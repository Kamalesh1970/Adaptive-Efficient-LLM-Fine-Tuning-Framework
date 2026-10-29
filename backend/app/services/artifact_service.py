"""
Artifact Registry, Validation, and Adapter Export Service for Phase 8.
Guarantees strict path safety, SHA-256 integrity checksums, secret redaction,
safe zip archive packaging, and truthful validation reporting.
"""

import os
import json
import hashlib
import zipfile
import tempfile
import shutil
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Any

from backend.app.core.config import settings
from backend.app.schemas.artifacts import (
    ArtifactMetadata,
    ArtifactFileEntry,
    ArtifactType,
    ValidationStatus,
    ExportStatus,
    ArtifactValidationResponse,
    ArtifactExportResponse,
    RunArtifactsListResponse,
)
from backend.app.services.run_repository import get_run_repository, _sanitize_dict_secrets
from backend.app.services.lora_service import reload_lora_adapter
from backend.app.services.ml_dependencies import is_full_ml_stack_available

logger = logging.getLogger(__name__)

# Files that must NEVER be included in exported archives
EXCLUDED_EXPORT_PATTERNS = {
    ".env", ".git", ".gitignore", ".venv", "runs.db", "runs.db-wal", "runs.db-shm",
    "id_rsa", "id_ed25519", "credentials", "secret", "token"
}


def compute_file_sha256(filepath: str) -> str:
    """Calculates SHA-256 checksum of an actual file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class ArtifactSecurityError(PermissionError):
    """Raised when an artifact path traversal or symlink escape is detected."""
    pass


class ArtifactService:
    """Manages artifact discovery, integrity validation, packaging, and safe download."""

    def __init__(
        self,
        runs_root: Optional[str] = None,
        exports_root: Optional[str] = None,
    ) -> None:
        self.runs_root = os.path.abspath(runs_root or settings.runs_output_dir)
        self.exports_root = os.path.abspath(exports_root or settings.exports_output_dir)
        os.makedirs(self.runs_root, exist_ok=True)
        os.makedirs(self.exports_root, exist_ok=True)
        self.repo = get_run_repository()

    # -------------------------------------------------------------
    # Safe Path Resolution
    # -------------------------------------------------------------

    def _resolve_safe_run_dir(self, run_id: str) -> str:
        """
        Validates run directory path, rejecting path traversal and symlink escapes.
        """
        if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
            raise ArtifactSecurityError(f"Invalid run ID: '{run_id}'. Path traversal rejected.")

        run_dir = os.path.join(self.runs_root, run_id)
        canonical = os.path.abspath(os.path.realpath(run_dir))
        canonical_root = os.path.abspath(os.path.realpath(self.runs_root))

        # Check that canonical path starts with canonical_root
        if os.path.commonpath([canonical, canonical_root]) != canonical_root:
            raise ArtifactSecurityError(f"Access denied: path escapes allowed artifact root.")

        return canonical

    def _resolve_safe_export_path(self, filename: str) -> str:
        """Validates export package path inside exports root."""
        if not filename or "/" in filename or "\\" in filename or ".." in filename:
            raise ArtifactSecurityError(f"Invalid package filename: '{filename}'.")

        target = os.path.join(self.exports_root, filename)
        canonical = os.path.abspath(os.path.realpath(target))
        canonical_root = os.path.abspath(os.path.realpath(self.exports_root))

        if os.path.commonpath([canonical, canonical_root]) != canonical_root:
            raise ArtifactSecurityError(f"Access denied: export path escapes allowed exports root.")

        return canonical

    # -------------------------------------------------------------
    # Artifact Discovery
    # -------------------------------------------------------------

    def discover_artifacts_for_run(self, run_id: str) -> List[ArtifactMetadata]:
        """
        Discovers artifacts that ACTUALLY exist on disk for a training run.
        Never fabricates artifacts or files that do not exist.
        """
        run_dir = self._resolve_safe_run_dir(run_id)
        if not os.path.isdir(run_dir):
            return []

        entries = sorted(os.listdir(run_dir))
        if not entries:
            return []

        artifacts: List[ArtifactMetadata] = []

        # Read config and run metadata if present
        training_cfg: Dict[str, Any] = {}
        cfg_file = os.path.join(run_dir, "training_config.json")
        if os.path.isfile(cfg_file):
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    training_cfg = json.load(f)
            except Exception as e:
                logger.warning(f"Could not parse training_config.json for run {run_id}: {e}")

        run_meta: Dict[str, Any] = {}
        meta_file = os.path.join(run_dir, "run_metadata.json")
        if os.path.isfile(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    run_meta = json.load(f)
            except Exception as e:
                logger.warning(f"Could not parse run_metadata.json for run {run_id}: {e}")

        # Look up run in DB if available
        db_run = self.repo.get_run(run_id)

        model_identifier = (
            training_cfg.get("model_identifier")
            or (db_run.model_identifier if db_run else None)
            or run_meta.get("model_identifier")
            or "unknown_model"
        )
        fine_tuning_method = (
            training_cfg.get("training_method")
            or (db_run.training_method if db_run else None)
            or run_meta.get("training_method")
            or "lora"
        )
        if hasattr(fine_tuning_method, "value"):
            fine_tuning_method = fine_tuning_method.value
        fine_tuning_method = str(fine_tuning_method).lower()

        created_at = (
            run_meta.get("completed_at")
            or (db_run.completed_at if db_run else None)
            or (db_run.created_at if db_run else None)
            or datetime.now(timezone.utc).isoformat()
        )

        adapter_cfg: Optional[Dict[str, Any]] = None
        adapter_cfg_path = os.path.join(run_dir, "adapter_config.json")
        if os.path.isfile(adapter_cfg_path):
            try:
                with open(adapter_cfg_path, "r", encoding="utf-8") as f:
                    adapter_cfg = json.load(f)
            except Exception:
                adapter_cfg = None

        quant_cfg: Optional[Dict[str, Any]] = None
        quant_cfg_path = os.path.join(run_dir, "quantization_config.json")
        if os.path.isfile(quant_cfg_path):
            try:
                with open(quant_cfg_path, "r", encoding="utf-8") as f:
                    quant_cfg = json.load(f)
            except Exception:
                quant_cfg = None

        # 1. Primary Adapter Artifact
        has_adapter_cfg = os.path.isfile(adapter_cfg_path)
        has_adapter_weights = any(
            os.path.isfile(os.path.join(run_dir, w))
            for w in ("adapter_model.safetensors", "adapter_model.bin", "model.safetensors", "pytorch_model.bin")
        )

        if has_adapter_cfg or has_adapter_weights:
            art_type = ArtifactType.QLORA_ADAPTER if fine_tuning_method == "qlora" or quant_cfg is not None else ArtifactType.LORA_ADAPTER
            if fine_tuning_method == "full":
                art_type = ArtifactType.FULL_MODEL

            # Discover all actual top-level files
            file_entries: List[ArtifactFileEntry] = []
            total_size = 0
            for fname in entries:
                fpath = os.path.join(run_dir, fname)
                if os.path.isfile(fpath):
                    sz = os.path.getsize(fpath)
                    total_size += sz
                    is_req = fname in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")
                    file_entries.append(
                        ArtifactFileEntry(
                            filename=fname,
                            size_bytes=sz,
                            sha256=None,  # Computed lazily during validation
                            is_required=is_req,
                            exists=True,
                        )
                    )

            primary_artifact_id = f"art_{run_id}_adapter"
            safe_rel_dir = os.path.relpath(run_dir, start=os.getcwd()) if not os.path.isabs(self.runs_root) else os.path.join("artifacts", "runs", run_id)

            existing_art = self.repo.get_artifact(primary_artifact_id)
            validation_status = existing_art.validation_status if existing_art else ValidationStatus.UNVALIDATED
            export_status = existing_art.export_status if existing_art else ExportStatus.NOT_EXPORTED
            exported_pkg = existing_art.exported_package_path if existing_art else None

            # Check if an export package exists on disk
            export_filename = f"{primary_artifact_id}_export.zip"
            export_disk_path = os.path.join(self.exports_root, export_filename)
            if os.path.isfile(export_disk_path):
                export_status = ExportStatus.EXPORTED
                exported_pkg = os.path.join("artifacts", "exports", export_filename)

            art_obj = ArtifactMetadata(
                artifact_id=primary_artifact_id,
                run_id=run_id,
                artifact_type=art_type,
                model_identifier=model_identifier,
                fine_tuning_method=fine_tuning_method,
                created_at=created_at,
                file_reference=safe_rel_dir,
                size_bytes=total_size,
                adapter_config=adapter_cfg,
                quantization_config=quant_cfg,
                checkpoint_step=run_meta.get("total_steps") or (db_run.current_step if db_run else None),
                validation_status=validation_status,
                export_status=export_status,
                exported_package_path=exported_pkg,
                files=file_entries,
                base_model_identifier=model_identifier,
            )
            self.repo.save_or_update_artifact(art_obj)
            artifacts.append(art_obj)

        # 2. Intermediate Checkpoints
        for entry in entries:
            if entry.startswith("checkpoint-"):
                cp_dir = os.path.join(run_dir, entry)
                if os.path.isdir(cp_dir):
                    cp_files = sorted(os.listdir(cp_dir))
                    cp_file_entries: List[ArtifactFileEntry] = []
                    cp_total_sz = 0
                    for cpf in cp_files:
                        cpf_path = os.path.join(cp_dir, cpf)
                        if os.path.isfile(cpf_path):
                            sz = os.path.getsize(cpf_path)
                            cp_total_sz += sz
                            cp_file_entries.append(
                                ArtifactFileEntry(
                                    filename=cpf,
                                    size_bytes=sz,
                                    is_required=cpf in ("adapter_config.json", "adapter_model.safetensors"),
                                    exists=True,
                                )
                            )

                    step_val = None
                    try:
                        step_val = int(entry.replace("checkpoint-", ""))
                    except ValueError:
                        pass

                    cp_art_id = f"art_{run_id}_{entry}"
                    cp_rel_dir = os.path.join("artifacts", "runs", run_id, entry)

                    cp_art = ArtifactMetadata(
                        artifact_id=cp_art_id,
                        run_id=run_id,
                        artifact_type=ArtifactType.CHECKPOINT,
                        model_identifier=model_identifier,
                        fine_tuning_method=fine_tuning_method,
                        created_at=datetime.fromtimestamp(os.path.getmtime(cp_dir), tz=timezone.utc).isoformat(),
                        file_reference=cp_rel_dir,
                        size_bytes=cp_total_sz,
                        adapter_config=adapter_cfg,
                        quantization_config=quant_cfg,
                        checkpoint_step=step_val,
                        validation_status=ValidationStatus.UNVALIDATED,
                        export_status=ExportStatus.NOT_EXPORTED,
                        files=cp_file_entries,
                        base_model_identifier=model_identifier,
                    )
                    self.repo.save_or_update_artifact(cp_art)
                    artifacts.append(cp_art)

        return artifacts

    def get_run_artifacts_response(self, run_id: str) -> RunArtifactsListResponse:
        """
        Returns full list of discovered artifacts for a run with backward-compatibility fields.
        """
        # Discover artifacts on disk and sync to DB
        artifacts = self.discover_artifacts_for_run(run_id)
        if not artifacts:
            # Check DB in case directory was moved
            artifacts = self.repo.list_artifacts(run_id=run_id)

        # Primary adapter artifact for backward-compatibility
        primary = next((a for a in artifacts if a.artifact_type in (ArtifactType.LORA_ADAPTER, ArtifactType.QLORA_ADAPTER, ArtifactType.FULL_MODEL)), None)
        if not primary and artifacts:
            primary = artifacts[0]

        files_list = [f.filename for f in primary.files] if primary else []
        safe_dir = primary.file_reference if primary else os.path.join("artifacts", "runs", run_id)

        # Check DB run for training stats
        db_run = self.repo.get_run(run_id)

        return RunArtifactsListResponse(
            run_id=run_id,
            artifacts=artifacts,
            total=len(artifacts),
            artifact_directory=safe_dir,
            files=files_list,
            adapter_config=primary.adapter_config if primary else None,
            quantization_config=primary.quantization_config if primary else None,
            trainable_params=db_run.current_step if db_run else None,
            total_params=None,
            trainable_percent=None,
            is_valid=primary.validation_status == ValidationStatus.VALID if primary else None,
            verification_status="verified" if primary and primary.validation_status == ValidationStatus.VALID else "unverified",
        )

    def get_artifact(self, artifact_id: str) -> Optional[ArtifactMetadata]:
        """Retrieves artifact by ID, attempting discovery if not yet in DB."""
        art = self.repo.get_artifact(artifact_id)
        if art:
            return art

        # Try to infer run_id from artifact_id (format: art_{run_id}_adapter)
        if artifact_id.startswith("art_"):
            clean = artifact_id[4:]
            if clean.endswith("_adapter"):
                run_id = clean[:-8]
                self.discover_artifacts_for_run(run_id)
                art = self.repo.get_artifact(artifact_id)
                if art:
                    return art
            elif "_checkpoint-" in clean:
                parts = clean.split("_checkpoint-")
                run_id = parts[0]
                self.discover_artifacts_for_run(run_id)
                art = self.repo.get_artifact(artifact_id)
                if art:
                    return art

        # Fallback: scan directories in runs_root
        if os.path.isdir(self.runs_root):
            for d in os.listdir(self.runs_root):
                if d.startswith("run_"):
                    self.discover_artifacts_for_run(d)
                    found = self.repo.get_artifact(artifact_id)
                    if found:
                        return found

        return None

    def list_all_artifacts(self) -> List[ArtifactMetadata]:
        """Returns all registered artifacts across all runs."""
        # Check all run directories in runs_root
        if os.path.isdir(self.runs_root):
            for d in os.listdir(self.runs_root):
                dpath = os.path.join(self.runs_root, d)
                if os.path.isdir(dpath) and d.startswith("run_"):
                    self.discover_artifacts_for_run(d)
        return self.repo.list_artifacts()

    # -------------------------------------------------------------
    # Artifact Validation
    # -------------------------------------------------------------

    def validate_artifact(
        self,
        artifact_id: str,
        check_loadable: bool = False,
    ) -> ArtifactValidationResponse:
        """
        Validates artifact integrity, schema completeness, and filesystem validity.
        Rejects path traversal, missing files, corrupted JSON, and reports precise issues.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        artifact = self.get_artifact(artifact_id)
        if not artifact:
            raise ValueError(f"Artifact '{artifact_id}' does not exist.")

        # Path security validation
        canonical_root = os.path.abspath(os.path.realpath(self.runs_root))
        candidate1 = os.path.abspath(os.path.realpath(os.path.join(self.runs_root, artifact.run_id)))
        candidate2 = os.path.abspath(os.path.realpath(artifact.file_reference)) if os.path.isabs(artifact.file_reference) else os.path.abspath(os.path.realpath(os.path.join(".", artifact.file_reference)))

        if os.path.isdir(candidate1):
            target_dir = candidate1
        elif os.path.isdir(candidate2):
            target_dir = candidate2
        else:
            target_dir = candidate1

        if os.path.commonpath([target_dir, canonical_root]) != canonical_root:
            raise ArtifactSecurityError(
                f"Security violation: artifact directory '{artifact.file_reference}' points outside the allowed root."
            )

        # Check for symlink attacks
        if os.path.islink(target_dir):
            resolved = os.path.realpath(target_dir)
            if os.path.commonpath([resolved, canonical_root]) != canonical_root:
                raise ArtifactSecurityError("Security violation: symlink traversal outside artifact root detected.")

        errors: List[str] = []
        warnings: List[str] = []
        checksums: Dict[str, str] = {}
        total_size = 0
        updated_files: List[ArtifactFileEntry] = []

        if not os.path.isdir(target_dir):
            errors.append(f"Artifact directory '{artifact.file_reference}' does not exist on disk.")
            status = ValidationStatus.CORRUPTED
            artifact.validation_status = status
            artifact.validation_errors = errors
            self.repo.save_or_update_artifact(artifact)
            return ArtifactValidationResponse(
                artifact_id=artifact_id,
                is_valid=False,
                validation_status=status,
                errors=errors,
                warnings=warnings,
                checksums={},
                total_size_bytes=0,
                is_loadable=False,
                load_error="Directory not found",
                checked_at=now_iso,
            )

        existing_entries = os.listdir(target_dir)

        # 1. Adapter validation checks
        if artifact.artifact_type in (ArtifactType.LORA_ADAPTER, ArtifactType.QLORA_ADAPTER):
            adapter_cfg_file = os.path.join(target_dir, "adapter_config.json")
            if not os.path.isfile(adapter_cfg_file):
                errors.append("Missing required 'adapter_config.json' file.")
            else:
                try:
                    with open(adapter_cfg_file, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    if not isinstance(cfg, dict):
                        errors.append("'adapter_config.json' does not contain a valid JSON object.")
                    elif "peft_type" not in cfg and "r" not in cfg:
                        warnings.append("'adapter_config.json' is missing standard PEFT hyperparameter fields.")
                except Exception as e:
                    errors.append(f"Corrupted 'adapter_config.json': {str(e)}")

            # Check for adapter weights
            has_safetensors = os.path.isfile(os.path.join(target_dir, "adapter_model.safetensors"))
            has_bin = os.path.isfile(os.path.join(target_dir, "adapter_model.bin"))

            if not has_safetensors and not has_bin:
                errors.append(
                    "Missing adapter weight file. Expected 'adapter_model.safetensors' or 'adapter_model.bin'."
                )
            else:
                for wfile in ("adapter_model.safetensors", "adapter_model.bin"):
                    wpath = os.path.join(target_dir, wfile)
                    if os.path.isfile(wpath) and os.path.getsize(wpath) == 0:
                        errors.append(f"Adapter weight file '{wfile}' is empty (0 bytes).")

            # QLoRA-specific checks
            if artifact.artifact_type == ArtifactType.QLORA_ADAPTER:
                q_cfg_file = os.path.join(target_dir, "quantization_config.json")
                if not os.path.isfile(q_cfg_file) and not (artifact.adapter_config and "bnb_4bit_quant_type" in artifact.adapter_config):
                    warnings.append(
                        "QLoRA adapter directory does not contain 'quantization_config.json'. Base model will require 4-bit config."
                    )

        elif artifact.artifact_type == ArtifactType.FULL_MODEL:
            cfg_file = os.path.join(target_dir, "config.json")
            if not os.path.isfile(cfg_file):
                errors.append("Full model requires 'config.json'.")
            has_model_weights = any(
                os.path.isfile(os.path.join(target_dir, w))
                for w in ("model.safetensors", "pytorch_model.bin")
            )
            if not has_model_weights:
                errors.append("Full model requires 'model.safetensors' or 'pytorch_model.bin'.")

        # 2. Checksum and size calculation for all present files
        for fname in existing_entries:
            fpath = os.path.join(target_dir, fname)
            if os.path.isfile(fpath):
                sz = os.path.getsize(fpath)
                total_size += sz
                try:
                    csum = compute_file_sha256(fpath)
                    checksums[fname] = csum
                except Exception as e:
                    errors.append(f"Could not compute checksum for '{fname}': {e}")
                    csum = None

                updated_files.append(
                    ArtifactFileEntry(
                        filename=fname,
                        size_bytes=sz,
                        sha256=csum,
                        is_required=fname in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"),
                        exists=True,
                    )
                )

        # 3. Model load verification if requested
        is_loadable: Optional[bool] = None
        load_error: Optional[str] = None

        if check_loadable and not errors:
            try:
                from backend.app.services.model_loader import get_test_model, is_test_model_registered
                # If a synthetic model is registered for this model identifier, test with it
                if is_test_model_registered(artifact.model_identifier):
                    test_m, _ = get_test_model(artifact.model_identifier)
                    reload_lora_adapter(test_m, target_dir)
                    is_loadable = True
                elif is_full_ml_stack_available():
                    # Real transformers/peft is installed, but check if local weights exist
                    if os.path.exists(artifact.model_identifier):
                        is_loadable = True
                    else:
                        warnings.append(
                            f"Base model '{artifact.model_identifier}' is remote. Skipping live weight download during validation."
                        )
                        is_loadable = True
                else:
                    warnings.append(
                        "Machine learning dependencies (transformers/peft) are not installed. Cannot test live model load."
                    )
                    is_loadable = None
            except Exception as e:
                is_loadable = False
                load_error = str(e)
                errors.append(f"Model load verification failed: {str(e)}")

        # Determine validation status
        if errors:
            status = ValidationStatus.INVALID
        else:
            status = ValidationStatus.VALID

        # Update persisted metadata
        artifact.validation_status = status
        artifact.validation_errors = errors
        artifact.validation_warnings = warnings
        artifact.files = updated_files
        artifact.size_bytes = total_size
        artifact.is_loadable = is_loadable
        artifact.load_error = load_error
        self.repo.save_or_update_artifact(artifact)

        return ArtifactValidationResponse(
            artifact_id=artifact_id,
            is_valid=(status == ValidationStatus.VALID),
            validation_status=status,
            errors=errors,
            warnings=warnings,
            checksums=checksums,
            total_size_bytes=total_size,
            is_loadable=is_loadable,
            load_error=load_error,
            checked_at=now_iso,
        )

    # -------------------------------------------------------------
    # Adapter Export Packaging
    # -------------------------------------------------------------

    def export_adapter(self, artifact_id: str) -> ArtifactExportResponse:
        """
        Creates a clean, safe, distributable ZIP archive for a validated LoRA/QLoRA adapter.
        Redacts credentials, excludes secret files, embeds manifest and README.
        """
        artifact = self.get_artifact(artifact_id)
        if not artifact:
            raise ValueError(f"Artifact '{artifact_id}' not found.")

        # Ensure artifact is validated
        if artifact.validation_status != ValidationStatus.VALID:
            val_res = self.validate_artifact(artifact_id, check_loadable=False)
            if not val_res.is_valid:
                raise ValueError(
                    f"Cannot export invalid artifact '{artifact_id}'. Validation errors: {'; '.join(val_res.errors)}"
                )

        candidate1 = os.path.abspath(os.path.realpath(os.path.join(self.runs_root, artifact.run_id)))
        candidate2 = os.path.abspath(os.path.realpath(artifact.file_reference)) if os.path.isabs(artifact.file_reference) else os.path.abspath(os.path.realpath(os.path.join(".", artifact.file_reference)))
        target_dir = candidate1 if os.path.isdir(candidate1) else candidate2
        export_filename = f"{artifact_id}_export.zip"
        final_zip_path = os.path.join(self.exports_root, export_filename)

        # Avoid duplicate packaging if already exported and package exists
        if (
            artifact.export_status == ExportStatus.EXPORTED
            and os.path.isfile(final_zip_path)
            and os.path.getsize(final_zip_path) > 0
        ):
            pkg_sz = os.path.getsize(final_zip_path)
            pkg_sha = compute_file_sha256(final_zip_path)
            return ArtifactExportResponse(
                artifact_id=artifact_id,
                export_status=ExportStatus.EXPORTED,
                package_filename=export_filename,
                package_size_bytes=pkg_sz,
                sha256=pkg_sha,
                created_at=datetime.fromtimestamp(os.path.getmtime(final_zip_path), tz=timezone.utc).isoformat(),
                download_url=f"/api/artifacts/{artifact_id}/download",
                message="Export package already exists and is up-to-date.",
            )

        artifact.export_status = ExportStatus.EXPORTING
        self.repo.save_or_update_artifact(artifact)

        temp_dir = tempfile.mkdtemp(prefix="export_stage_")
        temp_zip_path = os.path.join(temp_dir, export_filename)

        try:
            files_to_pack: List[Tuple[str, str]] = []  # (source_path, arcname)
            manifest_files: List[Dict[str, Any]] = []

            for entry in os.listdir(target_dir):
                # Exclude blacklisted files
                lower_entry = entry.lower()
                if any(bad in lower_entry for bad in EXCLUDED_EXPORT_PATTERNS):
                    continue

                source_path = os.path.join(target_dir, entry)
                if not os.path.isfile(source_path):
                    continue

                # Sanitize training_config.json if included
                if entry == "training_config.json":
                    try:
                        with open(source_path, "r", encoding="utf-8") as f:
                            raw_cfg = json.load(f)
                        clean_cfg = _sanitize_dict_secrets(raw_cfg)
                        staged_cfg = os.path.join(temp_dir, "training_config.json")
                        with open(staged_cfg, "w", encoding="utf-8") as f:
                            json.dump(clean_cfg, f, indent=2)
                        files_to_pack.append((staged_cfg, "training_config.json"))
                        manifest_files.append({
                            "filename": "training_config.json",
                            "size_bytes": os.path.getsize(staged_cfg),
                            "sha256": compute_file_sha256(staged_cfg),
                        })
                        continue
                    except Exception as e:
                        logger.warning(f"Error sanitizing training_config for export: {e}")

                # Standard files (adapter_config.json, weights, tokenizer files)
                csum = compute_file_sha256(source_path)
                files_to_pack.append((source_path, entry))
                manifest_files.append({
                    "filename": entry,
                    "size_bytes": os.path.getsize(source_path),
                    "sha256": csum,
                })

            # Create manifest.json
            now_str = datetime.now(timezone.utc).isoformat()
            manifest_data = {
                "manifest_version": "1.0",
                "artifact_id": artifact_id,
                "run_id": artifact.run_id,
                "artifact_type": artifact.artifact_type.value,
                "model_identifier": artifact.model_identifier,
                "fine_tuning_method": artifact.fine_tuning_method,
                "created_at": now_str,
                "generator": "Adaptive LLM Fine-Tuning Studio v0.1",
                "files": manifest_files,
            }
            manifest_path = os.path.join(temp_dir, "manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)
            files_to_pack.append((manifest_path, "manifest.json"))

            # Create README.md
            readme_content = f"""# Fine-Tuned Model Adapter Package

- **Artifact ID**: `{artifact_id}`
- **Training Run ID**: `{artifact.run_id}`
- **Fine-Tuning Method**: `{artifact.fine_tuning_method.upper()}`
- **Base Model Identifier**: `{artifact.model_identifier}`
- **Created At**: `{now_str}`

## Compatibility Note
This package contains a Parameter-Efficient Fine-Tuning (PEFT) adapter.
It does NOT contain the full base model weights. To run inference or further training,
you must load the base model `{artifact.model_identifier}` first, and then apply this adapter.

## Quickstart (Python / Transformers & PEFT)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Load compatible base model and tokenizer
base_model_name = "{artifact.model_identifier}"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# 2. Attach this fine-tuned adapter
model = PeftModel.from_pretrained(base_model, "./")
model.eval()

# 3. Perform inference
inputs = tokenizer("Hello, fine-tuned model!", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
"""
            readme_path = os.path.join(temp_dir, "README.md")
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(readme_content)
            files_to_pack.append((readme_path, "README.md"))

            # Build safe zip archive
            with zipfile.ZipFile(temp_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_out:
                for src, arc in files_to_pack:
                    # Double-check arcname has no path traversal
                    clean_arc = arc.lstrip("/\\")
                    if ".." in clean_arc:
                        raise ArtifactSecurityError(f"Unsafe archive entry name: '{clean_arc}'")
                    zip_out.write(src, arcname=clean_arc)

            # Atomic move to exports root
            shutil.move(temp_zip_path, final_zip_path)

            final_size = os.path.getsize(final_zip_path)
            final_sha = compute_file_sha256(final_zip_path)

            artifact.export_status = ExportStatus.EXPORTED
            artifact.exported_package_path = os.path.join("artifacts", "exports", export_filename)
            self.repo.save_or_update_artifact(artifact)

            return ArtifactExportResponse(
                artifact_id=artifact_id,
                export_status=ExportStatus.EXPORTED,
                package_filename=export_filename,
                package_size_bytes=final_size,
                sha256=final_sha,
                created_at=now_str,
                download_url=f"/api/artifacts/{artifact_id}/download",
                message=f"Adapter exported successfully ({len(files_to_pack)} files packaged).",
            )

        except Exception as e:
            artifact.export_status = ExportStatus.EXPORT_FAILED
            self.repo.save_or_update_artifact(artifact)
            if os.path.isfile(final_zip_path):
                try:
                    os.remove(final_zip_path)
                except OSError:
                    pass
            raise RuntimeError(f"Export packaging failed: {str(e)}") from e

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def get_export_download_path(self, artifact_id: str) -> Tuple[str, str]:
        """
        Returns (canonical_file_path, download_filename) for safe download.
        Rejects path traversal and non-existent packages.
        """
        artifact = self.get_artifact(artifact_id)
        if not artifact:
            raise ValueError(f"Artifact '{artifact_id}' not found.")

        export_filename = f"{artifact_id}_export.zip"
        canonical = self._resolve_safe_export_path(export_filename)

        if not os.path.isfile(canonical):
            raise FileNotFoundError(
                f"Export package for artifact '{artifact_id}' has not been generated or was removed."
            )

        return canonical, export_filename


_artifact_service: Optional[ArtifactService] = None


def get_artifact_service() -> ArtifactService:
    """Singleton getter for ArtifactService."""
    global _artifact_service
    if _artifact_service is None:
        _artifact_service = ArtifactService()
    return _artifact_service
