"""
Training Job Lifecycle Manager (Phase 4).
Orchestrates training runs through explicit lifecycle states:
QUEUED -> VALIDATING -> LOADING_MODEL -> PREPARING_DATASET -> TRAINING -> SAVING -> COMPLETED.
Provides thread-safe job tracking, safe cancellation, artifact validation, and telemetry.
"""

import os
import json
import uuid
import time
import logging
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from backend.app.core.config import settings
from backend.app.schemas.training import (
    TrainingConfigSchema,
    TrainingMethod,
    TrainingRunStatus,
    TrainingMetricPoint,
    TrainingRunResponse,
    TrainingArtifactMetadata,
)
from backend.app.services.model_loader import (
    ModelLoaderService,
    get_model_loader_service,
    ModelLoadError,
    QuantizationDeviceError,
)
from backend.app.services.lora_service import (
    inject_lora_adapters,
    prepare_and_inject_qlora_adapters,
    save_and_verify_adapter,
    reload_lora_adapter,
    LoRAConfigurationError,
)
from backend.app.services.ml_dependencies import (
    verify_qlora_dependencies_or_raise,
    BitsAndBytesMissingError,
    CudaUnavailableForQuantizationError,
)
from backend.app.services.training_dataset_service import (
    TrainingDatasetService,
    get_training_dataset_service,
)
from backend.app.services.training_engine import (
    TrainingEngine,
    TrainingCancelledException,
    TrainingExecutionError,
)

logger = logging.getLogger(__name__)


def _sanitize_string(val: str, secrets_to_remove: List[Optional[str]]) -> str:
    """Removes sensitive credentials or tokens from error messages and logs."""
    cleaned = val
    for secret in secrets_to_remove:
        if secret and len(secret) > 4:
            cleaned = cleaned.replace(secret, "[REDACTED]")
    return cleaned


class TrainingJob:
    """Encapsulates state and telemetry for a single training run."""

    def __init__(
        self,
        run_id: str,
        config: TrainingConfigSchema,
        output_dir: str,
        hf_token: Optional[str] = None,
    ) -> None:
        self.run_id = run_id
        self.config = config
        self.output_dir = output_dir
        self._hf_token = hf_token

        self.status = TrainingRunStatus.QUEUED
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None

        self.current_step: int = 0
        self.max_steps: int = config.max_steps or 0
        self.current_epoch: float = 0.0
        self.num_epochs: float = float(getattr(config, "num_epochs", 0) or 0)
        self.current_loss: Optional[float] = None
        self.loss_history: List[TrainingMetricPoint] = []
        self.elapsed_time_seconds: float = 0.0
        self.error_message: Optional[str] = None
        self.device: str = "cpu"

        self.trainable_params: Optional[int] = None
        self.total_params: Optional[int] = None
        self.trainable_percent: Optional[float] = None
        self.quantization_info: Optional[Dict[str, Any]] = None
        if config.training_method == TrainingMethod.QLORA:
            self.quantization_info = config.to_quantization_dict()

        self._cancel_event = threading.Event()
        self._lock = threading.Lock()
        self.artifact_files: List[str] = []
        self.adapter_config: Dict[str, Any] = {}

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def request_cancellation(self) -> bool:
        """Requests cancellation if the job is in a cancellable state."""
        with self._lock:
            cancellable_states = {
                TrainingRunStatus.QUEUED,
                TrainingRunStatus.VALIDATING,
                TrainingRunStatus.LOADING_MODEL,
                TrainingRunStatus.PREPARING_DATASET,
                TrainingRunStatus.TRAINING,
            }
            if self.status in cancellable_states:
                self._cancel_event.set()
                return True
            return False

    def to_response(self) -> TrainingRunResponse:
        """Converts internal job state to typed TrainingRunResponse."""
        with self._lock:
            cancellable = self.status in {
                TrainingRunStatus.QUEUED,
                TrainingRunStatus.VALIDATING,
                TrainingRunStatus.LOADING_MODEL,
                TrainingRunStatus.PREPARING_DATASET,
                TrainingRunStatus.TRAINING,
            }
            return TrainingRunResponse(
                run_id=self.run_id,
                status=self.status,
                config=self.config,
                created_at=self.created_at,
                started_at=self.started_at,
                completed_at=self.completed_at,
                current_step=self.current_step,
                max_steps=self.max_steps,
                current_epoch=self.current_epoch,
                num_epochs=self.num_epochs,
                current_loss=self.current_loss,
                loss_history=list(self.loss_history),
                elapsed_time_seconds=round(self.elapsed_time_seconds, 2),
                error_message=self.error_message,
                device=self.device,
                trainable_params=self.trainable_params,
                total_params=self.total_params,
                trainable_percent=self.trainable_percent,
                cancellable=cancellable,
                quantization_info=self.quantization_info,
            )


class JobManager:
    """Manages training jobs lifecycle, concurrency, and persistence."""

    def __init__(
        self,
        runs_output_dir: Optional[str] = None,
        model_loader: Optional[ModelLoaderService] = None,
        dataset_service: Optional[TrainingDatasetService] = None,
        engine: Optional[TrainingEngine] = None,
    ) -> None:
        self.runs_output_dir = runs_output_dir or settings.runs_output_dir
        self.model_loader = model_loader or get_model_loader_service()
        self.dataset_service = dataset_service or get_training_dataset_service()
        self.engine = engine or TrainingEngine()

        os.makedirs(self.runs_output_dir, exist_ok=True)
        self._jobs: Dict[str, TrainingJob] = {}
        self._idempotency_map: Dict[str, str] = {}
        self._lock = threading.Lock()

    def create_job(
        self,
        config: TrainingConfigSchema,
        hf_token: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> TrainingJob:
        """
        Creates and registers a new training job.
        Validates configuration before registration.
        Guarantees duplicate submission prevention: returns existing job if idempotency_key is repeated.
        """
        if idempotency_key:
            with self._lock:
                if idempotency_key in self._idempotency_map:
                    existing_run_id = self._idempotency_map[idempotency_key]
                    existing_job = self._jobs.get(existing_run_id)
                    if existing_job:
                        return existing_job

        if config.training_method not in (TrainingMethod.LORA, TrainingMethod.QLORA):
            raise ValueError(
                f"Supported methods are LoRA and QLoRA fine-tuning. Requested: {config.training_method.value}"
            )

        if not config.model_identifier:
            raise ValueError("model_identifier must be provided in configuration.")

        if not config.dataset_reference:
            raise ValueError("dataset_reference must be provided in configuration.")

        # Check for duplicate submission with identical active job
        with self._lock:
            active_statuses = {
                TrainingRunStatus.QUEUED,
                TrainingRunStatus.VALIDATING,
                TrainingRunStatus.LOADING_MODEL,
                TrainingRunStatus.PREPARING_DATASET,
                TrainingRunStatus.TRAINING,
            }
            for j in self._jobs.values():
                if j.status in active_statuses and (
                    j.config.project_name == config.project_name
                    and j.config.model_identifier == config.model_identifier
                    and j.config.dataset_reference == config.dataset_reference
                    and j.config.training_method == config.training_method
                ):
                    if idempotency_key:
                        self._idempotency_map[idempotency_key] = j.run_id
                    return j

        run_id = f"run_{uuid.uuid4().hex[:12]}"
        output_dir = os.path.join(self.runs_output_dir, run_id)

        job = TrainingJob(
            run_id=run_id,
            config=config,
            output_dir=output_dir,
            hf_token=hf_token,
        )

        with self._lock:
            self._jobs[run_id] = job
            if idempotency_key:
                self._idempotency_map[idempotency_key] = run_id

        return job

    def get_job(self, run_id: str) -> Optional[TrainingJob]:
        """Retrieves job by run_id."""
        with self._lock:
            return self._jobs.get(run_id)

    def list_jobs(self) -> List[TrainingJob]:
        """Retrieves all registered jobs."""
        with self._lock:
            return list(self._jobs.values())

    def cancel_job(self, run_id: str) -> bool:
        """Requests cancellation for a running or queued job."""
        job = self.get_job(run_id)
        if not job:
            return False
        return job.request_cancellation()

    def start_job(self, run_id: str, background: bool = True) -> TrainingJob:
        """Starts executing the job lifecycle either in a background thread or synchronously."""
        job = self.get_job(run_id)
        if not job:
            raise KeyError(f"Training run '{run_id}' not found.")

        if background:
            thread = threading.Thread(
                target=self._execute_job_lifecycle,
                args=(job,),
                daemon=True,
                name=f"TrainingRunner-{run_id}",
            )
            thread.start()
        else:
            self._execute_job_lifecycle(job)

        return job

    def _execute_job_lifecycle(self, job: TrainingJob) -> None:
        """Executes the full training lifecycle stages with strict error and cancellation handling."""
        start_time = time.time()
        job.started_at = datetime.now(timezone.utc).isoformat()
        secrets = [job._hf_token]

        try:
            # 1. VALIDATING STAGE
            job.status = TrainingRunStatus.VALIDATING
            if job.is_cancelled:
                raise TrainingCancelledException("Run cancelled during validation.")

            # Validate output directory writability
            try:
                os.makedirs(job.output_dir, exist_ok=True)
                test_file = os.path.join(job.output_dir, ".write_test")
                with open(test_file, "w") as f:
                    f.write("ok")
                os.remove(test_file)
            except Exception as e:
                raise PermissionError(f"Cannot write to output directory '{job.output_dir}': {e}") from e

            # 2. LOADING_MODEL STAGE
            job.status = TrainingRunStatus.LOADING_MODEL
            if job.is_cancelled:
                raise TrainingCancelledException("Run cancelled before model loading.")

            is_qlora = (job.config.training_method == TrainingMethod.QLORA)
            quant_cfg = job.config.get_quantization_config() if is_qlora else None

            model, tokenizer, device_used = self.model_loader.load_model_and_tokenizer(
                model_identifier=job.config.model_identifier,
                hf_token=job._hf_token,
                quantization_config=quant_cfg,
                is_qlora=is_qlora,
            )
            job.device = device_used

            if job.is_cancelled:
                raise TrainingCancelledException("Run cancelled after model loading.")

            # Inject LoRA or QLoRA adapters
            if is_qlora:
                peft_model, param_summary = prepare_and_inject_qlora_adapters(
                    base_model=model,
                    config=job.config,
                )
            else:
                peft_model, param_summary = inject_lora_adapters(
                    base_model=model,
                    config=job.config,
                )
            job.trainable_params = param_summary.trainable_params
            job.total_params = param_summary.total_params
            job.trainable_percent = param_summary.trainable_percent

            # 3. PREPARING_DATASET STAGE
            job.status = TrainingRunStatus.PREPARING_DATASET
            if job.is_cancelled:
                raise TrainingCancelledException("Run cancelled before dataset preparation.")

            prepared_dataset = self.dataset_service.prepare_dataset(
                dataset_id=job.config.dataset_reference,
                tokenizer=tokenizer,
                config=job.config,
            )

            # 4. TRAINING STAGE
            job.status = TrainingRunStatus.TRAINING
            if job.is_cancelled:
                raise TrainingCancelledException("Run cancelled before training steps.")

            def step_callback(step: int, loss: float, epoch: float, elapsed: float) -> None:
                with job._lock:
                    job.current_step = step
                    job.current_loss = loss
                    job.current_epoch = epoch
                    job.elapsed_time_seconds = elapsed

            loss_history = self.engine.train(
                model=peft_model,
                prepared_dataset=prepared_dataset,
                config=job.config,
                is_cancelled_fn=lambda: job.is_cancelled,
                on_step_callback=step_callback,
            )
            job.loss_history = loss_history

            # 5. SAVING STAGE
            job.status = TrainingRunStatus.SAVING
            sanitized_cfg = job.config.model_dump()
            # Ensure no credentials in dumped config
            sanitized_cfg.pop("hf_token", None)

            # Save adapter artifacts & sanitized training config
            saved_files = save_and_verify_adapter(
                model=peft_model,
                output_directory=job.output_dir,
                sanitized_config=sanitized_cfg,
                quantization_config=job.quantization_info,
            )
            job.artifact_files = saved_files

            # Save metrics.json
            metrics_path = os.path.join(job.output_dir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump([m.model_dump() for m in loss_history], f, indent=2)
            if "metrics.json" not in job.artifact_files:
                job.artifact_files.append("metrics.json")

            # Save run_metadata.json
            run_meta = {
                "run_id": job.run_id,
                "training_method": job.config.training_method.value,
                "quantization": job.quantization_info,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "device": job.device,
                "trainable_params": job.trainable_params,
                "total_params": job.total_params,
                "trainable_percent": job.trainable_percent,
                "final_loss": job.current_loss,
                "total_steps": job.current_step,
            }
            meta_path = os.path.join(job.output_dir, "run_metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(run_meta, f, indent=2)
            if "run_metadata.json" not in job.artifact_files:
                job.artifact_files.append("run_metadata.json")

            # Read adapter config for metadata
            adapter_cfg_path = os.path.join(job.output_dir, "adapter_config.json")
            if os.path.isfile(adapter_cfg_path):
                with open(adapter_cfg_path, "r", encoding="utf-8") as f:
                    job.adapter_config = json.load(f)

            # Verification: Reload adapter on base model to guarantee integrity
            reload_lora_adapter(model, job.output_dir)

            # 6. COMPLETED
            job.elapsed_time_seconds = time.time() - start_time
            job.completed_at = datetime.now(timezone.utc).isoformat()
            job.status = TrainingRunStatus.COMPLETED
            logger.info(f"Training run {job.run_id} completed successfully.")

        except TrainingCancelledException as e:
            job.status = TrainingRunStatus.CANCELLED
            job.completed_at = datetime.now(timezone.utc).isoformat()
            job.elapsed_time_seconds = time.time() - start_time
            clean_err = _sanitize_string(str(e), secrets)
            job.error_message = clean_err
            logger.warning(f"Training run {job.run_id} was cancelled: {clean_err}")

        except Exception as e:
            job.status = TrainingRunStatus.FAILED
            job.completed_at = datetime.now(timezone.utc).isoformat()
            job.elapsed_time_seconds = time.time() - start_time
            clean_err = _sanitize_string(str(e), secrets)
            job.error_message = clean_err
            logger.error(f"Training run {job.run_id} failed: {clean_err}", exc_info=True)

    def get_artifacts_metadata(self, run_id: str) -> Optional[TrainingArtifactMetadata]:
        """Retrieves and validates artifact metadata for a completed run."""
        job = self.get_job(run_id)
        if not job or job.status != TrainingRunStatus.COMPLETED:
            return None

        # Verify artifacts exist on disk
        if not os.path.isdir(job.output_dir):
            return None

        files = sorted(os.listdir(job.output_dir))
        if "adapter_config.json" not in files:
            return None

        # Use safe relative directory identifier
        safe_rel_dir = os.path.join("artifacts", "runs", run_id)

        return TrainingArtifactMetadata(
            run_id=job.run_id,
            artifact_directory=safe_rel_dir,
            files=files,
            adapter_config=job.adapter_config,
            quantization_config=job.quantization_info,
            trainable_params=job.trainable_params or 0,
            total_params=job.total_params or 0,
            trainable_percent=job.trainable_percent or 0.0,
            is_valid=True,
            verification_status="verified",
            created_at=job.completed_at or job.created_at,
        )


_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Returns singleton JobManager instance."""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager
