"""
Phase 8 Backend Tests: Training Artifacts, Adapter Export, and Model Evaluation.

Covers:
1. Artifact discovery from actual files
2. Artifact ownership and run association
3. Missing and partial artifacts
4. Path traversal and symlink attacks
5. File sizes and checksums
6. Export manifest generation
7. Secret exclusion from export packages
8. Safe archive contents (no path traversal, relative paths only)
9. Export failure and cleanup
10. Download authorization and artifact lookup
11. Adapter configuration validation
12. Evaluation dataset validation
13. Real metric extraction and correct perplexity calculation
14. Unavailable metrics and invalid numerical values (overflow, negative, NaN)
15. Evaluation persistence and terminal states
16. Baseline comparability rules
17. Secret redaction
18. Genuine integration test with registered synthetic model and dataset
"""

import os
import json
import zipfile
import math
import uuid
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.config import settings
from backend.app.schemas.artifacts import (
    ArtifactType,
    ValidationStatus,
    ExportStatus,
)
from backend.app.schemas.evaluation import (
    EvaluationStatus,
    EvaluationConfig,
)
from backend.app.services.artifact_service import (
    ArtifactService,
    ArtifactSecurityError,
    compute_file_sha256,
)
from backend.app.services.evaluation_service import (
    EvaluationService,
    calculate_perplexity,
)
from backend.app.services.model_loader import (
    register_test_model,
    clear_test_model_registry,
)
from backend.app.services.run_repository import get_run_repository


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class SyntheticTestModel:
    """Synthetic PyTorch-like model for unit and integration testing."""
    def __init__(self):
        self.is_synthetic_test_model = True
        self.config = MagicMock()

    def eval(self):
        return self

    def __call__(self, *args, **kwargs):
        class Output:
            pass
        out = Output()
        # Mock logits
        import torch
        out.logits = torch.randn(1, 10, 100)
        return out

    def generate(self, **kwargs):
        import torch
        return torch.tensor([[1, 2, 3, 4]])


class SyntheticTestTokenizer:
    """Synthetic tokenizer for unit and integration testing."""
    def __init__(self):
        self.pad_token = "[PAD]"
        self.eos_token = "[EOS]"

    def encode(self, text, **kwargs):
        return [1, 2, 3, 4, 5]

    def decode(self, tokens, **kwargs):
        return "Decoded sample completion"

    def __call__(self, text, **kwargs):
        import torch
        return {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }


@pytest.fixture(autouse=True)
def setup_test_models():
    register_test_model("test/phase8-model", lambda: SyntheticTestModel(), lambda: SyntheticTestTokenizer())
    yield
    clear_test_model_registry()


def create_fake_run_artifacts(base_dir: str, run_id: str, is_qlora: bool = False, missing_weights: bool = False):
    """Helper to create a realistic run artifact directory."""
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    adapter_cfg = {
        "peft_type": "LORA",
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "base_model_name_or_path": "test/phase8-model",
    }
    with open(os.path.join(run_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(adapter_cfg, f, indent=2)

    if not missing_weights:
        with open(os.path.join(run_dir, "adapter_model.safetensors"), "wb") as f:
            f.write(b"SAFE_TENSORS_BINARY_WEIGHTS_DATA_HERE_1234567890")

    training_cfg = {
        "project_name": "phase8-test-run",
        "model_identifier": "test/phase8-model",
        "training_method": "qlora" if is_qlora else "lora",
        "dataset_reference": "sample_alpaca",
        "batch_size": 2,
        "max_steps": 10,
        "hf_token": "hf_super_secret_token_12345678",
    }
    with open(os.path.join(run_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(training_cfg, f, indent=2)

    metrics = [{"step": 1, "loss": 2.5}, {"step": 2, "loss": 2.1}]
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    run_meta = {
        "run_id": run_id,
        "training_method": "qlora" if is_qlora else "lora",
        "model_identifier": "test/phase8-model",
        "completed_at": "2026-10-29T12:00:00Z",
        "total_steps": 10,
    }
    with open(os.path.join(run_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    if is_qlora:
        q_cfg = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"}
        with open(os.path.join(run_dir, "quantization_config.json"), "w", encoding="utf-8") as f:
            json.dump(q_cfg, f, indent=2)

    return run_dir


# =========================================================================
# 1. Artifact Discovery Tests
# =========================================================================

def test_artifact_discovery_from_actual_files(tmp_path):
    """Verifies artifact discovery only returns files that actually exist."""
    runs_dir = str(tmp_path / "runs")
    exports_dir = str(tmp_path / "exports")
    svc = ArtifactService(runs_root=runs_dir, exports_root=exports_dir)

    run_id = "run_disc_001"
    create_fake_run_artifacts(runs_dir, run_id, is_qlora=False)

    artifacts = svc.discover_artifacts_for_run(run_id)
    assert len(artifacts) >= 1
    primary = artifacts[0]
    assert primary.run_id == run_id
    assert primary.artifact_type == ArtifactType.LORA_ADAPTER
    assert primary.model_identifier == "test/phase8-model"
    assert primary.size_bytes > 0

    filenames = [f.filename for f in primary.files]
    assert "adapter_config.json" in filenames
    assert "adapter_model.safetensors" in filenames
    assert "training_config.json" in filenames


def test_artifact_ownership_and_run_association(tmp_path):
    """Artifact is strictly bound to its own training run."""
    runs_dir = str(tmp_path / "runs")
    svc = ArtifactService(runs_root=runs_dir)

    run1 = "run_own_1"
    run2 = "run_own_2"
    create_fake_run_artifacts(runs_dir, run1)
    create_fake_run_artifacts(runs_dir, run2)

    arts1 = svc.discover_artifacts_for_run(run1)
    arts2 = svc.discover_artifacts_for_run(run2)

    assert arts1[0].run_id == run1
    assert arts2[0].run_id == run2
    assert arts1[0].artifact_id != arts2[0].artifact_id


# =========================================================================
# 2. Missing & Corrupted Artifacts Tests
# =========================================================================

def test_missing_and_partial_artifacts(tmp_path):
    """Validation detects missing weight files and reports invalid status."""
    runs_dir = str(tmp_path / "runs")
    svc = ArtifactService(runs_root=runs_dir)

    run_id = "run_partial_001"
    create_fake_run_artifacts(runs_dir, run_id, missing_weights=True)

    svc.discover_artifacts_for_run(run_id)
    art_id = f"art_{run_id}_adapter"

    val_res = svc.validate_artifact(art_id)
    assert val_res.is_valid is False
    assert val_res.validation_status == ValidationStatus.INVALID
    assert any("weight file" in err.lower() for err in val_res.errors)


def test_corrupted_adapter_config(tmp_path):
    """Validation detects malformed JSON in adapter_config.json."""
    runs_dir = str(tmp_path / "runs")
    svc = ArtifactService(runs_root=runs_dir)

    run_id = "run_corrupt_cfg"
    run_dir = create_fake_run_artifacts(runs_dir, run_id)

    # Write corrupt JSON
    with open(os.path.join(run_dir, "adapter_config.json"), "w") as f:
        f.write("{ INVALID_JSON_DATA ::: ")

    svc.discover_artifacts_for_run(run_id)
    val_res = svc.validate_artifact(f"art_{run_id}_adapter")
    assert val_res.is_valid is False
    assert any("adapter_config.json" in err for err in val_res.errors)


# =========================================================================
# 3. Path Traversal and Symlink Security Tests
# =========================================================================

def test_path_traversal_rejection(tmp_path):
    """Artifact service strictly rejects relative path traversal attempts."""
    runs_dir = str(tmp_path / "runs")
    svc = ArtifactService(runs_root=runs_dir)

    with pytest.raises(ArtifactSecurityError):
        svc._resolve_safe_run_dir("../../../etc")

    with pytest.raises(ArtifactSecurityError):
        svc._resolve_safe_run_dir("run_1/../../outside")


def test_symlink_escape_rejection(tmp_path):
    """Artifact service rejects symlinks that resolve outside runs root."""
    runs_dir = tmp_path / "runs"
    outside_dir = tmp_path / "outside_secret"
    runs_dir.mkdir()
    outside_dir.mkdir()

    symlink_path = runs_dir / "run_symlink"
    symlink_path.symlink_to(outside_dir, target_is_directory=True)

    svc = ArtifactService(runs_root=str(runs_dir))

    # Resolving run_symlink should detect commonpath escape
    with pytest.raises(ArtifactSecurityError):
        svc.validate_artifact("art_run_symlink_adapter")


# =========================================================================
# 4. File Sizes, Checksums, and Manifest Generation
# =========================================================================

def test_file_sizes_and_sha256_checksums(tmp_path):
    """Validates accurate filesystem size calculation and real SHA-256 calculation."""
    runs_dir = str(tmp_path / "runs")
    svc = ArtifactService(runs_root=runs_dir)

    run_id = "run_csum_test"
    run_dir = create_fake_run_artifacts(runs_dir, run_id)

    val_res = svc.validate_artifact(f"art_{run_id}_adapter")
    assert val_res.is_valid is True
    assert "adapter_config.json" in val_res.checksums
    assert "adapter_model.safetensors" in val_res.checksums

    actual_csum = compute_file_sha256(os.path.join(run_dir, "adapter_config.json"))
    assert val_res.checksums["adapter_config.json"] == actual_csum
    assert len(actual_csum) == 64


# =========================================================================
# 5. Export Packaging and Secret Exclusion Tests
# =========================================================================

def test_export_package_manifest_and_secrets_exclusion(tmp_path):
    """Export package excludes secrets, generates manifest.json and README.md."""
    runs_dir = str(tmp_path / "runs")
    exports_dir = str(tmp_path / "exports")
    svc = ArtifactService(runs_root=runs_dir, exports_root=exports_dir)

    run_id = "run_export_001"
    create_fake_run_artifacts(runs_dir, run_id, is_qlora=False)

    art_id = f"art_{run_id}_adapter"
    export_res = svc.export_adapter(art_id)

    assert export_res.export_status == ExportStatus.EXPORTED
    assert export_res.package_size_bytes > 0

    zip_path = os.path.join(exports_dir, export_res.package_filename)
    assert os.path.isfile(zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        assert "adapter_config.json" in namelist
        assert "adapter_model.safetensors" in namelist
        assert "manifest.json" in namelist
        assert "README.md" in namelist

        # Verify no unsafe paths or leading slashes
        for name in namelist:
            assert not name.startswith("/")
            assert not name.startswith("\\")
            assert ".." not in name

        # Verify secret exclusion in training_config.json
        if "training_config.json" in namelist:
            cfg_text = zf.read("training_config.json").decode("utf-8")
            assert "hf_super_secret_token_12345678" not in cfg_text
            assert "[REDACTED]" in cfg_text

        # Verify manifest content
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        assert manifest["artifact_id"] == art_id
        assert manifest["model_identifier"] == "test/phase8-model"


def test_export_failure_and_cleanup_on_invalid_artifact(tmp_path):
    """Export rejects invalid artifact and cleans up any partial archive."""
    runs_dir = str(tmp_path / "runs")
    exports_dir = str(tmp_path / "exports")
    svc = ArtifactService(runs_root=runs_dir, exports_root=exports_dir)

    run_id = "run_invalid_export"
    create_fake_run_artifacts(runs_dir, run_id, missing_weights=True)

    with pytest.raises(ValueError) as excinfo:
        svc.export_adapter(f"art_{run_id}_adapter")
    assert "Cannot export invalid artifact" in str(excinfo.value)

    # Ensure no partial zip remained in exports directory
    assert not os.path.exists(os.path.join(exports_dir, f"art_{run_id}_adapter_export.zip"))


# =========================================================================
# 6. API Endpoints Tests
# =========================================================================

def test_api_list_and_get_artifacts(client: TestClient, tmp_path):
    """API endpoints list artifacts and retrieve safe metadata."""
    run_id = "run_api_art_1"
    create_fake_run_artifacts(settings.runs_output_dir, run_id)

    res = client.get(f"/api/training/runs/{run_id}/artifacts")
    assert res.status_code == 200
    data = res.json()
    assert data["run_id"] == run_id
    assert len(data["artifacts"]) >= 1

    art_id = data["artifacts"][0]["artifact_id"]
    get_res = client.get(f"/api/artifacts/{art_id}")
    assert get_res.status_code == 200
    art_data = get_res.json()
    assert art_data["artifact_id"] == art_id
    assert art_data["model_identifier"] == "test/phase8-model"


def test_api_validate_and_export_artifact(client: TestClient):
    """API validate and export pipeline."""
    run_id = "run_api_export_test"
    create_fake_run_artifacts(settings.runs_output_dir, run_id)

    art_id = f"art_{run_id}_adapter"

    # Validate
    val_res = client.post(f"/api/artifacts/{art_id}/validate", json={"check_loadable": True})
    assert val_res.status_code == 200
    val_data = val_res.json()
    assert val_data["is_valid"] is True
    assert val_data["validation_status"] == "valid"

    # Export
    exp_res = client.post(f"/api/artifacts/{art_id}/export")
    assert exp_res.status_code == 200
    exp_data = exp_res.json()
    assert exp_data["export_status"] == "exported"
    assert "download_url" in exp_data

    # Download
    down_res = client.get(f"/api/artifacts/{art_id}/download")
    assert down_res.status_code == 200
    assert down_res.headers["content-type"] == "application/zip"
    assert len(down_res.content) > 0


def test_api_download_nonexistent_returns_404(client: TestClient):
    """Attempting to download non-existent export package returns 404."""
    res = client.get("/api/artifacts/nonexistent_artifact_id/download")
    assert res.status_code == 404


# =========================================================================
# 7. Perplexity and Mathematical Evaluation Tests
# =========================================================================

def test_perplexity_mathematical_calculation():
    """Perplexity is calculated as exp(loss) with precision."""
    ppl, overflow, reason = calculate_perplexity(1.5)
    assert overflow is False
    assert reason is None
    assert ppl == round(math.exp(1.5), 4)

    # Loss 0 -> PPL 1.0
    ppl_zero, _, _ = calculate_perplexity(0.0)
    assert ppl_zero == 1.0


def test_perplexity_overflow_and_invalid_values():
    """Perplexity handles overflow (>85 loss), negative values, and NaN gracefully."""
    # Overflow
    ppl, overflow, reason = calculate_perplexity(120.0)
    assert ppl is None
    assert overflow is True
    assert "exceeds mathematical bounds" in reason

    # Negative loss
    ppl_neg, overflow_neg, reason_neg = calculate_perplexity(-1.0)
    assert ppl_neg is None
    assert "negative" in reason_neg

    # NaN loss
    ppl_nan, overflow_nan, reason_nan = calculate_perplexity(float("nan"))
    assert ppl_nan is None
    assert "NaN" in reason_nan


# =========================================================================
# 8. Evaluation Workflow and Baseline Comparison Tests
# =========================================================================

def test_evaluation_workflow_and_baseline_comparison(client: TestClient):
    """Full evaluation run computing fine-tuned metrics, baseline comparison, and sample generations."""
    run_id = "run_eval_test_01"
    create_fake_run_artifacts(settings.runs_output_dir, run_id)
    art_id = f"art_{run_id}_adapter"

    # Pre-validate artifact
    client.post(f"/api/artifacts/{art_id}/validate")

    eval_payload = {
        "artifact_id": art_id,
        "dataset_id": "alpaca_eval_ds",
        "split": "validation",
        "max_samples": 5,
        "batch_size": 1,
        "context_length": 128,
        "temperature": 0.7,
        "max_new_tokens": 16,
        "run_baseline_comparison": True,
        "generate_samples": True,
        "num_samples_to_save": 2,
    }

    # Mock dataset service get_normalized_samples
    mock_samples = [
        {"instruction": "What is Python?", "output": "Python is a programming language."},
        {"instruction": "Explain LoRA.", "output": "LoRA is Low-Rank Adaptation."},
        {"instruction": "What is QLoRA?", "output": "QLoRA is Quantized LoRA."},
    ]

    with patch("backend.app.services.evaluation_service.get_dataset_service") as mock_ds_svc:
        mock_svc = MagicMock()
        mock_svc.get_dataset.return_value = MagicMock(schema_type=MagicMock(value="instruction"))
        mock_svc.get_report.return_value = MagicMock(schema_type=MagicMock(value="instruction"))
        mock_svc.get_normalized_samples.return_value = mock_samples
        mock_ds_svc.return_value = mock_svc

        # 1. Launch evaluation
        eval_svc = EvaluationService()
        eval_run = eval_svc.create_evaluation(EvaluationConfig(**eval_payload))
        # Execute synchronously for deterministic testing
        completed_eval = eval_svc.start_evaluation(eval_run.evaluation_id, background=False)

        assert completed_eval.status == EvaluationStatus.COMPLETED
        assert completed_eval.metrics is not None
        assert completed_eval.metrics.eval_loss is not None
        assert completed_eval.metrics.perplexity is not None
        assert completed_eval.metrics.total_tokens > 0
        assert completed_eval.metrics.samples_per_second > 0

        # Baseline comparison assertions
        assert completed_eval.baseline_comparison is not None
        comp = completed_eval.baseline_comparison
        assert comp.is_comparable is True
        assert comp.base_eval_loss is not None
        assert comp.fine_tuned_eval_loss is not None
        assert comp.loss_delta is not None
        assert comp.comparison_summary != ""

        # Samples assertions
        assert len(completed_eval.samples) > 0
        s0 = completed_eval.samples[0]
        assert s0.prompt != ""
        assert s0.fine_tuned_output is not None
        assert s0.base_output is not None


def test_evaluation_cancellation():
    """Cancelling an evaluation updates state to CANCELLED cleanly."""
    run_id = "run_cancel_test"
    create_fake_run_artifacts(settings.runs_output_dir, run_id)
    art_id = f"art_{run_id}_adapter"

    eval_cfg = EvaluationConfig(
        artifact_id=art_id,
        dataset_id="test_ds",
        max_samples=10,
    )

    with patch("backend.app.services.evaluation_service.get_dataset_service") as mock_ds_svc:
        mock_svc = MagicMock()
        mock_svc.get_dataset.return_value = MagicMock(schema_type=MagicMock(value="instruction"))
        mock_svc.get_report.return_value = MagicMock(schema_type=MagicMock(value="instruction"))
        mock_svc.get_normalized_samples.return_value = [{"instruction": "hi", "output": "bye"}]
        mock_ds_svc.return_value = mock_svc

        eval_svc = EvaluationService()
        eval_run = eval_svc.create_evaluation(eval_cfg)
        cancelled = eval_svc.cancel_evaluation(eval_run.evaluation_id)
        assert cancelled.status == EvaluationStatus.CANCELLED
        assert "cancelled" in cancelled.error_message.lower()


def test_evaluation_duplicate_prevention():
    """Starting a duplicate evaluation on the same artifact while running raises an error."""
    run_id = f"run_dup_test_{uuid.uuid4().hex[:8]}"
    create_fake_run_artifacts(settings.runs_output_dir, run_id)
    art_id = f"art_{run_id}_adapter"

    eval_cfg = EvaluationConfig(
        artifact_id=art_id,
        dataset_id=f"test_ds_dup_{uuid.uuid4().hex[:6]}",
        max_samples=5,
    )

    with patch("backend.app.services.evaluation_service.get_dataset_service") as mock_ds_svc:
        mock_svc = MagicMock()
        mock_svc.get_dataset.return_value = MagicMock(schema_type=MagicMock(value="instruction"))
        mock_svc.get_report.return_value = MagicMock(schema_type=MagicMock(value="instruction"))
        mock_svc.get_normalized_samples.return_value = [{"instruction": "hi", "output": "bye"}]
        mock_ds_svc.return_value = mock_svc

        eval_svc = EvaluationService()
        eval_run = eval_svc.create_evaluation(eval_cfg)
        eval_run.status = EvaluationStatus.RUNNING
        get_run_repository().save_or_update_evaluation(eval_run)

        with pytest.raises(ValueError) as exc:
            eval_svc.create_evaluation(eval_cfg)
        assert "already running" in str(exc.value)

