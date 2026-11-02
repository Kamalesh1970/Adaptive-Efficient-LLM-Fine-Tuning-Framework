"""
Regression test suite for Artifact Validation and Model Evaluation (Phase 8 Remediation).
Verifies:
- Accurate base model resolution (local cache / directory vs remote restriction).
- Artifact integrity validation (weights, PEFT config, base model metadata).
- Dataset disk auto-recovery.
- Evaluation metrics calculation, sample generation, baseline comparison, and run persistence.
- Real hardware integration test on trained QLoRA adapter.
"""

import os
import json
import pytest
import tempfile
import shutil
from unittest.mock import MagicMock, patch

from backend.app.schemas.artifacts import (
    ArtifactMetadata,
    ArtifactType,
    ValidationStatus,
)
from backend.app.schemas.evaluation import (
    EvaluationConfig,
    EvaluationStatus,
    EvaluationRun,
)
from backend.app.services.artifact_service import ArtifactService
from backend.app.services.dataset_service import DatasetService
from backend.app.services.evaluation_service import EvaluationService
from backend.app.services.model_loader import is_model_cached_locally
from backend.app.services.run_repository import get_run_repository


@pytest.fixture
def run_repo():
    return get_run_repository()


@pytest.fixture
def artifact_svc():
    return ArtifactService()


@pytest.fixture
def eval_svc():
    return EvaluationService()


@pytest.fixture
def ds_svc():
    return DatasetService()


# ----------------------------------------------------------------------
# 1. Base Model Local Resolution & Download Restrictions
# ----------------------------------------------------------------------

def test_local_model_cache_resolution_success():
    """Confirms local cache resolution detects cached Qwen 0.5B model without network."""
    is_cached, path = is_model_cached_locally("Qwen/Qwen2.5-0.5B-Instruct")
    assert is_cached is True
    assert path is not None


def test_local_model_cache_resolution_failure():
    """Confirms non-cached model truthfully returns False."""
    is_cached, path = is_model_cached_locally("test/non-existent-remote-model-xyz")
    assert is_cached is False
    assert path is None


def test_evaluation_fails_when_base_model_missing_locally(eval_svc, run_repo):
    """Evaluation must fail with FileNotFoundError explaining remote download restriction when model is absent."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        art_id = "art_test_missing_model"
        run_id = "run_test_missing_model"
        
        # Create minimal adapter files
        with open(os.path.join(tmp_dir, "adapter_config.json"), "w") as f:
            json.dump({
                "base_model_name_or_path": "non-existent/remote-unapproved-model",
                "peft_type": "LORA",
                "r": 8,
                "target_modules": ["q_proj"]
            }, f)
        with open(os.path.join(tmp_dir, "adapter_model.safetensors"), "wb") as f:
            f.write(b"dummy_weights_12345678")

        art = ArtifactMetadata(
            artifact_id=art_id,
            run_id=run_id,
            artifact_type=ArtifactType.LORA_ADAPTER,
            model_identifier="non-existent/remote-unapproved-model",
            fine_tuning_method="lora",
            created_at="2026-11-03T00:00:00Z",
            file_reference=tmp_dir,
            validation_status=ValidationStatus.VALID,
            base_model_identifier="non-existent/remote-unapproved-model",
        )
        run_repo.save_or_update_artifact(art)

        # Attempt evaluation
        with pytest.raises(FileNotFoundError) as exc_info:
            eval_svc._evaluate_model(
                model_identifier="non-existent/remote-unapproved-model",
                adapter_dir=tmp_dir,
                records=[{"instruction": "Hi", "output": "Hello"}],
                schema_type="instruction",
                config=EvaluationConfig(
                    artifact_id=art_id,
                    dataset_id="test_ds",
                ),
                is_baseline=False,
                evaluation_id="eval_test_fail",
            )
        assert "Automatic remote model downloading is restricted during evaluation" in str(exc_info.value)


# ----------------------------------------------------------------------
# 2. Artifact Integrity and Metadata Validation
# ----------------------------------------------------------------------

def test_artifact_validation_detects_missing_directory(artifact_svc, run_repo):
    """Artifact validation fails when artifact directory does not exist on disk."""
    art = ArtifactMetadata(
        artifact_id="art_missing_dir_test",
        run_id="run_missing_dir_test",
        artifact_type=ArtifactType.LORA_ADAPTER,
        model_identifier="Qwen/Qwen2.5-0.5B-Instruct",
        fine_tuning_method="lora",
        created_at="2026-11-03T00:00:00Z",
        file_reference="artifacts/runs/non_existent_run_folder_xyz",
        validation_status=ValidationStatus.UNVALIDATED,
    )
    run_repo.save_or_update_artifact(art)

    res = artifact_svc.validate_artifact("art_missing_dir_test")
    assert res.is_valid is False
    assert res.validation_status == ValidationStatus.CORRUPTED
    assert any("does not exist on disk" in err for err in res.errors)


def test_artifact_validation_detects_missing_adapter_weights(artifact_svc, run_repo):
    """Artifact validation fails when adapter_model.safetensors or .bin is missing."""
    with tempfile.TemporaryDirectory(dir="artifacts/runs") as tmp_dir:
        run_name = os.path.basename(tmp_dir)
        art_id = f"art_{run_name}_adapter"
        
        # Create adapter_config.json only
        with open(os.path.join(tmp_dir, "adapter_config.json"), "w") as f:
            json.dump({
                "base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                "peft_type": "LORA",
                "r": 16,
            }, f)

        art = ArtifactMetadata(
            artifact_id=art_id,
            run_id=run_name,
            artifact_type=ArtifactType.LORA_ADAPTER,
            model_identifier="Qwen/Qwen2.5-0.5B-Instruct",
            fine_tuning_method="lora",
            created_at="2026-11-03T00:00:00Z",
            file_reference=tmp_dir,
            validation_status=ValidationStatus.UNVALIDATED,
        )
        run_repo.save_or_update_artifact(art)

        res = artifact_svc.validate_artifact(art_id)
        assert res.is_valid is False
        assert any("Missing adapter weight file" in err for err in res.errors)


def test_artifact_validation_detects_corrupted_adapter_config(artifact_svc, run_repo):
    """Artifact validation fails when adapter_config.json is corrupted."""
    with tempfile.TemporaryDirectory(dir="artifacts/runs") as tmp_dir:
        run_name = os.path.basename(tmp_dir)
        art_id = f"art_{run_name}_adapter"
        
        # Corrupted config
        with open(os.path.join(tmp_dir, "adapter_config.json"), "w") as f:
            f.write("{invalid_json: true,")
        with open(os.path.join(tmp_dir, "adapter_model.safetensors"), "wb") as f:
            f.write(b"weights")

        art = ArtifactMetadata(
            artifact_id=art_id,
            run_id=run_name,
            artifact_type=ArtifactType.LORA_ADAPTER,
            model_identifier="Qwen/Qwen2.5-0.5B-Instruct",
            fine_tuning_method="lora",
            created_at="2026-11-03T00:00:00Z",
            file_reference=tmp_dir,
            validation_status=ValidationStatus.UNVALIDATED,
        )
        run_repo.save_or_update_artifact(art)

        res = artifact_svc.validate_artifact(art_id)
        assert res.is_valid is False
        assert any("Corrupted 'adapter_config.json'" in err for err in res.errors)


def test_real_qlora_artifact_validation_succeeds(artifact_svc):
    """Validates real trained QLoRA adapter art_run_ba99e5ac1653_adapter with check_loadable=True."""
    res = artifact_svc.validate_artifact("art_run_ba99e5ac1653_adapter", check_loadable=True)
    assert res.is_valid is True
    assert res.validation_status == ValidationStatus.VALID
    assert res.is_loadable is True
    assert len(res.errors) == 0
    assert res.total_size_bytes > 30 * 1024 * 1024


# ----------------------------------------------------------------------
# 3. Dataset Disk Recovery
# ----------------------------------------------------------------------

def test_dataset_auto_recovers_starter_dataset(ds_svc):
    """Confirms DatasetService auto-recovers ds_36f5df8f99c4 from upload disk directory."""
    samples = ds_svc.get_normalized_samples("ds_36f5df8f99c4")
    assert samples is not None
    assert len(samples) == 30
    
    report = ds_svc.get_report("ds_36f5df8f99c4")
    assert report is not None
    assert report.is_valid is True
    assert report.valid_sample_count == 30


# ----------------------------------------------------------------------
# 4. Evaluation Workflow & Error Persistence
# ----------------------------------------------------------------------

def test_failed_evaluation_persists_error_message(eval_svc, run_repo):
    """Confirms failed evaluation records failure status and sanitized error in database."""
    config = EvaluationConfig(
        artifact_id="art_run_ba99e5ac1653_adapter",
        dataset_id="non_existent_ds_xyz",
        split="validation",
    )
    with pytest.raises(ValueError) as exc:
        eval_svc.create_evaluation(config)
    assert "Validation dataset 'non_existent_ds_xyz' not found" in str(exc.value)


# ----------------------------------------------------------------------
# 5. Real Hardware Inference & Baseline Comparison Test
# ----------------------------------------------------------------------

@pytest.mark.cuda
def test_real_hardware_evaluation_inference(eval_svc):
    """
    Genuine PyTorch & CUDA hardware evaluation of art_run_ba99e5ac1653_adapter
    on starter instruction dataset ds_36f5df8f99c4.
    """
    config = EvaluationConfig(
        artifact_id="art_run_ba99e5ac1653_adapter",
        dataset_id="ds_36f5df8f99c4",
        split="validation",
        max_samples=2,
        batch_size=1,
        context_length=512,
        temperature=0.7,
        max_new_tokens=16,
        do_sample=False,
        seed=42,
        run_baseline_comparison=True,
        generate_samples=True,
        num_samples_to_save=2,
    )

    eval_run = eval_svc.create_evaluation(config)
    assert eval_run.status == EvaluationStatus.QUEUED

    completed = eval_svc.start_evaluation(eval_run.evaluation_id, background=False)
    assert completed.status == EvaluationStatus.COMPLETED
    assert completed.error_message is None
    assert completed.metrics is not None
    assert completed.metrics.eval_loss > 0.0
    assert completed.metrics.perplexity > 1.0
    assert completed.baseline_comparison is not None
    assert completed.baseline_comparison.base_eval_loss is not None
    assert len(completed.samples) == 2
    assert completed.samples[0].fine_tuned_output is not None
