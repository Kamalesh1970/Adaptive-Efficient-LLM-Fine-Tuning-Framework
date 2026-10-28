"""
Phase 6: Training Run Preview, Resource Validation, and Launch Control Tests.

Comprehensive backend tests verifying:
- Valid run plan generation via /api/training/preview and /api/training/validate-run
- Invalid model rejection / warnings
- Invalid dataset rejection / warnings
- Invalid training configuration rejection
- CPU-only environment compatibility handling
- GPU available detection and reporting
- GPU detection error handling
- Missing training dependencies detection (torch, transformers, peft)
- Missing quantization dependencies (bitsandbytes for QLoRA vs LoRA)
- Unknown compatibility reporting when parameters or device are unknown
- Invalid output directory permissions handling
- Secret exclusion (tokens not present in RunPlan or responses)
- Launch rejection on failed preflight validation
- Duplicate launch prevention via idempotency key
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.schemas.preflight import PreflightStatus, RunPlan
from backend.app.schemas.training import TrainingConfigSchema, TrainingMethod
from backend.app.services.model_loader import register_test_model, clear_test_model_registry
from backend.app.services.job_manager import get_job_manager


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


SAMPLE_VALID_CONFIG: dict = {
    "project_name": "phase6-test-project",
    "model_identifier": "meta-llama/Meta-Llama-3-8B",
    "training_method": "lora",
    "dataset_reference": "sample_alpaca_dataset",
    "training_length_type": "steps",
    "training_length": 100,
    "max_steps": 100,
    "num_epochs": 3.0,
    "learning_rate": 0.0002,
    "batch_size": 2,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "context_length": 2048,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "v_proj"],
    "optimizer": "adamw_8bit",
    "mixed_precision": "fp16",
    "quantization_type": "nf4",
    "gradient_checkpointing": True,
    "seed": 42,
    "hf_token_status": "present_in_env",
    "advanced_enabled": False,
}

SAMPLE_GPU_HARDWARE = {
    "pytorch_available": True,
    "cuda_available": True,
    "gpu_available": True,
    "gpu_count": 1,
    "device_type": "cuda",
    "primary_gpu_name": "NVIDIA GeForce RTX 2050",
    "gpu_name": "NVIDIA GeForce RTX 2050",
    "primary_gpu_vram_gb": 4.0,
    "environment_type": "cuda_accelerated",
    "detection_status": "gpu_detected",
    "diagnostic_message": "Detected 1 GPU device(s). Primary: NVIDIA GeForce RTX 2050 (4.0 GB VRAM).",
}

SAMPLE_CPU_HARDWARE = {
    "pytorch_available": True,
    "cuda_available": False,
    "gpu_available": False,
    "gpu_count": 0,
    "device_type": "cpu",
    "primary_gpu_name": None,
    "gpu_name": None,
    "primary_gpu_vram_gb": None,
    "environment_type": "cpu_only",
    "detection_status": "cpu_only_build",
    "diagnostic_message": "PyTorch was built without CUDA support (CPU-only build).",
}


class TestPhase6RunPlan:
    def test_valid_run_plan_structure(self, client: TestClient):
        """1. Valid run plan contains all required serializable fields and excludes secrets."""
        with patch("backend.app.services.preflight_service.get_system_hardware_info", return_value=SAMPLE_GPU_HARDWARE):
            payload = {
                "config": SAMPLE_VALID_CONFIG,
                "hf_token": "hf_super_secret_token_never_expose",
            }
            res = client.post("/api/training/preview", json=payload)
            assert res.status_code == 200
            data = res.json()

            # Verify all required top-level fields in RunPlan
            assert "run_configuration" in data
            assert "selected_model" in data
            assert "dataset_reference" in data
            assert "fine_tuning_method" in data
            assert "device_information" in data
            assert "validation_results" in data
            assert "warnings" in data
            assert "resource_compatibility" in data
            assert "output_configuration" in data
            assert "can_launch" in data
            assert "created_at" in data

            # Verify method and model
            assert data["fine_tuning_method"] == "lora"
            assert data["selected_model"]["id"] == "meta-llama/Meta-Llama-3-8B"

            # Verify secret exclusion
            dumped = json.dumps(data)
            assert "hf_super_secret_token_never_expose" not in dumped
            assert "token" not in data["run_configuration"] or data["run_configuration"].get("token") is None

    def test_validate_run_endpoint(self, client: TestClient):
        """Test POST /api/training/validate-run returns validation response with plan."""
        with patch("backend.app.services.preflight_service.get_system_hardware_info", return_value=SAMPLE_GPU_HARDWARE):
            res = client.post("/api/training/validate-run", json={"config": SAMPLE_VALID_CONFIG})
            assert res.status_code == 200
            data = res.json()
            assert "can_launch" in data
            assert "blocking_count" in data
            assert "warning_count" in data
            assert "checks" in data
            assert "plan" in data
            assert isinstance(data["checks"], list)


class TestPhase6ValidationChecks:
    def test_invalid_model(self, client: TestClient):
        """2. Invalid / unsupported model fails model check in preflight."""
        # google/gemma-2-2b only supports lora and qlora, not full_finetuning
        bad_config = {
            **SAMPLE_VALID_CONFIG,
            "model_identifier": "google/gemma-2-2b",
            "training_method": "full_finetuning",
        }
        res = client.post("/api/training/preflight", json=bad_config)
        assert res.status_code == 200
        data = res.json()
        model_check = next(c for c in data["checks"] if c["check_id"] == "model_catalog")
        assert model_check["status"] == "fail"
        assert data["can_launch"] is False

    def test_invalid_dataset(self, client: TestClient):
        """3. Invalid dataset with errors in report fails dataset validation."""
        mock_report = MagicMock()
        mock_report.is_valid = False
        mock_report.invalid_sample_count = 12
        mock_report.total_samples = 50
        mock_report.original_filename = "corrupted_dataset.jsonl"

        mock_ds_svc = MagicMock()
        mock_ds_svc.get_dataset.return_value = mock_report
        mock_ds_svc.get_report.return_value = mock_report

        with patch("backend.app.services.preflight_service.get_dataset_service", return_value=mock_ds_svc):
            res = client.post("/api/training/preflight", json=SAMPLE_VALID_CONFIG)
            assert res.status_code == 200
            data = res.json()
            ds_check = next(c for c in data["checks"] if c["check_id"] == "dataset_validation")
            assert ds_check["status"] == "fail"
            assert "12 invalid sample(s)" in ds_check["message"]
            assert data["can_launch"] is False

    def test_invalid_training_configuration(self, client: TestClient):
        """4. Invalid training configuration (e.g. negative learning rate) rejected."""
        bad_config = {**SAMPLE_VALID_CONFIG, "learning_rate": -0.05}
        res = client.post("/api/training/preflight", json=bad_config)
        assert res.status_code == 400

    def test_cpu_only_environment(self, client: TestClient):
        """5. CPU-only environment reports CPU mode warning without crashing."""
        with patch("backend.app.services.preflight_service.get_system_hardware_info", return_value=SAMPLE_CPU_HARDWARE):
            res = client.post("/api/training/preflight", json=SAMPLE_VALID_CONFIG)
            assert res.status_code == 200
            data = res.json()
            device_check = next(c for c in data["checks"] if c["check_id"] == "device")
            assert device_check["status"] == "warn"
            assert "CPU execution mode" in device_check["message"]

    def test_gpu_available(self, client: TestClient):
        """6. GPU available environment passes device check and reports GPU name."""
        with patch("backend.app.services.preflight_service.get_system_hardware_info", return_value=SAMPLE_GPU_HARDWARE):
            res = client.post("/api/training/preflight", json=SAMPLE_VALID_CONFIG)
            assert res.status_code == 200
            data = res.json()
            device_check = next(c for c in data["checks"] if c["check_id"] == "device")
            assert device_check["status"] == "pass"
            assert "NVIDIA GeForce RTX 2050" in device_check["message"]

    def test_gpu_detection_error(self, client: TestClient):
        """7. Hardware detection error yields fail status for compute device."""
        error_hw = {
            "pytorch_available": True,
            "cuda_available": False,
            "gpu_available": False,
            "gpu_count": 0,
            "device_type": "cpu",
            "detection_status": "detection_error",
            "diagnostic_message": "CUDA driver version is insufficient for CUDA runtime version",
        }
        with patch("backend.app.services.preflight_service.get_system_hardware_info", return_value=error_hw):
            res = client.post("/api/training/preflight", json=SAMPLE_VALID_CONFIG)
            assert res.status_code == 200
            data = res.json()
            device_check = next(c for c in data["checks"] if c["check_id"] == "device")
            assert device_check["status"] == "fail"
            assert "Hardware detection error" in device_check["message"]
            assert data["can_launch"] is False

    def test_missing_training_dependencies(self, client: TestClient):
        """8. Missing training dependencies (transformers, peft) yields fail check."""
        mock_deps = {
            "is_ready": False,
            "missing_critical": ["transformers", "peft"],
            "versions": {"torch": "2.1.0"},
            "bitsandbytes_available": False,
        }
        with patch("backend.app.services.preflight_service.check_ml_dependencies", return_value=mock_deps):
            res = client.post("/api/training/preflight", json=SAMPLE_VALID_CONFIG)
            assert res.status_code == 200
            data = res.json()
            ml_check = next(c for c in data["checks"] if c["check_id"] == "ml_core_deps")
            assert ml_check["status"] == "fail"
            assert "Critical ML packages missing" in ml_check["message"]
            assert data["can_launch"] is False

    def test_missing_quantization_dependencies(self, client: TestClient):
        """9. Missing bitsandbytes is distinguished from missing GPU and fails QLoRA."""
        mock_deps = {
            "is_ready": True,
            "missing_critical": [],
            "versions": {"torch": "2.1.0", "transformers": "4.40.0", "peft": "0.10.0"},
            "bitsandbytes_available": False,
        }
        qlora_config = {**SAMPLE_VALID_CONFIG, "training_method": "qlora"}
        with patch("backend.app.services.preflight_service.check_ml_dependencies", return_value=mock_deps):
            with patch("backend.app.services.preflight_service.get_system_hardware_info", return_value=SAMPLE_GPU_HARDWARE):
                res = client.post("/api/training/preflight", json=qlora_config)
                assert res.status_code == 200
                data = res.json()
                bnb_check = next(c for c in data["checks"] if c["check_id"] == "bitsandbytes")
                assert bnb_check["status"] == "fail"
                assert "bitsandbytes is required" in bnb_check["message"]
                # Notice device check still passes!
                device_check = next(c for c in data["checks"] if c["check_id"] == "device")
                assert device_check["status"] == "pass"
                assert data["can_launch"] is False

    def test_unknown_compatibility(self, client: TestClient):
        """10. Unknown model parameter count reports VRAM compatibility as unknown."""
        custom_config = {**SAMPLE_VALID_CONFIG, "model_identifier": "custom-org/unknown-model-398"}
        with patch("backend.app.services.preflight_service.get_system_hardware_info", return_value=SAMPLE_GPU_HARDWARE):
            res = client.post("/api/training/preflight", json=custom_config)
            assert res.status_code == 200
            data = res.json()
            vram_check = next(c for c in data["checks"] if c["check_id"] == "vram")
            assert vram_check["status"] == "warn"
            assert vram_check["detail"]["vram_status"] == "unknown"

    def test_invalid_output_directory(self, client: TestClient):
        """11. Output directory write permission failure fails output_dir check."""
        with patch("os.makedirs"):
            with patch("builtins.open", side_effect=PermissionError("Permission denied")):
                res = client.post("/api/training/preflight", json=SAMPLE_VALID_CONFIG)
                assert res.status_code == 200
                data = res.json()
                out_check = next(c for c in data["checks"] if c["check_id"] == "output_dir")
                assert out_check["status"] == "fail"
                assert data["can_launch"] is False

    def test_secret_exclusion_in_run_plan(self, client: TestClient):
        """12. Secrets, HF tokens, and passwords are fully stripped from RunPlan."""
        payload = {
            "config": {
                **SAMPLE_VALID_CONFIG,
                "hf_token": "hf_leaked_secret_token",
            },
            "hf_token": "hf_leaked_secret_token",
        }
        res = client.post("/api/training/preview", json=payload)
        assert res.status_code == 200
        plan = res.json()
        plan_str = json.dumps(plan)
        assert "hf_leaked_secret_token" not in plan_str


class TestPhase6LaunchControl:
    def test_launch_rejection_on_failed_preflight(self, client: TestClient):
        """13. Launch endpoint rejects configs that fail preflight checks."""
        # Missing model_identifier will fail preflight
        bad_payload = {
            "config": {
                **SAMPLE_VALID_CONFIG,
                "model_identifier": None,
            }
        }
        res = client.post("/api/training/runs", json=bad_payload)
        assert res.status_code == 400
        data = res.json()
        assert "rejected by preflight validation" in data["detail"]["message"]

    def test_duplicate_launch_prevention(self, client: TestClient):
        """14. Repeated launch requests with same idempotency_key return existing job without duplication."""
        import torch.nn as nn
        register_test_model("test/idempotent-model", lambda: nn.Sequential(nn.Linear(8, 8)), lambda: MagicMock())

        payload = {
            "config": {
                **SAMPLE_VALID_CONFIG,
                "project_name": "idempotency-test",
                "model_identifier": "test/idempotent-model",
                "dataset_reference": "sample_alpaca_dataset",
                "batch_size": 1,
                "max_steps": 2,
            },
            "idempotency_key": "unique_client_key_998877",
        }

        # First request creates the job
        res1 = client.post("/api/training/runs", json=payload)
        assert res1.status_code == 201
        run1 = res1.json()
        run_id_1 = run1["run_id"]

        # Second request with the same idempotency key returns the SAME job
        res2 = client.post("/api/training/runs", json=payload)
        assert res2.status_code == 201
        run2 = res2.json()
        run_id_2 = run2["run_id"]

        assert run_id_1 == run_id_2

        clear_test_model_registry()
