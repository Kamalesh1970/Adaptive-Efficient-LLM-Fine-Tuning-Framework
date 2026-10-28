"""
Phase 6: Preflight Validation Endpoint Tests.

Tests the /api/training/preflight endpoint for correct aggregation of
model, dataset, device, dependency, and VRAM checks.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# Minimal valid config payload (no model, no dataset — both selectively set)
# ---------------------------------------------------------------------------

MINIMAL_CONFIG: dict = {
    "project_name": "preflight-test",
    "model_identifier": None,
    "training_method": "lora",
    "dataset_reference": None,
    "training_length_type": "steps",
    "training_length": 100,
    "max_steps": 100,
    "num_epochs": 3.0,
    "learning_rate": 0.0002,
    "batch_size": 2,
    "gradient_accumulation_steps": 1,
    "context_length": 2048,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "optimizer": "adamw_8bit",
    "mixed_precision": "fp16",
    "quantization_type": "nf4",
    "gradient_checkpointing": True,
    "seed": 42,
    "hf_token_status": "not_detected",
    "advanced_enabled": False,
}


class TestPreflightEndpointSchemaValidation:
    def test_invalid_payload_type_returns_422(self, client: TestClient) -> None:
        """Non-object payload (e.g. a list) must be rejected with HTTP 422."""
        res = client.post(
            "/api/training/preflight",
            content=b'["not", "an", "object"]',
            headers={"Content-Type": "application/json"},
        )
        assert res.status_code == 422

    def test_missing_required_field_returns_400(self, client: TestClient) -> None:
        """Payload missing project_name should be rejected."""
        bad_payload = {k: v for k, v in MINIMAL_CONFIG.items() if k != "project_name"}
        bad_payload["learning_rate"] = -0.001  # invalid value
        res = client.post("/api/training/preflight", json=bad_payload)
        assert res.status_code == 400


class TestPreflightResponseStructure:
    def test_response_has_correct_top_level_fields(self, client: TestClient) -> None:
        """Response must include checks, can_launch, blocking_count, warning_count."""
        res = client.post("/api/training/preflight", json=MINIMAL_CONFIG)
        assert res.status_code == 200
        data = res.json()
        assert "checks" in data
        assert "can_launch" in data
        assert "blocking_count" in data
        assert "warning_count" in data
        assert "method" in data
        assert data["method"] == "lora"

    def test_each_check_has_required_fields(self, client: TestClient) -> None:
        """Every check object must have check_id, label, status, message."""
        res = client.post("/api/training/preflight", json=MINIMAL_CONFIG)
        assert res.status_code == 200
        checks = res.json()["checks"]
        assert len(checks) > 0
        for check in checks:
            assert "check_id" in check
            assert "label" in check
            assert "status" in check
            assert "message" in check
            assert check["status"] in ("pass", "fail", "warn", "skip")

    def test_check_ids_are_unique(self, client: TestClient) -> None:
        """No two checks should share the same check_id."""
        res = client.post("/api/training/preflight", json=MINIMAL_CONFIG)
        ids = [c["check_id"] for c in res.json()["checks"]]
        assert len(ids) == len(set(ids))


class TestPreflightNoModelNoDataset:
    def test_no_model_is_fail(self, client: TestClient) -> None:
        """With no model_identifier, model_selected check must be 'fail'."""
        res = client.post("/api/training/preflight", json=MINIMAL_CONFIG)
        assert res.status_code == 200
        data = res.json()
        model_check = next(c for c in data["checks"] if c["check_id"] == "model_selected")
        assert model_check["status"] == "fail"

    def test_no_dataset_is_fail(self, client: TestClient) -> None:
        """With no dataset_reference, dataset_selected check must be 'fail'."""
        res = client.post("/api/training/preflight", json=MINIMAL_CONFIG)
        assert res.status_code == 200
        data = res.json()
        ds_check = next(c for c in data["checks"] if c["check_id"] == "dataset_selected")
        assert ds_check["status"] == "fail"

    def test_can_launch_is_false_with_no_model_no_dataset(self, client: TestClient) -> None:
        """can_launch must be False when model and dataset are missing."""
        res = client.post("/api/training/preflight", json=MINIMAL_CONFIG)
        assert res.status_code == 200
        assert res.json()["can_launch"] is False

    def test_blocking_count_positive_with_no_model_no_dataset(self, client: TestClient) -> None:
        """blocking_count must be >= 1 when model and dataset are absent."""
        res = client.post("/api/training/preflight", json=MINIMAL_CONFIG)
        assert res.json()["blocking_count"] >= 1


class TestPreflightWithValidModel:
    def test_valid_model_passes_model_selected_check(self, client: TestClient) -> None:
        """Known model in catalog yields pass for model_selected."""
        payload = {**MINIMAL_CONFIG, "model_identifier": "meta-llama/Llama-3.2-1B"}
        res = client.post("/api/training/preflight", json=payload)
        assert res.status_code == 200
        data = res.json()
        model_check = next(c for c in data["checks"] if c["check_id"] == "model_selected")
        assert model_check["status"] == "pass"

    def test_unknown_model_produces_catalog_warn(self, client: TestClient) -> None:
        """Unregistered model_identifier yields warn for model_catalog check."""
        payload = {**MINIMAL_CONFIG, "model_identifier": "some-org/unknown-model-xyz"}
        res = client.post("/api/training/preflight", json=payload)
        assert res.status_code == 200
        data = res.json()
        catalog_check = next((c for c in data["checks"] if c["check_id"] == "model_catalog"), None)
        assert catalog_check is not None
        assert catalog_check["status"] == "warn"

    def test_config_schema_check_always_passes_for_valid_config(self, client: TestClient) -> None:
        """config_schema check always passes when configuration is schema-valid."""
        res = client.post("/api/training/preflight", json=MINIMAL_CONFIG)
        config_check = next(c for c in res.json()["checks"] if c["check_id"] == "config_schema")
        assert config_check["status"] == "pass"


class TestPreflightQLoRAMethodChecks:
    def test_qlora_bitsandbytes_check_is_included(self, client: TestClient) -> None:
        """For qlora method, bitsandbytes check is present (not skipped)."""
        payload = {**MINIMAL_CONFIG, "training_method": "qlora"}
        res = client.post("/api/training/preflight", json=payload)
        assert res.status_code == 200
        data = res.json()
        bnb_check = next(c for c in data["checks"] if c["check_id"] == "bitsandbytes")
        assert bnb_check["status"] in ("pass", "fail")  # not skip

    def test_lora_bitsandbytes_check_is_skipped(self, client: TestClient) -> None:
        """For lora method, bitsandbytes check status is 'skip'."""
        payload = {**MINIMAL_CONFIG, "training_method": "lora"}
        res = client.post("/api/training/preflight", json=payload)
        assert res.status_code == 200
        data = res.json()
        bnb_check = next(c for c in data["checks"] if c["check_id"] == "bitsandbytes")
        assert bnb_check["status"] == "skip"

    def test_qlora_without_cuda_fails_method_compat(self, client: TestClient) -> None:
        """QLoRA with CPU-only hardware must yield fail for method_device_compat."""
        cpu_hw = {
            "pytorch_available": True,
            "cuda_available": False,
            "gpu_available": False,
            "gpu_count": 0,
            "device_type": "cpu",
            "primary_gpu_name": None,
            "primary_gpu_vram_gb": None,
            "environment_type": "cpu_only",
            "detection_status": "cpu_only_build",
            "diagnostic_message": "CPU only",
        }
        with patch(
            "backend.app.services.preflight_service.get_system_hardware_info",
            return_value=cpu_hw,
        ):
            payload = {**MINIMAL_CONFIG, "training_method": "qlora"}
            res = client.post("/api/training/preflight", json=payload)
        assert res.status_code == 200
        data = res.json()
        compat_check = next(c for c in data["checks"] if c["check_id"] == "method_device_compat")
        assert compat_check["status"] == "fail"
        assert data["can_launch"] is False


class TestPreflightHFToken:
    def test_hf_token_check_warns_without_token(self, client: TestClient) -> None:
        """Without a token, hf_token check returns warn."""
        res = client.post("/api/training/preflight", json=MINIMAL_CONFIG)
        token_check = next(c for c in res.json()["checks"] if c["check_id"] == "hf_token")
        assert token_check["status"] == "warn"

    def test_hf_token_check_passes_with_env_token(self, client: TestClient) -> None:
        """With hf_token_status=present_in_env, hf_token check passes."""
        payload = {**MINIMAL_CONFIG, "hf_token_status": "present_in_env"}
        res = client.post("/api/training/preflight", json=payload)
        token_check = next(c for c in res.json()["checks"] if c["check_id"] == "hf_token")
        assert token_check["status"] == "pass"

    def test_hf_token_from_session_passes(self, client: TestClient) -> None:
        """A hf_token provided in the payload (session token) yields pass."""
        payload = {**MINIMAL_CONFIG, "hf_token": "hf_fake_token_12345"}
        res = client.post("/api/training/preflight", json=payload)
        token_check = next(c for c in res.json()["checks"] if c["check_id"] == "hf_token")
        assert token_check["status"] == "pass"


class TestPreflightDeviceCheck:
    def test_device_check_passes_with_cuda_gpu(self, client: TestClient) -> None:
        """With a CUDA GPU detected, device check should pass."""
        gpu_hw = {
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
            "diagnostic_message": "Detected 1 GPU device(s).",
        }
        with patch(
            "backend.app.services.preflight_service.get_system_hardware_info",
            return_value=gpu_hw,
        ):
            res = client.post("/api/training/preflight", json=MINIMAL_CONFIG)
        assert res.status_code == 200
        device_check = next(c for c in res.json()["checks"] if c["check_id"] == "device")
        assert device_check["status"] == "pass"

    def test_device_check_warns_without_pytorch(self, client: TestClient) -> None:
        """Without pytorch, device check is fail (not warn)."""
        no_torch_hw = {
            "pytorch_available": False,
            "cuda_available": False,
            "gpu_available": False,
            "gpu_count": 0,
            "device_type": "cpu",
            "primary_gpu_name": None,
            "primary_gpu_vram_gb": None,
            "environment_type": "no_pytorch",
            "detection_status": "no_pytorch",
            "diagnostic_message": "PyTorch not installed.",
        }
        with patch(
            "backend.app.services.preflight_service.get_system_hardware_info",
            return_value=no_torch_hw,
        ):
            res = client.post("/api/training/preflight", json=MINIMAL_CONFIG)
        assert res.status_code == 200
        device_check = next(c for c in res.json()["checks"] if c["check_id"] == "device")
        assert device_check["status"] == "fail"


class TestPreflightVRAMEstimate:
    def test_estimated_vram_included_in_response(self, client: TestClient) -> None:
        """estimated_vram_gb should be None when no model is selected."""
        res = client.post("/api/training/preflight", json=MINIMAL_CONFIG)
        assert res.status_code == 200
        data = res.json()
        # No model_identifier → no param count → None
        assert data["estimated_vram_gb"] is None

    def test_vram_check_skips_without_gpu(self, client: TestClient) -> None:
        """VRAM check returns skip when no GPU is available and no estimate possible."""
        cpu_hw = {
            "pytorch_available": True,
            "cuda_available": False,
            "gpu_available": False,
            "gpu_count": 0,
            "device_type": "cpu",
            "primary_gpu_name": None,
            "primary_gpu_vram_gb": None,
            "environment_type": "cpu_only",
            "detection_status": "cpu_only_build",
            "diagnostic_message": "CPU only.",
        }
        with patch(
            "backend.app.services.preflight_service.get_system_hardware_info",
            return_value=cpu_hw,
        ):
            res = client.post("/api/training/preflight", json=MINIMAL_CONFIG)
        assert res.status_code == 200
        vram_check = next(c for c in res.json()["checks"] if c["check_id"] == "vram")
        assert vram_check["status"] == "skip"

    def test_blocking_count_matches_fail_check_count(self, client: TestClient) -> None:
        """blocking_count must equal the number of checks with status=='fail'."""
        res = client.post("/api/training/preflight", json=MINIMAL_CONFIG)
        data = res.json()
        actual_fails = sum(1 for c in data["checks"] if c["status"] == "fail")
        assert data["blocking_count"] == actual_fails

    def test_warning_count_matches_warn_check_count(self, client: TestClient) -> None:
        """warning_count must equal the number of checks with status=='warn'."""
        res = client.post("/api/training/preflight", json=MINIMAL_CONFIG)
        data = res.json()
        actual_warns = sum(1 for c in data["checks"] if c["status"] == "warn")
        assert data["warning_count"] == actual_warns
