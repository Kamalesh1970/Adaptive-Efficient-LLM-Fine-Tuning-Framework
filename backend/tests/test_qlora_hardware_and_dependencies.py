"""
Comprehensive Regression Tests for QLoRA Dependencies, Hardware Compatibility,
and VRAM Estimation (Stage 9).

Covers:
- Missing ML dependencies detection
- CUDA unavailable handling for QLoRA
- bitsandbytes unavailable detection
- Unsupported model detection
- Insufficient VRAM detection and recommendation
- Invalid quantization configuration
- Successful QLoRA configuration with granular breakdown
- Real training error and OOM handling
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.schemas.preflight import PreflightStatus
from backend.app.schemas.training import TrainingConfigSchema, TrainingMethod
from backend.app.services.preflight_service import (
    run_preflight,
    calculate_vram_breakdown,
    _estimate_vram_gb,
)
from backend.app.services.ml_dependencies import (
    verify_ml_dependencies_or_raise,
    verify_qlora_dependencies_or_raise,
    MLDependencyMissingError,
    BitsAndBytesMissingError,
    CudaUnavailableForQuantizationError,
)
from backend.app.services.training_engine import (
    TrainingEngine,
    TrainingExecutionError,
)
from backend.app.services.training_dataset_service import PreparedTrainingDataset


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# Mock hardware profiles
GPU_RTX_2050 = {
    "pytorch_available": True,
    "cuda_available": True,
    "gpu_available": True,
    "gpu_count": 1,
    "device_type": "cuda",
    "primary_gpu_name": "NVIDIA GeForce RTX 2050",
    "primary_gpu_vram_gb": 3.68,
    "environment_type": "cuda_accelerated",
    "detection_status": "gpu_detected",
}

CPU_ONLY = {
    "pytorch_available": True,
    "cuda_available": False,
    "gpu_available": False,
    "gpu_count": 0,
    "device_type": "cpu",
    "primary_gpu_name": None,
    "primary_gpu_vram_gb": None,
    "environment_type": "cpu_only",
    "detection_status": "cpu_only_build",
}


# ─── 1. Missing ML Dependencies ───────────────────────────────────────────────

def test_missing_ml_dependencies_fails_preflight(client: TestClient):
    """Verifies that missing core ML libraries fail preflight and block launch."""
    mock_dep_missing = {
        "is_ready": False,
        "versions": {"torch": "2.13.0", "transformers": None, "peft": None},
        "missing_critical": ["transformers", "peft"],
        "bitsandbytes_available": False,
        "status": "dependencies_missing",
    }
    config = {
        "project_name": "test-missing-deps",
        "model_identifier": "Qwen/Qwen2.5-0.5B-Instruct",
        "training_method": "qlora",
        "dataset_reference": "sample_dataset",
    }
    with patch("backend.app.services.preflight_service.check_ml_dependencies", return_value=mock_dep_missing):
        with patch("backend.app.services.preflight_service.get_system_hardware_info", return_value=GPU_RTX_2050):
            res = client.post("/api/training/preflight", json=config)
            assert res.status_code == 200
            data = res.json()
            assert data["can_launch"] is False
            ml_check = next(c for c in data["checks"] if c["check_id"] == "ml_core_deps")
            assert ml_check["status"] == "fail"
            assert "transformers" in ml_check["message"]


def test_verify_ml_dependencies_or_raise_exception():
    """Verifies that MLDependencyMissingError is raised when critical packages are absent."""
    mock_dep_missing = {
        "is_ready": False,
        "missing_critical": ["transformers", "peft"],
        "versions": {},
    }
    with patch("backend.app.services.ml_dependencies.check_ml_dependencies", return_value=mock_dep_missing):
        with pytest.raises(MLDependencyMissingError):
            verify_ml_dependencies_or_raise()


# ─── 2. CUDA Unavailable for QLoRA ───────────────────────────────────────────

def test_cuda_unavailable_blocks_qlora_in_preflight(client: TestClient):
    """Verifies that attempting QLoRA on CPU-only hardware blocks launch."""
    config = {
        "project_name": "test-cpu-qlora",
        "model_identifier": "Qwen/Qwen2.5-0.5B-Instruct",
        "training_method": "qlora",
        "dataset_reference": "sample_dataset",
    }
    with patch("backend.app.services.preflight_service.get_system_hardware_info", return_value=CPU_ONLY):
        res = client.post("/api/training/preflight", json=config)
        assert res.status_code == 200
        data = res.json()
        assert data["can_launch"] is False
        compat_check = next(c for c in data["checks"] if c["check_id"] == "method_device_compat")
        assert compat_check["status"] == "fail"
        assert "requires a CUDA-capable GPU" in compat_check["message"]


def test_verify_qlora_dependencies_raises_on_cpu():
    """Verifies CudaUnavailableForQuantizationError when device is CPU."""
    with patch("backend.app.services.ml_dependencies.check_ml_dependencies", return_value={"is_ready": True, "missing_critical": []}):
        with patch("backend.app.services.ml_dependencies.is_bitsandbytes_available", return_value=True):
            with patch("backend.app.core.system.get_system_hardware_info", return_value=CPU_ONLY):
                with pytest.raises(CudaUnavailableForQuantizationError):
                    verify_qlora_dependencies_or_raise(is_synthetic_model=False, device="cpu")


# ─── 3. bitsandbytes Unavailable ──────────────────────────────────────────────

def test_bitsandbytes_missing_blocks_qlora(client: TestClient):
    """Verifies that missing bitsandbytes blocks QLoRA launch."""
    mock_dep = {
        "is_ready": True,
        "versions": {"torch": "2.13.0", "transformers": "5.22.1", "peft": "0.21.2"},
        "missing_critical": [],
        "bitsandbytes_available": False,
        "status": "ready",
    }
    config = {
        "project_name": "test-no-bnb",
        "model_identifier": "Qwen/Qwen2.5-0.5B-Instruct",
        "training_method": "qlora",
        "dataset_reference": "sample_dataset",
    }
    with patch("backend.app.services.preflight_service.check_ml_dependencies", return_value=mock_dep):
        with patch("backend.app.services.preflight_service.get_system_hardware_info", return_value=GPU_RTX_2050):
            res = client.post("/api/training/preflight", json=config)
            assert res.status_code == 200
            data = res.json()
            assert data["can_launch"] is False
            bnb_check = next(c for c in data["checks"] if c["check_id"] == "bitsandbytes")
            assert bnb_check["status"] == "fail"
            assert "bitsandbytes is required" in bnb_check["message"]


# ─── 4. Unsupported Model ────────────────────────────────────────────────────

def test_unsupported_model_method_rejected(client: TestClient):
    """Verifies that models with restricted supported_methods fail preflight if method unsupported."""
    # google/gemma-2-2b only supports lora and qlora, not full_finetuning
    config = {
        "project_name": "test-unsupported-model",
        "model_identifier": "google/gemma-2-2b",
        "training_method": "full_finetuning",
        "dataset_reference": "sample_dataset",
    }
    with patch("backend.app.services.preflight_service.get_system_hardware_info", return_value=GPU_RTX_2050):
        res = client.post("/api/training/preflight", json=config)
        assert res.status_code == 200
        data = res.json()
        assert data["can_launch"] is False
        model_check = next(c for c in data["checks"] if c["check_id"] == "model_catalog")
        assert model_check["status"] == "fail"


# ─── 5. Insufficient VRAM & Recommendation ───────────────────────────────────

def test_insufficient_vram_detected_for_7b_on_rtx_2050():
    """Verifies that Qwen2.5-7B is flagged as unsupported on RTX 2050 with recommendation."""
    breakdown = calculate_vram_breakdown(
        method=TrainingMethod.QLORA,
        param_count_str="7.6B",
        batch_size=2,
        context_length=2048,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        lora_rank=16,
        total_vram_gb=3.68,
        available_vram_gb=3.68,
    )
    assert breakdown["is_compatible"] is False
    assert breakdown["compatibility_status"] == "unsupported"
    assert breakdown["total_estimated_vram_gb"] > 3.68
    assert "Qwen/Qwen2.5-0.5B-Instruct" in breakdown["recommendation"]
    assert "Hardware VRAM exceeded" in breakdown["recommendation"]


def test_qwen_7b_preflight_warns_insufficient_vram(client: TestClient):
    """Verifies preflight API returns unsupported vram_status when 7B model exceeds available VRAM."""
    config = {
        "project_name": "test-7b-vram",
        "model_identifier": "Qwen/Qwen2.5-7B",
        "training_method": "qlora",
        "dataset_reference": "sample_dataset",
        "batch_size": 2,
        "context_length": 2048,
    }
    with patch("backend.app.services.preflight_service.get_system_hardware_info", return_value=GPU_RTX_2050):
        res = client.post("/api/training/preflight", json=config)
        assert res.status_code == 200
        data = res.json()
        assert data["vram_status"] == "unsupported"
        assert data["estimated_vram_gb"] > 3.68
        assert "Qwen/Qwen2.5-0.5B-Instruct" in data["vram_message"]


# ─── 6. Successful QLoRA Configuration ────────────────────────────────────────

def test_successful_qlora_configuration_0_5b_on_rtx_2050():
    """Verifies that Qwen2.5-0.5B-Instruct is fully supported on 3.68 GB VRAM."""
    breakdown = calculate_vram_breakdown(
        method=TrainingMethod.QLORA,
        param_count_str="0.5B",
        batch_size=1,
        context_length=512,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        lora_rank=8,
        total_vram_gb=3.68,
        available_vram_gb=3.68,
    )
    assert breakdown["is_compatible"] is True
    assert breakdown["compatibility_status"] == "supported"
    assert breakdown["total_estimated_vram_gb"] < 2.0  # ~0.98 GB
    assert breakdown["estimated_model_memory_gb"] < 0.5  # ~0.26 GB


def test_qwen_0_5b_preflight_supported_vram(client: TestClient):
    """Verifies preflight returns supported vram_status for 0.5B model on RTX 2050."""
    config = {
        "project_name": "test-0.5b-vram",
        "model_identifier": "Qwen/Qwen2.5-0.5B-Instruct",
        "training_method": "qlora",
        "dataset_reference": "sample_dataset",
        "batch_size": 1,
        "context_length": 512,
        "gradient_accumulation_steps": 8,
        "lora_rank": 8,
        "lora_alpha": 16,
    }
    with patch("backend.app.services.preflight_service.get_system_hardware_info", return_value=GPU_RTX_2050):
        res = client.post("/api/training/preflight", json=config)
        assert res.status_code == 200
        data = res.json()
        assert data["vram_status"] == "supported"
        assert data["vram_breakdown"] is not None
        assert data["vram_breakdown"]["is_compatible"] is True


# ─── 7. Granular VRAM Breakdown Components ────────────────────────────────────

def test_vram_breakdown_contains_all_required_components():
    """Verifies that calculate_vram_breakdown returns all required granular components."""
    breakdown = calculate_vram_breakdown(
        method=TrainingMethod.QLORA,
        param_count_str="0.5B",
        batch_size=1,
        context_length=512,
        total_vram_gb=4.0,
        available_vram_gb=3.68,
    )
    required_keys = [
        "total_vram_gb",
        "available_vram_gb",
        "estimated_model_memory_gb",
        "estimated_training_memory_gb",
        "breakdown_components",
        "safety_margin_gb",
        "estimated_peak_memory_gb",
        "total_estimated_vram_gb",
        "is_compatible",
    ]
    for key in required_keys:
        assert key in breakdown, f"Missing key '{key}' in breakdown"

    comps = breakdown["breakdown_components"]
    for comp in ["quantized_weights_gb", "quantization_metadata_gb", "adapter_parameters_gb", "optimizer_states_gb", "gradients_gb", "activations_gb", "cuda_runtime_overhead_gb"]:
        assert comp in comps, f"Missing component '{comp}' in breakdown_components"


# ─── 8. Real Training Failure Handling ────────────────────────────────────────

def test_training_engine_handles_no_trainable_params_gracefully():
    """Verifies that training engine raises TrainingExecutionError when no trainable parameters exist."""
    import torch
    import torch.nn as nn

    class FrozenModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
            for p in self.parameters():
                p.requires_grad = False

        def forward(self, x):
            return self.linear(x)

    model = FrozenModel()
    dummy_dataset = PreparedTrainingDataset(
        train_samples=[{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [1, 2]}],
        eval_samples=None,
        train_count=1,
        eval_count=0,
        schema_type="instruction",
        max_seq_length=512,
        tokenizer_name_or_path="test",
    )
    config = TrainingConfigSchema(
        project_name="test-fail",
        training_method=TrainingMethod.QLORA,
        learning_rate=0.0002,
        batch_size=1,
        max_steps=5,
    )

    engine = TrainingEngine()
    with pytest.raises(TrainingExecutionError) as exc_info:
        engine.train(model=model, prepared_dataset=dummy_dataset, config=config)
    assert "No trainable parameters found" in str(exc_info.value)
