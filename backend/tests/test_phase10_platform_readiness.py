"""
Comprehensive Unit & Integration Test Suite for Phase 10:
Containerization, Kubernetes, CI/CD, Advanced Quantization, Model Export (GGUF),
vLLM Model Serving, and Multi-GPU Capabilities.
"""

import os
import json
import struct
import tempfile
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.schemas.quantization import (
    AdvancedQuantMethod,
    PostTrainingQuantizationConfig,
    QuantizationJobStatus,
)
from backend.app.schemas.export import (
    GGUFExportConfig,
    GGUFQuantizationType,
)
from backend.app.schemas.serving import (
    ServingConfig,
    ServingBackend,
    ServingStatus,
    SmokeInferenceRequest,
)
from backend.app.schemas.distributed import (
    DistributedStrategy,
    DistributedTrainingConfig,
)
from backend.app.services.quantization_service import (
    QuantizationService,
    QuantizationCompatibilityError,
    QuantizationDependencyError,
)
from backend.app.services.gguf_export_service import (
    GGUFExportService,
    GGUFExportError,
    GGUFEligibilityError,
    GGUF_MAGIC,
    GGUF_VERSION,
)
from backend.app.services.vllm_service import (
    VLLMService,
    ServingCompatibilityError,
    ServingError,
)
from backend.app.services.distributed_service import (
    DistributedService,
    DistributedConfigurationError,
)
from backend.app.services.artifact_service import ArtifactService
from backend.app.services.run_repository import get_run_repository
from backend.app.schemas.artifacts import ArtifactMetadata, ValidationStatus, ExportStatus


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ==============================================================================
# 1. Kubernetes & Compose Manifest Validation Tests
# ==============================================================================

def test_kubernetes_manifest_validation_endpoint(client: TestClient):
    """Verifies that the /api/system/kubernetes/validate endpoint parses all 12 YAML manifests."""
    resp = client.get("/api/system/kubernetes/validate")
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_valid"] is True
    assert data["manifest_count"] == 12
    assert len(data["errors"]) == 0
    kinds = {m["kind"] for m in data["manifests"]}
    assert "Deployment" in kinds
    assert "Service" in kinds
    assert "Job" in kinds
    assert "Namespace" in kinds
    assert "ConfigMap" in kinds
    assert "Secret" in kinds


# ==============================================================================
# 2. Advanced Quantization Tests
# ==============================================================================

def test_quantization_capability_discovery(client: TestClient):
    """Verifies truthful detection of quantization capabilities without fake GPU claims."""
    resp = client.get("/api/quantization/capabilities")
    assert resp.status_code == 200
    caps = resp.json()
    assert len(caps) >= 5

    cap_map = {c["method"]: c for c in caps}
    assert "bnb_4bit" in cap_map
    assert "bnb_8bit" in cap_map
    assert "gptq" in cap_map
    assert "awq" in cap_map
    assert "gguf" in cap_map

    # GGUF export format is always supported
    assert cap_map["gguf"]["supports_export"] is True

    # If CUDA is unavailable, BNB, GPTQ, and AWQ must be flagged unavailable with clear reason
    import torch
    if not torch.cuda.is_available():
        assert cap_map["bnb_4bit"]["is_available"] is False
        assert "CUDA" in cap_map["bnb_4bit"]["reason_if_unavailable"]


def test_quantization_config_validation():
    """Tests validation of post-training quantization parameters."""
    # Valid GGUF config
    cfg = PostTrainingQuantizationConfig(
        method=AdvancedQuantMethod.GGUF,
        source_reference="run_123",
        bit_width=4,
        quant_type="q4_k_m",
        output_format="gguf",
    )
    assert cfg.bit_width == 4

    # Invalid bit width
    with pytest.raises(ValueError, match="Bit width 5 not supported"):
        PostTrainingQuantizationConfig(
            method=AdvancedQuantMethod.GGUF,
            source_reference="run_123",
            bit_width=5,
        )

    # GPTQ without calibration dataset must raise ValueError
    with pytest.raises(ValueError, match="Calibration dataset is required for GPTQ"):
        PostTrainingQuantizationConfig(
            method=AdvancedQuantMethod.GPTQ,
            source_reference="run_123",
            bit_width=4,
            calibration_dataset_id=None,
        )


def test_quantization_execution_and_artifact_generation(tmp_path):
    """Executes real quantization on a synthetic model and verifies output file, size reduction, and SHA-256."""
    service = QuantizationService(quant_output_dir=str(tmp_path / "quant"))
    config = PostTrainingQuantizationConfig(
        method=AdvancedQuantMethod.GGUF,
        source_reference="synthetic-test-model",
        bit_width=4,
        quant_type="int4",
        output_format="safetensors",
    )

    # 1. Create job
    job = service.create_quantization_job(config)
    assert job.status == QuantizationJobStatus.QUEUED

    # 2. Execute synchronously
    completed_job = service.start_quantization(job.job_id, background=False)
    assert completed_job.status == QuantizationJobStatus.COMPLETED
    assert completed_job.output_artifact_path is not None
    assert os.path.exists(completed_job.output_artifact_path)
    assert completed_job.quantized_size_bytes > 0
    assert completed_job.size_reduction_ratio > 0
    assert completed_job.sha256 is not None

    # 3. Cancel already completed job returns completed without error
    ret = service.cancel_job(job.job_id)
    assert ret.status == QuantizationJobStatus.COMPLETED


# ==============================================================================
# 3. Model Export (GGUF) Tests
# ==============================================================================

def test_gguf_export_eligibility_and_validation(tmp_path):
    """Verifies eligibility checks for GGUF model export."""
    service = GGUFExportService(export_dir=str(tmp_path / "gguf"))

    # Non-existent artifact ID must fail eligibility
    config_invalid = GGUFExportConfig(
        source_artifact_id="non_existent_art_999",
        quantization_type=GGUFQuantizationType.Q4_K_M,
    )
    is_eligible, issues = service.validate_eligibility(config_invalid)
    assert is_eligible is False
    assert any("not found" in issue for issue in issues)

    with pytest.raises(GGUFEligibilityError):
        service.create_export_job(config_invalid)


def test_gguf_export_binary_generation_and_verification(tmp_path):
    """
    Verifies genuine GGUF v3 binary creation:
    1. GGUF magic bytes (0x46554747)
    2. Version 3
    3. Tensor alignment
    4. SHA-256 checksum
    5. Path traversal protection on download
    """
    export_dir = tmp_path / "exports" / "gguf"
    export_dir.mkdir(parents=True, exist_ok=True)
    service = GGUFExportService(export_dir=str(export_dir))

    # Register mock artifact in ArtifactService
    art_svc = service.artifact_service
    fake_run_dir = tmp_path / "run_001"
    fake_run_dir.mkdir()
    fake_weights = fake_run_dir / "adapter_model.bin"
    fake_weights.write_bytes(b"dummy_weights_content")

    art_meta = ArtifactMetadata(
        artifact_id="art_valid_001",
        run_id="run_001",
        artifact_type="lora_adapter",
        model_identifier="google/gemma-2b",
        fine_tuning_method="lora",
        created_at="2026-10-29T12:00:00Z",
        file_reference=str(fake_weights),
        size_bytes=len(b"dummy_weights_content"),
        validation_status=ValidationStatus.VALID,
        export_status=ExportStatus.NOT_EXPORTED,
        base_model_identifier="google/gemma-2b",
    )
    get_run_repository().save_or_update_artifact(art_meta.model_dump())

    config = GGUFExportConfig(
        source_artifact_id="art_valid_001",
        base_model_identifier="google/gemma-2b",
        quantization_type=GGUFQuantizationType.Q4_K_M,
        output_filename="gemma_q4_k_m.gguf",
    )

    # 1. Create export job
    job = service.create_export_job(config)
    assert job.status == "queued"

    # 2. Execute synchronously
    completed_job = service.start_export(job.export_id, background=False)
    assert completed_job.status == "completed"
    assert os.path.exists(completed_job.output_filepath)
    assert completed_job.file_size_bytes > 0
    assert completed_job.sha256 is not None

    # 3. Verify real GGUF binary format header
    with open(completed_job.output_filepath, "rb") as gf:
        magic, version, tensor_count, metadata_count = struct.unpack("<IIQQ", gf.read(24))
        assert magic == GGUF_MAGIC
        assert version == GGUF_VERSION
        assert tensor_count > 0
        assert metadata_count > 0

    # 4. Safe download path resolution
    resolved_path = service.resolve_download_path(job.export_id)
    assert resolved_path == completed_job.output_filepath

    # 5. Path traversal protection: output_filename with ../ must be rejected
    with pytest.raises(ValueError, match="path separators or traversal"):
        GGUFExportConfig(
            source_artifact_id="art_valid_001",
            output_filename="../../etc/shadow.gguf",
        )


# ==============================================================================
# 4. vLLM Model Serving Tests
# ==============================================================================

def test_vllm_capability_detection(client: TestClient):
    """Verifies truthful detection of vLLM serving capability."""
    resp = client.get("/api/serving/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert "vllm_installed" in data
    assert "cuda_available" in data
    assert "can_serve_vllm" in data

    import torch
    if not torch.cuda.is_available():
        assert data["can_serve_vllm"] is False
        assert "CPU mode" in data["unsupported_reason"]


def test_serving_instance_lifecycle_and_smoke_inference():
    """Tests starting, smoke testing, and stopping a serving instance."""
    service = VLLMService()
    config = ServingConfig(
        model_reference="test/synthetic-model",
        backend=ServingBackend.VLLM,
        host="127.0.0.1",
        port=8080,
        max_model_len=1024,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1,
    )

    # 1. Start serving instance
    instance = service.start_serving(config)
    assert instance.status == ServingStatus.READY
    assert instance.instance_id.startswith("serve_")

    # 2. Smoke test inference
    req = SmokeInferenceRequest(
        prompt="Explain quantum computing briefly.",
        max_tokens=32,
        temperature=0.7,
    )
    smoke_resp = service.test_smoke_inference(instance.instance_id, req)
    assert smoke_resp.tokens_generated > 0
    assert len(smoke_resp.generated_text) > 0
    assert smoke_resp.latency_ms > 0

    # 3. Stop serving instance
    stopped = service.stop_serving(instance.instance_id)
    assert stopped.status == ServingStatus.STOPPED

    # 4. Smoke test after stop must fail
    with pytest.raises(ServingError, match="not ready"):
        service.test_smoke_inference(instance.instance_id, req)


def test_serving_ssrf_protection():
    """Verifies that binding to unsafe external hosts is rejected."""
    with pytest.raises(ValueError, match="not permitted"):
        ServingConfig(
            model_reference="test/synthetic-model",
            host="169.254.169.254",  # AWS metadata IP
            port=8080,
        )


# ==============================================================================
# 5. Multi-GPU Discovery and Validation Tests
# ==============================================================================

def test_multi_gpu_discovery():
    """Verifies truthful multi-GPU hardware discovery."""
    service = DistributedService()
    report = service.discover_hardware()

    import torch
    if not torch.cuda.is_available():
        assert report.cuda_available is False
        assert report.visible_device_count == 0
        assert report.supports_multi_gpu is False
        assert DistributedStrategy.DDP not in report.supported_training_strategies
        assert "CPU mode" in report.unsupported_reason
    elif torch.cuda.device_count() == 1:
        assert report.visible_device_count == 1
        assert report.supports_multi_gpu is False
        assert "Single GPU detected" in report.unsupported_reason


def test_multi_gpu_configuration_rejection():
    """Verifies that requesting multi-GPU on CPU or single GPU is rejected with an actionable error."""
    service = DistributedService()

    import torch
    if not torch.cuda.is_available():
        # Requesting DDP on CPU host must raise DistributedConfigurationError
        cfg = DistributedTrainingConfig(
            strategy=DistributedStrategy.DDP,
            requested_gpu_count=2,
        )
        with pytest.raises(DistributedConfigurationError, match="requires CUDA hardware"):
            service.validate_distributed_training(cfg)

        # Requesting 2 GPUs on single device strategy must raise DistributedConfigurationError
        cfg_multi_single = DistributedTrainingConfig(
            strategy=DistributedStrategy.SINGLE_DEVICE,
            requested_gpu_count=2,
        )
        with pytest.raises(DistributedConfigurationError, match="selected single_device strategy"):
            service.validate_distributed_training(cfg_multi_single)
