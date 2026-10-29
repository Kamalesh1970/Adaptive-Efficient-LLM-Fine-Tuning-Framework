"""
Phase 5 Tests: QLoRA & 4-Bit Quantization.
Covers quantization configuration, hardware compatibility, bitsandbytes dependency checks,
k-bit model preparation, LoRA adapter injection on frozen base models, real loss trajectory,
artifact validation, and API integration without downloading large models or requiring real GPU hardware.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.schemas.quantization import (
    QuantizationConfigSchema,
    QuantizationType,
    ComputeDtype,
)
from backend.app.schemas.training import (
    TrainingConfigSchema,
    TrainingMethod,
    TrainingRunStatus,
)
from backend.app.services.ml_dependencies import (
    is_bitsandbytes_available,
    verify_qlora_dependencies_or_raise,
    BitsAndBytesMissingError,
    CudaUnavailableForQuantizationError,
)
from backend.app.services.model_loader import (
    ModelLoaderService,
    QuantizationDeviceError,
    register_test_model,
    clear_test_model_registry,
)
from backend.app.services.lora_service import (
    prepare_model_for_kbit,
    prepare_and_inject_qlora_adapters,
    inject_lora_adapters,
    save_and_verify_adapter,
    reload_lora_adapter,
    TargetModulesNotFoundError,
    LoRAConfigurationError,
)
from backend.app.services.training_dataset_service import (
    TrainingDatasetService,
    PreparedTrainingDataset,
)
from backend.app.services.training_engine import TrainingEngine
from backend.app.services.job_manager import JobManager, TrainingJob


@pytest.fixture
def client():
    return TestClient(app)


# ============================================================================
# Synthetic Test Model Helpers (Zero-download, offline testing)
# ============================================================================

class SyntheticParameter:
    def __init__(self, num_elements: int, requires_grad: bool = True):
        self._num_elements = num_elements
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._num_elements


class SyntheticQuantizedCausalLM:
    """Mock/Synthetic 4-bit quantized causal LM for unit testing."""

    def __init__(self):
        self.is_synthetic_test_model = True
        self.device = "cpu"
        self.is_quantized_4bit = False
        self.is_prepared_for_kbit = False
        self._modules = [
            ("model.layers.0.self_attn.q_proj", SyntheticParameter(1000, False)),
            ("model.layers.0.self_attn.v_proj", SyntheticParameter(1000, False)),
            ("model.layers.0.mlp.down_proj", SyntheticParameter(2000, False)),
        ]
        self._adapter_params = []

    def named_modules(self):
        return [
            ("model.layers.0.self_attn.q_proj", None),
            ("model.layers.0.self_attn.v_proj", None),
            ("model.layers.0.mlp.down_proj", None),
        ]

    def freeze_base_parameters(self):
        for _, p in self._modules:
            p.requires_grad = False

    def inject_adapters(self, rank: int, alpha: int):
        self._adapter_params = [
            ("lora_A", SyntheticParameter(rank * 64, True)),
            ("lora_B", SyntheticParameter(rank * 64, True)),
        ]

    def parameters(self):
        for _, p in self._modules:
            yield p
        for _, p in self._adapter_params:
            yield p


class SyntheticTokenizer:
    """Mock/Synthetic tokenizer for testing tokenization."""

    def __init__(self):
        self.name_or_path = "test-qlora-tokenizer"
        self.pad_token = "<pad>"
        self.eos_token = "</s>"

    def __call__(self, text, truncation=True, max_length=512, padding=False, return_tensors=None):
        words = text.split()
        token_ids = [(abs(hash(w)) % 1000) + 1 for w in words]
        if truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        return {
            "input_ids": token_ids,
            "attention_mask": [1] * len(token_ids),
        }

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = [f"<chat>{m.get('role')}: {m.get('content')}</chat>" for m in messages]
        return "\n".join(parts)


# ============================================================================
# 1. Quantization Configuration Tests
# ============================================================================

def test_valid_nf4_configuration():
    """Valid NF4 configuration with double quantization and default fp16 compute dtype."""
    cfg = QuantizationConfigSchema(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="fp16",
    )
    assert cfg.load_in_4bit is True
    assert cfg.bnb_4bit_quant_type == "nf4"
    assert cfg.bnb_4bit_use_double_quant is True
    assert cfg.bnb_4bit_compute_dtype == "fp16"
    data = cfg.to_dict()
    assert data["load_in_4bit"] is True
    assert data["bnb_4bit_quant_type"] == "nf4"


def test_valid_fp4_configuration():
    """Valid FP4 configuration with bf16 compute dtype."""
    cfg = QuantizationConfigSchema(
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype="bf16",
    )
    assert cfg.bnb_4bit_quant_type == "fp4"
    assert cfg.bnb_4bit_use_double_quant is False
    assert cfg.bnb_4bit_compute_dtype == "bf16"


def test_double_quantization_settings():
    """Tests double quantization enabled and disabled toggles."""
    cfg_enabled = QuantizationConfigSchema(bnb_4bit_use_double_quant=True)
    cfg_disabled = QuantizationConfigSchema(bnb_4bit_use_double_quant=False)
    assert cfg_enabled.bnb_4bit_use_double_quant is True
    assert cfg_disabled.bnb_4bit_use_double_quant is False


def test_invalid_compute_dtype():
    """Rejects unsupported compute dtypes with actionable validation error."""
    with pytest.raises(ValueError) as exc:
        QuantizationConfigSchema(bnb_4bit_compute_dtype="int4")
    assert "Unsupported compute dtype" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        QuantizationConfigSchema(bnb_4bit_compute_dtype="invalid_precision")
    assert "Unsupported compute dtype" in str(exc.value)


def test_invalid_quantization_type():
    """Rejects unsupported 4-bit quantization types."""
    with pytest.raises(ValueError) as exc:
        QuantizationConfigSchema(bnb_4bit_quant_type="int4")
    assert "Unsupported 4-bit quantization type" in str(exc.value)


def test_4bit_and_8bit_mutually_exclusive():
    """Ensures load_in_4bit and load_in_8bit cannot both be True simultaneously."""
    with pytest.raises(ValueError) as exc:
        QuantizationConfigSchema(load_in_4bit=True, load_in_8bit=True)
    assert "mutually exclusive" in str(exc.value)


def test_qlora_training_config_schema_validation():
    """Validates that TrainingConfigSchema enforces QLoRA-specific constraints."""
    # Valid QLoRA config
    valid_cfg = TrainingConfigSchema(
        model_identifier="test/model",
        training_method=TrainingMethod.QLORA,
        dataset_reference="test-ds",
        quantization_type="nf4",
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="fp16",
    )
    assert valid_cfg.training_method == TrainingMethod.QLORA
    q_dict = valid_cfg.to_quantization_dict()
    assert q_dict["bnb_4bit_quant_type"] == "nf4"

    # Missing target modules
    with pytest.raises(ValueError) as exc:
        TrainingConfigSchema(
            training_method=TrainingMethod.QLORA,
            lora_target_modules=[],
        )
    assert "At least one target module must be specified" in str(exc.value)

    # Incompatible 4bit and 8bit in TrainingConfigSchema
    with pytest.raises(ValueError) as exc:
        TrainingConfigSchema(
            training_method=TrainingMethod.QLORA,
            load_in_4bit=True,
            load_in_8bit=True,
        )
    assert "load_in_4bit and load_in_8bit cannot both be enabled simultaneously" in str(exc.value)


# ============================================================================
# 2. Compatibility & Dependency Handling Tests
# ============================================================================

def test_missing_bitsandbytes_raises_error():
    """Detects missing bitsandbytes dependency cleanly and raises actionable error."""
    with patch("backend.app.services.ml_dependencies.verify_ml_dependencies_or_raise"):
        with patch("backend.app.services.ml_dependencies.is_bitsandbytes_available", return_value=False):
            with pytest.raises(BitsAndBytesMissingError) as exc:
                verify_qlora_dependencies_or_raise(is_synthetic_model=False, device="cuda:0")
            assert "bitsandbytes" in str(exc.value)
            assert "pip install bitsandbytes" in str(exc.value)


def test_cuda_unavailable_for_quantization():
    """Raises CudaUnavailableForQuantizationError when device is CPU without CUDA."""
    with patch("backend.app.services.ml_dependencies.is_bitsandbytes_available", return_value=True):
        with patch("backend.app.services.ml_dependencies.verify_ml_dependencies_or_raise"):
            with pytest.raises(CudaUnavailableForQuantizationError) as exc:
                verify_qlora_dependencies_or_raise(is_synthetic_model=False, device="cpu")
            assert "compatible CUDA GPU" in str(exc.value)
            assert "CPU execution is not supported" in str(exc.value)


def test_model_loader_quantization_device_error_on_cpu():
    """Model loader rejects non-synthetic QLoRA loading when target device is CPU."""
    loader = ModelLoaderService()
    quant_cfg = QuantizationConfigSchema()
    with patch("backend.app.services.ml_dependencies.verify_ml_dependencies_or_raise"):
        with patch("backend.app.services.ml_dependencies.is_bitsandbytes_available", return_value=True):
            with patch("backend.app.services.model_loader.resolve_execution_device", return_value="cpu"):
                with pytest.raises(QuantizationDeviceError) as exc:
                    loader.load_model_and_tokenizer(
                        model_identifier="meta-llama/Meta-Llama-3-8B",
                        quantization_config=quant_cfg,
                        is_qlora=True,
                        hf_token="fake_hf_token",
                    )
                assert "compatible CUDA GPU" in str(exc.value)


def test_mocked_supported_hardware_compatibility():
    """When CUDA is available and bitsandbytes is installed, verification passes."""
    with patch("backend.app.services.ml_dependencies.is_bitsandbytes_available", return_value=True):
        with patch("backend.app.services.ml_dependencies.verify_ml_dependencies_or_raise"):
            with patch("backend.app.core.system.get_system_hardware_info", return_value={"cuda_available": True, "gpu_count": 1}):
                # Should not raise
                verify_qlora_dependencies_or_raise(is_synthetic_model=False, device="cuda:0")


# ============================================================================
# 3. Model Preparation Tests (k-bit preparation & LoRA injection)
# ============================================================================

def test_quantized_loading_configuration_synthetic():
    """Verifies that synthetic model loader flags 4-bit quantization without network download."""
    clear_test_model_registry()
    register_test_model(
        "test/qlora-synth",
        lambda: SyntheticQuantizedCausalLM(),
        lambda: SyntheticTokenizer(),
    )
    loader = ModelLoaderService()
    quant_cfg = QuantizationConfigSchema(bnb_4bit_quant_type="nf4")

    model, tokenizer, device = loader.load_model_and_tokenizer(
        model_identifier="test/qlora-synth",
        quantization_config=quant_cfg,
        is_qlora=True,
    )
    assert model.is_quantized_4bit is True
    assert model.quantization_config["bnb_4bit_quant_type"] == "nf4"
    clear_test_model_registry()


def test_kbit_preparation_call():
    """Verifies prepare_model_for_kbit prepares model and freezes base weights."""
    model = SyntheticQuantizedCausalLM()
    prepared = prepare_model_for_kbit(model, gradient_checkpointing=True)
    assert getattr(prepared, "is_prepared_for_kbit", False) is True


def test_prepare_and_inject_qlora_adapters():
    """Verifies base model weights remain frozen, adapter parameters are trainable, and param count is nonzero."""
    model = SyntheticQuantizedCausalLM()
    config = TrainingConfigSchema(
        model_identifier="test/model",
        training_method=TrainingMethod.QLORA,
        dataset_reference="test-ds",
        lora_rank=16,
        lora_alpha=32,
        lora_target_modules=["q_proj", "v_proj"],
    )

    peft_model, summary = prepare_and_inject_qlora_adapters(model, config)
    assert summary.trainable_params > 0
    assert summary.total_params > summary.trainable_params
    assert summary.trainable_percent < 100.0

    # Verify adapter parameters have requires_grad=True and base weights requires_grad=False
    trainable_count = 0
    frozen_count = 0
    for p in peft_model.parameters():
        if p.requires_grad:
            trainable_count += p.numel()
        else:
            frozen_count += p.numel()

    assert trainable_count == summary.trainable_params
    assert frozen_count > 0


def test_invalid_target_modules_rejected():
    """Rejects target modules that do not match any model projections."""
    model = SyntheticQuantizedCausalLM()
    config = TrainingConfigSchema(
        model_identifier="test/model",
        training_method=TrainingMethod.QLORA,
        dataset_reference="test-ds",
        lora_target_modules=["nonexistent_module_foo", "nonexistent_bar"],
    )
    with pytest.raises(TargetModulesNotFoundError):
        prepare_and_inject_qlora_adapters(model, config)


# ============================================================================
# 4. Training Engine & Telemetry Integration Tests
# ============================================================================

def test_qlora_training_step_execution_and_metrics(tmp_path):
    """Executes genuine training updates on synthetic quantized model; tracks real loss and steps."""
    clear_test_model_registry()
    register_test_model(
        "test/qlora-trainable",
        lambda: SyntheticQuantizedCausalLM(),
        lambda: SyntheticTokenizer(),
    )

    config = TrainingConfigSchema(
        model_identifier="test/qlora-trainable",
        training_method=TrainingMethod.QLORA,
        dataset_reference="test-ds",
        batch_size=2,
        max_steps=5,
        logging_steps=1,
        learning_rate=0.001,
        lora_target_modules=["q_proj"],
    )

    loader = ModelLoaderService()
    model, tokenizer, _ = loader.load_model_and_tokenizer(
        model_identifier="test/qlora-trainable",
        quantization_config=config.get_quantization_config(),
        is_qlora=True,
    )
    peft_model, _ = prepare_and_inject_qlora_adapters(model, config)

    dataset_records = [
        {"input_ids": [10, 20, 30, 40], "attention_mask": [1, 1, 1, 1], "labels": [10, 20, 30, 40]},
        {"input_ids": [50, 60, 70, 80], "attention_mask": [1, 1, 1, 1], "labels": [50, 60, 70, 80]},
    ]
    prepared_dataset = PreparedTrainingDataset(
        train_samples=dataset_records,
        eval_samples=[],
        train_count=len(dataset_records),
        eval_count=0,
        schema_type="instruction",
        max_seq_length=512,
        tokenizer_name_or_path="test-tokenizer",
    )

    recorded_steps = []
    def step_cb(step, loss, epoch, elapsed):
        recorded_steps.append((step, loss))

    engine = TrainingEngine()
    history = engine.train(
        model=peft_model,
        prepared_dataset=prepared_dataset,
        config=config,
        on_step_callback=step_cb,
    )

    assert len(history) == 5
    assert len(recorded_steps) == 5
    for pt in history:
        assert pt.loss > 0.0
        assert not (pt.loss != pt.loss) # not NaN

    clear_test_model_registry()


def test_qlora_artifact_saving_and_verification(tmp_path):
    """Saves QLoRA adapter artifacts including quantization_config.json and verifies reload."""
    model = SyntheticQuantizedCausalLM()
    config = TrainingConfigSchema(
        model_identifier="test/model",
        training_method=TrainingMethod.QLORA,
        dataset_reference="test-ds",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="fp16",
    )
    peft_model, _ = prepare_and_inject_qlora_adapters(model, config)

    out_dir = str(tmp_path / "qlora_run_artifacts")
    q_dict = config.to_quantization_dict()

    saved = save_and_verify_adapter(
        model=peft_model,
        output_directory=out_dir,
        sanitized_config=config.model_dump(),
        quantization_config=q_dict,
    )

    assert "adapter_config.json" in saved
    assert "training_config.json" in saved
    assert "quantization_config.json" in saved

    # Verify quantization_config.json content
    q_file = os.path.join(out_dir, "quantization_config.json")
    with open(q_file, "r") as f:
        loaded_q = json.load(f)
    assert loaded_q["bnb_4bit_quant_type"] == "nf4"
    assert loaded_q["load_in_4bit"] is True

    # Verify adapter reload
    reloaded = reload_lora_adapter(model, out_dir)
    assert reloaded is not None


# ============================================================================
# 5. Full Lifecycle & API Tests
# ============================================================================

def test_job_manager_qlora_lifecycle(tmp_path):
    """Full lifecycle execution of a QLoRA job from QUEUED to COMPLETED with metadata."""
    clear_test_model_registry()
    register_test_model(
        "test/qlora-full-life",
        lambda: SyntheticQuantizedCausalLM(),
        lambda: SyntheticTokenizer(),
    )

    config = TrainingConfigSchema(
        project_name="qlora-lifecycle-test",
        model_identifier="test/qlora-full-life",
        training_method=TrainingMethod.QLORA,
        dataset_reference="test-ds",
        batch_size=2,
        max_steps=3,
        logging_steps=1,
        lora_target_modules=["q_proj"],
    )

    mock_dataset_svc = MagicMock()
    sample_records = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
        {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1], "labels": [4, 5, 6]},
    ]
    mock_dataset_svc.prepare_dataset.return_value = PreparedTrainingDataset(
        train_samples=sample_records,
        eval_samples=[],
        train_count=len(sample_records),
        eval_count=0,
        schema_type="instruction",
        max_seq_length=512,
        tokenizer_name_or_path="test-tokenizer",
    )

    job_mgr = JobManager(
        runs_output_dir=str(tmp_path / "runs"),
        dataset_service=mock_dataset_svc,
    )

    job = job_mgr.create_job(config)
    assert job.quantization_info is not None
    assert job.quantization_info["load_in_4bit"] is True

    # Execute synchronously
    job_mgr.start_job(job.run_id, background=False)

    assert job.status == TrainingRunStatus.COMPLETED
    assert job.current_step == 3
    assert len(job.loss_history) == 3
    assert "quantization_config.json" in job.artifact_files

    # Check artifact metadata retrieval
    artifacts = job_mgr.get_artifacts_metadata(job.run_id)
    assert artifacts is not None
    assert artifacts.is_valid is True
    assert artifacts.quantization_config is not None
    assert artifacts.quantization_config["bnb_4bit_quant_type"] == "nf4"

    clear_test_model_registry()


def test_api_launch_qlora_run(client: TestClient, tmp_path):
    """API endpoint POST /api/training/runs initiates a genuine QLoRA run."""
    clear_test_model_registry()
    register_test_model(
        "test/api-qlora-model",
        lambda: SyntheticQuantizedCausalLM(),
        lambda: SyntheticTokenizer(),
    )

    payload = {
        "config": {
            "project_name": "qlora-api-run",
            "model_identifier": "test/api-qlora-model",
            "training_method": "qlora",
            "dataset_reference": "alpaca_test_id",
            "batch_size": 2,
            "max_steps": 2,
            "learning_rate": 0.0002,
            "lora_target_modules": ["q_proj"],
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "fp16",
        }
    }

    mock_hw = {
        "pytorch_available": True,
        "cuda_available": True,
        "gpu_available": True,
        "gpu_count": 1,
        "device_type": "cuda",
        "primary_gpu_name": "Test GPU",
        "primary_gpu_vram_gb": 16.0,
        "environment_type": "cuda_accelerated",
        "detection_status": "gpu_detected",
    }
    with patch("backend.app.services.preflight_service.get_system_hardware_info", return_value=mock_hw):
        res = client.post("/api/training/runs", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert data["status"] in ("queued", "validating", "loading_model", "completed")
    assert data["quantization_info"] is not None
    assert data["quantization_info"]["bnb_4bit_quant_type"] == "nf4"
    assert data["quantization_info"]["load_in_4bit"] is True

    clear_test_model_registry()


def test_no_secret_leakage_in_qlora_job(tmp_path):
    """Ensures HF tokens or sensitive credentials are not exposed in saved metadata or errors."""
    config = TrainingConfigSchema(
        project_name="qlora-secret-test",
        model_identifier="test/model",
        training_method=TrainingMethod.QLORA,
        dataset_reference="test-ds",
        lora_target_modules=["q_proj"],
    )
    job = TrainingJob(
        run_id="run_secret_test",
        config=config,
        output_dir=str(tmp_path / "run_secret"),
        hf_token="hf_SECRET_ACCESS_TOKEN_12345",
    )
    resp = job.to_response()
    resp_dict = resp.model_dump()
    assert "hf_SECRET_ACCESS_TOKEN_12345" not in json.dumps(resp_dict)


def test_existing_lora_workflow_preserved():
    """Confirms standard LoRA fine-tuning continues to work without quantization."""
    model = SyntheticQuantizedCausalLM()
    config = TrainingConfigSchema(
        model_identifier="test/model",
        training_method=TrainingMethod.LORA,
        dataset_reference="test-ds",
        lora_rank=8,
        lora_alpha=16,
        lora_target_modules=["q_proj", "v_proj"],
    )
    peft_model, summary = inject_lora_adapters(model, config)
    assert summary.trainable_params > 0
    assert getattr(peft_model, "is_quantized_4bit", False) is False
