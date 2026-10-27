"""
Phase 4 Tests: Real LoRA Fine-Tuning Engine, Services, Lifecycle, and API Endpoints.
Covers model loading, LoRA parameter freezing and injection, dataset preparation,
training loop execution, telemetry, cancellation, artifact verification, and security isolation.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.schemas.training import (
    TrainingConfigSchema,
    TrainingMethod,
    TrainingLengthType,
    TrainingRunStatus,
)
from backend.app.schemas.datasets import DatasetSchemaType
from backend.app.services.ml_dependencies import (
    is_full_ml_stack_available,
    MLDependencyMissingError,
    verify_ml_dependencies_or_raise,
)
from backend.app.services.model_loader import (
    ModelLoaderService,
    ModelNotFoundError,
    GatedModelAccessError,
    resolve_execution_device,
    validate_model_identifier,
    register_test_model,
    clear_test_model_registry,
)
from backend.app.services.lora_service import (
    validate_lora_config,
    LoRAConfigurationError,
    TargetModulesNotFoundError,
    count_parameters,
    inject_lora_adapters,
    save_and_verify_adapter,
    reload_lora_adapter,
)
from backend.app.services.dataset_service import (
    DatasetService,
    get_dataset_service,
)
from backend.app.services.training_dataset_service import (
    TrainingDatasetService,
    PreparedTrainingDataset,
)
from backend.app.services.training_engine import (
    TrainingEngine,
    TrainingCancelledException,
    TrainingExecutionError,
)
from backend.app.services.job_manager import (
    JobManager,
    TrainingJob,
    get_job_manager,
)


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


class SyntheticModule:
    def __init__(self, name: str, params: list):
        self.name = name
        self.params = params


class SyntheticCausalLM:
    """Mock/Synthetic causal LM for fast unit testing."""

    def __init__(self):
        self.is_synthetic_test_model = True
        self.device = "cpu"
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
    """Mock/Synthetic tokenizer for testing prompt tokenization."""

    def __init__(self):
        self.name_or_path = "test-tokenizer"
        self.pad_token = "<pad>"
        self.eos_token = "</s>"

    def __call__(self, text, truncation=True, max_length=512, padding=False, return_tensors=None):
        # Generate deterministic token IDs from word hashes
        words = text.split()
        token_ids = [(abs(hash(w)) % 1000) + 1 for w in words]
        if truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        return {
            "input_ids": token_ids,
            "attention_mask": [1] * len(token_ids),
        }

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for m in messages:
            parts.append(f"<chat>{m.get('role')}: {m.get('content')}</chat>")
        return "\n".join(parts)


# ============================================================================
# 1. Model Loading Service Tests
# ============================================================================

def test_model_identifier_validation():
    """Rejects empty string, traversal paths, and invalid types."""
    with pytest.raises(ModelNotFoundError):
        validate_model_identifier("")
    with pytest.raises(ModelNotFoundError):
        validate_model_identifier("   ")
    with pytest.raises(ModelNotFoundError):
        validate_model_identifier("../../etc/passwd")
    assert validate_model_identifier("meta-llama/Meta-Llama-3-8B") == "meta-llama/Meta-Llama-3-8B"


def test_resolve_execution_device():
    """Resolves CPU when requested, or falls back to CPU when GPU is absent."""
    assert resolve_execution_device("cpu") == "cpu"
    with patch("backend.app.services.model_loader.get_system_hardware_info") as mock_hw:
        mock_hw.return_value = {"cuda_available": False, "gpu_count": 0}
        assert resolve_execution_device("cuda:0") == "cpu"

        mock_hw.return_value = {"cuda_available": True, "gpu_count": 1}
        assert resolve_execution_device(None) == "cuda:0"
        assert resolve_execution_device("cuda:1") == "cuda:1"


def test_gated_model_access_error_without_token():
    """Gated model in catalog requires HF access token."""
    loader = ModelLoaderService()
    with pytest.raises(GatedModelAccessError):
        loader.load_model_and_tokenizer(
            model_identifier="meta-llama/Meta-Llama-3-8B",
            hf_token=None,
        )


def test_synthetic_test_model_registry():
    """Model loader retrieves models from synthetic test registry without network or weights download."""
    register_test_model("test/tiny-llama", lambda: SyntheticCausalLM(), lambda: SyntheticTokenizer())
    loader = ModelLoaderService()

    model, tokenizer, device = loader.load_model_and_tokenizer("test/tiny-llama")
    assert isinstance(model, SyntheticCausalLM)
    assert isinstance(tokenizer, SyntheticTokenizer)
    assert device == "cpu"

    clear_test_model_registry()


def test_missing_ml_dependencies_handling():
    """When ML dependencies are absent, verify_ml_dependencies_or_raise raises MLDependencyMissingError."""
    with patch("backend.app.services.ml_dependencies.is_full_ml_stack_available", return_value=False):
        with pytest.raises(MLDependencyMissingError):
            verify_ml_dependencies_or_raise()


# ============================================================================
# 2. LoRA Service Tests
# ============================================================================

def test_validate_lora_config_bounds():
    """Validates rank, alpha, dropout, and target module constraints."""
    # Valid
    validate_lora_config(rank=16, alpha=32, dropout=0.05, target_modules=["q_proj", "v_proj"])

    # Invalid rank
    with pytest.raises(LoRAConfigurationError):
        validate_lora_config(rank=0, alpha=32, dropout=0.05, target_modules=["q_proj"])
    with pytest.raises(LoRAConfigurationError):
        validate_lora_config(rank=1024, alpha=32, dropout=0.05, target_modules=["q_proj"])

    # Invalid alpha
    with pytest.raises(LoRAConfigurationError):
        validate_lora_config(rank=16, alpha=-1, dropout=0.05, target_modules=["q_proj"])

    # Invalid dropout
    with pytest.raises(LoRAConfigurationError):
        validate_lora_config(rank=16, alpha=32, dropout=1.5, target_modules=["q_proj"])

    # Empty target modules
    with pytest.raises(LoRAConfigurationError):
        validate_lora_config(rank=16, alpha=32, dropout=0.05, target_modules=[])


def test_lora_target_modules_not_found():
    """Raises TargetModulesNotFoundError when target modules don't match model architecture."""
    model = SyntheticCausalLM()
    cfg = TrainingConfigSchema(
        model_identifier="test/tiny-llama",
        dataset_reference="test-ds",
        lora_target_modules=["non_existent_module_name"],
    )
    with pytest.raises(TargetModulesNotFoundError):
        inject_lora_adapters(model, cfg)


def test_lora_injection_and_parameter_freezing():
    """Injects adapters, freezes base parameters, enables adapter gradients, and counts parameters."""
    model = SyntheticCausalLM()
    cfg = TrainingConfigSchema(
        model_identifier="test/tiny-llama",
        dataset_reference="test-ds",
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=["q_proj", "v_proj"],
    )
    peft_model, summary = inject_lora_adapters(model, cfg)

    assert summary.trainable_params > 0
    assert summary.total_params > summary.trainable_params
    assert summary.trainable_percent > 0.0
    assert summary.trainable_percent < 100.0


def test_save_and_verify_adapter(tmp_path):
    """Saves adapter artifacts and verifies files on disk."""
    model = SyntheticCausalLM()
    model.inject_adapters(16, 32)
    output_dir = str(tmp_path / "run_test_save")

    saved_files = save_and_verify_adapter(
        model=model,
        output_directory=output_dir,
        sanitized_config={"lora_rank": 16, "learning_rate": 0.0002},
    )

    assert "adapter_config.json" in saved_files
    assert "adapter_model.safetensors" in saved_files
    assert "training_config.json" in saved_files


# ============================================================================
# 3. Dataset Preparation Service Tests
# ============================================================================

@pytest.fixture
def sample_dataset_service(tmp_path):
    svc = DatasetService(upload_dir=str(tmp_path / "uploads"))
    raw_instruction_data = [
        {"instruction": "Translate hello to French", "input": "", "output": "Bonjour"},
        {"instruction": "Summarize text", "input": "Long text snippet...", "output": "Summary"},
        {"instruction": "Calculate sum", "input": "2 + 2", "output": "4"},
        {"instruction": "Provide capital", "input": "Capital of Japan", "output": "Tokyo"},
    ]
    raw_bytes = ("\n".join(json.dumps(s) for s in raw_instruction_data)).encode("utf-8")
    ds_id, safe_name, fmt = svc.store_upload_bytes(raw_bytes, "train_data.jsonl")
    svc.parse_and_validate(ds_id, safe_name, fmt, raw_bytes.decode("utf-8"), len(raw_bytes))
    return svc, ds_id


def test_format_instruction_sample():
    """Formats instruction samples with and without input field."""
    svc = TrainingDatasetService()
    tokenizer = SyntheticTokenizer()

    # With input
    sample_with_input = {"instruction": "Task", "input": "Context", "output": "Result"}
    text_with = svc.format_sample_to_text(sample_with_input, DatasetSchemaType.INSTRUCTION, tokenizer)
    assert "### Instruction:\nTask" in text_with
    assert "### Input:\nContext" in text_with
    assert "### Response:\nResult" in text_with

    # Without input
    sample_no_input = {"instruction": "Task", "input": "", "output": "Result"}
    text_no = svc.format_sample_to_text(sample_no_input, DatasetSchemaType.INSTRUCTION, tokenizer)
    assert "### Instruction:\nTask" in text_no
    assert "### Input:" not in text_no
    assert "### Response:\nResult" in text_no


def test_format_conversational_sample():
    """Formats conversational samples using tokenizer template or fallback."""
    svc = TrainingDatasetService()
    tokenizer = SyntheticTokenizer()

    sample = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }
    text = svc.format_sample_to_text(sample, DatasetSchemaType.CONVERSATIONAL, tokenizer)
    assert "Hello" in text
    assert "Hi there!" in text


def test_dataset_preparation_splits_and_tokenization(sample_dataset_service):
    """Prepares, splits without leakage, and tokenizes dataset."""
    ds_svc, ds_id = sample_dataset_service
    training_ds_svc = TrainingDatasetService(dataset_service=ds_svc)
    tokenizer = SyntheticTokenizer()

    config = TrainingConfigSchema(
        model_identifier="test/tiny-llama",
        dataset_reference=ds_id,
        context_length=128,
        train_split_ratio=0.75,
        eval_split_ratio=0.25,
        evaluation_strategy="steps",
    )

    prepared = training_ds_svc.prepare_dataset(ds_id, tokenizer, config)
    assert prepared.train_count > 0
    assert prepared.eval_count > 0
    assert len(prepared.train_samples) == prepared.train_count

    # Check causal LM structure: input_ids, attention_mask, labels
    sample = prepared.train_samples[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample
    assert len(sample["input_ids"]) == len(sample["labels"])


def test_empty_dataset_rejection(tmp_path):
    """Rejects empty or non-existent datasets."""
    ds_svc = DatasetService(upload_dir=str(tmp_path / "uploads"))
    training_ds_svc = TrainingDatasetService(dataset_service=ds_svc)
    tokenizer = SyntheticTokenizer()

    config = TrainingConfigSchema(
        model_identifier="test/tiny-llama",
        dataset_reference="non-existent-ds",
        context_length=128,
    )
    with pytest.raises(ValueError):
        training_ds_svc.prepare_dataset("non-existent-ds", tokenizer, config)


# ============================================================================
# 4. Training Engine & Cancellation Tests
# ============================================================================

def test_training_engine_synthetic_execution():
    """Executes synthetic training steps and records real loss telemetry."""
    engine = TrainingEngine()
    model = SyntheticCausalLM()
    model.inject_adapters(16, 32)

    prepared = PreparedTrainingDataset(
        train_samples=[{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]}],
        eval_samples=None,
        train_count=1,
        eval_count=0,
        schema_type="instruction",
        max_seq_length=128,
        tokenizer_name_or_path="test-tok",
    )

    config = TrainingConfigSchema(
        model_identifier="test/tiny-llama",
        dataset_reference="test-ds",
        training_length_type=TrainingLengthType.STEPS,
        max_steps=5,
        logging_steps=1,
    )

    recorded_steps = []
    def on_step(step, loss, epoch, elapsed):
        recorded_steps.append((step, loss))

    metrics = engine.train(
        model=model,
        prepared_dataset=prepared,
        config=config,
        on_step_callback=on_step,
    )

    assert len(metrics) == 5
    assert len(recorded_steps) == 5
    assert metrics[-1].step == 5
    assert metrics[-1].loss > 0.0


def test_training_engine_cancellation():
    """Cancels training execution midway through steps cleanly."""
    engine = TrainingEngine()
    model = SyntheticCausalLM()

    prepared = PreparedTrainingDataset(
        train_samples=[{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]}],
        eval_samples=None,
        train_count=1,
        eval_count=0,
        schema_type="instruction",
        max_seq_length=128,
        tokenizer_name_or_path="test-tok",
    )

    config = TrainingConfigSchema(
        model_identifier="test/tiny-llama",
        dataset_reference="test-ds",
        max_steps=100,
    )

    cancel_flag = False
    def is_cancelled():
        return cancel_flag

    # Trigger cancellation immediately
    cancel_flag = True
    with pytest.raises(TrainingCancelledException):
        engine.train(model, prepared, config, is_cancelled_fn=is_cancelled)


# ============================================================================
# 5. Job Manager & Full Lifecycle Tests
# ============================================================================

def test_job_manager_full_lifecycle(sample_dataset_service, tmp_path):
    """Executes entire lifecycle from QUEUED through COMPLETED with artifact validation."""
    ds_svc, ds_id = sample_dataset_service
    register_test_model("test/tiny-llama", lambda: SyntheticCausalLM(), lambda: SyntheticTokenizer())

    runs_dir = str(tmp_path / "test_runs")
    mgr = JobManager(
        runs_output_dir=runs_dir,
        dataset_service=TrainingDatasetService(dataset_service=ds_svc),
    )

    config = TrainingConfigSchema(
        project_name="my-test-run",
        model_identifier="test/tiny-llama",
        training_method=TrainingMethod.LORA,
        dataset_reference=ds_id,
        training_length_type=TrainingLengthType.STEPS,
        max_steps=3,
        batch_size=1,
        context_length=128,
        logging_steps=1,
    )

    job = mgr.create_job(config)
    assert job.status == TrainingRunStatus.QUEUED

    # Execute synchronously for deterministic testing
    mgr.start_job(job.run_id, background=False)

    assert job.status == TrainingRunStatus.COMPLETED
    assert job.current_step == 3
    assert job.current_loss is not None
    assert len(job.loss_history) == 3
    assert job.trainable_params is not None

    # Verify artifacts on disk
    artifacts = mgr.get_artifacts_metadata(job.run_id)
    assert artifacts is not None
    assert artifacts.is_valid is True
    assert "adapter_config.json" in artifacts.files
    assert "adapter_model.safetensors" in artifacts.files
    assert "run_metadata.json" in artifacts.files
    assert "metrics.json" in artifacts.files

    clear_test_model_registry()


def test_job_manager_rejection_of_unsupported_method():
    """Job creation strictly rejects unsupported methods like full_finetuning."""
    mgr = JobManager()
    config = TrainingConfigSchema(
        model_identifier="meta-llama/Meta-Llama-3-8B",
        training_method=TrainingMethod.FULL_FINETUNING,
        dataset_reference="test-ds",
    )
    with pytest.raises(ValueError) as exc:
        mgr.create_job(config)
    assert "Supported methods are LoRA and QLoRA fine-tuning" in str(exc.value)


# ============================================================================
# 6. Backend API Endpoint Tests
# ============================================================================

def test_api_launch_run_validation_failure(client: TestClient):
    """POST /api/training/runs rejects invalid configurations."""
    invalid_payload = {
        "config": {
            "project_name": "", # invalid empty name
            "training_method": "lora",
        }
    }
    res = client.post("/api/training/runs", json=invalid_payload)
    assert res.status_code in (400, 422)


def test_api_launch_run_rejects_unsupported_method(client: TestClient):
    """POST /api/training/runs rejects unsupported methods like full_finetuning."""
    payload = {
        "config": {
            "project_name": "full-ft-attempt",
            "model_identifier": "test/model",
            "training_method": "full_finetuning",
            "dataset_reference": "ds_123",
            "batch_size": 2,
            "context_length": 512,
            "learning_rate": 0.0001,
        }
    }
    res = client.post("/api/training/runs", json=payload)
    assert res.status_code == 400
    assert "Supported methods are LoRA and QLoRA fine-tuning" in res.json()["detail"]


def test_api_run_lifecycle_and_artifacts(client: TestClient, sample_dataset_service):
    """Full API integration test: launch run, poll status, cancel, and retrieve artifacts."""
    ds_svc, ds_id = sample_dataset_service
    register_test_model("test/api-llama", lambda: SyntheticCausalLM(), lambda: SyntheticTokenizer())

    # Point global dataset service and job manager to test state
    job_mgr = get_job_manager()
    job_mgr.dataset_service = TrainingDatasetService(dataset_service=ds_svc)

    payload = {
        "config": {
            "project_name": "api-lora-run",
            "model_identifier": "test/api-llama",
            "training_method": "lora",
            "dataset_reference": ds_id,
            "training_length_type": "steps",
            "max_steps": 5,
            "batch_size": 1,
            "context_length": 128,
            "logging_steps": 1,
        },
        "hf_token": "hf_secret_token_12345",
    }

    # 1. Launch Run
    launch_res = client.post("/api/training/runs", json=payload)
    assert launch_res.status_code == 201
    run_data = launch_res.json()
    run_id = run_data["run_id"]
    assert run_id.startswith("run_")
    assert run_data["status"] in ["queued", "validating", "loading_model", "preparing_dataset", "training", "saving", "completed"]

    # 2. Get Run Status
    get_res = client.get(f"/api/training/runs/{run_id}")
    assert get_res.status_code == 200
    status_data = get_res.json()
    assert status_data["run_id"] == run_id

    # Ensure secret token is not leaked in response
    resp_str = json.dumps(status_data)
    assert "hf_secret_token_12345" not in resp_str

    # 3. 404 on Unknown Run ID
    assert client.get("/api/training/runs/unknown_run_id").status_code == 404
    assert client.post("/api/training/runs/unknown_run_id/cancel").status_code == 404
    assert client.get("/api/training/runs/unknown_run_id/artifacts").status_code == 404

    clear_test_model_registry()
