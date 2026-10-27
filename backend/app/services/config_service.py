"""
Configuration Service for Training Config Defaults, Validation, and YAML Management (Phase 3).
"""

import os
import re
from typing import Dict, Any, List, Optional
import yaml
from pydantic import ValidationError

from backend.app.schemas.training import (
    TrainingConfigSchema,
    TrainingConfigValidationResult,
    TrainingMethod,
    TrainingLengthType,
)
from backend.app.core.system import get_system_hardware_info
from backend.app.services.model_catalog import get_model_catalog_service
from backend.app.services.dataset_service import get_dataset_service

SECRET_KEY_PATTERNS = {
    "token",
    "password",
    "secret",
    "api_key",
    "hf_token",
    "access_key",
    "credential",
    "auth",
}


def detect_hf_token_status() -> str:
    """Checks if a Hugging Face API token is present in the environment."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return "present_in_env"
    return "not_detected"


def get_default_training_config() -> TrainingConfigSchema:
    """
    Returns standard baseline training configuration.
    Explicitly starts with no model and no dataset selected.
    """
    return TrainingConfigSchema(
        project_name="my-finetuned-model",
        model_identifier=None,  # Clearly shows no model selected
        training_method=TrainingMethod.LORA,
        dataset_reference=None,  # Clearly shows no dataset selected
        dataset_subset=None,
        train_split="train",
        eval_split="none",
        training_length_type=TrainingLengthType.STEPS,
        training_duration_mode=TrainingLengthType.STEPS,
        training_length=100,
        max_steps=100,
        num_epochs=3.0,
        learning_rate=0.0002,
        batch_size=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        weight_decay=0.01,
        warmup_ratio=0.03,
        optimizer="adamw_8bit",
        seed=42,
        context_length=2048,
        max_seq_length=2048,
        truncation=True,
        packing=False,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        gradient_checkpointing=True,
        mixed_precision="fp16",
        quantization_type="nf4",
        advanced_enabled=False,
        hf_token_status=detect_hf_token_status(),
    )


def reset_training_config() -> TrainingConfigSchema:
    """
    Resets only the training configuration to baseline defaults.
    Does not touch model catalog or uploaded dataset files.
    """
    return get_default_training_config()


def validate_training_config(raw_data: Dict[str, Any]) -> TrainingConfigValidationResult:
    """
    Validates a training configuration dictionary against schema rules, model catalog,
    and dataset constraints. Distinguishes definite incompatibility from unknown compatibility.
    """
    errors: List[str] = []
    field_errors: Dict[str, List[str]] = {}
    warnings: List[str] = []
    parsed_config: Optional[TrainingConfigSchema] = None

    if isinstance(raw_data, TrainingConfigSchema):
        raw_data = raw_data.model_dump()

    try:
        parsed_config = TrainingConfigSchema(**raw_data)
    except ValidationError as e:
        for err in e.errors():
            loc = err["loc"]
            field_name = str(loc[-1]) if loc else "general"
            msg = err["msg"]
            if msg.startswith("Value error, "):
                msg = msg[len("Value error, "):]
            formatted_error = f"{field_name}: {msg}"
            errors.append(formatted_error)
            field_errors.setdefault(field_name, []).append(msg)

        return TrainingConfigValidationResult(
            is_valid=False,
            errors=errors,
            field_errors=field_errors,
            warnings=warnings,
            is_ready_for_training=False,
            config=None,
            effective_batch_size_per_device=None,
            total_effective_batch_size=None,
            detected_device_count=None,
        )

    # Hardware detection for effective batch size
    hw = get_system_hardware_info()
    cuda_available = hw.get("cuda_available", False)
    gpu_count = hw.get("gpu_count", 0)
    detected_device_count = gpu_count if (cuda_available and gpu_count > 0) else None

    effective_per_device = parsed_config.batch_size * parsed_config.gradient_accumulation_steps
    total_effective = (effective_per_device * detected_device_count) if detected_device_count is not None else None

    # Model reference and compatibility validation
    if parsed_config.model_identifier:
        model_service = get_model_catalog_service()
        model = model_service.get_model(parsed_config.model_identifier)
        if model:
            if parsed_config.training_method not in model.supported_methods:
                err_msg = f"Model '{model.name}' does not support method '{parsed_config.training_method.value}'"
                errors.append(f"training_method: {err_msg}")
                field_errors.setdefault("training_method", []).append(err_msg)
            if model.context_length is not None and parsed_config.context_length > model.context_length:
                warnings.append(
                    f"Configured context length ({parsed_config.context_length}) exceeds model native context ({model.context_length}). Verification status: RoPE scaling or truncation may be required."
                )
            if model.is_gated:
                warnings.append(f"Model '{model.name}' is gated. Ensure Hugging Face credentials are configured.")
        else:
            warnings.append(
                f"Model '{parsed_config.model_identifier}' is not registered in local catalog. Hardware and context compatibility has not yet been verified."
            )
    else:
        warnings.append("No base model selected. Select a model to proceed with fine-tuning.")

    # Dataset reference validation
    if parsed_config.dataset_reference:
        dataset_service = get_dataset_service()
        report = dataset_service.get_dataset(parsed_config.dataset_reference)
        if report:
            if not report.is_valid:
                err_msg = f"Selected dataset '{report.original_filename}' has validation errors and cannot be used for training"
                errors.append(f"dataset_reference: {err_msg}")
                field_errors.setdefault("dataset_reference", []).append(err_msg)
        else:
            warnings.append(
                f"Dataset '{parsed_config.dataset_reference}' not found in uploaded datasets. Remote Hugging Face dataset compatibility is unverified."
            )
    else:
        warnings.append("No dataset selected. Select or upload a dataset to proceed with fine-tuning.")

    # Phase notices
    warnings.append("Phase 4: Real LoRA fine-tuning engine active.")
    warnings.append("Phase 0 Notice: Training UI Skeleton active. Execution engine scheduled for Phase 4.")

    is_valid = len(errors) == 0
    is_ready_for_training = (
        is_valid
        and bool(parsed_config.model_identifier)
        and bool(parsed_config.dataset_reference)
        and parsed_config.training_method == TrainingMethod.LORA
    )

    return TrainingConfigValidationResult(
        is_valid=is_valid,
        errors=errors,
        field_errors=field_errors,
        warnings=warnings,
        is_ready_for_training=is_ready_for_training,
        config=parsed_config if is_valid else None,
        effective_batch_size_per_device=effective_per_device,
        total_effective_batch_size=total_effective,
        detected_device_count=detected_device_count,
    )


def export_training_config_yaml(config: TrainingConfigSchema) -> str:
    """
    Serializes a TrainingConfigSchema into a clean, safe YAML string.
    Guarantees secrets, API keys, passwords, and tokens are scrubbed from exported YAML.
    """
    raw_dict = config.model_dump()
    safe_dict: Dict[str, Any] = {}

    for k, v in raw_dict.items():
        k_lower = k.lower()
        # Strictly exclude any secret or token fields, except hf_token_status which is safe metadata
        if any(secret in k_lower for secret in SECRET_KEY_PATTERNS) and k != "hf_token_status":
            continue

        # Convert enums to string values
        if hasattr(v, "value"):
            safe_dict[k] = v.value
        else:
            safe_dict[k] = v

    header = (
        "# ====================================================================\n"
        "# Adaptive LLM Fine-Tuning Studio - Training Configuration\n"
        "# Safe YAML Export (Secrets excluded)\n"
        "# ====================================================================\n"
    )
    body = yaml.safe_dump(safe_dict, sort_keys=False, default_flow_style=False)
    return header + body


def parse_training_config_yaml(yaml_content: str) -> Dict[str, Any]:
    """
    Safely parses YAML content without executing arbitrary Python constructors or tags.
    Validates that root is a key-value mapping and scrubs any accidentally provided secrets.
    """
    if not isinstance(yaml_content, str):
        raise ValueError("YAML content must be a string")

    cleaned = yaml_content.strip()
    if not cleaned:
        raise ValueError("YAML content cannot be empty")

    try:
        # yaml.safe_load uses SafeLoader, completely preventing arbitrary code execution
        data = yaml.safe_load(cleaned)
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed YAML content: {str(exc)}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure: expected a mapping at root, got {type(data).__name__}")

    # Scrub any secret keys from input for security hygiene
    safe_dict: Dict[str, Any] = {}
    for k, v in data.items():
        k_str = str(k)
        if any(secret in k_str.lower() for secret in SECRET_KEY_PATTERNS) and k_str != "hf_token_status":
            continue
        safe_dict[k_str] = v

    return safe_dict
