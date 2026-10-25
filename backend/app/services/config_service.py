"""
Configuration Service for Training Config Defaults and Validation.
"""

import os
from typing import Dict, Any, List
from pydantic import ValidationError

from backend.app.schemas.training import (
    TrainingConfigSchema,
    TrainingConfigValidationResult,
    TrainingMethod,
    TrainingLengthType,
)


def detect_hf_token_status() -> str:
    """Checks if a Hugging Face API token is present in the environment."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return "present_in_env"
    return "not_detected"


def get_default_training_config() -> TrainingConfigSchema:
    """
    Returns standard baseline training configuration for Phase 0.
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
        training_length=100,
        batch_size=2,
        context_length=2048,
        learning_rate=0.0002,
        advanced_enabled=False,
        hf_token_status=detect_hf_token_status(),
    )


def validate_training_config(raw_data: Dict[str, Any]) -> TrainingConfigValidationResult:
    """
    Validates a training configuration dictionary against rules and requirements.
    """
    errors: List[str] = []
    warnings: List[str] = []
    parsed_config: Optional[TrainingConfigSchema] = None

    try:
        parsed_config = TrainingConfigSchema(**raw_data)
    except ValidationError as e:
        for err in e.errors():
            field_name = ".".join(str(loc) for loc in err["loc"])
            msg = err["msg"]
            errors.append(f"{field_name}: {msg}")
        return TrainingConfigValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            is_ready_for_training=False,
            config=None,
        )

    # Readiness & business rules check
    if not parsed_config.model_identifier:
        warnings.append("No base model selected. Select a model to proceed with fine-tuning.")

    if not parsed_config.dataset_reference:
        warnings.append("No dataset selected. Select or upload a dataset to proceed with fine-tuning.")

    # Phase 0 Notice
    warnings.append("Phase 0 Notice: Training UI Skeleton active. Execution engine scheduled for Phase 4.")

    # Even if valid, training is NOT executable in Phase 0
    is_ready = bool(
        parsed_config.model_identifier
        and parsed_config.dataset_reference
        and len(errors) == 0
        and False  # Strictly False in Phase 0
    )

    return TrainingConfigValidationResult(
        is_valid=True,
        errors=[],
        warnings=warnings,
        is_ready_for_training=is_ready,
        config=parsed_config,
    )
