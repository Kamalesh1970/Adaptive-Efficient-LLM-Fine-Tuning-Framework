"""
Training Configuration and Validation Schemas.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class TrainingMethod(str, Enum):
    """Allowed fine-tuning methods for Phase 0 UI."""
    LORA = "lora"
    QLORA = "qlora"
    FULL_FINETUNING = "full_finetuning"


class TrainingLengthType(str, Enum):
    """Unit of training duration."""
    STEPS = "steps"
    EPOCHS = "epochs"


class TrainingConfigSchema(BaseModel):
    """
    Initial Typed Training Configuration Schema.
    Contains only fields used by the Phase 0 Training UI.
    """
    project_name: str = Field(
        default="my-finetuned-model",
        description="Project identifier name"
    )
    model_identifier: Optional[str] = Field(
        default=None,
        description="Selected Hugging Face model identifier or local path"
    )
    training_method: TrainingMethod = Field(
        default=TrainingMethod.LORA,
        description="Fine-tuning method (UI option for Phase 0)"
    )
    dataset_reference: Optional[str] = Field(
        default=None,
        description="Selected dataset name, path, or uploaded reference"
    )
    dataset_subset: Optional[str] = Field(
        default=None,
        description="Optional dataset configuration subset"
    )
    train_split: str = Field(
        default="train",
        description="Dataset split used for training"
    )
    eval_split: str = Field(
        default="none",
        description="Dataset split used for evaluation ('none', 'test', 'validation')"
    )
    training_length_type: TrainingLengthType = Field(
        default=TrainingLengthType.STEPS,
        description="Duration type: 'steps' or 'epochs'"
    )
    training_length: int = Field(
        default=100,
        description="Number of steps or epochs to train"
    )
    batch_size: int = Field(
        default=2,
        description="Per-device training batch size"
    )
    context_length: int = Field(
        default=2048,
        description="Max token sequence context length"
    )
    learning_rate: float = Field(
        default=0.0002,
        description="Optimizer learning rate"
    )
    advanced_enabled: bool = Field(
        default=False,
        description="Whether advanced settings mode is active"
    )
    hf_token_status: str = Field(
        default="not_detected",
        description="Hugging Face API token status ('not_detected', 'present_in_env')"
    )

    @field_validator("project_name")
    @classmethod
    def validate_project_name(cls, v: str) -> str:
        cleaned = v.strip() if v else ""
        if not cleaned:
            raise ValueError("Project name cannot be empty or blank")
        return cleaned

    @field_validator("training_length")
    @classmethod
    def validate_training_length(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Training length must be greater than 0")
        if v > 1000000:
            raise ValueError("Training length exceeds maximum allowed value (1,000,000)")
        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Batch size must be greater than 0")
        if v > 512:
            raise ValueError("Batch size cannot exceed 512")
        return v

    @field_validator("context_length")
    @classmethod
    def validate_context_length(cls, v: int) -> int:
        if v < 128:
            raise ValueError("Context length must be at least 128 tokens")
        if v > 131072:
            raise ValueError("Context length cannot exceed 131,072 tokens")
        return v

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError("Learning rate must be strictly positive (> 0.0)")
        if v > 1.0:
            raise ValueError("Learning rate must be <= 1.0")
        return v


class TrainingConfigValidationResult(BaseModel):
    """Result of configuration validation."""
    is_valid: bool = Field(description="Whether configuration passes all schema rules")
    errors: List[str] = Field(default_factory=list, description="Validation failure messages")
    warnings: List[str] = Field(default_factory=list, description="Non-blocking recommendations")
    is_ready_for_training: bool = Field(
        default=False,
        description="Whether training can be started (always False in Phase 0)"
    )
    config: Optional[TrainingConfigSchema] = Field(
        default=None,
        description="Validated configuration instance if valid"
    )
