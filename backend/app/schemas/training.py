"""
Training Configuration and Validation Schemas for Adaptive LLM Fine-Tuning Studio (Phase 3).
"""

import math
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationInfo


class TrainingMethod(str, Enum):
    """Allowed fine-tuning methods."""
    LORA = "lora"
    QLORA = "qlora"
    FULL_FINETUNING = "full_finetuning"


class TrainingLengthType(str, Enum):
    """Unit of training duration."""
    STEPS = "steps"
    EPOCHS = "epochs"


class OptimizerType(str, Enum):
    """Supported optimizers."""
    ADAMW_8BIT = "adamw_8bit"
    ADAMW_TORCH = "adamw_torch"
    PAGED_ADAMW_8BIT = "paged_adamw_8bit"
    PAGED_ADAMW_32BIT = "paged_adamw_32bit"
    ADAMW_HF = "adamw_hf"
    SGD = "sgd"
    ADAFACTOR = "adafactor"


class MixedPrecisionPreference(str, Enum):
    """Supported precision preferences."""
    FP16 = "fp16"
    BF16 = "bf16"
    NONE = "none"
    FP32 = "fp32"


class QuantizationPreference(str, Enum):
    """Supported quantization types for QLoRA."""
    NF4 = "nf4"
    FP4 = "fp4"
    INT8 = "int8"
    NONE = "none"


ALLOWED_OPTIMIZERS = {opt.value for opt in OptimizerType}
ALLOWED_PRECISIONS = {prec.value for prec in MixedPrecisionPreference}
ALLOWED_QUANTIZATIONS = {q.value for q in QuantizationPreference}.union({"4bit", "8bit", None})

DEFAULT_LORA_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _check_finite(val: Any, field_name: str) -> None:
    """Rejects NaN and infinite numeric values."""
    if isinstance(val, bool):
        return
    if isinstance(val, (int, float)):
        if math.isnan(val):
            raise ValueError(f"{field_name} cannot be NaN")
        if math.isinf(val):
            raise ValueError(f"{field_name} cannot be infinite")


class TrainingConfigSchema(BaseModel):
    """
    Structured, typed, and validated Training Configuration Schema.
    """
    # General Settings
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
        description="Fine-tuning method: lora, qlora, full_finetuning"
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
    train_split_ratio: Optional[float] = Field(
        default=0.9,
        description="Proportion of dataset samples allocated to training"
    )
    eval_split_ratio: Optional[float] = Field(
        default=0.1,
        description="Proportion of dataset samples allocated to evaluation"
    )
    logging_steps: Optional[int] = Field(
        default=10,
        description="Interval of steps between recording metrics"
    )

    # Duration Settings
    training_length_type: TrainingLengthType = Field(
        default=TrainingLengthType.STEPS,
        description="Duration type: 'steps' or 'epochs'"
    )
    training_duration_mode: Optional[TrainingLengthType] = Field(
        default=None,
        description="Alias for training_length_type"
    )
    training_length: int = Field(
        default=100,
        description="Number of steps or epochs to train (legacy/shared field)"
    )
    max_steps: int = Field(
        default=100,
        description="Maximum training steps (when duration mode is steps)"
    )
    num_epochs: float = Field(
        default=3.0,
        description="Number of epochs to train (when duration mode is epochs)"
    )

    # Optimization Settings
    learning_rate: float = Field(
        default=0.0002,
        description="Optimizer learning rate"
    )
    batch_size: int = Field(
        default=2,
        description="Per-device training batch size"
    )
    per_device_train_batch_size: Optional[int] = Field(
        default=None,
        description="Alias for batch_size"
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        description="Number of gradient accumulation steps"
    )
    weight_decay: float = Field(
        default=0.01,
        description="Weight decay for optimizer regularization"
    )
    warmup_ratio: float = Field(
        default=0.03,
        description="Ratio of total training steps used for linear learning rate warmup"
    )
    optimizer: str = Field(
        default="adamw_8bit",
        description="Optimizer identifier (e.g. adamw_8bit, adamw_torch)"
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )

    # Sequence Settings
    context_length: int = Field(
        default=2048,
        description="Maximum token sequence context length"
    )
    max_seq_length: Optional[int] = Field(
        default=None,
        description="Alias for context_length"
    )
    truncation: bool = Field(
        default=True,
        description="Whether input sequences exceeding context length are truncated"
    )
    packing: bool = Field(
        default=False,
        description="Configuration preference for sequence packing"
    )

    # LoRA / QLoRA Settings
    lora_rank: int = Field(
        default=16,
        description="LoRA adapter rank dimension"
    )
    lora_alpha: int = Field(
        default=32,
        description="LoRA scaling alpha parameter"
    )
    lora_dropout: float = Field(
        default=0.05,
        description="LoRA adapter dropout probability"
    )
    lora_target_modules: List[str] = Field(
        default_factory=lambda: list(DEFAULT_LORA_MODULES),
        description="Target module names for LoRA weight updates"
    )

    # Memory and Precision Preferences
    gradient_checkpointing: bool = Field(
        default=True,
        description="Preference for gradient checkpointing to reduce VRAM consumption"
    )
    mixed_precision: str = Field(
        default="fp16",
        description="Mixed precision preference: fp16, bf16, or none"
    )
    quantization_type: Optional[str] = Field(
        default="nf4",
        description="Quantization preference (nf4, fp4, int8, none)"
    )
    load_in_4bit: bool = Field(
        default=True,
        description="Whether to load base model in 4-bit precision for QLoRA"
    )
    load_in_8bit: bool = Field(
        default=False,
        description="Whether to load base model in 8-bit precision"
    )
    bnb_4bit_quant_type: str = Field(
        default="nf4",
        description="4-bit quantization data format ('nf4' or 'fp4')"
    )
    bnb_4bit_use_double_quant: bool = Field(
        default=True,
        description="Whether to use nested/double quantization to compress quantization constants"
    )
    bnb_4bit_compute_dtype: str = Field(
        default="fp16",
        description="Compute precision dtype for 4-bit weights ('fp16', 'bf16', 'float32')"
    )

    # UI / Security State
    advanced_enabled: bool = Field(
        default=False,
        description="Whether advanced settings mode is active"
    )
    hf_token_status: str = Field(
        default="not_detected",
        description="Hugging Face API token status ('not_detected', 'present_in_env')"
    )

    @model_validator(mode="before")
    @classmethod
    def sync_aliases_and_check_numbers(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # Check for NaN and Inf in raw numeric fields and string representations
        for key, val in data.items():
            if isinstance(val, str) and val.strip().lower() in ("nan", "inf", "-inf", "+inf", "infinity", "-infinity"):
                raise ValueError(f"{key} cannot be NaN or infinite")
            if not isinstance(val, bool) and isinstance(val, (float, int)):
                _check_finite(val, key)

        # Sync batch size aliases
        if "per_device_train_batch_size" in data and "batch_size" not in data:
            data["batch_size"] = data["per_device_train_batch_size"]
        elif "batch_size" in data and "per_device_train_batch_size" not in data:
            data["per_device_train_batch_size"] = data["batch_size"]

        # Sync context length aliases
        if "max_seq_length" in data and "context_length" not in data:
            data["context_length"] = data["max_seq_length"]
        elif "context_length" in data and "max_seq_length" not in data:
            data["max_seq_length"] = data["context_length"]

        # Sync duration mode aliases
        if "training_duration_mode" in data and "training_length_type" not in data:
            data["training_length_type"] = data["training_duration_mode"]
        elif "training_length_type" in data and "training_duration_mode" not in data:
            data["training_duration_mode"] = data["training_length_type"]

        # Sync duration numbers
        mode = data.get("training_length_type") or data.get("training_duration_mode") or "steps"
        if mode == "steps":
            if "max_steps" in data and "training_length" not in data:
                data["training_length"] = data["max_steps"]
            elif "training_length" in data and "max_steps" not in data:
                data["max_steps"] = data["training_length"]
        elif mode == "epochs":
            if "num_epochs" in data and "training_length" not in data:
                try:
                    data["training_length"] = int(round(float(data["num_epochs"])))
                except (ValueError, TypeError):
                    pass
            elif "training_length" in data and "num_epochs" not in data:
                data["num_epochs"] = float(data["training_length"])

        return data

    @field_validator("project_name")
    @classmethod
    def validate_project_name(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Project name must be a string")
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Project name cannot be empty or blank")
        if len(cleaned) > 128:
            raise ValueError("Project name cannot exceed 128 characters")
        return cleaned

    @field_validator("model_identifier", "dataset_reference", "dataset_subset")
    @classmethod
    def validate_optional_references(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("Reference identifier must be a string")
        cleaned = v.strip()
        if not cleaned:
            return None
        return cleaned

    @field_validator("training_length", "max_steps")
    @classmethod
    def validate_training_length(cls, v: Optional[int], info: ValidationInfo) -> Optional[int]:
        if v is None:
            return None
        fname = info.field_name or "training_length"
        _check_finite(v, fname)
        if v <= 0:
            raise ValueError(f"{fname} must be greater than 0")
        if v > 1000000:
            raise ValueError(f"{fname} exceeds maximum allowed value (1,000,000)")
        return v

    @field_validator("num_epochs")
    @classmethod
    def validate_num_epochs(cls, v: float) -> float:
        _check_finite(v, "num_epochs")
        if v <= 0.0:
            raise ValueError("Epoch count must be strictly positive (> 0.0)")
        if v > 1000.0:
            raise ValueError("Epoch count cannot exceed 1,000 epochs")
        return v

    @field_validator("batch_size", "per_device_train_batch_size")
    @classmethod
    def validate_batch_size(cls, v: Optional[int], info: ValidationInfo) -> Optional[int]:
        if v is None:
            return None
        fname = info.field_name or "batch_size"
        _check_finite(v, fname)
        if v <= 0:
            raise ValueError(f"{fname} must be greater than 0")
        if v > 512:
            raise ValueError(f"{fname} cannot exceed 512")
        return v

    @field_validator("gradient_accumulation_steps")
    @classmethod
    def validate_gradient_accumulation(cls, v: int) -> int:
        _check_finite(v, "gradient_accumulation_steps")
        if v <= 0:
            raise ValueError("Gradient accumulation steps must be greater than 0")
        if v > 512:
            raise ValueError("Gradient accumulation steps cannot exceed 512")
        return v

    @field_validator("context_length", "max_seq_length")
    @classmethod
    def validate_context_length(cls, v: Optional[int], info: ValidationInfo) -> Optional[int]:
        if v is None:
            return None
        fname = info.field_name or "context_length"
        _check_finite(v, fname)
        if v < 128:
            raise ValueError(f"{fname} must be at least 128 tokens")
        if v > 131072:
            raise ValueError(f"{fname} cannot exceed 131,072 tokens")
        return v

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        _check_finite(v, "learning_rate")
        if v <= 0.0:
            raise ValueError("Learning rate must be strictly positive (> 0.0)")
        if v > 1.0:
            raise ValueError("Learning rate must be <= 1.0")
        return v

    @field_validator("weight_decay")
    @classmethod
    def validate_weight_decay(cls, v: float) -> float:
        _check_finite(v, "weight_decay")
        if v < 0.0:
            raise ValueError("Weight decay must be non-negative (>= 0.0)")
        if v > 1.0:
            raise ValueError("Weight decay cannot exceed 1.0")
        return v

    @field_validator("warmup_ratio")
    @classmethod
    def validate_warmup_ratio(cls, v: float) -> float:
        _check_finite(v, "warmup_ratio")
        if v < 0.0:
            raise ValueError("Warmup ratio must be non-negative (>= 0.0)")
        if v > 1.0:
            raise ValueError("Warmup ratio cannot exceed 1.0 (100% of training)")
        return v

    @field_validator("optimizer")
    @classmethod
    def validate_optimizer(cls, v: str) -> str:
        cleaned = v.strip().lower() if isinstance(v, str) else ""
        if cleaned not in ALLOWED_OPTIMIZERS:
            opts = ", ".join(sorted(ALLOWED_OPTIMIZERS))
            raise ValueError(f"Unsupported optimizer '{v}'. Allowed optimizers: {opts}")
        return cleaned

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, v: int) -> int:
        _check_finite(v, "seed")
        if v < 0:
            raise ValueError("Seed must be non-negative (>= 0)")
        if v > 2147483647:
            raise ValueError("Seed exceeds 32-bit signed integer range (2,147,483,647)")
        return v

    @field_validator("lora_rank")
    @classmethod
    def validate_lora_rank(cls, v: int) -> int:
        _check_finite(v, "lora_rank")
        if v < 1:
            raise ValueError("LoRA rank must be at least 1")
        if v > 512:
            raise ValueError("LoRA rank cannot exceed 512")
        return v

    @field_validator("lora_alpha")
    @classmethod
    def validate_lora_alpha(cls, v: int) -> int:
        _check_finite(v, "lora_alpha")
        if v < 1:
            raise ValueError("LoRA alpha must be at least 1")
        if v > 1024:
            raise ValueError("LoRA alpha cannot exceed 1,024")
        return v

    @field_validator("lora_dropout")
    @classmethod
    def validate_lora_dropout(cls, v: float) -> float:
        _check_finite(v, "lora_dropout")
        if v < 0.0:
            raise ValueError("LoRA dropout must be non-negative (>= 0.0)")
        if v > 1.0:
            raise ValueError("LoRA dropout cannot exceed 1.0")
        return v

    @field_validator("lora_target_modules")
    @classmethod
    def validate_target_modules(cls, v: List[str]) -> List[str]:
        if not isinstance(v, list):
            raise ValueError("Target modules must be a list of module name strings")
        cleaned_list: List[str] = []
        for mod in v:
            if not isinstance(mod, str) or not mod.strip():
                raise ValueError("Target module identifiers must be non-empty strings")
            cleaned_list.append(mod.strip())
        return cleaned_list

    @field_validator("mixed_precision")
    @classmethod
    def validate_mixed_precision(cls, v: str) -> str:
        cleaned = v.strip().lower() if isinstance(v, str) else ""
        if cleaned not in ALLOWED_PRECISIONS:
            prec = ", ".join(sorted(ALLOWED_PRECISIONS))
            raise ValueError(f"Unsupported mixed precision '{v}'. Allowed preferences: {prec}")
        return cleaned

    @field_validator("quantization_type")
    @classmethod
    def validate_quantization_type(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        cleaned = v.strip().lower()
        if cleaned not in ALLOWED_QUANTIZATIONS:
            quant = ", ".join(str(q) for q in sorted(ALLOWED_QUANTIZATIONS, key=lambda x: str(x)))
            raise ValueError(f"Unsupported quantization preference '{v}'. Allowed preferences: {quant}")
        return cleaned

    @model_validator(mode="after")
    def validate_cross_field_consistency(self) -> "TrainingConfigSchema":
        # Keep aliases synchronized
        if self.per_device_train_batch_size is None:
            self.per_device_train_batch_size = self.batch_size
        if self.max_seq_length is None:
            self.max_seq_length = self.context_length
        if self.training_duration_mode is None:
            self.training_duration_mode = self.training_length_type

        # LoRA/QLoRA module check
        if self.training_method in (TrainingMethod.LORA, TrainingMethod.QLORA):
            if not self.lora_target_modules or len(self.lora_target_modules) == 0:
                raise ValueError("At least one target module must be specified for LoRA or QLoRA fine-tuning")

        # QLoRA quantization preference check
        if self.training_method == TrainingMethod.QLORA:
            if not self.quantization_type or self.quantization_type == "none":
                raise ValueError("QLoRA requires an explicit quantization preference (e.g. 'nf4', 'fp4', or 'int8')")
            if self.load_in_4bit and self.load_in_8bit:
                raise ValueError("load_in_4bit and load_in_8bit cannot both be enabled simultaneously")
            if self.bnb_4bit_quant_type.lower() not in ("nf4", "fp4"):
                raise ValueError(f"Unsupported 4-bit quantization type '{self.bnb_4bit_quant_type}'. Must be 'nf4' or 'fp4'.")
            if self.bnb_4bit_compute_dtype.lower() not in ("fp16", "bf16", "float16", "float32", "bfloat16"):
                raise ValueError(f"Unsupported compute dtype '{self.bnb_4bit_compute_dtype}'.")

        return self

    def get_quantization_config(self) -> Any:
        """Returns typed QuantizationConfigSchema instance from configuration fields."""
        from backend.app.schemas.quantization import QuantizationConfigSchema
        return QuantizationConfigSchema(
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
        )

    def to_quantization_dict(self) -> Dict[str, Any]:
        """Serializes quantization configuration for artifacts and status reporting."""
        return {
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
        }


class TrainingConfigValidationResult(BaseModel):
    """Result of configuration validation."""
    is_valid: bool = Field(description="Whether configuration passes all schema rules")
    errors: List[str] = Field(default_factory=list, description="Validation failure messages")
    field_errors: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Field-specific validation failure messages"
    )
    warnings: List[str] = Field(default_factory=list, description="Non-blocking recommendations")
    is_ready_for_training: bool = Field(
        default=False,
        description="Whether training can be started (always False in Phase 3)"
    )
    config: Optional[TrainingConfigSchema] = Field(
        default=None,
        description="Validated configuration instance if valid"
    )
    effective_batch_size_per_device: Optional[int] = Field(
        default=None,
        description="Effective batch size per device (batch_size * gradient_accumulation_steps)"
    )
    total_effective_batch_size: Optional[int] = Field(
        default=None,
        description="Total effective batch size across all detected devices"
    )
    detected_device_count: Optional[int] = Field(
        default=None,
        description="Detected GPU device count, or None if unknown/CPU"
    )


class TrainingRunStatus(str, Enum):
    """Lifecycle statuses for a training run."""
    QUEUED = "queued"
    VALIDATING = "validating"
    LOADING_MODEL = "loading_model"
    PREPARING_DATASET = "preparing_dataset"
    TRAINING = "training"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingMetricPoint(BaseModel):
    """Telemetry data point recorded during training."""
    step: int = Field(description="Training step number")
    loss: float = Field(description="Calculated training loss")
    learning_rate: Optional[float] = Field(default=None, description="Current learning rate")
    epoch: Optional[float] = Field(default=None, description="Current epoch progress")
    timestamp: str = Field(description="ISO timestamp of metric recording")


class TrainingRunCreateRequest(BaseModel):
    """Request payload to initiate a training run."""
    config: TrainingConfigSchema = Field(description="Full training configuration")
    hf_token: Optional[str] = Field(default=None, description="Optional Hugging Face access token")


class TrainingRunResponse(BaseModel):
    """Comprehensive training run status and telemetry response."""
    run_id: str = Field(description="Unique training run identifier")
    status: TrainingRunStatus = Field(description="Current lifecycle status")
    config: TrainingConfigSchema = Field(description="Training configuration used for the run")
    created_at: str = Field(description="ISO timestamp of run creation")
    started_at: Optional[str] = Field(default=None, description="ISO timestamp of training start")
    completed_at: Optional[str] = Field(default=None, description="ISO timestamp of completion or termination")
    current_step: int = Field(default=0, description="Current step executed")
    max_steps: int = Field(default=0, description="Target training steps")
    current_epoch: Optional[float] = Field(default=0.0, description="Current epoch")
    num_epochs: Optional[float] = Field(default=0.0, description="Target epochs")
    current_loss: Optional[float] = Field(default=None, description="Latest recorded loss")
    loss_history: List[TrainingMetricPoint] = Field(default_factory=list, description="Historical loss trajectory")
    elapsed_time_seconds: float = Field(default=0.0, description="Elapsed runtime in seconds")
    error_message: Optional[str] = Field(default=None, description="Failure details if status is failed")
    device: str = Field(default="cpu", description="Execution device (e.g. cuda:0 or cpu)")
    trainable_params: Optional[int] = Field(default=None, description="Number of trainable adapter parameters")
    total_params: Optional[int] = Field(default=None, description="Total model parameter count")
    trainable_percent: Optional[float] = Field(default=None, description="Percentage of parameters that are trainable")
    cancellable: bool = Field(default=False, description="Whether the run can be cancelled in its current state")
    quantization_info: Optional[Dict[str, Any]] = Field(default=None, description="Quantization metadata if QLoRA is used")


class TrainingArtifactMetadata(BaseModel):
    """Validated artifact metadata after successful training."""
    run_id: str = Field(description="Run identifier associated with artifacts")
    artifact_directory: str = Field(description="Safe relative path to artifact storage")
    files: List[str] = Field(default_factory=list, description="List of generated artifact filenames")
    adapter_config: Dict[str, Any] = Field(default_factory=dict, description="LoRA adapter configuration dictionary")
    quantization_config: Optional[Dict[str, Any]] = Field(default=None, description="Quantization metadata if QLoRA was used")
    trainable_params: int = Field(description="Trainable parameter count")
    total_params: int = Field(description="Total parameter count")
    trainable_percent: float = Field(description="Trainable percentage")
    is_valid: bool = Field(default=True, description="Whether artifacts were successfully verified on disk")
    verification_status: str = Field(default="verified", description="Artifact verification status")
    created_at: str = Field(description="ISO creation timestamp")

