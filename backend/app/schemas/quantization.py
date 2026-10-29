"""
Quantization Configuration and Validation Schemas for QLoRA (Phase 5).
Defines typed models for 4-bit and 8-bit loading, NF4/FP4 quantization types,
double quantization, and compute dtypes via bitsandbytes.
"""

from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class QuantizationType(str, Enum):
    """Supported 4-bit quantization data formats."""
    NF4 = "nf4"
    FP4 = "fp4"


class ComputeDtype(str, Enum):
    """Supported compute precision dtypes for dequantized 4-bit weights."""
    FP16 = "fp16"
    BF16 = "bf16"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


ALLOWED_QUANT_TYPES = {"nf4", "fp4"}
ALLOWED_COMPUTE_DTYPES = {"fp16", "bf16", "float32", "float16", "bfloat16"}


class QuantizationConfigSchema(BaseModel):
    """
    Typed configuration schema for bitsandbytes quantization (QLoRA).
    """
    load_in_4bit: bool = Field(
        default=True,
        description="Whether to load base model weights in 4-bit precision"
    )
    load_in_8bit: bool = Field(
        default=False,
        description="Whether to load base model weights in 8-bit precision"
    )
    bnb_4bit_quant_type: str = Field(
        default="nf4",
        description="4-bit quantization data type: 'nf4' (NormalFloat4) or 'fp4' (Float4)"
    )
    bnb_4bit_use_double_quant: bool = Field(
        default=True,
        description="Whether to apply nested quantization to compress quantization constants"
    )
    bnb_4bit_compute_dtype: str = Field(
        default="fp16",
        description="Computation dtype for 4-bit dequantized forward pass: 'fp16', 'bf16', or 'float32'"
    )

    @field_validator("bnb_4bit_quant_type")
    @classmethod
    def validate_quant_type(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("bnb_4bit_quant_type must be a string")
        lower_v = v.lower().strip()
        if lower_v not in ALLOWED_QUANT_TYPES:
            raise ValueError(
                f"Unsupported 4-bit quantization type '{v}'. Supported types are: {', '.join(sorted(ALLOWED_QUANT_TYPES))}"
            )
        return lower_v

    @field_validator("bnb_4bit_compute_dtype")
    @classmethod
    def validate_compute_dtype(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("bnb_4bit_compute_dtype must be a string")
        lower_v = v.lower().strip()
        if lower_v not in ALLOWED_COMPUTE_DTYPES:
            raise ValueError(
                f"Unsupported compute dtype '{v}'. Supported compute dtypes are: {', '.join(sorted(ALLOWED_COMPUTE_DTYPES))}"
            )
        return lower_v

    @model_validator(mode="after")
    def validate_mutual_exclusivity(self) -> "QuantizationConfigSchema":
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("load_in_4bit and load_in_8bit are mutually exclusive. Cannot enable both simultaneously.")
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Returns safe dictionary representation for metadata serialization."""
        return {
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
        }


# ==============================================================================
# Phase 10: Advanced Quantization Schemas
# ==============================================================================

class AdvancedQuantMethod(str, Enum):
    """Supported quantization methods across training, post-training, and export."""
    BNB_4BIT = "bnb_4bit"
    BNB_8BIT = "bnb_8bit"
    GPTQ = "gptq"
    AWQ = "awq"
    GGUF = "gguf"


class QuantizationTarget(str, Enum):
    """Scope of quantization application."""
    TRAINING = "training"
    POST_TRAINING = "post_training"
    EXPORT = "export"
    SERVING = "serving"


class PostTrainingQuantizationConfig(BaseModel):
    """
    Configuration for post-training quantization jobs (GPTQ, AWQ, GGUF, BnB).
    """
    method: AdvancedQuantMethod = Field(
        description="Target quantization framework/method"
    )
    source_reference: str = Field(
        description="Source model identifier or trained run artifact ID"
    )
    bit_width: int = Field(
        default=4,
        description="Quantized bit width: 4 or 8"
    )
    quant_type: Optional[str] = Field(
        default="nf4",
        description="Specific quantization type (e.g. 'nf4', 'fp4', 'int8', 'q4_k_m', 'q4_0', 'q5_k_m')"
    )
    group_size: Optional[int] = Field(
        default=128,
        description="Group size for block-wise quantization (e.g. 32, 64, 128)"
    )
    compute_dtype: Optional[str] = Field(
        default="fp16",
        description="Compute precision during quantization"
    )
    calibration_dataset_id: Optional[str] = Field(
        default=None,
        description="Registered dataset ID to use for calibration (required for GPTQ/AWQ)"
    )
    calibration_samples: Optional[int] = Field(
        default=128,
        ge=1,
        le=2048,
        description="Number of calibration samples"
    )
    double_quant: bool = Field(
        default=True,
        description="Whether to enable nested quantization if supported"
    )
    output_format: str = Field(
        default="safetensors",
        description="Output format ('safetensors', 'bin', 'gguf')"
    )

    @field_validator("bit_width")
    @classmethod
    def validate_bits(cls, v: int) -> int:
        if v not in (2, 3, 4, 8, 16):
            raise ValueError(f"Bit width {v} not supported. Must be 2, 3, 4, 8, or 16.")
        return v

    @model_validator(mode="after")
    def validate_method_requirements(self) -> "PostTrainingQuantizationConfig":
        if self.method in (AdvancedQuantMethod.GPTQ, AdvancedQuantMethod.AWQ):
            if not self.calibration_dataset_id:
                raise ValueError(f"Calibration dataset is required for {self.method.value.upper()} quantization.")
        return self


class QuantizationCapability(BaseModel):
    """Detailed capability report for a quantization method on the current system."""
    method: AdvancedQuantMethod = Field(description="Method identifier")
    name: str = Field(description="Human-readable name")
    is_available: bool = Field(description="Whether the method is ready to execute on current system")
    required_libraries: list[str] = Field(default_factory=list, description="Libraries needed")
    installed_libraries: list[str] = Field(default_factory=list, description="Installed libraries")
    missing_libraries: list[str] = Field(default_factory=list, description="Missing libraries")
    requires_cuda: bool = Field(description="Whether CUDA hardware is mandatory")
    cuda_available: bool = Field(description="Whether CUDA is currently active")
    supports_training: bool = Field(description="Can be used during fine-tuning")
    supports_post_training: bool = Field(description="Can be used on existing artifacts")
    supports_inference: bool = Field(description="Supported for direct inference")
    supports_export: bool = Field(description="Supported as an export format")
    reason_if_unavailable: Optional[str] = Field(default=None, description="Explanation if disabled")


class QuantizationJobStatus(str, Enum):
    QUEUED = "queued"
    VALIDATING = "validating"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QuantizationJobRecord(BaseModel):
    """Persistent record of an advanced quantization operation."""
    job_id: str
    method: AdvancedQuantMethod
    source_reference: str
    config: PostTrainingQuantizationConfig
    status: QuantizationJobStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    elapsed_seconds: float = 0.0
    output_artifact_path: Optional[str] = None
    original_size_bytes: Optional[int] = None
    quantized_size_bytes: Optional[int] = None
    size_reduction_ratio: Optional[float] = None
    sha256: Optional[str] = None
    logs: list[str] = Field(default_factory=list)
    error_message: Optional[str] = None

