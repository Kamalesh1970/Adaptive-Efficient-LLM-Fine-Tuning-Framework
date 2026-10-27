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
