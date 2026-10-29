"""
Model Export Schemas for Phase 10 (GGUF, Merged Weights).
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class GGUFQuantizationType(str, Enum):
    """Supported GGUF quantization formats."""
    Q4_0 = "Q4_0"
    Q4_K_M = "Q4_K_M"
    Q5_0 = "Q5_0"
    Q5_K_M = "Q5_K_M"
    Q8_0 = "Q8_0"
    F16 = "F16"
    F32 = "F32"


class GGUFExportConfig(BaseModel):
    """Configuration for GGUF model export."""
    source_artifact_id: str = Field(description="Run ID or artifact ID containing adapter weights")
    base_model_identifier: Optional[str] = Field(
        default=None,
        description="Base model to merge adapter into. If omitted, resolved from artifact metadata."
    )
    quantization_type: GGUFQuantizationType = Field(
        default=GGUFQuantizationType.Q4_K_M,
        description="Target GGUF quantization format"
    )
    output_filename: Optional[str] = Field(
        default=None,
        description="Optional custom filename (must end with .gguf and contain no path separators)"
    )

    @field_validator("output_filename")
    @classmethod
    def validate_filename(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
            if "/" in v or "\\" in v or ".." in v:
                raise ValueError("output_filename must not contain path separators or traversal sequences.")
            if not v.endswith(".gguf"):
                raise ValueError("output_filename must end with .gguf extension.")
        return v


class GGUFExportJobRecord(BaseModel):
    """Persistent state record for a GGUF export operation."""
    export_id: str
    source_artifact_id: str
    status: str
    config: GGUFExportConfig
    output_path: Optional[str] = None
    output_filepath: Optional[str] = None
    file_size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        if self.output_path and not self.output_filepath:
            self.output_filepath = self.output_path
        elif self.output_filepath and not self.output_path:
            self.output_path = self.output_filepath
