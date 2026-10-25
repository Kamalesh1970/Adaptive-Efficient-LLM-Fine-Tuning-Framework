"""
Pydantic Schemas for Dataset Upload, Schema Detection, Validation, and Splits.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class DatasetFormat(str, Enum):
    """Supported file formats for dataset uploads."""
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"


class DatasetSchemaType(str, Enum):
    """Supported fine-tuning schema formats."""
    INSTRUCTION = "instruction"
    TEXT = "text"
    CONVERSATIONAL = "conversational"


class SampleValidationError(BaseModel):
    """Structured error entry for an invalid sample or malformed line."""
    sample_index: int = Field(description="1-indexed line number or sample index")
    message: str = Field(description="Description of the validation failure")
    sample_preview: Optional[str] = Field(default=None, description="Optional safe snippet of the malformed record")


class DatasetValidationReport(BaseModel):
    """Comprehensive validation report for an uploaded dataset."""
    dataset_id: str = Field(description="Unique server-assigned dataset identifier")
    original_filename: str = Field(description="Sanitized user-facing filename (no path)")
    file_format: DatasetFormat = Field(description="Detected and validated file format")
    file_size_bytes: int = Field(description="Total file size in bytes")
    detected_schema: Optional[DatasetSchemaType] = Field(
        default=None,
        description="Identified sample schema ('instruction', 'text', 'conversational')"
    )
    total_samples: int = Field(description="Total number of parsed records/lines")
    valid_sample_count: int = Field(description="Count of samples passing all schema rules")
    invalid_sample_count: int = Field(description="Count of samples with schema or structure errors")
    empty_sample_count: int = Field(description="Count of blank or empty records")
    duplicate_count: Optional[int] = Field(
        default=None,
        description="Count of duplicate samples identified by SHA-256 content hash"
    )
    is_valid: bool = Field(description="Whether the dataset contains valid data and is usable for fine-tuning")
    warnings: List[str] = Field(default_factory=list, description="Non-blocking cautionary warnings")
    errors: List[SampleValidationError] = Field(
        default_factory=list,
        description="Actionable errors preventing training"
    )
    preview_samples: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="First few valid normalized samples for UI inspection"
    )


class DatasetSummary(BaseModel):
    """High-level summary of an available dataset."""
    id: str = Field(description="Dataset identifier")
    filename: str = Field(description="Sanitized filename")
    format: DatasetFormat = Field(description="File format")
    size_bytes: int = Field(description="File size in bytes")
    uploaded_at: str = Field(description="ISO timestamp of upload")
    total_samples: int = Field(description="Total sample count")
    valid_sample_count: int = Field(description="Valid sample count")
    schema_type: Optional[DatasetSchemaType] = Field(default=None, description="Schema type")
    is_valid: bool = Field(description="Whether dataset passed validation")


class DatasetListResponse(BaseModel):
    """Response containing list of uploaded datasets."""
    datasets: List[DatasetSummary] = Field(description="List of available datasets")
    total: int = Field(description="Total count of datasets")


class DatasetSplitRequest(BaseModel):
    """Request payload for configuring dataset train/eval splits."""
    train_ratio: float = Field(default=0.9, ge=0.0, le=1.0, description="Proportion allocated to training")
    eval_ratio: float = Field(default=0.1, ge=0.0, le=1.0, description="Proportion allocated to evaluation")
    seed: int = Field(default=42, description="Random seed for deterministic split generation")
    shuffle: bool = Field(default=True, description="Whether to shuffle samples before splitting")


class DatasetSplitResult(BaseModel):
    """Deterministic train/evaluation split result."""
    dataset_id: str = Field(description="Target dataset identifier")
    total_samples: int = Field(description="Total valid samples available for splitting")
    train_count: int = Field(description="Number of samples allocated to training")
    eval_count: int = Field(description="Number of samples allocated to evaluation")
    train_indices: List[int] = Field(description="Deterministic 0-indexed indices for training")
    eval_indices: List[int] = Field(description="Deterministic 0-indexed indices for evaluation")
    has_overlap: bool = Field(default=False, description="Verification that splits are disjoint (leak-free)")
    warnings: List[str] = Field(default_factory=list, description="Warnings (e.g. small sample size)")


class DatasetValidationRequest(BaseModel):
    """Request payload to validate a dataset configuration or split selection."""
    train_split: str = Field(default="train", description="Target training split name")
    eval_split: str = Field(default="none", description="Target evaluation split name ('none', 'test', 'validation')")
    split_ratio: float = Field(default=0.9, ge=0.0, le=1.0, description="Train split ratio if splitting deterministically")
