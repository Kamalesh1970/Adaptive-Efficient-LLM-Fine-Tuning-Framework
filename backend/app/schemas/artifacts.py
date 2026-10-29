"""
Pydantic Schemas for Phase 8: Training Artifacts, Adapter Export, and Integrity Validation.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ArtifactType(str, Enum):
    """Classification of training artifacts."""
    LORA_ADAPTER = "lora_adapter"
    QLORA_ADAPTER = "qlora_adapter"
    FULL_MODEL = "full_model"
    CHECKPOINT = "checkpoint"
    TOKENIZER = "tokenizer"
    CONFIG_SNAPSHOT = "config_snapshot"


class ValidationStatus(str, Enum):
    """Integrity and validity status of an artifact."""
    UNVALIDATED = "unvalidated"
    VALID = "valid"
    INVALID = "invalid"
    CORRUPTED = "corrupted"


class ExportStatus(str, Enum):
    """Packaging and export status of an adapter."""
    NOT_EXPORTED = "not_exported"
    EXPORTING = "exporting"
    EXPORTED = "exported"
    EXPORT_FAILED = "export_failed"


class ArtifactFileEntry(BaseModel):
    """Metadata for an individual file within an artifact."""
    filename: str = Field(description="Name of file within the artifact directory")
    size_bytes: int = Field(default=0, description="Actual file size in bytes from the filesystem")
    sha256: Optional[str] = Field(default=None, description="SHA-256 integrity hash of the file")
    is_required: bool = Field(default=False, description="Whether this file is required for the artifact type")
    exists: bool = Field(default=True, description="Whether the file actually exists and is readable")


class ArtifactMetadata(BaseModel):
    """
    Complete safe metadata representing a training artifact.
    Does not expose arbitrary host filesystem paths or credentials.
    """
    artifact_id: str = Field(description="Unique artifact identifier (e.g. art_run123_adapter)")
    run_id: str = Field(description="Associated training run identifier")
    artifact_type: ArtifactType = Field(description="Type classification of the artifact")
    model_identifier: str = Field(description="Base model identifier used during training")
    fine_tuning_method: str = Field(description="Fine-tuning method (lora, qlora, full)")
    created_at: str = Field(description="UTC timestamp of artifact creation")
    file_reference: str = Field(description="Safe relative directory reference within the workspace")
    size_bytes: int = Field(default=0, description="Total size in bytes of existing artifact files")
    adapter_config: Optional[Dict[str, Any]] = Field(default=None, description="Adapter configuration metadata if applicable")
    quantization_config: Optional[Dict[str, Any]] = Field(default=None, description="Quantization metadata if QLoRA")
    checkpoint_step: Optional[int] = Field(default=None, description="Checkpoint training step if intermediate checkpoint")
    validation_status: ValidationStatus = Field(default=ValidationStatus.UNVALIDATED, description="Integrity validation status")
    export_status: ExportStatus = Field(default=ExportStatus.NOT_EXPORTED, description="Export packaging status")
    exported_package_path: Optional[str] = Field(default=None, description="Safe relative path to exported archive")
    files: List[ArtifactFileEntry] = Field(default_factory=list, description="Files discovered on disk")
    validation_errors: List[str] = Field(default_factory=list, description="List of integrity or schema errors")
    validation_warnings: List[str] = Field(default_factory=list, description="List of validation warnings")
    base_model_identifier: Optional[str] = Field(default=None, description="Reference to base model")
    is_loadable: Optional[bool] = Field(default=None, description="Whether artifact was successfully loaded/tested")
    load_error: Optional[str] = Field(default=None, description="Error encountered during load verification")


class RunArtifactsListResponse(BaseModel):
    """
    Response listing artifacts that actually exist for a run.
    Maintains backward compatibility with earlier Phase 4 TrainingArtifactMetadata fields.
    """
    run_id: str = Field(description="Associated training run ID")
    artifacts: List[ArtifactMetadata] = Field(default_factory=list, description="Artifacts discovered for the run")
    total: int = Field(default=0, description="Total number of artifacts for this run")

    # Backward compatibility with Phase 4/7 TrainingArtifactMetadata
    artifact_directory: Optional[str] = Field(default=None, description="Safe relative directory")
    files: Optional[List[str]] = Field(default=None, description="List of filenames in primary artifact directory")
    adapter_config: Optional[Dict[str, Any]] = Field(default=None)
    quantization_config: Optional[Dict[str, Any]] = Field(default=None)
    trainable_params: Optional[int] = Field(default=None)
    total_params: Optional[int] = Field(default=None)
    trainable_percent: Optional[float] = Field(default=None)
    is_valid: Optional[bool] = Field(default=None)
    verification_status: Optional[str] = Field(default=None)


class ArtifactValidationRequest(BaseModel):
    """Request payload for validating an artifact."""
    check_loadable: bool = Field(default=False, description="Whether to perform an active load check on the model")


class ArtifactValidationResponse(BaseModel):
    """Detailed response for artifact validation."""
    artifact_id: str = Field(description="Target artifact ID")
    is_valid: bool = Field(description="Whether the artifact passed all required validation checks")
    validation_status: ValidationStatus = Field(description="Updated validation status")
    errors: List[str] = Field(default_factory=list, description="Validation failure reasons")
    warnings: List[str] = Field(default_factory=list, description="Non-blocking validation warnings")
    checksums: Dict[str, str] = Field(default_factory=dict, description="SHA-256 checksums of verified files")
    total_size_bytes: int = Field(default=0, description="Total size in bytes of verified files")
    is_loadable: Optional[bool] = Field(default=None, description="Result of load test if requested")
    load_error: Optional[str] = Field(default=None, description="Error message if model loading failed")
    checked_at: str = Field(description="UTC timestamp of the validation check")


class ArtifactExportResponse(BaseModel):
    """Response describing the exported package."""
    artifact_id: str = Field(description="Target artifact ID")
    export_status: ExportStatus = Field(description="Packaging status")
    package_filename: str = Field(description="Name of the exported zip package")
    package_size_bytes: int = Field(description="Size in bytes of the exported package")
    sha256: str = Field(description="SHA-256 checksum of the package")
    created_at: str = Field(description="UTC creation timestamp")
    download_url: str = Field(description="Safe API endpoint URL to download the package")
    message: str = Field(description="Human-readable export summary")
