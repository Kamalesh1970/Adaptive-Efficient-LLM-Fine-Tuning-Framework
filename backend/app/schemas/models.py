"""
Pydantic Schemas for Foundation Model Catalog, Metadata, and Compatibility Validation.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from backend.app.schemas.training import TrainingMethod


class ModelVerificationStatus(str, Enum):
    """Registration and verification lifecycle of a catalog model entry."""
    CATALOG_REGISTERED = "catalog_registered"
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    EXPERIMENTAL = "experimental"


class CompatibilityStatus(str, Enum):
    """Result status of fine-tuning compatibility evaluation."""
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"
    REQUIRES_CREDENTIALS = "requires_credentials"


class ModelMetadata(BaseModel):
    """
    Structured foundation model metadata entry.
    Contains only verified catalog attributes. Unknown or unverified attributes
    are explicitly typed as None to prevent fabricating model specifications.
    """
    id: str = Field(description="Unique model identifier, e.g. 'meta-llama/Meta-Llama-3-8B'")
    name: str = Field(description="Human-readable model name")
    family: Optional[str] = Field(default=None, description="Architecture or model family (e.g. 'llama', 'qwen2')")
    source: str = Field(default="huggingface", description="Origin source or repository registry")
    task_type: str = Field(default="causal_lm", description="Supported model task type")
    parameter_count: Optional[str] = Field(default=None, description="Parameter scale (e.g. '8.0B', '0.5B')")
    context_length: Optional[int] = Field(default=None, description="Verified maximum token context length")
    is_gated: bool = Field(default=False, description="Whether access requires approved Hugging Face credentials")
    verification_status: ModelVerificationStatus = Field(
        default=ModelVerificationStatus.CATALOG_REGISTERED,
        description="Catalog verification level"
    )
    supported_methods: List[TrainingMethod] = Field(
        default_factory=lambda: [TrainingMethod.LORA, TrainingMethod.QLORA],
        description="Fine-tuning methods with verified architectural compatibility"
    )
    compatibility_notes: Optional[str] = Field(default=None, description="Detailed compatibility guidance")
    is_downloaded: bool = Field(
        default=False,
        description="Whether weights exist in local storage (strictly False in Phase 1)"
    )
    is_tested: bool = Field(
        default=False,
        description="Whether training execution was completed (strictly False in Phase 1)"
    )


class ModelCatalogResponse(BaseModel):
    """Response schema containing the registered model catalog list."""
    models: List[ModelMetadata] = Field(description="List of available model metadata entries")
    total: int = Field(description="Total number of models in catalog")


class ModelValidationRequest(BaseModel):
    """Request payload for validating model selection and method compatibility."""
    model_id: str = Field(description="Model identifier to validate")
    training_method: str = Field(description="Requested fine-tuning method ('lora', 'qlora', 'full_finetuning')")
    hf_token: Optional[str] = Field(default=None, description="Optional access token for gated models (masked/unverified)")
    context_length: Optional[int] = Field(default=None, description="Optional requested context length in tokens")

    @field_validator("model_id")
    @classmethod
    def validate_model_id_not_blank(cls, v: str) -> str:
        cleaned = v.strip() if v else ""
        if not cleaned:
            raise ValueError("Model identifier cannot be empty or blank")
        return cleaned


class ModelValidationResponse(BaseModel):
    """Structured response returned when evaluating model compatibility."""
    is_valid: bool = Field(description="Whether the model selection passes all validation and compatibility checks")
    model_id: Optional[str] = Field(default=None, description="Target model identifier")
    training_method: Optional[str] = Field(default=None, description="Requested fine-tuning method")
    compatibility_status: CompatibilityStatus = Field(description="Compatibility classification")
    errors: List[str] = Field(default_factory=list, description="Validation failure messages")
    warnings: List[str] = Field(default_factory=list, description="Non-blocking recommendations or cautionary notices")
    model: Optional[ModelMetadata] = Field(default=None, description="Model metadata if recognized in catalog")
    token_status: str = Field(
        default="not_provided",
        description="Token configuration state ('not_provided', 'configured_unverified', 'present_in_env')"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Sanitized validation details (never contains secrets, tokens, or credentials)"
    )
