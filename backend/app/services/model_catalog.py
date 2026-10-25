"""
Model Catalog Service.

Provides a modular, configurable catalog of causal language models,
stores metadata separated from execution loops, and performs strict compatibility validation.
"""

import os
from typing import Dict, List, Optional, Any
from backend.app.schemas.training import TrainingMethod
from backend.app.schemas.models import (
    ModelMetadata,
    ModelVerificationStatus,
    CompatibilityStatus,
    ModelValidationResponse,
)

# Representative initial catalog of causal language models.
# Separated from model downloading or weight-loading logic.
# All fields accurately distinguish verified facts from unverified/unknown attributes.
_INITIAL_MODELS: List[ModelMetadata] = [
    ModelMetadata(
        id="meta-llama/Meta-Llama-3-8B",
        name="Meta Llama 3 8B",
        family="llama",
        source="huggingface",
        task_type="causal_lm",
        parameter_count="8.0B",
        context_length=8192,
        is_gated=True,
        verification_status=ModelVerificationStatus.CATALOG_REGISTERED,
        supported_methods=[
            TrainingMethod.LORA,
            TrainingMethod.QLORA,
            TrainingMethod.FULL_FINETUNING,
        ],
        compatibility_notes="Gated model. Requires approved Hugging Face license and access credentials.",
        is_downloaded=False,
        is_tested=False,
    ),
    ModelMetadata(
        id="meta-llama/Llama-2-7b-hf",
        name="Llama 2 7B",
        family="llama",
        source="huggingface",
        task_type="causal_lm",
        parameter_count="6.7B",
        context_length=4096,
        is_gated=True,
        verification_status=ModelVerificationStatus.CATALOG_REGISTERED,
        supported_methods=[
            TrainingMethod.LORA,
            TrainingMethod.QLORA,
            TrainingMethod.FULL_FINETUNING,
        ],
        compatibility_notes="Gated model. Requires approved Hugging Face license and access credentials.",
        is_downloaded=False,
        is_tested=False,
    ),
    ModelMetadata(
        id="mistralai/Mistral-7B-v0.3",
        name="Mistral 7B v0.3",
        family="mistral",
        source="huggingface",
        task_type="causal_lm",
        parameter_count="7.2B",
        context_length=32768,
        is_gated=False,
        verification_status=ModelVerificationStatus.CATALOG_REGISTERED,
        supported_methods=[
            TrainingMethod.LORA,
            TrainingMethod.QLORA,
            TrainingMethod.FULL_FINETUNING,
        ],
        compatibility_notes="Open-weight causal model supporting sliding window attention.",
        is_downloaded=False,
        is_tested=False,
    ),
    ModelMetadata(
        id="Qwen/Qwen2.5-7B",
        name="Qwen 2.5 7B",
        family="qwen2",
        source="huggingface",
        task_type="causal_lm",
        parameter_count="7.6B",
        context_length=32768,
        is_gated=False,
        verification_status=ModelVerificationStatus.CATALOG_REGISTERED,
        supported_methods=[
            TrainingMethod.LORA,
            TrainingMethod.QLORA,
            TrainingMethod.FULL_FINETUNING,
        ],
        compatibility_notes="High-performance open-weights architecture with RoPE positional embeddings.",
        is_downloaded=False,
        is_tested=False,
    ),
    ModelMetadata(
        id="microsoft/Phi-3-mini-4k-instruct",
        name="Phi-3 Mini 4K Instruct",
        family="phi3",
        source="huggingface",
        task_type="causal_lm",
        parameter_count="3.8B",
        context_length=4096,
        is_gated=False,
        verification_status=ModelVerificationStatus.CATALOG_REGISTERED,
        supported_methods=[
            TrainingMethod.LORA,
            TrainingMethod.QLORA,
            TrainingMethod.FULL_FINETUNING,
        ],
        compatibility_notes="Compact 3.8B foundation model suitable for resource-constrained fine-tuning.",
        is_downloaded=False,
        is_tested=False,
    ),
    # Candidate small models for limited-VRAM / edge environments
    ModelMetadata(
        id="Qwen/Qwen2.5-0.5B",
        name="Qwen 2.5 0.5B",
        family="qwen2",
        source="huggingface",
        task_type="causal_lm",
        parameter_count="0.5B",
        context_length=32768,
        is_gated=False,
        verification_status=ModelVerificationStatus.CATALOG_REGISTERED,
        supported_methods=[
            TrainingMethod.LORA,
            TrainingMethod.QLORA,
            TrainingMethod.FULL_FINETUNING,
        ],
        compatibility_notes="Ultra-compact candidate for limited-VRAM (< 6 GB) environments and CPU testing.",
        is_downloaded=False,
        is_tested=False,
    ),
    ModelMetadata(
        id="HuggingFaceTB/SmolLM-135M",
        name="SmolLM 135M",
        family="smollm",
        source="huggingface",
        task_type="causal_lm",
        parameter_count="135M",
        context_length=2048,
        is_gated=False,
        verification_status=ModelVerificationStatus.CATALOG_REGISTERED,
        supported_methods=[
            TrainingMethod.LORA,
            TrainingMethod.QLORA,
            TrainingMethod.FULL_FINETUNING,
        ],
        compatibility_notes="Ultra-lightweight causal LM ideal for fast local prototyping and low-memory setups.",
        is_downloaded=False,
        is_tested=False,
    ),
    # Architecture with restricted method compatibility (full fine-tuning not supported)
    ModelMetadata(
        id="google/gemma-2-2b",
        name="Gemma 2 2B",
        family="gemma2",
        source="huggingface",
        task_type="causal_lm",
        parameter_count="2.6B",
        context_length=8192,
        is_gated=True,
        verification_status=ModelVerificationStatus.CATALOG_REGISTERED,
        supported_methods=[
            TrainingMethod.LORA,
            TrainingMethod.QLORA,
        ],
        compatibility_notes="Gated model requiring HF credentials. Full fine-tuning is unsupported for this architecture configuration.",
        is_downloaded=False,
        is_tested=False,
    ),
    # Architecture with unknown / unverified metadata
    ModelMetadata(
        id="custom/unverified-experimental-lm",
        name="Custom Experimental LM",
        family=None,  # Explicitly unknown
        source="local",
        task_type="causal_lm",
        parameter_count=None,  # Explicitly unknown
        context_length=None,  # Explicitly unverified
        is_gated=False,
        verification_status=ModelVerificationStatus.UNVERIFIED,
        supported_methods=[
            TrainingMethod.LORA,
        ],
        compatibility_notes="Experimental custom architecture. Context length and full fine-tuning compatibility are unverified.",
        is_downloaded=False,
        is_tested=False,
    ),
]


class ModelCatalogService:
    """
    Modular, in-memory model catalog registry and compatibility validation service.
    Can be configured dynamically with additional model definitions.
    """

    def __init__(self, initial_models: Optional[List[ModelMetadata]] = None) -> None:
        self._registry: Dict[str, ModelMetadata] = {}
        models_to_load = initial_models if initial_models is not None else _INITIAL_MODELS
        for model in models_to_load:
            self.register_model(model)

    def register_model(self, model: ModelMetadata) -> None:
        """Registers or updates a model metadata entry in the catalog."""
        self._registry[model.id] = model

    def get_all_models(self) -> List[ModelMetadata]:
        """Returns all registered model entries."""
        return list(self._registry.values())

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Looks up a model entry by identifier."""
        return self._registry.get(model_id)

    def has_model(self, model_id: str) -> bool:
        """Checks if a model identifier is registered."""
        return model_id in self._registry

    def validate_selection(
        self,
        model_id: Optional[str],
        training_method: Optional[str],
        hf_token: Optional[str] = None,
        context_length: Optional[int] = None,
    ) -> ModelValidationResponse:
        """
        Validates requested model selection, fine-tuning method compatibility,
        and required access credentials.
        Strictly prevents token leakage in responses or details.
        """
        errors: List[str] = []
        warnings: List[str] = []
        details: Dict[str, Any] = {}

        # 1. Validate model identifier
        cleaned_model_id = model_id.strip() if model_id else ""
        if not cleaned_model_id:
            return ModelValidationResponse(
                is_valid=False,
                model_id=None,
                training_method=training_method,
                compatibility_status=CompatibilityStatus.INCOMPATIBLE,
                errors=["Model identifier cannot be empty or blank."],
                warnings=warnings,
                model=None,
                token_status="not_provided",
                details={},
            )

        model = self.get_model(cleaned_model_id)
        if not model:
            return ModelValidationResponse(
                is_valid=False,
                model_id=cleaned_model_id,
                training_method=training_method,
                compatibility_status=CompatibilityStatus.INCOMPATIBLE,
                errors=[f"Unknown model identifier '{cleaned_model_id}'. Model is not recognized in catalog."],
                warnings=warnings,
                model=None,
                token_status="not_provided",
                details={},
            )

        # 2. Validate training method
        valid_methods = {m.value for m in TrainingMethod}
        cleaned_method = training_method.strip() if training_method else ""
        if not cleaned_method or cleaned_method not in valid_methods:
            return ModelValidationResponse(
                is_valid=False,
                model_id=cleaned_model_id,
                training_method=cleaned_method,
                compatibility_status=CompatibilityStatus.INCOMPATIBLE,
                errors=[
                    f"Invalid training method '{cleaned_method}'. Supported methods: {', '.join(sorted(valid_methods))}."
                ],
                warnings=warnings,
                model=model,
                token_status="not_provided",
                details={},
            )

        # 3. Check method compatibility with model
        supported_method_values = {m.value for m in model.supported_methods}
        if cleaned_method not in supported_method_values:
            if model.verification_status == ModelVerificationStatus.UNVERIFIED:
                return ModelValidationResponse(
                    is_valid=False,
                    model_id=cleaned_model_id,
                    training_method=cleaned_method,
                    compatibility_status=CompatibilityStatus.UNKNOWN,
                    errors=[
                        f"Compatibility for method '{cleaned_method}' with model '{cleaned_model_id}' is unverified."
                    ],
                    warnings=[f"Model '{cleaned_model_id}' contains unverified catalog specifications."],
                    model=model,
                    token_status="not_provided",
                    details={"supported_methods": list(supported_method_values)},
                )
            else:
                return ModelValidationResponse(
                    is_valid=False,
                    model_id=cleaned_model_id,
                    training_method=cleaned_method,
                    compatibility_status=CompatibilityStatus.INCOMPATIBLE,
                    errors=[
                        f"Model '{cleaned_model_id}' does not support '{cleaned_method}'. "
                        f"Supported methods: {', '.join(sorted(supported_method_values))}."
                    ],
                    warnings=warnings,
                    model=model,
                    token_status="not_provided",
                    details={"supported_methods": list(supported_method_values)},
                )

        # 4. Check credential requirements for gated models
        # Determine token status without exposing raw token
        has_supplied_token = bool(hf_token and hf_token.strip())
        has_env_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))

        if has_supplied_token:
            token_status = "configured_unverified"
        elif has_env_token:
            token_status = "present_in_env"
        else:
            token_status = "not_provided"

        if model.is_gated:
            if token_status == "not_provided":
                return ModelValidationResponse(
                    is_valid=False,
                    model_id=cleaned_model_id,
                    training_method=cleaned_method,
                    compatibility_status=CompatibilityStatus.REQUIRES_CREDENTIALS,
                    errors=[
                        f"Model '{cleaned_model_id}' requires approved Hugging Face gated access credentials. "
                        "Please configure a Hugging Face access token."
                    ],
                    warnings=["Gated access requires an authenticated Hugging Face account."],
                    model=model,
                    token_status=token_status,
                    details={"is_gated": True},
                )
            else:
                warnings.append(
                    "Hugging Face token configured. Credential verification with Hugging Face Hub is labeled as unverified in Phase 1."
                )

        # 5. Check context length configuration if provided
        if context_length is not None:
            if context_length <= 0:
                errors.append("Context length must be greater than 0.")
            elif model.context_length is not None and context_length > model.context_length:
                warnings.append(
                    f"Requested context length ({context_length}) exceeds model verified maximum ({model.context_length})."
                )
            elif model.context_length is None:
                warnings.append(
                    f"Model '{cleaned_model_id}' has unverified maximum context length."
                )

        # 6. Unverified catalog entry warning
        if model.verification_status == ModelVerificationStatus.UNVERIFIED:
            warnings.append(
                f"Model '{cleaned_model_id}' is registered with unverified specifications. Validate configuration before training."
            )

        # 7. Safe details (never contains secrets or raw tokens)
        details = {
            "family": model.family,
            "parameter_count": model.parameter_count,
            "context_length": model.context_length,
            "is_gated": model.is_gated,
            "supported_methods": list(supported_method_values),
        }

        is_valid = len(errors) == 0
        compatibility_status = (
            CompatibilityStatus.COMPATIBLE
            if is_valid
            else CompatibilityStatus.INCOMPATIBLE
        )

        return ModelValidationResponse(
            is_valid=is_valid,
            model_id=cleaned_model_id,
            training_method=cleaned_method,
            compatibility_status=compatibility_status,
            errors=errors,
            warnings=warnings,
            model=model,
            token_status=token_status,
            details=details,
        )


# Singleton instance
_catalog_instance: Optional[ModelCatalogService] = None


def get_model_catalog_service() -> ModelCatalogService:
    """Returns the default singleton model catalog service instance."""
    global _catalog_instance
    if _catalog_instance is None:
        _catalog_instance = ModelCatalogService()
    return _catalog_instance
