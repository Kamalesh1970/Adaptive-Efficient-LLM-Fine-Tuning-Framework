"""
Model Loading and Tokenizer Management Service (Phase 4).
Handles secure, validated loading of causal language models and tokenizers.
Guarantees trust_remote_code is disabled by default, prevents path traversal,
and supports device resolution between CUDA and CPU fallbacks.
"""

import os
from typing import Tuple, Any, Optional, Dict, Callable
from backend.app.core.config import settings
from backend.app.core.system import get_system_hardware_info
from backend.app.services.ml_dependencies import (
    is_full_ml_stack_available,
    verify_ml_dependencies_or_raise,
    MLDependencyMissingError,
)
from backend.app.services.model_catalog import get_model_catalog_service


class ModelLoadError(Exception):
    """Base exception for model loading failures."""
    pass


class ModelNotFoundError(ModelLoadError):
    """Raised when the requested model cannot be located locally or remotely."""
    pass


class GatedModelAccessError(ModelLoadError):
    """Raised when gated model authentication fails or token is missing."""
    pass


class ModelArchitectureUnsupportedError(ModelLoadError):
    """Raised when model is not a causal language model suitable for text generation."""
    pass


class UntrustedModelCodeError(ModelLoadError):
    """Raised when model requires executing custom remote Python code but trust_remote_code is disabled."""
    pass


# In-memory registry for synthetic test models to allow zero-download, offline unit testing
_test_model_registry: Dict[str, Tuple[Callable[[], Any], Callable[[], Any]]] = {}


def register_test_model(
    model_identifier: str,
    model_factory: Callable[[], Any],
    tokenizer_factory: Callable[[], Any],
) -> None:
    """Registers a synthetic model/tokenizer factory for unit testing without downloading weights."""
    _test_model_registry[model_identifier] = (model_factory, tokenizer_factory)


def clear_test_model_registry() -> None:
    """Clears registered synthetic test models."""
    _test_model_registry.clear()


def resolve_execution_device(requested: Optional[str] = None) -> str:
    """
    Resolves execution device. Returns 'cuda:0' or 'cuda' only if CUDA hardware is available.
    Falls back to 'cpu' gracefully for tiny models or when GPU is absent.
    """
    if requested and requested.lower() == "cpu":
        return "cpu"

    hw = get_system_hardware_info()
    if hw.get("cuda_available") and hw.get("gpu_count", 0) > 0:
        if requested and requested.startswith("cuda"):
            return requested
        return "cuda:0"
    return "cpu"


def validate_model_identifier(model_identifier: str) -> str:
    """Validates model identifier to prevent path traversal or unsafe execution."""
    if not isinstance(model_identifier, str):
        raise ModelNotFoundError("Model identifier must be a string.")

    cleaned = model_identifier.strip()
    if not cleaned:
        raise ModelNotFoundError("Model identifier cannot be empty.")

    # Guard against directory traversal attacks
    if ".." in cleaned:
        raise ModelNotFoundError("Invalid model identifier: path traversal tokens ('..') are not allowed.")

    return cleaned


class ModelLoaderService:
    """Service to load causal language models and tokenizers safely."""

    def __init__(self, allow_remote_code: bool = False) -> None:
        self.allow_remote_code = allow_remote_code or settings.allow_remote_code

    def load_model_and_tokenizer(
        self,
        model_identifier: str,
        hf_token: Optional[str] = None,
        device: Optional[str] = None,
        trust_remote_code: Optional[bool] = None,
        local_files_only: bool = False,
    ) -> Tuple[Any, Any, str]:
        """
        Loads a causal language model and tokenizer.
        Returns (model, tokenizer, device_used).
        Raises ModelLoadError subclasses on failure.
        """
        cleaned_id = validate_model_identifier(model_identifier)
        effective_trust = trust_remote_code if trust_remote_code is not None else self.allow_remote_code
        target_device = resolve_execution_device(device)

        # 1. Check synthetic test registry first (enables fast, offline unit testing)
        if cleaned_id in _test_model_registry:
            model_fac, tok_fac = _test_model_registry[cleaned_id]
            try:
                model = model_fac()
                tok = tok_fac()
                return model, tok, "cpu"
            except Exception as e:
                raise ModelLoadError(f"Failed to instantiate test model '{cleaned_id}': {str(e)}") from e

        # 2. Check for gated model credentials in catalog metadata
        catalog_svc = get_model_catalog_service()
        model_meta = catalog_svc.get_model(cleaned_id)
        token_to_use = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        if model_meta and model_meta.is_gated and not token_to_use:
            raise GatedModelAccessError(
                f"Model '{cleaned_id}' is gated. A valid Hugging Face access token is required."
            )

        # 3. Ensure ML dependencies are installed
        verify_ml_dependencies_or_raise()

        # 4. Import established ML libraries safely
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as e:
            raise MLDependencyMissingError(f"Failed to import PyTorch or Transformers: {str(e)}") from e

        # 5. Load Tokenizer
        tokenizer_kwargs: Dict[str, Any] = {
            "trust_remote_code": effective_trust,
            "local_files_only": local_files_only,
        }
        if token_to_use:
            tokenizer_kwargs["token"] = token_to_use

        try:
            tokenizer = AutoTokenizer.from_pretrained(cleaned_id, **tokenizer_kwargs)
            # Ensure pad_token is defined for batch padding in causal LM
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        except Exception as e:
            err_msg = str(e)
            if "requires `trust_remote_code=True`" in err_msg:
                raise UntrustedModelCodeError(
                    f"Model '{cleaned_id}' requires custom remote code execution, which is disabled for security."
                ) from e
            if "401" in err_msg or "403" in err_msg or "gated" in err_msg.lower():
                raise GatedModelAccessError(
                    f"Failed to authenticate access to model '{cleaned_id}'. Ensure a valid Hugging Face token is provided."
                ) from e
            if "not found" in err_msg.lower() or "repository not found" in err_msg.lower():
                raise ModelNotFoundError(f"Model or tokenizer '{cleaned_id}' was not found.") from e
            raise ModelLoadError(f"Failed to load tokenizer for '{cleaned_id}': {err_msg}") from e

        # 6. Load Causal Language Model
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": effective_trust,
            "local_files_only": local_files_only,
        }
        if token_to_use:
            model_kwargs["token"] = token_to_use

        # Determine precision & device placement
        if target_device.startswith("cuda"):
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = target_device
        else:
            model_kwargs["torch_dtype"] = torch.float32

        try:
            model = AutoModelForCausalLM.from_pretrained(cleaned_id, **model_kwargs)
            if not target_device.startswith("cuda"):
                model = model.to(target_device)
        except Exception as e:
            err_msg = str(e)
            if "requires `trust_remote_code=True`" in err_msg:
                raise UntrustedModelCodeError(
                    f"Model '{cleaned_id}' requires custom remote code execution, which is disabled for security."
                ) from e
            if "401" in err_msg or "403" in err_msg or "gated" in err_msg.lower():
                raise GatedModelAccessError(
                    f"Failed to authenticate access to model '{cleaned_id}'. Ensure a valid Hugging Face token is provided."
                ) from e
            if "not found" in err_msg.lower():
                raise ModelNotFoundError(f"Model '{cleaned_id}' was not found.") from e
            raise ModelLoadError(f"Failed to load causal language model for '{cleaned_id}': {err_msg}") from e

        return model, tokenizer, target_device


# Global service instance
_model_loader_instance: Optional[ModelLoaderService] = None


def get_model_loader_service() -> ModelLoaderService:
    """Returns singleton model loader service."""
    global _model_loader_instance
    if _model_loader_instance is None:
        _model_loader_instance = ModelLoaderService()
    return _model_loader_instance
