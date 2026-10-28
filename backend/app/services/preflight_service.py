"""
Preflight Validation Service for Adaptive LLM Fine-Tuning Studio (Phase 6).

Executes an ordered sequence of pre-launch checks against the training configuration,
hardware environment, ML dependency stack, output directory permissions, and model/dataset
registration state. Generates strongly-typed PreflightCheckResponse and serializable RunPlan.
"""

import os
import tempfile
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from backend.app.core.config import settings
from backend.app.schemas.preflight import (
    PreflightCheckResponse,
    PreflightCheckResult,
    PreflightStatus,
    RunPlan,
)
from backend.app.schemas.training import TrainingConfigSchema, TrainingMethod
from backend.app.core.system import get_system_hardware_info
from backend.app.services.ml_dependencies import check_ml_dependencies
from backend.app.services.model_catalog import get_model_catalog_service
from backend.app.services.dataset_service import get_dataset_service


# ─── VRAM Heuristics ──────────────────────────────────────────────────────────
_BYTES_PER_PARAM_FP16 = 2.0    # bfloat16 / float16
_BYTES_PER_PARAM_FP32 = 4.0    # full precision
_BYTES_PER_PARAM_4BIT = 0.5    # 4-bit NF4 quantized base weights
_LORA_OVERHEAD_MULTIPLIER = 1.35   # LoRA: base fp16 + adapter gradients
_QLORA_OVERHEAD_MULTIPLIER = 1.15  # QLoRA: 4-bit base + fp16 adapter
_FULL_FT_OVERHEAD_MULTIPLIER = 2.0 # full fine-tune: fp16 params + optimizer states


def _estimate_vram_gb(
    method: TrainingMethod,
    param_count_str: Optional[str],
    batch_size: int,
    context_length: int,
) -> Optional[float]:
    """
    Heuristically estimates VRAM requirement for the selected method and model size.
    Returns None if estimation is not possible (unknown parameter count).
    """
    params_billions: Optional[float] = None
    if param_count_str:
        pcs = param_count_str.strip().upper()
        try:
            if pcs.endswith("B"):
                params_billions = float(pcs[:-1])
            elif pcs.endswith("M"):
                params_billions = float(pcs[:-1]) / 1000.0
            elif pcs.endswith("T"):
                params_billions = float(pcs[:-1]) * 1000.0
            else:
                params_billions = float(pcs) / 1e9
        except (ValueError, IndexError):
            return None

    if params_billions is None:
        return None

    param_count = params_billions * 1e9

    if method == TrainingMethod.QLORA:
        base_bytes = param_count * _BYTES_PER_PARAM_4BIT
        total_bytes = base_bytes * _QLORA_OVERHEAD_MULTIPLIER
    elif method == TrainingMethod.LORA:
        base_bytes = param_count * _BYTES_PER_PARAM_FP16
        total_bytes = base_bytes * _LORA_OVERHEAD_MULTIPLIER
    else:  # full_finetuning
        base_bytes = param_count * _BYTES_PER_PARAM_FP16
        total_bytes = base_bytes * _FULL_FT_OVERHEAD_MULTIPLIER

    activation_bump = max(0.5, params_billions * 0.1 * batch_size)
    total_gb = (total_bytes / (1024 ** 3)) + activation_bump

    return round(total_gb, 1)


# ─── Individual Checks ────────────────────────────────────────────────────────

def _check_model_selected(config: TrainingConfigSchema) -> PreflightCheckResult:
    if config.model_identifier:
        return PreflightCheckResult(
            check_id="model_selected",
            label="Base Model",
            status=PreflightStatus.PASS,
            message=f"Model '{config.model_identifier}' is selected.",
        )
    return PreflightCheckResult(
        check_id="model_selected",
        label="Base Model",
        status=PreflightStatus.FAIL,
        message="No base model selected. Select a model to continue.",
    )


def _check_model_catalog(config: TrainingConfigSchema) -> Optional[PreflightCheckResult]:
    """Returns a check result only if model_identifier is set."""
    if not config.model_identifier:
        return None
    svc = get_model_catalog_service()
    model = svc.get_model(config.model_identifier)
    if not model:
        return PreflightCheckResult(
            check_id="model_catalog",
            label="Model Registration",
            status=PreflightStatus.WARN,
            message=(
                f"Model '{config.model_identifier}' is not registered in the local catalog. "
                "Hardware and context compatibility is unverified. The model may still be loadable "
                "from Hugging Face Hub if a token is configured."
            ),
        )
    # Method compatibility
    if config.training_method not in model.supported_methods:
        return PreflightCheckResult(
            check_id="model_catalog",
            label="Model Registration",
            status=PreflightStatus.FAIL,
            message=(
                f"Model '{model.name}' does not support '{config.training_method.value}'. "
                f"Supported methods: {', '.join(m.value for m in model.supported_methods)}."
            ),
            detail={"supported_methods": [m.value for m in model.supported_methods]},
        )
    # Gated model warning
    if model.is_gated:
        return PreflightCheckResult(
            check_id="model_catalog",
            label="Model Registration",
            status=PreflightStatus.WARN,
            message=(
                f"Model '{model.name}' is gated and requires a Hugging Face access token. "
                "Ensure the HF token is configured before launching."
            ),
            detail={"is_gated": True, "model_id": model.id},
        )
    return PreflightCheckResult(
        check_id="model_catalog",
        label="Model Registration",
        status=PreflightStatus.PASS,
        message=f"Model '{model.name}' is registered, supports {config.training_method.value}, and is accessible.",
        detail={"model_id": model.id, "family": model.family, "parameter_count": model.parameter_count},
    )


def _check_dataset_selected(config: TrainingConfigSchema) -> PreflightCheckResult:
    if config.dataset_reference:
        return PreflightCheckResult(
            check_id="dataset_selected",
            label="Training Dataset",
            status=PreflightStatus.PASS,
            message=f"Dataset reference '{config.dataset_reference}' is selected.",
        )
    return PreflightCheckResult(
        check_id="dataset_selected",
        label="Training Dataset",
        status=PreflightStatus.FAIL,
        message="No dataset selected. Upload or select a dataset to continue.",
    )


def _check_dataset_validation(config: TrainingConfigSchema) -> Optional[PreflightCheckResult]:
    """Returns a check result only if dataset_reference is set."""
    if not config.dataset_reference:
        return None
    svc = get_dataset_service()
    report = svc.get_dataset(config.dataset_reference) or svc.get_report(config.dataset_reference)
    if not report:
        return PreflightCheckResult(
            check_id="dataset_validation",
            label="Dataset Validation",
            status=PreflightStatus.WARN,
            message=(
                f"Dataset '{config.dataset_reference}' was not found in the uploaded dataset store. "
                "If using a Hugging Face Hub dataset, validation will be deferred to training time."
            ),
        )
    if not report.is_valid:
        error_count = report.invalid_sample_count
        return PreflightCheckResult(
            check_id="dataset_validation",
            label="Dataset Validation",
            status=PreflightStatus.FAIL,
            message=(
                f"Dataset '{report.original_filename}' has {error_count} invalid sample(s) "
                "and cannot be used for training. Re-upload a corrected dataset file."
            ),
            detail={"invalid_sample_count": error_count, "total_samples": report.total_samples},
        )
    return PreflightCheckResult(
        check_id="dataset_validation",
        label="Dataset Validation",
        status=PreflightStatus.PASS,
        message=(
            f"Dataset '{report.original_filename}' passed validation: "
            f"{report.valid_sample_count} valid samples, schema: {report.detected_schema or 'unknown'}."
        ),
        detail={
            "valid_sample_count": report.valid_sample_count,
            "schema": report.detected_schema,
            "format": report.file_format,
        },
    )


def _check_device(hw: Dict) -> PreflightCheckResult:
    detection_status = hw.get("detection_status")
    if detection_status == "detection_error":
        diagnostic = hw.get("diagnostic_message", "Hardware detection encountered an unexpected error.")
        return PreflightCheckResult(
            check_id="device",
            label="Compute Device",
            status=PreflightStatus.FAIL,
            message=f"Hardware detection error: {diagnostic}",
            detail={"detection_status": "detection_error", "diagnostic": diagnostic},
        )

    pytorch_ok = hw.get("pytorch_available", False)
    if not pytorch_ok:
        return PreflightCheckResult(
            check_id="device",
            label="Compute Device",
            status=PreflightStatus.FAIL,
            message=(
                "PyTorch is not installed in the runtime environment. "
                "Install it with CUDA support: pip install torch --index-url https://download.pytorch.org/whl/cu118"
            ),
            detail={"environment_type": hw.get("environment_type", "no_pytorch")},
        )

    cuda_ok = hw.get("cuda_available", False)
    gpu_count = hw.get("gpu_count", 0)
    if cuda_ok and gpu_count > 0:
        gpu_name = hw.get("primary_gpu_name") or hw.get("gpu_name") or "CUDA GPU"
        vram_gb = hw.get("primary_gpu_vram_gb")
        vram_str = f" ({vram_gb} GB VRAM)" if vram_gb is not None else ""
        return PreflightCheckResult(
            check_id="device",
            label="Compute Device",
            status=PreflightStatus.PASS,
            message=f"CUDA GPU detected: {gpu_name}{vram_str}. {gpu_count} device(s) available.",
            detail={
                "device_type": "cuda",
                "gpu_count": gpu_count,
                "primary_gpu_name": gpu_name,
                "primary_gpu_vram_gb": vram_gb,
            },
        )

    # CPU-only path
    diagnostic = hw.get("diagnostic_message", "CUDA is not available.")
    return PreflightCheckResult(
        check_id="device",
        label="Compute Device",
        status=PreflightStatus.WARN,
        message=f"CPU execution mode. {diagnostic} Training will be very slow on CPU.",
        detail={"device_type": "cpu", "detection_status": hw.get("detection_status")},
    )


def _check_method_device_compat(config: TrainingConfigSchema, hw: Dict) -> PreflightCheckResult:
    """Validates method vs device compatibility — QLoRA requires CUDA."""
    cuda_ok = hw.get("cuda_available", False)
    gpu_count = hw.get("gpu_count", 0)
    method = config.training_method

    if method == TrainingMethod.QLORA:
        if not cuda_ok or gpu_count <= 0:
            return PreflightCheckResult(
                check_id="method_device_compat",
                label="Method–Device Compatibility",
                status=PreflightStatus.FAIL,
                message=(
                    "QLoRA 4-bit quantization requires a CUDA-capable GPU. "
                    "bitsandbytes 4-bit operations are not supported on CPU. "
                    "Switch to LoRA method or attach a CUDA GPU."
                ),
                detail={"method": "qlora", "cuda_available": cuda_ok},
            )
        return PreflightCheckResult(
            check_id="method_device_compat",
            label="Method–Device Compatibility",
            status=PreflightStatus.PASS,
            message="QLoRA 4-bit quantization is compatible with the detected CUDA GPU.",
        )

    if method == TrainingMethod.FULL_FINETUNING:
        if not cuda_ok or gpu_count <= 0:
            return PreflightCheckResult(
                check_id="method_device_compat",
                label="Method–Device Compatibility",
                status=PreflightStatus.WARN,
                message=(
                    "Full fine-tuning on CPU is extremely slow and impractical for anything larger than ~125M parameters. "
                    "A CUDA GPU is strongly recommended."
                ),
                detail={"method": "full_finetuning", "cuda_available": False},
            )
    return PreflightCheckResult(
        check_id="method_device_compat",
        label="Method–Device Compatibility",
        status=PreflightStatus.PASS,
        message=f"{method.value.upper()} fine-tuning is compatible with the available compute device.",
    )


def _check_ml_deps(config: TrainingConfigSchema) -> List[PreflightCheckResult]:
    """Returns one check per critical ML dependency group."""
    results: List[PreflightCheckResult] = []
    dep_info = check_ml_dependencies()

    # Core trio: torch / transformers / peft
    if dep_info["is_ready"]:
        versions = dep_info.get("versions", {})
        ver_str = ", ".join(
            f"{k} {v}" for k, v in versions.items()
            if k in ("torch", "transformers", "peft") and v
        )
        results.append(PreflightCheckResult(
            check_id="ml_core_deps",
            label="Core ML Stack",
            status=PreflightStatus.PASS,
            message=f"Core ML stack is installed: {ver_str}.",
            detail={"versions": versions},
        ))
    else:
        missing = dep_info.get("missing_critical", [])
        results.append(PreflightCheckResult(
            check_id="ml_core_deps",
            label="Core ML Stack",
            status=PreflightStatus.FAIL,
            message=(
                f"Critical ML packages missing: {', '.join(missing)}. "
                "Install with: pip install torch transformers peft accelerate"
            ),
            detail={"missing": missing},
        ))

    # bitsandbytes — only relevant for QLoRA
    if config.training_method == TrainingMethod.QLORA:
        bnb_ok = dep_info.get("bitsandbytes_available", False)
        if bnb_ok:
            bnb_ver = dep_info.get("versions", {}).get("bitsandbytes")
            results.append(PreflightCheckResult(
                check_id="bitsandbytes",
                label="bitsandbytes (4-bit)",
                status=PreflightStatus.PASS,
                message=f"bitsandbytes{f' {bnb_ver}' if bnb_ver else ''} is available for QLoRA 4-bit quantization.",
            ))
        else:
            results.append(PreflightCheckResult(
                check_id="bitsandbytes",
                label="bitsandbytes (4-bit)",
                status=PreflightStatus.FAIL,
                message=(
                    "bitsandbytes is required for QLoRA 4-bit fine-tuning but is not installed. "
                    "Install with: pip install bitsandbytes"
                ),
            ))
    else:
        results.append(PreflightCheckResult(
            check_id="bitsandbytes",
            label="bitsandbytes (4-bit)",
            status=PreflightStatus.SKIP,
            message=f"bitsandbytes check skipped — not required for {config.training_method.value}.",
        ))

    return results


def _check_output_dir() -> PreflightCheckResult:
    """Verifies that the runs output directory is writable without exposing sensitive paths."""
    out_dir = settings.runs_output_dir
    try:
        os.makedirs(out_dir, exist_ok=True)
        # Test write permission using a temporary sentinel file
        test_file = os.path.join(out_dir, ".perm_test")
        try:
            with open(test_file, "w") as f:
                f.write("perm_test")
            os.remove(test_file)
        except (IOError, OSError, PermissionError) as e:
            return PreflightCheckResult(
                check_id="output_dir",
                label="Output Directory",
                status=PreflightStatus.FAIL,
                message=f"Runs output directory has insufficient write permissions: {str(e)}",
                detail={"writable": False},
            )
        return PreflightCheckResult(
            check_id="output_dir",
            label="Output Directory",
            status=PreflightStatus.PASS,
            message="Runs artifact directory is verified writable.",
            detail={"writable": True},
        )
    except Exception as e:
        return PreflightCheckResult(
            check_id="output_dir",
            label="Output Directory",
            status=PreflightStatus.FAIL,
            message=f"Output directory access error: {str(e)}",
            detail={"writable": False},
        )


def _check_vram(
    config: TrainingConfigSchema,
    hw: Dict,
    estimated_vram_gb: Optional[float],
) -> PreflightCheckResult:
    """Estimates VRAM sufficiency for the chosen method and model."""
    cuda_ok = hw.get("cuda_available", False)
    available_vram_gb = hw.get("primary_gpu_vram_gb")

    if not cuda_ok or available_vram_gb is None:
        if estimated_vram_gb is not None:
            return PreflightCheckResult(
                check_id="vram",
                label="VRAM Capacity",
                status=PreflightStatus.WARN,
                message=(
                    f"Estimated VRAM requirement: ~{estimated_vram_gb} GB. "
                    "No GPU detected — unable to verify VRAM sufficiency."
                ),
                detail={"estimated_gb": estimated_vram_gb, "available_gb": None, "vram_status": "unknown"},
            )
        return PreflightCheckResult(
            check_id="vram",
            label="VRAM Capacity",
            status=PreflightStatus.SKIP,
            message="VRAM check skipped — no CUDA GPU detected.",
            detail={"vram_status": "unknown"},
        )

    if estimated_vram_gb is None:
        return PreflightCheckResult(
            check_id="vram",
            label="VRAM Capacity",
            status=PreflightStatus.WARN,
            message=(
                f"Available VRAM: {available_vram_gb} GB. "
                "VRAM estimate unavailable — model parameter count is unknown."
            ),
            detail={"available_gb": available_vram_gb, "estimated_gb": None, "vram_status": "unknown"},
        )

    if estimated_vram_gb > available_vram_gb * 1.05:  # 5% margin
        return PreflightCheckResult(
            check_id="vram",
            label="VRAM Capacity",
            status=PreflightStatus.WARN,
            message=(
                f"Estimated requirement (~{estimated_vram_gb} GB) may exceed available VRAM "
                f"({available_vram_gb} GB). Training may fail with OOM errors. "
                f"Try reducing batch size, context length, or switching to {TrainingMethod.QLORA.value}."
            ),
            detail={"estimated_gb": estimated_vram_gb, "available_gb": available_vram_gb, "vram_status": "unsupported"},
        )

    return PreflightCheckResult(
        check_id="vram",
        label="VRAM Capacity",
        status=PreflightStatus.PASS,
        message=(
            f"Estimated VRAM requirement (~{estimated_vram_gb} GB) fits within "
            f"available VRAM ({available_vram_gb} GB)."
        ),
        detail={"estimated_gb": estimated_vram_gb, "available_gb": available_vram_gb, "vram_status": "supported"},
    )


def _check_hf_token(config: TrainingConfigSchema, hf_token: Optional[str]) -> PreflightCheckResult:
    """Checks Hugging Face token availability for gated model access."""
    has_env_token = config.hf_token_status == "present_in_env"
    has_in_memory = bool(hf_token and hf_token.strip())

    if has_env_token:
        return PreflightCheckResult(
            check_id="hf_token",
            label="HF Access Token",
            status=PreflightStatus.PASS,
            message="Hugging Face token detected in environment (HF_TOKEN or HUGGING_FACE_HUB_TOKEN).",
            detail={"source": "env"},
        )
    if has_in_memory:
        return PreflightCheckResult(
            check_id="hf_token",
            label="HF Access Token",
            status=PreflightStatus.PASS,
            message="Hugging Face token is configured for this session. Note: token is not verified against the Hub.",
            detail={"source": "session"},
        )
    # No token — check if model is gated
    if config.model_identifier:
        svc = get_model_catalog_service()
        model = svc.get_model(config.model_identifier)
        if model and model.is_gated:
            return PreflightCheckResult(
                check_id="hf_token",
                label="HF Access Token",
                status=PreflightStatus.FAIL,
                message=(
                    f"Model '{model.name}' is gated but no Hugging Face token is configured. "
                    "Provide a valid HF token to access this model."
                ),
                detail={"model_id": model.id, "is_gated": True},
            )
    return PreflightCheckResult(
        check_id="hf_token",
        label="HF Access Token",
        status=PreflightStatus.WARN,
        message=(
            "No Hugging Face token configured. Required for gated models (e.g. Llama, Gemma). "
            "Public models will still be accessible without a token."
        ),
    )


def _check_config_schema(config: TrainingConfigSchema) -> PreflightCheckResult:
    """Confirms the training configuration passed schema validation."""
    return PreflightCheckResult(
        check_id="config_schema",
        label="Configuration Schema",
        status=PreflightStatus.PASS,
        message=(
            f"Training configuration is valid: {config.training_method.value.upper()}, "
            f"LR={config.learning_rate}, batch={config.batch_size}, "
            f"context={config.context_length} tokens."
        ),
        detail={
            "training_method": config.training_method.value,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "context_length": config.context_length,
            "optimizer": config.optimizer,
        },
    )


# ─── Orchestrator ─────────────────────────────────────────────────────────────

def run_preflight(
    config: TrainingConfigSchema,
    hf_token: Optional[str] = None,
) -> PreflightCheckResponse:
    """
    Executes all preflight checks in priority order and returns an aggregated result.

    Order:
    1. Config schema validity
    2. Base model selected
    3. Model catalog registration and method compatibility
    4. Dataset selected
    5. Dataset validation report
    6. Compute device detection & CUDA availability
    7. Method–device compatibility
    8. ML dependency stack (core trio)
    9. bitsandbytes for QLoRA
    10. Output directory permissions
    11. VRAM capacity estimate
    12. HF token status
    """
    hw = get_system_hardware_info()
    checks: List[PreflightCheckResult] = []

    # 1. Config schema
    checks.append(_check_config_schema(config))

    # 2. Base model selected
    checks.append(_check_model_selected(config))

    # 3. Model catalog
    catalog_check = _check_model_catalog(config)
    if catalog_check:
        checks.append(catalog_check)

    # 4. Dataset selected
    checks.append(_check_dataset_selected(config))

    # 5. Dataset validation
    ds_validation = _check_dataset_validation(config)
    if ds_validation:
        checks.append(ds_validation)

    # 6. Compute device
    checks.append(_check_device(hw))

    # 7. Method–device compatibility
    checks.append(_check_method_device_compat(config, hw))

    # 8+9. ML dependencies (core + bitsandbytes)
    checks.extend(_check_ml_deps(config))

    # 10. Output directory permissions
    checks.append(_check_output_dir())

    # Determine parameter count for VRAM estimation
    param_count_str: Optional[str] = None
    if config.model_identifier:
        svc = get_model_catalog_service()
        model = svc.get_model(config.model_identifier)
        if model:
            param_count_str = model.parameter_count

    estimated_vram_gb = _estimate_vram_gb(
        method=config.training_method,
        param_count_str=param_count_str,
        batch_size=config.batch_size,
        context_length=config.context_length,
    )

    # 11. VRAM
    vram_check = _check_vram(config, hw, estimated_vram_gb)
    checks.append(vram_check)

    # 12. HF token
    checks.append(_check_hf_token(config, hf_token))

    # Aggregate results
    blocking = [c for c in checks if c.status == PreflightStatus.FAIL]
    warnings = [c for c in checks if c.status == PreflightStatus.WARN]
    can_launch = len(blocking) == 0

    vram_detail = vram_check.detail or {}
    vram_status = vram_detail.get("vram_status")
    vram_message = vram_check.message

    return PreflightCheckResponse(
        checks=checks,
        can_launch=can_launch,
        blocking_count=len(blocking),
        warning_count=len(warnings),
        method=config.training_method.value,
        estimated_vram_gb=estimated_vram_gb,
        vram_status=vram_status,
        vram_message=vram_message,
    )


# ─── RunPlan Generator ────────────────────────────────────────────────────────

def generate_run_plan(
    config: TrainingConfigSchema,
    hf_token: Optional[str] = None,
) -> RunPlan:
    """
    Generates a structured, serializable RunPlan from the current configuration.
    Strictly excludes access tokens, secrets, and sensitive absolute host paths.
    """
    preflight = run_preflight(config=config, hf_token=hf_token)
    hw = get_system_hardware_info()

    # 1. Sanitize run configuration (remove any possible secret fields)
    raw_config = config.model_dump()
    sanitized_config: Dict[str, Any] = {}
    for k, v in raw_config.items():
        if any(secret_term in k.lower() for secret_term in ("token", "secret", "password", "key", "auth")):
            if k == "hf_token_status":
                sanitized_config[k] = v
            continue
        sanitized_config[k] = v

    # Add token presence indicator without leaking secret
    has_token = bool(hf_token and hf_token.strip()) or config.hf_token_status == "present_in_env"
    sanitized_config["has_hf_token_configured"] = has_token

    # 2. Selected model summary
    selected_model_info: Optional[Dict[str, Any]] = None
    if config.model_identifier:
        svc = get_model_catalog_service()
        model = svc.get_model(config.model_identifier)
        if model:
            selected_model_info = {
                "id": model.id,
                "name": model.name,
                "family": model.family,
                "parameter_count": model.parameter_count,
                "context_length": model.context_length,
                "is_gated": model.is_gated,
                "supported_methods": [m.value for m in model.supported_methods],
            }
        else:
            selected_model_info = {
                "id": config.model_identifier,
                "name": config.model_identifier.split("/")[-1],
                "family": "custom",
                "parameter_count": None,
                "context_length": config.context_length,
                "is_gated": False,
                "supported_methods": [config.training_method.value],
            }

    # 3. Selected dataset summary
    dataset_info: Optional[Dict[str, Any]] = None
    if config.dataset_reference:
        dsvc = get_dataset_service()
        report = dsvc.get_dataset(config.dataset_reference) or dsvc.get_report(config.dataset_reference)
        if report:
            dataset_info = {
                "reference": config.dataset_reference,
                "filename": report.original_filename,
                "format": report.file_format,
                "total_samples": report.total_samples,
                "valid_sample_count": report.valid_sample_count,
                "is_valid": report.is_valid,
                "detected_schema": report.detected_schema,
            }
        else:
            dataset_info = {
                "reference": config.dataset_reference,
                "filename": config.dataset_reference,
                "format": "unknown",
                "total_samples": None,
                "valid_sample_count": None,
                "is_valid": None,
                "detected_schema": None,
            }

    # 4. Device information
    device_info = {
        "device_type": hw.get("device_type", "cpu"),
        "primary_gpu_name": hw.get("primary_gpu_name") or hw.get("gpu_name"),
        "gpu_count": hw.get("gpu_count", 0),
        "primary_gpu_vram_gb": hw.get("primary_gpu_vram_gb"),
        "cuda_available": hw.get("cuda_available", False),
        "cuda_version": hw.get("cuda_version"),
        "detection_status": hw.get("detection_status", "unknown"),
        "diagnostic_message": hw.get("diagnostic_message", ""),
    }

    # 5. Resource compatibility
    dep_info = check_ml_dependencies()
    resource_compat = {
        "pytorch_available": hw.get("pytorch_available", False),
        "cuda_available": hw.get("cuda_available", False),
        "gpu_count": hw.get("gpu_count", 0),
        "bitsandbytes_available": dep_info.get("bitsandbytes_available", False),
        "method": config.training_method.value,
        "method_supported": preflight.can_launch or not any(
            c.check_id in ("method_device_compat", "bitsandbytes") and c.status == PreflightStatus.FAIL
            for c in preflight.checks
        ),
        "vram_status": preflight.vram_status or "unknown",
        "estimated_vram_gb": preflight.estimated_vram_gb,
        "vram_message": preflight.vram_message,
    }

    # 6. Output configuration (relative, safe paths only)
    out_dir_relative = os.path.basename(settings.runs_output_dir)
    output_config = {
        "output_directory": out_dir_relative,
        "project_name": config.project_name,
        "logging_steps": getattr(config, "logging_steps", 1),
        "seed": config.seed,
        "mixed_precision": config.mixed_precision,
        "gradient_checkpointing": config.gradient_checkpointing,
        "optimizer": config.optimizer,
    }

    # 7. Collect warnings
    warnings_list = [c.message for c in preflight.checks if c.status == PreflightStatus.WARN]

    return RunPlan(
        run_configuration=sanitized_config,
        selected_model=selected_model_info,
        dataset_reference=dataset_info,
        fine_tuning_method=config.training_method.value,
        device_information=device_info,
        validation_results=preflight,
        warnings=warnings_list,
        resource_compatibility=resource_compat,
        output_configuration=output_config,
        can_launch=preflight.can_launch,
        blocking_count=preflight.blocking_count,
        warning_count=preflight.warning_count,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
