"""
Machine Learning Dependency Detection and Compatibility Utility (Phase 4).
Safely probes availability of PyTorch, Hugging Face Transformers, PEFT, and Accelerate.
Prevents unhandled crashes and provides clear dependency feedback.
"""

from typing import Dict, Any, Optional
import sys

class MLDependencyMissingError(RuntimeError):
    """Raised when machine learning execution is requested but required packages are missing."""
    pass


class BitsAndBytesMissingError(MLDependencyMissingError):
    """Raised when 4-bit or 8-bit quantization is requested but bitsandbytes is not installed."""
    pass


class CudaUnavailableForQuantizationError(RuntimeError):
    """Raised when QLoRA quantization is attempted on a CPU without CUDA GPU hardware."""
    pass


def is_torch_available() -> bool:
    """Checks if PyTorch is installed and importable."""
    return "torch" in sys.modules or _can_import("torch")


def is_transformers_available() -> bool:
    """Checks if Hugging Face Transformers is installed and importable."""
    return "transformers" in sys.modules or _can_import("transformers")


def is_peft_available() -> bool:
    """Checks if Hugging Face PEFT is installed and importable."""
    return "peft" in sys.modules or _can_import("peft")


def is_accelerate_available() -> bool:
    """Checks if Hugging Face Accelerate is installed and importable."""
    return "accelerate" in sys.modules or _can_import("accelerate")


def is_datasets_available() -> bool:
    """Checks if Hugging Face Datasets is installed and importable."""
    return "datasets" in sys.modules or _can_import("datasets")


def is_bitsandbytes_available() -> bool:
    """Checks if bitsandbytes quantization library is installed and importable."""
    return "bitsandbytes" in sys.modules or _can_import("bitsandbytes")


def is_full_ml_stack_available() -> bool:
    """Checks whether the core trio (PyTorch, Transformers, PEFT) is present."""
    return is_torch_available() and is_transformers_available() and is_peft_available()


def _can_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except (ImportError, Exception):
        return False


def get_ml_version_info() -> Dict[str, Optional[str]]:
    """Returns detected version strings for installed ML libraries."""
    versions: Dict[str, Optional[str]] = {
        "torch": None,
        "transformers": None,
        "peft": None,
        "accelerate": None,
        "datasets": None,
        "bitsandbytes": None,
    }
    for mod in versions.keys():
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            versions[mod] = None
    return versions


def check_ml_dependencies() -> Dict[str, Any]:
    """Returns availability and readiness of machine learning packages."""
    versions = get_ml_version_info()
    ready = is_full_ml_stack_available()
    missing = [pkg for pkg, ver in versions.items() if ver is None and pkg in ("torch", "transformers", "peft")]
    return {
        "is_ready": ready,
        "versions": versions,
        "missing_critical": missing,
        "bitsandbytes_available": is_bitsandbytes_available(),
        "status": "ready" if ready else "dependencies_missing",
    }


def verify_ml_dependencies_or_raise() -> None:
    """Raises MLDependencyMissingError if any of torch, transformers, or peft is missing."""
    info = check_ml_dependencies()
    if not info["is_ready"]:
        missing_str = ", ".join(info["missing_critical"])
        raise MLDependencyMissingError(
            f"Machine learning dependencies ({missing_str}) are not installed. "
            f"Install them via: pip install torch transformers peft accelerate"
        )


def verify_qlora_dependencies_or_raise(is_synthetic_model: bool = False, device: str = "cpu") -> None:
    """
    Verifies full software and hardware compatibility for QLoRA fine-tuning.
    Requires torch, transformers, peft, accelerate, bitsandbytes, and a CUDA GPU.
    """
    if is_synthetic_model:
        return

    # 1. Base ML libraries
    verify_ml_dependencies_or_raise()

    # 2. bitsandbytes library
    if not is_bitsandbytes_available():
        raise BitsAndBytesMissingError(
            "bitsandbytes is required for 4-bit/8-bit QLoRA fine-tuning but is not installed. "
            "Install it via: pip install bitsandbytes"
        )

    # 3. Hardware check: bitsandbytes 4-bit quantization natively requires CUDA
    from backend.app.core.system import get_system_hardware_info
    hw = get_system_hardware_info()
    cuda_avail = hw.get("cuda_available", False)
    gpu_count = hw.get("gpu_count", 0)

    if not cuda_avail or gpu_count <= 0 or (device and device.lower() == "cpu"):
        raise CudaUnavailableForQuantizationError(
            "QLoRA 4-bit fine-tuning requires a compatible CUDA GPU. "
            "CUDA hardware is not detected or device is set to CPU. "
            "CPU execution is not supported for 4-bit bitsandbytes quantization."
        )

