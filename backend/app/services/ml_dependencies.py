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
