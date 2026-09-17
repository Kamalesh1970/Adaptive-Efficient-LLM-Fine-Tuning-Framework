"""
System and Environment Status Utility.
"""

import sys
import platform
from typing import Dict, Any, Optional

def _get_torch_module() -> Optional[Any]:
    """Safely retrieves torch module if imported or present in sys.modules."""
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        return torch_mod
    try:
        import torch
        return torch
    except ImportError:
        return None


def get_system_status() -> Dict[str, Any]:
    """
    Collects system, Python, PyTorch, and CUDA status information.

    Returns:
        Dict containing key platform and framework environment details.
    """
    status: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "os_platform": platform.platform(),
        "pytorch_available": False,
        "pytorch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "cuda_device_count": 0,
        "cuda_device_names": [],
    }

    torch_mod = _get_torch_module()
    if torch_mod is not None:
        status["pytorch_available"] = True
        status["pytorch_version"] = getattr(torch_mod, "__version__", None)
        cuda_is_avail = False
        if hasattr(torch_mod, "cuda") and callable(getattr(torch_mod.cuda, "is_available", None)):
            cuda_is_avail = torch_mod.cuda.is_available()

        status["cuda_available"] = cuda_is_avail
        if cuda_is_avail:
            version_obj = getattr(torch_mod, "version", None)
            status["cuda_version"] = getattr(version_obj, "cuda", None) if version_obj else None
            count = torch_mod.cuda.device_count() if hasattr(torch_mod.cuda, "device_count") else 0
            status["cuda_device_count"] = count
            if hasattr(torch_mod.cuda, "get_device_name"):
                status["cuda_device_names"] = [
                    torch_mod.cuda.get_device_name(i) for i in range(count)
                ]

    return status


def format_system_status() -> str:
    """
    Returns a human-readable formatted string of system status.
    """
    status = get_system_status()

    lines = [
        "==================================================",
        "Adaptive Efficient LLM Fine-Tuning Framework Status",
        "==================================================",
        f"Python Version    : {status['python_version']}",
        f"OS Platform       : {status['os_platform']}",
        f"PyTorch Available : {status['pytorch_available']}",
    ]

    if status["pytorch_available"]:
        lines.append(f"PyTorch Version   : {status['pytorch_version']}")
        lines.append(f"CUDA Available    : {status['cuda_available']}")
        if status["cuda_available"]:
            lines.append(f"CUDA Version      : {status['cuda_version']}")
            lines.append(f"CUDA Device Count : {status['cuda_device_count']}")
            for idx, dev_name in enumerate(status["cuda_device_names"]):
                lines.append(f"  - Device [{idx}]   : {dev_name}")
    else:
        lines.append("PyTorch Version   : N/A")
        lines.append("CUDA Available    : N/A")

    lines.append("==================================================")
    return "\n".join(lines)
