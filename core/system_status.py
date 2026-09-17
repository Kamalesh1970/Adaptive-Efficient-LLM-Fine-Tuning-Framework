"""
System and Environment Status Utility.
"""

import sys
import platform
from typing import Dict, Any, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def get_system_status() -> Dict[str, Any]:
    """
    Collects system, Python, PyTorch, and CUDA status information.

    Returns:
        Dict containing key platform and framework environment details.
    """
    status: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "os_platform": platform.platform(),
        "pytorch_available": HAS_TORCH,
        "pytorch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "cuda_device_count": 0,
        "cuda_device_names": [],
    }

    if HAS_TORCH:
        status["pytorch_version"] = torch.__version__
        cuda_is_avail = torch.cuda.is_available()
        status["cuda_available"] = cuda_is_avail
        if cuda_is_avail:
            status["cuda_version"] = torch.version.cuda
            count = torch.cuda.device_count()
            status["cuda_device_count"] = count
            status["cuda_device_names"] = [
                torch.cuda.get_device_name(i) for i in range(count)
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
