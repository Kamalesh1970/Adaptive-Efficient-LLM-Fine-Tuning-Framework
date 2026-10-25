"""
System and Hardware Diagnostic Utility for Phase 0 Backend.

Safely probes the execution environment for:
- Python version and OS platform
- PyTorch availability and version
- CUDA availability, device count, device names, and VRAM capacity
Gracefully handles missing PyTorch, CPU-only environments, and unavailable CUDA.
"""

import sys
import platform
from typing import Dict, Any, Optional, List


def _safe_import_torch() -> Optional[Any]:
    """Safely retrieves torch module if available without raising an unhandled exception."""
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        return torch_mod
    try:
        import torch
        return torch
    except ImportError:
        return None


def get_system_hardware_info() -> Dict[str, Any]:
    """
    Collects actual system, PyTorch, and GPU hardware status.

    Returns:
        Dict containing verified platform and hardware details.
    """
    info: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "os_platform": platform.platform(),
        "pytorch_available": False,
        "pytorch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_count": 0,
        "gpu_devices": [],
        "primary_gpu_name": None,
        "primary_gpu_vram_gb": None,
        "environment_type": "cpu_only",
    }

    torch_mod = _safe_import_torch()
    if torch_mod is None:
        info["environment_type"] = "no_pytorch"
        return info

    info["pytorch_available"] = True
    info["pytorch_version"] = getattr(torch_mod, "__version__", None)

    cuda_is_avail = False
    if hasattr(torch_mod, "cuda") and callable(getattr(torch_mod.cuda, "is_available", None)):
        try:
            cuda_is_avail = bool(torch_mod.cuda.is_available())
        except Exception:
            cuda_is_avail = False

    info["cuda_available"] = cuda_is_avail

    if not cuda_is_avail:
        info["environment_type"] = "cpu_only"
        return info

    # CUDA is available; safely inspect devices
    info["environment_type"] = "cuda_accelerated"
    try:
        version_obj = getattr(torch_mod, "version", None)
        info["cuda_version"] = getattr(version_obj, "cuda", None) if version_obj else None
    except Exception:
        info["cuda_version"] = None

    device_count = 0
    if hasattr(torch_mod.cuda, "device_count") and callable(torch_mod.cuda.device_count):
        try:
            device_count = int(torch_mod.cuda.device_count())
        except Exception:
            device_count = 0

    info["gpu_count"] = device_count
    devices_list: List[Dict[str, Any]] = []

    for idx in range(device_count):
        dev_info: Dict[str, Any] = {
            "index": idx,
            "name": f"GPU {idx}",
            "total_vram_gb": None,
            "total_vram_mb": None,
        }
        if hasattr(torch_mod.cuda, "get_device_name") and callable(torch_mod.cuda.get_device_name):
            try:
                dev_info["name"] = torch_mod.cuda.get_device_name(idx)
            except Exception:
                pass

        if hasattr(torch_mod.cuda, "get_device_properties") and callable(torch_mod.cuda.get_device_properties):
            try:
                props = torch_mod.cuda.get_device_properties(idx)
                if hasattr(props, "total_memory"):
                    total_bytes = props.total_memory
                    dev_info["total_vram_gb"] = round(total_bytes / (1024 ** 3), 2)
                    dev_info["total_vram_mb"] = round(total_bytes / (1024 ** 2), 2)
            except Exception:
                pass

        devices_list.append(dev_info)

    info["gpu_devices"] = devices_list

    if devices_list:
        info["primary_gpu_name"] = devices_list[0].get("name")
        info["primary_gpu_vram_gb"] = devices_list[0].get("total_vram_gb")

    return info
