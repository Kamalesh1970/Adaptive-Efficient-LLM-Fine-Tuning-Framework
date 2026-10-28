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
        "cuda_build_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_available": False,
        "device_type": "cpu",
        "gpu_name": None,
        "gpu_count": 0,
        "gpu_devices": [],
        "primary_gpu_name": None,
        "primary_gpu_vram_gb": None,
        "total_vram_bytes": None,
        "free_vram_bytes": None,
        "detection_status": "unknown",
        "diagnostic_message": None,
        "environment_type": "cpu_only",
    }

    torch_mod = _safe_import_torch()
    if torch_mod is None:
        info["environment_type"] = "no_pytorch"
        info["detection_status"] = "no_pytorch"
        info["diagnostic_message"] = "PyTorch is not installed in the runtime environment."
        return info

    info["pytorch_available"] = True
    info["pytorch_version"] = getattr(torch_mod, "__version__", None)

    cuda_build = None
    try:
        version_obj = getattr(torch_mod, "version", None)
        cuda_build = getattr(version_obj, "cuda", None) if version_obj else None
    except Exception:
        cuda_build = None
    info["cuda_build_version"] = cuda_build
    info["cuda_version"] = cuda_build

    cuda_is_avail = False
    init_error = None
    if hasattr(torch_mod, "cuda") and callable(getattr(torch_mod.cuda, "is_available", None)):
        try:
            cuda_is_avail = bool(torch_mod.cuda.is_available())
        except Exception as e:
            cuda_is_avail = False
            init_error = str(e)
    else:
        init_error = "PyTorch cuda module does not provide is_available()"

    info["cuda_available"] = cuda_is_avail

    if init_error is not None:
        info["detection_status"] = "cuda_initialization_error"
        info["environment_type"] = "cpu_only"
        info["diagnostic_message"] = f"CUDA initialization failed: {init_error}"
        return info

    if not cuda_is_avail:
        info["environment_type"] = "cpu_only"
        if cuda_build is None:
            info["detection_status"] = "cpu_only_build"
            info["diagnostic_message"] = "PyTorch was built without CUDA support (CPU-only build)."
        else:
            info["detection_status"] = "cuda_unavailable"
            info["diagnostic_message"] = (
                f"PyTorch was built with CUDA {cuda_build}, but CUDA runtime is not available on this system."
            )
        return info

    # CUDA is available; safely inspect devices
    device_count = 0
    try:
        if hasattr(torch_mod.cuda, "device_count") and callable(torch_mod.cuda.device_count):
            device_count = int(torch_mod.cuda.device_count())
    except Exception as e:
        info["detection_status"] = "cuda_initialization_error"
        info["environment_type"] = "cpu_only"
        info["diagnostic_message"] = f"Failed to inspect CUDA device count: {str(e)}"
        return info

    info["gpu_count"] = device_count

    if device_count <= 0:
        info["detection_status"] = "zero_gpus_detected"
        info["environment_type"] = "cpu_only"
        info["diagnostic_message"] = "CUDA is available, but zero GPU devices are visible to the process."
        return info

    info["gpu_available"] = True
    info["device_type"] = "cuda"
    info["detection_status"] = "gpu_detected"
    info["environment_type"] = "cuda_accelerated"

    devices_list: List[Dict[str, Any]] = []
    for idx in range(device_count):
        dev_info: Dict[str, Any] = {
            "index": idx,
            "name": f"GPU {idx}",
            "total_vram_gb": None,
            "total_vram_mb": None,
            "total_vram_bytes": None,
            "free_vram_bytes": None,
        }
        if hasattr(torch_mod.cuda, "get_device_name") and callable(torch_mod.cuda.get_device_name):
            try:
                dev_info["name"] = torch_mod.cuda.get_device_name(idx)
            except Exception:
                pass

        total_bytes = None
        if hasattr(torch_mod.cuda, "get_device_properties") and callable(torch_mod.cuda.get_device_properties):
            try:
                props = torch_mod.cuda.get_device_properties(idx)
                if hasattr(props, "total_memory"):
                    total_bytes = int(props.total_memory)
            except Exception:
                pass

        free_bytes = None
        if hasattr(torch_mod.cuda, "mem_get_info") and callable(torch_mod.cuda.mem_get_info):
            try:
                free_b, tot_b = torch_mod.cuda.mem_get_info(idx)
                free_bytes = int(free_b)
                if total_bytes is None:
                    total_bytes = int(tot_b)
            except Exception:
                pass

        dev_info["total_vram_bytes"] = total_bytes
        dev_info["free_vram_bytes"] = free_bytes
        if total_bytes is not None:
            dev_info["total_vram_gb"] = round(total_bytes / (1024 ** 3), 2)
            dev_info["total_vram_mb"] = round(total_bytes / (1024 ** 2), 2)

        devices_list.append(dev_info)

    info["gpu_devices"] = devices_list

    if devices_list:
        primary = devices_list[0]
        info["gpu_name"] = primary.get("name")
        info["primary_gpu_name"] = primary.get("name")
        info["primary_gpu_vram_gb"] = primary.get("total_vram_gb")
        info["total_vram_bytes"] = primary.get("total_vram_bytes")
        info["free_vram_bytes"] = primary.get("free_vram_bytes")

    vram_desc = f" ({info['primary_gpu_vram_gb']} GB VRAM)" if info["primary_gpu_vram_gb"] is not None else ""
    info["diagnostic_message"] = f"Detected {device_count} GPU device(s). Primary: {info['gpu_name']}{vram_desc}."

    return info
