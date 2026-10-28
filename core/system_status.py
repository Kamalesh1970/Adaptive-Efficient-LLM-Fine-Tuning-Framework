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
        "cuda_build_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_available": False,
        "device_type": "cpu",
        "gpu_name": None,
        "gpu_count": 0,
        "cuda_device_count": 0,
        "cuda_device_names": [],
        "total_vram_bytes": None,
        "free_vram_bytes": None,
        "detection_status": "unknown",
        "diagnostic_message": None,
    }

    torch_mod = _get_torch_module()
    if torch_mod is None:
        status["detection_status"] = "no_pytorch"
        status["diagnostic_message"] = "PyTorch is not installed in the runtime environment."
        return status

    status["pytorch_available"] = True
    status["pytorch_version"] = getattr(torch_mod, "__version__", None)

    cuda_build = None
    try:
        version_obj = getattr(torch_mod, "version", None)
        cuda_build = getattr(version_obj, "cuda", None) if version_obj else None
    except Exception:
        cuda_build = None
    status["cuda_build_version"] = cuda_build
    status["cuda_version"] = cuda_build

    cuda_is_avail = False
    init_error = None
    if hasattr(torch_mod, "cuda") and callable(getattr(torch_mod.cuda, "is_available", None)):
        try:
            cuda_is_avail = bool(torch_mod.cuda.is_available())
        except Exception as e:
            cuda_is_avail = False
            init_error = str(e)
    else:
        init_error = "torch.cuda module does not provide is_available()"

    status["cuda_available"] = cuda_is_avail

    if init_error is not None:
        status["detection_status"] = "cuda_initialization_error"
        status["diagnostic_message"] = f"CUDA initialization failed: {init_error}"
        return status

    if not cuda_is_avail:
        if cuda_build is None:
            status["detection_status"] = "cpu_only_build"
            status["diagnostic_message"] = "PyTorch was built without CUDA support (CPU-only build)."
        else:
            status["detection_status"] = "cuda_unavailable"
            status["diagnostic_message"] = (
                f"PyTorch was built with CUDA {cuda_build}, but CUDA runtime is not available on this system."
            )
        return status

    # CUDA is available
    count = 0
    try:
        if hasattr(torch_mod.cuda, "device_count") and callable(torch_mod.cuda.device_count):
            count = int(torch_mod.cuda.device_count())
    except Exception as e:
        status["detection_status"] = "cuda_initialization_error"
        status["diagnostic_message"] = f"Failed to inspect CUDA device count: {str(e)}"
        return status

    status["gpu_count"] = count
    status["cuda_device_count"] = count

    if count <= 0:
        status["detection_status"] = "zero_gpus_detected"
        status["diagnostic_message"] = "CUDA is available, but zero GPU devices are visible to the process."
        return status

    status["gpu_available"] = True
    status["device_type"] = "cuda"
    status["detection_status"] = "gpu_detected"

    device_names = []
    for i in range(count):
        dev_name = f"GPU {i}"
        if hasattr(torch_mod.cuda, "get_device_name"):
            try:
                dev_name = torch_mod.cuda.get_device_name(i)
            except Exception:
                pass
        device_names.append(dev_name)

    status["cuda_device_names"] = device_names
    if device_names:
        status["gpu_name"] = device_names[0]

    # VRAM memory check for primary device
    try:
        if hasattr(torch_mod.cuda, "mem_get_info") and callable(torch_mod.cuda.mem_get_info):
            free_b, tot_b = torch_mod.cuda.mem_get_info(0)
            status["free_vram_bytes"] = int(free_b)
            status["total_vram_bytes"] = int(tot_b)
        elif hasattr(torch_mod.cuda, "get_device_properties") and callable(torch_mod.cuda.get_device_properties):
            props = torch_mod.cuda.get_device_properties(0)
            if hasattr(props, "total_memory"):
                status["total_vram_bytes"] = int(props.total_memory)
    except Exception:
        pass

    vram_desc = ""
    if status["total_vram_bytes"] is not None:
        total_gb = round(status["total_vram_bytes"] / (1024 ** 3), 2)
        vram_desc = f" ({total_gb} GB VRAM)"
    status["diagnostic_message"] = f"Detected {count} GPU device(s). Primary: {status['gpu_name']}{vram_desc}."

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
        lines.append(f"CUDA Build Version: {status['cuda_build_version']}")
        lines.append(f"CUDA Available    : {status['cuda_available']}")
        lines.append(f"GPU Available     : {status['gpu_available']}")
        lines.append(f"Device Type       : {status['device_type']}")
        lines.append(f"Detection Status  : {status['detection_status']}")
        if status["gpu_available"]:
            lines.append(f"CUDA Device Count : {status['cuda_device_count']}")
            for idx, dev_name in enumerate(status["cuda_device_names"]):
                lines.append(f"  - Device [{idx}]   : {dev_name}")
            if status["total_vram_bytes"] is not None:
                tot_gb = round(status["total_vram_bytes"] / (1024 ** 3), 2)
                lines.append(f"Total VRAM        : {tot_gb} GB ({status['total_vram_bytes']} bytes)")
            if status["free_vram_bytes"] is not None:
                free_gb = round(status["free_vram_bytes"] / (1024 ** 3), 2)
                lines.append(f"Free VRAM         : {free_gb} GB ({status['free_vram_bytes']} bytes)")
        if status["diagnostic_message"]:
            lines.append(f"Diagnostic Message: {status['diagnostic_message']}")
    else:
        lines.append("PyTorch Version   : N/A")
        lines.append("CUDA Available    : N/A")
        lines.append("GPU Available     : False")
        lines.append("Device Type       : cpu")
        if status["diagnostic_message"]:
            lines.append(f"Diagnostic Message: {status['diagnostic_message']}")

    lines.append("==================================================")
    return "\n".join(lines)
