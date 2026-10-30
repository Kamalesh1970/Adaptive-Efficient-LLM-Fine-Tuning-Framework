#!/usr/bin/env python3
"""
Hardware and CUDA Diagnostic Verification Script
Stage 2: Comprehensive PyTorch & CUDA Hardware Probe.
"""

import sys
import json

def run_cuda_diagnostic() -> dict:
    result = {
        "pytorch_version": None,
        "cuda_build_version": None,
        "cuda_available": False,
        "gpu_count": 0,
        "devices": [],
        "tensor_allocation_success": False,
        "tensor_operation_success": False,
        "diagnostic_message": "",
        "error": None,
    }

    try:
        import torch
    except ImportError as e:
        result["error"] = f"Failed to import torch: {e}"
        result["diagnostic_message"] = "PyTorch is not installed in the active environment."
        return result

    result["pytorch_version"] = torch.__version__
    result["cuda_build_version"] = torch.version.cuda

    try:
        cuda_avail = bool(torch.cuda.is_available())
        result["cuda_available"] = cuda_avail
    except Exception as e:
        result["error"] = str(e)
        result["diagnostic_message"] = f"Exception checking torch.cuda.is_available(): {e}"
        return result

    if not cuda_avail:
        result["diagnostic_message"] = (
            f"PyTorch built with CUDA {torch.version.cuda}, but CUDA runtime is not accessible. "
            "Host GPU communication or driver initialization is blocked."
        )
        return result

    try:
        count = torch.cuda.device_count()
        result["gpu_count"] = count
        for idx in range(count):
            name = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            free_b, tot_b = torch.cuda.mem_get_info(idx)
            result["devices"].append({
                "index": idx,
                "name": name,
                "total_memory_bytes": tot_b,
                "free_memory_bytes": free_b,
                "total_memory_gb": round(tot_b / (1024**3), 2),
                "free_memory_gb": round(free_b / (1024**3), 2),
                "compute_capability": f"{props.major}.{props.minor}",
            })
    except Exception as e:
        result["error"] = str(e)
        result["diagnostic_message"] = f"Failed to enumerate CUDA devices: {e}"
        return result

    # Perform tensor allocation & computation test
    try:
        t1 = torch.ones((128, 128), device="cuda", dtype=torch.float32)
        result["tensor_allocation_success"] = True
        t2 = t1 @ t1
        expected = float(128.0)
        actual = float(t2[0, 0].item())
        if abs(actual - expected) < 1e-4:
            result["tensor_operation_success"] = True
            result["diagnostic_message"] = (
                f"Successfully detected {count} CUDA device(s) and executed matrix multiplication on GPU 0 ({result['devices'][0]['name']})."
            )
        else:
            result["diagnostic_message"] = f"Tensor computation discrepancy: expected {expected}, got {actual}"
    except Exception as e:
        result["error"] = str(e)
        result["diagnostic_message"] = f"CUDA tensor operation failed: {e}"

    return result

if __name__ == "__main__":
    report = run_cuda_diagnostic()
    print("=" * 60)
    print("PHASE 12 CUDA & HARDWARE DIAGNOSTIC REPORT")
    print("=" * 60)
    print(f"PyTorch Version:         {report['pytorch_version']}")
    print(f"CUDA Build Version:      {report['cuda_build_version']}")
    print(f"CUDA Available:          {report['cuda_available']}")
    print(f"GPU Count:               {report['gpu_count']}")
    print(f"Tensor Allocation:       {'PASSED' if report['tensor_allocation_success'] else 'FAILED/BLOCKED'}")
    print(f"Tensor Operation:        {'PASSED' if report['tensor_operation_success'] else 'FAILED/BLOCKED'}")
    print(f"Diagnostic Message:      {report['diagnostic_message']}")
    if report["devices"]:
        for dev in report["devices"]:
            print(f" - GPU {dev['index']}: {dev['name']} ({dev['total_memory_gb']} GB total, {dev['free_memory_gb']} GB free, CC {dev['compute_capability']})")
    if report["error"]:
        print(f"Error Details:           {report['error']}")
    print("=" * 60)
    print(json.dumps(report, indent=2))
