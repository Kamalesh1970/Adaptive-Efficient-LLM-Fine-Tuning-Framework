"""
Multi-GPU and Distributed Capability Service (Phase 10: Multi-GPU).
Discovers PyTorch hardware devices, validates distributed configurations,
and enforces truthful capability reporting.
"""

import os
import logging
from typing import Optional, List, Dict, Any

from backend.app.schemas.distributed import (
    DistributedStrategy,
    GPUDeviceInfo,
    MultiGPUSystemReport,
    DistributedTrainingConfig,
)

logger = logging.getLogger(__name__)


class DistributedConfigurationError(Exception):
    """Raised when distributed configuration cannot be satisfied by hardware."""
    pass


class DistributedService:
    """
    Manages multi-GPU hardware discovery and strategy validation.
    """

    def discover_hardware(self) -> MultiGPUSystemReport:
        """
        Discovers actual visible GPU hardware via active PyTorch runtime.
        Never fabricates devices.
        """
        import torch

        cuda_ok = torch.cuda.is_available()
        count = torch.cuda.device_count() if cuda_ok else 0
        devices: List[GPUDeviceInfo] = []

        if cuda_ok and count > 0:
            for i in range(count):
                try:
                    name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)
                    mem_bytes = props.total_memory
                    mem_gb = round(mem_bytes / (1024 ** 3), 2)
                    devices.append(
                        GPUDeviceInfo(
                            index=i,
                            name=name,
                            total_memory_bytes=mem_bytes,
                            total_memory_gb=mem_gb,
                            is_available=True,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to query device {i}: {e}")

        supports_multi = count >= 2
        strategies = [DistributedStrategy.SINGLE_DEVICE]
        if supports_multi:
            strategies.extend([DistributedStrategy.DDP, DistributedStrategy.FSDP])

        unsupported_reason = None
        if not cuda_ok:
            unsupported_reason = "No CUDA hardware detected; operating in CPU mode."
        elif count == 1:
            unsupported_reason = "Single GPU detected (1 device). Multi-GPU strategies (DDP, FSDP, Tensor Parallelism > 1) require at least 2 visible GPUs."
        elif count == 0:
            unsupported_reason = "CUDA driver reports 0 visible devices."

        return MultiGPUSystemReport(
            cuda_available=cuda_ok,
            visible_device_count=count,
            devices=devices,
            supports_multi_gpu=supports_multi,
            supported_training_strategies=strategies,
            max_tensor_parallel_size=max(1, count),
            unsupported_reason=unsupported_reason,
        )

    def validate_distributed_training(self, config: DistributedTrainingConfig) -> None:
        """
        Validates distributed training configuration against actual visible hardware.
        Does NOT silently fall back to single device; raises actionable exception.
        """
        report = self.discover_hardware()

        if config.strategy != DistributedStrategy.SINGLE_DEVICE:
            if not report.cuda_available:
                raise DistributedConfigurationError(
                    f"Distributed training strategy '{config.strategy.value.upper()}' requires CUDA hardware. "
                    "Current system has no active CUDA support."
                )

            if report.visible_device_count < 2:
                raise DistributedConfigurationError(
                    f"Distributed training strategy '{config.strategy.value.upper()}' requires at least 2 GPUs, "
                    f"but only {report.visible_device_count} GPU(s) are visible to the process."
                )

            if config.requested_gpu_count > report.visible_device_count:
                raise DistributedConfigurationError(
                    f"Requested {config.requested_gpu_count} GPUs, but only {report.visible_device_count} are available."
                )

        elif config.requested_gpu_count > 1:
            raise DistributedConfigurationError(
                f"Requested {config.requested_gpu_count} GPUs but selected single_device strategy. "
                "Select 'ddp' or 'fsdp' for multi-GPU training."
            )


_distributed_service: Optional[DistributedService] = None


def get_distributed_service() -> DistributedService:
    global _distributed_service
    if _distributed_service is None:
        _distributed_service = DistributedService()
    return _distributed_service
