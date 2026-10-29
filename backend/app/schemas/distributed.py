"""
Multi-GPU and Distributed Execution Schemas for Phase 10.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class DistributedStrategy(str, Enum):
    """Supported distributed training strategies."""
    SINGLE_DEVICE = "single_device"
    DDP = "ddp"
    FSDP = "fsdp"


class GPUDeviceInfo(BaseModel):
    """Information about an individually detected GPU device."""
    index: int
    name: str
    total_memory_bytes: int
    total_memory_gb: float
    is_available: bool = True


class MultiGPUSystemReport(BaseModel):
    """Current hardware multi-GPU diagnostic status."""
    cuda_available: bool
    visible_device_count: int
    devices: List[GPUDeviceInfo] = Field(default_factory=list)
    supports_multi_gpu: bool
    supported_training_strategies: List[DistributedStrategy] = Field(default_factory=list)
    max_tensor_parallel_size: int
    unsupported_reason: Optional[str] = None


class DistributedTrainingConfig(BaseModel):
    """Configuration for distributed multi-GPU training."""
    strategy: DistributedStrategy = Field(
        default=DistributedStrategy.SINGLE_DEVICE,
        description="Distributed training strategy"
    )
    requested_gpu_count: int = Field(
        default=1,
        ge=1,
        le=64,
        description="Number of GPUs to allocate for distributed execution"
    )
    master_port: int = Field(
        default=29500,
        ge=1024,
        le=65535,
        description="Port for torch.distributed communication"
    )
    mixed_precision: str = Field(
        default="fp16",
        description="Precision format ('fp16', 'bf16', 'none')"
    )
    find_unused_parameters: bool = Field(
        default=False,
        description="DDP setting for unused parameter gradient sync"
    )
