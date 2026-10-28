"""
System and Health Schemas.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="ok", description="Health status string")
    app_name: str = Field(description="Name of the application")
    version: str = Field(description="Application version")


class GpuDeviceInfo(BaseModel):
    """GPU device descriptor."""
    index: int = Field(description="Device index")
    name: str = Field(description="Hardware device name")
    total_vram_gb: Optional[float] = Field(default=None, description="Total VRAM in gigabytes")
    total_vram_mb: Optional[float] = Field(default=None, description="Total VRAM in megabytes")
    total_vram_bytes: Optional[int] = Field(default=None, description="Total VRAM in bytes")
    free_vram_bytes: Optional[int] = Field(default=None, description="Free VRAM in bytes")


class SystemStatusResponse(BaseModel):
    """Detailed hardware and platform diagnostic status."""
    python_version: str = Field(description="Installed Python version")
    os_platform: str = Field(description="Operating system description")
    pytorch_available: bool = Field(description="Whether PyTorch is importable in the runtime")
    pytorch_version: Optional[str] = Field(default=None, description="PyTorch version string if available")
    cuda_build_version: Optional[str] = Field(default=None, description="CUDA build version against which PyTorch was compiled")
    cuda_available: bool = Field(description="Whether CUDA runtime acceleration is enabled")
    cuda_version: Optional[str] = Field(default=None, description="CUDA build or runtime version")
    gpu_available: bool = Field(default=False, description="Whether an active, usable GPU is detected")
    device_type: str = Field(default="cpu", description="Active device type: 'cuda' or 'cpu'")
    gpu_name: Optional[str] = Field(default=None, description="Name of the primary GPU device")
    gpu_count: int = Field(default=0, description="Number of detected GPU devices")
    gpu_devices: List[GpuDeviceInfo] = Field(default_factory=list, description="List of detected GPU devices")
    primary_gpu_name: Optional[str] = Field(default=None, description="Name of primary GPU")
    primary_gpu_vram_gb: Optional[float] = Field(default=None, description="Primary GPU VRAM capacity in GB")
    total_vram_bytes: Optional[int] = Field(default=None, description="Total VRAM of primary GPU in bytes")
    free_vram_bytes: Optional[int] = Field(default=None, description="Available free VRAM of primary GPU in bytes")
    detection_status: str = Field(default="unknown", description="Detection diagnostic status code")
    diagnostic_message: Optional[str] = Field(default=None, description="Human-readable diagnostic explanation")
    environment_type: str = Field(description="Execution environment: 'cuda_accelerated', 'cpu_only', or 'no_pytorch'")
