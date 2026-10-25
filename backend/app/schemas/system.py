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


class SystemStatusResponse(BaseModel):
    """Detailed hardware and platform diagnostic status."""
    python_version: str = Field(description="Installed Python version")
    os_platform: str = Field(description="Operating system description")
    pytorch_available: bool = Field(description="Whether PyTorch is importable in the runtime")
    pytorch_version: Optional[str] = Field(default=None, description="PyTorch version string if available")
    cuda_available: bool = Field(description="Whether CUDA acceleration is enabled")
    cuda_version: Optional[str] = Field(default=None, description="CUDA runtime version if available")
    gpu_count: int = Field(default=0, description="Number of detected GPU devices")
    gpu_devices: List[GpuDeviceInfo] = Field(default_factory=list, description="List of detected GPU devices")
    primary_gpu_name: Optional[str] = Field(default=None, description="Name of primary GPU")
    primary_gpu_vram_gb: Optional[float] = Field(default=None, description="Primary GPU VRAM capacity in GB")
    environment_type: str = Field(description="Execution environment: 'cuda_accelerated', 'cpu_only', or 'no_pytorch'")
