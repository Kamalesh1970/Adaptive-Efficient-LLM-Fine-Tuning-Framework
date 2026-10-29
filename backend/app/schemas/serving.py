"""
vLLM Model Serving Configuration & Lifecycle Schemas (Phase 10: Model Serving).
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class ServingStatus(str, Enum):
    """Lifecycle states for model serving instances."""
    VALIDATING = "validating"
    PREPARING_MODEL = "preparing_model"
    STARTING_SERVER = "starting_server"
    READY = "ready"
    STOPPED = "stopped"
    FAILED = "failed"


class ServingBackend(str, Enum):
    """Supported inference serving engines."""
    VLLM = "vllm"
    LLAMA_CPP_SERVER = "llama_cpp_server"


class ServingConfig(BaseModel):
    """Typed configuration for model serving."""
    model_reference: str = Field(description="Model identifier or run artifact ID")
    backend: ServingBackend = Field(default=ServingBackend.VLLM, description="Inference engine backend")
    host: str = Field(default="127.0.0.1", description="Bind host address")
    port: int = Field(default=8080, ge=1024, le=65535, description="Bind port")
    max_model_len: Optional[int] = Field(default=2048, ge=128, le=32768, description="Maximum context length")
    gpu_memory_utilization: float = Field(default=0.9, ge=0.1, le=0.98, description="Fraction of GPU memory for weights/KVs")
    tensor_parallel_size: int = Field(default=1, ge=1, le=16, description="Tensor parallel degree (multi-GPU)")
    quantization: Optional[str] = Field(default=None, description="Quantization mode (e.g. 'awq', 'gptq', 'fp8')")
    max_num_seqs: int = Field(default=256, ge=1, le=1024, description="Max concurrent sequences")

    @field_validator("host")
    @classmethod
    def validate_safe_host(cls, v: str) -> str:
        # SSRF Protection: ensure host is safe for server binding
        allowed_hosts = {"127.0.0.1", "localhost", "0.0.0.0"}
        if v.strip() not in allowed_hosts and not v.strip().startswith("10.") and not v.strip().startswith("192.168."):
            raise ValueError(f"Host '{v}' is not permitted for internal serving binding.")
        return v.strip()


class ServingInstanceRecord(BaseModel):
    """Persistent serving instance record."""
    instance_id: str
    model_reference: str
    backend: ServingBackend
    status: ServingStatus
    host: str
    port: int
    config: ServingConfig
    pid: Optional[int] = None
    created_at: str
    ready_at: Optional[str] = None
    stopped_at: Optional[str] = None
    error_message: Optional[str] = None
    logs: List[str] = Field(default_factory=list)


class SmokeInferenceRequest(BaseModel):
    """Bounded smoke test inference request payload."""
    prompt: str = Field(max_length=2000, description="Inference prompt (max 2000 chars)")
    max_tokens: int = Field(default=64, ge=1, le=256, description="Max tokens to generate (bounded)")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")


class SmokeInferenceResponse(BaseModel):
    """Response from smoke test inference request."""
    instance_id: str
    prompt: str
    generated_text: str
    tokens_generated: int
    latency_ms: float
    model_identifier: str
