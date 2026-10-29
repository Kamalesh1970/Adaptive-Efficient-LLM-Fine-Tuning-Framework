"""
Core application settings for Adaptive LLM Fine-Tuning Studio Backend.
"""

from typing import List
from pydantic import BaseModel, Field


class AppSettings(BaseModel):
    """FastAPI application settings."""
    app_name: str = Field(
        default="Adaptive LLM Fine-Tuning Studio",
        description="Application display name"
    )
    version: str = Field(
        default="0.1.0-phase0",
        description="Application version"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    host: str = Field(
        default="0.0.0.0",
        description="Bind host"
    )
    port: int = Field(
        default=8000,
        description="Bind port"
    )
    cors_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "*"
        ],
        description="Allowed CORS origins"
    )
    dataset_upload_dir: str = Field(
        default="datasets/uploads",
        description="Directory to store uploaded datasets"
    )
    max_upload_size_bytes: int = Field(
        default=50 * 1024 * 1024,
        description="Maximum dataset upload size in bytes (50MB)"
    )
    runs_output_dir: str = Field(
        default="artifacts/runs",
        description="Directory to store training run artifacts"
    )
    exports_output_dir: str = Field(
        default="artifacts/exports",
        description="Directory to store exported adapter packages"
    )
    allow_remote_code: bool = Field(
        default=False,
        description="Whether to permit execution of remote custom model code (strictly False by default)"
    )


settings = AppSettings()

