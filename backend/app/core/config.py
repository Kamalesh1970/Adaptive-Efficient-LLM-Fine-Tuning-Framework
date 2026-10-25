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


settings = AppSettings()
