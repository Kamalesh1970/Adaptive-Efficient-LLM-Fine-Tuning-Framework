"""
Basic Configuration System for Adaptive LLM Fine-Tuning Framework (Phase 0).
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Union
import yaml
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

from core.exceptions import ConfigurationError, ValidationError
from core.logger import get_logger

logger = get_logger("config")


class ProjectConfig(BaseModel):
    """Phase 0 Base Project Configuration Schema."""
    project_name: str = Field(
        default="Adaptive Efficient LLM Fine-Tuning Framework",
        description="Name of the project framework"
    )
    version: str = Field(
        default="0.1.0",
        description="Project version"
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Global seed for reproducibility"
    )
    device: str = Field(
        default="auto",
        description="Target execution device ('auto', 'cuda', 'cpu')"
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectConfig":
        """Builds and validates ProjectConfig from a dictionary."""
        try:
            config = cls(**data)
        except PydanticValidationError as e:
            raise ValidationError(f"Configuration validation error: {e}") from e

        config._apply_env_overrides()
        return config

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "ProjectConfig":
        """Loads ProjectConfig from a YAML file."""
        path = Path(yaml_path)
        if not path.is_file():
            raise ConfigurationError(f"Configuration file not found: {yaml_path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_dict = yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigurationError(f"Failed to parse YAML configuration: {e}") from e

        return cls.from_dict(raw_dict)

    def _apply_env_overrides(self) -> None:
        """Applies environment variable overrides to configuration attributes."""
        env_map = {
            "PROJECT_NAME": ("project_name", str),
            "PROJECT_VERSION": ("version", str),
            "SEED": ("seed", int),
            "DEVICE": ("device", str),
        }

        for env_var, (attr, conv) in env_map.items():
            val = os.environ.get(env_var)
            if val is not None:
                try:
                    setattr(self, attr, conv(val))
                    logger.info(f"Overrode config.{attr} from environment variable {env_var}={val}")
                except Exception as e:
                    raise ValidationError(
                        f"Failed to convert environment variable {env_var} value '{val}' to {conv.__name__}: {e}"
                    ) from e

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Saves configuration instance to a YAML file."""
        path = Path(yaml_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
