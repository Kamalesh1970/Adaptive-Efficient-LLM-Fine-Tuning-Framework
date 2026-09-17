"""
Unit Tests for Configuration & Exceptions Primitives (Phase 0).
"""

import os
import pytest
from core.exceptions import ProjectError, ConfigurationError, ValidationError
from core.config import ProjectConfig


def test_base_project_config():
    """Verify default ProjectConfig initialization."""
    cfg = ProjectConfig()
    assert cfg.project_name == "Adaptive Efficient LLM Fine-Tuning Framework"
    assert cfg.version == "0.1.0"
    assert cfg.seed == 42
    assert cfg.device == "auto"


def test_base_config_to_yaml(temp_dir):
    """Verify exporting ProjectConfig to YAML."""
    cfg = ProjectConfig(seed=123)
    yaml_path = temp_dir / "exported_project_config.yaml"
    cfg.to_yaml(yaml_path)

    assert yaml_path.exists()
    reloaded = ProjectConfig.from_yaml(yaml_path)
    assert reloaded.seed == 123
