"""
Pytest Fixtures for Phase 0 Project Foundation.
"""

import tempfile
from pathlib import Path
import pytest
import yaml

from core.config import ProjectConfig


@pytest.fixture
def sample_config_data():
    """Provides a sample ProjectConfig dictionary."""
    return {
        "project_name": "Adaptive LLM Test Framework",
        "version": "0.1.0-test",
        "seed": 12345,
        "device": "cpu",
    }


@pytest.fixture
def temp_yaml_config(sample_config_data):
    """Creates a temporary YAML file containing ProjectConfig data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(sample_config_data, tmp)
        tmp_path = tmp.name

    yield tmp_path

    path = Path(tmp_path)
    if path.exists():
        path.unlink()


@pytest.fixture
def temp_dir():
    """Provides a temporary directory path."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)
