"""
Comprehensive Pytest Test Suite for Phase 0 - Project Foundation.
"""

import os
import sys
import random
import logging
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.exceptions import (
    ProjectError,
    ConfigurationError,
    ValidationError,
    EnvironmentError,
)
from core.logger import get_logger
from core.config import ProjectConfig
from core.seed import set_seed
from core.system_status import get_system_status, format_system_status
from core.cli import main as cli_main


# 1. Project Structure Tests
def test_project_directories_exist():
    """Verify all required Phase 0 project directories exist."""
    base_dir = Path(__file__).resolve().parent.parent
    required_dirs = [
        "core",
        "core/lora",
        "core/qlora",
        "core/quantization",
        "core/training",
        "intelligence",
        "intelligence/hardware",
        "intelligence/memory",
        "intelligence/dataset",
        "intelligence/configuration",
        "evaluation",
        "evaluation/metrics",
        "evaluation/benchmarks",
        "evaluation/experiments",
        "models",
        "datasets",
        "dashboard",
        "deployment",
        "configs",
        "tests",
        "docs",
        "notebooks",
    ]
    for d in required_dirs:
        dir_path = base_dir / d
        assert dir_path.is_dir(), f"Required directory missing: {d}"


def test_required_files_exist():
    """Verify all required root files and .gitkeep placeholders exist."""
    base_dir = Path(__file__).resolve().parent.parent
    required_files = [
        "pyproject.toml",
        "requirements.txt",
        "README.md",
        ".gitignore",
        ".env.example",
        "models/.gitkeep",
        "datasets/.gitkeep",
        "dashboard/.gitkeep",
        "deployment/.gitkeep",
        "configs/.gitkeep",
        "docs/.gitkeep",
        "notebooks/.gitkeep",
    ]
    for f in required_files:
        file_path = base_dir / f
        assert file_path.is_file(), f"Required file missing: {f}"


# 2. Configuration System Tests
def test_default_config():
    """Verify default ProjectConfig initialization."""
    cfg = ProjectConfig()
    assert cfg.project_name == "Adaptive Efficient LLM Fine-Tuning Framework"
    assert cfg.version == "0.1.0"
    assert cfg.seed == 42
    assert cfg.device == "auto"


def test_config_from_yaml(temp_yaml_config):
    """Verify loading ProjectConfig from YAML."""
    cfg = ProjectConfig.from_yaml(temp_yaml_config)
    assert cfg.project_name == "Adaptive LLM Test Framework"
    assert cfg.version == "0.1.0-test"
    assert cfg.seed == 12345
    assert cfg.device == "cpu"


def test_config_missing_yaml():
    """Verify ConfigurationError raised when loading from non-existent file."""
    with pytest.raises(ConfigurationError) as exc_info:
        ProjectConfig.from_yaml("non_existent_file.yaml")
    assert "Configuration file not found" in str(exc_info.value)


def test_config_validation_error():
    """Verify ValidationError raised on invalid config parameter."""
    invalid_data = {"seed": -10}  # seed must be >= 0
    with pytest.raises(ValidationError) as exc_info:
        ProjectConfig.from_dict(invalid_data)
    assert "Configuration validation error" in str(exc_info.value)


def test_config_env_overrides(sample_config_data):
    """Verify environment variable overrides on ProjectConfig."""
    os.environ["SEED"] = "9999"
    os.environ["DEVICE"] = "cuda"

    try:
        cfg = ProjectConfig.from_dict(sample_config_data)
        assert cfg.seed == 9999
        assert cfg.device == "cuda"
    finally:
        os.environ.pop("SEED", None)
        os.environ.pop("DEVICE", None)


# 3. Logging Tests
def test_logger_initialization():
    """Verify logger creation and log level setting."""
    logger = get_logger("test_module", level="DEBUG")
    assert logger.name == "test_module"
    assert logger.level == logging.DEBUG


def test_logger_file_output(temp_dir):
    """Verify logger writes output to a specified log file."""
    log_file = temp_dir / "app.log"
    logger = get_logger("file_logger", level="INFO", log_file=str(log_file))
    logger.info("Phase 0 logger verification message")

    assert log_file.exists()
    assert "Phase 0 logger verification message" in log_file.read_text(encoding="utf-8")


# 4. Custom Exceptions Tests
def test_exception_hierarchy():
    """Verify custom exception inheritance."""
    assert issubclass(ConfigurationError, ProjectError)
    assert issubclass(ValidationError, ProjectError)
    assert issubclass(EnvironmentError, ProjectError)

    with pytest.raises(ProjectError):
        raise ConfigurationError("Config exception")

    with pytest.raises(ProjectError):
        raise ValidationError("Validation exception")

    with pytest.raises(ProjectError):
        raise EnvironmentError("Environment exception")


# 5. Reproducibility Tests
def test_set_seed_reproducibility():
    """Verify set_seed produces deterministic random numbers."""
    set_seed(42)
    val1 = random.randint(1, 100000)

    set_seed(42)
    val2 = random.randint(1, 100000)

    assert val1 == val2


def test_set_seed_status_dict():
    """Verify status dictionary returned by set_seed."""
    status = set_seed(100)
    assert status["seed"] == 100
    assert status["python_random"] is True


# 6. System Status Tests
def test_system_status_detection():
    """Verify get_system_status returns complete environment dictionary."""
    status = get_system_status()
    assert "python_version" in status
    assert "os_platform" in status
    assert "pytorch_available" in status
    assert "cuda_available" in status


def test_format_system_status_string():
    """Verify format_system_status output header and content."""
    formatted = format_system_status()
    assert "Adaptive Efficient LLM Fine-Tuning Framework Status" in formatted
    assert "Python Version" in formatted


def test_system_status_mocked_cuda():
    """Verify system status formatting when CUDA is mocked as available."""
    mock_torch = MagicMock()
    mock_torch.__version__ = "2.1.0-mock"
    mock_torch.cuda.is_available.return_value = True
    mock_torch.version.cuda = "12.1"
    mock_torch.cuda.device_count.return_value = 2
    mock_torch.cuda.get_device_name.side_effect = ["NVIDIA RTX 4090", "NVIDIA RTX 4090"]

    with patch.dict("sys.modules", {"torch": mock_torch}):
        status = get_system_status()
        assert status["pytorch_available"] is True
        assert status["cuda_available"] is True
        assert status["cuda_device_count"] == 2
        assert status["cuda_device_names"] == ["NVIDIA RTX 4090", "NVIDIA RTX 4090"]


# 7. CLI Entry Point Tests
def test_cli_execution(capsys):
    """Verify CLI main() outputs formatted status string to stdout."""
    cli_main()
    captured = capsys.readouterr()
    assert "Adaptive Efficient LLM Fine-Tuning Framework" in captured.out
    assert "Python Version" in captured.out
