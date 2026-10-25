"""
Pytest configuration and fixtures for backend test suite.
"""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from backend.app.main import create_app


@pytest.fixture(scope="session")
def app():
    """Builds test FastAPI application instance."""
    return create_app()


@pytest.fixture(scope="session")
def client(app):
    """Provides FastAPI TestClient instance."""
    return TestClient(app)


@pytest.fixture
def mock_cuda_environment(monkeypatch):
    """Mocks an active CUDA environment with 1 GPU and 16GB VRAM."""
    mock_torch = MagicMock()
    mock_torch.__version__ = "2.3.0+cu121"
    mock_torch.version.cuda = "12.1"
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 1
    mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 4090"

    mock_props = MagicMock()
    mock_props.total_memory = 24 * (1024 ** 3)  # 24 GB
    mock_torch.cuda.get_device_properties.return_value = mock_props

    monkeypatch.setattr("backend.app.core.system._safe_import_torch", lambda: mock_torch)
    return mock_torch


@pytest.fixture
def mock_cpu_only_environment(monkeypatch):
    """Mocks a CPU-only PyTorch environment where CUDA is unavailable."""
    mock_torch = MagicMock()
    mock_torch.__version__ = "2.3.0+cpu"
    mock_torch.cuda.is_available.return_value = False
    mock_torch.cuda.device_count.return_value = 0

    monkeypatch.setattr("backend.app.core.system._safe_import_torch", lambda: mock_torch)
    return mock_torch


@pytest.fixture
def mock_no_pytorch_environment(monkeypatch):
    """Mocks an environment where PyTorch is not installed at all."""
    monkeypatch.setattr("backend.app.core.system._safe_import_torch", lambda: None)
