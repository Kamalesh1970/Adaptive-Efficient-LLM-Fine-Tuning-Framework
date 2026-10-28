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
    """Mocks an active CUDA environment with 1 GPU and 24GB VRAM."""
    mock_torch = MagicMock()
    mock_torch.__version__ = "2.3.0+cu121"
    mock_torch.version.cuda = "12.1"
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 1
    mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 4090"
    mock_torch.cuda.mem_get_info.return_value = (20 * (1024 ** 3), 24 * (1024 ** 3))

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
    mock_torch.version.cuda = None
    mock_torch.cuda.is_available.return_value = False
    mock_torch.cuda.device_count.return_value = 0

    monkeypatch.setattr("backend.app.core.system._safe_import_torch", lambda: mock_torch)
    return mock_torch


@pytest.fixture
def mock_no_pytorch_environment(monkeypatch):
    """Mocks an environment where PyTorch is not installed at all."""
    monkeypatch.setattr("backend.app.core.system._safe_import_torch", lambda: None)


@pytest.fixture
def mock_cuda_unavailable_environment(monkeypatch):
    """Mocks an environment with CUDA-compiled PyTorch but no CUDA runtime/driver."""
    mock_torch = MagicMock()
    mock_torch.__version__ = "2.3.0+cu121"
    mock_torch.version.cuda = "12.1"
    mock_torch.cuda.is_available.return_value = False
    mock_torch.cuda.device_count.return_value = 0

    monkeypatch.setattr("backend.app.core.system._safe_import_torch", lambda: mock_torch)
    return mock_torch


@pytest.fixture
def mock_cuda_init_failure_environment(monkeypatch):
    """Mocks CUDA driver initialization error without crashing the health endpoint."""
    mock_torch = MagicMock()
    mock_torch.__version__ = "2.3.0+cu121"
    mock_torch.version.cuda = "12.1"
    mock_torch.cuda.is_available.side_effect = RuntimeError("CUDA driver initialization failed, status 802")

    monkeypatch.setattr("backend.app.core.system._safe_import_torch", lambda: mock_torch)
    return mock_torch


@pytest.fixture
def mock_zero_gpus_environment(monkeypatch):
    """Mocks CUDA available but 0 visible devices."""
    mock_torch = MagicMock()
    mock_torch.__version__ = "2.3.0+cu121"
    mock_torch.version.cuda = "12.1"
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 0

    monkeypatch.setattr("backend.app.core.system._safe_import_torch", lambda: mock_torch)
    return mock_torch


@pytest.fixture
def mock_multiple_gpus_environment(monkeypatch):
    """Mocks 2 CUDA GPUs."""
    mock_torch = MagicMock()
    mock_torch.__version__ = "2.3.0+cu121"
    mock_torch.version.cuda = "12.1"
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 2

    def get_name(idx):
        return f"NVIDIA RTX A6000 Device {idx}"
    mock_torch.cuda.get_device_name.side_effect = get_name

    def get_props(idx):
        props = MagicMock()
        props.total_memory = 48 * (1024 ** 3)
        return props
    mock_torch.cuda.get_device_properties.side_effect = get_props

    mock_torch.cuda.mem_get_info.return_value = (40 * (1024 ** 3), 48 * (1024 ** 3))

    monkeypatch.setattr("backend.app.core.system._safe_import_torch", lambda: mock_torch)
    return mock_torch
