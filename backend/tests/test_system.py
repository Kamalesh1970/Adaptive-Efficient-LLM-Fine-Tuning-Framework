"""
Tests for system status endpoint and hardware detection under varied environments.
"""

from fastapi.testclient import TestClient


def test_system_status_real_environment(client: TestClient):
    """Test /api/system/status on current runtime environment without failure."""
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert "python_version" in data
    assert "os_platform" in data
    assert "pytorch_available" in data
    assert "cuda_available" in data
    assert "environment_type" in data
    assert isinstance(data["gpu_devices"], list)


def test_system_status_no_pytorch(client: TestClient, mock_no_pytorch_environment):
    """Test /api/system/status gracefully reports missing PyTorch."""
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["pytorch_available"] is False
    assert data["pytorch_version"] is None
    assert data["cuda_available"] is False
    assert data["gpu_count"] == 0
    assert data["environment_type"] == "no_pytorch"


def test_system_status_cpu_only(client: TestClient, mock_cpu_only_environment):
    """Test /api/system/status gracefully reports CPU-only mode."""
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["pytorch_available"] is True
    assert data["pytorch_version"] == "2.3.0+cpu"
    assert data["cuda_available"] is False
    assert data["gpu_count"] == 0
    assert data["environment_type"] == "cpu_only"


def test_system_status_mocked_cuda(client: TestClient, mock_cuda_environment):
    """Test /api/system/status correctly reports detected CUDA GPU and VRAM."""
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["pytorch_available"] is True
    assert data["cuda_available"] is True
    assert data["cuda_version"] == "12.1"
    assert data["gpu_count"] == 1
    assert data["primary_gpu_name"] == "NVIDIA GeForce RTX 4090"
    assert data["primary_gpu_vram_gb"] == 24.0
    assert data["environment_type"] == "cuda_accelerated"
    assert len(data["gpu_devices"]) == 1
    assert data["gpu_devices"][0]["name"] == "NVIDIA GeForce RTX 4090"
    assert data["gpu_devices"][0]["total_vram_gb"] == 24.0
