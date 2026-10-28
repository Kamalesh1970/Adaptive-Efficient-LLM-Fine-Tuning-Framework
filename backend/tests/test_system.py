"""
Tests for system status endpoint and hardware detection under varied environments.
"""

from fastapi.testclient import TestClient


def test_system_status_real_environment(client: TestClient):
    """Test /api/system/status on current runtime environment detecting real hardware."""
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert "python_version" in data
    assert "os_platform" in data
    assert "pytorch_available" in data
    assert "cuda_available" in data
    assert "gpu_available" in data
    assert "device_type" in data
    assert "environment_type" in data
    assert isinstance(data["gpu_devices"], list)

    # In this environment with real NVIDIA RTX 2050
    if data["cuda_available"]:
        assert data["gpu_available"] is True
        assert data["device_type"] == "cuda"
        assert data["gpu_count"] >= 1
        assert "GeForce" in data["gpu_name"] or "NVIDIA" in data["gpu_name"]
        assert data["detection_status"] == "gpu_detected"
        assert data["total_vram_bytes"] is not None and data["total_vram_bytes"] > 0


def test_system_status_no_pytorch(client: TestClient, mock_no_pytorch_environment):
    """Test /api/system/status gracefully reports missing PyTorch."""
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["pytorch_available"] is False
    assert data["pytorch_version"] is None
    assert data["cuda_available"] is False
    assert data["gpu_available"] is False
    assert data["device_type"] == "cpu"
    assert data["gpu_count"] == 0
    assert data["detection_status"] == "no_pytorch"
    assert data["environment_type"] == "no_pytorch"
    assert "not installed" in data["diagnostic_message"]


def test_system_status_cpu_only_build(client: TestClient, mock_cpu_only_environment):
    """Test /api/system/status gracefully reports CPU-only build."""
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["pytorch_available"] is True
    assert data["pytorch_version"] == "2.3.0+cpu"
    assert data["cuda_build_version"] is None
    assert data["cuda_available"] is False
    assert data["gpu_available"] is False
    assert data["device_type"] == "cpu"
    assert data["gpu_count"] == 0
    assert data["detection_status"] == "cpu_only_build"
    assert data["environment_type"] == "cpu_only"
    assert "CPU-only" in data["diagnostic_message"]


def test_system_status_cuda_unavailable(client: TestClient, mock_cuda_unavailable_environment):
    """Test /api/system/status when PyTorch has CUDA build but runtime CUDA is unavailable."""
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["pytorch_available"] is True
    assert data["cuda_build_version"] == "12.1"
    assert data["cuda_available"] is False
    assert data["gpu_available"] is False
    assert data["device_type"] == "cpu"
    assert data["gpu_count"] == 0
    assert data["detection_status"] == "cuda_unavailable"
    assert "CUDA runtime is not available" in data["diagnostic_message"]


def test_system_status_cuda_initialization_failure(client: TestClient, mock_cuda_init_failure_environment):
    """Test /api/system/status handles CUDA driver/initialization exceptions gracefully."""
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["pytorch_available"] is True
    assert data["cuda_available"] is False
    assert data["gpu_available"] is False
    assert data["device_type"] == "cpu"
    assert data["gpu_count"] == 0
    assert data["detection_status"] == "cuda_initialization_error"
    assert "CUDA initialization failed" in data["diagnostic_message"]


def test_system_status_zero_visible_gpus(client: TestClient, mock_zero_gpus_environment):
    """Test /api/system/status when CUDA is available but zero devices are visible."""
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["pytorch_available"] is True
    assert data["cuda_available"] is True
    assert data["gpu_available"] is False
    assert data["device_type"] == "cpu"
    assert data["gpu_count"] == 0
    assert data["detection_status"] == "zero_gpus_detected"
    assert "zero GPU devices are visible" in data["diagnostic_message"]


def test_system_status_one_visible_gpu(client: TestClient, mock_cuda_environment):
    """Test /api/system/status correctly reports one detected CUDA GPU and VRAM metrics."""
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["pytorch_available"] is True
    assert data["cuda_available"] is True
    assert data["cuda_build_version"] == "12.1"
    assert data["cuda_version"] == "12.1"
    assert data["gpu_available"] is True
    assert data["device_type"] == "cuda"
    assert data["gpu_count"] == 1
    assert data["gpu_name"] == "NVIDIA GeForce RTX 4090"
    assert data["primary_gpu_name"] == "NVIDIA GeForce RTX 4090"
    assert data["primary_gpu_vram_gb"] == 24.0
    assert data["total_vram_bytes"] == 24 * (1024 ** 3)
    assert data["free_vram_bytes"] == 20 * (1024 ** 3)
    assert data["detection_status"] == "gpu_detected"
    assert data["environment_type"] == "cuda_accelerated"
    assert len(data["gpu_devices"]) == 1
    assert data["gpu_devices"][0]["name"] == "NVIDIA GeForce RTX 4090"
    assert data["gpu_devices"][0]["total_vram_gb"] == 24.0
    assert data["gpu_devices"][0]["total_vram_bytes"] == 24 * (1024 ** 3)
    assert data["gpu_devices"][0]["free_vram_bytes"] == 20 * (1024 ** 3)


def test_system_status_multiple_visible_gpus(client: TestClient, mock_multiple_gpus_environment):
    """Test /api/system/status correctly reports multiple detected CUDA GPUs."""
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["pytorch_available"] is True
    assert data["cuda_available"] is True
    assert data["gpu_available"] is True
    assert data["device_type"] == "cuda"
    assert data["gpu_count"] == 2
    assert data["gpu_name"] == "NVIDIA RTX A6000 Device 0"
    assert len(data["gpu_devices"]) == 2
    assert data["gpu_devices"][0]["name"] == "NVIDIA RTX A6000 Device 0"
    assert data["gpu_devices"][1]["name"] == "NVIDIA RTX A6000 Device 1"
    assert data["detection_status"] == "gpu_detected"
