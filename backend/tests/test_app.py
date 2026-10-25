"""
Tests for application startup and health endpoint.
"""

from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient):
    """Test root info endpoint returns 200 and online status."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "Adaptive LLM Fine-Tuning Studio" in data["app"]
    assert data["docs"] == "/docs"


def test_health_endpoint(client: TestClient):
    """Test /api/health endpoint structure and response."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app_name"] == "Adaptive LLM Fine-Tuning Studio"
    assert "version" in data
