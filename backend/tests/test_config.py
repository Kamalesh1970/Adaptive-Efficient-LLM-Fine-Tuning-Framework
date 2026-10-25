"""
Tests for training configuration defaults, validation logic, and schema edge cases.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.schemas.training import TrainingConfigSchema, TrainingMethod, TrainingLengthType


def test_get_training_config_defaults(client: TestClient):
    """Test /api/training/config/defaults returns valid unselected initial state."""
    response = client.get("/api/training/config/defaults")
    assert response.status_code == 200
    data = response.json()
    assert data["project_name"] == "my-finetuned-model"
    assert data["model_identifier"] is None  # Must show no model selected
    assert data["dataset_reference"] is None  # Must show no dataset selected
    assert data["training_method"] == "lora"
    assert data["training_length"] == 100
    assert data["training_length_type"] == "steps"
    assert data["batch_size"] == 2
    assert data["context_length"] == 2048
    assert data["learning_rate"] == 0.0002
    assert data["advanced_enabled"] is False


def test_validate_default_config_endpoint(client: TestClient):
    """Validating the initial defaults should be valid schema, but not ready for training."""
    defaults = client.get("/api/training/config/defaults").json()
    response = client.post("/api/training/config/validate", json=defaults)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is True
    assert data["is_ready_for_training"] is False
    assert any("model" in w.lower() for w in data["warnings"])
    assert any("dataset" in w.lower() for w in data["warnings"])


def test_validate_configured_model_and_dataset(client: TestClient):
    """When model and dataset are chosen, schema is valid, and Phase 0 notice is attached."""
    payload = {
        "project_name": "custom-llama3-instruct",
        "model_identifier": "meta-llama/Meta-Llama-3-8B",
        "training_method": "qlora",
        "dataset_reference": "yahma/alpaca-cleaned",
        "training_length_type": "steps",
        "training_length": 200,
        "batch_size": 4,
        "context_length": 4096,
        "learning_rate": 0.0001,
        "advanced_enabled": False,
    }
    response = client.post("/api/training/config/validate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is True
    # In Phase 0, is_ready_for_training remains False to prevent claiming execution is ready
    assert data["is_ready_for_training"] is False
    assert any("phase 0" in w.lower() for w in data["warnings"])


def test_validate_invalid_empty_project_name(client: TestClient):
    """Test validation fails when project name is whitespace or empty."""
    payload = {"project_name": "   "}
    response = client.post("/api/training/config/validate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is False
    assert any("project_name" in err.lower() for err in data["errors"])


def test_validate_invalid_learning_rate(client: TestClient):
    """Test validation fails for non-positive or excessive learning rate."""
    # Zero LR
    res1 = client.post("/api/training/config/validate", json={"learning_rate": 0.0})
    assert res1.json()["is_valid"] is False
    assert any("learning_rate" in err.lower() for err in res1.json()["errors"])

    # Negative LR
    res2 = client.post("/api/training/config/validate", json={"learning_rate": -0.001})
    assert res2.json()["is_valid"] is False
    assert any("learning_rate" in err.lower() for err in res2.json()["errors"])

    # Over 1.0
    res3 = client.post("/api/training/config/validate", json={"learning_rate": 2.5})
    assert res3.json()["is_valid"] is False
    assert any("learning_rate" in err.lower() for err in res3.json()["errors"])


def test_validate_invalid_context_length(client: TestClient):
    """Test validation fails if context length is less than 128 or exceeds 131,072."""
    res1 = client.post("/api/training/config/validate", json={"context_length": 64})
    assert res1.json()["is_valid"] is False
    assert any("context_length" in err.lower() for err in res1.json()["errors"])

    res2 = client.post("/api/training/config/validate", json={"context_length": 200000})
    assert res2.json()["is_valid"] is False


def test_validate_invalid_training_length(client: TestClient):
    """Test validation fails for non-positive training length."""
    res = client.post("/api/training/config/validate", json={"training_length": 0})
    assert res.json()["is_valid"] is False
    assert any("training_length" in err.lower() for err in res.json()["errors"])


def test_validate_invalid_training_method(client: TestClient):
    """Test validation fails for unrecognized training method."""
    res = client.post("/api/training/config/validate", json={"training_method": "super_tuning"})
    assert res.json()["is_valid"] is False
    assert any("training_method" in err.lower() for err in res.json()["errors"])


def test_schema_direct_instantiation_validation():
    """Test Pydantic model directly catches value errors."""
    with pytest.raises(ValueError):
        TrainingConfigSchema(project_name="")

    with pytest.raises(ValueError):
        TrainingConfigSchema(learning_rate=-0.05)

    with pytest.raises(ValueError):
        TrainingConfigSchema(batch_size=0)
