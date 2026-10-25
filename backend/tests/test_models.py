"""
Tests for Model Catalog, Model Lookup, and Compatibility Validation.
Verifies all Phase 1 requirements: catalog structure, compatibility rules,
gated credential security, error handling, and secret isolation.
"""

import os
import pytest
from fastapi.testclient import TestClient
from backend.app.schemas.models import (
    ModelMetadata,
    ModelVerificationStatus,
    CompatibilityStatus,
)
from backend.app.services.model_catalog import (
    ModelCatalogService,
    get_model_catalog_service,
)
from backend.app.schemas.training import TrainingMethod


def test_model_catalog_endpoint(client: TestClient):
    """Test GET /api/models returns registered model catalog with total count."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "total" in data
    assert data["total"] == len(data["models"])
    assert data["total"] >= 5

    # Verify key representative models exist
    model_ids = [m["id"] for m in data["models"]]
    assert "meta-llama/Meta-Llama-3-8B" in model_ids
    assert "mistralai/Mistral-7B-v0.3" in model_ids
    assert "microsoft/Phi-3-mini-4k-instruct" in model_ids
    assert "Qwen/Qwen2.5-0.5B" in model_ids  # Limited-VRAM candidate
    assert "google/gemma-2-2b" in model_ids  # Restricted method candidate


def test_known_model_lookup(client: TestClient):
    """Test GET /api/models/{model_id} returns exact validated metadata."""
    model_id = "meta-llama/Meta-Llama-3-8B"
    response = client.get(f"/api/models/{model_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == model_id
    assert data["name"] == "Meta Llama 3 8B"
    assert data["family"] == "llama"
    assert data["context_length"] == 8192
    assert data["is_gated"] is True
    assert data["is_downloaded"] is False
    assert data["is_tested"] is False
    assert "lora" in data["supported_methods"]
    assert "qlora" in data["supported_methods"]
    assert "full_finetuning" in data["supported_methods"]


def test_unknown_model_lookup(client: TestClient):
    """Test GET /api/models/{model_id} returns 404 for unrecognized model."""
    response = client.get("/api/models/nonexistent-org/fake-model-v9")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


def test_empty_model_identifier(client: TestClient):
    """Test lookup and validation fail clearly on empty model identifier."""
    # Empty GET identifier (whitespace)
    response_get = client.get("/api/models/%20")
    assert response_get.status_code in (400, 404)

    # Empty model_id in validate endpoint
    response_validate = client.post(
        "/api/models/validate",
        json={"model_id": "", "training_method": "lora"},
    )
    assert response_validate.status_code == 200
    data = response_validate.json()
    assert data["is_valid"] is False
    assert any("empty" in err.lower() for err in data["errors"])

    # Whitespace only model_id
    response_ws = client.post(
        "/api/models/validate",
        json={"model_id": "   ", "training_method": "lora"},
    )
    assert response_ws.status_code == 200
    assert response_ws.json()["is_valid"] is False


def test_valid_selection(client: TestClient):
    """Test validation succeeds for valid model and supported fine-tuning method."""
    payload = {
        "model_id": "mistralai/Mistral-7B-v0.3",
        "training_method": "lora",
        "context_length": 4096,
    }
    response = client.post("/api/models/validate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is True
    assert data["compatibility_status"] == "compatible"
    assert len(data["errors"]) == 0
    assert data["model"] is not None
    assert data["model"]["id"] == "mistralai/Mistral-7B-v0.3"


def test_invalid_training_method(client: TestClient):
    """Test validation reports error on invalid training method."""
    payload = {
        "model_id": "mistralai/Mistral-7B-v0.3",
        "training_method": "hyper_quantum_tuning",
    }
    response = client.post("/api/models/validate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is False
    assert any("invalid training method" in err.lower() for err in data["errors"])


def test_unsupported_model_method_combination(client: TestClient):
    """Test validation flags unsupported fine-tuning method on model with restricted compatibility."""
    # google/gemma-2-2b only supports lora and qlora in catalog, not full_finetuning
    payload = {
        "model_id": "google/gemma-2-2b",
        "training_method": "full_finetuning",
        "hf_token": "mock-token-for-test",
    }
    response = client.post("/api/models/validate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is False
    assert data["compatibility_status"] == "incompatible"
    assert any("does not support" in err.lower() for err in data["errors"])


def test_unknown_compatibility(client: TestClient):
    """Test validation for model with unverified status yields unknown status."""
    payload = {
        "model_id": "custom/unverified-experimental-lm",
        "training_method": "qlora",  # Only lora is listed, qlora is unverified
    }
    response = client.post("/api/models/validate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is False
    assert data["compatibility_status"] == "unknown"
    assert any("unverified" in err.lower() for err in data["errors"])


def test_gated_model_requires_credentials(client: TestClient, monkeypatch):
    """Test gated model without token fails validation with requires_credentials."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    payload = {
        "model_id": "meta-llama/Meta-Llama-3-8B",
        "training_method": "lora",
    }
    response = client.post("/api/models/validate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is False
    assert data["compatibility_status"] == "requires_credentials"
    assert any("gated" in err.lower() or "token" in err.lower() for err in data["errors"])


def test_gated_model_with_supplied_token_succeeds_unverified(client: TestClient):
    """Test gated model with supplied token passes validation and marks token unverified."""
    payload = {
        "model_id": "meta-llama/Meta-Llama-3-8B",
        "training_method": "lora",
        "hf_token": "hf_dummy_secret_token_123456789",
    }
    response = client.post("/api/models/validate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is True
    assert data["compatibility_status"] == "compatible"
    assert data["token_status"] == "configured_unverified"
    assert any("unverified" in w.lower() for w in data["warnings"])


def test_no_secret_leakage(client: TestClient):
    """Critical security test: Secret token must NEVER appear in response body, headers, or details."""
    secret_token = "hf_SUPER_SECRET_TOKEN_DO_NOT_LEAK_99999"
    payload = {
        "model_id": "meta-llama/Meta-Llama-3-8B",
        "training_method": "lora",
        "hf_token": secret_token,
    }
    response = client.post("/api/models/validate", json=payload)
    response_text = response.text

    assert secret_token not in response_text
    assert secret_token not in str(response.headers)
    assert "hf_token" not in response.json()


def test_catalog_entries_with_unknown_metadata(client: TestClient):
    """Test catalog models with unverified / unknown metadata are properly represented."""
    response = client.get("/api/models/custom/unverified-experimental-lm")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "custom/unverified-experimental-lm"
    assert data["family"] is None
    assert data["parameter_count"] is None
    assert data["context_length"] is None
    assert data["verification_status"] == "unverified"
    assert data["is_downloaded"] is False
    assert data["is_tested"] is False


def test_malformed_requests(client: TestClient):
    """Test malformed request handling."""
    # Non-JSON body
    response = client.post(
        "/api/models/validate",
        content="malformed body not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code in (400, 422)

    # Empty body
    response_empty = client.post(
        "/api/models/validate",
        json={},
    )
    assert response_empty.status_code == 200
    assert response_empty.json()["is_valid"] is False

    # Invalid context_length type
    response_type = client.post(
        "/api/models/validate",
        json={
            "model_id": "mistralai/Mistral-7B-v0.3",
            "training_method": "lora",
            "context_length": "invalid_integer_string",
        },
    )
    assert response_type.status_code == 200
    assert response_type.json()["is_valid"] is False
    assert any("context_length" in err.lower() for err in response_type.json()["errors"])


def test_context_length_warning_when_exceeding_verified_max(client: TestClient):
    """Test warning is issued when requested context exceeds model verified limit."""
    # Phi-3 mini verified context is 4096
    payload = {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "training_method": "lora",
        "context_length": 8192,
    }
    response = client.post("/api/models/validate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is True
    assert any("exceeds" in w.lower() for w in data["warnings"])


def test_configurable_catalog_service():
    """Test that ModelCatalogService is modular and can dynamically register new models."""
    custom_service = ModelCatalogService(initial_models=[])
    assert len(custom_service.get_all_models()) == 0

    new_model = ModelMetadata(
        id="community/tiny-lm-10m",
        name="Tiny LM 10M",
        family="tiny",
        source="huggingface",
        task_type="causal_lm",
        parameter_count="10M",
        context_length=1024,
        is_gated=False,
        verification_status=ModelVerificationStatus.CATALOG_REGISTERED,
        supported_methods=[TrainingMethod.LORA],
    )
    custom_service.register_model(new_model)
    assert custom_service.has_model("community/tiny-lm-10m")
    assert custom_service.get_model("community/tiny-lm-10m") == new_model
