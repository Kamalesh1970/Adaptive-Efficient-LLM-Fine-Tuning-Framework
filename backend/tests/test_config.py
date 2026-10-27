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


def test_validate_nan_and_infinity_rejection(client: TestClient):
    """Test validation strictly rejects NaN and infinite values."""
    # learning_rate NaN string
    res = client.post("/api/training/config/validate", json={"learning_rate": "nan"})
    assert res.json()["is_valid"] is False
    assert any("learning_rate" in err.lower() for err in res.json()["errors"])

    # learning_rate infinity
    res = client.post("/api/training/config/validate", json={"learning_rate": "inf"})
    assert res.json()["is_valid"] is False
    assert any("learning_rate" in err.lower() for err in res.json()["errors"])

    # batch_size NaN
    res = client.post("/api/training/config/validate", json={"batch_size": "nan"})
    assert res.json()["is_valid"] is False

    # Direct Pydantic model with float('nan')
    with pytest.raises(ValueError):
        TrainingConfigSchema(learning_rate=float("nan"))

    with pytest.raises(ValueError):
        TrainingConfigSchema(learning_rate=float("inf"))

    with pytest.raises(ValueError):
        TrainingConfigSchema(batch_size=float("nan"))


def test_validate_invalid_duration_mode(client: TestClient):
    """Test validation fails for invalid duration mode."""
    res = client.post("/api/training/config/validate", json={"training_duration_mode": "invalid_mode"})
    assert res.json()["is_valid"] is False


def test_lora_validation(client: TestClient):
    """Test LoRA validation: rank, alpha, dropout, target modules."""
    # Empty target modules for LoRA
    res = client.post(
        "/api/training/config/validate",
        json={"training_method": "lora", "lora_target_modules": []},
    )
    assert res.json()["is_valid"] is False

    # Rank 0
    res = client.post(
        "/api/training/config/validate",
        json={"training_method": "lora", "lora_rank": 0},
    )
    assert res.json()["is_valid"] is False

    # Alpha 0
    res = client.post(
        "/api/training/config/validate",
        json={"training_method": "lora", "lora_alpha": 0},
    )
    assert res.json()["is_valid"] is False

    # Dropout negative
    res = client.post(
        "/api/training/config/validate",
        json={"training_method": "lora", "lora_dropout": -0.1},
    )
    assert res.json()["is_valid"] is False

    # Dropout > 1.0
    res = client.post(
        "/api/training/config/validate",
        json={"training_method": "lora", "lora_dropout": 1.5},
    )
    assert res.json()["is_valid"] is False


def test_qlora_validation(client: TestClient):
    """Test QLoRA requires explicit quantization preference."""
    # QLoRA without quantization or none
    res = client.post(
        "/api/training/config/validate",
        json={"training_method": "qlora", "quantization_type": "none"},
    )
    assert res.json()["is_valid"] is False

    # QLoRA with unsupported quantization
    res = client.post(
        "/api/training/config/validate",
        json={"training_method": "qlora", "quantization_type": "2bit"},
    )
    assert res.json()["is_valid"] is False

    # QLoRA with valid nf4
    res = client.post(
        "/api/training/config/validate",
        json={"training_method": "qlora", "quantization_type": "nf4"},
    )
    assert res.json()["is_valid"] is True


def test_full_finetuning_validation(client: TestClient):
    """Test full fine-tuning does not require LoRA target modules."""
    res = client.post(
        "/api/training/config/validate",
        json={"training_method": "full_finetuning", "lora_target_modules": []},
    )
    assert res.json()["is_valid"] is True


def test_field_specific_validation_errors(client: TestClient):
    """Test field_errors maps invalid fields to their respective failure messages."""
    res = client.post(
        "/api/training/config/validate",
        json={
            "project_name": "",
            "learning_rate": -1.0,
            "context_length": 10,
        },
    )
    data = res.json()
    assert data["is_valid"] is False
    assert "field_errors" in data
    assert "project_name" in data["field_errors"]
    assert "learning_rate" in data["field_errors"]
    assert "context_length" in data["field_errors"]


def test_yaml_export_and_parse(client: TestClient):
    """Test exporting config to YAML and parsing it back."""
    defaults = client.get("/api/training/config/defaults").json()
    defaults["project_name"] = "yaml-test-run"
    defaults["learning_rate"] = 0.0003

    # Export
    export_res = client.post("/api/training/config/yaml/export", json=defaults)
    assert export_res.status_code == 200
    yaml_content = export_res.json()["yaml_content"]
    assert "project_name: yaml-test-run" in yaml_content
    assert "learning_rate: 0.0003" in yaml_content

    # Parse
    parse_res = client.post("/api/training/config/yaml/parse", json={"yaml_content": yaml_content})
    assert parse_res.status_code == 200
    parsed = parse_res.json()
    assert parsed["is_valid"] is True
    assert parsed["config"]["project_name"] == "yaml-test-run"
    assert parsed["config"]["learning_rate"] == 0.0003


def test_yaml_malformed_and_invalid_content(client: TestClient):
    """Test malformed YAML or non-mapping YAML is rejected safely."""
    # Malformed YAML
    bad_yaml = "project_name: [unclosed list"
    res1 = client.post("/api/training/config/yaml/parse", json={"yaml_content": bad_yaml})
    assert res1.json()["is_valid"] is False
    assert any("malformed" in err.lower() for err in res1.json()["errors"])

    # Non-mapping YAML (list instead of dict)
    list_yaml = "- item1\n- item2\n"
    res2 = client.post("/api/training/config/yaml/parse", json={"yaml_content": list_yaml})
    assert res2.json()["is_valid"] is False
    assert any("mapping" in err.lower() for err in res2.json()["errors"])

    # Invalid field values in YAML
    invalid_val_yaml = "project_name: test\nlearning_rate: -0.5\n"
    res3 = client.post("/api/training/config/yaml/parse", json={"yaml_content": invalid_val_yaml})
    assert res3.json()["is_valid"] is False
    assert any("learning_rate" in err.lower() for err in res3.json()["errors"])


def test_secret_exclusion_from_exports(client: TestClient):
    """Ensure Hugging Face tokens and secrets are scrubbed from exported YAML."""
    from backend.app.services.config_service import export_training_config_yaml

    config = TrainingConfigSchema(
        project_name="safe-export-test",
        learning_rate=0.0002,
    )
    yaml_str = export_training_config_yaml(config)
    assert "api_key" not in yaml_str.lower()
    assert "password" not in yaml_str.lower()


def test_reset_training_config_endpoint(client: TestClient):
    """Test /api/training/config/reset returns default baseline config."""
    res = client.post("/api/training/config/reset")
    assert res.status_code == 200
    data = res.json()
    assert data["project_name"] == "my-finetuned-model"
    assert data["model_identifier"] is None
    assert data["dataset_reference"] is None
    assert data["learning_rate"] == 0.0002


def test_stateless_isolated_configuration(client: TestClient):
    """Ensure validating one config does not mutate server state or leak to subsequent requests."""
    custom_payload = {
        "project_name": "isolated-test",
        "learning_rate": 0.0005,
    }
    client.post("/api/training/config/validate", json=custom_payload)

    # Defaults must remain untouched
    defaults = client.get("/api/training/config/defaults").json()
    assert defaults["project_name"] == "my-finetuned-model"
    assert defaults["learning_rate"] == 0.0002

