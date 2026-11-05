"""
Comprehensive Regression Tests:
Artifact Validation, Test Fixture Exclusion, Base Model Resolution, and Real QLoRA Evaluation.
"""

import os
import json
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.schemas.artifacts import ValidationStatus
from backend.app.services.artifact_service import (
    ArtifactService,
    is_test_fixture_artifact,
)
from backend.app.services.model_loader import is_model_cached_locally
from backend.app.services.run_repository import get_run_repository


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_is_test_fixture_classification():
    """Verifies that synthetic fixtures are correctly distinguished from production artifacts."""
    # Test fixture with test model identifier
    assert is_test_fixture_artifact(
        run_id="run_2f12c58d3f89",
        model_identifier="test/genuine-mini-lm",
        base_model_identifier="test/genuine-mini-lm",
        size_bytes=25,
        files=[{"filename": "adapter_model.safetensors", "size_bytes": 25}],
    ) is True

    # Test fixture with test run prefix
    assert is_test_fixture_artifact(
        run_id="run_dup_test_12345",
        model_identifier="Qwen/Qwen2.5-0.5B",
        base_model_identifier="Qwen/Qwen2.5-0.5B",
        size_bytes=48,
        files=[{"filename": "adapter_model.safetensors", "size_bytes": 48}],
    ) is True

    # Real trained QLoRA production artifact
    assert is_test_fixture_artifact(
        run_id="run_17f63adb9f6",
        model_identifier="Qwen/Qwen2.5-0.5B",
        base_model_identifier="Qwen/Qwen2.5-0.5B",
        size_bytes=35237104,
        files=[{"filename": "adapter_model.safetensors", "size_bytes": 35237104}],
    ) is False


def test_artifact_listing_with_exclude_test_fixtures(client: TestClient):
    """Verifies GET /api/artifacts?exclude_test_fixtures=true excludes fixtures."""
    res_all = client.get("/api/artifacts?exclude_test_fixtures=false")
    assert res_all.status_code == 200
    all_arts = res_all.json()
    assert len(all_arts) > 0

    res_filtered = client.get("/api/artifacts?exclude_test_fixtures=true")
    assert res_filtered.status_code == 200
    filtered_arts = res_filtered.json()

    # None of the filtered artifacts should be test fixtures
    for art in filtered_arts:
        assert art.get("is_test_fixture") is False
        assert not art["model_identifier"].startswith("test/")

    # The real trained run must be present in the filtered list
    target_ids = [a["artifact_id"] for a in filtered_arts]
    assert "art_run_17f63adb9f6_adapter" in target_ids or "art_run_17f63adbd9f6_adapter" in target_ids


def test_adapter_missing_base_model_safe_repair(tmp_path):
    """Missing base_model_name_or_path is repaired from training_config.json."""
    run_dir = tmp_path / "run_repair_test"
    run_dir.mkdir()

    # Write training_config.json with authoritative model
    with open(run_dir / "training_config.json", "w") as f:
        json.dump({"model_identifier": "test/authoritative-model"}, f)

    # Write adapter_config.json missing base_model_name_or_path
    with open(run_dir / "adapter_config.json", "w") as f:
        json.dump({"peft_type": "LORA", "r": 8}, f)

    with open(run_dir / "adapter_model.safetensors", "wb") as f:
        f.write(b"SAMPLE_ADAPTER_WEIGHT_DATA")

    art_svc = ArtifactService(runs_root=str(tmp_path))
    art_svc.discover_artifacts_for_run("run_repair_test")

    # Validate without load check
    res = art_svc.validate_artifact("art_run_repair_test_adapter", check_loadable=False)
    assert res.is_valid is True

    # Verify adapter_config.json was repaired on disk
    with open(run_dir / "adapter_config.json") as f:
        repaired_cfg = json.load(f)
    assert repaired_cfg["base_model_name_or_path"] == "test/authoritative-model"


def test_missing_base_model_fails_check_loadable(tmp_path):
    """When check_loadable=True and base model is not cached locally, validation fails truthfully."""
    run_dir = tmp_path / "run_uncached_test"
    run_dir.mkdir()

    with open(run_dir / "adapter_config.json", "w") as f:
        json.dump({"base_model_name_or_path": "nonexistent/uncached-remote-model"}, f)

    with open(run_dir / "adapter_model.safetensors", "wb") as f:
        f.write(b"SAMPLE_ADAPTER_WEIGHT_DATA")

    art_svc = ArtifactService(runs_root=str(tmp_path))
    art_svc.discover_artifacts_for_run("run_uncached_test")

    res = art_svc.validate_artifact("art_run_uncached_test_adapter", check_loadable=True)
    assert res.is_valid is False
    assert res.validation_status == ValidationStatus.INVALID
    assert res.is_loadable is False
    assert any("not found on local disk or in the local model cache" in err for err in res.errors)


def test_real_qlora_trained_artifact_integrity_and_evaluation():
    """
    Verifies the actual trained QLoRA artifact art_run_17f63adb9f6_adapter:
    - Base model: Qwen/Qwen2.5-0.5B
    - Size: > 33 MB
    - Local cache: Verified
    - Completed evaluation in repository with truthful metrics
    """
    art_svc = ArtifactService()
    art = art_svc.get_artifact("art_run_17f63adb9f6_adapter")
    assert art is not None, "art_run_17f63adb9f6_adapter must exist"
    assert art.base_model_identifier == "Qwen/Qwen2.5-0.5B"
    assert art.size_bytes > 30_000_000  # ~33.6 MB

    # Check local cache
    cached, _ = is_model_cached_locally("Qwen/Qwen2.5-0.5B")
    assert cached is True, "Base model Qwen/Qwen2.5-0.5B must be present in local cache"

    # Validation
    val_res = art_svc.validate_artifact("art_run_17f63adb9f6_adapter", check_loadable=False)
    assert val_res.is_valid is True
    assert val_res.validation_status == ValidationStatus.VALID

    # Check that live evaluation completed
    repo = get_run_repository()
    evals, _ = repo.list_evaluations(artifact_id="art_run_17f63adb9f6_adapter")
    assert len(evals) > 0, "A real evaluation for art_run_17f63adb9f6_adapter must exist"
    completed_eval = next((e for e in evals if e.status.value == "completed"), None)
    assert completed_eval is not None
    assert completed_eval.metrics.eval_loss is not None
    assert completed_eval.baseline_comparison is not None
    assert completed_eval.baseline_comparison.loss_delta < 0, "Fine-tuned model should show improvement over base"
