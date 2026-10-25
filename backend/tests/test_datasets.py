"""
Tests for Phase 2 Dataset Upload, Schema Detection, Validation, and Deterministic Splits.
Verifies format parsing (JSON, JSONL, CSV), security sandboxing, path isolation,
error reporting with line numbers, and leak-free split generation.
"""

import os
import json
import pytest
from fastapi.testclient import TestClient
from backend.app.schemas.datasets import DatasetFormat, DatasetSchemaType
from backend.app.services.dataset_service import DatasetService


@pytest.fixture
def temp_dataset_service(tmp_path):
    """Fixture providing an isolated DatasetService with a temporary upload dir."""
    return DatasetService(upload_dir=str(tmp_path), max_upload_size_bytes=1024 * 1024)


def test_valid_json_upload(client: TestClient):
    """Test uploading a valid instruction JSON array."""
    data = [
        {"instruction": "What is Python?", "input": "", "output": "Python is a programming language."},
        {"instruction": "Calculate 2+2", "input": "", "output": "4"},
    ]
    json_bytes = json.dumps(data).encode("utf-8")
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("train.json", json_bytes, "application/json")},
    )
    assert response.status_code == 200
    report = response.json()
    assert report["is_valid"] is True
    assert report["file_format"] == "json"
    assert report["detected_schema"] == "instruction"
    assert report["valid_sample_count"] == 2
    assert report["total_samples"] == 2
    assert report["invalid_sample_count"] == 0
    assert "dataset_id" in report


def test_valid_json_wrapped_object(client: TestClient):
    """Test uploading a wrapped JSON object containing a 'data' array."""
    data = {
        "version": "1.0",
        "data": [
            {"text": "Sample pre-training corpus line 1."},
            {"text": "Sample pre-training corpus line 2."},
        ],
    }
    json_bytes = json.dumps(data).encode("utf-8")
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("corpus.json", json_bytes, "application/json")},
    )
    assert response.status_code == 200
    report = response.json()
    assert report["is_valid"] is True
    assert report["detected_schema"] == "text"
    assert report["valid_sample_count"] == 2


def test_valid_jsonl_upload(client: TestClient):
    """Test uploading a valid conversational JSONL file."""
    lines = [
        json.dumps({
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }),
        json.dumps({
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Explain AI"},
                {"role": "assistant", "content": "AI is artificial intelligence."},
            ]
        }),
    ]
    jsonl_bytes = "\n".join(lines).encode("utf-8")
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("chat_data.jsonl", jsonl_bytes, "application/x-jsonlines")},
    )
    assert response.status_code == 200
    report = response.json()
    assert report["is_valid"] is True
    assert report["file_format"] == "jsonl"
    assert report["detected_schema"] == "conversational"
    assert report["valid_sample_count"] == 2


def test_valid_csv_upload(client: TestClient):
    """Test uploading a valid CSV file with instruction and output headers."""
    csv_content = "instruction,input,output\nTranslate to French,Hello,Bonjour\nWrite a poem,,Roses are red"
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("prompts.csv", csv_content.encode("utf-8"), "text/csv")},
    )
    assert response.status_code == 200
    report = response.json()
    assert report["is_valid"] is True
    assert report["file_format"] == "csv"
    assert report["detected_schema"] == "instruction"
    assert report["valid_sample_count"] == 2


def test_unsupported_file_type(client: TestClient):
    """Test uploading an unsupported file type (.parquet or .txt) returns 400."""
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("dataset.parquet", b"PAR1fakecontent", "application/octet-stream")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "unsupported" in data["detail"].lower()
    assert ".parquet" in data["detail"]


def test_empty_file(client: TestClient):
    """Test uploading an empty file returns validation failure without crash."""
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("empty.json", b"", "application/json")},
    )
    assert response.status_code == 200
    report = response.json()
    assert report["is_valid"] is False
    assert report["total_samples"] == 0
    assert any("empty" in e["message"].lower() for e in report["errors"])


def test_malformed_json(client: TestClient):
    """Test uploading malformed JSON syntax."""
    bad_json = b"[{'instruction': 'single quotes are invalid json'}"
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("bad.json", bad_json, "application/json")},
    )
    assert response.status_code == 200
    report = response.json()
    assert report["is_valid"] is False
    assert any("malformed json" in e["message"].lower() for e in report["errors"])


def test_malformed_jsonl_with_line_number_reporting(client: TestClient):
    """Test JSONL syntax error accurately reports the malformed line number."""
    lines = [
        json.dumps({"instruction": "Task 1", "output": "Output 1"}),
        "{ broken json line }",
        json.dumps({"instruction": "Task 3", "output": "Output 3"}),
    ]
    content = "\n".join(lines).encode("utf-8")
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("mixed.jsonl", content, "application/x-jsonlines")},
    )
    assert response.status_code == 200
    report = response.json()
    assert report["is_valid"] is False
    assert report["invalid_sample_count"] >= 1
    # Check that error explicitly points to line 2
    line_errors = [e for e in report["errors"] if e["sample_index"] == 2]
    assert len(line_errors) >= 1
    assert "line 2" in line_errors[0]["message"].lower()


def test_malformed_csv(client: TestClient):
    """Test empty CSV without header or rows."""
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("empty.csv", b"   \n  \n", "text/csv")},
    )
    assert response.status_code == 200
    report = response.json()
    assert report["is_valid"] is False
    assert any("empty" in e["message"].lower() or "no data" in e["message"].lower() for e in report["errors"])


def test_missing_required_fields(client: TestClient):
    """Test sample missing required output/response field is flagged."""
    data = [
        {"instruction": "Only instruction, missing output"},
        {"instruction": "Valid", "output": "Valid output"},
    ]
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("missing_fields.json", json.dumps(data).encode("utf-8"), "application/json")},
    )
    assert response.status_code == 200
    report = response.json()
    assert report["invalid_sample_count"] >= 1
    assert any("output" in e["message"].lower() for e in report["errors"])


def test_invalid_conversational_messages(client: TestClient):
    """Test conversational format with invalid roles or missing assistant response."""
    # Missing assistant turn
    bad_conv = [
        {
            "messages": [
                {"role": "user", "content": "Question?"},
                {"role": "invalid_role_xyz", "content": "Invalid!"},
            ]
        }
    ]
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("bad_chat.json", json.dumps(bad_conv).encode("utf-8"), "application/json")},
    )
    assert response.status_code == 200
    report = response.json()
    assert report["is_valid"] is False
    assert any("role" in e["message"].lower() or "assistant" in e["message"].lower() for e in report["errors"])


def test_dataset_schema_detection():
    """Unit test for schema detection logic."""
    service = DatasetService()
    assert service._detect_dataset_schema([{"instruction": "a", "output": "b"}]) == DatasetSchemaType.INSTRUCTION
    assert service._detect_dataset_schema([{"text": "some content"}]) == DatasetSchemaType.TEXT
    assert service._detect_dataset_schema([{"messages": [{"role": "user", "content": "hi"}]}]) == DatasetSchemaType.CONVERSATIONAL
    assert service._detect_dataset_schema([{"random_col": 123}]) is None


def test_dataset_normalization():
    """Test normalization maps aliases and preserves fields."""
    service = DatasetService()
    # Instruction format with prompt alias
    raw_inst = {"prompt": "Solve math", "context": "1+1", "response": "2"}
    norm_inst = service._normalize_sample(raw_inst, DatasetSchemaType.INSTRUCTION)
    assert norm_inst["instruction"] == "Solve math"
    assert norm_inst["input"] == "1+1"
    assert norm_inst["output"] == "2"

    # Conversational format with human/gpt aliases
    raw_chat = {
        "messages": [
            {"from": "human", "value": "Hey"},
            {"from": "gpt", "value": "Hello"},
        ]
    }
    norm_chat = service._normalize_sample(raw_chat, DatasetSchemaType.CONVERSATIONAL)
    assert norm_chat["messages"][0]["role"] == "user"
    assert norm_chat["messages"][1]["role"] == "assistant"


def test_dataset_identifier_generation():
    """Test server-side dataset identifier generation."""
    service = DatasetService()
    id1 = service.generate_dataset_id()
    id2 = service.generate_dataset_id()
    assert id1.startswith("ds_")
    assert id2.startswith("ds_")
    assert id1 != id2


def test_file_size_limit(temp_dataset_service):
    """Test that file exceeding max upload size is rejected."""
    # Service configured with 100 bytes limit
    small_service = DatasetService(upload_dir=temp_dataset_service.upload_dir, max_upload_size_bytes=100)
    with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
        small_service.store_upload_bytes(b"A" * 200, "large.json")


def test_path_traversal_attempts(temp_dataset_service):
    """Test that path traversal patterns in filenames are safely stripped."""
    safe_name = temp_dataset_service.sanitize_filename("../../../etc/passwd.json")
    assert ".." not in safe_name
    assert "/" not in safe_name
    assert "\\" not in safe_name
    assert safe_name == "passwd.json"


def test_duplicate_filenames_no_overwrite(temp_dataset_service):
    """Test that uploading two files with identical filenames creates distinct files."""
    data = b'[{"instruction": "task", "output": "done"}]'
    ds_id1, name1, _ = temp_dataset_service.store_upload_bytes(data, "dataset.json")
    ds_id2, name2, _ = temp_dataset_service.store_upload_bytes(data, "dataset.json")

    assert ds_id1 != ds_id2
    assert temp_dataset_service._file_map[ds_id1] != temp_dataset_service._file_map[ds_id2]


def test_dataset_lookup_and_unknown_id(client: TestClient):
    """Test GET /api/datasets/{id} returns 200 for known, 404 for unknown."""
    # Upload first
    data = json.dumps([{"instruction": "hi", "output": "bye"}]).encode("utf-8")
    up_res = client.post("/api/datasets/upload", files={"file": ("data.json", data, "application/json")})
    dataset_id = up_res.json()["dataset_id"]

    # Known lookup
    res_known = client.get(f"/api/datasets/{dataset_id}")
    assert res_known.status_code == 200
    assert res_known.json()["dataset_id"] == dataset_id

    # Unknown lookup
    res_unknown = client.get("/api/datasets/ds_nonexistent99")
    assert res_unknown.status_code == 404
    assert "not found" in res_unknown.json()["detail"].lower()


def test_validation_report_correctness_and_duplicates(client: TestClient):
    """Test report contains exact sample counts, format, and duplicate tracking."""
    data = [
        {"instruction": "task 1", "output": "res 1"},
        {"instruction": "task 1", "output": "res 1"},  # duplicate
        {"instruction": "task 2", "output": "res 2"},
    ]
    res = client.post(
        "/api/datasets/upload",
        files={"file": ("dups.json", json.dumps(data).encode("utf-8"), "application/json")},
    )
    assert res.status_code == 200
    report = res.json()
    assert report["total_samples"] == 3
    assert report["valid_sample_count"] == 3
    assert report["duplicate_count"] == 1
    assert any("duplicate" in w.lower() for w in report["warnings"])


def test_deterministic_split_and_no_train_eval_overlap(client: TestClient):
    """Test split creation guarantees 0% overlap and produces deterministic splits."""
    samples = [{"instruction": f"Task {i}", "output": f"Output {i}"} for i in range(20)]
    up_res = client.post(
        "/api/datasets/upload",
        files={"file": ("split_test.json", json.dumps(samples).encode("utf-8"), "application/json")},
    )
    dataset_id = up_res.json()["dataset_id"]

    # Request split: 80% train, 20% eval (seed 42)
    split_res = client.get(f"/api/datasets/{dataset_id}/splits?train_ratio=0.8&eval_ratio=0.2&seed=42")
    assert split_res.status_code == 200
    split_data = split_res.json()

    assert split_data["total_samples"] == 20
    assert split_data["train_count"] == 16
    assert split_data["eval_count"] == 4
    assert split_data["has_overlap"] is False

    # Verify zero overlap between train and eval indices
    train_set = set(split_data["train_indices"])
    eval_set = set(split_data["eval_indices"])
    assert len(train_set.intersection(eval_set)) == 0

    # Repeat with same seed to verify determinism
    repeat_res = client.get(f"/api/datasets/{dataset_id}/splits?train_ratio=0.8&eval_ratio=0.2&seed=42")
    assert repeat_res.json() == split_data


def test_split_validation_errors(client: TestClient):
    """Test invalid split ratios return 400 Bad Request."""
    data = json.dumps([{"instruction": "Task", "output": "Result"}]).encode("utf-8")
    up_res = client.post("/api/datasets/upload", files={"file": ("test.json", data, "application/json")})
    dataset_id = up_res.json()["dataset_id"]

    # Exceeding 1.0 total ratio
    bad_res = client.get(f"/api/datasets/{dataset_id}/splits?train_ratio=0.7&eval_ratio=0.5")
    assert bad_res.status_code == 400
    assert "exceeds" in bad_res.json()["detail"].lower()


def test_error_responses_without_filesystem_path_disclosure(client: TestClient):
    """Verify that error messages do not reveal absolute internal server paths."""
    res = client.get("/api/datasets/ds_not_found_123")
    assert res.status_code == 404
    assert "/home/" not in res.text
    assert "/tmp/" not in res.text
    assert "datasets/uploads" not in res.text
