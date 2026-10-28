"""
Phase 7 Tests: Real-Time Training Monitoring, Logs, Cancellation, and Training History.
Covers:
- Thread-safe SQLite RunRepository persistence
- Run history retrieval with pagination and status filtering
- Standardized training lifecycle event emission and cursor-based incremental retrieval
- Sanitized worker log streaming (redaction of secrets, HF tokens, Bearer headers)
- Verified checkpoint metadata discovery on disk
- Safe training run cancellation and idempotency
- Persistence recovery across simulated server restarts
- Concurrency and transaction safety under concurrent telemetry updates
"""

import os
import json
import time
import tempfile
import threading
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.schemas.training import (
    TrainingConfigSchema,
    TrainingMethod,
    TrainingLengthType,
    TrainingRunStatus,
)
from backend.app.schemas.history import (
    TrainingEvent,
    TrainingEventType,
    EventSeverity,
    TrainingLogEntry,
    RunHistoryItem,
    CheckpointMetadata,
)
from backend.app.services.run_repository import (
    RunRepository,
    _sanitize_string,
    _sanitize_dict_secrets,
)
from backend.app.services.job_manager import (
    JobManager,
    TrainingJob,
    get_job_manager,
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def temp_repo(tmp_path):
    """Provides an isolated SQLite repository for testing."""
    db_file = os.path.join(tmp_path, "test_runs.db")
    return RunRepository(db_path=db_file)


# ============================================================================
# 1. Secret Sanitization & Security Isolation Tests
# ============================================================================

def test_sanitize_string_scrubs_hf_tokens_and_bearer_headers():
    raw_text = "Exception occurred with hf_abcdefghijklmnopqrstuvwxyz123 and Bearer my-secret-token-123456"
    sanitized = _sanitize_string(raw_text)
    assert "hf_abcdefghijklmnopqrstuvwxyz123" not in sanitized
    assert "my-secret-token-123456" not in sanitized
    assert "[REDACTED]" in sanitized
    assert "Bearer [REDACTED]" in sanitized


def test_sanitize_string_custom_secret_redaction():
    custom_secret = "super_secret_hf_key_9999"
    raw_text = f"Connecting using {custom_secret} to load weights"
    sanitized = _sanitize_string(raw_text, secrets=[custom_secret])
    assert custom_secret not in sanitized
    assert "[REDACTED]" in sanitized


def test_sanitize_dict_secrets_recursively_redacts():
    nested_config = {
        "project_name": "fine-tune-llama",
        "hf_token": "hf_live_token_secret_12345678",
        "nested": {
            "token": "secret_abc",
            "api_key": "key_xyz",
            "learning_rate": 0.0002,
        },
        "log_messages": [
            "Loaded model with token hf_123456789012345678",
            "Safe log entry",
        ]
    }
    cleaned = _sanitize_dict_secrets(nested_config)
    assert cleaned["hf_token"] == "[REDACTED]"
    assert cleaned["nested"]["token"] == "[REDACTED]"
    assert cleaned["nested"]["api_key"] == "[REDACTED]"
    assert cleaned["nested"]["learning_rate"] == 0.0002
    assert "hf_123456789012345678" not in cleaned["log_messages"][0]
    assert "[REDACTED]" in cleaned["log_messages"][0]


# ============================================================================
# 2. SQLite RunRepository Unit Tests
# ============================================================================

def test_run_repository_init_creates_tables(temp_repo):
    assert os.path.exists(temp_repo.db_path)
    with temp_repo._get_connection() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()
        table_names = [t[0] for t in tables]
        assert "runs" in table_names
        assert "training_events" in table_names
        assert "training_logs" in table_names


def test_save_and_get_run(temp_repo):
    run_data = {
        "run_id": "run_test_001",
        "project_name": "llama3-lora",
        "model_identifier": "unsloth/llama-3-8b",
        "dataset_reference": "dataset_alpaca",
        "training_method": "lora",
        "status": "training",
        "config": {"model_identifier": "unsloth/llama-3-8b", "learning_rate": 0.0002},
        "created_at": "2026-10-28T12:00:00Z",
        "started_at": "2026-10-28T12:00:05Z",
        "current_step": 10,
        "max_steps": 50,
        "current_epoch": 0.6,
        "num_epochs": 3.0,
        "current_loss": 1.954,
        "device": "cuda:0",
    }
    temp_repo.save_or_update_run(run_data)

    retrieved = temp_repo.get_run("run_test_001")
    assert retrieved is not None
    assert retrieved["run_id"] == "run_test_001"
    assert retrieved["project_name"] == "llama3-lora"
    assert retrieved["current_step"] == 10
    assert retrieved["current_loss"] == 1.954
    assert retrieved["config"]["learning_rate"] == 0.0002


def test_list_runs_pagination_and_status_filter(temp_repo):
    # Insert multiple runs with different statuses
    for i in range(15):
        st = "completed" if i % 2 == 0 else "failed"
        temp_repo.save_or_update_run({
            "run_id": f"run_{i:03d}",
            "project_name": f"project_{i}",
            "model_identifier": "test/model",
            "dataset_reference": "test/data",
            "training_method": "lora",
            "status": st,
            "created_at": f"2026-10-28T12:{i:02d}:00Z",
        })

    # Page 1 (10 items)
    items, total = temp_repo.list_runs(page=1, page_size=10)
    assert total == 15
    assert len(items) == 10
    # Newest first
    assert items[0].run_id == "run_014"

    # Page 2 (5 items)
    items_p2, total_p2 = temp_repo.list_runs(page=2, page_size=10)
    assert total_p2 == 15
    assert len(items_p2) == 5

    # Status filter: completed (8 runs: 0, 2, 4, 6, 8, 10, 12, 14)
    completed_items, comp_total = temp_repo.list_runs(page=1, page_size=20, status_filter="completed")
    assert comp_total == 8
    assert all(r.status == "completed" for r in completed_items)

    # Status filter: failed (7 runs: 1, 3, 5, 7, 9, 11, 13)
    failed_items, fail_total = temp_repo.list_runs(page=1, page_size=20, status_filter="failed")
    assert fail_total == 7
    assert all(r.status == "failed" for r in failed_items)


def test_add_and_get_events_with_cursor_pagination(temp_repo):
    temp_repo.save_or_update_run({
        "run_id": "run_event_test",
        "project_name": "event-proj",
        "model_identifier": "test/model",
        "dataset_reference": "test/data",
        "training_method": "lora",
        "status": "training",
    })

    # Add 5 events
    event_ids = []
    for i in range(5):
        eid = f"evt_{i:03d}"
        event_ids.append(eid)
        temp_repo.add_event(TrainingEvent(
            event_id=eid,
            run_id="run_event_test",
            timestamp=f"2026-10-28T12:00:0{i}Z",
            event_type=TrainingEventType.TRAINING_STEP if i > 0 else TrainingEventType.TRAINING_STARTED,
            severity=EventSeverity.INFO,
            message=f"Step {i} completed",
            global_step=i,
        ))

    # Retrieve all
    evts, total, has_more = temp_repo.get_events("run_event_test", limit=10)
    assert total == 5
    assert len(evts) == 5
    assert not has_more

    # Retrieve with limit=2 (first page)
    evts_page1, _, has_more1 = temp_repo.get_events("run_event_test", limit=2)
    assert len(evts_page1) == 2
    assert has_more1
    assert evts_page1[0].event_id == "evt_000"
    assert evts_page1[1].event_id == "evt_001"

    # Incremental cursor retrieval from evt_001
    evts_page2, _, has_more2 = temp_repo.get_events("run_event_test", since_event_id="evt_001", limit=2)
    assert len(evts_page2) == 2
    assert has_more2
    assert evts_page2[0].event_id == "evt_002"
    assert evts_page2[1].event_id == "evt_003"


def test_add_and_get_logs_with_secret_scrubbing(temp_repo):
    temp_repo.save_or_update_run({
        "run_id": "run_log_test",
        "project_name": "log-proj",
        "model_identifier": "test/model",
        "dataset_reference": "test/data",
        "training_method": "lora",
        "status": "training",
    })

    secret_token = "hf_super_secret_training_token_999"
    temp_repo.add_log(
        run_id="run_log_test",
        entry=TrainingLogEntry(
            timestamp="2026-10-28T12:00:00Z",
            level="INFO",
            message=f"Authenticating worker with {secret_token}",
            stage="preparation",
        )
    )

    logs, total = temp_repo.get_logs("run_log_test")
    assert total == 1
    assert len(logs) == 1
    assert secret_token not in logs[0].message
    assert "[REDACTED]" in logs[0].message


# ============================================================================
# 3. Server Restart Persistence Recovery
# ============================================================================

def test_persistence_recovery_across_restart(tmp_path):
    db_file = os.path.join(tmp_path, "restart_test.db")
    runs_dir = os.path.join(tmp_path, "runs_out")
    os.makedirs(runs_dir, exist_ok=True)

    # 1. First JobManager session
    repo1 = RunRepository(db_path=db_file)
    mgr1 = JobManager(run_repo=repo1, runs_output_dir=runs_dir)

    cfg = TrainingConfigSchema(
        project_name="persistent-project",
        model_identifier="test/tiny-model",
        dataset_reference="test-dataset",
        training_method=TrainingMethod.LORA,
        max_steps=100,
        learning_rate=0.0003,
    )
    job = mgr1.create_job(config=cfg)
    run_id = job.run_id

    # Simulate some progress and state change
    job.status = TrainingRunStatus.TRAINING
    job.current_step = 42
    job.current_loss = 1.3456
    job.sync_to_db()

    job.emit_event(
        event_type=TrainingEventType.TRAINING_STEP,
        severity=EventSeverity.INFO,
        message="Step 42 telemetry",
        step=42,
        loss=1.3456,
    )
    job.add_log(level="INFO", message="Checkpoint reached step 42", stage="training")

    # 2. Simulate complete backend restart (new repository & job manager pointing to same SQLite DB)
    repo2 = RunRepository(db_path=db_file)
    mgr2 = JobManager(run_repo=repo2, runs_output_dir=runs_dir)

    # In memory, mgr2 has empty _jobs initially
    assert run_id not in mgr2._jobs

    # Fetching the job reconstructs it from persisted SQLite state
    recovered_job = mgr2.get_job(run_id)
    assert recovered_job is not None
    assert recovered_job.run_id == run_id
    assert recovered_job.config.project_name == "persistent-project"
    assert recovered_job.status == TrainingRunStatus.TRAINING
    assert recovered_job.current_step == 42
    assert recovered_job.current_loss == 1.3456
    assert recovered_job.config.model_identifier == "test/tiny-model"

    # Events and logs also persist across restart
    evts, e_tot, _ = repo2.get_events(run_id)
    assert e_tot >= 2  # QUEUED + Step 42
    assert any(e.global_step == 42 for e in evts)

    logs, l_tot = repo2.get_logs(run_id)
    assert l_tot >= 1
    assert any("step 42" in l.message for l in logs)


# ============================================================================
# 4. Multi-Threaded Concurrency & Transaction Safety
# ============================================================================

def test_concurrent_telemetry_and_event_writes(temp_repo):
    run_id = "run_concurrent_test"
    temp_repo.save_or_update_run({
        "run_id": run_id,
        "project_name": "concurrent-project",
        "model_identifier": "test/model",
        "dataset_reference": "test/dataset",
        "training_method": "lora",
        "status": "training",
    })

    num_threads = 6
    writes_per_thread = 25
    errors = []

    def worker_target(thread_idx: int):
        try:
            for i in range(writes_per_thread):
                step = thread_idx * writes_per_thread + i
                temp_repo.add_event(TrainingEvent(
                    event_id=f"evt_{thread_idx}_{i}",
                    run_id=run_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    event_type=TrainingEventType.TRAINING_STEP,
                    severity=EventSeverity.INFO,
                    message=f"Thread {thread_idx} wrote step {step}",
                    global_step=step,
                ))
                temp_repo.add_log(
                    run_id=run_id,
                    entry=TrainingLogEntry(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        level="INFO",
                        message=f"Worker log step {step}",
                        stage="training",
                    )
                )
                temp_repo.save_or_update_run({
                    "run_id": run_id,
                    "current_step": step,
                    "current_loss": 1.0 / (step + 1),
                })
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker_target, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Encountered concurrency errors: {errors}"

    evts, evt_count, _ = temp_repo.get_events(run_id, limit=500)
    assert evt_count == num_threads * writes_per_thread

    logs, log_count = temp_repo.get_logs(run_id, limit=500)
    assert log_count == num_threads * writes_per_thread


# ============================================================================
# 5. Checkpoint Metadata Discovery Tests
# ============================================================================

def test_checkpoint_discovery_on_disk(tmp_path):
    run_id = "run_cp_test"
    run_dir = os.path.join(tmp_path, "runs_out", run_id)
    os.makedirs(run_dir, exist_ok=True)

    # 1. Final adapter files
    adapter_cfg = {"peft_type": "LORA", "r": 16, "lora_alpha": 32}
    with open(os.path.join(run_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_cfg, f)
    with open(os.path.join(run_dir, "adapter_model.safetensors"), "wb") as f:
        f.write(b"0" * 1024)

    # 2. Intermediate checkpoint directory (checkpoint-50)
    cp_dir = os.path.join(run_dir, "checkpoint-50")
    os.makedirs(cp_dir, exist_ok=True)
    with open(os.path.join(cp_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_cfg, f)
    with open(os.path.join(cp_dir, "adapter_model.safetensors"), "wb") as f:
        f.write(b"0" * 512)

    db_path = os.path.join(tmp_path, "runs.db")
    repo = RunRepository(db_path=db_path)
    mgr = JobManager(run_repo=repo, runs_output_dir=os.path.join(tmp_path, "runs_out"))

    checkpoints = mgr.get_checkpoints(run_id)
    assert len(checkpoints) == 2

    cp_ids = [cp.checkpoint_id for cp in checkpoints]
    assert "final_adapter" in cp_ids
    assert "checkpoint-50" in cp_ids

    final_cp = next(cp for cp in checkpoints if cp.checkpoint_id == "final_adapter")
    assert final_cp.size_bytes >= 1024
    assert "adapter_config.json" in final_cp.files
    assert "adapter_model.safetensors" in final_cp.files

    step50_cp = next(cp for cp in checkpoints if cp.checkpoint_id == "checkpoint-50")
    assert step50_cp.step == 50
    assert step50_cp.size_bytes >= 512


# ============================================================================
# 6. REST API Endpoints Tests
# ============================================================================

def test_api_list_training_runs(client):
    res = client.get("/api/training/runs?page=1&page_size=10")
    assert res.status_code == 200
    data = res.json()
    assert "runs" in data
    assert "total" in data
    assert "page" in data
    assert "total_pages" in data
    assert isinstance(data["runs"], list)


def test_api_get_training_run_telemetry(client):
    mgr = get_job_manager()
    cfg = TrainingConfigSchema(
        project_name="telemetry-api-test",
        model_identifier="test/model",
        dataset_reference="test/data",
        training_method=TrainingMethod.LORA,
    )
    job = mgr.create_job(config=cfg)

    res = client.get(f"/api/training/runs/{job.run_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["run_id"] == job.run_id
    assert data["status"] == "queued"
    assert data["current_step"] == 0
    assert "loss_history" in data


def test_api_get_events_endpoint(client):
    mgr = get_job_manager()
    cfg = TrainingConfigSchema(
        project_name="events-api-test",
        model_identifier="test/model",
        dataset_reference="test/data",
        training_method=TrainingMethod.LORA,
    )
    job = mgr.create_job(config=cfg)
    job.emit_event(
        event_type=TrainingEventType.TRAINING_STARTED,
        severity=EventSeverity.INFO,
        message="Job initialization complete",
    )

    res = client.get(f"/api/training/runs/{job.run_id}/events")
    assert res.status_code == 200
    data = res.json()
    assert data["run_id"] == job.run_id
    assert len(data["events"]) >= 1
    assert any(e["event_type"] == "training_started" for e in data["events"])


def test_api_get_logs_endpoint(client):
    mgr = get_job_manager()
    cfg = TrainingConfigSchema(
        project_name="logs-api-test",
        model_identifier="test/model",
        dataset_reference="test/data",
        training_method=TrainingMethod.LORA,
    )
    job = mgr.create_job(config=cfg)
    job.add_log(level="INFO", message="Loading weights into device cpu", stage="loading_model")

    res = client.get(f"/api/training/runs/{job.run_id}/logs")
    assert res.status_code == 200
    data = res.json()
    assert data["run_id"] == job.run_id
    assert len(data["logs"]) >= 1
    assert any("Loading weights" in l["message"] for l in data["logs"])


def test_api_cancel_run_endpoint(client):
    mgr = get_job_manager()
    cfg = TrainingConfigSchema(
        project_name="cancel-api-test",
        model_identifier="test/model",
        dataset_reference="test/data",
        training_method=TrainingMethod.LORA,
    )
    job = mgr.create_job(config=cfg)
    # job is initially in QUEUED status

    res = client.post(f"/api/training/runs/{job.run_id}/cancel")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "cancelled"
    assert data["cancellable"] is False

    # Verify event recorded
    evts_res = client.get(f"/api/training/runs/{job.run_id}/events")
    evt_types = [e["event_type"] for e in evts_res.json()["events"]]
    assert "cancellation_requested" in evt_types
    assert "cancellation_completed" in evt_types
