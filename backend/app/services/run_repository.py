"""
SQLite-backed Persistent Run Repository for Phase 7.
Guarantees persistent run history, events, and logs across backend restarts.
Implements transactions, thread-safety, pagination, status filtering, and secret sanitization.
"""

import os
import json
import sqlite3
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

from backend.app.core.config import settings
from backend.app.schemas.history import (
    TrainingEvent,
    TrainingEventType,
    EventSeverity,
    TrainingLogEntry,
    RunHistoryItem,
)
from backend.app.schemas.artifacts import (
    ArtifactMetadata,
    ArtifactFileEntry,
    ArtifactType,
    ValidationStatus,
    ExportStatus,
)
from backend.app.schemas.evaluation import (
    EvaluationRun,
    EvaluationStatus,
    EvaluationConfig,
    EvaluationMetrics,
    EvaluationSample,
    BaselineComparison,
)


import re

HF_TOKEN_REGEX = re.compile(r"hf_[A-Za-z0-9_]{16,}")
BEARER_REGEX = re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.]{16,}")


def _sanitize_string(text: Optional[str], secrets: Optional[List[Optional[str]]] = None) -> Optional[str]:
    """Redacts known secrets and common token patterns from strings."""
    if not text:
        return text
    cleaned = str(text)
    if secrets:
        for secret in secrets:
            if secret and len(secret) > 4:
                cleaned = cleaned.replace(secret, "[REDACTED]")
    # Scrub Hugging Face tokens and Bearer headers automatically
    cleaned = HF_TOKEN_REGEX.sub("[REDACTED]", cleaned)
    cleaned = BEARER_REGEX.sub("Bearer [REDACTED]", cleaned)
    return cleaned


def _sanitize_dict_secrets(obj: Any) -> Any:
    """Recursively removes sensitive values (tokens, passwords, keys) from dict/list structures."""
    sensitive_keys = {"hf_token", "token", "password", "secret", "api_key", "authorization", "bearer"}
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if any(s in k.lower() for s in sensitive_keys):
                cleaned[k] = "[REDACTED]"
            else:
                cleaned[k] = _sanitize_dict_secrets(v)
        return cleaned
    elif isinstance(obj, list):
        return [_sanitize_dict_secrets(item) for item in obj]
    elif isinstance(obj, str):
        return _sanitize_string(obj)
    return obj


class RunRepository:
    """Thread-safe SQLite repository for training run state, events, and logs."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            os.makedirs(settings.runs_output_dir, exist_ok=True)
            self.db_path = os.path.join(settings.runs_output_dir, "runs.db")
        else:
            self.db_path = db_path
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        """Creates tables and indexes if they do not exist."""
        with self._lock, self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    project_name TEXT NOT NULL,
                    model_identifier TEXT NOT NULL,
                    dataset_reference TEXT NOT NULL,
                    training_method TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    elapsed_time_seconds REAL DEFAULT 0.0,
                    current_step INTEGER DEFAULT 0,
                    max_steps INTEGER DEFAULT 0,
                    current_epoch REAL DEFAULT 0.0,
                    num_epochs REAL DEFAULT 0.0,
                    current_loss REAL,
                    learning_rate REAL,
                    error_message TEXT,
                    device TEXT DEFAULT 'cpu',
                    latest_checkpoint TEXT,
                    output_dir TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs (created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_runs_status ON runs (status);

                CREATE TABLE IF NOT EXISTS training_events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    global_step INTEGER,
                    epoch REAL,
                    training_loss REAL,
                    learning_rate REAL,
                    checkpoint_path TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs (run_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_events_run_id ON training_events (run_id, timestamp);

                CREATE TABLE IF NOT EXISTS training_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    stage TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs (run_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_logs_run_id ON training_logs (run_id, log_id);

                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    model_identifier TEXT NOT NULL,
                    fine_tuning_method TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    file_reference TEXT NOT NULL,
                    size_bytes INTEGER DEFAULT 0,
                    adapter_config_json TEXT,
                    quantization_config_json TEXT,
                    checkpoint_step INTEGER,
                    validation_status TEXT NOT NULL,
                    export_status TEXT NOT NULL,
                    exported_package_path TEXT,
                    files_json TEXT NOT NULL,
                    validation_errors_json TEXT,
                    validation_warnings_json TEXT,
                    base_model_identifier TEXT,
                    is_loadable INTEGER,
                    load_error TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_artifacts_run_id ON artifacts (run_id);
                CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts (artifact_type);

                CREATE TABLE IF NOT EXISTS evaluations (
                    evaluation_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    elapsed_seconds REAL DEFAULT 0.0,
                    metrics_json TEXT,
                    samples_json TEXT,
                    baseline_comparison_json TEXT,
                    error_message TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_evaluations_run_id ON evaluations (run_id);
                CREATE INDEX IF NOT EXISTS idx_evaluations_artifact_id ON evaluations (artifact_id);
                CREATE INDEX IF NOT EXISTS idx_evaluations_status ON evaluations (status);

                CREATE TABLE IF NOT EXISTS quantization_jobs (
                    job_id TEXT PRIMARY KEY,
                    method TEXT NOT NULL,
                    source_reference TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    elapsed_seconds REAL DEFAULT 0.0,
                    output_artifact_path TEXT,
                    original_size_bytes INTEGER,
                    quantized_size_bytes INTEGER,
                    size_reduction_ratio REAL,
                    sha256 TEXT,
                    logs_json TEXT,
                    error_message TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_quant_jobs_status ON quantization_jobs (status);
                CREATE INDEX IF NOT EXISTS idx_quant_jobs_created ON quantization_jobs (created_at DESC);

                CREATE TABLE IF NOT EXISTS export_jobs (
                    export_id TEXT PRIMARY KEY,
                    export_type TEXT NOT NULL,
                    source_reference TEXT NOT NULL,
                    quantization_type TEXT,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    output_path TEXT,
                    file_size_bytes INTEGER,
                    sha256 TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    logs_json TEXT,
                    error_message TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_export_jobs_status ON export_jobs (status);

                CREATE TABLE IF NOT EXISTS serving_instances (
                    instance_id TEXT PRIMARY KEY,
                    model_reference TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    status TEXT NOT NULL,
                    host TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    config_json TEXT NOT NULL,
                    pid INTEGER,
                    created_at TEXT NOT NULL,
                    ready_at TEXT,
                    stopped_at TEXT,
                    error_message TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_serving_status ON serving_instances (status);
            """)

    def save_or_update_run(self, run_data: Dict[str, Any]) -> None:
        """Inserts or updates run record in the database."""
        now = datetime.now(timezone.utc).isoformat()
        run_id = run_data["run_id"]

        config_obj = run_data.get("config", {})
        if hasattr(config_obj, "model_dump"):
            config_dict = config_obj.model_dump(mode="json")
        elif hasattr(config_obj, "dict"):
            config_dict = config_obj.dict()
        elif isinstance(config_obj, dict):
            config_dict = config_obj
        else:
            config_dict = {}

        sanitized_cfg = _sanitize_dict_secrets(config_dict)
        config_json = json.dumps(sanitized_cfg)

        error_message = _sanitize_string(run_data.get("error_message"))

        with self._lock, self._get_connection() as conn:
            conn.execute("""
                INSERT INTO runs (
                    run_id, project_name, model_identifier, dataset_reference,
                    training_method, status, config_json, created_at, started_at,
                    completed_at, elapsed_time_seconds, current_step, max_steps,
                    current_epoch, num_epochs, current_loss, learning_rate,
                    error_message, device, latest_checkpoint, output_dir, updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(run_id) DO UPDATE SET
                    project_name = excluded.project_name,
                    model_identifier = excluded.model_identifier,
                    dataset_reference = excluded.dataset_reference,
                    training_method = excluded.training_method,
                    status = CASE 
                        WHEN runs.status IN ('completed', 'failed', 'cancelled') AND excluded.status NOT IN ('completed', 'failed', 'cancelled') 
                        THEN runs.status 
                        ELSE excluded.status 
                    END,
                    config_json = excluded.config_json,
                    started_at = COALESCE(excluded.started_at, runs.started_at),
                    completed_at = COALESCE(excluded.completed_at, runs.completed_at),
                    elapsed_time_seconds = excluded.elapsed_time_seconds,
                    current_step = excluded.current_step,
                    max_steps = excluded.max_steps,
                    current_epoch = excluded.current_epoch,
                    num_epochs = excluded.num_epochs,
                    current_loss = excluded.current_loss,
                    learning_rate = excluded.learning_rate,
                    error_message = excluded.error_message,
                    device = excluded.device,
                    latest_checkpoint = COALESCE(excluded.latest_checkpoint, runs.latest_checkpoint),
                    output_dir = excluded.output_dir,
                    updated_at = excluded.updated_at
            """, (
                run_id,
                run_data.get("project_name", "my-finetuned-model"),
                run_data.get("model_identifier", "unknown"),
                run_data.get("dataset_reference", "unknown"),
                run_data.get("training_method", "lora"),
                run_data.get("status", "queued"),
                config_json,
                run_data.get("created_at", now),
                run_data.get("started_at"),
                run_data.get("completed_at"),
                float(run_data.get("elapsed_time_seconds", 0.0) or 0.0),
                int(run_data.get("current_step", 0) or 0),
                int(run_data.get("max_steps", 0) or 0),
                float(run_data.get("current_epoch", 0.0) or 0.0),
                float(run_data.get("num_epochs", 0.0) or 0.0),
                run_data.get("current_loss"),
                run_data.get("learning_rate"),
                error_message,
                run_data.get("device", "cpu"),
                run_data.get("latest_checkpoint"),
                run_data.get("output_dir", ""),
                now,
            ))

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single run record by ID."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if not row:
                return None
            result = dict(row)
            try:
                result["config"] = json.loads(result["config_json"])
            except Exception:
                result["config"] = {}
            return result

    def list_runs(
        self,
        page: int = 1,
        page_size: int = 20,
        status_filter: Optional[str] = None,
    ) -> Tuple[List[RunHistoryItem], int]:
        """
        Lists runs in stable order (newest first) with pagination and optional status filter.
        Returns (list_of_items, total_count).
        """
        page = max(1, page)
        page_size = max(1, min(100, page_size))
        offset = (page - 1) * page_size

        with self._lock, self._get_connection() as conn:
            where_clause = ""
            params: List[Any] = []
            if status_filter and status_filter.strip() and status_filter.lower() != "all":
                where_clause = "WHERE status = ?"
                params.append(status_filter.lower().strip())

            count_query = f"SELECT COUNT(*) FROM runs {where_clause}"
            total = conn.execute(count_query, params).fetchone()[0]

            query = f"""
                SELECT run_id, project_name, model_identifier, dataset_reference,
                       training_method, status, created_at, started_at, completed_at,
                       elapsed_time_seconds, current_step, max_steps, current_epoch,
                       num_epochs, current_loss, learning_rate, error_message, device,
                       latest_checkpoint
                FROM runs
                {where_clause}
                ORDER BY created_at DESC, rowid DESC
                LIMIT ? OFFSET ?
            """
            rows = conn.execute(query, params + [page_size, offset]).fetchall()

            items = [
                RunHistoryItem(
                    run_id=row["run_id"],
                    project_name=row["project_name"],
                    model_identifier=row["model_identifier"],
                    dataset_reference=row["dataset_reference"],
                    training_method=row["training_method"],
                    status=row["status"],
                    created_at=row["created_at"],
                    started_at=row["started_at"],
                    completed_at=row["completed_at"],
                    elapsed_time_seconds=row["elapsed_time_seconds"],
                    current_step=row["current_step"],
                    max_steps=row["max_steps"],
                    current_epoch=row["current_epoch"],
                    num_epochs=row["num_epochs"],
                    current_loss=row["current_loss"],
                    learning_rate=row["learning_rate"],
                    error_message=row["error_message"],
                    device=row["device"],
                    latest_checkpoint=row["latest_checkpoint"],
                )
                for row in rows
            ]
            return items, total

    def add_event(self, event: TrainingEvent) -> None:
        """Stores a standardized training event."""
        clean_msg = _sanitize_string(event.message)
        with self._lock, self._get_connection() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO training_events (
                    event_id, run_id, timestamp, event_type, severity,
                    message, global_step, epoch, training_loss, learning_rate,
                    checkpoint_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.run_id,
                event.timestamp,
                event.event_type,
                event.severity.value if hasattr(event.severity, "value") else str(event.severity),
                clean_msg,
                event.global_step,
                event.epoch,
                event.training_loss,
                event.learning_rate,
                event.checkpoint_path,
            ))

    def get_events(
        self,
        run_id: str,
        since_event_id: Optional[str] = None,
        limit: int = 100,
    ) -> Tuple[List[TrainingEvent], int, bool]:
        """
        Retrieves training events in chronological order.
        Supports cursor/since_event_id filtering.
        Returns (events, total_run_events, has_more).
        """
        limit = max(1, min(500, limit))

        with self._lock, self._get_connection() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM training_events WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]

            if since_event_id:
                # Find rowid of since_event_id
                cursor = conn.execute(
                    "SELECT rowid FROM training_events WHERE event_id = ? AND run_id = ?",
                    (since_event_id, run_id),
                )
                since_row = cursor.fetchone()
                if since_row:
                    since_rowid = since_row[0]
                    rows = conn.execute(
                        """
                        SELECT event_id, run_id, timestamp, event_type, severity,
                               message, global_step, epoch, training_loss, learning_rate,
                               checkpoint_path
                        FROM training_events
                        WHERE run_id = ? AND rowid > ?
                        ORDER BY rowid ASC
                        LIMIT ?
                        """,
                        (run_id, since_rowid, limit + 1),
                    ).fetchall()
                else:
                    rows = []
            else:
                rows = conn.execute(
                    """
                    SELECT event_id, run_id, timestamp, event_type, severity,
                           message, global_step, epoch, training_loss, learning_rate,
                           checkpoint_path
                    FROM training_events
                    WHERE run_id = ?
                    ORDER BY rowid ASC
                    LIMIT ?
                    """,
                    (run_id, limit + 1),
                ).fetchall()

            has_more = len(rows) > limit
            selected_rows = rows[:limit]

            events = [
                TrainingEvent(
                    event_id=r["event_id"],
                    run_id=r["run_id"],
                    timestamp=r["timestamp"],
                    event_type=r["event_type"],
                    severity=EventSeverity(r["severity"]) if r["severity"] in ("info", "warning", "error") else EventSeverity.INFO,
                    message=r["message"],
                    global_step=r["global_step"],
                    epoch=r["epoch"],
                    training_loss=r["training_loss"],
                    learning_rate=r["learning_rate"],
                    checkpoint_path=r["checkpoint_path"],
                )
                for r in selected_rows
            ]

            return events, total, has_more

    def add_log(self, run_id: str, entry: TrainingLogEntry) -> None:
        """Appends a sanitized log entry for a run."""
        clean_msg = _sanitize_string(entry.message)
        with self._lock, self._get_connection() as conn:
            conn.execute("""
                INSERT INTO training_logs (run_id, timestamp, level, message, stage)
                VALUES (?, ?, ?, ?, ?)
            """, (
                run_id,
                entry.timestamp,
                entry.level,
                clean_msg,
                entry.stage,
            ))

    def get_logs(self, run_id: str, limit: int = 500) -> Tuple[List[TrainingLogEntry], int]:
        """Retrieves chronological logs for a run."""
        limit = max(1, min(2000, limit))
        with self._lock, self._get_connection() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM training_logs WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]

            rows = conn.execute(
                """
                SELECT timestamp, level, message, stage
                FROM training_logs
                WHERE run_id = ?
                ORDER BY log_id ASC
                LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()

            logs = [
                TrainingLogEntry(
                    timestamp=r["timestamp"],
                    level=r["level"],
                    message=r["message"],
                    stage=r["stage"],
                )
                for r in rows
            ]
            return logs, total

    # -------------------------------------------------------------
    # Phase 8: Artifacts Operations
    # -------------------------------------------------------------

    def save_or_update_artifact(self, artifact: ArtifactMetadata) -> None:
        """Saves or updates an artifact record in the database."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_connection() as conn:
            conn.execute("""
                INSERT INTO artifacts (
                    artifact_id, run_id, artifact_type, model_identifier,
                    fine_tuning_method, created_at, file_reference, size_bytes,
                    adapter_config_json, quantization_config_json, checkpoint_step,
                    validation_status, export_status, exported_package_path,
                    files_json, validation_errors_json, validation_warnings_json,
                    base_model_identifier, is_loadable, load_error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(artifact_id) DO UPDATE SET
                    file_reference = excluded.file_reference,
                    base_model_identifier = excluded.base_model_identifier,
                    validation_status = excluded.validation_status,
                    export_status = excluded.export_status,
                    exported_package_path = excluded.exported_package_path,
                    size_bytes = excluded.size_bytes,
                    files_json = excluded.files_json,
                    validation_errors_json = excluded.validation_errors_json,
                    validation_warnings_json = excluded.validation_warnings_json,
                    is_loadable = excluded.is_loadable,
                    load_error = excluded.load_error,
                    updated_at = excluded.updated_at
            """, (
                artifact.artifact_id,
                artifact.run_id,
                artifact.artifact_type.value,
                artifact.model_identifier,
                artifact.fine_tuning_method,
                artifact.created_at,
                artifact.file_reference,
                artifact.size_bytes,
                json.dumps(artifact.adapter_config) if artifact.adapter_config else None,
                json.dumps(artifact.quantization_config) if artifact.quantization_config else None,
                artifact.checkpoint_step,
                artifact.validation_status.value,
                artifact.export_status.value,
                artifact.exported_package_path,
                json.dumps([f.model_dump() for f in artifact.files]),
                json.dumps(artifact.validation_errors),
                json.dumps(artifact.validation_warnings),
                artifact.base_model_identifier,
                1 if artifact.is_loadable is True else (0 if artifact.is_loadable is False else None),
                artifact.load_error,
                now,
            ))

    def _row_to_artifact(self, row: sqlite3.Row) -> ArtifactMetadata:
        files_raw = json.loads(row["files_json"]) if row["files_json"] else []
        files = [ArtifactFileEntry(**f) for f in files_raw]

        is_loadable = None
        if row["is_loadable"] is not None:
            is_loadable = bool(row["is_loadable"])

        return ArtifactMetadata(
            artifact_id=row["artifact_id"],
            run_id=row["run_id"],
            artifact_type=ArtifactType(row["artifact_type"]),
            model_identifier=row["model_identifier"],
            fine_tuning_method=row["fine_tuning_method"],
            created_at=row["created_at"],
            file_reference=row["file_reference"],
            size_bytes=row["size_bytes"] or 0,
            adapter_config=json.loads(row["adapter_config_json"]) if row["adapter_config_json"] else None,
            quantization_config=json.loads(row["quantization_config_json"]) if row["quantization_config_json"] else None,
            checkpoint_step=row["checkpoint_step"],
            validation_status=ValidationStatus(row["validation_status"]),
            export_status=ExportStatus(row["export_status"]),
            exported_package_path=row["exported_package_path"],
            files=files,
            validation_errors=json.loads(row["validation_errors_json"]) if row["validation_errors_json"] else [],
            validation_warnings=json.loads(row["validation_warnings_json"]) if row["validation_warnings_json"] else [],
            base_model_identifier=row["base_model_identifier"],
            is_loadable=is_loadable,
            load_error=row["load_error"],
        )

    def get_artifact(self, artifact_id: str) -> Optional[ArtifactMetadata]:
        """Retrieves artifact metadata by ID."""
        with self._lock, self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
            if not row:
                return None
            return self._row_to_artifact(row)

    def list_artifacts(self, run_id: Optional[str] = None) -> List[ArtifactMetadata]:
        """Lists artifacts, optionally filtered by run ID."""
        with self._lock, self._get_connection() as conn:
            if run_id:
                rows = conn.execute(
                    "SELECT * FROM artifacts WHERE run_id = ? ORDER BY created_at DESC",
                    (run_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM artifacts ORDER BY created_at DESC"
                ).fetchall()
            return [self._row_to_artifact(r) for r in rows]

    # -------------------------------------------------------------
    # Phase 8: Evaluations Operations
    # -------------------------------------------------------------

    def save_or_update_evaluation(self, evaluation: EvaluationRun) -> None:
        """Saves or updates an evaluation run in the database."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_connection() as conn:
            conn.execute("""
                INSERT INTO evaluations (
                    evaluation_id, run_id, artifact_id, dataset_id, status,
                    config_json, created_at, started_at, completed_at,
                    elapsed_seconds, metrics_json, samples_json,
                    baseline_comparison_json, error_message, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evaluation_id) DO UPDATE SET
                    status = CASE 
                        WHEN evaluations.status IN ('completed', 'failed', 'cancelled') AND excluded.status NOT IN ('completed', 'failed', 'cancelled') 
                        THEN evaluations.status 
                        ELSE excluded.status 
                    END,
                    started_at = excluded.started_at,
                    completed_at = excluded.completed_at,
                    elapsed_seconds = excluded.elapsed_seconds,
                    metrics_json = excluded.metrics_json,
                    samples_json = excluded.samples_json,
                    baseline_comparison_json = excluded.baseline_comparison_json,
                    error_message = excluded.error_message,
                    updated_at = excluded.updated_at
            """, (
                evaluation.evaluation_id,
                evaluation.run_id,
                evaluation.artifact_id,
                evaluation.dataset_id,
                evaluation.status.value,
                json.dumps(evaluation.config.model_dump()),
                evaluation.created_at,
                evaluation.started_at,
                evaluation.completed_at,
                evaluation.elapsed_seconds,
                json.dumps(evaluation.metrics.model_dump()) if evaluation.metrics else None,
                json.dumps([s.model_dump() for s in evaluation.samples]),
                json.dumps(evaluation.baseline_comparison.model_dump()) if evaluation.baseline_comparison else None,
                _sanitize_string(evaluation.error_message),
                now,
            ))

    def _row_to_evaluation(self, row: sqlite3.Row) -> EvaluationRun:
        config_dict = json.loads(row["config_json"]) if row["config_json"] else {}
        metrics_dict = json.loads(row["metrics_json"]) if row["metrics_json"] else None
        samples_raw = json.loads(row["samples_json"]) if row["samples_json"] else []
        baseline_dict = json.loads(row["baseline_comparison_json"]) if row["baseline_comparison_json"] else None

        return EvaluationRun(
            evaluation_id=row["evaluation_id"],
            run_id=row["run_id"],
            artifact_id=row["artifact_id"],
            dataset_id=row["dataset_id"],
            status=EvaluationStatus(row["status"]),
            config=EvaluationConfig(**config_dict),
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            elapsed_seconds=row["elapsed_seconds"] or 0.0,
            metrics=EvaluationMetrics(**metrics_dict) if metrics_dict else None,
            samples=[EvaluationSample(**s) for s in samples_raw],
            baseline_comparison=BaselineComparison(**baseline_dict) if baseline_dict else None,
            error_message=row["error_message"],
        )

    def get_evaluation(self, evaluation_id: str) -> Optional[EvaluationRun]:
        """Retrieves evaluation run by ID."""
        with self._lock, self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM evaluations WHERE evaluation_id = ?",
                (evaluation_id,),
            ).fetchone()
            if not row:
                return None
            return self._row_to_evaluation(row)

    def list_evaluations(
        self,
        run_id: Optional[str] = None,
        artifact_id: Optional[str] = None,
        status_filter: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Tuple[List[EvaluationRun], int]:
        """Lists evaluations with optional filtering and pagination."""
        page = max(1, page)
        page_size = max(1, min(100, page_size))
        offset = (page - 1) * page_size

        where_clauses = []
        params: List[Any] = []

        if run_id:
            where_clauses.append("run_id = ?")
            params.append(run_id)
        if artifact_id:
            where_clauses.append("artifact_id = ?")
            params.append(artifact_id)
        if status_filter:
            where_clauses.append("status = ?")
            params.append(status_filter.lower())

        where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        with self._lock, self._get_connection() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM evaluations {where_str}",
                params,
            ).fetchone()[0]

            rows = conn.execute(
                f"""
                SELECT * FROM evaluations
                {where_str}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                params + [page_size, offset],
            ).fetchall()

            evaluations = [self._row_to_evaluation(r) for r in rows]
            return evaluations, total

    # ─── Phase 10: Quantization Jobs CRUD ───────────────────────────────────────

    def save_or_update_quantization_job(self, job_data: Dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        job_id = job_data["job_id"]
        config_obj = job_data.get("config", {})
        config_dict = config_obj if isinstance(config_obj, dict) else config_obj.model_dump(mode="json")
        config_json = json.dumps(_sanitize_dict_secrets(config_dict))
        logs_json = json.dumps(job_data.get("logs", []))

        method_val = job_data.get("method", "bnb_4bit")
        if hasattr(method_val, "value"):
            method_val = method_val.value
        status_val = job_data.get("status", "queued")
        if hasattr(status_val, "value"):
            status_val = status_val.value

        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO quantization_jobs (
                    job_id, method, source_reference, status, config_json,
                    created_at, started_at, completed_at, elapsed_seconds,
                    output_artifact_path, original_size_bytes, quantized_size_bytes,
                    size_reduction_ratio, sha256, logs_json, error_message, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status = CASE 
                        WHEN quantization_jobs.status IN ('completed', 'failed', 'cancelled') 
                        THEN quantization_jobs.status 
                        ELSE excluded.status 
                    END,
                    started_at = COALESCE(excluded.started_at, quantization_jobs.started_at),
                    completed_at = COALESCE(excluded.completed_at, quantization_jobs.completed_at),
                    elapsed_seconds = excluded.elapsed_seconds,
                    output_artifact_path = COALESCE(excluded.output_artifact_path, quantization_jobs.output_artifact_path),
                    original_size_bytes = COALESCE(excluded.original_size_bytes, quantization_jobs.original_size_bytes),
                    quantized_size_bytes = COALESCE(excluded.quantized_size_bytes, quantization_jobs.quantized_size_bytes),
                    size_reduction_ratio = COALESCE(excluded.size_reduction_ratio, quantization_jobs.size_reduction_ratio),
                    sha256 = COALESCE(excluded.sha256, quantization_jobs.sha256),
                    logs_json = excluded.logs_json,
                    error_message = COALESCE(excluded.error_message, quantization_jobs.error_message),
                    updated_at = excluded.updated_at;
                """,
                (
                    job_id,
                    str(method_val),
                    job_data.get("source_reference", ""),
                    str(status_val),
                    config_json,
                    job_data.get("created_at", now),
                    job_data.get("started_at"),
                    job_data.get("completed_at"),
                    float(job_data.get("elapsed_seconds", 0.0)),
                    job_data.get("output_artifact_path"),
                    job_data.get("original_size_bytes"),
                    job_data.get("quantized_size_bytes"),
                    job_data.get("size_reduction_ratio"),
                    job_data.get("sha256"),
                    logs_json,
                    job_data.get("error_message"),
                    now,
                ),
            )

    def get_quantization_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            row = conn.execute("SELECT * FROM quantization_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            d["config"] = json.loads(d.get("config_json") or "{}")
            d["logs"] = json.loads(d.get("logs_json") or "[]")
            return d

    def list_quantization_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM quantization_jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["config"] = json.loads(d.get("config_json") or "{}")
                d["logs"] = json.loads(d.get("logs_json") or "[]")
                result.append(d)
            return result

    # ─── Phase 10: Export Jobs CRUD (GGUF) ──────────────────────────────────────

    def save_or_update_export_job(self, export_data: Dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        export_id = export_data["export_id"]
        config_obj = export_data.get("config", {})
        config_dict = config_obj if isinstance(config_obj, dict) else config_obj.model_dump(mode="json")
        config_json = json.dumps(_sanitize_dict_secrets(config_dict))
        logs_json = json.dumps(export_data.get("logs", []))

        export_type = export_data.get("export_type", "gguf")
        if hasattr(export_type, "value"):
            export_type = export_type.value
        quant_type = export_data.get("quantization_type")
        if hasattr(quant_type, "value"):
            quant_type = quant_type.value
        status_val = export_data.get("status", "queued")
        if hasattr(status_val, "value"):
            status_val = status_val.value

        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO export_jobs (
                    export_id, export_type, source_reference, quantization_type,
                    status, config_json, output_path, file_size_bytes, sha256,
                    created_at, completed_at, logs_json, error_message, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(export_id) DO UPDATE SET
                    status = CASE 
                        WHEN export_jobs.status IN ('completed', 'failed', 'cancelled') 
                        THEN export_jobs.status 
                        ELSE excluded.status 
                    END,
                    output_path = COALESCE(excluded.output_path, export_jobs.output_path),
                    file_size_bytes = COALESCE(excluded.file_size_bytes, export_jobs.file_size_bytes),
                    sha256 = COALESCE(excluded.sha256, export_jobs.sha256),
                    completed_at = COALESCE(excluded.completed_at, export_jobs.completed_at),
                    logs_json = excluded.logs_json,
                    error_message = COALESCE(excluded.error_message, export_jobs.error_message),
                    updated_at = excluded.updated_at;
                """,
                (
                    export_id,
                    str(export_type),
                    export_data.get("source_reference") or export_data.get("source_artifact_id", ""),
                    str(quant_type) if quant_type else None,
                    str(status_val),
                    config_json,
                    export_data.get("output_path") or export_data.get("output_filepath"),
                    export_data.get("file_size_bytes"),
                    export_data.get("sha256"),
                    export_data.get("created_at", now),
                    export_data.get("completed_at"),
                    logs_json,
                    export_data.get("error_message"),
                    now,
                ),
            )

    def get_export_job(self, export_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            row = conn.execute("SELECT * FROM export_jobs WHERE export_id = ?", (export_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            d["config"] = json.loads(d.get("config_json") or "{}")
            d["logs"] = json.loads(d.get("logs_json") or "[]")
            d["source_artifact_id"] = d.get("source_reference")
            d["output_filepath"] = d.get("output_path")
            return d

    def list_export_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM export_jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["config"] = json.loads(d.get("config_json") or "{}")
                d["logs"] = json.loads(d.get("logs_json") or "[]")
                d["source_artifact_id"] = d.get("source_reference")
                d["output_filepath"] = d.get("output_path")
                result.append(d)
            return result

    # ─── Phase 10: Serving Instances CRUD (vLLM) ────────────────────────────────

    def save_or_update_serving_instance(self, inst_data: Dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        instance_id = inst_data["instance_id"]
        config_obj = inst_data.get("config", {})
        config_dict = config_obj if isinstance(config_obj, dict) else config_obj.model_dump(mode="json")
        config_json = json.dumps(_sanitize_dict_secrets(config_dict))

        backend_val = inst_data.get("backend", "vllm")
        if hasattr(backend_val, "value"):
            backend_val = backend_val.value
        status_val = inst_data.get("status", "validating")
        if hasattr(status_val, "value"):
            status_val = status_val.value

        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO serving_instances (
                    instance_id, model_reference, backend, status, host, port,
                    config_json, pid, created_at, ready_at, stopped_at, error_message, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(instance_id) DO UPDATE SET
                    status = excluded.status,
                    host = excluded.host,
                    port = excluded.port,
                    pid = COALESCE(excluded.pid, serving_instances.pid),
                    ready_at = COALESCE(excluded.ready_at, serving_instances.ready_at),
                    stopped_at = COALESCE(excluded.stopped_at, serving_instances.stopped_at),
                    error_message = COALESCE(excluded.error_message, serving_instances.error_message),
                    updated_at = excluded.updated_at;
                """,
                (
                    instance_id,
                    inst_data.get("model_reference", ""),
                    str(backend_val),
                    str(status_val),
                    inst_data.get("host", "127.0.0.1"),
                    int(inst_data.get("port", 8080)),
                    config_json,
                    inst_data.get("pid"),
                    inst_data.get("created_at", now),
                    inst_data.get("ready_at"),
                    inst_data.get("stopped_at"),
                    inst_data.get("error_message"),
                    now,
                ),
            )

    def get_serving_instance(self, instance_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            row = conn.execute("SELECT * FROM serving_instances WHERE instance_id = ?", (instance_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            d["config"] = json.loads(d.get("config_json") or "{}")
            return d

    def list_serving_instances(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM serving_instances ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["config"] = json.loads(d.get("config_json") or "{}")
                result.append(d)
            return result



_run_repository: Optional[RunRepository] = None


def get_run_repository() -> RunRepository:
    """Singleton getter for RunRepository."""
    global _run_repository
    if _run_repository is None:
        _run_repository = RunRepository()
    return _run_repository
