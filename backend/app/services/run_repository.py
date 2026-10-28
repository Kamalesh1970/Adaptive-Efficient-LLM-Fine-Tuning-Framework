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
            """)

    def save_or_update_run(self, run_data: Dict[str, Any]) -> None:
        """Inserts or updates run record in the database."""
        now = datetime.now(timezone.utc).isoformat()
        run_id = run_data["run_id"]

        config_obj = run_data.get("config", {})
        if hasattr(config_obj, "model_dump"):
            config_dict = config_obj.model_dump()
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
                    status = excluded.status,
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


_run_repository: Optional[RunRepository] = None


def get_run_repository() -> RunRepository:
    """Singleton getter for RunRepository."""
    global _run_repository
    if _run_repository is None:
        _run_repository = RunRepository()
    return _run_repository
