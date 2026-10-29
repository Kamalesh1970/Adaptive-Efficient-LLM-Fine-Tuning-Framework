"""
Training Job Lifecycle Manager (Phase 4 & Phase 7).
Orchestrates training runs through explicit lifecycle states:
QUEUED -> VALIDATING -> LOADING_MODEL -> PREPARING_DATASET -> TRAINING -> SAVING -> COMPLETED.
Provides thread-safe job tracking, safe cancellation, artifact validation, real-time telemetry,
structured event emission, sanitized logging, and SQLite persistence.
"""

import os
import json
import uuid
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone

from backend.app.core.config import settings
from backend.app.schemas.training import (
    TrainingConfigSchema,
    TrainingMethod,
    TrainingRunStatus,
    TrainingMetricPoint,
    TrainingRunResponse,
    TrainingArtifactMetadata,
)
from backend.app.schemas.history import (
    TrainingEvent,
    TrainingEventType,
    EventSeverity,
    TrainingLogEntry,
    CheckpointMetadata,
)
from backend.app.services.run_repository import (
    RunRepository,
    get_run_repository,
)
from backend.app.services.model_loader import (
    ModelLoaderService,
    get_model_loader_service,
    ModelLoadError,
    QuantizationDeviceError,
)
from backend.app.services.lora_service import (
    inject_lora_adapters,
    prepare_and_inject_qlora_adapters,
    save_and_verify_adapter,
    reload_lora_adapter,
    LoRAConfigurationError,
)
from backend.app.services.ml_dependencies import (
    verify_qlora_dependencies_or_raise,
    BitsAndBytesMissingError,
    CudaUnavailableForQuantizationError,
)
from backend.app.services.training_dataset_service import (
    TrainingDatasetService,
    get_training_dataset_service,
)
from backend.app.services.training_engine import (
    TrainingEngine,
    TrainingCancelledException,
    TrainingExecutionError,
)

logger = logging.getLogger(__name__)


def _sanitize_string(val: Optional[str], secrets_to_remove: List[Optional[str]]) -> str:
    """Removes sensitive credentials or tokens from error messages and logs."""
    if not val:
        return ""
    cleaned = str(val)
    for secret in secrets_to_remove:
        if secret and len(secret) > 4:
            cleaned = cleaned.replace(secret, "[REDACTED]")
    return cleaned


class TrainingJob:
    """Encapsulates state, event logging, and telemetry for a single training run."""

    def __init__(
        self,
        run_id: str,
        config: TrainingConfigSchema,
        output_dir: str,
        hf_token: Optional[str] = None,
        repo: Optional[RunRepository] = None,
    ) -> None:
        self.run_id = run_id
        self.config = config
        self.output_dir = output_dir
        self._hf_token = hf_token
        self.repo = repo or get_run_repository()

        self._lock = threading.RLock()
        self._status = TrainingRunStatus.QUEUED
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None

        self.current_step: int = 0
        self.max_steps: int = config.max_steps or 0
        self.current_epoch: float = 0.0
        self.num_epochs: float = float(getattr(config, "num_epochs", 0) or 0)
        self.current_loss: Optional[float] = None
        self.loss_history: List[TrainingMetricPoint] = []
        self.elapsed_time_seconds: float = 0.0
        self.error_message: Optional[str] = None
        self.device: str = "cpu"
        self.latest_checkpoint: Optional[str] = None

        self.trainable_params: Optional[int] = None
        self.total_params: Optional[int] = None
        self.trainable_percent: Optional[float] = None
        self.quantization_info: Optional[Dict[str, Any]] = None
        if config.training_method == TrainingMethod.QLORA:
            self.quantization_info = config.to_quantization_dict()

        self._cancel_event = threading.Event()
        self.artifact_files: List[str] = []
        self.adapter_config: Dict[str, Any] = {}
        self.events: List[TrainingEvent] = []
        self.logs: List[TrainingLogEntry] = []

    @property
    def status(self) -> TrainingRunStatus:
        return self._status

    @status.setter
    def status(self, new_status: Union[TrainingRunStatus, str]) -> None:
        if isinstance(new_status, str):
            new_status = TrainingRunStatus(new_status)
        terminal_statuses = {
            TrainingRunStatus.COMPLETED,
            TrainingRunStatus.FAILED,
            TrainingRunStatus.CANCELLED,
        }
        if hasattr(self, "_status") and self._status in terminal_statuses:
            if new_status != self._status:
                raise ValueError(
                    f"Invalid status transition: Run '{self.run_id}' is in terminal state '{self._status.value}' and cannot transition to '{new_status.value}'."
                )
        self._status = new_status

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def emit_event(
        self,
        event_type: Union[TrainingEventType, str],
        severity: EventSeverity = EventSeverity.INFO,
        message: str = "",
        step: Optional[int] = None,
        epoch: Optional[float] = None,
        loss: Optional[float] = None,
        lr: Optional[float] = None,
        checkpoint_path: Optional[str] = None,
    ) -> TrainingEvent:
        """Records an event and associated log entry, persisting both."""
        now = datetime.now(timezone.utc).isoformat()
        evt_type_val = event_type.value if hasattr(event_type, "value") else str(event_type)
        clean_msg = _sanitize_string(message, [self._hf_token])

        with self._lock:
            event = TrainingEvent(
                event_id=f"evt_{uuid.uuid4().hex[:12]}",
                run_id=self.run_id,
                timestamp=now,
                event_type=evt_type_val,
                severity=severity,
                message=clean_msg,
                global_step=step if step is not None else self.current_step,
                epoch=epoch if epoch is not None else self.current_epoch,
                training_loss=loss if loss is not None else self.current_loss,
                learning_rate=lr if lr is not None else getattr(self.config, "learning_rate", None),
                checkpoint_path=checkpoint_path or self.latest_checkpoint,
            )
            self.events.append(event)
            if checkpoint_path:
                self.latest_checkpoint = checkpoint_path

            log_entry = TrainingLogEntry(
                timestamp=now,
                level=severity.value.upper() if hasattr(severity, "value") else "INFO",
                message=clean_msg,
                stage=self.status.value,
            )
            self.logs.append(log_entry)

        try:
            self.repo.add_event(event)
            self.repo.add_log(self.run_id, log_entry)
            self.sync_to_db()
        except Exception as e:
            logger.warning(f"Error persisting training event: {e}")

        return event

    def add_log(self, level: str, message: str, stage: Optional[str] = None) -> TrainingLogEntry:
        """Appends a sanitized log entry for this job, persisting to SQLite."""
        now = datetime.now(timezone.utc).isoformat()
        clean_msg = _sanitize_string(message, [self._hf_token])
        log_entry = TrainingLogEntry(
            timestamp=now,
            level=level.upper(),
            message=clean_msg,
            stage=stage or self.status.value,
        )
        with self._lock:
            self.logs.append(log_entry)
        try:
            self.repo.add_log(self.run_id, log_entry)
        except Exception as e:
            logger.warning(f"Error persisting training log: {e}")
        return log_entry

    def sync_to_db(self) -> None:
        """Persists current state snapshot to the SQLite repository."""
        try:
            with self._lock:
                data = {
                    "run_id": self.run_id,
                    "project_name": self.config.project_name,
                    "model_identifier": self.config.model_identifier or "unknown",
                    "dataset_reference": self.config.dataset_reference or "unknown",
                    "training_method": self.config.training_method.value,
                    "status": self.status.value,
                    "config": self.config,
                    "created_at": self.created_at,
                    "started_at": self.started_at,
                    "completed_at": self.completed_at,
                    "elapsed_time_seconds": self.elapsed_time_seconds,
                    "current_step": self.current_step,
                    "max_steps": self.max_steps,
                    "current_epoch": self.current_epoch,
                    "num_epochs": self.num_epochs,
                    "current_loss": self.current_loss,
                    "learning_rate": getattr(self.config, "learning_rate", None),
                    "error_message": self.error_message,
                    "device": self.device,
                    "latest_checkpoint": self.latest_checkpoint,
                    "output_dir": self.output_dir,
                }
            self.repo.save_or_update_run(data)
        except Exception as e:
            logger.warning(f"Error syncing run to database: {e}")

    def request_cancellation(self) -> bool:
        """Requests cancellation if the job is in a cancellable state."""
        with self._lock:
            terminal_states = {
                TrainingRunStatus.COMPLETED,
                TrainingRunStatus.FAILED,
                TrainingRunStatus.CANCELLED,
            }
            if self.status in terminal_states:
                return False

            if not self._cancel_event.is_set():
                self._cancel_event.set()

            if self.status == TrainingRunStatus.QUEUED:
                self.status = TrainingRunStatus.CANCELLED
                self.completed_at = datetime.now(timezone.utc).isoformat()
                self.error_message = "Training run was cancelled before execution started."
                immediate_cancel = True
            else:
                immediate_cancel = False

        if immediate_cancel:
            self.emit_event(
                event_type=TrainingEventType.CANCELLATION_REQUESTED,
                severity=EventSeverity.WARNING,
                message="Cancellation requested by user.",
            )
            self.emit_event(
                event_type=TrainingEventType.CANCELLATION_COMPLETED,
                severity=EventSeverity.WARNING,
                message="Training run cancelled before start.",
            )
        else:
            self.emit_event(
                event_type=TrainingEventType.CANCELLATION_REQUESTED,
                severity=EventSeverity.WARNING,
                message="Cancellation requested by user. Terminating active training worker...",
            )

        return True

    def to_response(self) -> TrainingRunResponse:
        """Converts internal job state to typed TrainingRunResponse."""
        with self._lock:
            cancellable = self.status in {
                TrainingRunStatus.QUEUED,
                TrainingRunStatus.VALIDATING,
                TrainingRunStatus.LOADING_MODEL,
                TrainingRunStatus.PREPARING_DATASET,
                TrainingRunStatus.TRAINING,
            }
            cancellation_requested = self.is_cancelled and self.status != TrainingRunStatus.CANCELLED
            last_log_time = self.events[-1].timestamp if self.events else (self.completed_at or self.started_at or self.created_at)

            return TrainingRunResponse(
                run_id=self.run_id,
                status=self.status,
                config=self.config,
                created_at=self.created_at,
                started_at=self.started_at,
                completed_at=self.completed_at,
                current_step=self.current_step,
                max_steps=self.max_steps,
                current_epoch=self.current_epoch,
                num_epochs=self.num_epochs,
                current_loss=self.current_loss,
                loss_history=list(self.loss_history),
                elapsed_time_seconds=round(self.elapsed_time_seconds, 2),
                error_message=self.error_message,
                device=self.device,
                trainable_params=self.trainable_params,
                total_params=self.total_params,
                trainable_percent=self.trainable_percent,
                cancellable=cancellable,
                cancellation_requested=cancellation_requested,
                quantization_info=self.quantization_info,
                latest_checkpoint=self.latest_checkpoint,
                learning_rate=getattr(self.config, "learning_rate", None),
                last_log_time=last_log_time,
            )


class JobManager:
    """Manages training jobs lifecycle, concurrency, and SQLite persistence."""

    def __init__(
        self,
        runs_output_dir: Optional[str] = None,
        model_loader: Optional[ModelLoaderService] = None,
        dataset_service: Optional[TrainingDatasetService] = None,
        engine: Optional[TrainingEngine] = None,
        run_repo: Optional[RunRepository] = None,
        reconcile_on_init: bool = False,
    ) -> None:
        self.runs_output_dir = runs_output_dir or settings.runs_output_dir
        self.model_loader = model_loader or get_model_loader_service()
        self.dataset_service = dataset_service or get_training_dataset_service()
        self.engine = engine or TrainingEngine()
        self.run_repo = run_repo or get_run_repository()

        os.makedirs(self.runs_output_dir, exist_ok=True)
        self._jobs: Dict[str, TrainingJob] = {}
        self._active_threads: Dict[str, threading.Thread] = {}
        self._idempotency_map: Dict[str, str] = {}
        self._lock = threading.RLock()
        if reconcile_on_init:
            self.reconcile_interrupted_runs()

    def reconcile_interrupted_runs(self) -> int:
        """
        Reconciles orphaned runs from SQLite that were left in active/non-terminal
        states across a backend restart or shutdown. Marks them FAILED so they
        are not falsely reported as training when the process is not executing them.
        """
        active_states = [
            TrainingRunStatus.QUEUED.value,
            TrainingRunStatus.VALIDATING.value,
            TrainingRunStatus.LOADING_MODEL.value,
            TrainingRunStatus.PREPARING_DATASET.value,
            TrainingRunStatus.TRAINING.value,
            TrainingRunStatus.SAVING.value,
        ]
        reconciled = 0
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            for st in active_states:
                items, _ = self.run_repo.list_runs(status_filter=st, page=1, page_size=100)
                for item in items:
                    if item.run_id not in self._active_threads or not self._active_threads[item.run_id].is_alive():
                        persisted = self.run_repo.get_run(item.run_id)
                        if persisted:
                            persisted["status"] = TrainingRunStatus.FAILED.value
                            persisted["completed_at"] = now
                            persisted["error_message"] = "Training run was interrupted by backend server restart or process termination."
                            self.run_repo.save_or_update_run(persisted)
                            
                            fail_event = TrainingEvent(
                                event_id=f"evt_{uuid.uuid4().hex[:12]}",
                                run_id=item.run_id,
                                timestamp=now,
                                event_type=TrainingEventType.TRAINING_FAILED,
                                severity=EventSeverity.ERROR,
                                message="Training run was interrupted by backend server restart or process termination.",
                            )
                            self.run_repo.add_event(fail_event)
                            reconciled += 1
        return reconciled

    def create_job(
        self,
        config: TrainingConfigSchema,
        hf_token: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> TrainingJob:
        """
        Creates and registers a new training job.
        Validates configuration before registration.
        Guarantees duplicate submission prevention: returns existing job if idempotency_key is repeated.
        """
        if idempotency_key:
            with self._lock:
                if idempotency_key in self._idempotency_map:
                    existing_run_id = self._idempotency_map[idempotency_key]
                    existing_job = self._jobs.get(existing_run_id)
                    if existing_job:
                        return existing_job

        if config.training_method not in (TrainingMethod.LORA, TrainingMethod.QLORA):
            raise ValueError(
                f"Supported methods are LoRA and QLoRA fine-tuning. Requested: {config.training_method.value}"
            )

        if not config.model_identifier:
            raise ValueError("model_identifier must be provided in configuration.")

        if not config.dataset_reference:
            raise ValueError("dataset_reference must be provided in configuration.")

        with self._lock:
            active_statuses = {
                TrainingRunStatus.QUEUED,
                TrainingRunStatus.VALIDATING,
                TrainingRunStatus.LOADING_MODEL,
                TrainingRunStatus.PREPARING_DATASET,
                TrainingRunStatus.TRAINING,
            }
            for j in self._jobs.values():
                if j.status in active_statuses and (
                    j.config.project_name == config.project_name
                    and j.config.model_identifier == config.model_identifier
                    and j.config.dataset_reference == config.dataset_reference
                    and j.config.training_method == config.training_method
                ):
                    if idempotency_key:
                        self._idempotency_map[idempotency_key] = j.run_id
                    return j

        run_id = f"run_{uuid.uuid4().hex[:12]}"
        output_dir = os.path.join(self.runs_output_dir, run_id)

        job = TrainingJob(
            run_id=run_id,
            config=config,
            output_dir=output_dir,
            hf_token=hf_token,
            repo=self.run_repo,
        )

        with self._lock:
            self._jobs[run_id] = job
            if idempotency_key:
                self._idempotency_map[idempotency_key] = run_id

        # Initial persistence and event
        job.sync_to_db()
        job.emit_event(
            event_type=TrainingEventType.RUN_QUEUED,
            severity=EventSeverity.INFO,
            message=f"Training run {run_id} queued for project '{config.project_name}'.",
        )

        return job

    def get_job(self, run_id: str) -> Optional[TrainingJob]:
        """Retrieves job by run_id from memory or recovers it from persistent repository."""
        with self._lock:
            if run_id in self._jobs:
                return self._jobs[run_id]

        persisted = self.run_repo.get_run(run_id)
        if persisted:
            job = self._reconstruct_job_from_persisted(persisted)
            with self._lock:
                self._jobs[run_id] = job
            return job
        return None

    def _reconstruct_job_from_persisted(self, row: Dict[str, Any]) -> TrainingJob:
        """Rebuilds a TrainingJob instance from persisted SQLite state."""
        cfg_dict = row.get("config", {})
        if isinstance(cfg_dict, dict):
            clean_cfg = {}
            for k, v in cfg_dict.items():
                if isinstance(v, str) and "." in v:
                    for prefix in ("TrainingMethod.", "TrainingLengthType.", "QuantizationType."):
                        if prefix in v:
                            v = v.split(".")[-1].lower()
                clean_cfg[k] = v
            cfg_dict = clean_cfg

        try:
            config = TrainingConfigSchema(**cfg_dict) if cfg_dict else TrainingConfigSchema()
        except Exception:
            config = TrainingConfigSchema()

        output_dir = row.get("output_dir") or os.path.join(self.runs_output_dir, row["run_id"])
        job = TrainingJob(
            run_id=row["run_id"],
            config=config,
            output_dir=output_dir,
            repo=self.run_repo,
        )
        job.status = TrainingRunStatus(row["status"])
        job.created_at = row["created_at"]
        job.started_at = row["started_at"]
        job.completed_at = row["completed_at"]
        job.elapsed_time_seconds = float(row.get("elapsed_time_seconds", 0.0) or 0.0)
        job.current_step = int(row.get("current_step", 0) or 0)
        job.max_steps = int(row.get("max_steps", 0) or 0)
        job.current_epoch = float(row.get("current_epoch", 0.0) or 0.0)
        job.num_epochs = float(row.get("num_epochs", 0.0) or 0.0)
        job.current_loss = row.get("current_loss")
        job.error_message = row.get("error_message")
        job.device = row.get("device", "cpu")
        job.latest_checkpoint = row.get("latest_checkpoint")

        # Load historical loss metrics from metrics.json if present
        metrics_path = os.path.join(output_dir, "metrics.json")
        if os.path.isfile(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    job.loss_history = [TrainingMetricPoint(**item) for item in data]
            except Exception:
                job.loss_history = []

        return job

    def list_jobs(self) -> List[TrainingJob]:
        """Retrieves all registered in-memory jobs."""
        with self._lock:
            return list(self._jobs.values())

    def cancel_job(self, run_id: str) -> bool:
        """Requests cancellation for a running or queued job."""
        job = self.get_job(run_id)
        if not job:
            return False
        return job.request_cancellation()

    def start_job(self, run_id: str, background: bool = True) -> TrainingJob:
        """Starts executing the job lifecycle either in a background thread or synchronously."""
        with self._lock:
            job = self.get_job(run_id)
            if not job:
                raise KeyError(f"Training run '{run_id}' not found.")

            # Duplicate worker launch prevention:
            if run_id in self._active_threads and self._active_threads[run_id].is_alive():
                logger.warning(f"Training worker thread for '{run_id}' is already running. Duplicate start ignored.")
                return job

            if job.status != TrainingRunStatus.QUEUED:
                logger.warning(f"Cannot start training run '{run_id}' because status is already '{job.status.value}'.")
                return job

            if background:
                thread = threading.Thread(
                    target=self._execute_job_lifecycle,
                    args=(job,),
                    daemon=True,
                    name=f"TrainingRunner-{run_id}",
                )
                self._active_threads[run_id] = thread
                thread.start()
            else:
                self._execute_job_lifecycle(job)

        return job

    def _execute_job_lifecycle(self, job: TrainingJob) -> None:
        """Executes the full training lifecycle stages with strict event and cancellation handling."""
        start_time = time.time()
        job.started_at = datetime.now(timezone.utc).isoformat()
        secrets = [job._hf_token]

        try:
            # 1. VALIDATING STAGE
            job.status = TrainingRunStatus.VALIDATING
            job.emit_event(
                event_type=TrainingEventType.VALIDATION_STARTED,
                severity=EventSeverity.INFO,
                message="Starting configuration and output storage validation...",
            )

            if job.is_cancelled:
                raise TrainingCancelledException("Run cancelled during validation.")

            try:
                os.makedirs(job.output_dir, exist_ok=True)
                test_file = os.path.join(job.output_dir, ".write_test")
                with open(test_file, "w") as f:
                    f.write("ok")
                os.remove(test_file)
            except Exception as e:
                raise PermissionError(f"Cannot write to output directory '{job.output_dir}': {e}") from e

            job.emit_event(
                event_type=TrainingEventType.VALIDATION_COMPLETED,
                severity=EventSeverity.INFO,
                message="Configuration and storage validation succeeded.",
            )

            # 2. LOADING_MODEL STAGE
            job.status = TrainingRunStatus.LOADING_MODEL
            job.emit_event(
                event_type=TrainingEventType.MODEL_LOADING_STARTED,
                severity=EventSeverity.INFO,
                message=f"Loading base model '{job.config.model_identifier}' and tokenizer...",
            )

            if job.is_cancelled:
                raise TrainingCancelledException("Run cancelled before model loading.")

            is_qlora = (job.config.training_method == TrainingMethod.QLORA)
            quant_cfg = job.config.get_quantization_config() if is_qlora else None

            model, tokenizer, device_used = self.model_loader.load_model_and_tokenizer(
                model_identifier=job.config.model_identifier,
                hf_token=job._hf_token,
                quantization_config=quant_cfg,
                is_qlora=is_qlora,
            )
            job.device = device_used

            if job.is_cancelled:
                raise TrainingCancelledException("Run cancelled after model loading.")

            if is_qlora:
                peft_model, param_summary = prepare_and_inject_qlora_adapters(
                    base_model=model,
                    config=job.config,
                )
            else:
                peft_model, param_summary = inject_lora_adapters(
                    base_model=model,
                    config=job.config,
                )
            job.trainable_params = param_summary.trainable_params
            job.total_params = param_summary.total_params
            job.trainable_percent = param_summary.trainable_percent

            job.emit_event(
                event_type=TrainingEventType.MODEL_LOADING_COMPLETED,
                severity=EventSeverity.INFO,
                message=(
                    f"Model and adapters loaded on device '{device_used}'. "
                    f"Trainable params: {param_summary.trainable_params:,} ({param_summary.trainable_percent:.3f}%)"
                ),
            )

            # 3. PREPARING_DATASET STAGE
            job.status = TrainingRunStatus.PREPARING_DATASET
            job.emit_event(
                event_type=TrainingEventType.DATASET_PREP_STARTED,
                severity=EventSeverity.INFO,
                message=f"Tokenizing and preparing dataset '{job.config.dataset_reference}'...",
            )

            if job.is_cancelled:
                raise TrainingCancelledException("Run cancelled before dataset preparation.")

            prepared_dataset = self.dataset_service.prepare_dataset(
                dataset_id=job.config.dataset_reference,
                tokenizer=tokenizer,
                config=job.config,
            )

            job.emit_event(
                event_type=TrainingEventType.DATASET_PREP_COMPLETED,
                severity=EventSeverity.INFO,
                message=f"Dataset prepared with {prepared_dataset.train_count} train samples.",
            )

            # 4. TRAINING STAGE
            job.status = TrainingRunStatus.TRAINING
            job.emit_event(
                event_type=TrainingEventType.TRAINING_STARTED,
                severity=EventSeverity.INFO,
                message=f"Starting optimization loop on device '{job.device}'. Target steps: {job.max_steps}.",
            )

            if job.is_cancelled:
                raise TrainingCancelledException("Run cancelled before training steps.")

            configured_logging_steps = getattr(job.config, "logging_steps", None) or 1
            logging_interval = max(1, configured_logging_steps)

            def step_callback(step: int, loss: float, epoch: float, elapsed: float) -> None:
                with job._lock:
                    job.current_step = step
                    job.current_loss = loss
                    job.current_epoch = epoch
                    job.elapsed_time_seconds = elapsed

                if step % logging_interval == 0 or step == job.max_steps:
                    job.emit_event(
                        event_type=TrainingEventType.TRAINING_STEP,
                        severity=EventSeverity.INFO,
                        message=f"Step {step}/{job.max_steps} — loss: {loss:.4f}, epoch: {epoch:.2f}",
                        step=step,
                        epoch=epoch,
                        loss=loss,
                        lr=getattr(job.config, "learning_rate", None),
                    )

            loss_history = self.engine.train(
                model=peft_model,
                prepared_dataset=prepared_dataset,
                config=job.config,
                is_cancelled_fn=lambda: job.is_cancelled,
                on_step_callback=step_callback,
            )
            job.loss_history = loss_history

            # 5. SAVING STAGE
            job.status = TrainingRunStatus.SAVING
            job.emit_event(
                event_type=TrainingEventType.CHECKPOINT_SAVED,
                severity=EventSeverity.INFO,
                message="Saving fine-tuned adapter weights and metadata...",
            )

            sanitized_cfg = job.config.model_dump()
            sanitized_cfg.pop("hf_token", None)

            saved_files = save_and_verify_adapter(
                model=peft_model,
                output_directory=job.output_dir,
                sanitized_config=sanitized_cfg,
                quantization_config=job.quantization_info,
            )
            job.artifact_files = saved_files

            # Save metrics.json
            metrics_path = os.path.join(job.output_dir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump([m.model_dump() for m in loss_history], f, indent=2)
            if "metrics.json" not in job.artifact_files:
                job.artifact_files.append("metrics.json")

            # Save run_metadata.json
            run_meta = {
                "run_id": job.run_id,
                "training_method": job.config.training_method.value,
                "quantization": job.quantization_info,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "device": job.device,
                "trainable_params": job.trainable_params,
                "total_params": job.total_params,
                "trainable_percent": job.trainable_percent,
                "final_loss": job.current_loss,
                "total_steps": job.current_step,
            }
            meta_path = os.path.join(job.output_dir, "run_metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(run_meta, f, indent=2)
            if "run_metadata.json" not in job.artifact_files:
                job.artifact_files.append("run_metadata.json")

            adapter_cfg_path = os.path.join(job.output_dir, "adapter_config.json")
            if os.path.isfile(adapter_cfg_path):
                with open(adapter_cfg_path, "r", encoding="utf-8") as f:
                    job.adapter_config = json.load(f)

            # Verification: Reload adapter on base model to guarantee integrity
            reload_lora_adapter(model, job.output_dir)

            # 6. COMPLETED
            job.elapsed_time_seconds = time.time() - start_time
            job.completed_at = datetime.now(timezone.utc).isoformat()
            job.status = TrainingRunStatus.COMPLETED
            job.latest_checkpoint = "final_adapter"

            job.emit_event(
                event_type=TrainingEventType.TRAINING_COMPLETED,
                severity=EventSeverity.INFO,
                message=f"Training run {job.run_id} completed successfully in {job.elapsed_time_seconds:.1f}s.",
                checkpoint_path="final_adapter",
            )
            logger.info(f"Training run {job.run_id} completed successfully.")

        except TrainingCancelledException as e:
            job.status = TrainingRunStatus.CANCELLED
            job.completed_at = datetime.now(timezone.utc).isoformat()
            job.elapsed_time_seconds = time.time() - start_time
            clean_err = _sanitize_string(str(e), secrets)
            job.error_message = clean_err

            job.emit_event(
                event_type=TrainingEventType.CANCELLATION_COMPLETED,
                severity=EventSeverity.WARNING,
                message=f"Training run cancelled: {clean_err}",
            )
            logger.warning(f"Training run {job.run_id} was cancelled: {clean_err}")

        except Exception as e:
            job.status = TrainingRunStatus.FAILED
            job.completed_at = datetime.now(timezone.utc).isoformat()
            job.elapsed_time_seconds = time.time() - start_time
            clean_err = _sanitize_string(str(e), secrets)
            job.error_message = clean_err

            job.emit_event(
                event_type=TrainingEventType.TRAINING_FAILED,
                severity=EventSeverity.ERROR,
                message=f"Training run failed: {clean_err}",
            )
            logger.error(f"Training run {job.run_id} failed: {clean_err}", exc_info=True)
        finally:
            with self._lock:
                self._active_threads.pop(job.run_id, None)

    def get_artifacts_metadata(self, run_id: str) -> Optional[TrainingArtifactMetadata]:
        """Retrieves and validates artifact metadata for a completed run."""
        job = self.get_job(run_id)
        if not job or job.status != TrainingRunStatus.COMPLETED:
            return None

        if not os.path.isdir(job.output_dir):
            return None

        files = sorted(os.listdir(job.output_dir))
        if "adapter_config.json" not in files:
            return None

        safe_rel_dir = os.path.join("artifacts", "runs", run_id)

        return TrainingArtifactMetadata(
            run_id=job.run_id,
            artifact_directory=safe_rel_dir,
            files=files,
            adapter_config=job.adapter_config,
            quantization_config=job.quantization_info,
            trainable_params=job.trainable_params or 0,
            total_params=job.total_params or 0,
            trainable_percent=job.trainable_percent or 0.0,
            is_valid=True,
            verification_status="verified",
            created_at=job.completed_at or job.created_at,
        )

    def get_checkpoints(self, run_id: str) -> List[CheckpointMetadata]:
        """
        Discovers verified checkpoints for a run on disk.
        Safely inspects output directory without exposing internal filesystem paths.
        """
        job = self.get_job(run_id)
        output_dir = job.output_dir if job else os.path.join(self.runs_output_dir, run_id)
        if not os.path.isdir(output_dir):
            return []

        checkpoints: List[CheckpointMetadata] = []

        # 1. Final saved adapter
        adapter_cfg_file = os.path.join(output_dir, "adapter_config.json")
        if os.path.isfile(adapter_cfg_file):
            try:
                files = sorted(os.listdir(output_dir))
                total_size = sum(
                    os.path.getsize(os.path.join(output_dir, f))
                    for f in files
                    if os.path.isfile(os.path.join(output_dir, f))
                )
                created_iso = datetime.fromtimestamp(
                    os.path.getmtime(adapter_cfg_file),
                    tz=timezone.utc,
                ).isoformat()

                checkpoints.append(
                    CheckpointMetadata(
                        checkpoint_id="final_adapter",
                        run_id=run_id,
                        step=job.current_step if job else None,
                        epoch=job.current_epoch if job else None,
                        path=os.path.join("artifacts", "runs", run_id),
                        created_at=created_iso,
                        size_bytes=total_size,
                        files=files,
                        is_valid=True,
                    )
                )
            except Exception as e:
                logger.warning(f"Error checking final adapter checkpoint: {e}")

        # 2. Check for intermediate checkpoints
        try:
            entries = sorted(os.listdir(output_dir))
        except Exception:
            entries = []

        for entry in entries:
            if entry.startswith("checkpoint-"):
                cp_dir = os.path.join(output_dir, entry)
                if os.path.isdir(cp_dir):
                    try:
                        cp_files = sorted(os.listdir(cp_dir))
                        cp_size = sum(
                            os.path.getsize(os.path.join(cp_dir, f))
                            for f in cp_files
                            if os.path.isfile(os.path.join(cp_dir, f))
                        )
                        cp_time = datetime.fromtimestamp(
                            os.path.getmtime(cp_dir),
                            tz=timezone.utc,
                        ).isoformat()
                        step_val = None
                        try:
                            step_val = int(entry.replace("checkpoint-", ""))
                        except ValueError:
                            pass

                        checkpoints.append(
                            CheckpointMetadata(
                                checkpoint_id=entry,
                                run_id=run_id,
                                step=step_val,
                                epoch=None,
                                path=os.path.join("artifacts", "runs", run_id, entry),
                                created_at=cp_time,
                                size_bytes=cp_size,
                                files=cp_files,
                                is_valid=True,
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Error checking intermediate checkpoint {entry}: {e}")

        return checkpoints


_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Returns singleton JobManager instance."""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager
