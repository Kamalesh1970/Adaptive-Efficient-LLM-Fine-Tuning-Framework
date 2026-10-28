"""
Pydantic Schemas for Phase 7: Real-Time Training Monitoring, Logs, Checkpoints, and Run History.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class TrainingEventType(str, Enum):
    """Event types emitted during training run lifecycle."""
    RUN_QUEUED = "run_queued"
    VALIDATION_STARTED = "validation_started"
    VALIDATION_COMPLETED = "validation_completed"
    MODEL_LOADING_STARTED = "model_loading_started"
    MODEL_LOADING_COMPLETED = "model_loading_completed"
    DATASET_PREP_STARTED = "dataset_prep_started"
    DATASET_PREP_COMPLETED = "dataset_prep_completed"
    TRAINING_STARTED = "training_started"
    TRAINING_STEP = "training_step"
    CHECKPOINT_SAVED = "checkpoint_saved"
    CANCELLATION_REQUESTED = "cancellation_requested"
    CANCELLATION_COMPLETED = "cancellation_completed"
    TRAINING_COMPLETED = "training_completed"
    TRAINING_FAILED = "training_failed"


class EventSeverity(str, Enum):
    """Severity classification for training events and logs."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class TrainingEvent(BaseModel):
    """
    Standardized event record emitted during the training lifecycle.
    """
    event_id: str = Field(description="Unique event identifier")
    run_id: str = Field(description="Associated training run ID")
    timestamp: str = Field(description="UTC timestamp in ISO format")
    event_type: str = Field(description="Lifecycle event identifier")
    severity: EventSeverity = Field(default=EventSeverity.INFO, description="Event severity classification")
    message: str = Field(description="Sanitized, human-readable event description")
    global_step: Optional[int] = Field(default=None, description="Actual global step reached, if applicable")
    epoch: Optional[float] = Field(default=None, description="Actual epoch progress, if applicable")
    training_loss: Optional[float] = Field(default=None, description="Actual training loss at this step, if applicable")
    learning_rate: Optional[float] = Field(default=None, description="Actual learning rate at this step, if applicable")
    checkpoint_path: Optional[str] = Field(default=None, description="Relative checkpoint path or identifier, if applicable")


class TrainingEventListResponse(BaseModel):
    """Paginated or cursor-based list of training events."""
    run_id: str = Field(description="Associated training run ID")
    events: List[TrainingEvent] = Field(default_factory=list, description="List of recorded events")
    total: int = Field(description="Total events available for this run")
    has_more: bool = Field(default=False, description="Whether additional events exist past the current cursor")
    next_cursor: Optional[str] = Field(default=None, description="Cursor for the next batch of events")


class TrainingLogEntry(BaseModel):
    """Sanitized log entry from the training worker."""
    timestamp: str = Field(description="UTC timestamp of log entry")
    level: str = Field(default="INFO", description="Log level: INFO, WARNING, ERROR")
    message: str = Field(description="Sanitized log message")
    stage: Optional[str] = Field(default=None, description="Lifecycle stage when logged")


class TrainingLogListResponse(BaseModel):
    """List of sanitized training logs."""
    run_id: str = Field(description="Associated training run ID")
    logs: List[TrainingLogEntry] = Field(default_factory=list, description="Chronological log entries")
    total: int = Field(description="Total log lines available")


class CheckpointMetadata(BaseModel):
    """Metadata for a verified training checkpoint on disk."""
    checkpoint_id: str = Field(description="Checkpoint name or identifier (e.g. checkpoint-100 or final_adapter)")
    run_id: str = Field(description="Associated training run ID")
    step: Optional[int] = Field(default=None, description="Global step when checkpoint was taken")
    epoch: Optional[float] = Field(default=None, description="Epoch when checkpoint was taken")
    path: str = Field(description="Safe relative path to checkpoint directory")
    created_at: str = Field(description="ISO creation timestamp")
    size_bytes: int = Field(default=0, description="Total size in bytes of files in checkpoint")
    files: List[str] = Field(default_factory=list, description="List of checkpoint filenames")
    is_valid: bool = Field(default=True, description="Whether checkpoint directory is valid and readable")


class CheckpointListResponse(BaseModel):
    """List of verified checkpoints for a run."""
    run_id: str = Field(description="Associated training run ID")
    checkpoints: List[CheckpointMetadata] = Field(default_factory=list, description="Available checkpoints")
    total: int = Field(description="Count of checkpoints found")


class RunHistoryItem(BaseModel):
    """Summary item representing a persisted training run in history."""
    run_id: str = Field(description="Unique training run identifier")
    project_name: str = Field(description="Project name configured for the run")
    model_identifier: str = Field(description="Target model identifier")
    dataset_reference: str = Field(description="Target dataset reference")
    training_method: str = Field(description="Fine-tuning method: lora, qlora, full_finetuning")
    status: str = Field(description="Current or final status of the run")
    created_at: str = Field(description="ISO creation timestamp")
    started_at: Optional[str] = Field(default=None, description="ISO timestamp when training started")
    completed_at: Optional[str] = Field(default=None, description="ISO timestamp when run finished or terminated")
    elapsed_time_seconds: float = Field(default=0.0, description="Total elapsed runtime in seconds")
    current_step: int = Field(default=0, description="Last recorded global step")
    max_steps: int = Field(default=0, description="Configured total steps")
    current_epoch: Optional[float] = Field(default=None, description="Last recorded epoch")
    num_epochs: Optional[float] = Field(default=None, description="Configured total epochs")
    current_loss: Optional[float] = Field(default=None, description="Latest recorded training loss")
    learning_rate: Optional[float] = Field(default=None, description="Latest reported learning rate")
    error_message: Optional[str] = Field(default=None, description="Sanitized error summary if failed/cancelled")
    device: str = Field(default="cpu", description="Execution device")
    latest_checkpoint: Optional[str] = Field(default=None, description="Identifier of latest checkpoint")


class RunHistoryListResponse(BaseModel):
    """Paginated list of persistent training runs."""
    runs: List[RunHistoryItem] = Field(default_factory=list, description="List of persisted runs")
    total: int = Field(description="Total matching runs count")
    page: int = Field(default=1, description="Current page number (1-indexed)")
    page_size: int = Field(default=20, description="Number of items per page")
    total_pages: int = Field(default=1, description="Total number of pages")
