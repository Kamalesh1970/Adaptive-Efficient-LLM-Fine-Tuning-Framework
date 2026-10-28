"""
Trainer Callback and Metric Extraction Service for Phase 7.
Integrates with Hugging Face Trainer callbacks or custom training loops.
Provides real metric extraction, progress calculation with denominator validation,
and lifecycle event dispatching without metric fabrication.
"""

import math
from typing import Dict, Any, Optional, Callable
from backend.app.schemas.history import TrainingEventType, EventSeverity


def calculate_progress(
    current_step: Optional[int],
    max_steps: Optional[int],
) -> Optional[float]:
    """
    Calculates percentage progress strictly when denominator is valid.
    Never fabricates 100% or produces NaN/Inf.
    Returns None if denominator is missing, zero, negative, or step is missing.
    """
    if max_steps is None or max_steps <= 0:
        return None
    if current_step is None or current_step < 0:
        return None

    pct = (current_step / max_steps) * 100.0
    # Clamped between 0.0 and 100.0
    return round(min(100.0, max(0.0, pct)), 2)


def extract_trainer_metrics(
    logs: Optional[Dict[str, Any]] = None,
    state: Any = None,
) -> Dict[str, Any]:
    """
    Extracts actual training metrics from Hugging Face Trainer logs and state.
    Returns clean dictionary with real values or None when unavailable.
    Never fabricates metrics.
    """
    result: Dict[str, Any] = {
        "global_step": None,
        "max_steps": None,
        "epoch": None,
        "training_loss": None,
        "learning_rate": None,
    }

    # Extract state attributes if available
    if state is not None:
        step = getattr(state, "global_step", None)
        if isinstance(step, int) and step >= 0:
            result["global_step"] = step

        max_s = getattr(state, "max_steps", None)
        if isinstance(max_s, int) and max_s > 0:
            result["max_steps"] = max_s

        ep = getattr(state, "epoch", None)
        if isinstance(ep, (int, float)) and not math.isnan(ep):
            result["epoch"] = round(float(ep), 3)

    # Extract logs dictionary
    if logs and isinstance(logs, dict):
        # Loss extraction: 'loss' or 'train_loss'
        loss_val = logs.get("loss", logs.get("train_loss"))
        if isinstance(loss_val, (int, float)) and not math.isnan(loss_val) and not math.isinf(loss_val):
            result["training_loss"] = round(float(loss_val), 4)

        # Learning rate extraction: 'learning_rate' or 'lr'
        lr_val = logs.get("learning_rate", logs.get("lr"))
        if isinstance(lr_val, (int, float)) and not math.isnan(lr_val) and not math.isinf(lr_val) and lr_val >= 0:
            result["learning_rate"] = float(lr_val)

        # Epoch in logs overrides or provides fallback
        if result["epoch"] is None:
            ep = logs.get("epoch")
            if isinstance(ep, (int, float)) and not math.isnan(ep):
                result["epoch"] = round(float(ep), 3)

        # Step in logs
        if result["global_step"] is None:
            st = logs.get("step")
            if isinstance(st, int) and st >= 0:
                result["global_step"] = st

    return result


class StudioTrainerCallback:
    """
    Hugging Face Trainer compatible callback for telemetry and event dispatching.
    Safely integrates when transformers is available; functions as standalone handler if not.
    """

    def __init__(
        self,
        on_step_event_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_save_event_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.on_step_event_fn = on_step_event_fn
        self.on_save_event_fn = on_save_event_fn
        self.last_metrics: Dict[str, Any] = {}

    def on_log(self, args: Any, state: Any, control: Any, logs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Called by Hugging Face Trainer whenever a logging step occurs."""
        metrics = extract_trainer_metrics(logs=logs, state=state)
        self.last_metrics = metrics
        if self.on_step_event_fn:
            self.on_step_event_fn(metrics)

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Called by Hugging Face Trainer whenever a checkpoint is saved."""
        step = getattr(state, "global_step", None)
        cp_name = f"checkpoint-{step}" if step is not None else "checkpoint"
        if self.on_save_event_fn:
            self.on_save_event_fn(cp_name)
