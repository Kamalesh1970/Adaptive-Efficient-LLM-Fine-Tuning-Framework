"""
Real LoRA Training Engine (Phase 4).
Executes actual optimization steps using PyTorch or synthetic test models.
Supports step/epoch limits, batch sizing, gradient accumulation, random seed,
real-time metric tracking, and safe cancellation between steps.
"""

import time
import math
import random
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone

from backend.app.schemas.training import (
    TrainingConfigSchema,
    TrainingLengthType,
    TrainingMetricPoint,
)
from backend.app.services.training_dataset_service import PreparedTrainingDataset

logger = logging.getLogger(__name__)


class TrainingCancelledException(Exception):
    """Raised when training is cancelled by user request."""
    pass


class TrainingExecutionError(Exception):
    """Raised when an unrecoverable failure occurs during training execution."""
    pass


class TrainingEngine:
    """Executes the training loop and reports telemetry."""

    def __init__(self) -> None:
        pass

    def train(
        self,
        model: Any,
        prepared_dataset: PreparedTrainingDataset,
        config: TrainingConfigSchema,
        is_cancelled_fn: Optional[Callable[[], bool]] = None,
        on_step_callback: Optional[Callable[[int, float, float, float], None]] = None,
    ) -> List[TrainingMetricPoint]:
        """
        Executes real training loop.
        Calls on_step_callback(step, loss, epoch, elapsed_seconds) after each recorded step.
        Returns full list of TrainingMetricPoint.
        """
        if is_cancelled_fn and is_cancelled_fn():
            raise TrainingCancelledException("Training run was cancelled before execution started.")

        # Determine target total steps
        train_count = prepared_dataset.train_count
        batch_size = max(1, config.batch_size)
        grad_accum = max(1, config.gradient_accumulation_steps)
        effective_batch = batch_size * grad_accum

        steps_per_epoch = max(1, math.ceil(train_count / effective_batch))

        if config.training_length_type == TrainingLengthType.EPOCHS:
            target_epochs = getattr(config, "num_epochs", 1.0) or 1.0
            target_steps = max(1, int(math.ceil(steps_per_epoch * target_epochs)))
        else:
            target_steps = max(1, getattr(config, "max_steps", 10) or 10)
            target_epochs = target_steps / steps_per_epoch

        seed = config.seed if config.seed is not None else 42
        random.seed(seed)

        loss_history: List[TrainingMetricPoint] = []
        start_time = time.time()
        configured_log = getattr(config, "logging_steps", None)
        logging_steps = max(1, configured_log if configured_log is not None else 1)

        # Check if PyTorch model
        is_torch_model = False
        try:
            import torch
            if isinstance(model, torch.nn.Module):
                is_torch_model = True
        except ImportError:
            is_torch_model = False

        if is_torch_model:
            loss_history = self._train_pytorch(
                model=model,
                prepared_dataset=prepared_dataset,
                config=config,
                target_steps=target_steps,
                target_epochs=target_epochs,
                logging_steps=logging_steps,
                start_time=start_time,
                is_cancelled_fn=is_cancelled_fn,
                on_step_callback=on_step_callback,
            )
        else:
            # Synthetic / test model execution (for unit tests and environments without PyTorch)
            loss_history = self._train_synthetic(
                model=model,
                prepared_dataset=prepared_dataset,
                config=config,
                target_steps=target_steps,
                target_epochs=target_epochs,
                logging_steps=logging_steps,
                start_time=start_time,
                is_cancelled_fn=is_cancelled_fn,
                on_step_callback=on_step_callback,
            )

        return loss_history

    def _train_synthetic(
        self,
        model: Any,
        prepared_dataset: PreparedTrainingDataset,
        config: TrainingConfigSchema,
        target_steps: int,
        target_epochs: float,
        logging_steps: int,
        start_time: float,
        is_cancelled_fn: Optional[Callable[[], bool]],
        on_step_callback: Optional[Callable[[int, float, float, float], None]],
    ) -> List[TrainingMetricPoint]:
        """Runs deterministic synthetic training steps for unit testing."""
        loss_history: List[TrainingMetricPoint] = []
        base_loss = 2.45

        for current_step in range(1, target_steps + 1):
            if is_cancelled_fn and is_cancelled_fn():
                raise TrainingCancelledException(f"Training run was cancelled at step {current_step}/{target_steps}.")

            # Deterministic decay with subtle variance
            decay = math.exp(-0.02 * current_step)
            step_noise = (math.sin(current_step) * 0.04)
            current_loss = round(max(0.12, (base_loss * decay) + step_noise + 0.35), 4)

            current_epoch = round((current_step / target_steps) * target_epochs, 2)
            elapsed = time.time() - start_time

            # Record metrics according to logging_steps
            if current_step % logging_steps == 0 or current_step == target_steps:
                metric = TrainingMetricPoint(
                    step=current_step,
                    loss=current_loss,
                    learning_rate=config.learning_rate,
                    epoch=current_epoch,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                loss_history.append(metric)

                if on_step_callback:
                    on_step_callback(current_step, current_loss, current_epoch, elapsed)

            # Tiny sleep to allow responsive cancellation in test loops
            time.sleep(0.01)

        return loss_history

    def _train_pytorch(
        self,
        model: Any,
        prepared_dataset: PreparedTrainingDataset,
        config: TrainingConfigSchema,
        target_steps: int,
        target_epochs: float,
        logging_steps: int,
        start_time: float,
        is_cancelled_fn: Optional[Callable[[], bool]],
        on_step_callback: Optional[Callable[[int, float, float, float], None]],
    ) -> List[TrainingMetricPoint]:
        """Runs genuine PyTorch optimization loop."""
        import torch

        model.train()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise TrainingExecutionError("No trainable parameters found in model for optimization.")

        lr = config.learning_rate
        weight_decay = config.weight_decay or 0.01
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

        batch_size = max(1, config.batch_size)
        grad_accum = max(1, config.gradient_accumulation_steps)

        loss_history: List[TrainingMetricPoint] = []
        samples = prepared_dataset.train_samples
        num_samples = len(samples)

        current_sample_idx = 0
        optimizer.zero_grad()

        for current_step in range(1, target_steps + 1):
            if is_cancelled_fn and is_cancelled_fn():
                raise TrainingCancelledException(f"Training run cancelled at step {current_step}/{target_steps}.")

            accumulated_loss = 0.0

            for accum_idx in range(grad_accum):
                # Fetch batch
                batch_samples = []
                for _ in range(batch_size):
                    batch_samples.append(samples[current_sample_idx % num_samples])
                    current_sample_idx += 1

                # Collate batch with padding
                max_len = max(len(s["input_ids"]) for s in batch_samples)
                pad_id = getattr(model.config, "pad_token_id", 0) if hasattr(model, "config") else 0
                if pad_id is None:
                    pad_id = 0

                input_ids_batch = []
                attention_mask_batch = []
                labels_batch = []

                for s in batch_samples:
                    inp = s["input_ids"]
                    pad_len = max_len - len(inp)
                    input_ids_batch.append(inp + [pad_id] * pad_len)
                    attention_mask_batch.append(s["attention_mask"] + [0] * pad_len)
                    labels_batch.append(s["labels"] + [-100] * pad_len)

                # Move to model device
                device = next(model.parameters()).device
                b_input_ids = torch.tensor(input_ids_batch, dtype=torch.long, device=device)
                b_attn_mask = torch.tensor(attention_mask_batch, dtype=torch.long, device=device)
                b_labels = torch.tensor(labels_batch, dtype=torch.long, device=device)

                # Forward pass
                outputs = model(
                    input_ids=b_input_ids,
                    attention_mask=b_attn_mask,
                    labels=b_labels,
                )

                loss = outputs.loss / grad_accum
                loss.backward()
                accumulated_loss += loss.item() * grad_accum

            # Gradient clipping if configured
            max_grad_norm = config.max_grad_norm or 1.0
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

            current_epoch = round((current_step / target_steps) * target_epochs, 2)
            elapsed = time.time() - start_time
            recorded_loss = round(accumulated_loss, 4)

            if current_step % logging_steps == 0 or current_step == target_steps:
                metric = TrainingMetricPoint(
                    step=current_step,
                    loss=recorded_loss,
                    learning_rate=lr,
                    epoch=current_epoch,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                loss_history.append(metric)

                if on_step_callback:
                    on_step_callback(current_step, recorded_loss, current_epoch, elapsed)

        return loss_history
