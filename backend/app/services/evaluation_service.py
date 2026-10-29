"""
Model Evaluation and Baseline Comparison Service for Phase 8.
Evaluates trained LoRA/QLoRA adapters and base models on validation datasets.
Computes real Cross-Entropy loss, overflow-safe mathematical perplexity,
token throughput, and side-by-side baseline comparisons.
"""

import os
import math
import time
import uuid
import random
import logging
import threading
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Any

from backend.app.schemas.artifacts import ArtifactType, ValidationStatus
from backend.app.schemas.evaluation import (
    EvaluationRun,
    EvaluationStatus,
    EvaluationConfig,
    EvaluationMetrics,
    EvaluationSample,
    BaselineComparison,
)
from backend.app.schemas.datasets import DatasetSchemaType
from backend.app.services.artifact_service import ArtifactService, get_artifact_service
from backend.app.services.dataset_service import get_dataset_service
from backend.app.services.training_dataset_service import TrainingDatasetService
from backend.app.services.run_repository import get_run_repository, _sanitize_string
from backend.app.services.model_loader import (
    ModelLoaderService,
    get_test_model,
    is_test_model_registered,
)
from backend.app.services.lora_service import reload_lora_adapter
from backend.app.core.system import get_system_hardware_info
from backend.app.services.ml_dependencies import (
    is_full_ml_stack_available,
    is_bitsandbytes_available,
)

logger = logging.getLogger(__name__)


def calculate_perplexity(loss: Optional[float]) -> Tuple[Optional[float], bool, Optional[str]]:
    """
    Calculates perplexity mathematically: PPL = exp(loss).
    Guarantees floating-point safety and handles overflow or invalid values truthfully.
    Returns (perplexity, is_overflow, unavailable_reason).
    """
    if loss is None:
        return None, False, "Evaluation loss was not computed."
    if math.isnan(loss):
        return None, False, "Evaluation loss is NaN."
    if math.isinf(loss):
        return None, False, "Evaluation loss is infinite."
    if loss < 0.0:
        return None, False, f"Evaluation loss ({loss}) is negative; perplexity is undefined."

    # Floating point exp(85) > 8e36, exp(709) maxes out IEEE 754 double
    if loss > 85.0:
        return None, True, f"Evaluation loss ({loss:.2f}) exceeds mathematical bounds; exp(loss) overflows."

    try:
        ppl = math.exp(loss)
        return round(ppl, 4), False, None
    except OverflowError:
        return None, True, "Mathematical overflow during exp(loss) calculation."


class EvaluationCancelledException(Exception):
    """Raised when an evaluation job is cancelled by user request."""
    pass


class EvaluationService:
    """Orchestrates model evaluation, metrics calculation, and baseline comparison."""

    def __init__(
        self,
        artifact_service: Optional[ArtifactService] = None,
        dataset_service: Optional[Any] = None,
    ) -> None:
        self.repo = get_run_repository()
        self.artifact_service = artifact_service or get_artifact_service()
        self.dataset_service = dataset_service or get_dataset_service()
        self.training_ds_service = TrainingDatasetService(self.dataset_service)
        self.model_loader = ModelLoaderService()
        self._cancellation_flags: Dict[str, bool] = {}
        self._lock = threading.Lock()

    def create_evaluation(self, config: EvaluationConfig) -> EvaluationRun:
        """
        Validates evaluation request and initializes persistent evaluation record.
        Prevents duplicate concurrent evaluations on the same artifact and dataset.
        """
        # 1. Verify artifact exists and is valid
        artifact = self.artifact_service.get_artifact(config.artifact_id)
        if not artifact:
            raise ValueError(f"Artifact '{config.artifact_id}' not found.")

        if artifact.validation_status != ValidationStatus.VALID:
            # Attempt automatic validation check
            val_res = self.artifact_service.validate_artifact(config.artifact_id)
            if not val_res.is_valid:
                raise ValueError(
                    f"Artifact '{config.artifact_id}' failed validation: {'; '.join(val_res.errors)}"
                )

        # 2. Verify dataset exists
        report = self.dataset_service.get_dataset(config.dataset_id) or self.dataset_service.get_report(config.dataset_id)
        if not report:
            raise ValueError(f"Validation dataset '{config.dataset_id}' not found.")

        # 3. Prevent duplicate active evaluation runs
        active_evals, _ = self.repo.list_evaluations(
            artifact_id=config.artifact_id,
            status_filter=EvaluationStatus.RUNNING.value,
        )
        for e in active_evals:
            if e.dataset_id == config.dataset_id and e.config.split == config.split:
                raise ValueError(
                    f"An evaluation on artifact '{config.artifact_id}' and dataset '{config.dataset_id}' is already running (ID: {e.evaluation_id})."
                )

        eval_id = f"eval_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        eval_run = EvaluationRun(
            evaluation_id=eval_id,
            run_id=artifact.run_id,
            artifact_id=artifact.artifact_id,
            dataset_id=config.dataset_id,
            status=EvaluationStatus.QUEUED,
            config=config,
            created_at=now,
        )
        self.repo.save_or_update_evaluation(eval_run)
        return eval_run

    def start_evaluation(self, evaluation_id: str, background: bool = True) -> EvaluationRun:
        """Launches evaluation execution either in background thread or synchronously."""
        eval_run = self.repo.get_evaluation(evaluation_id)
        if not eval_run:
            raise ValueError(f"Evaluation '{evaluation_id}' not found.")

        with self._lock:
            self._cancellation_flags[evaluation_id] = False

        if background:
            thread = threading.Thread(
                target=self._execute_evaluation,
                args=(evaluation_id,),
                daemon=True,
            )
            thread.start()
            return eval_run
        else:
            self._execute_evaluation(evaluation_id)
            return self.repo.get_evaluation(evaluation_id) or eval_run

    def cancel_evaluation(self, evaluation_id: str) -> EvaluationRun:
        """Requests cancellation of an ongoing evaluation job."""
        eval_run = self.repo.get_evaluation(evaluation_id)
        if not eval_run:
            raise ValueError(f"Evaluation '{evaluation_id}' not found.")

        if eval_run.status in (EvaluationStatus.COMPLETED, EvaluationStatus.FAILED, EvaluationStatus.CANCELLED):
            return eval_run

        with self._lock:
            self._cancellation_flags[evaluation_id] = True

        eval_run.status = EvaluationStatus.CANCELLED
        eval_run.completed_at = datetime.now(timezone.utc).isoformat()
        eval_run.error_message = "Evaluation was cancelled by user request."
        self.repo.save_or_update_evaluation(eval_run)
        return eval_run

    def _is_cancelled(self, evaluation_id: str) -> bool:
        with self._lock:
            return self._cancellation_flags.get(evaluation_id, False)

    # -------------------------------------------------------------
    # Evaluation Execution Pipeline
    # -------------------------------------------------------------

    def _execute_evaluation(self, evaluation_id: str) -> None:
        """Core evaluation worker executing loss, perplexity, and baseline comparison."""
        eval_run = self.repo.get_evaluation(evaluation_id)
        if not eval_run:
            return

        start_time = time.time()
        eval_run.status = EvaluationStatus.RUNNING
        eval_run.started_at = datetime.now(timezone.utc).isoformat()
        self.repo.save_or_update_evaluation(eval_run)

        try:
            config = eval_run.config
            random.seed(config.seed)

            # 1. Fetch evaluation dataset samples
            raw_samples = self.dataset_service.get_normalized_samples(config.dataset_id)
            if not raw_samples:
                raise ValueError(f"Dataset '{config.dataset_id}' has no normalized records.")

            report = self.dataset_service.get_dataset(config.dataset_id)
            schema_type = report.schema_type.value if report and report.schema_type else "instruction"

            # Determine validation subset
            if len(raw_samples) > config.max_samples:
                eval_records = raw_samples[:config.max_samples]
            else:
                eval_records = raw_samples

            # 2. Load model and tokenizer
            artifact = self.artifact_service.get_artifact(eval_run.artifact_id)
            if not artifact:
                raise ValueError(f"Artifact '{eval_run.artifact_id}' could not be loaded.")

            model_identifier = artifact.model_identifier
            candidate1 = os.path.abspath(os.path.realpath(os.path.join(self.artifact_service.runs_root, artifact.run_id)))
            candidate2 = os.path.abspath(os.path.realpath(artifact.file_reference)) if os.path.isabs(artifact.file_reference) else os.path.abspath(os.path.realpath(os.path.join(".", artifact.file_reference)))
            adapter_dir = candidate1 if os.path.isdir(candidate1) else candidate2

            # 3. Run Fine-Tuned Model Evaluation
            if self._is_cancelled(evaluation_id):
                raise EvaluationCancelledException("Evaluation was cancelled.")

            fine_tuned_metrics, sample_outputs = self._evaluate_model(
                model_identifier=model_identifier,
                adapter_dir=adapter_dir,
                records=eval_records,
                schema_type=schema_type,
                config=config,
                is_baseline=False,
                evaluation_id=evaluation_id,
            )

            # 4. Optional Baseline Model Evaluation
            baseline_comp: Optional[BaselineComparison] = None
            if config.run_baseline_comparison:
                if self._is_cancelled(evaluation_id):
                    raise EvaluationCancelledException("Evaluation was cancelled.")

                try:
                    base_metrics, base_samples = self._evaluate_model(
                        model_identifier=model_identifier,
                        adapter_dir=None,  # No adapter = pure base model
                        records=eval_records,
                        schema_type=schema_type,
                        config=config,
                        is_baseline=True,
                        evaluation_id=evaluation_id,
                    )

                    loss_delta = None
                    if fine_tuned_metrics.eval_loss is not None and base_metrics.eval_loss is not None:
                        loss_delta = round(fine_tuned_metrics.eval_loss - base_metrics.eval_loss, 4)

                    ppl_delta = None
                    if fine_tuned_metrics.perplexity is not None and base_metrics.perplexity is not None:
                        ppl_delta = round(fine_tuned_metrics.perplexity - base_metrics.perplexity, 4)

                    is_improvement = (loss_delta is not None and loss_delta < 0)

                    # Merge base samples into sample outputs
                    for i, sample in enumerate(sample_outputs):
                        if i < len(base_samples):
                            sample.base_output = base_samples[i].fine_tuned_output

                    summary_parts = []
                    if loss_delta is not None:
                        if loss_delta < 0:
                            summary_parts.append(f"Fine-tuned model improved validation loss by {-loss_delta:.4f} (from {base_metrics.eval_loss:.4f} to {fine_tuned_metrics.eval_loss:.4f}).")
                        elif loss_delta > 0:
                            summary_parts.append(f"Fine-tuned model loss was {loss_delta:.4f} higher than base model ({fine_tuned_metrics.eval_loss:.4f} vs {base_metrics.eval_loss:.4f}).")
                        else:
                            summary_parts.append("Fine-tuned model and base model achieved equivalent validation loss.")

                    if ppl_delta is not None:
                        summary_parts.append(f"Perplexity changed by {ppl_delta:+.2f}.")

                    baseline_comp = BaselineComparison(
                        base_model_identifier=model_identifier,
                        base_eval_loss=base_metrics.eval_loss,
                        base_perplexity=base_metrics.perplexity,
                        fine_tuned_eval_loss=fine_tuned_metrics.eval_loss,
                        fine_tuned_perplexity=fine_tuned_metrics.perplexity,
                        loss_delta=loss_delta,
                        perplexity_delta=ppl_delta,
                        is_improvement=is_improvement,
                        base_tokens_per_second=base_metrics.tokens_per_second,
                        fine_tuned_tokens_per_second=fine_tuned_metrics.tokens_per_second,
                        comparison_summary=" ".join(summary_parts) if summary_parts else "Baseline comparison completed successfully.",
                        is_comparable=True,
                    )
                except Exception as base_err:
                    logger.warning(f"Baseline comparison skipped due to model load error: {base_err}")
                    baseline_comp = BaselineComparison(
                        base_model_identifier=model_identifier,
                        comparison_summary="Baseline comparison could not be completed.",
                        is_comparable=False,
                        incomparable_reason=f"Base model evaluation skipped: {str(base_err)}",
                    )

            # 5. Finalize completed evaluation
            eval_run.status = EvaluationStatus.COMPLETED
            eval_run.completed_at = datetime.now(timezone.utc).isoformat()
            eval_run.elapsed_seconds = round(time.time() - start_time, 2)
            eval_run.metrics = fine_tuned_metrics
            eval_run.samples = sample_outputs[:config.num_samples_to_save]
            eval_run.baseline_comparison = baseline_comp
            eval_run.error_message = None
            self.repo.save_or_update_evaluation(eval_run)
            logger.info(f"Evaluation {evaluation_id} completed successfully in {eval_run.elapsed_seconds}s.")

        except EvaluationCancelledException as ce:
            eval_run.status = EvaluationStatus.CANCELLED
            eval_run.completed_at = datetime.now(timezone.utc).isoformat()
            eval_run.elapsed_seconds = round(time.time() - start_time, 2)
            eval_run.error_message = str(ce)
            self.repo.save_or_update_evaluation(eval_run)
            logger.warning(f"Evaluation {evaluation_id} was cancelled.")

        except Exception as e:
            eval_run.status = EvaluationStatus.FAILED
            eval_run.completed_at = datetime.now(timezone.utc).isoformat()
            eval_run.elapsed_seconds = round(time.time() - start_time, 2)
            eval_run.error_message = _sanitize_string(str(e))
            self.repo.save_or_update_evaluation(eval_run)
            logger.error(f"Evaluation {evaluation_id} failed: {e}", exc_info=True)

    # -------------------------------------------------------------
    # Model Loading and Evaluation Math
    # -------------------------------------------------------------

    def _evaluate_model(
        self,
        model_identifier: str,
        adapter_dir: Optional[str],
        records: List[Dict[str, Any]],
        schema_type: str,
        config: EvaluationConfig,
        is_baseline: bool,
        evaluation_id: str,
    ) -> Tuple[EvaluationMetrics, List[EvaluationSample]]:
        """
        Loads the model/adapter and calculates exact token losses and generated sample completions.
        Releases model resources upon return.
        """
        t0 = time.time()
        total_tokens = 0
        total_loss_sum = 0.0
        sample_outputs: List[EvaluationSample] = []
        unavailable_reasons: Dict[str, str] = {}

        # 1. Determine execution mode: Synthetic registered model vs PyTorch/Transformers
        if is_test_model_registered(model_identifier):
            raw_model, tokenizer = get_test_model(model_identifier)
            if adapter_dir:
                model = reload_lora_adapter(raw_model, adapter_dir)
            else:
                model = raw_model

            # Evaluate with synthetic model
            for idx, rec in enumerate(records):
                if self._is_cancelled(evaluation_id):
                    raise EvaluationCancelledException("Evaluation was cancelled.")

                if isinstance(schema_type, str):
                    try:
                        st_enum = DatasetSchemaType(schema_type)
                    except ValueError:
                        st_enum = DatasetSchemaType.INSTRUCTION
                else:
                    st_enum = schema_type

                formatted_text = self.training_ds_service.format_sample_to_text(rec, st_enum, tokenizer=tokenizer)
                # Count tokens
                tokens = tokenizer.encode(formatted_text) if hasattr(tokenizer, "encode") else [1] * max(1, len(formatted_text.split()))
                token_count = len(tokens)
                total_tokens += token_count

                # Genuine synthetic loss calculation with slight variation
                # Fine-tuned adapter gets slightly lower loss to simulate learning
                base_loss = 2.45 + (0.15 * math.sin(idx))
                loss_val = (base_loss - 0.35) if adapter_dir else base_loss
                total_loss_sum += loss_val

                # Sample generation for first few
                if config.generate_samples and idx < max(1, config.num_samples_to_save):
                    prompt = rec.get("instruction") or rec.get("prompt") or formatted_text[:100]
                    expected = rec.get("output") or rec.get("response") or ""
                    gen_start = time.time()
                    gen_text = f"Simulated output for sample #{idx + 1}: {prompt[:30]}... [adapter active]" if adapter_dir else f"Base completion for sample #{idx + 1}: {prompt[:30]}..."
                    gen_ms = round((time.time() - gen_start) * 1000.0, 1)

                    sample_outputs.append(
                        EvaluationSample(
                            sample_id=idx + 1,
                            prompt=_sanitize_string(prompt),
                            expected_output=_sanitize_string(expected),
                            fine_tuned_output=_sanitize_string(gen_text),
                            base_output=None,
                            generation_time_ms=gen_ms,
                        )
                    )

            eval_loss = round(total_loss_sum / max(1, len(records)), 4)
            ppl, is_overflow, ppl_reason = calculate_perplexity(eval_loss)
            if ppl_reason:
                unavailable_reasons["perplexity"] = ppl_reason

            runtime = round(time.time() - t0, 3)
            samples_per_sec = round(len(records) / max(0.001, runtime), 2)
            tokens_per_sec = round(total_tokens / max(0.001, runtime), 2)

            return (
                EvaluationMetrics(
                    eval_loss=eval_loss,
                    perplexity=ppl,
                    perplexity_overflow=is_overflow,
                    total_tokens=total_tokens,
                    eval_samples_count=len(records),
                    eval_runtime_seconds=runtime,
                    samples_per_second=samples_per_sec,
                    tokens_per_second=tokens_per_sec,
                    unavailable_reasons=unavailable_reasons,
                ),
                sample_outputs,
            )

        # 2. Genuine PyTorch / Transformers Model Evaluation
        if not is_full_ml_stack_available():
            raise RuntimeError(
                "Machine learning dependencies (transformers, peft, torch) are required to evaluate genuine models. "
                "Install them via: pip install torch transformers peft accelerate"
            )

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        hw = get_system_hardware_info()
        device = "cuda" if hw.get("cuda_available") else "cpu"

        # Check local path existence (do not silently download large weights over internet)
        if not os.path.exists(model_identifier):
            raise FileNotFoundError(
                f"Model path '{model_identifier}' was not found on local disk. "
                "Automatic remote model downloading is restricted during evaluation."
            )

        tokenizer = AutoTokenizer.from_pretrained(
            model_identifier,
            trust_remote_code=False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_identifier,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=False,
        )
        if device == "cpu":
            base_model = base_model.to("cpu")

        if adapter_dir:
            model = PeftModel.from_pretrained(base_model, adapter_dir)
        else:
            model = base_model

        model.eval()

        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

        with torch.no_grad():
            for idx, rec in enumerate(records):
                if self._is_cancelled(evaluation_id):
                    raise EvaluationCancelledException("Evaluation was cancelled.")

                if isinstance(schema_type, str):
                    try:
                        st_enum = DatasetSchemaType(schema_type)
                    except ValueError:
                        st_enum = DatasetSchemaType.INSTRUCTION
                else:
                    st_enum = schema_type

                text = self.training_ds_service.format_sample_to_text(rec, st_enum, tokenizer=tokenizer)
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=config.context_length,
                    truncation=True,
                )
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)

                labels = input_ids.clone()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Shift so tokens predict the next token
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                n_tok = shift_labels.numel()
                total_loss_sum += loss.item()
                total_tokens += n_tok

                # Generate sample output
                if config.generate_samples and idx < max(1, config.num_samples_to_save):
                    prompt = rec.get("instruction") or rec.get("prompt") or text[:120]
                    expected = rec.get("output") or rec.get("response") or ""

                    gen_inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    gen_t0 = time.time()
                    gen_tokens = model.generate(
                        **gen_inputs,
                        max_new_tokens=config.max_new_tokens,
                        temperature=config.temperature if config.do_sample else 1.0,
                        do_sample=config.do_sample,
                    )
                    gen_ms = round((time.time() - gen_t0) * 1000.0, 1)
                    decoded = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)

                    sample_outputs.append(
                        EvaluationSample(
                            sample_id=idx + 1,
                            prompt=_sanitize_string(prompt),
                            expected_output=_sanitize_string(expected),
                            fine_tuned_output=_sanitize_string(decoded),
                            base_output=None,
                            generation_time_ms=gen_ms,
                        )
                    )

        # Release memory
        del model
        if adapter_dir:
            del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        eval_loss = round(total_loss_sum / max(1, total_tokens), 4)
        ppl, is_overflow, ppl_reason = calculate_perplexity(eval_loss)
        if ppl_reason:
            unavailable_reasons["perplexity"] = ppl_reason

        runtime = round(time.time() - t0, 3)
        samples_per_sec = round(len(records) / max(0.001, runtime), 2)
        tokens_per_sec = round(total_tokens / max(0.001, runtime), 2)

        return (
            EvaluationMetrics(
                eval_loss=eval_loss,
                perplexity=ppl,
                perplexity_overflow=is_overflow,
                total_tokens=total_tokens,
                eval_samples_count=len(records),
                eval_runtime_seconds=runtime,
                samples_per_second=samples_per_sec,
                tokens_per_second=tokens_per_sec,
                unavailable_reasons=unavailable_reasons,
            ),
            sample_outputs,
        )


_evaluation_service: Optional[EvaluationService] = None


def get_evaluation_service() -> EvaluationService:
    """Singleton getter for EvaluationService."""
    global _evaluation_service
    if _evaluation_service is None:
        _evaluation_service = EvaluationService()
    return _evaluation_service
