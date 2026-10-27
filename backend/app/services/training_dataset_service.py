"""
Dataset Preparation Service for LoRA Fine-Tuning (Phase 4).
Connects validated normalized datasets from Phase 2 to model tokenizers and training batches.
Supports instruction, conversational, and raw text schemas with sequence truncation and split verification.
"""

import copy
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from backend.app.schemas.training import TrainingConfigSchema
from backend.app.schemas.datasets import DatasetSchemaType
from backend.app.services.dataset_service import DatasetService, get_dataset_service

logger = logging.getLogger(__name__)


@dataclass
class PreparedTrainingDataset:
    """Encapsulates tokenized training and evaluation datasets."""
    train_samples: List[Dict[str, Any]]
    eval_samples: Optional[List[Dict[str, Any]]]
    train_count: int
    eval_count: int
    schema_type: str
    max_seq_length: int
    tokenizer_name_or_path: str


class PyTorchTokenizedDataset:
    """Lightweight PyTorch-compatible dataset for tokenized causal LM samples."""

    def __init__(self, samples: List[Dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        try:
            import torch
            return {
                "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(item["labels"], dtype=torch.long),
            }
        except ImportError:
            return item


class TrainingDatasetService:
    """Prepares and tokenizes validated datasets for causal language model training."""

    def __init__(self, dataset_service: Optional[DatasetService] = None) -> None:
        self.dataset_service = dataset_service or get_dataset_service()

    def format_sample_to_text(
        self,
        sample: Dict[str, Any],
        schema_type: DatasetSchemaType,
        tokenizer: Any,
    ) -> str:
        """
        Converts a normalized sample into a single text sequence for causal LM training.
        """
        if schema_type == DatasetSchemaType.INSTRUCTION:
            instruction = str(sample.get("instruction") or "").strip()
            input_text = str(sample.get("input") or "").strip()
            output = str(sample.get("output") or "").strip()

            if not instruction and not output:
                raise ValueError("Malformed instruction sample: missing both instruction and output.")

            if input_text:
                return (
                    f"Below is an instruction that describes a task, paired with an input that provides further context. "
                    f"Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Input:\n{input_text}\n\n"
                    f"### Response:\n{output}"
                )
            else:
                return (
                    f"Below is an instruction that describes a task. "
                    f"Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Response:\n{output}"
                )

        elif schema_type == DatasetSchemaType.CONVERSATIONAL:
            messages = sample.get("messages", [])
            if not messages or not isinstance(messages, list):
                raise ValueError("Malformed conversational sample: missing messages list.")

            # Attempt tokenizer's apply_chat_template if available
            if hasattr(tokenizer, "apply_chat_template") and callable(tokenizer.apply_chat_template):
                try:
                    formatted = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    if formatted:
                        return formatted
                except Exception as e:
                    logger.debug(f"Tokenizer chat_template unavailable or failed: {e}. Using fallback.")

            # Documented Fallback: Standard ChatML template
            formatted_parts = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                formatted_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            return "\n".join(formatted_parts)

        elif schema_type == DatasetSchemaType.TEXT:
            text = sample.get("text", "")
            if not text or not str(text).strip():
                raise ValueError("Malformed text sample: text field is empty.")
            return str(text)

        else:
            raise ValueError(f"Unsupported dataset schema type: {schema_type}")

    def tokenize_text(
        self,
        text: str,
        tokenizer: Any,
        max_seq_length: int,
    ) -> Dict[str, List[int]]:
        """
        Tokenizes text with sequence truncation and builds input_ids, attention_mask, labels.
        """
        if not text or not text.strip():
            raise ValueError("Cannot tokenize empty text string.")

        # Call tokenizer
        if hasattr(tokenizer, "__call__"):
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_seq_length,
                padding=False,
                return_tensors=None,
            )
            input_ids = encoded.get("input_ids", [])
            attention_mask = encoded.get("attention_mask", [1] * len(input_ids))
        else:
            raise TypeError("Provided tokenizer is not callable.")

        if not input_ids:
            raise ValueError("Tokenization resulted in empty token sequence.")

        # For causal language modeling, labels are identical to input_ids
        labels = list(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def prepare_dataset(
        self,
        dataset_id: str,
        tokenizer: Any,
        config: TrainingConfigSchema,
    ) -> PreparedTrainingDataset:
        """
        Loads, validates, splits, and tokenizes a dataset according to the training configuration.
        """
        report = self.dataset_service.get_report(dataset_id)
        if not report:
            raise ValueError(f"Dataset '{dataset_id}' not found. Please upload and validate first.")

        if not report.is_valid:
            raise ValueError(f"Dataset '{dataset_id}' failed validation: {report.errors}")

        if report.valid_sample_count == 0:
            raise ValueError(f"Dataset '{dataset_id}' has 0 valid samples.")

        normalized_samples = self.dataset_service.get_normalized_samples(dataset_id)
        if not normalized_samples:
            raise ValueError(f"Normalized samples for dataset '{dataset_id}' could not be retrieved.")

        schema_type = report.detected_schema
        if not schema_type:
            raise ValueError(f"Dataset '{dataset_id}' does not have a detected schema.")

        max_seq_length = config.context_length
        if max_seq_length <= 0:
            raise ValueError(f"Invalid max sequence length (context_length): {max_seq_length}")

        # Compute deterministic splits
        train_ratio = getattr(config, "train_split_ratio", 0.9)
        if train_ratio is None:
            train_ratio = 0.9
        eval_ratio = getattr(config, "eval_split_ratio", 0.1)
        if eval_ratio is None:
            eval_ratio = 0.1 if getattr(config, "eval_split", "none") != "none" else 0.0

        split_result = self.dataset_service.compute_splits(
            dataset_id=dataset_id,
            train_ratio=train_ratio,
            eval_ratio=eval_ratio,
            seed=config.seed or 42,
            shuffle=True,
        )

        if split_result.has_overlap:
            raise RuntimeError("Data leakage detected: train and eval splits share sample indices.")

        raw_train_samples = [normalized_samples[i] for i in split_result.train_indices]
        raw_eval_samples = [normalized_samples[i] for i in split_result.eval_indices]

        # Tokenize training samples
        tokenized_train: List[Dict[str, Any]] = []
        for sample in raw_train_samples:
            try:
                formatted_text = self.format_sample_to_text(sample, schema_type, tokenizer)
                tokens = self.tokenize_text(formatted_text, tokenizer, max_seq_length)
                tokenized_train.append(tokens)
            except Exception as e:
                logger.warning(f"Skipping malformed train sample: {e}")

        if not tokenized_train:
            raise ValueError(
                f"Tokenized training dataset is empty after processing {len(raw_train_samples)} samples."
            )

        # Tokenize evaluation samples if configured
        tokenized_eval: Optional[List[Dict[str, Any]]] = None
        eval_strategy = getattr(config, "evaluation_strategy", "no")
        use_eval = eval_strategy not in ("no", None) and len(raw_eval_samples) > 0

        if use_eval or len(raw_eval_samples) > 0:
            eval_list: List[Dict[str, Any]] = []
            for sample in raw_eval_samples:
                try:
                    formatted_text = self.format_sample_to_text(sample, schema_type, tokenizer)
                    tokens = self.tokenize_text(formatted_text, tokenizer, max_seq_length)
                    eval_list.append(tokens)
                except Exception as e:
                    logger.warning(f"Skipping malformed eval sample: {e}")
            if eval_list:
                tokenized_eval = eval_list

        tokenizer_name = getattr(tokenizer, "name_or_path", "custom_tokenizer")

        return PreparedTrainingDataset(
            train_samples=tokenized_train,
            eval_samples=tokenized_eval,
            train_count=len(tokenized_train),
            eval_count=len(tokenized_eval) if tokenized_eval else 0,
            schema_type=schema_type.value if hasattr(schema_type, "value") else str(schema_type),
            max_seq_length=max_seq_length,
            tokenizer_name_or_path=tokenizer_name,
        )


_training_dataset_service: Optional[TrainingDatasetService] = None


def get_training_dataset_service() -> TrainingDatasetService:
    """Returns singleton instance of TrainingDatasetService."""
    global _training_dataset_service
    if _training_dataset_service is None:
        _training_dataset_service = TrainingDatasetService()
    return _training_dataset_service
