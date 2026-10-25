"""
Dataset Upload, Schema Detection, Validation, Normalization, and Split Service.
Handles secure file storage, format parsing (JSON, JSONL, CSV), schema detection,
deterministic splitting without data leakage, and secret/path isolation.
"""

import os
import csv
import json
import uuid
import hashlib
import random
from io import StringIO
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

from backend.app.core.config import settings
from backend.app.schemas.datasets import (
    DatasetFormat,
    DatasetSchemaType,
    SampleValidationError,
    DatasetValidationReport,
    DatasetSummary,
    DatasetSplitResult,
)


class DatasetService:
    """
    Manages dataset file uploads, multi-format parsing, schema validation,
    normalization, and deterministic split generation.
    """

    ALLOWED_EXTENSIONS = {
        ".json": DatasetFormat.JSON,
        ".jsonl": DatasetFormat.JSONL,
        ".csv": DatasetFormat.CSV,
    }

    VALID_CONVERSATIONAL_ROLES = {"user", "assistant", "system", "human", "gpt", "bot"}

    def __init__(
        self,
        upload_dir: Optional[str] = None,
        max_upload_size_bytes: Optional[int] = None,
    ) -> None:
        self.upload_dir = upload_dir or settings.dataset_upload_dir
        self.max_upload_size_bytes = (
            max_upload_size_bytes
            if max_upload_size_bytes is not None
            else settings.max_upload_size_bytes
        )
        os.makedirs(self.upload_dir, exist_ok=True)

        # In-memory registry of validated dataset reports
        self._reports: Dict[str, DatasetValidationReport] = {}
        # Storage mapping of dataset_id -> stored filename (relative to upload_dir)
        self._file_map: Dict[str, str] = {}
        # In-memory store of parsed valid normalized samples
        self._normalized_cache: Dict[str, List[Dict[str, Any]]] = {}
        # Upload timestamps
        self._timestamps: Dict[str, str] = {}

    def sanitize_filename(self, filename: str) -> str:
        """Sanitizes user-provided filename to prevent path traversal."""
        cleaned = os.path.basename(filename)
        # Strip traversal tokens and special characters
        cleaned = cleaned.replace("..", "").replace("/", "").replace("\\", "").strip()
        return cleaned or "dataset_upload"

    def get_format_from_filename(self, filename: str) -> DatasetFormat:
        """Validates extension and returns DatasetFormat or raises ValueError."""
        lower_name = filename.lower()
        for ext, fmt in self.ALLOWED_EXTENSIONS.items():
            if lower_name.endswith(ext):
                return fmt
        _, file_ext = os.path.splitext(lower_name)
        ext_str = f" '{file_ext}'" if file_ext else ""
        raise ValueError(
            f"Unsupported file format{ext_str}. Supported formats: {', '.join(sorted(self.ALLOWED_EXTENSIONS.keys()))}"
        )


    def generate_dataset_id(self) -> str:
        """Generates a secure server-side dataset identifier."""
        return f"ds_{uuid.uuid4().hex[:12]}"

    def store_upload_bytes(
        self,
        raw_bytes: bytes,
        original_filename: str,
    ) -> Tuple[str, str, DatasetFormat]:
        """
        Validates size and extension, stores bytes securely, and returns
        (dataset_id, sanitized_filename, format).
        """
        if len(raw_bytes) > self.max_upload_size_bytes:
            raise ValueError(
                f"File size ({len(raw_bytes)} bytes) exceeds maximum allowed limit ({self.max_upload_size_bytes} bytes)."
            )

        safe_filename = self.sanitize_filename(original_filename)
        file_format = self.get_format_from_filename(safe_filename)
        dataset_id = self.generate_dataset_id()

        # Prevent collisions by prefixing with unique dataset_id
        stored_filename = f"{dataset_id}_{safe_filename}"
        stored_path = os.path.join(self.upload_dir, stored_filename)

        with open(stored_path, "wb") as f:
            f.write(raw_bytes)

        self._file_map[dataset_id] = stored_filename
        self._timestamps[dataset_id] = datetime.now(timezone.utc).isoformat()

        return dataset_id, safe_filename, file_format

    def parse_and_validate(
        self,
        dataset_id: str,
        original_filename: str,
        file_format: DatasetFormat,
        content: str,
        file_size_bytes: int,
    ) -> DatasetValidationReport:
        """Parses raw content into samples, detects schema, and validates data integrity."""
        # 1. Check for empty file
        if not content or not content.strip():
            report = DatasetValidationReport(
                dataset_id=dataset_id,
                original_filename=original_filename,
                file_format=file_format,
                file_size_bytes=file_size_bytes,
                detected_schema=None,
                total_samples=0,
                valid_sample_count=0,
                invalid_sample_count=0,
                empty_sample_count=1,
                duplicate_count=0,
                is_valid=False,
                warnings=[],
                errors=[
                    SampleValidationError(
                        sample_index=1,
                        message="Dataset file is completely empty or contains only whitespace.",
                    )
                ],
                preview_samples=[],
            )
            self._reports[dataset_id] = report
            return report

        # 2. Format-specific record parsing
        raw_records: List[Dict[str, Any]] = []
        parsing_errors: List[SampleValidationError] = []
        empty_count = 0

        if file_format == DatasetFormat.JSON:
            try:
                parsed_json = json.loads(content)
                if isinstance(parsed_json, list):
                    for idx, item in enumerate(parsed_json, start=1):
                        if isinstance(item, dict):
                            raw_records.append(item)
                        else:
                            parsing_errors.append(
                                SampleValidationError(
                                    sample_index=idx,
                                    message=f"Sample at index {idx} must be a JSON object, got {type(item).__name__}.",
                                )
                            )
                elif isinstance(parsed_json, dict):
                    # Check for standard wrapped object keys (e.g. {"data": [...]})
                    for key in ("data", "samples", "rows", "instances", "train"):
                        if key in parsed_json and isinstance(parsed_json[key], list):
                            for idx, item in enumerate(parsed_json[key], start=1):
                                if isinstance(item, dict):
                                    raw_records.append(item)
                                else:
                                    parsing_errors.append(
                                        SampleValidationError(
                                            sample_index=idx,
                                            message=f"Sample at index {idx} in '{key}' array must be a JSON object.",
                                        )
                                    )
                            break
                    else:
                        parsing_errors.append(
                            SampleValidationError(
                                sample_index=1,
                                message="JSON object must contain an array of sample objects under 'data', 'samples', or 'rows'.",
                            )
                        )
                else:
                    parsing_errors.append(
                        SampleValidationError(
                            sample_index=1,
                            message=f"JSON root must be an array or wrapped object, got {type(parsed_json).__name__}.",
                        )
                    )
            except json.JSONDecodeError as e:
                parsing_errors.append(
                    SampleValidationError(
                        sample_index=e.lineno or 1,
                        message=f"Malformed JSON on line {e.lineno}: {e.msg}",
                    )
                )

        elif file_format == DatasetFormat.JSONL:
            lines = content.splitlines()
            for line_idx, line in enumerate(lines, start=1):
                clean_line = line.strip()
                if not clean_line:
                    empty_count += 1
                    continue
                try:
                    item = json.loads(clean_line)
                    if isinstance(item, dict):
                        raw_records.append(item)
                    else:
                        parsing_errors.append(
                            SampleValidationError(
                                sample_index=line_idx,
                                message=f"Line {line_idx} must be a JSON object, got {type(item).__name__}.",
                            )
                        )
                except json.JSONDecodeError as e:
                    parsing_errors.append(
                        SampleValidationError(
                            sample_index=line_idx,
                            message=f"Malformed JSONL syntax on line {line_idx}: {e.msg}",
                            sample_preview=clean_line[:60],
                        )
                    )

        elif file_format == DatasetFormat.CSV:
            try:
                reader = csv.DictReader(StringIO(content))
                if not reader.fieldnames:
                    parsing_errors.append(
                        SampleValidationError(
                            sample_index=1,
                            message="CSV header is missing or empty.",
                        )
                    )
                else:
                    for row_idx, row in enumerate(reader, start=2):  # Row 1 is header
                        if not any(v.strip() for v in row.values() if v):
                            empty_count += 1
                            continue
                        raw_records.append(dict(row))
            except Exception as e:
                parsing_errors.append(
                    SampleValidationError(
                        sample_index=1,
                        message=f"Failed to parse CSV file: {str(e)}",
                    )
                )

        # 3. If parsing failed completely
        if parsing_errors and not raw_records:
            report = DatasetValidationReport(
                dataset_id=dataset_id,
                original_filename=original_filename,
                file_format=file_format,
                file_size_bytes=file_size_bytes,
                detected_schema=None,
                total_samples=len(parsing_errors),
                valid_sample_count=0,
                invalid_sample_count=len(parsing_errors),
                empty_sample_count=empty_count,
                duplicate_count=0,
                is_valid=False,
                warnings=[],
                errors=parsing_errors,
                preview_samples=[],
            )
            self._reports[dataset_id] = report
            return report

        # 4. Schema Detection across raw records
        detected_schema = self._detect_dataset_schema(raw_records)
        sample_errors: List[SampleValidationError] = list(parsing_errors)
        valid_normalized_samples: List[Dict[str, Any]] = []
        warnings: List[str] = []

        if detected_schema is None:
            sample_errors.append(
                SampleValidationError(
                    sample_index=1,
                    message=(
                        "Unable to identify a supported fine-tuning schema. "
                        "Supported schemas: Instruction (instruction, optional input, output), "
                        "Text (text/content), or Conversational (messages array with role and content)."
                    ),
                )
            )
        else:
            # Validate and normalize each sample
            seen_hashes = set()
            duplicate_count = 0

            for idx, raw_sample in enumerate(raw_records, start=1):
                errs = self._validate_sample_for_schema(raw_sample, detected_schema, idx)
                if errs:
                    sample_errors.extend(errs)
                else:
                    normalized = self._normalize_sample(raw_sample, detected_schema)
                    valid_normalized_samples.append(normalized)

                    # Content hashing for duplicate detection
                    sample_hash = self._hash_sample(normalized, detected_schema)
                    if sample_hash in seen_hashes:
                        duplicate_count += 1
                    else:
                        seen_hashes.add(sample_hash)

        valid_count = len(valid_normalized_samples)
        invalid_count = len(sample_errors)
        total_samples = valid_count + invalid_count

        # Small dataset warnings
        if valid_count > 0 and valid_count < 10:
            warnings.append(
                f"Dataset contains only {valid_count} valid samples. "
                "Training on small datasets may lead to overfitting; evaluation splits may be negligible."
            )

        if duplicate_count > 0:
            warnings.append(
                f"Identified {duplicate_count} duplicate samples based on content hash."
            )

        is_valid = (
            detected_schema is not None
            and valid_count > 0
            and len(parsing_errors) == 0
            and (invalid_count == 0 or (valid_count / total_samples >= 0.8))  # At least 80% valid to consider usable
        )

        preview = valid_normalized_samples[:3]
        report = DatasetValidationReport(
            dataset_id=dataset_id,
            original_filename=original_filename,
            file_format=file_format,
            file_size_bytes=file_size_bytes,
            detected_schema=detected_schema,
            total_samples=total_samples,
            valid_sample_count=valid_count,
            invalid_sample_count=invalid_count,
            empty_sample_count=empty_count,
            duplicate_count=duplicate_count if detected_schema else None,
            is_valid=is_valid,
            warnings=warnings,
            errors=sample_errors,
            preview_samples=preview,
        )

        self._reports[dataset_id] = report
        self._normalized_cache[dataset_id] = valid_normalized_samples
        return report

    def _detect_dataset_schema(
        self, samples: List[Dict[str, Any]]
    ) -> Optional[DatasetSchemaType]:
        """Examines samples to detect instruction, conversational, or text schema."""
        if not samples:
            return None

        # Inspect first non-empty samples
        for s in samples[:10]:
            if not isinstance(s, dict):
                continue

            # Conversational format: messages or conversations array
            if ("messages" in s and isinstance(s["messages"], list)) or (
                "conversations" in s and isinstance(s["conversations"], list)
            ):
                return DatasetSchemaType.CONVERSATIONAL

            # Instruction format keys
            instruction_keys = {"instruction", "prompt", "question", "user_prompt"}
            output_keys = {"output", "response", "target", "answer", "completion"}
            if any(k in s for k in instruction_keys) or any(k in s for k in output_keys):
                return DatasetSchemaType.INSTRUCTION

            # Text format
            if "text" in s or "content" in s:
                return DatasetSchemaType.TEXT

        return None

    def _validate_sample_for_schema(
        self,
        sample: Dict[str, Any],
        schema_type: DatasetSchemaType,
        index: int,
    ) -> List[SampleValidationError]:
        """Validates a single record against the expected schema format."""
        errors: List[SampleValidationError] = []

        if schema_type == DatasetSchemaType.INSTRUCTION:
            inst = (
                sample.get("instruction")
                or sample.get("prompt")
                or sample.get("question")
                or sample.get("user_prompt")
            )
            out = (
                sample.get("output")
                or sample.get("response")
                or sample.get("target")
                or sample.get("answer")
                or sample.get("completion")
            )

            if inst is None or not str(inst).strip():
                errors.append(
                    SampleValidationError(
                        sample_index=index,
                        message="Instruction schema requires a non-empty string for 'instruction' or 'prompt'.",
                    )
                )
            if out is None or not str(out).strip():
                errors.append(
                    SampleValidationError(
                        sample_index=index,
                        message="Instruction schema requires a non-empty string for 'output' or 'response'.",
                    )
                )

        elif schema_type == DatasetSchemaType.TEXT:
            text_val = sample.get("text") or sample.get("content")
            if text_val is None or not str(text_val).strip():
                errors.append(
                    SampleValidationError(
                        sample_index=index,
                        message="Text schema requires a non-empty string in the 'text' or 'content' field.",
                    )
                )

        elif schema_type == DatasetSchemaType.CONVERSATIONAL:
            messages = sample.get("messages") or sample.get("conversations")
            if not isinstance(messages, list) or len(messages) == 0:
                errors.append(
                    SampleValidationError(
                        sample_index=index,
                        message="Conversational schema requires a non-empty 'messages' or 'conversations' list.",
                    )
                )
            else:
                has_user = False
                has_assistant = False
                for m_idx, m in enumerate(messages):
                    if not isinstance(m, dict):
                        errors.append(
                            SampleValidationError(
                                sample_index=index,
                                message=f"Message {m_idx} must be a dictionary with role and content.",
                            )
                        )
                        continue
                    role = str(m.get("role") or m.get("from") or "").strip().lower()
                    content = str(m.get("content") or m.get("value") or "").strip()

                    if role not in self.VALID_CONVERSATIONAL_ROLES:
                        errors.append(
                            SampleValidationError(
                                sample_index=index,
                                message=f"Invalid conversational role '{role}' in message {m_idx}. Allowed: {sorted(self.VALID_CONVERSATIONAL_ROLES)}.",
                            )
                        )
                    if not content:
                        errors.append(
                            SampleValidationError(
                                sample_index=index,
                                message=f"Empty content for message {m_idx} with role '{role}'.",
                            )
                        )

                    if role in ("user", "human"):
                        has_user = True
                    if role in ("assistant", "gpt", "bot"):
                        has_assistant = True

                if not has_user:
                    errors.append(
                        SampleValidationError(
                            sample_index=index,
                            message="Conversational format must contain at least one user or human message.",
                        )
                    )
                if not has_assistant:
                    errors.append(
                        SampleValidationError(
                            sample_index=index,
                            message="Conversational format must contain at least one assistant or gpt response.",
                        )
                    )

        return errors

    def _normalize_sample(
        self,
        sample: Dict[str, Any],
        schema_type: DatasetSchemaType,
    ) -> Dict[str, Any]:
        """Converts raw sample to deterministic normalized structure."""
        if schema_type == DatasetSchemaType.INSTRUCTION:
            inst = (
                sample.get("instruction")
                or sample.get("prompt")
                or sample.get("question")
                or sample.get("user_prompt")
            )
            inp = sample.get("input") or sample.get("context")
            out = (
                sample.get("output")
                or sample.get("response")
                or sample.get("target")
                or sample.get("answer")
                or sample.get("completion")
            )
            return {
                "instruction": str(inst).strip(),
                "input": str(inp).strip() if inp and str(inp).strip() else None,
                "output": str(out).strip(),
            }

        elif schema_type == DatasetSchemaType.TEXT:
            text_val = sample.get("text") or sample.get("content")
            return {"text": str(text_val).strip()}

        elif schema_type == DatasetSchemaType.CONVERSATIONAL:
            messages = sample.get("messages") or sample.get("conversations") or []
            normalized_messages = []
            for m in messages:
                if isinstance(m, dict):
                    role = str(m.get("role") or m.get("from") or "user").strip().lower()
                    if role in ("human", "user"):
                        role = "user"
                    elif role in ("gpt", "assistant", "bot"):
                        role = "assistant"
                    elif role == "system":
                        role = "system"

                    content = str(m.get("content") or m.get("value") or "").strip()
                    normalized_messages.append({"role": role, "content": content})
            return {"messages": normalized_messages}

        return sample

    def _hash_sample(
        self,
        normalized: Dict[str, Any],
        schema_type: DatasetSchemaType,
    ) -> str:
        """Computes deterministic SHA-256 hash for duplicate detection."""
        if schema_type == DatasetSchemaType.INSTRUCTION:
            text = f"{normalized.get('instruction', '')}\n{normalized.get('input', '')}\n{normalized.get('output', '')}"
        elif schema_type == DatasetSchemaType.TEXT:
            text = normalized.get("text", "")
        elif schema_type == DatasetSchemaType.CONVERSATIONAL:
            text = "\n".join(
                f"{m.get('role', '')}:{m.get('content', '')}"
                for m in normalized.get("messages", [])
            )
        else:
            text = json.dumps(normalized, sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_report(self, dataset_id: str) -> Optional[DatasetValidationReport]:
        """Retrieves validation report by dataset ID."""
        return self._reports.get(dataset_id)

    def list_datasets(self) -> List[DatasetSummary]:
        """Returns summaries of all processed datasets."""
        summaries = []
        for d_id, r in self._reports.items():
            summaries.append(
                DatasetSummary(
                    id=d_id,
                    filename=r.original_filename,
                    format=r.file_format,
                    size_bytes=r.file_size_bytes,
                    uploaded_at=self._timestamps.get(d_id, ""),
                    total_samples=r.total_samples,
                    valid_sample_count=r.valid_sample_count,
                    schema_type=r.detected_schema,
                    is_valid=r.is_valid,
                )
            )
        return summaries

    def compute_splits(
        self,
        dataset_id: str,
        train_ratio: float = 0.9,
        eval_ratio: float = 0.1,
        seed: int = 42,
        shuffle: bool = True,
    ) -> DatasetSplitResult:
        """
        Computes deterministic, non-overlapping train and evaluation splits.
        Guarantees zero sample leakage between train and evaluation sets.
        """
        report = self.get_report(dataset_id)
        if not report:
            raise KeyError(f"Dataset '{dataset_id}' not found.")

        valid_count = report.valid_sample_count
        if valid_count == 0:
            raise ValueError(
                f"Cannot generate splits: dataset '{dataset_id}' has 0 valid samples."
            )

        if train_ratio < 0.0 or train_ratio > 1.0 or eval_ratio < 0.0 or eval_ratio > 1.0:
            raise ValueError("Split ratios must be between 0.0 and 1.0.")

        if (train_ratio + eval_ratio) > 1.0001:
            raise ValueError(
                f"Total split ratio ({train_ratio + eval_ratio:.2f}) exceeds 1.0."
            )

        indices = list(range(valid_count))
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(indices)

        eval_count = int(round(valid_count * eval_ratio))
        # Ensure eval doesn't consume all if train_ratio > 0
        if train_ratio > 0 and eval_count >= valid_count:
            eval_count = valid_count - 1

        train_count = valid_count - eval_count if train_ratio + eval_ratio >= 0.999 else int(round(valid_count * train_ratio))

        eval_indices = sorted(indices[:eval_count])
        train_indices = sorted(indices[eval_count : eval_count + train_count])

        # Verification of no overlap (guaranteed disjoint subsets)
        overlap = set(train_indices).intersection(set(eval_indices))
        has_overlap = len(overlap) > 0

        warnings: List[str] = []
        if eval_count == 0 and eval_ratio > 0:
            warnings.append(
                f"Evaluation split has 0 samples because total valid samples ({valid_count}) is too small."
            )
        elif eval_count < 5 and eval_count > 0:
            warnings.append(
                f"Evaluation split is very small ({eval_count} samples). Statistical significance will be limited."
            )

        return DatasetSplitResult(
            dataset_id=dataset_id,
            total_samples=valid_count,
            train_count=len(train_indices),
            eval_count=len(eval_indices),
            train_indices=train_indices,
            eval_indices=eval_indices,
            has_overlap=has_overlap,
            warnings=warnings,
        )


# Singleton instance
_dataset_service_instance: Optional[DatasetService] = None


def get_dataset_service() -> DatasetService:
    """Returns singleton dataset service instance."""
    global _dataset_service_instance
    if _dataset_service_instance is None:
        _dataset_service_instance = DatasetService()
    return _dataset_service_instance
