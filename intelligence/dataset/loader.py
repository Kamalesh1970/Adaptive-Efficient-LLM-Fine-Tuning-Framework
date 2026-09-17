"""
Dataset Loader Abstraction supporting JSON, JSONL, CSV, and Hugging Face Dataset objects.
"""

import json
import csv
from pathlib import Path
from typing import Union, List, Dict, Any, Optional

from core.exceptions import DatasetError
from core.logger import get_logger

logger = get_logger("dataset_loader")


class DatasetLoader:
    """Abstraction for loading datasets from files or Hugging Face dataset objects."""

    SUPPORTED_FORMATS = {".json", ".jsonl", ".csv"}

    @classmethod
    def load(cls, source: Union[str, Path, Any]) -> List[Dict[str, Any]]:
        """
        Loads dataset records into a unified List[Dict[str, Any]].

        Args:
            source: Filepath string/Path or Hugging Face Dataset instance.

        Returns:
            List of raw data record dictionaries.

        Raises:
            DatasetError: If file not found, format unsupported, empty, or malformed.
        """
        # If input is a Hugging Face Dataset object (or list of dicts)
        if not isinstance(source, (str, Path)):
            return cls._load_from_hf_or_iterable(source)

        path = Path(source)
        if not path.exists():
            raise DatasetError(f"Dataset file not found at path: {source}")

        if not path.is_file():
            raise DatasetError(f"Dataset path is not a valid file: {source}")

        suffix = path.suffix.lower()
        if suffix not in cls.SUPPORTED_FORMATS:
            raise DatasetError(f"Unsupported dataset file format '{suffix}'. Supported formats: {cls.SUPPORTED_FORMATS}")

        # Check for 0-byte empty file
        if path.stat().st_size == 0:
            raise DatasetError(f"Dataset file is completely empty (0 bytes): {source}")

        if suffix == ".json":
            return cls._load_json(path)
        elif suffix == ".jsonl":
            return cls._load_jsonl(path)
        elif suffix == ".csv":
            return cls._load_csv(path)

        raise DatasetError(f"Unhandled file extension: {suffix}")

    @classmethod
    def _load_json(cls, path: Path) -> List[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise DatasetError(f"Malformed JSON file at {path}: {e}") from e

        if isinstance(data, dict):
            # Single object wrapped in list
            return [data]
        elif isinstance(data, list):
            if len(data) == 0:
                raise DatasetError(f"JSON dataset file contains empty list: {path}")
            return data
        else:
            raise DatasetError(f"Expected list or dict in JSON file {path}, got {type(data).__name__}")

    @classmethod
    def _load_jsonl(cls, path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        record = json.loads(line_str)
                        if isinstance(record, dict):
                            records.append(record)
                        else:
                            raise DatasetError(f"JSONL record at line {line_idx} must be a dict")
                    except json.JSONDecodeError as je:
                        raise DatasetError(f"Malformed JSONL syntax at line {line_idx} in {path}: {je}") from je
        except DatasetError:
            raise
        except Exception as e:
            raise DatasetError(f"Failed to read JSONL file at {path}: {e}") from e

        if not records:
            raise DatasetError(f"JSONL file contains no valid JSON objects: {path}")

        return records

    @classmethod
    def _load_csv(cls, path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise DatasetError(f"CSV file has no header row: {path}")
                for row in reader:
                    records.append(dict(row))
        except DatasetError:
            raise
        except Exception as e:
            raise DatasetError(f"Malformed CSV file at {path}: {e}") from e

        if not records:
            raise DatasetError(f"CSV file contains no data rows: {path}")

        return records

    @classmethod
    def _load_from_hf_or_iterable(cls, obj: Any) -> List[Dict[str, Any]]:
        """Handles Hugging Face Dataset or iterable collection of dicts."""
        if hasattr(obj, "to_list") and callable(obj.to_list):
            data = obj.to_list()
        elif isinstance(obj, list):
            data = obj
        elif hasattr(obj, "__iter__"):
            data = list(obj)
        else:
            raise DatasetError(f"Unrecognized dataset object type: {type(obj).__name__}")

        if not data:
            raise DatasetError("Iterable dataset object is empty")

        return [dict(item) if isinstance(item, dict) else {"content": str(item)} for item in data]
