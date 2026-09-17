"""
Dataset Cleaner and Duplicate Detector.
"""

import hashlib
import re
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel, Field

from intelligence.dataset.schema import NormalizedSample, validate_raw_sample
from core.logger import get_logger

logger = get_logger("dataset_cleaner")


class CleaningOptions(BaseModel):
    """Configuration settings for dataset cleaning."""
    enable_cleaning: bool = Field(default=True, description="Enable cleaning steps")
    remove_invalid: bool = Field(default=True, description="Remove samples that fail schema validation")
    remove_duplicates: bool = Field(default=False, description="Remove duplicate content samples")
    normalize_whitespace: bool = Field(default=True, description="Normalize extra whitespace in text fields")


class CleaningStats(BaseModel):
    """Detailed statistics captured during dataset cleaning."""
    original_count: int = 0
    invalid_count: int = 0
    duplicate_count: int = 0
    clean_count: int = 0
    duplicate_indices: List[int] = Field(default_factory=list)
    invalid_indices: List[int] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class DatasetCleaner:
    """Performs deterministic validation, whitespace normalization, and duplicate detection."""

    def __init__(self, options: Optional[CleaningOptions] = None):
        self.options = options or CleaningOptions()

    @staticmethod
    def compute_sample_hash(sample: NormalizedSample) -> str:
        """Computes deterministic SHA-256 hash of normalized text content."""
        content = sample.get_text_content().lower()
        # Collapse whitespace for hash comparison
        collapsed = re.sub(r"\s+", " ", content).strip()
        return hashlib.sha256(collapsed.encode("utf-8")).hexdigest()

    def _normalize_sample_whitespace(self, sample: NormalizedSample) -> NormalizedSample:
        """Normalizes excessive internal whitespace in text fields."""
        if not self.options.normalize_whitespace:
            return sample

        def clean_str(val: Optional[str]) -> Optional[str]:
            if val is None:
                return None
            return re.sub(r"[ \t]+", " ", val).strip()

        if sample.instruction or sample.output or sample.input:
            return NormalizedSample(
                schema_type=sample.schema_type,
                instruction=clean_str(sample.instruction),
                input=clean_str(sample.input),
                output=clean_str(sample.output),
            )
        elif sample.messages:
            cleaned_msgs = [
                {"role": m["role"], "content": clean_str(m.get("content", "")) or ""}
                for m in sample.messages
            ]
            return NormalizedSample(
                schema_type=sample.schema_type,
                messages=cleaned_msgs,
            )
        return sample

    def process(
        self, raw_samples: List[Dict[str, Any]], normalized_samples: List[NormalizedSample]
    ) -> Tuple[List[NormalizedSample], CleaningStats]:
        """
        Cleans and deduplicates a normalized dataset.

        Args:
            raw_samples: List of raw input sample dictionaries.
            normalized_samples: Corresponding list of NormalizedSample objects.

        Returns:
            Tuple of (cleaned NormalizedSample list, CleaningStats object).
        """
        stats = CleaningStats(original_count=len(raw_samples))
        seen_hashes: Dict[str, int] = {}
        clean_samples: List[NormalizedSample] = []

        for idx, (raw, norm) in enumerate(zip(raw_samples, normalized_samples)):
            # 1. Validation check
            validation_errors = validate_raw_sample(raw)
            if validation_errors:
                stats.invalid_count += 1
                stats.invalid_indices.append(idx)
                if self.options.enable_cleaning and self.options.remove_invalid:
                    logger.debug(f"Filtering invalid sample at index {idx}: {validation_errors}")
                    continue

            # 2. Whitespace normalization
            cleaned_norm = self._normalize_sample_whitespace(norm) if self.options.enable_cleaning else norm

            # 3. Duplicate detection
            sample_hash = self.compute_sample_hash(cleaned_norm)
            if sample_hash in seen_hashes:
                stats.duplicate_count += 1
                stats.duplicate_indices.append(idx)
                if self.options.enable_cleaning and self.options.remove_duplicates:
                    logger.debug(f"Filtering duplicate sample at index {idx} (duplicate of {seen_hashes[sample_hash]})")
                    continue
            else:
                seen_hashes[sample_hash] = idx

            clean_samples.append(cleaned_norm)

        stats.clean_count = len(clean_samples)
        return clean_samples, stats
