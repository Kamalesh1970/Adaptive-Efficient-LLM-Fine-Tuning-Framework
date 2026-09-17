"""
High-Level Dataset & Tokenization Pipeline Orchestrator.
"""

from typing import Union, List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field

from intelligence.dataset.loader import DatasetLoader
from intelligence.dataset.schema import NormalizedSample, validate_raw_sample
from intelligence.dataset.normalizer import DatasetNormalizer
from intelligence.dataset.cleaner import DatasetCleaner, CleaningOptions, CleaningStats
from intelligence.dataset.tokenizer import TokenizerWrapper, MockTokenizer, TokenizedSample
from intelligence.dataset.analyzer import SequenceAnalyzer
from intelligence.dataset.profile import DatasetProfile
from core.logger import get_logger

logger = get_logger("dataset_pipeline")


class PipelineResult(BaseModel):
    """Container holding outputs of the dataset processing pipeline."""
    normalized_dataset: List[NormalizedSample] = Field(description="Normalized sample list")
    tokenized_dataset: List[TokenizedSample] = Field(description="Tokenized sample list")
    profile: DatasetProfile = Field(description="Structured dataset profile summary")
    cleaning_stats: CleaningStats = Field(description="Detailed cleaning statistics")
    validation_errors: List[str] = Field(default_factory=list, description="Validation issues encountered")


class DatasetPipeline:
    """
    High-level orchestrator executing the full dataset processing flow:
    Raw Data -> Load -> Schema Detect/Validate -> Normalize -> Clean -> Tokenize -> Analyze -> Profile
    """

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        max_sequence_length: int = 512,
        cleaning_options: Optional[CleaningOptions] = None,
        length_bins: Optional[List[int]] = None,
    ):
        raw_tokenizer = tokenizer if tokenizer is not None else MockTokenizer(max_length=max_sequence_length)
        if isinstance(raw_tokenizer, TokenizerWrapper):
            self.tokenizer_wrapper = raw_tokenizer
        else:
            self.tokenizer_wrapper = TokenizerWrapper(raw_tokenizer, max_length=max_sequence_length)

        self.max_sequence_length = max_sequence_length
        self.cleaner = DatasetCleaner(cleaning_options or CleaningOptions())
        self.length_bins = length_bins or [128, 256, 512, 1024]

    def process(self, source: Union[str, Path, List[Dict[str, Any]], Any]) -> PipelineResult:
        """
        Executes end-to-end dataset loading, normalization, cleaning, tokenization, and profiling.

        Args:
            source: Filepath, raw dict list, or Hugging Face dataset object.

        Returns:
            PipelineResult instance containing processed outputs and profile.
        """
        # 1. Load raw data
        if isinstance(source, list) and all(isinstance(i, dict) for i in source):
            raw_samples = source
        else:
            raw_samples = DatasetLoader.load(source)

        logger.info(f"Loaded {len(raw_samples)} raw dataset samples.")

        # 2. Collect validation issues across dataset
        validation_errors: List[str] = []
        for idx, sample in enumerate(raw_samples):
            errs = validate_raw_sample(sample)
            for e in errs:
                validation_errors.append(f"Sample [{idx}]: {e}")

        # 3. Normalize raw samples
        normalized_samples = DatasetNormalizer.normalize_dataset(raw_samples)

        # 4. Clean & Deduplicate
        clean_normalized, cleaning_stats = self.cleaner.process(raw_samples, normalized_samples)
        logger.info(f"Cleaned dataset: {cleaning_stats.clean_count} valid samples remaining.")

        # 5. Tokenize
        tokenized_samples = self.tokenizer_wrapper.encode_dataset(clean_normalized)

        # 6. Analyze sequence lengths and padding
        length_stats = SequenceAnalyzer.analyze_lengths(tokenized_samples, custom_bins=self.length_bins)
        padding_stats = SequenceAnalyzer.analyze_padding(tokenized_samples, sequence_capacity=self.max_sequence_length)

        # 7. Construct DatasetProfile
        detected_schema = clean_normalized[0].schema_type.value if clean_normalized else "unknown"

        profile = DatasetProfile(
            num_samples=length_stats.total_samples,
            total_tokens=length_stats.total_tokens,
            min_length=length_stats.min_length,
            max_length=length_stats.max_length,
            mean_length=length_stats.mean_length,
            median_length=length_stats.median_length,
            std_length=length_stats.std_length,
            duplicate_count=cleaning_stats.duplicate_count,
            invalid_count=cleaning_stats.invalid_count,
            estimated_padding_ratio=padding_stats.estimated_padding_ratio,
            schema_type=detected_schema,
            length_distribution=length_stats.length_distribution,
            percentiles=length_stats.percentiles,
        )

        return PipelineResult(
            normalized_dataset=clean_normalized,
            tokenized_dataset=tokenized_samples,
            profile=profile,
            cleaning_stats=cleaning_stats,
            validation_errors=validation_errors,
        )
