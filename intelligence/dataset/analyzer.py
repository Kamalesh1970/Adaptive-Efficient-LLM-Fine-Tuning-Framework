"""
Sequence Length and Padding Efficiency Analyzer.
"""

import math
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from intelligence.dataset.tokenizer import TokenizedSample
from core.exceptions import DatasetError
from core.logger import get_logger

logger = get_logger("sequence_analyzer")


class SequenceLengthStats(BaseModel):
    """Statistical summary of token sequence lengths."""
    total_samples: int = 0
    total_tokens: int = 0
    min_length: int = 0
    max_length: int = 0
    mean_length: float = 0.0
    median_length: float = 0.0
    std_length: float = 0.0
    percentiles: Dict[str, float] = Field(default_factory=dict)
    length_distribution: Dict[str, int] = Field(default_factory=dict)


class PaddingAnalysis(BaseModel):
    """Estimate of potential padding token waste under uniform sequence padding."""
    sequence_capacity: int = Field(description="Max sequence length capacity assumed for batching")
    total_allocated_tokens: int = Field(description="Total token slots reserved (num_samples * sequence_capacity)")
    actual_tokens: int = Field(description="Actual non-padding content tokens")
    estimated_padding_tokens: int = Field(description="Estimated padding tokens inserted")
    estimated_padding_ratio: float = Field(description="Ratio of padding tokens to total allocated tokens")
    note: str = Field(
        default="Estimated padding token waste assuming static uniform padding without sequence packing.",
        description="Disclaimer note"
    )


class SequenceAnalyzer:
    """Computes distribution statistics and padding efficiency estimates."""

    DEFAULT_BINS = [128, 256, 512, 1024]

    @classmethod
    def analyze_lengths(
        cls,
        tokenized_samples: List[TokenizedSample],
        custom_bins: Optional[List[int]] = None,
    ) -> SequenceLengthStats:
        """
        Calculates length distribution statistics for tokenized samples.

        Args:
            tokenized_samples: List of TokenizedSample objects.
            custom_bins: Optional list of upper bin boundary thresholds.

        Returns:
            SequenceLengthStats instance.
        """
        if not tokenized_samples:
            return SequenceLengthStats()

        lengths = [s.token_count for s in tokenized_samples]
        lengths.sort()

        n = len(lengths)
        total_tokens = sum(lengths)
        min_len = lengths[0]
        max_len = lengths[-1]
        mean_len = float(total_tokens) / n

        # Median
        if n % 2 == 1:
            median_len = float(lengths[n // 2])
        else:
            median_len = float(lengths[n // 2 - 1] + lengths[n // 2]) / 2.0

        # Standard Deviation
        variance = sum((x - mean_len) ** 2 for x in lengths) / n
        std_len = math.sqrt(variance)

        # Percentiles
        percentile_keys = [50, 75, 90, 95, 99]
        percentiles_dict: Dict[str, float] = {}
        for p in percentile_keys:
            idx = int(round((p / 100.0) * (n - 1)))
            idx = max(0, min(n - 1, idx))
            percentiles_dict[f"p{p}"] = float(lengths[idx])

        # Length Distribution Bins
        bin_edges = sorted(custom_bins or cls.DEFAULT_BINS)
        distribution: Dict[str, int] = {}

        prev_edge = 0
        for edge in bin_edges:
            bin_label = f"{prev_edge + 1}-{edge}" if prev_edge > 0 else f"0-{edge}"
            count = sum(1 for x in lengths if prev_edge < x <= edge)
            distribution[bin_label] = count
            prev_edge = edge

        # Remaining > max bin edge
        distribution[f">{prev_edge}"] = sum(1 for x in lengths if x > prev_edge)

        return SequenceLengthStats(
            total_samples=n,
            total_tokens=total_tokens,
            min_length=min_len,
            max_length=max_len,
            mean_length=round(mean_len, 2),
            median_length=round(median_len, 2),
            std_length=round(std_len, 2),
            percentiles=percentiles_dict,
            length_distribution=distribution,
        )

    @classmethod
    def analyze_padding(
        cls, tokenized_samples: List[TokenizedSample], sequence_capacity: int = 512
    ) -> PaddingAnalysis:
        """
        Estimates potential padding token waste for a given sequence length capacity.

        Args:
            tokenized_samples: List of TokenizedSample objects.
            sequence_capacity: Target batch sequence capacity length.

        Returns:
            PaddingAnalysis instance.
        """
        if not tokenized_samples:
            return PaddingAnalysis(
                sequence_capacity=sequence_capacity,
                total_allocated_tokens=0,
                actual_tokens=0,
                estimated_padding_tokens=0,
                estimated_padding_ratio=0.0,
            )

        num_samples = len(tokenized_samples)
        total_allocated = num_samples * sequence_capacity

        actual_tokens = sum(min(s.token_count, sequence_capacity) for s in tokenized_samples)
        padding_tokens = max(0, total_allocated - actual_tokens)
        padding_ratio = float(padding_tokens) / total_allocated if total_allocated > 0 else 0.0

        return PaddingAnalysis(
            sequence_capacity=sequence_capacity,
            total_allocated_tokens=total_allocated,
            actual_tokens=actual_tokens,
            estimated_padding_tokens=padding_tokens,
            estimated_padding_ratio=round(padding_ratio, 4),
        )
