"""
Dataset Profile Structured Container.
"""

import json
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class DatasetProfile(BaseModel):
    """Structured dataset characteristics profile serializable to JSON."""
    num_samples: int = Field(default=0, description="Total clean samples in dataset")
    total_tokens: int = Field(default=0, description="Sum of all tokens across dataset")
    min_length: int = Field(default=0, description="Minimum sequence token length")
    max_length: int = Field(default=0, description="Maximum sequence token length")
    mean_length: float = Field(default=0.0, description="Mean sequence token length")
    median_length: float = Field(default=0.0, description="Median sequence token length")
    std_length: float = Field(default=0.0, description="Standard deviation of sequence lengths")
    duplicate_count: int = Field(default=0, description="Count of duplicate content samples detected")
    invalid_count: int = Field(default=0, description="Count of invalid samples detected during schema validation")
    estimated_padding_ratio: float = Field(default=0.0, description="Estimated ratio of padding waste")
    schema_type: str = Field(default="unknown", description="Identified dataset schema pattern")
    length_distribution: Dict[str, int] = Field(default_factory=dict, description="Binned token length count distribution")
    percentiles: Dict[str, float] = Field(default_factory=dict, description="Token length percentiles (p50, p75, p90, p95, p99)")

    def to_json(self, indent: int = 2) -> str:
        """Serializes profile instance to JSON string."""
        return json.dumps(self.model_dump(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "DatasetProfile":
        """Deserializes DatasetProfile instance from JSON string."""
        data = json.loads(json_str)
        return cls(**data)
