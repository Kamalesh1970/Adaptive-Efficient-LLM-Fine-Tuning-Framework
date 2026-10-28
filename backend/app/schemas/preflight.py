"""
Pre-Launch Preflight Check and Run Plan Schemas for Adaptive LLM Fine-Tuning Studio (Phase 6).

Defines strongly-typed structures for the training run preflight validation result,
resource compatibility assessment, and serializable RunPlan.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PreflightStatus(str, Enum):
    """Overall outcome status of a single preflight check."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"
    VALIDATED = "validated"
    INVALID = "invalid"
    NOT_CHECKED = "not_checked"


class PreflightCheckResult(BaseModel):
    """Result of a single preflight validation check."""
    check_id: str = Field(description="Machine-readable check identifier")
    label: str = Field(description="Human-readable check label")
    status: PreflightStatus = Field(description="pass, fail, warn, skip, unknown, unsupported")
    message: str = Field(description="Detailed human-readable explanation of the outcome")
    detail: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional structured supplemental detail data",
    )


class PreflightCheckResponse(BaseModel):
    """
    Aggregated result of all pre-launch preflight checks.

    Fields:
    - checks: Ordered list of individual check results.
    - can_launch: True only if all required checks pass (no blocking failures).
    - blocking_count: Number of hard-blocking FAIL results.
    - warning_count: Number of non-blocking WARN results.
    - method: Fine-tuning method that was validated.
    - estimated_vram_gb: Rough VRAM estimate, or None if not determinable.
    - vram_status: Compatibility classification ('supported', 'unsupported', 'unknown', or None).
    - vram_message: Explanation of VRAM assessment.
    """
    checks: List[PreflightCheckResult] = Field(
        default_factory=list,
        description="Ordered list of all individual preflight check results",
    )
    can_launch: bool = Field(
        description="Whether all required preflight checks passed and training can be launched",
    )
    blocking_count: int = Field(
        default=0,
        description="Number of hard-blocking FAIL results that prevent launch",
    )
    warning_count: int = Field(
        default=0,
        description="Number of non-blocking WARN results for informational awareness",
    )
    method: Optional[str] = Field(
        default=None,
        description="Fine-tuning method being validated (lora, qlora, full_finetuning)",
    )
    estimated_vram_gb: Optional[float] = Field(
        default=None,
        description="Rough estimated VRAM requirement for the selected configuration, or None",
    )
    vram_status: Optional[str] = Field(
        default=None,
        description="VRAM compatibility classification: 'supported', 'unsupported', or 'unknown'",
    )
    vram_message: Optional[str] = Field(
        default=None,
        description="Explanation of VRAM assessment",
    )


class RunPlan(BaseModel):
    """
    Structured, serializable pre-launch training plan.

    Contains complete run configuration, model and dataset references, hardware info,
    validation results, warnings, resource compatibility, and output configuration.
    Guaranteed not to contain secrets, access tokens, or sensitive absolute host paths.
    """
    run_configuration: Dict[str, Any] = Field(
        description="Sanitized training configuration dictionary without tokens or credentials",
    )
    selected_model: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata summary for selected base model",
    )
    dataset_reference: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata summary for selected dataset",
    )
    fine_tuning_method: str = Field(
        description="Fine-tuning method: 'lora', 'qlora', or 'full_finetuning'",
    )
    device_information: Dict[str, Any] = Field(
        description="Detected compute device and GPU hardware info",
    )
    validation_results: PreflightCheckResponse = Field(
        description="Complete preflight check results",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Aggregated warnings from validation and configuration checks",
    )
    resource_compatibility: Dict[str, Any] = Field(
        description="Detailed resource compatibility assessment",
    )
    output_configuration: Dict[str, Any] = Field(
        description="Output directory and logging configuration (safe relative paths)",
    )
    can_launch: bool = Field(
        description="Whether this plan meets all prerequisites for launching",
    )
    blocking_count: int = Field(
        default=0,
        description="Count of blocking validation failures",
    )
    warning_count: int = Field(
        default=0,
        description="Count of non-blocking warnings",
    )
    created_at: str = Field(
        description="ISO 8601 UTC timestamp of plan generation",
    )


class RunValidationResponse(BaseModel):
    """Response returned by /api/training/validate-run."""
    can_launch: bool = Field(description="Whether the run can proceed to launch")
    blocking_count: int = Field(description="Number of blocking issues preventing launch")
    warning_count: int = Field(description="Number of warnings")
    checks: List[PreflightCheckResult] = Field(description="List of all check outcomes")
    plan: RunPlan = Field(description="Generated RunPlan")
