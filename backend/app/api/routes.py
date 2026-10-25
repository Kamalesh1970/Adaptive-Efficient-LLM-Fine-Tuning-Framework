"""
API Route Handlers for Adaptive LLM Fine-Tuning Studio (Phase 0).
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status

from backend.app.core.config import settings
from backend.app.core.system import get_system_hardware_info
from backend.app.schemas.system import HealthResponse, SystemStatusResponse
from backend.app.schemas.training import (
    TrainingConfigSchema,
    TrainingConfigValidationResult,
)
from backend.app.services.config_service import (
    get_default_training_config,
    validate_training_config,
)

router = APIRouter(prefix="/api", tags=["Studio API"])


@router.get("/health", response_model=HealthResponse, summary="Service Health Check")
def get_health() -> HealthResponse:
    """Returns the operational health and version of the studio backend."""
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.version,
    )


@router.get("/system/status", response_model=SystemStatusResponse, summary="Hardware and Environment Status")
def get_system_status() -> SystemStatusResponse:
    """
    Returns actual system, Python, PyTorch, and CUDA hardware status.
    Gracefully handles missing PyTorch and CPU-only setups.
    """
    try:
        hardware_info = get_system_hardware_info()
        return SystemStatusResponse(**hardware_info)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to probe system hardware: {str(e)}",
        )


@router.get(
    "/training/config/defaults",
    response_model=TrainingConfigSchema,
    summary="Get Baseline Training Configuration",
)
def get_training_config_defaults() -> TrainingConfigSchema:
    """Returns default baseline training configuration with initial unselected state."""
    return get_default_training_config()


@router.post(
    "/training/config/validate",
    response_model=TrainingConfigValidationResult,
    summary="Validate Training Configuration",
)
def post_validate_training_config(config_data: Dict[str, Any]) -> TrainingConfigValidationResult:
    """Validates provided training configuration against parameters and readiness criteria."""
    return validate_training_config(config_data)
