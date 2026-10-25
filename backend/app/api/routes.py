"""
API Route Handlers for Adaptive LLM Fine-Tuning Studio (Phase 0).
"""

from typing import Dict, Any, Optional
import json
from email.parser import BytesParser
from email import policy
from fastapi import APIRouter, HTTPException, status, Request, Query

from backend.app.core.config import settings
from backend.app.core.system import get_system_hardware_info
from backend.app.schemas.system import HealthResponse, SystemStatusResponse
from backend.app.schemas.training import (
    TrainingConfigSchema,
    TrainingConfigValidationResult,
)
from backend.app.schemas.models import (
    ModelMetadata,
    ModelCatalogResponse,
    ModelValidationResponse,
    CompatibilityStatus,
)
from backend.app.schemas.datasets import (
    DatasetValidationReport,
    DatasetListResponse,
    DatasetSplitResult,
    DatasetValidationRequest,
)
from backend.app.services.model_catalog import get_model_catalog_service
from backend.app.services.dataset_service import get_dataset_service
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


# ---------------------------------------------------------
# Phase 1: Model Selection & Compatibility Endpoints
# ---------------------------------------------------------

@router.get(
    "/models",
    response_model=ModelCatalogResponse,
    summary="List Available Model Catalog",
)
def get_model_catalog() -> ModelCatalogResponse:
    """
    Returns the catalog of registered causal language models suitable
    for fine-tuning experimentation.
    """
    service = get_model_catalog_service()
    models = service.get_all_models()
    return ModelCatalogResponse(models=models, total=len(models))


@router.get(
    "/models/{model_id:path}",
    response_model=ModelMetadata,
    summary="Get Model Metadata by Identifier",
)
def get_model_metadata(model_id: str) -> ModelMetadata:
    """
    Returns metadata for a specific supported model identifier.
    Returns 400 if model identifier is empty, or 404 if not found in catalog.
    """
    cleaned_id = model_id.strip() if model_id else ""
    if not cleaned_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model identifier cannot be empty.",
        )

    service = get_model_catalog_service()
    model = service.get_model(cleaned_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{cleaned_id}' not found in catalog.",
        )
    return model


@router.post(
    "/models/validate",
    response_model=ModelValidationResponse,
    summary="Validate Model Selection and Compatibility",
)
def post_validate_model_selection(
    payload: Dict[str, Any],
) -> ModelValidationResponse:
    """
    Validates a requested model selection and fine-tuning method.
    Evaluates method compatibility, architectural constraints, and gated access credentials.
    Access tokens are strictly sanitized and never leaked in logs or error responses.
    """
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request payload must be a JSON object.",
        )

    service = get_model_catalog_service()
    model_id = payload.get("model_id")
    training_method = payload.get("training_method")
    hf_token = payload.get("hf_token")
    context_length = payload.get("context_length")

    # Validate context_length type if provided
    if context_length is not None and not isinstance(context_length, int):
        try:
            context_length = int(context_length)
        except (ValueError, TypeError):
            return ModelValidationResponse(
                is_valid=False,
                model_id=str(model_id) if model_id else None,
                training_method=str(training_method) if training_method else None,
                compatibility_status=CompatibilityStatus.INCOMPATIBLE,
                errors=["context_length must be an integer."],
                model=None,
                token_status="not_provided",
            )

    return service.validate_selection(
        model_id=str(model_id) if model_id is not None else None,
        training_method=str(training_method) if training_method is not None else None,
        hf_token=str(hf_token) if hf_token is not None else None,
        context_length=context_length,
    )


# ---------------------------------------------------------
# Phase 2: Dataset Upload, Validation & Split Endpoints
# ---------------------------------------------------------

@router.post(
    "/datasets/upload",
    response_model=DatasetValidationReport,
    summary="Upload and Validate Dataset File",
)
async def upload_dataset(
    request: Request,
    filename: Optional[str] = Query(default=None),
) -> DatasetValidationReport:
    """
    Uploads a dataset file (JSON, JSONL, CSV), validates structure,
    detects fine-tuning schema, and returns a detailed validation report.
    Never exposes internal filesystem paths in response.
    """
    service = get_dataset_service()
    content_type = request.headers.get("content-type", "")
    raw_body = await request.body()

    if len(raw_body) > service.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({len(raw_body)} bytes) exceeds maximum upload limit ({service.max_upload_size_bytes} bytes).",
        )

    file_bytes: Optional[bytes] = None
    target_filename: Optional[str] = None

    if "multipart/form-data" in content_type:
        try:
            msg = BytesParser(policy=policy.default).parsebytes(
                f"Content-Type: {content_type}\r\n\r\n".encode("latin-1") + raw_body
            )
            for part in msg.iter_parts():
                fn = part.get_filename()
                if fn:
                    target_filename = fn
                    file_bytes = part.get_payload(decode=True)
                    break
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to parse multipart form data: {str(e)}",
            )
    elif "application/json" in content_type:
        try:
            json_body = json.loads(raw_body.decode("utf-8"))
            if isinstance(json_body, dict) and "filename" in json_body and "content" in json_body:
                target_filename = json_body["filename"]
                content_val = json_body["content"]
                file_bytes = content_val.encode("utf-8") if isinstance(content_val, str) else json.dumps(content_val).encode("utf-8")
        except Exception:
            pass

    if file_bytes is None:
        target_filename = filename or request.headers.get("x-filename") or "dataset.json"
        file_bytes = raw_body

    if not target_filename:
        target_filename = "dataset.json"

    # Validate file extension
    try:
        file_format = service.get_format_from_filename(target_filename)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # Store bytes securely (prevents path traversal, avoids filename collision)
    try:
        dataset_id, safe_name, fmt = service.store_upload_bytes(file_bytes, target_filename)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(e),
        )

    # Decode text safely
    try:
        text_content = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text_content = file_bytes.decode("latin-1")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unable to decode text file: {str(e)}",
            )

    report = service.parse_and_validate(
        dataset_id=dataset_id,
        original_filename=safe_name,
        file_format=fmt,
        content=text_content,
        file_size_bytes=len(file_bytes),
    )
    return report


@router.get(
    "/datasets",
    response_model=DatasetListResponse,
    summary="List Registered Datasets",
)
def list_datasets() -> DatasetListResponse:
    """Returns all uploaded and processed dataset summaries."""
    service = get_dataset_service()
    datasets = service.list_datasets()
    return DatasetListResponse(datasets=datasets, total=len(datasets))


@router.get(
    "/datasets/{dataset_id}",
    response_model=DatasetValidationReport,
    summary="Get Dataset Validation Report",
)
def get_dataset_report(dataset_id: str) -> DatasetValidationReport:
    """Returns detailed validation report for a specific dataset ID."""
    service = get_dataset_service()
    report = service.get_report(dataset_id)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found.",
        )
    return report


@router.post(
    "/datasets/{dataset_id}/validate",
    response_model=DatasetValidationReport,
    summary="Validate Dataset Configuration and Splits",
)
def validate_dataset(
    dataset_id: str,
    payload: Optional[DatasetValidationRequest] = None,
) -> DatasetValidationReport:
    """Re-validates dataset configuration or custom split requests."""
    service = get_dataset_service()
    report = service.get_report(dataset_id)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found.",
        )
    return report


@router.get(
    "/datasets/{dataset_id}/splits",
    response_model=DatasetSplitResult,
    summary="Compute Deterministic Train/Evaluation Splits",
)
def get_dataset_splits(
    dataset_id: str,
    train_ratio: float = Query(default=0.9, ge=0.0, le=1.0),
    eval_ratio: float = Query(default=0.1, ge=0.0, le=1.0),
    seed: int = Query(default=42),
    shuffle: bool = Query(default=True),
) -> DatasetSplitResult:
    """
    Computes deterministic, non-overlapping train and evaluation splits.
    Guarantees 0% data leakage across split partitions.
    """
    service = get_dataset_service()
    if not service.get_report(dataset_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found.",
        )

    try:
        return service.compute_splits(
            dataset_id=dataset_id,
            train_ratio=train_ratio,
            eval_ratio=eval_ratio,
            seed=seed,
            shuffle=shuffle,
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found.",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

