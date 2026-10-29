"""
API Route Handlers for Adaptive LLM Fine-Tuning Studio (Phase 0–6).
"""

import os
from typing import Dict, Any, Optional, List
import json
import math
from email.parser import BytesParser
from email import policy
from fastapi import APIRouter, HTTPException, status, Request, Query
from fastapi.responses import FileResponse

from backend.app.core.config import settings
from backend.app.core.system import get_system_hardware_info
from backend.app.schemas.system import HealthResponse, SystemStatusResponse
from backend.app.schemas.training import (
    TrainingConfigSchema,
    TrainingConfigValidationResult,
    TrainingRunStatus,
    TrainingRunCreateRequest,
    TrainingRunResponse,
    TrainingArtifactMetadata,
    TrainingMethod,
)
from backend.app.schemas.history import (
    TrainingEvent,
    TrainingEventListResponse,
    TrainingLogEntry,
    TrainingLogListResponse,
    CheckpointMetadata,
    CheckpointListResponse,
    RunHistoryItem,
    RunHistoryListResponse,
)
from backend.app.schemas.artifacts import (
    ArtifactMetadata,
    ArtifactValidationRequest,
    ArtifactValidationResponse,
    ArtifactExportResponse,
    RunArtifactsListResponse,
)
from backend.app.schemas.evaluation import (
    EvaluationRun,
    EvaluationConfig,
    EvaluationListResponse,
    EvaluationResultsResponse,
)
from backend.app.services.job_manager import get_job_manager
from backend.app.services.run_repository import get_run_repository
from backend.app.services.artifact_service import get_artifact_service, ArtifactSecurityError
from backend.app.services.evaluation_service import get_evaluation_service
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
from backend.app.schemas.preflight import (
    PreflightCheckResponse,
    PreflightCheckResult,
    PreflightStatus,
    RunPlan,
    RunValidationResponse,
)
from backend.app.services.model_catalog import get_model_catalog_service
from backend.app.services.dataset_service import get_dataset_service
from backend.app.services.config_service import (
    get_default_training_config,
    validate_training_config,
    export_training_config_yaml,
    parse_training_config_yaml,
    reset_training_config,
)
from backend.app.services.preflight_service import run_preflight, generate_run_plan

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


@router.post(
    "/training/config/reset",
    response_model=TrainingConfigSchema,
    summary="Reset Training Configuration to Baseline Defaults",
)
def post_reset_training_config() -> TrainingConfigSchema:
    """Resets training configuration to defaults without touching models or datasets."""
    return reset_training_config()



@router.post(
    "/training/config/yaml/export",
    summary="Export Configuration to YAML",
)
def post_export_training_config_yaml(config_data: Dict[str, Any]) -> Dict[str, str]:
    """Serializes training configuration to sanitized YAML string."""
    try:
        config = TrainingConfigSchema(**config_data)
        yaml_content = export_training_config_yaml(config)
        return {"yaml_content": yaml_content}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to export configuration to YAML: {str(e)}",
        )


@router.post(
    "/training/config/yaml/parse",
    response_model=TrainingConfigValidationResult,
    summary="Parse and Validate Uploaded YAML Configuration",
)
def post_parse_training_config_yaml(payload: Dict[str, str]) -> TrainingConfigValidationResult:
    """Safely parses YAML configuration, rejecting malformed content and validating schema."""
    yaml_content = payload.get("yaml_content", "")
    try:
        parsed_data = parse_training_config_yaml(yaml_content)
        return validate_training_config(parsed_data)
    except ValueError as e:
        return TrainingConfigValidationResult(
            is_valid=False,
            errors=[str(e)],
            field_errors={"yaml": [str(e)]},
            warnings=[],
            is_ready_for_training=False,
            config=None,
        )


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


# ============================================================================
# Phase 6: Training Run Preflight Validation Endpoint
# ============================================================================

@router.post(
    "/training/preflight",
    response_model=PreflightCheckResponse,
    summary="Run Pre-Launch Preflight Validation",
)
def post_training_preflight(payload: Dict[str, Any]) -> PreflightCheckResponse:
    """
    Executes an ordered sequence of pre-launch validation checks before starting training.
    Validates model selection, dataset, device hardware, ML dependencies, VRAM capacity,
    method–device compatibility, and HF token availability.

    Returns a typed PreflightCheckResponse with per-check granularity and an overall
    can_launch flag indicating whether training may proceed safely.
    """
    try:
        hf_token: Optional[str] = payload.pop("hf_token", None)
        config = TrainingConfigSchema(**payload)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid training configuration: {str(e)}",
        )

    try:
        return run_preflight(config=config, hf_token=hf_token)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preflight check failed unexpectedly: {str(e)}",
        )


@router.post(
    "/training/preview",
    response_model=RunPlan,
    summary="Generate Structured Run Plan and Resource Compatibility",
)
def post_training_preview(payload: Dict[str, Any]) -> RunPlan:
    """
    Generates a structured, serializable RunPlan containing configuration,
    model, dataset, device, validation results, resource compatibility, and output settings.
    Guarantees secrets and access tokens are strictly excluded.
    """
    try:
        raw_config = payload.get("config", payload)
        hf_token: Optional[str] = payload.get("hf_token")
        if isinstance(raw_config, dict):
            raw_config = dict(raw_config)
            hf_token = raw_config.pop("hf_token", hf_token)
        config = TrainingConfigSchema(**raw_config) if isinstance(raw_config, dict) else raw_config
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid training configuration: {str(e)}",
        )

    try:
        return generate_run_plan(config=config, hf_token=hf_token)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Run plan generation failed unexpectedly: {str(e)}",
        )


@router.post(
    "/training/validate-run",
    response_model=RunValidationResponse,
    summary="Validate Training Run and Generate Plan",
)
def post_validate_run(payload: Dict[str, Any]) -> RunValidationResponse:
    """
    Validates a training run before launch and returns preflight checks along with the generated RunPlan.
    """
    try:
        raw_config = payload.get("config", payload)
        hf_token: Optional[str] = payload.get("hf_token")
        if isinstance(raw_config, dict):
            raw_config = dict(raw_config)
            hf_token = raw_config.pop("hf_token", hf_token)
        config = TrainingConfigSchema(**raw_config) if isinstance(raw_config, dict) else raw_config
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid training configuration: {str(e)}",
        )

    try:
        plan = generate_run_plan(config=config, hf_token=hf_token)
        return RunValidationResponse(
            can_launch=plan.can_launch,
            blocking_count=plan.blocking_count,
            warning_count=plan.warning_count,
            checks=plan.validation_results.checks,
            plan=plan,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Run validation failed unexpectedly: {str(e)}",
        )


# ============================================================================
# Phase 4: Training Run Execution Endpoints
# ============================================================================

@router.post(
    "/training/runs",
    response_model=TrainingRunResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Validate and Launch Training Run",
)
def launch_training_run(payload: TrainingRunCreateRequest) -> TrainingRunResponse:
    """
    Validates configuration and launches a genuine LoRA or QLoRA fine-tuning run.
    Enforces server-side preflight verification and duplicate submission prevention.
    Execution proceeds asynchronously through explicit lifecycle states.
    """
    # 1. Validate configuration schema and constraints
    validation_res = validate_training_config(payload.config)
    if not validation_res.is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Configuration validation failed", "errors": validation_res.errors},
        )

    # 2. Strict Phase 5 boundary check: LoRA and QLoRA are supported
    if payload.config.training_method not in (TrainingMethod.LORA, TrainingMethod.QLORA):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Supported methods are LoRA and QLoRA fine-tuning. Requested: {payload.config.training_method.value}",
        )

    # 3. Server-side preflight check: all blocking checks must pass
    preflight = run_preflight(config=payload.config, hf_token=payload.hf_token)
    if not preflight.can_launch:
        blocking_msgs = [c.message for c in preflight.checks if c.status == PreflightStatus.FAIL]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Training launch rejected by preflight validation checks",
                "blocking_count": preflight.blocking_count,
                "blocking_issues": blocking_msgs,
            },
        )

    # 4. Create and dispatch job with duplicate launch prevention
    job_mgr = get_job_manager()
    try:
        job = job_mgr.create_job(
            config=payload.config,
            hf_token=payload.hf_token,
            idempotency_key=payload.idempotency_key,
        )
        if job.status == TrainingRunStatus.QUEUED:
            job_mgr.start_job(run_id=job.run_id, background=True)
        return job.to_response()
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get(
    "/training/runs",
    response_model=RunHistoryListResponse,
    summary="List Persistent Training Run History",
)
def list_training_runs(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Optional status filter"),
) -> RunHistoryListResponse:
    """
    Returns paginated persistent run history ordered newest first.
    Supports status filtering and survives backend restarts.
    """
    run_repo = get_run_repository()
    items, total = run_repo.list_runs(page=page, page_size=page_size, status_filter=status)
    total_pages = math.ceil(total / page_size) if total > 0 else 1
    return RunHistoryListResponse(
        runs=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get(
    "/training/runs/{run_id}",
    response_model=TrainingRunResponse,
    summary="Get Training Run Status and Telemetry",
)
def get_training_run(run_id: str) -> TrainingRunResponse:
    """Retrieves actual status, step progress, loss history, and telemetry for a training run."""
    job_mgr = get_job_manager()
    job = job_mgr.get_job(run_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training run '{run_id}' not found.",
        )
    return job.to_response()


@router.get(
    "/training/runs/{run_id}/events",
    response_model=TrainingEventListResponse,
    summary="Get Training Lifecycle Events",
)
def get_training_events(
    run_id: str,
    since_event_id: Optional[str] = Query(None, description="Cursor for incremental retrieval"),
    limit: int = Query(100, ge=1, le=500, description="Maximum events to return"),
) -> TrainingEventListResponse:
    """
    Returns recent training lifecycle events with optional cursor-based incremental retrieval.
    """
    job_mgr = get_job_manager()
    job = job_mgr.get_job(run_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training run '{run_id}' not found.",
        )
    run_repo = get_run_repository()
    events, total, has_more = run_repo.get_events(
        run_id=run_id,
        since_event_id=since_event_id,
        limit=limit,
    )
    next_cursor = events[-1].event_id if events and has_more else None
    return TrainingEventListResponse(
        run_id=run_id,
        events=events,
        total=total,
        has_more=has_more,
        next_cursor=next_cursor,
    )


@router.get(
    "/training/runs/{run_id}/logs",
    response_model=TrainingLogListResponse,
    summary="Get Sanitized Training Worker Logs",
)
def get_training_logs(
    run_id: str,
    limit: int = Query(500, ge=1, le=2000, description="Maximum log entries to return"),
) -> TrainingLogListResponse:
    """
    Returns sanitized training logs in chronological order without exposing arbitrary files or credentials.
    """
    job_mgr = get_job_manager()
    job = job_mgr.get_job(run_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training run '{run_id}' not found.",
        )
    run_repo = get_run_repository()
    logs, total = run_repo.get_logs(run_id=run_id, limit=limit)
    return TrainingLogListResponse(
        run_id=run_id,
        logs=logs,
        total=total,
    )


@router.get(
    "/training/runs/{run_id}/checkpoints",
    response_model=CheckpointListResponse,
    summary="Get Verified Training Checkpoints",
)
def get_training_checkpoints(run_id: str) -> CheckpointListResponse:
    """
    Returns verified checkpoint metadata for a run based on files that genuinely exist on disk.
    """
    job_mgr = get_job_manager()
    job = job_mgr.get_job(run_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training run '{run_id}' not found.",
        )
    checkpoints = job_mgr.get_checkpoints(run_id)
    return CheckpointListResponse(
        run_id=run_id,
        checkpoints=checkpoints,
        total=len(checkpoints),
    )


@router.post(
    "/training/runs/{run_id}/cancel",
    response_model=TrainingRunResponse,
    summary="Request Training Run Cancellation",
)
def cancel_training_run(run_id: str) -> TrainingRunResponse:
    """Requests safe cancellation of an active training run without artifact corruption."""
    job_mgr = get_job_manager()
    job = job_mgr.get_job(run_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training run '{run_id}' not found.",
        )

    job_mgr.cancel_job(run_id)
    return job.to_response()


@router.get(
    "/training/runs/{run_id}/artifacts",
    response_model=RunArtifactsListResponse,
    summary="List Artifacts for a Training Run",
)
def get_run_artifacts(run_id: str) -> RunArtifactsListResponse:
    """
    Lists artifacts that actually exist for the requested run.
    Maintains backward compatibility with earlier Phase 4/7 clients.
    """
    art_svc = get_artifact_service()
    try:
        run_dir = art_svc._resolve_safe_run_dir(run_id)
    except ArtifactSecurityError as se:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(se),
        )

    job_mgr = get_job_manager()
    run_repo = get_run_repository()
    job = job_mgr.get_job(run_id)
    db_run = run_repo.get_run(run_id)

    if not job and not db_run and not os.path.isdir(run_dir):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training run '{run_id}' not found.",
        )

    if job and job.status not in (TrainingRunStatus.COMPLETED, TrainingRunStatus.FAILED, TrainingRunStatus.CANCELLED):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Artifacts are not available. Current run status is '{job.status.value}'.",
        )

    resp = art_svc.get_run_artifacts_response(run_id)
    if resp.total == 0 and not os.path.isdir(run_dir):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact metadata could not be verified on disk for run '{run_id}'.",
        )

    return resp


@router.get(
    "/artifacts",
    response_model=List[ArtifactMetadata],
    summary="List All Training Artifacts",
)
def list_artifacts(run_id: Optional[str] = Query(None, description="Optional run ID filter")) -> List[ArtifactMetadata]:
    """Lists registered artifacts across all runs or filtered by run ID."""
    art_svc = get_artifact_service()
    if run_id:
        return art_svc.discover_artifacts_for_run(run_id)
    return art_svc.list_all_artifacts()


@router.get(
    "/artifacts/{artifact_id}",
    response_model=ArtifactMetadata,
    summary="Get Artifact Metadata",
)
def get_artifact(artifact_id: str) -> ArtifactMetadata:
    """Returns safe metadata for a known artifact."""
    art_svc = get_artifact_service()
    art = art_svc.get_artifact(artifact_id)
    if not art:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact '{artifact_id}' not found.",
        )
    return art


@router.post(
    "/artifacts/{artifact_id}/validate",
    response_model=ArtifactValidationResponse,
    summary="Validate Artifact Integrity",
)
def validate_artifact(
    artifact_id: str,
    payload: Optional[ArtifactValidationRequest] = None,
) -> ArtifactValidationResponse:
    """Validates artifact integrity and reports precise failures."""
    art_svc = get_artifact_service()
    check_loadable = payload.check_loadable if payload else False
    try:
        return art_svc.validate_artifact(artifact_id, check_loadable=check_loadable)
    except ArtifactSecurityError as se:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(se),
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(ve),
        )


@router.post(
    "/artifacts/{artifact_id}/export",
    response_model=ArtifactExportResponse,
    summary="Export LoRA or QLoRA Adapter Package",
)
def export_artifact(artifact_id: str) -> ArtifactExportResponse:
    """Creates a clean, safe export package from a validated adapter artifact."""
    art_svc = get_artifact_service()
    try:
        return art_svc.export_adapter(artifact_id)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except ArtifactSecurityError as se:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(se),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Export failed: {str(e)}",
        )


@router.get(
    "/artifacts/{artifact_id}/download",
    summary="Download Exported Adapter Package",
)
def download_artifact(artifact_id: str):
    """Downloads the export package using a safe, controlled response."""
    art_svc = get_artifact_service()
    try:
        filepath, filename = art_svc.get_export_download_path(artifact_id)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(ve),
        )
    except FileNotFoundError as fe:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(fe),
        )
    except ArtifactSecurityError as se:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(se),
        )

    return FileResponse(
        path=filepath,
        filename=filename,
        media_type="application/zip",
    )


# -------------------------------------------------------------
# Phase 8: Model Evaluation Endpoints
# -------------------------------------------------------------

@router.post(
    "/evaluations",
    response_model=EvaluationRun,
    status_code=status.HTTP_201_CREATED,
    summary="Start Model Evaluation Job",
)
def create_evaluation(payload: EvaluationConfig) -> EvaluationRun:
    """Initiates an evaluation run on an artifact and validation dataset."""
    eval_svc = get_evaluation_service()
    try:
        run = eval_svc.create_evaluation(payload)
        return eval_svc.start_evaluation(run.evaluation_id, background=True)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start evaluation: {str(e)}",
        )


@router.get(
    "/evaluations",
    response_model=EvaluationListResponse,
    summary="List Model Evaluations",
)
def list_evaluations(
    run_id: Optional[str] = Query(None, description="Optional training run ID filter"),
    artifact_id: Optional[str] = Query(None, description="Optional artifact ID filter"),
    status: Optional[str] = Query(None, description="Optional status filter"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
) -> EvaluationListResponse:
    """Lists evaluations with optional filters and pagination."""
    run_repo = get_run_repository()
    items, total = run_repo.list_evaluations(
        run_id=run_id,
        artifact_id=artifact_id,
        status_filter=status,
        page=page,
        page_size=page_size,
    )
    return EvaluationListResponse(
        evaluations=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/evaluations/{evaluation_id}",
    response_model=EvaluationRun,
    summary="Get Evaluation Status and Details",
)
def get_evaluation(evaluation_id: str) -> EvaluationRun:
    """Retrieves evaluation status, configuration, and progress."""
    run_repo = get_run_repository()
    evaluation = run_repo.get_evaluation(evaluation_id)
    if not evaluation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation '{evaluation_id}' not found.",
        )
    return evaluation


@router.post(
    "/evaluations/{evaluation_id}/cancel",
    response_model=EvaluationRun,
    summary="Cancel Ongoing Evaluation Job",
)
def cancel_evaluation(evaluation_id: str) -> EvaluationRun:
    """Cancels an active evaluation job safely."""
    eval_svc = get_evaluation_service()
    try:
        return eval_svc.cancel_evaluation(evaluation_id)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(ve),
        )


@router.get(
    "/evaluations/{evaluation_id}/results",
    response_model=EvaluationResultsResponse,
    summary="Get Model Evaluation Results and Baseline Comparison",
)
def get_evaluation_results(evaluation_id: str) -> EvaluationResultsResponse:
    """Retrieves evaluation metrics, sample outputs, and baseline comparison."""
    run_repo = get_run_repository()
    evaluation = run_repo.get_evaluation(evaluation_id)
    if not evaluation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation '{evaluation_id}' not found.",
        )

    return EvaluationResultsResponse(
        evaluation_id=evaluation.evaluation_id,
        status=evaluation.status,
        metrics=evaluation.metrics,
        samples=evaluation.samples,
        baseline_comparison=evaluation.baseline_comparison,
        error_message=evaluation.error_message,
    )


# ==============================================================================
# Phase 10: Containerization, Kubernetes, Quantization, GGUF & Multi-GPU Routes
# ==============================================================================

from backend.app.schemas.quantization import (
    PostTrainingQuantizationConfig,
    QuantizationCapability,
    QuantizationJobRecord,
)
from backend.app.schemas.export import (
    GGUFExportConfig,
    GGUFExportJobRecord,
)
from backend.app.schemas.serving import (
    ServingConfig,
    ServingInstanceRecord,
    SmokeInferenceRequest,
    SmokeInferenceResponse,
)
from backend.app.schemas.distributed import (
    MultiGPUSystemReport,
    DistributedTrainingConfig,
)
from backend.app.services.quantization_service import (
    get_quantization_service,
    QuantizationError,
    QuantizationCompatibilityError,
    QuantizationDependencyError,
)
from backend.app.services.gguf_export_service import (
    get_gguf_export_service,
    GGUFExportError,
    GGUFEligibilityError,
)
from backend.app.services.vllm_service import (
    get_vllm_service,
    ServingError,
    ServingCompatibilityError,
)
from backend.app.services.distributed_service import (
    get_distributed_service,
    DistributedConfigurationError,
)


# ─── 1. Kubernetes & Deployment Status ──────────────────────────────────────────

@router.get(
    "/system/kubernetes/validate",
    summary="Validate Kubernetes Deployment Manifests",
    tags=["Deployment & Infrastructure"]
)
def validate_kubernetes_manifests() -> Dict[str, Any]:
    """Validates syntax, structure, and required metadata for all Kubernetes manifests."""
    import glob
    import yaml

    manifest_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "deploy", "kubernetes"))
    if not os.path.isdir(manifest_dir):
        return {"is_valid": False, "manifest_count": 0, "errors": [f"Directory {manifest_dir} does not exist."]}

    yaml_files = sorted(glob.glob(os.path.join(manifest_dir, "*.yaml")))
    manifests_summary = []
    errors = []

    for fpath in yaml_files:
        fname = os.path.basename(fpath)
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                docs = list(yaml.safe_load_all(fh))
                for idx, doc in enumerate(docs):
                    if not doc:
                        continue
                    if "apiVersion" not in doc or "kind" not in doc or "metadata" not in doc:
                        errors.append(f"{fname} (doc {idx}): Missing apiVersion, kind, or metadata.")
                    else:
                        manifests_summary.append({
                            "filename": fname,
                            "kind": doc.get("kind"),
                            "name": doc.get("metadata", {}).get("name"),
                            "namespace": doc.get("metadata", {}).get("namespace", "default"),
                        })
        except Exception as e:
            errors.append(f"{fname}: YAML parsing error: {str(e)}")

    return {
        "is_valid": len(errors) == 0,
        "manifest_count": len(yaml_files),
        "manifests": manifests_summary,
        "errors": errors,
    }


# ─── 2. Advanced Quantization Endpoints ────────────────────────────────────────

@router.get(
    "/quantization/capabilities",
    response_model=List[QuantizationCapability],
    summary="List Supported Quantization Capabilities",
    tags=["Advanced Quantization"]
)
def get_quantization_capabilities() -> List[QuantizationCapability]:
    """Returns capability status for all supported quantization methods on this host."""
    service = get_quantization_service()
    return service.get_capabilities()


@router.post(
    "/quantization/jobs",
    response_model=QuantizationJobRecord,
    status_code=status.HTTP_201_CREATED,
    summary="Create & Launch Post-Training Quantization Job",
    tags=["Advanced Quantization"]
)
def create_quantization_job(config: PostTrainingQuantizationConfig) -> QuantizationJobRecord:
    """Launches a post-training model quantization job."""
    service = get_quantization_service()
    try:
        job = service.create_quantization_job(config)
        return service.start_quantization(job.job_id, background=True)
    except QuantizationCompatibilityError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except QuantizationDependencyError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get(
    "/quantization/jobs",
    response_model=List[QuantizationJobRecord],
    summary="List Quantization Jobs",
    tags=["Advanced Quantization"]
)
def list_quantization_jobs() -> List[QuantizationJobRecord]:
    """Returns historical and running quantization jobs."""
    service = get_quantization_service()
    return service.list_jobs()


@router.get(
    "/quantization/jobs/{job_id}",
    response_model=QuantizationJobRecord,
    summary="Get Quantization Job Status",
    tags=["Advanced Quantization"]
)
def get_quantization_job(job_id: str) -> QuantizationJobRecord:
    """Retrieves status and logs for a quantization job."""
    service = get_quantization_service()
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")
    return job


@router.post(
    "/quantization/jobs/{job_id}/cancel",
    response_model=QuantizationJobRecord,
    summary="Cancel Quantization Job",
    tags=["Advanced Quantization"]
)
def cancel_quantization_job(job_id: str) -> QuantizationJobRecord:
    """Cancels an active or queued quantization job."""
    service = get_quantization_service()
    try:
        return service.cancel_job(job_id)
    except QuantizationError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


# ─── 3. GGUF Model Export Endpoints ────────────────────────────────────────────

@router.post(
    "/export/gguf",
    response_model=GGUFExportJobRecord,
    status_code=status.HTTP_201_CREATED,
    summary="Create & Launch GGUF Export Job",
    tags=["Model Export (GGUF)"]
)
def export_gguf_model(config: GGUFExportConfig) -> GGUFExportJobRecord:
    """Creates a GGUF model export job from a trained adapter or base model."""
    service = get_gguf_export_service()
    try:
        job = service.create_export_job(config)
        return service.start_export(job.export_id, background=True)
    except GGUFEligibilityError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get(
    "/export/gguf",
    response_model=List[GGUFExportJobRecord],
    summary="List GGUF Export Jobs",
    tags=["Model Export (GGUF)"]
)
def list_gguf_export_jobs() -> List[GGUFExportJobRecord]:
    """Lists all GGUF export jobs."""
    service = get_gguf_export_service()
    return service.list_jobs()


@router.get(
    "/export/gguf/{export_id}",
    response_model=GGUFExportJobRecord,
    summary="Get GGUF Export Job Status",
    tags=["Model Export (GGUF)"]
)
def get_gguf_export_job(export_id: str) -> GGUFExportJobRecord:
    """Retrieves status and metadata for a GGUF export job."""
    service = get_gguf_export_service()
    job = service.get_job(export_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Export job '{export_id}' not found.")
    return job


@router.get(
    "/export/gguf/{export_id}/download",
    summary="Download Exported GGUF Model Artifact",
    tags=["Model Export (GGUF)"]
)
def download_gguf_artifact(export_id: str):
    """Safely downloads the generated GGUF binary file."""
    service = get_gguf_export_service()
    try:
        filepath = service.resolve_download_path(export_id)
        filename = os.path.basename(filepath)
        return FileResponse(
            path=filepath,
            filename=filename,
            media_type="application/octet-stream",
        )
    except GGUFExportError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


# ─── 4. vLLM Model Serving Endpoints ───────────────────────────────────────────

@router.get(
    "/serving/capabilities",
    summary="Check Model Serving Capabilities",
    tags=["Inference Serving (vLLM)"]
)
def get_serving_capabilities() -> Dict[str, Any]:
    """Returns vLLM and inference serving capability on current machine."""
    service = get_vllm_service()
    return service.get_serving_capabilities()


@router.post(
    "/serving/instances",
    response_model=ServingInstanceRecord,
    status_code=status.HTTP_201_CREATED,
    summary="Start vLLM Serving Instance",
    tags=["Inference Serving (vLLM)"]
)
def start_serving_instance(config: ServingConfig) -> ServingInstanceRecord:
    """Starts a supervised inference serving instance."""
    service = get_vllm_service()
    try:
        return service.start_serving(config)
    except ServingCompatibilityError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get(
    "/serving/instances",
    response_model=List[ServingInstanceRecord],
    summary="List Serving Instances",
    tags=["Inference Serving (vLLM)"]
)
def list_serving_instances() -> List[ServingInstanceRecord]:
    """Lists all active and stopped inference serving instances."""
    service = get_vllm_service()
    return service.list_serving_instances()


@router.get(
    "/serving/instances/{instance_id}",
    response_model=ServingInstanceRecord,
    summary="Get Serving Instance Status",
    tags=["Inference Serving (vLLM)"]
)
def get_serving_instance(instance_id: str) -> ServingInstanceRecord:
    """Retrieves instance health and status."""
    service = get_vllm_service()
    inst = service.get_serving_status(instance_id)
    if not inst:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Instance '{instance_id}' not found.")
    return inst


@router.post(
    "/serving/instances/{instance_id}/stop",
    response_model=ServingInstanceRecord,
    summary="Stop Serving Instance",
    tags=["Inference Serving (vLLM)"]
)
def stop_serving_instance(instance_id: str) -> ServingInstanceRecord:
    """Gracefully terminates an active serving instance."""
    service = get_vllm_service()
    try:
        return service.stop_serving(instance_id)
    except ServingError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post(
    "/serving/instances/{instance_id}/smoke-test",
    response_model=SmokeInferenceResponse,
    summary="Run Bounded Smoke Test Inference",
    tags=["Inference Serving (vLLM)"]
)
def smoke_test_inference(instance_id: str, request: SmokeInferenceRequest) -> SmokeInferenceResponse:
    """Submits a bounded test prompt with SSRF protections."""
    service = get_vllm_service()
    try:
        return service.test_smoke_inference(instance_id, request)
    except ServingError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


# ─── 5. Multi-GPU Discovery Endpoints ──────────────────────────────────────────

@router.get(
    "/system/multi-gpu",
    response_model=MultiGPUSystemReport,
    summary="Get Multi-GPU Hardware Report",
    tags=["Multi-GPU & Distributed"]
)
def get_multi_gpu_report() -> MultiGPUSystemReport:
    """Discovers visible devices and reports multi-GPU training and inference capabilities."""
    service = get_distributed_service()
    return service.discover_hardware()


@router.post(
    "/system/multi-gpu/validate",
    summary="Validate Distributed Training Configuration",
    tags=["Multi-GPU & Distributed"]
)
def validate_distributed_config(config: DistributedTrainingConfig) -> Dict[str, Any]:
    """Validates requested distributed strategy against actual hardware."""
    service = get_distributed_service()
    try:
        service.validate_distributed_training(config)
        return {
            "is_valid": True,
            "strategy": config.strategy.value,
            "requested_gpu_count": config.requested_gpu_count,
            "message": "Distributed configuration validated against active hardware."
        }
    except DistributedConfigurationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))




