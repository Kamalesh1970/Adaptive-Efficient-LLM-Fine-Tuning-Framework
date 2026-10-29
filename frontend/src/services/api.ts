/// <reference types="vite/client" />
import {
  HealthStatus,
  SystemStatus,
  TrainingConfig,
  ValidationResult,
  ModelCatalogResponse,
  ModelMetadata,
  ModelValidationRequest,
  ModelValidationResponse,
  DatasetValidationReport,
  DatasetListResponse,
  DatasetSplitResult,
  TrainingRunCreateRequest,
  TrainingRunResponse,
  TrainingArtifactMetadata,
  PreflightCheckResponse,
  RunPlan,
  RunValidationResponse,
  TrainingEventListResponse,
  TrainingLogListResponse,
  CheckpointListResponse,
  RunHistoryListResponse,
  ArtifactMetadata,
  RunArtifactsListResponse,
  ArtifactValidationResponse,
  ArtifactExportResponse,
  EvaluationConfig,
  EvaluationRun,
  EvaluationListResponse,
  EvaluationResultsResponse,
} from '../types/training';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

export class ApiError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${BASE_URL}${endpoint}`;
  try {
    const headers: Record<string, string> = {};
    if (!(options?.body instanceof FormData)) {
      headers['Content-Type'] = 'application/json';
    }

    const res = await fetch(url, {
      ...options,
      headers: {
        ...headers,
        ...options?.headers,
      },
    });

    if (!res.ok) {
      let errorMsg = `Server responded with ${res.status}`;
      try {
        const errorData = await res.json();
        if (errorData.detail) {
          errorMsg = typeof errorData.detail === 'string' ? errorData.detail : JSON.stringify(errorData.detail);
        }
      } catch {
        // Fallback to status text
      }
      throw new ApiError(errorMsg, res.status);
    }

    return await res.json();
  } catch (err: unknown) {
    if (err instanceof ApiError) {
      throw err;
    }
    const message = err instanceof Error ? err.message : 'Failed to connect to backend server';
    throw new ApiError(message);
  }
}

export async function fetchHealth(): Promise<HealthStatus> {
  return request<HealthStatus>('/health');
}

export async function fetchSystemStatus(): Promise<SystemStatus> {
  return request<SystemStatus>('/system/status');
}

export async function fetchDefaultConfig(): Promise<TrainingConfig> {
  return request<TrainingConfig>('/training/config/defaults');
}

export async function validateConfig(config: TrainingConfig): Promise<ValidationResult> {
  return request<ValidationResult>('/training/config/validate', {
    method: 'POST',
    body: JSON.stringify(config),
  });
}

export async function exportConfigYaml(config: TrainingConfig): Promise<{ yaml_content: string }> {
  return request<{ yaml_content: string }>('/training/config/yaml/export', {
    method: 'POST',
    body: JSON.stringify(config),
  });
}

export async function parseConfigYaml(yamlContent: string): Promise<ValidationResult> {
  return request<ValidationResult>('/training/config/yaml/parse', {
    method: 'POST',
    body: JSON.stringify({ yaml_content: yamlContent }),
  });
}

export async function resetConfig(): Promise<TrainingConfig> {
  return request<TrainingConfig>('/training/config/reset', {
    method: 'POST',
  });
}

// ---------------------------------------------------------
// Phase 1: Model Selection & Compatibility Endpoints
// ---------------------------------------------------------

export async function fetchModelCatalog(): Promise<ModelCatalogResponse> {
  return request<ModelCatalogResponse>('/models');
}

export async function fetchModelMetadata(modelId: string): Promise<ModelMetadata> {
  return request<ModelMetadata>(`/models/${encodeURIComponent(modelId)}`);
}

export async function validateModelSelection(
  req: ModelValidationRequest
): Promise<ModelValidationResponse> {
  return request<ModelValidationResponse>('/models/validate', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

// ---------------------------------------------------------
// Phase 2: Dataset Upload, Validation & Split Endpoints
// ---------------------------------------------------------

export async function uploadDatasetFile(file: File): Promise<DatasetValidationReport> {
  const formData = new FormData();
  formData.append('file', file, file.name);
  return request<DatasetValidationReport>('/datasets/upload', {
    method: 'POST',
    body: formData,
  });
}

export async function fetchDatasets(): Promise<DatasetListResponse> {
  return request<DatasetListResponse>('/datasets');
}

export async function fetchDatasetReport(datasetId: string): Promise<DatasetValidationReport> {
  return request<DatasetValidationReport>(`/datasets/${encodeURIComponent(datasetId)}`);
}

export async function fetchDatasetSplits(
  datasetId: string,
  trainRatio = 0.9,
  evalRatio = 0.1,
  seed = 42
): Promise<DatasetSplitResult> {
  const query = new URLSearchParams({
    train_ratio: trainRatio.toString(),
    eval_ratio: evalRatio.toString(),
    seed: seed.toString(),
  });
  return request<DatasetSplitResult>(`/datasets/${encodeURIComponent(datasetId)}/splits?${query.toString()}`);
}

// ---------------------------------------------------------
// Phase 4: Training Run Execution Endpoints
// ---------------------------------------------------------

export async function launchTrainingRun(
  req: TrainingRunCreateRequest
): Promise<TrainingRunResponse> {
  return request<TrainingRunResponse>('/training/runs', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

export async function fetchTrainingRun(runId: string): Promise<TrainingRunResponse> {
  return request<TrainingRunResponse>(`/training/runs/${encodeURIComponent(runId)}`);
}

export async function cancelTrainingRun(runId: string): Promise<TrainingRunResponse> {
  return request<TrainingRunResponse>(`/training/runs/${encodeURIComponent(runId)}/cancel`, {
    method: 'POST',
  });
}

export async function fetchRunArtifacts(runId: string): Promise<TrainingArtifactMetadata> {
  return request<TrainingArtifactMetadata>(`/training/runs/${encodeURIComponent(runId)}/artifacts`);
}

export async function fetchArtifactsForRun(runId: string): Promise<RunArtifactsListResponse> {
  return request<RunArtifactsListResponse>(`/training/runs/${encodeURIComponent(runId)}/artifacts`);
}

// ---------------------------------------------------------
// Phase 6: Pre-Launch Preflight Validation
// ---------------------------------------------------------

export async function runPreflight(
  config: TrainingConfig,
  hfToken?: string | null,
): Promise<PreflightCheckResponse> {
  const payload: Record<string, unknown> = { ...config };
  if (hfToken && hfToken.trim()) {
    payload['hf_token'] = hfToken.trim();
  }
  return request<PreflightCheckResponse>('/training/preflight', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function fetchRunPreview(
  config: TrainingConfig,
  hfToken?: string | null,
): Promise<RunPlan> {
  const payload: Record<string, unknown> = { config };
  if (hfToken && hfToken.trim()) {
    payload['hf_token'] = hfToken.trim();
  }
  return request<RunPlan>('/training/preview', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function validateRun(
  config: TrainingConfig,
  hfToken?: string | null,
): Promise<RunValidationResponse> {
  const payload: Record<string, unknown> = { config };
  if (hfToken && hfToken.trim()) {
    payload['hf_token'] = hfToken.trim();
  }
  return request<RunValidationResponse>('/training/validate-run', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

// ---------------------------------------------------------
// Phase 7: Real-Time Training Monitoring & History Endpoints
// ---------------------------------------------------------

export async function fetchRunHistory(
  page = 1,
  pageSize = 20,
  status?: string
): Promise<RunHistoryListResponse> {
  const params = new URLSearchParams({
    page: page.toString(),
    page_size: pageSize.toString(),
  });
  if (status && status !== 'all') {
    params.append('status', status);
  }
  return request<RunHistoryListResponse>(`/training/runs?${params.toString()}`);
}

export async function fetchRunEvents(
  runId: string,
  sinceEventId?: string,
  limit = 100
): Promise<TrainingEventListResponse> {
  const params = new URLSearchParams({
    limit: limit.toString(),
  });
  if (sinceEventId) {
    params.append('since_event_id', sinceEventId);
  }
  return request<TrainingEventListResponse>(
    `/training/runs/${encodeURIComponent(runId)}/events?${params.toString()}`
  );
}

export async function fetchRunLogs(
  runId: string,
  limit = 500
): Promise<TrainingLogListResponse> {
  const params = new URLSearchParams({
    limit: limit.toString(),
  });
  return request<TrainingLogListResponse>(
    `/training/runs/${encodeURIComponent(runId)}/logs?${params.toString()}`
  );
}

export async function fetchRunCheckpoints(
  runId: string
): Promise<CheckpointListResponse> {
  return request<CheckpointListResponse>(
    `/training/runs/${encodeURIComponent(runId)}/checkpoints`
  );
}

// ---------------------------------------------------------
// Phase 8: Artifacts, Export & Model Evaluation
// ---------------------------------------------------------

export async function fetchAllArtifacts(runId?: string): Promise<ArtifactMetadata[]> {
  const query = runId ? `?run_id=${encodeURIComponent(runId)}` : '';
  return request<ArtifactMetadata[]>(`/artifacts${query}`);
}

export async function fetchArtifactDetails(artifactId: string): Promise<ArtifactMetadata> {
  return request<ArtifactMetadata>(`/artifacts/${encodeURIComponent(artifactId)}`);
}

export async function validateArtifact(
  artifactId: string,
  checkLoadable = false
): Promise<ArtifactValidationResponse> {
  return request<ArtifactValidationResponse>(`/artifacts/${encodeURIComponent(artifactId)}/validate`, {
    method: 'POST',
    body: JSON.stringify({ check_loadable: checkLoadable }),
  });
}

export async function exportArtifact(artifactId: string): Promise<ArtifactExportResponse> {
  return request<ArtifactExportResponse>(`/artifacts/${encodeURIComponent(artifactId)}/export`, {
    method: 'POST',
  });
}

export function getArtifactDownloadUrl(artifactId: string): string {
  return `${BASE_URL}/artifacts/${encodeURIComponent(artifactId)}/download`;
}

export async function createEvaluation(config: EvaluationConfig): Promise<EvaluationRun> {
  return request<EvaluationRun>('/evaluations', {
    method: 'POST',
    body: JSON.stringify(config),
  });
}

export async function fetchEvaluations(params?: {
  run_id?: string;
  artifact_id?: string;
  status?: string;
  page?: number;
  page_size?: number;
}): Promise<EvaluationListResponse> {
  const searchParams = new URLSearchParams();
  if (params?.run_id) searchParams.append('run_id', params.run_id);
  if (params?.artifact_id) searchParams.append('artifact_id', params.artifact_id);
  if (params?.status) searchParams.append('status', params.status);
  if (params?.page) searchParams.append('page', params.page.toString());
  if (params?.page_size) searchParams.append('page_size', params.page_size.toString());

  const queryStr = searchParams.toString() ? `?${searchParams.toString()}` : '';
  return request<EvaluationListResponse>(`/evaluations${queryStr}`);
}

export async function fetchEvaluationDetails(evaluationId: string): Promise<EvaluationRun> {
  return request<EvaluationRun>(`/evaluations/${encodeURIComponent(evaluationId)}`);
}

export async function cancelEvaluation(evaluationId: string): Promise<EvaluationRun> {
  return request<EvaluationRun>(`/evaluations/${encodeURIComponent(evaluationId)}/cancel`, {
    method: 'POST',
  });
}

export async function fetchEvaluationResults(evaluationId: string): Promise<EvaluationResultsResponse> {
  return request<EvaluationResultsResponse>(`/evaluations/${encodeURIComponent(evaluationId)}/results`);
}


