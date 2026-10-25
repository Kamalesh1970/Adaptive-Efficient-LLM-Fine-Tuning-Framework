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
    train_ratio: train_ratio.toString(),
    eval_ratio: eval_ratio.toString(),
    seed: seed.toString(),
  });
  return request<DatasetSplitResult>(`/datasets/${encodeURIComponent(datasetId)}/splits?${query.toString()}`);
}
