/// <reference types="vite/client" />
import { HealthStatus, SystemStatus, TrainingConfig, ValidationResult } from '../types/training';

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
    const res = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
      },
      ...options,
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
