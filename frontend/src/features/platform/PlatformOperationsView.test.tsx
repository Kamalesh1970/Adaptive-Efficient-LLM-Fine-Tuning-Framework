import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { PlatformOperationsView } from './PlatformOperationsView';
import * as api from '../../services/api';
import {
  KubernetesValidationResponse,
  QuantizationCapability,
  MultiGPUSystemReport,
} from '../../types/training';

vi.mock('../../services/api');

const mockK8sValidation: KubernetesValidationResponse = {
  is_valid: true,
  manifest_count: 12,
  manifests: [
    { filename: 'backend-deployment.yaml', kind: 'Deployment', name: 'adaptive-studio-backend' },
    { filename: 'frontend-deployment.yaml', kind: 'Deployment', name: 'adaptive-studio-frontend' },
    { filename: 'training-job.yaml', kind: 'Job', name: 'adaptive-studio-cpu-training-job' },
    { filename: 'gpu-training-job.yaml', kind: 'Job', name: 'adaptive-studio-gpu-training-job' },
  ],
  errors: [],
};

const mockQuantCaps: QuantizationCapability[] = [
  {
    method: 'bnb_4bit',
    name: 'BitsAndBytes 4-Bit (NF4 / FP4)',
    is_available: false,
    required_libraries: ['bitsandbytes', 'torch'],
    installed_libraries: [],
    missing_libraries: ['bitsandbytes'],
    requires_cuda: true,
    cuda_available: false,
    supports_training: true,
    supports_post_training: true,
    supports_inference: true,
    supports_export: false,
    reason_if_unavailable: 'Requires CUDA-compatible GPU hardware',
  },
  {
    method: 'gguf',
    name: 'GGUF Format (llama.cpp compatible)',
    is_available: true,
    required_libraries: ['torch'],
    installed_libraries: ['torch', 'numpy'],
    missing_libraries: [],
    requires_cuda: false,
    cuda_available: false,
    supports_training: false,
    supports_post_training: true,
    supports_inference: true,
    supports_export: true,
    reason_if_unavailable: null,
  },
];

const mockGpuReport: MultiGPUSystemReport = {
  gpu_count: 0,
  cuda_available: false,
  devices: [],
  ddp_supported: false,
  fsdp_supported: false,
  recommended_strategy: 'none',
  summary: 'CPU-only execution environment. 0 GPUs detected.',
};

describe('Phase 10: PlatformOperationsView Component Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.validateKubernetesManifests).mockResolvedValue(mockK8sValidation);
    vi.mocked(api.fetchQuantizationCapabilities).mockResolvedValue(mockQuantCaps);
    vi.mocked(api.fetchQuantizationJobs).mockResolvedValue([]);
    vi.mocked(api.fetchGGUFExportJobs).mockResolvedValue([]);
    vi.mocked(api.fetchServingCapabilities).mockResolvedValue({
      is_available: false,
      version: null,
      cuda_available: false,
      supported_model_types: ['llama', 'mistral', 'qwen'],
      reason_if_unavailable: 'Requires CUDA GPU hardware',
    });
    vi.mocked(api.fetchServingInstances).mockResolvedValue([]);
    vi.mocked(api.fetchMultiGPUSystemReport).mockResolvedValue(mockGpuReport);
  });

  it('renders platform overview and defaults to Kubernetes manifest tab', async () => {
    render(<PlatformOperationsView />);

    expect(screen.getByText('Platform & Infrastructure')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('12/12 Valid YAML')).toBeInTheDocument();
      expect(screen.getByText('backend-deployment.yaml')).toBeInTheDocument();
      expect(screen.getByText('frontend-deployment.yaml')).toBeInTheDocument();
    });
  });

  it('navigates to Advanced Quantization tab and displays truthful capabilities', async () => {
    render(<PlatformOperationsView />);

    const quantTabBtn = screen.getByTestId('subtab-quantization');
    fireEvent.click(quantTabBtn);

    await waitFor(() => {
      expect(screen.getByText('BitsAndBytes 4-Bit (NF4 / FP4)')).toBeInTheDocument();
      expect(screen.getByText('GGUF Format (llama.cpp compatible)')).toBeInTheDocument();
      expect(screen.getByText('Requires CUDA-compatible GPU hardware')).toBeInTheDocument();
    });
  });

  it('navigates to GGUF Model Export tab and shows empty state', async () => {
    render(<PlatformOperationsView />);

    const exportTabBtn = screen.getByTestId('subtab-export');
    fireEvent.click(exportTabBtn);

    await waitFor(() => {
      expect(screen.getByText('GGUF Model Container Pipeline (GGUF v3)')).toBeInTheDocument();
      expect(screen.getByText(/No GGUF export jobs recorded/i)).toBeInTheDocument();
    });
  });

  it('navigates to Inference Serving tab and shows vLLM hardware prerequisites', async () => {
    render(<PlatformOperationsView />);

    const servingTabBtn = screen.getByTestId('subtab-serving');
    fireEvent.click(servingTabBtn);

    await waitFor(() => {
      expect(screen.getByText('vLLM Inference Engine')).toBeInTheDocument();
      expect(screen.getByText('Unavailable (CUDA required)')).toBeInTheDocument();
    });
  });

  it('navigates to Multi-GPU Diagnostics tab and truthfully flags CPU mode', async () => {
    render(<PlatformOperationsView />);

    await waitFor(() => {
      expect(screen.getByText('12/12 Valid YAML')).toBeInTheDocument();
    });

    const gpuTabBtn = screen.getByTestId('subtab-multigpu');
    fireEvent.click(gpuTabBtn);

    await waitFor(() => {
      expect(screen.getByText(/PyTorch Distributed Hardware Discovery/i)).toBeInTheDocument();
      expect(screen.getByText(/CPU-only/i)).toBeInTheDocument();
      expect(screen.getByText(/False \(CPU Only\)/i)).toBeInTheDocument();
      expect(screen.getByText(/Host system is operating truthfully in CPU mode/i)).toBeInTheDocument();
    });
  });

  it('refreshes all platform diagnostics when refresh button is clicked', async () => {
    render(<PlatformOperationsView />);

    await waitFor(() => {
      expect(screen.getByText('12/12 Valid YAML')).toBeInTheDocument();
    });

    const refreshBtn = screen.getByTestId('refresh-platform-btn');
    expect(refreshBtn).not.toBeDisabled();
    fireEvent.click(refreshBtn);

    await waitFor(() => {
      expect(api.validateKubernetesManifests).toHaveBeenCalledTimes(2);
      expect(api.fetchQuantizationCapabilities).toHaveBeenCalledTimes(2);
      expect(api.fetchMultiGPUSystemReport).toHaveBeenCalledTimes(2);
    });
  });
});
