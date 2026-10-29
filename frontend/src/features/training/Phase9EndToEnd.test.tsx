import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import App from '../../App';
import * as api from '../../services/api';

vi.mock('../../services/api', () => ({
  fetchHealth: vi.fn(),
  fetchSystemStatus: vi.fn(),
  fetchDefaultConfig: vi.fn(),
  validateConfig: vi.fn(),
  fetchModelCatalog: vi.fn(),
  fetchDatasets: vi.fn(),
  runPreflightCheck: vi.fn(),
  generateRunPlan: vi.fn(),
  createTrainingRun: vi.fn(),
  fetchTrainingRun: vi.fn(),
  fetchTrainingEvents: vi.fn(),
  fetchTrainingLogs: vi.fn(),
  fetchRunCheckpoints: vi.fn(),
  fetchRunArtifacts: vi.fn(),
  fetchAllArtifacts: vi.fn(),
  fetchRunHistory: vi.fn(),
  fetchEvaluations: vi.fn(),
  fetchEvaluationDetails: vi.fn(),
  createEvaluation: vi.fn(),
  startEvaluation: vi.fn(),
  validateArtifact: vi.fn(),
  exportArtifact: vi.fn(),
}));

describe('Phase 9 End-to-End Frontend Integration Verification', () => {
  const honestCpuStatus = {
    python_version: '3.12.3',
    os_platform: 'Linux-x86_64',
    pytorch_available: true,
    pytorch_version: '2.13.0+cu130',
    cuda_build_version: '13.0',
    cuda_available: false,
    cuda_version: null,
    gpu_available: false,
    device_type: 'cpu',
    gpu_name: null,
    gpu_count: 0,
    gpu_devices: [],
    primary_gpu_name: null,
    primary_gpu_vram_gb: null,
    total_vram_bytes: 0,
    free_vram_bytes: 0,
    detection_status: 'cuda_unavailable',
    environment_type: 'cpu_only',
  };

  const mockDefaultConfig = {
    project_name: 'e2e-studio-project',
    model_identifier: 'google/gemma-2b',
    training_method: 'lora' as const,
    dataset_reference: 'dataset_123',
    dataset_subset: null,
    train_split: 'train',
    eval_split: 'none',
    training_length_type: 'steps' as const,
    training_length: 50,
    batch_size: 2,
    context_length: 1024,
    learning_rate: 0.0002,
    advanced_enabled: false,
    hf_token_status: 'not_detected',
  };

  beforeEach(() => {
    vi.clearAllMocks();

    vi.mocked(api.fetchHealth).mockResolvedValue({
      status: 'ok',
      app_name: 'Adaptive LLM Fine-Tuning Studio',
      version: '0.9.0-phase9',
    });

    vi.mocked(api.fetchSystemStatus).mockResolvedValue(honestCpuStatus);
    vi.mocked(api.fetchDefaultConfig).mockResolvedValue(mockDefaultConfig);
    vi.mocked(api.fetchModelCatalog).mockResolvedValue({ models: [], total: 0 });
    vi.mocked(api.fetchDatasets).mockResolvedValue({ datasets: [], total: 0 });

    vi.mocked(api.validateConfig).mockResolvedValue({
      is_valid: true,
      errors: [],
      warnings: [],
      is_ready_for_training: true,
      config: mockDefaultConfig,
    });

    vi.mocked(api.fetchRunHistory).mockResolvedValue({
      runs: [
        {
          run_id: 'run_test_phase9_001',
          project_name: 'e2e-studio-project',
          model_identifier: 'google/gemma-2b',
          dataset_reference: 'dataset_123',
          training_method: 'lora',
          status: 'completed',
          current_step: 50,
          max_steps: 50,
          current_loss: 1.15,
          elapsed_time_seconds: 120,
          device: 'cpu',
          created_at: '2026-10-29T12:00:00Z',
          completed_at: '2026-10-29T12:05:00Z',
        },
      ],
      total: 1,
      page: 1,
      page_size: 20,
      total_pages: 1,
    });

    vi.mocked(api.fetchAllArtifacts).mockResolvedValue([
      {
        artifact_id: 'art_001',
        run_id: 'run_test_phase9_001',
        artifact_type: 'lora_adapter',
        file_reference: '/workspace/artifacts/runs/run_test_phase9_001/adapter_model.bin',
        fine_tuning_method: 'lora',
        size_bytes: 3500,
        model_identifier: 'google/gemma-2b',
        validation_status: 'valid',
        export_status: 'not_exported',
        created_at: '2026-10-29T12:05:00Z',
      },
    ]);

    vi.mocked(api.fetchEvaluations).mockResolvedValue({
      evaluations: [
        {
          evaluation_id: 'eval_phase9_001',
          run_id: 'run_test_phase9_001',
          artifact_id: 'art_001',
          dataset_id: 'ds_test',
          status: 'completed',
          config: {
            artifact_id: 'art_001',
            dataset_id: 'ds_test',
          },
          created_at: '2026-10-29T12:06:00Z',
          elapsed_seconds: 3.5,
          samples: [],
          metrics: {
            eval_loss: 1.25,
            perplexity: 3.49,
            total_tokens: 1200,
            eval_samples_count: 50,
            eval_runtime_seconds: 3.3,
            samples_per_second: 15.2,
            tokens_per_second: 363.6,
            unavailable_reasons: {},
          },
        },
      ],
      total: 1,
      page: 1,
      page_size: 10,
    });
  });

  it('1. verifies honest hardware reporting in sidebar and status badges', async () => {
    await act(async () => {
      render(<App />);
    });

    await waitFor(() => {
      expect(screen.getByTestId('backend-status-badge')).toHaveTextContent('Connected');
    });

    // Device indicator should honestly report CPU without fabricated GPU metrics
    expect(screen.getByTestId('sidebar-device-status')).toHaveTextContent(/cpu/i);
    expect(screen.queryByText('NVIDIA RTX 2050')).not.toBeInTheDocument();
  });

  it('2. verifies complete navigation across studio tabs (Configure -> History -> Artifacts -> Evaluation)', async () => {
    await act(async () => {
      render(<App />);
    });

    // 1. History Tab
    const historyNav = screen.getByTestId('nav-history');
    await act(async () => {
      fireEvent.click(historyNav);
    });

    await waitFor(() => {
      expect(screen.getByText('Training Run History')).toBeInTheDocument();
      expect(screen.getByText('e2e-studio-project')).toBeInTheDocument();
    });

    // 2. Artifacts Tab
    const artifactsNav = screen.getByTestId('nav-models');
    await act(async () => {
      fireEvent.click(artifactsNav);
    });

    await waitFor(() => {
      expect(screen.getByText('Training Artifacts & Adapters')).toBeInTheDocument();
    });

    // 3. Evaluation Tab
    const headerTabs = screen.getAllByRole('button');
    const evalTab = headerTabs.find((b) => b.textContent?.includes('Evaluation'));
    if (evalTab) {
      await act(async () => {
        fireEvent.click(evalTab);
      });
      await waitFor(() => {
        expect(screen.getByText('Model Evaluation')).toBeInTheDocument();
      });
    }

    // 4. Return to Configure
    const newTrainingNav = screen.getByTestId('nav-new-training');
    await act(async () => {
      fireEvent.click(newTrainingNav);
    });

    await waitFor(() => {
      expect(screen.getByTestId('sidebar-device-status')).toBeInTheDocument();
    });
  });

  it('3. verifies disconnected backend gracefully notifies the user without crashing', async () => {
    vi.mocked(api.fetchHealth).mockRejectedValueOnce(new Error('Connection refused'));
    vi.mocked(api.fetchSystemStatus).mockRejectedValueOnce(new Error('Connection refused'));

    await act(async () => {
      render(<App />);
    });

    await waitFor(() => {
      expect(screen.getByTestId('backend-status-badge')).toHaveTextContent('Offline');
    });
  });
});
