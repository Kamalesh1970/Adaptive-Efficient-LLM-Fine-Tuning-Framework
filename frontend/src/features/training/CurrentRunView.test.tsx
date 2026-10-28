import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { CurrentRunView } from './CurrentRunView';
import * as api from '../../services/api';
import {
  TrainingRunResponse,
  TrainingArtifactMetadata,
  TrainingEventListResponse,
  TrainingLogListResponse,
  CheckpointListResponse,
} from '../../types/training';

vi.mock('../../services/api');

const mockBaseConfig = {
  project_name: 'test-project',
  model_identifier: 'test/tiny-llama',
  training_method: 'lora' as const,
  dataset_reference: 'ds_test_123',
  dataset_subset: null,
  train_split: 'train',
  eval_split: 'none',
  training_length_type: 'steps' as const,
  training_length: 50,
  max_steps: 50,
  num_epochs: 3.0,
  learning_rate: 0.0002,
  batch_size: 2,
  context_length: 1024,
  lora_rank: 16,
  lora_alpha: 32,
  lora_dropout: 0.05,
  advanced_enabled: false,
  hf_token_status: 'not_detected',
};

const mockRunningResponse: TrainingRunResponse = {
  run_id: 'run_abc123456789',
  status: 'training',
  config: mockBaseConfig,
  created_at: '2026-10-27T12:00:00Z',
  started_at: '2026-10-27T12:00:05Z',
  completed_at: null,
  current_step: 25,
  max_steps: 50,
  current_epoch: 1.5,
  num_epochs: 3.0,
  current_loss: 1.8452,
  loss_history: [
    { step: 10, loss: 2.1245, learning_rate: 0.0002, epoch: 0.6, timestamp: '2026-10-27T12:00:10Z' },
    { step: 20, loss: 1.9540, learning_rate: 0.0002, epoch: 1.2, timestamp: '2026-10-27T12:00:20Z' },
    { step: 25, loss: 1.8452, learning_rate: 0.0002, epoch: 1.5, timestamp: '2026-10-27T12:00:25Z' },
  ],
  elapsed_time_seconds: 25.4,
  error_message: null,
  device: 'cuda:0',
  trainable_params: 4194304,
  total_params: 8030000000,
  trainable_percent: 0.0522,
  cancellable: true,
};

const mockCompletedResponse: TrainingRunResponse = {
  ...mockRunningResponse,
  status: 'completed',
  current_step: 50,
  completed_at: '2026-10-27T12:00:50Z',
  current_loss: 1.2401,
  cancellable: false,
};

const mockArtifacts: TrainingArtifactMetadata = {
  run_id: 'run_abc123456789',
  artifact_directory: 'artifacts/runs/run_abc123456789',
  files: ['adapter_config.json', 'adapter_model.safetensors', 'metrics.json', 'run_metadata.json'],
  adapter_config: { peft_type: 'LORA', r: 16, lora_alpha: 32 },
  trainable_params: 4194304,
  total_params: 8030000000,
  trainable_percent: 0.0522,
  is_valid: true,
  verification_status: 'verified',
  created_at: '2026-10-27T12:00:50Z',
};

const mockEventsResponse: TrainingEventListResponse = {
  run_id: 'run_abc123456789',
  events: [
    {
      event_id: 'evt_001',
      run_id: 'run_abc123456789',
      timestamp: '2026-10-27T12:00:05Z',
      event_type: 'run_started',
      severity: 'info',
      message: 'Training execution started on device cuda:0',
      global_step: 0,
      epoch: 0.0,
      training_loss: null,
      learning_rate: null,
      checkpoint_path: null,
    },
    {
      event_id: 'evt_002',
      run_id: 'run_abc123456789',
      timestamp: '2026-10-27T12:00:25Z',
      event_type: 'training_step',
      severity: 'info',
      message: 'Step 25/50 — loss: 1.8452, epoch: 1.50',
      global_step: 25,
      epoch: 1.5,
      training_loss: 1.8452,
      learning_rate: 0.0002,
      checkpoint_path: null,
    },
  ],
  total: 2,
  has_more: false,
  next_cursor: null,
};

const mockLogsResponse: TrainingLogListResponse = {
  run_id: 'run_abc123456789',
  logs: [
    {
      timestamp: '2026-10-27T12:00:05Z',
      level: 'INFO',
      message: 'Allocating model tensors on device cuda:0',
      stage: 'loading_model',
    },
    {
      timestamp: '2026-10-27T12:00:25Z',
      level: 'INFO',
      message: 'Optimizer step 25 completed smoothly',
      stage: 'training',
    },
  ],
  total: 2,
};

const mockCheckpointsResponse: CheckpointListResponse = {
  run_id: 'run_abc123456789',
  checkpoints: [
    {
      checkpoint_id: 'final_adapter',
      run_id: 'run_abc123456789',
      step: 50,
      epoch: 3.0,
      path: 'artifacts/runs/run_abc123456789',
      created_at: '2026-10-27T12:00:50Z',
      size_bytes: 16777216,
      files: ['adapter_config.json', 'adapter_model.safetensors'],
      is_valid: true,
    },
  ],
  total: 1,
};

describe('Phase 7: CurrentRunView Component Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.fetchRunEvents).mockResolvedValue(mockEventsResponse);
    vi.mocked(api.fetchRunLogs).mockResolvedValue(mockLogsResponse);
    vi.mocked(api.fetchRunCheckpoints).mockResolvedValue(mockCheckpointsResponse);
  });

  it('renders placeholder when no run is active and navigates to configuration', () => {
    const onNavigate = vi.fn();
    render(<CurrentRunView activeRunId={null} onNavigateToConfigure={onNavigate} />);

    expect(screen.getByTestId('current-run-placeholder')).toBeInTheDocument();
    expect(screen.getByText('No Active Training Run')).toBeInTheDocument();

    const goBtn = screen.getByTestId('btn-go-to-configure');
    fireEvent.click(goBtn);
    expect(onNavigate).toHaveBeenCalled();
  });

  it('fetches and displays running job telemetry, progress bar, real loss, step counts, and speed/ETA', async () => {
    vi.mocked(api.fetchTrainingRun).mockResolvedValue(mockRunningResponse);

    await act(async () => {
      render(<CurrentRunView activeRunId="run_abc123456789" />);
    });

    await waitFor(() => {
      expect(screen.getByTestId('run-id-display')).toHaveTextContent('run_abc123456789');
    });

    expect(screen.getByText('Training In Progress')).toBeInTheDocument();
    expect(screen.getByTestId('step-count-display')).toHaveTextContent('25 / 50');
    expect(screen.getByTestId('current-loss-display')).toHaveTextContent('1.8452');
    expect(screen.getByTestId('elapsed-time-display')).toHaveTextContent('25.4s');
    expect(screen.getByTestId('device-display')).toHaveTextContent('cuda:0');
    expect(screen.getByTestId('trainable-params-count')).toHaveTextContent('4,194,304');
    expect(screen.getByTestId('trainable-percent')).toHaveTextContent('(0.052%)');

    // Speed & ETA check
    expect(screen.getByTestId('speed-display')).toHaveTextContent('0.98 steps/s');
    expect(screen.getByTestId('eta-display')).toHaveTextContent('ETA: 25s');

    // Telemetry stream table
    const table = screen.getByTestId('loss-history-table');
    expect(table).toBeInTheDocument();
    expect(screen.getByText('2.1245')).toBeInTheDocument();
    expect(screen.getByText('1.9540')).toBeInTheDocument();

    // Progress bar fill
    expect(screen.getByTestId('progress-bar-fill')).toHaveStyle({ width: '50%' });
  });

  it('switches between monitoring tabs (Events, Worker Logs, Checkpoints)', async () => {
    vi.mocked(api.fetchTrainingRun).mockResolvedValue(mockRunningResponse);

    await act(async () => {
      render(<CurrentRunView activeRunId="run_abc123456789" />);
    });

    await waitFor(() => {
      expect(screen.getByTestId('tab-events')).toBeInTheDocument();
    });

    // 1. Switch to Events tab
    await act(async () => {
      fireEvent.click(screen.getByTestId('tab-events'));
    });
    expect(screen.getByTestId('events-panel')).toBeInTheDocument();
    expect(screen.getByText('Step 25/50 — loss: 1.8452, epoch: 1.50')).toBeInTheDocument();

    // 2. Switch to Logs tab
    await act(async () => {
      fireEvent.click(screen.getByTestId('tab-logs'));
    });
    expect(screen.getByTestId('logs-panel')).toBeInTheDocument();
    expect(screen.getByText(/Allocating model tensors on device cuda:0/)).toBeInTheDocument();

    // 3. Switch to Checkpoints tab
    await act(async () => {
      fireEvent.click(screen.getByTestId('tab-checkpoints'));
    });
    expect(screen.getByTestId('checkpoints-panel')).toBeInTheDocument();
    expect(screen.getByText('final_adapter')).toBeInTheDocument();
  });

  it('handles cancellation click and updates status without corrupting view', async () => {
    vi.mocked(api.fetchTrainingRun).mockResolvedValue(mockRunningResponse);
    vi.mocked(api.cancelTrainingRun).mockResolvedValue({
      ...mockRunningResponse,
      status: 'cancelled',
      cancellable: false,
    });

    await act(async () => {
      render(<CurrentRunView activeRunId="run_abc123456789" />);
    });

    await waitFor(() => {
      expect(screen.getByTestId('btn-cancel-run')).toBeInTheDocument();
    });

    const cancelBtn = screen.getByTestId('btn-cancel-run');
    await act(async () => {
      fireEvent.click(cancelBtn);
    });

    expect(api.cancelTrainingRun).toHaveBeenCalledWith('run_abc123456789');
  });

  it('renders verified artifacts panel with relative path and file list upon completion', async () => {
    vi.mocked(api.fetchTrainingRun).mockResolvedValue(mockCompletedResponse);
    vi.mocked(api.fetchRunArtifacts).mockResolvedValue(mockArtifacts);

    await act(async () => {
      render(<CurrentRunView activeRunId="run_abc123456789" />);
    });

    await waitFor(() => {
      expect(screen.getByTestId('artifacts-panel')).toBeInTheDocument();
    });

    expect(screen.getByText('LoRA Adapters Saved and Verified Successfully')).toBeInTheDocument();
    expect(screen.getByTestId('artifact-directory-path')).toHaveTextContent('artifacts/runs/run_abc123456789');
    expect(screen.getByTestId('artifact-files-list')).toHaveTextContent('adapter_model.safetensors');
    expect(screen.getByTestId('artifact-files-list')).toHaveTextContent('adapter_config.json');
  });

  it('displays actionable error banner when run status is failed', async () => {
    const mockFailedResponse: TrainingRunResponse = {
      ...mockRunningResponse,
      status: 'failed',
      cancellable: false,
      error_message: 'CUDA out of memory: tried to allocate 2.4 GiB on device cuda:0',
    };
    vi.mocked(api.fetchTrainingRun).mockResolvedValue(mockFailedResponse);

    await act(async () => {
      render(<CurrentRunView activeRunId="run_abc123456789" />);
    });

    await waitFor(() => {
      expect(screen.getByTestId('run-failure-banner')).toBeInTheDocument();
    });

    expect(screen.getByText(/CUDA out of memory/)).toBeInTheDocument();
  });
});
