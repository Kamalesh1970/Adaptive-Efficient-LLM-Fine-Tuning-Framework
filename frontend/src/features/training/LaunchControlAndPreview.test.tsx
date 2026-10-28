import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { RunPreviewPanel } from './RunPreviewPanel';
import { PreflightModal } from './PreflightModal';
import * as api from '../../services/api';
import {
  TrainingConfig,
  SystemStatus,
  ValidationResult,
  PreflightCheckResponse,
} from '../../types/training';

vi.mock('../../services/api', () => ({
  fetchHealth: vi.fn(),
  fetchSystemStatus: vi.fn(),
  fetchDefaultConfig: vi.fn(),
  validateConfig: vi.fn(),
  fetchModelCatalog: vi.fn(),
  validateModelSelection: vi.fn(),
  fetchDatasets: vi.fn(),
  uploadDatasetFile: vi.fn(),
  fetchDatasetReport: vi.fn(),
  fetchDatasetSplits: vi.fn(),
  resetConfig: vi.fn(),
  launchTrainingRun: vi.fn(),
  runPreflight: vi.fn(),
  fetchRunPreview: vi.fn(),
  validateRun: vi.fn(),
}));

const sampleConfig: TrainingConfig = {
  project_name: 'preview-test-project',
  model_identifier: 'meta-llama/Meta-Llama-3-8B',
  training_method: 'lora',
  dataset_reference: 'sample_alpaca_dataset',
  dataset_subset: null,
  train_split: 'train',
  eval_split: 'none',
  training_length_type: 'steps',
  training_duration_mode: 'steps',
  training_length: 250,
  max_steps: 250,
  num_epochs: 3.0,
  learning_rate: 0.0003,
  batch_size: 4,
  per_device_train_batch_size: 4,
  gradient_accumulation_steps: 2,
  weight_decay: 0.01,
  warmup_ratio: 0.03,
  optimizer: 'adamw_8bit',
  seed: 42,
  context_length: 4096,
  max_seq_length: 4096,
  truncation: true,
  packing: false,
  lora_rank: 32,
  lora_alpha: 64,
  lora_dropout: 0.05,
  lora_target_modules: ['q_proj', 'v_proj'],
  gradient_checkpointing: true,
  mixed_precision: 'fp16',
  quantization_type: 'nf4',
  advanced_enabled: false,
  hf_token_status: 'not_detected',
};

const sampleGpuStatus: SystemStatus = {
  python_version: '3.12.3',
  os_platform: 'Linux-x86_64',
  pytorch_available: true,
  pytorch_version: '2.3.0',
  cuda_build_version: '12.1',
  cuda_available: true,
  cuda_version: '12.1',
  gpu_available: true,
  device_type: 'cuda',
  gpu_name: 'NVIDIA GeForce RTX 2050',
  gpu_count: 1,
  gpu_devices: [
    {
      index: 0,
      name: 'NVIDIA GeForce RTX 2050',
      total_vram_gb: 4.0,
      total_vram_mb: 4096.0,
      total_vram_bytes: 4294967296,
      free_vram_bytes: 3500000000,
    },
  ],
  primary_gpu_name: 'NVIDIA GeForce RTX 2050',
  primary_gpu_vram_gb: 4.0,
  total_vram_bytes: 4294967296,
  free_vram_bytes: 3500000000,
  detection_status: 'gpu_detected',
  environment_type: 'cuda_accelerated',
};

const validResult: ValidationResult = {
  is_valid: true,
  errors: [],
  warnings: [],
  is_ready_for_training: true,
  config: sampleConfig,
};

const invalidResult: ValidationResult = {
  is_valid: false,
  errors: ['learning_rate must be > 0.0', 'batch_size cannot exceed 128'],
  field_errors: {
    learning_rate: ['learning_rate must be > 0.0'],
    batch_size: ['batch_size cannot exceed 128'],
  },
  warnings: [],
  is_ready_for_training: false,
  config: sampleConfig,
};

const passingPreflightResponse: PreflightCheckResponse = {
  checks: [
    {
      check_id: 'config_schema',
      label: 'Configuration Schema',
      status: 'pass',
      message: 'Training configuration is valid.',
      detail: null,
    },
    {
      check_id: 'model_selected',
      label: 'Base Model',
      status: 'pass',
      message: "Model 'meta-llama/Meta-Llama-3-8B' is selected.",
      detail: null,
    },
    {
      check_id: 'dataset_selected',
      label: 'Training Dataset',
      status: 'pass',
      message: "Dataset 'sample_alpaca_dataset' is selected.",
      detail: null,
    },
    {
      check_id: 'device',
      label: 'Compute Device',
      status: 'pass',
      message: 'CUDA GPU detected: NVIDIA GeForce RTX 2050.',
      detail: null,
    },
    {
      check_id: 'method_device_compat',
      label: 'Method–Device Compatibility',
      status: 'pass',
      message: 'LORA is compatible with CUDA device.',
      detail: null,
    },
    {
      check_id: 'ml_core_deps',
      label: 'Core ML Stack',
      status: 'pass',
      message: 'PyTorch 2.3.0, transformers, peft verified.',
      detail: null,
    },
    {
      check_id: 'bitsandbytes',
      label: 'bitsandbytes (4-bit)',
      status: 'skip',
      message: 'Skipped for LoRA.',
      detail: null,
    },
    {
      check_id: 'vram',
      label: 'VRAM Capacity',
      status: 'pass',
      message: 'Estimated ~2.1 GB fits within 4.0 GB.',
      detail: null,
    },
  ],
  can_launch: true,
  blocking_count: 0,
  warning_count: 0,
  method: 'lora',
  estimated_vram_gb: 2.1,
  vram_status: 'supported',
};

describe('Phase 6: Frontend Training Run Preview, Resource Validation & Launch Control', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // 1. Preview updates
  it('1. updates run preview with model, dataset, method, duration, and hyperparameter data', () => {
    render(
      <RunPreviewPanel
        config={sampleConfig}
        systemStatus={sampleGpuStatus}
        backendOnline={true}
        validationResult={validResult}
      />
    );

    expect(screen.getByTestId('preview-model-identifier')).toHaveTextContent('meta-llama/Meta-Llama-3-8B');
    expect(screen.getByTestId('preview-dataset-reference')).toHaveTextContent('sample_alpaca_dataset');
    expect(screen.getByTestId('preview-method')).toHaveTextContent('LoRA');
    expect(screen.getByTestId('preview-duration')).toHaveTextContent('250 steps');
    expect(screen.getByTestId('preview-context-length')).toHaveTextContent('4096 tokens');
    expect(screen.getByTestId('preview-learning-rate')).toHaveTextContent('0.0003');
    // Effective batch size: 4 per device * 2 accum * 1 GPU = 8
    expect(screen.getByTestId('preview-effective-batch')).toHaveTextContent('8');
  });

  // 2. Validation loading state
  it('2. displays validation in progress loading state when validating', () => {
    render(
      <RunPreviewPanel
        config={sampleConfig}
        systemStatus={sampleGpuStatus}
        backendOnline={true}
        validationResult={validResult}
        isValidating={true}
      />
    );

    expect(screen.getByTestId('launch-control-state')).toHaveTextContent('Validation in progress');
    const startBtn = screen.getByTestId('btn-start-training');
    expect(startBtn).toBeDisabled();
    expect(startBtn).toHaveTextContent('Validation in progress...');
  });

  // 3. Validation errors
  it('3. displays validation errors, Invalid readiness badge, and blocks launch', () => {
    render(
      <RunPreviewPanel
        config={sampleConfig}
        systemStatus={sampleGpuStatus}
        backendOnline={true}
        validationResult={invalidResult}
      />
    );

    expect(screen.getByTestId('validation-status-invalid')).toHaveTextContent('2 Error(s)');
    expect(screen.getByTestId('preview-launch-readiness')).toHaveTextContent('Invalid');
    expect(screen.getByTestId('launch-control-state')).toHaveTextContent('Blocked by validation');
    expect(screen.getByTestId('btn-start-training')).toBeDisabled();
  });

  // 4. Unknown compatibility
  it('4. displays Unknown readiness status when backend is offline or compatibility cannot be verified', () => {
    render(
      <RunPreviewPanel
        config={sampleConfig}
        systemStatus={null}
        backendOnline={false}
        validationResult={null}
      />
    );

    expect(screen.getByTestId('preview-launch-readiness')).toHaveTextContent('Unknown');
    expect(screen.getByTestId('preview-dependency-compat')).toHaveTextContent('Unknown');
    expect(screen.getByTestId('hardware-device-name')).toHaveTextContent('Backend Unavailable');
    expect(screen.getByTestId('btn-start-training')).toBeDisabled();
  });

  // 5. Launch button state
  it('5. enables launch button when ready for confirmation and disables when incomplete', () => {
    const handleStart = vi.fn();
    const { rerender } = render(
      <RunPreviewPanel
        config={sampleConfig}
        systemStatus={sampleGpuStatus}
        backendOnline={true}
        validationResult={validResult}
        onStartTraining={handleStart}
      />
    );

    // Ready for confirmation state
    expect(screen.getByTestId('launch-control-state')).toHaveTextContent('Ready for confirmation');
    const startBtn = screen.getByTestId('btn-start-training');
    expect(startBtn).not.toBeDisabled();
    fireEvent.click(startBtn);
    expect(handleStart).toHaveBeenCalledOnce();

    // Configuration incomplete state (no model)
    rerender(
      <RunPreviewPanel
        config={{ ...sampleConfig, model_identifier: null }}
        systemStatus={sampleGpuStatus}
        backendOnline={true}
        validationResult={validResult}
      />
    );
    expect(screen.getByTestId('launch-control-state')).toHaveTextContent('Configuration incomplete');
    expect(screen.getByTestId('btn-start-training')).toBeDisabled();
  });

  // 6. Explicit confirmation
  it('6. requires explicit user confirmation before enabling Confirm & Launch in PreflightModal', () => {
    const onConfirm = vi.fn();
    render(
      <PreflightModal
        isOpen={true}
        isLoading={false}
        result={passingPreflightResponse}
        error={null}
        onConfirmLaunch={onConfirm}
        onClose={vi.fn()}
        onRerun={vi.fn()}
      />
    );

    const confirmBtn = screen.getByTestId('btn-confirm-launch');
    const confirmCheckbox = screen.getByTestId('checkbox-explicit-confirm');

    // Initially confirmed
    expect(confirmCheckbox).toBeChecked();
    expect(confirmBtn).not.toBeDisabled();

    // Unchecking explicit confirmation disables launch button
    fireEvent.click(confirmCheckbox);
    expect(confirmCheckbox).not.toBeChecked();
    expect(confirmBtn).toBeDisabled();

    // Checking it again re-enables launch button
    fireEvent.click(confirmCheckbox);
    expect(confirmCheckbox).toBeChecked();
    expect(confirmBtn).not.toBeDisabled();

    fireEvent.click(confirmBtn);
    expect(onConfirm).toHaveBeenCalledOnce();
  });

  // 7. Duplicate submission prevention
  it('7. prevents duplicate submissions during active launch', async () => {
    vi.mocked(api.launchTrainingRun).mockReturnValue(new Promise(() => {}) as any);

    render(
      <RunPreviewPanel
        config={sampleConfig}
        systemStatus={sampleGpuStatus}
        backendOnline={true}
        validationResult={validResult}
        isLaunching={true}
      />
    );

    const startBtn = screen.getByTestId('btn-start-training');
    expect(startBtn).toBeDisabled();
    expect(screen.getByTestId('launch-control-state')).toHaveTextContent('Launching');
    expect(startBtn).toHaveTextContent('Launching Fine-Tuning...');
  });

  // 8. Launch failure
  it('8. renders launch failure state and error banner when launch API fails', () => {
    render(
      <RunPreviewPanel
        config={sampleConfig}
        systemStatus={sampleGpuStatus}
        backendOnline={true}
        validationResult={validResult}
        launchError="Preflight rejected: bitsandbytes missing for QLoRA"
      />
    );

    expect(screen.getByTestId('launch-control-state')).toHaveTextContent('Launch failed');
    expect(screen.getByTestId('launch-error-banner')).toHaveTextContent(
      'Preflight rejected: bitsandbytes missing for QLoRA'
    );
  });

  // 9. Actual run ID display
  it('9. displays actual run ID and Run Created state upon successful launch', () => {
    render(
      <RunPreviewPanel
        config={sampleConfig}
        systemStatus={sampleGpuStatus}
        backendOnline={true}
        validationResult={validResult}
        createdRunId="run_actual_lora_998811"
      />
    );

    expect(screen.getByTestId('launch-control-state')).toHaveTextContent('Run created');
    expect(screen.getByTestId('created-run-banner')).toBeInTheDocument();
    expect(screen.getByTestId('created-run-id')).toHaveTextContent('run_actual_lora_998811');
  });

  // 10. Backend unavailable state
  it('10. handles backend unavailable state cleanly without fabricating results', () => {
    render(
      <RunPreviewPanel
        config={sampleConfig}
        systemStatus={null}
        backendOnline={false}
        validationResult={null}
      />
    );

    expect(screen.getByTestId('hardware-device-name')).toHaveTextContent('Backend Unavailable');
    expect(screen.getByTestId('preview-launch-readiness')).toHaveTextContent('Unknown');
    expect(screen.getByTestId('btn-start-training')).toBeDisabled();
    expect(screen.getByText(/Connect to FastAPI backend to inspect compute hardware/)).toBeInTheDocument();
  });
});
