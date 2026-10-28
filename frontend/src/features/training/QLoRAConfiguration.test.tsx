import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { ParametersCard } from './ParametersCard';
import { RunPreviewPanel } from './RunPreviewPanel';
import { CurrentRunView } from './CurrentRunView';
import * as api from '../../services/api';
import {
  TrainingConfig,
  SystemStatus,
  ValidationResult,
  TrainingRunResponse,
  TrainingArtifactMetadata,
} from '../../types/training';

vi.mock('../../services/api');

const baseConfig: TrainingConfig = {
  project_name: 'test-qlora-project',
  model_identifier: 'meta-llama/Meta-Llama-3-8B',
  training_method: 'qlora',
  dataset_reference: 'yahma/alpaca-cleaned',
  dataset_subset: null,
  train_split: 'train',
  eval_split: 'none',
  training_length_type: 'steps',
  training_duration_mode: 'steps',
  training_length: 50,
  max_steps: 50,
  num_epochs: 3.0,
  learning_rate: 0.0002,
  batch_size: 2,
  per_device_train_batch_size: 2,
  gradient_accumulation_steps: 1,
  weight_decay: 0.01,
  warmup_ratio: 0.03,
  optimizer: 'adamw_8bit',
  seed: 42,
  context_length: 2048,
  max_seq_length: 2048,
  truncation: true,
  packing: false,
  lora_rank: 16,
  lora_alpha: 32,
  lora_dropout: 0.05,
  lora_target_modules: ['q_proj', 'v_proj'],
  gradient_checkpointing: true,
  mixed_precision: 'fp16',
  quantization_type: 'nf4',
  load_in_4bit: true,
  load_in_8bit: false,
  bnb_4bit_quant_type: 'nf4',
  bnb_4bit_use_double_quant: true,
  bnb_4bit_compute_dtype: 'fp16',
  advanced_enabled: true,
  hf_token_status: 'not_detected',
};

const cudaSystemStatus: SystemStatus = {
  python_version: '3.12.3',
  os_platform: 'Linux',
  pytorch_available: true,
  pytorch_version: '2.4.0',
  cuda_available: true,
  cuda_version: '12.4',
  gpu_available: true,
  device_type: 'cuda',
  gpu_name: 'NVIDIA RTX 4090',
  gpu_count: 1,
  gpu_devices: [
    { index: 0, name: 'NVIDIA RTX 4090', total_vram_gb: 24.0, total_vram_mb: 24576 },
  ],
  primary_gpu_name: 'NVIDIA RTX 4090',
  primary_gpu_vram_gb: 24.0,
  environment_type: 'bare_metal',
};

const cpuSystemStatus: SystemStatus = {
  python_version: '3.12.3',
  os_platform: 'Linux',
  pytorch_available: true,
  pytorch_version: '2.4.0',
  cuda_available: false,
  cuda_version: null,
  gpu_available: false,
  device_type: 'cpu',
  gpu_name: null,
  gpu_count: 0,
  gpu_devices: [],
  primary_gpu_name: null,
  primary_gpu_vram_gb: null,
  environment_type: 'cpu_only',
};

const validResult: ValidationResult = {
  is_valid: true,
  errors: [],
  field_errors: {},
  warnings: [],
  is_ready_for_training: true,
  config: baseConfig,
  effective_batch_size_per_device: 2,
  total_effective_batch_size: 2,
  detected_device_count: 1,
};

describe('Phase 5: QLoRA & 4-Bit Quantization Frontend Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('ParametersCard QLoRA Controls', () => {
    it('renders QLoRA quantization controls when training_method is qlora', () => {
      const handleChange = vi.fn();
      render(
        <ParametersCard
          config={baseConfig}
          onChange={handleChange}
          systemStatus={cudaSystemStatus}
          backendOnline={true}
        />
      );

      expect(screen.getByTestId('qlora-quantization-section')).toBeInTheDocument();
      expect(screen.getByTestId('select-qlora-precision')).toHaveValue('4bit');
      expect(screen.getByTestId('select-quantization')).toHaveValue('nf4');
      expect(screen.getByTestId('select-compute-dtype')).toHaveValue('fp16');
      expect(screen.getByTestId('checkbox-double-quant')).toBeChecked();
      expect(screen.getByTestId('qlora-compatibility-badge')).toHaveTextContent('Supported');
    });

    it('displays honest Unsupported compatibility warning when CUDA is missing', () => {
      const handleChange = vi.fn();
      render(
        <ParametersCard
          config={baseConfig}
          onChange={handleChange}
          systemStatus={cpuSystemStatus}
          backendOnline={true}
        />
      );

      expect(screen.getByTestId('qlora-compatibility-badge')).toHaveTextContent('Unsupported');
      expect(screen.getByTestId('qlora-hardware-warning')).toBeInTheDocument();
      expect(screen.getByTestId('qlora-hardware-warning')).toHaveTextContent(
        'Bitsandbytes 4-bit quantization requires a CUDA GPU'
      );
    });

    it('allows switching between 4-bit and 8-bit precision', () => {
      const handleChange = vi.fn();
      render(
        <ParametersCard
          config={baseConfig}
          onChange={handleChange}
          systemStatus={cudaSystemStatus}
          backendOnline={true}
        />
      );

      fireEvent.change(screen.getByTestId('select-qlora-precision'), {
        target: { value: '8bit' },
      });
      expect(handleChange).toHaveBeenCalledWith({
        load_in_4bit: false,
        load_in_8bit: true,
        quantization_type: 'int8',
      });
    });

    it('allows changing 4-bit quantization format (FP4 / NF4)', () => {
      const handleChange = vi.fn();
      render(
        <ParametersCard
          config={baseConfig}
          onChange={handleChange}
          systemStatus={cudaSystemStatus}
          backendOnline={true}
        />
      );

      fireEvent.change(screen.getByTestId('select-quantization'), {
        target: { value: 'fp4' },
      });
      expect(handleChange).toHaveBeenCalledWith({
        bnb_4bit_quant_type: 'fp4',
        quantization_type: 'fp4',
      });
    });

    it('allows toggling double quantization and changing compute dtype', () => {
      const handleChange = vi.fn();
      render(
        <ParametersCard
          config={baseConfig}
          onChange={handleChange}
          systemStatus={cudaSystemStatus}
          backendOnline={true}
        />
      );

      // Toggle double quant
      fireEvent.click(screen.getByTestId('checkbox-double-quant'));
      expect(handleChange).toHaveBeenCalledWith({ bnb_4bit_use_double_quant: false });

      // Change compute dtype
      fireEvent.change(screen.getByTestId('select-compute-dtype'), {
        target: { value: 'bf16' },
      });
      expect(handleChange).toHaveBeenCalledWith({ bnb_4bit_compute_dtype: 'bf16' });
    });
  });

  describe('RunPreviewPanel QLoRA and Hardware Enforcement', () => {
    it('displays detailed QLoRA quantization configuration in run summary preview', () => {
      render(
        <RunPreviewPanel
          config={baseConfig}
          systemStatus={cudaSystemStatus}
          backendOnline={true}
          validationResult={validResult}
        />
      );

      expect(screen.getByTestId('preview-qlora-details')).toBeInTheDocument();
      expect(screen.getByTestId('preview-qlora-details')).toHaveTextContent('4-Bit NF4');
      expect(screen.getByTestId('preview-qlora-details')).toHaveTextContent('Enabled (Nested)');
      expect(screen.getByTestId('preview-qlora-details')).toHaveTextContent('FP16');
      expect(screen.getByTestId('preview-qlora-compat')).toHaveTextContent('Supported');
    });

    it('disables Start Fine-Tuning button when running on CPU hardware with QLoRA', () => {
      render(
        <RunPreviewPanel
          config={baseConfig}
          systemStatus={cpuSystemStatus}
          backendOnline={true}
          validationResult={validResult}
        />
      );

      const startBtn = screen.getByTestId('btn-start-training');
      expect(startBtn).toBeDisabled();
      expect(screen.getByTestId('preview-qlora-compat')).toHaveTextContent('Unsupported');
      expect(screen.getByTestId('preview-qlora-compat')).toHaveTextContent(
        'QLoRA 4-bit fine-tuning requires CUDA GPU acceleration'
      );
    });

    it('enables Start Fine-Tuning button when running on CUDA GPU with valid configuration', () => {
      const handleStart = vi.fn();
      render(
        <RunPreviewPanel
          config={baseConfig}
          systemStatus={cudaSystemStatus}
          backendOnline={true}
          validationResult={validResult}
          onStartTraining={handleStart}
        />
      );

      const startBtn = screen.getByTestId('btn-start-training');
      expect(startBtn).not.toBeDisabled();
      fireEvent.click(startBtn);
      expect(handleStart).toHaveBeenCalled();
    });
  });

  describe('CurrentRunView QLoRA Telemetry & Artifact Verification', () => {
    it('displays QLoRA quantization telemetry badge and details for active runs', async () => {
      const qloraRun: TrainingRunResponse = {
        run_id: 'run_qlora_test123',
        status: 'training',
        config: baseConfig,
        created_at: '2026-10-27T12:00:00Z',
        started_at: '2026-10-27T12:00:05Z',
        completed_at: null,
        current_step: 10,
        max_steps: 50,
        current_epoch: 0.6,
        num_epochs: 3.0,
        current_loss: 2.1542,
        loss_history: [
          { step: 10, loss: 2.1542, learning_rate: 0.0002, epoch: 0.6, timestamp: '2026-10-27T12:00:10Z' },
        ],
        elapsed_time_seconds: 12.5,
        error_message: null,
        device: 'cuda:0',
        trainable_params: 4194304,
        total_params: 8030000000,
        trainable_percent: 0.0522,
        cancellable: true,
        quantization_info: {
          load_in_4bit: true,
          bnb_4bit_quant_type: 'nf4',
          bnb_4bit_use_double_quant: true,
          bnb_4bit_compute_dtype: 'fp16',
        },
      };

      vi.mocked(api.fetchTrainingRun).mockResolvedValue(qloraRun);

      await act(async () => {
        render(<CurrentRunView activeRunId="run_qlora_test123" />);
      });

      await waitFor(() => {
        expect(screen.getByTestId('run-quantization-info')).toBeInTheDocument();
      });

      expect(screen.getByTestId('run-quantization-info')).toHaveTextContent('QLoRA Quantized');
      expect(screen.getByTestId('run-quantization-info')).toHaveTextContent('NF4 (4-Bit)');
      expect(screen.getByTestId('run-quantization-info')).toHaveTextContent('Active');
      expect(screen.getByTestId('run-quantization-info')).toHaveTextContent('FP16');
    });

    it('displays QLoRA specific title and verified artifacts when completed', async () => {
      const completedQloraRun: TrainingRunResponse = {
        run_id: 'run_qlora_done',
        status: 'completed',
        config: baseConfig,
        created_at: '2026-10-27T12:00:00Z',
        started_at: '2026-10-27T12:00:05Z',
        completed_at: '2026-10-27T12:01:00Z',
        current_step: 50,
        max_steps: 50,
        current_loss: 1.25,
        loss_history: [
          { step: 50, loss: 1.25, learning_rate: 0.0002, epoch: 3.0, timestamp: '2026-10-27T12:01:00Z' },
        ],
        elapsed_time_seconds: 55.0,
        device: 'cuda:0',
        trainable_params: 4194304,
        total_params: 8030000000,
        trainable_percent: 0.0522,
        cancellable: false,
        quantization_info: {
          load_in_4bit: true,
          bnb_4bit_quant_type: 'nf4',
        },
      };

      const qloraArtifacts: TrainingArtifactMetadata = {
        run_id: 'run_qlora_done',
        artifact_directory: 'artifacts/runs/run_qlora_done',
        files: [
          'adapter_config.json',
          'adapter_model.safetensors',
          'metrics.json',
          'quantization_config.json',
          'run_metadata.json',
          'training_config.json',
        ],
        adapter_config: { peft_type: 'LORA', r: 16, lora_alpha: 32 },
        quantization_config: { load_in_4bit: true, bnb_4bit_quant_type: 'nf4' },
        trainable_params: 4194304,
        total_params: 8030000000,
        trainable_percent: 0.0522,
        is_valid: true,
        verification_status: 'verified',
        created_at: '2026-10-27T12:01:00Z',
      };

      vi.mocked(api.fetchTrainingRun).mockResolvedValue(completedQloraRun);
      vi.mocked(api.fetchRunArtifacts).mockResolvedValue(qloraArtifacts);

      await act(async () => {
        render(<CurrentRunView activeRunId="run_qlora_done" />);
      });

      await waitFor(() => {
        expect(screen.getByTestId('artifacts-panel')).toBeInTheDocument();
      });

      expect(screen.getByTestId('artifacts-panel')).toHaveTextContent(
        'QLoRA 4-Bit Adapters Saved and Verified Successfully'
      );
      expect(screen.getByTestId('artifact-files-list')).toHaveTextContent('quantization_config.json');
    });
  });
});
