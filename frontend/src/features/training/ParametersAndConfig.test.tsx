import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { ParametersCard } from './ParametersCard';
import { ConfigToolbar } from './ConfigToolbar';
import { RunPreviewPanel } from './RunPreviewPanel';
import { TrainingPage } from './TrainingPage';
import * as api from '../../services/api';
import { TrainingConfig, SystemStatus, ValidationResult } from '../../types/training';

vi.mock('../../services/api', () => ({
  fetchHealth: vi.fn(),
  fetchSystemStatus: vi.fn(),
  fetchDefaultConfig: vi.fn(),
  validateConfig: vi.fn(),
  fetchModelCatalog: vi.fn(),
  fetchModelMetadata: vi.fn(),
  validateModelSelection: vi.fn(),
  fetchDatasets: vi.fn(),
  uploadDatasetFile: vi.fn(),
  fetchDatasetReport: vi.fn(),
  fetchDatasetSplits: vi.fn(),
  exportConfigYaml: vi.fn(),
  parseConfigYaml: vi.fn(),
  resetConfig: vi.fn(),
}));

const mockBaseConfig: TrainingConfig = {
  project_name: 'test-project',
  model_identifier: 'meta-llama/Meta-Llama-3-8B',
  training_method: 'lora',
  dataset_reference: 'ds_test123',
  dataset_subset: null,
  train_split: 'train',
  eval_split: 'none',
  training_length_type: 'steps',
  training_duration_mode: 'steps',
  training_length: 150,
  max_steps: 150,
  num_epochs: 3.0,
  learning_rate: 0.0003,
  batch_size: 4,
  per_device_train_batch_size: 4,
  gradient_accumulation_steps: 2,
  weight_decay: 0.02,
  warmup_ratio: 0.05,
  optimizer: 'adamw_8bit',
  seed: 42,
  context_length: 4096,
  max_seq_length: 4096,
  truncation: true,
  packing: false,
  lora_rank: 32,
  lora_alpha: 64,
  lora_dropout: 0.1,
  lora_target_modules: ['q_proj', 'v_proj'],
  gradient_checkpointing: true,
  mixed_precision: 'bf16',
  quantization_type: 'nf4',
  advanced_enabled: false,
  hf_token_status: 'not_detected',
};

const mockSystemStatus: SystemStatus = {
  python_version: '3.12.0',
  os_platform: 'Linux-6.5.0',
  pytorch_available: true,
  pytorch_version: '2.4.0',
  cuda_available: true,
  cuda_version: '12.1',
  gpu_count: 2,
  gpu_devices: [
    { index: 0, name: 'NVIDIA RTX 4090', total_vram_gb: 24.0, total_vram_mb: 24576 },
    { index: 1, name: 'NVIDIA RTX 4090', total_vram_gb: 24.0, total_vram_mb: 24576 },
  ],
  primary_gpu_name: 'NVIDIA RTX 4090',
  primary_gpu_vram_gb: 24.0,
  environment_type: 'bare_metal',
};

describe('Phase 3: Training Parameters & Configuration Management Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    vi.mocked(api.fetchDefaultConfig).mockResolvedValue(mockBaseConfig);
    vi.mocked(api.resetConfig).mockResolvedValue(mockBaseConfig);
    vi.mocked(api.validateConfig).mockResolvedValue({
      is_valid: true,
      errors: [],
      warnings: [],
      is_ready_for_training: false,
      config: mockBaseConfig,
      effective_batch_size_per_device: 8,
      total_effective_batch_size: 16,
      detected_device_count: 2,
    });
    vi.mocked(api.fetchModelCatalog).mockResolvedValue({ models: [], total: 0 });
    vi.mocked(api.fetchDatasets).mockResolvedValue({ datasets: [], total: 0 });
  });

  it('renders Simple mode with all core fields and controls', () => {
    const handleChange = vi.fn();
    render(<ParametersCard config={mockBaseConfig} onChange={handleChange} />);

    expect(screen.getByTestId('simple-parameters-view')).toBeInTheDocument();
    expect(screen.getByTestId('input-project-name')).toHaveValue('test-project');
    expect(screen.getByTestId('duration-type-steps')).toBeInTheDocument();
    expect(screen.getByTestId('duration-type-epochs')).toBeInTheDocument();
    expect(screen.getByTestId('input-training-length')).toHaveValue(150);
    expect(screen.getByTestId('input-batch-size')).toHaveValue(4);
    expect(screen.getByTestId('input-grad-accum')).toHaveValue(2);
    expect(screen.getByTestId('input-context-length')).toHaveValue(4096);
    expect(screen.getByTestId('input-learning-rate')).toHaveValue(0.0003);
  });

  it('toggles between Simple and Advanced modes and renders advanced controls', async () => {
    const handleChange = vi.fn();
    render(<ParametersCard config={mockBaseConfig} onChange={handleChange} />);

    // Click Advanced tab
    await act(async () => {
      fireEvent.click(screen.getByTestId('param-tab-advanced'));
    });

    expect(handleChange).toHaveBeenCalledWith({ advanced_enabled: true });
    expect(screen.getByTestId('advanced-parameters-view')).toBeInTheDocument();
    expect(screen.getByTestId('input-lora-rank')).toHaveValue(32);
    expect(screen.getByTestId('input-lora-alpha')).toHaveValue(64);
    expect(screen.getByTestId('input-lora-dropout')).toHaveValue(0.1);
    expect(screen.getByTestId('input-target-modules')).toHaveValue('q_proj, v_proj');
    expect(screen.getByTestId('select-optimizer')).toHaveValue('adamw_8bit');
    expect(screen.getByTestId('input-weight-decay')).toHaveValue(0.02);
    expect(screen.getByTestId('input-warmup-ratio')).toHaveValue(0.05);
    expect(screen.getByTestId('input-seed')).toHaveValue(42);
    expect(screen.getByTestId('select-mixed-precision')).toHaveValue('bf16');
    expect(screen.getByTestId('toggle-gradient-checkpointing')).toBeChecked();
    expect(screen.getByTestId('toggle-packing')).not.toBeChecked();
  });

  it('updates shared configuration on user input changes in Simple mode', () => {
    const handleChange = vi.fn();
    render(<ParametersCard config={mockBaseConfig} onChange={handleChange} />);

    // Change project name
    fireEvent.change(screen.getByTestId('input-project-name'), {
      target: { value: 'custom-project-name' },
    });
    expect(handleChange).toHaveBeenCalledWith({ project_name: 'custom-project-name' });

    // Change learning rate
    fireEvent.change(screen.getByTestId('input-learning-rate'), {
      target: { value: '0.0001' },
    });
    expect(handleChange).toHaveBeenCalledWith({ learning_rate: 0.0001 });

    // Change batch size
    fireEvent.change(screen.getByTestId('input-batch-size'), {
      target: { value: '8' },
    });
    expect(handleChange).toHaveBeenCalledWith({
      batch_size: 8,
      per_device_train_batch_size: 8,
    });
  });

  it('switches duration between steps and epochs and updates unit inputs', async () => {
    const handleChange = vi.fn();
    const { rerender } = render(
      <ParametersCard config={mockBaseConfig} onChange={handleChange} />
    );

    // Switch to epochs
    await act(async () => {
      fireEvent.click(screen.getByTestId('duration-type-epochs'));
    });
    expect(handleChange).toHaveBeenCalledWith({
      training_length_type: 'epochs',
      training_duration_mode: 'epochs',
      training_length: 3,
    });

    // Rerender with epochs config
    rerender(
      <ParametersCard
        config={{ ...mockBaseConfig, training_length_type: 'epochs', num_epochs: 5.5 }}
        onChange={handleChange}
      />
    );
    expect(screen.getByTestId('input-training-length')).toHaveValue(5.5);

    // Change epochs number
    fireEvent.change(screen.getByTestId('input-training-length'), {
      target: { value: '4.2' },
    });
    expect(handleChange).toHaveBeenCalledWith({
      num_epochs: 4.2,
      training_length: 4,
    });
  });

  it('handles method-specific controls: disables LoRA in full fine-tuning and shows quantization in QLoRA', () => {
    const handleChange = vi.fn();

    // 1. Full Fine-Tuning mode
    const { rerender } = render(
      <ParametersCard
        config={{ ...mockBaseConfig, training_method: 'full_finetuning', advanced_enabled: true }}
        onChange={handleChange}
      />
    );
    expect(screen.getByTestId('notice-full-finetuning')).toBeInTheDocument();
    expect(screen.queryByTestId('input-lora-rank')).not.toBeInTheDocument();

    // 2. QLoRA mode
    rerender(
      <ParametersCard
        config={{
          ...mockBaseConfig,
          training_method: 'qlora',
          quantization_type: 'nf4',
          advanced_enabled: true,
        }}
        onChange={handleChange}
      />
    );
    expect(screen.getByTestId('qlora-quantization-section')).toBeInTheDocument();
    expect(screen.getByTestId('select-quantization')).toHaveValue('nf4');
    expect(screen.getByTestId('input-lora-rank')).toBeInTheDocument();
  });

  it('displays field-specific validation errors when invalid values are entered', () => {
    const handleChange = vi.fn();
    const fieldErrors = {
      project_name: ['Project name cannot be empty or blank'],
      learning_rate: ['Learning rate must be strictly positive (> 0.0)'],
      context_length: ['Context length must be at least 128 tokens'],
    };

    render(
      <ParametersCard
        config={mockBaseConfig}
        onChange={handleChange}
        fieldErrors={fieldErrors}
      />
    );

    expect(screen.getByTestId('error-project_name')).toHaveTextContent(
      'Project name cannot be empty or blank'
    );
    expect(screen.getByTestId('error-learning_rate')).toHaveTextContent(
      'Learning rate must be strictly positive (> 0.0)'
    );
    expect(screen.getByTestId('error-context_length')).toHaveTextContent(
      'Context length must be at least 128 tokens'
    );
  });

  it('updates RunPreviewPanel from actual selections and calculates effective batch size', () => {
    const validationResult: ValidationResult = {
      is_valid: true,
      errors: [],
      warnings: [],
      is_ready_for_training: false,
      config: mockBaseConfig,
      detected_device_count: 2,
    };

    render(
      <RunPreviewPanel
        config={mockBaseConfig}
        systemStatus={mockSystemStatus}
        backendOnline={true}
        validationResult={validationResult}
      />
    );

    expect(screen.getByTestId('preview-model-identifier')).toHaveTextContent('meta-llama/Meta-Llama-3-8B');
    expect(screen.getByTestId('preview-dataset-reference')).toHaveTextContent('ds_test123');
    expect(screen.getByTestId('preview-method')).toHaveTextContent('LoRA');
    expect(screen.getByTestId('preview-duration')).toHaveTextContent('150 steps');
    // Effective batch size: 4 batch * 2 accum * 2 devices = 16
    expect(screen.getByTestId('preview-effective-batch')).toHaveTextContent('16');
    expect(screen.getByTestId('preview-context-length')).toHaveTextContent('4096 tokens');
    expect(screen.getByTestId('preview-learning-rate')).toHaveTextContent('0.0003');
    expect(screen.getByTestId('validation-status-valid')).toBeInTheDocument();
  });

  it('shows per-device batch and explains total is unknown when device count is unavailable', () => {
    const validationResult: ValidationResult = {
      is_valid: true,
      errors: [],
      warnings: [],
      is_ready_for_training: false,
      config: mockBaseConfig,
      detected_device_count: null,
    };

    render(
      <RunPreviewPanel
        config={mockBaseConfig}
        systemStatus={{ ...mockSystemStatus, cuda_available: false, gpu_count: 0 }}
        backendOnline={true}
        validationResult={validationResult}
      />
    );

    // Per device: 4 batch * 2 accum = 8
    expect(screen.getByTestId('preview-effective-batch')).toHaveTextContent('8 (per-device)');
    expect(screen.getByText(/total unknown — no GPU detected/i)).toBeInTheDocument();
  });

  it('exports configuration to YAML through Save YAML button', async () => {
    vi.mocked(api.exportConfigYaml).mockResolvedValue({
      yaml_content: 'project_name: "test-project"\nlearning_rate: 0.0003\n',
    });

    // Mock createObjectURL & revokeObjectURL
    const createObjectURLMock = vi.fn().mockReturnValue('blob:http://localhost/mock-blob');
    const revokeObjectURLMock = vi.fn();
    window.URL.createObjectURL = createObjectURLMock;
    window.URL.revokeObjectURL = revokeObjectURLMock;

    render(
      <ConfigToolbar
        config={mockBaseConfig}
        onReset={vi.fn()}
        onLoadConfig={vi.fn()}
      />
    );

    await act(async () => {
      fireEvent.click(screen.getByTestId('btn-save-yaml'));
    });

    expect(api.exportConfigYaml).toHaveBeenCalledWith(mockBaseConfig);
    expect(screen.getByTestId('yaml-load-success')).toHaveTextContent('exported to YAML');
  });

  it('loads valid YAML file and updates configuration', async () => {
    const onLoadConfig = vi.fn();
    const validConfig: TrainingConfig = {
      ...mockBaseConfig,
      project_name: 'loaded-from-yaml',
      learning_rate: 0.0001,
    };

    vi.mocked(api.parseConfigYaml).mockResolvedValue({
      is_valid: true,
      errors: [],
      warnings: [],
      is_ready_for_training: false,
      config: validConfig,
    });

    render(
      <ConfigToolbar
        config={mockBaseConfig}
        onReset={vi.fn()}
        onLoadConfig={onLoadConfig}
      />
    );

    const file = new File(['project_name: loaded-from-yaml\nlearning_rate: 0.0001\n'], 'config.yaml', {
      type: 'text/yaml',
    });

    const fileInput = screen.getByTestId('yaml-file-input');
    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [file] } });
    });

    await waitFor(() => {
      expect(onLoadConfig).toHaveBeenCalledWith(validConfig);
      expect(screen.getByTestId('yaml-load-success')).toBeInTheDocument();
    });
  });

  it('rejects malformed YAML or invalid field values with clear error feedback', async () => {
    const onLoadConfig = vi.fn();

    vi.mocked(api.parseConfigYaml).mockResolvedValue({
      is_valid: false,
      errors: ['Malformed YAML syntax at line 2', 'learning_rate must be > 0.0'],
      warnings: [],
      is_ready_for_training: false,
      config: null,
    });

    render(
      <ConfigToolbar
        config={mockBaseConfig}
        onReset={vi.fn()}
        onLoadConfig={onLoadConfig}
      />
    );

    const file = new File(['invalid: [unclosed'], 'bad.yaml', { type: 'text/yaml' });
    const fileInput = screen.getByTestId('yaml-file-input');

    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [file] } });
    });

    await waitFor(() => {
      expect(screen.getByTestId('yaml-load-error')).toHaveTextContent('Malformed YAML syntax');
      expect(onLoadConfig).not.toHaveBeenCalled();
    });
  });

  it('resets configuration to baseline defaults when Reset Defaults is clicked', async () => {
    const handleReset = vi.fn();
    render(
      <ConfigToolbar
        config={mockBaseConfig}
        onReset={handleReset}
        onLoadConfig={vi.fn()}
      />
    );

    await act(async () => {
      fireEvent.click(screen.getByTestId('btn-reset-defaults'));
    });

    expect(handleReset).toHaveBeenCalledTimes(1);
  });

  it('guarantees Start Fine-Tuning button is strictly disabled in Phase 3', () => {
    render(
      <RunPreviewPanel
        config={mockBaseConfig}
        systemStatus={mockSystemStatus}
        backendOnline={true}
        validationResult={{
          is_valid: true,
          errors: [],
          warnings: [],
          is_ready_for_training: false, // Strictly false in Phase 3
          config: mockBaseConfig,
        }}
      />
    );

    const startBtn = screen.getByTestId('btn-start-training');
    expect(startBtn).toBeDisabled();
    expect(screen.getByTestId('phase0-preview-notice')).toHaveTextContent(
      'Phase 3 Configuration System Active'
    );
  });

  it('handles backend unavailable state in TrainingPage gracefully', async () => {
    vi.mocked(api.fetchDefaultConfig).mockRejectedValue(new Error('Network error'));
    vi.mocked(api.validateConfig).mockRejectedValue(new Error('Network error'));

    await act(async () => {
      render(
        <TrainingPage
          systemStatus={null}
          backendOnline={false}
          onRefreshSystemStatus={vi.fn()}
        />
      );
    });

    expect(screen.getByText('Backend Disconnected')).toBeInTheDocument();
    expect(screen.getByTestId('parameters-card')).toBeInTheDocument();
    expect(screen.getByTestId('btn-start-training')).toBeDisabled();
  });
});
