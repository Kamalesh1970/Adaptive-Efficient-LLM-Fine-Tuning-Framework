import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import App from './App';
import * as api from './services/api';

vi.mock('./services/api', () => ({
  fetchHealth: vi.fn(),
  fetchSystemStatus: vi.fn(),
  fetchDefaultConfig: vi.fn(),
  validateConfig: vi.fn(),
}));

describe('Adaptive LLM Fine-Tuning Studio - Phase 0 UI Skeleton', () => {
  const mockSystemStatus = {
    python_version: '3.12.3',
    os_platform: 'Linux-x86_64',
    pytorch_available: true,
    pytorch_version: '2.3.0',
    cuda_available: true,
    cuda_version: '12.1',
    gpu_count: 1,
    gpu_devices: [
      { index: 0, name: 'NVIDIA RTX 4090', total_vram_gb: 24.0, total_vram_mb: 24576.0 }
    ],
    primary_gpu_name: 'NVIDIA RTX 4090',
    primary_gpu_vram_gb: 24.0,
    environment_type: 'cuda_accelerated',
  };

  const mockDefaultConfig = {
    project_name: 'my-finetuned-model',
    model_identifier: null,
    training_method: 'lora' as const,
    dataset_reference: null,
    dataset_subset: null,
    train_split: 'train',
    eval_split: 'none',
    training_length_type: 'steps' as const,
    training_length: 100,
    batch_size: 2,
    context_length: 2048,
    learning_rate: 0.0002,
    advanced_enabled: false,
    hf_token_status: 'not_detected',
  };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.fetchHealth).mockResolvedValue({
      status: 'ok',
      app_name: 'Adaptive LLM Fine-Tuning Studio',
      version: '0.1.0-phase0',
    });
    vi.mocked(api.fetchSystemStatus).mockResolvedValue(mockSystemStatus);
    vi.mocked(api.fetchDefaultConfig).mockResolvedValue(mockDefaultConfig);
    vi.mocked(api.validateConfig).mockResolvedValue({
      is_valid: true,
      errors: [],
      warnings: ['No base model selected', 'No dataset selected', 'Phase 0 UI Prototype active'],
      is_ready_for_training: false,
      config: mockDefaultConfig,
    });
  });

  it('renders application sidebar with branding, navigation items, and backend status', async () => {
    await act(async () => {
      render(<App />);
    });

    expect(screen.getByTestId('sidebar')).toBeInTheDocument();
    expect(screen.getByText('Adaptive Studio')).toBeInTheDocument();
    expect(screen.getByText('LLM Fine-Tuning Studio')).toBeInTheDocument();

    // Navigation items
    expect(screen.getByTestId('nav-new-training')).toBeInTheDocument();
    expect(screen.getByTestId('nav-models')).toBeInTheDocument();
    expect(screen.getByTestId('nav-datasets')).toBeInTheDocument();
    expect(screen.getByTestId('nav-history')).toBeInTheDocument();
    expect(screen.getByTestId('nav-settings')).toBeInTheDocument();

    // Legal links
    expect(screen.getByTestId('link-privacy')).toBeInTheDocument();
    expect(screen.getByTestId('link-terms')).toBeInTheDocument();

    // Verify backend status indicator
    await waitFor(() => {
      expect(screen.getByTestId('backend-status-badge')).toHaveTextContent('Connected');
    });
  });

  it('renders header with title, subtitle, and tabs', async () => {
    await act(async () => {
      render(<App />);
    });

    expect(screen.getByText('Train')).toBeInTheDocument();
    expect(screen.getByText('Configure and start fine-tuning')).toBeInTheDocument();
    expect(screen.getByTestId('tab-configure')).toBeInTheDocument();
    expect(screen.getByTestId('tab-current-run')).toBeInTheDocument();
    expect(screen.getByTestId('tab-history')).toBeInTheDocument();
  });

  it('renders all key configuration sections and persistent run preview panel', async () => {
    await act(async () => {
      render(<App />);
    });

    expect(screen.getByTestId('model-card')).toBeInTheDocument();
    expect(screen.getByTestId('dataset-card')).toBeInTheDocument();
    expect(screen.getByTestId('parameters-card')).toBeInTheDocument();
    expect(screen.getByTestId('config-toolbar')).toBeInTheDocument();
    expect(screen.getByTestId('run-preview-panel')).toBeInTheDocument();
  });

  it('verifies default configuration state indicates no model and no dataset selected', async () => {
    await act(async () => {
      render(<App />);
    });

    await waitFor(() => {
      expect(screen.getByTestId('status-model-none')).toHaveTextContent('None Selected');
      expect(screen.getByTestId('status-dataset-none')).toHaveTextContent('None Selected');
    });

    const modelSelect = screen.getByTestId('model-select') as HTMLSelectElement;
    expect(modelSelect.value).toBe('');

    const datasetSelect = screen.getByTestId('dataset-select') as HTMLSelectElement;
    expect(datasetSelect.value).toBe('');
  });

  it('guarantees the Start Fine-Tuning action button is strictly disabled in Phase 0', async () => {
    await act(async () => {
      render(<App />);
    });

    const startBtn = screen.getByTestId('btn-start-training');
    expect(startBtn).toBeDisabled();
    expect(screen.getByTestId('phase0-preview-notice')).toBeInTheDocument();
  });

  it('allows toggling between Simple and Advanced parameters', async () => {
    await act(async () => {
      render(<App />);
    });

    expect(screen.getByTestId('simple-parameters-view')).toBeInTheDocument();

    // Click Advanced tab
    await act(async () => {
      fireEvent.click(screen.getByTestId('param-tab-advanced'));
    });
    expect(screen.getByTestId('advanced-parameters-placeholder')).toBeInTheDocument();

    // Click back to Simple tab
    await act(async () => {
      fireEvent.click(screen.getByTestId('param-tab-simple'));
    });
    expect(screen.getByTestId('simple-parameters-view')).toBeInTheDocument();
  });

  it('updates run preview when model and dataset are selected, keeping start button safely disabled', async () => {
    await act(async () => {
      render(<App />);
    });

    const modelSelect = screen.getByTestId('model-select');
    await act(async () => {
      fireEvent.change(modelSelect, { target: { value: 'meta-llama/Meta-Llama-3-8B' } });
    });

    const datasetSelect = screen.getByTestId('dataset-select');
    await act(async () => {
      fireEvent.change(datasetSelect, { target: { value: 'yahma/alpaca-cleaned' } });
    });

    // Verify preview reflects selection
    expect(screen.getByTitle('meta-llama/Meta-Llama-3-8B')).toBeInTheDocument();
    expect(screen.getByTitle('yahma/alpaca-cleaned')).toBeInTheDocument();

    // Button MUST remain disabled in Phase 0
    const startBtn = screen.getByTestId('btn-start-training');
    expect(startBtn).toBeDisabled();
  });

  it('renders Privacy Policy page when clicked and returns to studio view', async () => {
    await act(async () => {
      render(<App />);
    });

    await act(async () => {
      fireEvent.click(screen.getByTestId('link-privacy'));
    });

    const page = screen.getByTestId('privacy-policy-page');
    expect(page).toBeInTheDocument();
    expect(screen.getByRole('heading', { level: 1, name: 'Privacy Policy' })).toBeInTheDocument();

    // Return back to studio
    await act(async () => {
      fireEvent.click(screen.getByText('Back to Studio'));
    });

    expect(screen.getByTestId('training-page')).toBeInTheDocument();
  });

  it('renders Terms and Conditions page when clicked and returns to studio view', async () => {
    await act(async () => {
      render(<App />);
    });

    await act(async () => {
      fireEvent.click(screen.getByTestId('link-terms'));
    });

    const page = screen.getByTestId('terms-conditions-page');
    expect(page).toBeInTheDocument();
    expect(screen.getByRole('heading', { level: 1, name: 'Terms and Conditions' })).toBeInTheDocument();

    // Return back to studio
    await act(async () => {
      fireEvent.click(screen.getByText('Back to Studio'));
    });

    expect(screen.getByTestId('training-page')).toBeInTheDocument();
  });

  it('handles backend offline status gracefully without throwing', async () => {
    vi.mocked(api.fetchHealth).mockRejectedValue(new Error('Connection refused'));
    vi.mocked(api.fetchSystemStatus).mockRejectedValue(new Error('Connection refused'));

    await act(async () => {
      render(<App />);
    });

    await waitFor(() => {
      expect(screen.getByTestId('backend-status-badge')).toHaveTextContent('Offline');
    });
  });
});
