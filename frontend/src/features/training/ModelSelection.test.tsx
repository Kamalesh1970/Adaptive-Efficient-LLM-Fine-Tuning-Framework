import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { ModelCard } from './ModelCard';
import { TrainingPage } from './TrainingPage';
import * as api from '../../services/api';
import { ModelMetadata, TrainingConfig, ModelValidationResponse } from '../../types/training';

vi.mock('../../services/api', () => ({
  fetchHealth: vi.fn(),
  fetchSystemStatus: vi.fn(),
  fetchDefaultConfig: vi.fn(),
  validateConfig: vi.fn(),
  fetchModelCatalog: vi.fn(),
  fetchModelMetadata: vi.fn(),
  validateModelSelection: vi.fn(),
}));

describe('Phase 1: Model Selection, Catalog Integration, and Compatibility Tests', () => {
  const mockCatalog: ModelMetadata[] = [
    {
      id: 'meta-llama/Meta-Llama-3-8B',
      name: 'Meta Llama 3 8B',
      family: 'llama',
      source: 'huggingface',
      task_type: 'causal_lm',
      parameter_count: '8.0B',
      context_length: 8192,
      is_gated: true,
      verification_status: 'catalog_registered',
      supported_methods: ['lora', 'qlora', 'full_finetuning'],
      compatibility_notes: 'Gated model requiring approved license.',
      is_downloaded: false,
      is_tested: false,
    },
    {
      id: 'mistralai/Mistral-7B-v0.3',
      name: 'Mistral 7B v0.3',
      family: 'mistral',
      source: 'huggingface',
      task_type: 'causal_lm',
      parameter_count: '7.2B',
      context_length: 32768,
      is_gated: false,
      verification_status: 'catalog_registered',
      supported_methods: ['lora', 'qlora', 'full_finetuning'],
      compatibility_notes: 'Open weights model.',
      is_downloaded: false,
      is_tested: false,
    },
    {
      id: 'google/gemma-2-2b',
      name: 'Gemma 2 2B',
      family: 'gemma2',
      source: 'huggingface',
      task_type: 'causal_lm',
      parameter_count: '2.6B',
      context_length: 8192,
      is_gated: true,
      verification_status: 'catalog_registered',
      supported_methods: ['lora', 'qlora'], // full_finetuning excluded
      compatibility_notes: 'Full fine-tuning is unsupported.',
      is_downloaded: false,
      is_tested: false,
    },
    {
      id: 'custom/unverified-experimental-lm',
      name: 'Custom Experimental LM',
      family: null,
      source: 'local',
      task_type: 'causal_lm',
      parameter_count: null,
      context_length: null,
      is_gated: false,
      verification_status: 'unverified',
      supported_methods: ['lora'],
      compatibility_notes: 'Unverified custom architecture.',
      is_downloaded: false,
      is_tested: false,
    },
  ];

  const mockDefaultConfig: TrainingConfig = {
    project_name: 'my-finetuned-model',
    model_identifier: null,
    training_method: 'lora',
    dataset_reference: null,
    dataset_subset: null,
    train_split: 'train',
    eval_split: 'none',
    training_length_type: 'steps',
    training_length: 100,
    batch_size: 2,
    context_length: 2048,
    learning_rate: 0.0002,
    advanced_enabled: false,
    hf_token_status: 'not_detected',
  };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.fetchDefaultConfig).mockResolvedValue(mockDefaultConfig);
    vi.mocked(api.validateConfig).mockResolvedValue({
      is_valid: true,
      errors: [],
      warnings: [],
      is_ready_for_training: false,
      config: mockDefaultConfig,
    });
    vi.mocked(api.fetchModelCatalog).mockResolvedValue({
      models: mockCatalog,
      total: mockCatalog.length,
    });
    vi.mocked(api.validateModelSelection).mockResolvedValue({
      is_valid: true,
      model_id: 'mistralai/Mistral-7B-v0.3',
      training_method: 'lora',
      compatibility_status: 'compatible',
      errors: [],
      warnings: [],
      model: mockCatalog[1],
      token_status: 'not_provided',
      details: {},
    });
  });

  it('renders model selector with options from catalog', () => {
    render(
      <ModelCard
        config={mockDefaultConfig}
        onChange={vi.fn()}
        catalog={mockCatalog}
      />
    );

    const select = screen.getByTestId('model-select') as HTMLSelectElement;
    expect(select).toBeInTheDocument();
    expect(screen.getByText(/-- No model selected --/)).toBeInTheDocument();
    expect(screen.getByText(/Meta Llama 3 8B/)).toBeInTheDocument();
    expect(screen.getByText(/Mistral 7B v0.3/)).toBeInTheDocument();
  });

  it('handles catalog loading state gracefully', () => {
    render(
      <ModelCard
        config={mockDefaultConfig}
        onChange={vi.fn()}
        catalog={[]}
        isLoadingCatalog={true}
      />
    );

    const select = screen.getByTestId('model-select');
    expect(select).toBeDisabled();
    expect(screen.getByTestId('catalog-loading')).toBeInTheDocument();
    expect(screen.getByText(/Loading verified model catalog/)).toBeInTheDocument();
  });

  it('renders empty catalog when no models are available', () => {
    render(
      <ModelCard
        config={mockDefaultConfig}
        onChange={vi.fn()}
        catalog={[]}
        isLoadingCatalog={false}
      />
    );

    const select = screen.getByTestId('model-select');
    expect(select).not.toBeDisabled();
    expect(screen.getByTestId('model-card')).toBeInTheDocument();
  });

  it('handles API failure with clear error notice and retry trigger', () => {
    const handleRetry = vi.fn();
    render(
      <ModelCard
        config={mockDefaultConfig}
        onChange={vi.fn()}
        catalog={[]}
        catalogError="Failed to connect to backend model service"
        onRetryCatalog={handleRetry}
      />
    );

    expect(screen.getByTestId('catalog-error')).toBeInTheDocument();
    expect(screen.getByText(/Failed to connect to backend model service/)).toBeInTheDocument();

    const retryBtn = screen.getByRole('button', { name: /Retry/i });
    fireEvent.click(retryBtn);
    expect(handleRetry).toHaveBeenCalledTimes(1);
  });

  it('displays model metadata summary when a catalog model is selected', () => {
    const selectedConfig: TrainingConfig = {
      ...mockDefaultConfig,
      model_identifier: 'meta-llama/Meta-Llama-3-8B',
    };

    render(
      <ModelCard
        config={selectedConfig}
        onChange={vi.fn()}
        catalog={mockCatalog}
      />
    );

    expect(screen.getByTestId('model-metadata-summary')).toBeInTheDocument();
    expect(screen.getByText('8.0B')).toBeInTheDocument();
    expect(screen.getByText('llama')).toBeInTheDocument();
    expect(screen.getByText(/8,192 tokens/)).toBeInTheDocument();
    expect(screen.getByText(/Gated Access/)).toBeInTheDocument();
    expect(screen.getByText(/Not Downloaded/)).toBeInTheDocument();
  });

  it('renders training method selector and fires onChange', () => {
    const handleChange = vi.fn();
    render(
      <ModelCard
        config={mockDefaultConfig}
        onChange={handleChange}
        catalog={mockCatalog}
      />
    );

    const methodSelect = screen.getByTestId('method-select');
    expect(methodSelect).toBeInTheDocument();

    fireEvent.change(methodSelect, { target: { value: 'qlora' } });
    expect(handleChange).toHaveBeenCalledWith({ training_method: 'qlora' });
  });

  it('renders validation success feedback when configuration is compatible', () => {
    const compatibleConfig: TrainingConfig = {
      ...mockDefaultConfig,
      model_identifier: 'mistralai/Mistral-7B-v0.3',
      training_method: 'lora',
    };

    const validation: ModelValidationResponse = {
      is_valid: true,
      model_id: 'mistralai/Mistral-7B-v0.3',
      training_method: 'lora',
      compatibility_status: 'compatible',
      errors: [],
      warnings: ['Recommended batch size: 2-4'],
      model: mockCatalog[1],
      token_status: 'not_provided',
      details: {},
    };

    render(
      <ModelCard
        config={compatibleConfig}
        onChange={vi.fn()}
        catalog={mockCatalog}
        modelValidation={validation}
      />
    );

    expect(screen.getByTestId('validation-feedback')).toBeInTheDocument();
    expect(screen.getByText(/Configuration Compatible/)).toBeInTheDocument();
    expect(screen.getByText(/Recommended batch size/)).toBeInTheDocument();
  });

  it('renders validation failure and error banners for unsupported method', () => {
    const incompatibleConfig: TrainingConfig = {
      ...mockDefaultConfig,
      model_identifier: 'google/gemma-2-2b',
      training_method: 'full_finetuning',
    };

    const validation: ModelValidationResponse = {
      is_valid: false,
      model_id: 'google/gemma-2-2b',
      training_method: 'full_finetuning',
      compatibility_status: 'incompatible',
      errors: ["Model 'google/gemma-2-2b' does not support 'full_finetuning'."],
      warnings: [],
      model: mockCatalog[2],
      token_status: 'not_provided',
      details: {},
    };

    render(
      <ModelCard
        config={incompatibleConfig}
        onChange={vi.fn()}
        catalog={mockCatalog}
        modelValidation={validation}
      />
    );

    expect(screen.getByTestId('validation-error-banner')).toBeInTheDocument();
    expect(screen.getByText(/Method Compatibility Error/)).toBeInTheDocument();
    expect(screen.getByText(/does not support 'full_finetuning'/)).toBeInTheDocument();
  });

  it('renders credential requirement banner for gated models missing token', () => {
    const gatedConfig: TrainingConfig = {
      ...mockDefaultConfig,
      model_identifier: 'meta-llama/Meta-Llama-3-8B',
    };

    const validation: ModelValidationResponse = {
      is_valid: false,
      model_id: 'meta-llama/Meta-Llama-3-8B',
      training_method: 'lora',
      compatibility_status: 'requires_credentials',
      errors: ['Model requires approved Hugging Face gated access credentials.'],
      warnings: [],
      model: mockCatalog[0],
      token_status: 'not_provided',
      details: {},
    };

    render(
      <ModelCard
        config={gatedConfig}
        onChange={vi.fn()}
        catalog={mockCatalog}
        modelValidation={validation}
      />
    );

    expect(screen.getByTestId('validation-error-banner')).toBeInTheDocument();
    expect(screen.getByText(/Hugging Face Access Token Required/)).toBeInTheDocument();
  });

  it('provides masked token input and ensures token is not exposed in DOM or debug logs', () => {
    const secretToken = 'hf_test_super_secret_token_12345';
    let currentToken = '';

    const { rerender } = render(
      <ModelCard
        config={mockDefaultConfig}
        onChange={vi.fn()}
        catalog={mockCatalog}
        hfToken={currentToken}
        onHfTokenChange={(val) => {
          currentToken = val;
        }}
      />
    );

    const tokenInput = screen.getByTestId('hf-token-input') as HTMLInputElement;
    expect(tokenInput).toBeInTheDocument();
    expect(tokenInput.type).toBe('password'); // Masked by default

    // User types in token
    fireEvent.change(tokenInput, { target: { value: secretToken } });

    rerender(
      <ModelCard
        config={mockDefaultConfig}
        onChange={vi.fn()}
        catalog={mockCatalog}
        hfToken={secretToken}
        onHfTokenChange={vi.fn()}
      />
    );

    // Verify token status badge reflects in-memory unverified configuration
    expect(screen.getByTestId('hf-token-badge')).toHaveTextContent('Configured (In-Memory, Unverified)');

    // Toggle mask button switches type to text and back to password
    const toggleBtn = screen.getByTestId('btn-toggle-token-mask');
    fireEvent.click(toggleBtn);
    expect((screen.getByTestId('hf-token-input') as HTMLInputElement).type).toBe('text');

    fireEvent.click(toggleBtn);
    expect((screen.getByTestId('hf-token-input') as HTMLInputElement).type).toBe('password');
  });

  it('integration: selecting model in TrainingPage triggers catalog validation and updates preview', async () => {
    await act(async () => {
      render(
        <TrainingPage
          systemStatus={null}
          backendOnline={true}
          onRefreshSystemStatus={vi.fn()}
        />
      );
    });

    // Wait for catalog to load
    await waitFor(() => {
      expect(screen.getByText(/Meta Llama 3 8B/)).toBeInTheDocument();
    });

    const modelSelect = screen.getByTestId('model-select');
    await act(async () => {
      fireEvent.change(modelSelect, { target: { value: 'mistralai/Mistral-7B-v0.3' } });
    });

    // Verify preview reflects selection
    await waitFor(() => {
      expect(screen.getByTitle('mistralai/Mistral-7B-v0.3')).toBeInTheDocument();
    });
  });
});
