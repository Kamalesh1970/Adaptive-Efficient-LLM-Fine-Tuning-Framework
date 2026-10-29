import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { EvaluationView } from './EvaluationView';
import * as api from '../../services/api';
import { 
  ArtifactMetadata, 
  DatasetSummary, 
  EvaluationRun,
  EvaluationListResponse
} from '../../types/training';

vi.mock('../../services/api');

const mockArtifacts: ArtifactMetadata[] = [
  {
    artifact_id: 'art_eval_001',
    run_id: 'run_eval_001',
    artifact_type: 'lora_adapter',
    model_identifier: 'meta-llama/Llama-3-8B',
    fine_tuning_method: 'lora',
    created_at: '2026-10-28T14:30:00Z',
    file_reference: 'artifacts/run_eval_001',
    size_bytes: 14500000,
    checkpoint_step: 100,
    validation_status: 'valid',
    export_status: 'not_exported',
  },
];

const mockDatasets: DatasetSummary[] = [
  {
    id: 'dataset_alpaca_eval',
    filename: 'alpaca_clean_eval.jsonl',
    format: 'jsonl',
    total_samples: 50,
    valid_sample_count: 50,
    size_bytes: 12500,
    schema_type: 'alpaca' as any,
    is_valid: true,
    uploaded_at: '2026-10-28T14:00:00Z',
  },
];

const mockCompletedEvaluation: EvaluationRun = {
  evaluation_id: 'eval_run_001',
  run_id: 'run_eval_001',
  artifact_id: 'art_eval_001',
  dataset_id: 'dataset_alpaca_eval',
  status: 'completed',
  elapsed_seconds: 4.25,
  config: {
    artifact_id: 'art_eval_001',
    dataset_id: 'dataset_alpaca_eval',
    split: 'validation',
    max_samples: 20,
    batch_size: 2,
    temperature: 0.7,
    max_new_tokens: 64,
    run_baseline_comparison: true,
    generate_samples: true,
  },
  metrics: {
    eval_loss: 0.7241,
    perplexity: 2.06,
    perplexity_overflow: false,
    eval_runtime_seconds: 4.25,
    tokens_per_second: 38.4,
    samples_per_second: 4.7,
    eval_samples_count: 20,
    total_tokens: 163,
    unavailable_reasons: {},
  },
  baseline_comparison: {
    base_model_identifier: 'meta-llama/Llama-3-8B',
    is_comparable: true,
    base_eval_loss: 1.1524,
    base_perplexity: 3.17,
    fine_tuned_eval_loss: 0.7241,
    fine_tuned_perplexity: 2.06,
    loss_delta: -0.4283,
    perplexity_delta: -1.11,
    is_improvement: true,
    base_tokens_per_second: 39.8,
    fine_tuned_tokens_per_second: 38.4,
    comparison_summary: 'Fine-tuned adapter achieved lower validation loss (-0.4283) compared to base model.',
  },
  samples: [
    {
      sample_id: 1,
      prompt: 'Summarize the significance of PEFT fine-tuning.',
      fine_tuned_output: 'PEFT fine-tuning updates only a small subset of parameters, reducing GPU memory by up to 80%.',
      base_output: 'Fine-tuning is a technique to adapt models to specific downstream tasks.',
      expected_output: 'Parameter-efficient fine-tuning minimizes hardware requirements while maintaining performance.',
      generation_time_ms: 120,
    },
  ],
  created_at: '2026-10-28T16:00:00Z',
  started_at: '2026-10-28T16:00:02Z',
  completed_at: '2026-10-28T16:00:10Z',
  error_message: null,
};

const mockListResponse: EvaluationListResponse = {
  evaluations: [mockCompletedEvaluation],
  total: 1,
  page: 1,
  page_size: 10,
};

describe('Phase 8: EvaluationView Component Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.fetchAllArtifacts).mockResolvedValue(mockArtifacts);
    vi.mocked(api.fetchDatasets).mockResolvedValue({
      datasets: mockDatasets,
      total: 1,
    } as any);
    vi.mocked(api.fetchEvaluations).mockResolvedValue(mockListResponse);
    vi.mocked(api.fetchEvaluationDetails).mockResolvedValue(mockCompletedEvaluation);
  });

  it('renders initial setup form with artifact and dataset selectors', async () => {
    render(<EvaluationView />);

    await waitFor(() => {
      expect(screen.getByText(/1\. Select Trained Artifact/i)).toBeInTheDocument();
      expect(screen.getByText(/2\. Select Validation Dataset/i)).toBeInTheDocument();
    });

    expect(screen.getByText(/Evaluation Setup/i)).toBeInTheDocument();
  });

  it('populates initialArtifactId when provided as prop', async () => {
    render(<EvaluationView initialArtifactId="art_eval_001" />);

    await waitFor(() => {
      const artifactSelect = screen.getByTestId('select-artifact') as HTMLSelectElement;
      expect(artifactSelect.value).toBe('art_eval_001');
    });
  });

  it('starts evaluation and calls createEvaluation API', async () => {
    vi.mocked(api.createEvaluation).mockResolvedValue(mockCompletedEvaluation);

    render(<EvaluationView initialArtifactId="art_eval_001" />);

    await waitFor(() => {
      expect(screen.getByTestId('select-dataset')).toBeInTheDocument();
      expect(screen.getAllByText(/art_eval_001/i).length).toBeGreaterThan(0);
    });

    const datasetSelect = screen.getByTestId('select-dataset');
    fireEvent.change(datasetSelect, { target: { value: 'dataset_alpaca_eval' } });

    const launchBtn = screen.getByTestId('start-evaluation-btn');
    expect(launchBtn).not.toBeDisabled();
    fireEvent.click(launchBtn);

    await waitFor(() => {
      expect(api.createEvaluation).toHaveBeenCalledWith(expect.objectContaining({
        artifact_id: 'art_eval_001',
        dataset_id: 'dataset_alpaca_eval',
        split: 'validation',
        max_samples: 20,
        batch_size: 2,
        run_baseline_comparison: true,
      }));
    });
  });

  it('displays real evaluation metrics truthfully', async () => {
    render(<EvaluationView />);

    await waitFor(() => {
      expect(screen.getByTestId('eval-metrics-panel')).toBeInTheDocument();
    });

    expect(screen.getByTestId('metric-eval-loss')).toHaveTextContent('0.7241');
    expect(screen.getByTestId('metric-perplexity')).toHaveTextContent('2.06');
    expect(screen.getByTestId('metric-throughput')).toHaveTextContent('38.4');
  });

  it('displays baseline comparison metrics side-by-side when valid', async () => {
    render(<EvaluationView />);

    await waitFor(() => {
      expect(screen.getByTestId('baseline-comparison-card')).toBeInTheDocument();
    });

    expect(screen.getByText('1.1524')).toBeInTheDocument();
    expect(screen.getByText(/Fine-tuned adapter achieved lower validation loss/i)).toBeInTheDocument();
  });

  it('displays qualitative generated sample outputs', async () => {
    render(<EvaluationView />);

    await waitFor(() => {
      expect(screen.getByTestId('eval-samples-panel')).toBeInTheDocument();
    });

    expect(screen.getByText('Summarize the significance of PEFT fine-tuning.')).toBeInTheDocument();
    expect(screen.getByText(/reducing GPU memory by up to 80%/i)).toBeInTheDocument();
    expect(screen.getByText(/Fine-tuning is a technique to adapt models/i)).toBeInTheDocument();
  });

  it('handles perplexity overflow honestly', async () => {
    const overflowEvaluation: EvaluationRun = {
      ...mockCompletedEvaluation,
      evaluation_id: 'eval_overflow_002',
      metrics: {
        ...mockCompletedEvaluation.metrics!,
        eval_loss: 92.5,
        perplexity: null,
        perplexity_overflow: true,
      },
      baseline_comparison: null,
    };

    vi.mocked(api.fetchEvaluations).mockResolvedValue({
      evaluations: [overflowEvaluation],
      total: 1,
      page: 1,
      page_size: 10,
    });
    vi.mocked(api.fetchEvaluationDetails).mockResolvedValue(overflowEvaluation);

    render(<EvaluationView />);

    await waitFor(() => {
      expect(screen.getByTestId('metric-perplexity')).toHaveTextContent('Overflow');
    });
  });

  it('allows cancelling an evaluation run if running and cancellable', async () => {
    const runningEvaluation: EvaluationRun = {
      ...mockCompletedEvaluation,
      evaluation_id: 'eval_running_003',
      status: 'running',
    };

    vi.mocked(api.fetchEvaluations).mockResolvedValue({
      evaluations: [runningEvaluation],
      total: 1,
      page: 1,
      page_size: 10,
    });
    vi.mocked(api.fetchEvaluationDetails).mockResolvedValue(runningEvaluation);
    vi.mocked(api.cancelEvaluation).mockResolvedValue({
      ...runningEvaluation,
      status: 'cancelled',
    });

    render(<EvaluationView />);

    await waitFor(() => {
      expect(screen.getByTestId('cancel-eval-btn')).toBeInTheDocument();
    });

    const cancelBtn = screen.getByTestId('cancel-eval-btn');
    fireEvent.click(cancelBtn);

    await waitFor(() => {
      expect(api.cancelEvaluation).toHaveBeenCalledWith('eval_running_003');
    });
  });
});
