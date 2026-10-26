import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { DatasetCard } from './DatasetCard';
import { TrainingPage } from './TrainingPage';
import * as api from '../../services/api';
import { 
  TrainingConfig, 
  DatasetSummary, 
  DatasetValidationReport, 
  DatasetSplitResult 
} from '../../types/training';

vi.mock('../../services/api', () => ({
  fetchHealth: vi.fn(),
  fetchSystemStatus: vi.fn(),
  fetchDefaultConfig: vi.fn(),
  validateConfig: vi.fn(),
  fetchModelCatalog: vi.fn(),
  fetchModelMetadata: vi.fn(),
  validateModelSelection: vi.fn(),
  uploadDatasetFile: vi.fn(),
  fetchDatasets: vi.fn(),
  fetchDatasetReport: vi.fn(),
  fetchDatasetSplits: vi.fn(),
}));

describe('Phase 2: Dataset Upload, Validation and Split Selection Tests', () => {
  const mockDefaultConfig: TrainingConfig = {
    project_name: 'my-finetuned-model',
    model_identifier: 'mistralai/Mistral-7B-v0.3',
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

  const mockDatasets: DatasetSummary[] = [
    {
      id: 'ds_test_1',
      filename: 'alpaca_clean.json',
      format: 'json',
      size_bytes: 1024,
      uploaded_at: '2026-10-26T00:00:00Z',
      total_samples: 50,
      valid_sample_count: 50,
      schema_type: 'instruction',
      is_valid: true,
    },
    {
      id: 'ds_test_2',
      filename: 'broken_data.jsonl',
      format: 'jsonl',
      size_bytes: 512,
      uploaded_at: '2026-10-26T00:05:00Z',
      total_samples: 10,
      valid_sample_count: 2,
      schema_type: null,
      is_valid: false,
    },
  ];

  const mockValidationReport: DatasetValidationReport = {
    dataset_id: 'ds_test_1',
    original_filename: 'alpaca_clean.json',
    file_format: 'json',
    file_size_bytes: 1024,
    detected_schema: 'instruction',
    total_samples: 50,
    valid_sample_count: 50,
    invalid_sample_count: 0,
    empty_sample_count: 0,
    duplicate_count: 0,
    is_valid: true,
    warnings: [],
    errors: [],
    preview_samples: [
      { instruction: 'Translate to French', input: 'Hello', output: 'Bonjour' },
    ],
  };

  const mockSplitResult: DatasetSplitResult = {
    dataset_id: 'ds_test_1',
    total_samples: 50,
    train_count: 45,
    eval_count: 5,
    train_indices: Array.from({ length: 45 }, (_, i) => i),
    eval_indices: Array.from({ length: 5 }, (_, i) => i + 45),
    has_overlap: false,
    warnings: [],
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
      models: [],
      total: 0,
    });
    vi.mocked(api.fetchDatasets).mockResolvedValue({
      datasets: mockDatasets,
      total: mockDatasets.length,
    });
    vi.mocked(api.uploadDatasetFile).mockResolvedValue(mockValidationReport);
    vi.mocked(api.fetchDatasetReport).mockResolvedValue(mockValidationReport);
    vi.mocked(api.fetchDatasetSplits).mockResolvedValue(mockSplitResult);
  });

  it('renders upload control, dropzone, and supported format info', () => {
    render(
      <DatasetCard
        config={mockDefaultConfig}
        onChange={vi.fn()}
        datasets={mockDatasets}
      />
    );

    expect(screen.getByTestId('dataset-card')).toBeInTheDocument();
    expect(screen.getByTestId('upload-dropzone')).toBeInTheDocument();
    expect(screen.getByText(/Drag & drop dataset files here/)).toBeInTheDocument();
    expect(screen.getByText(/\.json, \.jsonl, \.csv/)).toBeInTheDocument();
  });

  it('handles client-side unsupported format feedback immediately', async () => {
    render(
      <DatasetCard
        config={mockDefaultConfig}
        onChange={vi.fn()}
      />
    );

    const input = screen.getByTestId('file-input');
    const fakeParquetFile = new File(['fake content'], 'test.parquet', { type: 'application/octet-stream' });

    await act(async () => {
      fireEvent.change(input, { target: { files: [fakeParquetFile] } });
    });

    expect(screen.getByTestId('dataset-error-banner')).toBeInTheDocument();
    expect(screen.getByText(/Unsupported format 'test.parquet'/)).toBeInTheDocument();
  });

  it('handles file size limit validation feedback', async () => {
    render(
      <DatasetCard
        config={mockDefaultConfig}
        onChange={vi.fn()}
      />
    );

    const input = screen.getByTestId('file-input');
    // Create a mock large file > 50MB
    const largeFile = new File(['large'], 'huge.json', { type: 'application/json' });
    Object.defineProperty(largeFile, 'size', { value: 60 * 1024 * 1024 });

    await act(async () => {
      fireEvent.change(input, { target: { files: [largeFile] } });
    });

    expect(screen.getByTestId('dataset-error-banner')).toBeInTheDocument();
    expect(screen.getByText(/exceeds the 50MB size limit/)).toBeInTheDocument();
  });

  it('handles upload failure and displays error message', () => {
    render(
      <DatasetCard
        config={mockDefaultConfig}
        onChange={vi.fn()}
        uploadError="Server connection refused during upload"
      />
    );

    expect(screen.getByTestId('dataset-error-banner')).toBeInTheDocument();
    expect(screen.getByText(/Server connection refused during upload/)).toBeInTheDocument();
  });

  it('renders upload success and validated dataset metrics', () => {
    render(
      <DatasetCard
        config={{ ...mockDefaultConfig, dataset_reference: 'ds_test_1' }}
        onChange={vi.fn()}
        uploadSuccess={true}
        selectedReport={mockValidationReport}
      />
    );

    expect(screen.getByTestId('upload-status')).toBeInTheDocument();
    expect(screen.getByTestId('dataset-validation-summary')).toBeInTheDocument();
    expect(screen.getByTestId('dataset-schema-badge')).toHaveTextContent('instruction');
    expect(screen.getByText('alpaca_clean.json')).toBeInTheDocument();
    expect(screen.getAllByText('50').length).toBeGreaterThanOrEqual(1); // valid/total count
  });

  it('renders validation error banner when dataset has invalid samples', () => {
    const brokenReport: DatasetValidationReport = {
      dataset_id: 'ds_test_2',
      original_filename: 'broken_data.jsonl',
      file_format: 'jsonl',
      file_size_bytes: 512,
      detected_schema: null,
      total_samples: 10,
      valid_sample_count: 2,
      invalid_sample_count: 8,
      empty_sample_count: 0,
      duplicate_count: 0,
      is_valid: false,
      warnings: [],
      errors: [
        { sample_index: 2, message: 'Malformed JSON syntax on line 2' },
        { sample_index: 5, message: 'Missing required output field' },
      ],
      preview_samples: [],
    };

    render(
      <DatasetCard
        config={{ ...mockDefaultConfig, dataset_reference: 'ds_test_2' }}
        onChange={vi.fn()}
        selectedReport={brokenReport}
      />
    );

    expect(screen.getByTestId('dataset-error-banner')).toBeInTheDocument();
    expect(screen.getByText(/Malformed JSON syntax on line 2/)).toBeInTheDocument();
    expect(screen.getByText(/Missing required output field/)).toBeInTheDocument();
  });

  it('allows selecting an existing dataset and fires onSelectDataset', () => {
    const handleSelect = vi.fn();
    const handleChange = vi.fn();

    render(
      <DatasetCard
        config={mockDefaultConfig}
        onChange={handleChange}
        datasets={mockDatasets}
        onSelectDataset={handleSelect}
      />
    );

    const select = screen.getByTestId('dataset-select');
    fireEvent.change(select, { target: { value: 'ds_test_1' } });

    expect(handleChange).toHaveBeenCalledWith({ dataset_reference: 'ds_test_1' });
    expect(handleSelect).toHaveBeenCalledWith('ds_test_1');
  });

  it('renders split selection controls and updates deterministic statistics', () => {
    const handleSplitChange = vi.fn();

    render(
      <DatasetCard
        config={{ ...mockDefaultConfig, dataset_reference: 'ds_test_1', train_split: 'train', eval_split: 'validation' }}
        onChange={vi.fn()}
        splitResult={mockSplitResult}
        trainRatio={0.9}
        evalRatio={0.1}
        onSplitChange={handleSplitChange}
      />
    );

    expect(screen.getByTestId('train-split-select')).toBeInTheDocument();
    expect(screen.getByTestId('eval-split-select')).toBeInTheDocument();
    expect(screen.getByTestId('split-stats')).toBeInTheDocument();
    expect(screen.getByText('45')).toBeInTheDocument(); // train count
    expect(screen.getByText('5')).toBeInTheDocument();  // eval count
    expect(screen.getByText(/0% Overlap/)).toBeInTheDocument();
  });

  it('displays backend unavailable state when offline', () => {
    render(
      <DatasetCard
        config={mockDefaultConfig}
        onChange={vi.fn()}
        backendOnline={false}
      />
    );

    expect(screen.getByText(/Backend is offline/)).toBeInTheDocument();
  });

  it('integration: uploading a file in TrainingPage updates dataset state and run preview', async () => {
    await act(async () => {
      render(
        <TrainingPage
          systemStatus={null}
          backendOnline={true}
          onRefreshSystemStatus={vi.fn()}
        />
      );
    });

    const fileInput = screen.getByTestId('file-input');
    const validFile = new File(['[{"instruction":"hi","output":"bye"}]'], 'prompts.json', { type: 'application/json' });

    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [validFile] } });
    });

    // Verify uploadDatasetFile called
    await waitFor(() => {
      expect(api.uploadDatasetFile).toHaveBeenCalledTimes(1);
    });

    // Verify run preview reflects dataset
    await waitFor(() => {
      expect(screen.getByTitle('ds_test_1')).toBeInTheDocument();
      expect(screen.getByText(/50 valid/)).toBeInTheDocument();
    });

    // Start training action MUST remain strictly disabled
    const startBtn = screen.getByTestId('btn-start-training');
    expect(startBtn).toBeDisabled();
  });
});
