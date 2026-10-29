import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ArtifactsView } from './ArtifactsView';
import * as api from '../../services/api';
import { ArtifactMetadata, ArtifactExportResponse, ArtifactValidationResponse } from '../../types/training';

vi.mock('../../services/api');

const mockArtifacts: ArtifactMetadata[] = [
  {
    artifact_id: 'art_run_001_lora',
    run_id: 'run_001',
    artifact_type: 'lora_adapter',
    model_identifier: 'meta-llama/Llama-3-8B',
    fine_tuning_method: 'lora',
    created_at: '2026-10-28T14:30:00Z',
    file_reference: 'artifacts/run_001',
    size_bytes: 14500000,
    adapter_config: {
      r: 16,
      lora_alpha: 32,
      lora_dropout: 0.05,
      target_modules: ['q_proj', 'v_proj'],
    },
    checkpoint_step: 100,
    validation_status: 'valid',
    export_status: 'not_exported',
    validation_errors: [],
    validation_warnings: [],
    files: [
      {
        filename: 'adapter_config.json',
        size_bytes: 650,
        sha256: 'a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2',
        is_required: true,
        exists: true,
      },
      {
        filename: 'adapter_model.safetensors',
        size_bytes: 14499350,
        sha256: 'f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5',
        is_required: true,
        exists: true,
      },
    ],
  },
  {
    artifact_id: 'art_run_002_corrupt',
    run_id: 'run_002',
    artifact_type: 'qlora_adapter',
    model_identifier: 'mistralai/Mistral-7B-v0.1',
    fine_tuning_method: 'qlora',
    created_at: '2026-10-28T15:00:00Z',
    file_reference: 'artifacts/run_002',
    size_bytes: 400,
    checkpoint_step: null,
    validation_status: 'invalid',
    export_status: 'not_exported',
    validation_errors: ['adapter_model.safetensors missing on disk'],
    validation_warnings: [],
  },
];

describe('Phase 8: ArtifactsView Component Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.fetchAllArtifacts).mockResolvedValue(mockArtifacts);
    vi.mocked(api.getArtifactDownloadUrl).mockImplementation((id) => `/api/artifacts/${id}/download`);
  });

  it('renders loading state initially then displays artifact list', async () => {
    render(<ArtifactsView />);
    expect(screen.getByText(/Scanning disk for validated artifacts/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('art_run_001_lora')).toBeInTheDocument();
      expect(screen.getByText('art_run_002_corrupt')).toBeInTheDocument();
    });

    expect(screen.getByText('meta-llama/Llama-3-8B')).toBeInTheDocument();
    expect(screen.getByText('mistralai/Mistral-7B-v0.1')).toBeInTheDocument();
  });

  it('renders empty state when no artifacts are found', async () => {
    vi.mocked(api.fetchAllArtifacts).mockResolvedValue([]);
    render(<ArtifactsView />);

    await waitFor(() => {
      expect(screen.getByText(/No artifacts found/i)).toBeInTheDocument();
    });
  });

  it('filters artifacts by search query', async () => {
    render(<ArtifactsView />);

    await waitFor(() => {
      expect(screen.getByText('art_run_001_lora')).toBeInTheDocument();
    });

    const searchInput = screen.getByTestId('search-artifacts-input');
    fireEvent.change(searchInput, { target: { value: 'mistral' } });

    expect(screen.queryByText('art_run_001_lora')).not.toBeInTheDocument();
    expect(screen.getByText('art_run_002_corrupt')).toBeInTheDocument();
  });

  it('filters artifacts by status filter dropdown', async () => {
    render(<ArtifactsView />);

    await waitFor(() => {
      expect(screen.getByText('art_run_001_lora')).toBeInTheDocument();
    });

    const statusSelect = screen.getByTestId('filter-status-select');
    fireEvent.change(statusSelect, { target: { value: 'invalid' } });

    expect(screen.queryByText('art_run_001_lora')).not.toBeInTheDocument();
    expect(screen.getByText('art_run_002_corrupt')).toBeInTheDocument();
  });

  it('opens details modal when Inspect button is clicked and shows manifest with SHA-256', async () => {
    render(<ArtifactsView />);

    await waitFor(() => {
      expect(screen.getByText('art_run_001_lora')).toBeInTheDocument();
    });

    const inspectButtons = screen.getAllByTestId('inspect-artifact-btn');
    fireEvent.click(inspectButtons[0]);

    await waitFor(() => {
      expect(screen.getByTestId('artifact-details-modal')).toBeInTheDocument();
      expect(screen.getByText('adapter_config.json')).toBeInTheDocument();
      expect(screen.getByText('adapter_model.safetensors')).toBeInTheDocument();
    });
  });

  it('triggers artifact validation and updates view status', async () => {
    const mockValResponse: ArtifactValidationResponse = {
      artifact_id: 'art_run_001_lora',
      is_valid: true,
      validation_status: 'valid',
      errors: [],
      warnings: [],
      checksums: {},
      total_size_bytes: 14500000,
      checked_at: '2026-10-28T15:00:00Z',
    };
    vi.mocked(api.validateArtifact).mockResolvedValue(mockValResponse);

    render(<ArtifactsView />);

    await waitFor(() => {
      expect(screen.getByText('art_run_001_lora')).toBeInTheDocument();
    });

    const validateButtons = screen.getAllByTestId('validate-artifact-btn');
    fireEvent.click(validateButtons[0]);

    await waitFor(() => {
      expect(api.validateArtifact).toHaveBeenCalledWith('art_run_001_lora', true);
      expect(screen.getByTestId('artifact-notification')).toBeInTheDocument();
    });
  });

  it('triggers export and handles export package creation', async () => {
    const mockExportResponse: ArtifactExportResponse = {
      artifact_id: 'art_run_001_lora',
      export_status: 'exported',
      package_filename: 'art_run_001_lora.zip',
      package_size_bytes: 14000000,
      sha256: 'deadbeef12345678',
      created_at: '2026-10-28T15:30:00Z',
      download_url: '/api/artifacts/art_run_001_lora/download',
      message: 'Packaged successfully',
    };
    vi.mocked(api.exportArtifact).mockResolvedValue(mockExportResponse);

    render(<ArtifactsView />);

    await waitFor(() => {
      expect(screen.getByText('art_run_001_lora')).toBeInTheDocument();
    });

    const exportButtons = screen.getAllByTestId('export-artifact-btn');
    fireEvent.click(exportButtons[0]);

    await waitFor(() => {
      expect(api.exportArtifact).toHaveBeenCalledWith('art_run_001_lora');
      expect(screen.getByText(/Packaged successfully/i)).toBeInTheDocument();
    });
  });

  it('navigates to evaluation when Evaluate button is clicked', async () => {
    const mockNavigate = vi.fn();
    render(<ArtifactsView onNavigateToEvaluation={mockNavigate} />);

    await waitFor(() => {
      expect(screen.getByText('art_run_001_lora')).toBeInTheDocument();
    });

    const evalButtons = screen.getAllByTestId('evaluate-artifact-btn');
    fireEvent.click(evalButtons[0]);

    expect(mockNavigate).toHaveBeenCalledWith('art_run_001_lora');
  });

  it('displays API error banner when fetching artifacts fails', async () => {
    vi.mocked(api.fetchAllArtifacts).mockRejectedValue(new Error('Network failure: 500'));
    render(<ArtifactsView />);

    await waitFor(() => {
      expect(screen.getByText(/Network failure: 500/i)).toBeInTheDocument();
    });
  });
});
