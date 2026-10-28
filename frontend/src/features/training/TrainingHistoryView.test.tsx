import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { TrainingHistoryView } from './TrainingHistoryView';
import * as api from '../../services/api';
import { RunHistoryListResponse } from '../../types/training';

vi.mock('../../services/api');

const mockHistoryResponse: RunHistoryListResponse = {
  runs: [
    {
      run_id: 'run_phase7_001',
      project_name: 'llama3-fine-tune',
      model_identifier: 'unsloth/llama-3-8b-bnb-4bit',
      dataset_reference: 'dataset_alpaca_clean',
      training_method: 'qlora',
      status: 'completed',
      created_at: '2026-10-28T10:00:00Z',
      started_at: '2026-10-28T10:00:05Z',
      completed_at: '2026-10-28T10:05:00Z',
      elapsed_time_seconds: 295.4,
      current_step: 100,
      max_steps: 100,
      current_epoch: 3.0,
      num_epochs: 3.0,
      current_loss: 0.8421,
      learning_rate: 0.0002,
      error_message: null,
      device: 'cuda:0',
      latest_checkpoint: 'final_adapter',
    },
    {
      run_id: 'run_phase7_002',
      project_name: 'tiny-test-run',
      model_identifier: 'test/tiny-llama',
      dataset_reference: 'dataset_quick_test',
      training_method: 'lora',
      status: 'failed',
      created_at: '2026-10-28T09:00:00Z',
      started_at: '2026-10-28T09:00:02Z',
      completed_at: '2026-10-28T09:00:15Z',
      elapsed_time_seconds: 13.2,
      current_step: 5,
      max_steps: 50,
      current_epoch: 0.1,
      num_epochs: 3.0,
      current_loss: null,
      learning_rate: 0.0002,
      error_message: 'CUDA OOM error',
      device: 'cuda:0',
      latest_checkpoint: null,
    },
  ],
  total: 2,
  page: 1,
  page_size: 10,
  total_pages: 1,
};

describe('Phase 7: TrainingHistoryView Component Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders history table with run records, duration, loss, and status badges', async () => {
    vi.mocked(api.fetchRunHistory).mockResolvedValue(mockHistoryResponse);
    const onSelectRun = vi.fn();

    await act(async () => {
      render(<TrainingHistoryView onSelectRun={onSelectRun} />);
    });

    await waitFor(() => {
      expect(screen.getByTestId('training-history-view')).toBeInTheDocument();
      expect(screen.getByTestId('history-table')).toBeInTheDocument();
    });

    expect(screen.getByText('run_phase7_001')).toBeInTheDocument();
    expect(screen.getByText('run_phase7_002')).toBeInTheDocument();
    expect(screen.getByText('llama3-fine-tune')).toBeDefined();
    expect(screen.getByText('0.8421')).toBeInTheDocument();
    expect(screen.getAllByText('Completed').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Failed').length).toBeGreaterThan(0);
  });

  it('navigates to run monitor when inspect button is clicked', async () => {
    vi.mocked(api.fetchRunHistory).mockResolvedValue(mockHistoryResponse);
    const onSelectRun = vi.fn();

    await act(async () => {
      render(<TrainingHistoryView onSelectRun={onSelectRun} />);
    });

    await waitFor(() => {
      expect(screen.getByTestId('btn-inspect-run_phase7_001')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByTestId('btn-inspect-run_phase7_001'));
    expect(onSelectRun).toHaveBeenCalledWith('run_phase7_001');
  });

  it('filters history runs by status when dropdown value changes', async () => {
    vi.mocked(api.fetchRunHistory).mockResolvedValue(mockHistoryResponse);
    const onSelectRun = vi.fn();

    await act(async () => {
      render(<TrainingHistoryView onSelectRun={onSelectRun} />);
    });

    await waitFor(() => {
      expect(screen.getByTestId('history-status-filter')).toBeInTheDocument();
    });

    const select = screen.getByTestId('history-status-filter');
    await act(async () => {
      fireEvent.change(select, { target: { value: 'completed' } });
    });

    expect(api.fetchRunHistory).toHaveBeenCalledWith(1, 10, 'completed');
  });

  it('handles empty history state gracefully', async () => {
    vi.mocked(api.fetchRunHistory).mockResolvedValue({
      runs: [],
      total: 0,
      page: 1,
      page_size: 10,
      total_pages: 1,
    });

    await act(async () => {
      render(<TrainingHistoryView onSelectRun={vi.fn()} />);
    });

    await waitFor(() => {
      expect(screen.getByTestId('history-empty')).toBeInTheDocument();
    });

    expect(screen.getByText('No Training Runs Found')).toBeInTheDocument();
  });

  it('handles API errors with banner notification', async () => {
    vi.mocked(api.fetchRunHistory).mockRejectedValue(new Error('Network error loading history'));

    await act(async () => {
      render(<TrainingHistoryView onSelectRun={vi.fn()} />);
    });

    await waitFor(() => {
      expect(screen.getByTestId('history-error-banner')).toBeInTheDocument();
    });

    expect(screen.getByText('Network error loading history')).toBeInTheDocument();
  });

  it('refreshes history when refresh button is clicked', async () => {
    vi.mocked(api.fetchRunHistory).mockResolvedValue(mockHistoryResponse);

    await act(async () => {
      render(<TrainingHistoryView onSelectRun={vi.fn()} />);
    });

    await waitFor(() => {
      expect(screen.getByTestId('history-refresh-btn')).toBeInTheDocument();
    });

    await act(async () => {
      fireEvent.click(screen.getByTestId('history-refresh-btn'));
    });

    expect(api.fetchRunHistory).toHaveBeenCalledTimes(2);
  });
});
