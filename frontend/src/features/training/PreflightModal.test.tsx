import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { PreflightModal } from './PreflightModal';
import { PreflightCheckResponse } from '../../types/training';

// ─── Mock data ─────────────────────────────────────────────────────────────

const mockPassResult: PreflightCheckResponse = {
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
      message: "Model 'meta-llama/Llama-3.2-1B' is selected.",
      detail: null,
    },
    {
      check_id: 'dataset_selected',
      label: 'Training Dataset',
      status: 'pass',
      message: 'Dataset reference is selected.',
      detail: null,
    },
    {
      check_id: 'device',
      label: 'Compute Device',
      status: 'pass',
      message: 'CUDA GPU detected: NVIDIA GeForce RTX 2050 (4.0 GB VRAM).',
      detail: { device_type: 'cuda', gpu_count: 1, primary_gpu_vram_gb: 4.0 },
    },
    {
      check_id: 'method_device_compat',
      label: 'Method–Device Compatibility',
      status: 'pass',
      message: 'LORA fine-tuning is compatible.',
      detail: null,
    },
    {
      check_id: 'ml_core_deps',
      label: 'Core ML Stack',
      status: 'pass',
      message: 'torch 2.1.0, transformers 4.40.0, peft 0.10.0.',
      detail: null,
    },
    {
      check_id: 'bitsandbytes',
      label: 'bitsandbytes (4-bit)',
      status: 'skip',
      message: 'Skipped — not required for lora.',
      detail: null,
    },
    {
      check_id: 'vram',
      label: 'VRAM Capacity',
      status: 'pass',
      message: 'Estimated ~1.5 GB fits within 4.0 GB.',
      detail: { estimated_gb: 1.5, available_gb: 4.0 },
    },
    {
      check_id: 'hf_token',
      label: 'HF Access Token',
      status: 'warn',
      message: 'No HF token configured.',
      detail: null,
    },
  ],
  can_launch: true,
  blocking_count: 0,
  warning_count: 1,
  method: 'lora',
  estimated_vram_gb: 1.5,
};

const mockBlockedResult: PreflightCheckResponse = {
  checks: [
    {
      check_id: 'config_schema',
      label: 'Configuration Schema',
      status: 'pass',
      message: 'Valid configuration.',
      detail: null,
    },
    {
      check_id: 'model_selected',
      label: 'Base Model',
      status: 'fail',
      message: 'No base model selected.',
      detail: null,
    },
    {
      check_id: 'dataset_selected',
      label: 'Training Dataset',
      status: 'fail',
      message: 'No dataset selected.',
      detail: null,
    },
    {
      check_id: 'device',
      label: 'Compute Device',
      status: 'warn',
      message: 'CPU execution mode. Training will be slow.',
      detail: { device_type: 'cpu' },
    },
    {
      check_id: 'method_device_compat',
      label: 'Method–Device Compatibility',
      status: 'pass',
      message: 'Compatible.',
      detail: null,
    },
    {
      check_id: 'ml_core_deps',
      label: 'Core ML Stack',
      status: 'fail',
      message: 'Missing: torch, transformers.',
      detail: { missing: ['torch', 'transformers'] },
    },
    {
      check_id: 'bitsandbytes',
      label: 'bitsandbytes (4-bit)',
      status: 'skip',
      message: 'Skipped.',
      detail: null,
    },
    {
      check_id: 'vram',
      label: 'VRAM Capacity',
      status: 'skip',
      message: 'No GPU.',
      detail: null,
    },
    {
      check_id: 'hf_token',
      label: 'HF Access Token',
      status: 'warn',
      message: 'No token.',
      detail: null,
    },
  ],
  can_launch: false,
  blocking_count: 3,
  warning_count: 2,
  method: 'lora',
  estimated_vram_gb: null,
};

// ─── Base props ─────────────────────────────────────────────────────────────

const noop = vi.fn();

function makeProps(overrides: Partial<Parameters<typeof PreflightModal>[0]> = {}) {
  return {
    isOpen: true,
    isLoading: false,
    result: null,
    error: null,
    onConfirmLaunch: noop,
    onClose: noop,
    onRerun: noop,
    ...overrides,
  };
}

describe('Phase 6: PreflightModal Component Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // ─── Visibility ───────────────────────────────────────────────────────────

  it('does not render when isOpen is false', () => {
    render(<PreflightModal {...makeProps({ isOpen: false })} />);
    expect(screen.queryByTestId('preflight-modal')).not.toBeInTheDocument();
  });

  it('renders modal when isOpen is true', () => {
    render(<PreflightModal {...makeProps()} />);
    expect(screen.getByTestId('preflight-modal')).toBeInTheDocument();
  });

  it('shows title and subtitle', () => {
    render(<PreflightModal {...makeProps()} />);
    expect(screen.getByText('Pre-Launch Resource Validation')).toBeInTheDocument();
    expect(screen.getByText(/Verifying hardware/)).toBeInTheDocument();
  });

  // ─── Loading state ────────────────────────────────────────────────────────

  it('shows loading spinner when isLoading=true', () => {
    render(<PreflightModal {...makeProps({ isLoading: true })} />);
    expect(screen.getByTestId('preflight-loading')).toBeInTheDocument();
    expect(screen.getByText('Running preflight checks…')).toBeInTheDocument();
  });

  it('hides loading spinner when result is available', () => {
    render(<PreflightModal {...makeProps({ isLoading: false, result: mockPassResult })} />);
    expect(screen.queryByTestId('preflight-loading')).not.toBeInTheDocument();
  });

  // ─── Error state ──────────────────────────────────────────────────────────

  it('renders API error banner with message', () => {
    render(
      <PreflightModal
        {...makeProps({
          error: 'Backend is offline',
          result: null,
          isLoading: false,
        })}
      />,
    );
    expect(screen.getByTestId('preflight-api-error')).toBeInTheDocument();
    expect(screen.getByText('Backend is offline')).toBeInTheDocument();
  });

  // ─── Pass result ──────────────────────────────────────────────────────────

  it('renders summary banner as "ready" when can_launch is true', () => {
    render(<PreflightModal {...makeProps({ result: mockPassResult })} />);
    const banner = screen.getByTestId('preflight-summary-banner');
    expect(banner).toBeInTheDocument();
    expect(screen.getByTestId('preflight-launch-status')).toHaveTextContent(
      'All checks passed — ready to launch',
    );
  });

  it('renders all check rows for a passing result', () => {
    render(<PreflightModal {...makeProps({ result: mockPassResult })} />);
    // Should render one row per check in mockPassResult
    for (const check of mockPassResult.checks) {
      expect(screen.getByTestId(`preflight-check-${check.check_id}`)).toBeInTheDocument();
    }
  });

  it('shows VRAM estimate when available', () => {
    render(<PreflightModal {...makeProps({ result: mockPassResult })} />);
    expect(screen.getByTestId('preflight-vram-estimate')).toBeInTheDocument();
    expect(screen.getByText('~1.5 GB')).toBeInTheDocument();
  });

  it('does not show VRAM estimate when null', () => {
    render(<PreflightModal {...makeProps({ result: mockBlockedResult })} />);
    expect(screen.queryByTestId('preflight-vram-estimate')).not.toBeInTheDocument();
  });

  it('renders correct status badge for each check', () => {
    render(<PreflightModal {...makeProps({ result: mockPassResult })} />);
    // pass check
    expect(screen.getByTestId('check-status-config_schema')).toHaveTextContent('pass');
    // warn check
    expect(screen.getByTestId('check-status-hf_token')).toHaveTextContent('warn');
    // skip check
    expect(screen.getByTestId('check-status-bitsandbytes')).toHaveTextContent('skip');
  });

  // ─── Blocked result ───────────────────────────────────────────────────────

  it('renders summary banner as blocked when can_launch is false', () => {
    render(<PreflightModal {...makeProps({ result: mockBlockedResult })} />);
    expect(screen.getByTestId('preflight-launch-status')).toHaveTextContent(
      '3 blocking issue(s) — cannot launch',
    );
  });

  it('renders fail check rows for blocking issues', () => {
    render(<PreflightModal {...makeProps({ result: mockBlockedResult })} />);
    expect(screen.getByTestId('check-status-model_selected')).toHaveTextContent('fail');
    expect(screen.getByTestId('check-status-dataset_selected')).toHaveTextContent('fail');
    expect(screen.getByTestId('check-status-ml_core_deps')).toHaveTextContent('fail');
  });

  // ─── Confirm Launch button ────────────────────────────────────────────────

  it('enables Confirm & Launch button when can_launch is true', () => {
    const onConfirm = vi.fn();
    render(
      <PreflightModal {...makeProps({ result: mockPassResult, onConfirmLaunch: onConfirm })} />,
    );
    const btn = screen.getByTestId('btn-confirm-launch');
    expect(btn).not.toBeDisabled();
    fireEvent.click(btn);
    expect(onConfirm).toHaveBeenCalledOnce();
  });

  it('disables Confirm & Launch button when can_launch is false', () => {
    render(<PreflightModal {...makeProps({ result: mockBlockedResult })} />);
    const btn = screen.getByTestId('btn-confirm-launch');
    expect(btn).toBeDisabled();
  });

  it('disables Confirm & Launch button during loading', () => {
    render(<PreflightModal {...makeProps({ isLoading: true, result: mockPassResult })} />);
    expect(screen.getByTestId('btn-confirm-launch')).toBeDisabled();
  });

  // ─── Action buttons ───────────────────────────────────────────────────────

  it('calls onClose when Cancel is clicked', () => {
    const onClose = vi.fn();
    render(<PreflightModal {...makeProps({ onClose })} />);
    fireEvent.click(screen.getByTestId('btn-cancel-preflight'));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('calls onClose when X button is clicked', () => {
    const onClose = vi.fn();
    render(<PreflightModal {...makeProps({ onClose })} />);
    fireEvent.click(screen.getByTestId('btn-close-preflight'));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('calls onClose when backdrop is clicked', () => {
    const onClose = vi.fn();
    render(<PreflightModal {...makeProps({ onClose })} />);
    fireEvent.click(screen.getByTestId('preflight-modal-backdrop'));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('calls onRerun when Re-run Checks is clicked', () => {
    const onRerun = vi.fn();
    render(<PreflightModal {...makeProps({ result: mockPassResult, onRerun })} />);
    fireEvent.click(screen.getByTestId('btn-rerun-preflight'));
    expect(onRerun).toHaveBeenCalledOnce();
  });

  it('calls onClose on Escape key press', async () => {
    const onClose = vi.fn();
    render(<PreflightModal {...makeProps({ onClose })} />);
    await act(async () => {
      fireEvent.keyDown(window, { key: 'Escape' });
    });
    expect(onClose).toHaveBeenCalledOnce();
  });

  // ─── Expandable detail ────────────────────────────────────────────────────

  it('expands check detail on "Show details" click', async () => {
    render(<PreflightModal {...makeProps({ result: mockPassResult })} />);
    // device check has detail { device_type, gpu_count, primary_gpu_vram_gb }
    const deviceRow = screen.getByTestId('preflight-check-device');
    const showBtn = deviceRow.querySelector('button');
    expect(showBtn).toBeInTheDocument();
    fireEvent.click(showBtn!);
    await waitFor(() => {
      expect(deviceRow).toHaveTextContent('device_type');
    });
  });
});
