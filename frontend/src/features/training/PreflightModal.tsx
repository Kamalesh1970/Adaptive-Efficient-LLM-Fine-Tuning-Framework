import React, { useEffect, useState } from 'react';
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  SkipForward,
  Loader2,
  Cpu,
  Layers,
  HardDrive,
  Zap,
  Lock,
  Package,
  MemoryStick,
  Play,
  X,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  Sparkles,
} from 'lucide-react';
import { PreflightCheckResult, PreflightCheckResponse, PreflightStatus } from '../../types/training';
import { Badge } from '../../components/common/Badge';

interface PreflightModalProps {
  /** Whether the modal is visible */
  isOpen: boolean;
  /** Whether the preflight check is actively loading */
  isLoading: boolean;
  /** The result from the backend preflight endpoint, or null while loading */
  result: PreflightCheckResponse | null;
  /** Error message if the preflight API call itself failed */
  error: string | null;
  /** Called when the user confirms launch (only enabled when can_launch) */
  onConfirmLaunch: () => void;
  /** Called when the user dismisses / cancels the modal */
  onClose: () => void;
  /** Called when the user requests re-running the preflight checks */
  onRerun: () => void;
}

// ─── Icon map per check_id ─────────────────────────────────────────────────

const CHECK_ICONS: Record<string, React.FC<{ className?: string }>> = {
  config_schema: Sparkles,
  model_selected: Layers,
  model_catalog: Layers,
  dataset_selected: HardDrive,
  dataset_validation: HardDrive,
  device: Cpu,
  method_device_compat: Zap,
  ml_core_deps: Package,
  bitsandbytes: Package,
  vram: MemoryStick,
  hf_token: Lock,
};

const DEFAULT_ICON = CheckCircle2;

// ─── Status indicator ──────────────────────────────────────────────────────

function StatusIcon({ status }: { status: PreflightStatus }) {
  switch (status) {
    case 'pass':
      return <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />;
    case 'fail':
      return <XCircle className="w-4 h-4 text-rose-400 shrink-0" />;
    case 'warn':
      return <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />;
    case 'skip':
      return <SkipForward className="w-4 h-4 text-zinc-500 shrink-0" />;
  }
}

function statusBorderClass(status: PreflightStatus): string {
  switch (status) {
    case 'pass':
      return 'border-emerald-500/20 bg-emerald-950/10';
    case 'fail':
      return 'border-rose-500/30 bg-rose-950/15';
    case 'warn':
      return 'border-amber-500/25 bg-amber-950/10';
    case 'skip':
      return 'border-studio-border/30 bg-studio-input/30';
  }
}

function statusTextClass(status: PreflightStatus): string {
  switch (status) {
    case 'pass':
      return 'text-emerald-400';
    case 'fail':
      return 'text-rose-400';
    case 'warn':
      return 'text-amber-400';
    case 'skip':
      return 'text-zinc-500';
  }
}

// ─── Single check row ──────────────────────────────────────────────────────

function CheckRow({ check }: { check: PreflightCheckResult }) {
  const [expanded, setExpanded] = useState(false);
  const Icon = CHECK_ICONS[check.check_id] ?? DEFAULT_ICON;
  const hasDetail = check.detail && Object.keys(check.detail).length > 0;

  return (
    <div
      className={`rounded-lg border p-3 transition-colors ${statusBorderClass(check.status)}`}
      data-testid={`preflight-check-${check.check_id}`}
    >
      <div className="flex items-start gap-3">
        {/* Check-type icon */}
        <Icon className="w-4 h-4 text-studio-muted shrink-0 mt-0.5" />

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <span
              className="text-[11px] font-semibold text-studio-text font-mono uppercase tracking-wide"
              data-testid={`check-label-${check.check_id}`}
            >
              {check.label}
            </span>
            <div className="flex items-center gap-1.5 shrink-0">
              <StatusIcon status={check.status} />
              <span
                className={`text-[10px] font-bold uppercase font-mono ${statusTextClass(check.status)}`}
                data-testid={`check-status-${check.check_id}`}
              >
                {check.status}
              </span>
            </div>
          </div>
          <p
            className="text-[11px] text-studio-muted mt-1 leading-relaxed"
            data-testid={`check-message-${check.check_id}`}
          >
            {check.message}
          </p>

          {/* Expandable detail */}
          {hasDetail && (
            <button
              type="button"
              onClick={() => setExpanded((prev) => !prev)}
              className="mt-1.5 text-[10px] text-studio-muted hover:text-studio-text inline-flex items-center gap-0.5 transition-colors cursor-pointer"
            >
              {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              {expanded ? 'Hide details' : 'Show details'}
            </button>
          )}
          {expanded && hasDetail && (
            <div className="mt-2 p-2 rounded bg-black/30 border border-studio-border/30 font-mono text-[10px] text-studio-muted space-y-0.5">
              {Object.entries(check.detail!).map(([k, v]) => (
                <div key={k} className="flex gap-2">
                  <span className="text-studio-text/60 shrink-0">{k}:</span>
                  <span className="break-all">{String(v)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Main modal ────────────────────────────────────────────────────────────

export const PreflightModal: React.FC<PreflightModalProps> = ({
  isOpen,
  isLoading,
  result,
  error,
  onConfirmLaunch,
  onClose,
  onRerun,
}) => {
  // Close on Escape
  useEffect(() => {
    if (!isOpen) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const canLaunch = result?.can_launch ?? false;
  const blocking = result?.blocking_count ?? 0;
  const warnings = result?.warning_count ?? 0;
  const estimatedVram = result?.estimated_vram_gb;

  return (
    /* Backdrop */
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      data-testid="preflight-modal"
      role="dialog"
      aria-modal="true"
      aria-labelledby="preflight-modal-title"
    >
      {/* Translucent overlay */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
        data-testid="preflight-modal-backdrop"
      />

      {/* Panel */}
      <div className="relative z-10 w-full max-w-xl max-h-[90vh] flex flex-col bg-studio-card border border-studio-cardBorder rounded-2xl shadow-2xl shadow-black/60 overflow-hidden">

        {/* Header */}
        <div className="px-5 py-4 border-b border-studio-border flex items-start justify-between gap-4 shrink-0">
          <div>
            <h2
              id="preflight-modal-title"
              className="text-sm font-bold text-studio-text flex items-center gap-2"
            >
              <Sparkles className="w-4 h-4 text-studio-accent" />
              Pre-Launch Resource Validation
            </h2>
            <p className="text-[11px] text-studio-muted mt-0.5">
              Verifying hardware, dependencies, and configuration before fine-tuning.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-studio-muted hover:text-studio-text transition-colors p-1 rounded cursor-pointer"
            data-testid="btn-close-preflight"
            aria-label="Close preflight modal"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body — scrollable */}
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-3 min-h-0">

          {/* Loading skeleton */}
          {isLoading && (
            <div className="py-10 flex flex-col items-center gap-4" data-testid="preflight-loading">
              <Loader2 className="w-8 h-8 text-studio-accent animate-spin" />
              <div className="text-center space-y-1">
                <p className="text-sm font-semibold text-studio-text">Running preflight checks…</p>
                <p className="text-xs text-studio-muted">
                  Inspecting hardware, dependencies, and configuration.
                </p>
              </div>
            </div>
          )}

          {/* API error */}
          {!isLoading && error && (
            <div
              className="p-4 rounded-xl bg-rose-950/30 border border-rose-500/30 text-xs text-rose-300 space-y-2"
              data-testid="preflight-api-error"
            >
              <div className="flex items-center gap-2 font-semibold text-rose-400">
                <AlertTriangle className="w-4 h-4" />
                Preflight check failed to run
              </div>
              <p className="font-mono break-words">{error}</p>
              <p className="text-rose-300/70">
                The backend may be offline. Start the FastAPI server and re-run checks.
              </p>
            </div>
          )}

          {/* Summary banner */}
          {!isLoading && result && (
            <>
              <div
                className={`p-3 rounded-xl border flex items-center justify-between gap-4 ${
                  canLaunch
                    ? 'bg-emerald-950/20 border-emerald-500/30'
                    : 'bg-rose-950/20 border-rose-500/30'
                }`}
                data-testid="preflight-summary-banner"
              >
                <div className="flex items-center gap-2">
                  {canLaunch ? (
                    <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                  ) : (
                    <XCircle className="w-5 h-5 text-rose-400" />
                  )}
                  <div>
                    <p
                      className={`text-xs font-bold ${canLaunch ? 'text-emerald-300' : 'text-rose-300'}`}
                      data-testid="preflight-launch-status"
                    >
                      {canLaunch ? 'All checks passed — ready to launch' : `${blocking} blocking issue(s) — cannot launch`}
                    </p>
                    {warnings > 0 && (
                      <p className="text-[10px] text-amber-400 mt-0.5">
                        {warnings} warning(s) noted — review before continuing
                      </p>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-1.5 shrink-0">
                  {canLaunch ? (
                    <Badge variant="success">Ready</Badge>
                  ) : (
                    <Badge variant="error">{blocking} Blocked</Badge>
                  )}
                  {warnings > 0 && <Badge variant="warning">{warnings} Warn</Badge>}
                </div>
              </div>

              {/* VRAM estimate */}
              {estimatedVram !== null && estimatedVram !== undefined && (
                <div
                  className="flex items-center justify-between px-3 py-2 rounded-lg bg-studio-input/60 border border-studio-border text-xs font-mono"
                  data-testid="preflight-vram-estimate"
                >
                  <span className="text-studio-muted flex items-center gap-1.5">
                    <MemoryStick className="w-3.5 h-3.5 text-studio-teal" />
                    Estimated VRAM requirement
                  </span>
                  <span className="font-bold text-studio-text">~{estimatedVram} GB</span>
                </div>
              )}

              {/* Check list */}
              <div className="space-y-2">
                {result.checks.map((check) => (
                  <CheckRow key={check.check_id} check={check} />
                ))}
              </div>
            </>
          )}
        </div>

        {/* Footer actions */}
        <div className="px-5 py-4 border-t border-studio-border shrink-0 flex items-center justify-between gap-3">
          <button
            type="button"
            onClick={onRerun}
            disabled={isLoading}
            className="px-3 py-2 rounded-lg text-xs font-semibold bg-studio-input hover:bg-studio-cardBorder text-studio-muted border border-studio-border inline-flex items-center gap-1.5 transition-colors disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
            data-testid="btn-rerun-preflight"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
            Re-run Checks
          </button>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onClose}
              className="px-3 py-2 rounded-lg text-xs font-semibold bg-studio-input hover:bg-studio-cardBorder text-studio-text border border-studio-border transition-colors cursor-pointer"
              data-testid="btn-cancel-preflight"
            >
              Cancel
            </button>

            <button
              type="button"
              disabled={!canLaunch || isLoading}
              onClick={onConfirmLaunch}
              title={
                !canLaunch
                  ? `${blocking} blocking issue(s) must be resolved before launching`
                  : 'Launch real LoRA / QLoRA fine-tuning'
              }
              className={`px-4 py-2 rounded-lg text-xs font-bold inline-flex items-center gap-1.5 transition-all shadow-md ${
                canLaunch && !isLoading
                  ? 'bg-studio-accent hover:bg-studio-accentHover text-studio-bg shadow-studio-accent/20 cursor-pointer'
                  : 'bg-studio-cardBorder/60 text-studio-muted border border-studio-border/60 cursor-not-allowed opacity-60'
              }`}
              data-testid="btn-confirm-launch"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              Confirm &amp; Launch
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
