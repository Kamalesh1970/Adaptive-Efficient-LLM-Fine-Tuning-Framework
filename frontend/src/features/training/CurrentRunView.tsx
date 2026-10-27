import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Activity,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Clock,
  Cpu,
  Layers,
  StopCircle,
  FileCheck,
  ArrowLeft,
  RefreshCw,
  Terminal,
} from 'lucide-react';
import { Badge } from '../../components/common/Badge';
import {
  TrainingRunResponse,
  TrainingArtifactMetadata,
} from '../../types/training';
import {
  fetchTrainingRun,
  cancelTrainingRun,
  fetchRunArtifacts,
} from '../../services/api';

interface CurrentRunViewProps {
  activeRunId: string | null;
  onNavigateToConfigure?: () => void;
}

export const CurrentRunView: React.FC<CurrentRunViewProps> = ({
  activeRunId,
  onNavigateToConfigure,
}) => {
  const [run, setRun] = useState<TrainingRunResponse | null>(null);
  const [artifacts, setArtifacts] = useState<TrainingArtifactMetadata | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isCancelling, setIsCancelling] = useState(false);
  const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const isTerminalStatus = (status?: string) => {
    return status === 'completed' || status === 'failed' || status === 'cancelled';
  };

  const loadRunDetails = useCallback(async (runId: string) => {
    try {
      const runData = await fetchTrainingRun(runId);
      setRun(runData);
      setError(null);

      // If run just completed, fetch verified artifacts
      if (runData.status === 'completed') {
        try {
          const artifactData = await fetchRunArtifacts(runId);
          setArtifacts(artifactData);
        } catch {
          // Artifacts might be finishing flush
        }
      }
      return runData;
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to retrieve run telemetry';
      setError(msg);
      return null;
    }
  }, []);

  // Polling loop for active runs
  useEffect(() => {
    if (!activeRunId) {
      setRun(null);
      setArtifacts(null);
      return;
    }

    let isSubscribed = true;
    setIsLoading(true);

    const poll = async () => {
      const runData = await loadRunDetails(activeRunId);
      if (isSubscribed) {
        setIsLoading(false);
      }

      if (isSubscribed && runData && !isTerminalStatus(runData.status)) {
        pollTimerRef.current = setTimeout(poll, 1500);
      }
    };

    poll();

    return () => {
      isSubscribed = false;
      if (pollTimerRef.current) {
        clearTimeout(pollTimerRef.current);
      }
    };
  }, [activeRunId, loadRunDetails]);

  const handleCancel = async () => {
    if (!run || !run.cancellable || isCancelling) return;
    setIsCancelling(true);
    try {
      const updated = await cancelTrainingRun(run.run_id);
      setRun(updated);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Cancellation request failed';
      setError(msg);
    } finally {
      setIsCancelling(false);
    }
  };

  // 1. Placeholder state when no run is active
  if (!activeRunId) {
    return (
      <div
        className="flex-1 flex items-center justify-center p-8 text-center"
        data-testid="current-run-placeholder"
      >
        <div className="max-w-md space-y-4 p-8 rounded-xl bg-studio-card border border-studio-cardBorder shadow-xl">
          <Activity className="w-10 h-10 text-studio-accent mx-auto" />
          <h2 className="text-base font-bold text-studio-text">No Active Training Run</h2>
          <p className="text-xs text-studio-muted leading-relaxed">
            Configure your model, dataset, and training parameters in the Configure tab, then click &quot;Start Fine-Tuning&quot; to initiate a real LoRA training run.
          </p>
          {onNavigateToConfigure && (
            <button
              type="button"
              onClick={onNavigateToConfigure}
              className="px-4 py-2 bg-studio-accent hover:bg-studio-accentHover text-studio-bg rounded-lg text-xs font-semibold inline-flex items-center gap-1.5 transition-colors cursor-pointer"
              data-testid="btn-go-to-configure"
            >
              <ArrowLeft className="w-3.5 h-3.5" /> Go to Configuration
            </button>
          )}
        </div>
      </div>
    );
  }

  // 2. Active run monitor view
  const progressPercent =
    run && run.max_steps > 0
      ? Math.min(100, Math.round((run.current_step / run.max_steps) * 100))
      : 0;

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return <Badge variant="success">Completed</Badge>;
      case 'training':
        return <Badge variant="default">Training In Progress</Badge>;
      case 'saving':
        return <Badge variant="default">Saving Adapter</Badge>;
      case 'loading_model':
        return <Badge variant="default">Loading Model</Badge>;
      case 'preparing_dataset':
        return <Badge variant="default">Tokenizing Dataset</Badge>;
      case 'validating':
        return <Badge variant="default">Validating Job</Badge>;
      case 'queued':
        return <Badge variant="default">Queued</Badge>;
      case 'failed':
        return <Badge variant="error">Failed</Badge>;
      case 'cancelled':
        return <Badge variant="default">Cancelled</Badge>;
      default:
        return <Badge variant="default">{status}</Badge>;
    }
  };

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6 max-w-6xl mx-auto w-full" data-testid="current-run-monitor">
      {/* Top Banner Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-studio-cardBorder gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-bold text-studio-text flex items-center gap-2">
              <Activity className="w-5 h-5 text-studio-accent" />
              Fine-Tuning Run Monitor
            </h1>
            {run && getStatusBadge(run.status)}
          </div>
          <div className="text-xs text-studio-muted font-mono mt-1 flex items-center gap-2">
            <span>Run ID:</span>
            <span className="text-studio-text font-semibold" data-testid="run-id-display">
              {run ? run.run_id : activeRunId}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {run && run.cancellable && (
            <button
              type="button"
              disabled={isCancelling}
              onClick={handleCancel}
              className="px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-rose-500/20 hover:bg-rose-500/30 text-rose-400 border border-rose-500/30 inline-flex items-center gap-1.5 transition-colors cursor-pointer"
              data-testid="btn-cancel-run"
            >
              <StopCircle className="w-3.5 h-3.5" />
              <span>{isCancelling ? 'Cancelling...' : 'Cancel Run'}</span>
            </button>
          )}

          {onNavigateToConfigure && (
            <button
              type="button"
              onClick={onNavigateToConfigure}
              className="px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-studio-card hover:bg-studio-cardBorder text-studio-text border border-studio-border inline-flex items-center gap-1.5 transition-colors cursor-pointer"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Back to Config</span>
            </button>
          )}
        </div>
      </div>

      {/* Global Error Banner */}
      {error && (
        <div className="p-3 bg-rose-950/30 border border-rose-500/30 rounded-xl text-xs text-rose-300 flex items-center gap-2" data-testid="run-api-error">
          <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Failure Message */}
      {run && run.status === 'failed' && run.error_message && (
        <div className="p-4 bg-rose-950/40 border border-rose-500/40 rounded-xl text-xs text-rose-200 space-y-1.5" data-testid="run-failure-banner">
          <div className="font-bold flex items-center gap-1.5 text-rose-400">
            <XCircle className="w-4 h-4" /> Training Execution Failed
          </div>
          <p className="font-mono text-[11px] leading-relaxed break-words bg-black/40 p-2.5 rounded border border-rose-900/50">
            {run.error_message}
          </p>
        </div>
      )}

      {/* Progress & Loss Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* Step Progress Card */}
        <div className="p-4 rounded-xl bg-studio-card border border-studio-cardBorder shadow space-y-2">
          <div className="text-[11px] uppercase tracking-wider font-mono text-studio-muted flex items-center justify-between">
            <span>Step Progress</span>
            <span className="font-semibold text-studio-text">{progressPercent}%</span>
          </div>
          <div className="text-xl font-bold font-mono text-studio-text" data-testid="step-count-display">
            {run ? `${run.current_step} / ${run.max_steps}` : '—'}
          </div>
          <div className="w-full bg-studio-input rounded-full h-1.5 overflow-hidden border border-studio-border/30">
            <div
              className="bg-studio-accent h-full transition-all duration-300"
              style={{ width: `${progressPercent}%` }}
              data-testid="progress-bar-fill"
            />
          </div>
        </div>

        {/* Current Loss Card */}
        <div className="p-4 rounded-xl bg-studio-card border border-studio-cardBorder shadow space-y-2">
          <div className="text-[11px] uppercase tracking-wider font-mono text-studio-muted flex items-center justify-between">
            <span>Training Loss</span>
            <Activity className="w-3.5 h-3.5 text-studio-accent" />
          </div>
          <div className="text-xl font-bold font-mono text-studio-text" data-testid="current-loss-display">
            {run && run.current_loss !== null && run.current_loss !== undefined
              ? run.current_loss.toFixed(4)
              : '—'}
          </div>
          <p className="text-[10px] text-studio-muted">
            {run?.loss_history && run.loss_history.length > 0
              ? `${run.loss_history.length} checkpoint metric(s)`
              : 'Awaiting loss calculations'}
          </p>
        </div>

        {/* Elapsed Time Card */}
        <div className="p-4 rounded-xl bg-studio-card border border-studio-cardBorder shadow space-y-2">
          <div className="text-[11px] uppercase tracking-wider font-mono text-studio-muted flex items-center justify-between">
            <span>Elapsed Time</span>
            <Clock className="w-3.5 h-3.5 text-studio-teal" />
          </div>
          <div className="text-xl font-bold font-mono text-studio-text" data-testid="elapsed-time-display">
            {run ? `${run.elapsed_time_seconds.toFixed(1)}s` : '0.0s'}
          </div>
          <p className="text-[10px] text-studio-muted font-mono">
            {run?.status ? `State: ${run.status}` : 'Pending'}
          </p>
        </div>

        {/* Execution Device & Parameters */}
        <div className="p-4 rounded-xl bg-studio-card border border-studio-cardBorder shadow space-y-2">
          <div className="text-[11px] uppercase tracking-wider font-mono text-studio-muted flex items-center justify-between">
            <span>Execution Device</span>
            <Cpu className="w-3.5 h-3.5 text-studio-accent" />
          </div>
          <div className="text-xl font-bold font-mono text-studio-text capitalize" data-testid="device-display">
            {run ? run.device : 'cpu'}
          </div>
          <p className="text-[10px] text-studio-muted font-mono truncate" title={run?.config.model_identifier || ''}>
            Model: {run?.config.model_identifier || 'None'}
          </p>
        </div>
      </div>

      {/* Trainable Parameters Telemetry */}
      {run && run.trainable_params !== null && run.trainable_params !== undefined && (
        <div className="p-4 rounded-xl bg-studio-input/80 border border-studio-border text-xs flex flex-wrap items-center justify-between gap-4 font-mono">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-studio-accent" />
            <span className="text-studio-muted">Trainable Adapter Parameters:</span>
            <span className="text-studio-text font-bold" data-testid="trainable-params-count">
              {run.trainable_params.toLocaleString()}
            </span>
            <span className="text-emerald-400 font-semibold" data-testid="trainable-percent">
              ({run.trainable_percent?.toFixed(3)}%)
            </span>
          </div>
          <div className="text-studio-muted text-[11px]">
            Total Base Model Parameters: <span className="text-studio-text">{run.total_params?.toLocaleString()}</span>
          </div>
        </div>
      )}

      {/* Loss History Telemetry Stream */}
      <div className="p-4 rounded-xl bg-studio-card border border-studio-cardBorder shadow space-y-3">
        <div className="flex items-center justify-between pb-2 border-b border-studio-cardBorder">
          <h3 className="text-xs font-bold text-studio-text uppercase tracking-wider font-mono flex items-center gap-1.5">
            <Terminal className="w-3.5 h-3.5 text-studio-teal" />
            Real Loss Telemetry Stream
          </h3>
          {isLoading && (
            <span className="text-[10px] text-studio-muted font-mono flex items-center gap-1">
              <RefreshCw className="w-3 h-3 animate-spin" /> Live Updates
            </span>
          )}
        </div>

        {run && run.loss_history && run.loss_history.length > 0 ? (
          <div className="max-h-60 overflow-y-auto font-mono text-xs border border-studio-border/40 rounded-lg">
            <table className="w-full text-left border-collapse" data-testid="loss-history-table">
              <thead>
                <tr className="bg-studio-bg/60 text-studio-muted text-[10px] uppercase border-b border-studio-border">
                  <th className="py-2 px-3">Step</th>
                  <th className="py-2 px-3">Loss</th>
                  <th className="py-2 px-3">Learning Rate</th>
                  <th className="py-2 px-3">Epoch</th>
                  <th className="py-2 px-3">Timestamp</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-studio-border/20 text-[11px]">
                {run.loss_history.map((pt) => (
                  <tr key={pt.step} className="hover:bg-studio-input/50 transition-colors">
                    <td className="py-1.5 px-3 font-semibold text-studio-text">{pt.step}</td>
                    <td className="py-1.5 px-3 text-studio-accent font-bold">{pt.loss.toFixed(4)}</td>
                    <td className="py-1.5 px-3 text-studio-muted">{pt.learning_rate ?? run.config.learning_rate}</td>
                    <td className="py-1.5 px-3 text-studio-muted">{pt.epoch ?? '—'}</td>
                    <td className="py-1.5 px-3 text-studio-muted text-[10px]">{pt.timestamp}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="py-8 text-center text-xs text-studio-muted font-mono">
            No telemetry points recorded yet. Awaiting training steps...
          </div>
        )}
      </div>

      {/* Validated Artifacts Panel (Only upon completion) */}
      {run && run.status === 'completed' && artifacts && (
        <div
          className="p-5 rounded-xl bg-emerald-950/20 border border-emerald-500/30 shadow space-y-4"
          data-testid="artifacts-panel"
        >
          <div className="flex items-center justify-between pb-3 border-b border-emerald-500/20">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              <h3 className="text-sm font-bold text-studio-text">
                LoRA Adapters Saved and Verified Successfully
              </h3>
            </div>
            <Badge variant="success">Verified On Disk</Badge>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
            <div className="space-y-1">
              <span className="text-studio-muted text-[11px]">Relative Artifact Directory:</span>
              <div className="p-2 bg-black/40 rounded border border-studio-border text-emerald-300 break-all" data-testid="artifact-directory-path">
                {artifacts.artifact_directory}
              </div>
            </div>
            <div className="space-y-1">
              <span className="text-studio-muted text-[11px]">Verified Files:</span>
              <ul className="p-2 bg-black/40 rounded border border-studio-border text-studio-text space-y-0.5 text-[11px]" data-testid="artifact-files-list">
                {artifacts.files.map((f) => (
                  <li key={f} className="flex items-center gap-1.5">
                    <FileCheck className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                    <span>{f}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
