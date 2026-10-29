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
  List,
  ScrollText,
  HardDrive,
  Gauge,
  Box,
  BarChart2,
} from 'lucide-react';
import { Badge } from '../../components/common/Badge';
import {
  TrainingRunResponse,
  TrainingArtifactMetadata,
  TrainingEvent,
  TrainingLogEntry,
  CheckpointMetadata,
} from '../../types/training';
import {
  fetchTrainingRun,
  cancelTrainingRun,
  fetchRunArtifacts,
  fetchRunEvents,
  fetchRunLogs,
  fetchRunCheckpoints,
} from '../../services/api';

interface CurrentRunViewProps {
  activeRunId: string | null;
  onNavigateToConfigure?: () => void;
  onNavigateToArtifacts?: () => void;
  onNavigateToEvaluation?: (runId?: string) => void;
}

type ActiveMonitorTab = 'telemetry' | 'events' | 'logs' | 'checkpoints';

export const CurrentRunView: React.FC<CurrentRunViewProps> = ({
  activeRunId,
  onNavigateToConfigure,
  onNavigateToArtifacts,
  onNavigateToEvaluation,
}) => {
  const [run, setRun] = useState<TrainingRunResponse | null>(null);
  const [artifacts, setArtifacts] = useState<TrainingArtifactMetadata | null>(null);
  const [events, setEvents] = useState<TrainingEvent[]>([]);
  const [logs, setLogs] = useState<TrainingLogEntry[]>([]);
  const [checkpoints, setCheckpoints] = useState<CheckpointMetadata[]>([]);
  const [activeTab, setActiveTab] = useState<ActiveMonitorTab>('telemetry');

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isCancelling, setIsCancelling] = useState(false);
  const [autoScrollLogs, setAutoScrollLogs] = useState(true);

  const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const logsEndRef = useRef<HTMLDivElement | null>(null);

  const isTerminalStatus = (status?: string) => {
    return status === 'completed' || status === 'failed' || status === 'cancelled';
  };

  const loadRunDetails = useCallback(async (runId: string) => {
    try {
      const runData = await fetchTrainingRun(runId);
      setRun(runData);
      setError(null);

      // If run completed, fetch verified artifacts
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

  const loadEvents = useCallback(async (runId: string) => {
    try {
      const data = await fetchRunEvents(runId, undefined, 200);
      setEvents(data.events);
    } catch {
      // Non-fatal if events endpoint is temporarily pending
    }
  }, []);

  const loadLogs = useCallback(async (runId: string) => {
    try {
      const data = await fetchRunLogs(runId, 500);
      setLogs(data.logs);
    } catch {
      // Non-fatal if logs endpoint is temporarily pending
    }
  }, []);

  const loadCheckpoints = useCallback(async (runId: string) => {
    try {
      const data = await fetchRunCheckpoints(runId);
      setCheckpoints(data.checkpoints);
    } catch {
      // Non-fatal
    }
  }, []);

  // Polling loop for active runs and associated sub-resources
  useEffect(() => {
    if (!activeRunId) {
      setRun(null);
      setArtifacts(null);
      setEvents([]);
      setLogs([]);
      setCheckpoints([]);
      return;
    }

    let isSubscribed = true;
    setIsLoading(true);

    const poll = async () => {
      const runData = await loadRunDetails(activeRunId);
      if (isSubscribed) {
        setIsLoading(false);
      }

      // Fetch supplementary sub-resources
      if (isSubscribed) {
        loadEvents(activeRunId);
        loadLogs(activeRunId);
        loadCheckpoints(activeRunId);
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
  }, [activeRunId, loadRunDetails, loadEvents, loadLogs, loadCheckpoints]);

  // Auto-scroll logs container when new logs arrive if enabled
  useEffect(() => {
    if (autoScrollLogs && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScrollLogs]);

  const handleCancel = async () => {
    if (!run || !run.cancellable || isCancelling) return;
    setIsCancelling(true);
    try {
      const updated = await cancelTrainingRun(run.run_id);
      setRun(updated);
      await loadEvents(run.run_id);
      await loadLogs(run.run_id);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Cancellation request failed';
      setError(msg);
    } finally {
      setIsCancelling(false);
    }
  };

  // Speed and ETA calculation with strict no-fabrication rule
  const calculateSpeedAndEta = () => {
    if (
      !run ||
      !run.elapsed_time_seconds ||
      run.elapsed_time_seconds <= 0 ||
      !run.current_step ||
      run.current_step <= 0
    ) {
      return {
        speed: 'Unavailable',
        eta: 'Unavailable',
      };
    }
    const speedVal = run.current_step / run.elapsed_time_seconds;
    const speedStr = `${speedVal.toFixed(2)} steps/s`;

    const remainingSteps = run.max_steps - run.current_step;
    if (remainingSteps <= 0) {
      return {
        speed: speedStr,
        eta: '0s',
      };
    }
    const etaSeconds = remainingSteps / speedVal;
    let etaStr = '';
    if (etaSeconds < 60) {
      etaStr = `${Math.round(etaSeconds)}s`;
    } else {
      const mins = Math.floor(etaSeconds / 60);
      const secs = Math.round(etaSeconds % 60);
      etaStr = `${mins}m ${secs}s`;
    }
    return {
      speed: speedStr,
      eta: etaStr,
    };
  };

  const formatFileSize = (bytes: number) => {
    if (bytes <= 0) return '0 B';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
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
            Configure your model, dataset, and training parameters in the Configure tab, then click &quot;Start Fine-Tuning&quot; to initiate a real LoRA or QLoRA training run.
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

  const { speed, eta } = calculateSpeedAndEta();

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

  const getSeverityBadge = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'error':
        return <Badge variant="error">Error</Badge>;
      case 'warning':
        return <Badge variant="warning">Warning</Badge>;
      default:
        return <Badge variant="default">Info</Badge>;
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
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-4">
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
              ? `${run.loss_history.length} telemetry metric(s)`
              : 'Awaiting loss calculations'}
          </p>
        </div>

        {/* Speed & ETA Card (Real-time telemetry, never fabricated) */}
        <div className="p-4 rounded-xl bg-studio-card border border-studio-cardBorder shadow space-y-2">
          <div className="text-[11px] uppercase tracking-wider font-mono text-studio-muted flex items-center justify-between">
            <span>Speed & ETA</span>
            <Gauge className="w-3.5 h-3.5 text-studio-teal" />
          </div>
          <div className="text-base font-bold font-mono text-studio-text truncate" data-testid="speed-display">
            {speed}
          </div>
          <p className="text-[10px] text-studio-muted font-mono" data-testid="eta-display">
            ETA: <span className="text-studio-text font-semibold">{eta}</span>
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

        {/* Execution Device */}
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

      {/* Quantization Configuration Telemetry */}
      {run && run.quantization_info && (
        <div className="p-3.5 rounded-xl bg-studio-input/60 border border-studio-border text-xs flex flex-wrap items-center justify-between gap-3 font-mono" data-testid="run-quantization-info">
          <div className="flex items-center gap-2">
            <Badge variant="info">QLoRA Quantized</Badge>
            <span className="text-studio-muted">Quantization Format:</span>
            <span className="text-studio-text font-semibold uppercase">
              {String(run.quantization_info['bnb_4bit_quant_type'] || 'NF4').toUpperCase()} (4-Bit)
            </span>
            <span className="text-studio-muted">• Double Quant:</span>
            <span className={run.quantization_info['bnb_4bit_use_double_quant'] ? 'text-emerald-400' : 'text-zinc-500'}>
              {run.quantization_info['bnb_4bit_use_double_quant'] ? 'Active' : 'Off'}
            </span>
          </div>
          <div className="text-studio-muted text-[11px]">
            Compute Dtype: <span className="text-studio-text uppercase font-semibold">{String(run.quantization_info['bnb_4bit_compute_dtype'] || 'FP16').toUpperCase()}</span>
          </div>
        </div>
      )}

      {/* Detailed Monitoring Navigation Tabs */}
      <div className="flex items-center gap-2 border-b border-studio-cardBorder pb-2">
        <button
          type="button"
          onClick={() => setActiveTab('telemetry')}
          className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold flex items-center gap-1.5 transition-colors cursor-pointer ${
            activeTab === 'telemetry'
              ? 'bg-studio-accent/20 text-studio-accent border border-studio-accent/40'
              : 'text-studio-muted hover:text-studio-text hover:bg-studio-card border border-transparent'
          }`}
          data-testid="tab-telemetry"
        >
          <Terminal className="w-3.5 h-3.5" />
          <span>Loss Stream ({run?.loss_history?.length || 0})</span>
        </button>

        <button
          type="button"
          onClick={() => {
            setActiveTab('events');
            if (activeRunId) loadEvents(activeRunId);
          }}
          className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold flex items-center gap-1.5 transition-colors cursor-pointer ${
            activeTab === 'events'
              ? 'bg-studio-accent/20 text-studio-accent border border-studio-accent/40'
              : 'text-studio-muted hover:text-studio-text hover:bg-studio-card border border-transparent'
          }`}
          data-testid="tab-events"
        >
          <List className="w-3.5 h-3.5" />
          <span>Events ({events.length})</span>
        </button>

        <button
          type="button"
          onClick={() => {
            setActiveTab('logs');
            if (activeRunId) loadLogs(activeRunId);
          }}
          className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold flex items-center gap-1.5 transition-colors cursor-pointer ${
            activeTab === 'logs'
              ? 'bg-studio-accent/20 text-studio-accent border border-studio-accent/40'
              : 'text-studio-muted hover:text-studio-text hover:bg-studio-card border border-transparent'
          }`}
          data-testid="tab-logs"
        >
          <ScrollText className="w-3.5 h-3.5" />
          <span>Worker Logs ({logs.length})</span>
        </button>

        <button
          type="button"
          onClick={() => {
            setActiveTab('checkpoints');
            if (activeRunId) loadCheckpoints(activeRunId);
          }}
          className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold flex items-center gap-1.5 transition-colors cursor-pointer ${
            activeTab === 'checkpoints'
              ? 'bg-studio-accent/20 text-studio-accent border border-studio-accent/40'
              : 'text-studio-muted hover:text-studio-text hover:bg-studio-card border border-transparent'
          }`}
          data-testid="tab-checkpoints"
        >
          <HardDrive className="w-3.5 h-3.5" />
          <span>Checkpoints ({checkpoints.length})</span>
        </button>
      </div>

      {/* Tab 1: Loss History Telemetry Stream */}
      {activeTab === 'telemetry' && (
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
      )}

      {/* Tab 2: Lifecycle Events */}
      {activeTab === 'events' && (
        <div className="p-4 rounded-xl bg-studio-card border border-studio-cardBorder shadow space-y-3" data-testid="events-panel">
          <div className="flex items-center justify-between pb-2 border-b border-studio-cardBorder">
            <h3 className="text-xs font-bold text-studio-text uppercase tracking-wider font-mono flex items-center gap-1.5">
              <List className="w-3.5 h-3.5 text-studio-teal" />
              Standardized Training Lifecycle Events
            </h3>
            <button
              type="button"
              onClick={() => activeRunId && loadEvents(activeRunId)}
              className="px-2.5 py-1 rounded bg-studio-input hover:bg-studio-border text-studio-text text-[11px] font-mono inline-flex items-center gap-1 cursor-pointer"
              data-testid="btn-refresh-events"
            >
              <RefreshCw className="w-3 h-3" /> Refresh Events
            </button>
          </div>

          {events.length > 0 ? (
            <div className="max-h-80 overflow-y-auto font-mono text-xs border border-studio-border/40 rounded-lg">
              <table className="w-full text-left border-collapse" data-testid="events-table">
                <thead>
                  <tr className="bg-studio-bg/60 text-studio-muted text-[10px] uppercase border-b border-studio-border">
                    <th className="py-2 px-3">Timestamp</th>
                    <th className="py-2 px-3">Severity</th>
                    <th className="py-2 px-3">Event Type</th>
                    <th className="py-2 px-3">Message</th>
                    <th className="py-2 px-3">Step</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-studio-border/20 text-[11px]">
                  {events.map((evt) => (
                    <tr key={evt.event_id} className="hover:bg-studio-input/50 transition-colors">
                      <td className="py-1.5 px-3 text-studio-muted text-[10px] whitespace-nowrap">{evt.timestamp}</td>
                      <td className="py-1.5 px-3">{getSeverityBadge(evt.severity)}</td>
                      <td className="py-1.5 px-3 text-studio-text font-semibold uppercase text-[10px]">{evt.event_type}</td>
                      <td className="py-1.5 px-3 text-studio-text">{evt.message}</td>
                      <td className="py-1.5 px-3 text-studio-muted">{evt.global_step ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="py-8 text-center text-xs text-studio-muted font-mono">
              No lifecycle events recorded yet for this run.
            </div>
          )}
        </div>
      )}

      {/* Tab 3: Sanitized Worker Logs */}
      {activeTab === 'logs' && (
        <div className="p-4 rounded-xl bg-studio-card border border-studio-cardBorder shadow space-y-3" data-testid="logs-panel">
          <div className="flex items-center justify-between pb-2 border-b border-studio-cardBorder">
            <h3 className="text-xs font-bold text-studio-text uppercase tracking-wider font-mono flex items-center gap-1.5">
              <ScrollText className="w-3.5 h-3.5 text-studio-teal" />
              Sanitized Worker Logs
            </h3>
            <div className="flex items-center gap-3">
              <label className="text-[11px] font-mono text-studio-muted flex items-center gap-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={autoScrollLogs}
                  onChange={(e) => setAutoScrollLogs(e.target.checked)}
                  className="rounded border-studio-border bg-studio-input cursor-pointer"
                />
                Auto-scroll
              </label>
              <button
                type="button"
                onClick={() => activeRunId && loadLogs(activeRunId)}
                className="px-2.5 py-1 rounded bg-studio-input hover:bg-studio-border text-studio-text text-[11px] font-mono inline-flex items-center gap-1 cursor-pointer"
                data-testid="btn-refresh-logs"
              >
                <RefreshCw className="w-3 h-3" /> Refresh Logs
              </button>
            </div>
          </div>

          {logs.length > 0 ? (
            <div
              className="max-h-80 overflow-y-auto p-3 bg-black/60 rounded-lg border border-studio-border/60 font-mono text-[11px] text-zinc-300 space-y-1"
              data-testid="logs-container"
            >
              {logs.map((log, idx) => (
                <div key={idx} className="leading-relaxed hover:bg-white/5 px-1 rounded flex items-start gap-2">
                  <span className="text-zinc-500 text-[10px] shrink-0">{log.timestamp}</span>
                  <span
                    className={`font-semibold shrink-0 ${
                      log.level === 'ERROR'
                        ? 'text-rose-400'
                        : log.level === 'WARNING'
                        ? 'text-amber-400'
                        : 'text-emerald-400'
                    }`}
                  >
                    [{log.level}]
                  </span>
                  {log.stage && <span className="text-zinc-400 shrink-0">({log.stage})</span>}
                  <span className="break-all">{log.message}</span>
                </div>
              ))}
              <div ref={logsEndRef} />
            </div>
          ) : (
            <div className="py-8 text-center text-xs text-studio-muted font-mono">
              No worker logs recorded yet.
            </div>
          )}
        </div>
      )}

      {/* Tab 4: Verified Checkpoints */}
      {activeTab === 'checkpoints' && (
        <div className="p-4 rounded-xl bg-studio-card border border-studio-cardBorder shadow space-y-3" data-testid="checkpoints-panel">
          <div className="flex items-center justify-between pb-2 border-b border-studio-cardBorder">
            <h3 className="text-xs font-bold text-studio-text uppercase tracking-wider font-mono flex items-center gap-1.5">
              <HardDrive className="w-3.5 h-3.5 text-studio-teal" />
              Verified Disk Checkpoints
            </h3>
            <button
              type="button"
              onClick={() => activeRunId && loadCheckpoints(activeRunId)}
              className="px-2.5 py-1 rounded bg-studio-input hover:bg-studio-border text-studio-text text-[11px] font-mono inline-flex items-center gap-1 cursor-pointer"
              data-testid="btn-refresh-checkpoints"
            >
              <RefreshCw className="w-3 h-3" /> Refresh
            </button>
          </div>

          {checkpoints.length > 0 ? (
            <div className="space-y-3" data-testid="checkpoints-list">
              {checkpoints.map((cp) => (
                <div
                  key={cp.checkpoint_id}
                  className="p-3 bg-studio-input/60 border border-studio-border rounded-lg text-xs font-mono space-y-2"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-studio-text">{cp.checkpoint_id}</span>
                    <Badge variant="success">Verified On Disk</Badge>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[11px] text-studio-muted">
                    <div>Step: <span className="text-studio-text font-semibold">{cp.step ?? 'Final'}</span></div>
                    <div>Size: <span className="text-studio-text font-semibold">{formatFileSize(cp.size_bytes)}</span></div>
                    <div>Created: <span className="text-studio-text">{cp.created_at}</span></div>
                    <div>Path: <span className="text-studio-accent truncate block">{cp.path}</span></div>
                  </div>
                  <div className="text-[11px] text-studio-muted">
                    Files ({cp.files.length}): <span className="text-studio-text">{cp.files.join(', ')}</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="py-8 text-center text-xs text-studio-muted font-mono">
              No checkpoints found on disk yet for this run.
            </div>
          )}
        </div>
      )}

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
                {run.config.training_method === 'qlora'
                  ? 'QLoRA 4-Bit Adapters Saved and Verified Successfully'
                  : 'LoRA Adapters Saved and Verified Successfully'}
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
                {(artifacts.files || []).map((f) => (
                  <li key={f} className="flex items-center gap-1.5">
                    <FileCheck className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                    <span>{f}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3 pt-2 border-t border-emerald-500/20">
            {onNavigateToArtifacts && (
              <button
                type="button"
                onClick={onNavigateToArtifacts}
                className="px-3.5 py-1.5 bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 border border-emerald-500/40 rounded-lg text-xs font-semibold inline-flex items-center gap-1.5 transition-colors cursor-pointer"
                data-testid="btn-view-run-artifacts"
              >
                <Box className="w-3.5 h-3.5" /> Open in Artifact Registry
              </button>
            )}
            {onNavigateToEvaluation && (
              <button
                type="button"
                onClick={() => onNavigateToEvaluation(run?.run_id)}
                className="px-3.5 py-1.5 bg-studio-card hover:bg-studio-cardHover text-studio-text border border-studio-border rounded-lg text-xs font-semibold inline-flex items-center gap-1.5 transition-colors cursor-pointer"
                data-testid="btn-eval-run-artifacts"
              >
                <BarChart2 className="w-3.5 h-3.5 text-studio-accent" /> Evaluate This Adapter
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
