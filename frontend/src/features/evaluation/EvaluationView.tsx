import React, { useState, useEffect, useCallback } from 'react';
import { 
  Play, 
  BarChart2, 
  CheckCircle2, 
  AlertTriangle, 
  XCircle, 
  Clock, 
  RefreshCw, 
  Scale, 
  FileText,
  StopCircle,
  TrendingDown,
  TrendingUp,
  Sliders
} from 'lucide-react';
import { 
  ArtifactMetadata, 
  DatasetSummary, 
  EvaluationConfig, 
  EvaluationRun, 
  EvaluationStatus 
} from '../../types/training';
import { 
  fetchAllArtifacts, 
  fetchDatasets, 
  createEvaluation, 
  fetchEvaluations, 
  fetchEvaluationDetails, 
  cancelEvaluation 
} from '../../services/api';

interface EvaluationViewProps {
  initialArtifactId?: string | null;
  onNavigateToArtifacts?: () => void;
}

export const EvaluationView: React.FC<EvaluationViewProps> = ({
  initialArtifactId = null,
  onNavigateToArtifacts,
}) => {
  // Selection states
  const [artifacts, setArtifacts] = useState<ArtifactMetadata[]>([]);
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [selectedArtifactId, setSelectedArtifactId] = useState<string>(initialArtifactId || '');
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');

  // Configuration states
  const [split, setSplit] = useState<string>('validation');
  const [maxSamples, setMaxSamples] = useState<number>(20);
  const [batchSize, setBatchSize] = useState<number>(2);
  const [temperature] = useState<number>(0.7);
  const [maxNewTokens, setMaxNewTokens] = useState<number>(64);
  const [runBaselineComparison, setRunBaselineComparison] = useState<boolean>(true);
  const [generateSamples, setGenerateSamples] = useState<boolean>(true);

  // Active / historical runs
  const [activeEvaluation, setActiveEvaluation] = useState<EvaluationRun | null>(null);
  const [pastEvaluations, setPastEvaluations] = useState<EvaluationRun[]>([]);
  const [, setLoading] = useState<boolean>(true);
  const [starting, setStarting] = useState<boolean>(false);
  const [cancelling, setCancelling] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Load available artifacts & datasets
  const loadPrerequisites = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [artData, dsData] = await Promise.all([
        fetchAllArtifacts(),
        fetchDatasets(),
      ]);
      setArtifacts(artData);
      setDatasets(dsData.datasets || []);

      if (initialArtifactId) {
        setSelectedArtifactId(initialArtifactId);
      } else if (artData.length > 0 && !selectedArtifactId) {
        // Prefer first valid artifact
        const firstValid = artData.find((a) => a.validation_status === 'valid') || artData[0];
        setSelectedArtifactId(firstValid.artifact_id);
      }

      if (dsData.datasets && dsData.datasets.length > 0 && !selectedDatasetId) {
        setSelectedDatasetId(dsData.datasets[0].id);
      }

      // Load past evaluations
      const evals = await fetchEvaluations({ page_size: 10 });
      setPastEvaluations(evals.evaluations || []);
      if (evals.evaluations && evals.evaluations.length > 0 && !activeEvaluation) {
        setActiveEvaluation(evals.evaluations[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load evaluation setup.');
    } finally {
      setLoading(false);
    }
  }, [initialArtifactId]);

  useEffect(() => {
    loadPrerequisites();
  }, [loadPrerequisites]);

  // Poll status if evaluation is running or queued
  useEffect(() => {
    if (!activeEvaluation) return;
    if (activeEvaluation.status !== 'queued' && activeEvaluation.status !== 'running') return;

    const interval = setInterval(async () => {
      try {
        const updated = await fetchEvaluationDetails(activeEvaluation.evaluation_id);
        setActiveEvaluation(updated);
        if (updated.status === 'completed' || updated.status === 'failed' || updated.status === 'cancelled') {
          // Refresh list of evaluations
          const list = await fetchEvaluations({ page_size: 10 });
          setPastEvaluations(list.evaluations || []);
        }
      } catch (err) {
        console.error('Error polling evaluation status:', err);
      }
    }, 1500);

    return () => clearInterval(interval);
  }, [activeEvaluation]);

  const handleStartEvaluation = async () => {
    if (!selectedArtifactId || !selectedDatasetId) {
      setError('Please select both a validated artifact and a validation dataset.');
      return;
    }

    setStarting(true);
    setError(null);

    const config: EvaluationConfig = {
      artifact_id: selectedArtifactId,
      dataset_id: selectedDatasetId,
      split,
      max_samples: maxSamples,
      batch_size: batchSize,
      temperature,
      max_new_tokens: maxNewTokens,
      run_baseline_comparison: runBaselineComparison,
      generate_samples: generateSamples,
      num_samples_to_save: 3,
    };

    try {
      const run = await createEvaluation(config);
      setActiveEvaluation(run);
      // Refresh past evals list
      const list = await fetchEvaluations({ page_size: 10 });
      setPastEvaluations(list.evaluations || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start evaluation job.');
    } finally {
      setStarting(false);
    }
  };

  const handleCancelEvaluation = async () => {
    if (!activeEvaluation) return;
    setCancelling(true);
    try {
      const res = await cancelEvaluation(activeEvaluation.evaluation_id);
      setActiveEvaluation(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cancel evaluation.');
    } finally {
      setCancelling(false);
    }
  };

  const selectedArtifact = artifacts.find((a) => a.artifact_id === selectedArtifactId);
  const isValidArtifact = selectedArtifact?.validation_status === 'valid';

  const getStatusBadge = (status: EvaluationStatus) => {
    switch (status) {
      case 'completed':
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded text-xs font-medium bg-emerald-950/80 text-emerald-400 border border-emerald-800/60" data-testid="eval-status-completed">
            <CheckCircle2 className="w-3.5 h-3.5" /> Completed
          </span>
        );
      case 'running':
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded text-xs font-medium bg-cyan-950/80 text-cyan-400 border border-cyan-800/60" data-testid="eval-status-running">
            <RefreshCw className="w-3.5 h-3.5 animate-spin" /> Evaluating...
          </span>
        );
      case 'queued':
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded text-xs font-medium bg-amber-950/80 text-amber-400 border border-amber-800/60" data-testid="eval-status-queued">
            <Clock className="w-3.5 h-3.5" /> Queued
          </span>
        );
      case 'failed':
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded text-xs font-medium bg-rose-950/80 text-rose-400 border border-rose-800/60" data-testid="eval-status-failed">
            <XCircle className="w-3.5 h-3.5" /> Failed
          </span>
        );
      case 'cancelled':
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded text-xs font-medium bg-zinc-800 text-zinc-400 border border-zinc-700" data-testid="eval-status-cancelled">
            <StopCircle className="w-3.5 h-3.5" /> Cancelled
          </span>
        );
    }
  };

  return (
    <div className="flex-1 flex flex-col min-w-0 overflow-y-auto bg-studio-bg text-studio-text p-6 space-y-6" data-testid="evaluation-view">
      {/* View Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-studio-border pb-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2 bg-studio-card border border-studio-border rounded-lg text-studio-accent">
              <BarChart2 className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-studio-text flex items-center gap-2">
                <span>Model Evaluation & Baseline Suite</span>
                <span className="text-xs px-2 py-0.5 rounded font-mono bg-studio-card border border-studio-border text-studio-muted">
                  Phase 8
                </span>
              </h1>
              <p className="text-xs text-studio-muted mt-0.5">
                Evaluate fine-tuned model adapters against validation datasets and compute honest baseline comparisons.
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {onNavigateToArtifacts && (
            <button
              onClick={onNavigateToArtifacts}
              className="px-3 py-1.5 rounded-lg border border-studio-border bg-studio-card hover:bg-studio-cardHover text-xs font-medium text-studio-text transition-colors"
              data-testid="btn-navigate-to-artifacts"
            >
              Registry Artifacts
            </button>
          )}

          {pastEvaluations.length > 0 && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-studio-muted font-mono">Past Runs:</span>
              <select
                value={activeEvaluation?.evaluation_id || ''}
                onChange={(e) => {
                  const found = pastEvaluations.find((ev) => ev.evaluation_id === e.target.value);
                  if (found) setActiveEvaluation(found);
                }}
                className="bg-studio-card border border-studio-border rounded-md px-2.5 py-1.5 text-xs text-studio-text font-mono focus:outline-hidden focus:border-studio-accent"
                data-testid="past-evaluations-select"
              >
                {pastEvaluations.map((ev) => (
                  <option key={ev.evaluation_id} value={ev.evaluation_id}>
                    {ev.evaluation_id} ({ev.status}) - {new Date(ev.created_at).toLocaleTimeString()}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
      </div>

      {error && (
        <div className="p-3.5 bg-rose-950/40 border border-rose-800/60 rounded-xl text-rose-300 text-xs flex items-center justify-between" data-testid="eval-error-banner">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{error}</span>
          </div>
          <button onClick={() => setError(null)} className="text-xs underline hover:opacity-80">
            Dismiss
          </button>
        </div>
      )}

      {/* Main Grid: Left Config Panel, Right Results Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Configuration Studio (5 Cols) */}
        <div className="lg:col-span-5 space-y-5">
          <div className="bg-studio-card border border-studio-border rounded-xl p-5 space-y-4">
            <div className="flex items-center gap-2 border-b border-studio-border pb-3">
              <Sliders className="w-4 h-4 text-studio-accent" />
              <h2 className="text-sm font-semibold text-studio-text">Evaluation Setup</h2>
            </div>

            {/* Step 1: Select Artifact */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-studio-text flex items-center justify-between">
                <span>1. Select Trained Artifact</span>
                {selectedArtifact && (
                  <span className={`text-[10px] font-mono px-1.5 py-0.2 rounded ${isValidArtifact ? 'bg-emerald-950 text-emerald-400' : 'bg-rose-950 text-rose-400'}`}>
                    {selectedArtifact.validation_status}
                  </span>
                )}
              </label>
              <select
                value={selectedArtifactId}
                onChange={(e) => setSelectedArtifactId(e.target.value)}
                className="w-full bg-studio-bg border border-studio-border rounded-lg p-2 text-xs font-mono text-studio-text focus:outline-hidden focus:border-studio-accent"
                data-testid="select-artifact"
              >
                <option value="">-- Choose an artifact --</option>
                {artifacts.map((a) => (
                  <option key={a.artifact_id} value={a.artifact_id}>
                    {a.artifact_id} ({a.artifact_type.replace('_', ' ')}) - {a.model_identifier}
                  </option>
                ))}
              </select>

              {selectedArtifact && !isValidArtifact && (
                <p className="text-[11px] text-amber-400/90 flex items-center gap-1 mt-1" data-testid="artifact-unvalidated-warning">
                  <AlertTriangle className="w-3.5 h-3.5" />
                  <span>This artifact is not yet validated. Validate it in the Artifacts tab before evaluating.</span>
                </p>
              )}
            </div>

            {/* Step 2: Select Validation Dataset */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-studio-text">
                2. Select Validation Dataset
              </label>
              <select
                value={selectedDatasetId}
                onChange={(e) => setSelectedDatasetId(e.target.value)}
                className="w-full bg-studio-bg border border-studio-border rounded-lg p-2 text-xs font-mono text-studio-text focus:outline-hidden focus:border-studio-accent"
                data-testid="select-dataset"
              >
                <option value="">-- Choose a dataset --</option>
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.id} ({d.schema_type || 'unstructured'}) - {d.total_samples} samples
                  </option>
                ))}
              </select>
            </div>

            {/* Step 3: Split and Sample Bounds */}
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <label className="text-[11px] text-studio-muted font-mono">Dataset Split</label>
                <select
                  value={split}
                  onChange={(e) => setSplit(e.target.value)}
                  className="w-full bg-studio-bg border border-studio-border rounded-md px-2 py-1.5 text-xs text-studio-text"
                  data-testid="select-split"
                >
                  <option value="validation">Validation</option>
                  <option value="test">Test</option>
                  <option value="train">Train</option>
                </select>
              </div>

              <div className="space-y-1">
                <label className="text-[11px] text-studio-muted font-mono flex items-center justify-between">
                  <span>Sample Limit</span>
                  <span className="text-studio-text">{maxSamples}</span>
                </label>
                <input
                  type="range"
                  min="5"
                  max="100"
                  step="5"
                  value={maxSamples}
                  onChange={(e) => setMaxSamples(Number(e.target.value))}
                  className="w-full accent-studio-accent"
                  data-testid="range-max-samples"
                />
              </div>
            </div>

            {/* Batch Size & Max New Tokens */}
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <label className="text-[11px] text-studio-muted font-mono">Batch Size</label>
                <select
                  value={batchSize}
                  onChange={(e) => setBatchSize(Number(e.target.value))}
                  className="w-full bg-studio-bg border border-studio-border rounded-md px-2 py-1.5 text-xs text-studio-text"
                >
                  <option value={1}>1 (Low VRAM)</option>
                  <option value={2}>2 (Recommended)</option>
                  <option value={4}>4</option>
                  <option value={8}>8</option>
                </select>
              </div>

              <div className="space-y-1">
                <label className="text-[11px] text-studio-muted font-mono">Max Generated Tokens</label>
                <select
                  value={maxNewTokens}
                  onChange={(e) => setMaxNewTokens(Number(e.target.value))}
                  className="w-full bg-studio-bg border border-studio-border rounded-md px-2 py-1.5 text-xs text-studio-text"
                >
                  <option value={16}>16 tokens</option>
                  <option value={32}>32 tokens</option>
                  <option value={64}>64 tokens (default)</option>
                  <option value={128}>128 tokens</option>
                </select>
              </div>
            </div>

            {/* Checkboxes: Baseline & Samples */}
            <div className="space-y-2 pt-2 border-t border-studio-border/60">
              <label className="flex items-center gap-2 cursor-pointer text-xs text-studio-text select-none">
                <input
                  type="checkbox"
                  checked={runBaselineComparison}
                  onChange={(e) => setRunBaselineComparison(e.target.checked)}
                  className="rounded border-studio-border text-studio-accent focus:ring-0"
                  data-testid="checkbox-baseline-comparison"
                />
                <span className="font-medium">Run Side-by-Side Baseline Comparison</span>
              </label>
              <p className="text-[11px] text-studio-muted ml-5">
                Evaluates the compatible base model without the adapter on the exact same dataset to determine true fine-tuning impact.
              </p>

              <label className="flex items-center gap-2 cursor-pointer text-xs text-studio-text select-none pt-1">
                <input
                  type="checkbox"
                  checked={generateSamples}
                  onChange={(e) => setGenerateSamples(e.target.checked)}
                  className="rounded border-studio-border text-studio-accent focus:ring-0"
                />
                <span className="font-medium">Generate Qualitative Text Samples</span>
              </label>
            </div>

            {/* Start Button */}
            <div className="pt-3">
              <button
                onClick={handleStartEvaluation}
                disabled={starting || !selectedArtifactId || !selectedDatasetId || !isValidArtifact}
                className="w-full py-2.5 rounded-lg bg-studio-accent text-studio-bg font-semibold text-xs flex items-center justify-center gap-2 shadow-sm shadow-studio-accent/20 hover:bg-studio-accent/90 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                data-testid="start-evaluation-btn"
              >
                {starting ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Initiating Evaluation...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-current" />
                    <span>Run Model Evaluation</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Right: Evaluation Progress & Results Dashboard (7 Cols) */}
        <div className="lg:col-span-7 space-y-5">
          {activeEvaluation ? (
            <div className="space-y-5">
              {/* Active Evaluation Card Header */}
              <div className="bg-studio-card border border-studio-border rounded-xl p-5 space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-studio-border pb-3">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-sm font-semibold text-studio-text" data-testid="eval-run-id">
                        {activeEvaluation.evaluation_id}
                      </span>
                      {getStatusBadge(activeEvaluation.status)}
                    </div>
                    <p className="text-xs text-studio-muted mt-0.5">
                      Artifact: <span className="font-mono text-studio-text">{activeEvaluation.artifact_id}</span> • Dataset: <span className="font-mono text-studio-text">{activeEvaluation.dataset_id}</span>
                    </p>
                  </div>

                  {activeEvaluation.status === 'running' && (
                    <button
                      onClick={handleCancelEvaluation}
                      disabled={cancelling}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-rose-800 bg-rose-950/60 hover:bg-rose-900/60 text-rose-300 text-xs font-medium transition-colors"
                      data-testid="cancel-eval-btn"
                    >
                      <StopCircle className="w-3.5 h-3.5" />
                      <span>{cancelling ? 'Cancelling...' : 'Cancel Evaluation'}</span>
                    </button>
                  )}
                </div>

                {/* Running Banner */}
                {activeEvaluation.status === 'running' && (
                  <div className="p-4 bg-cyan-950/30 border border-cyan-800/60 rounded-lg flex items-center gap-3">
                    <RefreshCw className="w-5 h-5 text-cyan-400 animate-spin shrink-0" />
                    <div>
                      <p className="text-xs font-semibold text-cyan-300">Inference & Validation in Progress</p>
                      <p className="text-[11px] text-cyan-400/80">Computing Cross-Entropy loss, perplexity, and token throughput on {activeEvaluation.config.max_samples} held-out validation sequences...</p>
                    </div>
                  </div>
                )}

                {/* Failed Error Message */}
                {activeEvaluation.status === 'failed' && (
                  <div className="p-3.5 bg-rose-950/40 border border-rose-800/60 rounded-lg text-rose-300 text-xs space-y-1" data-testid="eval-failed-message">
                    <div className="font-semibold text-rose-400 flex items-center gap-1.5">
                      <AlertTriangle className="w-4 h-4" /> Evaluation Execution Failed
                    </div>
                    <p className="font-mono text-[11px] break-all">{activeEvaluation.error_message}</p>
                  </div>
                )}

                {/* Real Metrics Cards */}
                {activeEvaluation.metrics && (
                  <div className="space-y-4" data-testid="eval-metrics-panel">
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      {/* Evaluation Loss */}
                      <div className="p-3 bg-studio-bg/70 border border-studio-border rounded-lg space-y-1">
                        <span className="text-[10px] uppercase font-mono text-studio-muted">Eval Loss</span>
                        <p className="text-lg font-bold font-mono text-emerald-400" data-testid="metric-eval-loss">
                          {activeEvaluation.metrics.eval_loss !== null && activeEvaluation.metrics.eval_loss !== undefined
                            ? activeEvaluation.metrics.eval_loss.toFixed(4)
                            : '—'}
                        </p>
                        <span className="text-[10px] text-studio-muted">Cross-Entropy</span>
                      </div>

                      {/* Perplexity */}
                      <div className="p-3 bg-studio-bg/70 border border-studio-border rounded-lg space-y-1">
                        <span className="text-[10px] uppercase font-mono text-studio-muted">Perplexity</span>
                        <p className="text-lg font-bold font-mono text-cyan-400" data-testid="metric-perplexity">
                          {activeEvaluation.metrics.perplexity !== null && activeEvaluation.metrics.perplexity !== undefined
                            ? activeEvaluation.metrics.perplexity.toFixed(2)
                            : activeEvaluation.metrics.perplexity_overflow
                            ? 'Overflow'
                            : '—'}
                        </p>
                        <span className="text-[10px] text-studio-muted">exp(loss)</span>
                      </div>

                      {/* Throughput */}
                      <div className="p-3 bg-studio-bg/70 border border-studio-border rounded-lg space-y-1">
                        <span className="text-[10px] uppercase font-mono text-studio-muted">Throughput</span>
                        <p className="text-lg font-bold font-mono text-studio-text" data-testid="metric-throughput">
                          {activeEvaluation.metrics.tokens_per_second.toFixed(1)}
                        </p>
                        <span className="text-[10px] text-studio-muted">tokens/sec</span>
                      </div>

                      {/* Evaluated Tokens */}
                      <div className="p-3 bg-studio-bg/70 border border-studio-border rounded-lg space-y-1">
                        <span className="text-[10px] uppercase font-mono text-studio-muted">Total Tokens</span>
                        <p className="text-lg font-bold font-mono text-studio-text" data-testid="metric-total-tokens">
                          {activeEvaluation.metrics.total_tokens.toLocaleString()}
                        </p>
                        <span className="text-[10px] text-studio-muted">{activeEvaluation.metrics.eval_samples_count} sequences</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Side-by-Side Baseline Comparison Card */}
              {activeEvaluation.baseline_comparison && (
                <div className="bg-studio-card border border-studio-border rounded-xl p-5 space-y-4" data-testid="baseline-comparison-card">
                  <div className="flex items-center justify-between border-b border-studio-border pb-3">
                    <div className="flex items-center gap-2">
                      <Scale className="w-4 h-4 text-studio-accent" />
                      <h3 className="text-sm font-semibold text-studio-text">Baseline Comparison</h3>
                    </div>
                    {activeEvaluation.baseline_comparison.is_comparable && activeEvaluation.baseline_comparison.is_improvement !== null && (
                      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium ${
                        activeEvaluation.baseline_comparison.is_improvement
                          ? 'bg-emerald-950/80 text-emerald-400 border border-emerald-800/60'
                          : 'bg-amber-950/80 text-amber-400 border border-amber-800/60'
                      }`} data-testid="comparison-improvement-badge">
                        {activeEvaluation.baseline_comparison.is_improvement ? (
                          <>
                            <TrendingDown className="w-3.5 h-3.5" /> Improved Performance
                          </>
                        ) : (
                          <>
                            <TrendingUp className="w-3.5 h-3.5" /> Higher Loss
                          </>
                        )}
                      </span>
                    )}
                  </div>

                  {activeEvaluation.baseline_comparison.is_comparable ? (
                    <div className="space-y-4">
                      {/* Comparison Table */}
                      <div className="grid grid-cols-3 gap-3 text-center">
                        <div className="p-3 bg-studio-bg/80 rounded-lg border border-studio-border">
                          <span className="text-[10px] uppercase font-mono text-studio-muted">Metric</span>
                          <div className="text-xs font-medium text-studio-text mt-2 space-y-2">
                            <div>Eval Loss</div>
                            <div>Perplexity</div>
                            <div>Throughput</div>
                          </div>
                        </div>

                        <div className="p-3 bg-studio-bg/80 rounded-lg border border-studio-border">
                          <span className="text-[10px] uppercase font-mono text-studio-muted">Base Model</span>
                          <div className="text-xs font-mono text-studio-muted mt-2 space-y-2">
                            <div data-testid="base-loss">
                              {activeEvaluation.baseline_comparison.base_eval_loss !== null
                                ? activeEvaluation.baseline_comparison.base_eval_loss?.toFixed(4)
                                : '—'}
                            </div>
                            <div data-testid="base-ppl">
                              {activeEvaluation.baseline_comparison.base_perplexity !== null
                                ? activeEvaluation.baseline_comparison.base_perplexity?.toFixed(2)
                                : '—'}
                            </div>
                            <div>
                              {activeEvaluation.baseline_comparison.base_tokens_per_second !== null
                                ? `${activeEvaluation.baseline_comparison.base_tokens_per_second?.toFixed(1)} t/s`
                                : '—'}
                            </div>
                          </div>
                        </div>

                        <div className="p-3 bg-studio-accent/5 rounded-lg border border-studio-accent/30">
                          <span className="text-[10px] uppercase font-mono text-studio-accent font-semibold">Fine-Tuned Adapter</span>
                          <div className="text-xs font-mono font-semibold text-studio-text mt-2 space-y-2">
                            <div className="text-emerald-400" data-testid="finetuned-loss">
                              {activeEvaluation.baseline_comparison.fine_tuned_eval_loss?.toFixed(4)}
                              {activeEvaluation.baseline_comparison.loss_delta !== null && activeEvaluation.baseline_comparison.loss_delta !== undefined && (
                                <span className="text-[10px] ml-1 text-emerald-400/80">
                                  ({activeEvaluation.baseline_comparison.loss_delta > 0 ? '+' : ''}{activeEvaluation.baseline_comparison.loss_delta})
                                </span>
                              )}
                            </div>
                            <div className="text-cyan-400" data-testid="finetuned-ppl">
                              {activeEvaluation.baseline_comparison.fine_tuned_perplexity?.toFixed(2)}
                            </div>
                            <div>
                              {activeEvaluation.baseline_comparison.fine_tuned_tokens_per_second?.toFixed(1)} t/s
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Summary Text */}
                      <p className="text-xs text-studio-muted italic bg-studio-bg/40 p-2.5 rounded border border-studio-border" data-testid="comparison-summary">
                        "{activeEvaluation.baseline_comparison.comparison_summary}"
                      </p>
                    </div>
                  ) : (
                    <div className="p-3 bg-amber-950/40 border border-amber-800/60 rounded-lg text-amber-300 text-xs">
                      <p className="font-semibold">Baseline comparison unavailable:</p>
                      <p className="text-[11px] mt-0.5">{activeEvaluation.baseline_comparison.incomparable_reason || 'Base model could not be loaded for comparative inference.'}</p>
                    </div>
                  )}
                </div>
              )}

              {/* Sample Outputs Accordion */}
              {activeEvaluation.samples && activeEvaluation.samples.length > 0 && (
                <div className="bg-studio-card border border-studio-border rounded-xl p-5 space-y-3" data-testid="eval-samples-panel">
                  <div className="flex items-center gap-2 border-b border-studio-border pb-3">
                    <FileText className="w-4 h-4 text-studio-accent" />
                    <h3 className="text-sm font-semibold text-studio-text">Qualitative Generated Samples</h3>
                  </div>

                  <div className="space-y-3">
                    {activeEvaluation.samples.map((s) => (
                      <div key={s.sample_id} className="p-3.5 bg-studio-bg/60 border border-studio-border rounded-lg space-y-2 text-xs font-mono">
                        <div className="flex items-center justify-between text-studio-muted text-[11px]">
                          <span>Sample #{s.sample_id}</span>
                          <span>Latency: {s.generation_time_ms} ms</span>
                        </div>

                        {/* Input Prompt */}
                        <div>
                          <span className="text-[10px] uppercase text-studio-muted">Input Prompt:</span>
                          <div className="p-2 bg-black/40 rounded border border-studio-border text-studio-text whitespace-pre-wrap">
                            {s.prompt}
                          </div>
                        </div>

                        {/* Side by side outputs */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 pt-1">
                          {s.base_output && (
                            <div>
                              <span className="text-[10px] uppercase text-studio-muted">Base Model Output:</span>
                              <div className="p-2 bg-black/30 rounded border border-studio-border text-studio-muted whitespace-pre-wrap">
                                {s.base_output}
                              </div>
                            </div>
                          )}

                          <div className={s.base_output ? '' : 'md:col-span-2'}>
                            <span className="text-[10px] uppercase text-studio-accent font-semibold">Fine-Tuned Adapter Output:</span>
                            <div className="p-2 bg-studio-accent/5 rounded border border-studio-accent/30 text-studio-text whitespace-pre-wrap">
                              {s.fine_tuned_output}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="p-16 border border-dashed border-studio-border rounded-xl bg-studio-card/20 text-center space-y-3 flex flex-col items-center justify-center">
              <BarChart2 className="w-10 h-10 text-studio-muted stroke-[1.5]" />
              <p className="text-sm font-semibold text-studio-text">No active evaluation selected</p>
              <p className="text-xs text-studio-muted max-w-sm">
                Choose an artifact and dataset on the left, then click "Run Model Evaluation" to compute genuine loss and perplexity benchmarks.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
