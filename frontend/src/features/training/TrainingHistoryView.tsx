import React, { useState, useEffect, useCallback } from 'react';
import {
  History,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  AlertCircle,
  FileText,
} from 'lucide-react';
import { Badge } from '../../components/common/Badge';
import { RunHistoryItem } from '../../types/training';
import { fetchRunHistory } from '../../services/api';

interface TrainingHistoryViewProps {
  onSelectRun: (runId: string) => void;
}

export const TrainingHistoryView: React.FC<TrainingHistoryViewProps> = ({
  onSelectRun,
}) => {
  const [runs, setRuns] = useState<RunHistoryItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [pageSize] = useState(10);
  const [statusFilter, setStatusFilter] = useState('all');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadHistory = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await fetchRunHistory(page, pageSize, statusFilter);
      setRuns(data.runs);
      setTotal(data.total);
      setTotalPages(data.total_pages || 1);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to retrieve training history.';
      setError(msg);
      setRuns([]);
    } finally {
      setIsLoading(false);
    }
  }, [page, pageSize, statusFilter]);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  const handleStatusChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setStatusFilter(e.target.value);
    setPage(1);
  };

  const formatDuration = (seconds?: number | null) => {
    if (seconds === undefined || seconds === null || seconds <= 0) return 'Unavailable';
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(0);
    return `${mins}m ${secs}s`;
  };

  const formatTimestamp = (iso?: string | null) => {
    if (!iso) return 'Not recorded';
    try {
      const date = new Date(iso);
      return date.toLocaleString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
    } catch {
      return iso;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status.toLowerCase()) {
      case 'completed':
        return <Badge variant="success">Completed</Badge>;
      case 'training':
        return <Badge variant="default">Training</Badge>;
      case 'failed':
        return <Badge variant="error">Failed</Badge>;
      case 'cancelled':
        return <Badge variant="default">Cancelled</Badge>;
      case 'queued':
        return <Badge variant="default">Queued</Badge>;
      default:
        return <Badge variant="default">{status}</Badge>;
    }
  };

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6 max-w-6xl mx-auto w-full" data-testid="training-history-view">
      {/* Top Bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-studio-cardBorder gap-4">
        <div>
          <h1 className="text-lg font-bold text-studio-text flex items-center gap-2">
            <History className="w-5 h-5 text-studio-accent" />
            Training Run History
          </h1>
          <p className="text-xs text-studio-muted font-mono mt-0.5">
            Persistent archive of all fine-tuning runs, checkpoints, and telemetry.
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* Status Filter */}
          <div className="flex items-center gap-2">
            <label htmlFor="status-filter" className="text-xs text-studio-muted font-mono">
              Status:
            </label>
            <select
              id="status-filter"
              value={statusFilter}
              onChange={handleStatusChange}
              className="bg-studio-input border border-studio-border text-studio-text text-xs rounded-lg px-2.5 py-1.5 focus:border-studio-accent focus:outline-none font-mono cursor-pointer"
              data-testid="history-status-filter"
            >
              <option value="all">All Statuses</option>
              <option value="completed">Completed</option>
              <option value="training">Training</option>
              <option value="failed">Failed</option>
              <option value="cancelled">Cancelled</option>
              <option value="queued">Queued</option>
            </select>
          </div>

          <button
            type="button"
            onClick={() => loadHistory()}
            disabled={isLoading}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-studio-card hover:bg-studio-cardBorder text-studio-text border border-studio-border inline-flex items-center gap-1.5 transition-colors cursor-pointer"
            data-testid="history-refresh-btn"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="p-3 bg-rose-950/30 border border-rose-500/30 rounded-xl text-xs text-rose-300 flex items-center gap-2" data-testid="history-error-banner">
          <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* History Table */}
      <div className="rounded-xl bg-studio-card border border-studio-cardBorder shadow overflow-hidden">
        {isLoading && runs.length === 0 ? (
          <div className="py-16 text-center text-xs text-studio-muted font-mono space-y-2" data-testid="history-loading">
            <RefreshCw className="w-6 h-6 animate-spin mx-auto text-studio-accent" />
            <p>Loading persistent run history...</p>
          </div>
        ) : runs.length === 0 ? (
          <div className="py-16 text-center text-xs text-studio-muted font-mono space-y-3" data-testid="history-empty">
            <FileText className="w-8 h-8 text-studio-muted mx-auto" />
            <p className="font-semibold text-studio-text">No Training Runs Found</p>
            <p className="max-w-md mx-auto text-[11px] leading-relaxed">
              {statusFilter !== 'all'
                ? `No training runs found with status '${statusFilter}'. Try selecting 'All Statuses'.`
                : 'No fine-tuning runs have been executed yet. Launch a run from the Configure tab to see it archived here.'}
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse" data-testid="history-table">
              <thead>
                <tr className="bg-studio-bg/60 text-studio-muted text-[10px] uppercase font-mono border-b border-studio-border">
                  <th className="py-3 px-4">Run ID</th>
                  <th className="py-3 px-4">Model</th>
                  <th className="py-3 px-4">Dataset</th>
                  <th className="py-3 px-4">Method</th>
                  <th className="py-3 px-4">Status</th>
                  <th className="py-3 px-4">Started At</th>
                  <th className="py-3 px-4">Duration</th>
                  <th className="py-3 px-4">Final Loss</th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-studio-border/20 text-xs font-mono">
                {runs.map((r) => (
                  <tr
                    key={r.run_id}
                    className="hover:bg-studio-input/50 transition-colors"
                    data-testid={`history-row-${r.run_id}`}
                  >
                    <td className="py-2.5 px-4">
                      <div className="font-semibold text-studio-text truncate max-w-[150px]" title={r.project_name}>
                        {r.project_name}
                      </div>
                      <div className="text-[10px] text-studio-muted font-mono">
                        {r.run_id}
                      </div>
                    </td>
                    <td className="py-2.5 px-4 text-studio-muted truncate max-w-[140px]" title={r.model_identifier}>
                      {r.model_identifier}
                    </td>
                    <td className="py-2.5 px-4 text-studio-muted truncate max-w-[130px]" title={r.dataset_reference}>
                      {r.dataset_reference}
                    </td>
                    <td className="py-2.5 px-4 uppercase text-[11px]">
                      {r.training_method}
                    </td>
                    <td className="py-2.5 px-4">
                      {getStatusBadge(r.status)}
                    </td>
                    <td className="py-2.5 px-4 text-studio-muted text-[11px]">
                      {formatTimestamp(r.started_at || r.created_at)}
                    </td>
                    <td className="py-2.5 px-4 text-studio-muted text-[11px]">
                      {formatDuration(r.elapsed_time_seconds)}
                    </td>
                    <td className="py-2.5 px-4 text-studio-accent font-semibold">
                      {r.current_loss !== null && r.current_loss !== undefined
                        ? r.current_loss.toFixed(4)
                        : 'Unavailable'}
                    </td>
                    <td className="py-2.5 px-4 text-right">
                      <button
                        type="button"
                        onClick={() => onSelectRun(r.run_id)}
                        className="px-2.5 py-1 rounded bg-studio-input hover:bg-studio-border text-studio-text text-[11px] font-semibold inline-flex items-center gap-1 transition-colors cursor-pointer"
                        data-testid={`btn-inspect-${r.run_id}`}
                      >
                        <ExternalLink className="w-3 h-3" />
                        <span>Inspect</span>
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination Footer */}
        {total > 0 && (
          <div className="p-3 bg-studio-bg/40 border-t border-studio-cardBorder flex flex-col sm:flex-row items-center justify-between gap-3 text-xs font-mono text-studio-muted">
            <div>
              Showing <span className="text-studio-text font-semibold">{(page - 1) * pageSize + 1}</span> to{' '}
              <span className="text-studio-text font-semibold">{Math.min(page * pageSize, total)}</span> of{' '}
              <span className="text-studio-text font-semibold">{total}</span> runs
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                disabled={page <= 1 || isLoading}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                className="px-2.5 py-1 rounded bg-studio-input hover:bg-studio-border disabled:opacity-40 disabled:cursor-not-allowed text-studio-text text-xs inline-flex items-center gap-1 cursor-pointer"
                data-testid="history-prev-page"
              >
                <ChevronLeft className="w-3.5 h-3.5" /> Prev
              </button>
              <span>
                Page <span className="text-studio-text font-semibold">{page}</span> of{' '}
                <span className="text-studio-text font-semibold">{totalPages}</span>
              </span>
              <button
                type="button"
                disabled={page >= totalPages || isLoading}
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                className="px-2.5 py-1 rounded bg-studio-input hover:bg-studio-border disabled:opacity-40 disabled:cursor-not-allowed text-studio-text text-xs inline-flex items-center gap-1 cursor-pointer"
                data-testid="history-next-page"
              >
                Next <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
