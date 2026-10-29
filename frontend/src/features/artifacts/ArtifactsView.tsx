import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { 
  Box, 
  Search, 
  Filter, 
  RefreshCw, 
  ShieldCheck, 
  Download, 
  Play, 
  Eye, 
  CheckCircle2, 
  XCircle, 
  Clock, 
  HardDrive
} from 'lucide-react';
import { ArtifactMetadata, ValidationStatus, ArtifactType } from '../../types/training';
import { 
  fetchAllArtifacts, 
  validateArtifact, 
  exportArtifact, 
  getArtifactDownloadUrl 
} from '../../services/api';
import { ArtifactDetailsModal } from './ArtifactDetailsModal';

interface ArtifactsViewProps {
  initialRunId?: string | null;
  onNavigateToEvaluation?: (artifactId: string) => void;
}

export const ArtifactsView: React.FC<ArtifactsViewProps> = ({
  initialRunId = null,
  onNavigateToEvaluation,
}) => {
  const [artifacts, setArtifacts] = useState<ArtifactMetadata[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<'all' | ValidationStatus>('all');
  const [typeFilter, setTypeFilter] = useState<'all' | ArtifactType>('all');
  const [selectedArtifact, setSelectedArtifact] = useState<ArtifactMetadata | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [validatingId, setValidatingId] = useState<string | null>(null);
  const [exportingId, setExportingId] = useState<string | null>(null);
  const [notification, setNotification] = useState<{ type: 'success' | 'error' | 'info'; message: string } | null>(null);

  const loadArtifacts = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchAllArtifacts(initialRunId || undefined);
      setArtifacts(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load artifacts.');
    } finally {
      setLoading(false);
    }
  }, [initialRunId]);

  useEffect(() => {
    loadArtifacts();
  }, [loadArtifacts]);

  const handleValidate = async (artifactId: string) => {
    setValidatingId(artifactId);
    try {
      const res = await validateArtifact(artifactId, true);
      if (res.is_valid) {
        setNotification({
          type: 'success',
          message: `Artifact ${artifactId} passed integrity validation!`,
        });
      } else {
        setNotification({
          type: 'error',
          message: `Validation failed: ${res.errors.join('; ')}`,
        });
      }
      // Refresh artifacts to reflect updated validation status
      await loadArtifacts();
      if (selectedArtifact && selectedArtifact.artifact_id === artifactId) {
        setSelectedArtifact((prev) => prev ? {
          ...prev,
          validation_status: res.validation_status,
          validation_errors: res.errors,
          validation_warnings: res.warnings,
          size_bytes: res.total_size_bytes,
        } : null);
      }
    } catch (err) {
      setNotification({
        type: 'error',
        message: err instanceof Error ? err.message : 'Validation request failed.',
      });
    } finally {
      setValidatingId(null);
    }
  };

  const handleExport = async (artifactId: string) => {
    setExportingId(artifactId);
    try {
      const res = await exportArtifact(artifactId);
      setNotification({
        type: 'success',
        message: res.message || `Export created: ${res.package_filename}`,
      });
      await loadArtifacts();
      if (selectedArtifact && selectedArtifact.artifact_id === artifactId) {
        setSelectedArtifact((prev) => prev ? {
          ...prev,
          export_status: 'exported',
          exported_package_path: res.download_url,
        } : null);
      }
    } catch (err) {
      setNotification({
        type: 'error',
        message: err instanceof Error ? err.message : 'Export failed.',
      });
    } finally {
      setExportingId(null);
    }
  };

  const filteredArtifacts = useMemo(() => {
    return artifacts.filter((art) => {
      const matchesSearch =
        searchQuery === '' ||
        art.artifact_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
        art.run_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
        art.model_identifier.toLowerCase().includes(searchQuery.toLowerCase());

      const matchesStatus = statusFilter === 'all' || art.validation_status === statusFilter;
      const matchesType = typeFilter === 'all' || art.artifact_type === typeFilter;

      return matchesSearch && matchesStatus && matchesType;
    });
  }, [artifacts, searchQuery, statusFilter, typeFilter]);

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
  };

  return (
    <div className="flex-1 flex flex-col min-w-0 overflow-y-auto bg-studio-bg text-studio-text p-6 space-y-6" data-testid="artifacts-view">
      {/* View Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-studio-border pb-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2 bg-studio-card border border-studio-border rounded-lg text-studio-accent">
              <Box className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-studio-text flex items-center gap-2">
                <span>Training Artifacts & Adapters</span>
                <span className="text-xs px-2 py-0.5 rounded font-mono bg-studio-card border border-studio-border text-studio-muted">
                  Phase 8
                </span>
              </h1>
              <p className="text-xs text-studio-muted mt-0.5">
                Inspect, validate, export, and evaluate trained model artifacts and adapters.
              </p>
            </div>
          </div>
        </div>

        <button
          onClick={loadArtifacts}
          disabled={loading}
          className="self-start md:self-auto flex items-center gap-2 px-3 py-1.5 rounded-lg border border-studio-border bg-studio-card hover:bg-studio-cardHover text-xs font-medium transition-colors"
          data-testid="refresh-artifacts-btn"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Notification Banner */}
      {notification && (
        <div 
          className={`p-3 rounded-lg border text-xs flex items-center justify-between animate-in fade-in ${
            notification.type === 'success' 
              ? 'bg-emerald-950/60 border-emerald-800 text-emerald-300' 
              : notification.type === 'error'
              ? 'bg-rose-950/60 border-rose-800 text-rose-300'
              : 'bg-zinc-800 border-zinc-700 text-zinc-300'
          }`}
          data-testid="artifact-notification"
        >
          <span>{notification.message}</span>
          <button 
            onClick={() => setNotification(null)}
            className="text-xs underline ml-4 hover:opacity-80"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Filter and Search Bar */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 bg-studio-card/70 p-3 rounded-lg border border-studio-border">
        {/* Search */}
        <div className="relative flex-1">
          <Search className="w-4 h-4 text-studio-muted absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search by artifact ID, run ID, or base model..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-studio-bg border border-studio-border rounded-md pl-9 pr-3 py-1.5 text-xs text-studio-text placeholder-studio-muted focus:outline-hidden focus:border-studio-accent"
            data-testid="search-artifacts-input"
          />
        </div>

        {/* Status Filter */}
        <div className="flex items-center gap-2">
          <Filter className="w-3.5 h-3.5 text-studio-muted" />
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value as any)}
            className="bg-studio-bg border border-studio-border rounded-md px-2.5 py-1.5 text-xs text-studio-text focus:outline-hidden focus:border-studio-accent"
            data-testid="filter-status-select"
          >
            <option value="all">All Statuses</option>
            <option value="valid">Valid Only</option>
            <option value="unvalidated">Unvalidated</option>
            <option value="invalid">Invalid</option>
            <option value="corrupted">Corrupted</option>
          </select>

          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value as any)}
            className="bg-studio-bg border border-studio-border rounded-md px-2.5 py-1.5 text-xs text-studio-text focus:outline-hidden focus:border-studio-accent"
            data-testid="filter-type-select"
          >
            <option value="all">All Types</option>
            <option value="lora_adapter">LoRA Adapter</option>
            <option value="qlora_adapter">QLoRA Adapter</option>
            <option value="checkpoint">Checkpoint</option>
            <option value="full_model">Full Model</option>
          </select>
        </div>
      </div>

      {/* Artifacts Table */}
      {loading ? (
        <div className="p-12 text-center text-studio-muted flex flex-col items-center justify-center space-y-3">
          <RefreshCw className="w-6 h-6 animate-spin text-studio-accent" />
          <span className="text-xs">Scanning disk for validated artifacts...</span>
        </div>
      ) : error ? (
        <div className="p-6 bg-rose-950/40 border border-rose-800/60 rounded-xl text-rose-300 text-xs">
          Error: {error}
        </div>
      ) : filteredArtifacts.length === 0 ? (
        <div className="p-12 text-center border border-dashed border-studio-border rounded-xl bg-studio-card/30 space-y-2" data-testid="empty-artifacts">
          <HardDrive className="w-8 h-8 text-studio-muted mx-auto" />
          <p className="text-sm font-semibold text-studio-text">No artifacts found</p>
          <p className="text-xs text-studio-muted max-w-sm mx-auto">
            {searchQuery || statusFilter !== 'all' || typeFilter !== 'all'
              ? 'No artifacts matched your current filter criteria.'
              : 'Complete a training run in the Configure tab to generate validated model adapters.'}
          </p>
        </div>
      ) : (
        <div className="border border-studio-border rounded-xl overflow-hidden bg-studio-card/60">
          <table className="w-full text-left border-collapse text-xs" data-testid="artifacts-table">
            <thead>
              <tr className="bg-studio-bg/90 border-b border-studio-border text-[11px] font-mono text-studio-muted uppercase">
                <th className="py-3 px-4">Artifact ID / Run</th>
                <th className="py-3 px-4">Type</th>
                <th className="py-3 px-4">Base Model</th>
                <th className="py-3 px-4">Size</th>
                <th className="py-3 px-4">Validation</th>
                <th className="py-3 px-4">Export Status</th>
                <th className="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-studio-border/60">
              {filteredArtifacts.map((art) => {
                const isValid = art.validation_status === 'valid';
                const isExported = art.export_status === 'exported';
                const isValidating = validatingId === art.artifact_id;
                const isExporting = exportingId === art.artifact_id;

                return (
                  <tr 
                    key={art.artifact_id} 
                    className="hover:bg-studio-cardHover/50 transition-colors"
                    data-testid={`artifact-row-${art.artifact_id}`}
                  >
                    {/* Artifact ID & Run */}
                    <td className="py-3 px-4">
                      <div className="font-mono text-xs font-semibold text-studio-text" data-testid="artifact-id">
                        {art.artifact_id}
                      </div>
                      <div className="text-[11px] text-studio-muted flex items-center gap-1 font-mono">
                        Run: <span className="text-studio-muted">{art.run_id}</span>
                        {art.checkpoint_step && (
                          <span className="text-[10px] bg-studio-card px-1 rounded border border-studio-border">
                            step {art.checkpoint_step}
                          </span>
                        )}
                      </div>
                    </td>

                    {/* Type Badge */}
                    <td className="py-3 px-4">
                      <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-studio-card border border-studio-border text-studio-accent font-medium">
                        {art.artifact_type.replace('_', ' ')}
                      </span>
                    </td>

                    {/* Base Model */}
                    <td className="py-3 px-4 font-mono text-xs text-studio-text truncate max-w-[180px]" title={art.model_identifier}>
                      {art.model_identifier}
                    </td>

                    {/* Size */}
                    <td className="py-3 px-4 font-mono text-xs text-studio-muted">
                      {formatBytes(art.size_bytes)}
                    </td>

                    {/* Validation Status */}
                    <td className="py-3 px-4">
                      {art.validation_status === 'valid' ? (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-emerald-950/80 text-emerald-400 border border-emerald-800/60" data-testid="badge-valid">
                          <CheckCircle2 className="w-3 h-3" /> Valid
                        </span>
                      ) : art.validation_status === 'invalid' ? (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-rose-950/80 text-rose-400 border border-rose-800/60" data-testid="badge-invalid">
                          <XCircle className="w-3 h-3" /> Invalid
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-zinc-800 text-zinc-400 border border-zinc-700" data-testid="badge-unvalidated">
                          <Clock className="w-3 h-3" /> Unvalidated
                        </span>
                      )}
                    </td>

                    {/* Export Status */}
                    <td className="py-3 px-4">
                      {isExported ? (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-cyan-950/80 text-cyan-400 border border-cyan-800/60">
                          Exported
                        </span>
                      ) : (
                        <span className="text-[11px] text-studio-muted">
                          {art.export_status === 'exporting' ? 'Exporting...' : 'Not exported'}
                        </span>
                      )}
                    </td>

                    {/* Actions */}
                    <td className="py-3 px-4 text-right">
                      <div className="flex items-center justify-end gap-1.5">
                        {/* Inspect Details */}
                        <button
                          onClick={() => {
                            setSelectedArtifact(art);
                            setModalOpen(true);
                          }}
                          className="p-1.5 rounded-md hover:bg-studio-cardHover text-studio-muted hover:text-studio-text transition-colors"
                          title="Inspect Metadata & Files"
                          data-testid="inspect-artifact-btn"
                        >
                          <Eye className="w-3.5 h-3.5" />
                        </button>

                        {/* Validate */}
                        <button
                          onClick={() => handleValidate(art.artifact_id)}
                          disabled={isValidating}
                          className="p-1.5 rounded-md hover:bg-studio-cardHover text-studio-muted hover:text-emerald-400 transition-colors disabled:opacity-40"
                          title="Validate Artifact Integrity"
                          data-testid="validate-artifact-btn"
                        >
                          <ShieldCheck className={`w-3.5 h-3.5 ${isValidating ? 'animate-spin text-emerald-400' : ''}`} />
                        </button>

                        {/* Export */}
                        <button
                          onClick={() => handleExport(art.artifact_id)}
                          disabled={!isValid || isExporting}
                          title={!isValid ? 'Validate artifact before exporting' : 'Create export package'}
                          className="p-1.5 rounded-md hover:bg-studio-cardHover text-studio-muted hover:text-studio-accent transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                          data-testid="export-artifact-btn"
                        >
                          <Box className={`w-3.5 h-3.5 ${isExporting ? 'animate-spin text-studio-accent' : ''}`} />
                        </button>

                        {/* Download */}
                        {isExported && (
                          <a
                            href={getArtifactDownloadUrl(art.artifact_id)}
                            download
                            className="p-1.5 rounded-md hover:bg-studio-cardHover text-cyan-400 transition-colors"
                            title="Download Export Package"
                            data-testid="download-artifact-btn"
                          >
                            <Download className="w-3.5 h-3.5" />
                          </a>
                        )}

                        {/* Evaluate */}
                        <button
                          onClick={() => onNavigateToEvaluation?.(art.artifact_id)}
                          disabled={!isValid}
                          title={!isValid ? 'Validate artifact before evaluating' : 'Evaluate model on validation dataset'}
                          className="px-2 py-1 rounded bg-studio-accent text-studio-bg font-semibold text-[11px] flex items-center gap-1 hover:bg-studio-accent/90 transition-all disabled:opacity-30 disabled:cursor-not-allowed"
                          data-testid="evaluate-artifact-btn"
                        >
                          <Play className="w-2.5 h-2.5 fill-current" />
                          <span>Eval</span>
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Inspect Modal */}
      <ArtifactDetailsModal
        artifact={selectedArtifact}
        isOpen={modalOpen}
        onClose={() => {
          setModalOpen(false);
          setSelectedArtifact(null);
        }}
        onValidate={handleValidate}
        onExport={handleExport}
        onEvaluate={(id) => {
          setModalOpen(false);
          onNavigateToEvaluation?.(id);
        }}
        isValidating={validatingId === selectedArtifact?.artifact_id}
        isExporting={exportingId === selectedArtifact?.artifact_id}
      />
    </div>
  );
};
