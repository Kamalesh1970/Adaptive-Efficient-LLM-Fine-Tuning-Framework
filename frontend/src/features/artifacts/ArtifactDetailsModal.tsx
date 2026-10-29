import React from 'react';
import { 
  X, 
  CheckCircle2, 
  AlertTriangle, 
  XCircle, 
  Download, 
  Play, 
  FileCheck, 
  Box, 
  ShieldCheck, 
  Cpu
} from 'lucide-react';
import { ArtifactMetadata } from '../../types/training';
import { getArtifactDownloadUrl } from '../../services/api';

interface ArtifactDetailsModalProps {
  artifact: ArtifactMetadata | null;
  isOpen: boolean;
  onClose: () => void;
  onValidate: (artifactId: string) => void;
  onExport: (artifactId: string) => void;
  onEvaluate: (artifactId: string) => void;
  isValidating?: boolean;
  isExporting?: boolean;
}

export const ArtifactDetailsModal: React.FC<ArtifactDetailsModalProps> = ({
  artifact,
  isOpen,
  onClose,
  onValidate,
  onExport,
  onEvaluate,
  isValidating = false,
  isExporting = false,
}) => {
  if (!isOpen || !artifact) return null;

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
  };

  const getValidationBadge = () => {
    switch (artifact.validation_status) {
      case 'valid':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-emerald-950/80 text-emerald-400 border border-emerald-800/60" data-testid="modal-status-valid">
            <CheckCircle2 className="w-3.5 h-3.5" /> Validated
          </span>
        );
      case 'invalid':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-rose-950/80 text-rose-400 border border-rose-800/60" data-testid="modal-status-invalid">
            <XCircle className="w-3.5 h-3.5" /> Invalid
          </span>
        );
      case 'corrupted':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-red-950/80 text-red-400 border border-red-800/60" data-testid="modal-status-corrupted">
            <AlertTriangle className="w-3.5 h-3.5" /> Corrupted
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-zinc-800 text-zinc-300 border border-zinc-700" data-testid="modal-status-unvalidated">
            Unvalidated
          </span>
        );
    }
  };

  const getExportBadge = () => {
    switch (artifact.export_status) {
      case 'exported':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-cyan-950/80 text-cyan-400 border border-cyan-800/60">
            Exported
          </span>
        );
      case 'exporting':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-amber-950/80 text-amber-400 border border-amber-800/60">
            Packaging...
          </span>
        );
      case 'export_failed':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-rose-950/80 text-rose-400 border border-rose-800/60">
            Export Failed
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-zinc-800 text-zinc-400 border border-zinc-700">
            Not Exported
          </span>
        );
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4 overflow-y-auto" data-testid="artifact-details-modal">
      <div className="bg-studio-card border border-studio-border rounded-xl shadow-2xl max-w-3xl w-full max-h-[90vh] flex flex-col text-studio-text overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="px-6 py-4 border-b border-studio-border flex items-center justify-between bg-studio-bg/60">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-studio-card border border-studio-border rounded-lg text-studio-accent">
              <Box className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-semibold text-studio-text" data-testid="artifact-modal-id">
                  {artifact.artifact_id}
                </h2>
                {getValidationBadge()}
                {getExportBadge()}
              </div>
              <p className="text-xs text-studio-muted">
                Run ID: <span className="font-mono text-studio-text">{artifact.run_id}</span> • Type: <span className="uppercase font-mono text-[11px] text-studio-accent">{artifact.artifact_type.replace('_', ' ')}</span>
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-studio-muted hover:text-studio-text hover:bg-studio-cardHover transition-colors"
            data-testid="modal-close-btn"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-6 flex-1 text-xs">
          {/* Metadata Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="p-3 bg-studio-bg/60 rounded-lg border border-studio-border space-y-1">
              <span className="text-[11px] text-studio-muted uppercase tracking-wider font-mono">Base Model</span>
              <p className="font-mono text-studio-text truncate font-medium" title={artifact.model_identifier} data-testid="artifact-modal-model">
                {artifact.model_identifier}
              </p>
            </div>
            <div className="p-3 bg-studio-bg/60 rounded-lg border border-studio-border space-y-1">
              <span className="text-[11px] text-studio-muted uppercase tracking-wider font-mono">Method</span>
              <p className="font-mono text-studio-text uppercase font-medium">
                {artifact.fine_tuning_method}
              </p>
            </div>
            <div className="p-3 bg-studio-bg/60 rounded-lg border border-studio-border space-y-1">
              <span className="text-[11px] text-studio-muted uppercase tracking-wider font-mono">Total Size</span>
              <p className="font-mono text-studio-text font-medium" data-testid="artifact-modal-size">
                {formatBytes(artifact.size_bytes)}
              </p>
            </div>
            <div className="p-3 bg-studio-bg/60 rounded-lg border border-studio-border space-y-1">
              <span className="text-[11px] text-studio-muted uppercase tracking-wider font-mono">Created</span>
              <p className="font-mono text-studio-text truncate">
                {new Date(artifact.created_at).toLocaleString()}
              </p>
            </div>
          </div>

          {/* Validation Errors & Warnings */}
          {artifact.validation_errors && artifact.validation_errors.length > 0 && (
            <div className="p-3.5 bg-rose-950/40 border border-rose-800/60 rounded-lg text-rose-300 space-y-1" data-testid="artifact-modal-errors">
              <div className="flex items-center gap-1.5 font-semibold text-rose-400">
                <AlertTriangle className="w-4 h-4" /> Integrity Issues Detected
              </div>
              <ul className="list-disc list-inside space-y-0.5 text-[11px]">
                {artifact.validation_errors.map((err, i) => (
                  <li key={i}>{err}</li>
                ))}
              </ul>
            </div>
          )}

          {artifact.validation_warnings && artifact.validation_warnings.length > 0 && (
            <div className="p-3.5 bg-amber-950/40 border border-amber-800/60 rounded-lg text-amber-300 space-y-1" data-testid="artifact-modal-warnings">
              <div className="flex items-center gap-1.5 font-semibold text-amber-400">
                <AlertTriangle className="w-4 h-4" /> Warnings
              </div>
              <ul className="list-disc list-inside space-y-0.5 text-[11px]">
                {artifact.validation_warnings.map((warn, i) => (
                  <li key={i}>{warn}</li>
                ))}
              </ul>
            </div>
          )}

          {/* File Reference */}
          <div className="space-y-1">
            <span className="text-[11px] font-mono text-studio-muted uppercase">Safe Relative Directory</span>
            <div className="p-2.5 bg-black/40 border border-studio-border rounded-lg font-mono text-emerald-400 break-all select-all" data-testid="artifact-modal-path">
              {artifact.file_reference}
            </div>
          </div>

          {/* File Manifest */}
          {(() => {
            const manifestFiles = artifact.files || [];
            return (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-mono text-studio-muted uppercase flex items-center gap-1.5">
                    <FileCheck className="w-3.5 h-3.5" /> Discovered File Manifest ({manifestFiles.length})
                  </span>
                </div>
                <div className="border border-studio-border rounded-lg overflow-hidden">
                  <table className="w-full text-left border-collapse" data-testid="artifact-files-table">
                    <thead>
                      <tr className="bg-studio-bg/80 border-b border-studio-border text-[11px] font-mono text-studio-muted uppercase">
                        <th className="py-2 px-3">Filename</th>
                        <th className="py-2 px-3">Size</th>
                        <th className="py-2 px-3">SHA-256 Checksum</th>
                        <th className="py-2 px-3 text-right">Required</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-studio-border/50 text-[11px] font-mono">
                      {manifestFiles.map((f, idx) => (
                        <tr key={idx} className="hover:bg-studio-cardHover/40">
                          <td className="py-2 px-3 text-studio-text flex items-center gap-1.5">
                            <span className="w-1.5 h-1.5 rounded-full bg-studio-accent" />
                            {f.filename}
                          </td>
                          <td className="py-2 px-3 text-studio-muted">{formatBytes(f.size_bytes)}</td>
                          <td className="py-2 px-3 text-studio-muted/90 truncate max-w-[200px]" title={f.sha256 || 'Not computed'}>
                            {f.sha256 ? `${f.sha256.slice(0, 12)}...${f.sha256.slice(-6)}` : '—'}
                          </td>
                          <td className="py-2 px-3 text-right">
                            {f.is_required ? (
                              <span className="text-amber-400 text-[10px]">Required</span>
                            ) : (
                              <span className="text-studio-muted text-[10px]">Optional</span>
                            )}
                          </td>
                        </tr>
                      ))}
                      {manifestFiles.length === 0 && (
                        <tr>
                          <td colSpan={4} className="py-4 text-center text-studio-muted">
                            No files discovered in directory.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            );
          })()}

          {/* Adapter Configuration Snapshot */}
          {artifact.adapter_config && (
            <div className="space-y-1.5">
              <span className="text-[11px] font-mono text-studio-muted uppercase flex items-center gap-1">
                <Cpu className="w-3.5 h-3.5" /> Adapter Configuration Metadata
              </span>
              <pre className="p-3 bg-black/50 border border-studio-border rounded-lg text-studio-muted font-mono text-[11px] overflow-x-auto max-h-40">
                {JSON.stringify(artifact.adapter_config, null, 2)}
              </pre>
            </div>
          )}
        </div>

        {/* Footer Actions */}
        <div className="px-6 py-3.5 border-t border-studio-border bg-studio-bg/80 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <button
              onClick={() => onValidate(artifact.artifact_id)}
              disabled={isValidating}
              className="px-3 py-1.5 rounded-lg border border-studio-border bg-studio-card hover:bg-studio-cardHover text-studio-text text-xs font-medium flex items-center gap-1.5 transition-colors disabled:opacity-50"
              data-testid="modal-validate-btn"
            >
              <ShieldCheck className={`w-3.5 h-3.5 text-emerald-400 ${isValidating ? 'animate-spin' : ''}`} />
              <span>{isValidating ? 'Validating...' : 'Validate Integrity'}</span>
            </button>

            <button
              onClick={() => onExport(artifact.artifact_id)}
              disabled={isExporting || artifact.validation_status !== 'valid'}
              title={artifact.validation_status !== 'valid' ? 'Validate artifact before export' : undefined}
              className="px-3 py-1.5 rounded-lg border border-studio-border bg-studio-card hover:bg-studio-cardHover text-studio-text text-xs font-medium flex items-center gap-1.5 transition-colors disabled:opacity-50"
              data-testid="modal-export-btn"
            >
              <Box className={`w-3.5 h-3.5 text-studio-accent ${isExporting ? 'animate-spin' : ''}`} />
              <span>{isExporting ? 'Exporting...' : 'Export Package'}</span>
            </button>

            {artifact.export_status === 'exported' && (
              <a
                href={getArtifactDownloadUrl(artifact.artifact_id)}
                download
                className="px-3 py-1.5 rounded-lg border border-cyan-800/80 bg-cyan-950/60 hover:bg-cyan-900/60 text-cyan-300 text-xs font-medium flex items-center gap-1.5 transition-colors"
                data-testid="modal-download-btn"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Download ZIP</span>
              </a>
            )}
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => onEvaluate(artifact.artifact_id)}
              disabled={artifact.validation_status !== 'valid'}
              title={artifact.validation_status !== 'valid' ? 'Only validated artifacts can be evaluated' : undefined}
              className="px-4 py-1.5 rounded-lg bg-studio-accent text-studio-bg font-semibold text-xs flex items-center gap-1.5 shadow-sm shadow-studio-accent/20 hover:bg-studio-accent/90 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              data-testid="modal-evaluate-btn"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>Evaluate Model</span>
            </button>

            <button
              onClick={onClose}
              className="px-3 py-1.5 rounded-lg border border-studio-border text-studio-muted hover:text-studio-text text-xs transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
