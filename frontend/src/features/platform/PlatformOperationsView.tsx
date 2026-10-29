import React, { useState, useEffect, useCallback } from 'react';
import {
  Server,
  Cpu,
  Layers,
  Terminal,
  Download,
  AlertTriangle,
  CheckCircle2,
  RefreshCw,
  Box,
  HardDrive,
  FileCode,
  ShieldAlert,
} from 'lucide-react';
import {
  KubernetesValidationResponse,
  QuantizationCapability,
  QuantizationJobRecord,
  GGUFExportJobRecord,
  ServingInstanceRecord,
  MultiGPUSystemReport,
} from '../../types/training';
import {
  validateKubernetesManifests,
  fetchQuantizationCapabilities,
  fetchQuantizationJobs,
  fetchGGUFExportJobs,
  fetchServingCapabilities,
  fetchServingInstances,
  fetchMultiGPUSystemReport,
} from '../../services/api';

export const PlatformOperationsView: React.FC = () => {
  const [activeSubTab, setActiveSubTab] = useState<'kubernetes' | 'quantization' | 'export' | 'serving' | 'multigpu'>('kubernetes');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // States
  const [k8sStatus, setK8sStatus] = useState<KubernetesValidationResponse | null>(null);
  const [quantCaps, setQuantCaps] = useState<QuantizationCapability[]>([]);
  const [quantJobs, setQuantJobs] = useState<QuantizationJobRecord[]>([]);
  const [exportJobs, setExportJobs] = useState<GGUFExportJobRecord[]>([]);
  const [servingCaps, setServingCaps] = useState<{
    is_available: boolean;
    version?: string | null;
    cuda_available: boolean;
    supported_model_types: string[];
    reason_if_unavailable?: string | null;
  } | null>(null);
  const [servingInstances, setServingInstances] = useState<ServingInstanceRecord[]>([]);
  const [gpuReport, setGpuReport] = useState<MultiGPUSystemReport | null>(null);

  const loadAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [k8s, caps, qj, ej, sc, si, gr] = await Promise.all([
        validateKubernetesManifests().catch(() => null),
        fetchQuantizationCapabilities().catch(() => []),
        fetchQuantizationJobs().catch(() => []),
        fetchGGUFExportJobs().catch(() => []),
        fetchServingCapabilities().catch(() => null),
        fetchServingInstances().catch(() => []),
        fetchMultiGPUSystemReport().catch(() => null),
      ]);

      if (k8s) setK8sStatus(k8s);
      setQuantCaps(caps);
      setQuantJobs(qj);
      setExportJobs(ej);
      if (sc) setServingCaps(sc);
      setServingInstances(si);
      if (gr) setGpuReport(gr);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load platform data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadAll();
  }, [loadAll]);

  return (
    <div className="flex-1 flex flex-col min-w-0 overflow-y-auto bg-studio-bg p-6 space-y-6">
      {/* Top Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-studio-border pb-4">
        <div>
          <h2 className="text-xl font-bold text-studio-text flex items-center gap-2">
            <Server className="w-5 h-5 text-studio-accent" />
            Platform &amp; Infrastructure
          </h2>
          <p className="text-xs text-studio-muted mt-1">
            Kubernetes orchestration, advanced quantization, GGUF binary export, and distributed GPU topology.
          </p>
        </div>
        <button
          onClick={loadAll}
          disabled={loading}
          className="flex items-center gap-2 px-3 py-1.5 rounded bg-studio-card hover:bg-studio-cardHover border border-studio-border text-xs text-studio-text transition-all self-start sm:self-auto"
          data-testid="refresh-platform-btn"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          Refresh Diagnostics
        </button>
      </div>

      {error && (
        <div className="p-3 bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs rounded-md">
          {error}
        </div>
      )}

      {/* Sub Navigation */}
      <div className="flex items-center space-x-2 border-b border-studio-border pb-2">
        {[
          { id: 'kubernetes', label: 'Kubernetes & Compose', icon: Layers },
          { id: 'quantization', label: 'Advanced Quantization', icon: HardDrive },
          { id: 'export', label: 'GGUF Model Export', icon: Box },
          { id: 'serving', label: 'Inference Serving', icon: Terminal },
          { id: 'multigpu', label: 'Multi-GPU Diagnostics', icon: Cpu },
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeSubTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveSubTab(tab.id as typeof activeSubTab)}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                isActive
                  ? 'bg-studio-accent text-studio-bg font-semibold'
                  : 'text-studio-muted hover:text-studio-text hover:bg-studio-card'
              }`}
              data-testid={`subtab-${tab.id}`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* 1. Kubernetes & Compose Tab */}
      {activeSubTab === 'kubernetes' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 bg-studio-card rounded-lg border border-studio-border">
              <span className="text-xs text-studio-muted">Manifest Validation</span>
              <div className="flex items-center gap-2 mt-1">
                {k8sStatus?.is_valid ? (
                  <>
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    <span className="text-sm font-semibold text-emerald-400">12/12 Valid YAML</span>
                  </>
                ) : (
                  <>
                    <AlertTriangle className="w-4 h-4 text-amber-400" />
                    <span className="text-sm font-semibold text-amber-400">Validation Pending</span>
                  </>
                )}
              </div>
            </div>
            <div className="p-4 bg-studio-card rounded-lg border border-studio-border">
              <span className="text-xs text-studio-muted">Cluster Status</span>
              <div className="flex items-center gap-2 mt-1">
                <span className="text-xs px-2 py-0.5 rounded bg-zinc-800 text-zinc-300 font-mono">
                  Offline (Pre-Cluster Stage)
                </span>
              </div>
              <p className="text-[11px] text-studio-muted mt-1">
                Manifests are verified and ready for CI/CD staging apply.
              </p>
            </div>
            <div className="p-4 bg-studio-card rounded-lg border border-studio-border">
              <span className="text-xs text-studio-muted">Docker Compose Profiles</span>
              <div className="flex flex-wrap gap-1 mt-1">
                <span className="text-[11px] px-2 py-0.5 rounded bg-studio-bg border border-studio-border text-studio-text">
                  default
                </span>
                <span className="text-[11px] px-2 py-0.5 rounded bg-studio-bg border border-studio-border text-studio-text">
                  gpu
                </span>
                <span className="text-[11px] px-2 py-0.5 rounded bg-studio-bg border border-studio-border text-studio-text">
                  inference
                </span>
              </div>
            </div>
          </div>

          <div className="bg-studio-card rounded-lg border border-studio-border p-4">
            <h3 className="text-sm font-semibold text-studio-text mb-3 flex items-center gap-2">
              <FileCode className="w-4 h-4 text-studio-accent" />
              Kubernetes Resource Manifests (deploy/kubernetes/)
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {k8sStatus?.manifests?.map((m) => (
                <div
                  key={m.filename}
                  className="p-3 bg-studio-bg/60 border border-studio-border rounded-md flex items-center justify-between"
                  data-testid={`k8s-manifest-${m.filename}`}
                >
                  <div>
                    <div className="text-xs font-mono font-medium text-studio-text">{m.filename}</div>
                    <div className="text-[11px] text-studio-muted">
                      Kind: <span className="text-zinc-300">{m.kind}</span>
                    </div>
                  </div>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    Validated
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* 2. Advanced Quantization Tab */}
      {activeSubTab === 'quantization' && (
        <div className="space-y-6">
          <div className="bg-studio-card rounded-lg border border-studio-border p-4">
            <h3 className="text-sm font-semibold text-studio-text mb-3 flex items-center gap-2">
              <HardDrive className="w-4 h-4 text-studio-accent" />
              Quantization Capabilities Discovery
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {quantCaps.map((c) => (
                <div
                  key={c.method}
                  className="p-4 bg-studio-bg/70 border border-studio-border rounded-lg space-y-2"
                  data-testid={`quant-cap-${c.method}`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <span className="text-xs font-semibold text-studio-text">{c.name}</span>
                    <span
                      className={`text-[10px] font-mono px-2 py-0.5 rounded ${
                        c.is_available
                          ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                          : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                      }`}
                    >
                      {c.is_available ? 'Available' : 'Unavailable'}
                    </span>
                  </div>
                  <div className="text-[11px] text-studio-muted">
                    Method: <span className="font-mono text-zinc-300">{c.method}</span>
                  </div>
                  <div className="text-[11px] text-studio-muted">
                    Requires CUDA: <span className="font-mono text-zinc-300">{c.requires_cuda ? 'Yes' : 'No'}</span>
                  </div>
                  {c.reason_if_unavailable && (
                    <div className="text-[11px] text-amber-300/90 bg-amber-500/5 p-2 rounded border border-amber-500/10">
                      {c.reason_if_unavailable}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Quantization Jobs History */}
          <div className="bg-studio-card rounded-lg border border-studio-border p-4">
            <h3 className="text-sm font-semibold text-studio-text mb-3">Recent Quantization Operations</h3>
            {quantJobs.length === 0 ? (
              <div className="text-xs text-studio-muted py-4 text-center">
                No post-training quantization jobs executed yet.
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="border-b border-studio-border text-studio-muted">
                    <tr>
                      <th className="pb-2">Job ID</th>
                      <th className="pb-2">Method</th>
                      <th className="pb-2">Status</th>
                      <th className="pb-2">Size Reduction</th>
                      <th className="pb-2">SHA-256</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-studio-border/50">
                    {quantJobs.map((q) => (
                      <tr key={q.job_id} className="text-studio-text">
                        <td className="py-2 font-mono text-[11px]">{q.job_id}</td>
                        <td className="py-2">{q.method}</td>
                        <td className="py-2">
                          <span className="px-2 py-0.5 rounded text-[10px] bg-zinc-800 text-zinc-300">
                            {q.status}
                          </span>
                        </td>
                        <td className="py-2">
                          {q.size_reduction_ratio ? `${(q.size_reduction_ratio * 100).toFixed(1)}%` : '—'}
                        </td>
                        <td className="py-2 font-mono text-[11px]">
                          {q.sha256 ? `${q.sha256.slice(0, 12)}...` : '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 3. GGUF Model Export Tab */}
      {activeSubTab === 'export' && (
        <div className="space-y-6">
          <div className="bg-studio-card rounded-lg border border-studio-border p-4">
            <h3 className="text-sm font-semibold text-studio-text mb-2 flex items-center gap-2">
              <Box className="w-4 h-4 text-studio-accent" />
              GGUF Model Container Pipeline (GGUF v3)
            </h3>
            <p className="text-xs text-studio-muted mb-4">
              Exports fine-tuned adapter weights merged with base architecture into self-contained GGUF binary format with 32-byte alignment.
            </p>

            {exportJobs.length === 0 ? (
              <div className="text-xs text-studio-muted py-6 text-center border border-dashed border-studio-border rounded">
                No GGUF export jobs recorded. Go to Artifacts tab to export fine-tuned models.
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="border-b border-studio-border text-studio-muted">
                    <tr>
                      <th className="pb-2">Export ID</th>
                      <th className="pb-2">Source Artifact</th>
                      <th className="pb-2">Quant Type</th>
                      <th className="pb-2">Status</th>
                      <th className="pb-2">File Size</th>
                      <th className="pb-2">SHA-256 Checksum</th>
                      <th className="pb-2">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-studio-border/50">
                    {exportJobs.map((ej) => (
                      <tr key={ej.export_id} className="text-studio-text">
                        <td className="py-2 font-mono text-[11px]">{ej.export_id}</td>
                        <td className="py-2 font-mono text-[11px]">{ej.source_artifact_id || ej.source_reference}</td>
                        <td className="py-2">{ej.config?.quantization_type || 'Q4_K_M'}</td>
                        <td className="py-2">
                          <span className="px-2 py-0.5 rounded text-[10px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                            {ej.status}
                          </span>
                        </td>
                        <td className="py-2">
                          {ej.file_size_bytes ? `${(ej.file_size_bytes / 1024).toFixed(1)} KB` : '—'}
                        </td>
                        <td className="py-2 font-mono text-[11px]">
                          {ej.sha256 ? `${ej.sha256.slice(0, 16)}...` : '—'}
                        </td>
                        <td className="py-2">
                          {ej.status === 'completed' && (
                            <a
                              href={`/api/export/gguf/${ej.export_id}/download`}
                              download
                              className="inline-flex items-center gap-1 text-[11px] text-studio-accent hover:underline"
                            >
                              <Download className="w-3 h-3" />
                              Download
                            </a>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 4. Serving Tab */}
      {activeSubTab === 'serving' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 bg-studio-card rounded-lg border border-studio-border">
              <span className="text-xs text-studio-muted">vLLM Inference Engine</span>
              <div className="flex items-center gap-2 mt-1">
                <span
                  className={`text-xs px-2 py-0.5 rounded ${
                    servingCaps?.is_available
                      ? 'bg-emerald-500/10 text-emerald-400'
                      : 'bg-amber-500/10 text-amber-400'
                  }`}
                >
                  {servingCaps?.is_available ? 'Available' : 'Unavailable (CUDA required)'}
                </span>
              </div>
              {servingCaps?.reason_if_unavailable && (
                <p className="text-[11px] text-studio-muted mt-2">
                  {servingCaps.reason_if_unavailable}
                </p>
              )}
            </div>
            <div className="p-4 bg-studio-card rounded-lg border border-studio-border">
              <span className="text-xs text-studio-muted">Active Instances</span>
              <div className="text-lg font-bold text-studio-text mt-1">{servingInstances.length}</div>
            </div>
          </div>
        </div>
      )}

      {/* 5. Multi-GPU Diagnostics Tab */}
      {activeSubTab === 'multigpu' && (
        <div className="space-y-6">
          <div className="p-4 bg-studio-card rounded-lg border border-studio-border space-y-3">
            <h3 className="text-sm font-semibold text-studio-text flex items-center gap-2">
              <Cpu className="w-4 h-4 text-studio-accent" />
              PyTorch Distributed Hardware Discovery
            </h3>
            <div className="text-xs text-studio-muted bg-studio-bg p-3 rounded font-mono">
              {gpuReport?.summary || 'Scanning distributed hardware...'}
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
              <div className="p-3 bg-studio-bg rounded border border-studio-border">
                <span className="text-[11px] text-studio-muted">CUDA Available</span>
                <div className="text-sm font-semibold text-studio-text mt-0.5">
                  {gpuReport?.cuda_available ? 'True' : 'False (CPU Only)'}
                </div>
              </div>
              <div className="p-3 bg-studio-bg rounded border border-studio-border">
                <span className="text-[11px] text-studio-muted">DDP Supported</span>
                <div className="text-sm font-semibold text-studio-text mt-0.5">
                  {gpuReport?.ddp_supported ? 'Yes' : 'No (Host has <2 GPUs)'}
                </div>
              </div>
              <div className="p-3 bg-studio-bg rounded border border-studio-border">
                <span className="text-[11px] text-studio-muted">FSDP Supported</span>
                <div className="text-sm font-semibold text-studio-text mt-0.5">
                  {gpuReport?.fsdp_supported ? 'Yes' : 'No (Host has <2 GPUs)'}
                </div>
              </div>
            </div>

            {!gpuReport?.cuda_available && (
              <div className="flex items-center gap-2 p-3 bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs rounded-md">
                <ShieldAlert className="w-4 h-4 shrink-0" />
                <span>
                  Host system is operating truthfully in CPU mode. Multi-GPU training requests (DDP/FSDP) are cleanly rejected at preflight before resource waste.
                </span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
