import React from 'react';
import { 
  Cpu, 
  HardDrive, 
  Layers, 
  Clock, 
  Sparkles, 
  AlertTriangle,
  Lock
} from 'lucide-react';
import { Badge } from '../../components/common/Badge';
import { TrainingConfig, SystemStatus, ValidationResult } from '../../types/training';

interface RunPreviewPanelProps {
  config: TrainingConfig;
  systemStatus: SystemStatus | null;
  backendOnline: boolean;
  validationResult: ValidationResult | null;
  onStartTraining?: () => void;
}

export const RunPreviewPanel: React.FC<RunPreviewPanelProps> = ({
  config,
  systemStatus,
  backendOnline,
  validationResult,
  onStartTraining,
}) => {
  const hasModel = Boolean(config.model_identifier);
  const hasDataset = Boolean(config.dataset_reference);
  const isReady = Boolean(validationResult?.is_ready_for_training); // Strictly false in Phase 0

  const methodLabels: Record<string, string> = {
    lora: 'LoRA (16-bit)',
    qlora: 'QLoRA (4-bit NF4)',
    full_finetuning: 'Full Weights',
  };

  const hardwareName = backendOnline && systemStatus
    ? systemStatus.cuda_available
      ? systemStatus.primary_gpu_name || 'CUDA GPU Detected'
      : 'CPU Only (No GPU)'
    : 'Unknown (Backend Offline)';

  const vramDisplay = backendOnline && systemStatus
    ? systemStatus.cuda_available && systemStatus.primary_gpu_vram_gb
      ? `${systemStatus.primary_gpu_vram_gb.toFixed(1)} GB VRAM`
      : 'N/A (CPU Mode)'
    : 'Unavailable';

  const hasHfToken = config.hf_token_status === 'present_in_env';

  return (
    <div
      className="w-full lg:w-80 shrink-0 bg-studio-card border border-studio-cardBorder rounded-xl p-4 flex flex-col justify-between shadow-lg shadow-black/30"
      data-testid="run-preview-panel"
    >
      <div className="space-y-4">
        {/* Header */}
        <div className="border-b border-studio-border/60 pb-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold text-studio-text tracking-wider uppercase font-mono flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-studio-accent" />
              Run Summary Preview
            </h3>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-studio-input border border-studio-border text-studio-muted font-mono">
              Draft
            </span>
          </div>
          <p className="text-[11px] text-studio-muted mt-1">
            Real-time verification of training readiness
          </p>
        </div>

        {/* Configuration Parameters Checklist */}
        <div className="space-y-2.5 text-xs">
          {/* Model Status */}
          <div className="flex items-start justify-between py-1 border-b border-studio-border/30">
            <span className="text-studio-muted text-[11px] flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-studio-muted" /> Model:
            </span>
            <div className="text-right">
              {hasModel ? (
                <span className="font-mono text-studio-text text-[11px] max-w-[140px] truncate block" title={config.model_identifier!}>
                  {config.model_identifier}
                </span>
              ) : (
                <span className="text-amber-400 font-medium text-[11px]" data-testid="status-model-none">
                  None Selected
                </span>
              )}
            </div>
          </div>

          {/* Dataset Status */}
          <div className="flex items-start justify-between py-1 border-b border-studio-border/30">
            <span className="text-studio-muted text-[11px] flex items-center gap-1.5">
              <HardDrive className="w-3.5 h-3.5 text-studio-muted" /> Dataset:
            </span>
            <div className="text-right">
              {hasDataset ? (
                <span className="font-mono text-studio-text text-[11px] max-w-[140px] truncate block" title={config.dataset_reference!}>
                  {config.dataset_reference}
                </span>
              ) : (
                <span className="text-amber-400 font-medium text-[11px]" data-testid="status-dataset-none">
                  None Selected
                </span>
              )}
            </div>
          </div>

          {/* Fine-Tuning Method */}
          <div className="flex items-center justify-between py-1 border-b border-studio-border/30">
            <span className="text-studio-muted text-[11px]">Method:</span>
            <span className="font-mono text-studio-text text-[11px] font-medium text-studio-accent">
              {methodLabels[config.training_method] || config.training_method}
            </span>
          </div>

          {/* Training Length */}
          <div className="flex items-center justify-between py-1 border-b border-studio-border/30">
            <span className="text-studio-muted text-[11px] flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-studio-muted" /> Duration:
            </span>
            <span className="font-mono text-studio-text text-[11px]">
              {config.training_length} {config.training_length_type}
            </span>
          </div>

          {/* Batch Size */}
          <div className="flex items-center justify-between py-1 border-b border-studio-border/30">
            <span className="text-studio-muted text-[11px]">Batch Size:</span>
            <span className="font-mono text-studio-text text-[11px]">
              {config.batch_size} (per device)
            </span>
          </div>

          {/* Context Length */}
          <div className="flex items-center justify-between py-1 border-b border-studio-border/30">
            <span className="text-studio-muted text-[11px]">Context Length:</span>
            <span className="font-mono text-studio-text text-[11px]">
              {config.context_length} tokens
            </span>
          </div>

          {/* Learning Rate */}
          <div className="flex items-center justify-between py-1 border-b border-studio-border/30">
            <span className="text-studio-muted text-[11px]">Learning Rate:</span>
            <span className="font-mono text-studio-text text-[11px]">
              {config.learning_rate}
            </span>
          </div>

          {/* Advanced Settings */}
          <div className="flex items-center justify-between py-1 border-b border-studio-border/30">
            <span className="text-studio-muted text-[11px]">Advanced Mode:</span>
            <span className="text-studio-muted text-[11px]">
              {config.advanced_enabled ? 'Active' : 'Default Preset'}
            </span>
          </div>
        </div>

        {/* Hardware Status Breakdown */}
        <div className="p-3 rounded-lg bg-studio-input/80 border border-studio-border space-y-2 text-xs">
          <div className="text-[10px] font-bold text-studio-muted uppercase tracking-wider font-mono flex items-center justify-between">
            <span className="flex items-center gap-1">
              <Cpu className="w-3 h-3 text-studio-teal" /> Hardware Target
            </span>
            <Badge variant={backendOnline && systemStatus?.cuda_available ? 'success' : 'default'}>
              {backendOnline && systemStatus?.cuda_available ? 'GPU Accelerated' : 'CPU Mode'}
            </Badge>
          </div>

          <div className="text-[11px] space-y-1 font-mono">
            <div className="flex justify-between text-studio-text">
              <span className="text-studio-muted">Device:</span>
              <span className="truncate max-w-[150px]" title={hardwareName}>{hardwareName}</span>
            </div>
            <div className="flex justify-between text-studio-text">
              <span className="text-studio-muted">VRAM:</span>
              <span className={systemStatus?.cuda_available ? 'text-studio-accent font-semibold' : 'text-studio-muted'}>
                {vramDisplay}
              </span>
            </div>
            <div className="flex justify-between text-studio-text">
              <span className="text-studio-muted">HF Token:</span>
              <span className={hasHfToken ? 'text-emerald-400' : 'text-zinc-500'}>
                {hasHfToken ? 'Active' : 'Not Set'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Action Area & Disabled State Notice */}
      <div className="pt-4 border-t border-studio-border/60 mt-4 space-y-2.5">
        <button
          type="button"
          disabled={!isReady}
          onClick={onStartTraining}
          className={`w-full py-2.5 px-4 rounded-xl text-xs font-semibold flex items-center justify-center space-x-2 transition-all shadow-md ${
            isReady
              ? 'bg-studio-accent hover:bg-studio-accentHover text-studio-bg shadow-studio-accent/20 cursor-pointer'
              : 'bg-studio-cardBorder/60 text-studio-muted border border-studio-border/60 cursor-not-allowed opacity-75'
          }`}
          data-testid="btn-start-training"
          title={
            !hasModel
              ? 'Select a model first'
              : !hasDataset
              ? 'Select a dataset first'
              : 'Phase 0 UI Skeleton: Fine-tuning engine starts in Phase 4'
          }
        >
          <Lock className="w-3.5 h-3.5 mr-1" />
          <span>Start Fine-Tuning</span>
        </button>

        <div className="p-2 rounded bg-studio-bg/60 border border-studio-border text-[10px] text-studio-muted leading-relaxed" data-testid="phase0-preview-notice">
          <p className="flex items-center gap-1 font-semibold text-studio-text">
            <AlertTriangle className="w-3 h-3 text-amber-400 shrink-0" />
            Phase 0 Prototype
          </p>
          <p className="mt-0.5">
            The start button is disabled until models and datasets are selected. Actual ML execution loops will be enabled in Phase 4.
          </p>
        </div>
      </div>
    </div>
  );
};
