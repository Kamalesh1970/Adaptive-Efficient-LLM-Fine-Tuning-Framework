import React from 'react';
import {
  Cpu,
  HardDrive,
  Layers,
  Clock,
  Sparkles,
  AlertTriangle,
  Lock,
  CheckCircle2,
  XCircle,
  Sliders,
  Play,
  Loader2,
  Package,
} from 'lucide-react';
import { Badge } from '../../components/common/Badge';
import {
  TrainingConfig,
  SystemStatus,
  ValidationResult,
  ModelMetadata,
  ModelValidationResponse,
  DatasetValidationReport,
  DatasetSplitResult,
  PreflightCheckResponse,
  RunPlan,
} from '../../types/training';

interface RunPreviewPanelProps {
  config: TrainingConfig;
  systemStatus: SystemStatus | null;
  backendOnline: boolean;
  validationResult: ValidationResult | null;
  selectedModel?: ModelMetadata | null;
  modelValidation?: ModelValidationResponse | null;
  selectedDatasetReport?: DatasetValidationReport | null;
  splitResult?: DatasetSplitResult | null;
  hfToken?: string;
  isLaunching?: boolean;
  isValidating?: boolean;
  preflightResult?: PreflightCheckResponse | null;
  runPlan?: RunPlan | null;
  launchStatus?: 'incomplete' | 'validating' | 'blocked' | 'ready' | 'launching' | 'failed' | 'created';
  launchError?: string | null;
  createdRunId?: string | null;
  onStartTraining?: () => void;
}

export const RunPreviewPanel: React.FC<RunPreviewPanelProps> = ({
  config,
  systemStatus,
  backendOnline,
  validationResult,
  selectedModel,
  modelValidation,
  selectedDatasetReport,
  splitResult,
  hfToken = '',
  isLaunching = false,
  isValidating = false,
  preflightResult = null,
  launchError = null,
  createdRunId = null,
  onStartTraining,
}) => {
  const hasModel = Boolean(config.model_identifier);
  const hasDataset = Boolean(config.dataset_reference);
  const isReady = Boolean(validationResult?.is_ready_for_training); // Always false in Phase 3

  const methodLabels: Record<string, string> = {
    lora: 'LoRA (Low-Rank Adaptation)',
    qlora: 'QLoRA (Quantized LoRA)',
    full_finetuning: 'Full Weights Fine-Tuning',
  };

  // Effective batch size calculation:
  // per_device_train_batch_size × gradient_accumulation_steps × number_of_devices
  const perDeviceBatchSize = config.per_device_train_batch_size ?? config.batch_size ?? 1;
  const gradAccum = config.gradient_accumulation_steps ?? 1;
  const effectivePerDevice = perDeviceBatchSize * gradAccum;

  const detectedGpuCount =
    backendOnline && (systemStatus?.gpu_available || systemStatus?.cuda_available) && systemStatus.gpu_count > 0
      ? systemStatus.gpu_count
      : (validationResult?.detected_device_count ?? null);

  const totalEffectiveBatch =
    detectedGpuCount !== null && detectedGpuCount > 0
      ? effectivePerDevice * detectedGpuCount
      : null;

  const hardwareName =
    backendOnline && systemStatus
      ? (systemStatus.gpu_available || systemStatus.cuda_available)
        ? systemStatus.gpu_name || systemStatus.primary_gpu_name || 'CUDA GPU Detected'
        : 'CPU Only (No GPU Detected)'
      : 'Backend Unavailable';

  const hasEnvToken = config.hf_token_status === 'present_in_env';
  const hasInMemoryToken = Boolean(hfToken && hfToken.trim());
  const hasHfToken = hasInMemoryToken || hasEnvToken;

  // Consolidated Warnings
  const warningsList: string[] = [];
  if (validationResult?.warnings) {
    warningsList.push(...validationResult.warnings);
  }
  if (modelValidation?.warnings) {
    warningsList.push(...modelValidation.warnings);
  }
  if (selectedDatasetReport?.warnings) {
    warningsList.push(...selectedDatasetReport.warnings);
  }
  if (preflightResult?.checks) {
    for (const c of preflightResult.checks) {
      if (c.status === 'warn') {
        warningsList.push(c.message);
      }
    }
  }
  const uniqueWarnings = Array.from(new Set(warningsList));
  const warningsCount = uniqueWarnings.length;

  // Launch Readiness Calculation
  let launchReadiness: 'Validated' | 'Invalid' | 'Unsupported' | 'Unknown' | 'Not yet checked';
  let readinessDescription = '';

  if (!backendOnline) {
    launchReadiness = 'Unknown';
    readinessDescription = 'FastAPI backend unavailable';
  } else if (!hasModel || !hasDataset) {
    launchReadiness = 'Not yet checked';
    readinessDescription = 'Model and dataset selection required';
  } else if (validationResult && !validationResult.is_valid) {
    launchReadiness = 'Invalid';
    readinessDescription = `${validationResult.errors.length} configuration error(s)`;
  } else if (modelValidation && !modelValidation.is_valid) {
    if (modelValidation.compatibility_status === 'incompatible') {
      launchReadiness = 'Unsupported';
      readinessDescription = `Model incompatible with ${config.training_method}`;
    } else {
      launchReadiness = 'Invalid';
      readinessDescription = 'Model authentication / access requirements not met';
    }
  } else if (selectedDatasetReport && !selectedDatasetReport.is_valid) {
    launchReadiness = 'Invalid';
    readinessDescription = `Dataset contains ${selectedDatasetReport.invalid_sample_count} invalid sample(s)`;
  } else if (config.training_method === 'qlora' && systemStatus && !systemStatus.gpu_available && !systemStatus.cuda_available) {
    launchReadiness = 'Unsupported';
    readinessDescription = 'QLoRA 4-bit fine-tuning requires CUDA GPU';
  } else if (preflightResult) {
    if (preflightResult.can_launch) {
      launchReadiness = 'Validated';
      readinessDescription = 'All preflight resource checks passed';
    } else {
      launchReadiness = 'Invalid';
      readinessDescription = `${preflightResult.blocking_count} blocking preflight failure(s)`;
    }
  } else if (isReady) {
    launchReadiness = 'Validated';
    readinessDescription = 'Ready for preflight validation and launch';
  } else {
    launchReadiness = 'Not yet checked';
    readinessDescription = 'Awaiting validation checks';
  }

  // Dependency Compatibility Calculation
  let depCompatStatus: 'Validated' | 'Invalid' | 'Unsupported' | 'Unknown' | 'Not yet checked';
  let depCompatMessage = '';

  if (!backendOnline) {
    depCompatStatus = 'Unknown';
    depCompatMessage = 'Backend offline';
  } else if (preflightResult) {
    const mlCheck = preflightResult.checks.find((c) => c.check_id === 'ml_core_deps');
    const bnbCheck = preflightResult.checks.find((c) => c.check_id === 'bitsandbytes');

    if (mlCheck?.status === 'fail' || bnbCheck?.status === 'fail') {
      depCompatStatus = 'Invalid';
      depCompatMessage = mlCheck?.status === 'fail' ? mlCheck.message : (bnbCheck?.message || 'Missing dependencies');
    } else if (config.training_method === 'qlora' && systemStatus && !systemStatus.gpu_available && !systemStatus.cuda_available) {
      depCompatStatus = 'Unsupported';
      depCompatMessage = 'QLoRA requires CUDA GPU acceleration';
    } else {
      depCompatStatus = 'Validated';
      depCompatMessage = 'PyTorch, transformers, peft verified';
    }
  } else if (config.training_method === 'qlora' && systemStatus && !systemStatus.gpu_available && !systemStatus.cuda_available) {
    depCompatStatus = 'Unsupported';
    depCompatMessage = 'QLoRA requires CUDA GPU acceleration';
  } else if (systemStatus?.pytorch_available) {
    depCompatStatus = 'Validated';
    depCompatMessage = `PyTorch ${systemStatus.pytorch_version || 'installed'}`;
  } else if (systemStatus && !systemStatus.pytorch_available) {
    depCompatStatus = 'Invalid';
    depCompatMessage = 'PyTorch not installed in backend runtime';
  } else {
    depCompatStatus = 'Not yet checked';
    depCompatMessage = 'Awaiting preflight check';
  }

  // Launch Control State Determination
  let launchControlState:
    | 'Configuration incomplete'
    | 'Validation in progress'
    | 'Blocked by validation'
    | 'Ready for confirmation'
    | 'Launching'
    | 'Launch failed'
    | 'Run created';

  if (createdRunId) {
    launchControlState = 'Run created';
  } else if (isLaunching) {
    launchControlState = 'Launching';
  } else if (launchError) {
    launchControlState = 'Launch failed';
  } else if (isValidating) {
    launchControlState = 'Validation in progress';
  } else if (!hasModel || !hasDataset) {
    launchControlState = 'Configuration incomplete';
  } else if (
    launchReadiness === 'Invalid' ||
    launchReadiness === 'Unsupported' ||
    (validationResult && !validationResult.is_valid)
  ) {
    launchControlState = 'Blocked by validation';
  } else if (isReady || launchReadiness === 'Validated') {
    launchControlState = 'Ready for confirmation';
  } else {
    launchControlState = 'Configuration incomplete';
  }

  const getReadinessVariant = (status: 'Validated' | 'Invalid' | 'Unsupported' | 'Unknown' | 'Not yet checked') => {
    switch (status) {
      case 'Validated':
        return 'success';
      case 'Invalid':
      case 'Unsupported':
        return 'error';
      case 'Unknown':
        return 'warning';
      case 'Not yet checked':
      default:
        return 'default';
    }
  };

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
              Phase 6
            </span>
          </div>
          <p className="text-[11px] text-studio-muted mt-1">
            Real-time verification of training parameters
          </p>
        </div>

        {/* Configuration Parameters Checklist */}
        <div className="space-y-2.5 text-xs">
          {/* Validation Status */}
          <div className="flex items-center justify-between py-1 border-b border-studio-border/30">
            <span className="text-studio-muted text-[11px]">Validation:</span>
            <div>
              {validationResult ? (
                validationResult.is_valid ? (
                  <span
                    className="text-emerald-400 font-mono text-[11px] inline-flex items-center gap-1 font-medium"
                    data-testid="validation-status-valid"
                  >
                    <CheckCircle2 className="w-3 h-3" /> Valid Config
                  </span>
                ) : (
                  <span
                    className="text-rose-400 font-mono text-[11px] inline-flex items-center gap-1 font-medium"
                    data-testid="validation-status-invalid"
                  >
                    <XCircle className="w-3 h-3" /> {validationResult.errors.length} Error(s)
                  </span>
                )
              ) : (
                <span className="text-studio-muted text-[11px] font-mono">Unchecked</span>
              )}
            </div>
          </div>

          {/* Launch Readiness Status */}
          <div className="flex items-center justify-between py-1 border-b border-studio-border/30">
            <span className="text-studio-muted text-[11px] flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-studio-muted" /> Readiness:
            </span>
            <div className="text-right">
              <Badge
                variant={getReadinessVariant(launchReadiness)}
                data-testid="preview-launch-readiness"
              >
                {launchReadiness}
              </Badge>
              {readinessDescription && (
                <span className="block text-[10px] text-studio-muted font-mono truncate max-w-[150px] text-right mt-0.5" title={readinessDescription}>
                  {readinessDescription}
                </span>
              )}
            </div>
          </div>

          {/* Dependency Compatibility */}
          <div className="flex items-center justify-between py-1 border-b border-studio-border/30">
            <span className="text-studio-muted text-[11px] flex items-center gap-1.5">
              <Package className="w-3.5 h-3.5 text-studio-muted" /> Dependencies:
            </span>
            <div className="text-right">
              <Badge
                variant={getReadinessVariant(depCompatStatus)}
                data-testid="preview-dependency-compat"
              >
                {depCompatStatus}
              </Badge>
              {depCompatMessage && (
                <span className="block text-[10px] text-studio-muted font-mono truncate max-w-[140px] text-right mt-0.5" title={depCompatMessage}>
                  {depCompatMessage}
                </span>
              )}
            </div>
          </div>

          {/* Validation Warnings */}
          <div className="flex items-center justify-between py-1 border-b border-studio-border/30">
            <span className="text-studio-muted text-[11px] flex items-center gap-1.5">
              <AlertTriangle className="w-3.5 h-3.5 text-studio-muted" /> Warnings:
            </span>
            <div className="text-right" data-testid="preview-warnings">
              {warningsCount > 0 ? (
                <Badge variant="warning">
                  {warningsCount} Warning(s)
                </Badge>
              ) : (
                <span className="font-mono text-studio-muted text-[11px]">None</span>
              )}
            </div>
          </div>

          {/* Model Status */}
          <div className="flex items-start justify-between py-1 border-b border-studio-border/30">
            <span className="text-studio-muted text-[11px] flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-studio-muted" /> Model:
            </span>
            <div className="text-right">
              {hasModel ? (
                <div>
                  <span
                    className="font-mono text-studio-text text-[11px] max-w-[140px] truncate block font-medium"
                    title={config.model_identifier!}
                    data-testid="preview-model-identifier"
                  >
                    {config.model_identifier}
                  </span>
                  {selectedModel && (
                    <span className="text-[10px] text-studio-muted font-mono block">
                      {selectedModel.parameter_count || 'Unknown size'} • {selectedModel.family || 'Unknown'}
                    </span>
                  )}
                  {modelValidation && (
                    <div className="mt-0.5">
                      {modelValidation.is_valid ? (
                        <span className="text-[10px] text-emerald-400 font-mono inline-flex items-center gap-0.5">
                          <CheckCircle2 className="w-2.5 h-2.5" /> Compatible
                        </span>
                      ) : modelValidation.compatibility_status === 'requires_credentials' ? (
                        <span className="text-[10px] text-amber-400 font-mono inline-flex items-center gap-0.5">
                          <Lock className="w-2.5 h-2.5" /> Needs Token
                        </span>
                      ) : (
                        <span className="text-[10px] text-rose-400 font-mono inline-flex items-center gap-0.5">
                          <XCircle className="w-2.5 h-2.5" /> Incompatible
                        </span>
                      )}
                    </div>
                  )}
                </div>
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
                <div>
                  <span
                    className="font-mono text-studio-text text-[11px] max-w-[140px] truncate block font-medium"
                    title={config.dataset_reference!}
                    data-testid="preview-dataset-reference"
                  >
                    {selectedDatasetReport ? selectedDatasetReport.original_filename : config.dataset_reference}
                  </span>
                  {selectedDatasetReport && (
                    <span className="text-[10px] text-studio-muted font-mono block">
                      {selectedDatasetReport.valid_sample_count} valid • {selectedDatasetReport.detected_schema || 'unknown'}
                    </span>
                  )}
                  {splitResult && (
                    <span className="text-[10px] text-studio-teal font-mono block">
                      Splits: {splitResult.train_count} train / {splitResult.eval_count} eval
                    </span>
                  )}
                </div>
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
            <span
              className="font-mono text-studio-text text-[11px] font-medium text-studio-accent"
              data-testid="preview-method"
            >
              {methodLabels[config.training_method] || config.training_method}
            </span>
          </div>

          {/* Training Duration */}
          <div className="flex items-center justify-between py-1 border-b border-studio-border/30">
            <span className="text-studio-muted text-[11px] flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-studio-muted" /> Duration:
            </span>
            <span className="font-mono text-studio-text text-[11px]" data-testid="preview-duration">
              {config.training_length_type === 'steps'
                ? `${config.max_steps ?? config.training_length} steps`
                : `${config.num_epochs ?? config.training_length} epochs`}
            </span>
          </div>

          {/* Effective Batch Size */}
          <div className="py-1 border-b border-studio-border/30">
            <div className="flex items-center justify-between">
              <span className="text-studio-muted text-[11px]">Effective Batch:</span>
              <span
                className="font-mono text-studio-text text-[11px] font-semibold text-studio-accent"
                data-testid="preview-effective-batch"
              >
                {totalEffectiveBatch !== null
                  ? `${totalEffectiveBatch}`
                  : `${effectivePerDevice} (per-device)`}
              </span>
            </div>
            <p className="text-[10px] text-studio-muted font-mono text-right mt-0.5">
              {detectedGpuCount !== null && detectedGpuCount > 0
                ? `${perDeviceBatchSize} batch × ${gradAccum} accum × ${detectedGpuCount} GPU(s)`
                : `${perDeviceBatchSize} batch × ${gradAccum} accum (total unknown — no GPU detected)`}
            </p>
          </div>

          {/* Context Length */}
          <div className="flex items-center justify-between py-1 border-b border-studio-border/30">
            <span className="text-studio-muted text-[11px]">Context Length:</span>
            <span className="font-mono text-studio-text text-[11px]" data-testid="preview-context-length">
              {config.context_length} tokens
            </span>
          </div>

          {/* Learning Rate */}
          <div className="flex items-center justify-between py-1 border-b border-studio-border/30">
            <span className="text-studio-muted text-[11px]">Learning Rate:</span>
            <span className="font-mono text-studio-text text-[11px]" data-testid="preview-learning-rate">
              {config.learning_rate}
            </span>
          </div>

          {/* Relevant Advanced Settings */}
          <div className="py-1 border-b border-studio-border/30 space-y-1">
            <div className="flex items-center justify-between text-[11px]">
              <span className="text-studio-muted flex items-center gap-1">
                <Sliders className="w-3 h-3 text-studio-muted" /> Advanced:
              </span>
              <span className="font-mono text-studio-text text-[10px]">
                {config.training_method === 'full_finetuning'
                  ? 'Full Weights'
                  : `r=${config.lora_rank ?? 16}, α=${config.lora_alpha ?? 32}`}
              </span>
            </div>
            {config.training_method === 'qlora' && (
              <div className="space-y-1 pt-0.5 border-t border-studio-border/20" data-testid="preview-qlora-details">
                <div className="flex items-center justify-between text-[10px] text-studio-muted font-mono">
                  <span>Quant Type:</span>
                  <span className="text-studio-teal font-semibold">
                    {config.load_in_8bit ? 'Int8' : `4-Bit ${(config.bnb_4bit_quant_type || config.quantization_type || 'nf4').toUpperCase()}`}
                  </span>
                </div>
                <div className="flex items-center justify-between text-[10px] text-studio-muted font-mono">
                  <span>Double Quant:</span>
                  <span className={config.bnb_4bit_use_double_quant !== false ? 'text-emerald-400' : 'text-zinc-500'}>
                    {config.bnb_4bit_use_double_quant !== false ? 'Enabled (Nested)' : 'Disabled'}
                  </span>
                </div>
                <div className="flex items-center justify-between text-[10px] text-studio-muted font-mono">
                  <span>Compute Dtype:</span>
                  <span className="text-studio-text">
                    {(config.bnb_4bit_compute_dtype || 'fp16').toUpperCase()}
                  </span>
                </div>
              </div>
            )}
            <div className="flex items-center justify-between text-[10px] text-studio-muted font-mono">
              <span>Optimizer / Prec:</span>
              <span className="text-studio-text">
                {config.optimizer || 'adamw_8bit'} • {(config.mixed_precision || 'fp16').toUpperCase()}
              </span>
            </div>
            <div className="flex items-center justify-between text-[10px] text-studio-muted font-mono">
              <span>Checkpointing:</span>
              <span className={config.gradient_checkpointing !== false ? 'text-emerald-400' : 'text-zinc-500'}>
                {config.gradient_checkpointing !== false ? 'Enabled' : 'Disabled'}
              </span>
            </div>
          </div>
        </div>

        {/* Hardware Status Breakdown - only when detected by backend */}
        {/* Hardware Status Breakdown */}
        {!backendOnline ? (
          <div
            className="p-3 rounded-lg bg-studio-input/80 border border-rose-900/40 space-y-2 text-xs"
            data-testid="hardware-status-box"
          >
            <div className="text-[10px] font-bold text-rose-400 uppercase tracking-wider font-mono flex items-center justify-between">
              <span className="flex items-center gap-1">
                <Cpu className="w-3 h-3 text-rose-400" /> Backend Status
              </span>
              <Badge variant="error">Offline</Badge>
            </div>
            <div className="text-[11px] space-y-1 font-mono">
              <div className="flex justify-between text-studio-text">
                <span className="text-studio-muted">Device:</span>
                <span className="text-rose-400 font-semibold" data-testid="hardware-device-name">
                  Backend Unavailable
                </span>
              </div>
              <p className="text-[10px] text-studio-muted leading-tight">
                Connect to FastAPI backend to inspect compute hardware and enable fine-tuning.
              </p>
            </div>
          </div>
        ) : systemStatus ? (
          <div
            className="p-3 rounded-lg bg-studio-input/80 border border-studio-border space-y-2 text-xs"
            data-testid="hardware-status-box"
          >
            <div className="text-[10px] font-bold text-studio-muted uppercase tracking-wider font-mono flex items-center justify-between">
              <span className="flex items-center gap-1">
                <Cpu className="w-3 h-3 text-studio-teal" /> Hardware Detected
              </span>
              <Badge variant={systemStatus.gpu_available || systemStatus.cuda_available ? 'success' : 'default'}>
                {systemStatus.gpu_available || systemStatus.cuda_available ? 'GPU Accelerated' : 'CPU Mode'}
              </Badge>
            </div>

            <div className="text-[11px] space-y-1 font-mono">
              <div className="flex justify-between text-studio-text">
                <span className="text-studio-muted">Device:</span>
                <span className="truncate max-w-[150px]" title={hardwareName} data-testid="hardware-device-name">
                  {hardwareName}
                </span>
              </div>
              <div className="flex justify-between text-studio-text">
                <span className="text-studio-muted">Device Type:</span>
                <span className="text-studio-text font-semibold uppercase" data-testid="hardware-device-type">
                  {(systemStatus.device_type || (systemStatus.gpu_available ? 'cuda' : 'cpu')).toUpperCase()}
                </span>
              </div>
              <div className="flex justify-between text-studio-text">
                <span className="text-studio-muted">Detected GPUs:</span>
                <span className="text-studio-text font-semibold">
                  {systemStatus.gpu_count} device(s)
                </span>
              </div>
              {systemStatus.primary_gpu_vram_gb !== null && systemStatus.primary_gpu_vram_gb !== undefined && (
                <div className="flex justify-between text-studio-text">
                  <span className="text-studio-muted">VRAM Capacity:</span>
                  <span className="text-studio-text font-semibold">
                    {systemStatus.primary_gpu_vram_gb} GB
                  </span>
                </div>
              )}
              <div className="flex justify-between text-studio-text">
                <span className="text-studio-muted">HF Token:</span>
                <span className={hasHfToken ? 'text-emerald-400' : 'text-zinc-500'}>
                  {hasHfToken ? 'Configured (Unverified)' : 'Not Set'}
                </span>
              </div>
            </div>

            {/* QLoRA Hardware Compatibility Status */}
            {config.training_method === 'qlora' && (
              <div className="pt-2 border-t border-studio-border/40 text-[10px] font-mono space-y-1" data-testid="preview-qlora-compat">
                <div className="flex justify-between items-center">
                  <span className="text-studio-muted">QLoRA Compatibility:</span>
                  <Badge variant={systemStatus.gpu_available || systemStatus.cuda_available ? 'success' : 'error'}>
                    {systemStatus.gpu_available || systemStatus.cuda_available ? 'Supported' : 'Unsupported'}
                  </Badge>
                </div>
                {!(systemStatus.gpu_available || systemStatus.cuda_available) && (
                  <p className="text-[10px] text-rose-400 leading-tight">
                    QLoRA 4-bit fine-tuning requires CUDA GPU acceleration. Current hardware is CPU-only.
                  </p>
                )}
              </div>
            )}
          </div>
        ) : null}
      </div>

      {/* Action Area & Disabled State Notice */}
      <div className="pt-4 border-t border-studio-border/60 mt-4 space-y-2.5">
        {/* Launch Control State Indicator */}
        <div className="flex items-center justify-between text-[11px] font-mono py-1 px-0.5">
          <span className="text-studio-muted">Launch Control:</span>
          <span
            className={`font-semibold ${
              launchControlState === 'Ready for confirmation' || launchControlState === 'Run created'
                ? 'text-emerald-400'
                : launchControlState === 'Blocked by validation' || launchControlState === 'Launch failed'
                ? 'text-rose-400'
                : launchControlState === 'Validation in progress' || launchControlState === 'Launching'
                ? 'text-amber-400'
                : 'text-zinc-500'
            }`}
            data-testid="launch-control-state"
          >
            {launchControlState}
          </span>
        </div>

        {/* Created Run Banner */}
        {createdRunId && (
          <div
            className="p-2.5 rounded-lg bg-emerald-950/20 border border-emerald-500/30 text-[11px] font-mono text-emerald-300 space-y-1"
            data-testid="created-run-banner"
          >
            <div className="flex items-center gap-1.5 font-bold text-emerald-400">
              <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
              <span>Run Created Successfully</span>
            </div>
            <div className="text-[10px] break-all">
              ID: <span className="text-studio-text font-bold" data-testid="created-run-id">{createdRunId}</span>
            </div>
          </div>
        )}

        {/* Launch Error Banner */}
        {launchError && !isLaunching && (
          <div
            className="p-2.5 rounded-lg bg-rose-950/30 border border-rose-500/30 text-[11px] font-mono text-rose-300 space-y-1"
            data-testid="launch-error-banner"
          >
            <div className="flex items-center gap-1.5 font-bold text-rose-400">
              <XCircle className="w-3.5 h-3.5 shrink-0" />
              <span>Launch Failed</span>
            </div>
            <p className="text-[10px] text-rose-300/90 leading-tight break-words">{launchError}</p>
          </div>
        )}

        {(() => {
          const isQloraBlocked = config.training_method === 'qlora' && Boolean(backendOnline && systemStatus && !systemStatus.gpu_available && !systemStatus.cuda_available);
          const isDisabled = !isReady || isLaunching || isQloraBlocked || launchControlState === 'Blocked by validation' || launchControlState === 'Configuration incomplete' || launchControlState === 'Validation in progress';

          let btnTitle = 'Configure model and dataset to enable fine-tuning';
          if (isQloraBlocked) {
            btnTitle = 'QLoRA 4-bit fine-tuning requires CUDA GPU acceleration. Current hardware is CPU-only.';
          } else if (launchControlState === 'Blocked by validation') {
            btnTitle = 'Blocked by validation: resolve errors before launching.';
          } else if (launchControlState === 'Configuration incomplete') {
            btnTitle = 'Select base model and dataset to enable fine-tuning.';
          } else if (launchControlState === 'Validation in progress') {
            btnTitle = 'Validation in progress...';
          } else if (isReady || launchControlState === 'Ready for confirmation') {
            btnTitle = config.training_method === 'qlora' ? 'Launch real QLoRA 4-bit fine-tuning' : 'Launch real LoRA fine-tuning';
          }

          return (
            <button
              type="button"
              disabled={isDisabled}
              onClick={onStartTraining}
              className={`w-full py-2.5 px-4 rounded-xl text-xs font-semibold flex items-center justify-center space-x-2 transition-all shadow-md ${
                !isDisabled
                  ? 'bg-studio-accent hover:bg-studio-accentHover text-studio-bg shadow-studio-accent/20 cursor-pointer'
                  : 'bg-studio-cardBorder/60 text-studio-muted border border-studio-border/60 cursor-not-allowed opacity-75'
              }`}
              data-testid="btn-start-training"
              title={btnTitle}
            >
              {isLaunching ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 mr-1 animate-spin" />
                  <span>Launching Fine-Tuning...</span>
                </>
              ) : isValidating ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 mr-1 animate-spin" />
                  <span>Validation in progress...</span>
                </>
              ) : launchControlState === 'Run created' ? (
                <>
                  <CheckCircle2 className="w-3.5 h-3.5 mr-1 text-emerald-400" />
                  <span>Run Created</span>
                </>
              ) : !isDisabled ? (
                <>
                  <Play className="w-3.5 h-3.5 mr-1 fill-current" />
                  <span>Start Fine-Tuning</span>
                </>
              ) : (
                <>
                  <Lock className="w-3.5 h-3.5 mr-1" />
                  <span>Start Fine-Tuning</span>
                </>
              )}
            </button>
          );
        })()}

        <div
          className="p-2 rounded bg-studio-bg/60 border border-studio-border text-[10px] text-studio-muted leading-relaxed"
          data-testid="phase0-preview-notice"
        >
          <p className="flex items-center gap-1 font-semibold text-studio-text">
            <AlertTriangle className="w-3 h-3 text-amber-400 shrink-0" />
            Phase 3 Configuration System Active / Phase 6 Pre-Launch Resource Validation Active
          </p>
          <p className="mt-0.5">
            Clicking Start Fine-Tuning opens the preflight checklist to validate hardware, dependencies, and configuration before launching.
          </p>
        </div>
      </div>
    </div>
  );
};
