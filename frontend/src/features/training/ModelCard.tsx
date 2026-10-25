import React, { useState } from 'react';
import { 
  Box, 
  KeyRound, 
  ShieldAlert, 
  CheckCircle2, 
  XCircle, 
  AlertTriangle, 
  Eye, 
  EyeOff, 
  RefreshCw,
  Info,
  Lock
} from 'lucide-react';
import { Card } from '../../components/common/Card';
import { Badge } from '../../components/common/Badge';
import { 
  TrainingConfig, 
  TrainingMethod, 
  ModelMetadata, 
  ModelValidationResponse 
} from '../../types/training';

interface ModelCardProps {
  config: TrainingConfig;
  onChange: (updates: Partial<TrainingConfig>) => void;
  catalog?: ModelMetadata[];
  isLoadingCatalog?: boolean;
  catalogError?: string | null;
  onRetryCatalog?: () => void;
  modelValidation?: ModelValidationResponse | null;
  isValidatingModel?: boolean;
  hfToken?: string;
  onHfTokenChange?: (token: string) => void;
}

export const ModelCard: React.FC<ModelCardProps> = ({
  config,
  onChange,
  catalog = [],
  isLoadingCatalog = false,
  catalogError = null,
  onRetryCatalog,
  modelValidation = null,
  isValidatingModel = false,
  hfToken = '',
  onHfTokenChange,
}) => {
  const [showToken, setShowToken] = useState(false);

  // Fallback options in case catalog is loading or offline
  const fallbackModelOptions = [
    { value: '', label: '-- No model selected --' },
    { value: 'meta-llama/Meta-Llama-3-8B', label: 'meta-llama/Meta-Llama-3-8B (8.0B parameters)' },
    { value: 'meta-llama/Llama-2-7b-hf', label: 'meta-llama/Llama-2-7b-hf (6.7B parameters)' },
    { value: 'mistralai/Mistral-7B-v0.3', label: 'mistralai/Mistral-7B-v0.3 (7.2B parameters)' },
    { value: 'Qwen/Qwen2.5-7B', label: 'Qwen/Qwen2.5-7B (7.6B parameters)' },
    { value: 'microsoft/Phi-3-mini-4k-instruct', label: 'microsoft/Phi-3-mini-4k-instruct (3.8B parameters)' },
    { value: 'Qwen/Qwen2.5-0.5B', label: 'Qwen/Qwen2.5-0.5B (0.5B parameters)' },
    { value: 'google/gemma-2-2b', label: 'google/gemma-2-2b (2.6B parameters)' },
  ];

  const methodOptions: { value: TrainingMethod; label: string; desc: string }[] = [
    { value: 'lora', label: 'LoRA (Low-Rank Adaptation)', desc: 'Trains low-rank adapter matrices (16-bit base)' },
    { value: 'qlora', label: 'QLoRA (4-bit Quantized LoRA)', desc: 'Quantizes base model to 4-bit NF4 with 16-bit LoRA adapters' },
    { value: 'full_finetuning', label: 'Full Fine-Tuning', desc: 'Updates all model parameters (high VRAM requirement)' },
  ];

  // Lookup currently selected model metadata
  const selectedModel = catalog.find((m) => m.id === config.model_identifier) || null;

  // Determine token status
  const hasEnvToken = config.hf_token_status === 'present_in_env';
  const hasInMemoryToken = Boolean(hfToken && hfToken.trim());

  // Header status badge
  const getHeaderBadge = () => {
    if (!config.model_identifier) {
      return <Badge variant="warning">No Model Selected</Badge>;
    }
    if (isValidatingModel) {
      return <Badge variant="default">Validating...</Badge>;
    }
    if (modelValidation) {
      if (modelValidation.is_valid) {
        return <Badge variant="success">Compatible</Badge>;
      }
      if (modelValidation.compatibility_status === 'requires_credentials') {
        return <Badge variant="warning">Credentials Required</Badge>;
      }
      return <Badge variant="error">Incompatible</Badge>;
    }
    return <Badge variant="success">Model Selected</Badge>;
  };

  return (
    <Card
      title="Base Model & Architecture"
      subtitle="Select the pre-trained foundation model and fine-tuning adaptation method"
      icon={<Box className="w-4 h-4" />}
      data-testid="model-card"
      headerAction={getHeaderBadge()}
    >
      <div className="space-y-4 text-xs">
        {/* Catalog Error Banner */}
        {catalogError && (
          <div
            className="p-2.5 rounded-lg bg-rose-950/40 border border-rose-800/60 text-rose-300 flex items-center justify-between"
            data-testid="catalog-error"
          >
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400" />
              <span>{catalogError}</span>
            </div>
            {onRetryCatalog && (
              <button
                type="button"
                onClick={onRetryCatalog}
                className="px-2 py-1 rounded bg-rose-900/60 hover:bg-rose-800 text-[11px] font-medium flex items-center gap-1 transition-colors"
              >
                <RefreshCw className="w-3 h-3" /> Retry
              </button>
            )}
          </div>
        )}

        {/* Model Dropdown */}
        <div>
          <label className="block text-studio-text font-medium mb-1.5 flex items-center justify-between">
            <span>Base Model Identifier</span>
            <span className="text-[10px] text-studio-muted">
              {isLoadingCatalog
                ? 'Fetching catalog...'
                : catalog.length > 0
                ? `${catalog.length} models registered in catalog`
                : 'Hugging Face repository or local path'}
            </span>
          </label>

          <select
            value={config.model_identifier || ''}
            onChange={(e) => onChange({ model_identifier: e.target.value || null })}
            disabled={isLoadingCatalog}
            className="w-full bg-studio-input border border-studio-cardBorder rounded-lg px-3 py-2 text-studio-text focus:outline-none focus:border-studio-accent transition-colors disabled:opacity-60 disabled:cursor-not-allowed font-mono text-xs"
            data-testid="model-select"
          >
            {isLoadingCatalog ? (
              <option value="" disabled>-- Loading model catalog... --</option>
            ) : catalog.length > 0 ? (
              <>
                <option value="" className="bg-studio-card text-studio-muted">
                  -- No model selected --
                </option>
                {catalog.map((m) => (
                  <option key={m.id} value={m.id} className="bg-studio-card text-studio-text">
                    {m.name} ({m.id}) {m.parameter_count ? `[${m.parameter_count}]` : ''} {m.is_gated ? '🔒 Gated' : ''}
                  </option>
                ))}
              </>
            ) : (
              fallbackModelOptions.map((opt) => (
                <option key={opt.value} value={opt.value} className="bg-studio-card text-studio-text">
                  {opt.label}
                </option>
              ))
            )}
          </select>

          {/* Loading Indicator */}
          {isLoadingCatalog && (
            <p className="text-[11px] text-studio-muted mt-1 flex items-center gap-1.5" data-testid="catalog-loading">
              <RefreshCw className="w-3 h-3 animate-spin text-studio-accent" />
              Loading verified model catalog from backend...
            </p>
          )}

          {/* No Model Selected Hint */}
          {!config.model_identifier && !isLoadingCatalog && (
            <p className="text-[11px] text-amber-400/90 mt-1 flex items-center gap-1" data-testid="no-model-hint">
              <ShieldAlert className="w-3 h-3 shrink-0" />
              No base model selected. Training cannot begin until a model is chosen.
            </p>
          )}
        </div>

        {/* Model Details / Metadata Summary */}
        {selectedModel && (
          <div
            className="p-3 rounded-lg bg-studio-input/60 border border-studio-cardBorder space-y-2"
            data-testid="model-metadata-summary"
          >
            <div className="flex items-center justify-between border-b border-studio-border/40 pb-1.5">
              <span className="font-semibold text-studio-text text-xs flex items-center gap-1.5">
                <Info className="w-3.5 h-3.5 text-studio-teal" />
                {selectedModel.name}
              </span>
              <div className="flex items-center gap-1.5">
                {selectedModel.is_gated ? (
                  <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium bg-amber-950/70 text-amber-300 border border-amber-800/60 font-mono">
                    <Lock className="w-2.5 h-2.5" /> Gated Access
                  </span>
                ) : (
                  <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-emerald-950/70 text-emerald-300 border border-emerald-800/60 font-mono">
                    Open Weights
                  </span>
                )}
                <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-studio-cardBorder/80 text-studio-muted">
                  {selectedModel.verification_status}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] font-mono">
              <div>
                <span className="text-studio-muted block text-[10px]">Architecture</span>
                <span className="text-studio-text">{selectedModel.family || 'Unknown'}</span>
              </div>
              <div>
                <span className="text-studio-muted block text-[10px]">Parameters</span>
                <span className="text-studio-text">{selectedModel.parameter_count || 'Unknown'}</span>
              </div>
              <div>
                <span className="text-studio-muted block text-[10px]">Verified Context</span>
                <span className="text-studio-text">
                  {selectedModel.context_length ? `${selectedModel.context_length.toLocaleString()} tokens` : 'Unverified'}
                </span>
              </div>
              <div>
                <span className="text-studio-muted block text-[10px]">Local Weights</span>
                <span className="text-studio-muted">Not Downloaded</span>
              </div>
            </div>

            {selectedModel.compatibility_notes && (
              <p className="text-[10px] text-studio-muted pt-1 border-t border-studio-border/30 italic">
                {selectedModel.compatibility_notes}
              </p>
            )}
          </div>
        )}

        {/* Training Method */}
        <div>
          <label className="block text-studio-text font-medium mb-1.5 flex items-center justify-between">
            <span>Fine-Tuning Method</span>
            <span className="text-[10px] text-studio-muted font-mono">Configuration choices (Phase 1)</span>
          </label>
          <select
            value={config.training_method}
            onChange={(e) => onChange({ training_method: e.target.value as TrainingMethod })}
            className="w-full bg-studio-input border border-studio-cardBorder rounded-lg px-3 py-2 text-studio-text focus:outline-none focus:border-studio-accent transition-colors font-mono text-xs"
            data-testid="method-select"
          >
            {methodOptions.map((opt) => {
              const isSupported = selectedModel ? selectedModel.supported_methods.includes(opt.value) : true;
              return (
                <option key={opt.value} value={opt.value} className="bg-studio-card text-studio-text">
                  {opt.label} {!isSupported ? '(Unsupported for selected model)' : ''}
                </option>
              );
            })}
          </select>
          <p className="text-[10px] text-studio-muted mt-1 italic">
            * Note: Training methods are configuration choices in Phase 1. Execution loops will be implemented in Phase 4.
          </p>
        </div>

        {/* Validation Feedback Banner */}
        {config.model_identifier && (
          <div data-testid="validation-feedback">
            {isValidatingModel ? (
              <div className="p-2.5 rounded-lg bg-studio-input/70 border border-studio-border text-studio-muted flex items-center gap-2 text-[11px]">
                <RefreshCw className="w-3.5 h-3.5 animate-spin text-studio-accent" />
                <span>Evaluating model compatibility and credentials...</span>
              </div>
            ) : modelValidation ? (
              modelValidation.is_valid ? (
                <div className="p-2.5 rounded-lg bg-emerald-950/40 border border-emerald-800/60 text-emerald-300 space-y-1">
                  <div className="flex items-center gap-1.5 font-medium text-[11px]">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                    <span>Configuration Compatible: {modelValidation.training_method} supported</span>
                  </div>
                  {modelValidation.warnings.length > 0 && (
                    <ul className="text-[10px] text-emerald-200/80 list-disc list-inside space-y-0.5">
                      {modelValidation.warnings.map((w, idx) => (
                        <li key={idx}>{w}</li>
                      ))}
                    </ul>
                  )}
                </div>
              ) : (
                <div
                  className={`p-2.5 rounded-lg border space-y-1.5 ${
                    modelValidation.compatibility_status === 'requires_credentials'
                      ? 'bg-amber-950/50 border-amber-800/70 text-amber-200'
                      : 'bg-rose-950/40 border-rose-800/60 text-rose-300'
                  }`}
                  data-testid="validation-error-banner"
                >
                  <div className="flex items-center gap-1.5 font-medium text-[11px]">
                    {modelValidation.compatibility_status === 'requires_credentials' ? (
                      <Lock className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                    ) : (
                      <XCircle className="w-3.5 h-3.5 text-rose-400 shrink-0" />
                    )}
                    <span>
                      {modelValidation.compatibility_status === 'requires_credentials'
                        ? 'Hugging Face Access Token Required'
                        : modelValidation.compatibility_status === 'unknown'
                        ? 'Model Compatibility Unverified'
                        : 'Method Compatibility Error'}
                    </span>
                  </div>
                  {modelValidation.errors.length > 0 && (
                    <ul className="text-[10px] list-disc list-inside space-y-0.5">
                      {modelValidation.errors.map((err, idx) => (
                        <li key={idx}>{err}</li>
                      ))}
                    </ul>
                  )}
                </div>
              )
            ) : null}
          </div>
        )}

        {/* Hugging Face Token Configuration & Security */}
        <div className="pt-3 border-t border-studio-border/50 space-y-2.5">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-studio-muted">
              <KeyRound className="w-3.5 h-3.5 text-studio-muted" />
              <span className="text-[11px] font-medium text-studio-text">Hugging Face Access Token:</span>
            </div>
            <div>
              <span 
                className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono font-medium border ${
                  hasInMemoryToken
                    ? 'bg-cyan-950/70 text-cyan-400 border-cyan-800/60'
                    : hasEnvToken 
                    ? 'bg-emerald-950/70 text-emerald-400 border-emerald-800/60'
                    : 'bg-zinc-800/80 text-zinc-400 border-zinc-700/60'
                }`}
                data-testid="hf-token-badge"
              >
                {hasInMemoryToken
                  ? 'Configured (In-Memory, Unverified)'
                  : hasEnvToken
                  ? 'Token Detected (Environment)'
                  : 'Not Detected in ENV'}
              </span>
            </div>
          </div>

          {/* Masked In-Memory Token Input */}
          <div className="flex items-center gap-2">
            <div className="relative flex-1">
              <input
                type={showToken ? 'text' : 'password'}
                value={hfToken}
                onChange={(e) => onHfTokenChange && onHfTokenChange(e.target.value)}
                placeholder="Optional in-memory HF token for gated models (e.g. hf_...)"
                autoComplete="off"
                spellCheck="false"
                className="w-full bg-studio-input border border-studio-cardBorder rounded-lg pl-3 pr-9 py-1.5 text-studio-text text-xs font-mono placeholder:text-zinc-600 focus:outline-none focus:border-studio-accent transition-colors"
                data-testid="hf-token-input"
              />
              <button
                type="button"
                onClick={() => setShowToken(!showToken)}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-studio-muted hover:text-studio-text p-0.5 rounded"
                title={showToken ? 'Mask token' : 'Show token'}
                data-testid="btn-toggle-token-mask"
              >
                {showToken ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
              </button>
            </div>
          </div>

          <p className="text-[10px] text-studio-muted leading-relaxed">
            <span className="font-semibold text-zinc-400">Security guarantee:</span> Access tokens are stored strictly in component memory for this session, never committed or written to browser storage, and never logged or exposed in API errors. Real credentials are labeled as unverified in Phase 1.
          </p>
        </div>
      </div>
    </Card>
  );
};
