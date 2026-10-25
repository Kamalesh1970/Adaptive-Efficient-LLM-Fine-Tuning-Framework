import React from 'react';
import { Box, KeyRound, ShieldAlert } from 'lucide-react';
import { Card } from '../../components/common/Card';
import { Badge } from '../../components/common/Badge';
import { TrainingConfig, TrainingMethod } from '../../types/training';

interface ModelCardProps {
  config: TrainingConfig;
  onChange: (updates: Partial<TrainingConfig>) => void;
}

export const ModelCard: React.FC<ModelCardProps> = ({ config, onChange }) => {
  const modelOptions = [
    { value: '', label: '-- No model selected --' },
    { value: 'meta-llama/Meta-Llama-3-8B', label: 'meta-llama/Meta-Llama-3-8B (8.0B parameters)' },
    { value: 'meta-llama/Llama-2-7b-hf', label: 'meta-llama/Llama-2-7b-hf (6.7B parameters)' },
    { value: 'mistralai/Mistral-7B-v0.3', label: 'mistralai/Mistral-7B-v0.3 (7.2B parameters)' },
    { value: 'Qwen/Qwen2.5-7B', label: 'Qwen/Qwen2.5-7B (7.6B parameters)' },
    { value: 'microsoft/Phi-3-mini-4k-instruct', label: 'microsoft/Phi-3-mini-4k-instruct (3.8B parameters)' },
  ];

  const methodOptions: { value: TrainingMethod; label: string; desc: string }[] = [
    { value: 'lora', label: 'LoRA (Low-Rank Adaptation)', desc: 'Trains low-rank adapter matrices (16-bit base)' },
    { value: 'qlora', label: 'QLoRA (4-bit Quantized LoRA)', desc: 'Quantizes base model to 4-bit NF4 with 16-bit LoRA adapters' },
    { value: 'full_finetuning', label: 'Full Fine-Tuning', desc: 'Updates all model parameters (high VRAM requirement)' },
  ];

  const hasHfToken = config.hf_token_status === 'present_in_env';

  return (
    <Card
      title="Base Model & Architecture"
      subtitle="Select the pre-trained foundation model and fine-tuning adaptation method"
      icon={<Box className="w-4 h-4" />}
      data-testid="model-card"
      headerAction={
        <Badge variant={config.model_identifier ? 'success' : 'warning'}>
          {config.model_identifier ? 'Model Selected' : 'No Model Selected'}
        </Badge>
      }
    >
      <div className="space-y-4 text-xs">
        {/* Model Dropdown */}
        <div>
          <label className="block text-studio-text font-medium mb-1.5 flex items-center justify-between">
            <span>Base Model Identifier</span>
            <span className="text-[10px] text-studio-muted">Hugging Face repository or local path</span>
          </label>
          <select
            value={config.model_identifier || ''}
            onChange={(e) => onChange({ model_identifier: e.target.value || null })}
            className="w-full bg-studio-input border border-studio-cardBorder rounded-lg px-3 py-2 text-studio-text focus:outline-none focus:border-studio-accent transition-colors"
            data-testid="model-select"
          >
            {modelOptions.map((opt) => (
              <option key={opt.value} value={opt.value} className="bg-studio-card text-studio-text">
                {opt.label}
              </option>
            ))}
          </select>
          {!config.model_identifier && (
            <p className="text-[11px] text-amber-400/90 mt-1 flex items-center gap-1" data-testid="no-model-hint">
              <ShieldAlert className="w-3 h-3 shrink-0" />
              No base model selected. Training cannot begin until a model is chosen.
            </p>
          )}
        </div>

        {/* Training Method */}
        <div>
          <label className="block text-studio-text font-medium mb-1.5 flex items-center justify-between">
            <span>Fine-Tuning Method</span>
            <span className="text-[10px] text-studio-muted font-mono">UI options only in Phase 0</span>
          </label>
          <select
            value={config.training_method}
            onChange={(e) => onChange({ training_method: e.target.value as TrainingMethod })}
            className="w-full bg-studio-input border border-studio-cardBorder rounded-lg px-3 py-2 text-studio-text focus:outline-none focus:border-studio-accent transition-colors"
            data-testid="method-select"
          >
            {methodOptions.map((opt) => (
              <option key={opt.value} value={opt.value} className="bg-studio-card text-studio-text">
                {opt.label}
              </option>
            ))}
          </select>
          <p className="text-[10px] text-studio-muted mt-1 italic">
            * Note: Training methods are UI scaffolding in Phase 0. Actual weight adaptation is implemented in Phase 4.
          </p>
        </div>

        {/* Hugging Face Token Status */}
        <div className="pt-2 border-t border-studio-border/50 flex items-center justify-between">
          <div className="flex items-center space-x-2 text-studio-muted">
            <KeyRound className="w-3.5 h-3.5 text-studio-muted" />
            <span className="text-[11px]">Hugging Face Access Token:</span>
          </div>
          <div>
            <span 
              className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono font-medium border ${
                hasHfToken 
                  ? 'bg-emerald-950/70 text-emerald-400 border-emerald-800/60'
                  : 'bg-zinc-800/80 text-zinc-400 border-zinc-700/60'
              }`}
              data-testid="hf-token-badge"
            >
              {hasHfToken ? 'Token Detected (Environment)' : 'Not Detected in ENV'}
            </span>
          </div>
        </div>
      </div>
    </Card>
  );
};
