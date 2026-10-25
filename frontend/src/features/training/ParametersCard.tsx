import React, { useState } from 'react';
import { Sliders, Settings2, Info } from 'lucide-react';
import { Card } from '../../components/common/Card';
import { Badge } from '../../components/common/Badge';
import { TrainingConfig } from '../../types/training';

interface ParametersCardProps {
  config: TrainingConfig;
  onChange: (updates: Partial<TrainingConfig>) => void;
}

export const ParametersCard: React.FC<ParametersCardProps> = ({ config, onChange }) => {
  const [paramTab, setParamTab] = useState<'simple' | 'advanced'>('simple');

  return (
    <Card
      title="Hyperparameters & Training Settings"
      subtitle="Configure training duration, optimizer parameters, and context sequence lengths"
      icon={<Sliders className="w-4 h-4" />}
      data-testid="parameters-card"
      headerAction={
        <div className="flex items-center space-x-1 p-0.5 bg-studio-input border border-studio-cardBorder rounded-lg">
          <button
            type="button"
            onClick={() => setParamTab('simple')}
            className={`px-2.5 py-1 rounded text-[11px] font-medium transition-all ${
              paramTab === 'simple'
                ? 'bg-studio-accent text-studio-bg font-semibold shadow-xs'
                : 'text-studio-muted hover:text-studio-text'
            }`}
            data-testid="param-tab-simple"
          >
            Simple
          </button>
          <button
            type="button"
            onClick={() => setParamTab('advanced')}
            className={`px-2.5 py-1 rounded text-[11px] font-medium transition-all ${
              paramTab === 'advanced'
                ? 'bg-studio-accent text-studio-bg font-semibold shadow-xs'
                : 'text-studio-muted hover:text-studio-text'
            }`}
            data-testid="param-tab-advanced"
          >
            Advanced
          </button>
        </div>
      }
    >
      {paramTab === 'simple' ? (
        <div className="space-y-4 text-xs" data-testid="simple-parameters-view">
          {/* Project Name */}
          <div>
            <label className="block text-studio-text font-medium mb-1 flex items-center justify-between">
              <span>Project / Output Name</span>
              <span className="text-[10px] text-studio-muted font-mono">Checkpoints directory identifier</span>
            </label>
            <input
              type="text"
              value={config.project_name}
              onChange={(e) => onChange({ project_name: e.target.value })}
              className="w-full bg-studio-input border border-studio-cardBorder rounded-lg px-3 py-2 text-studio-text focus:outline-none focus:border-studio-accent font-mono transition-colors"
              placeholder="e.g. my-finetuned-llama3"
              data-testid="input-project-name"
            />
          </div>

          {/* Training Duration & Units Switcher */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <label className="block text-studio-text font-medium mb-1 flex items-center justify-between">
                <span>Training Duration</span>
                <div className="flex items-center space-x-1 bg-studio-input px-1.5 py-0.5 rounded border border-studio-border">
                  <button
                    type="button"
                    onClick={() => onChange({ training_length_type: 'steps' })}
                    className={`px-1.5 py-0.5 rounded text-[10px] font-mono transition-colors ${
                      config.training_length_type === 'steps'
                        ? 'bg-studio-accent/20 text-studio-accent font-semibold'
                        : 'text-studio-muted hover:text-studio-text'
                    }`}
                    data-testid="duration-type-steps"
                  >
                    Steps
                  </button>
                  <span className="text-studio-border">|</span>
                  <button
                    type="button"
                    onClick={() => onChange({ training_length_type: 'epochs' })}
                    className={`px-1.5 py-0.5 rounded text-[10px] font-mono transition-colors ${
                      config.training_length_type === 'epochs'
                        ? 'bg-studio-accent/20 text-studio-accent font-semibold'
                        : 'text-studio-muted hover:text-studio-text'
                    }`}
                    data-testid="duration-type-epochs"
                  >
                    Epochs
                  </button>
                </div>
              </label>
              <input
                type="number"
                min={1}
                max={1000000}
                value={config.training_length}
                onChange={(e) => onChange({ training_length: parseInt(e.target.value, 10) || 1 })}
                className="w-full bg-studio-input border border-studio-cardBorder rounded-lg px-3 py-2 text-studio-text focus:outline-none focus:border-studio-accent font-mono"
                data-testid="input-training-length"
              />
            </div>

            {/* Batch Size */}
            <div>
              <label className="block text-studio-text font-medium mb-1 flex items-center justify-between">
                <span>Per-Device Batch Size</span>
                <span className="text-[10px] text-studio-muted">micro-batch</span>
              </label>
              <input
                type="number"
                min={1}
                max={64}
                value={config.batch_size}
                onChange={(e) => onChange({ batch_size: parseInt(e.target.value, 10) || 1 })}
                className="w-full bg-studio-input border border-studio-cardBorder rounded-lg px-3 py-2 text-studio-text focus:outline-none focus:border-studio-accent font-mono"
                data-testid="input-batch-size"
              />
            </div>
          </div>

          {/* Context Length & Learning Rate */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <label className="block text-studio-text font-medium mb-1 flex items-center justify-between">
                <span>Context Length (Tokens)</span>
                <span className="text-[10px] text-studio-muted font-mono">128 - 131k</span>
              </label>
              <input
                type="number"
                step={256}
                min={128}
                max={131072}
                value={config.context_length}
                onChange={(e) => onChange({ context_length: parseInt(e.target.value, 10) || 128 })}
                className="w-full bg-studio-input border border-studio-cardBorder rounded-lg px-3 py-2 text-studio-text focus:outline-none focus:border-studio-accent font-mono"
                data-testid="input-context-length"
              />
            </div>

            <div>
              <label className="block text-studio-text font-medium mb-1 flex items-center justify-between">
                <span>Learning Rate</span>
                <span className="text-[10px] text-studio-muted font-mono">AdamW lr</span>
              </label>
              <input
                type="number"
                step="0.00005"
                min="0.000001"
                max="1.0"
                value={config.learning_rate}
                onChange={(e) => onChange({ learning_rate: parseFloat(e.target.value) || 0.0002 })}
                className="w-full bg-studio-input border border-studio-cardBorder rounded-lg px-3 py-2 text-studio-text focus:outline-none focus:border-studio-accent font-mono"
                data-testid="input-learning-rate"
              />
            </div>
          </div>

          <div className="p-2.5 rounded-lg bg-studio-bg/50 border border-studio-border/50 text-[11px] text-studio-muted flex items-center gap-2">
            <Info className="w-4 h-4 text-studio-teal shrink-0" />
            <span>
              Values are pre-filled with sensible draft defaults based on Unsloth standard recipes.
            </span>
          </div>
        </div>
      ) : (
        /* Advanced Mode Placeholder */
        <div className="p-6 text-center space-y-3 bg-studio-bg/40 rounded-lg border border-dashed border-studio-border" data-testid="advanced-parameters-placeholder">
          <Settings2 className="w-8 h-8 text-studio-muted mx-auto stroke-[1.5]" />
          <div>
            <h3 className="text-xs font-semibold text-studio-text">Advanced Hyperparameter Controls</h3>
            <p className="text-[11px] text-studio-muted mt-1 max-w-md mx-auto">
              Advanced training controls (gradient accumulation steps, warmup ratios, weight decay, FP16/BF16 mixed precision, and optimizer schedulers) will be configurable here in later phases.
            </p>
          </div>
          <Badge variant="info">Phase 0 Architectural Placeholder</Badge>
        </div>
      )}
    </Card>
  );
};
