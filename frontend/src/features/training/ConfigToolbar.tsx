import React, { useRef, useState } from 'react';
import { Upload, Download, RotateCcw, FileText, AlertCircle, CheckCircle2 } from 'lucide-react';
import { TrainingConfig } from '../../types/training';
import { exportConfigYaml, parseConfigYaml } from '../../services/api';

interface ConfigToolbarProps {
  config: TrainingConfig;
  onReset: () => void;
  onLoadConfig: (newConfig: TrainingConfig) => void;
  isResetting?: boolean;
}

export const ConfigToolbar: React.FC<ConfigToolbarProps> = ({
  config,
  onReset,
  onLoadConfig,
  isResetting = false,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [isExporting, setIsExporting] = useState(false);

  // Client-side fallback for safe YAML export without secrets
  const clientSafeYamlExport = (cfg: TrainingConfig): string => {
    const lines = [
      '# ====================================================================',
      '# Adaptive LLM Fine-Tuning Studio - Training Configuration',
      '# Safe YAML Export (Secrets excluded)',
      '# ====================================================================',
      `project_name: "${cfg.project_name || 'my-finetuned-model'}"`,
      `model_identifier: ${cfg.model_identifier ? `"${cfg.model_identifier}"` : 'null'}`,
      `training_method: "${cfg.training_method}"`,
      `dataset_reference: ${cfg.dataset_reference ? `"${cfg.dataset_reference}"` : 'null'}`,
      `dataset_subset: ${cfg.dataset_subset ? `"${cfg.dataset_subset}"` : 'null'}`,
      `train_split: "${cfg.train_split}"`,
      `eval_split: "${cfg.eval_split}"`,
      `training_length_type: "${cfg.training_length_type}"`,
      `training_duration_mode: "${cfg.training_duration_mode || cfg.training_length_type}"`,
      `training_length: ${cfg.training_length}`,
      `max_steps: ${cfg.max_steps ?? cfg.training_length}`,
      `num_epochs: ${cfg.num_epochs ?? 3.0}`,
      `learning_rate: ${cfg.learning_rate}`,
      `batch_size: ${cfg.batch_size}`,
      `per_device_train_batch_size: ${cfg.per_device_train_batch_size ?? cfg.batch_size}`,
      `gradient_accumulation_steps: ${cfg.gradient_accumulation_steps}`,
      `weight_decay: ${cfg.weight_decay}`,
      `warmup_ratio: ${cfg.warmup_ratio}`,
      `optimizer: "${cfg.optimizer}"`,
      `seed: ${cfg.seed}`,
      `context_length: ${cfg.context_length}`,
      `max_seq_length: ${cfg.max_seq_length ?? cfg.context_length}`,
      `truncation: ${cfg.truncation}`,
      `packing: ${cfg.packing}`,
      `lora_rank: ${cfg.lora_rank}`,
      `lora_alpha: ${cfg.lora_alpha}`,
      `lora_dropout: ${cfg.lora_dropout}`,
      `lora_target_modules: [${cfg.lora_target_modules?.map((m) => `"${m}"`).join(', ') || ''}]`,
      `gradient_checkpointing: ${cfg.gradient_checkpointing}`,
      `mixed_precision: "${cfg.mixed_precision}"`,
      `quantization_type: ${cfg.quantization_type ? `"${cfg.quantization_type}"` : 'null'}`,
      `advanced_enabled: ${cfg.advanced_enabled}`,
    ];
    return lines.join('\n') + '\n';
  };

  // Safe client-side parser fallback for offline mode
  const parseYamlOfflineFallback = (text: string): Partial<TrainingConfig> => {
    // Check for dangerous constructors or tags
    if (text.includes('!!') || text.includes('!ruby') || text.includes('!python')) {
      throw new Error('YAML tags and constructors are not permitted for security reasons.');
    }

    if (text.trim().startsWith('{')) {
      const parsed = JSON.parse(text);
      if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
        throw new Error('Invalid structure: expected key-value mapping.');
      }
      return parsed;
    }

    const lines = text.split('\n');
    const result: Record<string, unknown> = {};

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line || line.startsWith('#')) continue;

      // Detect list at root
      if (line.startsWith('-') && !line.includes(':')) {
        throw new Error('Invalid YAML structure: expected key-value mapping at root.');
      }

      const colonIdx = line.indexOf(':');
      if (colonIdx === -1) {
        throw new Error(`Malformed YAML on line ${i + 1}: expected 'key: value' mapping.`);
      }

      const key = line.slice(0, colonIdx).trim();
      let rawVal = line.slice(colonIdx + 1).trim();

      // Check for inline arrays e.g. [a, b]
      if (rawVal.startsWith('[') && rawVal.endsWith(']')) {
        const inner = rawVal.slice(1, -1).trim();
        result[key] = inner
          ? inner.split(',').map((item) => item.trim().replace(/^["']|["']$/g, ''))
          : [];
        continue;
      }

      if (rawVal.startsWith('"') && rawVal.endsWith('"')) {
        rawVal = rawVal.slice(1, -1);
      } else if (rawVal.startsWith("'") && rawVal.endsWith("'")) {
        rawVal = rawVal.slice(1, -1);
      }

      if (rawVal === 'null' || rawVal === '~' || rawVal === '') {
        result[key] = null;
      } else if (rawVal === 'true') {
        result[key] = true;
      } else if (rawVal === 'false') {
        result[key] = false;
      } else if (!isNaN(Number(rawVal)) && rawVal !== '') {
        result[key] = Number(rawVal);
      } else {
        result[key] = rawVal;
      }
    }

    if (Object.keys(result).length === 0) {
      throw new Error('YAML file is empty or contains no configuration keys.');
    }

    return result as Partial<TrainingConfig>;
  };

  // Export current config as a downloadable YAML file
  const handleSaveYaml = async () => {
    setIsExporting(true);
    setLoadError(null);
    try {
      let yamlString: string;
      try {
        const res = await exportConfigYaml(config);
        yamlString = res.yaml_content;
      } catch {
        // Fallback to client-safe export
        yamlString = clientSafeYamlExport(config);
      }

      const blob = new Blob([yamlString], { type: 'text/yaml;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `${config.project_name || 'training'}_config.yaml`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      setStatusMessage('Configuration exported to YAML file.');
      setTimeout(() => setStatusMessage(null), 3500);
    } catch (err: unknown) {
      setLoadError(err instanceof Error ? err.message : 'Failed to export YAML configuration.');
    } finally {
      setIsExporting(false);
    }
  };

  // Load configuration from local YAML / JSON file
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoadError(null);
    setStatusMessage(null);

    const reader = new FileReader();
    reader.onload = async (event) => {
      const text = event.target?.result as string;
      if (!text || !text.trim()) {
        setLoadError('Selected YAML file is completely empty.');
        return;
      }

      try {
        // Try server-side validated parsing first
        try {
          const res = await parseConfigYaml(text);
          if (res.is_valid && res.config) {
            onLoadConfig(res.config);
            setStatusMessage(`Successfully loaded configuration from ${file.name}`);
            setTimeout(() => setStatusMessage(null), 4000);
            return;
          } else if (!res.is_valid) {
            const errList = res.errors?.length
              ? res.errors.join(' | ')
              : 'Configuration failed schema validation.';
            setLoadError(`Invalid configuration file: ${errList}`);
            return;
          }
        } catch {
          // Backend unavailable: fall back to safe local parsing
          const updates = parseYamlOfflineFallback(text);
          // Check for invalid field values
          if (typeof updates.learning_rate === 'number' && updates.learning_rate <= 0) {
            throw new Error('Invalid learning_rate: must be strictly positive.');
          }
          if (typeof updates.context_length === 'number' && updates.context_length < 128) {
            throw new Error('Invalid context_length: must be at least 128 tokens.');
          }
          if (typeof updates.batch_size === 'number' && updates.batch_size <= 0) {
            throw new Error('Invalid batch_size: must be greater than 0.');
          }

          onLoadConfig({ ...config, ...updates });
          setStatusMessage(`Loaded configuration from ${file.name} (offline mode)`);
          setTimeout(() => setStatusMessage(null), 4000);
        }
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Failed to parse configuration file.';
        setLoadError(`YAML Load Error: ${msg}`);
      }
    };

    reader.onerror = () => {
      setLoadError('Failed to read configuration file from disk.');
    };

    reader.readAsText(file);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="space-y-2">
      <div
        className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-xl bg-studio-card border border-studio-cardBorder shadow-sm"
        data-testid="config-toolbar"
      >
        <div className="flex items-center space-x-2 text-xs text-studio-muted">
          <FileText className="w-4 h-4 text-studio-accent" />
          <span>Configuration Presets:</span>
        </div>

        <div className="flex items-center space-x-2">
          <input
            ref={fileInputRef}
            type="file"
            accept=".yaml,.yml,.json"
            className="hidden"
            onChange={handleFileChange}
            data-testid="yaml-file-input"
          />

          {/* Load YAML Button */}
          <button
            type="button"
            onClick={() => {
              setLoadError(null);
              fileInputRef.current?.click();
            }}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-studio-input border border-studio-border hover:border-studio-accent/40 text-studio-text hover:text-studio-accent text-xs font-medium transition-all cursor-pointer"
            data-testid="btn-load-yaml"
            title="Load configuration from YAML or JSON file"
          >
            <Upload className="w-3.5 h-3.5" />
            <span>Load YAML</span>
          </button>

          {/* Save YAML Button */}
          <button
            type="button"
            onClick={handleSaveYaml}
            disabled={isExporting}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-studio-input border border-studio-border hover:border-studio-accent/40 text-studio-text hover:text-studio-accent text-xs font-medium transition-all cursor-pointer disabled:opacity-50"
            data-testid="btn-save-yaml"
            title="Download current configuration as YAML (secrets excluded)"
          >
            <Download className="w-3.5 h-3.5" />
            <span>{isExporting ? 'Exporting...' : 'Save YAML'}</span>
          </button>

          {/* Reset Defaults Button */}
          <button
            type="button"
            onClick={() => {
              setLoadError(null);
              onReset();
            }}
            disabled={isResetting}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-studio-input border border-studio-border hover:border-amber-500/40 text-studio-muted hover:text-amber-400 text-xs font-medium transition-all disabled:opacity-50 cursor-pointer"
            data-testid="btn-reset-defaults"
            title="Reset configuration to baseline defaults"
          >
            <RotateCcw className={`w-3.5 h-3.5 ${isResetting ? 'animate-spin' : ''}`} />
            <span>Reset Defaults</span>
          </button>
        </div>
      </div>

      {/* Error Feedback */}
      {loadError && (
        <div
          className="p-2.5 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-center justify-between"
          data-testid="yaml-load-error"
        >
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{loadError}</span>
          </div>
          <button
            type="button"
            onClick={() => setLoadError(null)}
            className="text-rose-400 hover:text-rose-300 text-[11px] underline ml-2"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Success Feedback */}
      {statusMessage && (
        <div
          className="p-2.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs flex items-center space-x-2"
          data-testid="yaml-load-success"
        >
          <CheckCircle2 className="w-4 h-4 shrink-0" />
          <span>{statusMessage}</span>
        </div>
      )}
    </div>
  );
};
