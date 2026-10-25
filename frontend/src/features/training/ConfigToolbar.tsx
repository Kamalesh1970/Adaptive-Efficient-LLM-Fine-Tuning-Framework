import React, { useRef } from 'react';
import { Upload, Download, RotateCcw, FileText } from 'lucide-react';
import { TrainingConfig } from '../../types/training';

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

  // Export current config as a downloadable YAML-like file
  const handleSaveYaml = () => {
    const yamlLines = [
      '# Adaptive LLM Fine-Tuning Studio Draft Configuration',
      `project_name: "${config.project_name}"`,
      `model_identifier: ${config.model_identifier ? `"${config.model_identifier}"` : 'null'}`,
      `training_method: "${config.training_method}"`,
      `dataset_reference: ${config.dataset_reference ? `"${config.dataset_reference}"` : 'null'}`,
      `dataset_subset: ${config.dataset_subset ? `"${config.dataset_subset}"` : 'null'}`,
      `train_split: "${config.train_split}"`,
      `eval_split: "${config.eval_split}"`,
      `training_length_type: "${config.training_length_type}"`,
      `training_length: ${config.training_length}`,
      `batch_size: ${config.batch_size}`,
      `context_length: ${config.context_length}`,
      `learning_rate: ${config.learning_rate}`,
      `advanced_enabled: ${config.advanced_enabled}`,
    ];
    const blob = new Blob([yamlLines.join('\n')], { type: 'text/yaml;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `${config.project_name || 'training'}_config.yaml`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Load configuration from local YAML / JSON file
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target?.result as string;
        // Simple line parser for YAML key-values or JSON
        if (text.trim().startsWith('{')) {
          const parsed = JSON.parse(text);
          onLoadConfig({ ...config, ...parsed });
        } else {
          // Parse basic YAML key: value lines
          const updates: Partial<TrainingConfig> = {};
          text.split('\n').forEach((line) => {
            const clean = line.trim();
            if (!clean || clean.startsWith('#')) return;
            const colonIdx = clean.indexOf(':');
            if (colonIdx > 0) {
              const key = clean.slice(0, colonIdx).trim() as keyof TrainingConfig;
              let val = clean.slice(colonIdx + 1).trim();
              if (val.startsWith('"') && val.endsWith('"')) {
                val = val.slice(1, -1);
              }
              if (val === 'null') {
                (updates as any)[key] = null;
              } else if (val === 'true') {
                (updates as any)[key] = true;
              } else if (val === 'false') {
                (updates as any)[key] = false;
              } else if (!isNaN(Number(val)) && val !== '') {
                (updates as any)[key] = Number(val);
              } else {
                (updates as any)[key] = val;
              }
            }
          });
          onLoadConfig({ ...config, ...updates });
        }
      } catch (err) {
        alert('Failed to parse uploaded YAML file. Please check file formatting.');
      }
    };
    reader.readAsText(file);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
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
          onClick={() => fileInputRef.current?.click()}
          className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-studio-input border border-studio-border hover:border-studio-accent/40 text-studio-text hover:text-studio-accent text-xs font-medium transition-all"
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
          className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-studio-input border border-studio-border hover:border-studio-accent/40 text-studio-text hover:text-studio-accent text-xs font-medium transition-all"
          data-testid="btn-save-yaml"
          title="Download current configuration as YAML"
        >
          <Download className="w-3.5 h-3.5" />
          <span>Save YAML</span>
        </button>

        {/* Reset Defaults Button */}
        <button
          type="button"
          onClick={onReset}
          disabled={isResetting}
          className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-studio-input border border-studio-border hover:border-amber-500/40 text-studio-muted hover:text-amber-400 text-xs font-medium transition-all disabled:opacity-50"
          data-testid="btn-reset-defaults"
          title="Reset configuration to backend defaults"
        >
          <RotateCcw className={`w-3.5 h-3.5 ${isResetting ? 'animate-spin' : ''}`} />
          <span>Reset Defaults</span>
        </button>
      </div>
    </div>
  );
};
