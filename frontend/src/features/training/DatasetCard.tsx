import React, { useState } from 'react';
import { Database, UploadCloud, ChevronDown, ChevronRight, FileCode, AlertCircle } from 'lucide-react';
import { Card } from '../../components/common/Card';
import { Badge } from '../../components/common/Badge';
import { TrainingConfig } from '../../types/training';

interface DatasetCardProps {
  config: TrainingConfig;
  onChange: (updates: Partial<TrainingConfig>) => void;
}

export const DatasetCard: React.FC<DatasetCardProps> = ({ config, onChange }) => {
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const sampleDatasets = [
    { value: '', label: '-- No dataset selected --' },
    { value: 'yahma/alpaca-cleaned', label: 'yahma/alpaca-cleaned (51,760 instruction samples)' },
    { value: 'Open-Orca/OpenOrca', label: 'Open-Orca/OpenOrca (General instruction/reasoning)' },
    { value: 'HuggingFaceH4/no_robots', label: 'HuggingFaceH4/no_robots (High quality SFT dataset)' },
    { value: 'custom_local_upload', label: 'Custom Uploaded Dataset (Scaffolding)' },
  ];

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      onChange({ dataset_reference: `upload:${file.name}` });
    }
  };

  return (
    <Card
      title="Dataset & Data Preprocessing"
      subtitle="Select fine-tuning corpus or stage local datasets for tokenization"
      icon={<Database className="w-4 h-4" />}
      data-testid="dataset-card"
      headerAction={
        <Badge variant={config.dataset_reference ? 'success' : 'warning'}>
          {config.dataset_reference ? 'Dataset Selected' : 'No Dataset Selected'}
        </Badge>
      }
    >
      <div className="space-y-4 text-xs">
        {/* Dataset Selector */}
        <div>
          <label className="block text-studio-text font-medium mb-1.5 flex items-center justify-between">
            <span>Dataset Identifier</span>
            <span className="text-[10px] text-studio-muted">Hugging Face hub path or uploaded file</span>
          </label>
          <select
            value={config.dataset_reference || ''}
            onChange={(e) => onChange({ dataset_reference: e.target.value || null })}
            className="w-full bg-studio-input border border-studio-cardBorder rounded-lg px-3 py-2 text-studio-text focus:outline-none focus:border-studio-accent transition-colors"
            data-testid="dataset-select"
          >
            {sampleDatasets.map((ds) => (
              <option key={ds.value} value={ds.value} className="bg-studio-card text-studio-text">
                {ds.label}
              </option>
            ))}
          </select>
          {!config.dataset_reference && (
            <p className="text-[11px] text-amber-400/90 mt-1 flex items-center gap-1" data-testid="no-dataset-hint">
              <AlertCircle className="w-3 h-3 shrink-0" />
              No dataset selected. Select a dataset to configure training.
            </p>
          )}
        </div>

        {/* Drag and Drop Upload Visual Scaffolding */}
        <div>
          <label className="block text-studio-text font-medium mb-1.5 flex items-center justify-between">
            <span>Upload Dataset File</span>
            <span className="text-[10px] font-mono text-studio-muted">Phase 0 Scaffolding</span>
          </label>
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-lg p-4 text-center transition-all ${
              dragActive
                ? 'border-studio-accent bg-studio-accent/5'
                : 'border-studio-border hover:border-studio-accent/40 bg-studio-bg/40'
            }`}
            data-testid="upload-dropzone"
          >
            <UploadCloud className="w-6 h-6 mx-auto text-studio-muted mb-1.5" />
            <p className="text-[11px] text-studio-text font-medium">
              Drag & drop dataset files here, or{' '}
              <label className="text-studio-accent hover:underline cursor-pointer">
                browse files
                <input
                  type="file"
                  accept=".json,.jsonl,.csv,.parquet"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      onChange({ dataset_reference: `upload:${e.target.files[0].name}` });
                    }
                  }}
                />
              </label>
            </p>
            <p className="text-[10px] text-studio-muted mt-1 flex items-center justify-center gap-1">
              <FileCode className="w-3 h-3 text-studio-muted" />
              Supported formats: <span className="font-mono text-studio-textSecondary">.json, .jsonl, .csv, .parquet</span>
            </p>
            <p className="text-[9px] text-studio-muted/70 mt-1 italic">
              (Visual scaffolding for Phase 0. Actual server-side processing is implemented in Phase 2)
            </p>
          </div>
        </div>

        {/* Splits & Subset Selectors */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div>
            <label className="block text-studio-muted text-[11px] font-medium mb-1">
              Subset Name
            </label>
            <input
              type="text"
              placeholder="default"
              value={config.dataset_subset || ''}
              onChange={(e) => onChange({ dataset_subset: e.target.value || null })}
              className="w-full bg-studio-input border border-studio-cardBorder rounded-lg px-2.5 py-1.5 text-studio-text text-xs focus:outline-none focus:border-studio-accent font-mono"
            />
          </div>
          <div>
            <label className="block text-studio-muted text-[11px] font-medium mb-1">
              Training Split
            </label>
            <input
              type="text"
              value={config.train_split}
              onChange={(e) => onChange({ train_split: e.target.value })}
              className="w-full bg-studio-input border border-studio-cardBorder rounded-lg px-2.5 py-1.5 text-studio-text text-xs focus:outline-none focus:border-studio-accent font-mono"
            />
          </div>
          <div>
            <label className="block text-studio-muted text-[11px] font-medium mb-1">
              Evaluation Split
            </label>
            <select
              value={config.eval_split}
              onChange={(e) => onChange({ eval_split: e.target.value })}
              className="w-full bg-studio-input border border-studio-cardBorder rounded-lg px-2.5 py-1.5 text-studio-text text-xs focus:outline-none focus:border-studio-accent"
            >
              <option value="none">none (Skip eval)</option>
              <option value="test">test</option>
              <option value="validation">validation</option>
            </select>
          </div>
        </div>

        {/* Collapsible Advanced Dataset Section */}
        <div className="pt-2 border-t border-studio-border/50">
          <button
            type="button"
            onClick={() => setAdvancedOpen(!advancedOpen)}
            className="flex items-center space-x-1.5 text-studio-muted hover:text-studio-text transition-colors text-[11px] font-medium"
            data-testid="dataset-advanced-toggle"
          >
            {advancedOpen ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
            <span>Advanced Dataset Formatting Options</span>
          </button>

          {advancedOpen && (
            <div className="mt-3 p-3 rounded-lg bg-studio-bg/60 border border-studio-border/60 space-y-3" data-testid="dataset-advanced-panel">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                  <label className="block text-studio-muted text-[10px] mb-1">Prompt Column Name</label>
                  <input
                    type="text"
                    defaultValue="instruction"
                    placeholder="instruction"
                    disabled
                    className="w-full bg-studio-input/50 border border-studio-cardBorder/50 rounded px-2 py-1 text-studio-muted text-xs cursor-not-allowed font-mono"
                  />
                </div>
                <div>
                  <label className="block text-studio-muted text-[10px] mb-1">Response Column Name</label>
                  <input
                    type="text"
                    defaultValue="output"
                    placeholder="output"
                    disabled
                    className="w-full bg-studio-input/50 border border-studio-cardBorder/50 rounded px-2 py-1 text-studio-muted text-xs cursor-not-allowed font-mono"
                  />
                </div>
              </div>
              <p className="text-[10px] text-studio-muted italic">
                * Automatic column detection and conversational chat templating will be activated in Phase 2.
              </p>
            </div>
          )}
        </div>
      </div>
    </Card>
  );
};
