import React, { useState, useRef } from 'react';
import { 
  Database, 
  UploadCloud, 
  ChevronDown, 
  ChevronRight, 
  FileCode, 
  AlertCircle,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  RefreshCw,
  FileText,
  Sliders,
  Layers
} from 'lucide-react';
import { Card } from '../../components/common/Card';
import { Badge } from '../../components/common/Badge';
import { 
  TrainingConfig, 
  DatasetSummary, 
  DatasetValidationReport, 
  DatasetSplitResult 
} from '../../types/training';

interface DatasetCardProps {
  config: TrainingConfig;
  onChange: (updates: Partial<TrainingConfig>) => void;
  datasets?: DatasetSummary[];
  isLoadingDatasets?: boolean;
  selectedReport?: DatasetValidationReport | null;
  onUploadFile?: (file: File) => Promise<void>;
  isUploading?: boolean;
  uploadError?: string | null;
  uploadSuccess?: boolean;
  onSelectDataset?: (datasetId: string) => void;
  splitResult?: DatasetSplitResult | null;
  trainRatio?: number;
  evalRatio?: number;
  onSplitChange?: (trainRatio: number, evalRatio: number) => void;
  backendOnline?: boolean;
}

export const DatasetCard: React.FC<DatasetCardProps> = ({
  config,
  onChange,
  datasets = [],
  isLoadingDatasets = false,
  selectedReport = null,
  onUploadFile,
  isUploading = false,
  uploadError = null,
  uploadSuccess = false,
  onSelectDataset,
  splitResult = null,
  trainRatio = 0.9,
  evalRatio = 0.1,
  onSplitChange,
  backendOnline = true,
}) => {
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [clientFileError, setClientFileError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const fallbackDatasets = [
    { value: '', label: '-- No dataset selected --' },
    { value: 'yahma/alpaca-cleaned', label: 'yahma/alpaca-cleaned (51,760 instruction samples)' },
    { value: 'Open-Orca/OpenOrca', label: 'Open-Orca/OpenOrca (General instruction/reasoning)' },
    { value: 'HuggingFaceH4/no_robots', label: 'HuggingFaceH4/no_robots (High quality SFT dataset)' },
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

  const processFile = async (file: File) => {
    setClientFileError(null);
    const validExtensions = ['.json', '.jsonl', '.csv'];
    const fileName = file.name.toLowerCase();
    const hasValidExt = validExtensions.some((ext) => fileName.endsWith(ext));

    if (!hasValidExt) {
      setClientFileError(`Unsupported format '${file.name}'. Supported formats: .json, .jsonl, .csv`);
      return;
    }

    const MAX_SIZE = 50 * 1024 * 1024; // 50MB
    if (file.size > MAX_SIZE) {
      setClientFileError(`Upload too large: '${file.name}' exceeds the 50MB size limit.`);
      return;
    }

    if (onUploadFile) {
      await onUploadFile(file);
    } else {
      onChange({ dataset_reference: `upload:${file.name}` });
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const getHeaderBadge = () => {
    if (isUploading) {
      return <Badge variant="default">Uploading...</Badge>;
    }
    if (!config.dataset_reference) {
      return <Badge variant="warning">No Dataset Selected</Badge>;
    }
    if (selectedReport) {
      if (selectedReport.is_valid) {
        return <Badge variant="success">Validated</Badge>;
      }
      return <Badge variant="error">Invalid Dataset</Badge>;
    }
    return <Badge variant="success">Dataset Selected</Badge>;
  };

  const activeError = clientFileError || uploadError;

  return (
    <Card
      title="Dataset & Data Preprocessing"
      subtitle="Select fine-tuning corpus or upload local datasets for schema validation"
      icon={<Database className="w-4 h-4" />}
      data-testid="dataset-card"
      headerAction={getHeaderBadge()}
    >
      <div className="space-y-4 text-xs">
        {/* Backend offline note if needed */}
        {!backendOnline && (
          <div className="p-2 rounded bg-amber-950/40 border border-amber-800/60 text-amber-300 text-[11px] flex items-center gap-2">
            <AlertTriangle className="w-3.5 h-3.5 shrink-0 text-amber-400" />
            <span>Backend is offline. Upload and server-side validation are disabled.</span>
          </div>
        )}

        {/* Dataset Selector */}
        <div>
          <label className="block text-studio-text font-medium mb-1.5 flex items-center justify-between">
            <span>Dataset Identifier</span>
            <span className="text-[10px] text-studio-muted">
              {isLoadingDatasets
                ? 'Loading datasets...'
                : datasets.length > 0
                ? `${datasets.length} uploaded dataset(s) available`
                : 'Hugging Face hub path or uploaded file'}
            </span>
          </label>

          <select
            value={config.dataset_reference || ''}
            onChange={(e) => {
              const val = e.target.value || null;
              onChange({ dataset_reference: val });
              if (val && onSelectDataset) {
                onSelectDataset(val);
              }
            }}
            className="w-full bg-studio-input border border-studio-cardBorder rounded-lg px-3 py-2 text-studio-text focus:outline-none focus:border-studio-accent transition-colors font-mono text-xs"
            data-testid="dataset-select"
          >
            {datasets.length > 0 ? (
              <>
                <option value="" className="bg-studio-card text-studio-muted">
                  -- No dataset selected --
                </option>
                {datasets.map((ds) => (
                  <option key={ds.id} value={ds.id} className="bg-studio-card text-studio-text">
                    {ds.filename} ({ds.id}) [{ds.valid_sample_count} samples, {ds.format}]
                  </option>
                ))}
              </>
            ) : (
              fallbackDatasets.map((ds) => (
                <option key={ds.value} value={ds.value} className="bg-studio-card text-studio-text">
                  {ds.label}
                </option>
              ))
            )}
          </select>

          {!config.dataset_reference && (
            <p className="text-[11px] text-amber-400/90 mt-1 flex items-center gap-1" data-testid="no-dataset-hint">
              <AlertCircle className="w-3 h-3 shrink-0" />
              No dataset selected. Select or upload a dataset to configure training.
            </p>
          )}
        </div>

        {/* Drag and Drop Upload Area */}
        <div>
          <label className="block text-studio-text font-medium mb-1.5 flex items-center justify-between">
            <span>Upload Dataset File</span>
            <span className="text-[10px] font-mono text-studio-muted">Max 50 MB per file</span>
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
            {isUploading ? (
              <div className="py-2 space-y-2">
                <RefreshCw className="w-6 h-6 mx-auto animate-spin text-studio-accent" />
                <p className="text-[11px] text-studio-text font-medium" data-testid="upload-status">
                  Uploading and validating dataset...
                </p>
              </div>
            ) : (
              <>
                <UploadCloud className="w-6 h-6 mx-auto text-studio-muted mb-1.5" />
                <p className="text-[11px] text-studio-text font-medium">
                  Drag & drop dataset files here, or{' '}
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="text-studio-accent hover:underline cursor-pointer bg-transparent border-0 p-0 font-medium inline"
                  >
                    browse files
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    data-testid="file-input"
                    accept=".json,.jsonl,.csv"
                    className="hidden"
                    onChange={handleFileInputChange}
                  />
                </p>
                <p className="text-[10px] text-studio-muted mt-1 flex items-center justify-center gap-1">
                  <FileCode className="w-3 h-3 text-studio-muted" />
                  Supported formats:{' '}
                  <span className="font-mono text-studio-textSecondary">.json, .jsonl, .csv</span>
                </p>
              </>
            )}
          </div>

          {/* Upload Status / Error feedback */}
          {activeError && (
            <div
              className="mt-2 p-2.5 rounded-lg bg-rose-950/40 border border-rose-800/60 text-rose-300 text-[11px] flex items-center gap-2"
              data-testid="dataset-error-banner"
            >
              <XCircle className="w-4 h-4 shrink-0 text-rose-400" />
              <span>{activeError}</span>
            </div>
          )}

          {uploadSuccess && !activeError && (
            <div
              className="mt-2 p-2.5 rounded-lg bg-emerald-950/40 border border-emerald-800/60 text-emerald-300 text-[11px] flex items-center gap-2"
              data-testid="upload-status"
            >
              <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-400" />
              <span>Dataset uploaded and validated successfully.</span>
            </div>
          )}
        </div>

        {/* Validation Summary Card */}
        {selectedReport && (
          <div
            className="p-3 rounded-lg bg-studio-input/60 border border-studio-cardBorder space-y-2.5"
            data-testid="dataset-validation-summary"
          >
            <div className="flex items-center justify-between border-b border-studio-border/40 pb-1.5">
              <span className="font-semibold text-studio-text text-xs flex items-center gap-1.5">
                <FileText className="w-3.5 h-3.5 text-studio-teal" />
                {selectedReport.original_filename}
              </span>
              <div className="flex items-center gap-1.5">
                {selectedReport.detected_schema && (
                  <span
                    className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-studio-cardBorder/90 text-studio-accent font-medium border border-studio-border"
                    data-testid="dataset-schema-badge"
                  >
                    {selectedReport.detected_schema}
                  </span>
                )}
                <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-studio-cardBorder/80 text-studio-muted">
                  {selectedReport.file_format.toUpperCase()}
                </span>
                <span
                  className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-medium ${
                    selectedReport.is_valid
                      ? 'bg-emerald-950/70 text-emerald-300 border border-emerald-800/60'
                      : 'bg-rose-950/70 text-rose-300 border border-rose-800/60'
                  }`}
                >
                  {selectedReport.is_valid ? 'Valid' : 'Validation Failed'}
                </span>
              </div>
            </div>

            {/* Metrics Breakdown Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] font-mono" data-testid="dataset-sample-count">
              <div>
                <span className="text-studio-muted block text-[10px]">Total Records</span>
                <span className="text-studio-text">{selectedReport.total_samples.toLocaleString()}</span>
              </div>
              <div>
                <span className="text-studio-muted block text-[10px]">Valid Samples</span>
                <span className="text-emerald-400 font-medium">{selectedReport.valid_sample_count.toLocaleString()}</span>
              </div>
              <div>
                <span className="text-studio-muted block text-[10px]">Invalid Samples</span>
                <span className={selectedReport.invalid_sample_count > 0 ? 'text-rose-400 font-medium' : 'text-studio-muted'}>
                  {selectedReport.invalid_sample_count.toLocaleString()}
                </span>
              </div>
              <div>
                <span className="text-studio-muted block text-[10px]">Duplicates</span>
                <span className="text-studio-text">
                  {selectedReport.duplicate_count !== null ? selectedReport.duplicate_count : 'N/A'}
                </span>
              </div>
            </div>

            {/* Warnings List */}
            {selectedReport.warnings.length > 0 && (
              <div className="p-2 rounded bg-amber-950/30 border border-amber-800/50 space-y-1" data-testid="dataset-warning-banner">
                <span className="text-[10px] font-semibold text-amber-300 flex items-center gap-1">
                  <AlertTriangle className="w-3 h-3 text-amber-400" /> Warnings
                </span>
                <ul className="text-[10px] text-amber-200/80 list-disc list-inside space-y-0.5">
                  {selectedReport.warnings.map((w, idx) => (
                    <li key={idx}>{w}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Errors List */}
            {selectedReport.errors.length > 0 && (
              <div className="p-2 rounded bg-rose-950/30 border border-rose-800/50 space-y-1" data-testid="dataset-error-banner">
                <span className="text-[10px] font-semibold text-rose-300 flex items-center gap-1">
                  <XCircle className="w-3 h-3 text-rose-400" /> Sample & Parsing Errors
                </span>
                <ul className="text-[10px] text-rose-200/90 list-disc list-inside space-y-0.5 max-h-28 overflow-y-auto font-mono">
                  {selectedReport.errors.slice(0, 5).map((err, idx) => (
                    <li key={idx}>
                      [Line/Sample {err.sample_index}] {err.message}
                    </li>
                  ))}
                  {selectedReport.errors.length > 5 && (
                    <li className="text-studio-muted italic">
                      + {selectedReport.errors.length - 5} more error(s)...
                    </li>
                  )}
                </ul>
              </div>
            )}
          </div>
        )}

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
            <label className="block text-studio-muted text-[11px] font-medium mb-1 flex items-center justify-between">
              <span>Training Split</span>
              <span className="text-[10px] font-mono text-studio-accent">{Math.round(trainRatio * 100)}%</span>
            </label>
            <select
              value={config.train_split}
              onChange={(e) => onChange({ train_split: e.target.value })}
              className="w-full bg-studio-input border border-studio-cardBorder rounded-lg px-2.5 py-1.5 text-studio-text text-xs focus:outline-none focus:border-studio-accent font-mono"
              data-testid="train-split-select"
            >
              <option value="train">train (Default)</option>
              <option value="train_80">train_80 (80% deterministic split)</option>
              <option value="train_90">train_90 (90% deterministic split)</option>
            </select>
          </div>

          <div>
            <label className="block text-studio-muted text-[11px] font-medium mb-1 flex items-center justify-between">
              <span>Evaluation Split</span>
              <span className="text-[10px] font-mono text-studio-accent">{config.eval_split === 'none' ? '0%' : `${Math.round(evalRatio * 100)}%`}</span>
            </label>
            <select
              value={config.eval_split}
              onChange={(e) => {
                const val = e.target.value;
                onChange({ eval_split: val });
                if (onSplitChange) {
                  if (val === 'none') {
                    onSplitChange(1.0, 0.0);
                  } else {
                    onSplitChange(0.9, 0.1);
                  }
                }
              }}
              className="w-full bg-studio-input border border-studio-cardBorder rounded-lg px-2.5 py-1.5 text-studio-text text-xs focus:outline-none focus:border-studio-accent"
              data-testid="eval-split-select"
            >
              <option value="none">none (Skip eval)</option>
              <option value="validation">validation (10% deterministic split)</option>
              <option value="test">test (Holdout split)</option>
            </select>
          </div>
        </div>

        {/* Deterministic Split Statistics Preview */}
        {splitResult && (
          <div className="p-2.5 rounded-lg bg-studio-bg/60 border border-studio-border/60 text-[11px] font-mono flex items-center justify-between" data-testid="split-stats">
            <span className="text-studio-muted flex items-center gap-1">
              <Sliders className="w-3.5 h-3.5 text-studio-teal" /> Disjoint Splits:
            </span>
            <div className="flex items-center gap-3">
              <span className="text-studio-text">
                Train: <strong className="text-studio-accent">{splitResult.train_count}</strong>
              </span>
              <span className="text-studio-text">
                Eval: <strong className="text-studio-accent">{splitResult.eval_count}</strong>
              </span>
              <span className="text-emerald-400 text-[10px]">
                {splitResult.has_overlap ? '⚠️ Overlap Detected' : '✓ 0% Overlap'}
              </span>
            </div>
          </div>
        )}

        {/* Collapsible Advanced Dataset Section */}
        <div className="pt-2 border-t border-studio-border/50">
          <button
            type="button"
            onClick={() => setAdvancedOpen(!advancedOpen)}
            className="flex items-center space-x-1.5 text-studio-muted hover:text-studio-text transition-colors text-[11px] font-medium"
            data-testid="dataset-advanced-toggle"
          >
            {advancedOpen ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
            <span>Advanced Dataset Formatting & Splitting Options</span>
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
              <div className="flex items-center justify-between text-[10px] text-studio-muted pt-2 border-t border-studio-border/30">
                <span className="flex items-center gap-1">
                  <Layers className="w-3 h-3 text-studio-teal" /> Deterministic split seed:
                </span>
                <span className="font-mono text-studio-text">seed=42 (PRNG shuffle)</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </Card>
  );
};
