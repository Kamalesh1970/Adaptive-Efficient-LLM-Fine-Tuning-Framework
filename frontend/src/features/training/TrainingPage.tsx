import React, { useState, useEffect, useCallback } from 'react';
import { ModelCard } from './ModelCard';
import { DatasetCard } from './DatasetCard';
import { ParametersCard } from './ParametersCard';
import { ConfigToolbar } from './ConfigToolbar';
import { RunPreviewPanel } from './RunPreviewPanel';
import { AlertBanner } from '../../components/common/AlertBanner';
import { 
  TrainingConfig, 
  SystemStatus, 
  ValidationResult,
  ModelMetadata,
  ModelValidationResponse,
  DatasetSummary,
  DatasetValidationReport,
  DatasetSplitResult
} from '../../types/training';
import { 
  fetchDefaultConfig, 
  validateConfig, 
  fetchModelCatalog, 
  validateModelSelection,
  fetchDatasets,
  uploadDatasetFile,
  fetchDatasetReport,
  fetchDatasetSplits
} from '../../services/api';

interface TrainingPageProps {
  systemStatus: SystemStatus | null;
  backendOnline: boolean;
  onRefreshSystemStatus: () => void;
}

export const TrainingPage: React.FC<TrainingPageProps> = ({
  systemStatus,
  backendOnline,
}) => {
  const [config, setConfig] = useState<TrainingConfig>({
    project_name: 'my-finetuned-model',
    model_identifier: null,
    training_method: 'lora',
    dataset_reference: null,
    dataset_subset: null,
    train_split: 'train',
    eval_split: 'none',
    training_length_type: 'steps',
    training_length: 100,
    batch_size: 2,
    context_length: 2048,
    learning_rate: 0.0002,
    advanced_enabled: false,
    hf_token_status: 'not_detected',
  });

  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  const [isResetting, setIsResetting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Phase 1 Model Catalog & Validation state
  const [catalog, setCatalog] = useState<ModelMetadata[]>([]);
  const [isLoadingCatalog, setIsLoadingCatalog] = useState(false);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [modelValidation, setModelValidation] = useState<ModelValidationResponse | null>(null);
  const [isValidatingModel, setIsValidatingModel] = useState(false);
  const [hfToken, setHfToken] = useState<string>('');

  // Phase 2 Dataset Upload, Validation & Split state
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [isLoadingDatasets, setIsLoadingDatasets] = useState(false);
  const [selectedDatasetReport, setSelectedDatasetReport] = useState<DatasetValidationReport | null>(null);
  const [isUploadingDataset, setIsUploadingDataset] = useState(false);
  const [datasetUploadError, setDatasetUploadError] = useState<string | null>(null);
  const [datasetUploadSuccess, setDatasetUploadSuccess] = useState(false);
  const [splitResult, setSplitResult] = useState<DatasetSplitResult | null>(null);
  const [trainRatio, setTrainRatio] = useState<number>(0.9);
  const [evalRatio, setEvalRatio] = useState<number>(0.1);

  // Load model catalog from backend
  const loadCatalog = useCallback(async () => {
    setIsLoadingCatalog(true);
    setCatalogError(null);
    try {
      const res = await fetchModelCatalog();
      setCatalog(res.models);
    } catch {
      setCatalogError('Could not load model catalog from backend. Operating with fallback registry.');
    } finally {
      setIsLoadingCatalog(false);
    }
  }, []);

  // Load datasets list from backend
  const loadDatasets = useCallback(async () => {
    setIsLoadingDatasets(true);
    try {
      const res = await fetchDatasets();
      setDatasets(res.datasets);
    } catch {
      // Offline fallback
    } finally {
      setIsLoadingDatasets(false);
    }
  }, []);

  useEffect(() => {
    loadCatalog();
    loadDatasets();
  }, [loadCatalog, loadDatasets]);

  // Initialize defaults from backend on mount
  useEffect(() => {
    let isMounted = true;
    async function loadDefaults() {
      try {
        const defaults = await fetchDefaultConfig();
        if (isMounted) {
          setConfig(defaults);
        }
      } catch {
        if (isMounted) {
          setErrorMessage('Could not connect to backend server. Operating in offline draft mode.');
        }
      }
    }
    loadDefaults();
    return () => {
      isMounted = false;
    };
  }, []);

  // Validate entire training configuration whenever config changes
  useEffect(() => {
    let isMounted = true;
    async function runConfigValidation() {
      try {
        const res = await validateConfig(config);
        if (isMounted) {
          setValidationResult(res);
          setErrorMessage(null);
        }
      } catch {
        // Offline validation fallback
        if (isMounted) {
          const errors: string[] = [];
          if (!config.project_name.trim()) errors.push('project_name: Project name cannot be blank');
          if (config.context_length < 128) errors.push('context_length: Must be at least 128 tokens');
          if (config.learning_rate <= 0) errors.push('learning_rate: Must be > 0.0');

          setValidationResult({
            is_valid: errors.length === 0,
            errors,
            warnings: ['Backend offline: local verification only'],
            is_ready_for_training: false,
            config,
          });
        }
      }
    }
    runConfigValidation();
    return () => {
      isMounted = false;
    };
  }, [config]);

  // Validate specific model selection and method compatibility
  useEffect(() => {
    let isMounted = true;
    if (!config.model_identifier) {
      setModelValidation(null);
      return;
    }

    async function runModelValidation() {
      setIsValidatingModel(true);
      try {
        const res = await validateModelSelection({
          model_id: config.model_identifier!,
          training_method: config.training_method,
          hf_token: hfToken.trim() || undefined,
          context_length: config.context_length,
        });
        if (isMounted) {
          setModelValidation(res);
        }
      } catch {
        if (isMounted) {
          // Offline fallback validation
          const selected = catalog.find((m) => m.id === config.model_identifier);
          if (selected) {
            const isSupported = selected.supported_methods.includes(config.training_method);
            const needsToken = selected.is_gated && !hfToken.trim() && config.hf_token_status !== 'present_in_env';
            setModelValidation({
              is_valid: isSupported && !needsToken,
              model_id: selected.id,
              training_method: config.training_method,
              compatibility_status: !isSupported 
                ? 'incompatible' 
                : needsToken 
                ? 'requires_credentials' 
                : 'compatible',
              errors: !isSupported 
                ? [`Model '${selected.id}' does not support method '${config.training_method}'`] 
                : needsToken 
                ? ['Gated model requires Hugging Face access token'] 
                : [],
              warnings: ['Backend offline: local compatibility verification only'],
              model: selected,
              token_status: hfToken.trim() ? 'configured_unverified' : 'not_provided',
              details: {},
            });
          }
        }
      } finally {
        if (isMounted) {
          setIsValidatingModel(false);
        }
      }
    }

    runModelValidation();
    return () => {
      isMounted = false;
    };
  }, [config.model_identifier, config.training_method, config.context_length, hfToken, config.hf_token_status, catalog]);

  // Handle file upload
  const handleUploadDatasetFile = async (file: File) => {
    setIsUploadingDataset(true);
    setDatasetUploadError(null);
    setDatasetUploadSuccess(false);

    try {
      const report = await uploadDatasetFile(file);
      setSelectedDatasetReport(report);
      setDatasetUploadSuccess(true);
      setConfig((prev) => ({ ...prev, dataset_reference: report.dataset_id }));

      // Refresh datasets list
      await loadDatasets();

      // Compute initial splits if dataset has valid samples
      if (report.valid_sample_count > 0) {
        try {
          const splits = await fetchDatasetSplits(report.dataset_id, trainRatio, evalRatio);
          setSplitResult(splits);
        } catch {
          // Non-blocking
        }
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Dataset upload failed';
      setDatasetUploadError(msg);
    } finally {
      setIsUploadingDataset(false);
    }
  };

  // Handle selecting an existing dataset
  const handleSelectDataset = async (datasetId: string) => {
    if (!datasetId) {
      setSelectedDatasetReport(null);
      setSplitResult(null);
      return;
    }
    try {
      const report = await fetchDatasetReport(datasetId);
      setSelectedDatasetReport(report);
      if (report.valid_sample_count > 0) {
        try {
          const splits = await fetchDatasetSplits(datasetId, trainRatio, evalRatio);
          setSplitResult(splits);
        } catch {
          // Non-blocking
        }
      }
    } catch {
      // Offline fallback
    }
  };

  // Handle split change
  const handleSplitChange = async (tRatio: number, eRatio: number) => {
    setTrainRatio(tRatio);
    setEvalRatio(eRatio);
    if (config.dataset_reference && selectedDatasetReport?.valid_sample_count) {
      try {
        const splits = await fetchDatasetSplits(config.dataset_reference, tRatio, eRatio);
        setSplitResult(splits);
      } catch {
        // Non-blocking
      }
    }
  };

  const handleUpdateConfig = (updates: Partial<TrainingConfig>) => {
    setConfig((prev) => ({ ...prev, ...updates }));
  };

  const handleResetDefaults = async () => {
    setIsResetting(true);
    try {
      const defaults = await fetchDefaultConfig();
      setConfig(defaults);
      setHfToken('');
      setSelectedDatasetReport(null);
      setSplitResult(null);
      setDatasetUploadError(null);
      setDatasetUploadSuccess(false);
    } catch {
      // Fallback reset
      setConfig({
        project_name: 'my-finetuned-model',
        model_identifier: null,
        training_method: 'lora',
        dataset_reference: null,
        dataset_subset: null,
        train_split: 'train',
        eval_split: 'none',
        training_length_type: 'steps',
        training_length: 100,
        batch_size: 2,
        context_length: 2048,
        learning_rate: 0.0002,
        advanced_enabled: false,
        hf_token_status: 'not_detected',
      });
      setHfToken('');
      setSelectedDatasetReport(null);
      setSplitResult(null);
    } finally {
      setIsResetting(false);
    }
  };

  const selectedModel = catalog.find((m) => m.id === config.model_identifier) || null;

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6" data-testid="training-page">
      {/* Backend Offline Warning Banner */}
      {!backendOnline && (
        <AlertBanner
          type="warning"
          title="Backend Disconnected"
          message={errorMessage || "FastAPI backend is not detected at http://localhost:8000. Operating in offline draft mode."}
        />
      )}

      {/* Validation Errors Alert if any */}
      {validationResult && !validationResult.is_valid && validationResult.errors.length > 0 && (
        <AlertBanner
          type="error"
          title="Configuration Parameter Errors"
          message={validationResult.errors.join(' | ')}
        />
      )}

      {/* Toolbar for Presets */}
      <ConfigToolbar
        config={config}
        onReset={handleResetDefaults}
        onLoadConfig={(loaded) => setConfig(loaded)}
        isResetting={isResetting}
      />

      {/* Main Two-Column Workflow Layout */}
      <div className="flex flex-col lg:flex-row gap-6 items-start">
        {/* Left / Central Configuration Area */}
        <div className="flex-1 space-y-6 w-full">
          {/* 1. Model Selection Card */}
          <ModelCard 
            config={config} 
            onChange={handleUpdateConfig}
            catalog={catalog}
            isLoadingCatalog={isLoadingCatalog}
            catalogError={catalogError}
            onRetryCatalog={loadCatalog}
            modelValidation={modelValidation}
            isValidatingModel={isValidatingModel}
            hfToken={hfToken}
            onHfTokenChange={setHfToken}
          />

          {/* 2. Dataset Selection Card */}
          <DatasetCard 
            config={config} 
            onChange={handleUpdateConfig}
            datasets={datasets}
            isLoadingDatasets={isLoadingDatasets}
            selectedReport={selectedDatasetReport}
            onUploadFile={handleUploadDatasetFile}

            isUploading={isUploadingDataset}
            uploadError={datasetUploadError}
            uploadSuccess={datasetUploadSuccess}
            onSelectDataset={handleSelectDataset}
            splitResult={splitResult}
            trainRatio={trainRatio}
            evalRatio={evalRatio}
            onSplitChange={handleSplitChange}
            backendOnline={backendOnline}
          />

          {/* 3. Parameters Card */}
          <ParametersCard config={config} onChange={handleUpdateConfig} />
        </div>

        {/* Right-Side Persistent Preview Panel */}
        <RunPreviewPanel
          config={config}
          systemStatus={systemStatus}
          backendOnline={backendOnline}
          validationResult={validationResult}
          selectedModel={selectedModel}
          modelValidation={modelValidation}
          selectedDatasetReport={selectedDatasetReport}
          splitResult={splitResult}
          hfToken={hfToken}
          onStartTraining={() => {
            alert('Phase 2 Notice: Dataset validated and splits configured. Execution engine scheduled for Phase 4.');
          }}
        />
      </div>
    </div>
  );
};
