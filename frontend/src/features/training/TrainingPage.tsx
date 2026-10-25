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
  ModelValidationResponse
} from '../../types/training';
import { 
  fetchDefaultConfig, 
  validateConfig, 
  fetchModelCatalog, 
  validateModelSelection 
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

  useEffect(() => {
    loadCatalog();
  }, [loadCatalog]);

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

  const handleUpdateConfig = (updates: Partial<TrainingConfig>) => {
    setConfig((prev) => ({ ...prev, ...updates }));
  };

  const handleResetDefaults = async () => {
    setIsResetting(true);
    try {
      const defaults = await fetchDefaultConfig();
      setConfig(defaults);
      setHfToken('');
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
          <DatasetCard config={config} onChange={handleUpdateConfig} />

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
          hfToken={hfToken}
          onStartTraining={() => {
            alert('Phase 1 Notice: Model compatibility verified. Execution engine will be implemented in Phase 4.');
          }}
        />
      </div>
    </div>
  );
};
