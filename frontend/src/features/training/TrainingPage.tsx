import React, { useState, useEffect } from 'react';
import { ModelCard } from './ModelCard';
import { DatasetCard } from './DatasetCard';
import { ParametersCard } from './ParametersCard';
import { ConfigToolbar } from './ConfigToolbar';
import { RunPreviewPanel } from './RunPreviewPanel';
import { AlertBanner } from '../../components/common/AlertBanner';
import { TrainingConfig, SystemStatus, ValidationResult } from '../../types/training';
import { fetchDefaultConfig, validateConfig } from '../../services/api';

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

  // Validate configuration whenever config changes
  useEffect(() => {
    let isMounted = true;
    async function runValidation() {
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
    runValidation();
    return () => {
      isMounted = false;
    };
  }, [config]);

  const handleUpdateConfig = (updates: Partial<TrainingConfig>) => {
    setConfig((prev) => ({ ...prev, ...updates }));
  };

  const handleResetDefaults = async () => {
    setIsResetting(true);
    try {
      const defaults = await fetchDefaultConfig();
      setConfig(defaults);
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
    } finally {
      setIsResetting(false);
    }
  };

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
          <ModelCard config={config} onChange={handleUpdateConfig} />

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
          onStartTraining={() => {
            alert('Phase 0 Notice: Model training execution engine will be implemented in Phase 4.');
          }}
        />
      </div>
    </div>
  );
};
