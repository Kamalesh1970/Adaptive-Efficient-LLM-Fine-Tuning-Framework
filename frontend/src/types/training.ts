export type TrainingMethod = 'lora' | 'qlora' | 'full_finetuning';

export type TrainingLengthType = 'steps' | 'epochs';

export interface TrainingConfig {
  project_name: string;
  model_identifier: string | null;
  training_method: TrainingMethod;
  dataset_reference: string | null;
  dataset_subset: string | null;
  train_split: string;
  eval_split: string;
  training_length_type: TrainingLengthType;
  training_length: number;
  batch_size: number;
  context_length: number;
  learning_rate: number;
  advanced_enabled: boolean;
  hf_token_status: string;
}

export interface GpuDeviceInfo {
  index: number;
  name: string;
  total_vram_gb: number | null;
  total_vram_mb: number | null;
}

export interface SystemStatus {
  python_version: string;
  os_platform: string;
  pytorch_available: boolean;
  pytorch_version: string | null;
  cuda_available: boolean;
  cuda_version: string | null;
  gpu_count: number;
  gpu_devices: GpuDeviceInfo[];
  primary_gpu_name: string | null;
  primary_gpu_vram_gb: number | null;
  environment_type: string;
}

export interface HealthStatus {
  status: string;
  app_name: string;
  version: string;
}

export interface ValidationResult {
  is_valid: boolean;
  errors: string[];
  warnings: string[];
  is_ready_for_training: boolean;
  config: TrainingConfig | null;
}
