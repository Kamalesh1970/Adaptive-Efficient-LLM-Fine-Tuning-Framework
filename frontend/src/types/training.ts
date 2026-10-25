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

export type ModelVerificationStatus = 'catalog_registered' | 'verified' | 'unverified' | 'experimental';

export type CompatibilityStatus = 'compatible' | 'incompatible' | 'unknown' | 'requires_credentials';

export interface ModelMetadata {
  id: string;
  name: string;
  family: string | null;
  source: string;
  task_type: string;
  parameter_count: string | null;
  context_length: number | null;
  is_gated: boolean;
  verification_status: ModelVerificationStatus;
  supported_methods: TrainingMethod[];
  compatibility_notes: string | null;
  is_downloaded: boolean;
  is_tested: boolean;
}

export interface ModelCatalogResponse {
  models: ModelMetadata[];
  total: number;
}

export interface ModelValidationRequest {
  model_id: string;
  training_method: TrainingMethod;
  hf_token?: string | null;
  context_length?: number | null;
}

export interface ModelValidationResponse {
  is_valid: boolean;
  model_id: string | null;
  training_method: string | null;
  compatibility_status: CompatibilityStatus;
  errors: string[];
  warnings: string[];
  model: ModelMetadata | null;
  token_status: string;
  details: Record<string, unknown>;
}

export type DatasetFormat = 'json' | 'jsonl' | 'csv';

export type DatasetSchemaType = 'instruction' | 'text' | 'conversational';

export interface SampleValidationError {
  sample_index: number;
  message: string;
  sample_preview?: string | null;
}

export interface DatasetValidationReport {
  dataset_id: string;
  original_filename: string;
  file_format: DatasetFormat;
  file_size_bytes: number;
  detected_schema: DatasetSchemaType | null;
  total_samples: number;
  valid_sample_count: number;
  invalid_sample_count: number;
  empty_sample_count: number;
  duplicate_count: number | null;
  is_valid: boolean;
  warnings: string[];
  errors: SampleValidationError[];
  preview_samples: Record<string, unknown>[];
}

export interface DatasetSummary {
  id: string;
  filename: string;
  format: DatasetFormat;
  size_bytes: number;
  uploaded_at: string;
  total_samples: number;
  valid_sample_count: number;
  schema_type: DatasetSchemaType | null;
  is_valid: boolean;
}

export interface DatasetListResponse {
  datasets: DatasetSummary[];
  total: number;
}

export interface DatasetSplitResult {
  dataset_id: string;
  total_samples: number;
  train_count: number;
  eval_count: number;
  train_indices: number[];
  eval_indices: number[];
  has_overlap: boolean;
  warnings: string[];
}

