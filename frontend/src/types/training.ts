export type TrainingMethod = 'lora' | 'qlora' | 'full_finetuning';

export type TrainingLengthType = 'steps' | 'epochs';

export type OptimizerType =
  | 'adamw_8bit'
  | 'adamw_torch'
  | 'paged_adamw_8bit'
  | 'paged_adamw_32bit'
  | 'adamw_hf'
  | 'sgd'
  | 'adafactor';

export type MixedPrecision = 'fp16' | 'bf16' | 'none' | 'fp32';

export type QuantizationPreference = 'nf4' | 'fp4' | 'int8' | 'none';

export interface TrainingConfig {
  project_name: string;
  model_identifier: string | null;
  training_method: TrainingMethod;
  dataset_reference: string | null;
  dataset_subset: string | null;
  train_split: string;
  eval_split: string;
  training_length_type: TrainingLengthType;
  training_duration_mode?: TrainingLengthType;
  training_length: number;
  max_steps?: number;
  num_epochs?: number;
  learning_rate: number;
  batch_size: number;
  per_device_train_batch_size?: number;
  gradient_accumulation_steps?: number;
  weight_decay?: number;
  warmup_ratio?: number;
  optimizer?: string;
  seed?: number;
  context_length: number;
  max_seq_length?: number;
  truncation?: boolean;
  packing?: boolean;
  lora_rank?: number;
  lora_alpha?: number;
  lora_dropout?: number;
  lora_target_modules?: string[];
  gradient_checkpointing?: boolean;
  mixed_precision?: string;
  quantization_type?: string | null;
  load_in_4bit?: boolean;
  load_in_8bit?: boolean;
  bnb_4bit_quant_type?: string;
  bnb_4bit_use_double_quant?: boolean;
  bnb_4bit_compute_dtype?: string;
  advanced_enabled: boolean;
  hf_token_status: string;
}

export interface GpuDeviceInfo {
  index: number;
  name: string;
  total_vram_gb: number | null;
  total_vram_mb: number | null;
  total_vram_bytes?: number | null;
  free_vram_bytes?: number | null;
}

export interface SystemStatus {
  python_version: string;
  os_platform: string;
  pytorch_available: boolean;
  pytorch_version: string | null;
  cuda_build_version?: string | null;
  cuda_available: boolean;
  cuda_version: string | null;
  gpu_available: boolean;
  device_type: string;
  gpu_name?: string | null;
  gpu_count: number;
  gpu_devices: GpuDeviceInfo[];
  primary_gpu_name: string | null;
  primary_gpu_vram_gb: number | null;
  total_vram_bytes?: number | null;
  free_vram_bytes?: number | null;
  detection_status?: string;
  diagnostic_message?: string | null;
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
  field_errors?: Record<string, string[]>;
  warnings: string[];
  is_ready_for_training: boolean;
  config: TrainingConfig | null;
  effective_batch_size_per_device?: number | null;
  total_effective_batch_size?: number | null;
  detected_device_count?: number | null;
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

export type TrainingRunStatus =
  | 'queued'
  | 'validating'
  | 'loading_model'
  | 'preparing_dataset'
  | 'training'
  | 'saving'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface TrainingMetricPoint {
  step: number;
  loss: number;
  learning_rate?: number | null;
  epoch?: number | null;
  timestamp: string;
}

export interface TrainingRunCreateRequest {
  config: TrainingConfig;
  hf_token?: string | null;
}

export interface TrainingRunResponse {
  run_id: string;
  status: TrainingRunStatus;
  config: TrainingConfig;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  current_step: number;
  max_steps: number;
  current_epoch?: number | null;
  num_epochs?: number | null;
  current_loss?: number | null;
  loss_history: TrainingMetricPoint[];
  elapsed_time_seconds: number;
  error_message?: string | null;
  device: string;
  trainable_params?: number | null;
  total_params?: number | null;
  trainable_percent?: number | null;
  cancellable: boolean;
  quantization_info?: Record<string, unknown> | null;
}

export interface TrainingArtifactMetadata {
  run_id: string;
  artifact_directory: string;
  files: string[];
  adapter_config: Record<string, unknown>;
  quantization_config?: Record<string, unknown> | null;
  trainable_params: number;
  total_params: number;
  trainable_percent: number;
  is_valid: boolean;
  verification_status: string;
  created_at: string;
}

// ─── Phase 6: Preflight Validation Types ─────────────────────────────────────

export type PreflightStatus = 'pass' | 'fail' | 'warn' | 'skip';

export interface PreflightCheckResult {
  check_id: string;
  label: string;
  status: PreflightStatus;
  message: string;
  detail?: Record<string, unknown> | null;
}

export interface PreflightCheckResponse {
  checks: PreflightCheckResult[];
  can_launch: boolean;
  blocking_count: number;
  warning_count: number;
  method: string | null;
  estimated_vram_gb: number | null;
}
