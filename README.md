# Adaptive LLM Fine-Tuning Studio

An adaptive, resource-aware Large Language Model (LLM) fine-tuning studio and training workflow engine.

Inspired by modern fine-tuning workflows (such as Unsloth), the Studio provides an ergonomic web dashboard paired with an adaptive ML backend designed to automatically select training strategies, quantization levels (e.g. QLoRA 4-bit NF4, LoRA), and memory configurations tailored to hardware constraints and dataset characteristics.

> **Phase 3 Status: COMPLETED**  
> Scope: Typed, Validated Training Parameters & Configuration Management System, Simple and Advanced Parameter Modes, LoRA/QLoRA and Method-Specific Controls, Functional YAML Save/Load Management (Secrets Excluded), Run Preview Effective Batch Size Integration, and Comprehensive Pytest/Vitest Suites.  
> *Notice: In Phase 3, training parameter configuration, schema validation, and YAML presets management are fully operational. ML training execution loops and memory profiling are planned for Phase 4.*

---

## Architecture Overview

```text
adaptive-llm-finetuning/
├── frontend/                          # Vite + React 18 + TypeScript + Tailwind CSS UI
│   ├── src/
│   │   ├── components/
│   │   │   ├── common/                # Card, Badge, AlertBanner
│   │   │   └── layout/                # Sidebar (branding & nav), Header (tabs & draft indicator)
│   │   ├── features/
│   │   │   └── training/              # ModelCard, DatasetCard, ParametersCard, ConfigToolbar, RunPreviewPanel
│   │   ├── services/
│   │   │   └── api.ts                 # Typed fetch client communicating with FastAPI
│   │   ├── types/
│   │   │   └── training.ts            # TypeScript interfaces for configs, models, hardware, and validation
│   │   ├── App.tsx                    # Root studio container
│   │   ├── App.test.tsx               # Vitest studio integration test suite
│   │   └── features/training/
│   │       └── ModelSelection.test.tsx # Vitest Phase 1 model selection test suite
│   ├── package.json
│   ├── tailwind.config.js             # Curated dark slate/charcoal studio theme
│   └── vite.config.ts                 # Dev server with /api proxy to FastAPI
│
├── backend/                           # FastAPI Application & Microservices
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py              # GET /api/health, GET /api/system/status, GET/POST /api/models/*
│   │   ├── core/
│   │   │   ├── config.py              # Host, port, debug, CORS settings
│   │   │   └── system.py              # Hardware probe (Python, PyTorch, CUDA, VRAM, CPU fallback)
│   │   ├── schemas/
│   │   │   ├── system.py              # HealthResponse, SystemStatusResponse, GpuDeviceInfo
│   │   │   ├── training.py            # TrainingConfigSchema, ValidationResult, TrainingMethod
│   │   │   └── models.py              # ModelMetadata, ModelCatalogResponse, ModelValidationResponse
│   │   ├── services/
│   │   │   ├── config_service.py      # Baseline defaults and configuration validation
│   │   │   └── model_catalog.py       # Modular model catalog registry and compatibility rules
│   │   └── main.py                    # FastAPI app factory with CORS middleware
│   └── tests/                         # Pytest test suite (73 unit and integration tests)
│       ├── test_app.py                # Startup and health check
│       ├── test_config.py             # Config defaults, validation, and invalid values
│       ├── test_system.py             # Real probe and mocked CUDA / CPU / no-PyTorch environments
│       └── test_models.py             # Model catalog, lookup, validation, security, and error handling
│
├── core/                              # Base ML foundation utilities (seed, logger, config)
├── intelligence/                      # Dataset & tokenization intelligence pipeline
├── configs/
│   └── default_config.yaml            # Baseline YAML configuration
├── docs/
│   └── architecture.md                # System architectural specification
├── .env.example                       # Environment configuration template
├── pyproject.toml                     # Modern Python project metadata & Pytest configuration
├── README.md                          # Project documentation
└── requirements.txt                   # Backend Python dependencies
```

---

## Phase 1: Model Selection & Compatibility

### 1. Backend Model Catalog Structure

The catalog is modular and maintained in `backend/app/services/model_catalog.py`, separated cleanly from model-loading code or runtime weight allocations.

Each `ModelMetadata` schema entry contains validated attributes:
- **`id`**: Unique model identifier (e.g. `meta-llama/Meta-Llama-3-8B`, `mistralai/Mistral-7B-v0.3`).
- **`name`**: Human-readable display name.
- **`family`**: Architecture family (`llama`, `qwen2`, `phi3`, `mistral`), or `None` if unknown.
- **`source`**: Registry source (`huggingface`, `local`).
- **`task_type`**: Supported task (`causal_lm`).
- **`parameter_count`**: Parameter size (`8.0B`, `0.5B`), or `None` if unknown.
- **`context_length`**: Maximum verified token context length, or `None` if unverified.
- **`is_gated`**: Boolean flag indicating if gated access / license approval is required.
- **`verification_status`**: Registration status (`catalog_registered`, `verified`, `unverified`, `experimental`).
- **`supported_methods`**: Compatible fine-tuning methods (`lora`, `qlora`, `full_finetuning`).
- **`compatibility_notes`**: Architecture or license notes.
- **`is_downloaded`**: Strictly `False` in Phase 1 (no weights downloaded).
- **`is_tested`**: Strictly `False` in Phase 1 (no training executed).

### 2. Representative Model Catalog Entries

The catalog includes standard foundation models as well as lightweight models for constrained-VRAM environments:
- **`meta-llama/Meta-Llama-3-8B`**: Gated access, 8.0B parameters, 8,192 verified context length.
- **`meta-llama/Llama-2-7b-hf`**: Gated access, 6.7B parameters, 4,096 verified context length.
- **`mistralai/Mistral-7B-v0.3`**: Open weights, 7.2B parameters, 32,768 verified context length.
- **`Qwen/Qwen2.5-7B`**: Open weights, 7.6B parameters, 32,768 verified context length.
- **`microsoft/Phi-3-mini-4k-instruct`**: Open weights, 3.8B parameters, 4,096 verified context length.
- **`Qwen/Qwen2.5-0.5B`**: Open weights, 0.5B parameters, primary candidate for limited-VRAM (< 6 GB) environments and CPU testing.
- **`HuggingFaceTB/SmolLM-135M`**: Open weights, 135M parameters, ultra-lightweight candidate for rapid prototyping.
- **`google/gemma-2-2b`**: Gated access, 2.6B parameters, restricted to LoRA and QLoRA (`full_finetuning` unsupported).
- **`custom/unverified-experimental-lm`**: Catalog entry with unverified metadata (`family=None`, `context_length=None`, `parameter_count=None`, verification status `unverified`).

### 3. API Endpoints and Schemas

#### `GET /api/models`
- **Description**: Returns all registered models in the catalog.
- **Response**: `ModelCatalogResponse` (`models: List[ModelMetadata]`, `total: int`).

#### `GET /api/models/{model_id:path}`
- **Description**: Returns verified metadata for a specific model.
- **Status Codes**:
  - `200 OK`: Model found and metadata returned.
  - `400 Bad Request`: Model identifier is empty or whitespace.
  - `404 Not Found`: Model identifier not recognized in catalog.

#### `POST /api/models/validate`
- **Description**: Validates a requested model selection, fine-tuning method, context length, and credentials.
- **Request Body**: `ModelValidationRequest` (`model_id: str`, `training_method: str`, `hf_token: Optional[str]`, `context_length: Optional[int]`).
- **Response Body**: `ModelValidationResponse` (`is_valid: bool`, `compatibility_status: CompatibilityStatus`, `errors: List[str]`, `warnings: List[str]`, `model: Optional[ModelMetadata]`, `token_status: str`, `details: Dict[str, Any]`).

### 4. Compatibility Validation Behavior

1. **Unknown Model Identifier**: Returns `is_valid: false`, `compatibility_status: "incompatible"`, error indicating model is not registered.
2. **Empty Model Identifier**: Returns `is_valid: false`, error indicating model identifier cannot be empty.
3. **Invalid Training Method**: Validates against allowed methods (`lora`, `qlora`, `full_finetuning`). Returns error if unrecognized.
4. **Unsupported Model/Method Combination**: Checks requested method against model's `supported_methods`. If unsupported (e.g. `google/gemma-2-2b` + `full_finetuning`), returns `is_valid: false`, `compatibility_status: "incompatible"`.
5. **Unknown Compatibility**: If model has `verification_status == "unverified"` and method compatibility is not recorded, returns `compatibility_status: "unknown"` with cautionary notices.
6. **Gated Credentials Requirement**: If a model requires gated access (`is_gated: true`) and no token is supplied or present in environment, returns `is_valid: false`, `compatibility_status: "requires_credentials"`.
7. **Context Length Validation**: Warns if requested context length exceeds verified model limits, and errors if context length <= 0.

### 5. Hugging Face Token Security Decisions

- **Zero Secret Leakage**: Access tokens are NEVER echoed in API responses, error details, logs, or exception stack traces.
- **No Persistent Storage**: Tokens entered in the frontend are maintained exclusively in React component memory. They are NEVER written to `localStorage`, `sessionStorage`, or cookies.
- **Masked Input**: The UI input defaults to masked password mode (`type="password"`) with an eye toggle.
- **Clear Status Distinction**: Distinguishes "Not Configured", "Configured (In-Memory, Unverified)", and "Detected in Environment (Unverified)".
- **Offline / Non-testing of Real Credentials**: Phase 1 does not transmit tokens to external Hugging Face servers or verify real credentials.

### 6. How to Add a Catalog Entry

To register an additional model programmatically or statically, add a new `ModelMetadata` instance to `backend/app/services/model_catalog.py`:

```python
from backend.app.schemas.models import ModelMetadata, ModelVerificationStatus
from backend.app.schemas.training import TrainingMethod

new_model = ModelMetadata(
    id="organization/model-name",
    name="Model Display Name",
    family="llama",                      # Architecture family, or None if unknown
    source="huggingface",
    task_type="causal_lm",
    parameter_count="7.0B",              # Parameter count, or None if unknown
    context_length=4096,                 # Maximum verified context, or None if unverified
    is_gated=False,                      # Set True if gated license required
    verification_status=ModelVerificationStatus.CATALOG_REGISTERED,
    supported_methods=[
        TrainingMethod.LORA,
        TrainingMethod.QLORA,
        TrainingMethod.FULL_FINETUNING,
    ],
    compatibility_notes="Verified community model entry.",
    is_downloaded=False,                 # Strictly False until downloaded
    is_tested=False,                     # Strictly False until executed
)

# Register into the catalog service
service = get_model_catalog_service()
service.register_model(new_model)
```

### 7. Known Limitations

- **No Weight Loading**: Phase 1 manages metadata and compatibility rules only. Actual model weights are not downloaded or mapped into GPU memory.
- **Unverified Live Credentials**: Token verification against the Hugging Face Hub API is scheduled for future phases; tokens are treated as unverified.
- **Static VRAM Notice**: Hardware and VRAM estimation is planned for Phase 2; Phase 1 does not dynamically compute memory footprints.
- **No Training Execution**: The "Start Fine-Tuning" action is strictly disabled until Phase 4 execution loops are built.

---

## Phase 2: Dataset Upload, Validation & Split Selection

### 1. Supported File Formats

The Studio backend and training interface support local upload and validation for three standardized structured formats:
- **JSON (`.json`)**: Supports top-level JSON array of sample objects `[...]` or standard wrapped dictionaries containing sample lists (e.g. `{"data": [...]}`).
- **JSON Lines (`.jsonl`)**: Validates every non-empty line independently as valid JSON, with exact line-number error reporting on syntax or schema errors.
- **Comma-Separated Values (`.csv`)**: Validates CSV header structure and row formatting using Python's `csv.DictReader`.

*Note: Unsupported file formats (such as `.parquet`, `.arrow`, or plain `.txt`) are strictly rejected with an informative HTTP 400 response specifying supported extensions.*

### 2. Supported Fine-Tuning Schemas

Uploaded datasets are automatically analyzed using structural heuristics to detect fine-tuning sample representations:
- **Instruction Format (`instruction`)**:
  - Requires `instruction` (string) and `output` / `response` (string).
  - Optionally accepts `input` / `context` (string).
- **Conversational Format (`conversational`)**:
  - Requires `messages` or `conversations` (array of message objects).
  - Each message object must contain `role` (`system`, `user`, `assistant`) and non-empty `content`.
- **Text Format (`text`)**:
  - Requires a single primary text field (`text`, `content`, or `prompt`).

### 3. Normalization & Duplicate Detection

- **Normalization**: Preserves source records while standardizing fields into a deterministic internal format, retaining 0-indexed sample references for error reporting.
- **Duplicate Detection**: Computes SHA-256 fingerprints across serialized samples to count exact duplicate entries.

### 4. Upload Security & Storage Configuration

- **Storage Location**: Uploads are saved under `datasets/uploads/` (configurable via `AppSettings.dataset_upload_dir`), strictly isolated from public frontend assets.
- **Path Traversal Prevention**: User filenames are sanitized using `os.path.basename` and stored under server-generated UUID identifiers (`ds_<uuid>`).
- **File Size Limits**: Configurable upload threshold (default 50 MB / 52,428,800 bytes). Files exceeding the limit are rejected immediately.
- **Data Protection**: Internal filesystem paths are never disclosed in API responses.

### 5. Train / Evaluation Split Selection

- **Disjoint Split Guarantee**: Generates non-overlapping training and evaluation splits using deterministic seeded shuffling (default seed: 42) ensuring **0% data leakage**.
- **Minimum Size Safeguards**: Datasets with insufficient samples for meaningful evaluation (e.g. < 5 samples) display a visible UI warning rather than falsely claiming evaluation readiness.

### 6. Phase 2 Known Limitations

- **No Parquet Support**: Parquet and Arrow binary datasets are not supported in Phase 2.
- **No Tokenization Pipeline**: Token counting and truncation are planned for Phase 3.
- **No Training Loop**: Training execution is deferred to Phase 4.

---

## Installation

### 1. Backend Setup

Create and activate a Python virtual environment, then install dependencies:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Frontend Setup

Install frontend packages:

```bash
cd frontend
npm install
cd ..
```

---

## Phase 3: Training Parameters & Configuration Management

### 1. Configuration Schema

The training configuration system is typed, validated with Pydantic in the backend (`backend/app/schemas/training.py`), and mirrored in TypeScript (`frontend/src/types/training.ts`):

- **General Settings**:
  - `project_name` (`str`): Project and output directory identifier.
  - `model_identifier` (`Optional[str]`): Selected model reference from catalog or Hugging Face.
  - `training_method` (`TrainingMethod`): Fine-tuning strategy: `lora`, `qlora`, or `full_finetuning`.
  - `dataset_reference` (`Optional[str]`): Uploaded or configured dataset reference.
  - `dataset_subset` (`Optional[str]`): Optional dataset configuration subset.
  - `train_split` (`str`): Training partition identifier (default: `train`).
  - `eval_split` (`str`): Evaluation partition identifier (default: `none`).
  - `training_duration_mode` / `training_length_type` (`TrainingLengthType`): Duration mode: `steps` or `epochs`.
  - `max_steps` (`int`): Maximum training steps (when mode is `steps`).
  - `num_epochs` (`float`): Number of training epochs (when mode is `epochs`).

- **Optimization Settings**:
  - `learning_rate` (`float`): Optimizer learning rate (default: `0.0002`).
  - `per_device_train_batch_size` / `batch_size` (`int`): Per-device micro-batch size (default: `2`).
  - `gradient_accumulation_steps` (`int`): Number of gradient accumulation steps (default: `1`).
  - `weight_decay` (`float`): Regularization weight decay (default: `0.01`).
  - `warmup_ratio` (`float`): Warmup step ratio (default: `0.03`).
  - `optimizer` (`str`): Optimizer identifier (`adamw_8bit`, `adamw_torch`, `paged_adamw_8bit`, `paged_adamw_32bit`, `adamw_hf`, `sgd`, `adafactor`).
  - `seed` (`int`): Reproducibility random seed (default: `42`).

- **Sequence Settings**:
  - `context_length` / `max_seq_length` (`int`): Token sequence length (default: `2048`, limits: `128` - `131,072`).
  - `truncation` (`bool`): Input truncation toggle (default: `true`).
  - `packing` (`bool`): Sequence packing preference flag (default: `false`).

- **LoRA Adapter Settings**:
  - `lora_rank` (`int`): Adapter rank dimension (default: `16`, limits: `1` - `512`).
  - `lora_alpha` (`int`): Adapter scaling factor (default: `32`, limits: `1` - `1,024`).
  - `lora_dropout` (`float`): Adapter dropout probability (default: `0.05`, limits: `0.0` - `1.0`).
  - `lora_target_modules` (`List[str]`): Projection layers for weight updates (default: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).

- **Memory and Precision Preferences**:
  - `gradient_checkpointing` (`bool`): Activation recomputation preference (default: `true`).
  - `mixed_precision` (`str`): Precision preference (`fp16`, `bf16`, `fp32`, `none`).
  - `quantization_type` (`Optional[str]`): QLoRA quantization preference (`nf4`, `fp4`, `int8`, `none`).

### 2. Validation Rules & Field-Specific Errors

Server-side validation enforces strict constraints:
- **Project Name**: Non-empty, stripped, length $\le$ 128 characters.
- **Positive Numerics**: Strictly positive (> 0) values for step count, epoch count, batch sizes, gradient accumulation, context length, learning rate, LoRA rank, and LoRA alpha.
- **Ranges**: Warmup ratio and weight decay must be in $[0.0, 1.0]$. LoRA dropout in $[0.0, 1.0]$. Context length in $[128, 131072]$.
- **Rejection of Non-Finite Values**: NaN, Infinity, -Infinity strings and float values are strictly rejected.
- **Method-Specific Rules**:
  - `lora` and `qlora` require non-empty `lora_target_modules`.
  - `qlora` requires an explicit `quantization_type` (`nf4`, `fp4`, `int8`).
  - `full_finetuning` disables and does not require LoRA adapter parameters.
- **Model Metadata Compatibility**: Distinguishes definite incompatibility (e.g. model does not support method) from unverified compatibility (model not in catalog).
- **Field-Level Errors**: The `/api/training/config/validate` response includes a `field_errors` dictionary mapping field names to clear messages, rendered beneath each input.

### 3. Configuration API Endpoints

- `GET /api/training/config/defaults`: Returns baseline configuration with no model and no dataset selected.
- `POST /api/training/config/validate`: Validates configuration payload, returning `is_valid`, `errors`, `field_errors`, `warnings`, and effective batch size metrics.
- `POST /api/training/config/reset`: Resets training configuration to defaults without touching uploaded datasets or model catalog.
- `POST /api/training/config/yaml/export`: Serializes configuration to sanitized YAML string (scrubbing secrets and tokens).
- `POST /api/training/config/yaml/parse`: Safely parses uploaded YAML using `yaml.safe_load` and validates against schema.

### 4. Simple and Advanced Modes

- **Simple Mode**:
  - Project name input.
  - Steps vs Epochs switcher with dynamic unit labels and inputs.
  - Per-device batch size and gradient accumulation steps inputs.
  - Context length selector with model verified context notice.
  - Learning rate input.
- **Advanced Mode**:
  - Fine-Tuning method adapter controls: LoRA rank, alpha, dropout, and comma-separated target modules.
  - Method-aware disabled state for full fine-tuning.
  - Explicit quantization preference dropdown for QLoRA.
  - Optimizer selection dropdown (`adamw_8bit`, `adamw_torch`, etc.).
  - Weight decay, warmup ratio, and random seed inputs.
  - Memory and precision toggles: gradient checkpointing, mixed precision (`fp16`, `bf16`, `fp32`, `none`), sequence packing preference, and input truncation.

### 5. YAML Configuration Management

- **Export YAML**: Downloads current configuration to `.yaml` file. All secrets, API keys, and passwords are unconditionally scrubbed.
- **Load YAML**: Parses user-uploaded YAML file safely without executing arbitrary Python constructors. Returns clear validation feedback if malformed or invalid.
- **Reset Defaults**: Resets configuration to clean baseline defaults while preserving uploaded datasets and model catalog files.

### 6. Run Preview Integration

The right-side preview panel reflects actual shared state in real-time:
- Model and dataset selection status.
- Selected fine-tuning method.
- Duration in steps or epochs.
- Effective batch size calculation:
  $$\text{Effective Batch Size} = \text{per\_device\_batch\_size} \times \text{gradient\_accumulation\_steps} \times \text{number\_of\_devices}$$
  When device count is unknown (CPU or offline), displays the per-device batch and explicitly notes that total is unknown.
- Context length and learning rate.
- Relevant advanced settings summary (LoRA rank/alpha, quantization, optimizer, precision, gradient checkpointing).
- Hardware detection (only when detected by backend).
- Real validation status badge.
- **Start Fine-Tuning Button**: Strictly disabled with lock icon and tooltip: "Execution engine scheduled for Phase 4."

### 7. Known Limitations

- **No Training Execution**: Phase 3 provides parameter configuration, validation, and YAML presets only. Training execution loops (`transformers`, `peft`, `accelerate`) are scheduled for Phase 4.
- **No Memory Profiling**: Hardware VRAM memory estimation is planned for a future phase; no fabricated estimates are shown.

---

## How to Start the Application

### 1. Start the Backend API Server

From the project root:

```bash
source .venv/bin/activate
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Endpoint: [http://localhost:8000/api/health](http://localhost:8000/api/health)
- Models Catalog: [http://localhost:8000/api/models](http://localhost:8000/api/models)
- Datasets Upload: `POST /api/datasets/upload`
- Datasets List: `GET /api/datasets`
- Dataset Details: `GET /api/datasets/{dataset_id}`
- Dataset Validation: `POST /api/datasets/{dataset_id}/validate`
- Dataset Splits: `GET /api/datasets/{dataset_id}/splits`

### 2. Start the Frontend Development Server

In a separate terminal:

```bash
cd frontend
npm run dev
```

Open your browser at:
[http://localhost:5173](http://localhost:5173)

---

## How to Run Tests

### 1. Backend Tests (Pytest)

Run all backend unit and integration tests:

```bash
# Run quiet test summary across full test suite
pytest -q

# Run specific Phase 2 dataset test suite
pytest backend/tests/test_datasets.py -v

# Run specific model catalog test suite
pytest backend/tests/test_models.py -v
```

### 2. Frontend Tests (Vitest)

Run Vitest unit and integration tests:

```bash
cd frontend
npm test
```

### 3. Frontend Production Build Check

Verify TypeScript compilation and asset bundling:

```bash
cd frontend
npm run build
```

---

## Phase 4: Real LoRA Fine-Tuning Engine

Phase 4 implements a genuine, end-to-end LoRA fine-tuning engine for causal language models.

### 1. Supported Model Requirements
- **Architecture**: Decoder-only Causal Language Models (e.g. Llama 3, Mistral, Gemma, Qwen, TinyLlama).
- **Format**: Standard Hugging Face model repository or local directory.
- **Safety**: Untrusted remote code execution (`trust_remote_code`) is disabled by default. Directory traversal tokens (`..`) are strictly rejected.
- **Gated Models**: Requires Hugging Face credentials (`HF_TOKEN`). Tokens are kept strictly in memory or environment and never leaked into saved artifacts or API telemetry.

### 2. LoRA Adapter Configuration
- **Hyperparameters**:
  - `lora_rank` (r): 1 to 512 (default: 16)
  - `lora_alpha`: 1 to 1024 (default: 32)
  - `lora_dropout`: 0.0 to 1.0 (default: 0.05)
  - `lora_target_modules`: Target linear projection layers (e.g. `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`).
- **Adapter Injection**:
  - Inspects base model modules to verify target layers exist (`TargetModulesNotFoundError` if absent).
  - Freezes base model parameters (`requires_grad = False`).
  - Enables gradients only on LoRA matrices (`requires_grad = True`).
  - Calculates exact total parameters, trainable parameters, and trainable percentage.

### 3. Dataset Preparation
- Consumes normalized datasets from Phase 2 with deterministic train/eval split partitions (0% data leakage).
- **Instruction Format**: Formats task prompts with instruction, optional input context, and response.
- **Conversational Format**: Uses the tokenizer's chat template if available, with a standard ChatML fallback (`<|im_start|>role\ncontent<|im_end|>`).
- **Text Format**: Formats raw text samples for autoregressive modeling.
- **Sequence Truncation**: Enforces configured `context_length` (e.g. 2048).
- **Causal LM Examples**: Constructs `input_ids`, `attention_mask`, and aligned `labels` for cross-entropy optimization.

### 4. Training Engine & Telemetry
- Supports PyTorch with AdamW optimizer and synthetic test model execution for zero-download test suites.
- Tracks real step progress, current loss trajectory, learning rate, epoch count, and elapsed runtime.
- **Cancellation**: Supports responsive cancellation between optimization steps without artifact corruption.

### 5. Artifact Saving & Verification
- Output Directory: `artifacts/runs/{run_id}`
- Saved Artifacts:
  - `adapter_config.json`: PEFT LoRA adapter configuration.
  - `adapter_model.safetensors` / `adapter_model.bin`: Adapter weights.
  - `training_config.json`: Sanitized training hyperparameters (credentials redacted).
  - `metrics.json`: Full telemetry loss trajectory and step times.
  - `run_metadata.json`: Final metrics, parameter counts, execution device, and timestamps.
- **Disk Verification**: Verifies adapter weights exist and reloads adapter onto base model before marking run as `completed`.

### 6. Hardware & Dependency Considerations
- **Target GPU**: Tested for lightweight workstation cards including NVIDIA RTX 2050 (4 GB VRAM) using small batch sizes (`batch_size=1` or `2`, `gradient_accumulation_steps=4`).
- **CPU Execution**: Automatically selected when GPU is absent or disabled.
- **Optional ML Stack**: PyTorch, Transformers, and PEFT are probed safely. Unit tests run offline using synthetic models without GPU or internet.

### 7. Known Limitations (Phase 4 Boundaries)
- Full fine-tuning is deferred to future roadmap milestones.

---

## Phase 5: QLoRA and 4-Bit Quantization

Phase 5 extends the LoRA training engine with genuine QLoRA 4-bit and 8-bit quantization capabilities powered by `bitsandbytes` and `peft`.

### 1. LoRA versus QLoRA
| Feature | LoRA | QLoRA |
|---|---|---|
| **Base Model Precision** | 16-bit float (FP16 / BF16) or 32-bit | 4-bit (NF4 / FP4) or 8-bit (Int8) |
| **VRAM Footprint** | Full model weights (~16GB for 8B model) | Compressed weights (~4–5GB for 8B model) |
| **Adapter Parameters** | Trainable FP16/BF16 low-rank matrices | Trainable FP16/BF16 low-rank matrices |
| **Base Weight Freezing** | Base weights frozen (`requires_grad=False`) | Quantized weights frozen; gradients backprop through dequantization |
| **Hardware Requirement** | CUDA GPU or CPU | **CUDA GPU Required** (bitsandbytes 4-bit requires CUDA) |

### 2. Quantization Configuration
Supported parameters in `QuantizationConfigSchema` and `TrainingConfigSchema`:
- `load_in_4bit`: Enable 4-bit base model precision (default: `True`).
- `load_in_8bit`: Enable 8-bit base model precision (`load_in_4bit` and `load_in_8bit` are mutually exclusive).
- `bnb_4bit_quant_type`: Quantization format:
  - `nf4`: NormalFloat4 (statistically optimal for zero-mean, normally distributed neural network weights).
  - `fp4`: Float4 (standard 4-bit floating point).
- `bnb_4bit_use_double_quant`: Double / nested quantization compresses the quantization constants themselves, saving ~0.37 bits per parameter with no loss in model quality (default: `True`).
- `bnb_4bit_compute_dtype`: Computation precision for dequantized forward pass:
  - `fp16`: Float16 (standard for Pascal, Turing, Ampere GPUs).
  - `bf16`: BFloat16 (recommended for Ampere, Ada Lovelace, and Hopper architectures).
  - `float32`: 32-bit single precision.

### 3. Supported Hardware and Software Requirements
- **Operating System**: Linux (x86_64).
- **GPU Architecture**: NVIDIA CUDA-capable GPU with Compute Capability >= 7.0 (Turing or newer recommended).
- **VRAM**: Minimum 6 GB VRAM recommended for 7B/8B models with 4-bit NF4, batch size 1, and gradient checkpointing.
- **CPU Execution**: CPU execution is **not supported** for bitsandbytes 4-bit quantization. If CUDA hardware is absent, the backend rejects real QLoRA loading with an actionable `QuantizationDeviceError` / `CudaUnavailableForQuantizationError`, and the frontend UI disables the launch button with an honest explanation.

### 4. Dependency Installation
To install the optional GPU quantization dependencies:
```bash
pip install bitsandbytes accelerate
```

### 5. Running QLoRA Training
1. In the Studio UI, select a base model (e.g. `meta-llama/Meta-Llama-3-8B`) and a validated dataset.
2. Under **Fine-Tuning Method**, select `QLoRA`.
3. In **Parameters & Hyperparameters**:
   - Choose quantization format (`NF4` or `FP4`).
   - Enable or disable **Double Quantization**.
   - Select compute dtype (`FP16` or `BF16`).
   - Configure LoRA Rank ($r$), Alpha ($\alpha$), and Target Modules.
4. Verify the **Run Summary Preview** indicates `Supported` under QLoRA Compatibility.
5. Click **Start Fine-Tuning** to launch the asynchronous training job.

### 6. Artifact Format
QLoRA runs save PEFT-compatible artifacts in `artifacts/runs/{run_id}`:
- `adapter_config.json`: PEFT adapter hyperparameters ($r$, $\alpha$, target modules, task type).
- `adapter_model.safetensors`: Trained adapter weights.
- `training_config.json`: Sanitized training configuration (credentials excluded).
- `quantization_config.json`: Quantization metadata (`load_in_4bit`, `bnb_4bit_quant_type`, `bnb_4bit_use_double_quant`, `bnb_4bit_compute_dtype`).
- `metrics.json`: Step-by-step training loss and learning rate trajectory.
- `run_metadata.json`: Run status, device, trainable parameters count, and final loss.

### 7. Testing Commands
Run all Phase 5 backend tests:
```bash
pytest backend/tests/test_qlora.py -v
```

Run full backend suite:
```bash
pytest -q
```

Run full frontend suite:
```bash
cd frontend && npm test -- --run
```

Run frontend production build:
```bash
cd frontend && npm run build
```

---

## Phase 6: Training Run Preview, Resource Validation & Launch Control

### 1. Run Preview Behavior
The persistent **Run Summary Preview** panel integrates live backend verification data to provide real-time pre-launch transparency:
- **Selected Model Identifier**: Shows active model ID, parameter scale, family, and compatibility badges.
- **Selected Dataset Reference**: Displays uploaded or hub dataset, valid samples count, detected schema, and train/eval splits.
- **Fine-Tuning Method**: Displays selected method (`LoRA`, `QLoRA`, `Full Weights`).
- **Training Duration**: Displays step count or epoch count.
- **Effective Batch Size**: Dynamically computed as $\text{per\_device\_batch\_size} \times \text{gradient\_accumulation\_steps} \times \text{detected\_gpu\_count}$.
- **Context Length & Learning Rate**: Verified token limit and optimizer learning rate.
- **LoRA / Quantization Configuration**: Adapter rank ($r$), alpha ($\alpha$), dropout, target modules, quant format (`NF4`/`FP4`), double quantization, and compute dtype.
- **Actual Hardware & Device**: Genuine detected GPU (e.g. `NVIDIA GeForce RTX 2050 (3.68 GB VRAM)`), visible device count, or explicit `CPU Mode`.
- **Dependency Compatibility**: Explicit status (`Validated`, `Invalid`, `Unsupported`, `Unknown`, `Not yet checked`) reflecting PyTorch, transformers, PEFT, and bitsandbytes availability.
- **Validation Warnings**: Aggregates non-blocking warnings across configuration, dataset, model, and preflight checks.
- **Launch Readiness**: Unambiguous status indicator (`Validated`, `Invalid`, `Unsupported`, `Unknown`, `Not yet checked`).

### 2. Pre-Launch Validation Service (Preflight)
The preflight engine (`backend/app/services/preflight_service.py`) executes an ordered sequence of validation checks before any training run is initiated:
1. **Configuration Schema**: Validates parameter types, learning rates, batch sizes, and context limits against Pydantic schema constraints.
2. **Model Selection & Registration**: Verifies model exists in local catalog or hub, confirms method compatibility, checks gated access requirements, and verifies context limits.
3. **Dataset Validation**: Verifies dataset is registered, formatted, and contains > 0 valid samples. Blocks on invalid sample errors.
4. **Compute Device & CUDA Availability**: Inspects actual PyTorch CUDA availability, device count, device names, and VRAM capacity. Accurately flags detection errors.
5. **Method–Device Compatibility**: Enforces that QLoRA 4-bit fine-tuning requires CUDA hardware. Blocks QLoRA on CPU-only hosts with actionable guidance.
6. **Required Training Dependencies**: Checks for core trio (`torch`, `transformers`, `peft`). Distinguishes missing core dependencies from missing quantization libraries.
7. **Quantization Dependencies**: Verifies `bitsandbytes` installation when QLoRA is selected; skips when LoRA is selected. Missing `bitsandbytes` is never conflated with missing GPU hardware.
8. **Output Directory Permissions**: Tests filesystem write permissions for artifacts directory using sentinel write checks.
9. **Resource Compatibility Assessment (VRAM)**: Computes heuristic memory estimates based on model parameter scale, precision, and batch size. If parameter scale or hardware is unknown, reports VRAM compatibility honestly as `unknown`. Never fabricates memory guarantees.
10. **Hugging Face Token Status**: Checks for environment or in-memory credentials when gated models (e.g. Llama-3) are selected.

### 3. Structured RunPlan
`POST /api/training/preview` generates a serializable `RunPlan` object containing:
- `run_configuration`: Sanitized training hyperparameters (passwords, tokens, and secret keys strictly redacted).
- `selected_model`: Model metadata summary.
- `dataset_reference`: Dataset validation report summary.
- `fine_tuning_method`: Target fine-tuning method (`lora`, `qlora`, `full_finetuning`).
- `device_information`: Verified compute hardware specifications.
- `validation_results`: Complete ordered check results (`checks`, `can_launch`, `blocking_count`, `warning_count`).
- `warnings`: Aggregated non-blocking warnings.
- `resource_compatibility`: Structured breakdown of dependencies, VRAM estimate, and compatibility status.
- `output_configuration`: Safe relative artifact paths and logging settings.
- `can_launch`: Boolean readiness determination.

### 4. Launch Control Lifecycle & Duplicate Prevention
Training launch (`POST /api/training/runs`) enforces server-side validation and idempotency:
1. **Server-Side Verification**: Runs preflight validation on the backend. If any blocking check fails, returns HTTP 400 with detailed blocking issues.
2. **Explicit User Confirmation**: Preflight checklist modal requires explicit user review and confirmation before the launch request is dispatched.
3. **Duplicate Submission Prevention**: Clients transmit a unique `idempotency_key`. Repeated requests with the same key return the existing training job without starting a duplicate process.
4. **Actual Run Identifier**: Returns the real server-generated `run_id` (e.g. `run_20261028_...`). The frontend transitions directly to `CurrentRunView` to stream genuine telemetry.
5. **Launch Control States**:
   - `Configuration incomplete`: Model or dataset not selected (button disabled).
   - `Validation in progress`: Validating parameters or running preflight checks (disabled with spinner).
   - `Blocked by validation`: Blocking issues detected (disabled with explanation).
   - `Ready for confirmation`: All checks passed, ready for user review (button enabled).
   - `Launching`: Request in flight (disabled with spinner).
   - `Launch failed`: Request failed or rejected by server (shows error banner, allows retry).
   - `Run created`: Genuine run created and displayed.

### 5. API Endpoints

#### `POST /api/training/preflight`
- **Description**: Executes pre-launch preflight resource and configuration checks.
- **Request**: `TrainingConfigSchema` (with optional `hf_token`).
- **Response**: `PreflightCheckResponse` (`checks`, `can_launch`, `blocking_count`, `warning_count`, `estimated_vram_gb`, `vram_status`, `vram_message`).

#### `POST /api/training/preview`
- **Description**: Generates a structured, secret-sanitized `RunPlan`.
- **Request**: `{"config": TrainingConfigSchema, "hf_token": Optional[str]}`.
- **Response**: `RunPlan`.

#### `POST /api/training/validate-run`
- **Description**: Validates a training run and returns preflight checks alongside the generated `RunPlan`.
- **Request**: `{"config": TrainingConfigSchema, "hf_token": Optional[str]}`.
- **Response**: `RunValidationResponse` (`can_launch`, `blocking_count`, `warning_count`, `checks`, `plan`).

#### `POST /api/training/runs`
- **Description**: Validates configuration, enforces preflight checks, prevents duplicate submissions via `idempotency_key`, and launches training asynchronously.
- **Request**: `TrainingRunCreateRequest` (`config`, `hf_token`, `idempotency_key`).
- **Status Codes**:
  - `201 Created`: Training job registered and launched.
  - `400 Bad Request`: Configuration invalid or rejected by preflight validation.

### 6. Testing Commands

Run Phase 6 backend test suite:
```bash
pytest backend/tests/test_phase6_preflight_and_launch.py -v
```

Run preflight endpoint unit tests:
```bash
pytest backend/tests/test_preflight.py -v
```

Run full backend test suite:
```bash
pytest -q
```

Run Phase 6 frontend test suite:
```bash
cd frontend && npm test -- src/features/training/LaunchControlAndPreview.test.tsx --run
```

Run full frontend test suite:
```bash
cd frontend && npm test -- --run
```

Run frontend production build:
```bash
cd frontend && npm run build
```

---

## Phase 7: Live Training Monitoring, Event Streaming & Run History

### 1. Telemetry and Event Monitoring
- **Live Loss Curve & Telemetry**: Dynamic charting of step-by-step training loss, learning rate, and epochs with zero fake metrics.
- **Event Feeds & Logs**: Chronological event logs tracking state transitions (`validating`, `loading_model`, `preparing_dataset`, `training`, `saving_artifacts`, `completed`, `failed`, `cancelled`).
- **Checkpoints Tracking**: Periodic checkpoint indexing with step number, timestamp, file reference, and verification status.

### 2. SQLite Run Repository & Persistence
- SQLite database (`artifacts/runs/runs.db`) stores run metadata, sanitized configurations, metrics history, checkpoints, and evaluation results.
- **Restart Survival**: Run history and evaluation records survive server restarts without data loss.

---

## Phase 8: Artifact Management, Export & Model Evaluation

### 1. Artifact Discovery & Integrity Validation
- **Discovery**: Real-time discovery of adapter weights (`adapter_model.bin` / `adapter_model.safetensors`), `adapter_config.json`, `training_config.json`, and training metrics.
- **Integrity Validation**: SHA-256 checksum generation and verification of required PEFT files; corrupt or missing files flag validation failure.

### 2. Standalone Package Export
- **Archive Generation**: Packages verified artifacts into portable ZIP archives (`/api/artifacts/{id}/export`).
- **Security Protections**: Strictly excludes secrets, Hugging Face tokens, private keys, parent directory symlinks, and absolute path traversals.
- **Direct Download**: HTTP streaming download endpoint (`GET /api/artifacts/{id}/download`).

### 3. Quantitative Model Evaluation & Baseline Comparison
- **Real Model Inference**: Computes cross-entropy evaluation loss, perplexity ($e^{\mathcal{L}_{\text{eval}}}$), and generation throughput (tokens/sec).
- **Baseline Comparison**: Compares fine-tuned adapter against the frozen base model on identical evaluation subsets, computing loss delta and percentage improvement.

---

## Phase 9: End-to-End Integration, Reliability & Production Readiness

### 1. End-to-End Integration Matrix

| Module / Workflow | Frontend Entry Point | Backend Service / Endpoint | Persistence & Storage | Integration Status |
|---|---|---|---|---|
| **Model Selection** | `ModelSelection.tsx` | `GET /api/models`, `POST /api/models/validate` | In-memory catalog + HuggingFace Hub | Verified |
| **Dataset Upload & Normalization** | `DatasetSelection.tsx` | `POST /api/datasets/upload`, `GET /api/datasets/{id}` | `artifacts/datasets/` JSON storage | Verified |
| **Configuration & Preflight** | `ParametersAndConfig.tsx`, `PreflightModal.tsx` | `POST /api/training/preflight`, `POST /api/training/preview` | Client session state & validated schemas | Verified |
| **Training Launch & Lifecycle** | `LaunchControlAndPreview.tsx` | `POST /api/training/runs` | JobManager + `artifacts/runs/runs.db` | Verified |
| **Telemetry & Run Monitoring** | `CurrentRunView.tsx` | `GET /api/training/runs/{id}`, `/events`, `/logs` | SQLite run table + event records | Verified |
| **Artifact Management & Export** | `ArtifactsView.tsx` | `GET /api/artifacts`, `POST /api/artifacts/{id}/export` | `artifacts/runs/{id}/` + exported ZIPs | Verified |
| **Model Evaluation** | `EvaluationView.tsx` | `POST /api/evaluations`, `GET /api/evaluations` | SQLite evaluation table | Verified |
| **Run & Evaluation History** | `TrainingHistoryView.tsx` | `GET /api/training/history` | SQLite paginated query | Verified |

### 2. Hardware Detection & GPU Troubleshooting
The studio enforces strict truth-in-hardware reporting:
- **Environment Status**: The system detects hardware using PyTorch and `nvidia-smi`.
- **Known Driver Issue (`nvidia-smi` exit code 9)**:
  - When the Linux NVIDIA kernel module is unloaded or mismatched (`NVIDIA-SMI couldn't communicate with the NVIDIA driver`), PyTorch reports `torch.cuda.is_available() == False`.
  - The framework does **not** assume the NVIDIA RTX 2050 is available, does not claim GPU acceleration, and operates honestly in `cpu_only` mode with `detection_status: cuda_unavailable`.
- **Distinction Between Capabilities**:
  - **GPU Detection**: Hardware detection query via PyTorch CUDA.
  - **GPU Training**: Training execution using `device="cuda"`.
  - **QLoRA Compatibility**: Requires genuine CUDA hardware and `bitsandbytes`. When CUDA is unavailable, QLoRA is blocked with an actionable explanation.

### 3. Reliability and Restart Recovery
- **Terminal State Immutability**: Runs and evaluations in terminal states (`completed`, `failed`, `cancelled`) cannot transition back to active states.
- **Dead Worker Reconciliation**: If the backend restarts while a run was in an active state (`training`, `validating`, `preparing_dataset`), startup reconciliation detects that the thread no longer exists and marks the run `failed` with an informative recovery message ("interrupted by server restart").
- **Duplicate Prevention**: Re-launching an existing running job ID or evaluation is blocked with HTTP 400.

### 4. Verification & Testing Commands

Run all Phase 9 automated end-to-end integration tests:
```bash
.venv/bin/pytest backend/tests/test_phase9_e2e_integration.py -v
```

Run full backend test suite (233 tests across all phases):
```bash
.venv/bin/pytest -v
```

Run full frontend test suite (121 tests across 12 suites):
```bash
cd frontend && npm test -- --run
```

Run frontend production build & TypeScript check:
```bash
cd frontend && npm run build
```

---

## Development Roadmap

- [x] **Phase 0**: Project Foundation, FastAPI Backend Services, System Hardware Diagnostics, Training Configuration Schemas, and Dark-Themed Training UI Skeleton.
- [x] **Phase 1**: Model Selection, Modular Catalog Registry, Fine-Tuning Method Compatibility Checks, Masked Hugging Face Credential Security, and UI Integration.
- [x] **Phase 2**: Dataset Upload (JSON, JSONL, CSV), Robust Validation, Multi-Schema Detection, Deterministic Normalization, Train/Eval Split Selection (0% Overlap Guarantee), and Training Studio UI Integration.
- [x] **Phase 3**: Training Parameters & Configuration Management System (Simple/Advanced modes, LoRA adapter settings, YAML presets, effective batch size calculation, and full schema validation).
- [x] **Phase 4**: Real LoRA Fine-Tuning Engine (Modular model loader, PEFT adapter injection, parameter freezing, dataset preparation, optimizer loop, streaming telemetry, cancellation, artifact validation, and Current Run view).
- [x] **Phase 5**: QLoRA and 4-Bit Quantization (Typed quantization schema, bitsandbytes 4-bit NF4/FP4 loading, k-bit model preparation via PEFT, double quantization, compute dtype selection, honest hardware verification, streaming telemetry, and verified adapter artifacts).
- [x] **Phase 6**: Training Run Preview, Resource Validation & Launch Control (Pre-launch validation service, real run preview, structured RunPlan, explicit confirmation, duplicate launch prevention, and launch lifecycle integration).
- [x] **Phase 7**: Live Training Monitoring, Event Streaming & SQLite Run History.
- [x] **Phase 8**: Artifact Discovery, Integrity Validation, Standalone ZIP Export & Model Evaluation.
- [x] **Phase 9**: End-to-End Integration, Reliability, Hardware Reconciliation, and Production Readiness.
- [ ] **Phase 10**: Advanced Quantization, Model Export (GGUF/vLLM), Multi-GPU, and Deployment.



