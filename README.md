# Adaptive LLM Fine-Tuning Studio

An adaptive, resource-aware Large Language Model (LLM) fine-tuning studio and training workflow engine.

Inspired by modern fine-tuning workflows (such as Unsloth), the Studio provides an ergonomic web dashboard paired with an adaptive ML backend designed to automatically select training strategies, quantization levels (e.g. QLoRA 4-bit NF4, LoRA), and memory configurations tailored to hardware constraints and dataset characteristics.

> **Phase 2 Status: COMPLETED**  
> Scope: Dataset Upload (JSON, JSONL, CSV), Robust Parsing, Multi-Format Schema Detection, Deterministic Normalization, Train/Eval Split Selection (0% Data Leakage Guarantee), and Training Studio UI Integration.  
> *Notice: In Phase 2, dataset ingestion, validation, and split calculation are fully operational. Model training execution, LoRA/QLoRA loops, and hardware memory profiling are planned for subsequent phases.*

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

## Development Roadmap

- [x] **Phase 0**: Project Foundation, FastAPI Backend Services, System Hardware Diagnostics, Training Configuration Schemas, and Dark-Themed Training UI Skeleton.
- [x] **Phase 1**: Model Selection, Modular Catalog Registry, Fine-Tuning Method Compatibility Checks, Masked Hugging Face Credential Security, and UI Integration.
- [x] **Phase 2**: Dataset Upload (JSON, JSONL, CSV), Robust Validation, Multi-Schema Detection, Deterministic Normalization, Train/Eval Split Selection (0% Overlap Guarantee), and Training Studio UI Integration.
- [ ] **Phase 3**: Real-time Hardware Profiling & VRAM Memory Estimation Engine / Strategy Selector.
- [ ] **Phase 4**: Fine-Tuning Execution Engine (Hugging Face `transformers`, `peft` LoRA/QLoRA, `bitsandbytes` 4-bit NF4, streaming loss telemetry).
- [ ] **Phase 5**: Evaluation Framework, Benchmark Suites, and Training History.
- [ ] **Phase 6**: Model Exporter (LoRA merge, GGUF/vLLM export, and deployment).
