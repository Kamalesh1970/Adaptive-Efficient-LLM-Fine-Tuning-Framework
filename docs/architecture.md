# Adaptive Efficient LLM Fine-Tuning Framework: Architecture Overview

## Research Objective

Can a resource-aware adaptive fine-tuning framework automatically select training strategies and configurations based on the model, dataset, and available hardware while reducing computational cost without significantly degrading task performance?

## System Architecture

```text
adaptive-llm-finetuning/
├── backend/                   # FastAPI Backend Application
│   ├── app/
│   │   ├── api/routes.py      # REST API endpoints (/health, /system/status, /models/*, /training/*)
│   │   ├── schemas/           # Pydantic data models (models.py, training.py, system.py)
│   │   └── services/          # Business logic & catalogs (model_catalog.py, config_service.py)
│   └── tests/                 # Backend test suite (test_models.py, test_config.py, test_system.py)
├── frontend/                  # Vite + React 18 + TypeScript + Tailwind Studio UI
│   └── src/
│       ├── features/training/ # ModelCard, DatasetCard, ParametersCard, RunPreviewPanel
│       └── services/api.ts    # Typed fetch client
├── core/                      # Configuration, logging, exceptions, seed reproducibility, system status
│   ├── lora/                  # [Planned] LoRA parameter wrappers
│   ├── qlora/                 # [Planned] QLoRA quantized low-rank adaptation
│   ├── quantization/          # [Planned] Base model quantization wrappers
│   └── training/              # [Planned] Execution loops
├── intelligence/              # Adaptive Intelligence Layer
│   ├── dataset/               # Dataset & Tokenization Pipeline
│   ├── hardware/              # [Planned] Hardware profiler (Phase 2)
│   ├── memory/                # [Planned] Memory estimation engine (Phase 2)
│   └── configuration/         # [Planned] Strategy selector (Phase 3)
├── evaluation/                # [Planned] Metrics & benchmarking framework (Phase 5)
├── models/                    # Model registry and helper loaders
├── datasets/                  # Dataset storage
└── tests/                     # Foundation & pipeline test suite
```

## Phase 1: Model Selection & Compatibility Layer

### Flow Overview

```text
User Selects Model & Fine-Tuning Method
                   ↓
   Frontend React Client (ModelCard)
                   ↓
POST /api/models/validate (Payload: model_id, method, optional token)
                   ↓
      ModelCatalogService Lookup
                   ↓
     ├── 1. Identifier Check (known, non-empty)
     ├── 2. Method Validation (lora, qlora, full_finetuning)
     ├── 3. Architecture Compatibility Rule Evaluation
     ├── 4. Gated Access & In-Memory Token Evaluation
     └── 5. Context Length Verification
                   ↓
     ModelValidationResponse (Status, Errors, Warnings, Details)
                   ↓
UI Feedback & Persistent Run Preview Update
```

### Core Design Principles

1. **Separation of Metadata and Execution**: Model definitions contain validated architectural specifications without executing weight downloads or CUDA allocations in this phase.
2. **Explicit Verification Standards**: Metadata that has not been verified against empirical sources is explicitly labeled as `unverified` / `None` rather than fabricated.
3. **No Secret Leakage**: Access tokens are kept masked in UI memory, never saved to browser storage, and never echoed back in API responses, logs, or error stack traces.
4. **Offline Testability**: All model catalog lookups and validation endpoints run entirely offline with mocked data and unit tests.

---

## Phase 2: Dataset Upload, Validation & Split Selection Layer

### Flow Overview

```text
User Drag-and-Drops or Browses Dataset File (.json, .jsonl, .csv)
                   ↓
      Frontend Pre-Validation (Extension, Size < 50MB)
                   ↓
POST /api/datasets/upload (multipart/form-data)
                   ↓
           DatasetService
                   ↓
   ├── 1. Filename Sanitization & Path Traversal Prevention
   ├── 2. Upload Size Limit Enforcement (< 50MB)
   ├── 3. Unique Server-Side ID Generation (ds_<uuid>)
   ├── 4. Extension & Format Extraction (JSON, JSONL, CSV)
   ├── 5. Robust Format Parsing:
   │       • JSON: Array or wrapped object {"data": [...]}
   │       • JSONL: Line-by-line parsing with line-number error tracking
   │       • CSV: CSV DictReader with header/row validation
   ├── 6. Schema Detection:
   │       • instruction: instruction + optional input + output
   │       • text: text / content / prompt
   │       • conversational: messages / conversations with role + content
   ├── 7. Deterministic Normalization into internal sample representation
   ├── 8. Duplicate Detection (SHA-256 fingerprinting)
   └── 9. Validation Report Generation (errors, warnings, sample counts)
                   ↓
DatasetValidationReport JSON Response (no internal filesystem paths disclosed)
                   ↓
Frontend TrainingPage Updates Dataset State & Metrics Grid
                   ↓
Deterministic Split Selection (GET /api/datasets/{id}/splits)
   ├── Disjoint train/eval sample sets (0% overlap guarantee)
   └── Warns if sample count is insufficient for meaningful evaluation
                   ↓
RunPreviewPanel Reflects Verified Dataset, Schema & Disjoint Split Statistics
```

### Security & Integrity Standards

1. **Path Traversal Protection**: User-supplied filenames are sanitized (`os.path.basename`) and stored strictly using server-generated identifiers (`ds_<uuid>`) under the dedicated data directory (`datasets/uploads/`).
2. **Internal Path Masking**: Absolute server filesystem paths are never returned in client API responses or serialized error models.
3. **Deterministic Disjoint Splitting**: Pseudo-random split generation with configurable seeds guarantees training and evaluation sets are non-overlapping with 0% data leakage.
4. **No Arbitrary Code Execution**: Datasets are parsed purely as structured data records without eval or deserialization of executable code.

