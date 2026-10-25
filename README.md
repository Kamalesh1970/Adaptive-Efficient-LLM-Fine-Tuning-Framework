# Adaptive LLM Fine-Tuning Studio

An adaptive, resource-aware Large Language Model (LLM) fine-tuning studio and training workflow engine.

Inspired by modern fine-tuning workflows (such as Unsloth), the Studio provides an ergonomic web dashboard paired with an adaptive ML backend designed to automatically select training strategies, quantization levels (e.g. QLoRA 4-bit NF4, LoRA), and memory configurations tailored to hardware constraints and dataset characteristics.

> **Phase 0 Status: COMPLETED**  
> Scope: Project Foundation, FastAPI Backend Services, System Hardware Diagnostics, Training Configuration Schemas, and Dark-Themed Training UI Skeleton.  
> *Notice: In Phase 0, UI scaffolding and configuration validation are active. Actual ML execution loops, dataset upload processing, and model training are planned for later phases.*

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
│   │   │   └── training.ts            # TypeScript interfaces for configs, hardware, and validation
│   │   ├── App.tsx                    # Root studio container
│   │   └── App.test.tsx               # Vitest component test suite
│   ├── package.json
│   ├── tailwind.config.js             # Curated dark slate/charcoal studio theme
│   └── vite.config.ts                 # Dev server with /api proxy to FastAPI
│
├── backend/                           # FastAPI Application & Microservices
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py              # GET /api/health, GET /api/system/status, GET/POST /api/training/config/*
│   │   ├── core/
│   │   │   ├── config.py              # Host, port, debug, CORS settings
│   │   │   └── system.py              # Hardware probe (Python, PyTorch, CUDA, VRAM, CPU fallback)
│   │   ├── schemas/
│   │   │   ├── system.py              # HealthResponse, SystemStatusResponse, GpuDeviceInfo
│   │   │   └── training.py            # TrainingConfigSchema, ValidationResult, TrainingMethod
│   │   ├── services/
│   │   │   └── config_service.py      # Baseline defaults and validation rules
│   │   └── main.py                    # FastAPI app factory with CORS middleware
│   └── tests/                         # Pytest test suite (15 unit and integration tests)
│       ├── test_app.py                # Startup and health check
│       ├── test_config.py             # Config defaults, validation, and invalid values
│       └── test_system.py             # Real probe and mocked CUDA / CPU / no-PyTorch environments
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

## Prerequisites

- **Python**: 3.11+ (Python 3.12 recommended)
- **Node.js**: v18+ (Node v18.19+ and npm 9+ recommended)
- **Optional**: CUDA-capable GPU (NVIDIA GPU with compute capability 7.0+ for future training phases; CPU-only mode is fully supported for development and Phase 0 UI testing)

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

Install the frontend packages using npm:

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

The interactive OpenAPI documentation will be available at:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Endpoint: [http://localhost:8000/api/health](http://localhost:8000/api/health)
- System Hardware Endpoint: [http://localhost:8000/api/system/status](http://localhost:8000/api/system/status)

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

Run the backend test suite:

```bash
# Run quiet test summary
pytest -q

# Or run verbose tests for backend
pytest backend/tests/ -v
```

### 2. Frontend Tests (Vitest)

Run the Vitest test suite:

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

## Current Development Status (Phase 0)

### Implemented in Phase 0:
- **FastAPI Backend Foundation**:
  - `GET /api/health`: Service health and version check.
  - `GET /api/system/status`: Real environment probe returning Python version, OS platform, PyTorch availability, CUDA availability, GPU device names, and VRAM in GB. Gracefully handles CPU-only and environments lacking PyTorch.
  - `GET /api/training/config/defaults`: Returns baseline configuration starting with unselected model and dataset.
  - `POST /api/training/config/validate`: Validates configuration parameters (project name, positive learning rate, context length limits, duration) and attaches readiness criteria.
- **Dark-Themed Training UI Skeleton**:
  - **Left Sidebar**: Application branding ("Adaptive LLM Fine-Tuning Studio"), highlighted navigation items ("New Training", "Models", "Datasets", "Training History", "Settings"), and live backend connectivity indicator.
  - **Main Header**: "Train" title, subtitle, draft badge, and navigation tabs ("Configure", "Current Run", "History").
  - **Base Model Card**: Model selector with unselected initial state, fine-tuning method dropdown (LoRA, QLoRA, Full Fine-Tuning with clear Phase 0 UI indicators), and Hugging Face token environment detector.
  - **Dataset Card**: Dataset selection, drag-and-drop upload zone (Phase 0 visual scaffolding), supported formats hint (`.json`, `.jsonl`, `.csv`, `.parquet`), subset selector, training/evaluation split fields, and collapsible advanced formatting section.
  - **Parameters Card**: Simple tab (project name, training duration with steps vs. epochs switcher, batch size, context length, learning rate) and Advanced tab (Phase 0 placeholder).
  - **Configuration Toolbar**: Functional "Load YAML" (file picker and parser), "Save YAML" (client-side export/download), and "Reset Defaults" (fetches baseline from backend).
  - **Run Summary Preview**: Persistent right panel displaying real-time model status, dataset status, training duration, batch size, context length, hardware summary, VRAM, and a strictly disabled "Start Fine-Tuning" action button until required selections and backend execution are ready.
- **Frontend/Backend Integration**:
  - Typed API layer in `frontend/src/services/api.ts`.
  - Offline fallback and error banner when the FastAPI backend is not running.

---

## Planned Future Phases

- **Phase 1**: Dataset Intelligence & Tokenization Pipeline (Schema detection, normalization, SHA-256 deduplication, sequence length analysis, and padding waste profiling).
- **Phase 2**: Real-time Hardware Profiling & VRAM Memory Estimation Engine.
- **Phase 3**: Adaptive Strategy Selector & Automatic Hyperparameter Resolution.
- **Phase 4**: Fine-Tuning Execution Engine (Hugging Face `transformers`, `peft` LoRA/QLoRA, `bitsandbytes` 4-bit NF4, streaming loss telemetry).
- **Phase 5**: Evaluation Framework, Benchmark Suites, and Training History.
- **Phase 6**: Model Exporter (LoRA merge, GGUF/vLLM export, and deployment).

