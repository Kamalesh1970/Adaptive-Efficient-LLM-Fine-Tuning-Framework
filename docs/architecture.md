# Adaptive Efficient LLM Fine-Tuning Framework: Architecture Overview

## Research Objective

Can a resource-aware adaptive fine-tuning framework automatically select training strategies and configurations based on the model, dataset, and available hardware while reducing computational cost without significantly degrading task performance?

## System Architecture

```text
adaptive-llm-finetuning/
├── core/                  # Configuration, logging, exceptions, seed reproducibility, system status
│   ├── lora/              # [Planned] LoRA parameter wrappers
│   ├── qlora/             # [Planned] QLoRA quantized low-rank adaptation
│   ├── quantization/      # [Planned] Base model quantization wrappers
│   └── training/          # [Planned] Execution loops
├── intelligence/          # Adaptive Intelligence Layer
│   ├── dataset/           # Dataset & Tokenization Pipeline (Phase 1)
│   │   ├── loader.py      # Abstract loader for JSON, JSONL, CSV, and HF Datasets
│   │   ├── schema.py      # Schema type detection (Instruction, Conversational) & validation
│   │   ├── normalizer.py  # Unified NormalizedSample mapper
│   │   ├── cleaner.py     # SHA-256 duplicate detector & sample cleaner
│   │   ├── tokenizer.py   # TokenizerWrapper & MockTokenizer
│   │   ├── analyzer.py    # Sequence length & estimated padding efficiency analyzer
│   │   ├── profile.py     # DatasetProfile dataclass (JSON serializable)
│   │   └── pipeline.py    # End-to-end DatasetPipeline orchestrator
│   ├── hardware/          # [Planned] Hardware profiler
│   ├── memory/            # [Planned] Memory estimation engine
│   └── configuration/     # [Planned] Strategy selector
├── evaluation/            # [Planned] Metrics & benchmarking framework
├── models/                # [Planned] Model registry
├── datasets/              # Dataset storage
├── dashboard/             # [Planned] Monitoring dashboard
├── deployment/            # [Planned] Exporter modules
└── tests/                 # Unit & integration test suite
```

## Dataset & Tokenization Pipeline Flow (Phase 1)

```text
Raw Dataset (JSON/JSONL/CSV/HF)
           ↓
    DatasetLoader
           ↓
    Schema Detection & Validation
           ↓
    DatasetNormalizer (NormalizedSample)
           ↓
    DatasetCleaner (Deduplication via SHA-256)
           ↓
    TokenizerWrapper (TokenizedSample)
           ↓
    SequenceAnalyzer (Lengths, Bins, Padding Waste)
           ↓
    DatasetProfile (JSON Output)
```

## Phase 1 Design Principles

1. **Format Agnostic Input**: Loads JSON, JSONL, CSV, or Hugging Face dataset objects seamlessly.
2. **Deterministic Normalization**: Standardizes instruction (`instruction`, `input`, `output`) and conversational (`messages`) schemas into `NormalizedSample`.
3. **Traceable Cleaning**: Non-destructive cleaning with exact statistics on original count, invalid samples, duplicate count, and clean count.
4. **Offline Testability**: Includes `MockTokenizer` to enable full pipeline testing without downloading external model weights.
5. **Structured Profiling**: Produces a JSON-serializable `DatasetProfile` recording sample count, token count, percentiles, binned distribution, and padding waste ratio.
