"""
Comprehensive Test Suite for Phase 1 - Dataset & Tokenization Pipeline.
"""

import json
import csv
import pytest
from pathlib import Path

from core.exceptions import DatasetError, TokenizationError, ProjectError
from intelligence.dataset.schema import (
    SchemaType,
    NormalizedSample,
    detect_schema,
    validate_raw_sample,
)
from intelligence.dataset.loader import DatasetLoader
from intelligence.dataset.normalizer import DatasetNormalizer
from intelligence.dataset.cleaner import DatasetCleaner, CleaningOptions, CleaningStats
from intelligence.dataset.tokenizer import TokenizerWrapper, MockTokenizer, TokenizedSample
from intelligence.dataset.analyzer import SequenceAnalyzer, SequenceLengthStats, PaddingAnalysis
from intelligence.dataset.profile import DatasetProfile
from intelligence.dataset.pipeline import DatasetPipeline, PipelineResult


# ---------------------------------------------------------------------------
# 1. Dataset Loader Tests
# ---------------------------------------------------------------------------

def test_loader_json(temp_dir):
    json_path = temp_dir / "test_data.json"
    data = [{"instruction": "What is Python?", "output": "Python is a language."}]
    json_path.write_text(json.dumps(data), encoding="utf-8")

    loaded = DatasetLoader.load(json_path)
    assert len(loaded) == 1
    assert loaded[0]["instruction"] == "What is Python?"


def test_loader_jsonl(temp_dir):
    jsonl_path = temp_dir / "test_data.jsonl"
    lines = [
        json.dumps({"instruction": "Q1", "output": "A1"}),
        json.dumps({"instruction": "Q2", "output": "A2"}),
    ]
    jsonl_path.write_text("\n".join(lines), encoding="utf-8")

    loaded = DatasetLoader.load(jsonl_path)
    assert len(loaded) == 2
    assert loaded[1]["instruction"] == "Q2"


def test_loader_csv(temp_dir):
    csv_path = temp_dir / "test_data.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["instruction", "output"])
        writer.writerow(["Translate hello", "Bonjour"])

    loaded = DatasetLoader.load(csv_path)
    assert len(loaded) == 1
    assert loaded[0]["instruction"] == "Translate hello"


def test_loader_invalid_filepath():
    with pytest.raises(DatasetError) as exc_info:
        DatasetLoader.load("non_existent_file_path_xyz.json")
    assert "Dataset file not found" in str(exc_info.value)


def test_loader_unsupported_format(temp_dir):
    txt_path = temp_dir / "test_data.txt"
    txt_path.write_text("Plain text content", encoding="utf-8")

    with pytest.raises(DatasetError) as exc_info:
        DatasetLoader.load(txt_path)
    assert "Unsupported dataset file format" in str(exc_info.value)


def test_loader_malformed_json(temp_dir):
    bad_json = temp_dir / "bad.json"
    bad_json.write_text("{this is invalid json syntax", encoding="utf-8")

    with pytest.raises(DatasetError) as exc_info:
        DatasetLoader.load(bad_json)
    assert "Malformed JSON" in str(exc_info.value)


def test_loader_empty_file(temp_dir):
    empty_file = temp_dir / "empty.json"
    empty_file.write_text("", encoding="utf-8")

    with pytest.raises(DatasetError) as exc_info:
        DatasetLoader.load(empty_file)
    assert "completely empty" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 2. Schema Validation Tests
# ---------------------------------------------------------------------------

def test_schema_detection_instruction():
    sample = {"instruction": "Write code", "output": "print('hello')"}
    assert detect_schema(sample) == SchemaType.INSTRUCTION


def test_schema_detection_conversational():
    sample = {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}
    assert detect_schema(sample) == SchemaType.CONVERSATIONAL


def test_schema_validation_valid_instruction():
    sample = {"instruction": "Explain AI", "output": "AI stands for..."}
    errors = validate_raw_sample(sample)
    assert len(errors) == 0


def test_schema_validation_invalid_instruction():
    sample = {"instruction": "", "output": "Some response"}
    errors = validate_raw_sample(sample)
    assert len(errors) > 0
    assert "non-empty string" in errors[0]


def test_schema_validation_conversational_roles():
    bad_sample = {"messages": [{"role": "hacker", "content": "test"}]}
    errors = validate_raw_sample(bad_sample)
    assert len(errors) > 0
    assert "Invalid message role" in errors[0]


# ---------------------------------------------------------------------------
# 3. Normalization Tests
# ---------------------------------------------------------------------------

def test_normalization_instruction():
    raw = {"prompt": "What is 2+2?", "response": "4"}
    norm = DatasetNormalizer.normalize_sample(raw)
    assert norm.schema_type == SchemaType.INSTRUCTION
    assert norm.instruction == "What is 2+2?"
    assert norm.output == "4"


def test_normalization_conversational():
    raw = {"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello!"}]}
    norm = DatasetNormalizer.normalize_sample(raw)
    assert norm.schema_type == SchemaType.CONVERSATIONAL
    assert norm.messages[0]["role"] == "user"
    assert norm.messages[1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# 4. Cleaning & Duplicate Detection Tests
# ---------------------------------------------------------------------------

def test_cleaner_duplicate_detection():
    raw_samples = [
        {"instruction": "Capital of France?", "output": "Paris"},
        {"instruction": "Capital of France?", "output": "Paris"},  # Duplicate
        {"instruction": "Capital of Germany?", "output": "Berlin"},
    ]
    norm_samples = DatasetNormalizer.normalize_dataset(raw_samples)

    cleaner = DatasetCleaner(CleaningOptions(enable_cleaning=True, remove_duplicates=True))
    cleaned, stats = cleaner.process(raw_samples, norm_samples)

    assert stats.original_count == 3
    assert stats.duplicate_count == 1
    assert stats.clean_count == 2
    assert len(cleaned) == 2


def test_cleaner_invalid_sample_removal():
    raw_samples = [
        {"instruction": "Valid instruction", "output": "Valid output"},
        {"instruction": "", "output": ""},  # Invalid empty
    ]
    norm_samples = DatasetNormalizer.normalize_dataset(raw_samples)

    cleaner = DatasetCleaner(CleaningOptions(enable_cleaning=True, remove_invalid=True))
    cleaned, stats = cleaner.process(raw_samples, norm_samples)

    assert stats.invalid_count == 1
    assert stats.clean_count == 1


# ---------------------------------------------------------------------------
# 5. Tokenization Tests
# ---------------------------------------------------------------------------

def test_mock_tokenizer_basic():
    tokenizer = MockTokenizer(max_length=10)
    res = tokenizer.encode("Hello world python test", max_length=10, truncation=True)
    assert res["token_count"] == 6  # BOS + 4 words + EOS
    assert len(res["input_ids"]) == 6
    assert len(res["attention_mask"]) == 6


def test_tokenizer_wrapper_encode_sample():
    tokenizer = MockTokenizer(max_length=20)
    wrapper = TokenizerWrapper(tokenizer, max_length=20)
    sample = NormalizedSample(schema_type=SchemaType.INSTRUCTION, instruction="Test instruction", output="Test response")

    tokenized = wrapper.encode_sample(sample)
    assert isinstance(tokenized, TokenizedSample)
    assert tokenized.token_count > 0
    assert len(tokenized.input_ids) == tokenized.token_count


def test_tokenizer_empty_text_error():
    wrapper = TokenizerWrapper(MockTokenizer(), max_length=10)
    empty_sample = NormalizedSample(schema_type=SchemaType.INSTRUCTION)

    with pytest.raises(TokenizationError) as exc_info:
        wrapper.encode_sample(empty_sample)
    assert "empty text sample" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 6. Sequence Analyzer & Padding Tests
# ---------------------------------------------------------------------------

def test_sequence_length_statistics():
    tokenized = [
        TokenizedSample(input_ids=[1, 2, 3], attention_mask=[1, 1, 1], token_count=3),
        TokenizedSample(input_ids=[1, 2, 3, 4, 5, 6, 7], attention_mask=[1]*7, token_count=7),
        TokenizedSample(input_ids=[1, 2, 3, 4, 5], attention_mask=[1]*5, token_count=5),
    ]

    stats = SequenceAnalyzer.analyze_lengths(tokenized, custom_bins=[4, 8])
    assert stats.total_samples == 3
    assert stats.total_tokens == 15
    assert stats.min_length == 3
    assert stats.max_length == 7
    assert stats.mean_length == 5.0
    assert stats.median_length == 5.0


def test_padding_efficiency_analysis():
    tokenized = [
        TokenizedSample(input_ids=[1]*10, attention_mask=[1]*10, token_count=10),
        TokenizedSample(input_ids=[1]*20, attention_mask=[1]*20, token_count=20),
    ]
    padding = SequenceAnalyzer.analyze_padding(tokenized, sequence_capacity=50)
    assert padding.total_allocated_tokens == 100
    assert padding.actual_tokens == 30
    assert padding.estimated_padding_tokens == 70
    assert padding.estimated_padding_ratio == 0.7


# ---------------------------------------------------------------------------
# 7. DatasetProfile Serialization Tests
# ---------------------------------------------------------------------------

def test_dataset_profile_json_serialization():
    profile = DatasetProfile(
        num_samples=100,
        total_tokens=2500,
        min_length=10,
        max_length=100,
        mean_length=25.0,
        median_length=20.0,
        duplicate_count=2,
        invalid_count=1,
        estimated_padding_ratio=0.15,
        schema_type="instruction",
        length_distribution={"0-128": 100},
    )

    json_str = profile.to_json()
    assert '"num_samples": 100' in json_str

    reloaded = DatasetProfile.from_json(json_str)
    assert reloaded.num_samples == 100
    assert reloaded.estimated_padding_ratio == 0.15
    assert reloaded.schema_type == "instruction"


# ---------------------------------------------------------------------------
# 8. End-to-End Pipeline Integration Test
# ---------------------------------------------------------------------------

def test_end_to_end_dataset_pipeline(temp_dir):
    jsonl_path = temp_dir / "dataset.jsonl"
    samples = [
        {"instruction": "Explain quantum computing", "output": "Quantum computing uses qubits."},
        {"instruction": "Explain quantum computing", "output": "Quantum computing uses qubits."},  # Duplicate
        {"instruction": "Write a python function for fibonacci", "output": "def fib(n): return n if n <= 1 else fib(n-1)+fib(n-2)"},
    ]
    jsonl_path.write_text("\n".join(json.dumps(s) for s in samples), encoding="utf-8")

    pipeline = DatasetPipeline(
        max_sequence_length=128,
        cleaning_options=CleaningOptions(enable_cleaning=True, remove_duplicates=True),
    )

    result = pipeline.process(jsonl_path)

    assert isinstance(result, PipelineResult)
    assert result.cleaning_stats.original_count == 3
    assert result.cleaning_stats.duplicate_count == 1
    assert result.profile.num_samples == 2
    assert result.profile.schema_type == "instruction"
    assert len(result.normalized_dataset) == 2
    assert len(result.tokenized_dataset) == 2
