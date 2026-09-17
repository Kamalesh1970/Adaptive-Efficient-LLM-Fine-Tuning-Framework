"""
Dataset Normalization Layer.
"""

from typing import Dict, Any, Optional, List
from intelligence.dataset.schema import NormalizedSample, SchemaType, detect_schema
from core.exceptions import DatasetError


class DatasetNormalizer:
    """Normalizes arbitrary raw sample dictionaries into deterministic NormalizedSample objects."""

    @classmethod
    def normalize_sample(cls, sample: Dict[str, Any]) -> NormalizedSample:
        """
        Normalizes a single raw sample dictionary.

        Args:
            sample: Raw data record dict.

        Returns:
            NormalizedSample instance.
        """
        if not isinstance(sample, dict):
            raise DatasetError(f"Cannot normalize sample of type {type(sample).__name__}")

        schema_type = detect_schema(sample)

        if schema_type == SchemaType.CONVERSATIONAL:
            raw_messages = sample.get("messages") or sample.get("conversations") or []
            messages: List[Dict[str, str]] = []
            for msg in raw_messages:
                if isinstance(msg, dict):
                    role = str(msg.get("role") or msg.get("from") or "user").strip()
                    # Alias mapping for legacy conversation tags (e.g. human -> user, gpt -> assistant)
                    if role.lower() in ("human", "user"):
                        role = "user"
                    elif role.lower() in ("gpt", "assistant", "bot"):
                        role = "assistant"
                    elif role.lower() in ("system",):
                        role = "system"

                    content = str(msg.get("content") or msg.get("value") or "").strip()
                    messages.append({"role": role, "content": content})

            return NormalizedSample(
                schema_type=SchemaType.CONVERSATIONAL,
                messages=messages,
            )

        else:  # INSTRUCTION
            instruction = sample.get("instruction") or sample.get("prompt") or sample.get("user_prompt") or sample.get("question")
            inp = sample.get("input") or sample.get("context")
            output = sample.get("output") or sample.get("response") or sample.get("target") or sample.get("answer")

            return NormalizedSample(
                schema_type=SchemaType.INSTRUCTION,
                instruction=str(instruction).strip() if instruction is not None else None,
                input=str(inp).strip() if inp is not None and str(inp).strip() else None,
                output=str(output).strip() if output is not None else None,
            )

    @classmethod
    def normalize_dataset(cls, samples: List[Dict[str, Any]]) -> List[NormalizedSample]:
        """Normalizes an entire dataset of raw dict samples."""
        return [cls.normalize_sample(s) for s in samples]
