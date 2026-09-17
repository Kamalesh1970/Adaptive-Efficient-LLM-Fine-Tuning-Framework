"""
Dataset Schema Definitions and Validation Logic.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from core.exceptions import ValidationError, DatasetError


class SchemaType(str, Enum):
    INSTRUCTION = "instruction"
    CONVERSATIONAL = "conversational"


class NormalizedSample(BaseModel):
    """Unified normalized sample representation across dataset formats."""
    instruction: Optional[str] = Field(default=None, description="Task instruction prompt")
    input: Optional[str] = Field(default=None, description="Optional input context")
    output: Optional[str] = Field(default=None, description="Expected model target response")
    messages: Optional[List[Dict[str, str]]] = Field(default=None, description="List of chat messages [{'role': ..., 'content': ...}]")
    schema_type: SchemaType = Field(description="Identified schema pattern")

    def get_text_content(self) -> str:
        """Returns unified text representation for length analysis or hashing."""
        if self.schema_type == SchemaType.INSTRUCTION:
            parts = []
            if self.instruction:
                parts.append(self.instruction.strip())
            if self.input:
                parts.append(self.input.strip())
            if self.output:
                parts.append(self.output.strip())
            return "\n".join(parts)
        elif self.schema_type == SchemaType.CONVERSATIONAL and self.messages:
            return "\n".join(f"{m.get('role', '')}: {m.get('content', '')}".strip() for m in self.messages)
        return ""


def detect_schema(sample: Dict[str, Any]) -> SchemaType:
    """
    Detects whether a sample dict follows instruction or conversational format.

    Args:
        sample: Dictionary representing a raw dataset row.

    Returns:
        SchemaType (INSTRUCTION or CONVERSATIONAL).

    Raises:
        DatasetError: If sample structure does not match any recognized schema.
    """
    if not isinstance(sample, dict):
        raise DatasetError(f"Expected dict sample, got {type(sample).__name__}")

    if "messages" in sample and isinstance(sample["messages"], list):
        return SchemaType.CONVERSATIONAL

    # Check for instruction format keys
    instruction_keys = {"instruction", "output", "prompt", "response", "input"}
    if any(k in sample for k in instruction_keys):
        return SchemaType.INSTRUCTION

    raise DatasetError(f"Unable to detect valid schema for sample keys: {list(sample.keys())}")


def validate_raw_sample(sample: Dict[str, Any]) -> List[str]:
    """
    Validates raw sample fields against schema specifications.

    Returns:
        List of error description strings (empty if valid).
    """
    errors: List[str] = []

    if not isinstance(sample, dict):
        return [f"Sample must be a dictionary, got {type(sample).__name__}"]

    try:
        schema_type = detect_schema(sample)
    except DatasetError as e:
        return [str(e)]

    if schema_type == SchemaType.INSTRUCTION:
        # Check instruction / output fields
        inst = sample.get("instruction") or sample.get("prompt")
        out = sample.get("output") or sample.get("response")

        if inst is None or not isinstance(inst, str) or not inst.strip():
            errors.append("Instruction format requires a non-empty string for 'instruction' or 'prompt'")

        if out is None or not isinstance(out, str) or not out.strip():
            errors.append("Instruction format requires a non-empty string for 'output' or 'response'")

    elif schema_type == SchemaType.CONVERSATIONAL:
        messages = sample.get("messages")
        if not isinstance(messages, list) or len(messages) == 0:
            errors.append("Conversational format requires a non-empty 'messages' list")
        else:
            valid_roles = {"user", "assistant", "system"}
            has_assistant_response = False
            for idx, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    errors.append(f"Message at index {idx} must be a dictionary")
                    continue
                role = msg.get("role")
                content = msg.get("content")

                if role not in valid_roles:
                    errors.append(f"Invalid message role '{role}' at index {idx}. Allowed: {valid_roles}")
                if not isinstance(content, str) or not content.strip():
                    errors.append(f"Empty content for message role '{role}' at index {idx}")
                if role == "assistant" and isinstance(content, str) and content.strip():
                    has_assistant_response = True

            if not has_assistant_response:
                errors.append("Conversational messages must include at least one valid 'assistant' response")

    return errors
