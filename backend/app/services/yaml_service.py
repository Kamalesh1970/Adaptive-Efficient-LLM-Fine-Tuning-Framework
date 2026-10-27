"""
YAML Configuration Management Service for Adaptive LLM Fine-Tuning Studio.
Provides safe serialization, parsing, and secret sanitization without arbitrary constructors.
"""

from typing import Dict, Any, Union, Set
import yaml

from backend.app.schemas.training import (
    TrainingConfigSchema,
    TrainingConfigValidationResult,
)

# Secrets that must NEVER be written to or accepted from exported configuration
SECRET_KEYS: Set[str] = {
    "hf_token",
    "token",
    "auth_token",
    "password",
    "secret",
    "api_key",
    "access_token",
    "private_key",
    "client_secret",
}


def sanitize_secrets_dict(data: Any) -> Any:
    """
    Recursively strips any secret fields from a configuration dictionary.
    """
    if isinstance(data, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in data.items():
            if str(k).lower() in SECRET_KEYS:
                continue
            cleaned[k] = sanitize_secrets_dict(v)
        return cleaned
    elif isinstance(data, list):
        return [sanitize_secrets_dict(item) for item in data]
    return data


def serialize_training_config_to_yaml(
    config: Union[TrainingConfigSchema, Dict[str, Any]]
) -> str:
    """
    Safely exports training configuration to a clean YAML string.
    Guarantees secrets are stripped and safe_dump is used.
    """
    if isinstance(config, TrainingConfigSchema):
        raw_dict = config.model_dump()
    elif isinstance(config, dict):
        raw_dict = dict(config)
    else:
        raise ValueError("Configuration must be a TrainingConfigSchema or dictionary")

    # Sanitize any accidental secrets
    sanitized = sanitize_secrets_dict(raw_dict)

    # Convert enum values to string if any remain
    def _normalize_enums(obj: Any) -> Any:
        if hasattr(obj, "value"):
            return obj.value
        if isinstance(obj, dict):
            return {k: _normalize_enums(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_normalize_enums(i) for i in obj]
        return obj

    sanitized = _normalize_enums(sanitized)

    header = (
        "# ==========================================================\n"
        "# Adaptive LLM Fine-Tuning Studio - Training Configuration\n"
        "# Safe YAML Export (Secrets and tokens are strictly excluded)\n"
        "# ==========================================================\n\n"
    )

    yaml_str = yaml.safe_dump(
        sanitized,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )

    return header + yaml_str


def parse_training_config_from_yaml(yaml_content: str) -> Dict[str, Any]:
    """
    Safely parses YAML content using yaml.safe_load.
    Rejects malformed syntax, executes no custom tags or constructors,
    and strips any contained secrets.
    """
    if not isinstance(yaml_content, str):
        raise ValueError("YAML content must be a text string")

    stripped = yaml_content.strip()
    if not stripped:
        raise ValueError("YAML configuration content cannot be empty or blank")

    try:
        parsed = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        raise ValueError(f"Malformed YAML syntax: {str(e)}") from e

    if not isinstance(parsed, dict):
        raise ValueError(
            "Invalid YAML configuration: root element must be a key-value mapping (dictionary)"
        )

    # Sanitize any included secrets
    sanitized = sanitize_secrets_dict(parsed)

    # Backward compatibility handling
    if "training_length" in sanitized and "max_steps" not in sanitized:
        if sanitized.get("training_length_type") == "steps" or "training_length_type" not in sanitized:
            sanitized["max_steps"] = sanitized["training_length"]
        elif sanitized.get("training_length_type") == "epochs":
            sanitized["num_epochs"] = float(sanitized["training_length"])

    if "batch_size" in sanitized and "per_device_train_batch_size" not in sanitized:
        sanitized["per_device_train_batch_size"] = sanitized["batch_size"]

    if "context_length" in sanitized and "max_seq_length" not in sanitized:
        sanitized["max_seq_length"] = sanitized["context_length"]

    return sanitized
