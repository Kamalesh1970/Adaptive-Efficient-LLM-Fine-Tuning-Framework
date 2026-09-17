"""
Configuration Manager and Schemas for Adaptive LLM Fine-Tuning Framework.
"""

import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import yaml
from pydantic import BaseModel, Field, ValidationError

from core.exceptions import ConfigurationError
from core.logger import setup_logger

logger = setup_logger("config_manager")


class ModelConfig(BaseModel):
    name_or_path: str = Field(default="meta-llama/Llama-2-7b-hf", description="Hugging Face model identifier or local path")
    torch_dtype: str = Field(default="float16", description="PyTorch data type (float16, bfloat16, float32)")
    trust_remote_code: bool = Field(default=False, description="Allow custom code from Hugging Face hub")
    use_cache: bool = Field(default=False, description="Use KV cache during training")


class QuantizationConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable model quantization")
    load_in_4bit: bool = Field(default=True, description="Enable 4-bit quantization")
    load_in_8bit: bool = Field(default=False, description="Enable 8-bit quantization")
    bnb_4bit_quant_type: str = Field(default="nf4", description="4-bit quantization type (nf4 or fp4)")
    bnb_4bit_compute_dtype: str = Field(default="float16", description="Compute dtype for 4-bit base model")
    bnb_4bit_use_double_quant: bool = Field(default=True, description="Use nested quantization")


class LoraConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable LoRA fine-tuning")
    r: int = Field(default=16, ge=1, description="LoRA rank dimension")
    lora_alpha: int = Field(default=32, ge=1, description="LoRA scaling factor alpha")
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0, description="LoRA dropout rate")
    bias: str = Field(default="none", description="Bias type for LoRA")
    task_type: str = Field(default="CAUSAL_LM", description="Task type")
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"],
        description="List of target module names for LoRA adapters"
    )


class TrainingConfig(BaseModel):
    output_dir: str = Field(default="./outputs", description="Directory where model checkpoints will be saved")
    num_train_epochs: float = Field(default=3.0, gt=0.0, description="Total number of training epochs")
    per_device_train_batch_size: int = Field(default=2, ge=1, description="Batch size per GPU/CPU for training")
    gradient_accumulation_steps: int = Field(default=4, ge=1, description="Gradient accumulation steps")
    learning_rate: float = Field(default=0.0002, gt=0.0, description="Initial learning rate")
    weight_decay: float = Field(default=0.001, ge=0.0, description="Weight decay")
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=1.0, description="Warmup ratio")
    logging_steps: int = Field(default=10, ge=1, description="Logging frequency in steps")
    save_strategy: str = Field(default="steps", description="Checkpoint save strategy")
    save_steps: int = Field(default=100, ge=1, description="Save frequency in steps")
    evaluation_strategy: str = Field(default="no", description="Evaluation strategy")
    fp16: bool = Field(default=True, description="Enable fp16 mixed precision")
    bf16: bool = Field(default=False, description="Enable bf16 mixed precision")
    max_grad_norm: float = Field(default=0.3, ge=0.0, description="Maximum gradient norm")
    seed: int = Field(default=42, description="Random seed")


class HardwareLimits(BaseModel):
    max_memory_gb: float = Field(default=16.0, gt=0.0, description="Maximum allowed GPU memory in GB")
    fallback_to_cpu: bool = Field(default=False, description="Allow execution fallback to CPU if GPU unavailable")
    allow_tf32: bool = Field(default=True, description="Allow TF32 format on Ampere GPUs")


class AdaptiveSettings(BaseModel):
    auto_configure: bool = Field(default=True, description="Enable automatic adaptive hyperparameter resolution")
    memory_headroom_gb: float = Field(default=2.0, ge=0.0, description="Safety memory headroom in GB")
    optimization_target: str = Field(default="balanced", description="Optimization goal: speed, memory, balanced")


class FrameworkConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    hardware: HardwareLimits = Field(default_factory=HardwareLimits)
    adaptive: AdaptiveSettings = Field(default_factory=AdaptiveSettings)


class ConfigManager:
    """Manager for loading, validating, and overriding framework configuration."""

    def __init__(self, config: Optional[FrameworkConfig] = None):
        self._config = config or FrameworkConfig()

    @property
    def config(self) -> FrameworkConfig:
        return self._config

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "ConfigManager":
        """Loads configuration from a YAML file."""
        path = Path(yaml_path)
        if not path.is_file():
            raise ConfigurationError(f"Configuration file not found at path: {yaml_path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_dict = yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigurationError(f"Failed to parse YAML configuration file: {e}") from e

        return cls.from_dict(raw_dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigManager":
        """Constructs and validates FrameworkConfig from a raw dictionary."""
        try:
            config = FrameworkConfig(**data)
        except ValidationError as e:
            raise ConfigurationError(f"Configuration validation failed: {e}") from e
        
        manager = cls(config)
        manager.apply_env_overrides()
        return manager

    def apply_env_overrides(self) -> None:
        """Applies environment variable overrides to configuration."""
        env_mappings = {
            "ADAPTIVE_MODEL_NAME": ("model", "name_or_path"),
            "ADAPTIVE_OUTPUT_DIR": ("training", "output_dir"),
            "ADAPTIVE_LEARNING_RATE": ("training", "learning_rate", float),
            "ADAPTIVE_BATCH_SIZE": ("training", "per_device_train_batch_size", int),
            "ADAPTIVE_LORA_R": ("lora", "r", int),
            "ADAPTIVE_AUTO_CONFIGURE": ("adaptive", "auto_configure", lambda v: v.lower() in ("true", "1", "yes")),
        }

        for env_var, mapping in env_mappings.items():
            val = os.environ.get(env_var)
            if val is not None:
                section = mapping[0]
                field = mapping[1]
                converter = mapping[2] if len(mapping) > 2 else str
                try:
                    converted_val = converter(val)
                    sub_config = getattr(self._config, section)
                    setattr(sub_config, field, converted_val)
                    logger.info(f"Overrode config [{section}][{field}] from environment variable {env_var}={converted_val}")
                except Exception as e:
                    raise ConfigurationError(
                        f"Failed to parse environment variable {env_var} value '{val}': {e}"
                    ) from e

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Saves current configuration to YAML file."""
        path = Path(yaml_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self._config.model_dump(), f, default_flow_style=False)
