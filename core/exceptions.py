"""
Central Custom Exception Hierarchy for Adaptive LLM Fine-Tuning Framework.
"""

class ProjectError(Exception):
    """Base exception for all project-related errors."""
    pass


class ConfigurationError(ProjectError):
    """Raised when configuration loading or parsing fails."""
    pass


class ValidationError(ProjectError):
    """Raised when configuration parameter validation fails."""
    pass


class EnvironmentError(ProjectError):
    """Raised when environment detection or runtime dependencies fail."""
    pass


class DatasetError(ProjectError):
    """Raised when dataset loading, parsing, schema validation, or processing fails."""
    pass


class TokenizationError(ProjectError):
    """Raised when tokenization fails or tokenizer parameters are invalid."""
    pass
