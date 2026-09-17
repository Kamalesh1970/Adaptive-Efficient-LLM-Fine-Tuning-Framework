"""
Reusable Structured Logging Utility.
"""

import logging
import sys
from typing import Optional


def get_logger(
    name: str = "adaptive_llm",
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Retrieves or initializes a formatted logger instance.

    Args:
        name: Module name for the logger.
        level: Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        log_file: Optional path to write log messages.

    Returns:
        logging.Logger configured instance.
    """
    logger = logging.getLogger(name)

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    if logger.handlers:
        for handler in logger.handlers:
            handler.setLevel(numeric_level)
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


# Default logger instance
logger = get_logger()
