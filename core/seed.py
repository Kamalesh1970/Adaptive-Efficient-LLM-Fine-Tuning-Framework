"""
Reproducibility Seed Utility for Adaptive LLM Fine-Tuning Framework.
"""

import os
import random
from typing import Dict, Any, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from core.logger import get_logger

logger = get_logger("seed")


def set_seed(seed: int = 42) -> Dict[str, Any]:
    """
    Sets global seeds for Python random, NumPy, and PyTorch (CPU & CUDA if available).

    Args:
        seed: Integer seed value.

    Returns:
        Dict detailing which libraries were seeded.
    """
    status: Dict[str, Any] = {
        "seed": seed,
        "python_random": False,
        "numpy": False,
        "torch_cpu": False,
        "torch_cuda": False,
    }

    # 1. Python random module
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    status["python_random"] = True

    # 2. NumPy
    if HAS_NUMPY:
        np.random.seed(seed)
        status["numpy"] = True

    # 3. PyTorch & CUDA
    if HAS_TORCH:
        torch.manual_seed(seed)
        status["torch_cpu"] = True

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            status["torch_cuda"] = True

    logger.debug(f"Global seed set to {seed}. Seed status: {status}")
    return status
