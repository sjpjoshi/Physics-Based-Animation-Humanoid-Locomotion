"""Reproducibility helpers."""
from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and (if installed) PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
