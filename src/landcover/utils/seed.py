"""Reproducibility helpers."""

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and (if installed) PyTorch random number generators.

    Args:
        seed: Integer seed applied to every RNG.
        deterministic: If True, request deterministic cuDNN/PyTorch kernels.
            This is slower and some ops raise errors when no deterministic
            implementation exists, so it is off by default and should be used
            only for debugging reproducibility issues.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
