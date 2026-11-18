"""Reproducible DataLoader construction."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and ``random`` in a worker from PyTorch's per-worker seed.

    PyTorch derives each worker's seed from the loader's generator, so with a
    seeded generator the augmentation stream is reproducible run to run.
    """
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 4,
    drop_last: bool = False,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader whose shuffling and augmentation are seeded.

    Args:
        dataset: Any map-style dataset.
        batch_size: Samples per batch.
        shuffle: Shuffle each epoch (training) or keep order (evaluation).
        seed: Seed for the loader's ``torch.Generator``; controls the shuffle
            order and every worker's NumPy seed.
        num_workers: Worker processes (0 loads in the main process).
        drop_last: Drop the final incomplete batch (useful for BatchNorm).
        pin_memory: Use pinned host memory for faster GPU transfer.

    Returns:
        Configured ``torch.utils.data.DataLoader``.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    if num_workers == 0:
        np.random.seed(seed)
        random.seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=drop_last,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
