"""Joint image/mask transforms operating on NumPy arrays.

Geometric transforms are applied identically to the image and the mask so
that labels stay aligned. Randomness comes from ``numpy.random``, which the
DataLoader seeds per worker (see ``landcover.data.loaders``).
"""

from __future__ import annotations

import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def apply_d4(image: np.ndarray, mask: np.ndarray | None, rot: int, flip: bool):
    """Apply one element of the dihedral group D4 (rotations x horizontal flip).

    Args:
        image: ``(H, W, C)`` array.
        mask: ``(H, W)`` array or ``None``.
        rot: Number of 90-degree counter-clockwise rotations, 0..3.
        flip: Whether to flip horizontally after rotating.

    Returns:
        ``(image, mask)`` transformed (contiguous copies).
    """
    image = np.rot90(image, rot, axes=(0, 1))
    if mask is not None:
        mask = np.rot90(mask, rot, axes=(0, 1))
    if flip:
        image = image[:, ::-1]
        if mask is not None:
            mask = mask[:, ::-1]
    return np.ascontiguousarray(image), (np.ascontiguousarray(mask) if mask is not None else None)


class RandomD4:
    """Randomly rotate by a multiple of 90 degrees and optionally flip.

    Aerial orthophotos have no canonical orientation, so all eight D4 poses
    are equally plausible views of the same scene.
    """

    def __call__(self, image, mask=None):
        """Return the image/mask pair under a uniformly random D4 element."""
        return apply_d4(image, mask, int(np.random.randint(4)), bool(np.random.randint(2)))


class Normalize:
    """Convert uint8 HWC RGB to a normalised float CHW tensor; mask to int64.

    Uses ImageNet statistics because the encoder is ImageNet-pretrained.
    """

    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        """Store per-channel mean and std (in [0, 1] units)."""
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, image, mask=None):
        """Return ``(image_tensor, mask_tensor_or_None)``."""
        x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255)
        x = (x - self.mean) / self.std
        y = torch.from_numpy(np.ascontiguousarray(mask)).long() if mask is not None else None
        return x, y


class Compose:
    """Chain joint transforms that each take and return ``(image, mask)``."""

    def __init__(self, transforms):
        """Store the transforms in application order."""
        self.transforms = list(transforms)

    def __call__(self, image, mask=None):
        """Apply each transform in turn."""
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


def build_transforms(train: bool) -> Compose:
    """Return the weak training pipeline (D4 + normalise) or the eval pipeline."""
    return Compose(([RandomD4()] if train else []) + [Normalize()])
