"""PyTorch datasets over the processed 512x512 PNG tiles."""

from __future__ import annotations

import os
from pathlib import Path

from torch.utils.data import Dataset

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")


def load_tile(root: Path, tile_id: str, with_mask: bool = True):
    """Read one tile as ``(rgb_uint8_hwc, mask_uint8_hw or None)``.

    Args:
        root: Processed data folder containing ``images/`` and ``masks/``.
        tile_id: Tile ID without extension.
        with_mask: Whether to read the mask.

    Raises:
        FileNotFoundError: If a required PNG is missing or unreadable.
    """
    import cv2

    img = cv2.imread(str(Path(root) / "images" / f"{tile_id}.png"), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"image tile {tile_id} not found under {root}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if not with_mask:
        return img, None
    mask = cv2.imread(str(Path(root) / "masks" / f"{tile_id}.png"), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"mask tile {tile_id} not found under {root}")
    return img, mask


class TileDataset(Dataset):
    """Labeled or unlabeled LandCover.ai tiles.

    Each item is a dict with ``image`` (float tensor, 3x512x512), ``id`` and,
    when ``labeled`` is True, ``mask`` (int64 tensor, 512x512). For unlabeled
    data the mask is never read from disk, so no label information can leak
    into semi-supervised training.
    """

    def __init__(self, root: Path, ids: list[str], transform=None, labeled: bool = True):
        """Store the tile root, the ID list, a joint transform and labeled flag."""
        self.root = Path(root)
        self.ids = list(ids)
        self.transform = transform
        self.labeled = labeled

    def __len__(self) -> int:
        """Number of tiles."""
        return len(self.ids)

    def __getitem__(self, i: int) -> dict:
        """Load, transform and return tile ``i`` as a dict."""
        tid = self.ids[i]
        img, mask = load_tile(self.root, tid, with_mask=self.labeled)
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        item = {"image": img, "id": tid}
        if self.labeled:
            item["mask"] = mask
        return item
