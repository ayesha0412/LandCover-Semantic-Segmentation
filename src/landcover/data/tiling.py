"""Cut LandCover.ai orthophotos into 512x512 tiles using the official indexing.

The tile counter ``k`` reproduces the official ``split.py``: it advances over
every position of a ceil-sized grid, but only tiles fully inside the scene are
written. Tile IDs ``<scene>_<k>`` therefore match ``train/val/test.txt``
exactly. Unlike ``split.py``, images are stored as lossless PNG instead of
re-encoded JPEG, so pixels equal the decoded source orthophoto.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterator

import numpy as np

from landcover.constants import NUM_CLASSES

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

MANIFEST_FIELDS = ["tile_id", "scene", "k", "y", "x", "pixel_size_m", "nodata_px"] + [
    f"px_c{c}" for c in range(NUM_CLASSES)
]


def iter_tiles(image: np.ndarray, mask: np.ndarray, tile: int) -> Iterator[tuple[int, int, int, np.ndarray, np.ndarray]]:
    """Yield every full tile of a scene in official ``split.py`` order.

    Args:
        image: ``(H, W, C)`` scene array.
        mask: ``(H, W)`` label array with the same height and width.
        tile: Square tile size in pixels.

    Yields:
        ``(k, y, x, image_tile, mask_tile)`` where ``k`` is the official tile
        counter and ``(y, x)`` the top-left pixel. Tiles are views, not copies.

    Raises:
        ValueError: If image and mask sizes differ.
    """
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"image {image.shape[:2]} and mask {mask.shape[:2]} differ")
    h, w = mask.shape[:2]
    k = 0
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            if y + tile <= h and x + tile <= w:
                yield k, y, x, image[y:y + tile, x:x + tile], mask[y:y + tile, x:x + tile]
            k += 1


def read_scene(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Decode a scene with OpenCV, returning ``(image_bgr, mask, pixel_size_m)``.

    The image is kept in OpenCV's BGR order because it is written back with
    OpenCV, which yields a correctly ordered RGB PNG on disk.
    """
    import cv2
    import tifffile

    from landcover.data.inspect import pixel_size_from_tiff

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if img is None or mask is None:
        raise IOError(f"could not decode {image_path} / {mask_path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    with tifffile.TiffFile(image_path) as tif:
        px = pixel_size_from_tiff(tif)
    return img, mask, px


def tile_scene(scene: str, raw_dir: Path, out_dir: Path, tile: int, png_level: int = 3) -> list[dict]:
    """Write all full tiles of one scene as PNG and return manifest rows.

    Args:
        scene: Scene stem, e.g. ``M-33-20-D-c-4-2``.
        raw_dir: Folder with ``images/`` and ``masks/`` GeoTIFFs.
        out_dir: Output root; tiles go to ``out_dir/images`` and
            ``out_dir/masks`` as ``<scene>_<k>.png``.
        tile: Tile size in pixels.
        png_level: PNG zlib compression level (lossless at every level).

    Returns:
        One manifest dict per tile with position, pixel size, number of
        all-zero (no-data) RGB pixels and per-class pixel counts.
    """
    import cv2

    img, mask, px = read_scene(raw_dir / "images" / f"{scene}.tif", raw_dir / "masks" / f"{scene}.tif")
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_PNG_COMPRESSION, png_level]
    rows = []
    for k, y, x, it, mt in iter_tiles(img, mask, tile):
        tid = f"{scene}_{k}"
        if not cv2.imwrite(str(out_dir / "images" / f"{tid}.png"), it, params):
            raise IOError(f"failed to write image tile {tid}")
        if not cv2.imwrite(str(out_dir / "masks" / f"{tid}.png"), np.ascontiguousarray(mt), params):
            raise IOError(f"failed to write mask tile {tid}")
        counts = np.bincount(mt.ravel(), minlength=256)
        row = {"tile_id": tid, "scene": scene, "k": k, "y": y, "x": x, "pixel_size_m": px,
               "nodata_px": int(np.all(it == 0, axis=-1).sum())}
        row.update({f"px_c{c}": int(counts[c]) for c in range(NUM_CLASSES)})
        if counts[NUM_CLASSES:].any():
            raise ValueError(f"{tid} contains labels outside 0..{NUM_CLASSES - 1}")
        rows.append(row)
    return rows


def write_manifest(rows: list[dict], path: Path) -> None:
    """Write manifest rows to CSV sorted by ``(scene, k)`` for a stable order."""
    rows = sorted(rows, key=lambda r: (r["scene"], r["k"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_manifest(path: Path) -> list[dict]:
    """Read a tile manifest CSV, converting numeric columns to int/float."""
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            for key in r:
                if key in ("tile_id", "scene"):
                    continue
                r[key] = float(r[key]) if key == "pixel_size_m" else int(r[key])
            out.append(r)
    return out
