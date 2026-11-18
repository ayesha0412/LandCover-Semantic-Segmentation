"""Read-only inspection of the raw LandCover.ai orthophoto dataset.

These functions never modify the raw data. They establish facts the later
preprocessing milestone (M1) depends on: file pairing, image/mask geometry,
the mask label encoding, the official split's tile-naming convention, and
class pixel frequencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GEOTIFF_PIXEL_SCALE_TAG = 33550


@dataclass
class SceneReport:
    """Per-orthophoto inspection result."""

    name: str
    image_shape: tuple
    image_dtype: str
    mask_shape: tuple
    mask_dtype: str
    pixel_size_m: float | None
    mask_values: dict
    zero_image_pixel_fraction: float
    n_cols_grid: int
    n_rows_grid: int
    n_full_tiles: int
    errors: list = field(default_factory=list)


def pair_scene_files(images_dir: Path, masks_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """Match image and mask GeoTIFFs by file stem.

    Args:
        images_dir: Folder containing ``*.tif`` orthophotos.
        masks_dir: Folder containing ``*.tif`` label masks.

    Returns:
        ``(paired, images_only, masks_only)`` — sorted lists of scene stems
        present in both folders, only in ``images_dir``, and only in
        ``masks_dir`` respectively.
    """
    imgs = {p.stem for p in Path(images_dir).glob("*.tif")}
    msks = {p.stem for p in Path(masks_dir).glob("*.tif")}
    return sorted(imgs & msks), sorted(imgs - msks), sorted(msks - imgs)


def tile_grid(height: int, width: int, tile: int) -> tuple[int, int, int]:
    """Describe the tiling grid used by the official ``split.py``.

    The official script steps over ``range(0, H, tile)`` and
    ``range(0, W, tile)``, incrementing a counter ``k`` for every position but
    saving only tiles that are fully inside the image. Tile ``k`` is therefore
    at row ``k // n_cols`` and column ``k % n_cols`` of a ceil-sized grid.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        tile: Square tile size in pixels.

    Returns:
        ``(n_rows, n_cols, n_full)`` — ceil grid rows, ceil grid columns and
        number of tiles that fit entirely within the image.
    """
    n_rows = math.ceil(height / tile)
    n_cols = math.ceil(width / tile)
    n_full = (height // tile) * (width // tile)
    return n_rows, n_cols, n_full


def tile_index_to_origin(k: int, height: int, width: int, tile: int) -> tuple[int, int] | None:
    """Convert an official tile index ``k`` to its pixel origin.

    Args:
        k: Tile counter as written in the official split files
            (``<scene>_<k>``).
        height: Image height in pixels.
        width: Image width in pixels.
        tile: Square tile size in pixels.

    Returns:
        ``(y, x)`` of the tile's top-left corner, or ``None`` if ``k`` is out
        of range or refers to a partial (discarded) edge tile.
    """
    n_rows, n_cols, _ = tile_grid(height, width, tile)
    if k < 0 or k >= n_rows * n_cols:
        return None
    y, x = (k // n_cols) * tile, (k % n_cols) * tile
    if y + tile > height or x + tile > width:
        return None
    return y, x


def parse_tile_id(tile_id: str) -> tuple[str, int]:
    """Split an official tile ID such as ``M-33-20-D-c-4-2_17`` into parts.

    Args:
        tile_id: ``<scene stem>_<k>`` string from ``train/val/test.txt``.

    Returns:
        ``(scene_stem, k)``.

    Raises:
        ValueError: If the ID does not end in ``_<integer>``.
    """
    scene, sep, k = tile_id.strip().rpartition("_")
    if not sep or not scene or not k.isdigit():
        raise ValueError(f"Malformed tile id: {tile_id!r}")
    return scene, int(k)


def read_split_file(path: Path) -> list[str]:
    """Read a split file, returning non-empty stripped lines in file order."""
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]


def validate_splits(
    splits: dict[str, list[str]], scene_shapes: dict[str, tuple[int, int]], tile: int
) -> dict:
    """Check that official split IDs are well-formed, valid and disjoint.

    Args:
        splits: Mapping split name -> list of tile IDs.
        scene_shapes: Mapping scene stem -> ``(height, width)``.
        tile: Tile size in pixels.

    Returns:
        Dictionary with per-split counts, duplicates, unknown scenes, invalid
        (partial/out-of-range) tiles, pairwise overlaps, total valid full
        tiles in the dataset and how many of them no split uses.
    """
    report: dict = {"counts": {}, "duplicates": {}, "unknown_scene": {}, "invalid_tile": {}, "overlaps": {}}
    sets = {}
    for name, ids in splits.items():
        report["counts"][name] = len(ids)
        report["duplicates"][name] = len(ids) - len(set(ids))
        unknown, invalid = [], []
        for tid in ids:
            scene, k = parse_tile_id(tid)
            if scene not in scene_shapes:
                unknown.append(tid)
                continue
            h, w = scene_shapes[scene]
            if tile_index_to_origin(k, h, w, tile) is None:
                invalid.append(tid)
        report["unknown_scene"][name] = unknown
        report["invalid_tile"][name] = invalid
        sets[name] = set(ids)
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            report["overlaps"][f"{a}&{b}"] = len(sets[a] & sets[b])
    total_full = sum(tile_grid(h, w, tile)[2] for h, w in scene_shapes.values())
    used = set().union(*sets.values()) if sets else set()
    report["total_full_tiles"] = total_full
    report["unused_full_tiles"] = total_full - len(used)
    return report


def per_tile_class_counts(mask: np.ndarray, tile: int, num_values: int = 256) -> np.ndarray:
    """Count label values in every full tile of a 2-D mask.

    Args:
        mask: ``(H, W)`` integer label mask.
        tile: Tile size in pixels.
        num_values: Length of the histogram (labels must be < this).

    Returns:
        Array of shape ``(n_rows_full, n_cols_full, num_values)`` with pixel
        counts per tile, ordered like the row-major official grid restricted
        to full tiles.
    """
    nr, nc = mask.shape[0] // tile, mask.shape[1] // tile
    grid = mask[: nr * tile, : nc * tile].reshape(nr, tile, nc, tile)
    counts = np.zeros((nr, nc, num_values), dtype=np.int64)
    for v in np.unique(grid):
        counts[:, :, int(v)] = (grid == v).sum(axis=(1, 3))
    return counts


def pixel_size_from_tiff(tif) -> float | None:
    """Return the GeoTIFF X pixel size in map units (metres), if tagged."""
    tag = tif.pages[0].tags.get(GEOTIFF_PIXEL_SCALE_TAG)
    if tag is None:
        return None
    return float(tag.value[0])


def inspect_scene(image_path: Path, mask_path: Path, tile: int) -> tuple[SceneReport, np.ndarray, np.ndarray]:
    """Load one orthophoto/mask pair and compute geometry and label statistics.

    Pixels are decoded with OpenCV (the same decoder used by the official
    ``split.py``) because the orthophotos are JPEG-compressed TIFFs; GeoTIFF
    tags are read with ``tifffile`` without decoding pixel data.

    Args:
        image_path: Path to the RGB GeoTIFF.
        mask_path: Path to the label GeoTIFF.
        tile: Tile size in pixels.

    Returns:
        ``(report, tile_counts, tile_zero)`` where ``tile_counts`` is the
        output of :func:`per_tile_class_counts` for the mask (empty if the
        mask could not be reduced to a 2-D label map) and ``tile_zero`` is a
        ``(n_rows_full, n_cols_full)`` count of all-zero (no-data) RGB pixels
        per full tile.
    """
    import cv2
    import tifffile

    with tifffile.TiffFile(image_path) as tif:
        px = pixel_size_from_tiff(tif)
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if img is None or mask is None:
        raise IOError(f"OpenCV could not decode {image_path} or {mask_path}")
    if img.ndim == 3 and img.shape[2] >= 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
    errors = []
    if mask.ndim == 3:
        if np.all(mask == mask[..., :1]):
            mask = mask[..., 0]
        else:
            errors.append("mask has differing channels")
    if img.shape[:2] != mask.shape[:2]:
        errors.append(f"shape mismatch image {img.shape[:2]} vs mask {mask.shape[:2]}")
    vals, cnts = np.unique(mask, return_counts=True)
    zero = np.all(img == 0, axis=-1) if img.ndim == 3 else (img == 0)
    zero_frac = float(zero.mean())
    tile_zero = per_tile_class_counts(zero.astype(np.uint8), tile, 2)[:, :, 1]
    h, w = img.shape[:2]
    n_rows, n_cols, n_full = tile_grid(h, w, tile)
    counts = per_tile_class_counts(mask, tile) if mask.ndim == 2 and not errors else np.empty(0)
    rep = SceneReport(
        name=Path(image_path).stem,
        image_shape=tuple(img.shape),
        image_dtype=str(img.dtype),
        mask_shape=tuple(mask.shape),
        mask_dtype=str(mask.dtype),
        pixel_size_m=px,
        mask_values={int(v): int(c) for v, c in zip(vals, cnts)},
        zero_image_pixel_fraction=zero_frac,
        n_cols_grid=n_cols,
        n_rows_grid=n_rows,
        n_full_tiles=n_full,
        errors=errors,
    )
    return rep, counts, tile_zero
