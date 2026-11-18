"""Unit tests for raw-dataset inspection helpers (synthetic data, no GPU)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landcover.data.inspect import (  # noqa: E402
    pair_scene_files,
    parse_tile_id,
    per_tile_class_counts,
    tile_grid,
    tile_index_to_origin,
    validate_splits,
)


def test_tile_grid_matches_official_loop():
    """Grid must match split.py: ceil positions counted, only full tiles kept."""
    h, w, t = 1300, 1100, 512
    k_full = []
    k = 0
    for y in range(0, h, t):
        for x in range(0, w, t):
            if y + t <= h and x + t <= w:
                k_full.append((k, y, x))
            k += 1
    n_rows, n_cols, n_full = tile_grid(h, w, t)
    assert (n_rows, n_cols) == (3, 3)
    assert n_full == len(k_full) == 4
    for k, y, x in k_full:
        assert tile_index_to_origin(k, h, w, t) == (y, x)
    full_ks = {k for k, _, _ in k_full}
    for k in range(n_rows * n_cols):
        if k not in full_ks:
            assert tile_index_to_origin(k, h, w, t) is None
    assert tile_index_to_origin(-1, h, w, t) is None
    assert tile_index_to_origin(9, h, w, t) is None


def test_parse_tile_id():
    """Scene stems contain hyphens; only the final underscore separates k."""
    assert parse_tile_id("M-33-20-D-c-4-2_17") == ("M-33-20-D-c-4-2", 17)
    for bad in ("M-33-20", "M-33_x", "_3"):
        with pytest.raises(ValueError):
            parse_tile_id(bad)


def test_per_tile_class_counts():
    """Per-tile histograms must sum to tile area and match manual counts."""
    rng = np.random.default_rng(0)
    m = rng.integers(0, 5, size=(10, 13), dtype=np.uint8)
    c = per_tile_class_counts(m, 4, num_values=8)
    assert c.shape == (2, 3, 8)
    assert (c.sum(-1) == 16).all()
    for r in range(2):
        for q in range(3):
            manual = np.bincount(m[r * 4:(r + 1) * 4, q * 4:(q + 1) * 4].ravel(), minlength=8)
            assert (c[r, q] == manual).all()


def test_validate_splits_detects_problems():
    """Overlaps, duplicates, unknown scenes and partial tiles are all reported."""
    shapes = {"A": (1024, 1100), "B": (512, 512)}
    splits = {
        "train": ["A_0", "A_1", "A_0", "B_0"],
        "val": ["A_3", "A_2", "C_0"],
        "test": ["B_0"],
    }
    r = validate_splits(splits, shapes, 512)
    assert r["duplicates"]["train"] == 1
    assert r["unknown_scene"]["val"] == ["C_0"]
    assert r["invalid_tile"]["val"] == ["A_2"]
    assert r["overlaps"]["train&test"] == 1
    assert r["total_full_tiles"] == 5


def test_pair_scene_files(tmp_path):
    """Images and masks are paired by stem; orphans are reported."""
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    for n in ("a", "b"):
        (tmp_path / "images" / f"{n}.tif").touch()
    for n in ("b", "c"):
        (tmp_path / "masks" / f"{n}.tif").touch()
    paired, io, mo = pair_scene_files(tmp_path / "images", tmp_path / "masks")
    assert (paired, io, mo) == (["b"], ["a"], ["c"])
