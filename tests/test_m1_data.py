"""Unit tests for M1 tiling, subsets, transforms, dataset and loaders."""

import cv2
import numpy as np
import pytest
import torch

from landcover.data.dataset import TileDataset
from landcover.data.inspect import parse_tile_id, tile_index_to_origin
from landcover.data.loaders import build_loader
from landcover.data.splits import nested_labeled_subsets, subset_class_stats
from landcover.data.tiling import iter_tiles
from landcover.data.transforms import IMAGENET_MEAN, IMAGENET_STD, Normalize, RandomD4, apply_d4, build_transforms


def _scene(h=1300, w=1100, seed=0):
    """Random synthetic RGB scene and 5-class mask."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8), rng.integers(0, 5, (h, w), dtype=np.uint8)


def test_iter_tiles_official_order_and_origin():
    """Tile k must sit where tile_index_to_origin (official convention) says."""
    img, msk = _scene()
    tiles = list(iter_tiles(img, msk, 512))
    assert [t[0] for t in tiles] == [0, 1, 3, 4]
    for k, y, x, it, mt in tiles:
        assert tile_index_to_origin(k, 1300, 1100, 512) == (y, x)
        assert np.array_equal(it, img[y:y + 512, x:x + 512])
        assert np.array_equal(mt, msk[y:y + 512, x:x + 512])
    with pytest.raises(ValueError):
        list(iter_tiles(img, msk[:-1], 512))


def test_png_roundtrip_is_lossless(tmp_path):
    """PNG storage must reproduce image (BGR) and mask bytes exactly."""
    img, msk = _scene(512, 512)
    cv2.imwrite(str(tmp_path / "i.png"), img)
    cv2.imwrite(str(tmp_path / "m.png"), msk)
    assert np.array_equal(cv2.imread(str(tmp_path / "i.png"), cv2.IMREAD_UNCHANGED), img)
    assert np.array_equal(cv2.imread(str(tmp_path / "m.png"), cv2.IMREAD_UNCHANGED), msk)


def test_nested_subsets_properties():
    """Subsets are nested, correctly sized, seed-deterministic and order-invariant."""
    ids = [f"S_{i}" for i in range(7470)]
    a = nested_labeled_subsets(ids, [0.1, 0.2, 0.3, 1.0], seed=42)
    b = nested_labeled_subsets(list(reversed(ids)), [0.3, 0.1, 1.0, 0.2], seed=42)
    c = nested_labeled_subsets(ids, [0.1], seed=7)
    assert {f: len(v) for f, v in a.items()} == {0.1: 747, 0.2: 1494, 0.3: 2241, 1.0: 7470}
    assert a == b
    assert set(a[0.1]) < set(a[0.2]) < set(a[0.3]) < set(a[1.0]) == set(ids)
    assert a[0.1] != c[0.1]
    with pytest.raises(ValueError):
        nested_labeled_subsets(ids, [0.0], seed=1)
    with pytest.raises(ValueError):
        nested_labeled_subsets(ids + ids[:1], [0.1], seed=1)


def test_subset_class_stats():
    """Pixel shares and per-class tile presence are computed from the manifest."""
    man = {"a": {"px_c0": 3, "px_c1": 1, "px_c2": 0, "px_c3": 0, "px_c4": 0},
           "b": {"px_c0": 0, "px_c1": 0, "px_c2": 4, "px_c3": 0, "px_c4": 0}}
    s = subset_class_stats(["a", "b"], man)
    assert s["n_tiles"] == 2
    assert s["pixel_share_pct"] == [37.5, 12.5, 50.0, 0.0, 0.0]
    assert s["tiles_with_class"] == [1, 1, 1, 0, 0]


def test_d4_keeps_alignment_and_covers_group():
    """Image and mask must move together; all 8 D4 poses must be distinct and reachable."""
    rng = np.random.default_rng(1)
    msk = rng.integers(0, 5, (8, 8), dtype=np.uint8)
    img = np.stack([msk * 50, msk, rng.integers(0, 256, (8, 8), dtype=np.uint8)], -1).astype(np.uint8)
    poses = set()
    for rot in range(4):
        for flip in (False, True):
            i2, m2 = apply_d4(img, msk, rot, flip)
            assert np.array_equal(i2[..., 0], m2 * 50)
            poses.add(m2.tobytes())
    assert len(poses) == 8
    np.random.seed(0)
    seen = {RandomD4()(img, msk)[1].tobytes() for _ in range(200)}
    assert seen == poses


def test_normalize_values():
    """Normalisation uses ImageNet statistics, CHW layout and int64 masks."""
    img = np.full((4, 4, 3), 255, np.uint8)
    x, y = Normalize()(img, np.ones((4, 4), np.uint8))
    assert x.shape == (3, 4, 4) and x.dtype == torch.float32 and y.dtype == torch.int64
    expected = [(1 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]
    assert torch.allclose(x[:, 0, 0], torch.tensor(expected))


@pytest.fixture
def tile_root(tmp_path):
    """Folder with 6 synthetic 512x512 tiles in the processed layout."""
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    ids = []
    for i in range(6):
        img, msk = _scene(512, 512, seed=i)
        tid = f"S-{i}_{i}"
        cv2.imwrite(str(tmp_path / "images" / f"{tid}.png"), img)
        cv2.imwrite(str(tmp_path / "masks" / f"{tid}.png"), msk)
        ids.append(tid)
    return tmp_path, ids


def test_dataset_items(tile_root):
    """Labeled items carry RGB images and masks; unlabeled items never read masks."""
    root, ids = tile_root
    ds = TileDataset(root, ids, build_transforms(train=False))
    it = ds[0]
    assert it["image"].shape == (3, 512, 512) and it["mask"].shape == (512, 512)
    raw_bgr = cv2.imread(str(root / "images" / f"{ids[0]}.png"))
    x = it["image"] * torch.tensor(IMAGENET_STD).view(3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    assert torch.allclose(x[0] * 255, torch.from_numpy(raw_bgr[..., 2]).float(), atol=1e-3)
    (root / "masks" / f"{ids[1]}.png").unlink()
    u = TileDataset(root, ids, build_transforms(train=True), labeled=False)
    assert "mask" not in u[1]
    with pytest.raises(FileNotFoundError):
        TileDataset(root, ids, build_transforms(train=False))[1]


@pytest.mark.parametrize("workers", [0, 2])
def test_loader_reproducible(tile_root, workers):
    """Same seed gives identical order and augmentation; a different seed does not."""
    root, ids = tile_root

    def first_epoch(seed):
        dl = build_loader(TileDataset(root, ids, build_transforms(train=True)), 3, True, seed, num_workers=workers)
        return [(b["id"], b["mask"]) for b in dl]

    a, b, c = first_epoch(5), first_epoch(5), first_epoch(6)
    assert all(x[0] == y[0] and torch.equal(x[1], y[1]) for x, y in zip(a, b))
    assert any(x[0] != y[0] or not torch.equal(x[1], y[1]) for x, y in zip(a, c))
    assert sorted(i for ids_, _ in a for i in ids_) == sorted(ids)
    assert parse_tile_id(a[0][0][0])[0].startswith("S-")
