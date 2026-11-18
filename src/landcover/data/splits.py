"""Official splits and nested labeled/unlabeled training subsets.

Subsets are drawn at tile (image) level, matching the project question "only
10/20/30% of training images are labeled". A single seeded permutation of the
sorted training IDs is cut at increasing lengths, so every smaller labeled set
is contained in every larger one (nested). The unlabeled set for a fraction is
the complement of its labeled set within the training split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from landcover.constants import NUM_CLASSES


def nested_labeled_subsets(train_ids: list[str], fractions: list[float], seed: int) -> dict[float, list[str]]:
    """Build nested labeled subsets from one seeded permutation.

    Args:
        train_ids: All training tile IDs (order does not matter; they are
            sorted first so results depend only on the ID set and the seed).
        fractions: Labeled fractions in (0, 1], e.g. ``[0.1, 0.2, 0.3, 1.0]``.
        seed: Seed for ``numpy.random.default_rng``.

    Returns:
        Mapping fraction -> list of labeled IDs, each of size
        ``round(fraction * len(train_ids))``, in permutation order.

    Raises:
        ValueError: If a fraction is outside (0, 1] or IDs are duplicated.
    """
    ids = sorted(train_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate training IDs")
    perm = [ids[i] for i in np.random.default_rng(seed).permutation(len(ids))]
    out = {}
    for f in sorted(fractions):
        if not 0 < f <= 1:
            raise ValueError(f"fraction {f} outside (0, 1]")
        out[f] = perm[: int(round(f * len(ids)))]
    return out


def write_ids(ids: list[str], path: Path) -> None:
    """Write one tile ID per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n")


def read_ids(path: Path) -> list[str]:
    """Read one tile ID per line, ignoring blank lines."""
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]


def subset_class_stats(ids: list[str], manifest: dict[str, dict]) -> dict:
    """Summarise class composition of a tile subset from manifest counts.

    Args:
        ids: Tile IDs in the subset.
        manifest: Mapping tile_id -> manifest row with ``px_c{c}`` columns.

    Returns:
        Dict with tile count, per-class pixel share (%), and per-class number
        of tiles containing at least one pixel of that class.
    """
    px = np.zeros(NUM_CLASSES, dtype=np.int64)
    tiles = np.zeros(NUM_CLASSES, dtype=np.int64)
    for t in ids:
        c = np.array([manifest[t][f"px_c{i}"] for i in range(NUM_CLASSES)])
        px += c
        tiles += c > 0
    share = 100 * px / max(px.sum(), 1)
    return {"n_tiles": len(ids), "pixel_share_pct": [round(float(s), 3) for s in share],
            "tiles_with_class": tiles.tolist()}
