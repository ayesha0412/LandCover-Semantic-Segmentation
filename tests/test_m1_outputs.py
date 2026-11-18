"""Acceptance checks on real M1 outputs (skipped until prepare/validate have run)."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "outputs" / "m1" / "validation.json"
STATS = ROOT / "data" / "splits" / "seed42" / "subset_stats.json"


@pytest.mark.skipif(not VAL.exists(), reason="run scripts/validate_data.py first")
def test_validation_passed():
    """Every tile is an exact crop of the raw scene and the subsets are nested."""
    r = json.loads(VAL.read_text())
    assert r["ok"], r
    assert r["manifest_tiles"] == r["tile_checks"]["checked"] == 10674
    assert r["subsets"]["sizes"] == {"10": 747, "20": 1494, "30": 2241, "100": 7470}


@pytest.mark.skipif(not STATS.exists(), reason="run scripts/prepare_data.py first")
def test_every_labeled_subset_contains_all_classes():
    """Even the 10% subset must contain labeled pixels of every class."""
    s = json.loads(STATS.read_text())
    for name in ("labeled_10", "labeled_20", "labeled_30"):
        assert all(n > 0 for n in s[name]["tiles_with_class"]), (name, s[name])
