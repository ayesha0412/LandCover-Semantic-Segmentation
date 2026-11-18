"""M1 validation: prove processed tiles are exact, aligned copies of the raw scenes.

Usage:
    python scripts/validate_data.py --config configs/base.yaml [--workers 6]

For every scene the raw GeoTIFF is decoded again and every tile PNG on disk is
compared pixel-for-pixel (image and mask) with the corresponding crop. Mask
class counts are compared with the manifest, the labeled subsets are checked
for nesting and completeness, and (if an M0 inspection report exists)
per-split pixel totals are compared with it. Writes ``outputs/m1/validation.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from landcover.constants import NUM_CLASSES, SPLITS  # noqa: E402
from landcover.data.splits import read_ids  # noqa: E402
from landcover.data.tiling import iter_tiles, read_manifest, read_scene  # noqa: E402


def check_scene(scene: str, raw_dir: Path, proc_dir: Path, tile: int, rows: dict) -> dict:
    """Compare all tiles of one scene on disk with fresh crops of the raw data.

    Args:
        scene: Scene stem.
        raw_dir: Raw dataset folder.
        proc_dir: Processed tile folder.
        tile: Tile size.
        rows: Manifest rows of this scene keyed by tile ID.

    Returns:
        Dict with counts of checked tiles and lists of IDs failing each check.
    """
    import cv2

    img, mask, _ = read_scene(raw_dir / "images" / f"{scene}.tif", raw_dir / "masks" / f"{scene}.tif")
    res = {"checked": 0, "missing": [], "image_mismatch": [], "mask_mismatch": [], "count_mismatch": [],
           "bad_shape": [], "extra_in_manifest": []}
    seen = set()
    for k, _, _, it, mt in iter_tiles(img, mask, tile):
        tid = f"{scene}_{k}"
        seen.add(tid)
        di = cv2.imread(str(proc_dir / "images" / f"{tid}.png"), cv2.IMREAD_UNCHANGED)
        dm = cv2.imread(str(proc_dir / "masks" / f"{tid}.png"), cv2.IMREAD_UNCHANGED)
        if di is None or dm is None or tid not in rows:
            res["missing"].append(tid)
            continue
        if di.shape != (tile, tile, 3) or dm.shape != (tile, tile) or di.dtype != np.uint8 or dm.dtype != np.uint8:
            res["bad_shape"].append(tid)
            continue
        if not np.array_equal(di, it):
            res["image_mismatch"].append(tid)
        if not np.array_equal(dm, mt):
            res["mask_mismatch"].append(tid)
        cnt = np.bincount(dm.ravel(), minlength=256)
        if cnt[NUM_CLASSES:].any() or any(cnt[c] != rows[tid][f"px_c{c}"] for c in range(NUM_CLASSES)):
            res["count_mismatch"].append(tid)
        res["checked"] += 1
    res["extra_in_manifest"] = sorted(set(rows) - seen)
    return res


def main() -> int:
    """Run all validation checks and return non-zero if any fails."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "base.yaml")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    raw, proc = ROOT / cfg["paths"]["raw_dir"], ROOT / cfg["paths"]["processed_dir"]
    spl, tile, seed = ROOT / cfg["paths"]["splits_dir"], cfg["dataset"]["tile_size"], cfg["dataset"]["split_seed"]
    manifest = read_manifest(proc / "manifest.csv")
    by_scene = defaultdict(dict)
    for r in manifest:
        by_scene[r["scene"]][r["tile_id"]] = r
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {s: ex.submit(check_scene, s, raw, proc, tile, rows) for s, rows in by_scene.items()}
        per_scene = {s: f.result() for s, f in futs.items()}
    fail_keys = ["missing", "image_mismatch", "mask_mismatch", "count_mismatch", "bad_shape", "extra_in_manifest"]
    agg = {k: sum(len(v[k]) for v in per_scene.values()) for k in fail_keys}
    agg["checked"] = sum(v["checked"] for v in per_scene.values())
    n_png = {d: len(list((proc / d).glob("*.png"))) for d in ("images", "masks")}

    by_id = {r["tile_id"]: r for r in manifest}
    m0_path = ROOT / "outputs" / "m0" / "dataset_report.json"
    split_ok = {}
    if m0_path.exists():
        m0 = json.loads(m0_path.read_text())["split_stats"]
        for s in SPLITS:
            ids = read_ids(spl / f"{s}.txt")
            tot = [sum(by_id[t][f"px_c{c}"] for t in ids) for c in range(NUM_CLASSES)]
            ref = [int(m0[s]["pixels_per_value"].get(str(c), 0)) for c in range(NUM_CLASSES)]
            split_ok[s] = tot == ref

    train = set(read_ids(spl / "train.txt"))
    fr = sorted(int(round(f * 100)) for f in cfg["dataset"]["label_fractions"])
    lab = {p: read_ids(spl / f"seed{seed}" / f"labeled_{p}.txt") for p in fr}
    subset_ok = {
        "nested": all(set(lab[a]) <= set(lab[b]) for a, b in zip(fr, fr[1:])),
        "within_train": all(set(v) <= train for v in lab.values()),
        "no_duplicates": all(len(set(v)) == len(v) for v in lab.values()),
        "sizes": {p: len(v) for p, v in lab.items()},
        "complement_ok": all(
            set(lab[p]).isdisjoint(read_ids(spl / f"seed{seed}" / f"unlabeled_{p}.txt"))
            and len(lab[p]) + len(read_ids(spl / f"seed{seed}" / f"unlabeled_{p}.txt")) == len(train)
            for p in fr if p < 100
        ),
    }
    ok = (all(v == 0 for k, v in agg.items() if k != "checked") and agg["checked"] == len(manifest)
          and n_png["images"] == n_png["masks"] == len(manifest) and all(split_ok.values())
          and subset_ok["nested"] and subset_ok["within_train"] and subset_ok["no_duplicates"]
          and subset_ok["complement_ok"])
    report = {"ok": ok, "manifest_tiles": len(manifest), "png_files": n_png, "tile_checks": agg,
              "split_pixel_totals_match_m0": split_ok or "skipped (no outputs/m0/dataset_report.json)",
              "subsets": subset_ok,
              "failures_by_scene": {s: {k: v[k][:5] for k in fail_keys if v[k]} for s, v in per_scene.items()
                                    if any(v[k] for k in fail_keys)}}
    out = ROOT / "outputs" / "m1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
