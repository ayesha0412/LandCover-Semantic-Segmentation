"""M1 preprocessing: tile scenes, copy official splits, build nested labeled subsets.

Usage:
    python scripts/prepare_data.py --config configs/base.yaml [--workers 6] [--skip-tiling]

Outputs (paths from the config):
    data/processed/images/<id>.png, data/processed/masks/<id>.png
    data/processed/manifest.csv
    data/splits/{train,val,test}.txt
    data/splits/seed<S>/labeled_<P>.txt, unlabeled_<P>.txt
    data/splits/seed<S>/subset_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from landcover.constants import SPLITS  # noqa: E402
from landcover.data.inspect import pair_scene_files, read_split_file  # noqa: E402
from landcover.data.splits import nested_labeled_subsets, subset_class_stats, write_ids  # noqa: E402
from landcover.data.tiling import read_manifest, tile_scene, write_manifest  # noqa: E402


def run_tiling(raw_dir: Path, out_dir: Path, tile: int, workers: int) -> list[dict]:
    """Tile every paired scene in parallel and return all manifest rows."""
    scenes, img_only, msk_only = pair_scene_files(raw_dir / "images", raw_dir / "masks")
    if img_only or msk_only:
        raise RuntimeError(f"unpaired scenes: {img_only} {msk_only}")
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(tile_scene, s, raw_dir, out_dir, tile): s for s in scenes}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.extend(r)
            print(f"[{i}/{len(scenes)}] {futs[fut]}: {len(r)} tiles", flush=True)
    return rows


def main() -> int:
    """Run preprocessing according to the YAML config."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "base.yaml")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-tiling", action="store_true", help="Reuse an existing manifest.")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    raw = ROOT / cfg["paths"]["raw_dir"]
    proc = ROOT / cfg["paths"]["processed_dir"]
    spl = ROOT / cfg["paths"]["splits_dir"]
    tile = cfg["dataset"]["tile_size"]
    seed = cfg["dataset"]["split_seed"]
    t0 = time.time()
    manifest_path = proc / "manifest.csv"
    if args.skip_tiling:
        rows = read_manifest(manifest_path)
    else:
        rows = run_tiling(raw, proc, tile, args.workers)
        write_manifest(rows, manifest_path)
    print(f"{len(rows)} tiles in manifest ({time.time() - t0:.0f}s)")
    by_id = {r["tile_id"]: r for r in rows}
    splits = {s: read_split_file(raw / f"{s}.txt") for s in SPLITS}
    for s, ids in splits.items():
        missing = [t for t in ids if t not in by_id]
        if missing:
            raise RuntimeError(f"{len(missing)} {s} ids have no tile, e.g. {missing[:3]}")
        write_ids(ids, spl / f"{s}.txt")
    subsets = nested_labeled_subsets(splits["train"], cfg["dataset"]["label_fractions"], seed)
    train_set = set(splits["train"])
    stats = {s: subset_class_stats(ids, by_id) for s, ids in splits.items()}
    for f, lab in subsets.items():
        pct = int(round(f * 100))
        unl = sorted(train_set - set(lab))
        write_ids(lab, spl / f"seed{seed}" / f"labeled_{pct}.txt")
        if unl:
            write_ids(unl, spl / f"seed{seed}" / f"unlabeled_{pct}.txt")
        stats[f"labeled_{pct}"] = subset_class_stats(lab, by_id)
        stats[f"labeled_{pct}"]["n_unlabeled"] = len(unl)
    (spl / f"seed{seed}" / "subset_stats.json").write_text(json.dumps(stats, indent=2))
    for name, st in stats.items():
        print(f"{name:12s} n={st['n_tiles']:5d} share%={st['pixel_share_pct']} tiles_with={st['tiles_with_class']}")
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
