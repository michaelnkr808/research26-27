#!/usr/bin/env python3
"""Freeze the rung 1 episode order: a full random permutation of all GN-Bench test_unseen episode ids.

Source of the ids: the check 3 per-episode CSV (one row per GN-Matrix test_unseen parquet row, in parquet
row order, columns scene and trajectory_id). The id format is GN0's own, gn0/evaluation/common.py
get_episode_id: "<scene>_<trajectory_id>" (ref_json is "<scene>/<trajectory_id>.json"). setup.sh checks on
the machine that the set of ids equals the set GN0 builds from the pinned GN-Matrix parquet.

Any episode count N is the first N lines of the output; the timing batch is the first K.
The output holds InteriorGS scene names, so it stays under results/ (gitignored).

Usage:
  python make_episode_order.py --csv <check3_InteriorGS_test_unseen.csv> --out ../results/episode_order.txt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parent / "config.yaml"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_from_check3_csv(csv_path: Path, split: str) -> list[str]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows):
        if r["split"] != split:
            raise SystemExit(f"row {i}: split {r['split']!r}, expected {split!r}")
        if int(r["row_index"]) != i:
            raise SystemExit(f"row {i}: row_index {r['row_index']} out of parquet order")
    ids = [f"{r['scene']}_{r['trajectory_id']}" for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate episode ids in the CSV")
    return ids


def frozen_order(ids_in_row_order: list[str], seed: int) -> list[str]:
    perm = np.random.default_rng(seed).permutation(len(ids_in_row_order))
    return [ids_in_row_order[i] for i in perm]


def order_text(order: list[str]) -> str:
    return "".join(f"{i}\n" for i in order)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config", default=CONFIG, type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))["episodes"]

    csv_sha = sha256_file(args.csv)
    if csv_sha != cfg["source_sha256"]:
        raise SystemExit(f"CSV sha256 {csv_sha} != config episodes.source_sha256 {cfg['source_sha256']}")
    ids = ids_from_check3_csv(args.csv, cfg["split"])
    if len(ids) != cfg["n_total"]:
        raise SystemExit(f"{len(ids)} ids, config says n_total {cfg['n_total']}")
    order = frozen_order(ids, int(cfg["order_seed"]))
    text = order_text(order)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"wrote {args.out}: {len(order)} ids, sha256 {sha}")
    expected = cfg.get("order_sha256")
    if expected and expected != sha:
        print(f"MISMATCH: config episodes.order_sha256 is {expected}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
