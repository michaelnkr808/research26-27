#!/usr/bin/env python
"""List the InteriorGS scenes check 3 needs (CPU only, run before downloading).

Reads each split's GN-Matrix parquet, applies the same episode selection as
run_check3.py (c3lib.select_rows with the config seed) and writes the sorted
unique scene ids, one per line, to --out. Prints episode and scene counts per
split. Episodes in scenes that are not downloaded are skipped by
run_check3.py and reported, so the list has to be complete.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c3lib  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=HERE.parent.parent / "config.yaml")
    ap.add_argument("--gn-matrix-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = c3lib.load_config(args.config)
    import pyarrow.parquet as pq

    all_scenes = set()
    for split, spec in cfg["splits"].items():
        t = pq.read_table(args.gn_matrix_dir / f"{split}.parquet", columns=["scene"])
        rows = c3lib.select_rows(t.num_rows, spec, cfg["seed"])
        scenes = t.column("scene").to_pylist()
        sel = {str(scenes[int(r)]) for r in rows}
        all_scenes |= sel
        print(f"{split}: {t.num_rows} rows, {len(rows)} selected, {len(sel)} scenes")
    args.out.write_text("\n".join(sorted(all_scenes)) + "\n")
    print(f"total unique scenes: {len(all_scenes)} -> {args.out}")


if __name__ == "__main__":
    main()
