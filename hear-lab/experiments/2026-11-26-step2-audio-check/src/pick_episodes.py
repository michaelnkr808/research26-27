#!/usr/bin/env python3
"""Freeze the Step 2 episode order (PLAN.md, "Episodes"). Runs on the Mac.

Reads the route A check 3 train_seen table and scenes.jsonl (both local only, pinned by sha256 in config.yaml),
drops the possibly mirrored scenes by the check 3 rule, splits the eligible episodes by the visible flag at the
0.15 m tolerance, and writes a balanced, seeded, interleaved order of episode ids (step2lib.frozen_order).
Batch 1 is its first 12 ids, the full set its first full_set_n. Only the id list is written (and committed).

  python3.11 src/pick_episodes.py                 # write episode_order.txt and print its sha256
  python3.11 src/pick_episodes.py --check         # recompute and compare with the committed file and config
  python3.11 src/pick_episodes.py --print-scenes N   # scenes of the first N ids (no CSV needed)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
import step2lib as s2  # noqa: E402


def build(cfg: dict, exp: Path = EXP, check3_dir: Path | None = None) -> tuple:
    """check3_dir overrides the folder of both input files (for a worktree, whose gitignored results/ is empty:
    point it at the main checkout's copy)."""
    e = cfg["episodes"]
    csv_path = (exp / e["source_csv"]).resolve()
    sj_path = (exp / e["scenes_jsonl"]).resolve()
    if check3_dir is not None:
        csv_path, sj_path = Path(check3_dir) / csv_path.name, Path(check3_dir) / sj_path.name
    for p, key in ((csv_path, "source_csv_sha256"), (sj_path, "scenes_jsonl_sha256")):
        got = s2.sha256_file(p)
        if got != e[key]:
            raise SystemExit(f"{p.name}: sha256 {got} != config episodes.{key} {e[key]}")
    rows = s2.read_csv(csv_path)
    mirrored = s2.mirrored_scenes(sj_path.read_text(encoding="utf-8").splitlines())
    in_m = [r for r in rows if r["scene"] in mirrored]
    counts = {
        "n_rows": len(rows),
        "n_mirrored_scenes": len(mirrored),
        "n_rows_in_mirrored_scenes": len(in_m),
        "n_visible_in_mirrored_scenes": sum(r[e["visible_column"]] == "1" for r in in_m),
    }
    vis, nvis = s2.split_by_visibility(rows, mirrored, e["split"], e["visible_column"], e["tolerance_column"],
                                       e["tolerance_m"])
    counts["n_eligible_visible"] = len(vis)
    counts["n_eligible_not_visible"] = len(nvis)
    bad = {k: (v, e["expected"][k]) for k, v in counts.items() if v != e["expected"][k]}
    if bad:
        raise SystemExit(f"counts differ from config episodes.expected: {bad}")
    order = s2.frozen_order(vis, nvis, int(e["order_seed"]))
    if len(order) != int(e["order_length"]):
        raise SystemExit(f"order length {len(order)} != config {e['order_length']}")
    vset = set(vis)
    for i, eid in enumerate(order):
        if (eid in vset) != (i % 2 == 0):
            raise SystemExit(f"position {i}: interleave broken")
    return order, counts, mirrored


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=EXP / "config.yaml")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check3-dir", type=Path, default=None,
                    help="folder holding the check 3 train_seen CSV and scenes.jsonl (default: the config paths)")
    ap.add_argument("--print-scenes", type=int, default=None, metavar="N")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    out = EXP / cfg["episodes"]["order_file"]
    if a.print_scenes is not None:
        ids = [x for x in out.read_text(encoding="utf-8").split() if x][: a.print_scenes]
        seen = []
        for x in ids:
            if s2.scene_of(x) not in seen:
                seen.append(s2.scene_of(x))
        print("\n".join(seen))
        return 0
    order, counts, mirrored = build(cfg, check3_dir=a.check3_dir)
    text = s2.order_text(order)
    sha = s2.sha256_text(text)
    low = set(cfg["direction_check"]["low_wall_share_scenes"])
    full = order[: int(cfg["episodes"]["full_set_n"])]
    info = (f"{len(order)} ids, sha256 {sha}; counts {counts}; batch 1 scenes "
            f"{len({s2.scene_of(x) for x in order[: int(cfg['episodes']['batch1_n'])]})}, full set scenes "
            f"{len({s2.scene_of(x) for x in full})}; low wall-share scenes in the full set: "
            f"{sorted(low & {s2.scene_of(x) for x in full})}")
    if a.check:
        committed = out.read_text(encoding="utf-8")
        ok = committed == text and cfg["episodes"]["order_sha256"] == sha
        print(("OK " if ok else "MISMATCH ") + info)
        return 0 if ok else 1
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out.name}: " + info)
    if cfg["episodes"]["order_sha256"] not in ("pending", sha):
        print(f"MISMATCH: config episodes.order_sha256 is {cfg['episodes']['order_sha256']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
