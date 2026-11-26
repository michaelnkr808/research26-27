#!/usr/bin/env python
"""Check 3 summary: counts per split and the review images.

Reads <out-dir>/check3_<split>.csv written by run_check3.py, recomputes the
selected rows from the parquet and config.yaml (so missing episodes are
reported, not hidden), and writes:

* <out-dir>/summary.json and <out-dir>/summary.txt: per split, the number of
  selected / done episodes, status counts (ok, missing_id, missing_box),
  visible / not visible at the primary tolerance and at the two sensitivity
  tolerances, and each of those crossed with the within-3.0-m flags (by
  GN-Matrix path length and by straight-line distance); plus the per-scene
  frame diagnostic and timing from scenes.jsonl.
* <out-dir>/review/<split>/NN_row<row>.png: RGB (left) and depth (right) with
  the projected box points drawn (green: seen at the primary tolerance, red:
  inside the image but hidden), for n_review_images_per_split episodes per
  split, taken in order from c3lib.review_candidates (seeded with the
  config seed), skipping episodes whose box was not found.
* <out-dir>/review/review_index.csv: the chosen episodes with their flags
  and an empty by_eye_agrees column for the person checking them.

Runs anywhere with numpy, pandas/pyarrow and Pillow (no GPU).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(Path(os.environ.get("HEAR_LAB_SRC") or HERE.parents[3] / "src")))  # shared code in hear-lab/src (moved 2026-11-26)
import c3lib  # noqa: E402

EXP_DIR = HERE.parent.parent
MAX_DRAW_POINTS = 4000
DEPTH_VIS_MAX_M = 10.0
# run_check3.py clips depth at its DEPTH_MAX_M (100 m); empty pixels sit there.
EMPTY_DEPTH_M = 99.99


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=EXP_DIR / "config.yaml")
    ap.add_argument("--gn-matrix-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--splits", nargs="*", default=None)
    return ap.parse_args()


def b(x) -> bool:
    return str(x).strip() in ("1", "True", "true")


def split_summary(rows, expected, cfg):
    done = {int(r["row_index"]) for r in rows}
    exp = {int(r) for r in expected}
    status = Counter(r["status"] for r in rows)
    ok = [r for r in rows if r["status"] == "ok"]
    res = {
        "n_selected": len(exp), "n_done": len(done & exp),
        "n_not_done": len(exp - done), "n_extra_rows": len(done - exp),
        "status": dict(status),
        "n_duplicate_label_ids": sum(1 for r in rows if str(r["n_label_entries"]).isdigit() and int(r["n_label_entries"]) > 1),
        "n_ok_with_invalid_depth_hits": sum(1 for r in ok if int(r["n_invalid_depth"] or 0) > 0),
    }
    for tol in ("primary", "strict", "loose"):
        key = f"visible_{tol}"
        vis = [r for r in ok if b(r[key])]
        nvis = [r for r in ok if not b(r[key])]
        entry = {"tolerance_m": float(rows[0][f"depth_tol_{tol}_m"]) if rows else None,
                 "visible": len(vis), "not_visible": len(nvis)}
        for flag in ("within_radius_path", "within_radius_straight"):
            entry[flag] = {
                "visible": sum(b(r[flag]) for r in vis),
                "not_visible": sum(b(r[flag]) for r in nvis),
            }
        res[tol] = entry
    missing = [r for r in rows if r["status"] in ("missing_id", "missing_box")]
    res["missing"] = {
        "n": len(missing),
        "within_radius_path": sum(b(r["within_radius_path"]) for r in missing),
        "within_radius_straight": sum(b(r["within_radius_straight"]) for r in missing),
    }
    rs = [float(r["render_s"]) for r in ok if r.get("render_s")]
    res["render_s_median"] = float(np.median(rs)) if rs else None
    res["all_within_radius_path"] = sum(b(r["within_radius_path"]) for r in rows)
    res["all_within_radius_straight"] = sum(b(r["within_radius_straight"]) for r in rows)
    return res


def scenes_summary(path: Path):
    if not path.exists():
        return None
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not recs:
        return None
    wins = Counter()
    for r in recs:
        d = r.get("frame_diag") or {}
        tot = {k: (d.get(k) or {}).get("total", 0) for k in ("identity", "mirror_x", "mirror_y", "mirror_xy")}
        if any(tot.values()):
            wins[max(tot, key=tot.get)] += 1
    load = np.array([r["load_s"] for r in recs], dtype=float)
    return {"n_scene_loads": len(recs), "load_s_median": float(np.median(load)),
            "load_s_total": float(load.sum()), "frame_diag_most_splats_inside": dict(wins)}


def draw_review(npz_path: Path, rec: dict, out_png: Path):
    from PIL import Image, ImageDraw

    z = np.load(npz_path)
    rgb = z["rgb"].astype(np.uint8)
    depth = z["depth"].astype(np.float32)
    H, W = depth.shape
    dvis = np.clip(depth / DEPTH_VIS_MAX_M, 0.0, 1.0)
    dimg = (255 * (1.0 - dvis)).astype(np.uint8)
    dimg = np.stack([dimg] * 3, axis=-1)
    dimg[depth >= EMPTY_DEPTH_M] = (60, 0, 60)  # nothing rendered (purple)
    header = 44
    canvas = Image.new("RGB", (2 * W, H + header), (255, 255, 255))
    canvas.paste(Image.fromarray(rgb), (0, header))
    canvas.paste(Image.fromarray(dimg), (W, header))
    draw = ImageDraw.Draw(canvas)
    inside = z["in_image"].astype(bool)
    seen = z["seen_primary"].astype(bool)
    idx = np.flatnonzero(inside)
    if len(idx) > MAX_DRAW_POINTS:
        idx = idx[np.linspace(0, len(idx) - 1, MAX_DRAW_POINTS).astype(int)]
    for i in idx:
        color = (0, 200, 0) if seen[i] else (230, 0, 0)
        x, y = float(z["u"][i]), float(z["v"][i]) + header
        for off in (0, W):
            draw.rectangle([x + off - 1, y - 1, x + off + 1, y + 1], fill=color)
    text = (f"{rec['split']} row {rec['row_index']} traj {rec['trajectory_id']} scene {rec['scene']} "
            f"target '{rec['label_label']}' (id {rec['label_ins_id']})\n"
            f"visible@{rec['depth_tol_primary_m']} m: {rec['visible_primary']}  seen {rec['n_seen_primary']}"
            f"/{rec['n_box_points']} pts, in image {rec['n_in_image']}  "
            f"path {float(rec['path_length_m']):.2f} m  green=seen red=hidden")
    draw.multiline_text((4, 2), text, fill=(0, 0, 0))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


def main():
    args = parse_args()
    cfg = c3lib.load_config(args.config)
    import pyarrow.parquet as pq

    splits = args.splits or list(cfg["splits"].keys())
    out = args.out_dir
    summary = {"config": {k: v for k, v in cfg.items()}, "splits": {}}
    review_rows = []
    for split in splits:
        pq_path = args.gn_matrix_dir / f"{split}.parquet"
        n_rows = pq.ParquetFile(pq_path).metadata.num_rows
        expected = c3lib.select_rows(n_rows, cfg["splits"][split], cfg["seed"])
        rows = c3lib.read_rows(out / f"check3_{split}.csv")
        s = split_summary(rows, expected, cfg)
        by_row = {int(r["row_index"]): r for r in rows}
        chosen = []
        for r in c3lib.review_candidates(expected, cfg["n_review"], cfg["seed"]):
            rec = by_row.get(r)
            npz = out / "review_raw" / split / f"{r}.npz"
            if rec is None or rec["status"] != "ok" or not npz.exists():
                continue
            chosen.append((r, rec, npz))
            if len(chosen) >= cfg["n_review"]:
                break
        for k, (r, rec, npz) in enumerate(chosen):
            png = out / "review" / split / f"{k:02d}_row{r}.png"
            draw_review(npz, rec, png)
            review_rows.append({"split": split, "rank": k, "row_index": r,
                                "trajectory_id": rec["trajectory_id"], "scene": rec["scene"],
                                "label": rec["label_label"], "visible_primary": rec["visible_primary"],
                                "n_seen_primary": rec["n_seen_primary"], "image": str(png.relative_to(out)),
                                "by_eye_agrees": ""})
        s["n_review_images"] = len(chosen)
        summary["splits"][split] = s
    summary["scenes"] = scenes_summary(out / "scenes.jsonl")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if review_rows:
        (out / "review").mkdir(parents=True, exist_ok=True)
        idx_path = out / "review" / "review_index.csv"
        # Keep any by-eye answers already filled in (rerunning must not erase them).
        if idx_path.exists():
            with open(idx_path, "r", newline="", encoding="utf-8") as fh:
                old = {(r["split"], str(r["row_index"])): r.get("by_eye_agrees", "") for r in csv.DictReader(fh)}
            for r in review_rows:
                r["by_eye_agrees"] = old.get((r["split"], str(r["row_index"])), "") or ""
        with open(idx_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(review_rows[0].keys()))
            w.writeheader()
            w.writerows(review_rows)
    lines = []
    for split, s in summary["splits"].items():
        p = s["primary"]
        lines.append(f"{split}: selected {s['n_selected']}, done {s['n_done']}, not done {s['n_not_done']}, "
                     f"status {s['status']}")
        for tol in ("primary", "strict", "loose"):
            t = s[tol]
            lines.append(f"  tol {t['tolerance_m']} m: visible {t['visible']}, not visible {t['not_visible']}; "
                         f"within radius (path) visible {t['within_radius_path']['visible']}, "
                         f"not visible {t['within_radius_path']['not_visible']}; "
                         f"within radius (straight) visible {t['within_radius_straight']['visible']}, "
                         f"not visible {t['within_radius_straight']['not_visible']}")
        lines.append(f"  missing id or box: {s['missing']}")
        lines.append(f"  median render s per episode: {s['render_s_median']}")
        lines.append(f"  duplicate label ids: {s['n_duplicate_label_ids']}, ok rows hitting empty-depth pixels: "
                     f"{s['n_ok_with_invalid_depth_hits']}, review images: {s['n_review_images']}")
    lines.append(f"scenes: {summary['scenes']}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
