#!/usr/bin/env python3
"""Top-down picture per episode for the lead's own judgment (PLAN.md, item 6). No verdict is drawn.

On the occupancy map of the scene (occupancy.png, GN-Bench's pixel grid: GS x = left + col * scale,
GS y = top - row * scale), draws the listener path, the source (target box centre), and at every step a short
arrow to the listener's left or right: the side the interaural time difference says (cue side). Its colour says
whether that matches the true side: green match, red mismatch, grey not counted (occluded or near the median
plane), black no cue. Runs where the scene files are (the rented machine). Output is InteriorGS-derived: local
only, never in git.

  python draw_topdown.py --run-dir <render out> --poses-dir <export out> --scenes-root <InteriorGS> [--every 1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import step2lib as s2  # noqa: E402


def gs_to_pixel(xy, meta) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    col = (xy[:, 0] - float(meta["left"])) / float(meta["scale"])
    row = (float(meta["top"]) - xy[:, 1]) / float(meta["scale"])
    return np.stack([col, row], axis=1)


def colour(step: dict) -> str:
    if not step["counted"]:
        return "0.6"
    if step["cue_side"] == s2.NONE:
        return "k"
    return "tab:green" if step["correct"] else "tab:red"


def draw(ep_id: str, run_dir: Path, poses_dir: Path, scenes_root: Path, every: int = 1) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    p = json.loads((poses_dir / "poses" / f"{ep_id}.json").read_text(encoding="utf-8"))
    edir = run_dir / "episodes" / ep_id
    e = json.loads((edir / "episode.json").read_text(encoding="utf-8"))
    steps = [json.loads(l) for l in (edir / "steps.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    occ = np.array(Image.open(scenes_root / p["scene"] / "occupancy.png").convert("L"))
    meta = p["occupancy_meta"]
    eyes = np.array([f["gs_eye"][:2] for f in p["frames"]])
    pix = gs_to_pixel(eyes, meta)
    src = gs_to_pixel(e["source_gs"][:2], meta)[0]
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    ax.imshow(occ, cmap="gray", vmin=0, vmax=255)
    ax.plot(pix[:, 0], pix[:, 1], "-", color="tab:blue", lw=1, label="listener path")
    ax.plot(pix[0, 0], pix[0, 1], "o", color="tab:blue", ms=6, label="start")
    ax.plot(src[0], src[1], "*", color="gold", mec="k", ms=16, label="source (target box centre)")
    arrow_px = 0.25 / float(meta["scale"])
    for k, (f, s) in enumerate(zip(p["frames"], steps)):
        if k % every:
            continue
        _fw, left, _r, _u = s2.listener_axes(f["gs_forward"])
        if s["cue_side"] == s2.NONE:
            ax.plot(pix[k, 0], pix[k, 1], ".", color=colour(s), ms=4)
            continue
        d = s["cue_side"] * left[:2]
        # GS y grows up, image rows grow down
        ax.arrow(pix[k, 0], pix[k, 1], d[0] * arrow_px, -d[1] * arrow_px, color=colour(s), width=0.3,
                 head_width=2.0, length_includes_head=True)
    t = s2.tally(steps)
    ax.set_title(f"{ep_id}: arrows = cue side; green match, red mismatch, grey not counted\n"
                 f"counted {t['n_counted']}, correct {t['n_correct']}, occluded {t['n_occluded']} of {t['n_steps']}",
                 fontsize=9)
    ax.legend(loc="lower right", fontsize=7)
    pad = 40
    lo, hi = np.minimum(pix.min(0), src) - pad, np.maximum(pix.max(0), src) + pad
    ax.set_xlim(max(lo[0], 0), min(hi[0], occ.shape[1]))
    ax.set_ylim(min(hi[1], occ.shape[0]), max(lo[1], 0))
    (run_dir / "topdown").mkdir(exist_ok=True)
    out = run_dir / "topdown" / f"{ep_id}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--poses-dir", type=Path, required=True)
    ap.add_argument("--scenes-root", type=Path, required=True)
    ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args(argv)
    recs = [json.loads(l) for l in (a.run_dir / "episode_records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in recs:
        print(draw(r["episode_id"], a.run_dir, a.poses_dir, a.scenes_root, a.every))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
