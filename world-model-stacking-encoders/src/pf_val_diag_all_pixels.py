"""Pixel-level comparison, stored vs re-rendered val frames, per task and camera, all ten libero_goal tasks (pf-val diagnosis, all-task part).
Val demos only (demo_30..39; the loader asserts it). Writes pixels-all.json, heatmaps-all.png (mean |diff| per task and camera, upright) and
sidebyside-all-<task>.png (agentview and wrist, stored | re-render | 4x |diff|, three moments) for every task."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from pf_val_diag_pixels import CAMS, load_pairs


def lap(x):
    g = x.astype(np.float32).mean(-1)
    return float(np.mean([(ndimage.laplace(f, mode="nearest") ** 2).mean() for f in g]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--rr", required=True, help="dir of re-rendered hdf5 (same file names)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    rep = {}
    files = sorted(Path(args.raw).glob("*.hdf5"))
    fig, ax = plt.subplots(len(files), 2, figsize=(7, 3.3 * len(files)))
    for ti, p in enumerate(files):
        P = load_pairs(p, Path(args.rr) / p.name)
        task = p.stem.replace("_demo", "")
        rep[task] = {}
        for ci, c in enumerate(CAMS):
            s, r = P[c]["s"], P[c]["r"]
            sf, rf = s.astype(np.float32), r.astype(np.float32)
            ab = np.abs(rf - sf)
            heat = ab.mean((0, 3))
            up = np.rot90(heat, 2)
            sub = slice(None, None, 4)
            fl = np.sort(heat.ravel())[::-1]
            t = P[c]["t"]
            tb = np.minimum((t * 10).astype(int), 9)
            fm = ab.mean((1, 2, 3))
            mask = up > np.quantile(up, 0.98)
            lab, nl = ndimage.label(mask)
            boxes = []
            for k in range(1, nl + 1):
                ys, xs = np.where(lab == k)
                if len(ys) >= 6:
                    boxes.append({"y0": int(ys.min()), "y1": int(ys.max()), "x0": int(xs.min()), "x1": int(xs.max()), "n": int(len(ys)), "mean_abs_diff": float(up[lab == k].mean())})
            rep[task][c] = {
                "n_frames": int(len(s)), "mean_shift_rgb": (rf - sf).mean((0, 1, 2)).round(3).tolist(),
                "brightness_stored": float(sf.mean()), "brightness_rerender": float(rf.mean()),
                "mean_abs_diff": float(ab.mean()), "frac_pixels_gt8": float((ab.mean(-1) > 8).mean()),
                "top5pct_positions_share_of_diff": float(fl[: int(0.05 * fl.size)].sum() / fl.sum()),
                "laplacian_ratio_rerender_over_stored": lap(r[sub]) / lap(s[sub]),
                "mad_by_demo_tenth": [float(fm[tb == b].mean()) for b in range(10)],
                "top2pct_regions_upright": sorted(boxes, key=lambda b: -b["n"] * b["mean_abs_diff"])[:4],
            }
            im = ax[ti, ci].imshow(up, cmap="magma", vmin=0, vmax=20)
            ax[ti, ci].set_title(f"{task[:34]} | {c[:9]} mean {ab.mean():.2f}", fontsize=7); ax[ti, ci].axis("off")
            print(task, c, round(float(ab.mean()), 3), flush=True)
        # side by side, 3 moments, both cameras
        rows = []
        for tt in (0.2, 0.5, 0.8):
            tiles = []
            for c in CAMS:
                idx = np.where(P[c]["demo"] == P[c]["demo"][len(P[c]["demo"]) // 2])[0]
                k = idx[int(tt * (len(idx) - 1))]
                a, b = np.rot90(P[c]["s"][k], 2), np.rot90(P[c]["r"][k], 2)
                dd = np.clip(np.abs(a.astype(np.int16) - b.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
                tiles += [a, b, dd]
            rows.append(np.concatenate(tiles, axis=1))
        Image.fromarray(np.concatenate(rows, 0)).resize((128 * 6 * 2, 128 * 3 * 2), Image.NEAREST).save(out / f"sidebyside-all-{task}.png")
    plt.tight_layout(); plt.savefig(out / "heatmaps-all.png", dpi=80)
    (out / "pixels-all.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
