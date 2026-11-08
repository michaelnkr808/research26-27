"""Overlay of the static difference masks of pf_val_diag_mask.py on one stored agentview frame (upright), stove task, pf-val diagnosis.
Green = top-3% mask inside the bowl box, red = top-3% mask outside the bowl box. Writes mask-overlay-stove.png and lists the connected
regions of the outside-box mask (upright x/y boxes and mean abs difference) in mask-regions-stove.json."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
import pf_val_diag_features as F
from pf_val_diag_pixels import load_pairs, CAMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stored", required=True)
    ap.add_argument("--rerender", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    P = load_pairs(args.stored, args.rerender)
    c = CAMS[0]
    heat = np.abs(P[c]["r"].astype(np.float32) - P[c]["s"].astype(np.float32)).mean((0, 3))
    allk, vald, S, R = F.load(args.stored, args.rerender)
    y0, y1, x0, x1 = F.bowl_box(S, R, vald)[0]
    box = np.zeros(heat.shape, bool); box[y0:y1, x0:x1] = True
    m3 = ndimage.binary_dilation(heat >= np.percentile(heat, 97), iterations=1)
    inb, outb = m3 & box, m3 & ~box
    from PIL import Image
    k = 40
    img = np.rot90(P[c]["s"][k], 2).astype(np.float32).copy()
    ri, ro = np.rot90(inb, 2), np.rot90(outb, 2)
    img[ri] = 0.5 * img[ri] + 0.5 * np.array([0, 255, 0]); img[ro] = 0.5 * img[ro] + 0.5 * np.array([255, 0, 0])
    Image.fromarray(img.astype(np.uint8)).resize((128 * 5, 128 * 5), Image.NEAREST).save(Path(args.out) / "mask-overlay-stove.png")
    lab, n = ndimage.label(ro, structure=np.ones((3, 3)))
    regs = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(ys) >= 8:
            regs.append({"y0": int(ys.min()), "y1": int(ys.max()), "x0": int(xs.min()), "x1": int(xs.max()), "n_pixels": int(len(ys)), "mean_abs_diff": float(np.rot90(heat, 2)[lab == i].mean())})
    (Path(args.out) / "mask-regions-stove.json").write_text(json.dumps({"outside_bowl_box_top3pct_regions_upright": sorted(regs, key=lambda r: -r["n_pixels"]),
                                                                    "bowl_box_upright_y0y1x0x1": [int(128 - y1), int(128 - y0), int(128 - x1), int(128 - x0)]}, indent=1))
    print(sorted(regs, key=lambda r: -r["n_pixels"])[:10])


if __name__ == "__main__":
    main()
