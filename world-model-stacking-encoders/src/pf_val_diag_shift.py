"""Is the re-render shifted by a fraction of a pixel against the stored frame? (pf-val diagnosis, stove task only)

For a grid of (dy, dx) shifts in pixels (cubic interpolation, edges repeated) the re-rendered frame is shifted and compared with the
stored frame of the same moment: mean absolute difference over every val frame (n-1 per demo), per camera. Writes shift-stove.json and a plot.
The same scan is also run on a copy of the stored frame shifted against itself as a reference for how fast the MAD rises with a shift.
Array axes are the raw (OpenGL, upside-down) HDF5 axes: dy is along array rows, dx along columns.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from pf_val_diag_pixels import CAMS, load_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stored", required=True)
    ap.add_argument("--rerender", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=4, help="use every step-th frame")
    args = ap.parse_args()
    P = load_pairs(args.stored, args.rerender)
    grid = np.arange(-1.0, 1.01, 0.25)
    rep = {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for ci, c in enumerate(CAMS):
        s = P[c]["s"][:: args.step].astype(np.float32)
        r = P[c]["r"][:: args.step].astype(np.float32)
        mad = np.zeros((len(grid), len(grid)))
        for a, dy in enumerate(grid):
            for b, dx in enumerate(grid):
                sh = np.stack([ndimage.shift(f, (dy, dx, 0), order=3, mode="nearest") for f in r])
                mad[a, b] = np.abs(sh - s).mean()
        a, b = np.unravel_index(mad.argmin(), mad.shape)
        # reference: stored frame vs itself shifted by the same amounts (shows the cost of a shift when images are otherwise identical)
        ref = np.array([np.abs(np.stack([ndimage.shift(f, (dy, 0, 0), order=3, mode="nearest") for f in s[:: 4]]) - s[:: 4]).mean() for dy in grid])
        rep[c] = {"grid_px": grid.tolist(), "mad_dy_by_dx": mad.round(4).tolist(), "best_dy_dx": [float(grid[a]), float(grid[b])],
                  "mad_at_zero_shift": float(mad[len(grid) // 2, len(grid) // 2]), "mad_at_best": float(mad[a, b]),
                  "reference_stored_vs_itself_shifted_in_dy_mad": ref.round(4).tolist(), "n_frames": int(len(s))}
        im = ax[ci].imshow(mad, extent=[grid[0] - .125, grid[-1] + .125, grid[-1] + .125, grid[0] - .125], cmap="viridis")
        ax[ci].set_xlabel("dx (px)"); ax[ci].set_ylabel("dy (px)"); ax[ci].set_title(f"{c}: MAD vs shift of re-render"); plt.colorbar(im, ax=ax[ci], fraction=0.046)
        print(c, "zero", rep[c]["mad_at_zero_shift"], "best", rep[c]["mad_at_best"], rep[c]["best_dy_dx"])
    plt.tight_layout(); plt.savefig(Path(args.out) / "shift-stove.png", dpi=110)
    (Path(args.out) / "shift-stove.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
