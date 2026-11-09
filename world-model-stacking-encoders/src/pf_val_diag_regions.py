"""Which image region carries the stored-vs-re-render feature gap? Region swap test, stove task only (pf-val diagnosis).

Frames: every 3rd val frame (demo-major), so about 280 frames. Image regions: a 4 x 4 grid of 32 x 32 px cells per camera (raw array
coordinates; the arrays are upside-down, the report gives upright coordinates). Two directions per cell and camera:
  repair  the re-rendered frame with the STORED pixels of that cell pasted in
  damage  the stored frame with the RE-RENDERED pixels of that cell pasted in
The other camera stays as it is in the base condition (re-render for repair, stored for damage). Each edited frame is encoded with DINOv3
(same code as pf_val_diag_features.py), and the combined probe (stove train demos, alpha 0.1, same as pf_val_diag_probe.py) is evaluated on
the subset. Output: regions-stove.json, regions-stove.png (heatmaps of the R2 change per cell).
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import diagnostics as D
import pf_val_diag_features as F

CAMS = F.CAMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stored", required=True)
    ap.add_argument("--rerender", required=True)
    ap.add_argument("--feat", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--step", type=int, default=3)
    args = ap.parse_args()
    out, feat = Path(args.out), Path(args.feat)
    allk, vald, S, R = F.load(args.stored, args.rerender)
    Sv = {c: np.concatenate([S[c][d][: len(R[c][d])] for d in vald]) for c in CAMS}
    Rv = {c: np.concatenate([R[c][d] for d in vald]) for c in CAMS}
    sa, rr0, st0 = np.load(feat / "stored_all.npz"), np.load(feat / "rr.npz"), np.load(feat / "st_val.npz")
    vdem = rr0["demo"]
    with h5py.File(args.stored, "r") as fs:
        acts = {d: fs["data"][f"demo_{d}"]["actions"][:].astype(np.float32) for d in sorted(set(sa["demo"].tolist()))}
    dem = sa["demo"]
    train = dem < 30
    y = np.concatenate([acts[d] for d in sorted(acts)])
    yv = np.concatenate([acts[d][: int((vdem == d).sum())] for d in sorted(set(vdem.tolist()))])
    x = np.concatenate([sa[c] for c in CAMS], axis=1)
    mu, sd = x[train].mean(0), x[train].std(0) + 1e-6
    w, b = D.ridge_fit(((x - mu) / sd)[train], y[train], 0.1)
    pred = lambda zz: ((np.concatenate([zz[c] for c in CAMS], axis=1) - mu) / sd) @ w + b
    sub = np.arange(len(yv))[:: args.step]
    ysub = yv[sub]
    enc, torch = F.make_encoder(args.device)

    def r2(z):
        return float(D.r2(ysub, pred(z)).mean())

    base = {"rr": r2({c: rr0[c][sub] for c in CAMS}), "stored": r2({c: st0[c][sub] for c in CAMS})}
    print("subset baselines", base, flush=True)
    G = 4
    res = {"base_r2_subset": base, "n_frames": int(len(sub)), "cells": {}}
    for ci, cam in enumerate(CAMS):
        for gy in range(G):
            for gx in range(G):
                sl = (slice(None), slice(gy * 32, (gy + 1) * 32), slice(gx * 32, (gx + 1) * 32))
                rep = Rv[cam][sub].copy(); rep[sl] = Sv[cam][sub][sl]                  # repair
                dmg = Sv[cam][sub].copy(); dmg[sl] = Rv[cam][sub][sl]                  # damage
                zr = {c: rr0[c][sub] for c in CAMS}; zr[cam] = F.encode(enc, torch, rep)
                zd = {c: st0[c][sub] for c in CAMS}; zd[cam] = F.encode(enc, torch, dmg)
                key = f"{cam}|row{gy}|col{gx}"
                res["cells"][key] = {"repair_r2": r2(zr), "repair_gain_over_rr": r2(zr) - base["rr"],
                                     "damage_r2": r2(zd), "damage_loss_from_stored": r2(zd) - base["stored"],
                                     "mean_abs_pixel_diff_in_cell": float(np.abs(Rv[cam][sub][sl].astype(np.float32) - Sv[cam][sub][sl].astype(np.float32)).mean())}
                print(key, {k: round(v, 4) for k, v in res["cells"][key].items()}, flush=True)
                (out / "regions-stove.json").write_text(json.dumps(res, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(8, 7))
    for ci, cam in enumerate(CAMS):
        for ri, (k, ttl) in enumerate((("repair_gain_over_rr", "gain from pasting stored cell into re-render"), ("damage_loss_from_stored", "loss from pasting re-render cell into stored"))):
            m = np.array([[res["cells"][f"{cam}|row{gy}|col{gx}"][k] for gx in range(G)] for gy in range(G)])
            m = np.rot90(m, 2)                                                           # upright
            im = ax[ri, ci].imshow(m, cmap="coolwarm", vmin=-np.abs(m).max(), vmax=np.abs(m).max())
            for (i, j), v in np.ndenumerate(m):
                ax[ri, ci].text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=8)
            ax[ri, ci].set_title(f"{cam}\n{ttl}", fontsize=8); ax[ri, ci].set_xticks([]); ax[ri, ci].set_yticks([])
    plt.tight_layout(); plt.savefig(out / "regions-stove.png", dpi=110)


if __name__ == "__main__":
    main()
