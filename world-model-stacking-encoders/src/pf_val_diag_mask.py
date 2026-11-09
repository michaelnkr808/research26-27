"""Static difference-mask paste test, stove task, agentview only (pf-val diagnosis).

The mask is the set of pixel positions where the mean |re-render - stored| over all val frames (agentview) is highest: top 3%, 6%, 12%
and 25% of positions (mask dilated by 1 px). Two directions: repair = re-render with the stored pixels pasted in on the mask;
damage = stored with the re-rendered pixels pasted in on the mask. All 843 val frames; the wrist camera keeps its base-condition
features (re-render for repair, stored for damage). Probe: stove-train combined probe, alpha 0.1, as in pf_val_diag_probe.py.
Output: mask-stove.json.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage

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
    args = ap.parse_args()
    out, feat = Path(args.out), Path(args.feat)
    allk, vald, S, R = F.load(args.stored, args.rerender)
    c0 = CAMS[0]
    Sv = np.concatenate([S[c0][d][: len(R[c0][d])] for d in vald])
    Rv = np.concatenate([R[c0][d] for d in vald])
    heat = np.abs(Rv.astype(np.float32) - Sv.astype(np.float32)).mean((0, 3))
    sa, rr0, st0 = np.load(feat / "stored_all.npz"), np.load(feat / "rr.npz"), np.load(feat / "st_val.npz")
    vdem = rr0["demo"]
    with h5py.File(args.stored, "r") as fs:
        acts = {d: fs["data"][f"demo_{d}"]["actions"][:].astype(np.float32) for d in sorted(set(sa["demo"].tolist()))}
    train = sa["demo"] < 30
    y = np.concatenate([acts[d] for d in sorted(acts)])
    yv = np.concatenate([acts[d][: int((vdem == d).sum())] for d in sorted(set(vdem.tolist()))])
    x = np.concatenate([sa[c] for c in CAMS], axis=1)
    mu, sd = x[train].mean(0), x[train].std(0) + 1e-6
    w, b = D.ridge_fit(((x - mu) / sd)[train], y[train], 0.1)
    r2 = lambda z: float(D.r2(yv, ((np.concatenate([z[c] for c in CAMS], axis=1) - mu) / sd) @ w + b).mean())
    enc, torch = F.make_encoder(args.device)
    res = {"base": {"stored": r2({c: st0[c] for c in CAMS}), "rr": r2({c: rr0[c] for c in CAMS})}, "masks": {}}
    box = np.zeros_like(heat, dtype=bool)                      # the bowl box of pf_val_diag_features.py (raw coordinates, y0 y1 x0 x1)
    y0, y1, x0, x1 = F.bowl_box(S, R, vald)[0]
    box[y0:y1, x0:x1] = True
    res["bowl_box_raw_y0y1x0x1"] = [int(v) for v in (y0, y1, x0, x1)]
    plan = []
    for pct in (3, 6, 12, 25):
        m = ndimage.binary_dilation(heat >= np.percentile(heat, 100 - pct), iterations=1)
        plan.append((str(pct), m))
    m3 = ndimage.binary_dilation(heat >= np.percentile(heat, 97), iterations=1)
    plan.append(("3_inside_bowl_box", m3 & box))
    plan.append(("3_outside_bowl_box", m3 & ~box))
    plan.append(("6_outside_bowl_box", ndimage.binary_dilation(heat >= np.percentile(heat, 94), iterations=1) & ~box))
    for name, m in plan:
        pct = name
        rep = Rv.copy(); rep[:, m] = Sv[:, m]
        dmg = Sv.copy(); dmg[:, m] = Rv[:, m]
        zr = {c: rr0[c] for c in CAMS}; zr[c0] = F.encode(enc, torch, rep)
        zd = {c: st0[c] for c in CAMS}; zd[c0] = F.encode(enc, torch, dmg)
        res["masks"][str(pct)] = {"mask_fraction_of_pixels": float(m.mean()), "repair_r2": r2(zr), "repair_gain_over_rr": r2(zr) - res["base"]["rr"],
                                  "damage_r2": r2(zd), "damage_loss_from_stored": r2(zd) - res["base"]["stored"],
                                  "share_of_total_abs_diff_inside_mask": float(heat[m].sum() / heat.sum())}
        print(pct, res["masks"][str(pct)], flush=True)
        (out / "mask-stove.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
