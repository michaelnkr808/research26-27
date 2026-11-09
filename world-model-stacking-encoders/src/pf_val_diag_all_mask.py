"""Static-mask paste test per task on the all-task probe (pf-val diagnosis, all-task part).

Per task and per camera, the mask is the 3 % of pixel positions with the largest mean |re-render - stored| over that task's val frames
(dilated by 1 px). Repair: the re-rendered val frame with the STORED pixels pasted on the mask, for one camera at a time (the other camera
keeps the Job's re-rendered features). Control: the same number of pixels at random positions (fixed seed), agentview only. Every 3rd val
frame is used (compute budget); R2 of the all-task probe (alpha 0.1, train = stored train frames of all tasks) is reported on that subset for
stored, re-render and each repaired condition, per task and pooled over tasks. Val demos only; stored features from pf_val_diag_all_features.py.
Writes mask-all.json.
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
    ap.add_argument("--raw", required=True)
    ap.add_argument("--rr-hdf5", required=True)
    ap.add_argument("--rr-feats", required=True)
    ap.add_argument("--stored", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    out = Path(args.out)
    tasks = [p.stem for p in sorted(Path(args.raw).glob("*.hdf5"))]
    xtr = {c: [] for c in CAMS}
    ytr = []
    for stem in tasks:
        with h5py.File(Path(args.raw) / f"{stem}.hdf5", "r") as f:
            names = sorted(f["data"].keys(), key=F.dnum)
            sp = D.demo_split(names)
            acts = {F.dnum(d): f["data"][d]["actions"][:].astype(np.float32) for d in names if sp[d] != "test"}
        tr = np.load(Path(args.stored) / f"{stem}.train.npz")
        for c in CAMS:
            xtr[c].append(tr[c])
        ytr.append(np.concatenate([acts[d][tr["j"][tr["demo"] == d]] for d in sorted(set(tr["demo"].tolist()))]))
    xt = np.concatenate([np.concatenate(xtr[c]) for c in CAMS], axis=1)
    ytr = np.concatenate(ytr)
    mu, sd = xt.mean(0), xt.std(0) + 1e-6
    w, b = D.ridge_fit((xt - mu) / sd, ytr, 0.1)
    pred = lambda z: ((np.concatenate([z[c] for c in CAMS], axis=1) - mu) / sd) @ w + b
    enc, torch = F.make_encoder(args.device)
    res = {"alpha": 0.1, "step": args.step, "tasks": {}}
    pool = {k: [] for k in ("y", "stored", "rr", "rep_agent", "rep_wrist", "rnd_agent")}
    for stem in tasks:
        allk, vald, S, R = F.load(Path(args.raw) / f"{stem}.hdf5", Path(args.rr_hdf5) / f"{stem}.hdf5")
        Sv = {c: np.concatenate([S[c][d][: len(R[c][d])] for d in vald]) for c in CAMS}
        Rv = {c: np.concatenate([R[c][d] for d in vald]) for c in CAMS}
        va = np.load(Path(args.stored) / f"{stem}.val.npz")
        with h5py.File(Path(args.rr_feats) / f"{stem}.h5", "r") as fx:
            zr = {c: np.concatenate([fx[f"demo_{d}"][c]["global"][:].astype(np.float32) for d in range(30, 40)]) for c in CAMS}
        keep = np.concatenate([np.where((va["demo"] == d) & (va["j"] < (len(S[CAMS[0]][f"demo_{d}"]) - 1)))[0] for d in range(30, 40)])
        zs = {c: va[c][keep] for c in CAMS}
        with h5py.File(Path(args.raw) / f"{stem}.hdf5", "r") as f:
            yv = np.concatenate([f["data"][f"demo_{d}"]["actions"][: len(R[CAMS[0]][f"demo_{d}"])].astype(np.float32) for d in range(30, 40)])
        sub = np.arange(len(yv))[:: args.step]
        r = {"n_frames": int(len(sub))}
        z0 = {"stored": {c: zs[c][sub] for c in CAMS}, "rr": {c: zr[c][sub] for c in CAMS}}
        outp = {k: pred(v) for k, v in z0.items()}
        rng = np.random.default_rng(0)
        for ci, c in enumerate(CAMS):
            heat = np.abs(Rv[c].astype(np.float32) - Sv[c].astype(np.float32)).mean((0, 3))
            m = ndimage.binary_dilation(heat >= np.percentile(heat, 97), iterations=1)
            rep = Rv[c][sub].copy(); rep[:, m] = Sv[c][sub][:, m]
            z = dict(z0["rr"]); z[c] = F.encode(enc, torch, rep)
            key = "rep_agent" if ci == 0 else "rep_wrist"
            outp[key] = pred(z)
            r[f"mask_fraction_{c}"] = float(m.mean())
            if ci == 0:
                mr = np.zeros(heat.shape, bool)
                mr.ravel()[rng.choice(mr.size, int(m.sum()), replace=False)] = True
                rep2 = Rv[c][sub].copy(); rep2[:, mr] = Sv[c][sub][:, mr]
                z2 = dict(z0["rr"]); z2[c] = F.encode(enc, torch, rep2)
                outp["rnd_agent"] = pred(z2)
        ys = yv[sub]
        for k, v in outp.items():
            r[f"r2_{k}"] = float(D.r2(ys, v).mean())
        r["gap_rr"] = r["r2_rr"] - r["r2_stored"]
        for k in ("rep_agent", "rep_wrist", "rnd_agent"):
            r[f"gain_{k}"] = r[f"r2_{k}"] - r["r2_rr"]
        res["tasks"][stem] = r
        pool["y"].append(ys)
        for k, v in outp.items():
            pool[k].append(v)
        print(stem, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
        (out / "mask-all.json").write_text(json.dumps(res, indent=1))
    y = np.concatenate(pool["y"])
    P = {k: np.concatenate(v) for k, v in pool.items() if k != "y"}
    res["pooled"] = {f"r2_{k}": float(D.r2(y, v).mean()) for k, v in P.items()}
    res["pooled"]["gap_rr"] = res["pooled"]["r2_rr"] - res["pooled"]["r2_stored"]
    for k in ("rep_agent", "rep_wrist", "rnd_agent"):
        res["pooled"][f"gain_{k}"] = res["pooled"][f"r2_{k}"] - res["pooled"]["r2_rr"]
    (out / "mask-all.json").write_text(json.dumps(res, indent=1))
    print(res["pooled"])


if __name__ == "__main__":
    main()
