"""DINOv3 global features on the Mac for the pf-val diagnosis (experiments/2026-11-06-perturbed-frames).

Why on the Mac: only the stove task's stored HDF5 is on the Mac, and the stored (clean) DINOv3 feature files are on a cluster volume that
no readable pod mounts. So the stored frames of the stove task are encoded here, with the repo's own encoder code (extract_features.py,
dinov3 loader: 224 px bicubic, rotate 180 first, CLS token = "global"), and the re-rendered val frames likewise, plus a few edited
versions of the re-rendered (or stored) val frames for the cheap cause tests. Only the stove task is covered.

Conditions (all val frames, n-1 per demo, frame j of the re-render pairs with stored image j):
  stored            stored frames (all 50 demos, all n frames; the first 30 demos are probe training data)
  rr                re-rendered frames, as the Job wrote them
  rr_colormatch     per-camera, per-channel mean/std of the re-render matched to the stored val frames
  rr_blur / rr_sharpen   gaussian blur (sigma 0.7) / unsharp mask (sigma 1.0, amount 1.0) of the re-render
  rr_bowlpaste      re-render with the stored pixels of the bowl box pasted in (agentview only; wrist unchanged)
  st_bowlpaste      stored with the re-rendered pixels of the bowl box pasted in (agentview only)
  st_blur           stored frames blurred with the sigma that makes their Laplacian energy equal to the re-render's (per camera)
Features are written to --out as one npz per condition, keys "<cam>" -> [N, 1024] float32, plus "demo" and "j".
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
CAMS = ["agentview_rgb", "eye_in_hand_rgb"]


def dnum(k):
    return int(k.split("_")[-1])


def load(stored_path, rr_path):
    with h5py.File(stored_path, "r") as fs, h5py.File(rr_path, "r") as fr:
        import diagnostics as D
        names = sorted(fs["data"].keys(), key=dnum)
        sp = D.demo_split(names)                                                   # the tally's split; test demos are never read
        allk = [d for d in names if sp[d] != "test"]
        vald = sorted(fr["data"].keys(), key=dnum)
        assert all(sp[d] == "val" for d in vald)
        S = {c: {d: fs["data"][d]["obs"][c][:] for d in allk} for c in CAMS}
        R = {c: {d: fr["data"][d]["obs"][c][:] for d in vald} for c in CAMS}
    return allk, vald, S, R


def lap_energy(x):
    g = x.astype(np.float32).mean(-1)
    return float(np.mean([(ndimage.laplace(f, mode="nearest") ** 2).mean() for f in g]))


def blur(x, sg):
    return np.clip(np.rint(np.stack([ndimage.gaussian_filter(f.astype(np.float32), sigma=(sg, sg, 0)) for f in x])), 0, 255).astype(np.uint8)


def sharpen(x, sg=1.0, amount=1.0):
    xf = x.astype(np.float32)
    b = np.stack([ndimage.gaussian_filter(f, sigma=(sg, sg, 0)) for f in xf])
    return np.clip(np.rint(xf + amount * (xf - b)), 0, 255).astype(np.uint8)


def colormatch(x, ref_mean, ref_std):
    xf = x.astype(np.float32)
    m, s = xf.mean((0, 1, 2)), xf.std((0, 1, 2))
    return np.clip(np.rint((xf - m) / s * ref_std + ref_mean), 0, 255).astype(np.uint8)


def bowl_box(S, R, vald, pad=3):
    """Box (raw array coordinates, y0 y1 x0 x1) around the strongest blob of the mean |re-render - stored| map in the agentview."""
    c = CAMS[0]
    heat = np.mean([np.abs(R[c][d].astype(np.float32) - S[c][d][: len(R[c][d])].astype(np.float32)).mean((0, 3)) for d in vald], axis=0)
    sm = ndimage.gaussian_filter(heat, 2.0)
    y, x = np.unravel_index(sm.argmax(), sm.shape)
    lab, _ = ndimage.label(sm > 0.35 * sm.max())
    ys, xs = np.where(lab == lab[y, x])
    return (max(ys.min() - pad, 0), min(ys.max() + pad + 1, 128), max(xs.min() - pad, 0), min(xs.max() + pad + 1, 128)), heat


def make_encoder(device):
    import torch
    import extract_features as E
    enc = E.ENCODERS["dinov3"](device, torch.float32)
    return enc, torch


def encode(enc, torch, frames_raw, bs=64):
    out = []
    for i in range(0, len(frames_raw), bs):
        fr = np.ascontiguousarray(frames_raw[i:i + bs][:, ::-1, ::-1])            # rotate 180, as extract_file does
        g, _ = enc.encode(fr)
        out.append(g.float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stored", required=True)
    ap.add_argument("--rerender", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--only", default="", help="comma list of conditions (default all)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    allk, vald, S, R = load(args.stored, args.rerender)
    enc, torch = make_encoder(args.device)
    info = {"device": args.device, "torch": torch.__version__, "hf_id": enc.hf_id}
    box, heat = bowl_box(S, R, vald)
    info["bowl_box_raw_y0y1x0x1"] = [int(v) for v in box]
    y0, y1, x0, x1 = box
    # per-camera reference statistics and the blur sigma for st_blur
    Sv = {c: np.concatenate([S[c][d][: len(R[c][d])] for d in vald]) for c in CAMS}
    Rv = {c: np.concatenate([R[c][d] for d in vald]) for c in CAMS}
    ref = {c: (Sv[c].astype(np.float32).mean((0, 1, 2)), Sv[c].astype(np.float32).std((0, 1, 2))) for c in CAMS}
    sub = slice(None, None, 6)
    sig = {}
    for c in CAMS:
        target = lap_energy(Rv[c][sub])
        best = min(np.arange(0.3, 1.61, 0.1), key=lambda s: abs(lap_energy(blur(Sv[c][sub], s)) - target))
        sig[c] = float(round(best, 2))
        info[f"laplacian_energy_{c}"] = {"stored": lap_energy(Sv[c][sub]), "rerender": target, "stored_blurred_at_sigma": sig[c],
                                       "stored_blurred": lap_energy(blur(Sv[c][sub], best))}

    def val_cond(name):
        """frames per camera, in the order demo-major over the val demos (n-1 each)."""
        f = {}
        for c in CAMS:
            if name == "rr":
                f[c] = Rv[c]
            elif name == "rr_colormatch":
                f[c] = colormatch(Rv[c], *ref[c])
            elif name == "rr_blur":
                f[c] = blur(Rv[c], 0.7)
            elif name == "rr_sharpen":
                f[c] = sharpen(Rv[c])
            elif name == "rr_bowlpaste":
                x = Rv[c].copy()
                if c == CAMS[0]:
                    x[:, y0:y1, x0:x1] = Sv[c][:, y0:y1, x0:x1]
                f[c] = x
            elif name == "st_bowlpaste":
                x = Sv[c].copy()
                if c == CAMS[0]:
                    x[:, y0:y1, x0:x1] = Rv[c][:, y0:y1, x0:x1]
                f[c] = x
            elif name == "st_blur":
                f[c] = blur(Sv[c], sig[c])
            elif name == "st_val":
                f[c] = Sv[c]
            else:
                raise KeyError(name)
        return f

    demo_ids = np.concatenate([[dnum(d)] * len(R[CAMS[0]][d]) for d in vald])
    js = np.concatenate([np.arange(len(R[CAMS[0]][d])) for d in vald])
    conds = ["rr", "st_val", "rr_colormatch", "rr_blur", "rr_sharpen", "rr_bowlpaste", "st_bowlpaste", "st_blur"]
    if args.only:
        conds = args.only.split(",")
    t0 = time.time()
    if not args.only or "stored_all" in conds:
        z = {}
        for c in CAMS:
            z[c] = encode(enc, torch, np.concatenate([S[c][d] for d in allk]))
        z["demo"] = np.concatenate([[dnum(d)] * len(S[CAMS[0]][d]) for d in allk])
        z["j"] = np.concatenate([np.arange(len(S[CAMS[0]][d])) for d in allk])
        np.savez(out / "stored_all.npz", **z)
        print(f"stored_all done {time.time() - t0:.0f}s", flush=True)
    for name in [c for c in conds if c != "stored_all"]:
        f = val_cond(name)
        z = {c: encode(enc, torch, f[c]) for c in CAMS}
        z["demo"], z["j"] = demo_ids, js
        np.savez(out / f"{name}.npz", **z)
        print(f"{name} done {time.time() - t0:.0f}s", flush=True)
        if name == "rr" and name in conds:
            # a few example images of the edited conditions (upright, agentview, val demo_30 frame 40) for the report
            pass
    # save a strip of examples (upright): stored | rr | rr_colormatch | rr_blur | rr_sharpen | rr_bowlpaste | st_blur, for agentview frame 40 of demo_30
    from PIL import Image
    idx = int(np.where((demo_ids == 30) & (js == 40))[0][0])
    tiles = []
    for name in ["st_val", "rr", "rr_colormatch", "rr_blur", "rr_sharpen", "rr_bowlpaste", "st_blur"]:
        f = val_cond(name)
        tiles.append(np.rot90(f[CAMS[0]][idx], 2))
    Image.fromarray(np.concatenate(tiles, axis=1)).resize((128 * 7 * 3, 128 * 3), Image.NEAREST).save(out / "example-strip-agentview.png")
    (out / "info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
