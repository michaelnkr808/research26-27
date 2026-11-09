"""Pixel-level comparison of stored vs re-rendered val frames (experiments/2026-11-06-perturbed-frames, pf-val diagnosis).

Runs on the Mac, CPU only. Inputs: the stored LIBERO HDF5 of one task (only turn_on_the_stove is on the Mac) and the re-rendered HDF5 of
the same task from Job stacking-encoders-pf-val (stamp 20261108-181737). Val demos only (demo_30..demo_39); frame j of the re-render
is the render of states[j+1] and is compared with stored image j (PLAN amendment 2026-11-08), so n-1 frames per demo.

Writes results/pf-val-diagnosis/pixels-<task>.json plus PNGs (heatmaps, time curves, crops).
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage

CAMS = ["agentview_rgb", "eye_in_hand_rgb"]


def load_pairs(stored_path, rr_path):
    """{cam: (stored [N,128,128,3] uint8, rerender [N,...], demo_id [N], frame_idx [N], frac_time [N])} for the re-rendered demos."""
    out = {c: {"s": [], "r": [], "demo": [], "j": [], "t": []} for c in CAMS}
    with h5py.File(stored_path, "r") as fs, h5py.File(rr_path, "r") as fr:
        demos = sorted(fr["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        allk = sorted(fs["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        split_ok = all(allk.index(d) >= len(allk) - 20 and allk.index(d) < len(allk) - 10 for d in demos)   # val = demos 30..39 of 50
        assert split_ok, "re-rendered demos are not the val demos"
        for d in demos:
            for c in CAMS:
                r = fr["data"][d]["obs"][c][:]
                s = fs["data"][d]["obs"][c][: len(r)]
                n = len(r)
                out[c]["s"].append(s)
                out[c]["r"].append(r)
                out[c]["demo"] += [d] * n
                out[c]["j"] += list(range(n))
                out[c]["t"] += list(np.arange(n) / max(n - 1, 1))
    return {c: {k: (np.concatenate(v) if k in ("s", "r") else np.array(v)) for k, v in out[c].items()} for c in CAMS}


def hf_energy(x):
    """Mean squared Laplacian (high-frequency energy) per frame, on the grey image, 0-255 scale."""
    g = x.astype(np.float32).mean(-1)
    lap = ndimage.laplace(g, mode="nearest", axes=(1, 2)) if False else np.stack([ndimage.laplace(f, mode="nearest") for f in g])
    return (lap ** 2).mean((1, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stored", required=True)
    ap.add_argument("--rerender", required=True)
    ap.add_argument("--task", default="turn_on_the_stove")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P = load_pairs(args.stored, args.rerender)
    rep = {"task": args.task, "n_val_demos": len(set(P[CAMS[0]]["demo"])), "cams": {}}
    for c in CAMS:
        s = P[c]["s"].astype(np.float32)
        r = P[c]["r"].astype(np.float32)
        diff = r - s
        ab = np.abs(diff)
        info = {"n_frames": int(len(s))}
        info["mean_rgb_stored"] = s.mean((0, 1, 2)).round(3).tolist()
        info["mean_rgb_rerender"] = r.mean((0, 1, 2)).round(3).tolist()
        info["mean_shift_rgb_rerender_minus_stored"] = diff.mean((0, 1, 2)).round(3).tolist()
        info["brightness_stored"] = float(s.mean())
        info["brightness_rerender"] = float(r.mean())
        info["mean_abs_diff"] = float(ab.mean())
        info["median_frame_mad"] = float(np.median(ab.mean((1, 2, 3))))
        info["rms_diff"] = float(np.sqrt((diff ** 2).mean()))
        # share of pixels (per channel-mean) with abs diff above 8 and above 32 (0-255)
        pix = ab.mean(-1)
        info["frac_pixels_absdiff_gt8"] = float((pix > 8).mean())
        info["frac_pixels_absdiff_gt32"] = float((pix > 32).mean())
        # share of the total absolute difference contributed by the top 5% of pixels (by heatmap rank)
        heat = pix.mean(0)
        flat = np.sort(heat.ravel())[::-1]
        info["share_of_total_mad_in_top5pct_pixel_positions"] = float(flat[: int(0.05 * flat.size)].sum() / flat.sum())
        # high-frequency energy (Laplacian) stored vs re-render
        hs, hr = hf_energy(P[c]["s"]), hf_energy(P[c]["r"])
        info["laplacian_energy_stored_mean"] = float(hs.mean())
        info["laplacian_energy_rerender_mean"] = float(hr.mean())
        info["laplacian_energy_ratio_rerender_over_stored"] = float(hr.mean() / hs.mean())
        # does a small blur of both remove the difference? (separates fine texture from layout/colour differences)
        bl = lambda a, sg: np.stack([ndimage.gaussian_filter(f, sigma=(sg, sg, 0)) for f in a])
        sub = slice(None, None, 3)
        info["mad_after_blur_sigma"] = {str(sg): float(np.abs(bl(r[sub], sg) - bl(s[sub], sg)).mean()) for sg in (1.0, 2.0, 4.0)}
        # time trend inside demos: MAD by tenth of the demo, and per demo
        t = P[c]["t"]
        bins = np.minimum((t * 10).astype(int), 9)
        fm = ab.mean((1, 2, 3))
        info["mad_by_demo_tenth"] = [float(fm[bins == b].mean()) for b in range(10)]
        info["mad_per_demo"] = {d: float(fm[P[c]["demo"] == d].mean()) for d in sorted(set(P[c]["demo"]), key=lambda k: int(k.split("_")[-1]))}
        info["signed_mean_shift_by_demo_tenth"] = [float(diff[bins == b].mean()) for b in range(10)]
        # frame-to-frame jitter of the shift: correlation of per-frame MAD with the stored robot motion is not computed here
        # where in the image: peak boxes of the heatmap (top 2% of pixel positions, connected regions, upright image coordinates)
        up = np.rot90(heat, 2)                            # stored arrays are OpenGL (upside down); rotate 180 for display
        mask = up > np.quantile(up, 0.98)
        lab, nl = ndimage.label(mask)
        boxes = []
        for k in range(1, nl + 1):
            ys, xs = np.where(lab == k)
            if len(ys) >= 6:
                boxes.append({"y0": int(ys.min()), "y1": int(ys.max()), "x0": int(xs.min()), "x1": int(xs.max()), "n": int(len(ys)),
                              "mean_mad": float(up[lab == k].mean())})
        info["top2pct_heatmap_regions_upright_xy"] = sorted(boxes, key=lambda b: -b["n"] * b["mean_mad"])[:8]
        # a coarse 4 x 4 grid of mean abs diff (upright)
        info["mad_4x4_grid_upright"] = np.array([[up[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32].mean() for j in range(4)] for i in range(4)]).round(2).tolist()
        # signed mean diff heatmap summary
        sd = np.rot90(diff.mean(0).mean(-1), 2)
        info["signed_heat_min_max"] = [float(sd.min()), float(sd.max())]
        rep["cams"][c] = info
        # figures
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        im = ax[0].imshow(up, cmap="magma"); ax[0].set_title(f"{c}: mean |re-render - stored| (0-255)"); plt.colorbar(im, ax=ax[0], fraction=0.046)
        im = ax[1].imshow(sd, cmap="coolwarm", vmin=-np.abs(sd).max(), vmax=np.abs(sd).max()); ax[1].set_title("mean signed (re-render - stored)"); plt.colorbar(im, ax=ax[1], fraction=0.046)
        ax[2].plot(np.arange(10) / 10 + 0.05, info["mad_by_demo_tenth"], "o-"); ax[2].set_xlabel("fraction of demo"); ax[2].set_ylabel("mean abs diff"); ax[2].set_title("difference over time in a demo")
        for a in ax[:2]:
            a.axis("off")
        plt.tight_layout(); plt.savefig(out / f"heatmap-{args.task}-{c}.png", dpi=110); plt.close()
        # side-by-side crops at 5 moments (stored | re-render | |diff|*4), upright, upscaled 3x
        demos = sorted(set(P[c]["demo"]), key=lambda k: int(k.split("_")[-1]))
        rows = []
        for d, tt in zip(demos[::2][:5], (0.15, 0.35, 0.55, 0.75, 0.9)):
            idx = np.where(P[c]["demo"] == d)[0]
            k = idx[int(tt * (len(idx) - 1))]
            a, b = np.rot90(P[c]["s"][k], 2), np.rot90(P[c]["r"][k], 2)
            dd = np.clip(np.abs(a.astype(np.int16) - b.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
            rows.append(np.concatenate([a, b, dd], axis=1))
        from PIL import Image
        Image.fromarray(np.concatenate(rows, axis=0)).resize((384 * 3, 128 * 3 * 5), Image.NEAREST).save(out / f"sidebyside-{args.task}-{c}.png")
    (out / f"pixels-{args.task}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({c: {k: rep["cams"][c][k] for k in ("mean_shift_rgb_rerender_minus_stored", "mean_abs_diff", "laplacian_energy_ratio_rerender_over_stored", "mad_after_blur_sigma", "mad_by_demo_tenth")} for c in CAMS}, indent=1))


if __name__ == "__main__":
    main()
