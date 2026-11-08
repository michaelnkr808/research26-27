"""Validation check of the perturbed-frame pipeline (PLAN Q2, tolerance 0.01; experiments/2026-11-06-perturbed-frames).

Question: does the DINOv3 action probe score the same on the stored clean val frames and on clean val frames re-rendered through
our pipeline (LIBERO-Plus env, fixture copy, state restore, render at 256, area shrink to 128)? Pass if |difference| <= 0.01.

Two subcommands, run by Job stacking-encoders-pf-val (never a test demo, anywhere):
  render   re-render the clean val demos of all 10 libero_goal files and write them in the input contract of the PLAN amendment
           (2026-11-08): LIBERO layout in a `libero_goal` folder, original file stems and demo names, raw OpenGL orientation, frame j =
           render of states[j+1] (so n-1 frames per demo), area shrink to 128 x 128.
  probe    fit the D probe on the clean stored train features (ridge, the tally's alpha for dinov3|global), then report val R2 on stored
           val frames and on the re-rendered val frames, both on the same n-1 frames per demo, per camera and combined.
Extraction of the re-rendered frames is done between the two by extract_features.py (the Job script), with metadata that must equal the
clean cache's (checked in `probe` with pf_analysis.check_meta; any mismatch fails the run).
"""

import argparse
import json
import time
import traceback
from pathlib import Path

import h5py
import numpy as np

import diagnostics as D
import pf_render as R
from pf_smoke import maybe_override_camera, sha256

CAMS = D.CAMS                         # agentview_rgb, eye_in_hand_rgb
CAM_OF = dict(zip(R.CAM_KEYS, CAMS))  # env obs key -> LIBERO HDF5 key
SUITE = "libero_goal"
TALLY_N_FRAMES = 63728                # tally README / JSON
TALLY_SPLIT = {"train": 38464, "val": 12416, "test": 12848}
TOL = 0.01


def raw_files(raw_dir):
    return sorted(Path(raw_dir).glob("*.hdf5"))


def kept_demos(raw, clean_feat_path):
    """The tally's demo list (diagnostics.load keeps demos present in every encoder file): here the dinov3 file's demos, and the check
    that nothing is dropped."""
    demos = sorted(raw["data"].keys(), key=lambda k: int(k.split("_")[-1]))
    with h5py.File(clean_feat_path, "r") as f:
        kept = [d for d in demos if d in f]
    return demos, kept


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------

def render(args):
    out_dir = Path(args.out) / SUITE
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = {"files": {}, "checks": {}, "failures": []}
    t0 = time.time()
    for raw_path in raw_files(args.raw):
        stem = raw_path.stem
        base = stem[:-5] if stem.endswith("_demo") else stem
        info = rep["files"][stem] = {}
        with h5py.File(raw_path, "r") as fr:
            demos, kept = kept_demos(fr, Path(args.feats) / "dinov3" / SUITE / (stem + ".h5"))
            info["n_demos_raw"], info["n_demos_kept"] = len(demos), len(kept)
            if kept != demos:
                rep["failures"].append(f"{stem}: tally demo list differs from the raw file's ({len(kept)} vs {len(demos)})")
                break
            split = D.demo_split(kept)
            val = [d for d in kept if split[d] == "val"]
            nontest = [d for d in kept if split[d] != "test"]
            assert val and all(split[d] == "val" for d in val)
            parsed = {d: R.parse_model_file(fr["data"][d].attrs["model_file"]) for d in nontest}   # test demos are not read
            fixtures, finfo = R.differing_bodies([parsed[d] for d in nontest])
            info.update(base=base, val_demos=val, fixtures=fixtures, n_train_val_compared=len(nontest))
            env = R.make_env(base, 256)
            try:
                with h5py.File(out_dir / (stem + ".hdf5"), "w") as fo:
                    fo.create_group("data")
                    for d in val:
                        g, p = fr["data"][d], parsed[d]
                        states = g["states"][:]
                        n = len(states)
                        if states.shape[1] != len(env.sim.get_state().flatten()):
                            rep["failures"].append(f"{stem}/{d}: state length {states.shape[1]} != env {len(env.sim.get_state().flatten())}")
                            raise RuntimeError("state length")
                        env.reset()
                        ok, det = R.check_env_fixtures_before_copy(env, p, fixtures)          # Q20, before any copy
                        if not ok:
                            rep["failures"].append(f"{stem}/{d}: Q20 env fixtures differ from the demo: {json.dumps(det, default=str)[:600]}")
                            raise RuntimeError("Q20")
                        cam = maybe_override_camera(env, p, "validate")
                        info.setdefault("camera_override_used_in", [])
                        if cam["override_used"]:
                            info["camera_override_used_in"].append(d)
                        R.copy_fixtures(env, p, fixtures)
                        imgs = {c: np.zeros((n - 1, 128, 128, 3), np.uint8) for c in CAMS}
                        for j in range(n - 1):
                            obs = R.restore(env, states[j + 1])
                            for k, im in R.images(obs).items():
                                imgs[CAM_OF[k]][j] = R.shrink(im, "area")
                            if j % 10 == 0 or j == n - 2:
                                okf, detf = R.check_fixtures_after(env, p, fixtures)
                                if not okf:
                                    rep["failures"].append(f"{stem}/{d} j={j}: fixtures after restore differ: {json.dumps(detf)[:400]}")
                                    raise RuntimeError("fixtures")
                        dg = fo["data"].create_group(d)
                        ob = dg.create_group("obs")
                        for c in CAMS:
                            ob.create_dataset(c, data=imgs[c])
                        info.setdefault("frames", {})[d] = n - 1
                        print(f"[render] {stem} {d}: {n - 1} frames", flush=True)
            except RuntimeError:
                pass                                  # the failure text is already in rep["failures"]
            finally:
                env.close()
        if rep["failures"]:
            break
    rep["seconds"] = round(time.time() - t0, 1)
    rep["checks"]["render_all_files"] = {"pass": not rep["failures"] and len(rep["files"]) == 10, "n_files": len(rep["files"])}
    Path(args.out).joinpath("render-report.json").write_text(json.dumps(rep, indent=1, default=str))
    print("[render] done", json.dumps(rep["checks"]), rep["failures"][:3], flush=True)
    return 0 if rep["checks"]["render_all_files"]["pass"] else 3


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------

def probe(args):
    import pf_analysis as PA
    rep = {"checks": {}, "provenance": {}}
    alphas, r2m, sizes = PA.read_tally(args.tally_dir, SUITE, "global")
    alpha = alphas["dinov3|global"]
    rep["tally"] = {"alpha_dinov3_global": alpha, "files": [s[0] for s in sizes], "n_frames": sizes[0][1], "split_frames": sizes[0][2]}
    tally_val_curve = json.loads(next(Path(args.tally_dir).glob("tally-*.json")).read_text())["probes"]["dinov3|global"]["val_r2_by_alpha"]
    xs = {c: [] for c in CAMS}
    rr = {c: [] for c in CAMS}               # re-rendered val frames (n-1 per demo)
    st1 = {c: [] for c in CAMS}              # stored val frames, the same n-1
    y_val1, lag = [], []
    ytrva, sptrva = [], []
    metas = {}
    n_train = n_val = 0
    for raw_path in raw_files(args.raw):
        stem = raw_path.stem
        with h5py.File(raw_path, "r") as fr, h5py.File(Path(args.feats) / "dinov3" / SUITE / (stem + ".h5"), "r") as fc, \
                h5py.File(Path(args.rr_feats) / "dinov3" / SUITE / (stem + ".h5"), "r") as fx:
            demos, kept = kept_demos(fr, Path(args.feats) / "dinov3" / SUITE / (stem + ".h5"))
            assert kept == demos, (stem, "tally demo list != raw demo list")
            clean_meta = json.loads(fc.attrs["meta"])
            PA.check_meta(fx, clean_meta, f"{stem} re-rendered features")             # fails on any mismatch
            metas[stem] = {k: clean_meta[k] for k in PA.META_KEYS}
            split = D.demo_split(kept)
            val = [d for d in kept if split[d] == "val"]
            assert sorted(fx.keys()) == sorted(val), (stem, "re-rendered feature demos != val demos", sorted(fx.keys()), val)
            for d in kept:
                if split[d] == "test":
                    continue                                                           # test demos are never opened
                a = fr["data"][d]["actions"][:]
                n = len(a)
                zc = {c: fc[d][c]["global"][:].astype(np.float32) for c in CAMS}
                for c in CAMS:
                    assert len(zc[c]) == n
                    xs[c].append(zc[c])
                ytrva.append(a)
                sptrva += [split[d]] * n
                if split[d] == "train":
                    n_train += n
                else:
                    n_val += n
                    zr = {c: fx[d][c]["global"][:].astype(np.float32) for c in CAMS}
                    for c in CAMS:
                        assert len(zr[c]) == n - 1, (stem, d, len(zr[c]), n)
                        rr[c].append(zr[c])
                        st1[c].append(zc[c][: n - 1])
                    y_val1.append(a[: n - 1])
                    lag.append((stem, d, {c: (zr[c], zc[c]) for c in CAMS}))
    sp = np.array(sptrva)
    y = np.concatenate(ytrva).astype(np.float32)
    x = {c: np.concatenate(xs[c]) for c in CAMS}
    train = sp == "train"
    rep["checks"]["tally_frame_counts_train_val"] = {"pass": n_train == TALLY_SPLIT["train"] and n_val == TALLY_SPLIT["val"],
                                                    "train": n_train, "val": n_val, "tally": TALLY_SPLIT}
    rep["checks"]["meta_equals_clean_cache"] = {"pass": True, "keys": PA.META_KEYS, "meta": next(iter(metas.values()))}
    yv1 = np.concatenate(y_val1).astype(np.float32)
    zst = {c: np.concatenate(st1[c]) for c in CAMS}
    zrr = {c: np.concatenate(rr[c]) for c in CAMS}
    for c in CAMS:
        assert np.isfinite(zrr[c]).all() and np.isfinite(zst[c]).all()

    def fit_eval(cols, alpha_):
        cat = lambda dct: np.concatenate([dct[c] for c in cols], axis=1)
        xtv = cat(x)
        mu, sd = xtv[train].mean(0), xtv[train].std(0) + 1e-6           # D.standardize with train statistics
        w, b = D.ridge_fit(((xtv - mu) / sd)[train], y[train], alpha_)
        f = lambda z: D.r2(yv1, ((cat(z) - mu) / sd) @ w + b)
        return f(zst), f(zrr)

    res = {}
    for name, cols in (("combined", CAMS), ("agentview", [CAMS[0]]), ("wrist", [CAMS[1]])):
        if name == "combined":
            a_use, how = alpha, "tally alpha for dinov3|global"
        else:                                                              # single-camera input: no stored alpha -> tuned on clean val (amendment)
            xtv = np.concatenate([x[c] for c in cols], axis=1)
            xz = (xtv - xtv[train].mean(0)) / (xtv[train].std(0) + 1e-6)
            a_use, _ = PA.tune_alpha(xz, y, sp)
            how = "tuned on clean val, tally grid (PLAN amendment 2026-11-08)"
        r_st, r_rr = fit_eval(cols, a_use)
        res[name] = {"alpha": a_use, "alpha_source": how, "val_r2_stored": float(r_st.mean()), "val_r2_rerender": float(r_rr.mean()),
                     "difference_rerender_minus_stored": float(r_rr.mean() - r_st.mean()),
                     "abs_difference": float(abs(r_rr.mean() - r_st.mean())), "per_dim_stored": r_st.tolist(), "per_dim_rerender": r_rr.tolist()}
    rep["val_r2"] = res
    # consistency with the tally: val R2 at alpha on ALL stored val frames equals the tally's val curve value (up to fp16 / numerics)
    xall = np.concatenate([x[c] for c in CAMS], axis=1)
    xz = (xall - xall[train].mean(0)) / (xall[train].std(0) + 1e-6)
    w, b = D.ridge_fit(xz[train], y[train], alpha)
    va = sp == "val"
    mine = float(D.r2(y[va], xz[va] @ w + b).mean())
    theirs = float(tally_val_curve[str(alpha)] if str(alpha) in tally_val_curve else tally_val_curve[str(float(alpha))])
    rep["checks"]["stored_val_r2_matches_tally_curve"] = {"pass": abs(mine - theirs) < 1e-3, "gating": False, "mine": mine, "tally": theirs,
                                                          "alpha": alpha}
    # lag check on standardized features (pf_analysis.check_lag): re-render frame j closest to stored frame j
    lag_ok, lag_err = True, []
    stats = {c: (x[c][train].mean(0), x[c][train].std(0) + 1e-6) for c in CAMS}
    for stem, d, zz in lag:
        for c in CAMS:
            mu, sd = stats[c]
            try:
                PA.check_lag((zz[c][0] - mu) / sd, (zz[c][1] - mu) / sd, f"{stem}/{d}/{c}")
            except ValueError as e:
                lag_ok = False
                lag_err.append(str(e))
    rep["checks"]["lag_check_all_val_demos"] = {"pass": lag_ok, "n_demo_cams": len(lag) * 2, "errors": lag_err[:5]}
    gate = res["combined"]["abs_difference"] <= TOL
    rep["checks"]["validation_gate_combined_within_0.01"] = {"pass": bool(gate), "abs_difference": res["combined"]["abs_difference"], "tolerance": TOL}
    rep["gate_per_camera_within_0.01 (reported, not the gate)"] = {k: bool(res[k]["abs_difference"] <= TOL) for k in res}
    rep["n_val_frames_each"] = int(len(yv1))
    rep["all_checks_passed"] = all(c["pass"] for c in rep["checks"].values() if c.get("gating", True))
    Path(args.out).joinpath("probe-report.json").write_text(json.dumps(rep, indent=1, default=str))
    print("[probe] SUMMARY", json.dumps({"combined": {k: res["combined"][k] for k in ("val_r2_stored", "val_r2_rerender", "abs_difference", "alpha")},
                                         "agentview": {k: res["agentview"][k] for k in ("val_r2_stored", "val_r2_rerender", "abs_difference")},
                                         "wrist": {k: res["wrist"][k] for k in ("val_r2_stored", "val_r2_rerender", "abs_difference")},
                                         "checks": {k: v["pass"] for k, v in rep["checks"].items()}}), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["render", "probe"])
    ap.add_argument("--raw", required=True, help="dir with the 10 libero_goal HDF5 files")
    ap.add_argument("--feats", default="/mnt/features/stacking-encoders/features", help="clean feature cache root")
    ap.add_argument("--rr-feats", default="/work/feats_rr", help="re-rendered feature root (probe)")
    ap.add_argument("--tally-dir", default="/pf")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    try:
        return {"render": render, "probe": probe}[args.cmd](args)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        Path(args.out).joinpath(f"{args.cmd}-crash.txt").write_text(tb)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
