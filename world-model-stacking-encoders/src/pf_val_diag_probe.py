"""Feature-level and probe-level comparison of stored vs re-rendered val frames, stove task only (pf-val diagnosis).

Input: the npz files written by pf_val_diag_features.py (DINOv3 global features computed on the Mac) and the stored HDF5 (actions).
The probe is a ridge regression from the two cameras' global features (standardized with train statistics) to the 7-d action, the same
recipe as pf_validate.py / the tally (diagnostics.ridge_fit, D.r2), fitted on the stove task's stored TRAIN demos (demo_0..demo_29) only.
This is a single-task probe (about 2.7k training frames), not the tally's all-task probe, so absolute R2 values differ from the Job's.
Val demos are demo_30..demo_39, n-1 frames each. Test demos are never opened.

Writes results/pf-val-diagnosis/features-stove.json and figures.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import diagnostics as D

CAMS = D.CAMS
CONDS = ["st_val", "rr", "rr_colormatch", "rr_blur", "rr_sharpen", "rr_bowlpaste", "st_bowlpaste", "st_blur"]


def boot_ci(per_demo_stats, groups, fn, n_boot=1000, seed=0):
    """Demo-level bootstrap: resample demos with replacement; fn(index array of frames) -> scalar."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_of = {g: np.where(groups == g)[0] for g in ug}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(ug, len(ug), replace=True)
        out.append(fn(np.concatenate([idx_of[g] for g in pick])))
    return np.percentile(out, [2.5, 97.5]).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stored", required=True)
    ap.add_argument("--feat", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    feat, out = Path(args.feat), Path(args.out)
    sa = np.load(feat / "stored_all.npz")
    acts = {}
    with h5py.File(args.stored, "r") as fs:
        for d in sorted(set(sa["demo"].tolist())):
            acts[d] = fs["data"][f"demo_{d}"]["actions"][:].astype(np.float32)
    assert max(acts) == 39, "test demos (40..49) must not be present"
    Z = {c: sa[c] for c in CAMS}
    dem, jj = sa["demo"], sa["j"]
    y = np.concatenate([acts[d] for d in sorted(acts)])
    assert len(y) == len(dem)
    train = dem < 30
    cond = {n: np.load(feat / f"{n}.npz") for n in CONDS}
    vdem, vj = cond["rr"]["demo"], cond["rr"]["j"]
    yv = np.concatenate([acts[d][: int((vdem == d).sum())] for d in sorted(set(vdem.tolist()))])
    assert len(yv) == len(vdem)
    # stored val frames recomputed from stored_all must equal st_val (same images, same code)
    sel = np.concatenate([np.where((dem == d) & (jj < (vdem == d).sum()))[0] for d in sorted(set(vdem.tolist()))])
    chk = {c: float(np.abs(Z[c][sel] - cond["st_val"][c]).max()) for c in CAMS}
    rep = {"n_train_frames": int(train.sum()), "n_val_frames": int(len(yv)), "n_val_demos": int(len(set(vdem.tolist()))),
           "stored_all_vs_st_val_max_abs_feature_diff": chk, "probes": {}, "cosine": {}}

    def fit(cols, alpha):
        x = np.concatenate([Z[c] for c in cols], axis=1)
        mu, sd = x[train].mean(0), x[train].std(0) + 1e-6
        w, b = D.ridge_fit(((x - mu) / sd)[train], y[train], alpha)
        return lambda zz: ((np.concatenate([zz[c] for c in cols], axis=1) - mu) / sd) @ w + b

    # alpha: the tally's 0.1 (dinov3|global) and, as a sensitivity, the alpha with the best stored-val R2 over the tally grid
    for name, cols in (("combined", CAMS), ("agentview", [CAMS[0]]), ("wrist", [CAMS[1]])):
        alphas = {"0.1": 0.1}
        curve = {a: float(D.r2(yv, fit(cols, a)({c: cond["st_val"][c] for c in CAMS})).mean()) for a in D.ALPHAS}
        best = max(curve, key=curve.get)
        alphas[f"best_on_stored_val={best:g}"] = best
        rep["probes"][name] = {"stored_val_r2_by_alpha": {f"{a:g}": v for a, v in curve.items()}}
        for aname, a in alphas.items():
            pred = fit(cols, a)
            P = {n: pred({c: cond[n][c] for c in CAMS}) for n in CONDS}
            res = {}
            base = float(D.r2(yv, P["st_val"]).mean())
            for n in CONDS:
                r2m = float(D.r2(yv, P[n]).mean())
                gap = r2m - base
                ci = boot_ci(None, vdem, lambda ix, n=n: float(D.r2(yv[ix], P[n][ix]).mean() - D.r2(yv[ix], P["st_val"][ix]).mean()), args.n_boot)
                res[n] = {"val_r2": r2m, "minus_stored_val": gap, "gap_ci95_demo_bootstrap": ci,
                          "per_action_dim_r2": D.r2(yv, P[n]).round(4).tolist()}
            rep["probes"][name][aname] = res
            if name == "combined" and aname == "0.1":
                # where the gap sits: per demo, per tenth of the demo, per action dim (SSE excess of rr over stored, as share of val variance)
                ssv = ((yv - yv.mean(0)) ** 2).sum()
                e_st, e_rr = ((yv - P["st_val"]) ** 2).sum(1), ((yv - P["rr"]) ** 2).sum(1)
                tot = float((e_rr - e_st).sum())
                nfr = np.array([(vdem == d).sum() for d in sorted(set(vdem.tolist()))])
                rep["gap_location_combined_alpha0.1"] = {
                    "total_sse_excess_over_total_val_variance_sum": tot / float(ssv),
                    "per_demo_sse_excess_share_of_total": {int(d): float((e_rr - e_st)[vdem == d].sum() / tot) for d in sorted(set(vdem.tolist()))},
                    "per_demo_r2_stored": {int(d): float(D.r2(yv[vdem == d], P["st_val"][vdem == d]).mean()) for d in sorted(set(vdem.tolist()))},
                    "per_demo_r2_rerender": {int(d): float(D.r2(yv[vdem == d], P["rr"][vdem == d]).mean()) for d in sorted(set(vdem.tolist()))},
                    "per_tenth_of_demo_sse_excess_share_of_total": [],
                    "n_demos_where_rerender_worse_than_stored": int(sum((e_rr - e_st)[vdem == d].sum() > 0 for d in set(vdem.tolist()))),
                }
                frac = vj / np.array([(vdem == d).sum() - 1 for d in vdem])
                tb = np.minimum((frac * 10).astype(int), 9)
                rep["gap_location_combined_alpha0.1"]["per_tenth_of_demo_sse_excess_share_of_total"] = [float((e_rr - e_st)[tb == b].sum() / tot) for b in range(10)]
                # how much of the excess comes from frames whose predictions moved the most
                mv = np.abs(P["rr"] - P["st_val"]).sum(1)
                order = np.argsort(-mv)
                top = order[: len(order) // 10]
                rep["gap_location_combined_alpha0.1"]["top10pct_frames_by_prediction_shift_share_of_sse_excess"] = float((e_rr - e_st)[top].sum() / tot)
                # oracle bias test: subtract the mean feature shift (rr - stored, per camera, over val) from the re-rendered features
                shift = {c: cond["rr"][c].mean(0) - cond["st_val"][c].mean(0) for c in CAMS}
                fixed = {c: cond["rr"][c] - shift[c] for c in CAMS}
                Pfix = pred(fixed)
                rep["oracle_mean_shift_removed_combined_alpha0.1"] = {
                    "val_r2": float(D.r2(yv, Pfix).mean()), "minus_stored_val": float(D.r2(yv, Pfix).mean() - base),
                    "note": "oracle: the shift is estimated from the same val frames; diagnostic only, not a fix"}

    # per-frame cosine stored vs re-render, raw and train-standardized; compared with cosines inside the stored sequence
    for c in CAMS:
        zs, zr = cond["st_val"][c], cond["rr"][c]
        cs = lambda a, b: (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
        mu, sd = Z[c][train].mean(0), Z[c][train].std(0) + 1e-6
        cos_raw, cos_std = cs(zs, zr), cs((zs - mu) / sd, (zr - mu) / sd)
        # baselines: stored j vs stored j+1 and j+3 inside the same demo (natural change between neighbouring frames)
        nb = {}
        for lag in (1, 3):
            a, b = [], []
            for d in sorted(set(vdem.tolist())):
                ii = np.where(vdem == d)[0]
                a.append(cond["st_val"][c][ii[:-lag]]); b.append(cond["st_val"][c][ii[lag:]])
            a, b = np.concatenate(a), np.concatenate(b)
            nb[f"stored_j_vs_j+{lag}_raw"] = float(cs(a, b).mean())
            nb[f"stored_j_vs_j+{lag}_standardized"] = float(cs((a - mu) / sd, (b - mu) / sd).mean())
        frac = vj / np.array([(vdem == d).sum() - 1 for d in vdem])
        tb = np.minimum((frac * 10).astype(int), 9)
        shift = zr - zs
        mean_shift = shift.mean(0)
        rep["cosine"][c] = {
            "raw_mean": float(cos_raw.mean()), "raw_min": float(cos_raw.min()), "raw_p05": float(np.percentile(cos_raw, 5)),
            "standardized_mean": float(cos_std.mean()), "standardized_min": float(cos_std.min()), "standardized_p05": float(np.percentile(cos_std, 5)),
            "by_demo_standardized": {int(d): float(cos_std[vdem == d].mean()) for d in sorted(set(vdem.tolist()))},
            "by_tenth_standardized": [float(cos_std[tb == b].mean()) for b in range(10)],
            "baselines": nb,
            "share_of_squared_shift_that_is_the_mean_shift": float((mean_shift ** 2).sum() * len(shift) / (shift ** 2).sum()),
            "mean_shift_norm_over_mean_feature_norm": float(np.linalg.norm(mean_shift) / np.linalg.norm(zs, axis=1).mean()),
        }
        # edited conditions: cosine to stored (standardized) -- does the edit move the re-render toward the stored features?
        rep["cosine"][c]["standardized_cosine_to_stored_by_condition"] = {
            n: float(cs((cond[n][c] - mu) / sd, (zs - mu) / sd).mean()) for n in CONDS if n != "st_val"}
    (out / "features-stove.json").write_text(json.dumps(rep, indent=1))
    # figure: per-demo and per-tenth cosine
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for c in CAMS:
        ax[0].plot(range(30, 40), list(rep["cosine"][c]["by_demo_standardized"].values()), "o-", label=c)
        ax[1].plot(np.arange(10) / 10 + 0.05, rep["cosine"][c]["by_tenth_standardized"], "o-", label=c)
    ax[0].set_xlabel("val demo"); ax[1].set_xlabel("fraction of demo"); ax[0].set_ylabel("cosine, stored vs re-render (standardized)")
    ax[0].legend(); plt.tight_layout(); plt.savefig(out / "cosine-stove.png", dpi=110)
    print(json.dumps({"chk": chk, "combined_a0.1": {n: (v["val_r2"], v["minus_stored_val"], v["gap_ci95_demo_bootstrap"]) for n, v in rep["probes"]["combined"]["0.1"].items()},
                      "cos": {c: {k: rep["cosine"][c][k] for k in ("raw_mean", "standardized_mean", "baselines", "standardized_cosine_to_stored_by_condition")} for c in CAMS}}, indent=1))


if __name__ == "__main__":
    main()
