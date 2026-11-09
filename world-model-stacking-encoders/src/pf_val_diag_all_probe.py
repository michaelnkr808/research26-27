"""All-task probe analysis, stored vs re-rendered val frames (pf-val diagnosis, all-task part).

Reproduces the Job's probe (pf_validate.probe): ridge from the two cameras' DINOv3 global features (standardized with train statistics) to the
7-d action, fitted on the stored TRAIN frames of all ten libero_goal tasks, scored on the val frames (n-1 per demo) of the stored features
(computed on the Mac, pf_val_diag_all_features.py) and of the re-rendered features (the Job's own, from the PVC). Then breaks the gap down
by task, camera, demo and time in a demo, at several alphas. Train and val demos only; test demos' actions and images are never read.
Writes probe-all.json and figures.
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
JOB = {"combined": (0.6561556494914705, 0.6052785138247245), "agentview": (0.5428944204730787, 0.4141972015339723), "wrist": (0.6264840654828115, 0.574600884097765)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--stored", required=True, help="stored_all_tasks dir")
    ap.add_argument("--rr", required=True, help="dir of the Job's re-rendered feature h5 (same file stems + .h5)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--pixels-json", default="")
    args = ap.parse_args()
    out = Path(args.out)
    tasks = [p.stem for p in sorted(Path(args.raw).glob("*.hdf5"))]
    X = {c: [] for c in CAMS}; XV = {c: [] for c in CAMS}; RR = {c: [] for c in CAMS}
    y_tr, y_va_full, y_va, meta = [], [], [], []       # meta rows for val (n-1): task idx, demo, j, n
    xtr = {c: [] for c in CAMS}
    xva_full = {c: [] for c in CAMS}
    for ti, stem in enumerate(tasks):
        with h5py.File(Path(args.raw) / f"{stem}.hdf5", "r") as f:
            names = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
            sp = D.demo_split(names)
            acts = {int(d.split("_")[-1]): f["data"][d]["actions"][:].astype(np.float32) for d in names if sp[d] != "test"}
        tr, va = np.load(Path(args.stored) / f"{stem}.train.npz"), np.load(Path(args.stored) / f"{stem}.val.npz")
        assert tr["demo"].max() < 30 and va["demo"].min() >= 30 and va["demo"].max() <= 39
        for c in CAMS:
            xtr[c].append(tr[c]); xva_full[c].append(va[c])
        y_tr.append(np.concatenate([acts[d][tr["j"][tr["demo"] == d]] for d in sorted(set(tr["demo"].tolist()))]))
        y_va_full.append(np.concatenate([acts[d][va["j"][va["demo"] == d]] for d in sorted(set(va["demo"].tolist()))]))
        with h5py.File(Path(args.rr) / f"{stem}.h5", "r") as fx:
            for d in sorted(set(va["demo"].tolist())):
                n = int((va["demo"] == d).sum())
                zr = {c: fx[f"demo_{d}"][c]["global"][:].astype(np.float32) for c in CAMS}
                assert len(zr[CAMS[0]]) == n - 1
                for c in CAMS:
                    RR[c].append(zr[c]); XV[c].append(va[c][va["demo"] == d][: n - 1])
                y_va.append(acts[d][: n - 1])
                meta += [(ti, d, j, n) for j in range(n - 1)]
    cat = lambda dct: {c: np.concatenate(v) for c, v in dct.items()}
    xtr, xva_full, XV, RR = cat(xtr), cat(xva_full), cat(XV), cat(RR)
    ytr, yvf, yv = np.concatenate(y_tr), np.concatenate(y_va_full), np.concatenate(y_va)
    meta = np.array(meta)
    tk, dm, jj, nn = meta.T
    rep = {"n_train": int(len(ytr)), "n_val_full": int(len(yvf)), "n_val": int(len(yv)), "tasks": tasks}

    def fitter(cols, alpha):
        xt = np.concatenate([xtr[c] for c in cols], axis=1)
        mu, sd = xt.mean(0), xt.std(0) + 1e-6
        w, b = D.ridge_fit((xt - mu) / sd, ytr, alpha)
        return lambda z: ((np.concatenate([z[c] for c in cols], axis=1) - mu) / sd) @ w + b

    # check 1: stored val R2 on all n val frames equals the tally curve value at alpha 0.1 (0.6538305228198683)
    pr = fitter(CAMS, 0.1)
    rep["check_stored_full_val_r2_alpha0.1"] = {"mine": float(D.r2(yvf, pr(xva_full)).mean()), "tally_curve": 0.6538305228198683}
    # check 2: Job numbers
    chk = {}
    for name, cols, a in (("combined", CAMS, 0.1), ("agentview", [CAMS[0]], 0.01), ("wrist", [CAMS[1]], 0.01)):
        pr = fitter(cols, a)
        s, r = float(D.r2(yv, pr(XV)).mean()), float(D.r2(yv, pr(RR)).mean())
        chk[name] = {"alpha": a, "mine_stored": s, "job_stored": JOB[name][0], "mine_rerender": r, "job_rerender": JOB[name][1],
                     "max_abs_diff_to_job": max(abs(s - JOB[name][0]), abs(r - JOB[name][1]))}
    rep["check_against_job_probe_report"] = chk
    ntask = len(tasks)
    # alpha sweep, combined and per camera
    rep["alpha_sweep"] = {}
    for name, cols in (("combined", CAMS), ("agentview", [CAMS[0]]), ("wrist", [CAMS[1]])):
        rep["alpha_sweep"][name] = {}
        for a in (0.01, 0.1, 1, 10, 100, 1000):
            pr = fitter(cols, a)
            s, r = float(D.r2(yv, pr(XV)).mean()), float(D.r2(yv, pr(RR)).mean())
            rep["alpha_sweep"][name][f"{a:g}"] = {"stored": s, "rerender": r, "gap": r - s}
            print(name, a, round(s, 4), round(r, 4), flush=True)
    # breakdown at alpha 0.1 and at the alpha best on stored val
    best = max(rep["alpha_sweep"]["combined"].items(), key=lambda kv: kv[1]["stored"])[0]
    rep["best_alpha_on_stored_val_combined"] = float(best)
    SST = ((yv - yv.mean(0)) ** 2).sum(0)

    def contrib(m, ps_, pr_):
        """This subset's additive part of the overall mean-R2 change (re-render minus stored); parts over a partition sum to the overall gap.
        (A per-subset R2 is not used: some action dimensions are constant within a task, which makes a within-task R2 blow up.)"""
        return float(-(((yv[m] - pr_[m]) ** 2 - (yv[m] - ps_[m]) ** 2).sum(0) / SST).mean())

    def relerr(m, ps_, pr_):
        return float((((yv[m] - pr_[m]) ** 2).sum(0) / SST).sum() / (((yv[m] - ps_[m]) ** 2).sum(0) / SST).sum() - 1)

    ug = [(t, d) for t in range(ntask) for d in range(30, 40)]
    for aname, a in (("alpha0.1", 0.1), ("alpha_best", float(best))):
        pr = fitter(CAMS, a)
        ps, prr = pr(XV), pr(RR)
        e_s, e_r = ((yv - ps) ** 2).sum(1), ((yv - prr) ** 2).sum(1)
        tot = float((e_r - e_s).sum())
        B = {}
        B["task"] = {}
        for t in range(ntask):
            m = tk == t
            B["task"][tasks[t]] = {"gap_contribution_to_overall": contrib(m, ps, prr), "relative_error_increase": relerr(m, ps, prr),
                                   "share_of_total_excess_sse": float((e_r - e_s)[m].sum() / tot),
                                   "share_of_val_frames": float(m.mean()), "n_demos_worse": int(sum((e_r - e_s)[(tk == t) & (dm == d)].sum() > 0 for d in range(30, 40)))}
        gaps_by_demo = {f"{tasks[t]}|demo_{d}": float((e_r - e_s)[(tk == t) & (dm == d)].sum() / tot) for t, d in ug}
        B["n_demos_rerender_worse_of_100"] = int(sum(v > 0 for v in gaps_by_demo.values()))
        B["demo_share_of_excess_top10"] = dict(sorted(gaps_by_demo.items(), key=lambda kv: -kv[1])[:10])
        B["demo_share_of_excess_max_min"] = [max(gaps_by_demo.values()), min(gaps_by_demo.values())]
        frac = jj / (nn - 2).clip(1)
        tb = np.minimum((frac * 10).astype(int), 9)
        B["per_tenth_share_of_excess"] = [float((e_r - e_s)[tb == b].sum() / tot) for b in range(10)]
        B["per_tenth_gap_contribution"] = [contrib(tb == b, ps, prr) for b in range(10)]
        B["per_tenth_relative_error_increase"] = [relerr(tb == b, ps, prr) for b in range(10)]
        mv = np.abs(prr - ps).sum(1)
        top = np.argsort(-mv)[: len(mv) // 10]
        B["top10pct_frames_by_prediction_shift_share_of_excess"] = float((e_r - e_s)[top].sum() / tot)
        # camera breakdown
        B["camera"] = {}
        for name, cols in (("agentview", [CAMS[0]]), ("wrist", [CAMS[1]])):
            p1 = fitter(cols, a)
            B["camera"][name] = {"stored": float(D.r2(yv, p1(XV)).mean()), "rerender": float(D.r2(yv, p1(RR)).mean())}
            q_s, q_r = p1(XV), p1(RR)
            B["camera"][name]["per_task_gap_contribution"] = {tasks[t]: contrib(tk == t, q_s, q_r) for t in range(ntask)}
            B["camera"][name]["per_task_relative_error_increase"] = {tasks[t]: relerr(tk == t, q_s, q_r) for t in range(ntask)}
        # oracle mean offset
        shift = {c: RR[c].mean(0) - XV[c].mean(0) for c in CAMS}
        pf = pr({c: RR[c] - shift[c] for c in CAMS})
        B["oracle_mean_shift_removed"] = {"r2": float(D.r2(yv, pf).mean()), "gap": float(D.r2(yv, pf).mean() - D.r2(yv, ps).mean())}
        # feature cosine per task and camera (standardized with train statistics)
        B["cosine_standardized"] = {}
        for c in CAMS:
            mu, sd = xtr[c].mean(0), xtr[c].std(0) + 1e-6
            u, v = (XV[c] - mu) / sd, (RR[c] - mu) / sd
            cs = (u * v).sum(1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1)
            B["cosine_standardized"][c] = {tasks[t]: float(cs[tk == t].mean()) for t in range(ntask)}
            B["cosine_standardized"][c]["all"] = float(cs.mean())
        # bootstrap CI on the overall gap and per-task gap (resample demos within the 100 (task, demo) groups)
        gid = tk * 100 + dm
        idx_of = {g: np.where(gid == g)[0] for g in np.unique(gid)}
        rng = np.random.default_rng(0)
        ugs = np.array(sorted(idx_of))
        draws, tdraws = [], {t: [] for t in range(ntask)}
        for _ in range(args.n_boot):
            pick = rng.choice(ugs, len(ugs), replace=True)
            ix = np.concatenate([idx_of[g] for g in pick])
            draws.append(float(D.r2(yv[ix], prr[ix]).mean() - D.r2(yv[ix], ps[ix]).mean()))
        B["overall_gap"] = {"value": float(D.r2(yv, prr).mean() - D.r2(yv, ps).mean()), "ci95_demo_bootstrap": np.percentile(draws, [2.5, 97.5]).tolist()}
        for t in range(ntask):
            gt = ugs[ugs // 100 == t]
            dr = []
            for _ in range(args.n_boot // 2):
                pick = rng.choice(gt, len(gt), replace=True)
                ix = np.concatenate([idx_of[g] for g in pick])
                dr.append(float(-(((yv[ix] - prr[ix]) ** 2 - (yv[ix] - ps[ix]) ** 2).sum(0) / SST).mean()))
            B["task"][tasks[t]]["gap_ci95"] = np.percentile(dr, [2.5, 97.5]).tolist()
        rep[aname] = B
        print(aname, "overall", B["overall_gap"], flush=True)
    if args.pixels_json and Path(args.pixels_json).exists():
        from scipy.stats import spearmanr
        px = json.loads(Path(args.pixels_json).read_text())
        g = [rep["alpha0.1"]["task"][t]["relative_error_increase"] for t in tasks]
        rep["spearman_task_gap_vs_pixel_mad"] = {c: float(spearmanr(g, [px[t.replace('_demo', '')][c]["mean_abs_diff"] for t in tasks]).statistic) for c in CAMS}
        rep["spearman_task_gap_vs_laplacian_ratio"] = {c: float(spearmanr(g, [px[t.replace('_demo', '')][c]["laplacian_ratio_rerender_over_stored"] for t in tasks]).statistic) for c in CAMS}
    (out / "probe-all.json").write_text(json.dumps(rep, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    B = rep["alpha0.1"]
    ax[0].barh([t.replace("_demo", "")[:30] for t in tasks], [B["task"][t]["gap_contribution_to_overall"] for t in tasks]); ax[0].set_title("each task's part of the overall R2 gap, alpha 0.1", fontsize=9)
    ax[1].plot(np.arange(10) / 10 + 0.05, B["per_tenth_share_of_excess"], "o-"); ax[1].set_title("share of excess error by tenth of demo", fontsize=9)
    for name in ("combined", "agentview", "wrist"):
        ks = list(rep["alpha_sweep"][name]); ax[2].plot(range(len(ks)), [rep["alpha_sweep"][name][k]["gap"] for k in ks], "o-", label=name)
    ax[2].set_xticks(range(len(ks))); ax[2].set_xticklabels(ks); ax[2].set_xlabel("alpha"); ax[2].set_title("gap vs alpha"); ax[2].legend()
    plt.tight_layout(); plt.savefig(out / "probe-all.png", dpi=100)


if __name__ == "__main__":
    main()
