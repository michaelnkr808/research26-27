"""Target-side diagnostics (H-E) on cached frozen-encoder features: encoders as prediction *targets*.

Binding spec: pre-registration in LOG 2026-10-21 (i) as amended by (j); task list in
experiments/2026-10-21-target-diagnostics/PLAN.md. In liu2026what, encoder features are both a clean current input and a
noised future target of the WAM, with one loss term per source (App. E.1 Eq. 9). The feasibility tally only tested the
input role. For base D and partner B (both cams concatenated per encoder), at horizons k, on frames t with t + k inside
the same demo (never shifted across demos), with target Delta_E(t; k) = f_E(t + k) - f_E(t) at raw scale:
  - T0 (no fitting, train frames): per-channel variance and mean L2 norm of f per cam, per-channel variance of Delta,
    and B's share of the loss on the target [Delta_D, Delta_B] under each scaling.
  - T1a: ridge X = [D_t, B_t] -> Delta_B. T1b: D_t -> Delta_D and X -> Delta_D (input-side gain of B for D's future).
  - T2 (primary): reduced-rank ridge X -> target with per-block weights w (block scaled by sqrt(w)): ridge fit, SVD of
    the fitted values, keep the top-r right singular vectors. Scored on D's dims on test.
    interference(B) = R^2_D(target Delta_D only) - R^2_D(target [Delta_D, Delta_B]); positive = B as a co-target hurts.
    Scalings: a_prime (primary) w = 1/d_block (per-source mean MSE, Eq. 9 if L_FM averages channels);
    b w = 1/(total train variance of the block) (equalized); a_dprime w = 1 (pooled, if L_FM sums channels).
    alpha per condition and rank: minimizes that condition's own weighted val MSE of the rank-r model.
Primary number: interference under a_prime, r = 64, global features, averaged over k in {4, 16, 32}.

Ridge uses the formula of diagnostics.ridge_fit (w = (X'X + alpha n I)^-1 X'(y - ybar), b = ybar) through one
eigendecomposition of X'X per fit set, so every alpha, target and scaling costs a projection rather than a new solve.
float64 throughout the linear algebra; features are float32 in memory and cast chunk by chunk.

CIs: demo-level bootstrap from diagnostics.bootstrap_r2 (1,000 paired draws, seed 0). Every k has the same test demos
(asserted), so the draws are identical across k and k-averages are taken per draw.

Example:
  python target_diagnostics.py --feats /work/feats --suite libero_goal --raw /work/data/libero_goal \
      --base dinov3 --partner c-radiov4 --out target-libero_goal-dinov3-c-radiov4.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import diagnostics as D

KS = [4, 16, 32]
RANKS = [64, 16, 256]  # first is primary
SCALINGS = ["a_prime", "b", "a_dprime"]  # first is primary
SCALING_DOC = {"a_prime": "w = 1/d_block (per-source mean MSE; primary)",
               "b": "w = 1/(total train variance of the block) (per-source equalized)",
               "a_dprime": "w = 1 (pooled channels)"}
CHUNK = 4096


# ---------------------------------------------------------------------------
# Within-demo shifts and targets
# ---------------------------------------------------------------------------

def shift_rows(index, k):
    """Global frame indices t (and t + k) with t + k inside the same demo. index: (file, demo, length) per demo."""
    rows, start = [], 0
    for _, _, n in index:
        rows.append(np.arange(start, start + max(0, n - k)))
        start += n
    t = np.concatenate(rows).astype(np.int64)
    return t, t + k


def delta_fn(blocks, k):
    """Target function rows -> [f(t + k) - f(t) for f in blocks], float64."""
    return lambda r: np.concatenate([f[r + k].astype(np.float64) - f[r] for f in blocks], 1)


def col_moments(y_fn, rows):
    """Per-column mean and variance of y_fn over rows (two passes, float64)."""
    s = sum(y_fn(rows[i:i + CHUNK]).sum(0) for i in range(0, len(rows), CHUNK))
    mu = s / len(rows)
    ss = sum(((y_fn(rows[i:i + CHUNK]) - mu) ** 2).sum(0) for i in range(0, len(rows), CHUNK))
    return mu, ss / len(rows)


def materialize(y_fn, rows):
    return np.concatenate([y_fn(rows[i:i + CHUNK]) for i in range(0, len(rows), CHUNK)])


# ---------------------------------------------------------------------------
# Ridge and reduced-rank ridge
# ---------------------------------------------------------------------------

class Fit:
    """Eigendecomposition of X'X on one set of rows. Ridge for any alpha and target is then a projection."""

    def __init__(self, x, rows):
        d = x.shape[1]
        g, sx = np.zeros((d, d)), np.zeros(d)
        for i in range(0, len(rows), CHUNK):
            c = x[rows[i:i + CHUNK]].astype(np.float64)
            g += c.T @ c
            sx += c.sum(0)
        s, self.q = np.linalg.eigh(g)
        self.s = np.maximum(s, 0.0)
        self.x, self.rows, self.n, self.sx = x, rows, len(rows), sx

    def coef(self, y_fn):
        """C0 = Q'X'(y - ybar) and ybar on the fit rows (unscaled target)."""
        xty, sy = 0.0, 0.0
        for i in range(0, len(self.rows), CHUNK):
            r = self.rows[i:i + CHUNK]
            y = y_fn(r)
            xty = xty + self.x[r].astype(np.float64).T @ y
            sy = sy + y.sum(0)
        ybar = sy / self.n
        return self.q.T @ (xty - np.outer(self.sx, ybar)), ybar

    def proj(self, rows):
        """X[rows] Q, float64."""
        return np.concatenate([self.x[rows[i:i + CHUNK]].astype(np.float64) @ self.q for i in range(0, len(rows), CHUNK)])

    def shrink(self, alpha):
        return 1.0 / (self.s + alpha * self.n)


def ridge_select_fit(fit_tr, fit_trva, c_tr, ybar_tr, c_trva, ybar_trva, p_va, y_va, p_te):
    """diagnostics.ridge_select on precomputed fits: alpha by mean val R^2, refit on train+val, test predictions."""
    val_curve = {}
    for a in D.ALPHAS:
        val_curve[a] = float(D.r2(y_va, p_va @ (c_tr * fit_tr.shrink(a)[:, None]) + ybar_tr).mean())
    alpha = max(D.ALPHAS, key=lambda a: val_curve[a])  # first maximum, as in ridge_select
    return p_te @ (c_trva * fit_trva.shrink(alpha)[:, None]) + ybar_trva, alpha, val_curve


def top_right_sv(m, r):
    """Top-r right singular vectors of m (d x p), descending, via eigh of the smaller Gram matrix."""
    d, p = m.shape
    if p <= d:
        _, v = np.linalg.eigh(m.T @ m)
        return np.ascontiguousarray(v[:, ::-1][:, :r])  # a reversed view makes later matmuls skip BLAS (~800x slower)
    ev, u = np.linalg.eigh(m @ m.T)
    ev, u = ev[::-1][:r], u[:, ::-1][:, :r]
    keep = ev > max(ev[0], 0.0) * 1e-12  # directions with no fitted variance don't change W V V'
    return (m.T @ u[:, keep]) / np.sqrt(ev[keep])


def rrr_basis(fit, c0, sq, alpha, rmax):
    """For target scaled by sq (per column): C' = Q'W (so W = Q C') and the top right singular vectors V of X W."""
    cp = (c0 * sq[None, :]) * fit.shrink(alpha)[:, None]
    # (X W)'(X W) = C'' diag(s) C' because (XQ)'(XQ) = diag(s) on the fit rows.
    return cp, top_right_sv(np.sqrt(fit.s)[:, None] * cp, rmax)


def rrr_select(fit_tr, fit_trva, c_tr, ybar_tr, c_trva, ybar_trva, sq, ranks, p_va, y_va, p_te, n_eval):
    """Reduced-rank ridge for one target condition. Per rank: alpha minimizing the weighted val MSE of the rank-r model,
    refit on train+val. Returns {r: (test predictions of the first n_eval columns, unscaled; alpha; val MSE by alpha)}."""
    rmax = max(ranks)
    yc_va = (y_va - ybar_tr) * sq
    tot = (yc_va ** 2).sum()
    curves = {r: {} for r in ranks}
    for a in D.ALPHAS:
        cp, v = rrr_basis(fit_tr, c_tr, sq, a, rmax)
        av, yv = p_va @ (cp @ v), yc_va @ v  # prediction = av[:, :r] V_r' ; ||y - pred||^2 expanded
        for r in ranks:
            mse = (tot - 2 * (yv[:, :r] * av[:, :r]).sum() + (av[:, :r] ** 2).sum()) / len(y_va)
            curves[r][a] = float(mse)
    out, cache = {}, {}
    for r in ranks:
        alpha = min(D.ALPHAS, key=lambda a: curves[r][a])
        if alpha not in cache:
            cache[alpha] = rrr_basis(fit_trva, c_trva, sq, alpha, rmax)
        cp, v = cache[alpha]
        vr = v[:, :r]
        pred = (p_te @ (cp @ vr)) @ vr[:n_eval].T / sq[:n_eval] + ybar_trva[:n_eval]
        out[r] = (pred, alpha, curves[r])
    return out


# ---------------------------------------------------------------------------
# Scalings and bootstrap
# ---------------------------------------------------------------------------

def block_weights(scaling, dims, total_vars):
    if scaling == "a_prime":
        return [1.0 / d for d in dims]
    if scaling == "b":
        return [1.0 / v for v in total_vars]
    if scaling == "a_dprime":
        return [1.0 for _ in dims]
    raise ValueError(scaling)


def column_scale(weights, dims):
    """sqrt(w_k) for every column of block k."""
    return np.concatenate([np.full(d, np.sqrt(w)) for w, d in zip(weights, dims)])


def loss_share(weights, total_vars):
    """Share of the (initial, predict-the-mean) loss carried by each block: w_k V_k / sum_j w_j V_j."""
    wv = [w * v for w, v in zip(weights, total_vars)]
    return [x / sum(wv) for x in wv]


def interference_draws(y_d, pred_ref, pred_with, groups, n_boot, seed):
    """Paired bootstrap draws of R^2_D(ref) - R^2_D(with B), and of each R^2."""
    dr = D.bootstrap_r2(y_d, {"ref": pred_ref, "with": pred_with}, groups, n_boot, seed)
    return dr["ref"] - dr["with"], dr["ref"], dr["with"]


# ---------------------------------------------------------------------------

def edge(alpha):
    return alpha in (D.ALPHAS[0], D.ALPHAS[-1])


def run_type(feats, split, index, groups, base, partner, ftype, ks, ranks, n_boot, seed, log):
    """All measures for one feature type. Returns (JSON dict, per-k bootstrap draws for k-averaging)."""
    train = split == "train"
    f_d = np.concatenate([feats[base][ftype][c] for c in D.CAMS], 1)
    f_b = np.concatenate([feats[partner][ftype][c] for c in D.CAMS], 1)
    dd, db = f_d.shape[1], f_b.shape[1]
    x = np.concatenate([D.standardize(f_d, train), D.standardize(f_b, train)], 1)  # same X for every condition
    x_d = x[:, :dd]
    res = {"dims": {"D": dd, "B": db, "X": dd + db}, "t0": {"features": {}, "by_k": {}}, "by_k": {}}

    for e in (base, partner):
        st = {"cams": {}}
        for c in D.CAMS:
            f = feats[e][ftype][c][train].astype(np.float64)
            st["cams"][c] = {"mean_channel_var": float(f.var(0).mean()),
                             "mean_l2_norm": float(np.linalg.norm(f, axis=1).mean())}
        fc = np.concatenate([feats[e][ftype][c][train] for c in D.CAMS], 1).astype(np.float64)
        st["mean_channel_var"] = float(fc.var(0).mean())
        res["t0"]["features"][e] = st
        log(f"T0 {ftype} {e}: per-channel var {st['mean_channel_var']:.4g}, "
            "L2 norm " + ", ".join(f"{c} {v['mean_l2_norm']:.4g}" for c, v in st["cams"].items()))

    draws = {}
    test_demos = None
    for k in ks:
        t_rows, _ = shift_rows(index, k)
        sp = split[t_rows]
        tr, va, te = t_rows[sp == "train"], t_rows[sp == "val"], t_rows[sp == "test"]
        trva = t_rows[(sp == "train") | (sp == "val")]
        g_te = groups[te]
        demos = set(np.unique(g_te).tolist())
        assert demos == set(np.unique(groups[split == "test"]).tolist()), f"k={k}: a test demo has <= {k} frames"
        assert test_demos is None or demos == test_demos, f"k={k}: test demos differ across k"
        test_demos = demos
        y_fn = delta_fn([f_d, f_b], k)
        rk = {"frames": {"train": len(tr), "val": len(va), "test": len(te)}, "n_test_demos": len(demos)}

        # T0: target scale and loss share.
        _, var = col_moments(y_fn, tr)
        tv = [float(var[:dd].sum()), float(var[dd:].sum())]
        t0 = {"delta_mean_channel_var": {"D": tv[0] / dd, "B": tv[1] / db},
              "delta_total_var": {"D": tv[0], "B": tv[1]},
              "loss_share_B": {s: loss_share(block_weights(s, [dd, db], tv), tv)[1] for s in SCALINGS}}
        res["t0"]["by_k"][str(k)] = t0
        log(f"T0 {ftype} k={k}: delta per-channel var D {t0['delta_mean_channel_var']['D']:.4g} "
            f"B {t0['delta_mean_channel_var']['B']:.4g}; B loss share "
            + ", ".join(f"{s} {v:.3f}" for s, v in t0["loss_share_B"].items()))

        # Fits (X and D-only) on train and train+val; coefficients for the full target [Delta_D, Delta_B].
        tk = time.time()
        fx_tr, fx_trva = Fit(x, tr), Fit(x, trva)
        cx_tr, yb_tr = fx_tr.coef(y_fn)
        cx_trva, yb_trva = fx_trva.coef(y_fn)
        fd_tr, fd_trva = Fit(x_d, tr), Fit(x_d, trva)
        yd_fn = delta_fn([f_d], k)
        cd_tr, ybd_tr = fd_tr.coef(yd_fn)
        cd_trva, ybd_trva = fd_trva.coef(yd_fn)
        y_va, y_te = materialize(y_fn, va), materialize(y_fn, te)
        px_va, px_te = fx_tr.proj(va), fx_trva.proj(te)
        pd_va, pd_te = fd_tr.proj(va), fd_trva.proj(te)
        log(f"{ftype} k={k}: frames {rk['frames']}, fits {time.time() - tk:.0f}s")

        # T1a: X -> Delta_B. T1b: D -> Delta_D and X -> Delta_D.
        sb, sd = slice(dd, None), slice(0, dd)
        pred_1a, a_1a, vc_1a = ridge_select_fit(fx_tr, fx_trva, cx_tr[:, sb], yb_tr[sb], cx_trva[:, sb], yb_trva[sb],
                                                px_va, y_va[:, sb], px_te)
        pred_1b_d, a_1b_d, vc_1b_d = ridge_select_fit(fd_tr, fd_trva, cd_tr, ybd_tr, cd_trva, ybd_trva,
                                                      pd_va, y_va[:, sd], pd_te)
        pred_1b_x, a_1b_x, vc_1b_x = ridge_select_fit(fx_tr, fx_trva, cx_tr[:, sd], yb_tr[sd], cx_trva[:, sd],
                                                      yb_trva[sd], px_va, y_va[:, sd], px_te)

        # T2: reference (Delta_D only; one block, so the scaling can't matter: tested) and with B per scaling.
        sq_ref = column_scale(block_weights(SCALINGS[0], [dd], [tv[0]]), [dd])
        ref = rrr_select(fx_tr, fx_trva, cx_tr[:, sd], yb_tr[sd], cx_trva[:, sd], yb_trva[sd], sq_ref, ranks,
                         px_va, y_va[:, sd], px_te, dd)
        withb = {}
        for s in SCALINGS:
            sq = column_scale(block_weights(s, [dd, db], tv), [dd, db])
            withb[s] = rrr_select(fx_tr, fx_trva, cx_tr, yb_tr, cx_trva, yb_trva, sq, ranks, px_va, y_va, px_te, dd)
        log(f"{ftype} k={k}: T1+T2 fitted, {time.time() - tk:.0f}s since fits started")

        # Bootstrap (same draws for every predictor and every k).
        y_te_d, y_te_b = y_te[:, sd], y_te[:, sb]
        dr_b = D.bootstrap_r2(y_te_b, {"t1a": pred_1a}, g_te, n_boot, seed)
        preds_d = {"t1b_D": pred_1b_d, "t1b_X": pred_1b_x}
        preds_d.update({f"ref|{r}": ref[r][0] for r in ranks})
        preds_d.update({f"{s}|{r}": withb[s][r][0] for s in SCALINGS for r in ranks})
        dr_d = D.bootstrap_r2(y_te_d, preds_d, g_te, n_boot, seed)
        dk = {"t1a": dr_b["t1a"], "t1b_gain": dr_d["t1b_X"] - dr_d["t1b_D"]}

        r2_te = lambda y, p: float(D.r2(y, p).mean())
        rk["t1a"] = {"r2": r2_te(y_te_b, pred_1a), "ci95": D.ci(dr_b["t1a"]), "alpha": a_1a, "val_r2_by_alpha": vc_1a}
        rk["t1b"] = {"D_only": {"r2": r2_te(y_te_d, pred_1b_d), "ci95": D.ci(dr_d["t1b_D"]), "alpha": a_1b_d,
                                "val_r2_by_alpha": vc_1b_d},
                     "D_plus_B": {"r2": r2_te(y_te_d, pred_1b_x), "ci95": D.ci(dr_d["t1b_X"]), "alpha": a_1b_x,
                                  "val_r2_by_alpha": vc_1b_x}}
        rk["t1b"]["gain"] = {"value": rk["t1b"]["D_plus_B"]["r2"] - rk["t1b"]["D_only"]["r2"],
                             "ci95": D.ci(dk["t1b_gain"])}
        log(f"T1 {ftype} k={k}: T1a R2 {rk['t1a']['r2']:.3f} {rk['t1a']['ci95']}; T1b D {rk['t1b']['D_only']['r2']:.3f} "
            f"D+B {rk['t1b']['D_plus_B']['r2']:.3f} gain {rk['t1b']['gain']['value']:+.4f} {rk['t1b']['gain']['ci95']}")

        t2 = {"reference": {}, "with_B": {s: {} for s in SCALINGS}}
        for r in ranks:
            t2["reference"][str(r)] = {"r2_D": r2_te(y_te_d, ref[r][0]), "ci95": D.ci(dr_d[f"ref|{r}"]),
                                       "alpha": ref[r][1], "val_mse_by_alpha": ref[r][2]}
            for s in SCALINGS:
                pred, alpha, curve = withb[s][r]
                inter = dr_d[f"ref|{r}"] - dr_d[f"{s}|{r}"]
                dk[f"{s}|{r}"] = inter
                ent = {"r2_D": r2_te(y_te_d, pred), "ci95": D.ci(dr_d[f"{s}|{r}"]), "alpha": alpha,
                       "val_mse_by_alpha": curve}
                ent["interference"] = t2["reference"][str(r)]["r2_D"] - ent["r2_D"]
                ent["interference_ci95"] = D.ci(inter)
                t2["with_B"][s][str(r)] = ent
                log(f"T2 {ftype} k={k} r={r} {s:8s}: R2_D ref {t2['reference'][str(r)]['r2_D']:.4f} "
                    f"with B {ent['r2_D']:.4f} interference {ent['interference']:+.4f} {ent['interference_ci95']} "
                    f"(alpha ref {ref[r][1]:g}, with {alpha:g})")
        rk["t2"] = t2
        res["by_k"][str(k)] = rk
        draws[k] = dk
        del fx_tr, fx_trva, fd_tr, fd_trva, cx_tr, cx_trva, px_va, px_te, pd_va, pd_te, y_va, y_te

    # Averages over k (point values averaged; CIs from per-draw averages).
    kk = [str(k) for k in ks]
    avg = lambda f: float(np.mean([f(res["by_k"][k]) for k in kk]))
    dmean = lambda key: np.mean([draws[k][key] for k in ks], 0)
    km = {"t1a": {"r2": avg(lambda b: b["t1a"]["r2"]), "ci95": D.ci(dmean("t1a"))},
          "t1b_gain": {"value": avg(lambda b: b["t1b"]["gain"]["value"]), "ci95": D.ci(dmean("t1b_gain"))},
          "t2": {s: {} for s in SCALINGS}}
    for s in SCALINGS:
        for r in ranks:
            km["t2"][s][str(r)] = {
                "interference": avg(lambda b: b["t2"]["with_B"][s][str(r)]["interference"]),
                "ci95": D.ci(dmean(f"{s}|{r}")),
                "r2_D_ref": avg(lambda b: b["t2"]["reference"][str(r)]["r2_D"]),
                "r2_D_with_B": avg(lambda b: b["t2"]["with_B"][s][str(r)]["r2_D"])}
    km["t0_loss_share_B"] = {s: float(np.mean([res["t0"]["by_k"][k]["loss_share_B"][s] for k in kk])) for s in SCALINGS}
    res["k_mean"] = km
    return res


def alpha_edges(res):
    """Every alpha choice at the grid edge (recorded, not acted on: PLAN pitfalls)."""
    hits = []
    for t, rt in res["types"].items():
        for k, rk in rt["by_k"].items():
            for name, a in [("t1a", rk["t1a"]["alpha"]), ("t1b_D", rk["t1b"]["D_only"]["alpha"]),
                            ("t1b_X", rk["t1b"]["D_plus_B"]["alpha"])]:
                if edge(a):
                    hits.append(f"{t}|k={k}|{name}|alpha={a:g}")
            for r, e in rk["t2"]["reference"].items():
                if edge(e["alpha"]):
                    hits.append(f"{t}|k={k}|t2_ref|r={r}|alpha={e['alpha']:g}")
            for s, by_r in rk["t2"]["with_B"].items():
                for r, e in by_r.items():
                    if edge(e["alpha"]):
                        hits.append(f"{t}|k={k}|t2_{s}|r={r}|alpha={e['alpha']:g}")
    return hits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feats", required=True, help="root written by extract_features.py --out")
    p.add_argument("--suite", required=True)
    p.add_argument("--raw", required=True, help="dir with the suite's LIBERO HDF5 files (demo index)")
    p.add_argument("--base", default="dinov3")
    p.add_argument("--partner", required=True)
    p.add_argument("--types", nargs="+", default=["global", "mean"], help="first is primary")
    p.add_argument("--ks", nargs="+", type=int, default=KS)
    p.add_argument("--ranks", nargs="+", type=int, default=RANKS, help="first is primary")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    t0 = time.time()
    log = lambda m: print(f"[target {time.time() - t0:6.0f}s] {m}", flush=True)
    args.encoders = [args.base, args.partner]
    feats, _, _, split, index, meta, files = D.load(args)
    groups = np.repeat(np.arange(len(index)), [n for _, _, n in index])  # demo id per frame
    counts = {s: int((split == s).sum()) for s in ("train", "val", "test")}
    log(f"{len(split)} frames, {len(index)} demos from {len(files)} files, split {counts}")
    res = {"suite": args.suite, "files": files, "base": args.base, "partner": args.partner,
           "encoders": {e: meta[e] for e in args.encoders}, "git_commit": D.git_commit(),
           "config": {"prereg": "LOG 2026-10-21 (i) as amended by (j)", "ks": args.ks, "ranks": args.ranks,
                      "primary_rank": args.ranks[0], "scalings": SCALING_DOC, "primary_scaling": SCALINGS[0],
                      "types": args.types, "primary_type": args.types[0], "alphas": D.ALPHAS, "seed": args.seed,
                      "n_boot": args.n_boot},
           "n_frames": len(split), "split_frames": counts, "types": {}}
    for t in args.types:
        res["types"][t] = run_type(feats, split, index, groups, args.base, args.partner, t, args.ks, args.ranks,
                                   args.n_boot, args.seed, log)
        res["frames_by_k"] = {k: v["frames"] for k, v in res["types"][t]["by_k"].items()}
    prim = res["types"][args.types[0]]["k_mean"]["t2"][SCALINGS[0]][str(args.ranks[0])]
    res["primary"] = {"definition": f"interference under {SCALINGS[0]}, r = {args.ranks[0]}, {args.types[0]}, "
                                    f"mean over k in {args.ks}", "interference": prim["interference"],
                      "ci95": prim["ci95"]}
    res["alpha_edge_hits"] = alpha_edges(res)
    res["seconds"] = time.time() - t0
    log(f"PRIMARY {args.base}+{args.partner}: interference {prim['interference']:+.4f} CI95 {prim['ci95']}")
    log(f"alpha at grid edge: {res['alpha_edge_hits'] or 'none'}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
