"""Synthetic checks for target_diagnostics.py (cases with known answers). Run: python test_target_diagnostics.py

Checks 1-5 are the ones listed in experiments/2026-10-21-target-diagnostics/PLAN.md; 6-7 check the fast ridge against
diagnostics.ridge_select and run run_type end to end. Thresholds in check 4 were fixed before the first run.
"""

import json

import numpy as np

import diagnostics as D
import target_diagnostics as T


def fits(x, y, split):
    """Fits, coefficients and projections as run_type builds them, for an explicit target array y."""
    rows = np.arange(len(x))
    tr, va, te = rows[split == "train"], rows[split == "val"], rows[split == "test"]
    trva = rows[split != "test"]
    y_fn = lambda r: y[r].astype(np.float64)
    f_tr, f_trva = T.Fit(x, tr), T.Fit(x, trva)
    c_tr, yb_tr = f_tr.coef(y_fn)
    c_trva, yb_trva = f_trva.coef(y_fn)
    return dict(fit_tr=f_tr, fit_trva=f_trva, c_tr=c_tr, ybar_tr=yb_tr, c_trva=c_trva, ybar_trva=yb_trva,
                p_va=f_tr.proj(va), y_va=y[va].astype(np.float64), p_te=f_trva.proj(te)), te


def demo_split(n_demo, per):
    g = np.repeat(np.arange(n_demo), per)
    a, b = int(n_demo * 0.6), int(n_demo * 0.8)
    return g, np.where(g < a, "train", np.where(g < b, "val", "test"))


def main():
    rng = np.random.default_rng(0)

    # 1. Shift: feature = frame index within its demo, so Delta = k exactly; no pair crosses a demo boundary.
    lengths = [5, 40, 33, 32, 1, 17, 64]
    index = [("f", f"demo_{i}", n) for i, n in enumerate(lengths)]
    within = np.concatenate([np.arange(n) for n in lengths]).astype(np.float32)[:, None]
    demo_of = np.repeat(np.arange(len(lengths)), lengths)
    for k in T.KS:
        t, tk = T.shift_rows(index, k)
        assert len(t) == sum(max(0, n - k) for n in lengths), (k, len(t))
        assert (demo_of[t] == demo_of[tk]).all(), f"k={k}: a pair crosses a demo boundary"
        assert np.array_equal(T.delta_fn([within], k)(t)[:, 0], np.full(len(t), float(k))), k
    print(f"1 shift ok: Delta = k, no cross-demo pairs, counts = sum max(0, len - k) for k in {T.KS}")

    # Shared synthetic regression: 60 latent factors drive X; demo-structured split.
    g, split = demo_split(200, 40)
    n = len(g)
    z = rng.normal(size=(n, 60))
    x = D.standardize(z @ rng.normal(size=(60, 80)) + 0.1 * rng.normal(size=(n, 80)), split == "train").astype(np.float32)

    # 2. RRR at full rank (r >= target dims) equals ridge predictions at the same alpha. Both SVD branches:
    #    target narrower than X (p <= d) and wider (p > d).
    for p in (30, 120):
        y = z @ rng.normal(size=(60, p)) + 0.3 * rng.normal(size=(n, p))
        f, te = fits(x, y, split)
        sq = np.ones(p)
        out = T.rrr_select(**f, sq=sq, ranks=[p], n_eval=p)
        pred, alpha, _ = out[p]
        ridge = f["p_te"] @ (f["c_trva"] * f["fit_trva"].shrink(alpha)[:, None]) + f["ybar_trva"]
        err = np.abs(pred - ridge).max()
        assert err < 1e-8, (p, err)
        print(f"2 full-rank RRR == ridge (target dims {p}, X dims 80, alpha {alpha:g}): max |diff| {err:.1e}")

    # 3. Single-block invariance: a Delta_D-only reference gives the same alpha and R^2 under all three scalings.
    y = z[:, :20] @ rng.normal(size=(20, 30)) * 3.0 + 0.5 * rng.normal(size=(n, 30))
    f, te = fits(x, y, split)
    tv = [float(y[split == "train"].var(0).sum())]
    r2s = {}
    for s in T.SCALINGS:
        sq = T.column_scale(T.block_weights(s, [30], tv), [30])
        out = T.rrr_select(**f, sq=sq, ranks=[8, 16], n_eval=30)
        r2s[s] = [(out[r][1], D.r2(y[te], out[r][0]).mean()) for r in (8, 16)]
    for s in T.SCALINGS[1:]:
        for (a0, v0), (a1, v1) in zip(r2s[T.SCALINGS[0]], r2s[s]):
            assert a0 == a1 and abs(v0 - v1) < 1e-10, (s, a0, a1, v0, v1)
    print(f"3 single-block invariance ok: (alpha, R2) at r=8,16 {[(a, round(v, 4)) for a, v in r2s['a_prime']]} "
          f"identical under {T.SCALINGS}")

    # 4. Planted interference. Delta_D (30-d) depends on factors 0-19; Delta_B (60-d) on factors 20-59 at x10 scale.
    #    Thresholds fixed before running:
    #      r = 32: interference > 0.5 under a_dprime and a_prime; < 0.05 under b; b < min(a_prime, a_dprime) - 0.4.
    #      r = 64 (all 60 factors fit): |interference| < 0.02 under every scaling.
    #    The maps have orthonormal rows, so every factor carries the same variance within its block: under b each D
    #    direction has V/20 and each B direction V/40, so the top 32 hold all of D. (First run, 2026-10-21, used
    #    Gaussian maps: their uneven singular values put 6 of D's 20 directions below B's top ones, interference(b)
    #    was +0.082 and the test failed. The data was changed to match the premise; the thresholds were not.)
    dd, db = 30, 60
    orth = lambda a, b: np.linalg.qr(rng.normal(size=(b, a)))[0].T  # (a, b), orthonormal rows
    y_d = z[:, :20] @ orth(20, dd) + 0.1 * rng.normal(size=(n, dd))
    y_b = 10 * z[:, 20:60] @ orth(40, db) + 0.1 * rng.normal(size=(n, db))
    y = np.concatenate([y_d, y_b], 1)
    f, te = fits(x, y, split)
    f_ref, _ = fits(x, y_d, split)
    tr = split == "train"
    tv = [float(y_d[tr].var(0).sum()), float(y_b[tr].var(0).sum())]
    ranks = [32, 64]
    ref = T.rrr_select(**f_ref, sq=T.column_scale(T.block_weights("a_prime", [dd], tv[:1]), [dd]), ranks=ranks, n_eval=dd)
    inter = {}
    for s in T.SCALINGS:
        sq = T.column_scale(T.block_weights(s, [dd, db], tv), [dd, db])
        out = T.rrr_select(**f, sq=sq, ranks=ranks, n_eval=dd)
        inter[s] = {r: D.r2(y_d[te], ref[r][0]).mean() - D.r2(y_d[te], out[r][0]).mean() for r in ranks}
    share = {s: T.loss_share(T.block_weights(s, [dd, db], tv), tv)[1] for s in T.SCALINGS}
    print("4 planted interference: B loss share " + ", ".join(f"{s} {v:.3f}" for s, v in share.items()))
    for r in ranks:
        print(f"  r={r}: R2_D ref {D.r2(y_d[te], ref[r][0]).mean():.3f}; interference "
              + ", ".join(f"{s} {inter[s][r]:+.4f}" for s in T.SCALINGS))
    assert inter["a_dprime"][32] > 0.5 and inter["a_prime"][32] > 0.5, inter
    assert inter["b"][32] < 0.05 and inter["b"][32] < min(inter["a_prime"][32], inter["a_dprime"][32]) - 0.4, inter
    assert all(abs(inter[s][64]) < 0.02 for s in T.SCALINGS), inter
    assert abs(share["b"] - 0.5) < 1e-12, share
    print("  ok")

    # 5. Bootstrap pairing: two identical predictors in one bootstrap_r2 call (as run_type does) give an
    #    interference CI of exactly [0, 0].
    p_ref = ref[32][0]
    dr = D.bootstrap_r2(y_d[te], {"ref|32": p_ref, "a_prime|32": p_ref.copy()}, g[te], 1000, 0)
    c = D.ci(dr["ref|32"] - dr["a_prime|32"])
    assert c == [0.0, 0.0], c
    print(f"5 bootstrap pairing ok: identical predictors give CI {c}")

    # 6. Fast ridge (one eigendecomposition per fit set) == diagnostics.ridge_select. X is standardized on train only,
    #    so the train+val refit has non-centered X, as in the real run.
    y = z[:, :10] @ rng.normal(size=(10, 7)) + rng.normal(size=(n, 7))
    f, te = fits(x, y, split)
    pred, alpha, curve = T.ridge_select_fit(f["fit_tr"], f["fit_trva"], f["c_tr"], f["ybar_tr"], f["c_trva"],
                                            f["ybar_trva"], f["p_va"], f["y_va"], f["p_te"])
    r2_ref, alpha_ref, yhat_ref, curve_ref = D.ridge_select(x.astype(np.float64), y, split)
    err = np.abs(pred - yhat_ref).max()
    assert alpha == alpha_ref and err < 1e-8, (alpha, alpha_ref, err)
    assert all(abs(curve[a] - curve_ref[a]) < 1e-10 for a in D.ALPHAS)
    print(f"6 fast ridge == diagnostics.ridge_select: alpha {alpha:g}, max |pred diff| {err:.1e}")

    # 7. run_type end to end on random demo-structured features (2 cams, 2 feature types): runs, outputs finite and
    #    JSON-serializable, frame counts per split match the shift, b's loss share is 1/2, primary keys exist.
    #    A test demo of <= 32 frames must trip the assert.
    lengths = rng.integers(40, 70, size=50)
    index = [("task", f"demo_{i}", int(m)) for i, m in enumerate(lengths)]
    nf = int(lengths.sum())
    groups = np.repeat(np.arange(50), lengths)
    split7 = np.repeat(np.where(np.arange(50) < 30, "train", np.where(np.arange(50) < 40, "val", "test")), lengths)
    feats = {e: {t: {c: np.cumsum(rng.normal(size=(nf, dim)), 0).astype(np.float32) / 10 for c in D.CAMS}
                 for t in ("global", "mean")} for e, dim in (("dinov3", 12), ("partner", 20))}
    res = T.run_type(feats, split7, index, groups, "dinov3", "partner", "global", T.KS, [4, 8, 64], 200, 0,
                     lambda m: None)
    json.dumps(res)
    for k in T.KS:
        t, _ = T.shift_rows(index, k)
        exp = {s: int((split7[t] == s).sum()) for s in ("train", "val", "test")}
        assert res["by_k"][str(k)]["frames"] == exp, (k, res["by_k"][str(k)]["frames"], exp)
        assert abs(res["t0"]["by_k"][str(k)]["loss_share_B"]["b"] - 0.5) < 1e-12
    vals = [e["interference"] for by_r in res["k_mean"]["t2"].values() for e in by_r.values()]
    assert np.isfinite(vals).all() and len(vals) == 9, vals
    print(f"7 run_type end to end ok: k-mean interference (a_prime, r=64) "
          f"{res['k_mean']['t2']['a_prime']['64']['interference']:+.4f} CI {res['k_mean']['t2']['a_prime']['64']['ci95']}")
    short = [(s, d, 30 if i == 45 else m) for i, (s, d, m) in enumerate(index)]
    sl = np.array([m for _, _, m in short])
    keep = np.concatenate([np.arange(st, st + m) for st, m in zip(np.cumsum(lengths) - lengths, sl)])
    try:
        T.run_type({e: {t: {c: a[keep] for c, a in tv.items()} for t, tv in ev.items()} for e, ev in feats.items()},
                   split7[keep], short, np.repeat(np.arange(50), sl), "dinov3", "partner", "global", T.KS, [4], 50, 0,
                   lambda m: None)
        raise RuntimeError("short test demo did not trip the assert")
    except AssertionError as e:
        print(f"  short test demo trips the assert: {e}")
    print("all ok")


if __name__ == "__main__":
    main()
