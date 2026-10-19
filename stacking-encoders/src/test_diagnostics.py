"""Synthetic sanity checks for diagnostics.py (cases with known answers). Run: python test_diagnostics.py"""

import numpy as np

import diagnostics as D


def main():
    rng = np.random.default_rng(0)
    n = 1500
    split = np.array(["train"] * 900 + ["val"] * 300 + ["test"] * 300)
    train = split == "train"

    x, y = rng.normal(size=(n, 300)), rng.normal(size=(n, 300))
    q, _ = np.linalg.qr(rng.normal(size=(300, 300)))
    assert np.allclose(D.cka(x, x), 1)
    assert np.allclose(D.cka(x, 3 * x @ q), 1), "CKA must be rotation/scale invariant"
    bi, de = D.cka(x, y)
    assert abs(de) < 0.02 and bi > 0.1, (bi, de)  # biased CKA is inflated when d is not << n
    print(f"CKA ok: independent biased {bi:.3f} debiased {de:.4f}")

    # Action depends on a[:, :5]. B copies a[:, :25] (so it holds all of A's action info) plus noise.
    a = rng.normal(size=(n, 50))
    b = np.concatenate([a[:, :25], rng.normal(size=(n, 25))], 1)
    act = a[:, :5] @ rng.normal(size=(5, 7)) + 0.1 * rng.normal(size=(n, 7))
    A, B = D.standardize(a, train), D.standardize(b, train)
    gain = D.probe(np.concatenate([A, B], 1), act, split)[0]["r2_mean"] - D.probe(A, act, split)[0]["r2_mean"]
    assert abs(gain) < 0.05, gain
    print(f"redundant B: gain(B|A) {gain:+.3f} (expect ~0)")

    # B also carries z, which A lacks and which drives half the action variance.
    z = rng.normal(size=(n, 5))
    b2 = D.standardize(np.concatenate([a[:, :25], z, rng.normal(size=(n, 20))], 1), train)
    act2 = a[:, :5] @ rng.normal(size=(5, 7)) + z @ rng.normal(size=(5, 7))
    gain = D.probe(np.concatenate([A, b2], 1), act2, split)[0]["r2_mean"] - D.probe(A, act2, split)[0]["r2_mean"]
    assert 0.3 < gain < 0.7, gain
    print(f"informative B: gain(B|A) {gain:.3f} (expect ~0.5)")

    # Bootstrap: R^2 from per-demo statistics matches r2() on an explicit resample.
    groups = np.repeat(np.arange(60), 25)
    y, yhat = rng.normal(size=(1500, 7)), rng.normal(size=(1500, 7))
    stats = D.demo_stats(y, yhat, groups)
    ones = np.ones((1, 60))
    assert np.allclose(D.r2_from_stats(ones, *stats), D.r2(y, yhat).mean()), "unit weights must give the point R^2"
    pick = rng.integers(0, 60, size=60)
    w = np.bincount(pick, minlength=60)[None].astype(float)
    rows = np.concatenate([np.where(groups == g)[0] for g in pick])
    assert np.allclose(D.r2_from_stats(w, *stats), D.r2(y[rows], yhat[rows]).mean()), "weighted R^2 != explicit resample"
    print("bootstrap R^2 from demo stats ok")

    # Bootstrap with demo-correlated data: within-demo frames share a demo offset, so a frame-level
    # bootstrap would be overconfident. Informative B: gain CI excludes 0. Redundant B: CI covers ~0.
    n_demo, per = 300, 50
    g = np.repeat(np.arange(n_demo), per)
    sp = np.where(g < 180, "train", np.where(g < 240, "val", "test"))
    trn = sp == "train"
    lat = np.repeat(rng.normal(size=(n_demo, 10)), per, 0) + 0.3 * rng.normal(size=(n_demo * per, 10))
    za = lat[:, :5]; zb = lat[:, 5:]
    act3 = za @ rng.normal(size=(5, 7)) + 0.5 * zb @ rng.normal(size=(5, 7)) + 0.5 * rng.normal(size=(n_demo * per, 7))
    A3 = D.standardize(np.concatenate([za, rng.normal(size=(n_demo * per, 20))], 1), trn)
    B3 = D.standardize(np.concatenate([za + 0.1 * rng.normal(size=za.shape), zb], 1), trn)
    C3 = D.standardize(za + 0.1 * rng.normal(size=za.shape), trn)  # redundant with A3
    te = sp == "test"
    preds = {k: D.probe(x, act3, sp)[1] for k, x in
             {"A": A3, "A+B": np.concatenate([A3, B3], 1), "A+C": np.concatenate([A3, C3], 1)}.items()}
    dr = D.bootstrap_r2(act3[te], preds, g[te], n_boot=500, seed=0)
    ci_b, ci_c = D.ci(dr["A+B"] - dr["A"]), D.ci(dr["A+C"] - dr["A"])
    assert ci_b[0] > 0.05, ci_b
    assert ci_c[0] < 0.01 and ci_c[1] > -0.01 and ci_c[1] - ci_c[0] < 0.05, ci_c
    print(f"bootstrap gain CI: informative B {ci_b[0]:+.3f}..{ci_b[1]:+.3f}, redundant C {ci_c[0]:+.3f}..{ci_c[1]:+.3f}")
    print("all ok")


if __name__ == "__main__":
    main()
