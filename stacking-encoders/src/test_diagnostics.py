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
    r, _, _ = D.residualize(B, A, split)
    resid = D.probe(D.standardize(r, train), act, split)["r2_mean"]
    assert resid < 0.1, resid
    print(f"redundant B: resid(B|A) R2 {resid:.3f} (expect ~0)")

    # B also carries z, which A lacks and which drives half the action variance.
    z = rng.normal(size=(n, 5))
    b2 = D.standardize(np.concatenate([a[:, :25], z, rng.normal(size=(n, 20))], 1), train)
    act2 = a[:, :5] @ rng.normal(size=(5, 7)) + z @ rng.normal(size=(5, 7))
    r, _, _ = D.residualize(b2, A, split)
    resid = D.probe(D.standardize(r, train), act2, split)["r2_mean"]
    gain = D.probe(np.concatenate([A, b2], 1), act2, split)["r2_mean"] - D.probe(A, act2, split)["r2_mean"]
    assert 0.3 < resid < 0.7 and 0.3 < gain < 0.7, (resid, gain)
    print(f"informative B: resid(B|A) R2 {resid:.3f}, gain(B|A) {gain:.3f} (expect ~0.5)")
    print("all ok")


if __name__ == "__main__":
    main()
