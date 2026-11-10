"""Synthetic sanity checks for diagnostics.py (cases with known answers). Run from src/: python test_diagnostics.py

The last checks (synthetic control partners, experiments/2026-11-09-control-partners/) build a small LIBERO-shaped HDF5
fixture in a temp dir and run diagnostics.load and diagnostics.main on it.
"""

import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np

import diagnostics as D


def make_fixture(root, n_files=2, n_demo=10, length=60, dim=8, seed=0):
    """LIBERO-shaped files: <root>/raw/*.hdf5 (actions, joint/gripper states) and <root>/feats/dinov3/suite/*.h5
    (global, mean, grid per cam). Actions depend linearly on a latent that dinov3's features carry. Returns the paths."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "raw").mkdir(parents=True)
    (root / "feats" / "dinov3" / "suite").mkdir(parents=True)
    w_act = rng.normal(size=(dim, 7))
    for i in range(n_files):
        with h5py.File(root / "raw" / f"task{i}_demo.hdf5", "w") as fr, \
                h5py.File(root / "feats" / "dinov3" / "suite" / f"task{i}_demo.h5", "w") as ff:
            ff.attrs["meta"] = json.dumps({"encoder": "fixture-dinov3"})
            for d in range(n_demo):
                lat = np.cumsum(rng.normal(size=(length, dim)), 0) / 5
                g = fr.create_group(f"data/demo_{d}")
                g["actions"] = lat @ w_act + 0.3 * rng.normal(size=(length, 7))
                g["obs/joint_states"] = rng.normal(size=(length, 7))
                g["obs/gripper_states"] = rng.normal(size=(length, 2))
                for c in D.CAMS:
                    x = lat + 0.2 * rng.normal(size=lat.shape)
                    gc = ff.create_group(f"demo_{d}/{c}")
                    gc["global"], gc["mean"] = x.astype(np.float32), x.astype(np.float32)
                    gc["grid"] = np.stack([x, x], 1).astype(np.float32)
    return Namespace(feats=str(root / "feats"), raw=str(root / "raw"), suite="suite")


def check_synthetic():
    """Synthetic control partners: shapes, scale, determinism, known CKA and gain, and real-encoder loads unchanged."""
    with tempfile.TemporaryDirectory() as tmp:
        fx = make_fixture(tmp)
        args = lambda enc, seed=0: Namespace(**vars(fx), encoders=enc, seed=seed)
        types = ("global", "mean")
        base = D.load(args(["dinov3"]), types)
        specs = ["noise:64", "noise:64:0.25", "dup:dinov3", "randproj:dinov3:48", "shuffle:dinov3"]
        feats, actions, proprio, split, index, meta, files = D.load(args(["dinov3"] + specs), types)
        # Real encoders: the same arrays, labels, split and index whether or not synthetic partners are requested.
        assert all(np.array_equal(base[0]["dinov3"][t][c], feats["dinov3"][t][c]) for t in types for c in D.CAMS)
        assert np.array_equal(base[1], actions) and np.array_equal(base[2], proprio) and np.array_equal(base[3], split)
        assert base[4] == index and base[6] == files and set(base[5]) == {"dinov3"}
        print("synthetic: real-encoder load identical with and without synthetic partners")

        d = feats["dinov3"]
        train = split == "train"
        nf = len(actions)
        for t in types:
            for c in D.CAMS:
                assert feats["noise:64"][t][c].shape == (nf, 32) and feats["randproj:dinov3:48"][t][c].shape == (nf, 24)
                assert feats["dup:dinov3"][t][c].shape == d[t][c].shape == feats["shuffle:dinov3"][t][c].shape
                assert all(feats[e][t][c].dtype == np.float32 for e in feats)
        assert meta["noise:64"]["synthetic"] == "noise:64" and meta["noise:64"]["seed"] == 0
        assert meta["noise:64"]["reference"] == "dinov3" and meta["randproj:dinov3:48"]["source"] == "dinov3"
        json.dumps(meta)
        print("synthetic: shapes (n, N/2 per cam for noise and randproj, source dims for dup and shuffle), float32, meta ok")

        # Determinism: same seed same arrays, different seed different arrays; noise scale follows the reference.
        c0 = D.CAMS[0]
        again = D.load(args(["dinov3", "noise:64", "randproj:dinov3:48", "shuffle:dinov3"]), types)[0]
        other = D.load(args(["dinov3", "noise:64", "randproj:dinov3:48", "shuffle:dinov3"], seed=1), types)[0]
        for e in ("noise:64", "randproj:dinov3:48", "shuffle:dinov3"):
            assert np.array_equal(again[e]["global"][c0], feats[e]["global"][c0]), e
            assert not np.array_equal(other[e]["global"][c0], feats[e]["global"][c0]), e
        ref_var = d["global"][c0][train].astype(np.float64).var(0).mean()
        r1 = feats["noise:64"]["global"][c0][train].astype(np.float64).var(0).mean() / ref_var
        r2_ = feats["noise:64:0.25"]["global"][c0][train].astype(np.float64).var(0).mean() / ref_var
        assert abs(r1 - 1) < 0.1 and abs(r2_ - 0.25) < 0.05, (r1, r2_)
        print(f"synthetic: deterministic per seed; noise variance / reference variance {r1:.3f} (s=1), {r2_:.3f} (s=0.25)")

        # Known structure: dup = copy; shuffle = same frames permuted jointly over cams; randproj keeps the rank of its
        # source and the per-channel variance.
        assert np.array_equal(feats["dup:dinov3"]["global"][c0], d["global"][c0])
        pairs = lambda a, b: {(x.tobytes(), y.tobytes()) for x, y in zip(a, b)}
        sh = feats["shuffle:dinov3"]["global"]
        assert pairs(sh[D.CAMS[0]], sh[D.CAMS[1]]) == pairs(d["global"][D.CAMS[0]], d["global"][D.CAMS[1]])
        assert not np.array_equal(sh[c0], d["global"][c0])
        pj = feats["randproj:dinov3:48"]["global"][c0].astype(np.float64)
        sv = np.linalg.svd(pj - pj.mean(0), compute_uv=False)
        assert (sv > 1e-4 * sv[0]).sum() == d["global"][c0].shape[1], sv[:12]
        assert 0.3 < pj.var(0).mean() / d["global"][c0].astype(np.float64).var(0).mean() < 3
        print("synthetic: dup is a copy, shuffle permutes whole frames jointly over cams, randproj rank = source rank")

        # CKA with dinov3: dup 1, noise ~0 and shuffled ~0 (debiased), randproj high.
        x = d["mean"][c0]  # raw, as main() passes it to cka
        c_dup, c_noise, c_shuf, c_proj = (D.cka(x, feats[e]["mean"][c0])[1]
                                          for e in ("dup:dinov3", "noise:64", "shuffle:dinov3", "randproj:dinov3:48"))
        assert abs(c_dup - 1) < 1e-6 and abs(c_noise) < 0.05 and abs(c_shuf) < 0.05 and c_proj > 0.5, \
            (c_dup, c_noise, c_shuf, c_proj)
        print(f"synthetic: debiased CKA with dinov3: dup {c_dup:.3f}, noise {c_noise:+.3f}, shuffle {c_shuf:+.3f}, "
              f"randproj {c_proj:.3f}")

        # End to end through main(): gains, metadata in the result JSON, saved bootstrap draws.
        gains = {}
        for spec in ("dup:dinov3", "randproj:dinov3:48", "noise:64", "shuffle:dinov3"):
            out = Path(tmp) / f"res-{spec.replace(':', '-')}.json"
            sys.argv = ["diagnostics.py", "--feats", fx.feats, "--suite", fx.suite, "--raw", fx.raw, "--encoders",
                        "dinov3", spec, "--probe-types", "global", "--n-boot", "100", "--cka-n", "400", "--seed", "3",
                        "--save-draws", str(Path(tmp) / "draws.npz"), "--out", str(out)]
            D.main()
            r = json.loads(out.read_text())
            assert r["encoders"][spec]["synthetic"] == spec and r["encoders"][spec]["seed"] == 3 and r["seed"] == 3
            pr = r["pairs"][f"dinov3|{spec}|global"]
            gains[spec] = pr["gain_B_given_A"]
            assert pr["gain_ci95"][0] <= gains[spec] <= pr["gain_ci95"][1]
            assert not any("grid" in k and spec in k for k in r["cka"]), "no grid CKA for synthetic partners"
            dr = np.load(Path(tmp) / "draws.npz")
            assert abs(dr[f"dinov3|{spec}|global"].mean() - gains[spec]) < 0.02
        assert abs(gains["dup:dinov3"]) < 0.01, gains
        assert abs(gains["randproj:dinov3:48"]) < 0.02, gains
        assert gains["noise:64"] < 0.01 and gains["shuffle:dinov3"] < 0.01, gains
        print("synthetic: gain(B|dinov3) via main(): " + ", ".join(f"{k} {v:+.4f}" for k, v in gains.items()))

        # --synth-seed changes the synthetic draw only: same bootstrap seed, so the demo draws (and dinov3 alone) are unchanged.
        res = {}
        for ss in (None, 9):
            out = Path(tmp) / f"ss-{ss}.json"
            sys.argv = ["diagnostics.py", "--feats", fx.feats, "--suite", fx.suite, "--raw", fx.raw, "--encoders", "dinov3",
                        "noise:64", "--probe-types", "global", "--n-boot", "100", "--cka-n", "400", "--seed", "3",
                        "--out", str(out)] + ([] if ss is None else ["--synth-seed", str(ss)])
            D.main()
            res[ss] = json.loads(out.read_text())
        assert res[None]["encoders"]["noise:64"]["seed"] == 3 and res[9]["encoders"]["noise:64"]["seed"] == 9
        assert res[9]["seed"] == 3 and res[None]["probes"]["dinov3|global"] == res[9]["probes"]["dinov3|global"]
        assert res[None]["probes"]["noise:64|global"] != res[9]["probes"]["noise:64|global"]
        print("synthetic: --synth-seed changes the noise draw only (bootstrap seed and dinov3 results unchanged)")


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
    check_synthetic()
    print("all ok")


if __name__ == "__main__":
    main()
