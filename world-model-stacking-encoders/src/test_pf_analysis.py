"""Synthetic checks for pf_analysis.py (cases with known answers). Run: python test_pf_analysis.py

Needs numpy and h5py. Writes synthetic features in the real file layout under a temp dir and never reads
real files. The synthetic world mirrors the redesign: a raw file per task with 50 demos (so the real split,
train demo_0..29, val demo_30..39, test demo_40..49, applies), re-rendered clean features of the 40 train+val
demos in a phase-A root (the test demos in that file are NaN-poisoned: reading one fails loudly), re-rendered
test features in per-condition roots, and an optional stored-feature root (test demos poisoned too).
"""

import contextlib
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np

import diagnostics as D
import pf_analysis as P

SUITE = "libero_goal"
DIMS = {"dinov3": 8, "siglip2": 8, "sam3": 8, "c-radiov4": 14, "vggt-omega": 12}  # per camera
PASS_REF = -1.0   # a GATE_REF that makes the gate pass on any synthetic world (threshold -1.025)


def meta_for(e):
    return {"encoder": e, "hf_id": "hf/" + e, "resolution": 224, "rotate180": True, "dtype": "float32", "grid": 4,
            "frame_stride": 1, "cams": list(D.CAMS), "n_demos": 10}


def make_world(root, seed=0, flip=(), noise_scale=0.15, drop_last=True, cond_drop_last=True, n_files=2, n_demo=50,
               T=16, lag=False, cond_meta_patch=None, nan_in=None, trval_meta_patch=None, poison_test=True):
    """Synthetic raw files, re-rendered phase-A features, five test-condition roots and a stored-feature root.

    Latent (u, v): D sees only u, every partner sees v (extra action information) plus a little u.
    Re-rendered features = stored features (first T-1 frames) + small noise; the four perturbed types add
    noise_scale * sd noise to every encoder, and negate the features of the (encoder, camera index) pairs in flip.
    """
    root = Path(root)
    rng = np.random.default_rng(seed)
    stems = [f"task{i}" for i in range(n_files)]
    mix = {}
    for e, d in DIMS.items():
        for c in (0, 1):
            m = np.zeros((8, d))
            if e == "dinov3":
                m[:4] = rng.normal(size=(4, d))
            else:
                m[4:] = rng.normal(size=(4, d)); m[:4] = 0.3 * rng.normal(size=(4, d))
            mix[(e, c)] = m
    A, B = rng.normal(size=(4, 7)), rng.normal(size=(4, 7))
    raw_dir, feats_dir, stored_dir = root / "raw", root / "trval", root / "stored"
    raw_dir.mkdir(parents=True)
    true_actions, cond_feats = {}, {c: {} for c in [P.REF] + P.TYPES}
    names = [f"demo_{i}" for i in range(n_demo)]
    part_of = D.demo_split(names)
    tr_len = T - 1 if drop_last else T
    cd_len = T - 1 if cond_drop_last else T
    for fi, stem in enumerate(stems):
        with h5py.File(raw_dir / (stem + ".hdf5"), "w") as fr:
            fe, fs = {}, None
            for e in DIMS:
                (feats_dir / e / SUITE).mkdir(parents=True, exist_ok=True)
                fe[e] = h5py.File(feats_dir / e / SUITE / (stem + ".h5"), "w")
                m = meta_for(e)
                if trval_meta_patch and fi >= 1:
                    m.update(trval_meta_patch)
                fe[e].attrs["meta"] = json.dumps(m)
            (stored_dir / "dinov3" / SUITE).mkdir(parents=True, exist_ok=True)
            fs = h5py.File(stored_dir / "dinov3" / SUITE / (stem + ".h5"), "w")
            fs.attrs["meta"] = json.dumps(meta_for("dinov3"))
            for i, dn in enumerate(names):
                part = part_of[dn]
                lat = rng.normal(size=(T, 8)) + 0.3 * rng.normal(size=(1, 8))
                act = lat[:, :4] @ A + lat[:, 4:] @ B + 0.1 * rng.normal(size=(T, 7))
                g = fr.create_group(f"data/{dn}")
                g["actions"] = act.astype(np.float32)
                g["obs/joint_states"] = rng.normal(size=(T, 7)).astype(np.float32)
                g["obs/gripper_states"] = rng.normal(size=(T, 2)).astype(np.float32)
                true_actions[(stem, dn)] = act.astype(np.float32)
                for e, f in fe.items():
                    for c, cam in enumerate(D.CAMS):
                        x = (lat @ mix[(e, c)] + 0.3 * rng.normal(size=(T, DIMS[e]))).astype(np.float32)
                        if e == "dinov3":
                            sx = x.copy()
                            if part == "test" and poison_test:
                                sx[:] = np.nan
                            fs.create_dataset(f"{dn}/{cam}/global", data=sx)
                        src = x[:tr_len]
                        if lag and e == "dinov3":  # re-render frame j is stored frame j-1 (off by one)
                            src = x[np.maximum(np.arange(tr_len) - 1, 0)]
                        y = (src + 0.05 * rng.normal(size=src.shape)).astype(np.float32)
                        if part == "test":
                            if poison_test:
                                f.create_dataset(f"{dn}/{cam}/global", data=np.full((tr_len, DIMS[e]), np.nan, np.float32))
                                f.create_dataset(f"{dn}/{cam}/mean", data=np.full((tr_len, 3), np.nan, np.float32))
                            for cn in [P.REF] + P.TYPES:
                                z = x[:cd_len] + 0.05 * rng.normal(size=(cd_len, x.shape[1])).astype(np.float32)
                                if cn != P.REF:
                                    z = z + noise_scale * x.std(0) * rng.normal(size=z.shape).astype(np.float32)
                                    if (e, c) in flip:
                                        z = -z
                                if nan_in == (cn, e, c):
                                    z[0, 0] = np.nan
                                cond_feats[cn].setdefault((e, stem, dn), []).append(z.astype(np.float32))
                        elif part in ("train", "val"):
                            f.create_dataset(f"{dn}/{cam}/global", data=y)
                            f.create_dataset(f"{dn}/{cam}/mean", data=y[:, :3])
            for f in fe.values():
                f.close()
            fs.close()
    cond_roots, ref_test = {}, {e: {c: [] for c in D.CAMS} for e in DIMS}
    for cn, items in cond_feats.items():
        cond_roots[cn] = root / "cond" / cn
        for (e, stem, dn), out in items.items():
            p = cond_roots[cn] / e / SUITE
            p.mkdir(parents=True, exist_ok=True)
            with h5py.File(p / (stem + ".h5"), "a") as f:
                if "meta" not in f.attrs:
                    m = meta_for(e)
                    for k_, v_ in (cond_meta_patch or {}).items():
                        if v_ is None:
                            m.pop(k_)
                        else:
                            m[k_] = v_
                    f.attrs["meta"] = json.dumps(m)
                for cam, y in zip(D.CAMS, out):
                    f.create_dataset(f"{dn}/{cam}/global", data=y)
                    f.create_dataset(f"{dn}/{cam}/mean", data=y[:, :3])
                    if cn == P.REF:
                        ref_test[e][cam].append(y)
    ref_test = {e: {c: np.concatenate(v) for c, v in cv.items()} for e, cv in ref_test.items()}
    n_tv = tr_len
    expected = {"train": n_files * 30 * n_tv, "val": n_files * 10 * n_tv, "test": n_files * 10 * (T - 1)}
    # Reference metas in the smoke-report format (q23.details.<enc>.meta): the tally cache's, unpatched.
    (root / "ref_meta.json").write_text(json.dumps({"q23": {"details": {e: {"meta": meta_for(e)} for e in DIMS}}}))
    return {"feats": feats_dir, "raw": raw_dir, "stored": stored_dir, "ref_meta": root / "ref_meta.json", "conds": cond_roots, "actions": true_actions,
            "stems": stems, "ref_test": ref_test, "expected": expected, "T": T}


@contextlib.contextmanager
def gate_ref(v):
    old = P.GATE_REF
    P.GATE_REF = v
    try:
        yield
    finally:
        P.GATE_REF = old


def run_dev(w, out, **kw):
    kw.setdefault("stored_root", w["stored"])
    kw.setdefault("ref_meta_path", w["ref_meta"])
    return P.dev(w["feats"], w["raw"], SUITE, out, expected_frames=w["expected"], quiet=True, **kw)


def run_final(w, out, report, n_boot=200, **kw):
    kw.setdefault("expected_frames", w["expected"])
    kw.setdefault("ref_meta_path", w["ref_meta"])
    return P.run(w["feats"], w["raw"], SUITE, w["conds"], report, out, n_boot=n_boot, min_boot=100, quiet=True, **kw)


def full_run(w, out, dev_kw=None, **kw):
    """Dev (gate forced to pass) then final. Returns (final results, dev report, dev report path)."""
    out = Path(out)
    with gate_ref(PASS_REF):
        rep = run_dev(w, out.parent / (out.name + "_dev"), **(dev_kw or {}))
        path = out.parent / (out.name + "_dev") / "dev-report.json"
        res = run_final(w, out, path, **kw)
    return res, rep, path


def rd(draws_point, draws):
    """Helper: row 0 = point estimate, then draws."""
    return np.concatenate([[draws_point], draws])


def expect(fn, exc, text=None):
    try:
        fn()
    except exc as e:
        assert text is None or text in str(e), (text, str(e))
        return
    raise AssertionError(f"expected {exc.__name__}")


# ---------------------------------------------------------------------------

def test_rules_unit():
    rng = np.random.default_rng(0)
    nb = 400

    def arr(point, sd=0.01):
        return rd(point, point + sd * rng.normal(size=nb))

    # P: R clearly largest -> fires; next-largest is picked by point estimate.
    e = {"R": arr(0.30), "L": arr(0.05), "S": arr(0.10), "G": arr(0.02)}
    r = P.rule_p(e)
    assert r["R_is_largest"] and r["next_largest"] == "S" and r["fires"] and r["diff_ci95"][0] > 0.15, r
    # R largest at the point but the gap to the runner-up is within noise -> not a hit.
    e = {"R": arr(0.101, sd=0.05), "L": arr(0.10, sd=0.05), "S": arr(0.0), "G": arr(0.0)}
    e["R"] = e["L"] + 0.001 + 0.0 * e["R"]
    e["R"][1:] = e["L"][1:] + 0.001 + 0.05 * rng.normal(size=nb)
    r = P.rule_p(e)
    assert r["R_is_largest"] and r["diff_ci95"][0] < 0 and not r["fires"], r
    # Exact tie at the point estimate: R is not strictly largest -> no hit even if draws favour R.
    e = {"R": arr(0.2), "L": arr(0.2), "S": arr(0.0), "G": arr(0.0)}
    e["R"][1:] = 0.3 + 0.01 * rng.normal(size=nb)
    r = P.rule_p(e)
    assert not r["R_is_largest"] and not r["fires"], r
    # R is not the largest at all.
    e = {"R": arr(-0.2), "L": arr(0.1), "S": arr(0.0), "G": arr(0.0)}
    assert not P.rule_p(e)["fires"]
    print("rule P edge cases ok")

    # B: R's clause is CI based.
    g = {"R": arr(-0.05), "L": arr(0.08), "S": arr(0.07), "G": arr(0.06)}
    b = P.rule_b(g)
    assert b["R_clause"] and b["fires_point_reading"] and b["fires_ci_reading_strict"], b
    # R below 0 at the point but its CI covers 0 -> R clause fails.
    g = {"R": arr(-0.01, sd=0.05), "L": arr(0.03), "S": arr(0.02), "G": arr(0.01)}
    b = P.rule_b(g)
    assert not b["R_clause"] and not b["fires_point_reading"], b
    # Another partner slightly below 0 with a CI that covers 0: both readings fail (only two readings exist).
    g = {"R": arr(-0.05), "L": arr(0.03), "S": arr(-0.005, sd=0.05), "G": arr(0.01)}
    g["S"][1:] = -0.005 + 0.05 * rng.normal(size=nb)
    b = P.rule_b(g)
    assert b["R_clause"] and not b["fires_point_reading"] and not b["fires_ci_reading_strict"], b
    assert "fires_ci_reading_lenient" not in b and "others_ci_lenient" not in b
    # Another partner at exactly 0 at the point (>= 0) but a CI reaching below 0.
    g = {"R": arr(-0.05), "L": arr(0.03), "S": arr(0.0, sd=0.05), "G": arr(0.01)}
    g["S"][0] = 0.0
    b = P.rule_b(g)
    assert b["R_clause"] and b["fires_point_reading"] and not b["fires_ci_reading_strict"], b
    # Another partner clearly below 0: no reading fires.
    g = {"R": arr(-0.05), "L": arr(-0.2), "S": arr(0.02), "G": arr(0.01)}
    b = P.rule_b(g)
    assert b["R_clause"] and not (b["fires_point_reading"] or b["fires_ci_reading_strict"]), b
    print("rule B edge cases (point and CI-strict readings) ok")

    o = P.ordering({"R": arr(0.3), "L": arr(0.1), "S": arr(0.2), "G": arr(0.0)})
    assert o["ours_by_e_point"] == ["R", "S", "L", "G"] and o["paper_order"] == ["L", "S", "G", "R"], o
    print("ordering beside the paper's ok")


def test_quantities_known():
    """Hand-built per-demo stats where the answer is known exactly: pooled e is the equal-weight mean over types."""
    k, rng = 6, np.random.default_rng(1)
    n = np.full(k, 10.0)
    sy, syy = np.zeros((k, 7)), np.full((k, 7), 10.0)  # ss_tot = 10 * k per dim at full weight

    def stats(r2v):  # sres chosen so the R2 over all demos is exactly r2v
        return (n, sy, syy, np.full((k, 7), 10.0 * (1 - r2v)))

    r2s = {P.REF: {"D": 0.5, "D+R": 0.6, "D+L": 0.58}}
    drops = {"camera": 0.1, "light": 0.0, "background": 0.2, "noise": 0.1}  # D's drops
    extra_R = {"camera": 0.3, "light": 0.1, "background": 0.0, "noise": 0.2}  # D+R drops this much more
    for t in P.TYPES:
        r2s[t] = {"D": 0.5 - drops[t], "D+R": 0.6 - drops[t] - extra_R[t], "D+L": 0.58 - drops[t]}
    st = {c: {p: stats(v) for p, v in ps.items()} for c, ps in r2s.items()}
    q = P.quantities(st, P.make_weights(k, 50, 0))
    assert np.isclose(q["pooled_e"]["R"][0], np.mean(list(extra_R.values())))
    assert np.isclose(q["pooled_e"]["L"][0], 0.0)
    assert np.isclose(q["g"]["camera"]["R"][0], 0.1 - 0.3) and np.isclose(q["pooled_g"]["L"][0], 0.08)
    assert np.isclose(q["e"]["noise"]["R"][0], 0.2)
    # Missing a type must fail loudly rather than pool three.
    bad = {c: v for c, v in st.items() if c != "noise"}
    try:
        P.quantities(bad, P.make_weights(k, 50, 0))
        raise AssertionError("expected KeyError")
    except KeyError:
        pass
    print("drop / e / gain / pooled quantities on hand-built stats ok")


def test_bootstrap_pairing():
    w1, w2 = P.make_weights(20, 300, seed=5), P.make_weights(20, 300, seed=5)
    assert np.array_equal(w1, w2) and np.all(w1[0] == 1) and np.all(w1.sum(1) == 20)
    assert not np.array_equal(w1, P.make_weights(20, 300, seed=6))
    # Paired differences of two strongly related probes are far tighter than if the draws were independent.
    rng = np.random.default_rng(2)
    k = 30
    y = rng.normal(size=(k * 20, 7)); groups = np.repeat(np.arange(k), 20)
    p1 = y + rng.normal(size=y.shape); p2 = p1 + 0.1 * rng.normal(size=y.shape)
    W = P.make_weights(k, 400, 0)
    a = D.r2_from_stats(W, *D.demo_stats(y, p1, groups)); b = D.r2_from_stats(W, *D.demo_stats(y, p2, groups))
    paired = (b - a)[1:].std()
    indep = np.sqrt(a[1:].var() + b[1:].var())
    assert paired < 0.3 * indep, (paired, indep)
    print(f"bootstrap pairing: paired diff sd {paired:.4f} vs independent {indep:.4f}")
    # The floor on draws.
    with tempfile.TemporaryDirectory() as t:
        try:
            P.run(t + "/f", t + "/r", SUITE, {}, t + "/t", t + "/out", n_boot=999)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    print("n_boot floor (1,000) enforced ok")


def test_drift_known():
    rng = np.random.default_rng(3)
    k, per, d = 5, 40, 12
    groups = np.repeat(np.arange(k), per)
    z = rng.normal(size=(k * per, d)).astype(np.float32)
    shift = np.zeros(d, np.float32); shift[:3] = [1.0, 2.0, 2.0]  # ||shift||^2 = 9
    num, den = P.drift_parts(z, z + shift, groups, k)
    expect = 9.0 / ((z - z.mean(0)) ** 2).sum(1).mean()
    got = P.drift_draws(num, den, np.ones((1, k)))[0]
    assert np.isclose(got, expect, rtol=1e-4), (got, expect)
    assert abs(got - 9.0 / d) < 0.8  # unit-variance features: variance sum is about d
    num0, den0 = P.drift_parts(z, z, groups, k)
    assert P.drift_draws(num0, den0, np.ones((1, k)))[0] == 0
    # Bootstrap weights: a resample that drops demos changes the ratio only through those demos.
    W = P.make_weights(k, 100, 0)
    dd = P.drift_draws(num, den, W)
    assert np.all(np.isfinite(dd)) and dd[1:].std() > 0
    print(f"drift on a known shift: {got:.3f} (expected {expect:.3f}); zero for no shift ok")


def test_transform_and_pca():
    rng = np.random.default_rng(4)
    n, d = 600, 30
    base = rng.normal(size=(n, d)).astype(np.float32)
    base[:, 0] *= 5.0
    base[:, 1] *= 4.0
    train = np.arange(n) < 400
    tf = P.CamTransform(base[train], reduce_to=2)
    std = D.standardize(base, train)
    assert np.allclose(tf.standardize(base), std, atol=1e-5), "must equal diagnostics.standardize"
    # With standardization the two big columns lose their size advantage; make a correlated pair instead.
    c = rng.normal(size=(n, 1)).astype(np.float32)
    x = rng.normal(size=(n, d)).astype(np.float32) * 0.3
    x[:, 3] += 3 * c[:, 0]; x[:, 4] += 3 * c[:, 0]
    tf = P.CamTransform(x[train], reduce_to=1)
    ax = tf.pca[1][:, 0]
    assert abs(ax[3]) > 0.6 and abs(ax[4]) > 0.6 and abs(abs(ax[3]) - abs(ax[4])) < 0.15, ax
    # Fit on train only: changing val/test rows (or adding a huge test-only direction) changes nothing.
    x2 = x.copy()
    x2[~train, 10] += 1000 * rng.normal(size=(~train).sum())
    tf2 = P.CamTransform(x2[train], reduce_to=1)
    assert np.array_equal(tf.pca[1], tf2.pca[1]) and np.array_equal(tf.mu, tf2.mu) and np.array_equal(tf.sd, tf2.sd)
    # Applied unchanged to other data: the map is a fixed function of its input.
    assert np.array_equal(tf(x[:5]), tf2(x[:5]))
    # Fitting on all rows would have been different (the leak this guards against).
    tf3 = P.CamTransform(x2, reduce_to=1)
    assert not np.allclose(np.abs(tf3.pca[1][:, 0]), np.abs(ax), atol=0.05)
    assert tf.dim_out == 1 and P.CamTransform(x[train]).dim_out == d
    # Matched plan.
    t, red = P.matched_plan({"dinov3": 1024, "siglip2": 1024, "sam3": 1024, "c-radiov4": 2304, "vggt-omega": 2048})
    assert t == 1024 and red == ["c-radiov4", "vggt-omega"]
    t, red = P.matched_plan({"dinov3": 768, "siglip2": 1024, "sam3": 256, "c-radiov4": 1024, "vggt-omega": 256})
    assert t == 256 and red == ["c-radiov4", "dinov3", "siglip2"]
    assert P.matched_plan({"a": 5, "b": 5})[1] == []
    try:
        P.matched_plan({"dinov3": 1024, "sam3": 1024}, {"sam3": 1000})
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    print("standardize == diagnostics.standardize; PCA train-only, planted axis recovered; matched plan ok")



# ---------------------------------------------------------------------------
# Loader: split from all raw names, n-1 frames, counts, test guard
# ---------------------------------------------------------------------------

def test_loader_split_and_labels():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        w = make_world(t / "w", seed=31)
        # The phase-A files hold only the 40 train+val demos' real features (test demos are NaN-poisoned).
        # A split computed from the feature file's demos alone would be 24/8/8; ours comes from the raw names.
        with h5py.File(w["feats"] / "dinov3" / SUITE / "task0.h5", "r") as f:
            assert len(f.keys()) == 50 and np.isnan(f["demo_45"][D.CAMS[0]]["global"][:]).all()
        tr = P.load_rendered(w["feats"], w["raw"], SUITE, list(DIMS), ["global"], expected_frames=w["expected"])
        val_demos = sorted({d for _, d, _, part in tr.index if part == "val"}, key=P._dnum)
        train_demos = sorted({d for _, d, _, part in tr.index if part == "train"}, key=P._dnum)
        assert val_demos == P.VAL_DEMOS and train_demos == P.TRAIN_DEMOS, (val_demos, train_demos)
        assert not any(part == "test" for *_, part in tr.index) and set(tr.split) == {"train", "val"}
        Tr = w["T"] - 1
        assert tr.counts == {"train": 2 * 30 * Tr, "val": 2 * 10 * Tr}
        # Labels are actions[:n-1] of the raw file, in index order.
        want = np.concatenate([w["actions"][(s, d)][:-1] for s, d, _, _ in tr.index])
        assert np.allclose(tr.actions, want)
        # Features are the written ones.
        with h5py.File(w["feats"] / "siglip2" / SUITE / "task1.h5", "r") as f:
            first = [i for i, (s, d, _, _) in enumerate(tr.index) if s == "task1"][0]
            off = sum(T_ for _, _, T_, _ in tr.index[:first])
            assert np.array_equal(tr.feats["siglip2"]["global"][D.CAMS[1]][off:off + Tr], f["demo_0"][D.CAMS[1]]["global"][:])
        # The default expected counts are the real ones (38,164 / 12,316 / 12,748): a synthetic world must fail them.
        assert P.EXPECTED_FRAMES == {"train": 38164, "val": 12316, "test": 12748}
        expect(lambda: P.load_rendered(w["feats"], w["raw"], SUITE, list(DIMS), ["global"]), ValueError, "expected 38164")
        # A raw file with 40 demos gives the silent 24/8/8 split under diagnostics' rule: refused before any read.
        w40 = make_world(t / "w40", seed=32, n_demo=40, n_files=1)
        expect(lambda: P.load_rendered(w40["feats"], w40["raw"], SUITE, list(DIMS), ["global"], expected_frames=None),
               ValueError, "demo_30..demo_39")
        # ... and the split check comes first: with a feature file missing the error is still the split error.
        (w40["feats"] / "sam3" / SUITE / "task0.h5").unlink()
        expect(lambda: P.load_rendered(w40["feats"], w40["raw"], SUITE, list(DIMS), ["global"], expected_frames=None),
               ValueError, "demo_30..demo_39")
        # 45 demos give val demo_27..35: also refused.
        expect(lambda: P.file_split([f"demo_{i}" for i in range(45)], "x"), ValueError)
        P.file_split([f"demo_{i}" for i in range(50)], "ok")
        # A feature file missing a train/val demo, and one with n frames instead of n-1.
        w2 = make_world(t / "w2", seed=33, n_files=1)
        with h5py.File(w2["feats"] / "sam3" / SUITE / "task0.h5", "a") as f:
            del f["demo_7"]
        expect(lambda: P.load_rendered(w2["feats"], w2["raw"], SUITE, list(DIMS), ["global"], expected_frames=None), KeyError, "demo_7")
        w3 = make_world(t / "w3", seed=34, n_files=1, drop_last=False)
        expect(lambda: P.load_rendered(w3["feats"], w3["raw"], SUITE, list(DIMS), ["global"], expected_frames=None), ValueError, "n-1")
        print("loader: split from all raw names (val = demo_30..39 on a 40-demo feature file), labels, counts, n-1 ok")


def test_dev_never_opens_test():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        w = make_world(t / "w", seed=35, n_files=1)
        # The loader refuses a test phase outright.
        expect(lambda: P.load_rendered(w["feats"], w["raw"], SUITE, list(DIMS), ["global"], phases=("train", "val", "test"),
                                       expected_frames=None), P.NoTestAccessError)
        # The dev entry reads DEV_PHASES; if that ever included test the run would raise and leave no dir behind.
        old = P.DEV_PHASES
        P.DEV_PHASES = ("train", "val", "test")
        try:
            expect(lambda: run_dev(w, t / "o_guard"), P.NoTestAccessError)
        finally:
            P.DEV_PHASES = old
        assert not (t / "o_guard").exists()
        # Stored reads refuse a test demo too.
        expect(lambda: P.load_stored(w["stored"], SUITE, "dinov3", "global", [("task0", "demo_45", 15, "test")],
                                     meta_for("dinov3")), P.NoTestAccessError)
        # With NaN-poisoned test demos in every phase-A file and in the stored root, dev still runs: nothing opened them.
        rep = run_dev(w, t / "o_ok", stored_root=w["stored"])
        assert rep["n_demos"] == 40 and rep["lag_check"]["parts"] == ["train", "val"] and rep["lag_check"]["demos_checked"] == 40
        print("dev entry: test phase refused, guard raises and cleans up, poisoned test demos never read ok")


# ---------------------------------------------------------------------------
# Gate (check A) and the dev report
# ---------------------------------------------------------------------------

def test_gate():
    assert P.GATE_REF == 0.6562 and P.GATE_MARGIN == 0.025 and abs(P.gate_threshold() - 0.6312) < 1e-12
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        w = make_world(t / "w", seed=36, n_files=2)
        rep = run_dev(w, t / "d0")                     # the real constants: a synthetic world misses 0.6312
        g = rep["gate"]
        s = g["score"]
        assert g["reference"] == 0.6562 and g["margin"] == 0.025 and "probe-report.json" in g["reference_source"]
        assert g["passed"] is (s >= 0.6312)
        # The scored value is the train-only best-alpha curve value, not the train+val refit.
        tr = P.load_rendered(w["feats"], w["raw"], SUITE, ["dinov3"], ["global"], expected_frames=w["expected"])
        train, val = tr.split == "train", tr.split == "val"
        x = D.standardize(np.concatenate([tr.feats["dinov3"]["global"][c] for c in D.CAMS], 1), train)
        alpha, curve = P.tune_alpha(x, tr.actions, tr.split)
        assert g["alpha"] == alpha and np.isclose(s, max(curve.values()), atol=1e-6) and np.isclose(s, curve[alpha], atol=1e-6)
        trva = train | val
        wf, bf = D.ridge_fit(x[trva], tr.actions[trva], alpha)
        refit = float(D.r2(tr.actions[val], x[val] @ wf + bf).mean())
        assert refit > s + 1e-4, (refit, s)            # the refit has seen val: scoring it would be the wrong number
        assert rep["configs"]["full"]["combined"]["probes"]["D"]["val_r2"] == s
        # The pass mark: at or above passes, just below fails (the reference is moved so threshold = s -/+ 1e-4).
        with gate_ref(s + P.GATE_MARGIN - 1e-4):
            assert run_dev(w, t / "d_pass")["gate"]["passed"] is True
        with gate_ref(s + P.GATE_MARGIN + 1e-4):
            r_fail = run_dev(w, t / "d_fail")
            assert r_fail["gate"]["passed"] is False and r_fail["gate"]["score"] == s
        # The gate always scores the global features, whatever ftype the probes use.
        with gate_ref(PASS_REF):
            rm = run_dev(w, t / "d_mean", ftype="mean")
        assert rm["gate"]["score"] == s and rm["ftype"] == "mean"
        # Dev report content: every probe tuned, val R2 and clean val gains for the four pairs, all configs and views.
        for config, views in rep["configs"].items():
            for view, cv in views.items():
                assert set(cv["probes"]) == {"D", "D+R", "D+L", "D+S", "D+G"} and set(cv["val_gain"]) == set("RLSG")
                for name, info in cv["probes"].items():
                    curve_ = {float(a): v for a, v in info["val_r2_by_alpha"].items()}
                    assert info["alpha_source"] == "tuned-on-rerendered-val" and info["alpha"] in D.ALPHAS
                    assert info["alpha"] == max(curve_, key=lambda a: (curve_[a], -D.ALPHAS.index(a)))   # first best wins
                    assert info["val_r2"] == curve_[info["alpha"]]
                for b in "RLSG":
                    assert np.isclose(cv["val_gain"][b], cv["probes"]["D+" + b]["val_r2"] - cv["probes"]["D"]["val_r2"])
                    assert cv["val_gain"][b] > 0.05, (config, view, b)   # partners carry extra information in this world
        assert set(rep["configs"]) == {"full", "matched"}
        # Optional references are skipped with a note when their input is absent.
        assert rep["lag_check"]["passed"] is True and rep["cross_value"] is not None and "skipped" in rep["notes"]["tally_reference"]
        # Failing dev run: CLI-level contract is gate.passed False with the report written.
        assert (t / "d_fail" / "dev-report.json").exists()
        # Cross value (re-render-trained D scored on stored val features) when a stored root is given.
        with gate_ref(PASS_REF):
            rc = run_dev(w, t / "d_cross", stored_root=w["stored"])
        cv_ = rc["cross_value"]
        assert cv_["val_r2_rerender_features"] == s and np.isfinite(cv_["val_r2_stored_features"]) and cv_["val_r2_stored_features"] > 0.1
        # The stored features here are the re-render's source plus small noise, so the two are close.
        assert abs(cv_["val_r2_stored_features"] - s) < 0.1
        print(f"gate: scored {s:.4f} = train-only curve value (refit would be {refit:.4f}); passes at/above, fails just below; "
              "dev report complete ok")


def make_tally(d, gains, other_suite=True):
    """Synthetic tally JSONs with val curves whose value at the chosen alpha gives the stated gains."""
    d.mkdir(parents=True)
    curve = lambda v: {"0.01": v - 0.1, "0.1": v, "1": v - 0.2, "10": v - 0.3, "100.0": v - 0.4, "1000.0": v - 0.5,
                       "10000.0": v - 0.6, "100000.0": v - 0.7}
    for b, enc in P.PARTNERS.items():
        key = f"{min(P.DINO, enc)}+{max(P.DINO, enc)}"
        probes = {"dinov3|global": {"alpha": 0.1, "val_r2_by_alpha": curve(0.5)},
                  f"{key}|global": {"alpha": 0.1, "val_r2_by_alpha": curve(0.5 + gains[b])}}
        (d / f"tally-{SUITE}-{P.DINO}-{enc}.json").write_text(json.dumps({"suite": SUITE, "probes": probes}))
    if other_suite:
        (d / "tally-other-dinov3-sam3.json").write_text(json.dumps(
            {"suite": "libero_other", "probes": {"dinov3|global": {"alpha": 0.1, "val_r2_by_alpha": curve(9.0)}}}))


def test_tally_reference_is_optional():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        gains = {"R": 0.04, "G": 0.03, "S": 0.02, "L": 0.01}
        make_tally(t / "tally", gains)
        got = P.reference_tally_val_gains(t / "tally", SUITE, "global")
        assert all(abs(got[b] - gains[b]) < 1e-12 for b in gains), got
        w = make_world(t / "w", seed=37, n_files=2)
        with gate_ref(PASS_REF):
            rep = run_dev(w, t / "d", tally_dir=t / "tally")
            ref = rep["tally_reference"]
            assert ref["tally_order"] == ["R", "G", "S", "L"] and set(ref["ours_val_gain"]) == set("RLSG")
            # The tally changes nothing in our probes: same alphas, same val R2, same gate as without it.
            rep0 = run_dev(w, t / "d0")
        assert rep["configs"] == rep0["configs"] and rep["gate"] == rep0["gate"]
        print("tally val gains: shown beside ours, read-only, no effect on any alpha or on the gate ok")


# ---------------------------------------------------------------------------
# Final run: refuses without a passing dev report
# ---------------------------------------------------------------------------

def test_final_refuses_without_passing_dev():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        w = make_world(t / "w", seed=38, n_files=2)
        bad_conds = {c: t / "nowhere" for c in [P.REF] + P.TYPES}      # never reached if the gate refuses first

        def final_with(report, trval=None, conds=None, out="o"):
            return P.run(trval or w["feats"], w["raw"], SUITE, conds or bad_conds, report, t / out, n_boot=100, min_boot=100,
                         quiet=True, expected_frames=w["expected"], ref_meta_path=w["ref_meta"])

        # No report at all, or a file that is not a dev report.
        expect(lambda: final_with(t / "none.json"), P.GateError, "not found")
        (t / "junk.json").write_text(json.dumps({"kind": "other"}))
        expect(lambda: final_with(t / "junk.json"), P.GateError, "not a dev report")
        # A dev report whose gate failed (real constants on a synthetic world).
        rep = run_dev(w, t / "d_fail")
        assert rep["gate"]["passed"] is False
        expect(lambda: final_with(t / "d_fail" / "dev-report.json"), P.GateError, "did not pass")
        assert not (t / "o").exists(), "a refused run must leave no output dir"
        # Tampered reports: passed flipped to True with a failing score; changed constants; wrong root; wrong suite.
        with gate_ref(PASS_REF):
            run_dev(w, t / "d_ok")
        ok_path = t / "d_ok" / "dev-report.json"
        good = json.loads(ok_path.read_text())

        def write(mod):
            j = json.loads(json.dumps(good)); mod(j)
            p = t / f"tamper{len(list(t.glob('tamper*')))}.json"; p.write_text(json.dumps(j)); return p
        # The gate passed only under PASS_REF; under the real constants the report's constants differ from this code's.
        expect(lambda: final_with(ok_path), P.GateError, "constants")
        # Same constants but a score below threshold with passed forged to True.
        with gate_ref(PASS_REF):
            forged = write(lambda j: j["gate"].update(score=-5.0, passed=True))
            expect(lambda: final_with(forged), P.GateError, "below the threshold")
            # A passing report for ANOTHER train/val root is refused.
            w2 = make_world(t / "w2", seed=39, n_files=2)
            expect(lambda: final_with(ok_path, trval=w2["feats"]), P.GateError, "train/val root")
            expect(lambda: final_with(write(lambda j: j.update(suite="libero_other"))), P.GateError, "suite")
            expect(lambda: final_with(write(lambda j: j.update(pca_whiten=True))), P.GateError, "pca_whiten")
            expect(lambda: final_with(write(lambda j: j["n_frames"].update(train=1))), P.GateError, "frame counts")
        assert not (t / "o").exists()
        # With the passing report for the same root the run goes ahead.
        with gate_ref(PASS_REF):
            res = run_final(w, t / "o_good", ok_path)
        assert res["dev_gate"]["passed"] is True and res["trval_root"] == str(w["feats"].resolve())
        print("final run: refuses (no report, failed gate, forged score, changed constants, other root/suite) ok; passes with a good one")


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_e2e_planted_and_null():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        # Planted: R loses its information under every perturbation (features negated), others only get noise.
        wa = make_world(t / "a", seed=11, flip=[("c-radiov4", 0), ("c-radiov4", 1)])
        out = t / "a" / "out"
        res, dev_rep, dev_path = full_run(wa, out)
        full = res["configs"]["full"]["combined"]
        pe = {b: full["pooled_e"][b]["point"] for b in "RLSG"}
        print("planted pooled e:", {b: round(v, 3) for b, v in pe.items()})
        assert max(pe, key=pe.get) == "R" and pe["R"] > 0.3 > max(pe[b] for b in "LSG") + 0.1, pe
        assert full["rule_P"]["fires"], full["rule_P"]
        rb = full["rule_B"]
        assert rb["R_clause"] and rb["fires_point_reading"] and rb["fires_ci_reading_strict"], rb
        assert full["ordering"]["ours_by_e_point"][0] == "R"
        # Every probe's alpha was tuned on re-rendered val (no tally alpha anywhere), in every config and view.
        for config, views in res["configs"].items():
            for view, cv in views.items():
                for name, info in cv["probes"].items():
                    assert info["alpha_source"] == "tuned-on-rerendered-val" and "val_r2_by_alpha" in info, (config, view, name)
                    assert info["alpha"] == dev_rep["configs"][config][view]["probes"][name]["alpha"]
        assert res["alphas_match_dev"] is True
        assert not any("tally" in k for k in res) and "drift_pipeline_reference" not in res and "stored" not in json.dumps(res["cond_roots"])
        # Matched control exists, reduces only R and G, and gives the same verdict on this planted case.
        assert res["matched"]["reduced"] == ["c-radiov4", "vggt-omega"] and res["matched"]["target"] == 8
        m = res["configs"]["matched"]["combined"]
        assert m["probes"]["D+R"]["dim"] == 32 and m["probes"]["D+L"]["dim"] == 32 and m["probes"]["D+G"]["dim"] == 32
        assert full["probes"]["D+R"]["dim"] == 44 and full["probes"]["D+G"]["dim"] == 40
        assert m["rule_P"]["fires"] and m["rule_B"]["fires_point_reading"]
        # Both cameras were corrupted, so agentview and wrist alone show it too.
        assert res["configs"]["full"]["agentview"]["rule_P"]["fires"] and res["configs"]["full"]["wrist"]["rule_P"]["fires"]
        # Output files, and labels: the last frame of each test demo was dropped (T-1 frames).
        assert (out / "results.json").exists() and (out / "stats.npz").exists() and (out / "preds.npz").exists()
        pr = np.load(out / "preds.npz")
        ys = np.concatenate([wa["actions"][(s, f"demo_{i}")][:-1] for s in wa["stems"] for i in range(40, 50)])
        assert np.allclose(pr["y_test"], ys), "labels must be actions[:T-1] of the test demos in order"
        Tr = wa["T"] - 1
        assert res["n_test_demos"] == 20 and res["n_test_frames"] == 20 * Tr
        assert res["n_trainval_frames"] == {"train": 2 * 30 * Tr, "val": 2 * 10 * Tr}
        # Recomputing from the saved sufficient statistics reproduces the pooled numbers (no refit needed).
        stz = np.load(out / "stats.npz")
        W = stz["boot_weights"].astype(float)
        st = {c: {p: tuple(stz[f"full/combined/{c}/{p}/{nm}"] for nm in ("n", "sy", "syy", "sres"))
                  for p in ["D", "D+R", "D+L", "D+S", "D+G"]} for c in [P.REF] + P.TYPES}
        q = P.quantities(st, W)
        assert np.isclose(q["pooled_e"]["R"][0], pe["R"]) and np.allclose(W, P.make_weights(20, 200, 0))
        assert P.stat(q["pooled_e"]["R"]) == full["pooled_e"]["R"]
        assert W.shape == (201, 20) and res["seed"] == 0 and res["n_boot"] == 200
        # The clean re-render R2 of D matches an independent fit (diagnostics' own functions) on re-rendered train+val
        # features at the tuned alpha, scored on the clean re-rendered test features.
        tr = P.load_rendered(wa["feats"], wa["raw"], SUITE, ["dinov3"], ["global"], expected_frames=wa["expected"])
        xall = np.concatenate([tr.feats["dinov3"]["global"][c] for c in D.CAMS], 1)
        mu, sd = xall[tr.split == "train"].mean(0), xall[tr.split == "train"].std(0) + 1e-6
        a_ = full["probes"]["D"]["alpha"]
        w_, b_ = D.ridge_fit((xall - mu) / sd, tr.actions, a_)
        xt = np.concatenate([wa["ref_test"]["dinov3"][c] for c in D.CAMS], 1)
        want = float(D.r2(ys, ((xt - mu) / sd) @ w_ + b_).mean())
        assert np.isclose(full["r2"][P.REF]["D"]["point"], want, atol=1e-4), (full["r2"][P.REF]["D"]["point"], want)
        # Drift: perturbed types move features; the flipped encoder drifts the most (negated: about 4x the variance moves).
        dr = res["drift"]["combined"]["dinov3"]
        assert set(dr) == set(P.TYPES) and dr["noise"]["ci95"][0] > 0.0 and "re-rendered clean-train" in res["drift_note"]
        dd = {e: res["drift"]["combined"][e]["camera"]["point"] for e in DIMS}
        assert max(dd, key=dd.get) == "c-radiov4", dd
        # PCA explained variance is recorded for the reduced encoders only.
        pv = res["pca_explained_variance"]["matched"]["combined"]
        assert set(pv) == {"c-radiov4", "vggt-omega"} and all(0 < v["explained_variance_ratio"] <= 1
                                                              for e_ in pv.values() for v in e_.values())
        assert res["pca_explained_variance"]["full"]["combined"] == {}
        # Never overwrite.
        before = (out / "results.json").read_text()
        with gate_ref(PASS_REF):
            expect(lambda: run_final(wa, out, dev_path), FileExistsError)
        assert (out / "results.json").read_text() == before
        print("planted case: rule P and rule B fire (full and matched), alphas tuned, labels, saved stats, drift ok")

        # Nothing about the fit depends on the perturbed data: same clean data, different perturbations -> same clean numbers.
        wb = make_world(t / "b", seed=11, flip=[], noise_scale=1.5)
        rb2, _, _ = full_run(wb, t / "b" / "out")
        assert rb2["configs"]["full"]["combined"]["r2"][P.REF] == res["configs"]["full"]["combined"]["r2"][P.REF]
        assert rb2["configs"]["full"]["combined"]["probes"] == res["configs"]["full"]["combined"]["probes"]

        # Null: every partner is perturbed the same way, so R must not stand out.
        wn = make_world(t / "n", seed=12, flip=[])
        rn, _, dev_path_n = full_run(wn, t / "n" / "out")
        fn = rn["configs"]["full"]["combined"]
        print("null pooled e:", {b: round(fn["pooled_e"][b]["point"], 3) for b in "RLSG"})
        assert not fn["rule_P"]["fires"] and not fn["rule_B"]["fires_point_reading"], (fn["rule_P"], fn["rule_B"])
        assert not rn["configs"]["matched"]["combined"]["rule_P"]["fires"]
        print("null case: rule P and rule B do not fire ok")

        # Camera-specific damage: R is negated on the agentview only.
        wv = make_world(t / "v", seed=13, flip=[("c-radiov4", 0)])
        rv, _, _ = full_run(wv, t / "v" / "out")
        assert rv["configs"]["full"]["agentview"]["rule_P"]["fires"], rv["configs"]["full"]["agentview"]["rule_P"]
        assert not rv["configs"]["full"]["wrist"]["rule_P"]["fires"]
        assert rv["configs"]["full"]["wrist"]["pooled_e"]["R"]["point"] < 0.1
        print("views: agentview-only damage shows in agentview, not in wrist ok")

        # Secondary mean features through the same code; all sizes equal here, so no matched row is produced.
        # Drift is skipped for the mean features (not defined in the PLAN). A global-feature dev report gates it too.
        with gate_ref(PASS_REF):
            rm = run_final(wn, t / "n" / "out_mean", dev_path_n, ftype="mean", n_boot=100)
        assert "matched" not in rm["configs"] and rm["matched"]["reduced"] == [] and rm["ftype"] == "mean"
        assert rm["configs"]["full"]["combined"]["probes"]["D+R"]["alpha"] in D.ALPHAS
        assert rm["drift"] == {} and "skipped" in rm["drift_note"] and "not compared" in rm["alphas_match_dev"]
        assert set(rn["frames_per_demo_dropped_last"]) == {1}
        print("mean features (no matched row when sizes agree, drift skipped) ok")

        # Stored root given to the final run: lag check on train and val only; poisoned stored test demos never read.
        with gate_ref(PASS_REF):
            rs = run_final(wn, t / "n" / "out_stored", dev_path_n, stored_root=wn["stored"])
        assert rs["lag_check"]["parts"] == ["train", "val"] and rs["configs"] == rn["configs"]


def test_alignment_meta_finite():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        # n frames per demo instead of n-1 (Q14: exactly n-1) in the phase-A root, and in a condition root.
        w = make_world(t / "n", seed=21, drop_last=False, n_files=2)
        expect(lambda: run_dev(w, t / "o1"), ValueError, "n-1")
        assert not (t / "o1").exists()
        w = make_world(t / "nc", seed=21, cond_drop_last=False, n_files=2)
        expect(lambda: full_run(w, t / "o1c"), ValueError, "n-1")
        # Off by one: re-render frame j equals stored frame j-1 (dev, stored root, train and val demos).
        w = make_world(t / "lag", seed=22, lag=True, n_files=2)
        with gate_ref(PASS_REF):
            expect(lambda: run_dev(w, t / "o2", stored_root=w["stored"]), ValueError, "alignment")
            expect(lambda: run_dev(w, t / "o2b", stored_root=None), ValueError, "stored-feature root")   # lag check is required
            assert not (t / "o2b").exists()
        assert not (t / "o2").exists()
        rng = np.random.default_rng(0)
        st = rng.normal(size=(21, 6))
        P.check_lag(st[:20] + 0.01 * rng.normal(size=(20, 6)), st, "ok")
        expect(lambda: P.check_lag(st[np.maximum(np.arange(20) - 1, 0)], st, "lag-1"), ValueError)
        expect(lambda: P.check_lag(st[1:21], st, "lag+1"), ValueError)
        # Metadata of the test conditions: a mismatch on each required key, and a missing key, fail.
        for k_, v_ in [("dtype", "bfloat16"), ("hf_id", "other"), ("resolution", 448), ("rotate180", False),
                       ("grid", 2), ("frame_stride", 2), ("cams", ["x"]), ("encoder", "zzz")]:
            w = make_world(t / ("m_" + k_), seed=23, n_files=1, cond_meta_patch={k_: v_})
            expect(lambda: full_run(w, t / ("om_" + k_)), ValueError, f"'{k_}'")
        w = make_world(t / "m_missing", seed=23, n_files=1, cond_meta_patch={"dtype": None})
        expect(lambda: full_run(w, t / "om_missing"), ValueError, "missing")
        # Metadata of the phase-A root: the second file disagrees with the first.
        w = make_world(t / "m_tv", seed=23, n_files=2, trval_meta_patch={"resolution": 448})
        expect(lambda: run_dev(w, t / "om_tv"), ValueError, "'resolution'")
        # Non-finite values in a condition file, and in the phase-A root.
        w = make_world(t / "nan", seed=24, n_files=1, nan_in=("noise", "sam3", 1))
        expect(lambda: full_run(w, t / "on"), ValueError, "non-finite")
        w = make_world(t / "nanc", seed=24, n_files=1)
        with h5py.File(w["feats"] / "siglip2" / SUITE / "task0.h5", "a") as f:
            f["demo_0"][D.CAMS[0]]["global"][0, 0] = np.inf
        expect(lambda: run_dev(w, t / "ond"), ValueError, "non-finite")
        print("alignment (n-1 only, lag check on train/val), meta, finiteness ok")


def test_misc_switches():
    # --sizes: unknown encoder names are rejected.
    expect(lambda: P.matched_plan({"dinov3": 8, "sam3": 8}, {"sam": 8}), ValueError, "unknown encoder")
    # Rule P ties: the conservative next-largest is used and dict order is irrelevant.
    rng = np.random.default_rng(7)
    nb = 400
    base = {"R": rd(0.5, 0.5 + 0.01 * rng.normal(size=nb)), "G": rd(0.0, rng.normal(size=nb) * 0.01)}
    tight = rd(0.1, 0.1 + 0.001 * rng.normal(size=nb))
    wide = rd(0.1, 0.1 + 0.2 * rng.normal(size=nb))  # same point estimate, much wider CI -> lower CI bound of the diff is lower
    r1 = P.rule_p({"R": base["R"], "L": tight, "S": wide, "G": base["G"]})
    r2_ = P.rule_p({"G": base["G"], "S": wide, "L": tight, "R": base["R"]})
    assert r1["next_largest"] == r2_["next_largest"] == "S" and r1["diff_ci95"] == r2_["diff_ci95"], (r1, r2_)
    # tune_alpha matches diagnostics.ridge_select's choice and returns the validation curve.
    rng = np.random.default_rng(8)
    n = 600
    split = np.array(["train"] * 360 + ["val"] * 120 + ["test"] * 120)
    x = rng.normal(size=(n, 20)).astype(np.float32)
    y = (x[:, :3] @ rng.normal(size=(3, 7)) + rng.normal(size=(n, 7))).astype(np.float32)
    a_, curve = P.tune_alpha(x, y, split)
    ref = D.ridge_select(x, y, split)
    assert a_ == ref[1] and np.allclose([curve[a] for a in D.ALPHAS], [ref[3][a] for a in D.ALPHAS], atol=1e-5)
    # Whitening switch: one flag; default is unwhitened. Whitened scores have unit variance on the fitting data.
    z = (rng.normal(size=(500, 1)) * 2 + 0.5 * rng.normal(size=(500, 10))).astype(np.float32)  # one strong shared factor
    tf_u, tf_w = P.CamTransform(z, 4, whiten=False), P.CamTransform(z, 4, whiten=True)
    assert P.PCA_WHITEN is False and np.array_equal(P.CamTransform(z, 4).pca[1], tf_u.pca[1])
    assert np.allclose(tf_w(z).std(0), 1, atol=0.05)
    assert not np.allclose(tf_u(z).std(0), 1, atol=0.2)
    print("sizes check, rule P tie handling, tune_alpha == ridge_select, whitening switch ok")


def test_input_errors():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        w = make_world(t / "w", seed=15, flip=[], n_files=2)
        with gate_ref(PASS_REF):
            run_dev(w, t / "d")
            dp = t / "d" / "dev-report.json"
            # A condition missing.
            conds = {c: v for c, v in w["conds"].items() if c != "light"}
            expect(lambda: P.run(w["feats"], w["raw"], SUITE, conds, dp, t / "o1", min_boot=100, n_boot=100, quiet=True,
                                 expected_frames=w["expected"], ref_meta_path=w["ref_meta"]), KeyError)
            assert not (t / "o1").exists(), "a refused run must not leave an output dir behind"
            # A condition whose frame count disagrees with the re-render.
            p = w["conds"]["camera"] / "sam3" / SUITE / "task0.h5"
            with h5py.File(p, "a") as f:
                name = sorted(f.keys())[0]
                x = f[name][D.CAMS[0]]["global"][:]
                del f[name][D.CAMS[0]]["global"]
                f[name][D.CAMS[0]].create_dataset("global", data=x[:-2])
            try:
                run_final(w, t / "o2", dp)
                raise AssertionError("expected ValueError")
            except ValueError as e:
                assert "expected" in str(e) or "frame" in str(e), e
            assert not (t / "o2").exists(), "a failed run must remove its claimed dir so a retry can reuse the path"
            # Declared sizes that disagree with the files.
            w2 = make_world(t / "w2", seed=16, flip=[], n_files=2)
            run_dev(w2, t / "d2")
            expect(lambda: run_final(w2, t / "o3", t / "d2" / "dev-report.json", sizes={"sam3": 1024}), ValueError, "declared size")
            expect(lambda: run_dev(w, t / "o4", sizes={"sam3": 1024}), ValueError, "declared size")
            # The n_boot floor is checked before anything else.
            expect(lambda: P.run(t / "f", t / "r", SUITE, {}, t / "nodev.json", t / "out", n_boot=999), ValueError, "1000")
        print("input errors (missing condition, frame mismatch, declared sizes, n_boot floor) refused ok")


def test_review_fixes():
    """Fix round after the independent review: real tally JSONs, required lag check, tally-cache metas,
    stale dev report, and the gate runs before any test label is read."""
    here = Path(P.__file__).resolve().parent.parent
    # 1. The real tally JSONs (result numbers only, no frames): the four DINOv3 val values differ in the 8th digit.
    real = here / "experiments" / "2026-10-19-feasibility-tally" / "results"
    got = P.reference_tally_val_gains(real, SUITE, "global")
    assert set(got) == set("RLSG") and all(np.isfinite(v) for v in got.values()), got
    # 3. The committed reference metas load for all five encoders and carry every META_KEYS key.
    rm = P.load_reference_meta()
    assert set(rm) == set(DIMS) and all(k in rm[e] for e in rm for k in P.META_KEYS)
    assert P.REF_META_PATH.is_file() and "results/smoke/report.json" in str(P.REF_META_PATH)
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        w = make_world(t / "w", seed=41, n_files=2)
        # 1b. A failing optional tally reference is recorded in the report; dev still writes it and keeps its outputs.
        make_tally(t / "tally", {"R": 0.04, "G": 0.03, "S": 0.02, "L": 0.01}, other_suite=False)
        f0 = t / "tally" / f"tally-{SUITE}-dinov3-sam3.json"
        j = json.loads(f0.read_text())
        j["probes"]["dinov3|global"]["val_r2_by_alpha"]["0.1"] += 0.1
        f0.write_text(json.dumps(j))
        expect(lambda: P.reference_tally_val_gains(t / "tally", SUITE, "global"), ValueError, "differs")
        with gate_ref(PASS_REF):
            rep = run_dev(w, t / "d", tally_dir=t / "tally")
        assert (t / "d" / "dev-report.json").exists() and "failed" in rep["notes"]["tally_reference"] and "tally_reference" not in rep
        # 3b. Metas: every re-rendered file of one encoder disagrees with the tally cache's reference -> dev and final fail.
        for k_, v_ in [("resolution", 448), ("dtype", "bfloat16")]:
            ref = json.loads(w["ref_meta"].read_text())
            ref["q23"]["details"]["sam3"]["meta"][k_] = v_
            bad = t / f"ref_{k_}.json"
            bad.write_text(json.dumps(ref))
            with gate_ref(PASS_REF):
                expect(lambda: run_dev(w, t / f"dm_{k_}", ref_meta_path=bad), ValueError, f"'{k_}'")
                expect(lambda: run_final(w, t / f"fm_{k_}", t / "d" / "dev-report.json", ref_meta_path=bad), ValueError, f"'{k_}'")
        ref = json.loads(w["ref_meta"].read_text())
        del ref["q23"]["details"]["siglip2"]
        (t / "ref_missing.json").write_text(json.dumps(ref))
        expect(lambda: P.load_reference_meta(t / "ref_missing.json"), KeyError, "siglip2")
        # 2. The final run refuses a dev report without a passed lag check; dev refuses to run without a stored root.
        with gate_ref(PASS_REF):
            run_dev(w, t / "d_ok")
        ok = t / "d_ok" / "dev-report.json"
        good = json.loads(ok.read_text())
        assert good["lag_check"]["passed"] is True

        def write(name, mod):
            jj = json.loads(json.dumps(good)); mod(jj)
            p = t / name; p.write_text(json.dumps(jj)); return p
        with gate_ref(PASS_REF):
            expect(lambda: run_final(w, t / "f1", write("nolag.json", lambda x: x.update(lag_check=None))), P.GateError, "lag check")
            expect(lambda: run_final(w, t / "f2", write("lagf.json", lambda x: x["lag_check"].update(passed=False))), P.GateError, "lag check")
            # 4. Stale report: the curve differs from what the given features give (also under ftype=mean).
            def stale(x):
                x["gate"]["val_r2_by_alpha"] = {a: v + 0.01 for a, v in x["gate"]["val_r2_by_alpha"].items()}
            sp = write("stale.json", stale)
            expect(lambda: run_final(w, t / "f3", sp), P.GateError, "stale")
            expect(lambda: run_final(w, t / "f4", sp, ftype="mean"), P.GateError, "stale")
            # 5. check_dev_report runs before any test label is read: with test-label reading rigged to blow up,
            # a bad report still gives the gate error, and a good one reaches the rigged reader.
            old = P.load_test_layout
            def boom(*a, **k):
                raise AssertionError("test labels read")
            P.load_test_layout = boom
            try:
                expect(lambda: run_final(w, t / "f5", t / "nonexistent.json"), P.GateError, "not found")
                expect(lambda: run_final(w, t / "f6", sp), P.GateError, "stale")
                expect(lambda: run_final(w, t / "f7", write("nolag2.json", lambda x: x.update(lag_check=None))), P.GateError)
                expect(lambda: run_final(w, t / "f8", ok), AssertionError, "test labels read")
            finally:
                P.load_test_layout = old
    print("review fixes: real tally JSONs, tally failure recorded, lag required, tally-cache metas, stale report, gate before labels ok")


def main():
    test_rules_unit()
    test_quantities_known()
    test_bootstrap_pairing()
    test_drift_known()
    test_transform_and_pca()
    test_loader_split_and_labels()
    test_dev_never_opens_test()
    test_gate()
    test_tally_reference_is_optional()
    test_final_refuses_without_passing_dev()
    test_e2e_planted_and_null()
    test_alignment_meta_finite()
    test_misc_switches()
    test_input_errors()
    test_review_fixes()
    print("all ok")


if __name__ == "__main__":
    main()
