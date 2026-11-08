"""Synthetic checks for pf_analysis.py (cases with known answers). Run: python test_pf_analysis.py

Needs numpy and h5py. Writes synthetic features in the real file layout under a temp dir and never reads
real files.
"""

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np

import diagnostics as D
import pf_analysis as P

SUITE = "libero_goal"
DIMS = {"dinov3": 8, "siglip2": 8, "sam3": 8, "c-radiov4": 14, "vggt-omega": 12}  # per camera
ALPHA_TALLY = {"dinov3": 0.1, "c-radiov4+dinov3": 1.0, "dinov3+siglip2": 0.1, "dinov3+sam3": 0.01,
               "dinov3+vggt-omega": 1.0}


def make_world(root, seed=0, flip=(), noise_scale=0.15, drop_last=True, n_files=5, n_demo=10, T=40):
    """Synthetic clean features, raw files, tally JSONs and five condition roots.

    Latent (u, v): D sees only u, every partner sees v (extra action information) plus a little u.
    Condition features = clean + tiny noise (re-render); the four perturbed types add noise_scale * sd
    noise to every encoder, and negate the features of the (encoder, camera index) pairs in flip.
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
    raw_dir, feats_dir = root / "raw", root / "feats"
    raw_dir.mkdir(parents=True); (root / "tally").mkdir()
    true_actions, cond_feats = {}, {c: {} for c in [P.REF] + P.TYPES}
    n_test_from = n_demo - max(1, round(n_demo * 0.2))
    for stem in stems:
        with h5py.File(raw_dir / (stem + ".hdf5"), "w") as fr:
            fe = {}
            for e in DIMS:
                (feats_dir / e / SUITE).mkdir(parents=True, exist_ok=True)
                fe[e] = h5py.File(feats_dir / e / SUITE / (stem + ".h5"), "w")
            for e, f in fe.items():
                f.attrs["meta"] = json.dumps({"encoder": e})
            for i in range(n_demo):
                lat = rng.normal(size=(T, 8)) + 0.3 * rng.normal(size=(1, 8))
                act = lat[:, :4] @ A + lat[:, 4:] @ B + 0.1 * rng.normal(size=(T, 7))
                g = fr.create_group(f"data/demo_{i}")
                g["actions"] = act.astype(np.float32)
                g["obs/joint_states"] = rng.normal(size=(T, 7)).astype(np.float32)
                g["obs/gripper_states"] = rng.normal(size=(T, 2)).astype(np.float32)
                true_actions[(stem, f"demo_{i}")] = act.astype(np.float32)
                for e, f in fe.items():
                    cams = []
                    for c, cam in enumerate(D.CAMS):
                        x = (lat @ mix[(e, c)] + 0.3 * rng.normal(size=(T, DIMS[e]))).astype(np.float32)
                        f.create_dataset(f"demo_{i}/{cam}/global", data=x)
                        cams.append(x)
                        f.create_dataset(f"demo_{i}/{cam}/mean", data=x[:, :3])
                    if i >= n_test_from:
                        Tc = T - 1 if drop_last else T
                        for cn in [P.REF] + P.TYPES:
                            out = []
                            for c, x in enumerate(cams):
                                y = x[:Tc] + 0.05 * rng.normal(size=(Tc, x.shape[1])).astype(np.float32)
                                if cn != P.REF:
                                    y = y + noise_scale * x.std(0) * rng.normal(size=y.shape).astype(np.float32)
                                    if (e, c) in flip:
                                        y = -y
                                out.append(y)
                            cond_feats[cn][(e, stem, f"demo_{i}")] = out
            for f in fe.values():
                f.close()
    cond_roots = {}
    for cn, items in cond_feats.items():
        cond_roots[cn] = root / "cond" / cn
        for (e, stem, d), out in items.items():
            p = cond_roots[cn] / e / SUITE
            p.mkdir(parents=True, exist_ok=True)
            with h5py.File(p / (stem + ".h5"), "a") as f:
                for cam, y in zip(D.CAMS, out):
                    f.create_dataset(f"{d}/{cam}/global", data=y)
                    f.create_dataset(f"{d}/{cam}/mean", data=y[:, :3])
    for b in ("c-radiov4", "siglip2", "sam3", "vggt-omega"):
        key = "+".join(sorted(["dinov3", b]))
        probes = {"dinov3|global": {"alpha": 0.1}, f"{key}|global": {"alpha": ALPHA_TALLY[key]},
                  "dinov3|mean": {"alpha": 1.0}, f"{key}|mean": {"alpha": 1.0}}
        (root / "tally" / f"tally-libero_goal-dinov3-{b}.json").write_text(json.dumps({"probes": probes}))
    return {"feats": feats_dir, "raw": raw_dir, "tally": root / "tally", "conds": cond_roots,
            "actions": true_actions, "stems": stems}


def run_world(w, out, n_boot=200, **kw):
    return P.run(w["feats"], w["raw"], SUITE, w["conds"], w["tally"], out, n_boot=n_boot, min_boot=100, quiet=True, **kw)


def rd(draws_point, draws):
    """Helper: row 0 = point estimate, then draws."""
    return np.concatenate([[draws_point], draws])


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
    assert b["R_clause"] and b["fires_point_reading"] and b["fires_ci_reading_strict"] and b["fires_ci_reading_lenient"], b
    # R below 0 at the point but its CI covers 0 -> R clause fails.
    g = {"R": arr(-0.01, sd=0.05), "L": arr(0.03), "S": arr(0.02), "G": arr(0.01)}
    b = P.rule_b(g)
    assert not b["R_clause"] and not b["fires_point_reading"], b
    # Another partner slightly below 0 with a CI that covers 0: the readings split three ways.
    g = {"R": arr(-0.05), "L": arr(0.03), "S": arr(-0.005, sd=0.05), "G": arr(0.01)}
    g["S"][1:] = -0.005 + 0.05 * rng.normal(size=nb)
    b = P.rule_b(g)
    assert b["R_clause"] and not b["fires_point_reading"] and not b["fires_ci_reading_strict"] and b["fires_ci_reading_lenient"], b
    # Another partner at exactly 0 at the point (>= 0) but a CI reaching below 0.
    g = {"R": arr(-0.05), "L": arr(0.03), "S": arr(0.0, sd=0.05), "G": arr(0.01)}
    g["S"][0] = 0.0
    b = P.rule_b(g)
    assert b["R_clause"] and b["fires_point_reading"] and not b["fires_ci_reading_strict"], b
    # Another partner clearly below 0: no reading fires.
    g = {"R": arr(-0.05), "L": arr(-0.2), "S": arr(0.02), "G": arr(0.01)}
    b = P.rule_b(g)
    assert b["R_clause"] and not (b["fires_point_reading"] or b["fires_ci_reading_strict"] or b["fires_ci_reading_lenient"]), b
    print("rule B edge cases (point / CI-strict / CI-lenient readings) ok")

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


def test_e2e_planted_and_null():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        # Planted: R loses its information under every perturbation (features negated), others only get noise.
        wa = make_world(t / "a", seed=11, flip=[("c-radiov4", 0), ("c-radiov4", 1)])
        out = t / "a" / "out"
        res = run_world(wa, out)
        full = res["configs"]["full"]["combined"]
        pe = {b: full["pooled_e"][b]["point"] for b in "RLSG"}
        print("planted pooled e:", {b: round(v, 3) for b, v in pe.items()})
        assert max(pe, key=pe.get) == "R" and pe["R"] > 0.3 > max(pe[b] for b in "LSG") + 0.1, pe
        assert full["rule_P"]["fires"], full["rule_P"]
        rb = full["rule_B"]
        assert rb["R_clause"] and rb["fires_point_reading"] and rb["fires_ci_reading_strict"], rb
        assert full["ordering"]["ours_by_e_point"][0] == "R"
        # alphas: the combined full probes use the tally's, with provenance.
        for name, key in [("D", "dinov3"), ("D+R", "c-radiov4+dinov3"), ("D+L", "dinov3+siglip2"),
                          ("D+S", "dinov3+sam3"), ("D+G", "dinov3+vggt-omega")]:
            info = full["probes"][name]
            want = 0.1 if name == "D" else ALPHA_TALLY[key]
            assert info["alpha"] == want and info["alpha_source"] == "tally", (name, info)
        assert res["configs"]["full"]["agentview"]["probes"]["D"]["alpha_source"] == "tuned-on-clean-val"
        # Matched control exists, reduces only R and G, and gives the same verdict on this planted case.
        assert res["matched"]["reduced"] == ["c-radiov4", "vggt-omega"] and res["matched"]["target"] == 8
        m = res["configs"]["matched"]["combined"]
        assert m["probes"]["D+R"]["dim"] == 32 and m["probes"]["D+L"]["dim"] == 32 and m["probes"]["D+G"]["dim"] == 32
        assert m["probes"]["D+L"]["alpha_source"] == "tally" and m["probes"]["D+R"]["alpha_source"] == "tuned-on-clean-val"
        assert full["probes"]["D+R"]["dim"] == 44 and full["probes"]["D+G"]["dim"] == 40
        assert m["rule_P"]["fires"] and m["rule_B"]["fires_point_reading"]
        # Both cameras were corrupted, so agentview and wrist alone show it too.
        assert res["configs"]["full"]["agentview"]["rule_P"]["fires"] and res["configs"]["full"]["wrist"]["rule_P"]["fires"]
        # Output files, and labels: the last frame of each test demo was dropped (T-1 frames).
        assert (out / "results.json").exists() and (out / "stats.npz").exists() and (out / "preds.npz").exists()
        pr = np.load(out / "preds.npz")
        ys = np.concatenate([wa["actions"][(s, f"demo_{i}")][:-1] for s in wa["stems"] for i in (8, 9)])
        assert np.allclose(pr["y_test"], ys), "labels must be actions[:T-1] of the test demos in order"
        assert res["n_test_demos"] == 10 and res["n_test_frames"] == 10 * 39
        # Recomputing from the saved sufficient statistics reproduces the pooled numbers (no refit needed).
        stz = np.load(out / "stats.npz")
        W = stz["boot_weights"].astype(float)
        st = {c: {p: tuple(stz[f"full/combined/{c}/{p}/{nm}"] for nm in ("n", "sy", "syy", "sres"))
                  for p in ["D", "D+R", "D+L", "D+S", "D+G"]} for c in [P.STORED, P.REF] + P.TYPES}
        q = P.quantities(st, W)
        assert np.isclose(q["pooled_e"]["R"][0], pe["R"]) and np.allclose(W, P.make_weights(10, 200, 0))
        assert P.stat(q["pooled_e"]["R"]) == full["pooled_e"]["R"]
        # Same draws for every probe: R2 of two probes is computed with one W (rows agree) - and the saved W has 201 rows.
        assert W.shape == (201, 10) and res["seed"] == 0 and res["n_boot"] == 200
        # The stored-clean R2 matches an independent fit with diagnostics' own functions (alpha from the tally).
        feats, actions, _, split, index, _, _ = P.load_clean(wa["feats"], wa["raw"], SUITE, ["dinov3"], "global")
        x = D.standardize(np.concatenate([feats["dinov3"]["global"][c] for c in D.CAMS], 1), split == "train")
        trva = split != "test"
        w_, b_ = D.ridge_fit(x[trva], actions[trva], 0.1)
        demos = P.layout_test_demos(index, split)
        sel = np.concatenate([s + np.arange(n - 1) for _, _, n, s in demos])
        want = float(D.r2(actions[sel], x[sel] @ w_ + b_).mean())
        got = full["r2"][P.STORED]["D"]["point"]
        assert np.isclose(got, want, atol=1e-4), (got, want)
        # Drift: re-render sits near the stored frame; perturbed types move features more.
        dr = res["drift"]["combined"]["dinov3"]
        assert dr["rerender_vs_stored"]["point"] < 0.1 * dr["noise"]["point"], dr
        assert dr["noise"]["ci95"][0] > 0.0
        # The flipped encoder drifts the most (negated features: about 4x the variance moves).
        dd = {e: res["drift"]["combined"][e]["camera"]["point"] for e in DIMS}
        assert max(dd, key=dd.get) == "c-radiov4", dd

        # Never overwrite.
        before = (out / "results.json").read_text()
        try:
            run_world(wa, out)
            raise AssertionError("expected FileExistsError")
        except FileExistsError:
            pass
        assert (out / "results.json").read_text() == before
        print("planted case: rule P and rule B fire (full and matched), alphas from tally, labels, saved stats, drift ok")

        # Nothing about the fit depends on the perturbed data: same clean data, different perturbations -> same clean numbers.
        wb = make_world(t / "b", seed=11, flip=[], noise_scale=1.5)
        rb2 = run_world(wb, t / "b" / "out")
        assert rb2["configs"]["full"]["combined"]["r2"][P.STORED] == res["configs"]["full"]["combined"]["r2"][P.STORED]
        assert rb2["configs"]["full"]["combined"]["probes"] == res["configs"]["full"]["combined"]["probes"]

        # Null: every partner is perturbed the same way, so R must not stand out.
        wn = make_world(t / "n", seed=12, flip=[])
        rn = run_world(wn, t / "n" / "out")
        fn = rn["configs"]["full"]["combined"]
        print("null pooled e:", {b: round(fn["pooled_e"][b]["point"], 3) for b in "RLSG"})
        assert not fn["rule_P"]["fires"] and not fn["rule_B"]["fires_point_reading"], (fn["rule_P"], fn["rule_B"])
        assert not rn["configs"]["matched"]["combined"]["rule_P"]["fires"]
        print("null case: rule P and rule B do not fire ok")

        # Camera-specific damage: R is negated on the agentview only.
        wv = make_world(t / "v", seed=13, flip=[("c-radiov4", 0)])
        rv = run_world(wv, t / "v" / "out")
        assert rv["configs"]["full"]["agentview"]["rule_P"]["fires"], rv["configs"]["full"]["agentview"]["rule_P"]
        assert not rv["configs"]["full"]["wrist"]["rule_P"]["fires"]
        assert rv["configs"]["full"]["wrist"]["pooled_e"]["R"]["point"] < 0.1
        print("views: agentview-only damage shows in agentview, not in wrist ok")

        # Frame counts: T = n (nothing dropped) also works.
        wf = make_world(t / "f", seed=14, flip=[("c-radiov4", 0), ("c-radiov4", 1)], drop_last=False, n_files=3)
        rf = run_world(wf, t / "f" / "out")
        assert rf["n_test_frames"] == 6 * 40 and set(rf["frames_per_demo_dropped_last"]) == {0}
        # Secondary mean features through the same code; all sizes equal here, so no matched row is produced.
        rm = P.run(wf["feats"], wf["raw"], SUITE, wf["conds"], wf["tally"], t / "f" / "out_mean", ftype="mean",
                   n_boot=100, min_boot=100, quiet=True)
        assert "matched" not in rm["configs"] and rm["matched"]["reduced"] == [] and rm["ftype"] == "mean"
        assert rm["configs"]["full"]["combined"]["probes"]["D+R"]["alpha"] == 1.0
        print("T = n case and mean features (no matched row when sizes agree) ok")


def test_input_errors():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        w = make_world(t / "w", seed=15, flip=[], n_files=2)
        # A condition missing.
        conds = {c: v for c, v in w["conds"].items() if c != "light"}
        try:
            P.run(w["feats"], w["raw"], SUITE, conds, w["tally"], t / "o1", min_boot=100, n_boot=100, quiet=True)
            raise AssertionError("expected KeyError")
        except KeyError:
            pass
        assert not (t / "o1").exists(), "a refused run must not leave an output dir behind"
        # A condition whose frame count disagrees with the re-render.
        p = w["conds"]["camera"] / "sam3" / SUITE / "task0.h5"
        with h5py.File(p, "a") as f:
            name = sorted(f.keys())[0]
            x = f[name][D.CAMS[0]]["global"][:]
            del f[name][D.CAMS[0]]["global"]
            f[name][D.CAMS[0]].create_dataset("global", data=x[:-2])
        try:
            run_world(w, t / "o2")
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "expected" in str(e) or "frame" in str(e), e
        # Declared sizes that disagree with the files.
        w2 = make_world(t / "w2", seed=16, flip=[], n_files=2)
        try:
            run_world(w2, t / "o3", sizes={"sam3": 1024})
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
        # Tally alpha missing.
        (w2["tally"] / "tally-libero_goal-dinov3-sam3.json").unlink()
        try:
            run_world(w2, t / "o4")
            raise AssertionError("expected KeyError")
        except KeyError:
            pass
        print("input errors (missing condition, frame mismatch, declared sizes, missing alpha) refused ok")


def main():
    test_rules_unit()
    test_quantities_known()
    test_bootstrap_pairing()
    test_drift_known()
    test_transform_and_pca()
    test_e2e_planted_and_null()
    test_input_errors()
    print("all ok")


if __name__ == "__main__":
    main()
