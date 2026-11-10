"""Synthetic checks for feature_overlap.py (cases with known answers). Run: python -m pytest test_feature_overlap.py -q

One test per line of the PLAN's test list (experiments/2026-11-10-feature-overlap/PLAN.md, "Code needed"), plus checks that
the fast ridge and the bootstrap equal the diagnostics.py versions. Every threshold below was written before the first run
and is not changed to make a test pass (a failing test means a code or test-data bug, to be fixed and logged):

  1. exact linear relation (y = x W, 6 -> 4 columns): mean and pooled score >= 0.99;
     independent noise target: score between -0.03 and +0.01 (the PLAN's noise gate).
  2. planted asymmetry (A = [shared 4, extra 4], B = shared W exactly), base A, partner B:
     S2 = C(B in A) >= 0.95, S1 = C(A in B) <= 0.80, S3 >= +0.15.
  3. wide pure-noise source (300 columns, 600 train frames) predicting a structured target: score between -0.03 and +0.01.
  4. demo-correlated random-walk features (per-demo offset + walk, source and target unrelated across columns, 100 source
     columns, 40 demos): demo split score between -0.40 and +0.10; frame-level split on the same data >= 0.60; no demo in
     two sets.
  5. changing test frames (source and target) leaves the target components, coverage, validation curves and alpha exactly
     equal, and the test score does change.
  6. scrambling the test targets leaves the validation curves and alpha exactly equal, and the score changes.
  7. identical predictors: the difference interval is exactly [0.0, 0.0].
  8. equal test variance in every component: mean-of-components R^2 equals pooled R^2 to 1e-12.
  9. folds: per file with 50 demos fold 1 = diagnostics.demo_split; test blocks of folds 1, 2, 3 are demos 40-49, 0-9, 20-29;
     within a fold the labels partition the demos 30 / 10 / 10 and val and test are disjoint; the test sets of the three folds
     are pairwise disjoint.
 10. fast ridge vs diagnostics.ridge_select on the same arrays: same alpha, per-column test R^2 equal to 1e-8, validation
     curve equal to 1e-6 (first run: the curve differed by 2.6e-8 because ridge_select multiplies float32 arrays; the 1e-8
     written first was too tight for that reference, a test-tolerance fix, not a code change).
 11. bootstrap draws equal diagnostics.bootstrap_r2 to 1e-12 (same seed); with all-ones weights the pooled draw equals the
     point value to 1e-12.
 12. eigendecompositions per direction: 3 without the baseline (target PCA, source on train, source on train+val), 5 with
     it, the same with a 3-value alpha grid (one per source, not per alpha or target).
 13. dup and noise partners through main() on a small stationary fixture, fold 1: dup S1 and S2 >= 0.98 and |S3| <= 0.02;
     noise S1 and S2 in [-0.03, +0.01] and |S3| <= 0.04; split counts equal diagnostics.demo_split's; test demos 4;
     test evaluations 6 per type; saved draws have n_boot entries and S3 draws = S2 draws - S1 draws; folds 2 and 3 run.
 PR #21 review fixes (thresholds written before the run):
 14. run_type on planted lopsided two-camera data (A = [shared 4, extra 4] per camera, B = shared W): out["S2"] >= 0.95,
     out["S1"] <= 0.80, S3 >= +0.15, and the S2 definition names the base as source ("source A, target B"); this test must
     fail when S1 and S2 are swapped in run_type (checked by hand once).
 15. fold_blocks at every n from 10 to 120 and fold 1, 2, 3: test and validation positions inside [0, n), disjoint, sizes
     round(0.2 n) each, and fold 1 test = last round(0.2 n) demos; explicitly n = 44 and 53 (where the old fold 3 overlapped).
 16. with show_scores False the log lines contain no score ("S1", "S2", "score") and name coverage, alpha, grid-edge only.
 17. the result JSON stores the ordered test demo ids and their sha256; dup and noise runs of fold 1 give the same hash,
     fold 2 a different one; a context pair (base not dinov3) says no primary score applies.
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

import diagnostics as D
import feature_overlap as F


def make_index(n_files=1, n_demo=50, length=20):
    return [(f"task{f}", f"demo_{d}", length) for f in range(n_files) for d in range(n_demo)]


def groups_of(index):
    return np.repeat(np.arange(len(index)), [n for _, _, n in index])


def direction(x, y, split, groups, m, n_boot=50, aux=None, seed=0):
    k = len(np.unique(groups[split == "test"]))
    return F.run_direction(x, y, split, groups, m, F.boot_weights(k, n_boot, seed), aux)


def stationary_pair(rng, n, dx=6, dy=4):
    x = rng.normal(size=(n, dx))
    return x.astype(np.float32), (x @ rng.normal(size=(dx, dy))).astype(np.float32)


# 1
def test_exact_linear_and_independent_noise():
    index = make_index(n_demo=50, length=20)
    split, g, rng = F.fold_split(index, 1), groups_of(index), np.random.default_rng(0)
    x, y = stationary_pair(rng, len(split))
    r, _ = direction(x, y, split, g, 4)
    assert r["score"]["value"] >= 0.99 and r["score_pooled"]["value"] >= 0.99, r["score"]
    noise = rng.normal(size=y.shape).astype(np.float32)
    r, _ = direction(x, noise, split, g, 4)
    assert -0.03 <= r["score"]["value"] <= 0.01, r["score"]


# 2
def test_planted_asymmetry_sign():
    index = make_index(n_demo=50, length=20)
    split, g, rng = F.fold_split(index, 1), groups_of(index), np.random.default_rng(1)
    shared = rng.normal(size=(len(split), 4))
    a = np.concatenate([shared, rng.normal(size=(len(split), 4))], 1).astype(np.float32)
    b = (shared @ rng.normal(size=(4, 4))).astype(np.float32)
    s2, _ = direction(a, b, split, g, 4)  # C(B in A)
    s1, _ = direction(b, a, split, g, 4)  # C(A in B)
    s3 = s2["score"]["value"] - s1["score"]["value"]
    assert s2["score"]["value"] >= 0.95 and s1["score"]["value"] <= 0.80 and s3 >= 0.15, (s2["score"], s1["score"])


# 3
def test_wide_noise_source_reads_zero():
    index = make_index(n_demo=50, length=20)
    split, g, rng = F.fold_split(index, 1), groups_of(index), np.random.default_rng(2)
    assert (split == "train").sum() == 600
    target = rng.normal(size=(len(split), 4)) @ rng.normal(size=(4, 4))
    r, _ = direction(rng.normal(size=(len(split), 300)).astype(np.float32), target.astype(np.float32), split, g, 4)
    assert -0.03 <= r["score"]["value"] <= 0.01, r["score"]


# 4
def test_demo_split_blocks_demo_memorization_frame_split_does_not():
    n_demo, length = 40, 30
    index = make_index(n_demo=n_demo, length=length)
    g, rng = groups_of(index), np.random.default_rng(3)

    def walks(d):
        off = 3.0 * rng.normal(size=(n_demo, 1, d))
        walk = np.cumsum(0.1 * rng.normal(size=(n_demo, length, d)), 1)
        return (off + walk).reshape(-1, d).astype(np.float32)

    x, y = walks(100), walks(12)  # unrelated columns, both demo-correlated
    split = F.fold_split(index, 1)
    sets = {s: set(g[split == s]) for s in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
    r_demo, _ = direction(x, y, split, g, 8)
    assert -0.40 <= r_demo["score"]["value"] <= 0.10, r_demo["score"]
    frame_split = np.array(["train"] * len(g), dtype=object)
    u = rng.random(len(g))
    frame_split[u < 0.4] = "val"
    frame_split[u < 0.2] = "test"
    frame_split = frame_split.astype(str)
    assert set(g[frame_split == "train"]) & set(g[frame_split == "test"])  # the leaky split does share demos
    r_frame, _ = direction(x, y, frame_split, g, 8)
    assert r_frame["score"]["value"] >= 0.60, r_frame["score"]


# 5, 6
def _base_case():
    index = make_index(n_demo=50, length=20)
    split, g, rng = F.fold_split(index, 1), groups_of(index), np.random.default_rng(4)
    x = rng.normal(size=(len(split), 10)).astype(np.float32)
    y = (x[:, :5] @ rng.normal(size=(5, 8)) + rng.normal(size=(len(split), 8))).astype(np.float32)
    return split, g, x, y


def test_test_frames_change_nothing_fitted():
    split, g, x, y = _base_case()
    te = split == "test"
    rng = np.random.default_rng(5)
    x2, y2 = x.copy(), y.copy()
    x2[te] = rng.normal(size=x2[te].shape) * 5
    y2[te] = rng.normal(size=y2[te].shape) * 5
    train = split == "train"
    _, c1, v1, cov1, _ = F.prepare_target(y, train, 4)
    _, c2, v2, cov2, _ = F.prepare_target(y2, train, 4)
    assert np.array_equal(v1, v2) and cov1 == cov2
    r1, _ = direction(x, y, split, g, 4)
    r2, _ = direction(x2, y2, split, g, 4)
    assert r1["alpha"] == r2["alpha"] and r1["val_r2_by_alpha"] == r2["val_r2_by_alpha"]
    assert r1["score"]["value"] != r2["score"]["value"]


def test_scrambled_test_targets_leave_alpha():
    split, g, x, y = _base_case()
    te = np.flatnonzero(split == "test")
    y2 = y.copy()
    y2[te] = y[np.random.default_rng(6).permutation(te)]
    r1, _ = direction(x, y, split, g, 4)
    r2, _ = direction(x, y2, split, g, 4)
    assert r1["alpha"] == r2["alpha"] and r1["val_r2_by_alpha"] == r2["val_r2_by_alpha"]
    assert r1["score"]["value"] != r2["score"]["value"]


# 7
def test_identical_predictors_give_zero_interval():
    split, g, x, y = _base_case()
    te = split == "test"
    yhat = y[te].astype(np.float64) + np.random.default_rng(7).normal(size=y[te].shape)
    w = F.boot_weights(len(np.unique(g[te])), 200, 0)
    a = F.boot_scores(y[te].astype(np.float64), yhat, g[te], w)
    b = F.boot_scores(y[te].astype(np.float64), yhat.copy(), g[te], w)
    assert D.ci(a["mean"] - b["mean"]) == [0.0, 0.0] and D.ci(a["pooled"] - b["pooled"]) == [0.0, 0.0]


# 8
def test_mean_of_components_equals_pooled_when_variances_equal():
    rng = np.random.default_rng(8)
    y = rng.normal(size=(500, 6))
    y = (y - y.mean(0)) / y.std(0)  # every component has exactly the same test variance
    yhat = y + rng.normal(size=y.shape) * rng.uniform(0.1, 2.0, size=6)
    _, mean, pooled = F.summaries(y, yhat)
    assert abs(mean - pooled) < 1e-12, (mean, pooled)


# 9
def test_folds_are_disjoint_cover_and_fold1_is_demo_split():
    index = make_index(n_files=10, n_demo=50, length=5)
    g = groups_of(index)
    names = [f"demo_{d}" for d in range(50)]
    ref = D.demo_split(names)
    tests = []
    for fold, (blk) in ((1, range(40, 50)), (2, range(0, 10)), (3, range(20, 30))):
        sp = F.fold_split(index, fold)
        per_demo = sp[np.cumsum([0] + [n for _, _, n in index[:-1]])]
        for f in range(10):
            lab = per_demo[f * 50:(f + 1) * 50]
            assert {s: int((lab == s).sum()) for s in ("train", "val", "test")} == {"train": 30, "val": 10, "test": 10}
            assert [j for j in range(50) if lab[j] == "test"] == list(blk)
            if fold == 1:
                assert list(lab) == [ref[n] for n in names]
        tests.append(set(g[sp == "test"]))
        assert len(sp) == len(g)
    assert not (tests[0] & tests[1] or tests[0] & tests[2] or tests[1] & tests[2])


# 10
def test_fast_ridge_equals_diagnostics_ridge_select():
    split, g, x, y = _base_case()
    train = split == "train"
    xs, ys = D.standardize(x, train), D.standardize(y, train)
    r2_ref, alpha_ref, _, curve_ref = D.ridge_select(xs, ys, split)
    tr, va, te = (np.flatnonzero(split == s) for s in ("train", "val", "test"))
    fit_tr, fit_trva = F.make_fits(xs, tr, va)
    yhat, alpha, curve = F.select_and_predict(fit_tr, fit_trva, ys, va, te)
    assert alpha == alpha_ref
    assert np.allclose(D.r2(ys[te].astype(np.float64), yhat), r2_ref, atol=1e-8)
    assert all(abs(curve[a] - curve_ref[a]) < 1e-6 for a in D.ALPHAS)  # ridge_select multiplies float32 arrays


# 11
def test_bootstrap_equals_diagnostics_and_pooled_point_value():
    split, g, x, y = _base_case()
    te = split == "test"
    yhat = y[te].astype(np.float64) * 0.7
    k = len(np.unique(g[te]))
    mine = F.boot_scores(y[te].astype(np.float64), yhat, g[te], F.boot_weights(k, 300, 0))
    ref = D.bootstrap_r2(y[te].astype(np.float64), {"a": yhat}, g[te], 300, 0)["a"]
    assert np.allclose(mine["mean"], ref, atol=1e-12, rtol=0)
    one = F.boot_scores(y[te].astype(np.float64), yhat, g[te], np.ones((1, k)))
    _, mean, pooled = F.summaries(y[te], yhat)
    assert abs(one["pooled"][0] - pooled) < 1e-12 and abs(one["mean"][0] - mean) < 1e-12


# 12
def test_one_eigendecomposition_per_source(monkeypatch):
    split, g, x, y = _base_case()
    aux = np.random.default_rng(9).normal(size=(len(split), 4)).astype(np.float32)
    calls = []
    real = np.linalg.eigh
    monkeypatch.setattr(np.linalg, "eigh", lambda a: calls.append(a.shape) or real(a))
    direction(x, y, split, g, 4)
    assert len(calls) == 3, calls
    calls.clear()
    direction(x, y, split, g, 4, aux=aux)
    assert len(calls) == 5, calls
    calls.clear()
    monkeypatch.setattr(D, "ALPHAS", [1e-2, 1.0, 1e3])
    direction(x, y, split, g, 4, aux=aux)
    assert len(calls) == 5, calls


# 13
def make_fixture(root, n_files=2, n_demo=10, length=100, dim=8, seed=0):
    """As test_diagnostics.make_fixture but with a stationary latent (iid per frame), so a noise partner has no
    demo-level mean drift to be penalized for."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "raw").mkdir(parents=True)
    (root / "feats" / "dinov3" / "suite").mkdir(parents=True)
    for i in range(n_files):
        with h5py.File(root / "raw" / f"task{i}_demo.hdf5", "w") as fr, \
                h5py.File(root / "feats" / "dinov3" / "suite" / f"task{i}_demo.h5", "w") as ff:
            ff.attrs["meta"] = json.dumps({"encoder": "fixture-dinov3"})
            for d in range(n_demo):
                lat = rng.normal(size=(length, dim))
                g = fr.create_group(f"data/demo_{d}")
                g["actions"] = lat[:, :7] + 0.3 * rng.normal(size=(length, 7))
                g["obs/joint_states"] = rng.normal(size=(length, 7))
                g["obs/gripper_states"] = rng.normal(size=(length, 2))
                for c in D.CAMS:
                    x = (lat + 0.2 * rng.normal(size=lat.shape)).astype(np.float32)
                    gc = ff.create_group(f"demo_{d}/{c}")
                    gc["global"], gc["mean"] = x, x
    return str(root / "feats"), str(root / "raw")


def run_main(tmp, feats, raw, spec, fold=1, n_boot=60, extra=()):
    out = Path(tmp) / f"o-{spec.replace(':', '-')}-f{fold}.json"
    draws = Path(tmp) / f"d-{spec.replace(':', '-')}-f{fold}.npz"
    argv = ["feature_overlap.py", "--feats", feats, "--suite", "suite", "--raw", raw, "--base", "dinov3", "--partner", spec,
            "--fold", str(fold), "--m", "8", "--n-boot", str(n_boot), "--seed", "0", "--synth-seed", "0",
            "--save-draws", str(draws), "--out", str(out), *extra]
    old, sys.argv = sys.argv, argv
    try:
        F.main()
    finally:
        sys.argv = old
    return json.loads(out.read_text()), np.load(draws)


def test_control_partners_hit_gates_through_main(tmp_path):
    feats, raw = make_fixture(tmp_path)
    ref_split = {s: int(n) for s, n in zip(*np.unique(
        np.concatenate([np.repeat(list(D.demo_split(range(10)).values()), 100)] * 2), return_counts=True))}
    dup, dr = run_main(tmp_path, feats, raw, "dup:dinov3")
    s = {k: dup["types"]["global"][k]["value"] for k in ("S1", "S2", "S3")}
    assert s["S1"] >= 0.98 and s["S2"] >= 0.98 and abs(s["S3"]) <= 0.02, s
    noise, _ = run_main(tmp_path, feats, raw, "noise:32")
    s = {k: noise["types"]["global"][k]["value"] for k in ("S1", "S2", "S3")}
    assert -0.03 <= s["S1"] <= 0.01 and -0.03 <= s["S2"] <= 0.01 and abs(s["S3"]) <= 0.04, s
    for r in (dup, noise):
        assert r["split_frames"] == ref_split and r["n_test_demos"] == 4, r["split_frames"]
        assert r["types"]["global"]["test_evaluations"] == 6 and r["test_evaluations"]["count"] == 6
        assert r["config"]["n_boot"] == 60 and not r["config"]["n_boot_meets_standard"]
        assert set(r["types"]["global"]["directions"]) == {"dinov3->" + r["partner"], r["partner"] + "->dinov3"}
    assert all(len(dr[k]) == 60 for k in dr.files)
    assert np.allclose(dr["global|S3|score"], dr["global|dinov3->dup:dinov3|score"] - dr["global|dup:dinov3->dinov3|score"])
    for fold in (2, 3):
        r, _ = run_main(tmp_path, feats, raw, "dup:dinov3", fold=fold)
        assert r["config"]["fold"] == fold and r["split_frames"] == ref_split and r["n_test_demos"] == 4
        assert r["types"]["global"]["S2"]["value"] >= 0.98


# 14, 16
def lopsided_feats(n_demo=50, length=20, seed=11):
    rng = np.random.default_rng(seed)
    index = make_index(n_demo=n_demo, length=length)
    n = n_demo * length
    a, b = {}, {}
    for c in D.CAMS:
        shared = rng.normal(size=(n, 4))
        a[c] = np.concatenate([shared, rng.normal(size=(n, 4))], 1).astype(np.float32)
        b[c] = (shared @ rng.normal(size=(4, 4))).astype(np.float32)
    feats = {"A": {"global": a}, "B": {"global": b}}
    return feats, index, F.fold_split(index, 1), groups_of(index), rng.normal(size=(n, 3)).astype(np.float32)


def test_s1_s2_labels_not_swapped():
    feats, index, split, g, aux = lopsided_feats()
    out, _ = F.run_type(feats, split, index, g, aux, "A", "B", "global", 8, 50, 0, lambda msg: None)
    assert out["S2"]["value"] >= 0.95 and out["S1"]["value"] <= 0.80 and out["S3"]["value"] >= 0.15, (out["S1"], out["S2"])
    assert "source A, target B" in out["S2"]["definition"] and "source B, target A" in out["S1"]["definition"]
    assert "no primary score" in out["note"]  # base A is not dinov3


def test_log_withholds_scores_for_real_partners():
    feats, index, split, g, aux = lopsided_feats()
    lines = []
    F.run_type(feats, split, index, g, aux, "A", "B", "global", 8, 50, 0, lines.append, show_scores=False)
    assert lines and all("S1" not in m and "S2" not in m and "score:" not in m and "pooled" not in m for m in lines), lines
    assert all("coverage" in m and "alpha" in m and "grid-edge" in m for m in lines if "withheld" in m)
    shown = []
    F.run_type(feats, split, index, g, aux, "A", "B", "global", 8, 50, 0, shown.append)
    assert any("S2" in m for m in shown)


# 15
def test_fold_blocks_never_overlap_any_demo_count():
    for n in list(range(10, 121)) + [44, 53]:
        n_t = max(1, round(n * 0.2))
        for fold in (1, 2, 3):
            te, va = F.fold_blocks(n, fold)
            assert len(te) == n_t and len(va) == n_t and not set(te) & set(va), (n, fold)
            assert min(te) >= 0 and max(te) < n and min(va) >= 0 and max(va) < n, (n, fold)
        assert list(F.fold_blocks(n, 1)[0]) == list(range(n - n_t, n))
    assert F.fold_blocks(50, 3) == (range(20, 30), range(30, 40))  # the real file size is unchanged


# 17
def test_test_demo_ids_in_result_json(tmp_path):
    feats, raw = make_fixture(tmp_path)
    r1, _ = run_main(tmp_path, feats, raw, "dup:dinov3")
    r2, _ = run_main(tmp_path, feats, raw, "noise:32")
    r3, _ = run_main(tmp_path, feats, raw, "dup:dinov3", fold=2)
    assert len(r1["test_demos"]["ids"]) == 4 and r1["test_demos"]["ids"][0].startswith("task0_demo/demo_")
    assert r1["test_demos"] == r2["test_demos"] and r1["test_demos"]["sha256"] != r3["test_demos"]["sha256"]
    assert "no primary score" not in r1["config"]["primary"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
