"""Synthetic checks for lp_eval.py (cases with known answers). Run: python test_lp_eval.py"""

import json
import math
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

import lp_eval as L

CATS = ["Camera Viewpoints", "Language Instructions", "Sensor Noise"]
SEEDS = L.EVAL_SEEDS


def fake_classification(rng):
    """Two suites, three categories, uneven cell sizes (one small cell)."""
    out = {}
    for suite, sizes in [("libero_goal", [300, 120, 7]), ("libero_10", [250, 180, 90])]:
        eps, i = [], 1
        for cat, n in zip(CATS, sizes):
            for _ in range(n):
                eps.append({"id": i, "name": f"{suite}_task_{i}", "category": cat, "difficulty_level": 1})
                i += 1
        rng.shuffle(eps)
        out[suite] = eps
    return out


def rows_for(sub, config, seed, count_of, benchmark=L.LP):
    """One row per episode x eval seed; the episode succeeds under the first count_of(e) eval seeds."""
    return [L.result_row(benchmark, e["suite"], e["id"], e["category"], config, seed, es, f"{config}-s{seed}",
                         sub["subset_id"], j < count_of(e))
            for e in sub["episodes"] for j, es in enumerate(SEEDS)]


def synthetic_subset(n):
    eps = [{"suite": "libero_goal", "id": i, "name": f"t{i}", "category": CATS[i % 3]} for i in range(n)]
    return {"subset_id": L.subset_id(eps), "episodes": eps}


def plant(sub, k_none, k_dl, k_dr):
    """Table where D succeeds under 1 of 3 eval seeds on every episode; on seed s, config X differs from D by k_s
    thirds in total (k > 0: k episodes at 2/3; k < 0: |k| episodes at 0), so delta_s(X - D) = 100 k_s / (3n) points.
    k_none is given as D - none (none is lowered)."""
    keys = [(e["suite"], e["id"]) for e in sub["episodes"]]
    t = {}
    for s in L.TRAIN_SEEDS:
        t[(L.D, s)] = {k: 1 for k in keys}
        for cfg, k in [(L.NONE, -k_none[s]), (L.DL, k_dl[s]), (L.DR, k_dr[s])]:
            t[(cfg, s)] = {key: (1 + (1 if k > 0 else -1) if i < abs(k) else 1) for i, key in enumerate(keys)}
    return t


def main():
    rng = np.random.default_rng(0)
    cls = fake_classification(rng)
    sizes = Counter((s, e["category"]) for s in cls for e in cls[s])
    total = sum(sizes.values())

    # 1. Proportional allocation: sums to N, each cell within 1 of N * share (largest remainder), >= 1 per cell when
    #    N is small; sampled counts equal the allocation; deterministic per seed; exclusion respected.
    for n in [6, 7, 50, 101, 400]:
        al = L.allocate(dict(sizes), n)
        assert sum(al.values()) == n and min(al.values()) >= 1, (n, al)
        if all(n * v / total >= 1 for v in sizes.values()):
            assert all(abs(al[c] - n * sizes[c] / total) < 1 for c in sizes), (n, al)
    al = L.allocate(dict(sizes), 101)
    assert al[("libero_goal", "Sensor Noise")] == 1 and al[("libero_goal", "Camera Viewpoints")] == 32   # 30.3 -> 30 by
    # floor, then +1 each to the largest remainders; checked by hand: quotas 32.15, 13.03, 0.76, 26.79, 19.29, 9.65
    sub = L.make_subset(cls, 101, seed=7)
    cnt = Counter((e["suite"], e["category"]) for e in sub["episodes"])
    assert dict(cnt) == {c: v for c, v in al.items()} and sub["n"] == 101
    assert len({(e["suite"], e["id"]) for e in sub["episodes"]}) == 101
    assert L.make_subset(cls, 101, seed=7) == sub and L.make_subset(cls, 101, seed=8)["subset_id"] != sub["subset_id"]
    excl = [(e["suite"], e["id"]) for e in sub["episodes"]]
    sub2 = L.make_subset(cls, 400, seed=9, exclude=excl)
    assert not {(e["suite"], e["id"]) for e in sub2["episodes"]} & set(excl) and sub2["n"] == 400
    try:
        L.make_subset(cls, 900, seed=9, exclude=excl)        # the 7-episode cell can't give its share after exclusion
        raise RuntimeError("exhausted cell accepted")
    except AssertionError:
        pass
    with tempfile.TemporaryDirectory() as d:
        Path(d, "s.json").write_text(json.dumps(sub))
        assert L.load_subset(Path(d, "s.json"))["subset_id"] == sub["subset_id"]
    print("1 ok: proportional subset (sums to N, within 1 of share, >= 1 per cell, exclusion, determinism)")

    # 2. Records per episode x eval seed, averaged per episode before pairing. On seed s, D+R loses k_s eval-seed
    #    successes spread so that some episodes drop by 1/3 and some by 2/3 (k = 2, 5, 0), and on seed 2 one episode
    #    D failed everywhere is solved under one eval seed. Deltas must be exactly 100 * (-k/3)/n points.
    n = sub["n"]
    keys = [(e["suite"], e["id"]) for e in sub["episodes"]]
    base = {s: {k: int(v) for k, v in zip(keys, np.random.default_rng(s).integers(0, 4, n))} for s in range(3)}
    rows = []
    want = {}
    for s in range(3):
        rows += rows_for(sub, L.D, s, lambda e, s=s: base[s][(e["suite"], e["id"])])
        dr = dict(base[s])
        if s == 0:                                            # one episode -2/3 => k = 2, discordant 1
            k = next(k for k in keys if dr[k] >= 2); dr[k] -= 2; want[s] = (-2, 1, 0)
        if s == 1:                                            # 5 thirds over 3 episodes (2 + 2 + 1)
            ks = [k for k in keys if dr[k] >= 2][:3]
            for k, m in zip(ks, [2, 2, 1]):
                dr[k] -= m
            want[s] = (-5, 3, 0)
        if s == 2:
            k = next(k for k in keys if dr[k] == 0); dr[k] = 1; want[s] = (1, 0, 1)
        rows += rows_for(sub, L.DR, s, lambda e, dr=dr: dr[(e["suite"], e["id"])])
    rng.shuffle(rows)                                         # order must not matter
    table = L.lp_table(rows, sub)
    prim = L.paired(table, sub, L.DR, L.D)
    for s in range(3):
        k, worse, better = want[s]
        assert math.isclose(prim[s]["delta_pts"], 100 * k / 3 / n), (s, prim[s]["delta_pts"])
        assert prim[s]["delta_exact"] * 3 * n == 100 * k
        assert (prim[s]["a_worse"], prim[s]["a_better"]) == (worse, better)
    print("2 ok: per-episode averaging over eval seeds, planted per-seed drops exact ("
          + ", ".join(f"{prim[s]['delta_pts']:+.3f}" for s in range(3)) + " pts)")

    # 3. Identical configs give exactly zero.
    same = L.paired(table, sub, L.D, L.D)
    assert all(same[s]["delta_exact"] == 0 and same[s]["a_worse"] == same[s]["a_better"] == 0 for s in range(3))
    print("3 ok: identical configs give exactly 0")

    # 4. Bad inputs fail loudly: a missing eval seed, a duplicate eval seed, a foreign eval seed, a missing episode
    #    (all three eval seeds gone), a row from another subset.
    one = [r for r in rows if (r["config"], r["train_seed"]) == (L.D, 0)]
    rest = [r for r in rows if (r["config"], r["train_seed"]) != (L.D, 0)]
    k0 = (one[0]["suite"], one[0]["episode_id"])
    no_ep = [r for r in one if (r["suite"], r["episode_id"]) != k0]
    for bad, what in [(rest + one[1:], "missing eval seed"), (rest + one + one[:1], "duplicate"),
                      (rest + one + [dict(one[0], eval_seed=44)], "eval seed 44"), (rest + no_ep, "missing episode"),
                      (rest + [dict(one[0], subset_id="other")] + one[1:], "foreign subset")]:
        try:
            L.lp_table(bad, sub)
        except AssertionError:
            continue
        raise RuntimeError(f"lp_table accepted: {what}")
    print("4 ok: missing / duplicate / foreign eval seeds, missing episodes, foreign-subset rows rejected")

    # 5. Locked verdict on planted cases, boundaries included (n = 2000: seed delta = k/60 pts, average = K/180).
    big = synthetic_subset(2000)
    z = [0, 0, 0]
    cases = [
        ("pass, effect",              [200] * 3, z,                 [-200] * 3,          L.V_EFFECT, True, True),
        ("(a) mean exactly +2.7",     [162] * 3, z,                 [-200] * 3,          L.V_EFFECT, True, True),
        ("(a) mean just below",       [162, 162, 161], z,           [-200] * 3,          L.V_CAL_FAIL, False, True),
        ("(a) one seed exactly 0",    [0, 400, 400], z,             [-200] * 3,          L.V_CAL_FAIL, False, True),
        ("(a) one seed negative",     [-1, 400, 400], z,            [-200] * 3,          L.V_CAL_FAIL, False, True),
        ("(b) mean exactly -1.35",    [200] * 3, [-81] * 3,         [-200] * 3,          L.V_EFFECT, True, True),
        ("(b) mean just below",       [200] * 3, [-81, -81, -82],   [-200] * 3,          L.V_CAL_FAIL, True, False),
        ("(a) and (b) fail",          [0] * 3, [-300] * 3,          [-200] * 3,          L.V_CAL_FAIL, False, False),
        ("D+R mean exactly -2.7",     [200] * 3, z,                 [-162] * 3,          L.V_EFFECT, True, True),
        ("D+R mean just above -2.7",  [200] * 3, z,                 [-162, -162, -161],  L.V_NO_EFFECT, True, True),
        ("D+R one seed exactly 0",    [200] * 3, z,                 [0, -600, -600],     L.V_NO_EFFECT, True, True),
        ("D+R one seed positive",     [200] * 3, z,                 [1, -600, -600],     L.V_NO_EFFECT, True, True),
        ("D+R no drop",               [200] * 3, z,                 z,                   L.V_NO_EFFECT, True, True),
        ("D+R improves",              [200] * 3, z,                 [100] * 3,           L.V_NO_EFFECT, True, True),
    ]
    for name, kn, kl, kr, want_v, want_a, want_b in cases:
        v = L.verdict(plant(big, kn, kl, kr), big)
        assert (v["verdict"], v["calibration_a"], v["calibration_b"]) == (want_v, want_a, want_b), (name, v)
        assert all(math.isclose(x, k / 60, abs_tol=1e-12) for x, k in zip(v["pairs"]["DR-D"]["per_seed_pts"], kr)), name
    try:
        t = plant(big, [200] * 3, z, [-200] * 3)
        del t[(L.DR, 2)]
        L.verdict(t, big)
        raise RuntimeError("verdict ran without all three train seeds")
    except AssertionError:
        pass
    print(f"5 ok: locked verdict on {len(cases)} planted cases incl. every boundary; missing train seed rejected")

    # 6. Sizing helper recovers a planted sd: half the pilot episodes d = +1/3, half -1/3 => sample sd =
    #    (100/3) sqrt(n/(n-1)); sd* = 1.5 sd; N = ceil((sd*/1.35)^2).
    pilot = synthetic_subset(100)
    pk = [(e["suite"], e["id"]) for e in pilot["episodes"]]
    pt = {(L.D, 0): {k: 1 for k in pk}, (L.DR, 0): {k: (2 if i % 2 else 0) for i, k in enumerate(pk)}}
    zz = L.sizing(pt, pilot, 0)
    sd = 100 / 3 * math.sqrt(100 / 99)
    assert math.isclose(zz["sd_pts"], sd) and math.isclose(zz["sd_star_pts"], 1.5 * sd)
    assert zz["N"] == math.ceil((1.5 * sd / 1.35) ** 2) == 1386, zz   # (50.252 / 1.35)^2 = 1385.6 (first written
    # as 1403, a hand-arithmetic slip; the formula side of the assert was unchanged)
    print(f"6 ok: sizing recovers planted sd ({zz['sd_pts']:.3f} pts -> N {zz['N']})")

    # 7. Episode-level CI: constant d gives a zero-width interval at d; for +/-1 d the width matches the normal
    #    approximation over n episodes (not 3n rollouts).
    assert L.episode_ci([5.0] * 50, draws=200) == (5.0, 5.0)
    d = np.array([1.0, -1.0] * 200)
    lo, hi = L.episode_ci(d, draws=4000, seed=1)
    assert abs((hi - lo) / (2 * 1.96 * d.std() / math.sqrt(len(d))) - 1) < 0.1, (lo, hi)
    print("7 ok: CIs resample whole episodes")

    # 8. Secondary numbers on a hand case: config "x" solves every Camera episode under all eval seeds and nothing
    #    else on seed 0, everything under one eval seed on seed 1; clean LIBERO kept apart.
    cam = sum(e["category"] == "Camera Viewpoints" for e in sub["episodes"])
    r8 = rows_for(sub, "x", 0, lambda e: 3 * (e["category"] == "Camera Viewpoints")) + rows_for(sub, "x", 1, lambda e: 1)
    r8 += [L.result_row(L.CLEAN, "libero_goal", i, "", "x", 0, 41, "x-s0", "clean", i % 2) for i in range(10)]
    sec = L.secondary(r8, L.lp_table(r8, sub), sub)["x"]
    assert math.isclose(sec["lp_per_seed_pts"][0], 100 * cam / n) and math.isclose(sec["lp_per_seed_pts"][1], 100 / 3)
    assert math.isclose(sec["lp_per_category_pts"]["Camera Viewpoints"], (100 + 100 / 3) / 2)
    assert math.isclose(sec["lp_per_category_pts"]["Sensor Noise"], 100 / 6)
    assert sec["clean_per_seed_pts"] == {0: 50.0}
    print("8 ok: secondary per-config, per-category and clean numbers")

    # 9. CLI end to end: subset (with exclusion), sizing, analyze with the verdict on a full planted round.
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "cls.json").write_text(json.dumps(cls))
        run = lambda *a: subprocess.run([sys.executable, str(Path(__file__).with_name("lp_eval.py")), *map(str, a)],
                                        capture_output=True, text=True)
        r = run("subset", "--classification", d / "cls.json", "--n", 101, "--seed", 7, "--out", d / "pilot.json")
        assert r.returncode == 0, r.stderr
        assert json.loads((d / "pilot.json").read_text())["subset_id"] == sub["subset_id"]
        r = run("subset", "--classification", d / "cls.json", "--n", 400, "--seed", 9, "--exclude", d / "pilot.json",
                "--out", d / "final.json")
        assert r.returncode == 0, r.stderr
        fin = L.load_subset(d / "final.json")
        assert fin["subset_id"] == sub2["subset_id"] and fin["excluded_subsets"] == [sub["subset_id"]]
        prow = []
        for s in [0]:
            prow += rows_for(sub, L.D, s, lambda e: 1) + rows_for(sub, L.DR, s, lambda e: 2 if e["id"] % 2 else 0)
        (d / "pilot.jsonl").write_text("\n".join(json.dumps(x) for x in prow))
        r = run("sizing", "--subset", d / "pilot.json", "--results", d / "pilot.jsonl", "--out", d / "size.json")
        assert r.returncode == 0 and "N " in r.stdout, r.stderr
        assert json.loads((d / "size.json").read_text())["N"] >= 1
        t = plant(fin, [20] * 3, [0] * 3, [-20] * 3)          # n = 400: delta = k/12 pts -> D-none +1.67: (a) fails
        frows = []
        for (cfg, s), cnts in t.items():
            frows += rows_for(fin, cfg, s, lambda e, c=cnts: c[(e["suite"], e["id"])])
        for i in range(2):
            (d / f"r{i}.jsonl").write_text("\n".join(json.dumps(x) for x in frows[i::2]))
        r = run("analyze", "--subset", d / "final.json", "--results", d / "r0.jsonl", d / "r1.jsonl",
                "--ci-draws", 200, "--out", d / "p.json")
        assert r.returncode == 0, r.stderr
        assert L.FRAMING in r.stdout and L.V_CAL_FAIL in r.stdout and "-1.67" in r.stdout, r.stdout
        out = json.loads((d / "p.json").read_text())
        assert out["verdict"]["verdict"] == L.V_CAL_FAIL and not out["verdict"]["calibration_a"]
        assert [round(out["primary"][str(s)]["delta_pts"], 6) for s in range(3)] == [round(-20 / 12, 6)] * 3
    print("9 ok: CLI end to end (subset with exclusion, sizing, analyze + verdict)")
    print("all checks passed")


if __name__ == "__main__":
    main()
