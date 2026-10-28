"""Synthetic checks for lp_eval.py (cases with known answers). Run: python test_lp_eval.py"""

import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

import lp_eval as L

CATS = ["Camera Viewpoints", "Language Instructions", "Sensor Noise"]


def fake_classification(rng):
    """Two suites, three categories, uneven cell sizes (one cell smaller than per_cell)."""
    out = {}
    for suite, sizes in [("libero_goal", [30, 12, 3]), ("libero_10", [25, 18, 9])]:
        eps, i = [], 1
        for cat, n in zip(CATS, sizes):
            for _ in range(n):
                eps.append({"id": i, "name": f"{suite}_task_{i}", "category": cat, "difficulty_level": 1})
                i += 1
        rng.shuffle(eps)
        out[suite] = eps
    return out


def rows_for(sub, config, seed, success_of, benchmark=L.LP):
    return [L.result_row(benchmark, e["suite"], e["id"], e["category"], config, seed, 41, f"{config}-s{seed}",
                         sub["subset_id"], success_of(e)) for e in sub["episodes"]]


def main():
    rng = np.random.default_rng(0)
    cls = fake_classification(rng)

    # 1. Subset: min(per_cell, cell size) per (suite, category), deterministic for a seed, different across seeds,
    #    unique keys, id round-trips through load_subset.
    sub = L.make_subset(cls, 5, seed=7)
    cnt = Counter((e["suite"], e["category"]) for e in sub["episodes"])
    assert cnt[("libero_goal", "Sensor Noise")] == 3 and all(v == 5 for k, v in cnt.items() if k != ("libero_goal", "Sensor Noise"))
    assert sub["n"] == 28 and len({(e["suite"], e["id"]) for e in sub["episodes"]}) == 28
    assert L.make_subset(cls, 5, seed=7) == sub and L.make_subset(cls, 5, seed=8)["subset_id"] != sub["subset_id"]
    with tempfile.TemporaryDirectory() as d:
        Path(d, "s.json").write_text(json.dumps(sub))
        assert L.load_subset(Path(d, "s.json"))["subset_id"] == sub["subset_id"]
    print("1 ok: stratified subset (cell counts, determinism, id)")

    # 2. Planted drop: D+R equals D except that on seed s a known set of k_s episodes D solved is failed by D+R
    #    (k = 2, 5, 0), and on seed 2 one episode D failed is solved by D+R. Deltas must be exactly -k_s/n (+1/n on
    #    seed 2), with the right discordant counts, and pairing must be by seed (different k per seed).
    n = sub["n"]
    keys = [(e["suite"], e["id"]) for e in sub["episodes"]]
    base = {s: {k: int(v) for k, v in zip(keys, np.random.default_rng(s).random(n) < 0.6)} for s in range(3)}
    solved = {s: [k for k in keys if base[s][k]] for s in range(3)}
    failed2 = [k for k in keys if not base[2][k]]
    drop = {0: solved[0][:2], 1: solved[1][:5], 2: []}
    gain = {0: [], 1: [], 2: failed2[:1]}
    rows = []
    for s in range(3):
        rows += rows_for(sub, "dinov3", s, lambda e, s=s: base[s][(e["suite"], e["id"])])
        dr = {k: (0 if k in drop[s] else 1 if k in gain[s] else base[s][k]) for k in keys}
        rows += rows_for(sub, "dinov3+c-radiov4", s, lambda e, s=s, dr=dr: dr[(e["suite"], e["id"])])
    rng.shuffle(rows)                                                  # order must not matter
    table = L.lp_table(rows, sub)
    prim = L.paired(table, sub, "dinov3+c-radiov4", "dinov3")
    assert [prim[s]["delta"] for s in range(3)] == [-2 / n, -5 / n, 1 / n], [prim[s]["delta"] for s in range(3)]
    assert [(prim[s]["a_worse"], prim[s]["a_better"]) for s in range(3)] == [(2, 0), (5, 0), (0, 1)]
    assert sum(prim[1]["per_episode"].values()) == -5
    print("2 ok: planted per-seed drops recovered exactly (" + ", ".join(f"{prim[s]['delta']:+.4f}" for s in range(3)) + ")")

    # 3. Identical configs give exactly zero for every seed.
    same = L.paired(table, sub, "dinov3", "dinov3")
    assert all(same[s]["delta"] == 0.0 and same[s]["a_worse"] == same[s]["a_better"] == 0 for s in range(3))
    print("3 ok: identical configs give exactly 0")

    # 4. Bad inputs fail loudly: a missing episode, a duplicate rollout, a row from another subset.
    for bad, what in [(rows[1:], "missing"), (rows + rows[:1], "duplicate"),
                      ([dict(rows[0], subset_id="other")] + rows[1:], "subset")]:
        try:
            L.lp_table(bad, sub)
        except AssertionError:
            continue
        raise AssertionError(f"lp_table accepted a {what} row")
    print("4 ok: missing / duplicate / foreign-subset rows rejected")

    # 5. Secondary numbers on a hand case: config "x" solves every Camera episode and nothing else on seed 0,
    #    everything on seed 1; per-seed means, per-category means over seeds, and clean LIBERO kept apart.
    cam = sum(e["category"] == "Camera Viewpoints" for e in sub["episodes"])
    r5 = rows_for(sub, "x", 0, lambda e: e["category"] == "Camera Viewpoints") + rows_for(sub, "x", 1, lambda e: 1)
    r5 += [L.result_row(L.CLEAN, "libero_goal", i, "", "x", 0, 41, "x-s0", "clean", i % 2) for i in range(10)]
    sec = L.secondary(r5, L.lp_table(r5, sub), sub)["x"]
    assert sec["lp_per_seed"] == {0: cam / n, 1: 1.0} and sec["lp_mean"] == (cam / n + 1) / 2
    assert sec["lp_per_category"]["Camera Viewpoints"] == 1.0 and sec["lp_per_category"]["Sensor Noise"] == 0.5
    assert sec["clean_per_seed"] == {0: 0.5}
    print("5 ok: secondary per-config, per-category and clean numbers")

    # 6. CLI end to end: subset file, rollout JSONL files, analyze prints the framing and the three seed deltas.
    with tempfile.TemporaryDirectory() as d:
        Path(d, "cls.json").write_text(json.dumps(cls))
        r = subprocess.run([sys.executable, "lp_eval.py", "subset", "--classification", str(Path(d, "cls.json")),
                            "--per-cell", "5", "--seed", "7", "--out", str(Path(d, "sub.json"))], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert json.loads(Path(d, "sub.json").read_text())["subset_id"] == sub["subset_id"]
        for i in range(2):
            Path(d, f"r{i}.jsonl").write_text("\n".join(json.dumps(x) for x in rows[i::2]))
        r = subprocess.run([sys.executable, "lp_eval.py", "analyze", "--subset", str(Path(d, "sub.json")), "--results",
                            str(Path(d, "r0.jsonl")), str(Path(d, "r1.jsonl")), "--out", str(Path(d, "p.json"))],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert L.FRAMING in r.stdout and f"{-2 / n:+14.4f}{-5 / n:+14.4f}{1 / n:+14.4f}" in r.stdout, r.stdout
        out = json.loads(Path(d, "p.json").read_text())
        assert [out["primary"][str(s)]["delta"] for s in range(3)] == [-2 / n, -5 / n, 1 / n]
    print("6 ok: CLI end to end")
    print("all checks passed")


if __name__ == "__main__":
    main()
