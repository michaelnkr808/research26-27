"""LIBERO-Plus evaluation bookkeeping and the pre-registered analysis for the proxy replication (decision 003 and its
amendments; experiments/2026-10-27-proxy-calibration/PLAN.md, "Pre-registered outcome rule", incl. "Evaluation
protocol" (locked 2026-10-28 (g)) and "Sample-size rule" (locked 2026-10-28 (h))). This file implements that rule; it
never changes it.

This round is a proxy replication: its result is about the proxy (no pretrained Wan 2.2 video prior, much smaller
model) and decides whether a higher-fidelity, multi-seed WAM replication is worth running. A proxy that doesn't
reproduce the D+R drop does not show that the paper's result was training noise.

Parts:
  subset   one fixed, seeded LIBERO-Plus episode subset, sampled in proportion to each of the 28 (suite, perturbation
           category) cells' size in the full benchmark (total N, largest-remainder rounding, >= 1 per cell, one seeded
           generator), optionally excluding earlier subsets (the pilot's episodes). Written once, reused by every
           checkpoint. Episodes are keyed (suite, id) as in LIBERO-plus 4976dc3 task_classification.json.
  records  one JSON line per rollout (episode x eval seed): benchmark, suite, episode_id, category, config,
           train_seed, eval_seed, checkpoint, subset_id, success. Every subset episode runs under eval seeds 41, 42, 43.
  analyze  per-episode score = mean success over the three eval seeds. Paired per training seed:
           delta_s(A - B) = mean over the subset of score(A, s) - score(B, s), in percentage points, with discordant
           counts. Calibration (D - none, D+L - D), then the primary D+R - D, then the locked verdict. Secondary:
           per-config success per seed, per category; clean LIBERO as context. Optional CIs resample whole episodes.
  sizing   paired pilot (dinov3 and dinov3+c-radiov4, same train seed, same pilot episodes): sd of the per-episode
           difference, sd* = 1.5 sd, N = ceil((sd* / 1.35)^2).

Examples:
  python lp_eval.py subset --classification task_classification.json --n 300 --seed 0 --out lp_subset.json
  python lp_eval.py subset --classification task_classification.json --n 900 --seed 1 --exclude pilot.json --out final.json
  python lp_eval.py sizing --subset pilot.json --results pilot_rollouts/*.jsonl
  python lp_eval.py analyze --subset final.json --results rollouts/*.jsonl --out paired.json
  python lp_eval.py seednoise --a-dir rollouts/eval-dinov3-s1 --b-dir rollouts/eval-dinov3-s0 --subset full-subset.json

seednoise applies the locked seed-noise rule (PLAN "Seed-noise rule (locked 2026-11-06)") to the two finished dinov3
runs (A = training seed 1, B = training seed 0), each a directory of 8 shard files `*of8.jsonl`.
"""

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

LP = "libero_plus"
CLEAN = "libero"
FIELDS = ["benchmark", "suite", "episode_id", "category", "config", "train_seed", "eval_seed", "checkpoint",
          "subset_id", "success"]
FRAMING = ("Proxy replication: these numbers are about the proxy. Non-reproduction of the D+R drop would not show that "
           "the paper's result was training noise.")

EVAL_SEEDS = (41, 42, 43)            # paper Table 10 columns; every subset episode runs under all three
TRAIN_SEEDS = (0, 1, 2)
NONE, D, R, DR, DL = "none", "dinov3", "c-radiov4", "dinov3+c-radiov4", "dinov3+siglip2"

# Locked thresholds, percentage points (PLAN outcome rule, 2026-10-28 (f)). Exact decimals; compared as fractions so
# boundary values land on the side the rule states.
CAL_A_MIN_MEAN = Fraction("2.7")     # (a) D - none: every seed > 0 and mean >= +2.7
CAL_B_MIN_MEAN = Fraction("-1.35")   # (b) D+L - D: mean >= -1.35 (non-inferiority, half the harm threshold)
HARM_MAX_MEAN = Fraction("-2.7")     # D+R - D: every seed < 0 and mean <= -2.7 (half the paper's -5.41)
SE_TARGET = 1.35                     # sample-size rule: sd* / sqrt(N) <= 1.35 points
SD_INFLATION = 1.5                   # sd* = 1.5 sd (variance x 2.25)

# Seed-noise rule (PLAN "Seed-noise rule (locked 2026-11-06)"), percentage points, exact.
SN_T = Fraction("2.7")               # threshold on |noise delta| and floor on each seed's mean LIBERO-Plus success
SN_SHARDS = 8
SN_CANT_TELL = "can't tell; user decides"
SN_RETHINK = "3 seeds can't settle the question; rethink seeds (or N) before the other 13 runs"
SN_GO = "go ahead"
SN_GPU_NOTE = ("Note: the two training seeds may have trained on different GPU models (LOG 2026-11-06 (c)); "
               "noted, not corrected for.")
SN_PARTIAL = "PARTIAL INPUT: not a verdict"

V_CAL_FAIL = "calibration failed: don't interpret the D+R comparison from this proxy"
V_EFFECT = "substantial proxy effect (trigger for the higher-fidelity run, not statistical proof)"
V_NO_EFFECT = "the proxy does not show a substantial effect"


# ---------------------------------------------------------------------------
# Subset
# ---------------------------------------------------------------------------

def subset_id(episodes):
    """Stable id of an episode list: sha256 of the sorted (suite, id) keys, first 16 hex chars."""
    keys = sorted(f"{e['suite']}/{e['id']}" for e in episodes)
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]


def allocate(sizes, n):
    """Proportional allocation of n over cells {cell: size}: largest-remainder rounding of n * size / total, then at
    least 1 per cell (taken back from the cells with the smallest remainders among those holding > 1). Sums to n."""
    assert n >= len(sizes), f"n={n} is below one per cell ({len(sizes)} cells)"
    total = sum(sizes.values())
    quota = {c: Fraction(n * s, total) for c, s in sizes.items()}
    alloc = {c: math.floor(q) for c, q in quota.items()}
    rem = sorted(sizes, key=lambda c: (-(quota[c] - alloc[c]), c))          # largest remainder first, ties by name
    for c in rem[:n - sum(alloc.values())]:
        alloc[c] += 1
    for c in sorted(c for c in alloc if alloc[c] == 0):
        alloc[c] = 1
        donor = min((d for d in alloc if alloc[d] > 1), key=lambda d: (quota[d] - alloc[d], d))
        alloc[donor] -= 1
    assert sum(alloc.values()) == n
    return alloc


def make_subset(classification, n, seed, exclude=()):
    """n episodes sampled in proportion to the full benchmark's (suite, category) cell sizes, one seeded generator in a
    fixed cell order, never drawing a (suite, id) in `exclude`. Returns the subset dict (header + episodes sorted by
    suite, id)."""
    exclude = {(s, int(i)) for s, i in exclude}
    rng = np.random.default_rng(seed)
    cells = defaultdict(list)
    for suite in sorted(classification):
        for e in classification[suite]:
            cells[(suite, e["category"])].append({"suite": suite, "id": int(e["id"]), "name": e["name"],
                                                   "category": e["category"],
                                                   "difficulty_level": e.get("difficulty_level")})
    alloc = allocate({k: len(v) for k, v in cells.items()}, n)             # shares from the full benchmark
    eps = []
    for key in sorted(cells):
        c = sorted((e for e in cells[key] if (e["suite"], e["id"]) not in exclude), key=lambda e: e["id"])
        assert alloc[key] <= len(c), f"cell {key}: wants {alloc[key]}, only {len(c)} left after exclusion"
        pick = rng.choice(len(c), size=alloc[key], replace=False)
        eps += [c[i] for i in sorted(pick)]
    eps.sort(key=lambda e: (e["suite"], e["id"]))
    return {"subset_id": subset_id(eps), "seed": seed, "n": len(eps), "allocation": "proportional",
            "excluded": len(exclude),
            "cells": {f"{s}/{c}": sum(1 for e in eps if (e["suite"], e["category"]) == (s, c)) for s, c in sorted(cells)},
            "episodes": eps}


def load_subset(path):
    sub = json.loads(Path(path).read_text())
    assert subset_id(sub["episodes"]) == sub["subset_id"], f"{path}: episode list doesn't match its subset_id"
    return sub


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def result_row(benchmark, suite, episode_id, category, config, train_seed, eval_seed, checkpoint, subset_id_, success):
    """One rollout outcome (one episode under one eval seed), as written by the rollout code (one JSON line each)."""
    assert benchmark in (LP, CLEAN), benchmark
    return {"benchmark": benchmark, "suite": suite, "episode_id": int(episode_id), "category": category,
            "config": config, "train_seed": int(train_seed), "eval_seed": int(eval_seed), "checkpoint": str(checkpoint),
            "subset_id": subset_id_, "success": int(bool(success))}


def read_rows(paths):
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                missing = [f for f in FIELDS if f not in r]
                assert not missing, f"{p}: row without {missing}"
                rows.append(r)
    return rows


def lp_table(rows, subset):
    """{(config, train_seed): {(suite, episode_id): successes over eval seeds 41, 42, 43 (0..3)}} for LIBERO-Plus rows.
    Every (config, seed) must cover exactly the subset's episodes, each under exactly the three eval seeds once (a
    paired analysis on mismatched episodes is invalid, so this fails loudly instead of intersecting)."""
    want = {(e["suite"], e["id"]) for e in subset["episodes"]}
    raw = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["benchmark"] != LP:
            continue
        assert r["subset_id"] == subset["subset_id"], f"row from subset {r['subset_id']}, expected {subset['subset_id']}"
        assert r["eval_seed"] in EVAL_SEEDS, f"eval seed {r['eval_seed']} not in {EVAL_SEEDS}"
        key = (r["suite"], r["episode_id"])
        cell = raw[(r["config"], r["train_seed"])][key]
        assert r["eval_seed"] not in cell, \
            f"duplicate rollout {key} eval seed {r['eval_seed']} for {r['config']} seed {r['train_seed']}"
        cell[r["eval_seed"]] = r["success"]
    table = {}
    for cs, eps in raw.items():
        assert set(eps) == want, (f"{cs}: {len(want - set(eps))} subset episodes missing, "
                                  f"{len(set(eps) - want)} not in the subset")
        for k, by_seed in eps.items():
            assert set(by_seed) == set(EVAL_SEEDS), f"{cs} {k}: eval seeds {sorted(by_seed)}, need {EVAL_SEEDS}"
        table[cs] = {k: sum(by_seed.values()) for k, by_seed in eps.items()}
    return table


def score(successes):
    """Per-episode score: mean success over the three eval seeds, exact (0, 1/3, 2/3, 1)."""
    return Fraction(successes, len(EVAL_SEEDS))


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def episode_ci(d_pts, draws=10000, seed=0, level=0.95):
    """Percentile CI of the mean per-episode difference, resampling whole episodes (each with its three eval-seed
    rollouts already averaged into d), never rollouts. Describes episode sampling only, not training-seed noise."""
    d = np.asarray(d_pts, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(draws, len(d)))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def paired(table, subset, a, b, ci_draws=0):
    """Per training seed present for both configs: per-episode d = score(a) - score(b) (eval seeds averaged first),
    delta = mean d in percentage points (exact fraction kept for the verdict), discordant counts (a scored lower /
    higher than b)."""
    keys = sorted((e["suite"], e["id"]) for e in subset["episodes"])
    seeds = sorted({s for c, s in table if c == a} & {s for c, s in table if c == b})
    out = {}
    for s in seeds:
        d = [score(table[(a, s)][k]) - score(table[(b, s)][k]) for k in keys]
        delta = sum(d, Fraction(0)) * 100 / len(d)
        out[s] = {"delta_pts": float(delta), "delta_exact": delta, "n": len(d),
                  "a_worse": sum(x < 0 for x in d), "a_better": sum(x > 0 for x in d),
                  "per_episode_pts": {f"{k[0]}/{k[1]}": float(x * 100) for k, x in zip(keys, d)}}
        if ci_draws:
            out[s]["ci95_episodes_pts"] = episode_ci([float(x * 100) for x in d], draws=ci_draws, seed=s)
    return out


def _seed_deltas(table, subset, a, b):
    p = paired(table, subset, a, b)
    assert sorted(p) == list(TRAIN_SEEDS), f"{a} - {b}: train seeds {sorted(p)}, need {TRAIN_SEEDS}"
    vals = [p[s]["delta_exact"] for s in TRAIN_SEEDS]
    return p, vals, sum(vals, Fraction(0)) / len(vals)


def verdict(table, subset):
    """The locked outcome rule, applied in order: calibration (a) and (b), then D+R vs D."""
    pairs, exact = {}, {}
    for name, a, b in [("D-none", D, NONE), ("DL-D", DL, D), ("DR-D", DR, D)]:
        p, vals, mean = _seed_deltas(table, subset, a, b)
        exact[name] = (vals, mean)
        pairs[name] = {"per_seed_pts": [float(v) for v in vals], "mean_pts": float(mean),
                       "a_worse": [p[s]["a_worse"] for s in TRAIN_SEEDS],
                       "a_better": [p[s]["a_better"] for s in TRAIN_SEEDS]}
    a_ok = all(v > 0 for v in exact["D-none"][0]) and exact["D-none"][1] >= CAL_A_MIN_MEAN
    b_ok = exact["DL-D"][1] >= CAL_B_MIN_MEAN
    if not (a_ok and b_ok):
        v = V_CAL_FAIL
    elif all(x < 0 for x in exact["DR-D"][0]) and exact["DR-D"][1] <= HARM_MAX_MEAN:
        v = V_EFFECT
    else:
        v = V_NO_EFFECT
    return {"calibration_a": a_ok, "calibration_b": b_ok, "verdict": v, "pairs": pairs,
            "thresholds_pts": {"cal_a_min_mean": float(CAL_A_MIN_MEAN), "cal_b_min_mean": float(CAL_B_MIN_MEAN),
                               "harm_max_mean": float(HARM_MAX_MEAN)}}


def sizing(table, subset, train_seed, a=DR, b=D):
    """Sample-size rule: sd (sample sd, ddof 1) of the per-episode paired difference a - b in points on the pilot
    episodes (eval seeds averaged first), sd* = 1.5 sd, N = ceil((sd* / 1.35)^2)."""
    p = paired(table, subset, a, b)
    assert train_seed in p, f"pilot needs {a} and {b} at train seed {train_seed}"
    d = np.array(list(p[train_seed]["per_episode_pts"].values()))
    sd = float(d.std(ddof=1))
    sd_star = SD_INFLATION * sd
    return {"train_seed": train_seed, "n_pilot": len(d), "sd_pts": sd, "sd_star_pts": sd_star,
            "N": math.ceil((sd_star / SE_TARGET) ** 2), "se_target_pts": SE_TARGET}


def secondary(rows, table, subset):
    """Each config's LIBERO-Plus success (points) per seed, mean over seeds, per category (mean over seeds), from the
    per-episode scores; clean LIBERO success per seed as context."""
    cat = {(e["suite"], e["id"]): e["category"] for e in subset["episodes"]}
    configs = sorted({c for c, _ in table})
    out = {}
    for c in configs:
        seeds = sorted(s for cc, s in table if cc == c)
        per_seed = {s: float(np.mean([float(score(v)) for v in table[(c, s)].values()]) * 100) for s in seeds}
        per_cat = defaultdict(list)
        for s in seeds:
            by = defaultdict(list)
            for k, v in table[(c, s)].items():
                by[cat[k]].append(float(score(v)))
            for g, v in by.items():
                per_cat[g].append(float(np.mean(v)) * 100)
        clean = defaultdict(list)
        for r in rows:
            if r["benchmark"] == CLEAN and r["config"] == c:
                clean[r["train_seed"]].append(r["success"])
        out[c] = {"lp_per_seed_pts": per_seed, "lp_mean_pts": float(np.mean(list(per_seed.values()))),
                  "lp_per_category_pts": {g: float(np.mean(v)) for g, v in sorted(per_cat.items())},
                  "clean_per_seed_pts": {s: float(np.mean(v)) * 100 for s, v in sorted(clean.items())}}
    return out


def _row(label, vals, fmt):
    return f"  {label:<10}" + "".join(format(v, fmt) for v in vals)


def report(prim, sec, a, b, ver=None):
    lines = [FRAMING, ""]
    if ver:
        lines.append("PRE-REGISTERED RULE (percentage points; per-episode score = mean over eval seeds 41-43)")
        for name, label in [("D-none", "calibration (a) D - none"), ("DL-D", "calibration (b) D+L - D"),
                            ("DR-D", "primary D+R - D")]:
            x = ver["pairs"][name]
            lines.append(f"  {label:<28} seeds " + ", ".join(f"{v:+.2f}" for v in x["per_seed_pts"])
                         + f"   average {x['mean_pts']:+.2f}")
        lines.append(f"  (a) {'pass' if ver['calibration_a'] else 'FAIL'}   (b) {'pass' if ver['calibration_b'] else 'FAIL'}")
        lines.append(f"  VERDICT: {ver['verdict']}")
        lines.append("")
    lines.append(f"PAIRED: {a} - {b} on LIBERO-Plus, per training seed (same episodes), points")
    seeds = sorted(prim)
    lines.append(_row("seed", seeds, ">14"))
    lines.append(_row("delta", [prim[s]["delta_pts"] for s in seeds], ">+14.2f"))
    lines.append(_row("a lower", [prim[s]["a_worse"] for s in seeds], ">14d"))
    lines.append(_row("a higher", [prim[s]["a_better"] for s in seeds], ">14d"))
    lines.append(_row("episodes", [prim[s]["n"] for s in seeds], ">14d"))
    if seeds and "ci95_episodes_pts" in prim[seeds[0]]:
        lines.append("  95% CI over episodes (episode sampling only, not training-seed noise): "
                     + ", ".join(f"s{s} [{prim[s]['ci95_episodes_pts'][0]:+.2f}, {prim[s]['ci95_episodes_pts'][1]:+.2f}]"
                                 for s in seeds))
    if seeds:
        lines.append(f"  average of the seed deltas: {np.mean([prim[s]['delta_pts'] for s in seeds]):+.2f}")
    lines += ["", "SECONDARY: LIBERO-Plus success per config, points (per seed; mean over seeds)"]
    for c, v in sec.items():
        ps = ", ".join(f"s{s} {x:.2f}" for s, x in v["lp_per_seed_pts"].items())
        lines.append(f"  {c:<22} mean {v['lp_mean_pts']:.2f}  ({ps})")
    lines += ["", "SECONDARY: LIBERO-Plus success per category, points (mean over seeds)"]
    for c, v in sec.items():
        lines.append(f"  {c:<22} " + ", ".join(f"{g} {x:.2f}" for g, x in v["lp_per_category_pts"].items()))
    if any(v["clean_per_seed_pts"] for v in sec.values()):
        lines += ["", "CONTEXT: clean LIBERO success per config, points (per seed)"]
        for c, v in sec.items():
            lines.append(f"  {c:<22} " + ", ".join(f"s{s} {x:.2f}" for s, x in v["clean_per_seed_pts"].items()))
    return "\n".join(lines)


def seednoise_outcome(mean_a, mean_b, delta):
    """The locked seed-noise outcome from exact Fractions (points). Precedence: floor, then |delta| >= T, else go."""
    if mean_a < SN_T or mean_b < SN_T:
        return SN_CANT_TELL
    if abs(delta) >= SN_T:
        return SN_RETHINK
    return SN_GO


def _sn_load(directory):
    """(shard files, rows) of one seed's directory. Only `*of8.jsonl`, never `*.jsonl` (that would also match the
    `.errors.jsonl` files)."""
    files = sorted(Path(directory).glob("*of8.jsonl"))
    return files, read_rows(files)


def _sn_episodes(rows):
    """{(suite, episode_id): {eval_seed: success}} for LIBERO-Plus rows (duplicates are caught separately)."""
    eps = defaultdict(dict)
    for r in rows:
        if r["benchmark"] == LP:
            eps[(r["suite"], r["episode_id"])][r["eval_seed"]] = r["success"]
    return eps


def seednoise(a_dir, b_dir, subset_path=None):
    """Apply the locked seed-noise rule. Returns a dict with `problems` (completeness failures; empty = complete),
    the numbers computed on the episodes both seeds have, and `outcome` (None whenever there are problems)."""
    problems = []
    runs = {}
    for name, d in (("A", a_dir), ("B", b_dir)):
        files, rows = _sn_load(d)
        if len(files) != SN_SHARDS:
            problems.append(f"seed {name}: {len(files)} shard files matching *of8.jsonl in {d}, need {SN_SHARDS}")
        runs[name] = (files, rows)
    all_rows = runs["A"][1] + runs["B"][1]
    if not all_rows:
        problems.append("no rows at all")
    configs = sorted({r["config"] for r in all_rows})
    if len(configs) != 1:
        problems.append(f"configs {configs}, need exactly one (the same in both seeds)")
    tseeds = {name: sorted({r["train_seed"] for r in runs[name][1]}) for name in "AB"}
    for name in "AB":
        if len(tseeds[name]) != 1:
            problems.append(f"seed {name}: train_seed values {tseeds[name]}, need exactly one")
    if len(tseeds["A"]) == 1 and len(tseeds["B"]) == 1 and tseeds["A"] == tseeds["B"]:
        problems.append(f"A and B have the same train_seed {tseeds['A'][0]}, need different ones")
    sids = sorted({r["subset_id"] for r in all_rows})
    if len(sids) > 1:
        problems.append(f"subset_id values {sids}, need one")
    dups = 0
    for name in "AB":
        cnt = defaultdict(int)
        for r in runs[name][1]:
            cnt[(r["benchmark"], r["suite"], r["episode_id"], r["eval_seed"])] += 1
        dups += sum(1 for n in cnt.values() if n > 1)
    if dups:
        problems.append(f"{dups} duplicate (benchmark, suite, episode_id, eval_seed) keys within a seed")
    eps = {name: _sn_episodes(runs[name][1]) for name in "AB"}
    for name in "AB":
        bad = [k for k, v in eps[name].items() if set(v) != set(EVAL_SEEDS)]
        if bad:
            problems.append(f"seed {name}: {len(bad)} LIBERO-Plus episodes without exactly eval seeds {EVAL_SEEDS}, "
                            f"e.g. {bad[0]} has {sorted(eps[name][bad[0]])}")
    ka, kb = set(eps["A"]), set(eps["B"])
    if ka != kb:
        problems.append(f"episode sets differ: {len(ka - kb)} only in A, {len(kb - ka)} only in B")
    if subset_path:
        sub = load_subset(subset_path)
        want = {(e["suite"], e["id"]) for e in sub["episodes"]}
        for name in "AB":
            have = set(eps[name])
            if have != want:
                problems.append(f"seed {name}: episodes differ from the subset file: {len(want - have)} missing, "
                                f"{len(have - want)} extra")
        if sids and sids != [sub["subset_id"]]:
            problems.append(f"rows' subset_id {sids} differs from the subset file's {sub['subset_id']}")
    # numbers, on the episodes both seeds have (all of them when complete); mean over the eval seeds present
    common = sorted(ka & kb)
    sc = {name: {k: Fraction(sum(eps[name][k].values()), len(eps[name][k])) for k in common} for name in "AB"}
    n = len(common)
    zero = Fraction(0)
    mean_a = sum(sc["A"].values(), zero) * 100 / n if n else zero
    mean_b = sum(sc["B"].values(), zero) * 100 / n if n else zero
    d = [sc["A"][k] - sc["B"][k] for k in common]
    delta = sum(d, zero) * 100 / n if n else zero
    return {"problems": problems, "config": configs[0] if len(configs) == 1 else configs,
            "train_seed_a": tseeds["A"], "train_seed_b": tseeds["B"], "subset_id": sids,
            "shards": {"A": len(runs["A"][0]), "B": len(runs["B"][0])},
            "n_episodes": n, "n_episodes_a": len(ka), "n_episodes_b": len(kb),
            "clean_records_excluded": {name: sum(1 for r in runs[name][1] if r["benchmark"] == CLEAN) for name in "AB"},
            "mean_a_pts": mean_a, "mean_b_pts": mean_b, "delta_pts": delta,
            "a_lower": sum(x < 0 for x in d), "a_higher": sum(x > 0 for x in d), "equal": sum(x == 0 for x in d),
            "threshold_pts": SN_T, "outcome": None if problems else seednoise_outcome(mean_a, mean_b, delta)}


def seednoise_report(r, partial):
    lines = []
    if partial:
        lines += [SN_PARTIAL, ""]
    if r["problems"]:
        lines.append("Completeness problems (ignored because of --partial):" if partial
                     else "COMPLETENESS FAILURES (no outcome):")
        lines += [f"  - {p}" for p in r["problems"]]
        lines.append("")
    lines += [f"config {r['config']}; A = training seed {r['train_seed_a']}, B = training seed {r['train_seed_b']}; "
              f"subset_id {r['subset_id']}",
              f"shard files: A {r['shards']['A']}, B {r['shards']['B']}",
              f"LIBERO-Plus episodes compared: {r['n_episodes']} (A has {r['n_episodes_a']}, B has {r['n_episodes_b']})",
              f"clean-LIBERO records excluded from the rule: A {r['clean_records_excluded']['A']}, "
              f"B {r['clean_records_excluded']['B']}",
              f"mean LIBERO-Plus success (points): A {float(r['mean_a_pts']):.3f}, B {float(r['mean_b_pts']):.3f}",
              f"noise delta A - B (points, with sign): {float(r['delta_pts']):+.3f}",
              f"episodes with A lower: {r['a_lower']}, A higher: {r['a_higher']}, equal: {r['equal']}",
              f"threshold T = {float(r['threshold_pts'])} points",
              SN_GPU_NOTE]
    if partial:
        lines += ["", SN_PARTIAL + " (no outcome line is printed)"]
    elif not r["problems"]:
        lines += ["", f"OUTCOME: {r['outcome']}"]
    return "\n".join(lines)


def _jsonable(prim):
    return {s: {k: v for k, v in x.items() if k != "delta_exact"} for s, x in prim.items()}


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    s1 = sp.add_parser("subset")
    s1.add_argument("--classification", required=True, help="LIBERO-plus task_classification.json (4976dc3)")
    s1.add_argument("--n", type=int, required=True, help="total episodes, allocated in proportion to cell size")
    s1.add_argument("--seed", type=int, required=True)
    s1.add_argument("--exclude", nargs="*", default=[], help="subset files whose episodes must not be drawn (pilot)")
    s1.add_argument("--out", required=True)
    s2 = sp.add_parser("analyze")
    s2.add_argument("--subset", required=True)
    s2.add_argument("--results", nargs="+", required=True, help="rollout JSONL files")
    s2.add_argument("--a", default=DR)
    s2.add_argument("--b", default=D)
    s2.add_argument("--ci-draws", type=int, default=0, help="episode-level bootstrap draws for CIs (0 = none)")
    s2.add_argument("--no-verdict", action="store_true", help="skip the locked rule (e.g. incomplete config set)")
    s2.add_argument("--out", default=None, help="JSON with all numbers, incl. per-episode differences")
    s3 = sp.add_parser("sizing")
    s3.add_argument("--subset", required=True, help="the pilot's subset file")
    s3.add_argument("--results", nargs="+", required=True)
    s3.add_argument("--train-seed", type=int, default=0)
    s3.add_argument("--out", default=None)
    s4 = sp.add_parser("seednoise", help="locked seed-noise rule on two finished dinov3 runs (A = seed 1, B = seed 0)")
    s4.add_argument("--a-dir", required=True, help="shard directory of training seed 1 (A)")
    s4.add_argument("--b-dir", required=True, help="shard directory of training seed 0 (B)")
    s4.add_argument("--subset", default=None, help="full-subset.json to check the episode list against")
    s4.add_argument("--partial", action="store_true", help="skip completeness failures (format check); no outcome")
    s4.add_argument("--json-out", default=None, help="machine-readable summary")
    args = p.parse_args()

    if args.cmd == "seednoise":
        r = seednoise(args.a_dir, args.b_dir, args.subset)
        if args.partial:
            r["outcome"] = None
        print(seednoise_report(r, args.partial))
        if args.json_out and (args.partial or not r["problems"]):
            out = {k: (float(v) if isinstance(v, Fraction) else v) for k, v in r.items()}
            out.update(partial=args.partial, gpu_note=SN_GPU_NOTE)
            if args.partial:
                out["note"] = SN_PARTIAL
            Path(args.json_out).write_text(json.dumps(out, indent=1))
        if r["problems"] and not args.partial:
            sys.exit(1)
        return

    if args.cmd == "subset":
        raw = Path(args.classification).read_bytes()
        excl_subs = [load_subset(f) for f in args.exclude]
        excl = [(e["suite"], e["id"]) for s in excl_subs for e in s["episodes"]]
        sub = make_subset(json.loads(raw), args.n, args.seed, exclude=excl)
        sub["classification_sha256"] = hashlib.sha256(raw).hexdigest()
        sub["excluded_subsets"] = [s["subset_id"] for s in excl_subs]
        Path(args.out).write_text(json.dumps(sub, indent=1))
        print(f"wrote {args.out}: {sub['n']} episodes, subset_id {sub['subset_id']}, {len(excl)} episodes excluded")
        return
    sub = load_subset(args.subset)
    rows = read_rows(args.results)
    table = lp_table(rows, sub)
    if args.cmd == "sizing":
        z = sizing(table, sub, args.train_seed)
        print(f"pilot sizing (train seed {z['train_seed']}, {z['n_pilot']} episodes): sd {z['sd_pts']:.3f} pts, "
              f"sd* {z['sd_star_pts']:.3f} pts, N {z['N']} (SE target {SE_TARGET} pts)")
        if args.out:
            Path(args.out).write_text(json.dumps(dict(z, subset_id=sub["subset_id"]), indent=1))
        return
    prim, sec = paired(table, sub, args.a, args.b, ci_draws=args.ci_draws), secondary(rows, table, sub)
    ver = None if args.no_verdict else verdict(table, sub)
    print(report(prim, sec, args.a, args.b, ver))
    if args.out:
        Path(args.out).write_text(json.dumps({"framing": FRAMING, "a": args.a, "b": args.b,
                                              "subset_id": sub["subset_id"], "verdict": ver,
                                              "primary": _jsonable(prim), "secondary": sec}, indent=1))


if __name__ == "__main__":
    main()
