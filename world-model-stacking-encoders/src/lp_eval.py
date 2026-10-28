"""LIBERO-Plus evaluation bookkeeping and the primary paired analysis for the proxy replication (decision 003 and its
2026-10-28 amendment; experiments/2026-10-27-proxy-calibration/PLAN.md, "Pre-registered outcome rule").

This round is a proxy replication: its result is about the proxy (no pretrained Wan 2.2 video prior, much smaller
model) and decides whether a higher-fidelity, multi-seed WAM replication is worth running. A proxy that doesn't
reproduce the D+R drop does not show that the paper's result was training noise.

Three parts:
  subset   one fixed, seeded LIBERO-Plus episode subset, stratified by (suite, perturbation category), written once and
           reused by every checkpoint. Episodes are keyed (suite, id) as in LIBERO-plus 4976dc3
           libero/libero/benchmark/task_classification.json.
  records  one JSON line per rollout: benchmark, suite, episode_id, category, config, train_seed, eval_seed,
           checkpoint, subset_id, success. Rollout code (stage 3) writes these through `result_row`; aggregates are
           never the only thing saved.
  analyze  primary: for each training seed s, the per-episode paired difference success(A, s) - success(B, s) on the
           subset (A = dinov3+c-radiov4, B = dinov3 by default) and its mean; the three seed deltas are printed side by
           side. Secondary: each config's success (per seed and mean over seeds), per category; clean LIBERO as context.
           Three seeds can show a large, consistent drop; they don't support strong claims about seed variance.

Examples:
  python lp_eval.py subset --classification task_classification.json --per-cell 10 --seed 0 --out lp_subset.json
  python lp_eval.py analyze --subset lp_subset.json --results runs/*/rollouts.jsonl --out paired.json
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

LP = "libero_plus"
CLEAN = "libero"
FIELDS = ["benchmark", "suite", "episode_id", "category", "config", "train_seed", "eval_seed", "checkpoint",
          "subset_id", "success"]
FRAMING = ("Proxy replication: these numbers are about the proxy. Non-reproduction of the D+R drop would not show that "
           "the paper's result was training noise.")


# ---------------------------------------------------------------------------
# Subset
# ---------------------------------------------------------------------------

def subset_id(episodes):
    """Stable id of an episode list: sha256 of the sorted (suite, id) keys, first 16 hex chars."""
    keys = sorted(f"{e['suite']}/{e['id']}" for e in episodes)
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]


def make_subset(classification, per_cell, seed):
    """Up to `per_cell` episodes from every (suite, category) cell, sampled without replacement with one seeded
    generator in a fixed cell order. Returns the subset dict (header + episodes sorted by suite, id)."""
    rng = np.random.default_rng(seed)
    cells = defaultdict(list)
    for suite in sorted(classification):
        for e in classification[suite]:
            cells[(suite, e["category"])].append({"suite": suite, "id": int(e["id"]), "name": e["name"],
                                                   "category": e["category"],
                                                   "difficulty_level": e.get("difficulty_level")})
    eps = []
    for key in sorted(cells):
        c = sorted(cells[key], key=lambda e: e["id"])
        pick = rng.choice(len(c), size=min(per_cell, len(c)), replace=False)
        eps += [c[i] for i in sorted(pick)]
    eps.sort(key=lambda e: (e["suite"], e["id"]))
    return {"subset_id": subset_id(eps), "seed": seed, "per_cell": per_cell, "n": len(eps),
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
    """One rollout outcome, as written by the rollout code (one JSON line each)."""
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
    """{(config, train_seed): {(suite, episode_id): success}} for LIBERO-Plus rows. Every (config, seed) must cover
    exactly the subset's episodes, once each, from that subset (a paired analysis on mismatched episodes is invalid,
    so this fails loudly instead of intersecting)."""
    want = {(e["suite"], e["id"]) for e in subset["episodes"]}
    table = defaultdict(dict)
    for r in rows:
        if r["benchmark"] != LP:
            continue
        assert r["subset_id"] == subset["subset_id"], f"row from subset {r['subset_id']}, expected {subset['subset_id']}"
        key = (r["suite"], r["episode_id"])
        cell = table[(r["config"], r["train_seed"])]
        assert key not in cell, f"duplicate rollout {key} for {r['config']} seed {r['train_seed']}"
        cell[key] = r["success"]
    for cs, cell in table.items():
        assert set(cell) == want, (f"{cs}: {len(want - set(cell))} subset episodes missing, "
                                   f"{len(set(cell) - want)} not in the subset")
    return table


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def paired(table, subset, a, b):
    """Primary analysis. Per training seed present for both configs: per-episode d = success(a) - success(b) on the
    subset, its mean, and the discordant pairs (a fails where b succeeds, and the reverse)."""
    keys = sorted((e["suite"], e["id"]) for e in subset["episodes"])
    seeds = sorted({s for c, s in table if c == a} & {s for c, s in table if c == b})
    out = {}
    for s in seeds:
        d = np.array([table[(a, s)][k] - table[(b, s)][k] for k in keys])
        out[s] = {"delta": float(d.mean()), "n": len(d), "a_worse": int((d < 0).sum()), "a_better": int((d > 0).sum()),
                  "per_episode": {f"{k[0]}/{k[1]}": int(x) for k, x in zip(keys, d)}}
    return out


def secondary(rows, table, subset):
    """Each config's LIBERO-Plus success per seed, mean over seeds, per category (mean over seeds); clean LIBERO."""
    cat = {(e["suite"], e["id"]): e["category"] for e in subset["episodes"]}
    configs = sorted({c for c, _ in table})
    out = {}
    for c in configs:
        seeds = sorted(s for cc, s in table if cc == c)
        per_seed = {s: float(np.mean(list(table[(c, s)].values()))) for s in seeds}
        per_cat = defaultdict(list)
        for s in seeds:
            by = defaultdict(list)
            for k, v in table[(c, s)].items():
                by[cat[k]].append(v)
            for g, v in by.items():
                per_cat[g].append(float(np.mean(v)))
        clean = defaultdict(list)
        for r in rows:
            if r["benchmark"] == CLEAN and r["config"] == c:
                clean[r["train_seed"]].append(r["success"])
        out[c] = {"lp_per_seed": per_seed, "lp_mean": float(np.mean(list(per_seed.values()))),
                  "lp_per_category": {g: float(np.mean(v)) for g, v in sorted(per_cat.items())},
                  "clean_per_seed": {s: float(np.mean(v)) for s, v in sorted(clean.items())}}
    return out


def report(prim, sec, a, b):
    lines = [FRAMING, "", f"PRIMARY: paired {a} - {b} on LIBERO-Plus, per training seed (same episodes)"]
    seeds = sorted(prim)
    lines.append("  seed      " + "".join(f"{s:>14}" for s in seeds))
    lines.append("  delta     " + "".join(f"{prim[s]['delta']:>+14.4f}" for s in seeds))
    lines.append("  a worse   " + "".join(f"{prim[s]['a_worse']:>14d}" for s in seeds))
    lines.append("  a better  " + "".join(f"{prim[s]['a_better']:>14d}" for s in seeds))
    lines.append("  episodes  " + "".join(f"{prim[s]['n']:>14d}" for s in seeds))
    if seeds:
        lines.append(f"  (secondary) mean of the seed deltas: {np.mean([prim[s]['delta'] for s in seeds]):+.4f}")
    lines += ["", "SECONDARY: LIBERO-Plus success per config (per seed; mean over seeds)"]
    for c, v in sec.items():
        ps = ", ".join(f"s{s} {x:.4f}" for s, x in v["lp_per_seed"].items())
        lines.append(f"  {c:<22} mean {v['lp_mean']:.4f}  ({ps})")
    lines += ["", "SECONDARY: LIBERO-Plus success per category (mean over seeds)"]
    for c, v in sec.items():
        lines.append(f"  {c:<22} " + ", ".join(f"{g} {x:.4f}" for g, x in v["lp_per_category"].items()))
    if any(v["clean_per_seed"] for v in sec.values()):
        lines += ["", "CONTEXT: clean LIBERO success per config (per seed)"]
        for c, v in sec.items():
            lines.append(f"  {c:<22} " + ", ".join(f"s{s} {x:.4f}" for s, x in v["clean_per_seed"].items()))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    s1 = sp.add_parser("subset")
    s1.add_argument("--classification", required=True, help="LIBERO-plus task_classification.json (4976dc3)")
    s1.add_argument("--per-cell", type=int, required=True)
    s1.add_argument("--seed", type=int, default=0)
    s1.add_argument("--out", required=True)
    s2 = sp.add_parser("analyze")
    s2.add_argument("--subset", required=True)
    s2.add_argument("--results", nargs="+", required=True, help="rollout JSONL files")
    s2.add_argument("--a", default="dinov3+c-radiov4")
    s2.add_argument("--b", default="dinov3")
    s2.add_argument("--out", default=None, help="JSON with all numbers, incl. per-episode differences")
    args = p.parse_args()

    if args.cmd == "subset":
        raw = Path(args.classification).read_bytes()
        sub = make_subset(json.loads(raw), args.per_cell, args.seed)
        sub["classification_sha256"] = hashlib.sha256(raw).hexdigest()
        Path(args.out).write_text(json.dumps(sub, indent=1))
        print(f"wrote {args.out}: {sub['n']} episodes, subset_id {sub['subset_id']}")
        return
    sub = load_subset(args.subset)
    rows = read_rows(args.results)
    table = lp_table(rows, sub)
    prim, sec = paired(table, sub, args.a, args.b), secondary(rows, table, sub)
    print(report(prim, sec, args.a, args.b))
    if args.out:
        Path(args.out).write_text(json.dumps({"framing": FRAMING, "a": args.a, "b": args.b,
                                              "subset_id": sub["subset_id"], "primary": prim, "secondary": sec},
                                             indent=1))


if __name__ == "__main__":
    main()
