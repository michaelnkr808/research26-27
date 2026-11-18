"""Upper-bound LIBERO-Plus rate from the timing-only LIBERO-Plus shard (job-smoke, LP_TIMING=1), for job-lp-eval's caps.

Written 2026-11-17 for RUN-stage2-eval.md. Reads the timing-only records file (one row per episode: suite, category, wall_s;
no outcome) and the committed full-benchmark episode list, and prints per cell and in total:
  - cell weight n_cell = episodes of that (suite, category) cell in the 10,030 (each runs under 3 eval seeds),
  - R_weighted = 3600 x ENVS / (sum over cells of n_cell x wall_s(cell) / 10,030): episodes per pod-hour if every episode
    took its cell's timing-only wall time with ENVS simulator processes busy at once,
  - the upper-bound pod-hours for one checkpoint (30,090 rollouts) at R_weighted.
Timing-only episodes all run to their step limit, so these are upper bounds on time (stage 1: real clean rollouts ran about
2.6 times faster). Use the lower of R_weighted and the pass's raw rate (records / rollout seconds of the pass, from the pod
log's "[pass] lp-timing" line) as --lp-rate, rounded down.

  python3 lp_timing_rate.py --records smoke-lp-timing-try1.jsonl --envs 4 \
      [--subset ../../2026-10-27-proxy-calibration/results/full-benchmark/full-subset.json] [--pass-seconds S]
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SUBSET = HERE.parent.parent / "2026-10-27-proxy-calibration" / "results" / "full-benchmark" / "full-subset.json"
EVAL_SEEDS = 3


def rate(records, subset, envs):
    eps = subset["episodes"]
    cells = Counter((e["suite"], e["category"]) for e in eps)
    if any("success" in r or not r.get("timing_only") for r in records):
        raise SystemExit("a record holds 'success' or lacks timing_only: not a timing-only file; stop (no outcome may be read here)")
    rows = [r for r in records if r.get("benchmark") == "libero_plus"]            # lp_eval.LP
    wall = {}
    for r in rows:
        wall.setdefault((r["suite"], r["category"]), []).append(float(r["wall_s"]))
    missing = sorted(set(cells) - set(wall))
    if missing:
        raise SystemExit(f"{len(missing)} cells have no timing record, e.g. {missing[0]}; rerun the timing shard")
    n = sum(cells.values())
    mean_wall = sum(cells[c] * (sum(wall[c]) / len(wall[c])) for c in cells) / n
    r_w = 3600.0 * envs / mean_wall
    return {"cells": {f"{s}/{c}": {"n": cells[(s, c)], "wall_s": sum(wall[(s, c)]) / len(wall[(s, c)])}
                      for s, c in sorted(cells)},
            "episodes": n, "rollouts_per_checkpoint": n * EVAL_SEEDS, "weighted_mean_wall_s": mean_wall,
            "envs": envs, "R_weighted_per_pod_hour": r_w, "pod_hours_upper_bound": n * EVAL_SEEDS / r_w}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True)
    p.add_argument("--subset", default=str(DEFAULT_SUBSET))
    p.add_argument("--envs", type=int, required=True)
    p.add_argument("--pass-seconds", type=float, default=None, help="the lp-timing pass's wall seconds ([pass] line)")
    a = p.parse_args()
    recs = [json.loads(x) for x in Path(a.records).read_text().splitlines() if x.strip()]
    out = rate(recs, json.loads(Path(a.subset).read_text()), a.envs)
    for k, v in out["cells"].items():
        print(f"{k:<40} n {v['n']:>5}  wall_s {v['wall_s']:8.1f}")
    print(f"weighted mean wall_s {out['weighted_mean_wall_s']:.1f} at {out['envs']} envs -> R_weighted "
          f"{out['R_weighted_per_pod_hour']:.1f} episodes per pod-hour; {out['rollouts_per_checkpoint']} rollouts per "
          f"checkpoint = {out['pod_hours_upper_bound']:.1f} pod-hours (upper bound)")
    if a.pass_seconds:
        raw = 3600.0 * len(recs) / a.pass_seconds
        print(f"raw pass rate {raw:.1f} episodes per pod-hour ({len(recs)} records in {a.pass_seconds:.0f} s)")
        print(f"--lp-rate {math.floor(min(raw, out['R_weighted_per_pod_hour']))}")
    else:
        print(f"--lp-rate {math.floor(out['R_weighted_per_pod_hour'])} (no --pass-seconds given; R_weighted only)")


if __name__ == "__main__":
    main()
