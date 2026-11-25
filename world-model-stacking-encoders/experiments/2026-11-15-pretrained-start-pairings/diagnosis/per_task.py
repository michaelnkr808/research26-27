"""Per-task clean LIBERO success of stage 1 'none' seed 0 and the three stage 2 single-encoder seed 0 checkpoints, from the
rollout records committed under results/ (2,000 rollouts per checkpoint: 40 tasks x 50 trials, eval seed 41). Used by
diagnosis/2026-11-25-feature-path.md. Reads only; no third-party packages.

Run from the repo root:
  /opt/homebrew/bin/python3.11 -I world-model-stacking-encoders/experiments/2026-11-15-pretrained-start-pairings/diagnosis/per_task.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
BASE = EXP.parent / "2026-11-15-pretrained-start-baseline"
RECORDS = {
    "none": sorted((BASE / "results/clean-eval/records").glob("clean-none-s0-shard*of4-try1.jsonl")),
    "dinov3": sorted((EXP / "results/clean/dinov3-s0/records").glob("clean-shard*of4-try1.jsonl")),
    "c-radiov4": sorted((EXP / "results/clean/c-radiov4-s0/records").glob("clean-shard*of4-try1.jsonl")),
    "siglip2": sorted((EXP / "results/clean/siglip2-s0/records").glob("clean-shard*of4-try1.jsonl")),
}


def main():
    succ = {c: defaultdict(int) for c in RECORDS}
    n = {c: defaultdict(int) for c in RECORDS}
    steps_ok = {c: [] for c in RECORDS}
    instr = {}
    for c, files in RECORDS.items():
        assert len(files) == 4, (c, files)
        seen = set()
        for f in files:
            for line in f.read_text().splitlines():
                r = json.loads(line)
                k = (r["suite"], int(r["episode_id"]) // 1000)
                key = (r["suite"], int(r["episode_id"]), int(r["eval_seed"]))
                assert key not in seen, (c, key)
                seen.add(key)
                n[c][k] += 1
                succ[c][k] += int(r["success"])
                instr[k] = r["instruction"]
                if r["success"]:
                    steps_ok[c].append(int(r["steps"]))
        assert len(seen) == 2000, (c, len(seen))
    tasks = sorted(n["none"])
    rows = []
    for k in tasks:
        rows.append({"suite": k[0], "task": k[1], "instruction": instr[k], **{c: f"{succ[c][k]}/{n[c][k]}" for c in RECORDS}})
    summary = {}
    for c in RECORDS:
        rates = [succ[c][k] / n[c][k] for k in tasks]
        diff = [succ[c][k] / n[c][k] - succ["none"][k] / n["none"][k] for k in tasks]
        summary[c] = {"avg": round(sum(rates) / len(rates), 4),
                      "tasks_at_0": sum(1 for r in rates if r == 0), "tasks_below_50pct": sum(1 for r in rates if r < 0.5),
                      "tasks_at_100": sum(1 for r in rates if r == 1),
                      "tasks_worse_than_none_by_20pts_or_more": sum(1 for d in diff if d <= -0.2),
                      "tasks_better_than_none_by_20pts_or_more": sum(1 for d in diff if d >= 0.2),
                      "mean_steps_of_successes": round(sum(steps_ok[c]) / len(steps_ok[c]), 1)}
    json.dump({"summary": summary, "per_task": rows}, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
