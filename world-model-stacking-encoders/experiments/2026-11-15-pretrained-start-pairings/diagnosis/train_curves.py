"""Training-curve summary for the stage 2 feature-path diagnosis (diagnosis/2026-11-25-feature-path.md).

Reads the pod logs committed under results/ (stage 2 pairings seed 0 runs and the stage 1 'none' seed 0 run), takes every
logged step record (lines '[train  Ns] {"step": ...}'; the last record per step wins, as read_history does), and prints
window means of loss/action, loss/vae, each source's loss and grad_norm. The pod logs carry a step record every step
for the first steps of a process and every 100 steps after that (the full per-step history.jsonl is on the cluster
volume only); the windows below therefore average the logged steps only, and the script prints how many.

Run (from the repo root, no third-party packages):
  /opt/homebrew/bin/python3.11 -I world-model-stacking-encoders/experiments/2026-11-15-pretrained-start-pairings/diagnosis/train_curves.py
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
BASE = EXP.parent / "2026-11-15-pretrained-start-baseline"
LOGS = {
    "none": BASE / "results/full-train/stacking-encoders-baseline-full-train-vzc7p.log",
    "dinov3": EXP / "results/train/dinov3-s0/stacking-encoders-pairings-train-dinov3-s0-2cflk.log",
    "c-radiov4": EXP / "results/train/c-radiov4-s0/stacking-encoders-pairings-train-c-radiov4-s0-7jltw.log",
    "siglip2": EXP / "results/train/siglip2-s0/stacking-encoders-pairings-train-siglip2-s0-xv2mk.log",
    "dinov3+c-radiov4": EXP / "results/train/dinov3-c-radiov4-s0/stacking-encoders-pairings-train-dinov3-c-radiov4-s0-dj992.log",
    "dinov3+siglip2": EXP / "results/train/dinov3-siglip2-s0/stacking-encoders-pairings-train-dinov3-siglip2-s0-g7gbf.log",
}
WINDOWS = [(0, 1), (100, 1100), (1000, 5000), (5000, 10000), (10000, 15000), (15000, 20000), (19700, 21696)]
REC = re.compile(r'^\[train\s+\d+s\] (\{"step".*\})\s*$')
FIRST = re.compile(r"^\[train\s+\d+s\] \[first batch\] loss at start (.*)$")


def parse(path):
    by, first, resumed, hist_tail = {}, None, [], []
    lines = path.read_text().splitlines()
    in_hist = False
    for line in lines:
        m = REC.match(line)
        if m:
            r = json.loads(m.group(1))
            by[r["step"]] = r
        m = FIRST.match(line)
        if m:
            first = m.group(1)
        if "resuming from" in line and line.startswith("[train"):
            resumed.append(line.strip())
        if line.startswith("=== BIGFILE history.jsonl"):
            in_hist = True
            continue
        if in_hist:
            if line.startswith("=== END history.jsonl"):
                in_hist = False
            else:
                hist_tail.append(json.loads(line))
    return [by[k] for k in sorted(by)], first, resumed, hist_tail


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    """Sample standard deviation of the logged single-step values (each a batch of 128); 0.0 for fewer than 2."""
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def main():
    out = {}
    for cfg, path in LOGS.items():
        recs, first, resumed, tail = parse(path)
        keys = [k for k in recs[-1] if k.startswith("loss/")] + ["loss", "grad_norm"]
        win = {}
        for lo, hi in WINDOWS:
            sel = [r for r in recs if lo <= r["step"] < hi]
            win[f"{lo}-{hi - 1}"] = {"n_logged": len(sel), **{k: round(mean([r[k] for r in sel if k in r]), 5) for k in keys},
                                     **{f"sd/{k}": round(sd([r[k] for r in sel if k in r]), 5) for k in ("loss/action", "loss/vae")}}
        out[cfg] = {"log": str(path.relative_to(EXP.parent.parent.parent)), "n_records": len(recs),
                    "last_step": recs[-1]["step"], "resumed": resumed, "first_batch": first, "windows": win,
                    "history_tail_steps": [r["step"] for r in tail],
                    "history_tail_mean": {k: round(mean([r[k] for r in tail if k in r]), 5) for k in keys}}
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
