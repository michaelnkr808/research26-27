"""Feature z-score statistics as the trained checkpoints carry them (config.json printed into each training pod log under
results/train/; rollout.Policy reads the same dict from model.pt, pretrained_train.py builds both from one cfg). Used by
diagnosis/2026-11-25-feature-path.md to see whether any channel's training std is so small that a small live-vs-cache
difference would become a large z-score at inference. Reads only; no third-party packages.

Run from the repo root:
  /opt/homebrew/bin/python3.11 -I world-model-stacking-encoders/experiments/2026-11-15-pretrained-start-pairings/diagnosis/feature_stats.py
"""

import json
import math
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


def config_of(path):
    lines = path.read_text().splitlines()
    s = next(i for i, l in enumerate(lines) if l.startswith("=== FILE config.json"))
    e = next(i for i in range(s + 1, len(lines)) if lines[i].startswith("=== END config.json"))
    return json.loads("\n".join(lines[s + 1:e]))


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]


def main():
    cfgs = {k: config_of(p) for k, p in LOGS.items()}
    out = {"same_action_and_state_norm_as_none": {k: (c["action_norm"] == cfgs["none"]["action_norm"]
                                                       and c["state_norm"] == cfgs["none"]["state_norm"])
                                                  for k, c in cfgs.items()},
           "n_episodes": {k: c["n_episodes"] for k, c in cfgs.items()},
           "n_frames": {k: c["n_frames"] for k, c in cfgs.items()},
           "sources_in_order": {k: c["sources"] for k, c in cfgs.items()},
           "per_source": {}}
    for k, c in cfgs.items():
        for src, (mu, sd) in c["feature_stats"].items():
            ratio = [abs(m) / s for m, s in zip(mu, sd)]
            rec = {"D": len(mu), "sd_min": min(sd), "sd_p01": q(sd, 0.01), "sd_median": q(sd, 0.5), "sd_max": max(sd),
                   "abs_mu_over_sd_max": max(ratio), "abs_mu_over_sd_p99": q(ratio, 0.99),
                   "channels_sd_below_0.01": sum(1 for s in sd if s < 0.01),
                   "channels_abs_mu_over_sd_above_10": sum(1 for r in ratio if r > 10),
                   # one float16 step (relative 2**-10) of the channel mean, in units of that channel's std: the size of a
                   # z-score change caused by rounding alone
                   "max_fp16_step_of_mu_in_sd": max(abs(m) * 2 ** -10 / s for m, s in zip(mu, sd)),
                   "lowest_sd_channels": [{"channel": i, "mu": mu[i], "sd": sd[i]}
                                          for i in sorted(range(len(sd)), key=lambda i: sd[i])[:3]]}
            key = f"{src} (run {k})"
            out["per_source"][key] = {a: (round(b, 6) if isinstance(b, float) else b) for a, b in rec.items()}
        # identical stats for a source across runs (same cache, same episodes)
    same = {}
    for src in ("dinov3", "c-radiov4", "siglip2"):
        vals = [json.dumps(c["feature_stats"][src]) for c in cfgs.values() if src in c["feature_stats"]]
        same[src] = {"runs": len(vals), "identical": len(set(vals)) == 1}
    out["stats_identical_across_runs"] = same
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
