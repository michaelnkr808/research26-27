#!/usr/bin/env python3
"""Step 2 summary and verdict (PLAN.md, "Pass bar"). Runs anywhere (the Mac after the copy-off, or the machine).

Reads <run>/episode_records.jsonl (render_audio.py) and writes, into the same folder:
  summary.json         pooled tally and the verdict against the bar, per episode, per scene, visible and not
                       visible starts, the route A low wall-share scenes if any are in the set, timing and cost
  episode_tallies.csv  one row per episode: ids, counts, shares, timing (no coordinates, labels or boxes)
Both are committed. The verdict is computed only from counted steps; occluded steps are reported with no bar. If
any episode the run was asked for (run_config.json episode_ids) has no record, the verdict is INCOMPLETE.

  python3.11 src/summarize.py --run-dir results/<run> [--usd-per-hour X]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
import step2lib as s2  # noqa: E402

TALLY_KEYS = ("n_steps", "n_los", "n_occluded", "n_counted", "n_correct", "n_occluded_with_side",
              "n_occluded_correct", "n_counted_no_cue")


def add(a: dict, b: dict) -> dict:
    out = {k: a.get(k, 0) + b.get(k, 0) for k in TALLY_KEYS}
    out["share_correct"] = out["n_correct"] / out["n_counted"] if out["n_counted"] else None
    out["occluded_share_correct"] = (out["n_occluded_correct"] / out["n_occluded_with_side"]
                                     if out["n_occluded_with_side"] else None)
    out["wilson95_share_correct"] = s2.wilson(out["n_correct"], out["n_counted"])
    return out


def pool(records) -> dict:
    t = {k: 0 for k in TALLY_KEYS}
    for r in records:
        t = add(t, r)
    return add(t, {})


def summarize(cfg: dict, records: list, usd_per_hour=None, wall_s=None, expected_ids=None) -> dict:
    """expected_ids: the episode ids the run was asked for (run_config.json). If any of them has no record (a
    failed episode), the verdict is INCOMPLETE: a pass or fail is given only for the whole planned batch."""
    dc = cfg["direction_check"]
    pooled = pool(records)
    got = [r["episode_id"] for r in records]
    missing = sorted(set(expected_ids) - set(got)) if expected_ids is not None else []
    extra = sorted(set(got) - set(expected_ids)) if expected_ids is not None else []
    by_scene = defaultdict(list)
    for r in records:
        by_scene[r["scene"]].append(r)
    low = set(dc["low_wall_share_scenes"])
    out = {
        "bar": {"share_correct_at_least": dc["bar_share_correct"], "min_counted_steps_pooled": dc["min_counted_steps_pooled"],
                "min_lateral_deg": dc["min_lateral_deg"], "counted": "line of sight and |lateral angle| >= min_lateral_deg"},
        "n_episodes": len(records),
        "pooled": pooled,
        "verdict": ("INCOMPLETE" if missing or extra or expected_ids is None
                    else s2.verdict(pooled, float(dc["bar_share_correct"]), int(dc["min_counted_steps_pooled"]))),
        "episodes_expected": None if expected_ids is None else len(expected_ids),
        "episodes_missing": missing,
        "episodes_unexpected": extra,
        "visible_starts": pool([r for r in records if r["visible_start"]]),
        "not_visible_starts": pool([r for r in records if not r["visible_start"]]),
        "per_scene": {s: dict(pool(v), n_episodes=len(v)) for s, v in sorted(by_scene.items())},
        "low_wall_share_scenes_in_set": sorted(low & set(by_scene)),
        "low_wall_share_scenes": pool([r for r in records if r["scene"] in low]) if low & set(by_scene) else None,
        "per_episode": {r["episode_id"]: {k: r[k] for k in (*TALLY_KEYS, "share_correct", "occluded_share_correct")}
                        for r in records},
        "episodes_with_no_counted_step": sorted(r["episode_id"] for r in records if r["n_counted"] == 0),
        "one_channel": "interaural difference zero by construction: both ears of the one-channel clip are the same "
                       "signal, the mean of the two-ear render's left and right",
        "front_back": "not tested",
    }
    tot = [r["timing_s"]["episode_total"] for r in records]
    steps = sum(r["n_steps"] for r in records)
    out["timing"] = {
        "episode_total_s_sum": sum(tot), "episode_total_s_mean": sum(tot) / len(tot) if tot else None,
        "simulate_s_per_step_mean": (sum(r["timing_s"]["simulate_total"] for r in records) / steps) if steps else None,
        "scene_load_s_mean": (sum(r["timing_s"]["scene_load_mesh"] + r["timing_s"]["scene_load_context"] for r in records)
                              / len(records)) if records else None,
        "pose_export_render_s_mean": (sum(r["timing_s"]["pose_export_render"] for r in records) / len(records)) if records else None,
        "steps_total": steps,
        "wall_s_all_episodes": wall_s,
    }
    if usd_per_hour is not None and wall_s is not None and records:
        out["timing"]["usd_per_hour"] = usd_per_hour
        out["timing"]["usd_per_episode_audio_wall"] = usd_per_hour * wall_s / 3600.0 / len(records)
    return out


def write_csv(records: list, path: Path) -> None:
    cols = ["episode_id", "scene", "order_position", "visible_start", "n_frames", *TALLY_KEYS, "share_correct",
            "occluded_share_correct", "episode_total_s", "simulate_total_s", "scene_load_s", "pose_export_render_s"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in records:
            t = r["timing_s"]
            w.writerow([r["episode_id"], r["scene"], r["order_position"], int(r["visible_start"]), r["n_frames"],
                        *[r[k] for k in TALLY_KEYS],
                        "" if r["share_correct"] is None else f"{r['share_correct']:.4f}",
                        "" if r["occluded_share_correct"] is None else f"{r['occluded_share_correct']:.4f}",
                        f"{t['episode_total']:.3f}", f"{t['simulate_total']:.3f}",
                        f"{t['scene_load_mesh'] + t['scene_load_context']:.3f}", f"{t['pose_export_render']:.3f}"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=EXP / "config.yaml")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--usd-per-hour", type=float, default=None)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    recs = [json.loads(l) for l in (a.run_dir / "episode_records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    timing = json.loads((a.run_dir / "timing.json").read_text(encoding="utf-8")) if (a.run_dir / "timing.json").exists() else {}
    rc_path = a.run_dir / "run_config.json"
    expected = json.loads(rc_path.read_text(encoding="utf-8"))["episode_ids"] if rc_path.exists() else None
    out = summarize(cfg, recs, a.usd_per_hour, timing.get("wall_s_all_episodes"), expected)
    out["failed_episodes"] = timing.get("failed", {})
    (a.run_dir / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_csv(recs, a.run_dir / "episode_tallies.csv")
    p = out["pooled"]
    if out["verdict"] == "INCOMPLETE":
        print(f"verdict INCOMPLETE: missing {out['episodes_missing']}, unexpected {out['episodes_unexpected']}"
              + ("" if expected is not None else " (no run_config.json, so the planned episode set is unknown)"))
    print(f"verdict {out['verdict']}: {p['n_correct']} of {p['n_counted']} counted steps have the correct sign "
          f"(bar {cfg['direction_check']['bar_share_correct']}); {p['n_occluded']} occluded steps reported, no bar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
