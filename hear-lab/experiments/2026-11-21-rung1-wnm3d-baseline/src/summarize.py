#!/usr/bin/env python3
"""Summarize a rung 1 run: SR, SPL, OS, NE, TL overall and by the check 3 visible flag, with and without
the possibly mirrored scenes; Wilson 95% intervals; the pass verdict; time and cost per episode.

Inputs (all outside git):
  --run-dir      the run folder copied off the machine (gn0/log/*.json from GN0, rung1_episodes.jsonl
                 from the rung1 hook, episode_ids.txt)
  --order-file   results/episode_order.txt (frozen order; its sha256 must match config.yaml)
  --run-kind     timing or main; N is read from config.yaml (episodes.timing_k or episodes.n_main),
                 never from the command line, and the summary refuses while that key is pending
  --check3-csv   route-a check 3 check3_InteriorGS_test_unseen.csv (visible flag)
  --scenes-jsonl route-a check 3 scenes.jsonl (frame diagnostic for the mirrored scenes)
  --usd-per-hour the rented machine's hourly price (dph_total), for cost per episode
  --out          summary JSON to write (put it under results/ or the run folder, never in git)

Metric definitions follow GN0 gn0/analysis/results.py: per-episode values, NaN or inf read as 0,
averaged over episodes. SR, SPL and OS are reported in percent (as in the paper), NE and TL in metres.
Pass rule (PLAN.md): reproduced if the paper's 53.5 lies inside the 95% Wilson interval of our SR.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import yaml

from build_commands import PendingDecision, resolve_count

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parent / "config.yaml"

METRICS = ("success", "spl", "oracle_success", "distance_to_goal", "path_length")
FRAMES = ("identity", "mirror_x", "mirror_y", "mirror_xy")


class SummaryError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sanitize(value) -> float:
    """As GN0 gn0/analysis/results.py sanitize_metric: float, NaN/inf/unparseable -> 0.0."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) else 0.0


def wilson(successes: int, n: int, z: float) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, as fractions in [0, 1]."""
    if n <= 0:
        raise SummaryError("Wilson interval needs n > 0")
    if not 0 <= successes <= n:
        raise SummaryError(f"successes {successes} outside 0..{n}")
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def load_gn0_logs(log_dir: Path) -> dict[str, dict]:
    """Complete, consistent GN0 episode logs (the rule of gn0/evaluation/common.py load_finished_episode_ids)."""
    out = {}
    for p in sorted(Path(log_dir).glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(d, dict) or d.get("id") != p.stem:
            raise SummaryError(f"{p}: id does not match the file name")
        if d.get("status") != "complete" or d.get("episode_complete") is not True:
            continue
        out[p.stem] = d
    return out


def load_events(path: Path) -> dict[str, dict]:
    """Per-episode wall times from the rung1 hook; the last line wins if an episode appears twice."""
    out = {}
    if not Path(path).exists():
        return out
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue  # a half-written last line after a crash
        out[d["id"]] = d
    return out


def load_check3(csv_path: Path, visible_col: str, tol_col: str, tol: float) -> dict[str, dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        if r["status"] != "ok":
            raise SummaryError(f"check 3 row {r['row_index']} has status {r['status']}")
        if abs(float(r[tol_col]) - tol) > 1e-9:
            raise SummaryError(f"check 3 {tol_col} = {r[tol_col]}, expected {tol}")
        if r[visible_col] not in ("0", "1"):
            raise SummaryError(f"check 3 {visible_col} = {r[visible_col]!r}")
        out[f"{r['scene']}_{r['trajectory_id']}"] = {"scene": r["scene"], "visible": r[visible_col] == "1"}
    return out


def mirrored_scenes(scenes_jsonl: Path) -> set[str]:
    """Scenes where a mirrored frame puts the most splat centres inside the label boxes.

    Same rule as route-a summarize_check3.py scenes_summary: totals per frame, max() with identity first
    (ties go to identity), records with all-zero totals skipped. A scene counts if any record of it wins
    with a mirrored frame.
    """
    out = set()
    for line in Path(scenes_jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        d = r.get("frame_diag") or {}
        tot = {k: (d.get(k) or {}).get("total", 0) for k in FRAMES}
        if any(tot.values()) and max(tot, key=tot.get) != "identity":
            out.add(r["scene"])
    return out


def group_metrics(rows: list[dict], z: float) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    vals = {k: [sanitize(r.get(k, 0)) for r in rows] for k in METRICS}
    for k in ("success", "oracle_success"):
        bad = [v for v in vals[k] if v not in (0.0, 1.0)]
        if bad:
            raise SummaryError(f"{k} must be 0 or 1, got {bad[:3]}")
    s = int(sum(vals["success"]))
    o = int(sum(vals["oracle_success"]))
    sr_lo, sr_hi = wilson(s, n, z)
    os_lo, os_hi = wilson(o, n, z)
    return {
        "n": n,
        "successes": s,
        "sr": 100.0 * s / n,
        "sr_ci95": [100.0 * sr_lo, 100.0 * sr_hi],
        "spl": 100.0 * sum(vals["spl"]) / n,
        "os": 100.0 * o / n,
        "os_ci95": [100.0 * os_lo, 100.0 * os_hi],
        "ne": sum(vals["distance_to_goal"]) / n,
        "tl": sum(vals["path_length"]) / n,
    }


def timing(rows: list[dict], events: dict[str, dict], usd_per_hour: float | None) -> dict:
    walls = [events[r["id"]]["wall_s"] for r in rows if r["id"] in events]
    steady = [events[r["id"]]["wall_s"] for r in rows
              if r["id"] in events and not events[r["id"]].get("first_in_launch")]
    out = {"n_with_time": len(walls), "n_without_time": len(rows) - len(walls),
           "n_first_in_launch": len(walls) - len(steady)}
    if walls:
        out.update({"total_s": sum(walls), "mean_s": sum(walls) / len(walls),
                    "median_s": statistics.median(walls)})
    if steady:
        out.update({"mean_s_excl_first": sum(steady) / len(steady),
                    "median_s_excl_first": statistics.median(steady)})
    if usd_per_hour is not None and walls:
        out["usd_per_hour"] = usd_per_hour
        out["usd_per_episode"] = out["mean_s"] / 3600.0 * usd_per_hour
        if steady:
            out["usd_per_episode_excl_first"] = out["mean_s_excl_first"] / 3600.0 * usd_per_hour
    return out


def summarize(cfg: dict, run_dir: Path, order: list[str], n: int, check3: dict[str, dict],
              mirrored: set[str], usd_per_hour: float | None, run_kind: str = "main") -> dict:
    if not 1 <= n <= len(order):
        raise SummaryError(f"n {n} outside 1..{len(order)}")
    wanted = order[:n]
    wanted_set = set(wanted)
    logs = load_gn0_logs(Path(run_dir) / "gn0" / "log")
    extra = sorted(set(logs) - wanted_set)
    if extra:
        raise SummaryError(f"{len(extra)} finished episodes are not in the first {n} of the order, e.g. {extra[:3]}")
    ids_file = Path(run_dir) / "episode_ids.txt"
    if ids_file.exists():
        asked = [l.strip() for l in ids_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        if asked != wanted:
            raise SummaryError(f"{ids_file} is not the first {n} ids of the frozen order")
    missing_c3 = [i for i in wanted if i not in check3]
    if missing_c3:
        raise SummaryError(f"{len(missing_c3)} requested ids have no check 3 row, e.g. {missing_c3[:3]}")

    events = load_events(Path(run_dir) / "rung1_episodes.jsonl")
    z = float(cfg["pass_rule"]["wilson_z"])
    rows = []
    for i in wanted:
        if i in logs:
            r = dict(logs[i])
            r["visible"] = check3[i]["visible"]
            r["mirrored_scene"] = check3[i]["scene"] in mirrored
            rows.append(r)

    def pick(visible=None, drop_mirrored=False):
        return [r for r in rows
                if (visible is None or r["visible"] == visible) and not (drop_mirrored and r["mirrored_scene"])]

    groups = {}
    for name, vis in (("all", None), ("visible", True), ("not_visible", False)):
        groups[name] = group_metrics(pick(vis), z)
        groups[name + "_excl_mirrored"] = group_metrics(pick(vis, True), z)

    target = cfg["target"]
    complete = len(rows) == n
    overall = groups["all"]
    if run_kind != "main":
        verdict = "not applicable (timing run; the pass bar is on the main run)"
    elif not complete:
        verdict = "incomplete"
    else:
        lo, hi = overall["sr_ci95"]
        verdict = "reproduced" if lo <= float(target["sr"]) <= hi else "not reproduced"

    return {
        "run_kind": run_kind,
        "n_requested": n,
        "n_finished": len(rows),
        "complete": complete,
        "verdict": verdict,
        "pass_rule": cfg["pass_rule"]["statement"],
        "target": {k: target[k] for k in ("sr", "spl", "os", "ne", "tl")},
        "groups": groups,
        "stop_reasons": dict(Counter(str(r.get("stop", "unknown")) for r in rows)),
        "n_mirrored_scenes": len(mirrored),
        "timing": timing(rows, events, usd_per_hour),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--order-file", type=Path, required=True)
    ap.add_argument("--run-kind", choices=["timing", "main"], required=True)
    ap.add_argument("--check3-csv", type=Path, required=True)
    ap.add_argument("--scenes-jsonl", type=Path, required=True)
    ap.add_argument("--usd-per-hour", type=float)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    try:
        n = resolve_count(cfg, a.run_kind)
    except PendingDecision as exc:
        raise SummaryError(str(exc)) from exc

    for path, key in ((a.order_file, cfg["episodes"]["order_sha256"]),
                      (a.check3_csv, cfg["episodes"]["source_sha256"]),
                      (a.scenes_jsonl, cfg["breakdowns"]["mirrored_scenes"]["source_sha256"])):
        if sha256_file(path) != key:
            raise SummaryError(f"{path}: sha256 does not match config.yaml")
    order = [l for l in a.order_file.read_text(encoding="utf-8").splitlines() if l]
    vis = cfg["breakdowns"]["visible"]
    check3 = load_check3(a.check3_csv, vis["csv_column"], vis["tolerance_column"], float(vis["tolerance_m"]))
    mirrored = mirrored_scenes(a.scenes_jsonl)
    mcfg = cfg["breakdowns"]["mirrored_scenes"]
    if len(mirrored) != mcfg["n_scenes_expected"]:
        raise SummaryError(f"{len(mirrored)} mirrored scenes, config expects {mcfg['n_scenes_expected']}")
    n_mirr_eps = sum(1 for v in check3.values() if v["scene"] in mirrored)
    if n_mirr_eps != mcfg["n_test_unseen_episodes_expected"]:
        raise SummaryError(f"{n_mirr_eps} test_unseen episodes in mirrored scenes, config expects "
                           f"{mcfg['n_test_unseen_episodes_expected']}")

    out = summarize(cfg, a.run_dir, order, n, check3, mirrored, a.usd_per_hour, a.run_kind)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    g = out["groups"]["all"]
    if g["n"]:
        print(f"{out['n_finished']}/{out['n_requested']} episodes; SR {g['sr']:.1f} "
              f"(95% Wilson {g['sr_ci95'][0]:.1f} to {g['sr_ci95'][1]:.1f}); verdict: {out['verdict']}")
    else:
        print(f"0/{out['n_requested']} episodes finished; verdict: {out['verdict']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
