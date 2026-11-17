"""Per-task tally of the stage 1 baseline clean-LIBERO evaluation (read-only, stdlib only).

Run: /opt/homebrew/bin/python3.11 -I tally_per_task.py [records_dir] [out_json]

Reads the rollout records (JSONL) and reports, per suite and per task: successes,
trials, success rate, and mean episode length (the "steps" field) over all episodes
and over successful episodes only. Counts only records with "benchmark": "libero".

Task id: episode_id = task * 1000 + trial (src/rollout.py line 136), so
task = episode_id // 1000 and trial = episode_id % 1000. The task name is the
"instruction" string the records carry; if a task's records carry more than one
instruction (or none), the task is flagged. Duplicate (suite, episode_id) keys are
counted as they appear and flagged, not removed. The script tallies whatever
records it is given and lists the distinct checkpoints and eval seeds seen, so a
mixed directory is visible rather than silently merged. success must be 0 or 1
(anything else raises). Rates and means are rounded to 4 and 2 decimals.
mean_steps_success is None when a task has no successes.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from tally_proxy_clean import load, SUITES  # noqa: E402

DEF_DIR = os.path.join(ROOT, "experiments", "2026-11-15-pretrained-start-baseline",
                       "results", "clean-eval", "records")
DEF_OUT = os.path.join(ROOT, "experiments", "2026-11-15-pretrained-start-baseline",
                       "results", "clean-eval", "per-task-tally.json")
NOTE = ("Per-task tally of the stage 1 baseline clean-LIBERO evaluation. task = episode_id // 1000 "
        "(src/rollout.py line 136: episode_id = task * 1000 + trial). Task name = the records' "
        "instruction field. Mean steps are over all episodes and over successful episodes only. "
        "Duplicates are counted, not removed.")


def _mean(xs):
    return round(sum(xs) / len(xs), 2) if xs else None


def tally(recs):
    clean = [r for r in recs if r.get("benchmark") == "libero"]
    tasks = {}
    for r in clean:
        if r["success"] not in (0, 1) or isinstance(r["success"], bool):
            raise ValueError(f"success must be 0 or 1, got {r['success']!r} "
                             f"for key {(r['suite'], r['episode_id'])}")
        key = (r["suite"], int(r["episode_id"]) // 1000)
        t = tasks.setdefault(key, {"succ": [], "all": [], "names": set(), "eps": {}})
        t["all"].append(r["steps"])
        if r["success"]:
            t["succ"].append(r["steps"])
        if r.get("instruction"):
            t["names"].add(r["instruction"])
        t["eps"][r["episode_id"]] = t["eps"].get(r["episode_id"], 0) + 1
    suites = {}
    for (suite, tid), t in sorted(tasks.items()):
        n, k = len(t["all"]), len(t["succ"])
        dups = [{"episode_id": e, "count": c} for e, c in sorted(t["eps"].items()) if c > 1]
        names = sorted(t["names"])
        s = suites.setdefault(suite, {"tasks": []})
        s["tasks"].append({"task_id": tid, "task_name": names[0] if len(names) == 1 else None,
                           "names_seen": names if len(names) != 1 else None,
                           "succ": k, "n": n, "rate": round(k / n, 4),
                           "mean_steps_all": _mean(t["all"]), "mean_steps_success": _mean(t["succ"]),
                           "duplicates": dups})
    for suite, s in suites.items():
        n = sum(x["n"] for x in s["tasks"])
        k = sum(x["succ"] for x in s["tasks"])
        s.update({"succ": k, "n": n, "rate": round(k / n, 4), "n_tasks": len(s["tasks"])})
    n_all = sum(s["n"] for s in suites.values())
    k_all = sum(s["succ"] for s in suites.values())
    return {"note": NOTE, "n_clean_records": len(clean), "n_all_records": len(recs),
            "n_non_clean_ignored": len(recs) - len(clean),
            "checkpoints": sorted({r["checkpoint"] for r in clean}),
            "eval_seeds": sorted({r["eval_seed"] for r in clean}),
            "total": {"succ": k_all, "n": n_all},
            "missing_suites": [s for s in SUITES if s not in suites],
            "suites": {s: suites[s] for s in SUITES if s in suites}}


def render(t):
    lines = [f"clean libero records: {t['n_clean_records']} of {t['n_all_records']} "
             f"({t['n_non_clean_ignored']} non-clean ignored); total {t['total']['succ']}/{t['total']['n']}",
             f"checkpoints: {', '.join(t['checkpoints'])}; eval seeds: {t['eval_seeds']}"]
    if t["missing_suites"]:
        lines.append("MISSING " + ",".join(t["missing_suites"]))
    for suite, s in t["suites"].items():
        lines.append(f"\n{suite}: {s['succ']}/{s['n']} ({100 * s['rate']:.1f}%), {s['n_tasks']} tasks")
        for x in s["tasks"]:
            name = x["task_name"] if x["task_name"] else f"(names seen: {x['names_seen']})"
            ms = "n/a" if x["mean_steps_success"] is None else f"{x['mean_steps_success']:.1f}"
            flag = f" | DUPLICATES {len(x['duplicates'])}" if x["duplicates"] else ""
            lines.append(f"  task {x['task_id']}: {x['succ']}/{x['n']} ({100 * x['rate']:.0f}%) "
                         f"steps all {x['mean_steps_all']:.1f} succ {ms} | {name}{flag}")
    return "\n".join(lines)


def main(argv):
    d = argv[1] if len(argv) > 1 else DEF_DIR
    o = argv[2] if len(argv) > 2 else DEF_OUT
    t = tally(load(d))
    print(render(t))
    os.makedirs(os.path.dirname(o), exist_ok=True)
    with open(o, "w") as f:
        json.dump(t, f, indent=2)
        f.write("\n")
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main(sys.argv)
