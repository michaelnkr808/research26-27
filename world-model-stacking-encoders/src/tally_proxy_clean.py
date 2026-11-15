"""Tally the from-scratch proxy's clean-LIBERO rollouts (read-only, stdlib only).

Run: /opt/homebrew/bin/python3.11 -I tally_proxy_clean.py [rollouts_dir] [out_json]

Counts only records with "benchmark": "libero" (libero_plus is ignored).
Groups by (config, train_seed, checkpoint) then eval_seed. Reports rollouts and
successes per suite, the four-suite average (mean of the per-suite rates), and
flags duplicate (suite, episode_id, eval_seed, checkpoint) keys and uneven
per-suite counts. Duplicates are counted as they appear (not removed), so a
duplicate inflates n and the rate; it is flagged, not corrected.

The four-suite average is None unless all four suites are present. success
must be 0 or 1 (anything else raises). Rates are rounded to 4 decimals.

What these records are: subset_id "clean1" comes from src/rollout.py line 135,
f"clean{clean_per_task}", i.e. `--clean 1`: one trial per base task (init state
0 per the rollout docstring), 10 per suite, 40 per checkpoint, eval seed 41.
They are NOT comparable with a full 50-per-task clean-LIBERO number. One
rollout = 2.5 points of four-suite average, and the s0/s1 gap is one rollout.
Checkpoints: the path is all we know. LOG.md of 2026-10-27-proxy-calibration
(lines 429, 501-505) shows them as Wave-1 training outputs (s0-a40 is the A40
twin of seed 0), but whether they are the full-length runs is not verified.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEF_DIR = os.path.join(ROOT, "experiments", "2026-10-27-proxy-calibration",
                       "results", "full-benchmark", "rollouts")
DEF_OUT = os.path.join(ROOT, "experiments", "2026-11-15-pretrained-start-baseline",
                       "results", "proxy-clean-tally.json")
NOTE = ("subset_id clean1 (src/rollout.py line 135, f\"clean{clean_per_task}\", --clean 1): one trial "
        "per base task (init state 0), 10 per suite, 40 per checkpoint, eval seed 41. Not comparable with "
        "a full 50-per-task clean-LIBERO number. One rollout = 2.5 points of four-suite average; the s0/s1 "
        "gap is one rollout. Checkpoints: the path is all we know (proxy-calibration LOG.md lines 429, "
        "501-505 show Wave-1 training outputs; full-length not verified). Duplicates are counted, not removed.")
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def load(rollouts_dir):
    recs = []
    for p in sorted(glob.glob(os.path.join(rollouts_dir, "**", "*.jsonl"), recursive=True)):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs


def tally(recs):
    clean = [r for r in recs if r.get("benchmark") == "libero"]
    groups = {}
    for r in clean:
        ck = (r["config"], r["train_seed"], r["checkpoint"])
        groups.setdefault(ck, {}).setdefault(r["eval_seed"], []).append(r)
    out = []
    for (cfg, ts, ckpt), by_seed in sorted(groups.items()):
        seeds = []
        for es, rs in sorted(by_seed.items()):
            per = {}
            seen = {}
            for r in rs:
                s = per.setdefault(r["suite"], {"n": 0, "succ": 0})
                s["n"] += 1
                if r["success"] not in (0, 1) or isinstance(r["success"], bool):
                    raise ValueError(f"success must be 0 or 1, got {r['success']!r} "
                                     f"for key {(r['suite'], r['episode_id'], es, ckpt)}")
                s["succ"] += int(r["success"])
                k = (r["suite"], r["episode_id"])
                seen[k] = seen.get(k, 0) + 1
            dups = [{"suite": k[0], "episode_id": k[1], "count": c}
                    for k, c in sorted(seen.items(), key=lambda kv: str(kv[0])) if c > 1]
            for s in per.values():
                s["rate"] = round(s["succ"] / s["n"], 4)
            rates = [per[s]["succ"] / per[s]["n"] for s in SUITES if s in per]
            avg = round(sum(rates) / 4, 4) if len(rates) == 4 else None
            counts = {s: per[s]["n"] for s in per}
            seeds.append({"eval_seed": es, "per_suite": per, "four_suite_avg": avg,
                          "n_total": sum(counts.values()),
                          "duplicates": dups,
                          "uneven_counts": len(set(counts.values())) > 1,
                          "missing_suites": [s for s in SUITES if s not in per]})
        out.append({"config": cfg, "train_seed": ts, "checkpoint": ckpt, "eval_seeds": seeds})
    return {"note": NOTE, "n_clean_records": len(clean), "n_all_records": len(recs),
            "n_non_clean_ignored": len(recs) - len(clean), "checkpoints": out}


def render(t):
    lines = [f"clean libero records: {t['n_clean_records']} of {t['n_all_records']} "
             f"({t['n_non_clean_ignored']} non-clean ignored)"]
    for c in t["checkpoints"]:
        lines.append(f"\n{c['config']} train_seed={c['train_seed']} {c['checkpoint']}")
        for e in c["eval_seeds"]:
            cells = []
            for s in SUITES:
                p = e["per_suite"].get(s)
                cells.append(f"{s} {p['succ']}/{p['n']}" if p else f"{s} -")
            avg = "n/a" if e["four_suite_avg"] is None else f"{100 * e['four_suite_avg']:.1f}%"
            flags = []
            if e["duplicates"]:
                flags.append(f"DUPLICATES {len(e['duplicates'])}")
            if e["uneven_counts"]:
                flags.append("UNEVEN COUNTS")
            if e["missing_suites"]:
                flags.append("MISSING " + ",".join(e["missing_suites"]))
            lines.append(f"  eval_seed={e['eval_seed']}: " + "; ".join(cells)
                         + f" | 4-suite avg {avg}" + (" | " + " ".join(flags) if flags else ""))
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
