"""Read-only timing tables from the pilot-rollout2 result files (RUN.md step 9; LOG 2026-11-05 (c) option A).

    python pilot_timings.py --subset pilot-subset.json --results a.jsonl b.jsonl --logs a.log b.log \
        [--dip POD_INDEX:START_S:END_S ...]

Per result file (one pod): wall_s, steps and wall_s/steps by suite x category in JOB order (job order = the subset's
episodes in file order, each under eval seeds 41, 42, 43, then the clean trials; rollout.make_jobs). Caveats: `steps`
includes the 30 wait steps and `wall_s` includes building the env, so wall_s/steps is a rough number. The records carry
no timestamps. The file is written as episodes finish, so line i is the i-th completion; with the pod's progress lines
(done count per ~60 s on the rollout clock) the end time of line i is interpolated, to within about 60 s (job order is FIFO but estimated starts can invert), and start =
end - wall_s. `--dip POD:START:END` (rollout-clock seconds) lists the episodes whose estimated run overlapped that window.
Writes nothing; prints tables.
"""
import argparse
import json
import re
from collections import OrderedDict


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def progress(path):
    pts = [(0.0, 0)]
    for l in open(path):
        m = re.search(r"\[rollout\s+(\d+)s\] progress: (\d+) done", l)
        if m:
            pts.append((float(m.group(1)), int(m.group(2))))
        m = re.search(r"\[rollout\s+(\d+)s\] finished: (\d+) records", l)
        if m:
            pts.append((float(m.group(1)), int(m.group(2))))
    return pts


def end_time(pts, k):
    """Rollout-clock time at which the done count reached k (1-based), linear between progress lines."""
    for (t0, d0), (t1, d1) in zip(pts, pts[1:]):
        if d0 < k <= d1:
            return t0 + (t1 - t0) * (k - d0) / (d1 - d0)
    return pts[-1][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--logs", nargs="*", default=[])
    ap.add_argument("--dip", nargs="*", default=[])
    a = ap.parse_args()
    sub = json.load(open(a.subset))
    jobidx = {}
    for i, e in enumerate(sub["episodes"]):
        for j, s in enumerate((41, 42, 43)):
            jobidx[("libero_plus", e["suite"], int(e["id"]), s)] = 3 * i + j
    n_lp = 3 * len(sub["episodes"])
    dips = {}
    for d in a.dip:
        p, s, e = d.split(":")
        dips[int(p)] = (float(s), float(e))
    for pi, path in enumerate(a.results):
        recs = load(path)
        pts = progress(a.logs[pi]) if pi < len(a.logs) else None
        clean_seen = 0
        rows = []
        for line, r in enumerate(recs):
            key = (r["benchmark"], r["suite"], r["episode_id"], r["eval_seed"])
            if r["benchmark"] == "libero_plus":
                ji = jobidx[key]
            else:
                ji = None
            end = end_time(pts, line + 1) if pts else None
            rows.append({"r": r, "job": ji, "line": line, "end": end,
                         "start": end - r["wall_s"] if end is not None else None})
        # order of cells in job order (clean last)
        cells = OrderedDict()
        for x in sorted((x for x in rows if x["job"] is not None), key=lambda x: x["job"]):
            cells.setdefault((x["r"]["suite"], x["r"]["category"]), []).append(x)
        cells[("clean", "")] = [x for x in rows if x["job"] is None]
        print(f"\n=== {path}  ({len(recs)} records)")
        print("cell (suite / category)                    jobs        n   mean_wall_s  mean_steps  mean_wall/steps(s)  succ")
        for (suite, cat), xs in cells.items():
            if suite == "clean":
                jr = "clean"
            else:
                jr = f"{min(x['job'] for x in xs)}-{max(x['job'] for x in xs)}"
            w = sum(x["r"]["wall_s"] for x in xs) / len(xs)
            st = sum(x["r"]["steps"] for x in xs) / len(xs)
            ratio = sum(x["r"]["wall_s"] for x in xs) / sum(x["r"]["steps"] for x in xs)
            sc = sum(x["r"]["success"] for x in xs)
            print(f"{suite + ' / ' + cat:42s} {jr:9s} {len(xs):4d} {w:12.1f} {st:11.1f} {ratio:19.4f} {sc:5d}")
        if pi in dips:
            ds, de = dips[pi]
            print(f"\n-- episodes whose estimated run [start, end] overlaps rollout-clock {ds:.0f}-{de:.0f} s (pod {pi})")
            print("line job  suite / category                         eval_seed  start_s   end_s  wall_s  steps  wall/steps")
            for x in rows:
                if x["start"] is None or x["end"] < ds or x["start"] > de:
                    continue
                r = x["r"]
                print(f"{x['line']:4d} {str(x['job']):4s} {r['suite'] + ' / ' + r['category']:40s} {r['eval_seed']:4d} "
                      f"{x['start']:9.0f} {x['end']:7.0f} {r['wall_s']:7.1f} {r['steps']:6d} {r['wall_s'] / r['steps']:9.4f}")
    if len(a.results) == 2:
        sensor_noise_split(sub, a.results)


SLOW_FACTOR = 1.5  # the author's own cutoff for "slow", not a rule from PLAN or RUN.md


def sensor_noise_split(subset, result_paths):
    """Per Sensor Noise pilot episode: wall_s per eval seed on each pod, pod 1 / pod 0 ratio, and slow / normal.
    Slow = mean wall_s over both pods and 3 seeds > SLOW_FACTOR x the mean wall_s of the same suite's non-Sensor-Noise
    LIBERO-Plus records (both pods). Needs exactly two result files (pod 0, pod 1)."""
    data = []
    for p in result_paths:
        d, other = {}, {}
        for r in load(p):
            if r["benchmark"] != "libero_plus":
                continue
            if r["category"] == "Sensor Noise":
                d.setdefault((r["suite"], r["episode_id"]), {})[r["eval_seed"]] = r["wall_s"]
            else:
                other.setdefault(r["suite"], []).append(r["wall_s"])
        data.append((d, other))
    base = {s: sum(data[0][1][s] + data[1][1][s]) / len(data[0][1][s] + data[1][1][s]) for s in data[0][1]}
    print(f"\n=== Sensor Noise pilot episodes, per episode (pod 0 = first file, pod 1 = second)")
    print(f"'slow' = both-pod mean wall_s > {SLOW_FACTOR} x the suite's non-Sensor-Noise mean wall_s: "
          f"THE AUTHOR'S OWN CUTOFF, not a rule. Suite bases (s): "
          + ", ".join(f"{s} {b:.1f}" for s, b in sorted(base.items())))
    print("suite / episode           pod0 wall_s 41/42/43      pod1 wall_s 41/42/43      pod1/pod0   class")
    slow, normal, rs, rn = [], [], [], []
    for e in subset["episodes"]:
        if e["category"] != "Sensor Noise":
            continue
        k = (e["suite"], int(e["id"]))
        a, b = data[0][0][k], data[1][0][k]
        m0, m1 = sum(a.values()) / 3, sum(b.values()) / 3
        ratio = m1 / m0
        is_slow = (m0 + m1) / 2 > SLOW_FACTOR * base[e["suite"]]
        (slow if is_slow else normal).append(k)
        (rs if is_slow else rn).append(ratio)
        print(f"{k[0] + ' ' + str(k[1]):24s} " + "/".join(f"{a[s]:6.1f}" for s in (41, 42, 43)).ljust(25)
              + "  " + "/".join(f"{b[s]:6.1f}" for s in (41, 42, 43)).ljust(25) + f"  {ratio:6.2f}   "
              + ("slow" if is_slow else "normal"))
    print(f"slow: {len(slow)} of {len(slow) + len(normal)}: {slow}")
    print(f"normal: {len(normal)}: {normal}")
    print(f"pod1/pod0 ratio range: slow {min(rs):.2f}-{max(rs):.2f}, normal {min(rn):.2f}-{max(rn):.2f}")


if __name__ == "__main__":
    main()
