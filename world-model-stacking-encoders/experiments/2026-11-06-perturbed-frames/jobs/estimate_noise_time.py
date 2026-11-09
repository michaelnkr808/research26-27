#!/usr/bin/env python3
"""Estimate the time of the phase-B noise render (job-pfb-render, stage render-noise) from the committed draws and the phase-A pilot timings.

Usage: estimate_noise_time.py <draws.json> <pilot-noise-render-report.json> [--frames-per-demo 127.5]

Method (arithmetic on measured numbers; nothing is run):
  - draws.json: the file written by `pf_rerender.py draws` (config/draws.json). Every key ending in "/noise" is one test demo's drawn LIBERO-Plus noise
    episode; its name ends in _noise_<code>. The 100 test demos are expected.
  - Each test demo is assumed to have --frames-per-demo frames (12,748 / 100 = 127.5; the real count differs per demo, so the total is an estimate).
  - The pilot report (results/pf-main/phase-a/pfa-pilot/reports/pilot-noise-render-report.json) gives timing.noise_by_code[<code>].noise_seconds_per_frame,
    measured for the codes the pilot happened to draw. For a code the pilot did not draw, the mean of the pilot's measured per-frame costs of the codes in
    the same family is used (families by LIBERO-Plus code: 1-10 motion, 11-20 gaussian, 21-30 zoom, 31-40 fog, 41-50 glass), and a note is printed.
    A family with no measured code at all falls back to the largest measured cost (a pessimistic choice) and says so.
  - On top of the noise function: the base render and loop cost per frame of the pilot ((frame loop seconds - sum of the noise seconds) / frames), and the pilot's
    environment build and prepare cost per demo, times the number of demos.
  - Not included: the consistency check that every test render runs first (about 1 to 2 minutes, not timed on this machine class), the scene-file pre-pass, the publish copy.
Prints the estimate and a suggested NOISE_CAP (1.5 x the estimate + 1200 s, rounded up to 600 s, at least 3600 s). The cap is the lead's decision.
"""
import argparse
import json
import math
import re
import sys

FAMILIES = {1: "motion", 2: "gaussian", 3: "zoom", 4: "fog", 5: "glass"}


def family(code):
    return (code - 1) // 10 + 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("draws")
    ap.add_argument("pilot")
    ap.add_argument("--frames-per-demo", type=float, default=127.5)
    a = ap.parse_args(argv)

    draws = json.load(open(a.draws))["draws"]
    pilot = json.load(open(a.pilot))
    tm = pilot["timing"]
    measured = {int(c): v["noise_seconds_per_frame"] for c, v in tm["noise_by_code"].items() if v.get("noise_seconds_per_frame") is not None}
    noise_total = sum(v["noise_seconds"] for v in tm["noise_by_code"].values())
    base_per_frame = max(0.0, (tm["frame_loop_seconds"] - noise_total) / tm["frames"])
    pilot_demos = sum(len(f.get("per_demo", {})) for f in pilot["files"].values())
    env_per_demo = tm["env_build_and_prepare_seconds_total"] / pilot_demos if pilot_demos else 0.0

    codes = {}
    for k, d in draws.items():
        if not k.endswith("/noise"):
            continue
        m = re.search(r"_noise_(\d+)$", d["name"])
        if not m:
            sys.exit(f"cannot read a noise code from {d['name']!r}")
        codes[k] = int(m.group(1))
    if not codes:
        sys.exit("no noise draws in the draws file")

    fam_mean = {}
    for f in FAMILIES:
        vals = [v for c, v in measured.items() if family(c) == f]
        if vals:
            fam_mean[f] = sum(vals) / len(vals)
    worst = max(measured.values())

    per_code = {}
    for k, c in codes.items():
        per_code.setdefault(c, 0)
        per_code[c] += 1
    total_noise = 0.0
    rows, notes = [], []
    for c in sorted(per_code):
        n = per_code[c]
        if c in measured:
            s, src = measured[c], "measured in the pilot"
        elif family(c) in fam_mean:
            s, src = fam_mean[family(c)], f"NOT measured: mean of the pilot's {FAMILIES[family(c)]} codes {sorted(x for x in measured if family(x) == family(c))}"
            notes.append(f"code {c} ({n} demos): {src}")
        else:
            s, src = worst, f"NOT measured and no measured code in the {FAMILIES[family(c)]} family: the largest measured cost {worst:.3f} s/frame (pessimistic)"
            notes.append(f"code {c} ({n} demos): {src}")
        frames = n * a.frames_per_demo
        total_noise += frames * s
        rows.append((c, n, s, frames * s, src))

    n_demos = len(codes)
    frames_all = n_demos * a.frames_per_demo
    base = frames_all * base_per_frame
    env = n_demos * env_per_demo
    total = total_noise + base + env

    print(f"draws: {n_demos} noise draws (expected 100), {len(per_code)} distinct codes; frames per demo assumed {a.frames_per_demo} ({frames_all:.0f} frames in all)")
    print(f"pilot: {tm['frames']} frames, noise function cost measured for codes {sorted(measured)}; base render+loop {base_per_frame * 1000:.1f} ms per frame; env build+prepare {env_per_demo:.2f} s per demo")
    print("code  demos  s/frame   est. seconds  source")
    for c, n, s, sec, src in rows:
        print(f"{c:4d}  {n:5d}  {s:7.4f}  {sec:12.0f}  {src}")
    for n_ in notes:
        print("NOTE:", n_)
    print(f"noise function: {total_noise:.0f} s; base render and loop: {base:.0f} s; env build and prepare: {env:.0f} s")
    print(f"ESTIMATED noise render time: {total:.0f} s = {total / 3600:.2f} h (not counting the consistency check, about 1 to 2 min, nor the publish copy)")
    cap = max(3600, int(math.ceil((1.5 * total + 1200) / 600.0) * 600))
    print(f"suggested NOISE_CAP: {cap}   (1.5 x estimate + 1200 s, rounded up to 600 s, at least 3600; set it at the top of job-pfb-render.yaml's script)")
    print("GPU-idle note: the render holds the GPU at low use; if the estimate is above about 2 h, the pod's hours-long GPU average may fall under 40 percent (_standards/08), see PHASE-B.md section 6.")


if __name__ == "__main__":
    main()
