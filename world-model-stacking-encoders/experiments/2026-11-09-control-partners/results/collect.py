"""Collect the control-partners stage 1 results and apply the PLAN's gates and decision rule.

Run (from anywhere):  /opt/homebrew/bin/python3.11 -I collect.py > collect-report.txt

What it does
  1. Checks the saved pod log (control-partners-20261110.log) against its recorded sha256.
  2. Extracts the ten result JSONs printed in the log (after lines "=== result json <path>")
     into this folder under their original file names. Each must parse as JSON.
  3. Evaluates the four gates in PLAN.md "Decision rule". If any gate fails it prints the
     failure and stops: no decision-rule verdict is printed (PLAN: "debug and amend, do not
     interpret").
  4. If all gates pass: prints the numbers the rule uses, applies every clause of the rule
     word for word, the stage 2 trigger, the PLAN's predictions against the results, the
     alphas side by side, run times against the PLAN's estimate, and the log warnings.

Recorded real values (tally gains and target interferences) are read from the repo JSONs the
PLAN names, not retyped. Thresholds and predicted ranges are the PLAN's own numbers, typed
below from PLAN.md (they are the pre-registered constants, not results).
No timestamps or other run-dependent text is printed, so the output is reproducible.
"""
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))  # world-model-stacking-encoders/
LOGNAME = "control-partners-20261110.log"
LOG_SHA256 = "bee6c0d9d06b5cce18b35574f21b0610bde0421ac02c060a2c7aa7676df56499"
LOG_BYTES = 251089
TALLY_DIR = os.path.join(ROOT, "experiments", "2026-10-19-feasibility-tally", "results")
TARGET_DIR = os.path.join(ROOT, "experiments", "2026-10-21-target-diagnostics", "results")
# Job times given by the lead (kubectl, not in the pod log); the finish time is also the last log line.
JOB_START = "2026-11-10T09:26:18Z"
JOB_FINISH = "2026-11-10T15:37:23Z"
JOB_WALL_S = (15 * 3600 + 37 * 60 + 23) - (9 * 3600 + 26 * 60 + 18)

EXPECTED_JSONS = [
    "input-libero_goal-dinov3-c-radiov4-rerun.json",
    "input-libero_goal-dinov3-dup-dinov3.json",
    "input-libero_goal-dinov3-noise-4608-s0.json",
    "input-libero_goal-dinov3-randproj-dinov3-4608-s0.json",
    "input-libero_goal-dinov3-shuffle-dinov3-s0.json",
    "target-libero_goal-dinov3-dup-dinov3.json",
    "target-libero_goal-dinov3-noise-4608-0.3-s0.json",
    "target-libero_goal-dinov3-noise-4608-s0.json",
    "target-libero_goal-dinov3-randproj-dinov3-4608-s0.json",
    "target-libero_goal-dinov3-shuffle-dinov3-s0.json",
]
IN = {  # control label -> (spec, input json)
    "dup:dinov3": EXPECTED_JSONS[1],
    "noise:4608": EXPECTED_JSONS[2],
    "randproj:dinov3:4608": EXPECTED_JSONS[3],
    "shuffle:dinov3": EXPECTED_JSONS[4],
}
TG = {  # control label -> target json
    "dup:dinov3": EXPECTED_JSONS[5],
    "noise:4608:0.3": EXPECTED_JSONS[6],
    "noise:4608": EXPECTED_JSONS[7],
    "randproj:dinov3:4608": EXPECTED_JSONS[8],
    "shuffle:dinov3": EXPECTED_JSONS[9],
}
REAL = [  # letter, name, tally file stem, target file stem
    ("G", "VGGT-Omega", "vggt-omega"),
    ("R", "C-RADIOv4", "c-radiov4"),
    ("L", "SigLIP 2", "siglip2"),
    ("S", "SAM 3", "sam3"),
]
SCALINGS = ["a_prime", "b", "a_dprime"]
SCALING_NAME = {"a_prime": "a'", "b": "b", "a_dprime": "a''"}
KS = ["4", "16", "32"]
RANKS = ["64", "16", "256"]

# ---- PLAN constants (Decision rule and Predictions, PLAN.md) ----
M_IN = 0.005
M_T = 0.003
TARGET_BIG = 0.0100      # two thirds of R's 0.0151
TARGET_SMALL = 0.0050    # one third of R's
B_HI = 0.0177            # SAM 3's 0.0207 minus 0.003
B_LO = 0.0104            # half of SAM 3's 0.0207
TRIGGER = 0.002          # stage 2 trigger distance

out = []


def p(s=""):
    out.append(s)


def fx(x, d=4):
    return f"{x:+.{d}f}"


def cis(c, d=4):
    return f"[{c[0]:+.{d}f}, {c[1]:+.{d}f}]"


# =========================================================== 1. log check and extraction
logpath = os.path.join(HERE, LOGNAME)
raw = open(logpath, "rb").read()
sha = hashlib.sha256(raw).hexdigest()
p("CONTROL PARTNERS, STAGE 1: collection report")
p("=" * 60)
p("Input log: results/" + LOGNAME)
p(f"  bytes {len(raw)} (expected {LOG_BYTES}): {'ok' if len(raw) == LOG_BYTES else 'MISMATCH'}")
p(f"  sha256 {sha}")
p(f"  expected {LOG_SHA256}: {'ok' if sha == LOG_SHA256 else 'MISMATCH'}")
if len(raw) != LOG_BYTES or sha != LOG_SHA256:
    p("STOP: log does not match the recorded size/hash.")
    print("\n".join(l.rstrip() for l in out))
    sys.exit(1)

lines = raw.decode("utf-8").split("\n")
heads = [i for i, l in enumerate(lines) if l.startswith("=== result json ")]
J = {}
p()
p("Extracted result JSONs (written under their original file names, each parsed with json):")
for n, i in enumerate(heads):
    j = heads[n + 1] if n + 1 < len(heads) else len(lines)
    body = "\n".join(lines[i + 1:j]).lstrip()
    dec = json.JSONDecoder()
    obj, end = dec.raw_decode(body)
    text = body[:end]
    assert json.loads(text) == obj
    name = lines[i].split("/")[-1].strip()
    with open(os.path.join(HERE, name), "w") as f:
        f.write(text + "\n")
    J[name] = obj
    p(f"  {name}: {len(text)} chars, parsed ok")
missing = [n for n in EXPECTED_JSONS if n not in J]
extra = [n for n in J if n not in EXPECTED_JSONS]
if missing or extra or len(heads) != 10:
    p(f"STOP: expected the ten names; missing {missing}, extra {extra}")
    print("\n".join(l.rstrip() for l in out))
    sys.exit(1)

# ---- recorded real values from the repo JSONs the PLAN names ----
tally = {}
tgt_real = {}
for letter, name, stem in REAL:
    tally[letter] = json.load(open(os.path.join(TALLY_DIR, f"tally-libero_goal-dinov3-{stem}.json")))
    tgt_real[letter] = json.load(open(os.path.join(TARGET_DIR, f"target-libero_goal-dinov3-{stem}.json")))


def real_gain(letter):
    stem = [s for l, n, s in REAL if l == letter][0]
    pr = tally[letter]["pairs"][f"dinov3|{stem}|global"]
    return pr["gain_B_given_A"], pr["gain_ci95"]


def real_tgt(letter, sc, r="64"):
    v = tgt_real[letter]["types"]["global"]["k_mean"]["t2"][sc][r]
    return v["interference"], v["ci95"]


def real_share(letter):
    return tgt_real[letter]["types"]["global"]["k_mean"]["t0_loss_share_B"]["a_prime"]


# ---- accessors on the new JSONs ----
def in_gain(spec):
    # spec is the label ("noise:4608" ...); JSON key "dinov3|<spec>|global"
    d = J[IN[spec]]["pairs"][f"dinov3|{spec}|global"]
    return d["gain_B_given_A"], d["gain_ci95"]


def in_probe(spec, key):
    return J[IN[spec]]["probes"][key]


def combo_key(d, spec):
    ks = [k for k in d["probes"] if "+" in k and k.endswith("|global")]
    assert len(ks) == 1, ks
    return ks[0]


def tgt_k(spec, sc, r="64"):
    v = J[TG[spec]]["types"]["global"]["k_mean"]["t2"][sc][r]
    return v["interference"], v["ci95"]


def tgt_prim(spec):
    pr = J[TG[spec]]["primary"]
    return pr["interference"], pr["ci95"]


# =========================================================== 2. gates
p()
p("=" * 60)
p("GATES (PLAN: nothing is read unless all pass)")
gates_ok = True
fails = []


def check(label, ok, detail):
    global gates_ok
    p(f"  [{'pass' if ok else 'FAIL'}] {label}: {detail}")
    if not ok:
        gates_ok = False
        fails.append(label)


# Gate 1
p("Gate 1. Rerun D + C-RADIOv4 reproduces the tally JSON within 1e-5 (global probe R2s, gain, gain CI):")
rr = J[EXPECTED_JSONS[0]]
ty = tally["R"]
rr_combo = combo_key(rr, "c-radiov4")
ty_combo = combo_key(ty, "c-radiov4")
items = [
    ("R2 dinov3|global", rr["probes"]["dinov3|global"]["r2_mean"], ty["probes"]["dinov3|global"]["r2_mean"]),
    ("R2 c-radiov4|global", rr["probes"]["c-radiov4|global"]["r2_mean"], ty["probes"]["c-radiov4|global"]["r2_mean"]),
    (f"R2 {rr_combo}", rr["probes"][rr_combo]["r2_mean"], ty["probes"][ty_combo]["r2_mean"]),
    ("gain(R|D)", rr["pairs"]["dinov3|c-radiov4|global"]["gain_B_given_A"], ty["pairs"]["dinov3|c-radiov4|global"]["gain_B_given_A"]),
    ("gain CI low", rr["pairs"]["dinov3|c-radiov4|global"]["gain_ci95"][0], ty["pairs"]["dinov3|c-radiov4|global"]["gain_ci95"][0]),
    ("gain CI high", rr["pairs"]["dinov3|c-radiov4|global"]["gain_ci95"][1], ty["pairs"]["dinov3|c-radiov4|global"]["gain_ci95"][1]),
]
g1 = True
for lab, a, b in items:
    ok = abs(a - b) <= 1e-5
    g1 &= ok
    p(f"      {lab}: rerun {a:.8f}  tally {b:.8f}  |diff| {abs(a - b):.2e}")
check("Gate 1 rerun vs tally", g1, "max |diff| " + f"{max(abs(a - b) for _, a, b in items):.2e}" + " (limit 1e-5)")

# Gate 2
p("Gate 2. dup:dinov3:")
dup_in = J[IN["dup:dinov3"]]
ck = dup_in["cka"]
ckdev = max(abs(v[t] - 1.0) for v in ck.values() for t in ("biased", "debiased"))
check("Gate 2a CKA(D, dup) = 1.000 +/- 1e-6", ckdev <= 1e-6,
      f"max |CKA - 1| over global/mean, both cams, biased and debiased = {ckdev:.2e}")
dg, dgci = in_gain("dup:dinov3")
check("Gate 2b input |gain(dup|D)| < 0.005", abs(dg) < 0.005, f"gain {fx(dg, 5)} {cis(dgci, 5)}")
dup_tg = J[TG["dup:dinov3"]]
mx = 0.0
cnt = 0
for sc in SCALINGS:
    for r in RANKS:
        mx = max(mx, abs(dup_tg["types"]["global"]["k_mean"]["t2"][sc][r]["interference"]))
        cnt += 1
for k in KS:
    for sc in SCALINGS:
        for r in RANKS:
            mx = max(mx, abs(dup_tg["types"]["global"]["by_k"][k]["t2"]["with_B"][sc][r]["interference"]))
            cnt += 1
check("Gate 2c target |interference| < 1e-6 under every scaling", mx < 1e-6,
      f"max |interference| over {cnt} cells (3 scalings x 3 ranks x (k-mean + k=4,16,32)) = {mx:.2e}")

# Gate 3
p("Gate 3. noise:4608 and shuffle:dinov3: |debiased CKA with DINOv3| < 0.02 (global and mean, both cams):")
for spec in ("noise:4608", "shuffle:dinov3"):
    c = J[IN[spec]]["cka"]
    vals = {k.split("|", 2)[2]: v["debiased"] for k, v in c.items()}
    mxv = max(abs(v) for v in vals.values())
    check(f"Gate 3 {spec}", mxv < 0.02 and len(vals) == 4,
          "debiased " + ", ".join(f"{k} {v:+.5f}" for k, v in vals.items()) + f"; max |.| {mxv:.5f}")

# Gate 4
p("Gate 4. Split counts equal the tally's (38,464 / 12,416 / 12,848) and 100 test demos:")
want = {"train": 38464, "val": 12416, "test": 12848}
tally_split = tally["R"]["split_frames"]
g4 = tally_split == want
p(f"      tally split_frames (read from the tally JSON): {tally_split}")
for name in EXPECTED_JSONS:
    d = J[name]
    ok = d["split_frames"] == want and d["n_frames"] == 63728
    if name.startswith("input"):
        ok &= d["n_test_demos"] == 100
        demos = str(d["n_test_demos"])
    else:
        nd = [d["types"]["global"]["by_k"][k]["n_test_demos"] for k in KS]
        ok &= all(x == 100 for x in nd)
        demos = "/".join(map(str, nd)) + " (k=4/16/32)"
    g4 &= ok
    p(f"      {name}: split {d['split_frames']['train']}/{d['split_frames']['val']}/{d['split_frames']['test']}, test demos {demos}: {'ok' if ok else 'MISMATCH'}")
check("Gate 4 splits and demos", g4, "all ten JSONs")

p()
p("ALL GATES PASS" if gates_ok else "GATE FAILURE: " + "; ".join(fails))
if not gates_ok:
    p("Per the PLAN, no decision-rule verdict is read when a gate fails: debug and amend, do not interpret.")
    print("\n".join(l.rstrip() for l in out))
    sys.exit(1)

# =========================================================== 3. numbers the rule uses
p()
p("=" * 60)
p("NUMBERS THE RULE USES (from the ten JSONs; real partners from the repo JSONs the PLAN names)")
p()
p("Input side, global features, seed 0. gain(B|D) = R2(D+B) minus R2(D), 95% demo-level bootstrap CI")
p("  control               dims  gain(B|D) [CI]                 R2 D+B   R2 B alone  alpha D/B/D+B  CKA(D,B) debiased global ag/wrist, mean ag/wrist")
dims_in = {"dup:dinov3": 2048, "noise:4608": 4608, "randproj:dinov3:4608": 4608, "shuffle:dinov3": 2048}
for spec in IN:
    d = J[IN[spec]]
    g, gc = in_gain(spec)
    ck_ = d["cka"]
    ckv = [ck_[f"dinov3|{spec}|{t}|{c}"]["debiased"] for t in ("global", "mean") for c in ("agentview_rgb", "eye_in_hand_rgb")]
    ckey = combo_key(d, spec)
    p(f"  {spec:<21} {dims_in[spec]:>5}  {fx(g)} {cis(gc)}   {d['probes'][ckey]['r2_mean']:.4f}   {d['probes'][spec + '|global']['r2_mean']:+.4f}      "
      f"{d['probes']['dinov3|global']['alpha']:g}/{d['probes'][spec + '|global']['alpha']:g}/{d['probes'][ckey]['alpha']:g}        "
      + " ".join(f"{v:+.4f}" for v in ckv))
g, gc = rr["pairs"]["dinov3|c-radiov4|global"]["gain_B_given_A"], rr["pairs"]["dinov3|c-radiov4|global"]["gain_ci95"]
p(f"  {'C-RADIOv4 (rerun)':<21} {4608:>5}  {fx(g)} {cis(gc)}   {rr['probes'][rr_combo]['r2_mean']:.4f}   {rr['probes']['c-radiov4|global']['r2_mean']:+.4f}      "
  f"{rr['probes']['dinov3|global']['alpha']:g}/{rr['probes']['c-radiov4|global']['alpha']:g}/{rr['probes'][rr_combo]['alpha']:g}")
p("  Recorded real partners (tally JSONs in the repo): gain(B|D) [CI]")
for letter, name, stem in REAL:
    g, gc = real_gain(letter)
    p(f"    {letter} {name:<11} {fx(g)} {cis(gc)}")
p("  Recorded R2 of DINOv3 alone (global) in the rerun: " + f"{rr['probes']['dinov3|global']['r2_mean']:.4f} {cis(rr['probes']['dinov3|global']['r2_ci95'])}")

p()
p("Target side, interference of the control as co-target, global, r = 64 (primary), mean over k in {4, 16, 32}")
p("  Under a' (primary), b (equalized), a'' (pooled); 95% CI in brackets; B loss share under a' (T0)")
p("  control               a' [CI]                          b [CI]                           a'' [CI]                         share a'")
for spec in TG:
    cells = []
    for sc in SCALINGS:
        v, c = tgt_k(spec, sc)
        cells.append(f"{fx(v, 5)} {cis(c, 5)}")
    sh = J[TG[spec]]["types"]["global"]["k_mean"]["t0_loss_share_B"]["a_prime"]
    p(f"  {spec:<21} " + "  ".join(cells) + f"  {sh:.3f}")
p("  Recorded real partners (target JSONs in the repo):")
for letter, name, stem in REAL:
    cells = []
    for sc in SCALINGS:
        v, c = real_tgt(letter, sc)
        cells.append(f"{fx(v, 5)} {cis(c, 5)}")
    p(f"    {letter} {name:<11} " + "  ".join(cells) + f"  {real_share(letter):.3f}")
p("  Also recorded for each control (k-mean, global): T1a R2 of the control target alone; T1b gain(B|D) on D's future from adding B as input")
for spec in TG:
    km = J[TG[spec]]["types"]["global"]["k_mean"]
    p(f"    {spec:<21} T1a R2 {km['t1a']['r2']:.4f}   T1b gain {fx(km['t1b_gain']['value'], 5)} {cis(km['t1b_gain']['ci95'], 5)}")
p("  Reference R2_D (DINOv3's target alone, same inputs; r = 64, a', k-mean) per control, next to the recorded real partners' range:")
refs = {s_: J[TG[s_]]["types"]["global"]["k_mean"]["t2"]["a_prime"]["64"]["r2_D_ref"] for s_ in TG}
p("    " + "; ".join(f"{s_} {v:.4f}" for s_, v in refs.items()))
rr_ = [tgt_real[l]["types"]["global"]["k_mean"]["t2"]["a_prime"]["64"]["r2_D_ref"] for l, _, _ in REAL]
p(f"    recorded real partners: {min(rr_):.4f} to {max(rr_):.4f}")
p("  By k (4 / 16 / 32), r = 64, interference and alpha (reference alpha -> alpha with the control), for the two controls with alpha changes at the primary cell:")
for s_ in ("noise:4608", "shuffle:dinov3"):
    bk = J[TG[s_]]["types"]["global"]["by_k"]
    for sc in SCALINGS:
        p(f"    {s_:<15} {SCALING_NAME[sc]:<3} " + "; ".join(f"k={k} {bk[k]['t2']['with_B'][sc]['64']['interference']:+.5f} (alpha {bk[k]['t2']['reference']['64']['alpha']:g} -> {bk[k]['t2']['with_B'][sc]['64']['alpha']:g})" for k in KS))
p("  Largest recorded real value per scaling (r = 64, k-mean): " + "; ".join(f"{SCALING_NAME[sc]} {max(real_tgt(l, sc)[0] for l, _, _ in REAL):+.5f}" for sc in SCALINGS)
  + "; shuffle:dinov3 k-mean: " + "; ".join(f"{SCALING_NAME[sc]} {tgt_k('shuffle:dinov3', sc)[0]:+.5f}" for sc in SCALINGS))
p("  T1b gain(B|D) on D's future from adding the control as input (k-mean; NOT used by the rule): " + "; ".join(f"{s_} {J[TG[s_]]['types']['global']['k_mean']['t1b_gain']['value']:+.5f}" for s_ in ("noise:4608", "noise:4608:0.3", "shuffle:dinov3")))
p("  Secondary (not used by the rule), a' by rank, k-mean:")
for spec in TG:
    cells = []
    for r in RANKS:
        v, c = tgt_k(spec, "a_prime", r)
        cells.append(f"r={r} {fx(v, 5)}")
    p(f"    {spec:<21} " + "   ".join(cells))

# =========================================================== 4. decision rule
p()
p("=" * 60)
p("DECISION RULE, applied word for word (PLAN.md 'Decision rule')")
p(f"Margins: input m = {M_IN}; target m_T = {M_T}.")
p()
p("INPUT SIDE on gain(control|D), global, seed 0 (controls: noise:4608, shuffle:dinov3, randproj:dinov3:4608; dup is a gate, not a verdict input)")
vi = {s: in_gain(s) for s in ("noise:4608", "shuffle:dinov3", "randproj:dinov3:4608")}
neg_hits = []
for s in ("noise:4608", "shuffle:dinov3"):
    v, c = vi[s]
    hit = v <= -M_IN and c[1] < 0
    p(f"  Clause 'Can go negative: noise:4608 or shuffle:dinov3 has point value <= -0.005 and a CI whose upper end is below 0.'")
    p(f"    {s}: point {fx(v, 5)} <= -0.005: {v <= -M_IN}; CI upper {fx(c[1], 5)} < 0: {c[1] < 0}  -> {'FIRES' if hit else 'does not fire'}")
    if hit:
        neg_hits.append(s)
pos_hits = []
for s in vi:
    v, c = vi[s]
    hit = v >= M_IN and c[0] > 0
    p(f"  Clause 'Capacity floor above zero: noise:4608, shuffle:dinov3 or randproj:dinov3:4608 has point value >= +0.005 and a CI whose lower end is above 0.'")
    p(f"    {s}: point {fx(v, 5)} >= +0.005: {v >= M_IN}; CI lower {fx(c[0], 5)} > 0: {c[0] > 0}  -> {'FIRES' if hit else 'does not fire'}")
    if hit:
        pos_hits.append(s)
unres_in = []
for s in vi:
    v, c = vi[s]
    contains = [t for t in (-M_IN, M_IN) if c[0] <= t <= c[1]]
    p(f"  Clause 'Unresolved: a control whose CI contains -0.005 or +0.005 is assigned to neither side; stage 2 replicates it before any call that depends on it.'")
    p(f"    {s}: CI {cis(vi[s][1], 5)} contains {contains if contains else 'neither -0.005 nor +0.005'}")
    if contains:
        unres_in.append(s)
within = all(abs(vi[s][0]) <= M_IN for s in vi)
p("  Clause 'Floor near zero: no control is unresolved and none meets either condition above (every point value within +/-0.005). The input test cannot flag harm.'")
p(f"    no control unresolved: {not unres_in}; none meets either condition: {not neg_hits and not pos_hits}; every point value within +/-0.005: {within}")
floor_zero = (not unres_in) and (not neg_hits) and (not pos_hits) and within
p("  Clause 'Both of the first two can fire ...; then report both and call the result mixed.'")
mixed_in = bool(neg_hits) and bool(pos_hits)
p(f"    can-go-negative fires: {bool(neg_hits)}; capacity-floor fires: {bool(pos_hits)}  -> mixed: {mixed_in}")
if mixed_in:
    in_outcome = "mixed (report both)"
elif neg_hits:
    in_outcome = "Can go negative"
elif pos_hits:
    in_outcome = "Capacity floor above zero"
elif floor_zero:
    in_outcome = "Floor near zero"
else:
    in_outcome = "Unresolved (no clause met; stage 2 replicates before any call)"
p(f"  INPUT-SIDE OUTCOME (PLAN's label): {in_outcome}" + (f" (fired by {', '.join(neg_hits)})" if neg_hits else ""))
if unres_in:
    p(f"  Unresolved controls: {unres_in}")
p("  Clause needing the paired difference: under 'Capacity floor above zero' every real partner is re-read as gain minus the matched-width noise gain (paired CI)."
  + (" That clause did not fire, so the verdict does not depend on the paired CI." if not pos_hits else " That clause fired: the paired CI is needed and is NOT COMPUTED."))

p()
p("TARGET SIDE on interference under a' (r = 64, global, k-mean), seed 0")
n03, n03c = tgt_k("noise:4608:0.3", "a_prime")
rp, rpc = tgt_k("randproj:dinov3:4608", "a_prime")
dpv, dpc = tgt_k("dup:dinov3", "a_prime")
dup_zero = max(abs(dup_tg["types"]["global"]["k_mean"]["t2"][sc][r]["interference"]) for sc in SCALINGS for r in RANKS) < 1e-6
p("  Clause 'Block size and loudness explain R: both noise:4608:0.3 and randproj:dinov3:4608 have interference >= 0.0100 (two thirds of R's 0.0151). Then the target test measures the size and loudness of the co-target block.'")
big = n03 >= TARGET_BIG and rp >= TARGET_BIG
p(f"    noise:4608:0.3 {fx(n03, 5)} >= 0.0100: {n03 >= TARGET_BIG}; randproj {fx(rp, 5)} >= 0.0100: {rp >= TARGET_BIG}  -> {'FIRES' if big else 'does not fire'}")
p("  Clause 'Partner-specific: both have interference <= 0.0050 (one third of R's) and dup is 0. Then the test sees something about the partner beyond block size and loudness; partners are above the floor only if their value exceeds 0.0050 (R, L, G yes; S no).'")
small = n03 <= TARGET_SMALL and rp <= TARGET_SMALL and dup_zero
p(f"    noise:4608:0.3 {fx(n03, 5)} <= 0.0050: {n03 <= TARGET_SMALL}; randproj {fx(rp, 5)} <= 0.0050: {rp <= TARGET_SMALL}; dup is 0 (|x| < 1e-6 everywhere): {dup_zero}  -> {'FIRES' if small else 'does not fire'}")
p("    Real partners against 'exceeds 0.0050' (values read from the repo target JSONs):")
for letter, name, stem in REAL:
    v, c = real_tgt(letter, "a_prime")
    p(f"      {letter} {name}: {fx(v, 5)} exceeds 0.0050: {'yes' if v > TARGET_SMALL else 'no'}")
p("  Clause 'Mixed: anything else; report both numbers, no claim.'")
if big:
    t_outcome = "Block size and loudness explain R"
elif small:
    t_outcome = "Partner-specific"
else:
    t_outcome = "Mixed"
p(f"    TARGET-SIDE OUTCOME (PLAN's label, first part of the rule): {t_outcome}")
p("  Clause 'Can go negative: any control with CI upper end below -0.003. Floor near zero: every control's CI lower end is at or above -0.003. Unresolved: a control whose CI contains -0.003.' (all five controls, a', r = 64, k-mean)")
neg_t, unres_t = [], []
low_ok = True
for s in TG:
    v, c = tgt_k(s, "a_prime")
    h = c[1] < -M_T
    u = c[0] <= -M_T <= c[1]
    lo_ok = c[0] >= -M_T
    low_ok &= lo_ok
    p(f"    {s:<21} point {fx(v, 5)} CI {cis(c, 6)}: upper < -0.003: {h}; lower >= -0.003: {lo_ok}; CI contains -0.003: {u}")
    if h:
        neg_t.append(s)
    if u:
        unres_t.append(s)
if neg_t:
    neg_out = "Can go negative (fires for " + ", ".join(neg_t) + "; PLAN: look for a bug or a mechanism such as an alpha effect before calling it a finding)"
elif unres_t:
    neg_out = "Unresolved (" + ", ".join(unres_t) + ")"
elif low_ok:
    neg_out = "Floor near zero"
else:
    neg_out = "none of the three"
p(f"    NEGATIVITY OUTCOME (PLAN's label): {neg_out}")
bv, bc = tgt_k("noise:4608", "b")
p("  Clause 'Under b: if noise:4608 under b is within 0.003 of or above the smallest real value (>= 0.0177, SAM 3's 0.0207 minus 0.003), the b picture \"every partner costs about 0.02\" is the floor for any equal-weight co-target whose future is not predictable from the inputs; if it is <= 0.0104 (half of SAM 3's 0.0207) the b values are partner-specific; in between, mixed, no claim.'")
sam_b, _ = real_tgt("S", "b")
p(f"    noise:4608 under b (r = 64, k-mean): {fx(bv, 5)} {cis(bc, 5)}; recorded SAM 3 under b {sam_b:.5f} (minus 0.003 = {sam_b - 0.003:.4f}, half = {sam_b / 2:.4f}; PLAN literals {B_HI} and {B_LO})")
if bv >= B_HI:
    b_out = "the b picture is the floor for any equal-weight co-target"
elif bv <= B_LO:
    b_out = "the b values are partner-specific"
else:
    b_out = "mixed, no claim"
p(f"    >= {B_HI}: {bv >= B_HI}; <= {B_LO}: {bv <= B_LO}  -> {b_out}")

# ---- stage 2 trigger
p()
p("STAGE 2 TRIGGER (PLAN: seeds 1 and 2 for any control whose stage 1 value is within 0.002 of one of the thresholds, or an 'unresolved' control)")
p("  Checked: each control's point value against the thresholds the rule applies to that control; and, where the rule compares a CI end, that CI end too. |value - threshold| <= 0.002 triggers.")
trig = []


def tcheck(control, side, what, val, thr, thrlabel):
    d = abs(val - thr)
    hit = d <= TRIGGER + 1e-12
    p(f"    {side} {control:<21} {what:<16} {val:+.5f} vs {thrlabel:<24} |diff| {d:.5f}  {'TRIGGER' if hit else ''}")
    if hit:
        trig.append((side, control, what, val, thrlabel, d))


p("  Input side (thresholds -0.005 and +0.005 on the point value; 0 on the CI end of the 'fires' conditions; +/-0.005 on CI ends for 'unresolved'):")
for s in vi:
    v, c = vi[s]
    tcheck(s, "input", "point", v, -M_IN, "-0.005")
    tcheck(s, "input", "point", v, +M_IN, "+0.005")
    tcheck(s, "input", "CI lower end", c[0], -M_IN, "-0.005 (unresolved)")
    tcheck(s, "input", "CI lower end", c[0], +M_IN, "+0.005 (unresolved)")
    tcheck(s, "input", "CI upper end", c[1], -M_IN, "-0.005 (unresolved)")
    tcheck(s, "input", "CI upper end", c[1], +M_IN, "+0.005 (unresolved)")
    if s in ("noise:4608", "shuffle:dinov3"):
        tcheck(s, "input", "CI upper end", c[1], 0.0, "0 (negative cond.)")
    tcheck(s, "input", "CI lower end", c[0], 0.0, "0 (positive cond.)")
p("  Target side (0.0100 and 0.0050 for noise:4608:0.3 and randproj on a'; 0.0177 and 0.0104 for noise:4608 under b; -0.003 for every control's CI ends):")
for s, v in (("noise:4608:0.3", n03), ("randproj:dinov3:4608", rp)):
    tcheck(s, "target", "a' point", v, TARGET_BIG, "0.0100")
    tcheck(s, "target", "a' point", v, TARGET_SMALL, "0.0050")
tcheck("noise:4608", "target", "b point", bv, B_HI, "0.0177")
tcheck("noise:4608", "target", "b point", bv, B_LO, "0.0104")
for s in TG:
    if s == "dup:dinov3":
        continue
    v, c = tgt_k(s, "a_prime")
    tcheck(s, "target", "a' CI lower end", c[0], -M_T, "-0.003")
    tcheck(s, "target", "a' CI upper end", c[1], -M_T, "-0.003")
p()
if trig or unres_in or unres_t:
    p("  STAGE 2 TRIGGERED for:")
    for t in trig:
        p(f"    {t[0]} {t[1]}: {t[2]} {t[3]:+.5f}, threshold {t[4]}, |diff| {t[5]:.5f}")
    for s in unres_in:
        p(f"    input {s}: unresolved (CI contains +/-0.005)")
    for s in unres_t:
        p(f"    target {s}: unresolved (CI contains -0.003)")
    p("  (Stage 2 is a second Job and needs the user's OK first, per the PLAN.)")
else:
    p("  No control is within 0.002 of any threshold and none is unresolved: stage 2 is not triggered by the rule. (The PLAN also allows the 3 seeds up front if the user wants them, standards 07.)")

# =========================================================== 5. predictions
p()
p("=" * 60)
p("PREDICTIONS (PLAN, written before running; speculative) against the results")


def rng_line(label, val, lo, hi, extra=""):
    inside = lo <= val <= hi
    where = "inside" if inside else ("below" if val < lo else "above")
    p(f"  {label}: predicted [{lo:+g}, {hi:+g}]{extra}; result {fx(val, 5)}: {where} the range" + ("" if inside else "  (MISS)"))


nv, nc = vi["noise:4608"]
rng_line("Input noise:4608", nv, -0.03, 0.0, ", most likely -0.015 to -0.005")
p(f"    inside the 'most likely' band [-0.015, -0.005]: {-0.015 <= nv <= -0.005}" + ("" if -0.015 <= nv <= -0.005 else "  (more negative than the most-likely band, still inside the full range)"))
p(f"    PLAN's reading 'below -0.005 with CI under 0: the input test can go negative': point {fx(nv, 5)}, CI upper {fx(nc[1], 5)} -> {nv < -0.005 and nc[1] < 0}")
rng_line("Input dup:dinov3 (within +/-0.003)", dg, -0.003, 0.003, "; outside +/-0.005 would be a bug")
rv, rc = vi["randproj:dinov3:4608"]
rng_line("Input randproj:dinov3:4608", rv, -0.005, 0.005)
sv, sc_ = vi["shuffle:dinov3"]
rng_line("Input shuffle:dinov3", sv, -0.01, 0.0, ", nearer zero than noise:4608")
p(f"    nearer zero than noise:4608: |{sv:+.5f}| < |{nv:+.5f}|: {abs(sv) < abs(nv)}")
gR, gRc = rr["pairs"]["dinov3|c-radiov4|global"]["gain_B_given_A"], rr["pairs"]["dinov3|c-radiov4|global"]["gain_ci95"]
p(f"  Input D + C-RADIOv4 rerun: predicted +0.037 [+0.032, +0.042], reproducing the tally within 1e-5; result {fx(gR)} {cis(gRc)}; gate 1 passed.")
p("  Target interference, a', r = 64, mean over k:")
p(f"    dup:dinov3: predicted exactly 0 (|x| < 1e-6) under every scaling, rank and k; result max |x| {mx:.2e}: {'as predicted' if mx < 1e-6 else 'MISS'}")
rng_line("    randproj:dinov3:4608", rp, 0.000, 0.015, ", central 0.005")
rng_line("    noise:4608:0.3", n03, 0.002, 0.02, ", central 0.008")
n46, n46c = tgt_k("noise:4608", "a_prime")
rng_line("    noise:4608", n46, 0.005, 0.035, ", central 0.015")
p(f"      'larger than the 0.3 version': noise:4608 {fx(n46, 5)} > noise:4608:0.3 {fx(n03, 5)}: {n46 > n03}")
shv, shc = tgt_k("shuffle:dinov3", "a_prime")
rng_line("    shuffle:dinov3", shv, 0.003, 0.02)
rng_line("    noise:4608 under b", bv, 0.015, 0.03)
dbm = max(abs(dup_tg["types"]["global"]["k_mean"]["t2"]["b"][r]["interference"]) for r in RANKS)
p(f"    dup under b is 0: max |x| over ranks {dbm:.2e}: {'as predicted' if dbm < 1e-6 else 'MISS'}")

# =========================================================== 6. alphas
p()
p("=" * 60)
p("ALPHA CHOSEN WITH AND WITHOUT EACH CONTROL (the PLAN asks for them side by side)")
p("Input side, global probe (ridge strength chosen on validation demos):")
p("  control               alpha D alone  alpha control alone  alpha D+control")
for spec in IN:
    d = J[IN[spec]]
    ckey = combo_key(d, spec)
    p(f"  {spec:<21} {d['probes']['dinov3|global']['alpha']:>13g}  {d['probes'][spec + '|global']['alpha']:>19g}  {d['probes'][ckey]['alpha']:>15g}")
p(f"  {'C-RADIOv4 (rerun)':<21} {rr['probes']['dinov3|global']['alpha']:>13g}  {rr['probes']['c-radiov4|global']['alpha']:>19g}  {rr['probes'][rr_combo]['alpha']:>15g}")
p("Target side, T2 reduced-rank fits (alpha for DINOv3's target alone, same inputs (the inputs still include the control), 'ref', versus with the control as co-target under a' / b / a''); primary rank r = 64, then every cell where they differ:")
for spec in TG:
    p(f"  {spec}")
    jj = J[TG[spec]]["types"]["global"]["by_k"]
    for k in KS:
        ref = jj[k]["t2"]["reference"]["64"]["alpha"]
        wb = [jj[k]["t2"]["with_B"][sc]["64"]["alpha"] for sc in SCALINGS]
        p(f"    k={k:<2} r=64: ref {ref:g} | a' {wb[0]:g}  b {wb[1]:g}  a'' {wb[2]:g}")
    diffs = []
    for k in KS:
        for r in RANKS:
            ref = jj[k]["t2"]["reference"][r]["alpha"]
            for sc in SCALINGS:
                a = jj[k]["t2"]["with_B"][sc][r]["alpha"]
                if a != ref:
                    diffs.append(f"k={k} r={r} {SCALING_NAME[sc]}: ref {ref:g} -> {a:g}")
    ncell = len(KS) * len(RANKS) * len(SCALINGS)
    p(f"    cells (of {ncell}: 3 k x 3 ranks x 3 scalings) where alpha with the control differs from the reference: {len(diffs)}")
    for dline in diffs:
        p(f"      {dline}")
    t1 = [(k, jj[k]["t1b"]["D_only"]["alpha"], jj[k]["t1b"]["D_plus_B"]["alpha"]) for k in KS]
    p("    T1b input-side probe alphas D only / D+B: " + "; ".join(f"k={k} {a:g}/{b:g}" for k, a, b in t1))

# =========================================================== 7. run times
p()
p("=" * 60)
p("RUN TIMES against the PLAN's estimate (PLAN 'Cost'; 'seconds' field of each result JSON)")
est = {"dup:dinov3": 1374, "noise:4608": 1572, "randproj:dinov3:4608": 1572, "shuffle:dinov3": 1374}
p("Input side (PLAN basis: 1,374 s for 4,096-d runs, 1,572 s for 6,656-d runs):")
for spec in IN:
    s = J[IN[spec]]["seconds"]
    p(f"  {spec:<21} {s:7.0f} s   PLAN {est[spec]} s   ratio {s / est[spec]:.2f}")
s = rr["seconds"]
p(f"  {'C-RADIOv4 (rerun)':<21} {s:7.0f} s   PLAN 1572 s   ratio {s / 1572:.2f}")
est_t = {"dup:dinov3": 1850, "shuffle:dinov3": 1850, "noise:4608:0.3": 4190, "noise:4608": 4190, "randproj:dinov3:4608": 4190}
p("Target side (PLAN basis: about 1,850 s for 4,096-d runs, about 4,190 s for 6,656-d runs):")
for spec in TG:
    s = J[TG[spec]]["seconds"]
    p(f"  {spec:<21} {s:7.0f} s   PLAN about {est_t[spec]} s   ratio {s / est_t[spec]:.2f}")
tot_in = sum(J[IN[s]]["seconds"] for s in IN) + rr["seconds"]
tot_t = sum(J[TG[s]]["seconds"] for s in TG)
p(f"Sum of the ten script times: input {tot_in / 3600:.2f} h, target {tot_t / 3600:.2f} h, together {(tot_in + tot_t) / 3600:.2f} h.")
stg = re.findall(r"^staged (\S+): (\S+) in (\d+)s", raw.decode(), re.M)
stg_s = sum(int(x[2]) for x in stg)
p("Staging lines in the log: " + "; ".join(f"{a} {b} in {c} s" for a, b, c in stg) + f" (total {stg_s} s = {stg_s / 60:.1f} min; PLAN said staging and download about 5 min)")
p(f"Whole Job: started {JOB_START}, finished {JOB_FINISH} (times from the lead's kubectl read, not in the pod log; the finish equals the last log timestamp): "
  f"{JOB_WALL_S // 3600} h {JOB_WALL_S % 3600 // 60} min {JOB_WALL_S % 60} s = {JOB_WALL_S / 3600:.2f} h wall; PLAN estimate about 6.7 h (pessimistic about 10 h); ratio {JOB_WALL_S / 3600 / 6.7:.2f}.")
p(f"CPU-hours: 2 CPU requested x {JOB_WALL_S / 3600:.2f} h = {2 * JOB_WALL_S / 3600:.1f} CPU-h requested (not a measured use; no CPU reading was taken); PLAN estimate about 13 CPU-h.")

# =========================================================== 8. not computed, memory, warnings
p()
p("=" * 60)
p("NOT COMPUTED")
p("  Paired difference gain(C-RADIOv4|DINOv3) minus gain(control|DINOv3) with a CI: NOT COMPUTED. It needs the bootstrap draws files (input-draws-*.npz, written by --save-draws on the cluster volume")
p("  /mnt/features/stacking-encoders/control-partners/), which are not in this repo or on this machine. They would have to be copied off the volume. No unpaired interval is substituted for it.")
p("  Which rule clauses need it: only the 'Capacity floor above zero' re-reading (every real partner as gain minus matched-width noise gain, paired CI)"
  + (": that clause did not fire." if not pos_hits else ": that clause fired.") + " The prediction note that R's +0.037 is 'information on top of a width penalty (paired difference larger than +0.037)' cannot be checked without it.")
p("  Memory: UNKNOWN. No RSS reading was taken during the run (the lead's LOG entry says RSS was not checked) and the pod log prints only the node's MemTotal and MemFree, no pod memory figure, so nothing is inferred about peak memory against the 14Gi request.")
p("  Mean-type features and grid CKA for controls: out of scope in the PLAN. The input JSONs do carry mean-type CKA (printed above for sanity); no mean-type probes or targets were run.")
p("  Stage 2 (seeds 1 and 2) and the noise-matrix variation in the CIs: not run (the bootstrap CIs hold the noise draw fixed; PLAN 'Seeds').")

p()
p("LOG WARNINGS AND ERRORS (scan of the pod log for warn/error/traceback/killed/oom/nan/fail/edge/deprecat/exception/retry/notice; lines outside the JSON bodies)")
body_lines = set()
for n, i in enumerate(heads):
    j = heads[n + 1] if n + 1 < len(heads) else len(lines)
    for q in range(i + 1, j):
        if lines[q].startswith(("{", "}", " ", "\t")) or lines[q] in ("",):
            body_lines.add(q)
pat = re.compile(r"warn|error|traceback|killed|oom|\bnan\b|fail|edge|deprecat|exception|retry|notice", re.I)
hits = [(q + 1, l) for q, l in enumerate(lines) if q not in body_lines and pat.search(l)]
for q, l in hits:
    p(f"  log line {q}: {l[:200]}")
if not hits:
    p("  none")
strict = re.compile(r"traceback|killed|oom|\bnan\b|exception|error", re.I)
sh = [q + 1 for q, l in enumerate(lines) if q not in body_lines and strict.search(l)]
p("  Lines matching traceback/killed/oom/nan/exception/error outside the JSON bodies: " + (", ".join(map(str, sh)) if sh else "none"))

print("\n".join(l.rstrip() for l in out))
