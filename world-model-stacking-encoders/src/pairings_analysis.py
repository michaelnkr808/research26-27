"""Stage 2 analysis: the locked decision rule of experiments/2026-11-15-pretrained-start-pairings/PLAN.md.

This file implements the PLAN's "Comparison statistic (defined before any result)" (PLAN lines 115-122) and its
"Decision rule" (lines 153-212) with the options the user signed off in the "Amendments" (lines 315 onward):
threshold T3 = max(2.7, f(n) x sigma_hat), margins M2, calibration C-a (required), clean gate G1 at 36, coverage 1%,
SigLIP 2 alone and C-RADIOv4 alone trained (Open item b = yes), and the one-time extension (seeds 3 and 4 of dinov3
and dinov3+c-radiov4) with sign condition S-4of5, for a complete three-seed I only; and the three cases the user
decided in the amendment of 2026-11-17, item 3 (a) to (c) (PLAN lines 398-405; "DECIDED CASES" below). It never
changes the rule. Any
other design (a scope cut, other options) is a dated PLAN amendment first and a code change second.

Run (stdlib only):
  /opt/homebrew/bin/python3.11 -I pairings_analysis.py --input stage2-totals.json --out stage2-analysis.json
Exit 0: outcome computed (printed, and written to --out). Exit 2: the input is incomplete or malformed; nothing is
written and no outcome is given. Exit 3: outcome computed and written as for 0, but the input reaches a case the PLAN
leaves to the user; it is printed as "STOP FOR THE USER" and listed under "stops_for_user" (see "OPEN CASE" below).
The exit-3 stop is a reminder after the fact, not the enforcement point: the run session must stop and ask the user
at the LIBERO-Plus evaluation step itself, before that checkpoint is evaluated (PLAN line 405).

Input (one JSON object):
  {"checkpoints": [<entry>, ...]}
One entry per (config, train_seed) of the design: configs none, dinov3, dinov3+siglip2, dinov3+c-radiov4, siglip2,
c-radiov4, each with train seeds 0, 1, 2; plus, only once the extension has been run, seeds 3 and 4 of dinov3 and
dinov3+c-radiov4 (all four entries). A missing entry is an input error: a run that could not be produced is listed
with "status": "missing" (Design, "Training failures"; Decision rule item 6).
  {"config": "dinov3", "train_seed": 0, "status": "ok",
   "checkpoint": "<optional path or id, reported as given>",
   "clean": {"libero_spatial": {"successes": 451, "episodes": 500}, "libero_object": {...},
             "libero_goal": {...}, "libero_10": {...}},
   "gate_diagnosis": {"fault_found": false, "note": "..."},     # required only when B < 36 (item 5)
   "libero_plus": {"41": {"successes": 8012, "evaluated": 10030, "missing": 0},
                   "42": {...}, "43": {...}}}     # optional only for a checkpoint over the clean 1% line (item 3 (c))
  {"config": "dinov3+c-radiov4", "train_seed": 2, "status": "missing", "reason": "non-finite loss, rerun failed"}
"evaluated" counts the LIBERO-Plus episodes scored under that eval seed after reruns, "missing" those still missing
(rollout.py's errors file); evaluated + missing must be 10,030. Clean: the 2,000 clean rollouts, 500 per suite
(50 per task, 10 tasks), eval seed 41; "episodes" counts the clean rollouts of that suite that produced a record
after reruns, so 500 - episodes are missing (a rollout with no record: it raised or its shard did not finish; not a
rollout the policy failed); more than 500 is an input error. The counts come from the stage 1 tally
(tally_proxy_clean.py's per-suite counts or tally_per_task.py).

Statistic (PLAN lines 116-122): rate = successes / evaluated per eval seed, Y(c, s) = mean of the three rates, in
points. A checkpoint with more than 1% of its 30,090 LIBERO-Plus rollouts missing counts as a missing seed.
B(c, s) = mean of the four suite rates, each rate = successes / rollouts with a record; missing clean rollouts leave
the denominators and are reported per checkpoint (item 3 (c)). A checkpoint with more than 1% of its 2,000 clean
rollouts missing (21 or more) counts as a missing seed for the whole analysis, the LIBERO-Plus tests included. d_s(A - B) = Y(A, s) - Y(B, s) on the seeds both have; dbar their mean.
sigma_hat: pooled within-configuration SD over every configuration with at least 2 seeds. f(n) = 2 x sqrt(2 / n).
All comparisons are exact (fractions; the square root is compared by squaring), so values on a boundary land on the
side the rule states: dbar <= -T, dbar >= T, and the strict ">" of M2.

Output: per-checkpoint rates, Y, eval-seed SD, missing counts, B and gate; per-config Ybar and SD; sigma_hat; per test
the paired seeds, d_s, dbar, n, the threshold used and the result; the extension status; and the outcome label as the
PLAN's Decision rule item 8 names it: "C fails, under C-a", "H and K", "H, not K, and H_L", "H, not K, not H_L",
"N and K", "N, not K", "I".

When C fails (C-a), nothing is claimed (PLAN line 192): the harm test and "below both members" are output as
"not claimed (C fails)", and so is the helpful-pairing state (K, H_L or unresolved), a claim about the contrast; the
comparisons themselves are still reported, the computed values under the keys "harm_test_as_computed_not_claimed"
and "helpful_pairing_as_computed_not_claimed".

DECIDED CASES (PLAN amendment 2026-11-17, item 3; until then the script reported these as pending the user):
  (a) (line 399) If the outcome is "C fails, under C-a", including C not established (a missing none or dinov3 seed,
      item 6) and read as that branch, the extension of item 7 is not run, even when the three-seed harm test is I
      with all three seeds present. Status EXT_C_FAILS; seeds 3 and 4 supplied anyway are refused (input error).
  (b) (line 400) "Below both members" (dinov3+c-radiov4 against c-radiov4 alone) stays in. After an extension it
      uses seeds 0 to 2 only, paired by seed, with the values and thresholds computed before the extension (the
      three-seed sigma_hat, T and f(3)); it is stated only when H (on five seeds) holds and this three-seed test
      holds, and the output says the harm test is on five seeds and this comparison on three
      ("below_both_members_basis"). Under "C fails, under C-a" it is not stated (and no extension is run, (a)).
  (c) (lines 403-404) Clean coverage, as above: up to 20 missing of 2,000 leave B's denominators and are reported;
      21 or more make the checkpoint a missing seed for the whole analysis, with item 6's consequences (a missing
      dinov3+c-radiov4 or dinov3 seed: harm I, not qualifying for the extension; a missing none or dinov3 seed: C
      not established, read as "C fails"; a missing dinov3+siglip2 or dinov3 seed: helpful pairing unresolved).
OPEN CASE (line 405, not decided by the user; kept out of this script's decisions): whether a checkpoint over the
clean 1% line still gets its LIBERO-Plus evaluation. Its "libero_plus" may be given (validated and reported, not
used: the seed is missing either way) or left out. The gate still applies to it: a computable B below 36 needs a
"gate_diagnosis" (item 5; the user's answer on PR #59, 2026-11-17). Without a recorded answer the checkpoint is
listed under "stops_for_user" and the exit code is 3. The user's answer for that checkpoint is recorded in its entry:
  "user_decision_libero_plus": "evaluate" | "skip", "decided": "<date and the user's words>"
Then the stop clears (no exit 3) and the report prints the recorded decision under "user_decisions". The two fields
go together, and only on a checkpoint over the clean line (anywhere else they are an input error).
ONE-SEED REPORT (PLAN amendment 2026-11-18, item 3; selected only by the explicit flag --one-seed, since the amendment
does not say how the mode is chosen). Stage 2's first pass trains seed 0 of each configuration only; no test of the
decision rule is read from it. Run:
  /opt/homebrew/bin/python3.11 -I pairings_analysis.py --one-seed --input stage2-seed0.json --out stage2-seed0-report.json
Input: the format above with exactly one entry per configuration, all with "train_seed": 0 (seeds 1 and 2 are
deferred and are not entered at all, never as "status": "missing"). A seed 0 training run that still fails after its
one rerun is listed with "status": "missing" and a "reason", and is reported as "no checkpoint". Each entry with
"libero_plus" also carries the per-category counts, adding up to that eval seed's totals:
  "libero_plus_categories": {"41": {"Camera Viewpoints": {"successes": 1203, "evaluated": 1500}, ...},
                             "42": {...}, "43": {...}}
Without the flag, a one-seed input stops with an input error (the locked rule needs seeds 0 to 2); with the flag, any
entry at another seed stops with an input error. Output: per checkpoint clean B, Y with the SD across its three
eval-seed rates, the three rates, the per-category rates and coverage; the seven differences d_0 with the paper's
beside them and the three per-eval-seed differences; one label per comparison ("same direction as the paper",
"opposite direction", "no difference", or no label when a checkpoint is absent or counts as missing on coverage);
the one overall line, read only when dinov3 - none is labelled "same direction as the paper"; the plain count of
labels; and every test of the rule shown as not computed. Exit codes as above (3: the open clean-coverage case).
lp_eval.py is not imported: its paired/verdict work on per-episode tables of the proxy's subset rule, not on totals.
"""

import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

NONE, D, DL, DR, L_, R_ = "none", "dinov3", "dinov3+siglip2", "dinov3+c-radiov4", "siglip2", "c-radiov4"
CONFIGS = (NONE, D, DL, DR, L_, R_)          # Open item b = yes (sign-off amendment)
BASE_SEEDS = (0, 1, 2)                       # Open item c: three seeds
EXT_SEEDS = (3, 4)                           # Decision rule item 7
EXT_CONFIGS = (D, DR)
EVAL_SEEDS = ("41", "42", "43")              # LIBERO-Plus eval seeds (PLAN line 112)
LP_EPISODES = 10030
LP_ROLLOUTS = 3 * LP_EPISODES                # 30,090
MISSING_MAX = Fraction(1, 100)               # coverage: more than 1% missing = missing seed (Open item j)
CLEAN_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CLEAN_PER_SUITE = 500
CLEAN_ROLLOUTS = len(CLEAN_SUITES) * CLEAN_PER_SUITE   # 2,000; more than 1% (21 or more) missing = missing seed
GATE = Fraction(36)                          # G1 (item 5)
FLOOR = Fraction(27, 10)                     # 2.7: T3's floor and M2's margin
EXT_MIN_NEGATIVE = 4                         # S-4of5

C_FAILS = "C fails, under C-a"
LABELS = {("H", "K"): "H and K", ("H", "H_L"): "H, not K, and H_L", ("H", "unresolved"): "H, not K, not H_L",
          ("N", "K"): "N and K", ("N", "H_L"): "N, not K", ("N", "unresolved"): "N, not K"}

EXT_NA = "not applicable: the harm test is not I"
EXT_DUE = "due: complete three-seed I with C passed; train seeds 3 and 4 of dinov3 and dinov3+c-radiov4 (item 7)"
EXT_MISSING = "not qualifying: an I caused by a missing seed does not trigger the extension (item 7)"
EXT_C_FAILS = ("not run: C fails under C-a (or C is not established and is read as that branch); seeds 3 and 4 could "
               "not change the outcome or any claim (PLAN amendment 2026-11-17, item 3 (a))")
EXT_DONE = ("done: seeds 3 and 4 analysed; H and N on five seeds, C, K, H_L and the c-radiov4 comparison on seeds "
            "0 to 2 (two-stage design)")
NOT_CLAIMED = "not claimed (C fails)"

BELOW_BOTH_3 = ("H and the c-radiov4 comparison on seeds 0 to 2, paired by seed (three-seed sigma_hat, T and f(3))")
BELOW_BOTH_5 = ("the harm test H is on five seeds (seeds 0 to 4, re-pooled sigma_hat, f(5), S-4of5); the c-radiov4 "
                "comparison is on three (seeds 0 to 2 only, paired by seed, with the three-seed sigma_hat, T and f(3) "
                "computed before the extension) (PLAN amendment 2026-11-17, item 3 (b))")

STOP_CLEAN_LP = ("more than 1% of the 2,000 clean rollouts missing after reruns: a missing seed for the whole analysis "
                 "(PLAN amendment 2026-11-17, item 3 (c)). Whether this checkpoint still gets its LIBERO-Plus "
                 "evaluation is not decided by the user (still open, PLAN line 405): stop and ask the user")


class InputError(ValueError):
    pass


def f_of(n):
    """f(n) = 2 x sqrt(2 / n): two standard errors of a paired mean, in units of sigma (PLAN line 155)."""
    return 2 * math.sqrt(2 / n)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def _count(x, what):
    if isinstance(x, bool) or not isinstance(x, int) or x < 0:
        raise InputError(f"{what}: need a non-negative integer, got {x!r}")
    return x


def _clean_B(rec, where):
    """Per-suite clean rates over the rollouts with a record, B, and the missing counts (item 3 (c)). A suite with
    no record at all has no rate, and then B is None (the checkpoint is then far over the 1% line)."""
    clean = rec.get("clean")
    if not isinstance(clean, dict):
        raise InputError(f"{where}: no clean tally (the gate needs B for every checkpoint, item 5)")
    if sorted(clean) != sorted(CLEAN_SUITES):
        raise InputError(f"{where}: clean suites {sorted(clean)}, need {sorted(CLEAN_SUITES)}")
    rates, missing = {}, {}
    for suite in CLEAN_SUITES:
        s = clean[suite]
        if not isinstance(s, dict):
            raise InputError(f"{where}: clean {suite} is not an object")
        n = _count(s.get("episodes"), f"{where} clean {suite} episodes")
        k = _count(s.get("successes"), f"{where} clean {suite} successes")
        if n > CLEAN_PER_SUITE:
            raise InputError(f"{where}: clean {suite} has {n} episodes, more than the {CLEAN_PER_SUITE} of the "
                             "protocol (a rollout counted twice?)")
        if k > n:
            raise InputError(f"{where}: clean {suite} successes {k} > episodes {n}")
        rates[suite] = Fraction(100 * k, n) if n else None
        missing[suite] = CLEAN_PER_SUITE - n
    B = None if None in rates.values() else sum(rates.values(), Fraction(0)) / len(rates)
    return rates, B, missing


def _lp(rec, where):
    lp = rec.get("libero_plus")
    if not isinstance(lp, dict) or sorted(lp) != sorted(EVAL_SEEDS):
        got = sorted(lp) if isinstance(lp, dict) else lp
        raise InputError(f"{where}: libero_plus eval seeds {got}, need {list(EVAL_SEEDS)}")
    out = {}
    for es in EVAL_SEEDS:
        e = lp[es]
        if not isinstance(e, dict):
            raise InputError(f"{where}: libero_plus {es} is not an object")
        k = _count(e.get("successes"), f"{where} eval seed {es} successes")
        n = _count(e.get("evaluated"), f"{where} eval seed {es} evaluated")
        m = _count(e.get("missing"), f"{where} eval seed {es} missing")
        if n + m != LP_EPISODES:
            raise InputError(f"{where}: eval seed {es}: evaluated {n} + missing {m} != {LP_EPISODES}")
        if k > n:
            raise InputError(f"{where}: eval seed {es}: successes {k} > evaluated {n}")
        out[es] = (k, n, m)
    return out


DECISIONS = ("evaluate", "skip")


def _user_decision(rec, where):
    """The user's recorded answer to the open case (PLAN line 405), or None when not given."""
    d, words = rec.get("user_decision_libero_plus"), rec.get("decided")
    if d is None and words is None:
        return None
    if d not in DECISIONS:
        raise InputError(f'{where}: "user_decision_libero_plus" must be one of {list(DECISIONS)}, got {d!r}')
    if not isinstance(words, str) or not words.strip():
        raise InputError(f'{where}: "user_decision_libero_plus" needs "decided": "<date and the user\'s words>"')
    return {"user_decision_libero_plus": d, "decided": words}


def load(obj):
    """Validate the input and return {(config, seed): checkpoint dict}, and whether extension seeds are present."""
    if not isinstance(obj, dict) or not isinstance(obj.get("checkpoints"), list):
        raise InputError('input must be {"checkpoints": [...]}')
    by_key = {}
    for i, rec in enumerate(obj["checkpoints"]):
        if not isinstance(rec, dict):
            raise InputError(f"checkpoints[{i}]: not an object")
        c, s = rec.get("config"), rec.get("train_seed")
        where = f"{c} seed {s}"
        if c not in CONFIGS:
            raise InputError(f"checkpoints[{i}]: unknown config {c!r}; the design is {list(CONFIGS)}")
        if isinstance(s, bool) or not isinstance(s, int):
            raise InputError(f"checkpoints[{i}]: train_seed must be an integer, got {s!r}")
        if s not in BASE_SEEDS and not (c in EXT_CONFIGS and s in EXT_SEEDS):
            raise InputError(f"{where}: not in the design (seeds 0-2; 3-4 only for {list(EXT_CONFIGS)})")
        if (c, s) in by_key:
            raise InputError(f"{where}: listed twice")
        by_key[(c, s)] = _checkpoint(rec, where)
    lacking = [f"{c} seed {s}" for c in CONFIGS for s in BASE_SEEDS if (c, s) not in by_key]
    if lacking:
        raise InputError("no entry for " + ", ".join(lacking) + " (a run that could not be produced is listed with "
                         '"status": "missing"; a scope cut needs a PLAN amendment and a code change)')
    ext = [(c, s) for c in EXT_CONFIGS for s in EXT_SEEDS]
    present = [k for k in ext if k in by_key]
    if present and len(present) != len(ext):
        raise InputError("extension incomplete: entries for " + ", ".join(f"{c} seed {s}" for c, s in present)
                         + "; the extension is all four runs (seeds 3 and 4 of dinov3 and dinov3+c-radiov4)")
    return by_key, bool(present)


def _checkpoint(rec, where):
    status = rec.get("status")
    if status == "missing":
        reason = rec.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise InputError(f'{where}: "status": "missing" needs a "reason"')
        return {"status": "missing", "missing_reason": reason, "checkpoint": rec.get("checkpoint")}
    if status != "ok":
        raise InputError(f'{where}: status must be "ok" or "missing", got {status!r}')
    suite_rates, B, clean_missing = _clean_B(rec, where)
    clean_missing_total = sum(clean_missing.values())
    clean_short = Fraction(clean_missing_total, CLEAN_ROLLOUTS) > MISSING_MAX
    gate_pass = None if B is None else B >= GATE
    diag = rec.get("gate_diagnosis")
    if gate_pass is False:
        if not isinstance(diag, dict) or not isinstance(diag.get("fault_found"), bool):
            raise InputError(f"{where}: B = {float(B):.2f} < 36 and no gate diagnosis "
                             '("gate_diagnosis": {"fault_found": false, "note": ...}) (item 5)')
        if diag["fault_found"]:
            raise InputError(f"{where}: the gate diagnosis found a fault; fix it by PR, re-evaluate every affected "
                             "checkpoint in every configuration, and supply the new results (item 6)")
    out = {"status": "ok", "checkpoint": rec.get("checkpoint"),
           "clean_suite_rates": suite_rates, "B": B, "gate_pass": gate_pass,
           "gate_diagnosis": diag if gate_pass is False else None,
           "clean_missing_per_suite": clean_missing, "clean_missing_total": clean_missing_total,
           "clean_missing_frac": Fraction(clean_missing_total, CLEAN_ROLLOUTS), "stop": None,
           "lp_counts": None, "lp_rates": None, "missing_per_eval_seed": None, "missing_total": None,
           "missing_frac": None}
    decision = _user_decision(rec, where)
    if decision and not clean_short:
        raise InputError(f'{where}: "user_decision_libero_plus" is given, but only {clean_missing_total} clean '
                         "rollouts are missing; the decision exists only for a checkpoint over the clean 1% line")
    out["user_decision"] = None
    reasons = []
    if clean_short:
        reasons.append(f"clean coverage: {clean_missing_total} of {CLEAN_ROLLOUTS} clean rollouts missing after "
                       "reruns, more than 1% (PLAN amendment 2026-11-17, item 3 (c))")
        # The open case: the LIBERO-Plus evaluation of such a checkpoint is the user's call, so the script neither
        # requires nor refuses it. Given: validated and reported, not used (the seed is missing either way).
        supplied = ("; LIBERO-Plus results were supplied: reported, not used" if "libero_plus" in rec
                    else "; no LIBERO-Plus results supplied")
        if decision:
            out["user_decision"] = dict(decision, note="recorded user decision on the open case (PLAN line 405)"
                                        + supplied)
        else:
            out["stop"] = STOP_CLEAN_LP + supplied
    if not clean_short or "libero_plus" in rec:
        lp = _lp(rec, where)
        missing_total = sum(m for _, _, m in lp.values())
        out.update(lp_counts=lp, lp_rates={es: Fraction(100 * k, n) if n else None for es, (k, n, m) in lp.items()},
                   missing_per_eval_seed={es: lp[es][2] for es in EVAL_SEEDS}, missing_total=missing_total,
                   missing_frac=Fraction(missing_total, LP_ROLLOUTS))
        if Fraction(missing_total, LP_ROLLOUTS) > MISSING_MAX:
            reasons.append(f"coverage: {missing_total} of {LP_ROLLOUTS} LIBERO-Plus rollouts missing after reruns, "
                           "more than 1%")
    if reasons:
        out.update(status="missing", missing_reason="; ".join(reasons))
        return out
    rates = out["lp_rates"]
    vals = list(rates.values())
    out["Y"] = sum(vals, Fraction(0)) / len(vals)
    out["eval_seed_sd"] = _sd(vals)
    return out


# ---------------------------------------------------------------------------
# Statistic
# ---------------------------------------------------------------------------

def _sd(vals):
    """Sample SD (ddof 1), float; None for fewer than 2 values."""
    if len(vals) < 2:
        return None
    m = sum(vals, Fraction(0)) / len(vals)
    return math.sqrt(sum(((v - m) ** 2 for v in vals), Fraction(0)) / (len(vals) - 1))


def seeds_of(ck, config, seeds):
    return [s for s in seeds if (config, s) in ck and ck[(config, s)]["status"] == "ok"]


def pooled(ck, seed_sets):
    """Pooled within-configuration variance over every config with at least 2 seeds (PLAN line 120).
    seed_sets: {config: seeds considered}. Returns (variance as a Fraction, degrees of freedom, per-config detail)."""
    ss, df, detail = Fraction(0), 0, {}
    for c, seeds in seed_sets.items():
        ys = [ck[(c, s)]["Y"] for s in seeds_of(ck, c, seeds)]
        if len(ys) >= 2:
            m = sum(ys, Fraction(0)) / len(ys)
            ssc = sum(((y - m) ** 2 for y in ys), Fraction(0))
            ss += ssc
            df += len(ys) - 1
            detail[c] = {"n": len(ys), "df": len(ys) - 1, "ss": float(ssc)}
        else:
            detail[c] = {"n": len(ys), "df": 0, "ss": 0.0, "note": "fewer than 2 seeds, not pooled"}
    if df == 0:
        raise InputError("sigma_hat undefined: no configuration has 2 or more seeds")
    return ss / df, df, detail


def compare(ck, a, b, seeds):
    """Paired differences d_s(a - b) on the seeds (of those given) that both configs have."""
    sa, sb = seeds_of(ck, a, seeds), seeds_of(ck, b, seeds)
    both = [s for s in seeds if s in sa and s in sb]
    d = {s: ck[(a, s)]["Y"] - ck[(b, s)]["Y"] for s in both}
    dbar = sum(d.values(), Fraction(0)) / len(d) if d else None
    return {"a": a, "b": b, "seeds_required": list(seeds), "paired_seeds": both, "n": len(both),
            "complete": both == list(seeds), "d_s": d, "dbar": dbar}


def k_sq(V, n):
    """(f(n) x sigma_hat)^2 = 8 V / n, exact."""
    return Fraction(8) * V / n


def t3(V, n):
    """T3 = max(2.7, f(n) x sigma_hat), as a float for the report."""
    return max(float(FLOOR), math.sqrt(k_sq(V, n)))


def le_minus_t3(x, V, n):
    """x <= -max(2.7, sqrt(k)), exact."""
    return x <= -FLOOR and x * x >= k_sq(V, n)


def ge_t3(x, V, n):
    """x >= max(2.7, sqrt(k)), exact."""
    return x >= FLOOR and x * x >= k_sq(V, n)


def m2(x, V, n):
    """M2: x - f(n) x sigma_hat > -2.7, i.e. x + 2.7 > sqrt(k), exact."""
    y = x + FLOOR
    return y > 0 and y * y > k_sq(V, n)


def harm_like(cmp, V, min_negative=None):
    """The H test (item 4) on cmp: sign condition and dbar <= -T3. min_negative None = every paired d_s < 0
    (three seeds); an integer = at least that many negative (S-4of5 after the extension), defined only when all
    required seeds are paired. Computed on the paired seeds; returns (result or None if not computable, T used)."""
    n = cmp["n"]
    if n == 0:
        return None, None
    neg = sum(v < 0 for v in cmp["d_s"].values())
    if min_negative is None:
        sign = neg == n
    elif cmp["complete"]:
        sign = neg >= min_negative
    else:
        return None, t3(V, n)     # S-4of5 is defined for five seeds only
    return sign and le_minus_t3(cmp["dbar"], V, n), t3(V, n)


def _test_out(cmp, result, established, threshold=None, margin=None, extra=None):
    out = {"a": cmp["a"], "b": cmp["b"], "seeds_required": cmp["seeds_required"], "paired_seeds": cmp["paired_seeds"],
           "n": cmp["n"], "d_s_pts": {str(s): float(v) for s, v in cmp["d_s"].items()},
           "d_s_exact": {str(s): str(v) for s, v in cmp["d_s"].items()},
           "dbar_pts": None if cmp["dbar"] is None else float(cmp["dbar"]),
           "dbar_exact": None if cmp["dbar"] is None else str(cmp["dbar"]),
           "result_on_paired_seeds": result, "established": established}
    if threshold is not None:
        out["threshold_T_pts"] = threshold
    if margin is not None:
        out["margin"] = margin
    if extra:
        out.update(extra)
    return out


def _m2_report(cmp, V):
    n = cmp["n"]
    if n == 0:
        return None, None
    return m2(cmp["dbar"], V, n), {"rule": "M2: dbar - f(n) x sigma_hat > -2.7", "f_n": f_of(n),
                                   "f_n_sigma_hat_pts": math.sqrt(k_sq(V, n)),
                                   "lower_bound_pts": float(cmp["dbar"]) - math.sqrt(k_sq(V, n))}


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

def analyse(obj):
    ck, has_ext = load(obj)
    base = {c: BASE_SEEDS for c in CONFIGS}
    V3, df3, pool3 = pooled(ck, base)

    # C (calibration), item 4, under T3; C-a: required.
    c_cmp = compare(ck, D, NONE, BASE_SEEDS)
    c_res = None
    if c_cmp["n"]:
        c_res = all(v > 0 for v in c_cmp["d_s"].values()) and ge_t3(c_cmp["dbar"], V3, c_cmp["n"])
    c_holds = bool(c_cmp["complete"] and c_res)
    C = _test_out(c_cmp, c_res, c_cmp["complete"], t3(V3, c_cmp["n"]) if c_cmp["n"] else None)
    C["holds"] = c_holds

    # H / N / I on seeds 0-2.
    h_cmp = compare(ck, DR, D, BASE_SEEDS)
    h_res, h_T = harm_like(h_cmp, V3)
    n_res, n_m = _m2_report(h_cmp, V3)
    if not h_cmp["complete"]:
        harm3 = "I"
    elif h_res:
        harm3 = "H"
    elif n_res:
        harm3 = "N"
    else:
        harm3 = "I"
    H3 = _test_out(h_cmp, h_res, h_cmp["complete"], h_T, extra={"sign_condition": "every d_s < 0"})
    N3 = _test_out(h_cmp, None if n_res is None else (n_res and not h_res), h_cmp["complete"], margin=n_m)

    # H_L and K on seeds 0-2 (helpful pairing).
    l_cmp = compare(ck, DL, D, BASE_SEEDS)
    hl_res, hl_T = harm_like(l_cmp, V3)
    k_raw, k_m = _m2_report(l_cmp, V3)
    if not l_cmp["complete"]:
        helpful = "unresolved"
    elif hl_res:
        helpful = "H_L"
    elif k_raw:
        helpful = "K"
    else:
        helpful = "unresolved"
    HL = _test_out(l_cmp, hl_res, l_cmp["complete"], hl_T, extra={"sign_condition": "every d_s < 0"})
    K = _test_out(l_cmp, None if k_raw is None else (k_raw and not hl_res), l_cmp["complete"], margin=k_m)

    # Below both members (secondary; c-radiov4 alone is trained). Seeds 0-2 and the three-seed sigma_hat, T and
    # f(3), also after an extension (PLAN amendment 2026-11-17, item 3 (b)).
    r_cmp = compare(ck, DR, R_, BASE_SEEDS)
    hr_res, hr_T = harm_like(r_cmp, V3)
    HR = _test_out(r_cmp, hr_res, r_cmp["complete"], hr_T, extra={"sign_condition": "every d_s < 0"})

    # Extension status (item 7; under C fails, PLAN amendment 2026-11-17, item 3 (a)).
    if harm3 != "I":
        ext_status = EXT_NA
    elif not h_cmp["complete"]:
        ext_status = EXT_MISSING
    elif not c_holds:
        ext_status = EXT_C_FAILS
    else:
        ext_status = EXT_DUE

    harm, stage5 = harm3, None
    if has_ext:
        if ext_status != EXT_DUE:
            raise InputError(f"extension seeds supplied, but the three-seed result does not qualify ({ext_status})")
        seeds5 = BASE_SEEDS + EXT_SEEDS
        V5, df5, pool5 = pooled(ck, {c: (seeds5 if c in EXT_CONFIGS else BASE_SEEDS) for c in CONFIGS})
        h5 = compare(ck, DR, D, seeds5)
        h5_res, h5_T = harm_like(h5, V5, EXT_MIN_NEGATIVE)
        n5_res, n5_m = _m2_report(h5, V5)
        if not h5["complete"]:
            harm = "I"
        elif h5_res:
            harm = "H"
        elif n5_res:
            harm = "N"
        else:
            harm = "I"
        stage5 = {"sigma_hat_pts": math.sqrt(V5), "pooled_variance_exact": str(V5), "df": df5, "pool": pool5,
                  "H": _test_out(h5, h5_res, h5["complete"], h5_T,
                                 extra={"sign_condition": "S-4of5: at least 4 of 5 d_s < 0",
                                        "negative_count": sum(v < 0 for v in h5["d_s"].values())}),
                  "N": _test_out(h5, None if n5_res is None else (n5_res and not h5_res), h5["complete"],
                                 margin=n5_m),
                  "harm_outcome": harm}
        ext_status = EXT_DONE

    label = C_FAILS if not c_holds else ("I" if harm == "I" else LABELS[(harm, helpful)])
    below_both = bool(harm == "H" and hr_res and r_cmp["complete"])
    stops = [{"config": c, "train_seed": s, "stop": r["stop"]} for (c, s), r in _sorted(ck) if r.get("stop")]
    decisions = [dict(r["user_decision"], config=c, train_seed=s) for (c, s), r in _sorted(ck)
                 if r.get("user_decision")]
    return {"rule": "PLAN 2026-11-15-pretrained-start-pairings, Decision rule as signed off: T3, M2, C-a, G1 at 36, "
                    "coverage 1%, extension S-4of5 for a complete three-seed I",
            "outcome": label,
            "harm_outcome": harm if c_holds else NOT_CLAIMED,
            "harm_outcome_three_seeds": harm3 if c_holds else NOT_CLAIMED,
            "harm_test_as_computed_not_claimed": None if c_holds else {"final": harm, "three_seeds": harm3},
            "helpful_pairing": helpful if c_holds else NOT_CLAIMED,
            "helpful_pairing_as_computed_not_claimed": None if c_holds else helpful,
            "calibration_C_holds": c_holds,
            "below_both_members": below_both if c_holds else NOT_CLAIMED,
            "below_both_members_basis": NOT_CLAIMED if not c_holds else (BELOW_BOTH_5 if has_ext else BELOW_BOTH_3),
            "extension": ext_status,
            "stops_for_user": stops,
            "user_decisions": decisions,
            "sigma_hat_three_seeds_pts": math.sqrt(V3), "pooled_variance_three_seeds_exact": str(V3),
            "df_three_seeds": df3, "pool_three_seeds": pool3,
            "tests_three_seeds": {"C": C, "H": H3, "N": N3, "H_L": HL, "K": K, "H_vs_c-radiov4": HR},
            "extension_stage": stage5,
            "configs": _config_summary(ck, has_ext),
            "checkpoints": _ck_out(ck)}


def _config_summary(ck, has_ext):
    out = {}
    for c in CONFIGS:
        seeds = BASE_SEEDS + EXT_SEEDS if (has_ext and c in EXT_CONFIGS) else BASE_SEEDS
        ok = seeds_of(ck, c, seeds)
        ys = [ck[(c, s)]["Y"] for s in ok]
        out[c] = {"seeds_present": ok, "seeds_missing": [s for s in seeds if s not in ok],
                  "Ybar_pts": float(sum(ys, Fraction(0)) / len(ys)) if ys else None,
                  "sd_over_train_seeds_pts": _sd(ys)}
        if has_ext and c in EXT_CONFIGS:
            ys3 = [ck[(c, s)]["Y"] for s in seeds_of(ck, c, BASE_SEEDS)]
            out[c]["Ybar_seeds_0_2_pts"] = float(sum(ys3, Fraction(0)) / len(ys3)) if ys3 else None
    return out


def _sorted(ck):
    return sorted(ck.items(), key=lambda kv: (CONFIGS.index(kv[0][0]), kv[0][1]))


def _ck_out(ck):
    out = []
    for (c, s), r in _sorted(ck):
        e = {"config": c, "train_seed": s, "status": r["status"], "checkpoint": r.get("checkpoint")}
        if "missing_reason" in r:
            e["missing_reason"] = r["missing_reason"]
        if "clean_suite_rates" in r:
            fl = (lambda v: None if v is None else float(v))
            e.update(B_pts=fl(r["B"]), gate_G1_pass=r["gate_pass"], gate_diagnosis=r["gate_diagnosis"],
                     clean_suite_rates_pts={k: fl(v) for k, v in r["clean_suite_rates"].items()},
                     clean_missing_per_suite=r["clean_missing_per_suite"],
                     clean_missing_total=r["clean_missing_total"],
                     clean_missing_pct=float(r["clean_missing_frac"] * 100), stop_for_user=r["stop"],
                     user_decision=r["user_decision"],
                     lp_rates_pts=None if r["lp_rates"] is None else {es: fl(v) for es, v in r["lp_rates"].items()},
                     lp_missing_per_eval_seed=r["missing_per_eval_seed"], lp_missing_total=r["missing_total"],
                     lp_missing_pct=None if r["missing_frac"] is None else float(r["missing_frac"] * 100))
        if "Y" in r:
            e.update(Y_pts=float(r["Y"]), Y_exact=str(r["Y"]), eval_seed_sd_pts=r["eval_seed_sd"])
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def report(res):
    lines = [f"OUTCOME: {res['outcome']}",
             f"  harm test: {res['harm_outcome']} (three seeds: {res['harm_outcome_three_seeds']}); helpful pairing: "
             f"{res['helpful_pairing']}; C holds: {res['calibration_C_holds']}; below both members: "
             f"{res['below_both_members']}",
             f"  below both members basis: {res['below_both_members_basis']}",
             f"  extension: {res['extension']}"]
    if not res["calibration_C_holds"]:
        lines[1] = (f"  harm test: {NOT_CLAIMED}; helpful pairing: {NOT_CLAIMED}; below both members: "
                    f"{NOT_CLAIMED}; C holds: False (PLAN line 192: the comparisons below are reported, nothing is "
                    f"claimed)")
    for p in res["stops_for_user"]:
        lines.append(f"  STOP FOR THE USER: {p['config']} seed {p['train_seed']}: {p['stop']}")
    for p in res["user_decisions"]:
        lines.append(f"  USER DECISION (recorded): {p['config']} seed {p['train_seed']}: LIBERO-Plus evaluation "
                     f"'{p['user_decision_libero_plus']}', decided: {p['decided']} ({p['note']})")
    lines.append(f"  sigma_hat (seeds 0-2): {_fmt(res['sigma_hat_three_seeds_pts'])} pts, df {res['df_three_seeds']}")
    if res["extension_stage"]:
        e = res["extension_stage"]
        lines.append(f"  sigma_hat (re-pooled, extension): {_fmt(e['sigma_hat_pts'])} pts, df {e['df']}")
    lines.append("checkpoints (Y = mean of the three LIBERO-Plus rates; B = clean four-suite average):")
    for r in res["checkpoints"]:
        if "Y_pts" in r:
            lines.append(f"  {r['config']:<18} s{r['train_seed']}  Y {_fmt(r['Y_pts'])}  eval-seed sd "
                         f"{_fmt(r['eval_seed_sd_pts'])}  missing {r['lp_missing_total']}  B {_fmt(r['B_pts'], 2)}"
                         f"  clean missing {r['clean_missing_total']}"
                         f"  gate {'pass' if r['gate_G1_pass'] else 'FAIL (no fault found)'}")
        else:
            lines.append(f"  {r['config']:<18} s{r['train_seed']}  MISSING: {r['missing_reason']}")
    tests = [("seeds 0-2", res["tests_three_seeds"])]
    if res["extension_stage"]:
        tests.append(("extension, seeds 0-4", {k: res["extension_stage"][k] for k in ("H", "N")}))
    for stage, ts in tests:
        lines.append(f"tests ({stage}):")
        for name, t in ts.items():
            d = ", ".join(f"s{s} {v:+.3f}" for s, v in t["d_s_pts"].items())
            if "threshold_T_pts" in t:
                thr = f"T {_fmt(t['threshold_T_pts'])}"
            elif t.get("margin"):
                thr = f"dbar - f(n) sigma_hat {_fmt(t['margin']['lower_bound_pts'])} vs -2.7"
            else:
                thr = "threshold n/a"
            lines.append(f"  {name:<14} {t['a']} - {t['b']}: d_s [{d}]  dbar {_fmt(t['dbar_pts'])}  n {t['n']}  "
                         f"{thr}  result {t['result_on_paired_seeds']}  established {t['established']}")
    return "\n".join(lines)


def _jsonable(x):
    if isinstance(x, Fraction):
        return float(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


# ---------------------------------------------------------------------------
# One-seed report (PLAN amendment 2026-11-18, item 3). Selected only by the --one-seed flag.
# ---------------------------------------------------------------------------

ONE_SEED = 0
ONE_SEED_BASIS = "one training seed; training-seed noise not measured on this model"
EVAL_SD_NOTE = "the SD across evaluation seeds is shown and is not training-seed noise"
SAME, OPPOSITE, ZERO = "same direction as the paper", "opposite direction", "no difference"
NO_CHECKPOINT = "no checkpoint"
NO_LP = "no LIBERO-Plus result"
ST_OK, ST_COVERAGE = "ok", "counts as missing on coverage"
# The paper's LIBERO-Plus totals (PLAN amendment 2026-11-18, item 3: Table 1 p6 and Table 10 p23 of the paper).
PAPER_TOTAL = {NONE: Fraction("74.7"), D: Fraction("84.0"), L_: Fraction("83.1"), R_: Fraction("83.5"),
               DL: Fraction("85.83"), DR: Fraction("78.59")}
ONE_SEED_COMPARISONS = ((D, NONE), (L_, NONE), (R_, NONE), (DL, D), (DL, L_), (DR, D), (DR, R_))
CALIBRATION = (D, NONE)
PATTERN = ((DR, D), (DR, R_), (DL, D))       # the paper's composition pattern: three signs together
RULE_TESTS = ("C", "H", "N", "I", "H_L", "K", "below both members")
TESTS_NOT_COMPUTED = ("not computed: needs paired training seeds and sigma_hat, and this pass has one training seed "
                      "per configuration (PLAN amendment 2026-11-18, item 3)")
NOT_IN_PLAY = "not in play: they apply to a three-seed result only (PLAN amendment 2026-11-18, item 3)"
DEFERRED = ("train seeds 1 and 2 of every configuration, none included, are deferred: not trained, not tested, and "
            "not entered in the input (PLAN amendment 2026-11-18, item 3)")
STOP_CLEAN_LP_ONE = ("more than 1% of the 2,000 clean rollouts missing after reruns: counts as missing on clean "
                     "coverage (PLAN amendment 2026-11-17, item 3 (c)). Whether this checkpoint still gets its "
                     "LIBERO-Plus evaluation is not decided by the user (PLAN line 405): stop and ask the user")


def _dec(x, signed=True):
    """An exact decimal Fraction (the paper's values) as written, e.g. +9.3 or -5.41."""
    from decimal import Decimal
    s = format(Decimal(x.numerator) / Decimal(x.denominator), "f")
    return ("+" + s) if signed and x > 0 else s


def _pts(x, nd=2, signed=False):
    if x is None:
        return "n/a"
    return f"{float(x):+.{nd}f}" if signed else f"{float(x):.{nd}f}"


def _lp_categories(rec, where, lp_counts):
    """Per-category LIBERO-Plus counts per eval seed: {"41": {"<category>": {"successes": k, "evaluated": n}, ...}}.
    The categories are the same under every eval seed and add up to that eval seed's totals."""
    cats = rec.get("libero_plus_categories")
    if not isinstance(cats, dict) or sorted(cats) != sorted(EVAL_SEEDS):
        got = sorted(cats) if isinstance(cats, dict) else cats
        raise InputError(f"{where}: libero_plus_categories eval seeds {got}, need {list(EVAL_SEEDS)} (the one-seed "
                         "report gives the LIBERO-Plus per-category rates)")
    names, per_seed = None, {}
    for es in EVAL_SEEDS:
        d = cats[es]
        if not isinstance(d, dict) or not d:
            raise InputError(f"{where}: libero_plus_categories {es} is not a non-empty object")
        if names is None:
            names = sorted(d)
        elif sorted(d) != names:
            raise InputError(f"{where}: libero_plus_categories {es} has categories {sorted(d)}, eval seed "
                             f"{EVAL_SEEDS[0]} has {names}")
        rates, tk, tn = {}, 0, 0
        for g in names:
            e = d[g]
            if not isinstance(e, dict):
                raise InputError(f"{where}: libero_plus_categories {es} {g} is not an object")
            k = _count(e.get("successes"), f"{where} eval seed {es} category {g} successes")
            n = _count(e.get("evaluated"), f"{where} eval seed {es} category {g} evaluated")
            if k > n:
                raise InputError(f"{where}: eval seed {es} category {g}: successes {k} > evaluated {n}")
            rates[g] = Fraction(100 * k, n) if n else None
            tk, tn = tk + k, tn + n
        k_all, n_all, _ = lp_counts[es]
        if (tk, tn) != (k_all, n_all):
            raise InputError(f"{where}: eval seed {es}: categories add up to {tk} successes of {tn} evaluated, the "
                             f"eval seed's totals are {k_all} of {n_all}")
        per_seed[es] = rates
    mean = {g: (None if any(per_seed[es][g] is None for es in EVAL_SEEDS)
                else sum((per_seed[es][g] for es in EVAL_SEEDS), Fraction(0)) / len(EVAL_SEEDS)) for g in names}
    return {"per_eval_seed": per_seed, "mean_over_eval_seeds": mean}


def load_one_seed(obj):
    """Validate a one-seed input: exactly one entry per configuration, all at train seed 0. Returns {config: dict}."""
    if not isinstance(obj, dict) or not isinstance(obj.get("checkpoints"), list):
        raise InputError('input must be {"checkpoints": [...]}')
    by_config = {}
    for i, rec in enumerate(obj["checkpoints"]):     # the design check first, before any entry's details
        s = rec.get("train_seed") if isinstance(rec, dict) else None
        if isinstance(s, int) and not isinstance(s, bool) and s != ONE_SEED:
            raise InputError(f"{rec.get('config')} seed {s}: the one-seed report (--one-seed) takes train seed 0 "
                             "only; seeds 1 and 2 are deferred (PLAN amendment 2026-11-18, item 3). A three-seed "
                             "input is analysed by the locked rule, without --one-seed; an input with some seeds "
                             'added needs the "not tested" state by PR first (item 6)')
    for i, rec in enumerate(obj["checkpoints"]):
        if not isinstance(rec, dict):
            raise InputError(f"checkpoints[{i}]: not an object")
        c, s = rec.get("config"), rec.get("train_seed")
        where = f"{c} seed {s}"
        if c not in CONFIGS:
            raise InputError(f"checkpoints[{i}]: unknown config {c!r}; the design is {list(CONFIGS)}")
        if isinstance(s, bool) or not isinstance(s, int):
            raise InputError(f"checkpoints[{i}]: train_seed must be an integer, got {s!r}")
        if s != ONE_SEED:
            raise InputError(f"{where}: the one-seed report (--one-seed) takes train seed 0 only; seeds 1 and 2 are "
                             "deferred (PLAN amendment 2026-11-18, item 3). A three-seed input is analysed by the "
                             "locked rule, without --one-seed; an input with some seeds added needs the "
                             '"not tested" state by PR first (item 6)')
        if c in by_config:
            raise InputError(f"{where}: listed twice")
        by_config[c] = _one_checkpoint(rec, where)
    lacking = [f"{c} seed 0" for c in CONFIGS if c not in by_config]
    if lacking:
        raise InputError("no entry for " + ", ".join(lacking) + " (the one-seed report needs seed 0 of all six "
                         'configurations; a training run that still fails after its one rerun is listed with "status":'
                         ' "missing" and a "reason", and is reported as "no checkpoint")')
    return by_config


def _one_checkpoint(rec, where):
    if rec.get("status") == "missing":
        reason = rec.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise InputError(f'{where}: "status": "missing" needs a "reason"')
        return {"status": NO_CHECKPOINT, "reason": reason, "checkpoint": rec.get("checkpoint")}
    if "libero_plus_categories" in rec and "libero_plus" not in rec:
        raise InputError(f"{where}: libero_plus_categories given without libero_plus")
    r = _checkpoint(rec, where)            # validation, B, gate, coverage and the open case, as in the locked rule
    cats = _lp_categories(rec, where, r["lp_counts"]) if r["lp_counts"] is not None else None
    rates = r["lp_rates"]
    Y = None
    if rates is not None and None not in rates.values():
        Y = sum(rates.values(), Fraction(0)) / len(rates)
    clean_short = r["clean_missing_frac"] > MISSING_MAX
    lp_short = r["missing_frac"] is not None and r["missing_frac"] > MISSING_MAX
    reasons = []
    if clean_short:
        reasons.append(f"clean coverage: {r['clean_missing_total']} of {CLEAN_ROLLOUTS} clean rollouts missing after "
                       "reruns, more than 1%")
    if lp_short:
        reasons.append(f"LIBERO-Plus coverage: {r['missing_total']} of {LP_ROLLOUTS} rollouts missing after reruns, "
                       "more than 1%")
    return {"status": ST_COVERAGE if reasons else ST_OK, "coverage_reason": "; ".join(reasons) or None,
            "checkpoint": r["checkpoint"], "B": r["B"], "clean_suite_rates": r["clean_suite_rates"],
            "gate_pass": r["gate_pass"], "gate_diagnosis": r["gate_diagnosis"],
            "clean_missing_per_suite": r["clean_missing_per_suite"], "clean_missing_total": r["clean_missing_total"],
            "clean_missing_frac": r["clean_missing_frac"], "lp_rates": rates,
            "missing_per_eval_seed": r["missing_per_eval_seed"], "missing_total": r["missing_total"],
            "missing_frac": r["missing_frac"], "Y": Y,
            "eval_seed_sd": None if Y is None else _sd(list(rates.values())), "categories": cats,
            "stop": STOP_CLEAN_LP_ONE + ("; LIBERO-Plus results were supplied" if "libero_plus" in rec else
                                         "; no LIBERO-Plus results supplied") if r["stop"] else None,
            "user_decision": r["user_decision"]}


def _one_comparison(cks, a, b):
    ra, rb = cks[a], cks[b]
    paper = PAPER_TOTAL[a] - PAPER_TOTAL[b]
    out = {"comparison": f"{a} - {b}", "a": a, "b": b, "paper_difference_pts": float(paper),
           "paper_difference": _dec(paper), "basis": ONE_SEED_BASIS, "d_0_pts": None, "d_0_exact": None,
           "d_per_eval_seed_pts": None, "label": None, "no_label_reason": None, "value_text": None}
    absent = [c for c, r in ((a, ra), (b, rb)) if r["status"] == NO_CHECKPOINT]
    if absent:
        out.update(value_text=NO_CHECKPOINT, no_label_reason=NO_CHECKPOINT + ": " + ", ".join(absent))
        return out
    no_y = [c for c, r in ((a, ra), (b, rb)) if r["Y"] is None]
    if no_y:
        out.update(value_text=NO_LP, no_label_reason=NO_LP + ": " + ", ".join(no_y))
        return out
    d0 = ra["Y"] - rb["Y"]
    per = {es: (None if ra["lp_rates"][es] is None or rb["lp_rates"][es] is None
                else float(ra["lp_rates"][es] - rb["lp_rates"][es])) for es in EVAL_SEEDS}
    out.update(d_0_pts=float(d0), d_0_exact=str(d0), d_per_eval_seed_pts=per,
               value_text=f"d_0 {_pts(d0, signed=True)}")
    short = [c for c, r in ((a, ra), (b, rb)) if r["status"] == ST_COVERAGE]
    if short:
        out["no_label_reason"] = ", ".join(f"{c} seed 0 {ST_COVERAGE}" for c in short)
    elif d0 == 0:
        out["label"] = ZERO
    else:
        out["label"] = SAME if (d0 > 0) == (paper > 0) else OPPOSITE
    return out


def _overall(cmps):
    by = {(c["a"], c["b"]): c for c in cmps}
    vals = "; ".join(f"{by[k]['comparison']}: {by[k]['value_text']}, paper {by[k]['paper_difference']}"
                     for k in PATTERN)
    cal = by[CALIBRATION]
    pat = [by[k] for k in PATTERN]
    differing = []
    if cal["label"] is None:
        state = "not read"
        line = f"not read: the DINOv3-against-no-encoder comparison has no label ({vals})"
    elif cal["label"] != SAME:
        state = "not read"
        line = f"not read: DINOv3 alone is not above the no-encoder model at one training seed ({vals})"
    elif any(c["label"] is None for c in pat):
        state = "not available"
        line = ("not available: no label for " + ", ".join(c["comparison"] for c in pat if c["label"] is None)
                + f" ({vals})")
    elif all(c["label"] == SAME for c in pat):
        state = "appears"
        line = f"the paper's pattern appears at one training seed ({vals})"
    else:
        state = "does not appear"
        differing = [c["comparison"] for c in pat if c["label"] != SAME]
        line = ("the paper's pattern does not appear at one training seed, differing: "
                + ", ".join(f"{c['comparison']} ({c['label']})" for c in pat if c["label"] != SAME) + f" ({vals})")
    return {"line": line, "state": state, "differing": differing, "basis": ONE_SEED_BASIS,
            "comparisons": [by[k]["comparison"] for k in PATTERN],
            "read_only_when": "dinov3 - none is labelled " + repr(SAME) + " (calibration first, option C-a)"}


def analyse_one_seed(obj):
    cks = load_one_seed(obj)
    cmps = [_one_comparison(cks, a, b) for a, b in ONE_SEED_COMPARISONS]
    counts = {SAME: 0, OPPOSITE: 0, ZERO: 0, "no label": 0}
    for c in cmps:
        counts[c["label"] or "no label"] += 1
    fl = (lambda v: None if v is None else float(v))
    ck_out = []
    for c in CONFIGS:
        r = cks[c]
        e = {"config": c, "train_seed": ONE_SEED, "status": r["status"], "checkpoint": r.get("checkpoint"),
             "basis": ONE_SEED_BASIS}
        if r["status"] == NO_CHECKPOINT:
            e["reason"] = r["reason"]
        else:
            cats = r["categories"]
            e.update(coverage_reason=r["coverage_reason"], B_pts=fl(r["B"]),
                     clean_suite_rates_pts={k: fl(v) for k, v in r["clean_suite_rates"].items()},
                     gate_G1_pass=r["gate_pass"], gate_diagnosis=r["gate_diagnosis"],
                     Y_pts=fl(r["Y"]), Y_exact=None if r["Y"] is None else str(r["Y"]),
                     eval_seed_sd_pts=r["eval_seed_sd"], eval_seed_sd_note=EVAL_SD_NOTE,
                     lp_rates_pts=None if r["lp_rates"] is None else {k: fl(v) for k, v in r["lp_rates"].items()},
                     lp_per_category_pts=None if cats is None else {
                         "mean_over_eval_seeds": {g: fl(v) for g, v in cats["mean_over_eval_seeds"].items()},
                         "per_eval_seed": {es: {g: fl(v) for g, v in d.items()}
                                           for es, d in cats["per_eval_seed"].items()}},
                     coverage={"clean_missing_per_suite": r["clean_missing_per_suite"],
                               "clean_missing_total": r["clean_missing_total"], "clean_rollouts": CLEAN_ROLLOUTS,
                               "clean_missing_pct": float(r["clean_missing_frac"] * 100),
                               "lp_missing_per_eval_seed": r["missing_per_eval_seed"],
                               "lp_missing_total": r["missing_total"], "lp_rollouts": LP_ROLLOUTS,
                               "lp_missing_pct": fl(None if r["missing_frac"] is None else r["missing_frac"] * 100)},
                     stop_for_user=r["stop"], user_decision=r["user_decision"])
        ck_out.append(e)
    stops = [{"config": c, "train_seed": ONE_SEED, "stop": cks[c]["stop"]} for c in CONFIGS if cks[c].get("stop")]
    decisions = [dict(cks[c]["user_decision"], config=c, train_seed=ONE_SEED) for c in CONFIGS
                 if cks[c].get("user_decision")]
    return {"report": "one-seed report, PLAN 2026-11-15-pretrained-start-pairings, amendment 2026-11-18, item 3",
            "basis": ONE_SEED_BASIS, "train_seeds": [ONE_SEED], "deferred_seeds": DEFERRED,
            "overall": _overall(cmps), "label_counts": counts, "comparisons": cmps, "checkpoints": ck_out,
            "decision_rule_tests": {t: TESTS_NOT_COMPUTED for t in RULE_TESTS},
            "outcome_branches_item_8_and_extension_item_7": NOT_IN_PLAY,
            "stops_for_user": stops, "user_decisions": decisions}


def report_one_seed(res):
    lines = [f"ONE-SEED REPORT (PLAN amendment 2026-11-18, item 3): {ONE_SEED_BASIS}",
             f"  {DEFERRED}",
             f"  OVERALL: {res['overall']['line']}",
             "  labels: " + ", ".join(f"{k} {v}" for k, v in res["label_counts"].items()) + " (of 7 comparisons)"]
    for p in res["stops_for_user"]:
        lines.append(f"  STOP FOR THE USER: {p['config']} seed 0: {p['stop']}")
    for p in res["user_decisions"]:
        lines.append(f"  USER DECISION (recorded): {p['config']} seed 0: LIBERO-Plus evaluation "
                     f"'{p['user_decision_libero_plus']}', decided: {p['decided']}")
    lines.append(f"comparisons, d_0 = Y(a, 0) - Y(b, 0), paper's difference beside it ({ONE_SEED_BASIS}):")
    for c in res["comparisons"]:
        per = ("" if c["d_per_eval_seed_pts"] is None else "  per eval seed " + ", ".join(
            f"{es} {_pts(v, signed=True)}" for es, v in c["d_per_eval_seed_pts"].items()))
        lab = c["label"] if c["label"] else f"no label ({c['no_label_reason']})"
        lines.append(f"  {c['comparison']:<30} {c['value_text']:<16} paper {c['paper_difference']:<6}{per}  "
                     f"label: {lab}")
    lines.append(f"checkpoints, train seed 0 ({ONE_SEED_BASIS}; {EVAL_SD_NOTE}):")
    for e in res["checkpoints"]:
        if e["status"] == NO_CHECKPOINT:
            lines.append(f"  {e['config']:<18} {NO_CHECKPOINT}: {e['reason']}")
            continue
        cov = e["coverage"]
        gate = ("n/a" if e["gate_G1_pass"] is None else "pass" if e["gate_G1_pass"]
                else "below 36, diagnosis found no fault (item 5)")
        lp = ("LIBERO-Plus: none" if e["lp_rates_pts"] is None else
              f"Y {_pts(e['Y_pts'], 3)}  eval-seed SD {_pts(e['eval_seed_sd_pts'], 3)}  rates "
              + ", ".join(f"{es} {_pts(v, 3)}" for es, v in e["lp_rates_pts"].items())
              + f"  LIBERO-Plus missing {cov['lp_missing_total']} of {cov['lp_rollouts']}")
        lines.append(f"  {e['config']:<18} status {e['status']}  B {_pts(e['B_pts'])}  clean missing "
                     f"{cov['clean_missing_total']} of {cov['clean_rollouts']}  gate G1 {gate}  {lp}")
        lines.append("      clean suites: " + ", ".join(f"{k} {_pts(v)}" for k, v in
                                                       e["clean_suite_rates_pts"].items()))
        if e["lp_per_category_pts"]:
            lines.append("      per category (mean over eval seeds): " + ", ".join(
                f"{g} {_pts(v)}" for g, v in e["lp_per_category_pts"]["mean_over_eval_seeds"].items()))
        if e["coverage_reason"]:
            lines.append(f"      {ST_COVERAGE}: {e['coverage_reason']}")
    lines.append(f"decision rule tests ({', '.join(RULE_TESTS)}): {TESTS_NOT_COMPUTED}")
    lines.append(f"outcome branches (item 8) and extension (item 7): {NOT_IN_PLAY}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description="Stage 2 analysis: the locked decision rule (see the module docstring).")
    p.add_argument("--input", required=True, help="JSON of per-checkpoint totals (format in this file's docstring)")
    p.add_argument("--out", default=None, help="write the full analysis as JSON")
    p.add_argument("--one-seed", action="store_true",
                   help="the one-seed report of PLAN amendment 2026-11-18, item 3 (train seed 0 only, no verdict)")
    args = p.parse_args(argv)
    if args.one_seed:
        try:
            res = analyse_one_seed(json.loads(Path(args.input).read_text()))
        except InputError as e:
            print(f"INPUT ERROR (no report): {e}", file=sys.stderr)
            return 2
        print(report_one_seed(res))
        if args.out:
            Path(args.out).write_text(json.dumps(_jsonable(res), indent=1))
        return 3 if res["stops_for_user"] else 0
    try:
        res = analyse(json.loads(Path(args.input).read_text()))
    except InputError as e:
        print(f"INPUT ERROR (no outcome): {e}", file=sys.stderr)
        return 2
    print(report(res))
    if args.out:
        Path(args.out).write_text(json.dumps(_jsonable(res), indent=1))
    return 3 if res["stops_for_user"] else 0


if __name__ == "__main__":
    sys.exit(main())
