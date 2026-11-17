"""Stage 2 analysis: the locked decision rule of experiments/2026-11-15-pretrained-start-pairings/PLAN.md.

This file implements the PLAN's "Comparison statistic (defined before any result)" (PLAN lines 115-123) and its
"Decision rule" (lines 151-211) with the options the user signed off in the "Amendments" (lines 311 onward):
threshold T3 = max(2.7, f(n) x sigma_hat), margins M2, calibration C-a (required), clean gate G1 at 36, coverage 1%,
SigLIP 2 alone and C-RADIOv4 alone trained (Open item b = yes), and the one-time extension (seeds 3 and 4 of dinov3
and dinov3+c-radiov4) with sign condition S-4of5, for a complete three-seed I only. It never changes the rule. Any
other design (a scope cut, other options) is a dated PLAN amendment first and a code change second.

Run (stdlib only):
  /opt/homebrew/bin/python3.11 -I pairings_analysis.py --input stage2-totals.json --out stage2-analysis.json
Exit 0: outcome computed (printed, and written to --out). Exit 2: the input is incomplete or malformed; nothing is
written and no outcome is given.

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
                   "42": {...}, "43": {...}}}
  {"config": "dinov3+c-radiov4", "train_seed": 2, "status": "missing", "reason": "non-finite loss, rerun failed"}
"evaluated" counts the LIBERO-Plus episodes scored under that eval seed after reruns, "missing" those still missing
(rollout.py's errors file); evaluated + missing must be 10,030. Clean: the 2,000 clean rollouts, 500 per suite
(50 per task, 10 tasks), eval seed 41; the counts come from the stage 1 tally (tally_proxy_clean.py's per-suite
counts or tally_per_task.py).

Statistic (PLAN lines 116-122): rate = successes / evaluated per eval seed, Y(c, s) = mean of the three rates, in
points. A checkpoint with more than 1% of its 30,090 LIBERO-Plus rollouts missing counts as a missing seed.
B(c, s) = mean of the four suite rates. d_s(A - B) = Y(A, s) - Y(B, s) on the seeds both have; dbar their mean.
sigma_hat: pooled within-configuration SD over every configuration with at least 2 seeds. f(n) = 2 x sqrt(2 / n).
All comparisons are exact (fractions; the square root is compared by squaring), so values on a boundary land on the
side the rule states: dbar <= -T, dbar >= T, and the strict ">" of M2.

Output: per-checkpoint rates, Y, eval-seed SD, missing counts, B and gate; per-config Ybar and SD; sigma_hat; per test
the paired seeds, d_s, dbar, n, the threshold used and the result; the extension status; and the outcome label as the
PLAN's Decision rule item 8 names it: "C fails, under C-a", "H and K", "H, not K, and H_L", "H, not K, not H_L",
"N and K", "N, not K", "I".

When C fails (C-a), nothing is claimed (PLAN line 190): the harm test and "below both members" are output as
"not claimed (C fails)"; the comparisons themselves are still reported, the harm test's computed value under the key
"harm_test_as_computed_not_claimed".

UNRESOLVED, PENDING THE USER (the PLAN does not settle these; the script keeps the behaviour below until the user
answers, and lists them in every output under "unresolved_pending_user"):
  1. The extension when the harm test is a complete three-seed I but C fails under C-a (item 7, line 187, says "if
     the result is I"; item 8, line 190, sends "C fails" to the user's options). Status "user decides"; if seeds 3 and
     4 are supplied anyway they are analysed, and the label stays "C fails, under C-a" (C stays on seeds 0 to 2).
  2. Clean LIBERO with other than 500 rollouts in a suite is refused (input error). The PLAN has a coverage rule for
     LIBERO-Plus only (line 117), none for clean LIBERO.
  3. After the extension, "below both members" pairs the five-seed harm result with the three-seed C-RADIOv4-alone
     test (three-seed sigma_hat; line 187 keeps the re-pooled sigma_hat for H and N only; line 176 does not say which
     H to pair it with).
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
GATE = Fraction(36)                          # G1 (item 5)
FLOOR = Fraction(27, 10)                     # 2.7: T3's floor and M2's margin
EXT_MIN_NEGATIVE = 4                         # S-4of5

C_FAILS = "C fails, under C-a"
LABELS = {("H", "K"): "H and K", ("H", "H_L"): "H, not K, and H_L", ("H", "unresolved"): "H, not K, not H_L",
          ("N", "K"): "N and K", ("N", "H_L"): "N, not K", ("N", "unresolved"): "N, not K"}

EXT_NA = "not applicable: the harm test is not I"
EXT_DUE = "due: complete three-seed I with C passed; train seeds 3 and 4 of dinov3 and dinov3+c-radiov4 (item 7)"
EXT_MISSING = "not qualifying: an I caused by a missing seed does not trigger the extension (item 7)"
EXT_USER = ("user decides: complete three-seed I, but C fails under C-a; the PLAN does not say whether the extension "
            "applies then (item 7 'if the result is I' vs item 8 'C fails' options)")
EXT_DONE = "done: seeds 3 and 4 analysed; H and N on five seeds, C, K, H_L on seeds 0 to 2 (two-stage design)"
NOT_CLAIMED = "not claimed (C fails)"

PENDING_EXT_C_FAILS = ("extension when the harm test is a complete three-seed I but C fails under C-a: 'user decides' "
                       "(PLAN lines 187 and 190)")
PENDING_CLEAN_500 = ("clean LIBERO with other than 500 rollouts in a suite: refused; the PLAN has no clean coverage "
                     "rule (line 117 covers LIBERO-Plus only)")
PENDING_BELOW_BOTH_EXT = ("after the extension, 'below both members' pairs the five-seed harm result with the "
                          "three-seed C-RADIOv4-alone test (lines 176 and 187)")
PENDING = (PENDING_EXT_C_FAILS, PENDING_CLEAN_500, PENDING_BELOW_BOTH_EXT)


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
    clean = rec.get("clean")
    if not isinstance(clean, dict):
        raise InputError(f"{where}: no clean tally (the gate needs B for every checkpoint, item 5)")
    if sorted(clean) != sorted(CLEAN_SUITES):
        raise InputError(f"{where}: clean suites {sorted(clean)}, need {sorted(CLEAN_SUITES)}")
    rates = {}
    for suite in CLEAN_SUITES:
        s = clean[suite]
        if not isinstance(s, dict):
            raise InputError(f"{where}: clean {suite} is not an object")
        n = _count(s.get("episodes"), f"{where} clean {suite} episodes")
        k = _count(s.get("successes"), f"{where} clean {suite} successes")
        if n != CLEAN_PER_SUITE:
            raise InputError(f"{where}: clean {suite} has {n} episodes, need {CLEAN_PER_SUITE} (2,000 rollouts). "
                             f"Unresolved, pending the user: {PENDING_CLEAN_500}")
        if k > n:
            raise InputError(f"{where}: clean {suite} successes {k} > episodes {n}")
        rates[suite] = Fraction(100 * k, n)
    return rates, sum(rates.values(), Fraction(0)) / len(rates)


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
    suite_rates, B = _clean_B(rec, where)
    gate_pass = B >= GATE
    diag = rec.get("gate_diagnosis")
    if not gate_pass:
        if not isinstance(diag, dict) or not isinstance(diag.get("fault_found"), bool):
            raise InputError(f"{where}: B = {float(B):.2f} < 36 and no gate diagnosis "
                             '("gate_diagnosis": {"fault_found": false, "note": ...}) (item 5)')
        if diag["fault_found"]:
            raise InputError(f"{where}: the gate diagnosis found a fault; fix it by PR, re-evaluate every affected "
                             "checkpoint in every configuration, and supply the new results (item 6)")
    lp = _lp(rec, where)
    rates = {es: Fraction(100 * k, n) if n else None for es, (k, n, m) in lp.items()}
    missing_total = sum(m for _, _, m in lp.values())
    too_incomplete = Fraction(missing_total, LP_ROLLOUTS) > MISSING_MAX
    out = {"status": "ok", "checkpoint": rec.get("checkpoint"),
           "clean_suite_rates": suite_rates, "B": B, "gate_pass": gate_pass,
           "gate_diagnosis": diag if not gate_pass else None,
           "lp_counts": lp, "lp_rates": rates, "missing_per_eval_seed": {es: lp[es][2] for es in EVAL_SEEDS},
           "missing_total": missing_total, "missing_frac": Fraction(missing_total, LP_ROLLOUTS)}
    if too_incomplete:
        out.update(status="missing", missing_reason=f"coverage: {missing_total} of {LP_ROLLOUTS} LIBERO-Plus "
                   "rollouts missing after reruns, more than 1%")
        return out
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

    # Below both members (secondary; c-radiov4 alone is trained).
    r_cmp = compare(ck, DR, R_, BASE_SEEDS)
    hr_res, hr_T = harm_like(r_cmp, V3)
    HR = _test_out(r_cmp, hr_res, r_cmp["complete"], hr_T, extra={"sign_condition": "every d_s < 0"})

    # Extension status (item 7).
    if harm3 != "I":
        ext_status = EXT_NA
    elif not h_cmp["complete"]:
        ext_status = EXT_MISSING
    elif c_holds:
        ext_status = EXT_DUE
    else:
        ext_status = EXT_USER

    harm, stage5 = harm3, None
    if has_ext:
        if ext_status not in (EXT_DUE, EXT_USER):
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
                  "harm_outcome": harm if c_holds else NOT_CLAIMED}
        ext_status = EXT_DONE + ("" if c_holds else "; C fails under C-a, so the label stays C fails")

    label = C_FAILS if not c_holds else ("I" if harm == "I" else LABELS[(harm, helpful)])
    below_both = bool(harm == "H" and hr_res and r_cmp["complete"])
    applies = {PENDING_EXT_C_FAILS: ext_status == EXT_USER or (has_ext and not c_holds),
               PENDING_CLEAN_500: False,     # the input was accepted, so every clean suite had 500 rollouts
               PENDING_BELOW_BOTH_EXT: bool(has_ext and c_holds)}
    return {"rule": "PLAN 2026-11-15-pretrained-start-pairings, Decision rule as signed off: T3, M2, C-a, G1 at 36, "
                    "coverage 1%, extension S-4of5 for a complete three-seed I",
            "outcome": label,
            "harm_outcome": harm if c_holds else NOT_CLAIMED,
            "harm_outcome_three_seeds": harm3 if c_holds else NOT_CLAIMED,
            "harm_test_as_computed_not_claimed": None if c_holds else {"final": harm, "three_seeds": harm3},
            "helpful_pairing": helpful,
            "calibration_C_holds": c_holds,
            "below_both_members": below_both if c_holds else NOT_CLAIMED,
            "extension": ext_status,
            "unresolved_pending_user": [{"case": p, "applies_to_this_input": applies[p]} for p in PENDING],
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


def _ck_out(ck):
    out = []
    for (c, s), r in sorted(ck.items(), key=lambda kv: (CONFIGS.index(kv[0][0]), kv[0][1])):
        e = {"config": c, "train_seed": s, "status": r["status"], "checkpoint": r.get("checkpoint")}
        if "missing_reason" in r:
            e["missing_reason"] = r["missing_reason"]
        if "B" in r:
            e.update(B_pts=float(r["B"]), gate_G1_pass=r["gate_pass"], gate_diagnosis=r["gate_diagnosis"],
                     clean_suite_rates_pts={k: float(v) for k, v in r["clean_suite_rates"].items()},
                     lp_rates_pts={es: (None if v is None else float(v)) for es, v in r["lp_rates"].items()},
                     lp_missing_per_eval_seed=r["missing_per_eval_seed"], lp_missing_total=r["missing_total"],
                     lp_missing_pct=float(r["missing_frac"] * 100))
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
             f"  extension: {res['extension']}"]
    if not res["calibration_C_holds"]:
        lines[1] = (f"  harm test: {NOT_CLAIMED}; below both members: {NOT_CLAIMED}; C holds: False "
                    f"(PLAN line 190: the comparisons below are reported, nothing is claimed); helpful pairing: "
                    f"{res['helpful_pairing']}")
    for p in res["unresolved_pending_user"]:
        if p["applies_to_this_input"]:
            lines.append(f"  UNRESOLVED, PENDING THE USER (applies here): {p['case']}")
    lines.append(f"  sigma_hat (seeds 0-2): {_fmt(res['sigma_hat_three_seeds_pts'])} pts, df {res['df_three_seeds']}")
    if res["extension_stage"]:
        e = res["extension_stage"]
        lines.append(f"  sigma_hat (re-pooled, extension): {_fmt(e['sigma_hat_pts'])} pts, df {e['df']}")
    lines.append("checkpoints (Y = mean of the three LIBERO-Plus rates; B = clean four-suite average):")
    for r in res["checkpoints"]:
        if "Y_pts" in r:
            lines.append(f"  {r['config']:<18} s{r['train_seed']}  Y {_fmt(r['Y_pts'])}  eval-seed sd "
                         f"{_fmt(r['eval_seed_sd_pts'])}  missing {r['lp_missing_total']}  B {_fmt(r['B_pts'], 2)}"
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


def main(argv=None):
    p = argparse.ArgumentParser(description="Stage 2 analysis: the locked decision rule (see the module docstring).")
    p.add_argument("--input", required=True, help="JSON of per-checkpoint totals (format in this file's docstring)")
    p.add_argument("--out", default=None, help="write the full analysis as JSON")
    args = p.parse_args(argv)
    try:
        res = analyse(json.loads(Path(args.input).read_text()))
    except InputError as e:
        print(f"INPUT ERROR (no outcome): {e}", file=sys.stderr)
        return 2
    print(report(res))
    if args.out:
        Path(args.out).write_text(json.dumps(_jsonable(res), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
