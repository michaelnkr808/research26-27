"""Synthetic checks for pairings_analysis.py: every outcome branch of the locked stage 2 rule, the extension, the
three cases decided in the PLAN amendment of 2026-11-17 (item 3 (a) to (c); class DecidedCases), and missing or
incomplete input. Run: /opt/homebrew/bin/python3.11 -I test_pairings_analysis.py

Synthetic numbers only. Most checkpoints are built with 10,000 evaluated and 30 missing episodes per eval seed
(90 of 30,090 missing, under the 1% line), so Y = successes / 100 is an exact decimal and boundary cases are exact.
The default design: none 70, dinov3 80, dinov3+siglip2 80, dinov3+c-radiov4 75, c-radiov4 78 (constant over seeds),
siglip2 78 - e, 78, 78 + e with e = 1.5. Only siglip2 varies, so the pooled variance is 2 e^2 / 12 = e^2 / 6 and
f(3) x sigma_hat = 2 e / 3 exactly (1.0 at e = 1.5, 3.0 at e = 4.5).
"""
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pairings_analysis as P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NONE, D, DL, DR, L_, R_ = P.NONE, P.D, P.DL, P.DR, P.L_, P.R_


def ck(config, seed, y, B=80, miss=(30, 30, 30), per_eval=None, **extra):
    """One checkpoint entry. y in points (str or number); per_eval overrides the three eval-seed rates."""
    ys = per_eval or (y, y, y)
    lp = {}
    for es, yy, m in zip(P.EVAL_SEEDS, ys, miss):
        ev = P.LP_EPISODES - m
        succ = Fraction(str(yy)) * ev / 100
        lp[es] = {"successes": round(succ), "evaluated": ev, "missing": m}
    k = Fraction(str(B)) * 5
    assert k.denominator == 1
    clean = {s: {"successes": int(k), "episodes": 500} for s in P.CLEAN_SUITES}
    e = {"config": config, "train_seed": seed, "status": "ok", "clean": clean, "libero_plus": lp}
    e.update(extra)
    return e


def gone(config, seed, reason="non-finite loss, rerun failed"):
    return {"config": config, "train_seed": seed, "status": "missing", "reason": reason}


def design(e=1.5, **vals):
    """vals: config name with '+' and '-' replaced (dinov3_c_radiov4) -> value or list of three per-seed values."""
    base = {NONE: 70, D: 80, DL: 80, DR: 75, R_: 78, L_: None}
    for k, v in vals.items():
        base[{"none": NONE, "dinov3": D, "dl": DL, "dr": DR, "r": R_}[k]] = v
    out = []
    for c in P.CONFIGS:
        for s in P.BASE_SEEDS:
            if c == L_:
                y = Fraction(78) + (s - 1) * Fraction(str(e))
            else:
                v = base[c]
                y = v[s] if isinstance(v, (list, tuple)) else v
            out.append(ck(c, s, float(y) if isinstance(y, Fraction) else y))
    return out


def replace(entries, new):
    keys = {(n["config"], n["train_seed"]) for n in new}
    return [x for x in entries if (x["config"], x["train_seed"]) not in keys] + list(new)


def run(entries):
    return P.analyse({"checkpoints": entries})


def ext(entries, d_vals=(80, 80), dr_vals=(76, 76)):
    return entries + [ck(D, 3, d_vals[0]), ck(D, 4, d_vals[1]), ck(DR, 3, dr_vals[0]), ck(DR, 4, dr_vals[1])]


class Statistic(unittest.TestCase):
    def test_Y_is_mean_of_three_rates_and_eval_sd(self):
        es = design()
        es = replace(es, [ck(D, 0, None, per_eval=(80, 81, 82))])
        r = run(es)
        c = [x for x in r["checkpoints"] if x["config"] == D and x["train_seed"] == 0][0]
        self.assertEqual(c["Y_exact"], "81")
        self.assertAlmostEqual(c["eval_seed_sd_pts"], 1.0)
        self.assertEqual(c["lp_missing_total"], 90)

    def test_Y_is_mean_of_rates_not_pooled_rate(self):
        # uneven denominators: 9,730 / 10,030 / 10,030 evaluated, rates 0, 100, 100. Mean of rates = 200/3;
        # pooled successes / pooled evaluated = 20060 / 29790 = 67.34, which the rule does not use.
        es = replace(design(), [ck(D, 0, None, miss=(300, 0, 0), per_eval=(0, 100, 100))])
        c = [x for x in run(es)["checkpoints"] if x["config"] == D and x["train_seed"] == 0][0]
        self.assertEqual(c["Y_exact"], "200/3")
        self.assertNotAlmostEqual(c["Y_pts"], 100 * 20060 / 29790, places=3)

    def test_zero_difference_is_neither_harm_nor_gain(self):
        r = run(design(dr=[74, 74, 80]))          # d = -6, -6, 0: dbar -4 <= -T, but 0 is not < 0
        h = r["tests_three_seeds"]["H"]
        self.assertLessEqual(h["dbar_pts"], -h["threshold_T_pts"])
        self.assertFalse(h["result_on_paired_seeds"])
        self.assertEqual(r["harm_outcome"], "I")
        r = run(design(none=[70, 70, 80]))        # d(dinov3 - none) = 10, 10, 0: dbar >= T, but 0 is not > 0
        c = r["tests_three_seeds"]["C"]
        self.assertGreaterEqual(c["dbar_pts"], c["threshold_T_pts"])
        self.assertFalse(c["result_on_paired_seeds"])
        self.assertEqual(r["outcome"], "C fails, under C-a")

    def test_missing_episodes_leave_the_denominator(self):
        es = replace(design(), [ck(D, 0, 80, miss=(100, 0, 0))])
        lp = [x for x in es if x["config"] == D and x["train_seed"] == 0][0]["libero_plus"]
        r = run(es)
        c = [x for x in r["checkpoints"] if x["config"] == D and x["train_seed"] == 0][0]
        k = lp["41"]["successes"]
        self.assertAlmostEqual(c["lp_rates_pts"]["41"], 100 * k / 9930)

    def test_B_is_mean_of_four_suite_rates(self):
        es = design()
        e = ck(D, 0, 80)
        for s, k in zip(P.CLEAN_SUITES, (450, 400, 350, 300)):
            e["clean"][s]["successes"] = k
        r = run(replace(es, [e]))
        c = [x for x in r["checkpoints"] if x["config"] == D and x["train_seed"] == 0][0]
        self.assertEqual(c["B_pts"], 75.0)

    def test_sigma_hat_pooled(self):
        r = run(design(e=1.5))
        self.assertEqual(r["pooled_variance_three_seeds_exact"], str(Fraction(3, 8)))   # 2 * 1.5^2 / 12
        self.assertEqual(r["df_three_seeds"], 12)
        self.assertAlmostEqual(r["sigma_hat_three_seeds_pts"], math.sqrt(0.375))

    def test_sigma_hat_pools_configs_with_two_seeds(self):
        es = replace(design(e=1.5), [gone(L_, 2)])
        r = run(es)
        self.assertEqual(r["df_three_seeds"], 11)
        self.assertEqual(r["pooled_variance_three_seeds_exact"], str(Fraction(9, 8) / 11))  # (76.5, 78): ss 1.125

    def test_f_values(self):
        self.assertAlmostEqual(P.f_of(3), 1.63299, places=5)
        self.assertAlmostEqual(P.f_of(5), 1.26491, places=5)


class Outcomes(unittest.TestCase):
    def check(self, r, label, harm, helpful, c=True):
        shown, shown_l = (harm, helpful) if c else (P.NOT_CLAIMED, P.NOT_CLAIMED)
        self.assertEqual((r["outcome"], r["harm_outcome"], r["helpful_pairing"], r["calibration_C_holds"]),
                         (label, shown, shown_l, c))
        if not c:
            self.assertEqual(r["harm_test_as_computed_not_claimed"]["final"], harm)
            self.assertEqual(r["helpful_pairing_as_computed_not_claimed"], helpful)

    def test_H_and_K(self):
        r = run(design())                         # d(DR-D) = -5, d(DL-D) = 0
        self.check(r, "H and K", "H", "K")
        self.assertEqual(r["tests_three_seeds"]["H"]["threshold_T_pts"], 2.7)
        self.assertEqual(r["extension"], P.EXT_NA)

    def test_H_not_K_and_H_L(self):
        self.check(run(design(dl=76)), "H, not K, and H_L", "H", "H_L")

    def test_H_not_K_not_H_L(self):
        self.check(run(design(dl=78)), "H, not K, not H_L", "H", "unresolved")   # -2: not H_L; -2 - 1 < -2.7

    def test_N_and_K(self):
        self.check(run(design(dr=80)), "N and K", "N", "K")

    def test_N_not_K_unresolved(self):
        self.check(run(design(dr=80, dl=78)), "N, not K", "N", "unresolved")

    def test_N_not_K_with_H_L(self):
        self.check(run(design(dr=80, dl=76)), "N, not K", "N", "H_L")

    def test_I_by_size(self):
        r = run(design(dr=78))                    # -2: not <= -2.7; -2 - 1 = -3 not > -2.7
        self.check(r, "I", "I", "K")
        self.assertEqual(r["extension"], P.EXT_DUE)

    def test_I_by_sign(self):
        r = run(design(dr=[74, 74, 81]))          # d = -6, -6, +1: mean -3.67 but one positive
        self.check(r, "I", "I", "unresolved")     # DR's spread raises f(3) sigma_hat to 2.87, so K fails too
        self.assertFalse(r["tests_three_seeds"]["H"]["result_on_paired_seeds"])
        self.assertLess(r["tests_three_seeds"]["H"]["dbar_pts"], -r["tests_three_seeds"]["H"]["threshold_T_pts"])

    def test_C_fails_by_size(self):
        self.check(run(design(none=78)), "C fails, under C-a", "H", "K", c=False)   # 2 < 2.7

    def test_C_fails_by_sign(self):
        r = run(design(none=[72, 72, 81]))        # d = 8, 8, -1: mean 5 >= T3 (3.61) but one negative
        self.check(r, "C fails, under C-a", "H", "unresolved", c=False)
        c = r["tests_three_seeds"]["C"]
        self.assertGreaterEqual(c["dbar_pts"], c["threshold_T_pts"])
        self.assertFalse(c["result_on_paired_seeds"])

    def test_below_both_members(self):
        r = run(design())                         # DR 75 vs c-radiov4 78: d = -3
        self.assertTrue(r["below_both_members"])
        self.assertTrue(r["tests_three_seeds"]["H_vs_c-radiov4"]["result_on_paired_seeds"])
        r = run(design(r=76))                     # d = -1
        self.assertFalse(r["below_both_members"])
        self.assertEqual(r["outcome"], "H and K")  # secondary: does not change the outcome
        r = run(design(dr=80, r=90))              # H_R holds but H does not: not "below both"
        self.assertTrue(r["tests_three_seeds"]["H_vs_c-radiov4"]["result_on_paired_seeds"])
        self.assertFalse(r["below_both_members"])


class Boundaries(unittest.TestCase):
    def test_H_at_exactly_minus_2_7(self):
        self.assertEqual(run(design(dr=77.3))["harm_outcome"], "H")
        self.assertEqual(run(design(dr=77.31))["harm_outcome"], "I")

    def test_M2_strict(self):
        self.assertEqual(run(design(dr=78.3))["harm_outcome"], "I")    # -1.7 - 1.0 = -2.7, not > -2.7
        self.assertEqual(run(design(dr=78.31))["harm_outcome"], "N")

    def test_K_strict(self):
        self.assertEqual(run(design(dl=78.3))["helpful_pairing"], "unresolved")
        self.assertEqual(run(design(dl=78.31))["helpful_pairing"], "K")

    def test_C_at_exactly_2_7(self):
        self.assertTrue(run(design(none=77.3))["calibration_C_holds"])
        self.assertFalse(run(design(none=77.31))["calibration_C_holds"])

    def test_T3_raised_by_noise(self):
        r = run(design(e=4.5, dr=77))             # f(3) sigma_hat = 3.0 exactly; d = -3.0
        self.assertAlmostEqual(r["tests_three_seeds"]["H"]["threshold_T_pts"], 3.0)
        self.assertEqual(r["harm_outcome"], "H")
        r = run(design(e=4.5, dr=77.1))           # -2.9: past 2.7 (T1 would say H) but not past T3 = 3.0
        self.assertEqual(r["harm_outcome"], "I")
        self.assertAlmostEqual(r["tests_three_seeds"]["C"]["threshold_T_pts"], 3.0)

    def test_gate_at_exactly_36(self):
        es = replace(design(), [ck(D, 0, 80, B=36)])
        c = [x for x in run(es)["checkpoints"] if x["config"] == D and x["train_seed"] == 0][0]
        self.assertTrue(c["gate_G1_pass"])

    def test_gate_fail_without_fault_is_still_scored(self):
        es = replace(design(), [ck(D, 0, 80, B=35.8, gate_diagnosis={"fault_found": False, "note": "x"})])
        r = run(es)
        c = [x for x in r["checkpoints"] if x["config"] == D and x["train_seed"] == 0][0]
        self.assertFalse(c["gate_G1_pass"])
        self.assertEqual(r["outcome"], "H and K")

    def test_coverage_300_vs_301_missing(self):
        es = replace(design(), [ck(D, 1, 80, miss=(100, 100, 100))])
        self.assertEqual(run(es)["outcome"], "H and K")
        es = replace(design(), [ck(D, 1, 80, miss=(101, 100, 100))])
        r = run(es)
        c = [x for x in r["checkpoints"] if x["config"] == D and x["train_seed"] == 1][0]
        self.assertEqual(c["status"], "missing")
        self.assertEqual((r["outcome"], r["harm_outcome"], r["helpful_pairing"]),
                         ("C fails, under C-a", P.NOT_CLAIMED, P.NOT_CLAIMED))
        self.assertEqual(r["harm_test_as_computed_not_claimed"]["final"], "I")
        self.assertEqual(r["helpful_pairing_as_computed_not_claimed"], "unresolved")


class MissingSeeds(unittest.TestCase):
    def test_missing_harmful_seed_gives_I_and_no_extension(self):
        r = run(replace(design(), [gone(DR, 2)]))
        self.assertEqual((r["outcome"], r["harm_outcome"]), ("I", "I"))
        h = r["tests_three_seeds"]["H"]
        self.assertEqual((h["n"], h["established"], h["result_on_paired_seeds"]), (2, False, True))
        self.assertEqual(r["extension"], P.EXT_MISSING)

    def test_missing_none_seed_is_C_fails(self):
        r = run(replace(design(), [gone(NONE, 0)]))
        self.assertEqual(r["outcome"], "C fails, under C-a")
        self.assertTrue(r["tests_three_seeds"]["C"]["result_on_paired_seeds"])
        self.assertFalse(r["tests_three_seeds"]["C"]["established"])
        self.assertEqual(r["harm_outcome"], P.NOT_CLAIMED)
        self.assertEqual(r["harm_test_as_computed_not_claimed"]["final"], "H")

    def test_missing_helpful_seed_is_unresolved(self):
        r = run(replace(design(), [gone(DL, 1)]))
        self.assertEqual((r["outcome"], r["helpful_pairing"]), ("H, not K, not H_L", "unresolved"))
        r = run(replace(design(dr=80), [gone(DL, 1)]))
        self.assertEqual(r["outcome"], "N, not K")

    def test_missing_dinov3_seed(self):
        r = run(replace(design(), [gone(D, 2)]))
        self.assertEqual((r["outcome"], r["harm_outcome"], r["helpful_pairing"], r["extension"]),
                         ("C fails, under C-a", P.NOT_CLAIMED, P.NOT_CLAIMED, P.EXT_MISSING))
        self.assertEqual(r["helpful_pairing_as_computed_not_claimed"], "unresolved")
        self.assertEqual(r["harm_test_as_computed_not_claimed"]["final"], "I")

    def test_missing_c_radiov4_alone(self):
        r = run(replace(design(), [gone(R_, 0)]))
        self.assertEqual(r["outcome"], "H and K")
        self.assertFalse(r["tests_three_seeds"]["H_vs_c-radiov4"]["established"])
        self.assertFalse(r["below_both_members"])


class Extension(unittest.TestCase):
    def base(self, **kw):
        return design(dr=[76, 76, 81], **kw)      # d = -4, -4, +1: complete three-seed I

    def test_three_seed_I_makes_extension_due(self):
        r = run(self.base())
        self.assertEqual((r["outcome"], r["extension"]), ("I", P.EXT_DUE))

    def test_extension_H_with_4_of_5(self):
        r = run(ext(self.base(), dr_vals=(76, 76)))
        self.assertEqual((r["outcome"], r["harm_outcome"], r["harm_outcome_three_seeds"]), ("H and K", "H", "I"))
        h = r["extension_stage"]["H"]
        self.assertEqual((h["n"], h["negative_count"], h["dbar_exact"]), (5, 4, "-3"))
        self.assertEqual(r["extension_stage"]["df"], 16)
        self.assertEqual(r["extension_stage"]["pooled_variance_exact"], str(Fraction(49, 32)))  # (20 + 4.5) / 16
        self.assertEqual(r["extension"], P.EXT_DONE)

    def test_extension_keeps_three_seed_C_K_H_L(self):
        es = self.base()
        r3, r5 = run(es), run(ext(es))
        for t in ("C", "K", "H_L"):
            self.assertEqual(r3["tests_three_seeds"][t], r5["tests_three_seeds"][t])
        self.assertEqual(r3["sigma_hat_three_seeds_pts"], r5["sigma_hat_three_seeds_pts"])

    def test_extension_still_I(self):
        r = run(ext(self.base(), dr_vals=(80, 80)))
        self.assertEqual((r["outcome"], r["harm_outcome"]), ("I", "I"))
        self.assertEqual(r["extension"], P.EXT_DONE)

    def test_extension_N(self):
        r = run(ext(self.base(), dr_vals=(83, 83)))
        self.assertEqual((r["outcome"], r["harm_outcome"]), ("N and K", "N"))

    def test_extension_3_of_5_is_not_H(self):
        es = design(dr=[72, 72, 81])              # d = -8, -8, +1
        r = run(ext(es, dr_vals=(72, 81)))        # five: -8, -8, +1, -8, +1: mean -4.4, 3 negative
        h = r["extension_stage"]["H"]
        self.assertEqual((h["negative_count"], h["result_on_paired_seeds"]), (3, False))
        self.assertLess(h["dbar_pts"], -h["threshold_T_pts"])
        self.assertEqual(r["harm_outcome"], "I")

    def test_extension_missing_seed_gives_I(self):
        es = ext(self.base(), dr_vals=(76, 76))
        es = replace(es, [gone(DR, 4)])
        r = run(es)
        self.assertEqual((r["outcome"], r["harm_outcome"]), ("I", "I"))
        self.assertIsNone(r["extension_stage"]["H"]["result_on_paired_seeds"])
        self.assertFalse(r["extension_stage"]["H"]["established"])

    def test_extension_not_allowed_after_H(self):
        with self.assertRaisesRegex(P.InputError, "does not qualify"):
            run(ext(design()))

    def test_extension_not_allowed_after_missing_seed_I(self):
        with self.assertRaisesRegex(P.InputError, "does not qualify"):
            run(ext(replace(self.base(), [gone(DR, 2)])))

    def test_extension_incomplete(self):
        with self.assertRaisesRegex(P.InputError, "extension incomplete"):
            run(self.base() + [ck(D, 3, 80)])

    def test_extension_H_uses_repooled_sigma(self):
        # siglip2 e = 6 makes f(5) x sigma_hat exceed 2.7. Re-pooled (seeds 0-4): T = 3.24, dbar = -3.4, so H.
        # With the three-seed sigma_hat at n = 5, T would be 3.74 and the result I.
        es = design(e=6, dr=[74, 74, 81])
        self.assertEqual(run(es)["extension"], P.EXT_DUE)
        r = run(ext(es, dr_vals=(77, 77)))
        h = r["extension_stage"]["H"]
        s5, s3 = r["extension_stage"]["sigma_hat_pts"], r["sigma_hat_three_seeds_pts"]
        self.assertAlmostEqual(h["threshold_T_pts"], P.f_of(5) * s5)
        self.assertGreater(P.f_of(5) * s5, 2.7)
        self.assertGreater(P.f_of(5) * s3, -h["dbar_pts"])      # the three-seed sigma would block H
        self.assertEqual((h["dbar_exact"], h["negative_count"]), ("-17/5", 4))
        self.assertEqual((r["harm_outcome"], r["outcome"]), ("H", "H, not K, not H_L"))

    def test_extension_N_uses_repooled_sigma(self):
        # siglip2 e = 3; the extension seeds of dinov3+c-radiov4 (83, 90) widen the spread: f(5) x re-pooled
        # sigma_hat = 4.46, dbar = +0.4, so 0.4 - 4.46 < -2.7: I. With the three-seed sigma_hat N would hold.
        es = design(e=3, dr=[74, 74, 81])
        self.assertEqual(run(es)["extension"], P.EXT_DUE)
        r = run(ext(es, dr_vals=(83, 90)))
        st = r["extension_stage"]
        n = st["N"]
        self.assertAlmostEqual(n["margin"]["f_n_sigma_hat_pts"], P.f_of(5) * st["sigma_hat_pts"])
        self.assertGreater(n["margin"]["f_n_sigma_hat_pts"], 2.7)
        self.assertGreater(n["dbar_pts"] - P.f_of(5) * r["sigma_hat_three_seeds_pts"], -2.7)  # three-seed: N
        self.assertFalse(n["result_on_paired_seeds"])
        self.assertEqual(r["harm_outcome"], "I")

    def test_extension_zero_difference_is_not_negative(self):
        # five d_s: -6, -6, +1, -6, 0 (dbar -3.4 <= -T): only 3 negative, so not H under S-4of5
        es = design(dr=[74, 74, 81])
        r = run(ext(es, dr_vals=(74, 80)))
        h = r["extension_stage"]["H"]
        self.assertEqual((h["negative_count"], h["result_on_paired_seeds"]), (3, False))
        self.assertLessEqual(h["dbar_pts"], -h["threshold_T_pts"])
        self.assertEqual(r["harm_outcome"], "I")



class NotClaimedAndPending(unittest.TestCase):
    def test_C_fails_claims_nothing(self):
        r = run(design(none=78))                  # C fails; computed harm test H; DR below c-radiov4 too
        self.assertEqual(r["outcome"], "C fails, under C-a")
        self.assertEqual((r["harm_outcome"], r["harm_outcome_three_seeds"], r["below_both_members"]),
                         (P.NOT_CLAIMED,) * 3)
        self.assertEqual(r["harm_test_as_computed_not_claimed"], {"final": "H", "three_seeds": "H"})
        self.assertEqual((r["helpful_pairing"], r["helpful_pairing_as_computed_not_claimed"]), (P.NOT_CLAIMED, "K"))
        self.assertTrue(r["tests_three_seeds"]["H_vs_c-radiov4"]["result_on_paired_seeds"])  # still reported
        self.assertTrue(r["tests_three_seeds"]["K"]["result_on_paired_seeds"])               # still reported
        text = P.report(r)
        self.assertIn("harm test: not claimed (C fails); helpful pairing: not claimed (C fails); "
                      "below both members: not claimed (C fails)", text)
        self.assertNotIn("harm test: H", text)
        self.assertNotIn("helpful pairing: K", text)
        self.assertNotIn("below both members: True", text)

    def test_C_holds_states_results_plainly(self):
        r = run(design())
        self.assertEqual((r["harm_outcome"], r["below_both_members"]), ("H", True))
        self.assertIsNone(r["harm_test_as_computed_not_claimed"])
        self.assertEqual((r["helpful_pairing"], r["helpful_pairing_as_computed_not_claimed"]), ("K", None))
        self.assertIn("harm test: H", P.report(r))
        self.assertIn("helpful pairing: K", P.report(r))

    def test_no_pending_cases_remain(self):
        r = run(design())
        self.assertNotIn("unresolved_pending_user", r)
        self.assertEqual(r["stops_for_user"], [])
        text = P.report(r)
        self.assertNotIn("PENDING THE USER", text)
        self.assertNotIn("STOP FOR THE USER", text)

    def test_docstring_names_the_decided_and_open_cases(self):
        doc = " ".join(P.__doc__.split())
        self.assertNotIn("UNRESOLVED, PENDING THE USER", doc)
        for words in ("DECIDED CASES (PLAN amendment 2026-11-17, item 3", "the extension of item 7 is not run",
                      "seeds 0 to 2 only, paired by seed", "21 or more make the checkpoint a missing seed",
                      "OPEN CASE", "still gets its LIBERO-Plus evaluation",
                      "The exit-3 stop is a reminder after the fact", "must stop and ask the user at the LIBERO-Plus "
                      "evaluation step itself", '"user_decision_libero_plus": "evaluate" | "skip"'):
            self.assertIn(words, doc)


def clean_short(entry, per_suite):
    """Take per_suite[i] rollouts of clean suite i out as missing (no record), keeping the suite's success rate."""
    for s, m in zip(P.CLEAN_SUITES, per_suite):
        c = entry["clean"][s]
        rate = Fraction(c["successes"], c["episodes"])
        c["episodes"] -= m
        c["successes"] = int(rate * c["episodes"]) if c["episodes"] else 0
    return entry


def get_ck(r, config, seed):
    return [x for x in r["checkpoints"] if x["config"] == config and x["train_seed"] == seed][0]


class DecidedCases(unittest.TestCase):
    """PLAN amendment 2026-11-17, item 3 (lines 398-405). Each test fails on the script of PR #56, which reported
    these cases as pending the user (an EXT_USER status and analysed extension seeds under C fails; a
    "unresolved_pending_user" list; any clean suite other than 500 rollouts refused)."""

    def three_seed_I(self, **kw):
        return design(dr=[76, 76, 81], **kw)      # d(DR - D) = -4, -4, +1: complete three-seed I

    # (a) line 399: no extension under "C fails, under C-a".
    def test_a_C_fails_complete_I_no_extension(self):
        r = run(self.three_seed_I(none=79))       # d(D - none) = 1: C fails by size
        self.assertEqual((r["outcome"], r["extension"]), ("C fails, under C-a", P.EXT_C_FAILS))
        self.assertTrue(r["tests_three_seeds"]["H"]["established"])   # all three seeds present: still not run
        self.assertEqual(r["harm_test_as_computed_not_claimed"], {"final": "I", "three_seeds": "I"})
        self.assertIn("extension: not run: C fails under C-a", P.report(r))

    def test_a_C_fails_extension_seeds_refused(self):
        with self.assertRaisesRegex(P.InputError, "does not qualify.*C fails under C-a"):
            run(ext(self.three_seed_I(none=79)))

    def test_a_C_not_established_is_read_as_C_fails_no_extension(self):
        es = replace(self.three_seed_I(), [gone(NONE, 1)])   # C not established; harm test complete I
        r = run(es)
        self.assertFalse(r["tests_three_seeds"]["C"]["established"])
        self.assertTrue(r["tests_three_seeds"]["H"]["established"])
        self.assertEqual((r["outcome"], r["extension"]), ("C fails, under C-a", P.EXT_C_FAILS))
        with self.assertRaisesRegex(P.InputError, "does not qualify"):
            run(ext(es))

    def test_a_C_holds_complete_I_extension_still_due(self):
        self.assertEqual(run(self.three_seed_I())["extension"], P.EXT_DUE)

    # (b) line 400: below both members after the extension: seeds 0-2, paired, pre-extension sigma_hat, T, f(3).
    def test_b_after_extension_stated_with_five_and_three_seed_basis(self):
        # e = 1.5, dr 76, 76, 81, c-radiov4 80, 80, 85: d(DR - R) = -4 on every seed 0-2; H on five seeds
        r = run(ext(self.three_seed_I(r=[80, 80, 85]), dr_vals=(76, 76)))
        self.assertEqual((r["harm_outcome_three_seeds"], r["harm_outcome"]), ("I", "H"))
        hr = r["tests_three_seeds"]["H_vs_c-radiov4"]
        self.assertEqual((hr["paired_seeds"], hr["n"], hr["result_on_paired_seeds"]), ([0, 1, 2], 3, True))
        self.assertAlmostEqual(hr["threshold_T_pts"], max(2.7, P.f_of(3) * r["sigma_hat_three_seeds_pts"]))
        self.assertIs(r["below_both_members"], True)
        self.assertEqual(r["below_both_members_basis"], P.BELOW_BOTH_5)
        self.assertNotIn("unresolved_pending_user", r)
        text = P.report(r)
        self.assertIn("harm test H is on five seeds", text)
        self.assertIn("comparison is on three", text)

    def test_b_uses_pre_extension_threshold_not_repooled(self):
        # e = 3, dr 74, 74, 81, c-radiov4 = dr + 4 (d = -4 on seeds 0-2), extension dr 74, 74: H on five seeds.
        # Three-seed T = f(3) x three-seed sigma_hat = 4.30 > 4: not below both. With the re-pooled sigma_hat
        # (f(5): 3.00, or f(3): 3.87) it would be stated.
        r = run(ext(design(e=3, dr=[74, 74, 81], r=[78, 78, 85]), dr_vals=(74, 74)))
        self.assertEqual(r["harm_outcome"], "H")
        hr, s5 = r["tests_three_seeds"]["H_vs_c-radiov4"], r["extension_stage"]["sigma_hat_pts"]
        self.assertEqual((hr["n"], hr["dbar_exact"], hr["result_on_paired_seeds"]), (3, "-4", False))
        self.assertAlmostEqual(hr["threshold_T_pts"], P.f_of(3) * r["sigma_hat_three_seeds_pts"])
        self.assertGreater(hr["threshold_T_pts"], 4)
        self.assertLess(P.f_of(3) * s5, 4)        # the re-pooled sigma_hat would have passed it
        self.assertIs(r["below_both_members"], False)
        self.assertEqual(r["below_both_members_basis"], P.BELOW_BOTH_5)

    def test_b_without_extension_basis_is_three_seeds(self):
        r = run(design())
        self.assertEqual((r["below_both_members"], r["below_both_members_basis"]), (True, P.BELOW_BOTH_3))

    def test_b_not_stated_under_C_fails(self):
        r = run(design(none=78))
        self.assertEqual((r["below_both_members"], r["below_both_members_basis"]), (P.NOT_CLAIMED, P.NOT_CLAIMED))

    # (c) lines 403-404: missing clean rollouts.
    def test_c_20_missing_leave_the_denominators(self):
        e = clean_short(ck(D, 0, 80), (20, 0, 0, 0))          # libero_spatial: 480 with a record
        e["clean"]["libero_spatial"]["successes"] = 432       # 90% of 480
        r = run(replace(design(), [e]))
        c = get_ck(r, D, 0)
        self.assertEqual((c["status"], c["clean_missing_total"]), ("ok", 20))
        self.assertEqual(c["clean_missing_per_suite"]["libero_spatial"], 20)
        self.assertEqual(c["clean_suite_rates_pts"]["libero_spatial"], 90.0)   # 432 / 480, not 432 / 500
        self.assertEqual(c["B_pts"], (90 + 80 * 3) / 4)
        self.assertEqual((r["outcome"], r["stops_for_user"]), ("H and K", []))
        self.assertIn("clean missing 20", P.report(r))

    def test_c_21_missing_is_a_missing_seed_and_a_stop(self):
        e = clean_short(ck(DR, 1, 75), (6, 5, 5, 5))          # 21 missing across suites
        r = run(replace(design(), [e]))
        c = get_ck(r, DR, 1)
        self.assertEqual((c["status"], c["clean_missing_total"]), ("missing", 21))
        self.assertIn("clean coverage: 21 of 2000", c["missing_reason"])
        # item 6: a missing dinov3+c-radiov4 seed makes the harm outcome I, not qualifying for the extension
        self.assertEqual((r["outcome"], r["harm_outcome"], r["extension"]), ("I", "I", P.EXT_MISSING))
        self.assertFalse(r["tests_three_seeds"]["H"]["established"])
        self.assertEqual([(s["config"], s["train_seed"]) for s in r["stops_for_user"]], [(DR, 1)])
        self.assertIn("LIBERO-Plus results were supplied: reported, not used", r["stops_for_user"][0]["stop"])
        self.assertIn("STOP FOR THE USER: dinov3+c-radiov4 seed 1", P.report(r))
        with self.assertRaisesRegex(P.InputError, "does not qualify"):
            run(ext(replace(design(), [e])))

    def test_c_21_missing_includes_libero_plus_tests(self):
        # a dinov3 seed over the clean line drops out of C (C fails), H and the helpful pairing (unresolved),
        # even though its LIBERO-Plus results are complete
        r = run(replace(design(), [clean_short(ck(D, 2, 80), (21, 0, 0, 0))]))
        self.assertEqual((r["outcome"], r["helpful_pairing_as_computed_not_claimed"]), ("C fails, under C-a",
                                                                                         "unresolved"))
        self.assertEqual(r["harm_test_as_computed_not_claimed"]["final"], "I")
        for t in ("C", "H", "K", "H_L"):
            self.assertEqual(r["tests_three_seeds"][t]["paired_seeds"], [0, 1])
        r = run(replace(design(), [clean_short(ck(DL, 0, 80), (0, 0, 0, 21))]))
        self.assertEqual((r["outcome"], r["helpful_pairing"]), ("H, not K, not H_L", "unresolved"))

    def test_c_libero_plus_not_required_over_the_line(self):
        e = clean_short(ck(DR, 1, 75), (500, 0, 0, 0))        # a whole suite without a record: B undefined
        del e["libero_plus"]
        r = run(replace(design(), [e]))
        c = get_ck(r, DR, 1)
        self.assertEqual((c["status"], c["B_pts"], c["gate_G1_pass"], c["lp_rates_pts"]), ("missing", None, None, None))
        self.assertIn("no LIBERO-Plus results supplied", r["stops_for_user"][0]["stop"])
        self.assertEqual(r["harm_outcome"], "I")

    def test_c_libero_plus_still_required_under_the_line(self):
        e = clean_short(ck(DR, 1, 75), (20, 0, 0, 0))
        del e["libero_plus"]
        with self.assertRaisesRegex(P.InputError, "libero_plus eval seeds"):
            run(replace(design(), [e]))

    # The user's answers on PR #59, 2026-11-17 ("go with both of these").
    def test_c_over_line_low_B_still_needs_gate_diagnosis(self):
        e = clean_short(ck(DR, 1, 75, B=30), (21, 0, 0, 0))   # B 30 < 36 on the records that exist
        with self.assertRaisesRegex(P.InputError, "no gate diagnosis"):
            run(replace(design(), [e]))
        e["gate_diagnosis"] = {"fault_found": True}
        with self.assertRaisesRegex(P.InputError, "found a fault"):
            run(replace(design(), [e]))

    def test_c_over_line_gate_is_computed_and_reported(self):
        diag = {"fault_found": False, "note": "no fault in the fail-branch diagnosis"}
        e = clean_short(ck(DR, 1, 75, B=30, gate_diagnosis=diag), (21, 0, 0, 0))
        c = get_ck(run(replace(design(), [e])), DR, 1)
        self.assertEqual((c["status"], c["gate_G1_pass"], c["gate_diagnosis"]), ("missing", False, diag))
        self.assertAlmostEqual(c["B_pts"], 30, delta=0.1)
        c = get_ck(run(replace(design(), [clean_short(ck(DR, 1, 75), (21, 0, 0, 0))])), DR, 1)
        self.assertEqual((c["gate_G1_pass"], c["gate_diagnosis"]), (True, None))
        self.assertAlmostEqual(c["B_pts"], 80, delta=0.1)

    def test_c_over_line_libero_plus_validated_and_reported_not_used(self):
        e = clean_short(ck(DR, 1, None, miss=(30, 30, 30), per_eval=(60, 61, 62)), (21, 0, 0, 0))
        r = run(replace(design(), [e]))
        c = get_ck(r, DR, 1)
        self.assertEqual(c["status"], "missing")
        for es, y in zip(P.EVAL_SEEDS, (60, 61, 62)):
            self.assertAlmostEqual(c["lp_rates_pts"][es], y)
        self.assertEqual((c["lp_missing_per_eval_seed"], c["lp_missing_total"]), ({"41": 30, "42": 30, "43": 30}, 90))
        self.assertNotIn("Y_pts", c)                              # reported, not used
        self.assertEqual(r["tests_three_seeds"]["H"]["paired_seeds"], [0, 2])
        e["libero_plus"]["42"]["missing"] = 0                     # evaluated + missing != 10,030
        with self.assertRaisesRegex(P.InputError, "!= 10030"):
            run(replace(design(), [e]))

    def test_c_recorded_user_decision_clears_the_stop(self):
        for decision, keep_lp in (("evaluate", True), ("skip", False)):
            e = clean_short(ck(DR, 1, 75), (21, 0, 0, 0))
            if not keep_lp:
                del e["libero_plus"]
            e.update(user_decision_libero_plus=decision, decided="2026-11-20, the user: 'test words'")
            r = run(replace(design(), [e]))
            self.assertEqual(r["stops_for_user"], [])
            self.assertEqual([(d["config"], d["train_seed"], d["user_decision_libero_plus"], d["decided"])
                              for d in r["user_decisions"]], [(DR, 1, decision, "2026-11-20, the user: 'test words'")])
            self.assertEqual((r["outcome"], get_ck(r, DR, 1)["status"]), ("I", "missing"))   # still a missing seed
            text = P.report(r)
            self.assertNotIn("STOP FOR THE USER", text)
            self.assertIn(f"USER DECISION (recorded): dinov3+c-radiov4 seed 1: LIBERO-Plus evaluation '{decision}', "
                          "decided: 2026-11-20, the user: 'test words'", text)

    def test_c_without_decision_stop_unchanged(self):
        r = run(replace(design(), [clean_short(ck(DR, 1, 75), (21, 0, 0, 0))]))
        self.assertEqual((len(r["stops_for_user"]), r["user_decisions"]), (1, []))

    def test_c_user_decision_bad_input(self):
        e = clean_short(ck(DR, 1, 75), (21, 0, 0, 0))
        e["user_decision_libero_plus"] = "evaluate"
        with self.assertRaisesRegex(P.InputError, 'needs "decided"'):
            run(replace(design(), [e]))
        e.update(user_decision_libero_plus="maybe", decided="2026-11-20")
        with self.assertRaisesRegex(P.InputError, "must be one of"):
            run(replace(design(), [e]))
        e = ck(DR, 1, 75, user_decision_libero_plus="skip", decided="2026-11-20")   # not over the line
        with self.assertRaisesRegex(P.InputError, "only for a checkpoint over the clean 1% line"):
            run(replace(design(), [e]))

    def test_c_cli_exit_0_with_recorded_decision(self):
        with tempfile.TemporaryDirectory() as d:
            inp = os.path.join(d, "i.json")
            e = clean_short(ck(DR, 1, 75), (21, 0, 0, 0))
            e.update(user_decision_libero_plus="skip", decided="2026-11-20, the user: 'skip it'")
            with open(inp, "w") as f:
                json.dump({"checkpoints": replace(design(), [e])}, f)
            p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "pairings_analysis.py"), "--input", inp],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("USER DECISION (recorded)", p.stdout)
            self.assertNotIn("STOP FOR THE USER", p.stdout)

    def test_c_cli_exit_3_on_a_stop(self):
        with tempfile.TemporaryDirectory() as d:
            inp, out = os.path.join(d, "i.json"), os.path.join(d, "o.json")
            with open(inp, "w") as f:
                json.dump({"checkpoints": replace(design(), [clean_short(ck(DR, 1, 75), (21, 0, 0, 0))])}, f)
            p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "pairings_analysis.py"), "--input", inp,
                                "--out", out], capture_output=True, text=True)
            self.assertEqual(p.returncode, 3, p.stderr)
            self.assertIn("STOP FOR THE USER", p.stdout)
            with open(out) as f:
                self.assertEqual(len(json.load(f)["stops_for_user"]), 1)


class BadInput(unittest.TestCase):
    def bad(self, entries, pattern):
        with self.assertRaisesRegex(P.InputError, pattern):
            run(entries)

    def test_missing_entry(self):
        self.bad([x for x in design() if not (x["config"] == DL and x["train_seed"] == 2)], "no entry for")

    def test_whole_config_absent(self):
        self.bad([x for x in design() if x["config"] != L_], "no entry for siglip2 seed 0")

    def test_duplicate(self):
        self.bad(design() + [ck(D, 0, 80)], "listed twice")

    def test_unknown_config(self):
        self.bad(design() + [ck("dinov3+unitok", 0, 80)], "unknown config")

    def test_seed_outside_design(self):
        self.bad(design() + [ck(NONE, 3, 70)], "not in the design")

    def test_bad_status(self):
        e = ck(D, 0, 80)
        e["status"] = "done"
        self.bad(replace(design(), [e]), "status must be")

    def test_missing_without_reason(self):
        self.bad(replace(design(), [{"config": D, "train_seed": 0, "status": "missing"}]), "needs a")

    def test_clean_absent(self):
        e = ck(D, 0, 80)
        del e["clean"]
        self.bad(replace(design(), [e]), "no clean tally")

    def test_clean_over_500(self):
        e = ck(D, 0, 80)
        e["clean"]["libero_goal"]["episodes"] = 501
        self.bad(replace(design(), [e]), "more than the 500")

    def test_lp_counts_do_not_add_up(self):
        e = ck(D, 0, 80)
        e["libero_plus"]["42"]["missing"] = 0
        self.bad(replace(design(), [e]), "!= 10030")

    def test_lp_eval_seed_absent(self):
        e = ck(D, 0, 80)
        del e["libero_plus"]["43"]
        self.bad(replace(design(), [e]), "eval seeds")

    def test_successes_above_evaluated(self):
        e = ck(D, 0, 80)
        e["libero_plus"]["41"]["successes"] = 10001
        self.bad(replace(design(), [e]), "successes")

    def test_non_integer_count(self):
        e = ck(D, 0, 80)
        e["libero_plus"]["41"]["successes"] = 8000.5
        self.bad(replace(design(), [e]), "non-negative integer")

    def test_gate_fail_needs_diagnosis(self):
        self.bad(replace(design(), [ck(D, 0, 80, B=35.8)]), "no gate diagnosis")

    def test_gate_fault_found(self):
        self.bad(replace(design(), [ck(D, 0, 80, B=35.8, gate_diagnosis={"fault_found": True})]), "found a fault")

    def test_not_an_object(self):
        with self.assertRaises(P.InputError):
            P.analyse([])


class Cli(unittest.TestCase):
    def test_cli_writes_outcome_and_refuses_bad_input(self):
        with tempfile.TemporaryDirectory() as d:
            good, bad, out, out2 = (os.path.join(d, n) for n in ("g.json", "b.json", "o.json", "o2.json"))
            with open(good, "w") as f:
                json.dump({"checkpoints": design()}, f)
            with open(bad, "w") as f:
                json.dump({"checkpoints": design()[1:]}, f)
            script = os.path.join(HERE, "pairings_analysis.py")
            p = subprocess.run([sys.executable, "-I", script, "--input", good, "--out", out],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("OUTCOME: H and K", p.stdout)
            with open(out) as f:
                self.assertEqual(json.load(f)["outcome"], "H and K")
            p = subprocess.run([sys.executable, "-I", script, "--input", bad, "--out", out2],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 2)
            self.assertIn("INPUT ERROR", p.stderr)
            self.assertFalse(os.path.exists(out2))


# ---------------------------------------------------------------------------
# One-seed report (PLAN amendment 2026-11-18, item 3)
# ---------------------------------------------------------------------------

EXCLUDED = ("reproduced", "not reproduced", "replicates", "confirms", "consistent with the paper", "significant",
            "within the noise", "beyond the noise", "harmful", "helpful", "worse than each member alone")
CATS = ("Camera Viewpoints", "Sensor Noise")
ONE_DEFAULT = {NONE: 70, D: 80, L_: 78, R_: 78, DL: 81, DR: 75}


def with_cats(e):
    """Split each eval seed's counts over two categories: the first gets half (rounded down) of both."""
    e["libero_plus_categories"] = {}
    for es, t in e["libero_plus"].items():
        k, n = t["successes"], t["evaluated"]
        e["libero_plus_categories"][es] = {CATS[0]: {"successes": k // 2, "evaluated": n // 2},
                                           CATS[1]: {"successes": k - k // 2, "evaluated": n - n // 2}}
    return e


def one(**vals):
    """Six seed 0 entries; vals: config -> y, or a full entry dict (built by the caller)."""
    out = []
    for c in P.CONFIGS:
        v = vals.get(c, ONE_DEFAULT[c])
        out.append(v if isinstance(v, dict) else with_cats(ck(c, 0, v)))
    return out


def run1(entries):
    r = P.analyse_one_seed({"checkpoints": entries})
    return r, P.report_one_seed(r), json.dumps(P._jsonable(r))


def cmp_of(r, a, b):
    return [c for c in r["comparisons"] if c["a"] == a and c["b"] == b][0]


def walk(x):
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from walk(v)


class OneSeedReport(unittest.TestCase):
    def test_full_input_every_element(self):
        es = one(**{D: with_cats(ck(D, 0, None, per_eval=(79, 80, 81), B=92.4))})
        r, text, js = run1(es)
        basis = "one training seed; training-seed noise not measured on this model"
        self.assertEqual(r["basis"], basis)
        self.assertIn(basis, text)
        # per checkpoint
        self.assertEqual([e["config"] for e in r["checkpoints"]], list(P.CONFIGS))
        d = [e for e in r["checkpoints"] if e["config"] == D][0]
        self.assertEqual(d["train_seed"], 0)
        self.assertEqual(d["status"], "ok")
        self.assertEqual(d["basis"], basis)
        self.assertAlmostEqual(d["B_pts"], 92.4)
        self.assertEqual(set(d["clean_suite_rates_pts"]), set(P.CLEAN_SUITES))
        self.assertEqual(d["Y_exact"], "80")
        self.assertAlmostEqual(d["eval_seed_sd_pts"], 1.0)
        self.assertIn("not training-seed noise", d["eval_seed_sd_note"])
        self.assertEqual(sorted(d["lp_rates_pts"]), ["41", "42", "43"])
        self.assertAlmostEqual(d["lp_rates_pts"]["41"], 79.0)
        self.assertAlmostEqual(d["lp_rates_pts"]["43"], 81.0)
        cats = d["lp_per_category_pts"]
        self.assertEqual(sorted(cats["mean_over_eval_seeds"]), sorted(CATS))
        self.assertEqual(sorted(cats["per_eval_seed"]), ["41", "42", "43"])
        k = es[1]["libero_plus_categories"]["41"][CATS[0]]
        self.assertAlmostEqual(cats["per_eval_seed"]["41"][CATS[0]], 100 * k["successes"] / k["evaluated"])
        cov = d["coverage"]
        self.assertEqual((cov["lp_missing_total"], cov["lp_rollouts"]), (90, 30090))
        self.assertEqual((cov["clean_missing_total"], cov["clean_rollouts"]), (0, 2000))
        self.assertTrue(d["gate_G1_pass"])
        self.assertIn("per category (mean over eval seeds): Camera Viewpoints", text)
        self.assertIn("eval-seed SD 1.000", text)
        # seven comparisons with the paper's differences
        want = [("dinov3 - none", "+9.3", 10), ("siglip2 - none", "+8.4", 8), ("c-radiov4 - none", "+8.8", 8),
                ("dinov3+siglip2 - dinov3", "+1.83", 1), ("dinov3+siglip2 - siglip2", "+2.73", 3),
                ("dinov3+c-radiov4 - dinov3", "-5.41", -5), ("dinov3+c-radiov4 - c-radiov4", "-4.91", -3)]
        self.assertEqual([(c["comparison"], c["paper_difference"]) for c in r["comparisons"]],
                         [(n, p) for n, p, _ in want])
        for c, (n, p, d0) in zip(r["comparisons"], want):
            self.assertAlmostEqual(c["d_0_pts"], d0)
            self.assertAlmostEqual(c["paper_difference_pts"], float(Fraction(p)))
            self.assertEqual(c["basis"], basis)
            self.assertEqual(sorted(c["d_per_eval_seed_pts"]), ["41", "42", "43"])
            self.assertEqual(c["label"], "same direction as the paper")
            self.assertIn(n, text)
            self.assertIn(f"paper {p}", text)
        dn = cmp_of(r, D, NONE)["d_per_eval_seed_pts"]
        self.assertEqual([round(dn[s], 6) for s in ("41", "42", "43")], [9.0, 10.0, 11.0])
        # overall line, label count, tests not computed
        ov = r["overall"]["line"]
        self.assertTrue(ov.startswith("the paper's pattern appears at one training seed ("))
        for frag in ("dinov3+c-radiov4 - dinov3: d_0 -5.00, paper -5.41", "dinov3+c-radiov4 - c-radiov4: d_0 -3.00, "
                     "paper -4.91", "dinov3+siglip2 - dinov3: d_0 +1.00, paper +1.83"):
            self.assertIn(frag, ov)
        self.assertIn("OVERALL: " + ov, text)
        self.assertEqual(r["label_counts"], {"same direction as the paper": 7, "opposite direction": 0,
                                             "no difference": 0, "no label": 0})
        self.assertEqual(set(r["decision_rule_tests"]), {"C", "H", "N", "I", "H_L", "K", "below both members"})
        self.assertTrue(all(v.startswith("not computed") for v in r["decision_rule_tests"].values()))
        self.assertIn("not in play", r["outcome_branches_item_8_and_extension_item_7"])
        self.assertIn("decision rule tests", text)
        # nothing that reads as a verdict
        for key in ("outcome", "harm_outcome", "helpful_pairing", "calibration_C_holds", "below_both_members",
                    "tests_three_seeds", "extension", "sigma_hat_three_seeds_pts"):
            self.assertNotIn(key, r)
        self.assertNotIn("OUTCOME", text)

    def test_paper_differences_are_the_arithmetic(self):
        for a, b in P.ONE_SEED_COMPARISONS:
            self.assertEqual(Fraction(cmp_of(run1(one())[0], a, b)["paper_difference"]),
                             P.PAPER_TOTAL[a] - P.PAPER_TOTAL[b])

    def test_labels_three_cases(self):
        r, text, _ = run1(one(**{DL: 79, R_: 75}))      # DL - D = -1 (paper +1.83); DR - R = 0
        self.assertEqual(cmp_of(r, DL, D)["label"], "opposite direction")
        self.assertEqual(cmp_of(r, DR, R_)["label"], "no difference")
        self.assertEqual(cmp_of(r, DR, D)["label"], "same direction as the paper")
        self.assertEqual(cmp_of(r, R_, NONE)["label"], "same direction as the paper")
        self.assertIn("label: opposite direction", text)
        self.assertIn("label: no difference", text)
        # negative paper difference: a negative d_0 is the same direction, a positive one opposite
        r2, _, _ = run1(one(**{DR: 82}))
        self.assertEqual(cmp_of(r2, DR, D)["label"], "opposite direction")
        self.assertEqual(cmp_of(r2, DR, R_)["label"], "opposite direction")
        self.assertEqual(r["label_counts"], {"same direction as the paper": 5, "opposite direction": 1,
                                             "no difference": 1, "no label": 0})

    def test_overall_does_not_appear_names_the_differing(self):
        r, text, _ = run1(one(**{DL: 79, R_: 75}))
        ov = r["overall"]
        self.assertEqual(ov["state"], "does not appear")
        self.assertTrue(ov["line"].startswith("the paper's pattern does not appear at one training seed"))
        self.assertEqual(ov["differing"], ["dinov3+c-radiov4 - c-radiov4", "dinov3+siglip2 - dinov3"])
        self.assertIn("dinov3+c-radiov4 - c-radiov4 (no difference)", ov["line"])
        self.assertIn("dinov3+siglip2 - dinov3 (opposite direction)", ov["line"])
        self.assertNotIn("pattern appears", text)

    def test_overall_appears_only_when_all_three_and_calibration(self):
        appears = "the paper's pattern appears at one training seed"
        r, text, _ = run1(one())
        self.assertIn(appears, text)
        # small sizes still count by sign, and are shown as they are
        r, text, _ = run1(one(**{DL: 80.4, DR: 79.6, R_: 79.8}))
        self.assertEqual(r["overall"]["state"], "appears")
        self.assertIn("dinov3+siglip2 - dinov3: d_0 +0.40, paper +1.83", r["overall"]["line"])
        # calibration not same direction: not read, neither appears nor does not appear
        for dval in (69, 70):
            r, text, _ = run1(one(**{D: dval, DR: dval - 5, DL: dval + 1}))
            self.assertEqual(r["overall"]["state"], "not read")
            self.assertTrue(r["overall"]["line"].startswith(
                "not read: DINOv3 alone is not above the no-encoder model at one training seed ("))
            self.assertNotIn("appear", r["overall"]["line"])
            self.assertIn("paper -5.41", r["overall"]["line"])
        # calibration comparison without a label
        r, text, _ = run1(one(**{NONE: gone(NONE, 0)}))
        self.assertTrue(r["overall"]["line"].startswith(
            "not read: the DINOv3-against-no-encoder comparison has no label ("))
        self.assertNotIn("appear", r["overall"]["line"])
        # a pattern comparison without a label: not available
        r, text, _ = run1(one(**{R_: gone(R_, 0)}))
        self.assertEqual(r["overall"]["state"], "not available")
        self.assertTrue(r["overall"]["line"].startswith("not available"))
        self.assertIn("dinov3+c-radiov4 - c-radiov4: no checkpoint, paper -4.91", r["overall"]["line"])
        self.assertNotIn("appear", r["overall"]["line"])

    def test_coverage_missing_has_numbers_and_no_label(self):
        e = with_cats(ck(DR, 0, 75, miss=(200, 200, 0)))         # 400 of 30,090 missing, over 1%
        r, text, _ = run1(one(**{DR: e}))
        c = [x for x in r["checkpoints"] if x["config"] == DR][0]
        self.assertEqual(c["status"], "counts as missing on coverage")
        self.assertEqual(c["coverage"]["lp_missing_total"], 400)
        self.assertIsNotNone(c["Y_pts"])
        h = cmp_of(r, DR, D)
        self.assertIsNone(h["label"])
        self.assertIsNotNone(h["d_0_pts"])
        self.assertIn("counts as missing on coverage", h["no_label_reason"])
        self.assertEqual(r["overall"]["state"], "not available")
        self.assertEqual(r["label_counts"]["no label"], 2)

    def test_clean_short_stops_for_the_user(self):
        e = ck(D, 0, 80)
        del e["libero_plus"]
        e["clean"]["libero_goal"]["episodes"] = 479               # 21 missing: over the clean 1% line
        r, text, _ = run1(one(**{D: e}))
        self.assertEqual(len(r["stops_for_user"]), 1)
        self.assertIn("STOP FOR THE USER", text)
        self.assertEqual(cmp_of(r, D, NONE)["value_text"], "no LIBERO-Plus result")
        self.assertTrue(r["overall"]["line"].startswith("not read: the DINOv3-against-no-encoder comparison"))

    def test_excluded_words_absent(self):
        e_cov = with_cats(ck(DR, 0, 75, miss=(200, 200, 0)))
        e_short = ck(D, 0, 80)
        del e_short["libero_plus"]
        e_short["clean"]["libero_goal"]["episodes"] = 479
        cases = [one(), one(**{DL: 79, R_: 75}), one(**{D: 69}), one(**{NONE: gone(NONE, 0, "rerun failed")}),
                 one(**{DR: e_cov}), one(**{D: e_short}), one(**{D: with_cats(ck(D, 0, 80, B=35.8, gate_diagnosis={
                     "fault_found": False, "note": "checked"}))})]
        for es in cases:
            _, text, js = run1(es)
            for w in EXCLUDED:
                self.assertNotIn(w, text.lower(), w)
                self.assertNotIn(w, js.lower(), w)

    def test_deferred_seeds_not_recorded_as_missing(self):
        for es in (one(), one(**{DL: gone(DL, 0, "rerun failed")})):
            r, text, js = run1(es)
            self.assertEqual({e["train_seed"] for e in r["checkpoints"]}, {0})
            self.assertEqual(len(r["checkpoints"]), 6)
            self.assertFalse([d for d in walk(P._jsonable(r)) if d.get("status") == "missing"])
            self.assertEqual(r["train_seeds"], [0])
            self.assertIn("deferred", r["deferred_seeds"])
        self.assertEqual([e for e in r["checkpoints"] if e["config"] == DL][0]["status"], "no checkpoint")
        self.assertEqual(cmp_of(r, DL, D)["value_text"], "no checkpoint")
        self.assertIsNone(cmp_of(r, DL, D)["label"])

    def test_one_seed_mode_refuses_seeds_1_and_2(self):
        with self.assertRaisesRegex(P.InputError, "train seed 0 only"):
            P.analyse_one_seed({"checkpoints": design()})
        with self.assertRaisesRegex(P.InputError, "train seed 0 only"):
            P.analyse_one_seed({"checkpoints": one() + [with_cats(ck(D, 1, 80))]})
        with self.assertRaisesRegex(P.InputError, "train seed 0 only"):    # a deferred seed entered as missing
            P.analyse_one_seed({"checkpoints": one() + [gone(D, 1)]})

    def test_locked_rule_refuses_a_one_seed_input(self):
        with self.assertRaisesRegex(P.InputError, "no entry for"):
            P.analyse({"checkpoints": one()})

    def test_one_seed_input_errors(self):
        with self.assertRaisesRegex(P.InputError, "no entry for siglip2 seed 0"):
            P.analyse_one_seed({"checkpoints": [e for e in one() if e["config"] != L_]})
        e = ck(D, 0, 80)
        with self.assertRaisesRegex(P.InputError, "libero_plus_categories"):
            P.analyse_one_seed({"checkpoints": one(**{D: e})})
        e = with_cats(ck(D, 0, 80))
        e["libero_plus_categories"]["42"][CATS[0]]["successes"] += 1
        with self.assertRaisesRegex(P.InputError, "add up"):
            P.analyse_one_seed({"checkpoints": one(**{D: e})})
        with self.assertRaisesRegex(P.InputError, "listed twice"):
            P.analyse_one_seed({"checkpoints": one() + [with_cats(ck(D, 0, 80))]})

    def test_cli_mode_selection(self):
        script = os.path.join(HERE, "pairings_analysis.py")
        with tempfile.TemporaryDirectory() as d:
            one_in, three_in = os.path.join(d, "one.json"), os.path.join(d, "three.json")
            with open(one_in, "w") as f:
                json.dump({"checkpoints": one()}, f)
            with open(three_in, "w") as f:
                json.dump({"checkpoints": design()}, f)
            outs = [os.path.join(d, f"o{i}.json") for i in range(3)]
            p = subprocess.run([sys.executable, "-I", script, "--one-seed", "--input", one_in, "--out", outs[0]],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("the paper's pattern appears at one training seed", p.stdout)
            with open(outs[0]) as f:
                j = json.load(f)
            self.assertEqual(j["overall"]["state"], "appears")
            self.assertEqual(len(j["comparisons"]), 7)
            # a one-seed input without the mode, and a three-seed input with it: refused, nothing written
            p = subprocess.run([sys.executable, "-I", script, "--input", one_in, "--out", outs[1]],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 2)
            self.assertIn("INPUT ERROR", p.stderr)
            self.assertFalse(os.path.exists(outs[1]))
            p = subprocess.run([sys.executable, "-I", script, "--one-seed", "--input", three_in, "--out", outs[2]],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 2)
            self.assertIn("INPUT ERROR", p.stderr)
            self.assertIn("train seed 0 only", p.stderr)
            self.assertFalse(os.path.exists(outs[2]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
