"""Synthetic checks for pairings_analysis.py: every outcome branch of the locked stage 2 rule, the extension, and
missing or incomplete input. Run: /opt/homebrew/bin/python3.11 -I test_pairings_analysis.py

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
        self.assertEqual((r["outcome"], r["harm_outcome"], r["helpful_pairing"], r["calibration_C_holds"]),
                         (label, harm, helpful, c))

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
                         ("C fails, under C-a", "I", "unresolved"))


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
        self.assertEqual(r["harm_outcome"], "H")

    def test_missing_helpful_seed_is_unresolved(self):
        r = run(replace(design(), [gone(DL, 1)]))
        self.assertEqual((r["outcome"], r["helpful_pairing"]), ("H, not K, not H_L", "unresolved"))
        r = run(replace(design(dr=80), [gone(DL, 1)]))
        self.assertEqual(r["outcome"], "N, not K")

    def test_missing_dinov3_seed(self):
        r = run(replace(design(), [gone(D, 2)]))
        self.assertEqual((r["outcome"], r["harm_outcome"], r["helpful_pairing"], r["extension"]),
                         ("C fails, under C-a", "I", "unresolved", P.EXT_MISSING))

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

    def test_C_fails_with_three_seed_I_is_left_to_the_user(self):
        r = run(self.base(none=79))
        self.assertEqual((r["outcome"], r["extension"]), ("C fails, under C-a", P.EXT_USER))
        r = run(ext(self.base(none=79)))          # the user said yes: analysed, label stays
        self.assertEqual(r["outcome"], "C fails, under C-a")
        self.assertEqual(r["harm_outcome"], "H")
        self.assertTrue(r["extension"].startswith(P.EXT_DONE))


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

    def test_clean_short(self):
        e = ck(D, 0, 80)
        e["clean"]["libero_goal"]["episodes"] = 499
        self.bad(replace(design(), [e]), "need 500")

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
