"""Run: /opt/homebrew/bin/python3.11 -I test_tally_proxy_clean.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tally_proxy_clean as T


def rec(suite, ep, succ, es=41, bench="libero", ck="/c/a", seed=0):
    return {"benchmark": bench, "suite": suite, "episode_id": ep, "config": "x",
            "train_seed": seed, "eval_seed": es, "checkpoint": ck, "success": succ}


class Tally(unittest.TestCase):
    def test_average_is_mean_of_suite_rates(self):
        # spatial 1/1, object 0/2, goal 1/2, 10 0/1 -> (1 + 0 + .5 + 0)/4
        recs = [rec("libero_spatial", 1, 1), rec("libero_object", 1, 0), rec("libero_object", 2, 0),
                rec("libero_goal", 1, 1), rec("libero_goal", 2, 0), rec("libero_10", 1, 0)]
        e = T.tally(recs)["checkpoints"][0]["eval_seeds"][0]
        self.assertAlmostEqual(e["four_suite_avg"], 0.375)
        self.assertTrue(e["uneven_counts"])

    def test_libero_plus_ignored(self):
        recs = [rec("libero_spatial", 1, 1), rec("libero_spatial", 2, 1, bench="libero_plus")]
        t = T.tally(recs)
        self.assertEqual(t["n_clean_records"], 1)
        self.assertEqual(t["checkpoints"][0]["eval_seeds"][0]["per_suite"]["libero_spatial"]["n"], 1)

    def test_duplicate_flag_and_grouping(self):
        recs = [rec("libero_spatial", 1, 1), rec("libero_spatial", 1, 1),
                rec("libero_spatial", 1, 0, es=42), rec("libero_spatial", 1, 0, ck="/c/b")]
        t = T.tally(recs)
        self.assertEqual(len(t["checkpoints"]), 2)
        s41 = t["checkpoints"][0]["eval_seeds"][0]
        self.assertEqual(len(s41["duplicates"]), 1)
        self.assertEqual(s41["duplicates"][0]["count"], 2)
        self.assertEqual(t["checkpoints"][0]["eval_seeds"][1]["duplicates"], [])

    def test_missing_suite_gives_none_average(self):
        recs = [rec("libero_spatial", 1, 1), rec("libero_object", 1, 1), rec("libero_goal", 1, 1)]
        e = T.tally(recs)["checkpoints"][0]["eval_seeds"][0]
        self.assertIsNone(e["four_suite_avg"])
        self.assertEqual(e["missing_suites"], ["libero_10"])

    def test_duplicate_inflates_n_but_is_flagged(self):
        # Counting behaviour: duplicates are NOT removed, so n counts both rows.
        recs = [rec("libero_spatial", 1, 1), rec("libero_spatial", 1, 1)]
        e = T.tally(recs)["checkpoints"][0]["eval_seeds"][0]
        self.assertEqual(e["per_suite"]["libero_spatial"]["n"], 2)
        self.assertEqual(len(e["duplicates"]), 1)

    def test_bad_success_raises(self):
        with self.assertRaises(ValueError):
            T.tally([rec("libero_spatial", 1, 2)])

    def test_rates_rounded(self):
        recs = [rec("libero_spatial", i, int(i == 0)) for i in range(3)]
        e = T.tally(recs)["checkpoints"][0]["eval_seeds"][0]
        self.assertEqual(e["per_suite"]["libero_spatial"]["rate"], 0.3333)


if __name__ == "__main__":
    unittest.main()
