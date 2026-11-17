"""Run: /opt/homebrew/bin/python3.11 -I test_tally_per_task.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tally_per_task as T


def rec(suite, task, trial, succ, steps, name="do x", bench="libero"):
    return {"benchmark": bench, "suite": suite, "episode_id": task * 1000 + trial, "success": succ,
            "steps": steps, "instruction": name, "checkpoint": "/c/a", "eval_seed": 41}


def task0(t, suite="libero_spatial"):
    return t["suites"][suite]["tasks"][0]


class PerTask(unittest.TestCase):
    def test_counts_rate_and_means(self):
        recs = [rec("libero_spatial", 0, 0, 1, 100), rec("libero_spatial", 0, 1, 1, 200),
                rec("libero_spatial", 0, 2, 0, 300)]
        x = task0(T.tally(recs))
        self.assertEqual((x["succ"], x["n"], x["rate"]), (2, 3, 0.6667))
        self.assertEqual(x["mean_steps_all"], 200.0)
        self.assertEqual(x["mean_steps_success"], 150.0)

    def test_task_id_from_episode_id(self):
        recs = [rec("libero_goal", 3, 7, 1, 10, name="a"), rec("libero_goal", 4, 0, 0, 20, name="b")]
        tasks = T.tally(recs)["suites"]["libero_goal"]["tasks"]
        self.assertEqual([(x["task_id"], x["task_name"]) for x in tasks], [(3, "a"), (4, "b")])

    def test_no_successes_gives_none_success_mean(self):
        x = task0(T.tally([rec("libero_10", 0, 0, 0, 50)]), "libero_10")
        self.assertIsNone(x["mean_steps_success"])
        self.assertEqual(x["mean_steps_all"], 50.0)

    def test_suite_and_total_sums(self):
        recs = [rec("libero_spatial", 0, 0, 1, 1), rec("libero_spatial", 1, 0, 0, 1),
                rec("libero_object", 0, 0, 1, 1)]
        t = T.tally(recs)
        self.assertEqual((t["suites"]["libero_spatial"]["succ"], t["suites"]["libero_spatial"]["n"]), (1, 2))
        self.assertEqual((t["total"]["succ"], t["total"]["n"]), (2, 3))
        self.assertEqual(t["missing_suites"], ["libero_goal", "libero_10"])

    def test_libero_plus_ignored(self):
        recs = [rec("libero_spatial", 0, 0, 1, 1), rec("libero_spatial", 0, 1, 1, 1, bench="libero_plus")]
        t = T.tally(recs)
        self.assertEqual(t["n_clean_records"], 1)
        self.assertEqual(task0(t)["n"], 1)

    def test_duplicates_flagged_not_removed(self):
        recs = [rec("libero_spatial", 0, 0, 1, 1), rec("libero_spatial", 0, 0, 1, 1)]
        x = task0(T.tally(recs))
        self.assertEqual(x["n"], 2)
        self.assertEqual(x["duplicates"], [{"episode_id": 0, "count": 2}])

    def test_conflicting_names_flagged(self):
        recs = [rec("libero_spatial", 0, 0, 1, 1, name="a"), rec("libero_spatial", 0, 1, 1, 1, name="b")]
        x = task0(T.tally(recs))
        self.assertIsNone(x["task_name"])
        self.assertEqual(x["names_seen"], ["a", "b"])

    def test_bad_success_raises(self):
        with self.assertRaises(ValueError):
            T.tally([rec("libero_spatial", 0, 0, 2, 1)])


if __name__ == "__main__":
    unittest.main()
