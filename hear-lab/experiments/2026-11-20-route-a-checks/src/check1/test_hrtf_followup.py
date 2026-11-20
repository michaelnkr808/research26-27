"""Unit tests for hrtf_followup.py that need neither the RLR library nor any data (argument parsing, azimuth
geometry, the binding's HRTF call guard). Run: python3.11 -m unittest -v test_hrtf_followup   (from this folder)
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import hrtf_followup as hf  # noqa: E402
import pass_tests as pt  # noqa: E402
import rlr_binding as rlr  # noqa: E402

REQ = ["--rlr-root", "r", "--scenes-root", "s", "--parquet", "p.parquet", "--run-dir", "run"]


class HrtfFollowup(unittest.TestCase):
    def test_parse_ok(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "new"
            a = hf.parse_args(REQ + ["--hrtf", "k.sofa", "--out", str(out)])
            self.assertEqual(a.hrtf, Path("k.sofa"))
            self.assertEqual(a.out, out)

    def test_out_must_not_exist(self):
        with tempfile.TemporaryDirectory() as t, self.assertRaises(SystemExit):
            hf.parse_args(REQ + ["--hrtf", "k.sofa", "--out", t])

    def test_hrtf_required_and_sofa(self):
        with self.assertRaises(SystemExit):
            hf.parse_args(REQ + ["--out", "/nonexistent-dir-x/o"])
        with self.assertRaises(SystemExit):
            hf.parse_args(REQ + ["--hrtf", "k.wav", "--out", "/nonexistent-dir-x/o"])

    def test_azimuth_geometry(self):
        for heading in (0.0, 37.0, -120.0):
            lis = np.array([1.0, 2.0, 1.3])
            pos = dict(hf.sweep_positions(lis, heading, 2.0))
            self.assertEqual(list(pos)[0], "az000")
            self.assertEqual(len(pos), 12)
            fwd, left, right, _up = hf.poses.agent_axes(heading)
            self.assertTrue(np.allclose(pos["az000"] - lis, 2.0 * fwd))      # 0 deg: straight ahead
            self.assertTrue(np.allclose(pos["az090"] - lis, 2.0 * left))     # +90 deg: listener's left
            self.assertTrue(np.allclose(pos["az270"] - lis, 2.0 * right))    # 270 deg: listener's right
            self.assertTrue(np.allclose(pos["az180"] - lis, -2.0 * fwd))
            for p in pos.values():
                self.assertAlmostEqual(float(np.linalg.norm(p - lis)), 2.0, places=9)
        # heading 0: left is +y
        self.assertTrue(np.allclose(dict(hf.sweep_positions([0, 0, 0], 0.0, 2.0))["az090"], [0, 2, 0]))

    def test_test_line_labels_and_unchanged_pass_function(self):
        ir = np.zeros((2, 200), dtype=np.float32)
        ir[0, 10] = 1.0
        ir[1, 40] = 1.0
        t = hf.test_line(ir, 44100.0, "kemar/x/y", "az090")
        self.assertTrue(t["pass"])
        self.assertTrue(t["line"].startswith("[test] PASS left_ear_before_right_ear"))
        self.assertTrue(hf.test_line(ir, 44100.0, "r", "az270")["line"].startswith("[info] "))
        self.assertEqual(t["pass"], pt.test_left_before_right(ir, 44100.0, "r", True)["pass"])

    def test_binding_guard_and_signature(self):
        class Dummy(rlr.RLRContext):
            def __init__(self):
                self.log = lambda *a: None
        with self.assertRaises(ValueError):
            Dummy().set_listener_hrtf(0, "")
        with self.assertRaises(ValueError):
            Dummy().set_listener_hrtf(0, None)
        self.assertTrue(hasattr(rlr.RLRContext, "set_listener_hrtf"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
