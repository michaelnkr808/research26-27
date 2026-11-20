"""Local unit tests for check 3 (no simulator, no GPU).

Run from anywhere:
    C3_TEST_UNSEEN_PARQUET=/path/to/InteriorGS_test_unseen.parquet \
        python3.11 -m unittest -v test_c3lib   (from this folder)

Covers the projection and visibility rule on a synthetic camera and box, box
surface sampling, the episode selection (test_unseen count from the parquet;
the seeded train_seen sample code path, exercised on the test_unseen row
count because the train_seen parquet is not on this machine), the review
choice, the resume helpers and a summary smoke test. The simulator part of
run_check3.py cannot run here (needs Linux, CUDA, gsplat and InteriorGS
splat files).
"""
from __future__ import annotations

import csv
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c3lib  # noqa: E402

CONFIG = HERE.parent.parent / "config.yaml"


def axis_box(centre, size):
    c = np.asarray(centre, dtype=float)
    h = np.asarray(size, dtype=float) / 2
    return np.array([[c[0] + sx * h[0], c[1] + sy * h[1], c[2] + sz * h[2]]
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


class Camera(unittest.TestCase):
    def setUp(self):
        self.cfg = c3lib.load_config(CONFIG)
        self.intr = c3lib.camera_intrinsics(self.cfg["width"], self.cfg["height"], self.cfg["hfov_deg"])
        self.eye = np.array([0.0, 0.0, 1.3])
        self.view = c3lib.look_at_view(self.eye, self.eye + np.array([1.0, 0.0, 0.0]))
        self.tols = c3lib.tolerances(self.cfg)
        H, W = self.cfg["height"], self.cfg["width"]
        self.far = np.full((H, W), 50.0)
        self.valid = np.zeros((H, W), dtype=bool)

    def test_config_values(self):
        self.assertEqual((self.cfg["width"], self.cfg["height"], self.cfg["hfov_deg"]), (480, 360, 70.0))
        self.assertEqual(self.tols, {"primary": 0.15, "strict": 0.05, "loose": 0.30})
        self.assertEqual(self.cfg["spacing_m"], 0.05)

    def test_intrinsics_match_gn_bench(self):
        fovy = math.radians(70)
        self.assertAlmostEqual(self.intr["fy"], 360 / (2 * math.tan(fovy / 2)))
        fovx = 2 * math.atan(math.tan(fovy / 2) * 480 / 360)
        self.assertAlmostEqual(self.intr["fx"], 480 / (2 * math.tan(fovx / 2)))
        # square pixels follow from the simulator's fovx formula
        self.assertAlmostEqual(self.intr["fx"], self.intr["fy"])
        self.assertAlmostEqual(self.intr["fovy_deg"], 70.0)
        self.assertGreater(self.intr["fovx_deg"], 85.0)
        self.assertLess(self.intr["fovx_deg"], 87.0)

    def test_axes(self):
        _, u, v = c3lib.project_points(np.array([[5.0, 0.0, 1.3], [5.0, -1.0, 1.3], [5.0, 0.0, 2.3]]),
                                       self.view, self.intr)
        self.assertAlmostEqual(u[0], 240.0)
        self.assertAlmostEqual(v[0], 180.0)
        self.assertGreater(u[1], 240.0)   # world -y is image right when looking along +x
        self.assertLess(v[2], 180.0)      # up is image up

    def test_heading(self):
        np.testing.assert_allclose(c3lib.heading_forward(90.0), [0.0, 1.0, 0.0], atol=1e-12)

    def run_box(self, centre, size, depth):
        pts = c3lib.sample_box_surface(axis_box(centre, size), self.cfg["spacing_m"])
        return c3lib.evaluate_visibility(pts, self.view, self.intr, depth, self.valid, self.tols, "z")

    def test_box_ahead_seen(self):
        r = self.run_box([4.0, 0.0, 1.3], [1.0, 1.0, 1.0], self.far)
        self.assertTrue(r["visible_primary"])
        self.assertEqual(r["n_in_image"], r["n_box_points"])
        self.assertEqual(r["n_seen_primary"], r["n_box_points"])  # nothing nearer than 50 m

    def test_box_behind_not_seen(self):
        r = self.run_box([-4.0, 0.0, 1.3], [1.0, 1.0, 1.0], self.far)
        self.assertEqual(r["n_front"], 0)
        self.assertFalse(r["visible_primary"] or r["visible_strict"] or r["visible_loose"])

    def test_box_outside_fov_not_seen(self):
        r = self.run_box([4.0, 10.0, 1.3], [1.0, 1.0, 1.0], self.far)  # about 68 deg to the left
        self.assertEqual(r["n_front"], r["n_box_points"])
        self.assertEqual(r["n_in_image"], 0)
        self.assertFalse(r["visible_loose"])

    def test_box_occluded_not_seen(self):
        wall = np.full_like(self.far, 2.0)  # a nearer surface everywhere
        r = self.run_box([4.0, 0.0, 1.3], [1.0, 1.0, 1.0], wall)
        self.assertGreater(r["n_in_image"], 0)
        self.assertFalse(r["visible_primary"] or r["visible_strict"] or r["visible_loose"])

    def test_within_tolerance(self):
        # front face of the box at x = 3.5 m (z-depth 3.5); render says 3.4
        depth = np.full_like(self.far, 3.4)
        r = self.run_box([4.0, 0.0, 1.3], [1.0, 1.0, 1.0], depth)
        self.assertTrue(r["visible_primary"])    # 3.5 <= 3.4 + 0.15
        self.assertFalse(r["visible_strict"])    # 3.5 >  3.4 + 0.05
        self.assertTrue(r["visible_loose"])
        # only the front face is within 0.15 m (back face is 1 m further)
        self.assertLess(r["n_seen_primary"], r["n_in_image"])
        self.assertGreater(r["share_seen_loose"], 0.0)

    def test_partial_occlusion(self):
        depth = self.far.copy()
        depth[:, :240] = 2.0  # left half of the image blocked
        r = self.run_box([4.0, 0.0, 1.3], [1.0, 1.0, 1.0], depth)
        self.assertTrue(r["visible_primary"])
        seen_u = r["u"][r["seen_primary"]]
        self.assertTrue(np.all(seen_u >= 240))

    def test_invalid_depth_not_seen(self):
        invalid = np.ones_like(self.valid)
        pts = c3lib.sample_box_surface(axis_box([4.0, 0.0, 1.3], [1.0, 1.0, 1.0]), 0.05)
        r = c3lib.evaluate_visibility(pts, self.view, self.intr, self.far, invalid, self.tols, "z")
        self.assertFalse(r["visible_loose"])
        self.assertEqual(r["n_invalid_depth"], r["n_in_image"])

    def test_ray_depth_kind(self):
        # point off-axis: z-depth 4, ray distance larger; a ray-depth image of exactly
        # the ray distance must see it, a z-depth image of 4.0 read as ray must not at 0.05
        pts = np.array([[4.0, -2.0, 1.3]])
        ray = math.sqrt(16 + 4)
        H, W = self.far.shape
        r = c3lib.evaluate_visibility(pts, self.view, self.intr, np.full((H, W), ray), self.valid,
                                      {"t": 0.0}, "ray")
        self.assertTrue(r["visible_t"])
        r = c3lib.evaluate_visibility(pts, self.view, self.intr, np.full((H, W), 4.0), self.valid,
                                      {"t": 0.05}, "ray")
        self.assertFalse(r["visible_t"])


class Boxes(unittest.TestCase):
    def test_sampling_spacing_and_surface(self):
        corners = axis_box([1.0, 2.0, 0.5], [0.33, 1.2, 1.0])
        pts = c3lib.sample_box_surface(corners, 0.05)
        lo, hi = corners.min(0), corners.max(0)
        on_face = np.any(np.isclose(pts, lo) | np.isclose(pts, hi), axis=1)
        self.assertTrue(np.all(on_face))
        self.assertTrue(np.all((pts >= lo - 1e-9) & (pts <= hi + 1e-9)))
        # grid step along x on the bottom face is <= 5 cm
        xs = np.unique(np.round(pts[np.isclose(pts[:, 2], lo[2])][:, 0], 9))
        self.assertLessEqual(np.diff(xs).max(), 0.05 + 1e-9)
        self.assertEqual(len(np.unique(np.round(pts, 6), axis=0)), len(pts))

    def test_rotated_and_shuffled_box(self):
        a = math.radians(30)
        Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        corners = axis_box([0, 0, 0], [2.0, 0.5, 1.0]) @ Rz.T + np.array([3.0, 1.0, 0.5])
        corners = corners[np.random.default_rng(0).permutation(8)]
        o, edges = c3lib.box_frame(corners)
        self.assertEqual(sorted(np.round(np.linalg.norm(edges, axis=1), 6)), [0.5, 1.0, 2.0])
        inside = c3lib.points_in_box(np.array([[3.0, 1.0, 0.5], [6.0, 1.0, 0.5]]), corners)
        self.assertEqual(list(inside), [True, False])

    def test_point_box_distance(self):
        corners = axis_box([0, 0, 0], [2.0, 2.0, 2.0])
        self.assertAlmostEqual(c3lib.point_box_distance([0, 0, 0], corners), 0.0)
        self.assertAlmostEqual(c3lib.point_box_distance([4, 0, 0], corners), 3.0)
        self.assertAlmostEqual(c3lib.point_box_distance([4, 5, 0], corners), 5.0)
        flat = axis_box([0, 0, 0], [2.0, 2.0, 0.0])
        self.assertAlmostEqual(c3lib.point_box_distance([0, 0, 3], flat), 3.0)

    def test_flat_box(self):
        corners = axis_box([0, 0, 0.0], [1.0, 1.0, 0.0])
        pts = c3lib.sample_box_surface(corners, 0.05)
        self.assertEqual(len(pts), 21 * 21)

    def test_real_labels_file_parses(self):
        p = os.environ.get("C3_LABELS_JSON")
        if not p:
            self.fail("set C3_LABELS_JSON to an InteriorGS labels.json for this test")
        import json
        labels = json.loads(Path(p).read_text())
        n_box = 0
        for e in labels:
            c = c3lib.box_corners(e)
            if c is None:
                continue
            c3lib.box_frame(c)  # raises if not a box
            n_box += 1
        self.assertGreater(n_box, 0)


class Selection(unittest.TestCase):
    def setUp(self):
        self.cfg = c3lib.load_config(CONFIG)
        p = os.environ.get("C3_TEST_UNSEEN_PARQUET")
        if not p:
            self.fail("set C3_TEST_UNSEEN_PARQUET to the local InteriorGS_test_unseen.parquet")
        import pyarrow.parquet as pq
        self.n_rows = pq.ParquetFile(p).metadata.num_rows

    def test_test_unseen_all(self):
        self.assertEqual(self.cfg["splits"]["InteriorGS_test_unseen"], "all")
        rows = c3lib.select_rows(self.n_rows, "all", self.cfg["seed"])
        self.assertEqual(len(rows), self.n_rows)
        print(f"\n[selection] InteriorGS_test_unseen rows in local parquet: {self.n_rows}")

    def test_seeded_sample_code_path(self):
        n = self.cfg["splits"]["InteriorGS_train_seen"]
        self.assertEqual(n, 1000)
        a = c3lib.select_rows(self.n_rows, n, self.cfg["seed"])
        b = c3lib.select_rows(self.n_rows, n, self.cfg["seed"])
        c = c3lib.select_rows(self.n_rows, n, self.cfg["seed"] + 1)
        self.assertEqual(len(a), 1000)
        self.assertEqual(len(np.unique(a)), 1000)
        self.assertTrue(np.all(np.diff(a) > 0))
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))
        expect = np.sort(np.random.default_rng(20261120).choice(self.n_rows, 1000, replace=False))
        self.assertTrue(np.array_equal(a, expect))
        with self.assertRaises(ValueError):
            c3lib.select_rows(10, 1000, 1)

    def test_review_candidates(self):
        rows = c3lib.select_rows(self.n_rows, "all", self.cfg["seed"])
        r1 = c3lib.review_candidates(rows, 10, self.cfg["seed"])
        r2 = c3lib.review_candidates(list(reversed(rows)), 10, self.cfg["seed"])
        self.assertEqual(r1, r2)
        self.assertEqual(len(r1), 10 * c3lib.REVIEW_OVERSAMPLE)
        self.assertEqual(len(set(r1)), len(r1))


class CsvAndSummary(unittest.TestCase):
    def test_resume_and_summary(self):
        import summarize_check3 as S
        cfg = c3lib.load_config(CONFIG)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            p = out / "check3_X.csv"
            with open(p, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=c3lib.CSV_FIELDS)
                w.writeheader()
                base = {k: "" for k in c3lib.CSV_FIELDS}
                for i, (st, vis, near) in enumerate([("ok", 1, 1), ("ok", 0, 0), ("missing_id", "", 1)]):
                    r = dict(base, split="X", row_index=i, trajectory_id=i, scene="s", label_ins_id="1",
                             label_label="chair", status=st, n_label_entries=1 if st == "ok" else 0,
                             within_radius_path=near, within_radius_straight=near, path_length_m=2.0,
                             depth_tol_primary_m=0.15, depth_tol_strict_m=0.05, depth_tol_loose_m=0.3,
                             n_invalid_depth=0 if st == "ok" else "", render_s=0.1 if st == "ok" else "")
                    for t in ("primary", "strict", "loose"):
                        r[f"visible_{t}"] = vis
                    w.writerow(r)
            with open(p, "a") as fh:
                fh.write("X,3,3,s,1,chair,ok")  # torn line from a crash
            self.assertEqual(c3lib.done_keys(p), {("X", 0), ("X", 1), ("X", 2)})
            c3lib.repair_csv_tail(p)
            self.assertTrue(p.read_bytes().endswith(b"\n"))
            self.assertEqual(len(c3lib.read_rows(p)), 3)
            s = S.split_summary(c3lib.read_rows(p), [0, 1, 2, 3], cfg)
            self.assertEqual(s["n_selected"], 4)
            self.assertEqual(s["n_not_done"], 1)
            self.assertEqual(s["primary"]["visible"], 1)
            self.assertEqual(s["primary"]["not_visible"], 1)
            self.assertEqual(s["primary"]["within_radius_path"], {"visible": 1, "not_visible": 0})
            self.assertEqual(s["missing"]["n"], 1)
            # review drawing on a synthetic frame
            H, W = 360, 480
            np.savez_compressed(out / "f.npz", rgb=np.zeros((H, W, 3), np.uint8),
                                depth=np.full((H, W), 100.0, np.float32), u=np.array([10.0, 20.0]),
                                v=np.array([10.0, 20.0]), in_image=np.array([True, True]),
                                seen_primary=np.array([True, False]))
            rec = dict(c3lib.read_rows(p)[0], n_seen_primary=1, n_box_points=2, n_in_image=2)
            S.draw_review(out / "f.npz", rec, out / "r.png")
            self.assertTrue((out / "r.png").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
