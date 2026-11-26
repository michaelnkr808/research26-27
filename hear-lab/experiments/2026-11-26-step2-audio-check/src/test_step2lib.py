"""Unit tests for Step 2 (Mac, no RLR library, no simulator, no scene files).

Run from this folder:  /opt/homebrew/bin/python3.11 -m unittest -v test_step2lib

Covers: the side sign of the interaural time difference on synthetic two-ear responses with KEMAR-like delays
(left ear first for a source on the listener's left), several headings, both sides; the azimuth and heading
convention against the scene frame (right-handed, z up: looking along +y, +x is on the right), at both ends, and
through the renderer's look-at rule (a point on the audio's left lands on the image's left half); the heading gate
(camera forward against the direction of travel, and a quaternion read in the wrong order is caught); the RLR listener
quaternion; the counting rule; the one-channel downmix; the looping and the clip synthesis; the loudness
normalisation; the episode pick (balance, mirrored exclusion, seed); the tallies and the bar; the pose export
record maths; the summary and the manifest.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(Path(os.environ.get("HEAR_LAB_SRC") or HERE.parents[2] / "src")))

import step2lib as s2  # noqa: E402
import poses  # noqa: E402

SR = 44100.0
C = 343.0
HEAD_RADIUS = 0.0875
DC = {"onset_rel": 0.1, "pre_ms": 0.25, "post_ms": 2.0, "max_lag_ms": 1.0}


def kemar_like_ir(eye, heading_deg, source, sr=SR, reflections=True, seed=0):
    """A synthetic two-ear response built without step2lib: the ears are placed explicitly, 90 degrees counter-
    clockwise (left) and clockwise (right) of the heading in the z-up scene frame; the nearer ear gets the direct
    sound first; the delay difference follows Woodworth's spherical-head formula (a/c)(theta + sin theta) with
    theta the lateral angle, which gives about 0.66 ms at 90 degrees (MIT KEMAR in route A: 0.73 ms); the far ear is
    quieter and low-passed. Later reflections (from 3 ms after the direct sound) come from random sides."""
    rng = np.random.default_rng(seed)
    eye = np.asarray(eye, float)
    source = np.asarray(source, float)
    h = math.radians(heading_deg)
    ear_l = eye + HEAD_RADIUS * np.array([math.cos(h + math.pi / 2), math.sin(h + math.pi / 2), 0.0])
    ear_r = eye + HEAD_RADIUS * np.array([math.cos(h - math.pi / 2), math.sin(h - math.pi / 2), 0.0])
    dl, dr = np.linalg.norm(source - ear_l), np.linalg.norm(source - ear_r)
    v = source - eye
    lat = math.asin(abs(dl - dr) / (2 * HEAD_RADIUS)) if abs(dl - dr) < 2 * HEAD_RADIUS else math.pi / 2
    itd = HEAD_RADIUS / C * (lat + math.sin(lat))
    t_direct = np.linalg.norm(v) / C
    near_left = dl < dr
    t_l = t_direct + (0.0 if near_left else itd)
    t_r = t_direct + (itd if near_left else 0.0)
    n = int(0.05 * sr)
    ir = np.zeros((2, n))
    kern = np.exp(-np.arange(24) / 4.0) * np.cos(np.arange(24) * 0.9)
    far = np.convolve(kern, np.ones(4) / 4)[:24]
    g_far = 10 ** (-(8.0 * math.sin(lat)) / 20)
    for ch, t, near in ((0, t_l, near_left), (1, t_r, not near_left)):
        i = int(round(t * sr))
        ir[ch, i:i + 24] += (kern if near else far * g_far) / max(np.linalg.norm(v), 0.3)
    if reflections:
        for _ in range(30):
            t = t_direct + 0.003 + rng.uniform(0, 0.03)
            i = int(round(t * sr))
            d = int(rng.integers(-25, 26))
            a = rng.uniform(-0.3, 0.3) / max(np.linalg.norm(v), 0.3)
            ir[0, i:i + 24] += a * kern
            j = min(max(i + d, 0), n - 24)
            ir[1, j:j + 24] += a * kern
    return ir, ("left" if near_left else "right"), itd


class SideSign(unittest.TestCase):
    def test_known_answer_many_headings(self):
        eye = np.array([1.0, -2.0, 1.3])
        n = 0
        for heading in (0.0, 37.0, 90.0, 180.0, 270.0, -135.0):
            fwd = np.array([math.cos(math.radians(heading)), math.sin(math.radians(heading)), 0.0])
            for rel in (30, 60, 90, 120, 150, -30, -60, -90, -120, -150):
                a = math.radians(heading + rel)
                src = eye + 3.0 * np.array([math.cos(a), math.sin(a), 0.0]) + np.array([0, 0, -0.6])
                ir, near, itd = kemar_like_ir(eye, heading, src, seed=n)
                e = s2.estimate_itd(ir, SR, DC["onset_rel"], DC["pre_ms"], DC["post_ms"], DC["max_lag_ms"])
                d = s2.source_direction(eye, fwd, src)
                want = s2.LEFT if rel > 0 else s2.RIGHT
                self.assertEqual(d["true_side"], want, (heading, rel))
                self.assertEqual(near, "left" if rel > 0 else "right", (heading, rel))
                self.assertEqual(e["cue_side"], want, (heading, rel, e))
                # the estimate is the built-in delay to within 3 samples (the far ear's 4-tap smoothing adds
                # 1.5 samples of group delay, and the delays are rounded to whole samples)
                self.assertLessEqual(abs(abs(e["itd_ms"]) - 1000 * itd), 3000.0 / SR + 1e-9, (heading, rel, e, itd))
                n += 1
        self.assertEqual(n, 60)

    def test_left_ear_first_means_positive_itd(self):
        ir = np.zeros((2, 2000))
        ir[0, 500] = 1.0          # left ear (channel 0) first
        ir[1, 530] = 0.5          # right ear 30 samples later
        e = s2.estimate_itd(ir, SR, 0.1, 0.25, 2.0, 1.0)
        self.assertEqual(e["lag_samples"], 30)
        self.assertAlmostEqual(e["itd_ms"], 1000 * 30 / SR)
        self.assertEqual(e["cue_side"], s2.LEFT)
        self.assertGreater(e["ild_db"], 0)
        e2 = s2.estimate_itd(ir[::-1].copy(), SR, 0.1, 0.25, 2.0, 1.0)
        self.assertEqual(e2["cue_side"], s2.RIGHT)
        self.assertEqual(e2["lag_samples"], -30)

    def test_equal_ears_give_no_side(self):
        ir = np.zeros((2, 1000))
        ir[:, 300] = 1.0
        self.assertEqual(s2.estimate_itd(ir, SR, 0.1, 0.25, 2.0, 1.0)["cue_side"], s2.NONE)
        self.assertEqual(s2.estimate_itd(np.zeros((2, 100)), SR, 0.1, 0.25, 2.0, 1.0)["cue_side"], s2.NONE)

    def test_late_opposite_reflection_outside_window_is_ignored(self):
        ir = np.zeros((2, 4000))
        ir[0, 500] = 1.0
        ir[1, 520] = 0.6
        ir[1, 700] = 0.9          # 4.5 ms later, right-first reflection: outside the 2 ms window
        ir[0, 730] = 0.9
        self.assertEqual(s2.estimate_itd(ir, SR, 0.1, 0.25, 2.0, 1.0)["cue_side"], s2.LEFT)

    def test_lag_limit(self):
        ir = np.zeros((2, 4000))
        ir[0, 500] = 1.0
        ir[1, 600] = 1.0          # 2.3 ms: beyond max_lag 1 ms, no positive correlation within range
        e = s2.estimate_itd(ir, SR, 0.1, 0.25, 3.0, 1.0)
        self.assertLessEqual(abs(e["lag_samples"]), round(SR / 1000))


class Frames(unittest.TestCase):
    """InteriorGS frame: right-handed, z up; a viewer looking along +y has +x on the right. Both ends of every axis."""

    def side(self, heading, src, eye=(0.0, 0.0, 1.3)):
        fwd = np.array([math.cos(math.radians(heading)), math.sin(math.radians(heading)), 0.0])
        return s2.source_direction(np.array(eye), fwd, np.array(src))

    def test_facing_back_plus_y(self):
        # standing in the plan facing +y ("back"): +x ("right") is on the listener's right, -x on the left
        self.assertEqual(self.side(90, (2, 0, 1.3))["true_side"], s2.RIGHT)
        self.assertEqual(self.side(90, (-2, 0, 1.3))["true_side"], s2.LEFT)
        self.assertAlmostEqual(self.side(90, (-2, 0, 1.3))["azimuth_deg"], 90.0)
        self.assertAlmostEqual(self.side(90, (2, 0, 1.3))["azimuth_deg"], -90.0)
        self.assertAlmostEqual(self.side(90, (0, 2, 1.3))["azimuth_deg"], 0.0)
        self.assertAlmostEqual(abs(self.side(90, (0, -2, 1.3))["azimuth_deg"]), 180.0)

    def test_facing_plus_x(self):
        self.assertEqual(self.side(0, (0, 2, 1.3))["true_side"], s2.LEFT)
        self.assertEqual(self.side(0, (0, -2, 1.3))["true_side"], s2.RIGHT)

    def test_facing_minus_x_and_minus_y(self):
        self.assertEqual(self.side(180, (0, 2, 1.3))["true_side"], s2.RIGHT)
        self.assertEqual(self.side(180, (0, -2, 1.3))["true_side"], s2.LEFT)
        self.assertEqual(self.side(270, (2, 0, 1.3))["true_side"], s2.LEFT)
        self.assertEqual(self.side(270, (-2, 0, 1.3))["true_side"], s2.RIGHT)

    def test_forward_vector_from_a_frame_map(self):
        # a non-unit forward with a z part (as GS target - eye can be) gives the same answer
        d = s2.source_direction(np.zeros(3), np.array([0.0, 2.014, 0.3]), np.array([-1.0, 0.0, 0.0]))
        self.assertEqual(d["true_side"], s2.LEFT)
        self.assertAlmostEqual(d["azimuth_deg"], 90.0)

    def test_lateral_and_elevation(self):
        d = self.side(0, (math.cos(math.radians(30)), math.sin(math.radians(30)), 1.3))
        self.assertAlmostEqual(d["lateral_deg"], 30.0)
        d = self.side(0, (math.cos(math.radians(150)), math.sin(math.radians(150)), 1.3))
        self.assertAlmostEqual(d["lateral_deg"], 30.0)    # behind-left: same lateral angle as front-left
        d = self.side(0, (0.0, 1.0, 2.3))                   # 45 degrees up on the left
        self.assertAlmostEqual(d["lateral_deg"], 45.0)
        self.assertAlmostEqual(d["elevation_deg"], 45.0)

    def test_counting_rule(self):
        self.assertTrue(s2.is_counted(True, 20.0, 20.0))
        self.assertTrue(s2.is_counted(True, -80.0, 20.0))
        self.assertFalse(s2.is_counted(True, 19.9, 20.0))     # front zone, |azimuth| < 20
        self.assertFalse(s2.is_counted(True, -10.0, 20.0))    # rear zone, |azimuth| > 160 (lateral 10)
        self.assertFalse(s2.is_counted(False, 90.0, 20.0))    # occluded

    def test_rlr_listener_quaternion(self):
        for h in (0.0, 30.0, 90.0, 180.0, 270.0, -45.0):
            fwd = np.array([math.cos(math.radians(h)), math.sin(math.radians(h)), 0.0])
            q = poses.listener_quat_from_forward(fwd * 1.7)
            np.testing.assert_allclose(q, poses.listener_quat_wxyz(h), atol=1e-9)
            m = poses.quat_wxyz_to_matrix(q)
            self.assertAlmostEqual(float(np.linalg.det(m)), 1.0, places=9)
            _f, left, right, up = s2.listener_axes(fwd)
            np.testing.assert_allclose(m @ np.array([1.0, 0, 0]), right, atol=1e-9)   # library local +x = right
            np.testing.assert_allclose(m @ np.array([-1.0, 0, 0]), left, atol=1e-9)   # local -x = left
            np.testing.assert_allclose(m @ np.array([0, 1.0, 0]), up, atol=1e-9)
            np.testing.assert_allclose(m @ np.array([0, 0, 1.0]), -fwd, atol=1e-9)    # local +z = back
            # a source on the world left is at negative local x in the listener's frame
            src_local = m.T @ (2.0 * left)
            self.assertLess(src_local[0], 0)
            src_local = m.T @ (2.0 * right)
            self.assertGreater(src_local[0], 0)

    def test_pose_export_records_under_a_gnbench_style_map(self):
        import export_poses as ep
        # GN-Bench world to GS is x_gs = 2 cx - (a_x x + b_x), y_gs = 2 cy - (a_y y + b_y); with a_x, a_y near -1 it
        # is a shift (route A check 1 scene); with a_x, a_y near +1 a 180 degree turn. Neither mirrors.
        for ax_, ay_ in ((-0.99, -1.0), (1.0, 1.01)):
            to_gs = lambda p: np.array([10.0 - (ax_ * p[0] + 0.3), 4.0 - (ay_ * p[1] - 0.2), p[2]])  # noqa: E731
            from scipy.spatial.transform import Rotation as R
            yaw = 30.0
            rec = ep.frame_records([[1.0, 2.0, 1.3]], [R.from_euler("z", yaw, degrees=True).as_quat()], to_gs)[0]
            self.assertAlmostEqual(rec["world_yaw_deg"], yaw)
            world_left_point = np.array([1.0, 2.0, 1.3]) + 2.0 * np.array([-math.sin(math.radians(yaw)), math.cos(math.radians(yaw)), 0])
            d = s2.source_direction(rec["gs_eye"], rec["gs_forward"], to_gs(world_left_point))
            self.assertEqual(d["true_side"], s2.LEFT)


class CameraAndHeading(unittest.TestCase):
    """PR #95 review, finding 1: the audio's left must be the picture's left, and the camera forward must follow the
    path, whatever format the collector's rotations are in."""

    def test_left_point_lands_on_the_left_half_of_the_image(self):
        import c3lib
        intr = c3lib.camera_intrinsics(480, 360, 70)      # GN-Bench train_seen RGB sensor
        eye = np.array([1.0, -2.0, 1.3])
        for h in range(0, 360, 15):
            f = np.array([math.cos(math.radians(h)), math.sin(math.radians(h)), 0.0])
            _f, left, right, _u = s2.listener_axes(f)
            view = c3lib.look_at_view(eye, eye + f)          # copy of gsplat_renderer._build_look_at_matrix
            for vec, on_left in ((left, True), (right, False)):
                _cam, u, _v = c3lib.project_points((eye + 2 * f + vec)[None, :], view, intr)
                self.assertEqual(float(u[0]) < intr["cx"], on_left, (h, on_left, float(u[0])))

    def _path(self, n=40, step=0.05):
        return [np.array([1.0 + k * step, 2.0, 1.3]) for k in range(n)]

    def test_gate_passes_forward_along_the_path(self):
        import export_poses as ep
        eyes = self._path()
        g = ep.heading_gate(eyes, [[1.0, 0.0, 0.0]] * len(eyes), 0.5, 45.0, 5)
        self.assertTrue(g["ok"])
        self.assertEqual(g["n_gated"], len(eyes) - 5)
        self.assertEqual(g["n_start_blend_skipped"], 5)
        self.assertLess(g["max_angle_deg"], 1e-6)

    def test_gate_skips_start_blend_and_fails_a_sideways_camera(self):
        import export_poses as ep
        eyes = self._path()
        fw = [[0.0, 1.0, 0.0]] * 5 + [[1.0, 0.0, 0.0]] * (len(eyes) - 5)   # start heading sideways: allowed
        self.assertTrue(ep.heading_gate(eyes, fw, 0.5, 45.0, 5)["ok"])
        fw = [[1.0, 0.0, 0.0]] * 20 + [[0.0, 1.0, 0.0]] + [[1.0, 0.0, 0.0]] * (len(eyes) - 21)
        g = ep.heading_gate(eyes, fw, 0.5, 45.0, 5)
        self.assertFalse(g["ok"])
        self.assertEqual(g["n_over_limit"], 1)
        self.assertAlmostEqual(g["max_angle_deg"], 90.0)

    def test_gate_last_frame_and_not_moving(self):
        import export_poses as ep
        eyes = self._path(10) + [self._path(10)[-1]]       # a repeated last point: no motion there
        fw = [[1.0, 0.0, 0.0]] * len(eyes)
        g = ep.heading_gate(eyes, fw, 0.5, 45.0, 0)
        self.assertTrue(g["ok"])
        self.assertEqual(g["n_not_moving"], 1)
        self.assertIsNone(g["angles_deg"][-1])
        self.assertAlmostEqual(g["angles_deg"][-2], 0.0)   # second to last: lookahead clamped, uses the last step

    def test_gate_allows_a_smoothed_corner(self):
        import export_poses as ep
        eyes = [np.array([k * 0.05, 0.0, 1.3]) for k in range(30)] + [np.array([1.45, (k + 1) * 0.05, 1.3]) for k in range(30)]
        # heading as the collector computes it: towards the point 0.5 m ahead (before its yaw smoothing)
        e = np.array(eyes)
        fw = []
        for k in range(len(e)):
            j = min(k + 10, len(e) - 1)
            d = e[j] - e[k] if j > k else e[k] - e[k - 1]
            fw.append(d)
        self.assertTrue(ep.heading_gate(eyes, fw, 0.5, 45.0, 5)["ok"])

    def test_wrong_quaternion_order_is_caught(self):
        import export_poses as ep
        from scipy.spatial.transform import Rotation as R
        yaw = 60.0
        u = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0])
        eyes = [np.array([1.0, 2.0, 1.3]) + 0.05 * k * u for k in range(40)]   # a path heading 60 degrees
        q_xyzw = R.from_euler("z", yaw, degrees=True).as_quat()               # scipy order, as frame_records assumes
        recs = ep.frame_records(eyes, [q_xyzw] * len(eyes), lambda p: np.asarray(p, float))
        self.assertTrue(ep.heading_gate([r["gs_eye"] for r in recs], [r["gs_forward"] for r in recs], 0.5, 45.0, 5)["ok"])
        q_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]                 # the same rotation in w x y z order
        recs = ep.frame_records(eyes, [q_wxyz] * len(eyes), lambda p: np.asarray(p, float))
        g = ep.heading_gate([r["gs_eye"] for r in recs], [r["gs_forward"] for r in recs], 0.5, 45.0, 5)
        self.assertFalse(g["ok"])                                             # read as x y z w, it points elsewhere
        self.assertGreater(g["max_angle_deg"], 45.0)


class Signals(unittest.TestCase):
    def test_downmix_is_the_mean_and_has_no_interaural_difference(self):
        x = np.array([[1.0, 2.0, -3.0], [3.0, 0.0, 1.0]])
        np.testing.assert_allclose(s2.downmix(x), [2.0, 1.0, -1.0])
        rng = np.random.default_rng(1)
        two = rng.standard_normal((2, 3000))
        m = s2.downmix(two)
        e = s2.estimate_itd(np.stack([m, m]), SR, 0.1, 0.25, 2.0, 1.0)
        self.assertEqual(e["lag_samples"], 0)
        self.assertAlmostEqual(e["ild_db"], 0.0)
        with self.assertRaises(ValueError):
            s2.downmix(np.zeros((3, 4)))

    def test_looping(self):
        src = np.arange(10.0)
        np.testing.assert_array_equal(s2.looped(src, 8, 5), [8, 9, 0, 1, 2])
        np.testing.assert_array_equal(s2.looped(src, -3, 4), [7, 8, 9, 0])
        np.testing.assert_array_equal(s2.looped(src, 25, 3), [5, 6, 7])

    def test_synthesis_with_a_delta_is_the_looped_source(self):
        rng = np.random.default_rng(2)
        src = rng.standard_normal(1000)
        delta = np.zeros((2, 50))
        delta[:, 0] = 1.0
        out = s2.synthesize(src, [delta] * 7, 300, 123, 100)
        self.assertEqual(out.shape, (2, 7 * 300))
        np.testing.assert_allclose(out[0], s2.looped(src, 123, 2100), atol=1e-9)
        np.testing.assert_allclose(out[1], out[0], atol=1e-12)

    def test_synthesis_delay_uses_the_sound_before_the_episode(self):
        rng = np.random.default_rng(3)
        src = rng.standard_normal(700)
        h = np.zeros((1, 40))
        h[0, 25] = 0.5
        out = s2.synthesize(src, [h, h, h], 200, 0, 50)
        np.testing.assert_allclose(out[0], 0.5 * s2.looped(src, -25, 600), atol=1e-9)

    def test_crossfade(self):
        src = np.ones(400)
        a = np.zeros((1, 5)); a[0, 0] = 1.0
        b = np.zeros((1, 5)); b[0, 0] = 3.0
        out = s2.synthesize(src, [a, b], 100, 0, 50)[0]
        np.testing.assert_allclose(out[:100], 1.0)
        w = np.arange(51) / 50
        np.testing.assert_allclose(out[100:151], 1.0 * (1 - w) + 3.0 * w)
        np.testing.assert_allclose(out[151:], 3.0)
        with self.assertRaises(ValueError):
            s2.synthesize(src, [a], 10, 0, 11)

    def test_normalise_and_wav_conversion(self):
        x = np.sin(np.linspace(0, 200, 44100))
        y, g = s2.normalise_rms(x, -14.0)
        self.assertAlmostEqual(s2.rms_dbfs(y), -14.0, places=9)
        pcm = (np.array([0, 16384, -32768, 32767])).astype(np.int16)
        np.testing.assert_allclose(s2.to_float_mono(pcm), [0, 0.5, -1.0, 32767 / 32768])
        st = np.array([[0.5, -0.5], [1.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(s2.to_float_mono(st), [0.0, 0.5])
        with self.assertRaises(ValueError):
            s2.normalise_rms(np.zeros(10), -14.0)

    def test_pad_irs(self):
        p, lens = s2.pad_irs([np.ones((2, 3)), np.ones((2, 5))])
        self.assertEqual(p.shape, (2, 2, 5))
        np.testing.assert_array_equal(lens, [3, 5])
        self.assertEqual(float(p[0, :, 3:].sum()), 0.0)


def _scene_line(scene, identity, mx=0, my=0, mxy=0):
    return json.dumps({"scene": scene, "frame_diag": {"identity": {"total": identity}, "mirror_x": {"total": mx},
                                                       "mirror_y": {"total": my}, "mirror_xy": {"total": mxy}}})


def _rows(spec):
    out = []
    for i, (scene, traj, vis, status) in enumerate(spec):
        out.append({"split": "InteriorGS_train_seen", "row_index": str(i), "trajectory_id": str(traj), "scene": scene,
                    "status": status, "visible_primary": vis, "depth_tol_primary_m": "0.15"})
    return out


class EpisodePick(unittest.TestCase):
    def test_mirrored_rule(self):
        lines = [_scene_line("A", 10, 9), _scene_line("B", 100, 101), _scene_line("C", 5, 5),  # tie -> identity
                 _scene_line("D", 0, 0), _scene_line("E", 3, 0, 0, 4), "", _scene_line("F", 1, 0, 2)]
        self.assertEqual(s2.mirrored_scenes(lines), {"B", "E", "F"})

    def test_split_balance_and_exclusion(self):
        spec = [("S1_1", 1, "1", "ok"), ("S1_1", 2, "0", "ok"), ("S2_2", 3, "1", "ok"), ("S2_2", 4, "0", "ok"),
                ("S3_3", 5, "1", "ok"), ("S3_3", 6, "0", "ok"), ("S3_3", 7, "0", "ok"), ("S4_4", 8, "1", "missing_id"),
                ("S1_1", 9, "0", "ok"), ("S1_1", 10, "1", "ok")]
        vis, nvis = s2.split_by_visibility(_rows(spec), {"S2_2"}, "InteriorGS_train_seen", "visible_primary",
                                           "depth_tol_primary_m", 0.15)
        self.assertEqual(vis, ["S1_1_1", "S3_3_5", "S1_1_10"])
        self.assertEqual(nvis, ["S1_1_2", "S3_3_6", "S3_3_7", "S1_1_9"])
        order = s2.frozen_order(vis, nvis, 7)
        self.assertEqual(len(order), 6)
        vset = set(vis)
        for i, x in enumerate(order):
            self.assertEqual(x in vset, i % 2 == 0)
        self.assertNotIn("S2_2_3", order)
        self.assertEqual(len(set(order)), 6)
        self.assertEqual(order, s2.frozen_order(vis, nvis, 7))           # reproducible
        many_v = [f"V_{i}" for i in range(40)]
        many_n = [f"N_{i}" for i in range(60)]
        self.assertNotEqual(s2.frozen_order(many_v, many_n, 1), s2.frozen_order(many_v, many_n, 2))
        self.assertEqual(s2.scene_of("0754_840890_138"), "0754_840890")

    def test_wrong_split_or_tolerance_refused(self):
        rows = _rows([("A_1", 1, "1", "ok")])
        rows[0]["split"] = "InteriorGS_test_unseen"
        with self.assertRaises(ValueError):
            s2.split_by_visibility(rows, set(), "InteriorGS_train_seen", "visible_primary", "depth_tol_primary_m", 0.15)
        rows = _rows([("A_1", 1, "1", "ok")])
        rows[0]["depth_tol_primary_m"] = "0.3"
        with self.assertRaises(ValueError):
            s2.split_by_visibility(rows, set(), "InteriorGS_train_seen", "visible_primary", "depth_tol_primary_m", 0.15)

    def test_pick_script_end_to_end(self):
        import pick_episodes as pe
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            spec = [(f"S{i % 4}_x", i, "1" if i % 3 == 0 else "0", "ok") for i in range(30)]
            rows = _rows(spec)
            with open(td / "c3.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            (td / "scenes.jsonl").write_text("\n".join([_scene_line("S0_x", 1, 5), _scene_line("S1_x", 5),
                                                          _scene_line("S2_x", 5), _scene_line("S3_x", 5)]) + "\n")
            sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
            in_m = [r for r in rows if r["scene"] == "S0_x"]
            elig = [r for r in rows if r["scene"] != "S0_x"]
            nv = sum(r["visible_primary"] == "1" for r in elig)
            cfg = {"episodes": {"source_csv": str(td / "c3.csv"), "scenes_jsonl": str(td / "scenes.jsonl"),
                                "source_csv_sha256": sha(td / "c3.csv"), "scenes_jsonl_sha256": sha(td / "scenes.jsonl"),
                                "split": "InteriorGS_train_seen", "visible_column": "visible_primary",
                                "tolerance_column": "depth_tol_primary_m", "tolerance_m": 0.15, "order_seed": 5,
                                "order_length": 2 * min(nv, len(elig) - nv),
                                "expected": {"n_rows": 30, "n_mirrored_scenes": 1, "n_rows_in_mirrored_scenes": len(in_m),
                                             "n_visible_in_mirrored_scenes": sum(r["visible_primary"] == "1" for r in in_m),
                                             "n_eligible_visible": nv, "n_eligible_not_visible": len(elig) - nv}}}
            order, counts, mirrored = pe.build(cfg, exp=td)
            self.assertEqual(mirrored, {"S0_x"})
            self.assertEqual(len(order), cfg["episodes"]["order_length"])
            self.assertTrue(all(not x.startswith("S0_x") for x in order))
            self.assertEqual(order, pe.build(cfg, exp=td)[0])
            cfg["episodes"]["expected"]["n_rows"] = 31
            with self.assertRaises(SystemExit):
                pe.build(cfg, exp=td)

    def test_committed_order_is_balanced_and_matches_the_config(self):
        import yaml
        cfg = yaml.safe_load((HERE.parent / "config.yaml").read_text())
        text = (HERE.parent / cfg["episodes"]["order_file"]).read_text()
        self.assertEqual(s2.sha256_text(text), cfg["episodes"]["order_sha256"])
        ids = text.split()
        self.assertEqual(len(ids), cfg["episodes"]["order_length"])
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(cfg["episodes"]["batch1_n"] % 2, 0)       # even prefixes are balanced
        self.assertEqual(cfg["episodes"]["full_set_n"] % 2, 0)


class RunGate(unittest.TestCase):
    def test_pending_values_block_the_run(self):
        import render_audio as ra
        import yaml
        cfg = yaml.safe_load((HERE.parent / "config.yaml").read_text())
        pend = ra.pending_values(cfg)
        # the sound hashes were filled in from the lead's download; the lead's confirmation is still pending
        self.assertEqual(pend, ["direction_check.proposals_confirmed_by_lead"])
        self.assertEqual(ra.pending_values({"a": {"b": "x"}, "c": 1}), [])
        with tempfile.TemporaryDirectory() as td:
            rc = ra.main(["--n", "1", "--poses-dir", td, "--scenes-root", td, "--rlr-root", td, "--hrtf", td,
                          "--sound", td, "--out", str(Path(td) / "out")])
            self.assertEqual(rc, 2)
            self.assertFalse((Path(td) / "out").exists())


class TallyAndBar(unittest.TestCase):
    def step(self, counted, correct, los=True, cue=1, true=1):
        return {"counted": counted, "correct": correct, "los": los, "cue_side": cue, "true_side": true}

    def test_tally_and_verdict(self):
        steps = [self.step(True, True)] * 9 + [self.step(True, False, cue=-1)] + [self.step(False, False, los=False, cue=-1)] * 3
        t = s2.tally(steps)
        self.assertEqual((t["n_steps"], t["n_counted"], t["n_correct"], t["n_occluded"]), (13, 10, 9, 3))
        self.assertEqual(t["n_occluded_correct"], 0)
        self.assertEqual(s2.verdict(t, 0.9, 10), "PASS")      # exactly 90 percent passes
        self.assertEqual(s2.verdict(t, 0.91, 10), "FAIL")
        self.assertEqual(s2.verdict(t, 0.9, 11), "TOO_FEW")

    def test_summary(self):
        import summarize as sm
        import yaml
        cfg = yaml.safe_load((HERE.parent / "config.yaml").read_text())
        base = {"n_frames": 10, "video_fps": 10.0, "n_los": 10, "n_occluded": 0, "n_occluded_with_side": 0,
                "n_occluded_correct": 0, "n_counted_no_cue": 0,
                "timing_s": {"episode_total": 2.0, "simulate_total": 1.0, "scene_load_mesh": 0.2,
                             "scene_load_context": 0.1, "pose_export_render": 3.0}}
        recs = [dict(base, episode_id="A_1_1", scene="A_1", order_position=0, visible_start=True, n_steps=10,
                     n_counted=150, n_correct=140, share_correct=140 / 150, occluded_share_correct=None),
                dict(base, episode_id="0001_839920_5", scene="0001_839920", order_position=1, visible_start=False,
                     n_steps=10, n_counted=60, n_correct=50, share_correct=50 / 60, occluded_share_correct=None)]
        out = sm.summarize(cfg, recs, usd_per_hour=0.5, wall_s=36.0, expected_ids=["A_1_1", "0001_839920_5"])
        self.assertEqual(out["pooled"]["n_counted"], 210)
        self.assertEqual(out["verdict"], "PASS")           # 190 / 210 = 0.905
        part = sm.summarize(cfg, recs, expected_ids=["A_1_1", "0001_839920_5", "B_2_9"])
        self.assertEqual(part["verdict"], "INCOMPLETE")    # a failed episode: no pass or fail for the batch
        self.assertEqual(part["episodes_missing"], ["B_2_9"])
        self.assertEqual(sm.summarize(cfg, recs)["verdict"], "INCOMPLETE")   # planned set unknown
        self.assertEqual(out["low_wall_share_scenes_in_set"], ["0001_839920"])
        self.assertEqual(out["low_wall_share_scenes"]["n_counted"], 60)
        self.assertEqual(out["visible_starts"]["n_correct"], 140)
        self.assertAlmostEqual(out["timing"]["usd_per_episode_audio_wall"], 0.5 * 36 / 3600 / 2)
        with tempfile.TemporaryDirectory() as td:
            sm.write_csv(recs, Path(td) / "t.csv")
            with open(Path(td) / "t.csv", newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 2)
            self.assertNotIn("source_gs", rows[0])


class Manifest(unittest.TestCase):
    def test_roundtrip(self):
        import make_manifest as mm
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "a").mkdir()
            (td / "a" / "x.txt").write_text("hello")
            (td / "y.bin").write_bytes(b"\x00\x01")
            self.assertEqual(mm.main([str(td)]), 0)
            self.assertEqual(mm.check(td), [])
            (td / "y.bin").write_bytes(b"\x00\x02")
            (td / "z").write_text("new")
            bad = mm.check(td)
            self.assertIn("differs y.bin", bad)
            self.assertIn("not listed z", bad)


if __name__ == "__main__":
    unittest.main()
