"""Smoke test of render_audio.py, summarize.py and draw_topdown.py on the Mac, with a stand-in for the RLR library.

The RLR library runs only on Linux x64, so this test replaces rlr_binding.RLRContext with a small fake that
returns the synthetic KEMAR-like two-ear responses of test_step2lib.kemar_like_ir for the listener pose it was
given (it reads the heading back from the listener quaternion). The scene is a made-up one-room floor plan
written by the test (not InteriorGS data). This checks the code path end to end (poses file in, records, clips,
pictures, summary out) and that the quaternion, the true side and the cue side agree through it. It says nothing
about the real library's output; the preflight on the rented machine does that.

Run from this folder:  /opt/homebrew/bin/python3.11 -m unittest -v test_render_smoke
"""
from __future__ import annotations

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

import poses  # noqa: E402
import rlr_binding  # noqa: E402
import step2lib as s2  # noqa: E402
from test_step2lib import kemar_like_ir  # noqa: E402

import yaml  # noqa: E402


class FakeContext:
    """Stand-in for rlr_binding.RLRContext (same methods render_audio.py and its preflight call)."""

    def __init__(self, lib, overrides, log=print):
        self.config_used = {"hrtfRight": [1.0, 0, 0], "hrtfUp": [0, 1.0, 0], "hrtfBack": [0, 0, 1.0], **overrides}
        self.src, self.pose, self.hrtf, self.n_sim = None, None, None, 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        pass

    def add_mesh_object(self, v, f, cat):
        assert cat is None, "materials off: no categories"
        self.n_faces = len(f)
        return 0

    def add_source(self, p):
        self.src = np.asarray(p, float)
        return 0

    def add_listener(self, layout, p, q):
        assert layout == "binaural"
        self.pose = (np.asarray(p, float), np.asarray(q, float))
        return 0

    def set_listener_pose(self, li, p, q):
        self.pose = (np.asarray(p, float), np.asarray(q, float))

    def set_listener_hrtf(self, li, path):
        self.hrtf = str(path)

    def clear_sources(self):
        self.src = None

    def clear_listeners(self):
        self.pose = None

    def simulate(self):
        assert self.hrtf, "HRTF must be set"
        self.n_sim += 1
        return 0.001

    def get_ir(self, li, si):
        eye, q = self.pose
        m = poses.quat_wxyz_to_matrix(q)
        fwd = -(m @ np.array([0.0, 0.0, 1.0]))          # local +z is back
        heading = math.degrees(math.atan2(fwd[1], fwd[0]))
        ir, _near, _itd = kemar_like_ir(eye, heading, self.src, seed=self.n_sim)
        return ir.astype(np.float32)

    def segment_blocked(self, a, b, end_margin_m=0.0):
        return False


def write_scene(root: Path):
    """A made-up 8 m x 8 m room with one target box near the middle."""
    sd = root / "SCN_000001"
    sd.mkdir(parents=True)
    walls = []
    corners = [(0, 0), (8, 0), (8, 8), (0, 8)]
    for i in range(4):
        walls.append({"location": [list(corners[i]), list(corners[(i + 1) % 4])], "thickness": 0.1, "height": 2.8})
    (sd / "structure.json").write_text(json.dumps({"walls": walls, "holes": [],
                                                   "rooms": [{"profile": [list(c) for c in corners], "room_type": "x"}]}))
    bb = [{"x": 4.0, "y": 4.0, "z": 0.0}, {"x": 4.4, "y": 4.0, "z": 0.0}, {"x": 4.4, "y": 4.4, "z": 0.0},
          {"x": 4.0, "y": 4.4, "z": 0.0}, {"x": 4.0, "y": 4.0, "z": 0.8}, {"x": 4.4, "y": 4.0, "z": 0.8},
          {"x": 4.4, "y": 4.4, "z": 0.8}, {"x": 4.0, "y": 4.4, "z": 0.8}]
    (sd / "labels.json").write_text(json.dumps([{"ins_id": "7", "label": "thing", "bounding_box": bb}]))
    from PIL import Image
    Image.fromarray(np.full((160, 160), 255, dtype=np.uint8)).save(sd / "occupancy.png")
    return sd


def write_poses(pdir: Path, eid: str, n: int):
    """A path along y = 3.4 from x = 0.5 to x = 7.9 heading +x, ears 0.9 m above the box centre: the box centre (4.2, 4.2) is first nearly straight
    ahead (front zone, not counted), then to the left (counted), then nearly straight behind (rear zone, not
    counted)."""
    frames = []
    for k in range(n):
        x = 0.5 + 7.4 * k / (n - 1)
        frames.append({"frame": k, "gs_eye": [x, 3.4, 1.3], "gs_forward": [1.0, 0.0, 0.0]})
    p = {"episode_id": eid, "scene": "SCN_000001", "trajectory_id": eid.rsplit("_", 1)[1], "target_ins_id": "7",
         "goal_gs_xy": [3.0, 3.0], "listener_height_m": 1.3, "video_fps": 10.0, "n_frames": n, "n_poses": n,
         "error": None, "occupancy_meta": {"left": 0.0, "top": 8.0, "scale": 0.05, "width": 160, "height": 160},
         "render_s": 0.5, "frames": frames}
    (pdir / "poses").mkdir(parents=True, exist_ok=True)
    (pdir / "poses" / f"{eid}.json").write_text(json.dumps(p))


class RenderSmoke(unittest.TestCase):
    def test_episode_summary_picture(self):
        import draw_topdown
        import render_audio as ra
        import summarize as sm

        cfg = yaml.safe_load((HERE.parent / "config.yaml").read_text())
        orig_ctx, orig_lib = rlr_binding.RLRContext, ra._lib
        rlr_binding.RLRContext = FakeContext
        ra._lib = lambda root: None
        try:
            with tempfile.TemporaryDirectory() as td:
                td = Path(td)
                scenes = td / "scenes"
                write_scene(scenes)
                rlr_root = td / "rlr"
                (rlr_root / "RLRAudioPropagationPkg/data").mkdir(parents=True)
                (rlr_root / "RLRAudioPropagationPkg/data/mp3d_material_config.json").write_text(json.dumps(
                    {"materials": [{"name": "Default", "labels": ["default"]}, {"name": "Wall", "labels": ["wall"]},
                                   {"name": "Floor", "labels": ["floor", "ceiling"]}]}))
                eid = "SCN_000001_42"
                write_poses(td / "export", eid, 25)
                sr = int(cfg["rlr"]["sample_rate_hz"])
                rng = np.random.default_rng(0)
                source = rng.standard_normal(sr) * 0.1
                pf = ra.preflight(cfg, rlr_root, td / "x.sofa")
                self.assertTrue(pf["pass"], pf)
                self.assertEqual(len(pf["cases"]), 8)
                out = td / "run"
                out.mkdir()
                job = {"cfg": cfg, "episode_id": eid, "order_position": 0, "out": str(out),
                       "poses_file": str(td / "export" / "poses" / f"{eid}.json"), "scenes_root": str(scenes),
                       "rlr_root": str(rlr_root), "hrtf": str(td / "x.sofa"), "source": source}
                rec = ra.render_episode(job)
                self.assertEqual(rec["n_steps"], 25)
                self.assertTrue(rec["visible_start"])
                self.assertGreater(rec["n_counted"], 5)
                self.assertLess(rec["n_counted"], 25)                        # front and rear zones dropped
                self.assertEqual(rec["n_correct"], rec["n_counted"])      # the fake head is never wrong
                self.assertEqual(rec["clip_duration_s"], 2.5)               # 25 frames at 10 fps
                for k in ("source_gs", "goal_to_box_horiz_m", "mesh_report"):
                    self.assertNotIn(k, rec)                                 # box-derived values stay local
                steps = [json.loads(l) for l in (out / "episodes" / eid / "steps.jsonl").read_text().splitlines()]
                self.assertTrue(all(s["true_side"] == s2.LEFT for s in steps))
                self.assertFalse(steps[0]["counted"])                         # front zone
                self.assertFalse(steps[-1]["counted"])                        # rear zone
                self.assertTrue(steps[12]["counted"])
                from scipy.io import wavfile
                r2, two = wavfile.read(str(out / "clips" / f"{eid}_two_ear.wav"))
                r1, one = wavfile.read(str(out / "clips" / f"{eid}_one_channel.wav"))
                self.assertEqual((r1, r2), (sr, sr))
                self.assertEqual(two.shape, (25 * sr // 10, 2))
                np.testing.assert_allclose(one, two.mean(axis=1), atol=1e-6)
                self.assertEqual(np.load(out / "episodes" / eid / "brirs.npy").shape[0], 25)
                (out / "episode_records.jsonl").write_text(json.dumps(rec) + "\n")
                (out / "timing.json").write_text(json.dumps({"wall_s_all_episodes": 3.0, "failed": {}}))
                self.assertEqual(sm.main(["--run-dir", str(out), "--usd-per-hour", "0.2"]), 0)
                summ = json.loads((out / "summary.json").read_text())
                self.assertEqual(summ["verdict"], "TOO_FEW")                 # far fewer than 200 counted steps
                png = draw_topdown.draw(eid, out, td / "export", scenes)
                self.assertTrue(png.is_file())
        finally:
            rlr_binding.RLRContext, ra._lib = orig_ctx, orig_lib


if __name__ == "__main__":
    unittest.main()
