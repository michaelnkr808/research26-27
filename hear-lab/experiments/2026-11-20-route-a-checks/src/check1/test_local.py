"""Local unit test for check 1 (runs on the Mac, no RLR library needed).

Exports one real InteriorGS scene to triangles and checks counts, closure, the bounding box and that
every face has a material; runs the overlay on it and prints the wall share; checks the ctypes struct
sizes against the header layout and the listener quaternion. The RLR library itself (ctypes calls)
cannot run here: it ships Linux x64 binaries only.

Environment (all required):
  CHECK1_SCENE_DIR  InteriorGS/<scene> folder with structure.json, labels.json, occupancy.json, occupancy.png
  CHECK1_PARQUET    GN-Matrix InteriorGS_test_unseen.parquet
  CHECK1_RLR_ROOT   rlr-audio-propagation checkout (only its material JSON is read here)
  CHECK1_OUT        folder for the overlay image written by the test (outside the repo)
Run: python3.11 -m unittest -v test_local   (from this folder)
"""
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(Path(os.environ.get("HEAR_LAB_SRC") or HERE.parents[3] / "src")))  # shared code in hear-lab/src (moved 2026-11-26)

import geometry  # noqa: E402
import gnbench_copy as gb  # noqa: E402
import overlay  # noqa: E402
import pass_tests as pt  # noqa: E402
import poses  # noqa: E402
import rlr_binding as rlr  # noqa: E402
from run_check1 import choose_episode  # noqa: E402


def _env(name):
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"set {name} (see the module docstring)")
    return Path(v)


class Check1Local(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas as pd
        cls.scene_dir = _env("CHECK1_SCENE_DIR")
        cls.out = _env("CHECK1_OUT")
        cls.mat = _env("CHECK1_RLR_ROOT") / "RLRAudioPropagationPkg/data/mp3d_material_config.json"
        cls.cfg_all = yaml.safe_load((HERE.parent.parent / "config.yaml").read_text())
        cls.cfg = cls.cfg_all["check1"]
        cls.df = pd.read_parquet(_env("CHECK1_PARQUET"))
        cls.rows = overlay.scene_rows(cls.df, cls.scene_dir.name)
        cls.row = choose_episode(cls.rows, int(cls.cfg_all["seed"]))
        cls.meta = gb.load_occupancy_metadata(cls.scene_dir)
        cls.affine = gb.episode_affine(cls.row, cls.meta)
        cls.structure, cls.labels = geometry.load_scene_files(cls.scene_dir)
        cls.mesh = geometry.build_scene_mesh(
            cls.scene_dir, cls.mat, lambda p: gb.gs_points_to_world(p, cls.affine, cls.meta),
            exclude_ins_ids=[str(cls.row["label_ins_id"])])
        print(f"\n[setup] scene {cls.scene_dir.name}, episode trajectory_id {cls.row['trajectory_id']} "
              f"(goal {cls.row['label_ins_id']} {cls.row['label_label']!r}), affine {tuple(round(x, 5) for x in cls.affine)}")

    def test_counts(self):
        m = self.mesh
        kinds = {k: sum(1 for p in m.parts if p.kind == k) for k in ("wall", "floor", "ceiling", "box")}
        goal = str(self.row["label_ins_id"])
        n_boxes = sum(1 for e in self.labels if "bounding_box" in e
                      and str(e["label"]).strip().lower() not in geometry.STRUCTURAL_LABELS
                      and str(e["ins_id"]) != goal)
        print(f"[counts] parts {kinds}; {m.report['n_vertices']} vertices, {m.report['n_faces']} triangles; "
              f"holes {m.report['holes_per_type']} (unassigned {m.report['holes_unassigned']}); "
              f"labels skipped {m.report['labels_skipped']}")
        self.assertEqual(kinds["wall"], len(self.structure["walls"]))
        self.assertEqual(kinds["floor"], len(self.structure["rooms"]))
        self.assertEqual(kinds["ceiling"], len(self.structure["rooms"]))
        self.assertEqual(kinds["box"] + m.report["labels_skipped"]["degenerate"], n_boxes)
        self.assertEqual(m.report["holes_unassigned"], [])
        self.assertEqual(len(m.faces), m.report["n_faces"])
        self.assertTrue(np.all(m.faces < len(m.vertices)))
        v = m.vertices[m.faces.astype(np.int64)]
        area = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
        self.assertGreater(float(area.min()), 1e-10, "degenerate triangle")
        # 6 boxes faces x 2 triangles
        for p in m.parts:
            if p.kind == "box":
                self.assertEqual(p.face_end - p.face_start, 12)

    def test_every_face_has_material(self):
        m = self.mesh
        names = {mm["name"] for mm in json.loads(self.mat.read_text())["materials"]}
        mt = geometry.MaterialTable(self.mat)
        self.assertEqual(len(m.face_category), len(m.faces))
        self.assertEqual(len(m.face_material), len(m.faces))
        for cat, name in zip(m.face_category, m.face_material):
            self.assertTrue(cat)
            self.assertIn(name, names)
            self.assertEqual(mt.match(cat).material_name, name, "category passed must map to the same material")
        print(f"[materials] faces per material {geometry._count(m.face_material)}")
        # the rule itself
        self.assertEqual(mt.match("high chair").material_name, "wood, Thick")
        self.assertEqual(mt.match("floor").material_name, "Carpet")
        self.assertEqual(mt.match("wall").material_name, "Gypsum Board")
        self.assertEqual(mt.match("fruit").material_name, "Default")
        geometry.write_material_map_csv(m, self.out / "assumed_materials.csv")

    def test_closed_and_bbox(self):
        m = self.mesh
        for p in m.parts:
            if p.closed_expected:
                self.assertTrue(geometry.part_is_closed(m, p), f"{p.kind} {p.ident} not closed")
        lo, hi = m.vertices.min(0), m.vertices.max(0)
        self.assertTrue(np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)))
        self.assertTrue(np.all(hi - lo > 0.5), f"degenerate bbox {lo} {hi}")
        ztop = max(float(w["height"]) for w in self.structure["walls"])
        self.assertLess(abs(float(lo[2])), 0.01, "mesh should start at the floor, z=0")
        # floors at z=0 and ceilings at the top wall height close the room vertically
        for p in m.parts:
            if p.kind in ("floor", "ceiling"):
                z = m.vertices[m.faces[p.face_start:p.face_end].astype(np.int64)][..., 2]
                self.assertTrue(np.allclose(z, 0.0 if p.kind == "floor" else ztop, atol=1e-5))
        # walls' outer footprint encloses every floor vertex (floors lie inside the walled area)
        wall_v = np.concatenate([m.vertices[m.faces[p.face_start:p.face_end].astype(np.int64)].reshape(-1, 3)
                                 for p in m.parts if p.kind == "wall"])
        floor_v = np.concatenate([m.vertices[m.faces[p.face_start:p.face_end].astype(np.int64)].reshape(-1, 3)
                                  for p in m.parts if p.kind == "floor"])
        self.assertTrue(np.all(floor_v[:, :2] >= wall_v[:, :2].min(0) - 1e-6))
        self.assertTrue(np.all(floor_v[:, :2] <= wall_v[:, :2].max(0) + 1e-6))
        print(f"[bbox] world-frame mesh bbox min {np.round(lo, 3).tolist()} max {np.round(hi, 3).tolist()}; "
              f"walls closed {sum(geometry.part_is_closed(m, p) for p in m.parts if p.kind == 'wall')}/"
              f"{sum(p.kind == 'wall' for p in m.parts)}, boxes closed "
              f"{sum(geometry.part_is_closed(m, p) for p in m.parts if p.kind == 'box')}/{sum(p.kind == 'box' for p in m.parts)}")

    def test_frame_start_on_free_floor(self):
        from PIL import Image
        img = np.array(Image.open(self.scene_dir / "occupancy.png"))
        H, W = img.shape
        for r in self.rows.to_dict("records"):
            x, y = gb.transform_from_world_to_pixel(self.meta, W, H, (r["start_world_x"], r["start_world_y"]))
            self.assertGreaterEqual(int(img[y, x]), gb.PASSABLE_THRESHOLD)
        # GS -> world -> GS round trip through the copied GN-Bench functions
        p = np.array([[1.0, -2.0, 0.5], [5.0, 1.0, 2.0]])
        back = gb.world_points_to_gs(gb.gs_points_to_world(p, self.affine, self.meta), self.affine, self.meta)
        self.assertTrue(np.allclose(back, p, atol=1e-5))

    def test_overlay_runs(self):
        s = overlay.run_overlay(self.scene_dir, self.rows, self.cfg, self.out, affine_row=self.row)
        overlay.print_overlay_summary(s)
        self.assertTrue((self.out / f"{self.scene_dir.name}_overlay.png").exists())
        self.assertGreater(s["n_wall_points"], 0)
        self.assertTrue(0.0 <= s["wall_share_occupied"] <= 1.0)

    def test_struct_sizes_and_pose(self):
        self.assertEqual(rlr.struct_sizes(), rlr.EXPECTED_SIZES)
        for hdeg in (0.0, 37.0, -120.0):
            q = poses.listener_quat_wxyz(hdeg)
            M = poses.quat_wxyz_to_matrix(q)
            fwd, left, right, up = poses.agent_axes(hdeg)
            self.assertTrue(np.allclose(M @ [0, 0, -1], fwd, atol=1e-9))  # local forward is -back
            self.assertTrue(np.allclose(M @ [1, 0, 0], right, atol=1e-9))
            self.assertTrue(np.allclose(M @ [0, 1, 0], up, atol=1e-9))
            self.assertAlmostEqual(float(np.linalg.det(M)), 1.0, places=9)
        self.assertTrue(np.allclose(poses.agent_axes(0.0)[1], [0, 1, 0]))  # heading 0: left is +y

    def test_pass_test_functions(self):
        sr, c = 44100.0, 343.0
        ir = np.zeros((1, 4000), dtype=np.float32)
        d = 3.0
        ir[0, int(round(d / c * sr))] = 1.0
        ir[0, 3000] = 0.3
        t = pt.test_first_arrival(ir, d, sr, c, 1.0, "synthetic", True)
        print("[synthetic]", t["line"])
        self.assertTrue(t["pass"])
        b = np.zeros((2, 2000), dtype=np.float32)
        b[0, 100] = 1.0
        b[1, 126] = 0.7
        t = pt.test_left_before_right(b, sr, "synthetic", True)
        print("[synthetic]", t["line"])
        self.assertTrue(t["pass"])
        b[[0, 1]] = b[[1, 0]]
        self.assertFalse(pt.test_left_before_right(b, sr, "synthetic", True)["pass"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
