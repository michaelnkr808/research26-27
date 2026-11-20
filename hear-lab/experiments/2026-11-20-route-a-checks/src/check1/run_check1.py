"""Check 1 runner (PLAN.md, check 1). Runs on the rented Linux x64 machine; see RUN.md.

Steps: pick one GN-Matrix episode of --scene (numpy default_rng(config seed) over the scene's rows in
trajectory_id order); export the scene mesh in GN-Bench's world frame with that episode's affine; for
each settings set in config.yaml check1.settings, create an RLR context with those fields set
explicitly, write a mono and a binaural impulse response from the goal box centre to the start pose,
run the pass tests; then the floor plan overlay and the label_ins_id lookup for four scenes.
Everything is written under --out, which must not exist yet (results are never overwritten).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import geometry  # noqa: E402
import gnbench_copy as gb  # noqa: E402
import overlay  # noqa: E402
import pass_tests as pt  # noqa: E402
import poses  # noqa: E402
import rlr_binding as rlr  # noqa: E402

LOS_END_MARGIN_M = 0.01  # ray for the line-of-sight test stops 1 cm short of each end


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> dict:
    def g(*a):
        try:
            return subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as exc:  # noqa: BLE001
            return f"unavailable: {exc!r}"
    return {"commit": g("rev-parse", "HEAD"), "dirty_files": g("status", "--porcelain")}


class Log:
    def __init__(self, path: Path):
        self.fh = open(path, "a", buffering=1)

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.fh.write(s + "\n")


def choose_episode(rows: pd.DataFrame, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    i = int(rng.integers(len(rows)))
    return rows.iloc[i].to_dict()


def box_centre_gs(labels: list, ins_id: str):
    for e in labels:
        if str(e.get("ins_id")) == str(ins_id):
            if "bounding_box" not in e:
                return None, e
            return geometry.box_corners(e).mean(axis=0), e
    return None, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=HERE.parent.parent / "config.yaml")
    ap.add_argument("--rlr-root", type=Path, required=True, help="checkout of rlr-audio-propagation at the pinned commit")
    ap.add_argument("--scenes-root", type=Path, required=True, help="folder holding InteriorGS/<scene>/ folders")
    ap.add_argument("--parquet", type=Path, required=True, help="GN-Matrix InteriorGS_test_unseen.parquet")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-audio", action="store_true",
                    help="stop before loading the RLR library (geometry, poses and overlay only); for a dry run")
    a = ap.parse_args(argv)

    cfg_all = yaml.safe_load(a.config.read_text())
    cfg = cfg_all["check1"]
    seed = int(cfg_all["seed"])
    a.out.mkdir(parents=True, exist_ok=False)
    log = Log(a.out / "check1.log")
    lib_path = a.rlr_root / "RLRAudioPropagationPkg/libs/linux/x64/libRLRAudioPropagation.so"
    mat_path = a.rlr_root / "RLRAudioPropagationPkg/data/mp3d_material_config.json"
    scene_dir = a.scenes_root / a.scene
    sr = float(cfg["sample_rate_hz"])
    c = float(cfg["speed_of_sound_m_s"])
    all_tests = []

    info = {
        "argv": sys.argv, "config_path": str(a.config), "config_sha256": sha256(a.config),
        "config": cfg_all, "git": git_info(), "repo_commit_env": os.environ.get("REPO_COMMIT"), "python": sys.version, "numpy": np.__version__,
        "pandas": pd.__version__, "platform": platform.platform(), "machine": platform.machine(),
        "cpu_count": os.cpu_count(), "rlr_library": str(lib_path), "rlr_library_sha256": sha256(lib_path),
        "material_config": str(mat_path), "material_config_sha256": sha256(mat_path),
        "parquet": str(a.parquet), "parquet_sha256": sha256(a.parquet),
        "scene_files_sha256": {f: sha256(scene_dir / f) for f in
                               ("structure.json", "labels.json", "occupancy.json", "occupancy.png")},
        "struct_sizes": rlr.struct_sizes(), "struct_sizes_expected": rlr.EXPECTED_SIZES,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if info["struct_sizes"] != rlr.EXPECTED_SIZES:
        log("ABORT: ctypes struct sizes differ from the header layout", info["struct_sizes"])
        return 2

    # ---- episode, frame, mesh
    df = pd.read_parquet(a.parquet)
    rows = overlay.scene_rows(df, a.scene)
    row = choose_episode(rows, seed)
    meta = gb.load_occupancy_metadata(scene_dir)
    affine = gb.episode_affine(row, meta)
    info["episode"] = {k: row[k] for k in ("split", "trajectory_id", "scene", "label_ins_id", "label_label",
                                           "start_world_x", "start_world_y", "start_facing_heading_degrees",
                                           "goal_world_x", "goal_world_y", "instruction")}
    info["affine"] = list(map(float, affine))
    info["gnbench_copy_check"] = gb.verify_against_gnbench(scene_dir, row)
    log("[episode]", json.dumps(info["episode"], default=str))
    log("[frame] affine", info["affine"], "| copy-vs-GN-Bench check:", info["gnbench_copy_check"])

    structure, labels = geometry.load_scene_files(scene_dir)
    centre_gs, entry = box_centre_gs(labels, row["label_ins_id"])
    if centre_gs is None:
        log(f"FAIL source_placement: label_ins_id {row['label_ins_id']} "
            f"{'has no bounding_box' if entry else 'not in labels.json'}")
        (a.out / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
        return 1
    to_world = lambda p: gb.gs_points_to_world(p, affine, meta)  # noqa: E731
    source_pos = to_world(centre_gs[None, :])[0]
    heading = float(row["start_facing_heading_degrees"])
    h = float(cfg["listener_height_m"])
    start_pos = np.array([row["start_world_x"], row["start_world_y"], h])
    mesh = geometry.build_scene_mesh(scene_dir, mat_path, to_world, exclude_ins_ids=[str(row["label_ins_id"])])
    geometry.write_material_map_csv(mesh, a.out / "assumed_materials.csv")
    closed = {k: sum(1 for p in mesh.parts if p.kind == k and geometry.part_is_closed(mesh, p))
              for k in ("wall", "box")}
    mesh_report = dict(mesh.report, parts={k: sum(1 for p in mesh.parts if p.kind == k)
                                           for k in ("wall", "floor", "ceiling", "box")},
                       closed_parts=closed, bbox_min=mesh.vertices.min(0).tolist(),
                       bbox_max=mesh.vertices.max(0).tolist(),
                       faces_per_material=geometry._count(mesh.face_material),
                       goal_box_excluded=str(row["label_ins_id"]))
    (a.out / "mesh_report.json").write_text(json.dumps(mesh_report, indent=2, default=str))
    info["source_world"] = source_pos.tolist()
    info["listener_world"] = start_pos.tolist()
    info["source_listener_distance_m"] = float(np.linalg.norm(source_pos - start_pos))
    info["source_inside_other_boxes"] = geometry.boxes_containing(centre_gs, labels, [row["label_ins_id"]])
    if info["source_inside_other_boxes"]:
        log(f"[warn] source point lies inside other labels.json boxes: {info['source_inside_other_boxes']}")
    log(f"[mesh] {mesh.report['n_vertices']} vertices, {mesh.report['n_faces']} triangles, parts {mesh_report['parts']}")
    log(f"[pose] source (goal box centre, world) {np.round(source_pos, 3).tolist()}; listener {np.round(start_pos, 3).tolist()}, "
        f"heading {heading:.2f} deg; distance {info['source_listener_distance_m']:.3f} m; goal_world "
        f"({row['goal_world_x']:.3f}, {row['goal_world_y']:.3f})")

    occ_img = np.array(Image.open(scene_dir / "occupancy.png"))
    H_, W_ = occ_img.shape

    def free_floor(p) -> bool:
        x, y = gb.transform_from_world_to_pixel(meta, W_, H_, p)
        return bool(occ_img[y, x] >= gb.PASSABLE_THRESHOLD)

    path_pts = [np.array([x, y, h]) for x, y in zip(row["path_raster_world_x"], row["path_raster_world_y"])]

    # ---- the two settings runs
    run_records = {}
    settings = {} if a.no_audio else cfg["settings"]
    if a.no_audio:
        log("[rlr] --no-audio: library not loaded, no impulse responses (dry run)")
        info["no_audio"] = True
    else:
        lib = rlr.load_library(lib_path)
        log("[rlr] library loaded:", lib_path)
    for run_name, overrides in settings.items():
        rdir = a.out / run_name
        rdir.mkdir()
        ov = dict(overrides)
        ov["sampleRate"] = sr
        rec = {"overrides": ov}
        with rlr.RLRContext(lib, ov, log=log) as ctx:
            rec["config_defaults"] = ctx.config_defaults
            rec["config_used"] = ctx.config_used
            q = poses.listener_quat_wxyz(heading, ctx.config_used)
            ctx.set_material_database(mat_path)
            ctx.add_mesh_object(mesh.vertices, mesh.faces, mesh.face_category)
            if run_name == list(cfg["settings"].keys())[0]:
                ctx.write_scene_obj(a.out / "scene_mesh_from_library.obj")
            src = ctx.add_source(source_pos)
            rec["irs"] = {}
            for layout, nch in (("mono", 1), ("binaural", 2)):
                li = ctx.add_listener(layout, start_pos, q)
                wall = ctx.simulate()
                ir = ctx.get_ir(li, src)
                ctx.write_ir_wave(li, src, rdir / f"ir_{layout}.wav")
                ctx.write_ir_metrics(li, src, rdir / f"ir_{layout}_metrics.txt")
                np.save(rdir / f"ir_{layout}.npy", ir)
                t = pt.test_basic(ir, nch, f"ir_{layout}_basic", run_name)
                all_tests.append(t)
                log(t["line"])
                rec["irs"][layout] = {"wall_time_s": wall, "thread_count": ctx.config_used["threadCount"],
                                      "channels": ir.shape[0], "samples": ir.shape[1],
                                      "indirect_ray_efficiency": ctx.indirect_ray_efficiency(),
                                      "ir_count": ctx.ir_count()}
                log(f"[timing] {run_name} {layout}: {wall:.3f} s wall, threadCount {ctx.config_used['threadCount']}")
                ctx.clear_listeners()

            # first arrival vs distance, line of sight
            blocked = ctx.segment_blocked(start_pos, source_pos, LOS_END_MARGIN_M)
            los_pos, los_from = (start_pos, "episode start") if not blocked else (None, None)
            if blocked:
                for k, p in enumerate(path_pts):
                    if not ctx.segment_blocked(p, source_pos, LOS_END_MARGIN_M):
                        los_pos, los_from = p, f"episode path raster point {k}"
                        break
            rec["los"] = {"start_in_line_of_sight": not blocked, "listener": None if los_pos is None else los_pos.tolist(),
                          "listener_from": los_from}
            if los_pos is not None:
                li = ctx.add_listener("mono", los_pos, q)
                rec["los"]["wall_time_s"] = ctx.simulate()
                ir = ctx.get_ir(li, src)
                if los_from != "episode start":
                    ctx.write_ir_wave(li, src, rdir / "ir_mono_line_of_sight.wav")
                    np.save(rdir / "ir_mono_line_of_sight.npy", ir)
                ctx.clear_listeners()
                dist = float(np.linalg.norm(source_pos - los_pos))
            else:
                ir, dist = np.zeros((1, 0), dtype=np.float32), float("nan")
            t = pt.test_first_arrival(ir, dist, sr, c, float(cfg["first_arrival_tolerance_ms"]), run_name, los_pos is not None)
            t["listener_from"] = los_from
            all_tests.append(t)
            log(t["line"] + f" (listener: {los_from})")

            # left ear before right ear
            _f, left, _r, _u = poses.agent_axes(heading)
            off = float(cfg["side_source_offset_m"])
            placed = None
            for k, p in enumerate([start_pos] + path_pts):
                s_pos = p + off * left
                if free_floor(p) and free_floor(s_pos) and not ctx.segment_blocked(p, s_pos, LOS_END_MARGIN_M):
                    placed = (p, s_pos, "episode start" if k == 0 else f"episode path raster point {k - 1}")
                    break
            if placed is not None:
                ctx.clear_sources()
                s2 = ctx.add_source(placed[1])
                li = ctx.add_listener("binaural", placed[0], q)
                rec["left_test_wall_time_s"] = ctx.simulate()
                ir = ctx.get_ir(li, s2)
                ctx.write_ir_wave(li, s2, rdir / "ir_binaural_left_source.wav")
                np.save(rdir / "ir_binaural_left_source.npy", ir)
                rec["left_test"] = {"listener": placed[0].tolist(), "source": placed[1].tolist(),
                                    "listener_from": placed[2], "heading_deg": heading}
            else:
                ir = np.zeros((2, 0), dtype=np.float32)
            t = pt.test_left_before_right(ir, sr, run_name, placed is not None)
            all_tests.append(t)
            log(t["line"] + (f" (listener: {placed[2]})" if placed else ""))
        (rdir / "run_record.json").write_text(json.dumps(rec, indent=2, default=str))
        run_records[run_name] = rec

    # ---- overlay and label lookup, four scenes
    scenes = overlay.choose_overlay_scenes(df, a.scene, int(cfg["n_overlay_scenes"]), seed)
    info["overlay_scenes"] = scenes
    odir = a.out / "overlay"
    for s in scenes:
        sd = a.scenes_root / s
        if not all((sd / f).exists() for f in ("structure.json", "labels.json", "occupancy.json", "occupancy.png")):
            line = f"FAIL overlay [{s}]: scene files missing under {sd}"
            all_tests.append({"test": "overlay", "run": s, "pass": False, "line": line})
            log(line)
            continue
        summ = overlay.run_overlay(sd, overlay.scene_rows(df, s), cfg, odir,
                                   affine_row=row if s == a.scene else None)
        overlay.print_overlay_summary(summ)
        log(f"{'PASS' if summ['wall_share_pass'] else 'FAIL'} wall_share [{s}]: {summ['wall_share_occupied']:.4f} "
            f"(bar {summ['wall_share_pass_bar']}; GN-Bench not-passable share {summ['wall_share_not_passable']:.4f}); "
            f"label_ins_id misses {summ['label_ins_id_misses']} of {summ['n_episodes']}")
        all_tests.append({"test": "wall_share", "run": s, "pass": summ["wall_share_pass"], "summary": summ})

    info["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (a.out / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    (a.out / "tests.json").write_text(json.dumps(all_tests, indent=2, default=str))
    n_fail = sum(1 for t in all_tests if not t["pass"])
    log(f"[done] {len(all_tests) - n_fail} PASS, {n_fail} FAIL (automatic tests only; the by-eye overlay check "
        f"is for the lead, images in {odir})")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
