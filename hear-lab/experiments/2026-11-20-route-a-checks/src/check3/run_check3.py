#!/usr/bin/env python
"""Check 3 runner: is the target visible from each GN-Matrix start pose?

Runs on the rented Linux GPU machine inside GN0's 'bae' environment (see
RUN.md). For every selected episode it renders GN-Bench's own depth image at
the start pose, projects points sampled on the goal object's labels.json box
into that image, applies the PLAN's visibility rule at three depth
tolerances, and appends one row to <out-dir>/check3_<split>.csv.

Design: experiments/2026-11-20-route-a-checks/PLAN.md, "Check 3". Every
parameter comes from config.yaml (check3 block). The simulator code used is
GN-Bench-Tools 0ce8367 (GN_Bench/sims/GN_Bench_simulator/GN_Bench_simulator.py)
inside GN0 c927781:

* each scene's splat file is loaded once with the simulator's own steps
  (load_occupancy_metadata, GaussianModel.load_ply), then every episode of that
  scene sets the episode's own world-to-splat affine with the simulator's
  derive_affine_transform (exactly what GNBenchSim.reconfigure does per
  episode) and renders with GNBenchSim.get_observations_at;
* on the first scene the same episode is also rendered after a full
  GNBenchSim.reconfigure, and the run stops if the two renders differ;
* before any scene, a synthetic plane is rendered with the same gsplat call to
  measure the depth convention (z-depth or ray distance) and the value of
  empty pixels.

Resume-safe: rows are appended and fsynced one by one; on restart, episodes
already in the CSV are skipped. Nothing here downloads anything.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c3lib  # noqa: E402

EXP_DIR = HERE.parent.parent
SPLIT_YAML = {
    "InteriorGS_test_unseen": "test_unseen.yaml",
    "InteriorGS_test_seen": "test_seen.yaml",
    "InteriorGS_train_seen": "train_seen.yaml",
}
# Depth sensor settings added to the pinned config (it has only RGB_SENSOR):
# metric depth, no normalisation, clipped at gsplat's far plane (100 m in
# render_camera_gsplat). Empty pixels come out as exactly DEPTH_MAX_M.
DEPTH_MAX_M = 100.0
FRAME_DIAG_BOXES = 20
FRAME_DIAG_GAUSSIANS = 200_000
FRAME_DIAG_SKIP_LABELS = {"floor", "wall", "ceiling", "carpet", "rug"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # not a git checkout on the machine
        return f"unknown ({exc.__class__.__name__})"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=EXP_DIR / "config.yaml")
    ap.add_argument("--gn0-root", type=Path, required=True,
                    help="GN0 checkout at the pinned commit (holds configs/gn_bench/interiorgs)")
    ap.add_argument("--gn-matrix-dir", type=Path, required=True,
                    help="directory holding <split>.parquet, e.g. InteriorGS_test_unseen.parquet")
    ap.add_argument("--scenes-dir", type=Path, required=True,
                    help="InteriorGS root with one folder per scene")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--splits", nargs="*", default=None,
                    help="subset of the config's splits (default: all of them)")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many newly processed episodes (timing run)")
    ap.add_argument("--skip-reconfigure-check", action="store_true",
                    help="do not compare the light scene load with GNBenchSim.reconfigure")
    return ap.parse_args()


# ----------------------------------------------------------------------------
# simulator setup
# ----------------------------------------------------------------------------
def build_sim_config(gn0_root: Path, split: str, cfg: dict):
    from gn0.benchmark.config import get_config

    yaml_path = gn0_root / "configs" / "gn_bench" / "interiorgs" / SPLIT_YAML[split]
    full = get_config(str(yaml_path))
    task_cfg = full.TASK_CONFIG
    sim_cfg = task_cfg.SIMULATOR.clone()
    sim_cfg.defrost()
    rgb = sim_cfg.RGB_SENSOR
    got = (int(rgb.WIDTH), int(rgb.HEIGHT), float(rgb.HFOV))
    want = (cfg["width"], cfg["height"], cfg["hfov_deg"])
    if got != want:
        raise SystemExit(f"camera in {yaml_path} is {got}, config.yaml says {want}; stopping")
    radius = float(task_cfg.TASK.SUCCESS_DISTANCE)
    if abs(radius - cfg["success_radius_m"]) > 1e-9:
        raise SystemExit(f"SUCCESS_DISTANCE {radius} != config.yaml {cfg['success_radius_m']}")
    d = sim_cfg.DEPTH_SENSOR
    d.WIDTH, d.HEIGHT, d.HFOV = rgb.WIDTH, rgb.HEIGHT, rgb.HFOV
    d.MIN_DEPTH = 0.0
    d.MAX_DEPTH = DEPTH_MAX_M
    d.NORMALIZE_DEPTH = False
    sim_cfg.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    return sim_cfg, yaml_path


def sensor_uuids(sim, gsim):
    rgb_uuid = depth_uuid = None
    for uuid, sensor in sim.sensor_suite.sensors.items():
        if isinstance(sensor, gsim.GNBenchSimDepthSensor):
            depth_uuid = uuid
        elif isinstance(sensor, gsim.GNBenchSimRGBSensor):
            rgb_uuid = uuid
    if rgb_uuid is None or depth_uuid is None:
        raise SystemExit("simulator did not build both an RGB and a depth sensor")
    return rgb_uuid, depth_uuid


def load_scene_light(sim, gsim, scene_dir: Path):
    """GNBenchSim.reconfigure steps 1 and 3 (metadata, splats) only: the
    occupancy grids, BEV map and actor are not used by get_observations_at."""
    from GN_Bench.scene.gaussian_model import GaussianModel

    sim.meta = gsim.load_occupancy_metadata(scene_dir)
    sim.gaussians = GaussianModel(sh_degree=3)
    sim.gaussians.load_ply(str(scene_dir / "3dgs_compressed.ply"))
    sim.scene_rest_dim = int(sim.gaussians.get_features_rest.shape[1])
    sim._current_scene = str(scene_dir)
    sim.actor_enabled = False


def set_episode(sim, gsim, payload):
    """Per-episode world-to-splat affine, as GNBenchSim.reconfigure fits it."""
    points, pixels = sim._load_raster_points(payload)
    sim.affine = gsim.derive_affine_transform(points, pixels, sim.meta)


def render_start(sim, gsim, uuids, start_xy, heading_deg, height_m):
    from scipy.spatial.transform import Rotation as R

    pos = np.array([start_xy[0], start_xy[1], height_m], dtype=np.float32)
    rot = R.from_euler("z", heading_deg, degrees=True).as_quat()
    t0 = time.perf_counter()
    obs = sim.get_observations_at(pos, rot, keep_agent_at_new_pose=True)
    if obs is None:
        raise RuntimeError("get_observations_at returned None")
    rgb = np.asarray(obs[uuids[0]])
    depth = np.asarray(obs[uuids[1]])[..., 0].astype(np.float32)
    render_s = time.perf_counter() - t0
    # Same pose arithmetic as GNBenchSim._render_observations.
    world_fwd = R.from_quat(rot).apply(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    eye = gsim.transform_from_world_to_GS(pos, sim.affine, sim.meta)
    target = gsim.transform_from_world_to_GS(pos + world_fwd, sim.affine, sim.meta)
    return rgb, depth, eye, target, render_s


def check_camera_matches_renderer(eye, target, intr, view):
    """Our view matrix and K against gsplat_renderer._build_pinhole_camera."""
    import torch
    from GN_Bench.render.gsplat_renderer import _build_pinhole_camera

    vm, K = _build_pinhole_camera(np.array(eye, dtype=np.float32), np.array(target, dtype=np.float32),
                                  intr["width"], intr["height"], math.radians(intr["fovy_deg"]),
                                  math.radians(intr["fovx_deg"]), torch.device("cpu"))
    vm = vm[0].numpy().astype(np.float64)
    K = K[0].numpy().astype(np.float64)
    if not np.allclose(vm, view, atol=1e-4) or not np.allclose(K, intr["K"], rtol=1e-5, atol=1e-4):
        raise SystemExit(f"camera mismatch with the renderer:\nview {view}\nrenderer {vm}\nK {intr['K']}\nrenderer K {K}")


class _FakeGaussians:
    """Minimal object with the attributes render_camera_gsplat reads."""

    def __init__(self, xyz, device):
        import torch

        n = len(xyz)
        self.get_xyz = torch.tensor(xyz, dtype=torch.float32, device=device)
        self.get_scaling = torch.full((n, 3), 0.015, dtype=torch.float32, device=device)
        rot = torch.zeros((n, 4), dtype=torch.float32, device=device)
        rot[:, 0] = 1.0
        self.get_rotation = rot
        self.get_opacity = torch.full((n, 1), 0.99, dtype=torch.float32, device=device)
        self.get_features = torch.full((n, 1, 3), 0.5, dtype=torch.float32, device=device)
        self.active_sh_degree = 0


def depth_preflight(intr: dict, device) -> dict:
    """Render a synthetic plane x = 3 m in front of a camera at the origin with
    the simulator's own gsplat call, and decide z-depth vs ray distance."""
    import torch
    from GN_Bench.render.gsplat_renderer import render_camera_gsplat

    plane_x = 3.0
    ys, zs = np.meshgrid(np.arange(-6.0, 6.0, 0.02), np.arange(-3.0, 5.0, 0.02), indexing="ij")
    xyz = np.stack([np.full(ys.size, plane_x), ys.ravel(), zs.ravel()], axis=1)
    fake = _FakeGaussians(xyz, device)
    bg = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    W, H = intr["width"], intr["height"]
    fovy, fovx = math.radians(intr["fovy_deg"]), math.radians(intr["fovx_deg"])
    eye = np.array([0.0, 0.0, 1.3], dtype=np.float32)
    with torch.no_grad():
        _, depth = render_camera_gsplat(fake, eye, eye + np.array([1.0, 0.0, 0.0], dtype=np.float32),
                                        W, H, fovy, fovx, bg, device, render_depth=True)
        _, empty = render_camera_gsplat(fake, eye, eye + np.array([-1.0, 0.0, 0.0], dtype=np.float32),
                                        W, H, fovy, fovx, bg, device, render_depth=True)
    probes = []
    for col, row in [(W // 2, H // 2), (8, H // 2), (W - 9, H // 2), (8, 8), (W - 9, H - 9)]:
        xn = (col + 0.5 - intr["cx"]) / intr["fx"]
        yn = (row + 0.5 - intr["cy"]) / intr["fy"]
        ray_factor = math.sqrt(1.0 + xn * xn + yn * yn)
        d = float(depth[row, col])
        probes.append({"col": col, "row": row, "rendered": d, "z_expected": plane_x,
                       "ray_expected": plane_x * ray_factor})
    err_z = max(abs(p["rendered"] - p["z_expected"]) for p in probes)
    err_ray = max(abs(p["rendered"] - p["ray_expected"]) for p in probes)
    kind = "z" if err_z < err_ray else "ray"
    result = {"probes": probes, "max_err_z": err_z, "max_err_ray": err_ray, "depth_kind": kind,
              "empty_view_raw_min": float(np.min(empty)), "empty_view_raw_max": float(np.max(empty))}
    if min(err_z, err_ray) > 0.05:
        raise SystemExit(f"depth preflight: neither convention fits within 5 cm: {json.dumps(result)}")
    if float(np.max(empty)) > 1e-6:
        raise SystemExit(f"depth preflight: empty view is not zero before the simulator's mapping: {json.dumps(result)}")
    return result


def frame_diagnostic(sim, labels: list, rng) -> dict:
    """How many splat centres fall inside label boxes, for labels.json taken
    as is and for the three mirrored alternatives about the map centre. If the
    boxes are in the splat frame (as assumed), 'identity' should win."""
    xyz = sim.gaussians.get_xyz.detach().float().cpu().numpy()
    if len(xyz) > FRAME_DIAG_GAUSSIANS:
        xyz = xyz[rng.choice(len(xyz), FRAME_DIAG_GAUSSIANS, replace=False)]
    cx = sim.meta["left"] + sim.meta["right"]
    cy = sim.meta["top"] + sim.meta["bottom"]
    boxes = []
    for e in sorted(labels, key=lambda e: str(e.get("ins_id"))):
        c = c3lib.box_corners(e)
        if c is None or str(e.get("label", "")).lower() in FRAME_DIAG_SKIP_LABELS:
            continue
        ext = c.max(0) - c.min(0)
        if np.prod(ext) < 0.01:
            continue
        boxes.append(c)
        if len(boxes) >= FRAME_DIAG_BOXES:
            break
    variants = {"identity": (1, 1), "mirror_x": (-1, 1), "mirror_y": (1, -1), "mirror_xy": (-1, -1)}
    out = {"n_boxes": len(boxes), "n_gaussians_sampled": int(len(xyz))}
    for name, (sx, sy) in variants.items():
        counts = []
        for c in boxes:
            cc = c.copy()
            if sx < 0:
                cc[:, 0] = cx - cc[:, 0]
            if sy < 0:
                cc[:, 1] = cy - cc[:, 1]
            counts.append(int(c3lib.points_in_box(xyz, cc).sum()))
        out[name] = {"median": float(np.median(counts)) if counts else None,
                     "total": int(np.sum(counts)) if counts else 0}
    return out


# ----------------------------------------------------------------------------
# main loop
# ----------------------------------------------------------------------------
def fmt(x):
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, float):
        return f"{x:.6g}"
    return x


def main():
    args = parse_args()
    cfg = c3lib.load_config(args.config)
    tols = c3lib.tolerances(cfg)
    intr = c3lib.camera_intrinsics(cfg["width"], cfg["height"], cfg["hfov_deg"])
    splits = args.splits or list(cfg["splits"].keys())
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    import pyarrow.parquet as pq
    import torch
    import GN_Bench.sims.GN_Bench_simulator.GN_Bench_simulator as gsim
    from gn0.benchmark.dataset import GNMatrixDataset

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device.type != "cuda":
        raise SystemExit("no CUDA device; the GN-Bench gsplat renderer needs a GPU")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta = {
        "started_utc": stamp, "argv": sys.argv, "repo_commit": git_commit(),
        "config": {k: v for k, v in cfg.items()}, "intrinsics": {k: v for k, v in intr.items() if k != "K"},
        "depth_sensor": {"MIN_DEPTH": 0.0, "MAX_DEPTH": DEPTH_MAX_M, "NORMALIZE_DEPTH": False},
        "host": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                 "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
        "splits": {},
    }
    try:
        import gsplat
        meta["host"]["gsplat"] = getattr(gsplat, "__version__", "unknown")
    except Exception as exc:
        meta["host"]["gsplat"] = f"import failed: {exc}"
    pre = depth_preflight(intr, device)
    meta["depth_preflight"] = pre
    depth_kind = pre["depth_kind"]
    print(f"[check3] depth convention: {depth_kind} (max err z {pre['max_err_z']:.4f}, ray {pre['max_err_ray']:.4f})", flush=True)

    meta_dir = out / "run_meta"
    meta_dir.mkdir(exist_ok=True)
    meta_path = meta_dir / f"{stamp}.json"

    def save_meta():
        meta_path.write_text(json.dumps(meta, indent=2, default=str))

    processed = 0
    missing_scenes = set()
    reconfigure_checked = args.skip_reconfigure_check
    rng_diag = np.random.default_rng(cfg["seed"])
    scenes_log = open(out / "scenes.jsonl", "a", encoding="utf-8")

    for split in splits:
        spec = cfg["splits"][split]
        pq_path = args.gn_matrix_dir / f"{split}.parquet"
        table = pq.read_table(pq_path)
        rows_sel = c3lib.select_rows(table.num_rows, spec, cfg["seed"])
        review = set(c3lib.review_candidates(rows_sel, cfg["n_review"], cfg["seed"]))
        meta["splits"][split] = {"parquet": str(pq_path), "sha256": sha256(pq_path),
                                 "n_rows": table.num_rows, "spec": spec, "n_selected": int(len(rows_sel)),
                                 "first_rows": [int(r) for r in rows_sel[:10]]}
        save_meta()
        sim_cfg, yaml_path = build_sim_config(args.gn0_root, split, cfg)
        meta["splits"][split]["sim_yaml"] = str(yaml_path)
        sim = gsim.GNBenchSim(sim_cfg)
        uuids = sensor_uuids(sim, gsim)

        csv_path = out / f"check3_{split}.csv"
        c3lib.repair_csv_tail(csv_path)
        done = c3lib.done_keys(csv_path)
        new_file = not csv_path.exists() or csv_path.stat().st_size == 0
        fh = open(csv_path, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=c3lib.CSV_FIELDS)
        if new_file:
            writer.writeheader()
            fh.flush()
            os.fsync(fh.fileno())

        pending = [int(r) for r in rows_sel if (split, int(r)) not in done]
        print(f"[check3] {split}: {len(rows_sel)} selected, {len(done)} done, {len(pending)} pending", flush=True)
        if not pending:
            fh.close()
            continue
        recs = table.take(pending).to_pylist()
        by_scene = defaultdict(list)
        for r, rec in zip(pending, recs):
            by_scene[str(rec["scene"])].append((r, rec))

        for scene in sorted(by_scene):
            if args.limit is not None and processed >= args.limit:
                break
            scene_dir = args.scenes_dir / scene
            needed = ["3dgs_compressed.ply", "labels.json", "occupancy.json", "occupancy.png"]
            if not all((scene_dir / f).exists() for f in needed):
                missing_scenes.add(scene)
                print(f"[check3] scene {scene}: files missing, skipped", flush=True)
                continue
            t0 = time.perf_counter()
            load_scene_light(sim, gsim, scene_dir)
            load_s = time.perf_counter() - t0
            labels = json.loads((scene_dir / "labels.json").read_text(encoding="utf-8"))
            by_id = defaultdict(list)
            for e in labels:
                by_id[str(e.get("ins_id"))].append(e)
            diag = frame_diagnostic(sim, labels, rng_diag)
            n_g = int(sim.gaussians.get_xyz.shape[0])

            n_done_scene = 0
            for row_index, rec in by_scene[scene]:
                if args.limit is not None and processed >= args.limit:
                    break
                payload = GNMatrixDataset._build_trajectory_data(rec)
                set_episode(sim, gsim, payload)
                sx, sy = float(rec["start_world_x"]), float(rec["start_world_y"])
                hd = float(rec["start_facing_heading_degrees"])
                gx, gy = float(rec["goal_world_x"]), float(rec["goal_world_y"])
                straight = c3lib.straight_distance(sx, sy, gx, gy)
                plen = c3lib.polyline_length(rec["path_raster_world_x"], rec["path_raster_world_y"])
                ins = str(rec["label_ins_id"])
                entries = by_id.get(ins, [])
                row = {k: "" for k in c3lib.CSV_FIELDS}
                row.update({
                    "split": split, "row_index": row_index, "trajectory_id": rec["trajectory_id"],
                    "scene": scene, "label_ins_id": ins, "label_label": rec["label_label"],
                    "n_label_entries": len(entries),
                    "start_world_x": sx, "start_world_y": sy, "start_heading_deg": hd,
                    "goal_world_x": gx, "goal_world_y": gy,
                    "straight_dist_m": straight, "path_length_m": plen,
                    "within_radius_straight": straight <= cfg["success_radius_m"],
                    "within_radius_path": plen <= cfg["success_radius_m"],
                    "depth_tol_primary_m": tols["primary"], "depth_tol_strict_m": tols["strict"],
                    "depth_tol_loose_m": tols["loose"], "depth_kind": depth_kind,
                })
                corners = c3lib.box_corners(entries[0]) if entries else None
                if not entries:
                    row["status"] = "missing_id"
                elif corners is None:
                    row["status"] = "missing_box"
                else:
                    rgb, depth, eye, target, render_s = render_start(
                        sim, gsim, uuids, (sx, sy), hd, cfg["camera_height_m"])
                    view = c3lib.look_at_view(eye, target)
                    check_camera_matches_renderer(eye, target, intr, view)
                    if not reconfigure_checked:
                        meta["reconfigure_check"] = reconfigure_check(
                            gsim, sim_cfg, scene_dir, rec, payload, (sx, sy), hd, cfg, rgb, depth)
                        save_meta()
                        reconfigure_checked = True
                    invalid = depth >= DEPTH_MAX_M * (1.0 - 1e-6)
                    pts = c3lib.sample_box_surface(corners, cfg["spacing_m"])
                    vis = c3lib.evaluate_visibility(pts, view, intr, depth, invalid, tols, depth_kind)
                    row["status"] = "ok"
                    row["render_s"] = render_s
                    row["cam_to_box_centre_m"] = float(np.linalg.norm(corners.mean(0) - np.asarray(eye, dtype=np.float64)))
                    # Diagnostic: horizontal distance from the GN-Matrix goal position
                    # (mapped to the splat frame like the camera) to the target box.
                    goal_gs = np.asarray(gsim.transform_from_world_to_GS(
                        np.array([gx, gy, 0.0], dtype=np.float32), sim.affine, sim.meta), dtype=np.float64)
                    goal_gs[2] = float(corners[:, 2].mean())
                    row["goal_to_box_horiz_m"] = c3lib.point_box_distance(goal_gs, corners)
                    for k in ("n_box_points", "n_front", "n_in_image", "n_invalid_depth"):
                        row[k] = vis[k]
                    for name in tols:
                        row[f"n_seen_{name}"] = vis[f"n_seen_{name}"]
                        row[f"share_seen_{name}"] = vis[f"share_seen_{name}"]
                        row[f"visible_{name}"] = vis[f"visible_{name}"]
                    if row_index in review:
                        rdir = out / "review_raw" / split
                        rdir.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(
                            rdir / f"{row_index}.npz", rgb=rgb, depth=depth, u=vis["u"], v=vis["v"],
                            in_image=vis["in_image"], seen_primary=vis["seen_primary"],
                            eye_gs=np.asarray(eye), target_gs=np.asarray(target))
                writer.writerow({k: fmt(v) for k, v in row.items()})
                fh.flush()
                os.fsync(fh.fileno())
                processed += 1
                n_done_scene += 1
            scenes_log.write(json.dumps({"split": split, "scene": scene, "load_s": load_s,
                                         "n_gaussians": n_g, "episodes": n_done_scene,
                                         "frame_diag": diag}) + "\n")
            scenes_log.flush()
            print(f"[check3] {split} {scene}: {n_done_scene} episodes, load {load_s:.1f}s, "
                  f"processed {processed}", flush=True)
        fh.close()
        del sim
        torch.cuda.empty_cache()
        if args.limit is not None and processed >= args.limit:
            break

    scenes_log.close()
    meta["finished_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta["processed_this_run"] = processed
    meta["missing_scenes"] = sorted(missing_scenes)
    save_meta()
    if missing_scenes:
        print(f"[check3] {len(missing_scenes)} scenes had missing files: {sorted(missing_scenes)[:20]}", flush=True)
        sys.exit(2)
    print(f"[check3] done: {processed} episodes this run; meta {meta_path}", flush=True)


def reconfigure_check(gsim, sim_cfg, scene_dir, rec, payload, start_xy, heading, cfg, rgb_light, depth_light):
    """Render the first episode again after a full GNBenchSim.reconfigure
    (the path GN-Bench itself takes) and compare with the light scene load."""
    import torch

    c = sim_cfg.clone()
    c.defrost()
    c.SCENE = str(scene_dir)
    c.REF_JSON = f"{rec['scene']}/{rec['trajectory_id']}.json"
    c.TRAJECTORY_JSON = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    c.freeze()
    full = gsim.GNBenchSim(c)
    t0 = time.perf_counter()
    full.reconfigure(c)
    reconf_s = time.perf_counter() - t0
    rgb, depth, _, _, _ = render_start(full, gsim, sensor_uuids(full, gsim), start_xy, heading,
                                       cfg["camera_height_m"])
    d_rgb = int(np.abs(rgb.astype(np.int32) - rgb_light.astype(np.int32)).max())
    d_depth = float(np.abs(depth - depth_light).max())
    res = {"scene": str(scene_dir.name), "trajectory_id": rec["trajectory_id"], "reconfigure_s": reconf_s,
           "max_abs_diff_rgb": d_rgb, "max_abs_diff_depth_m": d_depth}
    del full
    torch.cuda.empty_cache()
    print(f"[check3] reconfigure check: {res}", flush=True)
    if d_rgb > 1 or d_depth > 1e-3:
        raise SystemExit(f"light scene load does not reproduce GNBenchSim.reconfigure: {res}")
    return res


if __name__ == "__main__":
    main()
