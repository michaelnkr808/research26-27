#!/usr/bin/env python
"""Export the per-frame listener poses of the released reference-trajectory renderer, and its videos.

Runs on the rented machine inside GN0's 'bae' environment, from the GN0 root (needs GN-Bench, gsplat, a CUDA GPU
and the InteriorGS scenes). It does what GN0's scripts/evaluation/render_dataset.sh does for train_seen
(gn0/evaluation/local.py render_trajectory_videos at c927781): the same config, dataset loading, Env and
TrajectoryVideoCollector with the config's EVAL.RENDERER settings; only the episode list differs (the Step 2
ids instead of an index range). The collector is subclassed only to record the positions and rotations it
renders, in its own order, so audio step k uses the pose of video frame k. Nothing in GN0 is changed.

For every frame it also writes the camera eye and forward in the InteriorGS (GS) frame computed with GN-Bench's
own transform_from_world_to_GS, exactly as GNBenchSim._render_observations places the camera.

Outputs under --out (all local only: they hold scene coordinates and renders):
  poses/<episode_id>.json   per-frame world position and quaternion, GS eye and forward, the episode's target
                            ins_id, goal point, occupancy metadata, timing
  videos/<scene>/<trajectory_id>.mp4   the released renderer's video of the same frames
  export_meta.json          config, renderer settings, GN0 and GN-Bench-Tools commits, the ids, timing
  alignment.json            (with --alignment-n) frame counts and pixel differences against reference videos

  python export_poses.py --step2-config <exp>/config.yaml --order <exp>/episode_order.txt --n 12 \
      --out /workspace/step2/out/batch1-v1/export [--alignment-n 2 --reference-videos DIR]
  python export_poses.py --step2-config ... --list-alignment 2     # print the alignment ids only (no GPU)
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def frame_records(positions, rotations, to_gs) -> list:
    """Per-frame records from the collector's positions (world, x y z) and rotations (scipy quaternions x y z w).
    to_gs maps a world point (3,) to GS (3,) (GN-Bench transform_from_world_to_GS with the episode's affine).
    Forward in world is the rotation applied to local +x (GNBenchSim._render_observations)."""
    from scipy.spatial.transform import Rotation as R

    out = []
    for k, (p, q) in enumerate(zip(positions, rotations)):
        p = np.asarray(p, dtype=np.float64).reshape(3)
        q = np.asarray(q, dtype=np.float64).reshape(4)
        fwd_w = R.from_quat(q).apply(np.array([1.0, 0.0, 0.0]))
        eye = np.asarray(to_gs(p.astype(np.float32)), dtype=np.float64)
        tgt = np.asarray(to_gs((p + fwd_w).astype(np.float32)), dtype=np.float64)
        out.append({
            "frame": k,
            "world_position": p.tolist(),
            "world_quat_xyzw": q.tolist(),
            "world_yaw_deg": math.degrees(math.atan2(fwd_w[1], fwd_w[0])),
            "gs_eye": eye.tolist(),
            "gs_forward": (tgt - eye).tolist(),
        })
    return out


def git_head(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unknown ({exc.__class__.__name__})"


def read_ids(order_file: Path, n: int) -> list:
    ids = [x for x in Path(order_file).read_text(encoding="utf-8").split() if x]
    if n > len(ids):
        raise SystemExit(f"asked for {n} ids, order has {len(ids)}")
    return ids[:n]


def video_frames(path: Path) -> list:
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f.astype(np.int16))
    cap.release()
    return frames


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step2-config", type=Path, required=True)
    ap.add_argument("--order", type=Path, required=True)
    ap.add_argument("--n", type=int, default=0, help="first N ids of the order")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--alignment-n", type=int, default=0,
                    help="also export the first N episodes of the released renderer's own order (render_dataset.sh "
                         "--start-idx 0 --end-idx N) and compare them with --reference-videos")
    ap.add_argument("--reference-videos", type=Path, help="folder with <scene>/<trajectory_id>.mp4 from route A check 2")
    ap.add_argument("--list-alignment", type=int, default=None, metavar="N",
                    help="print the ids of the first N episodes of the released render order and exit")
    a = ap.parse_args(argv)

    import yaml
    from gn0.benchmark import get_config
    from gn0.evaluation.common import get_episode_id
    from gn0.evaluation.local import load_dataset_split, resolve_renderer_settings, select_render_episodes

    s2cfg = yaml.safe_load(a.step2_config.read_text(encoding="utf-8"))
    pcfg = s2cfg["poses"]
    config = get_config(pcfg["exp_config"])
    split = load_dataset_split(config)
    by_id = {get_episode_id(e.ref_json): e for e in split.episodes}
    align_eps, _ = select_render_episodes(split, 0, max(a.alignment_n, a.list_alignment or 0) or 0)
    align_ids = [get_episode_id(e.ref_json) for e in align_eps]
    if a.list_alignment is not None:
        print("\n".join(align_ids[: a.list_alignment]))
        return 0
    if a.out is None:
        ap.error("--out is required")
    if a.out.exists():
        ap.error(f"--out {a.out} exists; results are never overwritten")

    settings = resolve_renderer_settings(config)
    bad = {k: (settings[k], v) for k, v in pcfg["renderer_expected"].items() if settings[k] != v}
    if bad:
        raise SystemExit(f"renderer settings differ from config poses.renderer_expected: {bad}")

    ids = read_ids(a.order, a.n) if a.n else []
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} ids not in the train_seen dataset, e.g. {missing[:3]}")
    want = list(dict.fromkeys(ids + align_ids[: a.alignment_n]))
    if a.alignment_n and a.reference_videos is None:
        ap.error("--alignment-n needs --reference-videos")

    from GN_Bench import Env
    from GN_Bench.sims.GN_Bench_simulator import GN_Bench_simulator as gsim
    from gn0.collection import TrajectoryVideoCollector

    class RecordingCollector(TrajectoryVideoCollector):
        """The released collector; _render_poses is wrapped only to keep the poses it renders."""

        def _render_poses(self, *, sim, positions, rotations):
            self.recorded = (list(positions), list(rotations))
            return super()._render_poses(sim=sim, positions=positions, rotations=rotations)

    a.out.mkdir(parents=True)
    (a.out / "poses").mkdir()
    split.episodes = [by_id[i] for i in want]
    t_env = time.perf_counter()
    env = Env(config.TASK_CONFIG, split)
    t_env = time.perf_counter() - t_env
    rec = RecordingCollector(a.out / "videos", fps=settings["video_fps"], codec=settings["video_codec"],
                             smooth_window=settings["smooth_window"], resample_step_m=settings["resample_step_m"],
                             heading_lookahead_m=settings["heading_lookahead_m"],
                             yaw_smooth_window=settings["yaw_smooth_window"])
    meta = {"step2_config": str(a.step2_config), "gn0_head": git_head(Path.cwd()),
            "gn_bench_tools_head": git_head(Path.cwd() / "GN-Bench-Tools"), "exp_config": pcfg["exp_config"],
            "renderer_settings": settings, "ids": ids, "alignment_ids": align_ids[: a.alignment_n],
            "env_build_s": t_env, "episodes": {}}
    done = set()
    for _ in range(len(env.episodes)):
        t0 = time.perf_counter()
        env.reset()
        t_reset = time.perf_counter() - t0
        ep = env.current_episode
        eid = get_episode_id(ep.ref_json)
        sim = env.sim
        start_z = float(np.asarray(sim.get_agent_state().position, dtype=np.float32)[2])
        t1 = time.perf_counter()
        try:
            video_path, n_frames = rec.collect(episode=ep, sim=sim)
            err = None
        except Exception as exc:  # noqa: BLE001 - record and go on; the episode is then not rendered
            video_path, n_frames, err = None, None, repr(exc)
        t_render = time.perf_counter() - t1
        positions, rotations = getattr(rec, "recorded", ([], []))
        rec.recorded = ([], [])
        to_gs = lambda p: gsim.transform_from_world_to_GS(p, sim.affine, sim.meta)  # noqa: E731
        frames = frame_records(positions, rotations, to_gs)
        goal = ep.goal_info["world"]
        goal_gs = np.asarray(to_gs(np.array([goal["x"], goal["y"], 0.0], dtype=np.float32)), dtype=np.float64)
        out = {
            "episode_id": eid, "scene": Path(ep.ref_json).parent.name, "trajectory_id": Path(ep.ref_json).stem,
            "target_ins_id": str(ep.label_info["ins_id"]), "goal_gs_xy": goal_gs[:2].tolist(),
            "listener_height_m": start_z, "video_fps": settings["video_fps"], "frame_period_s": 1.0 / settings["video_fps"],
            "n_frames": n_frames, "n_poses": len(frames), "video": None if video_path is None else str(video_path),
            "error": err, "affine": [float(x) for x in sim.affine],
            "occupancy_meta": {k: sim.meta[k] for k in ("left", "top", "scale", "width", "height")},
            "reset_s": t_reset, "render_s": t_render, "frames": frames,
        }
        (a.out / "poses" / f"{eid}.json").write_text(json.dumps(out, default=float), encoding="utf-8")
        meta["episodes"][eid] = {"n_frames": n_frames, "n_poses": len(frames), "error": err,
                                 "reset_s": t_reset, "render_s": t_render}
        print(f"[export] {eid}: {len(frames)} poses, {n_frames} frames, render {t_render:.1f} s"
              + (f", ERROR {err}" if err else ""), flush=True)
        done.add(eid)
    meta["missing"] = sorted(set(want) - done)
    (a.out / "export_meta.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    status = 0 if not meta["missing"] and all(v["error"] is None and v["n_frames"] == v["n_poses"]
                                              for v in meta["episodes"].values()) else 1

    if a.alignment_n:
        al = {}
        for eid in align_ids[: a.alignment_n]:
            p = json.loads((a.out / "poses" / f"{eid}.json").read_text(encoding="utf-8"))
            ref = a.reference_videos / p["scene"] / f"{p['trajectory_id']}.mp4"
            if not ref.is_file():
                al[eid] = {"pass": False, "why": f"no reference video {ref}"}
                continue
            fr, fn = video_frames(ref), video_frames(Path(p["video"]))
            mad = [float(np.abs(x - y).mean()) for x, y in zip(fr, fn)]
            al[eid] = {"reference_frames": len(fr), "new_frames": len(fn), "poses": p["n_poses"],
                       "pass": len(fr) == len(fn) == p["n_poses"],
                       "mean_abs_pixel_diff_max": max(mad) if mad else None,
                       "mean_abs_pixel_diff_median": float(np.median(mad)) if mad else None}
        (a.out / "alignment.json").write_text(json.dumps(al, indent=2), encoding="utf-8")
        print("[alignment] " + json.dumps(al), flush=True)
        if not all(v["pass"] for v in al.values()):
            status = 1
    print(f"[done] {len(done)} episodes, status {status}", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
