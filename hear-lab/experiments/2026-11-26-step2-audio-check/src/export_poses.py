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
  alignment.json            (with --alignment-n) frame counts and pixel differences against reference videos, gated
                            by config poses.alignment (PROPOSED thresholds, PLAN.md)

Per episode it also fails (error recorded, exit 1) when the renderer's camera calls do not match the frames one to
one, when the pose computed from the collector's rotations differs from what the renderer received, when a moving
frame's camera forward is more than poses.heading_gate.max_deg from the direction of travel (heading_gate()), or
when an ear height differs from poses.listener_height_m_expected.

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


def heading_gate(eyes, forwards, lookahead_m: float, max_deg: float, skip_first: int) -> dict:
    """Check every frame's camera forward against the direction of travel along the path (review of PR #95, 1).

    Direction of travel at frame k: from eye k to the point lookahead_m further along the polyline of eyes
    (clamped at the end), the rule the released collector uses for its yaw (_path_yaws); at the last frame, where
    no path is left ahead, the step from frame k - 1. Both horizontal. A frame is gated when it moves: the path
    ahead (or the last step) is at least 1 mm long. Not gated: the first skip_first frames, whose yaw the collector
    blends with the episode's start heading on purpose (smoothed[0] is the start yaw, and the yaw window spreads
    it over the next frames), and frames that do not move. The collector resamples the path every 0.05 m and drops
    repeated points, so it has no turning-in-place frames; any frame without motion is counted in
    n_not_moving (expected 0) and left out of the gate. Returns per-frame angles (None when not gated) and the
    counts; ok is False when any gated frame is more than max_deg off."""
    e = np.asarray(eyes, dtype=np.float64).reshape(-1, 3)[:, :2]
    f = np.asarray(forwards, dtype=np.float64).reshape(-1, 3)[:, :2]
    n = len(e)
    seg = np.linalg.norm(np.diff(e, axis=0), axis=1) if n > 1 else np.zeros(0)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    angles, n_gated, n_bad, n_not_moving, worst = [], 0, 0, 0, 0.0
    for k in range(n):
        if k < skip_first:
            angles.append(None)
            continue
        s = min(cum[-1], cum[k] + lookahead_m)
        if s > cum[k] + 1e-3:
            j = int(np.searchsorted(cum, s, side="right") - 1)
            j = max(0, min(j, n - 2))
            a = (s - cum[j]) / max(cum[j + 1] - cum[j], 1e-12)
            d = e[j] * (1 - a) + e[j + 1] * a - e[k]
        elif k > 0 and seg[k - 1] >= 1e-3:
            d = e[k] - e[k - 1]
        else:
            n_not_moving += 1
            angles.append(None)
            continue
        fk = f[k]
        if np.linalg.norm(fk) < 1e-9:
            n_gated += 1
            n_bad += 1
            angles.append(180.0)
            continue
        cosang = float(fk @ d / (np.linalg.norm(fk) * np.linalg.norm(d)))
        ang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
        angles.append(ang)
        n_gated += 1
        n_bad += int(ang > max_deg)
        worst = max(worst, ang)
    return {"angles_deg": angles, "n_frames": n, "n_gated": n_gated, "n_over_limit": n_bad,
            "n_not_moving": n_not_moving, "n_start_blend_skipped": min(skip_first, n), "max_angle_deg": worst,
            "max_deg": max_deg, "ok": n_bad == 0}


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

    # Record the eye and look-at target the renderer actually receives: GNBenchSim._render_observations calls the
    # module-level render_camera_gsplat(gaussians, gs_pos, gs_target, ...) (GN_Bench_simulator.py lines 979 to 990 at
    # 0ce8367; train_seen has one RGB sensor, so one call per frame). The wrapper passes every argument through
    # unchanged and only keeps copies of gs_pos and gs_target while the collector renders.
    renderer_calls = []
    _orig_render = gsim.render_camera_gsplat

    def _recording_render(gaussians, gs_pos, gs_target, *args, **kwargs):
        if renderer_calls is not None and getattr(_recording_render, "active", False):
            renderer_calls.append((np.asarray(gs_pos, dtype=np.float64).copy(),
                                   np.asarray(gs_target, dtype=np.float64).copy()))
        return _orig_render(gaussians, gs_pos, gs_target, *args, **kwargs)

    gsim.render_camera_gsplat = _recording_render

    class RecordingCollector(TrajectoryVideoCollector):
        """The released collector; _render_poses is wrapped only to keep the poses it renders and to switch the
        renderer-input recording on for exactly those renders."""

        def _render_poses(self, *, sim, positions, rotations):
            self.recorded = (list(positions), list(rotations))
            renderer_calls.clear()
            _recording_render.active = True
            try:
                return super()._render_poses(sim=sim, positions=positions, rotations=rotations)
            finally:
                _recording_render.active = False

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
        # The listener pose is taken from what the renderer received (eye, and target - eye as forward); the pose
        # computed from the collector's rotations is kept beside it as a cross-check of the quaternion format.
        calls = list(renderer_calls)
        renderer_check = {"n_calls": len(calls), "n_frames": len(frames)}
        if len(calls) == len(frames) and frames:
            d_eye = max(float(np.abs(c[0] - np.asarray(fr["gs_eye"])).max()) for c, fr in zip(calls, frames))
            d_fwd = max(float(np.abs((c[1] - c[0]) - np.asarray(fr["gs_forward"])).max()) for c, fr in zip(calls, frames))
            renderer_check.update({"max_abs_diff_eye": d_eye, "max_abs_diff_forward": d_fwd})
            for c, fr in zip(calls, frames):
                fr["gs_eye_from_rotation"], fr["gs_forward_from_rotation"] = fr["gs_eye"], fr["gs_forward"]
                fr["gs_eye"], fr["gs_forward"] = c[0].tolist(), (c[1] - c[0]).tolist()
                fr["renderer_gs_target"] = c[1].tolist()
            renderer_check["ok"] = d_eye < 1e-3 and d_fwd < 1e-3
        else:
            renderer_check["ok"] = False
            if err is None:
                err = f"renderer saw {len(calls)} camera calls for {len(frames)} frames"
        gate = heading_gate([fr["gs_eye"] for fr in frames], [fr["gs_forward"] for fr in frames],
                            float(pcfg["heading_gate"]["lookahead_m"]), float(pcfg["heading_gate"]["max_deg"]),
                            int(settings["yaw_smooth_window"]) // 2 + 1)
        if err is None and not gate["ok"]:
            err = (f"heading gate: {gate['n_over_limit']} of {gate['n_gated']} moving frames have the camera forward "
                   f"more than {gate['max_deg']} degrees from the direction of travel (max {gate['max_angle_deg']:.1f})")
        if err is None and not renderer_check["ok"]:
            err = f"renderer input differs from the pose computed from the collector's rotations: {renderer_check}"
        if err is None:
            bad_z = [fr["frame"] for fr in frames
                     if abs(float(fr["gs_eye"][2]) - float(pcfg["listener_height_m_expected"])) > 1e-4]
            if bad_z:
                err = f"ear height differs from {pcfg['listener_height_m_expected']} m on frames {bad_z[:5]}"
        goal = ep.goal_info["world"]
        goal_gs = np.asarray(to_gs(np.array([goal["x"], goal["y"], 0.0], dtype=np.float32)), dtype=np.float64)
        out = {
            "episode_id": eid, "scene": Path(ep.ref_json).parent.name, "trajectory_id": Path(ep.ref_json).stem,
            "target_ins_id": str(ep.label_info["ins_id"]), "goal_gs_xy": goal_gs[:2].tolist(),
            "listener_height_m": start_z, "video_fps": settings["video_fps"], "frame_period_s": 1.0 / settings["video_fps"],
            "n_frames": n_frames, "n_poses": len(frames), "video": None if video_path is None else str(video_path),
            "error": err, "affine": [float(x) for x in sim.affine],
            "occupancy_meta": {k: sim.meta[k] for k in ("left", "top", "scale", "width", "height")},
            "reset_s": t_reset, "render_s": t_render, "renderer_check": renderer_check,
            "heading_gate": {k: v for k, v in gate.items() if k != "angles_deg"}, "frames": frames,
        }
        for fr, ang in zip(frames, gate["angles_deg"]):
            fr["heading_vs_travel_deg"] = ang
        (a.out / "poses" / f"{eid}.json").write_text(json.dumps(out, default=float), encoding="utf-8")
        meta["episodes"][eid] = {"n_frames": n_frames, "n_poses": len(frames), "error": err,
                                 "reset_s": t_reset, "render_s": t_render, "renderer_check": renderer_check,
                                 "heading_gate": out["heading_gate"]}
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
            thr = pcfg["alignment"]
            med = float(np.median(mad)) if mad else None
            mx = max(mad) if mad else None
            al[eid] = {"reference_frames": len(fr), "new_frames": len(fn), "poses": p["n_poses"],
                       "frame_counts_equal": len(fr) == len(fn) == p["n_poses"],
                       "mean_abs_pixel_diff_max": mx, "mean_abs_pixel_diff_median": med,
                       "thresholds": thr,
                       "pass": (len(fr) == len(fn) == p["n_poses"] and med is not None
                                and med <= float(thr["max_median_frame_mad"]) and mx <= float(thr["max_frame_mad"]))}
        (a.out / "alignment.json").write_text(json.dumps(al, indent=2), encoding="utf-8")
        print("[alignment] " + json.dumps(al), flush=True)
        if not all(v["pass"] for v in al.values()):
            status = 1
    print(f"[done] {len(done)} episodes, status {status}", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
