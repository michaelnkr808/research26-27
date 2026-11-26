#!/usr/bin/env python3
"""Render Step 2's binaural room impulse responses and listening clips with RLR, and run the direction check.

Runs on the rented Linux x64 machine (the RLR library ships Linux binaries only), in the audio environment
(requirements-step2.txt), after export_poses.py. Design and pass bar: ../PLAN.md. Every parameter: ../config.yaml.

Per episode (one process per episode, threadCount 1 inside, as SoundSpaces 2.0):
  1. scene load: the InteriorGS floor plan and boxes as a triangle mesh in the InteriorGS (GS) frame
     (hear-lab/src/geometry.py, no frame map: the identity), the target's own box left out; the source at the
     centre of the target's labels.json box; one RLR context with the SoundSpaces 2.0 settings, materials off
     (default material on every face), one binaural listener with the MIT KEMAR HRTF.
  2. per video frame k, in the export's order: move the listener to the camera eye and forward of frame k
     (temporal coherence on), simulate, read the two-ear impulse response, test the line of sight, estimate the
     interaural time difference and compare its sign with the true side.
  3. the listening clips: the looped, loudness-normalised telephone ring through the per-step responses, one step
     per video frame, two-ear and one-channel (mean of the two ears).
Before any episode, a free-field preflight with the real library and HRTF checks the side sign at four headings.

Outputs under --out (results/<run>/ when copied back). Committed (allow list in results/.gitignore):
run_config.json, episode_records.jsonl, timing.json, preflight.json. Local only (InteriorGS-derived):
episodes/<id>/{episode.json, steps.jsonl, brirs.npy, brir_lengths.npy}, clips/*.wav, render.log.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
SHARED = Path(os.environ.get("HEAR_LAB_SRC") or HERE.parents[2] / "src")
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(SHARED))

import step2lib as s2  # noqa: E402

COMMITTED_FILES = ("run_config.json", "episode_records.jsonl", "timing.json", "preflight.json")


def pending_values(cfg: dict) -> list:
    out = []

    def walk(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, f"{prefix}{k}.")
            elif v == "pending":
                out.append(prefix + k)
    walk(cfg)
    return out


def load_source(path: Path, sr: int, target_dbfs: float) -> tuple:
    """The telephone ring as float64 mono at sr, RMS-normalised to target_dbfs over the whole clip."""
    from math import gcd
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    native_sr, data = wavfile.read(str(path))
    x = s2.to_float_mono(data)
    info = {"native_rate_hz": int(native_sr), "native_samples": int(len(x)), "native_dtype": str(data.dtype),
            "native_channels": 1 if np.asarray(data).ndim == 1 else int(np.asarray(data).shape[1]),
            "native_rms_dbfs": s2.rms_dbfs(x)}
    if native_sr != sr:
        g = gcd(int(native_sr), int(sr))
        x = resample_poly(x, sr // g, native_sr // g)
        info["resampled"] = f"scipy.signal.resample_poly up {sr // g} down {native_sr // g}"
    x, gain = s2.normalise_rms(x, target_dbfs)
    info.update({"rate_hz": int(sr), "samples": int(len(x)), "duration_s": len(x) / sr, "gain": gain,
                 "rms_dbfs": s2.rms_dbfs(x), "peak_abs": float(np.abs(x).max())})
    return x, info


# ----------------------------------------------------------------------------- RLR side (Linux only)
_LIB = None


def _lib(rlr_root: Path):
    global _LIB
    if _LIB is None:
        import rlr_binding as rlr
        _LIB = rlr.load_library(rlr_root / "RLRAudioPropagationPkg/libs/linux/x64/libRLRAudioPropagation.so")
    return _LIB


def preflight(cfg: dict, rlr_root: Path, hrtf: Path) -> dict:
    """Free field, direct sound only (indirect off), the run's settings otherwise; listener at (0, 0, 1.3) facing
    each heading in the GS frame; a source 2 m to its left, then to its right. The estimated side must match."""
    import poses
    import rlr_binding as rlr

    dc, pf = cfg["direction_check"], cfg["rlr"]["preflight"]
    sr = float(cfg["rlr"]["sample_rate_hz"])
    ov = dict(cfg["rlr"]["overrides"], indirect=False)
    res = {"overrides": ov, "cases": []}
    eye = np.array([0.0, 0.0, 1.3])
    with rlr.RLRContext(_lib(rlr_root), ov, log=lambda *x: None) as ctx:
        res["config_used"] = ctx.config_used
        for h in pf["headings_deg"]:
            fwd = np.array([math.cos(math.radians(h)), math.sin(math.radians(h)), 0.0])
            _f, left, _r, _u = s2.listener_axes(fwd)
            q = poses.listener_quat_from_forward(fwd, ctx.config_used)
            for name, sgn in (("left", 1.0), ("right", -1.0)):
                src = eye + sgn * float(pf["distance_m"]) * left
                ctx.clear_sources()
                ctx.clear_listeners()
                si = ctx.add_source(src)
                li = ctx.add_listener("binaural", eye, q)
                ctx.set_listener_hrtf(li, hrtf)
                ctx.simulate()
                ir = ctx.get_ir(li, si)
                d = s2.source_direction(eye, fwd, src)
                e = s2.estimate_itd(ir, sr, dc["onset_rel"], dc["window_pre_ms"], dc["window_post_ms"], dc["max_lag_ms"])
                res["cases"].append({"heading_deg": h, "source": name, "true_side": d["true_side"],
                                     "azimuth_deg": d["azimuth_deg"], "itd_ms": e["itd_ms"], "ild_db": e["ild_db"],
                                     "cue_side": e["cue_side"], "pass": e["cue_side"] == d["true_side"]})
    res["pass"] = all(c["pass"] for c in res["cases"])
    return res


def render_episode(job: dict) -> dict:
    """One episode; returns the committed record. Writes the local files under job['out']."""
    import c3lib
    import geometry
    import poses
    import rlr_binding as rlr
    from scipy.io import wavfile

    cfg, eid = job["cfg"], job["episode_id"]
    dc, rc = cfg["direction_check"], cfg["rlr"]
    sr = int(rc["sample_rate_hz"])
    out = Path(job["out"])
    edir = out / "episodes" / eid
    edir.mkdir(parents=True)
    t_start = time.perf_counter()
    p = json.loads(Path(job["poses_file"]).read_text(encoding="utf-8"))
    if p["error"] is not None or p["n_frames"] != p["n_poses"] or p["n_poses"] == 0:
        raise RuntimeError(f"{eid}: pose export not usable (error {p['error']}, frames {p['n_frames']}, poses {p['n_poses']})")
    if abs(float(p["listener_height_m"]) - float(cfg["poses"]["listener_height_m_expected"])) > 1e-5:
        raise RuntimeError(f"{eid}: listener height {p['listener_height_m']}")
    bad_z = [fr["frame"] for fr in p["frames"]
             if abs(float(fr["gs_eye"][2]) - float(cfg["poses"]["listener_height_m_expected"])) > 1e-4]
    if bad_z:
        raise RuntimeError(f"{eid}: ear height differs from {cfg['poses']['listener_height_m_expected']} m on frames {bad_z[:5]}")
    fps = float(p["video_fps"])
    N = sr / fps
    if abs(N - round(N)) > 1e-9:
        raise RuntimeError(f"step length {N} samples is not an integer")
    N = int(round(N))

    # 1. scene load
    t0 = time.perf_counter()
    scene_dir = Path(job["scenes_root"]) / p["scene"]
    _structure, labels = geometry.load_scene_files(scene_dir)
    entries = [e for e in labels if str(e.get("ins_id")) == p["target_ins_id"]]
    corners = c3lib.box_corners(entries[0]) if entries else None
    if corners is None:
        raise RuntimeError(f"{eid}: target box not found")
    src = corners.mean(axis=0)
    mat = Path(job["rlr_root"]) / "RLRAudioPropagationPkg/data/mp3d_material_config.json"
    mesh = geometry.build_scene_mesh(scene_dir, mat, lambda q: np.asarray(q, dtype=np.float64),
                                     exclude_ins_ids=[p["target_ins_id"]] if cfg["source"]["exclude_target_box_from_mesh"] else [])
    goal = np.array([p["goal_gs_xy"][0], p["goal_gs_xy"][1], float(corners[:, 2].mean())])
    local = {"episode_id": eid, "scene": p["scene"], "source_gs": src.tolist(), "n_label_entries": len(entries),
             "goal_to_box_horiz_m": c3lib.point_box_distance(goal, corners),
             "source_inside_other_boxes": geometry.boxes_containing(src, labels, [p["target_ins_id"]]),
             "mesh_report": mesh.report}
    t_mesh = time.perf_counter() - t0
    if rc["materials"]:
        raise RuntimeError("materials on is not implemented (the PLAN uses SoundSpaces 2.0's materials off)")
    t0 = time.perf_counter()
    ctx = rlr.RLRContext(_lib(Path(job["rlr_root"])), dict(rc["overrides"]), log=lambda *x: None)
    ctx.add_mesh_object(mesh.vertices, mesh.faces, None)
    si = ctx.add_source(src)
    f0 = p["frames"][0]
    li = ctx.add_listener("binaural", f0["gs_eye"], poses.listener_quat_from_forward(f0["gs_forward"], ctx.config_used))
    ctx.set_listener_hrtf(li, job["hrtf"])
    t_setup = time.perf_counter() - t0

    # 2. per step
    irs, steps, sim_s, chk_s = [], [], [], []
    with open(edir / "steps.jsonl", "w", encoding="utf-8") as fh:
        for fr in p["frames"]:
            eye, fwd = np.asarray(fr["gs_eye"], dtype=np.float64), np.asarray(fr["gs_forward"], dtype=np.float64)
            ctx.set_listener_pose(li, eye, poses.listener_quat_from_forward(fwd, ctx.config_used))
            sim_s.append(ctx.simulate())
            t1 = time.perf_counter()
            ir = ctx.get_ir(li, si)
            los = not ctx.segment_blocked(eye, src, float(dc["los_end_margin_m"]))
            d = s2.source_direction(eye, fwd, src)
            e = s2.estimate_itd(ir, sr, dc["onset_rel"], dc["window_pre_ms"], dc["window_post_ms"], dc["max_lag_ms"])
            counted = s2.is_counted(los, d["lateral_deg"], dc["min_lateral_deg"])
            rec = {"step": fr["frame"], "time_s": fr["frame"] / fps, "los": los, **d,
                   "itd_ms": e["itd_ms"], "ild_db": e["ild_db"], "onset_index": e["onset_index"],
                   "cue_side": e["cue_side"], "counted": counted,
                   "correct": bool(counted and e["cue_side"] == d["true_side"]),
                   "ir_samples": int(ir.shape[1]), "simulate_s": sim_s[-1]}
            fh.write(json.dumps(rec) + "\n")
            steps.append(rec)
            irs.append(ir)
            chk_s.append(time.perf_counter() - t1)
    ctx.close()

    # 3. clips and saved responses
    t0 = time.perf_counter()
    source = job["source"]
    offset = int(np.random.default_rng(int(cfg["seed"]) + int(job["order_position"])).integers(N))
    clip = s2.synthesize(source, irs, N, offset, int(round(float(cfg["clips"]["crossfade_s"]) * sr)))
    clip *= 10.0 ** (float(cfg["clips"]["gain_db"]) / 20.0)
    mono = s2.downmix(clip)
    (out / "clips").mkdir(exist_ok=True)
    wavfile.write(str(out / "clips" / f"{eid}_two_ear.wav"), sr, clip.T.astype(np.float32))
    wavfile.write(str(out / "clips" / f"{eid}_one_channel.wav"), sr, mono.astype(np.float32))
    t_clip = time.perf_counter() - t0
    t0 = time.perf_counter()
    padded, lens = s2.pad_irs(irs)
    np.save(edir / "brirs.npy", padded)
    np.save(edir / "brir_lengths.npy", lens)
    t_save = time.perf_counter() - t0
    local.update({"start_offset_samples": offset, "step_samples": N, "clip_samples": int(clip.shape[1]),
                  "clip_peak_abs": float(np.abs(clip).max()), "video_frames": p["n_frames"]})
    (edir / "episode.json").write_text(json.dumps(local, indent=1, default=float), encoding="utf-8")

    t = s2.tally(steps)
    sa = np.asarray(sim_s)
    return {
        "episode_id": eid, "scene": p["scene"], "order_position": int(job["order_position"]),
        "visible_start": int(job["order_position"]) % 2 == 0, "n_frames": p["n_frames"], "video_fps": fps,
        "clip_duration_s": clip.shape[1] / sr, **t,
        "timing_s": {"pose_export_render": p["render_s"], "scene_load_mesh": t_mesh, "scene_load_context": t_setup,
                     "simulate_total": float(sa.sum()), "simulate_median": float(np.median(sa)),
                     "simulate_p90": float(np.percentile(sa, 90)), "simulate_max": float(sa.max()),
                     "step_checks_total": float(np.sum(chk_s)), "clips": t_clip, "save_brirs": t_save,
                     "episode_total": time.perf_counter() - t_start},
        "brir_samples_max": int(lens.max()), "brir_bytes": int(padded.nbytes),
    }


def main(argv=None) -> int:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=EXP / "config.yaml")
    ap.add_argument("--order", type=Path, default=None, help="default: the config's episodes.order_file")
    ap.add_argument("--n", type=int, required=True, help="first N ids of the order (batch 1: 12)")
    ap.add_argument("--poses-dir", type=Path, required=True, help="export_poses.py output folder (holds poses/)")
    ap.add_argument("--scenes-root", type=Path, required=True)
    ap.add_argument("--rlr-root", type=Path, required=True)
    ap.add_argument("--hrtf", type=Path, required=True)
    ap.add_argument("--sound", type=Path, required=True, help="the extracted telephone.wav")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--preflight-only", action="store_true")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    pend = pending_values(cfg)
    if pend:
        print(f"REFUSED: config values still pending: {pend}", file=sys.stderr)
        return 2
    if a.out.exists():
        ap.error(f"--out {a.out} exists; results are never overwritten")
    order_file = a.order or (EXP / cfg["episodes"]["order_file"])
    order_txt = order_file.read_text(encoding="utf-8")
    if s2.sha256_text(order_txt) != cfg["episodes"]["order_sha256"]:
        print("REFUSED: episode order sha256 differs from the config", file=sys.stderr)
        return 2
    ids = [x for x in order_txt.split() if x][: a.n]
    lib_path = a.rlr_root / "RLRAudioPropagationPkg/libs/linux/x64/libRLRAudioPropagation.so"
    hashes = {"library": s2.sha256_file(lib_path), "hrtf": s2.sha256_file(a.hrtf), "sound_wav": s2.sha256_file(a.sound)}
    want = {"library": cfg["pins"]["rlr_library_sha256"], "hrtf": cfg["rlr"]["hrtf"]["sha256"],
            "sound_wav": cfg["sound"]["wav_sha256"]}
    bad = {k: (hashes[k], want[k]) for k in hashes if hashes[k] != want[k]}
    if bad:
        print(f"REFUSED: sha256 mismatch {bad}", file=sys.stderr)
        return 2

    import rlr_binding as rlr
    if rlr.struct_sizes() != rlr.EXPECTED_SIZES:
        print(f"REFUSED: ctypes struct sizes {rlr.struct_sizes()}", file=sys.stderr)
        return 2
    a.out.mkdir(parents=True)
    log = open(a.out / "render.log", "a", buffering=1, encoding="utf-8")

    def say(*x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True)
        log.write(s + "\n")

    sr = int(cfg["rlr"]["sample_rate_hz"])
    source, src_info = load_source(a.sound, sr, float(cfg["sound"]["loudness"]["target_dbfs"]))
    t_pf = time.perf_counter()
    pf = preflight(cfg, a.rlr_root, a.hrtf)
    pf["wall_s"] = time.perf_counter() - t_pf
    (a.out / "preflight.json").write_text(json.dumps(pf, indent=2, default=float), encoding="utf-8")
    for c in pf["cases"]:
        say(f"[preflight] heading {c['heading_deg']:>3} source {c['source']:5s}: itd {c['itd_ms']:+.3f} ms, "
            f"ild {c['ild_db']:+.1f} dB, {'PASS' if c['pass'] else 'FAIL'}")
    run_config = {
        "argv": sys.argv, "config": cfg, "config_sha256": s2.sha256_file(a.config), "order_sha256": s2.sha256_text(order_txt),
        "n": a.n, "episode_ids": ids, "hashes": hashes, "source": src_info,
        "repo_commit": os.environ.get("REPO_COMMIT", "not set"), "python": sys.version, "numpy": np.__version__,
        "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "workers": a.workers or int(cfg["run"]["workers"]), "struct_sizes": rlr.struct_sizes(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    import scipy
    run_config["scipy"] = scipy.__version__
    (a.out / "run_config.json").write_text(json.dumps(run_config, indent=2, default=float), encoding="utf-8")
    if not pf["pass"]:
        say("[preflight] FAIL: the side sign is wrong in free field; no episode rendered. Stop and tell the lead.")
        return 3
    if a.preflight_only:
        say("[preflight] PASS (preflight only)")
        return 0

    jobs = []
    for i, eid in enumerate(ids):
        jobs.append({"cfg": cfg, "episode_id": eid, "order_position": i, "out": str(a.out),
                     "poses_file": str(a.poses_dir / "poses" / f"{eid}.json"), "scenes_root": str(a.scenes_root),
                     "rlr_root": str(a.rlr_root), "hrtf": str(a.hrtf), "source": source})
    workers = run_config["workers"]
    t_all = time.perf_counter()
    records, failed = [], {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as ex:
        futs = {ex.submit(render_episode, j): j["episode_id"] for j in jobs}
        for f in as_completed(futs):
            eid = futs[f]
            try:
                r = f.result()
                records.append(r)
                say(f"[episode] {eid}: {r['n_steps']} steps, counted {r['n_counted']}, correct {r['n_correct']}, "
                    f"occluded {r['n_occluded']}, total {r['timing_s']['episode_total']:.1f} s")
            except Exception as exc:  # noqa: BLE001 - record and continue; a failed episode is reported
                failed[eid] = repr(exc)
                say(f"[episode] {eid}: FAILED {exc!r}")
    wall = time.perf_counter() - t_all
    records.sort(key=lambda r: r["order_position"])
    with open(a.out / "episode_records.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, default=float) + "\n")
    timing = {"wall_s_all_episodes": wall, "workers": workers, "n_done": len(records), "failed": failed,
              "preflight_s": pf["wall_s"],
              "per_episode_total_s": {r["episode_id"]: r["timing_s"]["episode_total"] for r in records},
              "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (a.out / "timing.json").write_text(json.dumps(timing, indent=2, default=float), encoding="utf-8")
    say(f"[done] {len(records)} episodes in {wall:.1f} s with {workers} workers; {len(failed)} failed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
