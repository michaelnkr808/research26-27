"""Diagnostic for the failed left_ear_before_right_ear test of check1-run1 (not a replacement for it).

Same scene, episode, mesh, listener and source as the run's left-ear test (read from the run's
run_info.json and <settings>/run_record.json), simulated again with the real library in variants that
separate the possible causes:
  full_scene              the run's own setup (should reproduce the saved IR)
  no_boxes                walls, floors and ceilings only: all labels.json boxes removed
  full_scene_direct_only  full mesh, indirect = false (no reflections, no diffraction)
  free_field_direct_only  no mesh at all, indirect = false
  sweep_free_field        free field, direct only, source 2 m away at azimuths 0..330 deg (step 30,
                          +90 = the listener's left), at the run's heading and at heading 0
For each impulse response it prints, per ear: onset at 0.1 x the channel's own peak (the pass test's
measure), onset at 0.1 x the larger of the two ear peaks (common threshold), the time of each ear's
peak, the interaural delay from cross-correlation of the first 3 ms after the first non-zero sample
(positive = right ear later), and the level difference in dB over that window (positive = left louder).
Writes everything under --out, which must not exist. Settings: config.yaml check1.settings[--settings]
plus sampleRate, exactly as run_check1.py, with the variant's extra fields logged.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
from run_check1 import Log, choose_episode  # noqa: E402

XCORR_WINDOW_MS = 3.0


def ear_stats(ir: np.ndarray, sr: float) -> dict:
    L, R = ir[0].astype(np.float64), ir[1].astype(np.float64)
    gpk = max(np.abs(L).max(), np.abs(R).max())
    out = {"peak_L": float(np.abs(L).max()), "peak_R": float(np.abs(R).max())}
    if gpk <= 0:
        out["empty"] = True
        return out
    first = int(min(np.argmax(L != 0), np.argmax(R != 0)))
    ms = lambda i: None if i is None else 1000.0 * i / sr  # noqa: E731

    def common_onset(x):
        hit = np.abs(x) >= 0.1 * gpk
        return int(np.argmax(hit)) if hit.any() else None

    n = int(XCORR_WINDOW_MS * sr / 1000.0)
    wl, wr = L[first:first + n], R[first:first + n]
    cc = np.correlate(wr, wl, "full")
    lag = int(np.argmax(np.abs(cc))) - (len(wl) - 1)
    el, er = float(np.sum(wl ** 2)), float(np.sum(wr ** 2))
    out.update({
        "first_nonzero_ms": ms(first),
        "onset_own_peak_L_ms": ms(pt.first_arrival_index(L)), "onset_own_peak_R_ms": ms(pt.first_arrival_index(R)),
        "onset_common_L_ms": ms(common_onset(L)), "onset_common_R_ms": ms(common_onset(R)),
        "peak_time_L_ms": ms(int(np.argmax(np.abs(L)))), "peak_time_R_ms": ms(int(np.argmax(np.abs(R)))),
        "xcorr_itd_ms_R_minus_L": 1000.0 * lag / sr,
        "ild_db_L_over_R": 10.0 * math.log10(el / er) if el > 0 and er > 0 else None,
    })
    return out


def fmt(name: str, s: dict) -> str:
    if s.get("empty"):
        return f"{name:34s} empty IR"
    f = lambda v: "  none" if v is None else f"{v:6.3f}"  # noqa: E731
    return (f"{name:34s} first {f(s['first_nonzero_ms'])} | own-peak onset L {f(s['onset_own_peak_L_ms'])} "
            f"R {f(s['onset_own_peak_R_ms'])} | common onset L {f(s['onset_common_L_ms'])} R {f(s['onset_common_R_ms'])} "
            f"| peak L {f(s['peak_time_L_ms'])} R {f(s['peak_time_R_ms'])} | xcorr R-L {s['xcorr_itd_ms_R_minus_L']:+.3f} ms "
            f"| ILD {s['ild_db_L_over_R'] if s['ild_db_L_over_R'] is None else round(s['ild_db_L_over_R'], 1)} dB")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=HERE.parent.parent / "config.yaml")
    ap.add_argument("--rlr-root", type=Path, required=True)
    ap.add_argument("--scenes-root", type=Path, required=True)
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True, help="the check1-run1 output folder")
    ap.add_argument("--settings", default="library_defaults")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    cfg_all = yaml.safe_load(a.config.read_text())
    cfg = cfg_all["check1"]
    sr = float(cfg["sample_rate_hz"])
    a.out.mkdir(parents=True, exist_ok=False)
    log = Log(a.out / "diagnose.log")
    run_info = json.loads((a.run_dir / "run_info.json").read_text())
    rec = json.loads((a.run_dir / a.settings / "run_record.json").read_text())
    lt = rec["left_test"]
    lis, src, heading = np.array(lt["listener"]), np.array(lt["source"]), float(lt["heading_deg"])
    scene = run_info["episode"]["scene"]
    scene_dir = a.scenes_root / scene
    lib_path = a.rlr_root / "RLRAudioPropagationPkg/libs/linux/x64/libRLRAudioPropagation.so"
    mat_path = a.rlr_root / "RLRAudioPropagationPkg/data/mp3d_material_config.json"

    df = pd.read_parquet(a.parquet)
    row = choose_episode(overlay.scene_rows(df, scene), int(cfg_all["seed"]))
    if int(row["trajectory_id"]) != int(run_info["episode"]["trajectory_id"]):
        log("ABORT: episode differs from the run's", row["trajectory_id"], run_info["episode"]["trajectory_id"])
        return 2
    meta = gb.load_occupancy_metadata(scene_dir)
    affine = gb.episode_affine(row, meta)
    to_world = lambda p: gb.gs_points_to_world(p, affine, meta)  # noqa: E731
    full = geometry.build_scene_mesh(scene_dir, mat_path, to_world, exclude_ins_ids=[str(row["label_ins_id"])])
    _, labels = geometry.load_scene_files(scene_dir)
    no_boxes = geometry.build_scene_mesh(scene_dir, mat_path, to_world,
                                         exclude_ins_ids=[str(e.get("ins_id")) for e in labels])
    log(f"[setup] scene {scene}, trajectory {row['trajectory_id']}, listener {lis.tolist()}, source {src.tolist()}, "
        f"heading {heading}; full mesh {full.report['n_faces']} triangles, no-boxes mesh {no_boxes.report['n_faces']}")

    lib = rlr.load_library(lib_path)
    base = dict(cfg["settings"][a.settings])
    base["sampleRate"] = sr
    results = {}

    def simulate(name, mesh, extra, pairs, hd):
        ov = dict(base, **extra)
        log(f"[variant] {name}: overrides {ov}")
        outs = []
        try:
            with rlr.RLRContext(lib, ov, log=log) as ctx:
                ctx.set_material_database(mat_path)
                if mesh is not None:
                    ctx.add_mesh_object(mesh.vertices, mesh.faces, mesh.face_category)
                q = poses.listener_quat_wxyz(hd, ctx.config_used)
                for tag, lpos, spos in pairs:
                    ctx.clear_sources()
                    ctx.clear_listeners()
                    si = ctx.add_source(spos)
                    li = ctx.add_listener("binaural", lpos, q)
                    ctx.simulate()
                    ir = ctx.get_ir(li, si)
                    np.save(a.out / f"{name}_{tag}.npy", ir)
                    s = ear_stats(ir, sr)
                    s["distance_m"] = float(np.linalg.norm(spos - lpos))
                    s["expected_direct_ms"] = 1000.0 * s["distance_m"] / float(cfg["speed_of_sound_m_s"])
                    outs.append((tag, s))
                    log(fmt(f"{name} {tag}", s) + f" | expected direct {s['expected_direct_ms']:.3f} ms")
        except Exception as exc:  # noqa: BLE001 - record and continue with the other variants
            log(f"[variant] {name} failed: {exc!r}")
            results[name] = {"error": repr(exc)}
            return
        results[name] = {"overrides": ov, "heading_deg": hd, "irs": dict(outs)}

    one = [("left_test", lis, src)]
    simulate("full_scene", full, {}, one, heading)
    saved = np.load(a.run_dir / a.settings / "ir_binaural_left_source.npy")
    rep = np.load(a.out / "full_scene_left_test.npy") if (a.out / "full_scene_left_test.npy").exists() else None
    if rep is not None:
        same = rep.shape == saved.shape and bool(np.array_equal(rep, saved))
        log(f"[reproduce] full_scene identical to the saved run IR: {same}"
            + ("" if same or rep.shape != saved.shape else f" (max abs diff {float(np.abs(rep - saved).max()):.3g})"))
    simulate("no_boxes", no_boxes, {}, one, heading)
    simulate("full_scene_direct_only", full, {"indirect": False}, one, heading)
    simulate("free_field_direct_only", None, {"indirect": False}, one, heading)
    off = float(cfg["side_source_offset_m"])
    for hd in sorted({heading, 0.0}):
        fwd, left, _r, _u = poses.agent_axes(hd)
        pairs = []
        for az in range(0, 360, 30):
            t = math.radians(az)
            pairs.append((f"az{az:03d}", lis, lis + off * (math.cos(t) * fwd + math.sin(t) * left)))
        simulate(f"sweep_free_field_heading{int(hd)}", None, {"indirect": False}, pairs, hd)
    (a.out / "diagnose_summary.json").write_text(json.dumps(results, indent=2, default=str))
    log(f"[done] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
