"""Measured-HRTF follow-up to the failed left_ear_before_right_ear test of check 1 (not a replacement for it).

Question: the diagnostic (diagnose_left_ear.py) showed that the library's default HRTF gives a level
difference between the ears but almost no interaural time difference (within +-0.09 ms at every angle in free
field). With a measured HRTF (MIT KEMAR, normal pinna, SOFA file), does the library produce a proper
interaural time difference, and does check 1's left_ear_before_right_ear test pass?

For each HRTF condition in [default, kemar] and each settings set in config.yaml check1.settings, with the same
listener, source and heading as the run recorded (<run-dir>/<settings>/run_record.json "left_test"):
  sweep_free_field  no mesh, indirect = false, source 2 m (check1.side_source_offset_m) away at azimuths
                    0..330 deg step 30 (+90 = the listener's left, poses.py convention), heading 0
  left_test         the check 1 left-ear placement in the full scene mesh (goal box excluded, as the run),
                    with check 1's settings unchanged; pass_tests.test_left_before_right applied unchanged
Per impulse response it prints and writes the ear_stats fields (diagnose_left_ear.ear_stats), the condition
and a PASS/FAIL line from pass_tests.test_left_before_right (a test line for left_test and az090, where the
left ear is expected first; an info line, same function, for the other azimuths).
Reproduction check: for the default condition the free-field numbers should reproduce the earlier diagnostic
(sweep_free_field_heading0 in diagnose_left_ear.py's output); the log header says so.
Writes everything under --out, which must not exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pass_tests as pt  # noqa: E402
import poses  # noqa: E402
from diagnose_left_ear import ear_stats, fmt  # noqa: E402

CONDITIONS = ("default", "kemar")
AZIMUTHS_DEG = tuple(range(0, 360, 30))
TEST_TAGS = ("left_test", "az090")  # IRs where the left ear is expected first


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=HERE.parent.parent / "config.yaml")
    ap.add_argument("--rlr-root", type=Path, required=True)
    ap.add_argument("--scenes-root", type=Path, required=True)
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True, help="the check 1 run output folder, e.g. results/check1-run2 (needs run_info.json and <settings>/run_record.json)")
    ap.add_argument("--hrtf", type=Path, required=True, help="the SOFA file for the kemar condition (mit_kemar_normal_pinna.sofa)")
    ap.add_argument("--out", type=Path, required=True, help="must not exist")
    return ap


def parse_args(argv=None) -> argparse.Namespace:
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.out.exists():
        ap.error(f"--out {a.out} already exists; results are never overwritten")
    if a.hrtf.suffix.lower() != ".sofa":
        ap.error(f"--hrtf {a.hrtf} must end in .sofa (the library picks the format from the extension)")
    return a


def sweep_positions(listener, heading_deg: float, offset_m: float, azimuths=AZIMUTHS_DEG):
    """(tag, listener position, source position) per azimuth, the triples simulate() unpacks; +90 deg is the listener's left (poses.agent_axes convention)."""
    listener = np.asarray(listener, dtype=np.float64)
    fwd, left, _r, _u = poses.agent_axes(heading_deg)
    out = []
    for az in azimuths:
        t = math.radians(az)
        out.append((f"az{az:03d}", listener, listener + offset_m * (math.cos(t) * fwd + math.sin(t) * left)))
    return out


def test_line(ir: np.ndarray, sr: float, run: str, tag: str) -> dict:
    t = pt.test_left_before_right(ir, sr, run, True)
    kind = "test" if tag in TEST_TAGS else "info"
    t["line"] = f"[{kind}] " + t["line"]
    t["expected_left_first"] = tag in TEST_TAGS
    return t


def main(argv=None) -> int:
    a = parse_args(argv)
    import pandas as pd
    import yaml
    import geometry
    import gnbench_copy as gb
    import overlay
    import rlr_binding as rlr
    from run_check1 import Log, choose_episode

    cfg_all = yaml.safe_load(a.config.read_text())
    cfg = cfg_all["check1"]
    sr = float(cfg["sample_rate_hz"])
    a.out.mkdir(parents=True, exist_ok=False)
    log = Log(a.out / "hrtf_followup.log")
    run_info = json.loads((a.run_dir / "run_info.json").read_text())
    scene = run_info["episode"]["scene"]
    scene_dir = a.scenes_root / scene
    lib_path = a.rlr_root / "RLRAudioPropagationPkg/libs/linux/x64/libRLRAudioPropagation.so"
    mat_path = a.rlr_root / "RLRAudioPropagationPkg/data/mp3d_material_config.json"
    if not a.hrtf.is_file():
        log(f"ABORT: SOFA file {a.hrtf} not found")
        return 2
    sofa_sha = sha256(a.hrtf)
    log("[header] measured-HRTF follow-up to check 1 (see the module docstring of hrtf_followup.py)")
    log(f"[header] reproduction check: for the default condition the free-field sweep (heading 0, direct only) should "
        f"reproduce the earlier diagnostic (diagnose_left_ear.py, sweep_free_field_heading0): level difference "
        f"between the ears, interaural delay of largest magnitude 0.091 ms over all angles")
    log(f"[header] SOFA file {a.hrtf} sha256 {sofa_sha}")
    log(f"[header] library {lib_path} sha256 {sha256(lib_path)}")
    log(f"[header] config {a.config} sha256 {sha256(a.config)}; sample rate {sr}; speed of sound {cfg['speed_of_sound_m_s']}")
    log(f"[header] settings sets {json.dumps(cfg['settings'])}; run dir {a.run_dir}; conditions {list(CONDITIONS)}")

    df = pd.read_parquet(a.parquet)
    row = choose_episode(overlay.scene_rows(df, scene), int(cfg_all["seed"]))
    if int(row["trajectory_id"]) != int(run_info["episode"]["trajectory_id"]):
        log("ABORT: episode differs from the run's", row["trajectory_id"], run_info["episode"]["trajectory_id"])
        return 2
    meta = gb.load_occupancy_metadata(scene_dir)
    affine = gb.episode_affine(row, meta)
    full = geometry.build_scene_mesh(scene_dir, mat_path, lambda p: gb.gs_points_to_world(p, affine, meta),
                                     exclude_ins_ids=[str(row["label_ins_id"])])
    log(f"[setup] scene {scene}, trajectory {row['trajectory_id']}, full mesh {full.report['n_faces']} triangles")

    lib = rlr.load_library(lib_path)
    off = float(cfg["side_source_offset_m"])
    results = {"sofa": str(a.hrtf), "sofa_sha256": sofa_sha, "runs": {}}

    def simulate(cond, sname, variant, mesh, extra, pairs, hd):
        ov = dict(cfg["settings"][sname])
        ov["sampleRate"] = sr
        ov.update(extra)
        key = f"{cond}/{sname}/{variant}"
        log(f"[variant] {key}: configuration overrides {ov}, heading {hd}, HRTF {'library default' if cond == 'default' else a.hrtf}")
        irs = {}
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
                    if cond == "kemar":
                        ctx.set_listener_hrtf(li, a.hrtf)
                    ctx.simulate()
                    ir = ctx.get_ir(li, si)
                    np.save(a.out / f"{cond}_{sname}_{variant}_{tag}.npy", ir)
                    s = ear_stats(ir, sr)
                    s["distance_m"] = float(np.linalg.norm(spos - lpos))
                    t = test_line(ir, sr, f"{cond}/{sname}/{variant}", tag) if ir.shape[0] >= 2 and ir.shape[1] else None
                    irs[tag] = {"condition": cond, "stats": s, "test": t}
                    log(fmt(f"{cond} {sname} {variant} {tag}", s))
                    if t:
                        log("    " + t["line"])
        except Exception as exc:  # noqa: BLE001 - record and continue
            log(f"[variant] {key} failed: {exc!r}")
            results["runs"][key] = {"error": repr(exc), "irs": irs}
            return
        results["runs"][key] = {"overrides": ov, "heading_deg": hd, "irs": irs}

    for sname in cfg["settings"]:
        rec = json.loads((a.run_dir / sname / "run_record.json").read_text())
        lt = rec.get("left_test")
        if not lt:
            log(f"[skip] {sname}: run_record.json has no left_test placement")
            continue
        lis, src, heading = np.array(lt["listener"]), np.array(lt["source"]), float(lt["heading_deg"])
        log(f"[setup] {sname}: left_test listener {lis.tolist()}, source {src.tolist()}, heading {heading}")
        for cond in CONDITIONS:
            simulate(cond, sname, "sweep_free_field", None, {"indirect": False},
                     sweep_positions(lis, 0.0, off), 0.0)
            simulate(cond, sname, "left_test", full, {}, [("left_test", lis, src)], heading)

    log("[summary] free-field sweep, interaural delay (xcorr, right minus left, ms) at az090 (left) and az270 (right), "
        "largest |delay| over all azimuths, and the check 1 test result")
    for key, r in results["runs"].items():
        irs = r.get("irs", {})
        if "sweep_free_field" in key:
            d = {t: v["stats"].get("xcorr_itd_ms_R_minus_L") for t, v in irs.items()}
            vals = [abs(x) for x in d.values() if x is not None]
            log(f"[summary] {key}: az090 {d.get('az090')}, az270 {d.get('az270')}, max |itd| {max(vals) if vals else None}")
        else:
            for t, v in irs.items():
                log(f"[summary] {key}: {v['test']['line'] if v['test'] else 'no result'}")
    (a.out / "hrtf_followup_summary.json").write_text(json.dumps(results, indent=2, default=str))
    log(f"[done] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
