"""First smoke test of the perturbed-frame experiment (experiments/2026-11-06-perturbed-frames, PLAN Handoff steps 1 and 2).

Runs in the pod of Job stacking-encoders-pf-smoke (the proxy eval Jobs' environment). One task file, three train/val demos,
never a test demo. Stages (each records its checks in report.json; a failed gating check stops the run, PLAN Q8):
  q23     secondary mean-feature size per encoder, read from the cached feature h5 files (read only)
  scenes  scene XML of light and background variants vs the base scene
  demos   the tally's demo-level split (diagnostics.demo_split), 3 train/val demos, fixture list from model_file differences
  clean   Q20 check, fixture copy, state restore, clean 256 render, two shrink methods vs the stored 128 frames (Q22)
  camera  zero-change camera name: state length, frame match, camera pose vs the demo's saved pose (Q2 override only if it differs)
  perturb one frame each for camera, light, background and noise (noise at 256, then shrunk)
The validation check (D probe val R^2 within 0.01) is a later step and is not done here.
"""

import argparse
import difflib
import hashlib
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

import diagnostics            # the tally's own split (src/diagnostics.py, same sha256 as the proxy-src ConfigMap's copy)
import pf_render as R

ENCODERS = {"dinov3": "cephfs", "sam3": "cephfs", "vggt-omega": "cephfs", "siglip2": "rbd", "c-radiov4": "rbd"}
RAW_SHA256 = "387fc10747696b80dea6ed8d7f2beaa162bf92ae11750b241073cbd33aac73d5"   # PLAN Feasibility 7b
CAM_TOL = 1e-4          # camera pose difference above this is a real difference; the 4-decimal rounding gives <= ~5e-5
N_FRAMES = 6            # stored moments per demo


class Stop(Exception):
    pass


REPORT = {"checks": {}, "stages": {}, "stopped_at": None, "notes": []}


def check(name, ok, gating=True, **details):
    REPORT["checks"][name] = {"pass": bool(ok), "gating": gating, **details}
    print(f"[check] {name}: {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok and gating:
        raise Stop(name)


def stage(name):
    def deco(fn):
        def run(*a, **k):
            if REPORT["stopped_at"]:
                REPORT["stages"][name] = {"skipped": f"stopped at {REPORT['stopped_at']}"}
                return None
            t0 = time.time()
            print(f"=== stage {name}", flush=True)
            try:
                out = fn(*a, **k)
                REPORT["stages"][name] = {"seconds": round(time.time() - t0, 1), "ok": True}
                return out
            except Stop as e:
                REPORT["stopped_at"] = str(e)
                REPORT["stages"][name] = {"seconds": round(time.time() - t0, 1), "ok": False, "stopped_by": str(e)}
            except Exception as e:                                      # a crash is a failed check, and stops (Q8)
                tb = traceback.format_exc()
                print(tb, flush=True)
                REPORT["checks"][f"{name}_no_exception"] = {"pass": False, "gating": True, "error": repr(e), "traceback": tb}
                REPORT["stopped_at"] = f"{name}_no_exception"
                REPORT["stages"][name] = {"seconds": round(time.time() - t0, 1), "ok": False, "error": repr(e)}
            return None
        return run
    return deco


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# q23: mean-feature sizes
# ---------------------------------------------------------------------------

@stage("q23")
def stage_q23(args):
    roots = {"cephfs": args.feats_cephfs, "rbd": args.feats_rbd}
    sizes, per_enc = {}, {}
    for enc, vol in ENCODERS.items():
        p = Path(roots[vol]) / enc / "libero_goal" / (args.task + ".h5")
        e = {"path": str(p), "exists": p.exists()}
        per_enc[enc] = e
        if not p.exists():
            continue
        with h5py.File(p, "r") as f:
            e["n_demos"] = len(f.keys())
            d0 = f["demo_0"]
            e["cams"] = {}
            for cam in diagnostics.CAMS:
                e["cams"][cam] = {"mean_shape": list(d0[cam]["mean"].shape), "global_shape": list(d0[cam]["global"].shape),
                                  "mean_dtype": str(d0[cam]["mean"].dtype)}
            try:
                e["meta"] = json.loads(f.attrs["meta"])
            except Exception as ex:
                e["meta"] = f"unreadable: {ex!r}"
        dims = {c: v["mean_shape"][-1] for c, v in e["cams"].items()}
        sizes[enc] = dims
    check("q23_all_five_feature_files_read", len(sizes) == len(ENCODERS), files={k: v["exists"] for k, v in per_enc.items()})
    check("q23_both_cameras_same_size", all(len(set(d.values())) == 1 for d in sizes.values()), gating=False, sizes=sizes)
    mean_dim = {e: list(d.values())[0] for e, d in sizes.items()}
    smallest = min(mean_dim.values())
    REPORT["q23"] = {"mean_dim_per_camera": mean_dim, "smallest": smallest,
                     "differ": len(set(mean_dim.values())) > 1,
                     "reduce_to_smallest_with_PCA": sorted(e for e, v in mean_dim.items() if v > smallest),
                     "details": per_enc}
    print("[q23]", json.dumps({k: REPORT["q23"][k] for k in ("mean_dim_per_camera", "smallest", "differ")}), flush=True)


# ---------------------------------------------------------------------------
# scenes: light and background XML vs base
# ---------------------------------------------------------------------------

def _flatten(root):
    out, counts = {}, Counter()

    def walk(e, path):
        nm = e.get("name") or e.get("id") or e.get("file") or ""
        k = f"{path}/{e.tag}" + (f"[{nm}]" if nm else "")
        counts[k] += 1
        if counts[k] > 1:
            k += f"#{counts[k]}"
        out[k] = (dict(e.attrib), (e.text or "").strip())
        for c in e:
            walk(c, k)
    walk(root, "")
    return out


def xml_diff(base_path, var_path, cap=60):
    import xml.etree.ElementTree as ET
    a, b = _flatten(ET.parse(base_path).getroot()), _flatten(ET.parse(var_path).getroot())
    removed = sorted(set(a) - set(b))
    added = sorted(set(b) - set(a))
    changed = []
    for k in sorted(set(a) & set(b)):
        (aa, at), (ba, bt) = a[k], b[k]
        for attr in sorted(set(aa) | set(ba)):
            if aa.get(attr) != ba.get(attr):
                changed.append({"element": k, "attr": attr, "base": aa.get(attr), "variant": ba.get(attr)})
        if at != bt:
            changed.append({"element": k, "attr": "#text", "base": at[:80], "variant": bt[:80]})
    ta, tb = Path(base_path).read_text().splitlines(), Path(var_path).read_text().splitlines()
    ud = list(difflib.unified_diff(ta, tb, "base", "variant", lineterm="", n=0))
    return {"n_changed_attrs": len(changed), "n_added": len(added), "n_removed": len(removed),
            "changed_attrs": changed[:cap], "added": added[:cap], "removed": removed[:cap],
            "unified_diff_head": ud[:cap], "unified_diff_lines": len(ud),
            "base_bytes": Path(base_path).stat().st_size, "variant_bytes": Path(var_path).stat().st_size}


@stage("scenes")
def stage_scenes(args, envs):
    base = args.task_base
    names = {"base": base, "light_1": f"{base}_light_1", "light_11": f"{base}_light_11",
             "background_table_1": f"{base}_table_1", "background_tb_13": f"{base}_tb_13"}
    info = {}
    for label, nm in names.items():
        e = envs.get(nm)
        inn = R.inner(e)
        info[label] = {"bddl": nm, "problem_name": getattr(e, "problem_name", None), "scene_xml": inn._arena_xml,
                       "scene_properties": inn._arena_properties}
        if label != "base":
            envs.release(nm)              # keep memory low: one env at a time besides the base
    base_xml = info["base"]["scene_xml"]
    for label in list(info):
        if label == "base":
            continue
        info[label]["diff_vs_base"] = xml_diff(base_xml, info[label]["scene_xml"])
        d = info[label]["diff_vs_base"]
        print(f"[scenes] {label}: {Path(info[label]['scene_xml']).name}: {d['n_changed_attrs']} changed attrs, "
              f"{d['n_added']} added, {d['n_removed']} removed elements", flush=True)
    REPORT["scenes"] = info
    check("scenes_read", all("diff_vs_base" in v for k, v in info.items() if k != "base"))


# ---------------------------------------------------------------------------
# demos
# ---------------------------------------------------------------------------

@stage("demos")
def stage_demos(args, ctx):
    fr = ctx["raw"]
    demos = sorted(fr["data"].keys(), key=lambda k: int(k.split("_")[-1]))
    # the tally's own filter keeps demos that every encoder's feature file has; here the five files all have all 50 demos
    split = diagnostics.demo_split(demos)
    counts = Counter(split.values())
    train = [d for d in demos if split[d] == "train"]
    val = [d for d in demos if split[d] == "val"]
    chosen = [train[0], train[-1], val[0]]
    check("demos_none_is_test", all(split[d] != "test" for d in chosen), chosen=chosen, split_of_chosen={d: split[d] for d in chosen},
          split_counts=dict(counts), n_demos=len(demos))
    nontest = [d for d in demos if split[d] != "test"]
    # model_file differences across train+val demos only (the 10 test demos are not opened at all)
    parsed_all = {d: R.parse_model_file(fr["data"][d].attrs["model_file"]) for d in nontest}
    fixtures, info = R.differing_bodies([parsed_all[d] for d in nontest])
    check("demos_fixture_list_found", len(fixtures) >= 1, fixtures=fixtures, info=info, n_demos_compared=len(nontest),
          note="from train+val demos only; the test demos were not read")
    cams_equal = all(np.allclose(parsed_all[d]["cameras"]["agentview"]["pos"], parsed_all[nontest[0]]["cameras"]["agentview"]["pos"], atol=0)
                     and np.allclose(parsed_all[d]["cameras"]["agentview"]["quat"], parsed_all[nontest[0]]["cameras"]["agentview"]["quat"], atol=0)
                     for d in nontest)
    REPORT["demos"] = {"chosen": chosen, "split_counts": dict(counts), "fixtures": fixtures, "agentview_camera_same_in_all_nontest": bool(cams_equal)}
    ctx.update(chosen=chosen, parsed=parsed_all, fixtures=fixtures, split=split)


# ---------------------------------------------------------------------------
# clean render, Q20, fixtures, resize
# ---------------------------------------------------------------------------

def frame_indices(n):
    return np.unique(np.linspace(0, n - 2, N_FRAMES).round().astype(int)).tolist()   # image k pairs with state k+1; last image dropped


def maybe_override_camera(env, parsed, label):
    """Q2: set the saved pose only if the env's agentview pose differs from the demo's saved pose (> CAM_TOL)."""
    cur = R.camera_pose(env)
    sv = parsed["cameras"]["agentview"]
    dpos, dq = float(np.abs(cur["pos"] - sv["pos"]).max()), float(np.abs(cur["quat"] - sv["quat"]).max())
    differs = max(dpos, dq) > CAM_TOL
    if differs:
        R.apply_saved_camera(env, parsed)
    return {"env_pos": cur["pos"].tolist(), "saved_pos": sv["pos"].tolist(), "env_quat": cur["quat"].tolist(),
            "saved_quat": sv["quat"].tolist(), "pos_max_diff": dpos, "quat_max_diff": dq, "env_fovy": cur["fovy"],
            "tolerance": CAM_TOL, "differs": bool(differs), "override_used": bool(differs), "label": label}


@stage("clean")
def stage_clean(args, ctx, envs):
    fr, chosen, fixtures = ctx["raw"], ctx["chosen"], ctx["fixtures"]
    env = envs.get(args.task_base)
    state_len_env = len(env.sim.get_state().flatten())
    per_demo, cam_info = {}, {}
    samples = {"area": {R.AGENT: [], R.WRIST: []}, "bicubic": {R.AGENT: [], R.WRIST: []}}
    direct = {R.AGENT: [], R.WRIST: []}
    rot = {R.AGENT: [], R.WRIST: []}
    keep = {}
    joint_diffs = []
    for d in chosen:
        g = fr["data"][d]
        parsed = ctx["parsed"][d]
        states = g["states"][:]
        n = len(states)
        check(f"clean_state_length_{d}", states.shape[1] == state_len_env, state_len_stored=int(states.shape[1]), state_len_env=state_len_env)
        env.reset()
        ok, det = R.check_env_fixtures_before_copy(env, parsed, fixtures)         # Q20, first, before any copy
        check(f"q20_env_fixtures_vs_demo_{d}", ok, details=det)
        cam_info[d] = maybe_override_camera(env, parsed, "clean")
        R.copy_fixtures(env, parsed, fixtures)
        ks = frame_indices(n)
        stored = {R.AGENT: g["obs"]["agentview_rgb"], R.WRIST: g["obs"]["eye_in_hand_rgb"]}
        jstored = g["obs"]["joint_states"]
        rec = {"n_states": int(n), "frames": ks}
        for j, k in enumerate(ks):
            obs = R.restore(env, states[k + 1])
            im = R.images(obs)
            if j in (0, len(ks) - 1):
                okf, detf = R.check_fixtures_after(env, parsed, fixtures)
                check(f"fixtures_after_restore_clean_{d}_k{k}", okf, details=detf)
            try:
                joint_diffs.append(float(np.abs(np.asarray(obs["robot0_joint_pos"]) - jstored[k]).max()))
            except Exception as ex:
                REPORT["notes"].append(f"joint_pos check unavailable: {ex!r}")
            for cam in R.CAM_KEYS:
                st = stored[cam][k]
                for m in ("area", "bicubic"):
                    samples[m][cam].append(R.mae(R.shrink(im[cam], m), st))
            keep[(d, k)] = {"stored": {c: np.array(stored[c][k]) for c in R.CAM_KEYS}, "im256": im}
        per_demo[d] = rec
    # Q22: pick the method with the lower mean abs difference (both cameras, all frames); a tie goes to area
    allv = {m: float(np.mean(samples[m][R.AGENT] + samples[m][R.WRIST])) for m in samples}
    chosen_method = "bicubic" if allv["bicubic"] < allv["area"] else "area"
    REPORT["resize"] = {"mae_all": allv, "mae_agentview": {m: float(np.mean(samples[m][R.AGENT])) for m in samples},
                        "mae_wrist": {m: float(np.mean(samples[m][R.WRIST])) for m in samples},
                        "chosen": chosen_method, "n_frames_per_camera": len(samples["area"][R.AGENT]),
                        "rule": "lower mean abs pixel difference vs the stored 128 frames (0-255, all frames, both cameras); tie -> area"}
    print("[resize]", json.dumps(REPORT["resize"]), flush=True)
    # orientation: the re-render must be closer to the stored frame than to the stored frame turned by 180 degrees
    for (d, k), v in keep.items():
        for cam in R.CAM_KEYS:
            sm = R.shrink(v["im256"][cam], chosen_method)
            direct[cam].append(R.mae(sm, v["stored"][cam]))
            rot[cam].append(R.mae(sm, np.rot90(v["stored"][cam], 2)))
    check("orientation_matches_stored", all(np.mean(direct[c]) < np.mean(rot[c]) for c in R.CAM_KEYS),
          mae_direct={c: float(np.mean(direct[c])) for c in R.CAM_KEYS}, mae_vs_stored_rotated_180={c: float(np.mean(rot[c])) for c in R.CAM_KEYS})
    REPORT["clean"] = {"per_demo": per_demo, "camera_pose_vs_saved": cam_info, "max_joint_pos_diff_vs_stored_joint_states": max(joint_diffs) if joint_diffs else None}
    check("restored_arm_joints_equal_stored", bool(joint_diffs) and max(joint_diffs) < 1e-5, gating=False,
          max_diff=max(joint_diffs) if joint_diffs else None, n=len(joint_diffs))
    ctx.update(method=chosen_method, keep=keep, clean_ks={d: per_demo[d]["frames"] for d in chosen})
    make_clean_pngs(ctx, chosen_method, args.out)


def upright(a):
    return np.rot90(a, 2)


def tile_rows(rows, scale=2):
    """rows: list of lists of HxWx3 uint8 images; returns one uint8 image, nearest-neighbour upscaled."""
    h = max(im.shape[0] for r in rows for im in r)
    w = max(im.shape[1] for r in rows for im in r)
    canvas = np.zeros((h * len(rows), w * max(len(r) for r in rows), 3), np.uint8)
    for i, r in enumerate(rows):
        for j, im in enumerate(r):
            canvas[i * h:i * h + im.shape[0], j * w:j * w + im.shape[1]] = im
    return np.kron(canvas, np.ones((scale, scale, 1), np.uint8))


def make_clean_pngs(ctx, method, out):
    for d in ctx["chosen"]:
        ks = ctx["clean_ks"][d]
        k = ks[len(ks) // 2]
        v = ctx["keep"][(d, k)]
        rows = []
        for cam in R.CAM_KEYS:
            a, b = R.shrink(v["im256"][cam], "area"), R.shrink(v["im256"][cam], "bicubic")
            st = v["stored"][cam]
            diff = np.clip(np.abs(R.shrink(v["im256"][cam], method).astype(int) - st.astype(int)) * 4, 0, 255).astype(np.uint8)
            rows.append([upright(st), upright(a), upright(b), upright(diff)])
        Image.fromarray(tile_rows(rows)).save(Path(out) / f"clean-{d}-k{k}.png", optimize=True)
    REPORT["pngs_clean"] = "columns: stored 128, area-shrunk re-render, bicubic-shrunk re-render, |chosen - stored| x4; rows: agentview, wrist; turned upright (180 deg)"


# ---------------------------------------------------------------------------
# camera: the zero-change camera name
# ---------------------------------------------------------------------------

@stage("camera")
def stage_camera(args, ctx, envs):
    fr, fixtures, method = ctx["raw"], ctx["fixtures"], ctx["method"]
    d = ctx["chosen"][0]
    g, parsed = fr["data"][d], ctx["parsed"][d]
    states = g["states"][:]
    name = f"{args.task_base}_view_0_0_100_0_0_initstate_0"
    env0, clean = envs.get(name), envs.get(args.task_base)
    n_env0 = len(env0.sim.get_state().flatten())
    check("camera_zero_change_state_length", n_env0 == states.shape[1], env=n_env0, stored=int(states.shape[1]), name=name)
    env0.reset()
    pose = maybe_override_camera(env0, parsed, "zero_change_name")
    R.copy_fixtures(env0, parsed, fixtures)
    clean.reset()
    pose_clean = maybe_override_camera(clean, parsed, "clean_base_name")
    R.copy_fixtures(clean, parsed, fixtures)
    ks = ctx["clean_ks"][d]
    diffs256, vs_stored = {R.AGENT: [], R.WRIST: []}, {R.AGENT: [], R.WRIST: []}
    for k in ks:
        i0 = R.images(R.restore(env0, states[k + 1]))
        ic = R.images(R.restore(clean, states[k + 1]))
        for cam in R.CAM_KEYS:
            diffs256[cam].append(R.mae(i0[cam], ic[cam]))
            vs_stored[cam].append(R.mae(R.shrink(i0[cam], method), g["obs"]["agentview_rgb" if cam == R.AGENT else "eye_in_hand_rgb"][k]))
    okf, detf = R.check_fixtures_after(env0, parsed, fixtures)
    check("fixtures_after_restore_zero_change_camera", okf, details=detf)
    mean_clean_vs_stored = REPORT["resize"]["mae_agentview"][method]
    envs.release(name)
    REPORT["camera"] = {"name": name, "pose_zero_change_env": pose, "pose_clean_env": pose_clean,
                        "mae256_zero_change_vs_clean_name": {c: float(np.mean(v)) for c, v in diffs256.items()},
                        "mae128_zero_change_vs_stored": {c: float(np.mean(v)) for c, v in vs_stored.items()},
                        "mae128_clean_name_vs_stored_agentview_all_demos": mean_clean_vs_stored,
                        "override_used_zero_change": pose["override_used"], "override_used_clean": pose_clean["override_used"]}
    # frames of the zero-change name match the clean name (same camera pose after rounding) and the stored frames about as well
    check("camera_zero_change_frame_matches_clean", float(np.mean(diffs256[R.AGENT])) < 1.0, gating=True,
          mae256_agentview=float(np.mean(diffs256[R.AGENT])), threshold=1.0,
          note="threshold 1.0 on the 0-255 scale chosen here for 'same picture up to rounding of the camera pose'")
    if pose["override_used"] or pose_clean["override_used"]:
        REPORT["notes"].append("Q2 camera-pose override WAS USED (env pose differed from the demo's saved pose by more than the tolerance)")


# ---------------------------------------------------------------------------
# perturbed frames: one each
# ---------------------------------------------------------------------------

def pick_names(base, classification):
    cat = lambda c: [x["name"] for x in classification["libero_goal"] if x["category"] == c and x["name"].startswith(base + "_")]
    cams = cat("Camera Viewpoints")
    non_zero = [n for n in cams if n.split("_view_")[1].split("_")[0] != "0" or n.split("_view_")[1].split("_")[1] != "0"]
    return {"camera": (non_zero or cams)[0], "light": f"{base}_light_1", "background": f"{base}_table_1",
            "background_wall": f"{base}_tb_13", "noise": cat("Sensor Noise")[0]}


@stage("perturb")
def stage_perturb(args, ctx, envs):
    fr, fixtures, method = ctx["raw"], ctx["fixtures"], ctx["method"]
    names = pick_names(args.task_base, ctx["classification"])
    d = ctx["chosen"][0]
    g, parsed = fr["data"][d], ctx["parsed"][d]
    states = g["states"][:]
    ks = ctx["clean_ks"][d]
    k = ks[len(ks) // 2]
    out, shots, info = {}, {}, {}
    for cond in ["clean", "camera", "light", "background", "background_wall", "noise"]:
        nm = args.task_base if cond == "clean" else names[cond]
        env = envs.get(nm)
        check(f"perturb_state_length_{cond}", len(env.sim.get_state().flatten()) == states.shape[1], env_name=nm,
              env=len(env.sim.get_state().flatten()), stored=int(states.shape[1]))
        env.reset()
        if cond == "clean":
            maybe_override_camera(env, parsed, "perturb_clean")
        R.copy_fixtures(env, parsed, fixtures)
        im = R.images(R.restore(env, states[k + 1]))
        okf, detf = R.check_fixtures_after(env, parsed, fixtures)
        check(f"fixtures_after_restore_{cond}", okf, details=detf, env_name=nm)
        if cond == "noise":
            code = int(nm.split("_noise_")[1])
            im[R.AGENT] = R.apply_noise(im[R.AGENT], code, seed=0)
            info["noise_code"] = code
        shots[cond] = {c: R.shrink(im[c], method) for c in R.CAM_KEYS}
        out[cond] = {"env_name": nm, "scene_xml": R.inner(env)._arena_xml.split("/assets/")[-1]}
        if cond != "clean":
            envs.release(nm)
    stored = {R.AGENT: g["obs"]["agentview_rgb"][k], R.WRIST: g["obs"]["eye_in_hand_rgb"][k]}
    for cond in shots:
        out[cond]["mae_vs_stored_clean_frame"] = {c: R.mae(shots[cond][c], stored[c]) for c in R.CAM_KEYS}
        out[cond]["mae_vs_clean_rerender"] = {c: R.mae(shots[cond][c], shots["clean"][c]) for c in R.CAM_KEYS}
    REPORT["perturb"] = {"demo": d, "k": int(k), "resize": method, "conditions": out, **info}
    rows = [[upright(stored[c])] + [upright(shots[cond][c]) for cond in shots] for c in R.CAM_KEYS]
    Image.fromarray(tile_rows(rows)).save(Path(args.out) / f"perturbed-{d}-k{k}.png", optimize=True)
    REPORT["pngs_perturbed"] = "columns: stored clean 128, clean re-render, camera, light, background (table texture), background (table+wall), noise; rows: agentview, wrist; turned upright (180 deg)"
    # picture-level sanity: the agentview of each perturbed condition should differ from the clean re-render, the wrist of camera
    # and noise should not (they act on the agentview only)
    for cond in ("camera", "noise"):
        REPORT["perturb"]["conditions"][cond]["wrist_unchanged_vs_clean"] = bool(out[cond]["mae_vs_clean_rerender"][R.WRIST] == 0.0)
    check("perturbed_agentview_differs_from_clean", all(out[c]["mae_vs_clean_rerender"][R.AGENT] > 0 for c in
                                                       ("camera", "light", "background", "background_wall", "noise")), gating=False,
          mae={c: out[c]["mae_vs_clean_rerender"][R.AGENT] for c in out})


# ---------------------------------------------------------------------------

class Envs:
    def __init__(self):
        self.d = {}

    def get(self, name):
        if name not in self.d:
            self.d[name] = R.make_env(name)
        return self.d[name]

    def release(self, name):
        e = self.d.pop(name, None)
        if e is not None:
            try:
                e.close()
            except Exception:
                pass

    def close(self):
        for e in list(self.d.values()):
            try:
                e.close()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="turn_on_the_stove_demo.hdf5")
    ap.add_argument("--task", default="turn_on_the_stove_demo")
    ap.add_argument("--task-base", default="turn_on_the_stove")
    ap.add_argument("--feats-cephfs", default="/mnt/features/stacking-encoders/features")
    ap.add_argument("--feats-rbd", default="/persistent/stacking-encoders/features")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    import mujoco, robosuite, torch, PIL
    REPORT["provenance"] = {"SRC_GIT_COMMIT": os.environ.get("SRC_GIT_COMMIT"), "mujoco": mujoco.__version__,
                            "robosuite": robosuite.__version__, "numpy": np.__version__, "pillow": PIL.__version__,
                            "h5py": h5py.__version__, "raw_file": args.raw, "raw_sha256": sha256(args.raw),
                            "diagnostics_sha256": sha256(diagnostics.__file__),
                            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    check("raw_file_is_the_inspected_one", REPORT["provenance"]["raw_sha256"] == RAW_SHA256, gating=False,
          sha256=REPORT["provenance"]["raw_sha256"], expected=RAW_SHA256)
    stage_q23(args)
    envs = Envs()
    ctx = {}
    try:
        with h5py.File(args.raw, "r") as raw:
            ctx["raw"] = raw
            from libero.libero import get_libero_path
            cls = Path(get_libero_path("benchmark_root")) / "benchmark" / "task_classification.json"
            ctx["classification"] = json.loads(cls.read_text())
            stage_scenes(args, envs)
            stage_demos(args, ctx)
            stage_clean(args, ctx, envs)
            stage_camera(args, ctx, envs)
            stage_perturb(args, ctx, envs)
    finally:
        envs.close()
    REPORT["all_checks_passed"] = all(c["pass"] for c in REPORT["checks"].values() if c.get("gating", True))
    REPORT["soft_checks_failed"] = [k for k, c in REPORT["checks"].items() if not c.get("gating", True) and not c["pass"]]
    REPORT["seconds_total"] = round(time.time() - t0, 1)
    (Path(args.out) / "report.json").write_text(json.dumps(REPORT, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print("[done] report written; all gating checks passed:", REPORT["all_checks_passed"], "; stopped_at:", REPORT["stopped_at"], flush=True)


if __name__ == "__main__":
    main()
