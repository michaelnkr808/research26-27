"""Step A diagnosis for the perturbed-frame smoke test: why does the stored bowl look speckled while the re-render looks smooth?
(experiments/2026-11-06-perturbed-frames, LOG 2026-11-08; user decision "Find the cause, then check".)

Same 3 train/val demos and 6 frames each as pf_smoke.py. For each stored moment k it renders
  - the env at native 128 x 128 (no shrink) and the env at 256 x 256 shrunk by area averaging (the approved pipeline),
  - states[k], states[k+1], states[k+2] (off-by-one check; states[k+1] must give the lowest difference to stored image k),
and compares with the stored 128 frame on the whole frame, per camera, and on a crop around the bowl (the projected bowl body
position, and a hand-placed box read off the smoke-test PNG). It also records the MuJoCo version and the bowl's mesh / material /
texture files as the env loads them, next to the same elements in the demo's own model_file. Never a test demo.
"""

import argparse
import hashlib
import json
import os
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

import diagnostics
import pf_render as R
from pf_smoke import frame_indices, maybe_override_camera, sha256, tile_rows, upright

HALF = 14                    # crop half-size in pixels at 128 x 128 (28 x 28 crop)
HAND_CENTER_UPRIGHT = (62, 72)   # (x, y) of the bowl in the upright 128 frame, read off the smoke-test PNG clean-demo_0-k47 (hand)
SCALE = 8                    # crop upscaling in the PNG (nearest neighbour)


def project(sim, cam_name, point, size):
    """Upright pixel (x right, y down) of a world point for a MuJoCo camera, at size x size."""
    m, d = sim.model, sim.data
    cid = m.camera_name2id(cam_name)
    R_ = np.array(d.cam_xmat[cid]).reshape(3, 3)
    pc = R_.T @ (np.asarray(point, float) - np.array(d.cam_xpos[cid]))
    f = (size / 2) / np.tan(np.radians(float(m.cam_fovy[cid])) / 2)
    return float(size / 2 + f * pc[0] / -pc[2]), float(size / 2 - f * pc[1] / -pc[2])


def crop_box(cx, cy, size=128, half=HALF):
    x0 = int(round(min(max(cx - half, 0), size - 2 * half)))
    y0 = int(round(min(max(cy - half, 0), size - 2 * half)))
    return x0, y0, x0 + 2 * half, y0 + 2 * half


def crop(img_upright, box):
    x0, y0, x1, y1 = box
    return img_upright[y0:y1, x0:x1]


def hf_energy(c):
    """Mean |Laplacian| of the grey crop: high when a crop is speckled, low when it is smooth."""
    g = c.astype(np.float32).mean(axis=2)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(np.abs(lap).mean())


def bowl_assets(xml_text, body_name, base_note):
    """Mesh / material / texture elements used by the geoms under the named body, with file existence, size and sha256 prefix."""
    root = ET.fromstring(xml_text)
    asset = root.find("asset")
    by = lambda tag: {e.get("name"): dict(e.attrib) for e in (asset.iter(tag) if asset is not None else []) if e.get("name")}
    meshes, mats, texs = by("mesh"), by("material"), by("texture")
    body = next((b for b in root.iter("body") if b.get("name") == body_name), None)
    if body is None:
        return {"error": f"no body {body_name}"}
    out = {"note": base_note, "geoms": []}

    def fileinfo(p):
        if not p:
            return None
        ex = os.path.isfile(p)
        return {"path": p, "exists": ex, "bytes": os.path.getsize(p) if ex else None,
                "sha256_16": hashlib.sha256(open(p, "rb").read()).hexdigest()[:16] if ex else None}

    for g in body.iter("geom"):
        e = {"geom": dict(g.attrib)}
        if g.get("mesh") in meshes:
            e["mesh"] = meshes[g.get("mesh")]
            e["mesh_file"] = fileinfo(meshes[g.get("mesh")].get("file"))
        mname = g.get("material")
        if mname in mats:
            e["material"] = mats[mname]
            t = mats[mname].get("texture")
            if t in texs:
                e["texture"] = texs[t]
                e["texture_file"] = fileinfo(texs[t].get("file"))
        out["geoms"].append(e)
    return out


def asset_attr_diff(env_xml, demo_xml, cap=60):
    """mesh / texture / material elements present in both XMLs whose attributes differ (file compared by basename only)."""
    def load(x):
        root = ET.fromstring(x)
        a = root.find("asset")
        d = {}
        for tag in ("mesh", "texture", "material"):
            for e in (a.iter(tag) if a is not None else []):
                if e.get("name"):
                    at = dict(e.attrib)
                    if "file" in at:
                        at["file"] = os.path.basename(at["file"])
                    d[(tag, e.get("name"))] = at
        return d
    a, b = load(env_xml), load(demo_xml)
    diffs = []
    for k in sorted(set(a) & set(b)):
        if a[k] != b[k]:
            diffs.append({"element": f"{k[0]}[{k[1]}]", "env": a[k], "demo_model_file": b[k]})
    return {"n_common": len(set(a) & set(b)), "n_only_env": len(set(a) - set(b)), "n_only_demo": len(set(b) - set(a)),
            "n_differing": len(diffs), "differing_head": diffs[:cap],
            "only_env_head": sorted(f"{t}[{n}]" for t, n in set(a) - set(b))[:cap],
            "only_demo_head": sorted(f"{t}[{n}]" for t, n in set(b) - set(a))[:cap]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--task-base", default="turn_on_the_stove")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import mujoco, robosuite
    rep = {"provenance": {"SRC_GIT_COMMIT": os.environ.get("SRC_GIT_COMMIT"), "mujoco": mujoco.__version__,
                          "mj_versionString": mujoco.mj_versionString(), "robosuite": robosuite.__version__,
                          "numpy": np.__version__, "raw_sha256": sha256(args.raw),
                          "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, "checks": {}}
    fr = h5py.File(args.raw, "r")
    demos = sorted(fr["data"].keys(), key=lambda k: int(k.split("_")[-1]))
    split = diagnostics.demo_split(demos)
    chosen = [[d for d in demos if split[d] == "train"][0], [d for d in demos if split[d] == "train"][-1],
              [d for d in demos if split[d] == "val"][0]]
    assert all(split[d] != "test" for d in chosen), chosen
    rep["checks"]["none_is_test"] = {"pass": True, "chosen": chosen, "split": {d: split[d] for d in chosen}}
    nontest = [d for d in demos if split[d] != "test"]
    parsed = {d: R.parse_model_file(fr["data"][d].attrs["model_file"]) for d in nontest}
    fixtures, _ = R.differing_bodies([parsed[d] for d in nontest])
    rep["fixtures"] = fixtures

    e128, e256 = R.make_env(args.task_base, 128), R.make_env(args.task_base, 256)
    envxml = e128.sim.model.get_xml()
    bnames = [b.get("name") for b in ET.fromstring(envxml).iter("body") if b.get("name") and "bowl" in b.get("name")]
    bowl_body = next(n for n in bnames if n.endswith("_main"))
    rep["bowl_body"] = bowl_body
    demo_xml = fr["data"][chosen[0]].attrs["model_file"]
    demo_xml = demo_xml.decode() if isinstance(demo_xml, bytes) else demo_xml
    rep["bowl_assets_env"] = bowl_assets(envxml, bowl_body, "as the env loads them (LIBERO-Plus assets in the pod)")
    rep["bowl_assets_demo_model_file"] = bowl_assets(demo_xml, bowl_body, "as written in the demo's model_file (paths on the dataset author's machine; hashes only if the path exists here)")
    rep["asset_attr_diff_env_vs_demo_model_file"] = asset_attr_diff(envxml, demo_xml)
    print("[diag] assets read", flush=True)

    keys = ("native128", "area256")
    mae = {k: {R.AGENT: [], R.WRIST: []} for k in keys}
    crop_mae = {k: [] for k in keys}                 # bowl crop (projected box), agentview
    crop_mae_hand = {k: [] for k in keys}            # bowl crop (hand box)
    hf = {"stored": [], "native128": [], "area256": []}
    off = {k: {o: {R.AGENT: [], R.WRIST: []} for o in (0, 1, 2)} for k in keys}
    off_lowest = []
    frames_rec, pics = [], []
    for d in chosen:
        g = fr["data"][d]
        p = parsed[d]
        states = g["states"][:]
        n = len(states)
        for env in (e128, e256):
            env.reset()
            maybe_override_camera(env, p, "diag")
            R.copy_fixtures(env, p, fixtures)
        stored = {R.AGENT: g["obs"]["agentview_rgb"], R.WRIST: g["obs"]["eye_in_hand_rgb"]}
        for k in frame_indices(n):
            st = {c: np.array(stored[c][k]) for c in R.CAM_KEYS}

            def render(state):
                a = R.images(R.restore(e128, state))
                b = {c: R.shrink(v, "area") for c, v in R.images(R.restore(e256, state)).items()}
                return {"native128": a, "area256": b}

            r1 = render(states[k + 1])
            for key in keys:
                for c in R.CAM_KEYS:
                    mae[key][c].append(R.mae(r1[key][c], st[c]))
            bpos = np.array(e128.sim.data.body_xpos[e128.sim.model.body_name2id(bowl_body)])
            cx, cy = project(e128.sim, "agentview", bpos, 128)
            box = crop_box(cx, cy)
            hbox = crop_box(*HAND_CENTER_UPRIGHT)
            su = upright(st[R.AGENT])
            cr = {"stored": crop(su, box), **{key: crop(upright(r1[key][R.AGENT]), box) for key in keys}}
            crh = {"stored": crop(su, hbox), **{key: crop(upright(r1[key][R.AGENT]), hbox) for key in keys}}
            for key in keys:
                crop_mae[key].append(R.mae(cr[key], cr["stored"]))
                crop_mae_hand[key].append(R.mae(crh[key], crh["stored"]))
            for key in hf:
                hf[key].append(hf_energy(cr[key]))
            rec = {"demo": d, "k": int(k), "bowl_world_pos": bpos.tolist(), "bowl_projected_xy_upright128": [cx, cy], "box_proj": list(box),
                   "box_hand": list(hbox)}
            if k + 2 <= n - 1:                       # off-by-one check: states[k], [k+1], [k+2] against stored image k
                per = {}
                for o in (0, 1, 2):
                    ro = r1 if o == 1 else render(states[k + o])
                    per[o] = {}
                    for key in keys:
                        per[o][key] = {c: R.mae(ro[key][c], st[c]) for c in R.CAM_KEYS}
                        for c in R.CAM_KEYS:
                            off[key][o][c].append(per[o][key][c])
                tot = {o: float(np.mean([per[o]["native128"][c] for c in R.CAM_KEYS])) for o in (0, 1, 2)}
                rec["offset_mae_native128_mean_of_cams"] = tot
                rec["offset_lowest"] = int(min(tot, key=tot.get))
                off_lowest.append(rec["offset_lowest"])
            frames_rec.append(rec)
            pics.append((d, k, cr))
    def agg(v):
        return float(np.mean(v))
    rep["frames"] = frames_rec
    rep["mae_vs_stored"] = {key: {"overall": agg(mae[key][R.AGENT] + mae[key][R.WRIST]), "agentview": agg(mae[key][R.AGENT]),
                                  "wrist": agg(mae[key][R.WRIST]), "bowl_crop_projected_box": agg(crop_mae[key]),
                                  "bowl_crop_hand_box": agg(crop_mae_hand[key])} for key in keys}
    rep["n_moments"] = len(frames_rec)
    rep["bowl_crop_high_frequency_energy_mean_abs_laplacian"] = {k: agg(v) for k, v in hf.items()}
    rep["off_by_one"] = {"n_moments_checked": len(off_lowest), "lowest_offset_per_moment": off_lowest,
                         "mae_by_offset": {key: {str(o): {"overall": agg(off[key][o][R.AGENT] + off[key][o][R.WRIST]),
                                                         "agentview": agg(off[key][o][R.AGENT]), "wrist": agg(off[key][o][R.WRIST])}
                                                for o in (0, 1, 2)} for key in keys}}
    m1 = {key: {o: rep["off_by_one"]["mae_by_offset"][key][str(o)]["overall"] for o in (0, 1, 2)} for key in keys}
    rep["checks"]["offset_k_plus_1_is_lowest"] = {"pass": all(min(m1[key], key=m1[key].get) == 1 for key in keys) and all(o == 1 for o in off_lowest),
                                                  "overall_mae": m1}
    # PNGs: bowl crops, stored | native128 | 256 -> area (rows: moments), and a full-frame overview of one moment
    sel = pics[1::4]
    rows = [[np.ascontiguousarray(c[key]) for key in ("stored", "native128", "area256")] for _, _, c in sel]
    img = tile_rows(rows, scale=SCALE)
    Image.fromarray(img).save(out / "bowl-crops.png", optimize=True)
    rep["bowl_crops_png"] = {"columns": "stored | native 128 | 256 -> area", "rows": [(d, int(k)) for d, k, _ in sel],
                             "crop": f"{2 * HALF} x {2 * HALF} px at 128, upright (turned 180), shown x{SCALE}; box = bowl body projected through the agentview camera"}
    full = []
    for d in chosen[:1]:
        for k in frame_indices(len(fr["data"][d]["states"]))[2:3]:
            states = fr["data"][d]["states"][:]
            a = R.images(R.restore(e128, states[k + 1]))[R.AGENT]
            b = R.shrink(R.images(R.restore(e256, states[k + 1]))[R.AGENT], "area")
            full.append([upright(np.array(fr["data"][d]["obs"]["agentview_rgb"][k])), upright(a), upright(b)])
    Image.fromarray(tile_rows(full, scale=3)).save(out / "overview-agentview.png", optimize=True)
    rep["overview_png"] = "stored | native 128 | 256 -> area, agentview, upright, x3"
    for e in (e128, e256):
        e.close()
    fr.close()
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print("[diag] SUMMARY", json.dumps({"mae_vs_stored": rep["mae_vs_stored"], "hf_energy": rep["bowl_crop_high_frequency_energy_mean_abs_laplacian"],
                                       "off_by_one": rep["checks"]["offset_k_plus_1_is_lowest"], "assets_diff": {k: v for k, v in
                                       rep["asset_attr_diff_env_vs_demo_model_file"].items() if k.startswith("n_")}}), flush=True)


if __name__ == "__main__":
    main()
