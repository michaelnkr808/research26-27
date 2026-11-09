"""Main-run render of the perturbed-frame test (experiments/2026-11-06-perturbed-frames, PLAN Amendment 2026-11-09 and (b)).

Generalises pf_validate.render() (which stays as the record of the pf-val Job) to every split and every condition:

  render             --split train|val|test  --condition clean|camera|light|background|noise  --raw <libero_goal dir>
                     --out <root>  --config <yaml|json>  [--max-demos N]
                     Writes <root>/libero_goal/<stem>.hdf5 in the render input contract (LIBERO layout, original stems and demo names,
                     raw OpenGL orientation, frame j = render of states[j+1] so n-1 frames per demo, each demo's last frame dropped,
                     render 256 then area shrink to 128), and <root>/render-report.json.
                     A test-split render first runs the consistency check against --consistency-ref and stops before any test frame
                     is drawn if it fails.
  check-consistency  re-render demo_30 of each of the ten tasks (clean, all frames, both cameras) and compare the uint8 output
                     byte for byte with the earlier run in --consistency-ref; also compare the recorded software environments.
  compare            the same comparisons between two existing render outputs, without rendering (report-only with --report-only:
                     phase A uses it to compare its new val frames with the pf-val val frames).
  draws              list the perturbation draws (one per test demo and type) from the config seed; reads demo names only.

Rules kept from pf_validate.render: the split is computed from ALL demo names of the raw file and checked (train demo_0..29, val
demo_30..39, test demo_40..49) before any demo content is read; a train or val run cannot open a test demo (pf_render.RawFile); per
demo: state-length and dimension check, reset, Q20 before-copy check, camera-pose override when the demo's saved pose differs (one
shared prepare()), fixture copy, fixture check after restore every 10th frame and the last. The first failure stops the run with a
non-zero exit; nothing is swallowed. Resume is per file: output goes to <stem>.hdf5.partial and is renamed at the end; a file whose
final output exists and whose report entry is complete (same settings) is skipped.

Needs the pod environment of the proxy eval Jobs (LIBERO-Plus @ 4976dc3 on PYTHONPATH, LIBERO_CONFIG_PATH set, MUJOCO_GL=egl).
Nothing here renders on the Mac; the tests (test_pf_render.py) run the logic against a stub simulator.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from importlib import metadata as importlib_metadata
from pathlib import Path

import h5py
import numpy as np

import diagnostics as D
import pf_render as R
from pf_smoke import maybe_override_camera, sha256

SUITE = "libero_goal"
CAMS = D.CAMS                                  # agentview_rgb, eye_in_hand_rgb
if tuple(R.HDF5_KEY[k] for k in R.CAM_KEYS) != tuple(CAMS):
    raise ImportError("pf_render.HDF5_KEY and diagnostics.CAMS disagree")
CONFIG_REQUIRED = {"seed_draw": int, "seed_noise": int, "resize": str}
CONFIG_OPTIONAL = {"suite": str, "raw_sha256": dict, "consistency_demo": str, "n_files": int}
RC_OK, RC_CHECK, RC_CRASH, RC_CONSISTENCY, RC_USAGE = 0, 3, 4, 5, 2


class ConsistencyFailure(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# config and small helpers
# ---------------------------------------------------------------------------

def load_config(path):
    """Config as a dict (yaml or json). Required: seed_draw, seed_noise (ints), resize (area|bicubic). Optional: suite,
    raw_sha256 {stem: hex} (gating: the raw files must hash to these), consistency_demo (default demo_30), n_files (default 10).
    Unknown keys are an error (a typo must not silently become a default)."""
    text = Path(path).read_text()
    if str(path).endswith((".yaml", ".yml")):
        import yaml
        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a mapping")
    for k, t in {**CONFIG_REQUIRED, **CONFIG_OPTIONAL}.items():
        if k in cfg and (not isinstance(cfg[k], t) or isinstance(cfg[k], bool)):
            raise ValueError(f"config {k}: expected {t.__name__}, got {cfg[k]!r}")
    missing = [k for k in CONFIG_REQUIRED if k not in cfg]
    unknown = [k for k in cfg if k not in CONFIG_REQUIRED and k not in CONFIG_OPTIONAL]
    if missing or unknown:
        raise ValueError(f"config: missing {missing}, unknown {unknown}")
    if cfg["resize"] not in ("area", "bicubic"):
        raise ValueError(f"config resize: {cfg['resize']!r}")
    cfg.setdefault("suite", SUITE)
    cfg.setdefault("consistency_demo", "demo_30")
    cfg.setdefault("n_files", 10)
    return cfg


def canon(obj):
    return json.dumps(obj, sort_keys=True, default=str)


def digest(obj):
    return hashlib.sha256(canon(obj).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def base_of(stem):
    return stem[:-5] if stem.endswith("_demo") else stem


def raw_files(raw_dir, n_files=None):
    files = sorted(Path(raw_dir).glob("*.hdf5"))
    if not files:
        raise R.RenderCheckError(f"no .hdf5 files in {raw_dir}")
    if n_files is not None and len(files) != n_files:
        raise R.RenderCheckError(f"expected {n_files} raw files in {raw_dir}, found {len(files)}")
    return files


def load_classification(path=None):
    if path is None:
        from libero.libero import get_libero_path
        path = Path(get_libero_path("benchmark_root")) / "benchmark" / "task_classification.json"
    return json.loads(Path(path).read_text())


def code_hashes():
    here = Path(__file__).resolve().parent
    return {n: sha256(here / n) for n in ("pf_render.py", "pf_rerender.py", "pf_smoke.py", "diagnostics.py") if (here / n).exists()}


# ---------------------------------------------------------------------------
# environment record (software compared strictly, hardware recorded)
# ---------------------------------------------------------------------------

LIBS = {"mujoco": ["mujoco"], "robosuite": ["robosuite"], "numpy": ["numpy"], "Pillow": ["Pillow", "pillow"], "h5py": ["h5py"],
        "scipy": ["scipy"], "scikit-image": ["scikit-image"], "opencv": ["opencv-python", "opencv-python-headless", "opencv-contrib-python"],
        "Wand": ["Wand"], "PyOpenGL": ["PyOpenGL"], "PyYAML": ["PyYAML", "pyyaml"]}


def lib_version(dists):
    for d in dists:
        try:
            return importlib_metadata.version(d)
        except importlib_metadata.PackageNotFoundError:
            continue
    return None


def libero_root():
    try:
        import libero.libero as L
        return Path(L.__file__).resolve().parent
    except Exception:
        return None


def libero_fingerprint(root):
    """sha256 over (relative path, file sha256) of libero/libero/envs/**/*.py and benchmark/task_classification.json. The
    LIBERO-Plus tarball carries no .git, so the commit string is the setup script's; this fingerprint is the check that the code
    on disk is the same."""
    if root is None:
        return None
    files = sorted((root / "envs").rglob("*.py")) + [root / "benchmark" / "task_classification.json"]
    h = hashlib.sha256()
    for f in files:
        if f.exists():
            h.update(f"{f.relative_to(root)}:{sha256(f)}\n".encode())
    return h.hexdigest()


def bundle_info(scripts_dir):
    """env_bundle.sh exports neither the bundle key nor the tar checksum (KEY is a local shell variable; `build` prints the tar's
    sha256, `restore` prints only the path). So: the key is recomputed here exactly as the script does (first 12 hex of sha256 of
    setup_env.sh + setup_sim.sh + env_bundle.sh concatenated) when the three files are in scripts_dir, and the tar sha256 is taken
    from ENV_BUNDLE_SHA256 when the Job sets it. LP_COMMIT (setup_sim.sh) is read from the same directory."""
    out = {"key": None, "tar_sha256": os.environ.get("ENV_BUNDLE_SHA256"), "lp_commit_in_setup_sim": None}
    if scripts_dir:
        d = Path(scripts_dir)
        names = ("setup_env.sh", "setup_sim.sh", "env_bundle.sh")
        if all((d / n).exists() for n in names):
            out["key"] = hashlib.sha256(b"".join((d / n).read_bytes() for n in names)).hexdigest()[:12]
            m = re.search(r"^LP_COMMIT=([0-9a-f]{40})", (d / "setup_sim.sh").read_text(), re.M)
            out["lp_commit_in_setup_sim"] = m.group(1) if m else None
    return out


def nvidia_smi():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True,
                           timeout=30)
        if r.returncode != 0:
            return {"error": f"nvidia-smi exit {r.returncode}: {r.stderr.strip()[:200]}"}
        rows = [[x.strip() for x in line.split(",")] for line in r.stdout.strip().splitlines() if line.strip()]
        return {"gpu_name": rows[0][0], "driver_version": rows[0][1], "n_gpus_visible": len(rows)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:200]}


def gl_strings():
    """GL vendor, renderer and version of the context that is current in this thread (call after a first render). Best effort: if
    PyOpenGL or a current context is missing the reason is recorded instead of the strings."""
    try:
        from OpenGL import GL

        def g(e):
            v = GL.glGetString(e)
            return v.decode() if isinstance(v, bytes) else (None if v is None else str(v))
        out = {"vendor": g(GL.GL_VENDOR), "renderer": g(GL.GL_RENDERER), "version": g(GL.GL_VERSION)}
        if all(v is None for v in out.values()):
            out["error"] = "glGetString returned None (no current GL context)"
        return out
    except Exception as e:
        return {"vendor": None, "renderer": None, "version": None, "error": f"{type(e).__name__}: {e}"[:200]}


def collect_environment(scripts_dir=None):
    """{"software": {...}, "hardware": {...}} for the report. See pf_render.compare_provenance for how the two halves are compared."""
    libs = {k: lib_version(v) for k, v in LIBS.items()}
    libs["python"] = sys.version.split()[0]
    try:
        import wand.version
        libs["ImageMagick"] = str(wand.version.MAGICK_VERSION)
    except Exception:
        libs["ImageMagick"] = None
    b = bundle_info(scripts_dir)
    lp = os.environ.get("LP_COMMIT") or b["lp_commit_in_setup_sim"]
    sw = {"libraries": libs, "libero_plus_commit": lp, "libero_source_fingerprint": libero_fingerprint(libero_root()),
          "bundle_key": b["key"], "bundle_tar_sha256": b["tar_sha256"], "image_digest": os.environ.get("IMAGE_DIGEST"),
          "mujoco_gl": os.environ.get("MUJOCO_GL")}
    hw = {"nvidia_smi": nvidia_smi()}
    return {"software": sw, "hardware": hw,
            "notes": ["libero_plus_commit comes from LP_COMMIT or setup_sim.sh (the tarball has no .git); libero_source_fingerprint is the "
                      "check on the files",
                      "env_bundle.sh exports no bundle checksum: bundle_key is recomputed as the script does it, bundle_tar_sha256 needs "
                      "ENV_BUNDLE_SHA256 from the Job"]}


# ---------------------------------------------------------------------------
# the render itself
# ---------------------------------------------------------------------------

def prepare(env, parsed, fixtures, label, override_camera=True):
    """The one place that gets an env ready for a demo (PLAN, reviewer's notes): reset, Q20 check of the env's own fixtures before
    any copy, the camera-pose override when the env's agentview pose differs from the demo's saved pose (skipped for the camera
    condition, where the pose is the perturbation), then the fixture copy. Returns the camera record."""
    env.reset()
    ok, det = R.check_env_fixtures_before_copy(env, parsed, fixtures)
    if not ok:
        raise R.RenderCheckError(f"{label}: Q20 env fixtures differ from the demo: {json.dumps(det, default=str)[:600]}")
    if override_camera:
        cam = maybe_override_camera(env, parsed, label)
    else:
        cur, sv = R.camera_pose(env), parsed["cameras"]["agentview"]
        cam = {"override_used": False, "override_skipped": "camera condition", "env_pos": cur["pos"].tolist(), "saved_pos": sv["pos"].tolist(),
               "env_quat": cur["quat"].tolist(), "saved_quat": sv["quat"].tolist(), "label": label}
    R.copy_fixtures(env, parsed, fixtures)
    return cam


def render_demo(ctx, stem, base, d, arrays, parsed, fixtures, base_env, drawn):
    """All n-1 frames of one demo under ctx['condition']. Returns ({cam hdf5 key: (n-1,128,128,3) uint8}, info)."""
    cond, cfg = ctx["condition"], ctx["cfg"]
    states, actions = arrays["states"], arrays["actions"]
    n = len(states)
    if len(actions) != n:
        raise R.RenderCheckError(f"{stem}/{d}: {len(actions)} actions vs {n} states")
    own = None
    label = f"{stem}/{d}/{cond}"
    info = {"frames": n - 1}
    try:
        if cond in ("camera", "light", "background"):
            info["episode"] = drawn
            env = own = R.make_env(drawn["name"])
            try:
                info["scene_xml"] = R.inner(env)._arena_xml.split("/assets/")[-1]
            except Exception:
                info["scene_xml"] = None
        else:
            env = base_env
        if cond == "noise":
            info["episode"] = drawn
            info["noise_code"] = code = R.noise_code(drawn["name"])
        env_len = len(env.sim.get_state().flatten())
        if states.ndim != 2 or states.shape[1] != env_len:
            raise R.RenderCheckError(f"{label}: stored state shape {states.shape} vs env state length {env_len}")
        info["camera"] = prepare(env, parsed, fixtures, label, override_camera=(cond != "camera"))
        out = {c: np.zeros((n - 1, R.OUT_SIZE, R.OUT_SIZE, 3), np.uint8) for c in CAMS}
        for j in range(n - 1):
            obs = R.restore(env, states[j + 1])
            if j == 0:
                ctx["on_first_render"]()
            ims = R.images(obs)
            if cond == "noise":
                ims[R.AGENT] = R.apply_noise(ims[R.AGENT], code, R.noise_seed(cfg["seed_noise"], stem, d, j))
            for k, im in ims.items():
                out[R.HDF5_KEY[k]][j] = R.shrink(im, cfg["resize"])
            if j % 10 == 0 or j == n - 2:
                okf, detf = R.check_fixtures_after(env, parsed, fixtures)
                if not okf:
                    raise R.RenderCheckError(f"{label} j={j}: fixtures after restore differ: {json.dumps(detf, default=str)[:400]}")
        return out, info
    finally:
        if own is not None:
            own.close()


def output_complete(final, entry):
    """True if the final file holds exactly the demos of the report entry with the recorded frame counts."""
    try:
        with h5py.File(final, "r") as f:
            if sorted(f["data"].keys()) != sorted(entry["frames"]):
                return False
            return all(f["data"][d]["obs"][c].shape == (n, R.OUT_SIZE, R.OUT_SIZE, 3) for d, n in entry["frames"].items() for c in CAMS)
    except Exception:
        return False


def render_run(raw_dir, out_root, split, condition, cfg, classification, env_info, *, max_demos=None, only_demos=None,
               report_name="render-report.json"):
    """Render every raw file of the directory for one split and one condition; write the report after each file; raise on the first
    failure (after recording it in the report). Returns the report."""
    if condition != "clean" and split != "test":
        raise R.RenderCheckError("perturbed conditions are rendered for the test split only (PLAN)")
    if condition != "clean" and condition not in R.PTYPES:
        raise R.RenderCheckError(f"unknown condition {condition}")
    if condition != "clean" and classification is None:
        raise R.RenderCheckError("a perturbed condition needs the classification file")
    out_root = Path(out_root)
    out_dir = out_root / cfg["suite"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rpath = out_root / report_name
    t0 = time.time()
    files = raw_files(raw_dir, None if only_demos is not None else cfg["n_files"])
    if only_demos is not None:
        files = [f for f in files if f.stem in only_demos]
    chash = code_hashes()
    rep = {"schema": 1, "split": split, "condition": condition, "max_demos": max_demos, "only_demos": only_demos,
           "config": cfg, "config_sha256": digest(cfg), "code_sha256": chash, "environment": env_info, "files": {}, "failures": [],
           "draws": {}, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    rep["resume_notes"] = []
    if rpath.exists():                                  # resume: same run only, same software only
        prev = json.loads(rpath.read_text())
        if (prev.get("split"), prev.get("condition")) != (split, condition):
            raise R.RenderCheckError(f"{out_root} holds a {prev.get('split')}/{prev.get('condition')} run, not {split}/{condition}")
        cmp = R.compare_provenance(prev.get("environment") or {}, env_info)
        if cmp["stop"]:
            raise R.RenderCheckError(f"resume: software differs from the earlier run in this directory: {cmp['stop']}")
        rep["files"] = prev.get("files", {})
        rep["draws"] = prev.get("draws", {})
        rep["resume_notes"] = prev.get("resume_notes", []) + ([f"hardware differs from the earlier part: {cmp['recorded']}"] if cmp["recorded"] else [])
    if cfg.get("raw_sha256"):                           # gating hash check of the raw files (reviewer note)
        for f in files:
            want = cfg["raw_sha256"].get(f.stem)
            if want is None or sha256(f) != want:
                raise R.RenderCheckError(f"raw file {f.name}: sha256 does not match the config")
    rep["checks"] = {"raw_sha256_gating": {"pass": True, "enabled": bool(cfg.get("raw_sha256"))}}

    def on_first_render():
        if "gl" not in env_info["hardware"]:
            env_info["hardware"]["gl"] = gl_strings()
    ctx = {"condition": condition, "cfg": cfg, "on_first_render": on_first_render}
    scene_reports = {}
    try:
        if split == "test":        # phase B: scene files of all ten tasks first (inputs only), so a mismatch stops before any frame is drawn
            for raw_path in files:
                with R.RawFile(raw_path, "test") as rf:
                    nontest = rf.names("train") + rf.names("val")
                    pn = {d: R.parse_model_file(rf.model_file(d)) for d in nontest}
                    fx, _ = R.differing_bodies(list(pn.values()))
                    pt = {d: R.parse_model_file(rf.model_file(d)) for d in rf.names("test")}
                    for p in list(pn.values()) + list(pt.values()):
                        R.reject_other_rotations(p, fx)
                    scene_reports[raw_path.stem] = R.scene_file_check(pn, pt, fx)
                    print(f"[render] {raw_path.stem}: scene files of {len(pt)} test demos match the {len(pn)} train/val demos "
                          f"outside the fixture list {fx}", flush=True)
        for raw_path in files:
            stem = raw_path.stem
            base = base_of(stem)
            with R.RawFile(raw_path, split) as rf:      # split computed from all names and checked here, before any content
                demos = rf.names(split)
                if only_demos is not None:
                    demos = [d for d in demos if d in only_demos[stem]]
                    if demos != list(only_demos[stem]):
                        raise R.RenderCheckError(f"{stem}: requested demos {only_demos[stem]} are not all {split} demos")
                elif max_demos is not None:
                    demos = demos[:max_demos]
                nontest = rf.names("train") + rf.names("val")
                parsed = {d: R.parse_model_file(rf.model_file(d)) for d in nontest}     # test demos are not read in a train/val run
                fixtures, _ = R.differing_bodies([parsed[d] for d in nontest])
                for d in nontest:
                    R.reject_other_rotations(parsed[d], fixtures)       # (in a test run the pre-pass checked fixtures against all 50 demos)
                drawn = {}
                if condition != "clean":
                    drawn = {d: R.draw_episode(classification, base, stem, d, condition, cfg["seed_draw"]) for d in demos}
                    rep["draws"].update({f"{stem}/{d}/{condition}": v for d, v in drawn.items()})
                settings = digest({"split": split, "condition": condition, "demos": demos, "cfg": cfg, "code": chash, "max_demos": max_demos,
                                   "drawn": {d: v["name"] for d, v in drawn.items()}, "only": only_demos})
                final = out_dir / (stem + ".hdf5")
                old = rep["files"].get(stem)
                if old and old.get("complete") and old.get("settings_sha256") == settings and final.exists() and output_complete(final, old):
                    old["skipped_on_resume"] = True
                    print(f"[render] {stem}: complete from an earlier run, skipped", flush=True)
                    continue
                info = rep["files"][stem] = {"complete": False, "base": base, "split": split, "condition": condition, "demos": demos,
                                             "fixtures": fixtures, "n_nontest_compared": len(nontest), "settings_sha256": settings,
                                             "frames": {}, "per_demo": {}, "camera_override_used_in": []}
                if split == "test":                     # phase B: first read of test inputs (model_file text only), logged
                    test_parsed = {d: R.parse_model_file(rf.model_file(d)) for d in rf.names("test")}
                    for d in test_parsed:
                        R.reject_other_rotations(test_parsed[d], fixtures)
                    info["scene_file_check"] = scene_reports[stem]
                t1 = time.time()
                base_env = R.make_env(base, R.RENDER) if condition in ("clean", "noise") else None
                partial = out_dir / (stem + ".hdf5.partial")
                try:
                    with h5py.File(partial, "w") as fo:
                        fo.create_group("data")
                        for d in demos:
                            arrays = rf.demo_arrays(d)
                            imgs, dinfo = render_demo(ctx, stem, base, d, arrays, parsed.get(d) or test_parsed[d], fixtures, base_env, drawn.get(d))
                            dg = fo["data"].create_group(d)
                            ob = dg.create_group("obs")
                            for c in CAMS:
                                ob.create_dataset(c, data=imgs[c])
                            info["frames"][d] = dinfo["frames"]
                            info["per_demo"][d] = dinfo
                            if dinfo.get("camera", {}).get("override_used"):
                                info["camera_override_used_in"].append(d)
                            print(f"[render] {stem} {d} {condition}: {dinfo['frames']} frames", flush=True)
                finally:
                    if base_env is not None:
                        base_env.close()
                os.replace(partial, final)
                info["complete"] = True
                info["seconds"] = round(time.time() - t1, 1)
                touched = [(k, n) for k, n in rf.access_log if rf.split[n] == "test"]
                info["test_demo_reads"] = {"arrays": [n for k, n in touched if k == "arrays"], "model_file": [n for k, n in touched if k == "model_file"]}
                if split != "test" and touched:
                    raise R.SplitAccessError(f"{stem}: a {split} run touched test demos: {touched}")
                write_json(rpath, rep)
        rep["checks"]["render_all_files"] = {"pass": True, "n_files": len(files)}
    except BaseException as e:
        rep["failures"].append(f"{type(e).__name__}: {e}"[:2000])
        rep["checks"]["render_all_files"] = {"pass": False}
        raise
    finally:
        rep["seconds"] = round(time.time() - t0, 1)
        write_json(rpath, rep)
    return rep


# ---------------------------------------------------------------------------
# consistency check (PLAN Amendment 2026-11-09 (b))
# ---------------------------------------------------------------------------

def load_ref_environment(ref_root):
    p = Path(ref_root) / "render-report.json"
    if not p.exists():
        raise ConsistencyFailure(f"no render-report.json in the reference {ref_root}")
    return json.loads(p.read_text()).get("environment") or {}


def pixel_comparison(ref_root, new_root, suite, stems, demos_by_stem=None):
    out = {}
    for stem in stems:
        a, b = Path(ref_root) / suite / (stem + ".hdf5"), Path(new_root) / suite / (stem + ".hdf5")
        if not a.exists() or not b.exists():
            out[stem] = {"identical": False, "missing_file": [str(p) for p in (a, b) if not p.exists()]}
            continue
        out[stem] = R.compare_frames(a, b, None if demos_by_stem is None else demos_by_stem[stem])
    return out


def run_consistency(raw_dir, ref_root, out_root, cfg, env_info, *, render=render_run):
    """Software comparison first (stops before anything is rendered), then re-render demo_30 of each task (clean, all frames, both
    cameras) and compare the uint8 output byte for byte with the reference run. Writes consistency-report.json in out_root and
    raises ConsistencyFailure on any software difference or any differing value."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rep = {"reference": str(ref_root), "demo": cfg["consistency_demo"], "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "environment": env_info}
    path = out_root / "consistency-report.json"
    try:
        ref_env = load_ref_environment(ref_root)
        rep["provenance_comparison"] = cmp = R.compare_provenance(ref_env, env_info)
        if cmp["stop"]:
            raise ConsistencyFailure("software environment differs from the reference run: " + "; ".join(cmp["stop"]))
        stems = [f.stem for f in raw_files(raw_dir, cfg["n_files"])]
        only = {s: [cfg["consistency_demo"]] for s in stems}
        sub = out_root / ("consistency-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
        render(raw_dir, sub, "val", "clean", cfg, None, env_info, only_demos=only)
        rep["pixels"] = pix = pixel_comparison(ref_root, sub, cfg["suite"], stems, only)
        rep["provenance_comparison"] = R.compare_provenance(ref_env, env_info)     # again: the GL strings exist now (recorded only)
        rep["hardware_differences_recorded"] = rep["provenance_comparison"]["recorded"]
        bad = [s for s, r in pix.items() if not r["identical"]]
        rep["identical"] = not bad
        if bad:
            raise ConsistencyFailure(f"pixels differ from the reference in {len(bad)} of {len(stems)} files: {bad}")
        return rep
    except BaseException as e:
        rep.setdefault("identical", False)
        rep["failure"] = f"{type(e).__name__}: {e}"[:2000]
        raise
    finally:
        rep["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(path, rep)


def compare_dirs(ref_dir, new_dir, ref_report=None, new_report=None, suite=SUITE):
    """Render-free comparison of two render outputs: environment records (software difference listed under `stop`, hardware under
    `recorded`) and the pixels of every demo present in either file (a demo in one file only counts as a difference), per task file."""
    rr = Path(ref_report or Path(ref_dir) / "render-report.json")
    nr = Path(new_report or Path(new_dir) / "render-report.json")
    ref_env = json.loads(rr.read_text()).get("environment") or {} if rr.exists() else {}
    new_env = json.loads(nr.read_text()).get("environment") or {} if nr.exists() else {}
    stems = sorted(p.stem for p in (Path(ref_dir) / suite).glob("*.hdf5"))
    stems_new = sorted(p.stem for p in (Path(new_dir) / suite).glob("*.hdf5"))
    out = {"reference": str(ref_dir), "new": str(new_dir), "reference_report_found": rr.exists(), "new_report_found": nr.exists(),
           "provenance_comparison": R.compare_provenance(ref_env, new_env), "files_only_in_reference": sorted(set(stems) - set(stems_new)),
           "files_only_in_new": sorted(set(stems_new) - set(stems))}
    out["pixels"] = pixel_comparison(ref_dir, new_dir, suite, sorted(set(stems) & set(stems_new)))
    out["identical"] = bool(out["pixels"]) and all(r["identical"] for r in out["pixels"].values()) and not out["files_only_in_reference"] \
        and not out["files_only_in_new"]
    return out


# ---------------------------------------------------------------------------
# draws
# ---------------------------------------------------------------------------

def list_draws(raw_dir, cfg, classification, ptypes=R.PTYPES):
    files = []
    for p in raw_files(raw_dir, cfg["n_files"]):
        with R.RawFile(p, "test") as rf:                # names only: no demo content is read here
            files.append((p.stem, base_of(p.stem), rf.names("test")))
    return {"seed_draw": cfg["seed_draw"], "draws": R.draw_all(classification, files, cfg["seed_draw"], ptypes)}


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_render(args):
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    classification = load_classification(args.classification) if args.condition != "clean" else None
    env_info = collect_environment(args.env_scripts_dir)
    if args.split == "test":
        if not args.consistency_ref:
            print("a test-split render needs --consistency-ref (the phase A val render); no frame was drawn", flush=True)
            return RC_USAGE
        try:
            run_consistency(args.raw, args.consistency_ref, out / "consistency", cfg, env_info)
        except ConsistencyFailure as e:
            print(f"[consistency] FAILED, no test frame drawn: {e}", flush=True)
            return RC_CONSISTENCY
    elif args.consistency_ref:
        print("--consistency-ref only applies to a test-split render", flush=True)
        return RC_USAGE
    try:
        render_run(args.raw, out, args.split, args.condition, cfg, classification, env_info, max_demos=args.max_demos)
    except R.RenderCheckError as e:
        print(f"[render] check failed, stopping: {e}", flush=True)
        return RC_CHECK
    print("[render] done", flush=True)
    return RC_OK


def cmd_check(args):
    cfg = load_config(args.config)
    env_info = collect_environment(args.env_scripts_dir)
    try:
        run_consistency(args.raw, args.consistency_ref, args.out, cfg, env_info)
    except ConsistencyFailure as e:
        print(f"[consistency] FAILED: {e}", flush=True)
        return RC_CONSISTENCY
    print("[consistency] every value identical", flush=True)
    return RC_OK


def cmd_compare(args):
    out = compare_dirs(args.ref_dir, args.new_dir, args.ref_report, args.new_report)
    out["mode"] = "report-only" if args.report_only else "strict"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, out)
    n_bad = sum(1 for r in out["pixels"].values() if not r["identical"])
    print(f"[compare] files compared {len(out['pixels'])}, with differences {n_bad}, software differences "
          f"{len(out['provenance_comparison']['stop'])}, hardware differences {len(out['provenance_comparison']['recorded'])}", flush=True)
    if args.report_only:
        return RC_OK
    return RC_OK if out["identical"] and not out["provenance_comparison"]["stop"] else RC_CONSISTENCY


def cmd_draws(args):
    cfg = load_config(args.config)
    out = list_draws(args.raw, cfg, load_classification(args.classification))
    write_json(args.out, out)
    print(f"[draws] wrote {len(out['draws'])} draws to {args.out}", flush=True)
    return RC_OK


def build_parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, out=True):
        p.add_argument("--raw", required=True, help="dir with the 10 libero_goal HDF5 files")
        p.add_argument("--config", required=True)
        p.add_argument("--classification", default=None, help="task_classification.json (default: LIBERO-Plus's own, via LIBERO_CONFIG_PATH)")
        p.add_argument("--env-scripts-dir", default="/src", help="dir with setup_env.sh, setup_sim.sh, env_bundle.sh (for the bundle key)")
        if out:
            p.add_argument("--out", required=True)
    r = sub.add_parser("render")
    common(r)
    r.add_argument("--split", required=True, choices=["train", "val", "test"])
    r.add_argument("--condition", required=True, choices=["clean", "camera", "light", "background", "noise"])
    r.add_argument("--max-demos", type=int, default=None, help="render only the first N demos of the split in each file (pilot)")
    r.add_argument("--consistency-ref", default=None, help="test split: the phase A val render (a render output root) to compare with first")
    c = sub.add_parser("check-consistency")
    common(c)
    c.add_argument("--consistency-ref", required=True)
    m = sub.add_parser("compare")
    m.add_argument("--ref-dir", required=True)
    m.add_argument("--new-dir", required=True)
    m.add_argument("--ref-report", default=None)
    m.add_argument("--new-report", default=None)
    m.add_argument("--out", required=True)
    m.add_argument("--report-only", action="store_true")
    d = sub.add_parser("draws")
    common(d)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return {"render": cmd_render, "check-consistency": cmd_check, "compare": cmd_compare, "draws": cmd_draws}[args.cmd](args)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        out = Path(getattr(args, "out", "."))
        if args.cmd != "compare":
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{args.cmd}-crash.txt").write_text(tb)
        return RC_CRASH


if __name__ == "__main__":
    raise SystemExit(main())
