"""Render helpers for the perturbed-frame test (experiments/2026-11-06-perturbed-frames, PLAN Handoff steps 1-2).

Re-render a stored libero_goal demo moment inside a LIBERO-Plus env (clean, camera, light, background or noise):
  reset  ->  (Q20 check)  ->  copy the demo's fixture body pos/quat from its `model_file`  ->  set_init_state(states[k+1])
  ->  read the 256 x 256 images  ->  (noise at 256)  ->  shrink to 128 x 128.
Image k of a stored demo pairs with stored state k+1; the last image has no state (PLAN Q14, Feasibility 7b).

Needs the pod environment of the proxy eval Jobs (LIBERO-Plus @ 4976dc3 on PYTHONPATH, LIBERO_CONFIG_PATH set, MUJOCO_GL=egl).
The pure functions (XML parsing, resize, fixture comparison) import nothing from the simulator, so they can be tested anywhere.

Added 2026-11-09 for the main render (src/pf_rerender.py; PLAN Amendment 2026-11-09 and (b)): explicit exceptions instead of asserts
(RenderCheckError, SceneMismatch, SplitAccessError), per-frame noise seeds with the global RNG restored, the draw of perturbation
episodes (draw_episode, draw_all), RawFile (split computed from all names and checked, test demos unreadable in train/val runs),
scene_file_check (test model_file text against train/val), and the comparisons behind the render-consistency check
(compare_provenance, compare_frames). The older callers (pf_smoke, pf_validate, pf_diag) use the helpers above unchanged.
"""

import hashlib
import json
import re
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

RENDER = 256          # LIBERO_ENV_RESOLUTION, as src/rollout.py (PLAN Q15)
OUT_SIZE = 128        # stored frames are 128 x 128
AGENT, WRIST = "agentview_image", "robot0_eye_in_hand_image"
CAM_KEYS = (AGENT, WRIST)
HDF5_KEY = {AGENT: "agentview_rgb", WRIST: "eye_in_hand_rgb"}   # env obs key -> LIBERO HDF5 key (one place; = diagnostics.CAMS order)
IDENT_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
OTHER_ROT_ATTRS = ("euler", "axisangle", "xyaxes", "zaxis")   # rotation forms that copy_fixtures (pos and quat only) would miss
POS_TOL = 1e-5        # model_file holds 6 significant digits (inference plus recall; PLAN Feasibility 6b); values ~1 m


class RenderCheckError(RuntimeError):
    """A per-demo or per-file check of the render pipeline failed. Explicit exceptions, not asserts: `python -O` strips asserts.
    (RuntimeError subclass, so the pf_validate.render handler that catches RuntimeError treats it as before.)"""


class SceneMismatch(RenderCheckError):
    """Scene files (model_file strings) differ in a way the fixture list does not cover."""


class SplitAccessError(RenderCheckError):
    """Something tried to open a demo that the current run's split does not allow (the test-set guard)."""


# ---------------------------------------------------------------------------
# model_file (scene XML stored in the HDF5) -> bodies and cameras
# ---------------------------------------------------------------------------

def _vec(s, n, default=None):
    if s is None:
        return default
    v = np.array([float(x) for x in s.split()])
    if len(v) != n:
        raise ValueError(f"expected {n} numbers, got {s!r}")
    return v


def parse_model_file(xml_text):
    """{'bodies': {name: {'pos', 'quat', 'has_quat'}}, 'cameras': {name: {'pos', 'quat'}}} from a MuJoCo XML string.

    Bodies without a name are skipped. A body without a quat attribute gets the identity quaternion (1 0 0 0), the MuJoCo
    default (recalled; PLAN Q19). Duplicate names would make a dict ambiguous, so they raise.
    """
    if isinstance(xml_text, bytes):
        xml_text = xml_text.decode()
    root = ET.fromstring(xml_text)
    bodies, cams = {}, {}
    for b in root.iter("body"):
        name = b.get("name")
        if name is None:
            continue
        if name in bodies:
            raise ValueError(f"duplicate body name {name}")
        bodies[name] = {"pos": _vec(b.get("pos"), 3, np.zeros(3)), "quat": _vec(b.get("quat"), 4, IDENT_QUAT.copy()),
                        "has_quat": b.get("quat") is not None,
                        "other_rot": sorted(a for a in OTHER_ROT_ATTRS if b.get(a) is not None)}
    for c in root.iter("camera"):
        name = c.get("name")
        if name is not None:
            cams[name] = {"pos": _vec(c.get("pos"), 3), "quat": _vec(c.get("quat"), 4)}
    return {"bodies": bodies, "cameras": cams}


def differing_bodies(parsed_list):
    """Names of bodies whose pos or quat differs between the parsed model files (the fixture list, PLAN Q19), and a check
    that every parsed file has the same set of body names. Returns (sorted names, per-name info for the report)."""
    names0 = set(parsed_list[0]["bodies"])
    for i, p in enumerate(parsed_list[1:], 1):
        if set(p["bodies"]) != names0:
            only0, onlyi = sorted(names0 - set(p["bodies"])), sorted(set(p["bodies"]) - names0)
            raise SceneMismatch(f"model files differ in their body names (file 0 only: {only0[:5]}, file {i} only: {onlyi[:5]})")
    diff, info = [], {}
    for n in sorted(names0):
        pos = np.stack([p["bodies"][n]["pos"] for p in parsed_list])
        quat = np.stack([p["bodies"][n]["quat"] for p in parsed_list])
        pos_spread, quat_spread = float(np.ptp(pos, axis=0).max()), float(np.ptp(quat, axis=0).max())
        if pos_spread > 0 or quat_spread > 0:
            diff.append(n)
            info[n] = {"pos_spread_max_m": pos_spread, "quat_spread_max": quat_spread,
                       "pos_min": pos.min(axis=0).tolist(), "pos_max": pos.max(axis=0).tolist()}
    return diff, info


def quat_close(a, b, atol=POS_TOL):
    """Equal as rotations: q and -q are the same rotation. Returns (ok, max abs difference of the better sign)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = min(np.abs(a - b).max(), np.abs(a + b).max())
    return bool(d <= atol), float(d)


# ---------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------

def shrink(img, method):
    """256 x 256 x 3 uint8 -> 128 x 128 x 3 uint8. 'area' = mean of each 2 x 2 block (what area averaging is for an exact
    factor of 2), rounded; 'bicubic' = PIL BICUBIC (Pillow antialiases when it shrinks)."""
    if not (img.shape == (RENDER, RENDER, 3) and img.dtype == np.uint8):
        raise ValueError(f"expected a {RENDER} x {RENDER} x 3 uint8 image, got {img.shape} {img.dtype}")
    if method == "area":
        s = OUT_SIZE
        return np.rint(img.reshape(s, 2, s, 2, 3).astype(np.float32).mean(axis=(1, 3))).astype(np.uint8)
    if method == "bicubic":
        return np.asarray(Image.fromarray(img).resize((OUT_SIZE, OUT_SIZE), Image.BICUBIC))
    raise ValueError(method)


def mae(a, b):
    """Mean absolute pixel difference on the 0-255 scale."""
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


# ---------------------------------------------------------------------------
# simulator side
# ---------------------------------------------------------------------------

def bddl_path(name, suite="libero_goal"):
    from pathlib import Path
    from libero.libero import get_libero_path
    return str(Path(get_libero_path("bddl_files")) / suite / f"{name}.bddl")


def make_env(name, size=RENDER):
    """LIBERO-Plus offscreen env for a benchmark episode name (e.g. turn_on_the_stove, turn_on_the_stove_light_1,
    turn_on_the_stove_view_0_0_100_0_0_initstate_0). Both cameras render at size x size."""
    from libero.libero.envs import OffScreenRenderEnv
    return OffScreenRenderEnv(bddl_file_name=bddl_path(name), camera_heights=size, camera_widths=size)


def inner(env):
    return env.env


def fixture_root_bodies(env):
    """{fixture object name: root body name} of the env (e.g. wooden_cabinet_1 -> wooden_cabinet_1_main)."""
    return {n: o.root_body for n, o in inner(env).fixtures_dict.items()}


def read_bodies(env, body_names):
    """Current body_pos and body_quat in the env's sim for the named bodies."""
    m = env.sim.model
    out = {}
    for n in body_names:
        i = m.body_name2id(n)
        out[n] = {"pos": np.array(m.body_pos[i], float), "quat": np.array(m.body_quat[i], float)}
    return out


def bddl_region_xy(env, fixture_name):
    """(x_range, y_range) in world metres allowed by the bddl region of an `On(fixture, region)` init predicate: the bddl
    ranges are offsets from the target table, so the table offset (workspace_offset) is added. One range expected."""
    e = inner(env)
    regs = e.parsed_problem["regions"]
    st = [s for s in e.parsed_problem["initial_state"] if s[0].lower() == "on" and s[1] == fixture_name]
    if len(st) != 1:
        raise RenderCheckError(f"expected one On({fixture_name}, region) init predicate, got {st}")
    ranges = regs[st[0][2]]["ranges"]
    if len(ranges) != 1:
        raise RenderCheckError(f"expected one range for region {st[0][2]}, got {ranges}")
    x0, y0, x1, y1 = ranges[0]
    off = np.array(e.workspace_offset, float)
    return (x0 + off[0], x1 + off[0]), (y0 + off[1], y1 + off[1]), st[0][2], list(map(float, ranges[0]))


def check_env_fixtures_before_copy(env, parsed_demo, fixtures):
    """Q20: after reset, before any copy. Env fixture z and quat must equal the demo's model_file values (identity where
    absent) and x, y must lie inside the bddl region. Returns (ok, details)."""
    rb = fixture_root_bodies(env)
    inv = {v: k for k, v in rb.items()}
    missing = [f for f in fixtures if f not in inv]
    if missing:
        return False, {"error": f"differing bodies are not fixture root bodies of the env: {missing}", "env_roots": rb}
    cur = read_bodies(env, fixtures)
    ok_all, det = True, {}
    for f in fixtures:
        want = parsed_demo["bodies"][f]
        (x0, x1), (y0, y1), region, rng = bddl_region_xy(env, inv[f])
        p, q = cur[f]["pos"], cur[f]["quat"]
        z_ok = abs(p[2] - want["pos"][2]) <= POS_TOL
        q_ok, q_d = quat_close(q, want["quat"])
        in_xy = (x0 - POS_TOL <= p[0] <= x1 + POS_TOL) and (y0 - POS_TOL <= p[1] <= y1 + POS_TOL)
        ok = bool(z_ok and q_ok and in_xy)
        ok_all &= ok
        det[f] = {"ok": ok, "z_ok": bool(z_ok), "quat_ok": q_ok, "xy_in_region": bool(in_xy),
                  "env_pos": p.tolist(), "demo_pos": want["pos"].tolist(), "env_quat": q.tolist(),
                  "demo_quat": want["quat"].tolist(), "demo_quat_present": want["has_quat"],
                  "z_diff": float(abs(p[2] - want["pos"][2])), "quat_max_diff": q_d,
                  "region": region, "bddl_range_offsets_xmin_ymin_xmax_ymax": rng,
                  "world_x_range": [x0, x1], "world_y_range": [y0, y1],
                  "demo_xy_in_region": bool((x0 - POS_TOL <= want["pos"][0] <= x1 + POS_TOL)
                                            and (y0 - POS_TOL <= want["pos"][1] <= y1 + POS_TOL))}
    return bool(ok_all), det


def copy_fixtures(env, parsed_demo, fixtures):
    """Write the demo's body pos and quat for each fixture into the env's sim model (after reset, before set_init_state)."""
    m = env.sim.model
    for f in fixtures:
        i = m.body_name2id(f)
        m.body_pos[i] = parsed_demo["bodies"][f]["pos"]
        m.body_quat[i] = parsed_demo["bodies"][f]["quat"]


def check_fixtures_after(env, parsed_demo, fixtures):
    """After the copy and the state restore: each fixture's pos and quat in the sim equal the demo's model_file values."""
    cur = read_bodies(env, fixtures)
    ok_all, det = True, {}
    for f in fixtures:
        want = parsed_demo["bodies"][f]
        pd = float(np.abs(cur[f]["pos"] - want["pos"]).max())
        q_ok, qd = quat_close(cur[f]["quat"], want["quat"])
        ok = bool(pd <= POS_TOL and q_ok)
        ok_all &= ok
        det[f] = {"ok": ok, "pos_max_diff": pd, "quat_max_diff": qd}
    return bool(ok_all), det


def restore(env, state):
    """set_init_state with a length check; returns the raw observation (images upside down, OpenGL convention)."""
    state = np.asarray(state, float)
    cur = len(env.sim.get_state().flatten())
    if cur != len(state):
        raise RenderCheckError(f"state length: env {cur}, stored {len(state)}")
    return env.set_init_state(state)


def images(obs):
    """Copies of the two 256 x 256 uint8 camera images of an observation."""
    return {k: np.array(obs[k], copy=True) for k in CAM_KEYS}


def camera_pose(env, name="agentview"):
    m = env.sim.model
    i = m.camera_name2id(name)
    return {"pos": np.array(m.cam_pos[i], float), "quat": np.array(m.cam_quat[i], float),
            "fovy": float(m.cam_fovy[i])}


def apply_saved_camera(env, parsed_demo, name="agentview"):
    """Q2 override: set the env's agentview camera to the pose saved in the demo's model_file."""
    m = env.sim.model
    i = m.camera_name2id(name)
    m.cam_pos[i] = parsed_demo["cameras"][name]["pos"]
    m.cam_quat[i] = parsed_demo["cameras"][name]["quat"]



def noise_seed(base_seed, stem, demo, frame):
    """Seed for one frame's noise: a pure function of (config seed, task file stem, demo name, frame index), so a frame gets the
    same noise whatever else ran before it (resume, order, --max-demos). 32 bits, what np.random.seed takes."""
    h = hashlib.sha256(f"{int(base_seed)}|{stem}|{demo}|{int(frame)}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def apply_noise(img, code, seed, ew=None):
    """LIBERO-Plus's own sensor noise (libero.libero.envs.env_wrapper, ControlEnv.step), on a 256 x 256 uint8 image.
    code 1-10 motion blur, 11-20 Gaussian, 21-30 zoom, 31-40 fog, 41-50 glass blur. The functions draw from the global
    numpy RNG (verified in the source: np.random.uniform and np.random.randint only), so it is seeded here with `seed` (use
    noise_seed per frame) and the caller's global RNG state is saved before and restored after, also when a noise function
    raises. `ew` replaces the env_wrapper module (tests only)."""
    if ew is None:
        import libero.libero.envs.env_wrapper as ew
    if not 1 <= int(code) <= 50:
        raise ValueError(code)
    saved = np.random.get_state()
    try:
        np.random.seed(int(seed))
        pil = Image.fromarray(img)
        if code <= 10:
            out = ew.motion_blur(pil, severity=code)
        elif code <= 20:
            out = ew.gaussian_blur(pil, severity=code - 10)
        elif code <= 30:
            out = ew.zoom_blur(pil, severity=code - 20)
        elif code <= 40:
            out = ew.fog(pil, severity=code - 30)
        else:
            out = ew.glass_blur(pil, severity=code - 40)
        return np.asarray(out).astype(np.uint8)
    finally:
        np.random.set_state(saved)


# ---------------------------------------------------------------------------
# draws of perturbation settings (PLAN Q3, Q4): one per (test demo, type), uniform over LIBERO-Plus's own episode names
# ---------------------------------------------------------------------------

PTYPES = ("camera", "light", "background", "noise")
# type -> (task_classification.json category, regex for the part of the name after "<base task>_")
EPISODE_PATTERN = {
    "camera": ("Camera Viewpoints", r"view_\d+_\d+_\d+_\d+_\d+_initstate_0"),
    "light": ("Light Conditions", r"light_\d+"),
    "background": ("Background Textures", r"(?:table|tb)_\d+"),     # table texture (table_N) and wall theme (tb_N), pooled
    "noise": ("Sensor Noise", r"view_0_0_100_0_0_initstate_0_noise_(\d+)"),
}


def episode_names(classification, base, ptype, suite="libero_goal"):
    """Sorted names of LIBERO-Plus's own episodes of one type for one base task. Every name of that category that starts with
    the base task name must fit the expected pattern (so a camera or noise name with initstate other than 0 is an error, not
    skipped). Background pools table_N and tb_N, the two kinds in the benchmark's Background Textures category."""
    cat, pat = EPISODE_PATTERN[ptype]
    rx = re.compile(pat)
    names = []
    for e in classification[suite]:
        if e["category"] != cat or not e["name"].startswith(base + "_"):
            continue
        if not rx.fullmatch(e["name"][len(base) + 1:]):
            raise RenderCheckError(f"{ptype} episode name {e['name']!r} does not fit {pat!r}")
        names.append(e["name"])
    names = sorted(set(names))
    if not names:
        raise RenderCheckError(f"no {ptype} episodes for {base}")
    return names


def noise_code(name):
    m = re.search(r"_noise_(\d+)$", name)
    if not m:
        raise ValueError(name)
    return int(m.group(1))


def draw_episode(classification, base, stem, demo, ptype, seed, suite="libero_goal"):
    """One uniform draw over episode_names(...). Pure function of (seed, stem, demo, type) and the name list: no RNG state, so
    the same inputs give the same draw in any order. Returns the name, its index in the sorted list, the list size and a hash of
    the list (so a changed classification file shows up)."""
    names = episode_names(classification, base, ptype, suite)
    h = hashlib.sha256(f"{int(seed)}|draw|{stem}|{demo}|{ptype}".encode()).digest()
    i = int.from_bytes(h[:8], "big") % len(names)       # modulo bias at most len(names) / 2^64
    return {"name": names[i], "index": i, "n_choices": len(names),
            "list_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest()}


def draw_all(classification, files, seed, ptypes=PTYPES):
    """files: [(stem, base task name, [test demo names])]. Returns {"stem/demo/type": draw_episode(...)}."""
    out = {}
    for stem, base, demos in files:
        for d in demos:
            for t in ptypes:
                out[f"{stem}/{d}/{t}"] = draw_episode(classification, base, stem, d, t, seed)
    return out


# ---------------------------------------------------------------------------
# raw-file access with the split guard (PLAN Amendment 2026-11-09 section 7, "silent split error", and R7)
# ---------------------------------------------------------------------------

def demo_index(name):
    return int(name.split("_")[-1])


def split_checked(all_names):
    """{demo: train|val|test} from ALL demo names of a raw file with diagnostics.demo_split (the tally's own split), then the
    explicit check that train is exactly demo_0..demo_29, val exactly demo_30..demo_39 and test exactly demo_40..demo_49.
    Raises SplitAccessError otherwise. Needs names only, so it runs before any demo content is read."""
    import diagnostics as D
    ordered = sorted(all_names, key=demo_index)
    split = D.demo_split(ordered)
    want = {"train": [f"demo_{i}" for i in range(0, 30)], "val": [f"demo_{i}" for i in range(30, 40)],
            "test": [f"demo_{i}" for i in range(40, 50)]}
    for sp, w in want.items():
        got = [d for d in ordered if split[d] == sp]
        if got != w:
            raise SplitAccessError(f"split {sp}: expected {w[0]}..{w[-1]} ({len(w)} demos), got {got[:3]}..{got[-3:]} "
                                   f"({len(got)} demos) from {len(ordered)} names")
    return split


class RawFile:
    """A raw LIBERO HDF5 opened for a run of one split. The h5py handle is private; the only ways to read content are
    demo_arrays (states and actions, only for demos of the run's split) and model_file (the scene text; allowed for train and
    val demos in any run, and for test demos only in a test run). A train or val run therefore cannot open a test demo's group:
    the call raises SplitAccessError before h5py is touched. Every access is logged in `access_log`. Listing names is free."""

    def __init__(self, path, run_split, opener=None):
        if run_split not in ("train", "val", "test"):
            raise ValueError(run_split)
        if opener is None:
            import h5py
            opener = h5py.File
        self.path, self.run_split = str(path), run_split
        self._f = opener(path, "r")
        self._data = self._f["data"]
        self.all_names = sorted(self._data.keys(), key=demo_index)       # listing the keys opens no demo group
        self.access_log = []
        try:
            self.split = split_checked(self.all_names)                    # before any content is read
        except Exception:
            self._f.close()
            raise

    def names(self, split):
        return [d for d in self.all_names if self.split[d] == split]

    def model_file(self, name):
        if name not in self.split:
            raise SplitAccessError(f"unknown demo {name}")
        if self.split[name] == "test" and self.run_split != "test":
            raise SplitAccessError(f"{name} is a test demo; a {self.run_split} run may not open it")
        self.access_log.append(("model_file", name))
        return self._data[name].attrs["model_file"]

    def demo_arrays(self, name):
        if name not in self.split:
            raise SplitAccessError(f"unknown demo {name}")
        if self.split[name] != self.run_split:
            raise SplitAccessError(f"{name} is a {self.split[name]} demo; a {self.run_split} run may not read its states or actions")
        self.access_log.append(("arrays", name))
        g = self._data[name]
        return {"states": g["states"][:], "actions": g["actions"][:]}

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def reject_other_rotations(parsed, fixtures):
    """copy_fixtures copies pos and quat only; a fixture body that carries euler, axisangle, xyaxes or zaxis would be placed
    wrongly without notice (reviewer note, LOG 2026-11-08)."""
    bad = {f: parsed["bodies"][f]["other_rot"] for f in fixtures if parsed["bodies"][f]["other_rot"]}
    if bad:
        raise RenderCheckError(f"fixture bodies with rotation attributes other than quat: {bad}")


def scene_file_check(nontest, test, fixtures):
    """PLAN Amendment 2026-11-09 item 8 (decided): compare every test demo's model_file (parsed) against the train and val demos of
    the same file with differing_bodies. Stop (SceneMismatch) if the body-name sets differ, or if a body outside `fixtures` differs
    in pos or quat. Reads inputs only. nontest, test: {demo: parse_model_file(...)}. Returns a report dict."""
    base = list(nontest.values())
    fx = set(fixtures)
    rep, extra_all = {}, {}
    for d, p in test.items():
        diff, _ = differing_bodies(base + [p])                    # raises SceneMismatch on a body-name set difference
        extra = [b for b in diff if b not in fx]
        rep[d] = {"differing": diff, "outside_fixture_list": extra}
        if extra:
            extra_all[d] = extra
    if extra_all:
        raise SceneMismatch(f"test demos differ from the train/val demos in bodies outside the fixture list {sorted(fx)}: {extra_all}")
    return {"n_test_compared": len(test), "n_nontest": len(base), "per_demo": rep, "ok": True}


# ---------------------------------------------------------------------------
# render consistency (PLAN Amendment 2026-11-09 (b)): comparison of environment records and of the pixels
# ---------------------------------------------------------------------------

def _flat(d, pre=""):
    r = {}
    for k, v in d.items():
        if isinstance(v, dict):
            r.update(_flat(v, pre + k + "."))
        else:
            r[pre + k] = v
    return r


def compare_provenance(ref, new):
    """Compare two environment records ({"software": {...}, "hardware": {...}}, as pf_rerender.collect_environment writes them).
    Software (libraries, LIBERO-Plus commit and source fingerprint, environment bundle, image digest, MUJOCO_GL): any difference,
    or a field the reference did not record, goes into `stop`. Hardware (GPU name, driver, GL strings): differences go into
    `recorded` only; the pixel comparison decides. Returns {"stop": [...], "recorded": [...], "same": [...]}."""
    out = {"stop": [], "recorded": [], "same": []}
    fr, fn = _flat(ref.get("software") or {}), _flat(new.get("software") or {})
    for k in sorted(set(fr) | set(fn)):
        if k not in fr:
            out["stop"].append(f"software {k}: not recorded in the reference (now {fn[k]!r})")
        elif k not in fn:
            out["stop"].append(f"software {k}: not recorded now (reference {fr[k]!r})")
        elif fr[k] != fn[k]:
            out["stop"].append(f"software {k}: reference {fr[k]!r} vs now {fn[k]!r}")
        else:
            out["same"].append(k)
    hr, hn = _flat(ref.get("hardware") or {}), _flat(new.get("hardware") or {})
    for k in sorted(set(hr) | set(hn)):
        if hr.get(k) != hn.get(k):
            out["recorded"].append(f"hardware {k}: reference {hr.get(k)!r} vs now {hn.get(k)!r}")
        else:
            out["same"].append("hardware." + k)
    return out


def compare_frames(ref_path, new_path, demos=None):
    """Byte-for-byte comparison of the uint8 128 x 128 render outputs of two HDF5 files (LIBERO layout, data/<demo>/obs/<cam>).
    demos: names to compare, default every demo present in either file (one present on one side only is reported missing). Returns {"identical": bool, "demos": {demo: {cam: {...}}}}.
    Any difference in any value, a demo missing in either file, or a shape difference makes identical False."""
    import h5py
    res = {"identical": True, "demos": {}, "missing": []}
    with h5py.File(ref_path, "r") as a, h5py.File(new_path, "r") as b:
        da, db = a["data"], b["data"]
        names = list(demos) if demos is not None else sorted(set(da.keys()) | set(db.keys()), key=demo_index)
        if not names:
            res["identical"] = False
            res["missing"].append("no demos to compare")
        for d in names:
            if d not in da or d not in db:
                res["identical"] = False
                res["missing"].append(d)
                continue
            res["demos"][d] = {}
            for c in HDF5_KEY.values():
                x, y = da[d]["obs"][c][:], db[d]["obs"][c][:]
                if x.shape != y.shape or x.dtype != y.dtype:
                    r = {"identical": False, "shape_ref": list(x.shape), "shape_new": list(y.shape), "n_different": None}
                else:
                    diff = x != y
                    nd = int(diff.sum())
                    r = {"identical": nd == 0, "n_different": nd, "n_values": int(x.size),
                         "max_abs_diff": int(np.abs(x.astype(np.int16) - y.astype(np.int16)).max()) if nd else 0,
                         "n_frames_different": int(diff.reshape(len(x), -1).any(axis=1).sum()) if nd else 0}
                res["demos"][d][c] = r
                res["identical"] &= r["identical"]
    return res
