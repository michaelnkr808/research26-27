"""Render helpers for the perturbed-frame test (experiments/2026-11-06-perturbed-frames, PLAN Handoff steps 1-2).

Re-render a stored libero_goal demo moment inside a LIBERO-Plus env (clean, camera, light, background or noise):
  reset  ->  (Q20 check)  ->  copy the demo's fixture body pos/quat from its `model_file`  ->  set_init_state(states[k+1])
  ->  read the 256 x 256 images  ->  (noise at 256)  ->  shrink to 128 x 128.
Image k of a stored demo pairs with stored state k+1; the last image has no state (PLAN Q14, Feasibility 7b).

Needs the pod environment of the proxy eval Jobs (LIBERO-Plus @ 4976dc3 on PYTHONPATH, LIBERO_CONFIG_PATH set, MUJOCO_GL=egl).
The pure functions (XML parsing, resize, fixture comparison) import nothing from the simulator, so they can be tested anywhere.
"""

import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

RENDER = 256          # LIBERO_ENV_RESOLUTION, as src/rollout.py (PLAN Q15)
OUT_SIZE = 128        # stored frames are 128 x 128
AGENT, WRIST = "agentview_image", "robot0_eye_in_hand_image"
CAM_KEYS = (AGENT, WRIST)
IDENT_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
POS_TOL = 1e-5        # model_file holds 6 significant digits (inference plus recall; PLAN Feasibility 6b); values ~1 m


# ---------------------------------------------------------------------------
# model_file (scene XML stored in the HDF5) -> bodies and cameras
# ---------------------------------------------------------------------------

def _vec(s, n, default=None):
    if s is None:
        return default
    v = np.array([float(x) for x in s.split()])
    assert len(v) == n, (s, n)
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
        assert name not in bodies, f"duplicate body name {name}"
        bodies[name] = {"pos": _vec(b.get("pos"), 3, np.zeros(3)), "quat": _vec(b.get("quat"), 4, IDENT_QUAT.copy()),
                        "has_quat": b.get("quat") is not None}
    for c in root.iter("camera"):
        name = c.get("name")
        if name is not None:
            cams[name] = {"pos": _vec(c.get("pos"), 3), "quat": _vec(c.get("quat"), 4)}
    return {"bodies": bodies, "cameras": cams}


def differing_bodies(parsed_list):
    """Names of bodies whose pos or quat differs between the parsed model files (the fixture list, PLAN Q19), and a check
    that every parsed file has the same set of body names. Returns (sorted names, per-name info for the report)."""
    names0 = set(parsed_list[0]["bodies"])
    for p in parsed_list[1:]:
        assert set(p["bodies"]) == names0, "model files differ in their body names"
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
    assert img.shape == (RENDER, RENDER, 3) and img.dtype == np.uint8, (img.shape, img.dtype)
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
    assert len(st) == 1, (fixture_name, st)
    ranges = regs[st[0][2]]["ranges"]
    assert len(ranges) == 1, (st[0][2], ranges)
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
    assert cur == len(state), f"state length: env {cur}, stored {len(state)}"
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


def apply_noise(img, code, seed):
    """LIBERO-Plus's own sensor noise (libero.libero.envs.env_wrapper, ControlEnv.step), on a 256 x 256 uint8 image.
    code 1-10 motion blur, 11-20 Gaussian, 21-30 zoom, 31-40 fog, 41-50 glass blur. The functions draw from the global
    numpy RNG, so it is seeded here."""
    import libero.libero.envs.env_wrapper as EW
    np.random.seed(seed)
    pil = Image.fromarray(img)
    if code <= 10:
        out = EW.motion_blur(pil, severity=code)
    elif code <= 20:
        out = EW.gaussian_blur(pil, severity=code - 10)
    elif code <= 30:
        out = EW.zoom_blur(pil, severity=code - 20)
    elif code <= 40:
        out = EW.fog(pil, severity=code - 30)
    elif code <= 50:
        out = EW.glass_blur(pil, severity=code - 40)
    else:
        raise ValueError(code)
    return np.asarray(out).astype(np.uint8)
