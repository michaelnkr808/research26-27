"""Logic tests for pf_render.py and pf_rerender.py (experiments/2026-11-06-perturbed-frames). Run: python test_pf_render.py

The Mac has no MuJoCo or robosuite, so nothing here renders. The render loop is run against a STUB simulator (a small fake env whose
"images" are a deterministic function of the restored state). These are LOGIC tests (split guard, draws, RNG handling, pairing of
image k with state k+1, scene-file comparison, resume, consistency comparison, exit codes). They are not a stand-in for any rendered
result: no image from here is a render, and none is kept.

Real inputs used (train and val demos only; a test demo's group is never opened, and the spy below proves it):
  - the ten downloaded original libero_goal files (read only), for the split guard and the train/val scene comparison;
  - LIBERO-Plus's task_classification.json from the source copy, for the draws (synthetic fallback if the file is gone).
The LIBERO-Plus noise functions are NOT importable here (they import wand, cv2 and robosuite); apply_noise itself is the real function,
run with stand-in blur functions that draw from the global numpy RNG as the real ones do.

`python test_pf_render.py --real-only` runs only the train/val scene comparison over the ten downloaded files and prints each file's
fixture list (compared with results/pf-val/20261108-181737/render-report.json).
"""

import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

import pf_render as R
import pf_rerender as PR

SCRATCH = "/private/tmp/claude-501/-Users-michaelr808-Desktop-research-research26-27"
REAL_DIR = Path(SCRATCH) / "299bb972-847f-46ad-ae82-6ea1b7e3cb50/scratchpad/libero-datasets/libero_goal"
CLASSIFICATION = Path(SCRATCH) / "783ff06c-cba7-4999-9fa7-738b18aab793/scratchpad/libero-plus-src/libero/libero/benchmark/task_classification.json"
VAL_REPORT = Path(__file__).resolve().parents[1] / "experiments/2026-11-06-perturbed-frames/results/pf-val/20261108-181737/render-report.json"
CFG = {"seed_draw": 123, "seed_noise": 7, "resize": "area", "suite": "libero_goal", "consistency_demo": "demo_30", "n_files": 3}


def expect(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError(f"expected {exc.__name__}")


# ---------------------------------------------------------------------------
# synthetic raw files (LIBERO layout) and a stub simulator
# ---------------------------------------------------------------------------

def model_xml(i, move=None, rename=None, camera_dx=0.0):
    """Scene XML of a synthetic demo i. Fixtures (flat_stove_1_main, wooden_cabinet_1_main) move with i; mount_main and bowl_main
    do not. `move`: {body: new pos string}, `rename`: {old: new}."""
    pos = {"flat_stove_1_main": f"{-0.2 + 0.001 * i:.6f} 0.3 0.9", "wooden_cabinet_1_main": f"0.1 {-0.3 + 0.002 * i:.6f} 0.9",
           "mount_main": "0 0 0.5", "bowl_main": "0.1 0.2 0.95"}
    pos.update(move or {})
    names = {k: (rename or {}).get(k, k) for k in pos}
    quat = {"flat_stove_1_main": "", "wooden_cabinet_1_main": ' quat="0 0 0 1"', "mount_main": "", "bowl_main": ""}
    bodies = "".join(f'<body name="{names[k]}" pos="{pos[k]}"{quat[k]}/>' for k in pos)
    return (f'<mujoco><worldbody>{bodies}<camera name="agentview" pos="{0.658613 + camera_dx} 0 1.61035" quat="0.638018 0.30485 0.30485 0.638018"/>'
            f'</worldbody></mujoco>')


def make_raw(root, stems=("task_a_demo", "task_b_demo", "task_c_demo"), n_demo=50, state_len=79, seed=0, xml=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for s in stems:
        with h5py.File(root / (s + ".hdf5"), "w") as f:
            g = f.create_group("data")
            for i in range(n_demo):
                n = 12 + (i % 5)
                d = g.create_group(f"demo_{i}")
                d.create_dataset("states", data=rng.normal(size=(n, state_len)))
                d.create_dataset("actions", data=rng.normal(size=(n, 7)))
                d.attrs["model_file"] = (xml or model_xml)(i)
    return root


STUB_SPEC = {"fix": {"flat_stove_1_main": (0.9, [1, 0, 0, 0]), "wooden_cabinet_1_main": (0.9, [0, 0, 0, 1])},
             "cam": ([0.658613, 0.0, 1.61035], [0.638018, 0.30485, 0.30485, 0.638018])}


def spec_from_xml(xml):
    """Stub spec (what the stub env's own reset puts at the fixtures and the camera) taken from one demo's scene text."""
    p = R.parse_model_file(xml)
    fx = {n: (float(b["pos"][2]), b["quat"].tolist()) for n, b in p["bodies"].items() if n in ("flat_stove_1_main", "wine_rack_1_main", "wooden_cabinet_1_main")}
    return {"fix": fx, "cam": (p["cameras"]["agentview"]["pos"].tolist(), p["cameras"]["agentview"]["quat"].tolist())}


def stub_image(state, key, variant, shift):
    seed = int.from_bytes(hashlib.sha256(np.asarray(state, float).tobytes() + key.encode() + variant.encode()).digest()[:8], "big")
    im = np.random.default_rng(seed).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    if shift:
        im[5, 5, 0] ^= 1                      # one value of one pixel
    return im


class StubEnv:
    def __init__(self, name, state_len, log, shift=0, bad_z=False, spec=STUB_SPEC):
        self.name, self.log, self.shift, self.state_len, self.bad_z, self.spec = name, log, shift, state_len, bad_z, spec
        fx = list(spec["fix"])
        self.model = SimpleNamespace(
            names=fx + ["mount_main"], body_pos=np.zeros((len(fx) + 1, 3)), body_quat=np.tile([1.0, 0, 0, 0], (len(fx) + 1, 1)),
            cam_pos=np.zeros((1, 3)), cam_quat=np.zeros((1, 4)), cam_fovy=np.full(1, 45.0))
        self.model.body_name2id = lambda n: self.model.names.index(n)
        self.model.camera_name2id = lambda n: 0
        self.sim = SimpleNamespace(model=self.model, get_state=lambda: SimpleNamespace(flatten=lambda: np.zeros(state_len)))
        self.env = SimpleNamespace(
            fixtures_dict={n + "_obj": SimpleNamespace(root_body=n) for n in fx},
            parsed_problem={"regions": {"reg": {"ranges": [[-1.0, -1.0, 1.0, 1.0]]}},
                            "initial_state": [("On", n + "_obj", "reg") for n in fx]},
            workspace_offset=[0.0, 0.0, 0.0], _arena_xml="/x/assets/scenes/" + name + ".xml")

    def reset(self):
        for n, (z, q) in self.spec["fix"].items():
            i = self.model.body_name2id(n)
            self.model.body_pos[i] = [0.05, 0.05, z + (0.01 if self.bad_z else 0.0)]
            self.model.body_quat[i] = q
        pos, quat = self.spec["cam"]
        self.model.cam_pos[0] = np.array(pos) + (1e-3 if "_light_" in self.name else 0.0)      # light envs: pose off by more than 1e-4
        self.model.cam_quat[0] = quat
        if "_view_" in self.name:
            self.model.cam_pos[0] = np.array(pos) + 0.1                                          # the camera perturbation

    def set_init_state(self, state):
        v = self.name
        perturbed = any(t in v for t in ("_light_", "_table_", "_tb_"))
        return {"agentview_image": stub_image(state, "a", v if (perturbed or "_view_" in v) else "", self.shift),
                "robot0_eye_in_hand_image": stub_image(state, "w", v if perturbed else "", self.shift)}

    def close(self):
        self.log["closed"].append(self.name)


class Stub:
    """Installs the stub env, a numpy-RNG stand-in for the LIBERO-Plus noise functions, and records calls."""

    def __init__(self, state_len=79, shift=0, bad_z=False, fail_on_make=None, spec=STUB_SPEC):
        self.log = {"make": [], "closed": [], "noise": []}
        self.state_len, self.shift, self.bad_z, self.fail_on_make, self.spec = state_len, shift, bad_z, fail_on_make, spec

    def __enter__(self):
        self.orig_make, self.orig_noise = R.make_env, R.apply_noise

        def make(name, size=R.RENDER):
            self.log["make"].append(name)
            if self.fail_on_make and self.fail_on_make(name):
                raise RuntimeError("stub: cannot build env " + name)
            return StubEnv(name, self.state_len, self.log, self.shift, self.bad_z, self.spec)

        def blur(pil, severity=1):
            a = np.asarray(pil).astype(int)
            return np.clip(a + np.random.randint(-severity, severity + 1, a.shape) + int(np.random.uniform(-1, 1)), 0, 255).astype(np.uint8)
        ew = SimpleNamespace(motion_blur=blur, gaussian_blur=blur, zoom_blur=blur, fog=blur, glass_blur=blur)

        def noise(img, code, seed):
            assert img.shape == (256, 256, 3), f"noise must be applied to the 256 x 256 render before the shrink, got {img.shape}"
            self.log["noise"].append((code, seed))
            return self.orig_noise(img, code, seed, ew=ew)
        R.make_env, R.apply_noise = make, noise
        return self

    def __exit__(self, *a):
        R.make_env, R.apply_noise = self.orig_make, self.orig_noise


ENV = {"software": {"libraries": {"mujoco": "3.3.2", "robosuite": "1.4.0", "numpy": "1.26.4", "Pillow": "10.0"},
                    "libero_plus_commit": "4976dc3", "libero_source_fingerprint": "f" * 8, "bundle_key": "abc", "bundle_tar_sha256": None,
                    "image_digest": "sha256:1", "mujoco_gl": "egl"},
       "hardware": {"nvidia_smi": {"gpu_name": "NVIDIA L4", "driver_version": "550.1"}}}


def env_info(**sw):
    e = json.loads(json.dumps(ENV))
    for k, v in sw.items():
        if k.startswith("lib_"):
            e["software"]["libraries"][k[4:]] = v
        elif k.startswith("hw_"):
            e["hardware"]["nvidia_smi"][k[3:]] = v
        else:
            e["software"][k] = v
    return e


def classification_for_tests():
    if CLASSIFICATION.exists():
        return json.loads(CLASSIFICATION.read_text()), "real LIBERO-Plus task_classification.json"
    cl = {"libero_goal": []}
    for b in ("task_a", "task_b", "task_c"):
        for i in range(1, 8):
            cl["libero_goal"] += [{"name": f"{b}_light_{i}", "category": "Light Conditions"}, {"name": f"{b}_table_{i}", "category": "Background Textures"},
                                  {"name": f"{b}_tb_{i}", "category": "Background Textures"},
                                  {"name": f"{b}_view_{i}_0_100_1_0_initstate_0", "category": "Camera Viewpoints"},
                                  {"name": f"{b}_view_0_0_100_0_0_initstate_0_noise_{i * 3}", "category": "Sensor Noise"}]
    return cl, "SYNTHETIC classification (the LIBERO-Plus file was not found)"


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_apply_noise():
    img = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)

    def blur(pil, severity=1):
        a = np.asarray(pil).astype(int)
        return np.clip(a + np.random.randint(-3, 4, a.shape), 0, 255).astype(np.uint8)

    def bad(pil, severity=1):
        np.random.rand(3)
        raise RuntimeError("boom")
    ew = SimpleNamespace(motion_blur=blur, gaussian_blur=blur, zoom_blur=blur, fog=blur, glass_blur=blur)
    np.random.seed(999)
    before = np.random.get_state()
    a = R.apply_noise(img, 15, 41, ew=ew)
    after = np.random.get_state()
    assert before[0] == after[0] and (before[1] == after[1]).all() and before[2:] == after[2:], "global RNG state changed"
    b = R.apply_noise(img, 15, 41, ew=ew)
    c = R.apply_noise(img, 15, 42, ew=ew)
    assert (a == b).all(), "same seed, different output"
    assert not (a == c).all(), "different seed, same output"
    assert a.dtype == np.uint8 and a.shape == img.shape
    for code in (1, 10, 11, 25, 35, 50):
        R.apply_noise(img, code, 1, ew=ew)
    for code in (0, 51):
        expect(ValueError, R.apply_noise, img, code, 1, ew=ew)
    ew_bad = SimpleNamespace(motion_blur=bad, gaussian_blur=bad, zoom_blur=bad, fog=bad, glass_blur=bad)
    st = np.random.get_state()
    expect(RuntimeError, R.apply_noise, img, 5, 3, ew=ew_bad)
    st2 = np.random.get_state()
    assert (st[1] == st2[1]).all() and st[2] == st2[2], "RNG state not restored after an exception"
    s1, s2 = R.noise_seed(7, "t", "demo_40", 3), R.noise_seed(7, "t", "demo_40", 3)
    assert s1 == s2 and len({R.noise_seed(7, "t", "demo_40", k) for k in range(200)}) == 200
    assert R.noise_seed(7, "t", "demo_40", 3) != R.noise_seed(8, "t", "demo_40", 3) != R.noise_seed(7, "t2", "demo_40", 3)
    try:
        import libero.libero.envs.env_wrapper  # noqa
        real = "importable"
    except Exception as e:
        real = f"NOT importable here ({type(e).__name__}: {e})"
    print(f"apply_noise: real function, stand-in blur functions; seed determinism, RNG-state restore (also on error), code range ok. "
          f"LIBERO-Plus env_wrapper {real}")


def test_draws():
    cl, which = classification_for_tests()
    if "real" in which:
        bases = sorted(v["base"] for v in json.loads(VAL_REPORT.read_text())["files"].values())      # the ten task names
        assert len(bases) == 10
    else:
        bases = ["task_a", "task_b", "task_c"]
    sizes = {}
    for base in bases:
        for t in R.PTYPES:
            names = R.episode_names(cl, base, t)
            sizes[(base, t)] = len(names)
            assert names == sorted(names) and len(set(names)) == len(names)
            for n in names:
                assert n.startswith(base + "_")
                if t in ("camera", "noise"):
                    assert "_initstate_0" in n
                if t == "noise":
                    assert 1 <= R.noise_code(n) <= 50
    # determinism, order independence, membership
    demos = [f"demo_{i}" for i in range(40, 50)]
    base = bases[0]
    d1 = R.draw_all(cl, [("stem", base, demos)], 123)
    d2 = R.draw_all(cl, [("stem", base, list(reversed(demos)))], 123)
    assert d1 == d2, "draws depend on order"
    assert json.dumps(d1, sort_keys=True) == json.dumps(R.draw_all(cl, [("stem", base, demos)], 123), sort_keys=True)
    d3 = R.draw_all(cl, [("stem", base, demos)], 124)
    assert {k: v["name"] for k, v in d1.items()} != {k: v["name"] for k, v in d3.items()}, "seed has no effect"
    for key, v in d1.items():
        stem, d, t = key.split("/")
        names = R.episode_names(cl, base, t)
        assert v["name"] in names and names[v["index"]] == v["name"] and v["n_choices"] == len(names)
    # a draw is a draw from the full list, not three fixed ones: over many pseudo-demos every light and every background name appears
    for t in ("light", "background", "camera", "noise"):
        names = R.episode_names(cl, base, t)
        seen = {R.draw_episode(cl, base, "s", f"demo_{i}", t, 5)["name"] for i in range(25 * len(names))}
        assert seen == set(names), (t, len(seen), len(names))
    bgs = R.episode_names(cl, base, "background")
    assert any("_table_" in n for n in bgs) and any("_tb_" in n for n in bgs), "background should pool table_N and tb_N"
    assert len(R.episode_names(cl, base, "light")) > 3 and len(bgs) > 3
    # a name with initstate other than 0 in the camera or noise category is an error, not skipped
    bad = {"libero_goal": [{"name": f"{base}_view_0_0_100_2_4_initstate_1", "category": "Camera Viewpoints"}]}
    expect(R.RenderCheckError, R.episode_names, bad, base, "camera")
    print(f"draws ({which}): deterministic, order independent, from the full list per (task, type); list sizes for {base}: "
          + ", ".join(f"{t} {sizes[(base, t)]}" for t in R.PTYPES))
    return sizes


def test_split_guard_synthetic():
    with tempfile.TemporaryDirectory() as t:
        raw = make_raw(Path(t) / "raw")
        with R.RawFile(raw / "task_a_demo.hdf5", "val") as rf:
            assert rf.names("val") == [f"demo_{i}" for i in range(30, 40)] and rf.names("train") == [f"demo_{i}" for i in range(30)]
            assert rf.names("test") == [f"demo_{i}" for i in range(40, 50)]
            rf.demo_arrays("demo_30")
            expect(R.SplitAccessError, rf.demo_arrays, "demo_45")          # test demo
            expect(R.SplitAccessError, rf.model_file, "demo_45")
            expect(R.SplitAccessError, rf.demo_arrays, "demo_3")           # train demo in a val run
            rf.model_file("demo_3")                                         # scene text of a train demo is fine in a val run
            assert all(R.demo_index(n) < 40 for _, n in rf.access_log)
        with R.RawFile(raw / "task_a_demo.hdf5", "train") as rf:
            expect(R.SplitAccessError, rf.demo_arrays, "demo_30")
            expect(R.SplitAccessError, rf.model_file, "demo_49")
        with R.RawFile(raw / "task_a_demo.hdf5", "test") as rf:
            rf.model_file("demo_49"); rf.demo_arrays("demo_49"); rf.model_file("demo_0")
            expect(R.SplitAccessError, rf.demo_arrays, "demo_0")
        # files whose split is not train 0-29 / val 30-39 / test 40-49 are refused at open, before anything is read
        bad = {"short": make_raw(Path(t) / "b1", ("s_demo",), n_demo=49), "long": make_raw(Path(t) / "b2", ("s_demo",), n_demo=60)}
        for k, d in bad.items():
            e = expect(R.SplitAccessError, R.RawFile, d / "s_demo.hdf5", "val")
        # same count but a gap in the names (demo_17 missing, demo_50 present): the sorted split puts the wrong demos in val
        gap = Path(t) / "gap"
        gap.mkdir()
        with h5py.File(gap / "s_demo.hdf5", "w") as f:
            g = f.create_group("data")
            for i in list(range(0, 17)) + list(range(18, 51)):
                g.create_group(f"demo_{i}").attrs["model_file"] = model_xml(i)
        expect(R.SplitAccessError, R.RawFile, gap / "s_demo.hdf5", "val")
        # a run on a bad file refuses before reading content
        e = expect(R.SplitAccessError, PR.render_run, bad["short"], Path(t) / "o", "val", "clean", dict(CFG, n_files=1), None, env_info())
    print("split guard (synthetic): val/train runs cannot read or open test demos; files with a wrong split refused at open")


class Spy:
    """Logs every demo group name that gets opened through RawFile on the real files."""
    def __init__(self, prefix):
        self.prefix, self.log = str(prefix), []

    def __enter__(self):
        self.orig = h5py.File
        spy = self

        class SG:
            def __init__(s, g):
                s._g = g

            def keys(s):
                return s._g.keys()

            def __getitem__(s, k):
                spy.log.append(k)
                return s._g[k]

        class SF:
            def __init__(s, f):
                s._f = f

            def __getitem__(s, k):
                g = s._f[k]
                return SG(g) if k == "data" else g

            def close(s):
                s._f.close()

        def patched(path, mode="r", *a, **k):
            f = spy.orig(path, mode, *a, **k)
            return SF(f) if mode == "r" and str(path).startswith(spy.prefix) else f
        h5py.File = patched
        return self

    def __exit__(self, *a):
        h5py.File = self.orig


def real_state_len(path):
    with h5py.File(path, "r") as f:
        return f["data"]["demo_0"]["states"].shape[1]


def test_real_files_guard_and_render():
    if not REAL_DIR.exists():
        print("real-file guard test SKIPPED: downloaded files not found")
        return
    stem = "turn_on_the_stove_demo"
    with tempfile.TemporaryDirectory() as t:
        raw = Path(t) / "raw"
        raw.mkdir()
        (raw / (stem + ".hdf5")).symlink_to(REAL_DIR / (stem + ".hdf5"))
        n = real_state_len(raw / (stem + ".hdf5"))
        with h5py.File(raw / (stem + ".hdf5"), "r") as f0:
            spec = spec_from_xml(f0["data"]["demo_0"].attrs["model_file"])      # a train demo
        cfg = dict(CFG, n_files=1)
        test_names = [f"demo_{i}" for i in range(40, 50)]
        for split in ("val", "train"):
            with Spy(raw) as spy, Stub(state_len=n, spec=spec) as stub:
                rep = PR.render_run(raw, Path(t) / ("out_" + split), split, "clean", cfg, None, env_info(), max_demos=2)
            opened = set(spy.log)
            assert opened and not (opened & set(test_names)), f"{split} run opened test demos: {opened & set(test_names)}"
            want = {"val": ["demo_30", "demo_31"], "train": ["demo_0", "demo_1"]}[split]
            assert rep["files"][stem]["demos"] == want and rep["files"][stem]["test_demo_reads"] == {"arrays": [], "model_file": []}
            assert rep["files"][stem]["fixtures"] == ["flat_stove_1_main", "wine_rack_1_main", "wooden_cabinet_1_main"]
            with h5py.File(Path(t) / ("out_" + split) / "libero_goal" / (stem + ".hdf5"), "r") as f:
                with h5py.File(raw / (stem + ".hdf5"), "r") as fr:
                    for d in want:
                        n_states = fr["data"][d]["states"].shape[0]
                        assert f["data"][d]["obs"]["agentview_rgb"].shape == (n_states - 1, 128, 128, 3)
            print(f"real file, {split} run (stub simulator, 2 demos): groups opened = {sorted(opened, key=R.demo_index)[:6]}..., none of demo_40..49")
        # the same run through a spy where RawFile is asked for a test demo: refused before h5py is touched
        with Spy(raw) as spy:
            with R.RawFile(raw / (stem + ".hdf5"), "val") as rf:
                expect(R.SplitAccessError, rf.demo_arrays, "demo_40")
                expect(R.SplitAccessError, rf.model_file, "demo_45")
            assert not (set(spy.log) & set(test_names))


def test_scene_file_check():
    def parsed(i, **kw):
        return R.parse_model_file(model_xml(i, **kw))
    nontest = {f"demo_{i}": parsed(i) for i in range(40)}
    fixtures, _ = R.differing_bodies(list(nontest.values()))
    assert fixtures == ["flat_stove_1_main", "wooden_cabinet_1_main"], fixtures
    ok = R.scene_file_check(nontest, {f"demo_{i}": parsed(i) for i in range(40, 50)}, fixtures)
    assert ok["ok"] and ok["n_test_compared"] == 10
    # planted: a non-fixture body moved in one test demo
    moved = {f"demo_{i}": parsed(i) for i in range(40, 50)}
    moved["demo_44"] = parsed(44, move={"mount_main": "0 0 0.5001"})
    e = expect(R.SceneMismatch, R.scene_file_check, nontest, moved, fixtures)
    assert "mount_main" in str(e) and "demo_44" in str(e)
    moved["demo_44"] = parsed(44, move={"bowl_main": "0.1 0.2 0.96"})
    expect(R.SceneMismatch, R.scene_file_check, nontest, moved, fixtures)
    # planted: a renamed body
    renamed = {f"demo_{i}": parsed(i) for i in range(40, 50)}
    renamed["demo_47"] = parsed(47, rename={"bowl_main": "bowl_main_2"})
    e = expect(R.SceneMismatch, R.scene_file_check, nontest, renamed, fixtures)
    assert "body names" in str(e)
    # a fixture moving alone is fine; a fixture body gaining an euler attribute is refused by reject_other_rotations
    R.reject_other_rotations(nontest["demo_0"], fixtures)
    p = R.parse_model_file(model_xml(0).replace('<body name="flat_stove_1_main" ', '<body name="flat_stove_1_main" euler="0 0 1" '))
    expect(R.RenderCheckError, R.reject_other_rotations, p, fixtures)
    # the explicit exception survives python -O (an assert would be stripped)
    code = ("import pf_render as R, sys\n"
            "a = R.parse_model_file('<mujoco><worldbody><body name=\"x\" pos=\"0 0 0\"/></worldbody></mujoco>')\n"
            "b = R.parse_model_file('<mujoco><worldbody><body name=\"y\" pos=\"0 0 0\"/></worldbody></mujoco>')\n"
            "try:\n    R.differing_bodies([a, b])\nexcept R.SceneMismatch:\n    print('raised')\n    sys.exit(0)\nprint('NOT raised')\nsys.exit(1)\n")
    r = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent))
    assert r.returncode == 0 and "raised" in r.stdout, (r.stdout, r.stderr)
    print("scene-file comparison: unchanged passes; planted moved non-fixture body and planted renamed body both stop; body-name check survives python -O")


def test_provenance_and_frames():
    base = env_info()
    c = R.compare_provenance(base, env_info())
    assert not c["stop"] and not c["recorded"]
    c = R.compare_provenance(base, env_info(lib_mujoco="3.3.3"))
    assert len(c["stop"]) == 1 and "mujoco" in c["stop"][0]
    assert R.compare_provenance(base, env_info(libero_plus_commit="beef"))["stop"]
    assert R.compare_provenance(base, env_info(libero_source_fingerprint="0"))["stop"]
    assert R.compare_provenance(base, env_info(bundle_key="zzz"))["stop"]
    c = R.compare_provenance(base, env_info(hw_driver_version="560.0", hw_gpu_name="NVIDIA A10"))
    assert not c["stop"] and len(c["recorded"]) == 2, c
    ref = json.loads(json.dumps(base))
    del ref["software"]["libraries"]["Pillow"]
    c = R.compare_provenance(ref, base)
    assert any("Pillow" in s and "not recorded in the reference" in s for s in c["stop"])
    assert R.compare_provenance({}, base)["stop"], "an empty reference (the pf-val report has none) must not pass silently"
    # strict mode: null on both sides is not "equal"; non-strict only lists it
    for key in ("image_digest", "bundle_key", "libero_source_fingerprint"):
        nul = env_info(**{key: None})
        assert not R.compare_provenance(nul, nul)["stop"] and R.compare_provenance(nul, nul)["unverified"]
        assert R.compare_provenance(nul, nul, strict=True)["stop"], key
        assert R.compare_provenance(base, nul, strict=True)["stop"] and R.compare_provenance(nul, base, strict=True)["stop"]
    for lib in ("mujoco", "robosuite"):
        nul = env_info(**{"lib_" + lib: None})
        assert R.compare_provenance(nul, nul, strict=True)["stop"], lib
    assert not R.compare_provenance(base, env_info(), strict=True)["stop"]
    # shrink's error is a RuntimeError subclass again (pf_validate.render catches RuntimeError)
    e = expect(R.RenderCheckError, R.shrink, np.zeros((128, 128, 3), np.uint8), "area")
    assert isinstance(e, RuntimeError)
    with tempfile.TemporaryDirectory() as t:
        def write(p, arr_a, arr_w):
            with h5py.File(p, "w") as f:
                g = f.create_group("data")
                for d, (a, w) in {"demo_30": (arr_a, arr_w), "demo_31": (arr_a, arr_w)}.items():
                    o = g.create_group(d).create_group("obs")
                    o.create_dataset("agentview_rgb", data=a)
                    o.create_dataset("eye_in_hand_rgb", data=w)
        rng = np.random.default_rng(1)
        a = rng.integers(0, 256, (6, 128, 128, 3), dtype=np.uint8)
        w = rng.integers(0, 256, (6, 128, 128, 3), dtype=np.uint8)
        write(Path(t) / "ref.h5", a, w)
        write(Path(t) / "same.h5", a.copy(), w.copy())
        a2 = a.copy(); a2[3, 17, 90, 1] ^= 1
        write(Path(t) / "onepx.h5", a2, w)
        w2 = w.copy(); w2[0, 0, 0, 0] ^= 128
        write(Path(t) / "wrist.h5", a, w2)
        write(Path(t) / "shape.h5", a[:5], w[:5])
        assert R.compare_frames(Path(t) / "ref.h5", Path(t) / "same.h5")["identical"]
        r = R.compare_frames(Path(t) / "ref.h5", Path(t) / "onepx.h5")
        assert not r["identical"] and r["demos"]["demo_30"]["agentview_rgb"]["n_different"] == 1 and r["demos"]["demo_30"]["eye_in_hand_rgb"]["identical"]
        assert not R.compare_frames(Path(t) / "ref.h5", Path(t) / "wrist.h5")["identical"]
        assert not R.compare_frames(Path(t) / "ref.h5", Path(t) / "shape.h5")["identical"]
        with h5py.File(Path(t) / "one_demo.h5", "w") as f:             # a demo present in the reference only (and the other way round)
            o = f.create_group("data").create_group("demo_30").create_group("obs")
            o.create_dataset("agentview_rgb", data=a)
            o.create_dataset("eye_in_hand_rgb", data=w)
        r1 = R.compare_frames(Path(t) / "ref.h5", Path(t) / "one_demo.h5")
        r2 = R.compare_frames(Path(t) / "one_demo.h5", Path(t) / "ref.h5")
        assert not r1["identical"] and r1["missing"] == ["demo_31"] and not r2["identical"] and r2["missing"] == ["demo_31"]
        assert not R.compare_frames(Path(t) / "ref.h5", Path(t) / "same.h5", ["demo_99"])["identical"]
    print("consistency comparison: identical passes; one changed value in one pixel fails; library, LIBERO-Plus, bundle difference stops; "
          "driver/GPU difference only recorded; empty or incomplete reference stops")


def test_tolerance():
    """PLAN Amendment 2026-11-09 (d): pass if every differing value is off by at most 1 and fewer than 1e-6 of an array's values differ."""
    assert R.MAX_ABS_DIFF_TOL == 1 and R.FRACTION_TOL == 1e-6
    n_frames = 24                                     # 24 * 128 * 128 * 3 = 1,179,648 values: one differing value is 8.5e-7 < 1e-6
    rng = np.random.default_rng(5)
    a = rng.integers(2, 250, (n_frames, 128, 128, 3), dtype=np.uint8)
    w = rng.integers(2, 250, (n_frames, 128, 128, 3), dtype=np.uint8)

    def write(p, arr_a, arr_w, demos=("demo_30", "demo_31")):
        with h5py.File(p, "w") as f:
            g = f.create_group("data")
            for d in demos:
                o = g.create_group(d).create_group("obs")
                o.create_dataset("agentview_rgb", data=arr_a)
                o.create_dataset("eye_in_hand_rgb", data=arr_w)

    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        write(t / "ref.h5", a, w)
        write(t / "same.h5", a.copy(), w.copy())
        r = R.compare_frames(t / "ref.h5", t / "same.h5")
        assert r["verdict"] == "identical" and r["identical"] and r["within_tolerance"]
        assert r["demos"]["demo_30"]["eye_in_hand_rgb"]["within_tolerance"]
        # one value of one frame off by one: within tolerance, reported
        w1 = w.copy(); w1[7, 40, 50, 2] += 1
        write(t / "one.h5", a, w1)
        r = R.compare_frames(t / "ref.h5", t / "one.h5")
        c = r["demos"]["demo_30"]["eye_in_hand_rgb"]
        assert r["verdict"] == "within_tolerance" and not r["identical"] and r["within_tolerance"], r
        assert c["n_different"] == 1 and c["max_abs_diff"] == 1 and c["n_frames_different"] == 1 and c["within_tolerance"] and not c["identical"]
        assert 0 < c["fraction_different"] < 1e-6 and c["n_values"] == n_frames * 128 * 128 * 3
        assert r["demos"]["demo_30"]["agentview_rgb"]["identical"] and r["tolerance"] == {"max_abs_diff": 1, "fraction": 1e-6}
        # off by 2: fails
        w2 = w.copy(); w2[7, 40, 50, 2] += 2
        write(t / "two.h5", a, w2)
        r = R.compare_frames(t / "ref.h5", t / "two.h5")
        c = r["demos"]["demo_30"]["eye_in_hand_rgb"]
        assert r["verdict"] == "exceeds_tolerance" and not r["within_tolerance"] and c["max_abs_diff"] == 2 and not c["within_tolerance"]
        # off by one in more than 1e-6 of the values (two values of 1.18 million): fails
        w3 = w.copy(); w3[7, 40, 50, 2] += 1; w3[8, 41, 50, 2] += 1
        write(t / "many.h5", a, w3)
        r = R.compare_frames(t / "ref.h5", t / "many.h5")
        c = r["demos"]["demo_30"]["eye_in_hand_rgb"]
        assert r["verdict"] == "exceeds_tolerance" and c["max_abs_diff"] == 1 and c["fraction_different"] > 1e-6 and not c["within_tolerance"]
        # a failing array anywhere fails the whole file, a within-tolerance one next to it does not rescue it
        w4 = w1.copy(); w4[0, 0, 0, 0] ^= 128
        write(t / "mixed.h5", a, w4)
        assert R.compare_frames(t / "ref.h5", t / "mixed.h5")["verdict"] == "exceeds_tolerance"
        # shape, dtype, missing demo: never covered by the tolerance
        write(t / "shape.h5", a[:23], w[:23])
        r = R.compare_frames(t / "ref.h5", t / "shape.h5")
        assert r["verdict"] == "exceeds_tolerance" and not r["demos"]["demo_30"]["agentview_rgb"]["within_tolerance"]
        write(t / "dtype.h5", a.astype(np.int16), w.astype(np.int16))
        r = R.compare_frames(t / "ref.h5", t / "dtype.h5")
        assert r["verdict"] == "exceeds_tolerance" and not r["demos"]["demo_30"]["agentview_rgb"]["within_tolerance"]
        write(t / "one_demo.h5", a, w, demos=("demo_30",))
        for x, y in (("ref.h5", "one_demo.h5"), ("one_demo.h5", "ref.h5")):
            r = R.compare_frames(t / x, t / y)
            assert r["verdict"] == "exceeds_tolerance" and r["missing"] == ["demo_31"] and not r["within_tolerance"]
        with h5py.File(t / "nokey.h5", "w") as f:
            o = f.create_group("data").create_group("demo_30").create_group("obs")
            o.create_dataset("agentview_rgb", data=a)
        with h5py.File(t / "ref1.h5", "w") as f:
            o = f.create_group("data").create_group("demo_30").create_group("obs")
            o.create_dataset("agentview_rgb", data=a)
            o.create_dataset("eye_in_hand_rgb", data=w)
        expect(KeyError, R.compare_frames, t / "ref1.h5", t / "nokey.h5")
        # file-level verdict
        ok = {"f1": {"verdict": "identical"}, "f2": {"verdict": "within_tolerance"}}
        assert PR.pixel_verdict(ok) == "within_tolerance" and PR.pixel_verdict({"f1": {"verdict": "identical"}}) == "identical"
        assert PR.pixel_verdict({**ok, "f3": {"verdict": "exceeds_tolerance"}}) == "exceeds_tolerance" and PR.pixel_verdict({}) == "exceeds_tolerance"
        assert PR.pixel_comparison(t, t / "nowhere", "libero_goal", ["x"])["x"]["verdict"] == "exceeds_tolerance"
    # code hashes: recorded, never a stop; they are not part of the environment record that compare_provenance reads
    d = PR.code_hash_differences({"pf_render.py": "aa", "pf_rerender.py": "bb"}, {"pf_render.py": "aa", "pf_rerender.py": "cc"})
    assert len(d) == 1 and "pf_rerender.py" in d[0] and not PR.code_hash_differences({"a": "1"}, {"a": "1"})
    assert not R.compare_provenance({**env_info(), "code_sha256": {"x": "1"}}, {**env_info(), "code_sha256": {"x": "2"}}, strict=True)["stop"]
    print("tolerance: identical; one value off by one in 1.18 M values passes as within_tolerance and is reported; off by 2, off by one in "
          "more than 1e-6 of the values, shape, dtype, missing demo, missing camera key fail; code-hash difference recorded, not a stop")


def read_frames(root, stem, d, cam):
    with h5py.File(Path(root) / "libero_goal" / (stem + ".hdf5"), "r") as f:
        return f["data"][d]["obs"][cam][:]


def test_render_contract_and_resume():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        raw = make_raw(t / "raw")
        stems = ["task_a_demo", "task_b_demo", "task_c_demo"]
        with Stub() as stub:
            rep = PR.render_run(raw, t / "val", "val", "clean", CFG, None, env_info())
            assert stub.log["make"] == [s[:-5] for s in stems] and stub.log["closed"] == stub.log["make"]
        # contract: n-1 frames per demo, image j = render of states[j+1], uint8 128 x 128, raw (not flipped), original stems and demo names
        for s in stems:
            with h5py.File(raw / (s + ".hdf5"), "r") as fr:
                assert rep["files"][s]["demos"] == [f"demo_{i}" for i in range(30, 40)]
                for d in ("demo_30", "demo_39"):
                    st = fr["data"][d]["states"][:]
                    a = read_frames(t / "val", s, d, "agentview_rgb")
                    w = read_frames(t / "val", s, d, "eye_in_hand_rgb")
                    assert a.shape == (len(st) - 1, 128, 128, 3) and a.dtype == np.uint8
                    for j in (0, len(st) - 2):
                        assert (a[j] == R.shrink(stub_image(st[j + 1], "a", "", 0), "area")).all(), "image j must be the render of states[j+1]"
                        assert (w[j] == R.shrink(stub_image(st[j + 1], "w", "", 0), "area")).all()
            with h5py.File(t / "val" / "libero_goal" / (s + ".hdf5"), "r") as f:
                assert sorted(f["data"].keys(), key=R.demo_index) == [f"demo_{i}" for i in range(30, 40)]
        rep0 = json.loads((t / "val" / "render-report.json").read_text())
        assert all(rep0["files"][s]["complete"] for s in stems) and rep0["checks"]["render_all_files"]["pass"]
        assert rep0["files"]["task_a_demo"]["fixtures"] == ["flat_stove_1_main", "wooden_cabinet_1_main"]
        # resume: nothing to do the second time
        inode = {s: (t / "val/libero_goal" / (s + ".hdf5")).stat().st_mtime_ns for s in stems}
        with Stub() as stub:
            rep = PR.render_run(raw, t / "val", "val", "clean", CFG, None, env_info())
            assert stub.log["make"] == [], "complete files must be skipped"
        assert all(rep["files"][s].get("skipped_on_resume") for s in stems)
        assert inode == {s: (t / "val/libero_goal" / (s + ".hdf5")).stat().st_mtime_ns for s in stems}
        # a .partial file with junk is redone; final file deleted -> redone; incomplete report entry -> redone; others skipped
        (t / "val/libero_goal/task_a_demo.hdf5").unlink()
        (t / "val/libero_goal/task_a_demo.hdf5.partial").write_bytes(b"junk from a killed run")
        r = json.loads((t / "val/render-report.json").read_text())
        r["files"]["task_b_demo"]["complete"] = False
        (t / "val/render-report.json").write_text(json.dumps(r))
        before_c = (t / "val/libero_goal/task_c_demo.hdf5").stat().st_mtime_ns
        with Stub() as stub:
            rep = PR.render_run(raw, t / "val", "val", "clean", CFG, None, env_info())
            assert stub.log["make"] == ["task_a", "task_b"], stub.log["make"]
        assert not (t / "val/libero_goal/task_a_demo.hdf5.partial").exists() and (t / "val/libero_goal/task_a_demo.hdf5").exists()
        assert (t / "val/libero_goal/task_c_demo.hdf5").stat().st_mtime_ns == before_c
        # killed mid-run: the first file stays complete, the second leaves only a .partial, the rerun finishes
        with Stub(fail_on_make=lambda n: n == "task_b") as stub:
            expect(RuntimeError, PR.render_run, raw, t / "mid", "val", "clean", CFG, None, env_info())
        assert (t / "mid/libero_goal/task_a_demo.hdf5").exists() and not (t / "mid/libero_goal/task_b_demo.hdf5").exists()
        rr = json.loads((t / "mid/render-report.json").read_text())
        assert rr["files"]["task_a_demo"]["complete"] and not rr["files"]["task_b_demo"]["complete"] and rr["failures"] and not rr["checks"]["render_all_files"]["pass"]
        with Stub() as stub:
            PR.render_run(raw, t / "mid", "val", "clean", CFG, None, env_info())
            assert stub.log["make"] == ["task_b", "task_c"]
        # a different run in the same directory, or a different software environment, is refused
        with Stub():
            expect(R.RenderCheckError, PR.render_run, raw, t / "mid", "train", "clean", CFG, None, env_info())
            expect(R.RenderCheckError, PR.render_run, raw, t / "mid", "val", "clean", CFG, None, env_info(lib_mujoco="9"))
        # --max-demos pilot: first N demos, recorded
        with Stub():
            rp = PR.render_run(raw, t / "pilot", "val", "clean", CFG, None, env_info(), max_demos=2)
        assert rp["files"]["task_a_demo"]["demos"] == ["demo_30", "demo_31"] and rp["max_demos"] == 2
    print("render contract (stub simulator): n-1 frames, image j from states[j+1], uint8 128x128, original stems and demo names; "
          "resume skips complete files and redoes .partial, deleted and incomplete ones; a killed run resumes; mixed runs refused")


def test_first_failure_stops():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        raw = make_raw(t / "raw")
        with Stub(bad_z=True) as stub:
            e = expect(R.RenderCheckError, PR.render_run, raw, t / "o", "val", "clean", CFG, None, env_info())
            assert "Q20" in str(e) and stub.log["make"] == ["task_a"], "must stop at the first failing demo of the first file"
        rep = json.loads((t / "o/render-report.json").read_text())
        assert rep["failures"] and "Q20" in rep["failures"][0] and not rep["files"]["task_a_demo"]["complete"]
        assert not (t / "o/libero_goal/task_a_demo.hdf5").exists()
        with Stub(state_len=80):
            e = expect(R.RenderCheckError, PR.render_run, raw, t / "o2", "val", "clean", CFG, None, env_info())
            assert "state shape" in str(e)
        # exit code through the CLI
        cfgp = t / "cfg.json"
        cfgp.write_text(json.dumps(CFG))
        orig_env, orig_cl = PR.collect_environment, PR.load_classification
        PR.collect_environment = lambda scripts_dir=None: env_info()
        PR.load_classification = lambda p=None: {"libero_goal": []}
        try:
            with Stub(bad_z=True):
                rc = PR.main(["render", "--split", "val", "--condition", "clean", "--raw", str(raw), "--out", str(t / "o3"), "--config", str(cfgp)])
            assert rc == PR.RC_CHECK
            with Stub():
                rc = PR.main(["render", "--split", "val", "--condition", "camera", "--raw", str(raw), "--out", str(t / "o4"), "--config", str(cfgp)])
            assert rc == PR.RC_CHECK                                         # perturbed conditions are test-only
            rc = PR.main(["render", "--split", "train", "--condition", "clean", "--raw", str(raw), "--out", str(t / "o5"), "--config", str(cfgp),
                          "--consistency-ref", str(t)])
            assert rc == PR.RC_USAGE
        finally:
            PR.collect_environment, PR.load_classification = orig_env, orig_cl
    print("first failure stops: Q20 failure and state-length mismatch raise at the first demo, later files untouched, report records it, "
          "CLI exits non-zero (3 check, 4 crash, 2 usage)")


def test_camera_override_and_perturbed_conditions():
    cl = {"libero_goal": [e for b in ("task_a", "task_b", "task_c") for e in synthetic_names(b)]}
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        raw = make_raw(t / "raw")
        cfg = CFG
        with Stub() as stub:
            clean = PR.render_run(raw, t / "clean", "test", "clean", cfg, None, env_info())
            assert stub.log["make"] == ["task_a", "task_b", "task_c"]
        reps = {}
        for cond in ("camera", "light", "background", "noise"):
            with Stub() as stub:
                reps[cond] = PR.render_run(raw, t / cond, "test", cond, cfg, cl, env_info())
                made = stub.log["make"]
            f = reps[cond]["files"]["task_a_demo"]
            drawn = [R.draw_episode(cl, "task_a", "task_a_demo", d, cond, cfg["seed_draw"])["name"] for d in f["demos"]]
            if cond == "noise":
                assert made == ["task_a", "task_b", "task_c"], "noise uses the base env"
                assert [R.noise_code(x) for x in drawn] == [f["per_demo"][d]["noise_code"] for d in f["demos"]]
                assert len(stub.log["noise"]) == sum(f2["frames"][d] for f2 in reps[cond]["files"].values() for d in f2["frames"])
            else:
                assert [m for m in made if m.startswith("task_a")] == drawn, "one new env per (demo, type), built from the drawn episode"
                assert sorted(stub.log["closed"]) == sorted(made), "every per-demo env is closed"
                if cond == "camera":
                    assert all(f["per_demo"][d]["camera"].get("override_skipped") for d in f["demos"])
                if cond == "light":
                    assert all(f["per_demo"][d]["camera"]["override_used"] for d in f["demos"]), "Q2 override applies to non-camera envs"
            assert f["demos"] == [f"demo_{i}" for i in range(40, 50)]
            assert f["scene_file_check"]["ok"] and f["test_demo_reads"]["arrays"] == f["demos"]
            for k, v in reps[cond]["draws"].items():
                assert v == R.draw_episode(cl, k.split("/")[0][:-5], k.split("/")[0], k.split("/")[1], cond, cfg["seed_draw"])
        # wrist and agentview relations to the clean re-render (stub images; logic only)
        s, d = "task_a_demo", "demo_40"
        ca, cw = read_frames(t / "clean", s, d, "agentview_rgb"), read_frames(t / "clean", s, d, "eye_in_hand_rgb")
        na, nw = read_frames(t / "noise", s, d, "agentview_rgb"), read_frames(t / "noise", s, d, "eye_in_hand_rgb")
        assert (nw == cw).all() and not (na == ca).all(), "noise: wrist equals clean, agentview differs"
        ka, kw = read_frames(t / "camera", s, d, "agentview_rgb"), read_frames(t / "camera", s, d, "eye_in_hand_rgb")
        assert (kw == cw).all() and not (ka == ca).all(), "camera: wrist equals clean, agentview differs"
        # noise is a pure function of (seed, stem, demo, frame): a pilot with --max-demos 1 gives the same demo_40 frames
        with Stub():
            PR.render_run(raw, t / "noise_pilot", "test", "noise", cfg, cl, env_info(), max_demos=1)
        assert (read_frames(t / "noise_pilot", s, d, "agentview_rgb") == na).all()
        # the noise changes with the config seed
        with Stub():
            PR.render_run(raw, t / "noise_seed", "test", "noise", dict(cfg, seed_noise=8), cl, env_info(), max_demos=1)
        assert not (read_frames(t / "noise_seed", s, d, "agentview_rgb") == na).all()
        # a planted non-fixture change in a test demo's scene file stops the test render before any frame is drawn
        raw2 = make_raw(t / "raw2", xml=lambda i: model_xml(i, move={"mount_main": "0 0 0.7"} if i == 43 else None))
        with Stub() as stub:
            expect(R.SceneMismatch, PR.render_run, raw2, t / "bad", "test", "clean", cfg, None, env_info())
            assert stub.log["make"] == [], "no env built, so no frame drawn"
        assert not list((t / "bad/libero_goal").glob("*"))
        # the same planted change in a TRAIN demo makes mount_main part of the fixture list (the list is read from train/val demos), and then the
        # Q20 check refuses it because mount_main is not a fixture root body of the env: a stop, not a silent pass
        raw3 = make_raw(t / "raw3", xml=lambda i: model_xml(i, move={"mount_main": "0 0 0.7"} if i == 3 else None))
        with Stub() as stub:
            e = expect(R.RenderCheckError, PR.render_run, raw3, t / "ok3", "val", "clean", cfg, None, env_info(), max_demos=1)
        assert "not fixture root bodies" in str(e), e
    print("perturbed conditions (stub simulator): camera/light/background build one env per (demo, type) from the drawn episode and close it; "
          "noise uses the base env with per-frame seeds independent of order; Q2 override applied except in the camera condition; "
          "scene mismatch stops before any env is built; perturbed train/val renders refused")


def synthetic_names(b):
    out = []
    for i in range(1, 8):
        out += [{"name": f"{b}_light_{i}", "category": "Light Conditions"}, {"name": f"{b}_table_{i}", "category": "Background Textures"},
                {"name": f"{b}_tb_{i}", "category": "Background Textures"},
                {"name": f"{b}_view_{i * 10}_0_100_1_0_initstate_0", "category": "Camera Viewpoints"},
                {"name": f"{b}_view_0_0_100_0_0_initstate_0_noise_{i * 6}", "category": "Sensor Noise"}]
    return out


def test_consistency_end_to_end():
    cl = {"libero_goal": [e for b in ("task_a", "task_b", "task_c") for e in synthetic_names(b)]}
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        raw = make_raw(t / "raw")
        cfgp = t / "cfg.json"
        cfgp.write_text(json.dumps(CFG))
        clp = t / "cl.json"
        clp.write_text(json.dumps(cl))
        with Stub():
            ra = PR.render_run(raw, t / "phaseA", "trainval", "clean", CFG, None, env_info())   # phase A: train+val, one root = the reference
        assert ra["files"]["task_a_demo"]["demos"] == [f"demo_{i}" for i in range(40)]
        with h5py.File(t / "phaseA/libero_goal/task_a_demo.hdf5", "r") as f:
            assert sorted(f["data"].keys(), key=R.demo_index) == [f"demo_{i}" for i in range(40)]
        orig_env = PR.collect_environment, PR.load_classification
        cur = {"env": env_info()}
        PR.collect_environment = lambda scripts_dir=None: cur["env"]
        try:
            # identical renderer -> passes, only demo_30 of each task is re-rendered
            with Stub() as stub:
                rc = PR.main(["check-consistency", "--raw", str(raw), "--config", str(cfgp), "--consistency-ref", str(t / "phaseA"), "--out", str(t / "c1")])
                assert rc == PR.RC_OK and stub.log["make"] == ["task_a", "task_b", "task_c"], (rc, stub.log["make"])
            rep = json.loads((t / "c1/consistency-report.json").read_text())
            assert rep["identical"] and all(r["demos"].keys() == {"demo_30"} for r in rep["pixels"].values())
            # one value of one pixel differs -> fails
            with Stub(shift=1) as stub:
                rc = PR.main(["check-consistency", "--raw", str(raw), "--config", str(cfgp), "--consistency-ref", str(t / "phaseA"), "--out", str(t / "c2")])
            assert rc == PR.RC_CONSISTENCY
            rep = json.loads((t / "c2/consistency-report.json").read_text())
            assert not rep["identical"] and "pixels differ" in rep["failure"]
            # tolerance end to end: with a looser fraction (the stub's frames are small) one value off by one per frame passes as
            # within_tolerance, the report names the verdict and the per-array flags; with the real constants it fails (c2 above)
            old_frac = R.FRACTION_TOL
            R.FRACTION_TOL = 1e-3
            try:
                with Stub(shift=1):
                    rc = PR.main(["check-consistency", "--raw", str(raw), "--config", str(cfgp), "--consistency-ref", str(t / "phaseA"),
                                  "--out", str(t / "c2b")])
            finally:
                R.FRACTION_TOL = old_frac
            rep = json.loads((t / "c2b/consistency-report.json").read_text())
            assert rc == PR.RC_OK and rep["verdict"] == "within_tolerance" and not rep["identical"] and rep["within_tolerance"], rep.get("failure")
            assert rep["tolerance"] == {"max_abs_diff": 1, "fraction": 1e-3}
            assert all(c["within_tolerance"] and c["max_abs_diff"] == 1 for r in rep["pixels"].values() for c in r["demos"]["demo_30"].values())
            assert json.loads((t / "c1/consistency-report.json").read_text())["verdict"] == "identical"
            assert json.loads((t / "c2/consistency-report.json").read_text())["verdict"] == "exceeds_tolerance"
            # a code-hash difference (phase B runs newer code than phase A) is recorded, not a stop, in strict mode
            shutil.copytree(t / "phaseA", t / "phaseA_code")
            rp = t / "phaseA_code/render-report.json"
            rj = json.loads(rp.read_text())
            rj["code_sha256"]["pf_rerender.py"] = "0" * 64
            rp.write_text(json.dumps(rj))
            with Stub():
                rc = PR.main(["check-consistency", "--raw", str(raw), "--config", str(cfgp), "--consistency-ref", str(t / "phaseA_code"),
                              "--out", str(t / "c2c")])
            rep = json.loads((t / "c2c/consistency-report.json").read_text())
            assert rc == PR.RC_OK and rep["verdict"] == "identical" and any("pf_rerender.py" in x for x in rep["code_differences_recorded"])
            # the config settings hash must match the reference's: a different config stops before any render
            rj["config_sha256"] = "f" * 64
            rp.write_text(json.dumps(rj))
            with Stub() as stub:
                rc = PR.main(["check-consistency", "--raw", str(raw), "--config", str(cfgp), "--consistency-ref", str(t / "phaseA_code"),
                              "--out", str(t / "c2d")])
            assert rc == PR.RC_CONSISTENCY and stub.log["make"] == []
            assert "config settings hash" in json.loads((t / "c2d/consistency-report.json").read_text())["failure"]
            # library version differs -> stops before rendering anything
            cur["env"] = env_info(lib_robosuite="1.4.1")
            with Stub() as stub:
                rc = PR.main(["check-consistency", "--raw", str(raw), "--config", str(cfgp), "--consistency-ref", str(t / "phaseA"), "--out", str(t / "c3")])
                assert rc == PR.RC_CONSISTENCY and stub.log["make"] == [], "a software difference must stop before any render"
            assert "robosuite" in json.loads((t / "c3/consistency-report.json").read_text())["failure"]
            # driver and GPU differ, pixels identical -> passes, difference recorded
            cur["env"] = env_info(hw_driver_version="999", hw_gpu_name="NVIDIA A10")
            with Stub():
                rc = PR.main(["check-consistency", "--raw", str(raw), "--config", str(cfgp), "--consistency-ref", str(t / "phaseA"), "--out", str(t / "c4")])
            assert rc == PR.RC_OK
            assert len(json.loads((t / "c4/consistency-report.json").read_text())["hardware_differences_recorded"]) == 2
            cur["env"] = env_info()
            # a test-split render: the check runs first; on failure no test demo is read and no env is built
            PR.load_classification = lambda p=None: cl
            test_argv = ["render", "--split", "test", "--condition", "clean", "--raw", str(raw), "--config", str(cfgp)]
            with Spy(raw) as spy, Stub(shift=1) as stub:
                rc = PR.main(test_argv + ["--out", str(t / "B1"), "--consistency-ref", str(t / "phaseA")])
            assert rc == PR.RC_CONSISTENCY and not (set(spy.log) & {f"demo_{i}" for i in range(40, 50)}), spy.log
            assert not list((t / "B1/libero_goal").glob("*")) if (t / "B1/libero_goal").exists() else True
            with Stub() as stub:
                rc = PR.main(test_argv + ["--out", str(t / "B2")])
                assert rc == PR.RC_USAGE and stub.log["make"] == []
            cur["env"] = env_info(mujoco_gl="osmesa")
            with Spy(raw) as spy, Stub() as stub:
                rc = PR.main(test_argv + ["--out", str(t / "B3"), "--consistency-ref", str(t / "phaseA")])
            assert rc == PR.RC_CONSISTENCY and stub.log["make"] == [] and not (set(spy.log) & {f"demo_{i}" for i in range(40, 50)})
            # an error inside the check (Q20 fails in the val re-render) also exits 5, before any test read
            cur["env"] = env_info()
            with Spy(raw) as spy, Stub(bad_z=True) as stub:
                rc = PR.main(test_argv + ["--out", str(t / "B5"), "--consistency-ref", str(t / "phaseA")])
            assert rc == PR.RC_CONSISTENCY and not (set(spy.log) & {f"demo_{i}" for i in range(40, 50)}), (rc, spy.log)
            assert len(stub.log["make"]) == 1, "only the first val env of the check was built; no test env"
            with Stub() as stub:
                rc = PR.main(["check-consistency", "--raw", str(raw), "--config", str(cfgp), "--consistency-ref", str(t / "missing_ref"),
                              "--out", str(t / "c5")])
            assert rc == PR.RC_CONSISTENCY and stub.log["make"] == []
            # a reference without a camera key: error inside the pixel comparison, still exit 5
            shutil.copytree(t / "phaseA", t / "phaseA_nokey")
            with h5py.File(t / "phaseA_nokey/libero_goal/task_b_demo.hdf5", "a") as f:
                del f["data"]["demo_30"]["obs"]["eye_in_hand_rgb"]
            with Stub():
                rc = PR.main(["check-consistency", "--raw", str(raw), "--config", str(cfgp), "--consistency-ref", str(t / "phaseA_nokey"),
                              "--out", str(t / "c6")])
            assert rc == PR.RC_CONSISTENCY
            # strict: a null bundle key on both sides is not "same" (reference report patched, since a main-run render refuses to write one)
            shutil.copytree(t / "phaseA", t / "phaseA_nullbundle")
            rj = json.loads((t / "phaseA_nullbundle/render-report.json").read_text())
            rj["environment"]["software"]["bundle_key"] = None
            (t / "phaseA_nullbundle/render-report.json").write_text(json.dumps(rj))
            cur["env"] = env_info(bundle_key=None)
            with Stub() as stub:
                rc = PR.main(["check-consistency", "--raw", str(raw), "--config", str(cfgp), "--consistency-ref", str(t / "phaseA_nullbundle"),
                              "--out", str(t / "c7")])
            assert rc == PR.RC_CONSISTENCY and stub.log["make"] == []
            cur["env"] = env_info()
            # perturbed test render: needs --expect-draws; a draws file that differs is refused before the check and before any test read
            rc = PR.main(["draws", "--raw", str(raw), "--config", str(cfgp), "--classification", str(clp), "--out", str(t / "draws_ok.json")])
            assert rc == PR.RC_OK
            dok = json.loads((t / "draws_ok.json").read_text())
            bad = json.loads(json.dumps(dok))
            k0 = "task_b_demo/demo_42/camera"
            other = [n["name"] for n in cl["libero_goal"] if n["name"].startswith("task_b_view_") and n["name"] != bad["draws"][k0]["name"]][0]
            bad["draws"][k0]["name"] = other
            (t / "draws_bad.json").write_text(json.dumps(bad))
            pert = ["render", "--split", "test", "--condition", "camera", "--raw", str(raw), "--config", str(cfgp), "--classification", str(clp)]
            with Stub() as stub:
                rc = PR.main(pert + ["--out", str(t / "P0"), "--consistency-ref", str(t / "phaseA")])
                assert rc == PR.RC_USAGE and stub.log["make"] == []
            with Spy(raw) as spy, Stub() as stub:
                rc = PR.main(pert + ["--out", str(t / "P1"), "--consistency-ref", str(t / "phaseA"), "--expect-draws", str(t / "draws_bad.json")])
            assert rc == PR.RC_CHECK and stub.log["make"] == [] and not (set(spy.log) & {f"demo_{i}" for i in range(40, 50)})
            with Stub() as stub:
                rc = PR.main(pert + ["--out", str(t / "P2"), "--consistency-ref", str(t / "phaseA"), "--expect-draws", str(t / "draws_ok.json")])
            assert rc == PR.RC_OK
            rp2 = json.loads((t / "P2/render-report.json").read_text())
            assert rp2["files"]["task_b_demo"]["per_demo"]["demo_42"]["episode"]["name"] == dok["draws"][k0]["name"]
            assert rp2["pilot"] is False and rp2["not_for_training"] is False
            # pilots on val: every perturbed condition, small, own root, marked, timed; clean pilot, no --max-demos, test split refused
            for cond in ("camera", "light", "background", "noise"):
                with Stub() as stub:
                    rc = PR.main(["render", "--split", "val", "--condition", cond, "--pilot", "--max-demos", "1", "--raw", str(raw),
                                  "--config", str(cfgp), "--classification", str(clp), "--out", str(t / ("pilot_" + cond))])
                assert rc == PR.RC_OK, cond
                pr = json.loads((t / ("pilot_" + cond) / "render-report.json").read_text())
                assert pr["pilot"] is True and pr["not_for_training"] is True and pr["split"] == "val"
                assert pr["timing"]["frames"] > 0 and pr["timing"]["seconds_per_frame"] is not None
                if cond == "noise":
                    assert pr["timing"]["noise_by_code"] and all(v["noise_seconds_per_frame"] is not None for v in pr["timing"]["noise_by_code"].values())
                assert pr["files"]["task_a_demo"]["demos"] == ["demo_30"] and pr["files"]["task_a_demo"]["test_demo_reads"]["arrays"] == []
            with Stub():
                for argv in (["--split", "val", "--condition", "camera", "--pilot"],                       # no --max-demos
                             ["--split", "val", "--condition", "clean", "--pilot", "--max-demos", "1"],
                             ["--split", "test", "--condition", "camera", "--pilot", "--max-demos", "1"]):
                    rc = PR.main(["render", *argv, "--raw", str(raw), "--config", str(cfgp), "--classification", str(clp), "--out", str(t / "pilot_bad")])
                    assert rc in (PR.RC_CHECK, PR.RC_USAGE), (argv, rc)
                # a pilot refuses a root that holds main-run output, and a main run refuses a pilot root
                rc = PR.main(["render", "--split", "val", "--condition", "camera", "--pilot", "--max-demos", "1", "--raw", str(raw),
                              "--config", str(cfgp), "--classification", str(clp), "--out", str(t / "phaseA")])
                assert rc == PR.RC_CHECK
                rc = PR.main(["render", "--split", "val", "--condition", "clean", "--raw", str(raw), "--config", str(cfgp),
                              "--out", str(t / "pilot_camera")])
                assert rc == PR.RC_CHECK
            cur["env"] = env_info()
            with Stub() as stub:
                rc = PR.main(test_argv + ["--out", str(t / "B4"), "--consistency-ref", str(t / "phaseA")])
            assert rc == PR.RC_OK
            rb = json.loads((t / "B4/render-report.json").read_text())
            assert rb["files"]["task_a_demo"]["demos"] == [f"demo_{i}" for i in range(40, 50)]
            assert stub.log["make"][:3] == ["task_a", "task_b", "task_c"] and len(stub.log["make"]) == 6, "consistency (3 envs) then test render (3 envs)"
            assert (t / "B4/consistency/consistency-report.json").exists()
            # main-run render refuses at startup if a strict provenance field is null; a pilot does not
            for key in ("image_digest", "bundle_key", "libero_source_fingerprint"):
                with Stub() as stub:
                    expect(R.RenderCheckError, PR.render_run, raw, t / ("nullA_" + key), "trainval", "clean", CFG, None, env_info(**{key: None}))
                    assert stub.log["make"] == [] and not list((t / ("nullA_" + key) / "libero_goal").glob("*"))
            with Stub() as stub:
                expect(R.RenderCheckError, PR.render_run, raw, t / "nullA_lib", "trainval", "clean", CFG, None, env_info(lib_mujoco=None))
                expect(R.RenderCheckError, PR.render_run, raw, t / "nullB", "test", "clean", CFG, None, env_info(image_digest=None))
                assert stub.log["make"] == []
                PR.render_run(raw, t / "nullpilot", "val", "camera", CFG, cl, env_info(image_digest=None), max_demos=1, pilot=True)
            cur["env"] = env_info(image_digest=None)
            with Stub() as stub:
                rc = PR.main(["render", "--split", "trainval", "--condition", "clean", "--raw", str(raw), "--config", str(cfgp),
                              "--out", str(t / "nullcli")])
            assert rc == PR.RC_CHECK and stub.log["make"] == []
            cur["env"] = env_info()
            # --demos: a val-only reference (the pf-val case) against a trainval render
            with Stub():
                PR.render_run(raw, t / "valonly", "val", "clean", CFG, None, env_info())
            rc = PR.main(["compare", "--ref-dir", str(t / "valonly"), "--new-dir", str(t / "phaseA"), "--out", str(t / "cmp_u.json"), "--report-only"])
            assert rc == PR.RC_OK and not json.loads((t / "cmp_u.json").read_text())["identical"]
            rc = PR.main(["compare", "--ref-dir", str(t / "valonly"), "--new-dir", str(t / "phaseA"), "--out", str(t / "cmp_v.json"),
                          "--report-only", "--demos", "val"])
            cv = json.loads((t / "cmp_v.json").read_text())
            assert rc == PR.RC_OK and cv["identical"] and cv["demos_compared"] == [f"demo_{i}" for i in range(30, 40)]
            with Stub(shift=1):
                PR.render_run(raw, t / "phaseA_s", "trainval", "clean", CFG, None, env_info())
            rc = PR.main(["compare", "--ref-dir", str(t / "valonly"), "--new-dir", str(t / "phaseA_s"), "--out", str(t / "cmp_w.json"),
                          "--report-only", "--demos", "val"])
            assert not json.loads((t / "cmp_w.json").read_text())["identical"], "a shifted pixel must still show with the filter"
            assert PR.parse_demos("demo_30, demo_31") == ["demo_30", "demo_31"] and PR.parse_demos(None) is None
            expect(ValueError, PR.parse_demos, "30")
            # report-only compare (what phase A does against the pf-val val frames): never fails on differences, lists them
            rc = PR.main(["compare", "--ref-dir", str(t / "phaseA"), "--new-dir", str(t / "phaseA"), "--out", str(t / "cmp1.json"), "--report-only"])
            assert rc == PR.RC_OK and json.loads((t / "cmp1.json").read_text())["identical"]
            with Stub(shift=1):
                PR.render_run(raw, t / "phaseA2", "trainval", "clean", CFG, None, env_info())
            rc = PR.main(["compare", "--ref-dir", str(t / "phaseA"), "--new-dir", str(t / "phaseA2"), "--out", str(t / "cmp2.json"), "--report-only"])
            assert rc == PR.RC_OK and not json.loads((t / "cmp2.json").read_text())["identical"]
            rc = PR.main(["compare", "--ref-dir", str(t / "phaseA"), "--new-dir", str(t / "phaseA2"), "--out", str(t / "cmp3.json")])
            assert rc == PR.RC_CONSISTENCY
            # draws command lists names only
            with Spy(raw) as spy:
                rc = PR.main(["draws", "--raw", str(raw), "--config", str(cfgp), "--classification", str(clp), "--out", str(t / "draws.json")])
            assert rc == PR.RC_OK and spy.log == []
            dj = json.loads((t / "draws.json").read_text())
            assert len(dj["draws"]) == 3 * 10 * 4 and dj["seed_draw"] == CFG["seed_draw"]
        finally:
            PR.collect_environment, PR.load_classification = orig_env
    print("consistency end to end (stub simulator): identical passes; one changed pixel fails; library difference stops before any render; "
          "driver/GPU difference recorded and pixels decide; test render runs the check first and reads no test demo when it fails; "
          "report-only compare never fails; draws command opens no demo group")


def test_load_config():
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "c.json"
        p.write_text(json.dumps({"seed_draw": 1, "seed_noise": 2, "resize": "area"}))
        c = PR.load_config(p)
        assert c["n_files"] == 10 and c["consistency_demo"] == "demo_30"
        y = Path(t) / "c.yaml"
        y.write_text("seed_draw: 1\nseed_noise: 2\nresize: area\n")
        assert PR.load_config(y)["seed_noise"] == 2
        for bad in ({"seed_draw": 1, "seed_noise": 2}, {"seed_draw": 1, "seed_noise": 2, "resize": "area", "typo": 3},
                    {"seed_draw": "1", "seed_noise": 2, "resize": "area"}, {"seed_draw": 1, "seed_noise": 2, "resize": "nearest"}):
            p.write_text(json.dumps(bad))
            expect(ValueError, PR.load_config, p)
    print("config: required keys, unknown keys, types and resize value checked; yaml and json both load")


def real_files_fixture_check():
    """Train/val scene comparison over the ten downloaded files: the fixture list of each file (differing_bodies over the 40 train and
    val demos) must equal `fixtures` in the pf-val render report, and the 10 val demos, treated as if they were the files to check,
    must pass scene_file_check against the 30 train demos with that list (a real-data run of the phase B check that touches no test demo)."""
    if not REAL_DIR.exists():
        print("real-file scene comparison SKIPPED: downloaded files not found at", REAL_DIR)
        return False
    ref = json.loads(VAL_REPORT.read_text())["files"]
    files = sorted(REAL_DIR.glob("*.hdf5"))
    assert len(files) == 10, len(files)
    print(f"{'file':58s} fixtures (ours)  == pf-val report")
    for p in files:
        with Spy(REAL_DIR) as spy:
            with R.RawFile(p, "val") as rf:
                train, val = rf.names("train"), rf.names("val")
                parsed = {d: R.parse_model_file(rf.model_file(d)) for d in train + val}
            fixtures, info = R.differing_bodies([parsed[d] for d in train + val])
            for d in train + val:
                R.reject_other_rotations(parsed[d], fixtures)
            rep = R.scene_file_check({d: parsed[d] for d in train}, {d: parsed[d] for d in val}, fixtures)
        assert not (set(spy.log) & {f"demo_{i}" for i in range(40, 50)}), "a test demo was opened"
        want = ref[p.stem]["fixtures"]
        assert fixtures == want, (p.stem, fixtures, want)
        assert ref[p.stem]["n_train_val_compared"] == len(train + val) == 40
        assert rep["ok"] and all(not v["outside_fixture_list"] for v in rep["per_demo"].values())
        print(f"{p.stem:58s} {fixtures}  == report: True  (40 demos; val-vs-train scene_file_check ok; opened only demo_0..39)")
    return True


def main():
    if "--real-only" in sys.argv:
        real_files_fixture_check()
        return
    test_load_config()
    test_apply_noise()
    test_draws()
    test_split_guard_synthetic()
    test_scene_file_check()
    test_provenance_and_frames()
    test_tolerance()
    test_render_contract_and_resume()
    test_first_failure_stops()
    test_camera_override_and_perturbed_conditions()
    test_consistency_end_to_end()
    test_real_files_guard_and_render()
    real_files_fixture_check()
    print("all ok")


if __name__ == "__main__":
    main()
