"""Unit tests for check_settings.py, build_commands.py and the rung1 episode filter hook."""
import copy
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

import build_commands as B
import check_settings as C

HERE = Path(__file__).resolve().parent
CFG = yaml.safe_load((HERE.parent / "config.yaml").read_text())
sys.path.insert(0, str(HERE / "rung1_hook"))
import rung1_filter as F  # noqa: E402


# ---------------------------------------------------------------- config values
def test_current_config_matches_plan():
    assert C.check_config(CFG) == []


@pytest.mark.parametrize("key,value", [
    ("server.cfg_scale", 4.0),
    ("server.cfg_scale", 5),                       # int instead of float: types must match
    ("server.num_inference_steps", 8),
    ("server.dit_step_mask", [1] * 16),
    ("server.enable_dit_cache", False),
    ("server.disable_torch_compile", False),
    ("client.remote_min_chunk_translation", 0.0),  # the released eval_remote.sh value, not the paper's
    ("client.max_actions_per_query", 4),
    ("client.stall_patience", 1),
    ("benchmark.max_episode_steps", 400),
    ("benchmark.success_distance_m", 2.5),
    ("checkpoint.revision", "main"),
    ("episodes.order_seed", 1),
    ("target.sr", 53.0),
    ("breakdowns.visible.tolerance_m", 0.3),
    ("server.inference_seed", 42),
])
def test_any_changed_value_fails(key, value):
    cfg = copy.deepcopy(CFG)
    cur = cfg
    *parents, last = key.split(".")
    for p in parents:
        cur = cur[p]
    cur[last] = value
    errs = C.check_config(cfg)
    assert errs and any(key in e for e in errs)


def test_missing_key_fails():
    cfg = copy.deepcopy(CFG)
    del cfg["server"]["enable_cfg"]
    assert any("server.enable_cfg missing" in e for e in C.check_config(cfg))


def test_decided_values_recorded():
    """PLAN amendments: stall recovery on, K = 30 (2026-11-21); N = 1000 (2026-11-22)."""
    assert CFG["client"]["enable_stall_recovery"] is True
    assert CFG["episodes"]["timing_k"] == 30
    assert CFG["episodes"]["n_main"] == 1000


@pytest.mark.parametrize("key,value", [
    ("client.enable_stall_recovery", False),
    ("client.enable_stall_recovery", "pending"),
    ("client.enable_stall_recovery", 1),
    ("episodes.timing_k", 40),
    ("episodes.timing_k", "pending"),
])
def test_decided_values_cannot_change(key, value):
    cfg = copy.deepcopy(CFG)
    sec, k = key.split(".")
    cfg[sec][k] = value
    assert any(key in e for e in C.check_config(cfg))


@pytest.mark.parametrize("value,ok", [("pending", True), (1000, True), (5000, True), (0, False), (5001, False), (True, False)])
def test_episode_count_open_values(value, ok):
    cfg = copy.deepcopy(CFG)
    cfg["episodes"]["n_main"] = value
    assert (C.check_config(cfg) == []) is ok


# ---------------------------------------------------------------- commands
def _sections(stall):
    rd = "/workspace/rung1/runs/x/gn0"
    return {
        "SERVER": list(C.EXPECTED_SERVER_ARGV),
        "SERVER_ENV": list(C.EXPECTED_SERVER_ENV),
        "SERVER_CWD": ["/workspace/WNM-3D"],
        "SERVER_BANNER": list(C.EXPECTED_SERVER_BANNER),
        "CLIENT": C.expected_client_argv(rd, stall),
        "CLIENT_ENV": [f"PYTHONPATH={HERE / 'rung1_hook'}", "RUNG1_EPISODE_IDS=a", "RUNG1_EVENTS=b",
                       "RUNG1_HOOK_STATUS=c", "RUNG1_LAUNCH_ID=d"],
        "CLIENT_CWD": ["/workspace/GN0"],
        "CLIENT_BANNER": C.expected_client_banner(stall),
        "END": [],
    }


@pytest.mark.parametrize("stall", [True, False])
def test_expected_sections_pass(stall):
    assert C.check_commands(_sections(stall), stall) == []


def test_extra_client_flag_fails():
    s = _sections(False)
    s["CLIENT"] += ["--remote-max-yaw", "0.1"]
    assert C.check_commands(s, False)


def test_stall_flag_must_match_decision():
    assert C.check_commands(_sections(False), True)
    assert C.check_commands(_sections(True), False)


def test_server_flag_change_fails():
    s = _sections(False)
    s["SERVER"] = [a if a != "5.0" else "4.0" for a in s["SERVER"]]
    assert C.check_commands(s, False)


def test_missing_hook_env_fails():
    s = _sections(False)
    s["CLIENT_ENV"] = s["CLIENT_ENV"][1:]
    assert any("PYTHONPATH" in e for e in C.check_commands(s, False))


def test_banner_change_fails():
    s = _sections(False)
    s["CLIENT_BANNER"][6] = "Remote minimum chunk translation: 0 m"
    assert C.check_commands(s, False)


def test_parse_print_output():
    raw = b"SERVER\0bash\0x\0END\0"
    assert C.parse_print_output(raw) == {"SERVER": ["bash", "x"], "END": []}
    with pytest.raises(ValueError):
        C.parse_print_output(b"SERVER\0bash\0")
    with pytest.raises(ValueError):
        C.parse_print_output(b"bash\0END\0")


def test_run_eval_print_commands_match_plan():
    """The real run_eval.sh, through bash: both stall variants, and refusal while pending."""
    assert C.check_run_eval(CFG, C.CONFIG) == []


def test_run_eval_refuses_override_outside_print_mode():
    res = subprocess.run(["bash", str(C.RUN_EVAL), "--run-name", "x", "--stall-override-for-check", "true"],
                         capture_output=True, env=dict(os.environ, RUNG1_PYTHON=sys.executable))
    assert res.returncode != 0 and b"only allowed with --print-commands" in res.stderr


def test_build_commands_pending_and_override_rules(tmp_path):
    pend = copy.deepcopy(CFG)
    pend["client"]["enable_stall_recovery"] = "pending"
    with pytest.raises(B.PendingDecision):
        B.resolve_stall_recovery(pend)
    assert B.resolve_stall_recovery(pend, True) is True
    assert B.resolve_stall_recovery(CFG) is True                       # decided: on
    cfg = copy.deepcopy(CFG)
    cfg["client"]["enable_stall_recovery"] = False
    assert B.resolve_stall_recovery(cfg) is False
    with pytest.raises(ValueError):
        B.resolve_stall_recovery(cfg, True)
    with pytest.raises(SystemExit):
        B.main(["--emit", "client", "--result-dir", "/x/gn0", "--stall-override", "true"])  # no --for-check
    p = tmp_path / "pending.yaml"
    p.write_text(yaml.safe_dump(pend))
    assert B.main(["--config", str(p), "--emit", "client", "--result-dir", "/x/gn0"]) == 2   # pending
    assert B.main(["--emit", "client", "--result-dir", "/x/gn0"]) == 0                       # decided


def test_fmt_num():
    assert B.fmt_num(5.0) == "5.0" and B.fmt_num(0.15) == "0.15" and B.fmt_num(16) == "16"
    with pytest.raises(TypeError):
        B.fmt_num(True)


def test_committed_count_rules():
    assert B.committed_count_keys(CFG, 30) == ["episodes.timing_k"]   # K = 30 committed
    with pytest.raises(B.PendingDecision):          # n_main pending, 40 committed nowhere
        B.committed_count_keys(CFG, 40)
    pend = copy.deepcopy(CFG)
    pend["episodes"]["timing_k"] = "pending"
    with pytest.raises(B.PendingDecision):          # both pending
        B.committed_count_keys(pend, 30)
    cfg = copy.deepcopy(CFG)
    cfg["episodes"]["timing_k"] = 40
    assert B.committed_count_keys(cfg, 40) == ["episodes.timing_k"]
    with pytest.raises(B.PendingDecision):          # not a committed number
        B.committed_count_keys(cfg, 41)
    cfg["episodes"]["n_main"] = 1000
    assert B.committed_count_keys(cfg, 1000) == ["episodes.n_main"]
    assert B.resolve_count(cfg, "timing") == 40 and B.resolve_count(cfg, "main") == 1000
    assert B.resolve_count(CFG, "main") == 1000                      # real config: N = 1000
    pend_main = copy.deepcopy(CFG)
    pend_main["episodes"]["n_main"] = "pending"
    with pytest.raises(B.PendingDecision):
        B.resolve_count(pend_main, "main")
    assert B.main(["--emit", "check-n", "--n", "1000"]) == 0        # real config: N = 1000
    assert B.main(["--emit", "check-n", "--n", "40"]) == 2          # real config: 40 not committed
    assert B.main(["--emit", "check-n", "--n", "30"]) == 0          # real config: K = 30


def test_run_eval_refuses_uncommitted_n():
    """Run mode stops before anything starts when --n is not a committed count (both pending in the copy).

    The stall-recovery decision is set in a copy of the folder so that the count gate is what refuses.
    """
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        exp = Path(d) / "exp"
        shutil.copytree(HERE.parent, exp, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "results"))
        cfg = copy.deepcopy(CFG)
        cfg["episodes"]["timing_k"] = "pending"
        (exp / "config.yaml").write_text(yaml.safe_dump(cfg))
        env = dict(os.environ, RUNG1_PYTHON=sys.executable, RUNG1_ROOT=str(Path(d) / "root"))
        cmd = ["bash", str(exp / "src" / "run_eval.sh"), "--run-name", "x", "--n", "40", "--repo-commit", "abc"]
        res = subprocess.run(cmd, capture_output=True, env=env)
        assert res.returncode != 0 and b"not a committed episode count" in res.stderr
        cfg["episodes"]["timing_k"] = 40
        (exp / "config.yaml").write_text(yaml.safe_dump(cfg))
        if sys.platform != "linux":   # off Linux the next gate (platform) stops it; on the machine it would really start
            res = subprocess.run(cmd, capture_output=True, env=env)
            assert b"--n 40 matches episodes.timing_k" in res.stdout      # passes the count gate
            assert b"needs the rented Linux machine" in res.stderr
            assert not (Path(d) / "root").exists()                      # nothing was started or written
        res = subprocess.run(cmd[:5] + ["41"] + cmd[6:], capture_output=True, env=env)
        assert res.returncode != 0 and b"not a committed episode count" in res.stderr


def test_meta_refuses_changed_settings(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["client"]["enable_stall_recovery"] = True
    cfg["episodes"]["timing_k"] = 10
    cfg["episodes"]["n_main"] = 20
    m1 = B.build_meta(cfg, b"a", "/r/gn0", "run", 10, "abc", "L1", True)
    prev = tmp_path / "run_meta.jsonl"
    prev.write_text(__import__("json").dumps(m1) + "\n")
    B.check_previous(B.build_meta(cfg, b"a", "/r/gn0", "run", 10, "abc", "L2", True), prev)  # same: fine
    with pytest.raises(SystemExit):
        B.check_previous(B.build_meta(cfg, b"a", "/r/gn0", "run", 20, "abc", "L2", True), prev)
    with pytest.raises(SystemExit):
        B.check_previous(B.build_meta(cfg, b"b", "/r/gn0", "run", 10, "abc", "L2", True), prev)


# ---------------------------------------------------------------- pinned sources and order
def _fake_sources(tmp_path):
    wnm, gn0 = tmp_path / "wnm", tmp_path / "gn0"
    for (root, rel), lines in C.SOURCE_LINES.items():
        p = (wnm if root == "wnm" else gn0) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + "\n")
    y = {"EVAL": {"STALL_PATIENCE": 16, "EARLY_STOP_STEPS": 500},
         "TASK_CONFIG": {"ENVIRONMENT": {"MAX_EPISODE_STEPS": 500},
                         "SIMULATOR": {"FORWARD_STEP_SIZE": 0.25, "TURN_ANGLE": 15,
                                       "RGB_SENSOR": {"WIDTH": 480, "HEIGHT": 360, "HFOV": 70}},
                         "TASK": {"SUCCESS_DISTANCE": 3.0, "SUCCESS": {"SUCCESS_DISTANCE": 3.0},
                                  "SPL": {"SUCCESS_DISTANCE": 3.0}, "ORACLE_SUCCESS": {"SUCCESS_DISTANCE": 3.0}},
                         "DATASET": {"DATA_PATH": "data/datasets/GN_Matrix/InteriorGS/InteriorGS_test_unseen.parquet",
                                     "SCENES_DIR": "data/scene_datasets/InteriorGS"}}}
    yp = gn0 / CFG["client"]["exp_config"]
    yp.parent.mkdir(parents=True, exist_ok=True)
    yp.write_text(yaml.safe_dump(y))
    return wnm, gn0, yp, y


def test_sources_pass_and_fail(tmp_path):
    wnm, gn0, yp, y = _fake_sources(tmp_path)
    assert C.check_sources(wnm, gn0, CFG) == []
    sh = wnm / "scripts/inference/wnm_3d_server.sh"
    sh.write_text(sh.read_text().replace("DIT_STEP_MASK=(1 1 1 0 0 0 1 0 0 0 1 0 0 1 1 1)", "DIT_STEP_MASK=(1)"))
    assert any("DIT_STEP_MASK" in e for e in C.check_sources(wnm, gn0, CFG))
    wnm, gn0, yp, y = _fake_sources(tmp_path / "b")
    y["TASK_CONFIG"]["ENVIRONMENT"]["MAX_EPISODE_STEPS"] = 400
    yp.write_text(yaml.safe_dump(y))
    assert any("MAX_EPISODE_STEPS" in e for e in C.check_sources(wnm, gn0, CFG))


@pytest.mark.skipif(not os.environ.get("RUNG1_PINNED_WNM") or not os.environ.get("RUNG1_PINNED_GN0"),
                    reason="set RUNG1_PINNED_WNM and RUNG1_PINNED_GN0 to check the real pinned copies")
def test_real_pinned_sources():
    assert C.check_sources(Path(os.environ["RUNG1_PINNED_WNM"]), Path(os.environ["RUNG1_PINNED_GN0"]), CFG) == []


def test_setup_script_pins_match_config(tmp_path):
    assert C.check_setup_script(CFG) == []
    bad = tmp_path / "setup.sh"
    bad.write_text(C.SETUP_SH.read_text().replace('WNM_HF_HUB="0.36.2"', 'WNM_HF_HUB="2.4.0"'))
    assert any("WNM_HF_HUB" in e for e in C.check_setup_script(CFG, bad))
    bad.write_text(C.SETUP_SH.read_text().replace('CKPT_REVISION="', 'XCKPT_REVISION="'))
    assert any("CKPT_REVISION not set" in e for e in C.check_setup_script(CFG, bad))


def test_order_check(tmp_path):
    p = tmp_path / "order.txt"
    p.write_text("a\nb\n")
    errs = C.check_order(p, CFG)
    assert any("sha256" in e for e in errs) and any("lines" in e for e in errs)


# ---------------------------------------------------------------- episode filter hook
class _Ep:
    def __init__(self, scene, traj):
        self.ref_json = f"{scene}/{traj}.json"


def _id(ref_json):
    p = Path(ref_json)
    return f"{p.parent.name}_{p.stem}"


def test_select_in_order_keeps_frozen_order_and_skips_finished():
    eps = [_Ep("s1", 1), _Ep("s1", 2), _Ep("s2", 3), _Ep("s2", 4)]
    pending = [e for e in eps if _id(e.ref_json) != "s2_3"]
    sel, st = F.select_in_order(eps, pending, ["s2_4", "s2_3", "s1_1"], {"s2_3"}, _id)
    assert [_id(e.ref_json) for e in sel] == ["s2_4", "s1_1"]
    assert st == {"n_requested": 3, "n_dataset": 4, "n_already_finished": 1, "n_selected_now": 2}


def test_select_in_order_rejects_unknown_and_foreign():
    eps = [_Ep("s1", 1), _Ep("s1", 2)]
    with pytest.raises(F.FilterError, match="not in the dataset"):
        F.select_in_order(eps, eps, ["s1_9"], set(), _id)
    with pytest.raises(F.FilterError, match="not requested"):
        F.select_in_order(eps, eps[1:], ["s1_2"], {"s1_1"}, _id)


def test_install_wraps_gn0_common(tmp_path, monkeypatch):
    calls = {}
    common = types.ModuleType("gn0.evaluation.common")
    common.get_episode_id = _id
    common.load_finished_episode_ids = lambda log_dir: set()

    def select_eval_episodes(ds, log_dir, s, e):
        calls["select"] = (s, e)
        return list(ds.episodes), len(ds.episodes), 0, 0

    def atomic_write_json(path, payload):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(__import__("json").dumps(payload))

    common.select_eval_episodes = select_eval_episodes
    common.atomic_write_json = atomic_write_json
    pkg = types.ModuleType("gn0"); ev = types.ModuleType("gn0.evaluation")
    pkg.evaluation = ev; ev.common = common
    monkeypatch.setitem(sys.modules, "gn0", pkg)
    monkeypatch.setitem(sys.modules, "gn0.evaluation", ev)
    monkeypatch.setitem(sys.modules, "gn0.evaluation.common", common)

    ids = tmp_path / "ids.txt"; ids.write_text("s2_2\ns1_1\n")
    monkeypatch.setenv("RUNG1_EPISODE_IDS", str(ids))
    monkeypatch.setenv("RUNG1_EVENTS", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("RUNG1_HOOK_STATUS", str(tmp_path / "status.json"))
    monkeypatch.setenv("RUNG1_LAUNCH_ID", "L1")
    F.install()

    ds = types.SimpleNamespace(episodes=[_Ep("s1", 1), _Ep("s2", 2), _Ep("s3", 3)])
    log_dir = tmp_path / "gn0" / "log"
    sel, end, skipped, fin = common.select_eval_episodes(ds, log_dir, 0, None)
    assert [_id(e.ref_json) for e in sel] == ["s2_2", "s1_1"] and skipped == 0
    assert (tmp_path / "status.json").exists()
    with pytest.raises(F.FilterError):
        common.select_eval_episodes(ds, log_dir, 5, None)
    common.select_eval_episodes(ds, log_dir, 0, None)
    common.atomic_write_json(log_dir / "s2_2.json", {"id": "s2_2", "status": "complete", "success": 1.0, "stop": "x"})
    common.atomic_write_json(tmp_path / "other.json", {"id": "zz", "status": "complete"})   # not in log/: ignored
    common.atomic_write_json(log_dir / "s1_1.json", {"id": "s1_1", "status": "complete", "success": 0.0})
    lines = [__import__("json").loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [l["id"] for l in lines] == ["s2_2", "s1_1"]
    assert [l["first_in_launch"] for l in lines] == [True, False]
    assert all(l["wall_s"] >= 0 and l["launch_id"] == "L1" for l in lines)
    assert lines[0]["success"] == 1.0 and (log_dir / "s1_1.json").exists()


def test_sitecustomize_exits_when_filter_fails(tmp_path):
    """Without gn0 installed the filter cannot attach: the process must stop with code 97, not run on."""
    env = dict(os.environ, PYTHONPATH=str(HERE / "rung1_hook"), RUNG1_EPISODE_IDS=str(tmp_path / "none.txt"),
               RUNG1_EVENTS=str(tmp_path / "e"), RUNG1_HOOK_STATUS=str(tmp_path / "s"))
    res = subprocess.run([sys.executable, "-c", "print('ran')"], capture_output=True, env=env)
    assert res.returncode == 97 and b"ran" not in res.stdout
    env.pop("RUNG1_EPISODE_IDS")
    res = subprocess.run([sys.executable, "-c", "print('ran')"], capture_output=True, env=env)
    assert res.returncode == 0 and b"ran" in res.stdout


def test_append_line_numpy_scalars(tmp_path):
    import importlib, json as _json, sys as _sys
    import numpy as np
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).parent / "rung1_hook"))
    rung1_filter = importlib.import_module("rung1_filter")
    p = tmp_path / "ev.jsonl"
    rung1_filter.append_line(p, {"spl": np.float32(0.5), "success": np.float64(1.0), "n": np.int64(3)})
    assert _json.loads(p.read_text()) == {"spl": 0.5, "success": 1.0, "n": 3}
