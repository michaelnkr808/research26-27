#!/usr/bin/env python3
"""Assert that every rung 1 setting matches the values fixed in PLAN.md. Fails loudly on any difference.

What it checks:
1. config.yaml against the PLAN values written out below (EXPECTED), independently of the scripts.
2. The commands run_eval.sh would execute: it runs "bash run_eval.sh --print-commands" and compares the
   server and client argv, their environment and the banner lines it will look for, against argv written
   out below. While client.enable_stall_recovery is pending, both variants are checked and run_eval.sh
   must refuse to build a run command.
3. With --wnm-root and --gn0-root: the defaults inside the pinned WNM-3D and GN0 sources that the plan
   relies on but no flag sets (the DiT step mask, seed 1140, the benchmark YAML, eval_remote.sh defaults).
4. With --order-file: the frozen episode order (count, uniqueness, sha256).

Usage:
  python check_settings.py [--config ../config.yaml] [--wnm-root DIR --gn0-root DIR] [--order-file FILE]
Exit 0 only if nothing differs.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
CONFIG = EXP_DIR / "config.yaml"
RUN_EVAL = HERE / "run_eval.sh"

PENDING = "pending"

# ------------------------------------------------------------------ PLAN values (item 1 of the brief)
EXPECTED = {
    "pins.wnm_3d": "7fe2d5f93a8dc4e2a5a2ec6fe174746928ecdcbb",
    "pins.gn0": "c927781",
    "pins.gn_bench_tools": "0ce8367",
    "checkpoint.repo": "TeleEmbodied/WNM-3D",
    "checkpoint.revision": "5b8c08dd52028ef4a29e7b87966f27582032fb4f",
    "checkpoint.include": "wnm_3d_stage3_release/**",
    "checkpoint.path": "checkpoints/wnm_3d_stage3_release",
    "data.gn_matrix.revision": "50a68253b5436f0595779cd3f91baf65228a0238",
    "data.gn_matrix.file": "InteriorGS/InteriorGS_test_unseen.parquet",
    "data.gn_matrix.local_dir": "data/datasets/GN_Matrix",
    "data.gn_matrix.sha256": "de4fd64a4e27ef552017eebf43aeb02992618b49c8800aa9bbc09dd30a0e1347",
    "data.interiorgs.revision": "334dfeea4e0241033b4e5de97c01bc7c9c080530",
    "data.interiorgs.local_dir": "data/scene_datasets/InteriorGS",
    "env.wnm_3d.huggingface_hub": "0.36.2",
    "env.wnm_3d.pyarrow": "23.0.1",
    "env.wnm_3d.torch": "2.8.0",
    "env.bae.torch": "2.4.0",
    "server.script": "scripts/inference/wnm_3d_server.sh",
    "server.cuda_devices": "0",
    "server.num_replicas": 1,
    "server.gpus_per_replica": 1,
    "server.base_port": 8000,
    "server.num_inference_steps": 16,
    "server.dit_step_mask": [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1],
    "server.enable_dit_cache": True,
    "server.enable_cfg": True,
    "server.cfg_scale": 5.0,
    "server.disable_torch_compile": True,
    "server.inference_seed": 1140,
    "server.history_sampling": "uniform",
    "server.env": {"HF_HUB_OFFLINE": "1"},
    "client.script": "scripts/evaluation/eval_remote.sh",
    "client.exp_config": "configs/gn_bench/interiorgs/test_unseen.yaml",
    "client.num_gpus": 1,
    "client.procs_per_gpu": 1,
    "client.server_host": "127.0.0.1",
    "client.server_port": 8000,
    "client.max_actions_per_query": 8,
    "client.stall_patience": 16,
    "client.remote_min_chunk_translation": 0.15,
    "client.remote_stop_eps": "1e-3",
    "client.remote_translation_frame": "chunk_start",
    "client.remote_max_translation": 0.0,
    "client.remote_max_yaw": 0.0,
    "client.action_format": "auto",
    "client.save_media": "none",
    "client.remote_send_state": False,
    "client.dagger": False,
    "client.wam_trace_mode": "none",
    "benchmark.max_episode_steps": 500,
    "benchmark.early_stop_steps": 500,
    "benchmark.stall_patience": 16,
    "benchmark.success_distance_m": 3.0,
    "benchmark.forward_step_size_m": 0.25,
    "benchmark.turn_angle_deg": 15,
    "benchmark.rgb": {"width": 480, "height": 360, "hfov_deg": 70},
    "episodes.split": "InteriorGS_test_unseen",
    "episodes.n_total": 5000,
    "episodes.order_seed": 20261121,
    "episodes.order_sha256": "25366209f38a242dbcec94bfed26092a9937dbf5adabbcd149f4558caed72728",
    "episodes.source_sha256": "270297ecd2f9a3f87a7a2f4dc974c0ccea4e5219ad6709a6e2888333015510e7",
    "episodes.id_format": "{scene}_{trajectory_id}",
    "target.split": "Unseen",
    "target.sr": 53.5,
    "target.spl": 46.6,
    "target.os": 62.2,
    "target.ne": 4.1,
    "target.tl": 10.2,
    "pass_rule.ci_level": 0.95,
    "pass_rule.wilson_z": 1.959963984540054,
    "breakdowns.visible.csv_column": "visible_primary",
    "breakdowns.visible.tolerance_column": "depth_tol_primary_m",
    "breakdowns.visible.tolerance_m": 0.15,
    "breakdowns.mirrored_scenes.n_scenes_expected": 39,
    "breakdowns.mirrored_scenes.n_test_unseen_episodes_expected": 233,
    "breakdowns.mirrored_scenes.source_sha256": "f7931ea18109e7f45f5eaff10ed6ca681dfbd5b149fbe1212ec40717920ce0a3",
    "compute.provider": "vast.ai",
    "compute.gpu": {"type": "H100 SXM", "count": 1, "on_demand": True},
    "paths.hf_token_file": "/dev/shm/hft",
}

# Keys the lead has not decided yet: allowed values.
OPEN_KEYS = {
    "client.enable_stall_recovery": "pending, true or false",
    "episodes.n_main": "pending or an integer from 1 to 5000",
    "episodes.timing_k": "pending or an integer from 1 to 5000",
}

EXPECTED_SERVER_ARGV = [
    "bash", "scripts/inference/wnm_3d_server.sh",
    "--model-path", "checkpoints/wnm_3d_stage3_release",
    "--cuda-devices", "0",
    "--num-replicas", "1",
    "--base-port", "8000",
    "--num-inference-steps", "16",
    "--enable-dit-cache",
    "--enable-cfg",
    "--cfg-scale", "5.0",
    "--disable-torch-compile",
]
EXPECTED_SERVER_ENV = ["HF_HUB_OFFLINE=1"]


def expected_client_argv(result_dir: str, stall_recovery: bool) -> list[str]:
    argv = [
        "bash", "scripts/evaluation/eval_remote.sh",
        "--exp-config", "configs/gn_bench/interiorgs/test_unseen.yaml",
        "--num-gpus", "1",
        "--result-dir", result_dir,
        "--max-actions-per-query", "8",
        "--stall-patience", "16",
        "--remote-min-chunk-translation", "0.15",
    ]
    if stall_recovery:
        argv.append("--enable-stall-recovery")
    return argv


EXPECTED_SERVER_BANNER = [
    "  Checkpoint     : checkpoints/wnm_3d_stage3_release",
    "  Replicas       : 1",
    "  CUDA pool      : 0",
    "  GPUs/replica   : 1",
    "  History mode   : uniform",
    "  Sampling       : 8/16",
    "  DiT cache      : true",
    "  CFG            : on (scale 5.0)",
    "  Torch compile  : disabled",
]


def expected_client_banner(stall_recovery: bool) -> list[str]:
    return [
        "Experiment config: configs/gn_bench/interiorgs/test_unseen.yaml",
        "Action format: auto",
        "Max actions per query: 8",
        "Stall patience: 16",
        f"Stall recovery: {'true' if stall_recovery else 'false'}",
        "Remote stop epsilon: 1e-3",
        "Remote minimum chunk translation: 0.15 m",
        "WAM trace mode: none",
        "DAgger: false",
        "Media output: none",
        "Evaluating episodes from 0 to end",
        "Total GN-Bench workers: 1",
    ]


# ------------------------------------------------------------------ helpers
_MISSING = object()


def get_path(cfg: dict, dotted: str):
    cur = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return _MISSING
        cur = cur[k]
    return cur


def same(a, b) -> bool:
    """Equal with types: 1 is not True, 5 is not 5.0 unless both are numbers of the same kind."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return type(a) is type(b) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def check_config(cfg: dict) -> list[str]:
    errs = []
    for key, want in EXPECTED.items():
        got = get_path(cfg, key)
        if got is _MISSING:
            errs.append(f"config: {key} missing (expected {want!r})")
        elif not same(got, want):
            errs.append(f"config: {key} = {got!r}, expected {want!r}")
    stall = get_path(cfg, "client.enable_stall_recovery")
    if not (stall == PENDING or isinstance(stall, bool)):
        errs.append(f"config: client.enable_stall_recovery = {stall!r}, allowed: {OPEN_KEYS['client.enable_stall_recovery']}")
    for key in ("episodes.n_main", "episodes.timing_k"):
        v = get_path(cfg, key)
        ok = v == PENDING or (isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 5000)
        if not ok:
            errs.append(f"config: {key} = {v!r}, allowed: {OPEN_KEYS[key]}")
    shards = get_path(cfg, "checkpoint.shard_sha256")
    if not (isinstance(shards, dict) and len(shards) == 6 and all(re.fullmatch(r"[0-9a-f]{64}", str(v)) for v in shards.values())):
        errs.append("config: checkpoint.shard_sha256 must list 6 shards with sha256 values")
    return errs


def parse_print_output(raw: bytes) -> dict[str, list[str]]:
    headers = {"SERVER", "SERVER_ENV", "SERVER_CWD", "SERVER_BANNER", "CLIENT", "CLIENT_ENV", "CLIENT_CWD",
               "CLIENT_BANNER", "END"}
    items = raw.decode("utf-8").split("\0")
    if items and items[-1] == "":
        items.pop()
    sections: dict[str, list[str]] = {}
    cur = None
    for it in items:
        if it in headers:
            if it in sections:
                raise ValueError(f"section {it} printed twice")
            sections[it] = []
            cur = it
        elif cur is None:
            raise ValueError(f"output before the first section: {it!r}")
        else:
            sections[cur].append(it)
    if "END" not in sections:
        raise ValueError("no END marker: run_eval.sh output incomplete")
    return sections


def check_commands(sections: dict[str, list[str]], stall_recovery: bool) -> list[str]:
    errs = []
    tag = f"[stall recovery {'on' if stall_recovery else 'off'}]"
    if sections.get("SERVER") != EXPECTED_SERVER_ARGV:
        errs.append(f"server argv {tag}: {sections.get('SERVER')} != {EXPECTED_SERVER_ARGV}")
    if sections.get("SERVER_ENV") != EXPECTED_SERVER_ENV:
        errs.append(f"server env {tag}: {sections.get('SERVER_ENV')} != {EXPECTED_SERVER_ENV}")
    client = sections.get("CLIENT") or []
    try:
        rd = client[client.index("--result-dir") + 1]
    except (ValueError, IndexError):
        rd = None
    if rd is None or not rd.endswith("/gn0"):
        errs.append(f"client argv {tag}: --result-dir missing or not <run>/gn0: {rd!r}")
    elif client != expected_client_argv(rd, stall_recovery):
        errs.append(f"client argv {tag}: {client} != {expected_client_argv(rd, stall_recovery)}")
    flags = [a for a in client if a.startswith("--")]
    if len(flags) != len(set(flags)):
        errs.append(f"client argv {tag}: a flag is passed twice: {flags}")
    env = sections.get("CLIENT_ENV") or []
    want_pp = f"PYTHONPATH={HERE / 'rung1_hook'}"
    if want_pp not in env:
        errs.append(f"client env {tag}: {want_pp} missing: {env}")
    for name in ("RUNG1_EPISODE_IDS", "RUNG1_EVENTS", "RUNG1_HOOK_STATUS", "RUNG1_LAUNCH_ID"):
        if not any(e.startswith(name + "=") for e in env):
            errs.append(f"client env {tag}: {name} missing")
    if sections.get("SERVER_BANNER") != EXPECTED_SERVER_BANNER:
        errs.append(f"server banner {tag}: {sections.get('SERVER_BANNER')} != {EXPECTED_SERVER_BANNER}")
    if sections.get("CLIENT_BANNER") != expected_client_banner(stall_recovery):
        errs.append(f"client banner {tag}: {sections.get('CLIENT_BANNER')} != {expected_client_banner(stall_recovery)}")
    return errs


def run_print(config: Path, stall_override: str | None, python: str = sys.executable) -> subprocess.CompletedProcess:
    cmd = ["bash", str(RUN_EVAL), "--print-commands"]
    if stall_override is not None:
        cmd += ["--stall-override-for-check", stall_override]
    env = dict(os.environ, RUNG1_PYTHON=python)
    return subprocess.run(cmd, capture_output=True, env=env)


def check_run_eval(cfg: dict, config: Path) -> list[str]:
    errs = []
    if Path(config).resolve() != CONFIG.resolve():
        errs.append(f"run_eval.sh always reads {CONFIG}; checking another config file is not supported")
        return errs
    stall = cfg["client"]["enable_stall_recovery"]
    if stall == PENDING:
        refused = run_print(config, None)
        if refused.returncode == 0 or b"pending" not in refused.stderr:
            errs.append("run_eval.sh built a client command although client.enable_stall_recovery is pending")
        variants = [True, False]
    else:
        variants = [bool(stall)]
    for v in variants:
        res = run_print(config, ("true" if v else "false") if stall == PENDING else None)
        if res.returncode != 0:
            errs.append(f"run_eval.sh --print-commands failed (exit {res.returncode}): {res.stderr.decode()[-500:]}")
            continue
        try:
            sections = parse_print_output(res.stdout)
        except ValueError as exc:
            errs.append(f"run_eval.sh output: {exc}")
            continue
        errs += check_commands(sections, v)
    return errs


# ------------------------------------------------------------------ pinned sources
SOURCE_LINES = {
    # WNM-3D 7fe2d5f
    ("wnm", "scripts/inference/wnm_3d_server.sh"): [
        'GPUS_PER_REPLICA="1"', 'HOST="127.0.0.1"', 'ENABLE_DIT_CACHE="true"', 'NUM_INFERENCE_STEPS="16"',
        "DIT_STEP_MASK=(1 1 1 0 0 0 1 0 0 0 1 0 0 1 1 1)", 'ENABLE_CFG="true"', 'CFG_SCALE="5.0"',
        'DISABLE_TORCH_COMPILE="true"', 'HISTORY_SAMPLING="uniform"', 'RETURN_SCALED_ACTION="false"',
        'RESIZE_INPUT_TO_CHECKPOINT_RESOLUTION="false"', 'NAV_ACTION_SCALE=""', 'MAX_CHUNK_SIZE=""',
    ],
    ("wnm", "gammanav/vln/model/wnm_3d/action_head/wan_flow_matching_action_tf.py"): [
        "self.seed = 1140",
    ],
    # GN0 c927781
    ("gn0", "scripts/evaluation/eval_remote.sh"): [
        'STALL_RECOVERY_FLAG=""', "MAX_ACTIONS_PER_QUERY=8", 'ACTION_FORMAT="auto"', 'REMOTE_SEND_STATE_FLAG=""',
        'WAM_TRACE_MODE="none"', 'DAGGER_FLAG=""', 'START_IDX="0"', 'END_IDX=""', "PROCS_PER_GPU=1",
        "STALL_PATIENCE=16", 'SERVER_HOST="127.0.0.1"', 'SERVER_PORT="8000"', 'SERVER_PORTS=""',
        'REMOTE_REQUEST_TIMEOUT="600"', 'REMOTE_CONNECT_TIMEOUT="600"', 'REMOTE_METADATA_TIMEOUT="2"',
        'REMOTE_STOP_EPS="1e-3"', 'REMOTE_MIN_CHUNK_TRANSLATION="0"', 'REMOTE_TRANSLATION_FRAME="chunk_start"',
        'REMOTE_MAX_TRANSLATION="0.0"', 'REMOTE_MAX_YAW="0.0"', 'SAVE_MEDIA="none"',
        "CUDA_VISIBLE_DEVICES=$gpu_id python -m gn0.evaluation.remote \\",
    ],
    ("gn0", "gn0/agents/remote.py"): ["chunk_stop_translation: float = 0.15,"],
    ("gn0", "gn0/evaluation/common.py"): [
        'return f"{ref_path.parent.name}_{ref_path.stem}"', "def select_eval_episodes(", "def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:",
        "def load_finished_episode_ids(log_dir: Path) -> set[str]:",
    ],
    ("gn0", "gn0/evaluation/remote.py"): [
        "from gn0.evaluation.common import (", "    atomic_write_json,", "    select_eval_episodes,",
        "global_episodes, resolved_end_idx, skipped, finished_count = select_eval_episodes(",
        "save_path = log_dir / f\"{results['id']}.json\"", "atomic_write_json(save_path, results)",
        "STALL_RECOVERY_STEPS = 8", "STALL_RECOVERY_BACKTRACK_METERS = 0.08", "STALL_RECOVERY_MAX_ATTEMPTS = 2",
    ],
}


def check_sources(wnm_root: Path, gn0_root: Path, cfg: dict) -> list[str]:
    errs = []
    roots = {"wnm": Path(wnm_root), "gn0": Path(gn0_root)}
    for (root, rel), lines in SOURCE_LINES.items():
        p = roots[root] / rel
        if not p.is_file():
            errs.append(f"source: {p} not found")
            continue
        have = {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()}
        for ln in lines:
            if ln.strip() not in have:
                errs.append(f"source: {p}: line not found: {ln!r}")
    yml = Path(gn0_root) / cfg["client"]["exp_config"]
    if not yml.is_file():
        errs.append(f"source: {yml} not found")
        return errs
    y = yaml.safe_load(yml.read_text(encoding="utf-8"))
    b = cfg["benchmark"]
    t = y["TASK_CONFIG"]
    checks = {
        "EVAL.STALL_PATIENCE": (y["EVAL"]["STALL_PATIENCE"], b["stall_patience"]),
        "EVAL.EARLY_STOP_STEPS": (y["EVAL"]["EARLY_STOP_STEPS"], b["early_stop_steps"]),
        "ENVIRONMENT.MAX_EPISODE_STEPS": (t["ENVIRONMENT"]["MAX_EPISODE_STEPS"], b["max_episode_steps"]),
        "TASK.SUCCESS_DISTANCE": (t["TASK"]["SUCCESS_DISTANCE"], b["success_distance_m"]),
        "TASK.SUCCESS.SUCCESS_DISTANCE": (t["TASK"]["SUCCESS"]["SUCCESS_DISTANCE"], b["success_distance_m"]),
        "TASK.SPL.SUCCESS_DISTANCE": (t["TASK"]["SPL"]["SUCCESS_DISTANCE"], b["success_distance_m"]),
        "TASK.ORACLE_SUCCESS.SUCCESS_DISTANCE": (t["TASK"]["ORACLE_SUCCESS"]["SUCCESS_DISTANCE"], b["success_distance_m"]),
        "SIMULATOR.FORWARD_STEP_SIZE": (t["SIMULATOR"]["FORWARD_STEP_SIZE"], b["forward_step_size_m"]),
        "SIMULATOR.TURN_ANGLE": (t["SIMULATOR"]["TURN_ANGLE"], b["turn_angle_deg"]),
        "RGB_SENSOR.WIDTH": (t["SIMULATOR"]["RGB_SENSOR"]["WIDTH"], b["rgb"]["width"]),
        "RGB_SENSOR.HEIGHT": (t["SIMULATOR"]["RGB_SENSOR"]["HEIGHT"], b["rgb"]["height"]),
        "RGB_SENSOR.HFOV": (t["SIMULATOR"]["RGB_SENSOR"]["HFOV"], b["rgb"]["hfov_deg"]),
        "DATASET.DATA_PATH": (t["DATASET"]["DATA_PATH"],
                              f"{cfg['data']['gn_matrix']['local_dir']}/{cfg['data']['gn_matrix']['file']}"),
        "DATASET.SCENES_DIR": (t["DATASET"]["SCENES_DIR"], cfg["data"]["interiorgs"]["local_dir"]),
    }
    for name, (got, want) in checks.items():
        if got != want:
            errs.append(f"source: {yml}: {name} = {got!r}, expected {want!r}")
    return errs


SETUP_SH = HERE / "setup.sh"
SETUP_VARS = {
    "WNM3D_COMMIT": "pins.wnm_3d",
    "GN0_COMMIT": "pins.gn0",
    "GN_BENCH_TOOLS_COMMIT": "pins.gn_bench_tools",
    "CKPT_REPO": "checkpoint.repo",
    "CKPT_REVISION": "checkpoint.revision",
    "CKPT_INCLUDE": "checkpoint.include",
    "TOKENIZER_REPO": "tokenizer.repo",
    "TOKENIZER_REVISION": "tokenizer.revision",
    "GN_MATRIX_REPO": "data.gn_matrix.repo",
    "GN_MATRIX_REVISION": "data.gn_matrix.revision",
    "GN_MATRIX_FILE": "data.gn_matrix.file",
    "GN_MATRIX_SHA256": "data.gn_matrix.sha256",
    "INTERIORGS_REPO": "data.interiorgs.repo",
    "INTERIORGS_REVISION": "data.interiorgs.revision",
    "N_SCENES_EXPECTED": "data.interiorgs.n_scenes_expected",
    "WNM_TORCH": "env.wnm_3d.torch",
    "WNM_TORCHVISION": "env.wnm_3d.torchvision",
    "WNM_TORCHAUDIO": "env.wnm_3d.torchaudio",
    "WNM_TORCH_INDEX": "env.wnm_3d.torch_index",
    "WNM_CUDA_TOOLKIT": "env.wnm_3d.cuda_toolkit",
    "WNM_HF_HUB": "env.wnm_3d.huggingface_hub",
    "WNM_PYARROW": "env.wnm_3d.pyarrow",
    "BAE_TORCH": "env.bae.torch",
    "BAE_TORCHVISION": "env.bae.torchvision",
    "BAE_TORCHAUDIO": "env.bae.torchaudio",
    "BAE_TORCH_INDEX": "env.bae.torch_index",
    "BAE_CUDA_TOOLKIT": "env.bae.cuda_toolkit",
    "GSPLAT_INDEX": "env.bae.gsplat_index",
}


def check_setup_script(cfg: dict, setup_sh: Path = SETUP_SH) -> list[str]:
    """The pins hard-coded at the top of setup.sh must equal config.yaml."""
    errs = []
    text = Path(setup_sh).read_text(encoding="utf-8")
    found = dict(re.findall(r'^([A-Z0-9_]+)="([^"]*)"$', text, flags=re.M))
    for var, key in SETUP_VARS.items():
        want = get_path(cfg, key)
        if var not in found:
            errs.append(f"setup.sh: {var} not set")
        elif want is _MISSING or found[var] != str(want):
            errs.append(f"setup.sh: {var}={found[var]!r} but config {key} = {want!r}")
    return errs


def check_order(order_file: Path, cfg: dict) -> list[str]:
    errs = []
    p = Path(order_file)
    if not p.is_file():
        return [f"order: {p} not found"]
    data = p.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != cfg["episodes"]["order_sha256"]:
        errs.append(f"order: sha256 {sha} != config {cfg['episodes']['order_sha256']}")
    ids = data.decode("utf-8").splitlines()
    if len(ids) != cfg["episodes"]["n_total"] or len(set(ids)) != len(ids):
        errs.append(f"order: {len(ids)} lines, {len(set(ids))} unique; expected {cfg['episodes']['n_total']} unique")
    return errs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--wnm-root", type=Path)
    ap.add_argument("--gn0-root", type=Path)
    ap.add_argument("--order-file", type=Path)
    a = ap.parse_args(argv)
    if (a.wnm_root is None) != (a.gn0_root is None):
        ap.error("give both --wnm-root and --gn0-root, or neither")

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    errs = check_config(cfg)
    n_checked = len(EXPECTED) + len(OPEN_KEYS) + 1
    errs += check_run_eval(cfg, a.config)
    errs += check_setup_script(cfg)
    parts = ["config.yaml", "run_eval.sh commands", "setup.sh pins"]
    if a.wnm_root is not None:
        errs += check_sources(a.wnm_root, a.gn0_root, cfg)
        parts.append("pinned sources")
    if a.order_file is not None:
        errs += check_order(a.order_file, cfg)
        parts.append("episode order")

    pending = [k for k in OPEN_KEYS if get_path(cfg, k) == PENDING]
    if errs:
        for e in errs:
            print(f"MISMATCH: {e}", file=sys.stderr)
        print(f"FAILED: {len(errs)} difference(s) from PLAN.md", file=sys.stderr)
        return 1
    print(f"OK: {n_checked} config values, " + ", ".join(parts) + " match PLAN.md")
    if pending:
        print("still pending the lead's answer (no run can start): " + ", ".join(pending))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
