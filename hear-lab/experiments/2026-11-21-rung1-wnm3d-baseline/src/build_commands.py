#!/usr/bin/env python3
"""Build the exact server and client commands for rung 1 from config.yaml.

run_eval.sh executes what this prints and adds nothing of its own, so check_settings.py can test the
commands by asking run_eval.sh to print them (--print-commands).

  python build_commands.py --emit server
  python build_commands.py --emit client --result-dir DIR
  python build_commands.py --emit server-banner | client-banner
  python build_commands.py --emit meta --result-dir DIR --run-name NAME --n N --repo-commit SHA --launch-id ID [--previous FILE]

Command arrays and banner lines are printed NUL-separated. While client.enable_stall_recovery is
"pending", every emit that depends on it fails, unless --stall-override is given together with
--for-check (only used to print commands for checking, never to run).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
CONFIG = EXP_DIR / "config.yaml"

PENDING = "pending"


class PendingDecision(Exception):
    pass


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fmt_num(x) -> str:
    """Numbers as the shell scripts expect them: 5.0 -> '5.0', 0.15 -> '0.15', 16 -> '16'."""
    if isinstance(x, bool):
        raise TypeError("bool is not a number here")
    return repr(float(x)) if isinstance(x, float) else str(int(x))


def resolve_stall_recovery(cfg: dict, override: bool | None = None) -> bool:
    value = cfg["client"]["enable_stall_recovery"]
    if isinstance(value, bool):
        if override is not None and override != value:
            raise ValueError("--stall-override differs from the decided config value")
        return value
    if value == PENDING:
        if override is None:
            raise PendingDecision(
                "client.enable_stall_recovery is pending the lead's answer; no run can start"
            )
        return bool(override)
    raise ValueError(f"client.enable_stall_recovery must be true, false or 'pending', got {value!r}")


def server_argv(cfg: dict) -> list[str]:
    s = cfg["server"]
    argv = [
        "bash", s["script"],
        "--model-path", cfg["checkpoint"]["path"],
        "--cuda-devices", str(s["cuda_devices"]),
        "--num-replicas", fmt_num(s["num_replicas"]),
        "--base-port", fmt_num(s["base_port"]),
        "--num-inference-steps", fmt_num(s["num_inference_steps"]),
        "--enable-dit-cache" if s["enable_dit_cache"] else "--disable-dit-cache",
        "--enable-cfg" if s["enable_cfg"] else "--disable-cfg",
        "--cfg-scale", fmt_num(s["cfg_scale"]),
        "--disable-torch-compile" if s["disable_torch_compile"] else "--enable-torch-compile",
    ]
    return argv


def server_env(cfg: dict) -> list[str]:
    return [f"{k}={v}" for k, v in sorted(cfg["server"]["env"].items())]


def client_argv(cfg: dict, result_dir: str, stall_recovery: bool) -> list[str]:
    c = cfg["client"]
    argv = [
        "bash", c["script"],
        "--exp-config", c["exp_config"],
        "--num-gpus", fmt_num(c["num_gpus"]),
        "--result-dir", str(result_dir),
        "--max-actions-per-query", fmt_num(c["max_actions_per_query"]),
        "--stall-patience", fmt_num(c["stall_patience"]),
        "--remote-min-chunk-translation", fmt_num(c["remote_min_chunk_translation"]),
    ]
    if stall_recovery:
        argv.append("--enable-stall-recovery")
    return argv


def server_banner(cfg: dict) -> list[str]:
    """Lines wnm_3d_server.sh prints (lines 241 to 275) that must appear verbatim in server.log."""
    s = cfg["server"]
    mask = s["dit_step_mask"]
    steps = int(s["num_inference_steps"])
    sampling = f"{sum(mask[:steps])}/{steps}" if s["enable_dit_cache"] else f"{steps}/{steps}"
    cfg_line = f"on (scale {fmt_num(s['cfg_scale'])})" if s["enable_cfg"] else "off"
    return [
        f"  Checkpoint     : {cfg['checkpoint']['path']}",
        f"  Replicas       : {s['num_replicas']}",
        f"  CUDA pool      : {s['cuda_devices']}",
        f"  GPUs/replica   : {s['gpus_per_replica']}",
        f"  History mode   : {s['history_sampling']}",
        f"  Sampling       : {sampling}",
        f"  DiT cache      : {'true' if s['enable_dit_cache'] else 'false'}",
        f"  CFG            : {cfg_line}",
        f"  Torch compile  : {'disabled' if s['disable_torch_compile'] else 'enabled'}",
    ]


def client_banner(cfg: dict, stall_recovery: bool) -> list[str]:
    """Lines eval_remote.sh prints (lines 275 to 309) that must appear verbatim in client.log."""
    c = cfg["client"]
    return [
        f"Experiment config: {c['exp_config']}",
        f"Action format: {c['action_format']}",
        f"Max actions per query: {c['max_actions_per_query']}",
        f"Stall patience: {c['stall_patience']}",
        f"Stall recovery: {'true' if stall_recovery else 'false'}",
        f"Remote stop epsilon: {c['remote_stop_eps']}",
        f"Remote minimum chunk translation: {fmt_num(c['remote_min_chunk_translation'])} m",
        f"WAM trace mode: {c['wam_trace_mode']}",
        "DAgger: false",
        f"Media output: {c['save_media']}",
        "Evaluating episodes from 0 to end",
        f"Total GN-Bench workers: {int(c['num_gpus']) * int(c['procs_per_gpu'])}",
    ]


META_COMPARE_KEYS = ("config_sha256", "n", "order_sha256", "stall_recovery", "server_argv",
                     "server_env", "client_argv", "repo_commit")


def build_meta(cfg: dict, config_bytes: bytes, result_dir: str, run_name: str, n: int,
               repo_commit: str, launch_id: str, stall_recovery: bool) -> dict:
    return {
        "kind": "launch",
        "run_name": run_name,
        "launch_id": launch_id,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_commit": repo_commit,
        "config_sha256": sha256_bytes(config_bytes),
        "order_sha256": cfg["episodes"]["order_sha256"],
        "order_seed": cfg["episodes"]["order_seed"],
        "n": int(n),
        "stall_recovery": stall_recovery,
        "server_argv": server_argv(cfg),
        "server_env": server_env(cfg),
        "client_argv": client_argv(cfg, result_dir, stall_recovery),
        "server_banner": server_banner(cfg),
        "client_banner": client_banner(cfg, stall_recovery),
        "settings": {k: cfg[k] for k in ("pins", "checkpoint", "server", "client", "benchmark", "env", "data")},
        "deviations": [
            "client --remote-min-chunk-translation 0.15 (paper Appendix B.1) where eval_remote.sh sets 0",
            "episodes restricted to the first n of the frozen order by the rung1_hook sitecustomize (GN0 files unchanged)",
            "server run with HF_HUB_OFFLINE=1 (no model setting changes)",
        ],
    }


def check_previous(meta: dict, previous_path: Path) -> None:
    """A resumed run must use the same settings as every earlier launch in the same run folder."""
    if not previous_path.exists():
        return
    for line in previous_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        prev = json.loads(line)
        if prev.get("kind") != "launch":
            continue
        for k in META_COMPARE_KEYS:
            if prev.get(k) != meta.get(k):
                raise SystemExit(
                    f"refusing to resume: {k} differs from launch {prev.get('launch_id')} in {previous_path}"
                )


def emit_nul(items: list[str]) -> None:
    sys.stdout.write("".join(f"{x}\0" for x in items))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--emit", required=True,
                    choices=["server", "server-env", "client", "server-banner", "client-banner", "meta"])
    ap.add_argument("--result-dir")
    ap.add_argument("--run-name")
    ap.add_argument("--n", type=int)
    ap.add_argument("--repo-commit")
    ap.add_argument("--launch-id")
    ap.add_argument("--previous", type=Path)
    ap.add_argument("--stall-override", choices=["true", "false"])
    ap.add_argument("--for-check", action="store_true")
    a = ap.parse_args(argv)

    if a.stall_override is not None and not a.for_check:
        ap.error("--stall-override is only allowed with --for-check")
    override = None if a.stall_override is None else a.stall_override == "true"
    config_bytes = Path(a.config).read_bytes()
    cfg = yaml.safe_load(config_bytes.decode("utf-8"))

    try:
        if a.emit == "server":
            emit_nul(server_argv(cfg))
        elif a.emit == "server-env":
            emit_nul(server_env(cfg))
        elif a.emit == "server-banner":
            emit_nul(server_banner(cfg))
        else:
            stall = resolve_stall_recovery(cfg, override)
            if a.emit == "client":
                if not a.result_dir:
                    ap.error("--result-dir is required")
                emit_nul(client_argv(cfg, a.result_dir, stall))
            elif a.emit == "client-banner":
                emit_nul(client_banner(cfg, stall))
            elif a.emit == "meta":
                if a.for_check:
                    ap.error("meta is never built for a check")
                for k in ("result_dir", "run_name", "n", "repo_commit", "launch_id"):
                    if getattr(a, k) in (None, ""):
                        ap.error(f"--{k.replace('_', '-')} is required")
                meta = build_meta(cfg, config_bytes, a.result_dir, a.run_name, a.n,
                                  a.repo_commit, a.launch_id, stall)
                if a.previous is not None:
                    check_previous(meta, a.previous)
                print(json.dumps(meta, sort_keys=True))
    except PendingDecision as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
