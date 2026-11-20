"""Start-weights key check for check 2 (PLAN.md, Amendment 2026-11-20 (b), step 3).

The trainer loads the stage 3 checkpoint with load_state_dict(..., strict=False), shard by
shard, and does not print what did not match (WNM-3D 7fe2d5f, gammanav/vln/experiment/
base.py lines 499 to 538). This script repeats that on the CPU, from the same Hydra
settings the launcher produces, and records the missing and unexpected keys.

    python check2_keycheck.py --wnm3d-root /workspace/WNM-3D \
        --dry-run-log /workspace/check2/runs/dryrun.log --out /workspace/check2/keycheck

Steps:
1. Take the Hydra overrides from the "Command:" line the launcher prints with DRY_RUN=true
   (launcher lines 258 to 262): every argument after .../experiment/experiment.py.
2. Compose gammanav/vln/configs/conf.yaml with them, apply apply_action_overrides as
   experiment.py does, and build the model with hydra.utils.instantiate(cfg.model), as
   base.py create_model does (line 500). Nothing is moved to the GPU.
3. For each shard in model.safetensors.index.json, in sorted order as base.py does
   (line 517), call model.load_state_dict(shard, strict=False) and keep what it returns.
4. Never-loaded keys = model state_dict keys that no shard provided; unexpected keys = the
   union of the per-call unexpected keys. Write keycheck.json (counts and full lists).

Exit code: 0 when both counts are 0; 2 when either is not 0 (stop before timing, report
to the lead); 3 when loading raised (for example a shape mismatch), with the error in
keycheck.json.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shlex
import sys
import time
import traceback

ENTRY_SUFFIX = os.path.join("experiment", "experiment.py")


def overrides_from_dry_run(text: str) -> list[str]:
    """Hydra overrides from the launcher's DRY_RUN output (the line starting "Command:")."""
    lines = [ln for ln in text.splitlines() if ln.startswith("Command:")]
    if len(lines) != 1:
        raise ValueError(f"expected exactly one 'Command:' line, found {len(lines)}")
    tokens = shlex.split(lines[0][len("Command:"):])
    idx = [i for i, t in enumerate(tokens) if t.endswith(ENTRY_SUFFIX)]
    if len(idx) != 1:
        raise ValueError("could not find the experiment.py entry point in the command")
    overrides = tokens[idx[0] + 1:]
    if not any(o.startswith("pretrained_model_path=") for o in overrides):
        raise ValueError("dry-run command has no pretrained_model_path override")
    return overrides


def pretrained_path(overrides: list[str]) -> str:
    for o in overrides:
        if o.startswith("pretrained_model_path="):
            return o.split("=", 1)[1]
    raise ValueError("no pretrained_model_path override")


def shard_files(ckpt_dir: str) -> tuple[list[str], dict]:
    with open(os.path.join(ckpt_dir, "model.safetensors.index.json"), encoding="utf-8") as fh:
        index = json.load(fh)
    return sorted(set(index["weight_map"].values())), index


def check_keys(model, ckpt_dir: str, load_file, log=print) -> dict:
    """Load every shard into model with strict=False, as base.py does, and collect keys."""
    model_keys = set(model.state_dict().keys())
    shards, index = shard_files(ckpt_dir)
    index_keys = set(index["weight_map"].keys())
    provided: set[str] = set()
    unexpected: set[str] = set()
    per_shard = []
    for shard in shards:
        t0 = time.time()
        sd = load_file(os.path.join(ckpt_dir, shard))
        keys = set(sd.keys())
        result = model.load_state_dict(sd, strict=False)
        missing_here = list(getattr(result, "missing_keys", result[0]))
        unexpected_here = list(getattr(result, "unexpected_keys", result[1]))
        provided |= keys
        unexpected |= set(unexpected_here)
        per_shard.append(
            {
                "shard": shard,
                "n_keys": len(keys),
                "n_unexpected": len(unexpected_here),
                "n_missing_in_this_call": len(missing_here),
                "seconds": round(time.time() - t0, 3),
            }
        )
        log(f"[keycheck] {shard}: {len(keys)} keys, {len(unexpected_here)} unexpected")
        del sd
        gc.collect()
    never_loaded = sorted(model_keys - provided)
    return {
        "checkpoint_dir": os.path.abspath(ckpt_dir),
        "n_shards": len(shards),
        "n_model_state_keys": len(model_keys),
        "n_checkpoint_keys": len(provided),
        "index_matches_shards": index_keys == provided,
        "n_missing": len(never_loaded),
        "n_unexpected": len(unexpected),
        "missing_keys": never_loaded,
        "unexpected_keys": sorted(unexpected),
        "per_shard": per_shard,
        "pass": not never_loaded and not unexpected,
    }


def build_model(wnm3d_root: str, overrides: list[str]):
    """Build the model exactly as experiment.py + base.py create_model do (on the CPU)."""
    sys.path.insert(0, wnm3d_root)
    os.chdir(wnm3d_root)
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from gammanav.vln.utils.action_args_override_utils import apply_action_overrides

    config_dir = os.path.join(wnm3d_root, "gammanav", "vln", "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="conf", overrides=overrides)
    cfg = apply_action_overrides(cfg)
    model = instantiate(cfg.model)
    return model, cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="check 2 start-weights key check")
    ap.add_argument("--wnm3d-root", required=True)
    ap.add_argument("--dry-run-log", required=True, help="saved output of the DRY_RUN=true launch")
    ap.add_argument("--out", required=True, help="output folder for keycheck.json")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    out_path = os.path.join(os.path.abspath(a.out), "keycheck.json")
    wnm3d_root = os.path.abspath(a.wnm3d_root)
    t0 = time.time()
    report: dict = {"started": t0}
    try:
        with open(a.dry_run_log, encoding="utf-8") as fh:
            overrides = overrides_from_dry_run(fh.read())
        report["overrides"] = overrides
        ckpt = pretrained_path(overrides)
        ckpt = ckpt if os.path.isabs(ckpt) else os.path.join(wnm3d_root, ckpt)
        model, _ = build_model(wnm3d_root, overrides)
        report["build_seconds"] = round(time.time() - t0, 1)
        head_cfg = getattr(getattr(model, "action_head", None), "config", None)
        # base.py lines 533 to 538 inject LoRA after loading when this is true; recorded only
        report["defer_lora_injection"] = bool(getattr(head_cfg, "defer_lora_injection", False))
        from safetensors.torch import load_file

        report.update(check_keys(model, ckpt, load_file))
        code = 0 if report["pass"] else 2
    except Exception as exc:
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        report["pass"] = False
        code = 3
    report["seconds"] = round(time.time() - t0, 1)
    report["exit_code"] = code
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(
        f"[keycheck] pass={report['pass']} missing={report.get('n_missing')} "
        f"unexpected={report.get('n_unexpected')} exit={code} -> {out_path}"
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
