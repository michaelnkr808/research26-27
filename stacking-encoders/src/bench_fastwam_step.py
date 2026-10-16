"""Single-GPU timing/memory benchmark for one Fast-WAM training step.

Builds the Fast-WAM model (Wan2.2-TI2V-5B video expert + 1B action expert)
with RANDOM weights. Timing and memory don't depend on weight values, so no
DiT checkpoint is downloaded; only the frozen Wan2.2 VAE is fetched. Feeds
synthetic LIBERO-shaped batches (2 cams concatenated -> 224x448, 33 frames,
32-step actions, cached-text-shaped context) and times forward+backward.

No optimizer step: full AdamW state for 6B params does not fit on one 48 GB
GPU. Optimizer/ZeRO memory is computed analytically from the param counts
this script reports.

Usage (inside the FastWAM repo root, so its configs resolve):
    python bench_fastwam_step.py --fastwam-root . --ckpt false --batches 1 2 4 8 16
Prints one line starting with RESULT_JSON at the end.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time

import torch


def str2bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def build_model(fastwam_root: str, ckpt: bool):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    sys.path.insert(0, os.path.join(fastwam_root, "src"))
    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    with initialize_config_dir(config_dir=os.path.abspath(os.path.join(fastwam_root, "configs")), version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[
                "task=libero_uncond_2cam224_1e-4",
                f"model.mot_checkpoint_mixed_attn={str(ckpt).lower()}",
                "model.skip_dit_load_from_pretrain=true",
                "model.load_text_encoder=false",
                "model.action_dit_pretrained_path=null",
            ],
        )
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
    # Mirror Wan22Trainer._apply_dit_only_train_mode: only MoT (video+action experts) + proprio encoder train.
    model.requires_grad_(False)
    model.dit.requires_grad_(True)
    if getattr(model, "proprio_encoder", None) is not None:
        model.proprio_encoder.requires_grad_(True)
    model.train()
    return model, cfg


def make_sample(cfg, batch: int, text_dim: int):
    d = cfg.data.train
    h, w = (int(x) for x in d.video_size)
    t = int(d.num_frames)
    action_horizon = t - 1
    return {
        "video": torch.rand(batch, 3, t, h, w) * 2 - 1,
        "action": torch.randn(batch, action_horizon, int(d.processor.action_output_dim)),
        "proprio": torch.randn(batch, t, int(d.processor.proprio_output_dim)),
        "context": torch.randn(batch, int(d.context_len), text_dim),
        "context_mask": torch.ones(batch, int(d.context_len), dtype=torch.bool),
    }


def timed(fn, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times, torch.cuda.max_memory_allocated() / 1024**3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastwam-root", required=True)
    ap.add_argument("--ckpt", type=str2bool, required=True, help="gradient checkpointing on/off")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    t_build = time.perf_counter()
    model, cfg = build_model(args.fastwam_root, args.ckpt)
    build_s = time.perf_counter() - t_build

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    text_dim = int(model.text_dim)
    weights_gib = torch.cuda.memory_allocated() / 1024**3

    out = {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_mem_total_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "fastwam_commit": os.environ.get("FASTWAM_COMMIT") or subprocess.run(
            ["git", "-C", args.fastwam_root, "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "grad_checkpointing": args.ckpt,
        "params_total": n_total,
        "params_trainable": n_train,
        "weights_resident_gib": weights_gib,
        "model_build_s": build_s,
        "warmup": args.warmup,
        "iters": args.iters,
        "seed": args.seed,
        "results": [],
    }
    print(f"[bench] gpu={out['gpu']} params_total={n_total/1e9:.3f}B trainable={n_train/1e9:.3f}B "
          f"weights={weights_gib:.1f}GiB ckpt={args.ckpt}", flush=True)

    for b in args.batches:
        sample = make_sample(cfg, b, text_dim)
        rec = {"micro_batch": b}

        def vae_only():
            with torch.no_grad():
                model.build_inputs(sample)

        def train_step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model.training_loss(sample)
            loss.backward()
            model.zero_grad(set_to_none=True)

        try:
            vt, _ = timed(vae_only, 1, args.iters)
            st, peak = timed(train_step, args.warmup, args.iters)
            rec.update(
                status="ok",
                vae_encode_s_median=sorted(vt)[len(vt) // 2],
                step_s=st,
                step_s_median=sorted(st)[len(st) // 2],
                samples_per_s=b / sorted(st)[len(st) // 2],
                peak_alloc_gib=peak,
            )
            print(f"[bench] B={b} step_median={rec['step_s_median']:.3f}s "
                  f"({rec['samples_per_s']:.2f} samp/s) vae={rec['vae_encode_s_median']:.3f}s "
                  f"peak={peak:.1f}GiB", flush=True)
        except torch.OutOfMemoryError as e:
            rec.update(status="oom", error=str(e).splitlines()[0][:300])
            print(f"[bench] B={b} OOM", flush=True)
        out["results"].append(rec)
        model.zero_grad(set_to_none=True)
        del sample
        gc.collect()
        torch.cuda.empty_cache()
        if rec["status"] == "oom":
            break

    print("RESULT_JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
