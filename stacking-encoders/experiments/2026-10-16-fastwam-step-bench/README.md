# 2026-10-16 — Fast-WAM training-step benchmark (1 GPU)

**Purpose:** Replace back-of-envelope estimates with measured per-GPU throughput and memory for one Fast-WAM training step, so we can estimate wall-clock time for full and reduced-scale WAM runs on Nautilus.

**What runs:** `../../src/bench_fastwam_step.py`, from Fast-WAM at commit `7faa71108368fbb3b6885649f112af607427a2d4` (github.com/yuantianyuan01/FastWAM).
- Random weights. Timing and memory don't depend on weight values, so only the frozen Wan2.2 VAE (1.41 GB, ModelScope) is downloaded.
- Synthetic LIBERO-shaped batch, per `libero_uncond_2cam224_1e-4`: 2 cams concatenated to 224×448, 33 frames, 32×7 actions, 8-d proprio, 128×4096 text context.
- Same trainable set as their trainer: MoT (video + action experts) + proprio encoder.
- Forward + backward only, with no optimizer step. Optimizer/ZeRO memory is computed analytically from the reported param counts.
- Micro-batches 1, 2, 4, 8, 16, each with gradient checkpointing off and on. Stops at the first OOM.

**Hardware:** 1× L40 or L40S 48 GB (`nvidia.com/gpu` + `gpu.product` affinity), image `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`. Manifests: `job-l40.yaml` and `job-a40.yaml` (`nvidia.com/a40`). Both are submitted; the first to start is kept and the other deleted. The GPU actually used is recorded in the log's `nvidia-smi` output.
- The first attempt (`job.yaml`, 1× RTX A6000) sat Pending for ~1 h with no free A6000. It was deleted unrun, and the user approved switching cards.
- Timings on an L40 are not A6000 timings. Converting speed requires an assumed L40:A6000 throughput ratio, and that ratio gets labeled as an assumption. Memory numbers transfer directly.

**Command:**
```
kubectl create configmap stacking-encoders-bench-src -n aria-world --from-file=../../src/bench_fastwam_step.py
kubectl apply -f job-l40.yaml
kubectl logs -n aria-world job/stacking-encoders-fastwam-step-bench-l40 > results/job.log
kubectl delete job -n aria-world stacking-encoders-fastwam-step-bench-l40
kubectl delete configmap -n aria-world stacking-encoders-bench-src
```

**Commit:** see git log for the commit that adds this folder (made before submission).

## Predictions (written before running; speculative)
- Trainable params ≈ 6B. Resident bf16 weights + grads ≈ 24 GB, plus the VAE.
- Checkpointing off: fits B=1–2, OOM by B=4–8.
- Checkpointing on: fits up to B=8–16.
- Throughput with checkpointing ≈ 1–1.5 samples/s/GPU. Basis: ~3.2e13 FLOPs/sample (6·5B·882 video tokens + VAE), ×1.33 for recompute, at ~50 TFLOPS effective on an A6000.
- Implied full LIBERO run (20k steps × global batch 128 = 2.56M samples) on 8× A6000: ~3–4 days. Excludes ZeRO communication overhead and evaluation.

## Outcome
(pending)
