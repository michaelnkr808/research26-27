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

## v5: A6000 with low CPU/memory request (`job-a6000-lowcpu.yaml`), submitted 2026-10-16
- The L40 and A40 Jobs were deleted unrun after 1.5–2.5 h Pending.
- Prometheus (public, ~23:05Z) showed 8 free A6000s on schedulable nodes. All of those nodes had ≤1.8 CPU, or ≤9 Gi RAM, left. Our 4 CPU / 40 Gi request couldn't fit anywhere.
- Now 1 CPU / 16 Gi. The script builds the model under `torch.device("cuda")` (falls back to a CPU bf16 build) to keep host RAM low. `build_mode` is recorded in the results.
- Not submitted: the v4 24 GB variant (`job-24gb.yaml`). Prepared only; the user did not ask for it.

## v4: 24 GB layer-scaled variant (`job-24gb.yaml`)
- Same benchmark, built with 2/4/8 of 30 layers (`--num-layers`) on a 3090/4090/A5000, then extrapolated linearly to 30.
- Prediction (speculative): time and memory per layer are linear to within ~10%, and the 8-layer, checkpointing-off model fits B≥4 on 24 GB.
- Caveat: an extrapolation, not a measurement of the full model. Fixed costs (VAE, embeddings, head) appear as the intercept of the fit.

## Outcome (v5 run, 2026-10-16 23:15:39Z → 23:40:28Z, node nrp-01.csumb.edu, 1× RTX A6000)
Raw log: `results/job.log` (pulled 2026-10-17, ~19 h after completion, before the 24 h TTL). Job and ConfigMap deleted afterwards.

**Measured** (random weights, synthetic LIBERO-shaped input, fwd+bwd only, no optimizer step, median of 5 iters, 1 GPU, 1 run):
| micro-batch | step time | samples/s | VAE encode (inside step) | peak GPU mem |
|---|---|---|---|---|
| 1 | 0.808 s | 1.24 | 0.251 s | 23.9 GiB |
| 2 | 1.525 s | 1.31 | 0.479 s | 29.0 GiB |

- Params: 6.725B total; 6.021B trainable (MoT + proprio). Resident bf16 weights: 12.5 GiB. Built on the GPU (`gpu_context`), no fallback needed.
- **B=4 crashed** with `RuntimeError: size of tensor a (2) must match ... b (4)`. Fast-WAM's VAE encode is `torch.compile`d with CUDA graphs, and those don't tolerate batch-size changes within one process. This is an artifact of my benchmark design, not a memory limit. B≥4 is unmeasured.
- **Gradient checkpointing unsupported** in this code path: `ValueError: Wan MoT gradient checkpointing is not supported by the compiled tensor core.` Both build modes failed, so the checkpointing-on runs produced no data.

**Derived (speculative; from the 2 points above):**
- ~+5.1 GiB of activations per extra sample (no checkpointing).
- Full LIBERO run = 2.56M samples. On 8× A6000 at ~1.3 samples/s/GPU, that is ~68 h of fwd+bwd. Optimizer step and ZeRO communication are not measured, so **~3–3.5 days per full run**.
- VAE encoding is ~31% of step time. Caching latents offline could cut that.
- Micro-batch 2 + ZeRO-1 optimizer shard probably fits in 48 GB (~29 GiB + ~8 GiB).
- 24 GB cards can't hold even B=1 without sharding weights/grads (ZeRO-3).

**Vs. predictions:** throughput 1.24–1.31 samples/s (predicted 1–1.5). Full-run 3–3.5 days (predicted 3–4). Prediction "checkpointing fits B=8–16" untestable (unsupported). Prediction "no-checkpointing OOM by B=4–8" untested (crash was shape-related).

**Open:** Fast-WAM's config uses micro-batch 16/GPU. At ~5.1 GiB/sample that implies >80 GiB of activations, which conflicts with their stated 8-GPU single-node setup. Either the per-sample figure doesn't scale linearly, or their hardware is larger than assumed. Unresolved.
