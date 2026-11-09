# DINOv3 retrain Jobs (PLAN Amendment 2026-11-09 (c), signed off 2026-11-09)

Written 2026-11-09, Mac only. Nothing here has been submitted or dry-run. Submitting needs the user's go, brought by the lead (PLAN section 7 item 3 is still open; the user chose "Phase B first, then the 13"). Paths are relative to this experiment folder.

## The two Jobs

Both train config `dinov3` for training seeds 0 and 1 as one Indexed Job (2 completions, parallelism 2, 1 GPU per pod), `backoffLimit: 0`, deadline 129600 s (36 h from creation).

| Job | Manifest | Recipe | Run dirs | Role |
|---|---|---|---|---|
| `stacking-encoders-proxy-train-retrain-dinov3-l16` | `job-train-retrain-dinov3-l16.yaml` | `--layers 16`, 10 epochs (21,696 steps), rest default (option B1) | `/mnt/proxy/runs/dinov3-l16-s0`, `-s1` | Submit first |
| `stacking-encoders-proxy-train-retrain-dinov3-e20` | `job-train-retrain-dinov3-e20.yaml` | `--epochs 20` (43,392 steps), default size (option A1) | `/mnt/proxy/runs/dinov3-e20-s0`, `-s1` | Fallback only |

Deadline arithmetic (speculative): about 8 h per run (2 x the measured 0.64 to 0.65 s/step over 21,696 steps for the 16-layer guess, or 43,392 steps for the 20-epoch run, plus about 0.2 h setup) x 2 = 16 h, plus 20 h for the queue (round 1 waited 3 h to 9 h or more) = 36 h. The latest a pod can start and still finish is about 28 h after creation. RUN.md stop 5 (one run over 48 h) is not close.

## Order, and the exact fallback trigger

1. Submit the 16-layer Job only. Do not submit both.
2. Use the 20-epoch Job only if the 16-layer Job FAILS BEFORE COMPLETING ITS FIRST EPOCH (about 2,170 steps; the log prints a line at step 0 and then about every 433 steps) for an environment reason. The evidence that counts, from the saved pod log (`kubectl logs <pod> > results/logs/...`) and the pod status:
   - CUDA out of memory: a traceback ending in `torch.OutOfMemoryError: CUDA out of memory` (or `RuntimeError: CUDA out of memory`) after the `FLAGS:` line, pod phase `Failed`, container exit code 1 (not 137), no `wrote /mnt/proxy/runs/dinov3-l16-s... params` line, and the last `"step":` line below 2169 (or none).
   - Builder rejection: an `AssertionError` or other exception whose traceback passes through `build_model` in `proxy_wam.py` (for example the head-dim assertion at line 376), under the same conditions.
   Either index failing counts for both seeds: the two seeds must use one recipe, and with `backoffLimit: 0` a failed pod ends the Job and terminates the other pod. Then submit the 20-epoch Job (new run dirs; the failed 16-layer dirs are left alone, deleting data is gated).
3. NOT triggers: the Job trains past the first epoch and later fails or scores badly (a poor score is never a reason to switch: that would be a second retrain, which the amendment forbids; if the level target X = 27 is missed the user decides, PLAN section 3); `DeadlineExceeded` in the queue; a node or GPU fault; pod `OOMKilled` / exit 137 (host RAM, not the layer count); an `ABORT:` line from the guard (exit 2 or 3); a ConfigMap or PVC error; the pip setup failing. For those the lead diagnoses and resubmits the SAME 16-layer manifest, or asks the user. (This reading of "environment reason" is the worker's; the lead confirms.)
4. Open question the lead may want settled before submitting: the GPU list (the amendment's training list) includes 24 GB cards (RTX 3090, 4090, A5000). The 8-layer model fit a 24 GB card; 16 layers may not. If the scheduler puts one pod on a 24 GB card and it runs out of memory while a 48 GB card would have fit, the fallback rule fires for a reason the card choice caused (speculative; GPU memory for 16 layers is unmeasured). Dropping the 24 GB cards would avoid that but lengthens the queue and is a change to the amendment's list, so it is not done here.

## Pre-flight (before any submit; the lead runs kubectl)

1. ConfigMap `stacking-encoders-proxy-src` still has the same 14 keys as at LOG 2026-11-06 (a) 2(b) (and `proxy_wam.py`, `fastwam_data.py`, `diagnostics.py`, `t5_cache.py`, `setup_env.sh` among them): `kubectl get configmap stacking-encoders-proxy-src -n aria-world -o jsonpath='{.data}'` keys, or `-o json` and list `.data` keys.
2. `/src/proxy_wam.py` in the cluster equals main's. Main's sha256 at the base of this branch (cb9e329) is `7169332991dc7c1ebba548d31fa35c7074e7981c35b56b18946edd534fc339a4` (verified on the Mac, `shasum -a 256 world-model-stacking-encoders/src/proxy_wam.py`). On the cluster: `kubectl get configmap stacking-encoders-proxy-src -n aria-world -o jsonpath='{.data.proxy_wam\.py}' | shasum -a 256`; the hash must match main's at submit time (recompute main's then). If it differs, stop: either recreate the ConfigMap from main (then SRC_GIT_COMMIT differs from 2bab64f even though training code is unchanged; record it) or ask.
3. The out dirs are absent or empty: on the PVC, `/mnt/proxy/runs/dinov3-l16-s0`, `-s1` (and `dinov3-e20-s0`, `-s1` for the fallback) must not hold `ckpt.pt` or `model.pt`. The Job's own guard aborts with exit 3 if they do (that needs no pod, but a quick read-only check beforehand saves a queue wait). No earlier Job wrote there (verified from the manifests in `jobs/rendered/`, which use `runs/pilot-*`, `pilot2-*`, `dinov3-s0`, `dinov3-s0-a40`, `dinov3-s1`; not checked on the PVC).
4. Fill the code commit at submit: `sed "s/FILL_AT_SUBMIT/<sha>/" jobs/job-train-retrain-dinov3-l16.yaml > jobs/rendered/job-train-retrain-dinov3-l16.yaml` (commit the rendered file as submitted). The pod aborts (exit 2) if the placeholder is left in.
5. `kubectl create --dry-run=server -f` on the rendered file, then the user's go, then create.
6. GPU accounting (cap: 8 GPUs across all our running Jobs, `_standards/08`): the Job asks for 2 GPUs. Count the GPUs our other Jobs hold at the moment of submit (phase B perturbed-frames Jobs, any eval Jobs) and submit only if that count plus 2 is at most 8. Pending pods hold no GPU, but they will start the moment a node has one, so count them as if running. The user's "Phase B first" choice means: do not submit while phase B GPU Jobs run. Jobs per sweep: 1 (the 20-epoch fallback is a second Job only if the first failed).

## What to check about 15 minutes after a pod starts training

- GPU use over 10 minutes (Prometheus `DCGM_FI_DEV_GPU_UTIL{pod=~"stacking-encoders-proxy-train-retrain-dinov3-l16-.*"}`, or `kubectl exec <pod> -- nvidia-smi` if the node has no DCGM series): at or above 40% (wave 1 measured 76 to 97%). Below 40% over 10 minutes: recheck the pods, then stop and report.
- Step rate: the log line has `"step"` and `"sec"` (seconds on the training clock); sec per step between two lines. Wave 1 at 8 layers measured 0.64 to 0.65 s/step. For 16 layers expect more, up to about twice (speculative, about 1.3 s/step, which gives about 8 h). Much slower than that (say over 2 s/step, which would be about 12 h) is worth telling the user; the deadline still holds up to about 1.5 s/step with a short queue.
- GPU memory: `nvidia-smi` memory used against the card's total, in the first minutes. Close to full is a warning that the fallback may fire.
- The first lines of the log: `=== RETRAIN B1 (16 layers) index ...`, `SRC_GIT_COMMIT=<sha> RESUME=0`, `FLAGS: python /src/proxy_wam.py ... --layers 16 --out /mnt/proxy/runs/dinov3-l16-s<seed>`, then `nvidia-smi`, and later `[proxy ...] 1712 episodes, 277713 frames, ... in_dim ..., 21696 steps` (a wrong step count means wrong flags), then the step lines.
- When it finishes: save each pod's log to `results/logs/` within 24 h (the Job's ttl); look for `wrote /mnt/proxy/runs/dinov3-l16-s<seed>, <n> params` (the parameter count should be about twice the 8-layer 31,647,943; speculative). The pip freeze is copied to `/mnt/proxy/logs/pip-env-retrain-l16-dinov3-s<seed>.log` (round 1 lost it).

## The two evaluations afterwards (a later task; not done here)

Reuse the L4-only eval pattern, `jobs/rendered/job-eval-dinov3-s0-l4.yaml` (gpu.product NVIDIA-L4 only, as Amendment 2026-11-07), with the full-benchmark subset ConfigMap `stacking-encoders-proxy-subset-full` and the same sha256 check. Fields that change, per recipe actually trained (16-layer: `dinov3-l16-s0`, `dinov3-l16-s1`; fallback: `dinov3-e20-s0`, `dinov3-e20-s1`):

- `metadata.name` (a new name per seed or per recipe) and the header comment.
- `ITEMS`: the run dir names, `<run dir>@<shard>/8` for all 8 shards (the old Jobs had `dinov3-s0-a40@0/8` etc.); `completions` and `parallelism` to match, parallelism within the 8-GPU cap.
- The output directory in the script (`OUT=$R/rollouts/eval-dinov3-s0/...` and the `mkdir -p`): a new directory per run (for example `eval-dinov3-l16-s0`), and the log file name prefix (`eval-dinov3-s0-...`), so the old round-1 records are not overwritten or mixed in.
- `SRC_GIT_COMMIT` and the ConfigMap `stacking-encoders-proxy-src` contents at submit time (the eval needs `rollout.py`; for a 16-layer model the rollout code must build the same model from the run dir's `config.json`, whose `model_kw` carries `layers`; not checked in this task, check before submitting), and `activeDeadlineSeconds` (the L4 evals took days; size it as for the old Jobs).
- Memory: the old evals used 28Gi / 33Gi with a 15 to 19 GB working set; a 16-layer policy is bigger on the GPU, not necessarily in host RAM; check at the 30-minute look.
- The seed-noise read afterwards (`jobs/job-seednoise-read.yaml`, `jobs/SEEDNOISE.md`) must be pointed at the new record directories (its A = training seed 1, B = training seed 0 directories are hard-coded for round 1).
- Unchanged: the subset file and its sha256 (`9f4baa42...`), eval seeds 41, 42, 43, `--order-seed 0`, `--clean 1`, `--envs 4`, `--cuda-graph`, the image, the PVC, the node exclusions.
