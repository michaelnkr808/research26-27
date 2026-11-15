# LOG: 2026-11-12 pretrained-start sizing check

## 2026-11-15 [agent, lead, implementation] (run1 on Modal, one L40S: done; slice copy pod on Nautilus: done and deleted)

The user's go, in the implementation window on 2026-11-14 (local time), after the provider (Modal), card (L40S), price ($1.95/h) and estimated total (about $2 to $3) were stated: "yes". The same go covered merging PRs #29 and #31, the slice copy pod and the Wan 2.2 whole-transformer download Job on Nautilus (PR #32, separate). Code at main `7f58dbf` (PR #31 merge); Fast-WAM `7faa711`.

**Day's prices** (`modal billing rates`, read 2026-11-15 ~07:50 UTC, verified): L40S $1.95/h; CPU $0.0473/core/h; memory $0.008/GiB/h; volumes $0.09/GiB/month. Runpod and Vast.ai prices were not read today (the PLAN Step 1 comparison was skipped; Modal had been chosen in RUN.md).

**Step 3, data slice (Nautilus, RUN-slice.md)** (verified):
- `src/test_make_sizing_slice.py`: `all ok`. ConfigMap `stacking-encoders-sizing-slice-src` and Pod `stacking-encoders-sizing-slice` created 07:50:44 UTC; script started 07:52:53, `=== READY` 07:57:07. Log saved: `results/slice-pod.log`.
- 16 episodes, seed 0, pool 1,712, all four suites, 15 tasks, 2,686 frames; DINOv3 only; no `vae_b`. 30 chunks; slice 2.4G on the pod, 2.3G on the Mac (`du -sh`). Episode ids in `slice.json` (on the Modal volume `stacking-sizing:data/sizing-slice/slice.json`, and in `results/slice-pod.log`).
- Copy: 6 chunk transfers failed with `unexpected EOF` from `kubectl exec` and were refetched (part000 twice, part002, part005 twice, part018); every chunk then matched `SHA256SUMS`, tar sha256 `5301654e1ffc81c30105293ab80919a94d0cf5bbf4297947a0aa2dade499f7f7` matched `tar.sha256`, and `MANIFEST.sha256` checked with no bad lines.
- Pod and ConfigMap deleted ~08:04 UTC. Local copy is in the session scratchpad only (not durable).

**Step 2, volume**: `modal volume create stacking-sizing`; slice put at `data/sizing-slice` (layout checked: `dinov3 vae meta t5 MANIFEST.sha256 slice.json`).

**Step 4, stage** (07:51 to 07:54 UTC, CPU; `results/stage.log`). Deviation from RUN.md: run with `--revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e` (pinned) instead of the default `main`; that sha was `main` when read on 2026-11-14. Files and sha256 (verified, written to `wan/Wan2.2-TI2V-5B/manifest.json` on the volume):
- `diffusion_pytorch_model-00001-of-00003.safetensors` 9,825,014,472 B, `720b06c4ade5e87c1246bba8ac95b664c638749cd9b102cf84d823bb44c026a1`
- `diffusion_pytorch_model-00003-of-00003.safetensors` 178,558,176 B, `6306f7894c345de9093ad588771c2abfaeb668a81f7a6d9a918bd26ba3568e49` (needed by `_needed_shards` for blocks 0-7 and non-block tensors)
- `diffusion_pytorch_model.safetensors.index.json` 72,865 B, `bfa2337f1163e195d24151a72298daf34a620543898109be47e414c8daa5b3fe`
- `config.json` 251 B, `d1fea36899d00c2501b836c13ad65af56e2f9529ba622e50886d3f5c3e6c02bc`
- Download 37.5 s.

**Step 5, preflight** (`results/check.log`): `missing: []`; both configs see 4 suites, 16 episodes, 2,686 frames, 15 tasks; dims `vae: 48` and `dinov3: 1024`; local disk free 512 GiB.

**Step 6, run1** (`modal run --detach modal_app.py::run --run-id run1`; app `ap-DNNMVNwyfx5oKVUEJtp1aP`; start 08:07:46 UTC, launcher done in 371 s, exit 0, app stopped ~08:14 UTC; inner and outer limits not reached). Results fetched to `results/run1/` (21 files). Card: NVIDIA L40S, 47,695,921,152 B (44.42 GiB), driver 610.57.04, torch 2.7.1+cu128. Keep tag "first 8, user's choice". All numbers below verified from `results/run1/measure-*.json`, `load-*.json` and `run.log`.

| | `none` | `dinov3` |
|---|---|---|
| trainable params | 1,681,051,847 | 1,706,221,767 |
| tensors loaded / partly / new / skipped | 231 / 0 / 232 / 594 | 228 / 3 / 232 / 594 |
| loaded tensors re-read and compared | max abs diff 0 | max abs diff 0 |
| peak reserved, m=1 (larger of micro-step and full step) | 34,554,773,504 B (32.18 GiB) | 35,095,838,720 B (32.69 GiB) |
| sweep | m=1..16 ok, m=32 OOM | m=1..16 ok, m=32 OOM |
| m* (0.9 x memory rule; m=16 above 0.9) | 8 | 8 |
| peak reserved at m*=8, global-batch steps | 40,963,670,016 B (38.15 GiB) | 41,557,164,032 B (38.70 GiB) |
| s per optimizer step, batch 128 (m=8, accumulation 16), 2 timed | 3.907 (3.902, 3.911) | 3.994 (3.992, 3.997) |
| loss at start, model total (vae / action) | 3.735 (2.634 / 1.101) | 5.784 (2.908 / 0.994; dinov3 1.882) |
| predict-zero total (vae) | 2.526 (1.492) | 4.589 (1.774; dinov3 1.882) |

**Full-run cost** (PLAN "Decision rule" arithmetic, card-hours = S x t_step / 3600, S = 21,696; run this session): `none` 23.54 L40S-hours, $45.91 at $1.95; `dinov3` 24.07 L40S-hours, $46.94. Measured on L40S only; not transferred to A40/A6000/L40 (PLAN: no assumed ratios). Excluded as the PLAN lists: evaluation, data-loading stalls (host batch assembly was 1 ms (`none`) and 14 ms (`dinov3`) median per micro-batch on the slice), checkpoint writes, startup, failures, and storage/transfer of the full caches.

**Predictions vs result** (PLAN "Predictions"):
1. Loading as predicted: kept blocks and non-block tensors load exactly; with DINOv3, 3 tensors partly loaded (the widened slices). Confirmed.
2. Trainable 1.6 to 1.9B: 1.68B and 1.71B. Confirmed.
3, 5. Peak at m=1 25 to 34 GiB, fits 48 GB at m between 4 and 32: 32.2 GiB, m*=8. Confirmed.
6. 6 to 18 s per step on A6000: 3.9 s on L40S. Different card; not comparable by the PLAN's own rule.
7. 36 to 108 A6000-hours: 23.5 L40S-hours. Below the range, on a different card.
8. One encoder adds a few percent: +2.2% time, +0.55 GiB peak at m*. Confirmed.
9. Video loss above predict-zero at start: 2.634 vs 1.492 (`none`). Confirmed.

**Decision rule** (PLAN): fits the 48 GB class at m*=8, so full runs can use the 48 GB class Nautilus has. One full run (~$46 on Modal L40S) is above the user's ~$30/month paid budget, so by the rule full runs go to Nautilus or the PI ask, not the user's money. These are the rule's outputs; the user decides.

**Step 8, cost** (`modal billing report --for today`, read ~08:20 UTC): metered $0.34 for the day across stage, check and run1 (run1: L40S $0.2026, CPU $0.0393, memory $0.0399; stage and check about $0.01), all covered by credits, billed $0.00. Estimate was $2 to $3; actual far lower because the run took 6 minutes, not 20 to 35 (the sweep stopped at m=32 and each process was short).

**Step 9, cleanup: not done.** Volume `stacking-sizing` still holds the weights (about 10 GB) and the slice (2.3 GB), about $1.1/month at $0.09/GiB. Deleting is the user's call (RUN.md step 9).

**Not done or deviations**: Runpod/Vast.ai prices not read; revision pinned at stage; slice local copy only in scratchpad.

Next: the user decides whether to keep the Modal volume. Planning reads the cost result for the PI meeting and the full-run plan.
