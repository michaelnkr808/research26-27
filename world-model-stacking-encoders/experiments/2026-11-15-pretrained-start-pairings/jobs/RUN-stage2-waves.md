# Stage 2 wave plan: a proposal for the user's sweep OK

Written 2026-11-17 by a worker, Mac only. Nothing has been submitted. This is the wave and sub-wave split, with Job counts, that PLAN Order of work step 5 asks the user to OK before wave 1 (the Nautilus gate in `STANDARDS.md` line 49: more than 8 GPUs at once, or a sweep of more than 20 Jobs, is a stop). How to render and submit each Job: `RUN-stage2-train.md` and `RUN-stage2-eval.md`. Labels: arithmetic is speculative unless a measured input is named; the limits are the PLAN's.

## In plain words

Seventeen new training runs, one per configuration and seed, plus an evaluation of every trained model on clean LIBERO and then on all of LIBERO-Plus. Two limits shape the order: at most 8 GPUs at once for everything we run (training and evaluation together), and disk space on the current volume, which holds at most 4 training runs at once until the second volume exists. Wave 1 trains seed 0 of the five encoder configurations, in two parts because of the disk; meanwhile the stage 1 model gets its LIBERO-Plus evaluation, which also tells us how long a LIBERO-Plus evaluation takes. Wave 2 trains the remaining twelve runs, in two halves of six if the second volume exists, or in about four smaller groups if it does not. LIBERO-Plus evaluation, not training, is likely the larger cost, and its size is only known after the first one (PLAN amendment 2026-11-17 item 1); the measured cost goes to the user before wave 2's evaluations.

## Limits used (sources)

- **GPUs**: at most 8 at once, training and evaluation pods together (`STANDARDS.md` line 49; PLAN Compute "GPUs at once"). Every Job of this sweep is labelled `experiment=stacking-pairings`, and so are its pods.
- **Disk**: 190 GiB free on `stacking-encoders-proxy` (LOG 2026-11-17, cache pre-check, read by hand), at most 40 GB per running training job (PLAN Hand-off step 4), so at most 4 training runs at once (LOG 2026-11-17's reading). A finished run keeps about 26.7 GB (`ckpt.pt` about 20 GB + `model.pt` about 6.7 GB; stage 1 PLAN lines 83 and 87) until its `ckpt.pt` may be deleted (PLAN amendment 2026-11-15 later, item 3: finished, evaluation loaded `model.pt`, sha256 logged, backup copy verified, and the user's OK for that deletion or batch). The second volume (1Ti, the implementation session's recommended size; the user chose "a second one" and its creation needs the user's go) removes the disk limit for wave 2.
- **Card set**: training on L40 or L40S (stage 1's set); every evaluation pod on L4.
- **Jobs at once**: far under NRP's 400.

## Wave 1: seed 0 of the five encoder configurations (the design locks when the first of these starts)

| sub-wave | training Jobs (GPUs) | evaluation running at the same time (L4 GPUs) | GPUs at most | disk at most |
|---|---|---|---|---|
| 1a | `dinov3`, `dinov3+c-radiov4`, `dinov3+siglip2`, `c-radiov4` (4) | smoke of `none` (LIBERO-Plus timing shard only, 1), then LIBERO-Plus of `none` seed 0 at parallel 4 (4) | 8 | 4 x 40 = 160 GB |
| 1b | `siglip2` (1), once 1a's runs have finished (free space then about 204 - 4 x 26.7 = 97 GB, arithmetic) | wave 1a's smokes, clean and LIBERO-Plus Jobs, at most 7 L4 pods | 8 | 40 GB + the finished runs |

Why this order: the three configurations the main question needs (`dinov3`, the harmful pair and the helpful pair) start first; `c-radiov4` alone joins them because the "below both members" comparison uses it; `siglip2` alone is first in the scope-cut order the plan recommended (Open item l). This is a choice of order, not of scope; wave 1b runs in full. **If the second volume exists before wave 1, all five start together** (5 training GPUs + 3 L4 pods for `none`'s LIBERO-Plus).

Evaluations in wave 1, per finished checkpoint: smoke (1 L4 pod, under 1 h expected), clean (4 L4 pods; stage 1: about 1.1 h each), gate, then LIBERO-Plus (16 shards, parallel set by the GPUs left). **No verdict, rule change, recipe change or wave 2 change from wave 1** (PLAN Decision rule item 6).

**Wave 1 Jobs: 22** (5 training, 6 smoke: five configurations plus `none`'s timing shard, 5 clean, 6 LIBERO-Plus including `none` seed 0). Pods: 5 + 6 + 20 + 6 x 16 = 127 (retries not counted).

## Wave 2: seeds 1 and 2 of all six configurations, 12 runs, in full regardless of wave 1 (PLAN Decision rule item 6)

**Option A, with the second volume (recommended; the user's "a second one", go for creating it still needed):**

| sub-wave | training Jobs | evaluation pods at the same time | GPUs at most |
|---|---|---|---|
| 2a | seed 1 of `none`, `dinov3`, `siglip2`, `c-radiov4`, `dinov3+siglip2`, `dinov3+c-radiov4` (6) | at most 2 L4 (wave 1's LIBERO-Plus shards) | 8 |
| 2b | seed 2 of the six (6), when 2a's pods are gone | at most 2 L4 (wave 1 and 2a evaluations) | 8 |

Disk on the second volume: 6 x 40 = 240 GB at the peak plus the finished runs (12 x 26.7 = 320 GB if no `ckpt.pt` is deleted), under 1Ti (arithmetic).

**Option B, no second volume (the current volume only):** each sub-wave starts only after the previous runs' `ckpt.pt` files are deleted under the conditions above (each deletion batch needs the user's OK in the lead's window). Free space then shrinks by about 6.7 GB per finished run (its `model.pt`), so the sub-waves are about 4, 4, 3 and 1 runs (or 4, 3, 3, 2), 4 sub-waves in all (arithmetic from 204 GB, wave 1's five `model.pt`, and 40 GB per running job; whether stage 1's own `ckpt.pt` is deleted changes it by one run). Each sub-wave also waits for the previous one's evaluation and backup, so this is much slower.

Sub-wave order within wave 2 is by seed (all of seed 1, then all of seed 2), so a cut-short wave never leaves one configuration with more seeds than another.

**Wave 2 Jobs: 36** (12 training, 12 clean, 12 LIBERO-Plus). Pods: 12 + 48 + 12 x 16 = 252.

## Totals and the gate

| | Jobs | pods (no retries) |
|---|---|---|
| training | 17 | 17 |
| smoke (incl. the LIBERO-Plus timing shard) | 6 | 6 |
| clean LIBERO (4 shards each) | 17 (`none` seed 0 was done in stage 1) | 68 |
| LIBERO-Plus (16 shards each, proposed) | 18 | 288 |
| **all** | **58** | **379** |

Over 20 Jobs, so the user's OK for the sweep as a whole is needed (`STANDARDS.md` line 49). The extension, if the three-seed outcome is a complete I (PLAN Decision rule item 7, allowed by the user): 4 more training, 4 clean and 4 LIBERO-Plus Jobs (12), with their own OK at that time.

Time (speculative): a training run took 32.2 h on an L40 for `none` (stage 1 LOG); encoder runs are expected a few percent slower. LIBERO-Plus per checkpoint is not measured; for scale only, the PLAN's arithmetic gives about 50 to 140 L4 pod-hours per checkpoint (amendment 2026-11-17 item 1), so 18 checkpoints are roughly 900 to 2,500 L4 pod-hours, weeks of wall time at 2 to 4 L4 pods at once. The timing shard and `none` seed 0's evaluation replace this with measured numbers before any wave 2 evaluation.

## What the user must OK, step by step

| step | needs from the user | where |
|---|---|---|
| before wave 1 | the sweep as a whole: this plan, 58 Jobs (+12 if the extension), never more than 8 GPUs at once, L40/L40S for training and L4 for evaluation | lead's window (PLAN Order of work step 5) |
| before wave 1 (or before wave 2 at the latest) | the go to create the second volume (name and size, 1Ti proposed, `rook-cephfs`, ReadWriteMany) | lead's window; it is a cluster action |
| during wave 1 | nothing new if the runs pass their first-minutes checks; a stop comes to the user if micro-batch 4 does not fit, the projection is over the rendered cap, data wait is over 10% of step time, free space would fall under 40 GB per running job, a model does not fit on the L4, or GPU use stays under 40% at the most environments tried | `RUN-stage2-train.md` step 8, `RUN-stage2-eval.md` step 8 |
| after each checkpoint's evaluation | each `ckpt.pt` deletion or batch (only after evaluation loaded `model.pt`, sha256 logged, backup copy verified) | lead's window; `_standards/08`: agents never delete PVC data without asking |
| before the first deletion | the backup to Ceph S3 (the user said "sure a backup is fine"); its setup (bucket, credentials as a Secret) is not in these manifests, and if it is awkward it comes back to the user | separate task |
| before wave 2's evaluations | the measured evaluation cost of the whole design, and any scope cut (no cut was pre-registered) | lead's window; dated PLAN amendment stating what had been seen |
| if a checkpoint is more than 1% short on clean rollouts | whether it still gets its LIBERO-Plus evaluation (open in the PLAN) | lead's window |
| if the outcome is a complete three-seed I | the extension's Jobs (allowed in the rule; the Jobs still count toward the gate) | lead's window |
