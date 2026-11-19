# Stage 2 wave plan: one training seed per configuration, a proposal for the user's sweep OK

First written 2026-11-17 by a worker, Mac only; cut to the one-seed first pass on 2026-11-19 by a worker, Mac only. Nothing has been submitted. This is the order, the Job counts and the GPU counts that PLAN Order of work step 5 asks the user to OK before the first training run (the Nautilus gate in `STANDARDS.md` line 49: more than 8 GPUs at once, or a sweep of more than 20 Jobs, is a stop). How to render and submit each Job: `RUN-stage2-train.md` and `RUN-stage2-eval.md`. Labels: arithmetic is speculative unless a measured input is named; the limits are the PLAN's.

Source of the cut: PLAN "### 2026-11-18: one training seed per configuration for now, on the mentor's advice relayed by the user; wave 2 deferred" (PR #63, merged as e2193aa), points 2 (a) and 5. The earlier version of this file planned 17 training runs and 58 Jobs in two waves; that is no longer the plan.

## In plain words

Five new training runs, one per configuration, all with training seed 0: `dinov3`, `siglip2`, `c-radiov4`, `dinov3+siglip2` and `dinov3+c-radiov4`. The no-encoder model (`none`) seed 0 is stage 1's run and is not trained again. All six seed 0 checkpoints get a smoke check, clean LIBERO (five of them; `none`'s was stage 1's) and the full LIBERO-Plus benchmark. One limit shapes the order: at most 8 GPUs at once for everything we run, training and evaluation together. Wave 1 runs in two parts with the same seed and the same source commit. Part A trains the three configurations that do not read the SigLIP 2 feature cache and can start once the user gives the go. Part B trains the two that do read it, and waits until the SigLIP 2 cache is repaired and has passed its value check and its freshness check. LIBERO-Plus evaluation, not training, is likely the larger cost, and its size is only known after the first one (PLAN amendment 2026-11-17 item 1).

## Limits used (sources)

- **GPUs**: at most 8 at once, training and evaluation pods together (`STANDARDS.md` line 49; PLAN Compute "GPUs at once"). Every Job of this sweep is labelled `experiment=stacking-pairings`, and so are its pods.
- **Disk**: the second volume `stacking-encoders-runs` exists (1Ti, `rook-cephfs`, ReadWriteMany, created 2026-11-18 and empty then; LOG 2026-11-17 entry, "Second volume"), so training runs no longer have to be split by free space. The old reason for splitting wave 1 into sub-waves 1a and 1b (about 190 GiB free on `stacking-encoders-proxy`, room for 4 training runs at once) no longer applies; render with `--runs-pvc stacking-encoders-runs`. Peak is at most 5 x 40 GB = 200 GB while all five train (40 GB per running training job, PLAN Hand-off step 4) and 5 x 26.7 GB = about 134 GB for five finished runs if no `ckpt.pt` is deleted (about 20 GB `ckpt.pt` + about 6.7 GB `model.pt` per run; stage 1 PLAN lines 83 and 87), both under 1Ti (arithmetic). The free-space stop (40 GB per running job, `RUN-stage2-train.md` steps 5 and 8) is unchanged. A `ckpt.pt` may be deleted only as in PLAN amendment 2026-11-15 later, item 3 (finished, evaluation loaded `model.pt`, sha256 logged, backup copy verified, and the user's OK).
- **Card set**: training on L40 or L40S (stage 1's set); every evaluation pod on L4.
- **Jobs at once**: far under NRP's 400.

## Wave 1, part A and part B (the design locks when the first training run starts)

Wave 1 is the only wave now. Both parts use the same seed (0) and the same source commit `$C` (`RUN-stage2-train.md` step 2), so the five runs differ only in configuration.

| part | training Jobs (GPUs) | when it may start | evaluation at the same time (L4 GPUs) |
|---|---|---|---|
| A | `dinov3`, `c-radiov4`, `dinov3+c-radiov4` (3) | after the user's go for the sweep. These three do not read `feats/siglip2`; the DINOv3, C-RADIOv4 and VAE caches passed the value check (LOG 2026-11-19: every episode of those blocks, 0 gaps) | smoke of `none` (LIBERO-Plus timing shard only, 1), then LIBERO-Plus of `none` seed 0 at parallel 4 (4); then each finished run's smoke, clean and LIBERO-Plus |
| B | `siglip2`, `dinov3+siglip2` (2) | only after the SigLIP 2 repair, the value check and the freshness check have **all** passed and the LOG records that the hold on `feats/siglip2` is lifted. Until then no Job reads `feats/siglip2` for training or evaluation (LOG 2026-11-18 and 2026-11-19). Each of those steps has its own go from the user | the same, with whatever is free |

Part B does not wait for part A to finish; it waits only for those three conditions and for the GPU count. If part B is ready before part A is submitted, all five start together (5 training GPUs + 3 L4 pods for `none`'s LIBERO-Plus). Part A is not held back for part B: a delay in the repair does not delay part A.

Why this order: part A holds the three configurations that the pattern in PLAN amendment point 3 needs without SigLIP 2 (`dinov3` alone, `c-radiov4` alone and the harmful pair), so the cache repair is not on its path. This is a choice of order, not of scope: part B runs in full.

Evaluations in wave 1, per finished checkpoint: smoke (1 L4 pod; unmeasured for the encoder configurations), clean (4 L4 pods; stage 1: about 1.1 h each, PLAN "Clean evaluation time per checkpoint on L4"), gate, then LIBERO-Plus (16 shards, parallel set by the GPUs left). **No verdict, rule change, recipe change or change of what is trained from wave 1 results** (PLAN Decision rule item 6; PLAN amendment 2026-11-18 point 2 (b): a one-seed result is reported as "same pattern as the paper or not", with no yes/no verdict).

**Wave 1 Jobs: 22** (5 training, 6 smoke: five configurations plus `none`'s timing shard, 5 clean, 6 LIBERO-Plus including `none` seed 0). Pods: 5 + 6 + 20 + 6 x 16 = 127 (retries not counted).

## Not planned now: more seeds

Seeds 1 and 2 (12 runs) are deferred by the PLAN amendment of 2026-11-18, not dropped. Nothing of them is rendered, sized or submitted, and this file plans none of it. Whether to add seeds, and for which configurations, is the user's decision after the one-seed results are seen, made by a dated PLAN amendment (amendment point 3, "Constraint on adding seeds later"). A new wave plan and a new sweep OK would be needed then.

## Every transition, counted against the 8-GPU cap (PR #60 review item 3)

The rule behind all rows: at every moment, training pods + L4 pods (smoke, clean, LIBERO-Plus) of this sweep + any other stacking pod is at most 8, and every submit or `parallelism` patch is preceded by the GPU count of `RUN-stage2-train.md` step 5. A Job's pods count from the moment they are Pending (a Pending pod can start at any time). Clean Jobs take `--parallel` 1 to 4 and LIBERO-Plus Jobs `--parallel` 1 to 16, so evaluation always fits into what training leaves; when it does not fit, it waits.

| moment | training pods | L4 pods allowed | total at most |
|---|---|---|---|
| part A starts (3 training) | 3 | 1 (`none` smoke, LIBERO-Plus timing shard only) | 4 |
| `none`'s smoke has ended, its rate measured | 3 | 4 (`none` seed 0 LIBERO-Plus, `--parallel 4`), plus 1 spare for a finished run's smoke | 8 |
| one part A run ends (and so on for each) | 2, 1, 0 | its smoke (1), then its clean (`--parallel` up to what is free), then its LIBERO-Plus; `none`'s LIBERO-Plus keeps its 4 | 8 |
| part B starts while part A still trains (all five training) | 5 | 3 (lower `none`'s LIBERO-Plus `parallelism` to 3, let running clean pods finish, no new evaluation Job until the count allows) | 8 |
| part B starts after part A has ended | 2 | up to 6 | 8 |
| part B runs end | 1, 0 | up to 7, then 8 (`kubectl patch` parallelism up if wanted) | 8 |

Lowering a LIBERO-Plus Job's `parallelism` below half the rendered value stretches its run past the Job deadline's margin; re-render it with the new `--parallel` (same `EVAL_TRY`, same `--shards`: finished shards are skipped) instead of only patching (PR #60 review item 7).

## Totals and the gate

| | Jobs | pods (no retries) |
|---|---|---|
| training | 5 | 5 |
| smoke (incl. the LIBERO-Plus timing shard) | 6 | 6 |
| clean LIBERO (4 shards each) | 5 (`none` seed 0 was done in stage 1) | 20 |
| LIBERO-Plus (16 shards each, proposed) | 6 | 96 |
| **all** | **22** | **127** |

Over 20 Jobs, so the user's OK for the sweep as a whole is needed (`STANDARDS.md` line 49). The PLAN's extension (Decision rule item 7) belongs to a three-seed outcome and is not in play with one seed (PLAN amendment 2026-11-18, point 3).

Time and cost (speculative; every input is a figure already in the PLAN, named here, and is recomputed for 5 training runs and 6 checkpoints):
- Training, at least: about 118 L40S-hours (5 x 23.54, the `none` sizing figure), or about 120 (5 x 24.07, the `dinov3` sizing figure); if every run lands on an L40 at stage 1's measured `none` rate, about 161 L40-hours (5 x 32.18, 32.2 h per run, stage 1 LOG via PLAN "Card-hours per training run"). The pairs and single SigLIP 2 and C-RADIOv4 are unmeasured and expected slower (PLAN Prediction 7); no encoder configuration has been timed in a full run on any card. (PLAN amendment 2026-11-18 point 5 gives the same three totals.)
- Clean evaluation: about 22 L4 GPU-hours (5 x 4.4, stage 1's measured `none` rate, PLAN "Clean evaluation time per checkpoint on L4"); the encoder configurations are unmeasured.
- Smoke per configuration: unmeasured.
- LIBERO-Plus: unmeasured. For scale only, PLAN amendment 2026-11-18 point 5 gives about 300 to 320 L4 pod-hours at the clean rate and about 840 at the timing-only rate for 6 checkpoints (it is not an estimate). The timing shard and `none` seed 0's evaluation replace this with measured numbers.

## What the user must OK, step by step

| step | needs from the user | where |
|---|---|---|
| before any Job of the sweep (part A or part B) | the go for the sweep as a whole: this plan, 22 Jobs, never more than 8 GPUs at once, L40/L40S for training and L4 for evaluation. **No Job is created without this go, given in the experiment lead's window** | lead's window (PLAN Order of work step 5) |
| before part B | the go for each of the three steps that lift the hold on `feats/siglip2`: the repair Job, then the value check, then the freshness check; part B starts only after the LOG records that the hold is lifted | lead's window; `RUN-siglip2-repair.md`, `RUN-cache-value-check.md`, `RUN-cache-fresh-check.md` |
| during wave 1 | nothing new if the runs pass their first-minutes checks; a stop comes to the user if micro-batch 4 does not fit, the projection is over the rendered cap, data wait is over 10% of step time, free space would fall under 40 GB per running job, a model does not fit on the L4, or GPU use stays under 40% at the most environments tried | `RUN-stage2-train.md` step 8, `RUN-stage2-eval.md` step 8 |
| after each checkpoint's evaluation | each `ckpt.pt` deletion or batch (only after evaluation loaded `model.pt`, sha256 logged, backup copy verified) | lead's window; `_standards/08`: agents never delete PVC data without asking |
| before the first deletion | the backup to Ceph S3 (the user said "sure a backup is fine"); its setup (bucket, credentials as a Secret) is not in these manifests, and if it is awkward it comes back to the user | separate task |
| when the first LIBERO-Plus cost is measured | the measured evaluation cost, and any scope cut (no cut was pre-registered; the user decides it by a dated amendment) | lead's window |
| if a checkpoint is more than 1% short on clean rollouts | whether it still gets its LIBERO-Plus evaluation (open in the PLAN) | lead's window |
| after the six checkpoints' one-seed results | whether to add seeds, start stage 3 or stop (PLAN amendment 2026-11-18, point 3); nothing is pre-committed | lead's window |
