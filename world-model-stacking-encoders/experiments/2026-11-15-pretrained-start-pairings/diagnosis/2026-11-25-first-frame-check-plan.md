# Plan: first-frame feature check (are the encoder inputs in a rollout farther from training than the VAE inputs?)

2026-11-25, [agent, lead, planning], branch base `6bdff16`. A diagnostic check inside stage 2 (`../PLAN.md`), written and merged before any code or Job for it exists. It changes no stage 2 rule, result, checkpoint or configuration. Labels: **verified** (read this session in the named file), **recalled**, **speculative**.

## In plain words

Each single-encoder model scores below the no-encoder model. The no-GPU diagnosis (`2026-11-25-feature-path.md` in this folder; PR #91, open at the time of writing) ruled out a failure to fit and a bug in the feature code, and left three explanations: (B) the pictures differ between training (compressed 512 x 512 dataset video shrunk to 224) and rollouts (256 x 256 simulator renders shrunk to 224) and the encoders react to that more than the VAE does; (A) the model with encoders copes worse off the demonstrations (our fusion choices or the 8-layer model); or training-seed variance. This check measures one thing: at the first frame of a rollout, how far each encoder's features sit from the training features of the same task, with the VAE's latents as the yardstick. It can point to B or away from it. It cannot prove either, and it says nothing about A against seed variance.

## Where the method comes from

The procedure is the experiment session's proposal in the diagnosis note (section 3, "The discriminating check"). It is our own; no paper is followed, and no paper gives the bar below. The bar is the planning lead's choice, accepted by the user in chat on 2026-11-25 (see "Sign-off").

## What exists when this is written

Stage 2 results already seen by the planning lead (verified, stage 2 `LOG.md` on main at `6bdff16`): clean LIBERO `none` 92.55, `dinov3` 85.7, `c-radiov4` 76.6, `siglip2` 72.9; LIBERO-Plus `none` 50.83, `c-radiov4` 44.61; and the per-task clean table in the diagnosis note. No measurement of this check exists: no live feature has been compared with a training feature on real frames (verified, the note's items 4 and 5).

## Design

Inputs are real data, the real simulator, the real encoder weights and the checkpoints' own normalization statistics. Four channel groups: the VAE (48 channels, raw, as the model sees them) and DINOv3, C-RADIOv4 and SigLIP 2 (each z-scored with the statistics in its single-encoder checkpoint's config; the note's item 18 found them identical across runs).

1. **Live frames.** For each of the 40 clean LIBERO tasks and init states 0 to 4 (200 frames), build the first policy frame exactly as a rollout does (`rollout.libero_env`, reset, `set_init_state`, 30 dummy steps, `rollout.process_obs`) and run `rollout.Featurizer` at batch 1 on an L4, the evaluation card. No policy step is taken and no success is recorded.
2. **Training frames.** Frame 0 of every training episode of the same task. Encoders: the cached features, z-scored. VAE: frame 0 encoded alone (a one-frame clip through the training-side decode and `to_224`), not the cached clip-encoded latent.
3. **Distance.** For group g with C channels, d(x, y) = the L2 distance between the flattened (C, 14, 28) arrays divided by sqrt(C x 14 x 28).
4. **Per frame.** For live frame i of task k: r(i, g) = (distance from i to its nearest training frame 0 of task k) / ref(k, g), where ref(k, g) is the median, over the training frame 0s of task k, of each one's distance to its nearest other training frame 0 of task k.
5. **Per group.** m(g) = the median of r(i, g) over the 200 live frames.
6. **The statistic.** For each encoder e: **R(e) = m(e) / m(VAE)**. Uncertainty: a 95% percentile interval from 1,000 bootstrap draws that resample the 40 tasks with replacement (seed 0), reported beside each R(e).

## Predictions (written before any measurement)

The lead has no confident prediction (speculative): m(VAE) above 1, because the live frame comes after 30 settling steps from an evaluation init state and through a different pixel path. If B is the cause, R(e) well above 1, largest for SigLIP 2 and C-RADIOv4 (the lowest on clean LIBERO). If it is not, R(e) near or below 1.

## Reading rule (fixed before the run; read per encoder, from the point estimate)

- **R(e) at or below 1.2: "near".** The live inputs of encoder e are about as far from training as the VAE's. Counts against B for that encoder.
- **R(e) at or above 1.5: "clearly larger".** B is the lead for that encoder.
- **Between 1.2 and 1.5: "unclear".**
- If the bootstrap interval crosses 1.2 or 1.5, the zone is reported as borderline in those words. The zone itself still comes from the point estimate.

## What each outcome leads to (written before the run)

- **Any encoder "clearly larger":** the planning lead proposes the settling run for that encoder and `none`: clean rollouts with training-like frame preparation against the normal rollouts (a code change and a GPU evaluation; its own design by PR and the user's go). No fix to the rollout frame path is made from this check alone. Any such fix would change the evaluation protocol with test outcomes known, so it is the user's decision by a dated amendment to `../PLAN.md`.
- **All three "near":** B is not supported. No settling run is proposed from this check. What remains (A or seed variance) goes to the user with the choices already open: fusion variants, a deeper model, a second training seed.
- **Mixed or "unclear":** reported as it is; the user decides whether the settling run is worth doing anyway.
- In every case the stage 2 decision rule, the one-seed reporting rule (amendment 2026-11-18) and every stage 2 result stand unchanged, and nothing from this check goes into README key findings without the user's sign-off.

## Also reported, descriptive only (no bar, no decision made from them)

- The median and 90th percentile of r per group, per suite and over all 200 frames.
- Per encoder, the Spearman rank correlation over the 40 tasks between the per-task ratio (median r of the encoder over the task's 5 frames, divided by the VAE's) and the per-task clean success drop (`none` minus the encoder, from the committed clean records). Under B the tasks with the farthest features would be the tasks that collapse (speculative). Five frames per task, so it is noisy.
- The single-frame VAE latents of step 2 against the cached clip-encoded ones (largest absolute difference and relative L2). This tests the note's item 15, a recalled claim. If they differ, it is reported to the user at once, because it would affect `none` too; what follows is a separate decision.
- The live z-scores of DINOv3 channel 757 (median and largest absolute value; the note's item 19).

## Limits (stated before the run)

- A distance between inputs is not the model's sensitivity to them. "Near" does not prove the model is unaffected, and "clearly larger" does not prove harm.
- The encoders have 1,024 or 1,152 channels and the VAE 48 (the note, section 1), and the encoders are built to ignore small pixel changes while the VAE is not, so r may behave differently across groups for reasons unrelated to B (speculative). The per-task reference distance removes scale, not this.
- Only the first frame is measured. Features later in a rollout, after the robot has moved off the demonstrations, are not.
- The 1.2 and 1.5 edges are our own choice, with no outside calibration.
- One training seed per configuration; this check does not change that.

## Test-set use

The live frames are first frames of clean LIBERO evaluation init states (test-set inputs). No rollout is run and no success outcome is produced. The descriptive correlation uses clean outcomes already read and logged. Logged in the stage 2 `LOG.md` at submit. No hyperparameter, checkpoint, recipe or threshold is chosen from it.

## Compute

One small Job (speculative: under one L4 GPU-hour, not sized; the experiment session sizes it and checks the utilization rules of `_standards/08`). The diagnosis note says an L4 is needed for the live features (its item 13); the experiment session's message of 2026-11-25 said "CPU only". The experiment session resolves this before preparing the Job; a CPU run of the encoders would add its own rounding and is not this design. Caches and the dataset are on PVC `stacking-encoders-proxy`.

## Hand-off

The experiment session writes the script and the Job by PR and submits only on the user's go in its own window. The result goes in the stage 2 `LOG.md` and a short README in this folder, with every number, the commit and the Job's times.

## Sign-off

The user, in the planning window, 2026-11-25, to the lead's question "Run the measuring check with that bar?", after the lead described three zones ("about as far from training as the VAE's" / "at least half again as far" / "in between is unclear"): **"1. yes"**. The 1.5 edge is that wording. The 1.2 figure for "about as far", the bootstrap interval and the descriptive items were not spelled out in chat; they are the lead's and stand only with the user's merge OK on this PR.

## Amendments

(none)
