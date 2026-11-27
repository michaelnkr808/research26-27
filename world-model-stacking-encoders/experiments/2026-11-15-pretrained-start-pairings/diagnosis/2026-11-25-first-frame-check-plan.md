# Plan: first-frame feature check (are the encoder inputs in a rollout farther from training than the VAE inputs?)

2026-11-25, [agent, lead, planning], branch base `6bdff16`. A diagnostic check inside stage 2 (`../PLAN.md`), written and merged before any code or Job for it exists. It changes no stage 2 rule, result, checkpoint or configuration. Labels: **verified** (read this session in the named file), **recalled**, **speculative**.

## In plain words

Each single-encoder model scores below the no-encoder model. The no-GPU diagnosis (`2026-11-25-feature-path.md` in this folder; PR #91, open at the time of writing) ruled out a failure to fit and a bug in the feature code, and left three explanations: (B) the pictures differ between training (compressed 512 x 512 dataset video shrunk to 224) and rollouts (256 x 256 simulator renders shrunk to 224) and the encoders react to that more than the VAE does; (A) the model with encoders copes worse off the demonstrations (our fusion choices or the 8-layer model); or training-seed variance. This check measures one thing: at the first frame of a rollout, how far each encoder's features sit from the training features of the same task, with the VAE's latents as the yardstick. It can point to B or away from it. It cannot prove either, and it says nothing about A against seed variance.

## Where the method comes from

The procedure is the experiment session's proposal in the diagnosis note (section 3, "The discriminating check"). It is our own; no paper is followed, and no paper gives the bar below. The bar is the planning lead's choice, accepted by the user in chat on 2026-11-25 (see "Sign-off").

## What exists when this is written

Stage 2 results already seen by the planning lead (verified, stage 2 `LOG.md` on main at `6bdff16`): clean LIBERO `none` 92.55, `dinov3` 85.7, `c-radiov4` 76.6, `siglip2` 72.9; LIBERO-Plus `none` 50.83, `c-radiov4` 44.61; and the per-task clean table in the diagnosis note. No measurement of this check exists: no live feature has been compared with a training feature on real frames (verified, the note's items 4 and 5). One earlier result in this direction bears on it (verified, direction `README.md`, Key findings, "Absolute probe R² values in the tally are specific to the stored renderer"): in the perturbed-frames experiment the same DINOv3 probe lost about 0.05 R² between the stored frames and our re-render of the same moments. That was the older 128-pixel stored-frame path, not this one, so it is a reason to look, not a measurement of this path.

## Design

Inputs are real data, the real simulator, the real encoder weights and the checkpoints' own normalization statistics. Four channel groups: the VAE (48 channels, raw, as the model sees them) and DINOv3, C-RADIOv4 and SigLIP 2 (each z-scored with the statistics in its single-encoder checkpoint's config; the note's item 18 found them identical across runs).

1. **Live frames.** For each of the 40 clean LIBERO tasks and init states 0 to 4 (200 frames), build the first policy frame exactly as a rollout does (the LIBERO-Plus environment at commit `4976dc3`, `rollout.libero_env`, reset, `set_init_state`, 30 dummy steps, `rollout.process_obs`) and run `rollout.Featurizer` at batch 1 on an L4, the evaluation card (the note, section 3, step 1). No policy step is taken and no success is recorded.
2. **Training frames.** Frame 0 of every training episode of the same task (the task of each episode from `/mnt/proxy/feats/meta/<suite>.npz`, as in the note). Encoders: the cached features, z-scored. VAE: frame 0 encoded alone (a one-frame clip through the training-side decode and `to_224`), not the cached clip-encoded latent. Statistics: for encoder e, the `feature_stats` in the config of its own single-encoder seed 0 checkpoint (`pairings-<e>-s0`), on both the live and the training side.
3. **Distance.** For group g with C channels, d(x, y) = the L2 distance between the flattened (C, 14, 28) arrays divided by sqrt(C x 14 x 28).
4. **Per frame.** For live frame i of task k: r(i, g) = (distance from i to its nearest training frame 0 of task k) / ref(k, g), where ref(k, g) is the median, over the training frame 0s of task k, of each one's distance to its nearest other training frame 0 of task k. "Nearest" is the smallest distance; ties need no rule. A task with fewer than 2 training episodes, or with ref(k, g) = 0 for any group, is left out of every group's statistic and named in the report (none is expected: 1,712 training episodes over 40 tasks).
5. **Per group.** m(g) = the median of r(i, g) over all live frames pooled (200 if no task is left out), each frame counted once.
6. **The statistic.** For each encoder e: **R(e) = m(e) / m(VAE)**. Uncertainty: a 95% percentile interval (2.5th to 97.5th) from 1,000 bootstrap draws (seed 0). Each draw resamples the tasks with replacement, takes all 5 live frames of every drawn task (a task drawn twice counts twice), and recomputes m(e), m(VAE) and R(e) from them; ref(k, g) is not recomputed. Reported beside each R(e).

## Predictions (written before any measurement)

The lead has no confident prediction (speculative): m(VAE) above 1, because the live frame comes after 30 settling steps from an evaluation init state and through a different pixel path. If B is the cause, R(e) well above 1, largest for SigLIP 2 and C-RADIOv4 (the lowest on clean LIBERO). If it is not, R(e) near or below 1. These predictions carry no consequence; only the reading rule does.

## Reading rule (fixed before the run; read per encoder, from the point estimate)

- **R(e) <= 1.2: "near".** The live inputs of encoder e are about as far from training as the VAE's, or closer (any value below 1 is "near" too). Counts against B for that encoder.
- **R(e) >= 1.5: "clearly larger".** B is the lead for that encoder.
- **1.2 < R(e) < 1.5: "unclear".**
- If the bootstrap interval crosses 1.2 or 1.5, the zone is reported as borderline in those words. The zone and the next step still come from the point estimate; a borderline interval changes neither.
- **DINOv3 only:** R is also computed with channel 757 left out (the near-constant channel, the note's item 19). The zone uses all channels, as the model sees them. If the zone differs with channel 757 left out, DINOv3 is read as "unclear" and the report says the reading depends on that channel.

## What each outcome leads to (written before the run)

Exactly one of the three cases applies, checked in this order.
- **1. At least one encoder "clearly larger"** (whatever the others are): the planning lead proposes the settling run for that encoder and `none`: clean rollouts with training-like frame preparation against the normal rollouts (a code change and a GPU evaluation; its own design by PR and the user's go). No fix to the rollout frame path is made from this check alone. Any such fix would change the evaluation protocol with test outcomes known, so it is the user's decision by a dated amendment to `../PLAN.md`.
- **2. All three "near":** B is not supported. No settling run is proposed from this check. What remains (A or seed variance) goes to the user with the choices already open: fusion variants, a deeper model, a second training seed.
- **3. Anything else** (no encoder "clearly larger", at least one "unclear"): reported as it is; the user decides whether the settling run is worth doing anyway.
- A settling run, if one follows, uses the standard clean evaluation (all 50 init states per task), and nothing in its design is chosen from the 200 frames of this check beyond the decision to run it.
- In every case the stage 2 decision rule, the one-seed reporting rule (amendment 2026-11-18) and every stage 2 result stand unchanged, and nothing from this check goes into README key findings without the user's sign-off.

## Also reported, descriptive only (no bar, no decision made from them)

- The median and 90th percentile of r per group, per suite and over all 200 frames.
- Per encoder, the Spearman rank correlation over the 40 tasks between the per-task ratio (median r of the encoder over the task's 5 frames, divided by the VAE's) and the per-task clean success drop (`none` minus the encoder, from the committed clean records). Under B the tasks with the farthest features would be the tasks that collapse (speculative). Five frames per task, so it is noisy.
- The single-frame VAE latents of step 2 against the cached clip-encoded ones (largest absolute difference and relative L2). This tests the note's item 15, a recalled claim. If they differ, it is reported to the user at once, because it would affect `none` too; what follows is a separate decision. R(e) stays readable either way, because this check encodes the VAE frame alone on both sides.
- The live z-scores of DINOv3 channel 757 (median and largest absolute value; the note's item 19).

## Limits (stated before the run)

- A distance between inputs is not the model's sensitivity to them. "Near" does not prove the model is unaffected, and "clearly larger" does not prove harm.
- The encoders have 1,024 or 1,152 channels and the VAE 48 (the note, section 1), and the encoders are built to ignore small pixel changes while the VAE is not, so r may behave differently across groups for reasons unrelated to B (speculative). The per-task reference distance removes scale, not this.
- **The VAE yardstick does not cancel scene differences.** The live frame comes after 30 settling steps from an evaluation init state; the training frame is demonstration frame 0. The encoders respond to object and arm pose, the VAE to pixel layout, so these differences can move the groups by different amounts. Dividing by ref(k, g), each group's own spacing between demonstrations of the task, takes out part of this and not all. So "clearly larger" can come from scene differences alone, and the check cannot separate them from the pixel path. Here "pixel source" includes any difference in how the simulator renders the scene.
- The reference distance leaves the frame itself out and the live distance does not, and nearest-neighbour spacing depends on a group's dimension, so r is not exactly 1 even with no shift, by an amount that can differ by group. There is no control in this design that gives the value of R under "no pixel difference" or under a known one.
- Encoder rounding at batch 1 on the L4 is a live-only difference (the note's item 13) and is inside R.
- Only the first frame is measured. Features later in a rollout, after the robot has moved off the demonstrations, are not.
- The 1.2 and 1.5 edges are our own choice, with no outside calibration.
- One training seed per configuration; this check does not change that.

## Test-set use

The live frames are first frames of clean LIBERO evaluation init states (test-set inputs). No rollout is run and no success outcome is produced. The descriptive correlation uses clean outcomes already read and logged. Logged in the stage 2 `LOG.md` at submit. No hyperparameter, checkpoint, recipe or threshold is chosen from it. These 200 first frames will have been looked at before any settling run; that run's design says so and takes nothing from them (see the outcome list).

## Compute

One small Job (speculative: under one L4 GPU-hour, not sized; the experiment session sizes it and checks the utilization rules of `_standards/08`). The diagnosis note says an L4 is needed for the live features (its item 13); the experiment session's message of 2026-11-25 said "CPU only". The experiment session resolves this before preparing the Job; a CPU run of the encoders would add its own rounding and is not this design. Caches and the dataset are on PVC `stacking-encoders-proxy`.

## Hand-off

The experiment session writes the script and the Job by PR and submits only on the user's go in its own window. Its PR and LOG entry pin the source commit, the image, the environment commit, the dataset path and the three checkpoint paths with their sha256. The script is merged before the Job runs and before any number is read. Before the full Job, a dry run on a few frames: the live frame and the training frame 0 of the same task are saved and looked at side by side (orientation, camera, colours), per `_standards/07` "Sanity checks before scaling". The result goes in the stage 2 `LOG.md` and a short README in this folder, with every number, the commit and the Job's times.

**A question for the experiment session before it writes code (from the review of this PR):** can the simulator be set to a demonstration's own frame-0 state and rendered as a rollout renders it, as the perturbed-frames experiment did for the older data? If so, the live and training frames would show the same scene, and the distance between them would be the pixel path alone. That matched version is not in this design. If it is feasible at similar cost, the experiment session tells the planning lead, and it is added only by a dated amendment below, with the user's OK, before any number of this check is read.

## Sign-off

The user, in the planning window, 2026-11-25, to the lead's question "Run the measuring check with that bar?", after the lead described three zones ("about as far from training as the VAE's" / "at least half again as far" / "in between is unclear"): **"1. yes"**. The 1.5 edge is that wording. The 1.2 figure for "about as far", the bootstrap interval, the channel 757 rule, the order of the three outcome cases and the descriptive items were not spelled out in chat; they are the lead's and stand only with the user's merge OK on this PR.

## Amendments

### 2026-11-25: per-channel value ranges added as a descriptive item (the mentor's question; no number of this check exists yet; no change to the statistic, the reading rule or the outcome list)

Written by the planning lead, branch base `5ad2879`, before any script or Job for this check exists. The design above is unchanged.

**Why.** The mentor's reply on the stage 2 results (direction `LOG.md`, 2026-11-25, "the mentor's Slack reply") asks, among her theories: "Is the normalization being done? Or are the features in very different ranges". Normalization is done on both sides (verified in code: Wan's latent scale on the VAE channels, `src/extract_fastwam.py:99`, `src/rollout.py:229`; the per-channel z-score on encoder channels, the diagnosis note's items 17 to 19). A first read from the committed training logs (direction `LOG.md`, 2026-11-25, "the mentor's range question") puts the overall scale of the VAE channels and the z-scored encoder channels about level (predict-zero loss ratios of 1.03 to 1.13 on the first batch; an inference from logged losses, not a measurement of the values). Nobody has looked channel by channel. This check already loads both kinds of values, so the numbers are added here. The user, planning window, 2026-11-25, to "Add the channel-by-channel range numbers to the first-frame check?": **"1. yes sure we can do this"**.

**What is computed.** Four groups as in the Design. Values: VAE channels after Wan's scale, encoder channels after the z-score, each rounded through float16 as the cache and the rollout do, on both sets.
- **Two sets of frames.** (T) the training frame 0s of step 2, every one counted once (about 1,700). The encoder values are the cached training inputs. The VAE values are the single-frame encodings of step 2, the way a rollout encodes, not the cached clip-encoded training inputs. (L) the live first frames of step 1, every one counted once (200). A task left out under step 4 is left out of both sets.
- **Per channel.** For a group, a set and a channel, pool every value of that channel over all frames of the set and all 14 x 28 positions (both cameras together, as the model receives the grid). From that pool: the arithmetic mean, the population standard deviation (divide by N), and the largest absolute value.
- **Per group.** The overall mean square: the mean of the squared values over all channels, frames and positions of the group in that set.
- **Reported per group and set:** the median, smallest and largest per-channel standard deviation; the median and largest absolute per-channel mean; the largest absolute value; the overall mean square; and, per encoder group, the ratio of its overall mean square to the VAE group's on the same set. The full per-channel table goes in the result file.
- **Flags, fixed now (the lead's own choices, no paper behind them).** A channel is flagged, with the test named, if:
  - (a) on set T its standard deviation is below 0.25 or above 4, or its absolute mean is above 2;
  - (b) its standard deviation on set L is below a quarter of, or above four times, its standard deviation on set T. If the standard deviation on set T is below 1e-6, test (b) is not applied and the channel is flagged as constant on set T;
  - (c) its mean on set L differs from its mean on set T by more than 2 standard deviations of set T (same exception as in (b)).
  The report lists flagged channels by group and index, with the count and the share of the group's channels per test.
- **One wording, fixed now.** A group pair is called "in very different ranges" if an encoder group's overall mean square on set T is more than 4 times, or less than a quarter of, the VAE group's (a factor of 2 in root mean square). Otherwise the report says the overall ranges are comparable and gives the ratio.
- The lead's expectation (speculative): DINOv3 channel 757, whose raw training spread is near zero (the diagnosis note's item 19), is the likeliest channel to be flagged under (b) or (c).

**Reading.** Descriptive only. No zone, no next step and no change to R(e) follows from these numbers, whatever the count of flags. The flag list, the counts and the group ratios go to the user as they are; what follows, if anything, is a separate decision.

**Limits.**
- Means, standard deviations and the largest absolute value only; the shape of each channel's distribution is not described.
- "As the model receives them" holds for set L and for the encoders on set T. The VAE values on set T are rollout-style single-frame encodings; the comparison with the cached clip-encoded latents (under "Also reported, descriptive only" in the design; the note's item 15, a recalled claim) says how far those are from the real training inputs.
- Current-frame inputs only. The noised future-step inputs and the future targets (means over several frames) are not loaded by this check.
- Equal ranges at the input do not mean equal influence: the model's learned input projection can rescale any channel. This item answers the mentor's question about ranges, not how the model weighs the groups.
- The VAE channels have no unit-variance target, so tests (a) to (c) are looser for a VAE-against-encoder difference than the group ratio; the ratio is the direct answer to that question.
- Set T has about 1,700 frames and set L 200; each frame counts once within its own set.

**Sign-off.** The flag thresholds, the mean-shift test and the "very different ranges" wording were not spelled out in chat; they are the lead's and stand only with the user's merge OK on this PR.

**Cost.** None beyond the check: the same arrays, a few more summary statistics.

### 2026-11-27: eight readings of the arithmetic, fixed before any run (the experiment session's questions from PR #100; no number of this check exists; no change to the statistic, the zones or the outcome list)

Written by the planning lead, branch base `4f7a977`. The script for this check is up for review as PR #100 (head `8952520`); nothing has run on the cluster. Its implementer found eight places where the text above does not spell out the arithmetic, took a reading of each, and the experiment session asked the planning lead to confirm or correct them. All eight are confirmed as the reading of the design; two get one more sentence (b, h). The planning lead read the plan and the parts of the script named below, not the whole script.

- **a. DINOv3 without channel 757.** It is treated as a group of its own with C = 1,023: the channel is dropped from the live and the training arrays, the distance of step 3 uses that C, and ref(k, g) and r(i, g) are recomputed for it. Its R divides by the same m(VAE).
- **b. Which groups can leave a task out.** Only the four groups of the Design (VAE, DINOv3, C-RADIOv4, SigLIP 2). The variant of (a) uses the tasks they keep and cannot remove one, so it cannot move the main statistic. If that variant's reference distance is 0 for a task that is kept, the script stops with a failed check and no number is read (as the script on PR #100 does); what follows is a dated amendment, not a silent rule.
- **c. "The bootstrap interval crosses 1.2 or 1.5".** The interval's two ends (2.5th and 97.5th percentile) fall in different zones under the reading rule's edges (at or below 1.2; strictly between; at or above 1.5).
- **d. Percentiles and medians.** numpy's default (linear interpolation between order statistics) for the 2.5th, 97.5th and 90th percentiles, and `numpy.median` for every median.
- **e. The VAE comparison of "Also reported" (the note's item 15).** The relative L2 is the L2 norm of (single-frame latent minus cached clip-encoded latent) divided by the L2 norm of the cached clip-encoded latent. The cached latent is the reference because it is what the model was trained on. Reported over the training frame 0s: the largest and the median of each measure, and the count of episodes where the two latents are identical.
- **f. The bootstrap draw.** One resampled task list per draw, shared by all groups and all encoders, so m(e) and m(VAE) in a ratio come from the same frames. This is what step 6 says for one encoder; it is the same list for all three. The draw is pinned as the script on PR #100 has it (read by this PR's reviewer and by the lead at `8952520`): the kept tasks ordered by suite then task index, K of them; one `numpy.random.default_rng(0)` for all 1,000 draws; each draw is `integers(0, K, K)` indices into that list.
- **g. Decoding training frame 0.** The training-side decode may stop after frame 0 instead of decoding whole episodes, on the condition the script already carries: in the pod it compares that frame bit for bit with frame 0 of a full decode on one episode per suite and exits on any difference. This is a sample of four episodes, not every episode; the report says so.
- **h. The Spearman item.** It is computed on the Mac from the committed clean records and the result file, into a second JSON that carries the result file's sha256, because the clean records are not on the volume. Ties get their average rank and the correlation is the Pearson correlation of the ranks; where either side is constant it is reported as not computable (verified in the script at `8952520`, `rankdata` and `spearman`). It runs over the kept tasks, so n is below 40 if a task is left out, and n is reported. It stays descriptive: no bar, no decision.

Also from the experiment session (2026-11-27): the matched-scene version asked about in the Hand-off still has no route from this data (the LeRobot set stores only the 8-number robot state), so it is not added.

**Sign-off.** These eight readings were not spelled out in chat; they stand only with the user's merge OK on this PR.

**Cost.** None.
