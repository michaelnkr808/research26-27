# Plan: imagined-future test (on training clips, does acting on its own imagined future cost an encoder model more than it costs the no-encoder model?)

2026-11-25, [agent, lead, planning], branch base `47c0c04`. A diagnostic test inside stage 2 (`../PLAN.md`), written and merged before any code or Job for it exists. It changes no stage 2 rule, result, checkpoint or configuration. Labels: **verified** (read this session in the named file), **recalled**, **speculative**.

## In plain words

Each single-encoder model scores below the no-encoder model `none`, although all of them fit the training actions about equally well. One explanation the user and the lead picked on 2026-11-25 (direction `LOG.md`, "hypotheses for why performance fell", candidate 2): in a rollout the model has to imagine the future from noise, the action part looks at that imagined future, and an 8-layer model may imagine 1,024 or 1,152 encoder channels badly. Training never shows this, because in training the future the action part sees is the true future with some noise added. This test runs each trained model the way a rollout runs it, but on training clips, where the true actions and the true future are known. It asks whether the encoder models then act worse than `none`, and if so whether handing them the true future removes the difference. No simulator, no test data, no retraining.

## Where the method comes from

The test is our own; no paper is followed. The inference procedure it copies is the paper's as implemented in `src/proxy_wam.py` (`infer_actions`: 10 joint denoising steps, `INFER_STEPS = 10`, line 70, App. E.3 of liu2026what; verified). The bars are the planning lead's choice (see "Sign-off").

## What is known when this is written

- Verified in code: the action tokens attend to all video tokens in `FastWAMJoint` (`experiments/2026-11-12-pretrained-start-sizing/step2-reading.md` line 76); at inference video and actions start from noise and are denoised jointly, with latent step 0 reset to the observation after every step (`src/proxy_wam.py`, `infer_actions`); in training the video and action noise levels are drawn separately and the future latent steps are the true future at the drawn level (`training_loss`). A rollout executes the first 10 actions of each 32-step chunk (`src/rollout.py` line 74, `REPLAN = 10`).
- Stage 2 results seen by the planning lead: clean LIBERO `none` 92.55, `dinov3` 85.7, `c-radiov4` 76.65, `siglip2` 72.9; LIBERO-Plus `none` 50.83, `c-radiov4` 44.61; the diagnosis note `2026-11-25-feature-path.md` (final training action loss about the same across runs; VAE-channel video loss higher in every encoder run; the per-task clean table). No measurement of this test exists.

## Design

**Checkpoints.** Seed 0 of all six configurations: `none` (`/mnt/proxy/runs/baseline-none-s0/model.pt`), `dinov3`, `c-radiov4`, `siglip2`, `dinov3+c-radiov4`, `dinov3+siglip2` (`/mnt/runs/runs/pairings-<config>-s0/model.pt`), each loaded with its own config. The reading rule covers the three single encoders. The two pairs are computed and reported descriptively only.

**Clips (training data only).** For each of the 40 tasks: its training episodes sorted by id, 5 drawn without replacement with `numpy.random.default_rng(0)` (all of them if a task has fewer than 5), tasks processed in sorted order with one generator. For each drawn episode of length n, three windows, starting at frames 0, floor(n / 3) and floor(2n / 3). 600 clips if every task has 5 episodes. Every checkpoint gets the same clips.

**Inputs.** For each clip, exactly what a training step receives for that window, built by the training data code (`proxy_wam.Data`, `to_device`): the cached latents (VAE channels and the configuration's z-scored encoder channels, 3 latent steps), the T5 context, the proprio state, the normalized 32-step action chunk and the padding masks.

**Starting noise.** Per clip, seeded by (noise seed, clip index): the action noise (32, 7) and the VAE-channel video noise (48, 3, 14, 28) from one generator, so every checkpoint starts from the same action and VAE-channel noise; each encoder's channels from a second generator seeded by (noise seed, clip index, source name). Noise seed A = 0 for everything; noise seed B = 1 is used only for the control below.

**Three arms.** Each is the `infer_actions` loop with its 10 steps and both schedules unchanged; only what the future latent steps (1 and 2) hold differs.
- **I, imagined (what a rollout does).** Unmodified: the future starts as noise and is denoised jointly.
- **T, true future.** Before the first step and after every step, latent steps 1 and 2 are set to the true cached future, noised (same video noise) to the lowest noise level of the inference video schedule, and that level is passed as the video timestep at every step. The action part runs as in arm I.
- **N, no future information.** Latent steps 1 and 2 are held at the starting noise and the video timestep at the schedule's first (highest) level at every step. The action part runs as in arm I.

**Action error.** For a clip and an arm: the mean, over the 7 action dimensions and the valid steps among the first 10 of the chunk (not `action_is_pad`), of the squared difference between the arm's output chunk and the demonstration's normalized chunk. A clip with no valid step among the first 10 is dropped for every checkpoint and counted in the report. E(c, arm) is the mean over the remaining clips, each counted once. The same over all 32 steps is reported beside it, descriptively.

**Statistics, per single encoder e.**
- **Q(e) = E(e, I) / E(none, I).**
- **Q_T(e) = E(e, T) / E(none, T).**
- **Control, N0 = E(none, I) with noise seed B / E(none, I) with noise seed A:** how far the ratio moves from sampling noise alone.
- Uncertainty: a 95% percentile interval for each ratio from 1,000 bootstrap draws (seed 0) that resample the 40 tasks with replacement, taking all clips of a drawn task (a task drawn twice counts twice).

## Predictions (written before any measurement; they carry no consequence)

The lead's guess (speculative): "about the same" for all three single encoders, because every clip here was trained on and all runs fit the training actions equally. If so, the test rules out the simple form of candidate 2 and leaves the form in which imagination fails only in states the model has not seen.

## Reading rule (fixed before the run; per single encoder, from the point estimates)

Step 1.
- **Q(e) <= 1.2: "about the same".**
- **Q(e) >= 1.5: "clearly worse".**
- **1.2 < Q(e) < 1.5: "unclear".**

Step 2, only for an encoder that is "clearly worse".
- **Q_T(e) <= 1.2: "goes away with the true future".**
- **Q_T(e) >= 1.5: "stays with the true future".**
- **1.2 < Q_T(e) < 1.5: "attribution unclear".**

**Noise guard.** If N0 is below 1 / 1.1 or above 1.1, the measure is too noisy for these edges: every zone is reported as "not firm" and case 1 below applies. A bootstrap interval that crosses an edge is reported as borderline in those words and changes neither the zone nor the next step.

## What each outcome leads to (written before the run; exactly one case applies, checked in this order)

1. **The noise guard trips.** Reported as it is. No next step is proposed from this test; the user decides whether a larger clip set is worth running.
2. **At least one encoder is "clearly worse" and "goes away with the true future".** Candidate 2 is supported for that encoder: on inputs it was trained on, acting on its own imagined future costs it clearly more than it costs `none`, and the true future removes the difference. The lead brings the user the choice of a training-side change that reduces the action part's dependence on the imagined future (one existing example: in Fast-WAM's base model the action tokens attend only to the first video frame; `step2-reading.md` line 76, verified). Any such change is a new training experiment with its own design by PR and the user's go.
3. **Otherwise, at least one encoder is "clearly worse".** The deficit is present in inference mode on familiar inputs and the imagined future does not explain it, or the attribution is unclear. Reported; the lead and the experiment session look for the cause in the inference path before anything else is proposed.
4. **All three are "about the same".** The simple form of candidate 2 is not supported. What remains goes to the user with the choices already open: imagination or acting that fails only in unfamiliar states (this test cannot see it), the feature loss crowding out the backbone (candidate 1), the pixel source (the first-frame check), our choices where the paper is silent.
5. **Anything else.** Reported as it is; the user decides.

In every case the stage 2 decision rule, the one-seed reporting rule (`../PLAN.md`, amendment 2026-11-18) and every stage 2 result stand unchanged, and nothing from this test goes into README key findings without the user's sign-off.

## Also reported, descriptive only (no bar, no decision made from them)

- E for every arm and all six checkpoints, on the first 10 steps and on all 32.
- **Recovery share**, per checkpoint: (E(c, N) - E(c, I)) / (E(c, N) - E(c, T)), how much of the benefit of knowing the future the model's own imagination recovers; not computed where the denominator is not positive.
- **Imagination error**, per checkpoint and source group, after arm I: the mean squared difference between the imagined latent steps 1 and 2 and the true cached ones over the valid future steps, beside the mean square of the true values (the error of imagining zero). For the VAE channels also the ratio of each encoder checkpoint's error to `none`'s. This bears on candidate 1 (does the model imagine the picture worse once it must also imagine encoder features), descriptively.
- Q(e) by task, and per encoder the Spearman rank correlation over the 40 tasks between the per-task Q(e) and the per-task clean success drop (`none` minus the encoder, from the committed clean records).
- Q(e) by window position (0, one third, two thirds).

## Limits (stated before the run)

- **Open loop, familiar inputs.** Every clip was trained on, and each is judged from a demonstration state. A rollout visits states no demonstration contains, and its mistakes feed back. "About the same" here does not show the imagined future is harmless in a rollout; it shows only that the harm, if any, needs unfamiliar states.
- The demonstration's action is one valid action among several; squared error against it also charges a model for other valid choices. The same holds for every checkpoint, so the ratios are still comparable, but the floor is not zero.
- Arm T uses a regime training visited rarely: a nearly clean future with an action at any noise level (that low video noise levels are rare under the training sampler is recalled, not re-read). Arm N holds the video at the highest level throughout, also a corner of what training saw.
- The inputs are cached training inputs, not live rollout inputs; the pixel-source question is the first-frame check's, not this test's.
- The 1.2 and 1.5 edges and the 1.1 guard are our own choices; the only calibration is the control N0.
- One training seed per configuration.

## Test-set use

None. Training clips only; no simulator, no evaluation init state. The per-task correlation uses clean outcomes already read and logged.

## Compute

One GPU Job, no simulator (speculative: a few GPU-hours at most on one 24 GB card; not sized). All six checkpoints run on the same card model with the same batch size. The experiment session sizes it from a dry run on 8 clips and checks the utilization rules of `_standards/08`. Caches are on PVC `stacking-encoders-proxy`, the stage 2 checkpoints on `stacking-encoders-runs`.

## Hand-off

The experiment session writes the script in this folder by PR (no edit to `src/`) and submits only on the user's go in its own window. Done checks before the full Job: its arm I output equals `proxy_wam.infer_actions` on the same inputs and noise (largest absolute difference reported, expected 0); the lowest and highest levels of the inference video schedule are printed, with the share of training draws at or below the lowest (from the training sampler); a dry run on 8 clips. Its PR and LOG entry pin the source commit, the image, the six checkpoint paths with their sha256 and the card model. The script is merged before the Job runs and before any number is read. The result goes in the stage 2 `LOG.md` and a short README in this folder.

## Sign-off

The user, planning window, 2026-11-25: to the lead's description of the test ("run each model the way a rollout runs it [on training clips], and compare its actions to the demo's. Then do it again while handing the model the real future") and "Want me to write that design now?": **"2. yes sure write that"**. Not spelled out in chat, and standing only with the user's merge OK on this PR: the 1.2 and 1.5 edges, the noise guard, arm N, the clip draw, the first-10-steps error, the order of the outcome cases and the descriptive items.

## Amendments

(none)
