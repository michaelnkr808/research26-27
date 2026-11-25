# Plan: imagined-future test (on training clips, does acting on its own imagined future cost an encoder model more than it costs the no-encoder model?)

2026-11-25, [agent, lead, planning], branch base `47c0c04`. A diagnostic test inside stage 2 (`../PLAN.md`), written and merged before any code or Job for it exists. It changes no stage 2 rule, result, checkpoint or configuration. Labels: **verified** (read this session in the named file), **recalled**, **speculative**.

## In plain words

Each single-encoder model scores below the no-encoder model `none`, although all of them fit the training actions about equally well. One explanation the user and the lead picked on 2026-11-25 (direction `LOG.md`, "hypotheses for why performance fell", candidate 2): in a rollout the model has to imagine the future from noise, the action part looks at that imagined future, and an 8-layer model may imagine 1,024 or 1,152 encoder channels badly. Training never shows this, because in training the future the action part sees is the true future with some noise added. This test runs each trained model the way a rollout runs it, but on training clips, where the true actions and the true future are known. It asks whether the encoder models then act worse than `none`, and if so whether swapping the imagined future for the true one, at the same noise levels, removes the difference. No simulator, no test data, no retraining.

## Where the method comes from

The test is our own; no paper is followed. The inference procedure it copies is the paper's as implemented in `src/proxy_wam.py` (`infer_actions`: 10 joint denoising steps, `INFER_STEPS = 10`, line 70, App. E.3 of liu2026what; verified). The bars are the planning lead's choice (see "Sign-off").

## What is known when this is written

- Verified in code: the action tokens attend to all video tokens in `FastWAMJoint` (`experiments/2026-11-12-pretrained-start-sizing/step2-reading.md` line 76); at inference video and actions start from noise and are denoised jointly, with latent step 0 reset to the observation before the first step and after every step (`src/proxy_wam.py`, `infer_actions`); in training the video and action noise levels are drawn separately and the future latent steps are the true future at the drawn level (`training_loss`). A rollout executes the first 10 actions of each 32-step chunk (`src/rollout.py` line 74, `REPLAN = 10`) and runs inference under bf16 autocast (`Policy.act`, `src/rollout.py` lines 332-334).
- The training sampler for the video noise level is sigma = 5u / (1 + 4u) with u uniform (`experiments/2026-11-15-pretrained-start-baseline/fastwam-reading.md` line 51, verified there), so the share of training draws at or below a level s is s / (5 - 4s) (arithmetic): about 0.64 at s = 0.9, about 0.10 at s = 0.36.
- Stage 2 results seen by the planning lead: clean LIBERO `none` 92.55, `dinov3` 85.7, `c-radiov4` 76.65, `siglip2` 72.9; LIBERO-Plus `none` 50.83, `c-radiov4` 44.61; the diagnosis note `2026-11-25-feature-path.md` (final training action loss about the same across runs; VAE-channel video loss higher in every encoder run; the per-task clean table). No measurement of this test exists.
- Added 2026-11-25 after the design's review, before merge (the experiment session's report of 17:40 UTC; stage 2 `LOG.md`, last entry at main `44741f4`, read by the planning lead): `dinov3` LIBERO-Plus is Y = 41.741, with the three evaluation seeds far apart (41 / 42 / 43: 49.970 / 34.536 / 40.718, SD 7.768) where `none` has SD 3.041 and `c-radiov4` 1.245, and the same order in every shard and category. Verified in code by the planning lead: `request_noise(eval_seed, channels)` (`src/rollout.py` lines 112-118) seeds one video and one action generator with the evaluation seed, and `Policy.start_noise` (lines 306-311) caches the result, so a rollout uses the same starting-noise tensor at every policy call of every episode under that evaluation seed. The evaluation seed also seeds the environment (line 368, the experiment session's reading, not re-read here). Clean LIBERO is run under evaluation seed 41 only. `dinov3+c-radiov4` clean LIBERO: 83.5.

## Design

**Checkpoints.** Seed 0 of all six configurations: `none` (`/mnt/proxy/runs/baseline-none-s0/model.pt`), `dinov3`, `c-radiov4`, `siglip2`, `dinov3+c-radiov4`, `dinov3+siglip2` (`/mnt/runs/runs/pairings-<cfgk>-s0/model.pt`, where cfgk is the configuration name with "+" written as "-"), each loaded with its own config. The reading rule covers the three single encoders. The two pairs are computed and reported descriptively only.

**Clips (training data only).** Tasks are ordered by (suite name, instruction string). Within a task, its training episodes are ordered by episode id. One generator, `numpy.random.default_rng(0)`, is used for all tasks in that order: for each task, `rng.permutation(number of episodes)[:5]` picks 5 episodes (all of them if a task has fewer than 5; none is expected), then sorted by episode id. For each picked episode of length n, three windows, starting at frames 0, floor(n / 3) and floor(2n / 3). The clip index is the position in the list ordered by (task order, episode id, window). 600 clips if every task has 5. Every checkpoint gets the same clips in the same order.

**Inputs.** For each clip, exactly what a training step receives for that window, built by the training data code (`proxy_wam.Data` with the checkpoint's own `cfg["suites"]` and sources, `to_device`, with `data.context_dev` set as `train()` sets it): the cached latents (VAE channels and the configuration's z-scored encoder channels, 3 latent steps), the T5 context, the proprio state, the normalized 32-step action chunk and the padding masks. The script asserts that the `Data` object's action, state and feature statistics equal the checkpoint's `cfg["action_norm"]`, `cfg["state_norm"]` and `cfg["feature_stats"]`. The observation is `latents[:, :, 0:1]`.

**Precision and batching.** Every arm runs under `torch.autocast("cuda", dtype=torch.bfloat16)` as `Policy.act` does, on the eager path, with one fixed batch size and the clips in clip-index order, on the same card model for all six checkpoints.

**Starting noise.** For noise seed s and clip index i, generator k is a CPU `torch.Generator` seeded with `int(numpy.random.SeedSequence([s, i, k]).generate_state(1)[0])`. Generator k = 0 draws the VAE-channel video noise (48, 3, 14, 28) first and then the action noise (32, 7), so every checkpoint starts from the same VAE-channel and action noise. Generator k = 1, 2, 3 draws the channels of `dinov3`, `c-radiov4`, `siglip2` in that fixed numbering. Two noise seeds, A = 0 and B = 1.

**Three arms.** Write `tv`, `dv`, `ta`, `da` for the schedules `infer_actions` builds with 10 steps, x for the clip's cached latents and z for its starting video noise. In every arm the action part is exactly `infer_actions`': the action schedule is unchanged and `infer_action_scheduler.step` is called every step. In every arm `infer_video_scheduler.step` is also called on the whole video tensor every step, as `infer_actions` does; the arms differ only in what is written into latent steps 1 and 2 afterwards and in the video timestep passed to the model.
- **I, imagined (what a rollout does).** Unmodified `infer_actions`.
- **M, true future at matched noise.** Before step k (k = 0..9), latent steps 1 and 2 are set to `model.train_video_scheduler.add_noise(x, z, tv[k])` restricted to those steps (the same z at every step); the video timestep passed is `tv[k]`, as in arm I. After the scheduler step they are overwritten again before the next step. Latent step 0 is reset to the observation as in arm I. The only difference from arm I is the content of the future: the true future at the level the schedule expects, where arm I holds the model's own partly denoised guess. The cached future is written as it is, also where a future latent step is padding (as training inputs are); the report counts the clips with a padded future latent step.
- **N, no future information.** Before every step, latent steps 1 and 2 are set to z and the video timestep passed is `tv[0]`.

**Action error.** For a clip, an arm and a noise seed: the mean, over action dimensions 0 to 5 (the six motion dimensions) and the valid steps among the first 10 of the chunk (not `action_is_pad`), of the squared difference between the arm's output chunk and the demonstration's normalized chunk. The gripper dimension is left out of the primary error because it is close to two-valued and rollouts binarize it; it is reported separately (below). A clip with no valid step among the first 10 is dropped for every checkpoint and counted in the report. E(c, arm) is the mean over the remaining clips and over both noise seeds.

**Statistics, per single encoder e.**
- **Q(e) = E(e, I) / E(none, I).** Also Q_A(e) and Q_B(e), the same from noise seed A alone and B alone.
- **G(e) = 1 - (E(e, M) - E(none, M)) / (E(e, I) - E(none, I))**, the share of the gap to `none` that the true future closes. Computed only for an encoder that is "clearly worse" in step 1, where the denominator is positive.
- **Q_N(e) = E(e, N) / E(none, N)**, reported; it adds a label, not a case.
- Uncertainty: a 95% percentile interval for Q(e) and G(e) from 1,000 bootstrap draws (seed 0). Each draw resamples the 40 tasks with replacement and takes all clips of a drawn task (a task drawn twice counts twice); the numerator and the denominator of a ratio use the same draw and the same dropped clips. The interval covers which clips were sampled, not the noise seed and not the training seed.

## Predictions (written before any measurement; they carry no consequence)

The lead's guess (speculative): "about the same" for all three single encoders, because every clip here was trained on and all runs fit the training actions equally. If so, the test rules out the simple form of candidate 2 and leaves the form in which imagination fails only in states the model has not seen.

## Reading rule (fixed before the run; per single encoder, from the point estimates)

Step 1, from Q(e).
- **Q(e) <= 1.2: "about the same"** (reported as "better than none" if Q(e) <= 1 / 1.2; same zone for the cases below).
- **Q(e) >= 1.5: "clearly worse".**
- **1.2 < Q(e) < 1.5: "unclear".**
- **Firmness.** The zone is "not firm", and the encoder is treated as "unclear" in the cases below, if Q_A(e) and Q_B(e) do not both fall in the same zone as Q(e), or if the zone is "clearly worse" and the bootstrap interval of Q(e) includes 1.0.

Step 2, only for an encoder that is firmly "clearly worse", from G(e).
- **G(e) >= 2/3: "goes away with the true future".**
- **G(e) <= 1/3: "stays with the true future".**
- **1/3 < G(e) < 2/3: "attribution unclear".**
- Label, no change of case: if Q_N(e) >= 1.5 the report adds "also present with no future information".

## What each outcome leads to (written before the run; exactly one case applies, checked in this order)

1. **At least one encoder is "clearly worse" and "goes away with the true future".** Candidate 2 is supported for the encoders named, and only for them: on inputs they were trained on, acting on their own imagined future costs them clearly more than it costs `none`, and the true future at the same noise levels closes most of the gap. The other encoders are reported with their own zones. This is a lead for the next design, not a finding: one training seed per configuration (the one-seed override in `STANDARDS.md` allows descriptions, not claims). The lead brings the user the choice of a training-side change that reduces the action part's dependence on the imagined future (one existing example: in Fast-WAM's base model the action tokens attend only to the first video frame; `step2-reading.md` line 76, verified). Any such change is a new training experiment with its own design by PR and the user's go.
2. **Otherwise, at least one encoder is "clearly worse".** The deficit is present in inference mode on familiar inputs and the imagined future does not explain it, or the attribution is unclear. Reported with G(e) and the Q_N label; the lead and the experiment session look for the cause in the inference path before anything else is proposed.
3. **Otherwise, all three are "about the same".** The simple form of candidate 2 is not supported. What remains goes to the user with the choices already open: imagination or acting that fails only in unfamiliar states (this test cannot see it), the feature loss crowding out the backbone (candidate 1), the pixel source (the first-frame check), our choices where the paper is silent.
4. **Anything else** (some encoder "unclear" or "not firm", none "clearly worse"). Reported as it is; the user decides, including whether a larger clip set is worth running.

In every case the stage 2 decision rule, the one-seed reporting rule (`../PLAN.md`, amendment 2026-11-18) and every stage 2 result stand unchanged, and nothing from this test goes into README key findings without the user's sign-off.

## Also reported, descriptive only (no bar, no decision made from them)

- E for every arm, both noise seeds and all six checkpoints, on the first 10 steps and on all 32; the same with all 7 dimensions; and the gripper alone as the share of valid steps where the output's gripper sign equals the demonstration's.
- E(none, I) under noise seed B over seed A: how far a ratio moves from inference noise alone (it says nothing about clip sampling or training seeds).
- **Recovery share**, per checkpoint: (E(c, N) - E(c, I)) / (E(c, N) - E(c, M)), how much of the benefit of knowing the future the model's own imagination recovers; not computed where the denominator is not positive.
- **Imagination error**, per checkpoint and source group, after arm I: the mean squared difference between the imagined latent steps 1 and 2 and the true cached ones over the valid future steps, beside the mean square of the true values (the error of imagining zero). For the VAE channels also the ratio of each encoder checkpoint's error to `none`'s. This bears on candidate 1 (does the model imagine the picture worse once it must also imagine encoder features), descriptively.
- Q(e) by task, and per encoder the Spearman rank correlation over the 40 tasks between the per-task Q(e) and the per-task clean success drop (`none` minus the encoder, from the committed clean records).
- Q(e) by window position (0, one third, two thirds).
- **Rollout noise draws** (added 2026-11-25 before merge, see "What is known"): arm I run three more times per checkpoint with the rollout's own starting noise, `rollout.request_noise(s, channels)` for s = 41, 42, 43, used as `Policy.act` uses it and the same tensor for every clip. Reported per checkpoint: E(c, I) under each of the three, the largest over the smallest, and the same per task; beside it the same ratio between noise seeds A and B (per-clip noise). It asks whether one fixed noise tensor moves a model's actions on familiar clips, and whether it moves `dinov3` more than `none`. No bar and no case depends on it.

## Limits (stated before the run)

- **Open loop, familiar inputs.** Every clip was trained on, and each is judged from a demonstration state. A rollout visits states no demonstration contains, and its mistakes feed back. "About the same" here does not show the imagined future is harmless in a rollout; it shows only that the harm, if any, needs unfamiliar states.
- **One training seed.** A Q(e) above 1 can come from training-seed variance alone, which is unmeasured for every stage 2 configuration (the diagnosis note already shows two encoder runs ending with a slightly higher training action loss). Nothing in this test separates that from imagination; it is why case 1 is a lead and not a finding.
- **What a closed gap shows.** G(e) near 1 says the deficit is tied to the imagined content. It does not say whether the encoder model imagines worse or uses an imperfect future worse; the imagination-error item speaks to the first, descriptively. A shared error floor (the demonstration's action is one valid action among several, and squared error charges a model for other valid choices) pulls ratios toward 1, which is why step 2 uses a share of the gap and not a ratio.
- Arm M's noise levels are the inference schedule's own, all inside the range training draws from; arm N holds the video at the schedule's highest level, a common training regime (see the sampler above). How the inference schedule's levels are spaced is not readable in this repo (Fast-WAM's scheduler source is not in it); the done checks print them.
- The rollout-noise item holds the environment fixed (there is none) and uses familiar clips. A flat result there does not show the evaluation-seed spread on LIBERO-Plus is not a noise effect: the spread could need unfamiliar states, or come from the environment seeding. A spread there is a lead only.
- The inputs are cached training inputs, not live rollout inputs; the pixel-source question is the first-frame check's, not this test's.
- The 1.2 and 1.5 edges and the 1/3 and 2/3 edges are our own choices. They are not calibrated for power: nothing says how large a Q matters for closed-loop success, and the 8-clip dry run cannot tell.

## Test-set use

None. Training clips only; no simulator, no evaluation init state. The per-task correlation uses clean outcomes already read and logged.

## Compute

One GPU Job, no simulator (speculative: several GPU-hours at most on one 24 GB card, from 90 model passes per clip and checkpoint: 3 arms x 2 noise seeds x 10 steps, plus 3 x 10 for the rollout-noise item; not sized. An earlier version of this line said 50, an arithmetic slip). The experiment session sizes it from a dry run on 8 clips and checks the utilization rules of `_standards/08`. Caches are on PVC `stacking-encoders-proxy`, the stage 2 checkpoints on `stacking-encoders-runs`.

## Hand-off

The experiment session writes the script in this folder by PR (no edit to `src/`) and submits only on the user's go in its own window. Done checks before the full Job:
- its arm I output equals `proxy_wam.infer_actions` on the same inputs and noise under the same autocast (largest absolute difference reported, expected 0);
- `add_noise` at the schedule's level for a fully clean sample returns the cached future (if the scheduler has such a level; otherwise the smallest level is printed with the difference);
- the ten levels of `tv` are printed, each with the share of training draws at or below it;
- one clip's inputs (observation latents, proprio, action chunk) are compared with the cache files they came from;
- a dry run on 8 clips.

Its PR and LOG entry pin the source commit, the image, the six checkpoint paths with their sha256, the cache and T5-cache meta files with their sha256, the batch size and the card model. The script is merged before the Job runs and before any number is read. The result goes in the stage 2 `LOG.md` and a short README in this folder.

## Sign-off

The user, planning window, 2026-11-25: to the lead's description of the test ("run each model the way a rollout runs it [on training clips], and compare its actions to the demo's. Then do it again while handing the model the real future") and "Want me to write that design now?": **"2. yes sure write that"**. Not spelled out in chat, and standing only with the user's merge OK on this PR: the 1.2 and 1.5 edges, the gap-share edges, the firmness rule, arms M and N as defined, the clip draw, the first-10-steps error on the six motion dimensions, the order of the outcome cases and the descriptive items. The rollout-noise item was added after the read-only review and after the lead first asked for the merge OK; no reviewer has read it.

## Amendments

(none)
