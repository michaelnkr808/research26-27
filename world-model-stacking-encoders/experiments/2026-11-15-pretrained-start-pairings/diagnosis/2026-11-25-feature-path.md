# Stage 2 diagnosis: why each single encoder scores below `none` (training curves and the feature path, training vs inference)

2026-11-25, [agent, worker] on branch `worktree-agent-a8d641581bdaf9ff1` from main `6bdff16`. Reading only, on the Mac: no kubectl, no cluster, no GPU, no new Job, no edit to `src/`. Every number below was computed this session by the helper scripts in this folder from files committed in the repo, or is quoted from a named file. Labels: **verified** (ran it this session, or read it in the code or a committed log), **recalled** (memory, not checked), **speculative** (a hypothesis).

The question (from the lead): each single-encoder model (DINOv3, C-RADIOv4, SigLIP 2, fused by channel-wise concatenation on the VAE grid) scores below the no-encoder model `none` on clean LIBERO (85.7, 76.6, 72.9 against 92.55; LOG 2026-11-23 entries, lines 832, 845, 867) and `c-radiov4` is below `none` on LIBERO-Plus (44.61 against 50.83; LOG 2026-11-25, line 916-919), while the paper has them level or above. Is it (A) model capacity or our fusion choices, (B) a mismatch between the cached features used in training and the features computed live in rollouts, or (C) cannot tell?

## In plain words

The encoder models learned the action task on the training data just as well as `none` did, so they did not fail to fit. Every step that turns a camera frame into feature channels is the same code, the same weights and the same numbers in training and in rollouts; the caches were recomputed on the rollout card type and came out bit-identical. The one difference left is where the pixels come from: training frames are compressed dataset video, rollout frames are fresh simulator renders. That difference also hits `none`, and nobody has measured whether the encoders react to it more than the VAE does. So the answer is C, with a narrow B (a bug or drift in the feature path) ruled out, and one cluster check named below that separates the rest.

## 1. Training curves (seed 0, all six runs)

Source logs (verified, committed pod logs; each run is one fresh process, no resume, 21,696 optimizer steps of 128):
- `none`: `experiments/2026-11-15-pretrained-start-baseline/results/full-train/stacking-encoders-baseline-full-train-vzc7p.log` (training code `d699682`)
- `dinov3`: `results/train/dinov3-s0/stacking-encoders-pairings-train-dinov3-s0-2cflk.log`
- `c-radiov4`: `results/train/c-radiov4-s0/stacking-encoders-pairings-train-c-radiov4-s0-7jltw.log`
- `siglip2`: `results/train/siglip2-s0/stacking-encoders-pairings-train-siglip2-s0-xv2mk.log`
- `dinov3+c-radiov4`: `results/train/dinov3-c-radiov4-s0/stacking-encoders-pairings-train-dinov3-c-radiov4-s0-dj992.log`
- `dinov3+siglip2`: `results/train/dinov3-siglip2-s0/stacking-encoders-pairings-train-dinov3-siglip2-s0-g7gbf.log` (try 2; try 1 was OOM-killed before its first checkpoint and moved aside)

The five stage 2 runs used training code `c2a2e16`. `git diff d699682 c2a2e16` on the feature-path files changes only file writing (fsync) and the encoder revision pins; `git diff c2a2e16 HEAD` on `rollout.py`, `extract_features.py`, `fastwam_data.py`, `extract_fastwam.py`, `proxy_wam.py`, `pretrained_train.py`, `pretrained_sizing.py` is empty (verified this session).

Computed by `diagnosis/train_curves.py` (verified, run this session with `/opt/homebrew/bin/python3.11 -I`). The pod logs carry one step record per 100 steps after the first few, so each window averages the logged single-step values only (20 in the last window, 50 in the 15,000-19,999 window); each value is the loss of one batch of 128 with that step's own noise and timestep draws (the draws differ across configurations, PLAN "Pairing across configurations"). The full per-step `history.jsonl` is on the cluster volume only and was not needed.

| Config | Action loss, steps 19,700-21,600 mean (sd of single steps) | Action loss, 15,000-19,999 | VAE-channel video loss, 19,700-21,600 (sd) | Feature loss per source, 19,700-21,600 | Action loss at step 0 |
|---|---|---|---|---|---|
| `none` | 0.02638 (0.00394) | 0.03032 | 0.16931 (0.0127) | none | 1.66506 |
| `dinov3` | 0.02804 (0.00407) | 0.03231 | 0.19198 (0.01562) | dinov3 0.58508 | 1.95776 |
| `c-radiov4` | 0.02828 (0.00315) | 0.03271 | 0.18540 (0.01485) | c-radiov4 0.57368 | 1.74537 |
| `siglip2` | 0.02608 (0.00272) | 0.03119 | 0.18955 (0.01498) | siglip2 0.75142 | 1.74537 |
| `dinov3+c-radiov4` | 0.02958 (0.00378) | 0.03257 | 0.18620 (0.01682) | dinov3 0.72701, c-radiov4 0.90958 | 1.79678 |
| `dinov3+siglip2` | 0.02818 (0.00375) | 0.03143 | 0.18665 (0.01675) | dinov3 0.71529, siglip2 1.08472 | 1.79678 |

Reading (verified arithmetic on the table unless marked):
- **Did the encoder runs train as well as `none` on the action objective? Yes.** In the last window each single-encoder run's mean action loss is within about 0.002 of `none`'s and each pair's within about 0.003, against a single-step sd of about 0.003 to 0.004 (standard error of a 20-step mean about 0.0009). `siglip2`, the worst on clean LIBERO, has the lowest final action loss of the four single configurations. The ordering of training action loss does not follow the ordering of clean success.
- **The video loss on the VAE channels is higher in every encoder run**, by about 0.016 to 0.023 in the last window (about 10 to 13% of `none`'s value), and it is higher already in the 5,000-9,999 and later windows (`train_curves.py` output). So the shared backbone does give up some video-prediction fit when it also predicts 1,024 or 1,152 feature channels (verified as a training-loss fact; whether that matters for closed-loop success is speculative).
- **The feature losses flatten after about step 10,000** at about 0.57 to 0.75 for single sources and about 0.72 to 1.08 per source in the pairs (from about 1.6 to 1.9 at step 0), so in the encoder runs the feature terms are most of the total loss (for `dinov3`, 0.585 of a total 0.805 in the last window).
- Gradient norms at the end are similar (`none` 0.138, encoder runs 0.168 to 0.178, last-window means), far under the clip at 1.0.
- What the training curves cannot show: there is no validation split (stage 1 design, all 1,712 episodes trained on), so nothing here measures how either model behaves on states the demonstrations do not contain. Closed-loop rollouts leave the demonstration states after the first mistakes.

### Where the clean-LIBERO drop sits (per task)

Computed by `diagnosis/per_task.py` (verified, from the committed records: `none` `experiments/2026-11-15-pretrained-start-baseline/results/clean-eval/records/clean-none-s0-shard*of4-try1.jsonl`; encoders `results/clean/<config>-s0/records/clean-shard*of4-try1.jsonl`; 2,000 rollouts each, the same 40 tasks x 50 init states, eval seed 41, all on L4 cards per the pod logs). Four-suite averages reproduce the LOG: 0.9255, 0.857, 0.7665, 0.729.

| | `none` | `dinov3` | `c-radiov4` | `siglip2` |
|---|---|---|---|---|
| tasks at 50/50 | 16 | 15 | 8 | 18 |
| tasks below 25/50 | 0 | 5 | 7 | 10 |
| tasks 10 or more successes below `none` | 0 | 9 | 12 | 14 |
| tasks 10 or more successes above `none` | 0 | 1 | 0 | 0 |

Tasks where some encoder model has 20 or fewer successes of 50:

| Suite, task | Instruction | `none` | `dinov3` | `c-radiov4` | `siglip2` |
|---|---|---|---|---|---|
| libero_10 0 | turn on the stove and put the moka pot on it | 28/50 | 36/50 | 23/50 | 18/50 |
| libero_10 2 | put the yellow and white mug in the microwave and close it | 35/50 | 18/50 | 23/50 | 4/50 |
| libero_10 3 | put both moka pots on the stove | 43/50 | 32/50 | 19/50 | 28/50 |
| libero_10 4 | put both the alphabet soup and the cream cheese box in the basket | 46/50 | 46/50 | 34/50 | 8/50 |
| libero_10 5 | put both the alphabet soup and the tomato sauce in the basket | 48/50 | 33/50 | 32/50 | 15/50 |
| libero_10 7 | put the white mug on the left plate and put the yellow and white mug on the right plate | 46/50 | 20/50 | 37/50 | 17/50 |
| libero_10 8 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 34/50 | 20/50 | 34/50 | 0/50 |
| libero_goal 0 | open the middle drawer of the cabinet | 49/50 | 38/50 | 23/50 | 0/50 |
| libero_goal 1 | open the top drawer and put the bowl inside | 48/50 | 46/50 | 9/50 | 14/50 |
| libero_goal 7 | put the wine bottle on the rack | 33/50 | 39/50 | 26/50 | 13/50 |
| libero_object 1 | pick up the bbq sauce and place it in the basket | 45/50 | 20/50 | 50/50 | 50/50 |
| libero_object 4 | pick up the cream cheese and place it in the basket | 47/50 | 14/50 | 50/50 | 50/50 |
| libero_spatial 2 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 35/50 | 49/50 | 4/50 | 25/50 |
| libero_spatial 7 | pick up the black bowl on the ramekin and place it on the plate | 43/50 | 29/50 | 48/50 | 20/50 |
| libero_spatial 8 | pick up the black bowl on the stove and place it on the plate | 47/50 | 49/50 | 1/50 | 31/50 |

Reading: the loss is concentrated in a few tasks per encoder, and the tasks differ by encoder (C-RADIOv4 nearly fails libero_spatial 2 and 8, which `dinov3` does almost perfectly; `dinov3` drops two libero_object tasks that `siglip2` and `c-radiov4` do perfectly; `siglip2` gets libero_object 500/500). Many tasks stay at or near 50/50 for every model (verified, the table). A feature path that is broken everywhere (wrong normalization, wrong channel order, wrong camera) would be expected to hurt nearly every task (speculative). Task-specific collapses fit several explanations: scene-specific feature shift (a B variant), worse generalization off the demonstrations (A), or training-seed variance, which is not measured for any stage 2 configuration (one seed each; speculative).

LIBERO-Plus (quoted from LOG 2026-11-25, line 919, not recomputed): `c-radiov4` is above `none` on Background Textures and Sensor Noise and below on the other five categories.

## 2. Feature path, training vs inference, item by item

"Training" = how the cached inputs were made and fed (`extract_fastwam.py`, `extract_features.py`, `fastwam_data.py`, `proxy_wam.py`, `pretrained_train.py`); "inference" = `rollout.py` (Featurizer, Policy). File:line are at HEAD `6bdff16`, which equals the training code (`c2a2e16`) and the stage 2 evaluation code (`2a9c058`, from the eval pod logs' `SRC_GIT_COMMIT`) on every file named here (empty `git diff`, verified). Paths are under `world-model-stacking-encoders/src/`. Marks: **match**, **mismatch**, **unclear**. "Shared with none" means the same difference is in the `none` model's VAE path.

| # | Item | Training side | Inference side | Mark | Evidence |
|---|---|---|---|---|---|
| 1 | Encoder checkpoint, variant and revision | `extract_features.py:80-84` (REVISIONS), `:90` DINOv3 ViT-L/16, `:112` SigLIP 2 so400m-patch14-384, `:135` C-RADIOv4-SO400M; called via `extract_fastwam.py:159` (no variant) | same loaders via `rollout.py:221` (`ENCODERS[s](device, ENC_DTYPE[s])`, no variant) | match | verified: same functions; the freshness check after the SigLIP 2 repair logged the pinned revision loaded for each encoder and found all 24 sampled episode-source pairs bit-identical to the caches (`results/cache-fresh-check/stacking-encoders-pairings-cache-fresh-check-rw7pl.log`, `=== OK`, L4) |
| 2 | Encoder compute dtype | `--dtype` in the extraction Jobs: dinov3 fp32 (`experiments/2026-10-27-proxy-calibration/jobs/rendered/job-extract-dinov3.yaml:47`), c-radiov4 fp16 (`job-extract-c-radiov4.yaml:47`), siglip2 fp16 (`jobs/rendered/job-siglip2-reextract.yaml:80`, `job-siglip2-repair.yaml:94`) | `rollout.py:82` ENC_DTYPE dinov3 fp32, c-radiov4 fp16, siglip2 fp16 | match | verified (read both sides) |
| 3 | GPU type (fp16 numerics differ across cards) | on the 8 sampled episodes per source, the caches as now stored equal an L4 recompute bit for bit (`results/cache-fresh-check/stacking-encoders-pairings-cache-fresh-check-rw7pl.log:19` NVIDIA L4, summary `=== OK`); SigLIP 2 cache re-extracted on L4s after an earlier card-numerics mismatch (LOG 2026-11-18 to 2026-11-20) | stage 2 clean evals ran on L4 (e.g. `results/clean/dinov3-s0/stacking-encoders-pairings-clean-dinov3-s0-0-xntzt.log:27`; the same nvidia-smi line in each `results/clean/*/` pod log) | match | verified |
| 4 | Pixel source | dataset video: 512 x 512 AV1, 20 fps, decoded `rgb24` (`fastwam_data.py:4`, `:59-80`, `:76`) | simulator render at 256 x 256 (`rollout.py:71`, `:367`) | **mismatch, shared with none** | verified (code). Size of its effect on encoder features vs on VAE latents: not measured anywhere (the parity test feeds identical uint8 frames to both paths, `test_feature_parity.py:38-40`) |
| 5 | Resize to 224 per camera | float `[0,1]` torch bilinear with antialias (`fastwam_data.py:83-86`), then rounded to uint8 for encoders (`extract_fastwam.py:109`) | PIL bilinear center-crop-resize on uint8 (`rollout.py:171-179`, `:194-195`) | **mismatch (small), shared with none** | verified by the earlier test: at most 1 uint8 level on about 20% of values on the same input (`test_feature_parity.py:30-37`, `test_prep_gap_bounded`, accepted by the user 2026-11-17, direction LOG line 1947); not rerun this session (no torch env on the Mac, no pip installs allowed) |
| 6 | Orientation | dataset frames upright, no flip (`fastwam_data.py:5-6`) | both cameras rotated 180 degrees (`rollout.py:194-195`) | match | verified by eye in earlier sessions on all four suites, sim first frame next to training frame 0 (direction LOG 2026-10-29 (a), lines 639, 646); not redone here |
| 7 | Color order | RGB (`fastwam_data.py:76` `rgb24`) | sim observation used as delivered, no channel swap (`rollout.py:194-195`, `:227`) | match | code verified; that robosuite's `*_image` is RGB is recalled, and a swap would also have shown in the item 6 side-by-side (speculative) |
| 8 | Cameras and their order | agentview then wrist (`fastwam_data.py:26`; `extract_fastwam.py:108-115`) | agentview then wrist (`rollout.py:194-195`, `:233-237`) | match | verified |
| 9 | uint8 rounding before the encoder | `extract_fastwam.py:109` | inputs already uint8 (`rollout.py:179`) | match | verified |
| 10 | Encoder-internal resize and normalization | `extract_features.py:50-57`: bicubic to 224 / 384 / 448 (`:92`, `:115`, `:138`), processor mean and std (DINOv3, SigLIP 2), none for C-RADIOv4 (`:148`) | the same `encode` closures (`rollout.py:235`) | match | verified (one function); code-path parity at the same batch: max abs difference 0 for all three encoders on CPU (PR #55, direction LOG line 1947; not rerun) |
| 11 | Tokens kept | DINOv3 drops CLS and registers (`extract_features.py:96`, `:103`); SigLIP 2 all patch tokens (`:126`); C-RADIOv4 `spatial` (`:151`) | same functions | match | verified |
| 12 | Resize to the VAE grid (14 x 14 per camera, cameras side by side to 14 x 28) | `fastwam_data.py:101-107` (bilinear, `align_corners=False`), called at `extract_fastwam.py:113`, concatenated on width at `:115` | same function at `rollout.py:236`, concatenated on width at `:237` | match | verified |
| 13 | Encoder batch size | `--batch 32` (`job-extract-dinov3.yaml:47`, `job-extract-c-radiov4.yaml:47`, `job-siglip2-reextract.yaml:80`), chunked at `extract_fastwam.py:111-112` | one call per batch of pending requests (`rollout.py:314-317`, per camera `:233-235`), at most 4 per pod (`ENVS=4`, line 3 of each `results/clean/*/` eval log) | unclear (small) | on CPU the encoders differ by 1 to 2 fp16 steps across batch sizes (`test_feature_parity.py:18-20`, `:61-62`); not measured on L4 |
| 14 | Time assignment | latent step 0 = features of frame t (`proxy_wam.py:163`, placed at `:259`); VAE step 0 = frame t of the 9-frame clip (`fastwam_data.py:89-98`, `extract_fastwam.py:99`); steps 1-2 = means of t+4..t+16 and t+20..t+32 (`proxy_wam.py:125-134`, `:160-163`), used only as noised targets | step 0 = the current frame (`rollout.py:399-400`, `:317-322`); steps 1-2 start as noise and are denoised and discarded (`proxy_wam.py:450-465`) | match | verified. Features enter at inference only through latent step 0, the clean slot (`proxy_wam.py:413` training, `:452` and `:464` inference) |
| 15 | VAE latent of step 0 | encoded inside a 9-frame clip, bf16 (`extract_fastwam.py:94-101`; `job-extract-vae.yaml:47` `--dtype bf16`) | the single current frame, bf16 (`rollout.py:220`, `:228-231`) | match, shared with none | code verified; that the Wan 2.2 VAE's first latent step depends only on the first frame (causal) is recalled, not checked |
| 16 | float16 rounding | caches stored float16 (`extract_fastwam.py:115`), read and cast to float32 (`proxy_wam.py:349`) | `.half().float()` on VAE and features (`rollout.py:231`, `:237`) | match | verified |
| 17 | z-score statistics: where computed | per episode count, sum, sum of squares in float64 over all frames, positions and both cameras (`extract_fastwam.py:182-188`); pooled over the 1,712 training episodes, sd + 1e-6 (`proxy_wam.py:196-205`) | read from the checkpoint's config (`rollout.py:287-288`) | match | verified (code) |
| 18 | z-score statistics: same values | written into `cfg["feature_stats"]` (`pretrained_train.py:554`), saved in `config.json` (`:610`) and in `model.pt` with the same cfg (`:644`) | `Policy.load` reads `model.pt` config (`rollout.py:293-298`) | match | verified by code (one dict); `diagnosis/feature_stats.py` (run this session) shows each source's stats identical across every run that uses it (DINOv3 in 3 runs, C-RADIOv4 in 2, SigLIP 2 in 2), and action and proprio normalization identical to `none` in all runs |
| 19 | z-score: where applied | after float16, on the device (`proxy_wam.py:350-353`) | after float16 (`rollout.py:319-321`) | match | verified. One flag: DINOv3 channel 757 has training mean 3.4e-5 and sd 2.0e-4 (`feature_stats.py`; every other DINOv3 channel has sd above 0.039, and SigLIP 2 and C-RADIOv4 have no channel under 0.28). A live value off by 0.001 in that channel becomes about 5 z-units. Whether live values move that much is not measured (unclear, speculative risk; DINOv3 is the least-affected encoder) |
| 20 | Channel order in the concatenation | VAE first, then the config string's order (`proxy_wam.py:106`, `:195`, `:221-227`) | `[VAE] + cfg["sources"][1:]` in order (`rollout.py:284`, `:298`, `:318-322`), width asserted (`:323`) | match | verified |
| 21 | Tensor layout | (B, 3, 14, 28, C) permuted to (B, C, 3, 14, 28) (`proxy_wam.py:354`) | (B, 14, 28, D) permuted to (B, D, 1, 14, 28) (`rollout.py:321`) | match | verified |
| 22 | What the feature channels hold at a step | step 0: clean z-scored features of frame t; steps 1-2: noised z-scored future means, one noise tensor and one video timestep over all channels (`proxy_wam.py:411-413`); loss on steps 1-2 only (`:421`) | step 0: clean z-scored live features, reset after each of 10 denoising steps; steps 1-2: N(0, 1) start noise over all channels (`rollout.py:112-118`, `proxy_wam.py:450-465`) | match | verified; as the paper's p15 describes (PLAN fusion table row 8) |
| 23 | Model precision | bf16 autocast (`pretrained_train.py:132`) | bf16 autocast (`rollout.py:250`, `:333`) | match, shared with none | verified |
| 24 | Model build | `pretrained_sizing.py:99-103` calls `proxy_wam.build_model` | `rollout.py:296` calls the same with `cfg["in_dim"]`, `cfg["model_kw"]` | match | verified (code); `test_pretrained_train_sources.py` check 5 asserts the widened model loads with every parameter equal (not rerun) |

Summary: every item from the encoder weights to the model input is the same on both sides, and the two numeric checks that compare real caches with recomputation (freshness on L4, 8 sampled episodes per source) and the two code paths (parity, synthetic frames on CPU) found no difference. The differences that remain are before the encoder (items 4 and 5: dataset video vs live render, and the resizer), shared with `none`, plus two small unclears (item 13 batch numerics, item 19 the one near-constant DINOv3 channel).

## 3. Answer: C (cannot tell yet), with the narrow form of B ruled out

The encoder models fit the action objective on the training data as well as `none` (final-window action loss within about 0.002 of `none` for every single-encoder run, and `siglip2`, the worst on clean LIBERO, has the lowest), so capacity in the sense of failing to learn the task is not supported (verified, section 1). A mismatch in the feature path itself (wrong weights, dtype, card numerics, preprocessing inside the encoder, grid resize, time step, normalization statistics, channel order or layout) is ruled out by reading both sides item by item and by the two earlier numeric checks, cache freshness on L4 and code-path parity (verified, section 2). What is left cannot be separated with what is on the Mac: either (A) our fusion choices make the model generalize worse once a rollout leaves the demonstration states (the training loss cannot show this because there is no validation split; the higher VAE-channel loss in every encoder run and the large share of the loss taken by the feature terms are consistent with it, speculative), or (B, broad form) the encoder features react more than the VAE latents to the one input difference that remains, dataset video frames in training versus simulator renders at inference (items 4 and 5; shared with `none`, effect on each channel group never measured), or the gap is partly training-seed variance, which is unmeasured with one seed per configuration (speculative; the per-task pattern, collapses on different tasks for each encoder, fits all three).

**The single check that would settle A against B:** measure how far the live inputs at inference sit from the training inputs, per channel group, at a state both share. For each of the 40 clean tasks and a few init states (for example 0 to 4), build the first policy frame exactly as a rollout does (LIBERO-Plus env at 4976dc3, `rollout.libero_env`, `env.reset`, `set_init_state`, 30 dummy steps, `rollout.process_obs`) and pass it through `rollout.Featurizer` and the checkpoints' z-score statistics (from `config.json`, already in the committed training logs). On the training side, read frame 0 of every training episode of the same task: `/mnt/proxy/feats/{vae,vae_b}/<suite>/ep<id>.npy[0, 0]` (VAE latent step 0) and `/mnt/proxy/feats/{dinov3,c-radiov4,siglip2}/<suite>/ep<id>.npy[0]`, tasks from `/mnt/proxy/feats/meta/<suite>.npz`, z-scored the same way. For each channel group (VAE 48; each encoder), compute the distance from each live frame to its nearest training frame 0 of the same task, divided by the median nearest-neighbour distance among that task's training frame 0s (leave one out). Also report the live z-scores of DINOv3 channel 757. If the encoder groups' ratios sit near the VAE group's ratio, the live features are as close to training as the VAE latents are, broad B is unlikely, and A (or seed variance) is what remains. If the encoder groups' ratios are clearly larger than the VAE group's, B is the lead, and a rollout with training-like frame preparation would be the causal follow-up. The reading thresholds should be fixed before the run (the lead's or user's call, not set here). This needs the cluster: the caches are only on PVC `stacking-encoders-proxy`, and the simulator, the Wan 2.2 VAE file and the encoder weights are in the eval image and on the volume, not on this Mac. CPU is enough for about 200 frames (speculative estimate). Separating A from seed variance afterwards needs a second training seed of at least one encoder configuration (PLAN, wave 2 deferred), which is not a single check.

## Files

- `diagnosis/train_curves.py`: section 1 training table (reads the six committed training pod logs).
- `diagnosis/per_task.py`: section 1 per-task clean table (reads the committed clean-eval records).
- `diagnosis/feature_stats.py`: section 2 items 18 and 19 (reads the `config.json` printed into the training pod logs).

Run each from the repo root with `/opt/homebrew/bin/python3.11 -I <path>`; standard library only; each prints JSON.
