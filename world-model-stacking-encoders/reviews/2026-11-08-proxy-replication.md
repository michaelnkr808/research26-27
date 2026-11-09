# Review of the proxy replication (experiment 2026-10-27-proxy-calibration)

[agent, review] Written 2026-11-08. Read-only review: nothing was run on the cluster, nothing outside this file was changed. Labels: verified (read in the file or ran it here), recalled (from memory or from a LOG claim I did not re-check), speculative (my inference or arithmetic on assumed inputs).

## Scope and base commit

- Base: main at commit bde59a7 (`git rev-parse HEAD` first returned 729c3c6; fast-forwarded with `git merge --ff-only bde59a7`; verified).
- State reviewed: the experiment LOG through entry 2026-11-09 (o) (12 of 16 evaluation shards done, 0 errors), PLAN with all amendments, RUN.md, README, `jobs/` and `jobs/rendered/`, `results/pilot`, `results/full-benchmark`, `results/logs` (listing and the files cited), decisions 001 and 003, the paper note and `liu2026what-table10-composition.csv`, `_standards/` 01-08, the direction STANDARDS, README, PLAN, and the code and tests named in the task (`src/proxy_policy.py`, `proxy_wam.py`, `rollout.py`, `lp_eval.py`, `t5_cache.py`, `fastwam_data.py`, `profile_policy.py`, `pilot_timings.py` and the four test files). Line numbers below are in those files at bde59a7.
- Not reviewed: `extract_fastwam.py` and `extract_features.py` (not in the task list; the rollout featurizer imports them, so "same preprocessing as training" is taken from `rollout.py` docstrings and LOG, not from a read of the extractor), the Fast-WAM source (not in the repo and not installed), the real rollout JSONLs (on the PVC, not in the repo, and the user's rule is that no verdict is seen before all 16 shards are in).
- Tests, exact outcome (verified, Homebrew `/opt/homebrew/bin/python3.11`, Python 3.11.14, numpy 1.26.4, torch 2.11.0, pytest 7.1.2, run from `world-model-stacking-encoders/`; nothing installed):
  - `python -m pytest src/test_proxy_policy.py src/test_proxy_wam.py src/test_rollout.py src/test_lp_eval.py -q` stops at collection: `2 errors`, both `ModuleNotFoundError: No module named 'fastwam'` (from `src/proxy_wam.py:52`, imported by `test_proxy_wam.py:16` and `test_rollout.py:12`). FastWAM at 7faa711 is not installed in this Python (RUN.md "Session start" step 2 says it must be installed with `pip install --no-deps -e`). That is an environment finding: the two test files that cover the training model and the rollout code could not be run here.
  - Run file by file: `pytest src/test_lp_eval.py -q` gives `1 passed` (pytest collects only `test_seednoise`, `test_lp_eval.py:313`; checks 1-9 are inside `main()`, `:59`, and only run as a script). `python src/test_lp_eval.py` prints `1 ok` ... `10 ok` and `all checks passed`, including `5 ok: locked verdict on 14 planted cases incl. every boundary` and `10 ok: seednoise (3 outcomes, exact boundaries, floor precedence, completeness failures, --partial, .errors.jsonl ignored, clean rows excluded)`. `pytest src/test_proxy_policy.py -q` gives `no tests ran` (script-style); `python src/test_proxy_policy.py` prints checks `1 ok` ... `9 end to end ok` and `all ok` (this tests the superseded `proxy_policy.py`, not the code that trained the runs).
  - `test_proxy_wam.py` (12 checks) and `test_rollout.py` were not run by me. Their passes are recorded in LOG/PLAN only (recalled).
  - `git status --porcelain` was empty after the runs.

## Summary

- The code implements what PLAN and decision 003 say, with small, mostly disclosed differences. The locked outcome rule and the locked seed-noise rule are implemented exactly (thresholds as exact fractions, pairing, refusal when incomplete, `*of8.jsonl` only). I found no bug that would silently change a result. Details: sections 1 and 3.
- The biggest undisclosed or weakly disclosed items are minor: (a) the per-source feature loss is a mean over that source's channels (so a 1,152-channel source and a 1,024-channel source get equal weight); the code says so, PLAN does not list it as a choice where the paper is silent; (b) the 3 training seeds are paired by data order only, not by initialisation or noise, which weakens the PLAN's note that the same-seed comparison is probably less noisy than the cross-seed check; (c) clean LIBERO is 40 trials at one eval seed while PLAN:35 still says 50 per task; (d) the training environment's resolved package versions are not recorded (`pip freeze` goes to an emptyDir).
- Statistics (speculative arithmetic, section 4): with 3 training seeds the harm rule fires almost always if the proxy's true D+R drop equals the paper's 5.41 points and the per-seed noise is 2.7 points or less, about half the time if the true drop is half that, and about 3 to 10 percent of the time when there is no drop and the noise is 2.7 to 5.4 points. The training-seed noise itself is not measured yet; the seed-noise result (pending) is the first number that will say.
- Compute: no GPU-hour cap exists, the 8-GPU concurrency gate was respected as far as the LOG shows, and manifests meet the 20 percent request/limit rule. Gaps: no per-Job estimated-versus-actual GPU-hour record, the NRP Violations page was never read for these Jobs, an L40 pod ran about 3.5 hours after the watcher flagged it, and two pods ran for hours with no owning Job (disclosed, user decided).
- Documentation: LOG is append-only with labels. README, PLAN and the experiment README status still say 7 of 16 shards (the LOG is at 12 of 16), and the direction LOG has no summary of experiment entries (m), (n), (o). Two points were decided by an agent that touch the user's gates: the seed-noise rule was locked on the user's silence, and the first training Job was submitted on a message that did not name it (both flagged in the LOG itself).
- Outlook (speculative): the replication can say whether the proxy shows a D+R drop, not whether the paper's drop is real. The biggest risks are throughput (about 13 more checkpoints at roughly 76 GPU-hours of evaluation each, on L4 cards only) and whether a 32M-parameter from-scratch model scores high enough on LIBERO-Plus for any comparison to mean something. The seed-noise result answers the second within about a day.

## 1 Plan vs code

### 1.1 Architecture, encoders, fusion, normalisation, losses, length, seeds

| PLAN / decision 003 says | Code | Verdict |
|---|---|---|
| Proxy is Fast-WAM's `FastWAMJoint` (video DiT + action DiT, MoT, joint attention, first-frame-causal video mask), small, random init (PLAN:129) | `build_model` `proxy_wam.py:372-387`: `WanVideoDiT` hidden 384, 8 layers, 3 heads x 128; `ActionDiT` width 128; `MoT`; `FastWAMJoint(..., video_train_shift=5.0, action_train_shift=1.0, loss_lambda_* 1.0)`. 31,647,943 params (LOG 2026-11-06 (c), 2026-11-07 (b)) | matches (verified in code; param count from LOG) |
| Configs none, D, R, D+R, D+L; 3 training seeds 0/1/2 | `config_sources` `:104-107` ('none' -> [], else split on '+'); `lp_eval.py:54-55` names `dinov3`, `c-radiov4`, `dinov3+c-radiov4`, `dinov3+siglip2`; `TRAIN_SEEDS=(0,1,2)` `:54` | matches |
| Channel-wise concat [VAE 48 | source 1 | source 2], one expanded patch embedding and head, full 14x14 grid per camera (PLAN:94-97) | `Data.channel_slices` `:221-227`, `in_dim` `:218`, GRID `(14, 28)` `:64`; `LatentsOnly` `:366-369` | matches |
| Per-channel z-score with training-set statistics (PLAN:133, flagged as our choice) | stats summed over all 1,712 episodes `:196-206`; applied in `to_device` `:350-353` and in `rollout.py:316-318` with the checkpoint's saved stats; actions/proprio min/max over all training frames `:188-189` | matches; train-only, no LIBERO-Plus data touched (verified) |
| Future feature steps: step 0 = frame t clean, step 1 = mean of t+4..t+16, step 2 = mean of t+20..t+32 (PLAN:95, :134) | `feature_steps` `:125-134`, `future_means` `:137-157`, `feature_steps_cached` `:160-163` | matches (my reading of the index algebra agrees: group g starts at t+16g) |
| Loss = Fast-WAM's video loss split per source (Eq. 9, weight 1 each) + action flow matching, padded steps masked, first latent step excluded | `training_loss` `:396-427`: `pred_v[:, :, 1:]` `:415`; per-source loss `:419-422`; action loss `:423-426`; `sum(parts.values())` `:427` | matches, with the channel-mean point below |
| Flow matching, not MSE regression (user choice, PLAN:84) | Fast-WAM schedulers; `train_video_scheduler.add_noise/training_target` `:401-408` | matches |
| T5: umT5-XXL per-token, cross-attended, prompt template, 128 tokens, padded positions zeroed (PLAN:85, :126) | `t5_cache.py:25-50` (DEFAULT_PROMPT, `load_context` zeroes `~mask`, mask all ones); `to_device` `:357-358` | matches |
| Optimiser per paper Table 6: AdamW lr 1e-4 -> 1e-6 cosine, 5% warmup, wd 0.01, betas (0.9, 0.95), clip 1.0, bf16, batch 128, 10 epochs (PLAN:58) | `lr_at` `:466-471`; AdamW `:478`; `clip_grad_norm_(.., 1.0)` `:501`; bf16 autocast `:497`; defaults `--epochs 10 --batch-size 128` `:529-530`; 21,696 steps (277,713 frames x 10 / 128; LOG 2026-11-06 (a)) | matches |
| Seeds: seed sets data order and init; "same everything across configs" (PLAN:26) | `make_plan(lengths, epochs, bs, buffer_eps, seed)` `:270-286` depends on seed and episode list only (verified); `torch.manual_seed(seed)` `:476` then `build_model(data.in_dim, ...)` `:477` | data order shared across configs at one seed (verified); init and noise draws are not (see 3.7) |
| Batches from 16-episode buffers, not a full shuffle (PLAN:135) | `make_plan` `:278-285`, `--buffer-episodes 16` `:531` | matches and disclosed |
| No checkpoint selection | `torch.save(... out / "model.pt")` after the last step `:574`; `ckpt.pt` is deleted `:577-578`; rollout loads `model.pt` only `rollout.py:291` | matches (verified) |

Points where code and PLAN differ or PLAN is silent:

1. Per-source loss is a mean over the source's own channels and valid future steps (`proxy_wam.py:420-422`, docstring `:21-22`). Each source therefore carries equal weight whatever its channel count (D 1,024, R and L 1,152). The paper's Eq. 9 is a sum of per-source flow-matching losses and the paper does not say how each term is reduced (paper note, App. E.1/B.2: verified in the note). This is a choice that bears on suspect (2) of PLAN Next 2 (double counting) and is not listed under "Our choices where the paper is silent" (PLAN:131-135). Disclosed only in the code docstring. Low to medium.
2. The VAE channels get one combined loss term like any other source (slice `VAE`, `:419`), i.e. the video term is split by source, and the VAE term is a mean over 48 channels. Fast-WAM's own loss is recovered when only VAE channels exist (the docstring `:398-399` and the test claim byte-equality; recalled, test not run here).
3. `loss_lambda_video` / `loss_lambda_action` are passed as 1.0 but never read by `training_loss`; the sum is unweighted (`:427`). The values are 1.0, so there is no effect. Low.

### 1.2 Rollouts: what is evaluated

| PLAN says | `rollout.py` | Verdict |
|---|---|---|
| LIBERO-Plus, all 10,030 episodes x eval seeds 41/42/43 per checkpoint (Amendment 2026-11-06, PLAN:168-181) | `make_jobs` `:122-137` (every subset episode x every eval seed); `--eval-seeds 41,42,43` `:641`; subset = `results/full-benchmark/full-subset.json`, subset_id 91072e5da8499c99 (LOG 2026-11-06 (d)); `lp_eval.make_subset` with n = total returns every episode (`allocate` `:92-107`, test 1b ran and passed) | matches |
| Fast-WAM eval path: render 256, rotate 180, resize 224, replan every 10, 30 wait steps, no ensembling, 10 inference steps, text guidance 1.0 (PLAN:59, :122) | `RENDER, SIZE, NUM_STEPS_WAIT, REPLAN` `:68-71`; `process_obs` `:189-195`; chunk[:10] `:395`; `INFER_STEPS=10` (`proxy_wam.py:70`); no CFG anywhere | matches (as documented; Fast-WAM source not available to me, recalled) |
| Max steps 400 for spatial/object/goal, 700 for libero_10 (PLAN:59) | `MAX_STEPS` `:72`; loop bound `max_steps + NUM_STEPS_WAIT` `:386`. Pilot records show steps 430 / 730, consistent (LOG pilot_timings table) | matches. Whether Fast-WAM counts the 30 wait steps inside or outside the 400 is recalled, not checked |
| Success = the env's `done` | `res["success"] = bool(done)` `:405`; `done` from the last `env.step`, break on first True `:397-398`; wait-phase `done` ignored `:388` | matches |
| LIBERO-Plus episode ids -> tasks, init state 0, instruction with appended perturbation text stripped except for language perturbations (PLAN:63, App. C.2) | `libero_env` `:341-366` (`i = episode_id - 1`, `task.name == job["name"]` assert `:346-347`, `get_task_init_states(i)[0]` `:353`); `lp_instruction` `:99-106` | matches |
| Same episodes and seeds for every checkpoint | job list depends on subset, `--clean`, clean task list, eval seeds, `--order-seed` only (`:727-728`); shard = `[i::n]` of a shuffled list with a fixed seed `:140-149` | matches (LOG 2026-11-07 (c) check 1) |
| Clean LIBERO: "4 suites x 10 tasks x N rollouts. The paper uses 50 per task (2,000). Same per seed here." (PLAN:35) | `--clean 1` in the eval manifests: 40 trials, one per task, eval seed 41 only (`make_jobs` `:130-135`; LOG 2026-11-06 (d); PLAN:166 and :179 restate "40 trials") | deviation from the PLAN:35 wording, disclosed in the amendment and LOG, but PLAN:35 is not marked superseded. Clean LIBERO is context only in the rule, so low |

Other notes on the rollout path (verified in code): eval-seed handling per episode (`random/np/torch` seeded with the eval seed `:377-379`, `env.seed` `:365`, policy start noise from the eval seed `:109-115`, drawn once per (seed, channels) `:303-308`). So the three eval seeds differ in the policy's starting noise and in global RNG state; the simulator itself is otherwise fixed by the init state. The noise tensor is identical for every inference call and every episode under one eval seed (as the file says Fast-WAM does). This matters for section 4: the paper's eval-seed sd (0.04 to 1.01) also reflects only a few noise draws.

### 1.3 `lp_eval.py` against the locked rules

Outcome rule (PLAN:155-162) vs `verdict` `:236-255` and constants `:57-63` (verified):

| Locked text | Code | Match |
|---|---|---|
| (a) D - none: all three seeds > 0 and mean >= +2.7 | `a_ok = all(v > 0 ...) and mean >= CAL_A_MIN_MEAN` (Fraction 2.7) `:245` | yes |
| (b) D+L - D: mean >= -1.35 | `b_ok = mean >= CAL_B_MIN_MEAN` (Fraction -1.35) `:246` | yes |
| fail either -> "don't interpret the D+R comparison from this proxy" | `V_CAL_FAIL` `:77`, `:247-248` | yes |
| pass -> D+R - D: all three < 0 and mean <= -2.7 -> "substantial proxy effect (trigger ..., not proof)" else "does not show a substantial effect" | `:249-252`, strings `:78-79` | yes |
| Δ_s = mean over episodes of score(A,s) - score(B,s), per-episode score = mean of the three eval seeds, in percentage points, with discordant counts | `score` `:192-194`, `paired` `:211-226`, discordant `a_worse/a_better` `:222-223` | yes |
| Whole-episode resampling for any CI | `episode_ci` `:201-208` | yes |
| Refuse missing, duplicate, foreign rows | `lp_table` `:166-189` (asserts on subset_id, eval seed set, duplicates, episode coverage) | yes |
| Sizing helper, sd* = 1.5 sd, N = ceil((sd*/1.35)^2) | `sizing` `:258-267` | yes (now moot after the full-benchmark amendment) |

Seed-noise rule (PLAN:215-227) vs `seednoise` `:364-449`, `seednoise_outcome` `:339-345`, CLI `:516-529` (verified):

| Locked text | Code | Match |
|---|---|---|
| Compare `dinov3` seed 1 (A) vs seed 0 (B), full benchmark, per-episode score = mean over eval seeds 41-43, delta A - B with sign and the numbers of episodes where A is lower / higher | defaults `SN_CONFIG/A/B` `:68`; `delta`, `a_lower`, `a_higher` `:436-441`, `:448` | yes |
| Threshold T = 2.7 on \|delta\| (at or above -> "rethink") | `abs(delta) >= SN_T` `:343` | yes (>=, Fraction) |
| Floor: either seed's mean LIBERO-Plus success below 2.7 -> "can't tell; user decides", whatever the delta | checked first `:341-342` (`<`, so exactly 2.7 passes) | yes |
| Otherwise "go ahead" | `:345` | yes |
| Refuse unless every shard is complete | exactly 8 `*of8.jsonl` per seed `:374-375`; one config, one train seed per run with the expected values, one LIBERO-Plus subset_id, no duplicate keys, every episode has exactly eval seeds 41-43, the two seeds cover the same episodes, with `--subset` the file's episode list, without it exactly 10,030 episodes `:372-429`; any problem -> no outcome and exit 1 `:527-528` | yes, stricter than the text |
| Reads only `*of8.jsonl`, never `.errors.jsonl` | `Path(directory).glob("*of8.jsonl")` `:351`; the writer's error file is `<out>.errors.jsonl` = `...6of8.jsonl.errors.jsonl` (`rollout.py:505`), which does not end in `of8.jsonl` (verified; test 10 plants one and it is ignored) | yes |
| Clean-LIBERO rows excluded from the rule | `_sn_episodes` keeps `benchmark == LP` only `:355-361` | yes |

No deviation found between the PLAN text and either rule's code. Two small observations: (i) `--partial` bypasses every failure and prints no outcome, correct and tested; (ii) the script does not check that the files were written by the checkpoint paths the LOG names (it checks `train_seed` and `config` in the rows instead), which is the right invariant.

### 1.4 Deviations list and whether the LOG discloses each

| # | Deviation or choice | Where disclosed |
|---|---|---|
| 1 | Random-init ~32M-parameter model, no Wan 2.2 prior, no pretrained action expert | PLAN:8, :98, :129; STANDARDS-level framing in decision 003 amendment. Disclosed |
| 2 | Per-channel z-score of source features | PLAN:133, :90, Next 2 suspect (1). Disclosed |
| 3 | Time averaging of future features in groups of four | PLAN:95, :134. Disclosed (flagged speculative) |
| 4 | Encoder input resolution: DINOv3 224 (14x14), C-RADIOv4 448 (28x28 -> bilinear 14x14), SigLIP 2 384 (27x27 -> 14x14) | PLAN:132; Next 2 suspect (3). Disclosed |
| 5 | 16-episode shuffle buffers | PLAN:135; code docstring. Disclosed |
| 6 | Per-source loss is a channel mean | code docstring only (`proxy_wam.py:21-22`). Not in PLAN or LOG as a choice |
| 7 | Clean LIBERO 40 trials, one eval seed | LOG 2026-11-05 (d) section 4, PLAN:166, :179. Disclosed, but PLAN:35 still says 50 per task |
| 8 | Evaluation on the full benchmark instead of the proportional subset | PLAN Amendment 2026-11-06, user decision with quote. Disclosed |
| 9 | Seed 0 trained on an A40, seed 1 on an RTX 3090; seed 1 shard 6's first 135 records and seed 0 shard 0's first 1,359 records came from other card models than the rest | LOG 2026-11-06 (c), 2026-11-07 (e), (i), (o). Disclosed; records do not store the GPU model, so the split of rows is only in the LOG |
| 10 | Rollouts use the shuffled job order (`--order-seed 0`) | LOG 2026-11-06 (b), user OK. Disclosed |
| 11 | CUDA-graph replay of the denoising loop | `rollout.py:238-273`, LOG 2026-11-03; `profile_policy.py` compares to eager (recalled). Disclosed |

## 2 Fidelity to the paper

Source of the paper's recipe: `literature/notes/liu2026-what-visual-representations-wams.md` (checked by the earlier agent against arXiv HTML v1) and `liu2026what-table10-composition.csv`. Table 10: D+R 78.59 (sd 1.01), D+L 85.83 (sd 0.28); Table 1 D 84.0, VAE-only 74.7 (note). The harm threshold 2.7 = half of 84.0 - 78.59 = 5.41 (verified arithmetic).

Where the proxy matches the paper (verified from the note and PLAN): joint video-action denoising as the default formulation (App. B.1); flow matching; channel-wise concatenation with future-feature prediction and current features kept clean (Sec. 3.3, App. B.2); per-source feature losses with weight 1.0 (App. B.2, Eq. 9); LIBERO training on Fast-WAM's 1,712 episodes at 224 per view with 9 frames at offsets 0, 4, ..., 32 (Table 6/7); Table 6 optimiser values; evaluation protocol from App. E.3 (10 inference steps, replan 10, 30 wait steps, no ensembling, binarised gripper, eval seeds 41-43, 10,030 episodes).

Differences, each with a reason on record:

| Difference | Recorded? | Reason on record |
|---|---|---|
| ~32M-parameter model from random init instead of a Wan 2.2-based WAM (a 5B-class video expert, recalled) | yes (PLAN:8, :98) | compute; no code or checkpoints released (note, verified 2026-10-17). The one gap "we can't close" |
| One training checkpoint per config in the paper; we train three seeds | yes (decision 003) | the point of the round |
| Feature scale handling | yes (z-score per channel) | paper says only "source-specific processing"; our choice, flagged |
| Which grid reaches the model for R and L | yes (448/384 input resized to 14x14) | paper says only that features are aligned to the VAE grid and that spatial resizing is bilinear (App. B.2; docstring `fastwam_data.py:grid_to_vae`); our choice |
| Time alignment of features to the 3 latent steps | yes (flagged speculative) | paper states averaging in groups of four only for REPA |
| Encoder checkpoints (DINOv3 vit-L/16 `facebook/dinov3-vitl16-pretrain-lvd1689m`, C-RADIOv4 pinned revision, SigLIP 2 so400m) | yes (LOG, STANDARDS "variants often unstated") | paper says "primarily large (L) variants", no checkpoints named (note App. A) |
| Equal loss weight per source and per-channel-mean reduction | partly: weight 1.0 yes (suspect 2), the reduction no | see 1.1 point 1 |
| Frame source: training frames are AV1-compressed 512x512 videos resized with torchvision antialias; rollout frames are 256x256 renders resized with PIL bilinear | yes (PLAN:50-52: "Fast-WAM's own train/eval difference included") | follow Fast-WAM; a codec/resize gap that affects all configs but could interact with encoders differently (speculative) |
| Action expert initialisation | yes (PLAN:98) | paper silent |
| LIBERO-Plus eval episodes and seeds | none (matches) | n/a |
| Clean LIBERO 40 trials vs the paper's 2,000 | yes | context only |

Undocumented or thinly documented shortcuts (flag list): (1) the channel-mean loss reduction (1.1 point 1); (2) training environment: top-level pins in `src/setup_env.sh:6-12` and `requirements.txt` are exact, but transitive packages and `fastwam`'s `--no-deps` install are not pinned and the `pip freeze` line (`setup_env.sh:12`) goes to `/work/pip-env.log` on an emptyDir, so it is lost when the pod ends (LOG 2026-11-06 (a) note (a)); the two seed-check runs' resolved environment is unrecorded; (3) no ablation shows the 224 vs 448 choice for C-RADIOv4 matters or not; this is the very grid-mismatch suspect the planned Next 2 would test, and it means a null here cannot rule out that our own choices hid the paper's drop (the PLAN says so at Next 2). I found no shortcut that the repo rule "fidelity over cheapness" forbids and that is hidden from the record, other than (1) and (2).

## 3 Correctness risks

3.1 Eval-seed handling (verified): per-episode seeding is independent of batching (`rollout.py:376-379`, `:303-308`); episodes under the three eval seeds are separate jobs with distinct keys (`make_jobs` asserts uniqueness `:136`). The three eval seeds are three draws of the same start-noise tensor family, not three independent environments. No bug.

3.2 Episode and task indexing across shards (verified): classification id k maps to task index k-1 with a name assert (`:345-347`); the pod list is the same shuffled list on every pod, `[i::n]` (`:727-728`), so shards partition it (tested by 2b/2c in `test_rollout.py`, not run by me); the manifests give shard i the same job list for both seeds (LOG 2026-11-07 (c) check 1; I confirmed the code path). `check_resume` (`:152-161`) refuses a file with records outside the shard's job list. Totals: 3,767 jobs in shards 0 and 1, 3,766 in the other six = 30,130 = 30,090 LIBERO-Plus + 40 clean (arithmetic, matches LOG (o)).

3.3 Success counting (verified): `done` from the env, no post-hoc override; unfinished episodes are failures only by reaching `max_steps + 30`; episodes that raise are not recorded as failures but go to the errors file (`:529-533`), and `lp_table` then refuses the incomplete table (`lp_eval.py:184`). No silent zero-fill. Zero errors in all progress lines the LOG cites (LOG (n), (o)).

3.4 Deduplication when a shard is resubmitted (verified): resume skips keys already in the shard's own file (`done_keys` `:474-481`, `run_jobs` `:488`); `check_resume` blocks changed settings. If two shard files ever held the same key, `lp_table` (`:179-180`) and `seednoise` (`:399-406`) fail loudly; neither dedupes silently. Hazards, all handled in the LOG: a half-written last line would crash startup (no try/except in `done_keys`, `:479`), checked clean by a read-only Job before resuming shard 6 (LOG 2026-11-07 (e)) and by a read-only exec for shard 0 (LOG (k), (o)); concurrent writers to one file would corrupt it (the orphan-pod episode in LOG (k)-(m) had one pod per file). One residual risk (speculative): a resumed shard mixes card models (135 A40 rows then L4 rows for seed 1 shard 6; 1,359 L40 rows then L4 rows for seed 0 shard 0); rows carry no GPU field, so any later GPU-numerics question can only be answered from the LOG.

3.5 `.errors.jsonl` (verified): never read by `seednoise`. `analyze` is passed whatever `--results` files the user lists; RUN.md step 12 shows `results/rollouts/*.jsonl`, which would also match `*.errors.jsonl`. Those rows have keys `job` and `traceback`, so `read_rows` fails its `FIELDS` assert (`lp_eval.py:160-161`) rather than miscounting. Fail-safe, but the RUN.md glob should say `*of8.jsonl`. Low.

3.6 z-scoring and leakage (verified): feature and action/proprio statistics come from the 1,712 training episodes only (`proxy_wam.py:188-189`, `:196-206`) and the checkpoint carries them (`:558-559`); rollout uses the saved stats (`rollout.py:282-285`). LIBERO-Plus perturbs the same 40 base tasks that appear in training; that is by design (the paper does the same), not leakage by this code. No tuning on the evaluation set: model size, learning-rate schedule and epochs are fixed from the paper and one 500-step pilot whose results never enter the analysis (PLAN:150). The 40 clean tasks and the LIBERO-Plus tasks overlap with training scenes (recalled), so clean LIBERO is not a held-out test either.

3.7 Pairing across configs at one training seed (speculative, from code reading): the PLAN note (PLAN:227) says D and D+R at the same seed are probably less noisy to compare than seed 1 vs seed 0, because `make_plan` depends only on the seed. That part is verified (`proxy_wam.py:270-286`). But `torch.manual_seed(seed)` (`:476`) is followed by `build_model(data.in_dim, ...)` (`:477`) and by `torch.randn_like` draws shaped by the channel count (`:401`, `:406`); different `in_dim` per config means different random initial weights and different noise draws from the same seed. So only the data order is shared. If init and noise matter more than data order, the same-seed D+R - D delta is as noisy as the cross-seed delta, and the seed-noise check is not conservative. Not measurable until more runs finish.

3.8 Checkpoint selection (verified): none; final step only.

3.9 Training seeds vs hardware (speculative): the 13 remaining runs will land on whatever of the 3090/4090/A5000/L40/L40S (and named A40) cards free up; config is not randomised across card models, and seed 0 vs 1 already differ. bf16 non-determinism across card models adds noise that is not separable from "training-seed" noise. Low.

3.10 Potential environment drift between pods (speculative): `job-train.template.yaml` installs packages with pip on each pod and the rollout pods restore a prebuilt bundle (LOG 2026-11-06 (a) check (c)); the training pip path could resolve a different transitive version on a later date. Medium to low; the fix is to save `pip freeze` to the PVC.

## 4 Statistics

Inputs that exist (all from files in the repo): paper D+R - D = 84.0 - 78.59 = 5.41 points; harm threshold 2.7; calibration margin 1.35; paper D - VAE-only = 84.0 - 74.7 = 9.3 points; paper D+L - D about +1.8 points; paper eval-seed sd across the 16 Table 10 rows: mean 0.41, largest 1.01 (D+R), smallest 0.04 (arithmetic from the CSV). From our runs: the only measured variance numbers are timings; the pilot's sd (3.33) rests on one successful rollout in 600 and is explicitly unusable (LOG 2026-11-05 (d) section 3). No full-run success rate has been read yet, so our own training-seed variance is unmeasured (verified: no such number in LOG or results).

Episode sampling noise (speculative bound): scores per episode lie in {0, 1/3, 2/3, 1}, so the per-episode paired difference lies in [-1, 1]; with 10,030 episodes, SE of a paired mean difference = sd(d)/sqrt(10,030) = sd(d) x 0.00999. Even sd(d) = 1 gives 1.0 point; a plausible sd(d) of 0.3 to 0.5 (a quarter of episodes discordant) gives 0.3 to 0.5 points. That is small against 2.7 and 1.35. The real uncertainty is the run-to-run (training) noise.

Power of the locked rule with 3 training seeds (speculative; simulated, 2,000,000 draws per cell, assumptions: the three per-seed deltas are independent Normal(mu, s^2); s is the sd of one D+R - D delta from training noise, i.e. a difference of two runs; the rule fires when all three deltas are below 0 and their mean is at most -2.7). I use s = 2.7 as a scale because that is the boundary at which the seed-noise rule says "rethink"; 1.35 and 5.4 are half and double.

| true mean delta mu | s = 1.35 | s = 2.7 | s = 5.4 |
|---|---|---|---|
| 0 (no effect) | 0.000 | 0.033 | 0.099 |
| -1.35 | 0.041 | 0.158 | 0.184 |
| -2.7 | 0.496 | 0.428 | 0.300 |
| -5.41 (paper's size) | 1.000 | 0.914 | 0.579 |

Reading: if the proxy's true drop matched the paper's and training noise is at or below the seed-noise boundary, 3 seeds detect it about 9 times in 10 or better; if the true drop is only half (2.7), it is close to a coin flip at any s; at s = 5.4 the false-positive rate is about 1 in 10 and power for the paper's size is about 0.58. Calibration (simulated the same way): (a) D - none passes in about 1.0, 1.0 and 0.88 of draws at s = 1.35 / 2.7 / 5.4 if the true gain is the paper's +9.3, and 0.99 / 0.83 / 0.50 if it is half; (b) D+L - D non-inferiority passes in 0.96 / 0.81 / 0.67 of draws at s = 1.35 / 2.7 / 5.4 if the true effect is 0 and 1.00 / 0.98 / 0.84 if it is the paper's +1.8. So a false calibration failure is plausible when s is 2.7 or more; the PLAN already says (b) is tight and fails safe.

Two effects the table does not include (speculative): (i) selection: D+R is the one outlier among the paper's single additions to D (the other six, D+V, D+L, D+S, D+G, D+H, D+U, sit between 84.40 and 86.12 against D's 84.0, verified from the CSV and Table 1 value in the note), so its true drop is probably smaller than 5.41 (winner's curse) and the paper's own eval-seed sd (1.01) does not cover training noise; (ii) the proxy lacks the pretrained prior, so its true drop may be smaller still or zero. Both push toward the mu = -2.7 row or higher, where power is about 0.3 to 0.5. Verdict: 3 seeds can detect a drop as large as the paper's if training noise is moderate; they cannot reliably detect half of it. The seed-noise result (one cross-seed delta, one degree of freedom) will indicate s only roughly: a single draw from Normal(0, 2 s_run^2) has a 31 percent chance of falling below 0.4 of its sd, so a "go ahead" is weak evidence. The PLAN already calls it "a rough guide" (PLAN:197).

## 5 Compute and cluster compliance

Manifests (verified by grep of `jobs/rendered/*.yaml`; `sleep` and `tolerations:` appear in no command, only comments):

| Item | Finding |
|---|---|
| Job, not bare pod; `restartPolicy: Never`; `backoffLimit` 4 (08 template says 2, rule is "> 0"); `ttlSecondsAfterFinished` 86400 | pass (e.g. `job-eval-dinov3-s0-l4.yaml:61-68`) |
| Requests vs limits within 20% | CPU 4 / 4800m (1.20), memory 28Gi / 33Gi (1.18), ephemeral 50Gi / 58Gi (1.16) in all eval manifests; training 28Gi / 33Gi, ephemeral 20Gi / 24Gi; pass |
| Sized from measured use | CPU: pods used 3.3-3.9 of 4 cores (82-99%, LOG (n)/(m)); memory: working set 13.3-19.1 GB of 28Gi = 30.1 GB (44-63%, LOG (f)-(h)); both inside NRP's 20-200% / 20-150%. Memory was raised from 20Gi to 28Gi on a reviewer's drift extrapolation (LOG 2026-11-06 (e)); the measured plateau (13-19 GB) stays above 20% of the request. Pass |
| `activeDeadlineSeconds` | training 129,600 s (36 h) against 4-6.4 h estimates; eval 345,600 s (96 h) against about 9.5 h per pod; counted from Job creation (RUN.md notes this). Generous, consistent with the queue waits seen (up to about 9 h). Pass |
| No `tolerations`, no `sleep`, us-west, driver > 569, two bad hosts excluded, GPU product list valid | pass; the A40/A6000 `gpu.product` error was found 2026-11-06 and fixed by named-resource twins (PLAN Amendment (b)) |
| Gate: at most 8 GPUs at once | respected as far as recorded: LOG (a) 02:40 computed 9 > 8 and avoided it; (g) "4 + 2 + 2 = 8"; (n) "4 + 3 + 1 = 8". The LOG's totals are by parallelism, not by running pods |
| Gate: sweep of more than 20 Jobs | ambiguous in STANDARDS:48; concurrent Jobs stay well under 20. The round has created more than 20 Job objects in total (36 files in `jobs/rendered/`, several resubmits), which I read as not a sweep. Low |

GPU utilisation (08: above 40%, ideally near 100%):
- Training: 76-97% (A40 and 3090 runs; watcher polls, LOG 2026-11-07 (b)); pass.
- Evaluation: whole-run in-pod means for finished L4 pods 46.8-51.7% (LOG (k), (n), (o)); windows of 30-45% and some 5-minute windows lower (LOG (h)). The L4 nodes `nautilus-it-gpu05/06` export no DCGM series (LOG (h), `_standards/08`), so NRP's portal code cannot see these pods at all (recalled from `_standards/08` item 3); our own sampling is the only check. The standard says "ideally near 100%"; the evaluation workload is CPU-bound by design (policy busy 47-52%, mean batch 1.2-1.4), so average use near 50% is what it can reach on an L4 (speculative). Pass on the stated 40% line, thin margin.
- Fast cards failed: the pilot L40 pod averaged 42.6% (LOG 2026-11-05 (b)); the A40 eval pod 19.95% over 10 minutes (LOG 2026-11-07 (d)); the L40 pod 19.4% over 3 h and 22.8% over 6 h (LOG (i), (k)). The L4-only decision (user: "Swap to L4-only (Recommended)", LOG (k), PLAN Amendment 2026-11-07) follows from this and is consistent with 08.
- Process lapse on the L40 pod (verified from LOG (i)/(k) times): the watcher exited 4 (GPU idle) at 18:40:53 on 2026-11-07, the lead looked at about 19:50-20:05 (laptop asleep from about 12:00), the user decided at about 22:06, and the pod was deleted at 22:07; NRP's warning email arrived before that. RUN.md stop 2 says to stop and tell the user on watcher exit 4. The risk that fast cards would fail again was written down at 04:51 the same day (LOG (d): "the regular list still includes the fastest cards") but the list was not narrowed until after the email. Medium.
- Orphaned pods (verified, LOG (k), (m), (n)): `--cascade=orphan` on the two eval Jobs left two L4 pods with no owning Job. They lost the Job's `backoffLimit`, `activeDeadlineSeconds` and retry, and fall under NRP's "bare pod destroyed in 6 hours" wording (`_standards/08` table). Both passed 6 h without removal and finished. The "Swap to L4-only" option text the user chose did not state this consequence (LOG (k) option text vs (m) lead correction). Disclosed; the user then chose "Let them finish". Low now, medium as a pattern.
- Pre-flight item 8 (`_standards/08` line 220: check the Violations page about 30-40 minutes after launch): I find no entry recording a read of the Violations page for any eval Job; the A40 entry says "Not known whether NRP's Violations page flagged this pod. The user can check" (LOG (d)); (m)/(n) note the portal could not be read (NRP web services offline for maintenance until Monday, November 9, LOG (n)). Medium; the user's NRP idle-GPU warning email is the one event that did reach the user.
- Idle-pod rules: no interactive or sleeping pods in the manifests. The perturbed-frames validation Job ran into `DeadlineExceeded` while Pending (LOG (n)); no GPU was held.

Estimated vs actual GPU-hours (STANDARDS:49: "Log estimated vs. actual GPU-hours per run anyway"):
- Training: estimated 4.3-6.4 h per run before submit (LOG 2026-11-06 (a) section 6); actual 3.9 h (seed 1, 3090) and 4.0 h (seed 0, A40) (LOG (c), 2026-11-07 (b)). Recorded.
- Evaluation: estimated 7.8-8.4 h running per shard plus 0.2 h setup (LOG 2026-11-06 (d)) from the pilot speed of 449-484 episodes per hour; actual on L4: 385-411 episodes per hour, 9.2-9.7 h per shard (LOG (k), (n)). Seven finished L4 shards in LOG (k)/(i) total 238,329 s = 66.2 GPU-h; the four shards in (n) add about 38.2 GPU-h (3,766 records / the logged episodes per hour: 9.25 + 9.68 + 9.51 + 9.71, arithmetic). So 12 shards are about 115 GPU-h including the L40 and A40 partial pods (about 4 h and 0.3 h, estimated from the LOG), plus 7.9 GPU-h of training, excluding setup of about 8 minutes per pod, extraction, pilot and the idle Pending time (arithmetic, speculative where I estimated the L40 and A40 hours).
- The PLAN's whole-round estimate (959-1,044 GPU-h for 15 checkpoints, PLAN:181) versus the running average of about 76 GPU-h per checkpoint (8 shards x 9.46 h) gives about 1,135 GPU-h, 9 to 18 percent above the estimate (arithmetic from the numbers above). I find no entry that records estimated versus actual per Job in one place, nor a running total for the round. Finding: the standard asks for it "per run", the LOG has the pieces but not the comparison. Low to medium.
- Budget: STANDARDS:44 says no GPU-hour cap, only the 8-GPU / 20-Job gate and the `08` utilisation rules (override table at STANDARDS:60). The round is within the gate as recorded. The user waived RUN.md's 7-day stop 5 for this round (PLAN:187; RUN.md itself is unchanged and still says 7 days). Projection (speculative): 13 more checkpoints x about 76 GPU-h = about 990 GPU-h of rollouts plus 13 training runs (4 h for D, probably more for the larger configs; the pilot D+R pace on a 3090 was 1.06 s/step = 6.4 h, recalled from LOG); at the full 8 GPUs that is at least 123 h of perfect occupancy for rollouts alone (about 5 days); with the L4-only list (the LOG reports 2 of 536 nodes match, unchecked) and queue waits of 3 to 9 h seen so far, weeks are more likely.

## 6 Documentation and process

- Append-only (verified): the experiment LOG appends corrections as later entries ((o) corrects (n)) or as dated "Review corrections" blocks inside an entry; I found no silent rewrite in the text I read. PLAN keeps the locked rule text and adds dated amendments (PLAN:164-187, :247-260). Mixed date convention: entry (d) of 2026-11-05 is dated by Pacific date, later ones by UTC, and entry "2026-11-09 (o)" is dated by UTC while the commit is 2026-11-08 17:01 PST; the LOG itself says so at (d). Low.
- Labels: most substantive claims in the entries carry verified/speculative or an explicit "(speculative)". A few reasoning sentences in (h), (m) and (n) say "Reading (speculative)" correctly. No unlabelled number found.
- Decisions with who decided: user decisions are quoted verbatim in PLAN amendments and LOG entries (full benchmark, A40 race, L4-only, parallelism, "Let them finish"). Gaps (agent decisions on points that are the user's under `05` or the memory rule "questions aren't go-aheads"):
  1. Seed-noise rule locked on the user's silence: the user said "i just want you to stay as faithful to the paper as possible", the lead replied that it would lock threshold option 1 and floor option 2 unless the user objected, and treated silence as agreement (LOG 2026-11-06 (c) section 5, which flags it). Locking a threshold is a change to the evaluation protocol (gate in `05`). The user has not confirmed it in words. Medium.
  2. The wave-1 training Job was submitted on the user's "yeah the full benchmark is fine just get started on it now while i go to sleep", which did not name the training Job (LOG 2026-11-06 (a) section 1, which flags it). Low (the direction PLAN item 1 had said the Job needs an OK).
  3. The `--cascade=orphan` consequence was not in the option text the user chose (LOG (k) vs (m)). Low, disclosed.
  4. The standing approval "go raise eval parallelism after the GPU check" was used for several raises within the 8-GPU gate. Fine, but each raise is the lead's judgement of "after the GPU check", with in-pod 5-minute samples on pods that had run 6-11 minutes (LOG (h) s1-main -1). Low.
- Verification gates: no proxy result is in README key findings (verified: Key findings contain only the diagnostics results); no data deletion; no push. Nothing about outcomes was read before the rule was locked (LOG (c), (l) say so).
- Status staleness (verified): direction `README.md:3` and `PLAN.md:8` and the experiment `README.md` status say "7 of 16 shards done" as of 2026-11-07 22:26 UTC; the experiment LOG is at 12 of 16 (entry (o)). `PLAN.md:11` still says the experiment LOG "has no entry yet" for the phase-1 approval and Jobs, but (m) and (n) exist now. The README gap is dated, so not false, but `05` asks for the status to be current at session end. The direction `LOG.md` has no summary entry for experiment entries (m), (n), (o) (the last proxy-related direction entries are the 2026-11-07 status refresh and "seed-noise script merged"); `04` asks for one. `_standards/08` line 6 still says the NRP-code findings are "on branch `proxy-calibration-nrp-policy-j`, ... not yet merged into main"; LOG (k) says they were merged (d498883). `RUN.md` still describes the 40% floor as a hard stop over 10-minute windows (stop 2, stop 10, watcher exit 4) after LOG (j) showed the code averages over 1 day / 3 h / 6 h; the LOG records the open item (k). Low to medium.
- Open item for the user (not a defect): LOG (i)/(h) record the user's choice "Keep running, ask NRP"; I find no record that NRP support was asked or answered.

## 7 Does this line look promising?

All of this is speculative.

Plain version: the experiment is a smaller stand-in for the paper's robot model, built from the same parts and trained with the paper's settings, and it is being scored on exactly the paper's test (all 10,030 LIBERO-Plus episodes, three evaluation seeds). Two copies of the plain DINOv3 version have been trained and almost all of their test runs are finished. The first real number, how much two identical-recipe copies differ from each other, comes out when the last four shards finish, probably within about a day. That number tells us whether three copies per setting can show a drop of the paper's size at all.

Will it settle whether D+R hurts? Not the paper's question. At best it settles whether the stand-in shows the drop. If it does (and the calibration holds), that is a trigger for a higher-fidelity run, not proof. If it does not, the paper's drop might still be real, because the stand-in lacks the pretrained video model and uses our own choices for scale, grid and loss reduction, which are exactly the three suspects in PLAN Next 2. So a null here is weak evidence; the design says so already, and the code and rule texts do not overclaim.

Biggest risks, in order:
1. Throughput. About 13 more checkpoints at roughly 76 GPU-hours of evaluation each, on L4 cards only, with a queue of hours per shard, is probably weeks, not days, unless more cards qualify. The user has waived the 7-day line, but the schedule is the thing most likely to stall the round.
2. Signal floor. The 500-step pilot models almost never succeeded (1 of 600 rollouts). The full 21,696-step models are untested at scale; a 32M-parameter from-scratch model may score low on perturbed scenes, which makes every paired difference small and the floor rule (mean at least 2.7 points) the first thing to trip. The seed-noise result reads this out directly.
3. Our own choices may hide the effect (the three fusion suspects), which is why Next 2 exists.
4. Training noise unknown: at the seed-noise boundary (2.7 points) three seeds are adequate for the paper's full drop and weak for half of it (section 4).

What looks good: the analysis code is exact, tested and fails loudly; results do not depend on batching or order; the 40% rule and the 8-GPU gate are being managed; every user decision is quoted.

## Fix list

| # | file:line | What is wrong | Standard or PLAN line it touches | Suggested fix | Severity |
|---|---|---|---|---|---|
| 1 | `README.md:3`, `PLAN.md:8`, `experiments/2026-10-27-proxy-calibration/README.md:5` | Status says 7 of 16 shards (2026-11-07 22:26); LOG is at 12 of 16; `PLAN.md:11` says the experiment LOG has no entry for the phase-1 approval, but (m) and (n) exist | `05` end-of-session step 3; `04` "keep README current" | Refresh the three status lines from LOG (o); drop the "no entry yet" clause | medium |
| 2 | `LOG.md` (direction) | No direction-level summary of experiment LOG (m), (n), (o) (L40 pod, orphan pods, shards 8-12 done, A40/validation Job expiry) | `04` "short summary in the direction LOG" | Add one `[agent]` summary entry pointing to (m)-(o) | medium |
| 3 | `PLAN.md:215-227` (seed-noise rule) and experiment LOG 2026-11-06 (c) §5 | Threshold and floor locked on the user's silence | `05` verification gate "changing ... evaluation protocol"; memory rule "approval must be in-session" | Ask the user to confirm T = 2.7 and the floor in words before the verdict is acted on; record the answer as an amendment | medium |
| 4 | `src/proxy_wam.py:420-422` and PLAN:131-135 | Per-source loss reduced as a channel mean; not listed among choices where the paper is silent; bears on suspect (2) of Next 2 | `01`/`06` "log changes"; fidelity rule | Add one line to PLAN "Our choices where the paper is silent" | low |
| 5 | `src/setup_env.sh:12`, `jobs/job-train.template.yaml`, LOG 2026-11-06 (a) note (a) | `pip freeze` goes to `/work/pip-env.log` on an emptyDir; resolved transitive versions of the training environment are not recorded | `03` "pin the environment", `07` "record exact versions" | Print the freeze into the pod log (`tee`) or copy it to the PVC; for the 13 remaining runs record it | medium |
| 6 | `PLAN.md:35` vs `:166`, `:179` | "50 per task, same per seed" never marked superseded by the 40-trial clean list | `04` locked text + amendments | Add a one-line dated note that clean LIBERO is 40 trials at eval seed 41 (context only) | low |
| 7 | `proxy_wam.py:476-477` and `PLAN.md:227` | Same-seed configs share data order but not initialisation or noise draws; the note that the same-seed comparison is "probably less noisy" is not supported by the code | `01` "report uncertainty" | Soften the PLAN note, or seed the init per module name so configs share the common weights; do not change before the 13 runs unless the user decides (it changes the design) | medium |
| 8 | `_standards/08-compute-nautilus.md:6` | Says the NRP-code notes are on an unmerged branch; they were merged (LOG (k)) | `08` accuracy | Update the sentence | low |
| 9 | `RUN.md` stop 2, stop 10, watcher exit 4, "Rollouts and the 40% GPU floor" | Still treats 40% over 10-minute windows as a hard stop; LOG (j)/(k) show NRP's code averages over hours | `08` GPU section | Update after the user answers the open NRP question (LOG (k)) | low |
| 10 | Experiment LOG, whole round | No per-Job estimated-vs-actual GPU-hours table or running total; actual evaluation cost is 9-18 percent above the PLAN estimate | `STANDARDS.md:49`, `07` "Compute and cost" | Add a short table (Job, estimate, actual GPU-h) at round end; start now with the numbers in section 5 of this review | low |
| 11 | Experiment LOG entries (d)/(i)/(k) | Watcher exit 4 on the L40 pod at 18:40 was acted on about 3.5 h later; the fast-card risk was recorded at 04:51 but the list was narrowed only after NRP's email | RUN.md stop 2; `08` | Narrow the remaining shards to L4 only (done); add a push notification or a shorter poll on watcher exit 4 | medium |
| 12 | Eval Jobs, `_standards/08` pre-flight item 8 | No record of reading the Violations page 30-40 minutes after launch for any eval Job | `08` pre-flight 8 | Ask the user to read https://nrp.ai/userinfo after maintenance ends and log it | medium |
| 13 | `RUN.md` step 12 | Example glob `results/rollouts/*.jsonl` matches `*.errors.jsonl` | `03` | Change to `*of8.jsonl`; `read_rows` would fail loudly anyway | low |
| 14 | `jobs/rendered/job-eval-dinov3-s0-l4.yaml:62`, `job-eval-dinov3-s1-l4.yaml:57` | Live parallelism (4 and 3) differs from the committed files (3 and 3 then 2) | `03` "config is the source of truth" | Add the live values to the LOG (done in (n)); optionally commit a note in the manifest header | low |
| 15 | `src/test_lp_eval.py:59`, `:313`, `src/test_proxy_policy.py` | pytest collects 1 of 10 checks in `test_lp_eval.py` and none in `test_proxy_policy.py`; the checks run only as scripts; `test_proxy_wam.py` and `test_rollout.py` need `fastwam` | `07` "sanity checks" | Name the script commands in RUN.md "Session start", or wrap each check in a `test_*` function | low |
| 16 | Rollout rows (`rollout.py:467-471`) | Rows carry no GPU model or pod name; mixed-card shards can only be reconstructed from the LOG | `07` "record hardware" | Add `gpu` (from `torch.cuda.get_device_name`) to the record for the remaining 13 checkpoints (a code change, so the user decides) | low |
