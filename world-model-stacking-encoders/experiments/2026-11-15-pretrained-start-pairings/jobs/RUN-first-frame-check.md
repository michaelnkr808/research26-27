# First-frame feature check: steps for the lead

Written 2026-11-26 by a worker on the Mac. Nothing here has been run on the cluster and no kubectl command was used. Two Jobs, one L4 each: `jobs/job-first-frame-check-dryrun.yaml` (Job `stacking-encoders-pairings-first-frame-check-dryrun`) first, then `jobs/job-first-frame-check.yaml` (Job `stacking-encoders-pairings-first-frame-check`). Both run `src/first_frame_check.py run`, which implements the merged plan `diagnosis/2026-11-25-first-frame-check-plan.md` (Design 1 to 6, reading rule, descriptive items, amendment of 2026-11-25) as written. Paths are relative to this experiment folder unless they start with `src/` (= `world-model-stacking-encoders/src/`); `$C` is the source commit. Labels: **verified** (read in the code or a committed file this session, or run on the Mac), **recalled**, **speculative**.

**Each submit needs the user's go in the lead's window.** State before asking, per Job: one L4 GPU; dry run about 15 to 30 min of pod time, full run about 25 to 65 min (both speculative, basis in the manifest headers; the dry run measures the parts that set the full run's time); the GPU is mostly idle by design (batch-1 encoder calls on 200 frames and one-frame VAE encodes; the plan needs the L4 for the live features), so the whole-pod GPU mean will likely be under 40% (speculative), in a pod far shorter than the portal's 180-minute minimum for the idle warning; reads the caches, the dataset and the three checkpoints (the checkpoints only for their sha256 and `feature_stats`); writes one new dated folder under `/mnt/proxy/first-frame-check/` and a log under `/mnt/proxy/logs/`, nothing else. Test-set use: first frames of clean LIBERO evaluation init states 0 to 4, no rollout, no success outcome (plan "Test-set use"); log it in the stage 2 `LOG.md` at submit.

## Steps

1. **The user's go**, after the statement above (dry run first; the full run is a second go or the same go with the dry run named as its gate, the user's choice).
2. **Source commit** `$C`: a clean, pulled `main` that contains this branch (the script is merged before either Job runs): `git rev-parse HEAD`. Rerun the Mac tests there: `/opt/homebrew/bin/python3.11 -I world-model-stacking-encoders/src/test_first_frame_check.py` (18 tests).
3. **ConfigMap** `stacking-encoders-pairings-firstframe-src`, 12 keys (the 11 keys of the eval ConfigMap `stacking-encoders-pairings-eval-src`, which are rollout.py's import chain plus the env-bundle scripts, and `first_frame_check.py`). From `jobs/`:
   ```
   S=../../../src
   K="first_frame_check.py rollout.py lp_eval.py proxy_wam.py t5_cache.py diagnostics.py extract_fastwam.py extract_features.py fastwam_data.py setup_env.sh setup_sim.sh env_bundle.sh"
   kubectl create configmap stacking-encoders-pairings-firstframe-src -n aria-world $(for f in $K; do printf -- '--from-file=%s=%s/%s ' $f $S $f; done) --dry-run=server
   ```
   then the same without `--dry-run=server`. Read each key back and compare with `$C`:
   ```
   for f in $K; do
     a=$(kubectl get configmap stacking-encoders-pairings-firstframe-src -n aria-world -o jsonpath="{.data.${f//./\\.}}" | shasum -a 256 | cut -c1-64)
     b=$(git show $C:world-model-stacking-encoders/src/$f | shasum -a 256 | cut -c1-64)
     [ "$a" = "$b" ] && echo "ok $f" || echo "DIFF $f"
   done
   ```
   Any `DIFF`: stop. The 11 shared files had the same sha256 on main `99664a1` as the ones rendered into `jobs/rendered/job-smoke-dinov3-s0.yaml` (verified, `shasum -a 256` this session), so the env bundle key stays `9182f8131559`.
4. **Render** both manifests (from this experiment folder):
   ```
   K="first_frame_check.py rollout.py lp_eval.py proxy_wam.py t5_cache.py diagnostics.py extract_fastwam.py extract_features.py fastwam_data.py setup_env.sh setup_sim.sh env_bundle.sh"
   for J in job-first-frame-check-dryrun job-first-frame-check; do
     Y=jobs/rendered/$J.yaml
     sed "s/{name: SRC_GIT_COMMIT, value: FILL_AT_SUBMIT}/{name: SRC_GIT_COMMIT, value: $C}/" jobs/$J.yaml > $Y
     for f in $K; do
       h=$(git show $C:world-model-stacking-encoders/src/$f | shasum -a 256 | cut -c1-64)
       sed "s/FILL_SHA_$f /$h /" $Y > $Y.tmp && mv $Y.tmp $Y
     done
     grep -c "value: $C}" $Y           # must print 1
     grep -c FILL_ $Y                  # must print 0
     grep -cE '^ +[0-9a-f]{64}  ' $Y   # must print 12
   done
   ```
   Commit the rendered files as submitted.
5. **Dry run.** Check our running GPU count first (at most 8, `STANDARDS.md`) and that Secret `stacking-encoders-hf` exists. `kubectl create --dry-run=server -f jobs/rendered/job-first-frame-check-dryrun.yaml`, then `kubectl create -f ...`. Note the creation time (queue wait and run time go into `compute/nautilus-job-times.csv`). One check about 20 min after it starts running, no loop. Save the log from the pod at once (container logs vanish within about a day):
   ```
   mkdir -p results/first-frame-check
   P=$(kubectl get pods -n aria-world -l job-name=stacking-encoders-pairings-first-frame-check-dryrun -o name | head -1)
   kubectl logs $P -n aria-world > results/first-frame-check/dryrun-${P#pod/}.log
   kubectl get job stacking-encoders-pairings-first-frame-check-dryrun -n aria-world -o yaml > results/first-frame-check/dryrun-job-status.yaml
   ```
   Decode the PNGs from the log (the trap prints each between `=== PNG <name> sha256 ...` and `=== END PNG`):
   ```
   awk '/^=== PNG /{n=$3; getline; print > ("results/first-frame-check/dryrun-" n ".b64")}' results/first-frame-check/dryrun-*.log
   for f in results/first-frame-check/dryrun-*.png.b64; do base64 -d -i "$f" -o "${f%.b64}"; rm "$f"; done
   shasum -a 256 results/first-frame-check/dryrun-*.png   # compare with the sha256 on each "=== PNG" line
   ```
   **What to look at** in each PNG (top row: the live frame, init state 0, agentview | wrist, 224 px as `process_obs` gives it; bottom row: the training frame 0 of the task's lowest-id episode, agentview | wrist, the uint8 224 px frame the encoders saw): orientation (both rows upright and the same way up: the rollout rotates the sim frames 180 degrees, the dataset frames are upright, `src/fastwam_data.py` lines 5-6); camera (agentview left, wrist right, in both rows; the wrist view shows the gripper); colours (the same objects in the same hue in both rows: no red/blue swap); framing (the table and robot at about the same place and size). The scenes themselves may differ a little (the live frame is after 30 settling steps from an evaluation init state, the training frame is demonstration frame 0; plan Limits). Then read `result.json` (between `=== FILE result.json` and `=== END result.json`): it must hold no statistic (`statistics_keys` only), and `pipeline_checks` must show `frame0_decode` all true, `live_shapes` `[48, 14, 28]`, `[1024, 14, 28]`, `[1024, 14, 28]`, `[1152, 14, 28]` (VAE, DINOv3, C-RADIOv4, SigLIP 2; widths recalled from the diagnosis note section 1), `train_episodes` per suite (434, 457, 433, 388 expected, experiment LOG line 96), `pins.encoder_loaded_commits` equal to `pins.encoder_revisions`, `pins.gpu` an NVIDIA L4. **Anything off in a PNG or a check: stop and ask the user; do not run the full Job.**
6. **Full-run time from the dry run.** From the dry run's `timing`: live time about `live_frame_wall_s.median` x 200 / 3, training time about `train_s` / `train_episodes` x 1,712, plus `featurizer_load_s`, the sha/restore phase from the PHASE lines, and a few minutes of statistics. If that is above the full manifest's `CHECK_CAP_S` (5,400 s), re-render a larger cap and pod deadline (timing only, no number of the check is involved) and say so in the LOG.
7. **Full run.** Same as step 5 with `jobs/rendered/job-first-frame-check.yaml` (Job `stacking-encoders-pairings-first-frame-check`). Save the log, then cut the result out of it and check its hash (the trap prints the file, then one newline, then the END line):
   ```
   L=results/first-frame-check/full-<pod>.log
   awk '/^=== END result.json/{p=0} p{print} /^=== FILE result.json/{p=1}' $L | perl -pe 'chomp if eof' > results/first-frame-check/result.json
   shasum -a 256 results/first-frame-check/result.json   # must equal the sha256 on the "=== FILE result.json" line
   ```
8. **Spearman on the Mac** (descriptive item 2; the committed clean records are in the repo, not on the volume):
   `/opt/homebrew/bin/python3.11 -I world-model-stacking-encoders/src/first_frame_check.py correlate --result <experiment>/results/first-frame-check/result.json --out <experiment>/results/first-frame-check/spearman.json`. It reads the `none` records under `../2026-11-15-pretrained-start-baseline/results/clean-eval/records/` and each encoder's under `results/clean/<e>-s0/records/` (verified present in git), and writes the result's sha256 into its output.
9. **Read it against the plan's reading rule** (fixed before the run; nothing here changes it). In `result.json` `statistics`:
   - `reading.<encoder>`: `R` (point estimate), `interval` (2.5th to 97.5th percentile of 1,000 task-level draws), `zone` from the point estimate (`near` R <= 1.2, `clearly larger` R >= 1.5, else `unclear`), `wording` with "borderline: the bootstrap interval crosses 1.2 / 1.5" when it does. A borderline interval changes neither the zone nor the next step. DINOv3 also has `R_without_757` and `zone_without_757`; if they differ from the all-channel zone, `reading` is `unclear` and the wording says the reading depends on that channel.
   - `outcome`: the plan's case 1, 2 or 3, checked in that order on the `reading` values, with the plan's next step in words. The next step itself (a settling run proposal, or the open choices) is the planning lead's and the user's, not this script's.
   - `left_out_tasks`: tasks with fewer than 2 training episodes or a zero reference distance in any group (none expected); name them in the report.
   - Descriptive, no bar and no decision: `r_quantiles` (median and 90th percentile of r per group, per suite and all), `spearman.json` (step 8), `vae_single_vs_clip` (item 15: **if the single-frame and cached clip-encoded VAE latents differ, report it to the user at once**: it would affect `none` too), `dinov3_757_live_z`, and `ranges` (the amendment: `summary` per group and set, `mean_square_ratio_to_vae`, `wording` per encoder from set T, `flags` per group and test with counts and shares, `per_channel` the full table). The flag list, counts and ratios go to the user as they are.
   - The printed `=== SUMMARY` block in the log repeats these in short form.
   The result goes into the stage 2 `LOG.md` and a short README in this folder with every number, `$C`, the image, the pins in `result.json` `pins`, and the Jobs' times (plan Hand-off); nothing goes into README key findings without the user's sign-off.
10. **Delete** both Jobs after their logs are saved (`kubectl delete job <name> -n aria-world`, graceful, never `--force`) and the ConfigMap (`kubectl delete configmap stacking-encoders-pairings-firstframe-src -n aria-world`). The output folders on the PVC stay.

## Exit codes (`=== exit <rc>`; podFailurePolicy fails the Job at once on 2, 3, 10, 13, 15, 19; backoffLimit 1)

| rc | Meaning | What to do |
|---|---|---|
| 0 | done | steps 5 to 9 |
| 3 | a pipeline check failed (`CHECK FAILED` line): a cache file missing or of the wrong length, a checkpoint config that is not the expected single-encoder seed 0 run, a loaded encoder commit that is not the pin, not an L4, the frame-0 decode check, the clean task list, an existing output folder | read the line; stop and ask the user; change nothing on the PVC |
| 2 | the script failed (traceback, or bad arguments) | read the traceback, fix by PR, ask before resubmitting |
| 10 | not rendered, MODE wrong, or `/src/setup_sim.sh` does not pin LIBERO-Plus 4976dc3 | re-render (step 4) |
| 13 | a model.pt sha256 differs from the pin | stop and ask the user |
| 15 | a model.pt is missing | stop and ask |
| 19 | a `/src` key differs from the rendered hash | re-read the ConfigMap (step 3), re-render (step 4) |
| 16 | the sha step (1,800 s) or the check (`CHECK_CAP_S`) timed out; retried once in a new pod (a new dated folder, nothing overwritten) | read the PHASE and `[ffc ...]` lines; ask the user before raising a cap |
| 1 | Python died outside main, the env restore failed, or `nvidia-smi` failed (a faulty node); retried once | read the log |
| DeadlineExceeded | pod over its deadline, or Job over 22 h from creation (queue included) | ask the user |

## Plan item to implementation

All functions are in `src/first_frame_check.py` unless named otherwise; the tests are in `src/test_first_frame_check.py`.

| Plan item | What it says (short) | Implemented by | Tested on the Mac by |
|---|---|---|---|
| Design, groups | VAE raw (48 ch); DINOv3, C-RADIOv4, SigLIP 2 z-scored with each encoder's own single-encoder seed 0 `feature_stats` | `GROUPS`; `load_stats` (reads `model.pt` config as `rollout.Policy.load` does; checks config, seed 0, sources); `run`: `zscore` (float32 on the device, as `rollout.Policy.act` and `proxy_wam.to_device`) on both sides | (cluster only) |
| Design 1 | 40 clean tasks x init states 0-4, first policy frame exactly as a rollout (LIBERO-Plus 4976dc3, `libero_env`, reset, `set_init_state`, 30 dummy steps, `process_obs`), `rollout.Featurizer` at batch 1 on an L4; no policy step | `clean_jobs` (`rollout.make_jobs`, eval seed 41, trial = init state); `live_worker_init` + `live_frame` (seeds as `rollout.run_episode`, then `rollout.first_frame`) in spawned processes; `run`: one `rollout.Featurizer` with the three encoders, called with one frame at a time; L4 and encoder revision checks; LIBERO-Plus commit checked by the manifest and `run` | (cluster only; the dry run) |
| Design 2 | frame 0 of every training episode of the task (task from `meta/<suite>.npz`); encoders: cached features z-scored; VAE: frame 0 encoded alone through the training-side decode and `to_224`, not the cached clip latent | `run` (suite loop: `fastwam_data.read_suite` + meta npz, ids/tasks/lengths cross-checked); `train_frame0` (`frame0_episode` + `fastwam_data.decode_episode`, `to_224`, `video_clips(num_frames=1)`, cached `[0]` with length checks); the Featurizer's VAE in bf16, `.half()` rounding; `frame0_decode_check` (frame-0 shortcut equals the full decode, bit for bit) | (cluster only; the dry run) |
| Design 3 | d = L2 of flattened (C, 14, 28) / sqrt(C x 14 x 28) | `norm_dist`, `flatten` | `test_distance_normalization` |
| Design 4 | r = d(nearest training frame 0) / ref; ref = median of each training frame's distance to its nearest other one; ties no rule; tasks with < 2 episodes or ref = 0 in any group left out and named | `nearest_dists`, `task_ratios`; `assemble` (`left_out_tasks`); `run` (a left-out task is also out of the range sets) | `test_nearest_matches_brute_force_and_duplicates_are_zero`, `test_task_ratios_leave_self_out_reference`, `test_task_left_out_when_too_few_episodes_or_ref_zero`, `test_assemble_m_R_and_left_out_task_named` |
| Design 5 | m(g) = median of r over all live frames pooled, each once | `pooled_median` (in `assemble`) | `test_pooled_median_counts_each_frame_once` |
| Design 6 | R(e) = m(e) / m(VAE); 95% percentile interval, 1,000 draws, seed 0, tasks resampled with replacement, all 5 frames per drawn task, ref not recomputed | `ratio_R`, `bootstrap` (in `assemble`) | `test_bootstrap_resamples_tasks_with_ref_fixed` |
| Reading rule, zones | <= 1.2 near (below 1 too), >= 1.5 clearly larger, between unclear; from the point estimate | `zone`, `read_encoder` | `test_zones_and_borderline_wording` |
| Reading rule, borderline | interval crossing 1.2 or 1.5 reported as borderline in those words; zone and next step unchanged | `crossed_edges`, `borderline_wording`, `read_encoder` | `test_zones_and_borderline_wording` |
| Reading rule, DINOv3 channel 757 | R also without channel 757; zone uses all channels; zone differs -> read unclear, depends on that channel | `drop_channel` + `task_ratios` on the `NO757` group in `run`; `assemble`; `read_encoder(R_no757=...)` | `test_channel_757_rule` |
| Outcome cases 1 to 3 | checked in order: any clearly larger; all near; anything else | `outcome_case` (case and the plan's words; the actions are the lead's and the user's) | `test_outcome_cases_in_order` |
| Also reported 1 | median and 90th percentile of r per group, per suite and over all 200 | `r_summaries`, `quantiles` (in `assemble`; also for the without-757 variant) | `test_assemble_m_R_and_left_out_task_named` |
| Also reported 2 | Spearman over 40 tasks: per-task ratio (median r of the encoder over 5 frames / the VAE's) vs per-task clean drop (none minus encoder, committed records) | `task_level_ratios` (pod, in `assemble`: `per_task_ratio`); `correlate`, `task_success`, `spearman`, `rankdata` (Mac, step 8) | `test_task_level_ratio_and_757_live_and_latent_diff`, `test_spearman_with_ties_and_constant`, `test_correlate_on_fake_records` |
| Also reported 3 | single-frame VAE latents vs the cached clip-encoded ones: largest absolute difference and relative L2 | `latent_diff`, `latent_diff_summary`; per episode in `run` (`vae_single_vs_clip`) | `test_task_level_ratio_and_757_live_and_latent_diff` |
| Also reported 4 | live z-scores of DINOv3 channel 757: median and largest absolute value | `channel_757_live` (`dinov3_757_live_z`) | `test_task_level_ratio_and_757_live_and_latent_diff` |
| Amendment, values | VAE after Wan's scale, encoders after the z-score, each rounded through float16 as the cache and the rollout do | live: `rollout.Featurizer` output (`.half().float()`), then `zscore`; training: cached float16 then `zscore`, VAE single-frame `.half().float()` | (cluster only) |
| Amendment, two sets | T: training frame 0s (encoders cached, VAE single-frame); L: live frames; each frame once; left-out tasks out of both | `run` (`stats["T"]`, `stats["L"]`, only included tasks) | (cluster only) |
| Amendment, per channel | pool over frames and all 14 x 28 positions; arithmetic mean, population sd, largest absolute value | `ChannelStats` (`add`, `mean`, `sd`, `maxabs`, `table`) | `test_channel_stats_against_numpy` |
| Amendment, per group | overall mean square | `ChannelStats.mean_square` | `test_channel_stats_against_numpy` |
| Amendment, reported per group and set | median/smallest/largest per-channel sd; median and largest absolute per-channel mean; largest absolute value; mean square; encoder/VAE mean-square ratio on the same set; full per-channel table in the result | `group_summary`, `ranges_report` (`summary`, `mean_square_ratio_to_vae`, `per_channel`) | `test_ranges_report_end_to_end` |
| Amendment, flag (a) | on T: sd < 0.25 or > 4, or abs mean > 2 | `channel_flags` (`a`), `flag_report` | `test_flag_tests_and_constant_exception` |
| Amendment, flag (b) | sd_L < sd_T / 4 or > 4 sd_T; sd_T < 1e-6: not applied, flagged constant on T | `channel_flags` (`b`, `constant_T`) | `test_flag_tests_and_constant_exception` |
| Amendment, flag (c) | abs(mean_L - mean_T) > 2 sd_T; same exception | `channel_flags` (`c`, `constant_T`) | `test_flag_tests_and_constant_exception` |
| Amendment, flag report | flagged channels by group and index, count and share per test | `flag_report` (in `ranges_report`: `flags`) | `test_flag_tests_and_constant_exception` |
| Amendment, wording | "in very different ranges" if an encoder's mean square on T is > 4 or < 1/4 of the VAE's; else "the overall ranges are comparable" with the ratio | `range_wording` (in `ranges_report`: `wording`) | `test_mean_square_ratio_wording`, `test_ranges_report_end_to_end` |
| Amendment, reading | descriptive only, no zone or next step | nothing in `outcome_case` reads `ranges` | by construction |
| Hand-off, dry run | a few frames; live frame and training frame 0 of the same task side by side (orientation, camera, colours) | `run --dry-run` (one task per suite), `save_side_by_side`; the manifest prints the PNGs; statistics withheld | (the dry run, step 5) |
| Hand-off, pins | source commit, image, environment commit, dataset path, three checkpoints with sha256 | manifests (env and header, sha256 checked in the pod, exit 13); `run` writes them to `pins` | (cluster only) |
| Predictions, Limits, Test-set use, Compute, Sign-off | no computation | not code; the Compute open question (L4, not CPU) is followed: the manifests pin an L4 and `run` refuses another GPU | n/a |

## What could not be confirmed from the code (for the lead)

- **Training frame 0 pixels** come from the dataset video through `fastwam_data.decode_episode` (verified, code), cut after frame 0 by `frame0_episode` to avoid decoding every frame of 1,712 episodes. That the cut decode gives the same frame-0 pixels and the same `to_224` output as a full decode is expected from the code (same seek, same decoder; speculative until run) and is **checked in the pod** for the first used episode of each suite (`frame0_decode_check`, exit 3 on any difference).
- **Whether the caches hold frame 0 for every training episode**: not checkable on the Mac. `check_stage2_cache.py` checked every file's header and length earlier (experiment LOG, around lines 96 and 141; recalled from the LOG, not re-read in full), and the script checks every file it opens (exists, first dimension = episode length, 14 x 28 grid; exit 3 otherwise).
- **1,712 training episodes over 40 tasks** and the per-suite counts are from the experiment LOG (line 96, verified as quoted there); the script records the counts it finds.
- **Reading `feature_stats` with `torch.load(..., mmap=True)`** assumes the checkpoints are in torch's zipfile format (torch's default since 1.6; recalled, not checked). If not, the load raises and the script exits 2 (no fallback, no stand-in).
- **Interpretations I made where the plan does not spell out the arithmetic** (please confirm or tell me to change): the DINOv3-without-757 variant has its own reference distance, computed with the channel dropped on both sides; the left-out rule uses the four groups only; "the interval crosses 1.2 or 1.5" means the interval's ends fall in different zones (1.2: lo <= 1.2 < hi; 1.5: lo < 1.5 <= hi); percentiles are numpy's linear interpolation; item 15's relative L2 is ||single - cached|| / ||cached||; the 757 live z-scores report the signed median, the median of absolute values and the largest absolute value; each bootstrap draw is shared by all groups; the Spearman is computed on the Mac by a separate `correlate` command, because the committed clean records are not on the cluster volume (its output is a second JSON with the result's sha256, not part of `result.json`).
- **The live frames are built in spawned simulator processes** (as rollouts run them), with `rollout.first_frame` plus `torch.manual_seed` (`first_frame` itself seeds only `random` and `numpy`; `run_episode` seeds all three before building the env). That the simulator does not use torch's generator is recalled, not checked; seeding it anyway keeps the call identical to a rollout's.
- **One Featurizer with all three encoders** instead of one per configuration: each encoder runs its own `encode` on the same uint8 frames (verified, `rollout.Featurizer.__call__`), so the outputs should not depend on which other encoders are loaded (speculative for GPU kernel choice).
- **The plan's open question to the experiment session** (a matched version: the simulator set to a demonstration's frame-0 state): the LeRobot dataset this check reads exposes `action` and `observation.state` (8 values: end-effector pose and gripper) per frame (verified, `fastwam_data.read_suite`), not the full MuJoCo state, so it cannot be done from this dataset as read here; whether the original LIBERO HDF5 demos (which the perturbed-frames experiment rendered from saved states, `src/pf_rerender.py`) can be matched to these no-noops episodes is not known. Not in this design; for the planning lead.
- **GPU utilization**: the whole-pod mean will be low (speculative); see the statement at the top.
