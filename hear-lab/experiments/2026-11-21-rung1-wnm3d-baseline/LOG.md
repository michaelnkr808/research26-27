# Experiment log: rung 1, released WNM-3D at inference time in GN-Bench

Step-by-step record of running this experiment (Jobs, errors, reruns, exact numbers). Append-only, newest at the bottom. The direction's `../../LOG.md` keeps short summaries that point here. Entry template: `_standards/04-documentation.md`.

## 2026-11-21 [agent, worker] (folder written, episode order frozen, nothing rented)
**Did:** wrote PLAN, config, setup, run, check and summary scripts and their unit tests from the approved rung 1 plan (direction LOG, 2026-11-21 experiment entry). Generated the frozen order on the Mac from the route-a check 3 CSV (sha256 270297ec...510e7): 5000 ids, seed 20261121, sha256 25366209f38a242dbcec94bfed26092a9937dbf5adabbcd149f4558caed72728; first three ids 0060_839918_78, 0401_840132_178, 0098_839977_50. The 39 possibly mirrored scenes recomputed from check 3 `scenes.jsonl` by the check 3 rule hold 233 test_unseen episodes (50 visible), as check 3 reported.
**Checks:** unit tests pass on the Mac; `check_settings.py` passes, also against the local pinned copies of WNM-3D 7fe2d5f and GN0 c927781.
**Open for the lead:** stall recovery on or off; K (timing batch); N (main run). No run can start while stall recovery is pending.

## 2026-11-21 [agent, worker] (decisions recorded, nothing rented)
The user's answers relayed by the lead ("yes go for it"): stall recovery on (as released), K = 30. Recorded in `config.yaml` and as a PLAN amendment; `check_settings.py` asserts both. N still pending. No run had started.

## 2026-11-22 [agent, lead] (timing rental: setup, three fixes, 30 timing episodes; instance destroyed)
**Rental (approved by the user in the experiment window).** The user's "okay" to the quoted Japan H100 (offer 59816255, about 2.24 USD/h with disk); that offer was gone, the user's "yes" to the twin listing 59816165 (about 2.44 USD/h); at create time that id was gone too and the same machine (155837, Japan, reliability 0.9928, 28 cores, 252 GB RAM listed) was back as offer 59816179 at a lower price, and was rented: instance 61899040, 1x H100 SXM 80 GB, driver 595.91.07, on-demand, 200 GB disk, image `nvidia/cuda:12.9.1-devel-ubuntu22.04`, 2.3704 USD/h (dph_total incl. disk). 224 threads and 2015 GB RAM visible in the container. Created 2026-11-22T02:55:26Z, destroyed 2026-11-22T04:44:50Z: 1.823 h, 4.32 USD by the hourly rate. Account credit 189.976 USD before create, 185.702 USD just before destroy, 185.678 USD 20 s after (4.298 USD; late charges may still settle). Estimate given beforehand: about 1 h setup plus 1 to 2.5 h of episodes, 6 to 9 USD at 2.44 USD/h, hard stop 4 h (not from a timed batch; none existed). Actual 1.82 h.

**Setup** (`setup.sh`, all stages; log `env/setup_console.log`). Environments 02:56 to 03:05 Z; checkpoint (6 shards, sha256 OK) 03:05 to 03:08 Z. Three stops, each fixed by the lead as a commit on branch `hear-lab-rung1-tokenizer-path` (no model or evaluation setting changed):
1. Tokenizer (03:07 Z): the released checkpoint's `experiment_cfg/conf.yaml` names `tokenizer_path: ./checkpoints/umt5-xxl`, not the Hub id the script expected (the worker had flagged this as unknown). Fix d70ebc0: download google/umt5-xxl at the pinned 66cb9e7 (same five files) into `checkpoints/umt5-xxl`, as WNM-3D `docs/DOWNLOADS.md` lines 106 to 107 and route-a check 2 did; sha256 of the five files in `env/tokenizer.sha256`.
2. InteriorGS download (03:11 Z): Hugging Face rate limit (HTTP 429, 1000 API requests per 5 min) when about 333 of the 839 scene folders had any finished file (the "344" in the console is a file count within one 1,000-file batch; reviewer recount, PR #83). Fix 599f4ca: retry up to 12 attempts 330 s apart. Attempts: exit 123, exit 123, exit 0, exit 0; all 839 scenes present at 03:46 Z. The retry's 429 test matched the digits "429" in file hashes, so attempt 4 was extra (about 6 min by the attempt log times; attempt 2 had really failed, so attempt 3 was needed); tightened in eefdecc to "429 Client Error|Too Many Requests". Order check against the pinned parquet passed; token check passed (no token file left; repeated before destroy).
3. Episode hook (04:16 Z, during timing run 1): `rung1_filter.append_line` crashed the GN0 client with "Object of type float32 is not JSON serializable" after 18 of 30 episodes (an episode ended by the stall early stop at step 250). GN0's own episode file was written before the crash; only the hook's timing line for that episode is missing. Fix eefdecc: convert numpy scalars exactly (`.item()`), refuse anything else; unit test added (66 passed, 1 skipped).

**Timing run 1** (`runs/timing`, commit 599f4ca, launch 20261122T034655Z). Server ready after 170 s. 18 of 30 finished, then the crash above. Resume with the fixed code refused by `run_eval.sh` (repo commit differs from the first launch, by design), so the batch was rerun in full under a new run name rather than bypassing the check.

**Timing run 2** (`runs/timing-r2`, commit eefdecc, launch 20261122T041840Z): server ready after 150 s, client exit 0, 30 of 30 finished, no foreign ids. Settings as in `run_meta.jsonl` (stall recovery on, stop at 0.15 m, the released server flags). Results (`verified`, `summarize.py --run-kind timing`; not the pass bar, which is on the main run): SR 60.0 (18/30, Wilson 95% 42.3 to 75.4), SPL 59.2, OS 63.3, NE 3.43, TL 6.14. Visible starts 7/9 succeed, not visible 11/21. Stop reasons: chunk_low_translation 19, distance_to_goal_unchanged 9, action_near_zero 2. Timing: mean 45.1 s per episode (44.1 excluding the first of the launch), median 40.3 s; 1354 s for 30; 0.0297 USD per episode at 2.3704 USD/h. GPU utilisation (nvidia-smi every 5 s while the model was loaded) averaged 40%; one client alternates rendering and inference.

**Repeatability** (`verified`): the 18 episodes that finished in both runs have identical success, SPL, oracle success, distance to goal, path length and stop reason.

**Unexplained** (`speculative` at most): run 1 was about twice as slow on the same episodes (median 76.8 s against 40.9 s; GPU average 21% against 40%). The server's per-call timings show model calls about 1.4x slower in run 1, and the client side outside model calls slower by more; the slowdown varied widely by episode (some were faster) and lasted the whole run, which fits a busy host better than a warm-up (reviewer reading, PR #83). Run 2's 12 episodes not seen before had median 34.3 s, but that does not rule out a cache effect: the scenes were downloaded minutes before run 1 on a machine with about 2 TB of RAM, so the page cache was most likely warm in both runs. Cause not found. The main-run cost is therefore quoted as a range.

**Main-run projection** (`speculative`, from these two runs): at run 2's speed 1,000 episodes take about 12.5 h, about 30 USD plus about 1 h of setup (about 2.4 USD); at run 1's speed about twice that. All 5,000 would be about 151 USD with setup at run 2's speed (inside the remaining credit of about 186 USD, with little left over) and about 271 USD at run 1's speed (above it). N is the user's call.

**Metric note** (`verified`, reviewer, PR #83): one timing-r2 episode has an infinite distance to goal and no SPL; `summarize.py`, like GN0's own results script, counts both as 0, which pulls the mean NE down slightly. To be stated next to the NE comparison with the paper on the main run.

**Outputs:** `~/Desktop/hear-lab-rung1-results/` (env/, runs/timing, runs/timing-r2, copy manifest: 123 files, sha256 checked on the Mac, 0 failed; `timing-instance.json`, not in git). No instances remain.

## 2026-11-22 [agent, lead] (main run started: N = 1000; in progress at session close)
**Rental (approved by the user in the experiment window: "yes merge and rent it sure", after the quote of the Japan H100 at about 2.31 USD/h, hard stop 65 USD).** Instance 62043528, offer 59816179, machine 155837 (the timing machine), 1x H100 SXM 80 GB, Japan, reliability 0.9931, on-demand, 200 GB disk, image `nvidia/cuda:12.9.1-devel-ubuntu22.04`, 2.3704 USD/h (dph_total incl. disk). Created 2026-11-22T21:36:59Z; listing end 2026-11-23T11:00Z. Credit 185.545 USD before create.
**Setup:** 21:37 to 22:17 Z, all stages passed (InteriorGS: 4 attempts on rate limits); token check passed.
**Run:** `run_eval.sh --run-name main --n 1000 --repo-commit 4d9ceed` (main after PR #84). Server ready 22:20:38Z; filter selected 1000 of 5000. First 6 min: 11 episodes; GPU 45 to 100 % in samples.
**Copy-off:** a Mac-side copier (background task of the lead session) tars `env/` and `runs/main` to `~/Desktop/hear-lab-rung1-results/main-sync/` every 30 min; at the end it verifies a sha256 manifest and destroys the instance only if all 1000 finished with client exit 0 and every file verified. Handoff note: `~/Desktop/hear-lab-rung1-results/HANDOFF.md`.
**Next:** summarize (`--run-kind main`), the verdict against the pass bar, cost, this LOG and the README, PR with review.

## 2026-11-23 [agent, lead] (main run finished: 1000 of 1000, verdict reproduced; instance destroyed)
**Run:** `runs/main`, launch 20261122T221747Z, repo commit 4d9ceed. Server ready 2026-11-22T22:20:38Z; end record 2026-11-23T14:19:20Z: n_finished 1000, n_requested 1000, n_extra 0, client_exit 0. Settings as in `run_meta.jsonl` (stall recovery on, the paper's 0.15 m block stop rule, the released server flags).

**Compute (Vast.ai):** instance 62043528, machine 155837 (Japan), 1x H100 SXM 80 GB, on-demand, 200 GB disk, 2.3704 USD/h (dph_total incl. disk). Created 2026-11-22T21:36:59Z, destroyed 2026-11-23T14:33:12Z by the guarded copier: 16.94 h, 40.15 USD by the hourly rate. Credit 185.545 USD before create, 145.497 USD just before destroy, 145.470 USD 30 s after (40.07 USD; late charges may still settle). Hard stop approved: 65 USD. Setup about 0.7 h, episodes 15.98 h (sum of per-episode wall time 57,518 s). No instances remain (`vastai show instances` empty, 14:33Z).

**Listing end:** the host's listing end was 2026-11-23T11:00Z at create time, before the projected finish; the host extended it to 2026-11-24T00:00Z during the night (seen 13:31Z), so the run finished on one instance with no resume.

**Copy-off:** the first copier (30 rounds of 30 min) ran out at 13:01Z with 920 episodes copied and the run still going (943 done at 13:31Z). Restarted by the lead as `main_copier2.sh` (same logic, 16 rounds of 15 min). When MAIN_EXIT appeared it built the sha256 manifest on the instance, copied, verified on the Mac (1056 OK, 0 bad; re-checked by the lead: 0 bad), ran the token check (no token files left), and destroyed the instance. Across both copiers 15 of the 35 rounds left the copied count unchanged (counts in order: 11 37 37 37 37 37 190 190 250 283 308 308 308 387 387 387 497 522 547 572 572 638 669 704 745 791 791 854 854 920 920 920 920 992 1000; reviewer, PR #86); most likely tar exits non-zero when a file changes during the read and the round's unpack is skipped (`speculative`). The final copy ran after the run ended, so the verified copy is not affected. Outputs: `~/Desktop/hear-lab-rung1-results/main-sync/` (env/, runs/main, `copy_manifest_main.sha256`, `runs/main/summary.json`; not in git).

**GPU:** spot checks with nvidia-smi at 23:00Z, 04:01Z and 13:31Z read 47, 46 and 45 % (one client alternates rendering and inference, as in the timing run).

**Results** (`verified`, `summarize.py --run-kind main`, first 1000 of the frozen order; the visible / not-visible labels come from the route-a check 3 file, as fixed in `PLAN.md`, and were not re-derived for this run):

| | n | SR (95% Wilson) | SPL | OS | NE | TL |
|---|---|---|---|---|---|---|
| Paper, Table 1 Unseen (5,000 episodes) | 5000 | 53.5 | 46.6 | 62.2 | 4.1 | 10.2 |
| All | 1000 | 55.0 (51.9 to 58.1), 550 successes | 51.2 | 61.5 | 4.04 | 9.07 |
| Target visible at start | 220 | 65.0 (58.5 to 71.0) | 61.4 | 70.0 | 3.31 | 6.82 |
| Target not visible at start | 780 | 52.2 (48.7 to 55.7) | 48.3 | 59.1 | 4.25 | 9.71 |
| All, excluding the 39 possibly mirrored scenes | 958 | 55.1 (52.0 to 58.2) | 51.4 | 61.4 | 4.05 | 8.99 |

**Verdict: reproduced**, under the paper's 0.15 m block stop rule (the one deviation from the released command; see `PLAN.md`) (the pass bar in `PLAN.md`: the paper's 53.5 lies inside our 95% Wilson interval, 51.9 to 58.1). SPL, OS, NE and TL have no pass bar; SPL is higher than the paper's and TL shorter (no explanation tested; the 0.15 m stop rule ends some episodes earlier than the released command would, `speculative`).

Stop reasons: chunk_low_translation 641, distance_to_goal_unchanged 305, action_near_zero 38, environment_done 16.

**Metric note** (`verified`): one episode (0968_841900_242) has an infinite distance to goal and an undefined SPL (path length 0); `summarize.py`, like GN0's own results script, counts both as 0, which lowers the mean NE by about 0.005 (reviewer, PR #86). The paper's NE 4.1 is compared with this.

**Repeatability** (`verified`): the 30 timing-r2 episodes are the first 30 of the main run; all 30 have identical success, SPL, oracle success, distance to goal, path length and stop reason in both runs.

**Timing:** mean 57.5 s per episode, median 35.8 s; 0.038 USD per episode at 2.3704 USD/h. Slower on average than timing-r2 (45.1 s mean) and faster than timing run 1; cause not studied.

**Next:** PR with review. For planning: rung 1 passed, so rung 2 (render and check the audio on a small set) can start once planning hands it over; the not-visible split above is the no-audio reference for the research question.
