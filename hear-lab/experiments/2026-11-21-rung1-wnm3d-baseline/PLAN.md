# Plan: rung 1, the released WNM-3D result at inference time in GN-Bench

Written and committed before running. Once the experiment starts, don't edit this design: add changes below as dated amendments.

**Question:** does the released WNM-3D stage 3 checkpoint, run as released in GN-Bench on our rented hardware, reach the success rate the WNM-3D paper reports on unseen scenes? Every later rung adds audio to this model and reads its effect on GN-Bench test_unseen, so we first need to know that our set-up reproduces the paper's number without audio. No training and no audio in this rung.

**Target (fixed before any run):** WNM-3D paper (arXiv 2608.07267 v3) Table 1, Unseen, WNM-3D row (the same row in Table 4): SR 53.5, SPL 46.6, OS 62.2, NE 4.1, TL 10.2. The paper evaluates each of the 5,000 Unseen episodes once with inference seed 1140 and reports point estimates only (Appendix B.1).

## Design

**Split:** GN-Bench test_unseen (GN-Matrix `InteriorGS/InteriorGS_test_unseen.parquet`, 5,000 episodes) only. This is where later rungs are read and where route-a check 3's visible flag exists.

**Episodes:** a random sample, drawn and frozen now. `src/make_episode_order.py` builds a full random permutation of all 5,000 test_unseen episode ids with `numpy.random.default_rng(20261121).permutation(5000)`, applied to the ids in parquet row order. Any episode count N is the first N lines of that order, and the timing batch is the first K. The list is `results/episode_order.txt` (gitignored: it holds InteriorGS scene names and the InteriorGS terms are not read yet); it is regenerated from the seed whenever needed and checked by sha256.
- Seed 20261121, 5,000 ids, sha256 `25366209f38a242dbcec94bfed26092a9937dbf5adabbcd149f4558caed72728`.
- Source of the ids: the route-a check 3 per-episode CSV (`check3_InteriorGS_test_unseen.csv`, 2026-11-20 rerun, sha256 `270297ecd2f9a3f87a7a2f4dc974c0ccea4e5219ad6709a6e2888333015510e7`). It has one row per row of the GN-Matrix test_unseen parquet at revision 50a68253, in parquet order (check 3 read that same file, sha256 `de4fd64a...`), so its ids are GN-Matrix's ids. It was used because the parquet itself is not on the Mac. The id format is GN0's own (`{scene}_{trajectory_id}`, `gn0/evaluation/common.py` line 27). `setup.sh` (stage `verify`) checks on the machine that the frozen order holds exactly the ids of the downloaded parquet.
- **N (main run): open.** The lead picks it after the timing rental, from the measured time and cost per episode. **K (timing batch): open**, picked before the timing rental.
- Both are committed to `config.yaml` (`episodes.n_main`, `episodes.timing_k`) before the run they govern. `run_eval.sh` refuses any `--n` that is not one of the committed values (so nothing runs while both are pending), and `summarize.py` reads N from `config.yaml` for the run kind, not from the command line.

**What runs:** the released single-replica policy server (WNM-3D `scripts/inference/wnm_3d_server.sh`) and the released GN0 remote evaluation client (`scripts/evaluation/eval_remote.sh` with `configs/gn_bench/interiorgs/test_unseen.yaml`), both on one GPU, as in the WNM-3D README single-GPU commands. `src/run_eval.sh` starts both, restricted to the first N episodes of the frozen order. GN0 writes one result file per episode as it finishes and skips finished episodes on a restart, so a crashed run resumes with the same command. No WNM-3D or GN0 file is changed.

**Settings** (every one is in `config.yaml` and asserted by `src/check_settings.py`, which also checks the pinned source lines on the machine before a run):

| Setting | Value | Source |
|---|---|---|
| Checkpoint | `TeleEmbodied/WNM-3D` `wnm_3d_stage3_release`, revision 5b8c08dd52028ef4a29e7b87966f27582032fb4f, six shards checked by sha256 | WNM-3D `docs/DOWNLOADS.md` line 48; shard sha256 from route-a check 2 |
| Code | WNM-3D 7fe2d5f, GN0 c927781, GN-Bench-Tools 0ce8367 | route-a pins |
| Server replicas | 1 replica on GPU 0, port 8000 | README lines 131 to 136 |
| Inference steps | 16, with the fixed DiT step mask 1 1 1 0 0 0 1 0 0 0 1 0 0 1 1 1 (8 DiT calls) | `wnm_3d_server.sh` lines 34 and 35 |
| DiT cache | on | `wnm_3d_server.sh` line 33 |
| CFG | on, scale 5.0 | `wnm_3d_server.sh` lines 36 and 37 |
| torch.compile | off | `wnm_3d_server.sh` line 38 |
| Inference seed | 1140, a fresh generator per sampling call | action head `wan_flow_matching_action_tf.py` lines 289 and 1499; paper Appendix B.1 |
| Actions per query | at most 8 | `eval_remote.sh` line 18 |
| Stall patience | 16 | `eval_remote.sh` line 37, `test_unseen.yaml` line 3 |
| Episode step limit | MAX_EPISODE_STEPS 500 (EARLY_STOP_STEPS 500) | `test_unseen.yaml` lines 4 and 8 |
| Success radius | 3.0 m (success, SPL, oracle success) | `test_unseen.yaml` lines 28, 41, 43, 45 |
| Block-translation STOP | **0.15 m (deviation, below)** | paper Appendix B.1; GN0 `gn0/agents/remote.py` line 128 |
| Stall recovery | **open (below)** | README line 155 passes it |
| Client GPU slots | 1 (one worker) | README lines 163 and 164 |

**Deviation from the released command, fixed now:** the stop rule follows the paper. Appendix B.1 says the evaluator emits STOP when the cumulative translation of a returned block is below 0.15 m. GN0's `RemoteNavAgent` defaults to 0.15 m (`gn0/agents/remote.py` line 128), but `eval_remote.sh` sets `REMOTE_MIN_CHUNK_TRANSLATION="0"` (disabled, line 46) and passes it explicitly, and the README command does not override it. We pass `--remote-min-chunk-translation 0.15`. Recorded as a deviation from the released `eval_remote.sh`.

**Open, pending the lead's answer: `--enable-stall-recovery`.** The released README client command passes it (README line 155); the paper does not mention it. With it, a stalled agent backs up 8 x 0.08 m on the first two stall triggers and stops on the third (`gn0/evaluation/remote.py` lines 36 to 38 and 531 to 557); without it, the first trigger stops the episode. It is the config key `client.enable_stall_recovery`, set to `pending`; `run_eval.sh` refuses to start any run (timing or main) until it is `true` or `false`.

**Other choices that change no model setting** (listed so nothing is hidden):
- Episode restriction: GN0's client has only a contiguous start/end range. `src/rung1_hook/` is put on the client's `PYTHONPATH` for that one call; its `sitecustomize.py` wraps two functions of `gn0/evaluation/common.py` in memory: `select_eval_episodes` keeps only the requested ids (and fails if one is missing from the dataset), `atomic_write_json` also appends one line per finished episode with its wall time. If the hook cannot attach, the client exits at once rather than running unfiltered episodes.
- The server runs with `HF_HUB_OFFLINE=1`, so anything not already downloaded and pinned fails loudly instead of being fetched.
- flash-attn pinned to 2.8.3.post2, the version route-a check 2 installed (`docs/INSTALLATION.md` section 1.5 gives none).
- Library repairs from route-a check 2 in the `wnm_3d` environment: huggingface_hub 0.36.2 (the upgrade in `docs/DOWNLOADS.md` breaks transformers 4.51.3) and pyarrow 23.0.1 (newer needs numpy 2).
- InteriorGS: only the 839 test_unseen scenes are downloaded (whole scene folders), not the full dataset that `docs/DOWNLOADS.md` downloads.
- GN-Bench shuffles episode order inside a run (`SHUFFLE: true`), so episodes finish in a random order; each sampling call seeds its own generator with 1140, so we expect the order not to change an episode's outcome beyond GPU non-determinism (`speculative`, not tested).

## Pass bar and predictions (before running)

**Reproduced** if the paper's 53.5 lies inside the 95% Wilson score interval of our measured success rate on the first N episodes of the frozen order (interval closed at both ends). Otherwise **not reproduced**; a run that did not finish all N episodes gets no verdict (`incomplete`). SPL, OS, NE and TL are reported next to the paper's values with no pass bar of their own.

For scale (arithmetic only): at a success rate near 53.5 %, the 95% interval is about plus or minus 3 points for N = 1,000 and about plus or minus 4.4 points for N = 500.

**Also reported** (`src/summarize.py`):
- Success (and SPL, OS, NE, TL) split by the route-a check 3 visible flag at tolerance 0.15 m (`visible_primary`), each with a Wilson interval.
- The same with and without the 39 possibly mirrored scenes (233 test_unseen episodes; route-a check 3 frame diagnostic, rule as in `summarize_check3.py`).
- Seconds per episode (mean, median, and without the first episode of each launch, which includes start-up) and USD per episode from the instance's hourly price.
- Stop reasons, as GN0 records them.

What results would mean: inside the interval, our set-up is good enough to read later rungs against the paper's number. Below it, something in our set-up differs (the stop rule, stall recovery, hardware, library versions) and later rungs compare against our own measured baseline, not the paper's. Above it would also be a mismatch to explain.

## Compute

Vast.ai, one H100 SXM, single GPU, on-demand (the server was validated on an H100 80 GB at about 27 GiB, WNM-3D `docs/INSTALLATION.md` line 39; GN-Bench renders on the same GPU).
1. **Timing rental** (needs its own OK): setup plus the first K episodes, to measure setup time and seconds per episode.
2. **Main run** (needs its own OK): the first N episodes, N picked by the lead from step 1.
Results are copied off the machine and checked before it is destroyed; they live outside every git worktree (`~/Desktop/hear-lab-rung1-results/`). Procedure: `src/RUN.md`.

## Scripts
- `src/make_episode_order.py`: writes the frozen order (done once on the Mac; sha256 above).
- `src/setup.sh`: environments, pinned downloads, checks.
- `src/run_eval.sh`: server plus client on the first N episodes; resumable; writes `run_meta.jsonl` (every setting passed) per launch.
- `src/build_commands.py`: builds the exact commands from `config.yaml`.
- `src/check_settings.py`: asserts every setting (config, commands, setup pins, pinned sources, episode order).
- `src/summarize.py`: metrics, intervals, verdict, breakdowns, time and cost.

**Status:** planned. Stall recovery and K decided (amendment 2026-11-21 below). Waiting on: the timing rental OK, then N after it.

## Amendments
<!-- ### YYYY-MM-DD: what changed, why, and whether any results had been seen yet -->

### 2026-11-21: stall recovery on, K = 30 (the user's answers, before any run)
The lead put two recommendations to the user, who answered in the lead's window on 2026-11-21 ("yes go for it"):
- **Stall recovery stays on**, as in the released WNM-3D README client command (README line 155): `client.enable_stall_recovery: true`, so the client passes `--enable-stall-recovery`. The paper does not mention it, so this follows the released command, not the paper; it is not counted as a deviation from the released command.
- **Timing batch K = 30**: `episodes.timing_k: 30`, the first 30 ids of the frozen order. `run_eval.sh` accepts `--n 30` for the timing run.
- **N for the main run stays open** (`episodes.n_main: pending`), picked by the lead after the timing rental.

No run had started and no results existed when this was decided. `check_settings.py` now asserts both decided values.
