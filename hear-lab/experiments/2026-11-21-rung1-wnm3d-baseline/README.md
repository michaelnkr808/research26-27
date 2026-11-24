# Rung 1: released WNM-3D at inference time in GN-Bench test_unseen

**Results:** main run done (2026-11-23, Vast.ai, 1x H100 SXM, 2.3704 USD/h, 16.94 h, 40.15 USD; plus the timing batch on 2026-11-22, 1.82 h, 4.32 USD). Verdict **reproduced**: on the first 1000 episodes of the frozen order the released model succeeded in 550 (SR 55.0, Wilson 95% 51.9 to 58.1), which contains the paper's Unseen 53.5. Target visible at start: SR 65.0 (n 220); not visible: 52.2 (n 780). SPL 51.2 (paper 46.6), OS 61.5 (62.2), NE 4.04 (4.1), TL 9.07 (10.2). The 30 timing episodes repeated identically. Detail in `LOG.md` 2026-11-23. Design and pass bar in `PLAN.md`, every parameter in `config.yaml`.

What it does: runs the released WNM-3D stage 3 checkpoint, unchanged and without audio, in GN-Bench test_unseen on one rented H100, and checks whether the paper's Unseen success rate (53.5) lies inside the 95% interval of ours. One deviation from the released command (the paper's 0.15 m block-translation stop rule); stall recovery on, as released.

## Contents
- `config.yaml`: all settings (asserted by `src/check_settings.py`).
- `src/RUN.md`: how to set up and run on the rented machine, copy back and summarize.
- `src/`: `make_episode_order.py`, `setup.sh`, `run_eval.sh`, `build_commands.py`, `rung1_hook/` (episode filter for the GN0 client), `check_settings.py`, `summarize.py`, unit tests.
- `results/`: raw outputs, on the lead's Mac only (see "Outputs and what is in git" below). `results/episode_order.txt` is regenerated with `src/make_episode_order.py` (sha256 in `config.yaml`).
- `env/`: environment lists (pip freeze and conda list of both environments) for the timing and main rentals.

## Local checks (Mac)
Need Python 3.11 with numpy, pyyaml and pytest (what `src/` imports). Run from this folder:
~~~bash
cd hear-lab/experiments/2026-11-21-rung1-wnm3d-baseline
python3.11 -m pytest -q src
python3.11 src/check_settings.py
~~~

## Runs

Run name, command, repo commit, outputs (relative to this folder), LOG entry. All three used `src/RUN.md` (setup step 2, run step 3, copy back step 4, summary step 5) on one rented H100 SXM.
- **timing** (timing run 1, 18 of 30 episodes, then a hook crash): `run_eval.sh --run-name timing --n 30 --repo-commit 599f4ca6de4a071a1e9ef021c255b7dc9620fc36`. Outputs `results/timing/runs/timing/` (not in git). LOG 2026-11-22 (timing rental).
- **timing-r2** (all 30 episodes, after the fix): `run_eval.sh --run-name timing-r2 --n 30 --repo-commit eefdecc0fc2c35b0b0ae4094db34600c2299656a`. Outputs `results/timing/runs/timing-r2/` (tracked there: `summary.json`, `rung1_episodes.jsonl`). LOG 2026-11-22 (timing rental).
- **main** (1,000 episodes): `run_eval.sh --run-name main --n 1000 --repo-commit 4d9ceed6bef180d4bc344510ac9bd59fbeab1509`. Outputs `results/main/runs/main/` (tracked there: `summary.json`, `rung1_episodes.jsonl`). LOG 2026-11-22 (main run started) and 2026-11-23 (main run finished).

## Outputs and what is in git

- Raw outputs exist in one copy, on the lead's Mac. One set of raw outputs was already lost on 2026-11-20 in the sister experiment `../2026-11-20-route-a-checks/` (a worktree removal deleted gitignored files). Where a backup goes is still the user's choice.
- The user's answer on InteriorGS (2026-11-24, `../../data/README.md`): scores, summaries and per-episode tables (episode ids, success, distances) may be committed; scene images, rendered audio, geometry, floor plans, meshes and boxes stay out. `results/.gitignore` ignores everything except an allow list: the two copy manifests (`results/timing/copy_manifest.sha256`, `results/main/copy_manifest_main.sha256`) and, for timing-r2 and main, `summary.json` and `rung1_episodes.jsonl`. Left out on purpose: logs, `run_meta.jsonl`, `episode_ids.txt`, `results/episode_order.txt` (scene names; regenerable), the timing run 1 files, `partial-copy/`, instance records, and the `gn0/` folder (GN0's own per-episode files).
- Environment lists, committed in `env/`: `timing-*` and `main-*`, one pip freeze and one conda list per environment (`wnm_3d`, `bae`). Searched for tokens, passwords, home paths and `@` before committing: nothing secret (the only `@` lines are conda `file:///home/conda/feedstock_root/...` build paths and git pins).

## Main run (reproduce)
- Command on the rented machine: `run_eval.sh --run-name main --n 1000 --repo-commit 4d9ceed6bef180d4bc344510ac9bd59fbeab1509` (see `src/RUN.md`).
- Summary on the Mac: `src/summarize.py --run-kind main` as in `src/RUN.md` step 5, with `--usd-per-hour 2.3703703703703707`.
- Outputs: `results/main/` (sha256 manifest `copy_manifest_main.sha256` and `runs/main/summary.json` are in git; the rest is not).
- LOG entries: 2026-11-22 (timing rental, main run start), 2026-11-23 (main run result).
