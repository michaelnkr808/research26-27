# Rung 1: released WNM-3D at inference time in GN-Bench test_unseen

**Results:** main run done (2026-11-23, Vast.ai, 1x H100 SXM, 2.3704 USD/h, 16.94 h, 40.15 USD; plus the timing batch on 2026-11-22, 1.82 h, 4.32 USD). Verdict **reproduced**: on the first 1000 episodes of the frozen order the released model succeeded in 550 (SR 55.0, Wilson 95% 51.9 to 58.1), which contains the paper's Unseen 53.5. Target visible at start: SR 65.0 (n 220); not visible: 52.2 (n 780). SPL 51.2 (paper 46.6), OS 61.5 (62.2), NE 4.04 (4.1), TL 9.07 (10.2). The 30 timing episodes repeated identically. Detail in `LOG.md` 2026-11-23. Design and pass bar in `PLAN.md`, every parameter in `config.yaml`.

What it does: runs the released WNM-3D stage 3 checkpoint, unchanged and without audio, in GN-Bench test_unseen on one rented H100, and checks whether the paper's Unseen success rate (53.5) lies inside the 95% interval of ours. One deviation from the released command (the paper's 0.15 m block-translation stop rule); stall recovery on, as released.

## Contents
- `config.yaml`: all settings (asserted by `src/check_settings.py`).
- `src/RUN.md`: how to set up and run on the rented machine, copy back and summarize.
- `src/`: `make_episode_order.py`, `setup.sh`, `run_eval.sh`, `build_commands.py`, `rung1_hook/` (episode filter for the GN0 client), `check_settings.py`, `summarize.py`, unit tests.
- `results/`: gitignored. `episode_order.txt` (regenerate with `src/make_episode_order.py`; sha256 in `config.yaml`). Run outputs go to `~/Desktop/hear-lab-rung1-results/`, outside every worktree.

## Local checks (Mac)
~~~bash
cd hear-lab/experiments/2026-11-21-rung1-wnm3d-baseline
/opt/homebrew/bin/python3.11 -m pytest -q src
/opt/homebrew/bin/python3.11 src/check_settings.py
~~~

## Main run (reproduce)
- Command on the rented machine: `run_eval.sh --run-name main --n 1000 --repo-commit 4d9ceed6bef180d4bc344510ac9bd59fbeab1509` (see `src/RUN.md`).
- Summary on the Mac: `src/summarize.py --run-kind main` as in `src/RUN.md` step 5, with `--usd-per-hour 2.3703703703703707`.
- Outputs: `~/Desktop/hear-lab-rung1-results/main-sync/` (not in git; sha256 manifest `copy_manifest_main.sha256`, summary in `runs/main/summary.json`).
- LOG entries: 2026-11-22 (timing rental, main run start), 2026-11-23 (main run result).
