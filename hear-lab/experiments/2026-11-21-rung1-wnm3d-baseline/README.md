# Rung 1: released WNM-3D at inference time in GN-Bench test_unseen

**Results:** timing batch only (2026-11-22, Vast.ai, 1x H100 SXM, 2.3704 USD/h, 1.82 h, 4.32 USD). On the first 30 episodes of the frozen order the released model succeeded in 18 (SR 60.0, Wilson 95% 42.3 to 75.4; the paper's Unseen 53.5), at about 45 s and 0.03 USD per episode; a repeat of 18 episodes gave identical outcomes. This is not the pass-bar test, which is on the main run (N not yet chosen). Detail in `LOG.md` 2026-11-22. Design and pass bar in `PLAN.md`, every parameter in `config.yaml`.

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

<!-- After the run: command to run, commit hash, outcome, where the outputs are, and links to the LOG entries (see _standards/03-reproducibility-and-data.md). -->
