# Rung 1 on the rented machine: how to run it

What this does: sets up WNM-3D and GN0 on one Vast.ai H100, then runs the released server and client on the first N (or K) episodes of the frozen test_unseen order (`../PLAN.md`). Every parameter is in `../config.yaml`. Renting, the hourly price and destroying the instance are the experiment lead's (`../../../STANDARDS.md`, "Where experiments run"); each rental needs its own OK.

Placeholders: `<host>`, `<port>` (SSH, from `vastai ssh-url <id>`), `<id>` (instance), `<REPO>` (this repo on the Mac), `<K>` or `<N>` (episode count the lead picked), `<RUN>` (`timing` or `main`).

Before anything: `client.enable_stall_recovery` in `config.yaml` must be `true` or `false` (the lead's answer), committed. `run_eval.sh` refuses to start while it is `pending`.

## 0. On the Mac

~~~bash
cd <REPO>/hear-lab/experiments/2026-11-21-rung1-wnm3d-baseline
git rev-parse HEAD                                    # record in LOG.md; passed to run_eval.sh as --repo-commit
/opt/homebrew/bin/python3.11 -m pytest -q src
/opt/homebrew/bin/python3.11 src/make_episode_order.py \
  --csv ~/Desktop/hear-lab-route-a-results/2026-11-20-check3-rerun/check3_full/check3_InteriorGS_test_unseen.csv \
  --out results/episode_order.txt                     # must print sha256 25366209...2728 and no MISMATCH
/opt/homebrew/bin/python3.11 src/check_settings.py --order-file results/episode_order.txt
rm -rf src/__pycache__ src/rung1_hook/__pycache__ .pytest_cache
~~~

## 1. Upload the folder and the token (from the Mac)

~~~bash
ssh -p <port> root@<host> mkdir -p /workspace/rung1/exp
rsync -av -e "ssh -p <port>" --exclude __pycache__ --exclude .pytest_cache \
  <REPO>/hear-lab/experiments/2026-11-21-rung1-wnm3d-baseline/ root@<host>:/workspace/rung1/exp/
~~~

The Hugging Face token (InteriorGS is gated) goes to tmpfs only, from the Mac's saved token file, without entering a command line or the transcript, as on rental 5 of route-a:

~~~bash
ssh -p <port> root@<host> 'umask 077 && cat > /dev/shm/hft' < ~/.cache/huggingface/token
~~~

`setup.sh` reads it only for the InteriorGS download and deletes it when that stage ends.

## 2. Setup (on the machine, in tmux)

~~~bash
tmux new -s rung1
cd /workspace/rung1/exp/src
bash setup.sh 2>&1 | tee -a /workspace/rung1/env/setup_console.log
~~~

Stages (each can be rerun alone: `bash setup.sh downloads verify`): `record`, `apt`, `conda`, `env-wnm`, `env-bae`, `downloads`, `verify`, `token-check`. It stops on: a HEAD that is not the pin, a checkpoint or parquet sha256 mismatch, a scene count other than 839, a missing simulator file, the frozen order not matching the parquet's ids, any `check_settings.py` difference, a pre-existing `sitecustomize` in the `bae` environment, a token file left on disk.

Read `/workspace/rung1/env/checkpoint_conf_keys.txt`: if the checkpoint names a tokenizer other than `google/umt5-xxl` or a path that does not exist, stop and tell the lead (the server runs offline).

## 3. Run (on the machine, in tmux)

~~~bash
cd /workspace/rung1/exp/src
bash run_eval.sh --run-name <RUN> --n <K or N> --repo-commit <sha from step 0>
~~~

What it does, in order: `check_settings.py` against config, commands, setup pins, pinned sources and order; refuses if anything answers on port 8000; writes `/workspace/rung1/runs/<RUN>/episode_ids.txt` (first N of the order) and appends every setting to `run_meta.jsonl`; starts the server and waits for `/healthz` (up to 30 min); checks the server's banner lines; starts the client with the episode filter; waits for the filter to attach (up to 15 min) and checks the client's banner lines; waits for the client; appends an end record (finished count, no foreign ids).

If the machine or the run drops, run the same command again: finished episodes are skipped. A resumed launch must have the same settings, N and repo commit, or it refuses.

While it runs (second tmux window): progress with GN0's own summary, and the GPU:

~~~bash
source /workspace/miniforge3/etc/profile.d/conda.sh && conda activate bae
cd /workspace/GN0 && python -m gn0.analysis.results --path /workspace/rung1/runs/<RUN>/gn0
tail -3 /workspace/rung1/runs/<RUN>/rung1_episodes.jsonl
nvidia-smi
~~~

Look at GPU use 15 minutes in; if the GPU idles, say so in LOG.md.

## 4. Copy back and destroy

On the machine:

~~~bash
cd /workspace/rung1
find env runs -type f | sort > copy_list.txt
xargs sha256sum < copy_list.txt > copy_manifest.sha256
~~~

On the Mac (outside every git worktree):

~~~bash
mkdir -p ~/Desktop/hear-lab-rung1-results
rsync -av -e "ssh -p <port>" --files-from=:/workspace/rung1/copy_list.txt root@<host>:/workspace/rung1/ ~/Desktop/hear-lab-rung1-results/
rsync -av -e "ssh -p <port>" root@<host>:/workspace/rung1/copy_manifest.sha256 ~/Desktop/hear-lab-rung1-results/
cd ~/Desktop/hear-lab-rung1-results && shasum -a 256 -c copy_manifest.sha256 | grep -vc ': OK$'   # must print 0
~~~

Then `bash setup.sh token-check` on the machine once more, read the credit, destroy the instance, read the credit again, and write machine, price, hours, cost into LOG.md.

## 5. Summarize (on the Mac)

~~~bash
cd <REPO>/hear-lab/experiments/2026-11-21-rung1-wnm3d-baseline
C3=~/Desktop/hear-lab-route-a-results/2026-11-20-check3-rerun/check3_full
/opt/homebrew/bin/python3.11 src/summarize.py --run-dir ~/Desktop/hear-lab-rung1-results/runs/<RUN> \
  --order-file results/episode_order.txt --n <K or N> \
  --check3-csv $C3/check3_InteriorGS_test_unseen.csv --scenes-jsonl $C3/scenes.jsonl \
  --usd-per-hour <dph_total of the instance> --out ~/Desktop/hear-lab-rung1-results/runs/<RUN>/summary.json
~~~

The verdict is only meaningful for the main run (the pass bar is on the first N). For the timing run, read `timing` in the summary: seconds and USD per episode, with and without the first episode of each launch.
