# How to run the sizing check on Modal (one L40S)

Code: `src/pretrained_sizing.py` (model, loader, sanity, per-process measurement, launcher), `modal_app.py` here (volume, image, CPU staging, GPU run, fetch). Design: `PLAN.md`; loading facts: `step2-reading.md`.

Nothing below runs without the user's go in the implementation session's chat, given after that session states the provider (Modal), card (L40S), the day's hourly price and the estimated total (PLAN Step 1; `STANDARDS.md`, Paid compute). Steps 2, 3 and 7 move data or only read; steps 4, 5 and 6 start containers and bill (4 and 5 on CPU only, 6 on the GPU).

Choices this run uses (the user's, 2026-11-12): blocks 0-7 kept, tagged "first 8, user's choice" in every results file; extra encoder DINOv3 (1,024 channels); the proxy's precision (fp32 weights, bf16 autocast, fp32 AdamW); extra-channel slices of the widened patch embedding and head start at zero; action expert, action encoder and head and proprio encoder fresh as the proxy builds them (Fast-WAM itself derives its action expert from the Wan DiT, `step2-reading.md` Q4; not done here).

All commands run from this folder (`world-model-stacking-encoders/experiments/2026-11-12-pretrained-start-sizing/`), with the Modal CLI 1.6.4 logged in (`modal profile current` shows the workspace).

## 1. Day's price (PLAN Step 1)
```
modal billing rates
```
Note the L40S, CPU and memory rates in this folder's `LOG.md` with the date, next to the Runpod and Vast.ai prices.

## 2. Volume and data slice (no compute)
The slice is the separate worker's local folder `sizing-slice/` (16 training episodes, proxy cache layout: `meta/`, `vae/` and `vae_b/`, `dinov3/`, `t5/`; about 2.5 GB).
```
modal volume create stacking-sizing
modal volume put stacking-sizing <path to>/sizing-slice data/sizing-slice
modal volume ls stacking-sizing data/sizing-slice
```
Normal output of the last line: the folders `meta`, `vae` (and `vae_b` if the slice uses it), `dinov3`, `t5`. If they appear one level deeper (`data/sizing-slice/sizing-slice/...`), remove with `modal volume rm -r stacking-sizing data/sizing-slice` and repeat `put` with a trailing `/` on the local path.

## 3. Commit the code
`modal_app.py`'s `run` refuses to start if `src/` or `modal_app.py` has uncommitted changes, so the results name a real commit (`SRC_GIT_COMMIT` is passed into the container, which has no `.git`).

## 4. Stage the Wan 2.2 weights (CPU only, bills CPU time)
```
modal run modal_app.py::stage
```
Downloads from Hugging Face `Wan-AI/Wan2.2-TI2V-5B` (Apache-2.0) only `config.json`, the safetensors index and the shards that hold blocks 0-7 and the non-block tensors: shard 1, 9,825,014,472 bytes (`step2-reading.md` Q1, Q7). Checks the size, writes `wan/Wan2.2-TI2V-5B/manifest.json` on the volume with the resolved Hugging Face revision and each file's sha256. Copy the revision and sha256 into `LOG.md` (and into `data/README.md` if the weights are used later, `_standards/07`). Time: a few minutes (not measured).

## 5. Preflight (CPU only, on the run image)
```
modal run modal_app.py::check
```
The first call builds the run image (Python 3.11, torch 2.7.1 + torchvision 0.22.1 CUDA 12.8 wheels, the rest of `experiments/2026-10-27-proxy-calibration/requirements.txt`, Fast-WAM `7faa711` from its GitHub tarball with the pinned sha256, `--no-deps`). Whether Modal bills image builds is read on the day. Normal output: a JSON with the Wan revision, `"missing": []`, for `none` and `dinov3` the slice's suites, 16 episodes and dims (`dinov3: 1024`), and `local_disk_free_gib`. If the free disk is under 40 GiB, stop: the saved model + AdamW state (about 20 GB) won't fit and the GPU run would refuse to start (set `ephemeral_disk` on `run_sizing` then).

## 6. The run (one L40S, bills GPU, CPU and memory)
```
modal run --detach modal_app.py::run --run-id run1
```
`--detach` keeps the run going if the terminal disconnects; the fetch at the end then doesn't happen, so use step 7.

Limits (PLAN "Cost and time cap"): Modal function `timeout=5400` s on `run_sizing` (outer, 90 minutes; container start and image pull before the function starts are outside it, so the billed total can be a few minutes longer); inside it `timeout --signal=TERM --kill-after=60 3600` around the launcher (inner, 60 minutes). Order: config `none`, then `dinov3`; per config the sweep m = 1, 2, 4, ..., 128 (one process each, stopping at the first `oom` or `error`; the m = 1 process loads Wan, writes `load-<config>.json` and `sanity-<config>.json` and saves the state to container-local `/tmp/sizing-state`), then the full optimizer steps at global batch 128 with m*, and once more at m*/2 if that ends in `oom`. Each sweep row is written to the volume's `results/run1/measure-<config>.json` as its process ends, `run.log` holds every line, and the volume is committed every 2 minutes, so a run stopped by either limit keeps what it has. The function refuses a run id that already has results.

Watch (optional, no extra billing): `modal app logs <app id>` or the dashboard (`modal dashboard`). The app stops by itself when the function returns or hits its timeout; confirm with `modal app list` that nothing from `stacking-encoders-sizing` is still running, and `modal app stop <app id>` if it is.

## 7. Fetch the results (no compute)
```
modal volume get stacking-sizing results/run1 results/
```
or `modal run modal_app.py::fetch --run-id run1` (keeps any local file that already exists). Results land in `results/run1/`: `measure-none.json`, `measure-dinov3.json`, `load-*.json`, `sanity-*.json`, `rows/` (one JSON per process) and `run.log`. A rerun uses a new run id (`_standards/03`, never overwrite).

## 8. Cost after the run (STANDARDS.md: estimated vs actual dollars in LOG.md)
```
modal billing report --for today --show-resources --resolution h
modal billing summary
```
The report splits the day's cost by resource (L40S, CPU, memory); the summary shows the month's total against the $30 Starter credit. The same figures are on the dashboard's usage page (`modal dashboard`). Billing data may lag the run by some time (recalled, not checked); if the figure is missing, read it again later and note when it was read.

## 9. Clean up
Volume storage may bill (recalled; check `modal billing rates`). After the results are fetched and logged, delete the weights unless a real run will reuse them soon (the user's call):
```
modal volume rm -r stacking-sizing wan
```
The data slice and results are small; keep or remove them the same way.
