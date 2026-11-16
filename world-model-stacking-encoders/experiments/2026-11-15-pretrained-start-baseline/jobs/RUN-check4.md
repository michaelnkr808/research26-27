# Sanity check 4 on the cluster: submit steps for the lead

Written 2026-11-15 by a worker, Mac only. Nothing here has been submitted or dry-run, and no kubectl command was run to write it. Two Jobs, in order: `jobs/job-check4-train.yaml` (Job `stacking-encoders-baseline-check4-train`, one L40 or L40S) and then, only after it succeeds, `jobs/job-check4-smoke.yaml` (Job `stacking-encoders-baseline-check4-smoke`, one L4). Each header has what it runs, every resource number's source and the deadline arithmetic. Paths are relative to this experiment folder unless they start with `src/`. `<scratch>` is the lead's session scratchpad. `$C` is the source commit named in step 2. The format follows `jobs/RUN-sanity.md`, whose steps are the model.

**Each submit needs the user's go in the lead's window** (a question or an approval of these files is not a go; memory "ask before new cluster work"). State, before asking: the Job, the card (L40 or L40S; L4), the pod time and GPU-hours below, and the GPU count (step 6). The design is already locked (the sanity Job ran 2026-11-16); nothing here changes the PLAN.

**`ckpt.pt` and `model.pt` from this check stay on the PVC at `/mnt/proxy/runs/baseline-check4-none-s0-try<TRY>/` until the user OKs deleting them** (PLAN "Checkpointing and resume", "At the end"; deleting data is gated, `../../../_standards/05`). Nothing in either Job deletes anything. They are about 20 GB and 6.7 GB (PLAN arithmetic, speculative until written).

## What the two Jobs do, and what each costs (estimates, speculative; sources in the headers)

| | Train | Smoke |
|---|---|---|
| Card | L40 or L40S by `gpu.product` | L4 only |
| Runs | `pretrained_train.py`, 300 optimizer steps, a checkpoint every 100, micro-batch 8, `--out` on the PVC | `rollout.py run --timing-only --clean 1` = 40 clean rollouts, 4 environments, CUDA graph, 600 s monitor windows |
| Pod time, expected | about 42 min (three writes) if writes run near 100 MB/s | about 25 to 55 min |
| Pod time, worst-case allowance | 73.5 min (writes at 30 MB/s; header arithmetic) | restore 9 min + load up to 4 min + rollouts, capped at 90 min per pass |
| GPU-hours | about 0.6 (expected), 1.2 (worst case) | about 0.4 to 0.9 |
| Pod deadline / Job deadline | 9000 s (150 min) / 81000 s (22.5 h, queue included) | 7800 s (130 min) / 79200 s (22 h) |
| Needs the user's go | yes | yes, after the train Job succeeded |

**Checkpoint write count (the user's decision):** `CKPT_EVERY` is `"75"` in the committed file, so 300 steps give THREE timed writes, at steps 75, 150 and 225; step 300 ends the run and writes `model.pt` (`src/pretrained_train.py` line 324, `s + 1 < stop`).

## Before submitting the train Job

1. **The user's go**, in this window, after the statement above.

2. **Name the source commit** (after this branch is merged and pulled, on a clean `main`): `git rev-parse HEAD` gives `$C`. The pods read only ConfigMaps, never git.

3. **ConfigMaps.** Two, both in `aria-world`.
   1. `stacking-encoders-baseline-src` (7 keys) **already exists** from the sanity Job (LOG 2026-11-16: keys equal `c0ad302`'s). Reuse it unchanged for the train Job: read each key back and compare with these hashes, which equal `src/` at `85da306` (verified on the Mac, `shasum -a 256`; `src/` last changed at `6875f53`, so they must equal `$C`'s too: recompute at `$C`):
      `kubectl get configmap stacking-encoders-baseline-src -n aria-world -o jsonpath='{.data.pretrained_train\.py}' | shasum -a 256` (escape the dot as `\.`), for each of:
      - `pretrained_train.py` `e79ad083f8ac02af236e78de74c5b85af7271f0c86356d59086198973cfe6473`
      - `pretrained_sizing.py` `c14ccec63e66075c561573c315ed11a0abf3ce5e0fabd3751dbb599f64a52bb7`
      - `action_expert_init.py` `32e325315b766d601ac0fe60cb79e1c3cbf60caee96cd460682bf63c2431c3cf`
      - `proxy_wam.py` `e09543d5bf0bf8ce2a7d1db9048f8c7819366f97fb4ae0c552e802412e32fd0a`
      - `diagnostics.py` `7af1f7aa2d595d9e0b7f1fb462f7901c6fa9594d2da7895b0168f0ef3ea2d70d`
      - `t5_cache.py` `61181a74af8825eff0e0b8cb3a220f8483d9f1982642cd90b753cb9e5a87d708`
      - `setup_env.sh` `880c47cbe741b7fbcd98f44f85e7a1d14747d6c837e0493f4642e5ebbc18ace8`
      If any differs, stop and ask (the pod also prints the 7 hashes, so the log records what ran).
   2. `stacking-encoders-baseline-eval-src` (11 keys, **new, create it before the smoke Job, any time before step 12**): the import chain of `src/rollout.py` plus the three scripts of the env-bundle restore. Why a separate ConfigMap: `baseline-src` has no `rollout.py`, `setup_sim.sh` or `env_bundle.sh`; `stacking-encoders-proxy-src` (14 keys) holds an older `rollout.py` without `--timing-only` and other Jobs mount it, so it is not touched (header of the smoke Job). The 11 keys and their hashes at `85da306` (verified on the Mac; 171,962 bytes in all, far under the 1 MiB limit):
      - `rollout.py` `a097769427138a53828055e888be936545a9feef67e4e488b2fbb4ffceaf7227`
      - `lp_eval.py` `6f7ed893bc826ad8d0465ea5f34992d1fb548a5114a00d0f233055795512db6d`
      - `proxy_wam.py` `e09543d5bf0bf8ce2a7d1db9048f8c7819366f97fb4ae0c552e802412e32fd0a`
      - `t5_cache.py` `61181a74af8825eff0e0b8cb3a220f8483d9f1982642cd90b753cb9e5a87d708`
      - `diagnostics.py` `7af1f7aa2d595d9e0b7f1fb462f7901c6fa9594d2da7895b0168f0ef3ea2d70d`
      - `extract_fastwam.py` `42eb27b49051f5c7401e55c2730b3d4d3bf3371f28340b40317b8e87ccec66f0`
      - `extract_features.py` `03734ba84a8696e0d9f605ec89c11b50f033af9ea41cb4ef841c3c30742bd6a1`
      - `fastwam_data.py` `81993810937d2692597d1d161e438f7688dc7809d0017628d9603ef1d196e363`
      - `setup_env.sh` `880c47cbe741b7fbcd98f44f85e7a1d14747d6c837e0493f4642e5ebbc18ace8`
      - `setup_sim.sh` `bf9651099926914da0e4b32b04d317260a38be8d4b87dcba9813637d1ed91ae3`
      - `env_bundle.sh` `f3017146fd7e89b79709f5639c003d0688ccafce8257705c566290ccb99b4ef9`
      **The last three must be exactly these**: the prebuilt rollout environment `/mnt/proxy/env/env-9182f8131559.tar` is keyed by their combined hash (`src/env_bundle.sh` line 16; proxy LOG line 55), and the restore fails if they differ. The smoke pod prints the key it computes.
      Steps: write the 11 files from `$C` into an empty `<scratch>/baseline-eval-src` (`git show $C:world-model-stacking-encoders/src/$f > ...`, from the repo root); compare with the hashes above; check the name is free (`kubectl get configmap stacking-encoders-baseline-eval-src -n aria-world`: NotFound); `kubectl create configmap stacking-encoders-baseline-eval-src -n aria-world --request-timeout=30s --from-file=<scratch>/baseline-eval-src --dry-run=server`, then without `--dry-run=server`; read every key back and compare. Record in the LOG that it holds `$C`'s files. Keep it until the full evaluation Job is prepared.
   Option: add `rollout.py`, `lp_eval.py`, `extract_fastwam.py`, `extract_features.py`, `fastwam_data.py`, `setup_sim.sh`, `env_bundle.sh` as keys of `baseline-src` instead of a second ConfigMap (one ConfigMap for the whole experiment; the train Job's `sha256sum` line would then miss the extra keys, harmless). Not done here: it modifies an existing object, so it is the lead's call.

4. **PVC free space.** The train Job writes up to about 40 GB at its peak (`ckpt.pt` plus `ckpt.pt.tmp` during the second write), then about 27 GB stays (`ckpt.pt` 20 GB, `model.pt` 6.7 GB). The pod aborts with exit 14 before starting if the PVC has under 60 GiB free, and prints the figure. The last known free space was 216G (sanity pod log line 26, 2026-11-16, `results/sanity/stacking-encoders-baseline-sanity-cj4h7.log`); the Wan shards and any later Jobs may have changed it. No separate pod is needed for this check. If it exits 14, tell the user: the PVC is nearly full and nothing may be deleted without their OK.

5. **Render** the train manifest with the commit filled, replacing only the env value line:
   ```
   mkdir -p jobs/rendered
   sed "s/{name: SRC_GIT_COMMIT, value: FILL_AT_SUBMIT}/{name: SRC_GIT_COMMIT, value: $C}/" jobs/job-check4-train.yaml > jobs/rendered/job-check4-train.yaml
   grep -c "value: $C}" jobs/rendered/job-check4-train.yaml    # must print 1
   grep -c FILL_AT_SUBMIT jobs/rendered/job-check4-train.yaml   # must print 0
   ```
   `grep -n FILL_AT_SUBMIT jobs/job-check4-train.yaml` on the unrendered file shows exactly one line, the env value (verified). Never edit the original for one run; commit the rendered file as submitted. `TRY` stays `"1"` for the first submit; a resubmit needs `"2"`, `"3"`, ... (the pod aborts with exit 11 if `/mnt/proxy/runs/baseline-check4-none-s0-try<TRY>` exists). `MICRO_BATCH` stays `"8"` (measured fit on the L40, LOG 2026-11-16).

6. **GPU count:** the GPUs our Jobs hold or are Pending for, plus 1, must be at most 8 (`STANDARDS.md` line 49).

7. **Server-side dry run, then submit:** `kubectl create --dry-run=server -f jobs/rendered/job-check4-train.yaml`, then `kubectl create -f jobs/rendered/job-check4-train.yaml`. Note the creation time in the LOG. The Job deadline (81,000 s, 22.5 h) counts from now, queue included.

## While the train Job runs

8. **One wait, in the background**, the loop of `jobs/RUN-sanity.md` step 7 with `J=stacking-encoders-baseline-check4-train` and `END=$(( $(date +%s) + 81500 ))` (and "wait timed out (81500 s)" in its message). Same behaviour: 5-minute checks, one `END:` line (`Job succeeded`, `Job failed <reason>`, `kubectl failed`, or timeout); a `kubectl-error` line means the OIDC login expired: stop, ask the user to re-login, restart the loop. If the wait will pass about 50 min, write the handoff note first.

9. **The 15-minute GPU-use check** (`STANDARDS.md` line 48): about 15 min after the pod is Running: `kubectl get pods -n aria-world -l job-name=stacking-encoders-baseline-check4-train -o name`, then `kubectl logs <pod> -n aria-world | grep -E '^=== PHASE|^\[gpu\]|^\[train|\[ckpt\]|\[projection\]|^PVC free' | tail -40`. Expect: `PVC free: <n> GiB (need 60)`; phases `setup` and `stage-wan` at 0% for about 2 min (sanity: 33 s and 91 s); then `train` at 95 to 100% GPU with about 5 s per step; after step 100 a `[ckpt] step 100 written to ... in <s>s` line and a `[projection] with checkpoint writes:` line. What counts:
   - `device=cpu`, a CUDA error, or `STOP:`: delete the Job and tell the user.
   - The first `[ckpt]` write time is the number this check exists for. The binding limit is the in-pod `timeout` on the training command (7,200 s from Python start), which fires before the 9,000 s pod deadline: 60 s build + 1,512 s of steps + the writes + the `model.pt` write must fit, so with three writes a write time over about 1,300 s (about 15 MB/s) ends the pod with exit 124 and no `model.pt`. If the first `[ckpt]` time is over about 700 s (the allowance), tell the user and decide with them whether to let it run; the measured write time sets the full run's checkpoint cost (step 11).
   - Steps slower than 6 s each, or `data_wait_s` above about 1 s: note it in the LOG.
   If still Running at 30 to 40 min, check the Violations page (`_standards/08` pre-flight item 8).

## After the train Job ends

10. **Save the pod log right away** (logs vanish within about a day): `mkdir -p results/check4`, then `kubectl logs <pod> -n aria-world > results/check4/<pod>.log` (the pod, not `job/`), and `kubectl get job stacking-encoders-baseline-check4-train -n aria-world -o yaml > results/check4/train-job-status.yaml`. The same log is on the PVC (`/mnt/proxy/logs/baseline-check4-none-s0-try<TRY>-<date>.log`, GPU samples `...-gpu-<date>.csv`). If the pod hit its own deadline the EXIT trap did not run: the log ends where the kill came and `train.json`/`model.pt` may be missing.

11. **Extract the four small files** (`config.json`, `load.json`, `train.json`, `history.jsonl`) from the saved log with the loop of `RUN-sanity.md` step 10, changing the file list and the output names (`results/check4/train-<name>`): they sit between `=== FILE <name> sha256 ... bytes ...` and `=== END <name>`; the sha256 and byte count of each extract must equal its `=== FILE` line. Note the `=== MODEL <sha256> .../model.pt` line: the sha256 of the 6.7 GB `model.pt` on the PVC, for the record.
    **What to log in this experiment's `LOG.md`** (the lead's entry): the card name from `nvidia-smi` and the pod's node; the PVC free space printed at start; each `[ckpt] step <n> written ... in <s>s` time (and the size from `ls -la`: `ckpt.pt`, `model.pt`) and the effective MB/s; the `[projection] after 20 steps` and `[projection] with checkpoint writes` lines. **Do not take the second line's full-run total as the projection:** `project_with_writes` counts writes at THIS run's `--ckpt-every` (289 writes at 75 over 21,696 steps), not the full run's. Take the measured write time from the `[ckpt]` lines, set the full run's K by the PLAN rule (the multiple of 100 optimizer steps closest to 45 minutes of the card's measured step time, kept between 30 and 60 minutes, so 700 at 3.907 s/step; PLAN "Parameters set from measurements"), and recompute: full-run time = 21,696 x s/step + (number of writes at that K) x measured write time; Open item 4's limit is about twice that; s/step over the 300 steps from `history.jsonl` and peak `peak_reserved_bytes`; `data_wait_s`; `[gpu-summary]` per phase and `all`; the `[cgroup] memory.peak` line against the 24Gi request (to size the full run; the PR #39 note that a resume needs host RAM for a 20 GB checkpoint is separate); the exit code; estimated vs actual pod time and GPU-hours (estimates above); the loss at step 300 against step 0 (not an outcome; the LR is still in warmup).

12. **Delete the train Job** once the log is saved and the files extracted: `kubectl delete job stacking-encoders-baseline-check4-train -n aria-world` (graceful, never `--force`); recheck `kubectl get pods -n aria-world -l job-name=stacking-encoders-baseline-check4-train` shows nothing left. **Do not delete anything on the PVC.**

## The smoke Job (only after the train Job succeeded, and with the user's separate go)

13. **Preconditions:** the train Job exited 0 and `model.pt` exists (the smoke pod aborts with exit 15 if not; its own `ls` line shows the size). `stacking-encoders-baseline-eval-src` exists and its keys match step 3.2. The L4 queue may be long: the same selector (L4, driver > 569, us-west) matched only two nodes on 2026-11-09 and that Job never started (`experiments/2026-11-06-perturbed-frames/LOG.md` line 438); the Job deadline (22 h) allows for it, and widening the card list would break the L4-only decision, so say so to the user at the go. State to the user: one L4, about 25 to 55 min of pod time (speculative), about 0.4 to 0.9 GPU-hours, timing-only (no outcomes are computed), 40 rollouts, a test-task use that is logged as such.

14. **Environments per pod:** the Job runs one pass at `ENVS` `"4"`, the proxy's value (`experiments/2026-10-27-proxy-calibration/jobs/rendered/job-eval-dinov3-s1-l4.yaml` line 94). The PLAN raises environments until the GPU is at or above 40% over 10-minute windows (PLAN "Parameters set from measurements"). **If the smoke's GPU is under 40% over 10-minute windows, the lead brings it to the user for a separate smoke Job at more environments** (a new `SMOKE_TRY`, a higher `ENVS`, the user's own go; it costs another restore and model load, about 15 to 20 min of mostly idle pod). The pod has 4 CPUs (the proxy's), so more than 4 environments may not raise GPU use; raising CPUs is a further manifest change. A pass that finishes in under 10 minutes yields no 600-s window; the pod's own 30-s nvidia-smi samples (`[gpu-summary] rollout-envs4`) cover it.

15. **Render, dry run, submit:**
    ```
    sed "s/{name: SRC_GIT_COMMIT, value: FILL_AT_SUBMIT}/{name: SRC_GIT_COMMIT, value: $C}/" jobs/job-check4-smoke.yaml > jobs/rendered/job-check4-smoke.yaml
    grep -c "value: $C}" jobs/rendered/job-check4-smoke.yaml    # must print 1
    grep -c FILL_AT_SUBMIT jobs/rendered/job-check4-smoke.yaml   # must print 0
    kubectl create --dry-run=server -f jobs/rendered/job-check4-smoke.yaml
    kubectl create -f jobs/rendered/job-check4-smoke.yaml
    ```
    `TRAIN_TRY` must equal the train Job's `TRY`; `SMOKE_TRY` is `"1"` for the first submit (the pod aborts with exit 11 if its records file exists, so a rerun needs `"2"`). Check the GPU count (step 6). Wait with the step 8 loop with `J=stacking-encoders-baseline-check4-smoke` and `END=$(( $(date +%s) + 79700 ))`.

16. **The 15-minute check** (same grep with `-l job-name=stacking-encoders-baseline-check4-smoke`, pattern `'^=== PHASE|^\[gpu\]|^\[rollout|TIMING restore|resources:'`): `restore` phase (GPU 0%, up to about 9 min: `TIMING restore: ...`), then `rollout-envs4`: the `checkpoint ...` line (its `[rollout <s>s]` timestamp is the model-load time), the `progress:` lines (episodes/h, mean batch, policy busy), `episode (...): <steps> steps, wall <s>s` per episode, and every 600 s a `resources:` line (GPU util mean/max, pod CPU cores of 4, cgroup memory now/peak). `device=cpu`, a CUDA out-of-memory, a CUDA-graph capture error, `ERROR in`: see the exit-code table; tell the user.

17. **After it ends:** **first confirm each `.errors.jsonl` is absent or empty** (the log's `=== FILE` list shows whether one exists; a non-empty one means episodes raised, and whether raising correlates with success is not excluded: robosuite's own done handling past a success was recalled, not read, PR #46 review correction) **before reading any timing**. Then save the log (`kubectl logs <pod> -n aria-world > results/check4/<pod>.log`, and the Job status yaml), extract the records (`smoke-try<N>.jsonl`, plus any `.errors.jsonl`; they are printed between `=== FILE` markers; check sha256 and byte counts as in step 11), delete the Job as in step 12. **What to log in the LOG:** per-episode wall times (`wall_s` in the JSONL: the fixed step count per suite means these are upper bounds on a real episode, PLAN "Evaluation time"), their mean and spread per suite (libero_10 runs 730 steps, the other three 430); episodes per hour from the last `finished:` line; the model-load time; the `resources:` lines (GPU use per 10-minute window, pod CPU, memory) and `[gpu-summary]` for each rollout phase; the cgroup memory peak against the 28Gi request; GPU memory from the `mem max` figure; the L4 verdict (model fits? GPU at or above 40%?) against the PLAN rule; the projected wall time of 2,000 rollouts at that speed, **as an upper bound** and the shard count, environments per pod and deadline derived from it (logged before submitting the evaluation, PLAN "Evaluation time"). If the GPU stays under 40% after raising environments, or the model does not fit on an L4, that is a stop for the user (PLAN "Parameters set from measurements"; proxy RUN.md stop 10). The records hold no outcome field by construction (`rollout.py` `record`, line 551); do not compute one.

## Exit codes (pod log, last line `=== exit <rc>`)

Train Job:

| rc | Meaning | What to do |
|---|---|---|
| 0 | 300 steps, checkpoints, `model.pt` written, sha256 printed | step 11 |
| 2 | non-finite loss or grad norm (`STOP:`; `train.json` holds `"stopped"`) | sanity fail; stop, tell the user |
| 1, CUDA out of memory | `torch.OutOfMemoryError` in the train phase | PLAN micro-batch rule: after the user's OK, re-render with `MICRO_BATCH` `"4"` and the next `TRY`; if 4 also fails, stop and ask |
| 1, pip | `setup_env.sh` failed (last 50 lines printed) | diagnose; resubmit with the next `TRY` after the user's OK |
| 1, other | Python error (a failed `torch.save`, for example "No space left", "Disk quota exceeded") | read the traceback; a full PVC is a stop for the user; partial files stay (`ckpt.pt.tmp` too) until the user OKs deleting them |
| 10 | `SRC_GIT_COMMIT` not a 40-hex commit | re-render (step 5) |
| 11 | the out folder for this `TRY` exists | re-render with the next `TRY` |
| 12 / 13 | Wan staging failed / staged files do not match the sha256 | stop; tell the user (as RUN-sanity.md) |
| 14 | the PVC has under 60 GiB free | stop; tell the user (nothing may be deleted without their OK) |
| 124 | a step hit its `timeout` (pip 1800 s, a staging `dd` 900 s, training 7200 s, model sha256 900 s); the last `=== PHASE` line says which | most likely CephFS writes slower than the allowance; read the `[ckpt]` lines, tell the user before a resubmit |
| pod `DeadlineExceeded` | the 9000 s pod deadline fired; no `=== exit` line | as 124; read the PVC log (it keeps what was written) |
| 137 / `OOMKilled` | host memory over the 28Gi limit (a checkpoint write is the likely moment) | the `[cgroup] memory.peak` line may be missing; raise the request from the evidence and ask before resubmitting |
| Job `DeadlineExceeded` | 22.5 h from creation passed (almost surely in the queue) | ask the user whether to resubmit |

Smoke Job:

| rc | Meaning | What to do |
|---|---|---|
| 0 | all 40 records written | step 17 |
| 1, restore | `no env bundle ...` or an apt/unzip error in the restore | the three setup scripts in the ConfigMap differ from the prebuilt bundle's: compare with step 3.2 |
| 1, CUDA out of memory / graph capture | the model or its CUDA graphs do not fit an L4, or capture failed | an L4 verdict for the user (PLAN: leaving L4 is the user's call); a capture failure alone: re-render with `CUDA_GRAPH` `"0"` after the user's OK |
| 1, `SystemExit: ... episodes raised` | `.errors.jsonl` holds tracebacks (a simulator crash) | read them (printed in the log); tell the user |
| 10 / 11 / 15 | commit not 40-hex / records file exists (new `SMOKE_TRY`) / `model.pt` missing (check `TRAIN_TRY`) | fix and re-render |
| 124 | a rollout pass hit 5400 s | records so far are in the file; this is a time-per-episode finding (far slower than estimated): tell the user |
| pod `DeadlineExceeded` / 137 / Job `DeadlineExceeded` | as above (7800 s pod, 33Gi memory, 22 h Job) | as the train table |
