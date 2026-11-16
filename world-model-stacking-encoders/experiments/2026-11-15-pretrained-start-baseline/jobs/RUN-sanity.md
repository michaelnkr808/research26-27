# Sanity checks 1-3 on the cluster: submit steps for the lead

Written 2026-11-15 by a worker, Mac only. Nothing here has been submitted or dry-run, and no kubectl command was run to write it. The Job is `jobs/job-sanity.yaml` (Job `stacking-encoders-baseline-sanity`); its header has what it runs, every resource number's source and the deadline arithmetic. Paths are relative to this experiment folder unless they start with `src/` or `world-model-stacking-encoders/`. `<scratch>` is the lead's session scratchpad. `$C` is the source commit named in step 2.

**This Job is PLAN "Order of work" step 3. Submitting it starts the experiment, so the design locks** (PLAN header). Submit only with the user's go in the lead's window. It needs 1 GPU (L40 or L40S), about 0.5 h of pod time (speculative), and writes only small files to the PVC (step 13; `model.pt` stays on the pod's local disk, PR #40 review). After the report, the first phase runs the first 20 optimizer steps of the real run (`--max-steps 20`, lead's decision 2026-11-15); no test tasks are used.

## Before submitting

1. **Record the overfit pass mark in this folder's `LOG.md`** before the Job is created (PLAN sanity check 1: "Pass mark set by the implementation session before running it"). Proposed mark and its source are under "Pass and fail" below; the lead confirms or changes it, then writes it in the LOG, then submits.

2. **Name the source commit.** After this branch's PR is merged and pulled, on a clean `main`: `git rev-parse HEAD` gives `$C`. The pod reads only the ConfigMap, never git, so the ConfigMap must hold `$C`'s files exactly.

3. **Create the ConfigMap `stacking-encoders-baseline-src`** (lead's decision 2026-11-15: a separate ConfigMap holding only what this Job needs; `stacking-encoders-proxy-src` is not touched). Why not update `stacking-encoders-proxy-src`: other Jobs mount it at `/src` (the rollout Jobs import `proxy_wam.py` and `diagnostics.py` from it; the perturbed-frames Jobs check `extract_features.py` and the env-bundle scripts in it, `experiments/2026-11-06-perturbed-frames/jobs/PHASE-B.md` line 35), running pods see a replaced ConfigMap's files within about a minute (recalled), and main's `proxy_wam.py` (`e09543d5...` at `6875f53`) is not the file RETRAIN.md recorded (`7169332991...` at `cb9e329`), so replacing it would change other Jobs' code.
   The Job needs 7 keys, the import chain of `src/pretrained_train.py` (checked by grep: it imports `action_expert_init`, `pretrained_sizing`, `proxy_wam`, `diagnostics`; `proxy_wam` imports `t5_cache`) plus `setup_env.sh` for the pip environment: `pretrained_train.py`, `pretrained_sizing.py`, `action_expert_init.py`, `proxy_wam.py`, `diagnostics.py`, `t5_cache.py`, `setup_env.sh`.
   1. Write the 7 files from `$C` into an empty dir: for each `f`, `git show $C:world-model-stacking-encoders/src/$f > <scratch>/baseline-src/$f` (from the repo root).
   2. Check them against main: `shasum -a 256 <scratch>/baseline-src/*` must equal `git show origin/main:world-model-stacking-encoders/src/$f | shasum -a 256` for each (equal when `$C` is main's head; a difference means `src/` moved since `$C`, so stop and name the commit again). For reference, main at `6875f53` (verified on the Mac, `shasum -a 256`; recompute at `$C`):
      - `pretrained_train.py` `e79ad083f8ac02af236e78de74c5b85af7271f0c86356d59086198973cfe6473`
      - `pretrained_sizing.py` `c14ccec63e66075c561573c315ed11a0abf3ce5e0fabd3751dbb599f64a52bb7`
      - `action_expert_init.py` `32e325315b766d601ac0fe60cb79e1c3cbf60caee96cd460682bf63c2431c3cf`
      - `proxy_wam.py` `e09543d5bf0bf8ce2a7d1db9048f8c7819366f97fb4ae0c552e802412e32fd0a`
      - `diagnostics.py` `7af1f7aa2d595d9e0b7f1fb462f7901c6fa9594d2da7895b0168f0ef3ea2d70d`
      - `t5_cache.py` `61181a74af8825eff0e0b8cb3a220f8483d9f1982642cd90b753cb9e5a87d708`
      - `setup_env.sh` `880c47cbe741b7fbcd98f44f85e7a1d14747d6c837e0493f4642e5ebbc18ace8`
   3. Check the name is free: `kubectl get configmap stacking-encoders-baseline-src -n aria-world` should say NotFound. If it exists, read its keys and hashes and ask before replacing it.
   4. Dry run, then create: `kubectl create configmap stacking-encoders-baseline-src -n aria-world --request-timeout=30s --from-file=<scratch>/baseline-src --dry-run=server`, then the same without `--dry-run=server`. Size: the 7 files are 141,418 bytes at `6875f53` (far under the 1 MiB limit).
   5. Read back each key and compare with step 2: `kubectl get configmap stacking-encoders-baseline-src -n aria-world -o jsonpath='{.data.pretrained_train\.py}' | shasum -a 256` (escape the dot as `\.`), for all 7. All must match. The pod also prints the sha256 of the 7 `/src` files right after `nvidia-smi`, so the record is in the log.
   6. Record in the LOG that the ConfigMap holds `$C`'s files. Keep it after the Job is deleted until sanity check 4's Job is prepared (it needs the same keys); deleting a ConfigMap deletes no data.

4. **Render** with the commit filled, replacing only the env value line (PR #40 review item 1: a bare `s/FILL_AT_SUBMIT/.../` would also hit any other occurrence; the guard now checks for a 40-hex sha and holds no placeholder text, so it survives either way):
   ```
   mkdir -p jobs/rendered
   sed "s/{name: SRC_GIT_COMMIT, value: FILL_AT_SUBMIT}/{name: SRC_GIT_COMMIT, value: $C}/" jobs/job-sanity.yaml > jobs/rendered/job-sanity.yaml
   grep -c "value: $C}" jobs/rendered/job-sanity.yaml    # must print 1
   grep -c FILL_AT_SUBMIT jobs/rendered/job-sanity.yaml   # must print 0
   ```
   Never edit the original for one run; commit the rendered file as submitted. `TRY` stays `"1"` for the first submit; for a resubmit after a failed or deleted attempt, render with `TRY` set to `"2"`, `"3"`, ... (the pod aborts with exit 11 if `/mnt/proxy/runs/baseline-sanity-none-s0-try<TRY>` already exists, so an earlier attempt's files are never overwritten). `MICRO_BATCH` stays `"8"` unless the exit-code table says 4.

5. **GPU count:** the GPUs our Jobs hold or are Pending for, plus 1, must be at most 8 (`STANDARDS.md` line 49). The PVC gets only the logs and small result files (well under 100 MB, speculative). The pod's local disk holds the staged Wan files and `model.pt` (ephemeral request 38Gi, `/work` 24Gi).

6. **Server-side dry run, then submit:** `kubectl create --dry-run=server -f jobs/rendered/job-sanity.yaml`, then `kubectl create -f jobs/rendered/job-sanity.yaml`. Note the creation time in the LOG. The deadline (75,600 s, 21 h) counts from now, queue included.

## While it runs

7. **One wait, in the background** (lead's decision 2026-11-15): a plain bash 3.2 loop that reads the Job's status every 5 minutes and stops when `.status.succeeded` or `.status.failed` is set or a `Failed` condition appears (which also covers `DeadlineExceeded`), with an overall timeout. Run it with the Bash tool's background option; it prints one line per check and one final `END:` line:
   ```
   bash -c 'J=stacking-encoders-baseline-sanity; END=$(( $(date +%s) + 76000 ))
   while :; do
     S=$(perl -e "alarm 120; exec @ARGV" kubectl get job "$J" -n aria-world --request-timeout=60s -o jsonpath="{.status.succeeded}|{.status.failed}|{.status.conditions[?(@.type==\"Failed\")].reason}|{.status.active}" 2>&1) || S="kubectl-error: $S"
     echo "$(date -u +%H:%M:%S) succeeded|failed|failed-reason|active = $S"
     case "$S" in
       kubectl-error*) echo "END: kubectl failed (login expired or API unreachable): stop, ask the user to re-login, then rerun this loop"; exit 3 ;;
       *) IFS="|" read -r OK BAD WHY ACT <<< "$S"
          if [ -n "$OK" ] && [ "$OK" != 0 ]; then echo "END: Job succeeded"; exit 0; fi
          if { [ -n "$BAD" ] && [ "$BAD" != 0 ]; } || [ -n "$WHY" ]; then echo "END: Job failed ${WHY:-(pod failed)}"; exit 1; fi ;;
     esac
     [ $(date +%s) -lt $END ] || { echo "END: wait timed out (76000 s)"; exit 2; }
     sleep 300
   done'
   ```
   `END: Job succeeded` = complete; `END: Job failed BackoffLimitExceeded` = the pod failed (exit codes below); `END: Job failed DeadlineExceeded` = out of time (queue or run). A `kubectl-error` line ends the loop (`END: kubectl failed`, exit 3): the OIDC login has probably expired (kubectl then hangs or errors; `--request-timeout=60s` does not cover the login step, so `perl -e "alarm 120; exec @ARGV"` caps each call at 120 s, as macOS has no `timeout`). Then stop, ask the user to re-login, and start the loop again; the Job itself is unaffected. The timeout is the Job deadline (75,600 s from creation) plus about 7 min. The loop runs on the Mac, not in a Job, so its `sleep` is not the Job `sleep` ban. A wait this long outlasts an idle session: if it will run more than about 50 min, write the handoff note first (user's global rule). The loop was tested on the Mac's bash 3.2 with a fake `kubectl` that printed succeeded, failed and deadline statuses and one that failed (no cluster access).

8. **The 15-minute GPU-use check** (`STANDARDS.md` line 48; the lesson "check GPU util 15 min into any new job type"). About 15 min after the pod is Running: `kubectl get pods -n aria-world -l job-name=stacking-encoders-baseline-sanity -o name`, then `kubectl logs <pod> -n aria-world | grep -E '^=== PHASE|^\[gpu\]|^\[train' | tail -40`. What to expect (speculative): the `setup` and `stage-wan` phases at about 0% (pip install and the Wan copy, about 13 to 20 min), then high use in `first-batch` and `overfit`. What counts:
   - Still in `setup` after 30 min of Running: pip is stuck; look at the log, tell the user.
   - In `overfit`, the `[gpu]` samples should be well above 40%; the phase lasts only a few minutes. If they are under 40% there, note it in the LOG (it bears on the full run, which uses the same step code) and tell the user; the pod ends by itself, no action needed.
   - `device=cpu` or a CUDA error in the log: delete the Job (memory note: a GPU fault can make a script fall back to CPU).
   If the pod is still Running at 30 to 40 min, check the Violations page (`_standards/08` pre-flight item 8). The Prometheus alternative is the RETRAIN.md query with `pod=~"stacking-encoders-baseline-sanity-.*"`.

## After it ends

9. **Save the pod log right away** (container logs vanish within about a day; the Job's ttl is 24 h): `mkdir -p results/sanity`, then `kubectl logs <pod> -n aria-world > results/sanity/<pod>.log` (the pod, not `job/`), and `kubectl get job stacking-encoders-baseline-sanity -n aria-world -o yaml > results/sanity/job-status.yaml`. The same log is on the PVC as `/mnt/proxy/logs/baseline-sanity-none-s0-try<TRY>-<date>.log`, the GPU samples as `...-gpu-<date>.csv`, the pip freeze as `/mnt/proxy/logs/pip-env-baseline-sanity-none-s0-try<TRY>.log`. If the pod hit its own deadline (5400 s from start, pod reason `DeadlineExceeded`), the EXIT trap did not run: the pod log and the PVC log end where the kill came, and no result files were copied.

10. **Get the result files** (small, so they travel in the pod log; no reader pod). Both phases write to the pod's local `/work/out`; at exit the trap copies the small files to `/mnt/proxy/runs/baseline-sanity-none-s0-try<TRY>/` on the PVC and prints each of `first-batch/config.json`, `first-batch/load.json`, `first-batch/first-batch.json`, `first-batch/train.json`, `first-batch/history.jsonl`, `overfit/overfit.json` (those that exist) between `=== FILE <name> sha256 <hex> bytes <n>` and `=== END <name>`. Extract each from the saved log and check it:
    ```
    L=results/sanity/<pod>.log
    for f in first-batch/config.json first-batch/load.json first-batch/first-batch.json first-batch/train.json first-batch/history.jsonl overfit/overfit.json; do
      o=results/sanity/$(echo $f | tr / -)
      x=$(awk -v f="$f" '$0 == "=== END " f {p=0} p {print} index($0, "=== FILE " f " ") == 1 {p=1}' $L)
      case $f in *.jsonl) printf '%s\n' "$x" > $o ;; *) printf '%s' "$x" > $o ;; esac
      grep "^=== FILE $f " $L; shasum -a 256 $o; wc -c < $o
    done
    ```
    A file the pod never wrote has no `=== FILE` line and comes out empty; delete those. The sha256 and byte count of each extracted file must equal the `=== FILE` line's (the JSON files are written without a trailing newline, and `$(...)` drops the one the dump adds; `history.jsonl` ends in a newline, which the `*.jsonl` branch puts back; the JSON recipe was tested on the Mac on a fake log in this format and the extract was byte-identical, the `.jsonl` branch was tested the same way in the PR #40 fix commit). If one differs, or the log was cut, ask the user before creating a CPU reader pod to copy the files. `model.pt` was on the pod's local disk only and is gone with the pod (it is the first 20 steps of the real run, not needed).

11. **Judge checks 1 to 3** against "Pass and fail" below and write the outcome in this folder's `LOG.md` (the lead's entry; the JSONs and the log are committed under `results/sanity/`).

12. **Delete the Job** once the log is saved and the JSONs are extracted: `kubectl delete job stacking-encoders-baseline-sanity -n aria-world` (graceful; never `--force`). Recheck `kubectl get pods -n aria-world -l job-name=stacking-encoders-baseline-sanity` shows nothing left.

13. **PVC files left behind:** `/mnt/proxy/runs/baseline-sanity-none-s0-try<TRY>/` holds only the six small files of step 10 (no `model.pt`: it stayed on the pod's disk), plus the logs and GPU CSV under `/mnt/proxy/logs/`. Nothing needs deleting; if the user ever wants them gone, that is data deletion and needs the user's OK (`../../../_standards/05` gates).

## Exit codes (pod log, last line `=== exit <rc>`)

| rc | Meaning | What to do |
|---|---|---|
| 0 | both phases ran | judge the checks |
| 2 | non-finite loss or gradient norm (`STOP:` line; `first-batch.json` (at the report), `train.json` (in the 20 steps) or `overfit.json` holds `"stopped"`) | sanity fail; stop, tell the user |
| 1, CUDA out of memory | traceback ending in `torch.OutOfMemoryError` (or `CUDA out of memory`) in the first-batch or overfit phase | the PLAN's micro-batch rule ("Parameters set from measurements"): after the user's OK, re-render with `MICRO_BATCH` `"4"` and the next `TRY`; if 4 also runs out, stop and ask (the PLAN says so) |
| 1, pip | `setup_env.sh` failed (last 50 pip lines printed) | diagnose; resubmit with the next `TRY` after the user's OK |
| 1, other | any other Python error (traceback) | diagnose; tell the user before a resubmit |
| 10 | `SRC_GIT_COMMIT` is not a 40-hex commit (not filled, or filled wrong) | re-render (step 4) |
| 11 | the PVC sanity folder for this `TRY` exists | re-render with the next `TRY` |
| 12 | Wan staging failed: a `dd` error, a size mismatch, or the index/config copy failed | stop; tell the user (CephFS read problem or a damaged PVC copy) |
| 13 | staged Wan files do not match the sha256, or the check timed out (600 s) | stop; the PVC copy of the Wan files may be damaged; tell the user |
| 124 | a step hit its `timeout` (pip 1800 s, a staging `dd` 900 s, first-batch phase 1800 s, overfit phase 1200 s); the last `=== PHASE` line says which | look at the log around it (a CephFS stall, a hung pip); tell the user before a resubmit |
| pod `DeadlineExceeded` | the pod-level deadline (5400 s from pod start) fired; no `=== exit` line, the EXIT trap did not run | as 124; the Job shows `failed`, its reason `BackoffLimitExceeded` |
| 137 / `OOMKilled` | host memory over the 19Gi limit | raise the memory request from the log's evidence and ask before resubmitting |
| Job `DeadlineExceeded` | 21 h from creation passed (almost surely in the queue) | ask the user whether to resubmit (same `TRY` is fine if no folder was made: check the pod ever started) |

## Pass and fail

Any failed check is a stop: report to the user, no sanity check 4 until the user decides.

**Check 2, loss at start** (PLAN: "Repeated on the full data's first batch and logged; a non-finite loss stops"). Pass: `first-batch.json` exists with `loss_at_start.model` (`total`, `vae`, `action`) and `loss_at_start.predict_zero` all finite. The PLAN sets no numeric bar. Log them next to the sizing run's values on its batch of 16 (model total 3.735, VAE 2.634, action 1.101; predict-zero 2.526, VAE 1.492; sizing LOG line 38). They are not expected to match: this batch is 128 windows of the full data and the action expert is now derived from the Wan blocks, not fresh. A large gap is something to look at and tell the user, not a fail by the PLAN.

**Check 3, inputs inspected** (PLAN: "shapes, ranges, padding counts, task strings"). Pass when all hold:
- The first `[train ...]` line says `1712 episodes, 277713 frames, 40 tasks, in_dim 48; plan 21696 optimizer steps of 128, this run to step 20, micro-batch 8 x 16` (episodes, frames: `data/README.md` section 3; steps: PLAN recipe table; 40 = the paper's 40 tasks, Table 7, assuming the 40 instruction strings are all distinct, which is not checked here: the count is of distinct strings).
- `first-batch.json` `inputs`: `latents_host` shape `[128, 3, 14, 28, 48]` float16; `action` `[128, 32, 7]`; `action_is_pad` `[128, 32]`; `latent_is_pad` `[128, 2]`; `proprio` `[128, 8]`; `task` `[128]` (`proxy_wam.Data.assemble`).
- `action_range` and `proprio_range` inside [-1, 1] and both `*_abs_gt1_fraction` 0 (min/max normalization over the same 1,712 episodes maps every value into [-1, 1], `proxy_wam.MinMax`; anything outside is a normalization error).
- `channels/vae`: finite, `std_range` minimum above 0.
- `tasks`: real LIBERO instructions (read them), `n_distinct_tasks` above 1.
- Padding: `windows_with_padding` below 128 (some windows near episode ends are padded, not all).
- `load.json`: `all_loaded_diffs_zero` true; `counts_by_tag["loaded"]` 231 tensors (as the sizing run's `load-none.json`); the derived and fresh counts as the action-expert recipe says (`counts.derived`, `counts.fresh`).

**Check 1, overfit one batch. Proposed pass mark** (the lead records it in the LOG before submitting; PLAN sanity check 1 says it is set "from the proxy's check 9 (proxy PLAN line 137)"). From `overfit.json`, with `first_20_mean` and `last_20_mean` (means of the first and last 20 of the 300 steps, keys `loss`, `loss/vae`, `loss/action`) and `predict_zero` (keys `total`, `vae`, `action`, the same batch and draws):
1. `last loss/action < 0.1 x first loss/action`. Source: proxy check 9's action criterion, word for word (`src/test_proxy_wam.py`, check 9: "the action loss over the last 20 of 200 steps is below a tenth of the first 20").
2. Every term ends lower than it starts: `last[k] < first[k]` for `loss`, `loss/vae`, `loss/action`. Source: check 9's second criterion ("every other loss term ends lower than it starts").
3. `last loss < predict_zero total`, and `last loss < 0.5 x first loss`. Source: the PLAN's own words for this check ("the training loss falls well below its starting value and below the predict-zero baseline on that batch"); the 0.5 is the proxy's original check 9 criterion (total loss halves), which the proxy dropped only because its synthetic VAE latents were i.i.d. noise that a 0.46M-parameter test model could not memorize (check 9's history, proxy LOG line 500). Neither reason holds here (real latents, a 1.68B-parameter model; speculative that it then halves).
4. No `STOP:` (exit 2).
Pass = all four. A fail on 3 alone with 1 and 2 passing is still a fail under this mark; the lead may decide to weaken 3 before submitting (for example to "below predict-zero" only), but not after seeing the result (`../../../_standards/05`, no silent goalpost moves).

What is not tested here: whether the model learns on the full data (sanity check 4 and the full run), the full run's step time beyond 20 steps (the first phase gives 20 steps on this card: `s_step`, `data_wait_s` and `peak_reserved_bytes` per step in `history.jsonl`, and the `[projection] after 20 steps` log line; the lead logs them with the card name from `nvidia-smi`, and the PLAN's micro-batch rule reads `peak_reserved_bytes` against 0.9 x the card's memory), and rollouts.
