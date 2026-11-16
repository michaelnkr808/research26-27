# Final clean-LIBERO evaluation on the cluster: submit steps for the lead

Written 2026-11-16 by a worker, Mac only. Nothing here has been submitted, dry-run or run with kubectl. One Indexed Job, `jobs/job-clean-eval.yaml` (Job `stacking-encoders-baseline-clean-eval`): 4 pods on 4 L4 GPUs at once, each evaluating one shard of 500 of the 2,000 rollouts (4 suites x 10 tasks x 50 trials, init state = trial number, eval seed 41, real outcomes) of one checkpoint, `/mnt/proxy/runs/baseline-none-s0/model.pt`. The manifest header has every number's source and the deadline arithmetic. Paths are relative to this experiment folder unless they start with `src/` (that is `../../src/`). `$C` is the source commit (step 2). Format follows `RUN-check4.md` and `RUN-full-train.md`.

**The submit needs the user's go in the lead's window** (the user's 2026-11-16 delegation of the experiment covers running stage 1 within `_standards/08`; if the lead's reading of it is in doubt, ask). State before asking: the Job, 4 L4 GPUs at once, about 2.6 h of pod time per shard and about 10.4 GPU-hours in all (upper bound, from the smoke), the caps (pod 5.33 h, Job 48 h including queue), that the L4 queue may be long (the same selector matched two nodes on 2026-11-09), and that this is the evaluation's one test-set use. Nothing on the PVC is deleted by the Job.

**This is the test-set use of PLAN "Test-set use".** No checkpoint, hyperparameter or recipe choice is made from its outcomes. A re-evaluation of the same checkpoint is allowed at most once, only after a pipeline fault shown without test outcomes, with an amendment that says the first result had been seen. **Log the use in `LOG.md` with the date and the checkpoint (path, step 21,696, the `model.pt` sha256) at submit, before any outcome is read** (step 9 has the entry's contents).

## What differs from the smoke, in short
- Real outcomes (no `--timing-only`), `--clean 50`, `--clean-seed 41`, `--order-seed -1`, `--shard i/4`, one new records file per shard under `/mnt/proxy/rollouts/baseline-clean-eval/` (never the smoke's file).
- `--order-seed -1` gives every shard exactly 125 episodes of each suite (computed on the Mac; the default seed 0 would give 112 to 135 per suite per shard). See the manifest header.
- The pod checks the 11 `/src` keys against hashes rendered into the manifest (exit 19), computes `model.pt`'s sha256 and compares it with the value rendered from the training log (exit 13), and prints it.
- `backoffLimit` 6 (shared by the four indexes) with a `podFailurePolicy`: a retried pod resumes from its shard's records file (rollout.py skips keys already in it). **Every resubmit must use the same `$C` and the same `EVAL_TRY`** to resume; a new `EVAL_TRY` starts over and needs a LOG reason.
- At exit each pod prints its records file and any `.errors.jsonl` between `=== FILE ... sha256 ... bytes ... lines ...` and `=== END ...`, then `[errors] ...`.

## Before submitting

1. **The user's go**, after the statement above. Also confirm the full training Job ended `Complete` and its pod log was saved at once (`RUN-full-train.md` step 9), so the `=== MODEL` line is on disk.

2. **Source commit** `$C` (on a clean, pulled `main`): `git rev-parse HEAD`.

3. **ConfigMap read-back.** `stacking-encoders-baseline-eval-src` (11 keys) exists from 2026-11-16 (created from `bee6027`; LOG). Read every key back and compare with `$C`'s file (the keys are `src/` files, unchanged since unless the `git diff` says otherwise):
   ```
   for f in rollout.py lp_eval.py proxy_wam.py t5_cache.py diagnostics.py extract_fastwam.py extract_features.py fastwam_data.py setup_env.sh setup_sim.sh env_bundle.sh; do
     a=$(kubectl get configmap stacking-encoders-baseline-eval-src -n aria-world -o jsonpath="{.data.${f//./\\.}}" | shasum -a 256 | cut -c1-64)
     b=$(git show $C:world-model-stacking-encoders/src/$f | shasum -a 256 | cut -c1-64)
     [ "$a" = "$b" ] && echo "ok $f" || echo "DIFF $f"
   done
   ```
   Any `DIFF`: stop and ask the user (do not edit the ConfigMap without them). The last three keys must also reproduce the env bundle key `9182f8131559` (the pod prints the key it computes: `cat setup_env.sh setup_sim.sh env_bundle.sh | shasum -a 256 | cut -c1-12` gives it on the Mac). Do not edit the ConfigMap while the Job exists (a retried pod would exit 19).

4. **Render.** Take the `model.pt` sha256 from the saved training log (`grep '=== MODEL' results/full-train/<pod>.log`; the line is `=== MODEL <sha256>  /mnt/proxy/runs/baseline-none-s0/model.pt`). Put it in `$M` and check it is 64 hex characters. The hashes come from `$C`, not from the cluster:
   ```
   mkdir -p jobs/rendered; Y=jobs/rendered/job-clean-eval.yaml
   sed -e "s/{name: SRC_GIT_COMMIT, value: FILL_AT_SUBMIT}/{name: SRC_GIT_COMMIT, value: $C}/" -e "s/value: FILL_MODEL_SHA}/value: $M}/" jobs/job-clean-eval.yaml > $Y
   for f in rollout.py lp_eval.py proxy_wam.py t5_cache.py diagnostics.py extract_fastwam.py extract_features.py fastwam_data.py setup_env.sh setup_sim.sh env_bundle.sh; do
     h=$(git show $C:world-model-stacking-encoders/src/$f | shasum -a 256 | cut -c1-64)
     sed "s/FILL_SHA_$f /$h /" $Y > $Y.tmp && mv $Y.tmp $Y
   done
   grep -c "value: $C}" $Y            # must print 1
   grep -c "value: $M}" $Y            # must print 1
   grep -c FILL_ $Y                   # must print 0
   grep -cE '^ +[0-9a-f]{64}  ' $Y    # must print 11
   grep -n -i 'sleep\|toleration\|priorityClass' $Y   # must print nothing
   ```
   Never edit the original for one run; commit the rendered file as submitted.

5. **GPU count:** the GPUs our Jobs hold or are Pending for, plus 4, must be at most 8 (`STANDARDS.md` line 49); the training Job's pod must be gone first. Also list our pods and PVC space: `kubectl get pods -n aria-world`, and the records and logs are small (about 0.2 MB of records per shard), so no space gate is needed.

6. **Server dry run, then create:** `kubectl create --dry-run=server -f jobs/rendered/job-clean-eval.yaml`, then `kubectl create -f jobs/rendered/job-clean-eval.yaml`. Note the creation time in the LOG (the 48 h Job cap counts from it). The dry run is the first server check of `completionMode: Indexed` with `podFailurePolicy` and the `fieldRef` annotation `batch.kubernetes.io/job-completion-index`; if it is rejected, stop and report the message (do not patch the manifest on the fly).

## While it runs

7. **One wait, in the background**, ending only on the Job's `Complete` or `Failed` condition (retried pods raise `.status.failed` while the Job is alive, so `failed` is information only):
   ```
   bash -c 'J=stacking-encoders-baseline-clean-eval; END=$(( $(date +%s) + 173000 ))
   while :; do
     S=$(perl -e "alarm 120; exec @ARGV" kubectl get job "$J" -n aria-world --request-timeout=60s -o jsonpath="{.status.conditions[?(@.type==\"Complete\")].status}|{.status.conditions[?(@.type==\"Failed\")].reason}|{.status.failed}|{.status.active}|{.status.succeeded}" 2>&1) || S="kubectl-error: $S"
     echo "$(date -u +%H:%M:%S) complete|failed-reason|pods-failed|active|succeeded = $S"
     case "$S" in
       kubectl-error*) echo "END: kubectl failed (login expired or API unreachable): stop, ask the user to re-login, then rerun this loop"; exit 3 ;;
       *) IFS="|" read -r OK WHY NF ACT NS <<< "$S"
          if [ "$OK" = True ]; then echo "END: Job succeeded"; exit 0; fi
          if [ -n "$WHY" ]; then echo "END: Job failed $WHY"; exit 1; fi ;;
     esac
     [ $(date +%s) -lt $END ] || { echo "END: wait timed out (173000 s)"; exit 2; }
     sleep 300
   done'
   ```
   `pods-failed` above 0 with pods still `active` means a retry is running: read that pod's log, do not delete the Job. A `kubectl-error` line: the OIDC login expired; ask the user to re-login and restart the loop. The loop runs on the Mac (its `sleep` is not a Job sleep). If the wait will pass about 50 min, write the handoff note first. Never poll in a faster loop.

8. **The 15-minute GPU check** (`STANDARDS.md` line 48), about 15 min after each pod is Running: `kubectl get pods -n aria-world -l job-name=stacking-encoders-baseline-clean-eval -o name`, then for each pod `kubectl logs <pod> -n aria-world | grep -E '^=== PHASE|^MODE|^=== MODEL|equals the pinned|TIMING restore|^\[gpu\]|resources:|progress:' | tail -30`. Expect: `MODE: fresh`, `=== MODEL <sha256> ...` and `model.pt sha256 equals the pinned value`, phase `restore` at 0% for up to about 10 min, then `rollout-envs4` at about 90% with `progress:` lines near the smoke's 215 episodes per hour (a shard is a quarter of each suite, so the same mix). What counts:
   - `device=cpu`, a CUDA error or out-of-memory, a CUDA-graph capture error, `ERROR in`: delete the Job (never force) and tell the user.
   - Exit 13 (`ABORT: model.pt sha256 ... differs`): the Job fails by itself; tell the user, do not re-render with a different hash without understanding why the file differs from the training log's.
   - GPU under 40% over 10-minute windows once rollouts run: tell the user (leaving L4 or changing environments is their call, PLAN Open item 11).
   - Check the Violations page (`_standards/08` pre-flight item 8) once if a pod is still Running at 30 to 40 min.

## After it ends

9. **Save every pod log at once** (container logs vanish within about a day), the pod, not `job/`, including pods of failed attempts:
   ```
   mkdir -p results/clean-eval
   for p in $(kubectl get pods -n aria-world -l job-name=stacking-encoders-baseline-clean-eval -o name); do kubectl logs $p -n aria-world > results/clean-eval/${p#pod/}.log; done
   kubectl get job stacking-encoders-baseline-clean-eval -n aria-world -o yaml > results/clean-eval/job-status.yaml
   kubectl get pods -n aria-world -l job-name=stacking-encoders-baseline-clean-eval -o wide > results/clean-eval/pods.txt
   ```
   The same logs are on the PVC (`/mnt/proxy/logs/baseline-clean-eval-shard<i>of4-try1-<date>.log`, GPU samples `...-gpu-<date>.csv`). If a pod hit its own deadline the EXIT trap did not run: its log ends where the kill came; the records are on the PVC.

10. **Check the checkpoint.** Every pod log's `=== MODEL <sha256>` must equal the training log's: `grep -h '=== MODEL' results/full-train/*.log results/clean-eval/*.log | awk '{print $3}' | sort | uniq -c` shows one hash (the training line's own file path differs; compare the hash column only).

11. **Errors files first, before reading any outcome.** For each saved log: `grep -n '^\[errors\]\|^=== FILE\|^=== exit' results/clean-eval/<pod>.log`. A shard whose last pod exited 0 has `[errors] no .errors.jsonl`. If any pod printed `[errors] ... EXISTS`, extract that errors file (below), read it, and tell the user before computing anything: episodes that raised are not in the records, and whether raising correlates with success is not excluded (`RUN-check4.md` step 17). The pod exits 1 in that case, so the Job would not be Complete.

12. **Extract the four records files** from the saved logs of the pods that exited 0 (one per shard). Each file sits between `=== FILE <name> sha256 <h> bytes <n> lines <l>` and `=== END <name>`:
    ```
    mkdir -p results/clean-eval/records
    for i in 0 1 2 3; do
      name=clean-none-s0-shard${i}of4-try1.jsonl
      log=$(grep -l "^=== FILE $name sha256" $(grep -l '^=== exit 0$' results/clean-eval/*.log) | tail -1)
      awk -v n="$name" 'index($0, "=== FILE " n " ")==1 {on=1; next} $0=="=== END " n {on=0} on' $log > results/clean-eval/records/$name
      echo "$name from $log: $(shasum -a 256 < results/clean-eval/records/$name | cut -c1-64) $(wc -c < results/clean-eval/records/$name) bytes $(wc -l < results/clean-eval/records/$name) lines"
      grep "^=== FILE $name " $log
    done
    ```
    The sha256, byte count and line count of each extract must equal its `=== FILE` line, and each file must have **500 lines** (2,000 in all). If a shard has several pods, use the one that exited 0 (check `=== exit 0` at its end); an earlier failed pod's log holds only a partial file. Also `grep -c '"timing_only": true' results/clean-eval/records/*.jsonl` must print 0 for every file (real outcomes, not timing rows).

13. **Compute the four-suite average and per-suite rates** with the existing script `src/tally_proxy_clean.py` (`tally`, `render`; stdlib only; it keeps only `benchmark: "libero"` rows, groups by config, train seed and checkpoint, then eval seed, and reports successes per suite, the four-suite average as the mean of the per-suite rates, and flags duplicates, uneven counts and missing suites). **Give it the output path as the second argument: its default output file is the proxy's `results/proxy-clean-tally.json`, which a bare run would overwrite.** It reads every `*.jsonl` under the folder, so the errors files (if any) must not be in `records/`:
    ```
    /opt/homebrew/bin/python3.11 -I ../../src/tally_proxy_clean.py results/clean-eval/records results/clean-eval/clean-eval-tally.json
    ```
    Expect one checkpoint line (`none train_seed=0 /mnt/proxy/runs/baseline-none-s0`), `eval_seed=41`, 500 per suite (2,000 in all), no `DUPLICATES`, `UNEVEN COUNTS` or `MISSING` flags; anything else is a pipeline question for the user before any reading. The pass-bar number is its `4-suite avg`. Binomial standard error of the average for the report: sqrt(p(1-p)/2000) from that p (arithmetic, no new code). The script gives per-suite results, not per-task ones (`rollout.py` has no summary of its own); the PLAN also lists per-task success and mean episode length, which the records hold (`episode_id` = task index x 1000 + trial, `steps`) and no existing script tallies: say so to the user and ask before writing a tally (it would be a new script with a test, by PR).

14. **Delete the Job** once the logs are saved and the records extracted: `kubectl delete job stacking-encoders-baseline-clean-eval -n aria-world` (graceful, never `--force`); recheck `kubectl get pods -n aria-world -l job-name=stacking-encoders-baseline-clean-eval` shows nothing left. **Do not delete anything on the PVC** (the shards' records and logs stay).

15. **What to log in `LOG.md`** (the lead's one entry, with the test-set use): date, Job creation time, `$C`, checkpoint (`/mnt/proxy/runs/baseline-none-s0/model.pt`, step 21,696, sha256 equal to the training log's), eval seed 41, 2,000 rollouts in 4 shards of 500 with 125 per suite each, per-pod node and card, restore and load times, episodes per hour from each `finished:` line, `[gpu-summary]` per phase per pod (the 40% rule), `[cgroup]` memory peak and anon lines against the 28Gi request, exit codes and retries, the records files' sha256, then the four-suite average with its standard error and the per-suite rates, and the PLAN decision rule applied word for word (pass at 90 or above, 36 up to 90 conditional with LIBERO-Plus, below 36 fail; one number per rule, report which zone). Actual versus estimated pod time and GPU-hours (about 2.6 h and 10.4 GPU-hours, upper bound).

## Exit codes (pod log, last line `=== exit <rc>`)

| rc | Meaning | Job reaction | What to do |
|---|---|---|---|
| 0 | the shard's 500 records are written and printed | pod Succeeded | steps 9 to 13 |
| 1, restore | `no env bundle ...` or an apt/unzip error in the restore | retried | the three setup scripts in the ConfigMap differ from the prebuilt bundle's: compare with step 3 |
| 1, CUDA out of memory or graph capture | the model or its CUDA graphs do not fit | retried (will repeat) | the smoke fit on an L4 (16.5 of 23 GB); if it repeats, delete the Job and tell the user (leaving L4 is their call) |
| 1, `SystemExit: ... episodes raised` | `.errors.jsonl` holds tracebacks (a simulator crash); those episodes are not in the records | retried; a retry reruns them (resume skips only recorded keys) | step 11 before any outcome; tell the user |
| 1, `holds N records that are not in this shard's jobs` | `check_resume` refused the shard file (other order seed, shard count or settings) | retried until the budget is used | delete the Job; do not touch the PVC; tell the user |
| 1, other | Python error, a simulator worker died, nvidia-smi failed (faulty GPU node) | retried | read the traceback |
| 10 | `SRC_GIT_COMMIT` not 40-hex | fails the Job | re-render (step 4) |
| 13 | `model.pt` sha256 differs from the pinned one, or the pin was not rendered | fails the Job | step 8; tell the user |
| 15 | `model.pt` missing | fails the Job | did the training Job succeed? |
| 16 | the `model.pt` sha256 step (900 s) or the rollout pass (16,800 s cap) timed out | retried; a retry resumes | read the `progress:` lines; if all pods time out, per-episode time is far above the smoke's: tell the user |
| 19 | a `/src` key differs from the rendered hash, or a placeholder is left | fails the Job | step 3 and 4; tell the user |
| 137 / `OOMKilled` | host memory over the 33Gi limit | retried | the `[cgroup]` lines may be missing; raise from evidence and ask before resubmitting |
| pod `DeadlineExceeded` | the 19,200 s pod deadline fired; no `=== exit` line | retried | records so far are on the PVC; read the PVC log |
| Job `DeadlineExceeded` | 48 h from creation passed (queue plus runs) | Job failed | records so far are on the PVC; ask the user whether to resubmit with the same `EVAL_TRY` (it resumes) |
