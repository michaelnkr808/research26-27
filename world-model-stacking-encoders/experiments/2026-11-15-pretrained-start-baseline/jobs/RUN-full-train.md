# Full stage 1 baseline training on the cluster: submit steps for the lead

Written 2026-11-16 by a worker, Mac only. Nothing here has been submitted; the lead's server dry run of the manifest (podFailurePolicy included) passed 2026-11-16 and is repeated on the rendered file (step 5). Revised after the PR #47 review. One Job, `jobs/job-full-train.yaml` (Job `stacking-encoders-baseline-full-train`): config none, seed 0, all 21,696 optimizer steps, micro-batch 8, one L40 or L40S (whichever schedules first, the user 2026-11-16), output `/mnt/proxy/runs/baseline-none-s0` on the PVC. The manifest header has every number's source and the deadline arithmetic. Paths are relative to this experiment folder. `$C` is the source commit. Format follows `RUN-check4.md`.

**The submit needs the user's go in the lead's window** (an approval of these files is not a go). State before asking: the Job, the cards, about 24 h (L40S) to 31 h (L40) of pod time and the same in GPU-hours (LOG 2026-11-16 projection), the caps (62 h of training per pod, 96 h for the Job including queue and resumes), one GPU, and that nothing on the PVC is deleted.

## What differs from check 4, in short
- `src/pretrained_train.py` has a step watchdog (`--step-watchdog-s`, default 1800): if no optimizer step finishes in 30 min after the first one, the process logs `[watchdog] ...` and exits 17, which is retried (resume costs at most one checkpoint interval). So **the ConfigMap `stacking-encoders-baseline-src` must be updated from the merge commit `$C` before submit** (step 3).
- The pod checks the 7 ConfigMap keys against hashes rendered into the manifest (`sha256sum -c`, exit 19 on mismatch) before doing anything, so a ConfigMap edited during the run cannot silently change the code of a retried pod. **Every resubmit of this run must use the same `$C`.**
- A pod lost to a node reboot is retried (`backoffLimit` 2) and **resumes from `ckpt.pt` by itself**. A folder with files but no `ckpt.pt` is renamed to `baseline-none-s0.stale-<date>` and the run starts fresh (never deleted). The pod prints `MODE: fresh | resume | complete`.
- `--ckpt-every` is chosen in the pod: L40S 700, L40 500 (printed as `CARD ... -> --ckpt-every`). It is not part of the run identity, so a resume on the other card is allowed.
- `kubectl logs` shows only every 100th step line (the pod log would otherwise pass the kubelet's rotation size). The complete log is on the PVC: `/mnt/proxy/logs/baseline-none-s0-<date>.log` (one file per pod). The exit trap prints sha256, size and last 3 lines of `train.json` and `history.jsonl`; the full files stay in the out folder.

## Before submitting

1. **The user's go**, after the statement above.
2. **Source commit** `$C` (on a clean, pulled `main`): `git rev-parse HEAD`.
3. **Update ConfigMap `stacking-encoders-baseline-src`** (7 keys, exists) from `$C`: the watchdog changed `pretrained_train.py`, so the existing keys are stale. Replace the keys from `$C`'s files by your usual method (create from files with `--dry-run=server`, then apply), then read each key back and compare with `$C`'s file:
   ```
   for f in pretrained_train.py pretrained_sizing.py action_expert_init.py proxy_wam.py diagnostics.py t5_cache.py setup_env.sh; do
     a=$(kubectl get configmap stacking-encoders-baseline-src -n aria-world -o jsonpath="{.data.${f//./\\.}}" | shasum -a 256 | cut -c1-64)
     b=$(git show $C:world-model-stacking-encoders/src/$f | shasum -a 256 | cut -c1-64)
     [ "$a" = "$b" ] && echo "ok $f" || echo "DIFF $f"
   done
   ```
   Any `DIFF`: stop and ask. Do not edit this ConfigMap again while the Job exists (a retried pod would exit 19).
4. **Render** into `jobs/rendered/` the commit and the 7 key hashes (computed from `$C`, not from the cluster):
   ```
   mkdir -p jobs/rendered; Y=jobs/rendered/job-full-train.yaml
   sed "s/{name: SRC_GIT_COMMIT, value: FILL_AT_SUBMIT}/{name: SRC_GIT_COMMIT, value: $C}/" jobs/job-full-train.yaml > $Y
   for f in pretrained_train.py pretrained_sizing.py action_expert_init.py proxy_wam.py diagnostics.py t5_cache.py setup_env.sh; do
     h=$(git show $C:world-model-stacking-encoders/src/$f | shasum -a 256 | cut -c1-64)
     sed "s/FILL_SHA_$f /$h /" $Y > $Y.tmp && mv $Y.tmp $Y
   done
   grep -c "value: $C}" $Y        # must print 1
   grep -c FILL_ $Y               # must print 0
   grep -cE '^ +[0-9a-f]{64}  ' $Y   # must print 11 (7 /src keys + 4 Wan files)
   ```
   Never edit the original for one run; commit the rendered file as submitted.
5. **Server dry run, then create**: `kubectl create --dry-run=server -f jobs/rendered/job-full-train.yaml`, then `kubectl create -f jobs/rendered/job-full-train.yaml`. Note the creation time in the LOG; the 96 h Job cap counts from it. Check the GPU count first (ours plus 1 at most 8, `STANDARDS.md` line 49). **Before any manual resubmit, make sure no pod of the old Job remains** (`kubectl get pods -n aria-world -l job-name=stacking-encoders-baseline-full-train`): two pods on one `--out` would interleave `ckpt.pt.tmp` and `history.jsonl`. A pod stuck in Terminating on a lost node: wait (never force-delete, `_standards/08`). A resubmit resumes the same folder and must use the same `$C`.
6. **One wait, in the background**: a bash 3.2 loop, one check every 5 minutes, that ends **only on the Job's `Complete` or `Failed` condition** (a pod that failed and is being retried sets `.status.failed` while the Job is still alive, so `failed` is printed as information only):
   ```
   bash -c 'J=stacking-encoders-baseline-full-train; END=$(( $(date +%s) + 346000 ))
   while :; do
     S=$(perl -e "alarm 120; exec @ARGV" kubectl get job "$J" -n aria-world --request-timeout=60s -o jsonpath="{.status.conditions[?(@.type==\"Complete\")].status}|{.status.conditions[?(@.type==\"Failed\")].reason}|{.status.failed}|{.status.active}" 2>&1) || S="kubectl-error: $S"
     echo "$(date -u +%H:%M:%S) complete|failed-reason|pods-failed|active = $S"
     case "$S" in
       kubectl-error*) echo "END: kubectl failed (login expired or API unreachable): stop, ask the user to re-login, then rerun this loop"; exit 3 ;;
       *) IFS="|" read -r OK WHY NF ACT <<< "$S"
          if [ "$OK" = True ]; then echo "END: Job succeeded"; exit 0; fi
          if [ -n "$WHY" ]; then echo "END: Job failed $WHY"; exit 1; fi ;;
     esac
     [ $(date +%s) -lt $END ] || { echo "END: wait timed out (346000 s)"; exit 2; }
     sleep 300
   done'
   ```
   `pods-failed` above 0 with `active` 1 means a retry is running: read that pod's log, do not delete the Job. A `kubectl-error` line: the OIDC login expired, ask the user to re-login and restart the loop. The loop runs on the Mac (its `sleep` is not a Job sleep). A wait this long outlasts a session: write the handoff note first. The loop sees only Job status, so a hung pod shows up there only through the pod's own watchdog (exit 17) and the retry it causes.

## First checks

7. **15 minutes after the pod is Running** (`STANDARDS.md` line 48): `kubectl logs <pod> -n aria-world | grep -E '^CARD|^MODE|^STALE|^PVC free|^=== PHASE|^\[gpu\]|^\[train|\[ckpt\]|\[projection\]|\[watchdog\]|resuming|: OK$|FAILED' | tail -40` (get `<pod>` from `kubectl get pods -n aria-world -l job-name=stacking-encoders-baseline-full-train -o name`). Expect the card line and its `--ckpt-every`, `MODE: fresh`, a PVC line, phases `setup` and `stage-wan` at 0% for about 2 min, then `train` at 90 to 100%. `device=cpu`, a CUDA error or `STOP:`: delete the Job and tell the user. Then check the Violations page (`_standards/08`).
8. **First-minutes projection** (the `[projection] after 20 steps:` line, about 2.5 min into training): it prints `full run of 21696 steps = <h> h` (compute only). Add about 0.2 h for checkpoint writes. **If that total is over 62 h, stop and ask the user before anything else** (the training step would be killed by its 62 h `timeout`, exit 124, with no `model.pt`). Expected: about 24 h (L40S) or 30 h (L40). Also read the first `[ckpt] step <n> written ... in <s>s` line (about 21 s in check 4).

## What each exit code means (pod log last line `=== exit <rc>`; the Job's `podFailurePolicy` applies)

| rc | Meaning | Job result | What to do |
|---|---|---|---|
| 0 | all steps done, `model.pt` and its sha256 printed | Complete | step 9 |
| 2 | non-finite loss or grad norm (`STOP:`) | fails at once | stop; tell the user; `ckpt.pt` is kept; do not just resubmit (a resume repeats the same draws) |
| 10 | commit not 40-hex | fails at once | re-render |
| 13 | staged Wan files differ from the recorded sha256 | fails at once | stop; tell the user |
| 19 | a `/src` ConfigMap key differs from the hash rendered into the manifest | fails at once | the ConfigMap was edited after the render, or the render did not run: re-read it (step 3), re-render, resubmit with the same `$C` |
| 14 | PVC under 70 GiB free (fresh) or 40 GiB (resume) | fails at once | stop; tell the user; nothing may be deleted without their OK |
| 15 | `nvidia-smi` worked but the card is not an L40 or L40S | fails at once | tell the user |
| 18 | the training step hit its 62 h `timeout` (124, or 137 after the kill grace) | fails at once | the cap was used: tell the user; `ckpt.pt` is kept |
| 17 | the step watchdog fired (`[watchdog]` line: no optimizer step in 30 min, a hung loader or CephFS) | retried | the next pod resumes; if it fires again at the same step, tell the user |
| 16 | a setup step timed out (pip 1800 s, a staging `dd` 900 s, a sha256 check, the `model.pt` hash) | retried | nothing; a retry resumes. If all 3 pods hit it, tell the user |
| 1 | pip failed, `nvidia-smi` failed (a faulty GPU node), or a Python error (CUDA out of memory, a failed `torch.save`, "No space left", `... was written by a different run`) | counts toward the 2 retries | a retry resumes from `ckpt.pt`; the same traceback in two pods: stop and ask the user. A CUDA out-of-memory: PLAN micro-batch rule, `MICRO_BATCH` 4 only with the user's OK |
| 12 | Wan staging failed | counts toward the retries | as above |
| 137 / `OOMKilled` | host memory over 38Gi (this pod prints no `[cgroup]` or `=== exit` lines) | counts toward the retries | read the PVC log `/mnt/proxy/logs/baseline-none-s0-<date>.log` and earlier pods' `[cgroup]` lines; a resume loads about 20 GB into RAM; raise the request from the evidence, ask before resubmitting |
| node reboot, drain, preemption (`DisruptionTarget`) | pod lost | retried, does NOT count toward the 2 | nothing; check that the next pod shows `MODE: resume` |
| pod `DeadlineExceeded` (64 h) | the pod cap fired before the inner timeout; no exit line | counts toward the retries | read the PVC log |

A `model.pt` written but `train.json` missing (pod lost between the two writes): the next pod shows `MODE: resume`, retrains up to one interval (about 45 min) and rewrites `model.pt` the same way. Harmless.
| Job `DeadlineExceeded` (96 h from creation) | queue plus runs passed the cap | fails | ask the user whether to resubmit (it resumes) |

## How a resume shows up in the log
The new pod prints `MODE: resume`, a `ls -la` of the out folder (`ckpt.pt` size and time), then in the training output `resuming from /mnt/proxy/runs/baseline-none-s0/ckpt.pt at step <N>` (and `[resume] MICRO-BATCH CHANGED` only if the micro-batch differs, which it should not), and the step lines continue from `<N>`. Each resume also writes `config-resume-step<N>.json` in the out folder (printed by the exit trap) and appends to `history.jsonl` (a step re-run after a resume appears twice; the script keeps the last record per step in `train.json`). A pod that died before its first checkpoint instead prints `STALE: ... moving it to ...stale-<date>` and `MODE: fresh`. Each pod writes its own log file on the PVC. Log each resume in the LOG with the step and card.

## After the Job ends

9. **Save the logs right away** (container logs vanish within about a day): for every pod of the Job (`kubectl get pods -n aria-world -l job-name=stacking-encoders-baseline-full-train`), `kubectl logs <pod> -n aria-world > results/full-train/<pod>.log` (the pod, not `job/`), and `kubectl get job stacking-encoders-baseline-full-train -n aria-world -o yaml > results/full-train/job-status.yaml`. The pod log has the step lines only every 100th; the complete logs, `history.jsonl` and `train.json` are on the PVC (`/mnt/proxy/logs/baseline-none-s0-<date>.log`, `/mnt/proxy/runs/baseline-none-s0/`); copy them with a small CPU pod after asking the user (memory: external copies and new cluster work need their OK in this window), and check each copy against the sha256 and byte count the exit trap printed (`=== BIGFILE`). Log the `config.json` and `load.json` extracts as in `RUN-check4.md` step 11.
10. **What to log**: Job creation time, card and node of each pod, each `--ckpt-every`, each resume (step, card), mean s/step and `[gpu-summary]` per phase per pod, the write times, the `[cgroup] memory.peak` and anon lines of every pod against the 32Gi request (this sizes any later run and settles whether the portal's memory check is a concern), the exit codes, the `=== MODEL` sha256 of `model.pt`, total pod time and GPU-hours.
11. **Delete the Job** after the logs are saved: `kubectl delete job stacking-encoders-baseline-full-train -n aria-world` (graceful, never `--force`); `kubectl get pods -n aria-world -l job-name=stacking-encoders-baseline-full-train` must show nothing. **Delete nothing on the PVC.** `ckpt.pt` and `model.pt` stay until the user OKs deleting them (PLAN "Checkpointing and resume", "At the end"; the check-4 folder `baseline-check4-none-s0-try1` stays too).
