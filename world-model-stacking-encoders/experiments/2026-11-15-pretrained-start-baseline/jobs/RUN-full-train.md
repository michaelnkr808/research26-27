# Full stage 1 baseline training on the cluster: submit steps for the lead

Written 2026-11-16 by a worker, Mac only. Nothing here has been submitted or dry-run. One Job, `jobs/job-full-train.yaml` (Job `stacking-encoders-baseline-full-train`): config none, seed 0, all 21,696 optimizer steps, micro-batch 8, one L40 or L40S (whichever schedules first, the user 2026-11-16), output `/mnt/proxy/runs/baseline-none-s0` on the PVC. The manifest header has every number's source and the deadline arithmetic. Paths are relative to this experiment folder. `$C` is the source commit. Format follows `RUN-check4.md`.

**The submit needs the user's go in the lead's window** (an approval of these files is not a go). State before asking: the Job, the cards, about 24 h (L40S) to 31 h (L40) of pod time and the same in GPU-hours (LOG 2026-11-16 projection), the caps (62 h of training per pod, 96 h for the Job including queue and resumes), one GPU, and that nothing on the PVC is deleted. **`podFailurePolicy` has never been validated by a server** (step 5).

## What differs from check 4, in short
- A pod lost to a node reboot is retried (`backoffLimit` 2) and **resumes from `ckpt.pt` by itself**. A folder with files but no `ckpt.pt` is renamed to `baseline-none-s0.stale-<date>` and the run starts fresh (never deleted). The pod prints `MODE: fresh | resume | complete`.
- `--ckpt-every` is chosen in the pod: L40S 700, L40 500 (printed as `CARD ... -> --ckpt-every`). It is not part of the run identity, so a resume on the other card is allowed.
- `kubectl logs` shows only every 100th step line (the pod log would otherwise pass the kubelet's rotation size). The complete log is on the PVC: `/mnt/proxy/logs/baseline-none-s0-<date>.log` (one file per pod). The exit trap prints sha256, size and last 3 lines of `train.json` and `history.jsonl`; the full files stay in the out folder.

## Before submitting

1. **The user's go**, after the statement above.
2. **Source commit** `$C` (on a clean, pulled `main`): `git rev-parse HEAD`.
3. **ConfigMap `stacking-encoders-baseline-src`** (7 keys, already exists, unchanged). Read each key back and compare with `$C`'s file (the hashes in `RUN-check4.md` step 3.1 are the expected values if `src/` did not change since):
   ```
   for f in pretrained_train.py pretrained_sizing.py action_expert_init.py proxy_wam.py diagnostics.py t5_cache.py setup_env.sh; do
     a=$(kubectl get configmap stacking-encoders-baseline-src -n aria-world -o jsonpath="{.data.${f//./\\.}}" | shasum -a 256 | cut -c1-64)
     b=$(git show $C:world-model-stacking-encoders/src/$f | shasum -a 256 | cut -c1-64)
     [ "$a" = "$b" ] && echo "ok $f" || echo "DIFF $f"
   done
   ```
   Any `DIFF`: stop and ask.
4. **Render**, replacing only the env value line:
   ```
   mkdir -p jobs/rendered
   sed "s/{name: SRC_GIT_COMMIT, value: FILL_AT_SUBMIT}/{name: SRC_GIT_COMMIT, value: $C}/" jobs/job-full-train.yaml > jobs/rendered/job-full-train.yaml
   grep -c "value: $C}" jobs/rendered/job-full-train.yaml    # must print 1
   grep -c FILL_AT_SUBMIT jobs/rendered/job-full-train.yaml   # must print 0
   ```
5. **Server dry run, then create**: `kubectl create --dry-run=server -f jobs/rendered/job-full-train.yaml`. If it rejects `podFailurePolicy`, stop and tell the user (do not remove the rule without their OK). Then `kubectl create -f jobs/rendered/job-full-train.yaml`. Note the creation time in the LOG; the 96 h Job cap counts from it. Check the GPU count first (ours plus 1 at most 8, `STANDARDS.md` line 49). Resubmitting the same Job later (same name, after deleting the old Job) resumes the same folder.
6. **One wait, in the background**: the loop of `RUN-sanity.md` step 7 with `J=stacking-encoders-baseline-full-train` and `END=$(( $(date +%s) + 346000 ))`. If the wait passes about 50 min, write the handoff note first and end the session.

## First checks

7. **15 minutes after the pod is Running** (`STANDARDS.md` line 48): `kubectl logs <pod> -n aria-world | grep -E '^CARD|^MODE|^STALE|^PVC free|^=== PHASE|^\[gpu\]|^\[train|\[ckpt\]|\[projection\]|resuming' | tail -40` (get `<pod>` from `kubectl get pods -n aria-world -l job-name=stacking-encoders-baseline-full-train -o name`). Expect the card line and its `--ckpt-every`, `MODE: fresh`, a PVC line, phases `setup` and `stage-wan` at 0% for about 2 min, then `train` at 90 to 100%. `device=cpu`, a CUDA error or `STOP:`: delete the Job and tell the user. Then check the Violations page (`_standards/08`).
8. **First-minutes projection** (the `[projection] after 20 steps:` line, about 2.5 min into training): it prints `full run of 21696 steps = <h> h` (compute only). Add about 0.2 h for checkpoint writes. **If that total is over 62 h, stop and ask the user before anything else** (the training step would be killed by its 62 h `timeout`, exit 124, with no `model.pt`). Expected: about 24 h (L40S) or 30 h (L40). Also read the first `[ckpt] step <n> written ... in <s>s` line (about 21 s in check 4).

## What each exit code means (pod log last line `=== exit <rc>`; the Job's `podFailurePolicy` applies)

| rc | Meaning | Job result | What to do |
|---|---|---|---|
| 0 | all steps done, `model.pt` and its sha256 printed | Complete | step 9 |
| 2 | non-finite loss or grad norm (`STOP:`) | fails at once | stop; tell the user; `ckpt.pt` is kept; do not just resubmit (a resume repeats the same draws) |
| 10 | commit not 40-hex | fails at once | re-render |
| 13 | staged Wan files differ from the recorded sha256 | fails at once | stop; tell the user |
| 14 | PVC under 70 GiB free (fresh) or 40 GiB (resume) | fails at once | stop; tell the user; nothing may be deleted without their OK |
| 15 | card is not an L40 or L40S | fails at once | tell the user |
| 124 | a step hit its `timeout` (pip 1800 s, a staging `dd` 900 s, training 62 h, model sha256 900 s); the last `=== PHASE` line says which | fails at once | training: the 62 h cap was used, tell the user; a pip or staging hang: delete the Job and resubmit (the run resumes) after telling the user |
| 1 | pip failed, or a Python error (CUDA out of memory, a failed `torch.save`, "No space left", `... was written by a different run`) | counts toward the 2 retries | a retry resumes from `ckpt.pt`; if all 3 pods fail the same way, read the traceback and ask the user. A CUDA out-of-memory: PLAN micro-batch rule, `MICRO_BATCH` 4 only with the user's OK |
| 12 | Wan staging failed | counts toward the retries | as above |
| 137 / `OOMKilled` | host memory over 38Gi | counts toward the retries | a resume loads about 20 GB into RAM; read `[cgroup]` lines of earlier pods, raise the request from the evidence, ask before resubmitting |
| node reboot, drain, preemption (`DisruptionTarget`) | pod lost | retried, does NOT count toward the 2 | nothing; check that the next pod shows `MODE: resume` |
| pod `DeadlineExceeded` (64 h) | the pod cap fired before the inner timeout; no exit line | counts toward the retries | read the PVC log |
| Job `DeadlineExceeded` (96 h from creation) | queue plus runs passed the cap | fails | ask the user whether to resubmit (it resumes) |

## How a resume shows up in the log
The new pod prints `MODE: resume`, a `ls -la` of the out folder (`ckpt.pt` size and time), then in the training output `resuming from /mnt/proxy/runs/baseline-none-s0/ckpt.pt at step <N>` (and `[resume] MICRO-BATCH CHANGED` only if the micro-batch differs, which it should not), and the step lines continue from `<N>`. Each resume also writes `config-resume-step<N>.json` in the out folder (printed by the exit trap) and appends to `history.jsonl` (a step re-run after a resume appears twice; the script keeps the last record per step in `train.json`). A pod that died before its first checkpoint instead prints `STALE: ... moving it to ...stale-<date>` and `MODE: fresh`. Each pod writes its own log file on the PVC. Log each resume in the LOG with the step and card.

## After the Job ends

9. **Save the logs right away** (container logs vanish within about a day): for every pod of the Job (`kubectl get pods -n aria-world -l job-name=stacking-encoders-baseline-full-train`), `kubectl logs <pod> -n aria-world > results/full-train/<pod>.log` (the pod, not `job/`), and `kubectl get job stacking-encoders-baseline-full-train -n aria-world -o yaml > results/full-train/job-status.yaml`. The pod log has the step lines only every 100th; the complete logs, `history.jsonl` and `train.json` are on the PVC (`/mnt/proxy/logs/baseline-none-s0-<date>.log`, `/mnt/proxy/runs/baseline-none-s0/`); copy them with a small CPU pod after asking the user (memory: external copies and new cluster work need their OK in this window), and check each copy against the sha256 and byte count the exit trap printed (`=== BIGFILE`). Log the `config.json` and `load.json` extracts as in `RUN-check4.md` step 11.
10. **What to log**: Job creation time, card and node of each pod, each `--ckpt-every`, each resume (step, card), mean s/step and `[gpu-summary]` per phase per pod, the write times, the `[cgroup] memory.peak` and anon lines of every pod against the 32Gi request (this sizes any later run and settles whether the portal's memory check is a concern), the exit codes, the `=== MODEL` sha256 of `model.pt`, total pod time and GPU-hours.
11. **Delete the Job** after the logs are saved: `kubectl delete job stacking-encoders-baseline-full-train -n aria-world` (graceful, never `--force`); `kubectl get pods -n aria-world -l job-name=stacking-encoders-baseline-full-train` must show nothing. **Delete nothing on the PVC.** `ckpt.pt` and `model.pt` stay until the user OKs deleting them (PLAN "Checkpointing and resume", "At the end"; the check-4 folder `baseline-check4-none-s0-try1` stays too).
