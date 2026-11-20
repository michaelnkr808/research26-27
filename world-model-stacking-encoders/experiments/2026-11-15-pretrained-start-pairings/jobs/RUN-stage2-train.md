# Stage 2 training Jobs: steps for the lead

Written 2026-11-17 by a worker, Mac only. Nothing here has been submitted; the only kubectl use was `kubectl apply --dry-run=client` on rendered copies (that makes read-only discovery calls to the API server; nothing was created). One Job per (configuration, training seed), rendered from `jobs/job-train.template.yaml` by `jobs/render.sh`. The template is stage 1's `job-full-train.yaml` (which ran to exit 0) with the changes listed in its header. Paths are relative to this experiment folder. `$C` is the source commit. Which runs go in which wave, and what the user must OK first: `RUN-stage2-waves.md`.

**Every submit needs the user's go in the lead's window**, given for the sweep as a whole with the wave plan (`RUN-stage2-waves.md`; PLAN Order of work step 5). The design locks when the first wave 1 training run starts (PLAN Order of work step 6).

## Per-configuration values (render.sh table; arithmetic and labels in the template header)

| config | Job name | micro-batch | --ckpt-every L40S / L40 | time cap | pod / Job deadline | host memory req / limit |
|---|---|---|---|---|---|---|
| none (seeds 1, 2) | stacking-encoders-pairings-train-none-s<seed> | 8 | 700 / 500 | 62 h | 64 h / 96 h | 32Gi / 38Gi |
| dinov3 | ...-train-dinov3-s<seed> | 8 | 700 / 500 | 63 h | 65 h / 97 h | 32Gi / 38Gi |
| siglip2 | ...-train-siglip2-s<seed> | 8 | 700 / 500 | 63 h | 65 h / 97 h | 32Gi / 38Gi |
| c-radiov4 | ...-train-c-radiov4-s<seed> | 8 | 700 / 500 | 63 h | 65 h / 97 h | 32Gi / 38Gi |
| dinov3+siglip2 | ...-train-dinov3-siglip2-s<seed> | 8 | 600 / 500 | 65 h | 67 h / 99 h | 36Gi / 43Gi |
| dinov3+c-radiov4 | ...-train-dinov3-c-radiov4-s<seed> | 8 | 600 / 500 | 65 h | 67 h / 99 h | 36Gi / 43Gi |

Only `none` has measured step times on Nautilus. Everything for the encoder configurations is a linear extrapolation of the one sizing measurement for `dinov3` (+2.2% per step; speculative) and is checked in the first minutes (step 8). Out folder: `/mnt/proxy/runs/pairings-<cfgk>-s<seed>`, or `/mnt/runs/runs/pairings-<cfgk>-s<seed>` with `--runs-pvc` (second volume). `none` seed 0 is stage 1's run and render.sh refuses it.

## Once per stage (before the first training submit)

1. **The user's go for the sweep** (`RUN-stage2-waves.md`, "What the user must OK").
2. **Source commit** `$C` on a clean, pulled `main`: `git rev-parse HEAD`. Check that the 7 training files are the ones stage 1 ran, apart from one known change (17d9cbb is the last change before stage 1's full run):
   ```
   P=world-model-stacking-encoders/src
   git diff 17d9cbb $C -- $P/pretrained_sizing.py $P/action_expert_init.py $P/diagnostics.py $P/t5_cache.py $P/setup_env.sh   # must print nothing
   git diff 17d9cbb $C -- $P/pretrained_train.py $P/proxy_wam.py                                                              # must be only the fsync change
   ```
   The expected `pretrained_train.py` and `proxy_wam.py` diff is the fsync change made after the CephFS pool-full event (LOG 2026-11-18), and nothing else: each temporary file is flushed and `os.fsync`'ed before its rename onto the final name, so a write the storage refuses raises instead of renaming a file of zeros over a good one. In `pretrained_train.py`: `_write_json` writes through `open(tmp, "w")` with `f.flush()` and `os.fsync(f.fileno())` in place of `tmp.write_text(...)` (plus a two-line docstring, and the end-of-line comment `# same text and encoding as Path.write_text` on the `open` line), and the `model.pt` save at the end of `_main` runs inside `with open(tmp, "wb") as f:` followed by `f.flush()` and `os.fsync(f.fileno())`, `torch.save` still given the path `tmp` (the `open` line carries the end-of-line comment `# opened before the write so its fsync reports a refused write (see P.save_ckpt)`). In `proxy_wam.py`, in `save_ckpt` only: the same `with open(tmp, "wb") as f:` wrapper with `f.flush()` and `os.fsync(f.fileno())`, a local `import os`, and four docstring lines. It does not change what is saved: `test_train_save_fsync.py` checks `ckpt.pt`, `model.pt`, `config.json` and `load.json` against the code at `1b1fb13` (loaded content and raw bytes identical), and `test_none_invariance.py` passed on the change. Between 17d9cbb and `1b1fb13` these 7 files had no change (checked on the Mac). Any other difference in these 7 files: the PR that made it must have run `test_none_invariance.py` (PLAN code item 6); say so in the LOG.
3. **Create the ConfigMap** `stacking-encoders-pairings-train-src` (the same 7 keys as stage 1's `stacking-encoders-baseline-src`), from `$C`'s files, by your usual method (`--dry-run=server` first), then read each key back:
   ```
   for f in pretrained_train.py pretrained_sizing.py action_expert_init.py proxy_wam.py diagnostics.py t5_cache.py setup_env.sh; do
     a=$(kubectl get configmap stacking-encoders-pairings-train-src -n aria-world -o jsonpath="{.data.${f//./\\.}}" | shasum -a 256 | cut -c1-64)
     b=$(git show $C:world-model-stacking-encoders/src/$f | shasum -a 256 | cut -c1-64)
     [ "$a" = "$b" ] && echo "ok $f" || echo "DIFF $f"
   done
   ```
   Any `DIFF`: stop and ask. Never edit this ConfigMap while any training Job of the sweep exists (a retried pod would exit 19). A code change mid-sweep means a new ConfigMap name and a new `$C`, and the LOG says which runs used which.

## Per run

4. **Render** (from `jobs/`): `./render.sh train <config> <seed> --commit $C` (add `--runs-pvc stacking-encoders-runs`, or whatever name the user approves, once the second volume exists; add `--micro-batch 4` only under step 9). It prints the summary line and `remaining FILL_ placeholders:` (must be empty). Then:
   ```
   Y=rendered/job-train-<cfgk>-s<seed>.yaml
   grep -cE '^ +[0-9a-f]{64}  ' $Y                       # must print 11 (7 /src keys + 4 Wan files)
   grep -n -i 'sleep\|toleration\|priorityClass' $Y     # must print nothing
   ```
   Commit the rendered file as submitted. Never edit the template for one run.
5. **Caps before submit.** (a) GPUs: the GPUs our pods hold or are Pending for, plus 1, must be at most 8 (`STANDARDS.md` line 49). The pods of this sweep carry the label `experiment=stacking-pairings` (`kubectl get pods -n aria-world -l experiment=stacking-pairings`); add any other stacking pod (`kubectl get pods -n aria-world | grep stacking-encoders`). (b) Disk: free space on the runs volume must be at least 40 GB x (running training Jobs + this one) (PLAN Hand-off step 4; the LOG's 2026-11-17 reading is 190 GiB free on `stacking-encoders-proxy`). The pod's own guard (exit 14: 70 GiB fresh, 40 GiB resume) only sees its own need. Read free space with a small read-only pod or the last pod's `df` line; ask before any new pod.
6. **Server dry run, then create**: `kubectl create --dry-run=server -f $Y`, then `kubectl create -f $Y`. Note the creation time (the Job deadline counts from it). Before any manual resubmit, make sure no pod of the old Job remains (two pods on one out folder would interleave `ckpt.pt.tmp` and `history.jsonl`). A resubmit resumes the same folder and must use the same `$C`.
7. **One wait**: the background loop of `../../2026-11-15-pretrained-start-baseline/jobs/RUN-full-train.md` step 6 with `J=<this Job's name>` and `END` = the Job deadline + 400 s. Several training Jobs: one loop over all of them, one `kubectl get jobs -n aria-world -l experiment=stacking-pairings,kind=train` every 5 minutes, never one loop per Job. A wait over about 50 min: write the handoff note first.

## First minutes (PLAN Hand-off step 4; each of these is a stop for the user when it fails)

8. **About 15 min after the pod is Running**: `kubectl logs <pod> -n aria-world | grep -E '^CARD|^MODE|^STALE|^PVC free|^=== PHASE|^\[gpu\]|^\[train|\[ckpt\]|\[projection\]|\[first batch\]|\[watchdog\]|resuming|: OK$|FAILED' | tail -60`. Pod stdout keeps step lines 1 to 25 and every 100th. Log in this folder's `LOG.md`:
   - **Micro-batch fit**: the largest `peak_reserved_bytes` over steps 1 to 25 against 0.9 x the card's total memory (`nvidia-smi` line at the top of the log). Over 0.9 at micro-batch 8: delete the Job, re-render with `--micro-batch 4` (PLAN rule; the resume records the change), log it. A CUDA out-of-memory ends the pod with exit 20 (FailJob) and the same re-render applies. **Micro-batch 4 does not fit either: stop for the user.**
   - **Step time and projection**: the `[projection] after 20 steps` line (compute only) + about 0.25 h of checkpoint writes. **Over the rendered cap (62, 63 or 65 h): stop for the user.** Also check the rendered checkpoint interval lands within 30 to 60 min at the measured s/step; outside: re-render the interval (allowed on resume).
   - **Data wait**: the median of `data_wait_s` over steps 2 to 20 against the median `s_step`: `kubectl logs <pod> -n aria-world | grep -E '^\[train +[0-9]+s\] \{"step": ([2-9]|1[0-9]|20),' | sed 's/^[^{]*//' | /opt/homebrew/bin/python3.11 -I -c 'import sys,json,statistics as S; r=[json.loads(l) for l in sys.stdin]; w=S.median(x["data_wait_s"] for x in r); t=S.median(x["s_step"] for x in r); print(len(r),"steps, median wait",w,"median step",t,"ratio",w/t)'`. **Ratio over 0.10: stop for the user** (Open item j, chosen). Since 2026-11-20 the training pods request 2 CPUs with a 2400m limit (template CHANGES item 10; stage 1's `none` run peaked at 1.07 cores, the encoder configurations are unmeasured): with the ratio, report `kubectl top pod <pod> -n aria-world` against 2.4 cores (an instantaneous reading) and the throttled share from public Prometheus, `rate(container_cpu_cfs_throttled_periods_total{namespace="aria-world",pod="<pod>"}[10m]) / rate(container_cpu_cfs_periods_total{namespace="aria-world",pod="<pod>"}[10m])` (PR #70 review). CPU at the limit together with a high data wait points at the limit, and the fix to put to the user is a re-render at the old 4 / 4800m (a resource-only change, allowed on resume).
   - **GPU use**: `[gpu]` samples in the train phase near 90% (stage 1 89.7%). `device=cpu`, a CUDA error or `STOP:`: delete the Job, tell the user. GPU at 0% and no `[watchdog]` line 35 min into Running: delete the Job, tell the user.
   - **First-batch report**: the `[first batch] loss at start` line; the per-source channel statistics are in `first-batch.json` on the PVC (fault checks, PLAN Decision rule item 6).
   - **Free space**: the `PVC free:` line.
   - **Checkpoint write time** (the first write comes at the rendered `--ckpt-every`, 30 to 60 min in, so read it after this check): the first `[ckpt] step <n> written to ... in <s>s` line. Note it in the LOG next to stage 1's: about 21 s per write on an L40S in sanity check 4 (`../../2026-11-15-pretrained-start-baseline/LOG.md`, 2026-11-16) and about 26 to 28 s per write on the L40 in the full run (same LOG, 2026-11-17). Since PR #69 the fsync is inside this time (`pretrained_train.save` times the whole `save_ckpt` call), so it can be longer than stage 1's, and the `[projection] with checkpoint writes` line logged right after it uses it. The step watchdog does not beat during a save: a save that stalls past `--step-watchdog-s` (default 1,800 s in `pretrained_train.py`, not overridden by the template) ends the pod with exit 17 (`WATCHDOG_EXIT`), which the Job retries; the retry resumes from the previous `ckpt.pt`, because the rename never ran.
   - The first pair run's exit-trap `[cgroup]` lines size the rest (host memory is estimated, not measured, for every encoder configuration).
9. **Micro-batch 4 re-render**: `./render.sh train <config> <seed> --micro-batch 4 --commit $C` (same `$C`, and the same `--runs-pvc` as the first render, or none if it had none: a different volume would start a fresh run and ignore the existing `ckpt.pt`), delete the old Job (graceful), wait until its pod is gone, create the new one. The run resumes from `ckpt.pt` if one exists, else it starts fresh (a folder without `ckpt.pt` is renamed `.stale-<date>`, never deleted).

## Exit codes
As `../../2026-11-15-pretrained-start-baseline/jobs/RUN-full-train.md` ("What each exit code means"), with the stage 2 values: the time cap is the rendered one, and one new code:

| rc | Meaning | Job result | What to do |
|---|---|---|---|
| 20 | CUDA out of memory at the rendered micro-batch | fails at once | step 9 (re-render with 4); at 4: stop for the user |
| 1 with "CUDA out of memory" in the pod log | the same, but the traceback reached the PVC log after the script's check (tee lag) | retried at the same micro-batch | delete the Job and treat it as exit 20 |
| 10 | commit not 40-hex, or CONFIG / CFGK / SEED not rendered consistently | fails at once | re-render |
| 1 with an `OSError` (or a `RuntimeError` from `torch.save`) raised in `save_ckpt` in the pod log | the storage refused a checkpoint write (the fsync before the rename raised, PR #69, or the write itself failed); the previous `ckpt.pt` is untouched | retried; the retry resumes from the last good `ckpt.pt` | the refused write can leave a full-size `ckpt.pt.tmp` behind (the next save overwrites it). The retry's free-space guard counts it, so with little free space the retry stops with exit 14 (FailJob) on purpose: tell the user. Do not delete the `.tmp` by hand without the user's OK |

## After each Job ends
10. Save the logs at once (`RUN-full-train.md` step 9, with this Job's name; save into `results/train/<cfgk>-s<seed>/`). Log: creation time, card and node of each pod, each `--ckpt-every`, resumes, mean s/step and `[gpu-summary]` per phase, the `[cgroup]` peak and anon lines, the exit codes, the `=== MODEL` sha256 (it pins every evaluation of this checkpoint), pod time and GPU-hours against the projection.
11. Delete the Job (graceful). **Delete nothing on the PVC.** Each run's `ckpt.pt` stays until the conditions in PLAN amendment 2026-11-15 later, item 3 hold (run finished, its evaluation loaded `model.pt` and wrote a record, the sha256 logged, the backup copy exists and matches) and the user OKs that deletion or batch.
12. A run that fails for a reason that is not a fault is rerun once with the same seed and code (PLAN Design, "Training failures"); a second failure is a missing seed (PLAN Decision rule item 6).
