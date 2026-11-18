# SigLIP 2 damaged-episode repair on one L4: steps for the lead

Written 2026-11-18 by a worker, Mac only. Nothing here has been submitted. One Job, `jobs/job-siglip2-repair.yaml` (Job `stacking-encoders-pairings-siglip2-repair`), one pod, one NVIDIA L4. **Every command below runs from this experiment folder** (`world-model-stacking-encoders/experiments/2026-11-15-pretrained-start-pairings/`); `$C` is the source commit.

**What and why.** The cache check after the re-extraction found 8 SigLIP 2 libero_spatial episodes (ep 272 to 286, even ids) that are full-size files of zero bytes, left by the CephFS pool-full event (LOG 2026-11-18). The value check (`RUN-cache-value-check.md`) writes the full list of damaged episodes. The user approved repairing only the damaged episodes on an L4 (LOG 2026-11-18, decision 3; each new Job needs its own go). The re-extraction Job cannot do it (it refuses a non-empty folder, and the extractor skips any episode whose `.npy` exists), so this Job does it in three steps: `src/repair_stage2_episodes.py prepare` checks the list and that every listed episode really fails the value check, then renames the damaged `ep<id>.npy` and `ep<id>.stats.npy` into `/mnt/proxy/logs/siglip2-repair-<Job uid>/<suite>/` (never a delete); `extract_fastwam.py --episodes <ids>` recomputes them with the re-extraction's settings; `postcheck` value-checks them and checks that nothing else in the suite folder changed. That last check compares names, sizes, mtimes and entry types against a listing taken before the moves, and sha256 only for the two shard-of-2 `_meta` files; it does not compare the bytes of the other files (any write by this Job would change an mtime; hashing the folder would cost as much as a value check).

**The submit needs the user's go in the lead's window** (new cluster work; approving these files is not a go). Tell the user, before asking:
- one Job, one L4 GPU, about 5 minutes of pod time (about 0.1 GPU-hours) once it is scheduled, which can take hours on the two L4 nodes (manifest header: estimate and basis);
- the damaged files are moved aside by rename into `/mnt/proxy/logs/siglip2-repair-<Job uid>/` on the same volume; nothing is deleted, and they stay there until the user OKs deleting them;
- the list it repairs (step 2), and that nothing else in the cache changes (the post-check fails the Job if any other file in a repaired suite folder appears, disappears, or changes size, mtime or type, or if a shard-of-2 `_meta` file changes content);
- afterwards the value check and the freshness check run again, each with its own go (step 11).

## What is the same as the re-extraction, and what changes

Same as `jobs/job-siglip2-reextract.yaml`: image, `setup_env.sh`, env (`OMP_NUM_THREADS=3`, `MALLOC_ARENA_MAX=2`, `HF_HOME`, the HF secret), node affinity (`NVIDIA-L4`, driver > 569, us-west, the two fault hosts excluded), the `/src` hash check (exit 19), the `SRC_GIT_COMMIT` check (exit 10), the Job uid check (exit 11), the CUDA + L4 pre-check (exit 13), the `[gpu]` nvidia-smi sampler, the extractor call (`--source siglip2 --dtype fp16 --batch 32 --device cuda`, same `--root` and `--out`), resources (CPU 3 / 3600m, memory 12Gi / 14400Mi, ephemeral 30Gi / 34Gi; copied, since there is no measurement of this Job to size from), `backoffLimit: 2`, deadline 16 h from creation, TTL 1 day.

Changed:
- One pod, no shards. The extractor runs with `--episodes <ids>` and its default `--shard 0/1`. `--episodes` filters before the shard slicing, so every listed episode is computed, and the extractor writes a new `_meta.shard0of1.json` in each repaired suite (recording this commit and `device: cuda`). `_meta.shard0of2.json` and `_meta.shard1of2.json` are not touched: their sha256 is printed before and after, and a change fails the Job (exit 25).
- The ConfigMap `stacking-encoders-pairings-repair-src` has 8 keys: the five of the re-extraction, `repair_stage2_episodes.py`, `check_stage2_cache.py` (the driver imports its value check) and `damaged-episodes.txt` (the list, hash-checked like the code).
- The re-extraction's empty-folder guard (exit 12) and its `_meta` shard check (exit 14) are replaced by the driver's pre-check and post-check (exit 20 to 27).
- `NODE_NAME` from the downward API, for the log and the record.
- A per-pod time cap, `activeDeadlineSeconds: 3600` on the pod template (counted from pod start; the stage 2 templates have a pod deadline too, `CHECKLIST-08.md` "A deadline that bounds the Job"), so a hung pod cannot hold the L4 for the rest of the 16 h Job deadline. The re-extraction had none.

**Readers of the extra `_meta.shard0of1.json`** (every reader of `feats/<source>/<suite>/_meta.*.json` found by grep in `src/` and the stage 2 job files, read for this Job):
- `check_stage2_cache.py` `provenance()`: globs `_meta.shard*.json`, groups by (model, hf_id, resolution, dtype), and reports a gap only if the files disagree. Same values, so one group, "(3 file(s))" (test `MetaReaders`; the post-check calls it too).
- `check_cache_fresh.py` `check_provenance()`: globs the same pattern and checks each file's model and dtype against the encoder it recomputes with. The new file has the same values (test `MetaReaders`).
- Stage 2 training and evaluation (`proxy_wam.py` `Data`, used by `pretrained_train.py`, `pretrained_sizing.py` and `rollout.py`; `make_sizing_slice.py`; `lp_eval.py`): read `feats/meta/<suite>.npz`, `ep<id>.npy` and `ep<id>.stats.npy` by name; none reads the `_meta` JSON files.
- Job scripts: `job-siglip2-reextract.yaml` checks only `_meta.shard<i>of2.json`, by name; the layout pattern of `job-siglip2-delete.yaml` accepts `_meta.shard<i>of<n>.json`.

## Steps

1. **What must not be running.** `kubectl get jobs,pods -n aria-world`. No other Job may read or write `feats/siglip2` while this one runs:
   - the value check Job (`stacking-encoders-pairings-cache-value-check`) must have ended, its log saved and the list written (its RUN steps 5 and 6);
   - the freshness check (`stacking-encoders-pairings-cache-fresh-check`) reads `feats/siglip2` (its fixed draw includes ep000276, one of the listed files) and waits for the same two L4 nodes this Job needs. It must have ended (log saved) or be deleted (graceful) before step 5. Deleting it is a stop for the user's OK; it is rerun after the repair anyway (step 11);
   - no stage 2 Job of `siglip2` or `dinov3+siglip2` (still on hold).
   Count our GPUs: this pod plus the running ones, at most 8.
   **Ceph health** (the damage came from a CephFS pool-full event), from the Mac, public Prometheus, no kubectl:
   ```
   curl -sG https://prometheus.nrp-nautilus.io/api/v1/query --data-urlencode 'query=ceph_health_detail{namespace="rook",name=~"OSD_FULL|POOL_FULL|OSD_NEARFULL|POOL_NEARFULL"}'
   ```
   Read each result's `metric.name` and value (1 = the alarm is on). `OSD_FULL` or `POOL_FULL` on: do not submit; tell the user and check again later. `OSD_NEARFULL` or `POOL_NEARFULL` on: tell the user before the go (since PR #64 a refused write fails the pod loudly instead of leaving zeros, but the queue wait and the retries would be spent for nothing). An empty result or an error: say so to the user; do not read it as healthy.
2. **The list.** From `results/cache-value-check/damaged-episodes.txt` (value check RUN step 6):
   - every zero-page hit must have been confirmed by reading that file's bytes at the printed offset (value check RUN step 7; `od -A d -t x1 -j <offset> -N 64 <file>` in a read-only pod), and every "cannot read .npy header" hit by its first bytes (as on 2026-11-18). An unconfirmed line: stop.
   - keep the SigLIP 2 lines: `mkdir -p results/siglip2-repair; grep '^siglip2 ' results/cache-value-check/damaged-episodes.txt > results/siglip2-repair/damaged-episodes.txt`. Lines of another source (DINOv3, C-RADIOv4, VAE) are not repaired by this Job: record them and stop for the user.
   - check it: `/opt/homebrew/bin/python3.11 ../../src/repair_stage2_episodes.py check-list --list results/siglip2-repair/damaged-episodes.txt` prints one line per suite and `=== list OK: <n> episode(s)`. Anything else (exit 20): stop. An empty list means nothing to repair: do not submit.
   - if the list is not exactly the 8 known episodes, redo the manifest header's estimate (the episodes' frames at about 22 frames/s, plus the reads) and tell the user the new figure.
   - commit `results/siglip2-repair/damaged-episodes.txt` on main before step 3 (it is an input of the Job and is checked against `$C`).
3. **Source commit, code check and ConfigMap.** `$C` = `git rev-parse HEAD` on a clean, pulled `main` that contains this branch and the list.
   **Code check first:** the repaired episodes must be computed by the same feature code as the rest of the cache, which the re-extraction wrote at source commit `0c5d1586dd028447c388a2cdcd23efb30ac12cee` (LOG 2026-11-18, re-extraction steps 2-3; the first line of every log in `results/siglip2-reextract/`). The post-check compares only model, resolution and dtype, so it would not catch a change in preprocessing, a revision pin or a package pin.
   ```
   R0=0c5d1586dd028447c388a2cdcd23efb30ac12cee; P=world-model-stacking-encoders/src
   git diff $R0 $C -- $P/extract_features.py $P/fastwam_data.py $P/diagnostics.py $P/setup_env.sh   # must print nothing
   git diff $R0 $C -- $P/extract_fastwam.py                                                         # must be only the fsync change in save_npy
   ```
   The expected `extract_fastwam.py` diff is the PR #64 change: in `save_npy` only: `f.flush()` and `os.fsync(f.fileno())` before `os.replace` in both branches (in the `--stage-dir` branch, which this Job does not use, `shutil.copyfile` became `shutil.copyfileobj` into an open file so it can be fsynced), plus three docstring lines; it does not change the bytes written (`test_extract_fastwam_save.py`). Checked on the Mac for `0c5d158` against `ada298c`: the four-file diff is empty and the `extract_fastwam.py` diff is exactly this. Any other difference in these five files: stop and ask the user.
   Then:
   ```
   S=../../src; LST=results/siglip2-repair/damaged-episodes.txt
   kubectl create configmap stacking-encoders-pairings-repair-src -n aria-world --from-file=repair_stage2_episodes.py=$S/repair_stage2_episodes.py --from-file=check_stage2_cache.py=$S/check_stage2_cache.py --from-file=extract_fastwam.py=$S/extract_fastwam.py --from-file=extract_features.py=$S/extract_features.py --from-file=fastwam_data.py=$S/fastwam_data.py --from-file=diagnostics.py=$S/diagnostics.py --from-file=setup_env.sh=$S/setup_env.sh --from-file=damaged-episodes.txt=$LST --dry-run=server
   ```
   then the same without `--dry-run=server`. Read every key back against `$C`:
   ```
   E=world-model-stacking-encoders/experiments/2026-11-15-pretrained-start-pairings
   for f in repair_stage2_episodes.py check_stage2_cache.py extract_fastwam.py extract_features.py fastwam_data.py diagnostics.py setup_env.sh damaged-episodes.txt; do
     case $f in damaged-episodes.txt) p=$E/results/siglip2-repair/$f;; *) p=world-model-stacking-encoders/src/$f;; esac
     a=$(kubectl get configmap stacking-encoders-pairings-repair-src -n aria-world -o jsonpath="{.data.${f//./\\.}}" | shasum -a 256 | cut -c1-64)
     b=$(git show $C:$p | shasum -a 256 | cut -c1-64)
     [ "$a" = "$b" ] && echo "ok $f" || echo "DIFF $f"
   done
   ```
   Any `DIFF`: stop and ask. The list must match byte for byte, trailing newline included.
4. **Render** (from this experiment folder, with `$C` and `$E` as above):
   ```
   mkdir -p jobs/rendered; Y=jobs/rendered/job-siglip2-repair.yaml
   sed "s/{name: SRC_GIT_COMMIT, value: FILL_AT_SUBMIT}/{name: SRC_GIT_COMMIT, value: $C}/" jobs/job-siglip2-repair.yaml > $Y
   for f in repair_stage2_episodes.py check_stage2_cache.py extract_fastwam.py extract_features.py fastwam_data.py diagnostics.py setup_env.sh damaged-episodes.txt; do
     case $f in damaged-episodes.txt) p=$E/results/siglip2-repair/$f;; *) p=world-model-stacking-encoders/src/$f;; esac
     h=$(git show $C:$p | shasum -a 256 | cut -c1-64)
     sed "s/FILL_SHA_$f /$h /" $Y > $Y.tmp && mv $Y.tmp $Y
   done
   grep -c "value: $C}" $Y                 # must print 1
   grep -v '^#' $Y | grep -c FILL_         # must print 0
   grep -cE '^ +[0-9a-f]{64}  ' $Y         # must print 8
   ```
   Commit the rendered file as submitted.
5. **Server dry run, then create.** Recount GPUs (step 1). `kubectl create --dry-run=server -f jobs/rendered/job-siglip2-repair.yaml`, then `kubectl create -f jobs/rendered/job-siglip2-repair.yaml`. Note the creation time and the Job uid (`kubectl get job stacking-encoders-pairings-siglip2-repair -n aria-world -o jsonpath='{.metadata.uid}'`) in the LOG.
6. **One check, no loop.** About 30 minutes after creation: `kubectl get job,pods -n aria-world -l job-name=stacking-encoders-pairings-siglip2-repair -o wide`. Still Pending: check again at most once an hour. Pending for more than about 2 h: ask the user (do not widen the card list; L4 only is the point, as for the re-extraction). Once the pod has started, it ends within minutes. Do not use `kubectl wait --for=condition=complete` alone: a refusal makes the Job fail and that wait would sit until its timeout.
7. **Save the log right away** (container logs vanish within about a day), from the pod, not `job/`:
   ```
   kubectl get pods -n aria-world -l job-name=stacking-encoders-pairings-siglip2-repair
   kubectl logs <pod> -n aria-world > results/siglip2-repair/<pod>.log
   kubectl get job stacking-encoders-pairings-siglip2-repair -n aria-world -o yaml > results/siglip2-repair/job-status.yaml
   ```
   Save every pod's log if there was a retry. The full log is also on the PVC as `/mnt/proxy/logs/siglip2-repair-<Job uid>-<time>.log`.
8. **Read the result.** Expected, in order: eight `: OK` hash lines; the list; the nvidia-smi header naming an `NVIDIA L4`; `cuda True NVIDIA L4`. Under `=== prepare`: `first pod: no snapshot.json`, two `before: sha256` lines per suite, one `damaged, <k> gap(s)` line per episode, `snapshot written` with the file count per suite (870 for libero_spatial if the folder holds only 434 episodes x 2 files and the 2 `_meta` files), two `moved aside` lines per episode, `moves: <2n> file(s)`, the plan, `=== prepare OK`. Under `=== extract`: `0 of <n> episodes already done` and one `ep <id>: (<frames>, 14, 28, 1152)` line per episode near 22 frames/s. Under `=== postcheck`: `passes the value check` per episode, two `after: sha256` lines equal to the `before:` ones, the new meta with `'device': 'cuda'` and `$C`, the provenance line ending `(3 file(s))`, `files in siglip2/<suite>: before <N>, after <N+1>`, the record JSON, `=== repair post-check OK`, `=== done`. Record in the LOG: node, card, times, the episodes, frames/s, the `[gpu]` samples, GPU-hours against the estimate, and the record. The record file is `/mnt/proxy/logs/siglip2-repair-<Job uid>/repair-record.json` (the log prints it in full).
9. **Exit codes.** Every code in the `podFailurePolicy` list (10, 11, 13, 19 to 27) fails the Job at once; any other code (1, 2, 137, a signal, a pod killed by its 1-hour cap) is retried, up to `backoffLimit: 2`:
   - 10 not rendered; 19 a `/src` key differs from the render (code or list changed after the render): redo steps 3 and 4. 11 no Job uid label: ask. 13 no CUDA or not an L4: a node fault, ask before resubmitting. Nothing was moved in any of these.
   - 20 the list was refused (another source, a malformed or blank line, a duplicate, empty, an episode not in the suite's meta). 21 a listed episode passes the value check and was not moved by this Job (the list is stale or wrong). 22 an unexpected state: a `_meta.shard0of1.json` this Job did not write (a repair or extraction ran there before), a missing shard-of-2 `_meta`, a leftover `.tmp` of a listed episode, files in the Job's folder without a snapshot, a file that is not the one in the snapshot. A first pod that refuses with 20, 21 or 22 has moved nothing; on a retried pod, the earlier pod's moves stay as they are.
   - 23 a rename failed (cross-device or I/O). No copy was made; files already moved are in the Job's folder, the rest are in place. Ask.
   - 24 a listed episode still fails the value check after the extraction; 25 a shard-of-2 `_meta` changed; 26 another entry in the suite folder appeared, disappeared, or changed size, mtime or type against the snapshot listing (not a byte comparison), or a moved-aside copy is missing or has another size or mtime; 27 the new `_meta.shard0of1.json` is missing, not `device: cuda`, not `$C`, differs from the old ones in model, resolution or dtype, or the provenance check reports a gap. Every problem is printed as `PROBLEM (exit <k>)` and kept in `repair-record.json`.
   - 1 an uncaught exception (a traceback; the extractor or the driver crashed), 137 an out-of-memory kill, or the pod deadline (1 h from pod start, `DeadlineExceeded` on the pod): the next pod of the same Job resumes (see "Retries"); a third failure ends the Job.
10. **Delete the Job and the ConfigMap** after the logs are saved: `kubectl delete job stacking-encoders-pairings-siglip2-repair -n aria-world` (graceful, never `--force`), `kubectl delete configmap stacking-encoders-pairings-repair-src -n aria-world`. The moved-aside files stay in `/mnt/proxy/logs/siglip2-repair-<Job uid>/` until the user OKs deleting them (about 0.7 GB for the 8 known episodes); this file has no delete step on purpose.
11. **After the repair**, each with the user's go: the value check again (`RUN-cache-value-check.md`, `--values` over all caches; expected 0 gaps, SigLIP 2 provenance now 3 `_meta` files in each repaired suite with one model, resolution and dtype), then the freshness check again (`RUN-cache-fresh-check.md`, rendered to a new file; same seed, so the same 8 episodes, ep000276 among them; expected SigLIP 2 bit-identical on all 8). Until both pass, no Job may read `feats/siglip2` for training or evaluation.

## Retries
A pod of the same Job (same uid) after a crash or a node loss resumes from the files, not from memory. `snapshot.json` in the Job's folder says the pre-check passed. An episode with a copy in the Job's folder was moved by this Job, so it is not value-checked again before the extraction. A file still in the cache without its copy is moved; all moves finish before the extractor first runs, so such a file is always the original. The extractor skips an episode whose `ep<id>.npy` exists, so a repaired one is kept and a moved but unwritten one is computed. The marker `meta-shard0of1.expected`, written before the extractor runs, lets the retry accept the `_meta.shard0of1.json` that the earlier pod's extractor wrote. A new Job (new uid) after a failed one refuses (exit 22) if the failed one already ran the extractor, because of that `_meta.shard0of1.json`; how to go on from there is the user's call (the earlier Job's folder holds its moved files and its snapshot). If the failed Job stopped partway through the moves instead (for example exit 23 after one rename), a new Job moves the episode's remaining file into its own folder, so the moved-aside copies of one episode can sit in two Job folders; a later cleanup must look in both.

A pod that runs past its 1-hour cap (`activeDeadlineSeconds: 3600` on the pod template, counted from pod start) is killed by Kubernetes and counts as a failed pod; its code is not in the FailJob list, so the next pod of the same Job starts (each pod gets its own hour) and resumes from the files as above. A pod killed mid-rename leaves either the file in place or its copy in the Job's folder (a rename is atomic); one killed mid-write leaves at most a `.ep<id>.npy.tmp`, which the extractor reuses and replaces.

## GPU use and NRP's 40% rule
The pod holds the GPU for a few minutes, and the GPU is idle most of that time (setup, the pre-check and post-check reads); only the roughly 35 s of extraction runs it near the 90% the re-extraction showed. NRP's checks do not judge a pod this short: the Violations page skips pods younger than 30 minutes, and the warning email and the admin idle list need 180 minutes of runtime (`../../../_standards/08`, "GPU utilization"); the two matching L4 nodes (gpu05, gpu06) also export no DCGM series. What matters is not to leave the pod or the Job behind: if the pod is still Running more than about 30 minutes after it started, read the log (a stall) and ask the user. The `[gpu]` samples are for the LOG, not for a 15-minute check.

## Stop conditions (stop, record, ask the user; do not work around)
- A Job that reads `feats/siglip2` is running or queued (step 1), or the freshness check would have to be deleted.
- The list holds lines of other sources, unconfirmed zero-page hits, or other episodes than the 8 known (tell the user the new list and time before the go).
- Any non-zero exit (step 9), or a log that does not end with `=== done`.
- On a first pod: a `moves:` count other than 2 x the listed episodes (fewer only if the value check reported a missing file), or `already done` other than 0.
- A pod on a card other than an L4 (should be impossible with the affinity).
- Pending for more than about 2 h.
- Do not delete, copy back or edit any file in `/mnt/proxy/feats` or in the Job's folder by hand.
