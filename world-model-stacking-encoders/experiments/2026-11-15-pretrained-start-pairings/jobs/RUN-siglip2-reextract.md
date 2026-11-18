# SigLIP 2 cache re-extraction on L4s: steps for the lead

Written 2026-11-17 (US time; 2026-11-18 UTC) by a worker, Mac only. Nothing here has been submitted. Paths are relative to this experiment folder; `$C` is the source commit.

**What and why.** The cache freshness check (`LOG.md`, 2026-11-18 entry; `results/cache-fresh-check/`) found DINOv3 and C-RADIOv4 bit-identical to a fresh L4 recompute on all 8 checked episodes, but SigLIP 2 bit-identical on only 3 of 8 and past the `rel_max` bar on the other 5, most likely because some of the SigLIP 2 cache was written on other cards (an RTX 5000 Ada, an RTX 3090 and a stratus1 card before the final L4 try; direction `../../LOG.md` 2026-10-29 (h), (m), 2026-10-30 (b), (c)). The user chose in chat on 2026-11-18 ("yeah option 1"): delete the old SigLIP 2 cache, then recompute it in the same place on L4s. Two Jobs:

1. `jobs/job-siglip2-delete.yaml` (Job `stacking-encoders-pairings-siglip2-delete`): CPU only, deletes `/mnt/proxy/feats/siglip2` on PVC `stacking-encoders-proxy` and nothing else.
2. `jobs/job-siglip2-reextract.yaml` (Job `stacking-encoders-pairings-siglip2-reextract`): 2 shards, one NVIDIA L4 each, recomputes the cache into the same folder with the same code, model and settings as the Job that wrote it.

**The submit needs the user's go in the lead's window.** The user's choice of option 1 is the decision; confirm the go for these two Jobs before step 4, stating: one CPU-only deletion Job (minutes; deletes about 3,400 files, about 250 GB, irreversibly), then one extraction Job with 2 L4 GPUs for about 2 to 2.5 hours once scheduled (about 4 to 5 L4 GPU-hours), then the cache check and the freshness check (one CPU Job, one L4 for about 10 minutes). Deleting PVC data needs the user's OK (`../../../_standards/08`); option 1 is that OK for this folder only.

## What stays the same, and what changes

The original is `../2026-10-27-proxy-calibration/jobs/rendered/job-extract-siglip2.yaml`, the manifest of SigLIP 2 try 5, which finished the cache on an L4 (nautilus-it-gpu05) on 2026-10-30 (`../../LOG.md` 2026-10-30 (c)), rendered from `../2026-10-27-proxy-calibration/jobs/job-extract.template.yaml` by that experiment's `RUN.md` step 6.

Same: image `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`; `setup_env.sh` (unchanged since `439e52f`, so the same pip pins); `extract_fastwam.py --source siglip2 --dtype fp16 --batch 32` over the four suites; the model `google/siglip2-so400m-patch14-384` at resolution 384 (from its processor); output folder `/mnt/proxy/feats/siglip2/<suite>/` with `ep<id>.npy`, `ep<id>.stats.npy` and `_meta.shard<i>of<n>.json`; CPU 3 / 3600m, `OMP_NUM_THREADS=3`, `MALLOC_ARENA_MAX=2`, memory 12Gi / 14400Mi, ephemeral storage, `/work`, `/dev/shm`, driver > 569, us-west, deadline 16 h, TTL.

The code is the current main, not `439e52f`. What changed in the code path since then (verified, `git diff 439e52f HEAD`): `extract_features.py` pins the Hugging Face revision of SigLIP 2 to `e8e487298228002f3d8a82e0cd5c8ea9c567f57f` (the repo head since 2025-02-21, before the caches; the freshness check's `[load]` line confirmed the loaded commit) and loads the full model before taking its vision tower, which is the same module as before; `extract_fastwam.py` gained `--stage-dir` (not used here, so the write path is the old one). `fastwam_data.py` and `setup_env.sh` are unchanged; `diagnostics.py` changed, but the extractor uses only its `git_commit`, which reads `SRC_GIT_COMMIT`. This is the code the freshness check ran (`check_cache_fresh.py` calls the same `encoder_episode` and loader), and it reproduced DINOv3 and C-RADIOv4 bit for bit.

Every changed line of `jobs/job-siglip2-reextract.yaml` against the original, with its reason:

| Changed line(s) | Reason |
|---|---|
| Header comment (8 lines, replaces 5) | Says what this Job is, the user's decision, and that it runs only after the deletion; the old header described template rendering for any source. |
| `name: stacking-encoders-pairings-siglip2-reextract` | New Job name (stage 2 prefix), so it cannot collide with the old Job name or its logs in the LOG. |
| `completions: 2`, `parallelism: 2` | 2 shards: about 2 to 2.5 h per pod instead of 4 to 5 h for one, with 2 GPUs (see "Shards"). |
| `backoffLimit: 2` | The template's rule, one retry per shard (`backoffLimit: __SHARDS__`). |
| `podFailurePolicy` (4 lines) | The refusals (exit 10 to 14, 19) fail the Job at once instead of using up retries; a retry would refuse again. |
| `kubernetes.io/hostname NotIn` (1 line) | Excludes the two hosts with GPU faults (2026-11-03), as the freshness check manifest does (PR #62 review). |
| `python -c ... cuda ... L4 ... exit 13` | After setup: refuses unless CUDA is available and the device name's last word is "L4" (PR #62 review; a plain substring test would also match "L40"), so a GPU fault cannot silently fall back to CPU. |
| `--device cuda` | The extractor's default is `cuda` only if available, else `cpu`; naming it makes a missing GPU an error instead of a CPU run (PR #62 review). |
| `python -c ... _meta ... exit 14` | After the four suites: this shard's four `_meta.shard<i>of2.json` files must exist and record `"device": "cuda"` (the extractor writes `args.device` there). |
| `values: [NVIDIA-L4]` (replaces the 3 lines of 10 card types) | L4 only, the point of the re-extraction; same label and value as the freshness check, which scheduled on an L4 (nautilus-it-gpu06). Label key per `_standards/08` "Choosing GPUs". |
| `echo "=== SigLIP 2 re-extraction ..."` | Logs the shard, Job uid and source commit at the top of each pod log. |
| `[[ "$SRC_GIT_COMMIT" =~ ... ]]` | Refuses an unrendered manifest (exit 10); the commit goes into every `_meta` file. |
| `[ -n "$JOB_UID" ]` | Refuses if the pod lacks the Job uid label the stale-file guard needs (exit 11). |
| `sha256sum -c` block (8 lines) | The five `/src` keys must equal the hashes rendered from `$C` (exit 19), so the code is provably the current main. |
| `S=...; M=...` and the `if ... fi` block (11 lines) | The stale-file guard: if no pod of this Job has started before, `feats/siglip2` must be empty or absent, otherwise exit 12 without writing anything. A retried pod of the same Job finds its Job's marker and resumes (the extractor skips episodes already on disk, as before). |
| `touch $M-shard$I.started` | Writes this pod's marker in `/mnt/proxy/logs/` (not in the cache folder), before any cache file. |
| `nvidia-smi --query-gpu ... -l 30 &`, `SMI=$!; trap ...` | A `[gpu]` line every 30 s for the 15-minute utilization check: the L4 nodes gpu05/06 export no DCGM (`../2026-10-27-proxy-calibration/LOG.md` 2026-11-07 (h)); the sampler is killed at exit. |
| `echo "=== $s shard $I/2"` | Shard count. |
| `--shard $I/2` | Shard count. Each episode is computed alone (batches of 32 frames within one camera of one episode), so sharding does not change any value. |
| `SRC_GIT_COMMIT: FILL_AT_SUBMIT` | Rendered from `$C` at submit (step 3), as the template did. |
| `JOB_UID` env (4 lines) | The Job's uid from the pod label `batch.kubernetes.io/controller-uid` (downward API), for the guard's marker. |
| `configMap: {name: stacking-encoders-pairings-reextract-src}` | A new ConfigMap from the current main; the old `stacking-encoders-proxy-src` held `439e52f` (no revision pin) and may no longer exist. |

Not changed on purpose: the log file name pattern (`extract-siglip2-<shard>-<time>.log` in `/mnt/proxy/logs/`; the timestamp keeps it apart from the October logs) and the deadline (16 h from Job creation; covers a long L4 queue plus a retry).

### Shards
Two. One shard would take about 4 to 5 hours of pod time, already inside half a day, but two halve the wall time and one node fault costs half as much. More than two would need more L4s free at once (L4 with driver > 569 in us-west matched only two nodes on 2026-11-09, memory note "Nautilus GPU availability check"), and each pod repeats the setup and the four per-suite model loads. With 2 GPUs, the GPU count stays at most 8 only if the stage 2 Jobs then running use at most 6 (step 1).

### GPU-hour estimate (arithmetic; basis stated)
- Frames: 277,713 over 1,712 episodes (`../../LOG.md` 2026-10-29 (e)), about 138,900 per shard.
- Rate: 21.1 frames/s overall for SigLIP 2 on an L4 with the prefetch thread, CPU 3 (try 5, `../../LOG.md` 2026-10-30 (c)). That try covered only the libero_10 episodes after ep 114 (the long ones), so it is the closest measurement but not one over all suites. Lower reference: 17 frames/s on an L4 without prefetch (the probe, 2026-10-29 (e)).
- Per pod: 138,900 / 21.1 = 1.83 h, or 2.27 h at 17 frames/s; plus setup (pip about 33 s, stage 1), the SigLIP 2 download (a few GB, once per pod into `/work`) and four model loads (about 50 s each, the probe's SigLIP 2 time with load), about 5 to 10 minutes.
- Total: about 4 to 5 L4 GPU-hours; wall time about 2 to 2.5 h after both pods are scheduled. The deletion Job: minutes of 0.5 CPU.
- CephFS write load: about 146 MB per episode (C-RADIOv4, same width, 251 GB over 1,712 episodes, 2026-10-30 status line) every 7 to 8 s per pod, about 20 MB/s per pod.

## Steps

1. **Before anything: what else is running.** `kubectl get jobs,pods -n aria-world`. Count our GPUs: these two pods plus the running ones must be at most 8 (`../../STANDARDS.md` gate). **No Job that reads `feats/siglip2` may run or be submitted from step 4 until step 9 passes:** stage 2 configurations `siglip2` and `dinov3+siglip2` (training or their smoke Jobs) and any rollout of them. A running one: stop and ask the user.
2. **Source commit and ConfigMap.** `$C` = `git rev-parse HEAD` on a clean, pulled `main` that contains this branch. From `jobs/`:
   ```
   S=../../../src
   kubectl create configmap stacking-encoders-pairings-reextract-src -n aria-world --from-file=extract_fastwam.py=$S/extract_fastwam.py --from-file=extract_features.py=$S/extract_features.py --from-file=fastwam_data.py=$S/fastwam_data.py --from-file=diagnostics.py=$S/diagnostics.py --from-file=setup_env.sh=$S/setup_env.sh --dry-run=server
   ```
   then the same without `--dry-run=server`. Read each key back against `$C`:
   ```
   for f in extract_fastwam.py extract_features.py fastwam_data.py diagnostics.py setup_env.sh; do
     a=$(kubectl get configmap stacking-encoders-pairings-reextract-src -n aria-world -o jsonpath="{.data.${f//./\\.}}" | shasum -a 256 | cut -c1-64)
     b=$(git show $C:world-model-stacking-encoders/src/$f | shasum -a 256 | cut -c1-64)
     [ "$a" = "$b" ] && echo "ok $f" || echo "DIFF $f"
   done
   ```
   Any `DIFF`: stop and ask. Check `git show $C:world-model-stacking-encoders/src/extract_features.py | grep e8e487298228002f3d8a82e0cd5c8ea9c567f57f` prints the SigLIP 2 pin.
3. **Render the extraction Job** (from this experiment folder):
   ```
   mkdir -p jobs/rendered; Y=jobs/rendered/job-siglip2-reextract.yaml
   sed "s/{name: SRC_GIT_COMMIT, value: FILL_AT_SUBMIT}/{name: SRC_GIT_COMMIT, value: $C}/" jobs/job-siglip2-reextract.yaml > $Y
   for f in extract_fastwam.py extract_features.py fastwam_data.py diagnostics.py setup_env.sh; do
     h=$(git show $C:world-model-stacking-encoders/src/$f | shasum -a 256 | cut -c1-64)
     sed "s/FILL_SHA_$f /$h /" $Y > $Y.tmp && mv $Y.tmp $Y
   done
   grep -c "value: $C}" $Y           # must print 1
   grep -c FILL_ $Y                  # must print 0
   grep -cE '^ +[0-9a-f]{64}  ' $Y   # must print 5
   ```
   Commit the rendered file as submitted. The deletion Job has no placeholders and is submitted as written.
4. **Deletion.** `kubectl create --dry-run=server -f jobs/job-siglip2-delete.yaml`, then `kubectl create -f jobs/job-siglip2-delete.yaml`. Note the time in the LOG. One check after about 10 minutes: `kubectl get job stacking-encoders-pairings-siglip2-delete -n aria-world` (cap 1 h). Save the log right away (`mkdir -p results/siglip2-reextract`; `kubectl logs <pod> -n aria-world > results/siglip2-reextract/<pod>.log`; pod name from `kubectl get pods -n aria-world -l job-name=stacking-encoders-pairings-siglip2-delete -o name`). Read: the per-suite counts (expected 434, 457, 433, 388 episodes with as many stats files, a `_meta` file per suite and try, maybe a few `tmp`), `df` before and after (about 250 GB freed, speculative: the SigLIP 2 cache size was not recorded, C-RADIOv4's same-width cache is 234 GiB), the other feats folders still listed after, and the last line `=== deleted: /mnt/proxy/feats/siglip2 is empty`. The full listing is on the PVC at `/mnt/proxy/logs/siglip2-delete-<time>.list`. Then `kubectl delete job stacking-encoders-pairings-siglip2-delete -n aria-world`. **Never resubmit the deletion Job once step 5 has created the extraction Job** (it would delete fresh L4 files while pods write them and race their folder creation); from then on any restart is the user's decision (stop conditions).
5. **Extraction, only after step 4 completed with exit 0.** Recount GPUs (step 1). `kubectl create --dry-run=server -f jobs/rendered/job-siglip2-reextract.yaml`, then `kubectl create -f jobs/rendered/job-siglip2-reextract.yaml`. Note the time. If the deletion did not finish, the pods refuse to start (exit 12) and the Job fails at once; that is the guard working, not something to work around.
6. **GPU-utilization check about 15 minutes after each pod starts** (one check per pod, no loop). Why: on 2026-10-29 the encoder extraction ran the GPU under 40% twice, first because torch sized its thread pool to the node and was throttled at its CPU limit (fixed by `OMP_NUM_THREADS` = the CPU request, (h), (k)), then because the serial per-episode loop starved a fast GPU (fixed by the prefetch thread, (m), 2026-10-30 (a)); on L4s the encoders then ran at 94 to 100% (2026-10-29 (k)). NRP flags a GPU pod under 40% of its request and emails after a trailing 3 h average under 40%; an admin delete counts as a strike, and the account is assumed at 1 of 3 (`../../../_standards/08` "GPU utilization"). The L4 nodes export no DCGM, so read the pod's own lines:
   ```
   kubectl get pods -n aria-world -l job-name=stacking-encoders-pairings-siglip2-reextract -o wide
   kubectl logs <pod> -n aria-world | grep -E '^=== |fresh start|resume|ABORT|: OK$|already done|frames/s|^\[gpu\]' | tail -40
   kubectl top pod <pod> -n aria-world
   ```
   Expect: five `: OK` hash lines; `fresh start` on the first pod (`resume` on the second if the first wrote its marker earlier); `0 of <n> episodes already done` on every suite reached so far (anything else on a first pass: stop); the nvidia-smi header naming an `NVIDIA L4`; `ep ...: (n, 14, 28, 1152)` lines near 17 to 21 frames/s overall; `[gpu]` samples mostly 90 to 100% once the first suite runs (the setup and download minutes read 0). CPU about 1 to 2 cores of 3 (the encoders needed about one core without prefetch, 2026-10-29 (k)); memory see "Stop conditions". Then one more look after about 2.5 h; checks at most every 20 to 30 minutes.
7. **When the Job completes**: both pods `Completed`, each log ending with `=== done` after libero_10. Save each pod's log at once (`kubectl logs <pod> -n aria-world > results/siglip2-reextract/<pod>.log`, and `kubectl get job stacking-encoders-pairings-siglip2-reextract -n aria-world -o yaml > results/siglip2-reextract/job-status.yaml`). Record in the LOG: nodes and cards, start and end times, frames/s, GPU-hours (actual against this estimate), the `[gpu]` summary. Delete the Job (graceful, never `--force`) and the ConfigMap.
8. **Cache check**, `RUN-cache-check.md` unchanged (`jobs/job-cache-check.yaml`). Expected for SigLIP 2: 1,712 episode files and stats files with the right lengths, no gap; `_meta` model `google/siglip2-so400m-patch14-384`, resolution 384, dtype fp16, now as `_meta.shard0of2.json` and `_meta.shard1of2.json` per suite. DINOv3, C-RADIOv4 and the VAE as before.
9. **Freshness check**, `RUN-cache-fresh-check.md` steps 2 to 9 unchanged, with `jobs/job-cache-fresh-check.yaml` and its tolerance, rendered to a new file (for example `jobs/rendered/job-cache-fresh-check-after-reextract.yaml`) so the committed render of the first run stays. Same seed, so the same 8 episodes. **Expected: `=== OK`, and SigLIP 2 bit-identical (max abs diff 0, bit-equal 100%) on all 8 episodes**, as DINOv3 and C-RADIOv4 were on 2026-11-18. Passing the tolerance without bit-identity is not the expected result: record it and stop for the user.

## Stop conditions (stop, record, ask the user; do not work around)
- A Job that reads `feats/siglip2` is running or queued (step 1).
- The deletion Job exits non-zero: 20 (path, mount or symlink check), 23 (entries outside the cache layout; nothing was deleted), 24 (entries left after the delete), or any other code. Do not delete by hand.
- The deletion's counts differ from 434 / 457 / 433 / 388 episodes per suite: record it (the files are already gone); step 8 shows whether the new cache is complete.
- Extraction exit 13 (no CUDA or not an L4: a node fault; ask before resubmitting), 14 (a `_meta` file missing or not `device: cuda`), 12 (old files in `feats/siglip2`), 10 or 19 (render or ConfigMap mismatch: redo steps 2 and 3), 11 (no Job uid label: the guard cannot work on this cluster version; ask).
- Any suite on a first pass reporting episodes `already done`.
- A pod on a card other than an L4 (should be impossible with the affinity).
- GPU under 40% in the pod's own samples over 10 minutes, after its first 30 minutes (NRP's start-up exemption): delete the Job (graceful) and tell the user; finished episodes stay valid L4 output, but a new Job refuses to start on them (exit 12), so restarting means deleting again (step 4, only after the extraction Job is deleted and its pods are gone, and only on the user's OK) or a guard change by PR.
- CPU at the limit with throttling, or memory working set under 20% of the 12Gi request (about 2.4 GiB) for a long stretch: report it. The memory is sized for the long libero_10 episodes with prefetch (an OOM at 9Gi there, 2026-10-30 (c)), so the earlier suites may sit near or under 20%; the October try 5 ran only libero_10, so this was not seen then. Resizing is resource-only (no change to the features) but needs a new Job, so it is the user's call.
- OOMKilled, or a pod failure beyond the one retry per shard (CUDA errors on a node have happened, 2026-10-30 (c)).
- Pending for more than about 2 h: ask. Do not widen the card list; L4 only is the point.
- The Job fails or hits its deadline partway: ask. A new Job refuses to start on the partial folder by design; the options are (with the extraction Job deleted, its pods gone, and the user's OK) to delete again and rerun (at most about 5 GPU-hours lost) or to change the guard by PR.
- Cache check gaps, or the freshness check not bit-identical for SigLIP 2 on all 8 episodes (or any DINOv3 or C-RADIOv4 change).
- Free space on the PVC after step 7 more than a few GB below the free space before step 4 (`df` lines of the deletion log and the cache check's `volume:` line).
