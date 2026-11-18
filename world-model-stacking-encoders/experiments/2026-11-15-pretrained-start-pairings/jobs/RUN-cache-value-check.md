# Stage 2 cache value check: steps for the lead

Written 2026-11-18 by a worker, Mac only. Nothing here has been submitted. One CPU-only Job, `jobs/job-cache-value-check.yaml` (Job `stacking-encoders-pairings-cache-value-check`), runs `src/check_stage2_cache.py --values` read-only against PVC `stacking-encoders-proxy` (mounted `readOnly`), over all four sources (DINOv3, SigLIP 2, C-RADIOv4, VAE) and all four LIBERO suites. It does everything the header check does (`RUN-cache-check.md`) and then reads every episode file in full: for the encoders it recomputes each channel's count, sum and sum of squares and compares them with the `.stats.npy`; for every source (the VAE too) and the stats files it reports any all-zero 4096-byte page at a 4096-aligned offset; and it reports any non-finite value. Why: the 2026-11-18 cache check found 8 SigLIP 2 episodes written as full-size files of zeros during a CephFS pool-full event (LOG 2026-11-18), and the header check cannot see a zero stretch after a good header. Paths are relative to this experiment folder; `$C` is the source commit. Code delivery as in `RUN-cache-check.md` (ConfigMap mounted at `/src`), with a new ConfigMap name.

**The submit needs the user's go in the lead's window** (new cluster work; an approval of these files is not a go). State before asking: one CPU-only Job, 2 CPU and 2Gi requested, no GPU, reads the whole feature cache (about 756 GB) once and writes nothing, expected about 50 minutes if CephFS gives the 16-reader rate of the 2026-11-01 read bench (about 300 MB/s), about 1.5 h at half that, deadline 4 h. Its reads compete with any other Job reading `feats/` on the same CephFS (none should be running while `feats/siglip2` is on hold). The sizing and its assumptions are in the manifest header (Mac timing on synthetic files; CephFS speed not measured for this Job).

## Steps

1. **Source commit** `$C` on a clean, pulled `main` that contains `src/check_stage2_cache.py` with `--values`: `git rev-parse HEAD`; `python3.11 check_stage2_cache.py --help` from `src/` lists `--values`.
2. **Create the ConfigMap** (one key, new name; do not reuse `stacking-encoders-pairings-cachecheck-src`):
   ```
   kubectl create configmap stacking-encoders-pairings-valuecheck-src -n aria-world --from-file=check_stage2_cache.py=../../../src/check_stage2_cache.py --dry-run=server
   kubectl create configmap stacking-encoders-pairings-valuecheck-src -n aria-world --from-file=check_stage2_cache.py=../../../src/check_stage2_cache.py
   ```
   (run from `jobs/`; the path assumes that). Read it back and compare with `$C`'s file; a difference: stop and ask.
   ```
   kubectl get configmap stacking-encoders-pairings-valuecheck-src -n aria-world -o jsonpath='{.data.check_stage2_cache\.py}' | shasum -a 256
   git show $C:world-model-stacking-encoders/src/check_stage2_cache.py | shasum -a 256
   ```
3. **Server dry run, then create**: `kubectl create --dry-run=server -f jobs/job-cache-value-check.yaml`, then `kubectl create -f jobs/job-cache-value-check.yaml`. Note the creation time in the LOG.
4. **One check, no loop**: after about 60 minutes run `kubectl get job stacking-encoders-pairings-cache-value-check -n aria-world`. If it still shows active, check once more after another 60 minutes, then at most once per hour up to the 4 h deadline. Do not use `kubectl wait --for=condition=complete` alone: a gap makes the Job fail, and that wait would sit until its timeout. A failed Job is a result, not an error: read the log either way. A `kubectl` auth error: ask the user to re-login. Optional, once, about 15 minutes in: `kubectl top pod -n aria-world -l job-name=stacking-encoders-pairings-cache-value-check` (CPU use should be well above 0.4 cores, 20% of the 2 CPU request; far below that means CephFS is slow, record it).
5. **Save the log right away** (container logs vanish within about a day), using the pod, not `job/`:
   ```
   mkdir -p results/cache-value-check
   kubectl get pods -n aria-world -l job-name=stacking-encoders-pairings-cache-value-check
   kubectl logs <pod> -n aria-world > results/cache-value-check/<pod>.log
   kubectl get job stacking-encoders-pairings-cache-value-check -n aria-world -o yaml > results/cache-value-check/job-status.yaml
   ```
6. **Write the damaged-episode list** (the repair Job's input). Every episode with any gap, header or value, is printed once, uncapped, after the summary as `DAMAGED <source> <suite> <episode id>`:
   ```
   grep '^DAMAGED ' results/cache-value-check/<pod>.log | awk '{print $2, $3, $4}' > results/cache-value-check/damaged-episodes.txt
   wc -l < results/cache-value-check/damaged-episodes.txt
   grep '^=== damaged episodes' results/cache-value-check/<pod>.log
   ```
   The line count must equal the number on the `=== damaged episodes ... : <n>` line; if not, stop (the log is cut). The file has one line per episode, `<source> <suite> <episode id>` (for example `siglip2 libero_spatial 272`), and is written even when empty. The 8 SigLIP 2 libero_spatial episodes from 2026-11-18 (ep 272 to 286, even ids) are expected in it unless they were repaired first. Gaps that are not about one episode (a missing meta file, a provenance mismatch) have no `DAMAGED` line; they are `GAP` lines without an episode id.
7. **Read the log**: the per-source and suite blocks (episode counts, `provenance:` lines, `values: <k> of <n> episodes value-checked`, up to 15 `GAP` lines each), the `volume:` line, the `=== summary` table, the `values: ... GB read in ... s (... MB/s, 16 reader threads)` line, the `DAMAGED` list, the `[cgroup] memory.peak` line, and the last lines `=== OK: 0 gap(s)` or `=== FAIL: <n> gap(s)` followed by `=== exit <rc>`. Compare the printed `sha256sum` with step 2. What each value gap means:
   - `<k> all-zero 4096-byte page(s) in .npy, first at offset <o>` (or `in .stats.npy`): whole pages of zero bytes, the signature of a refused CephFS flush. Real float16 features never hold 2048 exact zeros in a row (speculative but safe: the intact synthetic and test caches have none).
   - `.stats.npy sum of squares differs` or `sum differs` (with the channel count and the largest difference against the fixed tolerance 1e-9): the data no longer matches the statistics written at extraction, for example a zero stretch that is not page-aligned, or a stats file from other data. Float64 summation-order noise is about 1e-15 (measured on synthetic files), far under the tolerance.
   - `.stats.npy count <c> but the .npy has <r> rows`: the stats and the data disagree on length.
   - `<k> non-finite value(s) in .npy`: inf or nan in the features.
   - `cannot read ... for the value check`: the file could not be read at all (permission or I/O error).
   A `values: <k> of <n>` line with k below n means k episodes had a readable float16 header of the right width; the others are named by their header `GAP` lines and were still scanned for zero pages.
   Record in the LOG: gaps per source and suite, the `DAMAGED` count, bytes read, seconds and MB/s (the first measured CephFS read rate for the whole cache, to correct the manifest header's projection), `memory.peak` and `anon` against the 2Gi request, the node, and estimated against actual run time.
8. **Exit codes**: 0 no gap; 1 a gap (the Job fails at once, no retry; the damaged episodes are in step 6's list) or an uncaught Python exception (a traceback in the log; tell these apart by the traceback and the missing `=== summary`); 2 a usage error (argparse); 137 an OOM kill (the memory limit was too small: record `memory.peak`, stop and ask before a resubmit with more); a pod lost to a node reboot is retried by itself. A deadline hit (`DeadlineExceeded` in `job-status.yaml`) means CephFS ran under about 55 MB/s: save the log and ask. Do not repair anything from this Job: it is read-only by design. A gap goes to the user with the log and the damaged-episode list.
9. **Delete the Job** after the log is saved: `kubectl delete job stacking-encoders-pairings-cache-value-check -n aria-world` (graceful, never `--force`), and the ConfigMap: `kubectl delete configmap stacking-encoders-pairings-valuecheck-src -n aria-world`. Nothing on the PVC is touched.

## What the check does not do
It checks that each file agrees with its own statistics and has no zero pages or non-finite values. It does not compare the cache against the encoders it came from (the freshness check, `RUN-cache-fresh-check.md`), so a file and stats both written from wrong features pass. The VAE cache has no stats files, so for the VAE only the zero-page and non-finite checks apply: a VAE file with a zero stretch that is not page-aligned and shorter than about two pages is not caught. A last partial page (under 4096 bytes at the end of a file) is not scanned for zeros; in encoder files the stats comparison covers it. Extra files (a leftover `.tmp`) are not reported.
