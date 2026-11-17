# Stage 2 cache pre-check: steps for the lead

Written 2026-11-17 by a worker, Mac only. Nothing here has been submitted. One CPU-only Job, `jobs/job-cache-check.yaml` (Job `stacking-encoders-pairings-cache-check`), runs `src/check_stage2_cache.py` read-only against PVC `stacking-encoders-proxy` (mounted `readOnly`). It checks, for DINOv3, SigLIP 2, C-RADIOv4 and the VAE cache and for each of the four LIBERO suites, that every episode in `feats/meta/<suite>.npz` has an `.npy` of length and shape (frames, 14, 28, D) float16 (VAE: (frames, 3, 14, 28, 48)) and, for the encoders, a `.stats.npy` (3, D) float64 with the expected count. It prints the `model`, `hf_id`, `resolution` and compute dtype of every `_meta.shard*.json`, and the PVC's free space. Paths are relative to this experiment folder; `$C` is the source commit. Code delivery follows the stage 1 Jobs (ConfigMap mounted at `/src`, `../2026-11-15-pretrained-start-baseline/jobs/RUN-full-train.md`).

**The submit needs the user's go in the lead's window** (new cluster work; an approval of these files is not a go). State before asking: one CPU-only Job, 500m CPU and 1Gi requested, no GPU, reads file headers only (the arrays are memory-mapped, a few thousand small reads), writes nothing, runs minutes, deadline 1 h.

## Steps

1. **Source commit** `$C` on a clean, pulled `main` that contains `src/check_stage2_cache.py`: `git rev-parse HEAD`.
2. **Create the ConfigMap** (one key, new name, created here; do not reuse the baseline's):
   ```
   kubectl create configmap stacking-encoders-pairings-cachecheck-src -n aria-world --from-file=check_stage2_cache.py=../../src/check_stage2_cache.py --dry-run=server
   kubectl create configmap stacking-encoders-pairings-cachecheck-src -n aria-world --from-file=check_stage2_cache.py=../../src/check_stage2_cache.py
   ```
   (run from `jobs/`; the path assumes that). Read it back and compare with `$C`'s file; a difference: stop and ask.
   ```
   kubectl get configmap stacking-encoders-pairings-cachecheck-src -n aria-world -o jsonpath='{.data.check_stage2_cache\.py}' | shasum -a 256
   git show $C:world-model-stacking-encoders/src/check_stage2_cache.py | shasum -a 256
   ```
3. **Server dry run, then create**: `kubectl create --dry-run=server -f jobs/job-cache-check.yaml`, then `kubectl create -f jobs/job-cache-check.yaml`. Note the creation time in the LOG.
4. **One check, no loop**: the Job should end within minutes. After about 10 minutes run `kubectl get job stacking-encoders-pairings-cache-check -n aria-world`; if it still shows active, check once more after 20 minutes (the Job cap is 1 h). Do not use `kubectl wait --for=condition=complete` alone: a gap makes the Job fail, and that wait would sit until its timeout. A failed Job is a result, not an error: read the log either way. A `kubectl` auth error: ask the user to re-login.
5. **Save the log right away** (container logs vanish within about a day), using the pod, not `job/`:
   ```
   kubectl get pods -n aria-world -l job-name=stacking-encoders-pairings-cache-check
   kubectl logs <pod> -n aria-world > results/cache-check/<pod>.log
   kubectl get job stacking-encoders-pairings-cache-check -n aria-world -o yaml > results/cache-check/job-status.yaml
   ```
6. **Read the log**: the per-source and suite blocks (episode counts, `provenance:` lines, up to 15 `GAP` lines each), the `volume:` line, the `=== summary` table, and the last line `=== OK: 0 gap(s)` or `=== FAIL: <n> gap(s)` followed by `=== exit <rc>`. Also compare the printed `sha256sum` with step 2. Check by hand: the SigLIP 2 `model` is the so400m variant (PLAN Open item f); the DINOv3 and C-RADIOv4 values match `data/README.md` section 4b (`facebook/dinov3-vitl16-pretrain-lvd1689m`, `nvidia/C-RADIOv4-SO400M`); every source reports one resolution; the VAE blocks may show no provenance (no `hf_id` in its meta) and that is not a gap. Free space against PLAN Open item g and Hand-off step 4.
7. **Exit codes**: 0 no gap; 1 a gap or an inconsistent or missing provenance file (the Job fails at once, no retry; the missing episodes are named in the log); 2 the script failed (a traceback, for example an unreadable meta file); a pod lost to a node reboot is retried by itself. Do not repair anything from this Job: it is read-only by design. A gap goes to the user with the log.
8. **Delete the Job** after the log is saved: `kubectl delete job stacking-encoders-pairings-cache-check -n aria-world` (graceful, never `--force`), and the ConfigMap if no longer needed: `kubectl delete configmap stacking-encoders-pairings-cachecheck-src -n aria-world`. Nothing on the PVC is touched.

## What the check does not do
It reads headers and stats files, not the feature values, so it does not catch corrupt contents inside a correctly shaped file (the memory-mapped header would still read fine). It does not compare the cache against the encoders it came from (the parity test, PLAN item 2). Extra files (for example a leftover `.tmp`) are not reported.
