# Encoder cache freshness check: steps for the lead

Written 2026-11-17 by a worker, Mac only. Nothing here has been submitted. One Job, `jobs/job-cache-fresh-check.yaml` (Job `stacking-encoders-pairings-cache-fresh-check`), one L4, runs `src/check_cache_fresh.py` against PVC `stacking-encoders-proxy`, mounted read-only. The user OK'd pinning the DINOv3 and SigLIP 2 revisions and this check in chat on 2026-11-17 ("sure pin it that sounds fine"); that is not a go to submit. Paths are relative to this experiment folder; `$C` is the source commit. Code delivery follows stage 1 (ConfigMap at `/src`, hashes rendered into the manifest, `../2026-11-15-pretrained-start-baseline/jobs/RUN-full-train.md` steps 3-4).

**The submit needs the user's go in the lead's window.** State before asking: one Job, one L4 GPU, expected 10 to 20 min of pod time (capped at 1 h per pod, one retry, 6 h including the queue), reads 8 episodes of video and their cached features, writes nothing to the PVC.

## What it checks, in one paragraph
The stage 2 features at training time come from the caches (built 2026-10-29 to 2026-10-31 from the repos' heads), and at rollout from the encoders loaded live. The loaders now pin the Hugging Face revisions (`src/extract_features.py` `REVISIONS`: DINOv3 `ea8dc2863c51be0a264bab82070e3e8836b02d51`, SigLIP 2 `e8e487298228002f3d8a82e0cd5c8ea9c567f57f`, C-RADIOv4 `c0457f5dc26ca145f954cd4fc5bb6114e5705ad8`). The pins were chosen from the Hub commit history (no commit to either repo after 2025-08-19 and 2025-02-21, long before the caches), so they should be the weights that built the caches. This Job confirms it from the data: for each of the four suites it draws 2 cached episodes with a fixed seed (numpy `default_rng(0)`, the same episodes for all three encoders), decodes and resizes their frames as extraction did, recomputes the features with the same function (`extract_fastwam.encoder_episode`), batch (32) and compute dtype (DINOv3 fp32, SigLIP 2 and C-RADIOv4 fp16) as the extraction Jobs, and compares them with the cached `.npy`. It also checks that each loaded model really came from the pinned commit (`config._commit_hash`) and that every `_meta.shard*.json` in the cache names the same model id and dtype.

## Pass tolerance (fixed 2026-11-17, before any run; do not change it after seeing results)
Per episode and encoder, over the cached (frames, 14, 28, D) float16 array:
- `cos_min` >= **0.999**: the lowest per-frame cosine similarity between the recomputed and the cached grid (each frame flattened).
- `rel_max` <= **0.02**: the largest absolute difference over the episode, divided by the largest absolute cached value.

Every one of the 24 episode-encoder comparisons must pass, with no missing file, no shape mismatch and no provenance or revision mismatch. The defaults are in the script (`COS_MIN`, `REL_MAX`) and asserted by `src/test_check_cache_fresh.py`.

Why these values. What the recompute cannot reproduce exactly, and its expected size:
1. **Card and kernel differences.** The caches were written by GPU pods, mostly on L4s (the final DINOv3, C-RADIOv4 and SigLIP 2 pods all ran on nautilus-it-gpu05: `../../LOG.md` 2026-10-29 (m), its 2026-10-30 02:40 UTC status line, and 2026-10-30 (c)), but earlier SigLIP 2 tries wrote some episodes on an RTX 5000 Ada, an RTX 3090 and a stratus1 card (`../../LOG.md` 2026-10-29 (h), 2026-10-29 (m), 2026-10-30 (b)), and the DINOv3 and C-RADIOv4 Jobs had earlier tries too (2026-10-29 (h), (k)) (their cards were not checked for this file), so some cached episodes may come from other cards. Different cards and cuDNN/cuBLAS algorithm choices change the order of fp16 accumulation (and TF32 in convolutions on these cards). Speculative size: of the order of the batch-size effect measured on CPU in `src/test_feature_parity.py` (1e-3 to 4e-3 of the max abs value, `BATCH_REL` comment), not measured on GPUs.
2. **Batch effects.** None by construction: same batch size (32) and the same chunk boundaries per camera as extraction.
3. **float16 storage.** Both sides are rounded to float16: up to about 5e-4 of each value's own size per side.
4. **Frame decoding and resize.** Same code (`fastwam_data.decode_episode`, `to_224` on CPU) and the same pinned `av` and `torch` versions (`src/setup_env.sh`), so expected identical.

So the numerical noise is expected below about 5e-3 relative max difference and a cosine above 0.9999; 0.02 and 0.999 leave about a fourfold margin. A real weight change (a different checkpoint or fine-tune) moves the features by an amount comparable to their size, which would give cosines well below 0.99 (speculative: no real alternative checkpoint exists for these repos to measure it; the fake test in `src/test_check_cache_fresh.py` uses a 30% weight change and fails all episodes). A change smaller than the tolerance (for example the same weights re-saved at another precision) would pass; that is accepted, since it is at the size of the fp16 noise the training features already carry. DINOv3 is computed in fp32 and should sit far inside the tolerance.

**If an encoder lands between the noise level and the bar** (for example `cos_min` between 0.99 and 0.999, or `rel_max` between 0.02 and 0.1), or fails only on episodes written by a non-L4 card: do not loosen the bar. Stop, record the numbers, and ask the user.

## Steps

1. **The user's go**, after the statement above.
2. **Source commit** `$C` on a clean, pulled `main` that contains this branch: `git rev-parse HEAD`.
3. **Create the ConfigMap** (6 keys, new name). Run from `jobs/`:
   ```
   S=../../../src
   kubectl create configmap stacking-encoders-pairings-freshcheck-src -n aria-world --from-file=check_cache_fresh.py=$S/check_cache_fresh.py --from-file=extract_features.py=$S/extract_features.py --from-file=extract_fastwam.py=$S/extract_fastwam.py --from-file=fastwam_data.py=$S/fastwam_data.py --from-file=diagnostics.py=$S/diagnostics.py --from-file=setup_env.sh=$S/setup_env.sh --dry-run=server
   ```
   then the same without `--dry-run=server`. Read each key back and compare with `$C`:
   ```
   for f in check_cache_fresh.py extract_features.py extract_fastwam.py fastwam_data.py diagnostics.py setup_env.sh; do
     a=$(kubectl get configmap stacking-encoders-pairings-freshcheck-src -n aria-world -o jsonpath="{.data.${f//./\\.}}" | shasum -a 256 | cut -c1-64)
     b=$(git show $C:world-model-stacking-encoders/src/$f | shasum -a 256 | cut -c1-64)
     [ "$a" = "$b" ] && echo "ok $f" || echo "DIFF $f"
   done
   ```
   Any `DIFF`: stop and ask.
4. **Render** into `jobs/rendered/` (from this experiment folder):
   ```
   mkdir -p jobs/rendered; Y=jobs/rendered/job-cache-fresh-check.yaml
   sed "s/{name: SRC_GIT_COMMIT, value: FILL_AT_SUBMIT}/{name: SRC_GIT_COMMIT, value: $C}/" jobs/job-cache-fresh-check.yaml > $Y
   for f in check_cache_fresh.py extract_features.py extract_fastwam.py fastwam_data.py diagnostics.py setup_env.sh; do
     h=$(git show $C:world-model-stacking-encoders/src/$f | shasum -a 256 | cut -c1-64)
     sed "s/FILL_SHA_$f /$h /" $Y > $Y.tmp && mv $Y.tmp $Y
   done
   grep -c "value: $C}" $Y           # must print 1
   grep -c FILL_ $Y                  # must print 0
   grep -cE '^ +[0-9a-f]{64}  ' $Y   # must print 6
   ```
   Commit the rendered file as submitted.
5. **Server dry run, then create**: `kubectl create --dry-run=server -f jobs/rendered/job-cache-fresh-check.yaml`, then `kubectl create -f jobs/rendered/job-cache-fresh-check.yaml`. Check our running GPU count first (at most 8, `STANDARDS.md`). The Secret `stacking-encoders-hf` must exist (DINOv3 is gated). Note the creation time in the LOG.
6. **One check, no loop**: once the pod is Running, after about 15 minutes run `kubectl get job stacking-encoders-pairings-cache-fresh-check -n aria-world` and `kubectl logs <pod> -n aria-world | grep -E '^\[load\]|^\[suite\]|^\[ep\]|^=== |^\[gpu\]|ABORT|device' | tail -40` (pod name: `kubectl get pods -n aria-world -l job-name=stacking-encoders-pairings-cache-fresh-check -o name`). Expect three `[load]` lines whose `revision` and `loaded` hashes are the pins above, then `[ep]` lines with GPU near 100%. If still active, check once more after 20 minutes (pod cap 1 h). A failed Job is a result: read the log either way.
7. **Save the log right away** (container logs vanish within about a day), from the pod, not `job/`:
   ```
   mkdir -p results/cache-fresh-check
   kubectl logs <pod> -n aria-world > results/cache-fresh-check/<pod>.log
   kubectl get job stacking-encoders-pairings-cache-fresh-check -n aria-world -o yaml > results/cache-fresh-check/job-status.yaml
   ```
   The JSON summary is in the log after `=== summary` (the pod's `/work` copy goes with the pod).
8. **Read the result**: the last lines are `=== OK: ...` or `=== FAIL: ...` (with `PROBLEM` lines) and `=== exit <rc>`. Record in the LOG: the three `[load]` lines, the chosen episodes, the worst `cos_min` and `rel_max` per encoder (`per_source` in the summary), the node and card, pod time and GPU-hours.
9. **Delete the Job** after the log is saved: `kubectl delete job stacking-encoders-pairings-cache-fresh-check -n aria-world` (graceful, never `--force`), and the ConfigMap: `kubectl delete configmap stacking-encoders-pairings-freshcheck-src -n aria-world`. Nothing on the PVC is touched.

## Exit codes (`=== exit <rc>`; podFailurePolicy fails the Job at once on 2, 3, 10, 19, 124)

| rc | Meaning | What to do |
|---|---|---|
| 0 | all 24 comparisons inside the tolerance, provenance and loaded revisions match | log it; the pins stand |
| 3 | an episode out of tolerance, a cached file missing or of the wrong shape, a `_meta` mismatch, or a loaded commit that is not the pin | stop, tell the user with the log; do not re-extract or edit anything without their OK |
| 2 | the check script failed (traceback; for example a dataset path or a gated download refused) | read the traceback, fix by PR, ask before resubmitting |
| 124 | the check hit its 2,400 s timeout | read the log (slow CephFS reads?), ask the user |
| 10, 19 | commit not rendered, or a `/src` key differs from the rendered hash | re-read the ConfigMap (step 3), re-render (step 4) |
| 16 | pip setup failed or timed out (counts toward the one retry) | if both pods hit it, read `tail` of the pip log in the pod log |
| 1 | Python died before `main` ran, or `nvidia-smi` failed (a faulty GPU node; counts toward the one retry) | read the log |
| `DeadlineExceeded` | pod over 1 h, or Job over 6 h from creation (queue included) | ask the user |

## If it does not schedule
L4 with driver > 569 in us-west matched only two nodes on 2026-11-09 (8 h Pending then). If it is still Pending after about 2 h, ask the user whether to widen the card list to the 24 GB set the extraction Jobs used (`../2026-10-27-proxy-calibration/jobs/rendered/job-extract-dinov3.yaml`); a card other than an L4 adds card-to-card numerics against the mostly-L4 caches (tolerance reasoning, item 1), which the bar is meant to absorb.

## What the check does not do
It recomputes 8 of the 1,712 episodes, so it would catch a different model, not a single corrupted file (`check_stage2_cache.py` covers shapes and lengths for every file). It does not check the VAE cache or T5. It does not test the rollout preprocessing gap (`src/test_feature_parity.py`, KNOWN GAP).
