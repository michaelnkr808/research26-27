# LOG: stage 2, encoder pairings on the 8-layer pretrained-start model

Step-by-step record of running this experiment (design in PLAN.md, results in README.md).

## 2026-11-17 [agent, lead, implementation] (cache pre-check on the cluster: 0 gaps)

**Run** (on the user's go in chat, "go"; verified, `results/cache-check/stacking-encoders-pairings-cache-check-gp9n2.log`, `results/cache-check/job-status.yaml`). Code is from PR #54's branch at `a87e9e254b200ef72f3a4d2b65fb6a3c790fa242` (reviewed, not yet merged; RUN step 1 asks for main, and this is the one departure). ConfigMap `stacking-encoders-pairings-cachecheck-src` read back with sha256 `d9d3d9d9...9e76aa`, equal to the commit's file and to the pod's printed hash. Server dry run OK. Job `stacking-encoders-pairings-cache-check` created 22:50:18 UTC, ran 22:50:30 to 22:56:04 UTC on a CPU node, exit 0, Complete 1/1. RUN step 2's relative path was wrong (`../../src` from `jobs/`); the lead used the absolute path, and the fix went back to the worker on PR #54.

**Outcome** (verified): `=== OK: 0 gap(s)`. Every meta episode in all four suites (goal 433, spatial 434, object 457, libero_10 388) has a complete `.npy` and, for the encoders, a `.stats.npy`, for dinov3, siglip2, c-radiov4 and vae. Provenance: dinov3 `facebook/dinov3-vitl16-pretrain-lvd1689m`, 224, fp32; siglip2 `google/siglip2-so400m-patch14-384` (so400m, as chosen), 384, fp16; c-radiov4 `nvidia/C-RADIOv4-SO400M`, 448, fp16; vae `Wan2.2_VAE (FastWAM loader)`, bf16. One file and one resolution per encoder per suite. These match `data/README.md` section 4b. The PLAN's "Cache check, by reading only" is now verified on the cluster for presence, shapes, lengths and stats files (not values).

**Free space** (read by hand, Hand-off step 4): `volume: 834 GiB used, 190 GiB free of 1024 GiB`. At 40 GB per running job, that allows at most 4 concurrent training runs on this volume, before any second volume (arithmetic).

Job and ConfigMap deleted after the log was saved. Nothing on the PVC was written.

**Next:** the user's merge OK for PRs #54-#57; the planning amendment (PR #58) and then the `pairings_analysis.py` update; wave manifests.

## 2026-11-17 [agent, lead, implementation] (stage 2 code merged, PRs #54-#57)

On the user's "merge": PR #54 (cache pre-check, merge `a26f64a`), #55 (feature parity and the frame-resize gap test, `397f95e`), #56 (`pairings_analysis.py`, `cdd7941`) and #57 (training tests with sources and none invariance, `be174b0`). Worker worktrees and branches removed. The three rule answers are still flagged "pending the user" in `pairings_analysis.py` until planning's amendment (PR #58) merges and the script is updated by PR.

## 2026-11-18 [agent, lead, implementation] (second volume created; encoder pinning dispatched; PRs #59, #60 open)

**Second volume** (on the user's go in chat, 2026-11-17: second volume "okay sure yeah as long as it doesn't violate nautilus' cluster protocols", size 1Ti "yeah sure as long as its fine with nautilus' policies"). Policy check: `_standards/08` lines 120 and 130 (no per-volume or per-namespace CephFS cap in the NRP docs; creating a PVC is a normal member action; `rook-cephfs` is US West, the jobs' region; 6-month purge for unaccessed volumes). PVC `stacking-encoders-runs`, `rook-cephfs`, ReadWriteMany, 1Ti, manifest `jobs/pvc-runs.yaml`. Server dry run OK. Created 2026-11-18 about 00:23 UTC, Bound (volume `pvc-5b9badfe-4856-457a-ba16-d3ccb283130d`), verified with `kubectl get pvc`. Empty. Stage 2 Jobs use it via `render.sh --runs-pvc stacking-encoders-runs` (PR #60).

**Open PRs:** #59 (`pairings_analysis.py` implements amendment 2026-11-17 item 3, plus the user's two answers on review questions: keep the gate diagnosis over the 1% clean line, and a recorded-decision field for the LIBERO-Plus open case; lead rerun 87 tests OK). #60 (stage 2 Job templates, `render.sh`, RUN steps, wave plan; Opus review approve with fixes, fixes in `180e0e3` and `e993a8f`; lead dry run OK; nothing submitted).

**Encoder pinning** (user: "sure pin it that sounds fine"): a worker is pinning the DINOv3 and SigLIP 2 Hugging Face revisions (C-RADIOv4 is already pinned) to the cache-time revisions, and preparing a check Job that recomputes features for a few cached episodes and compares them to the cache. The check Job needs the user's go before submission.
