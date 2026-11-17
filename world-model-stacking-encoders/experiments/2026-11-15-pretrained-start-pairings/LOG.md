# LOG: stage 2, encoder pairings on the 8-layer pretrained-start model

Step-by-step record of running this experiment (design in PLAN.md, results in README.md).

## 2026-11-17 [agent, lead, implementation] (cache pre-check on the cluster: 0 gaps)

**Run** (on the user's go in chat, "go"; verified, `results/cache-check/stacking-encoders-pairings-cache-check-gp9n2.log`, `results/cache-check/job-status.yaml`). Code is from PR #54's branch at `a87e9e254b200ef72f3a4d2b65fb6a3c790fa242` (reviewed, not yet merged; RUN step 1 asks for main, and this is the one departure). ConfigMap `stacking-encoders-pairings-cachecheck-src` read back with sha256 `d9d3d9d9...9e76aa`, equal to the commit's file and to the pod's printed hash. Server dry run OK. Job `stacking-encoders-pairings-cache-check` created 22:50:18 UTC, ran 22:50:30 to 22:56:04 UTC on a CPU node, exit 0, Complete 1/1. RUN step 2's relative path was wrong (`../../src` from `jobs/`); the lead used the absolute path, and the fix went back to the worker on PR #54.

**Outcome** (verified): `=== OK: 0 gap(s)`. Every meta episode in all four suites (goal 433, spatial 434, object 457, libero_10 388) has a complete `.npy` and, for the encoders, a `.stats.npy`, for dinov3, siglip2, c-radiov4 and vae. Provenance: dinov3 `facebook/dinov3-vitl16-pretrain-lvd1689m`, 224, fp32; siglip2 `google/siglip2-so400m-patch14-384` (so400m, as chosen), 384, fp16; c-radiov4 `nvidia/C-RADIOv4-SO400M`, 448, fp16; vae `Wan2.2_VAE (FastWAM loader)`, bf16. One file and one resolution per encoder per suite. These match `data/README.md` section 4b. The PLAN's "Cache check, by reading only" is now verified on the cluster for presence, shapes, lengths and stats files (not values).

**Free space** (read by hand, Hand-off step 4): `volume: 834 GiB used, 190 GiB free of 1024 GiB`. At 40 GB per running job, that allows at most 4 concurrent training runs on this volume, before any second volume (arithmetic).

Job and ConfigMap deleted after the log was saved. Nothing on the PVC was written.

**Next:** the user's merge OK for PRs #54-#57; the planning amendment (PR #58) and then the `pairings_analysis.py` update; wave manifests.
