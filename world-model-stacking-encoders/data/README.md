# Data provenance: Stacking Encoders

One entry per dataset, as `../../_standards/03` asks: source, access date, version, license, filtering applied, where it is stored. Every fact below points to the file or LOG entry it came from. "Not recorded" means the repo does not say; where the repo names a source page, the entry says to check it.

**Raw data is not in git.** `data/raw/` and `data/processed/` are empty (only `.gitkeep`) and `.gitignore` excludes their contents and all `*.h5` files. Everything below lives on Nautilus volumes in namespace `aria-world` (or was downloaded inside a Job to local disk and discarded). Large features are never committed. Result JSONs and logs that came out of them are in the experiment `results/` folders.

Cluster volumes are not archival (CephFS is purged after 6 months unaccessed; `pvc-features-cephfs.yaml`, header comment, citing `../../_standards/08`). Treat the paths below as the last known state, not a guarantee.

## 1. LIBERO demonstrations, libero_goal (HDF5)

- **Source:** Hugging Face dataset `yifengzhu-hf/LIBERO-datasets`, files `libero_goal/*.hdf5` (`experiments/2026-10-18-extract-libero-goal/job.yaml` line 55, `experiments/2026-10-19-feasibility-tally/extract-job.yaml`). To check the dataset page, use the Hugging Face page for that repo id.
- **Access dates:** first local download 2026-10-17: one file, `turn_on_the_stove_demo.hdf5`, for a smoke test (`../LOG.md` 2026-10-17 (c)). 2026-10-18 was only the Hugging Face API size check (6.37 GB, 10 files; extraction README, "What runs"). The extraction Job downloaded the suite on 2026-10-19 (Job ran about 06:53 to 07:47Z, extraction README, Results). Each later Job downloaded it again, with dates taken from the pod logs in `results/`: pilot diagnostics 2026-10-19 (10:24Z); tally extraction tries 2026-10-20; D+L/R and D+G diagnostics 2026-10-20; D+S diagnostics 2026-10-21; target-side Job 2026-10-27 (their `job.yaml` files). One file was also downloaded to a session scratchpad, not the repo, on 2026-10-27 (`../LOG.md` 2026-10-27 (g)).
- **Version:** Hugging Face revision `main` in every Job. No commit hash was recorded (`snapshot_download(..., revision="main")` in the Job yamls above). Size: 10 files, 6.37 GB (`experiments/2026-10-18-extract-libero-goal/README.md`).
- **License:** not recorded for this dataset. Check the dataset card on Hugging Face (repo id above). The related Fast-WAM re-render of LIBERO (section 3) is recorded as CC BY 4.0 (`experiments/2026-10-27-proxy-calibration/PLAN.md`, "Step 1 findings"); that does not by itself say anything about this HDF5 copy.
- **Layout** (verified on one file, `../LOG.md` 2026-10-17 (c)): `data/demo_k/obs/{agentview_rgb,eye_in_hand_rgb}`, 128x128 uint8, `macros_image_convention: opengl`.
- **Contents** (verified on one file, `turn_on_the_stove_demo.hdf5`, `../LOG.md` 2026-10-27 (g)): 50 demos of length 75 to 119 (median 89); per step, agentview and wrist RGB at 128x128, a 7-d action (3 translation, 3 rotation, gripper), reward 1 on the last step only, one language instruction per file. Whole suite: 10 task files, 500 demos, 63,728 timesteps, 127,456 frames over 2 cameras (`experiments/2026-10-18-extract-libero-goal/README.md`, Results).
- **Filtering applied:** no demos or frames dropped (all 10 files, every frame). Images are rotated 180 degrees (the HDF5 is stored in OpenGL orientation; `../STANDARDS.md`, Known pitfalls; extraction README, "What runs"). The split is made once, by demo, per task file: last 20% of demos test, previous 20% validation, rest train, giving 38,464 / 12,416 / 12,848 frames and 100 test demos (`experiments/2026-10-19-feasibility-tally/README.md`, Config). Frames within a demo are never split across sets.
- **Stored:** not stored on a volume. It is downloaded into the Job's local disk each time (`/work/data` in the Job yamls) and discarded with the pod.

## 2. LIBERO-Plus (benchmark and `task_classification.json`)

- **Source:** GitHub `sylvestf/LIBERO-plus` at commit `4976dc3` (dated 2026-01-21), read 2026-10-27; assets zip from Hugging Face `Sylvest/LIBERO-plus` (`experiments/2026-10-27-proxy-calibration/PLAN.md`, "Step 1 findings"; `src/setup_sim.sh` line 4).
- **Access date:** repo read 2026-10-27 (PLAN, "Step 1 findings"). The simulator was installed per pod inside Jobs from 2026-10-29 (`../LOG.md` 2026-10-29 (b) to (c)).
- **Version:** LIBERO-plus `4976dc3`. The Hugging Face assets repo has no revision recorded.
- **License:** the GitHub repo has no LICENSE file (GitHub reports none); the Hugging Face assets repo is tagged MIT. Treated as unstated for the code: evaluation use only, nothing redistributed (`experiments/2026-10-27-proxy-calibration/PLAN.md`, `src/setup_sim.sh` line 4, `../LOG.md` 2026-10-27 (n)).
- **Contents** (verified from `task_classification.json`, PLAN "Step 1 findings" and `../LOG.md` 2026-10-28 (c)): 10,030 episodes, 1 trial each, 4 suites x 7 perturbation categories = 28 cells. By category: camera 1,599, robot initial state 1,550, language 1,537, light 1,142, background 1,076, noise 1,601, layout 1,525.
- **`task_classification.json`:** from `libero/libero/benchmark/` in the LIBERO-plus repo at `4976dc3`. sha256 `faa87cce3e3ba434da01df7c77523a391b5f2912e4774330b0aa1be5f6a999e6`, checked on the user's download (`experiments/2026-10-27-proxy-calibration/LOG.md` 2026-11-06 (d), item 2; the same hash was seen 2026-10-28, `../LOG.md` 2026-10-28 (c)). The file itself is not committed (no LICENSE upstream). The user downloaded it after the 2026-11-06 lead entry said it was missing (`../LOG.md` 2026-11-06 [agent, lead]); where the user's copy is kept is not recorded. The full-benchmark episode list built from it is committed: `experiments/2026-10-27-proxy-calibration/results/full-benchmark/full-subset.json` (10,030 episodes, subset_id `91072e5da8499c99`, LOG 2026-11-06 (d)).
- **Filtering applied:** none to the benchmark itself. Evaluation uses the full 10,030 episodes under eval seeds 41, 42, 43 (`experiments/2026-10-27-proxy-calibration/PLAN.md`, Amendment 2026-11-06). Pilot runs used a seeded, stratified subset of 100 episodes (`results/pilot/pilot-subset.json`, `../LOG.md` 2026-10-31 (b)). Subset building is `src/lp_eval.py`.
- **Stored:** not a dataset on a volume. Assets are put on the proxy volume `stacking-encoders-proxy` by a simulator Job (`experiments/2026-10-27-proxy-calibration/jobs/job-sim.yaml`). The extracted instruction list `lp_instructions.txt` and `lp_tasks.json` are at `/mnt/proxy/` on that volume (`../LOG.md` 2026-10-29 (b) to (c)).

## 3. Fast-WAM LIBERO dataset (training data of the proxy round)

Not asked for by name, but it is the data the proxy-policy features come from, so it is recorded here.

- **Source:** Hugging Face dataset `yuanty/LIBERO-fastwam`, the data behind the paper's 1,712 training episodes (`experiments/2026-10-27-proxy-calibration/PLAN.md`, "Step 1 findings").
- **Access date:** download Job ran 2026-10-29 02:52 to 03:23 UTC (`../LOG.md` 2026-10-29 (a)).
- **Version:** revision `ee018b997c430bb12b5bf3c892d744798c5a2f91` (`experiments/2026-10-27-proxy-calibration/jobs/job-download.yaml` line 42). Downloaded by Job `job-download.yaml`; the `lerobot_v30` part is 4.71 GB (`../LOG.md` 2026-10-28 (f), second entry of that letter).
- **License:** CC BY 4.0 as a derivative of LIBERO; Fast-WAM code is MIT (PLAN "Step 1 findings").
- **Contents:** 434 + 457 + 433 + 388 = 1,712 episodes over four suites, 277,713 frames at 20 fps, regenerated with MuJoCo 3.3.2.
- **Filtering applied:** by the dataset's authors ("no_noops" suites; the exact filter is not documented in the Fast-WAM repo; PLAN "Step 1 findings"). None added by us. Frames are already upright, so the extractor's 180 degree rotation is off for this data (same PLAN section; `../LOG.md` 2026-10-28 (b)).
- **Stored:** PVC `stacking-encoders-proxy` at `/mnt/proxy/data/lerobot_v30/<suite>_no_noops_lerobot` (`experiments/2026-10-27-proxy-calibration/jobs/rendered/job-extract-dinov3.yaml`).

## 4. Cached frozen features

All features are produced by `src/extract_features.py` (libero_goal) or `src/extract_fastwam.py` (proxy round) from the data above. The source commit is written into each output file's metadata (`SRC_GIT_COMMIT`).

### 4a. libero_goal features (feasibility tally and target-side diagnostics)

Layout: `<volume>/stacking-encoders/features/<encoder>/libero_goal/*.h5`, one file per task. Both cameras, every frame, rotate-180 on, 4x4 grid + mean + global features (`experiments/2026-10-18-extract-libero-goal/README.md`).

| Encoder (dir name) | Variant and revision | Dtype | Size | Source commit | Volume and path | Made by |
|---|---|---|---|---|---|---|
| `siglip2` | `google/siglip2-large-patch16-384`, 384 px | fp16 | 4.4 G | `f4a7b82` | PVC `michael-persistent` (20Gi, RBD, ReadWriteOnce), `/persistent/stacking-encoders/features/siglip2` | `experiments/2026-10-18-extract-libero-goal/` |
| `c-radiov4` | `nvidia/C-RADIOv4-SO400M`, remote code `c0457f5` | fp16 (autocast) | 5.2 G | `f4a7b82` | same volume, `.../features/c-radiov4` | same |
| `dinov3` | `facebook/dinov3-vitl16-pretrain-lvd1689m` @ `ea8dc28`, 224 px | fp32 | 4.4 G | `0cd09b5` | PVC `stacking-encoders-features` (CephFS, 100Gi, ReadWriteMany; `pvc-features-cephfs.yaml`), `/mnt/features/stacking-encoders/features/dinov3` | `experiments/2026-10-19-feasibility-tally/` |
| `sam3` | `facebook/sam3` @ `3c879f3`, ViT backbone before the FPN neck, 1008 px | fp16 | 4.4 G | `9e3bdb2` | same CephFS volume, `.../features/sam3` | same |
| `vggt-omega` | `facebook/VGGT-Omega` `vggt_omega_1b_512.pt` @ `1041e80`, final aggregator layer, 512 px | fp16 (autocast) | 8.8 G | `0cd09b5` (first 5 files) and `9e3bdb2` (last 5) | same CephFS volume, `.../features/vggt-omega` | same |

Sources for the table: `experiments/2026-10-19-feasibility-tally/README.md` (Config, Failures and resource use), `../LOG.md` 2026-10-19/20 (f) and 2026-10-21 (g), and the Job yamls in those experiment folders. The D, S, G features total 17.6G on the CephFS volume (`../README.md`, Key files). The two extraction commits for VGGT-Omega gave identical features in a local check (max difference 0.0; tally README).

Encoder licenses and gating are in `../literature/encoder-weights.md` (DINOv3, SAM 3, VGGT-Omega are gated, access granted to the user 2026-10-19; `../LOG.md` 2026-10-19 (c)). DINOv3 was run in fp32 because fp16 overflowed (`../LOG.md` 2026-10-19/20 (f)).

Readers of these caches: `src/diagnostics.py`, `src/target_diagnostics.py`. Local checks used 3 to 12 demos copied to a session scratchpad, not stored in the repo.

### 4b. Proxy-round caches (all four LIBERO suites, Fast-WAM data)

Volume: PVC `stacking-encoders-proxy` (CephFS, `rook-cephfs`, ReadWriteMany, 1Ti; `experiments/2026-10-27-proxy-calibration/jobs/pvc-proxy.yaml`), mounted at `/mnt/proxy`. Features under `/mnt/proxy/feats/` (`experiments/2026-10-27-proxy-calibration/jobs/rendered/job-extract-dinov3.yaml`: `F=$R/feats`). All caches are stored and loaded as float16 (the `--dtype` flag in the table below is the compute dtype). Sizes from `du -sh` after extraction (GiB, as GB in the LOG): DINOv3 208G = 223 GB (`../LOG.md` 2026-10-30 (b)), C-RADIOv4 234G = 251 GB (`../LOG.md` 2026-10-30, the 03:05 UTC status line under 2026-10-29 (m)), SigLIP 2 234G = 251 GB (`../LOG.md` 2026-10-31 (a), SigLIP 2 step 6 checks line). VAE: 31.4 GB by arithmetic (proxy PLAN, "Storage" correction; it replaced the earlier 10.5 GB figure); a `du` for the VAE cache is not recorded.

| Cache | Encoder variant | Source commit in the rendered Job | Status |
|---|---|---|---|
| `feats/dinov3` | DINOv3, computed fp32, stored float16 (T, 14, 28, 1024) (`../LOG.md` 2026-10-30 (b)) | `2d0bcbd` (`job-extract-dinov3.yaml`) | done (experiment README status 2026-11-06: "all feature caches done"; `../LOG.md` 2026-11-01 (d) for the VAE and SigLIP 2 checks) |
| `feats/c-radiov4` | C-RADIOv4, computed fp16, stored float16 (T, 14, 28, 1152) (`../LOG.md` 2026-10-30, 03:05 UTC status line) | `2d0bcbd` | done |
| `feats/siglip2` | SigLIP 2 so400m, computed fp16, stored float16 (T, 14, 28, 1152) | `439e52f` (`job-extract-siglip2.yaml`). The OOM resubmits changed only memory and environment (`../LOG.md` 2026-10-30 (c)) and `../LOG.md` 2026-10-31 (b) records "src unchanged since 439e52f" | done, spot-checked (`../LOG.md` 2026-11-01 (d)) |
| `feats/vae`, `feats/vae_b` | Wan 2.2 VAE latents (48 ch), computed bf16, stored float16 (length, 3, 14, 28, 48) (`../LOG.md` 2026-11-01 (d)) | `2d0bcbd` (most of `feats/vae`, written 2026-10-29, `../LOG.md` 2026-10-29 (j) and (k)); `439e52f` (rest, plus `feats/vae_b`; its output was byte-identical to the earlier code, `../LOG.md` 2026-10-30 (a)); `dc30417` (shard 3, `job-extract-vae-s3.yaml`) | done; every episode in meta for all four suites (434, 457, 433, 388), checked 2026-11-01 (`../LOG.md` 2026-11-01 (d)) |
| `feats/t5` | umT5-XXL prompt embeddings | `2d0bcbd` (`../LOG.md` 2026-10-29 (g); `experiments/2026-10-27-proxy-calibration/jobs/rendered/job-t5.yaml` line 57) | done; 1,549 unique prompts, 1,550 files incl. `_meta.json` |

One known leftover: `feats/vae/libero_goal/.ep000159.npy.tmp`, harmless, not deleted (`../LOG.md` 2026-11-01 (d)). Training checkpoints and rollout records are also on this volume (PVC comment, `pvc-proxy.yaml`) and are listed in the proxy experiment's LOG, not here. The model weights downloaded for the proxy round (Wan 2.2 VAE, umT5-XXL, sha256-checked) are described in `experiments/2026-10-27-proxy-calibration/jobs/job-download.yaml`.
