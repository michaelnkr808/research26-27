# Phase A cluster Jobs (perturbed-frame test, redesign of Amendment 2026-11-09)

Written 2026-11-09 by the implementation worker pf-jobs-a. Base: main at 50b4905. **Nothing here has been submitted, no ConfigMap exists, and no kubectl command was run.** Every cluster step below needs the user's separate go in the session that submits it. Labels: `verified` (read or run in a repo file or script this session), `recalled`, `speculative` (arithmetic or guess). Numbers marked speculative were not measured on these Jobs.

## 1. What is here

| # | File | GPU | What it does | Needs |
|---|---|---|---|---|
| 1 | `job-pfa-render.yaml` | 1 L4 | Restore env, download the ten raw files, `pf_rerender.py render --split trainval --condition clean`, report-only `compare --demos val` against the pf-val render, publish render + report to the PVC, DINOv3 extraction in the same pod | nothing |
| 2 | `job-pfa-ext-siglip2-cradio.yaml` | 1 | SigLIP 2, then C-RADIOv4 (transformers 4.57.6 pins) | Job 1 finished with exit 0 |
| 3 | `job-pfa-ext-vggt.yaml` | 1 | VGGT-Omega (transformers 5.18.0 pins) | Job 1 |
| 4 | `job-pfa-ext-sam3-1.yaml`, `-2.yaml`, `-3.yaml` | 1 each | SAM 3 (5.18.0), three disjoint file shards | Job 1 |
| 5 | `job-pfa-dev.yaml` | 0 (CPU) | `pf_analysis.py dev` (train and val only, gate A, lag check) | Jobs 1 to 4 all finished with exit 0 |
| 6 | `job-pfa-pilot.yaml` | 1 L4 | OPTIONAL, the user decides: `render --pilot` for camera, light, background, noise on one val demo per task | nothing (independent of Job 1) |

Config: `../config/phase-a.yaml`. Jobs 2 to 5 do not wait on Job 1 inside the pod; each checks at the start for `/mnt/proxy/pf-main/phase-a/render/RENDER_COMPLETE` (written by Job 1 only after the render, its hashes and the verified copy are done) and aborts with exit 20 if it is missing. Job 5 additionally requires all 50 phase-A feature files (5 encoders x 10 stems).

Jobs counted for the 20-Job gate (`../../../STANDARDS.md`): phase A is 8 manifests (7 without the pilot). Phase B was estimated at 9 to 13 Jobs (PLAN Amendment 2026-11-09 section 8), so phase A + B can reach 15 to 21. Ask the user before phase B if the running total could pass 20 (whether earlier Jobs, such as the smoke test and pf-val, count toward the same sweep is not decided).

## 2. ConfigMap (not created; run by the lead after the user's go)

New ConfigMap `stacking-encoders-pf-main-src`, mounted at `/pf` (the existing `stacking-encoders-proxy-src` stays at `/src`: `diagnostics.py`, `extract_features.py`, `env_bundle.sh`, `setup_env.sh`, `setup_sim.sh` come from it; created per `../../2026-10-27-proxy-calibration/RUN.md` step 3). `diagnostics.py` is also put into `/pf` (same file) because `pf_rerender.code_hashes()` hashes the four code files from its own directory only, so without this copy `diagnostics.py` would be missing from the render report's `code_sha256` (verified, `src/pf_rerender.py:130-132`). Commit first, then, from the repo root `<repo>` = `/Users/michaelr808/Desktop/research/research26-27`:

```
kubectl create configmap stacking-encoders-pf-main-src -n aria-world --request-timeout=30s --from-file=pf_render.py=<repo>/world-model-stacking-encoders/src/pf_render.py --from-file=pf_rerender.py=<repo>/world-model-stacking-encoders/src/pf_rerender.py --from-file=pf_smoke.py=<repo>/world-model-stacking-encoders/src/pf_smoke.py --from-file=pf_analysis.py=<repo>/world-model-stacking-encoders/src/pf_analysis.py --from-file=diagnostics.py=<repo>/world-model-stacking-encoders/src/diagnostics.py --from-file=phase-a.yaml=<repo>/world-model-stacking-encoders/experiments/2026-11-06-perturbed-frames/config/phase-a.yaml --from-file=smoke-report.json=<repo>/world-model-stacking-encoders/experiments/2026-11-06-perturbed-frames/results/smoke/report.json --from-file=<repo>/world-model-stacking-encoders/experiments/2026-10-19-feasibility-tally/results/tally-libero_goal-dinov3-c-radiov4.json --from-file=<repo>/world-model-stacking-encoders/experiments/2026-10-19-feasibility-tally/results/tally-libero_goal-dinov3-sam3.json --from-file=<repo>/world-model-stacking-encoders/experiments/2026-10-19-feasibility-tally/results/tally-libero_goal-dinov3-siglip2.json --from-file=<repo>/world-model-stacking-encoders/experiments/2026-10-19-feasibility-tally/results/tally-libero_goal-dinov3-vggt-omega.json
```

Size against the 1 MiB limit (verified, `wc -c`): pf_render 27,665 + pf_rerender 42,142 + pf_smoke 26,842 + pf_analysis 58,612 + diagnostics 15,262 + config 2,142 + smoke report 43,516 + four tally JSONs 31,725 = 247,906 bytes, about 24% of the limit. `pf_validate.py` is not included (nothing here imports it).

The `/pf` keys are the only code the Jobs run besides `/src/extract_features.py`. Never edit `src/` between the ConfigMap and the commit hash in `SRC_GIT_COMMIT`; if a source file changes, delete and recreate the ConfigMap and use the new commit. Check after creating it: `kubectl get configmap stacking-encoders-pf-main-src -n aria-world -o json`, then compare each key's sha256 with the repo file. Phase B should mount a ConfigMap with these same files (same `pf_render.py`, `pf_rerender.py`, `pf_smoke.py`, `diagnostics.py`, `phase-a.yaml`) so that the render report's code hashes and settings hash agree.

`/src` check (the lead, before submitting): `stacking-encoders-proxy-src` must hold `extract_features.py` with sha256 prefix `03734ba84a86` (every extraction Job aborts with exit 21 otherwise) and `setup_env.sh`, `setup_sim.sh`, `env_bundle.sh` whose concatenation hashes to the existing bundle's key. The repo copies give key `9182f8131559` (verified, `cat setup_env.sh setup_sim.sh env_bundle.sh | shasum -a 256`), the same key the pf-val log shows (`/mnt/proxy/env/env-9182f8131559.tar`); the render Job aborts with exit 13 if `/mnt/proxy/env/env-<key>.tar` is missing for the ConfigMap's scripts.

## 3. Filling in SRC_GIT_COMMIT and submitting (the lead, after the go)

Each manifest has `SRC_GIT_COMMIT: FILL_AT_SUBMIT`; the script aborts with exit 11 if it is still that. Render at submit time and keep the rendered copies (as RUN.md does), do not edit the originals: `sed "s/FILL_AT_SUBMIT/<sha>/" jobs/job-pfa-render.yaml > jobs/rendered/job-pfa-render.yaml`, then `kubectl apply -f jobs/rendered/job-pfa-render.yaml`. `<sha>` is the commit that holds the ConfigMap sources. Record it in the experiment README (08 checklist item 5).

## 4. Order and stagger (shared cap: 8 GPUs at once in the namespace)

State at the lead's last read-only check, about 06:55 UTC 2026-11-09 (corrected from the first brief): 4 proxy eval pods Running (seed 0 shards on s0-l4 indices 1, 2, 3; seed 1 shard 2 on s1-l4 index 0). They are expected to finish later on 2026-11-09 (speculative). Recount the live GPU pods immediately before every submission; this plan only holds if the number below is true then.

GPU use of each step: Job 1 holds 1 for about 1.5 h (speculative), Job 2 holds 1 for about 0.8 h, Job 3 holds 1 for about 2.4 h, each SAM 3 shard holds 1 for about 1.9 h, pilot holds 1 for about 0.4 h, dev holds 0.

**Plan with the 4 evals still running (4 + phase A never above 8):**

| Step | Submit | GPUs by us | Total with 4 evals |
|---|---|---|---|
| 0 | (optional) pilot, once the user has said it runs | 1 | 5 |
| 1 | Job 1 (render), alongside the pilot if that runs | 1 (+1 pilot) | 5 (6 with pilot) |
| 2 | after Job 1 exits 0: Job 2, Job 3, SAM 3 shard 1, SAM 3 shard 2 | 4 | 8 |
| 3 | SAM 3 shard 3 when any of Jobs 2, 3, shard 1 or 2 has finished, or when an eval has finished | 1 | at most 8 |
| 4 | Job 5 (dev) after Jobs 1 to 4 have all exited 0 | 0 | unchanged |

Do not hold the render alone: the pilot may overlap with it because both are short, but if the count would pass 8 at any point, wait.

**If the evals are done (0 held):** after Job 1 exits 0 submit Jobs 2, 3 and the three SAM 3 shards together (5 GPUs, plus the pilot if still running: 6), then Job 5. Total never above 6.

**If some evals are done:** use the same table and add one extra step-2 or step-3 submission for each GPU that is free, keeping the sum of live GPU pods (evals + ours) at 8 or less.

Queue risk: the first pf-val submission sat Pending the whole 12 h (verified, direction LOG 2026-11-08). The Jobs carry a 48 h deadline from creation (queue included), which the user approved for pf-val; the work itself is capped stage by stage inside (section 7).

**Per Job type, check GPU use about 15 minutes into the extraction stage, and the Violations page 30 to 40 minutes after launch** (08 checklist item 8; pods under 30 minutes are not checked by the portal, the judgement is on hour-long averages):
- Job 1: the render and the restore are CPU-heavy and low in GPU use; check at the DINOv3 stage (about 1 h in, see the log line `=== compare` / `publish-render` followed by `extract-dinov3`), not at 15 minutes. Averaged over the whole pod the GPU may sit under 40% (speculative: restore, download, hash and render, about 40 to 60 minutes, followed by about 30 minutes of DINOv3). This is the cost of keeping the render and its extraction in one pod that the PLAN accepted; if the 3 h warning mail or the Violations page flags it, tell the user.
- Job 2 (first of the extraction-only type, with C-RADIOv4 and SigLIP 2): check about 15 minutes after the log shows `extract-siglip2` started; the earlier proxy extraction was CPU-throttled at 1 core (proxy LOG, 2026-10-29), so this Job asks for 2 CPU.
- Job 3, first SAM 3 shard: check once each. Later shards need only the Violations page.
- Use the namespace GPU dashboard (08) and the Violations page https://nrp.ai/userinfo. Delete finished Jobs afterwards (08 checklist item 9) after copying the reports.

## 5. Extraction flags next to the reference metas (must match exactly)

The reference metas are `results/smoke/report.json`, `q23.details.<encoder>.meta` (verified). `META_KEYS` (`src/pf_analysis.py:116`) = encoder, hf_id, resolution, rotate180, dtype, grid, frame_stride, cams. `extract_features.py` defaults: `--rotate180` on, `--grid 4`, `--frame-stride 1`, `--cams agentview_rgb eye_in_hand_rgb`, resolution from the encoder class (verified, `src/extract_features.py`). No Job overrides these four. Each Job checks the written files' metas against the reference with `pf_analysis.check_meta` before copying them to the PVC, and fails the stage (nothing copied) on any mismatch.

| Encoder | Flags used in the Job (after `--out /work/feats --hdf5 /work/pfa-trainval/libero_goal/*.hdf5`) | Reference meta (verified) | Pins and source of the flags |
|---|---|---|---|
| dinov3 | `--encoder dinov3 --dtype float32 --batch-size 64` | hf_id facebook/dinov3-vitl16-pretrain-lvd1689m, resolution 224, rotate180 true, dtype float32, grid 4, frame_stride 1, cams both | transformers 5.18.0 in `/work/extlib` (pf-val's pins, revision asserted `ea8dc28`); flags as pf-val and the tally |
| siglip2 | `--encoder siglip2 --variant google/siglip2-large-patch16-384 --dtype float16 --batch-size 64` | hf_id google/siglip2-large-patch16-384, resolution 384, dtype float16, rest as above | transformers 4.57.6 etc. from `2026-10-18-extract-libero-goal/job.yaml` |
| c-radiov4 | `--encoder c-radiov4 --dtype float16 --batch-size 64` | hf_id nvidia/C-RADIOv4-SO400M, resolution 448, dtype float16 | same pins as siglip2 |
| vggt-omega | `--encoder vggt-omega --dtype float16 --batch-size 16` | hf_id facebook/VGGT-Omega/vggt_omega_1b_512.pt, resolution 512, dtype float16 | transformers 5.18.0 plus the vggt-omega commit `48b23c8` tarball (`--no-deps`), from `2026-10-19-feasibility-tally/extract-job.yaml` |
| sam3 | `--encoder sam3 --dtype float16 --batch-size 8` | hf_id facebook/sam3, resolution 1008, dtype float16 | transformers 5.18.0 from the tally Job |

Batch sizes are not in the metas (only the keys above are checked); they are the old Jobs' values. The model revisions of SigLIP 2, C-RADIOv4, SAM 3 and VGGT-Omega are printed at the start of each Job (`revision <repo> <sha>`) and not enforced; compare them with PLAN Amendment 2026-11-09 section 6 (SAM 3 `3c879f3`, VGGT-Omega `1041e80`, C-RADIOv4 remote code `c0457f5`) in the log. Only DINOv3's revision (`ea8dc28`) is asserted, as pf-val did. The metas also hold `n_demos` (40 here vs the reference 50, not a checked key) and `git_commit` (the Job's `SRC_GIT_COMMIT`, not a checked key).

### Staging path rule (important for phase B)

`pf_analysis` compares the `source_file` of the extraction metas across roots and refuses duplicates. Every phase-A extraction reads its frames from the same local path, `/work/pfa-trainval/libero_goal/<stem>.hdf5` (verified in the Job scripts and asserted by the post-extraction meta check). The parent folder must keep the name `libero_goal`, because `extract_features.py` writes `<out>/<encoder>/<parent folder name>/<stem>.h5`. **Phase B must use a different staging path for each condition**, for example `/work/pfb-test-clean/libero_goal/<stem>.hdf5`, `/work/pfb-test-camera/libero_goal/...`, one per condition, in every phase-B extraction Job, and the same path for one condition across the five encoders.

## 6. Outputs on the proxy PVC `stacking-encoders-proxy` (1 Ti), `/mnt/proxy/pf-main/phase-a/`

- `render/` : `libero_goal/<stem>.hdf5` (ten files, 40 demos each), `render-report.json`, `render-sha256.txt`, and the marker `RENDER_COMPLETE`. Written by Job 1 via `render.part/` then a rename; Jobs never overwrite (Job 1 aborts with exit 14 if `render`, `render.part` or `feats/dinov3` exists). This is phase B's `--consistency-ref`. Keep it until phase B's check has run (Amendment 2026-11-09 (b)).
- `feats/<encoder>/libero_goal/<stem>.h5` : the phase-A feature root (`--trval-feats $PH/feats`). Each file is written by one pod, through `.partial` then a rename; a destination that exists is refused (exit 22 at start, or in the copy).
- `reports/<job>-<stamp>/` : each Job's own reports (`render-report.json`, `compare-vs-pf-val.json`, `setup-info.txt`, `render-sha256.txt`, the dev report).
- `dev/<stamp>/` : the full `dev` output directory. `logs/` (under `/mnt/proxy/logs/`) : one log per Job.
- Pilot: `/mnt/proxy/pf-main/pilot/<stamp>/<condition>/`, marked not for training.
- Each Job also prints a base64 tarball of its small reports between `=== BEGIN PFA_TGZ_B64` and `=== END PFA_TGZ_B64` at the end of its log (omitted if over 2 MB).

CephFS write stalls (08, proxy round): nothing large is written to CephFS directly. The render is written to `/work/render` on local disk and copied once; features go to `/work/feats` first. The only direct appends are the small log files. (The render copy, about 5 GB, and the feature copies are plain `cp` under `timeout`.)

Space expected (arithmetic, speculative): render HDF5s 50,480 states x 2 cameras x 128 x 128 x 3 bytes = 4.96 GB, uncompressed (the code calls `create_dataset` without compression, verified `src/pf_rerender.py:503`). Features: the PLAN's 49 GB for all five encoders over 228,440 images scaled to the phase-A 100,960 images = about 22 GB (the PLAN's number is itself a scaling of the tally's 27.2 GB, speculative). Reports and logs: well under 1 GB. Total about 27 GB; the Jobs abort if the PVC has less than 100 GB free. Local disk per pod is sized in each manifest (Job 1: 50 Gi request for the bundle, 6.4 GB of raw files, the render, the staged copy and the DINOv3 features, like pf-val's 50 Gi).

## 7. Caps, estimates and what each Job exits with

All estimates are speculative (rates from earlier runs on other cards, noise cost irrelevant here since phase A is clean).

| Job | Stage caps (seconds) | Estimate | GPU-h (pod time) |
|---|---|---|---|
| 1 render | pip 900, setup test 900, bundle hash 1800, download 1800, render 7200, compare 3600, publish 5400, DINOv3 7200, meta 900, copy 3600 | restore about 3.4 min (measured in pf-val), render 50,480 states at 48 per second = about 18 min, DINOv3 100,960 images at 55.7 per second = about 30 min | about 1.4 |
| 2 siglip2 + c-radiov4 | pip 1800, SigLIP 2 7200, C-RADIOv4 10800, meta 900, copy 3600 each | SigLIP 2 100,960 / 142 per second = 0.2 h, C-RADIOv4 100,960 / 62 = 0.45 h (TITAN RTX rates) | about 0.8 |
| 3 vggt | pip 1800 (+ vggt install 1800), extract 21600, meta 900, copy 3600 | 100,960 / 12.5 per second = 2.2 h (2080 Ti rate) | about 2.4 |
| 4 sam3 x3 | extract 18000 each | 100,960 / 5.5 per second = 5.1 h in total, about 1.7 h per shard | about 5.7 in total |
| 5 dev | pip 1800, download 1800, raw hash 900, dev 28800 | unknown (many ridge fits, 8 alphas, up to 6,656 columns) | 0 GPU |
| 6 pilot | restore, download, 4 x 3600 | a few minutes per condition | about 0.4 |

Total phase A without the pilot: about 10.3 GPU-h (speculative); the PLAN's section 8 estimate for train+val was about 8.5 extraction + 1 render.

Every stage's exit code is kept (`stage` function); the reports are copied to the PVC even after a failed stage; the Job exits with the worst code (a timeout is 124, a failed check is the script's own code). Exit 0 means every stage passed. For the dev Job the exit code is `pf_analysis.py dev`'s own (1 if gate A failed, after the report is written). Exit codes before the stages: 11 SRC_GIT_COMMIT unfilled, 12 PVC space, 13 env bundle, 14 render exists, 20 inputs missing, 21 extractor changed, 22 feature file exists.

Sizing notes. Memory and CPU for Jobs 3 and 4 are the tally Job's (measured there, 4Gi / 4800Mi, tally extract-job comments). Job 2 uses 2 CPU and 4Gi: the proxy extraction measured 2.4 GB resident for these two encoders (proxy LOG 2026-10-29), so 4Gi stays inside the 20 to 150% band. Job 1 and the pilot keep pf-val's 2 CPU / 12Gi (pf-val's actual memory use was not recorded; if the Violations page flags it, resize from Prometheus). The dev Job's 14Gi / 16Gi request is arithmetic only, not measured (speculative): float32 global features for the five encoders over 50,480 rows x 2 cameras x 7,424 dims = 3.0 GB, plus a standardized copy, the pair concatenation (at most 6,656 columns, 1.3 GB), the train-row copy and the ridge matrices, giving a peak of about 8 to 12 GB. It is the Job most likely to need a resize; a `--ftype mean` run needs about twice that memory and is not included.

## 8. Pre-flight checklist (08, adapted) and what the lead must do before submitting

1. The user's separate go (per the PLAN's handoff: "code changes, review, then the user's go"). Phase A is gated: nothing in phase B until gate A has passed.
2. Sources committed; `SRC_GIT_COMMIT` filled in the rendered copies; the commit hash recorded in the experiment README. The 08 checklist items 1 to 9 apply: Job not pod, 1 GPU, limits within 20% of requests (verified by `check.py` output in the LOG), no tolerations (none), no `sleep` (none), Violations page after launch, delete the Jobs afterwards.
3. Create `stacking-encoders-pf-main-src` (section 2) and verify the keys by sha256.
4. Confirm `stacking-encoders-proxy-src` holds the expected `extract_features.py` and env scripts (section 2).
5. Recount live GPU pods (section 4) and check free space on the PVC (`df` is also done by each Job).
6. Check that nothing already exists under `/mnt/proxy/pf-main/phase-a/`.
7. Read-only check before Job 2 to 5: Job 1 exited 0, the log shows `compare-pf-val` (or its stated skip), `publish-render`, `meta ok: dinov3`, and `render-report.json` shows no failures. Look at the compare result before relying on it: a pixel difference between this render and the pf-val render of the same val demos is a finding to report, not a failure of the Job (report-only, per Amendment (b) item 4).

If Job 1 fails after the render but before the publish, the render is lost with the pod (nothing is kept locally); no partial state remains on the PVC except possibly `render.part/`, which must be deleted by hand before a retry (with the user's OK). If it fails after the publish but in the DINOv3 stage, `render/` and `RENDER_COMPLETE` stay and Jobs 2 to 4 may run; DINOv3 must then be extracted by a small separate Job (not written; ask the lead), because Job 1 refuses to run again while `render/` exists.

## 9. Choices made here that the PLAN did not fix (for the lead)

See the report to the lead and the LOG entry "Look closely at": seeds 0/0; SAM 3 shard membership; resources of each Job; the cap values; the extra `diagnostics.py` in `/pf`; publishing through `render.part` and a marker; extraction in Job 1 gated on a successful publish; dev run for global features only; the 100 GB free-space abort; the revisions printed but not enforced (except DINOv3).
