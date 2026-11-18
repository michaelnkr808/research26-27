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

## 2026-11-18 [agent, lead, implementation] (PRs #59-#61 merged; encoder cache freshness check: DINOv3 and C-RADIOv4 bit-identical, SigLIP 2 out of tolerance on 5 of 8 episodes; stop for the user)

**Merged** on the user's "merge all 3": #59 (`4b024a8`), #60 (`6963830`), #61 (`5fca500`). Worktrees and branches removed.

**Run** (on the user's go, "and the go to the check job"; verified, `results/cache-fresh-check/stacking-encoders-pairings-cache-fresh-check-tcwjl.log`, `job-status.yaml`). `$C` = `5fca5003e4372f8a2615f8d841b88d7f2055e7f1`. ConfigMap `stacking-encoders-pairings-freshcheck-src`: all 6 keys read back equal to `$C`. Rendered `jobs/rendered/job-cache-fresh-check.yaml` (commit 1, hashes 6; one `FILL_` match left, in a header comment, so RUN step 4's "must print 0" check is off by that comment). Server dry run OK. Created 01:55:20 UTC; pod on nautilus-it-gpu06.fullerton.edu, **NVIDIA L4**; setup 01:56:23, check 01:56:46 to 02:02:57 (366.6 s), `=== exit 3`. Job failed at once (podFailurePolicy). About 0.12 L4 GPU-hours (arithmetic from the timestamps). Job and ConfigMap deleted after the log was saved.

**Loaded revisions** (verified, `[load]` lines): each loaded commit equals its pin. DINOv3 `ea8dc286...` fp32, SigLIP 2 `e8e48729...` fp16, C-RADIOv4 `c0457f5d...` fp16, all on cuda.

**Outcome** (verified): `=== FAIL: 5 episode(s) out of tolerance, 0 problem(s)`. Episodes (seed 0): spatial 276 and 368, object 123 and 140, goal 7 and 32, libero_10 251 and 314.
- DINOv3: 8 of 8 bit-identical to the cache (max abs diff 0).
- C-RADIOv4: 8 of 8 bit-identical.
- SigLIP 2: 3 of 8 bit-identical (spatial 276, libero_10 251 and 314). 5 FAIL: spatial 368 (cos_min 0.999327, cos_median 0.999938, rel_l2 3.67e-02, rel_max 1.33e-01, bit-equal 4.6%), object 123 (0.999829, 0.999975, 1.85e-02, 1.03e-01, 5.9%), object 140 (0.999869, 0.999973, 1.62e-02, 7.50e-02, 5.8%), goal 7 (0.999930, 0.999981, 1.19e-02, 4.80e-02, 6.0%), goal 32 (0.999856, 0.999979, 1.70e-02, 9.15e-02, 5.9%). Cosine passes the 0.999 bar everywhere. rel_l2 fails 0.02 on 1 episode, and rel_max fails 0.02 on all 5.
- First GPU measurement of the numeric noise: same card type (L4), same batch: 0. SigLIP 2 episodes not bit-equal: rel_l2 1.2e-2 to 3.7e-2, cosine at least 0.99933.

**Reading** (speculative, not verified per episode): the pins load the cache's weights; the DINOv3 and C-RADIOv4 bit-identity shows the pipeline is exact on an L4. The SigLIP 2 mismatches fit card-to-card fp16 numerics: SigLIP 2 try 1 wrote "all of libero_spatial and part of libero_object" on an RTX 5000 Ada (direction `LOG.md` 2026-10-29 (h), line 681), and later tries ran on other cards (RUN tolerance item 1). That does not explain why spatial 276 is bit-equal. Which card wrote each episode is not recorded in the cache. A weight change would give cosines far below these. Per the RUN file, a result past a bar is a stop for the user: no bar changed, no rerun.

**Next:** the user decides between accepting the SigLIP 2 card-numerics gap and re-extracting the SigLIP 2 cache on L4s.

## 2026-11-18 [agent, lead, implementation] (SigLIP 2 cache deleted; L4 re-extraction submitted, Pending)

On the user's choice ("yeah option 1", then "merge" for PR #62, merged as `0c5d158`), following `jobs/RUN-siglip2-reextract.md`:
- Step 1: no stacking Job or pod running. Steps 2-3: `$C` = `0c5d1586dd028447c388a2cdcd23efb30ac12cee`. ConfigMap `stacking-encoders-pairings-reextract-src`: 5 keys read back equal to `$C`, and the SigLIP 2 pin is present. Rendered `jobs/rendered/job-siglip2-reextract.yaml` (commit 1, no unfilled placeholder outside comments, 5 hashes).
- Step 4, deletion (verified, `results/siglip2-reextract/stacking-encoders-pairings-siglip2-delete-f4bnk.log`, `delete-job-status.yaml`): created 03:12:17 UTC, ran 03:13:55 to 03:15:02 UTC, exit 0. Listed 3,432 entries (`/mnt/proxy/logs/siglip2-delete-20261118-031355.list`): libero_spatial 434 episodes and 434 stats, libero_object 457 and 457, libero_goal 433 and 433, libero_10 388 and 388, one `_meta` each, no tmp. The cache was 234G. `/mnt/proxy` went from 834G used / 191G free to 606G / 419G. c-radiov4, dinov3, meta, t5, vae and vae_b are still listed. Ends `=== deleted: /mnt/proxy/feats/siglip2 is empty`. Job deleted.
- Step 5, extraction: `stacking-encoders-pairings-siglip2-reextract` (2 shards, L4 only) created 03:15:49 UTC. At 03:34 UTC both pods are **Pending** with no events (scheduler queue; L4 + us-west + driver > 569 is a narrow set, see the memory note on L4 queues).

**Until the freshness check passes**, no Job may read `feats/siglip2`: the `siglip2` and `dinov3+siglip2` stage 2 configurations are on hold (RUN step 1).

**Next (handoff):** once the pods start, a GPU check about 15 min in (RUN step 6). On completion, save the logs and delete the Job and ConfigMap (step 7). Then the cache check (step 8) and the freshness check rendered to a new file (step 9), with SigLIP 2 expected bit-identical on all 8 episodes. If it is still Pending after several hours, ask the user before widening the card set: widening would break the L4-only point of this re-extraction.

## 2026-11-18 [agent, lead, implementation] (SigLIP 2 re-extraction still Pending; left queued overnight)

At 04:52 UTC (about 96 min after creation) both pods of `stacking-encoders-pairings-siglip2-reextract` are still **Pending** (verified, `kubectl get pods` and pod events). The scheduler message: 0/540 nodes available; 292 fail the node affinity (L4, driver major > 569, us-west, two hostnames excluded), 226 untolerated taints, 20 unschedulable, and of the remaining two, 1 has insufficient `nvidia.com/gpu` and 1 insufficient cpu/memory. So the L4 + us-west + driver set leaves about two candidate nodes.

The lead's recommendation: leave it queued overnight; if still Pending in the morning, drop only the us-west term and keep L4-only (keeps the same-card-type point; cost: cross-region CephFS reads/writes, possibly slower). The user: "okay sure lets leave it queued overnight". The widening is **not approved**; ask again in the morning if still Pending.

**Next:** one check in the morning. If running, RUN step 6 (GPU check about 15 min in). If still Pending, ask the user about dropping the us-west term.

## 2026-11-18 [agent, lead, implementation] (SigLIP 2 re-extraction: shard 0 failed on a full NRP Ceph disk; Job suspended; overnight watcher started)

**What happened** (verified unless marked; pod logs in `results/siglip2-reextract/`, Ceph numbers from the public Prometheus, namespace `rook`):
- Shard 0 (`...-0-4wr4x`) got nautilus-it-gpu05.fullerton.edu (NVIDIA L4) after about 2 h 15 min Pending; container 05:31:05 to 05:44:33 UTC. Five hash lines OK, `fresh start`, 22.6 frames/s overall, `[gpu]` samples mostly 100%. It wrote the even episodes of libero_spatial up to ep 286, then exit 1: `tee: /mnt/proxy/logs/extract-siglip2-0-20261118-053105.log: No space left on device` and `PermissionError: [Errno 13]` opening `.ep000288.stats.npy.tmp`.
- The PVC is not full: `df` at 05:49 UTC shows 615G used, 410G free of 1.0T; `feats/siglip2` 15G.
- Cause (verified from the alarm history; the link to our ENOSPC is the lead's reading): NRP's main Ceph cluster has one disk at its full line. `osd.231` used fraction 0.9496 to 0.9500 over the last 8 h; `OSD_FULL` (HEALTH_ERR) and `POOL_FULL` were on at 23:43-23:51, 23:59-00:30 (several stretches), 00:42-01:30 (several), 01:43-01:44, 02:40-02:41, 05:28-05:31 and 05:46-05:49 UTC (1-minute steps); off all of 2026-11-17 before 23:43. Pool `nautilusfs-data0`: `percent_used` 0.969, `max_avail` 37.6 TB, so the pool as a whole has room. Next four disks: 0.9315, 0.9305, 0.9262, 0.9232.
- Kubernetes started a retry for shard 0 (`...-0-pzr5h`, gpu06, 05:49:05) and shard 1 got gpu05 (`...-1-q4h6t`, 05:49:44). The retry was still downloading the SigLIP 2 weights (1.9 GB `.incomplete` at 05:56; no episode yet, GPU 0%). Shard 1 ran at 21.5 frames/s with `[gpu]` at 100% and reached ep 163.
- State of the folder before the suspend (one `ls` through the shard 1 pod): no `ep000288` file, no `*.tmp` file, 217 stats files. A loop meant to list episodes without a stats file printed nothing, but its shell quoting was not checked, so pairing is **not verified** here; the cache check (RUN step 8) covers it.

**Decision.** The lead told the user the options (keep running with two failures left before the Job dies and a restart needs a new deletion; suspend and resume when the storage alarm has stayed off) and recommended suspending. The user: "yeah do what you think is best". Suspending is not a step of `jobs/RUN-siglip2-reextract.md` (its rule for a pod failure is stop and ask; the user was asked).

**Done:**
- Logs of all three pods and the Job status saved first (`results/siglip2-reextract/`, `job-status-before-suspend.yaml`).
- `kubectl patch job stacking-encoders-pairings-siglip2-reextract --type=merge -p '{"spec":{"suspend":true}}'`: condition `Suspended` since 05:58:00 UTC, `failed` 1, both running pods removed, the failed pod kept. The Job keeps its uid, so resumed pods take the guard's `resume` path and skip finished episodes. Recalled (Kubernetes docs, not tested here): a resume resets `startTime`, so the 16 h deadline restarts.
- Watcher `results/siglip2-reextract/watch_siglip2.py` (copy; it runs from the session scratchpad under `caffeinate`, pid 16009, started 05:59:52 UTC, status and events in `<scratchpad>/siglip2-watch/live/`). Rules: every 10 min ask Prometheus whether `OSD_FULL` or `POOL_FULL` was on in the last 2 h (no or thin data counts as not clear); when both were off for 2 h, set `suspend=false` (at most 4 resumes). After a resume, every 5 min: save the log of any finished pod; a filtered log and `kubectl top` 15 min after a pod starts; suspend again on a new pod failure, on an alarm in the last 10 min, or on a pod's `[gpu]` mean under 40% over 10 min on two checks after its first 30 min (the RUN stop rule says delete; the watcher suspends instead and stops). On Complete or Failed it saves logs and `job-status.yaml` and exits. It never deletes anything. If kubectl hangs (expired login) it stops calling kubectl for an hour and logs it. Both paths were dry-run once before the start.

**Mistake to record:** a diagnostic `kubectl exec ... env | grep HF` printed the Hugging Face token into this session's transcript. It was not written to any file (the committed logs were grepped for it).

**For planning (not acted on):** `stacking-encoders-runs` is on the same CephFS (`nautilusfs`), so stage 2 training writes would hit the same refusals while the alarm flaps.

**GPU-hours so far** (arithmetic from the container times): about 0.22 + 0.15 + 0.14 = 0.5 L4 GPU-hours.

**Next:** in the morning read `<scratchpad>/siglip2-watch/live/events.log`. If the Job resumed and finished: RUN steps 7 to 9. If still suspended: tell the user, and ask before anything else (resuming by hand, or NRP support about `osd.231`). The earlier open question (dropping the us-west term if Pending is long) is still not approved.

## 2026-11-18 [agent, lead, implementation] (SigLIP 2 re-extraction complete on L4s; Job and ConfigMap deleted; cache check and freshness check wait for the user's go)

**Outcome** (verified; pod logs, `job-status.yaml`, `watcher-events.log` and the two `check15-*.txt` files in `results/siglip2-reextract/`): the Job reached `Complete` at 12:33 UTC (CompletionsReached, succeeded 2, failed 1, the 05:44 storage failure).
- The watcher resumed the Job at 07:50:00 UTC after `OSD_FULL` and `POOL_FULL` had been off for 2 h (460 samples each). `status.startTime` became 07:50:01, so a resume does reset the deadline clock (verified; was "recalled" in the previous entry). The alarm has not come back: its only on-interval between 05:36 and 18:06 UTC is 05:46-05:49 (1-minute steps, 737 points).
- Shard 0 (`...-0-p9mdk`, nautilus-it-gpu06.fullerton.edu, NVIDIA L4): container 08:18:29 to 09:46:40 UTC. `resume` path, five hash lines OK. libero_spatial "144 of 217 episodes already done" (411 s for the rest, 22.1 frames/s), libero_object 0 of 229 (1,449 s, 23.2), libero_goal 0 of 217 (1,136 s, 23.6), libero_10 0 of 194 (2,252 s, 23.2). `_meta device check: 4 files, 0 not cuda`, `=== done`.
- Shard 1 (`...-1-rrbb4`, nautilus-it-gpu05.fullerton.edu, NVIDIA L4): Pending from 07:50 to 10:58; container 10:58:24 to 12:32:54 UTC. `resume` path, five hash lines OK. libero_spatial "82 of 217 episodes already done" (747 s, 22.2 frames/s), libero_object 0 of 228 (1,455 s, 23.2), libero_goal 0 of 216 (1,142 s, 22.9), libero_10 0 of 194 (2,288 s, 22.7). `_meta device check: 4 files, 0 not cuda`, `=== done`.
- Episode counts per suite, both shards: 434, 457, 433, 388 = 1,712 (arithmetic from the "of n" lines), the expected counts. The "already done" numbers on libero_spatial are the episodes the first pods wrote before the failure and the suspend (expected for a resume; RUN step 6's "0 already done" applies to a first pass).
- `[gpu]` samples (30 s): shard 0 mean 88.8% after the first 3 minutes, 17 of 171 samples under 40%; shard 1 mean 85.9%, 24 of 183 under 40% (single samples between episodes). The watcher's 15-minute checks: last-20-sample means 85% and 88%. No idle strike.
- GPU-hours (arithmetic from container times): 0.22 + 0.15 + 0.14 (first three pods) + 1.47 + 1.58 = about 3.6 L4 GPU-hours, against the RUN estimate of 4 to 5. Rate 22 to 23.6 frames/s against the 21.1 basis.

**RUN step 7 done:** logs saved by the watcher as each pod ended (both end with `=== done`); `kubectl delete job stacking-encoders-pairings-siglip2-reextract` and `kubectl delete configmap stacking-encoders-pairings-reextract-src` at about 18:06 UTC; no stacking Job or pod left. The watcher exited on its own at 12:33:35 UTC.

**Deviation from the RUN file, as recorded before:** the Job was suspended and resumed once (not a RUN step), so the cache was written by five pods of one Job on two L4 nodes (gpu05, gpu06) instead of two pods.

**Still on hold:** no Job may read `feats/siglip2` until RUN steps 8 (cache check, CPU) and 9 (freshness check, one L4 for about 10 minutes, SigLIP 2 expected bit-identical on all 8 episodes) pass. Both are new Jobs and wait for the user's go.

**Next:** the user's go for steps 8 and 9.
