# Experiment log: proxy-policy calibration round (decision 003)

Gritty, step-by-step record for this experiment (runs, errors, reruns, exact numbers). Append-only, newest at the bottom. The direction's `../../LOG.md` keeps short summaries that point here.

## Before 2026-11-04 (i): entries are in the direction LOG
This experiment started before experiments had their own LOG. Its history up to and including 2026-11-04 (h) stays in `../../LOG.md` (append-only, not copied here). The index below was made 2026-11-04 from each entry's heading; an entry that only mentions this experiment in passing may be missing. Some letters repeat on the same day (two sessions); the heading text tells them apart.

Design and locked protocol (no Jobs):
- 2026-10-27 (m): decision 003, go straight to the multi-seed proxy policy; plan written.
- 2026-10-27 (n): PLAN steps 1–3: facts, code, local smoke test.
- 2026-10-28 (untagged): smoke test rerun with real DINOv3.
- 2026-10-28 (b) "design review with the user": stage 1, proxy rebuilt on Fast-WAM's own code.
- 2026-10-28 (b) "reviewer feedback": reframed as a proxy replication; outcome rule amended before any training run.
- 2026-10-28 (c): paired-analysis code for the amended outcome rule.
- 2026-10-28 (d): full plan written up for the reviewer; auto-chain agreed.
- 2026-10-28 (e): second reviewer round (proxy wording, threshold, storage, orientation).
- 2026-10-28 (f) "decision rule locked": before the pilot.
- 2026-10-28 (f) "PLAN double-checked": autonomous run set up (RUN.md, manifests, watcher).
- 2026-10-28 (g) "NRP policy check" of the autonomous run.
- 2026-10-28 (g) "evaluation seeds and subset weighting locked".
- 2026-10-28 (h): sample-size rule locked (paired pilot).
- 2026-10-28 (i) "doc and PLAN reconciled"; (i) "NRP policy recheck".
- 2026-10-28 (j): RUN.md analysis-code step; (k): watcher survives a dropped laptop connection; (l): open NRP-usage risks.

Setup, simulator, feature caches (RUN steps 1–8):
- 2026-10-28 (m): run started (volume, download Job, step 7a analysis code).
- 2026-10-29 (a): step 2, orientation, rollout code, frame check on the Mac.
- 2026-10-29 (b)–(c): sim Job (try 1 OOM, try 2 works on the cluster).
- 2026-10-29 (d)–(e): probe (expired in queue, then done); T5 and extraction submitted.
- 2026-10-29 (f)–(m): extraction GPU-floor stops, thread oversubscription fix, VAE CephFS write stalls, SigLIP 2 borderline.
- 2026-10-30 (a)–(c): VAE and SigLIP 2 resubmitted; OOM fixes.
- 2026-10-31 (a)–(b), 2026-11-01 (a)–(b): VAE shard stalls on CephFS writes; fix = stage on local disk, then copy.
- 2026-11-01 (c)–(d): VAE shard 3 complete; cache checks pass.

Pilot training (RUN step 9):
- 2026-11-01 (e)–(h): pilot Jobs; stopped as disk-bound; CephFS read test; parallel-read loader; rerun.
- 2026-11-02 (a)–(f): D+R pilot2 OOM on an L4, resubmitted on 24/48 GB GPUs; D pilot2 done; D+R done.

Pilot rollouts:
- 2026-11-02 (g): first pilot rollout Job.
- 2026-11-03 (a)–(b): pre-check pass; stopped at ~25% GPU.
- 2026-11-03 (c)–(h): policy profiling and env-bundle build; launch-bound denoise and CPU noise found; fixes (noise cache, threads, CUDA graph); profile run 2 (graph == eager, GPU busy); pilot rollout 2 rendered.
- 2026-11-03 (i): pilot rollout 2 submitted; 21:04 GPU list widened.
- 2026-11-04 (a): a pod failed early in setup; deadline resubmit.
- 2026-11-04 (b): cause found: offline .deb install fails in fresh pods (Pre-Depends order).
- 2026-11-04 (c)–(h): env-build2 (two dpkg passes, still fails), local Docker test of fixes, apt-ordered install (b0b60f3), env-build3 hung in apt-get update (node network), time limit (2bab64f), env-build4 passed, pilot rollout 2 re-rendered.

Reviews and feedback that touch this experiment:
- 2026-11-03 (j): review of the proxy-calibration design against the standards.
- 2026-11-03 (k): presentation feedback; seed-noise check and perturbed-frame tests added to the direction PLAN.

## 2026-11-04 (i) [agent] (state at the switch to this LOG)
- Feature caches: all done (LOG 2026-11-01 (d)). Pilot training: D and D+R pilot2 checkpoints, training seed 0, done (LOG 2026-11-02 (c), (f)).
- Code: src at 2bab64f. Policy path unchanged since dbbc71d (start noise cached per eval seed, torch threads = pod CPUs, `--cuda-graph`); since then only `src/setup_sim.sh` changed: offline .debs installed by apt with sources off (b0b60f3), 10-min limit on the build's `apt-get update` (2bab64f). ConfigMap `stacking-encoders-proxy-src` = 2bab64f (14 keys, checked equal to the repo).
- Env bundle: `/mnt/proxy/env/env-9182f8131559.tar` (sha256 86afbf57…), built and restore-tested in a fresh container by env-build4 (Complete 21:59:03 UTC; restore: untar 41 s, debs + assets unzip + checks 487 s; log `results/logs/env-build4-restore-2qskz.log`). Earlier builds wrote a172458a0893 (2026-11-03) and 6549d0637e12 (env-build2) to the same folder; the current scripts don't use them (not checked whether still there).
- Known gap: the build's `apt-get install` download and `pip install` have no time limit.
- `jobs/rendered/job-pilot-rollout2.yaml`: SRC_GIT_COMMIT 2bab64f, 2 pods (one per checkpoint, all 340 jobs each), 4 envs, `--cuda-graph`, 4 CPU / 20Gi (limits 4800m / 24Gi), wide GPU list (3090, 4090, A5000, A6000, L40, L40S, A10, L4, A40), k8s-chase-ci-07 and ry-gpu-08 excluded, backoffLimit 4, deadline 36 h, output `rollouts/pilot-rollout2`. Server dry run OK. Not submitted: waiting on the user's OK.
- Nothing of ours running in aria-world (env-build2 failed, env-build3 deleted, env-build4 complete).
**Next:** user OK → submit pilot-rollout2, start `watch_job.sh stacking-encoders-proxy-pilot-rollout2 2 120`, check GPU util ~15 min into rollouts (after ~9 min setup + model load); < 40% → stop and report. Then per-episode timings, GPU, RSS, errors, success per checkpoint → sizing for stage 4 (sample-size rule, PLAN).

## 2026-11-04 (j) [agent] (pilot rollout 2 submitted)
- User OK (2026-11-04, "go for the pilot rollout").
- No Job of that name existed; ConfigMap re-checked equal to the repo src (2bab64f). `jobs/rendered/job-pilot-rollout2.yaml` as in (i): server dry run OK, submitted 22:08:45 UTC. Watcher `watch_job.sh stacking-encoders-proxy-pilot-rollout2 2 120` in the background.
**Next:** as (i): GPU util check ~15 min into rollouts (pod setup ~9 min + model load first); < 40% → stop and report. On watcher exit: per-episode timings, GPU, RSS, errors, success per checkpoint; then sizing.
- 2026-11-04 23:35 UTC: verified (kubectl, pod exec): pod 0 (pilot2-dinov3-s0) Pending ~71 min (scheduler: matching nodes short on GPU, CPU or memory), then running on nautilus-it-gpu06.fullerton.edu (NVIDIA L4, 23034 MiB) from 23:19:55. Setup in a real GPU pod passed: "TIMING restore: untar 119 s, debs + assets unzip + checks 400 s" (bundle 9182f8131559), fastwam ok, cuda True; rollouts started (340 jobs, 0 already done, 4 envs, torch threads 4, cuda graph True). At 23:34:37: 28 records in pilot2-dinov3-s0.0of1.jsonl; nvidia-smi snapshot 99% util, 5579 MiB used (one snapshot, not the 10-min check). Pod 1 still Pending. Note: gpu06 is one of the nodes that stalled VAE CephFS writes (LOG 2026-10-31); rollouts only append small JSONL lines, watch for stalls.
- 2026-11-04 23:52–23:58 UTC: verified (pod 0 exec): the watcher's gpu10m column is nan for this pod (no DCGM series for nautilus-it-gpu06; memory lists DCGM gaps on some nodes), so its exit-4 GPU check is blind here; measured in the pod instead. nvidia-smi 1 s samples, 23:52:39–23:57:39 (300 samples): mean 60.6%, median 61, 81% of samples ≥ 40%, 4% at 0. (One earlier snapshot at 23:52:22 read 20%.) Records: 28 at 23:34:37, 135 at 23:52:15, +51 in the 5-min window (~186 of 340); ~410 episodes/h average so far, on an L4. cgroup memory: current 24.81 GB, peak 25.77 GB = the 24Gi limit, but anon 12.39 GB vs file cache 10.94 GB (8.89 inactive); memory.events oom 0, oom_kill 0, max 0. So process memory ~12.4 GB of the 20Gi request; the rest is reclaimable page cache (the bundle untar + assets unzip). Per-process RSS: 4 env workers ~3.2–3.3 GB each, main rollout.py 2.6 GB (RSS counts shared pages). Pod 1 still Pending.

## 2026-11-05 (a) [agent] (pilot rollout 2: pod 0 (D) complete; watcher exit 6 was a post-completion artifact)
- verified (kubectl; log saved to `results/logs/pilot-rollout2-0-sstst.log`): pod 0 (pilot2-dinov3-s0, L4 on nautilus-it-gpu06) Completed, exit 0, container 23:19:55 → 00:12:12 UTC. rollout.py: "finished: 340 records, 0 errors in 2530s (484 episodes/h); 10165 policy calls, mean batch 1.6, policy busy 57%; cgroup memory now/peak GB (13.95, 25.77)". Progress lines: rate rose from ~360 to 484 episodes/h over the run; mean batch 1.6 throughout.
- Watcher exit 6 at 00:19:45 (cpu_ram_out=1): pod 0's 10-min CPU rate was 0.788 cores = 19.7% of the 4-CPU request, read 7 min after the pod finished. verified (Prometheus, 10-min CPU rate at 23:45 / 23:55 / 00:05 / 00:12 / 00:17): 2.93 / 3.77 / 2.83 / 2.84 / 1.52 cores, i.e. 70–95% of request while running; the low value is the window sliding past completion, not a usage problem. The watcher's rule doesn't exclude finished pods (noted; not changed).
- Success rates not read yet: the JSONL is on CephFS (/mnt/proxy/rollouts/pilot-rollout2/pilot2-dinov3-s0.0of1.jsonl) and no pod with the PVC is running. Plan: read it from pod 1 once it runs (exec), together with its own file. No new pod started for this.
- Watcher restarted in the background, delayed until pod 0's CPU series ages out of Prometheus (max 30 min) so the same artifact doesn't refire. Pod 1 (pilot2-dinov3+c-radiov4-s0) still Pending (~2 h 10 min).
**Next:** pod 1 runs → in-pod GPU check (DCGM may be blind again), read both JSONLs (success per checkpoint, per category, per eval seed; clean vs LIBERO-Plus; episode lengths) for sizing.

## 2026-11-05 (b) [agent] (one-shot status check: pilot rollout 2 finished, both pods Completed)
- verified (kubectl at 04:28:31 UTC; first kubectl calls at 04:26 failed with an OIDC connection reset / refused, then worked; Prometheus kube-state-metrics at 04:27 agreed): `job.batch/stacking-encoders-proxy-pilot-rollout2   Complete   2/2   DURATION 6h2m   AGE 6h19m`. Job created 22:08:45 UTC, completion time 04:11:33 UTC 2026-11-05 (Prometheus kube_job_status_completion_time); active 0, failed 0, succeeded 2.
- verified (kubectl): `pod/stacking-encoders-proxy-pilot-rollout2-1-lz8ts   0/1   Completed   0 restarts   node rci-tide-gpu-05.sdsu.edu`; pod 0 `-0-sstst` Completed on nautilus-it-gpu06.fullerton.edu. Pod 1 (D+R checkpoint, pilot2-dinov3+c-radiov4-s0): GPU NVIDIA L40 (nvidia-smi in its log); scheduled and started 03:16:47 UTC (Prometheus), so Pending ~5 h 8 min after the 22:08:45 submit; completed 04:11:12 UTC (~54 min running). Restarts 0 (Prometheus and kubectl).
- verified (log saved as `results/logs/pilot-rollout2-1-lz8ts.log`, via `kubectl logs`): setup "TIMING restore: untar 91 s, debs + assets unzip + checks 340 s"; rollouts began with 340 jobs, 0 already done, 4 envs. Final line: "[rollout 2813s] finished: 340 records, 0 errors in 2729s (449 episodes/h); 10397 policy calls, mean batch 1.6, policy busy 43%; cgroup memory now/peak GB (14.1, 25.77)". The log's own GPU summary (nvidia-smi, 30 s samples): "mean 42.6% over 94 samples". After the nvidia-smi summary the log ends with `340 /mnt/proxy/rollouts/pilot-rollout2/pilot2-dinov3+c-radiov4-s0.0of1.jsonl` (wc -l of the output file), the end date line, and `=== done`.
- Both checkpoints therefore have 340 records and 0 errors (pod 0: LOG 2026-11-05 (a)). Pod 1 ran 449 episodes/h on an L40 vs 484 on pod 0's L4 (verified, from the two logs); policy busy 43% vs 57%. Peak cgroup memory 25.77 GB in both pods; that it is mostly page cache is verified only for pod 0 (LOG 2026-11-04 (j)) and speculative for pod 1 (no anon/file split was measured there).
- GPU sampling in the Job: verified (read `jobs/rendered/job-pilot-rollout2.yaml` lines 62-63): the nvidia-smi 30 s sampler starts after `. /src/env_bundle.sh restore` and just before `python /src/rollout.py run`, so the log's "mean 42.6% over 94 samples" covers rollout.py only, including its model load, not the restore/setup period.
- verified (public Prometheus, after the fact, since the watcher's GPU check is blind on some nodes; this node has DCGM): query `avg_over_time(DCGM_FI_DEV_GPU_UTIL{pod="stacking-encoders-proxy-pilot-rollout2-1-lz8ts"}[10m])` as a range query, 03:17-04:12 UTC 2026-11-05, step 60 s; one series (gpu 3, rci-tide-gpu-05.sdsu.edu). This is the check RUN.md requires ("Rollouts and the 40% GPU floor"; step 9: pilot rollout must record GPU use, at least 40% over 10-min windows after startup; remedy is more environments per GPU; if that can't reach it, stop 10, stop and ask before stage 4). Trailing 10-min mean, UTC, %: 03:18-03:26 0.0 (setup, GPU idle); 03:27 1.2; 03:31 25.9; 03:33 38.9; 03:34 40.6 (first window at or above 40%); 03:35-03:48 44.2 to 55.4 (peak 55.4 at 03:44); then 03:49 39.3, 03:50 31.4, 03:51 25.2, 03:52 16.3, 03:53 9.7, 03:54 9.1 (minimum), 03:55 14.3, 03:56 20.0, 03:57 25.9, 03:58 29.7, 03:59 33.4, 04:00 38.2; 04:01 41.8 back above 40%; 04:02-04:12 45.4 to 58.4. Windows below 40% after the first reaching 40%: 03:49 through 04:00 (12 consecutive one-minute steps), minimum 9.1 at 03:54. (The windows before 03:34 include setup and the start of rollouts.)
- **The pilot did not meet the 40% floor** in pod 1: for 12 of the roughly 38 one-minute windows from 03:35 to 04:12 the 10-min mean was below 40%, down to 9.1%. The 42.6% whole-run mean hides this. This triggers RUN.md's stop-10 rule; no remedy was tried and nothing was changed (read-only check).
- The later slow stretch in the done count of the saved log `results/logs/pilot-rollout2-1-lz8ts.log` (about 1105 to 1706 s on the rollout clock: done 120 at 1105-1165 s, then 121 to 144 by 1706 s, verified, increments per ~60 s progress line, computed from the log: 0 to 5 in that stretch (0, 1, 0, 3, 1, 3, 4, 3, 4, 5), against 7 to 13 in the other lines apart from the 504-685 s stall (4, 0, 4, 0)) is roughly where the dip falls. With the rollout clock ending at 2813 s at the pod's 04:11:12 end, that is about 03:42:45 to 03:52:45; the time mapping is approximate (my arithmetic). Cause not determined; no guess made here.
- The earlier stall (about 504 to 685 s, done stays 48 to 52, about 03:32:45 to 03:35:45 by the same arithmetic) came during startup, when the 10-min mean was about 39 to 44%, not inside the 03:49 to 04:00 dip. It is a separate observation; cause not determined.
- Not done: success rates per checkpoint. The JSONLs are on CephFS and no pod was running to exec into; reading them needs a pod. No pod was started (task was read-only). Nothing on the cluster was changed.
**Next:** (1) Stop 10 (rollouts can't hold the GPU at 40% or more): the user decides; nothing is changed until then. (2) Reading the two JSONLs needs a pod; `_standards/08` prefers a Job, and it needs the user's OK before anyone prepares or submits it. (3) That read must produce, per RUN.md step 9: success per checkpoint, per category, per eval seed, and clean vs LIBERO-Plus; also the sd of the paired per-episode difference (D+R minus D, each episode averaged over eval seeds 41 to 43), sd*, N, and the cap (about 48 h on at most 8 GPUs, at the measured rollout speed).

## 2026-11-05 (c) [agent] (read-only diagnosis: why pilot rollout 2 pod 1 went mostly GPU-idle at about 03:43:30-03:53:30; corrected after review)
Read-only: public Prometheus and the saved logs only. No kubectl was needed; nothing on the cluster or in the code was changed. Times UTC, 2026-11-05 unless noted. All queries are range queries over the pod's run (about 03:15-04:15, step 60 or 120 s); `{P}` = `stacking-encoders-proxy-pilot-rollout2-1-lz8ts`, node `rci-tide-gpu-05.sdsu.edu`. The query scripts are in the session scratchpad only, not in the repo.

**Result: cause not determined, but the evidence rules out node CPU or memory contention, CPU throttling, memory trouble and local disk or pod-network I/O, and points at the workload (what it cannot rule out is a CephFS stall: node host-network bursts at 03:42 to 03:45 coincide with the dip start, source unattributed).** Most likely (speculative, not measured): jobs run in a fixed order (suite, then perturbation category), and for a stretch of consecutive jobs all 4 simulator workers were on episodes that cost much more CPU per simulator step, so the policy was called rarely and the GPU sat idle while the CPUs stayed busy.

1. Pod 1 during the dip (about 03:43:30 to 03:53:30) vs before and after.
- verified, GPU (re-measured at 30-s steps after review): `DCGM_FI_DEV_GPU_UTIL{pod="{P}"}`, range 03:40-03:58, step 30 s. 03:40-03:42:30 was 71 to 99; at 03:43:00 81 and from 03:43:30 about 0 (0, 0, 0, 0, 63 at 03:45:00, then 0 until 03:49; 1, 1, 6, 26 at 03:50:30, 0, 1, 0, 52 at 03:52:30, 5, 6 at 03:53:30); back to 43 and 81 at 03:54:00 and 03:54:30, then 67, 67, 87, 51. So the dip is about 03:43:30 to 03:53:45 with a few blips, about 10 minutes, which also fits the done-count stall (about 03:42:45 to 03:52:45 by my clock arithmetic) better than my first 03:46 start. The earlier startup stall shows the same way at 1-min steps: 03:33 to 03:35 read 1, 1, 1. So the 10-min mean below 40% (LOG (b)) comes from about 10 minutes of near-zero GPU use, not a general slowdown.
- verified, CPU was busy, not idle: `sum(rate(container_cpu_usage_seconds_total{pod="{P}",container!=""}[2m]))` = 3.2 to 4.7 cores through the dip (1-min steps, 03:44 to 03:53: 4.02, 4.01, 4.32, 3.24, 4.67, 3.77, 3.37, 3.96, 4.42, 3.73; the 4.67 at 03:48 is the peak of the run); from 03:27 on, outside the dip, 2.1 to 4.1. Split with `container_cpu_user_seconds_total` / `container_cpu_system_seconds_total` (same rate form): user 3.3 to 4.4 cores in the dip, system 0.02 to 0.07 at 03:46 to 03:50 (2-min steps; the lowest of the run). The pod was computing in user space, not waiting in the kernel.
- verified, no throttling: `sum(rate(container_cpu_cfs_throttled_periods_total{pod="{P}",container!=""}[2m])) / sum(rate(container_cpu_cfs_periods_total{pod="{P}",container!=""}[2m]))` and `sum(rate(container_cpu_cfs_throttled_seconds_total{...}[2m]))`: 0 in every window after 03:28 (0.006 at 03:26 is startup). The limit is 4.8 cores and the pod stayed under it.
- verified, no memory pressure: `container_memory_working_set_bytes` 15.7 GB flat through the dip (14.3 to 16.3 over the run); `container_memory_rss` 12.6 GB flat; `container_memory_cache` 10.6 GB flat; `container_memory_failures_total{failure_type="pgmajfault"}` rate 0 from 03:30 on. These cAdvisor numbers show no memory trouble, but "no reclaim" is too strong: the cgroup's own memory (the `cgroup memory now/peak GB` column in the progress lines) sat at 24 to 25.7 GB all run, close to the 24Gi limit, because the page cache fills the limit (LOG 2026-11-04 (j), pod 0: anon 12.4, cache 10.9 GB). Reclaim of cache may have been going on all run; it was not seen to stall anything, but I did not measure it.
- verified, no local disk or pod-network traffic: `container_fs_writes_bytes_total` rate 0.0005 MB/s all run after 03:28; `container_fs_reads_bytes_total` at most 0.9 MB/s and 0 to 0.0005 MB/s in 03:46-03:52; `container_network_receive_bytes_total` / `container_network_transmit_bytes_total` about 1e-5 MB/s. Caveat (reasoned from what the metrics count, not tested): these are the container's own filesystem and pod network counters. CephFS I/O to `/mnt/proxy` goes through the node's kernel client, so a CephFS write stall would not show in them. They only say local disk and pod network were quiet.
- verified, other DCGM fields (same `{pod="{P}"}` selector; power, PCIe TX and SM clock re-read at 30-s steps for 03:40-03:58 after review; the 1-min and 2-min figures I first gave were wrong in places): `DCGM_FI_DEV_FB_USED` 9.2 GB flat through the dip (8.1 to 9.6 over the run), no reset. `DCGM_FI_DEV_POWER_USAGE`: 190 to 244 W at 03:40 to 03:42:30; 105 to 145 W from 03:43 to 03:50:30 (mostly 105 to 125); then it climbs to 157, 158 (03:51), 144, 167 (03:52), 159, 175 (03:53) while util stays 0 to 52 at 03:51-03:53:30 (the reviewer's observation, confirmed), and 224, 220 at 03:54; the idle floor at 03:26 was ~104 W. `DCGM_FI_PROF_PCIE_TX_BYTES`: 4e6 to 6.5e6 B/s from 03:43:30 to 03:48:30 apart from one 1.7e8 sample at 03:46:30; then 1e7 to 2e7 with spikes of 2.1e8 (03:49:00), 3.1e8 (03:51:30), and 5.2e8, 5.7e8 at 03:53:00 and 03:53:30; 8e7 to 2.4e8 at 03:40 to 03:43:00 and up to 6.2e8 after 03:54. `DCGM_FI_DEV_SM_CLOCK`: 2.31 to 2.49 GHz for the whole of 03:40 to 03:54 (2.49 in 03:43 to 03:49), lows of 1.88 (03:55:00) and 1.94 GHz (03:56:00). `DCGM_FI_DEV_GPU_TEMP` 34 to 38 C in the dip (2-min steps); `DCGM_FI_DEV_MEM_COPY_UTIL` 0 in 03:48-03:52 (2-min steps). Reading: the GPU was not clock-throttled or out of memory, and it was nearly idle from 03:43:30 to about 03:50:30 (low power, tiny PCIe traffic). From about 03:51 there were short bursts of real work (power up, PCIe spikes) without util rising above ~50, so the pod was waking up gradually, not hard-stuck for the whole 10 minutes. Conclusion unchanged: idle because it was not asked to work, not because it was stuck or throttled.

2. Node `rci-tide-gpu-05` (24 CPUs, 4 GPUs).
- verified: `1 - avg(rate(node_cpu_seconds_total{mode="idle",instance=~"rci-tide-gpu-05.sdsu.edu.*"}[2m]))` = 0.31 to 0.46 over the whole window (0.31 to 0.43 in the dip), about 8 to 11 of 24 cores; not saturated. `node_load1{instance=~"rci-tide-gpu-05.sdsu.edu.*"}` 6.9 to 22 (lowest, 7 to 9, at 03:50 to 03:54 in the dip; highest, 22, at 03:34 in the earlier stall). `node_memory_MemAvailable_bytes` ~472 GB flat. `sum(rate(node_disk_written_bytes_total{...}[2m]))` ~0.1 MB/s in the dip. Node network (corrected after review): `sum(rate(node_network_receive_bytes_total{instance=~"rci-tide-gpu-05.sdsu.edu.*",device!~"lo|veth.*|cali.*"}[1m]))` was about 1 MB/s except 69 to 78 MB/s at 03:42 to 03:45, which is when the dip begins (03:43:30); `node_network_transmit_bytes_total` (same selector) 0.1 to 0.3 MB/s except 539 MB/s at 03:45 and 277 MB/s at 03:51 (and 11.7 at 03:41). Our pod's own network counters were ~0 then, so this is not our pod's pod-network traffic. Whose traffic it was I did not establish (I tried a per-pod `topk` over `container_network_*` with a `node=` label; it returned nothing, so my earlier "from other pods" was unverified). Host-network traffic includes the CephFS kernel client's traffic for any pod on the node, ours included, so a CephFS read or write by our pod would show here and not in the container counters. The time match of the 70 MB/s receive burst with the dip start, and the 539 and 277 MB/s transmit bursts at 03:45 and 03:51, is a coincidence I can't attribute. Speculative: it keeps the CephFS-stall idea alive (weak candidate below). Not checked: CephFS client or MDS metrics.
- verified, neighbours: `sum by (pod)(rate(container_cpu_usage_seconds_total{node="rci-tide-gpu-05.sdsu.edu",container!=""}[2m]))`: other jobs used 1 to 3.4 cores each (`reachy-dataset-xp1-e20-000-x9j5h` 2.4 to 3.4, dropping to 0.9 at 03:50 to 03:54; `ncv-lee-f2500-11-l2brf` ~1.2; `rajia-nov05-e3-bank-0-d2vkd` 0.2 to 1.2 from 03:42). `DCGM_FI_DEV_GPU_UTIL{Hostname="rci-tide-gpu-05.sdsu.edu"}` shows our pod alone on GPU 3; the other jobs are on GPUs 0, 1, 2. No CPU or GPU contention is visible. I did not look at CephFS-side or network-filesystem metrics for the node.

3. Pod 0 (`-0-sstst`, L4, nautilus-it-gpu06; no DCGM there).
- verified, per-60 s done increments from the saved logs (the command from the task):
  - pod 1: `8 8 8 8 8 4 0 4 0 7 7 7 11 12 12 12 0 1 0 3 1 3 4 3 4 5 12 12 12 13 7 9 13 11 11 12 10 8 8 11 9 9 12 11`
  - pod 0: `5 7 8 8 8 6 2 4 4 7 6 11 12 12 12 4 1 4 1 3 5 3 8 12 12 11 11 9 12 12 10 11 12 12 6 11 11 8 10 12 11`
- Pod 0 has both slow stretches as well, at the same place in the job order: a first one at done 40 to 56 (pod 1: 46 to 56) and a second at done 116 to about 145 (pod 1: 120 to 144). Pod 0's second stretch is 1044 to 1524 s on its rollout clock; the clock reads 2612 s at the "finished" line, shortly before the container ended at 00:12:12 on 2026-11-05, so that is about 23:46 to 23:54 on 2026-11-04 (my arithmetic).
- verified, same Prometheus pattern: `sum(rate(container_cpu_usage_seconds_total{pod="stacking-encoders-proxy-pilot-rollout2-0-sstst",container!=""}[2m]))` 2.3 to 4.6 cores over 23:46 to 23:54 on 2026-11-04 (4.64 at 23:48, the peak of its run); `container_cpu_cfs_throttled_seconds_total` rate 0 after 23:32; working set 15.5 to 15.8 GB flat; `container_fs_writes_bytes_total` rate 0.0005 MB/s; network ~1e-5 MB/s. Same flat stretches, busy CPUs, no visible I/O, on a different node and a different GPU model.
- verified (arithmetic from `results/pilot/pilot-subset.json` and `make_jobs` in rollout.py): both pods take all 340 jobs (shard 0/1) in the same order: the 100 subset episodes in file order (libero_10, libero_goal, libero_object, libero_spatial; within a suite by category), each with eval seeds 41, 42, 43 in a row, then 40 clean trials. 4 envs take the next jobs, so the done count is about the job index minus up to 4. Job index 45 to 56 = libero_10 "Sensor Noise" (episodes 15 to 18); 120 to 131 = libero_goal "Sensor Noise" (episodes 40 to 43); 132 to 143 = libero_goal "Objects Layout" (episodes 44 to 47); libero_10 "Objects Layout" is jobs 57 to 65 (episodes 19 to 21; corrected, I first wrote 57 to 62) and libero_10 "Light Conditions" 66 to 74. Exact dispatch counts (340 minus the `queued` column of the progress lines), pod 1: at 1105 s and 1165 s 124 dispatched, done 120 both times (so jobs 120 to 123, all Sensor Noise, were each running for at least a minute without finishing); 128 at 1345 s; 132 at 1465 s (all of Sensor Noise, job 131, dispatched by then); 136 at 1525 s; 143 at 1645 s (all of the libero_goal Objects Layout jobs dispatched); 148 at 1706 s (done 144). Done 128 at 1465 s, 132 at 1525 s, 144 at 1706 s. Pod 0: 129 dispatched at 1224 s, 133 at 1344 s, 141 at 1464 s, 149 at 1524 s (done 145). So Sensor Noise jobs ran from about 1105 to 1465 s in pod 1 (and about 1104 to 1344 s in pod 0), and Objects Layout jobs were dispatched from 1465 to 1645 s (pod 1) and 1344 to 1464 s (pod 0). In pod 1 the increments stayed low (4, 3, 4, 5 per minute at 1465 to 1706 s) while Objects Layout jobs were being dispatched; in pod 0 the same stage went 5, 3, then 8 (done 129, 134, 137, 145 at 1344 to 1524 s). Because completion order is not strictly dispatch order, the tail of the stretch can still contain slow Sensor Noise stragglers; I cannot tell from these counts whether Objects Layout is slow itself (pod 1: possibly; pod 0: less clear). So the claim is softened: Sensor Noise is the category that lines up in all four cases, Objects Layout is unresolved. Milder slowing (increments 7, 9 in pod 1 at done ~193 to 209; 8, 8 at ~266 to 282) lines up with libero_object and libero_spatial "Sensor Noise" (jobs 195 to 206, 270 to 278) as before. libero_10 "Light Conditions" (66 to 74) shows no visible slowing; libero_10 "Objects Layout" (57 to 65) went 7, 7, 7 per minute in pod 1 at done ~57 to 73, a little below the usual 8 to 12 (not clearly slow). So the flat stretches follow the job order, not the pod or node. The mapping is my arithmetic, not checked against per-episode `wall_s`.
- recalled (partial support from an earlier run, read again this session in `results/logs/stacking-encoders-proxy-sim.log` lines ~181-208, sim Job try 2 smoke of 2026-10-29): hold-still policy, OSMesa CPU rendering, no GPU, 4 envs at once, one episode per cell, so a different setup from the pilot (software rendering makes every step slower, tasks within a suite differ by category, and concurrent episodes share 4 CPUs). Seconds per episode, Sensor Noise vs the other categories of the same suite: libero_spatial 350.6 vs 274 to 291 (Light Conditions 570, an outlier); libero_object 215.7 vs 130 to 148; libero_goal 275.6 vs 196 to 217 (Objects Layout 217.0, in line with the others); libero_10 393.1 vs 264 to 431 (different base tasks, so not comparable). Same steps (430) in each suite. So Sensor Noise was 20 to 60% slower in three of four suites, and Objects Layout was about the same in libero_spatial (286.0 s vs 274 to 291) and libero_goal (217.0 vs 196 to 211), and about 10% higher in libero_object (147.8 vs 129 to 134). That supports Sensor Noise as the CPU-heavy category and weakens Objects Layout. It does not show where in the step the time goes, and with CPU rendering it is not the pilot's EGL setup.

4. What in rollout.py could keep the policy from being called while episodes keep running.
- Blocking JSONL write in the main loop (`fo.write(...); fo.flush()` in `run_jobs`, on CephFS): would stall every worker waiting for its chunk, so CPU would fall; it rose. It is also one short line per finished episode. Not supported by the evidence, but not excluded either, because the container counters can't see CephFS. Weak candidate.
- Episode setup: `run_episode` builds a new `OffScreenRenderEnv` for every job (`libero_env`: reads the bddl scene, compiles the model, new EGL context) and then 30 wait steps with no policy call. In principle the same in every category, so it does not explain a category-clustered dip alone; if scene size differs by category ("Objects Layout" names contain "add_N" / "moved") it could contribute. Speculative.
- Policy batching: the main loop runs a batch when every busy worker is waiting or 0.05 s have passed, so no deadlock; with CPU-bound workers the batches are just fewer. A symptom, not a cause. (The logged `mean batch` is a running average over the whole run, so it cannot show a dip in the stretch; I should not have cited it as evidence.)
- All 4 envs on CPU-heavy episodes at once: best fit to the evidence. Each worker asks the policy once per 10 simulator steps, so GPU use is proportional to simulator steps per second. If a category makes each step cost much more CPU, 4 workers at ~1 core each are CPU-bound and the GPU waits. Note on units (checked in rollout.py `run_episode`): `steps` in a record includes the 30 wait steps, and `wall_s` includes building the env, so seconds per step from them is a rough number. The LIBERO-Plus "Sensor Noise" category needs the ImageMagick `Wand` library (verified: `src/setup_sim.sh` installs libmagickwand-dev and Wand; LOG.md 2026-10-29 says the local smoke skipped Sensor Noise because wand was stubbed). That the perturbation is what costs the CPU time is speculative. Possible extra (speculative): the Job sets no `OMP_NUM_THREADS` / `MAGICK_THREAD_LIMIT` (verified by grep of `jobs/rendered/job-pilot-rollout2.yaml`; the workers only call `torch.set_num_threads(1)`), so ImageMagick or OpenCV could start up to 24 threads per worker on this node. CFS throttling was 0, so there is no direct sign of it.
- Because the 4 workers are in the same category at the same time (consecutive jobs), one slow category makes the whole pod's GPU use dip instead of averaging with fast categories.

Not available for the pilot: the actual per-episode `wall_s` and `steps` by category. Each JSONL record has both, so seconds per step by category from the two JSONLs on CephFS would test the Sensor Noise idea. It would not settle it alone: Sensor Noise episodes run as one consecutive block, so a pod-wide stall in those minutes (for example a CephFS stall) would also inflate their `wall_s`. Separating the two needs non-Sensor-Noise episodes that overlapped the block in time (with 4 envs, the episodes running next to a Sensor Noise one at the block edges) or a rerun in interleaved order (option B). Reading the JSONLs needs a pod with the PVC (none started; the user's OK is needed). The same read is needed for the success rates (LOG (b) Next 2), so it can be one Job.

5. Fix options (proposals only; nothing done).
- A. Read the two JSONLs first (cheap check; a CPU-only Job, no GPU): `wall_s / steps` by category and job index, to see whether Sensor Noise and Objects Layout episodes are slower per step (with the caveat above: a block-wide stall would look the same, and `steps` includes wait steps, `wall_s` env build). If they are not, this diagnosis is wrong and the next suspect is a CephFS write stall in those minutes; checking that would need per-record write timing, which the records don't have (a small timing log around `fo.flush()` in a rerun would).
- B. Interleave the job order (a fixed seeded shuffle or round-robin over categories; per rollout.py's header, an episode's result does not depend on batching or order). Cost: a small change in `make_jobs`, a new ConfigMap; no change to results. Effect: removes the deep dips, but the average stays about the same (pod 1 whole-run nvidia-smi mean 42.6%), so it would sit near the 40% floor with little margin.
- C. More environments per GPU (RUN.md's remedy), with CPU raised in proportion: the pod used up to 4.7 of its 4.8-core limit with 4 envs (3.2 to 4.7 in the dip, 2.1 to 4.1 in other minutes from 03:27), so 8 envs need about 8 CPUs or the CPU-bound stretches stay CPU-bound. Cost: larger CPU and RAM request (about 3 GB per worker more), so longer Pending (pod 1 waited about 5 h at 4 CPU / 20Gi). Effect: should raise the average if the GPU is the side waiting.
- D. Prefer slower GPUs for rollouts: like for like (each pod's own whole-run nvidia-smi summary line, 30 s samples, in the saved logs): pod 0 on the L4 `mean 56.7% over 88 samples`, pod 1 on the L40 `mean 42.6% over 94 samples`; policy busy 57% vs 43%. (I first compared the L4's 5-min in-pod sample, ~61%, with the L40's whole-run mean, which was not like for like.) Cost: no DCGM series at nautilus-it-gpu06, so the floor can't be checked in Prometheus there; the policy needs ~10 GB GPU memory so an L4 is enough. Effect: raises the percentage, not the useful work per GPU-hour; whether that satisfies NRP's floor is the user's call.
- E. Exclude `rci-tide-gpu-05`: not supported; pod 0 on another node has the same flat stretches.
- F. Cap threads in the workers (`OMP_NUM_THREADS=1`, `MAGICK_THREAD_LIMIT=1`, `OPENBLAS_NUM_THREADS=1`). Cheap, but a guess until A shows where the time goes.
- G. Report as is: the pilot is a sizing run; the user decides whether the 40% windows apply at this size (RUN.md says they do; stop 10).

Recommendation: A first (needed anyway for the success rates, and no GPU), then B plus C if the slow categories turn out CPU-heavy (B for the dips, C for the average). Not E. D only if the user accepts losing the Prometheus check.
**Next:** the user decides whether to approve the CPU-only Job that reads the two JSONLs (success rates and `wall_s` / `steps` by category), and which of B, C, D, F (if any) to apply before any stage 4 rollout. Nothing on the cluster or in the code has been changed.

## 2026-11-05 (d) [agent] (pilot read Job: JSONLs recovered; sizing, cap, stage 4 estimate, timing by category)
Worker "pilot-read". The user approved submitting the read Job in this session (2026-11-05: "go, submit the pilot-read job"). Paths relative to this folder. Nothing on the cluster was deleted; the other pod in the namespace (`vissam-feedback-stage1-v1-*`) was not touched. No GPU Job, no change to `src/` apart from the new read-only `src/pilot_timings.py`, no change to any locked rule text.

**1. The read Job**
- verified (read the whole manifest `jobs/rendered/job-pilot-read.yaml`, cherry-picked from 815416b): one container, image `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`, requests cpu 500m / mem 512Mi, limits 600m / 600Mi, no GPU, the PVC `stacking-encoders-proxy` mounted `readOnly: true` at /mnt/proxy, only `ls`, `sha256sum`, `wc -l`, `gzip | base64`; no writes, no sleep, no tolerations, us-west, deadline 7200 s, ttl 86400. Not edited.
- verified: `kubectl create --dry-run=server -f jobs/rendered/job-pilot-read.yaml` -> `job.batch/stacking-encoders-proxy-pilot-read created (server dry run)`; then `kubectl create -f ...` -> `job.batch/stacking-encoders-proxy-pilot-read created`. (The shell's `date -u` printed `Fri Nov  6 07:13:56 UTC 2026` at submit, which does not match the 2026-11-05 clock of the other entries; I did not investigate.)
- verified: one-shot checks: ContainerCreating for about 6 min (image pull, "Successfully pulled image ... in 3m1.838s"), then `pod/stacking-encoders-proxy-pilot-read-stvfp   0/1   Completed   0   6m32s`.
- verified: log saved with `kubectl logs job/stacking-encoders-proxy-pilot-read -n aria-world > results/logs/pilot-read-stvfp.log` (50,493 bytes, ends `=== done`).
- verified (from that log): `pilot2-dinov3-s0.0of1.jsonl` 165,253 bytes, SHA256 `c07ab4a83598fed2ae8e86856535f66e7c170db3c90f024c7082cf9479dc2c01`, LINES 340. `pilot2-dinov3+c-radiov4-s0.0of1.jsonl` 172,047 bytes, SHA256 `0670ec3e168912aceaf47ba108a7c4ca88fddba8cf46fed67a9fbc214660d111`, LINES 340.
- verified: decoded (base64, gunzip, in Python) into `results/pilot/`; the decoded bytes' sha256 equals the Job's printed value for both (script printed True, True), and:
  ```
  c07ab4a83598fed2ae8e86856535f66e7c170db3c90f024c7082cf9479dc2c01  pilot/pilot2-dinov3-s0.0of1.jsonl
  0670ec3e168912aceaf47ba108a7c4ca88fddba8cf46fed67a9fbc214660d111  pilot/pilot2-dinov3+c-radiov4-s0.0of1.jsonl
       340 pilot/pilot2-dinov3-s0.0of1.jsonl
       340 pilot/pilot2-dinov3+c-radiov4-s0.0of1.jsonl
  ```
  `.gitignore` does not ignore them (`git check-ignore` exit 1) and they are small text files, so they are committed.

**2. Test under Python 3.11** (venv in the session scratchpad, not the repo: `/opt/homebrew/bin/python3.11 -m venv <scratchpad>/venv311; pip install numpy==2.2.6`. `lp_eval.py` and `test_lp_eval.py` import only the stdlib and numpy; numpy is pinned as in requirements.txt.)
- verified: `python --version` -> `Python 3.11.14`.
- verified: `python world-model-stacking-encoders/src/test_lp_eval.py`:
  ```
  1 ok: proportional subset (sums to N, within 1 of share, >= 1 per cell, exclusion, determinism)
  2 ok: per-episode averaging over eval seeds, planted per-seed drops exact (-0.660, -1.650, +0.330 pts)
  3 ok: identical configs give exactly 0
  4 ok: missing / duplicate / foreign eval seeds, missing episodes, foreign-subset rows rejected
  5 ok: locked verdict on 14 planted cases incl. every boundary; missing train seed rejected
  6 ok: sizing recovers planted sd (33.501 pts -> N 1386)
  7 ok: CIs resample whole episodes
  8 ok: secondary per-config, per-category and clean numbers
  9 ok: CLI end to end (subset with exclusion, sizing, analyze + verdict)
  all checks passed
  ```

**3. Sizing (Step 9, locked rule; `--train-seed 0`, the pilot's train seed, per `lp_eval.py sizing --help`)**
- verified, command and output verbatim:
  ```
  python world-model-stacking-encoders/src/lp_eval.py sizing --subset results/pilot/pilot-subset.json --results results/pilot/pilot2-dinov3-s0.0of1.jsonl results/pilot/pilot2-dinov3+c-radiov4-s0.0of1.jsonl --train-seed 0
  pilot sizing (train seed 0, 100 episodes): sd 3.333 pts, sd* 5.000 pts, N 14 (SE target 1.35 pts)
  ```
  sd = 3.333, sd* = 1.5 x sd = 5.000, N = ceil((5.000 / 1.35)^2) = 14.
- verified (counted from the files): the pilot checkpoints almost never succeed. Pilot sanity counts (rollouts with success = 1), sizing inputs only, not results: `dinov3`: LIBERO-Plus 1 of 300 rollouts, clean LIBERO 0 of 40. `dinov3+c-radiov4`: LIBERO-Plus 0 of 300, clean 0 of 40. The one success is libero_goal episode 1410 ("Language Instructions"), eval seed 43, `dinov3` only. That is a per-episode difference of one third in one episode and 0 in the other 99, which gives an sd of about 3.3 points (reasoned, matches the printed value). **So sd, and therefore N = 14, rests on a single successful rollout out of 600. It is not a usable estimate of how much the two real (10-epoch) models will disagree. N = 14 is also below the proportional draw's 28-cell minimum (at least 1 per cell).** Flagged for the user; the rule is locked and I did not alter it or choose for the user. (speculative: 500-step checkpoints are too weak for a paired-difference sd to mean anything; fixing that would change the locked pilot design, so it needs the user's decision.)

**4. Cap and stage 4 estimate (RUN.md Step 9 cap: 15 checkpoints x (subset x 3 eval seeds + clean LIBERO) in about 48 h on at most 8 GPUs)**
- Method (verified arithmetic; the small script is in the session scratchpad only): speed from each pod's own log and records. GPU-seconds per rollout = the pod's rollout elapsed time (pod 0: 2530 s, pod 1: 2729 s, from the "finished ... in Ns" lines) split between LIBERO-Plus and clean in proportion to the sum of the records' `wall_s` of each kind. Budget = 8 GPUs x 48 h = 384 GPU-h. Per checkpoint: 3N LIBERO-Plus rollouts + 40 clean (the pilot's 40 clean trials; assumption: the final clean list is the same size, RUN.md does not fix its size). Cap N = (384 GPU-h x 3600 / 15 - clean cost) / (3 x GPU-s per LIBERO-Plus rollout). The estimate excludes pod startup (restore about 7 to 8 min plus model load per pod) and queue time.

| | pod 0 (D, L4) | pod 1 (D+R, L40) |
|---|---|---|
| rollout speed (log) | 484 episodes/h, 2530 s for 340 | 449 episodes/h, 2729 s for 340 |
| GPU-s per LIBERO-Plus / clean rollout | 7.64 / 5.97 | 8.32 / 5.85 |
| cap N (episodes) | 4012 | 3684 |
| stage 4 rollouts at N = 14 (42 + 40 per checkpoint x 15 checkpoints) | 2.3 GPU-h = 0.29 h on 8 GPUs | 2.4 GPU-h = 0.30 h on 8 GPUs |
| (reference) all 10,030 episodes | 959 GPU-h = 120 h on 8 GPUs | 1044 GPU-h = 130 h on 8 GPUs |
| (reference) SE at the cap with sd* = 5.0 | 0.079 pts | 0.082 pts |

- N = 14 does not exceed either cap (about 3700 to 4000), so the cap does not bind and no reduced N is needed. The cap N and its SE are listed for reference only: the SE uses sd* = 5.0 from the degenerate pilot above, so it is not a precision claim. The two speeds differ by about 8%; I did not pick one. Pod startup, queue time and any GPU-floor fix that changes envs per pod would move the real numbers (not estimated). Stage 4 training time is not in this estimate.

**5. Timing by suite x category (option A of LOG (c)); `src/pilot_timings.py` reproduces it**
- Command (verified; prints tables only, writes nothing): `python world-model-stacking-encoders/src/pilot_timings.py --subset results/pilot/pilot-subset.json --results results/pilot/pilot2-dinov3-s0.0of1.jsonl results/pilot/pilot2-dinov3+c-radiov4-s0.0of1.jsonl --logs results/logs/pilot-rollout2-0-sstst.log results/logs/pilot-rollout2-1-lz8ts.log --dip 1:1151:1766`. Rows are in job order (subset episode order x eval seeds, then 40 clean). Columns: n, mean `wall_s`, mean `steps`, mean `wall_s`/`steps`, successes. Caveats kept from LOG (c): `steps` includes the 30 wait steps and `wall_s` includes building the env, so wall/steps is rough. Almost every episode ran to the step limit, so `steps` is the same inside a suite (730 in libero_10, 430 elsewhere) and the wall_s differences are real time differences.
- Summary (verified; pod 0 = D on the L4, pod 1 = D+R on the L40; entries are mean wall_s, then mean wall_s/steps). The full table is the script output at the end of this entry.

| suite / category (jobs) | pod 0 | pod 1 |
|---|---|---|
| libero_10 / Sensor Noise (45-56) | 67.3 s, 0.092 | 91.9 s, 0.126 |
| libero_10 / other categories | 33-41 s, 0.045-0.056 | 29-39 s, 0.040-0.054 |
| libero_goal / Sensor Noise (120-131) | 138.1 s, 0.321 | 186.2 s, 0.433 |
| libero_goal / Objects Layout (132-143) | 15.4 s, 0.036 | 17.3 s, 0.040 |
| libero_goal / other categories | 18.6-20.7 s, 0.046-0.048 | 18.9-20.1 s, 0.044-0.047 |
| libero_object / Sensor Noise (195-206) | 27.9 s, 0.065 | 32.7 s, 0.076 |
| libero_object / other categories | 20.0-20.5 s, 0.046-0.048 | 18.9-19.5 s, 0.044-0.046 |
| libero_spatial / Sensor Noise (270-278) | 30.0 s, 0.070 | 35.9 s, 0.083 |
| libero_spatial / other categories | 20.7-22.3 s, 0.048-0.052 | 21.5-22.7 s, 0.050-0.053 |
| clean (40 trials) | 23.7 s, 0.047 | 23.3 s, 0.046 |

- verified: **Sensor Noise is the CPU-heavy category.** In all four suites and both pods its episodes are slower than the same suite's other categories: libero_10 about 2x, libero_object about 1.4 to 1.7x, libero_spatial about 1.4 to 1.7x, libero_goal about 7 to 10x on average (single libero_goal Sensor Noise episodes took up to about 400 s, wall/step about 0.92, against 15 to 20 s for the others). **Objects Layout is not slow** (libero_goal: 15.4 s and 17.3 s, the fastest category; in line with the rest of its suite in the other three). That settles the point LOG (c) left open: Objects Layout was not a cause; it only sat next to Sensor Noise in the job order.
- Records have no start or end times (verified: the keys are benchmark, suite, episode_id, category, config, train_seed, eval_seed, checkpoint, subset_id, success, steps, instruction, wall_s, final_state, worker_peak_rss_mb). So the overlap check uses estimated times: the file is written as episodes finish, so line i is the i-th completion; I interpolated its end time from the done counts in the pod's progress lines (about +-30 s) and set start = end - wall_s. Pod 1's GPU near-zero stretch (about 03:43:30 to 03:53:45 UTC per LOG (c)) maps to rollout clock about 1151 to 1766 s (it finished at 2813 s = 04:11:12; my arithmetic).
- verified, in that window (pod 1): the first four libero_goal Sensor Noise episodes (jobs 120 to 123) started at about 1098 to 1140 s and took 127 to 205 s each, jobs 124 and 125 took 127 s, and jobs 126, 127, 128 took 391 to 397 s each, ending at about 1682, 1706 and 1736 s. So Sensor Noise episodes were running in every part of the window, and the last of them finished after it. Only jobs 129 to 131 of the block finished normally (17 to 20 s).
- verified (the check LOG (c) asked for): the non-Sensor-Noise episodes that overlapped the block ran at normal speed. Pod 1: the libero_goal Objects Layout (jobs 132 to 143) and Light Conditions (144 to 152) episodes that ran between about 1480 and 1750 s took 14.9 to 23.5 s each (measured wall_s), wall/step 0.035 to 0.055, the same as in the rest of the run. So the slowness stayed with the Sensor Noise episodes and did not slow their neighbours on the same pod at the same time. That argues against a pod-wide stall in those minutes (for example a CephFS stall, which would have slowed all four workers) and for a per-episode cost of Sensor Noise. Caveat (reasoned): the placement of the neighbours in time depends on the +-30 s estimate, but their wall_s values are measured and short, which is what carries the argument.
- Does it support the CPU-heavy-category idea? Yes for Sensor Noise (verified above), no for Objects Layout. That the slowness comes from the perturbation's CPU work (Wand/ImageMagick) is still speculative; the table only shows that the category costs 1.4x to 10x the time per step. The GPU near-zero stretch then follows LOG (c)'s mechanism (speculative, consistent with the data): while 3 to 4 workers each ran a Sensor Noise episode at 0.3 to 0.9 s per step, the policy was called rarely.
- Pod 0 (L4) shows the same pattern in the table. It has no DCGM series, so no dip window was mapped for it and the overlap listing was run for pod 1 only.
- Consequence for LOG (c)'s fix options (not applied; the user chooses): option B (interleave the job order) fits the data, since the libero_goal Sensor Noise block costs 7 to 10x per episode and putting such episodes among fast ones would stop all four workers being slow together. That is a reading of the table, not a test.

**Next:** (1) The user decides how to handle the sizing: the locked rule gives N = 14, but sd rests on one successful rollout in 600 (500-step pilot checkpoints). Whether N = 14 is acceptable (the proportional draw needs at least 28 anyway), or the sizing basis should change (which amends the locked pilot design), is the user's call. (2) The user picks the GPU-floor fix (B and/or C of LOG (c)); the Sensor Noise result supports B. (3) The final subset draw (proportional, excluding the pilot episodes) has not been done; it waits on (1).

Output of `pilot_timings.py` (verbatim, full):
```

=== experiments/2026-10-27-proxy-calibration/results/pilot/pilot2-dinov3-s0.0of1.jsonl  (340 records)
cell (suite / category)                    jobs        n   mean_wall_s  mean_steps  mean_wall/steps(s)  succ
libero_10 / Background Textures            0-8          9         40.6       730.0              0.0557     0
libero_10 / Robot Initial States           9-20        12         33.5       730.0              0.0459     0
libero_10 / Camera Viewpoints              21-32       12         32.8       730.0              0.0450     0
libero_10 / Language Instructions          33-44       12         33.2       730.0              0.0454     0
libero_10 / Sensor Noise                   45-56       12         67.3       730.0              0.0922     0
libero_10 / Objects Layout                 57-65        9         32.8       730.0              0.0449     0
libero_10 / Light Conditions               66-74        9         34.3       730.0              0.0470     0
libero_goal / Background Textures          75-83        9         20.7       430.0              0.0480     0
libero_goal / Robot Initial States         84-95       12         20.0       430.0              0.0466     0
libero_goal / Camera Viewpoints            96-107      12         19.7       430.0              0.0459     0
libero_goal / Language Instructions        108-119     12         18.6       408.1              0.0456     1
libero_goal / Sensor Noise                 120-131     12        138.1       430.0              0.3212     0
libero_goal / Objects Layout               132-143     12         15.4       430.0              0.0359     0
libero_goal / Light Conditions             144-152      9         20.2       430.0              0.0470     0
libero_object / Background Textures        153-158      6         20.5       430.0              0.0477     0
libero_object / Robot Initial States       159-170     12         20.3       430.0              0.0471     0
libero_object / Camera Viewpoints          171-182     12         20.0       430.0              0.0465     0
libero_object / Language Instructions      183-194     12         20.0       430.0              0.0465     0
libero_object / Sensor Noise               195-206     12         27.9       430.0              0.0649     0
libero_object / Objects Layout             207-218     12         20.0       430.0              0.0465     0
libero_object / Light Conditions           219-227      9         20.0       430.0              0.0464     0
libero_spatial / Background Textures       228-236      9         20.7       430.0              0.0482     0
libero_spatial / Robot Initial States      237-245      9         22.3       430.0              0.0518     0
libero_spatial / Camera Viewpoints         246-257     12         21.7       430.0              0.0504     0
libero_spatial / Language Instructions     258-269     12         21.0       430.0              0.0488     0
libero_spatial / Sensor Noise              270-278      9         30.0       430.0              0.0698     0
libero_spatial / Objects Layout            279-290     12         21.9       430.0              0.0508     0
libero_spatial / Light Conditions          291-299      9         21.9       430.0              0.0510     0
clean /                                    clean       40         23.7       505.0              0.0470     0

=== experiments/2026-10-27-proxy-calibration/results/pilot/pilot2-dinov3+c-radiov4-s0.0of1.jsonl  (340 records)
cell (suite / category)                    jobs        n   mean_wall_s  mean_steps  mean_wall/steps(s)  succ
libero_10 / Background Textures            0-8          9         39.1       730.0              0.0536     0
libero_10 / Robot Initial States           9-20        12         29.5       730.0              0.0405     0
libero_10 / Camera Viewpoints              21-32       12         30.3       730.0              0.0415     0
libero_10 / Language Instructions          33-44       12         30.7       730.0              0.0421     0
libero_10 / Sensor Noise                   45-56       12         91.9       730.0              0.1259     0
libero_10 / Objects Layout                 57-65        9         34.4       730.0              0.0471     0
libero_10 / Light Conditions               66-74        9         31.6       730.0              0.0433     0
libero_goal / Background Textures          75-83        9         20.1       430.0              0.0468     0
libero_goal / Robot Initial States         84-95       12         19.2       430.0              0.0446     0
libero_goal / Camera Viewpoints            96-107      12         19.9       430.0              0.0463     0
libero_goal / Language Instructions        108-119     12         19.2       430.0              0.0447     0
libero_goal / Sensor Noise                 120-131     12        186.2       430.0              0.4330     0
libero_goal / Objects Layout               132-143     12         17.3       430.0              0.0403     0
libero_goal / Light Conditions             144-152      9         18.9       430.0              0.0439     0
libero_object / Background Textures        153-158      6         19.5       430.0              0.0455     0
libero_object / Robot Initial States       159-170     12         19.3       430.0              0.0450     0
libero_object / Camera Viewpoints          171-182     12         18.9       430.0              0.0441     0
libero_object / Language Instructions      183-194     12         19.0       430.0              0.0443     0
libero_object / Sensor Noise               195-206     12         32.7       430.0              0.0760     0
libero_object / Objects Layout             207-218     12         18.9       430.0              0.0440     0
libero_object / Light Conditions           219-227      9         19.1       430.0              0.0444     0
libero_spatial / Background Textures       228-236      9         21.6       430.0              0.0503     0
libero_spatial / Robot Initial States      237-245      9         22.3       430.0              0.0518     0
libero_spatial / Camera Viewpoints         246-257     12         21.5       430.0              0.0501     0
libero_spatial / Language Instructions     258-269     12         21.9       430.0              0.0509     0
libero_spatial / Sensor Noise              270-278      9         35.9       430.0              0.0834     0
libero_spatial / Objects Layout            279-290     12         22.7       430.0              0.0528     0
libero_spatial / Light Conditions          291-299      9         22.1       430.0              0.0515     0
clean /                                    clean       40         23.3       505.0              0.0461     0

-- episodes whose estimated run [start, end] overlaps rollout-clock 1151-1766 s (pod 1)
line job  suite / category                         eval_seed  start_s   end_s  wall_s  steps  wall/steps
 120 123  libero_goal / Sensor Noise                 41      1098    1225   127.4    430    0.2963
 121 120  libero_goal / Sensor Noise                 41      1102    1305   203.1    430    0.4723
 122 121  libero_goal / Sensor Noise                 42      1121    1325   204.2    430    0.4748
 123 122  libero_goal / Sensor Noise                 43      1140    1345   205.2    430    0.4771
 124 124  libero_goal / Sensor Noise                 42      1278    1405   127.0    430    0.2954
 125 125  libero_goal / Sensor Noise                 43      1298    1425   127.0    430    0.2954
 126 129  libero_goal / Sensor Noise                 41      1425    1445    19.9    430    0.0463
 127 130  libero_goal / Sensor Noise                 42      1448    1465    17.2    430    0.0401
 128 131  libero_goal / Sensor Noise                 43      1462    1480    18.0    430    0.0419
 129 132  libero_goal / Objects Layout               41      1480    1495    15.3    430    0.0355
 130 133  libero_goal / Objects Layout               42      1495    1510    14.9    430    0.0347
 131 134  libero_goal / Objects Layout               43      1510    1525    15.5    430    0.0360
 132 135  libero_goal / Objects Layout               41      1530    1545    15.1    430    0.0352
 133 136  libero_goal / Objects Layout               42      1550    1565    15.0    430    0.0349
 134 137  libero_goal / Objects Layout               43      1569    1585    15.7    430    0.0364
 135 138  libero_goal / Objects Layout               41      1578    1600    22.5    430    0.0522
 136 139  libero_goal / Objects Layout               42      1597    1615    18.3    430    0.0425
 137 140  libero_goal / Objects Layout               43      1611    1630    19.4    430    0.0450
 138 141  libero_goal / Objects Layout               41      1624    1645    21.2    430    0.0493
 139 142  libero_goal / Objects Layout               42      1640    1657    17.5    430    0.0406
 140 143  libero_goal / Objects Layout               43      1652    1669    17.7    430    0.0412
 141 126  libero_goal / Sensor Noise                 41      1285    1682   396.4    430    0.9219
 142 144  libero_goal / Light Conditions             41      1675    1694    18.3    430    0.0427
 143 127  libero_goal / Sensor Noise                 42      1309    1706   397.4    430    0.9243
 144 145  libero_goal / Light Conditions             42      1694    1711    16.6    430    0.0387
 145 146  libero_goal / Light Conditions             43      1698    1716    17.7    430    0.0412
 146 147  libero_goal / Light Conditions             41      1698    1721    23.2    430    0.0540
 147 148  libero_goal / Light Conditions             42      1709    1726    17.1    430    0.0399
 148 149  libero_goal / Light Conditions             43      1713    1731    18.2    430    0.0423
 149 128  libero_goal / Sensor Noise                 43      1345    1736   391.4    430    0.9103
 150 151  libero_goal / Light Conditions             42      1724    1741    17.2    430    0.0400
 151 150  libero_goal / Light Conditions             41      1722    1746    23.5    430    0.0547
 152 152  libero_goal / Light Conditions             43      1733    1751    18.1    430    0.0420
 153 153  libero_object / Background Textures        41      1735    1756    20.8    430    0.0483
 154 155  libero_object / Background Textures        43      1742    1761    18.9    430    0.0440
 155 154  libero_object / Background Textures        42      1745    1766    20.8    430    0.0483
 156 156  libero_object / Background Textures        41      1751    1771    19.9    430    0.0464
 157 158  libero_object / Background Textures        43      1758    1776    17.7    430    0.0413
 158 157  libero_object / Background Textures        42      1762    1781    19.2    430    0.0446
```
