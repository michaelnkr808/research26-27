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
