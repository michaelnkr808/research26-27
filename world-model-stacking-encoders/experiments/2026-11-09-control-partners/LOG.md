# Experiment log: control partners

Step-by-step record for this experiment (jobs, errors, reruns, exact numbers). Append-only, newest at the bottom. The direction's `../../LOG.md` keeps short summaries that point here.

The design is in `PLAN.md` and was signed off by the user 2026-11-09 (margins 0.005 input / 0.003 target, one seed first, global features only).

## 2026-11-10 [agent, lead] stage 1 submitted (ConfigMap, then Job)
**Goal:** submit stage 1 of the control-partners run (SYNTH_SEEDS "0"; one seed first) as one CPU Job.
**Did:** the user said in chat on 2026-11-10: "yeah if it only needs cpu then create the configmap to be a cpu only job and submit it". This moves the run ahead of perturbed-frames phase B's GPU work; the earlier order was only so the two would not compete for the user's attention, and this run needs no GPU. All facts below are verified by the implementation lead this session unless marked.
- Pre-checks at about 09:24 UTC (verified): main clean at 2274e32 and equal to origin/main; no ConfigMap `stacking-encoders-control-src` and no Job `stacking-encoders-control-partners` existed; no pod in namespace aria-world mounted PVC `michael-persistent`; the only other Job of ours active was `stacking-encoders-pfb-render` (pod Pending, uses PVC `stacking-encoders-proxy`, no conflict). Not checked before submit: that the feature folders are still on the PVCs (the Job's staging step fails fast if not) and whether the output folder `/mnt/features/stacking-encoders/control-partners/` already held files.
- ConfigMap `stacking-encoders-control-src` created with keys `diagnostics.py` and `target_diagnostics.py` from `src/` at main 2274e32. sha256 read back from the cluster equals the repo files: diagnostics.py 7af1f7aa2d595d9e0b7f1fb462f7901c6fa9594d2da7895b0168f0ef3ea2d70d, target_diagnostics.py 48bf7454d99cfe0f3a50205d1aa4bdb55d959042bd7af3f10cae42fe57fc46d7. Both files were last changed in 5b81152 (PR #9).
- Manifest rendered with `sed "s/SET-AT-SUBMISSION/2274e32b383b82c6aa46140bbbfb7051a4c7ed3c/" job.yaml > rendered/job.yaml`; the only difference from `job.yaml` is line 109 (SRC_GIT_COMMIT). The submitted copy is `rendered/job.yaml`. Server dry run passed, then `kubectl apply`: Job `stacking-encoders-control-partners` created 2026-11-10 09:26:04 UTC (time printed right after apply). SYNTH_SEEDS "0" (stage 1), 2 CPU / 14Gi requests, 16 h deadline counting from creation, backoffLimit 1.
- Pod `stacking-encoders-control-partners-b4ps5` was Running within 45 s on node `rci-nrp-cpu-04.sdsu.edu`. At 09:27 UTC the log showed nproc 2, the pip install done, and pip freeze with numpy==2.2.6, h5py==3.16.0, huggingface_hub==0.36.2. Nothing later was seen at the time of this entry.
- A background wait in the lead's session polls the Job status every 5 min and reports when it completes or fails.
**PR:** direct (chore; no PR).
**Found:** nothing yet; no result read.
**GPU-hours:** 0 (CPU only).
**Test-set use:** as in the tally and target experiments, see PLAN "What is computed" (probes scored on the libero_goal test demos, clean frames, stored features).
**Not done / unverified:** RSS against the 14Gi request in the first minutes not yet checked; no result read.
**Next:** check RSS and the first run's start; when the Job ends, collect logs and JSONs from `/mnt/features/stacking-encoders/control-partners/` into `results/`, check the four gates, apply the decision rule word for word, and explain to the user before anything goes to key findings. Stage 2 (seeds 1 and 2) needs the user's OK.
