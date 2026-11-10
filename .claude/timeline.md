# research26-27 timeline

## 2026-10-17 — stacking-encoders: direction scoped + Fast-WAM compute benchmark

- Scoped a new direction from Liu et al. 2026 (arXiv 2610.13024): can cheap frozen-encoder diagnostics predict which encoder combos help a World Action Model, and why generalists beat specialists. Plan: real (smaller-scale) WAM runs, not a non-WAM proxy.
- Set the Nautilus compute rule (no GPU-hour cap; ≤8 GPUs / ≤20 Jobs; no sleep in Jobs) as a STANDARDS.md override.
- A 1-GPU Fast-WAM benchmark ran on an A6000 after several attempts were blocked by a saturated cluster: ~1.3 samples/s/GPU, so a full run is ~3–3.5 days on 8× A6000 (estimate). Gradient checkpointing is unsupported in Fast-WAM's compiled path.
- Open decision: training hardware (8× A6000 vs ZeRO-3 on 3090s vs the A100 form).
- See: stacking-encoders/LOG.md, stacking-encoders/experiments/2026-10-16-fastwam-step-bench/README.md

## 2026-10-18 — stacking-encoders: proxy-policy ground truth + Phase 1 extraction script

- Reversed the 10-17 plan: ground truth will come from a small proxy policy trained on cached frozen encoder features and scored in LIBERO rollouts. The first test is whether it reproduces Liu et al.'s orderings. Full Fast-WAM runs (~3 days each) are too expensive for multi-seed grids. Liu et al. released no code or checkpoints.
- Transcribed the paper's Table 1 (single encoders) and Table 10 (16 composition results). Most gains are within ±1 pt; DINOv3+C-RADIOv4 at −5.4 is the outlier.
- Wrote `src/extract_features.py` and smoke-tested it locally on real LIBERO data (SigLIP 2, C-RADIOv4). LIBERO frames need a 180° rotation.
- Blocked on the user: HF gated access (DINOv3, SAM 3, VGGT-Ω) and an `hf-token` k8s Secret. Then a 1-GPU extraction Job, on go-ahead.
- See: stacking-encoders/decisions/001-ground-truth-model.md, stacking-encoders/literature/encoder-weights.md, stacking-encoders/LOG.md

## 2026-10-19 — stacking-encoders: first cached features + first diagnostics (pilot)

- Checked both of Liu et al.'s tables against the arXiv HTML: exact match. My "seed discrepancy" in the catch-up was wrong. The paper uses one trained checkpoint per config; the 3 seeds are evaluation seeds. Also found that the paper's default WAM is joint video–action denoising, not Fast-WAM direct prediction.
- Ran the first GPU extraction Job (TITAN RTX, fp16, 53 min, 90–99% util): SigLIP 2-L + C-RADIOv4 for libero_goal, 127k frames, 9.6 GB on the 20 GB PVC. 4 suites won't fit, so a storage decision is pending (CephFS recommended).
- Wrote `src/diagnostics.py` + synthetic tests and ran it as a CPU Job. Action-probe R²: proprio 0.28, SigLIP 0.61, RADIO 0.67, both 0.68. Gains are asymmetric (+0.069 vs +0.008). CKA 0.72–0.94. The residual-information metric returned 0 on real data (probe overfits; speculative cause: within-demo frame correlation).
- Open for the user: drop resid(B|A) in favor of gain(B|A)? Add demo-level bootstrap CIs; DINOv3 license + HF token Secret.
- See: stacking-encoders/LOG.md (2026-10-18/19 and 2026-10-19), stacking-encoders/experiments/2026-10-18-extract-libero-goal/README.md, stacking-encoders/experiments/2026-10-19-diagnostics-libero-goal/README.md

## 2026-10-19 — stacking-encoders: metric decided (gain + bootstrap CIs), HF access, CephFS drafted

- User decided to drop the residual-information metric and use gain(B|A) with demo-level bootstrap CIs. Implemented and tested in `src/diagnostics.py`. Explained gain as the "unique to B" slice: shared 0.60, RADIO-only 0.07, SigLIP-only 0.01.
- HF access granted for DINOv3, SAM 3, and VGGT-Ω. The user created the `stacking-encoders-hf` Secret.
- Drafted a 100Gi CephFS volume for features, following the NRP docs. Dry run passed; not created.
- Waiting on go-aheads: create the volume, rerun diagnostics with CIs, copy the libero_goal features, run DINOv3 extraction.
- See: stacking-encoders/LOG.md (2026-10-19 (b), (c)), stacking-encoders/data/pvc-features-cephfs.yaml

## 2026-10-21 — stacking-encoders: feasibility tally (decision 002) done, negative on the simple metric

- Extracted DINOv3, SAM 3, VGGT-Ω for libero_goal on Nautilus (~10 GPU-h over 3 tries; fp16 DINOv3 overflowed, try 2 over-requested memory, SAM 3 ran 5.5 frames/s) and ran CPU diagnostics for D+L, D+R, D+S, D+G.
- Result (n = 4, user signed off): neither CKA nor gain(B|D) flags the paper's harmful D+R. R is the most similar to D and adds the most to it. Every partner adds a small positive gain on frozen features.
- Decision 002 accepted; CephFS feature PVC created (17.6G of features kept). All Jobs and ConfigMaps cleaned up. NOTES.md is now git-ignored. A handoff block went to the planning agent; direction to be chosen with the mentor on 2026-10-22.
- See: stacking-encoders/experiments/2026-10-19-feasibility-tally/README.md, stacking-encoders/README.md (key findings), stacking-encoders/LOG.md (2026-10-19 (e) through 2026-10-21 (h))

## 2026-10-21 — stacking-encoders: planning session, reframed to encoders-as-targets (H-E)

- Read the liu2026what PDF. The WAM uses encoder features as clean inputs *and* noised future targets, with one loss term per source (Eq. 9) and no stated normalization. Future prediction carries most of DINOv3's gain (Table 9: 76.5 / 81.9 / 84.0). D+R (78.6) is below both D and R alone.
- Our tally tested encoders only as inputs, which is why nothing flagged D+R. Proposed H-E (target-side interference) and pre-registered T0/T1/T2 tests, then amended them to per-source losses after reading the PDF.
- Wrote the implementation handoff. The implementation agent has since built the script and tests (commits 2e03e18..acc9d0e).
- See: stacking-encoders/LOG.md (2026-10-21 (i), (j)), stacking-encoders/experiments/2026-10-21-target-diagnostics/PLAN.md, stacking-encoders/writing/hypotheses-draft.md (top section)

## 2026-10-27 — stacking-encoders: three research proposals for the mentor (Google Docs)

- Wrote the mentor's 6-section proposal for stacking encoders, and found prior work that narrows our novelty claim: KAGES (2026) already picks frozen encoders to fuse by marginal gain, so what stays ours is encoders as prediction targets inside a WAM. Also OSBORN, and ReWAM's feature calibration. Removed a ReWAM "+12.2 points" number that came from a search snippet.
- Scouted world-model and embedding papers (the mentor's reading list plus about 25 checked on arXiv) and wrote two more strong candidate proposals: which embedding a latent world model should plan in (controllable share), and why latent world models only plan a few steps ahead (oracle-embedding decomposition). They got their own folders, `world-model-embedding-choice/` and `world-model-planning-horizon/`.
- Moved from Claude Docs to the user's own Google Docs (school Drive). Each proposal is now a one-page doc of about 600 words (same link) plus a separate "full notes" doc. Nothing was run on the cluster.
- See: stacking-encoders LOG 2026-10-25 to 2026-10-27 (d) (folder renamed by the user to world-model-stacking-encoders/, not yet committed), world-model-embedding-choice/README.md, world-model-planning-horizon/README.md

## 2026-10-27 — world-model-stacking-encoders: target-side diagnostics, first cluster run (OOM)

- Re-checked everything before the run: tests pass under the Job's package versions, the code matches the committed version, the volumes are free, and 42 CPU nodes have room. The first capacity count was wrong because Prometheus lacks the GPU-node label. Fixed by taking the node list from kubectl.
- Submitted the Job with the user's OK. It ran out of memory (OOMKilled) 22 min in, on the first and smallest pair. Linux needed more than 10 GiB, where the Mac measurement said 6 GiB for the biggest pair. Deleted the Job (with the user's OK) and saved the log. No target-side result yet, and no NRP violation (the pod died before checks start).
- Also found that data loading is slow and nearly idle (~12 min per pair at ~0 CPU), which would fall below NRP's usage floor on later pairs. The resubmit needs Linux-measured memory, lower memory use, and faster loading.
- See: world-model-stacking-encoders/LOG.md 2026-10-27 (f), world-model-stacking-encoders/experiments/2026-10-21-target-diagnostics/RUN.md, results/try1-oom-20261027T185533Z.log

## 2026-10-27 — world-model-stacking-encoders: after the mentor meeting, learning LIBERO

- The proposal presentation went badly. The mentor said to focus on the benchmark: learn LIBERO, how stacking encoders relates to it, and the "intermediate steps" before the final result. The user will make their own slides from now on.
- Wrote RUN.md, the spec the implementation agent used for the target-side Job (that run is in the entry above).
- The user read LIBERO and LIBERO-Plus. We inspected a real libero_goal demo file and found from the paper's Table 1 that encoders mostly buy robustness to visual perturbations. The combo table gives only totals, so we don't know where D+R loses.
- See: world-model-stacking-encoders/LOG.md 2026-10-27 (g), experiments/2026-10-21-target-diagnostics/RUN.md

## 2026-10-28 — world-model-stacking-encoders: target-side test run (fails), proxy policy code started

- Fixed the OOM-killed target-diagnostics Job (lower-memory code with bit-identical outputs, features staged to local disk, sized with tracemalloc) and ran it: 5 h 45 m, peak 12.85 GiB. C-RADIOv4 is not the largest target-side interference (VGGT-Ω is), so H-E fails the pre-registered rule. In key findings with user OK; Job and ConfigMap deleted.
- Read the paper's evaluation: it trains once and reports only LIBERO-Plus totals for combos. Decision 003 (another session): multi-seed proxy policy.
- Proxy PLAN steps 1–3: the paper's 1,712 episodes are Fast-WAM's `yuanty/LIBERO-fastwam` (512→224, MuJoCo 3.3.2), not our HDF5; LIBERO-Plus is 10,030 × 1 trial. Wrote `src/proxy_policy.py` (9 checks pass) and smoke-tested it locally with real DINOv3 + C-RADIOv4 (Mac now logged in to HF).
- Open for the user: pixel stream in every config, text encoder choice; then step 4 extraction Jobs on a go-ahead.
- See: world-model-stacking-encoders/experiments/2026-10-21-target-diagnostics/README.md, experiments/2026-10-27-proxy-calibration/PLAN.md, LOG.md 2026-10-27 (h)–(n) and 2026-10-28

## 2026-10-28 — world-model-stacking-encoders: faithful proxy built on Fast-WAM, reviewer framing, autonomous run set up

- Reviewed the 8 proxy design choices with the user; decision: match the paper over cheap stand-ins (Wan 2.2 VAE latents, joint flow matching, the paper's T5, full 14×14 grids, Fast-WAM's data handling, ~1 TB feature cache).
- Rebuilt the proxy on Fast-WAM's own code (`src/proxy_wam.py`, `extract_fastwam.py`, `t5_cache.py`, `fastwam_data.py`; loss identical to Fast-WAM's), real-data smoke on one episode; added the paired per-seed analysis (`src/lp_eval.py`) after reviewer feedback reframed the round as a proxy replication.
- Wrote the full plan as a Google Doc for the reviewer; set up the autonomous run (RUN.md, stage 2 manifests, Prometheus watcher) with the user's pre-approval. Nothing submitted yet.
- See: world-model-stacking-encoders/experiments/2026-10-27-proxy-calibration/{PLAN.md,RUN.md}, writing/proxy-replication-plan-2026-10-28.html, LOG.md 2026-10-28 (b)–(f)

## 2026-10-28 — world-model-stacking-encoders: stand-in audit, reviewer rounds, decision rules locked

- Audited stand-ins: the DINOv3 stand-in only touched a smoke test (rerun with real DINOv3); the real caveat is that the earlier diagnostics used 128 px yifengzhu-hf data, not the paper's Fast-WAM data. Added "don't bury me in numbers" to the global and workspace CLAUDE.md.
- Folded five rounds of outside review into the proxy plan: proxy replication framing, action-expert claim corrected, ~1 TB volume, orientation checks, and a locked decision rule (paired per-seed; calibration D−none ≥ +2.7 all positive, D+L−D ≥ −1.35; D+R all negative and mean ≤ −2.7 as a trigger), eval seeds 41–43, proportional subset, paired D/D+R pilot for sizing (sd ×1.5, SE ≤ 1.35, 48 h / 8 GPU cap).
- RUN.md aligned (paired pilot, step 7a analysis-code checklist, round pre-approved); watcher fixed so a dropped laptop connection no longer looks like a vanished Job. Google Doc kept in sync. Next: user starts the autonomous run from RUN.md.
- See: world-model-stacking-encoders/experiments/2026-10-27-proxy-calibration/{PLAN.md,RUN.md}, decisions/003-proxy-policy-calibration.md, LOG.md 2026-10-28 (b)–(k)

## 2026-10-28 — world-model-stacking-encoders: NRP policy checks of the autonomous proxy run

- Rechecked the live NRP policy pages against the proxy run's manifests and runbook; server dry run of all stage 2 manifests passed.
- Fixed: download Job memory under NRP's 2 GB exempt size; watcher now also stops on CPU/RAM out of range (exit 6); runbook got a compliance checklist for manifests written mid-run and a ≥ 40% GPU requirement for rollout Jobs (stop 10).
- Open, awaiting the user: five usage risks for stages 3–4 (unkillable idle Job after kubectl expiry, CephFS contention with 8 trainers, simulator install location, one rollout file per pod, cleanup + NRP acknowledgment).
- See: world-model-stacking-encoders/experiments/2026-10-27-proxy-calibration/{RUN.md,watch_job.sh}, LOG.md 2026-10-28 (g), (i), (l)

## 2026-10-31 — Proxy run: feature extraction nearly done

- Extraction for DINOv3, C-RADIOv4 and SigLIP 2 finished and passed the file checks; the VAE is done except 20 episodes.
- User OK'd an extractor change (decode the next episode on a thread; VAE remainder into a fresh `vae_b` folder). Outputs were byte-identical to the old code; speed went up 2-3×; memory had to be raised after OOMs on long episodes.
- The last VAE shard stalled on CephFS writes on k8s-4090-01, as on the Fullerton nodes before, so the stalls follow the node, not the folder. VAE Job stopped (stop 2); waiting for the user to OK a resubmit that excludes those nodes, then the pilot.
- See: world-model-stacking-encoders/LOG.md 2026-10-30 (a)-(c), 2026-10-31 (a); experiments/2026-10-27-proxy-calibration/RUN.md

## 2026-11-01 — Proxy run: VAE resubmit stalls again; pilot sample drawn

- Resubmitted the last VAE shard as a single pod excluding the three stalling nodes (user OK). It queued ~3 h, was rejected twice by a fourth node (ry-gpu-08), then ran there and stalled on CephFS writes the same way. Stopped (stops 1/2). 11 libero_10 episodes left.
- So the stalls are not about specific nodes; the cause is unknown. Proposed fix, awaiting the user: write each episode to local disk, then copy it to CephFS in one go.
- Drew the 100-episode pilot LIBERO-Plus sample (size chosen by the agent, logged). Deleted the finished SigLIP 2 Job at the user's request.
- See: world-model-stacking-encoders/LOG.md 2026-10-31 (b), 2026-11-01 (a); experiments/2026-10-27-proxy-calibration/jobs/rendered/job-extract-vae-s3.yaml

## 2026-11-01 — stacking-encoders: VAE cache finished, pilot found disk-bound, loader rewritten

- Fixed VAE write stalls by staging on local disk; last VAE shard finished and all feature caches passed checks.
- First training pilot learned fine but its GPU sat idle about 2 h waiting on CephFS reads; second pilot cancelled.
- Read test showed parallel reads are about ten times faster; loader rewritten to read in parallel, same batches as before.
- Next: user OK to rerun both pilots with bigger memory and a GPU check 15 min in.
- See: world-model-stacking-encoders/LOG.md (2026-11-01 (b)-(g)), experiments/2026-10-27-proxy-calibration/RUN.md

## 2026-11-02 — stacking-encoders: slide prep for the reworked proposal

- Walked the user through the project story slide by slide for their new proposal deck (the user builds the slides): LIBERO groups, the paper's puzzle, the three cheap measures and their results table, why they missed the bad pair, the rerun (experiment 3) and its pre-registered rule.
- Fixed slide content: LIBERO-Long (not LIBERO-100) is the fourth group; LIBERO-Plus is the same tasks under disturbances, not a superset; harm was only measured on LIBERO-Plus after training (no clean-LIBERO number for the pair).
- A /btw side thread added a README key-findings note that the two answer keys differ in kind (df08d35). Pilot training runs finished and the pilot rollout was submitted by the run session.
- See: world-model-stacking-encoders/experiments/2026-10-19-feasibility-tally/README.md, experiments/2026-10-21-target-diagnostics/README.md, experiments/2026-10-27-proxy-calibration/PLAN.md

## 2026-11-02/03 — stacking-encoders: both pilots trained; pilot rollout stopped for low GPU use

- Reran both training pilots with the parallel loader. An L4 was too small for DINOv3 + C-RADIOv4, so the GPU list now has only full-24 GB / 48 GB cards. Both trained, about 12-18x faster than before, and a full run is now hours, not days.
- Queues were the main cost: 1.7-6.5 h per pod. My free-capacity script overstates what's free.
- Pilot simulator test: pre-check of all inputs passed. One pod ran with no errors, but the GPU was busy only ~25% (small batches, plus setup idle), so I deleted the Job.
- Next: user to choose. Proposed profiling a policy call locally and moving setup to a CPU Job, then rerunning the pilot rollout.
- See: world-model-stacking-encoders/LOG.md (2026-11-01 (h) to 2026-11-03 (b)), experiments/2026-10-27-proxy-calibration/jobs/rendered/

## 2026-11-03/04 — stacking-encoders: rollout GPU fix works; pilot rollout blocked by a setup bug

- Profiling on the cluster showed the policy wasted GPU time on tiny kernel launches and CPU noise prep. Fixed (cached noise, thread cap, CUDA graph replay); actions are identical to the old path and the GPU stays busy in a policy-only test.
- Pilot rollout 2 sat ~12 h in the queue (GPU list widened to A10/L4/A40, deadline raised to 36 h, both with user OK). When pods finally started, they failed in setup: the prebuilt env's offline package install breaks on a fresh machine. My earlier restore test missed it by running on an already-installed machine.
- Fix committed; an env rebuild Job that tests the restore on a truly fresh container is ready and waiting for the user's OK. User decided experiment READMEs get written at the end, consistently.
- See: world-model-stacking-encoders/LOG.md (2026-11-03 (c) to 2026-11-04 (b)), experiments/2026-10-27-proxy-calibration/jobs/rendered/job-env-build2.yaml

## 2026-11-03/04 — stacking-encoders: proposal slides finished; plan/archive doc layout

- Helped the user finish their reworked proposal deck in Vanshika's six-section format (they wrote it; I checked content against the project files, explained LIBERO demos vs rollouts and per-frame vs per-episode, and gave paste-ready related-works and citation lines).
- Recorded the written presentation feedback; most "design flaw" points were already handled (held-out demo split, conditional gain). User chose two new plan items: a seed-noise check before the full round and a cheap test on perturbed frames.
- User decided on a living root PLAN.md (plans only) with PLAN.md in every experiment folder, and LOG/READMEs as the archive. Set up in stacking-encoders; shared _standards/_template not updated yet (handoff prompt given for a docs agent).
- See: world-model-stacking-encoders/PLAN.md, world-model-stacking-encoders/LOG.md (2026-11-03 (k), (l))

## 2026-11-05 — stacking-encoders: lead session; pilot rollout 2 checked, GPU dip diagnosed, result read blocked on login

- Ran as the lead: workers did a status check and a read-only GPU-dip diagnosis, a separate review agent checked each, and both were corrected and merged. Both pilot runs finished cleanly; the second one's GPU sat mostly idle for ~10 min, which breaks the runbook's 40% rule.
- The dip looks like the workload, not the node: both pods slowed at the same points in their to-do list, where CPU-heavy sensor-noise scenes are bunched together (speculative until per-episode timings are read).
- User OK'd a tiny read-only Job to fetch the pilot result files; it wasn't submitted because kubectl's login refresh fails inside the worker's sandbox. The GPU fix waits on the timings.
- See: world-model-stacking-encoders/LOG.md (2026-11-05 [agent, lead]), experiments/2026-10-27-proxy-calibration/LOG.md (2026-11-05 (b), (c)), branch worktree-agent-a1345b33d3e874b25

## 2026-11-06 — root docs-audit lead: audit plan drafted (not dispatched); H100 reproduction estimate

- Read `_standards/` and every direction as the lead archiving session and drafted a three-worker docs audit: (1) commit the template files and check standards/template/root README agree, (2) bring embedding-choice and planning-horizon up to the new template (PLAN.md, README shape), (3) stacking-encoders parts the live run session isn't editing (finished experiments, a new `data/README.md`, STANDARDS tool fields, decisions and literature formats). Nothing dispatched or changed.
- Waiting on the user: how to handle the live stacking-encoders session (recommended: a handoff for its README/PLAN/LOG/proxy folder), keep vis-aria-depth out (separate app repo), whether the docs session that wrote the untracked `_template/PLAN.md` and `_template/experiments/_new-experiment/` is closed, OK to remove the old agent worktree, and whether the README feedback blocks stay. Found along the way: the standards rewrite landed inside an unrelated commit (8ec6494); stacking `references.bib` has only 2 entries.
- A worker estimated reproducing every GPU-runnable finding of liu2026what on one H100: years (about 3–16; core Tables 1 + 10 about 7 months to 3 years), all speculative. The paper gives no compute or step counts, and one 80 GB card is too tight for its recipe. The detail exists only in this session's scratchpad and in memory, and isn't in the stacking-encoders LOG yet.
- See: `_standards/04-documentation.md`, `_template/`, `world-model-stacking-encoders/README.md`

## 2026-11-06 — stacking-encoders lead: pilot read, full benchmark chosen, seed-noise training wave started

- Pulled the pilot results off the cluster. The short pilot models almost never succeed, so they couldn't size the experiment. The user chose to evaluate every model on the paper's full LIBERO-Plus benchmark.
- The GPU dip came from a few slow camera-noise episodes running back to back. Rollouts now shuffle their job order, with a guard against restarting on mismatched files. Merged and tested.
- Seed-noise check started: DINOv3 seed 1 trained (about 4 h on a 3090). Seed 0 was stuck in the queue, so it was resubmitted with the A40 allowed. The comparison rule is locked, anchored to the paper's numbers. The time limit can stretch.
- Seed 1's full-benchmark rollout is ready to go once the user downloads one LIBERO-Plus file. The permission system blocks agent downloads, so outside code and files now go through a user `!` command.
- See: world-model-stacking-encoders/LOG.md (2026-11-06 [agent, lead]), world-model-stacking-encoders/experiments/2026-10-27-proxy-calibration/LOG.md (2026-11-05 (d), 2026-11-06 (a)–(c)), experiments/2026-10-27-proxy-calibration/PLAN.md (Amendment 2026-11-06, Seed-noise rule)

## 2026-11-06: root docs-cleanup lead (done, merged to local main, not pushed)

- Ran the docs cleanup the earlier lead had planned. Workers did the edits; a fresh reviewer fact-checked each branch against its sources before merge. The checks caught real errors (wrong access dates, cache sizes marked "not recorded" that were in the LOG, a strike count dated as rechecked when it wasn't), all fixed before merging.
- User decisions: removed `world-model-embedding-choice/` and `world-model-planning-horizon/` (not interested; recoverable at 252524e); stacking `references.bib` cut to LIBERO, LIBERO-Plus and liu2026what (Agostinelli dropped from the bib only); formatter edits to two `src/` files discarded (layout only, code identical); three old stacking worktrees removed.
- Root: template PLAN.md and experiment folder committed; `.claude/worktrees/` and `vis-aria-depth/` ignored; CLAUDE.md, README and templates agree with `_standards/`; `_standards/08` brought in line with the stacking LOG, point-in-time facts dated, rules unchanged.
- Stacking: new `data/README.md` (provenance), STANDARDS tool fields, finished-experiment READMEs, README trimmed to `_standards/04` shape, 2026-11-06 mentor meeting logged and its three points added to PLAN Next.
- Open: the 1-of-3 idle-strike count dates from 2026-10-16 and the 2026-11-01 pilot may have added one (nobody has read the Violations page); no A40/A6000 pod has ever scheduled through `gpu.product`, though the current training list includes them; template lacks a `data/README.md` and owner/misuse slots; VAE cache size stays an estimate (user).
- See: world-model-stacking-encoders/LOG.md (the three 2026-11-06 [agent, docs] entries), `_standards/08-compute-nautilus.md`, commits a8483c9..55588db

## 2026-11-06: root docs-cleanup lead, follow-up (A40/A6000 mismatch confirmed in 08)

- Explained why A40/A6000 never schedule from our Jobs: those nodes offer the cards only under their own resource names, not the generic `nvidia.com/gpu` that our Jobs request. The current training and eval Jobs still listed them (harmless, adds no capacity).
- At the user's request, relayed this to the stacking implementation session. It ran a read-only Prometheus check (confirmed: generic GPU count 0 or absent on every A40/A6000 node), logged it as experiment LOG 2026-11-06 (f) with PLAN Amendment 2026-11-06 (b), and, with the user's OK, submitted an A40-named twin of the seed-0 training Job.
- A worker updated `_standards/08` to state the mismatch as verified, pointing at that LOG entry; whether a by-name request schedules is still untested. User decisions: assume 1 of 3 idle strikes; the LIBERO bib entries are fine as arXiv versions.
- Next: once the A40 twin runs (or clearly can't), update the "untested" line in `_standards/08`.
- See: `_standards/08-compute-nautilus.md` (special resources bullet), world-model-stacking-encoders/experiments/2026-10-27-proxy-calibration/LOG.md (2026-11-06 (f))

## 2026-11-06 — stacking-encoders implementation lead: seed-1 full-benchmark eval submitted; seed 0 training on an A40

- Resumed after a /clear. The user downloaded the LIBERO-Plus file. A worker built the full benchmark episode list, updated the cluster code in place, and submitted seed 1's evaluation as 8 pieces, starting with 2. A review found nothing serious. Memory was raised before any pod started (user's call).
- Found that the A40 we'd added for training never counted: A40/A6000 cards go out under their own resource name. With the user's OK we raced an A40-named copy of the seed-0 training job. It got a machine in about 25 min, and the original was deleted cleanly before it started.
- At close: seed 0 is training with its GPU busy, due about 2 h later. Seed 1's eval pods are still queued. The watchers end with the session, so next session: check pods, then do the GPU/memory/error check and scale up to 7 pods.
- See: world-model-stacking-encoders/experiments/2026-10-27-proxy-calibration/LOG.md (2026-11-06 (d)–(g)), PLAN.md "Amendment 2026-11-06 (b)"

## 2026-11-06: stacking-encoders planning lead: fusion idea planned, paper read for the mentor, perturbed-frame design drafted

- The user asked whether the paper's way of combining encoder features could explain why DINOv3 + C-RADIOv4 hurts. Recorded as a conditional next experiment: test three fusion suspects (scale, double counting as targets, grid alignment) only if the proxy finishes, passes calibration, and shows no drop.
- Read the paper for the mentor's points (workers, fact-checked against the PDF text): features are stacked along channels at matching grid locations; the paper uses no VLA (its model is a world action model); only channel concatenation was ever tried on pairs; encoders use the final layer only.
- Drafted the perturbed-frame test design (camera, light, background, noise; probe trained on clean, tested on perturbed; rule P decides, B reported; matched-size control). The LIBERO-Plus source was read (user cloned it). Most choices decided; it waits until the proxy is trained, and sign-off waits on a demo-file check and a few small choices.
- See: world-model-stacking-encoders/PLAN.md (Next 2-4), world-model-stacking-encoders/experiments/2026-11-06-perturbed-frames/PLAN.md, world-model-stacking-encoders/writing/hypotheses-draft.md ("Update 2026-11-06"), world-model-stacking-encoders/literature/notes/liu2026-what-visual-representations-wams.md ("Read for the mentor's 2026-11-06 points")

## 2026-11-07: stacking-encoders planning lead: NRP 40% GPU rule rechecked twice, docs corrected, L40 eval chunk stopped

- Status check for the user: the seed-noise evaluation (two DINOv3 seeds on the full LIBERO-Plus benchmark) was a bit under half done; the open issue was eval pods near or under 40% GPU.
- The user asked for NRP's real rule. The first worker and reviewer covered the public docs and nrp-site: 40% flags a pod, the limit is 4 pods outside the ranges at once, no kill documented (merged 266450e). The implementation session then found NRP's backend code (prp/k8s_portal): a warning email after about 3 h under 40%, an admin delete list, admin deletes recorded as strikes, and a three-strikes policy. A second worker plus two review rounds added that layer (merged eeb6a84).
- Decisions: "Soften, cap at 4" was given on the incomplete first picture and superseded. The user got NRP's warning email for the L40 pod and said "yep kill it"; the implementation session stopped it and is resubmitting the 7 unfinished chunks L4-only (user's pick there).
- See: _standards/08-compute-nautilus.md ([NRP code] section), world-model-stacking-encoders/LOG.md (2026-11-07 planning entries), world-model-stacking-encoders/STANDARDS.md (40% bullet)

## 2026-11-07: stacking-encoders slides lead: mentor deck (11/11/26 update) reviewed, six slide figures made and merged

- Walked the user through their own "Research Proposal" deck and its new 11/11/26 section (recap, dress-rehearsal trial runs, seed-noise check, perturbed-frame and combining-step follow-ups, compute bottlenecks, moving forward), checking every claim against the repo and giving plain, paste-ready lines. The user wrote the slides.
- Six figures (training curve, compute cost, GPU use during testing, noise-check ruler, combining grids, encoder layers) were made by workers, each checked by a separate reviewer; the reviews caught a misleading 5-minute GPU sample and swapped paper scores from the lead's own brief. Merged to main; copies on the user's Desktop.
- Decision (user): slide figures always get a copy in the repo (`writing/figures/`, scripts in `src/`), Desktop copy only for convenience.
- Found: the proxy PLAN calls D+L the paper's best DINOv3 pair, but D+U is higher in Table 10; D+L is best of the four we tested (noted in LOG, locked PLAN not edited).
- See: world-model-stacking-encoders/LOG.md (2026-11-07 [agent, slides]), world-model-stacking-encoders/writing/figures/

## 2026-11-07: stacking-encoders planning lead: perturbed-frame design finished, signed off and handed off

- Picked up after a /clear. Got status from the implementation session; evals of the two DINOv3 seed-noise models about half done, rest on L4s.
- Perturbed-frame test design: applied the second review's fixes plus one more review round; the user decided the last four open points (extra fixture check, rule B reported both ways, resize method picked by a smoke-test comparison, look up averaged-feature sizes first) and signed off. Handed to the implementation session, which got the user's OK for phase 1 (smoke test).
- README/PLAN status brought up to date twice (eval state, seed-noise script now exists).
- See: world-model-stacking-encoders/experiments/2026-11-06-perturbed-frames/PLAN.md, world-model-stacking-encoders/LOG.md (2026-11-07 planning entries)

## 2026-11-08: stacking-encoders planning lead: motion-alignment idea, prior-art search

- Held while the implementation session runs the evals, then took up the user's idea (passed over by the literature session) of trying Guo et al.'s label-free encoder scores on our encoders.
- Decision (user): pursue the robot version, a score for whether an encoder's features change where the scene moves, and check prior work first. A worker searched; a separate reviewer checked it against the papers in two rounds and overturned the first verdict: a simpler "how much do features change" score already exists and tracks robot success; the "where" version was not found.
- Decision (user): the prior-art file lives in `writing/` (the reading-list folder belongs to the literature session). Added as a candidate (PLAN Next 4); the user decides when to design it.
- See: world-model-stacking-encoders/writing/2026-11-08-motion-score-prior-art.md, world-model-stacking-encoders/LOG.md (2026-11-08 planning entries)

## 2026-11-09: stacking-encoders implementation lead: perturbed-frames check failed, diagnosed, plan amended; evals at 12 of 16

- Proxy evals: 12 of 16 DINOv3 shards done, all with 0 errors. Three seed-0 shards are running and seed 1 shard 2 is waiting for an L4. Logs are saved and the LOG is current through (o).
- Perturbed frames: the validation check failed. Our MuJoCo 3.3.2 re-renders lack the fine grain of the original LIBERO images, and the stored-trained probe can't tolerate the difference in any of the ten tasks (Mac diagnosis, two reviews).
- Decision (user): retrain the probes on our own re-renders. The plan amendment had two review rounds, the user decided 8 points (0.025 sanity margin, two-step run, scene-file check and others), and it's merged. The cluster go-ahead comes later.
- See: world-model-stacking-encoders/experiments/2026-11-06-perturbed-frames/PLAN.md (Amendment 2026-11-09), .../results/pf-val-diagnosis/README.md, world-model-stacking-encoders/experiments/2026-10-27-proxy-calibration/LOG.md (n)-(o)

## 2026-11-09: stacking-encoders planning: perturbed-frames test explained; re-render redesign checked against the literature

- Walked the user through the perturbed-frames test and why the probes now train on our own re-renders: the original demo pictures came from an unrecorded simulator setup, and mixing them with ours would blur the perturbation effect.
- A literature subagent found the redesign matches common practice (LIBERO demos are routinely redrawn, MuJoCo version is a known source of shift, robustness benchmarks keep one pipeline). Caveats for a write-up: our set-state route is less common than action replay, and the probe's sensitivity can inflate effects. Not independently reviewed.
- No decisions; nothing run.
- See: world-model-stacking-encoders/LOG.md (2026-11-09 planning entry), world-model-stacking-encoders/experiments/2026-11-06-perturbed-frames/PLAN.md

## 2026-11-09: stacking-encoders implementation: perturbed-frames redesign built, phase A run, gate passed

- Built the re-render redesign with five reviewed workers (render code, retrained-probe analysis, render-consistency check between phases, tolerance rule, phase A Job files); two Codex adversarial reviews shaped it.
- Ran phase A on the cluster: our own re-render of the training and validation frames, all five encoders extracted, a perturbation timing pilot, and the gate check. Everything exited cleanly, about 10 GPU-hours.
- Gate passed: a probe trained on our frames predicts actions slightly better than the original did on the original frames, so the re-render is a faithful stand-in. Phase B (test frames) waits for the user's go.
- Proxy evals: all 16 shards done with zero errors; the seed-noise read waits for the user's go.
- Decisions (user): render-consistency check added; deciding slice is full-size, global, both cameras; zero gain reported with rules unchanged; one-step pixel noise tolerated at one in a hundred thousand.
- See: world-model-stacking-encoders/experiments/2026-11-06-perturbed-frames/LOG.md (2026-11-09 lead entries), .../PLAN.md (Amendments 2026-11-09 (b)-(d)), .../jobs/PHASE-A.md, world-model-stacking-encoders/experiments/2026-10-27-proxy-calibration/LOG.md (p)-(r)

## 2026-11-09: stacking-encoders planning: pull-request workflow adopted, GitHub caught up, seed-noise verdict relayed

- The user asked whether agents should open PRs reviewed by other agents instead of merging straight to main; chose PRs for anything that feeds a research conclusion, direct merges for log and status chores. A worker wrote the rule into the standards and it went through the new process itself as PR #1 (two review rounds posted as PR comments, merged 7bfb568). Lead prompt and worker definition updated to match; main now blocks force-pushes and deletions.
- The first review round found GitHub main was three days and about 200 commits behind the Mac; scanned for secrets (none found) and pushed with the user's OK.
- Seed-noise verdict read from main: the locked rule says "go ahead", but both DINOv3 models succeed around a tenth as often as the paper's, so the pre-committed scale-reading exit applies. The user chose to amend first (train the DINOv3 models longer or larger, rerun the noise check, thresholds unchanged) before the other 13 runs; handed to the implementation session as direction, which drafts the amendment as a PR.
- The user lifted the two-worker cap for leads (high usage plan); all research leads told to re-read the lead prompt. Open: condense the rule text (follow-up PR), motion-score deferral unconfirmed, mention the case-study framing to the mentor.
- See: _standards/05-ai-assisted-work.md (Git section), world-model-stacking-encoders/LOG.md (2026-11-09 [agent, lead, planning] PR workflow), world-model-stacking-encoders/writing/2026-11-09-proxy-exit-options.md, https://github.com/michaelnkr808/research26-27/pull/1

## 2026-11-10: stacking-encoders implementation: phase B render resubmitted on a wider GPU set, queue of waiting experiments set

- The phase B render Job (perturbed frames, first read of test frames) never started: its L4-only rule matched just two nodes in the cluster and both stayed full for eight hours. With the user's OK it was deleted and resubmitted accepting five kinds of 24 GB card (PR #10, reviewed, merged, pushed); fresh 36 h deadline, pod still queued at close.
- The code bundle on the cluster keeps phase B's reviewed copy of the shared diagnostics file even though a later control-partners PR changed it on main; recorded as a provenance note in the experiment LOG.
- Planning handoff received: a CPU-only control-partners run (PR #9) joins the queue after phase B's GPU work. Order now: render, then SAM 3 / VGGT / final analysis, then control partners, then the 16-layer DINOv3 proxy retrain. Each cluster step waits for the user's go here; nothing proxy-side submitted.
- Lesson kept in memory: read the FailedScheduling event's "didn't match affinity" count early; a two-node pool is a rule problem, not a queue.
- See: world-model-stacking-encoders/experiments/2026-11-06-perturbed-frames/LOG.md (2026-11-10 entry), .../jobs/PHASE-B.md (note 2026-11-10), world-model-stacking-encoders/LOG.md (2026-11-10 implementation lead), world-model-stacking-encoders/experiments/2026-11-09-control-partners/PLAN.md

## 2026-11-09: stacking-encoders slides lead: deck checked, weekend changes summarized, 11/11 slide lines drafted

- Read the user's latest deck export and checked the 11/11 section against the repo; flagged a stale status line, the "best partner" wording (best of our four, not the paper's best), a few typos and the "same way the paper did it" line.
- Summarized what changed since the last deck edit: the case-study reframe (not yet told to the mentor), the perturbed-frames redesign and passed gate, the papers the user read, the review, and the seed-noise verdict (go ahead by the rule, but the proxy scores about a tenth of the paper, hence the amendment).
- Wrote paste-ready lines for a new Updated Question slide, a Noise Check result slide, the Disturbed Scenes rewrite, Related Works and Moving Forward. Two figures offered via the new PR route; the user had not picked yet.
- See: world-model-stacking-encoders/LOG.md (2026-11-09 [agent, slides]), world-model-stacking-encoders/decisions/004-paper-framing-case-study.md, world-model-stacking-encoders/experiments/2026-10-27-proxy-calibration/LOG.md (t)

## 2026-11-09: stacking-encoders literature: novelty check for planning, PR-rule replan

- Answered the planning session's three-question novelty check (search agent, independent reviewer, lead spot-check of the reviewer's new claims). No robot or WAM paper scores encoder pairs or a conditional gain before training. No multi-seed replication of composition results was found. Nothing ranks the paper's encoders under perturbation. Suggested narrowing: a conditional score for robot or WAM stacking.
- The review caught a withdrawn paper and several overreaches before the reply went out. Planning recorded the answer in writing/2026-11-09-plan-validity-review.md section 3.
- Replanned the unmerged Hewitt bookkeeping for the new PR rule. It and two new leads (Yardi 2025, Chun 2026) wait on the user's OK in this window, plus a question about pushing main after direct merges.
- See: world-model-stacking-encoders/LOG.md (2026-11-09 [agent, literature]), world-model-stacking-encoders/writing/2026-11-09-plan-validity-review.md
