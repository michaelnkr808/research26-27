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

## 2026-11-09: stacking-encoders planning: PR rule condensed, three methods memos merged, control-partners experiment designed

- Worked through the five loose ends from the plan-validity review. The pull-request rule in the standards was condensed to one bullet per topic (PR #6) and the stale "not yet merged" note in the compute standard fixed. The user confirmed deferring the motion-score design until the proxy result and chose to keep the layer study written down in PLAN for later.
- Three reviewer methods points became merged memos with a paper trail: the never-run standard baselines computed and compared against the paper's ordering (PR #7, blind baselines), a seed-power memo showing what drop sizes three or five seeds can detect under the measured seed noise (PR #8), and a control-partners experiment design with synthetic partners for both cheap tests (PR #9: noise, duplicate, random projection, shuffle). The user accepted the proposed margins, one seed first, global features only; the CPU Job is written but not submitted and waits in the implementation session's queue after phase B.
- Each PR took one read-only reviewer round posted as a PR comment, fixes by the same worker, then the user's "go for it". Three workers lost time on the Mac's Python setup, so the working interpreters are now in memory.
- Diagnosed the laptop fan: eight orphaned busy-loop shells left by a worker in the Arbor session; the user kills them with a `!` line since the classifier blocks cross-session kills.
- See: _standards/05-ai-assisted-work.md (Git section), world-model-stacking-encoders/writing/2026-11-09-blind-baselines.md, world-model-stacking-encoders/writing/2026-11-09-seed-power.md, world-model-stacking-encoders/experiments/2026-11-09-control-partners/PLAN.md, world-model-stacking-encoders/LOG.md (2026-11-09 [agent, lead, planning], PR rule condensed), https://github.com/michaelnkr808/research26-27/pull/6

## 2026-11-10: stacking-encoders literature lead: orientation only, four decisions put to the user

- Started a fresh literature-lead session after /clear (renamed vis-literature-agent). Read the standards, PLAN, and the literature LOG entries; no papers read, searched, or edited.
- Laid out the four items waiting on the user from the 2026-11-09 literature session: (1) Hewitt et al. 2021 bib entry and read marks, still on unmerged branch `worktree-agent-aec5f053ca0c12e3a` (566bf8b, worktree under .claude/worktrees/), to be redone as a PR; (2) Yardi 2025 and Chun 2026 as unread leads; (3) whether small direct-merge chores may be pushed to main right away; (4) Kornblith 2019 (CKA) to the top of the reading queue, bib entry once read.
- No decisions made; the user closed before answering.
- Open loops: the four questions above are unanswered; suggested next reads are Dong et al. 2026 then Wang et al. 2019. The Hewitt worktree is still present.
- See: world-model-stacking-encoders/LOG.md (2026-11-09 [agent, literature] novelty check), world-model-stacking-encoders/literature/reading-list/README.md

## 2026-11-10: stacking-encoders slides: session restarted, status refreshed for the 11/11 deck

- Restarted the slides-lead role after a /clear: read the standards, direction README, PLAN, recent LOG and the two earlier slides entries.
- Told the user what moved since the Nov 7 export (-10, still the newest): perturbed-frames phase B is resubmitted but has no output yet; the proxy retrain is set to a 16-layer DINOv3 and not submitted; the mentor has not heard about decision 004.
- No slide lines, figures, workers or cluster work this session.
- Open loops: the user picks what to do next for the 11/11 meeting (refreshed Noise Check and Disturbed Scenes lines, the two optional figures, or a check of a new deck export).
- See: world-model-stacking-encoders/LOG.md (2026-11-09 [agent, slides] and 2026-11-10 [agent, slides])

## 2026-11-10: stacking-encoders planning: path-to-paper memo, author email, feature-overlap design signed off

- Wrote and merged a path-to-paper memo (PR #11): three possible endings, milestones, and a speculative calendar (full result set around early December at the measured two-GPU pace). The user said the mentor's target is a CVPR workshop paper and that a measure of whether an encoder pair will be bad is the contribution the user wants.
- Un-parked two items at the user's request: a retrospective check against all of the paper's Table 10 compositions (needs three encoders we have not extracted, checkpoints unknown) and an email to the authors, which the user sent to two corresponding authors with the first authors copied.
- Designed a feature-overlap score on the four pairs already extracted (PR #14, merged after three review rounds). Decision: the user fixed the headline score as "rebuild the partner from DINOv3" and signed off; the lead had recommended the other direction. Handed to the implementation session as direction only.
- Wrote a dated note on decision 004 (PR #16). The reviewer caught two overstatements, now fixed; the note leaves open whether the pair-harm measure becomes the paper's headline.
- Open loops: PR #16 is open and waits on the user's "merge" (then pull, remove worktree agent-a63dd9603a5fb2949). This session's closing chore merge is on local main, not pushed. Ask the mentor on 2026-11-11 whether the pair-harm measure is in this paper and whether it is the headline. The overlap code and CPU Job need the user's go in the implementation window. The full Table 10 design waits a few days for the authors' reply.
- See: world-model-stacking-encoders/writing/2026-11-09-path-to-paper.md, world-model-stacking-encoders/experiments/2026-11-10-feature-overlap/PLAN.md, world-model-stacking-encoders/PLAN.md (Next 5, Next 6), world-model-stacking-encoders/LOG.md (2026-11-10 [agent, lead, planning] entries), https://github.com/michaelnkr808/research26-27/pull/16

## 2026-11-10: stacking-encoders review lead: cleanup after the four-audit review

- Short session for the review lead (renamed vis-review-agent). No new review task was assigned; the five review reports and the synthesis from 2026-11-08/09 are already on main (9cdc6ce..a7f10c2) and their LOG entry is written.
- Stopped two finished subagents (a fact-check reviewer and the perturbed-frames/docs audit worker) that were still listed in the session after reporting back; their worktrees had been removed earlier.
- Decisions relayed earlier and still owned by other sessions: seed-noise threshold wording before the proxy verdict (implementation), same-seed init mismatch in proxy_wam.py, Key findings headline rewording, PLAN Next 3 reframing around LiLa-WAM/ReWAM, backoffLimit exception.
- Open loops: no review in flight. Candidates offered to the user: review main's rewritten pf_analysis.py before the final perturbed-frames run, and recheck the synthesis fix list against current main (its status column is as of 43c3b7d). Any new report lands by PR per _standards/05.
- See: world-model-stacking-encoders/reviews/2026-11-09-synthesis.md, world-model-stacking-encoders/reviews/README.md, world-model-stacking-encoders/LOG.md (2026-11-09 [agent, review])

## 2026-11-10: stacking-encoders draft lead: writing plan, LaTeX skeleton, page-length question

- Started the paper's draft session (vis-draft-agent). Wrote the writing plan (outline with a slot per ending, figure list, citation list, venue facts checked on the CVPR site) and the LaTeX skeleton in the CVPR 2027 template under writing/paper/. Both went through two read-only review rounds and were merged on the user's OK (PRs #12, #13).
- Decisions: the user chose mix by section (the draft session writes setup, methods, limitations and supplement; the user writes abstract, introduction, related work and discussion from annotated outlines) and LaTeX in the repo, changed by PR.
- Answered whether figures would fill 7 to 10 pages: no, the limit is eight pages plus references and the planned sections already fill most of it. Corrected at close a claim that the pair-harm section adds a page (the plan takes it from sections 4 and 5).
- Open loops: draft sec/3_setup.tex (new branch, fact-check reviewer, PR). After the 2026-11-11 mentor meeting: framing (decision 004), whether PLAN Next 5 is in this paper, and length (four pages or fewer vs about eight). LIBERO/LIBERO-Plus read status still unresolved. This close merge is on local main, not pushed. The review session's own timeline entry was left uncommitted in the working tree for that session to commit.
- See: world-model-stacking-encoders/writing/2026-11-10-writing-plan.md, world-model-stacking-encoders/writing/paper/README.md, world-model-stacking-encoders/LOG.md (2026-11-10 [agent, lead, draft] entries)

## 2026-11-10: stacking-encoders implementation: control-partners results and stage 2, phase B render on an A6000 then a GPU fault, SAM 3 started

- Control-partners stage 1 finished; results with gates and the decision rule merged (PR #15), plus the paired comparison from fetched draws (PR #19, reviewer found a stronger pairing test). Input test can show harm; target test labelled partner-specific with a shuffled-control caveat. Stage 2 (seeds 1 and 2, PR #17) submitted and running, about 11 h.
- The phase B render sat Pending 12 h: manifest valid, but every usable node with its card types was full (free GPUs on reserved nodes). An RTX A6000 twin (PR #18, race note hardened after review) won at once in us-central; the render consistency check passed on the A6000 and four of five conditions published. Then the card failed mid-extraction and the script fell back to CPU; the lead saved the log and deleted the Job. Four SAM 3 Jobs submitted on the user's go (past the 20-Job gate); clean runs on an L4 at full use.
- A background duplicate lead session (`implementation-agent`) pushed overlapping logs; it stood down, the user confirmed this session as implementation lead, and a follow-up entry corrected its records.
- Open loops: the user's answer on phase B recovery (noise render retry, a new extraction Job for DINOv3, SigLIP 2, C-RADIOv4 on the four published renders) and on keeping the four renders; check the three queued SAM 3 Jobs and stage 2 (no session is watching them); the stray `.h5` files under `phase-b/render/`; then SAM 3 noise, VGGT-Omega, the final Job; proxy retrain and feature-overlap wait for the user.
- See: world-model-stacking-encoders/experiments/2026-11-06-perturbed-frames/LOG.md (2026-11-10 entries), world-model-stacking-encoders/experiments/2026-11-06-perturbed-frames/jobs/PHASE-B.md (sections 7, 10), world-model-stacking-encoders/experiments/2026-11-09-control-partners/README.md

## 2026-11-10: stacking-encoders implementation lead: phase B recovery Job, overlap code, restart after /clear

- Restarted after /clear and read the state back from the cluster: the A6000 had died mid-extraction (pre-reset), and two SAM 3 Jobs were lost to a node failure; resubmitted those two on the user's go.
- Two workers, both by reviewed PR: the feature-overlap code and tests (PR #21, merged) and one combined recovery Job that extracts the missing DINOv3, SigLIP 2 and C-RADIOv4 features from the four kept renders and then renders and extracts noise (PR #22, three review rounds, merged, submitted 21:48 UTC, Pending).
- Decisions: one combined Job instead of a noise-only retry (idle GPU); order went noise first, then back to extractions first after a reviewer showed noise first leaves the first 3 h about a quarter busy; the user wants no local runs on real data, so the overlap timing check became a small cluster sizing Job (PR #23, open).
- Open loops: PR #23 merge OK; recovery Job and SAM 3 background Pending (GPU check about 45 min after start; typed 3090 twin is an option if the wait drags); control-partners stage 2 collection overnight (then tell planning); sizing Job after stage 2 frees michael-persistent; SAM 3 noise, VGGT-Omega and the final Job once noise is published.
- See: world-model-stacking-encoders/experiments/2026-11-06-perturbed-frames/jobs/PHASE-B.md (section 11), world-model-stacking-encoders/experiments/2026-11-06-perturbed-frames/LOG.md, world-model-stacking-encoders/experiments/2026-11-10-feature-overlap/PLAN.md (amendment), https://github.com/michaelnkr808/research26-27/pull/23

## 2026-11-11: vis-meetings, slide prep for the 11/11 mentor meeting

- Went through four versions of the user's deck and wrote short, paste-ready lines for the new 11/11 slides (noise check status, our copy vs the paper's model, disturbed scenes, feature overlap); planning fact-checked every line and seven were fixed.
- Found out the proxy DINOv3 copies finish about 8% of LIBERO-Plus (and about 1 in 5 clean) vs the paper's 84%; the likely reason is the paper's pretrained Wan 2.2 start, but a pipeline bug is not ruled out. The paper trains 10 epochs at batch 128 like ours and states no model size.
- Made the "ours vs paper" figure (writing/figures/2026-11-11-ours-vs-paper.png, Desktop copy).
- Open loops: after the meeting, log the mentor's answers in the direction LOG and send planning one line; user decides on the two proposed proxy bug checks (implementation window) before the 16-layer retrain; PR #25 (control partners stage 2) and PR #23 (overlap sizing) await the user's OK.
- See: world-model-stacking-encoders/LOG.md (2026-11-11 vis-meetings entry), world-model-stacking-encoders/writing/figures/make_ours_vs_paper.py

## 2026-11-11: stacking-encoders implementation, PRs 24 and 25 merged, SAM 3 done, /doctor cleanup

- Merged PR #25 (control-partners stage 2: the two labels hold on all three seeds) and PR #24 (RTX 3090 backup Jobs for SAM 3) on the user's OK.
- All five SAM 3 extractions and the phase B recovery Job finished cleanly; four of five SAM 3 logs saved (camera's was already gone from the node).
- /doctor: install healthy and current; turned off the unused clangd-lsp and swift-lsp plugins and made auto mode the default permission mode (both in ~/.claude/settings.json).
- Open loops: VGGT-Omega and the feature-overlap sizing Job are on hold until the user says go; then the final phase B Job and the 16-layer DINOv3 retrain; turn off the Google Calendar connector in /mcp (user); pick 3-worker cap vs uncapped (global CLAUDE.md vs memory); push main.
- See: world-model-stacking-encoders/LOG.md (2026-11-11 implementation lead close), world-model-stacking-encoders/experiments/2026-11-06-perturbed-frames/LOG.md (2026-11-11 [agent, lead])

## 2026-11-11: stacking-encoders draft, paper setup section written, reviewed twice, merged (PR #20)

- Drafted the paper's setup section (source result, encoders, data, our proxy WAM, rules fixed in advance) in its own worktree; a fresh reviewer checked every sentence against the repo, the code and the arXiv PDF in two rounds. All paper numbers matched; nine overstatements fixed (mainly Fast-WAM's choices listed as the paper's, and Amendment (c) written as done).
- Found: SigLIP 2 Large in the frozen-feature tests vs So400m in the proxy, with no decision record; the paper states it; told the user and planning.
- User decisions: merge PR #20; seed-noise approval note in the supplement only; LIBERO and LIBERO-Plus cited (the user read both; relayed to the literature session for the bib notes); trimming the section (about 1.45 pages) waits on the mentor's length answer.
- Open loops: method halves of sections 4 to 6, limitations and the supplement trail, each by PR with a fresh reviewer; after the mentor meeting, record the framing, PLAN Next 5 and length answers and trim the setup if the paper is short; title and contributions still held for the mentor gate.
- See: world-model-stacking-encoders/writing/paper/sec/3_setup.tex, world-model-stacking-encoders/LOG.md (2026-11-10 [agent, lead, draft] setup section merged), PR #20 review comments

## 2026-11-11: stacking-encoders review session, control-partners stage 1 read requested, plan presented, not dispatched

- After a /clear, the planning lead resent its request for an independent read of the control-partners stage 1 result (labels vs the locked rule, the shuffled-DINOv3 caveat, what one seed can say). Read the standards, the experiment PLAN, README, both result reports and its LOG.
- Presented a plan (one review worker recomputing the rule from the raw JSONs, Codex review, one fresh fact-check reviewer, report in reviews/ by PR). The user closed the session before answering, so nothing was dispatched and no report exists.
- Open loops: re-ask the user's OK for the stage 1 read in the review session window (the relayed planning approval does not count), then run it; tell the planning lead when posted. Stage 2 of control-partners was still running at close.
- See: world-model-stacking-encoders/experiments/2026-11-09-control-partners/README.md, memory review-session-and-reviews-folder.md

## 2026-11-12: stacking-encoders planning session, mentor's notes, move to a pretrained Wan 2.2 start, Runpod and Modal set up

- Read the mentor's written notes on half the deck (fusion questions, start from pretrained weights at 8 layers, clean training and testing first) and answered the factual ones from the paper note. Rechecked that the paper still has no released code or weights.
- User decisions: drop the 16-layer from-scratch retrain; cheap tests are no longer the focus but may run in the background on Nautilus; start from the pretrained Wan 2.2 video transformer; investigate how the encoders are combined first; pay at most about $30 a month for short cloud jobs, keep long training on Nautilus, ask the PI for compute; applied for Modal academic credits.
- Set up Runpod (official plugin, signed in, empty account confirmed) and Modal (CLI installed, token verified). Nothing created or billed.
- Open loops: the plan is proposed but not dispatched, it needs the user's go in a planning session: (1) decisions/005 plus a dollar budget rule in STANDARDS, (2) a sizing check design for a pretrained 8-layer model, to run on Modal's free credit, ideally before the PI meeting. The rest of the mentor's notes have not arrived. PR #16 is still open. Local main is ahead of origin, unpushed.
- See: world-model-stacking-encoders/LOG.md (2026-11-12 [agent, lead, planning]), world-model-stacking-encoders/PLAN.md (Now, proxy item, 2026-11-11 line), memory paid-cloud-compute-runpod.md and stacking-encoders-mentor.md

## 2026-11-14: stacking-encoders implementation session (background copy) closed after the 2026-11-10 duplicate-lead overlap

- This session's context was reset on 2026-11-10 while it was the implementation lead. On waking it reconstructed the day's three cluster submissions (fetch-draws, control-partners stage 2, the A6000 render twin) from the cluster, decoded the stage 1 draw files with matching hashes, and pushed a direct-merge chore with LOG, PLAN and README entries plus the two rendered manifests (merge 767c790). Cluster use was read-only.
- The user's live session (vis-implementation-agent) then messaged that it was the real lead with the same work in flight. This session stood down, stopped its watcher, and the other session kept 767c790 as the record and added its own follow-up. No revert.
- Decisions: vis-implementation-agent is the implementation lead; this copy is closed. Memory updated with the lesson (check ListAgents and git log after a context reset before logging).
- Open loops: none for this session. Phase B and control-partners work continue in vis-implementation-agent.
- See: world-model-stacking-encoders/LOG.md (2026-11-10 [agent, implementation lead] A6000 twin entry), memory duplicate-lead-sessions.md

## 2026-11-15: stacking-encoders planning session, decision 006 merged, paper facts checked, stage 1 baseline design signed off and handed off

- Merged decision 006 (PR #30): the research question is the user's wording (why some encoder pairings work while others make it worse, with VAE-grid concatenation), the other 13 proxy runs are parked, and "done" is staged (baseline, pairings the paper's way, fusion variants, frozen-feature results as support). Stored the mentor's written message verbatim in the direction LOG and checked the paper's depth and clean-LIBERO numbers against the PDF: clean LIBERO cannot separate helpful from harmful pairings, the harm shows only on LIBERO-Plus.
- Relayed the user's decisions to the experiment session, which merged its sizing PRs, ran the sizing check (one full run of the 8-layer pretrained-start model is about a day on one L40S) and put the whole Wan 2.2 transformer on the cluster.
- Had the stage 1 baseline design written and reviewed twice (PR #34, merged): 8-layer model from pretrained Wan 2.2, no extra encoder, clean LIBERO. The user's choices: two-line pass bar (90 clear pass, under 36 fail, in between needs 27 on full LIBERO-Plus), action expert derived from the Wan 2.2 weights, training on Nautilus, one seed now and two alongside the pairing runs; the smaller items confirmed at the plan's recommendations.
- Open loops: the experiment session waits on the user to run its pinned `!` download of the Fast-WAM and LIBERO-Plus source archives in its window; then it reads them and brings the user the validation-split and image-augmentation choices, adds the user's confirmation of the small items to the baseline PLAN by PR, writes the training code by PR, and asks for a go before any cluster run. Planning's next piece is the stage 2 PLAN (DINOv3 alone, a helpful and the harmful pairing), after the baseline's result. PR #28 (draft session) is still open. This timeline commit is merged on local main but not pushed.
- See: world-model-stacking-encoders/experiments/2026-11-15-pretrained-start-baseline/PLAN.md (Amendments), world-model-stacking-encoders/LOG.md (the 2026-11-15 [agent, lead, planning] entries), world-model-stacking-encoders/decisions/006-research-question-and-done.md, memory planning-to-implementation-handoff.md

## 2026-11-15: stacking-encoders planning session (later), stage 2 pairings design signed off and merged, DreamWorld paper and code read, handed to the experiment session

- Checked on the experiment session (stage 1 code merged, sanity Job submitted, no result yet), then had the stage 2 design written and reviewed twice: DINOv3, SigLIP 2 and C-RADIOv4 alone, DINOv3 + SigLIP 2 as the helpful pairing and DINOv3 + C-RADIOv4 as the harmful one, three training seeds each, the full LIBERO-Plus benchmark on every final model, and a decision rule fixed before any result (PRs #41 and #43, merged).
- User decisions: the cautious rule, two extra seeds once if the answer is "can't tell", keep the cached SigLIP 2 variant and record it, a second volume plus a backup copy of each final model, delete resumable checkpoints after evaluation, no training-code change that the paper doesn't have ("stay faithful to the paper"), and no scope cut decided in advance.
- Read DreamWorld's paper and released code (PRs #42 and #44, merged). They do not settle what the stacking paper did where it is silent (feature width, normalization, resizing). Two of the lead's chat statements were wrong and were corrected after review: the stacking paper does not say it keeps full-width features, and DreamWorld upsamples where our code downsamples the partner encoders' grids.
- Drafted a general prompt for the presentations agent as a Worker variant, in chat only. It is not saved to the Arbor role file.
- Open loops: the experiment session runs stage 1 (sanity run, then the baseline); nothing of stage 2 runs before the baseline passes, and the sweep, the second volume and every checkpoint deletion need the user's go in that window, where it will also ask whether a deletion waits for a verified backup copy. Planning's next job is filling the stage 2 plan's blanks by dated amendment once the baseline's result and timings exist, then the stage 3 design. If the user wants the presentations agent prompt kept, it has to be copied from this session's chat into presentations-agent.md in Arbor's roles folder. PR #28 (draft session) is still open. The user's own edits to CLAUDE.md and _standards/05 are uncommitted and were left alone.
- See: world-model-stacking-encoders/experiments/2026-11-15-pretrained-start-pairings/PLAN.md (Decision rule, Amendments) and dreamworld-reading.md, world-model-stacking-encoders/LOG.md (2026-11-15 [agent, lead, planning], stage 2 pairings design), memory fidelity-over-cheapness.md and external-downloads-via-user.md

## 2026-11-15/16: stacking-encoders experiment session, stage 1 baseline code built and sanity checks 1-3 passed on Nautilus

- Took the stage 1 baseline hand-off from planning. Had Fast-WAM's training code read (no validation split, no image augmentation, 50 starting layouts per test task) and the old proxy's clean results counted; the user answered the remaining design questions (train on all episodes, reuse the saved image latents, keep the even action noise setting). PRs #35 to #40 merged: reading, tally, PLAN amendment, timing-only evaluation mode, training script with the Fast-WAM action-expert port, and the sanity Job.
- Submitted the sanity Job on the user's go (design locked from that run). It queued about 72 min, ran 8 min on an L40, and passed all three checks (overfit one batch, loss at start, inputs). Measured on the L40: about 5.0 s per optimizer step at micro-batch 8, peak 35.8 GiB, so about 30 L40-hours of steps for the full run.
- Answered planning's storage question (a second 1Ti CephFS volume is allowed, no documented quota) and received the stage 2 pairings hand-off; nothing of stage 2 is approved to run.
- Open loops: sanity check 4 (short training run with a small checkpoint interval to time a CephFS write, then the timing-only 40-rollout smoke on L4) needs its Job prepared and the user's go; ConfigMap stacking-encoders-baseline-src is kept on the cluster for it. The sanity results went to main as a direct LOG chore, not a reviewed PR; the user was offered a retroactive review. The three proxy retrain manifests share the render bug fixed in the sanity Job (logged, not fixed). Stage 2 code prep was offered and is waiting on the user's answer.
- See: world-model-stacking-encoders/experiments/2026-11-15-pretrained-start-baseline/LOG.md (last entry, Next line), jobs/RUN-sanity.md, results/sanity/, fastwam-reading.md; experiments/2026-11-15-pretrained-start-pairings/PLAN.md (Hand-off)

## 2026-11-16: stacking-encoders experiment session, baseline sanity check 4 run, all four sanity checks passed

- A worker prepared the sanity check 4 Jobs and a fresh reviewer checked them (PR #46, merged on the user's OK). Nothing was blocking; the fixes included a longer in-pod training timeout and the correct way to project the full run's checkpoint cost.
- The user chose three timed checkpoint writes (every 75 of 300 steps) and no second evaluation pass inside the smoke pod.
- Training half on an L40S: checkpoint writes to the cluster storage took about 20 s each, so the full baseline run is about a day on an L40S (a bit longer on an L40). Smoke on an L4: the model fits, the GPU stayed busy at 4 environments, and the 2,000-rollout evaluation will take at most about 9 L4-hours. Both Jobs were deleted; a new ConfigMap stacking-encoders-baseline-eval-src holds the evaluation code.
- Open loops: the user's go for the full training run, with two answers pending (L40 and L40S, or L40S only; time limit of about twice the projected time, Open item 4); its Job manifest goes by PR first. The check-4 ckpt.pt and model.pt (about 27 GB) stay on the PVC at /mnt/proxy/runs/baseline-check4-none-s0-try1/ until the user OKs deleting them. Host memory reached the pod limit in both Jobs without an OOM kill (likely page cache); log anonymous memory when sizing the full run. The user's own edits to CLAUDE.md and _standards/05 remain uncommitted and were left alone.
- See: world-model-stacking-encoders/experiments/2026-11-15-pretrained-start-baseline/LOG.md (the four 2026-11-16 entries), jobs/RUN-check4.md, results/check4/, PLAN.md "Parameters set from measurements"

## 2026-11-16: hear-lab planning session, direction folder set up (PR #45), Vast.ai access working, RMS-AQA scoped

- Wrote the hear-lab-experiment role prompt (Vast.ai first, Nautilus second, the user's OK before every rental or Job) and got the vastai tool reading the HEAR Lab team account and its 200 dollars of credit. Nothing has been rented.
- Set up the hear-lab direction folder for the question "what benefits does a world action model gain from spatial audio" (PI: Dongheon Lee). A fresh reviewer fact-checked it against the sources, the fixes went in, and PR #45 was merged on the user's "merge". The LOG entry (dated 2026-11-15) holds the RMS-AQA reference facts.
- Checked the official Interspeech 2027 dates: papers are due 2027-02-09, earlier than the "end of February" first assumed. The call for papers was not open yet, so the format is unchecked.
- Decisions from the user: "the cluster" means Vast.ai; experiments are run by the hear-lab-experiment session, not planning; hear-lab stays independent of the vis-lab work.
- Open loops: the RMS-AQA dataset access request (Hugging Face, PeacefulData/RMS-AQA) is waiting for approval; the PI has to say what "get it running" means (score an existing model, reproduce the trained baseline, or enter the challenge) and whether the 200 dollars should cover a training run. When both are back, planning writes the experiment PLAN.md and hands it to hear-lab-experiment. The literature survey waits for the user's go. hear-lab/STANDARDS.md still has the old deadline wording, to be fixed in its next pull request.
- See: hear-lab/LOG.md (2026-11-15 entries), hear-lab/PLAN.md, hear-lab/README.md, hear-lab/STANDARDS.md

## 2026-11-16: presentations session, picture-first deck for the PI meeting (compute funding ask)

- Read the user's deck export (Research Proposal-14, saved 2026-11-11) and the repo, and gave a picture-first slide plan with likely questions, weak spots and unchecked items for the PI meeting on Wednesday 2026-11-18.
- Made 13 new figures plus three reused ones in writing/figures/2026-11-16-pi-deck/ (script make_pi_deck_figures.py, tiles/, by-slide/ previews, README.txt), a Desktop copy at ~/Desktop/pi-deck-images, and an 8-slide Google Slides deck in the user's Drive built from a hand-written pptx (pip install was blocked).
- Checked and corrected: the stage 2 trim order is only a worker suggestion (the user decides later), the cluster-crowding and GPU-use arguments are weaker for the new model, and the deck's feature-overlap status (built) conflicts with the README (design only).
- Open loops: the user picks the dollar amount to ask for and fills it on slides 7 and 8; re-read Runpod, Modal and Vast.ai prices before Wednesday; stage 1 training should end about 2026-11-17 16:00 UTC (its clean-eval result may or may not exist by Wednesday); the figures folder and Desktop copy are uncommitted; deck has no speaker notes.
- See: world-model-stacking-encoders/writing/figures/2026-11-16-pi-deck/README.txt, experiments/2026-11-15-pretrained-start-pairings/PLAN.md (run counts, Open item l), experiments/2026-11-12-pretrained-start-sizing/README.md

## 2026-11-17: stacking-encoders experiment session, stage 1 baseline trained and passed clean LIBERO

- Watched the full baseline training on a Nautilus L40 to the end (about 32 h, exit 0, no restarts, GPU about 97%), saved its logs and deleted the Job; the trained model.pt stays on the PVC.
- On the user's "yep submit it", ran the 2,000-rollout clean LIBERO evaluation on 4 L4s (about an hour each, no errors, same model hash in every pod). Test-set use was logged before any outcome was read.
- Result: four-suite average 92.55%, above the locked 90% pass line, so stage 2 (encoder pairings) starts per the PLAN with this run as the no-encoder reference. A fresh reviewer recomputed it from the records; PR #52 merged on the user's OK.
- Open loops: tell the planning lead (vis-lab-planning) the result and that the experiment PLAN status is theirs to mark done; the user's answers on a per-task tally script and on deleting the check-4 checkpoint files (/mnt/proxy/runs/baseline-check4-none-s0-try1/). The presentations session's PI-deck note expected this result might not exist by Wednesday; it now does.
- See: world-model-stacking-encoders/experiments/2026-11-15-pretrained-start-baseline/LOG.md (2026-11-17 entries), README.md, results/clean-eval/

## 2026-11-17: stacking-encoders planning session, stage 1 pass recorded and the stage 2 plan amended (PR #58)

- Joined local main with GitHub's main at the start, answered a run of paper questions (Wan 2.2, VAE, flow matching, Fast-WAM), set up a literature session (vis-lab-literature) that then corrected three things said from memory, and sent a résumé summary to the résumé session.
- Stage 1 passed (92.55% on clean LIBERO against the 90 bar, no extra encoder, one training seed). Wrote the dated amendment to the stage 2 PLAN with the measured stage 1 numbers; a fresh reviewer checked it twice; merged as PR #58 on the user's OK. LOG entry, PLAN Now line and README status merged and pushed (c6b5f45).
- Decisions from the user: no extra seeds if the DINOv3 sanity check fails; the pair against C-RADIOv4 alone stays on three seeds with the thresholds computed before any extension; missing clean LIBERO episodes are rerun first, and a model more than 1% short drops out of the verdict (the strict rule, confirmed after its consequence was explained). The user also corrected a message to the mentor: the old 8% was LIBERO-Plus with DINOv3, the fair clean LIBERO comparison is about 20% to 92.5%.
- Open loops: the stage 3 design (fusion variants) is not started, and it begins by asking the user which variants to test. In the experiment window: the user's merge OK for stage 2 code PRs #54 to #57, then the go for the second volume and the sweep. LIBERO-Plus evaluation cost is still unmeasured. For the PI meeting on 2026-11-18: training is about 550 card-hours at the L40 rate, up from about 400. The direction PLAN.md "Now" section is long and partly out of date.
- See: world-model-stacking-encoders/LOG.md (2026-11-17 [agent, lead, planning]), world-model-stacking-encoders/experiments/2026-11-15-pretrained-start-pairings/PLAN.md (Amendments, 2026-11-17)
