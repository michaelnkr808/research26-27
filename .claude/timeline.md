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
