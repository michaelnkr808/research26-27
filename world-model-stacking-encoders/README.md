# Stacking Encoders

**Status:** exploring. Feasibility tally done (2026-10-21): on libero_goal, neither CKA nor gain(B|D) on frozen features flags the harmful D+R combo. Proposals presented to the mentor 2026-10-27; her advice: focus on the benchmark (LIBERO) and how stacking encoders relates to it. The target-side test (H-E, LOG 2026-10-21 (i)/(j)) has code and passing tests; run spec `experiments/2026-10-21-target-diagnostics/RUN.md`. First cluster try (2026-10-27) was OOM-killed on the first pair (Linux memory far above the Mac sizing); Job deleted, no target-side result yet. Try 2 ran to completion 2026-10-27: R is not the largest target-side interference (G is), so H-E fails the pre-registered rule (key findings updated with user OK). Job and ConfigMap deleted. Next (decision 003, user 2026-10-27): multi-seed proxy-policy calibration (none, D, R, D+R, D+L × 3 seeds, clean LIBERO + LIBERO-Plus); plan in `experiments/2026-10-27-proxy-calibration/PLAN.md`. PLAN steps 1–3 done 2026-10-27. 2026-10-28: user reviewed the 8 design choices and chose to match the paper (Wan 2.2 VAE latents, joint flow matching, the paper's T5, full 14 × 14 grids, Fast-WAM's data handling, ~756 GB feature cache on CephFS); stage 1 code rebuilt on Fast-WAM's own modules (`src/proxy_wam.py`, `extract_fastwam.py`, `t5_cache.py`, `fastwam_data.py`; 12 passing checks, real-data smoke on one episode). Next: step 4 extraction Jobs (needs go-ahead).
**Goal:** publish (user, 2026-10-19). Venue and timeline to discuss with the PhD mentor on Thursday 2026-10-22.
**Question (draft, not yet confirmed):** Can a cheap, *conditional* measure of what encoder B adds to encoder A (task-relevant information not recoverable from A), computed on frozen features from robot data, predict which encoder combinations improve a World Action Model (WAM) before training, where symmetric similarity (CKA) does not?
- Secondary, possibly a separate paper: why do general-purpose encoders match or beat specialists?
- **Hypothesis not chosen yet.** Candidates and definitions (similarity vs complementarity) are in `writing/hypotheses-draft.md`.
**Mentor feedback (2026-10-19), driving the current plan:**
- Low similarity ≠ complementarity, so define complementarity.
- Fix the hypothesis first.
- Start with 1 dataset and 5–10 encoders, CKA, and a manual tally.
- Position against agostinelli2022 (ensemble transferability metrics).
**Plan (decision 002, accepted by user 2026-10-19):**
1. ~~Feasibility tally, no training~~ done 2026-10-21: libero_goal, DINOv3 + 4 of 7 partners (L, R, S, G; V, U, H skipped). CKA and gain(B|D)/gain(D|B) with CIs, compared against the liu2026what Table 10 deltas.
2. Choose the hypothesis.
3. If justified, run the proxy-policy multi-seed study (decision 001).
**Reviewer feedback (2026-10-28, two rounds, accepted by user):** the proxy round is a *proxy replication*; if it doesn't reproduce the D+R drop, that doesn't show the paper's result was noise, only whether a higher-fidelity multi-seed WAM replication is worth running. Primary analysis = paired D+R − D per training seed on the same episodes (PLAN outcome rule, decision 003 amendment). Open for a workshop paper: five configs test one anomaly, not a predictor; a low-cost retrospective, exploratory check of our diagnostics against the paper's 16 composition results is a candidate next analysis (suggestive only; not started).
**What done looks like:** A conditional metric that ranks combo gains better than CKA, single-encoder score, and objective-overlap reasoning, on multi-seed ground truth. It must also explain harmful combos (e.g. D+R). Or a documented negative result.

## Key findings so far
Signed off by the user 2026-10-21 (input side) and 2026-10-27 (target side). Sources: `experiments/2026-10-19-feasibility-tally/` and `experiments/2026-10-21-target-diagnostics/` (one suite, 4 pairs, frozen features, linear probes). Paper deltas: liu2026what Tables 1 and 10 (`literature/liu2026what-table{1,10}-*.csv`).
- **The simplest form of the idea fails the cheap check** (verified, n = 4 pairs). The paper ranks the DINOv3 (D) pairs L (+1.83) > S (+0.49) > G (+0.40) > R (−5.41). Neither frozen-feature measure singles out the harmful D+R:
  - **CKA:** R is the most similar to D in all 6 feature-type × camera cells. But L, the paper's best pair, is nearly as similar (global agentview 0.576 vs 0.583), and G, the least similar by far (0.21 / 0.27), gave only a small gain.
  - **gain(B|D)** (global): R adds the most action-relevant information on top of D: +0.037 [95% CI +0.032, +0.042], vs S +0.033, L +0.017, G +0.014. That is the opposite of the paper's ranking for R.
- **On frozen features, every partner helps a little** (verified). All 4 gains are positive and small (+0.014 to +0.037 probe R²), and all partners probe about as well as D alone (R² 0.61–0.67; proprio 0.28). So the D+R drop does not show up in a linear readout of frozen features, and the mismatch is not explained by a broken encoder: the sanity checks match the pilot exactly.
- **The target-side version fails too** (verified, n = 4, pre-registered rule LOG 2026-10-21 (i)). In the WAM each encoder is also a future-feature target, so we measured how much adding B's future as a co-target hurts a rank-64 linear prediction of D's future. G interferes most (+0.034 [+0.033, +0.035] R²), then R (+0.015), L (+0.014), S (+0.003). R is not the largest, so H-E fails at this level; R's interference also grows, not shrinks, when encoders' losses are equalized. The ordering doesn't track the paper (L and R nearly tie here but are the paper's best and worst).
- **Not settled** (speculative): why D+R hurts. Neither the input nor the target role shows it on clean frozen features. Possible reasons:
  - noise from the paper's single training checkpoint per combo (its eval-seed sd for D+R is 1.01, the largest in Table 10, but training-seed noise is unmeasured);
  - training dynamics that frozen features can't show;
  - the perturbed LIBERO-Plus test conditions;
  - our measures are the wrong ones.

  The scales also differ: paper Δ is success-rate points, ours is probe R². Only orderings are compared.
- **Cost** (verified): large encoders are slow to extract (SAM 3 at 1008 px: 5.5 frames/s on a 2080 Ti; ~10 GPU-h for D, S, G on one suite). Diagnostics are cheap once features exist: 30–50 CPU-min per pair.
- Starting point: liu2026what (arXiv 2610.13024v1) judges complementarity qualitatively and uses one training checkpoint per config. See `literature/notes/liu2026-what-visual-representations-wams.md`.

## Key files
- Research proposal for the mentor: one-page version `writing/proposals-2026-10-27/p1-short.md` (Google Doc: https://docs.google.com/document/d/1JJj8KcgBusH65LHY6CsFEiFqACy2lQI3f-wcgI3WF40/edit); full notes `writing/proposals-2026-10-27/p1-stacking-encoders.html` (Google Doc: https://docs.google.com/document/d/1GgJm4coJ8Kn2J_D0IfIBYZpXSwlVU_3wVz17nUM7MBY/edit). Sibling candidate directions: `../world-model-embedding-choice/`, `../world-model-planning-horizon/`
- Research proposal draft for the mentor (2026-10-25, due 2026-10-27): Claude Doc https://claude.ai/artifact/B6uCkLxYV5FmWUUDKRQbzF (LOG 2026-10-25)
- `STANDARDS.md`: topic rules (compute budget + Nautilus rules filled; rest TBD)
- `LOG.md`: research log
- `decisions/`: decision records (001: proxy-policy ground truth; 002 (accepted 2026-10-19): feasibility tally before training; 003 (accepted 2026-10-27): go straight to the multi-seed proxy policy)
- `literature/`: paper notes and references.bib (incl. agostinelli2022, the closest prior work per the mentor)
- `writing/hypotheses-draft.md`: definitions, candidate hypotheses, the feasibility experiment, a novelty statement draft, and Thursday prep
- `experiments/2026-10-16-fastwam-step-bench/`: 1-GPU Fast-WAM training-step benchmark. A6000: ~1.3 samples/s/GPU, unreviewed, not a finding.
- `src/bench_fastwam_step.py`: the benchmark script
- `src/extract_features.py`: Phase 1 frozen-feature extraction (LIBERO HDF5 → per-encoder HDF5). Loaders: SigLIP 2, C-RADIOv4, DINOv3, SAM 3, VGGT-Ω. V-JEPA 2.1, UniTok and HRP are stubs.
- `experiments/2026-10-18-extract-libero-goal/`: first extraction Job. Features are on PVC `michael-persistent` at `/persistent/stacking-encoders/features/`.
- `src/target_diagnostics.py`, `src/test_target_diagnostics.py`: target-side diagnostics (H-E: T0 loss share, T1 target predictability, T2 reduced-rank interference) + synthetic tests; `experiments/2026-10-21-target-diagnostics/`: PLAN and Job manifest (not yet run)
- `src/diagnostics.py`, `src/test_diagnostics.py`: Phase 1 diagnostics (CKA, probes, gain, demo-level bootstrap CIs) + synthetic tests; `experiments/2026-10-19-diagnostics-libero-goal/`: first results (pilot)
- `experiments/2026-10-19-feasibility-tally/`: the decision-002 tally (D + L, R, S, G on libero_goal). Manifests, logs, result JSONs, full tables.
- `data/pvc-features-cephfs.yaml`: CephFS PVC `stacking-encoders-features` (100Gi, RWX). Holds the D, S, G features (17.6G) under `stacking-encoders/features/`. L and R stay on `michael-persistent`.
- `literature/liu2026what-table{1,10}-*.csv`: paper tables, checked against the arXiv HTML 2026-10-18
