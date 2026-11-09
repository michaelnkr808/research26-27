# Stacking Encoders

**Status:** exploring. The cheap frozen-feature tests (input side and target side) did not flag the harmful DINOv3 + C-RADIOv4 pair. The multi-seed proxy replication (stage 4) is running: as of 2026-11-07 ~22:26 UTC, both DINOv3 seeds are trained and 7 of 16 evaluation shards are done with 0 errors; the rest run on L4-only Jobs and two older L4 pods (finish time speculative, the reviewer's estimate in experiment LOG (k): about 06:30-07:00 UTC 2026-11-08; risk recorded in (k): NRP's 6-hour bare-pod rule could delete them first, at about 02:46 and 03:13 UTC, covered by the watcher and the standing one-shard L4 resubmit OK), details in `PLAN.md` and `experiments/2026-10-27-proxy-calibration/LOG.md` 2026-11-07 (k). The perturbed-frame test (`experiments/2026-11-06-perturbed-frames/`) was redesigned after its validation check failed (probes now trained on our own re-renders, user decision 2026-11-09); phase A (render, extraction, dev gate) was submitted 2026-11-09 07:34 UTC. Current agenda: `PLAN.md`. History: `LOG.md` (direction level), each experiment's own `LOG.md` (details).
**Goal:** publish (user, 2026-10-19). Venue and timeline to discuss with the PhD mentor on Thursday 2026-10-22.
**Question (draft, not yet confirmed):** Can a cheap, *conditional* measure of what encoder B adds to encoder A (task-relevant information not recoverable from A), computed on frozen features from robot data, predict which encoder combinations improve a World Action Model (WAM) before training, where symmetric similarity (CKA) does not?
- Secondary, possibly a separate paper: why do general-purpose encoders match or beat specialists?
- **Hypothesis not chosen yet.** Candidates and definitions (similarity vs complementarity) are in `writing/hypotheses-draft.md`.
**Mentor feedback (2026-10-19):** recorded in `LOG.md` 2026-10-19 (d), `decisions/002-feasibility-before-proxy-training.md` and `writing/hypotheses-draft.md`. Later mentor feedback: `LOG.md` 2026-10-27 (f) (mentor meeting outcome; run spec for the target-side diagnostics), 2026-10-27 (g) (LIBERO study session; the "intermediate steps" point), 2026-10-27 (h) (target-side Job try 2; the mentor also wants the target-encoder run) and 2026-11-06 [agent, docs] (mentor meeting 2026-11-06 recorded).
**Plan:** see `PLAN.md` (living agenda: now, next, parked, recently done).
**Reviewer feedback (2026-10-28):** recorded in `LOG.md` 2026-10-28 (b) (reviewer feedback: proxy round reframed as a proxy replication), 2026-10-28 (e) (second reviewer round) and 2026-10-28 (f) (decision rule locked before the pilot; two more reviewer rounds), and in the experiment PLAN's outcome rule and decision 003's amendment.
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
  The two answer keys differ in kind, too. Ours is per frame: the action the human teleoperator recorded at that frame of a clean LIBERO-Goal demo (or D's real future features), with no robot ever acting. The paper's is per episode: whether the trained robot, acting on its own in a perturbed LIBERO-Plus scene, completed the task (success or fail). Our probes measure whether frozen features describe clean demos; the paper's harm appears in closed-loop behaviour under perturbation.
- **Cost** (verified): large encoders are slow to extract (SAM 3 at 1008 px: 5.5 frames/s on a 2080 Ti; ~10 GPU-h for D, S, G on one suite). Diagnostics are cheap once features exist: 30–50 CPU-min per pair.
- Starting point: liu2026what (arXiv 2610.13024v1) judges complementarity qualitatively and uses one training checkpoint per config. See `literature/notes/liu2026-what-visual-representations-wams.md`.

## Key files
- Research proposal for the mentor: one-page version `writing/proposals-2026-10-27/p1-short.md` (Google Doc: https://docs.google.com/document/d/1JJj8KcgBusH65LHY6CsFEiFqACy2lQI3f-wcgI3WF40/edit); full notes `writing/proposals-2026-10-27/p1-stacking-encoders.html` (Google Doc: https://docs.google.com/document/d/1GgJm4coJ8Kn2J_D0IfIBYZpXSwlVU_3wVz17nUM7MBY/edit)
- `STANDARDS.md`: topic rules (scope, domain sources, baselines, metrics and pitfalls, tools and environment, compute budget and Nautilus rules are filled; only the figure/table convention is blank)
- `LOG.md`: research log for the direction (short session summaries; before 2026-11-04 it holds the full detail too). Each experiment folder has `PLAN.md`, `LOG.md`, `README.md`.
- `decisions/`: decision records (001: proxy-policy ground truth; 002 (accepted 2026-10-19): feasibility tally before training; 003 (accepted 2026-10-27): go straight to the multi-seed proxy policy)
- `literature/`: paper notes and references.bib (LIBERO, LIBERO-Plus, liu2026what and C-RADIOv4 [`ranzinger2026cradiov4`]; the agostinelli2022 note, the closest prior work per the mentor, is in `literature/notes/`, not in the bib)
- `writing/figures/`: slide figures for the mentor deck (2026-11-07), each with its script in `src/` (`plot_training_curve.py`, `plot_compute_cost.py`, `plot_slide_diagrams.py`); LOG 2026-11-07 [agent, slides]
- `writing/hypotheses-draft.md`: definitions, candidate hypotheses, the feasibility experiment, a novelty statement draft, and Thursday prep
- `experiments/2026-10-16-fastwam-step-bench/`: 1-GPU Fast-WAM training-step benchmark. A6000: ~1.3 samples/s/GPU, unreviewed, not a finding.
- `src/bench_fastwam_step.py`: the benchmark script
- `src/extract_features.py`: Phase 1 frozen-feature extraction (LIBERO HDF5 → per-encoder HDF5). Loaders: SigLIP 2, C-RADIOv4, DINOv3, SAM 3, VGGT-Ω. V-JEPA 2.1, UniTok and HRP are stubs.
- `experiments/2026-10-18-extract-libero-goal/`: first extraction Job. Features are on PVC `michael-persistent` at `/persistent/stacking-encoders/features/`.
- `src/target_diagnostics.py`, `src/test_target_diagnostics.py`: target-side diagnostics (H-E: T0 loss share, T1 target predictability, T2 reduced-rank interference) + synthetic tests; `experiments/2026-10-21-target-diagnostics/`: PLAN, Job manifest and results (ran 2026-10-27; see its `README.md`)
- `src/diagnostics.py`, `src/test_diagnostics.py`: Phase 1 diagnostics (CKA, probes, gain, demo-level bootstrap CIs) + synthetic tests; `experiments/2026-10-19-diagnostics-libero-goal/`: first results (pilot)
- `experiments/2026-10-19-feasibility-tally/`: the decision-002 tally (D + L, R, S, G on libero_goal). Manifests, logs, result JSONs, full tables.
- `data/README.md`: provenance of every dataset and feature cache (source, version, license, where stored)
- `experiments/2026-10-27-proxy-calibration/`: the multi-seed proxy replication (decision 003): proxy-policy training, rollouts on LIBERO-Plus, paired analysis. Running; see its `README.md` and `LOG.md`.
- `data/pvc-features-cephfs.yaml`: CephFS PVC `stacking-encoders-features` (100Gi, RWX). Holds the D, S, G features (17.6G) under `stacking-encoders/features/`. L and R stay on `michael-persistent`.
- `literature/liu2026what-table{1,10}-*.csv`: paper tables, checked against the arXiv HTML 2026-10-18
- `literature/reading-list/`: candidate papers for the user to read (50 leads in three themes, plus a top 10 in its `README.md`), 2026-11-07. Leads only, not citations. Owned by the literature-review session; other sessions read it but don't edit it.
