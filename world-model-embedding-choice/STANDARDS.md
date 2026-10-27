# World-Model Embedding Choice: Topic Standards

> These rules extend the general standards in `../_standards/`. A rule that *relaxes* a general rule must name it and say why.

## Scope
- **Research question:** see `README.md` (draft until the user and mentor confirm it).
- **In scope / out of scope:**
  - **In:** frozen pretrained image and video encoders; training-free scores on recorded trajectories (controllable share, action recoverability, multi-step versions); latent world models trained on frozen features (DINO-WM protocol) as ground truth; DINO-WM tasks (PushT, Wall, PointMaze), Meta-World, LIBERO.
  - **Out for now:** fine-tuning encoders; new planning costs (a crowded area, see `../world-model-planning-horizon/`); real-robot evaluation.

## Domain sources
- Key venues: ICML, NeurIPS, ICLR, CoRL, RSS, ICRA, CVPR. Most 2026 world-model work is arXiv-only; flag it as not peer reviewed (`02`).
- Canonical references: DINO-WM (ICML 2025), ALPS, Planning Limits, AD-WM (see `literature/references.bib`).
- Trusted datasets or benchmarks: DINO-WM datasets (OSF link in its GitHub README; **no data license stated**, check before use); Meta-World; LIBERO.
- Sources to treat with caution: per-encoder planning numbers from different papers use different planners and candidate sets; Planning Limits notes encoder comparisons depend on the candidate distribution. Don't pool them without saying so.

## Methods and evaluation
- Standard baselines (every comparison includes): single-frame state probe (ALPS-style); single-frame action probe; CKA to DINOv2; random ranking. Expensive reference: the trained world model's own prediction error.
- Metrics (and why): Spearman and Kendall tau between score and planning success (only the order matters for choosing an encoder); top-1 accuracy and regret of the score's pick (the actual decision); planning success rate and Chamfer distance as DINO-WM and ALPS report them (comparability).
- Required statistical tests or number of runs: bootstrap CIs over trajectories (frames within a trajectory are correlated); at least 3 training seeds for any world model used as ground truth (`07`). With fewer than about 10 encoders, report tallies and rank correlations with CIs, not significance.
- Known pitfalls: splitting probes by frame instead of trajectory (leakage); encoder dimension differences (reduce to a matched dimension, tune ridge alpha on validation); CLS vs patch features are different encoders for this purpose (DINO-WM Table 2).

## Tools and environment
- Languages, libraries, hardware: Python; reuse `../world-model-stacking-encoders/src/extract_features.py` (encoder loaders) and `../world-model-stacking-encoders/src/diagnostics.py` (ridge probes, demo-level bootstrap) where possible.
- Environment file: TBD
- Experiment tracker (W&B / MLflow / local logs): local JSON logs per run, as in stacking-encoders (TBD)
- Compute budget (GPU-hours / API $): **not set yet.** Set one before the first GPU run (`07`). Until then, any GPU Job is a stop-and-ask gate.
- Checkpoint storage location: TBD
- Nautilus setup (see `_standards/08`): every reportable run is a Job (never a trailing `sleep`), start on 1 GPU, utilization above 40%, requests and limits within 20%. PVC, image and GPU types TBD.

## Conventions
- Notation or terminology: "controllable share" = R-squared gain from adding the action when predicting the embedding change; "action recoverability" = inverse-dynamics probe R-squared. Never call either "complementarity" (that term belongs to stacking-encoders).
- Figure and table conventions: TBD
- Target output format or venue: TBD with the mentor.

## Overrides of general standards
| General rule | Override | Reason |
|---|---|---|
