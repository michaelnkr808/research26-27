# World-Model Planning Horizon: Topic Standards

> These rules extend the general standards in `../_standards/`. A rule that *relaxes* a general rule must name it and say why.

## Scope
- **Research question:** see `README.md` (draft until the user and mentor confirm it).
- **In scope / out of scope:**
  - **In:** the three-condition oracle comparison (learned predictor + embedding distance; real future embeddings + embedding distance; real future state + physical distance); cheap demo-based geometry checks; existing cost fixes (NGC, temporal-distance) applied as tests, not as our contribution; Meta-World, DINO-WM tasks, LIBERO.
  - **Out for now:** proposing a new planning cost (crowded); fine-tuning encoders; real-robot evaluation.

## Domain sources
- Key venues: ICML, NeurIPS, ICLR, CoRL, RSS, ICRA. Most sources are 2026 arXiv preprints; flag them (`02`).
- Canonical references: Planning Limits (arXiv 2609.39235) is the paper this project extends; NGC (2610.25013); DINO-WM (ICML 2025).
- Trusted datasets or benchmarks: Meta-World (Planning Limits' setting); DINO-WM datasets (no data license stated, check before use).
- Sources to treat with caution: Planning Limits mentions no code or data release (agent check of full text, 2026-10-27), so its candidate-sampling setup may need rebuilding; reproduce its numbers before comparing to them.

## Methods and evaluation
- Standard baselines: Planning Limits' own conditions (learned predictor; exact dynamics scored in state space); plain L2 and cosine costs.
- Metrics (and why): expert percentile and top-1 selection rate per encoder and per horizon k (Planning Limits' measures, for comparability); plannable range in steps under each condition; Kendall tau between embedding distance and steps-to-goal along demos (cheap geometry check); closed-loop success, mean and std over 3 seeds.
- Required statistical tests or number of runs: bootstrap CIs over start states or trajectories; at least 3 seeds for trained predictors (`07`). Encoder-level conclusions with about 8 encoders are tallies, not significance tests.
- Known pitfalls: the scoring rule changes between conditions (b) and (c) by design, so keep every other setting identical; candidate distribution changes encoder rankings (Planning Limits appendix), so fix it before seeing results; rendering must match the training image pipeline (resolution, crop).

## Tools and environment
- Languages, libraries, hardware: Python; reuse `../stacking-encoders/src/extract_features.py` (encoder loaders) and `../stacking-encoders/src/diagnostics.py` (ridge probes, demo-level bootstrap) where possible.
- Environment file: TBD
- Experiment tracker (W&B / MLflow / local logs): local JSON logs per run, as in stacking-encoders (TBD)
- Compute budget (GPU-hours / API $): **not set yet.** Set one before the first GPU run (`07`). Until then, any GPU Job is a stop-and-ask gate.
- Checkpoint storage location: TBD
- Nautilus setup (see `_standards/08`): every reportable run is a Job (never a trailing `sleep`), start on 1 GPU, utilization above 40%, requests and limits within 20%. PVC, image and GPU types TBD.

## Conventions
- Notation or terminology: condition (a) = learned predictor + embedding distance; (b) = real future embeddings + embedding distance; (c) = real future state + physical distance. Predictor error = (a) vs (b); geometry error = (b) vs (c); scoring-rule limit = (c) vs perfect.
- Figure and table conventions: TBD
- Target output format or venue: TBD with the mentor.

## Overrides of general standards
| General rule | Override | Reason |
|---|---|---|
