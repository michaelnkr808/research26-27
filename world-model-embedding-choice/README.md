# World-Model Embedding Choice

**Status:** exploring. Candidate direction (proposal 2 of 3 drafted 2026-10-27 for the PhD mentor). Nothing run. The user is reading the key papers before deciding whether to take it to the mentor.
**Question (draft, not confirmed):** Does the share of frame-to-frame embedding change that is caused by the action (computed on frozen embeddings of recorded trajectories, with no world model trained) predict how well a latent world model plans in that encoder, better than single-frame probes?
**What done looks like:** A training-free score that ranks frozen encoders by latent-world-model planning success better than a single-frame state probe, CKA and random, checked first against published per-encoder results (DINO-WM, ALPS, Planning Limits) and then against our own multi-seed DINO-WM runs. Or a documented negative result.

## The idea in plain words
A latent world model (e.g. DINO-WM) plans by imagining embeddings of future frames and picking actions whose imagined result is closest to the goal's embedding. If most of an embedding's change between frames comes from things the robot doesn't control (lighting, background, noise), different actions look alike and planning fails. We measure, per encoder, how much of the change the action explains ("controllable share") and how well the action can be read back from two consecutive embeddings ("action recoverability").

## Key findings so far
- None. Nothing has been run.

## Reading list (in order; see `writing/proposal-2026-10-27.html` for the full table)
1. DINO-WM (arXiv 2411.04983, ICML 2025): the setup; Table 2 encoder comparison.
2. ALPS (arXiv 2610.30120): closest threat; benchmarks 15 encoders; no probe-to-planning correlation (agent check of full text via fetch, 2026-10-27).
3. The Planning Limits of Latent World Models (arXiv 2609.39235): per-encoder plannable range (Table 2), usable as free labels.
4. AD-WM (arXiv 2609.30264): action-discriminative training; check whether its "elite regret" diagnostic overlaps this idea.
5. Toso et al. (arXiv 2602.18639): DINO-WM breaks under background and lighting changes.

## Key files
- `writing/proposal-short-2026-10-27.md`: one-page proposal (Google Doc: https://docs.google.com/document/d/1NW4Wbikj8b1M-V1QPiEmjetNig0gY8GXLG0gwbDu3oY/edit)
- `writing/proposal-2026-10-27.html`: full notes (Google Doc: https://docs.google.com/document/d/1okKCxvxoZk5LbEkR6V0REmOmYB5KFuCCF16mA7s65oQ/edit)
- `STANDARDS.md`: topic rules
- `LOG.md`: research log
- `literature/references.bib`: references (checked at arXiv abstract level unless noted)
- Related direction: `../stacking-encoders/` (shares encoder loaders and probe code: `src/extract_features.py`, `src/diagnostics.py`); `../world-model-planning-horizon/` (shares data and encoders)
