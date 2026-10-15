# Stacking Encoders

**Status:** exploring
**Question (draft, not yet confirmed):** Can cheap diagnostics on frozen vision encoders, computed on robot data, predict (a) which encoder combinations improve a World Action Model (WAM) before training, and (b) why general-purpose encoders match or beat specialists?
**What done looks like:** A ranked metric that predicts combo gains better than simple baselines (best-single-encoder, objective-overlap heuristic), evaluated against multi-seed WAM training runs. Or a documented negative result.

## Key findings so far
- None yet. Starting point: liu2026what (arXiv 2610.13024v1) judges complementarity qualitatively and uses one training checkpoint per config. See `literature/notes/liu2026-what-visual-representations-wams.md`.

## Key files
- `STANDARDS.md`: topic rules (not yet filled)
- `LOG.md`: research log
- `decisions/`: decision records
- `literature/`: paper notes and references.bib
