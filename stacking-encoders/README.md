# Stacking Encoders

**Status:** exploring. The compute-feasibility check is done, and choosing the training hardware is next (see LOG 2026-10-17).
**Question (draft, not yet confirmed):** Can cheap diagnostics on frozen vision encoders, computed on robot data, predict (a) which encoder combinations improve a World Action Model (WAM) before training, and (b) why general-purpose encoders match or beat specialists?
**What done looks like:** A ranked metric that predicts combo gains better than simple baselines (best-single-encoder, objective-overlap heuristic), evaluated against multi-seed WAM training runs. Or a documented negative result.

## Key findings so far
- None yet. Starting point: liu2026what (arXiv 2610.13024v1) judges complementarity qualitatively and uses one training checkpoint per config. See `literature/notes/liu2026-what-visual-representations-wams.md`.

## Key files
- `STANDARDS.md`: topic rules (compute budget + Nautilus rules filled; rest TBD)
- `LOG.md`: research log
- `decisions/`: decision records
- `literature/`: paper notes and references.bib
- `experiments/2026-10-16-fastwam-step-bench/`: 1-GPU Fast-WAM training-step benchmark. A6000: ~1.3 samples/s/GPU, unreviewed, not a finding.
- `src/bench_fastwam_step.py`: the benchmark script
