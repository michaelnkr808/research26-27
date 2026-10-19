# Stacking Encoders

**Status:** exploring. Ground truth will come from a proxy policy (decisions/001). First features cached (libero_goal, SigLIP 2-L + C-RADIOv4). First diagnostics run (SigLIP 2-L vs C-RADIOv4 on libero_goal). The residual-information metric failed on real data. Metric decided: gain(B|A) with demo-level bootstrap CIs; resid dropped. Next: implement CIs and rerun, create the CephFS feature PVC (drafted, not created), DINOv3 once the HF Secret exists (see LOG 2026-10-19 (b)).
**Question (draft, not yet confirmed):** Can cheap diagnostics on frozen vision encoders, computed on robot data, predict (a) which encoder combinations improve a World Action Model (WAM) before training, and (b) why general-purpose encoders match or beat specialists?
**What done looks like:** A ranked metric that predicts combo gains better than simple baselines (best-single-encoder, objective-overlap heuristic), evaluated against multi-seed WAM training runs. Or a documented negative result.

## Key findings so far
- None yet. Starting point: liu2026what (arXiv 2610.13024v1) judges complementarity qualitatively and uses one training checkpoint per config. See `literature/notes/liu2026-what-visual-representations-wams.md`.

## Key files
- `STANDARDS.md`: topic rules (compute budget + Nautilus rules filled; rest TBD)
- `LOG.md`: research log
- `decisions/`: decision records (001: proxy-policy ground truth)
- `literature/`: paper notes and references.bib
- `experiments/2026-10-16-fastwam-step-bench/`: 1-GPU Fast-WAM training-step benchmark. A6000: ~1.3 samples/s/GPU, unreviewed, not a finding.
- `src/bench_fastwam_step.py`: the benchmark script
- `src/extract_features.py`: Phase 1 frozen-feature extraction (LIBERO HDF5 → per-encoder HDF5)
- `experiments/2026-10-18-extract-libero-goal/`: first extraction Job. Features are on PVC `michael-persistent` at `/persistent/stacking-encoders/features/`.
- `src/diagnostics.py`, `src/test_diagnostics.py`: Phase 1 diagnostics + synthetic tests; `experiments/2026-10-19-diagnostics-libero-goal/`: first results (pilot)
- `literature/liu2026what-table{1,10}-*.csv`: paper tables, checked against the arXiv HTML 2026-10-18
