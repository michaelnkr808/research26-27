# Stacking Encoders

**Status:** exploring. Feasibility tally (decision 002) run on libero_goal: DINOv3 paired with SigLIP 2-L, C-RADIOv4, SAM 3, VGGT-Ω (V-JEPA 2.1, UniTok, HRP skipped). CKA, gain(B|D), gain(D|B) with demo-level CIs: `experiments/2026-10-19-feasibility-tally/`. Awaiting user review; no findings recorded. Next: user reviews the tally, then chooses the hypothesis and whether to run the proxy study (decision 001).
**Goal:** publish (user, 2026-10-19). Venue and timeline to discuss with the PhD mentor on Thursday 2026-10-22.
**Question (draft, not yet confirmed):** Can a cheap, *conditional* measure of what encoder B adds to encoder A (task-relevant information not recoverable from A), computed on frozen features from robot data, predict which encoder combinations improve a World Action Model (WAM) before training, where symmetric similarity (CKA) does not?
- Secondary, possibly a separate paper: why do general-purpose encoders match or beat specialists?
- **Hypothesis not chosen yet.** Candidates and definitions (similarity vs complementarity) are in `writing/hypotheses-draft.md`.
**Mentor feedback (2026-10-19), driving the current plan:**
- Low similarity ≠ complementarity, so define complementarity.
- Fix the hypothesis first.
- Start with 1 dataset and 5–10 encoders, CKA, and a manual tally.
- Position against agostinelli2022 (ensemble transferability metrics).
**Plan (proposed, decision 002):**
1. Feasibility tally, no training: libero_goal, DINOv3 + up to 7 partners. CKA and gain(B|D)/gain(D|B) with CIs, compared against the 7 two-encoder deltas in liu2026what Table 10.
2. Choose the hypothesis.
3. If justified, run the proxy-policy multi-seed study (decision 001).
**What done looks like:** A conditional metric that ranks combo gains better than CKA, single-encoder score, and objective-overlap reasoning, on multi-seed ground truth. It must also explain harmful combos (e.g. D+R). Or a documented negative result.

## Key findings so far
- None yet. Starting point: liu2026what (arXiv 2610.13024v1) judges complementarity qualitatively and uses one training checkpoint per config. See `literature/notes/liu2026-what-visual-representations-wams.md`.

## Key files
- `STANDARDS.md`: topic rules (compute budget + Nautilus rules filled; rest TBD)
- `LOG.md`: research log
- `decisions/`: decision records (001: proxy-policy ground truth; 002 (proposed): feasibility tally before training)
- `literature/`: paper notes and references.bib (incl. agostinelli2022, the closest prior work per the mentor)
- `writing/hypotheses-draft.md`: definitions, candidate hypotheses, the feasibility experiment, a novelty statement draft, and Thursday prep
- `experiments/2026-10-16-fastwam-step-bench/`: 1-GPU Fast-WAM training-step benchmark. A6000: ~1.3 samples/s/GPU, unreviewed, not a finding.
- `src/bench_fastwam_step.py`: the benchmark script
- `src/extract_features.py`: Phase 1 frozen-feature extraction (LIBERO HDF5 → per-encoder HDF5)
- `experiments/2026-10-18-extract-libero-goal/`: first extraction Job. Features are on PVC `michael-persistent` at `/persistent/stacking-encoders/features/`.
- `src/diagnostics.py`, `src/test_diagnostics.py`: Phase 1 diagnostics + synthetic tests; `experiments/2026-10-19-diagnostics-libero-goal/`: first results (pilot)
- `literature/liu2026what-table{1,10}-*.csv`: paper tables, checked against the arXiv HTML 2026-10-18
