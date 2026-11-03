# Plan: diagnostics pipeline pilot on libero_goal

*Reconstructed 2026-11-03 from the Purpose, What runs, and Predictions sections of `README.md`, which were written before the run. Not a new plan.*

**Question:** Does the diagnostics pipeline (CKA, action probes, gain, resid) work end to end on real cached features? This is a pipeline check on one pair (SigLIP 2 vs C-RADIOv4), not a test of any hypothesis.

**Design:** `src/diagnostics.py`. Debiased linear CKA per camera; ridge probes from features to the 7-d action, graded on held-out demos (split by demo, never by frame; ridge strength picked on a separate validation set). Synthetic tests in `src/test_diagnostics.py` pass first.

**Predictions (before running):** single-encoder probe R² 0.3 to 0.6, the two encoders close; CKA 0.4 to 0.7; small gains (< 0.05) in both directions.

**Status:** done 2026-10-19. Results in `README.md`.
