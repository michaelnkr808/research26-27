# Plan: feature extraction pilot on libero_goal

*Reconstructed 2026-11-03 from the Purpose, What runs, and Predictions sections of `README.md`, which were written before the run. Not a new plan.*

**Question:** Does the frozen-feature extraction pipeline work on a cluster GPU, and how fast and how large is it per encoder? The answer sizes the full extraction, and the features feed the first diagnostics.

**Design:** `src/extract_features.py` on LIBERO libero_goal (10 task files), SigLIP 2 and C-RADIOv4 (DINOv3 waited on the HF token), both cameras, every frame, fp16. Checked locally on one demo first.

**Predictions (before running):** about 89k frames, about 7 GB on disk, 100 to 300 frames/s per encoder.

**Status:** done 2026-10-19. Results in `README.md`.
