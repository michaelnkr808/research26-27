# Plan: Fast-WAM training-step benchmark

*Reconstructed 2026-11-03 from the Purpose, What runs, and Predictions sections of `README.md`, which were written before the run. Not a new plan.*

**Question:** How fast is one Fast-WAM training step on one GPU, and how much memory does it need? We need this to estimate how long full and reduced-scale WAM runs would take on Nautilus.

**Design:** Fast-WAM at `7faa711`, random weights (only the frozen Wan 2.2 VAE downloaded), synthetic LIBERO-shaped batches. Forward and backward only, micro-batches 1 to 16, gradient checkpointing off and on, stop at the first out-of-memory.

**Predictions (before running):** about 6B trainable parameters; with checkpointing, fits up to batch 8 to 16; about 1 to 1.5 samples/s per GPU.

**Status:** done. Results and the card changes along the way are in `README.md`.
