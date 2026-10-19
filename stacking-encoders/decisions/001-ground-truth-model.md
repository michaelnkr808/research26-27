# 001 — Ground truth from a proxy policy on cached frozen features
- **Date:** 2026-10-17
- **Status:** accepted (user, 2026-10-17)
- **Context:** The research question needs measured combo gains (success with A+B minus success with A), with several training seeds. One full Fast-WAM run is ~3–3.5 days on 8× A6000 (speculative, from the 1-GPU bench in `experiments/2026-10-16-fastwam-step-bench/`), and A6000 nodes are scarce. liu2026what released no code or checkpoints (verified, project page checked 2026-10-17), so we can't reuse their model.
- **Options considered:**
  1. Full Fast-WAM, multi-seed. Not feasible at grid scale.
  2. Reduced Fast-WAM (fewer layers or steps, cached VAE latents). Closer to the paper, still multi-GPU and costly per seed.
  3. Small proxy policy trained on cached frozen encoder features, scored by LIBERO simulator rollouts. Cheap enough for 3–5 seeds per config (speculative until a pilot is timed).
- **Decision & rationale:** Option 3. First test: can the proxy reproduce liu2026what's published orderings (e.g. DINOv3 >> VAE-only; adding C-RADIOv4 to DINOv3 hurts; adding SAM 3 or VGGT-Ω changes little)? Agreement is our evidence that proxy gains track WAM gains. Option 2 is kept in reserve for a few calibration runs.
- **Consequences:** Claims are about the proxy unless it agrees with the paper. Their integration differs from ours: per their project page, R²-WAM *predicts future encoder features* via shared visual tokens with channel-wise concatenation. A proxy that only takes features as input tests a different use of the encoder. Disagreement with the paper is ambiguous (proxy mismatch vs. their single-seed noise). Before Phase 1 is interpreted, their table needs a human check against the PDF.
- **Note 2026-10-19:** Sequencing changed by decision 002 (proposed). A no-training feasibility tally against liu2026what Table 10 comes first; proxy training follows only if that tally and the chosen hypothesis justify it. The ground-truth choice itself is unchanged.
