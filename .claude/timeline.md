# research26-27 timeline

## 2026-10-17 — stacking-encoders: direction scoped + Fast-WAM compute benchmark

- Scoped a new direction from Liu et al. 2026 (arXiv 2610.13024): can cheap frozen-encoder diagnostics predict which encoder combos help a World Action Model, and why generalists beat specialists. Plan: real (smaller-scale) WAM runs, not a non-WAM proxy.
- Set the Nautilus compute rule (no GPU-hour cap; ≤8 GPUs / ≤20 Jobs; no sleep in Jobs) as a STANDARDS.md override.
- A 1-GPU Fast-WAM benchmark ran on an A6000 after several attempts were blocked by a saturated cluster: ~1.3 samples/s/GPU, so a full run is ~3–3.5 days on 8× A6000 (estimate). Gradient checkpointing is unsupported in Fast-WAM's compiled path.
- Open decision: training hardware (8× A6000 vs ZeRO-3 on 3090s vs the A100 form).
- See: stacking-encoders/LOG.md, stacking-encoders/experiments/2026-10-16-fastwam-step-bench/README.md

## 2026-10-18 — stacking-encoders: proxy-policy ground truth + Phase 1 extraction script

- Reversed the 10-17 plan: ground truth will come from a small proxy policy trained on cached frozen encoder features and scored in LIBERO rollouts. The first test is whether it reproduces Liu et al.'s orderings. Full Fast-WAM runs (~3 days each) are too expensive for multi-seed grids. Liu et al. released no code or checkpoints.
- Transcribed the paper's Table 1 (single encoders) and Table 10 (16 composition results). Most gains are within ±1 pt; DINOv3+C-RADIOv4 at −5.4 is the outlier.
- Wrote `src/extract_features.py` and smoke-tested it locally on real LIBERO data (SigLIP 2, C-RADIOv4). LIBERO frames need a 180° rotation.
- Blocked on the user: HF gated access (DINOv3, SAM 3, VGGT-Ω) and an `hf-token` k8s Secret. Then a 1-GPU extraction Job, on go-ahead.
- See: stacking-encoders/decisions/001-ground-truth-model.md, stacking-encoders/literature/encoder-weights.md, stacking-encoders/LOG.md

## 2026-10-19 — stacking-encoders: first cached features + first diagnostics (pilot)

- Checked both of Liu et al.'s tables against the arXiv HTML: exact match. My "seed discrepancy" in the catch-up was wrong. The paper uses one trained checkpoint per config; the 3 seeds are evaluation seeds. Also found that the paper's default WAM is joint video–action denoising, not Fast-WAM direct prediction.
- Ran the first GPU extraction Job (TITAN RTX, fp16, 53 min, 90–99% util): SigLIP 2-L + C-RADIOv4 for libero_goal, 127k frames, 9.6 GB on the 20 GB PVC. 4 suites won't fit, so a storage decision is pending (CephFS recommended).
- Wrote `src/diagnostics.py` + synthetic tests and ran it as a CPU Job. Action-probe R²: proprio 0.28, SigLIP 0.61, RADIO 0.67, both 0.68. Gains are asymmetric (+0.069 vs +0.008). CKA 0.72–0.94. The residual-information metric returned 0 on real data (probe overfits; speculative cause: within-demo frame correlation).
- Open for the user: drop resid(B|A) in favor of gain(B|A)? Add demo-level bootstrap CIs; DINOv3 license + HF token Secret.
- See: stacking-encoders/LOG.md (2026-10-18/19 and 2026-10-19), stacking-encoders/experiments/2026-10-18-extract-libero-goal/README.md, stacking-encoders/experiments/2026-10-19-diagnostics-libero-goal/README.md

## 2026-10-19 — stacking-encoders: metric decided (gain + bootstrap CIs), HF access, CephFS drafted

- User decided to drop the residual-information metric and use gain(B|A) with demo-level bootstrap CIs. Implemented and tested in `src/diagnostics.py`. Explained gain as the "unique to B" slice: shared 0.60, RADIO-only 0.07, SigLIP-only 0.01.
- HF access granted for DINOv3, SAM 3, and VGGT-Ω. The user created the `stacking-encoders-hf` Secret.
- Drafted a 100Gi CephFS volume for features, following the NRP docs. Dry run passed; not created.
- Waiting on go-aheads: create the volume, rerun diagnostics with CIs, copy the libero_goal features, run DINOv3 extraction.
- See: stacking-encoders/LOG.md (2026-10-19 (b), (c)), stacking-encoders/data/pvc-features-cephfs.yaml

## 2026-10-21 — stacking-encoders: feasibility tally (decision 002) done, negative on the simple metric

- Extracted DINOv3, SAM 3, VGGT-Ω for libero_goal on Nautilus (~10 GPU-h over 3 tries; fp16 DINOv3 overflowed, try 2 over-requested memory, SAM 3 ran 5.5 frames/s) and ran CPU diagnostics for D+L, D+R, D+S, D+G.
- Result (n = 4, user signed off): neither CKA nor gain(B|D) flags the paper's harmful D+R. R is the most similar to D and adds the most to it. Every partner adds a small positive gain on frozen features.
- Decision 002 accepted; CephFS feature PVC created (17.6G of features kept). All Jobs and ConfigMaps cleaned up. NOTES.md is now git-ignored. A handoff block went to the planning agent; direction to be chosen with the mentor on 2026-10-22.
- See: stacking-encoders/experiments/2026-10-19-feasibility-tally/README.md, stacking-encoders/README.md (key findings), stacking-encoders/LOG.md (2026-10-19 (e) through 2026-10-21 (h))
