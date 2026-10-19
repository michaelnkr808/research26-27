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
