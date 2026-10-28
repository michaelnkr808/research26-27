# What Visual Representations Do World Action Models Need? (Liu et al., 2026)
- **Link/DOI:** https://arxiv.org/abs/2610.13024 (v1, 2026-10-09)
- **Tier:** 2 (preprint, not peer reviewed)
- **Read depth:** read by the user (confirmed 2026-10-16). The notes below were extracted by an agent on 2026-10-15; the numbers still need a check against the PDF before they go into a writeup.
- **Claim:** Adding frozen external visual features to a WAM (Wan 2.2-based) improves robustness. DINOv3 alone gives most of the gain, and further stacking adds little.
- **Method:** 13 frozen encoders (9 general-purpose, 4 specialized: SAM 3, VGGT-Ω, HRP, DynaFLIP). 5 fusion mechanisms; channel-wise concatenation is the default. Progressive composition (RQ3, App. E). Benchmarks: LIBERO, LIBERO-Plus (7 perturbation categories), RoboCasa GR1, RoboTwin 2.0, real Galbot G1.
- **Numbers as extracted (spot-check against the PDF):** LIBERO-Plus total: VAE-only 74.7, +DINOv3 84.0, R2-WAM (DINOv3+UniTok+V-JEPA 2.1) 87.3, +HRP 87.7. Adding C-RADIOv4 to DINOv3 gives 78.6 (hurts). Adding SAM 3 or VGGT-Ω to DINOv3 gives "little change". SigLIP 2 effects range from -0.4 to +1.8 depending on base. General-purpose avg 82.2 vs specialized 80.0.
- **Checked against the arXiv HTML v1 (2026-10-18, agent, by script):**
  - verified: Table 1 and Table 10 transcriptions match the HTML exactly.
  - verified: the general-purpose vs specialized averages (82.2 vs 80.0) recompute from Table 1.
  - verified, App. C.2: "Results use one trained checkpoint per configuration." The RQ1 and RQ3 SDs are "across three evaluation seeds".
  - verified, App. E.3: Table 10 uses "one fixed trained checkpoint, evaluated with seeds 41, 42, and 43". The SD "reflects evaluation-seed variability for a fixed trained checkpoint".
  - verified, App. C.2: overall LP success = total successes / total episodes (10,030 per seed).
  - verified, App. C.1, Table 7: LIBERO training uses 1,712 episodes, not the full 2,000. Images are 224×224 per view, global batch 128.
  - verified, App. B.1: the **default** formulation is joint video–action denoising. Fast-WAM-style direct prediction is one of three formulations. The 2026-10-16 (b) log entry implied direct prediction was the base model.
  - verified, App. A: "primarily large (L) variants", Qwen3.5-4B, final-layer token features. No exact checkpoints are named.
- **Evidence strength / limitations:** One trained checkpoint per config; the 3 seeds are evaluation seeds only, so training variance is unmeasured. Many composition deltas are under 2 points. Complementarity is explained qualitatively ("overlap in pretraining objectives can limit complementary gains"), with no similarity metric. One backbone. Compute not stated. **No code or checkpoints released** (verified 2026-10-17: the project page https://kailiu18.github.io/R-2WAM.github.io/ links only the paper and the website source; user confirmed).
- **Relevance to us:** Direct motivation. Their composition table is a free (but noisy, small-n) pilot target for any predictive metric.
- **Key quotes:** "The composition results suggest that overlap in pretraining objectives can limit complementary gains." (Sec. 4.4, as extracted; page unchecked)
- **Useful controlled pairs:** VGGT-Ω is DINOv3-initialized (per paper). SAM 3 uses a Perception Encoder backbone (per SAM 3 paper, arXiv 2511.16719, snippet only). HRP's base model is not stated in the paper and needs checking in Srirama et al. 2024.
- **From the project page (verified 2026-10-17):** R²-WAM "predict[s] future features" and integrates them "through shared visual tokens" via channel-wise concatenation. So the encoders may act as prediction targets, not just inputs. Check the PDF method section. Composition values are only in a figure image; the text gives VAE-only 74.7, DINOv3 84.0, SigLIP 2 deltas +1.8 / −0.4 / +0.2, R²-WAM 87.3, +HRP 87.7.

- **Read from the PDF (2026-10-21, LOG (j); verified, pdftotext of the user's copy):**
  - Channel-wise concatenation uses features as clean current inputs *and* noised future targets (Sec. 3.3, App. B.2).
  - Per-source feature losses are summed with weight 1.0 each (App. E.1 Eq. 9). Feature normalization is not stated ("source-specific processing", E.1).
  - Table 9 (DINOv3): current-only 76.5, future-only 81.9, both 84.0.
  - D+R 78.6 is below both D (84.0) and R (83.5) alone.

- **How the paper benchmarks (verified, arXiv HTML v1 text, Sec. 4.1, 4.4, App. C.2, D.1, E.3; read 2026-10-27):**
  - Train once on LIBERO (Fast-WAM's processed data, all four suites, 1,712 episodes). Then the *same* checkpoint is scored three ways, with no retraining:
    1. **LIBERO (clean):** 50 rollouts per task × 40 tasks = 2,000 rollouts, averaged over the four suites. Near saturation (~97–99%), so only in App. D, Table 8.
    2. **LIBERO-Plus (LP):** 10,030 episodes per evaluation seed across 7 perturbation categories (camera, robot initial state, language, light, background, noise, layout), 3 evaluation seeds (41, 42, 43). The perturbation descriptions appended to LP instructions are removed (rewritten instructions kept in the language category). LP is the headline metric.
    3. Other benchmarks (RoboCasa GR1, RoboTwin 2.0, real Galbot G1) for the single-source study and the final recipe only, not the composition study.
  - **Composition (RQ3, Fig. 4, Table 10):** only the LP **total** per combo. No per-category LP numbers and no clean-LIBERO numbers for any combination, including DINOv3 + C-RADIOv4. One trained checkpoint per combo.
  - Single sources do get per-category LP (Table 1) and clean LIBERO (Table 8). C-RADIOv4 alone: LP 83.5 (DINOv3 84.0), clean LIBERO 98.2; per category camera 62.6, robot 81.0, language 92.2, light 97.3, background 86.2, noise 86.6, layout 83.6. So C-RADIOv4 alone is about as good as DINOv3 (better on background and noise); only the pair is bad.
  - Composition eval settings (E.3): 10 inference steps, replan every 10 steps, 30 initial waiting steps, no action ensembling, text guidance 1.0, binarized gripper.
  - **Implication for us:** our diagnostics never run a policy. They compare frozen-feature probe scores on clean libero_goal training demos (one suite of four) against the paper's LP totals. The harm we try to predict is measured under perturbations we never look at.
