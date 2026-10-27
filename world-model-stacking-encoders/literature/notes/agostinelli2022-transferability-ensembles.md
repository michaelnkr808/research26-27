# Transferability Metrics for Selecting Source Model Ensembles (Agostinelli, Uijlings, Mensink, Ferrari, 2022)
- **Link/DOI:** https://arxiv.org/abs/2111.13011 (v2, 2022-03-31). CVPR 2022. Also https://research.google/pubs/transferability-metrics-for-selecting-source-model-ensembles/ (link from the user's PhD mentor, 2026-10-19).
- **Tier:** 1 (CVPR, peer reviewed). Read from the arXiv v2 PDF.
- **Read depth:** read (agent, 2026-10-19): abstract, Sec. 1–4.1, Appendix A and C, via PDF text extraction. Results tables only skimmed (numbers not transcribed). The user has not read it yet.
- **Claim:** Cheap transferability metrics can rank candidate *ensembles* of pre-trained source models for a target task without fine-tuning every ensemble. On semantic segmentation (17 target datasets, averaged), the selected ensemble beats a single source model from the same pool by 6.0% relative mean IoU, and a single larger model of similar capacity by 2.5%.
- **Method (verified, Sec. 3–4):**
  - Pool: 17 source datasets × 2 architectures × 2 pre-training schemes. Per target, 15 candidate source models (5 good + 10 random), ensembles of S = 3, so 455 candidate ensembles.
  - **Ensemble = average of output predictions.** Each member is fine-tuned on the target *separately*; the ensemble prediction is their mean (Sec. 4.1). This is late fusion of outputs, not feature fusion into one model.
  - Metrics extend LEEP, which needs each source model's predicted *label distribution* p_s(z|x) over its own source classes:
    - **MS-LEEP** assumes source models are independent, so it reduces to the sum of single-model LEEP scores (Eq. 4). It has **no interaction term**: it can't express redundancy or complementarity between members.
    - **E-LEEP** scores the averaged ensemble predictor (Eq. 5–6).
    - **IoU-EEP** and **SoftIoU-EEP** approximate mean IoU directly.
  - Evaluation: weighted Kendall τ between metric score and actual fine-tuned performance. Baseline metric: target-agnostic product of source accuracy × dataset size × number of classes.
  - Limitations (App. A): benefits shrink if the source pool isn't diverse. Appendix C: most of the gain comes from combining 3 models.
- **Evidence strength / limitations:** Strong venue, large pool. But one task family (segmentation), and members are fine-tuned rather than frozen. Their "diversity" comes from the source *dataset*, architecture, and pre-training choices, not from a measured feature-level complementarity.
- **Relevance to us (speculative, our analysis; to discuss with the user and mentor):** This is the closest prior work the mentor pointed to. It is the paper our novelty statement has to beat. Candidate differences:
  1. **Fusion type.** They average separately fine-tuned models' outputs. WAM encoder stacking fuses frozen *features* into one policy (channel concatenation in liu2026what). In feature fusion, adding a source can *hurt* (liu2026what Table 10: D+R −5.4), which output averaging of good members rarely does.
  2. **Metric family.** LEEP needs label predictions in a source label space. Vision foundation encoders (DINOv3, SAM 3, VGGT-Ω, V-JEPA 2.1) have no shared classifier head, so LEEP-style metrics don't apply as-is. We need feature-based measures (CKA, conditional probe gain).
  3. **Interactions.** MS-LEEP is additive by construction. Our question is explicitly about interaction: does B add something A lacks (gain(B|A)), and is that asymmetric?
  4. **Target.** Continuous actions and closed-loop success under perturbations (LIBERO-Plus), not per-pixel accuracy.
- **Not yet checked (needed before claiming novelty):** feature-level multi-encoder selection in multimodal LLMs and robotics (e.g. Cambrian-1, Theia, RADIO-style distillation; names recalled, not read). Single-model transferability metrics on features (LogME, H-score, OTCE, GBC) and whether any has a conditional/ensemble variant.
- **Key quotes:** "MS-LEEP can be seen as taking the best source models according to the single-model metric LEEP" (Sec. 3.4, p. 4).
