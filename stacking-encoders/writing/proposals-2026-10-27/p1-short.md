# Stacking Encoders in World Action Models
*Michael Ramirez, research proposal, 2026-10-27. Longer version with details: [full notes](https://docs.google.com/document/d/1GgJm4coJ8Kn2J_D0IfIBYZpXSwlVU_3wVz17nUM7MBY/edit).*
## Motivation
World Action Models (WAMs) predict future video and robot actions together. Liu et al. (2026) showed that adding frozen vision-encoder features helps a lot: DINOv3 raises LIBERO-Plus success from 74.7% to 84.0%. But adding a second encoder is unpredictable. Of the paper's 16 "add one more encoder" steps, 4 make results worse. DINOv3 + C-RADIOv4 drops to 78.6%, below either encoder alone, and the paper can only explain this qualitatively. Today the only way to know whether a combination helps is to train the whole model.
## Problem statement
We want a cheap test, run on frozen features before any training, that predicts whether adding encoder B to encoder A will help or hurt a WAM, and explains why the harmful combinations hurt.
## Related works
- **Liu et al. 2026** compare 13 encoders and their combinations in a WAM, but explain the results qualitatively, with one trained model per configuration.
- **KAGES (2026)** picks frozen encoders to fuse by how much each one adds, and shows more encoders can hurt. It covers image classification and retrieval only, with no actions or dynamics.
- **Agostinelli et al. 2022 and OSBORN (2023)** choose model ensembles without training, but for averaged classifier outputs, not features fused into one model.
- **Prismatic VLMs and Cambrian-1** fuse encoders in vision-language models and choose combinations by training each one.
- None of these covers the WAM case, where encoder features are also something the model must predict.
## Method
In Liu et al.'s WAM, each encoder's features are used twice: as an input describing the current frame, and as a target the model must predict for future frames. Most of DINOv3's benefit comes from the prediction role (76.5% input only, 81.9% prediction only, 84.0% both). Our first check tested only the input role, and it failed: neither similarity (CKA) nor "how much B adds to an action probe" flags C-RADIOv4 as harmful.
Our hypothesis is that a partner encoder hurts when it makes the prediction target worse: its future features are hard to predict, or its loss crowds out the model's ability to predict DINOv3's future. We measure this with simple linear models on frozen features:
1. **Loss share:** how much of the combined prediction target belongs to B.
2. **Predictability:** how well B's future features can be predicted.
3. **Interference:** how much worse a small model gets at predicting DINOv3's future when it must also predict B's, with and without rescaling each encoder.
## Experiments
1. **Done:** input-side check on LIBERO (DINOv3 with SigLIP 2, C-RADIOv4, SAM 3, VGGT-Ω). Negative result.
2. **Next:** target-side measures on the same features (CPU only; code written and tested).
3. Extend to all 7 DINOv3 pairs and the paper's 3- and 4-encoder combinations.
4. Train small policies with 3 or more seeds per combination, since the paper's numbers come from one trained model each.
5. If a pair is flagged, retrain it with per-encoder normalization to test the cause.
## Expected results
- **Evals:** rank correlation (Spearman, Kendall τ) between each measure and the real success gains, and how many harmful combinations it catches; policy success on LIBERO and LIBERO-Plus, mean ± std over seeds.
- **What we expect:** the interference measure flags C-RADIOv4 where similarity and input probes did not, and per-encoder normalization recovers part of the drop.
- **If not:** that is still a useful result. Either frozen-feature tests can't predict stacking harm, or the paper's drop is training noise.
## References
- Liu et al. 2026. What Visual Representations Do World Action Models Need? [arXiv 2610.13024](https://arxiv.org/abs/2610.13024)
- Liu and Jiang 2026. When More Foundation Models Means Less (KAGES). [arXiv 2608.17490](https://arxiv.org/abs/2608.17490)
- Agostinelli et al. 2022. Transferability Metrics for Selecting Source Model Ensembles. CVPR. [arXiv 2111.13011](https://arxiv.org/abs/2111.13011)
- K B et al. 2023. Building a Winning Team (OSBORN). ICCV. [arXiv 2309.02429](https://arxiv.org/abs/2309.02429)
- Karamcheti et al. 2024. Prismatic VLMs. ICML. [arXiv 2402.07865](https://arxiv.org/abs/2402.07865)
- Tong et al. 2024. Cambrian-1. NeurIPS. [arXiv 2406.16860](https://arxiv.org/abs/2406.16860)
- Kornblith et al. 2019. Similarity of Neural Network Representations Revisited (CKA). ICML. [arXiv 1905.00414](https://arxiv.org/abs/1905.00414)
