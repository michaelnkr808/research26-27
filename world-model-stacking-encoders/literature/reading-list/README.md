# Reading list (literature-review session)

**Owner:** the literature-review session (`literature-agent`, started 2026-11-07, user's request). Only that session edits files in this folder. Other sessions (implementation, planning, slides) may read it but don't edit it. If another session wants a change here, it goes through the user to the literature session. Paper notes (`../notes/`) and `../references.bib` stay shared, as before.

**What this is:** leads for the user to read, not citations. Every paper here was found by a search on 2026-11-07 (not from memory), its link was opened and the title confirmed, and its "why it might matter" note comes from the abstract only. Each theme list was then checked by a separate reviewer against every arXiv page and abstract, and the fixes were applied (direction `LOG.md` 2026-11-07 [agent, literature]). Nobody has read these papers in full. Don't cite anything from this folder until the user has read it and it has a note in `../notes/` and an entry in `../references.bib` (`_standards/02`).

## Files
- `2026-11-07-fusion.md` (18 papers): models that combine several vision encoders, multi-teacher distillation (RADIO family), and why adding an input or a loss can make a jointly trained model worse.
- `2026-11-07-measures.md` (16 papers): ways to compare representations beyond CKA, measures of what one representation adds to another, and choosing models or combinations without training (transferability).
- `2026-11-07-wam-context.md` (16 papers): world models built on pretrained encoder features, robustness of visual features under scene changes, and middle-layer features.

Touches key (used in every list): Q1 why a pair of encoders can hurt; Q2 measuring what encoder B adds to A; Q3 what is new vs prior model-selection work; Q4 the fusion-step suspects (feature scale, near-duplicate targets, grid resizing); Q5 the perturbed-frame test; Q6 middle-layer features; Q7 background on world action models.

## Top 10 (lead's pick, 2026-11-07, from abstracts only)
Ordered by what to read first. The theme file and rank are in brackets.

1. C-RADIOv4 (Tech Report), Ranzinger et al. 2026, https://arxiv.org/abs/2601.17237 [fusion 1]. The encoder in the harmful pair; its page names DINOv3 among its teachers. (Q1)
2. Metric-Guided Feature Fusion of Visual Foundation Models for Segmentation Tasks, Guo et al. 2026, CVPR 2026 Findings per the arXiv comment, https://arxiv.org/abs/2605.16864 [measures 3]. Says naive fusion of foundation models (DINOv3, SAM2) rarely helps and picks what to fuse with label-free metrics. Closest problem statement to ours, for segmentation. (Q1, Q2, Q3)
3. Conditional probing: measuring usable information beyond a baseline, Hewitt et al. 2021, EMNLP 2021 per the arXiv comment, https://arxiv.org/abs/2109.09234 [measures 1]. What a representation adds beyond a baseline: our gain(B|A) idea, in language models. (Q2)
4. Building a Winning Team: Selecting Source Model Ensembles using a Submodular Transferability Estimation Approach, Vimal K B et al. 2023, ICCV 2023, https://arxiv.org/abs/2309.02429 [measures 2]. Follow-up to agostinelli2022 that scores how well models in a group work together. (Q3)
5. Capturing Visual Environment Structure Correlates with Control Performance, Dong et al. 2026, ICLR 2026, https://arxiv.org/abs/2602.04880 [measures 5]. A cheap probe of frozen encoders that predicts robot policy performance; the abstract does not mention combining encoders. (Q3)
6. Encoder Winners Do Not Reliably Transfer Across VLA Backbone Scale: A Frozen-Backbone Grafting Diagnostic, Zeng et al. 2026, arXiv preprint, https://arxiv.org/abs/2606.14153 [fusion 4]. Swaps single frozen encoders into robot policies on LIBERO, scored by offline action error; the winner changes with backbone size. (Q2, Q3)
7. What Makes Training Multi-Modal Classification Networks Hard?, Wang et al. 2019, CVPR 2020, https://arxiv.org/abs/1905.12681 [fusion 3]. A model with more inputs can do worse than its best single input, blamed on inputs overfitting at different speeds. (Q1, Q4)
8. Quantifying & Modeling Multimodal Interactions: An Information Decomposition Framework, Liang et al. 2023, NeurIPS 2023, https://arxiv.org/abs/2302.12247 [measures 4]. Splits what two inputs tell you into shared, unique and only-together parts. (Q1, Q2)
9. What Makes Pre-Trained Visual Representations Successful for Robust Manipulation?, Burns et al. 2023, CoRL 2024, https://arxiv.org/abs/2312.12444 [wam-context 2]. Which pretrained features hold up under lighting, texture and distractor changes. (Q5)
10. Perception Encoder: The best visual embeddings are not at the output of the network, Bolya et al. 2025, NeurIPS 2025, https://arxiv.org/abs/2504.13181 [wam-context 1]. Strong features sit in intermediate layers. (Q6)

Also worth a skim if not read yet: Fast-WAM, https://arxiv.org/abs/2603.16666 [wam-context 5], because our proxy model is built on its code.

## Read so far
None yet. When the user reads a paper from here, the literature session marks it here and writes its note in `../notes/`.
