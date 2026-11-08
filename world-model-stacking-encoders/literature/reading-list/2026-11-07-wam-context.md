# Reading list: wam-context (world action models, perturbation robustness, middle-layer features)

This theme covers three things: how world models and world action models (models that predict future video and robot actions together) use or predict encoder features, how well pretrained visual features hold up when a scene changes (including benchmarks that change one thing at a time), and whether middle layers of an encoder carry better information than its output layer. Date: 2026-11-07.

Leads only: nothing here is read or cited yet (_standards/02). Every entry was found by web search this session and its arXiv abstract page was opened to confirm the title. Relevance notes come from arXiv abstracts. Venues are given only where I confirmed them on the page named in the entry.

Touches key: Q1 why a pair of encoders can hurt; Q2 measuring what encoder B adds to A; Q3 what is new vs prior model-selection work; Q4 the fusion-step suspects (feature scale, near-duplicate targets, grid resizing); Q5 the perturbed-frame test; Q6 middle-layer features; Q7 background on world action models.

Seeds not included: VC-1 (arXiv 2303.18240; found and title confirmed in search, left out for space), V-JEPA 2 (arXiv 2506.09985; title confirmed on arXiv, left out for space), Genie Envisioner (not searched, so unverified).

**Start here:** 1 (Perception Encoder, best features are not at the output), 2 (Burns et al., what makes encoders robust under scene shifts), 3 (DINO-WM, a world model whose prediction targets are encoder features).

### 1. Perception Encoder: The best visual embeddings are not at the output of the network
- Authors: Bolya et al. (2025)
- Venue: NeurIPS 2025 (oral); confirmed on https://neurips.cc/virtual/2025/oral/118806
- Link: https://arxiv.org/abs/2504.13181
- Link checked: 2026-11-07, title matches
- Touches: Q6
- Relevance: high
- Why it might matter: The abstract says strong general embeddings are hidden within the intermediate layers of the network, and introduces alignment methods to draw them out. That asks whether using only final-layer features, as the motivating paper does, leaves something useful unused.
- Read status: not read (abstract only)

### 2. What Makes Pre-Trained Visual Representations Successful for Robust Manipulation?
- Authors: Burns et al. (2023)
- Venue: CoRL 2024 (PMLR v270); confirmed on https://proceedings.mlr.press/v270/burns25a.html
- Link: https://arxiv.org/abs/2312.12444
- Link checked: 2026-11-07, title matches
- Touches: Q1 | Q5
- Relevance: high
- Why it might matter: The abstract says representations designed for manipulation do not necessarily generalize under lighting, texture, and distractor changes, and that emergent segmentation ability is a strong predictor of out-of-distribution generalization among ViT models. That is a possible explanation for why one encoder pair holds up under perturbation and another does not.
- Read status: not read (abstract only)

### 3. DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning
- Authors: Zhou et al. (2024)
- Venue: ICML 2025 (PMLR v267); confirmed on the volume index https://proceedings.mlr.press/v267/ (title and authors match)
- Link: https://arxiv.org/abs/2411.04983
- Link checked: 2026-11-07, title matches
- Touches: Q7 | Q3
- Relevance: high
- Why it might matter: The abstract says it models visual dynamics without reconstructing pixels, by predicting future patch features from DINOv2 learned from offline trajectories, and plans by treating goal features as prediction targets. It is an earlier world model whose prediction targets are pretrained-encoder features, like ours.
- Read status: not read (abstract only)

### 4. LIBERO-PRO: Towards Robust and Fair Evaluation of Vision-Language-Action Models Beyond Memorization
- Authors: Zhou et al. (2025)
- Venue: arXiv preprint; no venue found (arXiv, Crossref, OpenReview checked)
- Link: https://arxiv.org/abs/2510.03827
- Link checked: 2026-11-07, title matches
- Touches: Q5
- Relevance: high
- Why it might matter: The abstract says it extends LIBERO with perturbations along four dimensions (manipulated objects, initial states, task instructions, environments) and reports that models above 90% on standard LIBERO collapse under its generalized setting. Worth comparing with how LIBERO-Plus perturbs scenes when we design the perturbed-frame test.
- Read status: not read (abstract only)

### 5. Fast-WAM: Do World Action Models Need Test-time Future Imagination?
- Authors: Yuan et al. (2026)
- Venue: arXiv preprint; no venue found (arXiv, Crossref, OpenReview checked)
- Link: https://arxiv.org/abs/2603.16666
- Link checked: 2026-11-07, title matches
- Touches: Q7 | Q4
- Relevance: high
- Why it might matter: Our proxy model is built on this code. The abstract says removing video co-training during training causes a much larger drop than skipping future imagination at test time, and suggests the main value of video prediction is better world representations in training. That may make the prediction targets a place to look for why a stacked pair goes wrong (our speculation).
- Read status: not read (abstract only)

### 6. Layer by Layer: Uncovering Hidden Representations in Language Models
- Authors: Skean et al. (2025)
- Venue: ICML 2025 (PMLR v267); confirmed on https://proceedings.mlr.press/v267/skean25a.html and the arXiv comment ("update for ICML2025 camera-ready")
- Link: https://arxiv.org/abs/2502.02013
- Link checked: 2026-11-07, title matches
- Touches: Q6 | Q2
- Relevance: medium
- Why it might matter: The abstract says intermediate layers can encode richer representations than the last layer, and proposes layer-quality metrics based on information theory, geometry, and invariance to input perturbations. Those metrics might transfer to comparing encoder layers. It is mostly about language models; the abstract lists vision as one of its domains, so check how much vision it covers.
- Read status: not read (abstract only)

### 7. Back to the Features: DINO as a Foundation for Video World Models
- Authors: Baldassarre et al. (2025)
- Venue: arXiv preprint (venue not checked)
- Link: https://arxiv.org/abs/2507.19468
- Link checked: 2026-11-07, title matches
- Touches: Q7 | Q3
- Relevance: medium
- Why it might matter: The abstract describes DINO-world, a video world model trained to predict future frames in the latent space of DINOv2, which can be fine-tuned on observation-action trajectories and used for planning. Another example of a world model that predicts in encoder-feature space.
- Read status: not read (abstract only)

### 8. Decomposing the Generalization Gap in Imitation Learning for Visual Robotic Manipulation
- Authors: Xie et al. (2023)
- Venue: ICRA 2024; DOI 10.1109/ICRA57147.2024.10611331, confirmed on Crossref (https://api.crossref.org/works/10.1109/ICRA57147.2024.10611331)
- Link: https://arxiv.org/abs/2307.03659
- Link checked: 2026-11-07, title matches
- Touches: Q5
- Relevance: medium
- Why it might matter: The abstract studies single factors and sets of factors (examples named: lighting, camera placement), builds a simulated benchmark of 19 tasks with 11 factors, and finds a factor-difficulty ordering that is consistent between simulation and a real robot. Useful for organizing the perturbed-frame test by factor.
- Read status: not read (abstract only)

### 9. VAT: Vision Action Transformer by Unlocking Full Representation of ViT
- Authors: Li et al. (2025)
- Venue: arXiv preprint; submitted to ICLR 2026 and rejected (OpenReview forum TalHOvvLZu, venue field checked); no accepted venue found
- Link: https://arxiv.org/abs/2512.06013
- Link checked: 2026-11-07, title matches
- Touches: Q6
- Relevance: medium
- Why it might matter: The abstract argues that most robot methods use only the final ViT layer and discard useful information, and processes action tokens with visual features across all transformer layers; it reports results on LIBERO. The abstract does not say whether the ViT is frozen, so check before comparing with our setup.
- Read status: not read (abstract only)

### 10. SpawnNet: Learning Generalizable Visuomotor Skills from Pre-trained Networks
- Authors: Lin et al. (2023)
- Venue: ICRA 2024; DOI 10.1109/ICRA57147.2024.10610356, confirmed on Crossref (https://api.crossref.org/works/10.1109/ICRA57147.2024.10610356)
- Link: https://arxiv.org/abs/2307.03567
- Link checked: 2026-11-07, title matches
- Touches: Q6 | Q5
- Relevance: medium
- Why it might matter: The abstract says it identifies the key bottleneck in using a frozen pretrained backbone for policy learning and fuses multi-layer pretrained representations into a separate network for better categorical generalization. An older example of taking intermediate layers from a frozen encoder.
- Read status: not read (abstract only)

### 11. The Unsurprising Effectiveness of Pre-Trained Vision Models for Control
- Authors: Parisi et al. (2022)
- Venue: ICML 2022 (PMLR v162); confirmed on https://proceedings.mlr.press/v162/parisi22a.html
- Link: https://arxiv.org/abs/2203.03580
- Link checked: 2026-11-07, title matches
- Touches: Q6 | Q3
- Relevance: medium
- Why it might matter: The abstract says it isolates the importance of different representation training methods, data augmentations, and feature hierarchies for control. Early evidence on which encoders and which layers work for control.
- Read status: not read (abstract only)

### 12. From Pixels to Actions - Towards Native Vision-Language-Action Policies
- Authors: Fan et al. (2026)
- Venue: arXiv preprint (v1 2026-11-04); no accepted venue; OpenReview lists a withdrawn ICLR 2027 submission
- Link: https://arxiv.org/abs/2611.06918
- Link checked: 2026-11-07, title matches
- Touches: Q6 | Q5
- Relevance: medium
- Why it might matter: This is an encoder-free VLA, so it is a counterpoint to the frozen-encoder premise rather than evidence about frozen-encoder layers. The abstract says layer-wise probing shows action-critical signals often peak before the final visual representation, and it reports results across distribution shifts. It is only days old, so treat it with extra care.
- Read status: not read (abstract only)

### 13. Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think
- Authors: Yu et al. (2024)
- Venue: ICLR 2025 (Oral); confirmed in the arXiv comment on https://arxiv.org/abs/2410.06940
- Link: https://arxiv.org/abs/2410.06940
- Link checked: 2026-11-07, title matches
- Touches: Q4 | Q7
- Relevance: medium
- Why it might matter: The abstract says REPA aligns the projections of the denoiser's hidden states on noisy inputs with clean-image features from an external pretrained encoder. Encoder features enter as an alignment target, not an input, which is a different route than channel concatenation.
- Read status: not read (abstract only)

### 14. Unified Video Action Model
- Authors: Li et al. (2025)
- Venue: RSS 2025 (RSS XXI); confirmed on https://www.roboticsproceedings.org/rss21/p074.html
- Link: https://arxiv.org/abs/2503.00200
- Link checked: 2026-11-07, title matches
- Touches: Q7
- Relevance: medium
- Why it might matter: The abstract says it jointly optimizes video and action prediction with a joint latent representation and decoupled decoding, so action inference can bypass video generation. Background on the same family of models as Fast-WAM.
- Read status: not read (abstract only)

### 15. Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations
- Authors: Hu et al. (2024)
- Venue: ICML 2025 (PMLR v267); confirmed on https://proceedings.mlr.press/v267/hu25g.html and the arXiv comment ("ICML 2025 Spotlight Paper")
- Link: https://arxiv.org/abs/2412.14803
- Link checked: 2026-11-07, title matches
- Touches: Q7
- Relevance: low
- Why it might matter: The abstract hypothesizes that video-diffusion models produce representations holding both current static information and predicted future dynamics, and conditions an inverse dynamics model on them. It fine-tunes the video model on robot datasets and human manipulation data. Background on using a video model's own features rather than an outside encoder's.
- Read status: not read (abstract only)

### 16. VideoREPA: Learning Physics for Video Generation through Relational Alignment with Foundation Models
- Authors: Zhang et al. (2025)
- Venue: NeurIPS 2025; confirmed on https://neurips.cc/virtual/2025/loc/san-diego/poster/116051
- Link: https://arxiv.org/abs/2505.23656
- Link checked: 2026-11-07, title matches
- Touches: Q4 | Q7
- Relevance: low
- Why it might matter: The abstract says it distills physics understanding from video foundation models into text-to-video models by aligning token-level relations (a loss designed for fine-tuning). A different recipe from channel concatenation, so possibly useful as a contrast.
- Read status: not read (abstract only)
