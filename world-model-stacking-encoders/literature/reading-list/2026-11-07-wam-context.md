# Reading list: wam-context (world action models, perturbation robustness, middle-layer features)

This theme covers three things: how world models and world action models (models that predict future video and robot actions together) use or predict frozen vision-encoder features, how well pretrained visual features hold up when a scene changes (including benchmarks that change one thing at a time), and whether middle layers of an encoder carry better information than its output layer. Date: 2026-11-07.

Leads only: nothing here is read or cited yet (_standards/02). Every entry was found by web search this session and its arXiv abstract page was opened to confirm the title. Relevance notes come from abstracts or search summaries only. Seeds not included: VC-1 (arXiv 2303.18240; found and title confirmed in search, left out for space), V-JEPA 2 (arXiv 2506.09985; found and title confirmed on arXiv, left out for space), Genie Envisioner (not searched, so unverified).

**Start here:** 1 (From Pixels to Actions, middle-layer probing under shifts), 2 (Perception Encoder, best features are not at the output), 3 (LIBERO-PRO, one-factor perturbation benchmark).

### 1. From Pixels to Actions - Towards Native Vision-Language-Action Policies
- Authors: Fan et al. (2026)
- Venue: arXiv preprint, not peer reviewed (venue not checked)
- Link: https://arxiv.org/abs/2611.06918
- Link checked: 2026-11-07, title matches
- Touches: Q6 | Q5
- Relevance: high
- Why it might matter: Its abstract says layer-wise probing found that action-useful visual signals often peak well before the final layer, and it tests under distribution shifts. This is close to the mentor's middle-layer point. It was submitted 2026-11-04 and has no separate frozen encoder, so it differs from our setup.
- Read status: not read (abstract only)

### 2. Perception Encoder: The best visual embeddings are not at the output of the network
- Authors: Bolya et al. (2025)
- Venue: NeurIPS 2025 (oral); seen on the neurips.cc / nips.cc listing pages in web search results
- Link: https://arxiv.org/abs/2504.13181
- Link checked: 2026-11-07, title matches
- Touches: Q6
- Relevance: high
- Why it might matter: The title and abstract say the best features sit in middle layers of the encoder, not at the output. That asks whether using only final-layer features, as the motivating paper does, leaves something useful unused.
- Read status: not read (abstract only)

### 3. LIBERO-PRO: Towards Robust and Fair Evaluation of Vision-Language-Action Models Beyond Memorization
- Authors: Zhou et al. (2025)
- Venue: arXiv preprint, not peer reviewed (venue not checked)
- Link: https://arxiv.org/abs/2510.03827
- Link checked: 2026-11-07, title matches
- Touches: Q5
- Relevance: high
- Why it might matter: The abstract says it extends LIBERO with controlled perturbations (objects, initial states, instructions, environments) and that strong LIBERO models fail under them. Worth comparing with how LIBERO-Plus perturbs scenes when we design the perturbed-frame test.
- Read status: not read (abstract only)

### 4. What Makes Pre-Trained Visual Representations Successful for Robust Manipulation?
- Authors: Burns et al. (2023)
- Venue: CoRL 2024 (PMLR v270); seen on the proceedings.mlr.press page in web search results
- Link: https://arxiv.org/abs/2312.12444
- Link checked: 2026-11-07, title matches
- Touches: Q1 | Q5
- Relevance: high
- Why it might matter: The abstract reports that models built for manipulation did not generalize better under lighting, texture, and distractor shifts, and that emergent segmentation ability predicted robustness. That is a possible explanation for why one encoder pair holds up under perturbation and another does not.
- Read status: not read (abstract only)

### 5. Layer by Layer: Uncovering Hidden Representations in Language Models
- Authors: Skean et al. (2025)
- Venue: ICML 2025 (PMLR v267); seen on the proceedings.mlr.press page in web search results
- Link: https://arxiv.org/abs/2502.02013
- Link checked: 2026-11-07, title matches
- Touches: Q6 | Q2
- Relevance: medium
- Why it might matter: According to search summaries, it finds middle layers often beat the last layer and proposes ways to score layer quality, including robustness to input perturbations. Those scores might transfer to comparing encoder layers. It is mostly about language models, so check how much vision it covers.
- Read status: not read (abstract only)

### 6. Fast-WAM: Do World Action Models Need Test-time Future Imagination?
- Authors: Yuan et al. (2026)
- Venue: arXiv preprint, not peer reviewed (venue not checked)
- Link: https://arxiv.org/abs/2603.16666
- Link checked: 2026-11-07, title matches
- Touches: Q7 | Q4
- Relevance: high
- Why it might matter: Our proxy model is built on this code. According to search summaries of the abstract, video prediction during training matters much more than imagining the future at test time, which bears on how the feature-prediction targets could be where a stacked pair goes wrong.
- Read status: not read (abstract only)

### 7. DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning
- Authors: Zhou et al. (2024)
- Venue: arXiv preprint, not peer reviewed (venue not checked)
- Link: https://arxiv.org/abs/2411.04983
- Link checked: 2026-11-07, title matches
- Touches: Q7 | Q3
- Relevance: medium
- Why it might matter: By its title, a world model built on frozen pretrained (DINO) features rather than pixels. A likely earlier example of frozen-encoder features as the thing a world model works in, which is related to the motivating paper's setup. I only confirmed the title.
- Read status: not read (title only)

### 8. Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think
- Authors: Yu et al. (2025)
- Venue: ICLR 2025; seen on the proceedings.iclr.cc page in web search results
- Link: https://arxiv.org/abs/2410.06940
- Link checked: 2026-11-07, title matches
- Touches: Q4 | Q7
- Relevance: medium
- Why it might matter: REPA (per search summaries) pulls a generative model's hidden states toward a frozen encoder's features, mostly in the first few blocks. It is another way foundation-encoder features enter a generative model, and the choice of where to align may hint at which fusion steps are fragile.
- Read status: not read (abstract only)

### 9. Decomposing the Generalization Gap in Imitation Learning for Visual Robotic Manipulation
- Authors: Xie et al. (2023)
- Venue: arXiv preprint, not peer reviewed (venue not checked). A search summary said ICRA 2024, but that came from a secondary page and I did not confirm it on the paper or a venue site.
- Link: https://arxiv.org/abs/2307.03659
- Link checked: 2026-11-07, title matches
- Touches: Q5
- Relevance: medium
- Why it might matter: According to search summaries, it builds a benchmark that varies one factor at a time (lighting, camera position, background, and others). A good model for organizing the perturbed-frame test one factor at a time.
- Read status: not read (search summary only)

### 10. VAT: Vision Action Transformer by Unlocking Full Representation of ViT
- Authors: Li et al. (2025)
- Venue: arXiv preprint, not peer reviewed (venue not checked)
- Link: https://arxiv.org/abs/2512.06013
- Link checked: 2026-11-07, title matches
- Touches: Q6
- Relevance: medium
- Why it might matter: The abstract argues that most robot methods use only the final ViT layer and discard useful information, and it uses features from all layers. It reports results on LIBERO, so it is a robot-learning example of the middle-layer idea.
- Read status: not read (abstract only)

### 11. SpawnNet: Learning Generalizable Visuomotor Skills from Pre-trained Networks
- Authors: Lin et al. (2023)
- Venue: arXiv preprint, not peer reviewed (venue not checked)
- Link: https://arxiv.org/abs/2307.03567
- Link checked: 2026-11-07, title matches
- Touches: Q6 | Q5
- Relevance: medium
- Why it might matter: The abstract says it fuses multi-layer features from a frozen pretrained backbone into a separate policy network for better generalization. An older, simple example of taking intermediate layers from a frozen encoder.
- Read status: not read (abstract only)

### 12. The Unsurprising Effectiveness of Pre-Trained Vision Models for Control
- Authors: Parisi et al. (2022)
- Venue: ICML 2022 (PMLR v162); seen on the proceedings.mlr.press page in web search results
- Link: https://arxiv.org/abs/2203.03580
- Link checked: 2026-11-07, title matches
- Touches: Q6 | Q3
- Relevance: medium
- Why it might matter: The abstract says it studies training methods, augmentations, and feature hierarchies for control with pretrained vision models. Early evidence on which encoders and which layers work for control.
- Read status: not read (abstract only)

### 13. Unified Video Action Model
- Authors: Li et al. (2025)
- Venue: RSS 2025; seen on the roboticsproceedings.org page in web search results
- Link: https://arxiv.org/abs/2503.00200
- Link checked: 2026-11-07, title matches
- Touches: Q7
- Relevance: medium
- Why it might matter: Per search summaries, it learns video and action jointly and can skip video generation at inference. Background on the same family of models as Fast-WAM.
- Read status: not read (search summary only)

### 14. Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations
- Authors: Hu et al. (2024)
- Venue: ICML 2025 (PMLR v267); seen on the proceedings.mlr.press page in web search results
- Link: https://arxiv.org/abs/2412.14803
- Link checked: 2026-11-07, title matches
- Touches: Q7
- Relevance: low
- Why it might matter: Per search summaries, it feeds a policy from a video diffusion model's internal features, which encode the current scene and predicted motion. Background on using a video model's own features rather than an outside encoder's.
- Read status: not read (search summary only)

### 15. VideoREPA: Learning Physics for Video Generation through Relational Alignment with Foundation Models
- Authors: Zhang et al. (2025)
- Venue: NeurIPS 2025; seen on the neurips.cc poster page in web search results
- Link: https://arxiv.org/abs/2505.23656
- Link checked: 2026-11-07, title matches
- Touches: Q4 | Q7
- Relevance: low
- Why it might matter: Per search summaries, it aligns a video generator to video foundation-model features by matching relations between tokens. A different recipe from channel concatenation, so possibly useful as a contrast.
- Read status: not read (search summary only)
