# Reading list: prior art for a "motion fidelity" score (does a frozen encoder's feature change follow scene motion?)

Date: 2026-11-08. Scope: a novelty search for a proposed label-free score, modelled on the edge fidelity (EF) score of Guo et al. 2026 (guo2026metric, entry 3 of 2026-11-07-measures.md). For a frozen encoder, take consecutive frames of robot demos and ask whether the places where the feature map changes line up with where the scene moves (pixel change, optical flow, gripper or object motion), with no action labels. Long-term use: predict which encoders, or pairs, help a world action model before training. Nothing here is cited yet (_standards/02).

How entries were found and checked: web search (queries listed at the bottom), then each paper's arXiv API record (export.arxiv.org) was fetched on 2026-11-08 and its exact title confirmed. Read depth is stated per entry. Important limit: "skimmed" and "read" mean I read the method and evaluation sections only through the fetch tool, which returns a summary written by a small model, not the raw text; treat these as provisional until someone opens the paper. "Abstract only" means the arXiv abstract (or a search summary of it). Overlap labels are my judgement. Absence claims ("no paper does X") mean only that these searches found none (speculative); they do not prove none exists. Two of the lead's guesses were corrected: arXiv 2507.19507 is an unrelated physics thesis (Baldassarre et al. is 2507.19468), and TAP-Vid was seen only in search results, not opened, so it has no entry.

## Verdict

(a) Has anyone defined this score? Not that these searches found (speculative: absence of evidence). Several works sit next to it. Deep Feature Flow (2016) treats feature maps as moving along optical flow, and Motion-Focused Contrastive Learning (2021) aligns feature-map gradients with optical flow, but both use that as part of training one model, not as a score for comparing frozen encoders. FlowFeat (2025) maps frozen features to flow, again as training. Point-tracking probes (Aydemir et al. 2024) score frozen features over time, but need labelled tracks and a trained readout.

(b) Has anyone used a motion or dynamics property of frozen features to predict robot or world-model performance before training? Not found. The closest are Dong et al. 2026 (probe accuracy on simulator state labels from single images predicts policy success) and Burns et al. 2023 (emergent segmentation predicts out-of-distribution success). Both use static images and score one encoder at a time. Neither touches time or motion.

(c) Has anyone used a per-encoder or pair-level score to choose combinations? Partly. KAGES (Liu 2026) picks subsets of frozen encoders without training a classifier, but by kernel alignment with labels, on image classification, not for robots and not with a motion property. The OSBORN paper in the 2026-11-07 list is similar for ensembles of transfer models. Nothing found did this for robot world models.

Closest single work: CoRe-WAM (Zhou et al. 2026). It is the only one found that puts feature change between frames, matched to where content moved, inside a world action model. It is not a score, though, and it warns that a same-grid-cell difference mixes feature change with content moving. For the "score" side the closest is Dong et al. 2026 (cheap frozen-encoder predictor of control success).

What would still be new for us: a label-free, temporal score of frozen encoders, defined by comparing feature change to motion (ideally along flow, to avoid the same-cell confound CoRe-WAM points out), checked against world-model or policy outcomes, and extended to say something about pairs. Pieces of this exist in isolation; the combination, and the pair angle in particular, did not turn up. A caution: Dong et al. already show cheap proxies can be rank-correlated with policy success, so the claim that matters is that ours adds something their proxy and the segmentation proxy do not.

## Entries

Field "Overlap with our idea": same (it is the idea) / partial / adjacent (neighbouring idea only). No entry is "same".

### 1. CoRe-WAM: Correspondence-Aligned Temporal Residuals for World Action Models
- Authors: Zhou et al. (2026)
- Venue: arXiv 2609.27314v2, no venue note on the record
- Link: https://arxiv.org/abs/2609.27314
- Link checked: 2026-11-08 title matches
- Relevance: High
- Overlap with our idea: partial (closest in spirit, but a model component, not a score)
- Why it might matter: A world action model that takes the difference between current and past frame features, after matching each place to where it moved (a frozen Trace Anything model supplies the matching). Its argument is that a same-grid-cell difference mixes feature change with a different object moving into that cell. The paper says its difference "measures feature change after this association, not displacement itself", and reports no score of how well feature change tracks motion, no pre-training predictor and no encoder selection. It shows that feature change over time is already a design question in WAMs, and warns that a naive same-cell difference is confounded by motion; our score would need to handle that (compare along flow, not per fixed cell).
- Read status: Read (method and evaluation sections, through the fetch tool's summary of the page; a small-model summary, not my own full read)

### 2. Capturing Visual Environment Structure Correlates with Control Performance
- Authors: Dong et al. (2026)
- Venue: ICLR 2026 poster (per 2026-11-07 list); arXiv record has no venue note
- Link: https://arxiv.org/abs/2602.04880
- Link checked: 2026-11-08 title matches
- Relevance: High
- Overlap with our idea: adjacent for (b); no overlap for (a) and (c)
- Why it might matter: Probes nine frozen encoders by regressing simulator state (object pose, shape, material, lighting, joint angles, end-effector pose) from single images, then rank-correlates probe accuracy with diffusion-policy success on MetaWorld, RoboCasa and SimplerEnv. The state vector is static, the prediction is from one image, each encoder is scored alone, and it needs simulator state labels. So it is the closest pre-training predictor of robot performance, but it uses labels, no time, and no pairs. Our score would differ by being label-free, temporal and (if extended) pair-level. It is simulation-only for the predictor, so the real-robot scope is unchecked.
- Read status: Skimmed (method and evaluation via fetch tool summary; last 35k characters of the page, likely appendix, not read)

### 3. What Makes Pre-Trained Visual Representations Successful for Robust Manipulation?
- Authors: Burns et al. (2023)
- Venue: CoRL 2024 (PMLR v270, stated on the proceedings page seen in search; the arXiv record has only a page-count comment)
- Link: https://arxiv.org/abs/2312.12444
- Link checked: 2026-11-08 title matches
- Relevance: High
- Overlap with our idea: adjacent for (b)
- Why it might matter: Finds that emergent segmentation (Jaccard of the CLS attention map on PASCAL VOC, ViT models only) predicts out-of-distribution manipulation success better than ImageNet accuracy, in-distribution success or shape bias, with no training needed. All four of its metrics are static-image or policy-success metrics; none uses video, motion or flow, and it does not combine encoders. It is the template for "cheap frozen-feature property predicts robot success", which is the framing our score would be tested in. It is a property of the encoder alone, and covers only ViTs.
- Read status: Skimmed (metric definitions via fetch tool summary)

### 4. When More Foundation Models Means Less: Diagnosing and Addressing Multi-View Fusion Failure
- Authors: Liu and one co-author (2026)
- Venue: arXiv 2608.17490v1, no venue note on the record
- Link: https://arxiv.org/abs/2608.17490
- Link checked: 2026-11-08 title matches
- Relevance: High
- Overlap with our idea: adjacent for (c); closest on selecting encoders
- Why it might matter: Proposes KAGES, which greedily picks a subset of frozen encoders (from a pool of 10, including CLIP, SigLIP, DINOv2, MAE, SAM) by kernel alignment (a CKA-style score) between the summed encoder kernels and the label kernel, with no downstream training. It is training-free but uses labels, is defined on whole subsets rather than pairs, runs on image classification and retrieval, and mentions neither robotics nor time. It is the nearest thing to (c), and its argument that fusing more encoders can hurt matches our harmful pair. Ours would differ by using a motion property, not label alignment, and by targeting robot world models.
- Read status: Skimmed (method and datasets via fetch tool summary, first 100k of 165k characters)

### 5. FlowFeat: Pixel-Dense Embedding of Motion Profiles
- Authors: Araslanov et al. (2025)
- Venue: arXiv 2511.07696v1, no venue note on the record
- Link: https://arxiv.org/abs/2511.07696
- Link checked: 2026-11-08 title matches
- Relevance: Medium
- Overlap with our idea: adjacent for (a)
- Why it might matter: Trains a small decoder on top of a frozen encoder so that its dense features can be linearly turned into optical flow (flow from SEA-RAFT is the teacher). At test time it takes a single image. It evaluates on DAVIS video object segmentation, COCO-Stuff and NYUv2 depth. Flow is a training signal here, not a measuring stick for a frozen encoder, and the reading I did found no score of feature change against flow. It does show that a linear map from frozen features to flow is workable, which hints at a possible score variant (how well a linear map predicts flow from frozen features). That variant is our speculation.
- Read status: Skimmed (method and evaluation via fetch tool summary, first 100k of 108k characters)

### 6. Motion-Focused Contrastive Learning of Video Representations
- Authors: Li et al. (2021)
- Venue: ICCV 2021 Oral (stated in the arXiv comment)
- Link: https://arxiv.org/abs/2201.04029
- Link checked: 2026-11-08 title matches
- Relevance: Medium
- Overlap with our idea: adjacent for (a)
- Why it might matter: A training method for video encoders. Optical flow picks the clips to sample, and a loss aligns the feature map's gradient maps with optical flow so the backbone attends to high-motion places. This is the closest older thing to "feature-map change lines up with flow", but it is a loss used while training one encoder, not a score for comparing frozen encoders, and it has no robot or pair component. Useful as a source for how to define the alignment.
- Read status: Abstract only (arXiv abstract paraphrase and a search summary of the method)

### 7. Deep Feature Flow for Video Recognition
- Authors: Zhu et al. (2016)
- Venue: CVPR 2017 (recalled; the arXiv record has no venue note)
- Link: https://arxiv.org/abs/1611.07715
- Link checked: 2026-11-08 title matches
- Relevance: Low
- Overlap with our idea: adjacent for (a)
- Why it might matter: Computes features on key frames and moves them to other frames by warping with optical flow, on the assumption that deep features change smoothly along flow. It is the classic statement that feature maps can be carried along flow. It does not score how well a given encoder obeys that, but "how well do features warped along flow match the next frame's features" is a natural reading of our idea, and this is its origin. The warping step is something we could reuse.
- Read status: Abstract only (search summary)

### 8. What, Where, and How: Probing Spatiotemporal Representations in Video Foundation Models
- Authors: Musa et al. (2026)
- Venue: arXiv 2609.01551v1, no venue note on the record
- Link: https://arxiv.org/abs/2609.01551
- Link checked: 2026-11-08 title matches
- Relevance: Medium
- Overlap with our idea: adjacent for (a)
- Why it might matter: Probes layers of V-JEPA 2 and VideoMAE-v2 for camera motion, intuitive physics and anomaly detection. Camera motion is readable at about 60 to 70 percent of the network depth, and temporal features form smooth curves. These are video models, the probes are trained with labels, and nothing here compares feature change with flow or predicts robot performance. It is a hint that the middle layers carry motion, relevant to our layer choice (Q6).
- Read status: Abstract only

### 9. Can Visual Foundation Models Achieve Long-term Point Tracking?
- Authors: Aydemir et al. (2024)
- Venue: ECCV 2024 EVAL-FoMo workshop (stated in the arXiv comment)
- Link: https://arxiv.org/abs/2408.13575
- Link checked: 2026-11-08 title matches
- Relevance: Medium
- Overlap with our idea: adjacent for (a)
- Why it might matter: Tests frozen image foundation models (DINOv2, Stable Diffusion and others) on point tracking: zero-shot, with a light probe, and with LoRA. Stable Diffusion and DINOv2 features track well. This is the standard way the tracking field scores frozen features over time, but it needs labelled point tracks (or a trained readout) and ends in tracking accuracy, not a feature-change-against-motion score. It is the likely reference point a reviewer would raise against us, and a possible validity check: does our score rank encoders the way tracking accuracy does?
- Read status: Abstract only

### 10. Back to the Features: DINO as a Foundation for Video World Models
- Authors: Baldassarre et al. (2025)
- Venue: arXiv 2507.19468v1 (an earlier version was at an ICML 2025 workshop, per a search summary)
- Link: https://arxiv.org/abs/2507.19468
- Link checked: 2026-11-08 title matches
- Relevance: Medium
- Overlap with our idea: adjacent for (b)
- Why it might matter: DINO-world predicts future patch features of a frozen DINOv2 and is trained on large uncurated video. It reports forecasting quality and some action-conditioned planning, but I found no pre-training score of the encoder's motion behaviour, and the encoder is a single fixed choice. It supports the premise that features of frozen DINO can be predicted, which is the setting our score is meant to serve.
- Read status: Abstract only (search summary)

### 11. DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning
- Authors: Zhou et al. (2024)
- Venue: arXiv 2411.04983v2, no venue note on the record
- Link: https://arxiv.org/abs/2411.04983
- Link checked: 2026-11-08 title matches
- Relevance: Low
- Overlap with our idea: adjacent for (b)
- Why it might matter: A world model on frozen DINOv2 patch features used for planning. I did not see a pre-training predictor in what I read of it. Included because it was a lead and is the standard example of a frozen-feature world model.
- Read status: Abstract only (search summary)

### 12. Rethinking Representations for World-Action Modeling
- Authors: Jiang et al. (2026)
- Venue: arXiv 2609.38163v1, no venue note on the record
- Link: https://arxiv.org/abs/2609.38163
- Link checked: 2026-11-08 title matches
- Relevance: Medium
- Overlap with our idea: adjacent for (b)
- Why it might matter: Argues that reconstruction quality is a poor guide to which representation helps a world action model, comparing Video-VAE latents with DINOv3 features and a bottleneck trained with action gradients. Its motion-related analysis is on its own trained representation, not frozen encoders, and it uses no pre-training score. It is direct evidence that a cheap quality number can mislead for WAMs, and that a WAM on DINOv3 features is a live design.
- Read status: Skimmed (method and evaluation via fetch tool summary)

### 13. For Pre-Trained Vision Models in Motor Control, Not All Policy Learning Methods are Created Equal
- Authors: Hu et al. (2023)
- Venue: arXiv 2304.04591v2, no venue note on the record (ICML 2023, recalled)
- Link: https://arxiv.org/abs/2304.04591
- Link checked: 2026-11-08 title matches
- Relevance: Low
- Overlap with our idea: adjacent for (b)
- Why it might matter: Compares 14 pre-trained vision models under three policy-learning methods and finds the ranking depends on the method. A search summary says it found a link between linear-probe loss and success on one benchmark (I did not open the number). A caution for us: a predictor might rank encoders differently depending on the policy type.
- Read status: Abstract only

### 14. What do we learn from a large-scale study of pre-trained visual representations in sim and real environments?
- Authors: Silwal et al. (2023)
- Venue: arXiv 2310.02219v2, no venue note on the record
- Link: https://arxiv.org/abs/2310.02219
- Link checked: 2026-11-08 title matches
- Relevance: Low
- Overlap with our idea: adjacent for (b)
- Why it might matter: Shows simulation trends in pre-trained visual representations mostly carry over to the real robot. It uses sim performance as the proxy, so it is the "train then see" baseline, with no motion or label-free score.
- Read status: Abstract only

### 15. Where are we in the search for an Artificial Visual Cortex for Embodied Intelligence?
- Authors: Majumdar et al. (2023)
- Venue: arXiv 2303.18240v2, no venue note on the record (NeurIPS 2023, recalled)
- Link: https://arxiv.org/abs/2303.18240
- Link checked: 2026-11-08 title matches
- Relevance: Low
- Overlap with our idea: adjacent for (b)
- Why it might matter: Introduces CortexBench and VC-1 (masked autoencoding on egocentric video). No single representation wins everywhere. Video is used for pre-training only; motion is not measured and no pre-training predictor is offered.
- Read status: Abstract only

### 16. Probing the 3D Awareness of Visual Foundation Models
- Authors: El Banani et al. (2024)
- Venue: CVPR 2024 (stated in the arXiv comment)
- Link: https://arxiv.org/abs/2404.08636
- Link checked: 2026-11-08 title matches
- Relevance: Low
- Overlap with our idea: adjacent for (a)
- Why it might matter: Probes frozen features for 3D structure and for consistency of surfaces across viewpoints. Cross-view consistency is the nearest static analogue of our time-consistency idea. No motion, flow or policy performance.
- Read status: Abstract only

### 17. SPA: 3D Spatial-Awareness Enables Effective Embodied Representation
- Authors: Zhu et al. (2024)
- Venue: arXiv 2410.08208v3, no venue note on the record (ICLR 2025, recalled)
- Link: https://arxiv.org/abs/2410.08208
- Link checked: 2026-11-08 title matches
- Relevance: Low
- Overlap with our idea: adjacent for (b)
- Why it might matter: Trains a ViT with neural rendering on multi-view images and evaluates on 268 embodied tasks in eight simulators. A large benchmark that could serve as an outcome measure for validating a score, but it offers no pre-training predictor or motion property.
- Read status: Abstract only

### 18. Intuitive physics understanding emerges from self-supervised pretraining on natural videos
- Authors: Garrido et al. (2025)
- Venue: arXiv 2502.11831v1, no venue note on the record
- Link: https://arxiv.org/abs/2502.11831
- Link checked: 2026-11-08 title matches
- Relevance: Low
- Overlap with our idea: adjacent for (a)
- Why it might matter: Uses a violation-of-expectation test: models that predict in a learned representation space notice impossible events, while pixel-space predictors and multimodal language models stay near chance. A physical-understanding probe of video models via prediction surprise, not feature change against flow, and not tied to robot performance. Background for the claim that prediction in feature space captures physics.
- Read status: Abstract only

## Search terms

Web search (each run once, results scanned by title and summary):
- label-free score frozen visual encoder feature change between frames aligns with optical flow motion (nothing on target; found FlowFeat and flow-from-DINOv2 papers)
- predict robot policy performance from frozen visual representation properties without training (found Dong 2026)
- temporal consistency of frozen foundation model features video optical flow evaluation DINOv2 feature stability (nothing on target; DAVIS label propagation and tracking probes only)
- What makes pre-trained visual representations successful for robust manipulation Burns emergent segmentation
- choosing which vision encoders to combine for robot policy, predictor of ensemble or pair benefit without training (nothing on target)
- correlating dynamics-related property of pretrained visual representation with downstream world model or policy success rate (nothing on target)
- benchmark point tracking probe frozen image encoders temporal correspondence evaluation video representation TAP-Vid readout
- Guo metric-guided feature fusion visual foundation models segmentation edge fidelity label-free metrics (confirmed Guo's EF and SC metrics; the paper was not re-read here)
- feature-space temporal difference vs pixel motion alignment metric self-supervised video representation motion sensitivity (found MCL and others; none is a score for frozen encoders)
- Rethinking Representations for World-Action Modeling arXiv
- Motion-focused Contrastive Learning of Video Representations arXiv
- Baldassarre DINO-world video world model latent space pretrained DINOv2 features
- Deep Feature Flow video recognition warp feature maps with optical flow
- VideoGLUE video understanding evaluation of foundation models motion understanding frozen features (nothing on target; benchmark by task accuracy)
- do foundation model features encode motion? measure feature temporal difference correlation with frame difference robot manipulation video (found CoRe-WAM, What/Where/How; no correlation study)
- training-free predictor of which vision foundation models to ensemble or fuse for downstream robot learning, complementarity score encoders (extended mode; found KAGES, COVE)

arXiv API: an id_list lookup for every entry above (titles, authors, dates, comments confirmed), plus one keyword query (optical flow, frozen, representation, robot) that returned one unrelated result.

Seen in results but not given an entry (not opened by me): COVE (arXiv 2606.15765, orchestrating vision foundation models for dense prediction; a search summary says naive averaging gives marginal gains), RoboAct-CLIP (2504.02069, a search summary says it uses consecutive frame-feature differences to isolate motion for action recognition; title confirmed, role not checked), Recurrent Video Masked Autoencoders (2512.13684), Theia (2407.20179), Video Self-Distillation for Single-Image Encoders (2507.19272), "You Don't Need Strong Assumptions: Visual Representation Learning via Temporal Differences" (2606.15956), Making Foresight Actionable (2606.12217). RoboAct-CLIP and 2606.15956 are the ones most worth a second look for question (a).

Not searched: forward citations of Guo et al. 2026 and CoRe-WAM (could hold a closer match); Semantic Scholar; the physical-understanding benchmark literature beyond Garrido 2025 and Musa 2026.
