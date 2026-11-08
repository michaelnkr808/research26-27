# Prior art for a "motion fidelity" score (does a frozen encoder's feature change follow scene motion?)

Date: 2026-11-08 (corrected the same day after an independent review). Scope: a novelty search for a proposed label-free score, modelled on the edge fidelity (EF) score of Guo et al. 2026 (guo2026metric, entry 3 of `../literature/reading-list/2026-11-07-measures.md`). For a frozen encoder, take consecutive frames of robot demos and ask whether the places where the feature map changes line up with where the scene moves (pixel change, optical flow, gripper or object motion), with no action labels. Long-term use: predict which encoders, or pairs, help a world action model before training. Nothing here is cited yet (_standards/02).

How entries were found and checked: web search (queries listed at the bottom), then each paper's arXiv API record (export.arxiv.org) was fetched on 2026-11-08 and its exact title confirmed. Read status uses these labels. "verified (separate reviewer, raw arXiv HTML text, targeted sections; full read only for CoRe-WAM; 2026-11-08)" means a separate reviewer checked these claims against the raw arXiv HTML text (targeted sections; full read only for CoRe-WAM); I rechecked quotes in the reviewer's text extracts. "from the abstract" means the arXiv abstract or a search summary of it, not read further. "not read" means only the arXiv record was opened. Overlap labels are my judgement. Absence claims ("no paper does X") mean only that these searches found none (speculative). Two of the lead's guesses were corrected: arXiv 2507.19507 is an unrelated physics thesis (Baldassarre et al. is 2507.19468), and TAP-Vid was seen only in search results, not opened, so it has no entry. DynaFLIP (arXiv 2605.30350, title confirmed on its arXiv record) is already covered in `../literature/reading-list/2026-11-07-fusion.md` and has no entry here.

## Verdict

Corrected after an independent review (the first version said (b) was "not found"; that was wrong).

(a) Has anyone defined this score? Not the exact spatial score (does feature change happen where the scene moves, measured without labels on frozen encoders): not found (speculative; absence of evidence). The nearest neighbours are Tsagkas et al. 2025 (a label-free frame-to-frame cosine statistic of frozen features, but how much change, not where) and 4D-WAM (Yang 2026; feature changes between frames aligned with Trace Anything's motion, as a training loss and one diagnostic on a WAM, not an encoder score). Further out: Deep Feature Flow (2016) and Motion-Focused Contrastive Learning (2021) use flow with features inside training; FlowFeat (2025) trains a decoder toward flow; point-tracking probes (Aydemir 2024) score frozen features over time but need labels.

(b) Has anyone used a motion or dynamics-related property of frozen features to predict robot or world-model performance before training? Partly (verified by the independent reviewer). Tsagkas et al. correlate a label-free temporal statistic (average cosine similarity of frozen features between consecutive demo frames) negatively with policy success among the 14 PVRs on MetaWorld, within subgroups (the Fig. 4 caption omits the non-temporal ResNet subgroup, where no trend emerged). A caution from Yeom et al. 2026: in a small target-swap study (encoders used as auxiliary future-latent targets for pi0.5, mostly ID-fine-tuned, mostly one seed), probe R2 did not monotonically rank the resulting policies. Static predictors also exist (Dong et al. 2026, Burns et al. 2023).

(c) Has anyone used a per-encoder or pair-level score to choose combinations? Partly. KAGES (Liu and Jiang 2026) picks encoder subsets without training a classifier, by kernel alignment with labels, with a pairwise term in its set score, on image classification and not for robots or motion. The OSBORN paper in the 2026-11-07 list is similar for ensembles of transfer models. Nothing found did this for robot world models with a motion property (speculative).

Closest works. For the intended use (scoring encoders before training): Tsagkas et al. 2025. For the mechanism (feature change compared with motion): CoRe-WAM (Zhou 2026) together with 4D-WAM (Yang 2026); neither is an encoder score.

What stays new (speculative): a spatial score, meaning whether the feature change happens where the motion is, ideally measured along flow so that moving content is not confused with changing features (the confound CoRe-WAM points out), plus the pair angle. A label-free temporal statistic has already been correlated with policy success (Tsagkas), and a small target-swap study (Yeom) found probe R2 did not monotonically rank the resulting policies, so the claim that matters is that a spatial score adds something beyond Tsagkas's statistic and the static proxies of Dong and Burns.

## Entries

Field "Overlap with our idea": same (it is the idea) / partial / adjacent (neighbouring idea only). No entry is "same".

### 1. The Temporal Trap: Entanglement in Pre-Trained Visual Representations for Visuomotor Policy Learning
- Authors: Tsagkas et al. (2025)
- Venue: arXiv 2502.03270v3; no venue note on the record (the comment says it replaces an earlier paper and was split into two studies)
- Link: https://arxiv.org/abs/2502.03270
- Link checked: 2026-11-08 title matches
- Relevance: High
- Overlap with our idea: partial (a label-free temporal statistic of frozen features vs policy success; but it measures how much features change, not where, and uses no flow or pairs)
- Why it might matter: Closest work for the intended use (scoring encoders before training). Sec IV-B defines a label-free "average sequential cosine similarity" of frozen-PVR features between consecutive demo frames. Fig. 4 correlates it negatively with policy success across 14 PVRs on MetaWorld. The trend is among the 14 PVRs, within subgroups (the Fig. 4 caption omits the non-temporal ResNet subgroup, where no trend emerged). It measures how much features change overall, not where, and uses no flow, no motion ground truth and no pairs. So a label-free temporal feature statistic has already been correlated with policy success; what it does not do is check that the change happens where the scene moves.
- Read status: verified (separate reviewer, raw arXiv HTML text, targeted sections; full read only for CoRe-WAM; 2026-11-08); quotes rechecked by me in the reviewer's text extract

### 2. CoRe-WAM: Correspondence-Aligned Temporal Residuals for World Action Models
- Authors: Zhou et al. (2026)
- Venue: arXiv 2609.27314v2, no venue note on the record
- Link: https://arxiv.org/abs/2609.27314
- Link checked: 2026-11-08 title matches
- Relevance: High
- Overlap with our idea: partial (closest in spirit, but a model component, not a score)
- Why it might matter: A world action model that takes the difference between current and past frame features, after matching each place to where it moved (a frozen Trace Anything model supplies the matching). Its argument is that a same-grid-cell difference mixes feature change with a different object moving into that cell. The paper says its difference "measures feature change after this association, not displacement itself", and reports no score of how well feature change tracks motion, no pre-training predictor and no encoder selection. It shows that feature change over time is already a design question in WAMs, and warns that a naive same-cell difference is confounded by motion; our score would need to handle that (compare along flow, not per fixed cell).
- Read status: verified (separate reviewer, raw arXiv HTML text, targeted sections; full read only for CoRe-WAM; 2026-11-08); the CoRe-WAM text said "measures feature change after this association, not displacement itself"

### 3. 4D-WAM: Infusing Spatiotemporal Awareness into World Action Models through Trajectory Fields
- Authors: Yang et al. (2026)
- Venue: arXiv 2608.08023v2, no venue note on the record
- Link: https://arxiv.org/abs/2608.08023
- Link checked: 2026-11-08 title matches
- Relevance: High
- Overlap with our idea: partial (closest on mechanism; a training loss and one diagnostic, not an encoder score)
- Why it might matter: Sec 3.2 computes token-wise consecutive-frame feature changes and aligns them with Trace Anything's motion by cosine similarity, as a training loss for a WAM. Sec 3.1 and Fig. 2(c) measure the same kind of similarity between frame-to-frame feature differences of a pretrained WAM (Lingbot-VA) and of Trace Anything, as a diagnostic. This is the nearest mechanism to our score (feature change compared with a motion signal), but it is applied to a WAM's own representation and used as a training loss, not an encoder score computed on frozen encoders before training, and it does not compare pairs.
- Read status: verified (separate reviewer, raw arXiv HTML text, targeted sections; full read only for CoRe-WAM; 2026-11-08); Sec 3.1 and 3.2 text rechecked by me in the reviewer's text extract

### 4. What Makes Video World Model Latents Action-Relevant: Prediction over Reconstruction
- Authors: Yeom et al. (2026)
- Venue: CoRL 2026 (stated in the arXiv comment)
- Link: https://arxiv.org/abs/2606.07687
- Link checked: 2026-11-08 title matches
- Relevance: High
- Overlap with our idea: adjacent (an action-labelled dynamics probe of features of consecutive frames, and a caution)
- Why it might matter: Sec 4.6 is a small target-swap study: "we fine-tune pi0.5 on LIBERO with an auxiliary target from a frozen encoder", where the target is "used only during training" (Fig. 6), so the encoders are not the policy's visual input. Most targets are ID-fine-tuned checkpoints ("These couplings also differ in loss and weight, so their results are not a controlled comparison of target encoders across groups"), and the Limitations section says "few targets, two couplings, and mostly one policy seed" and that all three benchmarks are simulated. The paper also uses an inverse-dynamics probe (it needs action labels) on features of consecutive LIBERO frames. Sec 4.6: "Probe R2 also does not monotonically rank the policies, so action readout and closed-loop success should be evaluated separately." Keep this as a caution for our premise, but a weak one: it is not a test of frozen encoders as the policy's input. Separately, App C Table 9 says "Temporal cosine decreases monotonically with lambda; action R2 does not"; the outcome there is the action-probe R2, not policy success, and the paper limits its scope: "Cosine is a within-DIFF diagnostic" and frozen pretrained checkpoints' "cosine values would therefore not isolate the same intervention". Speculative: that this is a caution for Tsagkas-style statistics.
- Read status: verified (separate reviewer, raw arXiv HTML text, targeted sections; full read only for CoRe-WAM; 2026-11-08); quote rechecked by me in the reviewer's text extract

### 5. Reconstruction or Semantics? What Makes a Latent Space Useful for Robotic World Models
- Authors: Nilaksh et al. (2026)
- Venue: arXiv 2605.06388v1, no venue note on the record (comment: 9 pages)
- Link: https://arxiv.org/abs/2605.06388
- Link checked: 2026-11-08 title matches
- Relevance: Medium
- Overlap with our idea: adjacent (probe of frozen latents, with action labels, for six encoders)
- Why it might matter: Compares six reconstruction and semantic encoders as latent spaces for action-conditioned world models on Bridge V2, under a fixed protocol. Sec 2.2 uses an inverse-dynamics probe on frozen latents as "an initial diagnostic" of whether an encoder makes action-relevant change explicit. It needs action labels, is a diagnostic rather than a validated predictor, the probe uses no flow, and there are no pairs (the paper uses RAFT flow to grade rollout motion quality in an appendix). It shows encoder choice for world models is being compared on features, and is a candidate benchmark for our score.
- Read status: verified (separate reviewer, raw arXiv HTML text, targeted sections; full read only for CoRe-WAM; 2026-11-08); quote rechecked by me in the reviewer's text extract

### 6. Interpreting Physics in Video World Models
- Authors: Joseph et al. (2026)
- Venue: arXiv 2602.07050v1, no venue note on the record
- Link: https://arxiv.org/abs/2602.07050
- Link checked: 2026-11-08 title matches
- Relevance: Low
- Overlap with our idea: adjacent for (a)
- Why it might matter: Per the lead's note, probes video world models layer by layer for speed and direction. Only the arXiv record was opened (title and authors). Labelled probes on video models, apparently not a label-free score or a robot-performance link (speculative, not read).
- Read status: not read (arXiv record only)

### 7. V-JEPA Policy: Building Effective World-Action Models on Predictive Visual Latents
- Authors: Zhang et al. (2026)
- Venue: arXiv 2609.37250v2, comment: 19 pages, 5 figures, 11 tables
- Link: https://arxiv.org/abs/2609.37250
- Link checked: 2026-11-08 title matches
- Relevance: Low
- Overlap with our idea: adjacent for (b)
- Why it might matter: Per the lead's note, compares visual foundations inside one WAM, so it is outcome evidence for encoder choice in WAMs. Only the arXiv record was opened; whether it offers a pre-training predictor is not checked.
- Read status: not read (arXiv record only)

### 8. You Don't Need Strong Assumptions: Visual Representation Learning via Temporal Differences
- Authors: Daithankar et al. (2026)
- Venue: arXiv 2606.15956v1, no venue note on the record
- Link: https://arxiv.org/abs/2606.15956
- Link checked: 2026-11-08 title matches
- Relevance: Low
- Overlap with our idea: adjacent for (a)
- Why it might matter: A training method (an image encoder plus a motion encoder over RGB differences), not a frozen-encoder score. Not closer than CoRe-WAM.
- Read status: verified (separate reviewer, raw arXiv HTML text, targeted sections; full read only for CoRe-WAM; 2026-11-08)

### 9. RoboAct-CLIP: Video-Driven Pre-training of Atomic Action Understanding for Robotics
- Authors: Zhang et al. (2025)
- Venue: IROS 2025 (per the arXiv comment)
- Link: https://arxiv.org/abs/2504.02069
- Link checked: 2026-11-08 title matches
- Relevance: Low
- Overlap with our idea: adjacent for (a)
- Why it might matter: Differences CLIP frame embeddings inside a trained transformer for policy pretraining. Not a score of frozen encoders. Not closer than CoRe-WAM.
- Read status: verified (separate reviewer, raw arXiv HTML text, targeted sections; full read only for CoRe-WAM; 2026-11-08)

### 10. Capturing Visual Environment Structure Correlates with Control Performance
- Authors: Dong et al. (2026)
- Venue: ICLR 2026 poster (per 2026-11-07 list); arXiv record has no venue note
- Link: https://arxiv.org/abs/2602.04880
- Link checked: 2026-11-08 title matches
- Relevance: High
- Overlap with our idea: adjacent for (b); no overlap for (a) and (c)
- Why it might matter: Probes nine frozen encoders by regressing simulator state (object pose, shape, material, lighting, joint angles, end-effector pose) from single images, then compares probe accuracy with diffusion-policy success on MetaWorld, RoboCasa and SimplerEnv, reporting MMRV and Pearson r (App A.8 adds VC-1 and LIV). The state vector is static, the prediction is from one image, and each encoder is scored alone. It needs simulator state labels; it was checked against real-robot success on a 5-encoder subset on an xArm6 (Sec 5, "a strong correlation"), and in App A.9 with estimated real-world states. It is a close pre-training predictor of robot performance, but it uses labels, no time, and no pairs. Our score would differ by being label-free, temporal and (if extended) pair-level.
- Read status: verified (separate reviewer, raw arXiv HTML text, targeted sections; full read only for CoRe-WAM; 2026-11-08)

### 11. What Makes Pre-Trained Visual Representations Successful for Robust Manipulation?
- Authors: Burns et al. (2023)
- Venue: CoRL 2024 (PMLR v270; confirmed on https://proceedings.mlr.press/v270/burns25a.html per `../literature/reading-list/2026-11-07-wam-context.md`; the arXiv record has only a page-count comment)
- Link: https://arxiv.org/abs/2312.12444
- Link checked: 2026-11-08 title matches
- Relevance: High
- Overlap with our idea: adjacent for (b)
- Why it might matter: Finds that emergent segmentation (Jaccard of the CLS attention map on PASCAL VOC, ViT models only) predicts out-of-distribution manipulation success better than ImageNet accuracy, in-distribution success or shape bias, with no training needed. All four of its metrics are static-image or policy-success metrics; none uses video, motion or flow, and it does not combine encoders. It is the template for "cheap frozen-feature property predicts robot success", which is the framing our score would be tested in. It is a property of the encoder alone, and covers only ViTs.
- Read status: verified (separate reviewer, raw arXiv HTML text, targeted sections; full read only for CoRe-WAM; 2026-11-08)

### 12. When More Foundation Models Means Less: Diagnosing and Addressing Multi-View Fusion Failure
- Authors: Liu and Jiang (2026)
- Venue: arXiv 2608.17490v1, no venue note on the record
- Link: https://arxiv.org/abs/2608.17490
- Link checked: 2026-11-08 title matches
- Relevance: High
- Overlap with our idea: adjacent for (c); closest on selecting encoders
- Why it might matter: Proposes KAGES, which greedily picks a subset of frozen encoders (from a pool of 10, including CLIP, SigLIP, DINOv2, MAE, SAM) by kernel alignment (a CKA-style score) between the summed encoder kernels and the label kernel, with no downstream training. It is training-free but uses labels, runs on image classification and retrieval, and mentions neither robotics nor time. It also tests frozen-LLM fusion (Sec 4.3), and its set score has pairwise between-encoder kernel terms in the denominator (Sec 3.3), so it has a pair component. It is the nearest thing to (c), and its argument that fusing more encoders can hurt matches our harmful pair. Ours would differ by using a motion property, not label alignment, and by targeting robot world models.
- Read status: verified (separate reviewer, raw arXiv HTML text, targeted sections; full read only for CoRe-WAM; 2026-11-08)

### 13. FlowFeat: Pixel-Dense Embedding of Motion Profiles
- Authors: Araslanov et al. (2025)
- Venue: arXiv 2511.07696v1, no venue note on the record
- Link: https://arxiv.org/abs/2511.07696
- Link checked: 2026-11-08 title matches
- Relevance: Medium
- Overlap with our idea: adjacent for (a)
- Why it might matter: Trains a lightweight DPT decoder on top of a frozen encoder so its dense features carry motion information; SEA-RAFT optical flow is the training target ("teacher" in this paper means the EMA copy of the decoder). A linear operator is fit per frame pair, by ridge regression, on the trained decoder's features, not on frozen encoder features. Separately, the paper says one fixed linear layer fails: Sec 3, "Naïvely approximating optical flow with a single linear layer unsurprisingly fails"; Sec 4.4(a), "single linear layer (instead of a distribution) is futile". At test time it takes one image. It evaluates on DAVIS video object segmentation, COCO-Stuff and NYUv2 depth. So flow is a training signal here, not a measuring stick for a frozen encoder, and it gives no support for a linear-map-to-flow score; I am dropping that variant.
- Read status: verified (separate reviewer, raw arXiv HTML text, targeted sections; full read only for CoRe-WAM; 2026-11-08)

### 14. Motion-Focused Contrastive Learning of Video Representations
- Authors: Li et al. (2021)
- Venue: ICCV 2021 Oral (stated in the arXiv comment)
- Link: https://arxiv.org/abs/2201.04029
- Link checked: 2026-11-08 title matches
- Relevance: Medium
- Overlap with our idea: adjacent for (a)
- Why it might matter: A training method for video encoders. Optical flow picks the clips to sample, and a loss aligns the feature map's gradient maps with optical flow so the backbone attends to high-motion places. This is the closest older thing to "feature-map change lines up with flow", but it is a loss used while training one encoder, not a score for comparing frozen encoders, and it has no robot or pair component. Useful as a source for how to define the alignment.
- Read status: from the abstract (arXiv abstract paraphrase and a search summary of the method)

### 15. Deep Feature Flow for Video Recognition
- Authors: Zhu et al. (2016)
- Venue: CVPR 2017 (per CoRe-WAM's reference list, a citing paper; the arXiv record has no venue note)
- Link: https://arxiv.org/abs/1611.07715
- Link checked: 2026-11-08 title matches
- Relevance: Low
- Overlap with our idea: adjacent for (a)
- Why it might matter: Computes features on key frames and moves them to other frames by warping with optical flow, on the assumption that deep features change smoothly along flow. It is the classic statement that feature maps can be carried along flow. It does not score how well a given encoder obeys that, but "how well do features warped along flow match the next frame's features" is a natural reading of our idea, and this is its origin. The warping step is something we could reuse.
- Read status: from the abstract (arXiv record or a search summary of it); not read further

### 16. What, Where, and How: Probing Spatiotemporal Representations in Video Foundation Models
- Authors: Musa et al. (2026)
- Venue: arXiv 2609.01551v1, no venue note on the record
- Link: https://arxiv.org/abs/2609.01551
- Link checked: 2026-11-08 title matches
- Relevance: Medium
- Overlap with our idea: adjacent for (a)
- Why it might matter: Probes layers of V-JEPA 2 and VideoMAE-v2 for camera motion, intuitive physics and anomaly detection. Camera motion is readable at about 60 to 70 percent of the network depth, and temporal features form smooth curves. These are video models, the probes are trained with labels, and nothing here compares feature change with flow or predicts robot performance. It is a hint that the middle layers carry motion, relevant to our layer choice (Q6).
- Read status: from the abstract

### 17. Can Visual Foundation Models Achieve Long-term Point Tracking?
- Authors: Aydemir et al. (2024)
- Venue: ECCV 2024 EVAL-FoMo workshop (stated in the arXiv comment)
- Link: https://arxiv.org/abs/2408.13575
- Link checked: 2026-11-08 title matches
- Relevance: Medium
- Overlap with our idea: adjacent for (a)
- Why it might matter: Tests frozen image foundation models (DINOv2, Stable Diffusion and others) on point tracking: zero-shot, with a light probe, and with LoRA. Stable Diffusion and DINOv2 features track well. This is the standard way the tracking field scores frozen features over time, but scoring needs labelled point tracks, and the probing and LoRA settings train a readout (the abstract says the zero-shot setting involves "without any training", so it has no readout). It ends in tracking accuracy, not a feature-change-against-motion score. Speculative: zero-shot feature tracking scored against estimated tracks (not labels) would be close to a label-free version of our idea. It is also a possible validity check: does our score rank encoders the way tracking accuracy does?
- Read status: from the abstract (arXiv record or a search summary of it); not read further

### 18. Back to the Features: DINO as a Foundation for Video World Models
- Authors: Baldassarre et al. (2025)
- Venue: arXiv 2507.19468v1, no venue note on the record
- Link: https://arxiv.org/abs/2507.19468
- Link checked: 2026-11-08 title matches
- Relevance: Medium
- Overlap with our idea: adjacent for (b)
- Why it might matter: DINO-world predicts future patch features of a frozen DINOv2 and is trained on large uncurated video. It reports forecasting quality and some action-conditioned planning, but I found no pre-training score of the encoder's motion behaviour, and the encoder is a single fixed choice. It supports the premise that features of frozen DINO can be predicted, which is the setting our score is meant to serve.
- Read status: from the abstract (arXiv record or a search summary of it); not read further

### 19. DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning
- Authors: Zhou et al. (2024)
- Venue: arXiv 2411.04983v2, no venue note on the record
- Link: https://arxiv.org/abs/2411.04983
- Link checked: 2026-11-08 title matches
- Relevance: Low
- Overlap with our idea: adjacent for (b)
- Why it might matter: A world model on frozen DINOv2 patch features used for planning. I did not see a pre-training predictor in what I read of it. Included because it was a lead and is the standard example of a frozen-feature world model.
- Read status: from the abstract (arXiv record or a search summary of it); not read further

### 20. Rethinking Representations for World-Action Modeling
- Authors: Jiang et al. (2026)
- Venue: arXiv 2609.38163v1, no venue note on the record
- Link: https://arxiv.org/abs/2609.38163
- Link checked: 2026-11-08 title matches
- Relevance: Medium
- Overlap with our idea: adjacent for (a) and (b)
- Why it might matter: Argues that reconstruction quality is a poor guide to which representation helps a world action model, comparing Video-VAE latents (a pretrained encoder) with DINOv3 features and its own action-shaped bottleneck. Its Sec 4.4 and App E check, with action labels and after the fact, whether frame-to-frame representation change groups by commanded end-effector direction (t-SNE and 10-nearest-neighbour agreement), for Video-VAE variations as well as its own. That is adjacent to (a): representation change compared with a motion signal, but labelled, after training, and not a pre-training score. It cites Yeom et al. and Nilaksh et al. as probing studies. It also shows a WAM on DINOv3 features is a live design.
- Read status: verified (separate reviewer, raw arXiv HTML text, targeted sections; full read only for CoRe-WAM; 2026-11-08)

### 21. For Pre-Trained Vision Models in Motor Control, Not All Policy Learning Methods are Created Equal
- Authors: Hu et al. (2023)
- Venue: arXiv 2304.04591v2, no venue note on the record (ICML 2023, recalled)
- Link: https://arxiv.org/abs/2304.04591
- Link checked: 2026-11-08 title matches
- Relevance: Low
- Overlap with our idea: adjacent for (b)
- Why it might matter: Compares 14 pre-trained vision models under three policy-learning methods and finds the ranking depends on the method. A search summary says it found a link between linear-probe loss and success on one benchmark (I did not open the number). A caution for us: a predictor might rank encoders differently depending on the policy type.
- Read status: from the abstract (arXiv record or a search summary of it); not read further

### 22. What do we learn from a large-scale study of pre-trained visual representations in sim and real environments?
- Authors: Silwal et al. (2023)
- Venue: arXiv 2310.02219v2, no venue note on the record
- Link: https://arxiv.org/abs/2310.02219
- Link checked: 2026-11-08 title matches
- Relevance: Low
- Overlap with our idea: adjacent for (b)
- Why it might matter: Shows simulation trends in pre-trained visual representations mostly carry over to the real robot. It uses sim performance as the proxy, so it is the "train then see" baseline, with no motion or label-free score.
- Read status: from the abstract (arXiv record or a search summary of it); not read further

### 23. Where are we in the search for an Artificial Visual Cortex for Embodied Intelligence?
- Authors: Majumdar et al. (2023)
- Venue: arXiv 2303.18240v2, no venue note on the record (NeurIPS 2023, recalled)
- Link: https://arxiv.org/abs/2303.18240
- Link checked: 2026-11-08 title matches
- Relevance: Low
- Overlap with our idea: adjacent for (b)
- Why it might matter: Introduces CortexBench and VC-1 (masked autoencoding on egocentric video). No single representation wins everywhere. Video is used for pre-training only; motion is not measured and no pre-training predictor is offered.
- Read status: from the abstract (arXiv record or a search summary of it); not read further

### 24. Probing the 3D Awareness of Visual Foundation Models
- Authors: El Banani et al. (2024)
- Venue: CVPR 2024 (stated in the arXiv comment)
- Link: https://arxiv.org/abs/2404.08636
- Link checked: 2026-11-08 title matches
- Relevance: Low
- Overlap with our idea: adjacent for (a)
- Why it might matter: Probes frozen features for 3D structure and for consistency of surfaces across viewpoints. Cross-view consistency is the nearest static analogue of our time-consistency idea. No motion, flow or policy performance.
- Read status: from the abstract (arXiv record or a search summary of it); not read further

### 25. SPA: 3D Spatial-Awareness Enables Effective Embodied Representation
- Authors: Zhu et al. (2024)
- Venue: arXiv 2410.08208v3, no venue note on the record (ICLR 2025, recalled)
- Link: https://arxiv.org/abs/2410.08208
- Link checked: 2026-11-08 title matches
- Relevance: Low
- Overlap with our idea: adjacent for (b)
- Why it might matter: Trains a ViT with neural rendering on multi-view images and evaluates on 268 embodied tasks in eight simulators. A large benchmark that could serve as an outcome measure for validating a score, but it offers no pre-training predictor or motion property.
- Read status: from the abstract (arXiv record or a search summary of it); not read further

### 26. Intuitive physics understanding emerges from self-supervised pretraining on natural videos
- Authors: Garrido et al. (2025)
- Venue: arXiv 2502.11831v1, no venue note on the record
- Link: https://arxiv.org/abs/2502.11831
- Link checked: 2026-11-08 title matches
- Relevance: Low
- Overlap with our idea: adjacent for (a)
- Why it might matter: Uses a violation-of-expectation test: models that predict in a learned representation space notice impossible events, while pixel-space predictors and multimodal language models stay near chance. A physical-understanding probe of video models via prediction surprise, not feature change against flow, and not tied to robot performance. Background for the claim that prediction in feature space captures physics.
- Read status: from the abstract (arXiv record or a search summary of it); not read further

## Search terms

Web search (each run once, results scanned by title and summary):
- label-free score frozen visual encoder feature change between frames aligns with optical flow motion (nothing on target; found FlowFeat and flow-from-DINOv2 papers)
- predict robot policy performance from frozen visual representation properties without training (found Dong 2026)
- temporal consistency of frozen foundation model features video optical flow evaluation DINOv2 feature stability (nothing on target; DAVIS label propagation and tracking probes only)
- What makes pre-trained visual representations successful for robust manipulation Burns emergent segmentation
- choosing which vision encoders to combine for robot policy, predictor of ensemble or pair benefit without training (nothing on target)
- correlating dynamics-related property of pretrained visual representation with downstream world model or policy success rate (missed Tsagkas and Yeom; the independent reviewer found them)
- benchmark point tracking probe frozen image encoders temporal correspondence evaluation video representation TAP-Vid readout
- Guo metric-guided feature fusion visual foundation models segmentation edge fidelity label-free metrics (confirmed Guo's EF and SC metrics; the paper was not re-read here)
- feature-space temporal difference vs pixel motion alignment metric self-supervised video representation motion sensitivity (found MCL and others; none is a score for frozen encoders)
- Rethinking Representations for World-Action Modeling arXiv
- Motion-focused Contrastive Learning of Video Representations arXiv
- Baldassarre DINO-world video world model latent space pretrained DINOv2 features
- Deep Feature Flow video recognition warp feature maps with optical flow
- VideoGLUE video understanding evaluation of foundation models motion understanding frozen features (nothing on target; benchmark by task accuracy)
- do foundation model features encode motion? measure feature temporal difference correlation with frame difference robot manipulation video (found CoRe-WAM, What/Where/How; this query's top results also contained the Tsagkas paper's arXiv id 2502.03270, which I did not follow up in the first pass)
- training-free predictor of which vision foundation models to ensemble or fuse for downstream robot learning, complementarity score encoders (extended mode; found KAGES, COVE)

arXiv API: an id_list lookup for every entry above (titles, authors, dates, comments confirmed), plus one keyword query (optical flow, frozen, representation, robot) that returned one unrelated result. A second id_list lookup on 2026-11-08 confirmed the added entries (2502.03270, 2606.07687, 2605.06388, 2608.08023, 2602.07050, 2609.37250) and DynaFLIP (2605.30350).

Seen in results but not given an entry (not opened by me): COVE (arXiv 2606.15765, orchestrating vision foundation models for dense prediction; a search summary says naive averaging gives marginal gains), Recurrent Video Masked Autoencoders (2512.13684), Theia (2407.20179), Video Self-Distillation for Single-Image Encoders (2507.19272), Making Foresight Actionable (2606.12217).

Not searched: forward citations of Guo et al. 2026, CoRe-WAM, Tsagkas et al. and 4D-WAM (could hold a closer match); Semantic Scholar; the physical-understanding benchmark literature beyond Garrido 2025, Musa 2026 and Joseph 2026.
