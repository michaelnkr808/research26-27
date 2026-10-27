# Which Embedding Should a World Model Plan In?
*Michael Ramirez, research proposal draft, 2026-10-27. Longer version with details: [full notes](https://docs.google.com/document/d/1okKCxvxoZk5LbEkR6V0REmOmYB5KFuCCF16mA7s65oQ/edit).*
## Motivation
Latent world models such as DINO-WM plan inside a frozen encoder's embedding space. They imagine how the embedding changes under different actions and pick the actions whose result lands closest to a goal image's embedding. The choice of encoder matters a lot: in DINO-WM, PushT success is 0.90 with DINOv2 patch features but 0.20 with an ImageNet ResNet. Yet nobody can tell which encoder will plan well without training a world model on each one, and standard single-frame probes don't track planning. ALPS (2026) notes that the encoders with the best single-frame readout are not always the most predictable under learned dynamics.
## Problem statement
We want a cheap, training-free score, computed from frozen embeddings of recorded robot trajectories, that ranks encoders by how well a world model will plan in them.
## Related works
- **DINO-WM and V-JEPA 2** plan in frozen embedding spaces, but compare encoders only by training a world model on each.
- **ALPS and The Planning Limits of Latent World Models** benchmark 15 and 5 encoders for world models. Neither tests whether a cheap measure predicts planning success, or measures how much of the embedding is tied to actions.
- **Reconstruction or Semantics? and Video World Models Need Predictable Latents** already measure action recoverability with an inverse-dynamics probe on frozen latents (6 encoders), and how much of the latent change a linear, action-conditioned predictor explains. Neither tests whether these measures rank encoders by planning success before a world model is trained.
- **Representation Geometry Matters and One-Step Prediction Fidelity Is a Poor Proxy** look for cheap signals of planning quality. One-step prediction error does not track planning success across training checkpoints (r = −0.010), and latent-distance alignment is compared on only two encoders, without actions.
- **AD-WM and Sensorimotor World Models** show that action-relevance matters by adding action-recovery losses during training. They are training methods, not ways to pick a frozen encoder.
- **LogME and Agostinelli et al.** pick pretrained models without training, but for classification labels on single images, with no actions or dynamics.
## Method
Planning only works if different actions lead to clearly different embeddings. So a good embedding is one where most of the change from frame to frame is caused by the robot's actions, not by lighting, noise or background. We score each encoder on recorded trajectories with simple linear models:
1. **Controllable share:** how much better we can predict the next embedding change when we also know the action.
2. **Action recoverability:** how well the action can be read back from two consecutive embeddings. Nilaksh et al. (2026) report this on frozen latents but do not test it as a predictor of planning.
3. **Multi-step versions** of both, since planners look several steps ahead.
We then check whether these scores rank encoders the way their real planning success does.
## Experiments
1. Score 8 to 15 frozen encoders on DINO-WM's public datasets (PushT, Wall, PointMaze) and compare with the planning results already published in DINO-WM, ALPS and Planning Limits. Baselines: a single-frame state probe and latent-distance alignment (Hu et al. 2026). No training needed.
2. Train DINO-WM on about 8 encoders with 3 seeds each to get our own ground truth.
3. Add background and lighting changes to the frames (as in Toso et al. 2026) and check whether the score drops for the same encoders whose planning drops.
## Expected results
- **Evals:** rank correlation (Spearman, Kendall τ) between score and planning success; how often the score picks the best encoder; planning success rate as DINO-WM reports it.
- **What we expect:** the controllable-share score ranks encoders better than single-frame probes, recovers known gaps without training (such as DINOv2 patch features over its CLS token), and falls when distractors are added.
- **If not:** if single-frame probes do just as well, action-relevance adds nothing measurable, and we report that.
## References
- Zhou et al. 2025. DINO-WM. ICML. [arXiv 2411.04983](https://arxiv.org/abs/2411.04983)
- Liu et al. 2026. ALPS: Visual State Encoders for World Models. [arXiv 2610.30120](https://arxiv.org/abs/2610.30120)
- Alrasheed et al. 2026. The Planning Limits of Latent World Models. [arXiv 2609.39235](https://arxiv.org/abs/2609.39235)
- Qiu et al. 2026. AD-WM. [arXiv 2609.30264](https://arxiv.org/abs/2609.30264)
- Ivashkov et al. 2026. Sensorimotor World Models. [arXiv 2606.20104](https://arxiv.org/abs/2606.20104)
- Toso et al. 2026. Learning Invariant Visual Representations for Planning with JEPA World Models. [arXiv 2602.18639](https://arxiv.org/abs/2602.18639)
- Assran et al. 2025. V-JEPA 2. [arXiv 2506.09985](https://arxiv.org/abs/2506.09985)
- You et al. 2021. LogME. ICML. [arXiv 2102.11005](https://arxiv.org/abs/2102.11005)
- Nilaksh et al. 2026. Reconstruction or Semantics? What Makes a Latent Space Useful for Robotic World Models. [arXiv 2605.06388](https://arxiv.org/abs/2605.06388)
- Scardecchia and Yang 2026. Video World Models Need Predictable Latents. [arXiv 2610.34663](https://arxiv.org/abs/2610.34663)
- Hu et al. 2026. Representation Geometry Matters for Planning with JEPA World Models. [arXiv 2608.16287](https://arxiv.org/abs/2608.16287)
- Cheung and He 2026. One-Step Prediction Fidelity Is a Poor Proxy for Planning Performance in JEPA World Models. [arXiv 2610.19375](https://arxiv.org/abs/2610.19375)
