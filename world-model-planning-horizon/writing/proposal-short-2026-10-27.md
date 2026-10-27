# Why Do Latent World Models Only Plan a Few Steps Ahead?
*Michael Ramirez, research proposal draft, 2026-10-27. Longer version with details: [full notes](https://docs.google.com/document/d/1ieqcwt7FJyhLigUSYx809ItAz8lOE3kD8sOm0ldvhtU/edit).*
## Motivation
Latent world models only choose good actions for goals a few steps away. The Planning Limits paper (2026) finds reliable action ranking only for targets 5 to 10 steps ahead, while Meta-World task goals are 16 to 53 steps away. An 81-times larger predictor doesn't help, and the range depends on the encoder: at their threshold, DINOv2 reaches 10 steps and VideoMAEv2 only 5. Even with a perfect simulator, success falls from 92% to 41% as the goal moves from 5 to 20 steps ahead. But that test scores candidates by true physical distance, not by embedding distance, so it is still unknown how much of the limit comes from the embedding itself.
## Problem statement
For each frozen encoder, we want to split the planning limit into three parts: errors from the learned predictor, errors from the embedding's geometry (distance to the goal embedding ranking candidates wrongly), and the limit of scoring by distance to the goal at all.
## Related works
- **Planning Limits** measures the limit for 5 encoders. Its perfect-simulator test removes the predictor but also changes how candidates are scored, so geometry and scoring effects are mixed.
- **NGC, Monotone Planning Costs, TD-JEPA and Latent Projection (all 2026)** propose new planning costs. NGC, for example, raises far-goal success on TwoRoom from 27% to 95%. Each fixes the cost for one embedding; none checks, per encoder, whether geometry is the problem.
- **AD-WM** shows that low prediction error doesn't guarantee good action ranking, but doesn't trace failures to the embedding.
## Method
We run the same planning test three ways for each encoder, on the same start states, goals and candidate actions:
1. **Learned predictor + embedding distance:** the normal world model.
2. **Real future frames, embedded, + embedding distance:** perfect prediction with the same geometry. This is the new condition.
3. **Real future state + physical distance:** the Planning Limits test.
The gap between 1 and 2 measures predictor error, the gap between 2 and 3 measures geometry error, and 3 itself shows the scoring limit. We also test a cheap check on recorded demos: does embedding distance to the final frame shrink steadily as the robot gets closer?
## Experiments
1. Cheap geometry check on recorded demos for each encoder (the LIBERO features we already have, plus DINO-WM's datasets). CPU only.
2. The three-condition comparison on Meta-World for Planning Limits' 5 encoders plus DINOv3, SigLIP 2 and C-RADIO, at goal distances of 5 to 40 steps.
3. Test whether existing cost fixes (NGC, temporal distance) help most on the encoders with the largest geometry error.
## Expected results
- **Evals:** how often each condition ranks the expert's actions on top; the plannable range in steps; rank correlation between the cheap check and condition 2.
- **What we expect:** geometry error is large for some encoders and small for others, explaining part of why encoders differ, and the cheap check predicts it without a simulator.
- **If not:** if condition 2 is about as good as condition 3 for every encoder, the embedding isn't the problem and the new cost papers are fixing something small. That is also worth reporting.
## References
- Alrasheed et al. 2026. The Planning Limits of Latent World Models. [arXiv 2609.39235](https://arxiv.org/abs/2609.39235)
- You et al. 2026. Latent World Models Should Plan with Distances Along the Latent Manifold (NGC). [arXiv 2610.25013](https://arxiv.org/abs/2610.25013)
- Chahe et al. 2026. Latent World Models with Monotone Planning Costs. [arXiv 2608.09073](https://arxiv.org/abs/2608.09073)
- Bai and Xiong 2026. Temporal-Distance JEPA. [arXiv 2607.25337](https://arxiv.org/abs/2607.25337)
- Jang and Ryoo 2026. Latent Projection for Latent Planning. [arXiv 2610.26979](https://arxiv.org/abs/2610.26979)
- Qiu et al. 2026. AD-WM. [arXiv 2609.30264](https://arxiv.org/abs/2609.30264)
- Zhou et al. 2025. DINO-WM. ICML. [arXiv 2411.04983](https://arxiv.org/abs/2411.04983)
