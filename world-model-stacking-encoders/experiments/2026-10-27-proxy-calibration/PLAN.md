# Proxy-policy calibration round (decision 003)

**Status:** plan only. No code, no Jobs. Nothing here has been run.

## Question
Does a small policy trained on frozen encoder features, with 3 training seeds per config, reproduce the paper's composition ordering? And is D+R worse than D alone once training-seed noise is measured?

## Configs (user, 2026-10-27)
| Config | Encoders | Why |
|---|---|---|
| none | no external encoder (raw low-res pixels + proprio) | floor; the paper's "VAE-only" analogue (not identical) |
| D | DINOv3 | paper's single best |
| R | C-RADIOv4 | about equal to D alone in the paper (LP 83.5 vs 84.0) |
| D+R | DINOv3 + C-RADIOv4 | the harmful pair (LP 78.6, below both singles) |
| D+L | DINOv3 + SigLIP 2 | positive control, paper's best D pair (+1.8) |

3 training seeds each → 15 training runs. Under the 20-Job gate if each run is one Job; eval Jobs count separately (see Gates).

## Proxy design (draft, needs review before code)
- **Inputs per step:** for each encoder, both cameras (agentview, eye-in-hand), `grid` tokens (g × g, from `src/extract_features.py`) projected per source to a shared width, concatenated along channels (the paper's default fusion). Plus proprio and the language instruction embedding (frozen text encoder, same for all configs).
- **Outputs:** an action chunk (as Fast-WAM; chunk length to match their setup, to check), and future-feature predictions for every source in the config.
- **Loss:** action loss + Σ_sources feature-prediction loss, weight 1.0 each (paper App. E.1 Eq. 9). Per-source feature normalization: z-score per dimension on the training set. The paper doesn't state its normalization, so this is our choice; log it.
- **Model size:** small transformer, sized so one run fits on 1 GPU in hours, not days (speculative until timed).
- **Same everything across configs** except the encoder set: architecture, steps, LR, seeds 0/1/2, data order.

## Data
- All four LIBERO suites, matching the paper's 1,712 training episodes (Fast-WAM's processed data, paper App. C.1). **Open:** we extracted libero_goal from `yifengzhu-hf/LIBERO-datasets`; need to check whether Fast-WAM's 1,712-episode set is a filtered subset of these and how to get the same list.
- **Open: resolution.** The paper uses 224×224 per view. Rollout images must match the training images' resolution and orientation (OpenGL, rotate 180°), or the features shift between train and test.
- Extraction: D, R, L on libero_spatial, libero_object, libero_10. Rough cost from 2026-10-18 (verified, TITAN RTX): L ~142 frames/s, R ~62 frames/s on one suite. D not timed in that Job.

## Evaluation
- **Clean LIBERO:** 4 suites × 10 tasks × N rollouts. The paper uses 50 per task (2,000). Same per seed here.
- **LIBERO-Plus:** the paper uses 10,030 episodes × 3 eval seeds per checkpoint. For 15 checkpoints that is too much (speculative until a rollout is timed). Proposal: a fixed, stratified subset by perturbation category, the same subset for every checkpoint, sized after timing. Report per-category numbers (the paper gives none for combos).
- Encoders run live in the rollout loop, so eval is GPU work too.
- **Open:** LIBERO-Plus install and episode list. Not yet checked; the repo and its license need reading before anything else.

## Pre-registered outcome rule
Primary metric: LIBERO-Plus success on the stratified subset, mean over 3 training seeds, with a demo-free CI from seeds × episodes (method to fix before results).
1. **Calibration passes** if D > none and D+L ≥ D − (one seed sd). If it fails, the proxy doesn't track the paper and the D+R comparison isn't interpreted.
2. **If calibration passes:**
   - D+R < D by more than the training-seed spread → the harm is real in the proxy; then compare clean vs LP and per category to split explanations (2) and (3).
   - D+R within seed spread of D → consistent with explanation (1), single-checkpoint noise in the paper. Report as "the proxy doesn't reproduce it", not "the paper is wrong".
3. Clean-LIBERO results are secondary and reported alongside.

## Order of work
1. Read LIBERO-Plus repo and Fast-WAM data docs (no compute). Resolve the open items above.
2. Write the proxy training code + synthetic tests locally.
3. Local smoke test on libero_goal features (CPU or Mac, tiny).
4. **Ask:** extraction Jobs for 3 suites × D, R, L.
5. **Ask:** 1 pilot training run + timed rollouts (clean and LIBERO-Plus) on 1 GPU, to size everything.
6. **Ask:** the 15 training runs and eval.

## Gates
Every step marked Ask needs a go-ahead for that specific Job. STANDARDS gate: more than 8 GPUs at once or more than 20 Jobs in a sweep.
