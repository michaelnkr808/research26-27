# Proxy-policy calibration round (decision 003)

**Status:** steps 1–3 done 2026-10-27 (open items resolved; code + tests; local smoke test). Steps 4–6 need a go-ahead per Job. No Jobs.

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

## Step 1 findings: open items resolved (agent, 2026-10-27; repos read at LIBERO-plus `4976dc3` 2026-01-21, FastWAM `7faa711` 2026-08-20)
Sources: github.com/sylvestf/LIBERO-plus (README, `libero/libero/benchmark/task_classification.json`, `requirements.txt`); github.com/yuantianyuan01/FastWAM (README, `configs/data/libero_2cam.yaml`, `configs/task/libero_uncond_2cam224_1e-4.yaml`, `configs/sim_libero.yaml`, `experiments/libero/{libero_utils,eval_libero_single}.py`); HF `yuanty/LIBERO-fastwam` (README, `lerobot_v30/*/meta/info.json`); liu2026what arXiv HTML v1 Tables 6–7 and App. C.2, E.3.

**Training data: the paper's 1,712 episodes**
- verified: they are Fast-WAM's released dataset `yuanty/LIBERO-fastwam`, not `yifengzhu-hf/LIBERO-datasets`. Per-suite `total_episodes`: spatial 434, object 457, goal 433, libero_10 388 = **1,712** (277,713 frames, 20 fps). Suite dirs are `libero_*_no_noops_lerobot`; LeRobot v2.1 tar.gz (0.8–1.5 GB each) and v3.0 parquet + mp4.
- verified: regenerated with **MuJoCo 3.3.2**; the dataset README warns it is "MuJoCo-version sensitive". License CC BY 4.0 (derivative of LIBERO). FastWAM code: MIT.
- verified: features per frame: `observation.images.image` and `.wrist_image` **512×512×3 (video)**, `observation.state` (8: eef pose 6 + gripper 2), `action` (7: delta eef 6 + gripper).
- recalled: "no_noops" follows OpenVLA's regeneration (drop zero-action steps, drop demos whose replay fails). The exact filter isn't documented in the FastWAM repo. speculative: the counts differ from OpenVLA's because of the MuJoCo 3.3.2 replay.
- **Consequence:** our cached libero_goal features (500 demos, 128×128 HDF5 frames) are a different episode set and source resolution. To match the paper, extraction (step 4) should read Fast-WAM's dataset for **all four suites, libero_goal included**. This is how the PLAN's "matching the paper's 1,712 episodes" is met, not a change to it.

**Resolution and orientation**
- verified (paper Table 7; FastWAM `libero_2cam.yaml`): training images are resized from 512×512 to **224×224 per view**; the two views are concatenated horizontally for Fast-WAM's video model.
- verified (FastWAM `libero_utils.py`, `eval_libero_single.py`): rollouts render at **256×256** (`LIBERO_ENV_RESOLUTION = 256`, commented "resolution used to render training data"), rotate 180° (`[::-1, ::-1]`, "to match train preprocessing"), then center-crop-resize to 224.
- Discrepancy (verified that both exist, not resolved): the dataset metadata says 512×512, the eval comment says 256. Both end at 224, so it may not matter, but it's a train/eval difference in Fast-WAM itself.
- speculative (from the eval comment): frames in Fast-WAM's dataset are **already upright**, so our extractor's default `--rotate180` must be **off** for that data, unlike the raw HDF5. Check by viewing one decoded frame before extraction.
- Proposal for our rollouts (speculative, needs user OK as an eval detail): render at 512, rotate 180°, resize to each encoder's input size, so rollout frames go through exactly the same path as training frames.

**Action chunk and timing**
- verified (paper Table 6; FastWAM data config): action horizon **32 steps**; 9 video frames per view at offsets **0, 4, …, 32** of a 33-step window (`action_video_freq_ratio: 4`). So future targets are at t+4, …, t+32.
- verified (Table 6): AdamW, lr 1e-4 → 1e-6 cosine, 5% linear warmup, weight decay 0.01, betas (0.9, 0.95), grad clip 1.0, bf16, action loss weight 1.0; LIBERO batch 128, 10 epochs.
- verified (paper E.3; FastWAM `sim_libero.yaml`): eval replans every 10 steps, 30 initial wait steps, no action ensembling, binarized gripper. Max steps (FastWAM): 400 for spatial/object/goal, 700 for libero_10.
- verified (FastWAM): actions normalized min/max; the gripper sign is flipped in the dataloader and flipped back at execution.

**LIBERO-Plus**
- verified: 10,030 episodes, **1 trial each** (README: set `num_trials_per_task` 50 → 1); category and difficulty per episode in `task_classification.json`. By category: camera 1,599, robot initial state 1,550, language 1,537, light 1,142, background 1,076, noise 1,601, layout 1,525. By suite: spatial 2,402, object 2,518, goal 2,591, libero_10 2,519.
- verified: install = drop-in replacement for `libero` (`pip install -e .`), plus apt packages (libexpat1, libfontconfig1-dev, libpython3-stdlib, libmagickwand-dev), `extra_requirements.txt`, and `assets.zip` from HF `Sylvest/LIBERO-plus` unzipped into `libero/libero/`. Pins robosuite 1.4.0, numpy 1.22.4; MuJoCo not pinned.
- verified: **no LICENSE file in the GitHub repo** (GitHub reports none); the HF assets repo is tagged MIT. Treat the code license as unstated; fine for evaluation use, ask before redistributing anything from it.
- speculative: LIBERO-Plus should be run with `mujoco==3.3.2` (Fast-WAM's README instruction for LIBERO) so scenes match the training data. Not tested; check in step 5.
- Stratified subset: still open. The per-category counts above make a fixed per-category sample easy (e.g. N per category per suite, seeded). N is set after the step 5 timing.

## Steps 2–3: code and local smoke test (agent, 2026-10-27)
**Code:** `src/proxy_policy.py`, tests `src/test_proxy_policy.py` (commit d5d10b0). verified: all 9 synthetic checks pass (pixel tokens, windows never cross an episode end, train-only normalization, loss = action + Σ per-source means with weight 1.0, config wiring, seed-only data order, a planted signal is learned from the right stream and not without it, same-seed determinism, end to end through `main()`). Check 7 failed twice on its own premise before passing; the history is in the test file (thresholds unchanged).

**Design choices in the code that need the user's review** (none changes the configs, eval protocol or outcome rule):
1. **Pixel stream in every config** (raw frames, 180° rotated, block-averaged to 32×32 per cam, 4×4 patches): the analogue of the paper's VAE latents, which every config has. So "none" = pixels only, "D" = pixels + DINOv3. `--no-base-pixels` drops it. The PLAN table reads "none" as pixels + proprio but doesn't say whether pixels stay in D, R, etc.
2. Every stream, pixels included, is both input and future target (t+4 … t+32), as in the paper's default joint video–action formulation.
3. **Deterministic regression (MSE)** in place of flow matching, for both actions and future features. Simpler and cheaper; a proxy difference to state in any writeup.
4. Steps past an episode end are masked out of the loss (Fast-WAM pads them).
5. Feature z-score is per channel, pooled over positions and cams, train frames only. Actions min/max to [-1, 1] (as Fast-WAM); proprio = ee pose 6 + gripper 2 (as Fast-WAM), z-scored.
6. Fusion: per stream Linear → width 64, concatenated along channels, Linear → hidden 256; 4-layer pre-norm transformer over 32 grid tokens (2 cams × 4×4) + proprio + text + 32 action queries. ~3.6M params (none) to ~7.7M (two encoders). Size is a guess until step 5.
7. **Text encoder: open.** The model takes a frozen instruction embedding (`--text-emb`, .npz). Without it, a hash embedding is used (smoke only; it gives reworded LIBERO-Plus instructions unrelated vectors). A real choice is needed before step 5, and it must not be one of the compared encoders' own towers if that could favour a config (speculative concern).
8. Data loader reads the yifengzhu-hf HDF5 layout (the cache we have). A loader for Fast-WAM's LeRobot data (mp4 frames, `observation.state`, `action`) is needed at step 4 with the new extraction.

**User decisions on the design choices (2026-10-28):** stay as close to the paper as possible; cheapness is not a reason to deviate.
- Choice 1: base stream = frozen Wan 2.2 VAE latents (the VAE the paper uses; which of the two Wan 2.2 VAEs is still to confirm from the paper), encoded per frame, in every config. Replaces the pixel stream.
- Choice 3: flow matching, not MSE. Agent recommendation after the user's steer: joint flow matching on actions and future features, as in the paper's default joint video–action denoising (Eq. 1), pending user confirmation.
- Choice 7: the paper's T5 text encoder (umT5-XXL in Wan 2.2, recalled; to confirm), precomputed once per unique instruction, per-token, cross-attended.

**Choices 4–5 checked against the sources (agent, 2026-10-28; paper arXiv HTML v1, FastWAM `7faa711`):**
- Choice 3 confirmed by the user: joint flow matching on actions and future features (paper Eq. 1, 9; App. B.2).
- Choice 4, verified: Fast-WAM fills steps past the episode end with the last frame/action repeated (index clamp, `lerobot_dataset.py:658`), flags them `*_is_pad`, and drops them from the loss: action loss averaged over valid steps (`fastwam.py:590–596`), video loss over valid latent steps, where a latent step counts as padding only if all 4 of its frames are (`fastwam.py:420–457`). `skip_padding_as_possible: false` for LIBERO (no resampling). The paper says feature losses use "valid future targets" only (App. E.1, Eq. 9). Change: fill with the repeated last frame/action (we currently mask but fill differently) and keep the masks; same masking rule for VAE latents and features.
- Choice 5, verified: actions min/max to [-1, 1], not stepwise (`configs/data/libero_2cam.yaml`), as in our code. The paper does not say how external features are scaled before channel-wise concatenation (App. A: final-layer tokens; App. B.2, E.1: aligned to the VAE grid and concatenated; App. D calls them "raw" features in the sensitivity analysis). No R²-WAM code is public (project page, 2026-10-28). Our per-channel z-score stays, flagged as a guess.
- Also verified for choices 1 and 7: Fast-WAM's model is `Wan-AI/Wan2.2-TI2V-5B` (`configs/model/fastwam.yaml`) with a T5 embedding cache per instruction, `context_len: 128`. The paper doesn't name its Wan 2.2 variant; speculative: it uses the same 5B model, since it uses Fast-WAM's data and code lineage.

**Choice 6, model shape (user agreed, 2026-10-28):**
1. Full 14×14 grid per camera, no pooling (paper's DINOv3 grid is 14×14×1,024 per view, App. D, verified; Wan 2.2 5B VAE at 224 is also 14×14, recalled).
2. Time aligned to the VAE: 9 frames per view at t, t+4 … t+32 (Table 6, verified) → 1 current + 2 future latent steps (4× temporal compression, recalled); future features averaged in groups of 4 (stated for REPA, App. B.2; applying it to channel-wise concat is our reading, speculative).
3. Channel-wise concat of raw [VAE | source 1 | source 2] per grid cell, then one shared input projection and one output head (App. B.2), replacing per-stream projections to width 64.
4. Two experts (video, action) with joint attention, asymmetric mask (action sees video, not the reverse), T5 by cross-attention, conditioned on state and flow time (Sec. 3.1); separate video and action flow timesteps as in Fast-WAM.
5. No pretrained video prior (the paper starts from Wan 2.2): the main unavoidable gap. Size set from the step 5 pilot.
**Storage (user decision, 2026-10-28):** extract features once to disk and stream them during training (not live encoders: ~2,700× more encoder work, arithmetic). Full-grid float16 over 277,713 frames × 2 cams: DINOv3 223.0 GB, C-RADIOv4 250.8 GB, SigLIP 2-so400m 250.8 GB, VAE (48 ch) 10.5 GB (arithmetic, run 2026-10-28). Needs a ~750 GB CephFS PVC; `aria-world` has no storage ResourceQuota and holds 400Gi PVCs from other projects (kubectl, verified 2026-10-28). User: no need to ask the namespace admin. PVC created at step 4 with the go-ahead for that Job.

**Smoke test (verified, this Mac, 2026-10-27):** DINOv3 could not be used locally (gated; no HF token on this machine) and our cached libero_goal features are only on the cluster PVCs, so features were extracted locally for **3 demos of `turn_on_the_stove_demo.hdf5`** (265 timesteps): C-RADIOv4 (pinned revision, MPS, 5.2 frames/s) and SigLIP 2-**base** (cached; the real runs use the so400m variant; CPU, 48 frames/s). Configs none, c-radiov4, c-radiov4+siglip2 (same shape as D+R: two encoder streams of 1,152 and 768 channels), seed 0, 10 epochs, batch 32, 55 steps, default model size:
| config | params | wall (CPU / MPS) | train time (CPU / MPS) |
|---|---|---|---|
| none | 3.62M | 4 s / 8 s | 3.0 s / 6.0 s |
| c-radiov4 | 6.08M | 6 s / 5 s | 4.5 s / 4.1 s |
| c-radiov4+siglip2 | 7.72M | 7 s / 7 s | 5.7 s / 5.2 s |
| dinov3 | 5.81M | train 4.4 s / 3.6 s | (added 2026-10-28, see below) |
| dinov3+c-radiov4 | 8.27M | train 6.2 s / 5.2 s | |
| dinov3+siglip2 (base) | 7.45M | train 5.5 s / 4.8 s | |
About 0.1 s per batch-32 step for two encoder streams on CPU. Losses are not results (3 demos, 55 steps; encoder feature losses still ≈ 1.0).
**Update 2026-10-28 (verified):** the user logged this Mac in to Hugging Face with the token from the cluster Secret `stacking-encoders-hf` (user `michaelnkr808`, read; it reads `facebook/dinov3-vitl16-pretrain-lvd1689m` at `ea8dc28`). DINOv3 features extracted locally for the same 3 demos (CPU, 16.3 frames/s; MPS can't pool its 14×14 grid to 4×4), and the real configs dinov3, dinov3+c-radiov4 and dinov3+siglip2 (SigLIP 2-base) all train end to end on CPU and MPS (rows above).

**Sizing notes for step 5 (arithmetic from verified counts; speculative as estimates):**
- Full data = 277,713 frames. Grid features in RAM as float16 at 32 tokens per frame: DINOv3 (1,024) ≈ 18.2 GB, C-RADIOv4 (1,152) ≈ 20.5 GB, pixels ≈ 3.4 GB. D+R ≈ 42 GB of host RAM as written. Options before step 5: memory-map the caches, or pool to fewer tokens. Not decided.
- 10 epochs at batch 128 ≈ 21,700 steps per run.

## Stage 1: code for the faithful proxy (agent, 2026-10-28; FastWAM `7faa711`, liu2026what arXiv HTML v1)
Replaces `src/proxy_policy.py` (kept for history; its smoke results above are superseded). New code: `src/fastwam_data.py`, `src/extract_fastwam.py`, `src/t5_cache.py`, `src/proxy_wam.py`, tests `src/test_proxy_wam.py`.

**Facts found while porting (verified, FastWAM source unless noted):**
- Cameras: agentview + wrist, each 512² → 224² (bilinear, antialias), side by side to 224 × 448, [-1, 1] for the VAE (`configs/data/libero_2cam*.yaml`, `robot_video_dataset.py`). VAE latent grid 14 × 28, 48 channels, 4× time (`Wan2.2_VAE`, `WanVideoVAE38`: z_dim 48, ×16 spatial). DiT patch (1, 2, 2) → 98 tokens per latent step, 294 video tokens.
- Training frames in `yuanty/LIBERO-fastwam` are already upright (episode 432 checked by eye). Rollouts rotate both cameras 180° (`libero_utils.py:47–52`) and **invert the gripper sign** of predicted actions (`eval_libero_single.py:442`); the paper's composition evals also binarize the gripper, use 10 inference steps, replan every 10 steps, wait 30 steps, no action ensembling, text guidance 1.0 (App. E.3). Needed for the rollout code (stage 3).
- Correction to choice 4: padded actions are **not** the last action repeated. Fast-WAM clamps the index, then zeroes the 6 delta dims of padded steps and keeps the clamped gripper (`fastwam_processor.py`), and drops them from the loss. Images: last frame repeated; a future latent step is padding only if all 4 of its frames are.
- Proprio is min/max-normalized like actions (not z-scored as in the old proxy); stats over the whole training set; clamp ±5 (`normalizer.py`).
- VAE latents are computed per training window from that window's 9-frame clip (causal VAE: later latent steps depend on the clip start), so they are cached per start frame t, not per frame. Storage: 31.4 GB (not 10.5 GB); total for VAE + DINOv3 + C-RADIOv4 + SigLIP 2 ≈ 756 GB (arithmetic).
- T5: umT5-XXL encoder (`models_t5_umt5-xxl-enc-bf16`), 128 tokens, prompt "A video recorded from a robot's point of view executing the following instruction: {task}"; padded token positions zeroed and the mask set to all ones in the dataset. Proprio enters as one extra context token (Linear 8 → 4096).
- Fast-WAM's defaults download `Wan2.2_VAE.safetensors` from ModelScope `DiffSynth-Studio/Wan-Series-Converted-Safetensors` (sha256 `0e913a2c…`, matched after download). The paper doesn't name its Wan 2.2 variant; that it is TI2V-5B is still speculative.

**Design (as decided with the user, implemented):** the proxy *is* Fast-WAM's `FastWAMJoint` (video DiT + action DiT, mixture of transformers, joint attention, first-frame-causal video mask, action sees all video) built at small size from random weights: video width 384, action width 128 (paper 3:1), 8 layers, 3 heads × 128 (Wan's head size; Wan's 3D RoPE needs head_dim // 3 even), FFN ratios as Fast-WAM. Inputs channel-concatenated per grid cell [VAE 48 | sources]; one expanded patch embedding and head. Loss = Fast-WAM's (shift 5 video / 1 action, timestep weights, masked padding) with the video term split per source (Eq. 9). Size still set at the pilot.

**Our choices where the paper is silent (speculative, flagged for any writeup):**
- Encoders see Fast-WAM's 224² frame and resize it themselves (DINOv3 224 → 14 × 14; C-RADIOv4 448 → 28 × 28; SigLIP 2-so400m 384 → 27 × 27), then their grid is resized bilinearly to 14 × 14 per camera.
- Future features per latent step = mean of its 4 frames (stated for REPA, App. B.2).
- Source channels z-scored per channel with training-set stats (computed during extraction, no extra pass).
- Batches from 16-episode buffers (each episode read once per epoch), not a full shuffle.

**Checks (verified, `test_proxy_wam.py`, all 12 pass, CPU):** normalizer, padding masks and the full training loss equal Fast-WAM's own code (loss identical to Fast-WAM's `training_loss` on the same draws); attention mask; data plan coverage and determinism; z-score; overfit one batch; inference determinism; checkpoint resume reproduces a straight run bit for bit; `main()` end to end. Check 9 (overfit) was revised after two failed runs; history in the test and in LOG.

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
