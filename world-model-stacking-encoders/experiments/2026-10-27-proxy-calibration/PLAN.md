# Proxy-policy calibration round (decision 003)

**Status (2026-10-28):** steps 1–3 done; code rebuilt on Fast-WAM ("Stage 1" below) and eval bookkeeping written. Steps 4–6 are **pre-approved by the user and run autonomously** following `RUN.md` (same folder) until a stop condition there. Outcome rule, evaluation protocol and sample-size rule locked 2026-10-28 (f)–(h). Precondition for the step 5 sizing: `src/lp_eval.py` updated to the locked protocol with passing tests. No Jobs yet. (This status line is from 2026-10-28; current status is in `README.md`, run detail in `LOG.md`.)

## Question
Does a small policy trained on frozen encoder features, with 3 training seeds per config, reproduce the paper's composition ordering? And is D+R worse than D alone *in the proxy* once training-seed noise is measured?

**Framing (amended 2026-10-28 after reviewer feedback, user OK):** this is a **proxy replication**. The proxy (a Fast-WAM-based proxy using the paper's recipe where known; amended wording after the second review) has no pretrained Wan 2.2 video prior, is far smaller, and makes its own choices where the paper is silent (feature z-scoring, temporal averaging of future features) that may affect whether stacking hurts, so its result is about the proxy. Its job is to decide whether a **higher-fidelity, multi-seed WAM replication** (Wan 2.2-based model and training recipe matched to Liu et al. as closely as the available code permits) is warranted. It cannot by itself establish what happened in the paper's model.

## Configs (user, 2026-10-27)
| Config | Encoders | Why |
|---|---|---|
| none | no external encoder (Wan 2.2 VAE latents + proprio; pixels until 2026-10-28) | floor; the paper's "VAE-only" baseline |
| D | DINOv3 | paper's single best |
| R | C-RADIOv4 | about equal to D alone in the paper (LP 83.5 vs 84.0) |
| D+R | DINOv3 + C-RADIOv4 | the harmful pair (LP 78.6, below both singles) |
| D+L | DINOv3 + SigLIP 2 | positive control, paper's best D pair (+1.8) |

3 training seeds each → 15 training runs. Under the 20-Job gate if each run is one Job; eval Jobs count separately (see Gates).

## Proxy design (draft of 2026-10-27; superseded by "Stage 1" below)
- **Inputs per step:** for each encoder, both cameras (agentview, eye-in-hand), `grid` tokens (g × g, from `src/extract_features.py`) projected per source to a shared width, concatenated along channels (the paper's default fusion). Plus proprio and the language instruction embedding (frozen text encoder, same for all configs).
- **Outputs:** an action chunk (as Fast-WAM; chunk length to match their setup, to check), and future-feature predictions for every source in the config.
- **Loss:** action loss + Σ_sources feature-prediction loss, weight 1.0 each (paper App. E.1 Eq. 9). Per-source feature normalization: z-score per dimension on the training set. The paper doesn't state its normalization, so this is our choice; log it.
- **Model size:** small transformer, sized so one run fits on 1 GPU in hours, not days (speculative until timed).
- **Same everything across configs** except the encoder set: architecture, steps, LR, seeds 0/1/2, data order.

## Data
- All four LIBERO suites, matching the paper's 1,712 training episodes (Fast-WAM's processed data, paper App. C.1). Resolved in step 1: they are `yuanty/LIBERO-fastwam`, re-extracted for all four suites.
- Resolution and orientation: resolved in step 1 and Stage 1 (224 per view; training frames already upright; rollouts rotate 180° as Fast-WAM). Checks before extraction and before rollouts in Order of work 4–5.
- Extraction (superseded 2026-10-28): VAE, D, R, L on all four suites of Fast-WAM's data (`src/extract_fastwam.py`), sized from a timed probe (RUN.md step 4). Old reference speeds (2026-10-18, TITAN RTX, other resolution): L ~142 frames/s, R ~62 frames/s.

## Evaluation
- **Clean LIBERO:** 4 suites × 10 tasks × N rollouts. The paper uses 50 per task (2,000). Same per seed here.
- **LIBERO-Plus:** the paper uses 10,030 episodes × 3 eval seeds per checkpoint. For 15 checkpoints that is too much (speculative until a rollout is timed). Proposal: a fixed, stratified subset by perturbation category, the same subset for every checkpoint, sized after timing. Report per-category numbers (the paper gives none for combos). **Superseded by "Evaluation protocol (locked 2026-10-28 (g))" in the outcome rule section.**
- Encoders and the VAE run live on the current frame in the rollout loop, so eval is GPU work too.
- LIBERO-Plus install and episode list: resolved in step 1 (below).

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
- ~~Proposal: render at 512~~ Superseded 2026-10-28 (user: match the paper / Fast-WAM): rollouts follow Fast-WAM's eval path (render 256, rotate 180°, resize to 224), Fast-WAM's own train/eval resolution difference included.

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
- Choice 1: base stream = frozen Wan 2.2 VAE latents in every config, replacing the pixel stream. (Later corrected in Stage 1: encoded per 9-frame clip, not per frame; Fast-WAM's TI2V-5B VAE.)
- Choice 3: flow matching, not MSE: joint flow matching on actions and future features (paper Eq. 1), confirmed by the user below.
- Choice 7: the paper's T5 text encoder, precomputed once per unique instruction, per-token, cross-attended (umT5-XXL verified in Fast-WAM's loader, Stage 1).

**Choices 4–5 checked against the sources (agent, 2026-10-28; paper arXiv HTML v1, FastWAM `7faa711`):**
- Choice 3 confirmed by the user: joint flow matching on actions and future features (paper Eq. 1, 9; App. B.2).
- Choice 4, verified: Fast-WAM fills steps past the episode end with the last frame/action repeated (index clamp, `lerobot_dataset.py:658`), flags them `*_is_pad`, and drops them from the loss: action loss averaged over valid steps (`fastwam.py:590–596`), video loss over valid latent steps, where a latent step counts as padding only if all 4 of its frames are (`fastwam.py:420–457`). `skip_padding_as_possible: false` for LIBERO (no resampling). The paper says feature losses use "valid future targets" only (App. E.1, Eq. 9). Change: fill as Fast-WAM and keep the masks (corrected in Stage 1: padded actions get delta dims zeroed, gripper clamped; frames repeat the last one).
- Choice 5, verified: actions min/max to [-1, 1], not stepwise (`configs/data/libero_2cam.yaml`), as in our code. The paper does not say how external features are scaled before channel-wise concatenation (App. A: final-layer tokens; App. B.2, E.1: aligned to the VAE grid and concatenated; App. D calls them "raw" features in the sensitivity analysis). No R²-WAM code is public (project page, 2026-10-28). Our per-channel z-score stays, flagged as a guess.
- Also verified for choices 1 and 7: Fast-WAM's model is `Wan-AI/Wan2.2-TI2V-5B` (`configs/model/fastwam.yaml`) with a T5 embedding cache per instruction, `context_len: 128`. The paper doesn't name its Wan 2.2 variant; speculative: it uses the same 5B model, since it uses Fast-WAM's data and code lineage.

**Choice 6, model shape (user agreed, 2026-10-28):**
1. Full 14×14 grid per camera, no pooling (paper's DINOv3 grid is 14×14×1,024 per view, App. D, verified; Wan 2.2 5B VAE at 224 is 14×14 per view, verified in Stage 1).
2. Time aligned to the VAE: 9 frames per view at t, t+4 … t+32 (Table 6, verified) → 1 current + 2 future latent steps (4× temporal compression, recalled); future features averaged in groups of 4 (stated for REPA, App. B.2; applying it to channel-wise concat is our reading, speculative).
3. Channel-wise concat of raw [VAE | source 1 | source 2] per grid cell, then one shared input projection and one output head (App. B.2), replacing per-stream projections to width 64.
4. Two experts (video, action) with joint attention, asymmetric mask (action sees video, not the reverse), T5 by cross-attention, conditioned on state and flow time (Sec. 3.1); separate video and action flow timesteps as in Fast-WAM.
5. No pretrained video prior (the paper's video expert starts from Wan 2.2; the paper does not say how its action expert is initialized: Sec. 3.1 verbatim, App. B checked only via a summary, 2026-10-28): the main unavoidable gap. Size set from the step 5 pilot.
**Storage (user decision, 2026-10-28):** extract features once to disk and stream them during training (not live encoders: ~2,700× more encoder work, arithmetic). Full-grid float16 over 277,713 frames × 2 cams: DINOv3 223.0 GB, C-RADIOv4 250.8 GB, SigLIP 2-so400m 250.8 GB, VAE (48 ch) 10.5 GB (arithmetic, run 2026-10-28). Needs a ~750 GB CephFS PVC (superseded 2026-10-28 (e): caches total ≈ 756 GB, so request ~1 TB for headroom); `aria-world` has no storage ResourceQuota and holds 400Gi PVCs from other projects (kubectl, verified 2026-10-28). User: no need to ask the namespace admin. PVC created at step 4 with the go-ahead for that Job.

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

**Sizing notes for step 5 (2026-10-27, for the old 4×4 proxy; superseded by Stage 1 and RUN.md, kept for history):**
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
(Amended 2026-10-28 before any training run, after reviewer feedback, user OK. The original text is in git at 44d26c1. Changes: the paired per-seed comparison became the primary analysis, and the "within seed spread" branch no longer points to paper noise.)
(**Locked 2026-10-28 (f), user OK, after a third and fourth reviewer round, before the pilot.** The thresholds below do not change after the pilot or any later result. The previous wording, "D+L ≥ D − (one seed sd)" and an open minimum drop, is in git at 64edf4e.)

**Evaluation protocol (locked 2026-10-28 (g), user OK, after a fifth reviewer round, before the pilot):**
- **Evaluation seeds:** every selected LIBERO-Plus episode is run under all three evaluation seeds the paper uses, **41, 42, 43** (Table 10 columns; App. C.2, E.3), for every checkpoint, with the same seed IDs across checkpoints. Each episode's score per checkpoint = mean success over its three evaluation seeds. The pilot sizes N counting three rollouts per episode.
- **Subset weighting:** the subset is sampled **in proportion to each of the 28 suite × category cells' size in the full benchmark** (total N, largest-remainder rounding, at least 1 episode per cell; one seeded generator). A plain mean over the subset then estimates the benchmark's overall micro-average, so the 2.7 / 1.35-point thresholds are on the same scale as the paper's overall numbers. (Replaces equal-per-cell sampling with a simple average, which would over-weight small cells.)
- Δ_s below is computed on these per-episode scores.
- **Confidence intervals** (if any are added): resample whole episodes, each with its three evaluation-seed rollouts together; never resample rollouts as if independent.
- **Sample-size rule (locked 2026-10-28 (h), user OK, before the pilot):** paired pilot = two short training runs with the same training seed, D and D+R, both evaluated on the same pilot episodes (their own proportional sample) under eval seeds 41–43. Estimate sd of the paired per-episode difference d = score(D+R) − score(D), each episode averaged over its three eval seeds. Inflate: sd* = 1.5 × sd (variance × 2.25), because one short run may not show the full runs' disagreement. Choose N so SE = sd*/√N ≤ 1.35 percentage points for the per-seed paired mean (the proportional subset makes the plain mean the benchmark-weighted estimate). **Cap:** evaluating all 15 checkpoints (clean LIBERO + LIBERO-Plus subset × 3 eval seeds) must fit in roughly 48 h on ≤ 8 GPUs, from the pilot's rollout timing. If the rule's N exceeds the cap, use the largest affordable N and report the precision it actually gives (an operational choice, not one shown to be precise enough). **The final subset is drawn after sizing, independently, excluding the pilot episodes.** Pilot results never enter the effect analysis.

**The pilot is for sizing only** (model size, LIBERO-Plus subset size N by the sample-size rule above, GPU count). Its success rates do not count toward any result and cannot change this rule.

**All comparisons are paired per training seed:** for configs A and B, Δ_s(A − B) = mean over the fixed LIBERO-Plus subset of success(A, seed s) − success(B, seed s), in percentage points, s = 0, 1, 2. Every report shows the three Δ_s individually and their average. Confidence intervals over episodes describe episode sampling only and are never treated as evidence about training-seed variability. Three seeds are a small screening result, not an estimate of training variance.
Secondary: each config's LIBERO-Plus success per seed and averaged, per category; clean LIBERO as context (the paper reports it near saturation).

1. **Calibration** (a minimum check that the proxy responds to the controls, not proof that it is a good enough copy of the paper's model; the missing Wan 2.2 prior still limits what it can show). Passes only if both hold:
   - (a) D vs none: Δ_s(D − none) > 0 for all three seeds, and their average ≥ +2.7 points.
   - (b) D+L vs D, non-inferiority: average Δ_s(D+L − D) ≥ −1.35 points (half the harm threshold, so a loss that would count as harm for D+R can never count as "doesn't hurt" for D+L). The paper's +1.8 for D+L does not set this margin.
   If either fails: **"don't interpret the D+R comparison from this proxy"**, not "D+R is fine" and not "the proxy is useless". (Known risk: the 1.35 margin is tight with 3 seeds, so noise alone can fail (b); that fails safe.)
2. **If calibration passes,** primary question, D+R vs D:
   - **Substantial proxy effect** if Δ_s(D+R − D) < 0 for all three seeds **and** their average ≤ −2.7 points (half the paper's D+R − D drop of −5.41, Table 10 vs Table 1). Each seed's result shown, with discordant-episode counts. This is a **trigger for the higher-fidelity multi-seed WAM replication, not proof that the effect is statistically established.** Then compare clean vs LP and per category to split explanations (2) and (3).
   - Otherwise (any seed ≥ 0, or average above −2.7) → "the proxy does not show a substantial effect". This does **not** establish that the paper's result was training noise: the harm may need the pretrained video prior or the full-size model.
3. Overlap or feature interference (C-RADIOv4 distills DINOv3, SigLIP 2, SAM 3) stays a hypothesis in any writeup, not the explanation.

## Pilot sizing result, 2026-11-05

Added by the worker 2026-11-05 (d), RUN.md Step 9. A record of what the locked rule gave, not a change to it; full detail and commands in LOG 2026-11-05 (d). Paired pilot: `dinov3` vs `dinov3+c-radiov4`, train seed 0, 100 proportional pilot episodes x eval seeds 41-43 (`results/pilot/pilot-subset.json`, subset_id 73f25bc876f969b3), 500-step checkpoints. Result files recovered by the read Job (sha256 `c07ab4a8...` and `0670ec3e...`, 340 lines each). `lp_eval.py sizing`: **sd 3.333 pts, sd* 5.000 pts, N 14.** Pilot sanity counts (sizing inputs only): LIBERO-Plus successes 1 of 300 rollouts (`dinov3`) and 0 of 300 (`dinov3+c-radiov4`); clean LIBERO 0 of 40 each. **Caveat: sd comes from one successful rollout in 600, so N = 14 is not a usable estimate of full-run disagreement and is below the 28-cell minimum of the proportional draw; the user decides how to proceed (not decided).** Cap (15 checkpoints x (3N LIBERO-Plus rollouts + 40 clean) in 48 h on 8 GPUs = 384 GPU-h, from each pod's own rollout speed, excluding pod startup and queue; the 40 clean trials are eval seed 41 only, and with 3 seeds the caps would be about 3991 / 3665; configs not measured are assumed to run at D or D+R speed; config and GPU are confounded between the two pods; at N = 14 pod startup of about 8 min per pod is as large as the rollout time; training time excluded): pod 0 (D, L4, 484 episodes/h): cap N about 4011 (largest affordable); pod 1 (D+R, L40, 449 episodes/h): cap N about 3684. N = 14 is below both, so the cap does not bind. Stage 4 rollout estimate at N = 14: 2.3 GPU-h (pod 0 speed) or 2.4 GPU-h (pod 1 speed), about 0.3 h on 8 GPUs; all 10,030 episodes would be 959 or 1044 GPU-h (120 or 130 h on 8 GPUs). Timing by category (LOG (d)): in the cell means Sensor Noise takes 1.4x to 10x the time per step of the other categories (libero_goal the worst), but per episode only 8 of the 15 Sensor Noise pilot episodes are slow and the rest run at normal speed, so the cost looks tied to specific episodes, not the whole category; Objects Layout is not slow. The final subset has not been drawn.

## Amendment 2026-11-06: evaluation uses the full LIBERO-Plus benchmark (user decision of 2026-11-05)

Added by the worker "train-wave1" (LOG 2026-11-06 (a)). A dated record of a user decision; the locked text above is unchanged and is not edited.

**Decision (user, 2026-11-05, on how to size the LIBERO-Plus evaluation; the user chose the full benchmark over the compute-cap subset; per the lead's chat with the user, 2026-11-05, the user first said "i dont have a problem with running as many episodes as th ecompute cap allows" and then, offered full versus the cap, chose full):** "yeah the full benchmark is fine just get started on it now while i go to sleep". As the agent reads it: every checkpoint in the round is evaluated on all 10,030 LIBERO-Plus episodes (as in `task_classification.json`, LIBERO-plus `4976dc3`), each under eval seeds 41, 42 and 43, as the paper did.

**What this changes for the round**
- The proportional subset size N from the "Sample-size rule" and the roughly 48 h cap no longer set the LIBERO-Plus episode set. The set is the full benchmark: 10,030 episodes x 3 eval seeds = 30,090 rollouts per checkpoint.
- The pilot-episode exclusion in the sample-size rule becomes moot: nothing is sized from the pilot any more, and nothing is sampled. A subset file covering all 10,030 episodes (`lp_eval.py subset` with n = all) is still written once, its `subset_id` logged and the file committed or recorded, and reused by every checkpoint (RUN.md steps 9 and 11, "same subset file"). The pilot results still never enter the effect analysis.
- The weighting is the benchmark's own: a plain mean over the 10,030 episodes is the overall micro-average, the scale the 2.7 / 1.35-point thresholds are written on.

**What does not change:** the outcome rule, the thresholds (2.7 and 1.35 points), the paired per-training-seed analysis, per-episode scores averaged over the three eval seeds, and eval seeds 41-43 (all as locked in "Pre-registered outcome rule" above). The clean LIBERO list is unchanged as well (40 trials, one per task, eval seed 41). Confidence intervals, if any, still resample whole episodes.

**Rollout cost of the full benchmark** (copied from LOG 2026-11-05 (d) section 4; assumptions as stated there): 15 checkpoints x (3 x 10,030 LIBERO-Plus rollouts + 40 clean rollouts) = 959 GPU-h at pod 0's measured speed (D on an L4, 484 episodes/h) or 1044 GPU-h at pod 1's (D+R on an L40, 449 episodes/h); that is 120 h or 130 h on 8 GPUs. Assumptions: the speed is each pilot pod's own rollout time split between LIBERO-Plus and clean in proportion to the records' `wall_s`; the 40 clean trials are eval seed 41 only; configs not measured are assumed to run at D or D+R speed; config and GPU are confounded between the two pods; pod startup (about 8 min per pod) and queue time are excluded; training time is excluded. These figures are about 2.5 to 2.7 times the old 48 h cap of 384 GPU-h, and 120 to 130 h is already most of RUN.md's stop 5 line for the whole stage 4 (about 7 days). Whether that stop applies, and how to cut wall time (more GPUs is gated at 8), is not decided here.

**Training order (user, 2026-11-03):** the seed-noise check comes first. The full-length `dinov3` runs for training seeds 0 and 1 are trained before the other 13 runs; submitted as Job `stacking-encoders-proxy-train-wave1-dinov3` (LOG 2026-11-06 (a)).

**Training GPU list for all 15 runs (user, 2026-11-06: "Add A40, keep it for all"):** RTX 3090, RTX 4090, RTX A5000, RTX A6000, L40, L40S and A40 (the wave-1 list plus A40), driver > 569, us-west, hosts k8s-chase-ci-07 and ry-gpu-08 excluded. Seed 1 trained on a 3090; seed 0 was resubmitted as `stacking-encoders-proxy-train-wave1b-dinov3-s0` with this list (LOG 2026-11-06 (c)), so the two seed-check runs may be on different GPU models.

**Time limit (user, 2026-11-06, from the lead's chat):** asked whether RUN.md's roughly 7-day stage-4 limit (stop 5) can stretch for the full benchmark, the user said "yes we have time and your time estimates are always long". This is a user override of RUN.md stop 5 for this round (the full-benchmark evaluation of about 120 to 130 h on 8 GPUs, plus training, may exceed 7 days). RUN.md itself is not edited and still says about 7 days; this amendment overrides it. The other stop conditions (8 GPUs, 20 Jobs, GPU floor, and so on) are unchanged.

## DRAFT, needs the user's OK before any wave-1 rollout record is read: seed-noise comparison rule

**Superseded 2026-11-06: the rule is locked in "Seed-noise rule (locked 2026-11-06)" below. This draft stays as history.**

**Source of the idea (user's choice, 2026-11-03):** direction `PLAN.md` "Next" item 1 and direction LOG 2026-11-03 (k): "If two seeds of the same config differ by about as much as the harm threshold (half the paper's D+R − D drop), 3 seeds can't settle the question; rethink seeds or N before training the other 13 runs", and "write the comparison rule down before the runs finish". That wording points to threshold option 1 below (T = 2.7); options 2 and 3 are new in this draft.

**Status: a draft proposal by the worker "train-wave1". Not locked, not user-approved, not applied to anything. The user decides the choices marked OPTIONS.** It uses this PLAN's own constants: the harm threshold of 2.7 points (half the paper's D+R - D drop of -5.41, Table 10 vs Table 1) and the calibration margin of 1.35 points (half the harm threshold).

**What is compared.** The two full-length `dinov3` checkpoints, training seed 1 and training seed 0, on the full benchmark. For each of the 10,030 episodes, the score is the mean success over eval seeds 41, 42, 43 (so each checkpoint has one score per episode). Noise delta = mean over the 10,030 episodes of score(`dinov3`, seed 1) - score(`dinov3`, seed 0), in points, with the number of episodes where seed 1 scored lower and higher. Both runs have the same config, so the delta is training-seed noise (plus whatever eval-seed noise is left after averaging three seeds), not a config effect. Report it with its sign; the rule uses |delta|. It is a single draw, so it is a rough guide to the noise, not an estimate of it.

**Why it bears on the rule.** Each per-seed D+R - D comparison is a difference of two different configs trained at the same seed. If this delta alone is already about as big as the harm threshold, a per-seed D+R - D of -2.7 could appear from noise alone. (speculative, not tested: the same-seed comparison is probably less noisy than this cross-seed delta, because `make_plan` in `proxy_wam.py` (line 270) depends only on the seed and the episode list, so D and D+R at the same seed see the same batch order, while seed 1 vs seed 0 also changes the batch order. If so this check overstates the noise in the real comparison, i.e. it errs on the cautious side, toward "rethink seeds".)

**Result meaning "3 seeds can't settle the question; rethink seeds (or N) before the other 13 runs".** |noise delta| at or above the OPTIONS threshold T below. **Result meaning "go ahead":** |noise delta| below T, plus whatever the chosen floor option below requires (floor option 1: nothing more; option 2: both runs' mean LIBERO-Plus success at or above 2.7 points; option 3: the user's reading of the reported means). "Rethink seeds" is only a stop for the user's decision (for example more training seeds for D and D+R, which would change the locked 3-seed design); this draft does not pick a new design.

**OPTIONS for the threshold T**
1. T = 2.7 points (the harm threshold). Reading: noise between two same-config seeds is already as big as the effect the rule is looking for. Rough check (speculative, one draw, 1 degree of freedom, and it assumes the cross-seed delta is as noisy as the same-seed D+R - D comparison, which probably overstates the noise, see above): if the 3-seed average of the D+R - D deltas has a noise sd of about |noise delta| divided by sqrt(3), then at |noise delta| = 2.7 that sd is about 1.56 and a pure-noise average would fall at or below -2.7 about 4% of the time before the all-three-negative requirement. Tradeoff: a clean, easy-to-state line tied to the existing constant; with a single draw it can be crossed or missed by chance.
2. T = 1.35 points (the calibration margin). Reading: noise is as big as the margin that decides calibration (b), which the PLAN already calls tight. Tradeoff: more cautious and will fire more often, so it is likelier to ask for more seeds than needed; the cost is the other 13 runs being delayed or enlarged.
3. Two zones: |noise delta| < 1.35 go ahead; 1.35 to < 2.7 go ahead but state in the round's report that noise reaches the calibration margin; >= 2.7 rethink seeds. Tradeoff: matches the two constants already in the PLAN and avoids one hard edge; the middle zone needs the user's reading of it and may be harder to apply the same way later.

**OPTIONS for the floor condition** (a noise delta of about 0 means nothing if both runs score near 0; the 500-step pilot checkpoints had 1 success in 300 LIBERO-Plus rollouts, LOG 2026-11-05 (d)).
1. No floor condition. Tradeoff: a degenerate pair of runs would be read as "low noise".
2. If the `dinov3` mean LIBERO-Plus success is below 2.7 points (the harm threshold) in either seed, the comparison says nothing about noise at the scale of the rule and the answer is "can't tell; user decides". Tradeoff: ties the floor to an existing constant; it is a judgement call that a model scoring just above 2.7 is far enough from the floor.
3. Report the mean success next to the delta and let the user judge, with no automatic rule. Tradeoff: no new threshold, but not a pre-registered rule.

**Not covered by this draft (open):** whether to use the delta's episode-resampling interval (it describes episode sampling only, which on 10,030 episodes is narrow, and is not evidence about training-seed variability); and any per-category noise delta (small cells are noisier, and the rule is on the overall mean).

## Seed-noise rule (locked 2026-11-06)

**Provenance.** The user chose the seed-noise check on 2026-11-03 (direction `PLAN.md` "Next" item 1; LOG 2026-11-03 (k)). On 2026-11-06, in the lead's chat, the lead explained the draft above and recommended threshold option 1 (T = 2.7, the harm threshold) and floor option 2. The user answered: "i just want you to stay as faithful to the paper as possible". The lead replied that the paper has no training-seed check at all (one checkpoint per config) and that the options anchored to the paper's own numbers are threshold option 1 and floor option 2, and said it would lock those unless the user objected. The user did not object. Locked on that basis, 2026-11-06, before any wave-1 rollout record was read.

**What is compared.** The two full-length `dinov3` checkpoints, training seed 1 and training seed 0, on the full benchmark (10,030 LIBERO-Plus episodes). Each episode's score is the mean success over eval seeds 41, 42, 43. Noise delta = mean over the 10,030 episodes of score(`dinov3`, seed 1) - score(`dinov3`, seed 0), in percentage points, reported with its sign and with the numbers of episodes where seed 1 scored lower and higher. The rule uses |noise delta|. Seed 0 and seed 1 may have trained on different GPU models (LOG 2026-11-06 (c)); that is noted, not corrected for.

**Outcome.**
- T = 2.7 points (the harm threshold, half the paper's D+R - D drop of -5.41). If |noise delta| is at or above 2.7: "3 seeds can't settle the question; rethink seeds (or N) before the other 13 runs." This is a stop for the user's decision, not a new design.
- Floor: if either seed's mean LIBERO-Plus success is below 2.7 points, the answer is "can't tell; user decides", whatever the noise delta (a delta near 0 means nothing when both runs score near 0).
- Otherwise (|noise delta| below 2.7 and both means at or above 2.7): "go ahead" with the other 13 runs.
- The thresholds are not changed after seeing the result.

**Note (speculative, not tested).** This check is probably conservative: the same-seed D+R - D comparison is probably less noisy than the cross-seed delta, because `make_plan` in `proxy_wam.py` (line 270) depends only on the seed and the episode list, so D and D+R at the same seed see the same batch order, while seed 1 vs seed 0 also changes the batch order. So it errs toward "rethink seeds". A single delta from two seeds is one draw, a rough guide, not an estimate of the training-seed variance.

## Eval bookkeeping for the paired analysis (agent, 2026-10-28; after the reviewer amendment)
`src/lp_eval.py` + `src/test_lp_eval.py` (6 synthetic checks pass):
- `subset`: one fixed LIBERO-Plus subset, up to N episodes per (suite, category) cell (28 cells), one seeded generator, keyed (suite, id) from LIBERO-plus `4976dc3` `task_classification.json` (sha256 `faa87cce…`; not committed: the repo has no LICENSE). Written once with a `subset_id` hash and reused by every checkpoint. N set after the step 5 timing. ~~To change (locked protocol 2026-10-28 (g)): proportional allocation instead of up to N per cell; three eval seeds per episode (41–43) instead of one rollout; analysis averages eval seeds per episode before pairing; add the calibration pairs and the locked verdict.~~ **Done 2026-10-28 (RUN step 7a, LOG (m)):** proportional `make_subset` with pilot exclusion, records per episode × eval seed 41–43 averaged per episode, calibration pairs + locked verdict (thresholds as exact constants), `sizing` (sd, sd* = 1.5 sd, N), episode-level CIs; `test_lp_eval.py` 9 checks pass.
- Rollout code (stage 3) must write one JSON line per rollout: benchmark, suite, episode_id, category, config, train_seed, eval_seed, checkpoint, subset_id, success. Aggregates are never the only output.
- `analyze`: primary = per training seed, per-episode paired success(D+R) − success(D) on the subset, its mean and discordant counts; the three seed deltas side by side. Secondary: per-config success per seed and mean, per category; clean LIBERO as context. Refuses (fails loudly on) missing, duplicate or foreign-subset rows instead of intersecting. Prints the proxy-replication framing with every report.
- Checks: planted per-seed drops come out exactly (e.g. −2/n, −5/n, +1/n); identical configs give exactly 0; pairing by seed; secondary numbers on a hand case; CLI end to end.

## Order of work
1. Read LIBERO-Plus repo and Fast-WAM data docs (no compute). Resolve the open items above.
2. Write the proxy training code + synthetic tests locally.
3. Local smoke test on libero_goal features (CPU or Mac, tiny).
4. **Pre-approved (RUN.md stage 2):** volume, download, timed probe, T5 cache, extraction of VAE, D, R, L for all four suites (was: 3 suites × D, R, L). Before full extraction (added 2026-10-28 (e)): view frames from both cameras in all four suites of `yuanty/LIBERO-fastwam` to confirm they are upright and need no rotation (one episode, one camera checked so far). Volume ~1 TB, not ~760 GB. Headroom checked 2026-10-28 (f): `yuanty/LIBERO-fastwam` is 9.4 GB in total (lerobot_v30 4.7 GB; HF API tree listing), and `extract_fastwam.py` writes one episode at a time atomically (`.tmp` then rename, no merge step), so temporary files are one episode; ≈ 240 GB stays free after the ≈ 756 GB of caches.
5. **Pre-approved (RUN.md stage 3):** paired pilot (amended 2026-10-28 (h)): 2 short training runs, D and D+R, same training seed, + timed rollouts (clean and LIBERO-Plus, eval seeds 41–43) on the pilot's own episodes, sized by the sample-size rule. Before any rollout counts: pass one simulator frame through our rollout preprocessing (180° rotation) and compare it side by side with a training frame of the same task (added 2026-10-28 (e)).
6. **Pre-approved (RUN.md stage 4)** (threshold locked 2026-10-28 (f)): the 15 training runs and eval.

## Gates
User, 2026-10-28: steps 4–6 pre-approved and auto-chained ("move from stage to stage autonomously unless there is some major problem"); stop conditions and exact steps in `RUN.md`. Still gated: STANDARDS (more than 8 GPUs at once or more than 20 Jobs), anything outside RUN.md, protocol or outcome-rule changes, deleting data.

## Amendment 2026-11-06 (b): A40/A6000 are special resources

Added by the worker "train-a40" (LOG 2026-11-06 (f)). A dated record; the locked text above, including "Amendment 2026-11-06", is unchanged.

**Finding (verified, public Prometheus, run by the lead 2026-11-06 about 23:10 UTC; queries and results in LOG (f)):** A40 and RTX A6000 are NRP special resources (`nvidia.com/a40`, `nvidia.com/rtxa6000`). Every node that advertises either has `nvidia_com_gpu` allocatable 0 or absent. So a pod asking for `nvidia.com/gpu: 1` with `gpu.product` NVIDIA-A40 or NVIDIA-RTX-A6000 can never match those nodes.

**Consequence:** the A40 added to the training GPU list by "Amendment 2026-11-06" ("Add A40, keep it for all") added no capacity, and neither did the A6000 entry that was already in the list. The same two entries in the evaluation Job's GPU list add nothing. The lists still work for the other cards (RTX 3090, 4090, A5000, L40, L40S, and for evaluation A10, L4).

**User decision (2026-11-06, on the lead's option 1, race an A40-named twin of seed 0):** "I like 1. just be sure to properly kill the other one". Job `stacking-encoders-proxy-train-wave1c-dinov3-s0-a40` asks for `nvidia.com/a40: "1"`, has no `gpu.product` term, and writes `/mnt/proxy/runs/dinov3-s0-a40`; everything else equals wave 1b. Whichever of it and `stacking-encoders-proxy-train-wave1b-dinov3-s0` is scheduled first is kept; the other Job is deleted (graceful delete). If the twin wins, evaluation of seed 0 must point at `dinov3-s0-a40` (`rollout.py` takes config and seed from the run dir's `config.json`, not its name).

**Going forward:** future renders drop NVIDIA-A40 and NVIDIA-RTX-A6000 from `gpu.product` lists. A named-resource variant (`nvidia.com/a40` or `nvidia.com/rtxa6000`) is added only with the user's OK.

## Amendment 2026-11-07: remaining eval shards on L4 only (user decision; LOG 2026-11-07 (k))
A dated record; the locked text above is unchanged. Fast cards (L40, A40) ran the CPU-bound rollouts at about 20% GPU and drew NRP's idle-GPU warning email. The seven L4 shards averaged 47.7-51.7% (LOG (i), (k)). User, 2026-11-07, verbatim choice: "Swap to L4-only (Recommended)". The unfinished shards (seed 0: 0, 4, 6, 7; seed 1: 2, 4, 5) now run in Jobs `stacking-encoders-proxy-eval-dinov3-s0-l4` and `-s1-l4` with `gpu.product` NVIDIA-L4 only. Seed 0 shard 5 and seed 1 shard 3 finish in their original pods. Episodes, eval seeds, subset, order seed and outputs are unchanged, so the shards stay comparable. The GPU model is not part of the analysis, but it is recorded per pod in the logs.

### Amendment 2026-11-09 (c): train the DINOv3 proxy longer or larger before the other 13 runs (DRAFT for the user's sign-off)

**Status: DRAFT. Not signed off by the user, not applied, nothing submitted.** Written by an implementation worker on 2026-11-09 from the lead's brief. The direction (amend, rather than proceed as pre-registered or stop) is the user's choice made in the planning session's window, as relayed by the lead (the worker did not see that chat). The text below becomes binding only when the user signs it off in the implementation lead's window. Until then nothing runs on the cluster. The locked text above is not edited; where an older line is superseded in intent, a bracketed dated note says so (end of section 5). Labels: verified = read in a repo file this session (file named); recalled = from memory; speculative = reasoning or arithmetic on stated numbers, not tested. Paths are relative to this folder unless they start with `src/`, `writing/` or `literature/` (those are under `world-model-stacking-encoders/`).

**In plain words.** The seed-noise check printed "go ahead", but it only asks whether two runs of the same recipe agree with each other. It does not ask whether the recipe is any good. Both DINOv3 models succeed on about 7 to 8 episodes in 100 of the perturbed benchmark, where the paper's DINOv3 model succeeds on about 84 in 100. At that level, the 2.7-point effect we are hunting is about a third of the model's whole score, so the comparison is hard to read. This amendment proposes giving the DINOv3 proxy a bigger training budget or a bigger model, checking again that two seeds agree, and only then training the other 13 runs, all at the new recipe. No D+R model has been trained at full length yet, so this change cannot be steered by a D+R result.

#### 1. Why now, and why it is not post hoc

Verified facts (file pointers):
- No full-length D+R result exists. The only D+R checkpoint is the 500-step pilot (`results/pilot/`, PLAN "Pilot sizing result"), which the locked rule excludes from every effect analysis ("Pilot results never enter the effect analysis", sample-size rule; "The pilot is for sizing only"). Direction `PLAN.md` "Now" and `writing/2026-11-09-proxy-exit-options.md` ("Facts that apply to every exit") say the same. No eval of any config other than the two `dinov3` runs exists (LOG 2026-11-09 (t) reads only those).
- The seed-noise rule's thresholds stay as locked: T = 2.7 points, floor 2.7 points ("Seed-noise rule (locked 2026-11-06)"). The result that triggered this amendment (LOG 2026-11-09 (t), `results/full-benchmark/seednoise.json`): mean LIBERO-Plus success 8.315 points (training seed 1) and 7.155 points (training seed 0), noise delta +1.160 points, outcome "go ahead". The paper's DINOv3 total is 84.0 and its VAE-only baseline 74.7 (`literature/liu2026what-table1-single-source.csv`, rows DINOv3 and VAE-only, last column).
- The trigger is the absolute level of the two DINOv3 means, not the noise delta and not any D+R number. The options memo `writing/2026-11-09-proxy-exit-options.md` raised the scale issue before the verdict was read (its header: "Written before the seed-noise verdict was read by the planning session"; Exit 6 "The issue": the thresholds are absolute points from "models that succeed about 84% of the time", and a lower proxy score changes what they mean). The memo's Exit 2B ("train the DINOv3 models longer or larger, then rerun the noise check ... needs a written amendment, with thresholds restated as unchanged, before any D+R result exists") is the same move, written for the floor case.

Honest limits on the "not post hoc" claim:
- This is **not** Exit 2: the floor was not hit (both means are above 2.7). It borrows Exit 2B's form because the memo's Exit 6 identified the level problem in advance.
- The decision to retrain was made after the two DINOv3 means were read. What protects the comparison is that no D+R full-length result exists and nothing about the outcome rule changes. It does not protect against the possibility that the decision would have gone differently with other DINOv3 numbers. That is a data-dependent choice and is stated as one.
- The 8 and 7 point means are single numbers from two seeds, and "far below the paper" compares our small from-scratch proxy with a model that starts from a pretrained video prior (PLAN "Choice 6" item 5: the main unavoidable gap). So the low level may not be fixable by training budget at all (see section 3, if the level target is missed).

**What IS being changed (if the user signs off):** the DINOv3 proxy's training budget or size (one option from section 2), applied to both DINOv3 seed-check runs and then to all 15 runs.

**What is NOT changed:** the outcome rule and the calibration checks (a) at +2.7 and (b) at -1.35; the D+R harm threshold -2.7 and the all-three-seeds-negative requirement; the seed-noise thresholds T = 2.7 and floor 2.7; the evaluation protocol (full LIBERO-Plus, 10,030 episodes, subset file `results/full-benchmark/full-subset.json`, subset_id 91072e5da8499c99; eval seeds 41, 42, 43; the 40-trial clean LIBERO list; L4-only rollouts per "Amendment 2026-11-07"); training seeds (0, 1, 2); the paired per-seed analysis; the config list (none, D, R, D+R, D+L).

#### 2. What "longer or larger" could mean (the lead's options; none is picked)

Facts about the current recipe (verified, `src/proxy_wam.py` and the saved logs):
- Defaults: `--epochs` 10 (line 529), `--batch-size` 128 (line 530), `--hidden` 384 (line 535), `--layers` 8 (line 536), `--heads` 3 (line 537), `--head-dim` 128 (line 538); `build_model` at line 372 (action expert width = `hidden // 3`, line 553). 31,647,943 parameters for `dinov3` (both training logs, "wrote ... 31647943 params"). 21,696 steps (10 epochs of about 2,170 steps). LR 1e-4 with 5% warmup and a cosine decay to 1e-6 over the run's own total steps (`lr_at`, line 466; `total = len(plan)`, line 487), so a longer run gets its own full schedule, not an extension of the old one.
- The Job template passes extra flags through `__EXTRA__` on the `proxy_wam.py` line (`jobs/job-train.template.yaml:46`), so epochs or size can be set there without code changes. A new `--out` directory is required: `proxy_wam.py` resumes from `<out>/ckpt.pt` if it exists (lines 561-566), so reusing `runs/dinov3-s0-a40` or `runs/dinov3-s1` would resume the old, finished run (`jobs/rendered/job-train-wave1c-dinov3-s0-a40.yaml:24` makes the same point).
- Measured training cost of the current recipe (verified): seed 1 on an RTX 3090, 3 h 54 min wall from pod start, about 0.64 s per step (LOG 2026-11-06 (c) section 2); seed 0 on an A40, 4 h 00 min wall, 14,143.7 s of training loop, about 0.65 s per step, GPU use 76 to 97% across polls (LOG 2026-11-07 (b)). Waiting for data was 1,373 s and 1,750 s of those. So one run is about 4 GPU-hours.
- Eval cost per checkpoint (arithmetic from "Amendment 2026-11-06": 959 and 1044 GPU-h for 15 checkpoints; same figure as the options memo "Facts"): about 64 to 70 GPU-h per checkpoint. These come from the 500-step pilot's rollout speed, not from the full runs (how they move if success goes up and episodes end earlier is speculative and not measured). Training cost is not in them.
- Memory: the training Job list excludes L4 and A10 because D+R ran out of GPU memory there at the current size (LOG 2026-11-06 (a) section 3); `dinov3` at the current size trained on a 24 GB RTX 3090 (LOG (c)). Peak GPU memory of the full training runs was not logged in the files read (not verified).

**Does the training loss show a plateau? (this decides between more epochs and a larger model)**
- Data (verified): the raw logged lines in `results/logs/train-wave1-dinov3-s1-sh9hj.log` and `results/logs/train-wave1c-dinov3-s0-a40-bj6bf.log`; `src/plot_training_curve.py` plots the same logs, unsmoothed, into `writing/figures/2026-11-07-training-curve-trial-vs-full.png`. Each logged value is **one batch's loss**, not an average (`src/proxy_wam.py:503-504`; one line about every 433 steps, 51 per run), and the loss is flow matching at random noise levels, so single points are noisy.
- What the data show (arithmetic: I averaged the logged points in four step windows; windows hold 6 to 14 points, so this is rough): the total loss falls fast to about step 5,000, then changes little. Seed 1: 1.83 (steps 2,600-5,400), 1.64 (5,400-10,800), 1.65 (10,800-16,200), 1.59 (16,200-21,700). Seed 0: 1.80, 1.70, 1.58, 1.64. The action part: seed 1 0.090, 0.077, 0.065, 0.068; seed 0 0.091, 0.078, 0.072, 0.071. Point-to-point scatter inside a window is about +-0.15 in the total loss, as large as the later differences between windows.
- Reading (the numbers are verified; the interpretation is speculative): over the second half of the run the training loss is close to flat, with at most a small decline, even though the learning rate was being annealed to its floor, which usually pushes loss down at the end. So the data give no sign that the current model was still improving quickly when it stopped. They also cannot show that more epochs would be useless: no held-out loss was logged, the flat part may be the noise floor of the flow-matching loss rather than the model's limit, and closed-loop success can depend on small changes in action quality. What they do suggest: "same model, more epochs" has less going for it than it would if the curve were still dropping, so the larger-model options should not be dismissed.

Options:

**Option A (same model, more epochs).** Knob: `--epochs` (line 529) via `__EXTRA__`. A1 = 20 epochs (43,392 steps), A2 = 40 epochs (86,784 steps). Everything else as now.
- Training cost (speculative, assumes the measured 0.64 to 0.65 s per step holds): A1 about 7.7 to 7.9 h plus about 0.2 h setup, call it about 8 h per run, 2 runs = about 16 GPU-h. A2 about 16 h per run, 2 runs = about 32 GPU-h. Both far under RUN.md stop 5's "one training run > 48 h" (`RUN.md:28`).
- Re-eval: 2 checkpoints x 64 to 70 = about 128 to 139 GPU-h (arithmetic). Total for the re-check: A1 about 144 to 155 GPU-h, A2 about 160 to 171 GPU-h (speculative).
- Jobs: 1 training Job (Indexed, 2 completions, 2 GPUs) + 1 eval Job per checkpoint (8 shards each) = 3 Jobs minimum. Round 1 needed several eval Jobs per seed because shards were resubmitted (LOG 2026-11-07), so plan for up to about 6.
- Worked if: see section 3 (the re-check rule and the level target). The training loss is not evidence of success; as a diagnostic only, a clearly lower late-training loss than the current run would be a sign the extra epochs did something.

**Option B (larger model, same 10 epochs).** Knobs: `--hidden` (line 535), `--layers` (line 536), `--heads` (line 537) via `__EXTRA__`; the action width follows (`hidden // 3`, line 553), and `build_model` asserts `head_dim // 3` is even (line 375). B1 = deeper: `--layers 16`. B2 = wider: `--hidden 768 --heads 6` (same head size). The defaults are set up as 384 = 3 x 128, but doubling hidden and heads together was not run, so a short build-and-step check would be needed first (a gated Job; speculative that it works).
- Training cost (speculative arithmetic, not measured): B1 about twice the compute per step, so about 8 h per run (less if the step is partly data-bound; waiting for data was about 12% of the loop in seed 0, LOG 2026-11-07 (b)), 2 runs = about 16 GPU-h. B2 about 4 times the compute in the transformer layers, so roughly 2.5 to 4 times the step time, about 10 to 16 h per run, 2 runs = about 20 to 32 GPU-h. Parameter count would rise to roughly 2x (B1) or 4x (B2) of 31.6M (speculative).
- Memory (speculative): a larger model is likelier than the current one to need more than 24 GB, which would remove the RTX 3090, 4090 and A5000 from the training list and leave the L40, L40S and the A40 named resource (3 A40 nodes seen on 2026-11-06, LOG 2026-11-06 (g)). That can lengthen the queue (round-1 seed 0 waited about 9 h before the A40 route, LOG 2026-11-06 (c)). It also must fit D+R and D+L, which have wider inputs still.
- Re-eval and Jobs: as Option A. Re-check total: B1 about 144 to 155 GPU-h, B2 about 148 to 171 GPU-h (speculative). Eval cost may differ for a bigger policy, though the rollouts were CPU-limited (policy busy 43 to 57% in the pilot, LOG 2026-11-05 (b), (c)); not measured for a bigger model.
- Worked if: as Option A.

**Option C (both, e.g. 20 epochs and 16 layers).** Mentioned for completeness; not recommended as a first step because it changes two things at once and cannot say which one helped (speculative). Cost is roughly the product: about 16 h per run.

**The lead's reading (marked as such, not a pick).** (1) The loss data give no sign of a still-falling curve, which weakens the case for pure more-epochs (A1, A2) and leaves a model with more room (B) and "neither helps" as live possibilities. (2) A1 is the lowest queue risk (same hardware list, same code path) and the easiest to explain; A2 is likeliest to run into a deadline or queue problem. (3) B2 changes the most and has the largest unknowns (memory, step time, whether the builder accepts it). (4) None of the options addresses the missing pretrained prior, the one gap the PLAN already names as unavoidable; if that is the main reason for the low level, no budget in this list will close it, and section 3's level target will say so after about 150 to 170 GPU-h have been spent. A cheaper trial of whether training budget matters at all is not offered, because the only eval that counts is the full-benchmark one (about 64 to 70 GPU-h per checkpoint).

#### 3. The re-check rule after retraining (pre-registered now)

1. Train the chosen DINOv3 recipe for training seeds 0 and 1 (new `--out` directories, section 2), same code commit for both, same GPU-model list.
2. Evaluate both on the full benchmark exactly as before: same subset file (subset_id 91072e5da8499c99), eval seeds 41, 42, 43, the same 40-trial clean LIBERO list, L4-only rollouts (Amendment 2026-11-07).
3. Run `lp_eval.py seednoise` (config `dinov3`, A = training seed 1, B = training seed 0) with the locked thresholds unchanged: T = 2.7 points, floor 2.7 points. Outcomes as locked: |delta| at or above 2.7 means "3 seeds can't settle the question; rethink seeds (or N)" (the options memo Exit 1 applies); either mean below 2.7 means "can't tell; user decides"; otherwise "go ahead". The thresholds are not changed after seeing the result.
4. At most one retrain under this amendment. If the re-check does not give "go ahead" and meet the level target below, the next step is a user decision (listed below), not a second retrain.
5. Both pairs are reported. The original pair (10 epochs, 384 wide, 8 layers) keeps its recorded outcome "go ahead" (LOG 2026-11-09 (t)); the new pair's outcome is the one that governs the other 13 runs, because it is the recipe they use.
6. Honest risk, stated now: a better model has a higher mean, and absolute seed-to-seed differences may grow with it, so |delta| < 2.7 could be harder to meet at the new level (speculative; the old delta of 1.16 was at a mean of about 8). The thresholds stay anyway.

**Level target (the user sets it; STILL OPEN).** The seed-noise rule's floor (2.7) only guards against a degenerate model. The concern that started this amendment is the level (about 7 to 8 against the paper's 84), so a retrain that stays near 8 would pass the locked rule and change nothing. Options:
- L0. None. Only the locked rule (floor 2.7, T 2.7) decides. Tradeoff: a retrain that stays near 8 passes and the scale problem remains; section 4's secondary reading would then carry the caveat.
- L1. Both retrained models reach a mean LIBERO-Plus success of at least X points. Two ways to anchor X (arithmetic on repo numbers): X = 27 makes the 2.7-point harm threshold at most 10% of the model's own mean (today it is about a third of 8); X = about 42 makes it the same share of the model's mean as the paper's drop is of the paper's (5.41 / 84.0, about 6.4%, from the options memo Exit 6). Any other X is the user's call.
- L2. Clean-LIBERO success above Y points. Tradeoff: the clean list is 40 trials per checkpoint (one per task, eval seed 41 only; `results/full-benchmark/seednoise.json`, "clean_records_excluded": 40), so one trial is 2.5 points; too coarse to gate a decision. Clean success is a context number.
- **The lead's recommendation (not a decision):** L1 with X chosen by the user, the lead leaning to X = 27 as the middle value (low enough that a small from-scratch model might reach it, high enough that the harm threshold stops being a third of the score; whether it is reachable is speculative, there is no data on it), and L2 as context only. Reason: without a level target the amendment can spend about 150 to 170 GPU-h and end exactly where it started.
- **If the level target is not met,** the next step is a user decision, not another silent retrain. Options to list then: (i) proceed with the other 13 runs at the new recipe anyway, with the low level stated as the main caveat and the section 4 reading used; (ii) one more pre-registered, larger budget (needs a new amendment and its own sign-off); (iii) stop the proxy and rely on the perturbed-frames test (options memo Exits 1C and 2C); (iv) a higher-fidelity replication as its own proposal (options memo Exit 4A).

#### 4. The scale reading (options memo Exit 6, 6A and 6B), fixed now

The locked absolute rule stays decisive (memo Exit 6, "What stays fixed": "The thresholds are not changed. The locked absolute rule decides the pre-registered branch (calibration pass or fail, 'substantial proxy effect' or not, the seed-noise verdict and the floor). Nothing below can turn a failed branch into a pass or the reverse."). Added, labelled **secondary, post hoc scale reading, does not decide the branch** (the memo's recommendation: "6A with 6B ... with the choice of 6A fixed now"):
- **6A, relative reading.** Report each drop as a share of the proxy's own DINOv3 mean, next to the absolute value, with the same per-seed breakdown. For scale, the paper's D+R drop is about 6.4% of its DINOv3 score (arithmetic: 5.41 / 84.0). It is not a test; a ratio is unstable when the proxy's mean is small.
- **6B, place the proxy on the paper's scale.** One sentence in any report: how far below the paper's DINOv3 and VAE-only scores the proxy sits, and that the thresholds were set for the paper's range.
- **Missing power argument (memo Exit 6, "Acknowledged limit", restated).** The plan has a false-alarm argument for the 3-seed design (the draft seed-noise section above, option 1: a pure-noise 3-seed average reaches the harm line about 4% of the time, labelled speculative there) and says 3 seeds are a screening result ("Three seeds are a small screening result, not an estimate of training variance"). It has no detection-power argument: nowhere does it say how often a real drop of the paper's size would be seen with 3 seeds and our noise. So "the proxy does not show a substantial effect" cannot be read as evidence of no effect. State this in any write-up.
- One flagged departure from the memo's wording: the memo said 6A should be fixed "before the proxy's mean is read". The pilot-scale means (8.3 and 7.2) have already been read, but the means of the recipe that will actually be used for the 15 runs have not. 6A is fixed now, before those are read; the earlier reading is disclosed here.

#### 5. Consequences for the other 13 runs

- Not started before the re-check. Nothing of the 13 is trained or submitted until section 3's rule has been applied and the user has the result.
- If the amendment is accepted, the 13 remaining runs (`dinov3` seed 2, and three seeds each of `none`, `c-radiov4`, `dinov3+c-radiov4`, `dinov3+siglip2`) use the new recipe, so that the recipe is uniform across all 15, as the PLAN requires ("Same everything across configs except the encoder set", Proxy design). The retrained `dinov3` seeds 0 and 1 are the other two of the 15. Round 1's two original `dinov3` models and their evals are kept as a recorded pilot-scale result: not part of the final analysis, not deleted. Their results stay where they are: LOG 2026-11-09 (t), `results/full-benchmark/rollouts/eval-dinov3-s1/` and `.../eval-dinov3-s0/`, `results/full-benchmark/seednoise.json`, and the run directories `runs/dinov3-s1` and `runs/dinov3-s0-a40` on the PVC (not deleted; deleting data is gated).
- No code is changed by this amendment; options A and B need only flags (`__EXTRA__`), as far as read.
- **Lines superseded if accepted** (bracketed dated notes; the original lines are not edited): [2026-11-09 (c), if accepted: RUN.md Step 10, "with defaults (10 epochs, batch 128)", is replaced for this round by the chosen recipe; RUN.md itself is not edited]. [2026-11-09 (c), if accepted: PLAN "Amendment 2026-11-06", "Training order (user, 2026-11-03)", stays as history; the roles of final seeds 0 and 1 for `dinov3` move to the retrained pair]. [2026-11-09 (c), if accepted: PLAN "Seed-noise rule (locked 2026-11-06)" stays locked and is applied a second time, to the retrained pair, per section 3].

#### 6. Compute and gates

All arithmetic and speculative unless marked; eval figures are the options memo's 64 to 70 GPU-h per checkpoint.

| Option | Training, 2 runs | Re-eval, 2 checkpoints | Re-check total |
|---|---|---|---|
| A1, 20 epochs | about 16 GPU-h (about 8 h each) | about 128 to 139 GPU-h | about 144 to 155 GPU-h |
| A2, 40 epochs | about 32 GPU-h (about 16 h each) | same | about 160 to 171 GPU-h |
| B1, 16 layers | about 16 GPU-h (about 8 h each) | same | about 144 to 155 GPU-h |
| B2, hidden 768, 6 heads | about 20 to 32 GPU-h (10 to 16 h each) | same | about 148 to 171 GPU-h |

- The other 13 runs at the new recipe: training about 13 x (8 h for A1 or B1, 16 h for A2, 10 to 16 h for B2) = about 104 to 208 GPU-h, against about 52 at the current recipe (13 x 4 h); evaluation about 13 x 64 to 70 = about 830 to 910 GPU-h, unchanged by the choice (arithmetic). The full 15-checkpoint evaluation is the 959 to 1044 GPU-h already in "Amendment 2026-11-06".
- Gates (verified, `STANDARDS.md` and `RUN.md`): 8 GPUs at once. The re-check trains 2 pods at once and then evaluates at most 8 pods at once. 20 Jobs per sweep: about 3 to 6 Jobs for the re-check. Stop 5, "one training run > 48 h": the longest option is about 16 h (speculative), so it is not close. The user's override of RUN.md stop 5 for the whole stage ("Amendment 2026-11-06", "Time limit") still applies; it is not needed for the retrain itself. Training Job deadlines follow RUN.md (about 2x the estimate plus expected queue; queues of 3 to 9 hours were seen in round 1, LOG 2026-11-06 (a), (c)).
- Queue facts: evaluation is L4-only (Amendment 2026-11-07). Round 1's two evals ran from the first submit on 2026-11-06 (LOG (d)) to the last shard on 2026-11-09 (LOG (r)), so the re-check is better planned in days than in the 16 to 17 hours that 8 GPUs would give on paper (128 to 139 / 8, arithmetic). Training pods use the original GPU list minus the A40 and A6000 entries that match nothing (Amendment 2026-11-06 (b)); the named `nvidia.com/a40` route needs the user's OK each time.
- Changing the recipe and spending this compute is a protocol-changing step, so it needs the user's sign-off before any Job (PLAN "Gates"; `_standards/05` verification gates).

#### 7. For the user to decide (all STILL OPEN)

1. **Which option, if any:** A1 (20 epochs), A2 (40 epochs), B1 (16 layers), B2 (wider), C, or none. STILL OPEN. (The lead's reading is in section 2; no pick is made here.)
2. **The level target:** L0 none, L1 with X, or L2 clean-LIBERO as context; and the value of X. STILL OPEN.
3. **Go for the re-check:** whether to submit the retraining Jobs (the evals follow), and the spend of about 150 to 170 GPU-h in section 6. STILL OPEN.
4. **That the original DINOv3 pair becomes a recorded pilot-scale result and the 13 runs use the new recipe.** STILL OPEN.
