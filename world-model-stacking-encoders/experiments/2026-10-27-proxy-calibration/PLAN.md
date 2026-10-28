# Proxy-policy calibration round (decision 003)

**Status:** step 1 done (open items resolved below); steps 2–3 (code, local smoke test) in progress 2026-10-27. No Jobs.

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
