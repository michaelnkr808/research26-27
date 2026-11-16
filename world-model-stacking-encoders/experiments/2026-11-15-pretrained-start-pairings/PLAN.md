# Plan: stage 2, encoder pairings on the 8-layer pretrained-start model (channel-wise concatenation on the VAE grid)

Written 2026-11-15 by an agent (planning worker) as a design for the user to sign off. Design only: no code, no Job or pod file, nothing run, no cluster or paid-compute action, nothing downloaded. Claims are labelled verified (checked against the file and line, or the paper page, named), recalled (from memory or from a file's own recalled note, not checked here), or speculative (reasoning or arithmetic not yet measured). Predictions are speculative by definition. "The paper" is liu2026what (arXiv 2610.13024v1), read as a `pdftotext` conversion of the user's PDF (sha256 `211c647a...bf63024`) on 2026-11-15; page numbers are the PDF's printed ones, which equal the PDF page index in that conversion (verified: the printed footer of each page read here matches its index). "Fast-WAM" is Fast-WAM at commit `7faa711`, read from the extracted tarball in the session scratchpad (read only, nothing executed). Paths are relative to `world-model-stacking-encoders/` unless they start with `../`. The stage 1 plan is `experiments/2026-11-15-pretrained-start-baseline/PLAN.md`, cited as "stage 1 PLAN".

Once the experiment starts, which here means the first stage 2 training run on the cluster or a paid provider, this design is locked and changes become dated amendments at the bottom (`../../../_standards/04` line 12). The decision rule and its threshold option are locked earlier, at the user's sign-off (see "Decision rule").

## In plain words

Stage 1 trains the model with no extra encoder and checks that it can do the robot tasks at all. Stage 2 is the actual experiment: it trains the same model several more times, each time feeding it extra features from frozen image encoders the way the paper did (stacked as extra channels on the same grid as the video model's own latents). There are four main versions: no extra encoder (the stage 1 model), DINOv3 alone, DINOv3 paired with SigLIP 2 (a pairing the paper says helps a little), and DINOv3 paired with C-RADIOv4 (the pairing the paper says hurts). Each version is trained more than once with different random seeds, because the paper trained each version only once and so could not tell a real drop from luck. Every trained model is first tested on the normal LIBERO tasks to make sure it works, then on LIBERO-Plus, the harder perturbed benchmark where the paper saw the drop. The rule for deciding "yes, pairing DINOv3 with C-RADIOv4 really hurts on our model", "no it doesn't", or "can't tell" is written here before anything runs, so no result can move it.

## Question

On the 8-layer pretrained-start world action model of stage 1 (Fast-WAM's `FastWAMJoint`, video expert loaded from Wan 2.2 TI2V-5B blocks 0-7, action expert derived from those blocks by Fast-WAM's recipe, the stage 1 recipe throughout), with frozen encoder features added by channel-wise concatenation on the VAE latent grid:

1. **Primary:** is LIBERO-Plus success with DINOv3 + C-RADIOv4 lower than with DINOv3 alone by more than training-seed noise, as the paper reports (78.59 against 84.0, a drop of 5.41 points; verified, `literature/liu2026what-table10-composition.csv` line 10 and `literature/liu2026what-table1-single-source.csv` line 4, arithmetic)?
2. **Contrast:** does the pairing the paper reports as helpful (proposed: DINOv3 + SigLIP 2, 85.83, +1.83 over DINOv3; verified, table 10 CSV line 5, arithmetic) avoid that drop on the same model?
3. **Calibration:** does DINOv3 alone beat no extra encoder on LIBERO-Plus, as in the paper (84.0 against 74.7; verified, table 1 CSV lines 3-4)? If our model does not use DINOv3 the way the paper's does, the answer to 1 is not read as a reproduction or a non-reproduction.

This is stage 2 of decision 006 part (c) (`decisions/006-research-question-and-done.md` line 15, verified) under the user's research question, "Why do some encoder pairings work while others make it worse, in the context of VAE-grid feature concatenation?" (decision 006 line 11, verified). Stage 2 does not explain why; it establishes whether the pattern to be explained exists on our model, with training-seed noise measured. The why is stage 3 (fusion variants) and the cheap frozen-feature results (stage 4).

## What exists

**Stage 1 (verified, stage 1 PLAN and its `LOG.md`, `README.md`):** design signed off by the user 2026-11-15 (stage 1 PLAN "Amendments", lines 239-265); pass bar option 4 locked (B >= 90 pass; 36 <= B < 90 conditional pass that needs a LIBERO-Plus total of at least 27 under the full protocol; B < 36 fail; stage 1 PLAN line 245). Code merged: `src/pretrained_train.py` and `src/action_expert_init.py` (PR #39), `rollout.py --timing-only` (PR #38). Nothing has been run on the cluster (stage 1 `LOG.md` line 20, verified; README line 5). So no stage 1 number exists, and every stage 2 quantity that depends on one is a blank below.

**Sizing check (verified, `experiments/2026-11-12-pretrained-start-sizing/LOG.md` lines 26-56):** measured on one Modal L40S, configs `none` and `dinov3` only. Neither a two-encoder config nor SigLIP 2 or C-RADIOv4 alone was measured. `none`: 3.907 s per optimizer step of 128, 38.15 GiB peak at micro-batch 8, 1.83 GiB under the 0.9 limit, 23.54 L40S-hours per 10-epoch run. `dinov3`: 3.994 s per step (+2.2%), 38.70 GiB peak (+0.55 GiB), 1.28 GiB under the limit, 24.07 L40S-hours. Loading with DINOv3: 3 tensors "partly loaded" (the widened patch embedding and head), loaded parts re-read from the Wan file with max abs difference 0 (line 31-32). Not included: evaluation, data stalls, checkpoint writes, startup, failures (line 41).

**The paper's numbers this stage tests** (verified, the two CSVs; all LIBERO-Plus totals, mean of evaluation seeds 41, 42, 43 of **one trained checkpoint per row**, paper App. C.2, p17, and App. E.3, p22):

| Configuration | LIBERO-Plus total | Source |
|---|---|---|
| no extra encoder (VAE only) | 74.7 (sd 0.5) | table 1 CSV line 3 |
| DINOv3 | 84.0 (sd 0.4) | table 1 CSV line 4 |
| SigLIP 2 alone | 83.1 (sd 0.4) | table 1 CSV line 9 |
| C-RADIOv4 alone | 83.5 (sd 0.6) | table 1 CSV line 12 |
| DINOv3 + SigLIP 2 | 85.83 (sd 0.28) | table 10 CSV line 5 |
| DINOv3 + C-RADIOv4 | 78.59 (sd 1.01) | table 10 CSV line 10 |

The pair is below both of its members (78.59 against 84.0 and 83.5; arithmetic). The paper reports no clean-LIBERO number for any pair (Table 10 is LIBERO-Plus only; `literature/notes/liu2026-what-visual-representations-wams.md` line 34, verified there). Note on "the helpful pairing": in Table 10 the largest gain over DINOv3 is DINOv3 + UniTok (86.12, +2.12; table 10 CSV line 9, arithmetic), not DINOv3 + SigLIP 2; UniTok has no loader in `src/extract_features.py` (line 197, a stub, verified) and no cache, so DINOv3 + SigLIP 2 is the largest gain among cached encoders (Open item a). The Table 1 DINOv3 row and the Table 10 rows are separate training runs, so the 5.41 and 1.83 differences each compare two single trained checkpoints (speculative consequence: the paper cannot separate them from training-seed noise; `STANDARDS.md` line 37 lists "Single-checkpoint published labels" as a pitfall, verified).

### Cache check, by reading only

Question: can the existing caches under `/mnt/proxy/feats/` (PVC `stacking-encoders-proxy`) feed `pretrained_train.py` unchanged for `dinov3`, `siglip2`, `c-radiov4` and their pairs? Answer by reading: **yes, as far as the code and `data/README.md` show; not verified on the cluster.** Each point:

- **Folder names match config names** (verified). `pretrained_train.py` builds `P.Data(args.feats, ..., P.config_sources(args.config), args.t5)` (line 446); `config_sources` splits on `+` (`proxy_wam.py` lines 104-106); `Data._path` reads `<feats>/<source>/<suite>/ep<id>.npy` (lines 214-215). The caches are `feats/dinov3`, `feats/c-radiov4`, `feats/siglip2` (`data/README.md` lines 71-73). Rollout's `ENCODERS` and `ENC_DTYPE` use the same three names (`rollout.py` line 82).
- **Shapes and dtype** (verified by reading). Caches are (frames, 14, 28, D) float16, one row per frame (`extract_fastwam.py` docstring lines 9-10; `data/README.md` lines 71-73: DINOv3 D = 1024, C-RADIOv4 D = 1152, SigLIP 2 D = 1152). `Data.assemble` places them with `feature_steps_cached` on exactly that layout (`proxy_wam.py` lines 239-267, 160-163). Input widths: 48 + 1024 = 1072 (`dinov3`), 48 + 1152 = 1200 (`siglip2` or `c-radiov4` alone), 48 + 1024 + 1152 = 2224 (`dinov3+siglip2` and `dinov3+c-radiov4`) (arithmetic).
- **Normalization statistics** (verified by reading). Every encoder episode file is written after its `.stats.npy` (count, sum, sum of squares per channel; `extract_fastwam.py` lines 174-180), and `Data` sums them over all training episodes (`proxy_wam.py` lines 196-206). That the stats file exists for every episode of every encoder on the PVC: **not verified** (cluster).
- **Episode coverage.** The VAE cache was checked to hold every episode of all four suites (2026-11-01; `data/README.md` line 74). The encoder caches are recorded as "done" and SigLIP 2 as "spot-checked" (lines 71-73); a full per-episode count for each encoder against the meta files is **not recorded** in the files read. `Data.__init__` asserts VAE presence only (line 192-193); a missing encoder file would fail at the stats read (line 200) or later. **Not verified** (cluster).
- **Same extraction code as rollouts use** (verified this session by `git diff 2d0bcbd HEAD`): `src/extract_features.py` and `src/fastwam_data.py` are unchanged since commit `2d0bcbd`, from which the DINOv3 and C-RADIOv4 caches were made; `extract_fastwam.py` changed only in I/O (prefetch thread, `--out-name`, `--stage-dir`; commits `439e52f`, `dc30417`), whose VAE output was byte-identical to the earlier code (`data/README.md` line 74). The SigLIP 2 cache came from `439e52f` (line 73). So `rollout.Featurizer` (lines 213-238) computes features with the same encoder calls and the same grid resize as the training caches.
- **Encoder variants in the caches** (verified, `data/README.md` lines 71-73 and `src/extract_features.py`): DINOv3 `facebook/dinov3-vitl16-pretrain-lvd1689m` (line 76), computed fp32; C-RADIOv4 `nvidia/C-RADIOv4-SO400M` at pinned revision `c0457f5` (lines 115-118), computed fp16; SigLIP 2 **so400m** `google/siglip2-so400m-patch14-384` (line 96), computed fp16. The SigLIP 2 variant is a fidelity question (row 1 of the fusion table, Open item f). The `_meta.shard*.json` files on the PVC record `hf_id` and resolution (`extract_fastwam.py` lines 152-158); reading them is a cluster step.
- **Disk.** The PVC had 805G used and 220G free after the Wan 2.2 download (`data/README.md` line 79, verified). Stage 2 adds one `ckpt.pt` of about 20 GB and one `model.pt` of about 6.7 GB per run (stage 1 PLAN lines 83, 87, arithmetic there), and stage 1 keeps `ckpt.pt` until evaluation has loaded `model.pt` once and the user approves deleting it (stage 1 PLAN line 87). For 11 or 17 new runs that is about 294 or 454 GB (arithmetic, 26.7 GB x runs), more than 220G. Open item g.
- **Read load during training** (speculative arithmetic). A two-encoder run reads both caches once per epoch: about 223 + 251 = 474 GB per epoch (`data/README.md` line 67 sizes), about 4.7 TB over 10 epochs, about 55 MB/s averaged over a 24-hour run. The proxy measured CephFS at about 30 MB/s for one reader and about 300 MB/s for 16 (`proxy_wam.py` lines 293-295, from LOG 2026-11-01 (g)); the default 32 read threads (`pretrained_train.py` line 390) should keep up, but the sizing run measured host batch assembly only on a 16-episode slice (sizing LOG line 41). Measured in the first minutes of the first pair run (data wait is logged per step, `pretrained_train.py` line 311).

### What code exists and what is missing

What already supports extra encoders (verified by reading; none of it has been run for a pair on the pretrained model):
- `pretrained_train.py` takes `--config` with `+` (line 382), builds the model at `data.in_dim` (`build_and_init`, lines 171-174), and `pretrained_sizing.load_wan` widens `patch_embedding.weight`, `head.head.weight` and `head.head.bias` (the `WIDENED` names, line 64), copying the VAE slice from Wan and zeroing the extra slice (lines 227-235), then re-reads every loaded tensor from the file and checks max abs difference 0 and an all-zero extra slice (lines 147-176). The draws cover all channels (`draw_step`, lines 88-96, shape from `lat_shape_of`, line 99-100). The first-batch report gives per-source channel statistics (lines 225-229). `run_identity` includes `config` and `in_dim` (lines 410-417). The sizing run used this load path on real data for `dinov3` (sizing LOG line 31).
- `action_expert_init.init_action_expert` reads `model.video_expert.state_dict()` and skips `action_encoder.` and `head.` (lines 24, 118-134); the action expert has no patch embedding, so the widening should not touch it (speculative until a test shows it).
- `rollout.py`: `Featurizer` runs each source's encoder on the current frame and resizes its grid exactly as extraction does (lines 213-238); `Policy.act` z-scores each source with the training statistics stored in `model.pt` and concatenates in the config's order (lines 314-323); `Policy.load` builds the model from `cfg["in_dim"]` and `cfg["model_kw"]` (lines 292-298). The proxy's rollouts ran `dinov3` and the 500-step `dinov3+c-radiov4` pilot through this path (proxy PLAN line 166, verified).

What the implementation session must add (by PR, with CPU tests; not written here):
1. **Tests with encoder sources for `pretrained_train.py`.** The current test builds its fake cache with no sources (`test_pretrained_train.py` line 61, `fake_cache(..., [], ...)`, verified). Needed: a tiny run with one and with two fake sources: the widened load (VAE slice equals the fake Wan file, extra slices zero), the derived action expert unchanged by the widening, accumulation 8 x 16 and 4 x 32 against one batch of 128 with sources (same tolerance as PR #39, 1e-5 relative), resume, and `rollout.Policy.load` building the widened model from its `model.pt`.
2. **A training-versus-rollout feature parity test**: the same uint8 frames through `extract_fastwam.encoder_episode` and through `rollout.Featurizer` give the same float16 grid for each encoder (if an existing test already covers this, the session cites it instead).
3. **A cache pre-check script** for the cluster (reading only): for each encoder and suite, every meta episode has an `.npy` of the right length and shape and a `.stats.npy`; the `_meta.shard*.json` `model` and `resolution` fields; free space on the PVC.
4. **The stage 2 analysis**: `lp_eval.py`'s `paired` and `verdict` are written for the proxy's locked rule with fixed config names (lines 211-258, verified) and its `seednoise` for two `dinov3` runs (line 505). A new subcommand computes this PLAN's statistic and applies the locked rule word for word (per-checkpoint totals, paired per-seed differences, the threshold of the chosen option, the outcome), with synthetic tests for every outcome branch. The clean tally for the gate (stage 1's tally script or `lp_eval.py` addition) is reused.
5. **Job manifests and RUN steps** for the training waves and the evaluation shards, following `../../../_standards/08`.
6. **Only if the user chooses SigLIP 2 L (Open item f):** `extract_fastwam.py` passes no variant to the encoder loader (`ENCODERS[args.source](args.device, dtype)`, line 151, verified), so a variant flag, a separate cache folder name, and matching `ENC_DTYPE` and `Featurizer` entries in `rollout.py` are needed, plus an extraction Job.

## Design

### Configurations
All on the stage 1 model and recipe, unchanged except the input width (stage 1 PLAN "Design"; its departures 1 to 8 and its amendments carry over). Channel order is the config string's order after the VAE (`proxy_wam.py` lines 221-227).
- `none`: stage 1's baseline. Seed 0 from stage 1; seeds 1 and 2 trained here (the user's stage 1 answer, stage 1 PLAN line 248, verified).
- `dinov3`: DINOv3 alone.
- `dinov3+siglip2`: the helpful pairing, pending Open item a.
- `dinov3+c-radiov4`: the harmful pairing.
- Pending Open item b: `siglip2` and `c-radiov4` alone.

### Fusion as specified vs as implemented

Channel-wise concatenation, point by point. "Paper" quotes the PDF text; "Fast-WAM" only where it bears (Fast-WAM itself has no external encoders); "src" is our code. Verdicts: **match**, **departure**, or **not stated** (the paper does not say; our code makes a choice, named here).

| # | Item | Paper (page) | Fast-WAM | src (file, lines) | Verdict |
|---|---|---|---|---|---|
| 1 | Encoder layer and tokens | "token features from the final layer of each encoder" (App. A, p14); "We primarily use large (L) variants" (p14); DINOv3 on LIBERO: "Each view has a 14 x 14 grid of 1,024-dimensional features" (App. D.3, p21). Checkpoints not named. | n/a | `extract_features.py`: DINOv3 ViT-L/16 `last_hidden_state`, CLS and register tokens dropped (lines 76-87); SigLIP 2 so400m vision tower `last_hidden_state`, all patch tokens (lines 96-106); C-RADIOv4-SO400M `spatial` output (lines 115-130) | Final-layer patch tokens: **match** (that each model's output is taken after its final norm is recalled, not checked). DINOv3 ViT-L: **match** (consistent with p21's 14 x 14 x 1024). C-RADIOv4: variant **not stated**; no L variant exists, SO400M or H (recalled, `literature/encoder-weights.md` line 35). SigLIP 2: **departure** from "primarily large (L)": an L variant exists (`google/siglip2-large-patch16-384` was used for the libero_goal caches, `data/README.md` line 53, verified) and our proxy cache is so400m. Open item f. |
| 2 | Input resolution per encoder | Not stated. Observations are "224 x 224" per view (Table 7, p18); encoders "extract representations from the RGB observations available to the base WAM" (Sec. 3.2, p4). | Frames resized to 224 per camera, deterministic (`fastwam-reading.md` Q2) | Encoders get Fast-WAM's 224 frame as uint8 (`extract_fastwam.py` lines 18-19, 98-101) and resize it bicubically to their own size (`extract_features.py` line 52): DINOv3 224 (line 77), SigLIP 2 384 from its processor (line 98), C-RADIOv4 448 (line 116) | **Not stated.** Our choice: each encoder's native input size, by bicubic upsampling of the 224 frame. |
| 3 | Alignment to the VAE grid | "spatially align external features with the VAE latent grid" (Sec. 3.3, p5; App. B.2, p15). Method not stated. "spatial resizing is bilinear" is stated for REPA's teacher only (App. B.2, p14). | VAE latent grid 14 x 28 for the 224 x 448 two-camera frame | `fastwam_data.grid_to_vae` (lines 101-107): bilinear, `align_corners=False`, no antialias; DINOv3 14 x 14 per view, no resize; SigLIP 2 27 x 27 to 14 x 14; C-RADIOv4 28 x 28 to 14 x 14. The docstring (line 102) cites App. B.2 as if the paper stated bilinear for this step; the paper states it for REPA only. | **Not stated.** Our choice (borrowed from REPA). For DINOv3 no resize happens, so DINOv3 matches whatever the paper did if its grid was the same 14 x 14 per view (p21). That a factor-2 bilinear downsample with `align_corners=False` equals a 2 x 2 average (C-RADIOv4), while 27 to 14 drops part of the grid without antialias (SigLIP 2), is recalled behaviour of `F.interpolate`, not checked. |
| 4 | Layout of the two camera views | Not stated for features. LIBERO uses 2 RGB views (Table 7, p18). | Cameras concatenated horizontally to 224 x 448 before the VAE (`fastwam-reading.md` Q2 step 3, `libero_2cam.yaml:36`, verified there) | Per-camera 14 x 14 grids concatenated along the width, agentview then wrist (`extract_fastwam.py` lines 99-107; `rollout.py` lines 234-237), so each feature cell sits on the VAE cell of the same camera region (`proxy_wam.py` line 64, GRID 14 x 28) | **Not stated.** Our choice, following the VAE's layout. |
| 5 | Time matching | Not stated for channel-wise concatenation. Stated for REPA: "the observed temporal slice is preserved, and the eight future slices are averaged into two groups of four" (App. B.2, p14). Frames: 9 per view, offsets 0, 4, ..., 32 (Table 6, p17). | VAE: 9 frames to 3 latent steps (4x causal compression; `proxy_wam.py` line 63) | Latent step 0 = features of frame t; step 1 = mean of frames t+4 to t+16; step 2 = mean of t+20 to t+32 (`proxy_wam.py` lines 125-134, 160-163; docstring lines 18-20 call it our reading) | **Not stated.** Our choice (borrowed from REPA). Each latent step averages the same frames the VAE folds into that step (speculative reading of the VAE's 4x grouping). |
| 6 | Projection and normalization before combining | Input and output projections "expanded to accommodate the added features" (p15; p5). No per-source projector, LayerNorm or scaling is described for channel-wise concatenation (LayerNorm is stated for token-wise concatenation, p15). "Source-specific processing ... held fixed between training and evaluation" (App. E.1, p22) is named, not described. App. D.3 speaks of "raw current-frame DINOv3 features" and of "each mechanism's transformations of the raw features" (p21). | n/a | No per-source projection; the widened patch embedding is the only projection (`pretrained_sizing.py` lines 64, 128-143). Each source z-scored per channel with training-set mean and std, pooled over frames, positions and both cameras (`extract_fastwam.py` lines 174-180; `proxy_wam.py` lines 196-206, 347-353; at rollout `rollout.py` lines 287-288, 319-321). Channel counts stay unequal (1024, 1152, 1152). | Projection: **match** (one expanded input projection, as p15 reads; that it is one shared projection is the paper note's inference, line 45 there). Normalization: **not stated**; z-score is our choice (Open item h). |
| 7 | Widening while keeping the pretrained VAE weights | "expanded ... while retaining the pretrained VAE weights" (App. B.2, p15). Initialisation of the new slices not stated. | No widening in Fast-WAM | VAE input columns of `patch_embedding` and VAE output rows of `head.head` copied from Wan; extra columns and rows (weight and bias) set to zero (`pretrained_sizing.py` lines 227-235; `INIT_NOTE`, lines 66-69); checked against the file (lines 147-176; sizing LOG lines 31-32: max abs difference 0) | Retaining the VAE weights: **match**. Zero for the new slices: **not stated**, our choice (consequence: at step 0 the model ignores the features and predicts zero for them; sizing LOG line 52, verified). |
| 8 | Features as clean current inputs and noised future targets | "Observed features remain clean, whereas future VAE latents and external features are jointly noised and predicted. At inference, future features are initialized from noise and participate in joint denoising; only the VAE latents are decoded." (p15); Eq. 8, "The current state z~0 remains clean" (p22). Shared timestep and independent feature noise are stated for token-wise concatenation and the feature expert (p16); for channel-wise concatenation one tensor is "jointly noised" (p15). | First latent frame kept clean and out of the loss (`fastwam-reading.md` Q4, `fastwam.py:541-542, 574-577`) | One noise tensor over all channels, one video timestep per sample, latent step 0 reset to clean, loss on steps 1-2 only (`proxy_wam.py` lines 404-413, 421); at inference step 0 is reset to the observation after every step (lines 450-464) | **Match.** |
| 9 | Per-source feature loss weights | Eq. 9: L_aux = sum over sources of L_FM(f^k_{1:T}) (p22); feature losses weight 1.0 unless stated (App. B.2, p16); L = L_WAM + lambda_aux L_aux (Eq. 2, p5); L_FM is a "modality-specific mean-squared error" (p4); losses on "valid future targets" only (p22). Whether a source's L_FM is a mean over its own channels is not stated. | One mean over all video channels (`fastwam-reading.md` row j, `fastwam.py:427`) | Per-sample mean over each source's own channels and valid future steps, times the scheduler weight, batch mean, then summed with the VAE term and the action term, all weight 1 (`proxy_wam.py` lines 424-433) | Sum of per-source terms at weight 1: **match** to Eq. 9. Per-channel mean inside a term: **not stated**, our reading of "modality-specific". A **departure** from Fast-WAM's single mean for any fused config (identical only for `none`). Consequence: the 48 VAE channels and each encoder's 1,024 or 1,152 channels each count as one loss term. |
| 10 | Source order | "in a fixed source order" (Eq. 8, p22); the order itself not stated | n/a | VAE first, then the config string's order (`proxy_wam.py` lines 104-106, 221-227) | **Match** (fixed); the order used is **not stated**. |
| 11 | Encoder compute precision | Not stated | n/a | DINOv3 fp32 (fp16 overflowed, `data/README.md` line 61), C-RADIOv4 and SigLIP 2 fp16; all stored float16; rollouts use the same dtypes and round through float16 (`rollout.py` lines 82, 231-237) | **Not stated.** Train and rollout consistent. |
| 12 | Training and rollout image path | Not stated for encoders | Fast-WAM trains on 512 to 224 resized frames and evaluates on a 256 render rotated 180 degrees and center-crop-resized to 224 (proxy PLAN line 51, verified there) | Same as Fast-WAM for the VAE and for the encoders (`rollout.py` docstring lines 7-16) | Fast-WAM's own train/eval difference, accepted for the proxy (proxy PLAN line 54); carried over. |

**Where our code and the paper disagree (summary):** (a) SigLIP 2 so400m against the paper's "primarily large (L)" (row 1); (b) the `grid_to_vae` docstring cites App. B.2 for bilinear resizing, which the paper states for REPA only (row 3; the code is a choice, the citation overstates it); (c) per-source loss as a per-channel mean (row 9) is a departure from Fast-WAM and only a reading of the paper. Everything else is either a match or a choice where the paper is silent. The paper says its channel-wise design follows DreamWorld (Tan et al., 2026, arXiv 2603.00466; p15 and p5, verified in the reference list); no file in this repo has read DreamWorld (direction `LOG.md` line 274: "not checked", verified), and it may state the resize, normalization or time matching (speculative). Hand-off step 1.

### Training
- Exactly the stage 1 recipe: Table 6 and 7 settings and departures as in stage 1 PLAN lines 46-76 and its amendments (fp32 weights with bf16 autocast, action shift 1.0 and video shift 5.0, all 1,712 episodes, no validation split, cached latents, global batch 128 by accumulation, draws once per optimizer step keyed by (seed, step)).
- Micro-batch per card by stage 1's rule (the largest of 8 and 4 under 0.9 x the card's memory; stage 1 PLAN line 127). For the pairs this is not measured: the `dinov3` margin was 1.28 GiB, and a linear extrapolation of the measured +0.55 GiB per 1,024 channels gives about +1.2 GiB over `none` for 2,176 extra channels, leaving about 0.66 GiB (speculative arithmetic), so micro-batch 8 may fit on an L40S and may not on a card reporting less memory. Step time at micro-batch 4 is not measured.
- **Pairing across configurations.** All configurations use the same training seeds 0, 1, 2. With the same seed, the data order is identical across configs (`make_plan` depends only on the seed and the episode list, `proxy_wam.py` lines 270-286, verified), but the noise and timestep draws are not (the video noise tensor's shape includes the channel count, so the later draws shift; `pretrained_train.py` lines 88-96, verified), and the fresh parameters may differ (speculative). "Paired by training seed" therefore means shared data order only.
- Final checkpoint only (step 21,696). No intermediate checkpoint is evaluated.

### Evaluation (the stage 1 protocol, carried over)
- **Clean LIBERO first:** 50 rollouts per task, 40 tasks, 2,000 rollouts, eval seed 41, four-suite average (stage 1 PLAN line 101; paper App. C.2, p17).
- **Then LIBERO-Plus:** all 10,030 episodes under each of evaluation seeds 41, 42 and 43 (paper App. E.3, p22; the user's choice for the proxy, proxy PLAN lines 172-175, and for stage 1's option 4, stage 1 PLAN line 245). Inference settings as stage 1 (10 steps, replan every 10, 30 wait steps, no ensembling, text guidance 1.0, binarized gripper; paper App. E.3, p22).
- **On L4 cards only.** A move off L4 is a stop for the user (stage 1 PLAN Open item 11, confirmed, line 259). Whether an L4 holds the model plus the encoders of a pair (DINOv3 ViT-L in fp32 and C-RADIOv4-SO400M with fp32 weights under fp16 autocast, `extract_features.py` lines 119-130) and keeps the GPU above 40% is measured by a timing-only smoke per configuration (stage 1's mode, PR #38), not assumed.

### Comparison statistic (defined before any result)
- For configuration c and training seed s: **Y(c, s)** = the mean over evaluation seeds 41, 42, 43 of that checkpoint's LIBERO-Plus success rate, each rate being total successes over the 10,030 episodes (paper App. C.2, p17, "Overall LP success is the total number of successes divided by the total number of evaluated episodes"; App. E.3, p22, "The mean is computed from the three full-benchmark success rates"), in percentage points.
- Across training seeds: **Ybar(c)** = mean of Y(c, 0), Y(c, 1), Y(c, 2); its spread is the sample SD over the three seeds.
- Paired difference for configs A and B: **d_s(A - B) = Y(A, s) - Y(B, s)**, and **dbar(A - B)** = their mean over s.
- **Training-seed noise, sigma_hat:** the pooled within-configuration SD of Y(c, s) over all configurations that have three seeds (pooled variance = sum over c of sum over s of (Y(c, s) - Ybar(c))^2, divided by the sum over c of 2, i.e. 2 degrees of freedom per configuration; 8 with four configurations, 12 with six). If seeds were independent across configs, the SD of dbar for 3 seeds is sigma x sqrt(2/3) = 0.816 sigma (arithmetic).
- **Evaluation-seed noise is small by comparison:** the binomial SE of one checkpoint's total over 30,090 rollouts is at most about 0.29 points (sqrt(0.25 / 30,090), arithmetic). It is reported but not part of the rule.
- Clean-LIBERO statistic for the gate: **B(c, s)** = four-suite average on the 2,000 clean rollouts.

## Dependence on stage 1

Stage 1's locked rule decides whether stage 2 starts (stage 1 PLAN lines 163-171):
- **Stage 1 passes (B >= 90).** Stage 2 runs as designed. `none` seed 0 has not been scored on LIBERO-Plus in stage 1 (its Open item 6: only as option 4's condition, confirmed, stage 1 PLAN line 259), so its full LIBERO-Plus evaluation is the first evaluation of stage 2.
- **Conditional pass (36 <= B < 90 and the seed 0 LIBERO-Plus total >= 27).** Stage 2 runs; `none` seed 0 already has its LIBERO-Plus total and is not re-evaluated. Every report states the clean gap to the paper's 97.9. At a LIBERO-Plus level well below the paper's 74.7, a fixed 2.7-point threshold is a larger share of the score; threshold options 2 and 3 below are the ones that account for that.
- **Conditional pass with LIBERO-Plus below 27, or fail (B < 36).** Stage 2 does not start. The stage 1 fail options go to the user (stage 1 PLAN lines 166-171). If the user changes the model or recipe (more layers, longer training, other blocks), this PLAN is amended before it runs, with any number that depended on the old recipe re-derived.
- **Recipe changes made by stage 1 amendments** (for example micro-batch 4 on some card, a different time limit) carry over unchanged; stage 2 changes nothing stage 1 settles.

**Blanks that wait on stage 1** (each filled by a dated amendment below, not guessed):
- Card-hours per training run on the Nautilus cards actually used: ____ (stage 1's measured step time; the sizing figure, 23.54 L40S-hours for `none` and 24.07 for `dinov3`, is the only measured basis).
- Clean evaluation time per checkpoint on L4: ____ (stage 1 smoke and evaluation).
- LIBERO-Plus evaluation time per checkpoint on L4: ____ (stage 1's timing-only LIBERO-Plus shard if run, otherwise stage 2's own shard before submission).
- Stage 1 clean score B(none, 0): ____; its zone: ____; LIBERO-Plus total of `none` seed 0, if scored in stage 1: ____.
- Micro-batch and step time of each pair on the cards used: ____ (first minutes of each config's first run).

## Predictions

Written before any stage 2 or stage 1 result. All speculative.
1. **DINOv3 helps.** dbar(dinov3 - none) is positive in every seed, smaller than the paper's 9.3 points if our model sits below the paper's level. Reason: the paper's largest single effect; the 8-layer model may use features less.
2. **The helpful pairing is about neutral.** |dbar(dinov3+siglip2 - dinov3)| is within about 2 points, sign not predicted; the paper's +1.83 is one checkpoint against another and may be within training noise.
3. **The harmful pairing:** central guess that dbar(dinov3+c-radiov4 - dinov3) is negative, smaller in size than 5.41; whether it clears the threshold is roughly even odds. No measurement supports this; the paper's drop is one trained checkpoint.
4. **C-RADIOv4 and SigLIP 2 alone** (if trained): each between `none` and `dinov3`, close to `dinov3`, as in the paper's Table 1.
5. **Clean LIBERO:** every configuration within a few points of `none`; no configuration fails the clean gate if `none` passes stage 1. The paper's single encoders all sit at 98.0 to 99.1 clean (stage 1 PLAN line 149).
6. **Training-seed noise:** sigma_hat of 1 to 3 points on LIBERO-Plus. No source; the proxy's 1.16-point seed difference was at a mean of about 8 (proxy PLAN line 272) and does not transfer.
7. **Cost:** a pair run takes a few percent longer per step than `dinov3` (+2.2% for one encoder measured), unless it drops to micro-batch 4 or the loader cannot keep up with two caches.

## Decision rule

**The threshold option is the user's decision; it is not chosen here.** Everything else in this section is proposed; the user signs off the whole rule. Once signed off it is locked and changes only by a dated amendment that states whether any stage 2 result had been seen. No verdict is read from a single training seed (see "Order of work").

**Threshold T for "more than seed noise"** (in LIBERO-Plus points; options for the user):
1. **T = 2.7, fixed.** Half the paper's 5.41 drop; the threshold the user locked for the proxy (proxy PLAN line 160). Simple and already chosen once. Does not scale with our model's level or with measured noise.
2. **T = 0.032 x Ybar(dinov3).** Half the paper's relative drop (5.41 / 84.0 = 6.44%, halved; arithmetic), scaled to our DINOv3 level; equals 2.7 at the paper's 84.0. Fair if our model sits much lower; ignores measured noise.
3. **T = max(2.7, 2 x 0.816 x sigma_hat) = max(2.7, 1.63 x sigma_hat).** The proxy's 2.7 as a floor, raised to two standard errors of a 3-seed paired mean when the measured training-seed noise is larger. Uses the one thing the paper could not measure. sigma_hat from 3 seeds per configuration is itself rough (8 or 12 degrees of freedom when pooled).
4. **A one-sided interval:** reproduced only if dbar <= -2.7 and dbar + t x s_d / sqrt(3) < 0, with s_d the SD of the three d_s and t = 2.920 (one-sided 95%, 2 degrees of freedom; recalled value, the implementation checks it). The most conservative; with 3 seeds the interval is wide, so "inside the noise" becomes the likely outcome unless the drop is large.

**The worker's recommendation (for the user to accept or reject):** option 3. It keeps the user's own 2.7 and adds the training-seed noise this stage is designed to measure. If the user prefers one fixed number, option 1.

**The rule** (written with T from the chosen option; d_s and dbar as defined above, all from final checkpoints, full protocol):
- **Calibration (C):** passes if d_s(dinov3 - none) > 0 for all three seeds and dbar(dinov3 - none) >= T.
- **Harm reproduced (H):** d_s(dinov3+c-radiov4 - dinov3) < 0 for all three seeds **and** dbar(dinov3+c-radiov4 - dinov3) <= -T.
- **Harm not reproduced (N):** dbar(dinov3+c-radiov4 - dinov3) > -T/2.
- **Inside the noise (I):** neither H nor N.
- **Contrast (K):** the helpful pairing is "not harmful" if dbar(dinov3+siglip2 - dinov3) > -T/2 (the proxy's non-inferiority form, proxy PLAN line 157). The paper's +1.83 is not required, since it is not established beyond one checkpoint.
- **Below both members (secondary, only if C-RADIOv4 alone is trained, Open item b):** the same H test with `c-radiov4` in place of `dinov3`. Reported; it does not change the outcome below, but "worse than each member alone", the paper's pattern, can only be stated when it holds.
- **Clean gate (G), applied before a checkpoint's LIBERO-Plus evaluation.** Options for the user: G1 (recommended): every checkpoint needs B(c, s) >= 36, stage 1's fail line, as a check that the model works, not as a result; G2: no gate, clean reported as context only (the paper reports no clean numbers for pairs). With G1, a checkpoint below 36 gets the stage 1 fail-branch diagnosis, which uses no test outcomes (code reading, open-loop actions on training windows, training loss against predict-zero, a unit test; stage 1 PLAN line 166). A fault shown that way is fixed by PR and that checkpoint is re-evaluated at most once, both evaluations reported. If no fault is found, the low clean score is a result, reported as such, and the checkpoint is still scored on LIBERO-Plus so the paired statistic has no hole (the worker's recommendation; the alternative, skipping it, saves compute but leaves a missing seed, which the rule then treats as "inside the noise").

**What each outcome leads to** (written before any result):
- **C fails.** The model does not use DINOv3 as the paper's does. The pairing comparisons are reported, but no reproduction or non-reproduction is claimed ("don't interpret", the proxy's wording, proxy PLAN line 158). Options go to the user: deeper model (the mentor's "then increase depth"), or stop stage 2 here and write up what the 8-layer model shows.
- **C passes, H, and K.** The selective harm is reproduced on our model with training-seed noise measured: the pairing the paper reports as harmful hurts, the helpful one does not. Stage 3 (fusion variants) runs on these two pairings with its own PLAN. Per-category LIBERO-Plus results become the first look at why (secondary, not part of the rule).
- **C passes, H, not K.** Pairing DINOv3 with either partner hurts on our model; the paper's contrast does not hold here. Reported as found. Stage 3 is redesigned with the user (the question becomes why pairing hurts at all on this model, not why one pair does).
- **C passes, N.** The paper's drop does not appear on the 8-layer model across three seeds. This does **not** show the paper's result was noise: the 22 missing layers or our silent-point choices (fusion table rows 1, 3, 5, 6, 9) could matter. Options to the user: the same three configs on a deeper model; a fidelity change on a "not stated" row (for example SigLIP 2 L, or no z-score); or stop and report the non-reproduction with these limits.
- **C passes, I.** Inconclusive. Pre-registered now so that it is not a post hoc choice: the user may add training seeds 3 and 4 for `dinov3` and `dinov3+c-radiov4` once (4 runs, about 96 L40S-hours by the sizing basis, plus 4 evaluations), with the same rule applied to all five seeds, T recomputed by the same option from the same formula (with 5 seeds, the factor in option 3 becomes 2 x sqrt(2/5) = 1.26), and the result reported as a two-stage design. If still I, it is reported as inside the noise and the user decides. Whether to allow the extension is part of the sign-off (the worker recommends allowing it).

**Limits stated now:**
- 8 layers in both experts against the paper's 30 (stage 1 departure 1; the 30 is inferred, stage 1 PLAN line 62). The harm might depend on depth.
- The paper's numbers are one trained checkpoint per row with evaluation-seed SDs only (App. C.2, p17). A non-reproduction cannot say whether the paper's drop was training noise; a reproduction does not show the paper's checkpoint was typical.
- Three training seeds per configuration is a small screening result; sigma_hat is rough.
- The "not stated" rows of the fusion table are our choices; any of them could interact with the harm (stage 3's suspects: scale, double counting, grid mismatch; decision 006 line 16).
- The Wan 2.2 variant (TI2V-5B) is presumed, not stated by the paper (stage 1 PLAN line 62).

## Compute and cost

**Where:** training on Nautilus 48 GB-class cards (L40, L40S by `gpu.product`), as the user chose for stage 1 (stage 1 PLAN line 247); the named A40 and RTX A6000 resources only with the user's OK. Evaluation on Nautilus L4 only. No paid compute: "long training stays on Nautilus" (`STANDARDS.md` line 55, verified).

**Run counts** for each combination of Open items b and c. New training runs = (encoder configs x k) + (k - 1) for `none` (its seed 0 is stage 1's). Card-hours are training steps only, arithmetic from the sizing check's measured 23.54 (`none`) and 24.07 (`dinov3`) L40S-hours per run (sizing LOG line 41); pairs and single SigLIP 2 or C-RADIOv4 are not measured. Checkpoints to evaluate include `none` seed 0.

| Open b (configs) | k seeds | New training runs | L40S-hours of steps (arithmetic) | Checkpoints evaluated | Rollouts (clean + LIBERO-Plus) |
|---|---|---|---|---|---|
| no: `dinov3`, two pairs | 2 (needs an override) | 7 | about 165 to 169 | 8 | 256,720 |
| no | **3 (recommended)** | **11** | **about 259 to 265** | **12** | **385,080** |
| no | 5 | 19 | about 447 to 457 | 20 | 641,800 |
| yes: also `siglip2`, `c-radiov4` | 2 (needs an override) | 11 | about 259 to 265 | 12 | 385,080 |
| yes | **3 (recommended)** | **17** | **about 400 to 409** | **18** | **577,620** |
| yes | 5 | 29 | about 683 to 698 | 30 | 962,700 |

Each checkpoint is 2,000 clean plus 30,090 LIBERO-Plus rollouts (32,090). If `none` seed 0 was scored on LIBERO-Plus in stage 1, subtract 30,090 from the last column.

**Evaluation cost: blank until stage 1 measures it.** For scale only, at the proxy's rollout speed (484 episodes per L4 pod-hour, proxy PLAN line 166, a 31.6 M-parameter model), the recommended rows would be about 796 (11-run row) and 1,193 (17-run row) L4 pod-hours, about 100 and 150 hours of wall time on 8 L4 pods (arithmetic). The new model has about 53 times more parameters and adds encoder passes per call, so this is a floor of guesses, not an estimate (stage 1 PLAN line 113 makes the same point). Evaluation, not training, is likely the larger cost (speculative).

**Nautilus gates** (`STANDARDS.md` line 49, verified: "more than 8 GPUs at once, or a sweep of more than 20 Jobs" is a stop):
- **GPUs at once.** Wave 1 (seed 0 of each encoder config) is 3 or 5 training GPUs. Wave 2 (seeds 1 and 2 of every config, including `none`) is 8 runs with Open b "no" (exactly at the cap if all run at once) and 12 with "yes" (over the cap, so either two sub-waves of at most 8, or the user's explicit OK). Evaluation pods count toward the 8 when they run at the same time as training. The implementation session never exceeds 8 GPUs at once without the user's OK.
- **Jobs.** Even the smallest recommended design (11 training runs, 12 clean evaluations, 12 LIBERO-Plus evaluations, each sharded) is a sweep of more than 20 Jobs in total. **The user's OK for the stage 2 sweep as a whole is needed before the first wave**, stated with the wave plan and Job counts. Indexed Jobs (one Job, several completions) reduce the count, as the proxy did (proxy PLAN line 303).
- **Idle GPUs.** Evaluation pods are the 40% utilisation risk (stage 1 PLAN line 178); stage 1's rule applies: raise environments per GPU until the timing smoke shows at least 40% over 10-minute windows, otherwise stop for the user.

**Storage:** Open item g.

## Seeds

- **Training seeds 0, 1, 2 for every configuration** (`../../../_standards/07` line 9: "at least 3 seeds for any reported comparison", verified; `STANDARDS.md` line 31: at least 3 training seeds per combo, verified). The same seed numbers across configs, so each comparison is paired by data order (Design, "Pairing across configurations").
- `none` seed 0 is stage 1's run (stage 1 PLAN line 248); seeds 1 and 2 of `none` are trained here, as the user decided.
- Evaluation seeds 41, 42, 43 for LIBERO-Plus (paper App. E.3, p22); 41 for clean LIBERO (stage 1 PLAN line 101).
- Fewer than 3 training seeds needs a stated override of `_standards/07` and `STANDARDS.md` (decision 006 line 26, verified).

## Test-set use

Clean LIBERO and LIBERO-Plus rollouts are the test sets. Used for: the timing-only smoke per configuration (no outcome is recorded; stage 1 PLAN line 96, PR #38), the clean gate, and the LIBERO-Plus evaluation of each final checkpoint, each logged in this folder's `LOG.md` with date and checkpoint. No hyperparameter, checkpoint, recipe or threshold choice is made from any of them. A checkpoint is re-evaluated at most once, only after a pipeline fault has been shown without test outcomes (clean gate above). Seed 0 results of wave 1 are seen before waves 2 and 3 are trained; the rule is locked before wave 1, and no recipe or rule change is made from wave 1 outcomes (only pipeline faults shown without test outcomes are fixed, and both evaluations are reported). The paper's numbers are known in advance; they are the hypothesis, not a test-set use.

## Open, for the user

None of these is decided. Each has the lead's or the worker's recommendation and the cost of each side.

a. **The helpful pairing.** DINOv3 + SigLIP 2 is the lead's suggestion (decision 006 line 15), not yet the user's choice.
   - DINOv3 + SigLIP 2: +1.83 in Table 10; both caches exist; cost as the table above.
   - DINOv3 + UniTok: the largest gain over DINOv3 in Table 10 (+2.12); no loader (UniTok needs its repository's code, `literature/encoder-weights.md` line 11), no cache; costs a loader, a new download through the user, and an extraction Job (about 223 to 251 GB of disk by the other caches' sizes, speculative).
   - Recommendation (the worker's, agreeing with the lead): DINOv3 + SigLIP 2. Note: both gains are one checkpoint each, so neither pairing is established as helpful; the contrast in the rule only asks that it is not harmful.
b. **Whether SigLIP 2 alone and C-RADIOv4 alone are trained.** The mentor suggested testing encoders on their own (decision 005 line 5, verified). Yes: 6 more runs at k = 3 (17 instead of 11), about 140 more L40S-hours and 6 more checkpoint evaluations; gives "below each member alone", the paper's actual pattern. No: the harm can only be stated against DINOv3. The lead recommends yes; so does the worker.
c. **Training seeds per configuration.** The lead recommends 3 (the standards' minimum). Run counts for every combination are in "Compute and cost". 2 needs an override and makes sigma_hat close to unusable; 5 nearly doubles the cost.
d. **How much LIBERO-Plus.**
   - D1: the full benchmark (10,030 episodes x 3 evaluation seeds) on every trained checkpoint. The statistic is defined on it; it is the paper's protocol and the user's earlier choice for the proxy.
   - D2: the full benchmark on seed 0 of each configuration first, the other seeds after. Same total cost as D1, different order (this is item e).
   - D3: full on seed 0, a proportional subset on seeds 1 and 2. Cheaper, but mixes two protocols in one statistic; the rule would have to be rewritten for it.
   - Recommendation: D1, run in the order of D2.
e. **Order.** The lead recommends one seed of every configuration first to settle the mechanics (memory, step time, loader speed, L4 fit and GPU use for each pair), with the written rule that no verdict, no rule change and no recipe change is made from a single seed. The worker agrees and has written it into "Test-set use" and "Order of work".
f. **SigLIP 2 variant (new, from the fusion table row 1).** The cache is so400m; the paper says it "primarily" used L variants, and SigLIP 2 has an L. Options: keep so400m and record the departure (no cost, one silent-point difference in the helpful pairing); or extract SigLIP 2 L (a loader flag, an extraction Job on Nautilus, about 223 GB of disk if its width is 1,024 like DINOv3's, recalled width, and the disk is already short, item g). Recommendation: keep so400m for stage 2 and record it, because "primarily" is not "always" and the helpful pairing is the contrast, not the primary comparison; revisit if the contrast fails. The user's faithfulness rule points the other way, so the user decides.
g. **Disk (new).** 220G free on the PVC; 11 or 17 runs keep about 294 or 454 GB of checkpoints under stage 1's keep-until-loaded rule (arithmetic). Options: delete each run's `ckpt.pt` once its `model.pt` has loaded in evaluation (deleting data needs the user's OK; one OK for the whole stage would cover it), which leaves about 74 or 114 GB of `model.pt` files; or ask for a larger PVC. Recommendation: the first, with the OK given once at sign-off.
h. **Feature normalization (new, row 6).** The paper is silent and calls the features "raw" in one appendix (p21); our caches and rollouts z-score per channel. Options: keep z-score (as the proxy, no new code); or raw features (code change; DINOv3's and C-RADIOv4's raw scales differ, which is one of stage 3's suspects). Recommendation: keep z-score in stage 2 and test scale in stage 3, unless the DreamWorld reading (Hand-off step 1) states what was done.
i. **The inconclusive-case extension** (Decision rule, outcome I): allow two more seeds of `dinov3` and `dinov3+c-radiov4` once, or not. Recommendation: allow it, written now.
j. **Threshold T** (options 1 to 4) and **clean gate** (G1 or G2) in the Decision rule. Recommendations: option 3 and G1.

## Order of work

1. Reading, before the lock: DreamWorld (Tan et al. 2026, arXiv 2603.00466) for the channel-wise details the paper leaves out; any answer goes to the user as an amendment before the lock (Hand-off step 1).
2. The user signs off the rule (Open items a to j answered and recorded as a dated amendment).
3. Stage 1 result and zone known; the blanks above filled by a dated amendment.
4. Code by PR (Hand-off step 2), cache pre-check on the cluster (step 3).
5. The user's OK for the sweep (Nautilus gate), with the wave plan and Job counts.
6. Wave 1: seed 0 of every encoder configuration (and `none` seed 1 if the user wants it early). **The design locks when the first wave 1 training run starts.** First minutes: micro-batch, step time, data wait logged; projections logged. Then a timing-only smoke per configuration on L4, then clean evaluation, gate, then LIBERO-Plus. **No verdict from wave 1.**
7. Wave 2: seeds 1 and 2 of every configuration including `none` (in sub-waves of at most 8 GPUs unless the user OKs more), then their evaluations.
8. Apply the locked rule word for word with the analysis subcommand; report.

## Hand-off

The implementation session (a lead with its own workers) does the following, code by PR, and **runs nothing on the cluster or a paid provider except on the user's own go in that session**:
1. **Reading (no run).** DreamWorld (arXiv 2603.00466): what it states for channel-wise concatenation about grid resizing, feature normalization or projection, time alignment and the initialisation of the widened projections. A download goes through the user with a pinned command. Results to the planning lead as a short reading file with page numbers. (If the literature session should own this reading instead, the planning lead routes it.)
2. **Code, by PR, with CPU tests:** items 1 to 5 of "What the implementation session must add" (and item 6 only if the user chooses SigLIP 2 L).
3. **Cluster, reading only, first step on the cluster:** the cache pre-check (every episode of every encoder present with its stats file, shapes and lengths match meta; `_meta.shard*.json` `model` and `resolution`; free space). Until then the cache answer above is not verified.
4. **First minutes of each configuration's first training run:** micro-batch by stage 1's rule, seconds per step, data wait, peak memory, projection, all in this folder's `LOG.md`; a stop for the user if micro-batch 4 does not fit or the projection exceeds the agreed limit (stage 1's rule, about twice the projection).
5. **A timing-only smoke per configuration on L4** before its evaluation (fit, GPU use, time per episode); a move off L4 is a stop for the user.
6. This folder's `LOG.md` (step by step, exact numbers, estimated against actual card-hours) and `README.md` (results). The planning lead writes the direction `LOG.md` entry.

## Out of scope

Fusion variants and other integration mechanisms (stage 3); explaining why a pairing hurts (stage 3 and the frozen-feature results, stage 4); encoders other than DINOv3, SigLIP 2 and C-RADIOv4 (UniTok only if the user picks it under item a); three- or four-encoder compositions; depths other than 8; fine-tuning encoders (`STANDARDS.md` line 10); the paper's other benchmarks (RoboCasa, RoboTwin, real robot); any hyperparameter tuning; Job manifests (the implementation session's work).

## Status

Design only, written 2026-11-15. **Not signed off.** Nothing run, no code, no results. Waits on stage 1's result and on the user's answers to Open items a to j.

## Amendments
<!-- ### YYYY-MM-DD: what changed, why, and whether any results had been seen yet -->
