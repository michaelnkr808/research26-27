# Fast-WAM reading for the stage 1 baseline: validation split, augmentation, timestep draws, training-only behaviour, the action-expert preprocess, init states

Written 2026-11-15 by an agent (reading worker) for the baseline PLAN's Hand-off "Reading first" item (`PLAN.md` line 224), Open items 7 and 10, departures 2 and 7, and sanity check 4. Reading only: no Fast-WAM code was run, nothing was downloaded, no cluster was touched. The only thing executed was a small parser of our own (Python `zipfile` and `pickletools.genops`, which parse pickle opcodes without importing or calling anything) on the LIBERO-Plus init-state files (Q6); no file was unpickled.

Sources, read-only, extracted from the pinned tarballs by the lead:
- `FW/` = Fast-WAM at `7faa71108368fbb3b6885649f112af607427a2d4` (tarball sha256 `43102a7c...6f81520`). Paths are relative to the tree root.
- `LP/` = LIBERO-Plus at `4976dc30028e805ff8094b55501d532c48fec182` (tarball sha256 `8a31a8b0...bdc5cf9`). Paths relative to the tree root.
- Our code: paths relative to `world-model-stacking-encoders/` at `origin/main` `2636d0a`.

Labels: verified (read in the file and lines named, or computed this session from the files), recalled (memory, not checked), speculative (inference or arithmetic).

**The LIBERO training run.** Fast-WAM's README trains LIBERO with `bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4` (`FW/README.md:382, 388`: "a single node with 8 GPUs", verified). The joint variant we build has its own task file, `FW/configs/task/libero_joint_2cam224_1e-4.yaml`, identical to the uncond one except `override /model: fastwam_joint` (lines 4-5 in both; verified by reading both). Both use data config `libero_2cam` (line 4), batch 16 per GPU (line 9), 10 epochs (line 18), `weight_decay: 1e-2` (line 26), `gradient_accumulation_steps: 1` (line 25), on top of `FW/configs/train.yaml` (`mixed_precision: "bf16"` line 28, `seed: 42` line 29, `max_grad_norm: 1.0` line 30). Global batch is 16 x 8 GPUs = 128 (verified from those lines; matches the paper's 128).

## 1. Validation split (Open item 7)

**None for LIBERO.** (verified)
- `FW/configs/data/libero_2cam.yaml:33-34`: `val_set_proportion: 0.0`, `is_training_set: true`. Same in the LeRobot v3 variant (`FW/configs/data/libero_2cam_lerobot_v30.yaml:29-30`).
- With a proportion below 1e-6 every episode of every suite goes into the training set (`FW/src/fastwam/datasets/lerobot/base_lerobot_dataset.py:99-101`). Only a nonzero proportion triggers the split: a per-suite shuffle with `np.random.default_rng(seed)` (seed default 42) and the last `proportion` of episodes held out (lines 102-112). That path is not taken for LIBERO.
- The data config has no `val:` section, so `build_datasets` sets `val_ds = train_ds` (`FW/src/fastwam/runtime.py:430-433`). The "validation" dataset is the training set.
- For contrast, RoboTwin does hold out 1% (`FW/configs/data/robotwin.yaml:29` and its `val:` block at line 64, `val_set_proportion: 0.01` at line 92). This may be what the paper's "counted before the validation split" note refers to (speculative); for LIBERO the code splits nothing.

**What the "val" dataset is used for: monitoring only.** (verified)
- Every `eval_every` = 200 optimizer steps (task yaml line 22), `evaluate()` picks one random sample (index from a generator seeded by step + rank, `FW/src/fastwam/trainer.py:409-410`) from `val_dataset` (= the training set), computes its training loss, runs inference with seed 42, and logs PSNR/SSIM and action L1/L2 against ground truth (lines 399-588). The results are only logged and sent to wandb (lines 751-783).
- No checkpoint selection and no early stopping: weights and state are saved every `save_every` = 2000 steps (task yaml line 21; `trainer.py:785-793`) and at the end (lines 795-813); nothing reads the eval metrics to choose a checkpoint or stop. The loop ends only at `max_steps` (line 682).
- Side effect (verified): `evaluate()` calls `model.training_loss(sample)` (line 415), which draws noise and timesteps from the global RNG, so Fast-WAM's training draws are shifted every 200 steps. Irrelevant to us; noted because it is training-only behaviour.

**Consequence.** Training on all 1,712 episodes with no validation split matches Fast-WAM's LIBERO recipe. Departure 6 in the PLAN is not a departure from Fast-WAM (it may still differ from the paper, whose Table 7 note mentions a split without a size; speculative).

## 2. Image augmentation before the VAE (Open item 10, departure 7)

**No random image transform is applied in the LIBERO config.** Every transform on the training path is deterministic. (verified)

The full image path for a training sample, in order:
1. Frames decoded from the LeRobot videos (512 x 512 per camera, `libero_2cam.yaml:12, 16`). The LeRobot layer applies `image_transforms` only if one is passed (`FW/src/fastwam/datasets/lerobot/lerobot/lerobot_dataset.py:761-764`), and Fast-WAM's wrapper does not pass one (`base_lerobot_dataset.py:115-121`, no `image_transforms` argument, default `None`). (verified)
2. Processor `train_transforms` (`libero_2cam.yaml:58-61`), applied per camera when the processor is in train mode (`FW/src/fastwam/datasets/lerobot/processors/fastwam_processor.py:224-227`): `ToTensor` (uint8 to float / 255, `FW/src/fastwam/datasets/lerobot/transforms/image.py:6-13`) then `torchvision.transforms.Resize(size=[224, 224])` (deterministic; default bilinear with antialias in torchvision 0.22, recalled). `val_transforms` are identical (lines 62-65), so train and eval mode see the same frames. (verified)
3. Cameras concatenated horizontally to 224 x 448 (`concat_multi_camera: "horizontal"`, line 36; `FW/src/fastwam/datasets/lerobot/robot_video_dataset.py:193-195`). (verified)
4. `ResizeSmallestSideAspectPreserving` to [224, 448] (bicubic, antialias), `CenterCrop` [224, 448], `Normalize` mean 0.5 std 0.5 to [-1, 1] (`robot_video_dataset.py:85-93, 206-209`; definitions `FW/src/fastwam/datasets/dataset_utils.py:28-109`). All deterministic. At the input size 224 x 448 the resize scale is 1 and the crop is the whole frame (verified arithmetic from lines 46-50 and 80-81), so both are identities (that torchvision returns a same-size tensor unchanged is recalled; even if it resamples, sampling at scale 1 is an identity up to rounding, speculative).

A repo-wide grep of `FW/src`, `FW/scripts`, `FW/configs` for `random`, `Random`, `ColorJitter`, `Flip`, `augment` found no image augmentation; the only random calls in the data path are a retry on load errors (`robot_video_dataset.py:149, 290`, `base_lerobot_dataset.py:213`) and the instruction choice below. (verified)

Other data-side randomness, none of which changes the content seen in training (verified):
- Instruction: `augment_instruction` draws `np.random.rand() < drop_high_level_prob` (`fastwam_processor.py:144`), but the LIBERO config does not set it, so the default 1.0 applies (line 38) and the draw is always true: the instruction is always the plain task string (line 145). The prompt template is fixed (`robot_video_dataset.py:23, 230`).
- `skip_padding_as_possible: false` (`libero_2cam.yaml:35`), so padded windows are not resampled (`robot_video_dataset.py:132-133`).
- Text context: cached T5 embeddings, padded positions zeroed and the mask set to all ones (`robot_video_dataset.py:241-246`). Our `src/t5_cache.py` reproduces this (lines 27-33, 41-46, verified).

**Consequence for departure 7.** Caching the VAE latents and T5 embeddings drops no augmentation: Fast-WAM's LIBERO training sees exactly one deterministic view of each clip, as our cache does. The only remaining difference is the rounding noted in the PLAN (latents computed in bf16, stored float16; speculative effect negligible). Our cache's frame path, `src/fastwam_data.py:83-86` (ToTensor-equivalent, bilinear antialias Resize to 224, concat), matches steps 2 to 4 except that it skips the step-4 identity resize and crop (verified by reading; equality of pixels not run). Open item 10 has nothing to decide.

## 3. `sample_training_t` and the timestep draws (departure 2)

**Per-sample, independent, uniform in a warped variable; not stratified, not shared across the batch.** (verified)
- `FW/src/fastwam/models/wan22/schedulers/scheduler_continuous.py:31-37`: `u = torch.rand((batch_size,), dtype=float32)` on the model's device, `sigma = shift * u / (1 + (shift - 1) * u)` (the `_phi` warp, lines 18-19), `timestep = sigma * num_train_timesteps`, then `.to(dtype)`. One independent `u` per sample, from the global torch RNG; there is no generator argument.
- Two schedulers, two separate draws: video timesteps with `train_shift 5.0`, action timesteps with `train_shift 1.0` (uniform), both `num_train_timesteps 1000` (`FW/configs/model/fastwam_joint.yaml:47-55`; built in `FW/src/fastwam/models/wan22/fastwam.py:64-79`). Video and action timesteps of the same sample are independent of each other.
- Draw order inside `training_loss` (`fastwam.py:532-551`): video noise (`randn_like`), video timesteps, action noise, action timesteps. `src/proxy_wam.py` lines 401-409 makes the same calls in the same order (verified).
- The `dtype` passed is the model dtype (`input_latents.dtype`, `action.dtype`, `fastwam.py:536, 548`), which is bf16 in Fast-WAM's run (Q4 a). So Fast-WAM's timesteps and noise are bf16 tensors; ours are fp32.
- Loss weight per sample: `training_weight(t)` = a Gaussian bump in t, shifted and normalised (lines 39-47, constants from lines 21-29), computed in fp32. Our proxy calls the same method (`src/proxy_wam.py` lines 417, 426).
- Batch structure of Fast-WAM's own run: 8 ranks x 16 samples, each rank seeded `seed + rank` (`FW/src/fastwam/utils/pytorch_utils.py:25-32`), so each rank draws its 16 samples' noise and timesteps on its own; data-parallel averaging over 8 equal slices then equals the mean over 128 (recalled DeepSpeed/DDP behaviour). Fast-WAM itself never draws all 128 at once.

**Consequence for departure 2.** Since draws are i.i.d. per sample, drawing all 128 once per optimizer step and slicing per micro-batch is the same distribution as Fast-WAM's, and the PLAN's exactness argument holds (speculative only in the floating-point sense the PLAN already states). Implementation note: `sample_training_t` takes no generator (lines 31-37), so the training script must make the per-step draw itself (`torch.rand(128, generator=g)` then the scheduler's `_phi` and `* 1000`, same for the action scheduler) or wrap the call in a forked, reseeded RNG; it cannot pass a generator through.

## 4. Other randomness and training-only behaviour; every mismatch with `src/proxy_wam.py`

What Fast-WAM does (all verified unless labelled):
- **Dropout in Wan blocks or the MoT: none.** `FW/src/fastwam/models/wan22/wan_video_dit.py`, `action_dit.py`, `mot.py`, `fastwam.py`, `fastwam_joint.py` contain no `Dropout` (grep). The only dropout modules are in the T5 text encoder (`wan_video_text_encoder.py:54-123, 234-268`, rate 0.1) and the VAE (rate 0.0); both are frozen and in eval mode (`trainer.py:309-318` puts everything but `model.dit` and the proprio encoder in eval), and LIBERO training does not even load T5 (`fastwam_joint.yaml:5`, `load_text_encoder: false`). `self.training` branches in the DiT, action DiT and MoT only guard shapes, gradient checkpointing (off) and compile (off) (`wan_video_dit.py:472, 485`; `action_dit.py:266`; `mot.py:264, 679, 690`).
- **Text/condition dropout for CFG: none.** `training_loss` always passes the context and the proprio token (`fastwam.py:341-377, 522-572`); no random masking exists. LIBERO evaluation uses `text_cfg_scale: 1.0` (`FW/configs/sim_libero.yaml:34`), so no unconditional branch is ever needed.
- **EMA: none.** No EMA anywhere in `FW/src` or `FW/scripts` (grep for `ema`, `EMA`). Checkpoints hold the raw trained weights (`trainer.py:590-594`; writer `fastwam.py:1199-1209` per `../2026-11-12-pretrained-start-sizing/step2-reading.md` line 74), and evaluation loads those.
- **Optimizer:** `torch.optim.AdamW(dit params + proprio encoder params, lr=1e-4, weight_decay=1e-2, betas=(0.9, 0.95))`, one parameter group, **no weight-decay exclusions** (biases, norms and modulation tensors are all decayed) (`trainer.py:85-94`; wd from task yaml line 26; `train.yaml:31` default 0.0 is overridden).
- **LR schedule:** total steps = `ceil(len(dataset) / (16 x 8)) x 10` (`trainer.py:227-241`), where `len(dataset)` is the number of frames, every frame being a window start (`base_lerobot_dataset.py:186-187`); warmup = `int(0.05 x total)` (line 99) with `LinearLR(start_factor=1/warmup, end_factor=1.0)`, then `CosineAnnealingLR(T_max=total - warmup, eta_min=lr x 0.01 = 1e-6)` via `SequentialLR` (lines 243-276), stepped once per optimizer step (lines 62, 700-705).
- **Precision:** the model is built in bf16 (`runtime.py:34-40, 467-469`: `mixed_precision bf16` to `model_dtype=torch.bfloat16`), and the Accelerator runs DeepSpeed ZeRO-1 with `mixed_precision="bf16"` (`trainer.py:59-63`; `FW/scripts/accelerate_configs/accelerate_zero1_ds.yaml`; `FW/scripts/ds_configs/ds_zero1_config.json`). DeepSpeed's bf16 mode keeps fp32 master weights and fp32 optimizer state (recalled, not checked in DeepSpeed 0.18.7, the pin in `FW/pyproject.toml:15`).
- **Gradient clipping:** the trainer calls `accelerator.clip_grad_norm_(params, 1.0)` (`trainer.py:701`), but under DeepSpeed Accelerate leaves clipping to DeepSpeed's `gradient_clipping` setting (recalled), and `ds_zero1_config.json` has no `gradient_clipping` key (verified, lines 1-18; DeepSpeed's default is no clipping, recalled). Whether Fast-WAM's LIBERO run actually clipped at 1.0 is therefore **unresolved** (speculative either way); settling it needs a read of Accelerate 1.12.0's `clip_grad_norm_` and DeepSpeed 0.18.7's config defaults, which this reading did not download.
- **Loss weighting:** video loss = per-sample mean of MSE over channels and pixels and the non-padded future latent steps, times `training_weight(t_video)`, batch mean; action loss the same with padded action steps masked and `training_weight(t_action)`; total = `1.0 x video + 1.0 x action` (`fastwam.py:420-457, 574-608`; `lambda_video` default 1.0, `fastwam_joint.yaml:57-58` `lambda_action: 1.0`). The first latent frame is kept clean and left out of the loss (lines 541-542, 574-577).
- **Timestep shift:** video 5.0, action 1.0 (`fastwam_joint.yaml:47-55`). The README says action shift 1.0 is the new default and that "the original released checkpoints" used 5.0 (`FW/README.md:117, 128-131`). Which value the paper's runs used is not stated in the code (speculative: possibly 5.0 if they predate the change).
- **Data order:** a full random permutation of all frame windows per epoch (`FW/src/fastwam/utils/samplers.py:29-36`, seed `42 + epoch + offset`). The trainer never calls `set_epoch` itself; whether Accelerate's prepared loader does (otherwise every epoch repeats one order) is recalled as yes, not checked.

Comparison with `src/proxy_wam.py` (`training_loss` lines 396-427, `lr_at` lines 466-471, optimizer line 478, clip line 501; and `src/pretrained_sizing.py` `make_opt` lines 495-496, `micro_step` lines 499-506). Same: draw order and per-sample i.i.d. draws; schedulers (video shift 5, action shift 1, 1000 steps, same `training_weight`, line 386-387); masks and per-sample normalisation (`latent_pad` lines 119-122 equals `fastwam.py:442-443`); first latent frame clean and out of the loss; action-to-all-video attention (`FastWAMJoint`); AdamW lr, betas, wd 0.01, no decay exclusions (proxy decays every parameter too); eps default in both; 1e-6 floor; 5% warmup; no dropout, no EMA, no CFG dropout (verified by reading both).

Every mismatch found:

| # | What | Fast-WAM (`7faa711`) | Proxy / sizing code | Label and size |
|---|---|---|---|---|
| a | Weight precision | bf16 weights; DeepSpeed bf16 with fp32 master copy and optimizer state | fp32 weights, bf16 autocast, fp32 AdamW | verified for the build dtypes; DeepSpeed master weights recalled. This is PLAN departure 3 / Open item 8. |
| b | Dtype of noise and timesteps | bf16: `randn_like` of bf16 latents and actions, timesteps cast to bf16 (`fastwam.py:532-549`), so timesteps near 1000 are rounded to steps of 4 (bf16 has 8 significant bits, recalled) | fp32 noise and timesteps (`proxy_wam.py` lines 401-407, inputs are fp32) | verified dtype path; effect speculative (small). Follows from a. |
| c | Gradient clipping | `clip_grad_norm_(..., 1.0)` called, but possibly a no-op under DeepSpeed with no `gradient_clipping` in the DS config | `torch.nn.utils.clip_grad_norm_(..., 1.0)` always | unresolved (see above). The paper's Table 6 says 1.0, and the PLAN matches the paper. |
| d | Warmup formula | `LinearLR`: factor `1/w + (1 - 1/w) k / w` at optimizer step k | `(k + 1) / w` | verified; the two agree at k = 0 and differ by `k / w^2` of the base rate, at most about 1/w (about 0.09% of 1e-4 at w = 1,085; speculative arithmetic). Negligible. |
| e | Steps per run | `10 x ceil(frames / 128)`, last partial batch kept | `make_plan` drops the final < 128 leftover windows (lines 270-286) | verified formulas; with 277,713 frames (`data/README.md` line 39) that is 21,700 against the PLAN's 21,696 (speculative arithmetic; Accelerate's handling of the last partial batch across 8 ranks is recalled, not checked). |
| f | Data order | full shuffle of all windows per epoch | 16-episode buffers | verified; already PLAN departure 5. |
| g | RNG structure | each of 8 ranks draws for its own 16 samples from `seed + rank` | one generator per optimizer step for all 128 (the PLAN's design) | verified; same distribution, different numbers. |
| h | Global RNG consumed by evaluation | `evaluate()` every 200 steps runs a training loss on one sample (`trainer.py:414-416`) | no in-training evaluation | verified; no effect on the distribution. |
| i | Loss lambdas | `lambda_video x video + lambda_action x action` | `sum(parts.values())`, lambdas ignored (`proxy_wam.py` line 427) | verified; equal only because both lambdas are 1.0. A lambda change would need code. |

## 5. `scripts/preprocess_action_dit_backbone.py`

**What it computes** (all verified from `FW/scripts/preprocess_action_dit_backbone.py`):
- Inputs: `--model-config` (a model yaml with `video_dit_config` and `action_dit_config`, lines 99-122, 143), `--output` path, `--device` (default `cpu`, line 145), `--dtype` (default `float32`, line 146), `--apply-alpha-scaling` (default `true`, lines 147-151). The README's own command uses `--device cuda --dtype bfloat16` with `configs/model/fastwam.yaml` (`FW/README.md:199-205`).
- It builds the video expert by calling `load_wan22_ti2v_5b_components(...)` with the yaml's `video_dit_config` (lines 171-179) and builds a fresh `ActionDiT(**action_cfg)` (line 181), checking equal heads, head dim and layer count (lines 182-187).
- For every action-expert tensor except those starting `action_encoder.` or `head.` (`ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES`, `FW/src/fastwam/models/wan22/action_dit.py:32, 111-117`), it takes the video tensor of the same name (an error if missing, lines 196-198) and:
  - copies it if the shapes are equal (lines 201-203);
  - otherwise resizes it to the action shape one dimension at a time, in dimension order, each by 1-D linear interpolation with `align_corners=True`, in fp32 (lines 53-96, 205);
  - then, if alpha scaling is on, the tensor has 2 or more dimensions and its **last** dimension changed, multiplies by `alpha = sqrt(src_last / target_last)` (lines 206-208);
  - casts to the action tensor's dtype and moves to CPU (line 210).
- Output: a `torch.save` dict `{"policy": {skip_prefixes, alpha_scaling, interpolation}, "backbone_state_dict": {...}, "meta": {hidden_dim, ffn_dim, num_layers, num_heads, attn_head_dim, text_dim, freq_dim, eps}}` (lines 212-230). `ActionDiT.from_pretrained` checks `meta` against the config, requires exactly the backbone keys, and loads strict (`action_dit.py:149-232`).
- **Fresh tensors:** `action_encoder.weight/bias` and `head.weight/bias` (4 tensors; `nn.Linear` default init at construction, `action_dit.py:73, 97, 150`). The proprio encoder is fresh too, but it is not part of this script (`fastwam.py:59-60`).

Per-tensor effect at our sizes (video width 3072, FFN 14336; action width 1024, FFN 4096; both experts 24 heads x 128, so attention width 3072 in both; derived from `wan_video_dit.py:170-183, 197-210, 229-245, 366-379` and `action_dit.py:73-97`; verified shapes, the copy/interpolate/scale outcome is our application of the rule above):
- copied unchanged: self- and cross-attention `q/k/v` biases [3072] and `norm_q/norm_k` weights [3072];
- interpolated along the input dimension and scaled by sqrt(3): `q/k/v` weights [3072, 3072] to [3072, 1024], `text_embedding.2` and `time_embedding.2` weights, `ffn.0` weight ([14336, 3072] to [4096, 1024]), `time_projection.1` weight ([18432, 3072] to [6144, 1024]), and the block `modulation` [1, 6, 3072] to [1, 6, 1024] (a 3-D parameter, so the rule scales it too);
- scaled by sqrt(14336 / 4096) = sqrt(3.5): `ffn.2` weight [3072, 14336] to [1024, 4096];
- interpolated along the output dimension only, no scaling: `o` weights [3072, 3072] to [1024, 3072], `text_embedding.0` [3072, 4096] to [1024, 4096], `time_embedding.0` [3072, 256] to [1024, 256], and all shrunk 1-D tensors (biases of `o`, `ffn`, embeddings, `norm3` weight and bias, `time_projection.1` bias [18432] to [6144]).
- Count at 8 layers (speculative arithmetic): 27 tensors per block x 8 + 10 non-block = 226 backbone tensors (80 copied, 146 interpolated), plus 4 fresh. With the 2 proprio tensors that is 232, the number of new tensors the sizing run reported for its all-fresh action side (PLAN line 20), which cross-checks the key count.
- Two literal behaviours worth knowing (speculative as to intent): the `modulation` tensors get the sqrt(3) factor although they are added, not multiplied; and `time_projection.1` (six 3072-wide chunks stacked into 18432) is interpolated over the stacked axis, so values near chunk boundaries mix neighbouring chunks.

**Can it run at 8 layers keeping Wan blocks 0-7?** Yes, with a model yaml whose `video_dit_config.num_layers` and `action_dit_config.num_layers` are both 8 (verified mechanism): the video expert is built at 8 layers and loaded with `strict=False`, which fills blocks 0-7 and drops `blocks.8`-`blocks.29` (`FW/src/fastwam/models/wan22/helpers/loader.py:115-121`; `step2-reading.md` line 19), then action block i is made from video block i. Only the first N blocks are possible (no index remapping). But as written the script needs more than the transformer:
- `load_wan22_ti2v_5b_components` is called without `load_text_encoder`, so it defaults to `True` and fetches and loads the umT5-XXL encoder and tokenizer (`loader.py:150, 166-168, 191-204`), and it always fetches and loads the VAE (`loader.py:165, 210`), by default from ModelScope (`helpers/io.py:34-38`) into `./checkpoints/` unless `DIFFSYNTH_MODEL_BASE_PATH` points elsewhere (`io.py:48-52`). (verified)
- It reads all three DiT shards into one state dict (`io.py:117-122`) and checks a hash of the merged key names and shapes, so all three shards must be present (`loader.py:96-107`; `io.py:178-200`). They are on the PVC now (PLAN line 254, recalled from the implementation session's report). (verified mechanism)
- Memory: the full 30-block DiT in fp32 (about 20 GB) plus T5 cast to fp32 (about 23 GB from an 11.4 GB bf16 file) plus the VAE, all in RAM at once (speculative arithmetic from `step2-reading.md` lines 9 and 85).
- The `scripts/` folder is not part of the installed package (`FW/pyproject.toml`: packages found under `src`, `include = ["fastwam*"]`), so importing it needs the Fast-WAM tree root on `sys.path`. (verified)

**GPU:** not needed. The device defaults to `cpu` (line 145) and every operation is a tensor copy, `F.interpolate` or a multiply (lines 53-96, 196-210) (verified). Loading the VAE and T5 on CPU is expected to work but was not run (speculative).

**Deterministic:** yes for the tensor values. No random draw enters any saved tensor: values come only from the Wan weights; the `ActionDiT` random init supplies only shapes and dtypes (lines 189-210), and the randomly initialised skipped tensors are not saved (verified). CPU linear interpolation is deterministic (recalled). The pickle bytes of the output file may still differ between runs (recalled). Note the dtype: Fast-WAM's README command uses `bfloat16`, so their backbone was made from Wan weights rounded to bf16 before interpolation (`io.py:131-135` casts on read; README line 204); a `float32` run gives slightly different values (speculative: differences at bf16 rounding level).

**Import, one-off call, or port: port** (recommendation, speculative judgement).
- Calling it as a one-off would need the VAE and T5 downloaded or redirected (we hold neither in that layout), a custom 8-layer yaml, about 45 GB of RAM, and a second loader path for the produced `.pt`. It also routes through Fast-WAM's `strict=False` load, which reports nothing.
- Importing it needs the tarball root on `sys.path` and still drags in `load_wan22_ti2v_5b_components` unless only the helpers are imported.
- Porting is small: `_interpolate_last_dim` and `_resize_tensor_to_shape` (lines 53-96) and the copy/alpha rule (lines 196-210) are pure tensor functions (about 45 lines, MIT). The training script can apply them in-process to the video expert's tensors right after `load_wan` (blocks 0-7 already loaded in fp32, `src/pretrained_sizing.py` line 180), writing straight into `model.action_expert` by name, and record copied / interpolated / fresh counts in the load report.
- Tests for the port (CPU): equality with the original two helpers on random tensors of the real shapes (importing just those functions from the pinned tree is a test-only dependency), the 226 / 4 key split at 8 layers, and the shape and alpha factor of each tensor class above. Whether to build in fp32 (our recipe) or round to bf16 first (Fast-WAM's README) belongs with Open item 8.

## 6. Init states per clean task

**Every one of the 40 base-task `.pruned_init` files holds exactly 50 states.** (verified this session)

What `src/rollout.py` loads: for clean jobs, `torch.load(<init_states>/<suite>/<base task name>.pruned_init)[trial]` (lines 359-361), with base names from `clean_tasks` (10 per suite, asserted, lines 611-616). In LIBERO-Plus the `init_states` path is `libero/libero/init_files` (`LP/libero/libero/__init__.py:21, 32`), and the benchmark names each task's file `f"{task}.pruned_init"` (`LP/libero/libero/benchmark/__init__.py:95`). Under `LP/libero/libero/init_files/` there are exactly 10 `.pruned_init` files in each of `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` (40 in all); `libero_spatial`, `libero_object` and `libero_10` also hold unpruned `.init` files, which `rollout.py` does not read. That these 40 file stems are the base names `clean_tasks` produces is inferred from the naming rule and the count (speculative, not run; the LIBERO-Plus package was not imported).

How the count was derived, without unpickling: each file is a `torch.save` zip archive with only `data.pkl` and `version`, no tensor storage entries. A script of ours (`zipfile` to list entries, `pickletools.genops` to walk the opcodes of `data.pkl`) showed the same structure in all 40: one pickled NumPy array (globals `numpy.core.multiarray _reconstruct`, `numpy ndarray`, `numpy dtype`, `_codecs encode`), dtype string `f8` with `<` byte order, a shape tuple `(50, D)`, and a raw buffer of exactly `50 x D x 8` bytes (cross-check). The first axis is the state index that `[trial]` selects. D, the flattened simulator state size, varies by task:

| Suite | Files | Shape of each |
|---|---|---|
| libero_spatial | 10 | (50, 92) for all 10 |
| libero_object | 10 | (50, 110) for all 10 |
| libero_goal | 10 | (50, 79) for all 10 |
| libero_10 | 10 | (50, 47) x 3 (KITCHEN_SCENE3, KITCHEN_SCENE6, KITCHEN_SCENE8), (50, 51) KITCHEN_SCENE4, (50, 84) x 2 (LIVING_ROOM_SCENE1, LIVING_ROOM_SCENE5), (50, 123) x 2 (LIVING_ROOM_SCENE2, both tasks), (50, 71) LIVING_ROOM_SCENE6, (50, 45) STUDY_SCENE1 |

Caveat: these are the files in the pinned tarball. That the cluster image's installed LIBERO-Plus carries byte-identical files is recalled from the image build (same commit), not checked here.

**Consequence for sanity check 4** (PLAN lines 93-97): no task has more than 50 states, so the "start at state index 50" branch is impossible, and the smoke must use the timing-only mode. The final evaluation's `--clean 50` uses every state of every task.

## Consequences for the PLAN (for the lead and the user)

- **Open item 7:** Fast-WAM's LIBERO config holds out nothing (`val_set_proportion: 0.0`) and its "validation" set is the training set, used only for logged diagnostics; no checkpoint selection or early stopping. Training on all 1,712 episodes matches Fast-WAM; departure 6 is a departure from the paper's Table 7 note at most, not from the code.
- **Open item 10 / departure 7:** no random augmentation in LIBERO training; every frame transform is deterministic. The cached latents lose nothing, and item 10 has nothing to decide.
- **Departure 2:** timesteps are i.i.d. per sample (video shift 5, action shift 1), so drawing all 128 per optimizer step is the same distribution. The script must draw them itself, since `sample_training_t` takes no generator. Fast-WAM's own run draws per GPU slice of 16.
- **Dropout:** none in Wan blocks, the action expert or the MoT. No EMA, no condition dropout, no weight-decay exclusions, in Fast-WAM or the proxy.
- **New, for Open item 8 and the recipe table:** (a) Fast-WAM's noise and timesteps are bf16, a side effect of bf16 weights; (c) whether Fast-WAM's DeepSpeed run actually clipped gradients at 1.0 is unresolved (the DS config has no `gradient_clipping`); the PLAN follows the paper's 1.0, which stays the faithful choice either way; (e) Fast-WAM's step count would be 21,700, not 21,696, because it keeps the last partial batch (negligible; record or match). The README's note that the original release used action shift 5.0 (now 1.0) is worth a line in the PLAN, since the paper's runs may predate the change.
- **Open item 2 (decided: Fast-WAM's recipe):** port the interpolation and alpha rule into our script rather than run Fast-WAM's preprocess script, which also needs the VAE, T5 and all three shards in RAM. 226 tensors derived from blocks 0-7, 4 fresh, deterministic, no GPU.
- **Sanity check 4:** 50 init states per clean task, all 40 tasks; use the timing-only smoke branch.

Not answered: whether Accelerate 1.12.0 plus DeepSpeed 0.18.7 clip gradients with this DS config (needs their source); whether Accelerate's prepared loader reseeds Fast-WAM's sampler per epoch; which action shift the paper used.
