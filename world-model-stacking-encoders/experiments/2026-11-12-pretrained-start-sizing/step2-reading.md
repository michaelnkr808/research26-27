# Step 2 reading: Fast-WAM's loading path, compile status, released checkpoints, Wan 2.2 licence and size

Written 2026-11-12 by an agent (reading worker) for the sizing check's Step 2 (`PLAN.md`). Reading only: nothing was run, no Fast-WAM code executed, no weights or archives downloaded. Web facts come from Hugging Face's public listing API (file names and sizes) and small text files (`config.json`, the safetensors index, README), fetched on 2026-11-12. Labels: verified (read in the file or page named), recalled (memory), speculative (inference or arithmetic). Source paths below are relative to the Fast-WAM tree at commit `7faa711` (tarball sha256 matched the pin in `experiments/2026-10-27-proxy-calibration/requirements.txt`, per the lead), and `fw:` means `src/fastwam/models/wan22/`.

## 1. Which files hold the Wan 2.2 TI2V-5B DiT weights

- **Fast-WAM reads the DiT from `Wan-AI/Wan2.2-TI2V-5B`, files matching `diffusion_pytorch_model*.safetensors`** (verified: `fw:helpers/loader.py:126`; model id in `configs/model/fastwam_joint.yaml:2`). Not a converted copy: only the VAE and the T5 encoder are redirected to `DiffSynth-Studio/Wan-Series-Converted-Safetensors` (`loader.py:131-137`, verified), and the DiT line is not in the redirect dict.
- **Default download source is ModelScope**, not Hugging Face (verified: `fw:helpers/io.py:34-38`, default `"modelscope"` unless env `DIFFSYNTH_DOWNLOAD_SOURCE=huggingface`). Files land in `$DIFFSYNTH_MODEL_BASE_PATH/Wan-AI/Wan2.2-TI2V-5B/` (default `./checkpoints/`; `io.py:48-52, 101-114`, verified). A local copy is used without downloading if the glob matches (`io.py:54-71`, verified).
- **Actual files on Hugging Face** (verified, `https://huggingface.co/api/models/Wan-AI/Wan2.2-TI2V-5B/tree/main`): three shards `diffusion_pytorch_model-0000{1,2,3}-of-00003.safetensors` (9,825,014,472 + 9,995,661,736 + 178,558,176 bytes) and `diffusion_pytorch_model.safetensors.index.json`. Shard 1 holds blocks 0-13 plus the non-block tensors, shard 2 holds blocks 14-29 (index file, verified).
- **Format.** Safetensors, original Wan naming (not the diffusers `Wan2.2-TI2V-5B-Diffusers` layout): the index has 825 tensors named `blocks.N.self_attn.q.weight`, `blocks.N.ffn.0.weight`, `blocks.N.norm3.weight`, `blocks.N.modulation`, `head.head.weight`, `head.modulation`, `patch_embedding.*`, `text_embedding.0/2.*`, `time_embedding.0/2.*`, `time_projection.1.*` (verified, index json). The index's `total_size` is 19,999,150,848 bytes, which is 4 bytes per parameter for about 5.0 B parameters, so the stored tensors are float32 (speculative inference from size; the dtype field was not read).
- The loader also insists on a hash of the file's tensor names and shapes matching a registered value (`loader.py:36-45, 96-107`, verified), so the unmodified three shards pass; a truncated or re-saved file would not.

## 2. Parameter-name mapping, dtype casts, block indexing, subsets

- **No rename at all for the DiT.** The registry entry for `wan_video_dit` has no `state_dict_converter` (`loader.py:40-45`, verified; only the VAE has one). Checkpoint names are used as they are, and they match `WanVideoDiT`'s own parameter names one to one (verified by comparing the index key list with the module definitions: `fw:wan_video_dit.py:229-245` for blocks, `366-384` for the rest; `norm1`/`norm2` have no parameters, `norm3` has). The diffusers renaming function in `helpers/state_dict_converters.py:10-78` exists but the DiT path never calls it (verified: `grep` of the loader, only `wan_video_vae_state_dict_converter` is imported, `loader.py:9-11`).
- **Load call.** `model = WanVideoDiT(**config)`, `state_dict = load_state_dict(path, torch_dtype, "cpu")`, `model.load_state_dict(state_dict, strict=False)`, then `model.to(device, dtype)` (`loader.py:115-121`, verified). All shards are read and merged (`io.py:117-136`).
- **Dtype cast.** Every tensor is cast to the `torch_dtype` the caller passes while reading (`io.py:131-135`, verified). Fast-WAM's configs build in bf16 (`runtime.py:178`, verified); the proxy passes `torch.float32` (`src/proxy_wam.py` line 384, verified), so fp32 checkpoint to fp32 model is a no-op cast. No other cast is applied.
- **Block indexing.** Blocks are an `nn.ModuleList` named `blocks`, indices 0 to `num_layers - 1` (`fw:wan_video_dit.py:380-383`, verified). A checkpoint key `blocks.5.x` goes to model `blocks.5.x`; nothing re-indexes.
- **Can the loader take a subset of blocks?** Only the first N, by accident. Building with `num_layers=8` and loading with `strict=False` fills blocks 0-7 and silently drops the `blocks.8` to `blocks.29` keys (PyTorch's `strict=False` ignores unexpected keys; recalled behaviour, not run). Three limits (all verified from the code above):
  1. No other subset (evenly spaced, last 8) is supported: indices are not remapped, so it would need our own mapping (rename `blocks.{src}.` to `blocks.{dst}.`).
  2. `strict=False` also hides missing keys, and returns no report; the code ignores the return value. Step 4's loaded/new/skipped report must be our own.
  3. A shape mismatch still raises even with `strict=False` (PyTorch behaviour, recalled), so building the DiT with `in_dim = 48 + extra` fails on `patch_embedding.weight` (`[3072, 48, 1, 2, 2]` in the checkpoint) and `head.head.*` is unaffected only if `out_dim` stays 48. The widened input layer therefore needs our own code: build at 48, load, then widen; or load by hand.
- **The all-in-one entry point also loads other things.** `load_wan22_ti2v_5b_components` always downloads the VAE (`loader.py:165`), and builds the action expert separately (`fw:fastwam.py:137-143`); calling it would pull the VAE file and trip over the in-dim widening. Neither is needed with cached latents. Reusing only `WanVideoDiT(...)` plus our own safetensors read is the cleaner route (recommendation, speculative).
- **The trained Fast-WAM checkpoint uses different keys**: `{"mot": mot.state_dict(), "proprio_encoder": ..., "step", "torch_dtype"}` (`fw:fastwam.py:1199-1209`, verified), so video weights sit under `mixtures.video.*` and action weights under `mixtures.action.*` (from `MoT.mixtures`, `fw:mot.py:25`, verified). Relevant to Q6 only.

## 3. Expert configs for this checkpoint

Video expert (verified: `configs/model/fastwam_joint.yaml:12-33`, identical in `fastwam.yaml`; and Wan's own `config.json`: `dim 3072, ffn_dim 14336, freq_dim 256, in_dim 48, num_heads 24, num_layers 30, out_dim 48`):

| setting | value |
|---|---|
| width (`hidden_dim`) | 3072 |
| FFN size | 14336 |
| heads / head dim | 24 / 128 (24 x 128 = 3072) |
| layers | 30 |
| text dim | 4096 |
| freq dim | 256 |
| patch size | (1, 2, 2) |
| in / out channels | 48 / 48 |
| other flags | `eps 1e-6`, `seperated_timestep true`, `fuse_vae_embedding_in_latents true`, `video_attention_mask_mode first_frame_causal`, `action_conditioned false` |

Action expert (verified: same yaml, lines 35-47): `hidden_dim 1024`, `ffn_dim 4096`, `num_heads 24`, `attn_head_dim 128`, `num_layers 30`, `text_dim 4096`, `freq_dim 256`, `eps 1e-6`, `action_dim` taken from the data config (7 for LIBERO). Width ratio 3:1 and FFN ratio 4 as the proxy assumes (`src/proxy_wam.py` lines 373-384).

**Does MoT require equal depth? Yes** (verified): all experts must have the same number of layers, heads and head dimension (`fw:mot.py:33-47`; also `fw:fastwam.py:144-149`). Widths and FFN sizes may differ. This confirms the PLAN's "recalled, not checked" statement.

Detail for the 8-layer build: Wan's 3D RoPE splits `attn_head_dim 128` with `128 // 3 = 42` (even), which the proxy's assert allows. The proxy's FFN formula `round(hidden * 14336 / 3072)` at hidden 3072 gives exactly 14336 (verified arithmetic).

## 4. How the action expert, proprio encoder and other non-Wan parts are initialised

- **Action expert in Fast-WAM: not fresh, not a copy, but resized from the Wan video DiT.** `ActionDiT.from_pretrained` loads a prepared backbone file `checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt` (`configs/model/fastwam_joint.yaml:9`; `fw:action_dit.py:111-232`, verified). That file is produced locally by `scripts/preprocess_action_dit_backbone.py`, which loads the Wan video DiT, and for every action-expert tensor whose name exists in the video expert either copies it (same shape) or linearly interpolates it along each dimension (`align_corners=True`) and multiplies by `sqrt(video_width / action_width)` when the last dimension shrank (alpha scaling, default on) (`scripts/preprocess_action_dit_backbone.py:53-96, 139-236`, verified). It is not a separate downloaded checkpoint; the README tells the user to run it (README "Model Preparation", Step 2, verified).
  - Left random (kept fresh): any tensor starting `action_encoder.` or `head.` (`action_dit.py:32, 116`, verified): the 7-dim action in-projection and out-projection. Everything else in the action expert (blocks, text embedding, time embedding, time projection) comes from the resized Wan weights.
  - The prepared file is checked against the action config (hidden, FFN, layers, heads, head dim, text dim, freq dim, eps) and against missing or unexpected keys, then loaded strict (`action_dit.py:164-225`, verified). A mismatched layer count refuses to load.
  - If no path is given or `skip_dit_load_from_pretrain` is true, the action expert is random (`action_dit.py:130-138`, verified).
- **The proxy does not do this.** `build_model` constructs `ActionDiT(...)` directly, so the proxy's action expert is entirely random (`src/proxy_wam.py` lines 379-387, verified), and so was the video expert. For the sizing check the action expert's initialisation does not change cost, but it is the natural answer to Open item 5's "what to initialise new parts with": Fast-WAM's own recipe would rerun the preprocess script with the video expert built at 8 layers (it builds the video expert from the same config, `scripts/preprocess_action_dit_backbone.py:171-187`, so the output would be an 8-layer backbone from blocks 0-7 only). Our sizing script does not need it; a real run might.
- **Proprio encoder: fresh** `nn.Linear(proprio_dim, text_dim)` with PyTorch's default init, built in `FastWAM.__init__` (`fw:fastwam.py:58-62`, verified). It has no pretrained source in Fast-WAM; it is saved and loaded with the trained checkpoint only (`fastwam.py:1204-1205, 1219-1224`).
- **Other non-Wan parts.** The VAE and T5 are loaded from converted copies and are frozen (`loader.py:131-137`; trainer freezes everything but `model.dit` and the proprio encoder, `src/fastwam/trainer.py:312-318`, verified). Action-expert `text_embedding`, `time_embedding`, `time_projection` are resized Wan tensors, as above.

## 5. torch.compile, CUDA graphs and gradient checkpointing

- **In the proxy's path nothing is compiled.** The proxy builds `FastWAMJoint` directly (`src/proxy_wam.py` lines 383-387) without `compile_training_denoise`, whose default is `False` (`fw:fastwam.py:41, 88`, verified), and `MoT.compile_training_layers` is set from it (`fastwam.py:89`; initial `False`, `fw:mot.py:31`, verified). Even Fast-WAM's own Hydra route for the joint model has no switch: `create_fastwam_joint` does not take or pass `compile_training_denoise` (`runtime.py:163-245`, verified), and `fastwam_joint.yaml` has no such key (only `fastwam.yaml` has `compile_training_denoise: false`).
- **Where compile does exist** (all verified):
  - Training core, opt-in only: when `compile_training_denoise=True` and the model is in training mode, `MoT.forward_joint_core` wraps one joint layer with `torch.compile(..., fullgraph=True)` (`fw:mot.py:678-691`; default inductor backend, no CUDA graph there). This is the "compiled tensor core" the bench README quotes.
  - VAE encode, always: `FastWAM._encode_video_latents` compiles `vae.model.encode` with `backend="cudagraphs", fullgraph=True` the first time it is called (`fw:fastwam.py:247-260`). This is what crashed at B=4 in the 2026-10-16 bench (README line 62). It runs only inside `build_inputs` (the path that encodes raw video); the proxy's `training_loss` bypasses `build_inputs` and its `LatentsOnly` stand-in has no `.model.encode`, so this path is never reached.
  - Inference only: `compile_action_infer` paths (`fastwam.py:909-920, 1100-1141`), not used for training or sizing.
- **Would a batch-size change within one process break the sizing script?** Not from compile: with the defaults above the model runs in plain eager mode, so changing micro-batch within one process is a plain shape change. Per-process isolation (Step 6) is still right for allocator-fragmentation and OOM reasons, but it is not forced by any compiled path in what we call (speculative conclusion from reading the call graph; not run). If someone sets `compile_training_denoise=True`, a new batch size would trigger a recompile or a shape error (recalled behaviour of `torch.compile` with static shapes, not checked).
- **Gradient checkpointing is not available.** `MoT.__init__` raises `ValueError("Wan MoT gradient checkpointing is not supported by the compiled tensor core.")` whenever `mot_checkpoint_mixed_attn` is true, and again if either expert has `use_gradient_checkpointing` set (`fw:mot.py:28-29, 53-62`, verified). The yaml keeps `mot_checkpoint_mixed_attn: false`. Note the message blames the compiled core, but the check is unconditional, so it holds even with compile off. (The bench README, line 63, saw the same error.) Per-block `torch.utils.checkpoint` calls exist in `_apply_post_with_optional_checkpoint` (`mot.py:213-280`) but are unreachable through the constructor.

## 6. Does Fast-WAM publish a trained LIBERO checkpoint?

- **Yes, two for LIBERO** (verified: Fast-WAM README at `7faa711`, "Inference with Released Checkpoints", lines 272-301; Hugging Face file listing `https://huggingface.co/api/models/yuanty/fastwam/tree/main`, 2026-11-12; model page `https://huggingface.co/yuanty/fastwam`):
  - `libero_uncond_2cam224.pt`, 12,041,735,140 bytes (12.04 GB), the Fast-WAM base model (task config `libero_uncond_2cam224_1e-4` uses `override /model: fastwam`, `configs/task/libero_uncond_2cam224_1e-4.yaml:5`).
  - `libero_optional_idm_2cam224.pt`, 12,041,735,545 bytes, the optional-IDM variant (README table: 97.75 percent average success first-frame mode, 98.55 percent IDM mode, full LIBERO; as published, not reproduced here).
  - Matching `*_dataset_stats.json` (40,939 bytes each). A RoboTwin checkpoint (`robotwin_uncond_3cam_384.pt`, 12,041,813,092 bytes) is in the same repo. Repo total listed 36.1 GB.
- **Which parts it holds** (verified from the writer, `fw:fastwam.py:1199-1209`; trainer saves via this, `trainer.py:590-594`): a pickle with `mot` (both experts, `mixtures.video.*` and `mixtures.action.*`), `proprio_encoder`, `step`, `torch_dtype`. No VAE, no T5, no optimizer.
- **Depth: 30 layers in both experts** by config (`fastwam.yaml`/`fastwam_optional_idm.yaml:26,44` `num_layers: 30`, verified). Size check (speculative arithmetic): 12.04 GB at 2 bytes per parameter is about 6.02 B parameters, which equals the bench's measured trainable count (MoT plus proprio, 6.021 B, bench README line 60), so the file is consistent with a bf16 30-layer model.
- **Not our model class.** The released LIBERO checkpoints are the base and optional-IDM models, not `FastWAMJoint`. In the base model the action tokens attend only to the first video frame (`fw:fastwam.py:415-417`, verified), whereas the proxy's `FastWAMJoint` lets action attend to all video tokens (`fw:fastwam_joint.py:47-48`). Parameter shapes are the same, but the checkpoint was trained under a different mask. Whether to start from it is the user's decision after this check (PLAN Step 2).
- **Licence: not stated on the weights.** The model page's card metadata has no `license` field (HF API: `cardData.license` null; README in the repo has none, verified 2026-11-12). The code repository's `LICENSE` is MIT (verified: first line of `LICENSE` in the tarball: "MIT License, Copyright (c) 2026 The FastWAM Authors"). The Wan 2.2 base weights it was fine-tuned from are Apache 2.0 (Q7). Whether the MIT code licence covers the released weights is unclear (speculative); worth recording as unresolved in `data/README.md` if the checkpoint is ever used.
- The LIBERO dataset `yuanty/LIBERO-fastwam` is theirs (README lines 214, 212, verified), unrelated to the weights question.

## 7. Wan-AI/Wan2.2-TI2V-5B licence and size

Verified from `https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B` (HF API tree and card, 2026-11-12; repo last modified 2025-08-07):

- **Licence: Apache 2.0** (card metadata `license: apache-2.0`; README "License Agreement" section says the models are licensed under Apache 2.0, with a usage-conduct paragraph on lawful use).
- **Total of the files in the repo root: 34,180,085,475 bytes (34.18 GB)**, which is the five big files plus small ones: `models_t5_umt5-xxl-enc-bf16.pth` 11,361,920,418; `Wan2.2_VAE.pth` 2,818,839,170; the three DiT shards. Hugging Face's `usedStorage` field says 54.2 GB (includes other revisions or LFS history; not used here).
- **DiT weight files alone: 9,825,014,472 + 9,995,661,736 + 178,558,176 = 19,999,234,384 bytes (about 20.0 GB)**, fp32 (see Q1). The index file's `total_size` is 19,999,150,848, consistent (the difference is safetensors headers).
- For only blocks 0-7 plus the non-block tensors, the first shard (9.83 GB) is the only file needed (index file: blocks 0-13 and all non-block tensors are in shard 1; verified). Safetensors allows reading only the needed byte ranges, but a plain download fetches the whole 9.83 GB shard (recalled).

## Consequences for the sizing script

What the script must do for loading:

1. **Use our own loader, not `load_wan22_ti2v_5b_components`.** Build `WanVideoDiT` with the 8-layer config at width 3072 (as the proxy's `build_model` does, with `hidden=3072, ffn 14336, heads 24, head_dim 128, in_dim 48`), read the safetensors shard(s) with `safetensors.safe_open`, and copy tensors by name. Names need no renaming; the only mapping is block index (identity for blocks 0-7, `blocks.{src}` to `blocks.{dst}` for any other set). Only shard 1 is needed for blocks 0-7, 9.83 GB, not the full 20 GB or the 34 GB repo.
2. **Do not rely on Fast-WAM's `strict=False` load for the Step 4 report.** It returns nothing and ignores both missing and extra keys; the script must compute loaded / new / skipped itself, from the checkpoint index and the model's `state_dict`.
3. **Widen the input layer by hand.** Fast-WAM's loader would raise a shape error for `patch_embedding.weight` at 48 + extra channels. Build at 48, load, then copy into the widened `Conv3d` (VAE slice from the checkpoint, extra slice new). `head.head` stays at 48 output channels in the proxy's design unless the head is also widened for the source-channel loss (PLAN says the head is widened too); the same by-hand route applies.
4. **Pass `torch_dtype=torch.float32` and cast on read**: the checkpoint is fp32 on disk, matching the proxy's fp32 weights with bf16 autocast, so the loaded tensors are exact copies (Step 4's max-abs-difference-0 check is meaningful with no cast).
5. **Action expert and proprio encoder: initialise as the proxy does (random)** for the cost measurement; cost does not depend on it. Fast-WAM's own recipe (resized from the Wan video DiT, script above) is the reference for Open item 5; the 8-layer version would come from rerunning the preprocess script with `num_layers: 8`.
6. **Compile is off in this path**, so within-process batch-size changes are not blocked by it (the 2026-10-16 crash was the VAE's CUDA-graph compile, bypassed by `LatentsOnly`). Keep per-process isolation anyway (OOM and fragmentation, not compile). Do not set `compile_training_denoise`.

Contradictions or corrections to the PLAN:

- **Gradient checkpointing is unavailable** (`mot.py:28-29, 53-62`). Any fallback in the fit list or Open item 4 that mentions it must say it would need a code change in Fast-WAM's `MoT`; the only memory levers in the unmodified code are micro-batch size and precision.
- **Action expert is pretrained-from-Wan in Fast-WAM** (resized interpolation), not fresh or separate. The PLAN's list of possibilities ("from scratch, copied from the video expert, or from a separate checkpoint") is closest to "copied and resized from the video expert, with the 7-dim in/out projections fresh". The proxy's from-scratch action expert is therefore a difference from Fast-WAM, not just from the paper.
- **PLAN says "any 8 cost the same" and "Fast-WAM's config is not read here"**: now read, confirming all 30 blocks are identical in shape (one `DiTBlock` repeated, `wan_video_dit.py:380-383`), so the cost claim stands (the loader limit above is about correctness of the load, not cost).
- **PLAN predicts the first-8 subset loads through Fast-WAM's loader**: it does only for the first N blocks and only without widening; any other set or the widening needs own code, as above.
- **The released LIBERO checkpoints are base and optional-IDM, 30 layers, not `FastWAMJoint`**; the proxy's mask (action sees all video) differs from theirs (action sees the first frame). If the user later wants to start from a released checkpoint, that is a model-class decision, not a drop-in. Its weights licence is unstated on Hugging Face.
- **Default download source is ModelScope**; on a paid provider outside China a direct Hugging Face download of shard 1 is the simpler route (the files are the same repo id on both; recalled for ModelScope, verified for Hugging Face).
- **Total size of the repo is 34.18 GB, DiT 20.0 GB, DiT shard 1 9.83 GB**, which the PLAN's disk and download-time estimates (not given in files read here) should use.

Not answered: the licence of the released Fast-WAM weights (no statement exists on the pages read), and whether the ModelScope copy is byte-identical to Hugging Face (not read).
