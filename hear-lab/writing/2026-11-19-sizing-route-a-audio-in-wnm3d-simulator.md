# Sizing: route A, spatial audio added inside WNM-3D's own simulator (GN-Bench on InteriorGS) (2026-11-19)

Direction: hear-lab. Written by a planning worker for the hear-lab planning lead. This memo sizes one route only. A second memo sizes the other route; neither memo compares them or recommends one. That choice belongs to the user and the PI.

Labels: `verified` = read this session in the named file at the named commit, or in the named repository file; `recalled` = taken from an earlier hear-lab file (survey or LOG) or from memory, not reopened; `speculative` = this memo's own estimate or reading. Every effort figure is `speculative`. Nothing was run, installed or downloaded for this memo. Code was read from read-only copies at these pinned commits:

- GN0 `c927781` (github.com/TeleHuman/GN0)
- GN-Bench-Tools `0ce8367` (github.com/TeleHuman/GN-Bench-Tools, the GN0 submodule)
- WNM-3D `7fe2d5f` (github.com/TeleHuman/WNM-3D)
- rlr-audio-propagation `4fd446b` (github.com/facebookresearch/rlr-audio-propagation)
- habitat-gs `8387881` (github.com/zju3dv/habitat-gs)
- sound-spaces `287184f` (github.com/facebookresearch/sound-spaces)

Facts already checked in `literature/survey/2026-11-19-pi-followups-simulator-audio-papers-gaps.md` (Part A, "the PR #71 file" below) and `literature/survey/2026-11-18-novelty-and-simulator-gap.md` (Part 2, "the PR #68 file") are cited, not redone.

## What this route is

Keep everything WNM-3D already uses: GN-Bench as the simulator, InteriorGS Gaussian-splat scenes for the pictures, GN-Matrix for start poses, goals and expert paths, GN0's renderer and remote client, and WNM-3D's own converter and trainer. Add one new thing next to the renderer: an acoustic engine that, for every camera pose the renderer already visits, produces the sound the agent would hear from a sounding target object, once as two-ear (binaural) audio and once as one-channel audio. That audio is written beside each episode's video in the LeRobot dataset, read by the trainer next to the frames, and sent to the policy server next to the frames during closed-loop testing.

This is the PI's stated preference: "if you can use WNM-3D's own simulator for simulating audio-visual training data, then we will not use SoundSpaces 2.0" (LOG 2026-11-19, `recalled`). One caveat stays open from the PR #71 file (`recalled`): the most direct engine here is RLR-Audio-Propagation, which is the engine inside SoundSpaces 2.0. The pictures, scenes and episodes would be WNM-3D's own, but the audio physics would be the same as SoundSpaces 2.0's. Whether that counts as "not using SoundSpaces 2.0" is a question for the PI, not something this memo can settle.

What the route does not cover: the design of the audio encoder inside WNM-3D (only the place where it would attach is located, under "Pieces to build", piece 7), and the choice of research question.

## Pieces to build

### 1. Audio engine

What exists:
- GN-Bench renders only pictures. The simulator registers an RGB sensor and a depth sensor and nothing else (`GN_Bench/sims/GN_Bench_simulator/GN_Bench_simulator.py` lines 494 to 529, GN-Bench-Tools `0ce8367`, `verified`); the PR #71 file found no audio code in GN0, GN-Bench-Tools or WNM-3D (`recalled`).
- The simulator's renderer uses CUDA when present and falls back to CPU (`GN_Bench_simulator.py` line 541, `verified`). The agent's height is fixed at 1.3 m (`_load_initial_state`, line 780, `verified`).

Three candidate engines.

Option 1a, RLR-Audio-Propagation used on its own (rlr-audio-propagation `4fd446b`):
- The package ships binaries and C headers only, for "Linux - x64", under "CC-BY-NC" (`README.md`, `verified`). The C interface takes boxes or triangle meshes with a material name per face, sets sources and listeners by position and orientation, and returns impulse responses for every source and listener pair, with mono, binaural (HRTF) and ambisonic listener layouts (`RLRAudioPropagationPkg/headers/RLRAudioPropagation.h`, read in the PR #71 file, `recalled`; this session re-read the HRTF and thread settings: `RLRA_SetListenerHRTF` at line 342, `threadCount` default 1 at line 128, `sampleRate` default 44100 at line 130, `verified`).
- The engine models diffraction (sound bending around a doorway or a corner) and transmission (sound passing through walls). The bare library's own header gives the defaults: `diffraction` true, `maxDiffractionOrder` 10, `transmission` true, `indirectRayCount` 5000 (`RLRAudioPropagationPkg/headers/RLRAudioPropagation.h` lines 118, 126, 152 and 154; the C++ struct repeats them, `transmission = true` at line 620; `verified`). So option 1a has transmission on unless it is switched off. The habitat-gs wrapper documents different defaults (`docs/AUDIO.md` at `8387881`: transmission false, `verified`), and SoundSpaces' continuous simulator sets transmission true and 500 indirect rays (`soundspaces/continuous_simulator.py` lines 123 and 125, sound-spaces `287184f`, `verified`). Which settings the experiment uses is a choice to write down before rendering.
- Missing: no Python binding ships with the standalone package (PR #71 file, `recalled`). A thin caller is needed (ctypes over the C header, or a small C++ program), plus convolution of a dry clip with each impulse response and smoothing between steps (SoundSpaces 2.0 crossfades between steps, PR #71 file, `recalled`).

Option 1b, pyroomacoustics built from the floor plan (pip package, MIT; PR #71 file, `recalled`):
- Image-source plus ray-tracing engine for general polyhedral rooms, cross-platform, pip-installable, with binaural rendering from SOFA HRTF files (PR #71 file, `recalled`).
- Missing: per-wall materials must be set by hand in code (the repository's own `examples/room_from_stl.py` says so, PR #71 file, `recalled`). It is `recalled`, not checked, that it does not model diffraction. If that holds, a target behind a wall or around a corner would be heard only through reflections, which matters most for exactly the "target not visible at start" episodes this experiment is built around.

Option 1c, Habitat-GS (habitat-gs `8387881`):
- A habitat-sim fork that renders Gaussian splats, keeps habitat-sim's audio sensor (RLR inside, exposed to Python through pybind11 per `docs/AUDIO.md`, `verified`) behind a build flag that is off by default (`build.sh` lines 28 and 53 to 54, `HABITAT_WITH_AUDIO:=OFF`, `--with-audio`, `verified`). Its scene set includes some InteriorGS scenes as splats plus navmesh only, with no triangle mesh (PR #71 file, `recalled`), and its audio sensor needs a triangle mesh loaded as the scene to have any geometry (PR #71 file, `recalled`).
- Missing: it is a different simulator from GN-Bench. Used as the audio engine, it would run as a second simulator beside GN-Bench, with an exported mesh loaded as its scene and its agent kept in sync with GN-Bench's pose. Its one real gain over option 1a is a ready Python binding to RLR. The cost is a from-source C++ build of a habitat-sim fork with audio switched on.

Which looks most workable (`speculative`): option 1a. It is the only one that both runs inside GN-Bench's own Python process without a second simulator and models diffraction (per the documented defaults above). Option 1b is the easiest to install but likely the weakest for out-of-sight sounds. Option 1c gives a Python binding at the price of a second simulator and a C++ build; it is a fallback if the ctypes binding to the bare RLR library turns out to be hard. The biggest risk of option 1a is that the binary is archived (last push 2022-11-01, PR #71 file, `recalled`) and may not load on a current Linux image; see "Unknowns".

Effort for option 1a, `speculative`: 3 to 6 working days. Assumes the shipped `.so` loads on a rented Linux machine on the first or second try, the header is enough to write a ctypes binding without vendor docs, and one day goes to checking that mono and binaural outputs sound sane on a hand-placed source. Add 2 to 4 days if a C++ shim is needed instead of ctypes. Option 1c instead: 4 to 8 days (`speculative`), mostly the source build and the pose syncing.

### 2. Geometry and acoustic materials

What exists:
- InteriorGS ships per scene `structure.json` (room outlines, walls with segment, thickness and height, door and window openings, and instance boxes), `labels.json` (oriented 3D boxes with labels and instance ids), the splats and the occupancy map; no triangle meshes, no materials (PR #71 file, `recalled`; the gated files themselves were not opened).
- GN-Bench never opens `labels.json` or `structure.json`; it loads `occupancy.json`, `occupancy.png` and the splat file, and maps world coordinates to splat coordinates with a per-episode affine fit (`derive_affine_transform` line 271, `transform_from_world_to_GS` line 304, `load_occupancy_metadata` line 229, `GN_Bench_simulator.py`, `verified` for the function names and line numbers).
- The SAGE-3D convex collision meshes (USD, one per InteriorGS scene, CC BY-NC 4.0) exist; whether they include walls and floors is not known (PR #71 file, `recalled`).
- RLR's shipped `mp3d_material_config.json` matches material by label substring (wall, floor, ceiling, door, window, chair and others) (PR #71 file, `recalled`).

What is missing:
- A geometry exporter: walls from `structure.json` extruded to triangles, door and window holes left open, a floor and a ceiling per room, furniture as boxes from `labels.json` (each box face given a material). All in the same world frame that GN-Bench uses for the agent pose, so that the listener and the picture agree.
- A material map from InteriorGS labels and room types to RLR material names. This is a guess about real acoustics and must be stated as "assumed materials" wherever results appear.
- A choice about doors: the floor plan marks openings, and door open or closed state is not in the card's key list (PR #71 file, `recalled`). Treating all doors as open is the simple default.

Effort, `speculative`: 3 to 5 working days for floor plan plus boxes. Assumes `structure.json` is a clean, closed floor plan in metres in the same frame as GN-Bench's world (unknown, see "Unknowns"). Add 2 to 3 days to use the SAGE-3D collision meshes instead of boxes (needs a USD reader and a frame check).

### 3. Sound sources

What exists:
- Every GN-Matrix episode names its goal object by instance id and label (`label_ins_id`, `label_label`) and gives a goal position (PR #71 file, `recalled`; the loader that reads these columns is `gn0/benchmark/dataset.py`, GN0 `c927781`, `verified` for the goal and goal-pixel fields at lines 115 to 152).
- SoundSpaces ships "102 copyright-free sounds" (`soundspaces/README.md` line 50, sound-spaces `287184f`, `verified`); the repository is CC-BY-4.0 (`README.md` line 89 and `LICENSE`, `verified`). SAVi pairs sounds with object categories in `data/sounds/semantic_splits` (PR #68 file, `recalled`).

What is missing:
- Placement rule: the sound plays from the goal object's box centre (the instance named by `label_ins_id`, looked up in `labels.json`). Whether the ids match is `speculative` until the gated file is opened (PR #71 file).
- Clip choice: one dry, looping clip per goal category, or a category-independent clip set (for example one of a few "beacon" sounds). Category-matched clips need a mapping from InteriorGS goal labels (wardrobe, sofa and so on) to sounds; many furniture labels have no natural sound, so a SAVi-style category match will not cover all GN-Matrix goals (`speculative`). A category-independent clip avoids that and keeps the audio a pure "where" cue. That is a design choice for the lead.
- Licence: the SoundSpaces sounds are the obvious, already-cleared set, but using them may again raise the "is this SoundSpaces" question with the PI. Alternatives (Freesound CC0 clips, other public sound datasets) were not checked.
- Distractor sounds (as in SAVi) are optional and add effort; not sized.

Effort, `speculative`: 1 to 2 working days for placement and a clip table, assuming the instance ids match and the lead picks a clip policy.

### 4. Episodes and the target-visible-at-start flag

What exists:
- GN-Matrix episodes with start pose, start heading, goal position, goal object, instruction and a reference A* path (`path_raster_world`, `path_raster_pixel`), in splits train_seen, test_seen, test_unseen (PR #71 file, `recalled`).
- Success is "within 3.0 m of the goal position": the `SUCCESS` measure is configured with `SUCCESS_DISTANCE: 3.0` (`configs/gn_bench/interiorgs/train_seen.yaml` lines 47 to 48, GN0 `c927781`, `verified`) and compares distance to target with that radius (`Success` in `GN_Bench/tasks/nav/nav.py` lines 470 to 502, GN-Bench-Tools `0ce8367`, `verified`). The RGB sensor is 480 by 360 with a 70 degree field of view in the same config (`verified`).
- GN-Bench has a depth sensor class (`GNBenchSimDepthSensor`, line 512, `verified`), and the simulator can render observations at any pose without moving the agent (`get_observations_at`, line 1385, `verified`).

Two ways to define a "walk to the sounding object" episode:
- Reuse GN-Matrix episodes as they are. Same start, goal and expert path; the goal object becomes the sound source. Cheapest. But the GN-Matrix instruction names the goal and often its room ("Go to the wardrobe in the dining room", PR #71 file, `recalled`), so language alone already says where to go (see "Risks to the comparison").
- New episodes on the same scenes: new start poses chosen so that the visible and not-visible groups are both large enough, with the expert path from GN-Bench's existing A* planner (the renderer already smooths and resamples A* pixel paths, `gn0/collection/trajectory_video.py`, `verified`). This needs a new parquet in the GN-Matrix column layout, and new expert paths. The code that generated GN-Matrix was not found in the repositories read.

The visible-at-start flag (`speculative` design, all inputs `verified` to exist except the gated box file): at the start pose, the target counts as visible if (a) some part of its 3D box projects inside the 70 degree image, and (b) the box is not hidden: compare the projected box depth with GN-Bench's rendered depth at the start pose. A cheaper but weaker version uses only the floor plan walls (a 2D line-of-sight test), which ignores furniture. The rule has to be written down before any result is seen.

Also needed: drop episodes that start within the 3.0 m success radius (they test nothing about hearing); decide whether 3.0 m stays the success radius for this task (SAVi used 1.0 m, PR #68 file, `recalled`).

Effort, `speculative`: 3 to 5 working days to reuse GN-Matrix episodes and compute the flag with the depth test. Add 3 to 5 days if new episodes are needed for balance. Removing the target name from the instruction does not need new episodes or new code: the trainer already has `always_use_default_instruction`, `default_instruction` and `language_dropout_prob` (`WNM3DTransform._prepare_language`, `gammanav/vln/model/wnm_3d/transform/wnm_3d_cotrain.py` lines 219 to 245; defaults in `configs/model/wnm_3d/transform/wnm_3d_cotrain.yaml` lines 27 to 34, default text "Perform the default behavior."), GN-Matrix rows carry both `instruction` and `grounded_instruction` (`gn0/benchmark/dataset.py` lines 153 to 154), and the converter takes `--instruction-column` (all `verified`). A fixed "go to the sound" text is a config change. At inference time the same text would be sent by the GN0 client, which today forwards the episode instruction (`_query_new_actions`); that is a small change in piece 6.

### 5. Joining audio to rendering, to the LeRobot conversion, and to the trainer

What exists, rendering side (GN0 `c927781`, `verified`):
- `scripts/evaluation/render_dataset.sh` runs `python -m gn0.evaluation.local --render-dataset`, which calls `render_trajectory_videos` (`gn0/evaluation/local.py` lines 270 to 310). That uses `TrajectoryVideoCollector.collect` (`gn0/collection/trajectory_video.py`), which smooths the A* path, resamples it every `RESAMPLE_STEP_M` (0.05 m in `train_seen.yaml`), computes headings, renders one frame per pose with `sim.get_observations_at` (`_render_poses`), and writes only an MP4 at `VIDEO_FPS` (10.0 in the config). It saves no per-frame pose file.
- So frames are spaced by distance, not by time. The "10 frames per second" is a playback rate, not a physical clock. Audio needs a time base, so a convention has to be chosen: how many seconds of sound belong to each frame (`speculative` that this has to be decided; nothing in the code defines it).

What exists, conversion side (WNM-3D `7fe2d5f`, `scripts/data/convert_interiorgs_to_lerobot.py`, `verified`):
- It joins each MP4 to its GN-Matrix row, keeps every third frame by default (`--frame-stride` 3), and rebuilds camera poses from the annotation path, unless per-frame pose files `frame_NNNN_camera.json` sit next to the video, in which case it uses those (`load_rendered_camera_extrinsics`, lines 340 to 366). GN0's renderer as released does not write those files.
- It writes per-episode parquet (camera extrinsic, indices, task) and video, plus `meta/info.json` with `"has_audio": False` (line 710) and `meta/modality.json` with video, camera and annotation entries only (lines 734 to 738).
- When no per-frame pose files exist, the converter rebuilds the whole pose, not only the height, and differently from the renderer. It resamples the raw annotation path `path_raster_world` by arc length to the video's frame count and takes headings from that path (`convert_interiorgs_to_lerobot.py` lines 527 to 536, `resample_path` lines 276 to 300), at a camera height that defaults to 1.0 m (`--camera-height`). The renderer instead uses the line-of-sight-smoothed A* pixel path resampled every 0.05 m, with 0.5 m look-ahead headings and yaw smoothing (`gn0/collection/trajectory_video.py`, `_build_smoothed_positions`, `_line_of_sight_positions`, `_path_yaws`), at the simulator's 1.3 m. So the listener must take the renderer's pose in x, y, heading and height. And because the trainer computes action labels from these poses, adding pose files (as proposed below) changes the action labels too; see "Risks to the comparison".

What exists, trainer side (WNM-3D `7fe2d5f`, `verified`):
- The dataset class is `ShardedLeRobotSingleActionChunkDatasetInteriorGS` in `gammanav/vln/data/dataset/lerobot_sharded.py` (set in `gammanav/vln/configs/data/wnm_3d/interiorgs_wan22.yaml`). Actions are not stored; they are computed from camera extrinsics (class docstring, lines 32 to 41). `get_data_by_modality` (lines 516 to 531) knows four modalities: state, action, video, language, and raises an error for anything else. History frames are sampled uniformly over the past by default (`_select_history_indices`, lines 217 to 231).
- The modality metadata schema allows only video and annotation entries (`LeRobotModalityMetadata` in `gammanav/vln/data/schema/lerobot.py`).
- The data config lists modalities video, state, action, language (`interiorgs_wan22.yaml`), and the transform chain ends in `WNM3DTransform` and the collator (`gammanav/vln/model/wnm_3d/transform/wnm_3d_cotrain.py`).

What is missing:
- In GN0's renderer: at each rendered pose, also compute the impulse response from the source to the listener (mono and binaural), and write per-frame pose files so the converter uses exact poses. Write one audio file per episode (or per-frame audio windows) beside the MP4.
- In the converter: carry the audio through the same frame stride, write it into the LeRobot layout (an audio file per episode plus an index from frame to audio sample range), add an `audio` entry to `modality.json` and `info.json`. LeRobot v2.0 as used here has no audio feature type in this code (`verified` for this converter and schema; the upstream LeRobot format was not checked).
- In the trainer: an `audio` modality in `interiorgs_wan22.yaml`, a reader branch in `get_data_by_modality` that returns the audio window for each sampled frame index (history and target), an extension of `LeRobotModalityMetadata`, and the transform and collator changes in `wnm_3d_cotrain.py` so the audio tensor reaches the model. The no-audio arm uses the same data with the audio modality left out; the one-channel arm reads the mono version.

Effort, `speculative`: 4 to 7 working days. Assumes the engine and geometry pieces are done, one time convention is chosen up front, and audio is stored as plain wave files. Rendering the full data set is compute time, not counted here.

### 6. Closed-loop testing with audio through the GN-Bench WebSocket loop

What exists (`verified`):
- GN0 side: `gn0/agents/remote.py` `RemoteNavAgent._query_new_actions` (lines 481 to 530) takes the RGB observation and instruction and calls `RemoteNavClient.infer` in `gn0/remote/client.py` (lines 158 to 205), which packs `endpoint`, `images`, `instruction`, `session_id`, optional `state.nav_pose` into a msgpack-numpy message. Returned nav_delta actions become a new absolute pose (`_nav_delta_to_env_action`, lines 1180 to 1218), which the simulator accepts as the new pose (`GN_Bench_simulator.py` `step`, lines 819 to 835) and then clips against the occupancy grid when `COLLIDABLE` is true (lines 864 to 878). `COLLIDABLE` is true in all three InteriorGS configs (`train_seen.yaml` line 31, `test_seen.yaml` and `test_unseen.yaml` line 24) and by default (`GN_Bench/config/default.py` line 211, GN-Bench-Tools `0ce8367`). The audio step must use the pose after clipping. There is no physical clock in this loop either.
- Server side: `scripts/inference/wnm_3d_server.py` converts each request (`_convert_observation`, line 343), keeps a rolling frame buffer and picks history frames from it (`_select_past_target_buffer_frames`, line 425; `_select_history_indices`, line 449), and resets per `session_id`.

What is missing:
- On the GN0 side, an audio step after each pose change: compute the impulse response at the new pose, convolve the next stretch of the looping clip, and add it to the request (a new `audio` field in `RemoteNavClient.infer` and in `_query_new_actions`). The audio per step must follow the same time convention as the training data.
- On the server side, an audio buffer kept in step with the frame buffer, picked with the same history indices, and passed into the policy input.
- Logging of the visible-at-start flag per episode next to GN0's existing metrics, so success can be split.

Effort, `speculative`: 3 to 5 working days, assuming the engine wrapper from piece 1 runs inside the GN0 process. A Stage II (DAgger) training run for the audio arms would use this same loop to collect data, so this piece is also needed for training if Stage II is used.

### 7. Model side: where an audio input would attach (location only)

All in WNM-3D `7fe2d5f`, `verified` for the code paths; no design here.
- `gammanav/vln/model/wnm_3d/action_head/wan_flow_matching_action_tf.py`, `WANPolicyHead.forward` (from line 1135): builds the text condition (`encode_prompt`), encodes the video, and builds `past_obs_tokens` from past frames through the VGGT geometry adapter (`_build_vggt_past_obs_tokens`, line 732), then calls the DiT with `context=prompt_embs` and `past_obs_tokens=past_obs_tokens` (lines 1399 to 1424). This is where an audio feature would be computed from the batch.
- `gammanav/vln/model/wnm_3d/modules/wan_video_dit_action_casual_chunk.py`, `CausalWanModel`: the cross-attention context is built from the text embedding with the CLIP image embedding put in front (lines 1920 to 1924 and 2185 to 2189). Both forwards assert `context.shape[1] == self.text_len` before that (lines 2008 and 2184), so extra audio tokens would have to be joined after `text_embedding`, the way CLIP is, not into the raw text context. The second existing hook is `past_obs_tokens`, checked and put in front of the video tokens in the full-sequence forward (lines 2102 to 2180). Only the kv-cache forward `_forward_inference` rejects `past_obs_tokens` (line 2003); `CausalWanModel.forward` sends calls without a kv cache to the full-sequence path (lines 2285 to 2289), and WNM-3D's own inference builds `past_obs_tokens` and requires them (`wan_flow_matching_action_tf.py` lines 1773 to 1789). So this hook is in use at inference time too.
- `gammanav/vln/model/wnm_3d/model.py` (`WNM3DModel.prepare_input`, line 187) and `gammanav/vln/model/wnm_3d/inference_policy.py` carry batch fields to the head, and `scripts/inference/wnm_3d_server.py` feeds the policy at inference time. The VGGT geometry adapter (`modules/vggt_geometry_adapter.py`) is an existing example of an extra frozen encoder plus a trainable adapter added beside the video path.

Effort: not sized here; the encoder and its wiring are the research part and belong to the lead's design decision.

## Data, licences and access

- InteriorGS (huggingface.co/datasets/spatialverse/InteriorGS): gated with automatic approval after accepting the "InteriorGS Terms of Use" (PR #71 file, `recalled`). WNM-3D's `docs/DOWNLOADS.md` says to accept the terms and log in to Hugging Face before download (`verified`). Any cloud machine that renders data needs a Hugging Face token with that acceptance. The terms themselves were not read (the gated page refuses without access), so whether derived audio data may be published is unknown.
- GN-Matrix: not gated, CC BY-NC 4.0 (PR #71 file, `recalled`).
- SAGE-3D collision meshes (only if used): not gated, CC BY-NC 4.0 (PR #71 file, `recalled`).
- RLR-Audio-Propagation: CC-BY-NC, Linux x64 binaries (`README.md`, `verified`). Archived (PR #71 file, `recalled`).
- pyroomacoustics: MIT (PR #71 file, `recalled`).
- Sound clips: SoundSpaces' 102 "copyright-free" sounds in a CC-BY-4.0 repository (`verified`, see piece 3). Other sources not checked.
- WNM-3D code: Apache-2.0 for the original parts; the vendored VGGT-Omega and DINOv3 parts are under the FAIR Noncommercial Research License and the DINOv3 licence, and the model card lists `fair-noncommercial-research-license` (`README.md` lines 230 to 244, `docs/MODEL_CARD.md` line 2, `verified`).
- Training initialisation weights: Wan2.2-TI2V-5B, Wan2.1-I2V-14B-480P (image encoder), UMT5-XXL (tokenizer), and VGGT-Omega, which is gated and must be authorised separately (`docs/DOWNLOADS.md`, `verified`). `docs/DOWNLOADS.md` lines 53 to 54 say only that "evaluation and inference do not require" these weights; it says nothing about fine-tuning. What the code shows for fine-tuning from a released checkpoint (WNM-3D `7fe2d5f`, `verified` by reading, nothing run):
  - The launcher checks all of them before any run that is not a dry run (`scripts/train/wnm_3d_base.sh` lines 110 to 154). It downloads Wan2.2-TI2V-5B (UMT5 encoder and VAE files), Wan2.1-I2V-14B-480P (CLIP image encoder) and the UMT5-XXL tokenizer if they are missing (lines 121 to 142), and exits if `$VGGT_OMEGA_CKPT` is not a file (lines 143 to 147). So the launcher as released needs the VGGT-Omega file and the Wan and UMT5 downloads even when fine-tuning.
  - The model code does not read the standalone VGGT-Omega file when training starts from a released checkpoint. With `PRETRAINED_MODEL_PATH` set, the launcher adds `++action_head_cfg.config.skip_component_loading=true` (line 244); with that flag, `_init_vggt_geometry_modules` skips the standalone checkpoint and prints that its parameters "will be loaded from the full WNM-3D checkpoint" (`wan_flow_matching_action_tf.py` lines 515 to 527), and the text encoder, image encoder and VAE files are skipped the same way (line 342). `WNM3DModel.from_pretrained` also forces the flag (`model.py` line 329). Not checked: whether `dit_version=$WAN22_CKPT_DIR` or the tokenizer path (launcher lines 234 and 238) still read files from the Wan or UMT5 folders when the model is built.
  - Ways around the launcher check that the reviewer of PR #73 named, as untested readings of the code, not recommendations: (a) `DRY_RUN=true` skips the whole check block (line 110) and prints the full training command with its Hydra arguments (lines 258 to 262), which could then be run directly; (b) `VGGT_OMEGA_CKPT` could point at any existing file, since the launcher only tests `-f` and the file is not opened while `skip_component_loading` is true; (c) a one-line change to the launcher. The released checkpoint itself embeds the VGGT-Omega parameters and is under the FAIR Noncommercial Research License (`README.md` lines 238 to 242, `verified`).
  - The clean option is to request VGGT-Omega access on the Hugging Face account used on the rented machine. Whether to do that or use a workaround is the user's choice (and the PI's where it applies).
- All of the above are research or non-commercial terms, which fits a paper. Access steps on the user: accepting the InteriorGS terms on the Hugging Face account used on the rented machine, and, for training through the launcher as released, either VGGT-Omega access or one of the untested workarounds above (the user's choice).

## Compute and storage

What sources state (all `verified` in the named file):
- WNM-3D's released policy server was validated on one H100 80 GB and used about 27 GiB of GPU memory with one replica; each released stage checkpoint takes about 26 GiB on disk (`docs/INSTALLATION.md` lines 35 to 52, WNM-3D `7fe2d5f`).
- The Stage I launcher defaults to 8 GPUs, a per-device batch of 12 and up to 200,000 steps (`scripts/train/wnm_3d_stage1.sh`, `scripts/train/wnm_3d_base.sh`), and the install guide says other configurations "should be validated for both memory use and effective global batch size" (`docs/INSTALLATION.md`).
- RLR runs on CPU threads (`threadCount`, header line 128) on Linux x64 only. GN-Bench's splat renderer uses CUDA when available (`GN_Bench_simulator.py` line 541).

What is not stated anywhere read, and has to be timed on real data before any rental is sized:
- RLR impulse responses per second per CPU core on an InteriorGS scene built from the floor plan, at the ray counts that will be used. The bare library defaults to 5000 indirect rays (header line 118), while SoundSpaces' continuous simulator uses 500 (`continuous_simulator.py` line 123); the timing should be done at the count actually chosen, since it drives the cost.
- Splat frames per second for GN0's dataset renderer on the GPU type that would be rented.
- One training step's time and memory for the chosen fine-tuning set-up on the GPU type that would be rented, for each of the three arms (the audio arms add an encoder).
- Wall time of one closed-loop episode with audio.
- Size of InteriorGS on disk (not read), and audio storage, which follows from sample rate, channel count, bit depth and the seconds of audio per frame (a convention still to be chosen).

Budget: 200 US dollars on Vast.ai for all compute (`STANDARDS.md`, `verified`). Under this route, three kinds of work draw on it: data rendering (GPU for pictures, CPU for audio), training of three arms (with seeds per `_standards/07`), and closed-loop testing of three arms. Audio rendering is CPU work and can run on a cheaper CPU machine than the training, if the engine and the picture renderer are run as separate passes (`speculative`). Whether the whole plan fits in the budget cannot be said until the timings above exist.

## Unknowns and the smallest real check for each

Each check uses the real tools and the real data. None was run. The experiment session would run them later, with the user's OK.

1. Does the archived RLR binary load and produce mono and binaural impulse responses on a current Linux image? Check: on a rented Linux x64 CPU machine, load `libRLRAudioPropagation.so` through ctypes, build one real InteriorGS scene's walls and boxes, place one source and one listener, and write both impulse responses. Needs: a Vast.ai CPU instance, InteriorGS access (Hugging Face token with accepted terms) for one scene.
2. Is `structure.json` a clean, closed floor plan in the same world frame GN-Bench uses for the agent? Check: export one scene's walls, draw them over that scene's `occupancy.png` through GN-Bench's own `transform_from_world_to_pixel`, and look at the overlay for three or four scenes. Needs: InteriorGS access for those scenes; runs on CPU.
3. Do GN-Matrix `label_ins_id` values match the instance ids in `labels.json`? Check: look up every goal id of one GN-Matrix split in its scenes' `labels.json` and count misses. Needs: InteriorGS access; CPU.
4. How many episodes have the target visible at the start, in each split? Check: compute the flag with the depth test on the full test_unseen split and a sample of train_seen. Needs: InteriorGS access, one GPU instance that runs GN-Bench (the depth render uses the splat renderer).
5. How fast is RLR on these scenes, and how fast is the splat renderer? Check: time impulse responses on one full GN-Matrix episode's poses at the planned ray counts, and time `render_dataset.sh` on the same episode. Needs: one CPU instance and one GPU instance (or one GPU instance with enough CPU cores).
6. Do the SAGE-3D collision meshes include walls and floors (only if boxes from the floor plan are not enough)? Check: open one scene's `_collision.usd` with a USD reader and list the prims. Needs: a machine where a USD library can be installed (the Mac setup does not allow pip installs; a rented CPU instance does).
7. What do the InteriorGS terms allow for derived audio data and its release? Check: read the terms after accepting the gate. Needs: the user's Hugging Face account.
8. Does the PI accept RLR outside Habitat as "not using SoundSpaces 2.0", and SoundSpaces' sound clips? Check: ask the PI. No compute.
9. Is the WNM-3D training step affordable on one rented GPU at all with an added audio input? Check: one timed training step for the no-audio arm on the GPU type that would be rented, with the released checkpoint as the start. Needs: one large-memory GPU instance, the released checkpoint (about 26 GiB), the converted data for a few episodes. Also confirm, in the same run, which Wan and UMT5 files are read when the model is built from the released checkpoint (the open point under "Data, licences and access").
10. How do closed-loop step sizes compare with the training frame spacing? Training frames are spaced by distance (0.05 m resample, then every third frame kept), while closed-loop steps are whatever the policy returns. Check: run the released checkpoint on a few GN-Bench episodes and log the step lengths. Needs: one GPU instance, the released checkpoint, InteriorGS access.

## Risks to the comparison

These are the ways the no audio / one-channel / binaural comparison could be unfair or unconvincing to a reviewer under this route. All are `speculative` readings of the code and data described above.

- The instruction gives the answer away. GN-Matrix instructions name the target and often its room. With that text, the model can find the goal without hearing it, so all three arms may score about the same and the audio effect is hidden. The comparison needs an instruction that does not locate the target (for example a fixed "go to the sound" text, or the category only, as in SAVi). This is a design decision for the lead. The trainer's existing fixed-instruction switch (piece 4) makes this cheap to do; the risk is in forgetting to, or in using it for some arms and not others.
- The released checkpoint has already seen these episodes. The released WNM-3D weights were trained on InteriorGS train_seen data (PR #68 file, `recalled`). Reusing those episodes for audio training starts every arm from a model that memorised the paths. Results should be read on test_unseen, and the no-audio arm must get exactly the same extra fine-tuning as the audio arms, not be the released checkpoint as it is.
- Mono is not defined yet. "One-channel" can mean the binaural signal averaged to one channel, or a separate mono render. A separate render has different random ray noise, so the two arms would differ in more than the channel count. Averaging the binaural render is the cleanest "same render" choice. The same encoder must take both (mono duplicated to two channels, or a matched input), since the novelty claim rests on holding the encoder fixed (PR #68 file, `recalled`).
- Assumed materials and wall-only geometry. Acoustics come from guessed materials and a floor plan, not measured rooms. That is the same for all three arms, so it does not bias the comparison, but a reviewer may doubt that the out-of-sight results transfer to real rooms. With the bare library's default (transmission on), a hidden target is heard partly through walls as well as through openings and by diffraction; with transmission off, only the last two. How loud and how directional a hidden target sounds depends on that setting and on the guessed wall materials, so the setting must be fixed and reported.
- The engine choice decides the out-of-sight result. If an engine without diffraction is used (pyroomacoustics, per the `recalled` point above), out-of-sight targets get weaker direction cues than they would in reality, which could hide or inflate a binaural advantage exactly in the "not visible" group.
- No clock. Both the expert data and the closed loop are pose sequences without physical time. Audio per frame is a chosen window. If the window or step spacing differs between training and testing, the audio arms meet inputs at inference time that they never saw in training, and the vision-only arm does not have that problem.
- Action labels depend on the pose source. The trainer computes action labels from the per-frame camera poses. The released recipe rebuilds those poses from the annotation path, while this route would write the renderer's exact poses as pose files (piece 5), which changes the labels. All three arms, including no audio, must be trained from the same converted data with the same pose source, so that only the audio differs.
- Loudness as a progress meter. Expert paths go straight to the sounding target, so loudness rises along every training path. The audio arms may learn "louder means closer" without using direction. That is fine for one-channel against no audio but would blur what binaural adds. The visibility split partly guards against it; a reviewer may still ask.
- The visibility flag definition and group sizes. The split is only convincing if the flag rule is fixed before results and the not-visible group is large enough. GN-Matrix start poses were not built for this, so the groups may be very uneven (unknown 4). Episodes that start inside the 3.0 m success radius should be dropped.
- Success radius. 3.0 m is generous for "walk to the sounding object"; with a generous radius, small direction errors do not cost success, which can shrink differences between arms.
- Offline only. If only Stage I (offline expert data) is used for the audio arms, all arms suffer the usual compounding errors in closed loop; if Stage II (DAgger) is used, the audio arms need audio in the collection loop too (piece 6). Either is fair if all three arms get the same stages.
- One synthetic scene family. All scenes are InteriorGS, rebuilt from artist-made rooms. Claims stay inside that setting.

## Summary

Total effort for one person with coding agents, `speculative`: about 17 to 30 working days for pieces 1 to 6 with RLR used on its own, floor plan plus boxes, reused GN-Matrix episodes and the depth-based visibility flag. Each add-on from the pieces raises this: new episodes 3 to 5 days, collision meshes instead of boxes 2 to 3 days, a C++ shim for RLR 2 to 4 days. Any one of them alone adds 2 to 5 days; all three together add 7 to 12 days, for at most about 24 to 42 days. The audio encoder and its wiring into WNM-3D (piece 7) are not included. Compute time (rendering, three-arm training, closed-loop testing) is not included and cannot be sized until the timings under "Compute and storage" are measured.

The three biggest risks:
1. The archived RLR binary may not run on a current Linux image; the fallbacks are a habitat-sim audio build (a C++ build and a second simulator) or pyroomacoustics (likely no diffraction, which weakens exactly the out-of-sight case).
2. The GN-Matrix instructions name the target, and the released checkpoint has already trained on those episodes; without new instructions and a clean test split, the three arms may look the same for reasons that have nothing to do with audio.
3. Nothing in GN-Bench has a clock, so the time base for audio is a convention; a mismatch between training and closed-loop testing would hurt only the audio arms.

What would make this route fail outright: no InteriorGS access, or terms that forbid the derived data; RLR not loading on any available Linux machine while the fallbacks are judged too weak for out-of-sight sounds; `structure.json` not lining up with GN-Bench's world frame in a fixable way; or the PI deciding that RLR-based audio still counts as SoundSpaces 2.0.
