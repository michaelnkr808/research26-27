# Sizing memo, route B: move WNM-3D to Habitat and use SoundSpaces 2.0 for the audio (2026-11-19)

Written by a planning worker for the hear-lab planning lead. This memo sizes one route only. It does not compare it with the other route (building audio inside WNM-3D's own simulator) and does not recommend either. Nothing was installed, built, downloaded in bulk or run. Code was read as read-only copies at pinned commits in the planning session's scratchpad.

Labels: \`verified\` = read this session in the named file, commit or page; \`recalled\` = from memory or from an earlier survey file, not reopened; \`speculative\` = my inference or estimate. Every effort figure below is \`speculative\`.

Pinned commits read: WNM-3D \`7fe2d5f\`, GN0 \`c927781\`, GN-Bench-Tools \`0ce8367\`, GN0-VLN-CE \`ac4ebed\`, sound-spaces \`287184f\`, NaVLA \`436ad6b\`, habitat-gs \`8387881\`, rlr-audio-propagation \`4fd446b\`. Pages read: habitat-sim \`DATASETS.md\` (main, \`57ee494\`), niessner/Matterport \`README.md\` (\`8cd7c81\`), matterport/habitat-matterport-3dresearch \`README.md\` (\`9b74d5a\`), Hugging Face model API for \`TeleEmbodied/WNM-3D\` (revision \`5b8c08d\`) and \`facebook/VGGT-Omega\` (revision \`1041e80\`), GitHub refs API for habitat-sim and habitat-lab, HTTP HEAD of two SoundSpaces download files.

Builds on: \`hear-lab/literature/survey/2026-11-18-novelty-and-simulator-gap.md\` Part 2 and \`hear-lab/literature/survey/2026-11-19-pi-followups-simulator-audio-papers-gaps.md\` Part A. Facts taken from those files without reopening the source are marked \`recalled\`.

## What this route is

The idea in one line: keep WNM-3D as the model, but stop using its Gaussian-splat simulator for this project. Render the training data and run the closed-loop test in Habitat, where SoundSpaces 2.0 already produces sound that bounces around a real scanned house and arrives at two ears.

What has to happen, in order:

1. Install an audio-enabled habitat-sim (SoundSpaces 2.0's pinned build) on a Linux GPU machine, with Matterport3D or HM3D scenes.
2. Pick episodes where the goal is a sounding object, and compute for each one whether the target is visible from the start pose.
3. Walk an expert path through each training episode and record, for every kept frame, the camera image, the camera pose, a one-channel audio clip and a two-ear audio clip from the same scene and source.
4. Convert that into the LeRobot layout WNM-3D's trainer reads, plus an audio stream the trainer does not read yet.
5. Fine-tune three WNM-3D versions on the same episodes (no audio, one-channel, two-ear). The audio encoder and how it attaches are the research contribution and are not sized here; only the attach point is located.
6. Test each version in closed loop in Habitat through WNM-3D's existing WebSocket server, with a new Habitat client that also sends audio, and a rule for turning WNM-3D's (dx, dy, dyaw) output into Habitat motion.

Why this route exists (\`verified\`, from the 2026-11-18 survey Part 2 and this session's reading): WNM-3D's own simulator, GN-Bench, has no audio anywhere, while SoundSpaces 2.0's audio sensor lives inside habitat-sim. WNM-3D's server only needs RGB frames, an optional instruction and an optional pose, so it can be driven from any simulator that has a client.

## Pieces to build

### 1. Install SoundSpaces 2.0 with its pinned habitat-sim and habitat-lab, plus scene data

What exists (\`verified\`):
- \`sound-spaces\` \`INSTALLATION.md\` (\`287184f\`): conda env with Python 3.9 and cmake 3.14.0; habitat-sim cloned and checked out at branch \`RLRAudioPropagationUpdate\`, built with \`python setup.py install --headless --audio\`; habitat-lab at tag \`v0.2.2\`, with a manual edit (remove the FetchRobot import on line 36 of \`habitat/tasks/rearrange/rearrange_sim.py\`); then \`pip install -e .\` of sound-spaces. SoundSpaces 2.0 also needs the material file \`mp3d_material_config.json\` from rlr-audio-propagation, and GLIBC 2.29 or newer (older systems need a manual GLIBC build). The install test is \`python examples/minimal_example.py\`, which should write \`data/output.wav\`.
- GitHub refs API, read 2026-11-19: branch \`RLRAudioPropagationUpdate\` still exists at \`4f61e32\` (last commit 2022-11-04); habitat-sim tag \`v0.2.3\` is \`fffa543\`; habitat-lab tag \`v0.2.2\` is \`0f454f6\`. facebookresearch/habitat-sim is archived (\`recalled\` from the 2026-11-19 survey Part A).
- A second pinned stack exists and is closer to HM3D: NaVLA's \`install_env_bench.sh\` (\`436ad6b\`) builds habitat-sim at tag \`v0.2.3\` with \`--headless --audio\`, habitat-lab \`v0.2.3\`, goat-bench, and sound-spaces, in Python 3.9. So the audio sensor builds from a release tag, not only from the SoundSpaces branch, at least by that repository's account.
- habitat-gs (\`8387881\`) is a habitat-sim fork at version 0.3.3 (\`src_python/habitat_sim/__init__.py\` line 9) that keeps the audio sensor source; its \`docs/AUDIO.md\` says the sensor was added to habitat-sim through upstream PR #1646. So newer habitat-sim versions carry the audio code too (whether they build and run with it was not tested).
- The SoundSpaces Python package (\`soundspaces/\`) uses the old yacs-style habitat-lab config. The audio itself comes from \`habitat_sim.AudioSensorSpec\` in habitat-sim (\`SoundSpaces2.md\`), so a data renderer can call habitat-sim directly, as NaVLA's \`minavbench/build_datasets/replay_trajectory_with_audio.py\` does (lines 133 to 158).
- RLR is Linux x64 only and CC-BY-NC (\`recalled\`, 2026-11-19 survey Part A). WNM-3D needs a separate environment anyway: Python 3.11, PyTorch 2.8.0, CUDA 12.9 (\`WNM-3D/docs/INSTALLATION.md\`, \`7fe2d5f\`). Two environments talking over the WebSocket is the pattern WNM-3D already uses with GN0.

What is missing:
- A container image (or a scripted setup) that builds the audio habitat-sim on a Vast.ai host and passes the install test. Building habitat-sim from source needs EGL and OpenGL development libraries for headless rendering (GN0-VLN-CE \`docs/INSTALLATION.md\` lists them for its own habitat-sim build, \`verified\`). Whether a Vast.ai container exposes a working EGL device for habitat-sim is not known.
- A decision on which pinned stack to use (SoundSpaces branch with habitat-lab 0.2.2, or NaVLA's 0.2.3 tags). This is the lead's call; HM3D semantics loading differs by version (\`speculative\`).
- Scene data access (see "Data, licences and access").

Effort: 1 to 4 working days (\`speculative\`). Assumes one person with coding agents, a Linux GPU instance with root, and that one of the two pinned stacks builds without patching the C++ code. The upper end covers a GLIBC rebuild, EGL problems in the container, and making a reusable image. Waiting for dataset access is calendar time, not counted here.

### 2. Rendering the training data and converting it to WNM-3D's LeRobot format

What exists, WNM-3D side (\`verified\`, \`7fe2d5f\`):
- \`scripts/data/convert_interiorgs_to_lerobot.py\` joins GN0-rendered videos (\`<render-root>/<scene>/<trajectory_id>.mp4\`) with the GN-Matrix annotation parquet by (scene, trajectory_id). Per episode it writes one parquet with \`observation.camera_extrinsic\` (a 4 by 4 camera-to-world matrix per frame), \`episode_index\`, \`frame_index\`, \`task_index\` and \`annotation.language.language_instruction\` (lines 33 to 35, 501 to 582), and one H.264 mp4 under \`videos/chunk-XXX/observation.images.rgb/\`. It keeps every third source frame by default (\`--frame-stride 3\`). Poses come from GN0's per-frame camera sidecars when present (\`frame_XXXX_camera.json\`, row-vector transforms that it transposes), otherwise from the annotated path. \`meta/info.json\` declares LeRobot \`codebase_version\` v2.0 and only three feature groups: the RGB video (with \`"has_audio": False\`), the extrinsic, and the language index (lines 645 to 746). \`meta/modality.json\` maps \`video.rgb\`, \`camera.extrinsic\` and the language annotation.
- The camera convention the converter builds (lines 319 to 337): world z is up; the rotation columns are (lateral, up, forward); translation is (x, y, camera height).
- The trainer does not read stored actions. \`gammanav/vln/data/dataset/lerobot_sharded.py\` (class \`ShardedLeRobotSingleActionChunkDatasetInteriorGS\`, docstring lines 32 to 41, code lines 577 to 606) computes \`action.nav_delta\` as frame-to-frame [dx, dy, dyaw] from the camera extrinsics, after projecting camera centers into the start frame, smoothing (window 7, 2 passes) and scaling by \`nav_action_scale\` 4.0. \`state.nav_pose\` is also virtual. The data config is \`gammanav/vln/configs/data/wnm_3d/interiorgs_wan22.yaml\` (33 frames, 320 by 160 after resize, 8 actions per block, 4 blocks).

What exists, GN0 side (\`verified\`, \`c927781\`):
- Stage I data is not a policy rollout. \`gn0/collection/trajectory_video.py\` (\`TrajectoryVideoCollector\`) takes the episode's reference path, smooths it (window 9), resamples it every 0.05 m, sets heading from a 0.5 m look-ahead with yaw smoothing, teleports the agent to each pose and renders one frame per pose at 10 frames per second (defaults in \`gn0/evaluation/local.py\` lines 153 to 168). With the converter's stride of 3 that is one kept frame per 0.15 m of path and 3.33 frames per second, which matches the checkpoint metadata (480 by 360 at about 3.33 frames per second, \`recalled\` from the 2026-11-18 survey). GN-Bench renders 480 by 360 RGB with a 70 degree field of view and a 1.3 m agent height (\`recalled\`, 2026-11-18 and 2026-11-19 surveys).

What exists, Habitat audio side (\`verified\`):
- \`sound-spaces/soundspaces/continuous_simulator.py\` (\`287184f\`): \`ContinuousSoundSpacesSim\` adds one audio sensor with \`channelLayout\` Binaural, two channels, \`indirectRayCount\` 500, \`temporalCoherence\` True, \`transmission\` True, and \`enableMaterials\` False (lines 115 to 126); after a scene change it loads \`mp3d_material_config.json\` (line 333); it places the source at the goal position plus 1.5 m height (line 341); it convolves the dry clip with the impulse response for the length of one step and cross-fades with the previous step's response over 0.05 s (lines 47 to 55, 413 to 456). In continuous mode a step is 0.25 m and 0.25 s (\`ss_baselines/av_nav/config/default.py\` lines 216 to 222).
- \`SoundSpaces2.md\`: the channel layout can be Mono, Binaural or Ambisonics; the simulation is Monte Carlo path tracing, so results "vary slightly between runs"; default sample rate 44100, default thread count 1.
- NaVLA \`replay_trajectory_with_audio.py\` (\`436ad6b\`): follows a ShortestPathFollower path on GOAT-Bench episodes in HM3D, adds a binaural audio sensor at 1.5 m with materials enabled, convolves and writes video with audio at 16 kHz (lines 133 to 158, 211 to 213, 281 to 308, 409 to 418). The scene path is hard-coded to one HM3D val scene (line 139), so it is a working sketch, not a pipeline.

What is missing:
- A Habitat renderer that reproduces GN0's Stage I recipe: get the shortest path from the navmesh, smooth and resample it the same way, teleport the camera to each pose, and render RGB at the checkpoint's resolution and field of view. At every kept frame, also render the impulse response and convolve the source sound.
- One-channel and two-ear audio "from the same render". A habitat-sim audio sensor has one channel layout. Either two sensors (Mono and Binaural) are attached to the same agent, or the one-channel version is made from the binaural signal. Which one is a design decision for the lead (see "Risks to the comparison").
- An audio clock. GN0's frame rate is a fixed playback rate assigned to poses spaced 0.05 m apart, not a physical speed. The renderer has to choose how much audio belongs to each kept frame and keep the sound continuous across frames (the cross-fade above shows one way).
- Coordinate conversion from Habitat (y up, minus z forward, per the docstring of \`ContinuousSoundSpacesSim.step\`) to WNM-3D's z-up camera-to-world convention, checked by recomputing \`action.nav_delta\` on a rendered episode with WNM-3D's own reader.
- A Habitat-to-LeRobot converter. It must reproduce the converter's outputs exactly: the per-episode parquet columns, the mp4 layout and codec, \`meta/tasks.jsonl\`, \`meta/episodes.jsonl\`, \`meta/info.json\` and \`meta/modality.json\`, one shared resolution and frame rate across episodes (the converter raises an error otherwise), and the instruction text. It must also add what the current schema has no slot for: per-frame audio (both versions), sample rate, and the target-visible-at-start flag. Storing audio needs a new modality in \`modality.json\` and a reader change in \`lerobot_sharded.py\`, which is model-side code (piece 7).
- An instruction. GN-Matrix episodes always carry a language instruction; audio episodes do not. The same instruction must be used for all three versions.

Effort: 4 to 9 working days (\`speculative\`). Assumes piece 1 works, the episodes come from piece 3, and the converter reuses the existing WNM-3D script's metadata code. The range is mainly the audio clock, the coordinate check and the audio storage format.

### 3. Episode sets for "walk to a sounding object", and the target-visible-at-start flag

What exists (\`verified\` unless marked):
- SoundSpaces AudioGoal (Matterport3D and Replica): a goal position and a sound id; the target has no visual presence (\`recalled\`, 2026-11-18 survey). It suits "go to the sound", not "go to a sounding object you might see".
- SAVi semantic AudioGoal (Matterport3D only): config \`configs/semantic_audionav/savi/mp3d/semantic_audiogoal.yaml\` uses the grid simulator \`SoundSpacesSim\` with 1 m grid, 128 by 128 images, success 1 m measured to \`VIEW_POINTS\`, sounds from \`data/sounds/semantic_splits\`. Episode fields (\`soundspaces/tasks/semantic_audionav_task.py\`, \`recalled\` from the 2026-11-18 survey): \`object_category\`, \`sound_id\`, \`distractor_sound_id\`, \`offset\`, \`duration\`; goals carry \`view_points\`, positions from which "the object is visible". The dataset loader has a continuous hook (\`soundspaces/datasets/semantic_audionav_dataset.py\` lines 213 to 217, lifts goal and view points by 0.1 m), but the SAVi trainer config defaults to \`CONTINUOUS = False\` and I did not find SAVi code that uses \`ContinuousSoundSpacesSim\`; that simulator's code has no distractor handling in what I read. Start positions are grid nodes, which is fine as start poses for an on-the-fly renderer (\`speculative\`).
- The episode files are a public download: \`datasets.tar.xz\` and \`sounds.tar.xz\` on \`dl.fbaipublicfiles.com/SoundSpaces/\` both answered HTTP 200 to a HEAD request on 2026-11-19, with Content-Length 120,012,800 and 671,098,880 bytes. The README gives 77M and 13M for these; I did not reconcile the difference. Whether \`datasets.tar.xz\` contains the semantic (SAVi) split was not checked.
- NaVLA2's MINav (HM3D, habitat 0.2.3 with audio, a GOAT-Bench fork): code only. No episodes are released, two dataset-building scripts are stubs, and the evaluation config points at a missing \`data/datasets/diff_bench\` folder (\`recalled\`, 2026-11-18 survey). Its replay script shows that GOAT-Bench object episodes on HM3D (with \`view_points\` per goal, line 77) can be given a sound at the object's position. MINav's task is an instruction chain with discrete actions.

What each lacks:
- AudioGoal: an object to see; an instruction; a visibility flag.
- SAVi: Matterport3D only; no instruction (it gives a category sensor); no visibility flag; built for the grid simulator, so continuous use with SoundSpaces 2.0 needs checking; its sounds stop after \`duration\`, which is part of the SAVi task and changes what "target audible" means.
- MINav: no episodes; HM3D episodes would have to come from GOAT-Bench or be generated.

The target-visible-at-start flag (\`speculative\` on the method): no episode set stores it. The most direct computation renders a semantic image at the start pose and checks whether the target instance covers at least some threshold of pixels, at the same field of view as the RGB camera. This needs semantic annotations: Matterport3D's semantic mesh or HM3D-Semantics v0.2 (the \`*.semantic.glb\` and \`*.semantic.txt\` files, habitat-sim \`DATASETS.md\`). A cheaper but weaker proxy is "the start position is near a \`view_point\` and the object center is inside the camera's field of view", which ignores occlusion. The threshold is a choice the lead must fix before results are seen.

What is missing:
- An episode list with train, validation and test splits by scene, a fixed instruction, sound assignments, and the flag. A choice between SAVi on Matterport3D, GOAT-style object goals on HM3D, or newly generated episodes on either.
- A check that the split by visibility gives enough episodes in each half.

Effort: 3 to 7 working days (\`speculative\`). Assumes the episodes are reused (SAVi or GOAT-Bench) rather than generated from scratch, and that semantic rendering works in the chosen habitat-sim build. Generating new episodes on HM3D would push this to the high end or beyond.

### 4. A Habitat client for WNM-3D's inference server, for closed-loop testing

What exists (\`verified\`):
- Server, \`WNM-3D/scripts/inference/wnm_3d_server.py\` (\`7fe2d5f\`): msgpack-numpy over WebSocket (lines 49, 632 to 640); accepts RGB under several keys, an optional \`state.nav_pose\` [x, y, yaw], an optional instruction, a \`session_id\` (lines 13 to 16, 343 to 380, 515 to 540); returns an (8, 3) \`action.nav_delta\` block (\`recalled\`, 2026-11-18 survey). It keeps its own frame history, and errors on frames that are not the checkpoint's raw resolution unless started with \`--resize-input-to-checkpoint-resolution\` (\`recalled\`, same survey). Launcher defaults: 8 replicas on 8 GPUs; one replica with \`--cuda-devices 0 --num-replicas 1\` (\`wnm_3d_server.sh\`; \`docs/INSTALLATION.md\`).
- Client, GN0 \`gn0/remote/client.py\` (\`c927781\`): a 234-line synchronous WebSocket client that depends only on msgpack, numpy and websockets, written so a policy "can run outside the GN0 simulator environment" (docstring).
- Agent, GN0 \`gn0/agents/remote.py\` (\`RemoteNavAgent\`, line 102): builds the request from the observation (\`_query_new_actions\`, lines 481 to 571), keeps \`action_num\` actions of each returned block (default 1), stops when a block's total translation is below \`chunk_stop_translation\` (default 0.15 m), and turns each nav_delta into a pose action (lines 1180 to 1218). It imports \`GN_Bench.core.agent.Agent\` and GN-Bench's DAgger constants, so it cannot be used inside Habitat as it is.
- A Habitat harness from the same authors: GN0-VLN-CE (\`ac4ebed\`) runs Habitat VLN-CE evaluation and DAgger collection for GN-BAE, with \`gn0_vln_ce/env/habitat_env.py\` (loads legacy yacs or newer Hydra Habitat configs), \`gn0_vln_ce/agents/habitat_adapter.py\` (\`HabitatDaggerSimAdapter\`, line 58: "Adapter exposing BAE planner sim APIs on top of Habitat env/sim"), \`gn0_vln_ce/evaluators/ce_evaluator.py\`, and \`tools/build_occupancy_from_habitat.py\` (builds GN-Bench-style occupancy maps from the Habitat navmesh). Its documented install is habitat-sim and habitat-lab v0.1.7 in Python 3.9 (\`docs/INSTALLATION.md\`); \`configs/habitat/vln_r2r_hydra.yaml\` is labelled "for newer Habitat-Lab releases". It has no WebSocket client and no WNM-3D code (\`recalled\`, 2026-11-18 survey).

What is missing:
- A Habitat-side loop: reset an episode in the audio-enabled habitat-sim, render RGB (and audio), send them with the instruction and session id, apply the returned actions (piece 5), record success, distance and path length, and log the visibility flag per episode.
- An audio field in the request and in the server's \`_convert_observation\`. That server change belongs with the model-side work (piece 7).
- Version reconciliation: GN0-VLN-CE's harness targets habitat v0.1.7, SoundSpaces 2.0 targets v0.2.2 or v0.2.3. Porting the harness pieces that are needed (episode loop, metrics) is part of this piece.

Effort: 2 to 4 working days (\`speculative\`). Assumes the GN0 client is reused unchanged, the GN0 agent logic is ported without the DAgger parts, and the success metric is defined in the episode set of piece 3.

### 5. Converter from WNM-3D's (dx, dy, dyaw) output to Habitat actions

What exists (\`verified\` unless marked):
- In GN-Bench, nav_delta is not turned into discrete moves. \`RemoteNavAgent._nav_delta_to_env_action\` (GN0 \`gn0/agents/remote.py\` lines 1180 to 1218) rotates (dx, dy) into the world frame (by default from the pose at the start of the chunk), adds it to the position, applies dyaw, and sends a pose. \`GN_Bench_simulator.step\` (GN-Bench-Tools \`0ce8367\`, \`GN_Bench/sims/GN_Bench_simulator/GN_Bench_simulator.py\` lines 819 to 870) accepts that pose directly and then clips the motion against the occupancy grid. So WNM-3D's own benchmark executes continuous motion with collision clipping.
- The paper's VLN-CE converter (appendix B.2, \`recalled\` from the 2026-11-18 survey): accumulates the predicted (dx, dy) into a polyline and tracks it with a map-free kinematic beam search, 0.25 m forward moves, 15 degree turns on R2R, beam width 32, up to four forward moves per expansion, executing the first four primitives before replanning. Not released.
- GN0-VLN-CE \`gn0_vln_ce/agents/mpc.py\` (\`ac4ebed\`): \`smooth_polyline\`, \`Polyline2D\`, and \`plan_actions_beam_mpc\` ("Receding-horizon beam-search MPC to produce discrete actions", returns FWD, LEFT and RIGHT, no STOP), with defaults \`step_m\` 0.25, \`turn_deg\` 15.0, \`horizon\` 32, \`beam\` 300. It is the same kind of controller the paper describes, but its default beam width (300) is not the paper's (32), and the paper's per-expansion limit of four forward moves is not a parameter I saw. Whether it is the paper's code is not stated anywhere I read.

What is missing:
- A choice between two execution rules, which is the lead's: (a) continuous pose execution like GN-Bench, using the Habitat navmesh to stop or slide at walls (this keeps WNM-3D's training-time motion model, and SoundSpaces 2.0 can render audio at any position); or (b) discrete 0.25 m and 15 or 30 degree moves through a beam-search tracker like the paper's VLN-CE setup (this matches SoundSpaces' and VLN-CE's standard action spaces). The rule changes how comparable the numbers are to other papers, not the fairness between the three versions, as long as all three use it.
- A stop rule. GN0 stops on low chunk translation; SAVi measures success within 1 m of the view points. The two need to be reconciled.

Effort: 1 to 6 working days (\`speculative\`). The low end is rule (a), a port of about 40 lines of GN0 logic plus navmesh collision handling. The high end is rule (b) with \`mpc.py\` adapted and tuned to match the paper's described settings.

### 6. The transfer risk: fine-tuning a checkpoint trained only on InteriorGS on Habitat scenes

What exists (\`verified\`):
- The released checkpoints were trained only on InteriorGS LeRobot data (\`recalled\`, 2026-11-18 survey: the stage-3 \`experiment_cfg/conf.yaml\` lists only \`interiorgs_lerobot_seen\`; the model card says performance may not transfer to new simulators).
- \`docs/TRAINING.md\` (\`7fe2d5f\`): Stage I is offline supervised training on A* expert renders; Stage II is DAgger (roll out the Stage I policy, query the A* expert at visited states, fine-tune on the corrections); Stage III (reward-guided, DanceGRPO) lives in WNM-3D-RL. Continued training uses \`PRETRAINED_MODEL_PATH\`, and directories ending in \`_release\` "are not used by default for continued training".
- \`scripts/train/wnm_3d_base.sh\`: full fine-tuning (\`save_lora_only=false\`, line 221); defaults 8 GPUs, per-device batch 12, learning rate 1e-5; Stage I \`MAX_STEPS\` 200,000 and Stage II 70,000 (\`wnm_3d_stage1.sh\`, \`wnm_3d_stage2.sh\`). The launcher requires the VGGT-Omega checkpoint file to exist before any run (lines 142 to 146) and requires \`model.safetensors\` or \`model.safetensors.index.json\` under \`PRETRAINED_MODEL_PATH\` (lines 147 to 152). It also downloads Wan2.2-TI2V-5B, Wan2.1-I2V-14B-480P (for the image encoder) and the UMT5-XXL tokenizer if missing.
- Hugging Face API, 2026-11-19: \`TeleEmbodied/WNM-3D\` is not gated (revision \`5b8c08d\`); \`wnm_3d_stage3_release/\` holds six safetensors shards and \`model.safetensors.index.json\`, so it would pass the launcher's file check. \`facebook/VGGT-Omega\` is gated with manual approval (revision \`1041e80\`).
- \`docs/INSTALLATION.md\`: each released stage checkpoint is about 26 GiB; one server replica used about 27 GiB of GPU memory on an H100 80 GB; other GPU counts "should be validated for both memory use and effective global batch size".

What fine-tuning on Habitat scenes would involve, as the docs describe it (\`speculative\` beyond what is quoted): a Stage-I-style supervised run on the Habitat LeRobot data from piece 2, started from a released checkpoint through \`PRETRAINED_MODEL_PATH\`, with \`INTERIORGS_DATA_ROOT\` pointed at the Habitat data. Optionally a Stage-II-style DAgger round in Habitat, which needs an A* or navmesh expert on Habitat states; GN0-VLN-CE's \`HabitatDaggerSimAdapter\` and its occupancy builder already do this for GN-BAE on habitat v0.1.7. Stage III is out of reach for this budget (\`speculative\`). Every one of the three versions must get the same fine-tuning, so the no-audio version is also fine-tuned on Habitat data, not taken as released.

What is missing:
- The Habitat dataset (piece 2), the VGGT-Omega access, and a GPU configuration with a batch size and step count that fit the budget. The documented recipe is 8 GPUs; any smaller setup changes the effective batch and must be validated (the docs say so).
- A decision whether to start from the released Stage III checkpoint or an earlier stage, and whether to include a Habitat DAgger round.

Effort: 2 to 5 working days of engineering for the supervised fine-tuning path (\`speculative\`), not counting GPU time. Add 3 to 7 working days if a Habitat DAgger round is wanted (\`speculative\`, porting GN0-VLN-CE's adapter to the audio habitat-sim version).

### 7. Model side: where an audio input would attach (location only)

No encoder is designed here. Files and places, all \`verified\` at WNM-3D \`7fe2d5f\`:
- \`gammanav/vln/model/wnm_3d/modules/wan_video_dit_action_casual_chunk.py\`, class \`CausalWanModel\`: the conditioning sequence for cross-attention is built from the text embedding and, if present, a CLIP image embedding concatenated in front (\`context = self.text_embedding(context)\`, then \`torch.cat([clip_embedding, context], dim=1)\`, lines 1919 to 1924); every block applies \`self.cross_attn(self.norm3(x), context)\` (line 1621). Audio tokens projected to the same width could join this \`context\` the same way the CLIP tokens do.
- Same file, lines 2101 to 2127: \`past_obs_tokens\` (from the VGGT geometry adapter) are prepended to the video token sequence and get their own rotary position frames. This is the second existing pattern for an extra observation stream, with per-frame timing.
- \`gammanav/vln/model/wnm_3d/action_head/wan_flow_matching_action_tf.py\`, class \`WANPolicyHead\`: \`forward\` (line 1135) encodes the prompt (line 1179) and builds \`past_obs_tokens\` (line 1225) before calling the DiT with \`context=prompt_embs\` and \`past_obs_tokens=...\` (lines 1403 to 1422); the inference path does the same around lines 1733 to 1789. An audio encoder module and its call would sit here, next to \`_init_vggt_geometry_modules\` (line 493) and \`_build_vggt_past_obs_tokens\` (line 732).
- \`gammanav/vln/model/wnm_3d/transform/wnm_3d_cotrain.py\`: turns a sample into \`images\` and \`text\` (lines 183 to 188, 219 to 242); audio would need to pass through here.
- \`gammanav/vln/data/dataset/lerobot_sharded.py\` and \`gammanav/vln/configs/data/wnm_3d/interiorgs_wan22.yaml\`: a new audio modality aligned to the sampled frame indices.
- \`scripts/inference/wnm_3d_server.py\` (\`_convert_observation\`, line 343) and \`gammanav/vln/model/wnm_3d/inference_policy.py\` (\`WNM3DInferencePolicy\`): accept and buffer audio at inference time.

No effort figure: the encoder and its attachment are the research contribution and are out of scope for this sizing.

### Would UniWAM remove any of these pieces? (not sized)

UniWAM (\`recalled\`, 2026-11-18 survey: Wan2.2-TI2V-5B, navigation trained and evaluated in Habitat on HM3D v0.2 through a WebSocket policy interface, habitat-sim 0.3.3, checkpoints and data gated by an automatic Hugging Face agreement, 16 GPUs in the report's runs) would remove most of piece 4 (its Habitat client exists) and much of piece 6 (it is already trained on HM3D, so no InteriorGS-to-Habitat jump), and its relative SE(2) outputs are already executed in Habitat, which may cover piece 5 (how it executes them was not read). It would not remove pieces 1, 2 and 3: it has no audio, its habitat-sim 0.3.3 is not SoundSpaces 2.0's pinned build (habitat-gs shows the audio source exists at 0.3.3, \`verified\`, but nobody here has built it), and its task is instruction navigation to a pose near an object, not a sounding object. Piece 7 would be redone in a different codebase. It is also not WNM-3D, which the PI's messages take as the base (2026-11-19 LOG entry).

## Data, licences and access

Scene data (one is needed; \`verified\` from the pages named):
- Matterport3D: download requires signing the Terms of Use agreement form "using your institutional email addresses" and sending it to matterport3d@googlegroups.com; access is then granted by the maintainers (niessner/Matterport \`README.md\`, \`8cd7c81\`). The Habitat package is fetched with \`download_mp.py --task habitat\`, which "requires python 2.7", plus a separate scene dataset config file (habitat-sim \`DATASETS.md\`). SoundSpaces' Matterport3D-based task datasets and models are distributed under the Matterport3D Terms of Use and CC BY-NC-SA 3.0 US (\`sound-spaces/README.md\`). One Matterport3D example scene is downloadable without access (\`mp3d_example_scene\`, \`DATASETS.md\`).
- HM3D v0.2: "available exclusively for academic, non-commercial uses"; access through a request at matterport.com, then a Matterport API token used by habitat-sim's download utility; terms are the Matterport academic end-user licence (matterport/habitat-matterport-3dresearch \`README.md\`, \`9b74d5a\`; habitat-sim \`DATASETS.md\`). Stated sizes: train habitat files 27G, train semantic annotations 8.1G, val habitat files 3.3G, val semantic annotations 2.0G, minival habitat 390M.
- Which one fits depends on piece 3: SAVi episodes are Matterport3D; MINav and GOAT-Bench are HM3D.

Audio and episodes:
- SoundSpaces code: CC-BY-4.0 (\`sound-spaces/README.md\`). RLR library: CC-BY-NC, Linux x64 only (\`recalled\`, 2026-11-19 survey Part A). The material file \`mp3d_material_config.json\` ships with RLR.
- SoundSpaces episode and sound files: public downloads (HEAD returned HTTP 200, see piece 3). SoundSpaces 1.0's 867G of pre-rendered binaural impulse responses (\`soundspaces/README.md\`) are not needed when audio is rendered on the fly with SoundSpaces 2.0 (\`speculative\`, since the continuous simulator renders through the audio sensor).

Model weights:
- WNM-3D checkpoints: not gated; checkpoints are under the FAIR Noncommercial Research License and the DINOv3 License because they embed VGGT-Omega and DINOv3-derived parameters (\`docs/MODEL_CARD.md\`, \`7fe2d5f\`). Repository code is Apache-2.0 with third-party parts under their own terms (\`COPYRIGHT\`).
- VGGT-Omega (\`facebook/VGGT-Omega\`): gated, manual approval (Hugging Face API, 2026-11-19). The training launcher will not start without it (piece 6). This is a hard access requirement for any fine-tuning.
- Wan2.2-TI2V-5B, Wan2.1-I2V-14B-480P, UMT5-XXL: downloaded by the launcher (\`wnm_3d_base.sh\`); their gating was not checked this session.

All of this is non-commercial research use, which fits a university paper (\`speculative\`, not a legal reading).

## Compute and storage

Only figures a source states are given. Nothing here was timed.

- Budget: 200 US dollars of Vast.ai credit (\`hear-lab/STANDARDS.md\`). Not turned into hours here.
- Training recipe as documented: full fine-tuning, 8 GPUs, per-device batch 12, Stage I up to 200,000 steps, Stage II up to 70,000 steps (\`wnm_3d_stage1.sh\`, \`wnm_3d_stage2.sh\`, \`7fe2d5f\`). How many steps a Habitat fine-tune needs is unknown.
- Number of training runs: three versions, and \`_standards/07-ml-practices.md\` asks for at least 3 seeds per reported comparison (\`hear-lab/STANDARDS.md\`), so the comparison as specified is at least 9 fine-tuning runs, plus any Habitat DAgger rounds.
- Inference: one server replica used about 27 GiB of GPU memory on an H100 80 GB (\`docs/INSTALLATION.md\`). Closed-loop tests run the server and the Habitat client together.
- Rendering: RLR runs on CPU threads (\`threadCount\`, default 1, \`SoundSpaces2.md\`); \`ContinuousSoundSpacesSim\` uses 500 indirect rays with temporal coherence. RGB rendering in habitat-sim uses the GPU through EGL (\`recalled\`). The cost per rendered frame with audio was not found in any source and must be timed on a real scene.
- Storage, stated sizes: each WNM-3D stage checkpoint about 26 GiB (\`docs/INSTALLATION.md\`); training also writes checkpoints and optimizer states (\`SAVE_TOTAL_LIMIT\` 10 by default, \`wnm_3d_base.sh\`), size per saved step not stated; HM3D v0.2 train plus val habitat and semantic files 27G + 8.1G + 3.3G + 2.0G (\`habitat-matterport-3dresearch\` README); SoundSpaces episode and sound archives about 120 MB and 671 MB as served. Matterport3D's Habitat package size was not found. The Wan and UMT5 initialization weights and the rendered LeRobot data sizes were not found or estimated.
- Vast.ai specifics: instance disk is lost when the instance is destroyed (\`hear-lab/STANDARDS.md\`), so rendered data and checkpoints need to be copied off.

## Unknowns and the smallest real check for each

Each check uses the real tools and real data. None was run. Each needs the user's OK in the experiment session, and any rental needs the lead's OK first.

1. Does an audio-enabled habitat-sim build and render on a Vast.ai host (GLIBC, headless EGL inside a container)? Check: on one rented Linux GPU instance, follow \`sound-spaces/INSTALLATION.md\` (or NaVLA's \`install_env_bench.sh\`) and run \`examples/minimal_example.py\` on the public \`mp3d_example_scene\`; success is a written \`data/output.wav\` and an RGB frame. Needs: one Vast.ai GPU instance with root, no gated data.
2. Can one agent carry a Mono and a Binaural audio sensor at once, and how different are their impulse responses for the same pose? Check: on the same example scene, add both sensors, render at a few poses, compare the direct-path arrival times and levels, and render the same pose twice to see the run-to-run noise. Needs: the instance from check 1.
3. How long does one rendered frame with audio take at the chosen ray counts? Check: render one full SAVi or GOAT-style episode in one real scene at 480 by 360 with both audio versions and time it per frame. Needs: the instance plus Matterport3D or HM3D access (or the example scene for a first number, stated as such).
4. Does WNM-3D's reader recover sensible actions from Habitat poses? Check: convert that one rendered episode with the new converter, load it with WNM-3D's \`ShardedLeRobotSingleActionChunkDatasetInteriorGS\`, and compare the computed \`action.nav_delta\` with the known path (forward is positive dx, turns have the right sign). Needs: the \`wnm_3d\` environment on any machine; CPU is enough for the reader (\`speculative\`).
5. How far is the released checkpoint from Habitat scenes before any fine-tuning? Check: start one server replica with \`wnm_3d_stage3_release\`, send real Habitat frames from a few HM3D or Matterport3D val episodes through the new client, and look at the returned blocks and generated frames; then run a few closed-loop episodes without audio. Needs: one GPU with enough memory for the server (about 27 GiB observed), pieces 4 and 5 built.
6. Are the episode files what this route needs? Check: download the public \`datasets.tar.xz\`, list whether the semantic (SAVi) split is inside, and count episodes per split, scenes, categories and view points. Needs: no access, any machine.
7. How balanced is the visibility split? Check: compute the flag with the semantic-sensor method on the real validation episodes and count both halves; inspect a sample of start frames by eye against the flag. Needs: check 1's instance, scene and semantic data.
8. Will VGGT-Omega access be granted? Check: the user requests access on Hugging Face. Needs: the user's account; calendar time.
9. Does a single-GPU or few-GPU fine-tune of WNM-3D fit in memory, and what does one optimizer step cost? Check: one short real training run on Habitat data from \`wnm_3d_stage3_release\`, timing a real batch, as the standards require before any longer rental. Needs: VGGT-Omega access, the Habitat data from piece 2, one Vast.ai GPU instance sized from the server's observed memory and then measured.
10. Does SAVi's task logic work with \`ContinuousSoundSpacesSim\`, or must the episode loop be written fresh? Check: load SAVi episodes with \`CONTINUOUS True\` in the chosen stack and step one episode. Needs: check 1's instance and Matterport3D access.

## Risks to the comparison

These could make the no audio / one-channel / two-ear comparison unfair or unconvincing to a reviewer.

- What "one-channel" means. A mono sensor and a downmix of the binaural signal are different signals (the binaural one carries head-related filtering and level differences). If they come from two separate renders, Monte Carlo noise also differs. The memo cannot pick one; the lead should fix it and state it in the paper.
- Audio that names the target. In SAVi, each object category has its own sound. One-channel audio then tells the model what to look for even with no direction. A gain of one-channel over no audio could come from identity, not hearing. If the instruction text already names the object, audio adds less. The instruction must be identical across all three versions.
- Unequal training. The no-audio version must get the same Habitat fine-tuning, steps, data order and seeds as the audio versions. Comparing against the released checkpoint would credit audio with the gain from Habitat fine-tuning. A no-audio version that also lacks the audio encoder's extra parameters can be questioned; giving it a blank audio input of the same shape is one option (lead's choice).
- Seeds and budget. The standards ask for at least 3 seeds per reported comparison, so 9 runs. If the budget only covers one seed per version, a reviewer can say the difference is noise. NaVLA2's only mono against spatial result has no stated seeds (2026-11-19 LOG), so this is exactly the gap a reviewer will look at.
- Acoustic realism. \`ContinuousSoundSpacesSim\` sets \`enableMaterials\` False, so sound uses the default material unless that is changed; materials come from label matching even when enabled. The same for all versions, but a reviewer may ask whether the two-ear advantage survives realistic acoustics. Stochastic impulse responses should be rendered once and stored, so that all versions train on the same audio.
- The visibility flag. Its definition (pixel threshold, field of view, occlusion) decides the split. It must be fixed before results are seen. Start-visible episodes may be easy for every version, which would compress the difference there.
- Task too easy with audio. Without distractor sounds, a binaural model might solve the task by turning toward the loudest ear. That is a real finding, but a reviewer may want a distractor condition (SAVi has \`distractor_sound_id\`).
- Out-of-distribution visuals. Habitat's scanned-mesh images differ from the splat renders WNM-3D was trained on. This is the same for all three versions, but a weak no-audio baseline caused by poor transfer would inflate the audio gains. Reporting the no-audio Habitat success next to known WNM-3D numbers helps here.
- Execution rule. Discrete conversion (piece 5, rule b) adds a controller between the model and the world; continuous pose execution (rule a) is closer to training. Either is fair if all versions use it, but absolute numbers will not compare with VLN-CE or SAVi papers unless the rule matches theirs.
- Scene leakage. Train and test episodes must come from different scenes.

## Summary

Total effort for pieces 1 to 6: about 13 to 35 working days for one person with coding agents, or up to about 42 with a Habitat DAgger round (\`speculative\`). This excludes the audio encoder itself (piece 7, the research contribution), GPU time, and calendar time waiting for dataset and model access.

The three biggest risks:
1. Access and environment. The route needs Matterport3D or HM3D access, gated VGGT-Omega access (the training launcher will not start without it), and an archived audio branch of habitat-sim that must build on a rented Linux container.
2. Budget against the specified comparison. The documented training recipe is 8 GPUs and hundreds of thousands of steps, the comparison needs at least 9 fine-tuning runs by the standards, and the budget is 200 dollars. Rendering speed with audio is unknown. None of this has been timed.
3. Transfer from InteriorGS to Habitat. The released weights never saw Habitat or scanned-mesh scenes. If the fine-tuned no-audio version is weak in Habitat, the audio comparison runs on a poor base, and reviewers can read gains as repair of a broken baseline.

What would make this route fail outright:
- No access to either Matterport3D or HM3D, or VGGT-Omega access refused.
- No Vast.ai host on which the audio-enabled habitat-sim builds and renders headless.
- A timed batch showing that even one fine-tuning run per version does not fit in the 200 dollars, with no more budget available.
