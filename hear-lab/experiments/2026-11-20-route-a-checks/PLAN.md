# Plan: route A bring-up checks (sound engine, timed training step, visible starts)

Written and committed before running (2026-11-20, experiment session `hear-lab-experiment`). Once the experiment starts, don't edit this design: add changes below as dated amendments.

**Source of the task:** the planning lead's hand-off of 2026-11-19 (direction `LOG.md`, entry "the three route A checks handed to the experiment session"), on the lead's "yes" in the planning window. Unknowns are numbered as in `writing/2026-11-19-sizing-route-a-audio-in-wnm3d-simulator.md` ("the route A memo"). The lead's OK to write this plan and start the check 1 and check 3 code: experiment window, 2026-11-20 ("yes on both"). No rental is approved by this plan; each rental needs the lead's OK in the experiment window, with machine, price, hours and total stated first (`../../STANDARDS.md`).

**Question:** can route A (WNM-3D's own simulator, GN-Bench on InteriorGS, plus a sound engine) be built and afforded? Three checks feed decision `003` (route A or route B). None of them is a research result about audio.

## Check 1: does the sound engine run, and does the geometry line up (unknowns 1 and 2)

**Design.**
- Engine: the bare RLR-Audio-Propagation library at pinned commit `4fd446b` (github.com/facebookresearch/rlr-audio-propagation), `libRLRAudioPropagation.so` loaded through a ctypes binding written against its C header. No habitat-sim, no SoundSpaces Python code.
- Machine: a rented Linux x64 machine on Vast.ai (the library ships Linux x64 binaries only).
- Geometry: walls extruded from one real InteriorGS scene's `structure.json` (door and window openings left open), a floor and a ceiling per room, furniture as boxes from `labels.json`. Materials from RLR's shipped `mp3d_material_config.json` by label substring; anything unmatched gets the config's default. The material map is "assumed materials" and is written to the results.
- One source (at the box centre of the goal object of one real GN-Matrix episode in that scene) and one listener (at that episode's start pose, height 1.3 m, heading from the episode). Write one one-channel (mono listener) and one two-ear (binaural, the library's default HRTF) impulse response as WAV files, 44100 Hz (the library default).
- Settings: run the same source and listener twice and write both, so that the choice stays open: (a) the bare library defaults (transmission on, 5000 indirect rays, diffraction on, max diffraction order 10; header lines 118 to 154 at `4fd446b`), and (b) SoundSpaces' continuous simulator settings (transmission on, 500 indirect rays; `soundspaces/continuous_simulator.py` lines 123 and 125 at `287184f`). Every setting passed to the library is logged with the output. Thread count and wall time per impulse response are logged for both.
- Floor plan overlay: for four InteriorGS scenes (the scene above plus three others that appear in the GN-Matrix test_unseen split, chosen by sorted scene id with a fixed seed of 20261120), draw the `structure.json` walls onto that scene's `occupancy.png` through GN-Bench's own world-to-pixel mapping (from `occupancy.json`, the same code path the simulator uses), save the four overlay images, and compute one number per scene: the share of points sampled every 5 cm along wall centre lines that land within 2 pixels of an occupied pixel.
- Same run, unknown 3: look up every `label_ins_id` of the four scenes' GN-Matrix episodes in that scene's `labels.json` and count misses.

**Pass (before running).** Check 1 passes if all hold: the library loads and returns finite, non-zero impulse responses; mono has one channel and binaural two; the first arrival in the mono response lies within 1 ms of straight-line distance / 343 m/s when the source is in line of sight (if the chosen episode's source is not in line of sight, a second listener placement in line of sight is added for this test); for a source placed 2 m to the listener's left in open floor, the left ear's first arrival comes before the right ear's; the four overlays line up by eye (walls on occupancy borders, no flip, no scale or offset error) and the wall share is at least 0.9 in each scene. Any failure is reported with the error and the overlay images, and work stops for the lead before checks 2 and 3 spend money (the hand-off's stop rule). Fixable misalignments (a constant offset, a flip) are reported as such; fixing them is not done without the lead's OK, since the route A memo's fallbacks are the lead's call.

**Prediction:** I expect the library to load (`speculative`: it is archived since 2022 and its binary may need an older glibc). I have no expectation on the overlay.

## Check 3: how many starts have the target visible (unknown 4)

**Design.**
- Episodes: the full GN-Matrix `InteriorGS_test_unseen` split and a random sample of 1,000 episodes of `InteriorGS_train_seen` (numpy default_rng, seed 20261120, sampled without replacement by row). GN-Matrix and InteriorGS at the revisions downloaded, hashes recorded.
- At each episode's start pose (position, heading, camera height 1.3 m), render the depth image with GN-Bench's own simulator (`get_observations_at`, the episode's own world-to-splat transform) using the InteriorGS config's camera: 480 by 360, 70 degree horizontal field of view.
- Target box: the goal instance's oriented box from `labels.json`, found by `label_ins_id`. Episodes whose id is missing are counted and reported, not dropped silently.
- Proposed visibility rule (for this count only; the rule for the paper is set later in the experiment PLAN for the three-version comparison): sample points on the box surface on a grid of at most 5 cm spacing; project each into the image with the camera's pinhole intrinsics; a point is seen if it falls inside the image, lies in front of the camera, and its depth is at most the rendered depth at that pixel plus 0.15 m. The target is visible at the start if at least one point is seen. Also written per episode: the number of seen points, the share of the box's sampled points that are seen, and the result with tolerances 0.05 m and 0.30 m, so that a stricter or looser rule can be read off later without re-rendering.
- Also per episode: the start-to-goal straight-line distance and the geodesic path length from the GN-Matrix path, and a flag for starts within the 3.0 m success radius (`SUCCESS_DISTANCE`).

**Output:** one CSV per split with the flags above, and a summary of counts per split: visible, not visible, missing id, and how many of each start within 3.0 m. Ten start images (RGB and depth with the projected box drawn) per split are saved for a by-eye check of the rule.

**Pass (before running):** the counts are produced for both splits, and the saved images agree with the flag by eye in at least 9 of 10 per split. There is no target count; the counts are what the check reports.

**Prediction:** `speculative`, I expect most GN-Matrix starts to have the target out of view, since starts were not chosen for visibility.

## Check 2: one timed real training step (unknown 9)

**Design.** Start WNM-3D training (`scripts/train/wnm_3d_stage2.sh` path, i.e. the released launcher with `PRETRAINED_MODEL_PATH`) from the released stage 3 checkpoint (`TeleEmbodied/WNM-3D`, `wnm_3d_stage3_release`), no audio, on a few real InteriorGS train_seen episodes rendered with GN0's `render_dataset.sh` and converted with WNM-3D's `convert_interiorgs_to_lerobot.py`, both as released. Time real batches after warm-up and record seconds per step, peak GPU memory, and the largest per-device batch that fits, with every launcher setting that differs from the documented recipe (8 GPUs, per-device batch 12, ZeRO-2) written next to the numbers.

**Not decided yet (the lead settles these before check 2 is planned in detail and before its rental):** the GPU type and count it is timed on (recommended: one H200, because no affordable 8-GPU machine is on offer), and whether the launcher's VGGT-Omega file check stays as released (recommended: yes, download the real file; the lead's account has access). Check 2's detailed design is added as a dated amendment once these are settled, before it runs.

**Pass:** the step runs and the numbers above are recorded; if it does not fit in memory, that is the result, reported with the error.

## Compute

Vast.ai, on-demand, HEAR Lab team account. Checks 1 and 3 and the episode rendering for check 2 share one cheap single-GPU Linux machine (one rental, asked for separately); check 1 runs first and work stops if it fails. Check 2 is a second rental. Each rental's machine, price, hours, total and actual cost go in `LOG.md`. Results and logs are copied off each machine and checked before it is destroyed. SSH access is attached per instance (`vastai attach ssh`), because Vast.ai does not allow SSH keys on a team account (error 400, 2026-11-20).

**Status:** planned.

## Amendments
