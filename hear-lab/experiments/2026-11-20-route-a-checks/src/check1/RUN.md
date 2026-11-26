# Check 1: run instructions (route A bring-up checks)

What this does, in one breath: it builds a rough "acoustic shell" of one real InteriorGS apartment (walls with the door and window gaps left open, floor, ceiling, furniture as boxes), puts a sound at the goal object of one real GN-Matrix episode and a listener at that episode's start pose, and asks the archived RLR-Audio-Propagation library for the impulse response (the echo pattern a click would make) twice, once with the library's own defaults and once with SoundSpaces' settings. It then checks that the sound arrives when physics says it should and that a sound on the left reaches the left ear first, and separately draws the floor-plan walls on GN-Bench's occupancy map to show whether the geometry sits where the simulator thinks it does.

Design: `../../PLAN.md` (check 1). Parameters: `../../config.yaml` (`seed`, `check1`). This file says how to run it. Outputs go to this experiment's `results/` (`--out ../../results/<run-name>` below resolves to `results/<run-name>/` of the experiment folder). `results/.gitignore` ignores everything except a short allow list (sha256 manifests, `tests.json` of the check 1 rerun, the check 2 step summary); images, audio, meshes, floor plans and the check 3 per-episode CSVs (they hold InteriorGS labels and coordinates) stay out of git (policy and the user's answer of 2026-11-24 in `hear-lab/data/README.md`). Do not force-add anything.

## Files

Since 2026-11-26, `rlr_binding.py`, `geometry.py`, `gnbench_copy.py` and `poses.py` live in `hear-lab/src/` (shared with Step 2); the scripts here add that folder to `sys.path` (`HERE.parents[3] / "src"`, or `$HEAR_LAB_SRC` if set). A full clone of the repository (step 2 below) has it in place.

| file | what it is |
| --- | --- |
| `rlr_binding.py` | ctypes binding to `libRLRAudioPropagation.so`, written from `RLRAudioPropagationPkg/headers/RLRAudioPropagation.h` at `4fd446b` |
| `geometry.py` | `structure.json` + `labels.json` to triangles in GN-Bench's world frame, one RLR material category per face |
| `gnbench_copy.py` | GN-Bench's frame and pixel functions, copied verbatim with file and line numbers (why copied: see its docstring) |
| `overlay.py` | walls drawn on `occupancy.png` through GN-Bench's world-to-pixel mapping, wall share, `label_ins_id` lookup |
| `poses.py` | GN-Bench heading to the RLR listener quaternion |
| `pass_tests.py` | the PLAN's pass tests, printing PASS/FAIL with the numbers |
| `run_check1.py` | the runner |
| `test_local.py` | local unit test (no RLR library needed) |
| `requirements-check1.txt` | pinned Python packages |

## Frames (what was found in GN-Bench's code)

- GN-Bench (GN-Bench-Tools `0ce8367`, `GN_Bench/sims/GN_Bench_simulator/GN_Bench_simulator.py`) never reads `structure.json` or `labels.json`. The agent lives in a "world" frame (the GN-Matrix `start_world_*`, `goal_world_*`, `path_raster_world_*` columns). It reaches the splat ("GS") frame through a per-episode affine fit of the episode's `path_raster_world` against `path_raster_pixel` (`derive_affine_transform`, line 271) and `transform_from_world_to_GS` (line 304); z is not changed. The agent stands at z = 1.3 (`_load_initial_state`, line 775) and looks along its rotation applied to local +x, with rotation `R.from_euler("z", heading_degrees)` (`_render_observations`, line 912). So forward is (cos h, sin h, 0) in world, matching the parquet's `start_facing_unit_dx/dy`.
- World to occupancy pixel is `transform_from_world_to_pixel` (line 1413): `px = (-wx - lower_x)/scale_x - 0.5`, `py = (upper_y + wy)/scale_y - 0.5`, with `scale = span / image size` from `occupancy.json` `lower`/`upper` and the PNG size.
- `structure.json`, `labels.json` and `occupancy.json` share one frame (the InteriorGS frame; right-handed, z up, metres per the dataset card); a door box in `labels.json` sits exactly in a gap of a `structure.json` room outline in scene `0001_839920`. This code treats that frame as GN-Bench's GS frame and maps it to world with GN-Bench's own `transform_from_GS_to_world` (line 319) and the simulated episode's affine. For `0001_839920` the fitted affine amounts to about x_world = x_GS - 7.2, y_world = y_GS + 4.8 with a scale of about 1.007 (not exactly 1, because `derive_affine_transform` uses `occupancy.json` `scale` = 0.05 while the PNG's real pixel size is span / size, about 0.0503), and no mirror (determinant positive). All 38 test_unseen episodes of that scene give the same affine.
- Evidence that this is the right frame, from the local runs on `0001_839920` (we ran them): the walls drawn through GS to world to pixel land on the map's wall and border pixels with no flip, offset or scale error visible, and every episode start lands on a free pixel. Frames tried by hand before writing the code (a y flip only, or no flip) put the walls visibly off the map's walls.
- RLR listener orientation: a [w, x, y, z] quaternion from listener-local to world, with the library's default local axes (right +x, up +y, back +z). `poses.py` maps those onto the agent's right, up (+z) and back, so no HRTF axis setting is changed.

## On the rented machine (Vast.ai, Linux x64, fresh Ubuntu 22.04 or 24.04)

A CPU is enough for check 1. The lines below assume a root shell in `/workspace`.

1. System packages:

       apt-get update && apt-get install -y git python3 python3-venv python3-pip file binutils

2. This repository at the commit the lead names (the repo is public), and the pinned library:

       cd /workspace
       git clone https://github.com/michaelnkr808/research26-27.git
       git -C research26-27 checkout <COMMIT>       # the commit of this branch the lead names
       git clone https://github.com/facebookresearch/rlr-audio-propagation.git
       git -C rlr-audio-propagation checkout 4fd446b
       file rlr-audio-propagation/RLRAudioPropagationPkg/libs/linux/x64/libRLRAudioPropagation.so
       sha256sum rlr-audio-propagation/RLRAudioPropagationPkg/libs/linux/x64/libRLRAudioPropagation.so
       ldd rlr-audio-propagation/RLRAudioPropagationPkg/libs/linux/x64/libRLRAudioPropagation.so

   `file` must say "ELF 64-bit LSB shared object, x86-64" (if it says ASCII text, it is a Git LFS pointer: stop and report). The local read-only copy of this file is 7.3 MB. Its version strings ask for at most GLIBC_2.29 and it links only libc, libm, libpthread, libgcc_s and ld-linux, so Ubuntu 20.04 or newer should satisfy it (`speculative` until `ldd` shows no "not found").

3. Python environment:

       cd /workspace/research26-27/hear-lab/experiments/2026-11-20-route-a-checks
       python3 -m venv /workspace/venv-check1 && . /workspace/venv-check1/bin/activate
       pip install -r src/check1/requirements-check1.txt huggingface_hub
       pip freeze > /workspace/check1-pip-freeze.txt

4. Data (needs a Hugging Face token whose account accepted the InteriorGS terms; `hf auth login` asks for it). Only the four small files per scene are needed, no splats. The four overlay scenes are the main scene plus three that `overlay.choose_overlay_scenes` picks from the test_unseen scene ids with seed 20261120; a dry run of the code on the Mac picked `0470_840871`, `0748_841603`, `0756_840012`. The runner recomputes and logs them; if its list differs from this one, download what it names.

       hf auth login
       for s in 0001_839920 0470_840871 0748_841603 0756_840012; do
         hf download spatialverse/InteriorGS --repo-type dataset --local-dir /workspace/data/InteriorGS \
           --include "$s/structure.json" "$s/labels.json" "$s/occupancy.json" "$s/occupancy.png"
       done
       hf download TeleEmbodied/GN-Matrix --repo-type dataset --local-dir /workspace/data/GN_Matrix \
         --include "*InteriorGS_test_unseen.parquet"
       find /workspace/data -name InteriorGS_test_unseen.parquet

   The scene folder layout `InteriorGS/<scene>/{structure,labels,occupancy}.json, occupancy.png` is from the InteriorGS dataset card. The path of the parquet inside the GN-Matrix repo was not checked (hence the wildcard and `find`). The `hf` command syntax is `recalled`, not tested here; if `hf` is not found, the older name is `huggingface-cli download` with the same arguments.

5. Local unit test on the machine (same test as on the Mac):

       cd src/check1
       CHECK1_SCENE_DIR=/workspace/data/InteriorGS/0001_839920 \
       CHECK1_PARQUET=<path printed by find> \
       CHECK1_RLR_ROOT=/workspace/rlr-audio-propagation \
       CHECK1_OUT=/workspace/check1-unit-test-out \
       python -m unittest -v test_local

6. Dry run (everything except the library), then the real run. `--out` must not exist; results are never overwritten.

       python run_check1.py --rlr-root /workspace/rlr-audio-propagation \
         --scenes-root /workspace/data/InteriorGS --parquet <path printed by find> \
         --scene 0001_839920 --out /workspace/check1-dryrun --no-audio

       REPO_COMMIT=$(git -C /workspace/research26-27 rev-parse HEAD) \
       python run_check1.py --rlr-root /workspace/rlr-audio-propagation \
         --scenes-root /workspace/data/InteriorGS --parquet <path printed by find> \
         --scene 0001_839920 --out ../../results/check1-run1 2>&1 | tee /workspace/check1-run1-console.txt

7. Copy `results/check1-run1/`, `/workspace/check1-run1-console.txt` and `/workspace/check1-pip-freeze.txt` back to the Mac and check them before the machine is destroyed. Keep them out of git (see the top of this file).

## Expected outputs (`results/check1-run1/`)

- `check1.log`: everything printed, including every configuration field set (`[rlr] set ...`) and the full configuration passed to `RLRA_CreateContext`, the episode, the affine, timings, and one PASS/FAIL line per test.
- `run_info.json`: commit, config and its hash, hashes of the library, material JSON, parquet and scene files, Python and platform, episode, source and listener positions, struct sizes.
- `mesh_report.json`, `assumed_materials.csv` (the "assumed materials" table: label, category passed to the library, material picked, number of label matches, tie flag), `scene_mesh_from_library.obj` (the library's own dump of the mesh it received).
- One folder per settings set, `library_defaults/` and `soundspaces_continuous/`: `ir_mono.wav` (1 channel), `ir_binaural.wav` (2 channels), at 44100 Hz, 32-bit float as the library writes them; `.npy` copies; `*_metrics.txt` (the library's metrics, RT60 and so on); `ir_mono_line_of_sight.wav` only if the start was not in line of sight; `ir_binaural_left_source.wav`; `run_record.json` (configuration defaults and values used, wall time and thread count per impulse response, placements).
- `overlay/`: `<scene>_overlay.png` (occupancy map enlarged 4x; walls red, doors blue, windows and other openings cyan, magenta squares where a wall sample is more than 2 pixels from an occupied pixel), `<scene>_overlay_summary.json`, `<scene>_label_lookup.csv`, for the four scenes.
- `tests.json`: all test results.

Console lines to expect, per settings set: `PASS ir_mono_basic`, `PASS ir_binaural_basic`, `PASS first_arrival_vs_distance` (measured vs expected delay in ms), `PASS left_ear_before_right_ear` (left and right onsets); then per scene `PASS/FAIL wall_share`, and a final `[done] n PASS, m FAIL`. The by-eye overlay check is not automated: look at the four overlay images.

What the dry run on the Mac already showed for `0001_839920` (we ran it, `--no-audio`): the seed picks episode trajectory_id 465 (goal "Dining plate", ins_id 465); mesh of 18 walls, 2 floors, 2 ceilings and 505 boxes (4385 vertices, 6668 triangles), every wall and box closed; wall share 0.870 when "occupied" means value 0, 0.966 when it means GN-Bench's "not passable" (value below 200); 0 `label_ins_id` misses in 38 episodes.

## Choices this code made that the lead should confirm

1. "Occupied pixel" for the wall share. The InteriorGS card defines 0 = occupied, 127 = unknown, 255 = free; GN-Bench treats anything below 200 as not passable. Outer walls sit on the map's grey (unknown) border, so the two readings differ: on `0001_839920` the share is 0.870 (fails the 0.9 bar) with 0 only, and 0.966 (passes) with "not passable". The runner's PASS/FAIL line uses value 0; both numbers are written.
2. The goal object's own box is left out of the mesh, so that the source (at its box centre) is not sealed inside a closed box. The runner also warns if the source point lies inside any other box (none for the dry-run episode).
3. Boxes from `labels.json` labelled wall, floor, ceiling, room, door or window are not added (walls, floors and ceilings come from `structure.json`; door and window boxes would close the openings). Other labels, including "partition" and "curtain", are added.
4. First arrival = first sample at or above 0.1 times the channel's peak. Onsets at 0.01 and 0.5 times the peak are also written.
5. Binaural channel 0 is taken as the left ear. The header does not say; a left-before-right FAIL could mean swapped channels.
6. Episode: `numpy.random.default_rng(seed)` picks one row of the scene's test_unseen rows in trajectory_id order. If the start is not in line of sight of the source, the first point along the episode's own path that is becomes the second listener placement. The left-ear test uses the first point (start, then path points) where both the listener and the point 2 m to its left are on free occupancy pixels with nothing between them, heading kept at the episode's start heading.
7. Ceilings are at the highest wall height of the scene; holes are cut as the bounding rectangle of their profile in the wall's plane (one "HOLE" opening in `0001_839920` has a 63-point profile); floors and ceilings are single sheets.
8. `sampleRate` is set explicitly to `check1.sample_rate_hz` (44100, equal to the library default) in both runs; every field not named in `check1.settings` stays at the library default and is logged.

## Things already seen in the local data (for the lead)

- The `label_ins_id` lookup finds every id in `0001_839920` and the label text matches, but the box found is usually far from the episode's `goal_world`: median 5.3 m over the 38 episodes, 9 within 3 m (we ran this; per-episode distances are in the lookup CSV). Many episodes with different targets also share the same `goal_world`. Placing the source at the box centre (the PLAN's rule) therefore puts it away from the GN-Matrix goal; for the dry-run episode the source is about 7.6 m from `goal_world`. This bears on unknown 3 and on the source placement rule.
- The library binary contains the string `${CMAKE_INSTALL_PREFIX}/share/libmysofa/default.sofa`, so the "default HRTF" may be a file path fixed at build time. If the binaural run fails with an HRTF error, that is a check 1 finding to report, not something to work around.
- `gnbench_copy.verify_against_gnbench` compares the copied functions with GN-Bench's own when `GN_Bench` is importable (after the check 3 install); before that, `run_info.json` records "available: False".
