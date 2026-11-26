# hear-lab shared code

Code that more than one experiment uses (`../STANDARDS.md`, "Conventions": shared code lives in the direction's `src/`). Scripts add this folder to `sys.path` (from their own location, or from the environment variable `HEAR_LAB_SRC` when the folder was copied elsewhere). Pure Python and numpy unless noted.

| file | what it is | written for | used by |
|---|---|---|---|
| `rlr_binding.py` | ctypes binding to `libRLRAudioPropagation.so` (RLR-Audio-Propagation `4fd446b`; Linux x64 only) | route A check 1 | route A checks 1 and the HRTF follow-up, Step 2 |
| `geometry.py` | InteriorGS `structure.json` + `labels.json` to a triangle mesh with one material category per face | route A check 1 | route A check 1, Step 2 |
| `poses.py` | GN-Bench heading or forward vector to the RLR listener quaternion | route A check 1 | route A check 1, Step 2 |
| `gnbench_copy.py` | GN-Bench's frame and pixel functions, copied verbatim with line numbers | route A check 1 | route A check 1 |
| `c3lib.py` | check 3 camera, box and visibility helpers (its `load_config` reads the route A config) | route A check 3 | route A check 3, Step 2 (box helpers) |

Moved here from `experiments/2026-11-20-route-a-checks/src/check1/` and `src/check3/` on 2026-11-26 with `git mv` (history kept). The behaviour of the route A code is unchanged; Step 2 added, without changing anything route A calls: `RLRContext.set_listener_pose`, `add_mesh_object(..., face_category=None)` (the library's default material on every face), and `poses.axes_from_forward` / `poses.listener_quat_from_forward`. The route A tests for these modules stay in route A (`src/check1/test_local.py`, `src/check3/test_c3lib.py`); the Step 2 tests are in `experiments/2026-11-26-step2-audio-check/src/`.
