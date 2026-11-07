# Experiment log: perturbed-frame test

Step-by-step record of running this experiment (Jobs, errors, reruns, exact numbers). Append-only, newest at the bottom. The direction's `../../LOG.md` keeps short summaries that point here. Entry template: `_standards/04-documentation.md`.

## 2026-11-06 [agent, planning] design drafted
**Goal:** write the draft design for the perturbed-frame test (direction `PLAN.md` Next 3) after the user chose the perturbation types and the measurements.
**Did:** wrote `PLAN.md` (docs and source reading only). The LIBERO-Plus source copy was not available in the session, so the Feasibility section is marked pending, with the exact questions to answer. Nothing was run.
**Found:** from the repo's own files, the libero_goal HDF5 contents recorded in `../../data/README.md` list no simulator states, and Fast-WAM's dataset lists an 8-d `observation.state` (end-effector pose and gripper) but no simulator state (proxy `PLAN.md`, "Step 1 findings"). So whether a stored demo moment can be re-rendered under a new camera, light or texture is not settled by the repo (see `PLAN.md`, Feasibility).
**Next:** the user reviews the draft and answers the open questions in `PLAN.md`. After sign-off, hand to the implementation-agent session. The source check runs first once the LIBERO-Plus copy is available.
**Open questions:** listed in `PLAN.md`.
