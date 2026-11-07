# Experiment log: perturbed-frame test

Step-by-step record of running this experiment (Jobs, errors, reruns, exact numbers). Append-only, newest at the bottom. The direction's `../../LOG.md` keeps short summaries that point here. Entry template: `_standards/04-documentation.md`.

## 2026-11-06 [agent, planning] design drafted
**Goal:** write the draft design for the perturbed-frame test (direction `PLAN.md` Next 3) after the user chose the perturbation types and the measurements.
**Did:** wrote `PLAN.md` (docs and source reading only). The LIBERO-Plus source copy was not available in the session, so the Feasibility section is marked pending, with the exact questions to answer. Nothing was run.
**Found:** from the repo's own files, the libero_goal HDF5 contents recorded in `../../data/README.md` list no simulator states, and Fast-WAM's dataset lists an 8-d `observation.state` (end-effector pose and gripper) but no simulator state (proxy `PLAN.md`, "Step 1 findings"). So whether a stored demo moment can be re-rendered under a new camera, light or texture is not settled by the repo (see `PLAN.md`, Feasibility).
**Next:** the user reviews the draft and answers the open questions in `PLAN.md`. After sign-off, hand to the implementation-agent session. The source check runs first once the LIBERO-Plus copy is available.
**Open questions:** listed in `PLAN.md`.

## 2026-11-06 [agent, planning] review fixes to the draft
**Did:** corrected `PLAN.md` after review. Probe storage: `src/diagnostics.py` lines 176-197 keep predictions out of the JSON and discard weights, so no fitted probe exists and the analysis needs new code (refit with the stored alpha, save per-demo predictions). The HDF5 contents list in the data README is incomplete (the code reads `obs/joint_states` and `obs/gripper_states`), so it proves nothing about stored states. Costs redone with the recorded DINOv3 rate (about 66 frames/s fp32, `../2026-10-19-feasibility-tally/results/extract-try2-job.log` lines 186-195); diagnostics times are wall clock per pair. SigLIP 2 variant settled from `../2026-10-18-extract-libero-goal/job.yaml` line 60. Added: test-set use (pipeline check on train/val, tolerance to be set), re-render path questions, wrist camera, probe-capacity caveat, rule B defined, pair drift dropped, `rollout.py` pointer. The LIBERO-Plus source copy was still absent, so Feasibility stays pending.

## 2026-11-07 [agent, planning] user decisions recorded
**Did:** moved three user decisions into `PLAN.md` "Decided by the user": libero_goal HDF5 demos with the tally's split; rule P decides, B reported alongside, A not used; matched-dimension control added beside the full-size comparison (full-size primary). The control's method (PCA) and common size stay proposals. Status stays DRAFT (feasibility check pending, no sign-off). Remaining proposals: one draw per (demo, type), drift definition, per-camera reporting, clean re-render control with the 0.01 tolerance, sub-type sampling, predictions, fallbacks.
