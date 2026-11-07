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

## 2026-11-06 [agent, planning] feasibility read of LIBERO-Plus source @ 4976dc3
**Goal:** answer the Feasibility questions 0 to 8 in `PLAN.md` from the LIBERO-Plus source, and check the sub-type list. Reading only; nothing from the source was run, imported or installed.
**Did:** read the copy at `.../scratchpad/libero-plus-src` (`git log -1` = 4976dc3) with grep, sed, cat, jq. Wrote the answers into `PLAN.md` Feasibility (file:line for each), replaced "unverified" in the sub-type list with the check result, marked "Frames: which data" as decided, rewrote the open questions and the handoff step 1.
**Found (details and file:line in `PLAN.md`):**
- Camera and noise are runtime parameters read from the episode name (`libero/libero/envs/env_wrapper.py:207-221, 294-307`). Light and background are pre-built scene variants chosen by the first line of the bddl file (`libero_tabletop_manipulation.py:26230-26234`); the scene XML is in the assets zip, not in the repo, and no generator for it is in the repo.
- Camera moves the agentview only. Noise is applied to the agentview image only, after the render, in `step` and `reset`, and not on the `set_init_state` path. Field of view and roll are not in the code (README line 30 mentions field of view).
- The 1,000 light, table and tb bddl files of the 10 libero_goal base tasks are identical to the base bddl after line 1, and the init states come from the base task. Difficulty levels come from four models' accuracy (`index.html:437`), not from parameter strength; 121 of 279 libero_goal light episodes have no level.
- Dataset scripts write `states`, `robot_states`, `model_file` and bddl attributes (`scripts/create_dataset.py:261-267`). The Hugging Face files are not confirmed. The script drops the first 5 steps and the stored image k looks like the frame after action k, so it matches state k+1 (script reading; unrun).
- A set-state path (`env_wrapper.py:383-392`, used in `benchmark_scripts/render_single_task.py:70-74`) and a replay script (`scripts/create_dataset.py:145-189`) both exist.
- Risk for the clean control (6b): LIBERO-Plus's tabletop default agentview pose (`libero_tabletop_manipulation.py:306`) differs from the base-domain default (`bddl_base_domain.py:287-296`). MuJoCo is not pinned (robosuite 1.4.0 is, `requirements.txt:10`).
**Verified:** read in the source at the cited lines; bddl comparison, counts and ranges computed from files in the copy with cmp, grep and jq.
**Not done / open:** the assets zip (scene XML) and one Hugging Face HDF5 file were not read; whether a demo's state loads into a LIBERO-Plus env and renders the stored scene is untested.
**Next:** the user answers the open questions in `PLAN.md` (especially 2, 4, 14 to 17); then a small read-only step on one HDF5 file and the assets XML; smoke test after sign-off.
