# Experiment log: rung 1, released WNM-3D at inference time in GN-Bench

Step-by-step record of running this experiment (Jobs, errors, reruns, exact numbers). Append-only, newest at the bottom. The direction's `../../LOG.md` keeps short summaries that point here. Entry template: `_standards/04-documentation.md`.

## 2026-11-21 [agent, worker] (folder written, episode order frozen, nothing rented)
**Did:** wrote PLAN, config, setup, run, check and summary scripts and their unit tests from the approved rung 1 plan (direction LOG, 2026-11-21 experiment entry). Generated the frozen order on the Mac from the route-a check 3 CSV (sha256 270297ec...510e7): 5000 ids, seed 20261121, sha256 25366209f38a242dbcec94bfed26092a9937dbf5adabbcd149f4558caed72728; first three ids 0060_839918_78, 0401_840132_178, 0098_839977_50. The 39 possibly mirrored scenes recomputed from check 3 `scenes.jsonl` by the check 3 rule hold 233 test_unseen episodes (50 visible), as check 3 reported.
**Checks:** unit tests pass on the Mac; `check_settings.py` passes, also against the local pinned copies of WNM-3D 7fe2d5f and GN0 c927781.
**Open for the lead:** stall recovery on or off; K (timing batch); N (main run). No run can start while stall recovery is pending.
