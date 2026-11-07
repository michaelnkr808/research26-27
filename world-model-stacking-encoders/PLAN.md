# Plan: Stacking Encoders

Living agenda. Only work that is running, next, or parked lives here. Results go in the experiment's `README.md`, history in `LOG.md`, conclusions in `README.md` "Key findings". When an item finishes, replace it with one line under "Recently done" pointing to the LOG entry, and prune that list now and then. Every session that changes an item updates this file in the same commit as its LOG entry.

Each experiment folder has its own `PLAN.md` (the design, written before running), `LOG.md` (the step-by-step record of running it, with exact numbers), and `README.md` (what came out). The direction `LOG.md` keeps short session summaries and work outside any experiment (decisions, literature, feedback, plan changes).

## Now
- **Multi-seed proxy replication, stage 4** (decision 003): `experiments/2026-10-27-proxy-calibration/`. Evaluation is the full LIBERO-Plus benchmark (10,030 episodes × eval seeds 41–43) per checkpoint, by user decision (experiment PLAN "Amendment 2026-11-06"; it also records the stop-5 override and the training GPU list for all 15 runs). Seed-noise check first (Next 1, rule now locked): `dinov3` seed 1 trained 2026-11-06 (experiment LOG (c)); seed 0 is training as `stacking-encoders-proxy-train-wave1c-dinov3-s0-a40` on an A40 into `runs/dinov3-s0-a40`. It's an A40-named twin that won a race against wave 1b, which was deleted before it started (experiment PLAN "Amendment 2026-11-06 (b)", LOG (f), (g)); its eval must use that run dir. Seed 1's full-benchmark rollout (user OK 2026-11-06), 28Gi / 33Gi (experiment LOG (d), (e)): the original Job was split 2026-11-07 (user OK; experiment LOG 2026-11-07 (a)) into `stacking-encoders-proxy-eval-dinov3-s1-main` (shards 0-5, regular GPU pool, parallelism 2) and `stacking-encoders-proxy-eval-dinov3-s1-a40` (shards 6-7, `nvidia.com/a40`, parallelism 1), both submitted 02:40 UTC and Pending. Once a pod of each has run about 30 min: GPU, memory and errors check (A40 separately), then raise parallelism within the 8-GPU gate (training counts 1 while seed 0 trains).

## Next
1. **Seed-noise check before the full round** (presentation feedback, point 2; user chose 2026-11-03). Train the full-length D runs for training seeds 0 and 1 first, roll both out on the same episodes, and compare. If two seeds of the same config differ by about as much as the harm threshold (half the paper's D+R − D drop), 3 seeds can't settle the question; rethink seeds or N before training the other 13 runs. Costs no extra runs (both are among the 15); it only orders them. Started 2026-11-06: the rule is locked in the experiment PLAN "Seed-noise rule (locked 2026-11-06)", seed 1 is trained, and seed 0 is queued.
2. **Cheap tests on perturbed frames** (presentation feedback, point 3; user chose 2026-11-03). The paper's drop is measured under LIBERO-Plus perturbations; our frozen-feature tests used clean frames only. Render matched clean/perturbed LIBERO-Plus frames (same scene and state, one perturbation category at a time), extract D, R, L, S, G features, and measure how far each encoder's features (and each D+B pair's) move under each perturbation. Check whether D+R stands out. No policy training. Caveat: only 4 partners to compare against the paper, so an ordering match is suggestive, not evidence. To do first: a short design (which frames, which shift measure, what counts as "stands out") for the user to sign off.
3. **Check how liu2026what runs the encoders in parallel and how it combines them** (mentor, 2026-11-06; LOG 2026-11-06 [agent, docs] mentor meeting). The user's understanding, not yet checked against the paper: features from different encoders are combined at matching locations.
4. **Look at what happens between encoder layers, not just at the final output** (mentor, 2026-11-06; same LOG entry).
5. **Find out which VLA the paper uses** (the mentor asked; same LOG entry).

## Parked (not chosen; from presentation feedback, LOG 2026-11-03 (k))
- D+D redundancy control config in the proxy round.
- C-RADIOv4 as input only vs target only, to locate the harm (only useful if the proxy shows the drop).
- Dimension-matched features in the frozen-feature tests.
- Email the paper's authors for training-seed results or code (no code released, verified 2026-10-17). User's call.
- Retrospective check of our diagnostics against the paper's 16 composition results (reviewer, 2026-10-28; suggestive only).

## Recently done
- 2026-10-27: target-side interference test; C-RADIOv4 is not the largest interferer, H-E fails (`experiments/2026-10-21-target-diagnostics/`, LOG 2026-10-27).
- 2026-10-21: feasibility tally, DINOv3 + 4 partners; neither CKA nor gain flags D+R (`experiments/2026-10-19-feasibility-tally/`).
