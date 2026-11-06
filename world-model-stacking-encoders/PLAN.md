# Plan: Stacking Encoders

Living agenda. Only work that is running, next, or parked lives here. Results go in the experiment's `README.md`, history in `LOG.md`, conclusions in `README.md` "Key findings". When an item finishes, replace it with one line under "Recently done" pointing to the LOG entry, and prune that list now and then. Every session that changes an item updates this file in the same commit as its LOG entry.

Each experiment folder has its own `PLAN.md` (the design, written before running), `LOG.md` (the step-by-step record of running it, with exact numbers), and `README.md` (what came out). The direction `LOG.md` keeps short session summaries and work outside any experiment (decisions, literature, feedback, plan changes).

## Now
- **Multi-seed proxy replication** (decision 003): `experiments/2026-10-27-proxy-calibration/`. Pilot rollout 2's setup bug is fixed: the offline package install now uses apt (b0b60f3), and the rebuilt env bundle passed a fresh-container restore test (env-build4, 2026-11-04; experiment LOG). Pilot rollout 2 (src 2bab64f) submitted 2026-11-04 22:08 UTC; Job Complete 2/2 at 2026-11-05 04:11 UTC (340 records, 0 errors per checkpoint). Pod 1's GPU fell below the 40% floor mid-run (RUN.md stop 10); the read-only diagnosis points at CPU-heavy scenes bunched in the job order, not the node (speculative; experiment LOG 2026-11-05 (c)); the fix waits on per-episode timings. Reading the JSONLs: `jobs/rendered/job-pilot-read.yaml` (CPU-only, user OK 2026-11-05) is committed on branch worktree-agent-a1345b33d3e874b25, not submitted: the kubectl login refresh failed in the worker's sandbox. After the user refreshes the login: submit, decode, checksums, sizing, timings, cap. Then sizing for the full round (15 training runs: none, D, R, D+R, D+L × 3 seeds).

## Next
1. **Seed-noise check before the full round** (presentation feedback, point 2; user chose 2026-11-03). Train the full-length D runs for training seeds 0 and 1 first, roll both out on the same episodes, and compare. If two seeds of the same config differ by about as much as the harm threshold (half the paper's D+R − D drop), 3 seeds can't settle the question; rethink seeds or N before training the other 13 runs. Costs no extra runs (both are among the 15); it only orders them. To do first: write the comparison rule down before the runs finish. Needs the user's OK before submitting.
2. **Cheap tests on perturbed frames** (presentation feedback, point 3; user chose 2026-11-03). The paper's drop is measured under LIBERO-Plus perturbations; our frozen-feature tests used clean frames only. Render matched clean/perturbed LIBERO-Plus frames (same scene and state, one perturbation category at a time), extract D, R, L, S, G features, and measure how far each encoder's features (and each D+B pair's) move under each perturbation. Check whether D+R stands out. No policy training. Caveat: only 4 partners to compare against the paper, so an ordering match is suggestive, not evidence. To do first: a short design (which frames, which shift measure, what counts as "stands out") for the user to sign off.

## Parked (not chosen; from presentation feedback, LOG 2026-11-03 (k))
- D+D redundancy control config in the proxy round.
- C-RADIOv4 as input only vs target only, to locate the harm (only useful if the proxy shows the drop).
- Dimension-matched features in the frozen-feature tests.
- Email the paper's authors for training-seed results or code (no code released, verified 2026-10-17). User's call.
- Retrospective check of our diagnostics against the paper's 16 composition results (reviewer, 2026-10-28; suggestive only).

## Recently done
- 2026-10-27: target-side interference test; C-RADIOv4 is not the largest interferer, H-E fails (`experiments/2026-10-21-target-diagnostics/`, LOG 2026-10-27).
- 2026-10-21: feasibility tally, DINOv3 + 4 partners; neither CKA nor gain flags D+R (`experiments/2026-10-19-feasibility-tally/`).
