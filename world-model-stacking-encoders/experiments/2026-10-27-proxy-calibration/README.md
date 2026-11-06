# Proxy-policy calibration round (decision 003)

**Results:** none yet. The design, locked outcome rule, and sizing rule are in `PLAN.md`; the agent's run steps are in `RUN.md`; the step-by-step record is in `LOG.md` (this folder; it indexes the earlier entries, which stay in `../../LOG.md`).

Status 2026-11-06: all feature caches done; pilot training and pilot rollouts done and read (sizing result in `PLAN.md`; the pilot checkpoints almost never succeed, so the sizing is not usable). The user chose the full LIBERO-Plus benchmark (all 10,030 episodes x eval seeds 41-43) for every checkpoint (`PLAN.md` amendment 2026-11-06). Wave 1 of the round, full-length `dinov3` for training seeds 0 and 1, was submitted 2026-11-06 07:47 UTC and is still Pending in the queue (`LOG.md` 2026-11-06 (a)). A seed-noise comparison rule is drafted in `PLAN.md`, waiting for the user's OK. Pilot results are for sizing only and never count toward the effect analysis.

When the round finishes, its results go here, and the step-by-step result notes that built up in `PLAN.md` move here too, so `PLAN.md` goes back to being the design.
