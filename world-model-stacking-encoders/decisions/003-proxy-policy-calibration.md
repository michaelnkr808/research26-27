# 003 — Go straight to the multi-seed proxy policy (calibration round)
- **Date:** 2026-10-27
- **Status:** accepted (user, 2026-10-27)
- **Context:** Both cheap diagnostics failed to flag the harmful DINOv3 + C-RADIOv4 (D+R) pair on frozen libero_goal features: input side (LOG 2026-10-21) and target side (LOG 2026-10-27 (j)). The gap to the paper has three parts that need different fixes (speculative): (1) the paper's ground truth is one training checkpoint per combo, so D+R's −5.4 may be training-seed noise; (2) the harm may only appear on perturbed LIBERO-Plus scenes; (3) it may come from training dynamics a frozen linear readout can't show. A multi-seed proxy policy that is scored on both clean LIBERO and LIBERO-Plus can test all three.
- **Options considered:**
  1. Ask the authors for per-category D+R numbers. Free, but tests (2) only. Still worth doing in parallel (needs user OK, since it leaves the machine).
  2. Probe perturbed frames. Cheap, but tests (2) only.
  3. Multi-seed proxy policy (decision 001, option 3). Tests (1), (2) and (3).
- **Decision & rationale:** Option 3, now. This changes decision 002's sequence: step 2 (choose the hypothesis) is skipped for now, because no cheap measure predicts the paper's ordering, and we first need to know whether that ordering is real. User choices (2026-10-27):
  - Configs: none, D, R, D+R, D+L (D+SigLIP 2, the paper's best pair, as a positive control). 3 training seeds each = 15 training runs.
  - Eval: clean LIBERO and LIBERO-Plus.
  - Proxy uses features as input *and* as future-feature targets, with per-source losses summed (paper Eq. 9).
  - Data: all four LIBERO suites from the start, matching the paper's training set.
- **Consequences:**
  - New feature extraction for libero_spatial, libero_object, libero_10 (D, R, L). The proxy also needs the clean LIBERO and LIBERO-Plus simulators, plus live encoder inference in the rollout loop.
  - Calibration outcome rule (pre-registered, see `experiments/2026-10-27-proxy-calibration/PLAN.md`): the proxy is trusted only if it reproduces D > none and D+L ≥ D. D+R vs D is the question being asked, not a calibration criterion.
  - Claims are about the proxy unless it agrees with the paper (as in 001). If the proxy agrees on D > none and D+L ≥ D but D+R doesn't hurt across seeds, that points to explanation (1) and is reportable as such, with the caveat that the proxy is not the WAM.
  - Every cluster object (extraction Jobs, training Jobs, eval Jobs) still needs a go-ahead before it's prepared or submitted.
- **Amendment (2026-10-28, user OK, after two rounds of outside review; before any training run):**
  - The consequence above ("D+R doesn't hurt across seeds ... points to explanation (1) and is reportable as such") is withdrawn. The proxy lacks Wan 2.2's pretrained video prior and is much smaller, so a proxy that doesn't reproduce the drop doesn't show the paper's result was training noise. This round is a **proxy replication**; its outcome decides whether a **higher-fidelity, multi-seed WAM replication** (Wan 2.2-based, the paper's recipe as closely as available code permits) is warranted.
  - Primary analysis: paired D+R − D on the same LIBERO-Plus episodes, one delta per training seed, all three shown individually. Three seeds guide the next step; no strong variance claim. Details in the PLAN outcome rule.
  - Overlap/interference as the cause of D+R stays a hypothesis.
  - Scope note: five configs test one anomaly, not a general predictor. A workshop paper on "cheap diagnostics predict which encoder additions help" needs more combinations and a measure of the diagnostics' predictive performance. A low-cost first step: a retrospective, exploratory check of our diagnostics against the paper's 16 composition results (each addition's gain vs its parent composition, Table 10 / App. E). Because configurations share encoders and each has one checkpoint, any rank correlation is suggestive only, not validation. Not started; needs V-JEPA 2.1, UniTok, HRP features and multi-encoder parents, which we don't have.
