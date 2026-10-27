# 2026-10-19 — Phase 1 pilot diagnostics on libero_goal (CPU)

**Purpose:** Run the first diagnostics on the features cached by `../2026-10-18-extract-libero-goal/`, and check the diagnostic pipeline end to end. Go-ahead from the user 2026-10-19.

**This is a pipeline pilot, not a test of H1.** With 2 encoders there is one pair, and there is no ground truth for combo gains yet (the proxy policy isn't built). Whatever comes out describes SigLIP 2-L vs C-RADIOv4 on one suite. It can't be ranked against anything.

**What runs:** `../../src/diagnostics.py` (commit in `job.yaml` `SRC_GIT_COMMIT`). For each encoder pair (A, B):
- **CKA:** linear CKA, biased and debiased (unbiased HSIC, Song et al. 2012; formula recalled, checked by synthetic tests below). Computed per camera for global/mean/4×4-grid features on 5,000 random frames.
- **Action probes:** ridge from features (both cams concatenated) to the 7-d action at the same step. Test R², averaged over the 7 dims. Run for A, B, A+B, and a proprio reference (joint + gripper states).
- **gain(B|A)** = R²(A+B) − R²(A).
- **resid(B|A):** probe R² on the part of B that a least-squares map from A can't predict. **pred(A→B):** R² of that map.
- **Split:** by demo within each task file: last 20% test, previous 20% val (ridge α chosen from 1e-2…1e5), rest train. Refit on train+val. No frame-level split, because neighboring frames are near-duplicates.

**Checks before running:**
- Synthetic tests in `../../src/test_diagnostics.py`, all as expected:
  - CKA(X,X) = 1, and CKA is invariant to rotation and scale.
  - Independent 300-d features with n = 1,500: biased CKA 0.17, debiased ≈ 0.
  - B that contains A's action-relevant part: resid(B|A) R² 0.02.
  - B carrying extra information: resid 0.51 (expected ~0.5).
- **Bug caught by those tests:** the first version residualized with a val-tuned ridge. Shrinkage left a scaled copy of the predictable part in the residual, which standardization then restored (resid R² 0.97 instead of ~0). Fixed by using a near-unregularized map (α = 1e-6, n ≫ d).
- Local end-to-end run on 12 demos of `turn_on_the_stove` (CPU, fp32 features): see LOG.

**Hardware:** CPU only (no GPU nodes), 4 CPU / 16 Gi, `python:3.11.14-slim`. Reads features from `michael-persistent`; downloads libero_goal again for actions.

**Commands:**
```
kubectl create configmap stacking-encoders-diag-src -n aria-world --from-file=../../src/diagnostics.py
kubectl apply -f job.yaml
kubectl logs -n aria-world job/stacking-encoders-diag-libero-goal > results/job.log
kubectl delete job -n aria-world stacking-encoders-diag-libero-goal
kubectl delete configmap -n aria-world stacking-encoders-diag-src
```

## Predictions (written before running; speculative)
- Proprio probe R² is low to moderate (0.2–0.5). LIBERO actions are delta end-effector commands, which depend on the goal more than on the current pose.
- Single-encoder probe R² is 0.3–0.6, with SigLIP 2-L and RADIO within 0.05 of each other. Mean features probe about as well as global ones.
- Debiased CKA, SigLIP 2-L vs RADIO, is 0.4–0.7 on mean features. RADIO is distilled in part from SigLIP2-g, so overlap is expected.
- gain(B|A) is small in both directions (< 0.05 R²). resid(B|A) is well below the single-encoder R².

## Results (2026-10-19; one pair, one suite: a pipeline pilot, not a finding)
**Runs:**
- Submit 0 (10:23Z): failed in ~1 s, twice (backoffLimit 1). Exit 127: `free` isn't in `python:slim`. No compute used.
- Run 1 (10:24–10:43Z, commit d6e3905): `results/run1-*`.
- Run 2 (10:46–11:53Z, commit a89ece6): `results/run2-*`. The same metrics, except the residual map is a float64 pseudo-inverse, and the run logs each probe's val R² per α plus residual-map statistics.
- All other numbers are identical between runs 1 and 2. Cleanup done: Job and ConfigMap deleted, namespace empty.

**Data:** verified: 63,728 timesteps, 10 tasks, split by demo: 38,464 train / 12,416 val / 12,848 test frames.

**Action-probe test R²** (mean over 7 action dims; verified, both runs):

| features | proprio | SigLIP 2-L | C-RADIOv4 | both | gain(+RADIO \| SigLIP) | gain(+SigLIP \| RADIO) |
|---|---|---|---|---|---|---|
| global | 0.279 | 0.609 | 0.670 | 0.678 | +0.069 | +0.008 |
| mean | – | 0.645 | 0.668 | 0.682 | +0.037 | +0.014 |

**pred(A→B)** (test R² of a linear map from one encoder to the other): SigLIP→RADIO 0.79 global / 0.84 mean; RADIO→SigLIP 0.72 / 0.76.

**Debiased CKA, SigLIP 2-L vs RADIO** (5,000 frames; biased ≈ debiased):

| | global | mean | grid |
|---|---|---|---|
| agentview | 0.72 | 0.82 | 0.88 |
| wrist | 0.84 | 0.88 | 0.94 |

Sanity: self = 1, shuffled ≈ 0.

**resid(B|A) failed:** −0.000 in all 4 directions in both runs.
- Run 2 rules out the solver. The map is full rank (or 2 dims dropped), max |w| 0.7–4.8, and the residual keeps 11–18% of B's variance.
- The residual probe's val R² is negative at every α (−0.62 at α = 0.01, rising to −0.0002 at α = 1e5). The probe overfits the residual and finds no signal that generalizes to new demos.
- speculative cause: frames within a demo are near-duplicates, so the effective sample size is ~300 train demos, not 38k frames. That's far too few for a 2–4.6k-dim probe on a weak residual signal. The synthetic tests used i.i.d. frames, so they couldn't catch this.
- Methods note (verified by algebra, recalled standard result): under OLS, probing the residual of B given A explains the same variance as gain(B|A). So resid adds little beyond gain anyway.

**Against the predictions:**
- Proprio 0.28 was inside its range.
- Single-encoder R² 0.61–0.67 was above the predicted 0.3–0.6, and the encoders differed by up to 0.061 (predicted < 0.05).
- CKA 0.72–0.94 was above the predicted 0.4–0.7.
- Gains were < 0.05 except +0.069 (RADIO given SigLIP, global).

**Not measured:** uncertainty. The gain differences (e.g. +0.008 vs +0.069) need demo-level bootstrap CIs before anyone reads them.

GPU-hours: 0 (CPU only; ~1.5 h of 4-CPU time across both runs).
