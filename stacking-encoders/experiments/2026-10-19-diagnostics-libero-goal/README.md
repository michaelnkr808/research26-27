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
