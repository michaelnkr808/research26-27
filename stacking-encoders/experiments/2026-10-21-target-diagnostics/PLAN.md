# Handoff: target-side diagnostics (H-E) on libero_goal

Planning agent, 2026-10-21. For the implementation agent. The **binding** spec is the pre-registration in `LOG.md` 2026-10-21 (i) as amended by (j). This file turns it into tasks. If the two disagree, LOG wins; stop and ask the user.

## Read first
- `CLAUDE.md` → `README.md` and every file in `../../../_standards/` (session protocol: `05`; cluster rules: `08`).
- `STANDARDS.md`; `LOG.md` entries (e) through (j).
- `src/diagnostics.py` and `src/test_diagnostics.py` (code to reuse and imitate).
- `experiments/2026-10-19-feasibility-tally/README.md` and `diag-dlr-job.yaml` / `diag-ds-job.yaml` (Job templates).

## Why (one paragraph)
In liu2026what, encoder features are both a clean current input and a noised future target of the WAM (Sec. 3.3, App. B.2). Each source has its own flow-matching loss term at weight 1.0 (App. E.1 Eq. 9). No normalization is stated. The future-target role carries most of the gain: DINOv3 current-only 76.5, future-only 81.9, both 84.0 (Table 9). The tally tested only the input role, and nothing there flags D+R (78.6, below both D 84.0 and R 83.5 alone). H-E says a partner hurts by degrading the target: its future is hard to predict, or it competes with D's future for shared capacity under the loss weighting. This Job tests that cheaply on cached features.

## Scope
- **In:** a new script `src/target_diagnostics.py` and its synthetic tests `src/test_target_diagnostics.py`; local tests; one CPU Job manifest; after the user approves, running the Job and recording results.
- **Out:** any GPU work, new feature extraction, edits to `src/diagnostics.py` (import from it, don't change it), choosing the hypothesis, and editing README key findings (verification gate).

## Inputs (fixed by (e)/(i))
- libero_goal, 10 task files, 63,728 frames. Demo split from `diagnostics.demo_split` (unchanged). Both cams.
- Encoders: D = `dinov3`; partners `siglip2`, `c-radiov4`, `sam3`, `vggt-omega`.
- Feature type: **global** primary, **mean** secondary. Per encoder, the two cams are concatenated (D 2×1024, L 2×1024, S 2×1024, G 2×2048, R 2×2304).
- Reuse `diagnostics.load` (returns per-frame features, split, and `index` = (file, demo, length) per demo, in frame order). Frames are stored demo after demo, so a within-demo shift can be built from `index`.

## Definitions
- Horizons k ∈ {4, 16, 32}. For each k, keep frames t with t + k < demo length. **Never shift across demo boundaries.**
- Target Δ_E(t; k) = f_E(t + k) − f_E(t), raw scale (centered with train means only where a method needs it).
- Inputs X = [D_t, B_t] global, standardized on train (`diagnostics.standardize`). The **same X** goes into every T2 condition, so only the target changes.
- R² is the mean over target dims of per-dim test R² (`diagnostics.r2`).

## Measures
**T0, scale and loss share (no fitting; train frames):**
- Per encoder: mean per-channel variance of f, mean L2 norm of f per frame per cam, and mean per-channel variance of Δ for each k.
- Share of the loss on B's block in the concatenated target [Δ_D, Δ_B], under each scaling below: B's block weighted variance ÷ total.

**T1, target predictability (ridge, α on val per probe with `diagnostics.ALPHAS`, refit on train+val; same machinery as `ridge_select`):**
- T1a: X → Δ_B, test R².
- T1b: D_t alone → Δ_D vs X → Δ_D. The difference is the input-side gain of B for predicting D's future. Report both.

**T2, interference (primary measure), via reduced-rank ridge:**
1. Scale the target blocks with per-block weights w_k (multiply block k by sqrt(w_k)).
2. Fit full-rank ridge W = (XᵀX + α·n·I)⁻¹ Xᵀ Ỹ (same as `ridge_fit`).
3. Take the SVD of the fitted train(+val) values X W, and keep the top-r right singular vectors V_r.
4. W_r = W V_r V_rᵀ. Predict, undo the scaling, and score R² on **D's dims only**, on test.
5. **α selection:** on val, minimize the condition's own *weighted* val MSE of the rank-r model (the objective that condition trains on). Refit on train+val at that α.
- **Scalings** (w_k per block):
  - **(a′) primary:** per-source mean MSE, w_k = 1/d_k (Eq. 9 if L_FM averages channels).
  - **(b):** equalized, w_k = 1 / (total train variance of block k).
  - **(a″) secondary:** pooled, w_k = 1 (if L_FM sums channels).
- **Reference:** the same rank-r model with target Δ_D only. For a single block, the scaling can't change the result (test this).
- **interference(B; k, r, scaling) = R²_D(Δ_D only) − R²_D([Δ_D, Δ_B]).** Positive means adding B as a co-target hurts D's future prediction.
- **Rank r = 64 primary; 16 and 256 secondary.**
- **Primary number:** interference under (a′), r = 64, global, averaged over k ∈ {4, 16, 32}.

**CIs:**
- Demo-level bootstrap, 1,000 paired draws, seed 0, via `diagnostics.bootstrap_r2` (it resamples demos; give it the test Δ_D and the predictions).
- Average over k per draw. Same seed and same demo count give identical draws for every k. **Assert that the set of test demos is the same for all k**, i.e. every test demo has > 32 frames.
- 95% CI from `diagnostics.ci`.

## Synthetic tests (`src/test_target_diagnostics.py`, all must pass before any Job)
Follow `test_diagnostics.py` style: known-answer cases, asserts, a printed line per check.
1. **Shift:** a feature equal to the frame index within its demo gives Δ = k exactly. No frame pairs cross a demo boundary, and the frame count per k is Σ max(0, len − k).
2. **RRR at full rank** (r ≥ target dims) equals ridge predictions.
3. **Single-block invariance:** a Δ_D-only reference gives the same R² under all three scalings.
4. **Planted interference:** latent factors drive X. Δ_D depends on ~20 factors, and Δ_B on ~40 other factors at a large scale (e.g. ×10).
   - With r = 32: interference under (a″) and (a′) is clearly > 0, and under (b) it is smaller.
   - With r ≥ 60 (all factors fit): interference ≈ 0 under every scaling.
   - Pick the thresholds before running, and write them in the test.
5. **Bootstrap pairing:** the CI of an interference computed from two identical predictors is exactly [0, 0].

## Output
- JSON per pair, `results/target-libero_goal-dinov3-<B>.json`, with:
  - config: k, r, scalings, α grid, seed, n_boot, git commit, and frame counts per k per split;
  - T0 stats;
  - T1a/T1b R² + CI per k;
  - T2 per (feature type, k, r, scaling): reference R², with-B R², interference + CI, and the chosen α;
  - the k-averaged primary number + CI.
- `experiments/2026-10-21-target-diagnostics/README.md`, in the same shape as the tally README:
  - config, a primary table (B, T0 loss share under a′, T1a, T2 primary interference [CI], paper Δ), secondary tables, plain observations, caveats, resources.
  - Apply the **outcome rule from LOG (i)**:
    - R not the largest on T2 primary → H-E fails at this level.
    - R largest → weak support only (chance 1 in 4).
  - Don't argue beyond it.

## Cluster run (stop and ask the user before each cluster action)
- One CPU-only Job running the 4 pairs one after another. Base it on `diag-dlr-job.yaml`:
  - mounts `michael-persistent` read-only (L, R) and `stacking-encoders-features` (D, S, G);
  - us-west affinity; off GPU nodes; `python:3.11.14-slim`; numpy 2.2.6, h5py 3.16.0, huggingface-hub 0.36.2;
  - downloads libero_goal for the demo index;
  - code from a new ConfigMap `stacking-encoders-target-src` (both `diagnostics.py` and `target_diagnostics.py`, because the new one imports it);
  - `SRC_GIT_COMMIT` set; `backoffLimit: 1`; `activeDeadlineSeconds` set; **no sleep**.
- **Sizing:** measure peak RSS and wall time locally on a demo subset first. Size requests from that, with limits within 20% of requests (2 CPU / 11Gi worked for the tally). R global at 63,728 × 4,608 fp32 is ~1.2 GB; the k-shifted copies and fitted values multiply that.
- **Before submitting:** check CPU capacity via public Prometheus (memory note `nautilus-gpu-availability-check`, adapted for CPU nodes as in LOG (g)). Check that `michael-persistent` is free: it is RBD (ReadWriteOnce), so the Job won't mount it if another pod holds it.
- **After the Job:**
  - save the pod log to `results/`, then cut out the JSONs;
  - delete the Job and ConfigMap;
  - verify nothing of ours is left running in `aria-world`.

## Pitfalls
- Shift within demos only; split by demo, never by frame.
- Fit the standardization and the T0 variances on train only.
- Use `float64` for the SVD and the normal equations (R is 4,608-d). Feature dtypes differ (D fp32 on disk, others fp16), so cast on load.
- The α grid edge: if the chosen α sits at 0.01 or 1e5, record it. Don't widen the grid silently. A change is an amendment logged before rerunning.
- Don't change k, r, scalings, the primary measure or the outcome rule after seeing results. Any change is a dated LOG amendment that says it came after the results.

## Session end (protocol `05`)
- A `LOG.md` entry marked `[agent]`, with every number labelled verified, recalled or speculative, plus GPU/CPU-hours.
- README status line updated (not key findings).
- Commit, and leave a Next step.

## Estimated effort (speculative)
Code + tests ~2–3 h local; Job ~1–3 CPU-h total, to be measured by the local timing run.
