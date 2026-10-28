# 2026-10-21 — Target-side diagnostics (H-E) on libero_goal (no training)

**Purpose:** test the target role of each partner encoder. In liu2026what every encoder is both a current-frame input and a future-feature target of the WAM; the feasibility tally (`../2026-10-19-feasibility-tally/`) only tested the input role. Binding spec: LOG 2026-10-21 (i) as amended by (j); task list `PLAN.md`; run handoff `RUN.md`.

## In plain words (written for the user, 2026-10-27)
**The big idea.** The paper builds robot world models that see through several pretrained vision encoders at once. Adding a second encoder to DINOv3 usually helps a little, but adding C-RADIOv4 makes the robot much worse (−5.4 points). Training a model for every combination is expensive, so we want to spot a bad combination cheaply, from the encoders' features alone, before any training.

**How we score things.** We never run a robot. We compute cheap measures on frozen features from clean libero_goal training demos, then check whether their ranking of partners matches the ranking in the paper's Table 10. Those paper numbers are LIBERO-Plus success rates: the paper trains each combination once and tests it under perturbed conditions (camera, lighting, background, noise and so on). See `../../literature/notes/liu2026-what-visual-representations-wams.md`.

**Experiment 1, the input side (`../2026-10-19-feasibility-tally/`).** First guess: C-RADIOv4 hurts because it repeats what DINOv3 already says. We asked how much new information about the robot's actions each partner adds on top of DINOv3. C-RADIOv4 added the most of the four. So simple redundancy doesn't explain it.

**Experiment 2, the target side (this folder).** In these world models each encoder is also something the model must predict a moment ahead. Second guess: predicting C-RADIOv4's future is a burden that crowds out predicting DINOv3's future, like a student whose second exam eats into study time for the first. We built a capacity-limited linear stand-in and measured how much worse it got at DINOv3's future once each partner's future was added. Every partner cost a little, but VGGT-Ω cost the most (about twice C-RADIOv4), and VGGT-Ω helped in the paper. C-RADIOv4 and SigLIP 2 came out nearly tied, though they are the paper's worst and best partners. By the rule fixed in advance, this guess fails.

**Why, possibly (speculative).** Both tests look at clean, normal images, and there C-RADIOv4 looks like a fine partner (alone it is nearly as good as DINOv3 in the paper, 83.5 vs 84.0). So the damage probably isn't visible there. Candidates: (1) the paper trained each combination once, so DINOv3 + C-RADIOv4 may be one unlucky run; (2) something goes wrong during training that frozen features can't show; (3) the pair breaks mainly under LIBERO-Plus perturbations, which our clean-image probes never see. The paper reports only the LIBERO-Plus total for combinations, so the per-category breakdown that would test (3) isn't public.

## Config
- **Data:** libero_goal, 10 task files, 63,728 frames, 500 demos; same demo split as the tally (train/val/test 38,464 / 12,416 / 12,848 frames). Both cams concatenated per encoder.
- **Pairs:** base D (DINOv3) with B ∈ {L SigLIP 2, S SAM 3, G VGGT-Ω, R C-RADIOv4}. Features are the tally's caches. Dims (both cams): D 2,048; L 2,048; S 2,048; G 4,096; R 4,608.
- **Measures** (`../../src/target_diagnostics.py` at `52d1226`; outputs bit-identical to the pre-registered implementation at `b68ba54` on synthetic data, LOG 2026-10-27 (h)):
  - Target Δ_E(t; k) = f_E(t + k) − f_E(t), within a demo, k ∈ {4, 16, 32}. Inputs are [D, B] at t in every condition.
  - **T2 (primary):** reduced-rank ridge, r = 64. interference(B) = R²_D(target Δ_D only) − R²_D(target [Δ_D, Δ_B]); positive means B as a co-target hurts the prediction of D's future. Scalings: a′ (primary, w = 1/d per block), b (equalized block variance), a″ (pooled).
  - T0 (B's share of the initial loss), T1a (R² of Δ_B from [D, B]), T1b (gain of B as an input for Δ_D): reported, don't decide.
  - **Primary number:** interference under a′, r = 64, global features, mean over the 3 k. 95% CIs: demo-level bootstrap, 1,000 paired draws, seed 0 (fitted models held fixed).
- **Outcome rule (pre-registered, LOG 2026-10-21 (i), quoted):** "R not the largest on T2-raw → H-E fails at this level. ... R largest → weak support only. With n = 4, a random measure picks R 1 time in 4." Prediction also said R's interference shrinks under (b).
- **Run:** Job `stacking-encoders-target-diag` (`job.yaml`, try 2), submitted 2026-10-27 21:49:44Z at `567413d`, node-2-10.sdsc.optiputer.net, 2 CPU / 14Gi (limit 2.4 / 16Gi), features staged to local disk. Complete in 5 h 45 m, no restarts. Try 1 was OOM-killed (`results/try1-oom-20261027T185533Z.log`, LOG 2026-10-27 (f)).

## Results (verified, from `results/target-libero_goal-dinov3-*.json`; paper Δ = Table 10 mean − D alone 84.0, as verified in the tally README)

**Primary: T2 interference, global, r = 64, mean over k**

| B | interference a′ [95% CI] | under b | under a″ | B loss share (a′) | paper Δ vs D |
|---|---|---|---|---|---|
| G VGGT-Ω | **+0.0343** [+0.0332, +0.0354] | +0.0228 | +0.0462 | 0.717 | +0.40 |
| R C-RADIOv4 | +0.0151 [+0.0147, +0.0155] | +0.0251 | +0.0283 | 0.352 | −5.41 |
| L SigLIP 2 | +0.0141 [+0.0138, +0.0145] | +0.0288 | +0.0141 | 0.288 | +1.83 |
| S SAM 3 | +0.0032 [+0.0030, +0.0035] | +0.0207 | +0.0032 | 0.116 | +0.49 |

Reference R²_D (Δ_D only, r = 64, k-mean): 0.364–0.372 across pairs.

**Outcome: R is not the largest (G is; R is second, narrowly above L). Under the pre-registered rule, H-E fails at this level.** The secondary prediction also fails: R's interference grows under (b) (+0.0151 → +0.0251) instead of shrinking.

**Secondary** (global unless stated; k-mean)

| B | a′ r = 16 | a′ r = 256 | a′ by k (4 / 16 / 32) | mean-type a′ r = 64 | T1a R² | T1b gain |
|---|---|---|---|---|---|---|
| G | +0.0652 | +0.0187 | +0.0346 / +0.0362 / +0.0321 | +0.0777 | 0.433 | +0.0014 |
| R | +0.0145 | +0.0161 | +0.0162 / +0.0148 / +0.0143 | +0.0375 | 0.536 | +0.0103 |
| L | +0.0118 | +0.0175 | +0.0162 / +0.0134 / +0.0128 | +0.0595 | 0.495 | +0.0004 |
| S | +0.0035 | +0.0037 | +0.0021 / +0.0035 / +0.0041 | +0.0195 | 0.486 | +0.0073 |

α hit the grid edge only for `global|k=4|t1b_D` (α = 0.01) in every pair; recorded, not acted on (PLAN pitfalls).

## Observations (plain; n = 4, no significance claims across partners)
- Every partner interferes a little (all CIs above 0), so adding any co-target costs D some predictive accuracy at fixed rank. The sizes are small: 0.3 to 3.4 R² points against a reference of ~0.37.
- G stands out on the primary measure and carries most of the loss under a′ (share 0.72), but G helped in the paper (+0.40). R and L are nearly tied, yet they sit at opposite ends of the paper's ranking (−5.41 vs +1.83). The ordering doesn't track the paper's deltas.
- The bootstrap CIs are narrow because they resample test demos with the fitted models fixed; they don't include fitting variance or anything about the trained WAM.
- Dimension is a confound: the two widest partners (G 4,096, R 4,608) are first and second under a′. Not tested further.

## Caveats
- A linear, reduced-rank stand-in for the WAM's flow-matching objective, on clean libero_goal frames. It tests whether the target-side mechanism shows up in the features, not what happened inside the paper's model.
- Paper deltas are single trained checkpoints (Table 10 seeds are evaluation seeds).
- Encoder variants are our choices and are not confirmed to match the paper.

## Resources
- Cluster: ~11.5 CPU-h (2 CPU × 5 h 45 m), no GPU. Peak RSS 12.85 GiB (during D+R), estimate was 12.5–13 GiB; CPU 1.7–1.9 cores average. Estimated 1–3 CPU-h in PLAN, ~3 h wall locally (LOG (k)); actual wall 5 h 45 m (node ~4× slower than the Mac for this linear algebra).
- Try 1: ~0.4 CPU-h, OOM-killed.
