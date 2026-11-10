# Plan: one-way feature overlap (containment) score for a pair of frozen encoders

Written 2026-11-10 by an agent (planning worker) as a design for the user to sign off. Nothing here has been run on real features, and no cluster action has been taken. Claims are labeled verified (checked in this session against the file named), recalled, or speculative. Predictions are speculative by definition. The outcome rule is fixed now and is not to be edited after results (changes become dated amendments at the bottom, per `../../../_standards/04`). Paths are relative to `world-model-stacking-encoders/` unless they start with `src/`, `data/` etc. in this folder's parents.

## In plain words
The paper reports that stacking C-RADIOv4 on top of DINOv3 hurts, and says one reason may be that C-RADIOv4 was partly trained to copy DINOv3. If that is the reason, then C-RADIOv4's features should be easy to rebuild from DINOv3's features: it adds nothing new. So the idea is to measure that directly, with no robot and no labels. Take one encoder's features for every frame, fit a simple linear map to the other encoder's features on some demos, and see how well the map works on demos it has never seen. A high score means "most of what the second encoder says, the first one already says". Doing it in both directions gives a one-way overlap: B can be inside A without A being inside B (a wide encoder that learned from several teachers contains a lot but is not contained in any one of them).

One trap is avoided on purpose. The action-probe gains (how much adding B helps predict the robot's action) cannot be used for this: gain(D|B) is R2(D+B) minus R2(B), so the difference between the two directions is just R2(D) minus R2(B), the solo score again (`writing/2026-11-09-blind-baselines.md` line 113). This score never touches actions. It compares features with features.

With four pairs and one harmful label this is a tally, suggestive at most. It is groundwork for the Table 10 retrospective check, not a finding.

## Question
Does a feature-to-feature containment score, measured on frozen libero_goal features we already have, single out DINOv3 + C-RADIOv4 (the one pair the paper reports as harmful) among DINOv3 + {SigLIP 2, SAM 3, VGGT-Omega, C-RADIOv4}, and does it do so for a reason other than the measures we already have (debiased CKA, T1a, T1b, the solo score)?

## Why this is asked now
- `PLAN.md` Next 5 (line 25): the user un-parked a retrospective check against all of liu2026what Table 10, and the lead's candidate measure there is "a one-way overlap score, how much of B is already contained in A" (speculative, lead's idea).
- Direction `LOG.md` 2026-11-10 [agent, lead, planning] "ideas to strengthen the paper" (lines 1581-1592): the user decided "yes" to the one-way overlap score as a candidate measure, in the user's words "im not too sure what you mean on this but sure" (line 1583), so the plain-words section above is meant to be read at sign-off. The same entry says item 5 pulls part of decision 004's follow-up (the predictor) into the current paper (line 1589) and that a dated note on `decisions/004-paper-framing-case-study.md` goes by PR (still to open).
- Cheapness: the measure needs no new extraction. The four partners' features already exist (`data/README.md` lines 53-57).
- Status of the claim: with 4 pairs and one harmful label, `STANDARDS.md` line 31 applies word for word: "With 7 pairs, report the tally and don't claim significance." Here it is even fewer. Chance that a random score puts C-RADIOv4 at one named extreme of four partners is 1 in 4 (6 of 24 orderings; `writing/2026-11-09-blind-baselines.md` line 83, verified there). Nothing from this experiment goes into the README key findings without the user's sign-off at the verification gate (`../../../_standards/05`).

## Data (all existing, nothing extracted)
- libero_goal, 10 task files, 63,728 frames, 500 demos, 2 cameras. Tally split: per task file the last 20% of demos are test, the previous 20% validation, the rest train; 38,464 / 12,416 / 12,848 frames and 100 test demos (tally README line 6, verified; `src/diagnostics.py` lines 132-139 `demo_split`).
- Encoders and global feature size per camera (tally README lines 9-17): DINOv3 1,024, SigLIP 2 1,024, SAM 3 1,024, VGGT-Omega 2,048, C-RADIOv4 2,304. Both cameras concatenated: 2,048 / 2,048 / 2,048 / 4,096 / 4,608 (target-diagnostics README line 18). Variants are our choices and may differ from the paper's.
- Where: DINOv3, SAM 3 and VGGT-Omega on the CephFS PVC `stacking-encoders-features`; SigLIP 2 and C-RADIOv4 on `michael-persistent` (ReadWriteOnce) (`data/README.md` lines 53-57).

## The measure (fixed now)
Notation: D = DINOv3, B = a partner. For a source encoder X and target encoder Y, C(Y in X) is the held-out score of a linear map from X's features to Y's features, defined below. Three candidate pair scores, one of them primary:
- **S2 = C(B in D)** (PRIMARY). How much of the partner's feature content the base DINOv3 already provides: the "adding B gives nothing new" reading, which is how `PLAN.md` line 25 states the idea. Higher = more redundant partner = predicted worse. The paper's harmful pair is predicted to score highest.
- **S1 = C(D in B).** How much of DINOv3 is already inside the partner. The "teacher inside the student" reading of distillation. Secondary.
- **S3 = S2 minus S1.** Positive means the partner is more inside D than D is inside the partner. Secondary. This is the one-way part.

Only S2 can decide the outcome. S1 and S3 are reported and interpreted but carry no decision weight, so there is no choosing among three after the fact.

### Inputs
- **Global features, both cameras concatenated** (the primary feature type of the tally and of the control-partners design: `../2026-11-09-control-partners/PLAN.md` line 31). Reason: global is the pre-registered primary of both earlier experiments, it is what the probes saw, and the synthetic control partners exist for it. Mean features are a secondary repeat (same code, `--types mean`); grid features are not used.
- Each source feature is standardized per dimension with train-frame mean and standard deviation (`src/diagnostics.py` lines 241-243). This removes per-dimension scale, including the very large dimensions some transformers carry.

### The target (what gets predicted)
- Target Y's features are standardized per dimension on train frames, then projected onto Y's **top 64 principal components**, with the components computed on train frames only. Reason: a 2,304-d-per-camera target full of low-variance, mostly unpredictable dimensions (noise) would pull a plain average down, and the more such dimensions an encoder has the more it would be penalized for being wide. Fixing 64 for every encoder makes the targets comparable in size, drops the noise tail, and matches the rank 64 used as primary in the target-side test (`../2026-10-21-target-diagnostics/PLAN.md` line 51). Rank 64 is a choice, not a derived value; the share of Y's standardized variance that the 64 components cover on train frames ("coverage") is reported for every target so a reader can see how different the targets are.
- **Aggregation to one number:** the mean over the 64 components of the held-out R2 of each component (each component counts equally). A pooled variance-weighted R2 (1 minus total residual over total variance, over the same 64 components) is reported as a secondary because it is dominated by the first few components. Mean-of-components is primary because it asks "how many of B's main directions does D already give", and does not let one large component decide.
- R2 per component uses the test-set mean as the baseline, as `diagnostics.r2` does (lines 246-249).

### The map
- Ridge regression from the standardized source features (both cameras concatenated) to the 64 target components, with the same normal-equation form as `diagnostics.ridge_fit` (lines 252-256).
- **alpha** from the grid `diagnostics.ALPHAS` (0.01 to 1e5, factor 10), **chosen on the validation demos only** by mean validation R2, one alpha per direction and pair, then refit on train+val and scored once on the test demos (as `ridge_select`, lines 259-273). Test demos are never used to choose alpha, the components, the standardization or any threshold. If the chosen alpha sits at either grid edge it is recorded; the grid is not widened silently (target-diagnostics PLAN line 102).
- **Split by demo, never by frame.** The same demo split as the tally (above). Frames within a demo are highly correlated, so a frame-level split would leak (`STANDARDS.md` line 34). Components and standardization are fitted on train frames only.
- **Reduced-rank variant: not needed.** The target is already 64-dimensional, so the rank limit that the target-side test needed (a stand-in for the model's finite capacity) has no role here. The secondary all-dimensions version (below) uses plain ridge to the full target width.
- **Cross-split check.** The tally split is one split. Two more splits are run on the four real pairs, rotating which 20% of demos (in sorted demo order per task file) is test: fold 1 = the tally split (test block 5, validation block 4); fold 2 = test block 1, validation block 2; fold 3 = test block 3, validation block 4. This is the "more than one run" of `../../../_standards/07` (line 9) for a deterministic method whose only random part, apart from the controls, is which demos land in the test set. Folds 2 and 3 are not run for the controls.

### Confidence intervals
- Demo-level bootstrap over the test demos, **2,000 paired draws** (at least 1,000 are required by `STANDARDS.md` line 29), seed 0, with the fitted maps held fixed (the same scheme as `diagnostics.bootstrap_r2`, lines 303-316; the per-component mean R2 from per-demo sums is exactly what `r2_from_stats` computes, lines 296-300). Frames are never resampled.
- Every score for every pair in one run sees the same draws, so differences between partners (for the margin rule) are paired differences. Draws are saved (npz) so runs that cannot share a process (the controls) can still be differenced; same data, seed and number of test demos give the same draws (`../2026-11-09-control-partners/PLAN.md` line 34).
- Limit (as in the earlier experiments): these intervals do not include variation from refitting on different training demos; the cross-split check is the only handle on that.

### Secondary readings (reported, never decisive)
1. All-dimensions version: mean of per-dimension R2 over all target dimensions (no PCA), the quantity `diagnostics.py` already stores as `pred_A_to_B_r2` (lines 389-393). See the blindness section: this one already exists for the four pairs.
2. Pooled variance-weighted R2 over the 64 components.
3. Mean-type features instead of global.
4. **Trivial-structure baseline**, per target: held-out R2 for predicting the same 64 target components from only task identity (one-hot of the 10 task files), normalized time in the demo (frame index over demo length), and the proprioceptive state (joint + gripper, the proprio probe's input), by the same ridge. Reason: every encoder's top components may be dominated by which task and which phase of the motion the frame shows, which any encoder predicts; a high S2 could mean "B's main directions are easy-to-predict scene structure". Reported as headroom = S2 minus this baseline, per partner. It enters the outcome rule once (below).
5. **Width-matched sources** for S1 and S3 only (S2's source is always DINOv3, so its input is identical for all partners): source features replaced by their top 2,048 train principal components. Only VGGT-Omega (4,096) and C-RADIOv4 (4,608) are changed, since ridge is unchanged by rotating a full-width input.
6. **Context matrix**: C(Y in X) for all 20 ordered pairs of the five real encoders (global, tally split), to see whether C-RADIOv4 is simply "good at predicting and being predicted by everything" (richness) or specifically close to DINOv3.

## Confounds and how each is handled
| Confound | Handling |
|---|---|
| **Feature width** (a 4,608-d source predicts more easily than a 2,048-d one; a 4,608-d target has more to predict) | For the primary S2 the source is DINOv3 for every partner, so source width cannot differ. The target is fixed at 64 components for every partner. For S1 and S3 (varying source width), alpha is tuned on validation, the width-matched variant (secondary 5) is reported, and the noise control at C-RADIOv4's width must read near zero (gate). |
| **Per-dimension scale** (huge dimensions dominating) | Sources and targets are z-scored per dimension on train frames before anything else. |
| **Noise dimensions in the target** | Only the top 64 components are scored; the all-dimensions version is secondary and its disagreement with the primary is itself reported. The frame-shuffled and Gaussian-noise controls show what pure noise scores. |
| **Trivial shared structure** (task, motion phase, robot pose) | Trivial-structure baseline, secondary 4, and the headroom guard in the outcome rule. |
| **Richness** (a good encoder predicts anything) | Context matrix (secondary 6). The solo-score ordering is also compared (below). |
| **Leakage across frames in a demo** | Demo-level split and demo-level bootstrap; the synthetic test with demo-correlated features (below) fails if a frame-level split sneaks in. |
| **Redundancy is not usefulness** | S2 is high for a partner that repeats D and low for a partner that is mostly noise, since noise cannot be predicted from D. So S2 measures how much of B D already provides, not whether B is useful. This is by construction and is stated so that a low S2 is never read as "good partner". |

### Relation to measures we already have, and what would show the score is only one of them restated
- **Debiased CKA** (tally README lines 39-42): symmetric, one number per pair and camera, no map, no held-out frames, no interval. C-RADIOv4 has the highest CKA with DINOv3 in all 6 cells but by as little as 0.007 over SigLIP 2 in global agentview (tally README lines 77-78, verified). The four partners' CKA order (mean of two cameras, global) is R > L > S > G (blind-baselines lines 96-98 give 0.719, 0.684, 0.601, 0.240). S3 cannot be a function of CKA, because CKA is symmetric; a CI-separated nonzero S3 is information CKA cannot hold. S2 and S1 can be.
- **Solo action-probe score** (blind-baselines line 93: R 0.670, S 0.654, G 0.617, L 0.609): ordering R > S > G > L. If S2's order equals it, S2 may be a richness measure.
- **T1a and T1b** (target-diagnostics README lines 46-49): T1a is the R2 of a partner's future change from [D, B] now, which has B in the inputs; T1b is the gain of B as an input for D's future change. The new score is about the current frame, the source is D alone (S2) or B alone (S1), and no future or action is involved. Their order: T1b R > S > G > L (0.0103, 0.0073, 0.0014, 0.0004); T1a R > L > S > G (0.536, 0.495, 0.486, 0.433).
- **pred(A to B)** in `diagnostics.py` (lines 389-393): the secondary-1 quantity is this same number, already stored in the four tally JSONs. The new primary differs in the 64-component target, the CI, the cross-split check, the controls and the guards, not in the idea.
- **What would show S2 is merely a restatement (stated now):** (a) S2's order of the four partners equals CKA's order, or the solo order, or the T1a/T1b order, AND (b) S3's interval contains 0 for all four partners. In that case the report says "this score cannot be told apart from [that measure] with four partners" and does not call it a new measure. With only four partners a coincidence of orderings is likely (24 possible orders) and cannot be separated from a real agreement; the design admits this and relies on the three things CKA lacks: paired intervals on the margins, a nonzero asymmetry, and the controls. The real test of novelty is the 7-pair retrospective check.

## Control partners (known answers)
The synthetic partners already built for `../2026-11-09-control-partners/` are reused unchanged through `diagnostics.load` (`src/diagnostics.py` lines 55-125): `dup:dinov3`, `randproj:dinov3:4608`, `noise:4608`, `shuffle:dinov3`. They are run as partners of DINOv3 on the tally split (fold 1) only, synthetic seed 0, same bootstrap seed.

Gates: nothing about the real partners is read unless all pass (if one fails, debug and amend; do not interpret). Thresholds are written before any run and are not tuned.

| control | what it is | S2 = C(B in D) must read | S1 = C(D in B) must read |
|---|---|---|---|
| `dup:dinov3` | exact copy of D | at least 0.98 | at least 0.98 |
| `randproj:dinov3:4608` | fixed random linear mix of D, stretched to 4,608 dims | at least 0.95 (contained, though as wide as C-RADIOv4) | at least 0.95 (D is recoverable from the mix) |
| `noise:4608` | Gaussian noise at C-RADIOv4's width | within +/- 0.01 of 0 | within +/- 0.01 of 0 (a 4,608-d source of pure noise must not look informative: the width test) |
| `shuffle:dinov3` | D with frames permuted (real marginals, broken pairing) | within +/- 0.01 of 0 | within +/- 0.01 of 0 |

Also: split counts equal the tally's and 100 test demos; S3 within +/- 0.01 of 0 for each control. The numbers 0.98 and 0.95 are speculative expectations about how much shrinkage the ridge grid leaves in an exactly linear relation; they are fixed now as the bar. The controls test the ends of the scale only. There is no partner of known intermediate value, because a mixture of signal and noise does not have a clean known R2 once only the top components are scored; that is a stated limit.

## Predictions and outcome rule (fixed now)
### Evidence that already bears on the prediction (the prediction is not blind to these)
- CKA puts C-RADIOv4 highest with DINOv3 in all six cells (tally README line 77). Solo probe score puts C-RADIOv4 highest (blind-baselines line 93). gain(B|D) puts it highest (tally README lines 84-87). T1a and T1b put it highest (target-diagnostics README lines 46-49). The paper's own text says it distills DINOv3 (blind-baselines line 20). So "C-RADIOv4 is the extreme partner on a redundancy-style score" is the default expectation of everything we have, and a success here is unsurprising.
- **The tally result JSONs already contain `pred_A_to_B_r2` for every ordered D/partner pair, global and mean** (verified: key present in `../2026-10-19-feasibility-tally/results/tally-libero_goal-dinov3-*.json`; produced by `src/diagnostics.py` lines 389-393). It is the all-dimensions, no-interval version of secondary 1. I deliberately did not open these values for the other three partners.
- **Disclosure of an accidental sighting.** While checking that the key exists, a `grep` printed the four stored values from the D + C-RADIOv4 JSON (lines 289, 301, 309, 321: 0.805, 0.745, 0.833, 0.730). Assuming the keys are in the order D to R global, R to D global, D to R mean, R to D mean (not checked; speculative), DINOv3 predicts C-RADIOv4's dimensions a little better than the reverse in both feature types. The pilot README (`../2026-10-19-diagnostics-libero-goal/README.md` line 56, read) records the same direction for SigLIP 2 and C-RADIOv4 (SigLIP 2 to C-RADIOv4 0.79 global, C-RADIOv4 to SigLIP 2 0.72). Both lean toward "C-RADIOv4 is more inside the other encoder than the other is inside it", which is the S2 direction I chose as primary, and that choice was made from the wording of `PLAN.md` line 25 before the sighting, but a reader should not treat the primary as blind for the all-dimensions variant on this pair. Anyone who has already seen the other pairs' stored values should say so in an amendment.
- Recommended handling: nobody opens the stored `pred_A_to_B_r2` values for the other partners until the new run is complete; the implementation session compares them only afterwards, as the restatement check.

### Prediction (speculative, written before running)
On S2 (primary), C-RADIOv4 is the highest of the four partners, by a margin large enough for its paired interval to exclude 0 against the second highest. The reason is the paper's story: a partner trained to reproduce DINOv3's features should be largely rebuildable from DINOv3's. My confidence is moderate, about even odds, given the sighting above, which is as likely to mean "the all-dimensions version favors R, the 64-component version might not" as the opposite. I make no prediction for the other three partners' order or for the size of any score, except that the controls hit the gates. For S3 I expect a positive value for C-RADIOv4 (more of it inside D than D inside it), speculative and not decisive.

### Outcome rule (on S2, fold 1, global features, 2,000 paired draws; applied word for word)
Gates first (controls, above). Then, for the four real partners:
- **Flags D+R:** C-RADIOv4 has the highest S2; in every one of the 2,000 paired draws take the smallest of the three differences S2(R) minus S2(X), X over the other three partners; the 2.5th percentile of that smallest difference is above 0 (this accounts for the runner-up being picked from the same data; a plain 95% interval on the margin to the observed second place would be too optimistic); C-RADIOv4 is also highest in point value in folds 2 and 3; and C-RADIOv4 is also highest on the headroom score (S2 minus the trivial-structure baseline, fold 1, point value).
- **Does not flag:** for some other partner X, the paired difference S2(X) minus S2(R) has a 98.3% interval (95% corrected for the three comparisons) whose lower end is above 0 (the redundancy score puts a different partner clearly ahead of C-RADIOv4).
- **Unresolved:** anything else (C-RADIOv4 first only in point value, a margin interval that contains 0, or first in fold 1 but not in folds 2 or 3 or on headroom).

Chance: if the score were random, C-RADIOv4 would be the highest of four 1 time in 4; the interval and fold requirements make a chance "flags" rarer, by an amount not computed here. With four partners and one harmful label, even "flags" is a tally entry.

### What each outcome means for the Table 10 retrospective check (the lead's and user's call, not made here)
- **Flags:** S2 (and its extension to a stack, to be designed) stays the lead candidate measure for the retrospective check. This is the reason to ask the user to spend the three new extractions (V-JEPA 2.1, UniTok, HRP), and only after the restatement check and the novelty check in open question 7. No finding yet.
- **Does not flag:** the overlap-as-redundancy story, measured this way, does not single out the harmful pair, and the retrospective check should not be built around it; it would strengthen the README's current reading that the harm leaves no trace in clean frozen features, which stays the user's call.
- **Unresolved:** do not spend the three extractions on this score. Options for a follow-up are in open question 2 (different target rank, per-camera features); a follow-up is a new plan, not an edit of this one.
- In all cases: restatement result (a) and (b) above is reported next to the verdict.

## What code is needed (described, not written)
- A new script `src/feature_overlap.py` that imports `diagnostics` (loading including the synthetic partners, `demo_split`, `standardize`, `ALPHAS`, `ci`) the way `src/target_diagnostics.py` does, and does not change `diagnostics.py`.
  - Arguments: `--feats`, `--suite`, `--raw`, `--encoders` (reals plus optional synthetic specs, DINOv3 first), `--types`, `--fold` (1, 2 or 3), `--m` (default 64), `--n-boot` (default 2000), `--seed`, `--synth-seed`, `--save-draws`, `--out`.
  - A rotated demo split for folds 2 and 3, written in the new script, so the tally's `demo_split` stays untouched.
  - Per input encoder: standardize on train, one eigendecomposition of X'X for train and for train+val, reused for every alpha and every target, so cost is a projection per alpha (the same trick `target_diagnostics.py` uses; its docstring lines 19-21).
  - Per target encoder: train-only z-score and PCA, 64 components, coverage.
  - The trivial-structure baseline needs the task file name and demo length from `index` and the proprioceptive state that `diagnostics.load` already returns.
  - Output JSON: config, git commit, per ordered pair S values with alpha chosen and edge flags, the 64-component mean R2 and its interval, the pooled and all-dimensions versions, coverage, baselines, split counts, and the encoder meta for synthetic partners (already recorded by `diagnostics.load`).
- Tests in `src/test_feature_overlap.py` on synthetic data (all must pass before any Job; thresholds written in the test before it runs, in the style of `src/test_diagnostics.py`):
  1. Exact linear relation: Y = X W gives held-out S close to 1; independent noise gives close to 0 within a stated tolerance.
  2. Planted asymmetry: B is a function of A only, A has an extra independent block; C(B in A) close to 1 and C(A in B) clearly lower, and S3 has the right sign.
  3. Width: a wide pure-noise source predicting a fixed target reads close to 0, not positive (alpha chosen on validation does its job).
  4. Leakage: features that are a slow random walk inside each demo and unrelated across demos give held-out R2 close to 0 under the demo split; the test also asserts that no demo appears in two sets, and shows that a frame-level split on the same data would score high (so the test can fail).
  5. Train-only fitting: changing the test frames' values changes no standardization, component or alpha.
  6. Alpha uses validation only: scrambling test targets leaves the chosen alpha unchanged.
  7. Bootstrap pairing: two identical predictors give a difference interval of exactly [0, 0]; the mean-of-components R2 from per-demo sums equals a direct computation.
  8. Aggregation: when all 64 components have the same variance, mean-of-components equals the pooled R2.
  9. Fold split: every fold's test, validation and train demos are disjoint and cover all demos; fold 1 reproduces `diagnostics.demo_split`.
  10. Controls: with a synthetic partner spec the dup and noise cases give the gate values on a small fixture.
- Also, per `../../../_standards/07` ("sanity checks before scaling"), a local run on a handful of demos before the Job, timing it and measuring memory.

## Where it runs, and cost
- Features are on cluster volumes (above), so this is **one CPU-only Job** run by the implementation session, **only on the user's own go in that window** (ConfigMap, then Job; each cluster action is asked for). Reuse the control-partners Job (`../2026-11-09-control-partners/job.yaml`: 2 CPU / 14Gi requests, 2.4 / 16Gi limits, features staged to local disk, no sleep) as the template. `michael-persistent` is ReadWriteOnce, so check it is free before submitting (risk noted in the control-partners PLAN line 125). The code and plan land by PR first (`../../../_standards/05`: experiment code and outcome rules).
- Runs (about 14 invocations): the four real pairs in three folds, plus the four controls on fold 1, plus the five-encoder context matrix (the matrix reuses loaded features; if memory is tight it is a separate invocation).
- **Estimate (speculative), with its arithmetic.** Recorded per-pair script times of the tally diagnostics, which did more work than this experiment (CKA, three probes in two feature types, both directions of pred(A to B)), in seconds (`../2026-10-19-feasibility-tally/results/tally-libero_goal-dinov3-*.json`, field `seconds` at line 338, verified in this session): SigLIP 2 1,374; C-RADIOv4 1,572; SAM 3 1,750; VGGT-Omega 2,690. Their sum is 7,386 s = 2.05 h. Treating that as an upper bound for one fold of the four real pairs: three folds = 22,158 s = 6.15 h. The four controls match the pair sizes of the control-partners estimate (two 6,656-d at 1,572 s and two 4,096-d at 1,374 s per run, `../2026-11-09-control-partners/PLAN.md` lines 101 and 108-111): 5,892 s = 1.64 h. Staging and download about 0.1 h (same PLAN line 112). **Upper bound about 7.9 h wall, about 16 CPU-h at 2 CPU; times 1.5 for a slow node, about 12 h.** Reasons it should be well under that (speculative): no CKA, no action probes, one feature type, and one eigendecomposition per source reused for all alphas and targets instead of a fresh normal-equation solve per alpha (the old `ridge_select` re-forms X'X every time, which dominates its cost; target-diagnostics README line 65 records the target Job running about 4x slower than the Mac, which is why the first estimate there was too low). My guess is 1 to 3 h wall; the implementation session times a local subset run first and replaces this with a measured number.
- **Memory (speculative).** Global features only: the five real encoders total 14,848 columns over 63,728 frames, about 3.8 GB as float32, plus standardized copies; D + C-RADIOv4 peaked at 12.85 GiB in the target Job with k-shifted copies and fitted values (target-diagnostics README line 65) and ran within 11Gi in the tally diagnostics Job (control-partners PLAN line 115). This experiment's largest pair is the same 6,656 source dims, so 14Gi / 16Gi as in the control-partners Job is the starting size. Disk: raw libero_goal 5.9 GiB plus the staged features, as in that PLAN (line 126).
- No GPU. Nothing is stored in the repo except small result JSONs, draws and logs under `results/`.

## Not in scope
- No new encoder extraction: V-JEPA 2.1, UniTok and HRP stay stubs (`PLAN.md` line 25; their checkpoints are unknown).
- No training, no proxy policy, no perturbed frames, no GPU.
- No change to the README key findings, STANDARDS, PLAN or LOG of the direction, and no edit of `src/diagnostics.py` or `src/target_diagnostics.py`.
- No claim about any other Table 10 row. Only the four pairs for which we have features are scored.
- No stacks (three- and four-way compositions). A containment score for DINOv3 + UniTok plus a third encoder is part of the retrospective check's design, written only if this experiment says to continue.
- No action-probe quantity enters the score (the gain trap above).
- No per-camera, mean-type-as-primary, or grid-feature versions; no noise controls at other widths.
- No citation is made here. Debiased CKA is used throughout but its source (Kornblith et al. 2019, recalled) is not in `literature/references.bib`, which holds only liu2023libero, fei2025liberoplus, liu2026what, ranzinger2026cradiov4 and guo2026metric (verified); the same is noted in `STANDARDS.md` line 16. Predicting one network's features from another is a known way to compare representations (recalled, not checked, so no citation); a prior-art check is open question 7.

## Open questions for the user at sign-off
1. **Which direction is the headline score?** Options: S2 = C(B in D), "the partner is already inside DINOv3" (this plan; matches `PLAN.md` line 25 and gives every partner the same input), or S1 = C(D in B), "DINOv3 is already inside the partner" (the distillation reading). Recommendation: S2 primary, S1 and S3 reported, because the partner is the thing being added and S2 has no width confound. Changing it after results is not allowed.
2. **Target rank 64, equal-weight over components?** Options: 64 (this plan, matches the target-side test), 16 or 256, all dimensions. Recommendation: 64 as primary and the all-dimensions version as secondary; do not add ranks, so that the number of looks stays small.
3. **Three folds?** The cross-split check for the four real pairs costs about 4 extra hours in the upper-bound estimate. Recommendation: keep it; with a verdict resting on four numbers, the split is the cheapest honest noise check available.
4. **Context matrix (all 20 ordered pairs) included?** Recommendation: yes, it is what separates "C-RADIOv4 is close to DINOv3" from "C-RADIOv4 is close to everything".
5. **When does the Job run?** After perturbed-frames phase B and the control-partners Job in the implementation session's queue (direction `LOG.md` line 1535 gives that queue), on the user's go. Recommendation: after control-partners, reusing its template, so the same Job pre-checks apply.
6. **Stored `pred_A_to_B_r2` values.** Has the user or the lead already seen the stored values for pairs other than D + C-RADIOv4? Recommendation: say so in an amendment if yes; if no, nobody opens them before the new run is finished.
7. **Prior-art and citation check before this becomes a contribution.** The literature session should check whether regression-based or "model stitching"-style comparison of encoders is already established (recalled, unchecked), and Kornblith et al. 2019 should be added to `references.bib` first. Recommendation: yes, before any write-up; it does not block the run.

## Status
planned; awaiting the user's sign-off; no code, no Job.

## Amendments
<!-- ### YYYY-MM-DD: what changed, why, and whether any results had been seen yet -->
