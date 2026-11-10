# Plan: one-way feature overlap (containment) score for a pair of frozen encoders

Written 2026-11-10 by an agent (planning worker) as a design for the user to sign off; revised the same day after the round-1 review on PR #14. Nothing here has been run on real features, and no cluster action has been taken. Claims are labeled verified (checked against the file named), recalled, or speculative. Predictions are speculative by definition. The outcome rule is fixed at sign-off and is not edited after results (changes become dated amendments at the bottom, per `../../../_standards/04`). Paths are relative to `world-model-stacking-encoders/`.

## In plain words
Take one encoder's features for every frame, fit a simple linear map to another encoder's features on some demos, and see how well it predicts on demos it has never seen. A high score means "most of what the second encoder says, the first already says". Doing it in both directions gives a one-way overlap: encoder X can sit inside Y without Y sitting inside X.

Two different stories lead to two different directions, and the design keeps them apart:
- **S1 = C(D in B): DINOv3 is rebuilt from the partner.** The paper says C-RADIOv4 was distilled from DINOv3, SigLIP 2 and SAM 3 (`writing/2026-11-09-blind-baselines.md` line 20). A student trained to reproduce a teacher should contain it. So the distillation story predicts S1 high for C-RADIOv4, and it does not predict that C-RADIOv4 is itself inside DINOv3 (it also learned from two other teachers), so S3 (below) is not predicted positive.
- **S2 = C(B in D): the partner is rebuilt from DINOv3.** This tests a different story, "the added encoder brings little that DINOv3 lacks". It sits in tension with the tally, where C-RADIOv4 has the largest gain(B|D) of the four partners, though its interval overlaps SAM 3's (tally README lines 84-86, verified).

VGGT-Omega adds a second test of the distillation reading: its encoder is initialized from DINOv3 (`writing/2026-11-09-blind-baselines.md` line 18) and it was reported roughly harmless (+0.40, tally README line 42), so a high S1 for VGGT-Omega would cut against "S1 tracks harm".

Which one is the headline score is the user's decision at sign-off (open question 1). Everything below is written for a primary score P that is fixed as S1 or S2 before any run.

**Decision, 2026-11-10 (the user, in the lead's chat, before any run): P = S2, "rebuild the partner from DINOv3", C(B in D).** S1 and S3 are reported and carry no decision weight. The S1 reasoning below is kept for the record.

One trap is avoided on purpose. The action-probe gains cannot serve as the score: gain(D|B) is R2(D+B) minus R2(B), so the difference between the two directions is just R2(D) minus R2(B), the solo score again (`writing/2026-11-09-blind-baselines.md` line 113). This score never touches actions.

With four pairs and one harmful label this is a tally, suggestive at most (`STANDARDS.md` line 31: "With 7 pairs, report the tally and don't claim significance"). It is groundwork for the Table 10 retrospective check (direction `PLAN.md` Next 5), not a finding. Nothing goes into the README key findings without the user's sign-off at the verification gate (`../../../_standards/05`).

## Question
Does a feature-to-feature containment score, measured on frozen libero_goal features we already have, single out DINOv3 + C-RADIOv4 (the one pair the paper reports as harmful) among DINOv3 + {SigLIP 2, SAM 3, VGGT-Omega, C-RADIOv4}, and is it more than a restatement of a measure we already have?

## Why now
Direction `PLAN.md` Next 5 names a one-way overlap score as a candidate measure for the retrospective check; the user agreed on 2026-11-10 ("im not too sure what you mean on this but sure", direction `LOG.md` entry 2026-11-10 [agent, lead, planning] "ideas to strengthen the paper", line 1583), which is why the plain-words section is written for sign-off. That entry also notes that this pulls part of decision 004's follow-up into the current paper; a dated note on `decisions/004-paper-framing-case-study.md` is still to go by PR. No new extraction is needed (`data/README.md` lines 53-57). Chance that a random score puts C-RADIOv4 at one named extreme of four partners is 1 in 4 (`writing/2026-11-09-blind-baselines.md` line 83, verified there).

## Data (existing, nothing extracted)
- libero_goal, 10 task files, 63,728 frames, 500 demos, 2 cameras. Split per task file: last 20% of demos test, previous 20% validation, rest train; 38,464 / 12,416 / 12,848 frames, 100 test demos (tally README line 6; `src/diagnostics.py` lines 132-139).
- Global feature size per camera, then both cameras concatenated (tally README lines 9-17, target-diagnostics README line 18): DINOv3 1,024 / 2,048; SigLIP 2 1,024 / 2,048; SAM 3 1,024 / 2,048; VGGT-Omega 2,048 / 4,096; C-RADIOv4 2,304 / 4,608. Variants are our choices.
- Where: DINOv3, SAM 3, VGGT-Omega on CephFS PVC `stacking-encoders-features`; SigLIP 2 and C-RADIOv4 on `michael-persistent` (ReadWriteOnce) (`data/README.md` lines 53-57).

## The measure (fixed at sign-off)
C(Y in X) is the held-out score of a linear map from source X's features to target Y's features. Three candidate pair scores, D = DINOv3 and B = partner:
- **S1 = C(D in B)**, **S2 = C(B in D)**, **S3 = S2 minus S1** (positive: the partner is more inside D than D is inside it).
- **P, the primary score, is S1 or S2, chosen by the user before any run.** S3 and the unchosen score are reported and carry no decision weight. Higher P means more overlap, which is the predicted-worse end.
- **Fixed 2026-11-10 by the user: P = S2, before any run.** The consequences are listed under "Consequences of P = S2" in the outcome-rule section.

**Inputs.** Global features (the primary type of the tally and control-partners designs), both cameras concatenated, so each encoder is one 2,048-to-4,608-column matrix. Each source column is standardized with train-frame mean and standard deviation (`diagnostics.standardize`, lines 241-243), which removes per-dimension scale. Mean features are a secondary repeat: same code, fold 1, the four real pairs only.

**Target.** The target encoder's columns are standardized on train frames, then projected on the **top 64 principal components of the concatenated two-camera matrix**, components from train frames only. Reason: a very wide target has many low-variance, mostly unpredictable columns; scoring only 64 components makes targets the same size, drops the noise tail, and matches the rank-64 primary of the target-side test (`../2026-10-21-target-diagnostics/PLAN.md` line 51). 64 is a choice. The share of the target's variance the 64 components cover on train frames ("coverage") is reported for every target.

**Aggregation.** Mean over the 64 components of the held-out R2 of each component (each counts equally). Pooled variance-weighted R2 over the same components is a secondary. R2 uses the test-set mean as baseline, as `diagnostics.r2` does (lines 246-249).

**Map.** Ridge from source to the 64 components, same normal-equation form as `diagnostics.ridge_fit` (lines 252-256), float64 linear algebra, features held as float32. **alpha** from `diagnostics.ALPHAS`, **chosen on validation demos only** by mean validation R2, one alpha per direction and pair, then refit on train+val and scored on test (as `ridge_select`, lines 259-273). Alpha at a grid edge is recorded, never widened silently. Split by demo, never by frame (`STANDARDS.md` line 34). Reduced-rank is not needed: the target is already 64-dimensional, so the capacity limit that the target-side test modeled has no role.

**Cross-split check.** Rotate which 20% of demos (sorted order, per task file) is test, on the four real pairs: fold 1 = tally split (test block 5, validation block 4); fold 2 = test block 1, validation block 2; fold 3 = test block 3, validation block 4. This is the multi-run check of `../../../_standards/07` line 9 for a deterministic method. Folds 2 and 3 are not run for controls.

**Intervals.** Demo-level bootstrap over test demos, 2,000 paired draws (at least 1,000 required, `STANDARDS.md` line 29), seed 0, fitted maps held fixed (as `diagnostics.bootstrap_r2`, lines 303-316). All scores in one run share draws, so differences are paired; draws are saved so separate runs can be differenced. These intervals omit refit variance; the folds are the only handle on it.

**Secondary readings (reported, never decisive).**
1. All-dimensions mean of per-dimension R2 (no PCA; alpha chosen on validation by mean validation R2 over all dimensions). This is the quantity `diagnostics.py` already stores as `pred_A_to_B_r2` (lines 389-393).
2. Pooled variance-weighted R2.
3. Mean-type features (fold 1).
4. **Trivial-structure baseline** per target: held-out R2 of the same 64 components from task one-hot (10), frame index over demo length, and proprio (joint + gripper states, the proprio probe's input), each standardized on train, same ridge and alpha rule. Headroom = score minus this baseline.
5. **Context matrix (required, used in the rule):** C(Y in X) for all 20 ordered pairs of the five real encoders, global, fold 1.

## Confounds
| Confound | Handling |
|---|---|
| Source width (a 4,608-d source predicts more easily). | S2's source is DINOv3 for every partner. For S1 the sources differ (C-RADIOv4 is widest): alpha on validation, the noise control at 4,608 columns must read near zero (gate), and the width-matched guard in the rule (when P = S1). |
| Per-dimension scale; noise dimensions | z-scoring; only 64 components scored; all-dimensions version reported beside it. |
| Structure any encoder predicts (task, layout, phase) and encoder richness | Trivial-structure baseline (headroom must also favor C-RADIOv4 when P = S2) and the context matrix (residual specificity score in the rule). The linear baseline is weak, which is why the matrix carries the guard. |
| Leakage across frames of a demo | Demo-level split and bootstrap; a synthetic leakage test (below). |
| Overlap is not usefulness | Noise cannot be predicted from anything, so a mostly-noise partner scores low on S2. Neither score says a partner is useful. |

**Relation to measures we have, and when the score counts as a restatement.** Debiased CKA is symmetric, has no held-out frames and no interval; C-RADIOv4 is highest with DINOv3 in all 6 cells, by as little as 0.007 in global agentview (tally README lines 77-78); its four-partner order (mean of cameras, global) is R > L > S > G (`writing/2026-11-09-blind-baselines.md` lines 96-98). S3 cannot be a function of CKA. Other orders to compare: solo action-probe score R > S > G > L (blind-baselines line 93); T1a R > L > S > G and T1b R > S > G > L (target-diagnostics README lines 46-49; T1a has B among the inputs and predicts a future change, T1b is the input-side gain for D's future); and the gain(B|D) order R > S > L > G (global; blind-baselines line 94); for P = S1 also the source width order (R, then G, then the three 2,048-column encoders) and, as an after-the-run comparator only, the stored partner-to-DINOv3 all-dimensions values (not quoted here, not to be opened before the run is finished); and the stored all-dimensions DINOv3-to-partner order, which exists for all four partners in direction `LOG.md` (entry 2026-10-21 (i), about line 243) and which this plan does not quote. **If the order of the four partners on P equals the order of any of these, the report says "cannot be told apart from [that measure] with four partners"** and does not call P a new measure. With 24 possible orders a match is likely, and the design cannot separate coincidence from agreement; what P can add is paired intervals on margins, an S3 that CKA cannot hold, the controls, and the 7-pair retrospective check.

## Control partners (known answers)
The synthetic partners of `../2026-11-09-control-partners/` are reused unchanged through `diagnostics.load` (`src/diagnostics.py` lines 55-125 for the specs, 142-186 for loading): `dup:dinov3`, `randproj:dinov3:4608`, `noise:4608`, `shuffle:dinov3`, as partners of DINOv3, fold 1, synthetic seed 0.

| control | S1 and S2 must each read |
|---|---|
| `dup:dinov3` | at least 0.98 |
| `randproj:dinov3:4608` | at least 0.95 |
| `noise:4608` | between -0.03 and +0.01 |
| `shuffle:dinov3` | between -0.03 and +0.01 |

Also: S3 between -0.02 and +0.02 for dup and between -0.05 and +0.05 for randproj and between -0.04 and +0.04 for noise and shuffle (randproj's range is wider than the difference of its two ranges because both of its scores sit near 1 and the limit is a guess); split counts equal the tally's with 100 test demos. The thresholds are speculative guesses about ridge shrinkage and a possible mean shift of held-out R2 below zero, fixed now. **A gate failure permits code bug fixes only (each logged); thresholds are never changed.** **The controls are run and read before the real-pair outputs are opened.** There is no control of known intermediate value, because a signal-plus-noise mixture has no clean R2 once only top components are scored; the controls test the two ends of the scale only.

## Predictions and outcome rule (fixed at sign-off)

### What already bears on the prediction (not blind to these)
- C-RADIOv4 is highest on CKA (all 6 cells), solo probe score, gain(B|D), T1a and T1b (sources above). Everything we have favors it as the extreme partner on a similarity or redundancy-type score, so a "flags" result would be unsurprising.
- **The all-dimensions DINOv3-to-partner value for all four partners is printed in direction `LOG.md` (entry 2026-10-21 (i), about line 243) and is mentioned in `reviews/2026-11-08-finished-experiments.md`. This design is therefore NOT blind to the crude ordering of the S2 direction.** The tally JSONs also store `pred_A_to_B_r2` in both directions, global and mean, for each pair (key order verified from code: `src/diagnostics.py` lines 381-393 and 417, tally `diag-dlr-job.yaml` line 57; D to R global, R to D global, D to R mean, R to D mean).
- The writer saw four stored values for the DINOv3 + C-RADIOv4 pair by accident, and records the direction only: DINOv3 predicts C-RADIOv4 somewhat better than the reverse, in both feature types. The pilot README (`../2026-10-19-diagnostics-libero-goal/README.md` line 56) shows the same direction for SigLIP 2 and C-RADIOv4.
- **As far as the record shows, still unseen:** the 64-component score in either direction, the partner-to-DINOv3 direction for the other three partners, folds 2 and 3, the controls, and the context matrix.

### What each pattern for C-RADIOv4 would mean (written before the run; readings, not exhaustive)
| S1 (D in R) | S2 (R in D) | S3 | would mean |
|---|---|---|---|
| highest of four | not highest | at most 0 | The distillation story as stated: DINOv3 is largely rebuildable from C-RADIOv4, which also holds more than DINOv3 has. |
| not highest | highest of four | any | The added encoder is mostly rebuildable from DINOv3. Not the distillation test; in tension with its largest gain(B|D). |
| highest | highest | near 0 | Strong mutual overlap, near-duplicate content in this feature space. |
| not highest | not highest | any | This measure does not see an overlap story for the pair. |
| any | any | any, but C-RADIOv4 is not highest on the residual specificity score | Richness, not specificity: the flag fails and the outcome is "unresolved" (guard below). |

Prediction (speculative): if the user fixes P = S1, C-RADIOv4 is highest, at about even odds, and VGGT-Omega (DINOv3-initialized, roughly harmless) is not at or above it; a high S1 for VGGT-Omega would cut against "S1 tracks harm". If P = S2, I have no strong prior and note the tension above. I make no prediction on other partners or on score sizes beyond the control gates, and no prediction that S3 is positive.

2026-11-10 (after the decision, no new prediction): the user fixed P = S2, so the "if P = S2, I have no strong prior and note the tension above" statement is the one that applies; the P = S1 statement above stays as written before the decision.

### Outcome rule, written once for the primary score P (fold 1 = tally split, global, 2,000 paired draws)
Controls are read first. Four outcomes:
- **Invalid, no verdict:** any control gate fails.
- **Flags D+R:** all of: C-RADIOv4 has the highest P; in each paired draw take the smallest of the three differences P(R) minus P(X) over the other partners, and its 2.5th percentile is above 0 (this accounts for the runner-up being picked from the same data); C-RADIOv4 is also highest in point value in folds 2 and 3; and the **residual specificity score** from the context matrix, computed for every partner, is highest for C-RADIOv4 in fold 1 (point value): for P = S2, spec(B) = C(B in D) minus the mean of C(B in X) over the other three real encoders X (those besides D and B); for P = S1, spec(B) = C(D in B) minus the mean of C(Y in B) over the other three real encoders Y (those besides D and B). A partner that is simply easy to predict from, or to predict, everything has a high P but a low residual, so this follows the uniform-richness case without a separate sentence. **Headroom applies only when P = S2** (point value, fold 1: score minus the trivial-structure baseline, C-RADIOv4 highest): under S2 the targets differ by partner, so each has its own baseline; under S1 the target is DINOv3 for every partner, one constant baseline that cannot change the order, so the residual guard carries the shared-structure check. **When P = S1 a width-matched guard is also required:** every source (all four partners) cut to its top 2,048 train principal components, and C-RADIOv4 must still be highest on S1 in point value in fold 1 (ridge is unchanged by rotating a full-width source, so only VGGT-Omega and C-RADIOv4 are changed).
- **Does not flag:** one partner X is ahead of C-RADIOv4 on P with the 98.3% paired interval (95% corrected for three comparisons) excluding 0 in fold 1, and X is ahead in point value in folds 2 and 3.
- **Unresolved:** anything else.

Margin sizes are reported; no effect-size floor is used (the intervals hold fitted maps fixed and are tight, so a tiny margin can exclude 0). Chance of "C-RADIOv4 highest" is 1 in 4; the interval, fold and guard requirements make a chance "flags" rarer by an amount not computed. The restatement wording above is reported next to the verdict.

### What each outcome means for the retrospective check (the lead's and user's call, not made here)
- Flags: P stays the lead candidate for the retrospective check, which is the point at which to ask for the three extractions (V-JEPA 2.1, UniTok, HRP), after the novelty check (open question 5). No finding yet.
- Does not flag: the retrospective check should not be built around P as measured here; this agrees with the README's current reading that the harm leaves no trace in clean frozen features, which stays the user's call.
- Unresolved or invalid: do not spend the extractions on this score; a follow-up is a new plan.

### Consequences of P = S2 (2026-11-10, recorded after the user's decision; nothing above is changed)
- The headroom requirement applies (S2 form, fold 1).
- The residual specificity score uses the S2 form: spec(B) = C(B in D) minus the mean of C(B in X) over the other three real encoders X.
- The width-matched guard is not required, and its 4 runs are not part of the run (26 invocations; cost bound 52,296 s).
- Signed off by the user 2026-11-10 (in the lead's chat): the design, with P = S2.

## Test-set use (`../../../_standards/07` line 6)
Test demos are never used to choose alpha, components, standardization, P or any threshold. Test demos are scored once per ordered pair, fold and feature type at the chosen alpha, plus the secondaries on the same demos; each invocation logs its count of test evaluations in its result JSON and in this folder's `LOG.md`. Fold 1's test demos are the same 100 demos already scored by the pilot diagnostics, the tally and the target-side experiment, so reuse is high and nothing here is a fresh holdout; folds 2 and 3 score demos those earlier experiments used for training or validation.

## Code needed (described, not written)
A new `src/feature_overlap.py` importing `diagnostics` (loading with synthetic partners, `demo_split`, `standardize`, `ALPHAS`, `ci`) as `src/target_diagnostics.py` does, with `--fold`, `--m`, `--types`, `--n-boot`, `--synth-seed`, `--save-draws`, a rotated split written in the new file, one eigendecomposition of X'X per source reused for every alpha and target, and a result JSON holding config, commit, scores, alphas, edge flags, coverage, baselines, split counts and the count of test evaluations. Tests in `src/test_feature_overlap.py`, thresholds written before running, one line each: exact linear relation gives about 1 and independent noise about 0; planted asymmetry (B a function of A, A has an extra block) gives the right S3 sign; wide pure-noise source reads about 0; demo-correlated random-walk features score about 0 under the demo split while a frame-level split on the same data would score high (so the test can fail), and no demo is in two sets; changing test frames changes no standardization, component or alpha; scrambling test targets leaves alpha unchanged; identical predictors give a difference interval of exactly [0, 0]; mean-of-components equals pooled R2 when component variances are equal; folds are disjoint, cover all demos, and fold 1 equals `demo_split`; dup and noise specs on a small fixture hit the gate values. A local run on a few demos, timed and with memory measured, comes before the Job (`../../../_standards/07`, sanity checks).

## Where it runs, and cost
One CPU-only Job run by the implementation session, **only on the user's own go in that window** (ConfigMap, then Job; each asked for), using the control-partners `job.yaml` as the template; check that `michael-persistent` is free first (control-partners PLAN line 125). Code and plan land by PR first.

Invocations: 12 (four real pairs x three folds) + 4 controls + 6 context-matrix pairs (the four pairs with DINOv3 come from the fold-1 real-pair runs; the six pairs among the four partners are new, loaded two encoders at a time) + 4 mean-type repeats (fold 1) = 26, plus 4 width-matched runs (fold 1) if P = S1 = 30.

2026-11-10: with P = S2 the 4 width-matched runs are not part of the run, so the count is 26 invocations.

Estimate (speculative), arithmetic: recorded per-pair script times of the tally diagnostics, which did more work than this (CKA, three probes, both pred directions): SigLIP 2 1,374 s, C-RADIOv4 1,572 s, SAM 3 1,750 s, VGGT-Omega 2,690 s (`../2026-10-19-feasibility-tally/results/tally-libero_goal-dinov3-*.json`, `seconds`, line 338, verified); sum 7,386 s.
- Four real pairs x 3 folds: 22,158 s.
- Four controls (control-partners PLAN lines 101, 108-111: 2 x 1,572 + 2 x 1,374): 5,892 s.
- Context matrix, six new pairs each bounded by the slowest recorded pair time: 6 x 2,690 = 16,140 s.
- Mean-type repeat (fold 1, bounded the same): 7,386 s.
- Baselines about 360 s; staging about 360 s.
- Width-matched runs, only if P = S1, bounded like the mean-type repeat: 7,386 s.
- Total 22,158 + 5,892 + 16,140 + 7,386 + 720 = 52,296 s = 14.5 h; with the width-matched runs 59,682 s = **16.6 h wall upper bound, about 33 CPU-h at 2 CPU; x1.5 for a slow node about 25 h** (set the Job deadline accordingly). It should be far below this (no CKA or action probes, one decomposition per source instead of a solve per alpha; my guess 2 to 5 h); the implementation session replaces it with a locally timed number.

2026-11-10: with P = S2 the cost bound is the 52,296 s figure (14.5 h, about 29 CPU-h at 2 CPU, about 22 h with the 1.5x slow-node factor); the 59,682 s figure applied only if P = S1.

Memory (speculative): the context matrix loads two encoders at a time, which `diagnostics.load` allows (it loads only the encoders listed in `--encoders`, `src/diagnostics.py` lines 149-186), so the whole Job stays at the control-partners size, 14Gi/16Gi requests and limits within 20% (`../../../_standards/08`). The largest pair is VGGT-Omega + C-RADIOv4, 8,704 columns, about 2.2 GB as float32 against 1.7 GB for D + C-RADIOv4 (6,656 columns), which peaked at 12.85 GiB in the target Job with k-shifted copies this experiment does not make (target-diagnostics README line 65) and ran inside 11Gi in the tally diagnostics Job. That is a judgment, not a measurement: the implementation session measures the VGGT-Omega + C-RADIOv4 pair locally first, and if its peak exceeds 11Gi sizes the whole Job at 24Gi instead. macOS RSS undercounts Linux (`../2026-11-09-control-partners/job.yaml` lines 117-118), so the implementation session also watches RSS in the first minutes of the Job, as the control-partners Job does, and resizes to 24Gi if it nears the limit. No GPU.

## Not in scope
No new encoder extraction (V-JEPA 2.1, UniTok, HRP stay stubs); no training, proxy policy, perturbed frames or GPU; no change to README key findings, STANDARDS, direction PLAN or LOG, `src/diagnostics.py` or `src/target_diagnostics.py`; no claim about any other Table 10 row; no stacks (three- and four-way compositions); no action-probe quantity in the score; no per-camera or grid versions; No citation is made: debiased CKA is used throughout but Kornblith et al. 2019 (recalled) is not in `literature/references.bib` (which holds liu2023libero, fei2025liberoplus, liu2026what, ranzinger2026cradiov4, guo2026metric; verified; also `STANDARDS.md` line 16). Predicting one network's features from another is a known way to compare representations (recalled, unchecked, so uncited).

## Open questions for the user at sign-off
1. **Primary score P: S1 or S2?** This is the user's decision, not the lead's or the writer's, and it must be fixed before any run.
   - **Answered 2026-11-10 by the user (lead's chat): P = S2.** The question text and the options below are kept as asked.
   - S1 = C(D in B), DINOv3 inside the partner. Directly tests the paper's stated distillation story. Its crude ordering is, as far as the record shows, unseen for three partners. Cost: the sources differ in width (C-RADIOv4 widest); handled by validation alpha, the noise gate and a required width-matched guard (every source cut to its top 2,048 components).
   - S2 = C(B in D), the partner inside DINOv3. Same input for every partner, so no width confound, and it matches the "adds nothing new" reading. Cost: it is not the distillation test, it sits in tension with the largest gain(B|D), and its crude ordering is already in the record.
   - My recommendation (the writer's, not the lead's): S1 primary, S2 and S3 reported, with the width-matched guard as part of the S1 rule.
2. Target rank 64, equal-weight components? Recommend yes, with the all-dimensions version secondary and no other ranks.
3. Keep the three-fold check? Recommend yes; it is the cheapest honest noise check for a verdict resting on four numbers.
4. When does the Job run? Recommend after control-partners, reusing its template, on the user's go.
5. Prior-art and citation check (regression-based or stitching-style encoder comparison; add Kornblith et al. 2019 to `references.bib`) before any write-up. Recommend yes; it does not block the run.

## Status
planned; signed off by the user 2026-11-10 (design, with P = S2); no code, no Job.

## Amendments
<!-- ### YYYY-MM-DD: what changed, why, and whether any results had been seen yet -->

### Amendment 2026-11-10: sizing run on the cluster instead of locally
Additions only; no text above is edited. No result had been seen, no code run on real features, and no Job submitted when this was written.
- **The user's decision (2026-11-10, in the lead's chat):** "i would prefer to not run anything locally". This replaces the local timed and memory-measured run of the largest pair that "Code needed" and "Memory" above call for before the Job.
- **What moves:** the measurement goes from the Mac to one short CPU-only Job on the cluster, `job-sizing.yaml` in this folder. Real extracted features exist only on the cluster volumes, and no earlier run of this code on real features is recorded.
- **Also stands in for the "local run on a few demos" sanity check** of `../../../_standards/07` that "Code needed" above names, not only for the largest-pair measurement.
- **Test-set use (departure from "scored once" in "Test-set use" above):** the sizing run scores the fold-1 test demos of the VGGT-Omega + C-RADIOv4 pair (the script logs its count of test evaluations; the sizing Job log records it), and the full Job scores those demos again. Nothing is chosen from the sizing run's scores (no alpha, threshold, component or decision), the full Job's copy is the one analyzed, and the count is logged in `LOG.md` when it runs.
- **What it runs:** the single largest invocation the full Job runs: the VGGT-Omega + C-RADIOv4 context pair (base `vggt-omega`, partner `c-radiov4`, fold 1, global), through the same `run()` function and flags as `job.yaml` (`--m 64 --n-boot 2000 --seed 0 --synth-seed 0 --save-draws`), with only those two encoders staged (19.9 GiB of local disk, as in the full Job's largest step).
- **What it records:** staging time per encoder, wall time of the invocation, peak RSS from the `peak_rss_gib` field of its result JSON (only that field is printed), and the pod's cgroup peak if readable (it includes page cache from the copies, so it overstates the process).
- **How the full Job is then sized:** by the PLAN's existing rule, unchanged: a peak above 11Gi gives the whole Job 24Gi, otherwise 14Gi/16Gi (limits within 20% of requests); the full Job's `activeDeadlineSeconds` is set from the measured time (if the measured time implies a full-Job time above the PLAN's bound, the lead asks the user before submitting). Extrapolation: the measured wall time of this invocation (two directions on the widest pair) is multiplied by the number of invocations, 26, as an upper bound, since the other 25 are no wider (controls and real pairs with DINOv3 have fewer columns) and the Job also adds staging, which is read from the sizing log. The sizing Job itself requests 24Gi/28Gi on purpose, so that the largest pair cannot be killed before its peak is known; that is a property of the measurement run, not a sizing of the full Job.
- **Blindness:** its output JSON is a real-pair result and is covered by the blindness rule (review of PR #21): it is not opened or printed before the controls are read. The script already withholds real-pair scores from stdout, `job-sizing.yaml` does not print, hash or tar the JSON, and it stays on the volume in `feature-overlap-sizing`, a folder separate from the full Job's `feature-overlap`, so the full Job's empty-output check still passes. Only the control partners of the full Job are read first, as before.
- **Scheduling:** it waits for `michael-persistent` (ReadWriteOnce, holds C-RADIOv4) to be free, i.e. until control-partners stage 2 has ended (see "Where it runs" above), and runs only on the user's own go in the implementation window.
- **Not a substitute:** the synthetic timing in this folder's `LOG.md` (random data at the real widths, on the Mac) is an early read only and does not replace this measurement.
- **Changes no design, score, threshold, gate, invocation count or outcome rule.**
