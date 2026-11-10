# Plan: control partners for the two cheap frozen-feature tests

Written 2026-11-09 before any control run, as a design for the user to sign off. Nothing here has been run on real features. Claims are labeled verified (checked in this session), recalled, or speculative. Predictions are speculative by definition; the decision rule is fixed now and is not to be edited after results (changes become dated amendments below, per `../../../_standards/04`).

## In plain words
The two cheap tests (input side: does adding a partner help an action probe; target side: does adding a partner's future as a second prediction target hurt DINOv3's future) were only ever shown real partners, and every real partner came out "helps a little" or "costs a little". A thermometer that has only read normal temperatures hasn't shown it can read a fever. So we feed the tests fake partners whose true value we know: pure noise (useless), an exact copy of DINOv3 (adds nothing), a scrambled copy of DINOv3 (real-looking but unrelated to the frame), and a random mix of DINOv3 stretched to C-RADIOv4's width (nothing new, but just as wide). If the tests give such fakes the same small positive or small cost as the real partners, the tests can't flag a bad partner, and the finding "the cheap check fails" is really "the cheap check can't see anything".

## Question
Can the input-side test gain(B|D) and the target-side test T2 interference tell a partner that carries no useful information, or only extra width, from the real partners we measured (SigLIP 2, SAM 3, VGGT-Ω, C-RADIOv4), and how much of C-RADIOv4's measured gain and cost is just its 4,608 dimensions?

## Why this is needed (verified, reviewer text)
`../../reviews/2026-11-08-finished-experiments.md` line 23: "No control partner (a noise or duplicate B, a dimension-matched B) was run". Line 213 (target-side code review, item 5): no control partner "to show what interference a harmless second target of this size produces. Under b all four partners cost 0.021-0.029, which is a hint that 'any equally weighted second target costs about 0.02'". Same review, line 21: gain(B|D) "is very unlikely to go clearly negative"; lines 22 and 209: under the primary scaling interference follows B's loss share. The direction `README.md` finding "The simplest form of the idea fails the cheap check" rests on these two tests. This experiment does not edit that finding; the lead decides after the results.

## Controls (all generated in memory from DINOv3's loaded features; no new extraction)
Sizes are totals over the two cameras (the probe and the target see both cams concatenated). DINOv3 is 2,048 dims, C-RADIOv4 4,608 (tally README line 11, 13).

| control | spec | dims | what it is | why it is here |
|---|---|---|---|---|
| (i) noise | `noise:4608` | 4,608 | Gaussian noise, 2,304 per cam, per-channel variance equal to DINOv3's mean per-channel variance on train frames | Pure extra capacity, zero information. Answers "how much of R's gain is just its width" and "what does a useless target of R's size cost". |
| (ii) copy | `dup:dinov3` | 2,048 | exact copy of DINOv3 | Adds no new direction. Anchor: input gain should be about 0; target interference should be exactly 0 (see Predictions). A nonzero target value means a bug. |
| (iii) projection | `randproj:dinov3:4608` | 4,608 | fixed random linear map of each cam's DINOv3 features (1,024 to 2,304 dims, one Gaussian matrix per cam, entries N(0, 1/1024)) | As wide as R and made of real DINOv3 structure (correlated channels), but no information beyond DINOv3. Separates width from information. |
| (iv) shuffle | `shuffle:dinov3` | 2,048 | DINOv3 with the frames permuted (one permutation, both cams and both feature types together) | Real feature statistics, but the pairing with frame, action and future is broken. Compare with (i): Gaussian noise versus realistic noise. |

Added with a reason:
- (v) `noise:4608:0.3`, **target side only**. Same as (i) but variance 0.3 times DINOv3's per-channel variance. Reason: on the target side the scale of the co-target changes its share of the loss (reviewer lines 22 and 209), and (i) is far louder than any real partner (white noise differences have variance 2 sigma^2 = 0.12 per channel; C-RADIOv4's recorded difference variance averaged over k is 0.034, so 0.3 gives about 0.036, verified from `../2026-10-21-target-diagnostics/results/target-diag-20261027T214944Z.log` T0 lines). (v) is the "same width, same loudness as R, no structure" control. On the input side features are standardized, so scale is irrelevant and (v) would repeat (i).
- The real pair D + C-RADIOv4 is rerun on the input side with `--save-draws`. Reason: the tally did not save its bootstrap draws, so a paired confidence interval for "R's gain minus noise's gain" needs R's draws from the same seed. The rerun doubles as a check that the changed code reproduces the tally (gate below).

Not included, reasons: noise at other widths (2,048, 6,144; follow-up only if (i) is nonzero, to see whether it scales with width); a real dimension-matched partner (needs a new encoder or extraction, out of scope); `mean`-type features (secondary in both earlier experiments; `--probe-types`/`--types` can add them, cost below).

## What is computed
Data and split as in both earlier experiments: libero_goal, 63,728 frames, 500 demos, demo-level split 38,464 / 12,416 / 12,848 (tally README line 6). Global features (primary). Base is always DINOv3.

**Input side** (`src/diagnostics.py`, one invocation per control, `--probe-types global --n-boot 1000 --seed 0`):
- gain(control|D) = R2(D+control) minus R2(D), with its 95% demo-level bootstrap CI (probes fixed, test demos resampled). Same split, same CKA frames and same 1,000 bootstrap draws as the tally: the draws depend only on the number of test demos (100), `n_boot` and `--seed 0`, so they are identical across runs (verified from the code: `bootstrap_r2` builds its weights from the seed and the demo count only).
- Paired difference gain(R|D) minus gain(control|D) with a CI from the saved draws (`--save-draws`), from the D + C-RADIOv4 rerun.
- Also reported: R2 of the control alone, debiased CKA(D, control) for global and mean features (sanity: copy 1, noise 0, shuffle 0), alpha chosen. No grid CKA for synthetic partners (they have no grid features).

**Target side** (`src/target_diagnostics.py`, base DINOv3, partner = the control, `--types global --n-boot 1000 --seed 0`, default k in {4, 16, 32} and ranks 64 (primary), 16, 256):
- T2 interference of the control as co-target under the three scalings a' (primary, w = 1/d), b (equalized), a'' (pooled), averaged over k, with CIs. Compared against the recorded real values (`../2026-10-21-target-diagnostics/README.md` lines 33-36): G +0.0343, R +0.0151, L +0.0141, S +0.0032 under a'; b between 0.0207 and 0.0288; a'' G 0.0462, R 0.0283, L 0.0141, S 0.0032.
- Also: T0 loss share of the control under each scaling, T1a, T1b, and the alpha chosen with and without the control (the reviewer flagged that alpha is chosen per condition on its own weighted loss, line 210; a control whose interference comes with a larger alpha is showing an alpha effect, not capacity competition).

**Seeds.** The random controls (noise, projection, shuffle) are generated from `--synth-seed`; the bootstrap seed stays 0 in every run so the demo draws are the tally's. Stage 1 uses synthetic seed 0 for everything. The bootstrap CIs resample test demos with the fitted models fixed, so they do not include the variation from drawing a different noise matrix; stage 2 (below) measures that.

## Predictions (speculative, written before running)
Real values for orientation: input gain +0.014 (G) to +0.037 (R) (tally README lines 39-42); target interference under a' +0.003 (S) to +0.034 (G).

**Input side, gain(control|D), global**
- noise:4608: between -0.03 and 0.00, most likely -0.005 to -0.015. Reason (speculative): 4,608 irrelevant standardized columns are fitted on about 51,000 train+val frames, so the probe fits some noise and pays for it on the test demos, or shrinks DINOv3's own weights by choosing a larger ridge strength. The lead's alternative (a clearly positive gain) I think is unlikely (about one in ten) because ridge here is tuned on held-out demos.
  - below -0.005 with CI under 0: the input test can go negative. Width alone costs, so a partner of R's width that adds nothing would read as negative, and R's +0.037 is information on top of a width penalty (paired difference larger than +0.037).
  - within +/-0.005: floor near zero. The test cannot tell a harmless partner from one that adds nothing, and cannot go meaningfully negative at this size, so "every partner helps a little" says nothing about harm.
  - above +0.005 with CI above 0: part of every real partner's gain is capacity. Re-read the real partners as gain minus matched-noise gain (noise at their own width).
- dup:dinov3: within +/-0.003. It adds nothing, but ridge on two identical copies acts like half the penalty and the penalty grid is coarse (factor 10), so a small nonzero value is a resolution effect. Outside +/-0.005 means a bug: stop.
- randproj:dinov3:4608: between -0.005 and +0.005. If it sits clearly above noise's gain, wide real-structured columns help a little even with no new information (a finer effective ridge strength), which is part of what a wide real partner gets for free.
- shuffle:dinov3: between -0.01 and 0.00, nearer zero than noise:4608 because it has 2,048 dims and low effective rank.
- D + C-RADIOv4 rerun: gain +0.037 [+0.032, +0.042], reproducing the tally within 1e-5 (gate).

**Target side, interference under a', r = 64, mean over k**
- dup:dinov3: exactly 0 (|x| < 1e-6) under every scaling, rank and k. Derivation: the stacked target [Delta_D, Delta_D] has the same top-r directions, the same block weights and the same chosen alpha, so D's predictions are unchanged. Verified on toy data (`src/test_target_diagnostics.py` check 8). On real features a nonzero value is a bug: stop.
- randproj:dinov3:4608: between 0.000 and 0.015, central guess 0.005. It is fully predictable from D, so there is no competing signal, but its loss share (0.5 under a' at equal per-channel variance, about 0.69 under a'') and a random reweighting of D's directions (by up to about 3x, my estimate from the spread of a 1,024 to 2,304 Gaussian map) can shift which 64 directions are kept.
- noise:4608:0.3 (R's width and loudness, no structure): between 0.002 and 0.02, central guess 0.008. Only the B_{t+k} half of a noise target is unpredictable (the -B_t half is in the inputs, as for any partner), but their loss dominates the validation error that picks alpha, which may push alpha up and cost D accuracy.
- noise:4608 (louder): between 0.005 and 0.035, central 0.015; larger than the 0.3 version.
- shuffle:dinov3: between 0.003 and 0.02. Its target is the difference of two unrelated frames, so it is loud and unpredictable like noise but low-rank.
- Under b the hint from the review is that anything unpredictable of equal weight costs about 0.02. Prediction: noise:4608 under b lands within 0.015 to 0.03; dup under b is 0.

What each target outcome would mean in plain words:
- If dup is 0 while noise:0.3 and randproj are each about 0.015 or more (as big as R): the test measures how big and loud the second target block is, not what is in it. R's +0.0151 says nothing about R.
- If dup is 0 and both noise:0.3 and randproj are small (0.005 or less) while the real partners cost 0.014 to 0.034: the test does measure something about the partner beyond block size and loudness. S (0.003) would then be indistinguishable from the floor, and R, L and G stand above it.
- If a control comes out clearly negative: the test can show a helpful co-target, so it can go negative.

## Decision rule (fixed now; margins from the recorded real values, not tuned)
Margins: input m = 0.005 (about a third of the smallest real input gain, 0.014, and the size of the real gains' CI half-width); target m_T = 0.003 (the smallest real interference, S, 0.0032).

**Gates (nothing is read unless all pass; if one fails, debug and amend, do not interpret):**
1. Rerun D + C-RADIOv4 (input side) reproduces `../2026-10-19-feasibility-tally/results/tally-libero_goal-dinov3-c-radiov4.json` (global probe R2s and gain, and gain CI) within 1e-5.
2. dup:dinov3: CKA with DINOv3 = 1.000 (+/-1e-6); input |gain| < 0.005; target |interference| < 1e-6 under every scaling.
3. noise:4608 and shuffle:dinov3: |debiased CKA with DINOv3| < 0.02 (global and mean, both cams).
4. Split counts equal the tally's (38,464 / 12,416 / 12,848) and 100 test demos.

**Input side ("the test can go negative" versus "the floor is near zero"), on gain(control|D), global, seed 0:**
- *Can go negative:* noise:4608 or shuffle:dinov3 has point value <= -0.005 and a CI whose upper end is below 0.
- *Capacity floor above zero:* noise:4608, shuffle:dinov3 or randproj:dinov3:4608 has point value >= +0.005 and a CI whose lower end is above 0. Then every real partner is re-read as gain minus the matched-width noise gain (paired CI), and the 4,608-dim partner C-RADIOv4 is judged against noise:4608.
- *Unresolved:* a control whose CI contains -0.005 or +0.005 is assigned to neither side; stage 2 replicates it before any call that depends on it.
- *Floor near zero:* no control is unresolved and none meets either condition above (every point value within +/-0.005). The input test cannot flag harm.
- Both of the first two can fire (noise negative, randproj or shuffle positive); then report both and call the result mixed. dup:dinov3 is a gate (above), not a verdict input.

**Target side, on interference under a' (r = 64, global, k-mean), seed 0:**
- *Block size and loudness explain R:* both noise:4608:0.3 and randproj:dinov3:4608 have interference >= 0.0100 (two thirds of R's 0.0151). Then the target test measures the size and loudness of the co-target block.
- *Partner-specific:* both have interference <= 0.0050 (one third of R's) and dup is 0. Then the test sees something about the partner beyond block size and loudness; partners are above the floor only if their value exceeds 0.0050 (R, L, G yes; S no).
- *Mixed:* anything else; report both numbers, no claim.
- *Can go negative:* any control with CI upper end below -0.003. *Floor near zero:* every control's CI lower end is at or above -0.003. *Unresolved:* a control whose CI contains -0.003; stage 2 replicates it before either call. No control is built to help as a co-target, so "can go negative" can only fire by accident; if it fires, look for a bug or a mechanism (for example an alpha effect) before calling it a finding.
- Under b: if noise:4608 under b is within 0.003 of or above the smallest real value (>= 0.0177, SAM 3's 0.0207 minus 0.003), the b picture "every partner costs about 0.02" is the floor for any equal-weight co-target whose future is not predictable from the inputs; if it is <= 0.0104 (half of SAM 3's 0.0207) the b values are partner-specific; in between, mixed, no claim.

**Stage 2 (seed replicates).** Run seeds 1 and 2 of the random controls (a second Job, new name and output folder, the user's OK first) for any control whose stage 1 value is within 0.002 of one of the thresholds above, or for all of them if the user wants the standard 3 seeds up front (`../../../_standards/07`). A verdict stands only if the seeds agree on which side of each threshold they fall.

**What the outcomes do to the direction's finding (lead's call, not made here):** floor near zero on input and size-explained on target: the finding becomes "the cheap checks cannot flag a bad partner", no evidence either way about D + R. Can-go-negative and partner-specific: the finding stands and is stronger, and the controls give a floor to report next to each real number. Mixed: report each test separately.

## Cost
Everything is CPU, one Job, 10 runs (under the 20-Job gate), 2 CPU / 14Gi requests (`job.yaml`). Estimates are from the recorded run times of the two earlier Jobs; the new runs are the same shape (same frames, same input dims) as the real runs they are matched to.

Recorded (verified):
- Input side, script time per pair (the `seconds` field, line 338 of each `../2026-10-19-feasibility-tally/results/tally-libero_goal-dinov3-*.json`): SigLIP 2 1,374 s (D+L, 4,096 input dims), C-RADIOv4 1,572 s (6,656 dims), SAM 3 1,750 s, VGGT-Ω 2,690 s (6,144 dims). The tally README line 129 records the Job-level times (D+L and D+R together 49 min). Those runs included mean-type probes and grid CKA; ours skip both.
- Target side, global features only, time until the log line "loaded mean features" (global phase ends): `../2026-10-21-target-diagnostics/results/target-diag-20261027T214944Z.log` line 76 (SigLIP 2, 1,848 s), line 168 (SAM 3, 1,675 s), line 260 (VGGT-Ω, 3,804 s), line 352 (C-RADIOv4, 4,192 s). Whole-run time for all 4 pairs and both feature types: 5 h 45 m on 2 CPU, `../2026-10-21-target-diagnostics/README.md` line 25; 11.5 CPU-h and peak RSS 12.85 GiB (during D+R) at line 65. Staging: DINOv3 41 s (log line 24), C-RADIOv4 248 s (log line 307).

Estimate for stage 1 (speculative):

| part | runs | basis | time |
|---|---|---|---|
| input, copy and shuffle (4,096-d) | 2 | 1,374 s each | 0.8 h |
| input, noise, projection and the D+R rerun (6,656-d) | 3 | 1,572 s each | 1.3 h |
| target, copy and shuffle (4,096-d) | 2 | about 1,850 s each | 1.0 h |
| target, noise x2 and projection (6,656-d) | 3 | about 4,190 s each | 3.5 h |
| staging and download | | about 5 min | 0.1 h |
| **total** | 10 | | **about 6.7 h wall, about 13 CPU-h; pessimistic x1.5, about 10 h, 20 CPU-h** |

The x1.5 is a guess for a slower node (the target Job ran about 4x slower than the Mac). Stage 2 with seeds 1 and 2 of the three random controls on both sides adds about 2.5 h (input) and 8 h (target), about 10.5 h wall; its deadline would need raising to 24 h. Adding `mean`-type features would roughly double the target-side time (the recorded mean phase took 1,371 s for C-RADIOv4 and about 1,660 to 1,730 s for the 2,048-d partners). Memory: the largest synthetic pair has the same 6,656 input dims and frame count as D+R, which peaked at 12.85 GiB in the target Job (14Gi/16Gi, no OOM; `../2026-10-21-target-diagnostics/job.yaml` lines 75-80) and ran inside 11Gi/13Gi in the tally diagnostics Job (`../2026-10-19-feasibility-tally/diag-dlr-job.yaml` lines 66-70). Generating the largest synthetic partner takes 2 to 3 s and under 3 GiB transiently (measured locally at the real frame count on the Mac; macOS RSS undercounts, so watch RSS in the first minutes).

## Out of scope
- No new real encoders and no dimension-matched real encoder; no feature extraction; no GPU.
- No training, no proxy policy, no perturbed frames, no change to the input or target definitions, ranks, scalings, k, or the primary measures.
- No `mean`-type features and no grid CKA for the controls.
- No edit of the direction `README.md`, `PLAN.md` or `LOG.md`; no CKA confidence interval work.
- No cluster action without the user's separate go: creating the ConfigMap and submitting the Job are each asked for.

## Risks
- **RWO volume.** `michael-persistent` is ReadWriteOnce (RBD) and also holds the SigLIP 2 cache; only the D+R rerun needs it (C-RADIOv4 features), and DINOv3 comes from the ReadWriteMany CephFS volume. If another pod holds `michael-persistent` the pod stays in ContainerCreating. Check before submitting. Fallback: drop the volume and the rerun block, and compare controls with the tally's published CIs instead (an unpaired comparison, wider than the paired one).
- **Memory.** Sized from D+R (above). Features are staged to local disk (the target Job's try 1 loaded them in place over the network (a long idle load) and was OOM-killed at 9Gi/10Gi; try 2 staged them and resized; `../2026-10-21-target-diagnostics/job.yaml` line 11). Disk: raw libero_goal 5.9 + DINOv3 4.4 + C-RADIOv4 5.2 GiB, 20Gi emptyDir, 18Gi/21Gi ephemeral request/limit.
- **Reproducibility of the synthetic partners.** They are generated in the pod from DINOv3's loaded features with `np.random.default_rng([seed, crc32(spec), crc32(type), crc32(cam)])` (shuffle: `[seed, crc32(spec)]`), seed from `--synth-seed` (default `--seed`). The result JSON records, under `encoders[<spec>]`: the spec, the seed used, the NumPy version, the noise standard deviation per type and cam, the reference encoder, the source encoder, the dims per cam, and the first five entries of the permutation. The Job also records `SRC_GIT_COMMIT`, `SYNTH_SEEDS` and the pip freeze. NumPy does not guarantee identical random streams across versions, so a local rerun (the tests ran on NumPy 1.26.4) may differ from the Job (pinned 2.2.6); the record in the JSON is what to compare against.
- **Design risks (speculative).** (a) Noise on the target side is white, so its difference variance (2 sigma^2) is higher than real features' and does not grow with k; (v) matches R's level on average but not by k. (b) The shuffled partner's target is the difference of two unrelated frames, not a smooth future, so it is louder than a real partner's. (c) Alpha is chosen per condition on that condition's weighted loss, so a control's interference can be an alpha effect; the JSONs record both alphas and the analysis reports them side by side. (d) The CIs do not include the variation from drawing a different noise matrix (stage 2 measures it). (e) All of this is clean-frame linear probes on libero_goal with DINOv3 as base; it says nothing about training dynamics or perturbed frames.
- **Data revision.** The LIBERO download uses `revision="main"` (reviewer line 288, item 11). The Job prints the dataset sha so a later change is detectable.
- **Code change.** `src/diagnostics.py` gains the synthetic-partner option, `--synth-seed` and `--save-draws`; `src/target_diagnostics.py` gains `--synth-seed` and a help string. Real-encoder runs take the same path as before: verified by running main-branch and branch versions of both scripts on a two-real-encoder fixture and comparing every key of both result JSONs except wall time and the git commit string (identical), and by the existing synthetic tests passing before and after.

## Steps once the user signs off
1. Merge the branch by pull request (experiment code and plan), then set `SRC_GIT_COMMIT` in `job.yaml` to the merged commit. The user's go for each cluster action: ConfigMap `stacking-encoders-control-src` from `src/diagnostics.py` and `src/target_diagnostics.py`, then the Job.
2. Pre-checks as in `../2026-10-21-target-diagnostics/RUN.md` step 3 (nothing of ours running, `michael-persistent` free, features present, CPU capacity). First 10 minutes: pod phase, node, RSS against 14Gi.
3. Collect logs and JSONs from the CephFS folder `/mnt/features/stacking-encoders/control-partners/` into `results/` here; fill `README.md`; apply the decision rule word for word; explain to the user in plain words before anything goes into key findings.
