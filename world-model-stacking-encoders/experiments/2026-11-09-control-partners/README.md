# 2026-11-09 control partners for the input-side and target-side cheap tests

**Status:** stage 1 (seed 0) finished 2026-11-10; results collected and the PLAN's gates and decision rule applied (this file, `results/`). All four gates pass. Rule outcomes below. What the outcomes mean for the direction's key findings is the user's call and is not made here. The paired differences from the fetched draws are added below (PR #19, merged; no verdict changes). Stage 2 (seeds 1 and 2) was submitted 2026-11-10 17:23 UTC and is running; no stage 2 result exists.

**Question:** can gain(B|D) (input side) and T2 interference (target side) tell a partner that carries no useful information, or only extra width, from the real partners we measured, and how much of C-RADIOv4's gain and cost is its 4,608 dimensions? Design, predictions, decision rule and cost: `PLAN.md`. Job manifest: `job.yaml` (template); the submitted copy is `rendered/job.yaml` (commit sha filled in, submitted 2026-11-10). Code: `../../src/diagnostics.py` and `../../src/target_diagnostics.py` (synthetic-partner option; tests in `../../src/test_diagnostics.py` and `../../src/test_target_diagnostics.py`).

## Run (verified)
Job `stacking-encoders-control-partners`, pod `stacking-encoders-control-partners-b4ps5`, node `rci-nrp-cpu-04.sdsu.edu`, source commit `2274e32b383b82c6aa46140bbbfb7051a4c7ed3c` (recorded in every JSON), seed 0 for every random control and for the bootstrap (1,000 demo-level draws, 100 test demos), NumPy 2.2.6. Exit code 0; started 2026-11-10 09:26:18Z, finished 15:37:23Z (the lead's kubectl read; the finish equals the last pod-log line). LIBERO datasets revision `f13aa24a3da8c43c7225569f28c562979fa0e35a` (printed in the log).

## Files
- `results/control-partners-20261110.log`: the complete pod log, saved by the lead with `kubectl logs` (251,089 bytes, sha256 `bee6c0d9d06b5cce18b35574f21b0610bde0421ac02c060a2c7aa7676df56499`).
- `results/*.json`: the ten result JSONs, extracted from the log by `results/collect.py` under their original names (five `input-*`, five `target-*`). Each parses as JSON.
- `results/collect.py`: extraction, gates, every clause of the decision rule, stage 2 trigger, predictions, alphas, run times. Run `/opt/homebrew/bin/python3.11 -I collect.py`; its output is `results/collect-report.txt` (reproduced exactly). Recorded real values are read from the tally and target-diagnostics JSONs in the repo; the thresholds are the PLAN's constants.
- `results/stacking-encoders-control-fetch-draws.log` (90,500 bytes, sha256 `e324bad96de81df15929226d6e04223293848d49f48e11babfa9f2dec2d9e2fc`): pod log of the read-only Job `stacking-encoders-control-fetch-draws` (completed 2026-11-10): the volume's file listing, sha256 of all 15 files, and the base64 tarball of the five draw files.
- `results/draws-s1/input-draws-*.npz`: the five bootstrap-draw files (seed 0), byte for byte from that tarball (each about 14 KB). The repo `.gitignore` excludes `**/experiments/*/results/**/*.npz`, so they were added with `git add -f`.
- `results/paired.py` and `results/paired-report.txt`: hash checks, CI reproduction, pairing check and the paired differences (below). Run `/opt/homebrew/bin/python3.11 -I paired.py`; its output is `paired-report.txt` (reproduced exactly).

## Gates (all verified, from `results/collect-report.txt`)
| gate | compared | result |
|---|---|---|
| 1. rerun D + C-RADIOv4 reproduces the tally within 1e-5 (global probe R2s, gain, gain CI) | largest absolute difference over six numbers: 7.3e-8 (gain +0.03709637 vs +0.03709644) | pass |
| 2a. dup:dinov3 CKA with DINOv3 = 1.000 (+/-1e-6) | max abs deviation from 1 (global and mean, both cams, biased and debiased): 0 | pass |
| 2b. dup input abs(gain) < 0.005 | gain +0.0028 [+0.0016, +0.0040] | pass |
| 2c. dup target abs(interference) < 1e-6 under every scaling | max over 36 cells (3 scalings x 3 ranks x k-mean and k = 4, 16, 32): 2.2e-16 | pass |
| 3. noise:4608 and shuffle:dinov3 abs(debiased CKA with DINOv3) < 0.02 | largest: noise 0.0005, shuffle 0.0004 | pass |
| 4. splits 38,464 / 12,416 / 12,848 and 100 test demos | all ten JSONs match the tally's split and have 100 test demos | pass |

## Numbers the rule uses (verified)
**Input side**, global features, gain(B|D) = R2(D+B) minus R2(D), 95% demo-level bootstrap CI.

| control | dims | gain(B\|D) [95% CI] | alpha D alone / control alone / D+control |
|---|---|---|---|
| noise:4608 | 4,608 | -0.0269 [-0.0302, -0.0246] | 0.1 / 1000 / 0.1 |
| shuffle:dinov3 | 2,048 | -0.0031 [-0.0036, -0.0027] | 0.1 / 10000 / 0.1 |
| randproj:dinov3:4608 | 4,608 | +0.0011 [-0.0007, +0.0027] | 0.1 / 0.1 / 0.1 |
| dup:dinov3 (gate only) | 2,048 | +0.0028 [+0.0016, +0.0040] | 0.1 / 0.1 / 0.1 |
| C-RADIOv4 rerun | 4,608 | +0.0371 [+0.0325, +0.0423] | 0.1 / 0.1 / 0.1 |

Recorded real partners (tally JSONs): VGGT-Omega +0.0138 [+0.0105, +0.0176]; C-RADIOv4 +0.0371 [+0.0325, +0.0423]; SigLIP 2 +0.0168 [+0.0130, +0.0208]; SAM 3 +0.0326 [+0.0275, +0.0379]. Debiased CKA with DINOv3 (global agentview, wrist, mean agentview, wrist): dup 1, 1, 1, 1; noise +0.0005, +0.0000, +0.0001, -0.0002; randproj 0.997, 0.997, 0.997, 0.997; shuffle -0.0004, +0.0002, -0.0001, +0.0000.

**Target side**, interference of the control as co-target, global, r = 64, mean over k in {4, 16, 32}, 95% CI; "share" is the control's loss share under a' (T0).

| control | a' (primary) [CI] | b [CI] | a'' [CI] | share (a') |
|---|---|---|---|---|
| noise:4608:0.3 | +0.00071 [+0.00068, +0.00075] | +0.00074 [+0.00071, +0.00078] | +0.00196 [+0.00188, +0.00206] | 0.408 |
| noise:4608 | +0.00469 [+0.00354, +0.00574] | +0.00073 [+0.00070, +0.00077] | +0.01472 [+0.01287, +0.01652] | 0.684 |
| randproj:dinov3:4608 | +0.00076 [+0.00062, +0.00088] | +0.00076 [+0.00062, +0.00088] | +0.00126 [+0.00109, +0.00142] | 0.499 |
| shuffle:dinov3 | +0.08844 [+0.08556, +0.09133] | +0.04869 [+0.04741, +0.05010] | +0.08844 [+0.08556, +0.09133] | 0.684 |
| dup:dinov3 (gate only) | 0 (about 1e-17) | 0 | 0 | 0.500 |
| recorded G VGGT-Omega | +0.03430 [+0.03322, +0.03542] | +0.02279 | +0.04616 | 0.717 |
| recorded R C-RADIOv4 | +0.01508 [+0.01465, +0.01555] | +0.02514 | +0.02835 | 0.352 |
| recorded L SigLIP 2 | +0.01413 [+0.01376, +0.01454] | +0.02880 | +0.01413 | 0.288 |
| recorded S SAM 3 | +0.00322 [+0.00299, +0.00349] | +0.02072 | +0.00322 | 0.116 |

Secondary (not used by the rule), a' at r = 16 / 256: noise:4608:0.3 +0.00013 / +0.01052; noise:4608 +0.00048 / +0.04459; randproj +0.00009 / +0.00110; shuffle +0.10587 / +0.05366. T1a R2 of the control target alone and T1b gain are in `results/collect-report.txt`.

## Decision rule, clause by clause (PLAN wording quoted; verified)
**Input side**
- "*Can go negative:* noise:4608 or shuffle:dinov3 has point value <= -0.005 and a CI whose upper end is below 0." noise:4608: point -0.0269, CI upper -0.0246: **fires**. shuffle: point -0.0031 is not <= -0.005: does not fire.
- "*Capacity floor above zero:* noise:4608, shuffle:dinov3 or randproj:dinov3:4608 has point value >= +0.005 and a CI whose lower end is above 0." No control reaches +0.005 (largest point +0.0011, randproj): does not fire.
- "*Unresolved:* a control whose CI contains -0.005 or +0.005 ..." None: noise [-0.0302, -0.0246], shuffle [-0.0036, -0.0027], randproj [-0.0007, +0.0027].
- "*Floor near zero:* no control is unresolved and none meets either condition above ..." Not met (can-go-negative fired; noise's point value is outside +/-0.005).
- "Both of the first two can fire ...; then report both and call the result mixed." Only the first fired: not mixed.
- **Outcome, PLAN's label: "Can go negative".**

**Target side**, a', r = 64, global, k-mean
- "*Block size and loudness explain R:* both noise:4608:0.3 and randproj:dinov3:4608 have interference >= 0.0100 ..." noise:4608:0.3 +0.00071, randproj +0.00076: does not fire.
- "*Partner-specific:* both have interference <= 0.0050 (one third of R's) and dup is 0. ..." both are <= 0.0050 and dup is 0 (max 2.2e-16): **fires**. "partners are above the floor only if their value exceeds 0.0050 (R, L, G yes; S no)": checked against the recorded values, R +0.0151, L +0.0141, G +0.0343 exceed it, S +0.0032 does not.
- **Outcome, PLAN's label: "Partner-specific".**
- Facts about shuffle:dinov3 (verified; no interpretation): it is not an input to the "Partner-specific" clause or to the b clause (the PLAN names only noise:4608:0.3, randproj and dup for a', and noise:4608 for b); it enters only the negativity clause below. Its values (r = 64, k-mean) are +0.0884 under a', +0.0487 under b and +0.0884 under a''. Every recorded real partner is lower under each scaling: a' +0.0032 to +0.0343 (largest G), b +0.0207 to +0.0288 (largest L), a'' +0.0032 to +0.0462 (largest G). Under b its by-k values are +0.0467 (k = 4), +0.0470 (k = 16), +0.0524 (k = 32); alpha is unchanged (0.1 to 0.1) at k = 4 and k = 16 and rose from 0.1 to 1 at k = 32. The rule has no clause for a large positive control, and no call is made here.
- "*Can go negative* ... *Floor near zero:* every control's CI lower end is at or above -0.003. *Unresolved* ..." All five controls have a CI lower end at or above -0.003 (smallest: dup, -3.7e-17): none unresolved, none negative. **Outcome, PLAN's label: "Floor near zero".**
- "Under b: if noise:4608 under b is within 0.003 of or above the smallest real value (>= 0.0177 ...); if it is <= 0.0104 (half of SAM 3's 0.0207) the b values are partner-specific; in between, mixed, no claim." noise:4608 under b is +0.00073 [+0.00070, +0.00077], which is <= 0.0104: **"the b values are partner-specific"**.

## Stage 2 trigger
PLAN, verbatim: "Run seeds 1 and 2 of the random controls (a second Job, new name and output folder, the user's OK first) for any control whose stage 1 value is within 0.002 of one of the thresholds above, or for all of them if the user wants the standard 3 seeds up front (`../../../_standards/07`)." The decision rule also says an unresolved control is replicated by stage 2.

Unresolved controls: none (input or target). Point values within 0.002 of a threshold: **input shuffle:dinov3**, point -0.0031 against -0.005 (difference 0.0019). No other point value is within 0.002 of a threshold (among the comparisons the rule makes, every target-side point value is at least 0.0042 from its thresholds; the CI ends of the controls other than dup are at least 0.0036 from -0.003, and dup, which is deterministic, has a CI lower end 0.0030 from -0.003). Also within 0.002 when CI ends are compared against the thresholds the rule applies to them: input shuffle CI lower end -0.0036 against -0.005 (0.0014); input randproj CI lower end -0.0007 against 0 (0.0007, the "capacity floor" lower-end condition). Whether a trigger on these is binding for any call is not decided here: the input outcome "Can go negative" is fired by noise:4608 alone, which is far from every threshold (0.019 or more). Stage 2 is a second Job and needs the user's OK. (Reading of "value" as the point value, plus the CI ends the rule compares, is the worker's; the full list is in the report.) **Open reading for the user:** under the literal "one of the thresholds above", target noise:4608 under a' (+0.00469) is 0.0003 from the 0.0050 threshold, although no clause compares that control with that threshold (the a' clauses use noise:4608:0.3 and randproj only). Whether that counts as a trigger is the user's call; this README does not decide it.

## Predictions against the results (speculative predictions from the PLAN, results verified)
| prediction | predicted | result | |
|---|---|---|---|
| input noise:4608 | -0.03 to 0.00, most likely -0.015 to -0.005 | -0.0269 | inside the range, outside the most-likely band (more negative) |
| input dup | within +/-0.003 | +0.0028 | inside (by 0.0002) |
| input randproj | -0.005 to +0.005 | +0.0011 | inside |
| input shuffle | -0.01 to 0.00, nearer zero than noise | -0.0031 | inside; nearer zero than noise: yes |
| rerun D + C-RADIOv4 | +0.037 [+0.032, +0.042], within 1e-5 of the tally | +0.0371 [+0.0325, +0.0423] | as predicted (gate 1) |
| target dup, every scaling | 0 (abs < 1e-6) | 2.2e-16 at most | as predicted |
| target randproj a' | 0.000 to 0.015, central 0.005 | +0.00076 | inside (low end) |
| target noise:4608:0.3 a' | 0.002 to 0.02, central 0.008 | +0.00071 | below the range (miss) |
| target noise:4608 a' | 0.005 to 0.035, central 0.015 | +0.00469 | below the range (miss, by 0.0003); larger than the 0.3 version: yes |
| target shuffle a' | 0.003 to 0.02 | +0.08844 | above the range (miss, about four times the top) |
| target noise:4608 under b | 0.015 to 0.03 | +0.00073 | below the range (miss) |
| target dup under b | 0 | 7.4e-17 at most | as predicted |

## Alpha chosen with and without each control (verified)
Input side: D alone 0.1 in every run; D+control 0.1 for every control (the control alone chose 1000 for noise and 10000 for shuffle). So no input-side alpha shift. Target side (reduced-rank fits; 27 cells per control = 3 k x 3 ranks x 3 scalings, alpha with the control compared with DINOv3's target alone, same inputs (the inputs still include the control)): dup 0 cells differ, randproj 0, noise:4608:0.3 1 (k=4, r=256, a'': 0.1 to 1), noise:4608 8 (all a' or a'' cells, 0.1 to 1: k=4 r=64 a' and a'', k=4 r=16 a'', k=4 r=256 a' and a'', k=16 r=64 a'', k=16 r=256 a' and a''), shuffle 18 (a' in 7 of 9 cells, a'' in 7 of 9, b in 4 of 9, always 0.1 to 1). At the primary cell (r = 64, a', k = 4, 16, 32) the alpha rose from 0.1 to 1 for shuffle at all three k and for noise:4608 at k = 4 only. The PLAN's pre-registered sentence ("What is computed", target side): "a control whose interference comes with a larger alpha is showing an alpha effect, not capacity competition". As written it applies to shuffle:dinov3 at all three primary k (a', r = 64, alpha 0.1 to 1) and to noise:4608 at k = 4 (alpha 0.1 to 1). noise:4608 under a' by k (r = 64) is +0.0121 (k = 4, alpha 0.1 to 1), +0.0014 (k = 16) and +0.0005 (k = 32), so its k-mean +0.00469 is mostly the alpha-shifted k = 4 cell. noise:4608:0.3 and randproj, the two controls the "Partner-specific" label rests on, had no alpha change at the primary cell (r = 64, a', all three k). Reference R2_D of DINOv3's target alone (same inputs, r = 64, a', k-mean): dup 0.3603, randproj 0.3598, shuffle 0.3577, noise:4608:0.3 0.3412, noise:4608 0.3411, against 0.3642 to 0.3723 for the four recorded real partners. T1b gain(B|D) on D's future (not used by the rule): noise:4608 -0.0464, shuffle -0.0088. The full per-cell list and by-k values are in `results/collect-report.txt`. No alpha-effect call is made here.

## Run times against the PLAN's estimate (verified from the JSON `seconds` fields and the log)
Input runs took 300 to 531 s each, about a quarter to a third of the PLAN's 1,374 and 1,572 s (the PLAN's basis included mean-type probes and grid CKA; these runs skip both). Target runs: dup 1,314 s (PLAN about 1,850), shuffle 3,237 s (about 1,850), noise:4608:0.3 5,645 s, noise:4608 4,715 s, randproj 4,741 s (PLAN about 4,190 for each of the three). Staging 460 s (PLAN about 5 min with download). Whole Job 6 h 11 min wall, against the PLAN's about 6.7 h (pessimistic about 10 h); 2 CPU requested for 6.18 h = 12.4 CPU-h requested (not a measured use), against about 13.

## Paired differences (stage 1, seed 0 only; added after the draws were fetched; all verified, from `results/paired-report.txt`)
These are single-seed numbers (synthetic seed 0, bootstrap seed 0). They say nothing about the variation from drawing a different noise matrix; stage 2 (seeds 1 and 2 of all random controls) was submitted on 2026-11-10 at 17:23 UTC and is running; no stage 2 result exists and nothing in this section uses it.

**What the arrays are** (read from `src/diagnostics.py`, not guessed): each npz holds two arrays of 1,000 values, one per bootstrap draw of the 100 test demos. `dinov3|X|global` is R2(DINOv3 + X) minus R2(DINOv3 alone), the gain of X given DINOv3, one value per draw; `X|dinov3|global` is R2(DINOv3 + X) minus R2(X alone), the gain of DINOv3 given X. The DINOv3-alone R2 draws themselves are not saved. The 95% interval is `np.quantile(draws, [0.025, 0.975])` with NumPy's default linear interpolation, the same call as `ci()` in `src/diagnostics.py`.

**Integrity (verified):** the tarball decoded from the log has the printed sha256 (`896693fd...`), holds exactly the five npz files, each member matches the volume's printed sha256 and the committed file byte for byte. Trailing newline: each of the ten committed result JSONs (all five `input-*` and all five `target-*`) is the volume copy plus one final newline (repo file minus that newline hashes to the volume's printed sha256; the repo file as committed does not). So the JSONs are otherwise the volume's bytes.

**CIs reproduced (verified):** for all five runs, the CI recomputed from the draws equals the JSON's `gain_ci95` (both arrays) and `asym_ci95` to a difference of exactly 0.

**Pairing check (verified, with a limit):** the exact test the lead proposed (DINOv3-alone draw arrays identical across the five files) cannot be run, because those draws are not saved. What was checked instead. Weaker checks: (a) DINOv3 alone has the same R2 and the same bootstrap CI in all five JSONs (max abs difference exactly 0); (b) suggestive only: in the noise and shuffle runs (own R2 near 0), the R2(DINOv3) minus R2(X) draws agree to a max abs difference of 1.1e-5 (correlation 0.99999997, against about 0.03 when one array's order is scrambled); with different weights the two arrays would carry independent R2(DINOv3) fluctuations of sd about 0.012 and would differ by about that much, but the probes of X are not shown to predict a constant, so this is not a proof; (c) for the copy, R2(DINOv3) minus R2(copy) is exactly 0 in every draw (within one run). Data-level test on all ten arrays, (d): rebuild the bootstrap weights as `bootstrap_r2` does (one `default_rng(0)`, one `multinomial(100, 1/100, size=1000)` call), fit a linear regression of each saved array on the weight matrix using the first 700 draws, and score the last 300 (out-of-sample R2). Criterion, written before this script's own output was looked at but after the reviewer's numbers were known: in-order score above 0.8, and every one of 20 permuted draw orders and a shift by one below 0.2. Result: in generator order 0.941 to 0.989 on all ten arrays; best of 20 permuted orders -0.035 to -0.099 and shifted -0.105 to -0.278 (per-array values in `results/paired-report.txt`); the check passes. So pairing follows from the code and is confirmed at data level for all five files. This is statistical, not bit-exact per draw: bit-exact identity rests on the code (one generator, one call, nothing partner-dependent drawing from it, the same demo order because all five JSONs record the same frames, split, files and 100 test demos, and the same NumPy and source version in one pod). For scale only: scrambling the draw order of the noise array changes the paired noise interval only slightly (to [+0.0589, +0.0706] from [+0.0584, +0.0709]), because the noise gain has little spread of its own.

**Paired differences:** gain(C-RADIOv4|DINOv3) minus gain(control|DINOv3), per draw; point values from the JSONs; gain(C-RADIOv4|DINOv3) is +0.0371 (rerun JSON).
| control | gain(control\|DINOv3) | paired difference | 95% interval |
|---|---|---|---|
| noise:4608 | -0.0269 | +0.0640 | [+0.0584, +0.0709] |
| dup:dinov3 | +0.0028 | +0.0343 | [+0.0296, +0.0394] |
| randproj:dinov3:4608 | +0.0011 | +0.0360 | [+0.0315, +0.0414] |
| shuffle:dinov3 | -0.0031 | +0.0402 | [+0.0355, +0.0456] |

**Which rule clause they bear on:** the PLAN's only use of a paired difference is the "Capacity floor above zero" clause: "Then every real partner is re-read as gain minus the matched-width noise gain (paired CI), and the 4,608-dim partner C-RADIOv4 is judged against noise:4608." That clause did not fire in stage 1 (no control has a point value of +0.005 or more with a CI above 0), so no verdict changes; the numbers are reported for the record. The PLAN's other mention is a prediction note (input side): for noise below -0.005 with CI under 0, "R's +0.037 is information on top of a width penalty (paired difference larger than +0.037)". The stage 1 report said this could not be checked without the draws. This is not an independent confirmation: per draw, "paired difference larger than +0.037" is the same statement as "the noise control's gain is below 0", which stage 1's noise interval ([-0.0302, -0.0246]) already showed; the draws add the interval only. Noise:4608's gain is negative (the "Can go negative" outcome), and the paired difference against noise:4608 is +0.0640, interval [+0.0584, +0.0709]; the interval's lower end (+0.0584) is compared here with the point value +0.0371 of C-RADIOv4's own gain, which is a comparison of an interval end with a point value. That is what the PLAN's note says to expect. What it shows, and no more: against the width-matched noise partner, C-RADIOv4 comes out ahead by more than its own gain, because the noise partner loses ground. No claim is made here about the direction's key findings. Against the other three controls the difference is close to C-RADIOv4's own gain (the copy, the projection) or somewhat larger (the shuffle).

## Not computed, memory, warnings
- **Memory is unknown.** No RSS reading was taken during the run and the pod log prints only the node's MemTotal and MemFree, no pod memory figure; nothing is inferred.
- Not run in stage 1, by the PLAN's scope: mean-type probes and targets, grid CKA for controls, noise at other widths. Stage 2 (seeds 1 and 2 of all random controls) was submitted by the lead on 2026-11-10 at 17:23 UTC and is running; no stage 2 result exists and nothing in this file uses it. The paired differences are in "Paired differences" above.
- **Log warnings** (everything outside the JSON bodies): the pip "running as root" warning and the pip "new release" notice (harmless, setup); five lines "alpha at grid edge: ['global|k=4|t1b_D|alpha=0.01']" (one per target run; the same edge hit recorded in the earlier target Job, the PLAN's pitfalls say recorded, not acted on). No traceback, OOM, kill, NaN or exception.

## What the outcomes do to the direction's finding (the PLAN's own prior statement, not this file's)
Quoted from `PLAN.md`: "floor near zero on input and size-explained on target: the finding becomes "the cheap checks cannot flag a bad partner", no evidence either way about D + R. Can-go-negative and partner-specific: the finding stands and is stronger, and the controls give a floor to report next to each real number. Mixed: report each test separately." The PLAN labels this "lead's call, not made here"; the user's call is also needed before anything goes to key findings (`_standards/05`).

LOG: `LOG.md` entries "2026-11-10 [agent, worker] stage 1 results collected, gates and decision rule applied" and "2026-11-10 [agent, worker] stage 1 paired differences from the fetched draws".
