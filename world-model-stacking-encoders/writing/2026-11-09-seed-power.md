# How often would the 3-seed proxy rule catch a real drop? A first measurement (2026-11-09)

Written by an agent worker for the lead and the user. Status: informs the write-up and the user's later decisions. It does not amend the locked rule, and it recommends no change to any threshold. Labels: **verified** (computed by `src/power_seeds.py` this session or read in the cited file this session), **recalled**, **speculative**.

## In plain words

**The question.** The proxy replication trains DINOv3 alone and DINOv3 + C-RADIOv4 three times each, and the locked rule calls a drop "substantial" only if every seed shows a drop and the average drop is at least 2.7 points. The plan already argued about false alarms (how often noise alone would trigger that). It never said how often a real drop would be caught. This memo answers that, using the one measurement we have of how much two training runs of the same recipe disagree.

**The one-paragraph answer.** Our two DINOv3 runs differ by about 1.16 points. That is one pair, so the true run-to-run spread could be much smaller or much larger than it suggests. Taking it at face value, a real drop of the paper's size (5.41 points) would be caught essentially every time; a real drop of exactly the harm line (2.7 points) would be caught about half the time, however quiet the training noise is, because the line sits right at that drop; and a real drop of half the harm line (1.35 points) would almost never be flagged. If the true noise is twice or four times larger than this one pair suggests, the paper-size drop is still caught most of the time but not always, and the half-size drops get lost. Further up the range the data still allow (8x to 32x the estimate), the rule would miss most paper-size drops (flagged 0.368, 0.232 and 0.172 of the time at 8x, 16x and 32x), so the reassuring numbers rest on the noise being near this first estimate. A hypothetical five-seed variant (all five seeds must drop and the average must reach 2.7; this is not the locked rule) does cut false alarms, since with no real drop and 4x noise it fires 0.025 of the time against 0.088 for three seeds, but it catches fewer real drops once noise is large. All of this is before the DINOv3 models are retrained deeper (Amendment (c)); the numbers must be rerun on the retrained pair.

**What "no substantial effect" can and cannot mean.** It can mean: no drop as large as the paper's showed up in this proxy, and the proxy's noise was small enough that a paper-size drop would very likely have shown (if the noise is near our estimate or up to about twice it). It cannot mean "D+R does not hurt": a true drop of 2.7 points or less is missed about half the time or more, and a true drop of 1.35 points is missed almost always. It also cannot mean "the paper's result was training noise" (the plan already says this, PLAN pre-registered rule item 2, verified). And it is only as good as the noise estimate, which comes from one pair.

## Method

**Data (verified).** Two finished DINOv3 runs, training seed 1 and training seed 0, each evaluated on the same 10,030 LIBERO-Plus episodes under eval seeds 41, 42, 43; per-episode score = mean success over the three eval seeds. Records are under `experiments/2026-10-27-proxy-calibration/results/full-benchmark/rollouts/eval-dinov3-s1/` and `.../eval-dinov3-s0/`, the episode list is `full-subset.json`. The script calls `lp_eval.seednoise` (the locked check) on them and reproduces LOG 2026-11-09 (t): means 8.315 and 7.155, delta +1.160.

**Step 1, noise from the pair.** If a config's mean success moves with a training-seed sd of sigma, the gap between two seeds has sd sigma times the square root of 2. So the point estimate is sigma = |gap| / sqrt 2 = 0.820 points (verified). One gap is one draw, and the chi-square argument with one degree of freedom gives a 95% range of 0.366 to 26.17 points, i.e. 0.45x to 31.9x the estimate (verified, script output). That range is too wide to be useful: the data fit a spread of a few tenths of a point and also a spread above twenty points. The estimate also includes a little episode-sampling noise, so it is slightly high (the episode part of the gap has sd about 0.19 points, verified). Pulling the other way is a bias of single draws: |gap| / sqrt 2 from one pair has a median of about two thirds of the true sd (0.6745, checked in `--selftest`; the single draw is below the true sd about 68% of the time), so the point estimate is more likely low than high. Judgment: the 2x row is therefore a fair central reading of the noise, not a pessimistic one.

**Step 2, episode-sampling noise.** The mean over 10,030 episodes of one config has a standard error of about 0.18 points (empirical, using the 3-eval-seed score), or 0.26 to 0.28 by the plain success-rate formula that treats every episode as a single coin flip (verified). The simulation uses the empirical figure averaged over the two seeds (0.177), and sqrt 2 times that (0.251) for a D+R vs D gap. That is larger than the measured sd of the paired gap itself (0.189), so it errs toward more noise.

**Step 3, simulate the locked rule.** For each training seed s, the D+R minus D gap is drawn as: minus the true drop, plus training noise, plus episode noise. Training noise: each run's mean has sd sigma, independent across runs, so the gap has sd sigma times sqrt 2. Then the rule is applied exactly as `lp_eval.verdict` does it (`src/lp_eval.py` lines 249-252 (246-248 is the calibration check), verified by reading): "substantial proxy effect" if every seed's gap is below 0 and the mean gap is at most -2.7; anything else is "no substantial effect". The script's branch function is tested against `lp_eval.verdict` on 29 synthetic tables (all agree, verified in `--selftest`). The "no effect" branch is split in two for information: some seed at or above 0, or all seeds below 0 but the mean not low enough. Calibration is assumed to have passed, since the D+R comparison only applies then. True drops tried: 5.41 (the paper's D+R minus D, recalled from PLAN), 2.7 (the harm line), 1.35 (the calibration margin), 0. Seed sd tried: 0.25x, 0.5x, 1x, 2x, 4x of the 0.820 estimate, plus 8x, 16x and 32x for the 5.41 drop only (3 seeds), because the one-pair range runs to about 32x. 200,000 simulated rounds per cell, fixed RNG seed 20261109, so the Monte Carlo error is about 0.001.

**Assumptions in the simulation (speculative, not tested).** Training noise is Gaussian. It is independent between D and D+R at the same seed (the plan notes that the same seed may share batch order and so correlate the noise, which would make drops easier to catch; the script has a `--rho` option, not used in the table). D+R has the same noise as D. The 5-seed rows apply a generalised version of the rule (all five gaps below 0, mean at most -2.7); that generalisation is not locked and is shown only so the user can see what more seeds would buy.

## Results (copied from the script output)

Script output, head (verbatim):

```
== Measured: dinov3, training seed 1 (A) vs training seed 0 (B), 10030 LIBERO-Plus episodes
mean success A: 8.315 points   B: 7.155 points
between-seed delta A - B: +1.160 points   (episodes A lower 1016, A higher 1257, equal 7757)
paired per-episode difference: sd 18.94 points, so episode-sampling SE of the mean gap 0.189 points
episode-sampling SE of one config's mean over 10030 episodes: A 0.184, B 0.171 points (empirical sd of the 3-eval-seed score / sqrt n); binary-success formula sqrt(p(1-p)/n): A 0.276, B 0.257
== Training-seed sd from this one pair
point estimate |delta|/sqrt 2 = 0.820 points (includes a little episode-sampling noise; expected sd of the gap from episodes alone is 0.19)
95% chi-square (1 df) range: 0.366 to 26.17 points  (0.45x to 31.9x the estimate)
One pair gives a point estimate with no useful confidence interval: the data fit a training-seed sd
from a few tenths of a point to more than twenty points. The grid below covers 0.25x to 4x only.

== Power table (rng seed 20261109; seed sd grid = (0.25, 0.5, 1.0, 2.0, 4.0) x 0.820 points; episode-sampling SE of one mean = 0.177 points, of a paired gap sqrt2 x that = 0.251; rho = 0)
Locked rule, D+R vs D (lp_eval.verdict lines 249-252): effect iff every seed's gap < 0 and mean gap <= -2.7;
anything else is 'no substantial effect'. Calibration assumed passed.
```

#### 3 training seeds each of D and D+R (locked rule); 200000 simulated rounds per cell
| true drop (pts) | seed sd (pts) | multiple of estimate | P(effect) | P(no effect, some seed >= 0) | P(no effect, all < 0 but mean above -2.7) |
|---|---|---|---|---|---|
| 5.41 | 0.21 | 0.25x | 1.000 | 0.000 | 0.000 |
| 5.41 | 0.41 | 0.5x | 1.000 | 0.000 | 0.000 |
| 5.41 | 0.82 | 1x | 1.000 | 0.000 | 0.000 |
| 5.41 | 1.64 | 2x | 0.957 | 0.030 | 0.014 |
| 5.41 | 3.28 | 4x | 0.653 | 0.325 | 0.022 |
| 2.7 | 0.21 | 0.25x | 0.499 | 0.000 | 0.501 |
| 2.7 | 0.41 | 0.5x | 0.501 | 0.000 | 0.499 |
| 2.7 | 0.82 | 1x | 0.499 | 0.035 | 0.466 |
| 2.7 | 1.64 | 2x | 0.452 | 0.327 | 0.221 |
| 2.7 | 3.28 | 4x | 0.326 | 0.627 | 0.046 |
| 1.35 | 0.21 | 0.25x | 0.000 | 0.001 | 0.999 |
| 1.35 | 0.41 | 0.5x | 0.000 | 0.047 | 0.952 |
| 1.35 | 0.82 | 1x | 0.024 | 0.337 | 0.639 |
| 1.35 | 1.64 | 2x | 0.138 | 0.630 | 0.232 |
| 1.35 | 3.28 | 4x | 0.186 | 0.768 | 0.046 |
| 0 | 0.21 | 0.25x | 0.000 | 0.875 | 0.125 |
| 0 | 0.41 | 0.5x | 0.000 | 0.875 | 0.125 |
| 0 | 0.82 | 1x | 0.000 | 0.875 | 0.125 |
| 0 | 1.64 | 2x | 0.019 | 0.875 | 0.106 |
| 0 | 3.28 | 4x | 0.088 | 0.875 | 0.036 |

#### 5 training seeds each of D and D+R (hypothetical generalisation, not locked); 200000 simulated rounds per cell
| true drop (pts) | seed sd (pts) | multiple of estimate | P(effect) | P(no effect, some seed >= 0) | P(no effect, all < 0 but mean above -2.7) |
|---|---|---|---|---|---|
| 5.41 | 0.21 | 0.25x | 1.000 | 0.000 | 0.000 |
| 5.41 | 0.41 | 0.5x | 1.000 | 0.000 | 0.000 |
| 5.41 | 0.82 | 1x | 1.000 | 0.000 | 0.000 |
| 5.41 | 1.64 | 2x | 0.948 | 0.050 | 0.002 |
| 5.41 | 3.28 | 4x | 0.517 | 0.479 | 0.004 |
| 2.7 | 0.21 | 0.25x | 0.502 | 0.000 | 0.498 |
| 2.7 | 0.41 | 0.5x | 0.500 | 0.000 | 0.500 |
| 2.7 | 0.82 | 1x | 0.494 | 0.055 | 0.452 |
| 2.7 | 1.64 | 2x | 0.376 | 0.483 | 0.141 |
| 2.7 | 3.28 | 4x | 0.181 | 0.808 | 0.011 |
| 1.35 | 0.21 | 0.25x | 0.000 | 0.001 | 0.999 |
| 1.35 | 0.41 | 0.5x | 0.000 | 0.077 | 0.923 |
| 1.35 | 0.82 | 1x | 0.005 | 0.496 | 0.498 |
| 1.35 | 1.64 | 2x | 0.067 | 0.810 | 0.124 |
| 1.35 | 3.28 | 4x | 0.076 | 0.913 | 0.011 |
| 0 | 0.21 | 0.25x | 0.000 | 0.969 | 0.031 |
| 0 | 0.41 | 0.5x | 0.000 | 0.969 | 0.031 |
| 0 | 0.82 | 1x | 0.000 | 0.969 | 0.031 |
| 0 | 1.64 | 2x | 0.003 | 0.969 | 0.028 |
| 0 | 3.28 | 4x | 0.025 | 0.969 | 0.006 |

#### 3 training seeds (locked rule), true drop 5.41, noise further up the one-pair range; 200000 simulated rounds per cell
| true drop (pts) | seed sd (pts) | multiple of estimate | P(effect) | P(no effect, some seed >= 0) | P(no effect, all < 0 but mean above -2.7) |
|---|---|---|---|---|---|
| 5.41 | 6.56 | 8x | 0.368 | 0.626 | 0.006 |
| 5.41 | 13.12 | 16x | 0.232 | 0.767 | 0.001 |
| 5.41 | 26.24 | 32x | 0.172 | 0.828 | 0.000 |

**Reading the table (verified against the rows above).**
- At the point estimate (1x) with 3 seeds: a true drop of 5.41 is flagged 1.000 (to three decimals); a drop of 2.7 is flagged 0.499; a drop of 1.35 is flagged 0.024; no drop is flagged 0.000.
- At 2x the estimate: 5.41 is flagged 0.957; at 4x it is 0.653.
- The 2.7 row sits near 0.5 at small noise because the drop equals the line: the average gap lands on either side of -2.7 by coin flip. Any threshold rule flags a drop exactly at its line about half the time; this is a property of the line, not of the noise level or of where the line was set.
- False alarms with no real drop: 0.000 at the estimate, 0.019 at 2x, 0.088 at 4x (3 seeds). The plan's rough figure was "about 4%" for a pure-noise 3-seed average at a noise level chosen from a gap of 2.7 (speculative, PLAN). That figure lies between the 2x (0.019) and 4x (0.088) rows here, so the two are not in conflict; this is not a check of either.
- 5 seeds: at a drop of 5.41 and 4x noise, P(effect) falls from 0.653 (3 seeds) to 0.517 (5 seeds), and at a drop of 2.7 and 4x noise from 0.326 to 0.181. The "some seed at or above 0" column grows with the seed count, so the every-seed-negative requirement is what costs power. A rule on the average alone would behave differently; that is not computed here and is not proposed.

## Limits

1. **One seed pair.** The training-seed sd is a point estimate from a single gap with no usable interval (0.45x to 31.9x). The grid covers 0.25x to 4x only; the upper part of the plausible range is not in the table. Every row of the table is conditional on a noise level the data barely pin down.
2. **Different GPU models between the two seeds (verified: seed 1 trained on an RTX 3090 and seed 0 on an A40, proxy LOG 2026-11-06 (c), 2026-11-07 (b) and 2026-11-09 (u); LOG 2026-11-09 (t) carries the note).** So part of the 1.16-point gap may be hardware, not seed. Noted, not corrected for. This could push the estimate up or down for the same-GPU case; the direction is unknown (speculative).
3. **The absolute-points rule on a low-success proxy (proxy exit-options memo, Exit 6, verified by reading).** Both DINOv3 models succeed on about 7 to 8 episodes in 100, where the paper's succeed on about 84. A 2.7-point drop is about a third of the proxy's whole score. The thresholds are absolute points taken from the paper's scale; here a paper-size drop of 5.41 points may be impossible or mean something different (speculative). This memo uses the paper's drop sizes as the question asks; it does not say a 5.41-point drop is plausible at this success level.
4. **Rerun after the retrain.** The DINOv3 models are being retrained deeper (Amendment (c), signed off, LOG (v)). The noise level and success level will change. Rerun `power_seeds.py` on the retrained pair (change `--a-dir`, `--b-dir`, `--subset`) and replace the table; nothing here carries over by assumption.
5. **Calibration is assumed to pass,** and calibration can itself fail by noise (PLAN says the 1.35 margin is tight); that branch is not simulated.
6. **Where the plan's wording stands.** The plan has "Three seeds are a small screening result, not an estimate of training variance" (verified). The numbers here say the same thing quantitatively: with 3 seeds the rule is a reliable alarm for a paper-size drop only if noise stays near or below about twice the first estimate, and it is close to a coin flip at the harm line itself.

## How to reproduce

```
/Users/michaelr808/Desktop/research/research26-27/vis-aria-depth/.venv/bin/python3 -I world-model-stacking-encoders/src/power_seeds.py --selftest
/Users/michaelr808/Desktop/research/research26-27/vis-aria-depth/.venv/bin/python3 -I world-model-stacking-encoders/src/power_seeds.py
```

The interpreter above is the vis-aria-depth venv (numpy 2.4.6); any Python with numpy works, and plain `python` is not on PATH on the Mac. The record paths are set by `--a-dir` (training seed 1 records, default `.../rollouts/eval-dinov3-s1`), `--b-dir` (default `.../rollouts/eval-dinov3-s0`) and `--subset` (default `.../full-subset.json`), all under `experiments/2026-10-27-proxy-calibration/results/full-benchmark/`; point them at the retrained pair for the rerun. Needs numpy only. Reuses `src/lp_eval.py` by import (no side effects on import). No cluster work, no new rollouts, no test-set use beyond reading records already produced.
