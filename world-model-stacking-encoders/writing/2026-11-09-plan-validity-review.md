# Plan validity reviews, 2026-11-09

Note: the reviews were read-only; nothing here is a result; findings reported by subagents, spot-checked by the lead.

Pointers are file:line as the reviewers gave them, at the time of review (main 43c3b7d). Later commits moved lines. Labels are the reviewers' own: verified = read in a named file; speculative = the reviewer's judgment. Where the lead checked a point itself, it says "lead checked".

## Summary (plain language)

On 2026-11-09 the planning session asked three independent readers to poke holes in the direction: one for the logic of the plan, one for the design of the experiments, one for how new the idea is against the literature. The main worry is that the direction's big question (can we predict which encoder pairs help?) could not actually be answered by the work planned, and that everything rests on one reported drop from one training run in another paper. The experiment rules themselves were found sound and decisive, but the cutoff number (2.7 points) is asserted rather than justified, and three seeds may be too few to see an effect that size. Several docs were stale or wrong, and some planned exits ended in "the user decides" with no options. The literature check found no paper that scores encoder pairs before training for robots, so a narrower claim looks safe, but that is a lead, not a proof.

## 1. Direction logic and plan (reviewer 1)

Findings:
- The README question (a predictor) was not reachable by planned work. The only multi-seed ground truth planned is five proxy configs with two pairs (proxy PLAN:10-19); decisions/003:23 says "five configs test one anomaly, not a general predictor"; the item that adds combinations is parked (PLAN:24). Only the "documented negative result" half of "done" was reachable. (verified)
- Decision tree gaps. The exits "calibration fails" (proxy PLAN:158), "seed-noise rethink" (:222) and "floor" (:223) ended in "user decides" with no options. "Harm reproduces" triggers a "higher-fidelity multi-seed WAM replication" (:160, decisions/003:20) that was in no plan list and had no feasibility line. "No effect and no fusion suspect brings the drop back" had no next step (hypotheses-draft.md:78). (verified)
- Since decision 003 the direction had become "replicate and explain the paper's D+R result", while README and STANDARDS still said "draft, not yet confirmed". (speculative, stated plainly)
- Claimable contributions by outcome (speculative):
  - first multi-seed evidence on a harmful frozen-encoder stacking, with per-category localisation, if the drop reproduces;
  - a clean negative that conditional probing (Hewitt et al. 2021) is blind to it;
  - the target-side interference measure, the one measure that is not an obvious special case of prior work;
  - a modest fusion-design finding if a suspect brings the drop back.
  Not claimable: a predictor; the gain(B|A) construction itself.
- Stale or wrong items (verified): PF PLAN:3 called the signed-off amendment a draft; PF PLAN:7, 19, 96 pointed to "Next 3"; PLAN Now carried hour-by-hour status; README "L, the paper's best pair" was wrong (D+U is higher, LOG ~1315); hypotheses-draft old rule unmarked; STANDARDS lists agostinelli2022 as canonical though removed from the bib; Kornblith not added; the layer study (PLAN Next 3) had no design; "the harm" was used as if established.
- Three weakest points for an outside reviewer (reviewer's judgment, with cheap fixes):
  1. The whole direction rests on one number from one training run in another paper. Cheap fix: email the authors (parked, PLAN:23).
  2. Proxy fidelity and the success floor. Fix: pre-commit the exits.
  3. Cheap tests have no power and measure something different from closed-loop success. Fix: rewrite the question; per-encoder scoring against Table 1 if a predictor claim is wanted.

## 2. Experiment-design validity (reviewer 2)

Proxy rule and sizing:
- The proxy outcome rule is decisive (every result maps to one branch) and lp_eval.py matches it (lp_eval.py:236-255, exact Fractions). (verified)
- The 2.7-point harm threshold is asserted, not justified. It is an absolute success-rate figure from a paper whose models succeed about 84%; the PLAN never discusses what it means if the proxy's success scale differs. Pilot checkpoints: 1 success in 600 LIBERO-Plus rollouts (proxy PLAN:166). (verified gap)
- The locked sample-size rule sizes episode-sampling error, not training-seed noise; the pilot sizing was unusable. The PLAN has a false-alarm argument (:204) but no detection-power argument for 3 seeds. (verified)
- Calibration shows only that the proxy uses encoder features (a) and that adding a second encoder is not inherently harmful (b). (a) requires every seed, (b) only the average, an unexplained asymmetry; the R-alone config is used by no rule. (verified)
- Paired comparisons are apples to apples (same episodes, eval seeds 41-43, order seed, code commit) with an inherent confound: D+R also changes parameter count and adds a loss term. GPU model differs between training seeds (3090 vs A40) and within evals; recorded in LOG only, "noted, not corrected for". (verified)
- The seed-noise rule matches lp_eval.py seednoise (:339-345, 364-449) and is stricter in its completeness checks. The one constant 2.7 does three jobs (effect size, noise threshold, absolute floor). The floor is almost no floor (a model at 3% success passes); "go ahead" is possible with noise near the effect size or with both models near zero. (verified text, speculative judgment)

Perturbed-frames redesign:
- Test demos are locked in code until the final run (pf_analysis.py DEV_PHASES, allow_test=False; final refuses without a passing dev report, re-hashes features, first test-label read after all checks). Same renderer for train and test, with the Amendment (b) byte-level consistency check. Residual: the scene-XML diff gate for light/background variants is not implemented, and draws are uniform over all variant names. (verified)
- Rule P was computed in twelve slices while the PLAN named no deciding view (fixed by Amendment (c), user decision 2026-11-09). (verified)
- Rule P with 4 partners is a hint (1 in 4 by chance) and says nothing about closed-loop success. The paper-ordering comparison is descriptive only; S vs G in the paper is within single-run noise. The design cannot separate "a clean-trained linear readout is brittle to a low-level shift" from "action is no longer linearly decodable". (speculative, as given)

Across all experiments:
- Required baselines (3) objective-overlap label and (4) random ranking (STANDARDS.md "Methods and evaluation") have never been run anywhere. (verified)
- The README key finding stated R as top gainer without saying its CI overlaps S's (since fixed). Tally probe numbers are renderer-specific (since noted). (verified)

## 3. Literature (literature-agent session, 2026-11-09)

These are leads only. Every "nobody has done X" claim is speculative. Full reports are in that session's scratchpad, not in the repo.

- Q1, scoring pairs or a conditional gain before training: no robot, VLA or WAM paper found. Recent robot work scores one encoder at a time: Xie 2026 S2 (arXiv 2610.31670, App. B.6), Chun 2026 (arXiv 2610.29168, Sec. 4.2.2), and Yardi 2025, an action-score linear probe on 23 encoders (arXiv 2501.16389, Sec. III-D), which is closest to our probe. Combination scoring exists outside robotics or with training: Yang 2024 Law of Vision Representation (arXiv 2408.16357, fits a regression on fine-tuned configs) and Theia (arXiv 2407.20179, trains per teacher set). Suggested narrowing: claim a conditional score for robot/WAM stacking, not "scoring combinations".
- Q2, multi-seed replication of encoder-composition results: none found. Precedents for single-run encoder rankings flipping across seeds: Hu 2023 (arXiv 2304.04591, RL, Sec. 4.1; BC rankings argued steadier), Zeng 2026 (arXiv 2606.14153, top-1 not stable under seed perturbation, offline MSE), Burns 2023 (3 seeds, no ranking analysis), Kang 2026 (arXiv 2609.16745, abstract-level), Jiang 2026 (arXiv 2606.04233, abstract-level, LIBERO gains mostly not significant). arXiv 2606.13856 was withdrawn.
- Q3, clean-trained, perturbed-tested action probe: beyond arXiv 2606.07687, none found. Chun 2026 uses paired clean/perturbed LIBERO-Plus frames with a label-free geometry measure. ST-WAM (arXiv 2607.28993) finds frozen DINOv3 more stable than Wan2.2 VAE latents. ALPS (arXiv 2610.30120) is a clean state probe where C-RADIOv3 and DINOv3 lead. Tsagkas 2025 AFA (arXiv 2511.10762) trained policies under lighting/texture randomization. None orders SigLIP 2, SAM 3, VGGT-Omega, C-RADIOv4.
- Correction of the planning lead by the literature agent: "L > S > G > R" is the Table 10 pair ordering. Table 1's single-encoder order differs (C-RADIOv4 is highest of the four).

## What was done about it

- decisions/004 (reframing): in progress; it was not on main when this file was written.
- `writing/2026-11-09-proxy-exit-options.md`: options for the unfinished exits, pre-committed before the seed-noise verdict.
- LOG entry 2026-11-09 [agent, planning] docs: stale and wrong lines fixed (README, PLAN, STANDARDS, hypotheses-draft).
- perturbed-frames PLAN Amendment 2026-11-09 (c): names the deciding view for Rule P.
