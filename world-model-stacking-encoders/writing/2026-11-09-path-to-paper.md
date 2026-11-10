# Path to a paper: where the direction stands and what has to happen (2026-11-09)

Written for the user and the user's PhD mentor. It decides nothing. Labels: `verified` = read in a repo file (pointer given), `recalled` = from memory, `speculative` = judgement or arithmetic on assumptions. Paths are relative to `world-model-stacking-encoders/`. "Proxy PLAN" = `experiments/2026-10-27-proxy-calibration/PLAN.md`. "Perturbed PLAN" = `experiments/2026-11-06-perturbed-frames/PLAN.md`. "Exit memo" = `writing/2026-11-09-proxy-exit-options.md`. State is as of main at b805d4a.

## The short version

We are checking whether one bad result in someone else's paper is real. The paper says stacking two vision encoders, DINOv3 and C-RADIOv4, makes a robot model worse. Our cheap tests did not see that. So the paper we can write is a careful case study: does the drop reproduce when we train several times, and does it leave any trace in the frozen features? Which paper that is depends on three running experiments. All results could be in by mid-November at best, and early December is more realistic at the pace round 1 actually achieved (`speculative`, section 3). Writing time is not recorded anywhere, so submission is "result date plus writing time", which the mentor should help set.

## 1. The claim we are aiming at

The question, `README.md:5` and `decisions/004-paper-framing-case-study.md:9` (verified): "When frozen encoders are stacked as inputs and future-feature targets of a world action model, is the reported harm from pairing DINOv3 with C-RADIOv4 (liu2026what Table 10) reproducible across training seeds, and does it leave any trace in frozen features (clean, perturbed, or as co-targets), or does it arise only in training?" The predictor idea is the stated follow-up (`decisions/004-paper-framing-case-study.md:8`, option C).

The synthesis review's verdict line (`reviews/2026-11-09-synthesis.md:66`, speculative in the source): "If the proxy reproduces the drop and some cheap measure flags it, a CoRL or RSS paper is plausible, more so with more than 4 to 7 pairs. If the proxy does not reproduce it, the likely output is a careful replication note, workshop level on its own." Our clean-frame cheap tests did not flag the pair (`README.md:15`), so, in the author's reading (`speculative`), the first condition is at most half met in every ending; the perturbed-frames test could still count as a cheap measure that flags it. All venue, paper and headline judgments below are `speculative`.

**Ending (a): the drop reproduces across training seeds.** The locked rule calls it a "substantial proxy effect" if every seed drops and the average drop is at least 2.7 points, and says this is "a trigger for the higher-fidelity multi-seed WAM replication, not proof" (Proxy PLAN:160, verified). Paper (`speculative`): the case study. Headline (`speculative`): the reported harm shows up in a small stand-in model, the clean-frame cheap tests could not have predicted it, and the perturbed-frame result says whether any trace exists in frozen features. Venue: a robot-learning workshop or short paper is the safe reading. A main-conference paper needs more, for example a trace in the perturbed frames, more pairs, or the faithful replication, which the Exit memo prices at about 18 to 21 days of the whole 8-GPU allowance (Exit memo:75).

**Ending (b): the drop does not reproduce.** Proxy PLAN:161 gives "the proxy does not show a substantial effect" and adds that this "does not establish that the paper's result was training noise". Paper (`speculative`): the replication note. Headline (`speculative`): in a small from-scratch proxy with our stated choices, the pair did not clearly hurt, and the clean-frame cheap tests saw nothing either. The power memo limits the wording: "no substantial effect" cannot mean "D+R does not hurt" (`writing/2026-11-09-seed-power.md:11`). Venue: workshop or short note. Exit memo:95 calls this "a modest result, but a true one".

**Ending (c): the proxy cannot tell.** Either the retrained DINOv3 models miss the level target of 27 points (Proxy PLAN:385), or calibration fails (Proxy PLAN:155 to 157), or the noise or floor rules stop the round. Paper (`speculative`): the cheap-tests-plus-perturbed-frames-only note. Headline (`speculative`): the clean-frame tests could hardly have flagged a bad partner (if the control-partners run says so), and the perturbed-frame test does or does not single out the pair. Venue: short note or workshop. This is the thinnest ending.

## 2. Milestones in dependency order

**M1. Perturbed-frames phase B (rule P).** The test re-renders LIBERO-Plus scenes with picture changes and checks whether the DINOv3 plus C-RADIOv4 pair stands out. What decides it, verified: "D+R stands out if (i) e(R) is the largest of the four partners' e(B), and (ii) the lower end of the 95% paired-bootstrap CI of e(R) minus the next-largest e(B) is above 0" (Perturbed PLAN:96), read from one slice, "full-size features, global pooling, combined view" (Perturbed PLAN:526). With four partners an ordering match is a hint only (Perturbed PLAN:100). Depends on: phase A, done. Runs: implementation session. State: the first render Job sat Pending 8 h 25 min and was resubmitted 2026-11-10 05:29 UTC with a wider GPU list (`experiments/2026-11-06-perturbed-frames/LOG.md:438`). No test data read yet.

**M2. Control-partners run, then the headline re-read.** Fake partners of known value (noise, a DINOv3 copy, a random mix, a shuffled copy) go through the two cheap tests to learn whether those tests can flag a bad partner at all. Decision rule, verified: the input side "can go negative" if a noise or shuffle control reads at or below -0.005 with an interval under 0 (`experiments/2026-11-09-control-partners/PLAN.md:80`); if no control moves, the "floor near zero" reading says the input test cannot flag harm (line 83). The target side is "block size and loudness explain R" if both noise and projection controls reach 0.0100 or more (line 87), and "partner-specific" if both stay at 0.0050 or less and the DINOv3 copy reads 0 (line 88). What follows for the README finding is "lead's call, not made here" (line 95); the user decides afterwards. Depends on: nothing technical (CPU only); the plan puts it after phase B's GPU work. Runs: implementation session, with the user's go for the ConfigMap and again for the Job (line 133). State: designed, signed off, merged, not submitted.

**M3. DINOv3 16-layer retrain, its evals, and the re-check.** Two seeds, one Indexed Job, then full-benchmark evals of both, then the seed-noise check with thresholds unchanged (Proxy PLAN:320). What decides it: both retrained models must reach "a mean LIBERO-Plus success of at least 27 points", and missing it "is a decision for the user, not a second retrain" (Proxy PLAN:385). Depends on: phase B GPU Jobs finished and the user's go (`experiments/2026-10-27-proxy-calibration/jobs/RETRAIN.md:3`, "Phase B first"). Runs: implementation session. State: manifests written, not submitted. The 20-epoch fallback fires only on a first-epoch environment failure (RETRAIN.md:20).

**M4. The other 13 proxy runs and the paired analysis.** Five configs, three seeds, the locked outcome rule (Proxy PLAN:155 to 161): calibration first, then D+R against D. Depends on M3 returning "go ahead" with the level target met. Runs: implementation session for Jobs; planning reads the analysis. Calibration and the D+R comparison become readable together, so calibration is read first and the D+R lines withheld (Exit memo:57). State: not started.

**M5. Seed-power rerun.** The power memo says its numbers do not carry over: rerun `src/power_seeds.py` on the retrained pair (`writing/2026-11-09-seed-power.md:113`). Depends on M3's records. Runs: planning session. Duration: not recorded.

**M6. README key findings sign-off.** This is the `_standards/05` verification gate. The review asked to reword the headline "The simplest form of the idea fails the cheap check" (`README.md:15`) because the tests could hardly have gone negative (`reviews/2026-11-09-synthesis.md:80`, status "open" at line 82). The wording waits for M2 and M1, and for M4 if reached. Runs: planning drafts; the user signs.

**M7. Bib and citation hygiene.** Kornblith 2019 is not in `literature/references.bib` and CKA is used throughout (`STANDARDS.md:16`). agostinelli2022 stays out of the bib until read (`STANDARDS.md:15`). Yardi 2025 and Chun 2026 are unread leads (`decisions/004-paper-framing-case-study.md:14`). The Hewitt 2021 bib PR waits on the user's answer (`LOG.md:1566`, item 1). Runs: the literature session owns the reading list; the user reads. Needed before any write-up.

**M8. Mentor decisions.** Decision 004 has not been told to the mentor (`decisions/004-paper-framing-case-study.md:14`). Venue and timeline were to be discussed 2026-10-22 (`README.md:4`) and are still "to be decided with the mentor" (`STANDARDS.md:55`). Runs: the user. Nothing technical blocks this.

**M9. Outline, figures, draft, internal review, submission.** Nothing exists yet. The figure and table convention is blank (`STANDARDS.md:54`). Runs: planning drafts, the user edits and owns, a fresh read-only reviewer checks claims.

## 3. Calendar (`speculative` throughout)

Assumptions: nothing fails; day 0 is 2026-11-10 05:29 UTC, the render resubmission; the user's response time to each go is not recorded, so it is counted as zero; the 8-GPU allowance (`STANDARDS.md:48`) is shared by evals and training and nothing else uses it. The retrain is assumed to be submitted the moment phase B's GPU Jobs end. The implementation lead's recorded queue (direction `LOG.md:1535`) puts the control-partners CPU Job before the retrain; if that order is followed, the retrain starts 6.7 h (best) to 10 h (pessimistic) later (`experiments/2026-11-09-control-partners/PLAN.md:113`), which moves every M3 and later date by about that much. "Realistic" adds queue waits seen in the logs and, for evals and the 13 runs, round 1's measured pace. Inputs are cited under the table.

| Milestone | Best case | Realistic | Arithmetic |
|---|---|---|---|
| M1 phase B, last GPU Job | 2026-11-10 | 2026-11-11 | best: 6.4 h + X of 1.7 h = 8.1 h; realistic: 8.1 + 8.4 render queue + 4 VGGT/SAM 3 queue = 20.5 h |
| M1 final Job (rule P readable) | 2026-11-10 | 2026-11-11 | + 2 invocations of about 1 h: 10.1 h and 22.5 h |
| M2 control partners | 2026-11-10 | 2026-11-11 | starts at phase B GPU end; 6.7 h best, 10 h pessimistic (control-partners PLAN.md:113); CPU, off the critical path |
| M3 retrain training | 2026-11-10 | 2026-11-11 | best: 8 h after 8.1 h; realistic: 9 h queue + 8 h after 20.5 h |
| M3 evals and re-check | 2026-11-11 | 2026-11-14 | best: 128 to 139 GPU-h / 8 GPUs = 16 to 17.4 h; realistic: round 1's 64.7 h |
| M4 other 13 runs | 2026-11-16 | 2026-12-04 to 05 | 104 + 830 to 910 = 934 to 1014 GPU-h; best at 8 GPUs = 117 to 127 h; realistic at round 1's pace of about 2 GPUs = 467 to 507 h (19.5 to 21 days) after 11-14 |
| M5, analysis, README sign-off | not recorded | not recorded | no duration in any repo file |
| **First publishable result set, ending (a)** | **2026-11-16** | **2026-12-05** | M4 end, plus the unrecorded line above |
| **Submission, ending (a)** | **result set + W** | **result set + W** | W = writing, bib, review time; not recorded; for the user and mentor to set |
| First result set, ending (c), "stop the proxy" | 2026-11-11 | 2026-11-14 | latest of M1, M2 and the M3 re-check; submission = that date + W |
| First result set, ending (c), "proceed anyway" | same as (a) | same as (a) | the user decides; decision time not recorded |
| First result set, ending (c), "one more retrain" | 2026-11-17 | 2026-12-08 to 09 | (a) plus one cycle: best 8 + 17.4 = about 25 h; realistic 17 + 64.7 = about 82 h; plus a new amendment's sign-off, not recorded |

Inputs, all verified in the files named. Phase B critical path "6.4 h + X" (`experiments/2026-11-06-perturbed-frames/jobs/PHASE-B.md:53`); X is the noise-render estimate of 6068 s (`experiments/2026-11-06-perturbed-frames/LOG.md:431`); final Job about 1 h per invocation (PHASE-B.md:87), run twice (global, mean); render queue 8 h 25 min (perturbed LOG.md:438); phase A's SAM 3 shards "waited about 4 h" (PHASE-B.md:55). Retrain about 8 h per run including setup, both at once, queue "3 h to 9 h or more" (RETRAIN.md:15). Eval cost 64 to 70 GPU-h per checkpoint (Exit memo:17), so 128 to 139 for two, 16 to 17 h on 8 GPUs (Proxy PLAN:360). Round 1: first eval submitted 2026-11-06 21:24:11 (`experiments/2026-10-27-proxy-calibration/LOG.md:473`), last shard done 2026-11-09 14:04 UTC (LOG.md:890), which is 64.7 h. Thirteen runs at 8 h of training each is 104 GPU-h (Proxy PLAN:358). Round 1's pace is roughly 2 GPUs on average (128 to 139 GPU-h in 64.7 h), so the 13 runs take 934 to 1014 / 2 = 467 to 507 h, ending 2026-12-04 to 05; that is the realistic column.

If the queue improves (my assumption, not recorded anywhere): an average of 4 GPUs, twice round 1's pace, gives 234 to 254 h for the 13 runs, ending 2026-11-24 to 25, a first result set on 2026-11-25, and 2026-11-27 to 28 for the "one more retrain" row.

Ending (b) has the same data dates as (a). If the fusion-step test is then triggered, add about 190 to 210 GPU-h of rollouts per suspect (Exit memo:92, speculative there, design not written). No venue deadline is in the repo, so none is used; any the mentor names must be confirmed by the mentor.

## 4. Risks

All date effects and "pushes toward" readings in this section are `speculative`.

**The retrain misses 27 points.** A deeper model may still score low, and a better model may also have larger seed-to-seed gaps (Proxy PLAN:323, speculative there). Date effect: the 13 runs stop; the user chooses among four pre-written options (Proxy PLAN:330), and "one more retrain" costs about 1 to 3.4 days. Pushes toward (c).

**Queue waits.** The phase B render Job sat Pending 8 h 25 min (perturbed LOG.md:438); a round-1 training pod sat Pending 9 h and was deleted and resubmitted (proxy LOG.md:431) and eval pods over 3 h 54 min (proxy LOG.md:536). Each stage can pay this again. Date effect: roughly a day over the chain, as in the realistic column. No ending change by itself.

**Absolute scores about a tenth of the paper's.** Our two DINOv3 models succeed on about 7 to 8 episodes in 100, where the paper's succeed on about 84 (`writing/2026-11-09-seed-power.md:112`). The 2.7-point thresholds were set for the paper's range, which is why the level target exists. If the level stays low, a "no effect" reading is weak. Pushes toward (b) or (c).

**Only 4 partners and one LIBERO suite.** The cheap tests use libero_goal (`STANDARDS.md:17`) and four partners. "R is largest" has a 1 in 4 chance by chance (Perturbed PLAN:100), and the standards say not to claim significance with so few pairs (`STANDARDS.md:31`). No date effect. It lowers the venue ceiling in every ending.

**Idle-GPU strikes.** The account is at 1 of 3 idle strikes (`STANDARDS.md:61`). The render pod's noise stage may leave the GPU idle; PHASE-B.md:93 puts the worry at a noise render over about 3 h, and the current estimate is about 1.7 h (6068 s, perturbed LOG.md:431), under that line. A third strike is "flagged for review and potential suspension" (`STANDARDS.md:47`), so GPU work might stop; no date can be put on that. Pushes toward (c).

**The mentor does not accept the narrower framing.** The user chose option C without telling the mentor (`decisions/004-paper-framing-case-study.md:14`). The predictor paper is "Not deliverable from the five configs we are running" (same file, line 7). If the mentor wants that paper, the plan changes, not just the date. Best handled early.

**Noise too large for three seeds.** One seed pair gives no usable range (`writing/2026-11-09-seed-power.md:110`). If the retrained gap is 2.7 points or more, the rule says "rethink seeds", and the memo's first option is four more training runs, 260 to 280 GPU-h of rollouts (Exit memo:29), about a day and a half at 8 GPUs (260 to 280 / 8 = 33 to 35 h). Pushes toward (c).

## 5. The user's decision points, in the order they arrive

1. **Go for the control-partners ConfigMap, then the Job** (`experiments/2026-11-09-control-partners/PLAN.md:133`). Rule: that PLAN, lines 70 to 95.
2. **Go for the retrain Jobs**, once phase B's GPU Jobs finish (RETRAIN.md:3). Known by then: the GPU count in use and whether the render ran clean. Pre-committed options: RETRAIN.md, the order and fallback sections.
3. **What rule P said, and what the control partners said.** Known: both results. The README headline re-read is the user's call (control-partners PLAN:95), then the key-findings sign-off.
4. **The re-check outcome.** Known: both retrained means, the noise gap, the level against 27. "Go ahead" starts the 13 runs; a miss gives four options (Proxy PLAN:330); "rethink seeds" gives Exit 1 of the Exit memo; a floor miss gives Exit 2.
5. **Telling the mentor about decision 004 and choosing a venue.** This can happen now. The information that matters most arrives at decision 4, so the user may want the mentor's view before the 13 runs spend about 930 to 1,010 GPU-h (Proxy PLAN:358 and :360).
6. **The branch the 13 runs land in.** Known: all 15 evals. Calibration is read first, D+R held back (Exit memo:57). Options by branch: Exits 3, 4, 5 and 6 of the Exit memo. The harm-reproduces branch also asks whether to do the faithful replication (Exit 4, option 4A), a separate proposal.
7. **Hewitt PR, the two reading leads, and the final outline.** The Hewitt merge and the Yardi and Chun file already wait on the user (`LOG.md:1566`). The bib is needed before any draft.
