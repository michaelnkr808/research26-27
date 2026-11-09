# Proxy replication: what happens at each exit (options memo)

**Written before the seed-noise verdict was read by the planning session; no eval result of the final runs was opened for this file.** Nothing here is a decision. The user decides. Recommendations are the lead's, marked as such. Labels: verified = read in a named file this session; recalled = from project memory or earlier sessions, not re-read; speculative = reasoning, not tested. "Arithmetic" means a calculation on numbers already written in the repo. Paths are relative to `world-model-stacking-encoders/`; "proxy PLAN" is `experiments/2026-10-27-proxy-calibration/PLAN.md`.

## Summary in plain words

The proxy round was built so that each possible result leads to a next step, but several exits have no next step written down, and one points at work that no plan list contains. This memo writes the options for each exit down while nobody knows which exit we will land in, so the choice cannot be bent by the result. The most likely awkward cases are the first two: the two DINOv3 models may differ from each other by about 2.7 percentage points of task success (the size of the effect we are hunting), or they may sit at the "floor", meaning both succeed too rarely for the comparison to mean anything. If the rule says "rethink seeds", the cheapest honest fix is more training seeds for just the two configs that matter, and that has to be fixed before any DINOv3 + C-RADIOv4 result exists. The "harm reproduces" branch promises a more faithful replication of the paper's model, and the repo's own estimate says one such run takes days on eight scarce big cards, so that branch needs a real decision rather than a default. Last, the 2.7-point thresholds come from a paper whose models succeed most of the time while our small model trained from scratch may succeed far less often, so the memo proposes a clearly secondary "relative" reading and states that the locked absolute rule still decides.

**Key:** D = DINOv3, R = C-RADIOv4, L = SigLIP 2, "none" = no external encoder (the paper's VAE-only baseline, proxy PLAN:13). "Calibration" = does the proxy respond to the known controls at all (D beats none, D+L is not worse than D). "Floor" = both models succeed too rarely for the comparison to mean anything.

## Facts that apply to every exit

- verified (direction `PLAN.md`, "Now"): only the two `dinov3` seed-noise models are recorded as trained; their evals produce the seed-noise verdict. No full-length D+R model is recorded as trained (a 500-step pilot D+R checkpoint exists, proxy PLAN:166; it never enters the effect analysis, proxy PLAN:148 and 150). So any redesign decided now or right after the seed-noise verdict still happens before any D+R result exists. After the other runs are evaluated that stops being true, which matters for exits 3 and 5.
- verified (proxy PLAN:152): "Three seeds are a small screening result, not an estimate of training variance."
- verified (proxy PLAN, "Amendment 2026-11-06"): evaluation is the full LIBERO-Plus benchmark, 10,030 episodes times eval seeds 41, 42, 43, per checkpoint. So the "N" in "rethink seeds (or N)" (episodes) is already at its maximum; the lever left is the number of training seeds. (verified that N was the LIBERO-Plus subset size: `RUN.md:64`, proxy PLAN:148.)
- verified (`STANDARDS.md:48`): "Gate (stop and ask): more than 8 GPUs at once, or a sweep of more than 20 Jobs." `RUN.md:28` repeats these as a stop, plus "one training run > 48 h".
- Eval cost per checkpoint (arithmetic from proxy PLAN:181, which gives 959 GPU-h at one pod's speed or 1044 GPU-h at the other's for 15 checkpoints): about 64 to 70 GPU-h per checkpoint, excluding queue time, pod start and training. Those figures come from a 500-step pilot's rollout speed (proxy PLAN, "Pilot sizing result"), so they are estimates, not measurements of the full runs.
- No measured training cost per run is used in this memo (no figure quoted). The repo records the training Jobs in the experiment LOG (2026-11-06 (c), 2026-11-07 (b)); where training cost matters below it is marked "no figure used".

---

## Exit 1: the seed-noise rule says "rethink seeds (or N)"

**What the locked rule says.** proxy PLAN:222: "If |noise delta| is at or above 2.7: '3 seeds can't settle the question; rethink seeds (or N) before the other 13 runs.' This is a stop for the user's decision, not a new design." (verified)

**What the plan says happens next.** Nothing beyond the stop. The earlier draft (proxy PLAN:201) names "more training seeds for D and D+R" only as an example and says it "does not pick a new design". Direction `PLAN.md` Next 1 repeats "rethink seeds or N before training the other 13 runs". (verified)

**Options**
- **1A. Add training seeds for D and D+R only.** Cost, assuming 2 extra seeds each (seeds 3 and 4, so 5 each): 4 extra training runs and 4 extra checkpoints to evaluate, about 260 to 280 GPU-h of rollouts (arithmetic, 4 x 64 to 70), roughly a day and a half on 8 GPUs if the queue allows, plus training time (no figure used). GPU gate: training is at most 4 GPUs at once and evals can be held to 8 by capping parallelism, so the 8-GPU gate is not crossed. Job gate: 4 training Jobs plus 4 eval Jobs (one Indexed Job of 8 shards per checkpoint) is 8 Jobs; one Job per shard would be 4 + 32 = 36 Jobs and would need the user's OK. (How an Indexed Job counts against "20 Jobs" is not defined in `STANDARDS.md`; speculative.) With 1 extra seed each the figures are half. Main tradeoff: the calibration controls (none, D+L) stay at 3 seeds, and extra seeds average the noise down rather than remove it.
- **1B. Proceed with 3 seeds and report the result as underpowered.** Cost: nothing extra. Main tradeoff: if noise alone can produce a 2.7-point difference, a "substantial effect" verdict is weak evidence and a "no effect" verdict says very little; the headline of any write-up must say so, not a footnote.
- **1C. Stop the proxy and rely on the perturbed-frames test.** Cost: the remaining 13 runs are not trained, saving their training (no figure used) and about 13 x 64 to 70 GPU-h of rollouts (arithmetic). Main tradeoff: gives up the only experiment that measures actual task success; the perturbed-frames test measures features and a probe, not success (verified, direction `PLAN.md` "Now").

**Constraint on all three.** Any redesign must be written down and approved before any D+R result exists. That is satisfiable now; it stops being satisfiable once D+R evals are read.

**Recommendation (lead's, not a decision).** 1A with two extra seeds for each of D and D+R, Job shape fixed in advance as Indexed Jobs so the Job count stays under the gate, and the extra seeds named and committed before D+R training starts. It is the only option that answers the question the noise check raised.

## Exit 2: the floor says "can't tell"

**What the locked rule says.** proxy PLAN:223: "Floor: if either seed's mean LIBERO-Plus success is below 2.7 points, the answer is 'can't tell; user decides', whatever the noise delta (a delta near 0 means nothing when both runs score near 0)." (verified)

**What the plan says happens next.** The user decides; no options are written. (verified)

**Options**
- **2A. Check for a pipeline fault first, without reading success rates.** Look at error counts, episode completion, the action-sign and image-orientation conventions the LOG records, and that the checkpoints are the intended ones. Cost: hours of reading, no cluster use. Tradeoff: if a bug is found, the affected evals are rerun (about 64 to 70 GPU-h per checkpoint, arithmetic), the expensive but clean outcome.
- **2B. Train the DINOv3 models longer or larger, then rerun the noise check.** Cost: no training figure used; evals again at about 64 to 70 GPU-h per checkpoint. Tradeoff: changes the training budget after seeing a result, so it needs a written amendment, with thresholds restated as unchanged, before any D+R result exists.
- **2C. Stop the proxy and rely on the perturbed-frames test.** Cost: none beyond what is spent. Tradeoff: the proxy yields no usable comparison and the write-up says so.
- **2D. Read the result relatively (exit 6) and continue.** Tradeoff: contradicts the locked floor, written for exactly this case; allowed only as a labelled secondary reading, never as "go ahead".

**Recommendation (lead's).** 2A, then 2C if nothing is broken. 2B only with a written amendment; 2D never as an override of the floor.

## Exit 3: calibration fails after the full round

**What the locked rule says.** proxy PLAN:156 to 158: "(a) D vs none: Δ_s(D − none) > 0 for all three seeds, and their average ≥ +2.7 points. (b) D+L vs D, non-inferiority: average Δ_s(D+L − D) ≥ −1.35 points ... If either fails: 'don't interpret the D+R comparison from this proxy'." The PLAN itself says (b) is tight with 3 seeds, so noise alone can fail it, and that this "fails safe". (verified)

**What the plan says happens next.** Direction `PLAN.md` Next 2: "if calibration fails, no fusion tests." Nothing else. (verified, paraphrased from the conditional text)

**A timing problem (speculative).** When calibration can be read, all 15 evals exist, so the D+R comparison exists too. A pre-commitment that depends on not having seen D+R has to be enforced by procedure: read calibration first and withhold the D+R lines until the user has chosen. Whether `lp_eval.py analyze` can print calibration alone was not checked.

**Options**
- **3A. Accept the stop.** Report "don't interpret D+R from this proxy" and rely on the perturbed-frames test. Cost: none. Tradeoff: ends the proxy line with D+R uninterpreted.
- **3B. If only (b) failed narrowly, add training seeds for D and D+L only.** Cost: for 2 extra seeds each, 4 runs and about 260 to 280 GPU-h of rollouts (arithmetic), same shape as 1A. Tradeoff: changes the design after calibration was read; allowed only if committed before D+R is read.
- **3C. Treat a failure of (a), DINOv3 not clearly above no-encoder, as a proxy fault and diagnose it.** Cost: reading, possibly retraining (no figure used). Tradeoff: honest but open-ended.

**Recommendation (lead's).** 3A as the default. 3B only if (b) alone fails and the user wants to keep the proxy going, with D+R lines withheld until the user chooses.

## Exit 4: "Harm reproduces"

**What the locked rule says.** proxy PLAN:160: "Substantial proxy effect if Δ_s(D+R − D) < 0 for all three seeds and their average ≤ −2.7 points ... This is a trigger for the higher-fidelity multi-seed WAM replication, not proof that the effect is statistically established." (verified)

**What the plan says happens next.** Nothing is listed. Decision 003 (Amendment 2026-10-28) says the outcome decides whether a "higher-fidelity, multi-seed WAM replication (Wan 2.2-based, the paper's recipe as closely as available code permits)" is warranted, but no item in direction `PLAN.md` (Now, Next, Parked) is that replication. (verified)

**Feasibility, from what the repo records**
- verified (decision 001, Context): "One full Fast-WAM run is ~3–3.5 days on 8× A6000 (speculative, from the 1-GPU bench ...)". The direction LOG's A6000-bench entry has the same estimate, labelled speculative there, with 6.02B trainable parameters and "24 GB cards need ZeRO-3".
- verified (proxy PLAN, "Amendment 2026-11-06 (b)"): A40 and RTX A6000 are special resources (`nvidia.com/a40`, `nvidia.com/rtxa6000`); every node that advertises either has `nvidia_com_gpu` allocatable 0 or absent, so a normal GPU request cannot match them. The named-resource route exists and worked: seed 0 trained on an A40 requested as `nvidia.com/a40` (direction `PLAN.md` "Now"; proxy PLAN:255-257, which says an `nvidia.com/rtxa6000` variant is added only with the user's OK). The direction LOG also records free A6000s only on CPU-starved nodes.
- arithmetic: D and D+R at 3 seeds each is 6 runs; at 3 to 3.5 days each on 8 cards, one after another, about 18 to 21 days of wall time using the whole 8-GPU gate, with evaluation on top. The `STANDARDS.md` 8-GPU gate applies directly to such a replication. `RUN.md:28`'s "one training run > 48 h" stop governs this proxy round's Jobs, not a future experiment with its own plan; it only shows the scale (one such run is well beyond that line).
- recalled (project memory, 2026-11-05, labelled speculative there): the full-paper recipe on one H100 was estimated in years, and one 80 GB card is too tight for it.
- No cost figure exists in the repo for a multi-seed higher-fidelity replication as a whole.

**Options**
- **4A. Do it.** Cost: weeks of 8-GPU time (arithmetic above), scarce card types, a new plan and decision record, and the user's OK at the compute gate. Tradeoff: the most direct test of whether the drop belongs to the paper's model.
- **4B. Do the parked "C-RADIOv4 as input only vs target only" test instead.** Cost: small proxy runs; a 3-seed pair of new configs is 6 runs, about 380 to 420 GPU-h of rollouts (arithmetic) plus training (no figure used). Tradeoff: the parked entry says it is only useful if the proxy shows the drop, which is this case; it locates where the harm enters but stays inside the proxy.
- **4C. Stop at the proxy result and write it up.** Cost: none. Tradeoff: the result stays "a small proxy shows a drop", with the missing Wan 2.2 prior flagged.

**Recommendation (lead's).** 4B, and bring 4A to the user as a separate proposal with its own plan, compute estimate and card availability. 4C is an acceptable stopping point if time is short. 4A should not start from this memo alone.

## Exit 5: "No substantial effect", and none of the three fusion-step suspects brings the drop back

**What the locked rule says.** proxy PLAN:161: "Otherwise (any seed ≥ 0, or average above −2.7) → 'the proxy does not show a substantial effect'. This does not establish that the paper's result was training noise: the harm may need the pretrained video prior or the full-size model." (verified)

**What the plan says happens next.** Direction `PLAN.md` Next 2 (conditional): the fusion-step suspects test (scale, double counting, grid mismatch), design not yet written and to go to the user first. Nothing is written for what follows if all three come back empty. `writing/hypotheses-draft.md` "Update 2026-11-06" open question (a) says a null cannot separate the combining step from what the proxy lacks (the pretrained video prior, full size), and (b) says suspect 2 may not be testable by undoing one of our choices because the proxy already copies the paper's loss. (verified)

**Cost of the fusion test (speculative; design not written).** If each suspect means 3 D+R runs with one choice changed, that is 3 training runs and 3 checkpoints of rollouts per suspect, about 190 to 210 GPU-h of rollouts (arithmetic), about 580 to 630 for all three, and about 18 Jobs if each run and each eval is one Job, just under the 20-Job gate. No cost figure for the test exists in the repo.

**Options**
- **5A. Stop and report a bounded null.** The write-up says: in a small from-scratch proxy, with our channel scaling, grid resize and loss weights, D+R did not clearly hurt; this cannot separate the paper's result from missing pretraining. Cost: none. Tradeoff: a modest result, but a true one.
- **5B. Run the parked input-only vs target-only test.** Cost: as in 4B. Tradeoff: its parked condition ("only useful if the proxy shows the drop") is not met here, so it would be a deliberate change of that condition.
- **5C. Ask the paper's authors for training-seed results or code (parked, direction `PLAN.md`:24; the user's call; per-category results would be an addition to the parked item).** Cost: an email from the user. Tradeoff: slow and may go unanswered, but it is the only route to information about the real model.
- **5D. Go to the higher-fidelity replication (exit 4A).** Tradeoff: compute as in exit 4. The locked rule did not make this a trigger on this branch.

**Recommendation (lead's).** 5A plus 5C, with the perturbed-frames test named as the remaining evidence on whether D+R drifts more under perturbation. No 5D without a new decision.

## Exit 6: scale reading (new, from the 2026-11-09 methods review)

**The issue.** The 2.7 and 1.35 thresholds are absolute success-rate points. They are half of the paper's D+R minus D gap (proxy PLAN:222: "half the paper's D+R - D drop of -5.41"), measured on models that succeed about 84% of the time (verified, `literature/liu2026what-table1-single-source.csv`: DINOv3 total 84.0, VAE-only 74.7; the D+R value is taken from the PLAN, Table 10 was not re-read here). Our proxy has no pretrained video prior and is small (verified, decision 003 Amendment). If its DINOv3 mean success is far below the paper's, 2.7 points is a much bigger share of its own score, so the same absolute gap means something different.

**What stays fixed.** The thresholds are not changed. The locked absolute rule decides the pre-registered branch (calibration pass or fail, "substantial proxy effect" or not, the seed-noise verdict and the floor). Nothing below can turn a failed branch into a pass or the reverse.

**Options for a secondary reading** (reported next to the locked verdict, labelled secondary)
- **6A. Relative reading.** Report each drop as a share of the proxy's own DINOv3 mean, next to the absolute value. For scale, the paper's drop was about 6.4% of its DINOv3 score (arithmetic: 5.41 / 84.0). Cost: a few lines in the write-up, and if wanted one extra printed column in the analysis. Tradeoff: a ratio is unstable when the proxy's mean is small, so it is reported with the same per-seed breakdown and is not a test.
- **6B. Place the proxy on the paper's scale.** One sentence: how far below the paper's DINOv3 and VAE-only scores the proxy sits, and that the thresholds were set for the paper's range. Cost: none. Tradeoff: context only; gives no reading of the result.
- **6C. State the limit only, no secondary number.** The write-up says the thresholds are absolute points from a high-success model and that lower proxy scores change what they mean. Cost: none. Tradeoff: the reader cannot judge effect size.

**Recommendation (lead's).** 6A with 6B, both labelled "secondary, post hoc scale reading, does not decide the branch", with the choice of 6A fixed now, before the proxy's mean is read.

**Acknowledged limit to state in any write-up (not a new rule).** The plan has a false-alarm argument for the 3-seed design (proxy PLAN:204: a rough check that a pure-noise 3-seed average reaches the harm line about 4% of the time, labelled speculative there) and says 3 seeds are a screening result (proxy PLAN:152). It has no detection-power argument: nowhere in the sections read for this memo does it say how often a real drop of the paper's size would be seen with 3 seeds and our noise. So "no substantial effect" cannot be read as evidence of no effect. (verified for lines 152 and 204; the absence of a power argument is speculative beyond the sections read.)

---

## Recommendations in one place (lead's, for the user to accept or change)

1. Seed-noise says rethink: add 2 training seeds each for D and D+R as Indexed Jobs, fixed before any D+R result.
2. Floor says can't tell: look for a pipeline fault first (without reading rates), else stop and lean on the perturbed-frames test.
3. Calibration fails: stop (default); add seeds only if (b) alone failed narrowly, with the D+R lines withheld.
4. Harm reproduces: do the parked input-only vs target-only test, and bring the full replication to the user as a separate proposal with compute and card availability.
5. No effect and no fusion fix: report a bounded null, ask the authors, no escalation without a new decision.
6. Scale: keep the absolute rule decisive, add a labelled secondary relative reading and the proxy's position on the paper's scale, and state the missing power argument.
