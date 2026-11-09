# Synthesis of the four independent reviews (2026-11-09) [agent, review]

Written by a review worker for the lead. It merges four reports that were each fact-checked by a separate reviewer and corrected. It adds no new experiments. Labels follow the reports: `verified` means the report's author checked it against a file, a git object or a run; `recalled` means from memory; `speculative` means judgement. Every per-line call below is judgement, so every call is speculative. The facts under each call keep the reports' labels. Short names for the reports: FE = finished experiments, LP = literature and promise, PR = proxy replication, PF = perturbed frames and docs.

## Scope

- The four reports, all dated 2026-11-08, all written against main at `bde59a7`:
  - FE: `reviews/2026-11-08-finished-experiments.md` (the five finished experiments and their code).
  - LP: `reviews/2026-11-08-literature-and-promise.md` (novelty, the starting fact from the paper, prior work per planned step, citation check).
  - PR: `reviews/2026-11-08-proxy-replication.md` (the multi-seed proxy: code against plan, fidelity to the paper, statistics, compute).
  - PF: `reviews/2026-11-08-perturbed-frames-and-docs.md` (the perturbed-frames merge, `pf_analysis.py`, direction-level docs).
- Main is now at `43c3b7d`. Main moved 22 commits past `bde59a7` while the reviews ran. PF says this itself: the perturbed-frames analysis code was rewritten (probes now trained on our own re-renders), and the reviewed `pf_analysis.py` is the superseded version. Statuses in the fix list are as of `43c3b7d`.
- Out of scope for all four: `vis-aria-depth` (another direction) and live cluster state (no kubectl, no Prometheus).
- Where two reports could disagree I opened the file. I found no real disagreement. Two apparent ones dissolve: FE says the older Jobs used `backoffLimit` 1 (compliant) while PF says the perturbed-frames Jobs used 0 (not compliant), which are different Jobs; and FE says pytest collects nothing in two test files while PR says it collects one test in `test_lp_eval.py`, which is a different file (a grep on main also shows no `test_*` function in the two FE files).

## Verdict per research line

Plain frame: the project asks whether a cheap test on frozen image features can tell, before training, which pairs of vision encoders help a robot model and which hurt. The paper shows one pair (DINOv3 plus C-RADIOv4) hurting. Nobody here has yet shown that this harm is real in our setting.

### (a) The cheap frozen-feature tests (done)

Call: stop (more clean-frame linear tests); keep the finished results as weak evidence, not as "the idea fails".
Reasoning (speculative): the computations are right and honestly documented, but the two tests could hardly have found the effect. The input-side test asks how much a second encoder adds, and a tuned linear fit almost never gets worse when it is given more inputs, so it could say "helps a little" but not "hurts". The target-side test mostly follows how big each encoder's raw numbers are, so it measured scale, and with the scales equalised all four partners look alike. Think of two thermometers where one cannot read above normal and the other reads room temperature. They did not show a fever, which tells us little about the patient.
Would most change the call: a control partner (pure noise, or a copy of DINOv3) run through the same tests. If the tests can go negative on a bad partner, they are informative; if not, they are closed for good.
Rests on: FE Summary items 4 and 7; FE "Code review" (probes and gain, target side). Facts are verified in FE; "could hardly have found it" is FE's speculative reading of verified code.

### (b) The multi-seed proxy replication (running)

Call: keep going.
Reasoning (speculative): this is the one piece that can produce a trusted answer key, because the paper used one trained model per pair and never measured training-seed noise. The code matches the locked plan and fails loudly instead of guessing, and nothing in it would silently change a result. What it can show is whether our small stand-in model reproduces the drop, not whether the paper's drop is real. A null here is weak evidence, because the stand-in lacks the paper's pretrained video model and uses our own choices for scale, grid and loss. Three seeds can find a drop as big as the paper's if training noise is moderate, and only about a coin flip for half that size.
Would most change the call: the seed-noise result (two identical-recipe DINOv3 copies compared on the full benchmark). If the two copies differ by about as much as the harm threshold, three seeds cannot settle the question and the remaining 13 runs should wait.
Rests on: PR Summary and sections 1.3, 3.7, 4, 5, 7; LP section 5. Code-versus-plan matches are verified in PR; the power table is PR's own simulation (speculative); the throughput worry (about 13 more checkpoints at roughly 76 GPU-hours of rollouts each, L4 cards only) is PR arithmetic.

### (c) The perturbed-frame test (redesigned, phase A running)

Call: keep going.
Reasoning (speculative): our earlier tests used clean pictures while the paper's harm appears in perturbed scenes, so a measure that never sees perturbed pictures cannot predict a perturbation-driven drop. This test fixes that, and no prior work was found that does it per encoder pair. The first design failed its own validation (re-rendered pictures differed enough from the stored ones that the probe lost accuracy), and the redesign trains probes on our own re-renders, which is a sound, common-practice fix. The reviewed analysis code matched its plan with only small gaps, and main has since rewritten it, so that code review is partly out of date. With four partners any ordering match is suggestive only.
Would most change the call: the dev gate result (does the re-render consistency check pass its 0.025 threshold) and, after it, whether rule P singles out DINOv3 plus C-RADIOv4 at all.
Rests on: PF Part A (Spec 1-9), Part B (B1-B9), Summary item 1; LP section 3(a). Verified items are marked in PF; that main's new code is fine is not reviewed.

### (d) The fusion-step suspects (conditional, not designed)

Call: not enough evidence.
Reasoning (speculative): the idea is that the way channels are combined (scale, double counting, grid mismatch) may be why the pair hurts, and it only runs if the proxy shows no drop. No prior work tests these in a world model, so the question is open, but nothing has been designed or signed off. One review finding makes it more important: our own choices where the paper is silent (per-channel scaling, resizing, a channel-mean loss that gives each source equal weight) may be hiding the drop. The channel-mean choice is not yet listed among those choices in the PLAN.
Would most change the call: which branch of the locked outcome rule the proxy lands in. Only the "proxy does not show a substantial effect" branch triggers it.
Rests on: PR sections 1.1 (point 1) and 2; LP section 3(c); PLAN Next 2. Verified: the loss reduction in `proxy_wam.py:420-422`. The rest is speculative.

### (e) Intermediate layers (PLAN Next 3)

Call: rethink (reframe before starting).
Reasoning (speculative): "earlier layers help" is already reported for a single frozen DINOv3 inside a world action model, in LiLa-WAM (a large gain from block 17 over the final block) and ReWAM (averaging blocks 12 to 24 helps), each on about 10 tasks with seed counts not stated. Repeating that would not be new. What is not answered is whether layer choice changes pair effects, for example whether the DINOv3 plus C-RADIOv4 harm shrinks at intermediate layers. PLAN Next 3 names no prior work and has no pair framing.
Would most change the call: the user reading LiLa-WAM and ReWAM and deciding the pair-by-layer question is the interesting one; or the proxy showing no pair harm at all, which would make the layer question stand alone.
Rests on: LP section 3(d) and its fix-list row on PLAN:16; PLAN Next 3. The LiLa-WAM numbers are verified in LP from arXiv HTML; the reframing is judgement.

### (f) The motion-alignment score (PLAN Next 4)

Call: not enough evidence.
Reasoning (speculative): two independent searches found no label-free score of whether a frozen encoder's features change where the scene moves, so the spatial version looks new. But it is a score per encoder, while the main question is about pairs, and our only pair answers are the 7 pairs in the paper's Table 10. Tsagkas et al.'s whole-frame statistic is the required baseline, and that paper's caption and text disagree about ResNets, which the user should check when reading. The design has not started, so there is little to judge besides priority.
Would most change the call: whether a per-encoder score can first be checked against the paper's single-encoder results (Table 1 has more encoders than our four pairs) and whether it yields any pair rule.
Rests on: LP section 3(e); the motion-score prior-art file; PLAN Next 4. Search results are verified at the depth LP states (abstract level for most papers).

### (g) The main research question and publishability

Call: keep going (conditional; there is no positive evidence yet).
Reasoning (speculative): the exact question (a cheap conditional measure on frozen robot-data features that predicts which encoder pairs help a world action model) was not found in prior work, but the closest work (KAGES, Ding 2026, Wang ICLR 2026) already covers parts of it, and our conditional gain is close to conditional probing from 2021. So the defensible novelty is the combination: frozen encoders, a robot world model, predicting harm, checked against closed-loop success with seed replicates. Today the finished tests neither support nor kill the idea. If the proxy reproduces the drop and some cheap measure flags it, a CoRL or RSS paper is plausible, more so with more than 4 to 7 pairs. If the proxy does not reproduce it, the likely output is a careful replication note, workshop level on its own. The window is open but narrowing, since related papers keep appearing.
Would most change the call: whether the DINOv3 versus DINOv3 plus C-RADIOv4 gap survives training-seed noise in the proxy.
Rests on: LP Summary items 1 and 5 and sections 1 and 5; FE Summary item 8. Novelty claims are verified at abstract depth for most neighbours (LP says so); publishability is judgement.

## Decisions the user needs to make

Numbered, most urgent first. "Recommendation" is what the report authors suggested.

1. Confirm the seed-noise threshold and floor in words, before the verdict is read (time-critical). Source: PR fix 3, PR section 6. The rule was locked on the user's silence: the user said to stay faithful to the paper, the lead said it would lock the threshold at 2.7 points (half of the paper's drop) and the floor (either seed's LIBERO-Plus success below 2.7 points means "can't tell"), and treated silence as agreement. Locking a threshold is an evaluation-protocol change, which is a verification gate in `_standards/05`. The verdict was due within about a day of 2026-11-08 and decides whether the other 13 runs go ahead.
   - Option A: confirm both numbers as locked. Option B: change one, recorded as an amendment before looking at the result. Option C: ask for the threshold in other words (for example, "what difference between two identical-recipe copies would make you stop?").
   - Recommendation (PR): ask now, record the answer as an amendment, and only then run `lp_eval.py seednoise`. Do not read the verdict first.
2. Whether the same-seed initialisation mismatch matters. Source: PR 3.7, PR fix 7. Configs at one seed share data order but probably not initial weights or training noise, because the first layer resizes with the number of channels. The PLAN note that same-seed comparisons are "probably less noisy" is therefore not supported. This is speculative in PR (the fastwam module code was not read).
   - Option A: leave the design, soften the PLAN sentence, and treat the seed-noise number as the estimate. Option B: seed initialisation per module name so shared weights match across configs (changes the design for the 13 remaining runs). Option C: measure it with one extra run.
   - Recommendation (PR): option A unless the user wants to change the design; do not change before the 13 runs without a decision.
3. Key findings headline rewording. Source: FE fix list 1, 2, 3, 7 (high). The signed-off headline "The simplest form of the idea fails the cheap check" is stronger than the tests can support: the input measure could hardly go negative and the target measure mostly follows feature scale. The same section says "Neither frozen-feature measure singles out the harmful D+R" while gain(D|B) does put C-RADIOv4 last (only because it restates that encoder's solo score), and "R adds the most" sits inside overlapping confidence intervals with SAM 3.
   - Option A: reword to what was tested ("the two cheap linear measures, as built, do not single out D+R, and could hardly have done so") and add both limits. Option B: keep the headline and add the limits beneath. Option C: leave it and rely on the experiment READMEs.
   - Recommendation (FE): option A. Status on main: open.
4. Whether PLAN Next 3 is reframed around LiLa-WAM and ReWAM. Source: LP section 3(d) and fix list. Option A: reframe to "layers times pairs" and cite both as starting leads. Option B: keep it as an open layer study. Option C: drop it for now and spend the effort on the perturbed-frame test. Recommendation (LP): option A; the user should read both papers first, since the user has read neither.
5. Whether the early touch of the stove test demos needs recording beyond the LOG. Source: PF Spec 1. An early diagnosis script loaded all 50 stove demos' image arrays in two runs (and began encoding them in a third) before the loader was changed to read only train and val. No number came from it and the early script version was never committed, so the LOG is the only record. `_standards/07` says to log every test-set use.
   - Option A: nothing more, the LOG entry stands. Option B: add one line to the experiment README (what, when, no result used). Option C: also add a correction line under the LOG entry.
   - Recommendation (PF): low severity; option B is cheap and makes the record easy to find. Do not edit the old LOG line.
6. Redo the code review of the perturbed-frame analysis against main's rewritten `pf_analysis.py`? Source: PF Part B8 and its Next step. Option A: yes, same checks, about the same effort. Option B: only the checks that touch the final run (a fixed `primary` marker, pairing, leakage). Recommendation (PF): at least B before the final run.
7. `backoffLimit: 0` in the perturbed-frames Jobs against `_standards/08` line 31. Source: PF Standards 4. Option A: set it to 1 or more and make the scripts safe to rerun. Option B: write the exception into STANDARDS with the reason (a failed check is a result). Recommendation (PF): either, but not silent; it also applies to the new phase-A Jobs.
8. Read the NRP Violations page after maintenance ends and log it. Source: PR 5, PF Standards 8, FE section 5. It was never checked for any Job in this direction, and the account has had one idle-GPU warning. It needs the user's login. Recommendation: do it once and log the result.

## Cross-cutting patterns

LOG lines edited in place. FE found 19 commits that delete lines in the direction LOG, mostly same-session review fixes, none in the finished experiments' entries. PF found specific cases: direction LOG commits 28495a3, f85e5c0, 60be280, 8107099 and others, and in the perturbed-frames LOG c30d6de and e67cdaf, which changed counts with no correction line (`:219` lists neither change). The one worth noting is 60be280, which rewrote what the user's approval covered, though the earlier wording was never acted on. PR found the right pattern in the proxy LOG, where (o) corrects (n) with a dated correction. Rule (`04`): append a correction, never edit the old line.

backoffLimit 0. PF raised it for the three perturbed-frames Jobs (smoke, diag, val), whose headers say "Follows _standards/08" while `08` line 31 wants a value above 0; the new phase-A Jobs on main also use 0 (checked by grep). FE found the older Jobs compliant (`backoffLimit` 1). Either follow the rule or record the exception.

Missing estimated-versus-actual compute. PR found no per-Job table or running total for the proxy round (the actual evaluation cost runs 9 to 18 percent above the PLAN estimate). PF found the same for smoke, diag and pf-val. FE found utilisation not recorded for several older Jobs. The standard (`STANDARDS.md:49`) asks for it per run.

Stale status in README and PLAN. The direction README, direction PLAN and proxy README all still say 7 of 16 shards while the experiment LOG is at 12 of 16 (confirmed on main by grep). `PLAN.md:11` was marked superseded but its old text remains. `04` wants the status in one or two sentences and the PLAN free of results. `_standards/08:6` still calls an already merged branch "not yet merged". RUN.md still treats 40 percent over 10 minutes as a hard stop though the portal averages over hours.

Test files pytest cannot collect. FE: `test_diagnostics.py` and `test_target_diagnostics.py` print `all ok` as scripts but pytest reports "no tests ran". PR: `test_lp_eval.py` collects 1 of 10 checks and `test_proxy_policy.py` none, and `test_proxy_wam.py` and `test_rollout.py` need `fastwam` installed. PF: two tests read result files by a relative path. A quick pytest glance can mistake "nothing ran" for "passed".

Baselines required by STANDARDS never computed. FE: B's solo score, the objective-overlap label and a random ranking were not computed for the tally, and no control partner was run. The same baselines must exist before the proxy data is seen (FE fix 6). LP adds a baseline now needed for (f): Tsagkas et al.'s whole-frame statistic.

Unlabeled claims. FE: the target README Observations and some tally lines carry no verified or speculative label. PF: the experiment README's old status line stated a cause with no label (now fixed on main). Smaller: Key findings "the sanity checks match the pilot exactly" covers two encoders only.

## Fix list by owner

Severity: high = a reported finding could be wrong or its headline rests on a measure that could hardly show the effect; medium = narrower or weaker than stated, or a record gap; low = clarity or housekeeping. Source tags give the report and its item. "Status on main 43c3b7d" is open, fixed, partly fixed, or not rechecked; for items only PF checked, PF's status is copied; for the others I used a single grep on main where it took one step.

### Implementation session

| # | file:line | what is wrong | standard or PLAN line | suggested fix | severity | status on main 43c3b7d |
|---|---|---|---|---|---|---|
| I1 | `experiments/2026-11-06-perturbed-frames/LOG.md:155`, `:213`, `:220` [PF 2] | counts changed in place (37 to 36, 5 to 4, 90 to 95 demos, three to two tasks); `:219` lists neither | `04`, `06` | add correction lines naming each change | medium | not rechecked |
| I2 | `src/pf_render.py:133-140`, `:187-207` [PF 5] | fixture check reads back the arrays it just wrote (`body_pos`, `body_quat`), never world-frame `body_xpos`, `body_xquat` | experiment PLAN Handoff step 2 | compare world-frame values after `forward()` | medium | not rechecked (file changed on main) |
| I3 | `src/pf_smoke.py:34`, `:227-237` [PF 6] | CAM_TOL 1e-4 decides "differs"; a 5e-5 difference skips the user's camera override | experiment PLAN Q2 | ask for a tolerance or apply the override whenever the pose is saved; write an amendment | medium | open (CAM_TOL still at `pf_smoke.py:34`) |
| I4 | `jobs/job-pf-diag.yaml:5-10` [PF 8] | third cluster Job not named in the phase-1 approval | memory rule "ask before new cluster work"; `08` | state in the LOG which user decision covered it | medium | not rechecked |
| I5 | `jobs/job-pf-smoke.yaml:40`, `job-pf-diag.yaml:23`, `job-pf-val.yaml:45` and the phase-A Jobs [PF 11] | `backoffLimit: 0`; headers say "Follows _standards/08" | `08:31` | set above 0 or record the exception in each header and STANDARDS (decision 7) | medium | open (also in the `job-pfa-*` files) |
| I6 | `jobs/job-pf-val.yaml:33`, experiment PLAN `:119` [PF 10] | low GPU use expected, no recorded exception to the 40 percent rule | STANDARDS:47, `08` | record the measured use, or move CPU stages to a CPU Job | medium | not rechecked |
| I7 | experiment LOG pf-val entry (`LOG.md:193-200`) [PF 13] | no GPU or RAM measure, no Violations check, no estimate beside actual time for smoke, diag, pf-val | STANDARDS:49; `08` pre-flight 8 | add one line per Job | medium | not rechecked |
| I8 | `src/pf_validate.py:229-243`, diagnosis README `:26-37` [PF 23] | gate numbers, alpha sweep, per-camera and oracle numbers without demo-level CIs | STANDARDS:29 | add the bootstrap or label "point estimate, no CI" | medium | not rechecked |
| I9 | diagnosis README `:185` and `:92`, `LOG.md:190-221`, `src/pf_val_diag_all_features.py` [PF 21, 22] | commands and code commit absent; unpinned scratch venv; personal `/private/tmp/...` paths; no provenance written | `03`, `04` | add commands and commit, `pip freeze`, replace paths, write a run-info JSON | medium | not rechecked |
| I10 | `src/pf_analysis.py` `_run` output [PF 30] | no field says which (config, view, ftype) decides; six rule-P verdicts per run | PLAN "stands out" rule, Q12 | add a fixed `primary` key before the final run | medium | open (no `primary` in `pf_analysis.py` on main) |
| I11 | `src/setup_env.sh:12`, `jobs/job-train.template.yaml` [PR 5] | `pip freeze` goes to an emptyDir and is lost; transitive versions unrecorded | `03`, `07` | `tee` the freeze into the pod log or copy it to the PVC for the remaining 13 runs | medium | not rechecked |
| I12 | proxy eval watcher exit 4 on the L40 pod [PR 11] | exit acted on about 3.5 h later; fast-card risk recorded at 04:51 but the list narrowed only after the NRP email | RUN.md stop 2; `08` | shorter poll or a push notification on exit 4 (list already narrowed to L4) | medium | not rechecked |
| I13 | proxy eval Jobs, `08` pre-flight 8 [PR 12] | no record of reading the Violations page | `08` pre-flight 8 | see decision 8; log once read | medium | open (needs the user's login) |
| I14 | `src/test_diagnostics.py:8`, `src/test_target_diagnostics.py:34` [FE 5] | `pytest` collects zero tests (exit 5); no tests for `demo_split`, `load`, train-only standardisation, noise partner | `07`, `03` | add thin `test_*` wrappers or document "run as script from `src/`"; add the missing tests | medium | open (no `test_*` function in either file) |
| I15 | tally README and STANDARDS "Methods" [FE 6] | B's solo score, overlap label, random ranking, control partner never computed | STANDARDS standard baselines; `07` | compute them before the proxy comparison is read; add a one-line note to the tally README | medium | open |
| I16 | direction `LOG.md`, entries for smoke, pf-val, diagnosis and proxy (m)-(o) [PF 24, PR 2] | no direction-level summary | `04`, `05` End step 1 | one `[agent]` entry per session; add Job commit hashes to the experiment README | medium | partly fixed (direction LOG +18 lines, experiment LOG +97 lines; completeness not checked) |
| I17 | proxy and perturbed-frames commits cd1b1e6, de0ece4, 6714dc7 and the pf commits [PF 26] | items changed without a PLAN update in the same commit | `05` End step 2 | update the PLAN in the same commit | medium | not rechecked |
| I18 | `src/proxy_wam.py:476-477`, experiment `PLAN.md:227` [PR 7] | same-seed configs share data order only; the "less noisy" note is unsupported | `01` "report uncertainty" | soften the sentence (decision 2) | medium | not rechecked |
| I19 | `src/proxy_wam.py:420-422`, experiment PLAN `:131-135` [PR 4] | per-source loss is a channel mean; not listed among choices where the paper is silent | `01`/`06`; fidelity rule | add one line to the PLAN list | low | not rechecked |
| I20 | `src/target_diagnostics.py:152-165` [FE 8] | alpha chosen per condition on that condition's loss | none | note in README, or rerun G with the reference alpha | low | not rechecked |
| I21 | `src/extract_features.py:246-249`, `:94-96`, `:256` [FE 9] | finiteness check before the fp16 cast; default SigLIP 2 variant differs from the one used; metadata records compute dtype not stored dtype | `03` | check after cast; fix default; add `stored_dtype` | low | not rechecked |
| I22 | extract and tally Job yamls (`revision="main"`) [FE 11] | LIBERO dataset version not pinned, no checksum | `03`, `07` | record HF revision and sha256 of the ten files | low | not rechecked |
| I23 | five finished experiments, `08` pre-flight and Requests [FE 12] | CPU and RAM use not recorded for several Jobs; Violations page not checked | `08` | record peak RSS and CPU from Prometheus for new Jobs | low | not rechecked |
| I24 | tally README and target README Caveats [FE 4, 10, 14] | CIs cover test demos only; CKA has no CI; 128 px upsampling of 1.75x to 7.9x not listed; target Observations unlabeled | `01`, `05` | add the caveats and labels | low to medium | not rechecked |
| I25 | `experiments/2026-10-18-extract-libero-goal/README.md:36-39` [FE 15] | C-RADIOv4 throughput and storage outside the predictions, not flagged | `01` | add one line | low | not rechecked |
| I26 | five finished experiments: `LOG.md` stubs, no `config.*` [FE 16, 17] | stubs point into the direction LOG; parameters live in Job yamls | `04`, `03` | one-line note in each README that the Job yaml is the config | low | not rechecked |
| I27 | `experiments/2026-10-19-diagnostics-libero-goal/results/run1-*.json`, `run2-*.json` [FE 18] | made by a script version with `resid` that no longer exists | `03` | tag commits `d6e3905` and `a89ece6` | low | not rechecked |
| I28 | `src/pf_analysis.py:20` (main `:29`) [PF 29] | docstring says three readings of rule B, two implemented | PLAN amendment 2026-11-08 | change to "two readings" | low | open |
| I29 | `src/pf_analysis.py:383` [PF 31] | tally tolerance 1e-6 equals the cross-machine gap | none (fails closed) | raise to about 1e-5 or log both numbers | low | mostly moot on main (check removed) |
| I30 | `jobs/job-pf-val.yaml:16`, `:34` [PF 14] | stale comments ("12 h deadline", "NOT created yet") | `04` | update | low | open |
| I31 | `jobs/job-pf-smoke.yaml:76-86` [PF 15] | pod exits 0 whenever `report.json` exists | `08` Job design | `exit $RC` as pf-val does | low | not rechecked |
| I32 | `jobs/job-pf-val.yaml:73` [PF 12] | in-pod cap 4 h exceeds the 3 h idle-warning threshold | `08`, STANDARDS:47 | cap under 3 h or log the risk (the 4 h was the user's choice, so ask) | low | not rechecked |
| I33 | `src/pf_smoke.py:373-375` [PF 7] | MAE < 1.0 gate chosen by the agent, observed 0.0 | `05` | name it as agent-chosen or drop it | low | not rechecked |
| I34 | `LOG.md:194-195`, `features-run-info.json` [PF 9] | DINOv3 ran on Mac MPS after the option "No GPU" | `05` | add one line that "no GPU" meant no cluster GPU | low | not rechecked |
| I35 | diagnosis README `:13`, `:88`, `stored-originals.sha256:1`; `LOG.md:199`, `:201`, `:214` [PF 16-20] | five wrong or loose statements: "nine files have no Job-side hash" (all ten match), 12,416 vs 12,316 frames, "3 steps later" understates, 7.6 percent includes the bowl box, "all ten tasks" for position shift | `05` | correction lines for the LOG; direct fix for README results | low | not rechecked |
| I36 | proxy `RUN.md` step 12, stops 2 and 10; experiment `PLAN.md:34` vs `:166`, `:179`; manifests s0-l4, s1-l4 [PR 6, 9, 13, 14] | glob matches `.errors.jsonl`; 40 percent stop out of date; PLAN:34 says 50 per task and is not marked superseded; live parallelism differs from committed manifests | `03`, `08` | change glob to `*of8.jsonl`; update after the NRP question; add a superseded note; note live values | low | not rechecked |
| I37 | proxy cost and hardware records [PR 10, 16]; `src/test_lp_eval.py:59`, `:313`, `src/test_proxy_policy.py` [PR 15] | no per-Job estimate-vs-actual table; rollout rows carry no GPU model; pytest collects 1 of 10 checks | `STANDARDS.md:49`, `07` | add the table at round end; add `gpu` to rows for the remaining runs (the user decides); name script commands in RUN.md | low | not rechecked |
| I38 | perturbed-frames `src/pf_val_diag_*.py`, `pf_validate.py`, `pf_smoke.py` [PF 32] | smells: two Laplacian energies, hard-coded 30 and 40 split, duplicated standardise and bootstrap, test-script imports, dead code, literal report values | `07` (soft) | do not copy these patterns into the main render; fix the Laplacian wording | low | partly open (`TALLY_N_FRAMES` still unused on main) |
| I39 | `test_pf_render.py:38`, `test_pf_analysis.py` (main) [PF B9] | tests read `experiments/...` by a relative path, not self-contained | `07` | make paths relative to the file or skip when missing | low | not rechecked |
| I40 | commit `cbb8fe7` [PF Standards 9] | deleted a result PNG with no recorded OK | `05` gate | note it in the LOG (harm nil) | low | not rechecked |
| I41 | early stove diagnosis script [PF Spec 1] | loaded test-demo images; process-rule breach, no number used | `07` line 6 | see decision 5 | low | open |

### Planning session

| # | file:line | what is wrong | standard or PLAN line | suggested fix | severity | status on main 43c3b7d |
|---|---|---|---|---|---|---|
| P1 | `README.md:15-17` (Key findings headline and gain bullet) and tally README `:103-107` [FE 1, 2, 3, 7] | headline rests on measures that could hardly show the effect (input) or mostly track feature scale (target); "R adds the most" inside overlapping CIs; gain(D given B) omitted; "sanity checks match the pilot exactly" covers two encoders | `01`, `06`, STANDARDS Methods | reword (decision 3); name both limits and the solo-score link; state both gain directions | high | open (headline unchanged at `README.md:15`) |
| P2 | `README.md:19`, target README `:55-57` [FE 3] | "ordering doesn't track the paper" is a judgement; interference follows B's loss share and flattens under equalised scaling | `01`, `05` | replace with the measured fact (n = 4, rank correlations +0.6 and 0) and the scale note | high (merged with P1) | not rechecked |
| P3 | `README.md:28` [FE 13] | "30-50 CPU-min per pair" undercounts; the 49-minute Job covered two pairs; per pair it is about 46 to 90 CPU-minutes | `05` | rewrite with the JSON `seconds` figures | low | open |
| P4 | `README.md:3`, `PLAN.md:8`, `experiments/2026-10-27-proxy-calibration/README.md:5` [PF 25, PR 1] | "7 of 16 shards" while the experiment LOG is at 12 of 16; status paragraph far longer than one or two sentences; counts and ETA in PLAN | `04`, `05` End steps 2, 3 | two-sentence status with a pointer to the latest LOG entry; keep counts out of PLAN | medium | partly fixed (`PLAN.md:11` superseded but old text remains; "7 of 16" still in all three) |
| P5 | direction `LOG.md`, commit 60be280 (2026-11-04 env-build2 Next line) [PF 1] | approval-scope sentence rewritten in place | `04`, `06` | add a dated correction quoting the original; do not edit again | medium | not rechecked |
| P6 | direction `LOG.md`, commits 28495a3, f85e5c0, 8107099, 1519499 and others [PF 3] | other in-place edits of older entries | `04` | use correction lines from now on; no need to undo history | medium | not rechecked |
| P7 | `PLAN.md:16` (Next 3, layers) [LP] | names no prior work; LiLa-WAM and ReWAM already show intermediate DINOv3 blocks help | `01` survey first | add both as starting leads; reframe toward layers times pairs (decision 4) | medium | open (neither named in PLAN) |
| P8 | `writing/hypotheses-draft.md:55-61` [LP] | novelty statement compares only with agostinelli2022; out of sync with the 2026-10-27 proposal; "combining can hurt" and "no label space" do not separate us from KAGES | `01` | sync with `p1-short.md` related works or mark superseded; add conditional probing | medium | open (no KAGES in the draft) |
| P9 | experiment PLAN 2026-11-08 amendment, "Lead's readings" [PF C5] | alpha-tuning rule, Q12 as separate probes, drift full-size only are evaluation-protocol readings the user did not explicitly OK | `05` gate | ask the user to confirm in words | medium | not rechecked |
| P10 | `_standards/08-compute-nautilus.md:6` [PR 8] | says the NRP-code findings are on an unmerged branch; they were merged | `08` accuracy | update the sentence | low | open |
| P11 | `PLAN.md:26-32` "Recently done" [PF 27] | no line for the smoke test, validation failure or diagnosis | `04` | add one line each pointing to the experiment LOG | low | not rechecked |
| P12 | `README.md` Key files [PF 28] | missing perturbed-frames folder, `src/pf_*.py`, `lp_eval.py`, the prior-art file, `encoder-weights.md`, proxy plan HTML, `reviews/` | `04` | add one line each | low | not rechecked |
| P13 | `STANDARDS.md:15` [LP] | calls agostinelli2022 the closest prior work for combination choice; KAGES is closer for frozen-feature concatenation; no bib entry | `02` | list KAGES once the user has read it | low | not rechecked |
| P14 | `decisions/` [PF C8] | full benchmark, locked seed-noise rule, perturbed-frames design and unwhitened PCA live only in PLAN amendments and LOG | `04` | judge whether ADRs are needed | low | not rechecked |
| P15 | `writing/2026-11-08-motion-score-prior-art.md:32` [LP] | Tsagkas entry quotes only the figure caption; Sec. IV-B says the correlation is strong within ResNets | `02` contradicting evidence | add the sentence; the user checks when reading | low | not rechecked |

### Literature session

| # | file:line | what is wrong | standard or PLAN line | suggested fix | severity | status on main 43c3b7d |
|---|---|---|---|---|---|---|
| L1 | `literature/references.bib:21-29` (fei2025liberoplus) [LP] | published at CVPR 2026 under a different title; note says preprint | `02`: cite the published version | add the CVPR 2026 entry, keep the arXiv id, note the title change | medium | open (note still says preprint) |
| L2 | `literature/references.bib:11-19` (liu2023libero) [LP] | published at NeurIPS 2023 Datasets and Benchmarks | `02` | change to inproceedings, keep the arXiv id | low | open |
| L3 | `literature/reading-list/2026-11-07-measures.md`, `-fusion.md` [LP] | KAGES, Ding 2026, Wang ICLR 2026, Tang 2026, CUES and Hurt 2026 absent, although the first four are in the user's proposal | `02` | suggestion for this session, passed via the user | low | not rechecked |
| L4 | `literature/notes/agostinelli2022-transferability-ensembles.md:7-8`, `:21` [LP] | "15 candidates" and output averaging describe Sec. 4 only; "not yet checked" list still open | `02` faithful paraphrase | add the Sec. 5 setup; point to the reading list | low | not rechecked |
| L5 | `literature/notes/liu2026-what-visual-representations-wams.md:20`, `:4` [LP] | "SAM 3 uses a Perception Encoder backbone" rests on a snippet; "numbers still need a check" is stale | `02`, `04` | mark recalled; replace the stale line with a pointer | low | not rechecked |

## What this review did not check

- Live cluster state: no kubectl, no Prometheus, no NRP Violations page. Whether the seed-noise verdict has been produced, and how the last proxy shards ended, is not known to this synthesis.
- Result files on the PVCs: the real rollout JSONLs and cached features were not opened (the user's rule is that no verdict is seen before all 16 shards are in). PR checked code and LOG only; FE read the result JSONs that are in the repo.
- Fast-WAM source: not in the repo and not installed. `test_proxy_wam.py` and `test_rollout.py` could not be run by the PR reviewer; their passes are recorded in the LOG only. That the proxy follows Fast-WAM's own training and evaluation path rests on docstrings and the LOG (recalled).
- `extract_fastwam.py`, `extract_features.py` as used by the proxy rollout, and `bench_fastwam_step.py`: not read by the reviewers who would have needed them.
- Papers: most were read only at abstract depth, a few at targeted sections of arXiv HTML. Semantic Scholar's citation API was blocked (HTTP 429), so forward citations of liu2026what rest on web search only. LP's claims about SAM 3's backbone, the LIBERO-Plus per-factor findings, and Tsagkas's passage beyond Fig. 4 and Sec. IV-B are unverified. The user has read none of the leads named in LP.
- The reviewed perturbed-frames analysis code is the superseded version; main's rewritten 1,063-line file got only a supplementary test run and spot greps, not a review (PF B8).
- Whether any Violations strike exists, whether NRP was asked the open "keep running" question (PR section 6), and whether the NRP portal build matches the public source.
- Fix-list rows marked "not rechecked" were not looked at on main by this synthesis. This synthesis reran none of the reports' scripts; its only checks were greps on main at `43c3b7d` for the statuses noted.
