# Writing plan for the stacking-encoders paper (2026-11-10)

Written by the draft session (`vis-draft-agent`) for the user, from main at 744f288. It decides nothing about experiments, findings or venue; the user and the mentor decide those. Inputs: the planning session's handoff message to this session (2026-11-10, relayed ideas, not approvals), `writing/2026-11-09-path-to-paper.md` ("path memo"), `decisions/004-paper-framing-case-study.md`, `README.md`, `PLAN.md`, `STANDARDS.md`. Labels: `verified` = read in a repo file (pointer given) or a fetched primary page (named), `recalled` = from memory, `speculative` = judgment. Paths are relative to `world-model-stacking-encoders/`. "Proxy PLAN" = `experiments/2026-10-27-proxy-calibration/PLAN.md`; "perturbed PLAN" = `experiments/2026-11-06-perturbed-frames/PLAN.md`; "control PLAN" = `experiments/2026-11-09-control-partners/PLAN.md`. Milestones M1 to M9 and endings (a), (b), (c) are the path memo's (sections 1 and 2).

## The short version

The paper is a case study of one reported result: does the harm from stacking DINOv3 with C-RADIOv4 survive training seeds, and does it leave any trace in frozen features (`decisions/004`, verified). Some of the paper reads the same whatever the experiments say: the introduction's setting, the setup, the clean-frame measurements, the blind baselines, and the limits. That part can be drafted now. What the clean-frame result means waits on the control-partners run, and the perturbed frames, the proxy seeds and the headline wait on their own results. So the plan is one outline with a slot for each ending, the fixed part drafted first, and the results and discussion written when the answers arrive (`speculative`, the planning session's suggestion, adopted here). The slowest part is probably not the experiments but the citations: most related work, and every encoder we run except C-RADIOv4, still needs to be read before it can be cited (section 4).

Two gates come before the title and contributions are fixed (section 5): the mentor has not yet been told about decision 004's framing (`decisions/004:14`, verified), and pulling PLAN Next 5 (a pair-harm measure) into this paper is not yet reconciled with decision 004, which names the predictor as a follow-up; its dated note is a PR still to open (LOG 2026-11-10 [agent, lead, planning], Next 3, verified).

## 1. Venue and format (open: user and mentor)

- Target: a CVPR workshop (user, 2026-11-10; the mentor's goal; LOG 2026-11-10 [agent, lead, planning]). Which workshop, its deadline and its page limit are not in the repo (verified: no file records them).
- The CVPR 2027 workshop list does not exist yet: workshop proposals closed 2026-11-04 and proposal decisions are due 2026-12-22; workshops run 2027-06-20 and 21 in Seattle (verified, CVPR 2027 Call for Workshops page, https://cvpr.thecvf.com/Conferences/2027/CallForWorkshops, fetched 2026-11-10 by the PR #12 reviewer; this session's search summary of the Dates page agrees). So the 2026-11-11 mentor meeting can settle the kind of venue (length, archival or not), not a specific workshop.
- Archival workshop papers follow the CVPR 2027 style and length rules; accepted-paper metadata is due 2027-03-29 and camera-ready 2027-04-23, and every author needs a complete OpenReview profile (verified by the PR #12 reviewer on the CVPR 2027 Call for Workshops, 2026-11-10). Paper deadlines are set per workshop; non-archival tracks in 2026 ran into April and May (from a search summary only, not verified).
- Length decides prior publication for CVPR, not archival status: a peer-reviewed workshop paper longer than four pages, excluding references, counts as a publication whether or not it is in proceedings (verified by the PR #12 reviewer on the CVPR 2026 Author Guidelines, https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines, 2026-11-10; the 2027 guidelines were not checked). So a paper over four pages could block reusing the same material in a later CVPR paper on a pair-harm measure (the contribution the user wants, PLAN Next 5); a paper of four pages or fewer would not. Other venues have their own rules. This is a choice for the user and the mentor.
- The kit's main-conference limit is eight pages excluding references, and acknowledgments are left for the final copy (verified, `sec/1_intro.tex` lines 18 and 141 to 142 of the CVPR author kit, tag `CVPR2027-v1`). CVPR's double-blind rule says acknowledgments must not identify the authors (grant numbers are its example; PR #12 reviewer, 2026 guidelines). The NRP text names UC San Diego and NSF awards, so it goes in at camera-ready only.
- AI use: the CVPR 2026 main-conference guidelines allow any tools, with authors responsible for the content; citations to non-existent work can lead to rejection (PR #12 reviewer, 2026 guidelines). The chosen workshop's own rule still needs checking (`../_standards/05`, "For the human").
- Authorship and author order: not recorded; the user and the mentor decide.
- Anonymization: the repo is public, so the review version must not link to it.
- Where the draft lives: decided, section 6.

## 2. Outline, with a slot per ending

Each section lists what goes in, the files it is written from, and its state: **ready** (can be drafted now), **partial**, or **waits on Mx**. Page shares are for an 8-page paper (`speculative`); for a 4-page version, sections 3 and 4 shrink to one paragraph each and the supplement carries the rest.

**Title.** Set last, after the framing gate. Working title (`speculative`): "Does stacking DINOv3 with C-RADIOv4 hurt a world action model? A multi-seed case study."

**Abstract.** Written last, one sentence per section. Waits on all.

**1. Introduction (about 1 page). Partial.**
- In liu2026what's world action model (WAM), frozen vision encoders are both inputs and future-feature targets (paper note, Sec. 3.3 and App. B.2; do not generalize this to every WAM); liu2026what stacks encoders with DINOv3. Against DINOv3 alone in Table 1 (84.0; Table 10 has no DINOv3-alone row), most of the seven DINOv3 pairs in Table 10 change success a little, and DINOv3 + C-RADIOv4 is the lowest (our arithmetic on `literature/liu2026what-table1-single-source.csv` and `literature/liu2026what-table10-composition.csv`, verified). Each row is one trained checkpoint; its three seeds are evaluation seeds (`STANDARDS.md:19`, verified).
- Why it matters: if the harm is real, a cheap check before training would save the cost of training bad stacks; if it is seed noise, single-checkpoint composition tables can mislead the choice of encoders (`speculative`).
- The question, from `README.md:5`.
- What we did: clean-frame frozen-feature tests (input and target side), a perturbed-frame test, and a multi-seed proxy replication. The target-side test, the perturbed-frame test and the proxy each have a decision rule fixed before the data it decides was read; the input-side tally fixed its design in advance but had no decision rule ("No hypothesis is chosen or argued here", tally README:3). Exceptions are listed in the supplement (do-not-repeat list). Say "tests", not "probes": "probe" means a linear probe everywhere else in the paper.
- Contributions, slot per ending, after the framing gate: the clean-frame measurements and what the controls say they can detect (wording waits on M2 and M6); the perturbed-frame result (waits on M1); the multi-seed answer (waits on M3, M4); the pair-harm measure across Table 10 (slot, PLAN Next 5, not designed).

**2. Related work (about 0.75 page). Structure ready; text waits on reading (section 4).**
- Visual representations for robot policies and WAMs: liu2026what (in bib); Fast-WAM, our proxy's code base; Yardi et al. 2025 and Chun et al. 2026 (unread leads, `decisions/004:15`).
- Combining foundation models: C-RADIOv4 (`ranzinger2026cradiov4`, read) and Guo et al. 2026 (`guo2026metric`, read); KAGES (Liu and Jiang 2026, arXiv 2608.17490), in the user's own proposal as showing "more encoders can hurt" (`writing/proposals-2026-10-27/p1-short.md:9`); Ding 2026 and Wang ICLR 2026 (`reviews/2026-11-09-synthesis.md`, (g)).
- Choosing models or combinations without training: conditional probing (Hewitt et al. 2021; reported read by the user per the planning lead's recollection, unconfirmed: LOG 2026-11-09 [agent, slides] "Not verified"; bib PR waiting on the user in the literature session); CKA (Kornblith et al. 2019, not in bib, `STANDARDS.md:16`); agostinelli2022 (note exists, not read, `STANDARDS.md:15`). Dong et al. 2026, a cheap frozen-encoder probe that predicts robot policy performance (reading-list top 10, item 5). Scoring combinations by training them: Law of Vision Representation (arXiv 2408.16357) and Theia (arXiv 2407.20179) (LOG 2026-11-09 [agent, literature], Q1).
- Seed noise and replication: Hu et al. 2023 and Zeng et al. 2026 (leads, same LOG entry).
- Robustness under perturbation: LIBERO-Plus (`fei2025liberoplus`, in bib); Burns et al. 2023 (reading-list top 10, item 9); arXiv 2606.07687 (lead).
- The novelty sentence comes from the literature session's check (no multi-seed replication of an encoder-composition table found, no robot or WAM work scoring encoder pairs before training; both `speculative` absence claims, LOG 2026-11-09 [agent, literature]). Phrase it narrowly ("we are not aware of"), and have the literature session rerun the check before submission, since related papers keep appearing. These neighbours matter most if section 7 goes in.

**3. Setup (about 1 page). Mostly ready; the proxy's final depth waits on M3; licenses are not ready.**
- The source paper: its WAM, its encoders, Tables 1 and 10 (`literature/` CSVs, the paper note). The paper note says its numbers "still need a check against the PDF before they go into a writeup" (`literature/notes/liu2026-what-visual-representations-wams.md:4`, verified); do that check before this section is final.
- Our encoders: DINOv3, C-RADIOv4, SigLIP 2, SAM 3, VGGT-Omega; checkpoints (`literature/encoder-weights.md`). Spell out names; one-letter codes in tables only.
- Data: LIBERO-Goal demos for the frozen-feature tests (decision 002), the four LIBERO suites for proxy training, LIBERO-Plus for evaluation (proxy PLAN "Step 1 findings"). Licenses: the LIBERO data license is "not recorded" and LIBERO-Plus's is treated as "unstated" (`data/README.md:14`, `:25`, verified); resolve before submission.
- Our proxy and how it differs from the paper: built on Fast-WAM, Wan 2.2 VAE latents, T5 text, flow matching, no pretrained video prior, smaller, per-source z-scoring, temporal averaging of future features, resizing every partner's grid to the 14 x 14 VAE grid, and 16-episode buffers instead of a full shuffle, where the paper is silent (proxy PLAN:8, :83 to :85, :131 to :135; `PLAN.md` Next 2, suspect 3; `writing/figures/2026-11-07-combining-grids.png` shows the stacking step). Configs at one seed share data order but probably not initial weights (`reviews/2026-11-09-synthesis.md`, decision 2, open), so the paper must not describe the pairing as a shared initialization. Model depth after Amendment (c) waits on M3.
- Pre-registration: each rule was fixed before the data it decides was read; the exceptions go in the supplement trail (do-not-repeat list below). Whether this counts as a strength in the paper's framing is `speculative`; state it plainly and let the reader judge.

**4. Clean-frame tests (about 1.25 pages). Measurements ready; the reading waits on M2, the headline on M6.**
- Input side: gain(B|DINOv3) with demo-level bootstrap intervals, and CKA without intervals (none were computed, tally README:79) (`experiments/2026-10-19-feasibility-tally/README.md`). The tally's DINOv3 features are fp32 and the partners' fp16, a deviation from its pre-registration (tally README:107). Target side: the interference test (`experiments/2026-10-21-target-diagnostics/README.md`).
- The blind baselines next to the measures: single-encoder score, the paper's overlap reasoning coded as a label, random ranking (`writing/2026-11-09-blind-baselines.md`). They were computed after the tally, and the label is not blind to the paper's numbers (that memo, lines 5 to 7, verified); say both.
- The renderer caveat on absolute probe numbers (`README.md:28`).
- Slot, control partners (waits on M2). The control PLAN's decision rule sets each test's outcome (control PLAN:79 to 91); what the outcomes mean for the finding is the lead's proposed mapping, explicitly "not made here" (control PLAN:95, verified): floor near zero and size-explained: "the cheap checks cannot flag a bad partner", "no evidence either way about D + R"; can-go-negative and partner-specific: "the finding stands and is stronger, and the controls give a floor to report next to each real number"; mixed: "report each test separately". A further input-side outcome, a capacity floor above zero, re-reads every real partner as gain minus the matched-width noise gain (control PLAN:81, verified), which would change this section's gain numbers and table T1. The README headline the user signs off (M6) sets the wording; the paper says no more than that.

**5. Perturbed frames (about 1 page). Method ready; result waits on M1.**
- Design: four picture types, action probe trained on our own clean re-renders and tested on perturbed ones, feature drift as a sanity check (perturbed PLAN, Amendments 2026-11-09 and 2026-11-09 (b)). Report the redesign plainly: the first validation check failed its gate, the probes were retrained on re-renders with a new gate, and that gate passed (`PLAN.md:14`; perturbed LOG "gate A PASSED"); the render tolerance was set after the phase-A val comparison; no test-set result was seen in either step (perturbed LOG, "pf-tolerance" entry; the early stove test-demo load, perturbed LOG:204, is disclosed in the supplement).
- Rule P decides (perturbed PLAN:96), read only on the deciding slice: "full-size features, global pooling, combined view (both cameras)" (perturbed PLAN:526, Amendment 2026-11-09 (c)); the other eleven slices are robustness checks with no decision weight (:527); rule B is "reported alongside, not deciding" (:97); a partner whose re-measured clean gain differs from the tally is flagged (:528). With four partners an ordering match is a hint, not evidence (perturbed PLAN:100). The supplement lists every perturbed-frames amendment (2026-11-08, 2026-11-08 (b), 2026-11-09, (b), (c), (d)).
- Slot: C-RADIOv4 stands out under rule P / does not.

**6. Multi-seed proxy replication (about 1.5 pages). Method ready; results wait on M3, M4, M5.**
- Rules: calibration, the seed-noise rule (proxy PLAN:215, "Seed-noise rule (locked 2026-11-06)"), the level target (proxy PLAN:385), the outcome rule (proxy PLAN:155 to 162).
- Reporting duties already written in the proxy PLAN: every comparison shows the three per-seed deltas and their average, and episode intervals are never treated as evidence about training-seed variability (:152); overlap or feature interference "stays a hypothesis in any writeup, not the explanation" (:162); one sentence on how far below the paper's DINOv3 and VAE-only scores the proxy sits, and that the thresholds were set for the paper's range (:336, 6B); the relative reading 6A as a secondary, non-deciding number (:335).
- What happened so far, as a record: the first seed pair passed the noise rule but scored far below the paper, so the training was amended before the other runs (Amendment (c), PRs #3 and #5). Amendment (c) and 6A were fixed after the first full-length DINOv3 means were read (proxy PLAN:338 calls them "pilot-scale", a term the paper should avoid because "pilot" there means the 500-step sizing run, :148; "flagged departure"); disclose it.
- Slot (a), only if calibration passes (:159): "Substantial proxy effect" if every seed drops and the average drop is at least 2.7 points, each seed shown with discordant-episode counts; the PLAN's words are "a trigger for the higher-fidelity multi-seed WAM replication, not proof that the effect is statistically established" (proxy PLAN:160, verified).
- Slot (b): "the proxy does not show a substantial effect", which "does **not** establish that the paper's result was training noise: the harm may need the pretrained video prior or the full-size model" (:161, verified); the 3-seed design has no detection-power argument (:337), and the seed-power rerun (M5) sets the wording.
- Slot (c), the proxy cannot tell: say which stop fired. If the seed-noise re-check says "rethink seeds" or a seed falls below the floor: the rule and the user's choice then (proxy PLAN:215, "Seed-noise rule (locked 2026-11-06)"). If calibration fails: "don't interpret the D+R comparison from this proxy", "not 'D+R is fine' and not 'the proxy is useless'" (:158, verified). If the level target is missed: what the user chose among the pre-written options (proxy PLAN:330).

**7. Slot: a pair-harm measure across Table 10 (PLAN Next 5).** Not designed, and waits on the framing gate. If it runs and gives a result before the deadline, it becomes a section here (about 1 page, taken from sections 4 and 5); if not, one sentence in the discussion as future work. Its limits are already written in PLAN Next 5 and go in with it.

**8. Discussion (about 0.5 page). Waits on the ending.** All readings here are `speculative` until the results exist.
- (a): the proxy shows a substantial effect in the locked rule's sense; the rule's next step is the higher-fidelity multi-seed WAM replication (proxy PLAN:160). Whether frozen-feature checks are enough depends on section 5 and stays a reading, not a conclusion (proxy PLAN:8: the proxy "cannot by itself establish what happened in the paper's model").
- (b): the proxy does not show a substantial effect; what it cannot rule out (the missing video prior, the full-size model, :161). The fusion-step suspects (PLAN Next 2) are the follow-up that applies here: they run only in this branch (`PLAN.md` Next 2, verified).
- (c): what the cheap tests can and cannot say, and what it would take to answer the question.

**9. Limitations (about 0.5 page). Ready.** Four partners; one LIBERO suite for the frozen-feature tests; a small from-scratch proxy without the video prior; our own choices where the paper is silent; the source paper's one checkpoint per row; concatenated pairs differ in input width and no matched-dimension control was run in the tally (`STANDARDS.md:35`, tally README:17); the answer keys differ in kind, per-frame teleoperator actions on clean demos vs per-episode closed-loop success under perturbation (`README.md:27`), and in units, so only orderings are compared; absolute probe numbers specific to the renderer; three seeds as a screening result with no power argument. Sources: `README.md` key findings, path memo section 4, PLAN Next 5's limits, proxy PLAN:337.

**Acknowledgments. Ready, camera-ready only.** The NRP text, verbatim from `../_standards/08-compute-nautilus.md:155-156` (required in every paper that used NRP); the mentor and anyone else the user names.

**Supplement. Mostly ready.** The pre-registration trail (each rule, the date it was locked, the file) with its exceptions; amendments and why; full result tables; compute used, from the experiment LOGs; encoder checkpoints and licenses; the early load of the stove test demos before the loader was fixed (`reviews/2026-11-09-synthesis.md`, decision 5).

### Do-not-repeat list (corrections the source files still carry)

- "L, the paper's best pair": wrong. SigLIP 2 is the best of our four partners; DINOv3 + UniTok (+2.12) is the paper's best pair (`README.md:29`, verified). Still uncorrected in `experiments/2026-10-21-target-diagnostics/README.md:12`, `decisions/003-proxy-policy-calibration.md:10`, proxy PLAN:17 and its configs table, and `writing/hypotheses-draft.md:15` (verified).
- In Table 1 alone, C-RADIOv4 is the strongest single encoder of our four partners; "L > S > G > R" is the Table 10 pair ordering, not the single-encoder ordering (LOG 2026-11-09 [agent, literature], correction).
- "The ordering doesn't track the paper" (`README.md:19`) is a judgment (`reviews/2026-11-09-synthesis.md`, P2); write it as one.
- Pre-registration exceptions to disclose, not smooth over: blind baselines computed after the fact, label not blind to the paper (`writing/2026-11-09-blind-baselines.md:5-7`); the seed-noise threshold and floor locked on the user's silence (`reviews/2026-11-09-synthesis.md`, decision 1, open); Amendment (c) and 6A fixed after the first full-length DINOv3 means (proxy PLAN:338); the perturbed-frames gate margin moved from 0.01 to 0.025 after the first validation check failed, before any re-render-trained result (perturbed PLAN:466; `PLAN.md:14`), and the render tolerance was set after the phase-A val comparison (perturbed LOG "pf-tolerance"), with no test-set result seen in either (the early stove test-demo load is listed under the supplement); the control-partner target thresholds set as fractions of C-RADIOv4's already-known value (control PLAN:87-88).
- Same-seed configs probably do not share initial weights (speculative; the module code was not read; section 3 above).

## 3. Figures and tables

The figure convention in `STANDARDS.md:54` is blank. Proposed (`speculative`, user to agree, added by PR): every paper figure is made by a checked-in script in `src/` that reads result files, saves a vector PDF at the template's column width plus a PNG copy in `writing/figures/`, and uses one fixed color per encoder across all figures.

| # | What it shows | State | Source |
|---|---|---|---|
| F1 | Overview: encoders stacked as inputs and future-feature targets of the WAM, and where each of our three tests looks | new | could extend `src/plot_slide_diagrams.py` (the combining-grids schematic) |
| T1 | The paper's numbers for our four partners next to our clean-frame measures and the blind baselines | new script; CKA, target and baseline columns ready, gain column waits on M2 (capacity-floor re-read) | tally and target-diagnostics result JSONs; `src/tally_baselines.py` |
| F2 | Control partners next to the real partners, on both tests | waits on M2 | control-partners results; new script |
| F3 | Perturbed frames: each partner's score under rule P with intervals | waits on M1 | `src/pf_analysis.py` outputs; new script |
| F4 | Proxy success per config and seed, with the paper's values | waits on M4 | `src/lp_eval.py` outputs; new script; the slide schematic `writing/figures/2026-11-07-noise-check-ruler.png` is a starting point |
| A1 | Training curves | exists, slide style | `src/plot_training_curve.py`; restyle |
| A2 | Compute cost | exists, slide style | `src/plot_compute_cost.py`; restyle |

Not for the paper: the GPU-use plot (operations) and the encoder-layers diagram (no layer study was run).

## 4. Citations

Rule: cite only papers the user has read and that have an entry in `literature/references.bib` (`../_standards/02:10`; `literature/reading-list/README.md:5`). The reading list belongs to the literature session; this plan only lists what the paper needs.

- In the bib now (verified): `liu2026what`, `ranzinger2026cradiov4`, `guo2026metric` (read), and `liu2023libero`, `fei2025liberoplus`, whose entries say "not read in full" (`references.bib:18`, `:28`, verified). The user kept those two on purpose as the benchmark citations when the bib was cut (LOG 2026-11-06 [agent, docs], references.bib cut). The records conflict: LOG 2026-10-27 (g) says "the user read both" (`LOG.md:413`, verified). Question for the user: which is right, and does a dataset or benchmark citation need a full read under the user's bar? The bib stays the literature session's to change.
- Published versions (`../_standards/02`, ML-specific): LIBERO is cited by liu2026what as NeurIPS 2023 Datasets and Benchmarks, and LIBERO-Plus as CVPR 2026 under a different title; neither venue is confirmed from the proceedings yet (same LOG entry, Next). Confirm and update the entries before submission (literature session).
- Reported read, bib entry not merged: Hewitt et al. 2021 (unconfirmed, see section 2).
- Needed for the method, not read or not in the bib: Kornblith et al. 2019 (CKA); Fast-WAM (our code base); the models we run other than C-RADIOv4: DINOv3, SigLIP 2, SAM 3, VGGT-Omega, the Wan 2.2 VAE, the T5 text encoder; flow matching. A paper normally cites every model it runs (`recalled` convention). Under the user's bar each needs at least its method section read; whether a partial read meets the bar is the user's call.
- Needed for related work, unread leads: Yardi et al. 2025, Chun et al. 2026, KAGES, Ding 2026, Wang ICLR 2026, Law of Vision Representation, Theia, Dong et al. 2026, Burns et al. 2023, agostinelli2022, Hu et al. 2023, Zeng et al. 2026, arXiv 2606.07687.
- That is roughly 20 to 25 papers to read before section 2 and the setup can be cited in full (`speculative` count). This is likely the long pole.
- The bib's `note` fields hold our bookkeeping and print in the CVPR style's reference list, and the style prints no arXiv ID for `@misc` entries, while `../_standards/02` asks for a DOI, arXiv ID or stable URL on every citation (seen in the skeleton build and its review, PR #13); fix before submission (literature session's entries, or a cleaned copy for the paper).
- Page counts: the Mac builds with Tectonic (XeTeX), which needs a font setting to match the kit's Times; the final page count is checked with pdfLaTeX, as the kit's own CI uses (PR #13 review).

## 5. Order of work

1. With the mentor (2026-11-11, LOG 2026-11-09 [agent, lead, planning] path to paper, Next): decision 004's framing, whether PLAN Next 5 comes into this paper, and the kind of venue (four pages or fewer vs longer; archival or not). A specific workshop can be picked only after the CVPR 2027 list is out (decisions due 2026-12-22).
2. Skeleton in the CVPR 2027 template with every section above, each slot marked, source pointers as comments (PR #13).
3. Draft the ready parts: setup, clean-frame measurements, limitations, the method halves of sections 5 and 6, acknowledgments, supplement trail. Make F1 and T1's ready columns. The user writes the introduction (without contributions) and related work from annotated outlines as the reading allows.
4. A fresh read-only reviewer checks every sentence and number against the files and papers it cites, as in PRs #7 to #9 and #11; fixes on the same branch.
5. Fill slots as M1, M2, M3 to M5 land, using the locked rules' wording; make F2 to F4. The headline wording follows the user's M6 sign-off; results go into the paper only after the matching README finding is signed off (`../_standards/05`, verification gates).
6. Related work as the user reads (in parallel with 3 to 5); the literature session reruns the novelty check before submission.
7. Mentor read, revisions; OpenReview profiles for every author; final checks (anonymization, the workshop's AI-use rule, NRP text at camera-ready, bib notes and published versions, licenses); submission by the user.

Timing (`speculative`): the path memo's realistic date for the full result set is early December. Archival CVPR 2027 workshop papers have camera-ready on 2027-04-23, so their submission deadlines fall before that, set per workshop (verified dates above; the per-workshop deadlines are unknown until the list is out). That likely leaves room for the results sections, the mentor's read and revisions; if the chosen workshop's deadline is early, ending (c)'s smaller paper or a four-page version may be what fits.

## 6. Decisions

Decided by the user, 2026-11-10 (this session):
1. Who writes the prose: a mix by section. The draft session drafts setup, methods (sections 3 to 6), limitations and the supplement; the user writes the abstract, introduction, related work and discussion from annotated outlines; the draft session fact-checks everything.
2. Where the draft lives: LaTeX in the repo, `writing/paper/`, in the CVPR template, changes by PR; it can be synced to Overleaf for the mentor (PR #13).

Open, for the user:
3. With the mentor: framing (decision 004 and PLAN Next 5), kind of venue, author list (section 5, step 1).
4. Whether benchmark citations need a full read (section 4), and the reading for citations, through the literature session.
5. The figure convention line in `STANDARDS.md` (section 3), by PR.
