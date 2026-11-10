# Writing plan for the stacking-encoders paper (2026-11-10)

Written by the draft session (`vis-draft-agent`) for the user, from main at 744f288. It decides nothing about experiments, findings or venue; the user and the mentor decide those. Inputs: the planning session's handoff message to this session (2026-11-10, relayed ideas, not approvals), `writing/2026-11-09-path-to-paper.md` ("path memo"), `decisions/004-paper-framing-case-study.md`, `README.md`, `PLAN.md`, `STANDARDS.md`. Labels: `verified` = read in a repo file (pointer given), `recalled` = from memory, `speculative` = judgment. Paths are relative to `world-model-stacking-encoders/`. Milestones M1 to M9 and endings (a), (b), (c) are the path memo's (sections 1 and 2).

## The short version

The paper is a case study of one reported result: does the harm from stacking DINOv3 with C-RADIOv4 survive training seeds, and does it leave any trace in frozen features (`decisions/004`, verified). About half of the paper reads the same whatever the experiments say: the introduction, the setup, the clean-frame tests and their negative result, the blind baselines, and the limits. That half can be drafted now. The other half (perturbed frames, the proxy seeds, what the control partners mean for the headline) waits for results. So the plan is one outline with a slot for each ending, the fixed half drafted first, and only the results and discussion written when the answer arrives (`speculative`, the planning session's suggestion, adopted here). The slowest part is probably not the experiments but the citations: most related work and every encoder we run still needs to be read before it can be cited (section 4).

## 1. Venue and format (open: user and mentor)

- Target: a CVPR workshop (user, 2026-11-10; the mentor's goal). Which workshop, its deadline and its page limit are not in the repo (verified: no file records them).
- `recalled`, to check against the chosen workshop's call: CVPR workshops use the CVPR LaTeX author kit; many take full papers of up to 8 pages plus references, which appear in the workshop proceedings (archival); some take short papers or extended abstracts of about 4 pages, often non-archival; deadlines usually fall about February to March before a June conference; many are double-blind.
- Archival or not matters beyond this paper (`recalled` rule, `speculative` effect): an archival 8-page workshop paper usually counts as a prior publication under dual-submission rules, which could limit reusing the same material in a later main-conference paper on a pair-harm measure (the contribution the user wants, PLAN Next 5). A non-archival short paper usually does not. This is a choice for the user and the mentor.
- Authorship and author order: not recorded; the user and the mentor decide.
- Before submitting: the venue's AI-use policy (`../_standards/05`, "For the human"), anonymization if double-blind (the repo is public, so a link to it would de-anonymize), and the NRP acknowledgment (section 2, Acknowledgments).
- Where the draft lives: open, see section 6.

## 2. Outline, with a slot per ending

Each section lists what goes in, the files it is written from, and its state: **ready** (can be drafted now), **partial**, or **waits on Mx**. Page shares are for an 8-page paper (`speculative`); for a 4-page version, sections 3 and 4 shrink to one paragraph each and the appendix carries the rest.

**Title.** Set last. Working title (`speculative`): "Does stacking DINOv3 with C-RADIOv4 hurt a world action model? A multi-seed case study." Waits on the ending.

**Abstract.** Written last, one sentence per section. Waits on all.

**1. Introduction (about 1 page). Partial.**
- World action models (WAMs) take frozen vision encoders as inputs and as future-feature targets; liu2026what stacks encoders with DINOv3 and reports that most partners help a little and one, C-RADIOv4, hurts (Table 10; `literature/liu2026what-table10-composition.csv`, verified). Each row is one trained checkpoint; its three seeds are evaluation seeds (`STANDARDS.md:19`, verified).
- Why it matters: if the harm is real, a cheap check before training would save the cost of training bad stacks; if it is seed noise, single-checkpoint composition tables can mislead the choice of encoders.
- The question, from `README.md:5`.
- What we did: clean-frame frozen-feature tests (input and target side), a perturbed-frame test, and a multi-seed proxy replication, each with its rule fixed before results.
- Contributions, slot per ending: the clean-frame negative and what the controls say it can detect (ready, wording waits on M2 and M6); the perturbed-frame result (waits on M1); the multi-seed answer (waits on M3, M4); the pair-harm measure across Table 10 (slot, PLAN Next 5, not designed).

**2. Related work (about 0.75 page). Structure ready; text waits on reading (section 4).**
- Visual representations for robot policies and WAMs: liu2026what (in bib); Fast-WAM, our proxy's code base; Yardi et al. 2025 and Chun et al. 2026 (unread leads, `decisions/004:15`).
- Combining foundation models: C-RADIOv4 (`ranzinger2026cradiov4`, read) and Guo et al. 2026 (`guo2026metric`, read); multi-teacher distillation and fusion papers in the reading list (leads).
- Choosing models or combinations without training: conditional probing (Hewitt et al. 2021; read per the planning lead, bib PR waiting on the user, LOG 2026-11-09 [agent, literature]); CKA (Kornblith et al. 2019, not in bib, `STANDARDS.md:16`); agostinelli2022 (note exists, not read, `STANDARDS.md:15`).
- Seed noise and replication: Hu et al. 2023 and Zeng et al. 2026 (leads from the literature session's novelty check, LOG 2026-11-09 [agent, literature]).
- Robustness under perturbation: LIBERO-Plus (`fei2025liberoplus`, in bib); arXiv 2606.07687 (lead).
- The novelty sentence comes from the literature session's check (no multi-seed replication of an encoder-composition table found, no robot or WAM work scoring encoder pairs before training; both `speculative` absence claims, LOG 2026-11-09 [agent, literature]). Phrase it narrowly, as that entry suggests.

**3. Setup (about 1 page). Mostly ready; the proxy's final size waits on M3.**
- The source paper: its WAM, its encoders, Tables 1 and 10 (`literature/` CSVs, the paper note).
- Our encoders: DINOv3, C-RADIOv4, SigLIP 2, SAM 3, VGGT-Omega; checkpoints and licenses (`literature/encoder-weights.md`, `data/README.md`). Spell out names; the one-letter codes go in tables only.
- Data: LIBERO-Goal demos for the frozen-feature tests (decision 002), the four LIBERO suites for proxy training, LIBERO-Plus for evaluation (proxy PLAN "Step 1 findings").
- Our proxy and how it differs from the paper: built on Fast-WAM, Wan 2.2 VAE latents, T5 text, flow matching, no pretrained video prior, smaller, per-source z-scoring and grid resizing where the paper is silent (proxy PLAN "Framing" and "Choice" lines; `writing/figures/2026-11-07-combining-grids.png` shows the stacking step). Model depth after Amendment (c) waits on M3.
- Pre-registration: every decision rule was locked before its results, with dates; the full trail goes in the appendix. This is a strength of the paper and should be visible in the main text.

**4. Clean-frame tests (about 1.25 pages). Measurements ready; the reading waits on M2, the headline on M6.**
- Input side: CKA and gain(B|DINOv3) with demo-level bootstrap intervals (`experiments/2026-10-19-feasibility-tally/README.md`). Target side: the interference test (`experiments/2026-10-21-target-diagnostics/README.md`).
- The blind baselines next to the measures: single-encoder score, the paper's overlap reasoning coded as a label, random ranking (`writing/2026-11-09-blind-baselines.md`).
- The renderer caveat on absolute probe numbers (`README.md:28`).
- Slot, from the control-partners rule (`experiments/2026-11-09-control-partners/PLAN.md:95`, verified): floor near zero and size-explained gives "the cheap checks cannot flag a bad partner"; can-go-negative and partner-specific gives "the finding stands and is stronger"; mixed gives each test reported separately. Which wording the README headline takes is the user's call (M6); the synthesis review asked for a reword (`reviews/2026-11-09-synthesis.md`, decision 3). The paper uses whatever the README ends up saying, not stronger.

**5. Perturbed frames (about 1 page). Method ready; result waits on M1.**
- Design: four picture types, action probe trained on our own clean re-renders and tested on perturbed ones, feature drift as a sanity check (perturbed PLAN and Amendment 2026-11-09). Report the redesign plainly: the first validation check failed, the probes were retrained on re-renders, and the gate then passed (perturbed LOG "gate A PASSED").
- Rule P decides (perturbed PLAN:96). With four partners an ordering match is a hint, not evidence (perturbed PLAN:100).
- Slot: C-RADIOv4 stands out under rule P / does not.

**6. Multi-seed proxy replication (about 1.5 pages). Method ready; results wait on M3, M4, M5.**
- Rules: calibration, the seed-noise rule, the level target, the outcome rule (proxy PLAN:155 to 161, :385).
- What happened so far, as a record: the first seed pair passed the noise rule but scored far below the paper, so the training was amended before the other runs (Amendment (c), PRs #3 and #5). Say this in the paper; it is part of why the result can be trusted.
- Slot (a), substantial effect: every seed drops and the mean drop clears the threshold; the PLAN's own words are "a trigger for the higher-fidelity multi-seed WAM replication, not proof" (proxy PLAN:160, verified).
- Slot (b), no substantial effect: the paper must say this is not evidence of no effect, because the 3-seed design has no detection-power argument (proxy PLAN:337, verified) and the seed-power rerun (M5) sets the wording.
- Slot (c), the proxy cannot tell: say which stop fired (level target, calibration, noise or floor) and what the user chose then.

**7. Slot: a pair-harm measure across Table 10 (PLAN Next 5).** Not designed. If it runs and gives a result before the deadline, it becomes a section here (about 1 page, taken from sections 4 and 5); if not, one sentence in the discussion as future work. Its limits are already written in PLAN Next 5 and go in with it.

**8. Discussion (about 0.5 page). Waits on the ending.**
- (a): the harm shows up in training but not in frozen features (if section 5 is also negative), so cheap frozen-feature checks are not enough; the fusion-step suspects (PLAN Next 2) are the next step.
- (b): a single-checkpoint harm did not reproduce in our proxy; what our proxy cannot rule out (the missing video prior, model size).
- (c): what the cheap tests can and cannot say, and what it would take to answer the question.

**9. Limitations (about 0.5 page). Ready.** Four partners; one LIBERO suite for the frozen-feature tests; a small from-scratch proxy without the video prior; our own choices where the paper is silent; the source paper's one checkpoint per row; different units (probe fit vs success rate), so only orderings are compared; absolute probe numbers specific to the renderer; three seeds as a screening result. Sources: `README.md` key findings, path memo section 4, PLAN Next 5's limits.

**Acknowledgments. Ready.** The NRP text, verbatim from `../_standards/08-compute-nautilus.md:155-156` (required in every paper that used NRP); the mentor and anyone else the user names.

**Appendix. Mostly ready.** The pre-registration trail (each rule, date locked, file); amendments and why; full result tables; compute used, from the experiment LOGs; encoder checkpoints and licenses.

## 3. Figures and tables

The figure convention in `STANDARDS.md:54` is blank. Proposed (`speculative`, user to agree, added by PR): every paper figure is made by a checked-in script in `src/` that reads result files, saves a vector PDF at the template's column width plus a PNG copy in `writing/figures/`, and uses one fixed color per encoder across all figures.

| # | What it shows | State | Source |
|---|---|---|---|
| F1 | Overview: encoders stacked as inputs and future-feature targets of the WAM, and where each of our three tests looks | new | could extend `src/plot_slide_diagrams.py` (the combining-grids schematic) |
| T1 | The paper's numbers for our four partners next to our clean-frame measures and the blind baselines | new script, data ready | tally and target-diagnostics result JSONs; `src/tally_baselines.py` |
| F2 | Control partners next to the real partners, on both tests | waits on M2 | control-partners results; new script |
| F3 | Perturbed frames: each partner's score under rule P with intervals | waits on M1 | `src/pf_analysis.py` outputs; new script |
| F4 | Proxy success per config and seed, with the paper's values | waits on M4 | `lp_eval.py` outputs; new script; the slide schematic `writing/figures/2026-11-07-noise-check-ruler.png` is a starting point |
| A1 | Training curves | exists, slide style | `src/plot_training_curve.py`; restyle |
| A2 | Compute cost | exists, slide style | `src/plot_compute_cost.py`; restyle |

Not for the paper: the GPU-use plot (operations) and the encoder-layers diagram (no layer study was run).

## 4. Citations

Rule: cite only papers the user has read and that have an entry in `literature/references.bib` (`../_standards/02`; the user's bar since the 2026-11-06 bib cut, `STANDARDS.md:15` and LOG 2026-11-06 [agent, docs]). The reading list belongs to the literature session; this plan only lists what the paper needs.

- In the bib now (verified): `liu2026what`, `liu2023libero`, `fei2025liberoplus`, `ranzinger2026cradiov4`, `guo2026metric`.
- Read, bib entry not merged: Hewitt et al. 2021 (waits on the user's answer in the literature session).
- Needed for the method, not read or not in the bib: Kornblith et al. 2019 (CKA); Fast-WAM (our code base); the models we run: DINOv3, SigLIP 2, SAM 3, VGGT-Omega, the Wan 2.2 VAE, the T5 text encoder; flow matching. A paper normally cites every model it runs (`recalled` convention). Under the user's bar each needs at least its method section read. Whether a partial read meets the bar is the user's call.
- Needed for related work, unread leads: Yardi et al. 2025, Chun et al. 2026, agostinelli2022, Hu et al. 2023, Zeng et al. 2026, arXiv 2606.07687, and whatever the user picks from the reading list's top 10.
- That is roughly 15 to 20 papers to read before section 2 and the setup can be cited in full (`speculative` count). This is likely the long pole.

## 5. Order of work

1. Venue, archival or not, deadline, page limit, authorship: user and mentor (the mentor meeting is 2026-11-11, LOG 2026-11-09 [agent, lead, planning] path to paper, Next).
2. The user's two choices in section 6 (who writes, where).
3. Skeleton in the venue's template with every section above, each slot marked, and source pointers as comments.
4. Draft the ready parts: setup, clean-frame tests (measurements only), limitations, introduction without contributions, the method halves of sections 5 and 6, acknowledgments, appendix trail. Make T1 and F1.
5. A fresh read-only reviewer checks every sentence and number against the files and papers it cites, as in PRs #7 to #11; fixes on the same branch.
6. Fill slots as M1, M2, M3 to M5 land, using the locked rules' wording; make F2 to F4. The headline wording follows the user's M6 sign-off.
7. Related work as the user reads (in parallel with 4 to 6).
8. Mentor read, revisions, final checks (anonymization, AI-use policy, NRP text, bib), submission by the user.

Timing (`speculative`): the path memo's realistic date for the full result set is early December. If the workshop deadline is in February or March (typical, `recalled`, not confirmed), that leaves room for the results sections, the mentor's read and revisions. If the deadline is earlier, ending (c)'s smaller paper, or a short non-archival version, may be what fits.

## 6. Decisions for the user

1. Who writes the prose. (A) The draft session writes each section from the repo, the user rewrites it in their own words. (B) The user writes from an annotated skeleton (bullets with sources), the draft session fact-checks. (C) A mix by section, for example (A) for setup and appendix, (B) for introduction and discussion. The user will present and defend the paper, which argues for (B) or (C) (`speculative`, from how the user handles slides).
2. Where the draft lives. LaTeX in the repo (`writing/paper/`, every change reviewable by PR, can sync to Overleaf for the mentor); Overleaf alone; or a Google Doc in the user's Drive first, moved to LaTeX near the deadline. CVPR submissions use LaTeX (`recalled`).
3. With the mentor: which workshop, archival or not, page limit, deadline, author list.
4. Reading for citations (section 4), through the literature session.
5. The figure convention line in `STANDARDS.md` (section 3), by PR.
