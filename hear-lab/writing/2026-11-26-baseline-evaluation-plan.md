# Baseline evaluation plan, draft 1 (2026-11-26)

- **Status:** draft by the planning lead for the lead and the PI. Nothing here has been run, approved or rented. Choices marked "recommended" are the planning lead's; the lead and the PI decide.
- **What it answers:** the PI's first milestone after the 2026-11-26 meeting, quoted: "please prepare a baseline evaluation plan covering the models, test scenarios, question types, and metrics." (`LOG.md` 2026-11-26, PI direction entry. The PI's full message is kept outside this public repo.)
- **Sources:** the literature survey `literature/survey/2026-11-26-long-form-spatial-audio-qa-baselines.md` (pull request #98, open when this was written; its read depth and its reviewers' corrections are stated in the file) and the experiment session's two answers on the sound engine (`LOG.md` 2026-11-26, engine entry and ambisonics entry). The planning lead read the survey file, not the papers, model cards or code behind it. Every fact about a model, a benchmark or a metric below is from that file unless marked otherwise.
- **Labels:** `verified` = read by the planning lead this session in the named file; `recalled` = from memory; `speculative` = the planning lead's reading or proposal.

## 1. The question for this stage

Under which conditions do existing models that answer questions about spatial audio lose track of which sound source is which, and what kinds of mistakes do they make? The PI names three conditions: a source stops and is heard again later; two sources move along crossing paths; several sources of one class appear at different positions or times. Each is compared with a simpler control, at growing audio length.

What the survey says about prior work (`verified` in the survey file; a search result, not proof): no audio-only spatial question answering benchmark it read asks about any of the three conditions with per-source identity as ground truth, and all audio-only benchmarks it read use clips of at most 10 s. The closest are ST-OmniQA (audio-visual, 10 s), SonicWorld / Sonic-Bench (a two-clip diagnostic across a speech pause) and ST-AudioQA, which lists these cases as future work.

## 2. Models (recommended list)

| Model | Audio it takes | Longest input | Access | Role |
|---|---|---|---|---|
| MiDashengLM-Spatial | two-ear, 16 kHz | trained on 30 to 60 s clips; no enforced cap, not tested beyond | ungated, Apache-2.0 | first model to bring up |
| BAT | two-ear, 32 kHz | 10 s | encoder licence is non-commercial; needs LLaMA-2-7B, which has a manual-approval gate | second two-ear model |
| ST-AudioLM | first-order ambisonics, 32 kHz | 10 s | weights gated (auto-approve form, academic non-commercial); adapter on a separately fetched base | the one model the survey found that was built for moving sources |
| Spatial-Omni (7B variant) | first-order ambisonics, 16 kHz | 20 s window | ungated weights, non-commercial; base model fetched separately | second ambisonics model |
| Audio Flamingo 3 | one channel, 16 kHz | 10 min | ungated, non-commercial | control with no direction cue |

- Not available as baselines: Hear You Are and HLR-LLM, the two papers the PI linked on 2026-11-19. The survey found no released model for either.
- Two-ear models come first: the simulator renders two-ear audio today. The two ambisonics models need a small change to our engine binding and a format check per model (channel order, scaling, heading) before their scores count (the experiment session's answer, from the engine's header, not run).
- A second control (`speculative`, ours): each spatial model is also given the one-channel average of the same scene, to separate "cannot track" from "does not use direction".
- Access the lead would have to request: LLaMA-2-7B (for BAT) and the ST-AudioLM form. Not requested yet.
- Inference hardware and cost: not measured. The survey found one statement (ST-AudioLM's README: tested on one H100).

## 3. Test scenarios

Each hard scenario gets a control that differs in one thing only (the pairing is ours, `speculative`):

| Scenario (the PI's) | Control |
|---|---|
| A source stops and is heard again later | the same source and path, sounding without a gap |
| Two sources move along crossing paths | the same two sources moving without crossing |
| Several sources of one class at different positions or times | the same layout with sources of different classes |

- **Lengths:** the same scene design at 10 s, 30 s, 60 s and 120 s (`speculative`, to be confirmed). 10 s is what every audio-only benchmark in the survey uses; 60 s is the length in the PI's figure and MiDashengLM-Spatial's training length.
- **Rendering:** our sound engine in 3D scenes with the measured head (decision `003`), a listener that stays in place (our reading of the PI's figure), and moving sources made from one impulse response per position with a crossfade between positions. That is how the prior datasets in the survey make moving sources (DCASE 2020, SpatialScaper, ST-AudioQA), and how the experiment session says our binding can do it (not yet run for these cases).
- **Ground truth:** the generator writes, for every source, its identity, class, position over time and the intervals when it sounds. Every question and every score below is computed from that record.
- **Sound clips:** speech and other classes need clips with clear licences. Only a telephone ring is in hand. Open.
- **How many scenes:** not set. It is fixed, with the predictions and the pass rules, in the experiment `PLAN.md` before any scored run, after a timed batch gives the render cost.

## 4. Question types

- **Borrowed:** the six question dimensions of RMS-AQA (counting, location, temporal detection, spatial relation, temporal relation, action prediction) and its four-option multiple choice format, so that scoring is exact and needs no judge model. RMS-AQA's own question templates were not seen (gated card).
- **Ours (`speculative`, untested):** identity questions written from the ground truth, for example: is the sound at the end from the same source as the one at the start; how many different sources of this class were there; after the two paths cross, which side is each source on. Wrong options are built to be the specific mistakes we want to tell apart: two sources swapped, two sources merged into one, one source split into two, a source lost after silence.
- Every question carries tags: the scenario, the length, and whether answering needs source identity.

## 5. Metrics

- **Answer accuracy** per scenario, length and question type, with chance level stated (25% for four options). The headline is the gap between each hard scenario and its control, and how that gap changes with length.
- **Identity measures.** The PI names identity switches and re-identification accuracy. In the survey these exist only in tracking papers (identity switches, IDF1, HOTA, LOCATA's swap and fragmentation rates; arXiv 2506.11145 adapts HOTA to speakers who go silent and move), and they need a predicted identity at every time step, which a question answering model does not output. Two ways to get them (`speculative`, ours):
  1. From answers (recommended for this stage): re-identification accuracy is the accuracy on same-or-different-source questions; an identity switch is counted when the answer after a crossing or a gap assigns a source to the other source's position or class.
  2. From a timeline: ask the model to list each source with its position over time and score that list with the tracking metrics. Whether these models can produce such a list is unknown; an exploratory add-on.
- **Localization error:** accuracy on location questions with fixed direction bins shared by all models, since the models differ in how they report direction.
- **Error types:** the share of wrong answers of each kind (swapped, merged, split, lost), read from which wrong option was chosen.

## 6. Recordings longer than a model's limit

Most of the models take 10 to 20 s. Options: (a) score each model only up to its own limit; (b) cut the recording into pieces the model accepts, ask per piece, and combine the answers by a fixed rule written down beforehand; (c) pass a text summary from piece to piece. Recommended: report (a) and (b). (b) is the plain baseline that shows what is lost without memory across pieces. Whether (b) is a fair baseline is a question for the PI.

## 7. Checks before any score counts

1. The simulator's left and right cue is correct (the small Step 2 batch, pull request #95; not rented).
2. Each model first reproduces a number or an example from its own release, then passes a known-answer test on our renders (one source, clearly left or right).
3. The format check for the two ambisonics models.
4. Control accuracy is clearly above chance for a model before its hard-scenario gap is read. These models were trained on other simulators, so a model can fail on our audio for reasons that have nothing to do with identity.

## 8. Order of work (small steps, each checked before the next)

1. The small Step 2 batch: direction check and render time per step.
2. Bring up MiDashengLM-Spatial on its own examples, then the known-answer test on our renders.
3. A scene generator for the three scenarios and their controls at 10 s; a handful of scenes for the lead to listen to.
4. The experiment `PLAN.md`: questions, metrics, scene counts, predictions and pass rules, fixed before the scored run.
5. Add BAT, then Audio Flamingo 3 as the control, then the two ambisonics models; then the longer lengths.

Each rental needs the lead's OK in the experiment session (`STANDARDS.md`). No training is part of this stage.

## 9. Open questions

For the PI:
- Is cutting a long recording into pieces (section 6, option b) an acceptable baseline?
- Audio only, or audio with video? Does the listener stay in place?
- Does "our simulation environment" mean the sound engine in the 3D scenes we already use?
- Which sound classes matter (speech, animals, objects)?
- Does RMS-AQA come back (its question templates would help section 4, and access needs a challenge team registration)?
- Is the navigation work (decisions `001` to `003`, Steps 3 and 4) dropped or kept beside this? What does "world audio model" mean here?

For the lead: the access requests in section 2; whether the non-commercial licences are acceptable for a paper; the budget for this stage once a timed batch exists.

## 10. Borrowed and ours

- **Borrowed:** the models (released by their authors); moving sources from per-position impulse responses (DCASE 2020, SpatialScaper, ST-AudioQA); question dimensions and multiple choice format (RMS-AQA); the definitions of the identity metrics (tracking papers).
- **Ours, untested:** the three scenario and control pairs; the lengths; the identity questions and their wrong options; identity measures computed from answers; the one-channel control for each spatial model; the fixed rule for combining pieces.

## 11. Not verified

Nothing was run. Model facts are the survey's, at its stated read depth; no model was downloaded. Engine abilities for several, moving and intermittent sources and for ambisonics are the experiment session's reading of code and a header. Render cost, inference cost and scene counts are unknown.
