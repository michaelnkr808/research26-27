# HEAR Lab: spatial audio for world action models

**Status:** exploring. Three bring-up checks have run (`experiments/2026-11-20-route-a-checks/`), and Step 1 of the plan is done: the released WNM-3D result was reproduced with no training (`experiments/2026-11-21-rung1-wnm3d-baseline/`, PR #86). No experiment with audio has run. The research question, the base model and the setting are set by the lead (`decisions/001` and `002`, 2026-11-19; the PI has not confirmed the question). Work goes step by step (`PLAN.md`, four steps, called rungs before 2026-11-24): next is Step 2, rendering the audio for a small set of episodes and checking it. The audio-visual data comes from WNM-3D's own simulator with an added sound engine; Habitat with SoundSpaces 2.0 is the fallback (`decisions/003`). The PI's first task, RMS-AQA, is parked (`PLAN.md`, Parked). The literature survey and the novelty checks are in `literature/survey/` (PRs #49, #50, #51, #68 and #71; start with `literature/survey/2026-11-16-summary.md`). Current agenda: `PLAN.md`. History: `LOG.md`.

**Question:** What benefits does a world action model gain from being given spatial audio? In the lead's own words (2026-11-15): "relating spatial audio to world action models and investigating what benefits can be gained from providing world action models with spatial audio".

**Research question (`decisions/001-research-question.md`):** When the target is out of sight, does a navigation world action model reach a sounding object more often with spatial (two-ear) audio than with one-channel audio or with no audio? Three versions of one model (WNM-3D), scored on reaching a sounding object, with success split by whether the target was visible at the start. An encoder comparison and long-form audio are deferred extensions.

**What done looks like:** a paper submitted to Interspeech. The official Interspeech 2027 dates page gives the paper submission deadline as 2027-02-09 and the paper update deadline as 2027-02-16 (`verified`, read 2026-11-15; the lead had given "end of February"). The paper format is not checked yet because the call for papers was not open. What result the paper has to show is not defined yet, and has to be written down with the PI before any experiment that feeds the paper is designed (`../_standards/01-research-process.md`).

## Working definitions
These are starting points written from memory (`recalled`), not from a source. The literature survey replaces them with cited definitions.
- **World action model:** a model that predicts what a robot will observe next (usually video) together with the actions it should take.
- **Spatial audio:** sound recorded or simulated so that it keeps where each source is (direction and distance), for example two-ear (binaural) or multi-microphone recordings, as opposed to a single mixed-down channel.

## Open scope questions
Each open one becomes a decision record in `decisions/` once the lead and the PI settle it.
- **Which world action model the work builds on.** Decided by the lead: WNM-3D, a navigation model built on Wan 2.2 (`decisions/002-base-model-and-setting.md`).
- **Setting.** Decided by the lead: navigation in simulation, not robot-arm manipulation (`decisions/002`).
- **Where the audio-visual data comes from.** Decided by the lead: WNM-3D's own simulator (GN-Bench on InteriorGS scenes) with an added sound engine (RLR-Audio-Propagation) and a measured HRTF; Habitat with SoundSpaces 2.0 is the written fallback (`decisions/003-simulator-route-a.md`). Whether this meets the PI's condition on SoundSpaces 2.0 is still a question for the PI.
- **Which task set measures "benefit", and with which metrics.** Partly set: the task is walking to a sounding object, with success split by whether the target was visible at the start (`decisions/001`). The episodes, the metrics and the number of seeds are open and go in the experiment `PLAN.md`.
- **Which audio encoder.** Open.
- **What the paper has to show.** Open (see "What done looks like").

## First task from the PI: RMS-AQA
Parked by the lead on 2026-11-18. The PI's written feedback after the 2026-11-19 meeting did not bring it back, so it stays parked (`PLAN.md`, Parked; `decisions/001`).

Get the RMS-AQA benchmark running (`chen2026rmsaqa` in `literature/references.bib`). RMS-AQA is a two-stage multiple-choice question answering benchmark on 10-second, four-channel spatial audio clips of home scenes. Its training clips are simulated, its validation set mixes simulated and recorded clips, and its hidden evaluation set is recorded. It is also a live ICASSP 2027 Grand Challenge. The plan item and its open questions are in `PLAN.md`.

## Key findings so far
- None yet.

## Ownership
- Lead: the repo owner.
- PI: Dongheon Lee, Assistant Professor at PKNU, previously at Meta.
- Work is done mostly by AI agents under the lead's direction (`../_standards/05-ai-assisted-work.md`): a planning lead session (`hear-lab-planning`) and an experiment lead session (`hear-lab-experiment`).

## Key files
- `STANDARDS.md`: topic rules, including where experiments run and the compute budget
- `PLAN.md`: living agenda (now, next, parked, recently done)
- `LOG.md`: research log for the direction (short session summaries; detail is in each experiment's `LOG.md`)
- `experiments/`: one folder per experiment, each with `config.*`, `PLAN.md`, `LOG.md`, `README.md` (copy `experiments/_new-experiment/`)
- `decisions/`: decision records (`001` research question, `002` base model and setting, `003` simulator route)
- `literature/`: `reading-list.md` (what has been read), `references.bib`, and paper notes; `literature/survey/` holds the reviewed survey files
