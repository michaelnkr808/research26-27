# HEAR Lab: spatial audio for world action models

**Status:** exploring. Nothing has been run yet; the PI's first task is to get the RMS-AQA benchmark running. A literature survey and a prior-work check are done (`literature/survey/`, PRs #49, #50 and #51; start with `literature/survey/2026-11-16-summary.md`). Three candidate research questions go to the PI on 2026-11-19 (`LOG.md` 2026-11-18); the dataset request for RMS-AQA is still pending. Current agenda: `PLAN.md`. History: `LOG.md`.

**Question:** What benefits does a world action model gain from being given spatial audio? In the lead's own words (2026-11-15): "relating spatial audio to world action models and investigating what benefits can be gained from providing world action models with spatial audio".

**What done looks like:** a paper submitted to Interspeech. The official Interspeech 2027 dates page gives the paper submission deadline as 2027-02-09 and the paper update deadline as 2027-02-16 (`verified`, read 2026-11-15; the lead had given "end of February"). The paper format is not checked yet because the call for papers was not open. What result the paper has to show is not defined yet, and has to be written down with the PI before any experiment that feeds the paper is designed (`../_standards/01-research-process.md`).

## Working definitions
These are starting points written from memory (`recalled`), not from a source. The literature survey replaces them with cited definitions.
- **World action model:** a model that predicts what a robot will observe next (usually video) together with the actions it should take.
- **Spatial audio:** sound recorded or simulated so that it keeps where each source is (direction and distance), for example two-ear (binaural) or multi-microphone recordings, as opposed to a single mixed-down channel.

## Open scope questions
None of these are decided. Each one becomes a decision record in `decisions/` once the lead and the PI settle it.
- **Which world action model the work builds on.** Open. One candidate the lead named is Wan 2.2, the model already used in this repo's `../world-model-stacking-encoders/` direction.
- **Which benchmark or task set measures "benefit", and with which metrics.** Open. RMS-AQA (below) is a spatial audio question answering benchmark, not a world action model benchmark, so how it connects to the question is part of what has to be settled.
- **Where the spatial audio comes from.** Open. The RMS-AQA dataset is the first one the direction will handle.
- **What the paper has to show.** Open (see "What done looks like").

## First task from the PI: RMS-AQA
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
- `decisions/`: decision records
- `literature/`: `reading-list.md` (what has been read), `references.bib`, and paper notes; `literature/survey/` holds the reviewed survey files
