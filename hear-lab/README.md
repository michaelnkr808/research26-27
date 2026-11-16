# HEAR Lab: spatial audio for world action models

**Status:** exploring. The direction was set up on 2026-11-15 and nothing has been run yet. The first step is the literature survey. Current agenda: `PLAN.md`. History: `LOG.md`.

**Question:** What benefits does a world action model gain from being given spatial audio? In the lead's own words (2026-11-15): "relating spatial audio to world action models and investigating what benefits can be gained from providing world action models with spatial audio".

**What done looks like:** not defined yet. This has to be written down with the PI before any experiment is designed (`../_standards/01-research-process.md`). It needs to name the result or artifact that would answer the question and the evidence that would count.

## Working definitions
These are starting points written from memory (`recalled`), not from a source. The literature survey replaces them with cited definitions.
- **World action model:** a model that predicts what a robot will observe next (usually video) together with the actions it should take.
- **Spatial audio:** sound recorded or simulated so that it keeps where each source is (direction and distance), for example two-ear (binaural) or multi-microphone recordings, as opposed to a single mixed-down channel.

## Open scope questions
None of these are decided. Each one becomes a decision record in `decisions/` once the lead and the PI settle it.
- Which world action model (or models) the work builds on.
- Which benchmark or task set measures "benefit", and with which metrics.
- Where the spatial audio comes from: real recordings, simulation, or both.
- What the target output is (paper, venue, timeline).

## Key findings so far
- None yet.

## Ownership
- Lead: the repo owner.
- PI: the HEAR Lab PI.
- Work is done mostly by AI agents under the lead's direction (`../_standards/05-ai-assisted-work.md`): a planning lead session (`hear-lab-planning`) and an experiment lead session (`hear-lab-experiment`).

## Key files
- `STANDARDS.md`: topic rules, including where experiments run and the compute budget
- `PLAN.md`: living agenda (now, next, parked, recently done)
- `LOG.md`: research log for the direction (short session summaries; detail is in each experiment's `LOG.md`)
- `experiments/`: one folder per experiment, each with `config.*`, `PLAN.md`, `LOG.md`, `README.md` (copy `experiments/_new-experiment/`)
- `decisions/`: decision records
- `literature/`: paper notes and `references.bib`
