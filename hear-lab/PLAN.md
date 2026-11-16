# Plan: HEAR Lab

Living agenda. Only work that is running, next, or parked lives here. No results: those go in the experiment's `README.md`, history in `LOG.md`, conclusions in `README.md` "Key findings". When an item finishes, replace it with one line under "Recently done" pointing to its LOG entry, and prune that list now and then. When an item is dropped or parked, record why in the LOG or a decision record. Every session that changes an item updates this file in the same commit as its LOG entry.

## Now
- **First task from the PI: get the RMS-AQA benchmark running.** Planning stage, nothing run yet. What the planning lead read about the benchmark (code, data access, compute, challenge dates) is in the `LOG.md` entry for 2026-11-15.
  - **Blocked on:** dataset access. The dataset is gated with manual approval, so the lead has to request it on Hugging Face (`PeacefulData/RMS-AQA`) before anything can be downloaded.
  - **Open, for the lead and the PI:** (1) what "running" means: score an existing model on the validation set, reproduce the paper's trained baseline for one or more of its four models, or enter the challenge (submission deadline 2026-12-10 per the challenge site, which marks its dates as preliminary); (2) whether the 200 dollar budget is meant to cover a training run.
  - **Settled 2026-11-15:** by "the cluster" the PI means Vast.ai (per the lead), which matches the Vast.ai-first rule in `STANDARDS.md`.
  - Then: the planning lead writes the experiment `PLAN.md`, and the experiment lead builds and runs it.
- **Scope the direction with the PI.** Settle the open scope questions in `README.md`. Owner: the lead, with the planning lead writing up each answer as a decision record.
- **Literature survey** (`../_standards/01-research-process.md`: survey prior work before building). Not started; waits for the lead's go. Seeded by the four papers in `literature/reading-list.md`. It covers three things: audio as an input to robot policies and world models, spatial audio in embodied AI, and which world action models could take an audio input. Output: entries in `literature/references.bib`, the reading list kept current, and a short summary of what exists and what is missing.

## Next
1. Decision records for the base model, the benchmark, and the audio source, once the scope questions and the survey have answers.
2. Fill the **open** entries in `STANDARDS.md` from those decisions (by pull request).
3. Check the Interspeech 2027 deadline and paper format against the official call for papers, and work back a timeline from it.

## Parked
- None.

## Recently done
- None yet.
