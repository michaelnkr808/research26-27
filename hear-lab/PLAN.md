# Plan: HEAR Lab

Living agenda. Only work that is running, next, or parked lives here. No results: those go in the experiment's `README.md`, history in `LOG.md`, conclusions in `README.md` "Key findings". When an item finishes, replace it with one line under "Recently done" pointing to its LOG entry, and prune that list now and then. When an item is dropped or parked, record why in the LOG or a decision record. Every session that changes an item updates this file in the same commit as its LOG entry.

## Now
- **First task from the PI: get the RMS-AQA benchmark running.** Planning stage, nothing run yet. What the planning lead read about the benchmark (code, data access, compute, challenge dates) is in the `LOG.md` entry for 2026-11-15.
  - **Blocked on:** dataset access. Downloads are gated behind a registration form on Hugging Face (`PeacefulData/RMS-AQA`); the page says approval is automatic, but the dataset's setting is manual approval. The lead made the request on 2026-11-15. Still waiting on 2026-11-16: a test download with the lead's saved login was refused (`LOG.md` 2026-11-16). Found 2026-11-17: the challenge site also requires a team registration on a separate Google Form, and the email on the Hugging Face request must match the form's primary contact email exactly; the lead has not submitted that form, which may be why the request is still pending (`LOG.md` 2026-11-17). Decision for the lead and the PI: register a team, or email the organizers (`rmsaqa.2027@gmail.com`) for research-only access. The lead emailed the organizers on 2026-11-18 asking for research-only access; waiting for the reply.
  - **The lead's reading of the PI (2026-11-16, not yet confirmed by the PI):** the PI wants the benchmark set up on the cluster so that experiments can be run through it.
  - **Proposed bring-up check (planning lead, waits for the lead's OK):** train the Audio Flamingo 3 baseline with the paper's recipe and compare its validation score with the paper's row. No trained checkpoint is released, so nothing can be scored without training first. This needs the whole dataset (about 317 GB). Details and what was read: `LOG.md` 2026-11-16.
  - **Open, for the lead and the PI:** (1) confirm what "running" means, and whether the PI also wants a challenge entry (submission deadline 2026-12-10 per the challenge site, which marks its dates as preliminary); (2) whether the 200 dollar budget is meant to cover a training run (its cost is unknown until a real batch is timed); (3) whether the PI already has dataset access or can nudge the organizers; (4) where trained checkpoints and results are kept between Vast.ai rentals, since a rented machine keeps nothing once destroyed.
  - **Settled 2026-11-15:** by "the cluster" the PI means Vast.ai (per the lead), which matches the Vast.ai-first rule in `STANDARDS.md`.
  - Then: the planning lead writes the experiment `PLAN.md`, and the experiment lead session (`hear-lab-experiment`) builds and runs it. Experiments are not run from the planning session (the lead, 2026-11-15).
- **Scope the direction with the PI.** Next PI meeting: 2026-11-19 (weekly, Thursdays). The lead brings three candidate research questions and the open PI questions (`LOG.md` 2026-11-18). Settle the open scope questions in `README.md`. The lead's approach (2026-11-16): add an audio encoder to the world action model so it can also make decisions based on audio input (`LOG.md` 2026-11-16 addendum). Still to ask the PI: how RMS-AQA connects to world action models. The lead prefers Wan 2.2 as the base (2026-11-16, a preference, not yet a decision record). Opened by the survey: whether the setting is simulated navigation, where spatial audio data exists, or robot manipulation, where no public spatial audio data was found. Owner: the lead, with the planning lead writing up each answer as a decision record.
- **Choose the base model.** Owner: the lead ("ill decide on the base model in a bit", 2026-11-16). Candidates and what each releases: `literature/survey/2026-11-16-summary.md`. Open check: whether WNM-3D can be evaluated where SoundSpaces 2.0 renders audio.
- **Find the data.** No ready set with egocentric video, spatial audio and navigation actions was found in two search rounds; the lead expects one exists. Fallback: render with SoundSpaces 2.0 (`literature/survey/2026-11-16-spatial-audio-navigation-data.md`, closing section).

## Next
1. Decision records for the base model, the benchmark, and the audio source, once the scope questions and the survey have answers.
2. Fill the **open** entries in `STANDARDS.md` from those decisions (by pull request). The same pull request replaces the unchecked "end of February" deadline wording there with the checked date, 2027-02-09.
3. Work back a timeline from the Interspeech 2027 paper deadline (2027-02-09). When the call for papers opens, check the paper format and page limit; the dates page listed the call as "Soon" on 2026-11-15.

## Parked
- None.

## Recently done
- Prior-work check: audio in a world model has a small-scale precedent; mono against spatial audio, and audio in a large pretrained video world action model, were not found (PR #51). `LOG.md` 2026-11-16.
- Survey follow-up: navigation data, navigation world models, bibliography, summary, read-first list (PR #50). `LOG.md` 2026-11-16.
- Literature survey, three topic files, reviewed and merged (PR #49). `LOG.md` 2026-11-16.
- Direction folder set up and merged (PR #45). `LOG.md` 2026-11-15.
- Interspeech 2027 dates checked against the official dates page. `LOG.md` 2026-11-15.
