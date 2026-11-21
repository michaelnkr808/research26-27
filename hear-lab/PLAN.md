# Plan: HEAR Lab

Living agenda. Only work that is running, next, or parked lives here. No results: those go in the experiment's `README.md`, history in `LOG.md`, conclusions in `README.md` "Key findings". When an item finishes, replace it with one line under "Recently done" pointing to its LOG entry, and prune that list now and then. When an item is dropped or parked, record why in the LOG or a decision record. Every session that changes an item updates this file in the same commit as its LOG entry.

## Now
- **Simulator route: route A first, not yet committed.** The lead accepted the planning lead's recommendation on 2026-11-19 ("yeah thats fine"): start with WNM-3D's own simulator plus a sound engine (route A), commit to it only after real checks come back, and keep Habitat with SoundSpaces 2.0 (route B) as the fallback. Decision `003` is written when the checks are back. Sources: `writing/2026-11-19-sizing-comparison-simulator-routes.md` and the two route memos beside it (PR #73). The checks were handed to the experiment session (`hear-lab-experiment`) as direction on the lead's "yes" of 2026-11-19 (`LOG.md` 2026-11-19, checks handed off). No rental is approved: each one needs the lead's OK in that session. Status 2026-11-20 (end of day): checks 1 and 3 run and reproduced (PRs #75, #79, both merged; check 1 not passed by its own rule with the default head model, passes its ear test with a measured HRTF, PR #76); check 2 designed and coded (PRs #77, #78, user decisions recorded) and run on 2026-11-21 on one H200 (PR #80: the released config does not fit on one card; it fits with optimizer offload at about 35 s per step). See `LOG.md` 2026-11-20 [agent, experiment] entries. State on 2026-11-20 (the experiment session's report): hand-off received, a plan presented to the lead, nothing started; it waits on the lead's answers in its window (`LOG.md` 2026-11-20).
  1. Does the sound engine (RLR-Audio-Propagation) run on a rented Linux machine and return a one-channel and a two-ear impulse response for one real InteriorGS scene; do the floor plans line up with the simulator's occupancy map (route A memo, unknowns 1 and 2).
  2. One timed real training step from the released WNM-3D checkpoint on the GPU type that would be rented (unknown 9 in both memos). This decides whether three versions fit in the 200 dollars, and with how many seeds.
  3. Counts of visible and not-visible starts on the real episodes (route A memo, unknown 4).
- **Access on the lead's Hugging Face account.** VGGT-Omega: the lead's saved login already has access (a HEAD request for the checkpoint file returned a redirect to the file, 2026-11-19). InteriorGS: access granted (the same kind of request returned 200 on 2026-11-20; it was refused on 2026-11-19). Nothing is blocked on access.
- **For the PI (weekly, Thursdays):** whether using the SoundSpaces 2.0 sound engine outside Habitat counts as "not using SoundSpaces 2.0"; what "long-form" means for this work; what the paper has to show; the research question itself (decision `001` is the lead's, not yet confirmed by the PI).

## Next
1. Decision `003` (simulator route), once the checks are back.
2. The experiment `PLAN.md` for the three-version comparison, before anything runs: metrics and pass bars or predictions, the visibility rule, the one-channel definition, the instruction text, the pose source, the success radius, whether a DAgger stage is used, and the number of seeds against the budget (the general rule is at least 3; a relaxation is a recorded override in `STANDARDS.md`).
3. The audio encoder: which one and where it attaches (a decision record; the route memos locate the two attachment points in WNM-3D's code).
4. Fill the remaining **open** entries in `STANDARDS.md` from those decisions (by pull request).
5. Work back a timeline from the Interspeech 2027 paper deadline (2027-02-09). When the call for papers opens, check the paper format and page limit; the dates page listed the call as "Soon" on 2026-11-15.

## Parked
- **RMS-AQA benchmark (the PI's first task).** Parked by the lead on 2026-11-18 ("maybe just drop this for now"). The PI assigned it and has not been told yet; the lead raises it on 2026-11-19, and it comes back to "Now" if the PI wants it. State when parked: nothing run; dataset access still pending (the Hugging Face request of 2026-11-15, and the lead's email of 2026-11-18 to the organizers asking for research-only access; the challenge's team registration form was not submitted; the organizers answered on 2026-11-20 that the dataset is available only to registered challenge participants, `LOG.md` 2026-11-20); the proposed bring-up check (train the Audio Flamingo 3 baseline with the paper's recipe, about 317 GB of data) was never approved. What was read: `LOG.md` 2026-11-15 to 2026-11-18. Why parked: `LOG.md` 2026-11-18 addendum 2.

## Recently done
- Sizing of the two simulator routes and their comparison, reviewed and merged (PR #73). `LOG.md` 2026-11-19.
- Decisions `001` (research question) and `002` (base model WNM-3D, simulated navigation) accepted and merged with `README.md` and `STANDARDS.md` updates (PR #72). `LOG.md` 2026-11-19.
- PI meeting of 2026-11-19 and the PI's written feedback. `LOG.md` 2026-11-19 (PI feedback).
- Novelty check, WNM-3D simulator gap, adversarial second check (PR #68) and PI follow-ups (PR #71), merged. `LOG.md` 2026-11-19 [agent, literature].
- Prior-work check: audio in a world model has a small-scale precedent; mono against spatial audio, and audio in a large pretrained video world action model, were not found (PR #51). `LOG.md` 2026-11-16.
- Survey follow-up: navigation data, navigation world models, bibliography, summary, read-first list (PR #50). `LOG.md` 2026-11-16.
- Literature survey, three topic files, reviewed and merged (PR #49). `LOG.md` 2026-11-16.
- Direction folder set up and merged (PR #45). `LOG.md` 2026-11-15.
- Interspeech 2027 dates checked against the official dates page. `LOG.md` 2026-11-15.
