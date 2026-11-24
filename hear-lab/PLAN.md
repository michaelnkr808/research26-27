# Plan: HEAR Lab

Living agenda. Only work that is running, next, or parked lives here. No results: those go in the experiment's `README.md`, history in `LOG.md`, conclusions in `README.md` "Key findings". When an item finishes, replace it with one line under "Recently done" pointing to its LOG entry, and prune that list now and then. When an item is dropped or parked, record why in the LOG or a decision record. Every session that changes an item updates this file in the same commit as its LOG entry.

## Now
- **The ladder: four rungs, one at a time** (the lead, 2026-11-21: "I think we do need to go increamentally here"; rungs 2 to 4 answered "okay sure sounds good", "okay makes sense", "okay right"; on rung 1 the lead asked which paper is meant, which the planning lead reads as agreement, `LOG.md` 2026-11-21 ladder entry). Each rung gets its own experiment `PLAN.md` and the lead's OK in the experiment session before any rental; a rung starts only when the one below it works.
  1. **Reproduce the baseline, no training (in flight, handed to `hear-lab-experiment` on 2026-11-21; plan approved by the lead 2026-11-21, Unseen split, paper's stop rule, stall recovery on; folder merged (PR #82); timing batch of 30 done 2026-11-22 on Vast.ai, about 4 USD, 18 of 30 succeeded; waiting on the lead's choice of N for the main run, `LOG.md` 2026-11-22 entry).** Run the released WNM-3D stage 3 checkpoint closed-loop in its own simulator (GN-Bench) and compare the success rate with the WNM-3D paper's reported figure; report it split by the check 3 visible / not-visible flag; time and cost of an inference-time run.
  2. **Make the sound and check it.** Render two-ear and one-channel audio for a small set of episodes with the sound engine and a measured HRTF, and check that the direction cue points at the target. Built on route A (decision `003`).
  3. **Smallest training step.** Add audio by the published recipe (LoRA on the backbone plus new audio modules, `literature/survey/2026-11-21-new-input-training-recipes.md`, PR #81), on a small set, with one question: does the model use the sound when the target is out of sight. Decided at this rung: the audio window per step (Spatial-AST expects 10 s clips), and whether the no-audio version gets the same LoRA training.
  4. **The three-version comparison** (decision `001`).
- **Simulator route: route A, committed** (`decisions/003-simulator-route-a.md`; the lead, 2026-11-22: "commit to route A it sounds good"). WNM-3D's own simulator plus the RLR sound engine, two-ear audio with a measured HRTF; Habitat with SoundSpaces 2.0 is the written fallback. Carried into rung 2: the 39 scenes whose object boxes may be mirrored; the distance between the GN-Matrix goal position and the object's box; the transmission setting; the wall-overlap bar missed in 2 of 4 scenes; the lead's own look at the overlay and visibility images; the InteriorGS terms (not yet read).
- **Waiting for the lead's answer: not planning on full fine-tuning within the 200 dollars.** Check 2: the released full fine-tune does not fit one H200, and the offload fallback costs about 37 dollars per 1000 steps (`LOG.md` 2026-11-21, diagnosis entry). The planning lead recommends training by the published LoRA recipe instead (rung 3); how the model is trained is not decided.
- **For the PI (weekly, Thursdays):** the training-cost measurement and what it means for the 200 dollars; whether using the SoundSpaces 2.0 sound engine outside Habitat counts as "not using SoundSpaces 2.0"; what "long-form" means for this work; what the paper has to show; the research question itself (decision `001` is the lead's, not yet confirmed by the PI); whether to register an RMS-AQA challenge team or leave it parked.

## Next
1. The audio encoder: which one, the window per step, and where it attaches (a decision record before rung 3; the route memos locate the two attachment points in WNM-3D's code).
2. The experiment `PLAN.md` for rung 4, before anything runs: metrics and pass bars or predictions, the visibility rule, the one-channel definition, the instruction text, the pose source, the success radius, whether a DAgger stage is used, and the number of seeds against the budget (the general rule is at least 3; a relaxation is a recorded override in `STANDARDS.md`).
3. Fill the remaining **open** entries in `STANDARDS.md` from those decisions (by pull request).
4. Work back a timeline from the Interspeech 2027 paper deadline (2027-02-09). When the call for papers opens, check the paper format and page limit; the dates page listed the call as "Soon" on 2026-11-15.

## Parked
- **RMS-AQA benchmark (the PI's first task).** Parked by the lead on 2026-11-18 ("maybe just drop this for now"). The PI assigned it and has not been told yet; the lead raises it on 2026-11-19, and it comes back to "Now" if the PI wants it. State when parked: nothing run; dataset access still pending (the Hugging Face request of 2026-11-15, and the lead's email of 2026-11-18 to the organizers asking for research-only access; the challenge's team registration form was not submitted; the organizers answered on 2026-11-20 that the dataset is available only to registered challenge participants, `LOG.md` 2026-11-20); the proposed bring-up check (train the Audio Flamingo 3 baseline with the paper's recipe, about 317 GB of data) was never approved. What was read: `LOG.md` 2026-11-15 to 2026-11-18. Why parked: `LOG.md` 2026-11-18 addendum 2.

## Recently done
- Decision `003`: route A committed, measured HRTF. `LOG.md` 2026-11-22.
- The three route A checks (sound engine, timed training step, visible starts), run and reviewed (PRs #75 to #80). `LOG.md` 2026-11-20 and 2026-11-21.
- Hugging Face access for VGGT-Omega and InteriorGS on the lead's login. `LOG.md` 2026-11-20.
- Sizing of the two simulator routes and their comparison, reviewed and merged (PR #73). `LOG.md` 2026-11-19.
- Decisions `001` (research question) and `002` (base model WNM-3D, simulated navigation) accepted and merged with `README.md` and `STANDARDS.md` updates (PR #72). `LOG.md` 2026-11-19.
- PI meeting of 2026-11-19 and the PI's written feedback. `LOG.md` 2026-11-19 (PI feedback).
- Novelty check, WNM-3D simulator gap, adversarial second check (PR #68) and PI follow-ups (PR #71), merged. `LOG.md` 2026-11-19 [agent, literature].
- Prior-work check: audio in a world model has a small-scale precedent; mono against spatial audio, and audio in a large pretrained video world action model, were not found (PR #51). `LOG.md` 2026-11-16.
- Survey follow-up: navigation data, navigation world models, bibliography, summary, read-first list (PR #50). `LOG.md` 2026-11-16.
- Literature survey, three topic files, reviewed and merged (PR #49). `LOG.md` 2026-11-16.
- Direction folder set up and merged (PR #45). `LOG.md` 2026-11-15.
- Interspeech 2027 dates checked against the official dates page. `LOG.md` 2026-11-15.
