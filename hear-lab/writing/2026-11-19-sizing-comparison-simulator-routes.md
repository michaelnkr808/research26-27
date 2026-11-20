# The two simulator routes compared (2026-11-19)

Written by the planning lead from the two sizing memos in this folder:
- Route A: `2026-11-19-sizing-route-a-audio-in-wnm3d-simulator.md` (keep WNM-3D in its own simulator, GN-Bench on InteriorGS scenes, and add a sound engine).
- Route B: `2026-11-19-sizing-route-b-habitat-soundspaces.md` (move WNM-3D to Habitat and use SoundSpaces 2.0).

Nothing was run for any of the three files. Every effort figure is the memo's estimate from reading code and is `speculative`. The recommendation at the end is the planning lead's and is `speculative`. The choice between the routes belongs to the user and the PI and becomes decision `003`.

What the data has to serve (decision `001`): three versions of WNM-3D (no audio, one-channel audio, two-ear audio), trained on the same episodes, tested on walking to a sounding object, with success split by whether the target was visible at the start.

## The routes in short

**Route A.** WNM-3D's renderer, scenes, episodes, converter, trainer and closed-loop client stay as released. One new thing is added next to the renderer: an acoustic engine that computes, for every camera pose, what the agent would hear, from room geometry built out of the InteriorGS floor plans and object boxes with assumed materials. This is what the PI asked for, on the condition that it can produce the data ("if you can use WNM-3D's own simulator for simulating audio-visual training data, then we will not use SoundSpaces 2.0", `LOG.md` 2026-11-19).

**Route B.** The audio comes ready-made from SoundSpaces 2.0 inside Habitat, on scanned scenes (Matterport3D or HM3D) with existing sounding-object episode sets. What has to be built is everything that connects WNM-3D to Habitat: the data rendering and its conversion to WNM-3D's training format, a Habitat client for WNM-3D's server, and a converter from WNM-3D's movement output to Habitat actions. The released weights have never seen these scenes.

## What is the same in both routes

- **The audio encoder and where it attaches.** Both memos locate the same two places in WNM-3D's model code (piece 7 in each). Designing the encoder is not sized in either.
- **The sound engine.** The most direct engine in route A is RLR-Audio-Propagation, which is also the engine inside SoundSpaces 2.0 (archived, Linux x64 binaries, CC-BY-NC). Route A calls it directly; route B calls it through an audio-enabled habitat-sim build. In both, getting it to run on a rented Linux machine is the first technical unknown.
- **Access needed for any training.** WNM-3D's training launcher exits unless the VGGT-Omega checkpoint file is present, whether or not a released checkpoint is given as the start (`scripts/train/wnm_3d_base.sh` lines 143 to 147 at `7fe2d5f`; `verified` by the planning lead this session). `facebook/VGGT-Omega` is gated with manual approval (route B memo, Hugging Face API read 2026-11-19). `docs/DOWNLOADS.md` says only that evaluation and inference do not need it (lines 53 to 54). So this access is needed in both routes. The route A memo's sentence that fine-tuning from a released checkpoint does not need it is wrong on this point and is to be corrected.
- **The cost of training is unknown and is the largest open item.** The documented recipe is full fine-tuning on 8 GPUs with a per-device batch of 12 (both memos, `wnm_3d_base.sh`), and the install guide says other set-ups have to be validated. Three versions at the general rule of at least 3 seeds each is 9 runs. The budget is 200 US dollars. No training step has been timed, so nobody can say yet whether even one run per version fits.
- **Design choices that either route forces, and that belong in the experiment `PLAN.md`:** what "one-channel" means (both memos point to the two-ear render averaged to one channel as the "same render" choice, and warn against a separate render); an instruction that does not tell the model where the target is, and is identical across the three versions (what the instruction says also decides what the no-audio version can possibly do); a visibility rule fixed before any result is seen; the same fine-tuning for the no-audio version as for the audio versions; audio rendered once and stored so all versions train on the same signal; the success radius.

## Where they differ

| | Route A (own simulator plus sound engine) | Route B (Habitat plus SoundSpaces 2.0) |
|---|---|---|
| What already works | The whole WNM-3D pipeline: renderer, converter, trainer, closed-loop client. The released weights were trained on these scenes. | The audio-visual simulator and episode sets (SAVi on Matterport3D; MINav code and GOAT-Bench on HM3D). |
| Main things to build | Geometry and materials from floor plans and boxes; the engine wrapper; sound sources; audio in the rendering, the LeRobot conversion, the trainer's dataset class, and the closed-loop client. | The pinned SoundSpaces install; rendering and a pose-faithful conversion to WNM-3D's format; a Habitat client; the movement-to-action converter. |
| Effort, pieces 1 to 6 (`speculative`, from each memo) | About 17 to 30 working days, plus 5 to 10 if new episodes, collision meshes or a C++ shim are needed. | About 13 to 35 working days, up to about 42 with a DAgger round in Habitat. |
| Scene access | InteriorGS, gated with automatic approval. Its terms were not read, so whether derived audio data may be released is unknown. | Matterport3D (signed form sent by institutional email, python 2.7 downloader) or HM3D (access request, API token). Calendar time. |
| Does the released model already work there | Yes. | No. The weights have never seen Habitat scenes, so every version needs extra fine-tuning just to see the new rooms, before audio adds anything. |
| How believable the sound is | Lower. Walls from a floor plan, objects as boxes, materials guessed from labels. Nobody was found who has done room acoustics on these scenes (one precedent on Gaussian-splat reconstructions, pi-AVAS). A reviewer may doubt the out-of-sight result. | Higher and familiar. Scanned rooms, the pipeline the PI's two linked papers and NaVLA2 used. Note that the continuous SoundSpaces simulator turns materials off by default. |
| How results sit next to other papers | A new, self-made task. No other audio navigation paper uses it. | Episode sets other audio navigation papers use, though the movement rule would still differ from theirs. |
| Specific traps | The episode instructions name the target and often its room; the released checkpoint already trained on the train_seen episodes; the simulator has no clock, so seconds of audio per frame is a convention; the size of the not-visible group is unknown. | A weak no-audio version after transfer would make audio gains look like repair of a broken baseline; each object category has its own sound in SAVi, so one-channel audio can name the target. |
| The PI's condition | Meets it for scenes, pictures and episodes. The audio physics would still be SoundSpaces 2.0's engine; whether that counts is a question for the PI. | Does not meet it. |

## What decides it

The effort ranges overlap, so effort does not decide (`speculative`).

1. **Compute.** The budget is the tightest limit, and route B spends part of it teaching every version new rooms. Route A starts from a model that already navigates its scenes. This is the planning lead's inference (`speculative`); no timing exists for either.
2. **The PI's stated condition.** Route A is the one the PI asked for if it can produce the data.
3. **What kind of risk the user prefers.** Route A's risk is about the sound: an engine run outside its usual home, simplified geometry, and a self-made task a reviewer has not seen. Route B's risk is about access, an old pinned install, transfer, and budget.

## Recommendation (planning lead, `speculative`)

Start with route A, and only commit to it after the first three items below come back. They are checks on the real tools and real data, not small versions of the experiment. Each one needs the experiment session and the user's OK there; none is approved by this file.

1. **Ask the PI** whether RLR used outside Habitat, and SoundSpaces' sound clips, count as "not using SoundSpaces 2.0". No compute.
2. **Start the two access requests,** since both take the user's own account and one takes calendar time: accept the InteriorGS terms, and request VGGT-Omega access (needed for training in either route).
3. **Does the engine run, and does the geometry line up?** On a rented Linux x64 CPU machine: load the RLR library, build one real InteriorGS scene from its floor plan and boxes, and write a one-channel and a two-ear impulse response (route A memo, unknown 1); and draw the floor plan over the simulator's occupancy map for three or four scenes (unknown 2). If the engine does not load, the fallbacks in order are an audio-enabled habitat-sim or Habitat-GS build (route A memo, piece 1), then route B.
4. **Time one real training step** from the released checkpoint on the GPU type that would be rented (unknown 9 in both memos). This is the same in both routes and decides whether three versions fit in the budget at all, and with how many seeds. It needs the VGGT-Omega access from item 2.
5. **Count the visible and not-visible starts** on the real episodes (route A memo, unknown 4). If one group is tiny, new episodes are needed and route A's effort moves to the top of its range.

Route B stays the fallback. Its first check would be the SoundSpaces install on a rented machine (route B memo, unknown 1).

## Decisions this file does not make

- Route A or route B (the user and the PI; decision `003`).
- How many seeds, given the budget (the general rule is at least 3; a relaxation would be a recorded override in `STANDARDS.md`).
- The one-channel definition, the instruction text, the visibility rule, the success radius, and whether a DAgger stage is used.
- Whether any of the checks above is run, and on what machine.
