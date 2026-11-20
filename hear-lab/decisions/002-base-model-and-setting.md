# 002 — Base model WNM-3D, setting simulated navigation
- **Date:** 2026-11-19
- **Status:** proposed 2026-11-19, on the lead's "go" in the planning window to a plan that listed "WNM-3D as the base" and "navigation as the setting" among the decision records. It becomes accepted on the lead's merge OK for this pull request.
- **Context:** The direction needs a world action model to add audio to, and a setting to test it in. The lead's words, from `LOG.md`:
  - Setting, 2026-11-16: "why do we need a robot arm doing manipulation with spatial audio? i feel like that doesn't make sense for a task where audio would be useful ... i feel that adding audio to a WAM would help moreso with navigation over like the tasks that a robot arm would do".
  - Base family, 2026-11-16: "id prefer to work with wan 2.2".
  - Base model, 2026-11-18, after asking whether to build on FastWAM: "it makes more sense to do WNM-3D since its a navigation model"; and, after the simulator gap was explained again: "WNM-3D seems fine".
  - The PI's written feedback of 2026-11-19 takes WNM-3D as the base (the planning lead's reading of "if you can use WNM-3D's own simulator for simulating audio-visual training data, then we will not use SoundSpaces 2.0"). The PI gave no answer on the alternative the lead was to name, UniWAM.
  - The survey found no public real-robot manipulation dataset with spatial audio, and found spatial audio with video and actions only for simulated navigation (`literature/survey/2026-11-16-summary.md`).
- **Options considered** (facts from `literature/survey/2026-11-16-summary.md`, `2026-11-18-novelty-and-simulator-gap.md` part 2, and `LOG.md` 2026-11-18; `verified` there, nothing reproduced):
  1. **WNM-3D.** Navigation, built on Wan2.2-TI2V-5B. Code with training scripts for two of its three stages and checkpoints are released; the released weights are for non-commercial research use. Its released training and evaluation run in its own simulator (GN-Bench on InteriorGS Gaussian-splat scenes), which has no audio, and the released checkpoints were trained only on InteriorGS data. Chosen.
  2. **UniWAM.** Wan2.2-5B, trained and evaluated in Habitat on HM3D, code and gated checkpoints released. The closest complete Habitat stack found, but built for a different task, with no audio and a different habitat-sim version from the one SoundSpaces 2.0's audio needs. The named alternative.
  3. **AstraNav-World.** Evaluated in Habitat, but no training script for its video part and no video weights were found.
  4. **FastWAM.** Wan2.2-5B plus an action expert, code and checkpoints released, but a robot-arm model (LIBERO, RoboTwin 2.0). The lead asked about it and chose WNM-3D instead.
  5. **Plain Wan 2.2 with an own action head.** Listed as a candidate in the summary; not pursued.
  6. **Cosmos 3.** Raised in the planning window on 2026-11-19 as the alternative whose base model takes and generates two-channel audio natively. Not a navigation model; its released robot policy does not use audio; whether its two channels carry direction was not checked. Not pursued.
- **Decision & rationale:**
  - **(a) Setting:** navigation in simulation, not robot-arm manipulation. The lead's reason is quoted above; the survey's finding on where spatial audio data exists points the same way.
  - **(b) Base model:** WNM-3D. The lead's reason: it is a navigation model. The planning lead's supporting reasons (`speculative`): its training code is released, and its training data format (LeRobot) is not tied to one simulator.
  - **(c) The realistic form of the work** (from the summary, not a new decision): start from a released checkpoint and train an added audio part. WNM-3D was trained on 16 H100, far above the direction's 200 dollar budget. Whether that form fits the budget is unknown until a real batch is timed.
  - Nothing here is a result.
- **Consequences:**
  - `README.md` "Open scope questions": the base model and the setting are marked decided.
  - WNM-3D's simulator has no audio, so the audio-visual training data and the closed-loop test have to be built one of two ways: add a sound engine to WNM-3D's own scenes (the PI's preference), or move WNM-3D to Habitat with SoundSpaces 2.0. That choice is open and will be decision `003`, after the two sizing memos (`PLAN.md`).
  - If WNM-3D is moved to Habitat, its released weights have never seen those scenes; how well they transfer is unknown.
  - The released weights are non-commercial, so anything built on them carries that limit.
  - UniWAM stays the named alternative if neither simulator route is workable for WNM-3D (the planning lead's suggestion, not decided).
