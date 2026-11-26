# Step 2: make the sound and check it

**Results:** none yet. Design and pass bar in `PLAN.md`, run steps in `RUN.md`, parameters in `config.yaml`, progress in `LOG.md`.

What it is: a ringing telephone at the target object of GN-Bench train_seen episodes, rendered with RLR and a measured head (MIT KEMAR) at the camera pose of every frame of the episode's reference-path video, with a check that the left/right timing cue points to the true side.

## Files
- `episode_order.txt`: the frozen episode order (ids only, 686, balanced visible / not visible), made by `src/pick_episodes.py` from the route A check 3 train_seen table (local only).
- `src/step2lib.py`: the pure-numpy parts (sides and azimuths, the ITD estimate, the counting rule, looping, clip synthesis, downmix, episode pick, tallies and the bar). Tests: `src/test_step2lib.py`, `src/test_render_smoke.py` (the whole audio path with a stand-in for the Linux-only library).
- `src/export_poses.py` (GN0 `bae` environment, GPU): the released reference-path renderer, recording each frame's pose.
- `src/render_audio.py` (Linux, CPU): impulse responses, clips, the direction check, timing. `src/draw_topdown.py`: the pictures. `src/summarize.py`: the verdict and tallies. `src/make_manifest.py`: the sha256 manifest.
- Shared code in `../../src/` (`rlr_binding.py`, `geometry.py`, `poses.py`, `gnbench_copy.py`, `c3lib.py`, moved there from route A on 2026-11-26).

## Outputs and what is in git
Runs go to `results/<what>-v<N>/` in the main checkout. `results/.gitignore` ignores everything except the manifest, the run settings, the per-episode tallies, the summary, the preflight, the export's settings and alignment check, and the environment lists (`PLAN.md` section 8). Clips, impulse responses, videos, poses, per-step records and pictures are InteriorGS-derived and stay on the lead's Mac only, in one copy. To regenerate them: `RUN.md` steps 2 to 5 at the same commit, into a new run folder; the committed manifest lists every file with its size and sha256 so a loss is visible.

## Licences and terms to settle before audio leaves the lab
- MIT KEMAR HRTF (`mit_kemar_normal_pinna.sofa`): the licence is **not verified**. MIT's KEMAR page says the measurements are "made available to the research community"; no formal licence was found (route A, 2026-11-20). Fine inside the lab; settle it before any rendered audio leaves the lab (a paper's supplementary audio, a released dataset). Cite Gardner and Martin 1994.
- SoundSpaces telephone ring: SoundSpaces calls its sounds "copyright-free"; the SoundSpaces code is CC-BY-4.0. No licence file for the sounds themselves was read.
- InteriorGS-derived outputs (clips, pictures): may appear only as paper figures with a citation (`../../data/README.md`).
- RLR-Audio-Propagation is CC BY-NC 4.0 (`../../data/README.md`).

<!-- After the run: command to run, commit hash, outcome, where the outputs are, and links to the LOG entries. -->
