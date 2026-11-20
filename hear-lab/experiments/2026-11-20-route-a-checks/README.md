# Route A bring-up checks

Design in `PLAN.md` (with the 2026-11-20 amendment), parameters in `config.yaml`, the run record with exact numbers in `LOG.md`. Checks 1 and 3 ran on Vast.ai on 2026-11-20 (instance 61678754, 1x RTX 3090, 0.5354 USD/h, 0.95 h; account credit down 0.957 USD in total). A measured-HRTF follow-up to check 1 ran on Vast.ai on 2026-11-20 (instance 61700406, 1x GTX 1660 Super, CPU work only, 0.0692 USD/h, 0.053 h, about 0.004 USD). Check 2 has not run: the lead chose one H200 and the real VGGT-Omega file on 2026-11-20; its design amendment is next.

Commit run: 9144fe8cdad016cf0c01c1f68cf90837f68bb8e0 (code), plus the left-ear diagnostic script dfc332b. Raw outputs are kept out of git until the InteriorGS terms are read. The first set was lost locally on 2026-11-20 (see `LOG.md`); the check 1 outputs were regenerated identically on rental 3 and sit with the follow-up outputs in `~/Desktop/hear-lab-route-a-results/` on the experiment lead's machine; the 4 overlays and 20 check 3 review images were recovered to `~/Desktop/hear-lab-route-a-images/`; the check 3 per-episode CSVs need a GPU rerun.

## Results

**Check 1, sound engine and geometry.** Not passed by the PLAN's rule (8 PASS, 4 FAIL). What the run shows (`verified`):
- RLR-Audio-Propagation at 4fd446b loads on Ubuntu 22.04 (glibc 2.35) through the ctypes binding and returns one-channel and two-ear impulse responses for a real InteriorGS scene. First arrival matches the straight-line distance within 0.33 ms. One impulse response takes 0.02 to 0.23 s on one CPU thread depending on the settings.
- The left-ear-first test fails in both settings sets. The diagnostic shows why: the library's default head model, as configured, gives a strong, correctly signed level difference between the ears but almost no time difference (within plus or minus 0.09 ms at every angle in free field). So the two-ear audio from these settings carries a level cue and essentially no timing cue. Follow-up (2026-11-20, rental 3): with a measured HRTF file (MIT KEMAR, normal pinna) loaded through `RLRA_SetListenerHRTF`, the time difference becomes realistic (0.73 ms at 90 degrees, left ear first for a source on the left) and the check 1 left-ear test passes in both settings sets (left ear leads by 0.68 ms). The check 1 result with the default head model stays a FAIL. Which head model route A uses is for planning to decide. A higher spherical-harmonic order was not tried.
- Floor plans line up with GN-Bench's occupancy maps by eye in all four scenes (no flip, offset or scale error; looked at by the experiment lead agent, not yet by the user). The pre-set wall-share bar (0.9, occupied = value 0) passed in 2 of 4 scenes (0.928, 0.927) and failed in 2 (0.870, 0.857); under GN-Bench's "not passable" reading all four are 0.927 or more. The misses sit on grey unknown border cells and in openings.
- Every goal `label_ins_id` was found in `labels.json` (four scenes).

**Check 3, visible starts.** Passed by the PLAN's rule, provisionally: the by-eye review of the 20 images was done by the experiment lead agent and waits for the user. With the proposed depth rule (tolerance 0.15 m): test_unseen 1114 of 5000 starts have the target visible, 3886 not; the train_seen sample 351 of 1000 visible, 649 not. The split barely moves with the tolerance (0.05 m and 0.30 m in `LOG.md`). No goal id was missing. The GN-Matrix goal position sits a median 1.26 m (test_unseen) from the target object's box (horizontal distance to the box, zero inside it), with a quarter of episodes more than 3.6 m away. Limits: in 39 of 939 scenes a mirrored frame fits the object boxes better than the one used, so the flags of those 233 test_unseen and 41 train_seen episodes may be wrong; no test_unseen path is shorter than 3.01 m, so the path-based within-3 m flag is empty there.

**Check 2, timed training step.** Not run. 48 real train_seen episodes are rendered with GN0 as released and wait on the check 2 machine for conversion.

## What these results do not show

Nothing here is a result about audio helping navigation. The visibility rule is a proposal for this count; the rule for the paper is set later. Whether the missing timing cue matters for the research question has not been tested.

## LOG entries
`LOG.md`, 2026-11-20 entries (set-up, worker code, rental 1 and check 1, diagnosis and check 3).
