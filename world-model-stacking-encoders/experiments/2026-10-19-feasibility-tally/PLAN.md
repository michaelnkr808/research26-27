# Plan: feasibility tally on libero_goal (decision 002)

*Reconstructed 2026-11-03 from `README.md` and the choices fixed before running in `LOG.md` 2026-10-19 (e). Not a new plan.*

**Question:** Before training anything, do cheap frozen-feature measures (CKA similarity, and gain: how much a partner B improves an action probe over DINOv3 alone) flag the pair the paper found harmful, DINOv3 + C-RADIOv4?

**Design:** DINOv3 paired with SigLIP 2, C-RADIOv4, SAM 3, VGGT-Ω on libero_goal. Same demo-level held-out split as the pipeline pilot, ridge strength tuned per probe on validation, 95% demo-level bootstrap CIs. Global features primary, mean secondary. Compared against the paper's Table 10 deltas. No hypothesis chosen or argued (user instruction).

**What would count:** a measure that ranks C-RADIOv4 as the worst partner. With 4 pairs this is a sanity check, not evidence.

**Status:** done 2026-10-21. Neither measure flags C-RADIOv4. Results in `README.md`.
