# Research Log


## 2026-10-15 [agent]
**Goal:** Assess the proposed direction: frozen-feature similarity predicting WAM encoder-combo gains, and probing why general encoders beat specialists.
**Did:** Scaffolded the folder from `_template/`. Located the paper (arXiv 2610.13024v1) and extracted its setup and results via web fetch. Wrote a literature note and a BibTeX entry.
**Found:**
- verified (agent extraction from the arXiv HTML, not human-read): no quantitative complementarity measure in the paper; one training checkpoint per config; most composition deltas under 2 pts.
- speculative: those deltas may be within training-seed noise. Symmetric CKA may be the wrong tool; an asymmetric "residual information" measure fits "add B to A" better.
**Hypotheses (pre-registered, speculative):**
- H1: An asymmetric metric (action-relevant info in B not linearly predictable from A) correlates with combo gain better than CKA(A,B), and better than B's single-encoder score.
- H2: Specialists lose action-relevant info relative to their own initialization. Test on matched pairs, e.g. VGGT-Ω vs DINOv3, SAM 3 vs PE.
**Next:** Human reads the paper (Sec. 4.2, 4.4, App. E). Fill `STANDARDS.md` (compute budget especially). Check whether code/checkpoints are released. Do a prior-work pass on similarity-predicts-fusion (e.g. Cambrian-1, Theia, RADIO; all recalled, unverified).
**Open questions:** Compute available for multi-seed WAM training? Is a small proxy policy acceptable as ground truth?

## 2026-10-16 [agent]
**Goal:** Fold the compute setup into the plan.
**Did:** User confirmed they have read liu2026what, so I updated the note's read depth. Read the in-progress `../_standards/08-compute-nautilus.md` (not committed; I didn't edit it). Pointed this direction's `STANDARDS.md` at it.
**Found (speculative, from 08's GPU list):** Usable GPUs are mostly 24 GB-class cards (4090, A5000, 3090). A100 quota is 0 without an access form. Phase 1 (feature extraction, CKA, probes) fits single-GPU Jobs. Multi-seed retraining of a Wan 2.2-scale WAM likely doesn't fit, so ground truth probably needs a smaller proxy policy, the A100 form, or opportunistic Jobs that checkpoint often.
**Next:** User sets a GPU-hour budget and decides on the ground-truth model (proxy vs. full WAM). Then write `decisions/001-ground-truth-model.md`.
**Open questions:** Did the paper release code/checkpoints (user has read it)? Storage location for cached features?

## 2026-10-16 (b) [agent]
**Goal:** Set the compute rule and pick the ground-truth model.
**Did:** User agreed to a concurrency-based budget, now written into `STANDARDS.md` (override of `07`, with the no-`sleep` Job rule).
- Checked the Fast-WAM paper (arXiv 2603.16666v2, §4.1). verified: Wan2.2-5B backbone + 1B action expert = 6B params; 20k steps on LIBERO; training hardware not stated.
- Checked liu2026what via agent extraction (needs human confirmation). Its base WAM is "a video expert initialized from Wan 2.2" plus an action expert. It uses Fast-WAM's training datasets and follows "the direct-prediction paradigm of Fast-WAM" (App. B.1). It does not call its model Fast-WAM and gives no model size.
**Next:** User decides proxy-vs-WAM after a plain-language explanation. Then write `decisions/001-ground-truth-model.md`.
