# Research Log


## 2026-10-27 [agent] (direction created)
**Goal:** Set up this folder for a candidate direction drafted as a proposal for the PhD mentor (user request).
**Did:** Copied `_template/`. Moved the proposal draft here from `../stacking-encoders/writing/proposals-2026-10-27/` (now `writing/proposal-2026-10-27.html`; the Google Doc link is in README). Filled README (draft question, plain-language idea, reading list), STANDARDS (scope, sources, baselines, metrics, pitfalls; compute budget not set), and `literature/references.bib`.
**Found:** nothing run. Background and paper checks are in `../stacking-encoders/LOG.md` (2026-10-26, 2026-10-27, 2026-10-27 (b)). References were checked against arXiv abstract pages; only the papers marked in each bib note were read further.
**Next:** User reads the reading list in README and decides whether this goes to the mentor. Before any experiment: confirm the question, set a compute budget in STANDARDS, write paper notes in `literature/notes/`.
**Open questions:** see the proposal's Open questions section.

## 2026-10-27 (b) [agent] (one-page version)
**Did:** User asked for shorter proposals (the full draft was about 1,700 words). Wrote `writing/proposal-short-2026-10-27.md` (about 600 words, same six sections, no confidence labels; every number is one already checked in the full notes). Saved the long version as a separate Google Doc ("full notes") and rewrote the original Google Doc to the short version through the Google Docs connector (guarded by revision ID; read back and checked: headings, numbered lists, links). Links in README.
**Next:** unchanged.

## 2026-10-27 (rename) [agent]
**Did:** Folder renamed by the user from `stacking-encoders/` to `world-model-stacking-encoders/` (2026-10-27). Repo paths in README/STANDARDS/references were updated; older entries above keep the old path (append-only). Cluster resource names (PVC `stacking-encoders-features`, Secret `stacking-encoders-hf`, ConfigMaps, Job names, `/persistent/stacking-encoders/` and `/mnt/features/stacking-encoders/` paths) are unchanged on purpose: they name real objects on Nautilus.

## 2026-10-27 (c) [agent] (paper summaries, prior-work check, proposal update)
**Goal:** User asked what the proposal is and what to read, then for summaries of DINO-WM, ALPS and AD-WM, then a check that nobody has already done this, then to add the results to the one-pager for tomorrow's mentor meeting.
**Did:** Fetched DINO-WM, ALPS (incl. appendix) and AD-WM HTML. Ran 5 web searches and fetched 10 candidate papers. Edited the one-page Google Doc in place (guarded by revision ID; read back and checked bullets, bold labels, links): two new Related-works bullets, one sentence under Action recoverability, a Baselines sentence in Experiment 1, four references. Synced `writing/proposal-short-2026-10-27.md`; added 4 entries to `literature/references.bib`. Full-notes doc not changed.
**Found (verified via fetch, not a careful read):**
- ALPS: no probe-to-planning correlation in main text or appendix; no action-controllability or inverse-dynamics measure. Appendix S2 says lowest single-frame readout error is "not always the most predictable under learned dynamics."
- AD-WM: elite regret needs a trained predictor; success tracks regret (rho ~0.86) but not factual error (Table III); no frozen-encoder comparison.
- Nilaksh et al. 2605.06388 already compute "action recoverability" (IDM on frozen latents, 6 encoders) next to CEM planning, family-level only. Scardecchia & Yang 2610.34663 (posted 2026-10-26) define a "linear unpredictable share" on frozen latents with an action-conditioned linear predictor, one autoencoder. Neither correlates the measure with planning across encoders.
- No paper found that tests a training-free, action-based score as a predictor of world-model planning success across frozen encoders. Search coverage is not proof of absence.
**Next:** Mentor meeting. Afterwards write `literature/notes/` for ALPS, AD-WM, Nilaksh, Scardecchia; consider the realized-cost side of elite regret as an extra training-free score (speculative).
**Open questions:** Should score 2 be dropped as a contribution and kept only as a tested baseline?
