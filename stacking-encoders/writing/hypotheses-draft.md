# Hypotheses and scope (DRAFT for the user to choose from; not decided)

Written 2026-10-19 by an agent after mentor feedback. Nothing here is a finding. The user picks the hypothesis; this file lays out options.

## Mentor feedback (Slack, 2026-10-19, paraphrased; mentor hadn't read the paper yet)
1. Similarity metrics are a fine starting point. But **low similarity is not the same as complementarity**: different encodings don't necessarily add task-useful information. Define what "complementary" means and whether it helps.
2. **Define the hypothesis exactly first.** Then decide whether this is one experiment or a full paper. Start with **very small experiments: 1 dataset, 5–10 encoders**. Check what CKA gives, and compare and manually tally similarities and differences.
3. **Prior work exists** on selecting model combinations without training them (linked: agostinelli2022transferability). We must say how we differ, what WAMs introduce, and how our approach changes. "A simple transferability on a new task is usually credited less nowadays."
4. Meeting moved to Thursday. She wants the user's goals (publish? interests?) to make a plan.

User's goals (stated 2026-10-19): **wants to publish.** Hypothesis still to be chosen.

## Definitions (proposed)
- **Similarity** S(A, B): symmetric, task-agnostic agreement between two encoders' features on the same frames (debiased linear CKA). High S means the two encoders encode the frames alike.
- **Complementarity** C(B | A; T): task-relevant information in B that is not linearly recoverable from A, for task T.
  - Operationally, gain(B|A) = R²(probe on [A, B]) − R²(probe on A), on held-out demos, with demo-level bootstrap CIs. This is already implemented (`src/diagnostics.py`).
  - It is **asymmetric** (C(B|A) ≠ C(A|B)) and **task-dependent** (T = next actions; later, future-feature prediction).
- Why these differ: two encoders can be dissimilar (low S) because B encodes things T doesn't need, giving C ≈ 0. They can also be fairly similar overall (high S) while B still carries a small task-critical signal A lacks, giving C > 0.
  - Pilot (libero_goal, SigLIP 2-L vs C-RADIOv4, one pair, not a finding): CKA 0.72–0.94, yet gain(RADIO|SigLIP) = +0.069 vs gain(SigLIP|RADIO) = +0.008. One symmetric number can't express that asymmetry.

## Candidate hypotheses (pick one primary; the others become controls)
- **H-A (similarity fails):** Across encoder pairs, CKA(A, B) is not a reliable predictor of the success gain from adding B to A in a WAM. Its rank correlation with gain is no better than B's single-encoder score.
  - Easy to test, but a negative result alone is a weaker paper.
- **H-B (conditional task information predicts gains):** gain(B|A) on an action probe ranks "add B to A" combos by WAM success gain better than (i) CKA, (ii) B's single-encoder score, and (iii) the paper's objective-overlap reasoning.
  - This is the constructive version and the strongest publishable claim if true. *Recommended primary (agent's view).*
- **H-C (redundancy hurts feature fusion):** Adding a B that is highly similar to A *and* adds little conditional task information lowers success (e.g. D+R −5.4 in liu2026what), unlike output-averaging ensembles.
  - This is a mechanism claim that sets us apart from agostinelli2022, where averaging good members rarely hurts.
- **H-D (specialists discard task info; the original second question):** A specialist encoder (SAM 3, VGGT-Ω, HRP) carries less action-relevant information than the general encoder it was initialized from. Measured as probe R², on matched pairs (e.g. VGGT-Ω vs DINOv3).
  - A separate thread. Keep it out of the first paper unless it falls out cheaply.

## Smallest feasibility experiment (matches mentor point 2; no policy training)
- **Dataset:** LIBERO, libero_goal suite only (features for SigLIP 2-L and C-RADIOv4 already cached).
- **Encoders (5–10):** DINOv3 (D) as base, plus partners from liu2026what Table 10: SigLIP 2 (L), C-RADIOv4 (R), SAM 3 (S), VGGT-Ω (G), then V-JEPA 2.1 (V), UniTok (U), HRP (H) as loaders allow. That makes 5 now and 8 maximum.
- **Compute:** the CKA matrix, and gain(B|D) and gain(D|B) for each partner, with CIs. Then **manually tally** each pair: what's similar, what's different, which is larger.
- **Free labels, no training:** liu2026what Table 10 has the LIBERO-Plus success of D+B for B in {V, L, S, G, H, U, R}. That's 7 pairs with deltas vs D alone (84.0): +0.9, +1.8, +0.5, +0.4, +0.5, +2.1, −5.4.
- **Feasibility questions:**
  - (1) Does CKA order the 7 pairs anything like the deltas?
  - (2) Does gain(B|D)?
  - (3) Is D+R (−5.4) flagged by either?
- **Caveats (must be stated):**
  - n = 7 pairs gives no statistical power. This is a tally and a sanity check, not a test.
  - The labels come from one trained checkpoint per combo (evaluation seeds only).
  - Their numbers are on perturbed LIBERO-Plus after joint video–action training; our features come from clean libero_goal frames.
- **Outcome rule (proposed):** if neither metric gets D+R's sign or the rough ordering, rethink before building the proxy policy (decision 001). If one does, the proxy-policy multi-seed study becomes the paper's main experiment.

## Novelty statement (draft; needs the related-work check in `literature/notes/agostinelli2022-transferability-ensembles.md`)
> Prior transferability work (e.g. agostinelli2022) selects ensembles of fine-tuned models whose *outputs are averaged*, using label-space metrics that are additive across members. We study *feature-level* stacking of frozen vision foundation encoders inside a world action model. There, combining encoders can hurt, the encoders share no label space, and the useful quantity is *conditional* and asymmetric (what B adds to A for predicting actions or future features). We test whether a cheap conditional measure predicts closed-loop gains where symmetric similarity does not.

What "WAM introduces" (mentor's question), candidate answers to check:
- (a) Feature fusion into one model, not output averaging.
- (b) A control target (actions) plus a predictive target (future features). That gives two definitions of "task-relevant", and they may disagree.
- (c) Robustness under perturbations (LIBERO-Plus categories), so complementarity could be perturbation-specific.

## For Thursday (user prep)
- Goals: publish. Say which venue tier and timeline you have in mind (e.g. a workshop paper first vs a main conference).
- Bring: one primary hypothesis (H-B recommended), the definitions above, the 7-pair feasibility plan, and the draft novelty statement.
- Ask her: is the 7-pair tally against the paper's labels an acceptable first check, given that it's single-checkpoint? And which related work does she think is closest besides agostinelli2022?
