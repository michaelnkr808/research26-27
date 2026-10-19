# 002 — Feasibility tally (1 suite, 5–8 encoders, no training) before any proxy-policy training
- **Date:** 2026-10-19
- **Status:** accepted (user, 2026-10-19)
- **Context:** PhD mentor feedback (2026-10-19, see `writing/hypotheses-draft.md`):
  - Define the hypothesis and "complementarity" first.
  - Start with 1 dataset and 5–10 encoders; check CKA and tally similarities and differences by hand.
  - Position against prior ensemble-transferability work (agostinelli2022).
  
  The user wants to publish. Decision 001 (proxy-policy ground truth) is still the plan for the main study, but training a proxy across many combos and seeds is the expensive step. Its value depends on a hypothesis we haven't fixed yet.
- **Options considered:**
  1. Go straight to proxy-policy training on the 8 cached-feature encoders (decision 001 as is).
  2. **Feasibility first:**
     - Diagnostics only (CKA, gain(B|A), gain(A|B), bootstrap CIs) on libero_goal for DINOv3 + up to 7 partners.
     - Tally them against the 7 two-encoder deltas published in liu2026what Table 10.
     - Then decide the hypothesis and whether the proxy study is worth running.
  3. Broaden first: all 4 LIBERO suites for every encoder before any comparison.
- **Decision & rationale (proposed):** Option 2.
  - It is what the mentor asked for, needs no policy training, and reuses the code and features already in place.
  - It answers the cheapest form of the question first: does any of our cheap measures even get the sign of D+R (−5.4) and the rough ordering?
- **Consequences:**
  - Decision 001 still holds but is sequenced after this tally (see the note added there).
  - Pending go-aheads that serve this plan:
    - DINOv3 extraction (needed: D is the base of all 7 pairs).
    - SAM 3 and VGGT-Ω extraction.
    - The CPU diagnostics rerun with CIs.
  - The 4-suite storage work (CephFS PVC) is not needed until the proxy study. It is still useful if we extract many encoders.
  - The 7-pair tally has no statistical power: single-checkpoint labels, and a different data regime (LIBERO-Plus vs clean libero_goal). It's a sanity check, not evidence for a hypothesis.
