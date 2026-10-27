# World-Model Planning Horizon

**Status:** exploring. Candidate direction (proposal 3 of 3 drafted 2026-10-27 for the PhD mentor). Nothing run. The user is reading the key papers before deciding whether to take it to the mentor.
**Question (draft, not confirmed):** If a latent world model's predictions were perfect, how far ahead could it plan in each frozen encoder's embedding space, and does that "geometry horizon" explain why encoders differ in plannable range?
**What done looks like:** A per-encoder split of the planning-horizon limit into predictor error, embedding-geometry error and the scoring-rule limit, from an oracle experiment (real future embeddings scored by embedding distance), plus a cheap demo-based check that predicts the geometry part. Or a documented finding that geometry error is small.

## The idea in plain words
Latent world models only pick good actions for goals a few steps away. Planning Limits (arXiv 2609.39235) showed bigger predictors don't fix this, and that even a perfect simulator fails far from the goal, but its control scored by true physical distance, not embedding distance. We add the missing condition: real future frames, embedded and scored by distance to the goal embedding. Comparing the three conditions shows how much of the limit comes from the predictor, from the embedding's geometry, and from the scoring rule itself.

## Key findings so far
- None. Nothing has been run.

## Reading list (in order)
1. The Planning Limits of Latent World Models (arXiv 2609.39235): the measured limit, per-encoder ranges, and the exact-dynamics control this project extends.
2. Latent World Models Should Plan with Distances Along the Latent Manifold (NGC, arXiv 2610.25013): L2 distance fails far from the goal; graph-distance fix.
3. DINO-WM (arXiv 2411.04983): the planner setup.
4. AD-WM (arXiv 2609.30264): low prediction error does not mean good action ranking.
5. Cost-fix papers to skim: Monotone Planning Costs (2608.09073), TD-JEPA (2607.25337), Latent Projection (2610.26979).

## Key files
- `writing/proposal-short-2026-10-27.md`: one-page proposal (Google Doc: https://docs.google.com/document/d/1sKBxucnzdKOZNWzwJBmjCQ7ItSzjis-LZEnu3wLXHUg/edit)
- `writing/proposal-2026-10-27.html`: full notes (Google Doc: https://docs.google.com/document/d/1ieqcwt7FJyhLigUSYx809ItAz8lOE3kD8sOm0ldvhtU/edit)
- `STANDARDS.md`: topic rules
- `LOG.md`: research log
- `literature/references.bib`: references (checked at arXiv abstract level unless noted)
- Related directions: `../world-model-embedding-choice/` (same data and encoders), `../stacking-encoders/` (encoder loaders and probe code)
