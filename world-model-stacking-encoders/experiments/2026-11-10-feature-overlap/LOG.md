# Experiment log: one-way feature overlap (containment) score

Step-by-step record of running this experiment (Jobs, errors, reruns, exact numbers). Append-only, newest at the bottom. The direction's `../../LOG.md` keeps short summaries that point here. Entry template: `_standards/04-documentation.md`.

## 2026-11-10 [agent, worker, planning]
**Goal:** write the experiment design for the one-way overlap score (direction PLAN Next 5, candidate measure).
**Did:** wrote `PLAN.md`, `README.md` and this stub; revised the PLAN after the round-1 review on PR #14. No code, no run, no cluster action, no number computed on real features. Four stored values for the DINOv3 + C-RADIOv4 pair were seen by the writer by accident; direction only is recorded in `PLAN.md` ("What already bears on the prediction").
**PR:** #14 (opened by the lead).
**Found:** nothing (design only).
**Test-set use:** none. (Per run, record here: invocation, fold, feature type, number of test evaluations; see `PLAN.md` "Test-set use".)
**Next:** the user's sign-off on `PLAN.md`, including which score is primary (open question 1); then code and tests by PR; then a CPU Job on the user's go.

## 2026-11-10 [agent, worker, planning] decision recorded
**Did:** recorded the user's sign-off and decision (lead's chat, 2026-11-10): primary score P = S2, C(B in D), fixed before any run; the design is signed off. Consequences in `PLAN.md` ("Consequences of P = S2"): headroom applies, no width-matched runs, 26 invocations. No gate, threshold or rule changed. No results opened.
**Test-set use:** none.
