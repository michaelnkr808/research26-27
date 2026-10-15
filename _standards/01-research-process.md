# 01 — Research Process

## Before starting work
- **Write the question down.** Every direction's `README.md` starts with one or two sentences stating the research question. If you can't write it, the work isn't scoped yet.
- **Define what "done" looks like.** Name the result, artifact, or decision that would answer the question, and the evidence that would count.
- **Survey prior work first.** Do a literature/prior-art pass before building anything (see `02`). Record what already exists and how your work differs.
- **State hypotheses before running experiments.** Write the expected outcome in `LOG.md` *before* you see results, so hindsight can't rewrite the prediction.

## While doing the work
- **Start small, then scale.** Begin with the simplest baseline or smallest experiment that could fail informatively.
- **Change one thing at a time** when comparing approaches, or explicitly design for multiple factors.
- **Always compare against a baseline.** A number on its own is not a result.
- **Keep timeboxes.** When a sub-task goes well past its expected time, stop and log why before continuing.

## Drawing conclusions
- **Separate observation from interpretation.** "X went up 12%" is an observation. "Because of Y" is an interpretation and needs its own evidence.
- **Report uncertainty**: variance across runs/seeds, sample sizes, confidence intervals, or at minimum the number of trials.
- **Look for the boring explanation first**: bugs, data leakage, randomness, measurement error.
- **Every claim in a writeup traces back** to a log entry, an experiment folder, or a cited source.

## Status labels
Use these labels in `README.md` and `LOG.md`:
`exploring` → `scoped` → `in-progress` → `analyzing` → `writing` → `done` / `paused` / `abandoned` (always with a reason).
