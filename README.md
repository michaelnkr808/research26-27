# Research 2026–27

This is the root of all research for the 2026–27 cycle. Each **research direction** gets its own subfolder. The files in `_standards/` hold the **general rules**, which apply to every subfolder.

## How the rules fit together

1. **General standards** (`_standards/`): apply everywhere, always.
2. **Topic standards** (`<subfolder>/STANDARDS.md`): add rules specific to one direction, such as domain sources, methods, metrics, and tools.
3. **Conflicts:** a topic rule can make a general rule *stricter* or more specific. It can only *relax* a general rule if it states which rule it overrides and why.

## General standards

| File | Covers |
|---|---|
| [`_standards/01-research-process.md`](_standards/01-research-process.md) | Going from a question to a claim: scoping, hypotheses, experiments, conclusions |
| [`_standards/02-sources-and-citations.md`](_standards/02-sources-and-citations.md) | Which sources to trust, how to cite them, how to verify claims |
| [`_standards/03-reproducibility-and-data.md`](_standards/03-reproducibility-and-data.md) | Data, code, environments, and experiment tracking |
| [`_standards/04-documentation.md`](_standards/04-documentation.md) | Research logs, decision records, naming, and file layout |
| [`_standards/05-ai-assisted-work.md`](_standards/05-ai-assisted-work.md) | Agent session protocol, truthfulness rules, verification gates, git conventions |
| [`_standards/06-integrity-and-ethics.md`](_standards/06-integrity-and-ethics.md) | Honesty, attribution, data ethics, and negative results |
| [`_standards/07-ml-practices.md`](_standards/07-ml-practices.md) | Evaluation hygiene, seeds, ablations, experiment tracking, compute budgets |
| [`_standards/08-compute-nautilus.md`](_standards/08-compute-nautilus.md) | NRP Nautilus cluster: workload types, GPU utilization policy, storage, data movement, Job template |

## Layout of a research direction

```
<direction-name>/
├── STANDARDS.md      # topic rules (copy from _template/)
├── README.md         # question, status, key findings so far
├── LOG.md            # dated research log (append-only)
├── decisions/        # one file per significant decision (ADR-style)
├── literature/       # paper notes + references.bib
├── data/             # raw/ (never edited) and processed/
├── src/              # code
├── experiments/      # one folder per run or experiment, with config + results
└── writing/          # drafts, figures, final outputs
```

To start a new direction, copy `_template/` to a new folder named in `kebab-case`, then fill in `README.md` and `STANDARDS.md`.
