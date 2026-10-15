# 03 — Reproducibility and Data

**Goal:** someone else, or you in six months, can rerun any reported result from the files in the folder.

## Data
- **Raw data is read-only.** Keep it in `data/raw/` and never edit it in place. All cleaning happens in code that writes to `data/processed/`.
- **Record provenance.** Each dataset gets a `data/README.md` entry: source, access date, version, license, and any filtering applied.
- **Large or sensitive data stays out of version control.** Store a pointer instead: a path, a download script, or a checksum.
- **Never mix splits.** Fix the train/val/test split (or its equivalent) once, record how it was made, and don't touch test data during development.

## Code
- **All results come from code that is checked in.** Don't keep numbers produced by an unsaved notebook cell or one-off terminal commands.
- **Pin the environment.** Keep a `requirements.txt` / `environment.yml` / `pyproject.toml` with versions, and record the hardware when it matters.
- **Set and log random seeds.** Report results across multiple seeds wherever randomness matters.
- **Use version control (git).** Commit before every reported experiment, and record the commit hash in the experiment's record.
- **Notebooks are for exploration.** Anything that produces a reported result moves into a script or a cleanly rerunnable notebook.

## Experiments
Each experiment goes in its own folder, `experiments/YYYY-MM-DD-<short-name>/`, which contains:
- `config.*`: every parameter, so nothing is set by hand on the command line without being recorded
- `README.md`: hypothesis, command to run, commit hash, outcome, and a link to the matching `LOG.md` entry
- `results/`: raw outputs, metrics, and figures

**Never overwrite results.** A rerun gets a new folder, or a suffix such as `-v2`.
