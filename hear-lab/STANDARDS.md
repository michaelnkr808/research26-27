# HEAR Lab — Topic Standards

> These rules extend the general standards in `../_standards/`. A rule that *relaxes* a general rule must name it and say why.

Entries marked **open** are not decided yet. An open entry is filled by a pull request once the lead (and the PI where it applies) has decided it. Nobody fills one in by guessing.

## Scope
- **Research question:** What benefits does a world action model gain from being given spatial audio? (Full wording in `README.md`.) The specific question is in `decisions/001-research-question.md`.
- **In scope / out of scope:** partly set by `decisions/001`, `decisions/002` and `decisions/003`: simulated navigation with WNM-3D is in scope, in WNM-3D's own simulator with an added sound engine; the encoder comparison and long-form audio are deferred, and predicting future sound is set aside. The rest is **open**, to be set with the PI together with "what done looks like".

## Domain sources
- Key venues, journals, or conferences: **open**, filled from the literature survey.
- Canonical references or textbooks: **open**, filled from the literature survey.
- Trusted datasets or benchmarks: RMS-AQA (`chen2026rmsaqa`) was to be the first benchmark the direction handles, as the PI's first task; the lead parked it on 2026-11-18 (`PLAN.md`, Parked). Others: **open**, filled from the literature survey.
- Sources to treat with caution: **open**.

## Methods and evaluation
- Standard baselines: **open.** Whatever model is chosen, the same model without the audio input is a required baseline, so that any gain can be tied to the audio and not to other changes. Decision `001` adds the same model with one-channel audio as a second required baseline, so that any gain can be tied to the spatial part of the audio.
- Metrics (and why): **open.**
- Required statistical tests or number of runs: the general rule applies (`../_standards/07-ml-practices.md`: at least 3 seeds for any reported comparison).
- Known pitfalls in this field (leakage, confounds, etc.): **open**, filled from the literature survey.
- **Faithful experiments.** When reproducing a paper, stay as close to the original method as compute allows. No stand-in model, smaller dataset, or shortcut is used without the owner's OK first, and any that is used is named in the experiment's `README.md` with how it could change the result. A cheap test that does not represent the real experiment is not run in its place.

## Tools and environment
- Languages, libraries, hardware: **open** as a direction-wide list. Until it is set, each experiment records its own: the pinned repository commits, the checkpoint and dataset revisions and their checksums in its `config.yaml` and setup script, and the machine of each rental in its `LOG.md`.
- **Organized as in the stacking-encoders direction.** The planning lead asked the lead where a second copy of the raw outputs should be kept; the lead, 2026-11-23, in the planning window: "can we just organize it the same we do for the vis-lab" (`LOG.md` 2026-11-23 [agent, lead, planning], second entry). The entries from here to "Checkpoint storage location", the `data/README.md` and shared-code lines under "Conventions", and the naming override are the planning lead's reading of that answer. They are taken from `../world-model-stacking-encoders/STANDARDS.md` ("Tools and environment") and from what that direction keeps in git (its `experiments/*/results/`, `data/README.md`, `compute/` and `src/`), and are accepted by the lead's merge OK for the pull request that added them. Copied from that direction: local logs as the tracker, text results committed under `results/`, a ledger in `compute/`, the `data/README.md` form, shared code in `src/`, and snake_case Python file names (its practice; it lists no override for it). Hear-lab's own additions, not found there: the per-run `pip freeze` lists, the full-run manifest with a regenerate command, the list of what is not committed (that direction does commit some arrays and images), and the check for tokens. Where Vast.ai differs from the cluster, the entry says so.
- Environment file: no single direction-wide file (stacking-encoders keeps one `requirements.txt` for one experiment and pins in its Job files). Each experiment pins its own environment in its setup script and `config.yaml`. The full installed-package list of every run (`pip freeze`, plus the commit of each cloned repository) is saved with the run and committed under the experiment's `results/<run>/`.
- Experiment tracker (W&B / MLflow / local logs): local logs. No W&B or MLflow is used. Each run's config, source commit and metrics go into result files and logs under the experiment's `results/<run>/`, and the step-by-step record goes into the experiment `LOG.md` (`../_standards/04-documentation.md`).
- Rentals ledger: `compute/vastai-rentals.csv`, one row per rental: experiment, run, GPU, hourly price, start, end, hours, total cost, and the estimate beside the actual. Kept by the experiment lead, read from the `vastai` tool or the account's billing page, never estimated. It is the counterpart of the stacking-encoders `compute/nautilus-job-times.csv`, counting dollars against the 200 where that file counts queue hours. The per-run cost line in the experiment `LOG.md` (below, "Vast.ai") stays.
- Results storage:
  - Every run's outputs go in `experiments/<experiment>/results/<run>/` in the main checkout, one folder per rental or run. From 2026-11-23 a run folder is named `<what>-v<N>` (for example `main-v1`, `timing-v1`, `check2-v1`): a rerun of the same thing gets the next `-v<N>`, and a new kind of run gets a new `<what>` (the experiment lead's scheme). Folders made before that keep their names. Nothing is kept on the Desktop or in another folder outside the repo (the lead's request to the experiment session, `LOG.md` 2026-11-23 [agent, lead, planning], finding (3)). A rerun gets a new folder; a results folder is never renamed or overwritten (`../_standards/03-reproducibility-and-data.md`).
  - **Committed to git:** the text records a conclusion rests on. That is the per-episode records, the summaries and tallies, the logs, the config as run, the environment lists and the checksum manifests. GitHub is their second copy.
  - **Not committed:** checkpoints and model weights; arrays (`.npy`, `.npz`, already excluded by the repository `.gitignore`); video and audio renders; scene images and anything copied from the InteriorGS dataset (geometry, object boxes, floor plans); and the Vast.ai instance record (`instance.json`, `*-instance.json`), which holds the instance's Jupyter token, SSH host and port, and addresses (the image tag, offer, price and host reliability are copied from it into the experiment `LOG.md`). The repository is public (`gh repo view`, read 2026-11-23), so a committed file is a published file. On the InteriorGS material the experiment lead recommended, in its own window on 2026-11-23, committing scores and per-episode tables and keeping scene images, rendered audio and anything copied from the dataset out of the repository, released only as paper figures with a citation; the lead answered "yes sure thats fine" (the experiment session's report to the planning lead, not seen by the planning lead). The terms themselves are recorded in `data/README.md`. These files stay in the same `results/<run>/` folder, excluded by a `.gitignore` in `results/` that names the file types and no longer excludes the whole folder.
  - The files that are not committed exist in one copy, on the lead's machine. This is the same position as the stacking-encoders large files, which exist in one copy on a cluster volume. If they are lost they are regenerated by running again, at the cost of a rental (it happened once: `LOG.md` 2026-11-20 [agent, experiment], the entry "raw outputs lost and regenerated"). To make a loss visible, the committed manifest lists every file of the run, committed or not, with its size and sha256, and the experiment `README.md` gives the command that regenerates the ones not committed.
  - Before a results file is committed it is checked for tokens, keys, email addresses and absolute home paths. A file that contains one stays out of git and stays listed in the manifest.
  - Outputs are copied from the rented machine straight into the main checkout and checked against the manifest before the instance is destroyed. They are never kept only inside a worktree: removing a worktree deletes its ignored files (same `LOG.md` entry).
- Checkpoint storage location: no checkpoint has been trained yet. Released checkpoints are downloaded onto the rented machine for each run and are not kept. Where trained checkpoints are kept is **open** and is set before rung 3 is planned: a Vast.ai instance keeps nothing once it is destroyed, and the stacking-encoders answer (a Nautilus volume) needs the cluster. Checkpoints never go in git (`../_standards/05-ai-assisted-work.md`).

### Where experiments run
Two places are available. **Vast.ai comes first.** Every run is planned for Vast.ai by default. The Nautilus cluster is used only when the lead says so, or when the experiment lead has explained why it fits a particular run better and the lead agreed.

Only the experiment lead session (`hear-lab-experiment`) rents, submits, or manages compute.

**Vast.ai**
- Compute budget: 200 US dollars of credit on the HEAR Lab team account. The planning lead read this balance from the account with the `vastai` tool on 2026-11-15 (`verified`), and the lead said the same day that it is the whole compute budget for now. The lead, 2026-11-19: work within this credit before asking for more.
- Every rental needs the owner's OK first, as set out in "Rules for every agent" in the workspace `CLAUDE.md`. Before asking, the experiment lead states the machine (GPU, memory, disk, the host's reliability score), the hourly price, the expected hours, and the total. The time estimate comes from a real timed batch. One OK covers one rental.
- On-demand instances by default. An interruptible instance, which can be taken away mid-run, needs the owner's OK.
- Hosts can go offline, and destroying an instance deletes everything on it. Long runs save checkpoints, and results and logs are copied off the instance and checked before it is destroyed.
- A stopped instance still bills for its disk. An instance is destroyed once its work is saved, unless the owner says to keep it.
- No session stops or destroys an instance it did not create.
- The experiment's `LOG.md` records, for each run: GPU type, hourly price, hours, and total cost, with the estimate next to the actual (`../_standards/07-ml-practices.md`, "Estimate before launching").
- API keys and other tokens never go in the repo or in chat.

**Nautilus**
- `../_standards/08-compute-nautilus.md` applies in full to any run that uses the cluster, and is read before any cluster work. It does not apply to Vast.ai runs.
- Every new cluster Job needs the owner's OK first, as set out in "Rules for every agent" in the workspace `CLAUDE.md`.
- The namespace is shared with other directions in this repo. No session deletes a Job it did not create.
- PVC + storage class, image, allowed GPU types: **open**, set when the first cluster run is planned.

## Conventions
- Notation or terminology: working definitions are in `README.md` until the literature survey replaces them.
- Figure and table conventions: **open.**
- Data provenance: `data/README.md` has one entry for every dataset, released checkpoint and third-party asset a run has used (source, access date, version or revision, licence, filtering applied, where it is stored), in the form of the stacking-encoders `data/README.md` (`../_standards/03-reproducibility-and-data.md`). An entry is written before the first run that uses the item; for the items already used it is owed now.
- Shared code: code that more than one experiment uses lives in the direction's `src/`, as in stacking-encoders. An experiment's own `src/` holds what only that experiment uses.
- File names: Python source files and import packages are snake_case (override below). Files written by a third-party tool keep the names the tool gave them.
- Target output format or venue: a paper for Interspeech. Paper submission deadline 2027-02-09 and paper update deadline 2027-02-16, from the official Interspeech 2027 dates page (read 2026-11-15; the lead had given the end of February). The paper format is not checked: the call for papers was not open on 2026-11-15.

## Overrides of general standards
| General rule | Override | Reason |
|---|---|---|
| `04` Naming: "Folders and files: `kebab-case`, with no spaces." | Python source files and import packages use snake_case (for example `check_settings.py` and `rung1_hook/` in `experiments/2026-11-21-rung1-wnm3d-baseline/src/`). Output files written by a third-party tool keep the tool's names. Everything else stays kebab-case. | Python cannot import a module whose name has a hyphen, and PEP 8 names modules in snake_case (`recalled`). The stacking-encoders direction does the same in practice (`../world-model-stacking-encoders/src/fastwam_data.py`), without a listed override. Proposed by the planning lead after the 2026-11-23 audit; accepted by the lead's merge OK for the pull request that added this row. |
