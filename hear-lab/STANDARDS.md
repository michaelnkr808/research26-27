# HEAR Lab — Topic Standards

> These rules extend the general standards in `../_standards/`. A rule that *relaxes* a general rule must name it and say why.

Entries marked **open** are not decided yet. An open entry is filled by a pull request once the lead (and the PI where it applies) has decided it. Nobody fills one in by guessing.

## Scope
- **Research question:** What benefits does a world action model gain from being given spatial audio? (Full wording in `README.md`.)
- **In scope / out of scope:** **open.** To be set with the PI together with "what done looks like".

## Domain sources
- Key venues, journals, or conferences: **open**, filled from the literature survey.
- Canonical references or textbooks: **open**, filled from the literature survey.
- Trusted datasets or benchmarks: **open**, filled from the literature survey.
- Sources to treat with caution: **open**.

## Methods and evaluation
- Standard baselines: **open.** Whatever model is chosen, the same model without the audio input is a required baseline, so that any gain can be tied to the audio and not to other changes.
- Metrics (and why): **open.**
- Required statistical tests or number of runs: the general rule applies (`../_standards/07-ml-practices.md`: at least 3 seeds for any reported comparison).
- Known pitfalls in this field (leakage, confounds, etc.): **open**, filled from the literature survey.
- **Faithful experiments.** When reproducing a paper, stay as close to the original method as compute allows. No stand-in model, smaller dataset, or shortcut is used without the lead's OK first, and any that is used is named in the experiment's `README.md` with how it could change the result. A cheap test that does not represent the real experiment is not run in its place.

## Tools and environment
- Languages, libraries, hardware: **open.**
- Environment file: **open.**
- Experiment tracker (W&B / MLflow / local logs): **open.**
- Checkpoint storage location: **open.** Checkpoints never go in git (`../_standards/05-ai-assisted-work.md`).

### Where experiments run
Two places are available. **Vast.ai comes first.** Every run is planned for Vast.ai by default. The Nautilus cluster is used only when the lead says so, or when the experiment lead has explained why it fits a particular run better and the lead agreed.

Only the experiment lead session (`hear-lab-experiment`) rents, submits, or manages compute.

**Vast.ai**
- Compute budget: 200 US dollars of credit on the HEAR Lab team account, as read from the account on 2026-11-15 (`verified`). Whether more will be added is **open**.
- Every rental needs the lead's OK first, in the experiment lead's own session. Before asking, the experiment lead states the machine (GPU, memory, disk, the host's reliability score), the hourly price, the expected hours, and the total. The time estimate comes from a real timed batch. One OK covers one rental.
- On-demand instances by default. An interruptible instance, which can be taken away mid-run, needs the lead's OK.
- Hosts can go offline, and destroying an instance deletes everything on it. Long runs save checkpoints, and results and logs are copied off the instance and checked before it is destroyed.
- A stopped instance still bills for its disk. An instance is destroyed once its work is saved, unless the lead says to keep it.
- No session stops or destroys an instance it did not create.
- The experiment's `LOG.md` records, for each run: GPU type, hourly price, hours, and total cost, with the estimate next to the actual (`../_standards/07-ml-practices.md`, "Estimate before launching").
- API keys and other tokens never go in the repo or in chat.

**Nautilus**
- `../_standards/08-compute-nautilus.md` applies in full to any run that uses the cluster, and is read before any cluster work. It does not apply to Vast.ai runs.
- Every new cluster Job needs the lead's OK first, in the experiment lead's own session.
- The namespace is shared with other directions in this repo. No session deletes a Job it did not create.
- PVC + storage class, image, allowed GPU types: **open**, set when the first cluster run is planned.

## Conventions
- Notation or terminology: working definitions are in `README.md` until the literature survey replaces them.
- Figure and table conventions: **open.**
- Target output format or venue: **open.**

## Overrides of general standards
| General rule | Override | Reason |
|---|---|---|
| None | | |
