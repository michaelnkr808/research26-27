# <Direction Name> — Topic Standards

> These rules extend the general standards in `../_standards/`. A rule that *relaxes* a general rule must name it and say why.

## Scope
- **Research question:**
- **In scope / out of scope:**

## Domain sources
- Key venues, journals, or conferences:
- Canonical references or textbooks:
- Trusted datasets or benchmarks:
- Sources to treat with caution:

## Methods and evaluation
- Standard baselines:
- Metrics (and why):
- Required statistical tests or number of runs:
- Known pitfalls in this field (leakage, confounds, etc.):

## Tools and environment
- Languages, libraries, hardware:
- Environment file:
- Experiment tracker (W&B / MLflow / local logs):
- Compute budget (GPU-hours / API $): TBD (user). All GPU work runs on NRP Nautilus under `../_standards/08-compute-nautilus.md` (that file is in progress as of 2026-10-16).
- Checkpoint storage location: TBD. Personal PVC is only 20 GB, so cached features and checkpoints probably need CephFS or Ceph S3 (speculative sizing).

## Conventions
- Notation or terminology:
- Figure and table conventions:
- Target output format or venue:

## Overrides of general standards
| General rule | Override | Reason |
|---|---|---|
