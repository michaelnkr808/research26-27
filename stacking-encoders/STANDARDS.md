# Stacking Encoders — Topic Standards

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
- Compute budget: **no GPU-hour cap** (Nautilus is free; see override below). All GPU work follows `../_standards/08-compute-nautilus.md`. Specifically:
  - Every reportable run is a **Job** that runs a real command and exits. **Never `sleep infinity`, or any trailing `sleep`, in a Job** (ban risk).
  - Interactive pods are for short debugging only, and are deleted at session end (check `kubectl get pods -n aria-world`).
  - Utilization above 40% of requested GPUs; requests and limits within 20% of each other; start on 1 GPU.
  - **Gate (stop and ask):** more than 8 GPUs at once, or a sweep of more than 20 Jobs.
  - Log estimated vs. actual GPU-hours per run anyway, for planning.
- Checkpoint storage location: TBD. Personal PVC is only 20 GB, so cached features and checkpoints probably need CephFS or Ceph S3 (speculative sizing).

## Conventions
- Notation or terminology:
- Figure and table conventions:
- Target output format or venue:

## Overrides of general standards
| General rule | Override | Reason |
|---|---|---|
| `07` Compute and cost: "Each direction sets a compute budget: GPU-hours, API dollars, or both." | Budget is a concurrency cap (8 GPUs / 20 Jobs) plus the `08` utilization rules, not a GPU-hour total. | NRP Nautilus has no per-hour cost to us. The real risk is idle or under-utilized GPUs in the shared `aria-world` namespace (account at 1/3 idle strikes). Agreed with user 2026-10-16. |
