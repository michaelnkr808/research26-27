# Stacking Encoders: Topic Standards

> These rules extend the general standards in `../_standards/`. A rule that *relaxes* a general rule must name it and say why.

## Scope
- **Research question:** see `README.md` (draft until the hypothesis is chosen; candidates in `writing/hypotheses-draft.md`).
- **In scope / out of scope:**
  - **In:** frozen vision encoders; feature-level combination (concatenation); cheap pre-training diagnostics (CKA, conditional probe gain); LIBERO / LIBERO-Plus.
  - **Out for now:** training or fine-tuning encoders; output-averaging ensembles; real-robot evaluation; the specialist-vs-generalist question (H-D) unless it falls out cheaply.

## Domain sources
- Key venues, journals, or conferences: CoRL, RSS, ICRA (robot learning); NeurIPS, ICLR, ICML; CVPR, ICCV (transferability and model selection). arXiv preprints are common; flag them as not peer reviewed (`02`).
- Canonical references or textbooks:
  - liu2026what: the motivating paper; its tables are transcribed and checked in `literature/`.
  - agostinelli2022transferability: closest prior work on choosing model combinations without training.
  - For CKA: Kornblith et al. 2019 (recalled, not yet in references.bib; add before citing).
- Trusted datasets or benchmarks: LIBERO (4 suites; HDF5 demos), LIBERO-Plus (perturbation categories; the paper's evaluation). Feasibility work uses libero_goal only (decision 002).
- Sources to treat with caution:
  - liu2026what's composition numbers: one trained checkpoint per combo; the 3 seeds are evaluation seeds.
  - Encoder model cards: variants and resolutions are often unstated.
  - Anything recalled about related work (Cambrian-1, Theia, LogME, etc.) until it's read.

## Methods and evaluation
- Standard baselines (every comparison of predictors includes): (1) B's single-encoder score; (2) symmetric CKA(A, B); (3) the paper's objective-overlap reasoning, coded as a label before seeing results; (4) random ranking.
- Metrics (and why):
  - **Diagnostics:** debiased linear CKA (similarity); action-probe test R² and gain(B|A) (conditional task information, asymmetric).
  - **Predictor quality:** rank correlation (Spearman, or Kendall τ as in agostinelli2022) between predictor and measured combo gain, plus whether the sign of harmful combos is predicted. We use rank metrics because only the ordering matters for choosing combos.
- Required statistical tests or number of runs:
  - Demo-level bootstrap CIs (≥1,000 draws) on every probe number. Frames within a demo are correlated, so frame-level CIs are invalid.
  - Proxy-policy ground truth needs ≥3 training seeds per combo (`07`).
  - With 7 pairs, report the tally and don't claim significance.
- Known pitfalls in this field (leakage, confounds, etc.):
  - Treating low similarity as complementarity (mentor point 1).
  - Splitting probe train/test by frame instead of by demo (leakage).
  - Feature dimensionality differences, e.g. RADIO summary 2304-d vs 1024-d: concatenation changes probe capacity, so compare at matched dimensions or with regularization tuned on validation.
  - Single-checkpoint published labels.
  - Domain shift between clean libero_goal features and LIBERO-Plus labels.
  - Image orientation in LIBERO HDF5 (OpenGL convention; rotate 180°).

## Tools and environment
- Languages, libraries, hardware: Python 3.11.14; PyTorch 2.7.1 (CUDA 12.8 build on the cluster), h5py, transformers 4.57.6, timm, open_clip_torch, numpy 2.2.6, with the full pin list in `experiments/2026-10-27-proxy-calibration/requirements.txt` (versions as used on the Mac, 2026-10-28). One Job, the tally extraction Job, pinned `transformers==5.18.0` because SAM 3 is not in 4.57 (comment in `experiments/2026-10-19-feasibility-tally/extract-job.yaml`); it is not a direction-wide pin. Fast-WAM at commit `7faa711` is installed with `--no-deps`. Hardware: a Mac (CPU or MPS) for local tests and Nautilus GPUs and CPU nodes for Jobs (GPU model per run is in each experiment's LOG).
- Environment file: `experiments/2026-10-27-proxy-calibration/requirements.txt` (proxy-calibration code). For the earlier experiments there is no environment file; each Job yaml carries its own `pip install` pins (`experiments/*/job.yaml`, `extract-job.yaml`, `diag-*-job.yaml`), and the proxy round builds its environment with `src/setup_env.sh`.
- Experiment tracker (W&B / MLflow / local logs): local logs. No W&B or MLflow is used (Fast-WAM's wandb pin is left out on purpose; `experiments/2026-10-27-proxy-calibration/requirements.txt`). Each run's config, source commit and metrics go into result JSONs and pod logs under the experiment's `results/`, and the step-by-step record goes into the experiment `LOG.md` (`../_standards/04`).
- Compute budget: **no GPU-hour cap** (Nautilus is free; see override below). All GPU work follows `../_standards/08-compute-nautilus.md`. Specifically:
  - Every reportable run is a **Job** that runs a real command and exits. **Never `sleep infinity`, or any trailing `sleep`, in a Job** (ban risk).
  - Interactive pods are for short debugging only, and are deleted at session end (check `kubectl get pods -n aria-world`).
  - Utilization above 40% of requested GPUs; requests and limits within 20% of each other; start on 1 GPU. What NRP's rule is (rechecked 2026-11-07, `../_standards/08`, verified): the policy allows a user no more than 4 pods outside the ranges (GPU above 40% of the request, CPU 20-200%, RAM 20-150%) at once, the portal checks running pods every 10 minutes and skips pods younger than 30 minutes, and NRP says admins "will ban namespaces with under-utilized resources requests" and that namespaces that keep requesting more than they use "risk being banned". No source says what happens above 4 pods. No averaging window, further grace period, warning, or automatic kill is documented. Our own operating rule on this threshold is unchanged by this note. The 3-strike line in 08 is lab-reported, not in NRP's published policy; the user said 2026-11-06 to keep assuming 1 of 3 (speculative whether it is enforced).
  - **Gate (stop and ask):** more than 8 GPUs at once, or a sweep of more than 20 Jobs.
  - Log estimated vs. actual GPU-hours per run anyway, for planning.
- Checkpoint storage location: CephFS PVC `stacking-encoders-proxy` (1Ti, `rook-cephfs`), proxy training checkpoints at `/mnt/proxy/runs/<config>-s<seed>/ckpt.pt` and rollout records at `/mnt/proxy/rollouts/*.jsonl` (`experiments/2026-10-27-proxy-calibration/RUN.md` steps 10 and 11, `jobs/pvc-proxy.yaml`). Cached features are on that PVC and on `stacking-encoders-features` (see `data/README.md`). Checkpoints stay out of git. The personal PVC `michael-persistent` is only 20Gi (`experiments/2026-10-18-extract-libero-goal/job.yaml` line 5). CephFS is not archival (purged after 6 months unaccessed; `data/pvc-features-cephfs.yaml`).

## Conventions
- Notation or terminology: encoders by letter as in Table 10 (D=DINOv3, V=V-JEPA 2.1, L=SigLIP 2, S=SAM 3, G=VGGT-Ω, H=HRP, U=UniTok, R=C-RADIOv4). "Similarity" = CKA; "complementarity" = gain(B|A) on a named task target. Never use the two interchangeably.
- Figure and table conventions:
- Target output format or venue: a publication (user goal, 2026-10-19). Venue to be decided with the mentor.

## Overrides of general standards
| General rule | Override | Reason |
|---|---|---|
| `07` Compute and cost: "Each direction sets a compute budget: GPU-hours, API dollars, or both." | Budget is a concurrency cap (8 GPUs / 20 Jobs) plus the `08` utilization rules, not a GPU-hour total. | NRP Nautilus has no per-hour cost to us. The real risk is idle or under-utilized GPUs in the shared `aria-world` namespace (account at 1/3 idle strikes). Agreed with user 2026-10-16. |
