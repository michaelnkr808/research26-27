# Research Log


## 2026-10-15 [agent]
**Goal:** Assess the proposed direction: frozen-feature similarity predicting WAM encoder-combo gains, and probing why general encoders beat specialists.
**Did:** Scaffolded the folder from `_template/`. Located the paper (arXiv 2610.13024v1) and extracted its setup and results via web fetch. Wrote a literature note and a BibTeX entry.
**Found:**
- verified (agent extraction from the arXiv HTML, not human-read): no quantitative complementarity measure in the paper; one training checkpoint per config; most composition deltas under 2 pts.
- speculative: those deltas may be within training-seed noise. Symmetric CKA may be the wrong tool; an asymmetric "residual information" measure fits "add B to A" better.
**Hypotheses (pre-registered, speculative):**
- H1: An asymmetric metric (action-relevant info in B not linearly predictable from A) correlates with combo gain better than CKA(A,B), and better than B's single-encoder score.
- H2: Specialists lose action-relevant info relative to their own initialization. Test on matched pairs, e.g. VGGT-Ω vs DINOv3, SAM 3 vs PE.
**Next:** Human reads the paper (Sec. 4.2, 4.4, App. E). Fill `STANDARDS.md` (compute budget especially). Check whether code/checkpoints are released. Do a prior-work pass on similarity-predicts-fusion (e.g. Cambrian-1, Theia, RADIO; all recalled, unverified).
**Open questions:** Compute available for multi-seed WAM training? Is a small proxy policy acceptable as ground truth?

## 2026-10-16 [agent]
**Goal:** Fold the compute setup into the plan.
**Did:** User confirmed they have read liu2026what, so I updated the note's read depth. Read the in-progress `../_standards/08-compute-nautilus.md` (not committed; I didn't edit it). Pointed this direction's `STANDARDS.md` at it.
**Found (speculative, from 08's GPU list):** Usable GPUs are mostly 24 GB-class cards (4090, A5000, 3090). A100 quota is 0 without an access form. Phase 1 (feature extraction, CKA, probes) fits single-GPU Jobs. Multi-seed retraining of a Wan 2.2-scale WAM likely doesn't fit, so ground truth probably needs a smaller proxy policy, the A100 form, or opportunistic Jobs that checkpoint often.
**Next:** User sets a GPU-hour budget and decides on the ground-truth model (proxy vs. full WAM). Then write `decisions/001-ground-truth-model.md`.
**Open questions:** Did the paper release code/checkpoints (user has read it)? Storage location for cached features?

## 2026-10-16 (b) [agent]
**Goal:** Set the compute rule and pick the ground-truth model.
**Did:** User agreed to a concurrency-based budget, now written into `STANDARDS.md` (override of `07`, with the no-`sleep` Job rule).
- Checked the Fast-WAM paper (arXiv 2603.16666v2, §4.1). verified: Wan2.2-5B backbone + 1B action expert = 6B params; 20k steps on LIBERO; training hardware not stated.
- Checked liu2026what via agent extraction (needs human confirmation). Its base WAM is "a video expert initialized from Wan 2.2" plus an action expert. It uses Fast-WAM's training datasets and follows "the direct-prediction paradigm of Fast-WAM" (App. B.1). It does not call its model Fast-WAM and gives no model size.
**Next:** User decides proxy-vs-WAM after a plain-language explanation. Then write `decisions/001-ground-truth-model.md`.

## 2026-10-16 (c) [agent]
**Goal:** Find out whether a full Fast-WAM-scale run is feasible on Nautilus.
**Did:** Read the Fast-WAM GitHub README. Ran read-only kubectl queries (`get nodes`, `get resourcequota`, `get pods`) after the user logged in.
**Found:**
- verified (README): LIBERO training is "a single node with 8 GPUs", DeepSpeed ZeRO-1. No memory, batch, or time figures given.
- verified (kubectl, 2026-10-16): Ready nodes include 4× 8-GPU RTX A6000 (48 GB), 1× 8-GPU A40 (48 GB), 17× 4-GPU L40 (48 GB), 16× 8-GPU RTX 3090 (24 GB), and 8× 8-GPU A100-SXM4-80GB. Node counts are totals, not free capacity.
- verified: `aria-world` quota for a100/h100/h200/gh200 is 0. No pods were running in the namespace.
- Users are forbidden from `get node`, so taints and current allocation can't be checked from kubectl.
- speculative: per-GPU memory for 6B training with ZeRO-3 is ~12 GB plus activations, so 8× A6000 or 4× L40 nodes are the realistic targets without the A100 form.
- unclear: quota entries named `low-priority-ban` / `high-priority-ban` (pods: 0) may block `opportunistic` priority. Not checked.
**Next:** User decides on the "smaller real WAM runs" plan. Then a 1-GPU timing/memory Job (no sleep).

## 2026-10-16 (d) [agent]
**Goal:** Measure Fast-WAM per-GPU step time and memory (`experiments/2026-10-16-fastwam-step-bench/`).
**Did:** Committed the bench script and Job manifest (1eaecbf). Server-side dry run passed. Created ConfigMap `stacking-encoders-bench-src` and Job `stacking-encoders-fastwam-step-bench` (1× `nvidia.com/rtxa6000`, 4 CPU, 40Gi, 3 h `activeDeadlineSeconds`, no sleep). Job started 2026-10-16T19:29:57Z.
**Found:** verified: the pod has been Pending for 61 min as of 20:31Z. FailedScheduling reports "Insufficient nvidia.com/rtxa6000"; no A6000 is free. No GPU is held while Pending.
**Open:** Waiting on user: keep the A6000-only request, or resubmit accepting any 48 GB card (L40/L40S/A40/A6000)? If it never schedules, the Job dies at the 3 h cap (~22:30Z). The Job and ConfigMap still need manual deletion either way.
**Next:** On a user decision, either wait or resubmit. After it completes: save logs to `results/job.log`, delete the Job and ConfigMap, confirm no pods remain.

## 2026-10-17 [agent]
**Goal:** Collect the Fast-WAM step benchmark and turn it into training-time estimates.
**Did:**
- 2026-10-16: deleted the L40 and A40 Jobs unrun (user-approved).
- Live Prometheus showed free A6000s only on CPU/RAM-starved nodes, so I resubmitted with 1 CPU / 16 Gi and built the model on the GPU (ff1b457). It scheduled on nrp-01.csumb.edu and ran 23:15:39Z–23:40:28Z.
- My watch missed the run: the connection dropped, and the next watch printed nothing. User spotted the completion. Logs pulled ~19 h later (TTL 24 h) to `experiments/2026-10-16-fastwam-step-bench/results/job.log`.
- Deleted the Job and ConfigMap; namespace confirmed clean.
- **Correction:** my 23:15Z claim that k8s-a6000-01 fit 1 CPU was wrong. I ignored init-container requests; corrected free CPU was 0.87.
**Found:**
- verified (1 run, 5 iters, A6000): B=1 0.808 s/step (1.24 samples/s, 23.9 GiB peak); B=2 1.525 s/step (1.31 samples/s, 29.0 GiB peak). 6.02B trainable params. VAE ~31% of the step.
- verified: B=4 crashed (CUDA-graph shape issue in Fast-WAM's compiled VAE, an artifact of varying B in one process). Gradient checkpointing is rejected by Fast-WAM's compiled core.
- speculative: full LIBERO run ~3–3.5 days on 8× A6000. 24 GB cards need ZeRO-3.
**Next:**
- Settle hardware: 8× A6000 (scarce, CPU-starved nodes) vs. ZeRO-3 on 8× 3090 vs. the A100 form.
- Before any multi-GPU run: one B per process, and measure the optimizer step + ZeRO overhead.
- Not a finding yet. Single run; needs user review before it goes in the README key findings.
**Open questions:** How does Fast-WAM fit micro-batch 16/GPU? Can VAE latents be precomputed to save ~30%?
