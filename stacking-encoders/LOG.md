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

## 2026-10-17 (b) [agent]
**Goal:** Review the A6000 bench and pick the next direction.
**Did:** Re-checked the bench numbers against `results/job.log` (they match the 10-17 entry; no new run). Fetched the liu2026what project page. Wrote `decisions/001-ground-truth-model.md` (accepted by user). Updated the paper note.
**Found:**
- verified: no code or checkpoints for liu2026what. The page says R²-WAM predicts future encoder features through shared visual tokens. That is a different use of encoders than a feature-input proxy.
- verified: the composition values are in a figure image only. Exact per-config numbers need reading off the PDF figure or appendix tables.
**Decided:** Ground truth = small proxy policy on cached frozen features, scored by LIBERO/LIBERO-Plus rollouts, multi-seed. First test: does the proxy reproduce the paper's orderings? User is not filing the A100 form for now.
**Next:** (1) Extract the paper's composition numbers from the PDF into a table (human check). (2) List which encoders have downloadable weights. (3) Write the feature-extraction script for Phase 1 diagnostics. No cluster Job without user go-ahead.

## 2026-10-17 (c) [agent]
**Goal:** Transcribe the paper's Table 1, check encoder weight availability, and write the Phase 1 feature-extraction script.
**Did:**
- Transcribed Table 1 (single-source, LIBERO-Plus) from the user's screenshot to `literature/liu2026what-table1-single-source.csv`. Agent transcription; human check pending.
- Wrote `literature/encoder-weights.md` (HF API + GitHub READMEs).
- Wrote `src/extract_features.py`: frozen encoder over LIBERO HDF5, storing global/mean/4×4-grid features as fp16. Loaders for DINOv3, SigLIP 2, C-RADIOv4; stubs for the rest.
- Local CPU smoke test (scratch venv, not committed): downloaded one LIBERO file (`libero_goal/turn_on_the_stove_demo.hdf5`, 448 MB), 2 demos × 2 cams.
**Found:**
- verified: HDF5 layout `data/demo_k/obs/{agentview_rgb,eye_in_hand_rgb}`, 128×128 uint8; this file has 50 demos, 4460 steps. `macros_image_convention: opengl`. Visual check: raw agentview is upside down; rotate-180 looks upright (default on). Wrist cam judged the same way, less obvious.
- verified: SigLIP 2 (base-224) and C-RADIOv4-SO400M run end to end. No NaNs; features vary over time. Shapes: SigLIP global 768; RADIO summary 2304, tokens 1152.
- verified: CPU throughput (fp32, Mac) is 31 frames/s for SigLIP-base and 1.0 frames/s for RADIO-SO400M at 448. Not representative of GPU speed.
- verified: liu2026what's Table 1 is single-source only; the composition numbers are in a separate figure or appendix.
- speculative: storage ≈ 400k frames for LIBERO (2 cams) × ~33 KB/frame (4×4 grid, 1024-d fp16) ≈ 13 GB per encoder. Too much for the 20 GB PVC with several encoders, so it needs CephFS.
- DINOv3 is gated and there's no local HF token, so its loader is untested. HRP weights not found.
**Next:** (1) User verifies the Table 1 CSV and the weights table, and confirms encoder variants from the paper. (2) Get the composition numbers (App. E). (3) Accept licenses for DINOv3 / SAM 3 / VGGT-Ω and create an HF token Secret. (4) On go-ahead, a single-GPU extraction Job for 1 suite × 2–3 encoders.

## 2026-10-17 (d) [agent]
**Goal:** Record the paper's composition table and the user's weight/variant corrections.
**Did:** Transcribed Table 10 (RQ3 composition) from the user's screenshot to `literature/liu2026what-table10-composition.csv`. Updated `literature/encoder-weights.md` with the user-supplied corrections (HRP on the CMU site, DynaFLIP first-author upload, variants from App. A).
**Found:**
- verified (internal consistency): the mean and SD recomputed from the 3 seed columns match the transcribed Mean ± SD in all 16 rows. The SigLIP 2 deltas (+1.83, −0.44, +0.20) match the project-page text.
- verified (arithmetic on the table): 16 conditional gains, using D alone = 84.0 from Table 1. Most fall within ±1.2 points; outliers are D+R −5.41, D+U+V+G −2.36, D+U +2.12, D+L +1.83.
- speculative: D+R −5.4 is the largest effect. C-RADIOv4 is distilled partly from DINOv3-7B, so it fits "overlap hurts". Fusion or dimension effects (RADIO summary is 2304-d) are an alternative explanation.
- Not verified by the agent: the pasted weight and variant details (the HRP page loads; its contents weren't checked).
**Next:** User accepts the HF licenses (DINOv3, SAM 3, VGGT-Ω; request VGGT-Ω early) and creates a read token. Then, on go-ahead, a single-GPU extraction Job.

## 2026-10-18/19 [agent]
**Goal:** Resolve the "seeds" discrepancy I raised in the catch-up, check the Table 1/10 transcriptions, explain HF license steps, and run the first feature-extraction Job (user go-ahead 2026-10-18).
**Did:**
- Fetched the arXiv HTML v1 of liu2026what and script-checked both CSVs against it. Updated the paper note and the CSV headers.
- Rebuilt a pinned CPU venv (transformers 4.57.6, timm 1.0.30, open-clip-torch 3.3.0, torch 2.7.1) and re-smoke-tested SigLIP 2-L and C-RADIOv4.
- Fixed RADIO under fp16: it now uses fp32 weights + autocast (a cast to fp16 crashed). Added a non-finite check and a `SRC_GIT_COMMIT` env override (f4a7b82).
- Ran Job `stacking-encoders-extract-libero-goal` (89fb593). See `experiments/2026-10-18-extract-libero-goal/`.
**Found:**
- **Correction:** my catch-up claimed Table 10's 3 seeds contradicted "one checkpoint per config". Wrong. verified, App. C.2/E.3: one trained checkpoint per config, 3 *evaluation* seeds. The repo already said so.
- verified: Table 1 (14 rows) and Table 10 (16 rows) match the HTML exactly. The general 82.2 vs specialized 80.0 recomputes.
- verified (App. B.1): the paper's default WAM is joint video–action denoising, not Fast-WAM direct prediction. The 10-16 (b) entry implied otherwise. Also verified, App. C.1: LIBERO training uses 1,712 episodes. App. A names no exact checkpoints, only "primarily L".
- verified (CPU, 1 demo): fp16 vs fp32 cosine ≥ 0.9995 (SigLIP 2-L), ≥ 0.99999 (RADIO autocast). Not measured on GPU.
- verified (Job, TITAN RTX, fp16): 127,456 frames per encoder for libero_goal. SigLIP 2-L 142 frames/s, RADIO 62 frames/s. 4.4 G + 5.2 G on the PVC. GPU utilization 90–99%. 53 min total.
- verified (Prometheus ~06:45Z 10-19): no free Ampere+ `nvidia.com/gpu` on a node with ≥2 CPU / 10 Gi. Free A6000s were on a 0.07-CPU node.
- speculative: all 4 suites × 2 encoders ≈ 38 GB, too big for the 20 GB PVC.
**Next:**
- (1) User accepts HF licenses (DINOv3 first; VGGT-Ω early) and creates the `stacking-encoders-hf` Secret themselves.
- (2) Load the cached libero_goal features and run first diagnostics (CKA, residual probe) on SigLIP 2 vs RADIO. Local or 1-GPU, no new Job without go-ahead.
- (3) Decide feature storage for the 4-suite run (CephFS PVC vs. shrinking: fewer grid cells or frame stride).
**Open questions:** Is fp16 (vs the paper's unknown precision) acceptable for all encoders? Do we use the paper's 1,712-episode subset, and which episodes are they?

## 2026-10-19 [agent]
**Goal:** Run the first Phase 1 diagnostics (CKA, action probes, residual information) on the cached libero_goal features (user go-ahead).
**Did:**
- Wrote `src/diagnostics.py` and `src/test_diagnostics.py` (synthetic cases with known answers).
- Ran CPU Job `stacking-encoders-diag-libero-goal`: one failed submit, then run 1 and run 2. Details and tables in `experiments/2026-10-19-diagnostics-libero-goal/README.md`.
- Bugs caught along the way:
  - Synthetic test: a ridge-shrunk residual map leaked A's signal back after standardization.
  - Local 12-demo run: a key-ordering crash, and an overfit residual map when n < 10·dim (now guarded).
  - Cluster: `free` missing in `python:slim`.
**Found (verified, one pair / one suite, not a finding):**
- Action-probe test R² (global features): proprio 0.28, SigLIP 2-L 0.61, RADIO 0.67, both 0.68. Gains: +0.069 (RADIO given SigLIP) vs +0.008 (reverse). Mean features show the same direction.
- Debiased CKA SigLIP vs RADIO is 0.72–0.94: highest for grid and the wrist camera, lowest for global agentview.
- resid(B|A) = 0.000 everywhere, in both runs. Run 2 shows the map is well-conditioned, and the residual probe's val R² is negative at all α.
- speculative: within-demo frame correlation makes the effective n ~300 demos, so the probe overfits.
**Decision needed (verification gate: evaluation protocol):** drop the resid(B|A) probe and use gain(B|A) as the asymmetric metric? (They're equivalent under OLS.) Add demo-level bootstrap CIs to all probe numbers?
**Next:** (1) User decides on the metric change. (2) Add bootstrap CIs and rerun on the cached features (CPU). (3) DINOv3 once the HF token exists. (4) Storage decision for 4 suites (CephFS recommended).

## 2026-10-19 (b) [agent]
**Goal:** Explain the open metric decision, find the gated model pages, and draft feature storage on CephFS.
**Did:** Checked the HF API for `facebook/sam3` and `facebook/VGGT-Omega` (both exist, `gated: manual`). Read the NRP Ceph page and storage tutorial. Wrote `data/pvc-features-cephfs.yaml` (100Gi, `rook-cephfs`, ReadWriteMany). A server-side dry run passed. **Not created.**
**Decided (user, 2026-10-19):** drop resid(B|A); gain(B|A) is the asymmetric metric. Add demo-level bootstrap CIs to all probe numbers.
**Found:**
- verified (user): DINOv3 access granted. HF token and `stacking-encoders-hf` Secret not yet created.
- verified (kubectl): the namespace has 5 PVCs, all `rook-ceph-block`. No storage quota entries are visible.
- verified (NRP docs): `rook-cephfs` is US West, ReadWriteMany; conda/pip installs are prohibited. The docs give no region label value and no CephFS size limit.
**Next:** (1) User OKs creating the CephFS PVC. (2) Implement bootstrap CIs and remove resid locally, then ask before the CPU rerun Job. (3) User creates the HF token Secret; then a DINOv3 extraction Job on go-ahead.
