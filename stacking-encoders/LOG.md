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

## 2026-10-19 (c) [agent]
**Goal:** Correct the NRP-docs claim in the 2026-10-19 (b) entry.
**Correction to 2026-10-19 (b), "Found", NRP docs line:** "The docs give no region label value and no CephFS size limit" was wrong on both counts.
- verified (NRP docs, `userdocs/storage/intro.md`, site commit `f926199`): the region-affinity example uses `topology.kubernetes.io/region In [us-west]`. That matches `rook-cephfs` (US West). verified (kubectl, 2026-10-19): 288 nodes carry `us-west`.
- verified (same page): CephFS has a per-file size limit of 16 TB. No per-volume or per-namespace CephFS cap is documented.
- Implication: `data/pvc-features-cephfs.yaml` is unaffected, but any Job that mounts it should add `us-west` region affinity (see `_standards/08`, "CephFS in detail").
**Next:** unchanged from (b). (1) User OKs creating the CephFS PVC.

## 2026-10-19 (c) [agent]
**Goal:** Implement the decided metric change (gain + demo-level bootstrap CIs, drop resid), confirm the HF Secret, and prepare the storage decision.
**Did:**
- `src/diagnostics.py` (7dc80ca): removed resid(B|A). Added demo-level bootstrap CIs (1,000 draws by default): probes are kept fixed, test demos are resampled with replacement, and the same draws are used for every probe. Gives CIs for each probe R², each gain, and the asymmetry R²(B) − R²(A).
- `src/test_diagnostics.py`: new synthetic tests. Bootstrap R² from per-demo statistics matches an explicit resample exactly. With demo-correlated data, the informative-B gain CI is +0.175..+0.247 and the redundant-C CI is −0.001..−0.001. All pass.
- Smoke run of the full script on the 12-demo local set (from scratchpad): runs end to end. The numbers are meaningless with ~2 test demos.
- User created the `stacking-encoders-hf` Secret themselves. verified (kubectl metadata only): it exists (23:05Z) with one key `HF_TOKEN` of token-like length.
**Found:**
- verified (user): access granted to DINOv3, SAM 3, and VGGT-Ω.
- verified (kubectl): other lab members have personal 50Gi RBD PVCs. Ours would be the first CephFS PVC in `aria-world`. No storage quota is visible.
- Limit: the CIs cover test-demo sampling only, not probe-training variability.
- numpy 2.2 on macOS prints spurious matmul RuntimeWarnings (recalled: a known Accelerate issue). Results are unaffected (CKA self = 1, all asserts pass).
**Next (all need user go-ahead):** (1) Create the CephFS PVC `stacking-encoders-features` (`data/pvc-features-cephfs.yaml`). (2) CPU diagnostics rerun Job with CIs on the libero_goal features (can read `michael-persistent`). (3) Copy Job for the libero_goal features to CephFS, checksum-verified, originals kept. (4) DINOv3 extraction Job using the Secret.

## 2026-10-19 (d) [agent]
**Goal:** Update the direction's docs and specs for the PhD mentor's feedback (Slack, 2026-10-19), and help shape the hypothesis. No experiments run, no cluster actions.
**Did:**
- Read agostinelli2022 (arXiv v2 PDF; the mentor's link). Wrote `literature/notes/agostinelli2022-transferability-ensembles.md` and a BibTeX entry.
- Wrote `writing/hypotheses-draft.md`: mentor feedback, similarity vs complementarity definitions, candidate hypotheses H-A..H-D, the 7-pair feasibility tally, a novelty statement draft, and Thursday prep.
- Wrote `decisions/002-feasibility-before-proxy-training.md` (**proposed**). Added a sequencing note to 001.
- Updated `README.md` (goal: publish; question reframed around conditional complementarity; plan) and filled the empty `STANDARDS.md` sections (scope, sources, baselines, metrics, stats, pitfalls, notation).
**Found:**
- verified (agostinelli2022, Sec. 3.4/4.1):
  - Their ensembles average the outputs of separately fine-tuned models.
  - MS-LEEP assumes independent sources and reduces to a sum of single-model LEEP scores, so it has no interaction term.
  - LEEP-family metrics need label-space predictions.
- speculative: those three points are where our work can differ (feature fusion that can hurt; foundation encoders without label spaces; asymmetric conditional gain). The broader related-work check (Cambrian-1, Theia, feature-based LogME/H-score variants) is not done.
- verified (arithmetic on the transcribed Table 10, D alone = 84.0): the 7 two-encoder deltas are D+V +0.93, D+L +1.83, D+S +0.49, D+G +0.40, D+H +0.48, D+U +2.12, D+R −5.41.
**Decision needed (user):** (1) Approve decision 002? (2) Pick a primary hypothesis (agent suggests H-B). Both are verification gates (scope/question).
**Next:** User reviews the hypotheses draft before Thursday 2026-10-22. On go-ahead: DINOv3 extraction for libero_goal (D is the base of every pair), then SAM 3 / VGGT-Ω, then the diagnostics rerun with CIs for the 7-pair tally. The earlier pending go-aheads (CephFS PVC, copy Job) can wait unless many encoders are extracted.
**Open questions:** What does the mentor consider the closest prior work beyond agostinelli2022? Is a 7-pair, single-checkpoint tally acceptable as a first check?

## 2026-10-19 (e) [agent] (pre-registration, written before any tally run)
**Goal:** Feasibility tally (decision 002, accepted by user 2026-10-19). libero_goal only, base D = DINOv3 ViT-L/16, partners L, R, then S, G (V, U, H only if cheap). No policy training. No hypothesis is argued (user instruction): this produces numbers for a manual tally against the 7 liu2026what Table 10 deltas.
**Fixed before seeing any D results:**
- **Primary feature type: global** (as in the pilot). Mean-pooled is secondary. Grid is CKA-only (unchanged).
- Same frames and same demo split as the pilot. Per task file, demos sorted by index: last 20% test, previous 20% val, rest train. 63,728 timesteps, 2 cams. Rotate 180°, fp16 features, `src/diagnostics.py` unchanged apart from its inputs.
- Per pair (D, B): debiased linear CKA (5,000 frames, seed 0), probe test R² for D, B, D+B, gain(B|D), gain(D|B), demo-level bootstrap 95% CIs (1,000 draws, seed 0).
- Ridge α is selected on validation **separately for every probe**, including each concatenation (grid 1e-2…1e5), then refit on train+val. No matched-dimension control; dimension mismatches are reported, not corrected.
- Tally columns: CKA(D,B) (global; agentview and wrist reported separately), gain(B|D) [CI], gain(D|B) [CI], paper delta vs D alone = 84.0. Observations are qualitative only; no significance claims (n ≤ 7).
- Encoder variants (choices, not confirmed to match the paper): DINOv3 `facebook/dinov3-vitl16-pretrain-lvd1689m` at 224; SAM 3 `facebook/sam3` ViT backbone, pre-neck, at 1008 (no summary token, so global = patch mean); VGGT-Ω `vggt_omega_1b_512.pt` at 512, final aggregator layer (2048-d: frame + global-attention tokens concatenated), global = camera token.
**Did so far:** Wrote `sam3` and `vggt-omega` loaders in `src/extract_features.py`. Smoke-tested D/S/G with **random weights** on CPU (scratch venv: transformers 5.18.0, torch 2.7.1, vggt-omega @ 48b23c8). verified: shapes D global 1024 / tokens 14×14; S 1024 / 72×72; G 2048 / 16×16 at 256. Outputs finite. Real gated weights not yet loaded anywhere.
**Note:** this log has two entries headed "2026-10-19 (c)". The second (bootstrap implementation) is the one the README means. Left as-is (append-only).

## 2026-10-19/20 (f) [agent] (tally extraction: cluster events so far)
**Did (all approved by user in-session; times UTC 10-20):**
- Created CephFS PVC `stacking-encoders-features` (100Gi, `rook-cephfs`, Bound). Created ConfigMap `stacking-encoders-extract-src`.
- Try 1 (00:22, 2080 Ti): DINOv3 in fp16 gave non-finite features on batch 0. Deleted the Job and its retry pod at once. ~1 GPU-min. Log: `experiments/2026-10-19-feasibility-tally/results/extract-try1-job.log`.
- Try 2 (00:29–02:44, 2080 Ti, k8s-haosu-05): DINOv3 in fp32 finished all 10 files (65–72 frames/s, 4.4G). VGGT-Ω (fp16) finished 5 of 10 files at 11.9 frames/s (~3× slower than my estimate). Deleted by me after user approval, for under-used memory (below). ~2.25 GPU-h.
- Try 3 (submitted 02:47): 1 CPU / 4Gi (limit 1.2 / 4.8Gi), VGGT-Ω loader builds on the GPU and mmaps the checkpoint (9e3bdb2). Resumes by skipping finished files. Watchdog running locally: deletes the Job if GPU < 40% or memory working set < 20% of the request on 3 consecutive 5-min checks.
**Found:**
- verified (Prometheus, try 2): GPU util median 100%. CPU ~1.0 core of 2 requested. Host memory: DINOv3 stage RSS ~1.0 GiB, VGGT-Ω stage median 4.4 GiB (max 8.4) against a 16Gi request. **The memory request was under 20% use for the whole run.** The pod was over 30 min old from ~00:59, so it was probably flagged as a memory-low violation on the NRP portal from ~01:00 to 02:44 (not checked on the portal by the agent). My sizing error: I requested 16Gi for a short load peak.
- verified (try 3, 02:59): VGGT-Ω stage working set 1.42 GiB (35% of 4Gi), anon 1.25 GiB, GPU 100%, 0 OOM events.
- verified: the HF token reads all three gated repos. Revisions: DINOv3 `ea8dc28`, SAM 3 `3c879f3`, VGGT-Ω `1041e80`.
- verified (local, random full-size VGGT-Ω checkpoint): the new loader gives identical features to a plain strict load (max |diff| 0.0).
- **Deviation from pre-registration (e):** DINOv3 features are fp32, not fp16 (fp16 overflowed). L and R are fp16 (R via autocast).

## 2026-10-21 (g) [agent] (tally complete: D+S diagnostics, write-up)
**Did:**
- 10-20: extraction try 3 finished at 10:40Z. All 10 files for G and S; D was done in try 2. ~7.9 GPU-h at 99–100% GPU and 85–89% of the 4Gi request.
- 10-21: saved the try-3 log. Deleted the extraction Job and ConfigMap `stacking-encoders-extract-src`.
- With user approval ("D+S only"): CPU-capacity check via Prometheus found 28 us-west CPU-only nodes that fit 2.4 CPU / 13 GiB. Created ConfigMap `stacking-encoders-diag-src`. Submitted `diag-ds-job.yaml` (07:16Z, commit 1e342dc). It completed in 30 min. Saved the log and JSON, deleted the Job and ConfigMap. The namespace has no Jobs, pods or ConfigMaps of ours (verified 08:22Z).
- Wrote `experiments/2026-10-19-feasibility-tally/README.md`: config, commits, the 4-row tally (global and mean), plain observations, caveats, skipped encoders, failures, resource use.
**Found** (verified, `results/tally-libero_goal-dinov3-sam3.json`; observations only, no hypothesis chosen):
- D+S global: CKA 0.559 / 0.643; gain(S|D) +0.033 [+0.027, +0.038]; gain(D|S) +0.029 [+0.024, +0.034]; R² S 0.654, D+S 0.682.
- Across L, R, S, G: R has the highest CKA with D in all 6 feature-type × camera cells (margin 0.007–0.094) and the largest gain(B|D). The paper order is L > S > G > R. See the experiment README.
- The local Prometheus monitor's memory-low alert fired at 5 min (pip install stage). It was a false alarm: NRP skips pods under 30 min. I added a grace period to the monitor (scratchpad only).
**Resources (whole tally):** ~10.2 GPU-h (tries 1–3). CPU diagnostics 49 + 45 + 30 min at 2 cores.
**Next:** user reviews the tally (key findings need sign-off), then chooses the hypothesis and whether the proxy-policy study (decision 001) is worth running. Open: keep or delete the 17.6G on the CephFS PVC; check the NRP Violations page for the probable try-2 memory-low flag.

## 2026-10-21 (h) [agent] (key findings signed off; README brought up to date)
**Did:**
- The user reviewed the tally and asked for it to go into the README key findings. Wrote them with labels (verified / speculative) and n = 4 caveats.
- Updated the README status, the plan (step 1 done, decision 002 accepted), and key files (tally experiment, CephFS PVC manifest, loader list).
- Verified the 7 paper deltas against `literature/liu2026what-table{1,10}-*.csv`: Table 10 mean − D alone 84.0 matches all 7 values in the user's spec. The experiment README now cites this instead of "not re-checked".
- Made the HRP skip reason precise in the experiment README.
**Found:** in Table 10, D+R has the largest eval-seed sd (1.01) (verified, CSV).
**Stale, not edited (out of scope of this request):**
- `src/extract_features.py` HRP stub message ("no public release found 2026-10-17") predates the user's correction that the weights are on the CMU site.
- `STANDARDS.md` storage line still says checkpoint storage is TBD; the CephFS PVC now exists for features.
**Next:** user and mentor (Thursday 2026-10-22): choose the hypothesis and whether the proxy-policy study (decision 001) is worth running. Open: keep or delete the 17.6G on the CephFS PVC; check the NRP Violations page for the probable try-2 memory-low flag.

## 2026-10-21 (i) [agent] (paper check + pre-registration of target-side diagnostics; written before any target-side code or run)
**Goal:** Explain why the input-side tally can't flag D+R. Check how liu2026what uses encoder features. Pre-register cheap diagnostics that treat the encoders as **prediction targets**. No code written and nothing run yet.
**Found:**
- verified (`src/diagnostics.py` lines 6, 290–299): every probe so far predicts the 7-d action **at the same timestep** from current features. No future-feature or encoder-as-target diagnostic exists. `pred(D→B)` (same frame, global) is in the JSONs but not the tally tables: R 0.805, S 0.733, L 0.675, G 0.603 (read from `results/tally-*.json`, no CIs). It is a redundancy measure, not a target-predictability measure.
- verified (arXiv HTML v1, read through a summarising fetch; verbatim quotes not yet hand-checked against the PDF):
  - Encoder features are both input and target. Sec. 3.3: features "concatenated along the channel dimension"; the video expert's projections are "expanded to jointly predict VAE and external-feature flow velocities". App. B.2: "Observed features remain clean, whereas future VAE latents and external features are jointly noised and predicted."
  - App. B.2: auxiliary feature-prediction loss weight 1.0. **No per-encoder normalization, rescaling or loss weight is stated** for the concatenation path.
  - Table 6: action horizon 32 steps; 9 frames per view at offsets 0, 4, …, 32. Table 7 / App. D.3: 2 views (agent-view and wrist), 224×224. App. D.3: DINOv3 gives a 14×14 grid of 1,024-d features per view.
  - Sec. 4.4: the paper's only explanation for D+R is a teacher–student overlap (C-RADIOv4 distills DINOv3, SigLIP 2 and SAM 3).
- speculative: an input-side ridge probe on [D, B] can almost never score below D alone, so gain(B|D) can't predict harm by construction. Multi-output ridge also fits each output independently, so a linear probe can't show targets competing either. That needs shared capacity under one summed loss.
**Hypothesis under test (H-E, proposed; not chosen by the user):** a partner hurts when it makes the *prediction target* worse. Either its future features are hard to predict, or, under the paper's unnormalized concatenation, it takes a large share of the summed loss and crowds out D's own future prediction.
**Pre-registered (fixed now, before any code or result):**
- **Data, split, encoders:** as in (e): libero_goal, same demo split, D + {L, R, S, G}, global features primary, mean secondary. Both cams, concatenated as before.
- **Horizons:** k ∈ {4, 16, 32} steps (the paper's first offset, a middle one, and its last). Frames with t + k beyond the demo end are dropped. **Primary = mean over the 3 k.**
- **Target:** Δ_E = f_E(t + k) − f_E(t), so that "copy the current frame" scores R² = 0. Inputs: current global features, standardized on train.
- **T0 (loss share; no fitting):** each encoder's share of the total raw variance of Δ in the concatenated target [D, B], on train frames.
- **T1 (target predictability):** ridge from [D, B] at t to Δ_B. Test R² averaged over dims. α tuned on val per probe (same grid), refit on train+val.
- **T2 (target interference; primary measure):** reduced-rank ridge, rank r = 64 (secondary r ∈ {16, 256}). Closed form: ridge fit, then SVD of the train fitted values, then project, in the target scaling being tested. Inputs are [D, B] at t in every condition, so only the target changes.
  - interference(B) = R²_D(target Δ_D only) − R²_D(target [Δ_D, Δ_B]), on D's dims, test set.
  - Two scalings: (a) **raw**, as in the paper; (b) **per-encoder equalized**, each encoder's block rescaled to equal total train variance.
- **CIs:** demo-level bootstrap, 1,000 paired draws, seed 0, as in `src/diagnostics.py`.
- **Prediction (H-E):** on T2-raw (primary), R has the largest interference of the 4 partners, and it shrinks under (b).
- **Outcome rule:**
  - R not the largest on T2-raw → H-E fails at this level. The remaining explanations (single-checkpoint noise, training dynamics, LIBERO-Plus shift) need retraining or the authors' variance data.
  - R largest → weak support only. With n = 4, a random measure picks R 1 time in 4. It would justify a small causal test (with vs without per-encoder normalization in a trained model), not a claim.
  - T0, T1 and the secondary r, k and feature types are reported but don't decide the outcome.
- **Known limits:** global/mean features, not the paper's 14×14 grids. Linear model, not flow matching. Deltas, not noised targets. Clean libero_goal, not LIBERO-Plus.
**Next:** (1) user/mentor review on Thursday 2026-10-22. (2) On go-ahead, write `src/target_diagnostics.py` (reusing `diagnostics.py` loaders) plus synthetic tests, e.g. a planted high-variance block must cause interference under raw and not under equalized scaling. (3) CPU Job per pair after a Prometheus capacity check and user approval.
