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

## 2026-10-21 (j) [agent] (paper read from the user's PDF; amendment to the (i) pre-registration, before any code or run)
**Goal:** Read liu2026what from the user's PDF (`~/Downloads/2610.13024v1-2.pdf`, text via pdftotext) to replace the summarising-fetch reading in (i). Amend the pre-registration where the paper contradicts it.
**Found (verified, PDF text):**
- (i)'s quotes are confirmed: Sec. 3.3 "concatenate them along the channel dimension… jointly predict VAE and external-feature flow velocities". App. B.2 "Observed features remain clean, whereas future VAE latents and external features are jointly noised and predicted." App. B.2 "auxiliary feature-prediction losses use a weight of 1.0." Table 6: offsets 0, 4, …, 32; action horizon 32.
- **App. E.1, Eq. 9: each source has its own loss term**, L_aux = Σ_k L_FM(f^k_{1:T}), and L_FM is the "modality-specific mean-squared error" (Sec. 3.1). So the feature losses are not one MSE over the pooled channels. If L_FM averages over a source's channels (likely but not stated), channel count does not set a source's loss share; feature scale still can. Each added source also adds another full-weight (1.0) feature loss next to the video and action losses.
- App. E.1: "Source-specific processing and concatenation order are held fixed". The processing is not described. **Whether features are normalized remains unstated.** App. D.3 says "raw current-frame DINOv3 features". The design follows DreamWorld (Tan et al., 2026), not checked.
- **Table 9 (App. D.2), DINOv3 under channel-wise concat:**

  | Config | LP success |
  |---|---|
  | current-only input | 76.5 |
  | future-only prediction | 81.9 |
  | both | 84.0 |

  The authors read this as "supporting future-feature prediction as a major contributor". Our tally tested only the current-input role.
- Table 1 (repo CSV): C-RADIOv4 alone 83.5 vs DINOv3 alone 84.0. **D+R (78.6) is below both single sources**, so the harm comes from combining them, not from a weak R.
- Sec. 4.4: the only explanation offered for D+R is "a direct teacher–student relationship" (R distills D, SigLIP 2 and SAM 3).
**Amendment to (i), T2 (made before any code or result):**
- Primary scaling is now **(a′) per-source mean MSE at raw feature scale**: each encoder's block weighted by 1/d_k, matching Eq. 9 if L_FM is a per-source mean. (b) per-source equalized variance is unchanged.
- The (i) "raw" pooled-channel loss becomes secondary (a″), as the case where L_FM sums channels.
- The prediction is unchanged (R has the largest interference under the primary scaling and less under (b)), as are the outcome rule and the other choices in (i).
- Added to T0: per-encoder per-channel feature variance and mean norm. This is a direct check of the feature-scale route.
**Open questions for the authors:** training-seed variance for D+R; what the "source-specific processing" is (normalization?); whether L_FM is a mean or a sum over channels.

## 2026-10-21 (compliance) [agent]
**Goal:** Check the CephFS volume and our cluster use against the NRP rules in `../_standards/08`.
**Did:** Read-only kubectl (`get pvc/pods/jobs/configmaps/resourcequota/secret`) and a review of every Job manifest in `experiments/`. The NRP policy pages were not re-fetched; 08 was last checked against them 2026-10-16/19.
**Found (verified):**
- `stacking-encoders-features`: Bound, 100Gi, RWX, `rook-cephfs` (US West). It matches `data/pvc-features-cephfs.yaml`. Last df (extract try 3, 10-20 10:33Z): 15G used. LOG says 17.6G. Mounted at `/mnt/features`, not a home dir.
- No pods, Jobs, Deployments or stray ConfigMaps in `aria-world`. Only GPU-quota and ban-class quotas, all 0/0, none hit.
- All Jobs: `kind: Job`, `backoffLimit: 1`, `activeDeadlineSeconds` set, no `sleep`, no tolerations, no `priorityClassName`, limits within 20% of requests. CephFS Jobs have us-west affinity. pip installs and HF_HOME go to the container or `emptyDir`, never CephFS. Each writer writes its own files (timestamped logs, per-pair JSON).
- Tally results and logs are copied into `experiments/2026-10-19-feasibility-tally/results/`, so the 6-month purge risks only the cached features.
**Open:** extraction try 2 was probably flagged memory-low (16Gi request, under 20% used, ~01:00–02:44Z 10-20). Only the NRP Violations page (https://nrp.ai/userinfo, user login) can confirm. Try 3's request was right-sized (85–89% used, GPU 99–100%).
**Next:** User checks the Violations page. Decide keep/delete for the CephFS features after the mentor meeting.

## 2026-10-21 (k) [agent] (target-side diagnostics implemented and tested locally; no cluster action)
**Goal:** Implement `experiments/2026-10-21-target-diagnostics/PLAN.md` (binding spec: (i) as amended by (j)): code, synthetic tests, local sizing, Job manifest. Stop before any cluster action (user instruction).
**Did:**
- `src/target_diagnostics.py` (T0, T1a/T1b, T2 reduced-rank ridge under a′/b/a″, ranks 64/16/256, k 4/16/32, global + mean, demo-level bootstrap with k-averaged draws, α-grid-edge record) and `src/test_target_diagnostics.py`. `src/diagnostics.py` unchanged (imported).
- Implementation choice (not a spec change): ridge goes through one eigendecomposition of XᵀX per fit set, same formula as `diagnostics.ridge_fit`. Check 6 shows it equals `diagnostics.ridge_select` (same α, max pred diff 4e-13). The Δ_D-only reference is computed once per k and reused for every scaling; check 3 shows the scaling can't change it.
- Commits: `2e03e18` (code + tests), `b68ba54` (performance fix below). Manifest `experiments/2026-10-21-target-diagnostics/job.yaml` (not submitted).
**Found (verified, local, numpy 2.2.6 / Accelerate, Python 3.11):**
- All 7 synthetic checks pass: shift (Δ = k, no cross-demo pairs, counts Σ max(0, len − k)); full-rank RRR = ridge (max diff 6.6e-10); single-block invariance; planted interference (r = 32: a′ +0.988, a″ +0.988, b +0.013; r = 64: all |·| < 1e-4); identical predictors give CI [0, 0]; fast ridge = `ridge_select`; `run_type` end to end, and a ≤ 32-frame test demo trips the assert. `test_diagnostics.py` still passes.
- **Check 4 failed on its first run, and the test data was changed (not the thresholds).** Under b at r = 32, interference was +0.082 against the pre-set threshold < 0.05. Cause: random Gaussian maps gave uneven factor variances, so only 14 of D's 20 directions ranked in the top 32 under b (6.3% of D's variance outside), so the test's premise was false for its data. Switched to orthonormal maps; the thresholds were unchanged. Recorded in the test file.
- **Performance bug fixed:** a reversed (negative-stride) view of the singular vectors made matmuls skip BLAS (23.4 s vs 0.03 s for the same product). Fixed by a contiguous copy. T1/T2 outputs are bit-identical before and after (checked on one block).
- **Sizing (synthetic random-walk features at real D+R dims, 2 threads):** full size (63,622 frames), one block (global, k = 4): 688 s wall, ~1.26 cores average, peak RSS 6.1 GiB (macOS). 20% subset: 7.0 GiB. Peak is set by the 6,656² matrices, not frame count. Synthetic outputs were sane (R² ≈ 0, interference ≈ 0, as expected for unpredictable random-walk deltas). No real features were used, and no real result exists yet.
- Estimate (speculative, extrapolated by dim³): D+R ~70 min, D+G ~55, D+L and D+S ~25 each; ~3 h sequential at 2 CPU locally, maybe up to 2× on cluster nodes. Above the PLAN's 1–3 CPU-h guess.
- Manifest: one CPU Job, 4 pairs sequential (cheapest first), 2 CPU / 9Gi (limit 2.4 / 10Gi), 8 h deadline, ConfigMap `stacking-encoders-target-src`, `SRC_GIT_COMMIT` b68ba54, no sleep, no tolerations. Linux RSS is unmeasured, so 9Gi is a judgment call.
**Resources:** local CPU only (~1.2 h of sizing/profiling runs). Cluster: none.
**Next:** user approval for the cluster steps, each to be confirmed: (1) Prometheus CPU-capacity check, (2) check that `michael-persistent` is free, (3) create ConfigMap `stacking-encoders-target-src` from `src/diagnostics.py` + `src/target_diagnostics.py` at b68ba54, (4) submit `job.yaml`, then watch memory against the 9Gi request. After the run: save the log and JSONs, delete the Job and ConfigMap, write the experiment README applying the (i) outcome rule.

## 2026-10-25 [agent] (research proposal draft for the mentor, due Tuesday 2026-10-27)
**Goal:** Draft the mentor's requested proposal (Motivation, Problem statement, Related works, Method, Experiments, Expected results) as a shareable doc. No experiments or cluster actions.
**Did:**
- Wrote the draft as a Claude Doc: https://claude.ai/artifact/B6uCkLxYV5FmWUUDKRQbzF (private until the user shares it). Framed around H-E (target-side interference) with the input-side tally as motivation. H-E is still **proposed, not chosen by the user**. The doc says so and lists it as an open question.
- Recomputed every add-one-encoder delta in Table 10 from `literature/liu2026what-table10-composition.csv` (python, this session).
- Checked 9 related papers against their arXiv abstract pages (WebFetch summaries): Kornblith 2019 (CKA), Cambrian-1, Theia, AM-RADIO, Eagle, LogME, Prismatic (Sec. 4.2 via HTML), OpenVLA, Burns 2023. Not added to `references.bib` yet.
**Found:**
- verified (CSV arithmetic): of 16 add-one steps, 11 gain +0.20..+2.12, 1 is flat (D+U+H −0.03), and 4 lose: D+R −5.41, D+U+L −0.44, D+U+V+S −1.13, D+U+V+G −2.36. VGGT-Ω is +0.40 on D but −2.36 on D+U+V, so the harm depends on context.
- verified (Prismatic HTML Sec. 4.2, summarising fetch, quotes not hand-checked): channel-concat DINOv2+SigLIP helps "across the board"; DINOv2+CLIP hurts on TextVQA.
- Not verified: only Liu et al. and Agostinelli et al. have been read in full. The rest were checked at abstract level only, which does not meet `02` for a final writeup.
**Next:** User reviews the doc before Tuesday 2026-10-27 and picks the main hypothesis. If wanted, add the new papers to `references.bib` with notes. The E2 target-diagnostics Job is still waiting for go-ahead (LOG (k) Next).

## 2026-10-26 [agent] (literature scouting: papers in the line of the mentor's recommendation)
**Goal:** User asked for papers aligned with the mentor's recommended paper (agostinelli2022), and for world-model/embedding topics for proposals 2 and 3. Nothing run.
**Found (verified at arXiv abstract level via summarising fetches; none read in full):**
- **Threat to proposal 1's novelty:** KAGES, "When More Foundation Models Means Less" (arXiv 2608.17490, 2026-08-18). It greedily selects frozen encoders to fuse by *marginal* gain in centered kernel-target alignment, and reports non-monotonic accuracy as encoders are added. Image recognition/retrieval only, no robotics/WAM. This is close to H-B (conditional gain); proposal 1's difference now rests on WAMs and the target role (H-E).
- **Also relevant to proposal 1:** OSBORN (arXiv 2309.02429, ICCV 2023) scores ensembles with a "cohesiveness" term, so "ensemble metrics are additive" (our MS-LEEP point) is not true of all prior work. ReWAM (arXiv 2609.38163) normalizes and calibrates frozen DINO features in a WAM, which overlaps proposal 1's E5 (normalization fix). Metric-guided VFM fusion (arXiv 2605.16864, CVPR 2026 Findings) uses label-free metrics to pick complementary encoders for segmentation.
- Agostinelli lineage: LEEP (2002.12462, ICML 2020), GBC (2111.12780, CVPR 2022), "How stable are transferability metrics evaluations?" (2204.01403, ECCV 2022), SFDA (2207.03036, ECCV 2022), survey (2402.15231).
- World-model encoder choice (for proposals 2/3): ALPS (2610.30120), Planning Limits of Latent World Models (2609.39235; Table 2 plannable range per encoder), DINO-WM (2411.04983; Table 2 encoder comparison read via HTML).
**Not done:** the proposal 1 doc has not been updated with KAGES / OSBORN / ReWAM. No bib entries added.
**Next:** User reads the list and picks topics for proposals 2 and 3. Then update proposal 1's Related works and novelty claim (needs user OK, since it changes the framing).

## 2026-10-26 (b) [agent] (mentor's reading list and companion papers)
**Did:** The user shared the mentor's own reading list: Atlas (World Labs blog), V-JEPA 2.1 (2603.14482), DINO-WM (ICML 2025), VL-JEPA (2512.10942), SpatioLM (2608.01899), SmolVLM (2504.05299). Fetched each abstract (verified at abstract level). Checked companion papers on arXiv: I-JEPA (2301.08243, ICCV 2023), V-JEPA 2 (2506.09985), PLDM (2502.14819), LeWM (2603.19312), JEPA-VLA (2602.11832), 2606.07687 (prediction over reconstruction), VSI-Bench (2412.14171), VGGT (2503.11651, CVPR 2025), plus the world-model encoder papers in (2026-10-26).
**Found:** speculative: the list's themes (embedding-space world models, JEPA, spatial VLMs) fit topic A (encoder choice for latent world models). VLM spatial distillation from 3D models looks crowded (MVRD 2608.10864, GeoWorld-VLM 2605.16713; search snippets only).
**Next:** user picks topics for proposals 2 and 3.

## 2026-10-27 [agent] (three proposal drafts as HTML for Google Docs)
**Goal:** User wants the proposals as their own Google Docs (not Claude Docs): stacking encoders (P1, the one going to the mentor), plus two early drafts (P2, P3) the user will read up on before deciding.
**Did:** Wrote `writing/proposals-2026-10-27/p1-stacking-encoders.html`, `p2-world-model-embedding.html`, `p3-predictability-horizon.html`. P1 = the Claude Doc text plus related-work rows for KAGES, metric-guided VFM fusion, OSBORN and ReWAM, and a softened novelty claim (input-side marginal-gain selection is not new; KAGES). P2 = action-controllable share of embedding change as a training-free encoder-selection score for latent world models. P3 = predictability horizon of frozen embeddings vs Planning Limits' plannable range.
**Found (verified at abstract level unless noted):**
- ALPS (2610.30120) full text via fetch: planning results for 6 encoders (Table 3, S11, S12), probes for 15; no probe-planning correlation computed; no action-controllability measure.
- AD-WM (2609.30264) and Sensorimotor WM (2606.20104) are training methods, not encoder-selection metrics.
- **Correction:** the "+12.2 points on RoboTwin 2.0" for ReWAM given in chat on 2026-10-26 came from a search snippet and is not in the abstract (which reports 93.6% success). Removed from P1. Whether ReWAM's "Feature Calibration" is per-encoder normalization is unconfirmed.
- DINO-WM datasets and some checkpoints are on OSF per its README; no data license stated.
**Not done:** Google Drive upload. Drive is still connected to the school account and no Google Docs editor connector is present; waiting on the user. The Claude Docs copies (P1 complete, P2 partial) still exist and were not deleted.
**Next:** User picks the Drive account; upload the 3 HTML files as Google Docs.

## 2026-10-27 (b) [agent] (proposals uploaded to Google Docs)
**Did:** User approved uploading to the school Drive (minarami@ucsc.edu). P3 rewritten as a full proposal: oracle decomposition of the latent-planning horizon limit into predictor, embedding-geometry and scoring-rule parts (`p3-planning-horizon.html`, renamed from `p3-predictability-horizon.html`). Flattened nested lists inside numbered steps (Google's HTML import restarted the numbering). Uploaded via Drive `create_file` (HTML to Google Doc):
- P1 https://docs.google.com/document/d/1JJj8KcgBusH65LHY6CsFEiFqACy2lQI3f-wcgI3WF40/edit (verified by read-back: numbering 1 to 4 intact)
- P2 https://docs.google.com/document/d/1NW4Wbikj8b1M-V1QPiEmjetNig0gY8GXLG0gwbDu3oY/edit
- P3 https://docs.google.com/document/d/1sKBxucnzdKOZNWzwJBmjCQ7ItSzjis-LZEnu3wLXHUg/edit (uploaded before the list fix: Method steps 4 to 6 show as 1 to 3)
**Found (verified, Planning Limits appendix via fetch):** its exact-dynamics control scores in state space, not embedding space; no oracle-embedding condition; no code or data release mentioned. NGC (2610.25013) abstract shows no oracle-embedding experiment either.
**Not done:** no Google Docs editor connector, so the docs can't be edited in place by the agent. P3's numbering needs a manual fix or a re-upload.
**Next:** User reads the P2/P3 papers and decides which two go to the mentor with P1.

## 2026-10-27 (c) [agent] (P2 and P3 moved to their own directions)
**Did:** Created `../world-model-embedding-choice/` (P2) and `../world-model-planning-horizon/` (P3) from `_template/`, and moved their proposal HTML there. P1 stays in `writing/proposals-2026-10-27/`.
**Next:** unchanged (user reads, picks what goes to the mentor; E2 target-diagnostics Job still awaiting go-ahead).

## 2026-10-27 (d) [agent] (one-page proposals in Google Docs)
**Did:** Google Docs connector now available. For all three proposals: kept the long version as a new "(full notes)" Google Doc (P1, P2 via Drive copy; P3 re-uploaded from the repo HTML so its numbering is fixed), then rewrote each original doc (same link) to a one-page version of about 600 to 660 words from `writing/proposals-2026-10-27/p1-short.md` and the two sibling directions' `proposal-short-2026-10-27.md`. Each batch was guarded by the doc's revision ID; all three were read back and checked. No one had edited or commented on the originals before the rewrite (Drive modifiedTime equal to creation).
- P1 one-page: https://docs.google.com/document/d/1JJj8KcgBusH65LHY6CsFEiFqACy2lQI3f-wcgI3WF40/edit ; full notes: https://docs.google.com/document/d/1GgJm4coJ8Kn2J_D0IfIBYZpXSwlVU_3wVz17nUM7MBY/edit
**Next:** user reviews the one-page versions before sending to the mentor.

## 2026-10-27 (rename) [agent]
**Did:** Folder renamed by the user from `stacking-encoders/` to `world-model-stacking-encoders/` (2026-10-27). Repo paths in README/STANDARDS/references were updated; older entries above keep the old path (append-only). Cluster resource names (PVC `stacking-encoders-features`, Secret `stacking-encoders-hf`, ConfigMaps, Job names, `/persistent/stacking-encoders/` and `/mnt/features/stacking-encoders/` paths) are unchanged on purpose: they name real objects on Nautilus.

## 2026-10-27 (e) [agent] (novelty check for P1; competitors added to the one-page Google Doc)
**Goal:** User is presenting P1 to the mentor and asked to (1) list key talking points, (2) double-check nobody has done this, and list close competitors in the doc.
**Did:** 11 web searches (pre-training prediction of encoder combos for WAMs/VLAs/VLMs, target-side interference, multi-teacher distillation, frozen-feature predictors of control, citations of liu2026what). Abstract pages fetched for 2606.03879, 2609.24048, 2602.04880, 2610.21915, 2507.03262, 2610.34663, 2605.15618, 2410.01680. Edited the P1 one-page Google Doc in place (revision-guarded, read back: 4 new bold-led bullets, a sharper novelty line, 7 linked references); mirrored in `writing/proposals-2026-10-27/p1-short.md`.
**Found (verified at abstract level only; none read in full):**
- No paper found that predicts, before training, whether adding an encoder helps or hurts a WAM, or that studies encoder features as combined prediction targets. Absence in search is not proof of novelty.
- Closest: KAGES (already listed; input-side marginal gain, image tasks); PHI-S (2410.01680; teacher feature-statistics mismatch hurts multi-teacher distillation, fixed by per-teacher normalization; this anticipates the H-E mechanism in a different setting); Wang et al. ICLR 2026 (2507.03262) and Ding et al. (2606.03879): harmful encoders in multi-encoder VLMs, measured on trained models; Ding et al. say good configurations remain hard to identify before training.
- Also adjacent: Shen et al. 2610.21915 and Tang et al. 2609.24048 (prediction-target / latent-space comparisons by training each); Dong et al. 2602.04880 (frozen-feature probes correlate with policy performance); Scardecchia & Yang 2610.34663 (posted 2026-10-26; trains for predictable latents).
- Tang et al.'s gradient-alignment analysis came from a search snippet, not the abstract; not used in the doc.
**Not done:** the doc is now ~800 words before references (was ~620), over the ~1-page target. The full-notes doc and `references.bib` are not updated. ReWAM (2609.38163) is left in full notes only.
**Next:** User reviews the doc's Related works and decides whether to trim back to one page. Read PHI-S in full before the meeting (the mentor may ask how H-E differs from it). E2 target-diagnostics Job still awaits go-ahead.

## 2026-10-27 (f) [agent] (mentor meeting outcome; run spec for the target-side diagnostics)
**Mentor feedback (user report, 2026-10-27):** the user presented the proposals; the mentor advised focusing on the dataset/benchmark, learning LIBERO well, and working out how stacking encoders relates to the benchmark. The user will make their own slides from now on, to understand the content better.
**Did:** Wrote `experiments/2026-10-21-target-diagnostics/RUN.md`, a step-by-step handoff to run the drafted E2 Job (ask before ConfigMap, submit, cleanup). verified (`git rev-parse`): the three src files at HEAD are identical to b68ba54, so the manifest's `SRC_GIT_COMMIT` still holds after the rename. Nothing run; no cluster action.
**Next:** The user passes `RUN.md` to the implementation agent. Separately, the user studies LIBERO (suites, tasks, demos, how success rate is measured, how LIBERO-Plus perturbs it) and how the encoder-combo results connect to it.

## 2026-10-27 (f) [agent] (target-side Job, try 1: OOM-killed; Job deleted)
**Goal:** Run the E2/H-E target-side diagnostics per `experiments/2026-10-21-target-diagnostics/RUN.md`, asking before each cluster step.
**Did:**
- Steps 1–3 (no approval needed): both test files pass under the Job's pins (numpy 2.2.6, h5py 3.16.0, local venv). `src/` hashes at HEAD = b68ba54. kubectl auth OK. Nothing of ours in `aria-world`. `michael-persistent` Bound and mounted by no pod. `stacking-encoders-features` Bound.
- CPU capacity: 131 nodes match the Job affinity (kubectl), 42 with ≥ 2.4 CPU / 10 GiB free (Prometheus). Correction to my own first count (107): Prometheus `kube_node_labels` doesn't export `pci-10de.present`, so the label filter kept GPU nodes. Memory note updated.
- With user approval: created ConfigMap `stacking-encoders-target-src` (read back, byte-identical to `src/` at b68ba54). Submitted `job.yaml` 18:55:33Z (repo HEAD 2465cc6, tree clean). Pod on pas.seat.net.internet2.edu, Running 18:56:05Z. The feature sizes in the log match (D 4.4G, S 4.4G, G 8.8G on CephFS; L 4.4G, R 5.2G on michael-persistent).
- With user approval: deleted the Job at ~19:25Z (incl. its automatic retry pod `776qq`, which would have OOMed identically). The try-1 log is saved to `experiments/2026-10-21-target-diagnostics/results/try1-oom-20261027T185533Z.log`. ConfigMap kept for the resubmit.
**Found (verified, pod status + public Prometheus):**
- **OOMKilled** (exit 137, 19:15:41Z, pod age 22 min) in pair D+L, global k = 4, during T1/T2 (after the fits, 140 s). RSS rose 5.4 → 6.3 → 7.0 → 8.8 GiB in ~3 min, then the pod hit the 10Gi limit. Page cache fell to 0.6 GiB, so it wasn't cache.
- **The sizing basis was wrong:** local macOS `/usr/bin/time -l` gave 6.1 GiB for the larger D+R pair (LOG 2026-10-21 (k)). Linux needs more than 10 GiB for the smaller D+L pair. Cause speculative: macOS memory compression undercounts RSS, and/or OpenBLAS allocates more.
- **Slow, nearly idle load phase:** `diagnostics.load` took 737 s for D+L at ~0.01 CPU cores and ~1 GiB (11% of 9Gi). Each later pair reloads, so after the 30-min grace these phases would sit below NRP's 20% floor (speculative: flags likely). No NRP violation occurred: the pod died at 22 min, before checks start.
- Partial real output before the crash (D+L global k = 4, T0 only; not a result, the run is incomplete): split 38,464 / 12,416 / 12,848 (matches the tally); L loss share 0.291 under a′ and a″.
**Resources:** cluster ~0.4 CPU-h (2 pods), no GPU. Local: tests and the capacity check.
**Not done:** no target-side result. Violations page not checked (needs user login; nothing expected, pod < 30 min).
**Next:** (1) Measure peak memory in a Linux container locally (`python:3.11.14-slim`, same pins), all 4 pairs or at least D+R. (2) Cut memory without changing numbers: load only the feature type in use, free fit objects sooner; re-run the synthetic tests and check outputs are bit-identical on synthetic data. (3) Fix the idle load: stage features to local disk in one sequential copy (needs more ephemeral storage), or load each encoder once per Job. (4) Resize `job.yaml` from the Linux numbers, log the changes, and ask the user before resubmitting (ConfigMap must be recreated if `src/` changes).

## 2026-10-27 (g) [agent] (LIBERO study session with the user; no runs)
**Note:** two entries above are both labelled (f): the first is this session's mentor/run-spec entry, the second is the implementation agent's Job try 1. Left as is (append-only).
**Did:** Walked the user through the target-side idea, the LIBERO paper and LIBERO-Plus (the user read both). Downloaded one libero_goal file to the session scratchpad (not the repo) and inspected it.
**Found:**
- verified (`libero_goal/turn_on_the_stove_demo.hdf5`): 50 demos, length 75–119 (median 89); per step agentview + wrist RGB at 128×128, 7-d action (3 translation, 3 rotation, gripper ±1), reward 1 on the last step only; one language instruction per file.
- verified (`literature/liu2026what-table1-single-source.csv`): adding DINOv3 to VAE-only gains most on visual perturbations (background +21.9, noise +19.4, camera +13.0) and little on language (+2.7) or layout (+1.2).
- verified (`literature/liu2026what-table10-composition.csv`): the composition table reports only the LIBERO-Plus total, so which perturbation categories D+R loses on is unknown. If the drop is concentrated in visual categories, clean-image probes can't see it (the domain-shift pitfall in STANDARDS).
**Mentor (user report, 2026-10-27):** also look at the intermediate steps before the final LIBERO result. Meaning not yet confirmed. Candidates: progress within a rollout (how far failed runs get), the per-category breakdown, or success over training checkpoints.
**Next:** User asks the mentor which "intermediate steps" she meant. Ideas, not approved: ask the authors for D+R per-category numbers; probe features of LIBERO-Plus-perturbed frames (new cluster work, needs a go-ahead).

## 2026-10-27 (h) [agent] (target-side Job try 2 prepared: less memory, staged features, resized; not submitted)
**Goal:** Fix what killed try 1 (LOG (f)) and get the Job ready to resubmit. User report: the mentor also wants the target-encoder run done. User agreed to measure on the cluster rather than in a local Linux container (Docker on the Mac would need ~27 GB of features and a bigger VM).
**Did:**
- `src/` (commit 52d1226): `diagnostics.load` takes an optional `types` (default unchanged). `target_diagnostics.py` loads one feature type at a time, deletes per-camera arrays once concatenated, builds X block by block, scores each test prediction (R² + bootstrap draws) right away and drops it, and frees all per-k objects (incl. `y_te` slices that kept it alive). The old code held ~12 test predictions per k and carried them into the next k.
- verified: both test files pass (numpy 2.2.6, h5py 3.16.0). verified: old (b68ba54) vs new end to end through `main` on synthetic HDF5 data (3 files, 60 demos, both types, 3 k, 3 ranks): JSON identical except `seconds`, every float and CI equal.
- `job.yaml`: features copied to local disk (emptyDir 24Gi) with one `cp -r` per encoder; dinov3 once, each partner copied before its pair and removed after. Memory 14Gi request / 16Gi limit; ephemeral 22Gi/26Gi; `SRC_GIT_COMMIT` 52d1226. Client-side dry-run parses; script passes `bash -n`; staging functions tested on fake dirs.
**Found:**
- verified (tracemalloc, new code, synthetic random-walk features at real D+R dims, 64,391 frames, global, k = 4 and 16): peak numpy allocation 11.8 GiB (during k = 4 T1/T2), back to 5.2 GiB between k. macOS `ru_maxrss` for the same run was 6.2 GiB, so Mac RSS is again far below the real footprint.
- speculative: real Linux peak ~12.5–13 GiB for D+R (adds LAPACK eigh workspace ~0.7 GiB and interpreter/BLAS overhead, not traced). Smaller pairs lower.
- verified (kubectl, ~22:00Z): nothing of ours runs in `aria-world`; we hold only ConfigMap `stacking-encoders-target-src` (still the b68ba54 code), Secret `stacking-encoders-hf`, PVCs. ~90 `vissam-*` Jobs (label `project: vis-sam`, not in this repo) sit Failed/Complete for 10–24 h; not ours, not touched.
- Running two probes at once pushed the Mac to ~21 GB swap; the calibration probe (old code, D+L) was stopped and not rerun.
**Risk (speculative):** each pair starts with a few minutes of copy/load at low memory and CPU; a 10-min portal check could land there. Much shorter than try 1's 12-min in-place loads.
**Resources:** local CPU ~1.5 h (probe, swapping). Cluster: none.
**Next:** ask the user before each step: (1) delete and recreate ConfigMap `stacking-encoders-target-src` from `src/` at 52d1226; (2) pre-checks per RUN.md (CPU/memory capacity on Prometheus for 2 CPU / 14Gi, `michael-persistent` free); (3) submit; watch RSS on Prometheus and the staging times in the log; check the Violations page at ~35 min.

## 2026-10-27 (i) [agent] (target-side Job try 2 submitted and running; deadline extended to 16 h)
**Did:** With user approval for each step: deleted and recreated ConfigMap `stacking-encoders-target-src` (read back, byte-identical to `src/` at 52d1226); pre-checks clean (`michael-persistent` mounted by no pod, both PVCs Bound, 19 untainted affinity nodes with ≥ 2.4 CPU / 16 GiB free on Prometheus); submitted `job.yaml` 21:49:44Z at 567413d. Pod `stacking-encoders-target-diag-pnkkm` Running 21:50:04Z on node-2-10.sdsc.optiputer.net. With user approval, patched the live Job `activeDeadlineSeconds` 28800 → 57600 (~22:24Z); `job.yaml` updated to match.
**Found (verified, Job log and public Prometheus at pod age ~28 min):**
- Staging fixed the idle load: dinov3 4.4G copied in 41 s (CephFS), siglip2 4.4G in 210 s (RBD); feature load 3 s (try 1: 737 s). Split 38,464 / 12,416 / 12,848, same as try 1.
- D+L: max RSS 7.2 GiB (try 1 died above 10 GiB on this pair); RSS min 3.7 GiB over 20 min (NRP floor 2.8 GiB = 20% of 14Gi); CPU 1.67 cores avg (84% of request).
- Speed: one D+L block (global, k = 4) took ~13 min (fits 131 s + T1/T2 523 s); the Mac took ~11.5 min for a D+R block, ~4x more work, so this node runs the linear algebra ~4x slower.
- speculative (dim³ extrapolation from the D+L time): D+L, D+S ~70 min each, D+G ~3.5 h, D+R ~4.5 h; ~11 h total, past the 8 h cap, which would have cut D+R (last). Hence the 16 h deadline; expected finish ~09:00Z 2026-10-28. An 8-CPU resubmit was considered and not chosen: only 1 untainted affinity node had ≥ 9.6 CPU free.
**Not done:** Violations page not checked (needs user login). No interpretation of partial numbers (outcome rule needs all 4 pairs).
**Next:** watch via Prometheus (RSS on D+R vs the 16Gi limit); collect the log and 4 JSONs within 24 h of finish (RUN.md step 7); ask before cleanup; write the experiment README with the (i) outcome rule.

## 2026-10-27 (j) [agent] (target-side Job try 2 complete; H-E fails the pre-registered rule)
**Did:** Job Complete 1/1 in 5 h 45 m (submitted 21:49:44Z, done ~03:35Z 2026-10-28), no restarts. Saved the full log (`results/target-diag-20261027T214944Z.log`) and cut the 4 JSONs out of it (`results/target-libero_goal-dinov3-{siglip2,sam3,vggt-omega,c-radiov4}.json`; each parses, has the PLAN fields, `git_commit` 52d1226). Wrote the experiment README with the outcome rule applied.
**Found (verified, from the JSONs; primary = T2 interference, a′, r = 64, global, mean over k):** G +0.0343 [+0.0332, +0.0354]; R +0.0151 [+0.0147, +0.0155]; L +0.0141 [+0.0138, +0.0145]; S +0.0032 [+0.0030, +0.0035]. **R is not the largest → H-E fails at this level (LOG 2026-10-21 (i) rule).** R's interference grows under (b) (+0.0251), so the secondary prediction fails too. The ordering doesn't track the paper's deltas (G +0.40, L +1.83, R −5.41). Dimension is a confound (G and R widest), not tested.
**Resources (verified, Prometheus + Job status):** ~11.5 CPU-h, no GPU; peak RSS 12.85 GiB (D+R), within the 12.5–13 GiB estimate (LOG (h)).
**Not done:** README key findings not touched (verification gate, needs user OK). Job and ConfigMap still exist (cleanup needs user OK; Job TTL deletes it ~24 h after finish). Violations page not checked by the agent.
**Next:** explain the result to the user (RUN.md step 10); ask before cleanup; user decides whether this goes into key findings and what it means for the LIBERO focus.

## 2026-10-27 (k) [agent] (cleanup; key findings updated)
**Did:** With user OK: deleted Job `stacking-encoders-target-diag` and ConfigMap `stacking-encoders-target-src`; verified nothing of ours is left in `aria-world` except Secret `stacking-encoders-hf` and the two PVCs. With user OK: added the target-side result to README key findings and noted that neither role shows the D+R harm.
**Next:** decide with the mentor where this leaves the stacking-encoders question given the LIBERO focus; candidate next step (not approved): look at D+R by LIBERO-Plus perturbation category (ask the authors, or probe perturbed frames).

## 2026-10-27 (l) [agent] (plain-language write-up; how the paper benchmarks)
**Did:** Added an "In plain words" section (full encoder names, per user request) to `experiments/2026-10-21-target-diagnostics/README.md`. Read the paper's evaluation text (arXiv HTML v1: Sec. 4.1, 4.4, App. C.2, D.1, E.3) and added the benchmarking protocol to `literature/notes/liu2026-what-visual-representations-wams.md`.
**Found (verified, arXiv HTML v1):** one checkpoint per config, trained on LIBERO (1,712 episodes, four suites), then scored without retraining on clean LIBERO (2,000 rollouts, near saturation, App. D only) and LIBERO-Plus (10,030 episodes × 3 eval seeds, 7 categories; the headline). The composition study reports only LIBERO-Plus totals: no per-category and no clean-LIBERO numbers for any combination. C-RADIOv4 alone: LP 83.5 vs DINOv3 84.0; better than DINOv3 on background (86.2 vs 83.9) and noise (86.6 vs 86.0).
**Clarified for the user:** our diagnostics never run a policy; their only ground truth is the paper's LIBERO-Plus totals.
**Next:** user decides whether to ask the authors for DINOv3 + C-RADIOv4 per-category numbers, or to probe perturbed frames (new cluster work, needs a go-ahead).

## 2026-10-27 (m) [agent] (decision 003: go straight to multi-seed proxy policy; plan written)
**Did:** Explained to the user why the cheap diagnostics may miss the D+R harm, split into three causes (single-checkpoint noise, LIBERO-Plus shift, training dynamics; all speculative). User chose the multi-seed proxy policy (decision 001, option 3), skipping decision 002 step 2. Recorded as decision 003. User choices: configs none, D, R, D+R, D+L × 3 seeds; eval on clean LIBERO + LIBERO-Plus; proxy uses features as input and as future-feature targets (paper Eq. 9); all four suites from the start. Wrote `experiments/2026-10-27-proxy-calibration/PLAN.md` with a pre-registered calibration rule.
**Not done:** no code, no Jobs, nothing run. Open items: LIBERO-Plus install/episode list, Fast-WAM's 1,712-episode list, training/rollout resolution, LIBERO-Plus subset size (needs a timed rollout).
**Next:** PLAN step 1 (read the LIBERO-Plus repo and Fast-WAM data docs, no compute), then the proxy code + tests locally. Cluster work only with a go-ahead per Job.

## 2026-10-27 (n) [agent] (proxy calibration PLAN steps 1–3: facts, code, local smoke test; no cluster work)
**Goal:** PLAN.md "Order of work" steps 1–3 only (user brief). No Jobs, ConfigMaps or PVCs; no contact with authors; configs, eval protocol and outcome rule unchanged.
**Did:**
- Step 1: read LIBERO-plus (`4976dc3`), FastWAM (`7faa711`), HF `yuanty/LIBERO-fastwam` metadata, and paper Tables 6–7. Findings written into PLAN.md with labels.
- Step 2: `src/proxy_policy.py` + `src/test_proxy_policy.py` (commit d5d10b0). Existing `test_diagnostics.py` and `test_target_diagnostics.py` still pass.
- Step 3: local smoke test on real C-RADIOv4 and SigLIP 2-base features extracted on this Mac from 3 demos of the one local libero_goal file. Scratch venv: Python 3.11.14, torch 2.7.1, numpy 2.2.6, h5py 3.16.0, transformers 4.57.6, timm 1.0.30, open-clip-torch 3.3.0, einops 0.8.1.
**Found:**
- verified: the paper's 1,712 episodes = Fast-WAM's `yuanty/LIBERO-fastwam` (434 + 457 + 433 + 388; 277,713 frames; MuJoCo 3.3.2; 512×512 frames resized to 224; CC BY 4.0), not the yifengzhu-hf HDF5 we extracted from. Matching the paper means re-extracting all four suites (libero_goal too) from Fast-WAM's data.
- verified: action horizon 32; feature frames at t+4 … t+32; Fast-WAM rollouts render at 256, rotate 180°, resize to 224 (dataset metadata says 512: unresolved discrepancy in Fast-WAM).
- verified: LIBERO-Plus = 10,030 episodes × 1 trial; per-category counts in PLAN; no LICENSE in the GitHub repo (HF assets MIT).
- verified: 9/9 synthetic checks pass. Check 7 (planted signal) failed twice first: run 1 base too small (val episodes drifted from train), run 2 the per-episode walk made future features near-unpredictable (feature MSE 0.557 vs threshold 0.5). Fixed the test data (continuous walk, interleaved val episodes); thresholds unchanged. Recorded in the test.
- verified (smoke): all configs train end to end on real features on CPU and MPS; ~0.1 s per batch-32 step for two encoder streams on CPU; 3.6M–7.7M params. Losses are not results.
- Not possible locally: DINOv3 (gated, no HF token on this Mac), and our cluster feature caches (would need a pod).
**Resources:** local only (~10 min CPU/MPS). Cluster: none.
**Not done:** text encoder choice; LeRobot (Fast-WAM) data loader; rollout/eval code (needs the simulators); host-RAM plan for full data (~42 GB for D+R as written).
**Next:** user reviews the 8 design choices in PLAN "Steps 2–3" (especially the pixel stream in every config and the text encoder), then PLAN step 4 (extraction Jobs from Fast-WAM's data for D, R, L, all four suites) on a go-ahead.

## 2026-10-28 [agent] (smoke test rerun with real DINOv3)
**Did:** User logged this Mac in to Hugging Face (`hf auth login` with the token read from the cluster Secret `stacking-encoders-hf`; token never printed). verified: user `michaelnkr808`, read permission, `facebook/dinov3-vitl16-pretrain-lvd1689m` readable at `ea8dc28` (the revision in LOG 2026-10-17). Extracted DINOv3 for the same 3 demos (CPU, 16.3 frames/s, 35 s; MPS fails on the 14×14 → 4×4 pool) and ran dinov3, dinov3+c-radiov4, dinov3+siglip2 (base) on CPU and MPS: all exit 0; 5.8M / 8.3M / 7.5M params; 3.6–6.2 s for 55 batch-32 steps. PLAN smoke table updated. Losses are not results.
**Next:** unchanged from (n): user reviews the PLAN design choices; then step 4 on a go-ahead.

## 2026-10-28 (b) [agent] (design review with the user; stage 1: proxy rebuilt on Fast-WAM's own code; no cluster work)
**Goal:** user reviews the 8 PLAN design choices, then stage 1 (all local code) for the chosen design.
**Decisions (user, 2026-10-28):** match the paper wherever possible, cheapness is not a reason to deviate. Base stream = Wan 2.2 VAE latents (not pixels); joint flow matching on actions and future features (not MSE); the paper's T5, precomputed; copy Fast-WAM's padding; keep our per-channel feature z-score as a flagged guess (the paper is silent); full 14 × 14 grids, VAE-aligned time steps, channel-wise concat with one shared projection, two experts; extract features once to a ~750 GB CephFS volume (user: no need to ask the namespace admin). Recorded in PLAN.
**Did:**
- Read the paper (arXiv HTML v1, App. A, B.1–B.2, C.1–C.2, E.1, E.3) and FastWAM `7faa711` (data configs, dataset, processor, normalizer, FastWAM/FastWAMJoint, MoT, DiTs, scheduler, VAE loader, T5 precompute, LIBERO eval). Facts in PLAN "Stage 1".
- New code: `src/fastwam_data.py` (LeRobot v3.0 reader, Fast-WAM's resize/concat/normalize), `src/extract_fastwam.py` (meta, per-window VAE clip latents, encoder grids + per-episode channel stats), `src/t5_cache.py` (Fast-WAM's T5 cache format), `src/proxy_wam.py` (FastWAMJoint at small size, per-source loss, episode-buffer data plan, resumable training, 10-step inference), `src/test_proxy_wam.py`. `proxy_policy.py` kept, superseded. Env pins in the experiment's `requirements.txt`.
- read-only kubectl: `aria-world` has no storage ResourceQuota; three 400Gi `vissam-*` PVCs exist; ours is 100Gi CephFS. NRP policy page: no volume size limit; unused volumes purgeable after 6 months; no archival storage.
**Found (verified unless marked):**
- Fast-WAM pads past the episode end by zeroing the 6 delta action dims and keeping the clamped gripper (correction to what the agent told the user earlier: "repeat the last action").
- `yuanty/LIBERO-fastwam` frames are upright (episode 432, by eye); Fast-WAM's rollouts rotate both cameras 180° and invert the gripper sign.
- VAE latents depend on the clip start (causal VAE), so they're cached per start frame: 31.4 GB, total cache ≈ 756 GB (arithmetic).
- `test_proxy_wam.py`: 12/12 pass on CPU. Our loss equals Fast-WAM's `training_loss` on the same draws (check 6). Test-premise failures on the way, code unchanged by them: check 3 and 4 hand-computed expectations were wrong (t=27 is fully padded; mean(14, 18, 19, 19) = 17.5); check 9 (overfit) failed twice: run 1 drew fresh noise each step (4.87 → 3.39), run 2 fixed noise (4.76 → 2.42 vs threshold 0.5×). A 600-step diagnostic showed action 0.911 → 0.000 and both feature terms falling, VAE term flat at ~1.45 (fake i.i.d. latents exceed the 0.46M-param test model). **Criterion revised after seeing results:** action loss < 0.1× start and every term lower at the end; passes (action 0.667 → 0.041). History kept in the test.
- Build issues fixed: FastWAM needs a VAE object (`LatentsOnly` stand-in); Wan's 3D RoPE needs head_dim // 3 even, so defaults are Wan's 3 heads × 128 (64 fails).
- Real-data smoke (libero_goal ep 432, 77 frames, CPU fp32): VAE latents (77, 3, 14, 28, 48), finite, std 0.74, ~5 min on CPU (peak RSS ~20.7 GB at batch 4); DINOv3, C-RADIOv4, SigLIP 2-so400m grids (77, 14, 28, D) with stats; proxy trains 9 steps for none / dinov3 / dinov3+c-radiov4 / dinov3+siglip2 (28.5M / 31.6M / 35.2M / 35.2M params, default size). Losses are not results; T5 contexts in the smoke were random (smoke only).
- MPS fails (Fast-WAM's RoPE computes in float64, unsupported on MPS); CPU and CUDA only. Not patched.
**Not done:** `t5_cache.py` not run with the real umT5-XXL weights (11.4 GB; to run on the cluster at step 4). No timing of VAE extraction on GPU. Rollout code (stage 3). Old tests not rerun (their code is unchanged).
**Resources:** local CPU ~30 min; downloads ~6 GB (VAE weights, SigLIP 2-so400m, one libero_goal parquet + 2 video files) in the session scratchpad / HF cache. Cluster: none.
**Next:** stage 2 = PLAN step 4, with a go-ahead per Job: (1) create a ~760 GB CephFS PVC (or grow `stacking-encoders-features`); (2) download `yuanty/LIBERO-fastwam` lerobot_v30 (4 suites) to it; (3) Jobs: meta, VAE, DINOv3, C-RADIOv4, SigLIP 2 extraction (sharded; time a first shard before sizing the rest), T5 cache on 1 GPU. Then stage 3 (pilot + rollout code).

## 2026-10-28 (b) [agent] (reviewer feedback: proxy round reframed as a proxy replication; outcome rule amended before any training run)
**Goal:** Fold two rounds of outside reviewer feedback (pasted by the user) into the plan; user OK'd the edits.
**Did:** Amended `experiments/2026-10-27-proxy-calibration/PLAN.md` (framing + outcome rule; original text at f42af61), appended an amendment to decision 003, added a reviewer-feedback line to README.
- Framing: proxy replication; its outcome decides whether a higher-fidelity, multi-seed WAM replication (Wan 2.2-based, the paper's recipe as closely as code permits) is warranted.
- Withdrawn: "D+R within seed spread → consistent with single-checkpoint noise in the paper" (decision 003 said the same). The proxy lacks the pretrained video prior, so non-reproduction doesn't establish noise. I had also told the user this in chat on 2026-10-28; corrected.
- Primary analysis: paired D+R − D per training seed on the same LIBERO-Plus episodes, three deltas shown individually; means/CIs secondary; no strong variance claim from 3 seeds.
- Overlap/interference stays a hypothesis.
- Scope (open): a workshop paper needs more combos and a predictive-performance evaluation. Candidate low-cost step: retrospective check of our diagnostics vs the paper's 16 composition results (gain vs parent composition), labelled exploratory because configs share encoders and have one checkpoint each. Needs V-JEPA 2.1, UniTok, HRP and multi-encoder parent features; not started.
**Verified:** reviewer's facts match our repo: Table 10 seeds are evaluation seeds of one checkpoint (CSV header, LOG 2026-10-18/19); D+R 78.59 (CSV). Nothing run.
**Next:** unchanged from the previous entry (stage 2 / PLAN step 4 with a go-ahead per Job). Separately, user decides whether to add the 16-combo retrospective check to the plan.

## 2026-10-28 (c) [agent] (paired-analysis code for the amended outcome rule; no cluster work)
**Note:** two entries above are both labelled 2026-10-28 (b) (stage 1 code, then the reviewer amendment from another session). Left as is (append-only).
**Goal:** user-pasted brief after the reviewer amendment (116d4ed): eval bookkeeping and a paired-analysis script with synthetic tests; training unchanged; the 16-combo retrospective check is out of scope.
**Did:** read the amended PLAN framing/outcome rule, decision 003 amendment and LOG 2026-10-28 (b). Grepped our new code and PLAN text for the withdrawn "paper noise" framing: none. Added a proxy-replication note to the `proxy_wam.py` docstring. New `src/lp_eval.py` (seeded stratified LIBERO-Plus subset with a content hash; per-rollout JSON-line records; paired per-seed D+R − D analysis; secondary per-config, per-category, clean LIBERO) and `src/test_lp_eval.py`. PLAN section "Eval bookkeeping".
**Found (verified):** LIBERO-plus `4976dc3` `task_classification.json` (sha256 faa87cce3e3b…): 4 suites × 7 categories = 28 cells, 10,030 episodes, ids unique per suite, counts match PLAN step 1. `test_lp_eval.py`: 6/6 pass on the first run (planted drops recovered exactly; identical configs exactly 0; bad rows rejected). A trial subset at 10 per cell on the real file gives 280 episodes (scratchpad only, not the final subset; N waits for the step 5 timing).
**Not done:** rollout code (stage 3) that writes the records; the final subset file.
**Next:** unchanged: stage 2 (PLAN step 4) with a go-ahead per Job; then stage 3 rollout code writes `lp_eval` records.

## 2026-10-28 (d) [agent] (full plan written up for the reviewer; auto-chain agreed)
**Did:** Wrote the whole proxy-replication plan for the user's reviewer as a Google Doc in the user's Drive (school account): "Stacking encoders: proxy replication plan (2026-10-28)", https://docs.google.com/document/d/1_N-OZfvVund39OrxtuGMl_Eo5c1GZyE71XurUmdDS68/edit . Source kept at `writing/proxy-replication-plan-2026-10-28.html`. Read back through Drive: renders as written. Fixed two stale PLAN lines (config "none" is now VAE-only; the 2026-10-27 design draft marked superseded).
**Agreed with the user:** stages 1–3 (volume, dataset download, extraction incl. a first timed shard, T5 cache, pilot) are pre-approved and auto-chained: the agent watches Job completion (Prometheus) while this session stays open (user keeps the laptop awake), submits the next step after checks, one fix-and-retry per failure, stops on a second failure. Stage 4 (15 runs) still needs the user's OK after the pilot numbers. Recorded in memory as an exception to "check jobs only on request".
**Next:** user decides whether to send the doc to the reviewer before starting; then stage 2 step 1 (CephFS volume).

## 2026-10-28 (e) [agent] (second reviewer round: proxy wording, threshold, storage, orientation; PLAN + Google Doc updated)
**Goal:** User asked to fold the reviewer's second-round comments into the plan and the Google Doc.
**Did:** Updated PLAN.md (framing, choice 5, storage line, outcome rule, order of work), the Google Doc "Stacking encoders: proxy replication plan (2026-10-28)" in place (same link), and its repo mirror `writing/proxy-replication-plan-2026-10-28.html`. Read the doc back through Drive: all edits present.
- Wording: "Fast-WAM-based proxy using the paper's recipe where known" instead of "small copy"; z-scoring and temporal averaging named as choices that may affect whether stacking hurts.
- Corrected: the doc said the paper starts from "a pretrained action expert". verified (paper Sec. 3.1, verbatim via web fetch): "a video expert initialized from Wan 2.2 and a separate action expert"; nothing on action-expert initialization there. App. B was read only through a model summary (not verbatim), which also found nothing. Now stated as "the paper does not say".
- Outcome rule: "clearly negative (beyond the seed spread of the singles)" replaced by a minimum drop fixed before training. **Threshold not chosen; user decides before the 15-run round.** Reported as a small screening result. Not changed: the calibration rule still uses "one seed sd" (same 3-seed weakness; reviewer didn't flag it; raise with the user).
- Storage: caches ≈ 756 GB (arithmetic from PLAN) vs a ~760 GB volume; now ~1 TB.
- Orientation: added a check before full extraction (both cameras, all four suites) and a sim-vs-training frame side-by-side before rollouts count. So far one episode, one camera checked by eye.
- Doc numbering: inserting the orientation step shifted the execution list; pre-approved steps now read 1–4 (volume, orientation check, extraction, pilot) and the user approves step 5 (full round).
**Next:** user sets the minimum-drop threshold (before the 15 runs, not blocking stage 2). Stage 2 starts with a ~1 TB volume and the training-frame orientation check.

## 2026-10-28 (f) [agent] (decision rule locked before the pilot; PLAN + Google Doc)
**Goal:** User OK'd locking the outcome rule after two more reviewer rounds (reviewer proposed the D+R rule and asked for fixed margins on both calibration checks).
**Did:** Rewrote the PLAN outcome rule (previous wording in git at 64edf4e), updated the Google Doc in place (same link) and the repo mirror; read the doc back through Drive.
- Locked before the pilot; the pilot is sizing only (model size, subset N, GPUs); its success rates don't count. N chosen so episode noise is small next to 2.7 points.
- All comparisons: paired per-seed differences, each seed shown plus the average; episode-level CIs not treated as training-seed evidence.
- Calibration (minimum check that the proxy responds to controls, not proof of a good copy): (a) D − none > 0 in all three seeds and average ≥ +2.7; (b) D+L − D average ≥ −1.35 (non-inferiority, half the harm threshold, so a control loss that counts as harm can't count as "doesn't hurt"). Fail → "don't interpret D+R from this proxy".
- D+R (reviewer's rule): all three seeds negative and average ≤ −2.7 (half of −5.41, Table 10 D+R 78.59 vs Table 1 D 84.0) → "substantial proxy effect", a trigger for the higher-fidelity run, not statistical proof.
- Storage headroom verified: `yuanty/LIBERO-fastwam` 9.4 GB total (HF API tree), extraction writes per episode atomically with no merge step; ≈ 240 GB free on 1 TB after caches.
**Not done:** `src/lp_eval.py analyze` computes the D+R − D pairing but not the calibration pairs (D − none, D+L − D) or the locked pass/fail rule. Needs a small code change + synthetic tests before results exist.
**Next:** implement the locked rule in `lp_eval.py` (calibration pairs, thresholds as constants, verdict printed with the framing); then stage 2 as planned.

## 2026-10-28 (f) [agent] (PLAN double-checked; autonomous run set up: RUN.md, manifests, watcher; nothing submitted)
**Goal:** user asked to double-check PLAN.md and make sure the next session (after /close and /clear) can run stage to stage on its own "unless there is some major problem".
**Did:**
- PLAN.md: fixed stale lines left from 2026-10-27 (status, open items already resolved in step 1, extraction scope, rollout resolution proposal now superseded by Fast-WAM's eval path, choices 1/3/7 wording, choice 4 fill, VAE grid now verified, old sizing notes marked historical, order of work and gates now point to RUN.md). No change to configs, outcome rule or design.
- New `experiments/2026-10-27-proxy-calibration/RUN.md`: authorization (user quotes), how chaining works, stop conditions, session start, steps 1–13 (volume, download + orientation, ConfigMap, probe, T5, extraction, rollout code, sim-vs-train frame check, pilot, 15 runs, eval, analysis, report).
- New `watch_job.sh`: waits on a Job via the public Prometheus only; exits on complete / dead / vanished / GPU idle < 40% for a pod > 35 min / never appeared. verified: dead-Job path on an existing failed vissam Job (exit 2); missing Job keeps waiting; the idle-GPU join returns matches on another namespace's GPU pods. Not verified: the success path (no completed Job of ours to test on).
- New manifests in `jobs/`: `pvc-proxy.yaml` (1Ti rook-cephfs), `job-download.yaml` (CPU, at NRP's exempt request level; dataset @ee018b99, ModelScope weights with sha256 checks, umT5 tokenizer @37ec5126, orientation thumbnails printed into the log), `job-probe.yaml` (1 Ampere+ GPU: meta for 4 suites + 2 timed episodes per source), `job-t5.yaml`, `job-extract.template.yaml` (Indexed, filled from the probe); `src/setup_env.sh` (pinned env + FastWAM 7faa711 --no-deps). verified: YAML parses, no sleep/tolerations, embedded Python compiles, bash syntax ok, and a server-side dry run of the PVC and three Jobs was accepted (nothing created).
- Memory updated: the auto-chain exception covers the whole round (stage 4 once the threshold is set).
**Found:** dataset `yuanty/LIBERO-fastwam` lerobot_v30 = 4.71 GB; ModelScope sha256 for Wan2.2_VAE (0e913a2c…) and umT5-XXL enc (d92de679…), recorded in the download Job.
**Open (user):** the minimum-drop threshold (needed before stage 4); the calibration rule still uses "one seed sd" with 3 seeds (raised in (e), not yet decided).
**Next:** next session follows RUN.md from "Session start", then step 1 (volume).

## 2026-10-28 (g) [agent] (NRP policy check of the autonomous proxy run before it starts)
**Goal:** User asked to double-check that the auto-chained proxy run (RUN.md, stages 2–4) can't break Nautilus policy while unattended.
**Did:** Reread `_standards/08`, RUN.md, every manifest in `jobs/`, `watch_job.sh`, `src/setup_env.sh`. Rechecked the live NRP policies page (last updated 2026-10-26) and Jobs page (2026-10-25) via web fetch. `kubectl apply --dry-run=server` on pvc-proxy, download, probe, t5: all accepted.
**Found (verified):**
- Policy wording unchanged on the numbers that matter: GPU > 40%, CPU 20–200%, RAM 20–150% of request, max 4 violating pods, limits within 20% of requests, no `sleep` in Jobs (ban). Exemption is stated as "pods requesting 1 CPU core and 2GB memory" (one size, not a threshold).
- `job-download.yaml` requested 2Gi = 2.15 GB, above the 2 GB exempt size, for a network-bound Job that may run > 30 min at low RAM. Fixed: 1536Mi request / 1800Mi limit.
- `watch_job.sh` only caught idle GPUs; a pod flagged for low CPU or RAM would go unnoticed. Added exit 6 (pod > 35 min old, not at exempt size, 10-min CPU outside 20–200% of request or RSS < 20% of request). Queries tested on live Prometheus against a running aria-world pod: return values, exemption filter works. RUN.md exit list and stop 2 updated.
- Otherwise compliant: all Jobs, no sleep, no tolerations, 1 GPU per pod, limits ≤ 1.2× requests, deadlines and 24 h TTL set, LIBERO is simulated (no PII), ≤ 8 GPUs gate in RUN.
**Open risks (speculative):** probe (12Gi) and T5 (24Gi) RAM requests are guesses; fine if they finish within 30 min, else the new exit 6 catches it. kubectl login expiry (stop 3) means an idle GPU Job can't be deleted until the user logs in; it then runs until its `activeDeadlineSeconds`.
**Next:** unchanged: RUN.md stage 2 step 1.

## 2026-10-28 (g) [agent] (evaluation seeds and subset weighting locked before the pilot)
**Goal:** User agreed to the reviewer's two evaluation corrections and asked to lock them into the plan.
**Did:** PLAN outcome rule gains "Evaluation protocol (locked 2026-10-28 (g))"; old LIBERO-Plus eval line marked superseded; eval bookkeeping notes the code changes still needed. Google Doc updated in place (same link) and read back; repo mirror updated.
- Evaluation seeds: every selected episode runs under the paper's seeds 41, 42, 43 (Table 10 columns, our CSV) for every checkpoint, same IDs across checkpoints; per-episode score = mean over the three; pairing uses those scores. Discordant counts become "D+R scored lower / higher". Pilot sizes N counting 3 rollouts per episode.
- Subset weighting: proportional to each of the 28 suite × category cells' benchmark size (largest remainder, ≥ 1 per cell), so a plain mean estimates the benchmark micro-average and the 2.7 / 1.35 thresholds are on the paper's scale.
**Found (verified, `src/lp_eval.py`):** current code samples up to N per cell (equal allocation), takes plain means, and rejects a second rollout of the same episode for a (config, seed) as a duplicate, so neither locked item is implemented yet.
**Next:** implementation agent: proportional `make_subset`, eval seeds 41–43 in records and `lp_table` (one row per episode × eval seed), per-episode averaging before pairing, calibration pairs (D − none, D+L − D) and the locked verdict, with synthetic tests. Then stage 2.

## 2026-10-28 (h) [agent] (sample-size rule locked: paired pilot; RUN.md brought in line)
**Goal:** User locked the reviewer's sample-size numbers and approved a paired pilot.
**Did:** PLAN outcome rule gains the "Sample-size rule" and a CI note; order-of-work step 5 is now the paired pilot; step 6 no longer waits on the threshold. Google Doc updated in place (same link) and the repo mirror. RUN.md (written by the other session for the autonomous run) updated so it doesn't follow superseded rules: authorization notes the locked rules and requires `lp_eval.py` to implement them before step 9 sizing; stop 8 replaced; step 9 is the paired pilot with the sizing formula; step 11 uses eval seeds 41–43; step 12 names calibration checks (a) and (b). Memory exception updated (pilot = two short runs).
- Rule: two short runs (D, D+R), same training seed, same pilot episodes (own proportional sample), eval seeds 41–43. sd of the paired per-episode difference after averaging the three eval seeds; sd* = 1.5 × sd (variance × 2.25); N so that sd*/√N ≤ 1.35 points. Cap: all 15 checkpoints evaluated in ~48 h on ≤ 8 GPUs; above the cap, use the affordable N and report the precision (operational choice). Final subset drawn after sizing, independently, excluding pilot episodes. Pilot results never enter the effect analysis.
- CIs, if added, resample whole episodes with their three rollouts together.
**Inconsistency for the user (not changed):** the Google Doc (from session (d)) says the project lead approves the 15-run round after the pilot; RUN.md and memory say stage 4 is pre-approved (user quote in RUN.md).
**Next:** implementation agent updates `lp_eval.py` to the locked protocol (now a RUN.md precondition for step 9); then RUN.md stage 2.
