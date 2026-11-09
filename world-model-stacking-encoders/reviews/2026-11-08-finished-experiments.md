# Review of the five finished experiments and their code

Reviewer: audit-done worker `[agent, review]`, 2026-11-08. Read-only audit; no experiment was rerun on the cluster and no kubectl was used. Labels: **verified** = I checked it against a file, a git object, or a run I did this session; **recalled** = from memory, not checked; **speculative** = my judgement or a hypothesis.

## Scope and base commit

- Base: `main` at commit `bde59a7` (the worktree started at `729c3c6`; I ran `git merge --ff-only bde59a7`, and `git rev-parse HEAD` then printed `bde59a7e271fd13d03be88c12eb1e152fd2b0855`) (verified).
- Read: `CLAUDE.md`, `README.md`, `_standards/01` to `08`, `world-model-stacking-encoders/STANDARDS.md`, `README.md` (Key findings, lines 13-29), `PLAN.md`, `decisions/001`, `decisions/002`, the direction `LOG.md` entries 2026-10-15 to 2026-10-28 (full text of the entries on these five experiments), all five experiment folders (PLAN, LOG, README, Job yamls, `results/`), and `src/extract_features.py`, `src/diagnostics.py`, `src/test_diagnostics.py`, `src/target_diagnostics.py`, `src/test_target_diagnostics.py` (verified).
- Checked every number in Key findings and in the experiment READMEs against the result JSONs and logs with small scripts (kept in the session scratchpad, not in the repo). The scripts only read `results/*.json` and `results/*.log`.
- Not checked: `src/bench_fastwam_step.py` (the benchmark script itself), the Fast-WAM code, the cached feature files on the PVCs, the paper PDF (I used the repo's transcribed CSVs `literature/liu2026what-table{1,10}-*.csv`), anything on the cluster.
- Code identity: `src/diagnostics.py` at HEAD differs from the tally-time version (`7dc80ca`) only by the optional `types` argument of `load()` (verified, diff of the two blobs). `src/target_diagnostics.py`, `src/test_target_diagnostics.py` and `src/diagnostics.py` at HEAD are byte-identical to `52d1226`, the commit the target-side Job ran (verified, `git diff --stat 52d1226 HEAD` empty for these paths). `src/extract_features.py` at HEAD is byte-identical to `9e3bdb2`, the last tally extraction commit (verified).

## Summary

**Bottom line (verified unless marked):**

1. **No number in Key findings or in the experiment READMEs disagrees with the result files.** I spot-checked every number that appears in Key findings (`README.md:15-20, 28`) and the tables of the tally, pilot-diagnostics and target-side READMEs. All matched to the rounding shown. Two small wording issues (CPU-minutes vs wall minutes, and a "most" that sits inside overlapping CIs) are in the Fix list.
2. **I found no data leakage and no wrong CI unit in the code.** Splits are by demo, standardisation and the z-scoring statistics come from train frames only, ridge strength is chosen on a separate validation split, and the bootstrap resamples whole test demos with the same draws for every predictor. The CIs only cover test-demo sampling (not fitting variance); the LOG and target README say so, the tally README and Key findings do not.
3. **The experiments ran as planned, with disclosed deviations.** The deviations I found (DINOv3 stored from fp32 compute, extraction in three tries with two code commits, only 4 of 7 pairs, resid dropped after the pilot, target-side Job try 1 OOM and a rewritten try 2, test data changed after a failed check) are all in a LOG or README. Four of the five PLAN.md files are marked as reconstructed after the fact; the README predictions they were rebuilt from were committed before the results in all three cases I could date (verified, `git log --follow`).
4. **The weak point is what the measures can distinguish, not how they were computed.**
   - `gain(B|D)` is nearly the partner's own probe score, and an input-side ridge probe that is tuned on validation can hardly go negative, so "gain does not flag D+R" was close to guaranteed before the run (the pre-registration itself says this, LOG.md:(i)). Across the four partners gain(B|D) orders them the same way as B's solo R² except L and G, which are within noise.
   - The target-side "interference" under the primary scaling is ordered exactly like B's share of the raw loss (G 0.717 > R 0.352 > L 0.288 > S 0.116 gives interference G > R > L > S), i.e. it mostly measures how large B's raw feature changes are next to D's. Under the equalised scaling all four partners land between +0.021 and +0.029. So "R is not the largest" says R's raw features are smaller-scale than VGGT-Ω's, not that target competition is absent.
   - No control partner (a noise or duplicate B, a dimension-matched B) was run, and the STANDARDS baselines (B's solo score, objective-overlap label, random ranking) were not computed.
5. **Tests pass but `pytest` finds none.** Both test files are scripts with `main()` and no `test_*` functions, so `python -m pytest src/test_diagnostics.py src/test_target_diagnostics.py -q` prints `no tests ran in 0.11s`. Run as scripts under Python 3.11.14-like env (numpy 1.26.4), both print `all ok` (verified). The default `python` on this Mac does not exist and `python3` (3.14.3) has no numpy.
6. **Compute rules in 08 were followed on the manifests I could see** (Job not pod, `backoffLimit` 1, `activeDeadlineSeconds` set, no sleep, limits within 20% of requests). CPU and RAM utilisation against the 20-200% / 20-150% ranges is recorded only for the tally extraction try 3 and the target-side Job; for the pilot extraction, the pilot diagnostics and the three tally diagnostics Jobs it is not recorded, so I cannot confirm it.
7. **Is the direction alive?** Speculative: yes, but with no positive evidence yet. The cheap tests so far could not have shown the effect, so they do not kill the idea; the decisive pieces are the seed-noise and replication runs and a measure computed on perturbed frames. Details in "Does this line look promising?".

## 2026-10-16-fastwam-step-bench

**What it is:** a 1-GPU timing and memory benchmark of one Fast-WAM training step (random weights, synthetic LIBERO-shaped batch). It is not a finding; the README says so (`README.md:32`).

**1. Ran as planned?** Mostly, with the following deviations, all disclosed:

| Deviation | Where disclosed |
|---|---|
| PLAN.md is reconstructed 2026-11-03 from the README | `PLAN.md:3` |
| Planned L40/L40S (and an A40 twin); neither started; the result is from an A6000 Job (`job-a6000-lowcpu.yaml`, 1 CPU / 16 Gi) | `README.md:12-13, 16-24, 41-45` |
| Only micro-batch 1 and 2 measured. B=4 crashed on a CUDA-graph shape error caused by the benchmark design, B>=8 not reached | `README.md:62` |
| Gradient checkpointing unsupported by the compiled path, so the "checkpointing on" predictions are untestable | `README.md:63` |
| Single run, median of 5 iterations, forward and backward only, no optimizer step | `README.md:55` |

- The predictions were written before the run: the README exists in `1eaecbf` (2026-10-16 12:29), the run finished 2026-10-16 23:40Z (verified, `git log --follow`).
- The plan said "stop at the first out-of-memory". The run ended on a shape error, not on memory (`README.md:62`), so the memory-limit part of the question is unanswered.

**2. Numbers vs result file (verified, `results/job.log`):**
- B=1: 0.808 s, 1.24 samples/s, VAE 0.251 s, peak 23.9 GiB (log line 386). B=2: 1.525 s, 1.31 samples/s, VAE 0.479 s, peak 29.0 GiB (line 387). Params 6.725B total, 6.021B trainable, weights 12.5 GiB (line 385). GPU is `NVIDIA RTX A6000` (line 9). All match `README.md:58-61`.
- The derived figures are labelled speculative in the README and I recomputed the arithmetic: 2.56M samples / (8 x 1.3 samples/s) = 246,000 s = 68 h, which matches "~68 h of fwd+bwd" (`README.md:67`). The "+5.1 GiB per sample" is the difference of the two peaks (29.0 - 23.9), a two-point slope.
- Direction README: the only claim is the Key-files line "A6000: ~1.3 samples/s/GPU, unreviewed, not a finding" (`README.md:` Key files). Consistent.

**3. Method:** fine for its purpose, but the 3-3.5 day full-run figure rests on two points, ignores optimizer, ZeRO communication and evaluation (`README.md:67`), and the open question about micro-batch 16 per GPU is unresolved (`README.md:74`). Do not quote it as a measurement of a full run.

**4. Tests:** none apply (benchmark script not audited).

**5. Standards:** Job, `backoffLimit: 1`, `activeDeadlineSeconds: 10800`, no sleep, limits within 20% (`job-a6000-lowcpu.yaml:13-15, 61-63`: 1 CPU/1, 16 Gi/19 Gi, ephemeral 20/24 Gi) (verified). GPU utilisation and host RAM use were not recorded. The pod ran 24 min 49 s (23:15:39Z to 23:40:28Z), under NRP's 30-minute no-check window (`_standards/08`, "How the portal checks"), so the 20-150% RAM range could not have flagged it (verified arithmetic). The log was pulled ~19 h after completion, with a 24 h TTL (`README.md:53`); close to losing it.

## 2026-10-18-extract-libero-goal

**What it is:** first extraction Job: SigLIP 2-L and C-RADIOv4-SO400M features for libero_goal on one GPU. Feeds the pilot diagnostics.

**1. Ran as planned?**
- Data: libero_goal, 10 files, 500 demos, 63,728 steps x 2 cameras = 127,456 frames per encoder (verified, sum of the per-file counts in `results/job.log:172-181`, and the same counts for RADIO at 184-193).
- Encoders and variants as stated: `google/siglip2-large-patch16-384` at 384 px, `nvidia/C-RADIOv4-SO400M` at 448 px (`results/job.log:171, 183`). Compute dtype fp16, RADIO under autocast (`job.yaml:60-63`).
- Deviations / misses against the README predictions (`README.md:29-33`):
  - Frames 127k vs predicted 89k. Explained (`README.md:36`).
  - Storage 4.4 + 5.2 = 9.6 GB vs predicted ~7.2 GB. The README gives the actual figures (`README.md:39`) but does not call out the miss.
  - Throughput: SigLIP 2-L ~142 frames/s (inside the predicted 100-300), C-RADIOv4 ~62 frames/s, **below** the predicted 100-300 (`README.md:38`). Reported, not flagged as a miss.
- The predictions were committed in `89fb593` (2026-10-18 23:53 PDT) which is the minute the Job started (job started 06:53Z) (verified, `git log --follow`).
- Not disclosed in this README: the dataset is pulled with `revision="main"` (`job.yaml:55`), so the exact data version is not pinned. It is disclosed in `data/README.md:13`.

**2. Numbers vs result file:** 127,456 frames, 142 and 62 frames/s, TITAN RTX, 53 min: all match `results/job.log` (lines 9, 172-193; the Job ran 06:53 to 07:47Z per README). GPU utilisation 90-99% is from Prometheus and was not rechecked by me (the log does not contain it). The direction README does not cite this experiment's numbers.

**3. Method (`src/extract_features.py`):** 
- Frames are 128x128 in the HDF5 (`data/README.md:16`), rotated 180 degrees (`extract_features.py:240-241`; orientation was checked by eye on one file, LOG 2026-10-17 (c)), then bicubic-upsampled to the encoder's resolution (`extract_features.py:50-57`): DINOv3 224, SigLIP 2 384, C-RADIOv4 448, VGGT-Omega 512, SAM 3 1008. So the upsampling factor ranges from 1.75x to 7.9x. This is not listed as a caveat in the tally README (speculative impact: it could favour or hurt encoders unevenly).
- The finiteness check runs before the cast to fp16 (`extract_features.py:246-249`), so a value above 65,504 would become inf in the stored file without being caught there. The downstream results are finite, so this did not happen in these runs (verified: all result JSONs are finite).
- Storage is fp16 for every encoder (`extract_features.py:14, 249`), whereas the metadata field `dtype` (`extract_features.py:256`) and the tally/data READMEs list DINOv3 as "fp32". That column is the compute dtype, not the stored dtype (same file sizes, 4.4 G for both siglip2 and dinov3). Easy to misread.
- The default SigLIP 2 variant in the script is `google/siglip2-so400m-patch14-384` (`extract_features.py:94-96`), but the Job passes `--variant google/siglip2-large-patch16-384` (`job.yaml:60`). The file metadata records which was used, so this is only a trap for someone rerunning without `--variant`.

**4. Tests:** none for this script (checked locally on one demo, `README.md:13-17`).

**5. Standards:** Job, `backoffLimit: 1`, `activeDeadlineSeconds: 14400`, no sleep, limits within 20% (`job.yaml:12-14, 71-73`: 2 CPU/2, 10/12 Gi, ephemeral 14/16 Gi), image tag pinned, pip pins and `pip freeze` in the log (verified). GPU use above 40% is recorded; CPU and RAM use against the 20-200% / 20-150% ranges is not recorded for this 53-minute pod (could not verify). Estimated vs actual GPU-hours logged (`README.md:41`).

## 2026-10-19-diagnostics-libero-goal

**What it is:** the pipeline pilot of `diagnostics.py` on one pair (SigLIP 2-L vs C-RADIOv4). The README says plainly that it is a pipeline pilot, not a test of any hypothesis (`README.md:5`).

**1. Ran as planned?**
- Data and split as planned: 63,728 steps; train/val/test frames 38,464 / 12,416 / 12,848 (verified in both JSONs).
- Deviations, all disclosed:
  - Submit 0 failed in ~1 s (no `free` in `python:slim`); run 1 at `d6e3905`, run 2 at `a89ece6` with a changed residual map (`README.md:42-45`; JSON `git_commit` fields match, verified).
  - The residual probe `resid(B|A)` failed (-0.0002 in all four cells in both runs) and was dropped by the user's decision on 2026-10-19; `gain(B|A)` and demo-level bootstrap CIs replaced it in `7dc80ca` (`README.md:67-71, 88`). So the pilot JSONs contain `resid` fields and no CIs, and **the current `src/diagnostics.py` cannot regenerate them** (disclosed at `README.md:88`).
  - A synthetic-test bug in the first residual version was caught and fixed before the runs (`README.md:20`).
- The pilot's predictions (`README.md:34-38`) are compared with the outcome (`README.md:73-77`); three of them missed (single-encoder R2 above 0.6, CKA above 0.7, one gain above 0.05) and the README says so.
- The README was committed with the Job (`346dc48`, 2026-10-19 03:23) before the runs (verified).

**2. Numbers vs result file (verified, `results/run1-...json` and `run2-...json`; the two runs are identical except for the resid fields):**
- Proprio 0.2793; SigLIP 2 global 0.6092, C-RADIOv4 global 0.6696, both 0.6777; mean 0.6454 / 0.6680 / 0.6824. gain(RADIO|SigLIP) global 0.0686, gain(SigLIP|RADIO) 0.0081; mean 0.0371 and 0.0145. pred(A to B) 0.7915, 0.7185, 0.8366, 0.7626. CKA 0.717 / 0.842 (global), 0.824 / 0.883 (mean), 0.883 / 0.943 (grid); self = 1, shuffled about 0. Every value agrees with the README tables (`README.md:51-65`) to the digits shown.

**3. Method:** see "Code review". For this run: the headline "RADIO adds +0.069 on SigLIP but SigLIP adds only +0.008 on RADIO" is a single split with no CI (the README says "Not measured: uncertainty", `README.md:79`), so it should not be read as a result.

**4. Tests:** `test_diagnostics.py` of that date tested the resid map; the current file tests the replacement. Not relevant to this pilot's validity beyond the bug it caught.

**5. Standards:** `backoffLimit: 1`, `activeDeadlineSeconds: 10800`, limits within 20% (`job.yaml:12-13, 55-57`: 4 CPU/4, 16/19 Gi), CPU-only, pinned pip versions (verified). Run 2 took ~67 min with 4 CPU / 16 Gi requested; CPU and RAM use against NRP's ranges was not recorded (could not verify). GPU-hours 0 stated.

## 2026-10-19-feasibility-tally

**What it is:** decision-002 tally: DINOv3 (D) paired with SigLIP 2-L (L), C-RADIOv4 (R), SAM 3 (S), VGGT-Omega (G) on libero_goal; input-side CKA and gain(B|D) set beside the paper's Table 10 deltas.

**1. Ran as planned?** The pre-registration is LOG.md 2026-10-19 (e), lines 190-200, committed in `0cd09b5` together with the loaders, before any tally extraction ran (verified: `git blame` attributes those lines to `0cd09b5`; lines 190-318 of the LOG each belong to a single commit and I found no later edit).

| Item | Pre-registered | What happened | Disclosed? |
|---|---|---|---|
| Pairs | D with L, R, then S, G; V, U, H "only if cheap" | 4 pairs; V, U, H skipped | yes, `README.md:109-114` |
| DINOv3 dtype | fp16 | fp32 (fp16 overflowed on batch 0) | yes, LOG (f), `README.md:107` |
| Extraction | one Job | 3 tries, 2 code commits (`0cd09b5`, `9e3bdb2`), try 2 deleted for a 16 Gi RAM request that used under 20% | yes, `README.md:26, 116-125` |
| Feature type | global primary, mean secondary | as planned | |
| Matched dimensions | no control (pre-registered) | none; D+R 6,656-d vs D+L 4,096-d | yes, `README.md:17, 103` |
| Diagnostics code | `diagnostics.py` unchanged | `7dc80ca`, identical on all three Jobs (verified in each JSON, `git_commit`) | |
| Columns | CKA, gain(B|D), gain(D|B), paper delta | all present | but `README.md:89` says gain(D|B) was "not asked for"; LOG.md:197 lists it as a pre-registered column. Minor wording error. |

- n_frames 63,728, split 38,464 / 12,416 / 12,848, 100 test demos, seed 0, 1,000 bootstrap draws in all four JSONs (verified).
- The tally README (`README.md:11`) lists the hf ids and revisions; the JSON metadata matches (e.g. `facebook/dinov3-vitl16-pretrain-lvd1689m`, 224 px; VGGT-Omega metadata shows the first extraction commit `0cd09b5` because the file attributes are read from the first file only; verified).

**2. Numbers vs result files** (verified against `results/tally-libero_goal-dinov3-*.json`; I recomputed everything in `README.md:35-71` and Key findings):
- gain(B|D) global [95% CI]: L +0.0168 [+0.0130, +0.0208]; R +0.0371 [+0.0325, +0.0423]; S +0.0326 [+0.0275, +0.0379]; G +0.0138 [+0.0105, +0.0176]. The README shows +0.017, +0.037, +0.033, +0.014 with CIs that match to the digits printed (R's lower bound is 0.03248, so "+0.032" is right).
- gain(D|B) global: L +0.0575, R +0.0174, S +0.0288, G +0.0470. R2 D 0.6499; B: L 0.6092, R 0.6696, S 0.6536, G 0.6167; D+B: 0.6667 / 0.6870 / 0.6824 / 0.6637. Alphas as in the README table. Mean-type and single-probe CIs match.
- CKA global (agentview / wrist): L 0.576 / 0.792, R 0.583 / 0.854, S 0.559 / 0.643, G 0.213 / 0.267. Mean, grid values match (`README.md:49-56`). "R has the highest CKA with D in all 6 cells" is true (R beats the runner-up in every one of global, mean and grid on both cameras).
- Paper deltas: D+L 85.83, D+S 84.49, D+G 84.40, D+R 78.59 minus D alone 84.0 gives +1.83, +0.49, +0.40, -5.41 (verified against `literature/liu2026what-table10-composition.csv` and `table1-single-source.csv`).
- Sanity checks: proprio 0.2793 in all four JSONs; L and R single probes equal the pilot (0.609, 0.670); D's CI agrees across Jobs to ~1e-8 (verified).
- Direction README Key findings (`README.md:15-18`): every number matches. Wording issues are in the Fix list: "R adds the most action-relevant information" sits inside overlapping CIs with S (R [0.0325, 0.0423] vs S [0.0275, 0.0379]); the CKA margin of 0.007 in the quoted cell has no CI (stated in the tally README at `README.md:79`, not in Key findings).
- Resource figures in `README.md:121-129` (about 10.2 GPU-h; SAM 3 5.5 frames/s; VGGT-Omega 12.5 frames/s on the last tries) match the extract logs (`results/extract-try2-job.log`, `extract-try3-job.log`, verified lines for 11.9, 12.5 and 5.5 frames/s).

**3. Method** (see Code review for details): sound computation; the construct is limited.
- What the numbers say in my own arithmetic (verified from the JSONs, n = 4, so descriptive only): the Spearman rank correlation of each measure with the paper's order L > S > G > R is -0.4 for gain(B|D), -0.8 for B's solo probe R2, +0.8 for gain(D|B), +0.2 for global agentview CKA (lower CKA read as a better partner). gain(D|B) is simply R2(D+B) minus R2(B); it puts R last because R is the strongest solo probe (0.670), not because of any interaction. The tally README has gain(D|B) (`README.md:89-91`); Key findings leaves it out. If gain(D|B) is later quoted as the measure that "works", note that it was picked after seeing the table and that it is a restatement of B's solo score.
- Single split, single suite, single seed: the gain CIs cover test-demo sampling only (they do not include probe-fitting variance or other splits) (`diagnostics.py:204-239`). The tally README's Caveats (`README.md:98-107`) do not state this; the LOG does (2026-10-19 (c), "Limit").
- The probes see single frames and no language goal. libero_goal has the same scene for all ten tasks, so a frame alone is ambiguous about the goal early in a demo (speculative; it caps R2 around 0.65).
- Dimension mismatch is acknowledged and pre-registered (`README.md:103`). The Parked list has "Dimension-matched features" (`PLAN.md`).

**4. Tests:** see "Code review" and section 4 of the checks below.

**5. Standards:** manifests `diag-dlr-job.yaml`, `diag-dg-job.yaml`, `diag-ds-job.yaml`: `backoffLimit: 1`, deadlines 21,600 / 10,800 / 10,800 s, no sleep, 2 CPU / 2400m, 11 / 13 Gi (`diag-dlr-job.yaml:15-17, 68-70`) (verified). `extract-job.yaml`: 1 CPU / 1200m, 4 Gi / 4800 Mi, 1 GPU, `backoffLimit: 1`, deadline 46,800 s (`extract-job.yaml:16-19, 97-101`). Try 3 GPU 99-100% and RAM 85-89% of request are recorded (`README.md:126`, from Prometheus, not rechecked by me). The try-2 over-request (16 Gi for 1-4 GiB use) is disclosed as a likely memory-low flag, never confirmed on the NRP Violations page (`README.md:125`). RAM and CPU use for the three CPU diagnostics Jobs is not recorded.

## 2026-10-21-target-diagnostics

**What it is:** the target-side test (H-E). For base D and partner B, how much does adding B's future features as a second prediction target hurt a rank-64 linear prediction of D's future features?

**1. Ran as planned?** Pre-registration is LOG.md 2026-10-21 (i) (lines 240-267) and its amendment (j) (lines 269-291), committed in `c352d42` and `01c2cee`, both before the code (`2e03e18`) and the run (verified, blame). The experiment `PLAN.md` is a real handoff written before running (not reconstructed).

| Item | Spec | What happened | Disclosed? |
|---|---|---|---|
| Primary scaling | (a) raw pooled, amended to (a') `w = 1/d_block` before any code | (a') used, plus b and (a'') | yes, LOG (j) |
| Horizons / ranks | k in {4, 16, 32}, r = 64 primary (16, 256 secondary) | as specified; all JSONs `config.ks`, `config.ranks` (verified) | |
| Feature type | global primary, mean secondary | as specified | |
| Test 4 data | thresholds fixed first | data changed after check 4 failed on the first run, thresholds unchanged | yes, LOG (k) and `test_target_diagnostics.py:83-90` |
| Run | one CPU Job, ~3 h estimate | try 1 OOM-killed at ~22 min (Mac RSS undercounted Linux), try 2 at `52d1226` with staged features, 5 h 45 min wall, deadline raised live to 16 h | yes, `README.md:25`, LOG 2026-10-27 (f), (h), (i) |
| Code identity | `b68ba54` pre-registered | `52d1226` ran; outputs bit-identical on synthetic data | yes, `README.md:19`, LOG (h) |
| Housekeeping | | `job.yaml:12` header still says "NOT YET SUBMITTED" | yes, `README.md:69` |

- Test demos per k are asserted identical (`target_diagnostics.py:250-251`); JSON frame counts per k (37,264 / 12,016 / 12,448 at k=4, etc.) match `shift_rows` (verified).
- Alpha at the grid edge: only `global|k=4|t1b_D` (alpha 0.01) in each pair (`alpha_edge_hits` in every JSON, verified; README says the same, `README.md:51`).

**2. Numbers vs result files** (verified against all four `results/target-libero_goal-dinov3-*.json`). I recomputed the primary table and the secondary table:
- Primary a', r = 64, global, mean over k: G +0.0343 [+0.0332, +0.0354]; R +0.0151 [+0.0147, +0.0155]; L +0.0141 [+0.0138, +0.0145]; S +0.0032 [+0.0030, +0.0035]. Under b: G 0.0228, R 0.0251, L 0.0288, S 0.0207. Under a'': G 0.0462, R 0.0283, L 0.0141, S 0.0032. B loss share (a'): G 0.717, R 0.352, L 0.288, S 0.116. Reference R2_D 0.3642 (G, L), 0.3723 (R), 0.3698 (S). All match `README.md:33-38`.
- Secondary: r = 16 / 256 values, per-k values, mean-type values, T1a and T1b all match `README.md:46-49`.
- Key findings (`README.md:19`): G +0.034 [+0.033, +0.035], R +0.015, L +0.014, S +0.003, "R's interference grows under equalised losses" (0.0151 to 0.0251). All correct.
- The outcome rule (R not the largest on T2 primary, so H-E fails at this level) is applied correctly: G is the largest (verified). Peak RSS 12.85 GiB and 5 h 45 min are from the LOG and Prometheus, not rechecked by me.
- `README.md:55` says "the ordering doesn't track the paper's deltas". My Spearman for the a' primary ordering (S best, then L, R, G) against the paper order is +0.6 (n = 4), and 0 for the mean-type ordering. Neither is meaningful at n = 4; the sentence is a judgement, not a measurement, and should say so.
- `README.md:57` states "dimension is a confound: the two widest partners (G, R) are first and second under a'". Correct, and an even better predictor is the loss share (next section).

**3. Method:** see "Code review" for the long form. The two things that limit the claim:
- Under a' (`w = 1/d_block`, raw feature scale), interference is ordered exactly like B's loss share. G, R, L, S have shares 0.717, 0.352, 0.288, 0.116 and interference 0.0343, 0.0151, 0.0141, 0.0032 (verified). Under b, where every block is rescaled to equal total variance, all four partners are within +0.0207 to +0.0288. So the primary result is largely "which partner has the biggest raw feature changes", which is a property of each encoder's output scale, and the paper does not say how (or whether) it normalises (LOG (j)).
- The planted-interference test (`test_target_diagnostics.py:83-116`) is built exactly this way (B at x10 scale), so it confirms the code detects scale dominance, but the test cannot tell scale dominance from real target conflict.

**4. Tests:** `python src/test_target_diagnostics.py` prints seven passing checks and `all ok` (verified, below).

**5. Standards:** `job.yaml`: `backoffLimit: 1`, `activeDeadlineSeconds: 57600` (patched live from 28,800, logged, `job.yaml:21`), no sleep, 2 CPU / 2400m, 14 / 16 Gi, ephemeral 22 / 26 Gi, us-west affinity, read-only PVC mounts (`job.yaml:20-22, 81-87`) (verified). RAM: peak 12.85 GiB of 14 Gi and a minimum of 3.7 GiB (26%) over the first 20 minutes, CPU 1.67-1.9 of 2 cores (LOG 2026-10-27 (i), (j), from Prometheus; not rechecked). Try 1's idle 737 s load phase was the reason for staging the features to local disk (LOG (f)). The Violations page was never checked (needs the user's login; LOG (j) "Not done"). Observations in `README.md:53-57` carry no verified/speculative labels (the Results header says verified, the "Why, possibly" paragraph says speculative).

## Code review (diagnostics.py, target_diagnostics.py, extract_features.py)

The idea in one paragraph: the code asks how much a second encoder B adds, in a straight-line fit (ridge regression) from frozen features to the recorded teleoperator action, on top of encoder A, and compares that with how similar the two feature clouds are (CKA). The target-side code asks a different question: if a straight-line model with only 64 "directions" must predict both D's and B's future features, how much does D's own prediction get worse when B's future is also in the target.

### CKA (`diagnostics.py:130-151`)
- Definition: with linear kernels `K = XX'` and `L = YY'` on column-centred features, `CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) HSIC(L, L))`. The biased version uses double-centred Gram matrices (`:146-149`); the debiased version uses the unbiased HSIC of Song et al. with zeroed diagonals (`:136-141`, `:150`) (verified by reading; the formula matches the standard unbiased estimator, the match to the literature is recalled).
- Tests: CKA(X, X) = 1, invariance to rotation and scale, independent 300-d features give biased 0.167 and debiased -0.0001, and the in-run sanity (self = 1, shuffled about 0) (verified).
- Computed on 5,000 frames drawn at random from all splits, same frames for every pair (`:270-271`). This is unsupervised, so there is no leakage, but the frames are correlated within demos (effective sample size is closer to the number of demos than the number of frames, speculative), the debiasing assumes independent rows, and there is no CI (the tally README admits it, `README.md:79`).
- Features are only centred, not standardised, so the high-variance channels dominate the kernel. CKA is per camera and per feature type, while the probes use both cameras concatenated, so the two measures are not computed on the same input.

### Probes and gain (`diagnostics.py:158-199, 288-304`)
- Ridge: `W = (X'X + alpha*n*I)^-1 X'(y - mean(y))`, intercept `mean(y)` (`:169-173`). X is standardised with **train-frame** mean and sd only (`:158-160`, called with `train` at `:290, 293`), so val and test statistics never enter. Alpha is chosen from `ALPHAS` by mean validation R2 (`:176-190`); the final fit is on train+val at that alpha and is scored on test (`:187-190`). No test leakage (verified by reading).
- Split by demo within each task file (`demo_split`, `:57-64`): last 20% test, previous 20% val, rest train; frame counts match the JSONs. No frame-level split anywhere. The assertion at `:93` catches feature/action length mismatches.
- `R2` is the mean of per-dimension R2 over the 7 action dimensions (`:163-166`). Per-dimension values (C-RADIOv4 global): 0.70, 0.72, 0.71, 0.60, 0.40, 0.72, 0.83 (verified from the JSON), so rotation dimension 5 pulls the mean down.
- `gain(B|A) = R2(A+B) - R2(A)` (`:301-304`). Alpha is tuned separately for A, B and A+B, so the concatenated probe gets its own best alpha (as pre-registered). The probes are deterministic, so seeds only matter for the CKA subsample and the bootstrap (seed 0, recorded).
- Why gain cannot do what the tally asks: with the alpha tuned on validation, R2(A+B) is rarely below R2(A) because the penalised fit can always shrink B's weights. A negative gain would need overfitting that the validation step is built to avoid. The pre-registration says the same (LOG (i), "speculative: an input-side ridge probe ... can almost never score below D alone"). All four measured gains are positive (+0.014 to +0.037). So the tally could rank partners by how much they help the probe, but could not have shown "B hurts". It also ranks nearly like B's own probe score, because A+B contains B (verified: R2(B) order R, S, G, L vs gain(B|D) order R, S, L, G).
- Capacity: A+B has more inputs than A alone, and the dimensions differ by partner (D+R 6,656-d, D+L 4,096-d). Pre-registered and disclosed. There is no control partner (a pure-noise B of the same dimension, or a duplicate of D) to say how large a gain is expected by capacity alone (speculative: probably near zero with tuned alpha, but never measured).
- Alpha hit the lower grid edge (0.01) for several probes (tally README `:23`; D's val R2 is 0.648 at 0.01 and 0.654 at 0.1, so D itself is fine). Edge cases are recorded, not acted on, as the PLAN says.
- Numerics: `x.T @ x` and `solve` run on float32 arrays in `ridge_fit` (inputs are float32 from `load`, `:101`), with `alpha*n` of at least 384 added to the diagonal. I do not expect a problem at that regularisation, and the three tally Jobs reproduce D's CI to ~1e-8, but the target-side code casts to float64 on purpose (`target_diagnostics.py:21`). Low risk.

### Bootstrap (`diagnostics.py:204-239`)
- Per-demo sufficient statistics (frame count, sum y, sum y^2, sum of squared errors) and multinomial demo weights give the resampled R2 exactly (`r2_from_stats`); a unit test checks it against an explicit resample (`test_diagnostics.py:39-49`). The unit of resampling is the demo, and the same draws are used for all probes, so gains are paired. This is the right unit (frames within a demo are correlated).
- Limits (verified from the code): the fitted models are held fixed, so the CI covers which test demos were drawn (100 demos) and nothing else: not the choice of train/val/test demos, not the fitting. Tasks are not stratified in the resample. The CIs are therefore narrower than a split-resampling or cross-validated interval would be (speculative on size). Key findings quotes the CIs without this qualifier.
- Draw count 1,000 (>= STANDARDS minimum) (verified in every JSON).

### Target side (`target_diagnostics.py`)
- Targets: `Delta_E(t;k) = f_E(t+k) - f_E(t)`, built only inside a demo (`shift_rows`, `:53-60`; test 1 checks no pair crosses a demo boundary and the counts). Features X are standardised on train frames (`:236-237`), targets stay at raw scale. T0 variances are on train rows only (`:257-258`). No leakage found.
- Fast ridge: one eigendecomposition per fit set (`Fit`, `:84-114`), checked equal to `diagnostics.ridge_select` to 5e-14 (test 6).
- Reduced-rank step (`rrr_basis`, `rrr_select`, `:138-167`): fit ridge on the scaled target, take the top-r right singular vectors `V_r` of the fitted values, predict with `W V_r V_r'`. Block scaling multiplies block k by `sqrt(w_k)` (`:184-186`). Interference is `R2_D(target = Delta_D only) - R2_D(target = [Delta_D, Delta_B])` with the same input X = [D, B] in both (`:301-308, 327-330`), which is a good design: only the target changes.
- Things that weaken the reading:
  1. **Scale confound, shown above.** Under a' the weight is `1/d_block`, which equalises the *number of channels*, not the *variance*, so a block with larger raw Delta variance takes more of the rank budget. G's per-channel Delta variance is ~2.6x D's (0.168 vs 0.064 at k=16, global; verified from the JSON), and its loss share 0.717; S's is 0.14x (0.009 vs 0.064) with share 0.116.
  2. **Alpha is chosen per condition on that condition's own weighted loss** (`rrr_select` `:152-160`; PLAN `:44`). The with-B condition's alpha is dominated by B when B carries most of the loss, so part of the gap can be a different alpha, not capacity. In the JSONs the alphas are equal for k = 4 and 16 in all pairs, but at k = 32 the G pair chose 1 with B and 0.1 for the reference (verified). Small for the primary number.
  3. **R2 is the unweighted mean over D's 2,048 dimensions** (`D.r2`), whereas the WAM loss is a (weighted) sum of squared errors. Low-variance noisy dimensions count as much as large ones. Reference R2 is only ~0.37, so every figure is a small difference on a noisy base.
  4. **Rank 64 is an arbitrary capacity.** Secondary ranks show the ordering is stable in one respect (G largest at r = 16, 64 and 256) but the other three are close: r = 256 gives G 0.0187, L 0.0175, R 0.0161, S 0.0037. R vs L at r = 64 is "narrowly above" only on a test-demo CI.
  5. **No control partner** (for example D's own future as B, or a random projection of D) to show what interference a harmless second target of this size produces. Under b all four partners cost 0.021-0.029, which is a hint that "any equally weighted second target costs about 0.02".
- `alpha_edges` records every grid-edge choice (`:361-377`); the Job found one per pair. Fine.
- What the experiment can and cannot say: with the pre-registered rule it correctly shows R is not the largest at this level. It cannot say target-side competition is absent in a nonlinear flow-matching model with its own (unknown) normalisation. The README's own caveats say so (`README.md:60-62`).

### Extraction (`extract_features.py`)
- Right: rotate-180 flag, patch tokens skip CLS and register tokens for DINOv3 (`:80`), square-grid assertion (`:60-65`), finiteness check, atomic file writes via `.partial` and `os.replace` (`:231, 259`), resume by skipping existing files (`:303-305`), `SRC_GIT_COMMIT` and a per-file metadata blob with encoder id, resolution, dtype and commit.
- Issues are in the Extraction section above (upsampling from 128 px, finiteness check before the fp16 cast, stored-vs-compute dtype, default variant differs from the one used, `C-RADIOv4` remote code pinned at `c0457f5` (good) but the dataset revision `main` is not).
- DINOv3 `global` is the CLS token; SAM 3 `global` is the patch mean, so SAM 3's global and mean probes are identical by construction (`:150-151`, tally README `:70`); VGGT-Omega `global` is the camera token. The four "global" features are therefore different kinds of summaries, which is a limit on reading the global-feature tally as a like-for-like comparison.

### Tests (item 4 of the task)
- `python -m pytest src/test_diagnostics.py src/test_target_diagnostics.py -q` from the direction folder: **there is no `python` on this machine** (`zsh: command not found: python`). `python3` is 3.14.3 with no numpy and no pytest (`ModuleNotFoundError: No module named 'numpy'`, `No module named pytest`). The Homebrew 3.11 interpreter (`/opt/homebrew/opt/python@3.11/bin/python3.11`, numpy 1.26.4, h5py present, pytest present) runs: `pytest` prints **`no tests ran in 0.11s`** and exits with status 5 (verified), because both files define only `main()` (`test_diagnostics.py:8`, `test_target_diagnostics.py:34`) and no `test_*` function. Both files document `python test_x.py` as the way to run them (docstring line 1).
- Run as scripts from `src/` with that interpreter: `test_diagnostics.py` ends `all ok` (CKA independent biased 0.167 / debiased -0.0001; redundant B gain +0.000; informative B gain 0.529; bootstrap-from-stats equals explicit resample; gain CI informative +0.175..+0.247, redundant -0.001..-0.001). `test_target_diagnostics.py` ends `all ok` (seven checks; planted interference r=32: a' +0.9882, b +0.0126, a'' +0.9882; r=64 all about 0; identical predictors give CI [0, 0]; fast ridge equals `ridge_select` to 5.5e-14; end-to-end `run_type` finite, random-walk null interference +0.0003). Tests ran under numpy 1.26.4, not the pinned 2.2.6 of the Jobs. Verified.
- Do the tests exercise the claims?
  - Yes: CKA properties, gain on a redundant vs an informative B, per-demo bootstrap arithmetic, bootstrap CIs on demo-correlated data, shift correctness, fast ridge equals reference, full-rank RRR equals ridge, single-block scale invariance, bootstrap pairing, and a null (random walk) interference near zero.
  - No: `demo_split` and `load` (the code that creates the train/val/test boundary and aligns features to actions) have no test, so "split by demo, no leakage" rests on reading the code and on the printed frame counts. There is no test that train-only standardisation is used (a leakage regression would not be caught). There is no test that gain stays near zero for a pure-noise B at the real dimensions, and no coverage check of the bootstrap CI (the demo-correlated test only checks that an informative B's CI excludes 0 and a redundant B's covers about 0).
  - The planted-interference test uses a B at x10 scale, which exercises the scale-dominance case only, and the case that matters for the reading (equal-scale B whose future conflicts with D's) is not constructed.

## Standards compliance

- **Claim labels (05):** Key findings uses `verified` and `speculative` on each bullet (`README.md:15-20, 28`). Experiment READMEs label their numbers at section level (tally "all numbers verified from ...", extract/diag/bench per-bullet). The target README's Observations (`README.md:53-57`) and several qualitative sentences in the tally Observations (`README.md:73-96`) carry no label (the tally ones are arithmetic on the tables above them). Low.
- **LOG append-only (04):** the entries on these experiments, LOG.md:190-318, each come from one commit with no later edit (verified, `git blame`). The direction LOG as a whole has about 20 commits with deletions, mostly same-day corrections of the newest entries (for example `8107099` rewrote lines in the 2026-11-06 docs entry instead of adding a correction below it). None touched these experiments' entries. Low, and a pattern to keep an eye on. The experiment `LOG.md` files of all five are 2026-11-04 index stubs pointing into the direction LOG (`LOG.md:5-10` in each); 04 wants a step-by-step record there. The stub says why. Low.
- **PLAN before running (04/01):** target-diagnostics PLAN and the tally pre-registration (LOG (e), (i), (j)) were written before the runs. The other four PLAN.md files are labelled reconstructed (`PLAN.md:3`), and the README predictions they come from were committed with or before the Jobs (verified for extract `89fb593`, diagnostics `346dc48`, bench `1eaecbf`).
- **Commit hashes and provenance (03):** every Job yaml carries `SRC_GIT_COMMIT`; all result JSONs carry `git_commit` (verified); all hashes cited in the READMEs (`f4a7b82`, `d6e3905`, `a89ece6`, `7dc80ca`, `0cd09b5`, `9e3bdb2`, `52d1226`, `b68ba54`, `567413d`, `ff1b457`, `1eaecbf`) exist in the repo (verified, `git cat-file -t`). Model revisions: DINOv3 `ea8dc28`, SAM 3 `3c879f3`, VGGT-Omega `1041e80`, RADIO remote code `c0457f5`. The SigLIP 2 revision is not recorded. The LIBERO HDF5 dataset revision is `main` (not pinned, `job.yaml:55`, `extract-job.yaml:67`, `diag-*-job.yaml`); `data/README.md:13` records that fact but there is no hash or file checksum. The extraction log does list the ten file sizes (`results/job.log:158-167`), which is a weak fingerprint.
- **Seeds (07):** CKA subsample and bootstrap use seed 0 (recorded in every JSON); ridge is deterministic. Single data split; no cross-split replication. STANDARDS asks for multi-seed only for the proxy ground truth, so not a violation.
- **Config files (03):** none of the five has a `config.*`; the parameters are on the command lines in the Job yamls. 03 says "every parameter" goes in a config. Low.
- **Environment (03/07):** pip pins are in each Job yaml and `pip freeze` is in the logs (verified for extract and fastwam; tally and target Jobs pin numpy 2.2.6, h5py 3.16.0, huggingface-hub 0.36.2). Local tests here ran on numpy 1.26.4.
- **Baselines (STANDARDS "Methods"):** the standard baselines for a predictor comparison are B's single-encoder score, symmetric CKA, objective-overlap reasoning coded as a label before results, and random ranking. The tally compares CKA and gain against the paper's order informally; B's solo score is in the table but not compared; the overlap label and the random ranking are not computed. With n = 4 the README refuses a rank statistic (`README.md:102`), which is what STANDARDS asks. For the later proxy comparison these baselines must exist before the data is seen.
- **Nautilus 08 and STANDARDS (resources, utilisation, backoffLimit, deadlines):**

| Job | backoffLimit | activeDeadlineSeconds | limits within 20% | sleep | utilisation evidence |
|---|---|---|---|---|---|
| fastwam bench (`job-a6000-lowcpu.yaml`) | 1 | 10,800 | yes | no | none recorded; 24 min 49 s run is under the 30-minute no-check window |
| extract pilot (`job.yaml`) | 1 | 14,400 | yes | no | GPU 90-99% recorded; CPU/RAM not |
| diagnostics pilot | 1 | 10,800 | yes | no | not recorded |
| tally extraction (`extract-job.yaml`) | 1 | 46,800 | yes | no | try 3: GPU 99-100%, RAM 85-89%; try 2 over-requested RAM (disclosed) |
| tally diagnostics x3 | 1 | 21,600 / 10,800 / 10,800 | yes | no | not recorded |
| target diagnostics (`job.yaml`) | 1 | 57,600 (patched live) | yes | no | RAM min 26%, CPU 84-95% of request |

  The pre-flight step "check the Violations page 30-40 minutes after launch" (08, checklist item 8) was not done for any of these (needs the user's login); the tally and target LOGs say so. All Jobs and ConfigMaps were deleted afterwards (tally LOG: verified 2026-10-21 08:22Z; target LOG (k)).

## Does this line look promising?

*All of this section is speculative (my judgement), built on the verified facts above.*

**Plain-words answer.** The question is still alive, but nothing so far supports it, and nothing so far can sink it either. Think of it as having tried two thermometers on a patient and neither showed a fever: one thermometer could not read above normal, and the other measured the room temperature.

**Why the negative results are weak evidence against the idea.**
- The input-side measure (does B add action information on top of D) can hardly come out negative, so it can say "B helps a bit" but not "B hurts". It also tracks how good B is on its own. Where the ordering looked right (gain(D|B)), that is just B's solo score reversed.
- The target-side measure mostly follows each encoder's raw feature scale, and the paper does not say how it scales features. A different reasonable scaling gives a flat picture (all four partners about the same).
- Both measures use clean frames and a straight-line model on single frames; the paper's harm appears under perturbed LIBERO-Plus scenes and in a trained model. The paper reports only a LIBERO-Plus total for combinations, from one trained checkpoint each, so the only "answer key" is itself a single noisy draw (D+R's drop is 5.4 points, its evaluation-seed sd is 1.0; training-seed variation is unmeasured).
- n = 4 pairs, no baselines computed, no control partner.

**What would be needed to answer the main question** (a cheap, conditional, frozen-feature measure that predicts which combinations help a WAM):
1. **A ground truth we trust.** Training-seed noise for one config (the two DINOv3 seed-noise models and the replication in `experiments/2026-10-27-proxy-calibration/` are exactly this). If D+R vs D does not show a drop in the proxy, there is nothing to predict at our scale and the question changes; if it does, we need many more combinations (all 16 in Table 10 and more) with at least 3 seeds each, because a rank correlation over 4 to 7 pairs cannot separate a good measure from luck.
2. **Measures defined where the effect lives.** The perturbed-frames experiment is the right next probe (the same measures, computed on perturbed re-renders). A measure that never sees perturbed frames cannot predict a perturbation-driven drop.
3. **Measures that can go the right way.** Something that can be negative (for example a fusion-aware or training-dynamics-aware score), a scale-normalised version of the target-side measure, and the fusion-step tests already in PLAN Next 2 (scale, double counting, grid mismatch), because the harm might come from how channels are combined and not from which encoders they are.
4. **Controls and baselines fixed before the data is seen.** B's solo score, symmetric CKA, the objective-overlap label, random ranking, a noise or duplicate partner, and dimension-matched features; report rank correlation with a CI over combinations.
5. **A cheap way to estimate gains per combination** is the point of the whole line, so any candidate also needs a cost line (CPU-minutes per pair) next to its accuracy. The input-side tally is cheap (30-50 minutes on 2 cores per pair); the target-side one was not (about 11.5 CPU-hours for 4 pairs).

**One sentence for the lead.** The finished experiments are sound as computed and honestly documented, they test the idea only weakly, and the fastest route to an answer is the proxy replication plus a perturbed-frame measure with proper baselines, not more clean-frame linear probes.

## Fix list

Severity: high = a reported finding could be wrong; medium = a reported finding is weaker or narrower than stated; low = clarity, reproducibility or housekeeping. I found no high-severity item.

1. `world-model-stacking-encoders/README.md:15-18` (Key findings, gain bullet). Wording: "R adds the most action-relevant information ... +0.037 ... vs S +0.033" sits inside overlapping CIs (R [0.0325, 0.0423], S [0.0275, 0.0379]; `experiments/2026-10-19-feasibility-tally/README.md:86` says the CIs overlap). Also add that gain(B|D) orders the partners like B's own probe score. Breaks: `01` ("report uncertainty"), STANDARDS "Methods". Fix: reword to "R is at or near the top; CI overlaps S" and name the solo-score link. Severity: medium.
2. `world-model-stacking-encoders/README.md:15-17` and `experiments/2026-10-19-feasibility-tally/README.md:103-107`. The measure cannot go clearly negative (pre-registration says so, LOG.md (i)), so "fails the cheap check" is close to by-construction for harm. Add this limit to Key findings and to the tally Caveats. Breaks: `06` (report what happened, including what the test could detect). Severity: medium.
3. `world-model-stacking-encoders/README.md:19` and `experiments/2026-10-21-target-diagnostics/README.md:55-57`. Add: under a' interference follows B's loss share (G 0.717, R 0.352, L 0.288, S 0.116 gives interference 0.0343, 0.0151, 0.0141, 0.0032) and under b all four lie between 0.0207 and 0.0288; so the result is largely a feature-scale effect. Replace "doesn't track the paper's deltas" with the measured fact (n = 4, rank correlation +0.6 for a' order, 0 for mean-type order, no significance). Breaks: `01` (separate observation from interpretation), `05` (label claims). Severity: medium.
4. `experiments/2026-10-19-feasibility-tally/README.md:98-107` (Caveats) and `README.md` Key findings. Add that CIs are over test demos only (single split, fitted models fixed), and that CKA has no CI (the quoted 0.576 vs 0.583 cell is a tie). The caveat exists in LOG 2026-10-19 (c) and `target-diagnostics/README.md:56`, not here. Breaks: `01` ("report uncertainty"), STANDARDS "Required statistical tests". Severity: medium.
5. `world-model-stacking-encoders/src/test_diagnostics.py:8` and `src/test_target_diagnostics.py:34`. `pytest` collects zero tests, so the standard command runs nothing and exits with status 5 (no tests collected; verified), which a CI step or a quick glance can mistake for a pass. Fix: add thin `test_*` wrappers (for example `def test_all(): main()`), or state "run as a script from `src/`" in the README Key files; also add tests for `demo_split` and `load` (split boundaries, no frame in two splits), for train-only standardisation, and for a pure-noise B at the real dimensions. Breaks: `07` (sanity checks), `03` (rerunnable). Severity: medium.
6. STANDARDS "Methods and evaluation" (baselines) vs `experiments/2026-10-19-feasibility-tally/README.md`. B's solo score, the objective-overlap label and a random ranking are not computed; no control partner (noise, duplicate, dimension-matched). Fix: add them when the next comparison is run, and add a one-line note in the tally README that they were not run. Breaks: STANDARDS "Standard baselines", `07` ("always compare against a baseline"). Severity: medium (low for the finished runs, because the tally is labelled a sanity check; medium for the proxy round).
7. `world-model-stacking-encoders/README.md:18-20` and tally README `:89-91`. gain(D|B) is shown in the tally README but not in Key findings; it matches the paper's order (rho +0.8, n = 4) only because it equals R2(D+B) minus R2(B). Fix: state both directions, and that gain(D|B) is a restatement of B's solo score, before anyone quotes it. The tally README `:89` also calls gain(D|B) "not asked for" while LOG.md:197 pre-registered it. Severity: low.
8. `src/target_diagnostics.py:152-165` (alpha per condition on that condition's weighted loss; G at k = 32 chose 1 vs 0.1 for the reference). Note in the README, or rerun the G pair with the reference alpha forced; check the primary number moves little. Breaks: none; reduces a possible confound. Severity: low.
9. `src/extract_features.py:246-249`. Finiteness check runs before the fp16 cast on `:249`; check after the cast (or clip). `:94-96` default SigLIP 2 variant differs from the one used (`experiments/2026-10-18-extract-libero-goal/job.yaml:60`); change the default or record the used variant in the README. `:256` records compute `dtype`, not the stored fp16; rename or add a `stored_dtype` field. Breaks: `03` (provenance). Severity: low.
10. `experiments/2026-10-19-feasibility-tally/README.md` Caveats and `experiments/2026-10-21-target-diagnostics/README.md` Caveats. Add: source frames are 128x128 and are upsampled 1.75x to 7.9x to the encoders' input sizes (`data/README.md:16`, `extract_features.py:50-57`). Severity: low.
11. Job yamls: `experiments/2026-10-18-extract-libero-goal/job.yaml:55`, `experiments/2026-10-19-feasibility-tally/extract-job.yaml:67`, `diag-*-job.yaml` (`revision="main"`). The LIBERO HDF5 dataset version is not pinned and no checksum is recorded. Fix: record the HF revision hash and sha256 of the ten files next time they are downloaded. Breaks: `03` (data provenance), `07` (record dataset hash). Severity: low (disclosed in `data/README.md:13`).
12. 08 pre-flight and Requests section vs the extract pilot, pilot diagnostics and three tally diagnostics Jobs. CPU and RAM use against the 20-200% / 20-150% ranges is not recorded, and the Violations page was never checked after launch. Fix: record peak RSS and mean CPU from Prometheus for any new Job, and note "Violations page not checked (user login)" in the Job's LOG entry. Severity: low (no evidence of a violation; lab already at 1 of 3 strikes, so worth recording).
13. `world-model-stacking-encoders/README.md:28`. "30-50 CPU-min per pair" is 30-49 minutes of wall time on 2 cores (60-100 CPU-minutes). Fix the unit. Severity: low.
14. `experiments/2026-10-21-target-diagnostics/README.md:53-57`, tally Observations `:73-96`. Add verified/speculative labels (for example the "dimension is a confound" line is partly verified now). Breaks: `05`. Severity: low.
15. `experiments/2026-10-18-extract-libero-goal/README.md:36-39`. The README shows the C-RADIOv4 throughput (62 frames/s) and storage (9.6 GB) but does not flag them as outside the 100-300 frames/s and ~7 GB predictions. Add one line. Severity: low.
16. All five `experiments/*/LOG.md`: index stubs pointing into the direction LOG. 04 wants the step-by-step record in the experiment LOG. Fine for finished work if the stub says why (it does); for future experiments write the detail in the experiment LOG. Severity: low.
17. Five experiment folders have no `config.*` (03). Parameters live in Job yaml command lines. Severity: low; add a one-line note in each README saying the Job yaml is the config.
18. `experiments/2026-10-19-diagnostics-libero-goal/results/run1-*.json`, `run2-*.json`: produced by a script version with `resid` that no longer exists in `src/diagnostics.py` (README `:88` discloses). Fix: tag the commits `d6e3905` and `a89ece6` or keep the run script alongside the results. Severity: low.
