# 2026-10-19 — Feasibility tally on libero_goal (decision 002; no training)

**Purpose:** cheap check from decision 002 (accepted by user 2026-10-19). Frozen-feature diagnostics for DINOv3 (D) paired with each partner B, set beside the two-encoder deltas in liu2026what Table 10. **No hypothesis is chosen or argued here** (user instruction). With 4 pairs this is a sanity check, not evidence. Choices were fixed before running in LOG `2026-10-19 (e)`. Cluster events are in LOG `(f)` and `(g)`.

## Config
- **Data:** LIBERO libero_goal, all 10 task files, 63,728 timesteps, 2 cameras (agentview, wrist), frames rotated 180°. Same frames and same demo-level split as the pilot (`../2026-10-19-diagnostics-libero-goal/`): per task file, last 20% of demos test, previous 20% val, rest train. 38,464 / 12,416 / 12,848 frames, 100 test demos (verified, every result JSON).
- **Encoders** (variants are our choices; not confirmed to match the paper):

  | key | encoder | variant | input | feature dims per cam (global) | dtype | source |
  |---|---|---|---|---|---|---|
  | D | DINOv3 | `facebook/dinov3-vitl16-pretrain-lvd1689m` @ `ea8dc28` | 224 | 1024 | fp32 | this run |
  | L | SigLIP 2-L | as in `../2026-10-18-extract-libero-goal/` | | 1024 | fp16 | pilot cache |
  | R | C-RADIOv4 | as in `../2026-10-18-extract-libero-goal/` | | 2304 | fp16 (autocast) | pilot cache |
  | S | SAM 3 | `facebook/sam3` @ `3c879f3`, ViT backbone before the FPN neck | 1008 | 1024 (patch mean; S has no summary token, so global = mean) | fp16 | this run |
  | G | VGGT-Ω | `facebook/VGGT-Omega` `vggt_omega_1b_512.pt` @ `1041e80`, final aggregator layer | 512 | 2048 (camera token) | fp16 autocast | this run |

  Concatenated global probe input: D+L 4,096-d, D+S 4,096-d, D+G 6,144-d, D+R 6,656-d (verified, from the per-cam dims above). **These differ, and no matched-dimension control was run** (pre-registered).
- **Diagnostics:** `../../src/diagnostics.py` at `7dc80ca`, unchanged. Per pair:
  - Debiased linear CKA per camera on 5,000 frames (seed 0) for global, mean and 4×4-grid features.
  - Ridge probes to the 7-d action, both cams concatenated: D, B, D+B, plus proprio. Test R² averaged over dims.
  - gain(B|D) = R²(D+B) − R²(D), and gain(D|B) = R²(D+B) − R²(B).
  - 95% CIs from a demo-level bootstrap over test demos: 1,000 paired draws, seed 0.
- **Ridge α:** tuned on validation **separately for every probe and every pair** (grid 1e-2…1e5), then refit on train+val. The α picked for each probe is listed in the tables. α hit the bottom edge of the grid (0.01) for L global, R mean, and G global and mean.
- **Feature type:** global is primary, mean is secondary (pre-registered).
- **Code and manifests:**
  - Extraction: `src/extract_features.py` at `0cd09b5` (try 2: all of D, first 5 G files) and `9e3bdb2` (try 3: last 5 G files, all S). The VGGT-Ω loader change between these gave identical features in a local check (max |diff| 0.0, verified).
  - Manifests: `extract-job.yaml`, `diag-dlr-job.yaml`, `diag-dg-job.yaml`, `diag-ds-job.yaml`.
- **Hardware:**
  - Extraction: 1× RTX 2080 Ti, us-west, 1 CPU / 4Gi (try 3).
  - Diagnostics: CPU-only nodes, 2 CPU / 11Gi.
  - Features live on the CephFS PVC `stacking-encoders-features` (D 4.4G, S 4.4G, G 8.8G). L and R are read from `michael-persistent`.

## Tally (all numbers verified from `results/tally-libero_goal-dinov3-*.json`; paper Δ = Table 10 mean − D alone 84.0 from Table 1, verified 2026-10-21 against `../../literature/liu2026what-table{1,10}-*.csv`, which were checked against the arXiv HTML 2026-10-18)

**Global (primary)**

| B | CKA(D,B) agentview / wrist | gain(B\|D) [95% CI] | gain(D\|B) [95% CI] | R² D | R² B | R² D+B | α D / B / D+B | paper Δ vs D (84.0) |
|---|---|---|---|---|---|---|---|---|
| L SigLIP 2-L | 0.576 / 0.792 | +0.017 [+0.013, +0.021] | +0.057 [+0.052, +0.064] | 0.650 | 0.609 | 0.667 | 0.1 / 0.01 / 0.1 | +1.83 |
| R C-RADIOv4 | 0.583 / 0.854 | +0.037 [+0.032, +0.042] | +0.017 [+0.014, +0.021] | 0.650 | 0.670 | 0.687 | 0.1 / 0.1 / 0.1 | −5.41 |
| S SAM 3 | 0.559 / 0.643 | +0.033 [+0.027, +0.038] | +0.029 [+0.024, +0.034] | 0.650 | 0.654 | 0.682 | 0.1 / 0.1 / 0.1 | +0.49 |
| G VGGT-Ω | 0.213 / 0.267 | +0.014 [+0.010, +0.018] | +0.047 [+0.041, +0.054] | 0.650 | 0.617 | 0.664 | 0.1 / 0.01 / 0.1 | +0.40 |
| V V-JEPA 2.1 | not run | | | | | | | +0.93 |
| U UniTok | not run | | | | | | | +2.12 |
| H HRP | not run | | | | | | | +0.48 |

**Mean (secondary)**

| B | CKA(D,B) agentview / wrist | gain(B\|D) [95% CI] | gain(D\|B) [95% CI] | R² D | R² B | R² D+B | α D / B / D+B | paper Δ |
|---|---|---|---|---|---|---|---|---|
| L | 0.672 / 0.782 | +0.034 [+0.030, +0.038] | +0.031 [+0.027, +0.036] | 0.642 | 0.645 | 0.676 | 0.1 / 0.1 / 0.1 | +1.83 |
| R | 0.766 / 0.863 | +0.037 [+0.033, +0.042] | +0.011 [+0.006, +0.017] | 0.642 | 0.668 | 0.679 | 0.1 / 0.01 / 0.1 | −5.41 |
| S | 0.665 / 0.665 | +0.035 [+0.031, +0.040] | +0.024 [+0.019, +0.029] | 0.642 | 0.654 | 0.678 | 0.1 / 0.1 / 0.1 | +0.49 |
| G | 0.268 / 0.399 | +0.016 [+0.013, +0.019] | +0.031 [+0.025, +0.037] | 0.642 | 0.627 | 0.658 | 0.1 / 0.01 / 0.1 | +0.40 |

**Grid CKA (4×4, agentview / wrist):** L 0.830 / 0.868, R 0.929 / 0.898, S 0.896 / 0.802, G 0.452 / 0.484.
**Single-probe 95% CIs (global):**

| Probe | 95% CI |
|---|---|
| D | [0.623, 0.671] |
| L | [0.581, 0.630] |
| R | [0.645, 0.690] |
| S | [0.628, 0.674] |
| G | [0.589, 0.638] |

**Sanity checks** (verified):
- Proprio probe R² is 0.279 in every Job, matching the pilot.
- The L and R single-probe R² match the pilot (0.609, 0.670).
- S global and mean B-probes are identical (0.654, same CI), as expected, because S's global is the patch mean.
- D's probe CIs agree across the four Jobs to about 1e-8.

## Observations (plain; 4 pairs, no significance claims; no hypothesis argued)
The paper orders these 4 pairs L (+1.83) > S (+0.49) > G (+0.40) > R (−5.41).

**CKA:**
- R has the highest CKA with D in all 6 cells (global, mean and grid × 2 cams) (verified).
- R's margin over the next pair ranges from 0.007 (global agentview, over L) to 0.094 (mean agentview).
- No CIs were computed for CKA, so the 0.007 margin is not distinguishable from a tie here.
- L, the paper's top pair, is second or third, close to R.
- G has by far the lowest CKA and a small positive paper Δ.
- So CKA does not follow the paper order beyond R being first.

**gain(B|D):**
- Global orders the pairs R > S > L > G. Mean has R ≈ S ≈ L (+0.034 to +0.037, overlapping CIs) > G.
- The paper's only negative pair, R, has the largest gain(B|D) in both feature types (verified). Its CI overlaps S's in both.
- So gain(B|D) does not single out D+R as harmful. It places R at or near the top.

**gain(D|B)** (not asked for, but it is in the table):
- R is lowest in both types (global +0.017, mean +0.011) (verified).
- R also has the highest single-encoder R² (0.670), and gain(D|B) = R²(D+B) − R²(B) by definition.

**Feature types:**
- Global and mean give the same order on each measure (verified). L and G tie at +0.031 in mean gain(D|B).
- In gain(B|D), mean features bring L up close to R and S (+0.017 → +0.034).
- Grid CKA on agentview puts S (0.896) above L (0.830). Elsewhere L is above S.

## Caveats
- **Labels:**
  - The paper deltas come from one trained checkpoint per combo, evaluated with 3 eval seeds only. They are measured on perturbed LIBERO-Plus after joint video-action training.
  - Our features come from clean libero_goal frames, frozen, with linear probes on single-step actions. These are different data and a different regime (recalled from decision 002 and the user's spec).
- **n = 4 pairs** (3 encoders skipped, below). No rank statistic or significance is reported.
- **Dimension mismatch:** D+R (6,656-d) and D+G (6,144-d) probes have more inputs than D+L and D+S (4,096-d). α is tuned per probe, but nothing equalizes capacity.
- **α at the grid edge** (0.01) for L global, R mean and G. A lower α might score slightly higher (speculative).
- **Variants:**
  - Our variants may differ from the paper's (e.g. SAM 3 pre-neck patch mean, VGGT-Ω camera token).
  - D is fp32 while L, R, S and G are fp16. This deviates from the pre-registration, because fp16 DINOv3 overflowed (LOG (f)).

## Skipped encoders
- **V-JEPA 2.1:** only third-party HF conversions exist. The official weights need the repo's code and a choice of how to feed single frames to a video model.
- **UniTok:** needs its repo's code; not cheap.
- **HRP:** weights are on the CMU project site, but HRP fine-tunes existing encoders, so there are several checkpoints on different bases. The paper's base is unknown (user-supplied check 2026-10-17, `../../literature/encoder-weights.md`).

All three are recorded as not run, not as failures.

## Failures and resource use
**Extraction attempts:**

| Try | Outcome | GPU time |
|---|---|---|
| 1 | fp16 DINOv3 gave non-finite features on batch 0; deleted at once | ~1 GPU-min |
| 2 | D finished, plus 5 of 10 G files; deleted for an oversized memory request (16Gi for ~1–4 GiB use) | ~2.25 GPU-h |
| 3 | G and S finished at 1 CPU / 4Gi (02:47–10:40Z 10-20) | ~7.9 GPU-h |

- **Try 2:** probably flagged memory-low on the NRP portal from ~01:00 to 02:44Z 10-20. That was my sizing error (LOG (f)).
- **Try 3:** GPU stayed at 99–100% and memory at 85–89% of the request.
- **GPU total:** ~10.2 GPU-h.
- **Throughput:** VGGT-Ω ran at 12.5 frames/s and SAM 3 at 5.5 frames/s. Both were far slower than my estimate.
- **Diagnostics (CPU, 2 cores each):** D+L/D+R 49 min, D+G 45 min, D+S 30 min.
- **Watchdogs:** the local watchdogs stalled twice when the kubectl login expired. The Prometheus monitor kept watching.
- **Cleanup:** all Jobs and ConfigMaps are deleted (verified 2026-10-21 08:22Z). The CephFS PVC remains.

## Commands
```
kubectl create configmap stacking-encoders-extract-src -n aria-world --from-file=extract_features.py=../../src/extract_features.py
kubectl apply -f extract-job.yaml
kubectl create configmap stacking-encoders-diag-src -n aria-world --from-file=diagnostics.py=../../src/diagnostics.py
kubectl apply -f diag-dlr-job.yaml   # then diag-dg-job.yaml, diag-ds-job.yaml
kubectl logs -n aria-world <pod> > results/<name>-job.log
kubectl delete job -n aria-world <job>; kubectl delete configmap -n aria-world <configmap>
```
The JSONs in `results/` are cut from the `=== result json` blocks of the diagnostics logs.
