# 2026-10-18 — Phase 1 pilot: feature extraction on libero_goal (1 GPU)

**Purpose:** First cached-feature run for Phase 1. Checks that the pipeline works on GPU, and measures GPU throughput and storage per encoder, so the full 4-suite × N-encoder extraction can be sized. These features also feed the first CKA/probe diagnostics. Go-ahead from the user 2026-10-18.

**What runs:** `../../src/extract_features.py` at commit `f4a7b82` (passed to the Job as `SRC_GIT_COMMIT` and stored in each output file's metadata).
- Data: `yifengzhu-hf/LIBERO-datasets`, `libero_goal/*.hdf5` (10 files, 6.37 GB, from the HF API on 2026-10-18), downloaded inside the Job.
- Encoders: SigLIP 2 (`google/siglip2-large-patch16-384`) and C-RADIOv4 (`nvidia/C-RADIOv4-SO400M`, remote code pinned to `c0457f5`). DINOv3 is left out until the HF token Secret exists (gated repo).
- Variants: the paper says only "primarily large (L) variants" (App. A, verified against the arXiv HTML 2026-10-18). SigLIP 2-L at 384 px is our pick. RADIO has no L, so SO400M. **Neither is confirmed to be the paper's exact checkpoint.**
- Both cams, every frame, rotate-180 on, 4×4 grid + mean + global, stored as fp16. Compute in **fp16** (fp16 autocast for RADIO), not bf16, so that Turing cards (2080 Ti) can run it.

**Hardware:** 1× `nvidia.com/gpu`, Turing or newer (product list in `job.yaml`), 2 CPU / 10 Gi. Prometheus at about 23:40Z on 2026-10-18 showed no free Ampere+ GPU on a node with ≥2 CPU and ≥10 Gi left. Only 2080 Ti (2) and older Pascal cards qualified. The 3 free A6000s sat on a node with 0.07 CPU left.

**Local checks before submitting (CPU, Mac, scratch venv with the same pins as the Job):**
- verified: both encoders run end to end on 1 demo (`turn_on_the_stove`, 80 steps × 2 cams). No NaNs. SigLIP 2-L global 1024-d; RADIO summary 2304-d, tokens 1152-d.
- verified: fp16 vs fp32 on that demo. SigLIP 2-L min cosine ≥ 0.9995 (relative error ≤ 1.3%). RADIO under fp16 autocast, min cosine ≥ 0.99999. Caveat: CPU autocast may cast fewer ops than CUDA, so GPU fp16 error isn't measured. The script now raises an error on any non-finite feature.
- verified: file size per frame is 36.9 KB (SigLIP 2-L) and 43.8 KB (RADIO).
- Fixed: casting RADIO to fp16 failed ("expected m1 and m2 to have the same dtype"), because its remote code upcasts some inputs. It now keeps fp32 weights and runs under autocast.

**Commands:**
```
kubectl create configmap stacking-encoders-extract-src -n aria-world --from-file=../../src/extract_features.py
kubectl apply -f job.yaml
kubectl logs -n aria-world job/stacking-encoders-extract-libero-goal > results/job.log
kubectl delete job -n aria-world stacking-encoders-extract-libero-goal
kubectl delete configmap -n aria-world stacking-encoders-extract-src
```
The features stay on the PVC (`/persistent/stacking-encoders/features/`). They're too big for git. A copy of the log also goes to `/persistent/stacking-encoders/logs/`.

## Predictions (written before running; speculative)
- Frames: about 89k (500 demos × ~89 steps × 2 cams, extrapolated from the 50-demo file).
- Storage: about 3.3 GB (SigLIP 2-L) + 3.9 GB (RADIO) ≈ 7.2 GB on the PVC. The Job stops early if fewer than 10 GB are free.
- Throughput on a 2080 Ti in fp16: about 100–300 frames/s per encoder, so 5–15 min per encoder. Wall-clock under 45 min including pip and the 6.4 GB download.
- GPU utilization may fall below 100%, because frames are read and resized in the main process with no DataLoader. Check the dashboard at 30–40 min.
