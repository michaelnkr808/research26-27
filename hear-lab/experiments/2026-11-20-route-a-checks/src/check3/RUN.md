# Check 3 on the rented machine: how to run it

What this does: for every GN-Matrix start pose in the selected episodes, render GN-Bench's own depth image, project the goal object's box into it, and write whether any part of the box is in view (PLAN.md, "Check 3"). Code in this folder; every parameter comes from `../../config.yaml`.

The simulator part runs only on Linux with a CUDA GPU. The projection, visibility rule and episode selection were unit-tested on the Mac (`test_c3lib.py`).

Paths below assume the machine's home directory is `/root` and that this experiment folder was copied to `/root/route-a-checks` (its `config.yaml` plus `src/check3/`). Change them consistently if not.

## 0. Machine

A Vast.ai Linux x64 instance with one NVIDIA GPU, a driver that supports CUDA 12.4 (550 or newer), and enough disk for the scenes (size is measured in step 3 before downloading). Start from an image that has conda, or install Miniconda first:

```bash
cd /root
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3
source /root/miniconda3/etc/profile.d/conda.sh
nvidia-smi
```

## 1. Environment (GN0 docs/INSTALLATION.md at c927781)

These are GN0's own steps. WNM-3D's install doc was not among the pinned copies read for this code, so only GN0's is followed; check 3 uses only GN0 and GN-Bench-Tools.

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda create -n bae python=3.10 -y
conda activate bae

cd /root
git clone https://github.com/TeleHuman/GN0.git
cd GN0
git checkout c927781
git submodule update --init GN-Bench-Tools
git rev-parse HEAD                       # must start with c927781
git -C GN-Bench-Tools rev-parse HEAD     # must start with 0ce8367; if not, stop and tell the lead

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
conda install -c "nvidia/label/cuda-12.4.0" -c nvidia -c conda-forge cuda-toolkit=12.4 -y
pip install -e ./GN-Bench-Tools
pip install -r requirements.txt
pip install -e .
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124
pip install pyyaml huggingface_hub hf_transfer

python -c "import torch, gsplat, GN_Bench.sims.GN_Bench_simulator.GN_Bench_simulator as g, gn0.benchmark.config; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), gsplat.__version__)"
pip freeze > /root/check3_pip_freeze.txt
```

## 2. Code from this repo

From the Mac (the lead's machine), create the folder on the machine, then copy the experiment's config and this folder, and the shared `hear-lab/src/` (since 2026-11-26 `c3lib.py` lives there; the scripts find it through `HEAR_LAB_SRC`):

```bash
ssh -p <port> root@<host> mkdir -p /root/route-a-checks /root/hear-lab-src
scp -P <port> -r hear-lab/experiments/2026-11-20-route-a-checks/config.yaml hear-lab/experiments/2026-11-20-route-a-checks/src root@<host>:/root/route-a-checks/
scp -P <port> hear-lab/src/*.py root@<host>:/root/hear-lab-src/
# on the machine, in every shell that runs the check 3 scripts:
export HEAR_LAB_SRC=/root/hear-lab-src
```

Record the repo commit the copy came from (`git rev-parse HEAD` on the Mac) in LOG.md; the runner cannot see it on the machine and writes "unknown" in its run metadata.

On the machine, rerun the local tests (they need the test_unseen parquet from step 3 and one labels.json):

```bash
cd /root/route-a-checks/src/check3
C3_TEST_UNSEEN_PARQUET=/root/data/GN_Matrix/InteriorGS_test_unseen.parquet \
C3_LABELS_JSON=/root/data/InteriorGS/<any downloaded scene>/labels.json \
python -m unittest -v test_c3lib
```

## 3. Data

Both datasets are on Hugging Face. InteriorGS is gated: the token must belong to an account whose access request was approved. Do not commit anything downloaded here.

```bash
export HF_TOKEN=<token>          # the lead provides it; do not write it to a file in the repo
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /root/data/GN_Matrix /root/data/InteriorGS

# GN-Matrix: the two parquet files, wherever they sit in the dataset repo
huggingface-cli download TeleEmbodied/GN-Matrix --repo-type dataset \
  --include "*InteriorGS_test_unseen.parquet" "*InteriorGS_train_seen.parquet" \
  --local-dir /root/data/GN_Matrix_raw
find /root/data/GN_Matrix_raw -name "InteriorGS_*.parquet" -exec ln -sf {} /root/data/GN_Matrix/ \;
ls -l /root/data/GN_Matrix              # both files must be there; if train_seen is missing, stop and tell the lead
python -c "from huggingface_hub import HfApi; a=HfApi(); print('GN-Matrix', a.dataset_info('TeleEmbodied/GN-Matrix').sha); print('InteriorGS', a.dataset_info('spatialverse/InteriorGS').sha)" | tee /root/check3_dataset_revisions.txt

# Scenes the selected episodes use (applies the config's selection and seed)
cd /root/route-a-checks/src/check3
python list_scenes.py --gn-matrix-dir /root/data/GN_Matrix --out /root/check3_scenes.txt
```

Before downloading scenes, check the InteriorGS repo layout and the total size. The code expects `<scene>/3dgs_compressed.ply`, `labels.json`, `occupancy.json`, `occupancy.png` (InteriorGS README). If the listing shows another layout (for example one archive per scene), stop and tell the lead.

```bash
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
scenes = [s for s in open("/root/check3_scenes.txt").read().split() if s]
files = list(api.list_repo_tree("spatialverse/InteriorGS", repo_type="dataset", path_in_repo=scenes[0]))
print("example scene listing:", [(f.path, getattr(f, "size", None)) for f in files])
total = 0
for s in scenes:
    total += sum(getattr(f, "size", 0) or 0 for f in api.list_repo_tree("spatialverse/InteriorGS", repo_type="dataset", path_in_repo=s))
print(f"{len(scenes)} scenes, {total/1e9:.1f} GB")
EOF
df -h /root
```

Download only the needed files of the needed scenes (`structure.json` is not used by check 3):

```bash
cd /root
sed 's|$|/3dgs_compressed.ply|' /root/check3_scenes.txt >  /root/check3_include.txt
sed 's|$|/labels.json|'         /root/check3_scenes.txt >> /root/check3_include.txt
sed 's|$|/occupancy.json|'      /root/check3_scenes.txt >> /root/check3_include.txt
sed 's|$|/occupancy.png|'       /root/check3_scenes.txt >> /root/check3_include.txt
xargs -a /root/check3_include.txt -n 400 huggingface-cli download spatialverse/InteriorGS --repo-type dataset --local-dir /root/data/InteriorGS --include
ls /root/data/InteriorGS | wc -l       # should equal: wc -l < /root/check3_scenes.txt
```

## 4. Timing run first (20 episodes)

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate bae
cd /root/route-a-checks/src/check3
python run_check3.py --gn0-root /root/GN0 --gn-matrix-dir /root/data/GN_Matrix \
  --scenes-dir /root/data/InteriorGS --out-dir /root/check3_timing \
  --splits InteriorGS_test_unseen --limit 20 2>&1 | tee /root/check3_timing.log
python summarize_check3.py --gn-matrix-dir /root/data/GN_Matrix --out-dir /root/check3_timing --splits InteriorGS_test_unseen
```

What it checks before any episode, and stops on:

- Depth convention: renders a synthetic plane 3 m away with GN-Bench's gsplat call and decides z-depth vs ray distance (log line `depth convention: z`, details in `run_meta/*.json` under `depth_preflight`). Expected: `z`. It stops if neither fits within 5 cm or if empty pixels are not zero.
- Light scene load vs GN-Bench's own `GNBenchSim.reconfigure`: the first episode is rendered both ways and must match (`reconfigure_check` in the run metadata).
- Every rendered episode: our view matrix and intrinsics must equal the renderer's `_build_pinhole_camera`.

From the timing run, read `load_s_median` (scenes) and `median render s per episode` in `/root/check3_timing/summary.txt`, and estimate the full run as (number of scenes from list_scenes) x scene load + (number of episodes) x render. Tell the lead the estimate before the full run. Also look at the review images in `/root/check3_timing/review/` (there may be fewer than 10 in a 20-episode run) to see that the box points sit on the target object.

## 5. Full run (resume-safe)

```bash
tmux new -s check3
source /root/miniconda3/etc/profile.d/conda.sh && conda activate bae
cd /root/route-a-checks/src/check3
python run_check3.py --gn0-root /root/GN0 --gn-matrix-dir /root/data/GN_Matrix \
  --scenes-dir /root/data/InteriorGS --out-dir /root/check3_full 2>&1 | tee -a /root/check3_full.log
python summarize_check3.py --gn-matrix-dir /root/data/GN_Matrix --out-dir /root/check3_full
```

If the machine drops or the run stops, start the same command again: finished episodes are skipped (rows are written and fsynced one by one; a half-written last line is dropped). Exit code 2 means some scenes had missing files; download them and rerun.

## Outputs (in /root/check3_full)

- `check3_InteriorGS_test_unseen.csv`, `check3_InteriorGS_train_seen.csv`: one row per selected episode. Status `ok`, `missing_id` (label_ins_id not in labels.json) or `missing_box` (entry has no bounding_box). Visibility at 0.15 m (primary) and at 0.05 m and 0.30 m: seen point counts, share of box points seen, and the flag. Also straight-line start-to-goal distance, GN-Matrix path length, the two within-3.0-m flags, number of in-image points that hit empty depth, distance from the camera to the box centre, horizontal distance from the GN-Matrix goal position to the target box, and render time.
- `summary.json`, `summary.txt`: counts per split (visible, not visible, missing, and each by within 3.0 m), timing, and the per-scene frame diagnostic (how many splat centres fall inside label boxes as given vs mirrored; `identity` should win in nearly every scene).
- `review/<split>/NN_row<row>.png` (10 per split) and `review/review_index.csv` (fill in `by_eye_agrees`; rerunning the summary keeps answers already filled in). Green points: seen at 0.15 m; red: inside the image but hidden.
- `review_raw/`, `scenes.jsonl`, `run_meta/<time>.json` (config, parquet hashes, intrinsics, preflight, host, GPU, gsplat version).

Copy all of it off the machine before it is destroyed (`scp -r -P <port> root@<host>:/root/check3_full <somewhere outside the repo>`), together with `/root/check3_pip_freeze.txt`, `/root/check3_dataset_revisions.txt` and `/root/check3_scenes.txt`. The review images and `review_raw/` are InteriorGS renders: keep them out of git (the user's answer of 2026-11-24, `hear-lab/data/README.md`: images and rendered audio stay out; the per-episode CSVs carry InteriorGS labels, instance ids and scene coordinates and stay local too, PR #89 review). The rerun's outputs are in `results/2026-11-20-check3-rerun/check3_full/` of the experiment folder; nothing from that folder is tracked.
