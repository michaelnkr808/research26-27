# Check 2 on the rented machine: how to run it

What this does: one timed real training step of WNM-3D on one H200, started from the released stage 3 checkpoint, on the 48 train_seen episodes rendered on rental 2 (PLAN.md, "Amendment 2026-11-20 (b): check 2 detailed design"; parameters in `../../config.yaml`, block `check2`). Commands are in the amendment's order. Nothing here changes a WNM-3D file or the launcher.

Placeholders: `<host>` and `<port>` (SSH, from `vastai ssh-url <instance>` after `vastai attach ssh`), `<REPO>` (this repo on the lead's Mac), `<RENDERS>` (the folder on the Mac that holds `<scene>/<trajectory_id>.mp4` for the 48 renders), `<HF_TOKEN>` (the lead's Hugging Face token, typed only into the one VGGT-Omega command in step 4; never written into a file, never into this repo). Stop rules (minutes from instance start unless said): environment verified by 90, downloads done within 60 of starting them, conversion within 30, key check within 30, first logged step within 60 of each launch, hard stop at 300.

Machine paths: WNM-3D at `/workspace/WNM-3D`, everything for this check under `/workspace/check2`.

## 0. Before renting (on the Mac)

~~~bash
cd <REPO>
git rev-parse HEAD                                   # record in LOG.md: the commit src/check2 comes from
ls <RENDERS>/*/*.mp4 | wc -l                         # must be 48; if not, stop (re-rendering is a new run)
(cd <RENDERS> && find . -name '*.mp4' | sort | xargs md5 -r) > /tmp/check2_renders.md5
cd hear-lab/experiments/2026-11-20-route-a-checks/src/check2
/opt/homebrew/bin/python3.11 -m unittest -v test_check2   # all pass (the real-Trainer test is skipped on the Mac)
shasum -a 256 check2_steplog.py check2_keycheck.py sitecustomize.py test_check2.py > /tmp/check2_src.sha256
~~~

## 1. Machine record (on the machine)

~~~bash
mkdir -p /workspace/check2/env /workspace/check2/runs /workspace/check2/src
cd /workspace/check2/env
date -u +%FT%TZ > instance_start_seen.txt
nvidia-smi > nvidia-smi.txt; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > gpu.csv
df -h > df.txt; free -g > free.txt; nproc > nproc.txt; cat /etc/os-release > os.txt
~~~

The driver must support CUDA 12.9 (about 575 or newer). If not, stop.

## 2. Environment (WNM-3D docs/INSTALLATION.md section 1 at 7fe2d5f)

~~~bash
cd /workspace
wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /workspace/miniforge3
source /workspace/miniforge3/etc/profile.d/conda.sh
conda create -n wnm_3d python=3.11 -y
conda activate wnm_3d
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
conda install -c "nvidia/label/cuda-12.9.0" -c nvidia -c conda-forge cuda-toolkit=12.9 -y

git clone https://github.com/TeleHuman/WNM-3D.git /workspace/WNM-3D
cd /workspace/WNM-3D
git checkout 7fe2d5f93a8dc4e2a5a2ec6fe174746928ecdcbb
git rev-parse HEAD                                    # must be 7fe2d5f93a8dc4e2a5a2ec6fe174746928ecdcbb
python -m pip install -e .
python -m pip install flash-attn --no-build-isolation # if it starts compiling: MAX_JOBS=8 and watch the 90-minute stop
python -m pip install --upgrade huggingface_hub hf_xet

python - <<'PY'
import torch, gammanav, flash_attn, deepspeed
print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"WNM-3D package: {gammanav.__path__[0]}")
print(f"flash_attn {flash_attn.__version__} deepspeed {deepspeed.__version__} GPU {torch.cuda.get_device_name(0)}")
PY
~~~

`CUDA available` must be `True` and both imports must work. Without flash-attn the code falls back to PyTorch attention, which is not the released path: stop.

Record the environment (before PYTHONPATH is ever set):

~~~bash
cd /workspace/check2/env
pip freeze > pip_freeze.txt; conda list > conda_list.txt; nvcc --version > nvcc.txt
git -C /workspace/WNM-3D rev-parse HEAD > wnm3d_head.txt
python -c "import sitecustomize; print(sitecustomize.__file__)" > preexisting_sitecustomize.txt 2>&1 || true
cat preexisting_sitecustomize.txt                     # "No module named" means the image has none
~~~

## 3. Uploads (from the Mac)

~~~bash
cd <REPO>/hear-lab/experiments/2026-11-20-route-a-checks
rsync -av -e "ssh -p <port>" src/check2/ root@<host>:/workspace/check2/src/check2/
rsync -av -e "ssh -p <port>" /tmp/check2_src.sha256 root@<host>:/workspace/check2/env/
rsync -av -e "ssh -p <port>" <RENDERS>/ root@<host>:/workspace/check2/renders/
rsync -av -e "ssh -p <port>" /tmp/check2_renders.md5 root@<host>:/workspace/check2/env/
~~~

On the machine:

~~~bash
cd /workspace/check2/src/check2 && sha256sum -c /workspace/check2/env/check2_src.sha256
cd /workspace/check2/renders && find . -name '*.mp4' | sort | xargs md5sum > /workspace/check2/env/renders_machine.md5
diff <(awk '{print $1, $2}' /workspace/check2/env/check2_renders.md5) <(awk '{print $1, $2}' /workspace/check2/env/renders_machine.md5) && echo RENDERS-MATCH
~~~

Both checks must pass (md5 -r on the Mac and md5sum print the same "hash path" layout). Then rerun the unit tests on the machine, where the real-Trainer test also runs:

~~~bash
cd /workspace/check2/src/check2 && python -m unittest -v test_check2 2>&1 | tee /workspace/check2/env/unittest.txt
~~~

## 4. Downloads (pinned; about 51 GB)

~~~bash
cd /workspace/WNM-3D && mkdir -p checkpoints
hf download TeleEmbodied/GN-Matrix InteriorGS/InteriorGS_train_seen.parquet --repo-type dataset \
  --revision 50a68253b5436f0595779cd3f91baf65228a0238 --local-dir /workspace/check2/gn_matrix
hf download TeleEmbodied/WNM-3D --include "wnm_3d_stage3_release/**" --local-dir checkpoints \
  --revision 5b8c08dd52028ef4a29e7b87966f27582032fb4f
hf download Wan-AI/Wan2.2-TI2V-5B models_t5_umt5-xxl-enc-bf16.pth Wan2.2_VAE.pth \
  --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e --local-dir checkpoints/Wan2.2-TI2V-5B
hf download Wan-AI/Wan2.1-I2V-14B-480P models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
  --revision 6b73f84e66371cdfe870c72acd6826e1d61cf279 --local-dir checkpoints/Wan2.1-I2V-14B-480P
hf download google/umt5-xxl tokenizer_config.json tokenizer.json spiece.model special_tokens_map.json config.json \
  --revision 66cb9e7e85526fe440a945569e42c72fb6cbc0ad --local-dir checkpoints/umt5-xxl
~~~

VGGT-Omega (gated). Type the token only into this command; a leading space keeps it out of history where HISTCONTROL=ignorespace, and the history is cleared anyway:

~~~bash
 HF_TOKEN=<HF_TOKEN> hf download facebook/VGGT-Omega vggt_omega_1b_512.pt \
  --revision 1041e80fc0e911235d3426b0a3d9a81075111a53 --local-dir checkpoints
history -c
ls ~/.cache/huggingface/token ~/.cache/huggingface/stored_tokens 2>&1   # both must be "No such file"
env | grep -c HF_TOKEN                                                  # must print 0
~~~

Checksums (Hub LFS sha256 values, read 2026-11-20) and sizes:

~~~bash
cd /workspace/WNM-3D/checkpoints
cat > /workspace/check2/env/downloads.sha256 <<'EOF'
e0e0ff763ed8bb6b855782e10f1577d676bc7abce2df0e46cea5d340a05963a5  wnm_3d_stage3_release/model-00001-of-00006.safetensors
3dd6a6ed2735b920badc8b58864ac567f921b5e1ab537f9c755e35e6c65ebd8c  wnm_3d_stage3_release/model-00002-of-00006.safetensors
9d82ec1b43fcb2a8ce39fe692993c8d5944a7620f086b61257b2d79dbf69b156  wnm_3d_stage3_release/model-00003-of-00006.safetensors
936913a457e1d5bce09acc2b5044db1d34524698cc5a838282d1d196875a326c  wnm_3d_stage3_release/model-00004-of-00006.safetensors
d7dffc1c6e31c343e991d82062bfd7c27b120bed9139c832354621042933d508  wnm_3d_stage3_release/model-00005-of-00006.safetensors
4b3ca202db5df95e571095844f5073a47e761cb94ad7f5be8c237d67c32b3944  wnm_3d_stage3_release/model-00006-of-00006.safetensors
7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d  Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36  Wan2.2-TI2V-5B/Wan2.2_VAE.pth
628c9998b613391f193eb67ff68da9667d75f492911e4eb3decf23460a158c38  Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
EOF
sha256sum -c /workspace/check2/env/downloads.sha256
stat -c '%s %n' vggt_omega_1b_512.pt                     # must be 4576706117
sha256sum vggt_omega_1b_512.pt > /workspace/check2/env/vggt.sha256   # recorded (the anonymous Hub listing hides it)
du -sh /workspace/WNM-3D/checkpoints/* > /workspace/check2/env/checkpoint_sizes.txt
~~~

Any mismatch: stop.

## 5. Conversion (docs/TRAINING.md lines 66 to 74, paths only changed)

~~~bash
cd /workspace/WNM-3D
python scripts/data/convert_interiorgs_to_lerobot.py \
  --render-root /workspace/check2/renders \
  --annotations /workspace/check2/gn_matrix/InteriorGS/InteriorGS_train_seen.parquet \
  --output-root data/interiorgs_lerobot_check2 \
  --num-workers 8 2>&1 | tee /workspace/check2/env/convert.log
~~~

Checks (all must hold, else stop):

~~~bash
cd /workspace/WNM-3D/data/interiorgs_lerobot_check2
grep '^\[INFO\]' /workspace/check2/env/convert.log     # matched=48 short_videos=0 missing_videos=<rows - 48>
grep '^\[DONE\]' /workspace/check2/env/convert.log     # episodes=48
python - <<'PY' | tee /workspace/check2/env/convert_checks.txt
import json, pathlib, cv2, math
root = pathlib.Path(".")
info = json.load(open("meta/info.json"))
eps = [json.loads(l) for l in open("meta/episodes.jsonl")]
print("total_episodes", info["total_episodes"], "episodes.jsonl", len(eps))
print("total_frames", info["total_frames"], "sum length", sum(int(e["length"]) for e in eps))
print("parquets", len(list(root.glob("data/chunk-*/episode_*.parquet"))), "mp4s", len(list(root.glob("videos/chunk-*/*/episode_*.mp4"))))
print("pose_sources", info.get("conversion", {}).get("pose_sources"))
e = eps[0]
src = cv2.VideoCapture(f"/workspace/check2/renders/{e['scene']}/{e['trajectory']}.mp4")
out = next(root.glob("videos/chunk-*/*/episode_000000.mp4"))
n_src = int(src.get(cv2.CAP_PROP_FRAME_COUNT)); n_out = int(cv2.VideoCapture(str(out)).get(cv2.CAP_PROP_FRAME_COUNT))
print("episode 0 frames", n_out, "expected ceil(src/3)", math.ceil(n_src / 3))
PY
~~~

Expected: 48 and 48; equal frame totals; 48 parquets and 48 MP4s; pose_sources is the annotations one; episode 0 frames equal ceil(source / 3).

## 6. Dry run, then the start-weights key check

The key check needs the dry run's Hydra overrides, so the dry run comes first. Common settings for every launch (decided 2026-11-20: no gradient accumulation, GLOBAL_BATCH_SIZE not set):

~~~bash
cat > /workspace/check2/launch_env.sh <<'EOF'
export NUM_GPUS=1 SAVE_STRATEGY=no SEED=20261120 AUTO_DOWNLOAD_WEIGHTS=false DEEPSPEED_CFG=${DEEPSPEED_CFG:-zero2}
export INTERIORGS_DATA_ROOT=data/interiorgs_lerobot_check2
export PRETRAINED_MODEL_PATH=checkpoints/wnm_3d_stage3_release
export VGGT_OMEGA_CKPT=checkpoints/vggt_omega_1b_512.pt
export PYTHONUNBUFFERED=1
EOF
cd /workspace/WNM-3D && source /workspace/check2/launch_env.sh
DRY_RUN=true PER_DEVICE_BS=12 MAX_STEPS=60 OUTPUT_DIR=/workspace/check2/runs/dryrun \
  bash scripts/train/wnm_3d_stage2.sh | tee /workspace/check2/runs/dryrun.log
grep -o 'save_strategy=[^ ]*\|per_device_train_batch_size=[^ ]*\|global_batch_size=[^ ]*\|max_steps=[^ ]*\|deepspeed=[^ ]*\|bf16=[^ ]*\|num_frames=[^ ]*\|image_resolution_[a-z]*=[^ ]*\|pretrained_model_path=[^ ]*\|skip_component_loading=[^ ]*' /workspace/check2/runs/dryrun.log
~~~

Check against PLAN step 4: per-device 12, global 12, max_steps 60, zero2.json, bf16 true, 33 frames, 320 by 160, the stage 3 path, skip_component_loading true. Before the first real trial, also check that the trial's saved `experiment_cfg/conf.yaml` reads `save_strategy: 'no'` (a string); if it reads `false`, stop and ask.

Key check (CPU only, about 54 GB RAM; stop at 30 minutes):

~~~bash
cd /workspace/check2/src/check2
CUDA_VISIBLE_DEVICES= python check2_keycheck.py --wnm3d-root /workspace/WNM-3D \
  --dry-run-log /workspace/check2/runs/dryrun.log --out /workspace/check2/keycheck 2>&1 | tee /workspace/check2/keycheck.log
tail -1 /workspace/check2/keycheck.log
~~~

The last line must read `pass=True missing=0 unexpected=0 exit=0`. Exit 2 (keys missing or unexpected) or 3 (loading raised): stop before timing and report `/workspace/check2/keycheck/keycheck.json` to the lead; nothing is fixed.

## 7. Batch trials

One script runs a trial: `nvidia-smi` logging beside it, a timestamped log in which the progress bar's carriage-return updates become separate lines, then the summary.

~~~bash
cat > /workspace/check2/trial.sh <<'EOF'
# usage: bash /workspace/check2/trial.sh <trial_id> <per_device_bs> <max_steps> <warmup>
set -uo pipefail
ID=$1; BS=$2; STEPS=$3; WARM=$4
RUN=/workspace/check2/runs/$ID
[ -e "$RUN" ] && { echo "output folder exists: $RUN"; exit 1; }
cd /workspace/WNM-3D && source /workspace/check2/launch_env.sh
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 1 > "$RUN.nvidia-smi.csv" &
SMI=$!
date +%s.%N > "$RUN.launch_time"
PYTHONPATH=/workspace/check2/src/check2 PER_DEVICE_BS=$BS MAX_STEPS=$STEPS OUTPUT_DIR=$RUN \
  bash scripts/train/wnm_3d_stage2.sh 2>&1 \
  | stdbuf -oL tr '\r' '\n' \
  | while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done > "$RUN.log"
RC=${PIPESTATUS[0]}
kill $SMI
echo "exit $RC" >> "$RUN.log"
if grep -q -E 'CUDA out of memory|OutOfMemoryError' "$RUN.log"; then echo "RESULT $ID: OOM"
elif grep -q -E 'ZeRO-Offload with a client provided optimizer|ZeRORuntimeException' "$RUN.log"; then echo "RESULT $ID: CONFIG ERROR (not a fit result; stop for the lead)"
elif [ "$RC" -ne 0 ]; then echo "RESULT $ID: FAILED exit $RC (read the log; a missing-file error is reported as such)"
elif [ ! -s "$RUN/check2_steps.jsonl" ]; then echo "RESULT $ID: NO STEP LOG (stop; the hook did not attach)"
else echo "RESULT $ID: FITS"
  python /workspace/check2/src/check2/check2_steplog.py summarize "$RUN/check2_steps.jsonl" --warmup "$WARM" --batch "$BS" | tee "$RUN.summary.json"
fi
EOF
~~~

While a trial runs, in a second shell: once training starts, check that `/workspace/check2/runs/<id>/check2_steps.jsonl` exists and that `grep '\[check2\] step logger attached' /workspace/check2/runs/<id>.log` finds one line (if not, stop the trial: `pkill -f experiment.py`). Look at `nvidia-smi` by hand 3 minutes after the first step (if the timed window ends first, say so and use the CSV). Stop a launch that has no logged step 60 minutes after it started.

Order (PLAN step 4, "Batch-size search"):

~~~bash
bash /workspace/check2/trial.sh bs12 12 60 10                 # main timing: 10 warm-up, 50 timed
# if bs12 is OOM: down in this order, stop at the first FITS (that one is the main timing)
bash /workspace/check2/trial.sh bs8 8 60 10
bash /workspace/check2/trial.sh bs6 6 60 10
bash /workspace/check2/trial.sh bs4 4 60 10
bash /workspace/check2/trial.sh bs3 3 60 10
bash /workspace/check2/trial.sh bs2 2 60 10
bash /workspace/check2/trial.sh bs1 1 60 10
# if bs12 FITS: upward fit checks, 10 warm-up + 10 timed, stop at the first OOM
bash /workspace/check2/trial.sh bs16 16 20 10
bash /workspace/check2/trial.sh bs24 24 20 10
# only if bs1 is OOM with zero2: offload fallback, reported as a deviation, same downward order from 12
DEEPSPEED_CFG=zero2_offload bash /workspace/check2/trial.sh off_bs12 12 60 10   # then off_bs8, off_bs6, ...
~~~

A CONFIG ERROR from the offload trial is not "does not fit": stop and report it. OOM at batch 1 with offload as well: the result is "does not fit on one H200 with the released configs"; stop.

After each trial, keep the logs and delete the final model save (about 27 GB): `find /workspace/check2/runs/<id> -name '*.safetensors' -delete` (plus any `global_step*` folder). Record the minutes the save took (the last timestamped lines of the log).

## 8. Copy back and destroy

On the machine:

~~~bash
cd /workspace/check2
mkdir -p meta_copy && cp -r /workspace/WNM-3D/data/interiorgs_lerobot_check2/meta meta_copy/
find env runs keycheck keycheck.log meta_copy launch_env.sh trial.sh -type f \
  ! -name '*.safetensors' ! -name '*.pt' ! -name '*.bin' ! -path '*/global_step*' | sort > copy_list.txt
xargs sha256sum < copy_list.txt > copy_manifest.sha256
wc -l copy_list.txt
~~~

On the Mac:

~~~bash
mkdir -p ~/Desktop/hear-lab-route-a-results/check2-run1
rsync -av -e "ssh -p <port>" --files-from=:/workspace/check2/copy_list.txt root@<host>:/workspace/check2/ ~/Desktop/hear-lab-route-a-results/check2-run1/
rsync -av -e "ssh -p <port>" root@<host>:/workspace/check2/copy_manifest.sha256 ~/Desktop/hear-lab-route-a-results/check2-run1/
cd ~/Desktop/hear-lab-route-a-results/check2-run1 && shasum -a 256 -c copy_manifest.sha256 | grep -vc ': OK$'   # must print 0
~~~

Then on the machine repeat the token check of step 4 (`ls ~/.cache/huggingface/token ~/.cache/huggingface/stored_tokens`, both missing). Read the credit, destroy the instance (`vastai destroy instance <id>`), read the credit again, and write machine, price, hours, cost and every result into LOG.md (the lead's session).
