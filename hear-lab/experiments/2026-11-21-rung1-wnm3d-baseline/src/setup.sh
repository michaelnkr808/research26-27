#!/usr/bin/env bash
# Rung 1 environment and data setup on a fresh Vast.ai Linux machine (one NVIDIA GPU, driver for CUDA 12.9).
#
# Follows WNM-3D docs/INSTALLATION.md (sections 1 and 2) and docs/DOWNLOADS.md at 7fe2d5f, and GN0
# docs/INSTALLATION.md at c927781, as route-a checks 2 and 3 ran them, with the two library repairs check 2
# needed (huggingface_hub 0.36.2, pyarrow 23.0.1 in the wnm_3d environment).
#
# Usage (as root on the machine, from the uploaded experiment folder):
#   bash setup.sh                 # every stage in order
#   bash setup.sh <stage> ...     # stages: record apt conda env-wnm env-bae downloads verify token-check
#
# Hugging Face token: only the InteriorGS download needs it (gated). It is read from /dev/shm/hft
# (tmpfs, put there by the lead from the Mac), exported only inside the subshell of that one download,
# and the file is removed when the download stage ends, whatever the outcome. Nothing calls
# "hf auth login", so no token file is written under ~/.cache/huggingface; token-check verifies that.
#
# The pinned values below are also in config.yaml; check_settings.py fails if the two differ.

set -eo pipefail

WNM3D_REPO="https://github.com/TeleHuman/WNM-3D.git"
WNM3D_COMMIT="7fe2d5f93a8dc4e2a5a2ec6fe174746928ecdcbb"
GN0_REPO="https://github.com/TeleHuman/GN0.git"
GN0_COMMIT="c927781"
GN_BENCH_TOOLS_COMMIT="0ce8367"
CKPT_REPO="TeleEmbodied/WNM-3D"
CKPT_REVISION="5b8c08dd52028ef4a29e7b87966f27582032fb4f"
CKPT_INCLUDE="wnm_3d_stage3_release/**"
TOKENIZER_REPO="google/umt5-xxl"
TOKENIZER_REVISION="66cb9e7e85526fe440a945569e42c72fb6cbc0ad"
GN_MATRIX_REPO="TeleEmbodied/GN-Matrix"
GN_MATRIX_REVISION="50a68253b5436f0595779cd3f91baf65228a0238"
GN_MATRIX_FILE="InteriorGS/InteriorGS_test_unseen.parquet"
GN_MATRIX_SHA256="de4fd64a4e27ef552017eebf43aeb02992618b49c8800aa9bbc09dd30a0e1347"
INTERIORGS_REPO="spatialverse/InteriorGS"
INTERIORGS_REVISION="334dfeea4e0241033b4e5de97c01bc7c9c080530"
N_SCENES_EXPECTED="839"
WNM_TORCH="2.8.0"
WNM_TORCHVISION="0.23.0"
WNM_TORCHAUDIO="2.8.0"
WNM_TORCH_INDEX="https://download.pytorch.org/whl/cu129"
WNM_CUDA_TOOLKIT="12.9"
WNM_HF_HUB="0.36.2"
WNM_PYARROW="23.0.1"
BAE_TORCH="2.4.0"
BAE_TORCHVISION="0.19.0"
BAE_TORCHAUDIO="2.4.0"
BAE_TORCH_INDEX="https://download.pytorch.org/whl/cu124"
BAE_CUDA_TOOLKIT="12.4"
GSPLAT_INDEX="https://docs.gsplat.studio/whl/pt24cu124"

WS="/workspace"
WNM3D_ROOT="$WS/WNM-3D"
GN0_ROOT="$WS/GN0"
CONDA_DIR="$WS/miniforge3"
CONDA_SH="$CONDA_DIR/etc/profile.d/conda.sh"
ENV_DIR="$WS/rung1/env"
TOKEN_FILE="/dev/shm/hft"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SRC_DIR/.." && pwd)"

mkdir -p "$ENV_DIR"
log() { echo "[setup $(date -u +%FT%TZ)] $*" | tee -a "$ENV_DIR/setup.log"; }
die() { log "STOP: $*"; exit 1; }
activate() { set +u; source "$CONDA_SH"; conda activate "$1"; }

stage_record() {
    log "machine record"
    date -u +%FT%TZ > "$ENV_DIR/setup_start.txt"
    nvidia-smi > "$ENV_DIR/nvidia-smi.txt"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$ENV_DIR/gpu.csv"
    df -h > "$ENV_DIR/df.txt"; free -g > "$ENV_DIR/free.txt"; nproc > "$ENV_DIR/nproc.txt"
    cat /etc/os-release > "$ENV_DIR/os.txt"
    cat "$ENV_DIR/gpu.csv"
}

stage_apt() {
    log "apt packages (libgl1 and libglib2.0-0 were needed for cv2 in route-a check 3)"
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git wget curl libgl1 libglib2.0-0 > "$ENV_DIR/apt.log"
}

stage_conda() {
    if [[ -f "$CONDA_SH" ]]; then log "miniforge already at $CONDA_DIR"; return; fi
    log "installing Miniforge"
    cd "$WS"
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash Miniforge3-Linux-x86_64.sh -b -p "$CONDA_DIR"
}

stage_env_wnm() {
    log "wnm_3d environment (WNM-3D docs/INSTALLATION.md section 1)"
    activate base
    conda env list | grep -q '^wnm_3d ' || conda create -n wnm_3d python=3.11 -y
    activate wnm_3d
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install torch==$WNM_TORCH torchvision==$WNM_TORCHVISION torchaudio==$WNM_TORCHAUDIO --index-url "$WNM_TORCH_INDEX"
    conda install -c "nvidia/label/cuda-$WNM_CUDA_TOOLKIT.0" -c nvidia -c conda-forge cuda-toolkit=$WNM_CUDA_TOOLKIT -y
    [[ -d "$WNM3D_ROOT/.git" ]] || git clone "$WNM3D_REPO" "$WNM3D_ROOT"
    git -C "$WNM3D_ROOT" checkout "$WNM3D_COMMIT"
    [[ "$(git -C "$WNM3D_ROOT" rev-parse HEAD)" == "$WNM3D_COMMIT" ]] || die "WNM-3D HEAD is not $WNM3D_COMMIT"
    (cd "$WNM3D_ROOT" && python -m pip install -e .)
    MAX_JOBS=8 python -m pip install flash-attn --no-build-isolation
    # DOWNLOADS.md asks for "pip install --upgrade huggingface_hub hf_xet"; the upgrade to 2.x breaks
    # transformers 4.51.3 (route-a check 2), so the version pip install -e . resolved is kept.
    python -m pip install "huggingface_hub==$WNM_HF_HUB" hf_xet
    # pyarrow newer than 23 needs numpy 2; WNM-3D pins numpy 1.26.4 (route-a check 2).
    python -m pip install "pyarrow==$WNM_PYARROW"
    python -m pip check | tee "$ENV_DIR/pip_check_wnm_3d.txt"
    python - <<'PY' | tee "$ENV_DIR/verify_wnm_3d.txt"
import torch, gammanav, flash_attn, huggingface_hub, pyarrow
print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"WNM-3D package: {gammanav.__path__[0]}")
print(f"flash_attn {flash_attn.__version__} huggingface_hub {huggingface_hub.__version__} pyarrow {pyarrow.__version__}")
print(f"GPU {torch.cuda.get_device_name(0)}")
assert torch.cuda.is_available(), "CUDA not available"
PY
    python -m pip freeze > "$ENV_DIR/pip_freeze_wnm_3d.txt"
    conda list > "$ENV_DIR/conda_list_wnm_3d.txt"
}

stage_env_bae() {
    log "bae environment (WNM-3D docs/INSTALLATION.md section 2, GN0 docs/INSTALLATION.md)"
    [[ -d "$GN0_ROOT/.git" ]] || git clone "$GN0_REPO" "$GN0_ROOT"
    git -C "$GN0_ROOT" checkout "$GN0_COMMIT"
    git -C "$GN0_ROOT" submodule update --init GN-Bench-Tools
    [[ "$(git -C "$GN0_ROOT" rev-parse HEAD)" == "$GN0_COMMIT"* ]] || die "GN0 HEAD is not $GN0_COMMIT"
    [[ "$(git -C "$GN0_ROOT/GN-Bench-Tools" rev-parse HEAD)" == "$GN_BENCH_TOOLS_COMMIT"* ]] || die "GN-Bench-Tools HEAD is not $GN_BENCH_TOOLS_COMMIT"
    activate base
    conda env list | grep -q '^bae ' || conda create -n bae python=3.10 -y
    activate bae
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install torch==$BAE_TORCH torchvision==$BAE_TORCHVISION torchaudio==$BAE_TORCHAUDIO --index-url "$BAE_TORCH_INDEX"
    conda install -c "nvidia/label/cuda-$BAE_CUDA_TOOLKIT.0" -c nvidia -c conda-forge cuda-toolkit=$BAE_CUDA_TOOLKIT -y
    cd "$GN0_ROOT"
    python -m pip install -e ./GN-Bench-Tools
    python -m pip install -r requirements.txt
    python -m pip install -e .
    python -m pip install gsplat --index-url "$GSPLAT_INDEX"
    python -m pip install pyyaml   # for the rung 1 scripts (build_commands.py, check_settings.py)
    python -m pip check | tee "$ENV_DIR/pip_check_bae.txt" || true
    python -c "import torch, gsplat, yaml, GN_Bench.sims.GN_Bench_simulator.GN_Bench_simulator as g, gn0.evaluation.remote; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), gsplat.__version__)" \
        | tee "$ENV_DIR/verify_bae.txt"
    python -m pip freeze > "$ENV_DIR/pip_freeze_bae.txt"
    conda list > "$ENV_DIR/conda_list_bae.txt"
}

stage_downloads() {
    activate wnm_3d
    date -u +%FT%TZ > "$ENV_DIR/downloads_start.txt"
    trap 'rm -f "$TOKEN_FILE"' EXIT

    log "checkpoint $CKPT_REPO@$CKPT_REVISION ($CKPT_INCLUDE)"
    mkdir -p "$WNM3D_ROOT/checkpoints"
    hf download "$CKPT_REPO" --include "$CKPT_INCLUDE" --local-dir "$WNM3D_ROOT/checkpoints" --revision "$CKPT_REVISION"
    "$CONDA_DIR/envs/bae/bin/python" - "$EXP_DIR/config.yaml" > "$ENV_DIR/checkpoint.sha256" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
for name, sha in cfg["checkpoint"]["shard_sha256"].items():
    print(f"{sha}  wnm_3d_stage3_release/{name}")
PY
    (cd "$WNM3D_ROOT/checkpoints" && sha256sum -c "$ENV_DIR/checkpoint.sha256") || die "checkpoint sha256 mismatch"
    du -sh "$WNM3D_ROOT/checkpoints/wnm_3d_stage3_release" > "$ENV_DIR/checkpoint_size.txt"

    log "tokenizer path named by the checkpoint"
    grep -rn "tokenizer_path\|skip_component_loading\|eval_bf16" "$WNM3D_ROOT/checkpoints/wnm_3d_stage3_release/experiment_cfg/conf.yaml" \
        | tee "$ENV_DIR/checkpoint_conf_keys.txt" || true
    if grep -q "tokenizer_path: *$TOKENIZER_REPO *$" "$ENV_DIR/checkpoint_conf_keys.txt"; then
        # The server runs with HF_HUB_OFFLINE=1, so the tokenizer must be in the cache under refs/main;
        # it is fetched once here and accepted only if main is the pinned revision.
        hf download "$TOKENIZER_REPO" tokenizer_config.json tokenizer.json spiece.model special_tokens_map.json config.json \
            | tee "$ENV_DIR/tokenizer_download.txt"
        grep -q "$TOKENIZER_REVISION" "$ENV_DIR/tokenizer_download.txt" \
            || die "the tokenizer's main is no longer the pinned $TOKENIZER_REVISION; ask the lead"
    else
        log "the checkpoint does not name $TOKENIZER_REPO as its tokenizer: read $ENV_DIR/checkpoint_conf_keys.txt; if it names any other Hub id or a missing path, stop and tell the lead"
    fi

    log "GN-Matrix $GN_MATRIX_FILE at $GN_MATRIX_REVISION"
    mkdir -p "$GN0_ROOT/data/datasets/GN_Matrix"
    hf download "$GN_MATRIX_REPO" "$GN_MATRIX_FILE" --repo-type dataset --revision "$GN_MATRIX_REVISION" \
        --local-dir "$GN0_ROOT/data/datasets/GN_Matrix"
    echo "$GN_MATRIX_SHA256  $GN0_ROOT/data/datasets/GN_Matrix/$GN_MATRIX_FILE" | sha256sum -c - || die "GN-Matrix parquet sha256 mismatch"

    "$CONDA_DIR/envs/bae/bin/python" -c "import sys, pyarrow.parquet as pq; t = pq.read_table(sys.argv[1], columns=['scene']); print('\n'.join(sorted(set(t.column('scene').to_pylist()))))" \
        "$GN0_ROOT/data/datasets/GN_Matrix/$GN_MATRIX_FILE" > "$ENV_DIR/scenes.txt"
    [[ "$(wc -l < "$ENV_DIR/scenes.txt" | tr -d ' ')" == "$N_SCENES_EXPECTED" ]] || die "expected $N_SCENES_EXPECTED scenes in test_unseen"

    log "InteriorGS at $INTERIORGS_REVISION: whole folders of the $N_SCENES_EXPECTED test_unseen scenes"
    [[ -s "$TOKEN_FILE" ]] || die "no token at $TOKEN_FILE (the lead streams it there from the Mac)"
    mkdir -p "$GN0_ROOT/data/scene_datasets/InteriorGS"
    sed 's|$|/*|' "$ENV_DIR/scenes.txt" > "$ENV_DIR/interiorgs_include.txt"
    (
        HF_TOKEN="$(cat "$TOKEN_FILE")"
        export HF_TOKEN
        xargs -a "$ENV_DIR/interiorgs_include.txt" -n 200 \
            hf download "$INTERIORGS_REPO" --repo-type dataset --revision "$INTERIORGS_REVISION" \
            --local-dir "$GN0_ROOT/data/scene_datasets/InteriorGS" --include
    )
    rm -f "$TOKEN_FILE"
    trap - EXIT
    n_dirs="$(find "$GN0_ROOT/data/scene_datasets/InteriorGS" -mindepth 1 -maxdepth 1 -type d ! -name '.*' | wc -l | tr -d ' ')"
    [[ "$n_dirs" == "$N_SCENES_EXPECTED" ]] || die "$n_dirs scene folders, expected $N_SCENES_EXPECTED"
    missing=0
    while read -r s; do
        for f in 3dgs_compressed.ply occupancy.json occupancy.png; do   # what the GN-Bench simulator opens
            [[ -s "$GN0_ROOT/data/scene_datasets/InteriorGS/$s/$f" ]] || { echo "missing $s/$f"; missing=1; }
        done
    done < "$ENV_DIR/scenes.txt"
    [[ "$missing" == 0 ]] || die "scene files missing"
    du -sh "$GN0_ROOT/data/scene_datasets/InteriorGS" > "$ENV_DIR/interiorgs_size.txt"
    date -u +%FT%TZ > "$ENV_DIR/downloads_end.txt"
    stage_token_check
}

stage_verify() {
    log "episode ids: the frozen order must hold exactly the ids GN0 builds from the pinned parquet"
    "$CONDA_DIR/envs/bae/bin/python" - "$GN0_ROOT/data/datasets/GN_Matrix/$GN_MATRIX_FILE" "$EXP_DIR/results/episode_order.txt" <<'PY' | tee "$ENV_DIR/verify_order.txt"
import sys, pyarrow.parquet as pq
rows = pq.read_table(sys.argv[1], columns=["scene", "trajectory_id"]).to_pylist()
ids = [f"{r['scene']}_{r['trajectory_id']}" for r in rows]   # GN0 get_episode_id of "<scene>/<trajectory_id>.json"
order = [l for l in open(sys.argv[2]).read().splitlines() if l]
assert len(ids) == len(set(ids)) == 5000, (len(ids), len(set(ids)))
assert len(order) == len(set(order)) == 5000
assert set(ids) == set(order), "the frozen order and the parquet hold different ids"
print("ORDER-MATCHES-PARQUET 5000")
PY
    "$CONDA_DIR/envs/bae/bin/python" "$SRC_DIR/check_settings.py" --config "$EXP_DIR/config.yaml" \
        --wnm-root "$WNM3D_ROOT" --gn0-root "$GN0_ROOT" --order-file "$EXP_DIR/results/episode_order.txt" \
        | tee "$ENV_DIR/check_settings.txt"
    git -C "$WNM3D_ROOT" rev-parse HEAD > "$ENV_DIR/wnm3d_head.txt"
    git -C "$GN0_ROOT" rev-parse HEAD > "$ENV_DIR/gn0_head.txt"
    git -C "$GN0_ROOT/GN-Bench-Tools" rev-parse HEAD > "$ENV_DIR/gn_bench_tools_head.txt"
    env -u PYTHONPATH "$CONDA_DIR/envs/bae/bin/python" -c "import sitecustomize; print(sitecustomize.__file__)" \
        > "$ENV_DIR/preexisting_sitecustomize.txt" 2>&1 && die "the bae environment already has a sitecustomize module" || true
    log "setup verified"
}

stage_token_check() {
    local hfh="${HF_HOME:-$HOME/.cache/huggingface}" bad=0
    for f in "$TOKEN_FILE" "$hfh/token" "$hfh/stored_tokens"; do
        if [[ -e "$f" ]]; then echo "TOKEN FILE PRESENT: $f"; bad=1; else echo "no $f"; fi
    done | tee "$ENV_DIR/token_check.txt"
    grep -q "TOKEN FILE PRESENT" "$ENV_DIR/token_check.txt" && die "a token file is on disk; delete it" || true
}

STAGES=("$@")
[[ ${#STAGES[@]} -gt 0 ]] || STAGES=(record apt conda env-wnm env-bae downloads verify token-check)
for st in "${STAGES[@]}"; do
    case "$st" in
        record) stage_record ;;
        apt) stage_apt ;;
        conda) stage_conda ;;
        env-wnm) stage_env_wnm ;;
        env-bae) stage_env_bae ;;
        downloads) stage_downloads ;;
        verify) stage_verify ;;
        token-check) stage_token_check ;;
        *) die "unknown stage $st" ;;
    esac
done
