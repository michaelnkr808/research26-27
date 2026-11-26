# Step 2: how to run it

What this does, in one breath: on one rented Linux machine with a GPU, it renders the reference-path video of each chosen train_seen episode with GN0's own renderer and records the camera pose of every frame; then, on the CPU, it puts a looping telephone ring at the target object, renders what a KEMAR head at each frame's pose hears (RLR, SoundSpaces 2.0 settings), checks whether the left/right timing cue points to the right side, and writes clips, pictures and a summary. Design and pass bar: `PLAN.md`. Every parameter: `config.yaml`.

Renting, the hourly price and destroying the instance are the experiment lead's (`../../STANDARDS.md`, "Where experiments run"); each rental needs the lead's own OK. Nothing here rents anything.

Placeholders: `<REPO>` the research26-27 main checkout on the Mac (not a worktree: `git worktree remove` deletes gitignored files), `<COMMIT>` the commit of this folder that is run, `<host>` and `<port>` the instance's SSH address, `<RUN>` the run folder name (`batch1-v1` for the first batch, `full-v1` for the full set; a rerun gets the next `-v<N>`), `<N>` the number of episodes (12 for batch 1, `config.yaml` `episodes.full_set_n` for the full set), `<W>` the worker count (the machine's CPU cores, at most `<N>`).

## 0. On the Mac, before renting

**0a. The telephone ring** (third-party file, not in git). The SoundSpaces sounds archive is about 13 MB (`soundspaces/README.md` at `287184f`). The file name inside it is the AudioGoal default, `telephone.wav` under `sounds/1s_all/` (README tree, lines 51 to 53); the listing below confirms it before anything is extracted.

    mkdir -p <REPO>/hear-lab/experiments/2026-11-26-step2-audio-check/results/inputs/ss-sounds
    cd <REPO>/hear-lab/experiments/2026-11-26-step2-audio-check/results/inputs/ss-sounds
    curl -fL -o sounds.tar.xz http://dl.fbaipublicfiles.com/SoundSpaces/sounds.tar.xz
    ls -l sounds.tar.xz && file sounds.tar.xz             # expect "XZ compressed data", about 13 MB
    shasum -a 256 sounds.tar.xz                           # -> config.yaml sound.archive_sha256
    tar -tJf sounds.tar.xz | grep -i telephone            # expect sounds/1s_all/telephone.wav; if not, stop and tell the lead
    tar -xJf sounds.tar.xz sounds/1s_all/telephone.wav
    file sounds/1s_all/telephone.wav                      # expect RIFF WAVE audio; note the rate and channels
    shasum -a 256 sounds/1s_all/telephone.wav             # -> config.yaml sound.wav_sha256
    cp sounds/1s_all/telephone.wav ../telephone.wav

The URL is `http` as the SoundSpaces README gives it (not `https`); the sha256 check is what makes the download trustworthy. Write the two hashes into `config.yaml` (`sound.archive_sha256`, `sound.wav_sha256`, replacing `pending`) and commit that before the run, together with the lead's confirmation of the PROPOSED values (`direction_check.proposals_confirmed_by_lead`, PLAN.md): `render_audio.py` refuses to start while any value is `pending`, and it checks the wav's sha256 on the machine. If the listing has no telephone file, or the wav is not 1 s long, stop and tell the lead (PLAN.md names this file). `results/` is gitignored except its allow list, so the archive and the wav stay out of git. Licence: SoundSpaces calls its 102 sounds "copyright-free"; the code is CC-BY-4.0 (`data/README.md`); no separate licence for the sounds was read.

**0b. Unit tests** (Mac interpreter with numpy, `/opt/homebrew/bin/python3.11`):

    cd <REPO>/hear-lab/experiments/2026-11-26-step2-audio-check/src
    /opt/homebrew/bin/python3.11 -m unittest -v test_step2lib test_render_smoke
    cd .. && /opt/homebrew/bin/python3.11 src/pick_episodes.py --check     # the committed order, recomputed from the check 3 files

**0c. The reference videos** for the alignment check are route A check 2's renders, `../2026-11-20-route-a-checks/results/check2-render/` (local only; `<scene>/<trajectory_id>.mp4`). Check they are there: `ls ../2026-11-20-route-a-checks/results/check2-render/*/*.mp4 | wc -l` (48).

## 1. Machine

One Linux x64 instance with one NVIDIA GPU whose driver supports CUDA 12.4 (route A check 3 ran GN0's renderer on an RTX 3090), Ubuntu 22.04 (glibc 2.35; the RLR library ran there in route A), image `nvidia/cuda:12.4.1-devel-ubuntu22.04` or one with conda, at least 16 CPU cores for the audio workers, at least 32 GB of RAM (each worker holds one episode's impulse responses, about 0.4 MB per step, up to a few hundred MB for the longest paths), and about 60 GB of disk (the scenes of the batch, two conda environments, the impulse responses). On-demand. The GPU is used only by the pose export; the audio is CPU work.

## 2. Upload the code, the inputs and the token (from the Mac)

    ssh -p <port> root@<host> mkdir -p /workspace/step2/repo/hear-lab/experiments /workspace/step2/inputs /workspace/step2/reference_videos
    cd <REPO> && git rev-parse HEAD                      # must be <COMMIT>; record it in LOG.md
    rsync -av -e "ssh -p <port>" --exclude __pycache__ hear-lab/src root@<host>:/workspace/step2/repo/hear-lab/
    rsync -av -e "ssh -p <port>" --exclude __pycache__ --exclude results \
      hear-lab/experiments/2026-11-26-step2-audio-check root@<host>:/workspace/step2/repo/hear-lab/experiments/
    rsync -av -e "ssh -p <port>" --exclude __pycache__ --exclude results --exclude env \
      hear-lab/experiments/2026-11-20-route-a-checks root@<host>:/workspace/step2/repo/hear-lab/experiments/    # for step 3e only
    rsync -av -e "ssh -p <port>" hear-lab/experiments/2026-11-26-step2-audio-check/results/inputs/telephone.wav root@<host>:/workspace/step2/inputs/
    rsync -av -e "ssh -p <port>" hear-lab/experiments/2026-11-20-route-a-checks/results/check2-render/ root@<host>:/workspace/step2/reference_videos/

The Hugging Face token (InteriorGS is gated) goes to tmpfs only, from the Mac's saved token file, without entering a command line or a transcript (as rung 1):

    ssh -p <port> root@<host> 'umask 077 && cat > /dev/shm/hft' < ~/.cache/huggingface/token

## 3. Setup on the machine

Shell in `/workspace/step2`. `EXP=/workspace/step2/repo/hear-lab/experiments/2026-11-26-step2-audio-check`.

**3a. System packages and conda**

    export EXP=/workspace/step2/repo/hear-lab/experiments/2026-11-26-step2-audio-check
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y git wget curl file binutils libgl1 libglib2.0-0
    cd /workspace/step2
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/step2/miniconda3
    source /workspace/step2/miniconda3/etc/profile.d/conda.sh
    nvidia-smi
    mkdir -p /workspace/step2/out/<RUN>/env

**3b. GN0 and its `bae` environment** (GN0 `docs/INSTALLATION.md` at `c927781`, as route A check 3 and rung 1)

    cd /workspace/step2
    git clone https://github.com/TeleHuman/GN0.git && cd GN0
    git checkout c92778118542bf1dd7d437855d06e0d822e8361d
    git submodule update --init GN-Bench-Tools
    git rev-parse HEAD                       # c92778118542bf1dd7d437855d06e0d822e8361d
    git -C GN-Bench-Tools rev-parse HEAD     # 0ce83677532591c582b0c8a2879ab431a1efff18; if not, stop and tell the lead
    conda create -n bae python=3.10 -y && conda activate bae
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
    conda install -c "nvidia/label/cuda-12.4.0" -c nvidia -c conda-forge cuda-toolkit=12.4 -y
    pip install -e ./GN-Bench-Tools
    pip install -r requirements.txt
    pip install -e .
    pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124
    pip install pyyaml huggingface_hub
    python -c "import torch, gsplat, GN_Bench.sims.GN_Bench_simulator.GN_Bench_simulator as g, gn0.collection; print(torch.__version__, torch.cuda.get_device_name(0), gsplat.__version__)"
    pip freeze > /workspace/step2/out/<RUN>/env/bae-pip-freeze.txt
    conda list > /workspace/step2/out/<RUN>/env/bae-conda-list.txt
    conda deactivate

**3c. The audio environment, the RLR library and the HRTF**

    cd /workspace/step2
    conda create -n step2audio python=3.11 -y && conda activate step2audio
    pip install -r $EXP/src/requirements-step2.txt
    pip freeze > /workspace/step2/out/<RUN>/env/step2audio-pip-freeze.txt
    git clone https://github.com/facebookresearch/rlr-audio-propagation.git
    git -C rlr-audio-propagation checkout 4fd446b5abb5c71fb7a232a083bbddd65f25fc6f
    L=rlr-audio-propagation/RLRAudioPropagationPkg/libs/linux/x64/libRLRAudioPropagation.so
    file $L                                  # "ELF 64-bit LSB shared object, x86-64" (ASCII text = a Git LFS pointer: stop)
    sha256sum $L                             # 31e948eef4908d8cbb403b5f445d9d0eab59fc81b05a658538f8795984f9bfb4
    ldd $L | grep "not found" && echo "MISSING LIBRARIES: stop" || echo "ldd OK"
    mkdir -p hrtf && curl -fL -o hrtf/mit_kemar_normal_pinna.sofa https://sofacoustics.org/data/database/mit/mit_kemar_normal_pinna.sofa
    ls -l hrtf/mit_kemar_normal_pinna.sofa   # 1171764 bytes
    file hrtf/mit_kemar_normal_pinna.sofa    # "Hierarchical Data Format (version 5)"
    sha256sum hrtf/mit_kemar_normal_pinna.sofa   # e7035994f5fd754058424c061380ee92b1d5ed58fccef2887a4266916616acdf
    sha256sum inputs/telephone.wav           # must equal config.yaml sound.wav_sha256
    conda deactivate

Any hash or `file` result that differs: stop and tell the lead. (`render_audio.py` checks the library, HRTF and sound hashes again and refuses to run on a mismatch.)

**3d. Data** (GN-Matrix parquet; the InteriorGS scenes of the run's episodes and of the 2 alignment episodes)

    source /workspace/step2/miniconda3/etc/profile.d/conda.sh && conda activate bae
    cd /workspace/step2/GN0
    hf download TeleEmbodied/GN-Matrix InteriorGS/InteriorGS_train_seen.parquet --repo-type dataset \
      --revision 50a68253b5436f0595779cd3f91baf65228a0238 --local-dir data/datasets/GN_Matrix
    sha256sum data/datasets/GN_Matrix/InteriorGS/InteriorGS_train_seen.parquet | tee /workspace/step2/out/<RUN>/env/parquet.sha256
    python $EXP/src/export_poses.py --step2-config $EXP/config.yaml --order $EXP/episode_order.txt --list-alignment 2 \
      | tee /workspace/step2/alignment_ids.txt
    python $EXP/src/pick_episodes.py --print-scenes <N> > /workspace/step2/scenes.txt
    sed 's/_[^_]*$//' /workspace/step2/alignment_ids.txt >> /workspace/step2/scenes.txt
    sort -u -o /workspace/step2/scenes.txt /workspace/step2/scenes.txt && wc -l /workspace/step2/scenes.txt
    HF_TOKEN="$(cat /dev/shm/hft)" bash -c 'for s in $(cat /workspace/step2/scenes.txt); do
      hf download spatialverse/InteriorGS --repo-type dataset --revision 334dfeea4e0241033b4e5de97c01bc7c9c080530 \
        --local-dir data/scene_datasets/InteriorGS \
        --include "$s/3dgs_compressed.ply" "$s/labels.json" "$s/structure.json" "$s/occupancy.json" "$s/occupancy.png" || exit 1
    done'
    rm -f /dev/shm/hft
    for s in $(cat /workspace/step2/scenes.txt); do for f in 3dgs_compressed.ply labels.json structure.json occupancy.json occupancy.png; do
      [ -s data/scene_datasets/InteriorGS/$s/$f ] || echo "MISSING $s/$f"; done; done      # must print nothing
    ls ~/.cache/huggingface/token /dev/shm/hft 2>&1 | grep -v "No such file" && echo "TOKEN FILE PRESENT: delete it" || echo "no token file"

The alignment ids are printed by GN0's own episode order (`select_render_episodes(split, 0, 2)`); each must have a reference video `/workspace/step2/reference_videos/<scene>/<trajectory_id>.mp4` (route A check 2 rendered indices 0 to 47 of the same order). If the 429 rate limit of Hugging Face stops the loop, wait 5 minutes and run it again (it resumes).

**3e. Unit tests on the machine**

    conda activate step2audio
    cd $EXP/src && python -m unittest -v test_step2lib test_render_smoke 2>&1 | tail -3
    conda deactivate

Route A's check 1 local test on one downloaded scene, which exercises the shared code moved to `hear-lab/src/` on 2026-11-26 against real InteriorGS files (it could not run on the Mac, which no longer holds the check 1 scenes). It only filters the parquet by the scene, so the train_seen parquet and a train_seen scene serve:

    conda activate bae
    cd /workspace/step2/repo/hear-lab/experiments/2026-11-20-route-a-checks/src/check1
    S=$(head -1 /workspace/step2/scenes.txt)
    CHECK1_SCENE_DIR=/workspace/step2/GN0/data/scene_datasets/InteriorGS/$S \
    CHECK1_PARQUET=/workspace/step2/GN0/data/datasets/GN_Matrix/InteriorGS/InteriorGS_train_seen.parquet \
    CHECK1_RLR_ROOT=/workspace/step2/rlr-audio-propagation CHECK1_OUT=/workspace/step2/routea-test-out \
    python -c "import pandas, PIL, yaml" && python -m unittest -v test_local 2>&1 | tail -5   # expect OK; a failure is a finding for the lead
    conda deactivate

This check is optional: if `bae` lacks pandas, skip it and say so in LOG.md (do not install into `bae` after its package list was saved).

## 4. Pose export and videos (GPU, `bae` environment)

    conda activate bae && cd /workspace/step2/GN0
    python $EXP/src/export_poses.py --step2-config $EXP/config.yaml --order $EXP/episode_order.txt --n <N> \
      --alignment-n 2 --reference-videos /workspace/step2/reference_videos \
      --out /workspace/step2/out/<RUN>/export 2>&1 | tee /workspace/step2/out/<RUN>/export.log
    echo "exit $?"
    cat /workspace/step2/out/<RUN>/export/alignment.json
    conda deactivate

It stops if the renderer settings differ from `config.yaml` `poses.renderer_expected`, or if an id is not in the dataset. Exit 1 means an episode failed, a frame count differs from the pose count, or the alignment check failed (a frame count different from route A's video): stop and tell the lead. Watch `nvidia-smi` once a minute in: the export renders at roughly the speed route A check 2 did (48 episodes in about 5 minutes).

## 5. Audio, direction check, pictures, summary (CPU, `step2audio` environment)

    conda activate step2audio && cd $EXP
    export R=/workspace/step2/out/<RUN>
    REPO_COMMIT=<COMMIT> python src/render_audio.py --n <N> --poses-dir $R/export --scenes-root /workspace/step2/GN0/data/scene_datasets/InteriorGS \
      --rlr-root /workspace/step2/rlr-audio-propagation --hrtf /workspace/step2/hrtf/mit_kemar_normal_pinna.sofa \
      --sound /workspace/step2/inputs/telephone.wav --out $R/render --workers <W> 2>&1 | tee $R/render-console.log
    echo "exit $?"

Exit 2: refused (a pending config value, an order or file hash mismatch); exit 3: the preflight failed (the free-field side sign was wrong at some heading; no episode was rendered); exit 1: some episode failed (listed in `timing.json`). In each case stop and tell the lead. The first lines printed are the 8 preflight cases, each with its ITD (about +0.7 ms for a left source, -0.7 ms for a right one, if the library behaves as in route A).

    python src/draw_topdown.py --run-dir $R/render --poses-dir $R/export --scenes-root /workspace/step2/GN0/data/scene_datasets/InteriorGS
    python src/summarize.py --run-dir $R/render --usd-per-hour <dph_total of the instance>
    conda deactivate

`summarize.py` prints the verdict line; the numbers are in `$R/render/summary.json`.

## 6. Manifest, copy-off, check, destroy

On the machine:

    cd /workspace/step2/out/<RUN> && cp ../../alignment_ids.txt ../../scenes.txt env/
    conda activate step2audio && python $EXP/src/make_manifest.py /workspace/step2/out/<RUN>

On the Mac:

    mkdir -p <REPO>/hear-lab/experiments/2026-11-26-step2-audio-check/results/<RUN>
    rsync -av -e "ssh -p <port>" root@<host>:/workspace/step2/out/<RUN>/ <REPO>/hear-lab/experiments/2026-11-26-step2-audio-check/results/<RUN>/
    cd <REPO>/hear-lab/experiments/2026-11-26-step2-audio-check
    /opt/homebrew/bin/python3.11 src/make_manifest.py results/<RUN> --check          # must print "manifest OK"

Then, on the machine once more, `ls ~/.cache/huggingface/token /dev/shm/hft` (both absent), read the account credit, destroy the instance, read the credit again, and write GPU, hourly price, hours, total cost and the estimate beside it into `LOG.md` and the rentals ledger.

## 7. What gets committed

`git status` in the experiment folder shows only the allow-listed files of `results/.gitignore`: `<RUN>/manifest.sha256`, `<RUN>/render/{run_config.json, episode_records.jsonl, timing.json, preflight.json, summary.json, episode_tallies.csv}`, `<RUN>/export/{export_meta.json, alignment.json}`, `<RUN>/env/*`. Before committing, scan them (a hit stays out of git and stays listed in the manifest):

    /usr/bin/grep -rnE "hf_[A-Za-z0-9]{20,}|/Users/|/home/[a-z]|@[A-Za-z0-9.-]+\.[a-z]{2,}|BEGIN [A-Z ]*PRIVATE KEY" $(git ls-files --others --exclude-standard results/<RUN>) || echo "scan clean"

Everything else (poses, videos, per-step records, impulse responses, clips, top-down pictures, logs, `episode.json`) is InteriorGS-derived or holds scene coordinates and stays local. To regenerate them: steps 2 to 5 with the same `<COMMIT>` and a new `<RUN>` name.
