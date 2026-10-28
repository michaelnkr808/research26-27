#!/bin/bash
# Python environment for the proxy-replication Jobs (image pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime). Pins match
# experiments/2026-10-27-proxy-calibration/requirements.txt; torch/torchvision listed so pip keeps the image's CUDA build.
# FastWAM at 7faa711 (MIT) without its own deps (it pins deepspeed, wandb, torchcodec etc. that our code doesn't import).
set -euo pipefail
pip install --no-cache-dir -q torch==2.7.1 torchvision==0.22.1 numpy==2.2.6 h5py==3.16.0 transformers==4.57.6 \
  timm==1.0.30 open_clip_torch==3.3.0 einops==0.8.1 av==15.1.0 pandas==2.3.3 pyarrow==21.0.0 safetensors==0.5.3 \
  pillow==13.0.0 imageio==2.37.0 imageio-ffmpeg==0.6.0 omegaconf==2.3.0 hydra-core==1.3.2 regex==2025.11.3 \
  termcolor==2.5.0 rich==14.2.0 GitPython==3.1.45 huggingface_hub==0.36.2
pip install --no-cache-dir -q --no-deps \
  https://github.com/yuantianyuan01/FastWAM/archive/7faa71108368fbb3b6885649f112af607427a2d4.tar.gz
echo "=== pip freeze"; pip freeze
python -c "import fastwam, torch; print('fastwam ok, torch', torch.__version__, 'cuda', torch.cuda.is_available())"
