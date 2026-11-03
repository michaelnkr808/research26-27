#!/bin/bash
# Simulator for the proxy rollouts (RUN.md step 7), run after setup_env.sh in the same pod. Source it (". setup_sim.sh")
# so PYTHONPATH / LIBERO_CONFIG_PATH reach the caller.
# LIBERO-Plus github.com/sylvestf/LIBERO-plus @ 4976dc3 (no LICENSE file; evaluation use only, nothing redistributed) and
# its assets (HF Sylvest/LIBERO-plus assets.zip, copied to the PVC once by the sim Job). Installed per pod under /work
# (emptyDir): never pip or unzip many small files on CephFS (_standards/08).
# Pins: tested together on the Mac 2026-10-28 (Python 3.11, torch 2.7.1): LIBERO-Plus wants numpy 1.22.4 (old gym and
# robosuite), our code and tests pass on numpy 1.26.4; mujoco 3.3.2 as Fast-WAM's data (README: MuJoCo-version
# sensitive). The libero package has no top-level __init__.py, so pip -e maps nothing: the repo root goes on PYTHONPATH.
set -euo pipefail
W=${SIM_DIR:-/work/sim}
ASSETS=${LP_ASSETS:-/mnt/proxy/sim/assets.zip}
LP_COMMIT=4976dc30028e805ff8094b55501d532c48fec182
# SIM_DEBS (env_bundle.sh restore): install the bundle's saved .debs offline and skip pip (the bundle's venv has the
# packages); SIM_KEEP_DEBS (env_bundle.sh build): keep the downloaded .debs there for the bundle.
if [ -n "${SIM_DEBS:-}" ]; then
  dpkg -i $SIM_DEBS/*.deb >/dev/null
else
  KEEP=${SIM_KEEP_DEBS:-/var/cache/apt/archives}; mkdir -p $KEEP/partial
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends -o Dir::Cache::archives=$KEEP \
    -o APT::Keep-Downloaded-Packages=true curl unzip libmagickwand-dev libexpat1 \
    libfontconfig1 libgl1 libegl1 libglib2.0-0 libosmesa6 libgomp1 >/dev/null
  pip install --no-cache-dir -q numpy==1.26.4 mujoco==3.3.2 robosuite==1.4.0 bddl==1.0.1 future==1.0.0 easydict==1.9 \
    gym==0.25.2 cloudpickle==2.1.0 opencv-python==4.10.0.84 scikit-image==0.24.0 matplotlib==3.9.2 Wand==0.6.13 pyyaml
fi
mkdir -p $W
if [ ! -d $W/LIBERO-plus ]; then
  curl -sSL https://github.com/sylvestf/LIBERO-plus/archive/$LP_COMMIT.tar.gz | tar xz -C $W
  mv $W/LIBERO-plus-$LP_COMMIT $W/LIBERO-plus
fi
L=$W/LIBERO-plus/libero/libero
if [ ! -d $L/assets ]; then
  # the zip nests assets/ under inspire/hdd/.../LIBERO-plus-0/assets (checked 2026-10-28); move it to libero/libero/assets
  unzip -q $ASSETS -d $W/unzip
  mv "$(find $W/unzip -type d -path '*LIBERO-plus-0/assets' | head -1)" $L/assets
  rm -rf $W/unzip
fi
mkdir -p $W/libero-config
printf "benchmark_root: $L\nbddl_files: $L/bddl_files\ninit_states: $L/init_files\ndatasets: $L/../datasets\nassets: $L/assets\n" \
  > $W/libero-config/config.yaml
export LIBERO_CONFIG_PATH=$W/libero-config
export PYTHONPATH=$W/LIBERO-plus:${PYTHONPATH:-}
ls $L/assets
python -c "import numpy, mujoco, robosuite; from libero.libero import benchmark; print('sim ok: numpy', numpy.__version__, 'mujoco', mujoco.__version__, 'robosuite', robosuite.__version__)"
