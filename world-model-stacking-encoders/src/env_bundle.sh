#!/bin/bash
# Prebuilt rollout environment (LOG 2026-11-03 (b): each rollout pod spent ~13 min on setup with its GPU idle).
#   bash env_bundle.sh build     CPU Job: setup_env.sh + setup_sim.sh into a venv (/work/venv, on top of the image's
#                                torch), keep the .debs, tar venv + debs + the LIBERO-Plus checkout (without the 9.4 GB
#                                assets, which pods still unzip from assets.zip) to $BUNDLE_DIR/env-<key>.tar
#   . env_bundle.sh restore      GPU pod: untar to /work, install the .debs offline, activate the venv, then
#                                setup_sim.sh (assets unzip, LIBERO config, exports). Source it so the exports reach
#                                the caller.
# <key> = first 12 hex of sha256(setup_env.sh + setup_sim.sh + this file), so a bundle is only used by the scripts that
# built it; restore fails if that bundle is missing. Same image (pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime) and the
# same /work paths in both places: the venv's paths are absolute. The tar is written to local disk, then copied to
# CephFS in one cp (_standards/08: never pip or unpack many small files on CephFS; memory: CephFS write stalls).
set -euo pipefail
S=$(dirname "${BASH_SOURCE[0]}")
BUNDLE_DIR=${BUNDLE_DIR:-/mnt/proxy/env}
KEY=$(cat $S/setup_env.sh $S/setup_sim.sh $S/env_bundle.sh | sha256sum | cut -c1-12)
BUNDLE=$BUNDLE_DIR/env-$KEY.tar
export SIM_DIR=${SIM_DIR:-/work/sim}
now() { date +%s; }

case "${1:-}" in
build)
  t0=$(now)
  python -m venv --system-site-packages /work/venv
  set +u; . /work/venv/bin/activate; set -u
  bash $S/setup_env.sh > /work/pip-env.log 2>&1 || { tail -50 /work/pip-env.log; exit 1; }
  t1=$(now)
  SIM_KEEP_DEBS=/work/debs . $S/setup_sim.sh
  t2=$(now)
  echo "TIMING setup_env $((t1 - t0)) s, setup_sim $((t2 - t1)) s (apt + pip + checkout + assets unzip)"
  ls /work/debs/*.deb | wc -l
  mkdir -p /work/out
  tar cf /work/out/env.tar --exclude=sim/LIBERO-plus/libero/libero/assets -C /work venv debs sim/LIBERO-plus
  t3=$(now)
  du -sh /work/venv /work/debs /work/sim/LIBERO-plus/libero/libero/assets /work/out/env.tar
  mkdir -p $BUNDLE_DIR
  cp /work/out/env.tar $BUNDLE.part && mv $BUNDLE.part $BUNDLE
  t4=$(now)
  echo "TIMING tar $((t3 - t2)) s, copy to CephFS $((t4 - t3)) s -> $BUNDLE"
  sha256sum $BUNDLE /work/out/env.tar
  ;;
restore)
  [ -f $BUNDLE ] || { echo "no env bundle $BUNDLE for these scripts (run the env-build Job)"; return 1 2>/dev/null || exit 1; }
  t0=$(now)
  tar xf $BUNDLE -C /work
  t1=$(now)
  set +u; . /work/venv/bin/activate; set -u
  SIM_DEBS=/work/debs . $S/setup_sim.sh
  t2=$(now)
  echo "TIMING restore: untar $((t1 - t0)) s, debs + assets unzip + checks $((t2 - t1)) s ($BUNDLE)"
  python -c "import fastwam, torch; print('fastwam ok, torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  ;;
*)
  echo "usage: env_bundle.sh build | . env_bundle.sh restore"; exit 2 ;;
esac
