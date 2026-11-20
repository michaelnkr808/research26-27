#!/bin/bash
# Render a stage 2 Job template for one configuration and one training seed (RUN-stage2-train.md, RUN-stage2-eval.md).
# Written 2026-11-17. Runs on the Mac (bash 3.2) or Linux; reads only the templates and, with --commit, git objects. Writes one
# file into jobs/rendered/ (or --out-dir). It never contacts the cluster.
#
#   render.sh train <config> <seed>  [--micro-batch 8|4] [--runs-pvc NAME] [--commit C]
#   render.sh smoke <config>         [--clean-smoke 0|1] [--lp-timing 0|1] [--envs N] [--try T] [--model-sha SHA] [--runs-pvc NAME] [--commit C]
#   render.sh clean <config> <seed>  [--clean-rate R] [--parallel P (1-4)] [--envs N] [--try T] [--model-sha SHA] [--runs-pvc NAME] [--commit C]
#   render.sh lp    <config> <seed>  --lp-rate R [--shards N] [--parallel P] [--envs N] [--try T] [--model-sha SHA] [--runs-pvc NAME] [--commit C]
#   common: [--out-dir DIR] [--example] [--extension (required for seeds 3 and 4)]   (--example prefixes the file with a NOT-FOR-SUBMISSION banner)
#
# <config>: none, dinov3, siglip2, c-radiov4, dinov3+siglip2, dinov3+c-radiov4 (pretrained_train.py --config). Kubernetes names
# use CFGK = config with + replaced by - (dinov3+siglip2 -> dinov3-siglip2).
# Without --commit the file keeps FILL_AT_SUBMIT and FILL_SHA_* (the pod aborts on them: exit 10 / 19). With --commit C it fills
# the commit and the sha256 of each /src key from git show C:world-model-stacking-encoders/src/<key>, as stage 1's RUN did by hand.
# Rates: --clean-rate R and --lp-rate R are timing-only episodes per pod-hour from the config's smoke (RUN-stage2-eval.md).
set -euo pipefail

die() { echo "render.sh: $*" >&2; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)

[ $# -ge 2 ] || die "usage: render.sh train|smoke|clean|lp <config> [seed] [options] (see the header)"
KIND=$1; CONFIG=$2; shift 2
SEED=0
case $KIND in
  train|clean|lp) [ $# -ge 1 ] || die "$KIND needs a seed"; SEED=$1; shift ;;
  smoke) ;;
  *) die "kind must be train, smoke, clean or lp, not '$KIND'" ;;
esac
case $CONFIG in
  none|dinov3|siglip2|c-radiov4|dinov3+siglip2|dinov3+c-radiov4) ;;
  *) die "config '$CONFIG' is not one of none dinov3 siglip2 c-radiov4 dinov3+siglip2 dinov3+c-radiov4" ;;
esac
CFGK=${CONFIG//+/-}
[[ "$SEED" =~ ^[0-9]$ ]] || die "seed '$SEED' must be one digit"
if [ "$SEED" -ge 3 ]; then
  case $CONFIG in
    dinov3|dinov3+c-radiov4) ;;   # extension seeds: need --extension (checked after the options)
    *) die "seeds 3 and 4 exist only for dinov3 and dinov3+c-radiov4 (PLAN Decision rule item 7)" ;;
  esac
fi

MB=8; RUNS_PVC=stacking-encoders-proxy; COMMIT=""; MODEL_SHA=""; ENVS=4; TRY=1; CLEAN_RATE=""; LP_RATE=""; NSHARDS=16
PARALLEL=4; CLEAN_SMOKE=""; LP_TIMING=1; OUTDIR="$HERE/rendered"; EXAMPLE=0; EXTENSION=0
while [ $# -gt 0 ]; do
  case $1 in
    --micro-batch) MB=$2; shift 2 ;;
    --runs-pvc) RUNS_PVC=$2; shift 2 ;;
    --commit) COMMIT=$2; shift 2 ;;
    --model-sha) MODEL_SHA=$2; shift 2 ;;
    --envs) ENVS=$2; shift 2 ;;
    --try) TRY=$2; shift 2 ;;
    --clean-rate) CLEAN_RATE=$2; shift 2 ;;
    --lp-rate) LP_RATE=$2; shift 2 ;;
    --shards) NSHARDS=$2; shift 2 ;;
    --parallel) PARALLEL=$2; shift 2 ;;
    --clean-smoke) CLEAN_SMOKE=$2; shift 2 ;;
    --lp-timing) LP_TIMING=$2; shift 2 ;;
    --out-dir) OUTDIR=$2; shift 2 ;;
    --example) EXAMPLE=1; shift ;;
    --extension) EXTENSION=1; shift ;;
    *) die "unknown option '$1'" ;;
  esac
done
num() { [[ "$2" =~ ^[0-9]+$ ]] && [ "$2" -ge 1 ] || die "$1 must be a positive integer, not '$2'"; }
num --envs "$ENVS"; num --try "$TRY"; num --shards "$NSHARDS"; num --parallel "$PARALLEL"
[ "$MB" = 8 ] || [ "$MB" = 4 ] || die "--micro-batch must be 8 or 4 (PLAN micro-batch rule)"
[[ "$RUNS_PVC" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]] || die "--runs-pvc '$RUNS_PVC' is not a valid claim name"
[ -z "$COMMIT" ] || [[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "--commit must be a 40-hex commit"
[ -z "$MODEL_SHA" ] || [[ "$MODEL_SHA" =~ ^[0-9a-f]{64}$ ]] || die "--model-sha must be a sha256"
ceil() { echo $(( ($1 + $2 - 1) / $2 * $2 )); }   # ceil(a / b) * b, integers
if [ "$SEED" -ge 3 ] && [ "$EXTENSION" != 1 ]; then
  die "seed $SEED is an extension seed (PLAN Decision rule item 7): only after a complete three-seed I, with the user's OK for its Jobs; pass --extension"
fi

# Where runs live: stage 1's volume, or the second volume (PLAN amendment 2026-11-15 later, item 3), mounted at /mnt/runs
if [ "$RUNS_PVC" = stacking-encoders-proxy ]; then RUN_ROOT=/mnt/proxy/runs; else RUN_ROOT=/mnt/runs/runs; fi
if [ "$CONFIG" = none ] && [ "$SEED" = 0 ]; then
  RUN_DIR=/mnt/proxy/runs/baseline-none-s0                                     # stage 1's run (stage 1 LOG line 152)
  [ -z "$MODEL_SHA" ] || die "none seed 0 is pinned to stage 1's model.pt sha256; --model-sha is not accepted for it"
  MODEL_SHA=b25e63e61d6297d523eb75f9e334f179e5faac58c06aaa81da92d4ca3fcdc732
  [ "$KIND" != clean ] || die "none seed 0's clean LIBERO evaluation was stage 1's (2,000 of 2,000 records); a repeat would be a second test-set use. Its LIBERO-Plus evaluation and timing smoke are allowed"
else
  RUN_DIR=$RUN_ROOT/pairings-$CFGK-s$SEED
fi
[ -n "$MODEL_SHA" ] || MODEL_SHA=FILL_MODEL_SHA                                 # the pod aborts (13) until it is rendered

SED=()
add() { SED+=(-e "s|$1|$2|g"); }
add __CFG__ "$CONFIG"; add __CFGK__ "$CFGK"; add __SEED__ "$SEED"; add __RUNS_PVC__ "$RUNS_PVC"; add __ENVS__ "$ENVS"
add __RUN_DIR__ "$RUN_DIR"; add __MODEL_SHA__ "$MODEL_SHA"; add __TRY__ "$TRY"

case $KIND in
  train)
    [ "$CONFIG.$SEED" != none.0 ] || die "none seed 0 is stage 1's run (/mnt/proxy/runs/baseline-none-s0); it is not retrained"
    # per config: time cap (h), host memory, checkpoint interval per card; arithmetic in job-train.template.yaml's header
    # (host memory of the encoder configs: measured 2026-11-20, header HOST MEMORY; the first render's 32Gi / 38Gi and 36Gi / 43Gi were too low)
    case $CONFIG in
      none)                           CAP_H=62; MEMR=32Gi; MEML=38Gi; CK_S=700; CK_L=500 ;;
      dinov3|siglip2|c-radiov4)       CAP_H=63; MEMR=40Gi; MEML=48Gi; CK_S=700; CK_L=500 ;;
      dinov3+siglip2|dinov3+c-radiov4) CAP_H=65; MEMR=64Gi; MEML=76Gi; CK_S=600; CK_L=500 ;;
    esac
    CAP=$(( CAP_H * 3600 )); POD=$(ceil $(( CAP + 4680 )) 3600); JOBD=$(( (CAP_H + 34) * 3600 ))
    add __MICRO_BATCH__ "$MB"; add __CKPT_L40S__ "$CK_S"; add __CKPT_L40__ "$CK_L"; add __TRAIN_TIMEOUT_S__ "$CAP"
    add __POD_DEADLINE_S__ "$POD"; add __JOB_DEADLINE_S__ "$JOBD"; add __MEM_REQ__ "$MEMR"; add __MEM_LIM__ "$MEML"
    add __RUN_ROOT__ "$RUN_ROOT"
    TPL=job-train.template.yaml; NAME=job-train-$CFGK-s$SEED.yaml
    SUMMARY="train $CONFIG seed $SEED: micro-batch $MB, ckpt-every L40S $CK_S / L40 $CK_L, cap ${CAP_H} h, pod ${POD} s, Job ${JOBD} s, memory $MEMR / $MEML, out $RUN_ROOT/pairings-$CFGK-s$SEED"
    ;;
  smoke)
    [ "$SEED" = 0 ] || die "the smoke runs on seed 0"
    [ -n "$CLEAN_SMOKE" ] || { if [ "$CONFIG" = none ]; then CLEAN_SMOKE=0; else CLEAN_SMOKE=1; fi; }
    [[ "$CLEAN_SMOKE$LP_TIMING" =~ ^[01][01]$ ]] && [ "$CLEAN_SMOKE$LP_TIMING" != 00 ] || die "--clean-smoke / --lp-timing must be 0 or 1, not both 0"
    add __CLEAN_SMOKE__ "$CLEAN_SMOKE"; add __LP_TIMING__ "$LP_TIMING"
    TPL=job-smoke.template.yaml; NAME=job-smoke-$CFGK-s0.yaml
    SUMMARY="smoke $CONFIG seed 0: clean pass $CLEAN_SMOKE, LIBERO-Plus timing shard $LP_TIMING, envs $ENVS, try $TRY, checkpoint $RUN_DIR"
    ;;
  clean)
    if [ -z "$CLEAN_RATE" ]; then
      [ "$CONFIG" = none ] || die "clean needs --clean-rate R (timing-only clean episodes per pod-hour from this config's smoke)"
      CLEAN_RATE=215                                                             # stage 1's smoke, none (stage 1 LOG 2026-11-16)
    fi
    num --clean-rate "$CLEAN_RATE"
    CAP=$(ceil $(( (3600000 + CLEAN_RATE - 1) / CLEAN_RATE )) 100)               # 2 x 500 / R x 3,600 s
    [ "$PARALLEL" -le 4 ] || die "--parallel $PARALLEL exceeds the 4 clean shards"
    WAVES=$(( (4 + PARALLEL - 1) / PARALLEL )); [ "$WAVES" -ge 2 ] || WAVES=2
    POD=$(ceil $(( 1800 + CAP + 120 )) 600); JOBD=$(ceil $(( 72000 + (WAVES + 2) * POD )) 3600)
    add __ROLLOUT_CAP_S__ "$CAP"; add __POD_DEADLINE_S__ "$POD"; add __JOB_DEADLINE_S__ "$JOBD"; add __PARALLEL__ "$PARALLEL"
    TPL=job-clean-eval.template.yaml; NAME=job-clean-$CFGK-s$SEED.yaml
    SUMMARY="clean $CONFIG seed $SEED: 4 shards, parallel $PARALLEL, rate $CLEAN_RATE/h, cap $CAP s, pod $POD s, Job $JOBD s, envs $ENVS, try $TRY, checkpoint $RUN_DIR"
    ;;
  lp)
    [ -n "$LP_RATE" ] || die "lp needs --lp-rate R from the LIBERO-Plus timing shard (RUN-stage2-eval.md); LIBERO-Plus time is still blank in the PLAN"
    num --lp-rate "$LP_RATE"
    [ "$PARALLEL" -le "$NSHARDS" ] || die "--parallel $PARALLEL exceeds --shards $NSHARDS"
    PER=$(( (30090 + NSHARDS - 1) / NSHARDS ))
    CAP=$(ceil $(( (2 * PER * 3600 + LP_RATE - 1) / LP_RATE )) 100)
    POD=$(ceil $(( 1800 + CAP + 120 )) 600)
    WAVES=$(( (NSHARDS + PARALLEL - 1) / PARALLEL ))
    JOBD=$(ceil $(( 72000 + (WAVES + 2) * POD )) 3600)
    add __ROLLOUT_CAP_S__ "$CAP"; add __POD_DEADLINE_S__ "$POD"; add __JOB_DEADLINE_S__ "$JOBD"
    add __NSHARDS__ "$NSHARDS"; add __PARALLEL__ "$PARALLEL"
    TPL=job-lp-eval.template.yaml; NAME=job-lp-$CFGK-s$SEED.yaml
    SUMMARY="lp $CONFIG seed $SEED: $NSHARDS shards of <= $PER rollouts, parallel $PARALLEL, rate $LP_RATE/h, cap $CAP s, pod $POD s, Job $JOBD s, envs $ENVS, try $TRY, checkpoint $RUN_DIR"
    ;;
esac

mkdir -p "$OUTDIR"; OUT=$OUTDIR/$NAME; TMP=$OUT.tmp
{
  if [ "$EXAMPLE" = 1 ]; then
    echo "# EXAMPLE RENDER, NOT FOR SUBMISSION (written by render.sh $KIND $CONFIG ${SEED}; placeholders FILL_* left in: the pod aborts on them)."
    echo "# Example inputs are illustrations, not measurements: --clean-rate ${CLEAN_RATE:-none given}, --lp-rate ${LP_RATE:-none given} (215 is stage 1's"
    echo "# timing-only CLEAN rate for none; no LIBERO-Plus rate has been measured)."
  fi
  echo "# RENDERED by jobs/render.sh from jobs/$TPL: $SUMMARY"
  sed "${SED[@]}" "$HERE/$TPL"
} > "$TMP"
# second volume lines: kept only when the runs go to a second volume
if [ "$RUNS_PVC" = stacking-encoders-proxy ]; then grep -v '@RUNSVOL' "$TMP" > "$TMP.2"; mv "$TMP.2" "$TMP"; fi
# Hugging Face token: not mounted for config none (stage 1 mounted no secret)
if [ "$CONFIG" = none ]; then grep -v '@HFTOKEN' "$TMP" > "$TMP.2"; mv "$TMP.2" "$TMP"; fi
if [ -n "$COMMIT" ]; then
  git -C "$REPO" cat-file -e "$COMMIT^{commit}" || die "commit $COMMIT not found"
  sed "s/{name: SRC_GIT_COMMIT, value: FILL_AT_SUBMIT}/{name: SRC_GIT_COMMIT, value: $COMMIT}/" "$TMP" > "$TMP.2"; mv "$TMP.2" "$TMP"
  for f in $(grep -v '^ *#' "$TMP" | grep -o 'FILL_SHA_[A-Za-z0-9_.]*' | sed 's/^FILL_SHA_//' | sort -u); do
    h=$(git -C "$REPO" show "$COMMIT:world-model-stacking-encoders/src/$f" | shasum -a 256 | cut -c1-64)
    sed "s/FILL_SHA_$f /$h /" "$TMP" > "$TMP.2"; mv "$TMP.2" "$TMP"
  done
  [ "$(grep -c "value: $COMMIT}" "$TMP")" = 1 ] || die "commit not rendered exactly once"
  [ "$(grep -v '^ *#' "$TMP" | grep -c -E 'FILL_SHA_|FILL_AT_SUBMIT' || true)" = 0 ] || die "a FILL_SHA_ or FILL_AT_SUBMIT placeholder is left"
fi
LEFT=$(grep -v '^ *#' "$TMP" | grep -o '__[A-Z0-9_]*__' | sort -u | tr '\n' ' ' || true)
[ -z "$LEFT" ] || die "placeholders left in $OUT: $LEFT"
mv "$TMP" "$OUT"
echo "wrote $OUT"
echo "  $SUMMARY"
echo "  remaining FILL_ placeholders: $(grep -v '^ *#' "$OUT" | grep -o 'FILL_[A-Z_]*[A-Za-z0-9_.]*' | sort -u | tr '\n' ' ' || true)"
