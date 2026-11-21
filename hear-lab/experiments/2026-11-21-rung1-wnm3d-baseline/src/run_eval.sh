#!/usr/bin/env bash
# Rung 1: run the released WNM-3D single-replica server and the released GN0 remote eval client on the
# first N episodes of the frozen test_unseen order (PLAN.md, config.yaml).
#
# Usage (on the rented machine, after setup.sh):
#   bash run_eval.sh --run-name NAME --n N --repo-commit SHA
# Rerunning the same command resumes: GN0 skips episodes already in <run>/gn0/log/, and a resumed
# launch must have the same settings as the earlier ones (build_commands.py --previous).
#
# Check mode (any machine with bash and a Python that has PyYAML; runs nothing):
#   RUNG1_PYTHON=python3 bash run_eval.sh --print-commands [--stall-override-for-check true|false]
# prints the exact argv arrays and banner lines run mode would use, NUL-separated (check_settings.py).
#
# Environment overrides: RUNG1_ROOT, WNM3D_ROOT, GN0_ROOT, CONDA_SH, RUNG1_PYTHON.

set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SRC_DIR/.." && pwd)"
CONFIG="$EXP_DIR/config.yaml"
HOOK_DIR="$SRC_DIR/rung1_hook"

RUNG1_ROOT="${RUNG1_ROOT:-/workspace/rung1}"
WNM3D_ROOT="${WNM3D_ROOT:-/workspace/WNM-3D}"
GN0_ROOT="${GN0_ROOT:-/workspace/GN0}"
CONDA_SH="${CONDA_SH:-/workspace/miniforge3/etc/profile.d/conda.sh}"
PY="${RUNG1_PYTHON:-$(dirname "$CONDA_SH")/../../envs/bae/bin/python}"

RUN_NAME=""
N=""
REPO_COMMIT=""
PRINT=0
STALL_FOR_CHECK=""

die() { echo "ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-name) RUN_NAME="${2-}"; shift 2 ;;
        --n) N="${2-}"; shift 2 ;;
        --repo-commit) REPO_COMMIT="${2-}"; shift 2 ;;
        --print-commands) PRINT=1; shift ;;
        --stall-override-for-check) STALL_FOR_CHECK="${2-}"; shift 2 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

if [[ -n "$STALL_FOR_CHECK" && "$PRINT" -ne 1 ]]; then
    die "--stall-override-for-check is only allowed with --print-commands"
fi
if [[ "$PRINT" -eq 1 && -z "$RUN_NAME" ]]; then
    RUN_NAME="check"
fi
[[ -n "$RUN_NAME" ]] || die "--run-name is required"
[[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "--run-name may only hold letters, digits, dot, dash, underscore"

RUN_DIR="$RUNG1_ROOT/runs/$RUN_NAME"
RESULT_DIR="$RUN_DIR/gn0"

# Read a NUL-separated list printed by build_commands.py into a named array (bash 3.2 compatible).
read_nul_array() {
    local __name="$1"; shift
    local __tmp __item __rc
    __tmp="$(mktemp)"
    set +e
    "$@" > "$__tmp"
    __rc=$?
    set -e
    if [[ $__rc -ne 0 ]]; then
        rm -f "$__tmp"
        die "build_commands.py failed (exit $__rc): $*"
    fi
    eval "$__name=()"
    while IFS= read -r -d '' __item; do
        eval "$__name+=(\"\$__item\")"
    done < "$__tmp"
    rm -f "$__tmp"
}

BUILD=("$PY" "$SRC_DIR/build_commands.py" --config "$CONFIG")
CHECK_ARGS=()
if [[ -n "$STALL_FOR_CHECK" ]]; then
    CHECK_ARGS=(--stall-override "$STALL_FOR_CHECK" --for-check)
fi

read_nul_array SERVER_CMD "${BUILD[@]}" --emit server
read_nul_array SERVER_ENV "${BUILD[@]}" --emit server-env
read_nul_array SERVER_BANNER "${BUILD[@]}" --emit server-banner
read_nul_array CLIENT_CMD "${BUILD[@]}" --emit client --result-dir "$RESULT_DIR" ${CHECK_ARGS[@]+"${CHECK_ARGS[@]}"}
read_nul_array CLIENT_BANNER "${BUILD[@]}" --emit client-banner ${CHECK_ARGS[@]+"${CHECK_ARGS[@]}"}

HEALTH_URL="$("$PY" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["server"]["health_url"])' "$CONFIG")"
READY_TIMEOUT="$("$PY" -c 'import sys,yaml; print(int(yaml.safe_load(open(sys.argv[1]))["server"]["ready_timeout_s"]))' "$CONFIG")"
HOOK_WAIT="$("$PY" -c 'import sys,yaml; print(int(yaml.safe_load(open(sys.argv[1]))["client"]["hook_wait_s"]))' "$CONFIG")"
N_TOTAL="$("$PY" -c 'import sys,yaml; print(int(yaml.safe_load(open(sys.argv[1]))["episodes"]["n_total"]))' "$CONFIG")"
ORDER_FILE="$EXP_DIR/$("$PY" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["episodes"]["order_file"])' "$CONFIG")"

IDS_FILE="$RUN_DIR/episode_ids.txt"
CLIENT_ENV=(
    "PYTHONPATH=$HOOK_DIR"
    "RUNG1_EPISODE_IDS=$IDS_FILE"
    "RUNG1_EVENTS=$RUN_DIR/rung1_episodes.jsonl"
    "RUNG1_HOOK_STATUS=$RUN_DIR/launch_LAUNCHID/hook_status.json"
    "RUNG1_LAUNCH_ID=LAUNCHID"
)

if [[ "$PRINT" -eq 1 ]]; then
    printf 'SERVER\0';        printf '%s\0' "${SERVER_CMD[@]}"
    printf 'SERVER_ENV\0';    printf '%s\0' "${SERVER_ENV[@]}"
    printf 'SERVER_CWD\0';    printf '%s\0' "$WNM3D_ROOT"
    printf 'SERVER_BANNER\0'; printf '%s\0' "${SERVER_BANNER[@]}"
    printf 'CLIENT\0';        printf '%s\0' "${CLIENT_CMD[@]}"
    printf 'CLIENT_ENV\0';    printf '%s\0' "${CLIENT_ENV[@]}"
    printf 'CLIENT_CWD\0';    printf '%s\0' "$GN0_ROOT"
    printf 'CLIENT_BANNER\0'; printf '%s\0' "${CLIENT_BANNER[@]}"
    printf 'END\0'
    exit 0
fi

# ---------------------------------------------------------------- run mode
[[ "$(uname -s)" == "Linux" ]] || die "run mode needs the rented Linux machine"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
[[ -n "$REPO_COMMIT" ]] || die "--repo-commit is required (git rev-parse HEAD on the Mac; the machine has no repo)"
[[ "$N" =~ ^[1-9][0-9]*$ ]] || die "--n must be a positive integer"
(( N <= N_TOTAL )) || die "--n $N exceeds n_total $N_TOTAL"
[[ -f "$ORDER_FILE" ]] || die "episode order file missing: $ORDER_FILE (make_episode_order.py)"

# Every setting, the pinned sources and the order file, checked before anything starts.
"$PY" "$SRC_DIR/check_settings.py" --config "$CONFIG" --wnm-root "$WNM3D_ROOT" --gn0-root "$GN0_ROOT" \
    --order-file "$ORDER_FILE" || die "check_settings.py failed"

# Our sitecustomize must not shadow one the bae environment already has.
if env -u PYTHONPATH "$PY" -c 'import sitecustomize, sys; print(sitecustomize.__file__)' 2>/dev/null; then
    die "the bae environment already has a sitecustomize module; ours would shadow it"
fi

if curl -sf -o /dev/null "$HEALTH_URL"; then
    die "something already answers at $HEALTH_URL; stop it first (WNM-3D scripts/inference/wnm_3d_kill_remote_eval.sh)"
fi

LAUNCH_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LAUNCH_DIR="$RUN_DIR/launch_$LAUNCH_ID"
mkdir -p "$LAUNCH_DIR" "$RESULT_DIR"

# Episode list for this run: the first N ids of the frozen order. A resumed run must ask for the same list.
head -n "$N" "$ORDER_FILE" > "$LAUNCH_DIR/episode_ids.txt"
[[ "$(wc -l < "$LAUNCH_DIR/episode_ids.txt" | tr -d ' ')" == "$N" ]] || die "could not take $N ids from $ORDER_FILE"
if [[ -f "$IDS_FILE" ]]; then
    cmp -s "$IDS_FILE" "$LAUNCH_DIR/episode_ids.txt" || die "$IDS_FILE differs from the first $N ids of the order"
else
    cp "$LAUNCH_DIR/episode_ids.txt" "$IDS_FILE"
fi

# Launch record: every setting passed, appended to run_meta.jsonl (refuses if it differs from earlier launches).
"$PY" "$SRC_DIR/build_commands.py" --config "$CONFIG" --emit meta --result-dir "$RESULT_DIR" \
    --run-name "$RUN_NAME" --n "$N" --repo-commit "$REPO_COMMIT" --launch-id "$LAUNCH_ID" \
    --previous "$RUN_DIR/run_meta.jsonl" > "$LAUNCH_DIR/meta.json" || die "launch record refused (see above)"
cat "$LAUNCH_DIR/meta.json" >> "$RUN_DIR/run_meta.jsonl"

# Machine and source snapshot for this launch.
nvidia-smi > "$LAUNCH_DIR/nvidia-smi.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$LAUNCH_DIR/gpu.csv"
{
    echo "wnm_3d $(git -C "$WNM3D_ROOT" rev-parse HEAD)"
    echo "gn0 $(git -C "$GN0_ROOT" rev-parse HEAD)"
    echo "gn_bench_tools $(git -C "$GN0_ROOT/GN-Bench-Tools" rev-parse HEAD)"
} > "$LAUNCH_DIR/heads.txt"
(cd "$EXP_DIR" && find config.yaml src -type f ! -path '*/__pycache__/*' | sort | xargs sha256sum) > "$LAUNCH_DIR/src.sha256"
"$PY" -m pip freeze > "$LAUNCH_DIR/pip_freeze_bae.txt" 2>&1 || true
"$(dirname "$CONDA_SH")/../../envs/wnm_3d/bin/python" -m pip freeze > "$LAUNCH_DIR/pip_freeze_wnm_3d.txt" 2>&1 || true
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 \
    > "$LAUNCH_DIR/nvidia-smi.csv" 2>&1 &
SMI_PID=$!

SERVER_PID=""
CLIENT_PID=""
cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    [[ -n "$CLIENT_PID" ]] && kill -TERM "$CLIENT_PID" 2>/dev/null || true
    [[ -n "$SERVER_PID" ]] && kill -TERM "$SERVER_PID" 2>/dev/null || true
    [[ -n "$CLIENT_PID" ]] && wait "$CLIENT_PID" 2>/dev/null || true
    [[ -n "$SERVER_PID" ]] && wait "$SERVER_PID" 2>/dev/null || true
    kill "$SMI_PID" 2>/dev/null || true
    echo "[rung1] launch $LAUNCH_ID ended with exit $rc" | tee -a "$LAUNCH_DIR/run_eval.log"
    exit "$rc"
}
trap cleanup EXIT INT TERM

log() { echo "[rung1 $(date -u +%FT%TZ)] $*" | tee -a "$LAUNCH_DIR/run_eval.log"; }

check_banner() {   # logfile, lines...
    local logfile="$1"; shift
    local line
    for line in "$@"; do
        grep -Fqx -- "$line" "$logfile" || die "expected line not found in $logfile: '$line'"
    done
}

# ---- server
log "starting server: ${SERVER_CMD[*]}"
(
    set +u
    source "$CONDA_SH"
    conda activate wnm_3d
    set -u
    cd "$WNM3D_ROOT"
    exec env "${SERVER_ENV[@]}" "${SERVER_CMD[@]}"
) > "$LAUNCH_DIR/server.log" 2>&1 &
SERVER_PID=$!

t0=$(date +%s)
until curl -sf -o /dev/null "$HEALTH_URL"; do
    kill -0 "$SERVER_PID" 2>/dev/null || die "server exited before it was ready; see $LAUNCH_DIR/server.log"
    (( $(date +%s) - t0 < READY_TIMEOUT )) || die "server not ready after $READY_TIMEOUT s"
    sleep 10
done
log "server ready after $(( $(date +%s) - t0 )) s"
check_banner "$LAUNCH_DIR/server.log" "${SERVER_BANNER[@]}"

# ---- client
CLIENT_ENV=(
    "PYTHONPATH=$HOOK_DIR"
    "RUNG1_EPISODE_IDS=$IDS_FILE"
    "RUNG1_EVENTS=$RUN_DIR/rung1_episodes.jsonl"
    "RUNG1_HOOK_STATUS=$LAUNCH_DIR/hook_status.json"
    "RUNG1_LAUNCH_ID=$LAUNCH_ID"
)
log "starting client: ${CLIENT_CMD[*]}"
(
    set +u
    source "$CONDA_SH"
    conda activate bae
    set -u
    cd "$GN0_ROOT"
    exec env "${CLIENT_ENV[@]}" "${CLIENT_CMD[@]}"
) > "$LAUNCH_DIR/client.log" 2>&1 &
CLIENT_PID=$!

t0=$(date +%s)
until [[ -s "$LAUNCH_DIR/hook_status.json" ]]; do
    if ! kill -0 "$CLIENT_PID" 2>/dev/null; then
        wait "$CLIENT_PID" || true
        CLIENT_PID=""
        die "client exited before the episode filter attached; see $LAUNCH_DIR/client.log"
    fi
    (( $(date +%s) - t0 < HOOK_WAIT )) || die "episode filter did not attach within $HOOK_WAIT s"
    sleep 5
done
log "episode filter attached: $(cat "$LAUNCH_DIR/hook_status.json")"
check_banner "$LAUNCH_DIR/client.log" "${CLIENT_BANNER[@]}"

set +e
wait "$CLIENT_PID"
CLIENT_RC=$?
set -e
CLIENT_PID=""
log "client exited with $CLIENT_RC"

# ---- end record: finished ids must all be requested ones
"$PY" - "$RESULT_DIR/log" "$IDS_FILE" "$LAUNCH_ID" "$CLIENT_RC" >> "$RUN_DIR/run_meta.jsonl" <<'PYEOF'
import json, sys, time
from pathlib import Path
log_dir, ids_file, launch_id, rc = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
wanted = set(l.strip() for l in open(ids_file) if l.strip())
done = set()
for p in log_dir.glob("*.json"):
    d = json.loads(p.read_text())
    if d.get("status") == "complete" and d.get("episode_complete") is True and d.get("id") == p.stem:
        done.add(p.stem)
extra = sorted(done - wanted)
print(json.dumps({"kind": "end", "launch_id": launch_id, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "client_exit": rc, "n_requested": len(wanted), "n_finished": len(done & wanted),
                  "n_extra": len(extra)}, sort_keys=True))
if extra:
    sys.stderr.write(f"ERROR: {len(extra)} finished episodes were not requested, e.g. {extra[:3]}\n")
    sys.exit(3)
PYEOF
tail -1 "$RUN_DIR/run_meta.jsonl" | tee -a "$LAUNCH_DIR/run_eval.log"
exit "$CLIENT_RC"
