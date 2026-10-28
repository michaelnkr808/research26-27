#!/bin/bash
# Wait for a Job in aria-world to finish, using the public NRP Prometheus only (no kubectl: its OIDC login expires and
# then hangs; memory nautilus-kubectl-auth-expiry). For the auto-chained proxy round (RUN.md): run it in the background
# from the agent session; its exit wakes the agent.
#
#   watch_job.sh JOB_NAME COMPLETIONS [POLL_SECONDS]
#
# Exit codes: 0 all COMPLETIONS succeeded | 2 Job dead (no active pods, not complete, >= 1 failure; seen twice in a row)
#             3 Job metrics gone for 10 polls after being seen (deleted / TTL) | 4 GPU idle: a pod of this Job older
#             than 35 min averaged < 40% GPU utilization over the last 10 min (NRP's floor; _standards/08)
#             5 Job never appeared within 30 min of starting the watch | 6 CPU/RAM out of NRP range: a pod of this
#             Job older than 35 min, not exempt (requests > 1 CPU or > 2 GB), had 10-min CPU < 20% or > 200% of its
#             request, or RSS < 20% of its memory request (> 150% can't happen: limits are <= 1.2x requests)
# One status line per poll on stdout. If this machine can't reach Prometheus (laptop offline), the poll prints
# "offline" and counts toward nothing: no exit 3/5 from a dropped connection, and the 30-min clock for exit 5 pauses.
set -u
JOB=$1; N=$2; POLL=${3:-120}
PROM=${PROM:-https://prometheus.nrp-nautilus.io/api/v1/query}
q() {  # q PROMQL -> one number (sum over series), "nan" when there is no series, "err" when Prometheus can't be reached
  local body
  body=$(curl -sf --max-time 30 -G "$PROM" --data-urlencode "query=$1") || { echo err; return; }
  echo "$body" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    if d.get("status") != "success":
        print("err"); sys.exit()
    r = d["data"]["result"]
    print(sum(float(x["value"][1]) for x in r) if r else "nan")
except Exception:
    print("err")'
}
seen=0; gone=0; dead=0; t0=$(date +%s)
while true; do
  S=$(q "sum(kube_job_status_succeeded{namespace=\"aria-world\",job_name=\"$JOB\"})")
  F=$(q "sum(kube_job_status_failed{namespace=\"aria-world\",job_name=\"$JOB\"})")
  A=$(q "sum(kube_job_status_active{namespace=\"aria-world\",job_name=\"$JOB\"})")
  if [ "$S" = err ] || [ "$F" = err ] || [ "$A" = err ]; then
    echo "$(date -u +%FT%TZ) job=$JOB offline (Prometheus unreachable); not counted"
    t0=$((t0 + POLL)); sleep "$POLL"; continue
  fi
  # pods of this Job running for > 35 min with a 10-min mean GPU utilization below 40
  IDLE=$(q "count((avg_over_time(DCGM_FI_DEV_GPU_UTIL{namespace=\"aria-world\",pod=~\"$JOB-.*\"}[10m]) < 40) and on(pod) ((time() - kube_pod_start_time{namespace=\"aria-world\",pod=~\"$JOB-.*\"}) > 2100))")
  GPU=$(q "avg(avg_over_time(DCGM_FI_DEV_GPU_UTIL{namespace=\"aria-world\",pod=~\"$JOB-.*\"}[10m]))")
  # pods outside NRP's CPU (20-200%) / RAM (>= 20%) ranges, same age rule, skipping pods at the exempt size
  SEL="namespace=\"aria-world\",pod=~\"$JOB-.*\""
  REQC="sum by (pod) (max by (pod, container) (kube_pod_container_resource_requests{$SEL,resource=\"cpu\"}))"
  REQM="sum by (pod) (max by (pod, container) (kube_pod_container_resource_requests{$SEL,resource=\"memory\"}))"
  CPUR="(sum by (pod) (rate(container_cpu_usage_seconds_total{$SEL,container!=\"\"}[10m])) / $REQC)"
  MEMR="(sum by (pod) (max by (pod, container) (container_memory_rss{$SEL,container!=\"\"})) / $REQM)"
  OUT=$(q "count((($CPUR < 0.2) or on(pod) ($CPUR > 2) or on(pod) ($MEMR < 0.2)) and on(pod) ((time() - kube_pod_start_time{$SEL}) > 2100) and on(pod) (($REQC > 1) or on(pod) ($REQM > 2e9)))")
  echo "$(date -u +%FT%TZ) job=$JOB succeeded=$S/$N failed=$F active=$A gpu10m=$GPU idle_pods=$IDLE cpu_ram_out=$OUT"
  if [ "$S" = nan ] && [ "$A" = nan ]; then
    if [ $seen = 1 ]; then gone=$((gone + 1)); [ $gone -ge 10 ] && { echo "EXIT 3: metrics gone"; exit 3; }
    elif [ $(( $(date +%s) - t0 )) -gt 1800 ]; then echo "EXIT 5: never appeared"; exit 5; fi
  else
    seen=1; gone=0
    if python3 -c "import sys; sys.exit(0 if float('$S') >= $N else 1)"; then echo "EXIT 0: complete"; exit 0; fi
    if python3 -c "import sys; a, f = float('$A' if '$A' != 'nan' else 0), float('$F' if '$F' != 'nan' else 0); sys.exit(0 if a == 0 and f >= 1 else 1)"; then
      dead=$((dead + 1)); [ $dead -ge 2 ] && { echo "EXIT 2: dead (no active pods, failures, not complete)"; exit 2; }
    else dead=0; fi
    if [ "$IDLE" != nan ] && [ "$IDLE" != err ] && python3 -c "import sys; sys.exit(0 if float('$IDLE') >= 1 else 1)"; then
      echo "EXIT 4: GPU idle below 40% (NRP floor)"; exit 4
    fi
    if [ "$OUT" != nan ] && [ "$OUT" != err ] && python3 -c "import sys; sys.exit(0 if float('$OUT') >= 1 else 1)"; then
      echo "EXIT 6: CPU or RAM outside NRP range"; exit 6
    fi
  fi
  sleep "$POLL"
done
