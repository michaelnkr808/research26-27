#!/bin/bash
# usage: watch_stage2.sh <segment_seconds> <baseline_file>
# Polls our stage 2 Jobs and pods every 5 min; exits on the first change, on repeated kubectl failure, or at the segment end.
SEG=${1:-6000}; BASE=$2
END=$(( $(date +%s) + SEG ))
T=$(mktemp -d)
fails=0
kc() {
  kubectl "$@" > "$T/out" 2> "$T/err" &
  local pid=$!
  ( sleep 45; kill $pid 2>/dev/null ) > /dev/null 2>&1 &
  local wd=$!
  wait $pid; local rc=$?
  kill $wd 2>/dev/null; wait $wd 2>/dev/null
  return $rc
}
JJ='jsonpath={range .items[*]}{"job "}{.metadata.name}{" succeeded="}{.status.succeeded}{" failed="}{.status.failed}{" cond="}{.status.conditions[*].type}{"\n"}{end}'
PJ='jsonpath={range .items[*]}{"pod "}{.metadata.name}{" "}{.status.phase}{"\n"}{end}'
while :; do
  ok=1
  kc get jobs -n aria-world -l experiment=stacking-pairings --request-timeout=20s -o "$JJ" && cp "$T/out" "$T/j" || ok=0
  if [ $ok = 1 ]; then kc get pods -n aria-world -l experiment=stacking-pairings --request-timeout=20s -o "$PJ" && cp "$T/out" "$T/p" || ok=0; fi
  if [ $ok = 1 ] && [ -s "$T/j" ]; then
    fails=0
    grep -v " Succeeded$" "$T/p" > "$T/p2"; grep -v "cond=SuccessCriteriaMet Complete$" "$T/j" > "$T/j2"; cat "$T/j2" "$T/p2" | sort > "$T/cur"   # finished pods and Complete Jobs leave on their own (TTL): not an event   # finished pods leave the list on their own: not an event
    [ -s "$BASE" ] || cp "$T/cur" "$BASE"
    if ! cmp -s "$T/cur" "$BASE"; then
      echo "CHANGE at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      diff "$BASE" "$T/cur"
      cp "$T/cur" "$BASE"
      exit 0
    fi
  else
    fails=$((fails+1))
    if [ $fails -ge 4 ]; then echo "KUBECTL_FAIL at $(date -u +%Y-%m-%dT%H:%M:%SZ) ($fails in a row)"; head -c 600 "$T/err"; exit 3; fi
  fi
  if [ $(date +%s) -ge $END ]; then echo "SEGMENT end at $(date -u +%Y-%m-%dT%H:%M:%SZ), no change, fails=$fails"; exit 0; fi
  sleep 300
done
