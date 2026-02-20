#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ID:?Set RUN_ID}"
INTERVAL="${WATCH_INTERVAL_SECONDS:-20}"
NAME_PREFIX="people-clean-${RUN_ID}-shard-"

while true; do
  echo "===== $(date -u '+%Y-%m-%d %H:%M:%S UTC') ====="
  for pod_id in $(runpodctl get pod | awk "/${NAME_PREFIX}/ {print \$1}"); do
    echo "--- ${pod_id} ---"
    runpodctl get pod "$pod_id" -a | rg "NAME|STATUS|PORTS|VCPU|MEM"
  done
  echo ""
  sleep "$INTERVAL"
done
