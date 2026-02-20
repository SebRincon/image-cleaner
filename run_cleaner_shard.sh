#!/usr/bin/env bash
set -euo pipefail

S3_AWS_ACCESS_KEY_ID="${S3_AWS_ACCESS_KEY_ID:?Set S3_AWS_ACCESS_KEY_ID}"
S3_AWS_SECRET_ACCESS_KEY="${S3_AWS_SECRET_ACCESS_KEY:?Set S3_AWS_SECRET_ACCESS_KEY}"
S3_AWS_SESSION_TOKEN="${S3_AWS_SESSION_TOKEN:-}"

AWS_ACCESS_KEY_ID="${S3_AWS_ACCESS_KEY_ID}"
AWS_SECRET_ACCESS_KEY="${S3_AWS_SECRET_ACCESS_KEY}"
if [[ -n "${S3_AWS_SESSION_TOKEN:-}" ]]; then
  AWS_SESSION_TOKEN="$S3_AWS_SESSION_TOKEN"
fi

: "${AWS_ACCESS_KEY_ID:?Set S3_AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?Set S3_AWS_SECRET_ACCESS_KEY}"
: "${AWS_DEFAULT_REGION:=us-east-1}"
: "${SHARD_INDEX:?Set SHARD_INDEX}"
: "${SHARD_COUNT:?Set SHARD_COUNT}"

WORKDIR="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RUN_ID="${RUN_ID:-2026-02-20_peopleclean_v1}"
THRESHOLD="${THRESHOLD:-0.50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SRC_PREFIX="${SRC_PREFIX:-train/}"
DST_PREFIX="${DST_PREFIX:-cleaned/train/}"
MANIFEST_PREFIX="${MANIFEST_PREFIX:-cleaned/manifests/}"
MAX_IMAGES="${MAX_IMAGES:-0}"
EMIT_PRESIGNED_URLS="${EMIT_PRESIGNED_URLS:-0}"
DRY_RUN="${DRY_RUN:-0}"
PRESIGN_SECONDS="${PRESIGN_SECONDS:-604800}"

cd "$WORKDIR"
source .venv/bin/activate

if [[ -n "${AWS_SESSION_TOKEN_HEX:-}" ]]; then
  if command -v xxd >/dev/null 2>&1; then
    AWS_SESSION_TOKEN="$(printf '%s' "$AWS_SESSION_TOKEN_HEX" | xxd -r -p)"
  elif command -v python3 >/dev/null 2>&1; then
    AWS_SESSION_TOKEN="$(python3 - "$AWS_SESSION_TOKEN_HEX" <<'PY'
import sys
print(bytes.fromhex(sys.argv[1]).decode("utf-8"))
PY
)"
  else
    echo "xxd or python3 required to decode AWS_SESSION_TOKEN_HEX in pod" >&2
    exit 1
  fi
  export AWS_SESSION_TOKEN
fi

MONITOR_DIR="${MONITOR_DIR:-/tmp/people_clean_monitor}"
MONITOR_PORT="${MONITOR_PORT:-8000}"
RUN_POD_MONITOR="${ENABLE_POD_MONITOR:-1}"
STATUS_FILE="$MONITOR_DIR/status_shard_${SHARD_INDEX}_of_${SHARD_COUNT}.json"
LOG_FILE="$MONITOR_DIR/cleaner_shard_${SHARD_INDEX}_of_${SHARD_COUNT}.log"

if [[ "$RUN_POD_MONITOR" == "1" ]]; then
  mkdir -p "$MONITOR_DIR"
fi

ARGS=(
  --bucket hackathon-lens-correction-submissions
  --region "$AWS_DEFAULT_REGION"
  --src-prefix "$SRC_PREFIX"
  --dst-prefix "$DST_PREFIX"
  --manifest-prefix "$MANIFEST_PREFIX"
  --run-id "$RUN_ID"
  --shard-index "$SHARD_INDEX"
  --shard-count "$SHARD_COUNT"
  --threshold "$THRESHOLD"
  --batch-size "$BATCH_SIZE"
  --max-images "$MAX_IMAGES"
)

if [[ "$RUN_POD_MONITOR" == "1" ]]; then
  ARGS+=(--status-file "$STATUS_FILE")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  ARGS+=(--dry-run)
fi

if [[ "$EMIT_PRESIGNED_URLS" == "1" ]]; then
  ARGS+=(--emit-presigned-urls)
  ARGS+=(--presign-exp-seconds "$PRESIGN_SECONDS")
fi

if [[ "$RUN_POD_MONITOR" == "1" ]]; then
  (
    python scripts/clean_people_s3.py "${ARGS[@]}" > "$LOG_FILE" 2>&1
  ) &
  CLEANER_PID=$!

  python3 scripts/monitor_server.py \
    --status-file "$STATUS_FILE" \
    --log-file "$LOG_FILE" \
    --port "$MONITOR_PORT" \
    --status-title "shard-$SHARD_INDEX/$SHARD_COUNT" \
    > "$MONITOR_DIR/monitor_server.log" 2>&1 &
  SERVER_PID=$!

  wait "$CLEANER_PID"
  CLEAN_EXIT=$?
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  exit "$CLEAN_EXIT"
else
  python scripts/clean_people_s3.py "${ARGS[@]}"
fi
