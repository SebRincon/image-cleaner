#!/usr/bin/env bash
set -euo pipefail

: "${BUCKET:=hackathon-lens-correction-submissions}"
: "${REGION:=us-east-1}"
: "${RUN_ID:?Set RUN_ID}"
: "${SHARD_COUNT:=8}"
: "${MANIFEST_PREFIX:=cleaned/manifests/}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/scripts/merge_shard_manifests.py" \
  --bucket "$BUCKET" \
  --region "$REGION" \
  --run-id "$RUN_ID" \
  --manifest-prefix "$MANIFEST_PREFIX" \
  --shard-count "$SHARD_COUNT"
