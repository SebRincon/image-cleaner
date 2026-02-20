#!/usr/bin/env bash
set -euo pipefail

: "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY}"

RUN_ID="${RUN_ID:-2026-02-20_peopleclean_v1}"
SHARD_COUNT="${SHARD_COUNT:-4}"
REGION="${REGION:-us-east-1}"
CODE_DIR="${CODE_DIR:-/workspace/image-cleaner}"
GIT_REPO_URL="${GIT_REPO_URL:-}"
GIT_BRANCH="${GIT_BRANCH:-main}"
IMAGE_NAME="${RUNPOD_IMAGE_NAME:-runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04}"
GPU_TYPE="${RUNPOD_GPU_TYPE:-NVIDIA GeForce RTX 5090}"
CONTAINER_DISK_SIZE="${CONTAINER_DISK_SIZE:-50}"
FORCE_EXPLICIT_LAUNCH="${FORCE_EXPLICIT_LAUNCH:-0}"
RUNPOD_START_SSH="${RUNPOD_START_SSH:-0}"
RUNPOD_PORTS="${RUNPOD_PORTS:-8000/http}"
MONITOR_PORT="${MONITOR_PORT:-8000}"
ENABLE_POD_MONITOR="${ENABLE_POD_MONITOR:-1}"
SESSION_TOKEN_ARGS=()
SSH_FLAGS=()
PORT_FLAGS=()

if [[ -n "${RUNPOD_PORTS}" ]]; then
  PORT_FLAGS=("--ports" "$RUNPOD_PORTS")
fi

if [[ "$RUNPOD_START_SSH" == "1" ]]; then
  SSH_FLAGS+=("--startSSH")
fi

encode_session_token_hex() {
  local token="$1"
  if command -v xxd >/dev/null 2>&1; then
    printf '%s' "$token" | xxd -p | tr -d '\n'
  elif command -v python3 >/dev/null 2>&1; then
    python3 - "$token" <<'PY'
import sys
print(sys.argv[1].encode().hex())
PY
  else
    echo "xxd or python3 required to encode AWS_SESSION_TOKEN for runpodctl launch" >&2
    return 1
  fi
}

if [[ -n "${AWS_SESSION_TOKEN:-}" ]]; then
  if [[ "$AWS_SESSION_TOKEN" == *"="* ]]; then
    SESSION_TOKEN_HEX="$(encode_session_token_hex "$AWS_SESSION_TOKEN")"
    SESSION_TOKEN_ARGS+=(--env "AWS_SESSION_TOKEN_HEX=${SESSION_TOKEN_HEX}")
  else
    SESSION_TOKEN_ARGS+=(--env "AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN}")
  fi
fi

if ! command -v runpodctl >/dev/null 2>&1; then
  echo "runpodctl not found in PATH" >&2
  exit 1
fi

create_with_template() {
  local idx="$1"
  local pod_name="$2"
  runpodctl create pod \
    --templateId "$RUNPOD_TEMPLATE_ID" \
    --name "$pod_name" \
    --env "AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID" \
    --env "AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY" \
    --env "AWS_DEFAULT_REGION=$REGION" \
    --env "MONITOR_PORT=$MONITOR_PORT" \
    --env "ENABLE_POD_MONITOR=$ENABLE_POD_MONITOR" \
    --env "SHARD_INDEX=$idx" \
    --env "SHARD_COUNT=$SHARD_COUNT" \
    --env "RUN_ID=$RUN_ID" \
    --env "THRESHOLD=${THRESHOLD:-0.50}" \
    --env "BATCH_SIZE=${BATCH_SIZE:-16}" \
    --env "SRC_PREFIX=${SRC_PREFIX:-train/}" \
    --env "DST_PREFIX=${DST_PREFIX:-cleaned/train/}" \
    --env "MANIFEST_PREFIX=${MANIFEST_PREFIX:-cleaned/manifests/}" \
    --env "MAX_IMAGES=${MAX_IMAGES:-0}" \
    --env "DRY_RUN=${DRY_RUN:-0}" \
    --env "EMIT_PRESIGNED_URLS=${EMIT_PRESIGNED_URLS:-0}" \
    --env "PRESIGN_SECONDS=${PRESIGN_SECONDS:-604800}" \
    "${SESSION_TOKEN_ARGS[@]}" \
    "${SSH_FLAGS[@]}" \
    "${PORT_FLAGS[@]}" \
    --env "GIT_REPO_URL=$GIT_REPO_URL" \
    --env "GIT_BRANCH=$GIT_BRANCH" \
    --args "bash -lc \"$STARTUP_CMD\""
}

create_with_explicit() {
  local idx="$1"
  local pod_name="$2"
  runpodctl create pod \
    --imageName "$IMAGE_NAME" \
    --gpuType "$GPU_TYPE" \
    --containerDiskSize "$CONTAINER_DISK_SIZE" \
    --name "$pod_name" \
    --env "AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID" \
    --env "AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY" \
    --env "AWS_DEFAULT_REGION=$REGION" \
    --env "MONITOR_PORT=$MONITOR_PORT" \
    --env "ENABLE_POD_MONITOR=$ENABLE_POD_MONITOR" \
    --env "SHARD_INDEX=$idx" \
    --env "SHARD_COUNT=$SHARD_COUNT" \
    --env "RUN_ID=$RUN_ID" \
    --env "THRESHOLD=${THRESHOLD:-0.50}" \
    --env "BATCH_SIZE=${BATCH_SIZE:-16}" \
    --env "SRC_PREFIX=${SRC_PREFIX:-train/}" \
    --env "DST_PREFIX=${DST_PREFIX:-cleaned/train/}" \
    --env "MANIFEST_PREFIX=${MANIFEST_PREFIX:-cleaned/manifests/}" \
    --env "MAX_IMAGES=${MAX_IMAGES:-0}" \
    --env "DRY_RUN=${DRY_RUN:-0}" \
    --env "EMIT_PRESIGNED_URLS=${EMIT_PRESIGNED_URLS:-0}" \
    --env "PRESIGN_SECONDS=${PRESIGN_SECONDS:-604800}" \
    "${SESSION_TOKEN_ARGS[@]}" \
    "${SSH_FLAGS[@]}" \
    "${PORT_FLAGS[@]}" \
    --env "GIT_REPO_URL=$GIT_REPO_URL" \
    --env "GIT_BRANCH=$GIT_BRANCH" \
    --args "bash -lc \"$STARTUP_CMD\""
}

for ((i=0; i<SHARD_COUNT; i++)); do
  POD_NAME="people-clean-${RUN_ID}-shard-${i}"
  echo "Launching ${POD_NAME}"

  STARTUP_CMD="if [[ ! -f \"$CODE_DIR/bootstrap_cleaner.sh\" || ! -f \"$CODE_DIR/run_cleaner_shard.sh\" ]]; then "
  STARTUP_CMD+="if [[ -z \"$GIT_REPO_URL\" ]]; then "
  STARTUP_CMD+="echo \"Code not found at $CODE_DIR and GIT_REPO_URL not set\" >&2; exit 1; "
  STARTUP_CMD+="fi; "
  STARTUP_CMD+="mkdir -p \"$(dirname "$CODE_DIR")\" \"$CODE_DIR\"; "
  STARTUP_CMD+="git clone --depth 1 --branch \"$GIT_BRANCH\" \"$GIT_REPO_URL\" \"$CODE_DIR\"; "
  STARTUP_CMD+="fi; "
  STARTUP_CMD+="cd \"$CODE_DIR\" && bash bootstrap_cleaner.sh && bash run_cleaner_shard.sh"

  if [[ "$FORCE_EXPLICIT_LAUNCH" == "1" ]]; then
    echo "FORCE_EXPLICIT_LAUNCH=1, using image/gpu launch path"
    create_with_explicit "$i" "$POD_NAME"
  else
    if [[ -n "${RUNPOD_TEMPLATE_ID:-}" ]]; then
      if ! create_with_template "$i" "$POD_NAME"; then
        echo "Template launch failed for ${POD_NAME}, retrying with explicit image/gpu flags..."
        create_with_explicit "$i" "$POD_NAME"
      fi
    else
      echo "RUNPOD_TEMPLATE_ID not set, using explicit image/gpu launch path"
      create_with_explicit "$i" "$POD_NAME"
    fi
  fi
done
