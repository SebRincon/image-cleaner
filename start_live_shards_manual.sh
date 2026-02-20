#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/image-cleaner}"
KEY_PATH="${RUNPOD_KEY_PATH:-$HOME/.ssh/runpod_image_cleaner}"
GIT_BRANCH="${GIT_BRANCH:-master}"
POD_NAME_FILTER="${POD_NAME_FILTER:-}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -i "$KEY_PATH")

if [[ ! -f "$KEY_PATH" ]]; then
  echo "SSH key not found: $KEY_PATH" >&2
  exit 1
fi

if [[ ! -f .env ]]; then
  echo "Missing .env in current directory" >&2
  exit 1
fi

source .env

: "${RUN_ID:?Set RUN_ID in .env or environment}"
: "${SHARD_COUNT:?Set SHARD_COUNT in .env or environment}"
if [[ -z "${S3_AWS_ACCESS_KEY_ID:-}" || -z "${S3_AWS_SECRET_ACCESS_KEY:-}" ]]; then
  echo "S3_AWS_ACCESS_KEY_ID and S3_AWS_SECRET_ACCESS_KEY must be set in .env" >&2
  exit 1
fi

if [[ -z "$POD_NAME_FILTER" ]]; then
  POD_NAME_FILTER="people-clean-${RUN_ID}"
fi

pod_names=()
while IFS= read -r pod_name; do
  [[ -n "$pod_name" ]] && pod_names+=( "$pod_name" )
done < <(runpodctl get pod | awk 'NR>1 && $2 ~ /-shard-[0-9]+$/ {print $2}')

filtered=()
for pod_name in "${pod_names[@]}"; do
  if [[ "$pod_name" == *"$POD_NAME_FILTER"* ]]; then
    filtered+=( "$pod_name" )
  fi
done
pod_names=( "${filtered[@]}" )

if [[ ${#pod_names[@]} -eq 0 ]]; then
  pod_names=()
  while IFS= read -r pod_name; do
    [[ -n "$pod_name" ]] && pod_names+=( "$pod_name" )
  done < <(runpodctl get pod | awk 'NR>1 && $2 ~ /^people-clean-.*-shard-[0-9]+$/ {print $2}')

  if [[ ${#pod_names[@]} -eq 0 ]]; then
    echo "No people-clean shard pods found. Set POD_NAME_FILTER or start pods first." >&2
    exit 1
  fi

  echo "No pods matched '${POD_NAME_FILTER}'. Falling back to all people-clean shard pods."
fi

for pod_name in "${pod_names[@]}"; do
  shard_index="${pod_name##*shard-}"
  if ! connect_line="$(runpodctl ssh connect -v "$pod_name")"; then
    echo "Unable to get SSH info for $pod_name"
    continue
  fi
  pod_user="$(echo "$connect_line" | awk '{print $2}')"
  pod_ip="${pod_user#root@}"
  pod_port="$(echo "$connect_line" | awk '{print $4}')"

  if [[ "$pod_user" == "No" || -z "$pod_ip" || -z "$pod_port" ]]; then
    echo "Unable to resolve SSH for $pod_name: $connect_line"
    continue
  fi

  echo "Starting shard $shard_index on $pod_name ($pod_ip:$pod_port)"

  scp "${SSH_OPTS[@]}" -P "$pod_port" .env "root@$pod_ip:$WORKDIR/.env"
ssh "${SSH_OPTS[@]}" -p "$pod_port" "root@$pod_ip" "bash -s -- \"$shard_index\" \"$SHARD_COUNT\" \"$GIT_BRANCH\" \"$WORKDIR\"" <<'REMOTE'
set -euo pipefail

REQUESTED_SHARD_INDEX="$1"
REQUESTED_SHARD_COUNT="$2"
REQUESTED_GIT_BRANCH="$3"
WORKDIR="${4:-/workspace/image-cleaner}"

if [[ -z "$REQUESTED_SHARD_INDEX" || -z "$REQUESTED_SHARD_COUNT" ]]; then
  echo "Missing shard arguments for pod startup" >&2
  exit 1
fi

cd "$WORKDIR"

if [[ ! -f .env ]]; then
  echo "Missing .env on pod at $WORKDIR/.env" >&2
  exit 1
fi

set -a
source .env
set +a

SHARD_INDEX="$REQUESTED_SHARD_INDEX"
SHARD_COUNT="$REQUESTED_SHARD_COUNT"
GIT_BRANCH="${GIT_BRANCH:-$REQUESTED_GIT_BRANCH}"
RUN_ID="${RUN_ID:-2026-02-20_peopleclean_v1}"

export SHARD_INDEX
export SHARD_COUNT
export GIT_BRANCH
export RUN_ID

if [[ -d .git ]]; then
  git fetch origin --prune
  if git show-ref --verify --quiet "refs/heads/$GIT_BRANCH"; then
    git checkout "$GIT_BRANCH"
  else
    git checkout -b "$GIT_BRANCH"
  fi
  git reset --hard "origin/$GIT_BRANCH"
else
  if [[ -n "${GIT_REPO_URL:-}" ]]; then
    git clone --depth 1 --branch "$GIT_BRANCH" "$GIT_REPO_URL" "$WORKDIR"
  else
    echo "No git repo detected and GIT_REPO_URL not set" >&2
    exit 1
  fi
fi

bash bootstrap_cleaner.sh

pkill -f clean_people_s3.py || true
pkill -f monitor_server.py || true

mkdir -p /tmp/people_clean_monitor

export RUN_ID="${RUN_ID:?Set RUN_ID}"
export THRESHOLD="${THRESHOLD:-0.50}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export SRC_PREFIX="${SRC_PREFIX:-train/}"
export DST_PREFIX="${DST_PREFIX:-cleaned/train/}"
export MANIFEST_PREFIX="${MANIFEST_PREFIX:-cleaned/manifests/}"
export MAX_IMAGES="${MAX_IMAGES:-0}"

nohup bash run_cleaner_shard.sh > "/tmp/people_clean_monitor/cleaner_shard_${SHARD_INDEX}_of_${SHARD_COUNT}.log" 2>&1 < /dev/null &
echo "STARTED:$(hostname) shard=$SHARD_INDEX pid=$!"
REMOTE
done
