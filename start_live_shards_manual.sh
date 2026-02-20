#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/image-cleaner}"
KEY_PATH="${RUNPOD_KEY_PATH:-$HOME/.ssh/runpod_image_cleaner}"
GIT_BRANCH="${GIT_BRANCH:-master}"
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

for pod_name in $(runpodctl get pod | rg "people-clean-${RUN_ID}" | awk '{print $2}'); do
  shard_index="${pod_name##*shard-}"
  conn=$(runpodctl ssh connect "$pod_name")
  pod_ip=$(echo "$conn" | awk '{print $2}' | sed 's/root@//')
  pod_port=$(echo "$conn" | awk '{print $4}')

  if [[ "$pod_ip" == "No" ]]; then
    echo "Unable to resolve SSH for $pod_name"
    continue
  fi

  echo "Starting shard $shard_index on $pod_name ($pod_ip:$pod_port)"

  scp "${SSH_OPTS[@]}" -P "$pod_port" .env "root@$pod_ip:$WORKDIR/.env"
  ssh "${SSH_OPTS[@]}" -p "$pod_port" "root@$pod_ip" "bash -s" <<EOF
set -euo pipefail

cd "$WORKDIR"
source .env
git fetch origin --prune
if git show-ref --verify --quiet "refs/heads/$GIT_BRANCH"; then
  git checkout "$GIT_BRANCH"
else
  git checkout -b "$GIT_BRANCH"
fi
git reset --hard "origin/$GIT_BRANCH"

bash bootstrap_cleaner.sh

pkill -f \"clean_people_s3.py\" || true
pkill -f \"monitor_server.py\" || true

mkdir -p /tmp/people_clean_monitor

export SHARD_INDEX=\"$shard_index\"
export SHARD_COUNT=\"${SHARD_COUNT}\"
export RUN_ID=\"${RUN_ID}\"
export THRESHOLD=\"${THRESHOLD}\"
export BATCH_SIZE=\"${BATCH_SIZE}\"
export SRC_PREFIX=\"${SRC_PREFIX}\"
export DST_PREFIX=\"${DST_PREFIX}\"
export MANIFEST_PREFIX=\"${MANIFEST_PREFIX}\"
export MAX_IMAGES=\"${MAX_IMAGES}\"

nohup bash run_cleaner_shard.sh > \"/tmp/people_clean_monitor/cleaner_shard_${shard_index}_manual.log\" 2>&1 &
echo \"STARTED:$pod_name shard=$shard_index\"
EOF
done
