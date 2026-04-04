#!/usr/bin/env bash
# Training: Chameleon train host prep — rclone + read-only object store mount for data on GPU nodes.
# Secrets: create ~/training.env on the server with AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
# (and optional overrides below), or export variables before running.
set -euo pipefail

ENV_FILE="${TRAINING_ENV_FILE:-$HOME/training.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
fi

: "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID (e.g. in ~/training.env)}"
: "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY}"

RCLONE_REMOTE_NAME="${RCLONE_REMOTE_NAME:-rclone_s3}"
S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-https://chi.tacc.chameleoncloud.org:7480}"
OBJSTORE_BUCKET="${OBJSTORE_BUCKET:-objstore-proj28}"
MOUNTPOINT="${RCLONE_MOUNTPOINT:-/tmp/rclone-tests/object}"

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install nload fuse3 curl
curl -fsSL https://rclone.org/install.sh | sudo bash

if sudo test -f /etc/fuse.conf; then
  sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf || true
fi

mkdir -p ~/.config/rclone
rclone config delete "${RCLONE_REMOTE_NAME}" 2>/dev/null || true
rclone config create "${RCLONE_REMOTE_NAME}" s3 \
  provider Other \
  access_key_id "${AWS_ACCESS_KEY_ID}" \
  secret_access_key "${AWS_SECRET_ACCESS_KEY}" \
  endpoint "${S3_ENDPOINT_URL}" \
  acl private

sudo mkdir -p "${MOUNTPOINT}"
sudo chown -R "$(id -un):$(id -gn)" "${MOUNTPOINT}"

fusermount -uz "${MOUNTPOINT}" 2>/dev/null || true
rclone mount "${RCLONE_REMOTE_NAME}:${OBJSTORE_BUCKET}" "${MOUNTPOINT}" \
  --read-only \
  --allow-other \
  --vfs-cache-mode off \
  --dir-cache-time 10s \
  --daemon

echo "rclone: ${RCLONE_REMOTE_NAME}:${OBJSTORE_BUCKET} -> ${MOUNTPOINT}"
