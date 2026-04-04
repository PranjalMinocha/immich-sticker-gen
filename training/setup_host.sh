#!/usr/bin/env bash
# Chameleon GPU host prep: rclone S3 remote, sync bucket folders to local disk (SSD), optional read-only mount.
# Typical layout: Raw-Data/sa-1b-sample.tar.gz + Teacher-Embeddings/*.npy — tarball is extracted once into Raw-Data/extracted/.
# Secrets: ~/training.env with AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (see README).
# Duplicate at MLOps repo root: ../../setup_host.sh (keep in sync).
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

# Local staging (use NVMe/scratch if available, e.g. LOCAL_DATA_ROOT=/mnt/local/training-data)
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-$HOME/training-data}"
OBJSTORE_RAW_PREFIX="${OBJSTORE_RAW_PREFIX:-Raw-Data}"
OBJSTORE_TEACHER_PREFIX="${OBJSTORE_TEACHER_PREFIX:-Teacher-Embeddings}"
SA1B_SAMPLE_TAR="${SA1B_SAMPLE_TAR:-sa-1b-sample.tar.gz}"
RAW_EXTRACT_SUBDIR="${RAW_EXTRACT_SUBDIR:-extracted}"

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install nload fuse3 curl tar
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

mkdir -p "${LOCAL_DATA_ROOT}"
REMOTE_BASE="${RCLONE_REMOTE_NAME}:${OBJSTORE_BUCKET}"

echo "rclone sync: ${REMOTE_BASE}/${OBJSTORE_RAW_PREFIX} -> ${LOCAL_DATA_ROOT}/Raw-Data"
rclone sync "${REMOTE_BASE}/${OBJSTORE_RAW_PREFIX}" "${LOCAL_DATA_ROOT}/Raw-Data"

echo "rclone sync: ${REMOTE_BASE}/${OBJSTORE_TEACHER_PREFIX} -> ${LOCAL_DATA_ROOT}/Teacher-Embeddings"
rclone sync "${REMOTE_BASE}/${OBJSTORE_TEACHER_PREFIX}" "${LOCAL_DATA_ROOT}/Teacher-Embeddings"

TGZ="${LOCAL_DATA_ROOT}/Raw-Data/${SA1B_SAMPLE_TAR}"
EXTRACT_DIR="${LOCAL_DATA_ROOT}/Raw-Data/${RAW_EXTRACT_SUBDIR}"
MARKER="${EXTRACT_DIR}/.sa1b_sample_extracted"
if [[ -f "$TGZ" ]]; then
  if [[ ! -f "$MARKER" ]]; then
    echo "Extracting ${SA1B_SAMPLE_TAR} -> ${EXTRACT_DIR}"
    rm -rf "${EXTRACT_DIR}"
    mkdir -p "${EXTRACT_DIR}"
    tar -xzf "$TGZ" -C "${EXTRACT_DIR}"
    touch "${MARKER}"
  else
    echo "Skip tar extract (marker exists): ${MARKER}"
  fi
else
  echo "Note: no ${TGZ} — place images under ${EXTRACT_DIR} or adjust paths in YAML."
fi

# Optional: FUSE mount whole bucket (browsing / legacy). Training should use LOCAL_DATA_ROOT for I/O.
case "${RCLONE_ENABLE_MOUNT:-0}" in
  1|true|yes|YES) RCLONE_DO_MOUNT=1 ;;
  *) RCLONE_DO_MOUNT=0 ;;
esac
if [[ "$RCLONE_DO_MOUNT" == "1" ]]; then
  sudo mkdir -p "${MOUNTPOINT}"
  sudo chown -R "$(id -un):$(id -gn)" "${MOUNTPOINT}"
  fusermount -uz "${MOUNTPOINT}" 2>/dev/null || true
  rclone mount "${RCLONE_REMOTE_NAME}:${OBJSTORE_BUCKET}" "${MOUNTPOINT}" \
    --read-only \
    --allow-other \
    --vfs-cache-mode off \
    --dir-cache-time 10s \
    --daemon
  echo "rclone mount: ${REMOTE_BASE} -> ${MOUNTPOINT}"
else
  echo "rclone mount skipped (set RCLONE_ENABLE_MOUNT=1 to enable)."
fi

echo "Done. Local training data root: ${LOCAL_DATA_ROOT}"
echo "  Docker bind: -v ${LOCAL_DATA_ROOT}:/data:ro"
echo "  split_teacher example: image_root=/data/Raw-Data/${RAW_EXTRACT_SUBDIR}  teacher_root=/data/Teacher-Embeddings"
