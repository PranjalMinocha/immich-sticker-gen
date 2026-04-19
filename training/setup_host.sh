#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${TRAINING_ENV_FILE:-$HOME/training.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
fi

RCLONE_SYNC_ENABLED="${RCLONE_SYNC_ENABLED:-0}"
RCLONE_REMOTE_NAME="${RCLONE_REMOTE_NAME:-rclone_s3}"
S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-https://chi.tacc.chameleoncloud.org:7480}"
OBJSTORE_BUCKET="${OBJSTORE_BUCKET:-objstore-proj28}"
MOUNTPOINT="${RCLONE_MOUNTPOINT:-/tmp/rclone-tests/object}"

LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-$HOME/training-data}"
OBJSTORE_RAW_PREFIX="${OBJSTORE_RAW_PREFIX:-Raw-Data}"
OBJSTORE_TEACHER_PREFIX="${OBJSTORE_TEACHER_PREFIX:-Teacher-Embeddings}"
SA1B_SAMPLE_TAR="${SA1B_SAMPLE_TAR:-sa-1b-sample.tar.gz}"
RAW_EXTRACT_SUBDIR="${RAW_EXTRACT_SUBDIR:-extracted}"

# New manifest-driven layout support (works with user-managed mounted bucket)
MOUNTED_DATA_ROOT="${MOUNTED_DATA_ROOT:-$MOUNTPOINT}"
OBJSTORE_DATA_ROOT="${OBJSTORE_DATA_ROOT:-$MOUNTED_DATA_ROOT}"
DATASET_MANIFESTS_DIR_NAME="${DATASET_MANIFESTS_DIR_NAME:-dataset_manifests}"
TRAIN_MANIFEST_FILE="${TRAIN_MANIFEST_FILE:-train_manifest.csv}"
VAL_MANIFEST_FILE="${VAL_MANIFEST_FILE:-val_manifest.csv}"
IMAGES_DIR_NAME="${IMAGES_DIR_NAME:-images}"
ANNOTATIONS_DIR_NAME="${ANNOTATIONS_DIR_NAME:-annotations}"
TEACHER_EMBEDDINGS_DIR_NAME="${TEACHER_EMBEDDINGS_DIR_NAME:-Teacher-Embeddings}"
EMBEDDINGS_SUBDIR="${EMBEDDINGS_SUBDIR:-sa_000000}"
STRICT_CHECKS="${STRICT_CHECKS:-1}"

case "$RCLONE_SYNC_ENABLED" in
  1|true|yes|YES)
    : "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID (e.g. in ~/training.env)}"
    : "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY}"

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
    fi

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
    fi
    ;;
  *)
    echo "Skipping rclone sync/install (RCLONE_SYNC_ENABLED=${RCLONE_SYNC_ENABLED})."
    ;;
esac

MANIFEST_DIR="${OBJSTORE_DATA_ROOT}/${DATASET_MANIFESTS_DIR_NAME}"
TRAIN_MANIFEST_PATH="${MANIFEST_DIR}/${TRAIN_MANIFEST_FILE}"
VAL_MANIFEST_PATH="${MANIFEST_DIR}/${VAL_MANIFEST_FILE}"

if [[ ! -f "$TRAIN_MANIFEST_PATH" ]]; then
  echo "Train manifest not found: ${TRAIN_MANIFEST_PATH}"
  exit 1
fi
if [[ ! -f "$VAL_MANIFEST_PATH" ]]; then
  echo "Val manifest not found: ${VAL_MANIFEST_PATH}"
  exit 1
fi

OBJSTORE_DATA_ROOT="$OBJSTORE_DATA_ROOT" \
TRAIN_MANIFEST_PATH="$TRAIN_MANIFEST_PATH" \
VAL_MANIFEST_PATH="$VAL_MANIFEST_PATH" \
TEACHER_EMBEDDINGS_DIR_NAME="$TEACHER_EMBEDDINGS_DIR_NAME" \
EMBEDDINGS_SUBDIR="$EMBEDDINGS_SUBDIR" \
STRICT_CHECKS="$STRICT_CHECKS" \
python3 - <<'PY'
import csv
import os
from pathlib import Path

data_root = Path(os.environ["OBJSTORE_DATA_ROOT"]).resolve()
train_csv = Path(os.environ["TRAIN_MANIFEST_PATH"]).resolve()
val_csv = Path(os.environ["VAL_MANIFEST_PATH"]).resolve()
emb_dir_name = os.environ["TEACHER_EMBEDDINGS_DIR_NAME"]
emb_subdir = os.environ.get("EMBEDDINGS_SUBDIR", "")
strict = os.environ.get("STRICT_CHECKS", "1").lower() in {"1", "true", "yes"}

emb_root = data_root / emb_dir_name
if emb_subdir:
    emb_root = emb_root / emb_subdir


def s3_uri_to_local(uri: str) -> Path:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    parts = uri.split("/", 3)
    if len(parts) < 4:
        raise ValueError(f"Invalid s3 URI: {uri}")
    key = parts[3]
    return data_root / key


def parse_manifest(path: Path):
    missing = {"jpg": 0, "ann": 0, "npy": 0}
    n_rows = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_uri", "annotation_uri"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{path} must contain columns: image_uri, annotation_uri")

        for row in reader:
            image_uri = (row.get("image_uri") or "").strip()
            ann_uri = (row.get("annotation_uri") or "").strip()
            if not image_uri or not ann_uri:
                continue
            n_rows += 1

            jpg = s3_uri_to_local(image_uri)
            ann = s3_uri_to_local(ann_uri)
            npy = emb_root / f"{jpg.stem}.npy"

            if not jpg.exists():
                missing["jpg"] += 1
            if not ann.exists():
                missing["ann"] += 1
            if not npy.exists():
                missing["npy"] += 1

    return n_rows, missing


train_rows, train_missing = parse_manifest(train_csv)
val_rows, val_missing = parse_manifest(val_csv)

missing_total = {
    "jpg": train_missing["jpg"] + val_missing["jpg"],
    "ann": train_missing["ann"] + val_missing["ann"],
    "npy": train_missing["npy"] + val_missing["npy"],
}

print(f"Validated manifests: train_rows={train_rows} val_rows={val_rows} test_policy=val")
print(f"Missing: jpg={missing_total['jpg']} ann={missing_total['ann']} npy={missing_total['npy']}")

if strict and any(missing_total.values()):
    raise SystemExit("Strict checks enabled and missing files were found.")
PY

EMB_PATH="${OBJSTORE_DATA_ROOT}/${TEACHER_EMBEDDINGS_DIR_NAME}"
if [[ -n "$EMBEDDINGS_SUBDIR" ]]; then
  EMB_PATH="${EMB_PATH}/${EMBEDDINGS_SUBDIR}"
fi

echo "Done. CSV-manifest setup ready."
echo "Use these YAML values (no split_manifest needed):"
echo "  data.data_dir: ${OBJSTORE_DATA_ROOT}/${IMAGES_DIR_NAME}"
echo "  data.annotation_root: ${OBJSTORE_DATA_ROOT}/${ANNOTATIONS_DIR_NAME}"
echo "  data.objstore_local_root: ${OBJSTORE_DATA_ROOT}"
echo "  data.train_manifest_csv: ${TRAIN_MANIFEST_PATH}"
echo "  data.val_manifest_csv: ${VAL_MANIFEST_PATH}"
echo "  data.embeddings_dir: ${EMB_PATH}"
echo "Note: synthetic-data manifest is intentionally ignored."
