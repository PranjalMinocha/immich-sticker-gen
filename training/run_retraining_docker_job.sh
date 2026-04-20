#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  run_retraining_docker_job.sh \
    --run-id <id> \
    --pretrained-model-uri <mlflow-or-model-uri> \
    --output-json <path>

Environment (required):
  OBJSTORE_DATA_ROOT   Daemon-visible object-store mount, e.g. /mnt/objstore
  MOBILESAM_ROOT       Host path to MobileSAM repo
  MLFLOW_TRACKING_URI  MLflow tracking URI

Environment (optional):
  OUT_DIR              default: /tmp/retraining_out
  TRAINING_IMAGE       default: immich-sticker-train:nvidia
  BASE_CONFIG          default: training/configs/retraining_low_lr.yaml
  FORCE_QUALITY_GATE_PASS default: 0
EOF
}

RUN_ID=""
PRETRAINED_MODEL_URI=""
OUTPUT_JSON=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"; shift 2;;
    --pretrained-model-uri)
      PRETRAINED_MODEL_URI="$2"; shift 2;;
    --output-json)
      OUTPUT_JSON="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

[[ -n "$RUN_ID" ]] || { echo "--run-id required" >&2; exit 2; }
[[ -n "$PRETRAINED_MODEL_URI" ]] || { echo "--pretrained-model-uri required" >&2; exit 2; }
[[ -n "$OUTPUT_JSON" ]] || { echo "--output-json required" >&2; exit 2; }

OBJSTORE_DATA_ROOT="${OBJSTORE_DATA_ROOT:?set OBJSTORE_DATA_ROOT}"
MOBILESAM_ROOT="${MOBILESAM_ROOT:?set MOBILESAM_ROOT}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:?set MLFLOW_TRACKING_URI}"

OUT_DIR="${OUT_DIR:-/tmp/retraining_out}"
TRAINING_IMAGE="${TRAINING_IMAGE:-immich-sticker-train:nvidia}"
BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/training/configs/retraining_low_lr.yaml}"
FORCE_QUALITY_GATE_PASS="${FORCE_QUALITY_GATE_PASS:-0}"
FORCE_QUALITY_GATE_FAIL="${FORCE_QUALITY_GATE_FAIL:-0}"

mkdir -p "$OUT_DIR"
BASE_CKPT_DIR="$OUT_DIR/base"
mkdir -p "$BASE_CKPT_DIR"
BASE_CKPT_PATH="$BASE_CKPT_DIR/${RUN_ID}_mobile_sam_full.pt"
RENDERED_CONFIG="$OUT_DIR/${RUN_ID}_config.yaml"

echo "[retrain] Downloading pretrained model: ${PRETRAINED_MODEL_URI}"
docker run --rm \
  -v "$OUT_DIR:/out" \
  -e MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
  -e PRETRAINED_MODEL_URI="$PRETRAINED_MODEL_URI" \
  -e BASE_CKPT_PATH="/out/base/${RUN_ID}_mobile_sam_full.pt" \
  "$TRAINING_IMAGE" \
  python3 -c "from mlflow import artifacts as A; import os, shutil; uri=os.environ['PRETRAINED_MODEL_URI']; tracking=os.environ['MLFLOW_TRACKING_URI']; dst=os.environ['BASE_CKPT_PATH']; tmp=A.download_artifacts(artifact_uri=uri, tracking_uri=tracking); candidate=(os.path.join(tmp, os.path.basename(uri.rstrip('/'))) if os.path.isdir(tmp) else tmp); files=[f for f in os.listdir(tmp) if os.path.isfile(os.path.join(tmp,f))] if os.path.isdir(tmp) else []; candidate=(os.path.join(tmp, files[0]) if (os.path.isdir(tmp) and not os.path.isfile(candidate) and len(files)==1) else candidate); os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(candidate, dst); print(dst)"

EXTRA_RENDER_ARGS=()
if [[ "$FORCE_QUALITY_GATE_PASS" == "1" && "$FORCE_QUALITY_GATE_FAIL" == "1" ]]; then
  echo "FORCE_QUALITY_GATE_PASS and FORCE_QUALITY_GATE_FAIL cannot both be 1" >&2
  exit 2
fi
if [[ "$FORCE_QUALITY_GATE_PASS" == "1" ]]; then
  EXTRA_RENDER_ARGS+=("--force-quality-gate-pass")
fi
if [[ "$FORCE_QUALITY_GATE_FAIL" == "1" ]]; then
  EXTRA_RENDER_ARGS+=("--force-quality-gate-fail")
fi

echo "[retrain] Rendering config"
python3 "$REPO_ROOT/training/render_retraining_config.py" \
  --base-config "$BASE_CONFIG" \
  --output-config "$RENDERED_CONFIG" \
  --run-id "$RUN_ID" \
  --data-root /data \
  --output-dir /out \
  --mobilesam-root /mobilesam \
  --pretrained-checkpoint-path "/out/base/${RUN_ID}_mobile_sam_full.pt" \
  "${EXTRA_RENDER_ARGS[@]}"

echo "[retrain] Starting training container"
docker run --rm \
  --gpus all \
  --shm-size=10g \
  -v "$OBJSTORE_DATA_ROOT:/data:ro" \
  -v "$OUT_DIR:/out" \
  -v "$MOBILESAM_ROOT:/mobilesam:ro" \
  -v "$REPO_ROOT:/work" \
  -v "$REPO_ROOT/training/dataset_sa1b.py:/app/training/dataset_sa1b.py:ro" \
  -e MOBILESAM_ROOT=/mobilesam \
  -e MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
  -e PYTHONPATH=/work/training \
  "$TRAINING_IMAGE" \
  python3 /work/training/train.py --config "/out/$(basename "$RENDERED_CONFIG")" --num-workers 1 --run-id "$RUN_ID" --output-json "$OUTPUT_JSON"

echo "[retrain] Verifying result"
python3 "$REPO_ROOT/training/verify_retraining_result.py" \
  --result-json "$OUTPUT_JSON" \
  --check-mlflow \
  --expected-orchestrator-run-id "$RUN_ID"

echo "[retrain] Completed ${RUN_ID}"
