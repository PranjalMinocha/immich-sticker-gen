#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

IMAGE="${TRAINING_IMAGE:-immich-sticker-train:nvidia}"
RUN_ID="${RETRAIN_RUN_ID:-fail-$(date -u +%Y%m%dT%H%M%SZ)}"
OBJSTORE_DATA_ROOT="${OBJSTORE_DATA_ROOT:?set OBJSTORE_DATA_ROOT to mounted object-store root}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/../training_out}"
MOBILESAM_ROOT="${MOBILESAM_ROOT:?set MOBILESAM_ROOT to MobileSAM repo path}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:?set MLFLOW_TRACKING_URI}"
BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/configs/retraining_low_lr.yaml}"
DISABLE_PRETRAINED="${DISABLE_PRETRAINED:-1}"

mkdir -p "${OUT_DIR}"
RENDERED_CONFIG="${OUT_DIR}/${RUN_ID}_config_fail.yaml"
RESULT_JSON="${OUT_DIR}/${RUN_ID}_result.json"

echo "[1/3] Rendering strict fail config for ${RUN_ID}"
EXTRA_RENDER_ARGS=()
if [[ "${DISABLE_PRETRAINED}" == "1" ]]; then
  EXTRA_RENDER_ARGS+=("--disable-pretrained")
fi
docker run --rm \
  -v "${REPO_ROOT}:/work" \
  -v "${OUT_DIR}:/out" \
  "${IMAGE}" \
  python3 /work/training/render_retraining_config.py \
    --base-config "/work/training/configs/$(basename "${BASE_CONFIG}")" \
    --output-config "/out/$(basename "${RENDERED_CONFIG}")" \
    --run-id "${RUN_ID}" \
    --force-quality-gate-fail \
    "${EXTRA_RENDER_ARGS[@]}"

echo "[2/3] Running retraining expecting quality-gate failure"
docker run --rm \
  --gpus all \
  --shm-size=10g \
  -v "${OBJSTORE_DATA_ROOT}:/data:ro" \
  -v "${OUT_DIR}:/out" \
  -v "${MOBILESAM_ROOT}:/mobilesam:ro" \
  -v "${REPO_ROOT}:/work" \
  -v "${REPO_ROOT}/training/dataset_sa1b.py:/app/training/dataset_sa1b.py:ro" \
  -e MOBILESAM_ROOT=/mobilesam \
  -e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
  -e PYTHONPATH=/work/training \
  "${IMAGE}" \
  python3 /work/training/train.py --config "/out/$(basename "${RENDERED_CONFIG}")" --num-workers 1 --run-id "${RUN_ID}" --output-json "/out/$(basename "${RESULT_JSON}")"

echo "[3/3] Verifying failure result JSON + MLflow"
docker run --rm \
  -v "${OUT_DIR}:/out" \
  -v "${REPO_ROOT}:/work" \
  -e PYTHONPATH=/work/training \
  "${IMAGE}" \
  python3 /work/training/verify_retraining_result.py \
    --result-json "/out/$(basename "${RESULT_JSON}")" \
    --require-failed \
    --require-not-registered \
    --check-mlflow \
    --expected-orchestrator-run-id "${RUN_ID}"

echo "Quality-gate-failure retraining test passed for ${RUN_ID}"
echo "Result JSON: ${RESULT_JSON}"
