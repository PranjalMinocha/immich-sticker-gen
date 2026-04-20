# Retraining Testing (Training Team)

This document covers training-owned validation for retraining runs that are triggered by the data pipeline.

## What this validates

- Training job can consume run-scoped manifests under `retraining_runs/<run_id>/dataset_manifests/`.
- MLflow run is created with required metrics and checkpoint artifact.
- Model registry gets a new version when quality gates pass.
- Quality-gate-fail runs do not register a model.
- Model version tags include test metrics (`test_dice`, `test_iou`, `runtime_seconds`).
- Post-retrain gates include prompt robustness and hard subsets (small objects, low light).

## Pretrained source policy

- First retrain: use MLflow registry/bootstrapped source when no serving checkpoint exists in object storage.
- Subsequent retrains: use object-storage checkpoint URI when present (`PRETRAINED_MODEL_S3_URI` or `s3://{SERVING_MODEL_BUCKET}/{SERVING_MODEL_KEY}`).
- Fallback when object checkpoint is missing: registry alias `Production` -> latest registry version -> `BOOTSTRAP_MODEL_URI`.

The trigger resolves this source and passes it into the retraining command as `{pretrained_model_uri}`.

## Preconditions

- Training image built: `immich-sticker-train:nvidia`.
- Object-store root mounted in a Docker-visible path (recommended: `/mnt/objstore`, see `ops/systemd/README.md`).
- MobileSAM repo available on host (`-v <path>:/mobilesam:ro`).
- MLflow tracking URI reachable from the training container.
- Data team has already prepared run artifacts:
  - `retraining_runs/<run_id>/dataset_manifests/train_manifest.csv`
  - `retraining_runs/<run_id>/dataset_manifests/val_manifest.csv`
  - referenced image/annotation objects.
- Hard-subset manifests exist (or are generated) at:
  - `/data/eval_subsets/small_objects_manifest.csv`
  - `/data/eval_subsets/low_light_manifest.csv`

## Build hard subsets (20 each)

```bash
cd ~/immich-sticker-gen/training
python3 build_eval_subsets.py \
  --val-manifest /mnt/objstore/retraining_runs/<run_id>/dataset_manifests/val_manifest.csv \
  --output-dir /mnt/objstore/eval_subsets \
  --objstore-local-root /mnt/objstore \
  --data-dir /mnt/objstore \
  --small-count 20 \
  --low-light-count 20
```

## Happy-path smoke test

Set environment variables and run:

```bash
cd ~/immich-sticker-gen/training
OBJSTORE_DATA_ROOT=/tmp/rclone-tests/object \
OUT_DIR=/home/cc/training_out \
MOBILESAM_ROOT=/home/cc/MobileSAM-pytorch/MobileSAM \
MLFLOW_TRACKING_URI=http://129.114.27.60:8000 \
RETRAIN_RUN_ID=<existing-run-id> \
bash run_retraining_smoke_docker.sh
```

This script:

1. Renders run-specific config from `configs/retraining_low_lr.yaml`.
2. Runs `train.py` in docker.
3. Verifies output JSON + MLflow run + registry version linkage.

## Quality-gate-fail test

```bash
cd ~/immich-sticker-gen/training
OBJSTORE_DATA_ROOT=/tmp/rclone-tests/object \
OUT_DIR=/home/cc/training_out \
MOBILESAM_ROOT=/home/cc/MobileSAM-pytorch/MobileSAM \
MLFLOW_TRACKING_URI=http://129.114.27.60:8000 \
RETRAIN_RUN_ID=<existing-run-id> \
bash run_retraining_quality_gate_fail_docker.sh
```

This script forces strict thresholds and verifies:

- result JSON status is `failed`
- MLflow run exists
- model registry is not updated.

## Manual verifier

You can verify any produced result file with:

```bash
docker run --rm \
  -v /home/cc/training_out:/out \
  -v /home/cc/immich-sticker-gen/training:/work \
  immich-sticker-train:nvidia \
  python3 /work/verify_retraining_result.py \
    --result-json /out/<run>_result.json \
    --check-mlflow --check-registry --require-passed --require-registered
```

## Evidence to capture for reports

- Training command used (including run id).
- Result JSON file path and contents.
- MLflow run URL/id.
- Registered model name/version URL.
- Pass/fail for smoke and fail-path tests.
