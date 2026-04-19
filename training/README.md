# Training (Docker Only)

Training and tests run only in containers.

## 1) Build image

From repo root:

```bash
cd ~/immich-sticker-gen
DOCKER_BUILDKIT=1 docker build -f training/Dockerfile.nvidia -t immich-sticker-train:nvidia .
```

## 2) Prepare mounted data split manifest

Expected object-store layout under your mounted root:

- `images/`
- `annotations/`
- `dataset_manifests/train_manifest.csv`
- `dataset_manifests/val_manifest.csv`
- `Teacher-Embeddings/<subdir>/*.npy`

Generate split manifest (uses only train + val CSV; ignores synthetic-data manifest):

```bash
cd ~/immich-sticker-gen
OBJSTORE_DATA_ROOT=/path/to/mounted/bucket/root \
GENERATED_SPLIT_MANIFEST=/path/to/mounted/bucket/root/splits/split_manifest.json \
EMBEDDINGS_SUBDIR=sa_000000 \
STRICT_CHECKS=1 \
bash training/setup_host.sh
```

## 3) Run offline eval unit tests in container

```bash
docker run --rm \
  --gpus all \
  immich-sticker-train:nvidia \
  pytest -q test_offline_eval.py
```

## 4) Run smoke training in container

```bash
docker run --rm \
  --gpus all \
  --shm-size=10g \
  -v /path/to/mounted/bucket/root:/data:ro \
  -v /path/to/training_out:/out \
  -v /path/to/MobileSAM-pytorch/MobileSAM:/mobilesam:ro \
  -e MOBILESAM_ROOT=/mobilesam \
  -e MLFLOW_TRACKING_URI=http://129.114.27.60:8000 \
  immich-sticker-train:nvidia \
  python3 train.py --config configs/chameleon_nvidia.yaml --num-workers 1 --run-id smoke-phase2 --sample-window-size 5000 --output-json /out/sticker_training_result.json
```

Verify:

- `/path/to/training_out/sticker_training_result.json` exists
- JSON includes `metrics.dice`, `metrics.iou`, `metrics.runtimeSeconds`, `qualityGate.*`
- MLflow run appears at `http://129.114.27.60:8000`

## 5) Immich-triggered container contract

Immich job should invoke containerized training with:

```bash
python3 train.py \
  --config <trainingConfigPath> \
  --sample-window-size 5000 \
  --run-id <immich-run-id> \
  --output-json <resultJsonPath>
```

Required Immich config (`machineLearning.stickerTraining`):

- `enabled: true`
- `retrainThreshold: 5000`
- `sampleWindowSize: 5000`
- `pythonExecutable`
- `trainingScriptPath`
- `trainingConfigPath`
- `trainingWorkingDirectory`
- `resultJsonPath`
- `qualityGate.minDiceScore`
- `qualityGate.minIouScore`
- `qualityGate.maxRuntimeSeconds`

## 6) Provided configs

- NVIDIA: `training/configs/chameleon_nvidia.yaml`
- ROCm: `training/configs/chameleon_docker.yaml`
