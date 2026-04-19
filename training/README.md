# Training (Phase 2)

This is the training runner used by Immich retraining jobs.

## What it does

- Trains MobileSAM with Ray (`train.py`).
- Logs runs/metrics/artifacts to MLflow.
- Computes offline gates (`dice`, `iou`, `runtimeSeconds`).
- Registers model in MLflow Model Registry only when gates pass.
- Writes machine-readable result JSON for Immich.

## Prerequisites

- Python 3.10+
- NVIDIA instance (Quadro RTX 6000 supported) or AMD ROCm
- MobileSAM checkout (set `MOBILESAM_ROOT`)
- MLflow server: `http://129.114.27.60:8000`

Install deps:

```bash
cd ~/immich-sticker-gen
pip install -r training/requirements.txt
```

NVIDIA PyTorch (example):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Data setup (new mounted layout)

Expected mounted object store layout:

- `images/`
- `annotations/`
- `dataset_manifests/train_manifest.csv`
- `dataset_manifests/val_manifest.csv`
- `Teacher-Embeddings/<subdir>/*.npy`

Run setup to generate `split_manifest.json` from train/val CSVs:

```bash
cd ~/immich-sticker-gen
OBJSTORE_DATA_ROOT=/path/to/mounted/bucket/root \
EMBEDDINGS_SUBDIR=sa_000000 \
bash training/setup_host.sh
```

Notes:

- Third synthetic manifest is intentionally ignored.
- `test` split is set to `val` in generated split manifest.
- Default strict mode fails if any image/annotation/embedding file is missing.

## Recommended configs

- NVIDIA bare metal: `training/configs/chameleon_nvidia.yaml`
- ROCm/container: `training/configs/chameleon_docker.yaml`

After running `setup_host.sh`, update config paths using printed values:

- `data.data_dir`
- `data.annotation_root`
- `data.embeddings_dir`
- `data.split_manifest`

## Run training manually

```bash
cd ~/immich-sticker-gen/training
export MOBILESAM_ROOT=~/MobileSAM-pytorch/MobileSAM
export MLFLOW_TRACKING_URI=http://129.114.27.60:8000

python3 train.py --config /path/to/run.yaml --num-workers 1
```

Immich-compatible invocation (same script, adds orchestration args):

```bash
python3 train.py \
  --config /path/to/run.yaml \
  --sample-window-size 5000 \
  --run-id <immich-run-id> \
  --output-json /tmp/sticker_training_result.json
```

## Validate quickly

```bash
cd ~/immich-sticker-gen/training
python3 -m compileall offline_eval.py train.py sam_utils.py
pytest -q test_offline_eval.py
```

Smoke run:

```bash
python3 train.py \
  --config /path/to/run.yaml \
  --num-workers 1 \
  --run-id smoke-phase2 \
  --sample-window-size 5000 \
  --output-json /tmp/sticker_training_result.json
```

Verify:

- JSON has `metrics.dice`, `metrics.iou`, `metrics.runtimeSeconds`, `qualityGate.*`
- MLflow run appears at `http://129.114.27.60:8000`
- Model version is registered only when gate passes

## Immich integration fields (server config)

Set `machineLearning.stickerTraining`:

- `enabled: true`
- `retrainThreshold: 5000`
- `sampleWindowSize: 5000`
- `pythonExecutable`
- `trainingScriptPath` (path to `train.py`)
- `trainingConfigPath` (path to YAML)
- `trainingWorkingDirectory`
- `resultJsonPath`
- `qualityGate.minDiceScore`
- `qualityGate.minIouScore`
- `qualityGate.maxRuntimeSeconds`
