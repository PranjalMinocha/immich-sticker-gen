# Immich Sticker Gen - Training Pipeline

## Overview
Two-stage training pipeline for MobileSAM (Segment Anything Model) to generate sticker cutouts from photos.

## Training Modes
- **encoder_distill**: Train TinyViT encoder via knowledge distillation from ViT-H teacher
- **full_sam**: Fine-tune full MobileSAM (encoder + mask decoder) from scratch

## Data
- SA-1B dataset (1.1M images)
- Data splits: train/val/test (70%/10%/20%)
- Training data: `/data/Raw-Data/extracted/` (images)
- Teacher embeddings: `/data/Teacher-Embeddings/sa_000000/`
- Instance index: `/data/Raw-Data/extracted/sam_instance_index/sam_instances_v1.json`

## Docker
```bash
# Build
DOCKER_BUILDKIT=1 docker build -f training/Dockerfile -t immich-sticker-train:rocm .

# Run (2 GPUs with Ray Train)
docker run --rm --shm-size=10.24gb --device=/dev/kfd --device=/dev/dri --group-add video \
  -v ~/training-data:/data:ro \
  -v ~/MobileSAM-pytorch/MobileSAM:/mobilesam:ro \
  -v ~/training_out:/out \
  -e MLFLOW_TRACKING_URI=http://129.114.27.60:8000 \
  -e MOBILESAM_ROOT=/mobilesam \
  immich-sticker-train:rocm \
  python3 train.py --config /out/run.yaml --num-workers 2
```

## Configuration
- Config file: `~/training_out/run.yaml`
- MLflow: `http://129.114.27.60:8000/`
- Use `sam_instance_frac` to limit training data (e.g., 0.001 = 0.1%)

## Key Files
| File | Purpose |
|------|---------|
| `training/train.py` | Main entry point (uses Ray Train) |
| `training/training_core.py` | Utilities (MLflow, GPU monitoring) |
| `training/dataset_sa1b.py` | SA-1B data loading |
| `training/sam_utils.py` | SAM model helpers |

## Notes
- Uses **Ray Train** for multi-GPU training (`--num-workers N`)
- Logs: train loss, val/test IoU, epoch time, learning rate
- System metrics logged to MLflow with `system/` prefix
- GPU util via pyrsmi (ROCm) or rocm-smi CLI fallback