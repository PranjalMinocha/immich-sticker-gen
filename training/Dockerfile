# Training image: Ubuntu + PyTorch ROCm 6.0 wheels + training deps.
# Build from repository root (BuildKit reuses layers on rebuild when training/ code or requirements unchanged):
#   DOCKER_BUILDKIT=1 docker build -f training/Dockerfile -t immich-sticker-train:rocm .
#
# Run (example; data = LOCAL_DATA_ROOT from setup_host.sh). Use --shm-size for DataLoader workers.
#   docker run --rm -it --shm-size=8g --device=/dev/kfd --device=/dev/dri --group-add video \
#     -v ~/training-data:/data:ro \
#     -v ~/training_out:/out \
#     -e MLFLOW_TRACKING_URI=http://YOUR_MLFLOW_HOST:8000 \
#     -e MOBILESAM_ROOT=/mobilesam \
#     -v ~/MobileSAM-pytorch/MobileSAM:/mobilesam:ro \
#     immich-sticker-train:rocm \
#     python3 train.py --config configs/chameleon_docker.yaml
#
# Set data.data_dir and data.embeddings_dir under /data — see DATA.md.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    git \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch with ROCm 6.0 (match host driver stack on Chameleon).
RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

WORKDIR /app
COPY training/requirements.txt /app/training/requirements.txt
RUN pip3 install -r /app/training/requirements.txt

COPY training /app/training
WORKDIR /app/training

ENV PYTHONPATH=/app/training

CMD ["python3", "train.py", "--config", "configs/tinyvit_baseline.yaml"]
