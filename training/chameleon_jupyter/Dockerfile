# ROCm PyTorch + JupyterLab + MobileSAM-pytorch (recursive clone) for Chameleon GPU hosts.
# Build: docker compose build (from this directory on the train host).
# Graded training runs should use training/train_encoder.py in a container; this image is for interactive work.

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

RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

RUN pip3 install \
    opencv-python-headless \
    pycocotools \
    matplotlib \
    onnxruntime \
    onnx \
    timm \
    jupyterlab

WORKDIR /opt
ARG MOBILESAM_PYTORCH_REPO=https://github.com/ChaoningZhang/MobileSAM-pytorch.git
RUN git clone --recursive "${MOBILESAM_PYTORCH_REPO}" MobileSAM-pytorch

WORKDIR /opt/MobileSAM-pytorch
RUN pip install -e ./MobileSAM

WORKDIR /workspace
ENV PYTHONPATH=/opt/MobileSAM-pytorch/MobileSAM

EXPOSE 8888

CMD ["sh", "-c", \
     "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
      --ServerApp.token=${JUPYTER_TOKEN} --ServerApp.allow_origin='*'"]
