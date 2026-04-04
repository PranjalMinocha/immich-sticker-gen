FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
 
WORKDIR /app
 
# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
 
# Python deps — pinned for reproducibility
RUN pip install --no-cache-dir \
    # MobileSAM
    git+https://github.com/ChaoningZhang/MobileSAM.git \
    # ONNX
    onnx==1.16.0 \
    onnxruntime==1.18.0 \
    onnxruntime-gpu==1.18.0 \
    onnxoptimizer==0.3.13 \
    # Quantization / optimization
    onnxmltools==1.12.0 \
    # Data / vision
    numpy==1.26.4 \
    opencv-python-headless==4.9.0.80 \
    Pillow==10.3.0 \
    pycocotools==2.0.7 \
    requests==2.31.0 \
    timm==0.9.16
 
# Copy benchmark scripts
COPY download_data.py /app/download_data.py
COPY benchmark.py    /app/benchmark.py
 
# Data lives in a named volume mounted at /data
ENV DATA_DIR=/data
ENV MODEL_DIR=/data
 