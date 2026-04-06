FROM nvcr.io/nvidia/pytorch:24.05-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    git+https://github.com/ChaoningZhang/MobileSAM.git \
    onnx==1.16.0 \
    onnxruntime-gpu==1.19.2 \
    tensorrt \
    openvino \
    numpy==1.26.4 \
    Pillow==10.3.0 \
    pycocotools==2.0.7 \
    requests==2.31.0 \
    timm==0.9.16

ENV NVIDIA_PRODUCT_NAME=""

COPY download_data.py /app/download_data.py
COPY benchmark.py     /app/benchmark.py

WORKDIR /app
ENTRYPOINT []
