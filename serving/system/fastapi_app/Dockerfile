FROM quay.io/jupyter/scipy-notebook:python-3.12

ARG ROCM_VERSION=7.0.1
ARG AMDGPU_VERSION=7.0.1.70001

USER root

RUN wget https://repo.radeon.com/amdgpu-install/${ROCM_VERSION}/ubuntu/noble/amdgpu-install_${AMDGPU_VERSION}-1_all.deb && \
    apt-get update --yes && \
    apt-get install --yes ./amdgpu-install_${AMDGPU_VERSION}-1_all.deb && \
    amdgpu-install --usecase=rocm -y && \
    rm *.deb && \
    apt-get install --yes libgl1 libglib2.0-0

ENV PATH="/opt/venv/bin:/opt/rocm/bin:${PATH}"

RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.30.1 \
    numpy==1.26.4 \
    opencv-python-headless==4.9.0.80 \
    Pillow==10.3.0 \
    pydantic==2.7.1 \
    onnx==1.18.0 \
    coloredlogs flatbuffers packaging protobuf sympy

RUN pip install --no-cache-dir --no-index \
    --find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0 \
    onnxruntime-rocm

# context is the parent dir
COPY system/fastapi_app/app.py /app/app.py

WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
