FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.30.1 \
    onnxruntime==1.18.0 \
    numpy==1.26.4 \
    opencv-python-headless==4.9.0.80 \
    Pillow==10.3.0 \
    pydantic==2.7.1

COPY system/fastapi_app/app.py /app/app.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
