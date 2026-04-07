# serving/system/fastapi_app/Dockerfile
FROM nvcr.io/nvidia/tritonserver:24.05-py3

RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.30.1 \
    numpy==1.26.4 \
    Pillow==10.3.0 \
    pydantic==2.7.1 \
    onnxruntime-gpu==1.18.1

# context is the parent dir
COPY system/fastapi_app/app.py /app/app.py

WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
