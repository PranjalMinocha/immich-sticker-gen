"""
MobileSAM FastAPI inference endpoint — NVIDIA GPU (onnxruntime-gpu).

POST /predict
  Input:  { "image": "<base64 jpg>",
            "box": [x1,y1,x2,y2] }   OR   { "point": [x, y] }
  Output: { "mask": "<base64 png>",
            "inference_ms": float, "encoder_ms": float, "decoder_ms": float }
"""
from __future__ import annotations

import base64, io, os, time, warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

MODEL_DIR    = os.environ.get("MODEL_DIR", "/data")
ENCODER_ONNX = os.path.join(MODEL_DIR, "mobile_sam_encoder.onnx")
DECODER_ONNX = os.path.join(MODEL_DIR, "mobile_sam_decoder.onnx")

PIXEL_MEAN = np.array([123.675, 116.28,  103.53], dtype=np.float32)
PIXEL_STD  = np.array([ 58.395,  57.12,  57.375], dtype=np.float32)

PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

enc_sess = ort.InferenceSession(ENCODER_ONNX, providers=PROVIDERS)
dec_sess = ort.InferenceSession(DECODER_ONNX, providers=PROVIDERS)

app = FastAPI(title="MobileSAM API (CUDA)")


class PredictRequest(BaseModel):
    image: str
    box:   list[float] | None = None
    point: list[float] | None = None


class PredictResponse(BaseModel):
    mask:         str
    inference_ms: float
    encoder_ms:   float
    decoder_ms:   float


def _preprocess(image_rgb: np.ndarray, size: int = 1024) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded  = np.pad(resized,
                     ((0, size - new_h), (0, size - new_w), (0, 0)),
                     mode="constant").astype(np.float32)
    return ((padded - PIXEL_MEAN) / PIXEL_STD).transpose(2, 0, 1)[None]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t_total = time.perf_counter()

    image = np.array(Image.open(io.BytesIO(base64.b64decode(req.image))).convert("RGB"))
    orig_h, orig_w = image.shape[:2]

    t0 = time.perf_counter()
    (embedding,) = enc_sess.run(["image_embeddings"], {"image": _preprocess(image)})
    encoder_ms = (time.perf_counter() - t0) * 1e3

    if req.box:
        x1, y1, x2, y2 = req.box
        point_coords = np.array([[[x1,y1],[x2,y2],[0,0],[0,0],[0,0]]], dtype=np.float32)
        point_labels = np.array([[2, 3, -1, -1, -1]],                  dtype=np.float32)
    elif req.point:
        px, py = req.point
        point_coords = np.array([[[px,py],[0,0],[0,0],[0,0],[0,0]]], dtype=np.float32)
        point_labels = np.array([[1, -1, -1, -1, -1]],               dtype=np.float32)
    else:
        point_coords = np.array([[[orig_w/2,orig_h/2],[0,0],[0,0],[0,0],[0,0]]], dtype=np.float32)
        point_labels = np.array([[1, -1, -1, -1, -1]],                           dtype=np.float32)

    t0 = time.perf_counter()
    masks, _, _ = dec_sess.run(
        ["masks", "iou_predictions", "low_res_masks"],
        {
            "image_embeddings": embedding,
            "point_coords":     point_coords,
            "point_labels":     point_labels,
            "mask_input":       np.zeros((1,1,256,256), dtype=np.float32),
            "has_mask_input":   np.array([0],           dtype=np.float32),
            "orig_im_size":     np.array([orig_h, orig_w], dtype=np.float32),
        },
    )
    decoder_ms = (time.perf_counter() - t0) * 1e3

    buf = io.BytesIO()
    Image.fromarray((masks[0,0] > 0).astype(np.uint8) * 255).save(buf, format="PNG")

    return PredictResponse(
        mask         = base64.b64encode(buf.getvalue()).decode(),
        inference_ms = (time.perf_counter() - t_total) * 1e3,
        encoder_ms   = encoder_ms,
        decoder_ms   = decoder_ms,
    )


@app.get("/health")
def health():
    return {"status": "ok"}