"""
MobileSAM FastAPI inference endpoint.

POST /predict
  Input:  { "image": "<base64 jpg>", "box": [x1,y1,x2,y2] }
          OR { "image": "<base64 jpg>", "point": [x, y] }
  Output: { "mask": "<base64 png>", "inference_ms": float,
            "encoder_ms": float, "decoder_ms": float }
"""
from __future__ import annotations

import base64
import io
import os
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

# ── paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR    = os.environ.get("MODEL_DIR", "/data")
ENCODER_ONNX = os.path.join(MODEL_DIR, "mobile_sam_encoder.onnx")
DECODER_ONNX = os.path.join(MODEL_DIR, "mobile_sam_decoder.onnx")

# ── preprocessing ─────────────────────────────────────────────────────────────
PIXEL_MEAN = np.array([123.675, 116.28,  103.53 ], dtype=np.float32)
PIXEL_STD  = np.array([ 58.395,  57.12,   57.375], dtype=np.float32)

def preprocess(image_rgb: np.ndarray, size: int = 1024) -> np.ndarray:
    h, w   = image_rgb.shape[:2]
    scale  = size / max(h, w)
    new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded  = np.pad(resized,
                     ((0, size - new_h), (0, size - new_w), (0, 0)),
                     mode="constant").astype(np.float32)
    padded  = (padded - PIXEL_MEAN) / PIXEL_STD
    return padded.transpose(2, 0, 1)[None]          # [1,3,H,W]

# ── ONNX sessions ─────────────────────────────────────────────────────────────
_providers = ["CPUExecutionProvider"]
enc_sess = ort.InferenceSession(ENCODER_ONNX, providers=_providers)
dec_sess = ort.InferenceSession(DECODER_ONNX, providers=_providers)

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="MobileSAM API")

class PredictRequest(BaseModel):
    image: str                      # base64 JPEG
    box:   list[float] | None = None   # [x1,y1,x2,y2]
    point: list[float] | None = None   # [x, y]

class PredictResponse(BaseModel):
    mask:         str    # base64 PNG
    inference_ms: float
    encoder_ms:   float
    decoder_ms:   float

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t_total = time.perf_counter()

    # decode image
    img_bytes = base64.b64decode(req.image)
    image     = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    orig_h, orig_w = image.shape[:2]

    # encoder
    t0 = time.perf_counter()
    tensor = preprocess(image)
    (embedding,) = enc_sess.run(["image_embeddings"], {"image": tensor})
    encoder_ms = (time.perf_counter() - t0) * 1e3

    # build prompts
    if req.box:
        x1, y1, x2, y2 = req.box
        point_coords = np.array([[[x1,y1],[x2,y2],[0,0],[0,0],[0,0]]], dtype=np.float32)
        point_labels = np.array([[2, 3, -1, -1, -1]],                  dtype=np.float32)
    elif req.point:
        px, py = req.point
        point_coords = np.array([[[px,py],[0,0],[0,0],[0,0],[0,0]]], dtype=np.float32)
        point_labels = np.array([[1, -1, -1, -1, -1]],               dtype=np.float32)
    else:
        # centre point fallback
        point_coords = np.array([[[orig_w/2, orig_h/2],[0,0],[0,0],[0,0],[0,0]]], dtype=np.float32)
        point_labels = np.array([[1, -1, -1, -1, -1]],                            dtype=np.float32)

    mask_input    = np.zeros((1, 1, 256, 256), dtype=np.float32)
    has_mask      = np.array([0],              dtype=np.float32)
    orig_im_size  = np.array([orig_h, orig_w], dtype=np.float32)

    # decoder
    t0 = time.perf_counter()
    masks, _, _ = dec_sess.run(
        ["masks", "iou_predictions", "low_res_masks"],
        {
            "image_embeddings": embedding,
            "point_coords":     point_coords,
            "point_labels":     point_labels,
            "mask_input":       mask_input,
            "has_mask_input":   has_mask,
            "orig_im_size":     orig_im_size,
        },
    )
    decoder_ms = (time.perf_counter() - t0) * 1e3

    # encode mask as base64 PNG
    mask_bool = (masks[0, 0] > 0).astype(np.uint8) * 255
    buf = io.BytesIO()
    Image.fromarray(mask_bool).save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode()

    inference_ms = (time.perf_counter() - t_total) * 1e3
    return PredictResponse(
        mask         = mask_b64,
        inference_ms = inference_ms,
        encoder_ms   = encoder_ms,
        decoder_ms   = decoder_ms,
    )

@app.get("/health")
def health():
    return {"status": "ok"}
