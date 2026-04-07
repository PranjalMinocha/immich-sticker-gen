"""
MobileSAM FastAPI inference endpoint

BACKEND=onnx (default)      onnxruntime-gpu with encoder + decoder ONNX
BACKEND=pytorch             PyTorch, direct encoder/decoder calls (thread-safe)

POST /predict
  Input:  { "image": "<base64>", "bbox": [x, y, w, h] }
       OR { "image": "<base64>", "point_coords": [[x, y]] }
  Output: { "mask": "<base64 png>", "inference_ms": float, "encoder_ms": float, "decoder_ms": float }
"""
from __future__ import annotations

import base64, io, os, time

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

BACKEND   = os.environ.get("BACKEND",   "onnx")
MODEL_DIR = os.environ.get("MODEL_DIR", "/data")
CKPT_PATH = os.environ.get("CKPT_PATH", os.path.join(MODEL_DIR, "mobile_sam.pt"))

PIXEL_MEAN = np.array([123.675, 116.28,  103.53], dtype=np.float32)
PIXEL_STD  = np.array([ 58.395,  57.12,  57.375], dtype=np.float32)

if BACKEND == "pytorch":
    import torch
    from mobile_sam import sam_model_registry
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[pytorch backend] device={_device}, torch={torch.__version__}", flush=True)
    _sam = sam_model_registry["vit_t"](checkpoint=CKPT_PATH)
    _sam.to(_device).eval()
else:
    import onnxruntime as ort
    _PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    _enc_sess  = ort.InferenceSession(
        os.path.join(MODEL_DIR, "mobile_sam_encoder.onnx"), providers=_PROVIDERS)
    _dec_sess  = ort.InferenceSession(
        os.path.join(MODEL_DIR, "mobile_sam_decoder.onnx"), providers=_PROVIDERS)

app = FastAPI(title=f"MobileSAM API ({BACKEND})")


class PredictRequest(BaseModel):
    image:        str
    bbox:         list[float]       | None = None  # [x, y, w, h]
    point_coords: list[list[float]] | None = None  # [[x, y]]


class PredictResponse(BaseModel):
    mask:         str
    inference_ms: float
    encoder_ms:   float
    decoder_ms:   float
    iou_score:    float | None = None


def _preprocess(image_rgb: np.ndarray, size: int = 1024) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
    resized = np.array(Image.fromarray(image_rgb).resize((new_w, new_h), Image.BILINEAR), dtype=np.float32)
    padded = np.zeros((size, size, 3), dtype=np.float32)
    padded[:new_h, :new_w] = resized
    return ((padded - PIXEL_MEAN) / PIXEL_STD).transpose(2, 0, 1)[None]


def _infer_pytorch(image: np.ndarray, req: PredictRequest):
    orig_h, orig_w = image.shape[:2]
    t0 = time.perf_counter()
    image_tensor = torch.from_numpy(_preprocess(image)).to(_device)
    with torch.no_grad():
        embedding = _sam.image_encoder(image_tensor)
    encoder_ms = (time.perf_counter() - t0) * 1e3

    scale = 1024 / max(orig_h, orig_w)
    if req.bbox:
        x, y, w, h = [c * scale for c in req.bbox]
        coords = torch.tensor([[[x, y], [x + w, y + h]]], dtype=torch.float32, device=_device)
        labels = torch.tensor([[2, 3]],                   dtype=torch.int,     device=_device)
    else:
        px, py = req.point_coords[0][0] * scale, req.point_coords[0][1] * scale
        coords = torch.tensor([[[px, py]]],               dtype=torch.float32, device=_device)
        labels = torch.tensor([[1]],                      dtype=torch.int,     device=_device)

    t0 = time.perf_counter()
    with torch.no_grad():
        sparse, dense = _sam.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        masks, iou = _sam.mask_decoder(
            image_embeddings=embedding,
            image_pe=_sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
    decoder_ms = (time.perf_counter() - t0) * 1e3
    return (masks[0, 0] > _sam.mask_threshold).cpu().numpy(), encoder_ms, decoder_ms, float(iou[0, 0])


def _infer_onnx(image: np.ndarray, req: PredictRequest):
    orig_h, orig_w = image.shape[:2]
    t0 = time.perf_counter()
    (embedding,) = _enc_sess.run(["image_embeddings"], {"image": _preprocess(image)})
    encoder_ms = (time.perf_counter() - t0) * 1e3

    scale = 1024 / max(orig_h, orig_w)
    if req.bbox:
        x, y, w, h = [c * scale for c in req.bbox]
        x1, y1, x2, y2 = x, y, x + w, y + h
        point_coords = np.array([[[x1, y1], [x2, y2], [0, 0], [0, 0], [0, 0]]], dtype=np.float32)
        point_labels = np.array([[2, 3, -1, -1, -1]],                            dtype=np.float32)
    else:
        px, py = req.point_coords[0][0] * scale, req.point_coords[0][1] * scale
        point_coords = np.array([[[px, py], [0, 0], [0, 0], [0, 0], [0, 0]]], dtype=np.float32)
        point_labels = np.array([[1, -1, -1, -1, -1]],                         dtype=np.float32)

    t0 = time.perf_counter()
    masks, iou, _ = _dec_sess.run(
        ["masks", "iou_predictions", "low_res_masks"],
        {
            "image_embeddings": embedding,
            "point_coords":     point_coords,
            "point_labels":     point_labels,
            "mask_input":       np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input":   np.array([0],              dtype=np.float32),
            "orig_im_size":     np.array([orig_h, orig_w], dtype=np.float32),
        },
    )
    decoder_ms = (time.perf_counter() - t0) * 1e3
    return (masks[0, 0] > 0), encoder_ms, decoder_ms, float(iou[0, 0])


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.bbox and not req.point_coords:
        raise HTTPException(status_code=422, detail="Provide either 'bbox' or 'point_coords'")

    t_total = time.perf_counter()
    image = np.array(Image.open(io.BytesIO(base64.b64decode(req.image))).convert("RGB"))

    if BACKEND == "pytorch":
        mask, encoder_ms, decoder_ms, iou_score = _infer_pytorch(image, req)
    else:
        mask, encoder_ms, decoder_ms, iou_score = _infer_onnx(image, req)

    buf = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(buf, format="PNG")

    return PredictResponse(
        mask         = base64.b64encode(buf.getvalue()).decode(),
        inference_ms = (time.perf_counter() - t_total) * 1e3,
        encoder_ms   = encoder_ms,
        decoder_ms   = decoder_ms,
        iou_score    = iou_score,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
