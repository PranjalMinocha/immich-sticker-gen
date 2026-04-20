# serving/system/fastapi_app/app.py
"""
MobileSAM Ray Serve endpoint (PyTorch backend)

Model is pulled from MinIO at replica startup:
  MINIO_ENDPOINT / MODEL_BUCKET / MODEL_KEY  →  loaded into GPU

POST /predict
  Input:  { "image": "<base64>", "bbox": [x, y, w, h] }
       OR { "image": "<base64>", "point_coords": [[x, y]] }
  Output: { "mask": "<base64 png>", "inference_ms": float,
            "encoder_ms": float, "decoder_ms": float, "iou_score": float }
"""
from __future__ import annotations

import base64, io, os, time

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from ray import serve

MINIO_ENDPOINT   = os.environ.get("MINIO_ENDPOINT",   "http://minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MODEL_BUCKET     = os.environ.get("MODEL_BUCKET",     "models")
MODEL_KEY        = os.environ.get("MODEL_KEY",        "production/mobile_sam.pt")
LOCAL_CKPT       = "/tmp/mobile_sam.pt"

PIXEL_MEAN = np.array([123.675, 116.28,  103.53], dtype=np.float32)
PIXEL_STD  = np.array([ 58.395,  57.12,  57.375], dtype=np.float32)

app = FastAPI(title="MobileSAM API")


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


@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 1})
@serve.ingress(app)
class StickerGenModel:
    def __init__(self):
        import boto3
        import torch
        from mobile_sam import sam_model_registry

        boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
        ).download_file(MODEL_BUCKET, MODEL_KEY, LOCAL_CKPT)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._sam = sam_model_registry["vit_t"](checkpoint=LOCAL_CKPT)
        self._sam.to(self._device).eval()
        print(f"[StickerGenModel] ready on {self._device}", flush=True)

    @app.post("/predict", response_model=PredictResponse)
    def predict(self, req: PredictRequest):
        import torch

        if not req.bbox and not req.point_coords:
            raise HTTPException(status_code=422, detail="Provide 'bbox' or 'point_coords'")

        t_total = time.perf_counter()
        image   = np.array(Image.open(io.BytesIO(base64.b64decode(req.image))).convert("RGB"))
        orig_h, orig_w = image.shape[:2]
        scale   = 1024 / max(orig_h, orig_w)
        new_h   = int(orig_h * scale + 0.5)
        new_w   = int(orig_w * scale + 0.5)

        t0 = time.perf_counter()
        img_t = torch.from_numpy(_preprocess(image)).to(self._device)
        with torch.no_grad():
            embedding = self._sam.image_encoder(img_t)
        encoder_ms = (time.perf_counter() - t0) * 1e3

        if req.bbox:
            x, y, w, h = [c * scale for c in req.bbox]
            coords = torch.tensor([[[x, y], [x + w, y + h]]], dtype=torch.float32, device=self._device)
            labels = torch.tensor([[2, 3]],                   dtype=torch.int,     device=self._device)
        else:
            px, py = req.point_coords[0][0] * scale, req.point_coords[0][1] * scale
            coords = torch.tensor([[[px, py]]],               dtype=torch.float32, device=self._device)
            labels = torch.tensor([[1]],                      dtype=torch.int,     device=self._device)

        t0 = time.perf_counter()
        with torch.no_grad():
            sparse, dense = self._sam.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
            masks, iou = self._sam.mask_decoder(
                image_embeddings=embedding,
                image_pe=self._sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            masks = self._sam.postprocess_masks(masks, input_size=(new_h, new_w), original_size=(orig_h, orig_w))
        decoder_ms = (time.perf_counter() - t0) * 1e3

        mask = (masks[0, 0] > self._sam.mask_threshold).cpu().numpy()
        buf  = io.BytesIO()
        Image.fromarray(mask.astype(np.uint8) * 255).save(buf, format="PNG")

        return PredictResponse(
            mask         = base64.b64encode(buf.getvalue()).decode(),
            inference_ms = (time.perf_counter() - t_total) * 1e3,
            encoder_ms   = encoder_ms,
            decoder_ms   = decoder_ms,
            iou_score    = float(iou[0, 0]),
        )

    @app.get("/health")
    def health(self):
        return {"status": "ok"}


sticker_gen_app = StickerGenModel.bind()
