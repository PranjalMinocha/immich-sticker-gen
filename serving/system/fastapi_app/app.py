# serving/system/fastapi_app/app.py
"""
MobileSAM FastAPI endpoint (PyTorch backend)

At startup the model checkpoint is resolved via model_source_resolver:
  1. S3 object at PRETRAINED_MODEL_S3_URI (if set and exists)
  2. MLflow registry Production alias  (MODEL_REGISTRY_NAME / MODEL_REGISTRY_ALIAS)
  3. Latest registry version
  4. BOOTSTRAP_MODEL_URI
The resolved artifact is downloaded to CKPT_PATH (/data/mobile_sam.pt by default).

POST /predict
  Input:  { "image": "<base64>", "bbox": [x, y, w, h] }
       OR { "image": "<base64>", "point_coords": [[x, y]] }
  Output: { "mask": "<base64 png>", "inference_ms": float,
            "encoder_ms": float, "decoder_ms": float, "iou_score": float }
"""
from __future__ import annotations

import base64, io, os, time
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request, Response
from PIL import Image, ImageFile
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Prometheus metrics ────────────────────────────────────────────────────────
_REQUEST_COUNT   = Counter("sticker_requests_total",   "Total predict requests",            ["status"])
_ERROR_COUNT     = Counter("sticker_errors_total",     "Total predict errors")
_INFERENCE_MS    = Histogram("sticker_inference_ms",   "Total inference latency ms",
                             buckets=[10, 25, 50, 100, 200, 500, 1000, 2000, 5000])
_ENCODER_MS      = Histogram("sticker_encoder_ms",     "Image encoder latency ms",
                             buckets=[5, 10, 25, 50, 100, 250, 500, 1000])
_DECODER_MS      = Histogram("sticker_decoder_ms",     "Mask decoder latency ms",
                             buckets=[1, 5, 10, 25, 50, 100, 250])
_IOU_SCORE       = Histogram("sticker_iou_score",      "Predicted IoU score",
                             buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])

CKPT_PATH            = os.environ.get("CKPT_PATH", "/data/mobile_sam.pt")
MODEL_RELOAD_TOKEN   = os.environ.get("MODEL_RELOAD_TOKEN", "")
MLFLOW_TRACKING_URI  = os.environ.get("MLFLOW_TRACKING_URI", "")
MODEL_REGISTRY_NAME  = os.environ.get("MODEL_REGISTRY_NAME", "immich-sticker-mobilesam")
MODEL_REGISTRY_ALIAS = os.environ.get("MODEL_REGISTRY_ALIAS", "Production")
BOOTSTRAP_MODEL_URI  = os.environ.get(
    "BOOTSTRAP_MODEL_URI",
    "mlflow-artifacts:/2/94d1e731c6f64307908d893c9b1476dc/artifacts/checkpoints/mobile_sam_full.pt",
)
S3_ENDPOINT          = os.environ.get("S3_ENDPOINT", "")
S3_ACCESS_KEY        = os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET_KEY        = os.environ.get("S3_SECRET_KEY", "")

PIXEL_MEAN = np.array([123.675, 116.28,  103.53], dtype=np.float32)
PIXEL_STD  = np.array([ 58.395,  57.12,  57.375], dtype=np.float32)

app = FastAPI(title="MobileSAM API")

_sam = None
_device = None
_reload_lock = threading.Lock()


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


def _make_s3_client():
    import boto3
    kwargs: dict = {}
    if S3_ENDPOINT:
        kwargs["endpoint_url"] = S3_ENDPOINT
    if S3_ACCESS_KEY and S3_SECRET_KEY:
        kwargs["aws_access_key_id"] = S3_ACCESS_KEY
        kwargs["aws_secret_access_key"] = S3_SECRET_KEY
    return boto3.client("s3", **kwargs)


def _resolve_and_download_model() -> None:
    """Resolve the production model from MLflow/S3 and write it to CKPT_PATH."""
    if not MLFLOW_TRACKING_URI and not BOOTSTRAP_MODEL_URI:
        print("[server] no MLFLOW_TRACKING_URI or BOOTSTRAP_MODEL_URI set — using existing CKPT_PATH", flush=True)
        return

    from fastapi_app.model_source_resolver import resolve_pretrained_model_source

    s3 = _make_s3_client() if S3_ENDPOINT else None
    pretrained_s3_uri = os.environ.get("PRETRAINED_MODEL_S3_URI", "")

    try:
        resolved = resolve_pretrained_model_source(
            tracking_uri=MLFLOW_TRACKING_URI,
            model_name=MODEL_REGISTRY_NAME,
            preferred_alias=MODEL_REGISTRY_ALIAS,
            bootstrap_model_uri=BOOTSTRAP_MODEL_URI or None,
            object_store_model_uri=pretrained_s3_uri or None,
            object_store_client=s3,
        )
    except Exception as exc:
        print(f"[server] model resolution failed ({exc}), using existing CKPT_PATH", flush=True)
        return

    print(f"[server] resolved model via strategy={resolved.strategy} uri={resolved.source_uri} version={resolved.model_version}", flush=True)

    Path(CKPT_PATH).parent.mkdir(parents=True, exist_ok=True)

    if resolved.source_uri.startswith("s3://"):
        stripped = resolved.source_uri[5:]
        bucket, _, key = stripped.partition("/")
        s3_client = s3 or _make_s3_client()
        s3_client.download_file(bucket, key, CKPT_PATH)
    elif resolved.source_uri.startswith("models:/") or resolved.source_uri.startswith("runs:/") or resolved.source_uri.startswith("mlflow-artifacts:/"):
        import mlflow.artifacts as mlflow_artifacts
        tmp_dir = str(Path(CKPT_PATH).parent / "_resolve_tmp")
        downloaded = mlflow_artifacts.download_artifacts(
            artifact_uri=resolved.source_uri,
            tracking_uri=MLFLOW_TRACKING_URI,
            dst_path=tmp_dir,
        )
        downloaded_path = Path(downloaded)
        if downloaded_path.is_dir():
            candidates = list(downloaded_path.rglob("*.pt"))
            if not candidates:
                raise RuntimeError(f"No .pt file found in downloaded artifact dir: {downloaded_path}")
            downloaded_path = candidates[0]
        import shutil
        shutil.move(str(downloaded_path), CKPT_PATH)
    else:
        print(f"[server] unknown source_uri scheme, skipping download: {resolved.source_uri}", flush=True)

    print(f"[server] model written to {CKPT_PATH}", flush=True)


def _load_model():
    import torch
    from mobile_sam import sam_model_registry

    global _sam, _device

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _sam = sam_model_registry["vit_t"](checkpoint=CKPT_PATH)
    _sam.to(_device).eval()
    print(f"[server] loaded {CKPT_PATH} on {_device}", flush=True)


@app.on_event("startup")
def startup():
    _resolve_and_download_model()
    _load_model()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    import torch

    if not req.bbox and not req.point_coords:
        _REQUEST_COUNT.labels(status="error").inc()
        _ERROR_COUNT.inc()
        raise HTTPException(status_code=422, detail="Provide 'bbox' or 'point_coords'")

    t_total = time.perf_counter()
    image   = np.array(Image.open(io.BytesIO(base64.b64decode(req.image))).convert("RGB"))
    orig_h, orig_w = image.shape[:2]
    scale   = 1024 / max(orig_h, orig_w)
    new_h   = int(orig_h * scale + 0.5)
    new_w   = int(orig_w * scale + 0.5)

    t0 = time.perf_counter()
    img_t = torch.from_numpy(_preprocess(image)).to(_device)
    with torch.no_grad():
        embedding = _sam.image_encoder(img_t)
    encoder_ms = (time.perf_counter() - t0) * 1e3

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
        masks = _sam.postprocess_masks(masks, input_size=(new_h, new_w), original_size=(orig_h, orig_w))
    decoder_ms = (time.perf_counter() - t0) * 1e3

    mask = (masks[0, 0] > _sam.mask_threshold).cpu().numpy()
    buf  = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(buf, format="PNG")

    total_ms   = (time.perf_counter() - t_total) * 1e3
    iou_val    = float(iou[0, 0])

    _REQUEST_COUNT.labels(status="success").inc()
    _INFERENCE_MS.observe(total_ms)
    _ENCODER_MS.observe(encoder_ms)
    _DECODER_MS.observe(decoder_ms)
    _IOU_SCORE.observe(iou_val)

    return PredictResponse(
        mask         = base64.b64encode(buf.getvalue()).decode(),
        inference_ms = total_ms,
        encoder_ms   = encoder_ms,
        decoder_ms   = decoder_ms,
        iou_score    = iou_val,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/admin/reload")
def reload_model(token: str | None = None, x_model_reload_token: str | None = Header(default=None)):
    if MODEL_RELOAD_TOKEN:
        candidate = x_model_reload_token or token or ""
        if candidate != MODEL_RELOAD_TOKEN:
            raise HTTPException(status_code=401, detail="invalid reload token")

    with _reload_lock:
        _resolve_and_download_model()
        _load_model()
    return {"status": "reloaded", "ckpt_path": CKPT_PATH}
