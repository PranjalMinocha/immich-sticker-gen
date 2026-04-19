# serving/system/pipeline.py
"""
Pipeline: train → eval → promote → deploy

The pipeline container is a thin orchestrator — PyTorch runs on ray-workers.
All GPU work is submitted as @ray.remote functions that execute on the cluster.

Steps:
  1. train()   — remote Ray job; downloads/fine-tunes model → saves to MinIO models/candidate/
  2. eval()    — remote Ray job; loads candidate, runs forward pass, returns iou score
  3. promote() — copies models/candidate/ → models/production/ in MinIO (runs locally)
  4. deploy()  — applies serve_config.yaml to the cluster via Ray Serve REST API

Replace _train_remote and _eval_remote bodies with your own logic when you have
real training data and a custom model.
"""
from __future__ import annotations

import os, sys

import boto3
import ray
import requests

RAY_ADDRESS      = os.environ.get("RAY_ADDRESS",      "ray://ray-head:10001")
MINIO_ENDPOINT   = os.environ.get("MINIO_ENDPOINT",   "http://minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MODEL_BUCKET     = os.environ.get("MODEL_BUCKET",     "models")
EVAL_THRESHOLD   = float(os.environ.get("EVAL_THRESHOLD", "0.75"))

MOBILE_SAM_URL = (
    "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
)

_MINIO_CFG = dict(
    minio_endpoint   = MINIO_ENDPOINT,
    minio_access_key = MINIO_ACCESS_KEY,
    minio_secret_key = MINIO_SECRET_KEY,
    bucket           = MODEL_BUCKET,
)


# ── MinIO client ──────────────────────────────────────────────────────────────

def _s3():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )


# ── Step 1: Train (runs on ray-worker — torch available there) ────────────────

@ray.remote(num_gpus=1)
def _train_remote(cfg: dict) -> None:
    """
    Downloads the base checkpoint and uploads it as the training artifact.
    Replace with your fine-tuning loop once you have labelled training data.
    """
    import urllib.request
    import boto3 as _boto3

    s3 = _boto3.client(
        "s3",
        endpoint_url=cfg["minio_endpoint"],
        aws_access_key_id=cfg["minio_access_key"],
        aws_secret_access_key=cfg["minio_secret_key"],
    )
    local = "/tmp/mobile_sam.pt"
    print(f"[train] downloading base checkpoint …", flush=True)
    urllib.request.urlretrieve(cfg["model_url"], local)

    print(f"[train] uploading to {cfg['bucket']}/candidate/mobile_sam.pt", flush=True)
    s3.upload_file(local, cfg["bucket"], "candidate/mobile_sam.pt")
    print("[train] done", flush=True)


def train() -> None:
    ray.get(_train_remote.remote({**_MINIO_CFG, "model_url": MOBILE_SAM_URL}))


# ── Step 2: Eval (runs on ray-worker — torch available there) ─────────────────

@ray.remote(num_gpus=1)
def _eval_remote(cfg: dict) -> float:
    """
    Loads the candidate model and runs a synthetic forward pass.
    Returns a pseudo-iou score; replace with your real eval dataset.
    """
    import boto3 as _boto3
    import torch
    from mobile_sam import sam_model_registry

    s3 = _boto3.client(
        "s3",
        endpoint_url=cfg["minio_endpoint"],
        aws_access_key_id=cfg["minio_access_key"],
        aws_secret_access_key=cfg["minio_secret_key"],
    )
    local = "/tmp/candidate_model.pt"
    s3.download_file(cfg["bucket"], "candidate/mobile_sam.pt", local)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = sam_model_registry["vit_t"](checkpoint=local)
    model.to(device).eval()

    dummy = torch.rand(1, 3, 1024, 1024, device=device)
    with torch.no_grad():
        emb    = model.image_encoder(dummy)
        coords = torch.tensor([[[256.0, 256.0]]], device=device)
        labels = torch.tensor([[1]], dtype=torch.int, device=device)
        sparse, dense = model.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        _, iou = model.mask_decoder(
            image_embeddings=emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
    score = float(iou[0, 0].item())
    print(f"[eval] iou_score={score:.4f}", flush=True)
    return score


def eval_model() -> float:
    return ray.get(_eval_remote.remote(_MINIO_CFG))


# ── Step 3: Promote (runs locally — just S3 copy, no GPU needed) ─────────────

def promote() -> None:
    _s3().copy(
        {"Bucket": MODEL_BUCKET, "Key": "candidate/mobile_sam.pt"},
        MODEL_BUCKET,
        "production/mobile_sam.pt",
    )
    print("[pipeline] promoted candidate → production", flush=True)


# ── Step 4: Deploy via Ray Serve REST API ─────────────────────────────────────

def deploy() -> None:
    import yaml

    with open("/app/serve_config.yaml") as f:
        config = yaml.safe_load(f)

    resp = requests.put(
        "http://ray-head:8265/api/serve/applications/",
        json=config,
        timeout=30,
    )
    resp.raise_for_status()
    print("[pipeline] deployed immich-sticker-gen via Ray Serve", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ray.init(RAY_ADDRESS)
    print("[pipeline] connected to Ray cluster", flush=True)

    print("[pipeline] step 1/4 — train", flush=True)
    train()

    print("[pipeline] step 2/4 — eval", flush=True)
    score = eval_model()
    print(f"[pipeline] eval score={score:.4f}, threshold={EVAL_THRESHOLD}", flush=True)

    if score < EVAL_THRESHOLD:
        print("[pipeline] eval FAILED — model not promoted", flush=True)
        sys.exit(1)

    print("[pipeline] step 3/4 — promote", flush=True)
    promote()

    print("[pipeline] step 4/4 — deploy", flush=True)
    deploy()

    print("[pipeline] done", flush=True)
    ray.shutdown()


if __name__ == "__main__":
    main()
