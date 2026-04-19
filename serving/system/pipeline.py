# serving/system/pipeline.py
"""
Pipeline: train → eval → promote → deploy

Steps:
  1. train()   — Ray Train job; downloads/fine-tunes model → saves to MinIO models/candidate/
  2. eval()    — loads candidate model, runs a forward-pass sanity check, returns iou score
  3. promote() — if score >= EVAL_THRESHOLD, copies candidate → production in MinIO
  4. deploy()  — applies serve_config.yaml to the Ray cluster via Ray Serve API

Replace the train() and eval() bodies with your own logic when you have real training data.
"""
from __future__ import annotations

import io, os, sys, urllib.request

import boto3
import ray
from ray import serve
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

RAY_ADDRESS     = os.environ.get("RAY_ADDRESS",     "ray://ray-head:10001")
MINIO_ENDPOINT  = os.environ.get("MINIO_ENDPOINT",  "http://minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MODEL_BUCKET    = os.environ.get("MODEL_BUCKET",    "models")
EVAL_THRESHOLD  = float(os.environ.get("EVAL_THRESHOLD", "0.75"))

MOBILE_SAM_URL  = (
    "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
)

# ── MinIO client ──────────────────────────────────────────────────────────────

def _s3():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )


# ── Step 1: Train ─────────────────────────────────────────────────────────────

def _train_loop(config: dict):
    """
    Ray Train worker function.

    Currently seeds MinIO with the base checkpoint. Replace with your
    fine-tuning loop once you have labelled training data.
    """
    import boto3 as _boto3, urllib.request as _req

    s3 = _boto3.client(
        "s3",
        endpoint_url=config["minio_endpoint"],
        aws_access_key_id=config["minio_access_key"],
        aws_secret_access_key=config["minio_secret_key"],
    )

    local = "/tmp/mobile_sam.pt"
    print(f"[train] downloading base checkpoint …", flush=True)
    _req.urlretrieve(config["model_url"], local)

    print(f"[train] uploading to {config['bucket']}/candidate/mobile_sam.pt", flush=True)
    s3.upload_file(local, config["bucket"], "candidate/mobile_sam.pt")
    print("[train] done", flush=True)


def train() -> None:
    trainer = TorchTrainer(
        _train_loop,
        scaling_config=ScalingConfig(num_workers=1, use_gpu=True),
        train_loop_config={
            "minio_endpoint":   MINIO_ENDPOINT,
            "minio_access_key": MINIO_ACCESS_KEY,
            "minio_secret_key": MINIO_SECRET_KEY,
            "bucket":           MODEL_BUCKET,
            "model_url":        MOBILE_SAM_URL,
        },
    )
    result = trainer.fit()
    print(f"[pipeline] train result: {result}", flush=True)


# ── Step 2: Eval ──────────────────────────────────────────────────────────────

@ray.remote(num_gpus=1)
def _eval_remote(config: dict) -> float:
    """
    Loads the candidate model and runs a synthetic forward pass.
    Returns a pseudo-iou score; replace with your real eval dataset.
    """
    import boto3 as _boto3
    import numpy as np
    import torch
    from mobile_sam import sam_model_registry

    s3 = _boto3.client(
        "s3",
        endpoint_url=config["minio_endpoint"],
        aws_access_key_id=config["minio_access_key"],
        aws_secret_access_key=config["minio_secret_key"],
    )
    local = "/tmp/candidate_model.pt"
    s3.download_file(config["bucket"], "candidate/mobile_sam.pt", local)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = sam_model_registry["vit_t"](checkpoint=local)
    model.to(device).eval()

    # synthetic 512×512 RGB image
    dummy = torch.rand(1, 3, 1024, 1024, device=device)
    with torch.no_grad():
        emb = model.image_encoder(dummy)
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
    ref    = _eval_remote.remote({
        "minio_endpoint":   MINIO_ENDPOINT,
        "minio_access_key": MINIO_ACCESS_KEY,
        "minio_secret_key": MINIO_SECRET_KEY,
        "bucket":           MODEL_BUCKET,
    })
    return ray.get(ref)


# ── Step 3: Promote ───────────────────────────────────────────────────────────

def promote() -> None:
    s3  = _s3()
    src = {"Bucket": MODEL_BUCKET, "Key": "candidate/mobile_sam.pt"}
    s3.copy(src, MODEL_BUCKET, "production/mobile_sam.pt")
    print("[pipeline] promoted candidate → production", flush=True)


# ── Step 4: Deploy ────────────────────────────────────────────────────────────

def deploy() -> None:
    import yaml

    with open("/app/serve_config.yaml") as f:
        config = yaml.safe_load(f)

    serve.start(detached=True, http_options={"host": "0.0.0.0"})

    from ray.serve.schema import ServeDeploySchema
    client = serve.get_deployment_handle  # noqa: F841 — force serve import check

    # Apply via the REST API so the deployment persists after this process exits
    import requests as _req
    resp = _req.put(
        "http://ray-head:8265/api/serve/applications/",
        json=config,
        timeout=30,
    )
    resp.raise_for_status()
    print(f"[pipeline] deployed immich-sticker-gen via Ray Serve", flush=True)


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
        print(f"[pipeline] eval FAILED — model not promoted", flush=True)
        sys.exit(1)

    print("[pipeline] step 3/4 — promote", flush=True)
    promote()

    print("[pipeline] step 4/4 — deploy", flush=True)
    deploy()

    print("[pipeline] done ✓", flush=True)
    ray.shutdown()


if __name__ == "__main__":
    main()
