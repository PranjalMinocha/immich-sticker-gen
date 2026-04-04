#!/usr/bin/env python3
"""
Serving benchmark client for MobileSAM.

Simulates realistic request arrival patterns following the guide:
  - Serial (baseline):    one request at a time, 100 trials
  - Concurrent:           N workers sending continuously
  - Constant rate:        fixed inter-arrival time (Poisson λ → 1/λ gap)
  - Poisson arrivals:     exponentially distributed inter-arrival times

EXPERIMENT env var selects the target:
  fastapi_serial          fastapi_concurrent
  fastapi_constant_rate   fastapi_poisson
  triton_serial           triton_concurrent
  triton_constant_rate    triton_poisson

SERVER env vars:
  FASTAPI_URL   default http://fastapi_server:8000/predict
  TRITON_URL    default triton_server:8000
"""
from __future__ import annotations

import base64, concurrent.futures, json, os, time
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR     = Path(os.environ.get("DATA_DIR",   "/data"))
FASTAPI_URL  = os.environ.get("FASTAPI_URL",  "http://fastapi_server:8000/predict")
TRITON_URL   = os.environ.get("TRITON_URL",   "triton_server:8000")
TRITON_MODEL = os.environ.get("TRITON_MODEL", "mobile_sam")
EXPERIMENT   = os.environ.get("EXPERIMENT",   "fastapi_serial")

NUM_SERIAL_TRIALS  = int(os.environ.get("NUM_SERIAL_TRIALS",  "100"))
NUM_CONCURRENT     = int(os.environ.get("NUM_CONCURRENT",     "8"))
CONCURRENT_REQS    = int(os.environ.get("CONCURRENT_REQS",    "200"))
RATE_REQS_PER_SEC  = float(os.environ.get("RATE_REQS_PER_SEC", "5.0"))
RATE_DURATION_SEC  = float(os.environ.get("RATE_DURATION_SEC", "30.0"))

# ── data helpers ──────────────────────────────────────────────────────────────
def load_manifest():
    p = DATA_DIR / "manifest.json"
    if not p.exists():
        raise FileNotFoundError("manifest.json not found — run download_data.py first")
    return json.loads(p.read_text())

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def first_box(ann_path: str) -> list[float]:
    """Return bounding box [x1,y1,x2,y2] of first annotation mask."""
    import json
    from pycocotools import mask as mask_utils
    data = json.loads(Path(ann_path).read_text())
    anns = data.get("annotations", [])
    if not anns:
        return [-1, -1, -1, -1]
    seg = anns[0]["segmentation"]
    rle = mask_utils.frPyObjects(seg, seg["size"][0], seg["size"][1]) \
          if isinstance(seg["counts"], list) else seg
    m = mask_utils.decode(rle)
    if m.ndim == 3:
        m = m[..., 0]
    ys, xs = np.where(m)
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

# ── FastAPI client ─────────────────────────────────────────────────────────────
def fastapi_request(image_b64: str, box: list[float]) -> dict:
    import requests
    payload = {"image": image_b64, "box": box}
    t0 = time.perf_counter()
    resp = requests.post(FASTAPI_URL, json=payload, timeout=60)
    wall_ms = (time.perf_counter() - t0) * 1e3
    resp.raise_for_status()
    data = resp.json()
    return {
        "wall_ms":       wall_ms,
        "inference_ms":  data.get("inference_ms", wall_ms),
        "encoder_ms":    data.get("encoder_ms", 0),
        "decoder_ms":    data.get("decoder_ms", 0),
    }

# ── Triton client ──────────────────────────────────────────────────────────────
def triton_request(image_b64: str, box: list[float]) -> dict:
    from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
    client = InferenceServerClient(url=TRITON_URL)

    inp_img = InferInput("INPUT_IMAGE", [1, 1], "BYTES")
    inp_img.set_data_from_numpy(np.array([[image_b64]], dtype=object))

    inp_box = InferInput("BOX", [1, 4], "FP32")
    inp_box.set_data_from_numpy(np.array([box], dtype=np.float32))

    outputs = [
        InferRequestedOutput("MASK",       binary_data=False),
        InferRequestedOutput("ENCODER_MS", binary_data=False),
        InferRequestedOutput("DECODER_MS", binary_data=False),
    ]

    t0 = time.perf_counter()
    result = client.infer(model_name=TRITON_MODEL, inputs=[inp_img, inp_box], outputs=outputs)
    wall_ms = (time.perf_counter() - t0) * 1e3

    return {
        "wall_ms":      wall_ms,
        "encoder_ms":   float(result.as_numpy("ENCODER_MS")[0, 0]),
        "decoder_ms":   float(result.as_numpy("DECODER_MS")[0, 0]),
        "inference_ms": wall_ms,
    }

# ── arrival pattern runners ────────────────────────────────────────────────────
def run_serial(request_fn: Callable, image_b64: str, box: list[float]) -> list[dict]:
    """Baseline: one request at a time."""
    results = []
    # warmup
    for _ in range(3):
        request_fn(image_b64, box)
    for _ in range(NUM_SERIAL_TRIALS):
        results.append(request_fn(image_b64, box))
    return results


def run_concurrent(request_fn: Callable, image_b64: str, box: list[float]) -> list[dict]:
    """N workers each sending a new request as soon as the last completes."""
    results = []

    def worker(_):
        return request_fn(image_b64, box)

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CONCURRENT) as ex:
        futs = [ex.submit(worker, i) for i in range(CONCURRENT_REQS)]
        for f in concurrent.futures.as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"  request error: {e}")
    return results


def _rate_loop(request_fn: Callable, image_b64: str, box: list[float],
               poisson: bool) -> list[dict]:
    """Send requests at a target rate; optionally Poisson inter-arrival times."""
    results   = []
    interval  = 1.0 / RATE_REQS_PER_SEC
    deadline  = time.perf_counter() + RATE_DURATION_SEC

    def worker():
        return request_fn(image_b64, box)

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        futures = []
        while time.perf_counter() < deadline:
            futures.append(ex.submit(worker))
            gap = np.random.exponential(interval) if poisson else interval
            time.sleep(max(0, gap))
        for f in concurrent.futures.as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"  request error: {e}")
    return results


def run_constant_rate(request_fn, image_b64, box):
    return _rate_loop(request_fn, image_b64, box, poisson=False)


def run_poisson(request_fn, image_b64, box):
    return _rate_loop(request_fn, image_b64, box, poisson=True)

# ── summary ───────────────────────────────────────────────────────────────────
def summarise(experiment: str, results: list[dict], duration_sec: float | None = None):
    if not results:
        print("No results collected.")
        return
    wall  = np.array([r["wall_ms"]      for r in results])
    enc   = np.array([r["encoder_ms"]   for r in results])
    dec   = np.array([r["decoder_ms"]   for r in results])

    throughput = (len(wall) / (duration_sec * 1e3) * 1e3) \
                 if duration_sec else len(wall) / wall.sum() * 1e3

    print()
    print("=" * 55)
    print(f"Experiment : {experiment}")
    print("=" * 55)
    print(f"Requests completed           : {len(wall)}")
    print(f"Arrival pattern              : {experiment.split('_', 1)[1]}")
    print(f"Inference Latency (median)   : {np.percentile(wall,50):.2f} ms")
    print(f"Inference Latency (p95)      : {np.percentile(wall,95):.2f} ms")
    print(f"Inference Latency (p99)      : {np.percentile(wall,99):.2f} ms")
    print(f"Encoder Latency  (median)    : {np.percentile(enc, 50):.2f} ms")
    print(f"Decoder Latency  (median)    : {np.percentile(dec, 50):.2f} ms")
    print(f"Throughput                   : {throughput:.2f} req/s")
    print("=" * 55)
    print()

# ── entrypoint ────────────────────────────────────────────────────────────────
EXPERIMENT_MAP = {
    # FastAPI
    "fastapi_serial":        (fastapi_request, run_serial),
    "fastapi_concurrent":    (fastapi_request, run_concurrent),
    "fastapi_constant_rate": (fastapi_request, run_constant_rate),
    "fastapi_poisson":       (fastapi_request, run_poisson),
    # Triton
    "triton_serial":         (triton_request,  run_serial),
    "triton_concurrent":     (triton_request,  run_concurrent),
    "triton_constant_rate":  (triton_request,  run_constant_rate),
    "triton_poisson":        (triton_request,  run_poisson),
}

if __name__ == "__main__":
    if EXPERIMENT not in EXPERIMENT_MAP:
        raise ValueError(
            f"Unknown EXPERIMENT='{EXPERIMENT}'. "
            f"Choose from: {', '.join(EXPERIMENT_MAP)}"
        )

    pairs      = load_manifest()
    pair       = pairs[0]
    image_b64  = encode_image(pair["image_path"])
    box        = first_box(pair["annotation_path"])

    request_fn, runner = EXPERIMENT_MAP[EXPERIMENT]
    print(f"\n>>> Starting experiment: {EXPERIMENT}")
    print(f"    Image: {pair['image_path']}")
    print(f"    Box:   {box}\n")

    t_start = time.perf_counter()
    results = runner(request_fn, image_b64, box)
    duration = time.perf_counter() - t_start

    summarise(EXPERIMENT, results, duration_sec=duration)
