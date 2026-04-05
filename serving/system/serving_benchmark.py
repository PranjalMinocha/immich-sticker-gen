#!/usr/bin/env python3
"""
Serving benchmark client for MobileSAM.

EXPERIMENT env var:
  fastapi_serial, fastapi_concurrent, fastapi_poisson
  triton_serial,  triton_concurrent,  triton_poisson
"""
from __future__ import annotations

import base64, concurrent.futures, json, os, time
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

DATA_DIR          = Path(os.environ.get("DATA_DIR",   "/data"))
FASTAPI_URL       = os.environ.get("FASTAPI_URL",  "http://fastapi_server:8000/predict")
TRITON_URL        = os.environ.get("TRITON_URL",   "triton_server:8000")
TRITON_MODEL      = os.environ.get("TRITON_MODEL", "mobile_sam_gpu_1")
EXPERIMENT        = os.environ.get("EXPERIMENT",   "fastapi_serial")
NUM_SERIAL_TRIALS = int(os.environ.get("NUM_SERIAL_TRIALS",  "100"))
NUM_CONCURRENT    = int(os.environ.get("NUM_CONCURRENT",     "8"))
CONCURRENT_REQS   = int(os.environ.get("CONCURRENT_REQS",   "200"))
RATE_REQS_PER_SEC = float(os.environ.get("RATE_REQS_PER_SEC", "5.0"))
RATE_DURATION_SEC = float(os.environ.get("RATE_DURATION_SEC", "30.0"))


def load_manifest():
    p = DATA_DIR / "manifest.json"
    if not p.exists():
        raise FileNotFoundError("manifest.json not found — run init first")
    return json.loads(p.read_text())


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def first_box(ann_path: str) -> list[float]:
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


def fastapi_request(image_b64: str, box: list[float]) -> dict:
    import requests
    t0   = time.perf_counter()
    resp = requests.post(FASTAPI_URL, json={"image": image_b64, "box": box}, timeout=60)
    wall_ms = (time.perf_counter() - t0) * 1e3
    resp.raise_for_status()
    data = resp.json()
    return {
        "wall_ms":      wall_ms,
        "encoder_ms":   data.get("encoder_ms", 0),
        "decoder_ms":   data.get("decoder_ms", 0),
    }


def triton_request(image_b64: str, box: list[float]) -> dict:
    from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
    client  = InferenceServerClient(url=TRITON_URL)
    inp_img = InferInput("INPUT_IMAGE", [1, 1], "BYTES")
    inp_img.set_data_from_numpy(np.array([[image_b64]], dtype=object))
    inp_box = InferInput("BOX", [1, 4], "FP32")
    inp_box.set_data_from_numpy(np.array([box], dtype=np.float32))
    t0 = time.perf_counter()
    result = client.infer(
        model_name=TRITON_MODEL,
        inputs=[inp_img, inp_box],
        outputs=[
            InferRequestedOutput("MASK",       binary_data=False),
            InferRequestedOutput("ENCODER_MS", binary_data=False),
            InferRequestedOutput("DECODER_MS", binary_data=False),
        ],
    )
    wall_ms = (time.perf_counter() - t0) * 1e3
    return {
        "wall_ms":    wall_ms,
        "encoder_ms": float(result.as_numpy("ENCODER_MS")[0, 0]),
        "decoder_ms": float(result.as_numpy("DECODER_MS")[0, 0]),
    }


def run_serial(fn: Callable, image_b64: str, box: list[float]) -> list[dict]:
    for _ in range(3):           # warmup
        fn(image_b64, box)
    return [fn(image_b64, box) for _ in range(NUM_SERIAL_TRIALS)]


def run_concurrent(fn: Callable, image_b64: str, box: list[float]) -> list[dict]:
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CONCURRENT) as ex:
        futs = [ex.submit(fn, image_b64, box) for _ in range(CONCURRENT_REQS)]
        for f in concurrent.futures.as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"  error: {e}")
    return results


def run_poisson(fn: Callable, image_b64: str, box: list[float]) -> list[dict]:
    results  = []
    interval = 1.0 / RATE_REQS_PER_SEC
    deadline = time.perf_counter() + RATE_DURATION_SEC
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        futures = []
        while time.perf_counter() < deadline:
            futures.append(ex.submit(fn, image_b64, box))
            time.sleep(max(0, np.random.exponential(interval)))
        for f in concurrent.futures.as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"  error: {e}")
    return results


def summarise(results: list[dict], duration_sec: float):
    if not results:
        print("No results.")
        return
    wall = np.array([r["wall_ms"]    for r in results])
    enc  = np.array([r["encoder_ms"] for r in results])
    dec  = np.array([r["decoder_ms"] for r in results])
    print()
    print("=" * 55)
    print(f"Experiment : {EXPERIMENT}  |  model: {TRITON_MODEL if 'triton' in EXPERIMENT else 'fastapi'}")
    print("=" * 55)
    print(f"Requests completed           : {len(wall)}")
    print(f"Inference Latency (median)   : {np.percentile(wall,50):.2f} ms")
    print(f"Inference Latency (p95)      : {np.percentile(wall,95):.2f} ms")
    print(f"Inference Latency (p99)      : {np.percentile(wall,99):.2f} ms")
    print(f"Encoder Latency  (median)    : {np.percentile(enc, 50):.2f} ms")
    print(f"Decoder Latency  (median)    : {np.percentile(dec, 50):.2f} ms")
    print(f"Throughput                   : {len(wall)/duration_sec:.2f} req/s")
    print("=" * 55)
    print()


EXPERIMENT_MAP = {
    "fastapi_serial":     (fastapi_request, run_serial),
    "fastapi_concurrent": (fastapi_request, run_concurrent),
    "fastapi_poisson":    (fastapi_request, run_poisson),
    "triton_serial":      (triton_request,  run_serial),
    "triton_concurrent":  (triton_request,  run_concurrent),
    "triton_poisson":     (triton_request,  run_poisson),
}

if __name__ == "__main__":
    if EXPERIMENT not in EXPERIMENT_MAP:
        raise ValueError(f"Unknown EXPERIMENT='{EXPERIMENT}'. Choose from: {', '.join(EXPERIMENT_MAP)}")

    pairs     = load_manifest()
    image_b64 = encode_image(pairs[0]["image_path"])
    box       = first_box(pairs[0]["annotation_path"])

    fn, runner = EXPERIMENT_MAP[EXPERIMENT]
    print(f"\n>>> Starting: {EXPERIMENT}")
    print("Waiting 10s for server to be ready...")
    time.sleep(10)

    t0      = time.perf_counter()
    results = runner(fn, image_b64, box)
    summarise(results, time.perf_counter() - t0)