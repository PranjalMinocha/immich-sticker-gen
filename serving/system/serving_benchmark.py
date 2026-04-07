#!/usr/bin/env python3
"""
Serving benchmark client for MobileSAM.

EXPERIMENT env var:
  fastapi_serial, fastapi_concurrent, fastapi_poisson
  triton_serial,  triton_concurrent,  triton_poisson
"""
from __future__ import annotations

import base64, concurrent.futures, json, os, subprocess, time
from pathlib import Path

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


# ── FastAPI ───────────────────────────────────────────────────────────────────

def fastapi_request(image_b64: str, box: list[float]) -> dict:
    import requests
    t0   = time.perf_counter()
    resp = requests.post(FASTAPI_URL, json={"image": image_b64, "box": box}, timeout=60)
    wall_ms = (time.perf_counter() - t0) * 1e3
    resp.raise_for_status()
    data = resp.json()
    return {
        "wall_ms":    wall_ms,
        "encoder_ms": data.get("encoder_ms", 0),
        "decoder_ms": data.get("decoder_ms", 0),
    }


def run_serial(fn, image_b64: str, box: list[float]) -> tuple[list[dict], int]:
    for _ in range(3):
        fn(image_b64, box)
    results, errors = [], 0
    for _ in range(NUM_SERIAL_TRIALS):
        try:
            results.append(fn(image_b64, box))
        except Exception as e:
            print(f"  error: {e}")
            errors += 1
    return results, NUM_SERIAL_TRIALS


def run_concurrent(fn, image_b64: str, box: list[float]) -> tuple[list[dict], int]:
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CONCURRENT) as ex:
        futs = [ex.submit(fn, image_b64, box) for _ in range(CONCURRENT_REQS)]
        for f in concurrent.futures.as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"  error: {e}")
    return results, CONCURRENT_REQS


def run_poisson(fn, image_b64: str, box: list[float]) -> tuple[list[dict], int]:
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
    return results, len(futures)


def summarise(results: list[dict], total: int, duration_sec: float):
    errors = total - len(results)
    if not results:
        print("No results."); return
    wall = np.array([r["wall_ms"]    for r in results])
    enc  = np.array([r["encoder_ms"] for r in results])
    dec  = np.array([r["decoder_ms"] for r in results])
    print()
    print("=" * 55)
    print(f"Experiment : {EXPERIMENT}  |  model: fastapi")
    print("=" * 55)
    print(f"Requests completed           : {len(wall)} / {total}")
    print(f"Error rate                   : {errors/total*100:.1f}%  ({errors} errors)")
    print(f"Inference Latency (median)   : {np.percentile(wall,50):.2f} ms")
    print(f"Inference Latency (p95)      : {np.percentile(wall,95):.2f} ms")
    print(f"Inference Latency (p99)      : {np.percentile(wall,99):.2f} ms")
    print(f"Encoder Latency  (median)    : {np.percentile(enc, 50):.2f} ms")
    print(f"Decoder Latency  (median)    : {np.percentile(dec, 50):.2f} ms")
    print(f"Throughput                   : {len(wall)/duration_sec:.2f} req/s")
    print("=" * 55)
    print()


# ── Triton (perf_analyzer) ────────────────────────────────────────────────────

def generate_perf_input(image_b64: str, box: list[float]):
    path = DATA_DIR / "perf_input.json"
    path.write_text(json.dumps({
        "data": [{
            "INPUT_IMAGE": {"content": [image_b64], "shape": [1]},
            "BOX":         {"content": box,          "shape": [4]},
        }]
    }))


def run_perf_analyzer(extra_args: list[str]):
    cmd = [
        "perf_analyzer",
        "-u", TRITON_URL,
        "-m", TRITON_MODEL,
        "--input-data", str(DATA_DIR / "perf_input.json"),
        "-b", "1",
        "--measurement-interval", str(int(RATE_DURATION_SEC * 1000)),
    ] + extra_args
    print(f"\n>>> {' '.join(cmd)}\n")
    subprocess.run(cmd)


FASTAPI_EXPERIMENTS = {
    "fastapi_serial":     (fastapi_request, run_serial),
    "fastapi_concurrent": (fastapi_request, run_concurrent),
    "fastapi_poisson":    (fastapi_request, run_poisson),
}

TRITON_EXPERIMENTS = {
    "triton_serial":     ["--concurrency-range", "1"],
    "triton_concurrent": ["--concurrency-range", str(NUM_CONCURRENT)],
    "triton_poisson":    ["--request-rate-range", str(int(RATE_REQS_PER_SEC)),
                          "--request-distribution", "poisson"],
}

if __name__ == "__main__":
    all_experiments = list(FASTAPI_EXPERIMENTS) + list(TRITON_EXPERIMENTS)
    if EXPERIMENT not in all_experiments:
        raise ValueError(f"Unknown EXPERIMENT='{EXPERIMENT}'. Choose from: {', '.join(all_experiments)}")

    pairs     = load_manifest()
    image_b64 = encode_image(pairs[0]["image_path"])
    box       = first_box(pairs[0]["annotation_path"])

    print(f"\n>>> Starting: {EXPERIMENT}")
    print("Waiting 10s for server to be ready...")
    time.sleep(10)

    if EXPERIMENT in FASTAPI_EXPERIMENTS:
        fn, runner = FASTAPI_EXPERIMENTS[EXPERIMENT]
        t0             = time.perf_counter()
        results, total = runner(fn, image_b64, box)
        summarise(results, total, time.perf_counter() - t0)
    else:
        generate_perf_input(image_b64, box)
        run_perf_analyzer(TRITON_EXPERIMENTS[EXPERIMENT])
