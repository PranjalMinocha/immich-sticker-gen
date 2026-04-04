#!/usr/bin/env python3
"""
MobileSAM Inference Benchmark
==============================
Experiments (set via EXPERIMENT env var):
  pytorch_base      – plain PyTorch
  pytorch_compiled  – torch.compile (best effort, falls back if unsupported)
  onnx              – ONNX, no optimisation
  onnx_graph        – ONNX + graph optimisations
  onnx_dynquant     – ONNX + dynamic quantisation
  onnx_static_cons  – ONNX + static quant (conservative)
  onnx_static_agg   – ONNX + static quant (aggressive)
  ep_cpu            – ONNX Runtime, CPU execution provider
  ep_cuda           – ONNX Runtime, CUDA execution provider
  ep_tensorrt       – ONNX Runtime, TensorRT execution provider
  ep_openvino       – ONNX Runtime, OpenVINO execution provider
"""
from __future__ import annotations

import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import List

import cv2
import numpy as np
import onnx
import torch
from PIL import Image
from pycocotools import mask as mask_utils

# Silence repeated timm/mobile_sam registry warnings
warnings.filterwarnings(
    "ignore",
    message="Overwriting tiny_vit_.* in registry.*",
)

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/data"))
CKPT_PATH = MODEL_DIR / "mobile_sam.pt"
EXPERIMENT = os.environ.get("EXPERIMENT", "pytorch_base")

ONNX_RAW = MODEL_DIR / "mobile_sam_decoder.onnx"
ONNX_GRAPH = MODEL_DIR / "mobile_sam_decoder_graph.onnx"
ONNX_DYNQUANT = MODEL_DIR / "mobile_sam_decoder_dynquant.onnx"
ONNX_STATIC_CONS = MODEL_DIR / "mobile_sam_decoder_static_cons.onnx"
ONNX_STATIC_AGG = MODEL_DIR / "mobile_sam_decoder_static_agg.onnx"
ENCODER_ONNX = MODEL_DIR / "mobile_sam_encoder.onnx"

NUM_TRIALS = int(os.environ.get("NUM_TRIALS", "100"))
NUM_BATCHES = int(os.environ.get("NUM_BATCHES", "50"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))

# ── image preprocessing ───────────────────────────────────────────────────────
PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(3, 1, 1)
PIXEL_STD = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(3, 1, 1)


def preprocess_sam(image_rgb: np.ndarray, image_size: int = 1024) -> torch.Tensor:
    h, w = image_rgb.shape[:2]
    scale = image_size / max(h, w)
    new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.pad(
        resized,
        ((0, image_size - new_h), (0, image_size - new_w), (0, 0)),
        mode="constant",
    )
    x = torch.from_numpy(padded).permute(2, 0, 1).float()
    return (x - PIXEL_MEAN) / PIXEL_STD


# ── data helpers ─────────────────────────────────────────────────────────────
def load_manifest() -> List[dict]:
    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found in {DATA_DIR}. Run download_data.py first."
        )
    return json.loads(manifest_path.read_text())


def load_pair(pair: dict):
    image = np.array(Image.open(pair["image_path"]).convert("RGB"))
    ann_data = json.loads(Path(pair["annotation_path"]).read_text())
    return image, ann_data


def ann_to_mask(ann: dict) -> np.ndarray:
    seg = ann["segmentation"]
    if isinstance(seg["counts"], list):
        rle = mask_utils.frPyObjects(seg, seg["size"][0], seg["size"][1])
    else:
        rle = seg
    mask = mask_utils.decode(rle)
    return (mask[..., 0] if mask.ndim == 3 else mask).astype(bool)


def mask_to_box(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


# ── MobileSAM model loading ───────────────────────────────────────────────────
def load_mobilesam():
    from mobile_sam import sam_model_registry, SamPredictor

    sam = sam_model_registry["vit_t"](checkpoint=str(CKPT_PATH))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device).eval()
    predictor = SamPredictor(sam)
    return sam, predictor, device


# ── ONNX helpers ──────────────────────────────────────────────────────────────
def validate_onnx_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        onnx.load(str(path))
        return True
    except Exception:
        return False


def unlink_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


# ── ONNX export helpers ───────────────────────────────────────────────────────
def export_decoder_onnx(sam, path: Path, force: bool = False) -> None:
    """Export the MobileSAM decoder (prompt enc + mask dec) to ONNX."""
    from mobile_sam.utils.onnx import SamOnnxModel
    import onnx
    from onnx import shape_inference

    if force:
        unlink_if_exists(path)
    elif validate_onnx_file(path):
        print(f"Using existing valid decoder ONNX -> {path}")
        return
    else:
        unlink_if_exists(path)

    print(f"Exporting decoder ONNX -> {path}")
    onnx_model = SamOnnxModel(sam, return_single_mask=True)
    onnx_model.eval()

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size

    dummy = dict(
        image_embeddings=torch.randn(1, embed_dim, *embed_size, dtype=torch.float32),
        point_coords=torch.tensor(
            [[[100.0, 100.0], [700.0, 700.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]],
            dtype=torch.float32,
        ),
        point_labels=torch.tensor([[2.0, 3.0, -1.0, -1.0, -1.0]], dtype=torch.float32),
        mask_input=torch.zeros(1, 1, 256, 256, dtype=torch.float32),
        has_mask_input=torch.tensor([0.0], dtype=torch.float32),
        # IMPORTANT: TensorRT needs this shape tensor to be int32/int64, not float
        orig_im_size=torch.tensor([1024, 1024], dtype=torch.int64),
    )

    torch.onnx.export(
        onnx_model,
        tuple(dummy.values()),
        str(path),
        input_names=list(dummy.keys()),
        output_names=["masks", "iou_predictions", "low_res_masks"],
        dynamic_axes={
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    model = onnx.load(str(path))
    model = shape_inference.infer_shapes(model)
    onnx.save(model, str(path))

    onnx.load(str(path))
    print("  decoder export done with shape inference.")


def export_encoder_onnx(sam, path: Path, force: bool = False) -> None:
    """Export the TinyViT image encoder to ONNX."""
    if force:
        unlink_if_exists(path)
    elif validate_onnx_file(path):
        print(f"Using existing valid encoder ONNX -> {path}")
        return
    else:
        unlink_if_exists(path)

    print(f"Exporting encoder ONNX -> {path}")
    dummy = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)

    torch.onnx.export(
        sam.image_encoder,
        dummy,
        str(path),
        input_names=["image"],
        output_names=["image_embeddings"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "image_embeddings": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    onnx.load(str(path))
    print("  encoder export done.")


# ── ONNX optimisation helpers ─────────────────────────────────────────────────
def optimise_graph(src: Path, dst: Path) -> None:
    unlink_if_exists(dst)

    import onnxoptimizer

    model = onnx.load(str(src))
    passes = onnxoptimizer.get_fuse_and_elimination_passes()
    optimised = onnxoptimizer.optimize(model, passes)
    onnx.save(optimised, str(dst))
    print(f"Graph-optimised ONNX -> {dst}")


def quantise_dynamic(src: Path, dst: Path) -> None:
    unlink_if_exists(dst)

    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    print(f"Dynamic-quantised ONNX -> {dst}")


# ── Static quant calibration readers ──────────────────────────────────────────
def _encoder_calibration_data_reader(pairs, n=20):
    from onnxruntime.quantization import CalibrationDataReader

    class _Reader(CalibrationDataReader):
        def __init__(self):
            self._data = []
            for p in pairs[:n]:
                img, _ = load_pair(p)
                tensor = preprocess_sam(img).unsqueeze(0).numpy().astype(np.float32)
                self._data.append({"image": tensor})
            self._iter = iter(self._data)

        def get_next(self):
            return next(self._iter, None)

    return _Reader()


def _decoder_calibration_data_reader(pairs, sam, n=20):
    from onnxruntime.quantization import CalibrationDataReader

    class _Reader(CalibrationDataReader):
        def __init__(self):
            self._data = []
            device = next(sam.parameters()).device

            for p in pairs[:n]:
                img, ann = load_pair(p)
                anns = ann.get("annotations", [])
                if not anns:
                    continue

                gt = ann_to_mask(anns[0])
                box = mask_to_box(gt)

                with torch.no_grad():
                    image_tensor = preprocess_sam(img).unsqueeze(0).to(device)
                    image_embeddings = (
                        sam.image_encoder(image_tensor).detach().cpu().numpy().astype(np.float32)
                    )

                point_coords = np.array(
                    [[[box[0], box[1]], [box[2], box[3]], [0, 0], [0, 0], [0, 0]]],
                    dtype=np.float32,
                )
                point_labels = np.array([[2, 3, -1, -1, -1]], dtype=np.float32)
                mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
                has_mask_input = np.array([0], dtype=np.float32)
                # IMPORTANT: int64 for TensorRT shape tensor
                orig_im_size = np.array([img.shape[0], img.shape[1]], dtype=np.int64)

                self._data.append(
                    {
                        "image_embeddings": image_embeddings,
                        "point_coords": point_coords,
                        "point_labels": point_labels,
                        "mask_input": mask_input,
                        "has_mask_input": has_mask_input,
                        "orig_im_size": orig_im_size,
                    }
                )

            self._iter = iter(self._data)

        def get_next(self):
            return next(self._iter, None)

    return _Reader()


def quantise_static_encoder(src: Path, dst: Path, pairs, per_channel: bool = False) -> None:
    unlink_if_exists(dst)

    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    quantize_static(
        str(src),
        str(dst),
        calibration_data_reader=_encoder_calibration_data_reader(pairs),
        quant_format=QuantFormat.QOperator,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method=CalibrationMethod.MinMax,
    )
    label = "aggressive" if per_channel else "conservative"
    print(f"Static-quantised encoder ({label}) ONNX -> {dst}")


def quantise_static_decoder(
    src: Path, dst: Path, pairs, sam, per_channel: bool = False
) -> None:
    unlink_if_exists(dst)

    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    quantize_static(
        str(src),
        str(dst),
        calibration_data_reader=_decoder_calibration_data_reader(pairs, sam),
        quant_format=QuantFormat.QOperator,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method=CalibrationMethod.MinMax,
    )
    label = "aggressive" if per_channel else "conservative"
    print(f"Static-quantised decoder ({label}) ONNX -> {dst}")


# ── ONNX inference session factory ────────────────────────────────────────────
def make_ort_session(onnx_path: Path, providers=None):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = providers or ["CPUExecutionProvider"]
    return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)


# ── ONNX e2e inference (encoder + decoder) ────────────────────────────────────
def onnx_e2e(enc_sess, dec_sess, image: np.ndarray, box: np.ndarray) -> np.ndarray:
    tensor = preprocess_sam(image).unsqueeze(0).numpy().astype(np.float32)
    (embedding,) = enc_sess.run(["image_embeddings"], {"image": tensor})

    point_coords = np.array(
        [[[box[0], box[1]], [box[2], box[3]], [0, 0], [0, 0], [0, 0]]],
        dtype=np.float32,
    )
    point_labels = np.array([[2, 3, -1, -1, -1]], dtype=np.float32)
    mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    has_mask = np.array([0], dtype=np.float32)

    # IMPORTANT: TensorRT needs int32/int64 here
    orig_size = np.array([image.shape[0], image.shape[1]], dtype=np.int64)

    masks, _, _ = dec_sess.run(
        ["masks", "iou_predictions", "low_res_masks"],
        {
            "image_embeddings": embedding,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "mask_input": mask_input,
            "has_mask_input": has_mask,
            "orig_im_size": orig_size,
        },
    )
    return masks[0, 0] > 0


# ── metric helpers ────────────────────────────────────────────────────────────
def summarise(
    experiment: str,
    enc_path: Path,
    dec_path: Path,
    latencies: list,
    batch_times: list,
    batch_size: int,
) -> None:
    lat = np.array(latencies)
    blat = np.array(batch_times)
    enc_mb = enc_path.stat().st_size / 1e6 if enc_path.exists() else 0.0
    dec_mb = dec_path.stat().st_size / 1e6 if dec_path.exists() else 0.0

    print()
    print("=" * 55)
    print(f"Experiment : {experiment}")
    print("=" * 55)
    print(f"Encoder model                : {enc_path.name} ({enc_mb:.2f} MB)")
    print(f"Decoder model                : {dec_path.name} ({dec_mb:.2f} MB)")
    print(f"Total Model Size on Disk     : {enc_mb + dec_mb:.2f} MB")
    print(f"Inference Latency (median)   : {np.percentile(lat, 50) * 1e3:.2f} ms")
    print(f"Inference Latency (p95)      : {np.percentile(lat, 95) * 1e3:.2f} ms")
    print(f"Inference Latency (p99)      : {np.percentile(lat, 99) * 1e3:.2f} ms")
    print(f"Inference Throughput         : {len(lat) / lat.sum():.2f} FPS")
    if len(blat):
        fps = (batch_size * len(blat)) / blat.sum()
        print(f"Batch Throughput (encoder)   : {fps:.2f} FPS")
        print(f"Batch Latency (encoder, p50) : {np.percentile(blat, 50) * 1e3:.2f} ms")
        print(f"Effective Batch Size         : {batch_size}")
    print("=" * 55)
    print()


# ── experiment runners ────────────────────────────────────────────────────────
def run_pytorch(compile_model: bool = False) -> None:
    sam, predictor, device = load_mobilesam()
    compile_enabled = False
    original_encoder = sam.image_encoder

    if compile_model:
        try:
            sam.image_encoder = torch.compile(
                sam.image_encoder,
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=False,
            )
            predictor.model = sam
            compile_enabled = True
            print("torch.compile enabled for image_encoder")
        except Exception as e:
            print(f"torch.compile setup failed, using eager mode instead: {e}")
            sam.image_encoder = original_encoder
            predictor.model = sam

    pairs = load_manifest()
    single_img, single_ann = load_pair(pairs[0])
    box = mask_to_box(ann_to_mask(single_ann["annotations"][0]))

    with torch.no_grad():
        for _ in range(5):
            try:
                predictor.set_image(single_img)
                predictor.predict(box=box[None], multimask_output=False)
            except Exception as e:
                if compile_enabled:
                    print(f"Compiled encoder failed during warmup, falling back to eager: {e}")
                    sam.image_encoder = original_encoder
                    predictor.model = sam
                    compile_enabled = False
                    predictor.set_image(single_img)
                    predictor.predict(box=box[None], multimask_output=False)
                else:
                    raise

    latencies = []
    with torch.no_grad():
        for _ in range(NUM_TRIALS):
            t0 = time.perf_counter()
            predictor.set_image(single_img)
            predictor.predict(box=box[None], multimask_output=False)
            latencies.append(time.perf_counter() - t0)

    batch_imgs = [preprocess_sam(load_pair(p)[0]) for p in pairs[:BATCH_SIZE]]
    batch_tensor = torch.stack(batch_imgs).to(device)

    with torch.no_grad():
        sam.image_encoder(batch_tensor)

    batch_times = []
    with torch.no_grad():
        for _ in range(NUM_BATCHES):
            t0 = time.perf_counter()
            sam.image_encoder(batch_tensor)
            batch_times.append(time.perf_counter() - t0)

    summarise(EXPERIMENT, CKPT_PATH, CKPT_PATH, latencies, batch_times, BATCH_SIZE)


def run_onnx(enc_path: Path, dec_path: Path, providers=None) -> None:
    sam, _, _ = load_mobilesam()
    export_encoder_onnx(sam, enc_path)
    export_decoder_onnx(sam, dec_path)
    del sam

    enc_sess = make_ort_session(enc_path, providers)
    dec_sess = make_ort_session(dec_path, providers)

    pairs = load_manifest()
    single_img, single_ann = load_pair(pairs[0])
    box = mask_to_box(ann_to_mask(single_ann["annotations"][0]))

    for _ in range(5):
        onnx_e2e(enc_sess, dec_sess, single_img, box)

    latencies = []
    for _ in range(NUM_TRIALS):
        t0 = time.perf_counter()
        onnx_e2e(enc_sess, dec_sess, single_img, box)
        latencies.append(time.perf_counter() - t0)

    batch_imgs = [
        preprocess_sam(load_pair(p)[0]).unsqueeze(0).numpy().astype(np.float32)
        for p in pairs[:BATCH_SIZE]
    ]
    batch_arr = np.concatenate(batch_imgs, axis=0).astype(np.float32)

    batch_times = []
    effective_batch_size = BATCH_SIZE

    try:
        enc_sess.run(["image_embeddings"], {"image": batch_arr})
        for _ in range(NUM_BATCHES):
            t0 = time.perf_counter()
            enc_sess.run(["image_embeddings"], {"image": batch_arr})
            batch_times.append(time.perf_counter() - t0)
    except Exception as e:
        print(f"Batched encoder ONNX failed, falling back to batch size 1: {e}")
        single_batch = batch_imgs[0].astype(np.float32)
        effective_batch_size = 1

        enc_sess.run(["image_embeddings"], {"image": single_batch})
        for _ in range(NUM_BATCHES):
            t0 = time.perf_counter()
            enc_sess.run(["image_embeddings"], {"image": single_batch})
            batch_times.append(time.perf_counter() - t0)

    summarise(EXPERIMENT, enc_path, dec_path, latencies, batch_times, effective_batch_size)


def run_onnx_graph() -> None:
    sam, _, _ = load_mobilesam()
    export_encoder_onnx(sam, ENCODER_ONNX)
    export_decoder_onnx(sam, ONNX_RAW)
    del sam

    enc_graph = MODEL_DIR / "mobile_sam_encoder_graph.onnx"
    optimise_graph(ENCODER_ONNX, enc_graph)
    optimise_graph(ONNX_RAW, ONNX_GRAPH)

    run_onnx(enc_graph, ONNX_GRAPH)


def run_onnx_dynquant() -> None:
    sam, _, _ = load_mobilesam()
    export_encoder_onnx(sam, ENCODER_ONNX)
    export_decoder_onnx(sam, ONNX_RAW)
    del sam

    enc_dynq = MODEL_DIR / "mobile_sam_encoder_dynquant.onnx"
    quantise_dynamic(ENCODER_ONNX, enc_dynq)
    quantise_dynamic(ONNX_RAW, ONNX_DYNQUANT)

    run_onnx(enc_dynq, ONNX_DYNQUANT)


def run_onnx_static(aggressive: bool = False) -> None:
    sam, _, _ = load_mobilesam()
    export_encoder_onnx(sam, ENCODER_ONNX)
    export_decoder_onnx(sam, ONNX_RAW)

    pairs = load_manifest()
    suffix = "agg" if aggressive else "cons"
    enc_sq = MODEL_DIR / f"mobile_sam_encoder_static_{suffix}.onnx"
    dec_sq = MODEL_DIR / f"mobile_sam_decoder_static_{suffix}.onnx"

    quantise_static_encoder(ENCODER_ONNX, enc_sq, pairs, per_channel=aggressive)
    quantise_static_decoder(ONNX_RAW, dec_sq, pairs, sam, per_channel=aggressive)

    del sam
    run_onnx(enc_sq, dec_sq)


def run_ep(provider: str) -> None:
    import onnxruntime as ort

    provider_map = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
        "openvino": "OpenVINOExecutionProvider",
    }

    requested = provider_map[provider]
    available = ort.get_available_providers()
    print("Available ORT providers:", available)

    if requested not in available:
        raise RuntimeError(
            f"{requested} is not available in this container. "
            f"Available providers: {available}"
        )

    providers = [requested]
    if requested != "CPUExecutionProvider":
        providers.append("CPUExecutionProvider")

    sam, _, _ = load_mobilesam()
    export_encoder_onnx(sam, ENCODER_ONNX)
    export_decoder_onnx(sam, ONNX_RAW)
    del sam

    sess = make_ort_session(ENCODER_ONNX, providers)
    actual = sess.get_providers()
    print("Session providers:", actual)
    if actual[0] != requested:
        raise RuntimeError(
            f"Requested {requested}, but session is using {actual}"
        )

    run_onnx(ENCODER_ONNX, ONNX_RAW, providers=providers)


# ── entrypoint ────────────────────────────────────────────────────────────────
EXPERIMENT_MAP = {
    "pytorch_base": lambda: run_pytorch(compile_model=False),
    "pytorch_compiled": lambda: run_pytorch(compile_model=True),
    "onnx": lambda: run_onnx(ENCODER_ONNX, ONNX_RAW),
    "onnx_graph": run_onnx_graph,
    "onnx_dynquant": run_onnx_dynquant,
    "onnx_static_cons": lambda: run_onnx_static(aggressive=False),
    "onnx_static_agg": lambda: run_onnx_static(aggressive=True),
    "ep_cpu": lambda: run_ep("cpu"),
    "ep_cuda": lambda: run_ep("cuda"),
    "ep_tensorrt": lambda: run_ep("tensorrt"),
    "ep_openvino": lambda: run_ep("openvino"),
}

if __name__ == "__main__":
    fn = EXPERIMENT_MAP.get(EXPERIMENT)
    if fn is None:
        raise ValueError(
            f"Unknown experiment '{EXPERIMENT}'. "
            f"Choose from: {', '.join(EXPERIMENT_MAP)}"
        )

    print(f"\n>>> Starting experiment: {EXPERIMENT}\n")
    fn()
