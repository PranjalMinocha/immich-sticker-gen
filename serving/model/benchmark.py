#!/usr/bin/env python3
# serving/model/benchmark.py
"""
MobileSAM model benchmark.

EXPERIMENT env var: pytorch, onnx, onnx_quantized, tensorrt, openvino
"""
from __future__ import annotations

import json, os, time, warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

warnings.filterwarnings("ignore", message="Overwriting tiny_vit_.*")

DATA_DIR   = Path(os.environ.get("DATA_DIR",  "/data"))
MODEL_DIR  = Path(os.environ.get("MODEL_DIR", "/data"))
CKPT_PATH  = Path(os.environ.get("CKPT_PATH", str(MODEL_DIR / "mobile_sam.pt")))
EXPERIMENT = os.environ.get("EXPERIMENT", "pytorch")
NUM_TRIALS = int(os.environ.get("NUM_TRIALS", "100"))

ENCODER_ONNX      = MODEL_DIR / "mobile_sam_encoder.onnx"
DECODER_ONNX      = MODEL_DIR / "mobile_sam_decoder.onnx"
ENCODER_ONNX_QUANT = MODEL_DIR / "mobile_sam_encoder_quantized.onnx"
DECODER_ONNX_QUANT = MODEL_DIR / "mobile_sam_decoder_quantized.onnx"

PIXEL_MEAN = torch.tensor([123.675, 116.28,  103.53]).view(3, 1, 1)
PIXEL_STD  = torch.tensor([ 58.395,  57.12,  57.375]).view(3, 1, 1)


def preprocess(image_rgb: np.ndarray, size: int = 1024) -> torch.Tensor:
    h, w = image_rgb.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
    resized = np.array(Image.fromarray(image_rgb).resize((new_w, new_h), Image.BILINEAR))
    padded  = np.pad(resized, ((0, size-new_h), (0, size-new_w), (0, 0)), mode="constant")
    x = torch.from_numpy(padded).permute(2, 0, 1).float()
    return (x - PIXEL_MEAN) / PIXEL_STD


def load_sample():
    pair  = json.loads((DATA_DIR / "manifest.json").read_text())[0]
    image = np.array(Image.open(pair["image_path"]).convert("RGB"))
    ann   = json.loads(Path(pair["annotation_path"]).read_text())
    seg   = ann["annotations"][0]["segmentation"]
    rle   = mask_utils.frPyObjects(seg, seg["size"][0], seg["size"][1]) \
            if isinstance(seg["counts"], list) else seg
    m     = mask_utils.decode(rle)
    m     = m[..., 0] if m.ndim == 3 else m
    ys, xs = np.where(m)
    box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
    return image, box


def load_sam():
    from mobile_sam import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_t"](checkpoint=str(CKPT_PATH))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device).eval()
    return sam, SamPredictor(sam)


def export_encoder(sam, path: Path):
    if path.exists():
        print(f"Encoder already exists: {path}"); return
    import onnx
    print(f"Exporting encoder → {path}")
    torch.onnx.export(
        sam.image_encoder, torch.randn(1, 3, 1024, 1024), str(path),
        input_names=["image"], output_names=["image_embeddings"], opset_version=17,
    )
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(str(path))), str(path))


def export_decoder(sam, path: Path):
    if path.exists():
        print(f"Decoder already exists: {path}"); return
    import onnx
    print(f"Exporting decoder → {path}")
    from mobile_sam.utils.onnx import SamOnnxModel
    model      = SamOnnxModel(sam, return_single_mask=True).eval()
    embed_dim  = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    dummy = (
        torch.randn(1, embed_dim, *embed_size),
        torch.randint(0, 1024, (1, 5, 2), dtype=torch.float32),
        torch.randint(0, 4,    (1, 5),    dtype=torch.float32),
        torch.zeros(1, 1, 256, 256),
        torch.tensor([0.0]),
        torch.tensor([1024, 1024], dtype=torch.int64),
    )
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["image_embeddings","point_coords","point_labels",
                     "mask_input","has_mask_input","orig_im_size"],
        output_names=["masks","iou_predictions","low_res_masks"],
        dynamic_axes={"point_coords":{1:"num_points"},"point_labels":{1:"num_points"}},
        opset_version=17,
    )
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(str(path))), str(path))


def quantize(src: Path, dst: Path):
    if dst.exists():
        print(f"Quantized model already exists: {dst}"); return
    from onnxruntime.quantization import quantize_dynamic, QuantType
    print(f"Quantizing {src.name} → {dst.name}")
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gather"])


def ort_infer(enc_sess, dec_sess, image: np.ndarray, box: np.ndarray):
    tensor = preprocess(image).unsqueeze(0).numpy().astype(np.float32)
    (embed,) = enc_sess.run(["image_embeddings"], {"image": tensor})
    dec_sess.run(
        ["masks", "iou_predictions", "low_res_masks"],
        {
            "image_embeddings": embed,
            "point_coords":     np.array([[[box[0],box[1]],[box[2],box[3]],[0,0],[0,0],[0,0]]], dtype=np.float32),
            "point_labels":     np.array([[2, 3, -1, -1, -1]], dtype=np.float32),
            "mask_input":       np.zeros((1,1,256,256), dtype=np.float32),
            "has_mask_input":   np.array([0], dtype=np.float32),
            "orig_im_size":     np.array([image.shape[0], image.shape[1]], dtype=np.int64),
        },
    )


def summarise(latencies: list):
    lat = np.array(latencies) * 1e3
    print()
    print("=" * 55)
    print(f"Experiment : {EXPERIMENT}  |  checkpoint: {CKPT_PATH.name}")
    print("=" * 55)
    print(f"Requests completed         : {len(lat)}")
    print(f"Latency (median)           : {np.percentile(lat, 50):.2f} ms")
    print(f"Latency (p95)              : {np.percentile(lat, 95):.2f} ms")
    print(f"Latency (p99)              : {np.percentile(lat, 99):.2f} ms")
    print(f"Throughput                 : {1000/np.percentile(lat, 50):.2f} FPS")
    print("=" * 55)
    print()


def run_pytorch():
    sam, predictor = load_sam()
    image, box = load_sample()
    with torch.no_grad():
        for _ in range(3):
            predictor.set_image(image)
            predictor.predict(box=box[None], multimask_output=False)
        latencies = []
        for _ in range(NUM_TRIALS):
            t0 = time.perf_counter()
            predictor.set_image(image)
            predictor.predict(box=box[None], multimask_output=False)
            latencies.append(time.perf_counter() - t0)
    summarise(latencies)


def run_onnx(providers: list, enc_path: Path = ENCODER_ONNX, dec_path: Path = DECODER_ONNX,
             dec_providers: list = None):
    import onnxruntime as ort
    from mobile_sam import sam_model_registry
    sam_cpu = sam_model_registry["vit_t"](checkpoint=str(CKPT_PATH)).eval()
    export_encoder(sam_cpu, ENCODER_ONNX)
    export_decoder(sam_cpu, DECODER_ONNX)
    del sam_cpu

    if enc_path == ENCODER_ONNX_QUANT:
        quantize(ENCODER_ONNX, ENCODER_ONNX_QUANT)
        quantize(DECODER_ONNX, DECODER_ONNX_QUANT)

    available = ort.get_available_providers()
    print("Available ORT providers:", available)

    enc = ort.InferenceSession(str(enc_path), providers=providers)
    dec = ort.InferenceSession(str(dec_path), providers=dec_providers or providers)
    print("Active providers:", enc.get_providers())

    image, box = load_sample()
    for _ in range(3):
        ort_infer(enc, dec, image, box)
    latencies = []
    for _ in range(NUM_TRIALS):
        t0 = time.perf_counter()
        ort_infer(enc, dec, image, box)
        latencies.append(time.perf_counter() - t0)
    summarise(latencies)


def run_openvino():
    from mobile_sam import sam_model_registry
    from openvino import Core
    sam_cpu = sam_model_registry["vit_t"](checkpoint=str(CKPT_PATH)).eval()
    export_encoder(sam_cpu, ENCODER_ONNX)
    export_decoder(sam_cpu, DECODER_ONNX)
    del sam_cpu

    ie = Core()
    enc = ie.compile_model(ie.read_model(str(ENCODER_ONNX)), "CPU")
    dec = ie.compile_model(ie.read_model(str(DECODER_ONNX)), "CPU")

    image, box = load_sample()

    def infer():
        tensor = preprocess(image).unsqueeze(0).numpy().astype(np.float32)
        embed = enc({"image": tensor})["image_embeddings"]
        dec({
            "image_embeddings": embed,
            "point_coords":     np.array([[[box[0],box[1]],[box[2],box[3]],[0,0],[0,0],[0,0]]], dtype=np.float32),
            "point_labels":     np.array([[2, 3, -1, -1, -1]], dtype=np.float32),
            "mask_input":       np.zeros((1,1,256,256), dtype=np.float32),
            "has_mask_input":   np.array([0], dtype=np.float32),
            "orig_im_size":     np.array([image.shape[0], image.shape[1]], dtype=np.int64),
        })

    for _ in range(3):
        infer()
    latencies = []
    for _ in range(NUM_TRIALS):
        t0 = time.perf_counter()
        infer()
        latencies.append(time.perf_counter() - t0)
    summarise(latencies)


EXPERIMENT_MAP = {
    "pytorch":       run_pytorch,
    "onnx":          lambda: run_onnx(["CUDAExecutionProvider", "CPUExecutionProvider"]),
    "onnx_quantized":lambda: run_onnx(["CPUExecutionProvider"],
                                       ENCODER_ONNX_QUANT, DECODER_ONNX_QUANT),
    "tensorrt":      lambda: run_onnx(["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
                                      dec_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]),
    "openvino":      run_openvino,
}

if __name__ == "__main__":
    if EXPERIMENT not in EXPERIMENT_MAP:
        raise ValueError(f"Unknown EXPERIMENT='{EXPERIMENT}'. Choose from: {', '.join(EXPERIMENT_MAP)}")
    print(f"\n>>> Starting: {EXPERIMENT}")
    EXPERIMENT_MAP[EXPERIMENT]()
