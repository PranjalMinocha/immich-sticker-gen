#!/usr/bin/env python3
"""
Export MobileSAM encoder and decoder to ONNX.
Run once via the init container before starting servers.
"""
import os, warnings
warnings.filterwarnings("ignore")

import torch
from pathlib import Path

MODEL_DIR    = Path(os.environ.get("MODEL_DIR", "/data"))
CKPT_PATH    = MODEL_DIR / "mobile_sam.pt"
ENCODER_ONNX = MODEL_DIR / "mobile_sam_encoder.onnx"
DECODER_ONNX = MODEL_DIR / "mobile_sam_decoder.onnx"


def export_encoder(sam, path: Path):
    if path.exists():
        print(f"Encoder already exists: {path}")
        return
    print(f"Exporting encoder → {path}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            sam.image_encoder,
            torch.randn(1, 3, 1024, 1024),
            str(path),
            input_names=["image"],
            output_names=["image_embeddings"],
            opset_version=17,

        )
    print("  done.")


def export_decoder(sam, path: Path):
    if path.exists():
        print(f"Decoder already exists: {path}")
        return
    print(f"Exporting decoder → {path}")
    from mobile_sam.utils.onnx import SamOnnxModel
    onnx_model = SamOnnxModel(sam, return_single_mask=True).eval()
    embed_dim  = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    dummy = (
        torch.randn(1, embed_dim, *embed_size),
        torch.randint(0, 1024, (1, 5, 2), dtype=torch.float32),
        torch.randint(0, 4,    (1, 5),    dtype=torch.float32),
        torch.randn(1, 1, 256, 256),
        torch.tensor([1], dtype=torch.float32),
        torch.tensor([1024, 1024], dtype=torch.float32),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            onnx_model, dummy, str(path),
            input_names=["image_embeddings","point_coords","point_labels",
                         "mask_input","has_mask_input","orig_im_size"],
            output_names=["masks","iou_predictions","low_res_masks"],
            dynamic_axes={"point_coords":{1:"num_points"},"point_labels":{1:"num_points"}},
            opset_version=17,

        )
    print("  done.")


if __name__ == "__main__":
    from mobile_sam import sam_model_registry
    print(f"Loading checkpoint: {CKPT_PATH}")
    sam = sam_model_registry["vit_t"](checkpoint=str(CKPT_PATH))
    sam.eval()
    export_encoder(sam, ENCODER_ONNX)
    export_decoder(sam, DECODER_ONNX)
    print("Export complete.")