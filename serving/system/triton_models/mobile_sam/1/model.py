"""
Triton Python backend for MobileSAM.

Inputs:
  INPUT_IMAGE  – TYPE_STRING [1]   base64 JPEG
  BOX          – TYPE_FP32   [4]   [x1,y1,x2,y2]  (set all -1 to use centre point)

Outputs:
  MASK         – TYPE_STRING [1]   base64 PNG
  ENCODER_MS   – TYPE_FP32  [1]
  DECODER_MS   – TYPE_FP32  [1]
"""
import base64
import io
import json
import os
import time

import cv2
import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image

PIXEL_MEAN = np.array([123.675, 116.28,  103.53 ], dtype=np.float32)
PIXEL_STD  = np.array([ 58.395,  57.12,   57.375], dtype=np.float32)


def _preprocess(image_rgb: np.ndarray, size: int = 1024) -> np.ndarray:
    h, w   = image_rgb.shape[:2]
    scale  = size / max(h, w)
    new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded  = np.pad(resized,
                     ((0, size - new_h), (0, size - new_w), (0, 0)),
                     mode="constant").astype(np.float32)
    padded  = (padded - PIXEL_MEAN) / PIXEL_STD
    return padded.transpose(2, 0, 1)[None]   # [1,3,H,W]


class TritonPythonModel:
    def initialize(self, args):
        import onnxruntime as ort

        model_dir    = os.path.dirname(__file__)
        # ONNX files live in the shared data volume, path passed via env
        data_dir     = os.environ.get("MODEL_DIR", "/data")
        enc_path     = os.path.join(data_dir, "mobile_sam_encoder.onnx")
        dec_path     = os.path.join(data_dir, "mobile_sam_decoder.onnx")

        providers = ["CPUExecutionProvider"]
        self.enc_sess = ort.InferenceSession(enc_path, providers=providers)
        self.dec_sess = ort.InferenceSession(dec_path, providers=providers)

    def execute(self, requests):
        responses = []
        for request in requests:
            # --- unpack inputs ---
            img_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE")
            box_tensor = pb_utils.get_input_tensor_by_name(request, "BOX")

            img_b64 = img_tensor.as_numpy()[0, 0]
            if isinstance(img_b64, bytes):
                img_b64 = img_b64.decode()
            img_bytes = base64.b64decode(img_b64)
            image     = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
            orig_h, orig_w = image.shape[:2]

            box = box_tensor.as_numpy().flatten().tolist()

            # --- encoder ---
            t0 = time.perf_counter()
            tensor    = _preprocess(image)
            (embed,)  = self.enc_sess.run(["image_embeddings"], {"image": tensor})
            enc_ms    = (time.perf_counter() - t0) * 1e3

            # --- build prompts ---
            if all(v >= 0 for v in box):
                x1, y1, x2, y2 = box
                point_coords = np.array([[[x1,y1],[x2,y2],[0,0],[0,0],[0,0]]], dtype=np.float32)
                point_labels = np.array([[2, 3, -1, -1, -1]],                  dtype=np.float32)
            else:
                point_coords = np.array([[[orig_w/2, orig_h/2],[0,0],[0,0],[0,0],[0,0]]], dtype=np.float32)
                point_labels = np.array([[1, -1, -1, -1, -1]],                            dtype=np.float32)

            mask_input   = np.zeros((1,1,256,256), dtype=np.float32)
            has_mask     = np.array([0],           dtype=np.float32)
            orig_im_size = np.array([orig_h, orig_w], dtype=np.float32)

            # --- decoder ---
            t0 = time.perf_counter()
            masks, _, _ = self.dec_sess.run(
                ["masks", "iou_predictions", "low_res_masks"],
                {
                    "image_embeddings": embed,
                    "point_coords":     point_coords,
                    "point_labels":     point_labels,
                    "mask_input":       mask_input,
                    "has_mask_input":   has_mask,
                    "orig_im_size":     orig_im_size,
                },
            )
            dec_ms = (time.perf_counter() - t0) * 1e3

            # --- encode mask ---
            mask_bool = (masks[0, 0] > 0).astype(np.uint8) * 255
            buf = io.BytesIO()
            Image.fromarray(mask_bool).save(buf, format="PNG")
            mask_b64 = base64.b64encode(buf.getvalue()).decode()

            # --- pack outputs ---
            out_mask    = pb_utils.Tensor("MASK",       np.array([[mask_b64]], dtype=object))
            out_enc_ms  = pb_utils.Tensor("ENCODER_MS", np.array([[enc_ms]],   dtype=np.float32))
            out_dec_ms  = pb_utils.Tensor("DECODER_MS", np.array([[dec_ms]],   dtype=np.float32))
            responses.append(pb_utils.InferenceResponse(
                output_tensors=[out_mask, out_enc_ms, out_dec_ms]
            ))

        return responses
