"""
Triton Python backend for MobileSAM — NVIDIA GPU (onnxruntime-gpu).

Inputs:
  INPUT_IMAGE  TYPE_STRING [1]  base64 JPEG
  BOX          TYPE_FP32   [4]  [x1,y1,x2,y2]  (all -1 = centre point fallback)

Outputs:
  MASK         TYPE_STRING [1]  base64 PNG
  ENCODER_MS   TYPE_FP32   [1]
  DECODER_MS   TYPE_FP32   [1]
"""
import base64, io, os, time
import cv2
import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image

PIXEL_MEAN = np.array([123.675, 116.28,  103.53], dtype=np.float32)
PIXEL_STD  = np.array([ 58.395,  57.12,  57.375], dtype=np.float32)
PROVIDERS  = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _preprocess(image_rgb: np.ndarray, size: int = 1024) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded  = np.pad(resized,
                     ((0, size - new_h), (0, size - new_w), (0, 0)),
                     mode="constant").astype(np.float32)
    return ((padded - PIXEL_MEAN) / PIXEL_STD).transpose(2, 0, 1)[None]


class TritonPythonModel:
    def initialize(self, args):
        import onnxruntime as ort
        device_id = int(args.get("model_instance_device_id", 0))
        data_dir  = os.environ.get("MODEL_DIR", "/data")
        opts      = [{"device_id": device_id, "gpu_mem_limit": 2 * 1024 * 1024 * 1024}, {}]
        self.enc_sess = ort.InferenceSession(
            os.path.join(data_dir, "mobile_sam_encoder.onnx"),
            providers=PROVIDERS, provider_options=opts)
        self.dec_sess = ort.InferenceSession(
            os.path.join(data_dir, "mobile_sam_decoder.onnx"),
            providers=PROVIDERS, provider_options=opts)

    def execute(self, requests):
        responses = []
        for request in requests:
            img_b64 = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE").as_numpy()[0, 0]
            if isinstance(img_b64, bytes):
                img_b64 = img_b64.decode()
            image = np.array(Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB"))
            orig_h, orig_w = image.shape[:2]
            box = pb_utils.get_input_tensor_by_name(request, "BOX").as_numpy().flatten().tolist()

            t0 = time.perf_counter()
            (embed,) = self.enc_sess.run(["image_embeddings"], {"image": _preprocess(image)})
            enc_ms = (time.perf_counter() - t0) * 1e3

            scale = 1024 / max(orig_h, orig_w)
            if all(v >= 0 for v in box):
                x1, y1, x2, y2 = [c * scale for c in box]
                point_coords = np.array([[[x1,y1],[x2,y2],[0,0],[0,0],[0,0]]], dtype=np.float32)
                point_labels = np.array([[2, 3, -1, -1, -1]],                  dtype=np.float32)
            else:
                point_coords = np.array([[[orig_w/2*scale,orig_h/2*scale],[0,0],[0,0],[0,0],[0,0]]], dtype=np.float32)
                point_labels = np.array([[1, -1, -1, -1, -1]],                                       dtype=np.float32)

            t0 = time.perf_counter()
            masks, _, _ = self.dec_sess.run(
                ["masks", "iou_predictions", "low_res_masks"],
                {
                    "image_embeddings": embed,
                    "point_coords":     point_coords,
                    "point_labels":     point_labels,
                    "mask_input":       np.zeros((1,1,256,256), dtype=np.float32),
                    "has_mask_input":   np.array([0],           dtype=np.float32),
                    "orig_im_size":     np.array([orig_h, orig_w], dtype=np.float32),
                },
            )
            dec_ms = (time.perf_counter() - t0) * 1e3

            buf = io.BytesIO()
            Image.fromarray((masks[0,0] > 0).astype(np.uint8) * 255).save(buf, format="PNG")
            mask_b64 = base64.b64encode(buf.getvalue()).decode()

            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("MASK",       np.array([[mask_b64]], dtype=object)),
                pb_utils.Tensor("ENCODER_MS", np.array([[enc_ms]],   dtype=np.float32)),
                pb_utils.Tensor("DECODER_MS", np.array([[dec_ms]],   dtype=np.float32)),
            ]))
        return responses