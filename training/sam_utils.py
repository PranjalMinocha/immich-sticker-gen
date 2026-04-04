"""
MobileSAM `Sam` helpers: trainable forward (official `Sam.forward` is @torch.no_grad),
losses, IoU, merging a trained TinyViT state dict into a full checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def import_build_sam_vit_t(mobilesam_root: Path):
    root_str = str(mobilesam_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from mobile_sam.build_sam import build_sam_vit_t

    return build_sam_vit_t


def build_sam_tiny(
    mobilesam_root: Path,
    checkpoint_path: str | None,
    device: torch.device,
) -> torch.nn.Module:
    build = import_build_sam_vit_t(mobilesam_root)
    kw: Dict[str, Any] = {}
    if checkpoint_path:
        kw["checkpoint"] = checkpoint_path
    sam = build(**kw)
    return sam.to(device)


def forward_sam_trainable(
    sam: nn.Module,
    batched_input: List[Dict[str, Any]],
    multimask_output: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """
    Same logic as mobile_sam.modeling.sam.Sam.forward but gradients enabled.
    Each batch element must have: image (3,H,W float), original_size (h,w),
    boxes (1,4) optional, point_coords/point_labels optional,
    low_res_mask_gt (1,256,256) float in [0,1] for loss (optional for inference).
    """
    input_images = torch.stack(
        [sam.preprocess(x["image"].to(device)) for x in batched_input], dim=0
    )
    image_embeddings = sam.image_encoder(input_images)
    low_res_list: List[torch.Tensor] = []
    raw_outputs: List[Dict[str, torch.Tensor]] = []

    for image_record, curr_embedding in zip(batched_input, image_embeddings):
        if "point_coords" in image_record:
            pc = image_record["point_coords"].to(device)
            pl = image_record["point_labels"].to(device)
            points = (pc, pl)
        else:
            points = None
        bx = image_record.get("boxes", None)
        if bx is not None:
            bx = bx.to(device)
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=points,
            boxes=bx,
            masks=image_record.get("mask_inputs", None),
        )
        low_res_masks, iou_predictions = sam.mask_decoder(
            image_embeddings=curr_embedding.unsqueeze(0),
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        raw_outputs.append(
            {"low_res_logits": low_res_masks, "iou_predictions": iou_predictions}
        )
        # multimask_output False -> expect (B_prompt, 1, 256, 256)
        low_res_list.append(low_res_masks[:, 0:1])

    stacked = torch.cat(low_res_list, dim=0)
    return stacked, raw_outputs


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    t = targets
    inter = (probs * t).sum(dim=(1, 2, 3))
    denom = probs.pow(2).sum(dim=(1, 2, 3)) + t.pow(2).sum(dim=(1, 2, 3)) + eps
    return 1.0 - (2.0 * inter + eps) / denom


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """BCE + mean Dice on low-res (B,1,256,256)."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    dice = dice_loss_with_logits(logits, targets).mean()
    return bce + dice


@torch.no_grad()
def mean_iou_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.0,
) -> float:
    pred = (logits > threshold).float()
    t = (targets > 0.5).float()
    inter = (pred * t).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3)) - inter + 1e-6
    return float((inter / union).mean().item())


def strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        out[nk] = v
    return out


def merge_tinyvit_encoder_into_sam(
    sam: nn.Module,
    encoder_state: Dict[str, torch.Tensor],
    strict: bool = False,
) -> None:
    enc = strip_module_prefix(encoder_state)
    sam.image_encoder.load_state_dict(enc, strict=strict)


def save_sam_checkpoint(path: Path, sam: nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    m = sam.module if hasattr(sam, "module") else sam
    torch.save(m.state_dict(), path)
