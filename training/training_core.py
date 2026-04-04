"""
Shared training utilities: MLflow param flattening, TinyViT paths,
encoder distillation loss, encoder-only evaluation loop.
Used by `train.py` (single-GPU only).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_mobilesam_root(cfg: dict) -> Path:
    env = os.environ.get("MOBILESAM_ROOT") or os.environ.get("IMMICH_MS_ROOT")
    if cfg.get("mobilesam_root"):
        return Path(cfg["mobilesam_root"]).expanduser().resolve()
    if env:
        return Path(env).expanduser().resolve()
    sibling = _repo_root().parent / "MobileSAM-pytorch" / "MobileSAM"
    if sibling.is_dir():
        return sibling.resolve()
    raise FileNotFoundError(
        "TinyViT import path not found. Set mobilesam_root in YAML, or env MOBILESAM_ROOT, "
        "or place MobileSAM-pytorch next to immich-sticker-gen."
    )


def _import_tiny_vit(mobilesam_root: Path):
    root_str = str(mobilesam_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from mobile_sam.modeling.tiny_vit_sam import TinyViT

    return TinyViT


def encoder_distill_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match MobileSAM-pytorch train.py customized_mseloss."""
    return ((pred - target) ** 2).sum(dim=1).mean().sqrt()


def flatten_cfg(prefix: str, obj: Any, out: Dict[str, str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            flatten_cfg(key, v, out)
    elif isinstance(obj, (list, tuple)):
        out[prefix] = json.dumps(obj)
    elif obj is None:
        out[prefix] = "null"
    else:
        out[prefix] = str(obj)


def git_sha(repo: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
        return r.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def gpu_env_info() -> Dict[str, str]:
    info: Dict[str, str] = {
        "torch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_device_count": str(torch.cuda.device_count()) if torch.cuda.is_available() else "0",
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            info[f"gpu_{i}_name"] = torch.cuda.get_device_name(i)
    try:
        info["rocm_version"] = str(torch.version.hip) if hasattr(torch.version, "hip") and torch.version.hip else "n/a"
    except Exception:
        info["rocm_version"] = "n/a"
    return info


def _unwrap(model: nn.Module) -> nn.Module:
    return model


@torch.no_grad()
def evaluate_encoder(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Eval",
    show_progress: bool = False,
) -> Dict[str, float]:
    """Encoder eval on one process."""
    m = _unwrap(model)
    m.eval()
    total_loss = 0.0
    total_cos = 0.0
    n_batches = 0
    it = tqdm(loader, desc=desc, leave=False, disable=not show_progress)
    for imgs, target_feats, _ in it:
        imgs = imgs.to(device, non_blocking=True)
        target_feats = target_feats.to(device, non_blocking=True)
        pred = m(imgs)
        loss = encoder_distill_loss(pred, target_feats)
        pred_f = pred.flatten(1)
        tgt_f = target_feats.flatten(1)
        cos = torch.nn.functional.cosine_similarity(pred_f, tgt_f, dim=1).mean()
        total_loss += loss.item()
        total_cos += cos.item()
        n_batches += 1
        if show_progress:
            it.set_postfix(loss=f"{loss.item():.4f}", cos=f"{cos.item():.4f}")
    return {
        "loss": total_loss / max(n_batches, 1),
        "cosine_similarity": total_cos / max(n_batches, 1),
    }
