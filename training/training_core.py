"""
Shared training utilities: MLflow param flattening, TinyViT paths,
encoder distillation loss, encoder-only evaluation loop.
Used by `train.py` (single-GPU only).
"""
from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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


# AMD ROCm: MLflow model metrics (pyrsmi and/or rocm-smi CLI; Chameleon / Docker).
_rocm_smi_initialized = False
_gpu_util_backend: str = "none"  # pyrsmi | rocm_smi_cli | none
_gpu_util_init_detail: str = ""


def gpu_util_logging_status() -> Tuple[str, str]:
    """(backend, detail) for MLflow params — detail explains failures or fallback choice."""
    return _gpu_util_backend, _gpu_util_init_detail


def _rocm_smi_shutdown() -> None:
    global _rocm_smi_initialized
    if not _rocm_smi_initialized:
        return
    try:
        from pyrsmi import rocml

        rocml.smi_shutdown()
    except Exception:
        pass
    _rocm_smi_initialized = False


def _rocm_smi_cli_path() -> Optional[str]:
    return shutil.which("rocm-smi") or shutil.which("rocm_smi")


def _rocm_smi_cli_responds() -> bool:
    exe = _rocm_smi_cli_path()
    if not exe:
        return False
    try:
        r = subprocess.run(
            [exe, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return r.returncode == 0
    except Exception:
        return False


def _parse_rocm_smi_showuse(stdout: str, dev_idx: int) -> Optional[float]:
    """Parse `rocm-smi --showuse` text; return GPU use %% for dev_idx."""
    pat = re.compile(r"GPU\[(\d+)\][^\n]*GPU use \(%\):\s*(\d+)", re.IGNORECASE)
    by_id: Dict[int, float] = {}
    for m in pat.finditer(stdout):
        by_id[int(m.group(1))] = float(m.group(2))
    if dev_idx in by_id:
        return by_id[dev_idx]
    if by_id:
        return by_id.get(0, next(iter(by_id.values())))
    return None


def _sample_gpu_util_rocm_smi_cli(dev_idx: int) -> Optional[float]:
    exe = _rocm_smi_cli_path()
    if not exe:
        return None
    try:
        r = subprocess.run(
            [exe, "-d", str(dev_idx), "--showuse"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        text = (r.stdout or "") + (r.stderr or "")
        val = _parse_rocm_smi_showuse(text, dev_idx)
        if val is not None:
            return val
        r2 = subprocess.run(
            [exe, "--showuse"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        text2 = (r2.stdout or "") + (r2.stderr or "")
        return _parse_rocm_smi_showuse(text2, dev_idx)
    except Exception:
        return None


def init_rocm_smi_for_gpu_util_logging() -> bool:
    """
    Prefer pyrsmi (libamd_smi); if that fails (common in minimal Docker), fall back to
    ``rocm-smi --showuse`` subprocess parsing.
    """
    global _rocm_smi_initialized, _gpu_util_backend, _gpu_util_init_detail

    if _rocm_smi_initialized:
        _gpu_util_backend = "pyrsmi"
        return True
    if _gpu_util_backend == "rocm_smi_cli":
        return True

    try:
        from pyrsmi import rocml

        rocml.smi_initialize()
        _rocm_smi_initialized = True
        _gpu_util_backend = "pyrsmi"
        atexit.register(_rocm_smi_shutdown)
        return True
    except Exception as e:
        _gpu_util_init_detail = f"pyrsmi_init_failed: {type(e).__name__}: {e}"

    if _rocm_smi_cli_responds():
        _gpu_util_backend = "rocm_smi_cli"
        t = _gpu_util_init_detail
        _gpu_util_init_detail = (t + "; " if t else "") + "using_rocm_smi_cli_for_gpu_util_percent"
        return True

    _gpu_util_init_detail = (
        (_gpu_util_init_detail + "; ") if _gpu_util_init_detail else ""
    ) + "no_rocm_smi_in_path_or_rocm_smi_version_failed"
    return False


def sample_gpu_utilization_percent(device: torch.device) -> Optional[float]:
    """
    GFX utilization (0-100) for the training GPU (pyrsmi or rocm-smi CLI).
    Call after torch.cuda.synchronize() for a meaningful sample.
    """
    if device.type != "cuda":
        return None
    idx = device.index if device.index is not None else 0
    return sample_gpu_utilization_by_idx(idx)


def sample_gpu_utilization_by_idx(idx: int) -> Optional[float]:
    """GFX utilization (0-100) for a specific GPU index."""
    if _gpu_util_backend == "rocm_smi_cli":
        return _sample_gpu_util_rocm_smi_cli(idx)
    if not _rocm_smi_initialized:
        return None
    try:
        from pyrsmi import rocml

        u = rocml.smi_get_device_utilization(idx)
        if u is None or u < 0:
            return None
        return float(u)
    except Exception:
        return None


def sample_all_gpu_utilization() -> Dict[int, float]:
    """Sample utilization for all available GPUs. Returns dict of {gpu_idx: utilization}."""
    result = {}
    if not torch.cuda.is_available():
        return result
    num_gpus = torch.cuda.device_count()
    for idx in range(num_gpus):
        util = sample_gpu_utilization_by_idx(idx)
        if util is not None:
            result[idx] = util
    return result


def sample_gpu_memory_utilization_percent(device: torch.device) -> Optional[float]:
    """VRAM memory-busy percent (0–100) via pyrsmi only; None with CLI-only backend."""
    if device.type != "cuda" or _gpu_util_backend != "pyrsmi":
        return None
    if not _rocm_smi_initialized:
        return None
    try:
        from pyrsmi import rocml

        idx = device.index if device.index is not None else 0
        u = rocml.smi_get_device_memory_busy(idx)
        if u is None or u < 0:
            return None
        return float(u)
    except Exception:
        return None


def torch_cuda_memory_mib(device: torch.device) -> Tuple[Optional[float], Optional[float]]:
    """PyTorch HIP/CUDA memory (always loggable when training on GPU). Returns (allocated, reserved) MiB."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None, None
    alloc = float(torch.cuda.memory_allocated(device)) / (1024.0 * 1024.0)
    reserved = float(torch.cuda.memory_reserved(device)) / (1024.0 * 1024.0)
    return alloc, reserved


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
    """Encoder eval on one process (full loader)."""
    m = _unwrap(model)
    m.eval()
    total_loss = 0.0
    total_cos = 0.0
    n_batches = 0
    total = len(loader) if hasattr(loader, "__len__") else None
    it = tqdm(loader, desc=desc, leave=False, disable=not show_progress, total=total)
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
