#!/usr/bin/env python3
"""
Unified training entry: encoder distillation or full MobileSAM fine-tuning.
Config key: training.mode = encoder_distill | full_sam

Two-stage (distill then segment): run encoder_distill, then full_sam with model.mobile_sam_checkpoint
set to the first run's mobile_sam_full.pt (MLflow artifact or local path).

Artifacts: full MobileSAM state dict (mobile_sam_full.pt) logged to MLflow (plus split manifest).
System metrics (CPU/RAM/disk, optional GPU util) logged each epoch when psutil is installed.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import mlflow
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataset_sa1b import SA1BSamDataset, build_datasets, build_sam_loaders
from sam_utils import (
    build_sam_tiny,
    forward_sam_trainable,
    mean_iou_from_logits,
    merge_tinyvit_encoder_into_sam,
    save_sam_checkpoint,
    segmentation_loss,
    strip_module_prefix,
)
from training_core import (
    cleanup_distributed,
    encoder_distill_loss,
    evaluate_encoder,
    flatten_cfg,
    git_sha,
    gpu_env_info,
    setup_distributed,
    _import_tiny_vit,
    _repo_root,
    _resolve_mobilesam_root,
    _unwrap,
)

try:
    import psutil
except ImportError:
    psutil = None


def snapshot_system_metrics() -> Dict[str, float]:
    out: Dict[str, float] = {}
    if psutil is None:
        return out
    out["sys_cpu_percent"] = float(psutil.cpu_percent(interval=None))
    vm = psutil.virtual_memory()
    out["sys_ram_used_bytes"] = float(vm.used)
    out["sys_ram_percent"] = float(vm.percent)
    try:
        du = psutil.disk_usage("/")
        out["sys_disk_used_percent"] = float(du.percent)
    except OSError:
        pass
    return out


def try_rocm_gpu_util() -> Optional[float]:
    try:
        r = subprocess.run(
            ["rocm-smi", "--showuse"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode != 0:
            return None
        for line in r.stdout.splitlines():
            if "GPU use" in line or "GPU Utilization" in line:
                parts = line.replace("%", " ").split()
                for i, p in enumerate(parts):
                    if p.isdigit() and i > 0:
                        return float(p)
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


def log_system_metrics_mlflow(step: int, device: torch.device) -> None:
    for k, v in snapshot_system_metrics().items():
        mlflow.log_metric(k, v, step=step)
    if device.type == "cuda":
        mlflow.log_metric(
            "gpu_mem_allocated_bytes",
            float(torch.cuda.memory_allocated(device)),
            step=step,
        )
        mlflow.log_metric(
            "gpu_mem_reserved_bytes",
            float(torch.cuda.memory_reserved(device)),
            step=step,
        )
    gu = try_rocm_gpu_util()
    if gu is not None:
        mlflow.log_metric("gpu_util_percent_rocm_smi", gu, step=step)


def build_optimizer_sam(
    sam: nn.Module,
    opt_name: str,
    lr: float,
    wd: float,
    momentum: float,
) -> torch.optim.Optimizer:
    params = [p for p in sam.parameters() if p.requires_grad]
    if opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)


@torch.no_grad()
def eval_sam_loader_mean_iou(
    sam: nn.Module,
    loader: DataLoader,
    device: torch.device,
    multimask_output: bool,
) -> float:
    sam.eval()
    tot = 0.0
    n = 0
    for batch in loader:
        for b in batch:
            b["image"] = b["image"].to(device, non_blocking=True)
            b["boxes"] = b["boxes"].to(device, non_blocking=True)
            b["low_res_mask_gt"] = b["low_res_mask_gt"].to(device, non_blocking=True)
        logits, _ = forward_sam_trainable(sam, batch, multimask_output, device)
        tgt = torch.stack([b["low_res_mask_gt"] for b in batch], dim=0)
        tot += mean_iou_from_logits(logits, tgt) * len(batch)
        n += len(batch)
    return tot / max(n, 1)


def _collect_val_samples(loader: DataLoader, n: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for batch in loader:
        for b in batch:
            out.append(b)
            if len(out) >= n:
                return out
    return out


@torch.no_grad()
def log_sam_val_preview_artifacts(
    sam: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    multimask_output: bool,
    num_samples: int,
    staging_dir: Path,
) -> None:
    """
    Save validation images with box prompt, predicted mask overlay, and GT contour; log to MLflow.
    Prompt matches training: axis-aligned box from GT mask (see SA1BSamDataset).
    """
    if num_samples <= 0:
        return
    ar = mlflow.active_run()
    if ar is None:
        return

    sam.eval()
    samples = _collect_val_samples(val_loader, num_samples)
    if not samples:
        return

    ep_tag = f"epoch_{epoch:04d}"
    art_prefix = f"val_previews/{ep_tag}"
    stage = staging_dir / "mlflow_val_previews" / ep_tag
    stage.mkdir(parents=True, exist_ok=True)

    for i, rec in enumerate(samples[:num_samples]):
        img_t = rec["image"]
        box_t = rec["boxes"]
        gt_t = rec["low_res_mask_gt"]
        path_str = str(rec.get("path", f"sample_{i}"))
        stem = Path(path_str).stem

        one = [
            {
                "image": img_t.to(device, non_blocking=True),
                "original_size": rec["original_size"],
                "boxes": box_t.to(device, non_blocking=True),
                "low_res_mask_gt": gt_t.to(device, non_blocking=True),
            }
        ]
        logits, _ = forward_sam_trainable(sam, one, multimask_output, device)
        pred_lr = torch.sigmoid(logits[0, 0]).float().cpu().numpy()
        gt_lr = gt_t[0, 0].float().cpu().numpy()

        img = img_t.permute(1, 2, 0).cpu().numpy()
        img_u8 = np.clip(img, 0.0, 255.0).astype(np.uint8)
        h, w = img_u8.shape[:2]

        pred_up = cv2.resize(pred_lr, (w, h), interpolation=cv2.INTER_LINEAR)
        gt_up = cv2.resize(gt_lr, (w, h), interpolation=cv2.INTER_NEAREST)

        # BGR for cv2 drawing / imwrite
        vis = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR).astype(np.float32)
        pred_mask = (pred_up > 0.5).astype(np.float32)
        green = np.zeros_like(vis)
        green[:, :, 1] = pred_mask * 255.0
        vis = cv2.addWeighted(vis, 0.62, green, 0.38, 0)
        vis = np.clip(vis, 0, 255).astype(np.uint8)

        gt_bin = (gt_up > 0.5).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 0, 255), thickness=2)

        bx = box_t[0, 0].cpu().numpy()
        x0, y0, x1, y1 = int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 180, 255), 2)

        caption = f"{stem} | prompt: box [{x0},{y0},{x1},{y1}] (xyxy, resized)"
        cv2.putText(
            vis,
            caption[:120],
            (4, min(24, h - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        out_path = stage / f"sample_{i:02d}_{stem}.png"
        cv2.imwrite(str(out_path), vis)
        mlflow.log_artifact(str(out_path), artifact_path=art_prefix)


def train_sam_epochs(
    sam: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    train_cfg: dict,
    is_master: bool,
    world_size: int,
    multimask_output: bool,
    epoch_offset: int = 0,
    train_sampler: Optional[DistributedSampler] = None,
    staging_dir: Optional[Path] = None,
) -> None:
    show_tqdm = is_master and train_cfg.get("show_progress", True)
    log_iv = int(train_cfg.get("log_interval_batches", 50))
    global_step = epoch_offset * max(len(train_loader), 1)

    for epoch in range(1, epochs + 1):
        ep = epoch_offset + epoch
        sam.train()
        if train_sampler is not None:
            train_sampler.set_epoch(ep)
        epoch_loss = 0.0
        nb = 0
        t0 = time.perf_counter()
        bar = tqdm(train_loader, desc=f"SAM train ep {ep}", disable=not show_tqdm)
        for batch_idx, batch in enumerate(bar):
            for b in batch:
                b["image"] = b["image"].to(device, non_blocking=True)
                b["boxes"] = b["boxes"].to(device, non_blocking=True)
                b["low_res_mask_gt"] = b["low_res_mask_gt"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = forward_sam_trainable(sam, batch, multimask_output, device)
            tgt = torch.stack([b["low_res_mask_gt"] for b in batch], dim=0)
            loss = segmentation_loss(logits, tgt)
            loss.backward()
            optimizer.step()

            if world_size > 1:
                lt = loss.detach().clone()
                dist.all_reduce(lt, op=dist.ReduceOp.SUM)
                loss_reduced = lt / world_size
            else:
                loss_reduced = loss.detach()

            epoch_loss += loss_reduced.item()
            nb += 1
            global_step += 1

            if is_master and (batch_idx + 1) % log_iv == 0:
                mlflow.log_metric("sam_train_loss_batch", loss_reduced.item(), step=global_step)

            if show_tqdm:
                bar.set_postfix(loss=f"{loss_reduced.item():.4f}")

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.perf_counter() - t0
        if is_master:
            mlflow.log_metric("sam_train_loss_epoch", epoch_loss / max(nb, 1), step=ep)
            mlflow.log_metric("sam_epoch_time_sec", epoch_time, step=ep)
            log_system_metrics_mlflow(ep, device)
            if val_loader is not None:
                viou = eval_sam_loader_mean_iou(sam, val_loader, device, multimask_output)
                mlflow.log_metric("val_mean_iou_lowres", viou, step=ep)
                n_prev = int(train_cfg.get("val_preview_samples", 3))
                if n_prev > 0 and staging_dir is not None:
                    log_sam_val_preview_artifacts(
                        sam,
                        val_loader,
                        device,
                        ep,
                        multimask_output,
                        n_prev,
                        staging_dir,
                    )

        if world_size > 1:
            dist.barrier()


def run_encoder_distill(
    cfg: dict,
    cfg_path: Path,
    mobilesam_root: Path,
    device: torch.device,
    rank: int,
    world_size: int,
    local_rank: int,
    is_master: bool,
    backend: str,
) -> None:
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    out_cfg = cfg.get("output", {})
    output_dir = Path(out_cfg.get("dir", "./training_outputs")).expanduser()
    if is_master:
        output_dir.mkdir(parents=True, exist_ok=True)

    split_out = output_dir / "split_manifest.json"
    manifest_in = data_cfg.get("split_manifest")
    manifest_path = Path(manifest_in).resolve() if manifest_in else None

    _, train_ds, val_ds, test_ds, split_meta, jpg_splits = build_datasets(
        data_cfg=data_cfg,
        img_size=int(data_cfg.get("image_size", 1024)),
        seed=int(data_cfg.get("seed", 42)),
        train_frac=float(data_cfg.get("train_frac", 0.7)),
        val_frac=float(data_cfg.get("val_frac", 0.1)),
        test_frac=float(data_cfg.get("test_frac", 0.2)),
        split_manifest=manifest_path,
        split_manifest_out=split_out if is_master and manifest_path is None else None,
    )

    batch_size = int(train_cfg.get("batch_size", 8))
    per_gpu = max(batch_size // world_size, 1)
    num_workers = int(data_cfg.get("num_workers", 4))

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, shuffle=True, seed=int(data_cfg.get("seed", 42))
        )
    else:
        train_sampler = None

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=per_gpu,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(per_gpu, 1),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=max(per_gpu, 1),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    TinyViT = _import_tiny_vit(mobilesam_root)
    model_cfg = cfg["model"]
    model = TinyViT(
        img_size=int(model_cfg.get("img_size", 1024)),
        in_chans=int(model_cfg.get("in_chans", 3)),
        num_classes=int(model_cfg.get("num_classes", 1000)),
        embed_dims=list(model_cfg.get("embed_dims", [64, 128, 160, 320])),
        depths=list(model_cfg.get("depths", [2, 2, 6, 2])),
        num_heads=list(model_cfg.get("num_heads", [2, 4, 5, 10])),
        window_sizes=list(model_cfg.get("window_sizes", [7, 7, 14, 7])),
        mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
        drop_rate=float(model_cfg.get("drop_rate", 0.0)),
        drop_path_rate=float(model_cfg.get("drop_path_rate", 0.0)),
        use_checkpoint=bool(model_cfg.get("use_checkpoint", False)),
        mbconv_expand_ratio=float(model_cfg.get("mbconv_expand_ratio", 4.0)),
        local_conv_size=int(model_cfg.get("local_conv_size", 3)),
        layer_lr_decay=float(model_cfg.get("layer_lr_decay", 0.8)),
    )

    model = model.to(device)
    if world_size > 1 and torch.cuda.is_available():
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    opt_name = train_cfg.get("optimizer", "sgd").lower()
    lr = float(train_cfg.get("learning_rate", 0.05))
    wd = float(train_cfg.get("weight_decay", 5e-4))
    momentum = float(train_cfg.get("momentum", 0.9))
    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(train_cfg.get("scheduler_gamma", 0.5))
    )

    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or mlflow_cfg.get(
        "tracking_uri", "http://127.0.0.1:5000"
    )
    experiment_name = mlflow_cfg.get("experiment_name", "immich-sticker-encoder")
    run_name = mlflow_cfg.get("run_name")

    sha = git_sha(_repo_root())
    flat_params: Dict[str, str] = {}
    cfg_flat = {k: v for k, v in cfg.items() if k != "mobilesam_root"}
    flatten_cfg(
        "",
        {**cfg_flat, "git_sha": sha, "world_size": world_size, "backend": backend},
        flat_params,
    )

    t_wall_start = time.perf_counter()
    peak_mem = 0
    epochs = int(train_cfg.get("epochs", 8))
    log_interval = int(train_cfg.get("log_interval_batches", 50))
    show_tqdm = is_master and train_cfg.get("show_progress", True)

    if is_master:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)
        for k, v in flat_params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("config_path", str(cfg_path))
        mlflow.log_param("mobilesam_root", str(mobilesam_root))
        for k, v in split_meta.items():
            if k != "counts":
                mlflow.log_param(f"split_{k}", str(v))
        if "counts" in split_meta:
            for sk, sv in split_meta["counts"].items():
                mlflow.log_param(f"split_count_{sk}", sv)
        for k, v in gpu_env_info().items():
            mlflow.log_param(f"env_{k}", v)
        if split_out.is_file():
            mlflow.log_artifact(str(split_out), artifact_path="split")
        log_system_metrics_mlflow(0, device)

    try:
        epoch_bar = tqdm(
            range(1, epochs + 1),
            desc="Epochs",
            disable=not show_tqdm,
        )
        for epoch in epoch_bar:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            epoch_loss = 0.0
            epoch_batches = 0
            t_epoch = time.perf_counter()

            batch_bar = tqdm(
                train_loader,
                desc=f"Train ep {epoch}/{epochs}",
                leave=False,
                disable=not show_tqdm,
            )
            for batch_idx, (imgs, target_feats, _) in enumerate(batch_bar):
                imgs = imgs.to(device, non_blocking=True)
                target_feats = target_feats.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                pred = model(imgs)
                loss = encoder_distill_loss(pred, target_feats)
                loss.backward()
                optimizer.step()

                if world_size > 1:
                    lt = loss.detach().clone()
                    dist.all_reduce(lt, op=dist.ReduceOp.SUM)
                    loss_reduced = lt / world_size
                else:
                    loss_reduced = loss.detach()

                epoch_loss += loss_reduced.item()
                epoch_batches += 1

                if torch.cuda.is_available():
                    peak_mem = max(peak_mem, torch.cuda.max_memory_allocated(device))

                if is_master and (batch_idx + 1) % log_interval == 0:
                    mlflow.log_metric(
                        "train_loss_batch",
                        loss_reduced.item(),
                        step=(epoch - 1) * len(train_loader) + batch_idx,
                    )
                if show_tqdm:
                    batch_bar.set_postfix(loss=f"{loss_reduced.item():.4f}")

            scheduler.step()
            epoch_time = time.perf_counter() - t_epoch
            train_loss_epoch = epoch_loss / max(epoch_batches, 1)

            if is_master:
                mlflow.log_metric("train_loss_epoch", train_loss_epoch, step=epoch)
                mlflow.log_metric("epoch_time_sec", epoch_time, step=epoch)
                mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=epoch)
                log_system_metrics_mlflow(epoch, device)

            if world_size > 1:
                dist.barrier()
            if is_master:
                val_metrics = evaluate_encoder(
                    model,
                    val_loader,
                    device,
                    desc=f"Val ep {epoch}",
                    show_progress=show_tqdm,
                )
                mlflow.log_metric("val_embedding_loss", val_metrics["loss"], step=epoch)
                mlflow.log_metric(
                    "val_cosine_similarity", val_metrics["cosine_similarity"], step=epoch
                )
            if world_size > 1:
                dist.barrier()

        if world_size > 1:
            dist.barrier()

        if is_master:
            test_metrics = evaluate_encoder(
                model,
                test_loader,
                device,
                desc="Test (held-out)",
                show_progress=show_tqdm,
            )
        else:
            test_metrics = {"loss": 0.0, "cosine_similarity": 0.0}

        total_time = time.perf_counter() - t_wall_start

        if is_master:
            mlflow.log_metric("test_embedding_loss", test_metrics["loss"])
            mlflow.log_metric("test_cosine_similarity", test_metrics["cosine_similarity"])
            mlflow.log_metric("total_train_time_sec", total_time)
            mlflow.log_metric("peak_cuda_memory_bytes", float(peak_mem))
            mlflow.log_metric("epochs_completed", float(epochs))

            ckpt_enc = output_dir / "tinyvit_encoder_only.pth"
            st = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            torch.save(strip_module_prefix(st), ckpt_enc)

            sam_ckpt = cfg.get("model", {}).get("mobile_sam_checkpoint")
            if not sam_ckpt:
                raise ValueError(
                    "encoder_distill requires model.mobile_sam_checkpoint to assemble full MobileSAM for MLflow."
                )
            sam = build_sam_tiny(mobilesam_root, sam_ckpt, device)
            merge_tinyvit_encoder_into_sam(sam, strip_module_prefix(st), strict=False)
            sam.eval()

            multimask = bool(cfg.get("sam", {}).get("multimask_output", False))
            if data_cfg.get("annotation_root"):
                _, _, sam_test = build_sam_loaders(
                    data_cfg,
                    jpg_splits,
                    batch_size=int(train_cfg.get("eval_batch_size", train_cfg.get("batch_size", 8))),
                    num_workers=num_workers,
                    distributed_sampler_train=None,
                    device_is_cuda=torch.cuda.is_available(),
                )
                if sam_test is not None:
                    tiou = eval_sam_loader_mean_iou(sam, sam_test, device, multimask)
                    mlflow.log_metric("test_mean_iou_lowres", tiou)

            full_path = output_dir / "mobile_sam_full.pt"
            save_sam_checkpoint(full_path, sam)
            mlflow.log_artifact(str(full_path), artifact_path="checkpoints")

    finally:
        if is_master and mlflow.active_run():
            mlflow.end_run()
        cleanup_distributed()


def _sam_collate(batch: List[Any]) -> List[Any]:
    return batch


def run_full_sam(
    cfg: dict,
    cfg_path: Path,
    mobilesam_root: Path,
    device: torch.device,
    rank: int,
    world_size: int,
    local_rank: int,
    is_master: bool,
    backend: str,
) -> None:
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    if not data_cfg.get("annotation_root"):
        raise ValueError("full_sam requires data.annotation_root with mask JSON per image.")
    out_cfg = cfg.get("output", {})
    output_dir = Path(out_cfg.get("dir", "./training_outputs")).expanduser()
    if is_master:
        output_dir.mkdir(parents=True, exist_ok=True)

    split_out = output_dir / "split_manifest.json"
    manifest_in = data_cfg.get("split_manifest")
    manifest_path = Path(manifest_in).resolve() if manifest_in else None

    _, _, _, _, split_meta, jpg_splits = build_datasets(
        data_cfg=data_cfg,
        img_size=int(data_cfg.get("image_size", 1024)),
        seed=int(data_cfg.get("seed", 42)),
        train_frac=float(data_cfg.get("train_frac", 0.7)),
        val_frac=float(data_cfg.get("val_frac", 0.1)),
        test_frac=float(data_cfg.get("test_frac", 0.2)),
        split_manifest=manifest_path,
        split_manifest_out=split_out if is_master and manifest_path is None else None,
    )

    batch_size = int(train_cfg.get("batch_size", 4))
    per_gpu = max(batch_size // world_size, 1)
    num_workers = int(data_cfg.get("num_workers", 4))

    train_ds = SA1BSamDataset(jpg_splits["train"], data_cfg)
    val_ds = SA1BSamDataset(jpg_splits["val"], data_cfg)
    test_ds = SA1BSamDataset(jpg_splits["test"], data_cfg)
    train_sampler: Optional[DistributedSampler] = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, shuffle=True, seed=int(data_cfg.get("seed", 42))
        )
    sam_train = DataLoader(
        train_ds,
        batch_size=per_gpu,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=_sam_collate,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    sam_val = DataLoader(
        val_ds,
        batch_size=per_gpu,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_sam_collate,
        pin_memory=torch.cuda.is_available(),
    )
    sam_test = DataLoader(
        test_ds,
        batch_size=per_gpu,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_sam_collate,
        pin_memory=torch.cuda.is_available(),
    )

    sam_ckpt = cfg.get("model", {}).get("mobile_sam_checkpoint")
    sam = build_sam_tiny(mobilesam_root, sam_ckpt, device)
    if world_size > 1 and torch.cuda.is_available():
        try:
            sam = nn.SyncBatchNorm.convert_sync_batchnorm(sam)
        except Exception:
            pass
        sam = DDP(sam, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    opt_name = train_cfg.get("optimizer", "adamw").lower()
    lr = float(train_cfg.get("learning_rate", 1e-4))
    wd = float(train_cfg.get("weight_decay", 0.01))
    momentum = float(train_cfg.get("momentum", 0.9))
    optimizer = build_optimizer_sam(sam, opt_name, lr, wd, momentum)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(train_cfg.get("scheduler_gamma", 0.9))
    )

    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or mlflow_cfg.get(
        "tracking_uri", "http://127.0.0.1:5000"
    )
    experiment_name = mlflow_cfg.get("experiment_name", "immich-sticker-sam")
    run_name = mlflow_cfg.get("run_name")
    sha = git_sha(_repo_root())
    flat_params: Dict[str, str] = {}
    cfg_flat = {k: v for k, v in cfg.items() if k != "mobilesam_root"}
    flatten_cfg(
        "",
        {**cfg_flat, "git_sha": sha, "world_size": world_size, "backend": backend},
        flat_params,
    )
    multimask = bool(cfg.get("sam", {}).get("multimask_output", False))
    epochs = int(train_cfg.get("epochs", 8))

    if is_master:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)
        for k, v in flat_params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("config_path", str(cfg_path))
        for k, v in split_meta.items():
            if k != "counts":
                mlflow.log_param(f"split_{k}", str(v))
        if "counts" in split_meta:
            for sk, sv in split_meta["counts"].items():
                mlflow.log_param(f"split_count_{sk}", sv)
        for k, v in gpu_env_info().items():
            mlflow.log_param(f"env_{k}", v)
        if split_out.is_file():
            mlflow.log_artifact(str(split_out), artifact_path="split")
        log_system_metrics_mlflow(0, device)

    try:
        train_sam_epochs(
            sam,
            sam_train,
            sam_val,
            device,
            epochs,
            optimizer,
            scheduler,
            train_cfg,
            is_master,
            world_size,
            multimask,
            epoch_offset=0,
            train_sampler=train_sampler,
            staging_dir=output_dir if is_master else None,
        )
        if world_size > 1:
            dist.barrier()
        if is_master and sam_test is not None:
            tiou = eval_sam_loader_mean_iou(_unwrap(sam), sam_test, device, multimask)
            mlflow.log_metric("test_mean_iou_lowres", tiou)
            full_path = output_dir / "mobile_sam_full.pt"
            save_sam_checkpoint(full_path, sam)
            mlflow.log_artifact(str(full_path), artifact_path="checkpoints")
    finally:
        if is_master and mlflow.active_run():
            mlflow.end_run()
        cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified MobileSAM / TinyViT training")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg_path = Path(args.config).resolve()
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_top = cfg.get("training", {})
    mode = train_top.get("mode", "encoder_distill")

    train_cfg = cfg.get("train", {})
    backend = train_cfg.get("distributed_backend", "nccl")
    rank, world_size, local_rank = setup_distributed(backend)
    is_master = rank == 0

    mobilesam_root = _resolve_mobilesam_root(cfg)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if mode == "encoder_distill":
        run_encoder_distill(
            cfg, cfg_path, mobilesam_root, device, rank, world_size, local_rank, is_master, backend
        )
    elif mode == "full_sam":
        run_full_sam(
            cfg, cfg_path, mobilesam_root, device, rank, world_size, local_rank, is_master, backend
        )
    else:
        raise ValueError(f"Unknown training.mode: {mode}")


if __name__ == "__main__":
    main()
