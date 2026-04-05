#!/usr/bin/env python3
"""
Unified training entry: encoder distillation or full MobileSAM fine-tuning (single GPU only).
Config key: training.mode = encoder_distill | full_sam

Two-stage (distill then segment): run encoder_distill, then full_sam with training.pretrained_checkpoint_path
set to the first run's mobile_sam_full.pt (MLflow artifact or local path).
Weight loading: training.use_pretrained + training.pretrained_checkpoint_path only.
Student TinyViT in encoder_distill is always random-init unless extended elsewhere.

Artifacts: full MobileSAM state dict (mobile_sam_full.pt) logged to MLflow (plus split manifest).
System metrics (CPU/RAM/disk/GPU) use MLflow's collector (see start_run(log_system_metrics=True));
install **pyrsmi** on AMD/ROCm so GPU appears in the System metrics tab.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_sa1b import (
    SAM_ENCODER_PAD_SIDE,
    SA1BSamDataset,
    build_datasets,
    build_sam_loaders,
)
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
    encoder_distill_loss,
    evaluate_encoder,
    flatten_cfg,
    git_sha,
    gpu_env_info,
    _import_tiny_vit,
    _repo_root,
    _resolve_mobilesam_root,
)

def resolve_pretrained_checkpoint(cfg: dict) -> Optional[str]:
    """
    training.use_pretrained (default True) and training.pretrained_checkpoint_path.
    Returns None when use_pretrained is false (random-init SAM scaffold).
    """
    train_top = cfg.get("training") or {}
    use_pt = bool(train_top.get("use_pretrained", True))
    path = train_top.get("pretrained_checkpoint_path")
    if use_pt:
        if not path:
            raise ValueError(
                "training.use_pretrained is true (default) but training.pretrained_checkpoint_path is not set."
            )
        return str(path)
    return None


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


def _low_res_sam_to_unpadded_hw(
    low_res_hw: np.ndarray,
    unpadded_h: int,
    unpadded_w: int,
    *,
    full_side: int = SAM_ENCODER_PAD_SIDE,
    interpolation: int,
) -> np.ndarray:
    """
    SAM low-res masks (256²) align to the padded square input to the image encoder (1024²).
    Dataset images are nh×nw in the top-left of that square; map mask to the same layout.
    """
    lr = low_res_hw.shape[0]
    if low_res_hw.shape[1] != lr:
        raise ValueError(f"Expected square low-res mask, got {low_res_hw.shape}")
    up = cv2.resize(
        low_res_hw,
        (full_side, full_side),
        interpolation=interpolation,
    )
    return up[:unpadded_h, :unpadded_w]


@torch.no_grad()
def log_sam_val_preview_artifacts(
    sam: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    multimask_output: bool,
    num_samples: int,
    staging_dir: Path,
    preview_artifact_prefix: str = "val_previews",
) -> None:
    """
    Save validation images with box prompt, predicted mask overlay, and GT contour; log to MLflow.
    Prompt matches training: COCO bbox on the instance (or tight box on instance mask); see SA1BSamDataset.
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
    safe_stem = preview_artifact_prefix.strip("/").replace("/", "_")
    art_prefix = f"{preview_artifact_prefix}/{ep_tag}"
    stage = staging_dir / "mlflow_val_previews" / safe_stem / ep_tag
    stage.mkdir(parents=True, exist_ok=True)

    for i, rec in enumerate(samples[:num_samples]):
        img_t = rec["image"]
        box_t = rec["boxes"]
        gt_t = rec["low_res_mask_gt"]
        path_str = str(rec.get("path", f"sample_{i}"))
        stem = Path(path_str).stem
        ann_idx = rec.get("ann_idx")

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

        pred_up = _low_res_sam_to_unpadded_hw(
            pred_lr, h, w, interpolation=cv2.INTER_LINEAR
        )
        gt_up = _low_res_sam_to_unpadded_hw(
            gt_lr, h, w, interpolation=cv2.INTER_NEAREST
        )

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

        ann_part = f" ann={ann_idx}" if ann_idx is not None else ""
        caption = f"{stem}{ann_part} | box [{x0},{y0},{x1},{y1}] (xyxy, resized)"
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

        suffix = f"_a{ann_idx}" if ann_idx is not None else ""
        out_path = stage / f"sample_{i:02d}_{stem}{suffix}.png"
        cv2.imwrite(str(out_path), vis)
        mlflow.log_artifact(str(out_path), artifact_path=art_prefix)


@torch.no_grad()
def log_encoder_merged_sam_val_previews(
    student: nn.Module,
    mobilesam_root: Path,
    base_sam_checkpoint: Optional[str],
    sam_val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    multimask_output: bool,
    num_samples: int,
    staging_dir: Path,
) -> None:
    """
    Merge current TinyViT into a MobileSAM scaffold (pretrained .pt or random init) and log previews.
    """
    if num_samples <= 0 or mlflow.active_run() is None:
        return
    st = student.state_dict()
    enc_state = strip_module_prefix(st)
    sam = build_sam_tiny(mobilesam_root, base_sam_checkpoint, device)
    merge_tinyvit_encoder_into_sam(sam, enc_state, strict=False)
    try:
        log_sam_val_preview_artifacts(
            sam,
            sam_val_loader,
            device,
            epoch,
            multimask_output,
            num_samples,
            staging_dir,
            preview_artifact_prefix="val_previews_merged_sam",
        )
    finally:
        del sam


def train_sam_epochs(
    sam: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    train_cfg: dict,
    multimask_output: bool,
    epoch_offset: int = 0,
    staging_dir: Optional[Path] = None,
) -> None:
    show_tqdm = train_cfg.get("show_progress", True)
    log_iv = int(train_cfg.get("log_interval_batches", 50))
    global_step = epoch_offset * max(len(train_loader), 1)

    for epoch in range(1, epochs + 1):
        ep = epoch_offset + epoch
        sam.train()
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

            loss_reduced = loss.detach()
            epoch_loss += loss_reduced.item()
            nb += 1
            global_step += 1

            if batch_idx == 0 or (batch_idx + 1) % log_iv == 0:
                mlflow.log_metric("sam_train_loss_batch", loss_reduced.item(), step=global_step)

            if show_tqdm:
                bar.set_postfix(loss=f"{loss_reduced.item():.4f}")

        if scheduler is not None:
            scheduler.step()
            mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=ep)

        epoch_time = time.perf_counter() - t0
        mlflow.log_metric("sam_train_loss_epoch", epoch_loss / max(nb, 1), step=ep)
        mlflow.log_metric("sam_epoch_time_sec", epoch_time, step=ep)
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
                    preview_artifact_prefix="val_previews",
                )


def run_encoder_distill(
    cfg: dict,
    cfg_path: Path,
    mobilesam_root: Path,
    device: torch.device,
) -> None:
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    out_cfg = cfg.get("output", {})
    output_dir = Path(out_cfg.get("dir", "./training_outputs")).expanduser()
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
        split_manifest_out=split_out if manifest_path is None else None,
    )

    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 4))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(batch_size, 1),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=max(batch_size, 1),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    model_cfg_enc = cfg.get("model", {})
    encoder_sam_ckpt = resolve_pretrained_checkpoint(cfg)

    n_prev_enc = int(train_cfg.get("val_preview_samples", 3))
    multimask_enc = bool(cfg.get("sam", {}).get("multimask_output", False))
    sam_enc_val_loader: Optional[DataLoader] = None
    if (
        n_prev_enc > 0
        and data_cfg.get("annotation_root")
        and len(jpg_splits["val"]) > 0
    ):
        pv_ds = SA1BSamDataset(
            jpg_splits["val"],
            data_cfg,
            img_size=int(data_cfg.get("image_size", 1024)),
            progress_label="val_preview",
            split="val",
        )
        bs_pv = min(4, max(len(pv_ds), 1))
        sam_enc_val_loader = DataLoader(
            pv_ds,
            batch_size=bs_pv,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_sam_collate,
            pin_memory=pin,
        )

    TinyViT = _import_tiny_vit(mobilesam_root)
    model = TinyViT(
        img_size=int(model_cfg_enc.get("img_size", 1024)),
        in_chans=int(model_cfg_enc.get("in_chans", 3)),
        num_classes=int(model_cfg_enc.get("num_classes", 1000)),
        embed_dims=list(model_cfg_enc.get("embed_dims", [64, 128, 160, 320])),
        depths=list(model_cfg_enc.get("depths", [2, 2, 6, 2])),
        num_heads=list(model_cfg_enc.get("num_heads", [2, 4, 5, 10])),
        window_sizes=list(model_cfg_enc.get("window_sizes", [7, 7, 14, 7])),
        mlp_ratio=float(model_cfg_enc.get("mlp_ratio", 4.0)),
        drop_rate=float(model_cfg_enc.get("drop_rate", 0.0)),
        drop_path_rate=float(model_cfg_enc.get("drop_path_rate", 0.0)),
        use_checkpoint=bool(model_cfg_enc.get("use_checkpoint", False)),
        mbconv_expand_ratio=float(model_cfg_enc.get("mbconv_expand_ratio", 4.0)),
        local_conv_size=int(model_cfg_enc.get("local_conv_size", 3)),
        layer_lr_decay=float(model_cfg_enc.get("layer_lr_decay", 0.8)),
    )

    model = model.to(device)

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
    flatten_cfg("", {**cfg_flat, "git_sha": sha, "training_gpus": "1"}, flat_params)

    t_wall_start = time.perf_counter()
    peak_mem = 0
    epochs = int(train_cfg.get("epochs", 8))
    log_interval = int(train_cfg.get("log_interval_batches", 50))
    show_tqdm = train_cfg.get("show_progress", True)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=run_name, log_system_metrics=True)
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

    try:
        epoch_bar = tqdm(
            range(1, epochs + 1),
            desc="Epochs",
            disable=not show_tqdm,
        )
        for epoch in epoch_bar:
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

                loss_reduced = loss.detach()
                epoch_loss += loss_reduced.item()
                epoch_batches += 1

                if torch.cuda.is_available():
                    peak_mem = max(peak_mem, torch.cuda.max_memory_allocated(device))

                if batch_idx == 0 or (batch_idx + 1) % log_interval == 0:
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

            mlflow.log_metric("train_loss_epoch", train_loss_epoch, step=epoch)
            mlflow.log_metric("epoch_time_sec", epoch_time, step=epoch)
            mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=epoch)

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
            if sam_enc_val_loader is not None:
                log_encoder_merged_sam_val_previews(
                    model,
                    mobilesam_root,
                    encoder_sam_ckpt,
                    sam_enc_val_loader,
                    device,
                    epoch,
                    multimask_enc,
                    n_prev_enc,
                    output_dir,
                )

        test_metrics = evaluate_encoder(
            model,
            test_loader,
            device,
            desc="Test (held-out)",
            show_progress=show_tqdm,
        )

        total_time = time.perf_counter() - t_wall_start

        mlflow.log_metric("test_embedding_loss", test_metrics["loss"])
        mlflow.log_metric("test_cosine_similarity", test_metrics["cosine_similarity"])
        mlflow.log_metric("total_train_time_sec", total_time)
        mlflow.log_metric("peak_cuda_memory_bytes", float(peak_mem))
        mlflow.log_metric("epochs_completed", float(epochs))

        ckpt_enc = output_dir / "tinyvit_encoder_only.pth"
        st = model.state_dict()
        torch.save(strip_module_prefix(st), ckpt_enc)

        sam = build_sam_tiny(mobilesam_root, encoder_sam_ckpt, device)
        merge_tinyvit_encoder_into_sam(sam, strip_module_prefix(st), strict=False)
        sam.eval()

        multimask = bool(cfg.get("sam", {}).get("multimask_output", False))
        if data_cfg.get("annotation_root"):
            _, _, sam_test = build_sam_loaders(
                data_cfg,
                jpg_splits,
                batch_size=int(train_cfg.get("eval_batch_size", train_cfg.get("batch_size", 8))),
                num_workers=num_workers,
                device_is_cuda=torch.cuda.is_available(),
            )
            if sam_test is not None:
                tiou = eval_sam_loader_mean_iou(sam, sam_test, device, multimask)
                mlflow.log_metric("test_mean_iou_lowres", tiou)

        full_path = output_dir / "mobile_sam_full.pt"
        save_sam_checkpoint(full_path, sam)
        mlflow.log_artifact(str(full_path), artifact_path="checkpoints")

    finally:
        if mlflow.active_run():
            mlflow.end_run()


def _sam_collate(batch: List[Any]) -> List[Any]:
    return batch


def run_full_sam(
    cfg: dict,
    cfg_path: Path,
    mobilesam_root: Path,
    device: torch.device,
) -> None:
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    if not data_cfg.get("annotation_root"):
        raise ValueError("full_sam requires data.annotation_root with mask JSON per image.")
    out_cfg = cfg.get("output", {})
    output_dir = Path(out_cfg.get("dir", "./training_outputs")).expanduser()
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
        split_manifest_out=split_out if manifest_path is None else None,
    )

    batch_size = int(train_cfg.get("batch_size", 4))
    num_workers = int(data_cfg.get("num_workers", 4))

    img_sz = int(data_cfg.get("image_size", 1024))
    train_ds = SA1BSamDataset(
        jpg_splits["train"], data_cfg, img_size=img_sz, progress_label="train", split="train"
    )
    val_ds = SA1BSamDataset(
        jpg_splits["val"], data_cfg, img_size=img_sz, progress_label="val", split="val"
    )
    test_ds = SA1BSamDataset(
        jpg_splits["test"], data_cfg, img_size=img_sz, progress_label="test", split="test"
    )
    sam_train = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_sam_collate,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    sam_val = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_sam_collate,
        pin_memory=torch.cuda.is_available(),
    )
    sam_test = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_sam_collate,
        pin_memory=torch.cuda.is_available(),
    )

    sam_ckpt_resolved = resolve_pretrained_checkpoint(cfg)

    sam = build_sam_tiny(mobilesam_root, sam_ckpt_resolved, device)

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
    flatten_cfg("", {**cfg_flat, "git_sha": sha, "training_gpus": "1"}, flat_params)
    multimask = bool(cfg.get("sam", {}).get("multimask_output", False))
    epochs = int(train_cfg.get("epochs", 8))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=run_name, log_system_metrics=True)
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
            multimask,
            epoch_offset=0,
            staging_dir=output_dir,
        )
        if sam_test is not None:
            tiou = eval_sam_loader_mean_iou(sam, sam_test, device, multimask)
            mlflow.log_metric("test_mean_iou_lowres", tiou)
            full_path = output_dir / "mobile_sam_full.pt"
            save_sam_checkpoint(full_path, sam)
            mlflow.log_artifact(str(full_path), artifact_path="checkpoints")
    finally:
        if mlflow.active_run():
            mlflow.end_run()


def main() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError(
            "Multi-GPU is not supported. Run: python3 train.py --config <yaml> "
            "(do not use torchrun --nproc_per_node > 1)."
        )
    parser = argparse.ArgumentParser(description="Unified MobileSAM / TinyViT training (single GPU)")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg_path = Path(args.config).resolve()
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_top = cfg.get("training", {})
    mode = train_top.get("mode", "encoder_distill")

    mobilesam_root = _resolve_mobilesam_root(cfg)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if mode == "encoder_distill":
        run_encoder_distill(cfg, cfg_path, mobilesam_root, device)
    elif mode == "full_sam":
        run_full_sam(cfg, cfg_path, mobilesam_root, device)
    else:
        raise ValueError(f"Unknown training.mode: {mode}")


if __name__ == "__main__":
    main()
