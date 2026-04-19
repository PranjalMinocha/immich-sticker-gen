#!/usr/bin/env python3
"""
Ray Train entry point for MobileSAM training.
Uses TorchTrainer for multi-GPU training on AMD GPUs (ROCm).

Usage:
    python train.py --config configs/run.yaml

For hyperparameter tuning, use tune_train.py:
    python tune_train.py --config configs/tune_example.yaml
"""
from __future__ import annotations

import os
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_LOG_TO_STDERR"] = "0"
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS"] = "0"
os.environ["RAY_SCHEDULER_EVENTS"] = "0"
os.environ["RAY_AIR_SKIP_CONTROLLER_STATE_QUERY"] = "1"

import sys

class StderrFilter:
    def __init__(self, real_stderr):
        self._real = real_stderr
        self._skip_next_newline = False
        
    def write(self, text):
        if 'PlacementGroupCleaner' in text or 'Failed to query' in text:
            self._skip_next_newline = True
            return len(text)
        if self._skip_next_newline and text in ('\n', '\n\n', '\n\n\n'):
            self._skip_next_newline = False
            return len(text)
        self._real.write(text)
        return len(text)
        
    def flush(self):
        self._real.flush()
        
    def fileno(self):
        return self._real.fileno()
        
    def isatty(self):
        return self._real.isatty()

sys.stderr = StderrFilter(sys.stderr)

import warnings
warnings.filterwarnings("ignore")

import logging

import argparse
import json
import math
import re
import subprocess
import sys
import time
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader
from tqdm import tqdm

import ray
import ray.train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig

from dataset_sa1b import (
    SAM_ENCODER_PAD_SIDE,
    SA1BSamDataset,
    build_datasets,
    build_sam_loaders,
)
from sam_utils import (
    build_sam_tiny,
    mean_dice_from_logits,
    forward_sam_trainable,
    mean_iou_from_logits,
    merge_tinyvit_encoder_into_sam,
    save_sam_checkpoint,
    segmentation_loss,
    strip_module_prefix,
)
from offline_eval import OfflineEvalMetrics, OfflineEvalThresholds, evaluate_quality_gates
from training_core import (
    encoder_distill_loss,
    evaluate_encoder,
    flatten_cfg,
    git_sha,
    gpu_env_info,
    gpu_util_logging_status,
    init_rocm_smi_for_gpu_util_logging,
    sample_all_gpu_utilization,
    sample_gpu_memory_utilization_percent,
    sample_gpu_utilization_percent,
    torch_cuda_memory_mib,
    _import_tiny_vit,
    _repo_root,
    _resolve_mobilesam_root,
)


def resolve_pretrained_checkpoint(cfg: dict) -> Optional[str]:
    """training.use_pretrained and training.pretrained_checkpoint_path."""
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


_DEFAULT_SAM_INSTANCE_EVAL_BATCHES = 500


def effective_train_cfg_for_eval(train_cfg: dict, data_cfg: dict) -> dict:
    """Shallow copy of train with optional sam_instance_frac from data."""
    out = dict(train_cfg)
    if "sam_instance_frac" not in out and data_cfg.get("sam_instance_frac") is not None:
        out["sam_instance_frac"] = data_cfg["sam_instance_frac"]
    return out


def resolve_eval_max_batches(train_cfg: dict, loader: DataLoader) -> Optional[int]:
    """Batch cap for SAM IoU eval."""
    frac_raw = train_cfg.get("sam_instance_frac")
    if frac_raw is not None:
        frac = float(frac_raw)
        if not (0.0 < frac <= 1.0):
            raise ValueError("train.sam_instance_frac must be in (0, 1].")
        n_batches = len(loader)
        if n_batches <= 0:
            return None
        return max(1, int(math.ceil(frac * n_batches)))
    return _DEFAULT_SAM_INSTANCE_EVAL_BATCHES


def _log_mlflow_eval_batch_cap(train_cfg: dict) -> None:
    frac_raw = train_cfg.get("sam_instance_frac")
    if frac_raw is not None:
        mlflow.log_param("sam_instance_frac", str(frac_raw))
    else:
        mlflow.log_param(
            "sam_instance_frac",
            f"unset_default_{_DEFAULT_SAM_INSTANCE_EVAL_BATCHES}_batches",
        )


def _log_mlflow_gpu_metric_init() -> None:
    """AMD ROCm: pyrsmi and/or rocm-smi CLI."""
    init_rocm_smi_for_gpu_util_logging()
    backend, detail = gpu_util_logging_status()
    if backend == "pyrsmi":
        tag = "pyrsmi_rocm"
    elif backend == "rocm_smi_cli":
        tag = "rocm_smi_cli"
    else:
        tag = "unavailable"
    mlflow.log_param("gpu_util_mlflow_metrics", tag)
    mlflow.log_param("gpu_util_backend", backend)
    if detail:
        mlflow.log_param("gpu_util_init_detail", str(detail)[:900])


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
    max_batches: Optional[int] = None,
    show_progress: bool = True,
    desc: str = "SAM IoU",
) -> float:
    """Mean low-res mask IoU over a DataLoader."""
    sam.eval()
    tot = 0.0
    n = 0
    if max_batches is not None:
        iterable = islice(loader, max_batches)
        total = max_batches
    else:
        iterable = loader
        total = len(loader) if hasattr(loader, "__len__") else None
    bar = tqdm(iterable, desc=desc, total=total, disable=not show_progress, leave=False)
    for batch in bar:
        for b in batch:
            b["image"] = b["image"].to(device, non_blocking=True)
            b["boxes"] = b["boxes"].to(device, non_blocking=True)
            b["low_res_mask_gt"] = b["low_res_mask_gt"].to(device, non_blocking=True)
        logits, _ = forward_sam_trainable(sam, batch, multimask_output, device)
        tgt = torch.stack([b["low_res_mask_gt"] for b in batch], dim=0)
        tot += mean_iou_from_logits(logits, tgt) * len(batch)
        n += len(batch)
    return tot / max(n, 1)


@torch.no_grad()
def eval_sam_loader_mean_dice(
    sam: nn.Module,
    loader: DataLoader,
    device: torch.device,
    multimask_output: bool,
    max_batches: Optional[int] = None,
    show_progress: bool = True,
    desc: str = "SAM Dice",
) -> float:
    sam.eval()
    tot = 0.0
    n = 0
    if max_batches is not None:
        iterable = islice(loader, max_batches)
        total = max_batches
    else:
        iterable = loader
        total = len(loader) if hasattr(loader, "__len__") else None
    bar = tqdm(iterable, desc=desc, total=total, disable=not show_progress, leave=False)
    for batch in bar:
        for b in batch:
            b["image"] = b["image"].to(device, non_blocking=True)
            b["boxes"] = b["boxes"].to(device, non_blocking=True)
            b["low_res_mask_gt"] = b["low_res_mask_gt"].to(device, non_blocking=True)
        logits, _ = forward_sam_trainable(sam, batch, multimask_output, device)
        tgt = torch.stack([b["low_res_mask_gt"] for b in batch], dim=0)
        tot += mean_dice_from_logits(logits, tgt) * len(batch)
        n += len(batch)
    return tot / max(n, 1)


def _sam_collate(batch: List[Any]) -> List[Any]:
    return batch


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
    world_rank: int = 0,
    log_to_mlflow: bool = True,
) -> None:
    """Train SAM for specified epochs (adapted for Ray multi-worker)."""
    show_tqdm = train_cfg.get("show_progress", True)
    log_iv = int(train_cfg.get("log_interval_batches", 50))
    sam_eval_mb = (
        resolve_eval_max_batches(train_cfg, val_loader)
        if val_loader is not None
        else None
    )
    global_step = epoch_offset * max(len(train_loader), 1)

    for epoch in range(1, epochs + 1):
        ep = epoch_offset + epoch
        sam.train()
        epoch_loss = 0.0
        nb = 0
        epoch_gpu_util_sum = 0.0
        epoch_gpu_util_count = 0
        t0 = time.perf_counter()
        bar = tqdm(train_loader, desc=f"SAM train ep {ep}", disable=not show_tqdm, leave=False)
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

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            
            if log_to_mlflow:
                gpu_utils = sample_all_gpu_utilization()
                util_mem = sample_gpu_memory_utilization_percent(device)
                mem_alloc, mem_res = torch_cuda_memory_mib(device)
                
                for gpu_idx, util in gpu_utils.items():
                    mlflow.log_metric(f"system/gpu_{gpu_idx}_utilization_percent", util, step=global_step)
                    epoch_gpu_util_sum += util
                    epoch_gpu_util_count += 1

                if batch_idx == 0 or (batch_idx + 1) % log_iv == 0:
                    mlflow.log_metric("sam_train_loss_batch", loss_reduced.item(), step=global_step)
                    if util_mem is not None:
                        mlflow.log_metric(
                            "system/gpu_memory_utilization_percent", util_mem, step=global_step
                        )
                    if mem_alloc is not None:
                        mlflow.log_metric(
                            "system/gpu_torch_memory_allocated_mib", mem_alloc, step=global_step
                        )
                    if mem_res is not None:
                        mlflow.log_metric(
                            "system/gpu_torch_memory_reserved_mib", mem_res, step=global_step
                        )

            if show_tqdm:
                bar.set_postfix(loss=f"{loss_reduced.item():.4f}")

        if scheduler is not None:
            scheduler.step()
            if log_to_mlflow:
                mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=ep)

        epoch_time = time.perf_counter() - t0
        if log_to_mlflow:
            mlflow.log_metric("sam_train_loss_epoch", epoch_loss / max(nb, 1), step=ep)
            mlflow.log_metric("sam_epoch_time_sec", epoch_time, step=ep)
            if epoch_gpu_util_count > 0:
                mlflow.log_metric(
                    "system/gpu_utilization_percent_mean",
                    epoch_gpu_util_sum / epoch_gpu_util_count,
                    step=ep,
                )
        print(f"sam_train_loss_epoch={epoch_loss / max(nb, 1)}", file=sys.stderr)
        
        if world_rank == 0 and val_loader is not None and log_to_mlflow:
            viou = eval_sam_loader_mean_iou(
                sam,
                val_loader,
                device,
                multimask_output,
                max_batches=sam_eval_mb,
                show_progress=show_tqdm,
                desc=f"SAM val IoU ep {ep}",
            )
            mlflow.log_metric("val_mean_iou_lowres", viou, step=ep)
            print(f"val_mean_iou_lowres={viou}", file=sys.stderr)


def train_fn(config: Dict[str, Any]) -> None:
    """
    Training function for Ray TorchTrainer.
    Each worker runs this function independently.
    """
    world_rank = ray.train.get_context().get_world_rank()
    local_rank = ray.train.get_context().get_local_rank()
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if world_rank == 0:
        print(f"Worker {world_rank}/{local_rank}: Using device {device}", file=sys.stderr)
    
    cfg = config["cfg"]
    cfg_path = config["cfg_path"]
    mobilesam_root = config["mobilesam_root"]
    log_to_mlflow = config.get("log_to_mlflow", True)
    orchestrator_run_id = config.get("orchestrator_run_id")
    output_json_path = config.get("output_json_path")
    run_started_unix = int(time.time())
    
    data_cfg = cfg["data"]
    train_cfg = effective_train_cfg_for_eval(cfg["train"], data_cfg)
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

    sam_inst_frac = data_cfg.get("sam_instance_frac")
    if sam_inst_frac is not None and 0 < sam_inst_frac <= 1.0:
        n_train = max(1, int(len(train_ds) * sam_inst_frac))
        n_val = max(1, int(len(val_ds) * sam_inst_frac))
        n_test = max(1, int(len(test_ds) * sam_inst_frac))
        train_ds = torch.utils.data.Subset(train_ds, list(range(n_train)))
        val_ds = torch.utils.data.Subset(val_ds, list(range(n_val)))
        test_ds = torch.utils.data.Subset(test_ds, list(range(n_test)))
        if world_rank == 0:
            print(f"Using {sam_inst_frac*100:.1f}%: train={n_train}, val={n_val}, test={n_test}", file=sys.stderr)
    
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
    sam = nn.DataParallel(sam, device_ids=[local_rank])

    opt_name = train_cfg.get("optimizer", "adamw").lower()
    lr = float(train_cfg.get("learning_rate", 1e-4))
    wd = float(train_cfg.get("weight_decay", 0.01))
    momentum = float(train_cfg.get("momentum", 0.9))
    optimizer = build_optimizer_sam(sam, opt_name, lr, wd, momentum)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(train_cfg.get("scheduler_gamma", 0.9))
    )

    multimask = bool(cfg.get("sam", {}).get("multimask_output", False))
    epochs = int(train_cfg.get("epochs", 8))
    show_tqdm = train_cfg.get("show_progress", True) and world_rank == 0
    
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or mlflow_cfg.get(
        "tracking_uri", "http://127.0.0.1:5000"
    )
    experiment_name = mlflow_cfg.get("experiment_name", "immich-sticker-sam")
    run_name = mlflow_cfg.get("run_name") or orchestrator_run_id
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    if world_rank == 0:
        result_payload: Dict[str, Any] = {
            "status": "failed",
            "metrics": {},
            "qualityGate": {},
            "mlflow": {},
        }
        try:
            mlflow.start_run(run_name=run_name, log_system_metrics=True)
        except Exception as e:
            print(f"MLflow start failed: {e}", file=sys.stderr)
            if mlflow.active_run():
                mlflow.end_run()
            mlflow.start_run(run_name=run_name, log_system_metrics=False)

        if orchestrator_run_id:
            mlflow.set_tag("immich_sticker_training_run_id", orchestrator_run_id)
        
        sha = git_sha(_repo_root())
        flat_params: Dict[str, str] = {}
        cfg_flat = {k: v for k, v in cfg.items() if k != "mobilesam_root"}
        flatten_cfg("", {**cfg_flat, "git_sha": sha, "training_gpus": "ray"}, flat_params)
        
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
        _log_mlflow_eval_batch_cap(train_cfg)
        if split_out.is_file():
            mlflow.log_artifact(str(split_out), artifact_path="split")
        _log_mlflow_gpu_metric_init()
        
        mlflow.log_param("ray_worker_rank", world_rank)
        mlflow.log_param("ray_local_rank", local_rank)

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
        world_rank=world_rank,
        log_to_mlflow=log_to_mlflow,
    )

    if world_rank == 0:
        offline_eval_cfg = cfg.get("offline_eval", {})
        max_eval_batches = resolve_eval_max_batches(train_cfg, sam_test)

        test_iou = eval_sam_loader_mean_iou(
            sam,
            sam_test,
            device,
            multimask,
            max_batches=max_eval_batches,
            show_progress=show_tqdm,
            desc="SAM test IoU",
        )
        test_dice = eval_sam_loader_mean_dice(
            sam,
            sam_test,
            device,
            multimask,
            max_batches=max_eval_batches,
            show_progress=show_tqdm,
            desc="SAM test Dice",
        )

        runtime_seconds = max(1, int(time.time()) - run_started_unix)
        thresholds = OfflineEvalThresholds(
            min_dice=float(offline_eval_cfg.get("min_dice", 0.8)),
            min_iou=float(offline_eval_cfg.get("min_iou", 0.7)),
            max_runtime_seconds=int(offline_eval_cfg.get("max_runtime_seconds", 14_400)),
        )
        metrics = OfflineEvalMetrics(
            dice=float(test_dice),
            iou=float(test_iou),
            runtime_seconds=runtime_seconds,
        )
        quality_gate = evaluate_quality_gates(metrics, thresholds)

        if log_to_mlflow:
            mlflow.log_metric("test_mean_iou_lowres", test_iou)
            mlflow.log_metric("test_mean_dice_lowres", test_dice)
            mlflow.log_metric("runtime_seconds", runtime_seconds)
            mlflow.log_param("offline_eval_min_iou", thresholds.min_iou)
            mlflow.log_param("offline_eval_min_dice", thresholds.min_dice)
            mlflow.log_param("offline_eval_max_runtime_seconds", thresholds.max_runtime_seconds)

        full_path = output_dir / "mobile_sam_full.pt"
        save_sam_checkpoint(full_path, sam)
        mlflow.log_artifact(str(full_path), artifact_path="checkpoints")

        active_run = mlflow.active_run()
        run_id = active_run.info.run_id if active_run else None
        result_payload = {
            "status": "passed" if quality_gate["passed"] else "failed",
            "metrics": {
                "dice": round(test_dice, 6),
                "iou": round(test_iou, 6),
                "runtimeSeconds": runtime_seconds,
            },
            "qualityGate": {
                **quality_gate,
                "expectedMinDice": thresholds.min_dice,
                "expectedMinIou": thresholds.min_iou,
                "expectedMaxRuntimeSeconds": thresholds.max_runtime_seconds,
            },
            "mlflow": {
                "trackingUri": tracking_uri,
                "runId": run_id,
            },
        }

        registry_cfg = cfg.get("model_registry", {})
        register_enabled = bool(registry_cfg.get("enabled", True))
        if register_enabled and quality_gate["passed"] and run_id:
            model_name = str(registry_cfg.get("model_name", "immich-sticker-mobilesam"))
            client = MlflowClient(tracking_uri=tracking_uri)
            try:
                client.get_registered_model(model_name)
            except Exception:
                client.create_registered_model(model_name)
            model_uri = f"runs:/{run_id}/checkpoints/mobile_sam_full.pt"
            mv = mlflow.register_model(model_uri=model_uri, name=model_name)
            client.set_model_version_tag(model_name, str(mv.version), "quality_gate_passed", "true")
            if orchestrator_run_id:
                client.set_model_version_tag(model_name, str(mv.version), "immich_run_id", orchestrator_run_id)
            result_payload["mlflow"]["modelName"] = model_name
            result_payload["mlflow"]["modelVersion"] = str(mv.version)
            result_payload["mlflow"]["registered"] = True
        else:
            result_payload["mlflow"]["registered"] = False

        if output_json_path:
            out_path = Path(output_json_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")

        print("Training completed.", file=sys.stderr)
        mlflow.end_run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ray Train MobileSAM training")
    parser.add_argument("--config", type=str, required=False, default=None)
    parser.add_argument("--num-workers", type=int, default=2, help="Number of Ray workers (GPUs)")
    parser.add_argument("--sample-window-size", type=int, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    config_path = args.config or os.environ.get("IMMICH_STICKER_TRAINING_CONFIG")
    if not config_path:
        raise ValueError("Config file is required. Use --config or IMMICH_STICKER_TRAINING_CONFIG.")

    cfg_path = Path(config_path).resolve()
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.sample_window_size is not None:
        cfg.setdefault("data", {})["sample_window_size"] = int(args.sample_window_size)

    if args.output_json:
        cfg.setdefault("output", {})["result_json_path"] = str(args.output_json)

    train_top = cfg.get("training", {})
    mode = train_top.get("mode", "full_sam")
    
    out_cfg = cfg.get("output", {})
    output_dir = Path(out_cfg.get("dir", "./training_outputs")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if mode != "full_sam":
        raise ValueError(f"Ray Train currently only supports full_sam mode, got {mode}")

    mobilesam_root = _resolve_mobilesam_root(cfg)
    
    num_workers = args.num_workers
    if not torch.cuda.is_available():
        num_workers = 1
        print("CUDA not available, using 1 worker", file=sys.stderr)
    else:
        available_gpus = torch.cuda.device_count()
        num_workers = min(num_workers, available_gpus)
        print(f"Using {num_workers} workers (GPUs available: {available_gpus})", file=sys.stderr)

    config = {
        "cfg": cfg,
        "cfg_path": str(cfg_path),
        "mobilesam_root": str(mobilesam_root),
        "log_to_mlflow": True,
        "orchestrator_run_id": args.run_id,
        "output_json_path": args.output_json,
    }

    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=True,
    )

    run_config = RunConfig(
        name=f"ray-mobilesam-{int(time.time())}",
    )
    
    mlflow.enable_system_metrics_logging()

    ray.init(
        ignore_reinit_error=True,
        num_gpus=num_workers,
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_fn,
        train_loop_config=config,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    result = trainer.fit()
    
    ray.shutdown()


if __name__ == "__main__":
    main()
