#!/usr/bin/env python3
"""
MobileSAM-style TinyViT encoder distillation against ViT-H teacher .npy features.
Config-driven single entrypoint; MLflow logging; optional torchrun multi-GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import mlflow
from tqdm import tqdm

from dataset_sa1b import build_datasets


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


def setup_distributed(backend: str) -> tuple[int, int, int]:
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) <= 1:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Eval",
    show_progress: bool = False,
) -> Dict[str, float]:
    """Single-process eval (call on rank 0 only when using DDP)."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyViT encoder distillation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    cfg_path = Path(args.config).resolve()
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg.get("train", {})
    backend = train_cfg.get("distributed_backend", "nccl")
    rank, world_size, local_rank = setup_distributed(backend)
    is_master = rank == 0

    mobilesam_root = _resolve_mobilesam_root(cfg)
    TinyViT = _import_tiny_vit(mobilesam_root)

    data_cfg = cfg["data"]
    out_cfg = cfg.get("output", {})
    output_dir = Path(out_cfg.get("dir", "./training_outputs")).expanduser()
    if is_master:
        output_dir.mkdir(parents=True, exist_ok=True)

    split_out = output_dir / "split_manifest.json"
    manifest_in = data_cfg.get("split_manifest")
    manifest_path = Path(manifest_in).resolve() if manifest_in else None

    _, train_ds, val_ds, test_ds, split_meta = build_datasets(
        data_root=Path(data_cfg["root"]).expanduser().resolve(),
        shard_dirs=data_cfg["shard_dirs"],
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
        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=int(data_cfg.get("seed", 42)))
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=per_gpu,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(per_gpu, 1),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=max(per_gpu, 1),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

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

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
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
    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        momentum = float(train_cfg.get("momentum", 0.9))
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(train_cfg.get("scheduler_gamma", 0.5))
    )

    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or mlflow_cfg.get(
        "tracking_uri", "http://127.0.0.1:5000"
    )
    experiment_name = mlflow_cfg.get("experiment_name", "immich-sticker-encoder")
    run_name = mlflow_cfg.get("run_name") or None

    sha = git_sha(_repo_root())
    flat_params: Dict[str, str] = {}
    cfg_flat_source = {k: v for k, v in cfg.items() if k != "mobilesam_root"}
    flatten_cfg(
        "",
        {**cfg_flat_source, "git_sha": sha, "world_size": world_size, "backend": backend},
        flat_params,
    )

    t_wall_start = time.perf_counter()
    peak_mem = 0

    if is_master:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)
        for k, v in flat_params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("config_path", str(cfg_path))
        mlflow.log_param("mobilesam_root", str(mobilesam_root))
        if cfg.get("mobilesam_root") is not None:
            mlflow.log_param("mobilesam_root_config", str(cfg["mobilesam_root"]))
        for k, v in split_meta.items():
            if k != "counts":
                mlflow.log_param(f"split_{k}", str(v))
        if "counts" in split_meta:
            for sk, sv in split_meta["counts"].items():
                mlflow.log_param(f"split_count_{sk}", sv)
        env_info = gpu_env_info()
        for k, v in env_info.items():
            mlflow.log_param(f"env_{k}", v)
        if split_out.is_file():
            mlflow.log_artifact(str(split_out), artifact_path="split")

    epochs = int(train_cfg.get("epochs", 8))
    log_interval = int(train_cfg.get("log_interval_batches", 50))

    show_tqdm = is_master and train_cfg.get("show_progress", True)

    try:
        epoch_bar = tqdm(
            range(1, epochs + 1),
            desc="Epochs",
            position=0,
            leave=True,
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
                position=1 if show_tqdm else 0,
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

            if show_tqdm:
                epoch_bar.set_postfix(
                    train_loss=f"{train_loss_epoch:.4f}",
                    ep_time_s=f"{epoch_time:.0f}",
                )

            if world_size > 1:
                dist.barrier()
            if is_master:
                val_metrics = evaluate(
                    model,
                    val_loader,
                    device,
                    desc=f"Val ep {epoch}",
                    show_progress=show_tqdm,
                )
                mlflow.log_metric("val_embedding_loss", val_metrics["loss"], step=epoch)
                mlflow.log_metric("val_cosine_similarity", val_metrics["cosine_similarity"], step=epoch)
            if world_size > 1:
                dist.barrier()

        if world_size > 1:
            dist.barrier()
        if is_master:
            test_metrics = evaluate(
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
            mlflow.log_metric(
                "test_cosine_similarity", test_metrics["cosine_similarity"]
            )
            mlflow.log_metric("total_train_time_sec", total_time)
            mlflow.log_metric("peak_cuda_memory_bytes", float(peak_mem))
            mlflow.log_metric("epochs_completed", float(epochs))

            ckpt_path = output_dir / "tinyvit_encoder_final.pth"
            state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            torch.save(state, ckpt_path)
            mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")

    finally:
        if is_master and mlflow.active_run():
            mlflow.end_run()
        cleanup_distributed()


if __name__ == "__main__":
    main()
