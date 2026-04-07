#!/usr/bin/env python3
"""
Hyperparameter tuning for MobileSAM using Ray Tune.
Run as: python tune_train.py --config ~/training_out/run.yaml
"""
import os
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_LOG_TO_STDERR"] = "0"
os.environ["GLOG_minloglevel"] = "3"
os.environ["RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS"] = "0"

import argparse
import sys
import time
from pathlib import Path

import yaml
from ray import tune


def get_search_space(cfg: dict) -> dict:
    """Get search space from config."""
    tune_cfg = cfg.get("tune", {})
    search_space = tune_cfg.get("search_space", {})
    
    return {
        "learning_rate": tune.choice(search_space.get("learning_rates", [1e-4, 5e-5, 1e-5])),
        "weight_decay": tune.choice(search_space.get("weight_decays", [0.01, 0.001])),
        "batch_size": tune.choice(search_space.get("batch_sizes", [4, 2])),
        "scheduler_gamma": tune.choice(search_space.get("scheduler_gammas", [0.9])),
        "optimizer": tune.choice(search_space.get("optimizers", ["adamw"])),
    }


def train_with_config(config: dict, cfg: dict, cfg_path: Path) -> None:
    """Train SAM with given hyperparameters - simple version without Ray Train."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from dataset_sa1b import SA1BSamDataset, build_datasets
    from sam_utils import build_sam_tiny, forward_sam_trainable, segmentation_loss, mean_iou_from_logits
    from training_core import build_optimizer_sam, effective_train_cfg_for_eval
    
    hyperparams = config
    train_cfg = effective_train_cfg_for_eval(cfg["train"], cfg["data"])
    
    for k, v in hyperparams.items():
        if k in train_cfg:
            train_cfg[k] = v
    
    data_cfg = cfg["data"]
    _, _, _, _, _, jpg_splits = build_datasets(
        data_cfg=data_cfg,
        img_size=int(data_cfg.get("image_size", 1024)),
        seed=int(data_cfg.get("seed", 42)),
        train_frac=float(data_cfg.get("train_frac", 0.7)),
        val_frac=float(data_cfg.get("val_frac", 0.1)),
        test_frac=float(data_cfg.get("test_frac", 0.2)),
    )

    batch_size = int(train_cfg.get("batch_size", 4))
    img_sz = int(data_cfg.get("image_size", 1024))
    
    train_ds = SA1BSamDataset(jpg_splits["train"], data_cfg, img_size=img_sz, split="train")
    val_ds = SA1BSamDataset(jpg_splits["val"], data_cfg, img_size=img_sz, split="val")
    
    sam_inst_frac = data_cfg.get("sam_instance_frac")
    if sam_inst_frac and 0 < sam_inst_frac <= 1.0:
        n_train = max(1, int(len(train_ds) * sam_inst_frac))
        n_val = max(1, int(len(val_ds) * sam_inst_frac))
        train_ds = torch.utils.data.Subset(train_ds, list(range(n_train)))
        val_ds = torch.utils.data.Subset(val_ds, list(range(n_val)))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=lambda x: x)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=lambda x: x)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mobilesam_root = os.environ.get("MOBILESAM_ROOT", "/mobilesam")
    sam = build_sam_tiny(Path(mobilesam_root), None, device)
    sam.to(device)
    
    opt_name = train_cfg.get("optimizer", "adamw").lower()
    lr = float(train_cfg.get("learning_rate", 1e-4))
    wd = float(train_cfg.get("weight_decay", 0.01))
    optimizer = build_optimizer_sam(sam, opt_name, lr, wd, 0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(train_cfg.get("scheduler_gamma", 0.9)))
    
    epochs = int(train_cfg.get("epochs", 2))
    multimask = bool(cfg.get("sam", {}).get("multimask_output", False))
    metric = cfg.get("tune_metric", "val_mean_iou_lowres")
    
    for epoch in range(1, epochs + 1):
        sam.train()
        train_loss = 0.0
        nb = 0
        for batch in train_loader:
            for b in batch:
                b["image"] = b["image"].to(device)
                b["boxes"] = b["boxes"].to(device)
                b["low_res_mask_gt"] = b["low_res_mask_gt"].to(device)
            optimizer.zero_grad()
            logits, _ = forward_sam_trainable(sam, batch, multimask, device)
            tgt = torch.stack([b["low_res_mask_gt"] for b in batch])
            loss = segmentation_loss(logits, tgt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            nb += 1
        
        if scheduler:
            scheduler.step()
        
        sam.eval()
        tot_iou, n_iou = 0.0, 0
        for batch in val_loader:
            for b in batch:
                b["image"] = b["image"].to(device)
                b["boxes"] = b["boxes"].to(device)
                b["low_res_mask_gt"] = b["low_res_mask_gt"].to(device)
            logits, _ = forward_sam_trainable(sam, batch, multimask, device)
            tgt = torch.stack([b["low_res_mask_gt"] for b in batch])
            tot_iou += mean_iou_from_logits(logits, tgt) * len(batch)
            n_iou += len(batch)
        
        val_iou = tot_iou / max(n_iou, 1)
        tune.report(**{metric: val_iou, "train_loss": train_loss / max(nb, 1)})


def main():
    parser = argparse.ArgumentParser(description="Ray Tune hyperparameter tuning")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-trials", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    tune_cfg = cfg.get("tune", {})
    num_trials = args.num_trials
    epochs = args.epochs
    cfg["tune"]["enabled"] = False
    cfg["train"]["epochs"] = epochs
    
    train_top = cfg.get("training", {})
    mode = train_top.get("mode", "full_sam")
    metric = "val_mean_iou_lowres" if mode == "full_sam" else "val_embedding_loss"
    mode_str = "max" if mode == "full_sam" else "min"
    
    cfg["tune_metric"] = metric
    
    print(f"Tuning: {num_trials} trials, {epochs} epochs, metric={metric}")

    tuner = tune.Tuner(
        lambda c: train_with_config(c, cfg, cfg_path),
        tune_config=tune.TuneConfig(num_samples=num_trials, max_concurrent_trials=1),
        param_space=get_search_space(cfg),
    )

    results = tuner.fit()
    best = results.get_best_result(metric, mode_str)
    
    print(f"\nBest: {best.config}, {metric}={best.metrics.get(metric)}")
    
    out_dir = Path(cfg.get("output", {}).get("dir", "~/training_out"))
    (out_dir / "tune_results.json").write_text(str(best.config))


if __name__ == "__main__":
    main()