#!/usr/bin/env python3
import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Render run-specific retraining config from base YAML")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-root", default="/data")
    parser.add_argument("--output-dir", default="/out")
    parser.add_argument("--mobilesam-root", default="/mobilesam")
    parser.add_argument("--pretrained-checkpoint-path", default=None)
    parser.add_argument("--force-quality-gate-pass", action="store_true")
    parser.add_argument("--force-quality-gate-fail", action="store_true")
    parser.add_argument("--disable-pretrained", action="store_true")
    args = parser.parse_args()

    base_path = Path(args.base_config)
    out_path = Path(args.output_config)
    cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))

    data = cfg.setdefault("data", {})
    run_prefix = f"{args.data_root.rstrip('/')}/retraining_runs/{args.run_id}/dataset_manifests"
    data["train_manifest_csv"] = f"{run_prefix}/train_manifest.csv"
    data["val_manifest_csv"] = f"{run_prefix}/val_manifest.csv"

    output_cfg = cfg.setdefault("output", {})
    output_cfg["dir"] = args.output_dir
    cfg["mobilesam_root"] = args.mobilesam_root

    if args.disable_pretrained:
        training_top = cfg.setdefault("training", {})
        training_top["use_pretrained"] = False
    elif args.pretrained_checkpoint_path:
        training_top = cfg.setdefault("training", {})
        training_top["use_pretrained"] = True
        training_top["pretrained_checkpoint_path"] = args.pretrained_checkpoint_path

    if args.force_quality_gate_pass:
        offline_eval = cfg.setdefault("offline_eval", {})
        offline_eval["min_dice"] = 0.0
        offline_eval["min_iou"] = 0.0
        offline_eval["max_runtime_seconds"] = 86400
        offline_eval["enable_boundary_gate"] = True
        offline_eval["min_boundary_f1"] = 0.0
        offline_eval["enable_prompt_robustness_gate"] = True
        offline_eval["max_prompt_iou_drop"] = 1.0
        offline_eval["min_prompt_robust_iou"] = 0.0
        offline_eval["enable_small_object_gate"] = True
        offline_eval["min_small_object_iou"] = 0.0
        offline_eval["enable_low_light_gate"] = True
        offline_eval["min_low_light_iou"] = 0.0

    if args.force_quality_gate_fail:
        offline_eval = cfg.setdefault("offline_eval", {})
        offline_eval["min_dice"] = 0.99999
        offline_eval["min_iou"] = 0.99999
        offline_eval["max_runtime_seconds"] = 1
        offline_eval["enable_boundary_gate"] = True
        offline_eval["min_boundary_f1"] = 0.99999
        offline_eval["enable_prompt_robustness_gate"] = True
        offline_eval["max_prompt_iou_drop"] = 0.0
        offline_eval["min_prompt_robust_iou"] = 0.99999
        offline_eval["enable_small_object_gate"] = True
        offline_eval["min_small_object_iou"] = 0.99999
        offline_eval["enable_low_light_gate"] = True
        offline_eval["min_low_light_iou"] = 0.99999

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
