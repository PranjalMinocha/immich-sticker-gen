#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def _resolve_uri_to_local_path(uri: str, objstore_local_root: Path, data_dir: Path) -> Path:
    raw = (uri or "").strip()
    if raw.startswith("s3://"):
        parts = raw.split("/", 3)
        if len(parts) < 4:
            raise ValueError(f"Invalid s3 URI: {uri}")
        return (objstore_local_root / parts[3]).resolve()
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (data_dir / p).resolve()


def _annotation_candidates(annotation_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    anns = annotation_payload.get("annotations")
    if isinstance(anns, list):
        return [ann for ann in anns if isinstance(ann, dict)]
    if isinstance(annotation_payload, dict):
        return [annotation_payload]
    return []


def _bbox_area_ratio(ann: Dict[str, Any], image_w: int, image_h: int) -> Optional[float]:
    bbox = ann.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        w = float(bbox[2])
        h = float(bbox[3])
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0 or image_w <= 0 or image_h <= 0:
        return None
    return (w * h) / float(image_w * image_h)


@dataclass
class Candidate:
    image_uri: str
    annotation_uri: str
    ann_idx: int
    small_object_ratio: float
    mean_luminance: float


def _collect_candidates(
    manifest_path: Path,
    objstore_local_root: Path,
    data_dir: Path,
) -> List[Candidate]:
    out: List[Candidate] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {manifest_path}")
        if "image_uri" not in reader.fieldnames or "annotation_uri" not in reader.fieldnames:
            raise ValueError(f"CSV must include image_uri and annotation_uri: {manifest_path}")

        for row in reader:
            image_uri = (row.get("image_uri") or "").strip()
            annotation_uri = (row.get("annotation_uri") or "").strip()
            if not image_uri or not annotation_uri:
                continue

            image_path = _resolve_uri_to_local_path(image_uri, objstore_local_root, data_dir)
            annotation_path = _resolve_uri_to_local_path(annotation_uri, objstore_local_root, data_dir)
            if not image_path.is_file() or not annotation_path.is_file():
                continue

            with Image.open(image_path).convert("RGB") as img:
                w, h = img.size
                mean_luminance = float(np.asarray(img.convert("L"), dtype=np.float32).mean() / 255.0)

            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            anns = _annotation_candidates(payload)
            if not anns:
                continue

            ratio_choices: List[Tuple[int, float]] = []
            for idx, ann in enumerate(anns):
                ratio = _bbox_area_ratio(ann, w, h)
                if ratio is not None:
                    ratio_choices.append((idx, ratio))
            if not ratio_choices:
                continue

            ann_idx, ratio = min(ratio_choices, key=lambda it: it[1])
            out.append(
                Candidate(
                    image_uri=image_uri,
                    annotation_uri=annotation_uri,
                    ann_idx=ann_idx,
                    small_object_ratio=ratio,
                    mean_luminance=mean_luminance,
                )
            )
    return out


def _write_subset(path: Path, rows: List[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_uri", "annotation_uri", "ann_idx"])
        for row in rows:
            writer.writerow([row.image_uri, row.annotation_uri, row.ann_idx])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build small-object and low-light evaluation subsets")
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--objstore-local-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--small-count", type=int, default=20)
    parser.add_argument("--low-light-count", type=int, default=20)
    args = parser.parse_args()

    manifest_path = Path(args.val_manifest).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    objstore_local_root = Path(args.objstore_local_root).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()

    candidates = _collect_candidates(manifest_path, objstore_local_root, data_dir)
    if not candidates:
        raise RuntimeError("No valid candidates found while building eval subsets")

    small = sorted(candidates, key=lambda c: c.small_object_ratio)[: max(1, args.small_count)]
    low_light = sorted(candidates, key=lambda c: c.mean_luminance)[: max(1, args.low_light_count)]

    small_path = out_dir / "small_objects_manifest.csv"
    low_light_path = out_dir / "low_light_manifest.csv"
    _write_subset(small_path, small)
    _write_subset(low_light_path, low_light)

    summary = {
        "total_candidates": len(candidates),
        "small_objects_count": len(small),
        "low_light_count": len(low_light),
        "small_objects_manifest": str(small_path),
        "low_light_manifest": str(low_light_path),
        "small_object_ratio_min": min(c.small_object_ratio for c in small),
        "small_object_ratio_max": max(c.small_object_ratio for c in small),
        "low_light_luminance_min": min(c.mean_luminance for c in low_light),
        "low_light_luminance_max": max(c.mean_luminance for c in low_light),
    }
    summary_path = out_dir / "subset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
