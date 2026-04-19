import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError

from ingestion_config import IngestionConfig


@dataclass
class CheckResult:
    hard_fail_reasons: list[str]
    soft_warn_reasons: list[str]
    metrics: dict


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _average_hash(path: str, hash_size: int = 8) -> str:
    image = Image.open(path).convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32)
    threshold = float(pixels.mean())
    bits = (pixels > threshold).astype(np.uint8).flatten()
    return "".join(str(int(bit)) for bit in bits)


def _hamming_distance(hash_a: str, hash_b: str) -> int:
    return sum(1 for left, right in zip(hash_a, hash_b) if left != right)


def _laplacian_variance(gray_pixels: np.ndarray) -> float:
    center = gray_pixels
    laplacian = (
        -4.0 * center
        + np.roll(center, 1, axis=0)
        + np.roll(center, -1, axis=0)
        + np.roll(center, 1, axis=1)
        + np.roll(center, -1, axis=1)
    )
    return float(np.var(laplacian))


def validate_sample(
    image_path: str,
    annotation_path: str,
    config: IngestionConfig,
    seen_sha256: set[str],
    seen_ahash: list[str],
) -> CheckResult:
    hard_fail_reasons: list[str] = []
    soft_warn_reasons: list[str] = []
    metrics: dict = {}

    if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
        hard_fail_reasons.append("missing_or_empty_image")
        return CheckResult(hard_fail_reasons, soft_warn_reasons, metrics)

    if not os.path.exists(annotation_path) or os.path.getsize(annotation_path) == 0:
        hard_fail_reasons.append("missing_or_empty_annotation")
        return CheckResult(hard_fail_reasons, soft_warn_reasons, metrics)

    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    annotation_stem = os.path.splitext(os.path.basename(annotation_path))[0]
    if image_stem != annotation_stem:
        hard_fail_reasons.append("stem_mismatch")

    try:
        image = Image.open(image_path)
        image = image.convert("RGB")
        width, height = image.size
        metrics["width"] = width
        metrics["height"] = height
    except UnidentifiedImageError:
        hard_fail_reasons.append("image_decode_failed")
        return CheckResult(hard_fail_reasons, soft_warn_reasons, metrics)

    if width < config.min_width or height < config.min_height:
        hard_fail_reasons.append("image_too_small")
    if width > config.max_width or height > config.max_height:
        hard_fail_reasons.append("image_too_large")

    aspect_ratio = max(width / height, height / width)
    metrics["aspect_ratio"] = aspect_ratio
    if aspect_ratio > config.max_aspect_ratio:
        soft_warn_reasons.append("extreme_aspect_ratio")

    gray = np.asarray(image.convert("L"), dtype=np.float32)
    brightness = float(gray.mean() / 255.0)
    blur_variance = _laplacian_variance(gray)
    metrics["brightness"] = brightness
    metrics["blur_variance"] = blur_variance

    if brightness < config.min_brightness_warn or brightness > config.max_brightness_warn:
        soft_warn_reasons.append("extreme_brightness")
    if blur_variance < config.min_blur_variance_warn:
        soft_warn_reasons.append("blurry_image")

    try:
        with open(annotation_path, "r", encoding="utf-8") as file_handle:
            annotation_payload = json.load(file_handle)
    except json.JSONDecodeError:
        hard_fail_reasons.append("annotation_json_invalid")
        return CheckResult(hard_fail_reasons, soft_warn_reasons, metrics)

    annotations = annotation_payload.get("annotations")
    if not isinstance(annotations, list):
        hard_fail_reasons.append("annotations_not_list")
        annotations = []

    if len(annotations) < config.min_annotations:
        hard_fail_reasons.append("too_few_annotations")

    valid_bbox_count = 0
    bbox_area_ratios: list[float] = []
    for annotation in annotations:
        bbox = annotation.get("bbox") if isinstance(annotation, dict) else None
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        try:
            x, y, box_width, box_height = [float(value) for value in bbox]
        except (TypeError, ValueError):
            continue

        if box_width <= 0 or box_height <= 0:
            continue

        valid_bbox_count += 1
        area_ratio = (box_width * box_height) / float(width * height)
        bbox_area_ratios.append(area_ratio)

        if area_ratio < config.min_bbox_area_ratio:
            hard_fail_reasons.append("bbox_area_too_small")
        if area_ratio > config.max_bbox_area_ratio:
            hard_fail_reasons.append("bbox_area_too_large")
        if area_ratio < config.tiny_bbox_warn_ratio:
            soft_warn_reasons.append("tiny_bbox")
        if area_ratio > config.huge_bbox_warn_ratio:
            soft_warn_reasons.append("huge_bbox")

        if x + box_width < 0 or y + box_height < 0 or x > width or y > height:
            soft_warn_reasons.append("bbox_outside_image")

    metrics["annotation_count"] = len(annotations)
    metrics["valid_bbox_count"] = valid_bbox_count
    if bbox_area_ratios:
        metrics["bbox_area_ratio_min"] = min(bbox_area_ratios)
        metrics["bbox_area_ratio_max"] = max(bbox_area_ratios)
        metrics["bbox_area_ratio_mean"] = float(sum(bbox_area_ratios) / len(bbox_area_ratios))

    if valid_bbox_count == 0:
        hard_fail_reasons.append("no_valid_bbox")

    sha256_digest = _sha256_file(image_path)
    average_hash = _average_hash(image_path)
    metrics["sha256"] = sha256_digest
    metrics["ahash"] = average_hash

    if sha256_digest in seen_sha256:
        hard_fail_reasons.append("exact_duplicate_image")
    else:
        seen_sha256.add(sha256_digest)

    near_duplicate = any(
        _hamming_distance(average_hash, previous_hash) <= config.max_near_dup_distance
        for previous_hash in seen_ahash
    )
    if near_duplicate:
        soft_warn_reasons.append("near_duplicate_image")
    seen_ahash.append(average_hash)

    hard_fail_reasons = sorted(set(hard_fail_reasons))
    soft_warn_reasons = sorted(set(soft_warn_reasons))
    return CheckResult(hard_fail_reasons, soft_warn_reasons, metrics)
