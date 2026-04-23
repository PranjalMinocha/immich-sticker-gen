import json
import os
import shutil
import tarfile
import tempfile
from typing import Any, Optional

import numpy as np


def _parse_list(raw_value: Any) -> Optional[list]:
    if isinstance(raw_value, list):
        return raw_value
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, list):
            return parsed
    return None


def extract_request_features(bbox_raw: Any, point_coords_raw: Any) -> Optional[np.ndarray]:
    bbox = _parse_list(bbox_raw)
    if bbox is None or len(bbox) != 4:
        return None

    try:
        x, y, width, height = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None

    if width <= 0 or height <= 0:
        return None

    point_coords = _parse_list(point_coords_raw)
    if point_coords is None:
        point_count = 0.0
    else:
        point_count = float(len(point_coords))

    area = width * height
    aspect = width / max(height, 1e-9)
    center_x = x + (width / 2.0)
    center_y = y + (height / 2.0)

    vector = np.array(
        [[x, y, width, height, area, aspect, center_x, center_y, point_count]],
        dtype=np.float32,
    )
    return vector


def build_online_detector(x_ref: np.ndarray, ert: int = 300, window_size: int = 25):
    from alibi_detect.cd import CVMDriftOnline

    # CVMDriftOnline uses the Cramér-von Mises statistic (scipy-based, no PyTorch needed).
    # It monitors each feature independently; any drift across the feature set is flagged.
    return CVMDriftOnline(
        x_ref,
        ert=ert,
        window_sizes=[window_size],
    )


def save_detector(detector: Any, output_dir: str) -> None:
    from alibi_detect.saving import save_detector as alibi_save_detector

    os.makedirs(output_dir, exist_ok=True)
    alibi_save_detector(detector, output_dir)


def load_detector(detector_dir: str):
    from alibi_detect.saving import load_detector as alibi_load_detector

    if not os.path.isdir(detector_dir):
        return None
    return alibi_load_detector(detector_dir)


def upload_detector_artifact(s3_client: Any, bucket: str, key: str, detector_dir: str) -> None:
    if not os.path.isdir(detector_dir):
        raise FileNotFoundError(f"Detector directory does not exist: {detector_dir}")

    with tempfile.TemporaryDirectory(prefix="drift_artifact_upload_") as temp_dir:
        archive_path = os.path.join(temp_dir, "cd.tar.gz")
        with tarfile.open(archive_path, mode="w:gz") as archive:
            archive.add(detector_dir, arcname="cd")
        s3_client.upload_file(archive_path, bucket, key)


def download_detector_artifact(s3_client: Any, bucket: str, key: str, cache_root: str) -> str:
    os.makedirs(cache_root, exist_ok=True)
    detector_dir = os.path.join(cache_root, "cd")
    archive_path = os.path.join(cache_root, "cd.tar.gz")

    if os.path.isdir(detector_dir):
        shutil.rmtree(detector_dir)
    if os.path.exists(archive_path):
        os.remove(archive_path)

    s3_client.download_file(bucket, key, archive_path)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        base_dir = os.path.abspath(cache_root)
        for member in archive.getmembers():
            name = member.name.strip()
            if name in ("", ".", "./"):
                continue
            if os.path.isabs(name):
                raise ValueError(f"Unsafe absolute tar member path detected: {member.name}")
            candidate = os.path.abspath(os.path.join(base_dir, name))
            if os.path.commonpath([base_dir, candidate]) != base_dir:
                raise ValueError(f"Unsafe tar member path detected: {member.name}")
        archive.extractall(path=cache_root)
    os.remove(archive_path)

    if not os.path.isdir(detector_dir):
        raise RuntimeError(f"Downloaded artifact did not contain expected detector directory: {detector_dir}")

    return detector_dir
