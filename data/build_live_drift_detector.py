import argparse
import csv
import json
import os
import tempfile
from io import StringIO
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from live_drift import build_online_detector, extract_request_features, save_detector, upload_detector_artifact


S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
RAW_BUCKET = os.environ.get("RAW_BUCKET")


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    prefix = "s3://"
    if not isinstance(uri, str) or not uri.startswith(prefix):
        raise ValueError(f"Invalid S3 URI: {uri}")

    remainder = uri[len(prefix) :]
    bucket, sep, key = remainder.partition("/")
    if not sep or not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return bucket, key


def annotation_to_vector(annotation: Dict) -> Optional[np.ndarray]:
    if not isinstance(annotation, dict):
        return None

    bbox = annotation.get("bbox")
    point_coords = annotation.get("point_coords")

    if point_coords is None and isinstance(bbox, list) and len(bbox) == 4:
        try:
            x, y, width, height = [float(value) for value in bbox]
            point_coords = [[x + (width / 2.0), y + (height / 2.0)]]
        except (TypeError, ValueError):
            point_coords = None

    return extract_request_features(bbox, point_coords)


def _s3_client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )


def _read_manifest_rows(s3_client, bucket: str, manifest_key: str) -> Iterable[Dict[str, str]]:
    obj = s3_client.get_object(Bucket=bucket, Key=manifest_key)
    csv_text = obj["Body"].read().decode("utf-8")
    reader = csv.DictReader(StringIO(csv_text))
    for row in reader:
        yield row


def _load_json_payload(s3_client, bucket: str, key: str) -> Optional[Dict]:
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        text = obj["Body"].read().decode("utf-8")
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def build_and_save(
    artifact_bucket: str,
    artifact_key: str,
    manifest_key: str,
    max_vectors: int,
    window_size: int,
    ert: int,
) -> None:
    if not RAW_BUCKET:
        raise RuntimeError("RAW_BUCKET is required to load training manifest")
    if not artifact_bucket:
        raise RuntimeError("artifact_bucket is required")
    if not artifact_key:
        raise RuntimeError("artifact_key is required")

    s3_client = _s3_client()
    vectors = []
    manifests_seen = 0
    annotations_seen = 0
    annotations_invalid = 0

    for row in _read_manifest_rows(s3_client, RAW_BUCKET, manifest_key):
        manifests_seen += 1
        annotation_uri = row.get("annotation_uri")
        if not annotation_uri:
            continue

        try:
            ann_bucket, ann_key = parse_s3_uri(annotation_uri)
        except ValueError:
            continue

        payload = _load_json_payload(s3_client, ann_bucket, ann_key)
        if payload is None:
            continue

        annotations = payload.get("annotations")
        if not isinstance(annotations, list):
            continue

        for annotation in annotations:
            annotations_seen += 1
            vector = annotation_to_vector(annotation)
            if vector is None:
                annotations_invalid += 1
                continue
            vectors.append(vector[0])
            if len(vectors) >= max_vectors:
                break

        if len(vectors) >= max_vectors:
            break

    if len(vectors) < max(50, window_size):
        raise RuntimeError(
            "Not enough valid training-reference vectors to build detector. "
            f"Need at least {max(50, window_size)}, got {len(vectors)}"
        )

    x_ref = np.asarray(vectors, dtype=np.float32)
    detector = build_online_detector(x_ref=x_ref, ert=ert, window_size=window_size)

    with tempfile.TemporaryDirectory(prefix="drift_detector_build_") as temp_dir:
        detector_dir = os.path.join(temp_dir, "cd")
        save_detector(detector, detector_dir)
        upload_detector_artifact(s3_client, artifact_bucket, artifact_key, detector_dir)

    print(
        "Uploaded drift detector artifact to s3://{}/{} using {} vectors from {} manifest rows "
        "(annotations_seen={}, invalid_annotations={})".format(
            artifact_bucket,
            artifact_key,
            len(vectors),
            manifests_seen,
            annotations_seen,
            annotations_invalid,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and upload online drift detector from training manifest")
    parser.add_argument(
        "--artifact-bucket",
        default=os.environ.get("DRIFT_ARTIFACT_BUCKET", RAW_BUCKET),
        help="S3 bucket to store detector artifact",
    )
    parser.add_argument(
        "--artifact-key",
        default=os.environ.get("DRIFT_ARTIFACT_KEY", "drift_detectors/initial_training/cd.tar.gz"),
        help="S3 object key for detector artifact tarball",
    )
    parser.add_argument(
        "--manifest-key",
        default=os.environ.get("DRIFT_REF_MANIFEST_KEY", "dataset_manifests/train_manifest.csv"),
        help="S3 key for training manifest CSV",
    )
    parser.add_argument(
        "--max-vectors",
        type=int,
        default=int(os.environ.get("DRIFT_REF_MAX_ROWS", "100000")),
        help="Max reference vectors to load from training manifest annotations",
    )
    parser.add_argument("--window-size", type=int, default=25, help="MMD online detector window size")
    parser.add_argument("--ert", type=int, default=300, help="Expected run time between false positives")
    args = parser.parse_args()

    build_and_save(
        args.artifact_bucket,
        args.artifact_key,
        args.manifest_key,
        args.max_vectors,
        args.window_size,
        args.ert,
    )


if __name__ == "__main__":
    main()
