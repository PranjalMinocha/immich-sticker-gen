from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import requests


def deploy_model_from_mlflow_run(
    s3_client,
    raw_bucket: str,
    tracking_uri: str,
    run_id: str,
    model_artifact_path: str,
    serving_model_bucket: str,
    serving_model_key: str,
    local_dir: str,
) -> Dict[str, str]:
    from mlflow import artifacts as mlflow_artifacts

    artifact_uri = f"runs:/{run_id}/{model_artifact_path}"
    downloaded = mlflow_artifacts.download_artifacts(artifact_uri=artifact_uri, tracking_uri=tracking_uri, dst_path=local_dir)

    downloaded_path = Path(downloaded)
    if downloaded_path.is_dir():
        file_name = Path(model_artifact_path).name
        downloaded_path = downloaded_path / file_name

    if not downloaded_path.is_file():
        raise RuntimeError(f"Downloaded artifact is not a file: {downloaded_path}")

    target_bucket = serving_model_bucket or raw_bucket
    with downloaded_path.open("rb") as fp:
        s3_client.upload_fileobj(fp, target_bucket, serving_model_key)

    return {
        "artifact_uri": artifact_uri,
        "target_s3_uri": f"s3://{target_bucket}/{serving_model_key}",
        "downloaded_path": str(downloaded_path),
    }


def ping_serving_reload(reload_url: str, reload_token: str | None = None, timeout_seconds: float = 20.0) -> Tuple[bool, str]:
    if not reload_url:
        return False, "reload_url_unset"

    headers = {}
    if reload_token:
        headers["X-Model-Reload-Token"] = reload_token

    response = requests.post(reload_url, headers=headers, timeout=timeout_seconds)
    if response.status_code >= 300:
        return False, f"http_{response.status_code}:{response.text[:200]}"
    return True, response.text[:200]
