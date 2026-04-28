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
    backup_model_key: str | None = None,
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

    # Back up the current production model before overwriting.
    # A failed backup blocks the deploy so we can always roll back.
    backed_up = False
    if backup_model_key:
        try:
            s3_client.copy_object(
                Bucket=target_bucket,
                CopySource={"Bucket": target_bucket, "Key": serving_model_key},
                Key=backup_model_key,
            )
            # Verify the backup object actually landed.
            s3_client.head_object(Bucket=target_bucket, Key=backup_model_key)
            backed_up = True
            print(f"[model_deployer] Backup written: s3://{target_bucket}/{backup_model_key}")
        except Exception as exc:
            response = getattr(exc, "response", None)
            error_code = response.get("Error", {}).get("Code", "") if isinstance(response, dict) else ""
            if error_code in ("NoSuchKey", "404", "NotFound") or "NoSuchKey" in str(exc):
                # Source key absent — this is the very first deploy, no prior model to back up.
                print("[model_deployer] No existing production model to back up (first deploy).")
            else:
                raise RuntimeError(f"Failed to back up production model before deploy: {exc}") from exc

    # Upload the new model; restore backup if the upload fails.
    try:
        with downloaded_path.open("rb") as fp:
            s3_client.upload_fileobj(fp, target_bucket, serving_model_key)
    except Exception as upload_exc:
        if backed_up:
            print(f"[model_deployer] Upload failed ({upload_exc}); restoring backup.")
            try:
                s3_client.copy_object(
                    Bucket=target_bucket,
                    CopySource={"Bucket": target_bucket, "Key": backup_model_key},
                    Key=serving_model_key,
                )
            except Exception as restore_exc:
                raise RuntimeError(
                    f"Upload failed AND backup restore failed: upload={upload_exc} restore={restore_exc}"
                ) from restore_exc
        raise RuntimeError(f"Model upload failed: {upload_exc}") from upload_exc

    return {
        "artifact_uri": artifact_uri,
        "target_s3_uri": f"s3://{target_bucket}/{serving_model_key}",
        "backup_s3_uri": f"s3://{target_bucket}/{backup_model_key}" if backup_model_key else "",
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
