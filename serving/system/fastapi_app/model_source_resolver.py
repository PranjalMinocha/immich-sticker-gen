from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ResolvedModelSource:
    source_uri: str
    strategy: str
    model_name: str
    model_version: Optional[str]


def resolve_pretrained_model_source(
    tracking_uri: str,
    model_name: str,
    preferred_alias: str,
    bootstrap_model_uri: Optional[str],
    object_store_model_uri: Optional[str] = None,
    object_store_client=None,
    client_factory=None,
) -> ResolvedModelSource:
    if object_store_model_uri and _object_uri_exists(object_store_model_uri, object_store_client):
        return ResolvedModelSource(
            source_uri=object_store_model_uri,
            strategy="object_store_checkpoint",
            model_name=model_name,
            model_version=None,
        )

    if client_factory is None:
        from mlflow.tracking import MlflowClient

        client_factory = MlflowClient
    client = client_factory(tracking_uri=tracking_uri)

    if preferred_alias:
        try:
            mv = client.get_model_version_by_alias(model_name, preferred_alias)
            return ResolvedModelSource(
                source_uri=f"models:/{model_name}@{preferred_alias}",
                strategy="registry_alias",
                model_name=model_name,
                model_version=str(mv.version),
            )
        except Exception:
            pass

    try:
        versions = list(client.search_model_versions(f"name = '{model_name}'"))
    except Exception:
        versions = []

    if versions:
        latest = max(versions, key=lambda mv: int(mv.version))
        return ResolvedModelSource(
            source_uri=f"models:/{model_name}/{latest.version}",
            strategy="registry_latest_version",
            model_name=model_name,
            model_version=str(latest.version),
        )

    if bootstrap_model_uri:
        return ResolvedModelSource(
            source_uri=bootstrap_model_uri,
            strategy="bootstrap_uri",
            model_name=model_name,
            model_version=None,
        )

    raise RuntimeError(
        "Unable to resolve pretrained model source: registry has no versions and BOOTSTRAP_MODEL_URI is unset"
    )


def _object_uri_exists(object_uri: str, object_store_client=None) -> bool:
    if not object_uri.lower().startswith("s3://"):
        return False

    stripped = object_uri[5:]
    bucket, sep, key = stripped.partition("/")
    if not sep or not bucket or not key:
        return False

    if object_store_client is None:
        import boto3

        object_store_client = boto3.client("s3")

    try:
        object_store_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False
