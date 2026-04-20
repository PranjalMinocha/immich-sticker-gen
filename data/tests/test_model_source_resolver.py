import os
import unittest

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model_source_resolver import resolve_pretrained_model_source


class _ModelVersion:
    def __init__(self, version: str):
        self.version = version


class _ClientAlias:
    def __init__(self, tracking_uri=None):
        pass

    def get_model_version_by_alias(self, name, alias):
        return _ModelVersion("7")


class _ClientLatest:
    def __init__(self, tracking_uri=None):
        pass

    def get_model_version_by_alias(self, name, alias):
        raise RuntimeError("alias missing")

    def search_model_versions(self, query):
        return [_ModelVersion("3"), _ModelVersion("12")]


class _ClientEmpty:
    def __init__(self, tracking_uri=None):
        pass

    def get_model_version_by_alias(self, name, alias):
        raise RuntimeError("alias missing")

    def search_model_versions(self, query):
        return []


class _FakeS3Client:
    def __init__(self, existing_uris=None):
        self._existing = set(existing_uris or [])

    def head_object(self, Bucket, Key):
        uri = f"s3://{Bucket}/{Key}"
        if uri not in self._existing:
            raise RuntimeError("missing")


class ModelSourceResolverTests(unittest.TestCase):
    def test_prefers_object_store_checkpoint_when_present(self):
        resolved = resolve_pretrained_model_source(
            tracking_uri="http://mlflow:5000",
            model_name="immich-sticker-mobilesam",
            preferred_alias="Production",
            bootstrap_model_uri="mlflow-artifacts:/x",
            object_store_model_uri="s3://models/production/mobile_sam.pt",
            object_store_client=_FakeS3Client({"s3://models/production/mobile_sam.pt"}),
            client_factory=_ClientAlias,
        )
        self.assertEqual(resolved.strategy, "object_store_checkpoint")
        self.assertEqual(resolved.source_uri, "s3://models/production/mobile_sam.pt")

    def test_skips_object_store_checkpoint_when_missing(self):
        resolved = resolve_pretrained_model_source(
            tracking_uri="http://mlflow:5000",
            model_name="immich-sticker-mobilesam",
            preferred_alias="Production",
            bootstrap_model_uri="mlflow-artifacts:/x",
            object_store_model_uri="s3://models/production/mobile_sam.pt",
            object_store_client=_FakeS3Client(),
            client_factory=_ClientAlias,
        )
        self.assertEqual(resolved.strategy, "registry_alias")

    def test_prefers_alias_when_available(self):
        resolved = resolve_pretrained_model_source(
            tracking_uri="http://mlflow:5000",
            model_name="immich-sticker-mobilesam",
            preferred_alias="Production",
            bootstrap_model_uri="mlflow-artifacts:/x",
            client_factory=_ClientAlias,
        )
        self.assertEqual(resolved.strategy, "registry_alias")
        self.assertEqual(resolved.model_version, "7")

    def test_falls_back_to_latest_registry_version(self):
        resolved = resolve_pretrained_model_source(
            tracking_uri="http://mlflow:5000",
            model_name="immich-sticker-mobilesam",
            preferred_alias="Production",
            bootstrap_model_uri="mlflow-artifacts:/x",
            client_factory=_ClientLatest,
        )
        self.assertEqual(resolved.strategy, "registry_latest_version")
        self.assertEqual(resolved.model_version, "12")

    def test_falls_back_to_bootstrap_if_registry_empty(self):
        resolved = resolve_pretrained_model_source(
            tracking_uri="http://mlflow:5000",
            model_name="immich-sticker-mobilesam",
            preferred_alias="Production",
            bootstrap_model_uri="mlflow-artifacts:/2/run/artifacts/checkpoints/mobile_sam_full.pt",
            client_factory=_ClientEmpty,
        )
        self.assertEqual(resolved.strategy, "bootstrap_uri")


if __name__ == "__main__":
    unittest.main()
