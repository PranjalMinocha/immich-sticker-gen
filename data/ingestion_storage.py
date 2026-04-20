import os
import shutil

from ingestion_config import IngestionConfig


class StorageClient:
    def put_file(self, local_path: str, key: str) -> None:
        raise NotImplementedError

    def put_bytes(self, body: bytes, key: str) -> None:
        raise NotImplementedError


class LocalStorageClient(StorageClient):
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def _destination(self, key: str) -> str:
        key = key.lstrip("/")
        return os.path.join(self.root, key)

    def put_file(self, local_path: str, key: str) -> None:
        destination = self._destination(key)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy2(local_path, destination)

    def put_bytes(self, body: bytes, key: str) -> None:
        destination = self._destination(key)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, "wb") as file_handle:
            file_handle.write(body)


class S3StorageClient(StorageClient):
    def __init__(self, config: IngestionConfig):
        import boto3

        self.bucket = config.raw_bucket
        self.s3 = boto3.client(
            "s3",
            endpoint_url=config.s3_endpoint,
            aws_access_key_id=config.s3_access_key,
            aws_secret_access_key=config.s3_secret_key,
        )

    def put_file(self, local_path: str, key: str) -> None:
        self.s3.upload_file(local_path, self.bucket, key)

    def put_bytes(self, body: bytes, key: str) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=body)


def build_storage_client(config: IngestionConfig) -> StorageClient:
    if config.storage_backend == "local":
        return LocalStorageClient(config.local_store_root)
    if config.storage_backend == "s3":
        return S3StorageClient(config)
    raise ValueError(f"Unsupported storage backend: {config.storage_backend}")
