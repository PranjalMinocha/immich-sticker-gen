import argparse
import datetime as dt
import json
import os
import tarfile
import tempfile

import requests
from tqdm import tqdm

from ingestion_checks import CheckResult, validate_sample
from ingestion_config import load_config, validate_storage_config
from ingestion_report import build_summary, write_jsonl, write_summary
from ingestion_storage import build_storage_client


def download_file(url: str, local_filename: str) -> str:
    print(f"Downloading {url}...")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with open(local_filename, "wb") as file_handle, tqdm(
            desc=os.path.basename(local_filename),
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                file_handle.write(chunk)
                progress.update(len(chunk))
    return local_filename


def safe_extract_tar(archive_path: str, extract_dir: str) -> None:
    base_dir = os.path.abspath(extract_dir)
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            name = member.name.strip()
            # Common harmless root markers in tar archives
            if name in ("", ".", "./"):
                continue
            # Reject absolute tar paths
            if os.path.isabs(name):
                raise ValueError(f"Unsafe absolute tar member path detected: {member.name}")
            candidate = os.path.abspath(os.path.join(base_dir, name))
            # Robust traversal protection
            if os.path.commonpath([base_dir, candidate]) != base_dir:
                raise ValueError(f"Unsafe tar member path detected: {member.name}")
        archive.extractall(path=extract_dir)


def collect_pairs(extract_dir: str) -> tuple[dict[str, str], dict[str, str]]:
    image_by_stem: dict[str, str] = {}
    annotation_by_stem: dict[str, str] = {}

    for root, _, files in os.walk(extract_dir):
        for filename in files:
            lower = filename.lower()
            absolute_path = os.path.join(root, filename)
            stem, extension = os.path.splitext(os.path.basename(filename))
            if lower.endswith(".jpg"):
                image_by_stem.setdefault(stem, absolute_path)
            elif extension.lower() == ".json":
                annotation_by_stem.setdefault(stem, absolute_path)

    return image_by_stem, annotation_by_stem


def write_reports(
    run_id: str,
    records: list[dict],
    summary: dict,
    temp_root: str,
    storage_client,
) -> None:
    manifest_name = f"ingestion_run_{run_id}.jsonl"
    summary_name = f"ingestion_summary_{run_id}.json"
    manifest_path = os.path.join(temp_root, manifest_name)
    summary_path = os.path.join(temp_root, summary_name)

    write_jsonl(manifest_path, records)
    write_summary(summary_path, summary)

    storage_client.put_file(manifest_path, f"quality/manifests/{manifest_name}")
    storage_client.put_file(summary_path, f"quality/reports/{summary_name}")


def status_from_result(check_result: CheckResult) -> str:
    if check_result.hard_fail_reasons:
        return "hard_fail"
    if check_result.soft_warn_reasons:
        return "soft_warn"
    return "pass"


def ingest_archive(archive_path: str) -> dict:
    config = load_config()
    validate_storage_config(config)
    storage_client = build_storage_client(config)
    run_id = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    print(f"Starting ingestion run {run_id} (backend={config.storage_backend})")

    seen_sha256: set[str] = set()
    seen_ahash: list[str] = []
    records: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="ingest_extract_") as extract_dir, tempfile.TemporaryDirectory(
        prefix="ingest_reports_"
    ) as report_dir:
        safe_extract_tar(archive_path, extract_dir)
        image_by_stem, annotation_by_stem = collect_pairs(extract_dir)

        all_stems = sorted(set(image_by_stem.keys()) | set(annotation_by_stem.keys()))
        print(f"Found {len(all_stems)} candidate stems in archive")

        for stem in tqdm(all_stems, desc="Validating and routing"):
            image_path = image_by_stem.get(stem)
            annotation_path = annotation_by_stem.get(stem)

            if not image_path or not annotation_path:
                reasons = ["missing_image_for_annotation"] if not image_path else ["missing_annotation_for_image"]
                record = {
                    "sample_id": stem,
                    "status": "hard_fail",
                    "hard_fail_reasons": reasons,
                    "soft_warn_reasons": [],
                    "metrics": {},
                }
                records.append(record)

                present_file = image_path or annotation_path
                if present_file:
                    target_key = f"quarantine/{reasons[0]}/{os.path.basename(present_file)}"
                    storage_client.put_file(present_file, target_key)
                continue

            check_result = validate_sample(image_path, annotation_path, config, seen_sha256, seen_ahash)
            sample_status = status_from_result(check_result)
            record = {
                "sample_id": stem,
                "status": sample_status,
                "hard_fail_reasons": check_result.hard_fail_reasons,
                "soft_warn_reasons": check_result.soft_warn_reasons,
                "metrics": check_result.metrics,
            }
            records.append(record)

            image_name = os.path.basename(image_path)
            annotation_name = os.path.basename(annotation_path)

            if sample_status == "hard_fail":
                reason = check_result.hard_fail_reasons[0]
                storage_client.put_file(image_path, f"quarantine/{reason}/images/{image_name}")
                storage_client.put_file(annotation_path, f"quarantine/{reason}/annotations/{annotation_name}")
            else:
                storage_client.put_file(image_path, f"images/{image_name}")
                storage_client.put_file(annotation_path, f"annotations/{annotation_name}")

        summary = build_summary(records)
        summary["run_id"] = run_id
        summary["storage_backend"] = config.storage_backend
        write_reports(run_id, records, summary, report_dir, storage_client)

    print(json.dumps(summary, indent=2, sort_keys=True))

    if summary["hard_fail_rate"] > config.hard_fail_rate_max:
        raise RuntimeError(
            f"Hard fail rate {summary['hard_fail_rate']:.3f} exceeded threshold {config.hard_fail_rate_max:.3f}"
        )

    if summary["soft_warn_rate"] > config.soft_warn_rate_max:
        print(
            f"Warning: soft warn rate {summary['soft_warn_rate']:.3f} exceeded threshold "
            f"{config.soft_warn_rate_max:.3f}"
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest SA-1B tar archives with data quality checks. "
            "Use --url for remote source or --archive for local testing."
        )
    )
    parser.add_argument("--url", help="HTTP(S) URL to tar archive")
    parser.add_argument("--archive", help="Local path to tar archive")
    args = parser.parse_args()

    if bool(args.url) == bool(args.archive):
        raise ValueError("Provide exactly one of --url or --archive")

    archive_path = args.archive
    downloaded = False

    if args.url:
        archive_path = os.path.abspath("dataset_chunk.tar")
        download_file(args.url, archive_path)
        downloaded = True

    try:
        ingest_archive(archive_path)
        print("Ingestion complete.")
    finally:
        if downloaded and archive_path and os.path.exists(archive_path):
            os.remove(archive_path)


if __name__ == "__main__":
    main()
