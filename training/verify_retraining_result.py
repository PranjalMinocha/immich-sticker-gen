#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict

from mlflow.tracking import MlflowClient

from retraining_result_contract import validate_result_payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Result file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Result payload is not a JSON object")
    return payload


def _artifact_exists(client: MlflowClient, run_id: str, artifact_path: str) -> bool:
    parent, _, name = artifact_path.rpartition("/")
    listed = client.list_artifacts(run_id, path=parent or None)
    for item in listed:
        if item.path == artifact_path or item.path.endswith("/" + name):
            return True
    return False


def _verify_mlflow(payload: Dict[str, Any], args: argparse.Namespace) -> None:
    mlflow_payload = payload["mlflow"]
    tracking_uri = args.tracking_uri or mlflow_payload["trackingUri"]
    run_id = mlflow_payload["runId"]

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    _require(run.info.run_id == run_id, "MLflow run id mismatch")

    for metric_name in args.require_metric:
        _require(metric_name in run.data.metrics, f"Required MLflow metric missing: {metric_name}")

    if args.require_artifact:
        _require(
            _artifact_exists(client, run_id, args.require_artifact),
            f"Required MLflow artifact missing: {args.require_artifact}",
        )

    if args.expected_orchestrator_run_id:
        tag_value = run.data.tags.get("immich_sticker_training_run_id")
        _require(
            tag_value == args.expected_orchestrator_run_id,
            "MLflow run tag immich_sticker_training_run_id mismatch",
        )

    if args.check_registry and bool(mlflow_payload.get("registered")):
        model_name = mlflow_payload["modelName"]
        model_version = str(mlflow_payload["modelVersion"])
        mv = client.get_model_version(model_name, model_version)
        _require(mv.run_id == run_id, "Model version run_id mismatch")
        for required_tag in (
            "quality_gate_passed",
            "test_dice",
            "test_iou",
            "test_boundary_f1",
            "test_prompt_iou_drop",
            "test_prompt_robust_iou",
            "runtime_seconds",
        ):
            _require(required_tag in mv.tags, f"Missing model version tag: {required_tag}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify retraining output JSON plus MLflow/model-registry state")
    parser.add_argument("--result-json", required=True, help="Path to training output JSON")
    parser.add_argument("--tracking-uri", default=None, help="Override MLflow tracking URI")
    parser.add_argument("--require-passed", action="store_true", help="Require result.status='passed'")
    parser.add_argument("--require-failed", action="store_true", help="Require result.status='failed'")
    parser.add_argument("--require-registered", action="store_true", help="Require mlflow.registered=true")
    parser.add_argument("--require-not-registered", action="store_true", help="Require mlflow.registered=false")
    parser.add_argument("--check-mlflow", action="store_true", help="Verify MLflow run existence and required fields")
    parser.add_argument("--check-registry", action="store_true", help="Verify model registry version linkage")
    parser.add_argument(
        "--require-artifact",
        default="checkpoints/mobile_sam_full.pt",
        help="Artifact path that must exist in the MLflow run",
    )
    parser.add_argument(
        "--require-metric",
        action="append",
        default=[
            "test_mean_iou_lowres",
            "test_mean_dice_lowres",
            "test_boundary_f1_lowres",
            "test_prompt_robust_iou",
            "test_prompt_iou_drop",
            "runtime_seconds",
        ],
        help="Repeatable metric names that must exist in MLflow run",
    )
    parser.add_argument("--expected-orchestrator-run-id", default=None)
    args = parser.parse_args()

    _require(not (args.require_passed and args.require_failed), "Cannot require both passed and failed status")
    _require(
        not (args.require_registered and args.require_not_registered),
        "Cannot require both registered and not registered",
    )

    payload = _read_json(Path(args.result_json))
    errors = validate_result_payload(
        payload,
        require_passed=True if args.require_passed else (False if args.require_failed else None),
        require_registered=True if args.require_registered else (False if args.require_not_registered else None),
    )
    if errors:
        raise RuntimeError("Result payload validation failed: " + "; ".join(errors))

    if args.check_mlflow:
        _verify_mlflow(payload, args)

    print("Retraining result verification passed")
    print(f"runId={payload['mlflow']['runId']} status={payload['status']} registered={payload['mlflow'].get('registered')}")


if __name__ == "__main__":
    main()
