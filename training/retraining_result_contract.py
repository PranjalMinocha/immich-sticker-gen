from typing import Any, Dict, List, Optional


def validate_result_payload(
    payload: Dict[str, Any],
    require_passed: Optional[bool] = None,
    require_registered: Optional[bool] = None,
) -> List[str]:
    errors: List[str] = []

    if not isinstance(payload, dict):
        return ["result payload must be an object"]

    status = payload.get("status")
    if status not in ("passed", "failed"):
        errors.append("status must be 'passed' or 'failed'")

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        errors.append("metrics must be an object")
    else:
        for key in ("dice", "iou", "runtimeSeconds"):
            if key not in metrics:
                errors.append(f"metrics.{key} is required")

    quality_gate = payload.get("qualityGate")
    if not isinstance(quality_gate, dict):
        errors.append("qualityGate must be an object")
    else:
        if "passed" not in quality_gate:
            errors.append("qualityGate.passed is required")

    mlflow_payload = payload.get("mlflow")
    if not isinstance(mlflow_payload, dict):
        errors.append("mlflow must be an object")
    else:
        if not mlflow_payload.get("trackingUri"):
            errors.append("mlflow.trackingUri is required")
        if not mlflow_payload.get("runId"):
            errors.append("mlflow.runId is required")
        if "registered" not in mlflow_payload:
            errors.append("mlflow.registered is required")

    if require_passed is True and status != "passed":
        errors.append("expected passed training result")
    if require_passed is False and status != "failed":
        errors.append("expected failed training result")

    if isinstance(mlflow_payload, dict) and "registered" in mlflow_payload:
        registered = bool(mlflow_payload.get("registered"))
        if require_registered is True and not registered:
            errors.append("expected mlflow.registered=true")
        if require_registered is False and registered:
            errors.append("expected mlflow.registered=false")
        if registered:
            if not mlflow_payload.get("modelName"):
                errors.append("mlflow.modelName is required when registered=true")
            if not mlflow_payload.get("modelVersion"):
                errors.append("mlflow.modelVersion is required when registered=true")

    return errors
