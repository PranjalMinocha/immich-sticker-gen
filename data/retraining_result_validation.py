from typing import Any, Dict


def validate_training_result(payload: Dict[str, Any]) -> tuple[bool, str]:
    # Structural checks
    status = payload.get("status")
    if status != "passed":
        return False, f"quality_gate_failed:{status}"
    mlflow_payload = payload.get("mlflow")
    if not isinstance(mlflow_payload, dict):
        return False, "missing_mlflow_payload"
    run_id = mlflow_payload.get("runId")
    if not run_id:
        return False, "missing_mlflow_run_id"
    if mlflow_payload.get("registered") is not True:
        return False, "model_not_registered"
    if not mlflow_payload.get("modelVersion"):
        return False, "missing_model_version"
    if not mlflow_payload.get("modelName"):
        return False, "missing_model_name"

    # Quality gate: training script must report its own gate as passed AND supply metrics.
    # A structurally valid result with bad metrics must not proceed to deployment.
    # Note: qualityGate contains pass/fail flags (passDice, passIou, …); actual metric
    # values (dice, iou, runtimeSeconds, …) live at the top-level "metrics" key.
    quality_gate = payload.get("qualityGate")
    if not isinstance(quality_gate, dict):
        return False, "missing_quality_gate_block"
    if not quality_gate.get("passed"):
        reason = quality_gate.get("reason") or quality_gate.get("failReason") or "unspecified"
        return False, f"quality_gate_not_passed:{reason}"
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        return False, "missing_metrics"

    return True, ""
