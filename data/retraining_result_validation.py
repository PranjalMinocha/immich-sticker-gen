from typing import Any, Dict


def validate_training_result(payload: Dict[str, Any]) -> tuple[bool, str]:
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
    return True, ""
