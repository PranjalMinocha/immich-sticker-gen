from offline_eval import OfflineEvalMetrics, OfflineEvalThresholds, evaluate_quality_gates


def test_quality_gate_passes_when_all_metrics_meet_thresholds() -> None:
    metrics = OfflineEvalMetrics(dice=0.9, iou=0.8, runtime_seconds=100)
    thresholds = OfflineEvalThresholds(min_dice=0.8, min_iou=0.7, max_runtime_seconds=200)

    result = evaluate_quality_gates(metrics, thresholds)

    assert result["passed"] is True
    assert result["passDice"] is True
    assert result["passIou"] is True
    assert result["passRuntime"] is True


def test_quality_gate_fails_when_any_metric_misses_threshold() -> None:
    metrics = OfflineEvalMetrics(dice=0.9, iou=0.6, runtime_seconds=100)
    thresholds = OfflineEvalThresholds(min_dice=0.8, min_iou=0.7, max_runtime_seconds=200)

    result = evaluate_quality_gates(metrics, thresholds)

    assert result["passed"] is False
    assert result["passDice"] is True
    assert result["passIou"] is False
    assert result["passRuntime"] is True
