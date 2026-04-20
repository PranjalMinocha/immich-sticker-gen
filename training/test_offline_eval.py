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


def test_prompt_robustness_gate_blocks_when_iou_drop_too_high() -> None:
    metrics = OfflineEvalMetrics(
        dice=0.9,
        iou=0.8,
        runtime_seconds=100,
        prompt_iou_drop=0.25,
        prompt_robust_iou=0.55,
    )
    thresholds = OfflineEvalThresholds(
        min_dice=0.8,
        min_iou=0.7,
        max_runtime_seconds=200,
        enable_prompt_robustness_gate=True,
        max_prompt_iou_drop=0.15,
        min_prompt_robust_iou=0.5,
    )

    result = evaluate_quality_gates(metrics, thresholds)
    assert result["passed"] is False
    assert result["passPromptRobustness"] is False


def test_small_object_and_low_light_gates_can_pass_with_valid_subset_metrics() -> None:
    metrics = OfflineEvalMetrics(
        dice=0.9,
        iou=0.8,
        runtime_seconds=100,
        small_object_iou=0.62,
        low_light_iou=0.58,
    )
    thresholds = OfflineEvalThresholds(
        min_dice=0.8,
        min_iou=0.7,
        max_runtime_seconds=200,
        enable_small_object_gate=True,
        enable_low_light_gate=True,
        min_small_object_iou=0.55,
        min_low_light_iou=0.5,
    )

    result = evaluate_quality_gates(metrics, thresholds)
    assert result["passed"] is True
    assert result["passSmallObject"] is True
    assert result["passLowLight"] is True
