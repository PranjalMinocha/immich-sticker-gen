from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class OfflineEvalThresholds:
    min_dice: float
    min_iou: float
    max_runtime_seconds: int
    min_boundary_f1: Optional[float] = None
    max_prompt_iou_drop: Optional[float] = None
    min_prompt_robust_iou: Optional[float] = None
    min_small_object_iou: Optional[float] = None
    min_low_light_iou: Optional[float] = None
    enable_boundary_gate: bool = False
    enable_prompt_robustness_gate: bool = False
    enable_small_object_gate: bool = False
    enable_low_light_gate: bool = False


@dataclass(frozen=True)
class OfflineEvalMetrics:
    dice: float
    iou: float
    runtime_seconds: int
    boundary_f1: Optional[float] = None
    prompt_iou_drop: Optional[float] = None
    prompt_robust_iou: Optional[float] = None
    small_object_iou: Optional[float] = None
    low_light_iou: Optional[float] = None


def evaluate_quality_gates(metrics: OfflineEvalMetrics, thresholds: OfflineEvalThresholds) -> Dict[str, bool]:
    pass_dice = metrics.dice >= thresholds.min_dice
    pass_iou = metrics.iou >= thresholds.min_iou
    pass_runtime = metrics.runtime_seconds <= thresholds.max_runtime_seconds

    pass_boundary = True
    if thresholds.enable_boundary_gate:
        pass_boundary = (
            metrics.boundary_f1 is not None
            and thresholds.min_boundary_f1 is not None
            and metrics.boundary_f1 >= thresholds.min_boundary_f1
        )

    pass_prompt_robustness = True
    if thresholds.enable_prompt_robustness_gate:
        drop_ok = (
            metrics.prompt_iou_drop is not None
            and thresholds.max_prompt_iou_drop is not None
            and metrics.prompt_iou_drop <= thresholds.max_prompt_iou_drop
        )
        robust_ok = (
            metrics.prompt_robust_iou is not None
            and thresholds.min_prompt_robust_iou is not None
            and metrics.prompt_robust_iou >= thresholds.min_prompt_robust_iou
        )
        pass_prompt_robustness = drop_ok and robust_ok

    pass_small_object = True
    if thresholds.enable_small_object_gate:
        pass_small_object = (
            metrics.small_object_iou is not None
            and thresholds.min_small_object_iou is not None
            and metrics.small_object_iou >= thresholds.min_small_object_iou
        )

    pass_low_light = True
    if thresholds.enable_low_light_gate:
        pass_low_light = (
            metrics.low_light_iou is not None
            and thresholds.min_low_light_iou is not None
            and metrics.low_light_iou >= thresholds.min_low_light_iou
        )

    passed = (
        pass_dice
        and pass_iou
        and pass_runtime
        and pass_boundary
        and pass_prompt_robustness
        and pass_small_object
        and pass_low_light
    )
    result = {
        "passed": passed,
        "passDice": pass_dice,
        "passIou": pass_iou,
        "passRuntime": pass_runtime,
        "passBoundary": pass_boundary,
        "passPromptRobustness": pass_prompt_robustness,
        "passSmallObject": pass_small_object,
        "passLowLight": pass_low_light,
    }
    return result
