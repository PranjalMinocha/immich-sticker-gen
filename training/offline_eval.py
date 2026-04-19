from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class OfflineEvalThresholds:
    min_dice: float
    min_iou: float
    max_runtime_seconds: int


@dataclass(frozen=True)
class OfflineEvalMetrics:
    dice: float
    iou: float
    runtime_seconds: int


def evaluate_quality_gates(metrics: OfflineEvalMetrics, thresholds: OfflineEvalThresholds) -> Dict[str, bool]:
    pass_dice = metrics.dice >= thresholds.min_dice
    pass_iou = metrics.iou >= thresholds.min_iou
    pass_runtime = metrics.runtime_seconds <= thresholds.max_runtime_seconds
    passed = pass_dice and pass_iou and pass_runtime
    return {
        "passed": passed,
        "passDice": pass_dice,
        "passIou": pass_iou,
        "passRuntime": pass_runtime,
    }
