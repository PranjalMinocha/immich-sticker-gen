def should_trigger_retraining(ready_count: int, threshold: int) -> bool:
    return ready_count >= threshold
