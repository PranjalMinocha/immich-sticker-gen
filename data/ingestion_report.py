import json
import os
from collections import Counter


def write_jsonl(file_path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file_handle:
        for row in rows:
            file_handle.write(json.dumps(row, sort_keys=True))
            file_handle.write("\n")


def build_summary(rows: list[dict]) -> dict:
    total = len(rows)
    hard_failed = sum(1 for row in rows if row["status"] == "hard_fail")
    soft_warned = sum(1 for row in rows if row["status"] == "soft_warn")
    passed = sum(1 for row in rows if row["status"] == "pass")

    hard_fail_reasons = Counter()
    soft_warn_reasons = Counter()
    for row in rows:
        hard_fail_reasons.update(row.get("hard_fail_reasons", []))
        soft_warn_reasons.update(row.get("soft_warn_reasons", []))

    return {
        "total_samples": total,
        "passed_samples": passed,
        "soft_warn_samples": soft_warned,
        "hard_failed_samples": hard_failed,
        "hard_fail_rate": (hard_failed / total) if total else 0.0,
        "soft_warn_rate": (soft_warned / total) if total else 0.0,
        "top_hard_fail_reasons": hard_fail_reasons.most_common(10),
        "top_soft_warn_reasons": soft_warn_reasons.most_common(10),
    }


def write_summary(file_path: str, summary: dict) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, indent=2, sort_keys=True)
