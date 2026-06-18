from __future__ import annotations


def compute_case_metrics(history: list[dict], max_repair_iterations: int = 0) -> dict:
    if not history:
        raise ValueError("Cannot compute metrics without evaluation history.")

    final = history[-1]
    evaluation = final.get("evaluation") or {}
    if isinstance(evaluation, dict) and isinstance(evaluation.get("metrics"), dict):
        return dict(evaluation["metrics"])
    if isinstance(final.get("metrics"), dict):
        return dict(final["metrics"])
    raise ValueError("Evaluation history does not contain v0 case metrics.")
