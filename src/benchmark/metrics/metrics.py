from __future__ import annotations

from collections import Counter


def compute_case_metrics(history: list[dict], max_repair_iterations: int = 0) -> dict:
    if not history:
        raise ValueError("Cannot compute metrics without evaluation history.")

    final = history[-1]
    evaluation = final.get("evaluation") or {}
    if isinstance(evaluation, dict) and isinstance(evaluation.get("metrics"), dict):
        metrics = dict(evaluation["metrics"])
        _add_physical_repair_deltas(metrics, history)
        return metrics
    if isinstance(final.get("metrics"), dict):
        metrics = dict(final["metrics"])
        _add_physical_repair_deltas(metrics, history)
        return metrics
    raise ValueError("Evaluation history does not contain v0 case metrics.")


PHYSICAL_DELTA_TYPES = [
    "serious_collision",
    "room_boundary",
    "above_wall_height",
    "below_floor",
    "floating_or_vertical_inconsistency",
]


def _add_physical_repair_deltas(metrics: dict, history: list[dict]) -> None:
    initial_flags = _physical_flags(history[0]) if history else []
    final_flags = _physical_flags(history[-1]) if history else []
    initial_counts = _flag_type_counts(initial_flags)
    final_counts = _flag_type_counts(final_flags)
    for flag_type in PHYSICAL_DELTA_TYPES:
        prefix = "floating" if flag_type == "floating_or_vertical_inconsistency" else flag_type
        metrics[f"{prefix}_count_initial"] = initial_counts.get(flag_type, 0)
        metrics[f"{prefix}_count_final"] = final_counts.get(flag_type, 0)
        metrics[f"{prefix}_delta"] = final_counts.get(flag_type, 0) - initial_counts.get(flag_type, 0)
        if flag_type == "room_boundary":
            metrics["boundary_count_initial"] = metrics["room_boundary_count_initial"]
            metrics["boundary_count_final"] = metrics["room_boundary_count_final"]
            metrics["boundary_delta"] = metrics["room_boundary_delta"]

    initial_total = sum(initial_counts.values())
    final_total = sum(final_counts.values())
    metrics["physical_flag_count_initial"] = initial_total
    metrics["physical_flag_count_final"] = final_total
    metrics["physical_flag_delta"] = final_total - initial_total
    metrics["repair_helped_physical_flags"] = final_total < initial_total if len(history) > 1 else False
    metrics["repair_worsened_physical_flags"] = final_total > initial_total if len(history) > 1 else False

    final_confidence = Counter(str(flag.get("confidence") or "unknown") for flag in final_flags if isinstance(flag, dict))
    metrics.setdefault("high_confidence_physical_flag_count", final_confidence.get("high", 0))
    metrics.setdefault("low_confidence_physical_flag_count", final_confidence.get("low", 0))
    metrics.setdefault("fallback_physical_flag_count", _fallback_flag_count(final_flags))
    metrics.setdefault("fallback_metadata_conflict_count", _flag_code_count(final_flags, "fallback_metadata_conflict"))
    metrics["dense_collision_cluster_count"] = _dense_cluster_count(history)
    metrics["dense_collision_cluster_max_size"] = _dense_cluster_max_size(history)


def _physical_flags(history_entry: dict) -> list[dict]:
    evaluation = history_entry.get("evaluation") if isinstance(history_entry, dict) else {}
    debug = evaluation.get("debug_evidence") if isinstance(evaluation, dict) else {}
    flags = debug.get("physical_flags") if isinstance(debug, dict) else []
    return [flag for flag in flags if isinstance(flag, dict)] if isinstance(flags, list) else []


def _flag_type_counts(flags: list[dict]) -> Counter:
    return Counter(str(flag.get("type") or flag.get("code") or "unknown") for flag in flags if isinstance(flag, dict))


def _fallback_flag_count(flags: list[dict]) -> int:
    count = 0
    for flag in flags:
        source_kind = str(flag.get("source_kind") or "").lower()
        confidence = str(flag.get("confidence") or "").lower()
        if source_kind in {"object_position_extent_fallback", "fallback_default", "unknown"} or confidence == "low":
            count += 1
    return count


def _flag_code_count(flags: list[dict], code: str) -> int:
    return sum(1 for flag in flags if isinstance(flag, dict) and str(flag.get("code") or "") == code)


def _dense_cluster_actions(history: list[dict]) -> list[dict]:
    actions = []
    for item in history:
        feedback = item.get("feedback") if isinstance(item, dict) else {}
        for action in feedback.get("repair_actions", []) if isinstance(feedback, dict) else []:
            if isinstance(action, dict) and action.get("action") == "spread_dense_collision_cluster":
                actions.append(action)
    return actions


def _dense_cluster_count(history: list[dict]) -> int:
    return len(_dense_cluster_actions(history))


def _dense_cluster_max_size(history: list[dict]) -> int:
    sizes = [len(action.get("objects", [])) for action in _dense_cluster_actions(history) if isinstance(action.get("objects"), list)]
    return max(sizes) if sizes else 0
