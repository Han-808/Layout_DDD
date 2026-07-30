from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

from benchmark.utils.io import read_json, write_json


BENCHMARK_COLUMNS = [
    "case_id",
    "model",
    "input_level",
    "scene_representation_mode",
    "overall_valid",
    "validity_gate",
    "parse_success",
    "renderable",
    "judge_success",
    "vlm_valid",
    "room_consistency_score",
    "room_consistency_score_norm",
    "object_presence_rate",
    "specified_relation_pass_rate",
    "specified_attachment_pass_rate",
    "primary_score",
    "prompt_budget_exceeded",
    "prompt_budget_error_stage",
    "prompt_tokens_est",
    "request_max_tokens",
    "context_length",
    "prompt_budget_ok",
    "generation_truncated",
    "parse_error_kind",
    "aliasing_enabled",
    "num_aliases",
    "avg_canonical_object_id_length",
    "avg_model_object_id_length",
    "avg_canonical_category_length",
    "avg_model_category_length",
    "estimated_output_token_savings",
    "hierarchy_floor_objects_requested",
    "serious_collision_count_initial",
    "serious_collision_count_final",
    "serious_collision_delta",
    "room_boundary_count_initial",
    "room_boundary_count_final",
    "room_boundary_delta",
    "boundary_count_initial",
    "boundary_count_final",
    "boundary_delta",
    "above_wall_height_count_initial",
    "above_wall_height_count_final",
    "above_wall_height_delta",
    "below_floor_count_initial",
    "below_floor_count_final",
    "below_floor_delta",
    "floating_count_initial",
    "floating_count_final",
    "floating_delta",
    "dense_collision_cluster_count",
    "dense_collision_cluster_max_size",
    "fallback_physical_flag_count",
    "fallback_metadata_conflict_count",
    "high_confidence_physical_flag_count",
    "low_confidence_physical_flag_count",
    "repair_helped_physical_flags",
    "repair_worsened_physical_flags",
]


def aggregate_case_results(case_results: Iterable[dict]) -> dict:
    results = list(case_results)
    return {
        "num_cases": len(results),
        "overall": summarize_diagnostics(results),
        "by_input_mode": summarize_by_input_mode(results),
        "failure_breakdown": failure_breakdown(results),
        "evidence_flag_rates": evidence_flag_rates(results),
        "by_input_level": summarize_by_input_level(results),
        "case_results": results,
    }


def aggregate_case_result_files(paths: Iterable[str | Path]) -> dict:
    return aggregate_case_results(read_json(path) for path in paths)


def summarize_by_input_level(case_metrics: Iterable[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for item in case_metrics:
        grouped.setdefault(str(item.get("input_level") or "unknown"), []).append(item)

    summary = {}
    for input_level, rows in grouped.items():
        entry = {
            "num_cases": len(rows),
            "primary_score_mean": _mean(_values(rows, "primary_score")),
            "primary_score_std": _std(_values(rows, "primary_score")),
            "validity_gate_rate": _mean([1.0 if row.get("validity_gate") else 0.0 for row in rows]),
            "room_consistency_score_norm_mean": _mean(_values(rows, "room_consistency_score_norm")),
        }
        for key in [
            "object_presence_rate",
            "specified_relation_pass_rate",
            "specified_attachment_pass_rate",
        ]:
            values = _values(rows, key)
            if values:
                entry[f"{key}_mean"] = _mean(values)
        summary[input_level] = entry
    return summary


def summarize_by_input_mode(case_metrics: Iterable[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for item in case_metrics:
        mode = item.get("scene_representation_mode") or item.get("input_mode") or item.get("input_level") or "unknown"
        grouped.setdefault(str(mode), []).append(item)
    return {mode: summarize_diagnostics(rows) for mode, rows in grouped.items()}


def summarize_diagnostics(rows: list[dict]) -> dict:
    return {
        "num_cases": len(rows),
        "task_error_rate": _rate(rows, "task_error"),
        "parse_success_rate": _rate(rows, "parse_success"),
        "validity_gate_rate": _rate(rows, "validity_gate"),
        "renderable_rate": _rate(rows, "renderable"),
        "judge_success_rate": _rate(rows, "judge_success"),
        "vlm_valid_rate": _rate(rows, "vlm_valid"),
        "vlm_score_mean": _mean(_values_any(rows, ["vlm_score", "room_consistency_score"])),
        "vlm_confidence_mean": _mean(_values(rows, "vlm_confidence")),
        "object_presence_rate_mean": _mean(_values(rows, "object_presence_rate")),
        "malformed_json_rate": _rate(rows, "malformed_json"),
        "generation_truncated_rate": _rate(rows, "generation_truncated"),
        "finish_reason_length_rate": _rate(rows, "finish_reason_length"),
        "prompt_budget_exceeded_rate": _rate(rows, "prompt_budget_exceeded"),
        "context_pressure_rate": _context_pressure_rate(rows),
        "mean_generation_prompt_tokens_by_mode": _mean(_values(rows, "generation_prompt_tokens_est")),
        "max_generation_prompt_tokens_by_mode": _max(_values(rows, "generation_prompt_tokens_est")),
        "mean_repair_prompt_tokens": _mean(_values(rows, "repair_prompt_tokens_est")),
        "max_repair_prompt_tokens": _max(_values(rows, "repair_prompt_tokens_est")),
        "no_renderable_objects_rate": _hard_failure_rate(rows, "no_renderable_objects"),
        "overall_valid_rate": _rate(rows, "overall_valid"),
        "exact_object_id_match_rate_mean": _mean(_values(rows, "exact_object_id_match_rate")),
        "category_match_rate_mean": _mean(_values(rows, "category_match_rate")),
        "region_assignment_rate_mean": _mean(_values(rows, "region_assignment_rate")),
        "num_region_groups_mean": _mean(_values(rows, "num_region_groups")),
        "evidence_groups_selected_mean": _mean(_values(rows, "evidence_groups_selected")),
        "mean_collision_delta": _mean(_values(rows, "serious_collision_delta")),
        "cases_collision_improved": sum(1 for row in rows if _number(row.get("serious_collision_delta")) is not None and _number(row.get("serious_collision_delta")) < 0),
        "cases_collision_worsened": sum(1 for row in rows if _number(row.get("serious_collision_delta")) is not None and _number(row.get("serious_collision_delta")) > 0),
        "cases_boundary_improved": sum(
            1
            for row in rows
            if (delta := _number_any(row, ["room_boundary_delta", "boundary_delta"])) is not None and delta < 0
        ),
        "cases_height_improved": sum(
            1
            for row in rows
            if (
                (_number(row.get("above_wall_height_delta")) or 0)
                + (_number(row.get("below_floor_delta")) or 0)
            )
            < 0
        ),
        "cases_with_fallback_metadata_conflict": sum(
            1
            for row in rows
            if _number(row.get("fallback_metadata_conflict_count")) and _number(row.get("fallback_metadata_conflict_count")) > 0
        ),
        "cases_with_dense_collision_cluster": sum(1 for row in rows if _number(row.get("dense_collision_cluster_count")) and _number(row.get("dense_collision_cluster_count")) > 0),
        "cases_with_floating_evidence": sum(1 for row in rows if _number(row.get("floating_count_final")) and _number(row.get("floating_count_final")) > 0),
    }


def failure_breakdown(rows: list[dict]) -> dict:
    counts = {
        "task_error": sum(1 for row in rows if row.get("task_error")),
        "generation_error": sum(1 for row in rows if row.get("generation_error")),
        "schema_error": sum(1 for row in rows if _flag_count(row, "layout_sanity") > 0),
        "render_error": sum(1 for row in rows if row.get("render_error")),
        "judge_error": sum(1 for row in rows if row.get("judge_error")),
        "vlm_invalid": sum(1 for row in rows if row.get("judge_success") and row.get("vlm_valid") is False),
        "prompt_budget_exceeded": sum(1 for row in rows if row.get("prompt_budget_exceeded")),
    }
    denominator = len(rows)
    return {key: {"count": count, "rate": _safe_rate(count, denominator)} for key, count in counts.items()}


def evidence_flag_rates(rows: list[dict]) -> dict:
    keys = sorted({key for row in rows for key in _flag_counts(row)})
    denominator = len(rows)
    return {key: _safe_rate(sum(1 for row in rows if _flag_count(row, key) > 0), denominator) for key in keys}


def write_benchmark_metrics_outputs(case_metrics: list[dict], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    csv_path = out / "benchmark_metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BENCHMARK_COLUMNS)
        writer.writeheader()
        for row in case_metrics:
            writer.writerow({key: _csv_value(row.get(key)) for key in BENCHMARK_COLUMNS})

    summary = aggregate_case_results(case_metrics)
    summary_path = write_json(out / "benchmark_summary.json", summary)
    return csv_path, summary_path


def write_summary_outputs(summary: dict, out_dir: str | Path, json_name: str, csv_name: str) -> tuple[Path, Path]:
    results = list(summary.get("case_results", []))
    csv_path, summary_path = write_benchmark_metrics_outputs(results, out_dir)
    if json_name != "benchmark_summary.json":
        write_json(Path(out_dir) / json_name, summary)
    if csv_name != "benchmark_metrics.csv":
        _copy_csv(csv_path, Path(out_dir) / csv_name)
    return summary_path, csv_path


def _values(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        value = _number(row.get(key))
        if value is None:
            continue
        values.append(value)
    return values


def _values_any(rows: list[dict], keys: list[str]) -> list[float]:
    values = []
    for row in rows:
        for key in keys:
            value = _number(row.get(key))
            if value is None:
                continue
            values.append(value)
            break
    return values


def _number_any(row: dict, keys: list[str]) -> float | None:
    for key in keys:
        value = _number(row.get(key))
        if value is not None:
            return value
    return None


def _number(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rate(rows: list[dict], key: str) -> float | None:
    if not rows:
        return None
    present = [row for row in rows if row.get(key) is not None]
    if not present:
        return None
    return sum(1.0 for row in present if bool(row.get(key))) / float(len(present))


def _hard_failure_rate(rows: list[dict], code: str) -> float | None:
    if not rows:
        return None
    return _safe_rate(sum(1 for row in rows if code in row.get("hard_failure_codes", [])), len(rows))


def _flag_counts(row: dict) -> dict:
    counts = row.get("evidence_flag_counts")
    return counts if isinstance(counts, dict) else {}


def _flag_count(row: dict, key: str) -> int:
    value = _flag_counts(row).get(key, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_rate(count: int, denominator: int) -> float | None:
    return None if denominator <= 0 else float(count) / float(denominator)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _max(values: list[float]) -> float | None:
    return max(values) if values else None


def _context_pressure_rate(rows: list[dict]) -> float | None:
    pressured = 0
    present = 0
    for row in rows:
        estimated = row.get("prompt_tokens_est")
        budget = row.get("prompt_budget")
        if estimated is None or budget in {None, 0}:
            continue
        present += 1
        if float(estimated) / max(1.0, float(budget)) > 0.8:
            pressured += 1
    return _safe_rate(pressured, present)


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    mean = _mean(values)
    assert mean is not None
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _csv_value(value: object) -> object:
    return "" if value is None else value


def _copy_csv(src: Path, dst: Path) -> None:
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
