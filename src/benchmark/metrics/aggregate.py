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
    "validity_gate",
    "room_consistency_score",
    "room_consistency_score_norm",
    "object_presence_rate",
    "specified_relation_pass_rate",
    "specified_attachment_pass_rate",
    "primary_score",
]


def aggregate_case_results(case_results: Iterable[dict]) -> dict:
    results = list(case_results)
    return {
        "num_cases": len(results),
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
        value = row.get(key)
        if value is None:
            continue
        values.append(float(value))
    return values


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


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
