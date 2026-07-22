#!/usr/bin/env python3
"""Aggregate direct source-scene reports and camera-ablation event results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ARMS = ("global_raw", "visibility_raw", "visibility_highlight", "visibility_highlight_global")
METRICS = ("collision", "oob", "support")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    cases = _read_json(args.cases).get("cases", [])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source_rows = [_source_row(case, args.source_root, args.cases.parent) for case in cases]
    _write_tsv(args.out_dir / "source_metrics.tsv", source_rows)

    event_rows: list[dict[str, Any]] = []
    case_metadata = {case["case_id"]: case for case in cases}
    for case_id, case in case_metadata.items():
        path = args.ablation_root / case_id / "per_event.tsv"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                event_rows.append(
                    {
                        "case_id": case_id,
                        "base_case_id": case["base_case_id"],
                        "family": case["family"],
                        **row,
                    }
                )
    _write_tsv(args.out_dir / "camera_per_event.tsv", event_rows)

    summaries = _camera_summaries(event_rows)
    _write_tsv(args.out_dir / "camera_summary.tsv", summaries)
    contrasts = _controlled_contrasts(summaries)
    _write_tsv(args.out_dir / "controlled_contrasts.tsv", contrasts)
    (args.out_dir / "combined_results.json").write_text(
        json.dumps(
            {
                "source_metrics": source_rows,
                "camera_per_event": event_rows,
                "camera_summary": summaries,
                "controlled_contrasts": contrasts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"source cases: {len(source_rows)}")
    print(f"camera event rows: {len(event_rows)}")
    print(f"results: {args.out_dir}")


def _source_row(case: dict[str, Any], root: Path, fixture_root: Path) -> dict[str, Any]:
    fixture_dir = fixture_root / case["fixture_dir"]
    distortion = _read_json(fixture_dir / "distortion_manifest.json")
    annotation_path = fixture_dir / "reference_annotation.json"
    annotation = _read_json(annotation_path) if annotation_path.is_file() else {}
    expected = distortion.get("expected", {})
    oar_relations = annotation.get("oar_relations")
    oar_relations = oar_relations if isinstance(oar_relations, list) else []
    expected_oar_ids = expected.get("oar_invalid_relation_ids")
    if not isinstance(expected_oar_ids, list):
        expected_oar_ids = [
            _relation_id_at(oar_relations, int(value), family="oar")
            for value in expected.get("oar_invalid_relation_indices", [])
        ]
    expected_events = {
        "collision": sorted(
            "|".join(sorted(str(value) for value in pair))
            for pair in expected.get("collision_invalid_pairs", [])
        ),
        "oob": sorted(str(value) for value in expected.get("oob_invalid_object_ids", [])),
        "support": sorted(str(value) for value in expected.get("support_invalid_object_ids", [])),
        "oar": sorted(str(value) for value in expected_oar_ids),
    }
    expected_fields = {
        f"expected_{metric}_invalid": json.dumps(values, separators=(",", ":"))
        for metric, values in expected_events.items()
    }
    report_path = root / case["case_id"] / "evaluation_report.json"
    if not report_path.is_file():
        return {
            "case_id": case["case_id"],
            "base_case_id": case["base_case_id"],
            "family": case["family"],
            "status": "missing_report",
            **expected_fields,
        }
    report = _read_json(report_path)
    categories = report.get("category_reports", {})
    reports = report.get("reports", {})
    generic = reports.get("generic_validity", {}).get("metrics", {})
    row: dict[str, Any] = {
        "case_id": case["case_id"],
        "base_case_id": case["base_case_id"],
        "family": case["family"],
        "status": "completed",
        "target_object_fraction": case.get("severity", {}).get("target_object_fraction"),
        "target_relation_fraction": case.get("severity", {}).get("target_relation_fraction"),
        "benchmark_score": report.get("benchmark_score"),
        "prompt_fidelity": categories.get("prompt_fidelity", {}).get("score"),
        "structural_validity": categories.get("structural_validity", {}).get("score"),
        "visual_quality": categories.get("visual_quality", {}).get("score"),
        "oor_score": reports.get("oor", {}).get("score"),
        "oar_score": reports.get("oar", {}).get("score"),
        **expected_fields,
    }
    for metric in METRICS:
        metric_report = generic.get(metric, {})
        observed = _invalid_event_ids(metric, metric_report)
        row[f"{metric}_status"] = metric_report.get("status")
        row[f"{metric}_score"] = metric_report.get("score")
        row[f"{metric}_requires_vlm"] = metric_report.get("requires_vlm_count")
        row[f"observed_{metric}_invalid"] = json.dumps(observed, separators=(",", ":"))
        row[f"{metric}_invalid_verdicts"] = len(observed)
        row[f"{metric}_exact_match"] = observed == expected_events[metric]
    oar_checks = reports.get("oar", {}).get("checks", [])
    observed_oar = [
        str(check.get("relation_id") or f"oar_{index:03d}")
        for index, check in enumerate(oar_checks if isinstance(oar_checks, list) else [])
        if isinstance(check, dict) and check.get("passed") is False
    ]
    row["observed_oar_invalid"] = json.dumps(observed_oar, separators=(",", ":"))
    row["oar_exact_match"] = observed_oar == expected_events["oar"]
    row["all_controlled_metrics_exact_match"] = all(
        bool(row[f"{metric}_exact_match"]) for metric in METRICS
    ) and bool(row["oar_exact_match"])
    return row


def _relation_id_at(relations: list[Any], index: int, *, family: str) -> str:
    if 0 <= index < len(relations) and isinstance(relations[index], dict):
        relation_id = str(relations[index].get("relation_id") or "").strip()
        if relation_id:
            return relation_id
    return f"{family}_{index:03d}"


def _invalid_event_ids(metric: str, metric_report: dict[str, Any]) -> list[str]:
    items = metric_report.get("pairs") if isinstance(metric_report.get("pairs"), list) else metric_report.get("objects", [])
    event_ids: list[str] = []
    for item in items if isinstance(items, list) else []:
        judge = item.get("judge_result") if isinstance(item, dict) else None
        verdict = str(
            (judge or {}).get("verdict")
            or (judge or {}).get("label")
            or (item.get("final_verdict") if isinstance(item, dict) else None)
            or ""
        ).lower()
        if verdict != "invalid":
            continue
        if metric == "collision":
            event_ids.append(
                "|".join(sorted([str(item.get("object_a")), str(item.get("object_b"))]))
            )
        else:
            event_ids.append(str(item.get("object_id")))
    return sorted(event_ids)


def _camera_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for subset in ("all", str(row.get("family") or "unknown")):
            grouped[(subset, str(row.get("arm")), str(row.get("metric")))].append(row)
            grouped[(subset, str(row.get("arm")), "overall")].append(row)
    summaries: list[dict[str, Any]] = []
    subsets = sorted({key[0] for key in grouped})
    for subset in subsets:
        for arm in ARMS:
            for metric in ("overall", *METRICS):
                selected = grouped.get((subset, arm, metric), [])
                total = len(selected)
                resolved = sum(_int(row.get("resolved")) for row in selected)
                correct = sum(_int(row.get("match")) for row in selected)
                estimated = sum(_float(row.get("estimated_uncached_seconds")) for row in selected)
                summaries.append(
                    {
                        "subset": subset,
                        "arm": arm,
                        "metric": metric,
                        "total": total,
                        "resolved": resolved,
                        "correct": correct,
                        "accuracy_all": correct / total if total else None,
                        "coverage": resolved / total if total else None,
                        "tp": sum(row.get("gt_label") == "invalid" and row.get("predicted_label") == "invalid" for row in selected),
                        "fp": sum(row.get("gt_label") == "valid" and row.get("predicted_label") == "invalid" for row in selected),
                        "fn": sum(row.get("gt_label") == "invalid" and row.get("predicted_label") == "valid" for row in selected),
                        "tn": sum(row.get("gt_label") == "valid" and row.get("predicted_label") == "valid" for row in selected),
                        "mean_estimated_seconds": estimated / total if total else None,
                        "mean_images": sum(_float(row.get("image_count")) for row in selected) / total if total else None,
                    }
                )
    return summaries


def _controlled_contrasts(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["subset"], row["arm"], row["metric"]): row for row in summaries}
    comparisons = (
        ("camera_selection", "global_raw", "visibility_raw"),
        ("highlight", "visibility_raw", "visibility_highlight"),
        ("highlighted_global_context", "visibility_highlight", "visibility_highlight_global"),
    )
    rows: list[dict[str, Any]] = []
    for subset in sorted({row["subset"] for row in summaries}):
        for metric in ("overall", *METRICS):
            for variable, baseline_arm, treatment_arm in comparisons:
                baseline = lookup[(subset, baseline_arm, metric)]
                treatment = lookup[(subset, treatment_arm, metric)]
                rows.append(
                    {
                        "subset": subset,
                        "metric": metric,
                        "changed_variable": variable,
                        "baseline": baseline_arm,
                        "treatment": treatment_arm,
                        "event_count": treatment["total"],
                        "accuracy_delta": _delta(treatment["accuracy_all"], baseline["accuracy_all"]),
                        "coverage_delta": _delta(treatment["coverage"], baseline["coverage"]),
                        "mean_estimated_seconds_delta": _delta(
                            treatment["mean_estimated_seconds"], baseline["mean_estimated_seconds"]
                        ),
                        "mean_images_delta": _delta(treatment["mean_images"], baseline["mean_images"]),
                    }
                )
    return rows


def _delta(left: Any, right: Any) -> float | None:
    return None if left is None or right is None else float(left) - float(right)


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    return int(_float(value))


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


if __name__ == "__main__":
    main()
