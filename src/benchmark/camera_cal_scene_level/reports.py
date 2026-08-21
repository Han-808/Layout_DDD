"""Report, terminal-state, and summary projections for camera-cal runs.

The historical script remains the compatibility façade.  This module owns no
runner import and performs no evaluation; callers can inject the façade's
current IO, telemetry, and policy helpers through :class:`ReportsDependencies`
to preserve monkeypatch behavior and exact artifact ordering.
"""

from __future__ import annotations

from copy import deepcopy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from benchmark.camera_cal_scene_level.io import (
    atomic_write_json as _io_atomic_write_json,
    read_json as _io_read_json,
    utc_now as _io_utc_now,
)
from benchmark.camera_cal_scene_level.telemetry import (
    api_usage_summary as _telemetry_api_usage_summary,
    empty_metric_summary as _telemetry_empty_metric_summary,
    initial_image_count as _telemetry_initial_image_count,
    metric_failure_counts as _telemetry_metric_failure_counts,
    read_api_call_records as _telemetry_read_api_call_records,
    telemetry_by_metric as _telemetry_by_metric,
)
from benchmark.evaluator.scoring import MIN_PUBLISHABLE_SCORE_COVERAGE
from benchmark.visual_judge.contracts import (
    response_schema_audit_from_exception as _response_schema_audit_from_exception,
)


CASE_SCHEMA_VERSION = "camera_cal_scene_level_case_v5"
SUMMARY_SCHEMA_VERSION = "camera_cal_scene_level_summary_v2"


@dataclass(frozen=True)
class ReportsDependencies:
    """Runtime helpers that a compatibility façade may inject.

    The dependency object has no defaults on purpose.  The factory resolves
    package globals at call time, while a runner façade can construct one from
    its current globals for every call.  This keeps existing monkeypatch points
    live without importing or retaining the runner module here.
    """

    read_json: Callable[[Path], dict[str, Any]]
    atomic_write_json: Callable[[Path, Any], None]
    utc_now: Callable[[], str]
    api_usage_summary: Callable[[list[dict[str, Any]]], dict[str, Any]]
    read_api_call_records: Callable[[Path], list[dict[str, Any]]]
    empty_metric_summary: Callable[..., dict[str, Any]]
    telemetry_by_metric: Callable[
        [dict[str, Any]], dict[str, dict[str, int]]
    ]
    initial_image_count: Callable[[dict[str, Any]], int]
    metric_failure_counts: Callable[[dict[str, Any]], dict[str, int]]
    response_schema_audit_from_exception: Callable[
        [Exception], dict[str, Any] | None
    ]
    metric_score_is_publishable: Callable[[dict[str, Any]], bool]
    case_schema_version: str
    summary_schema_version: str
    minimum_publishable_coverage: float


def default_reports_dependencies() -> ReportsDependencies:
    """Resolve package defaults at invocation time, not import time."""

    return ReportsDependencies(
        read_json=read_json,
        atomic_write_json=atomic_write_json,
        utc_now=utc_now,
        api_usage_summary=api_usage_summary,
        read_api_call_records=read_api_call_records,
        empty_metric_summary=empty_metric_summary,
        telemetry_by_metric=telemetry_by_metric,
        initial_image_count=initial_image_count,
        metric_failure_counts=metric_failure_counts,
        response_schema_audit_from_exception=response_schema_audit_from_exception,
        metric_score_is_publishable=_metric_score_is_publishable,
        case_schema_version=CASE_SCHEMA_VERSION,
        summary_schema_version=SUMMARY_SCHEMA_VERSION,
        minimum_publishable_coverage=MIN_PUBLISHABLE_SCORE_COVERAGE,
    )


def collect_l1_engineering_failures(
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    """Collect fail-closed L1 transport/schema failures without hiding L3."""

    failures: list[dict[str, Any]] = []

    def visit(
        value: Any,
        *,
        path: tuple[str, ...],
        metric: str | None,
    ) -> None:
        if isinstance(value, dict):
            current_metric = (
                str(value["metric"])
                if value.get("metric") is not None
                else metric
            )
            if (
                value.get("route") == "vlm_adjudication_failed"
                or value.get("status") == "vlm_adjudication_failed"
            ):
                item = {
                    "path": ".".join(path),
                    "metric": current_metric,
                    "route": value.get("route"),
                    "status": value.get("status"),
                    "error": value.get("adjudication_error")
                    or (
                        value.get("evidence", {}).get("error")
                        if isinstance(value.get("evidence"), dict)
                        else None
                    ),
                }
                evidence = value.get("evidence")
                evidence = evidence if isinstance(evidence, dict) else {}
                audit = (
                    value.get("adjudication_failure_audit")
                    or evidence.get("adjudication_failure_audit")
                )
                if isinstance(audit, dict):
                    item["response_schema_validation"] = deepcopy(audit)
                failures.append(item)
            for key, child in value.items():
                visit(
                    child,
                    path=(*path, str(key)),
                    metric=current_metric,
                )
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(
                    child,
                    path=(*path, str(index)),
                    metric=metric,
                )

    visit(report, path=("l1",), metric=None)
    return failures


def binary_schema_validation_summary(
    report: dict[str, Any],
) -> dict[str, int]:
    """Count binary response attempts separately from logical Judge calls."""

    audits: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key in (
                "response_schema_validation",
                "adjudication_failure_audit",
            ):
                audit = value.get(key)
                if isinstance(audit, dict) and audit.get("policy") == (
                    "single_schema_repair_retry_v1"
                ):
                    audits.append(audit)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(report)
    return {
        "logical_binary_judge_calls": len(audits),
        "response_attempts": sum(
            int(item.get("attempt_count") or 0)
            for item in audits
        ),
        "schema_repair_retries": sum(
            int(item.get("repair_retry_count") or 0)
            for item in audits
        ),
        "schema_repair_recoveries": sum(
            item.get("recovered") is True for item in audits
        ),
        "schema_repair_failures": sum(
            item.get("repair_retry_count") == 1
            and item.get("recovered") is False
            for item in audits
        ),
    }


def record_case_cancellation(
    *,
    case: dict[str, Any],
    output_root: Path,
    dependencies: ReportsDependencies | None = None,
) -> dict[str, Any]:
    """Persist an explicit terminal record for a fail-fast cancellation."""

    deps = dependencies or default_reports_dependencies()
    case_id = str(case["case_id"])
    case_out = output_root / "cases" / case_id
    case_out.mkdir(parents=True, exist_ok=True)
    cancelled_at = deps.utc_now()
    record = {
        "schema_version": deps.case_schema_version,
        "case_id": case_id,
        "status": "cancelled",
        "reason": "cancelled_after_prior_case_failure",
        "cancelled_at": cancelled_at,
        "final_decision_status": "not_run",
        "api_usage": deps.api_usage_summary([]),
    }
    deps.atomic_write_json(case_out / "cancellation.json", record)
    manifest_path = case_out / "case_run_manifest.json"
    if manifest_path.is_file():
        manifest = deps.read_json(manifest_path)
        manifest.update(
            status="cancelled",
            completed_at=cancelled_at,
            final_decision_status="not_run",
            reason=record["reason"],
            api_usage=deepcopy(record["api_usage"]),
        )
    else:
        manifest = deepcopy(record)
        manifest["completed_at"] = cancelled_at
    deps.atomic_write_json(manifest_path, manifest)
    return record


def record_case_failure(
    *,
    case: dict[str, Any],
    output_root: Path,
    error: Exception,
    dependencies: ReportsDependencies | None = None,
) -> dict[str, Any]:
    """Persist API usage and an explicit terminal record for a failed case."""

    deps = dependencies or default_reports_dependencies()
    case_id = str(case["case_id"])
    case_out = output_root / "cases" / case_id
    case_out.mkdir(parents=True, exist_ok=True)
    api_calls_path = case_out / "api_calls.jsonl"
    api_usage_path = case_out / "api_usage.json"
    api_usage = deps.api_usage_summary(
        deps.read_api_call_records(api_calls_path)
    )
    deps.atomic_write_json(api_usage_path, api_usage)
    failure = {
        "schema_version": deps.case_schema_version,
        "case_id": case_id,
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "failed_at": deps.utc_now(),
        "api_calls_path": str(api_calls_path.resolve()),
        "api_usage_path": str(api_usage_path.resolve()),
        "api_usage": api_usage,
    }
    schema_audit = deps.response_schema_audit_from_exception(error)
    if schema_audit is not None:
        failure["response_schema_validation"] = schema_audit
    deps.atomic_write_json(case_out / "failure.json", failure)
    manifest_path = case_out / "case_run_manifest.json"
    if manifest_path.is_file():
        manifest = deps.read_json(manifest_path)
        manifest.update(
            status="failed",
            completed_at=failure["failed_at"],
            final_decision_status="unresolved",
            error_type=failure["error_type"],
            error=failure["error"],
            api_usage=api_usage,
            api_calls_path=str(api_calls_path.resolve()),
            api_usage_path=str(api_usage_path.resolve()),
        )
        if schema_audit is not None:
            manifest["binary_response_schema_validation"] = schema_audit
    else:
        manifest = deepcopy(failure)
        manifest.update(
            completed_at=failure["failed_at"],
            final_decision_status="unresolved",
        )
        if schema_audit is not None:
            manifest["binary_response_schema_validation"] = schema_audit
    deps.atomic_write_json(manifest_path, manifest)
    return failure


def l3_resolution_audit(
    scene_quality_report: dict[str, Any],
    *,
    metrics: tuple[str, ...],
    dependencies: ReportsDependencies | None = None,
) -> dict[str, Any]:
    """Resolve runner status without relabelling partial coverage as infra."""

    deps = dependencies or default_reports_dependencies()
    report_metrics = scene_quality_report.get("metrics")
    report_metrics = (
        report_metrics if isinstance(report_metrics, dict) else {}
    )
    unresolved: list[str] = []
    infrastructure_failures: list[str] = []
    partial_coverage: list[str] = []
    below_coverage_threshold: list[str] = []
    reasons: dict[str, list[str]] = {}
    coverage_warnings: dict[str, list[str]] = {}
    for metric in metrics:
        item = report_metrics.get(metric)
        metric_reasons: list[str] = []
        metric_coverage_warnings: list[str] = []
        if not isinstance(item, dict):
            metric_reasons.append("metric_report_missing")
            infrastructure_failures.append(metric)
        else:
            item_status = str(item.get("status") or "")
            terminal_state = str(item.get("terminal_state") or "")
            item_reason = str(item.get("reason") or "")
            coverage_threshold_failure = bool(
                item_status == "failed_coverage_threshold"
                or item_reason
                in {
                    "below_minimum_score_coverage",
                    "failed_coverage_threshold",
                }
            )
            explicit_infrastructure_failure = bool(
                item_status in {"error", "infrastructure_failure"}
                or (item_status == "failed" and not coverage_threshold_failure)
                or terminal_state == "infrastructure_failure"
                or bool(item.get("infrastructure_failures"))
            )
            if explicit_infrastructure_failure:
                infrastructure_failures.append(metric)
            score_publishable = deps.metric_score_is_publishable(item)
            if item_status not in {"evaluated", "partial"}:
                metric_reasons.append(
                    f"metric_status:{item_status or 'missing'}"
                )
            for field in (
                "coverage",
                "functional_check_coverage",
                "placement_check_coverage",
            ):
                coverage = item.get(field)
                if (
                    isinstance(coverage, dict)
                    and coverage.get("complete") is False
                ):
                    marker = f"{field}:incomplete"
                    if score_publishable:
                        metric_coverage_warnings.append(marker)
                    else:
                        metric_reasons.append(marker)
            if metric_coverage_warnings:
                partial_coverage.append(metric)
                coverage_warnings[metric] = list(
                    dict.fromkeys(metric_coverage_warnings)
                )
            elif (
                not explicit_infrastructure_failure
                and not score_publishable
                and any(
                    isinstance(item.get(field), dict)
                    and item[field].get("complete") is False
                    for field in (
                        "coverage",
                        "functional_check_coverage",
                        "placement_check_coverage",
                    )
                )
            ):
                below_coverage_threshold.append(metric)
                metric_reasons.append("coverage:below_publishable_threshold")
            elif (
                not explicit_infrastructure_failure
                and item_status in {"evaluated", "partial"}
                and not score_publishable
            ):
                metric_reasons.append("metric_score:unavailable")
            for field in (
                "functional_check_phase",
                "placement_check_phase",
                "group_phase",
                "cross_group_relation_phase",
            ):
                phase = item.get(field)
                if (
                    isinstance(phase, dict)
                    and str(phase.get("status") or "")
                    in {
                        "unresolved",
                        "terminal_contract_failure",
                        "infrastructure_failure",
                    }
                ):
                    phase_status = str(phase.get("status") or "")
                    metric_reasons.append(f"{field}:{phase_status}")
                    if phase_status in {
                        "terminal_contract_failure",
                        "infrastructure_failure",
                    }:
                        infrastructure_failures.append(metric)
        if metric_reasons:
            if metric not in infrastructure_failures:
                unresolved.append(metric)
            reasons[metric] = list(dict.fromkeys(metric_reasons))
    infrastructure_failures = list(dict.fromkeys(infrastructure_failures))
    unresolved = list(dict.fromkeys(unresolved))
    partial_coverage = list(dict.fromkeys(partial_coverage))
    below_coverage_threshold = list(
        dict.fromkeys(below_coverage_threshold)
    )
    return {
        "status": (
            "infrastructure_failure"
            if infrastructure_failures
            else "unresolved"
            if unresolved
            else "resolved"
        ),
        "unresolved_metrics": unresolved,
        "infrastructure_failure_metrics": infrastructure_failures,
        "partial_coverage_metrics": partial_coverage,
        "below_coverage_threshold_metrics": below_coverage_threshold,
        "reasons_by_metric": reasons,
        "coverage_warnings_by_metric": coverage_warnings,
        "minimum_publishable_coverage": deps.minimum_publishable_coverage,
        "policy": "terminal_binary_or_scoreable_partial_coverage_v3",
    }


def _metric_score_is_publishable(item: dict[str, Any]) -> bool:
    """Treat the evaluator's published score and threshold audit as authority."""

    score = item.get("score")
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(float(score))
    ):
        return False
    coverage = item.get("coverage")
    coverage = coverage if isinstance(coverage, dict) else {}
    scoring = item.get("scoring")
    scoring = scoring if isinstance(scoring, dict) else {}
    projections = [
        coverage,
        coverage.get("score_projection"),
        scoring.get("coverage_projection"),
    ]
    explicit_thresholds = [
        projection.get("coverage_threshold_passed")
        for projection in projections
        if isinstance(projection, dict)
        and isinstance(projection.get("coverage_threshold_passed"), bool)
    ]
    if explicit_thresholds:
        return all(explicit_thresholds)
    return True


def build_summary(
    *,
    case_records: list[dict[str, Any]],
    metrics: tuple[str, ...],
    elapsed_seconds: float,
    dependencies: ReportsDependencies | None = None,
) -> dict[str, Any]:
    """Aggregate case comparison, report, telemetry, and API usage outputs."""

    deps = dependencies or default_reports_dependencies()
    metric_summaries = {
        metric: deps.empty_metric_summary(total=len(case_records))
        for metric in metrics
    }
    successful = 0
    cancelled = 0
    grouping_failures = 0
    final_unresolved = 0
    final_infrastructure_failure = 0
    l3_unresolved_cases = 0
    l3_infrastructure_failure_cases = 0
    l1_engineering_failure_cases = 0
    binary_logical_judge_calls = 0
    binary_response_attempts = 0
    binary_schema_repair_retries = 0
    binary_schema_repair_recoveries = 0
    binary_schema_repair_failures = 0
    total_judge_calls = 0
    total_selector_calls = 0
    all_api_call_records: list[dict[str, Any]] = []
    latencies: list[float] = []
    for record in case_records:
        api_calls_path = record.get("api_calls_path")
        if isinstance(api_calls_path, str) and api_calls_path:
            all_api_call_records.extend(
                deps.read_api_call_records(Path(api_calls_path))
            )
        if record.get("status") == "cancelled":
            cancelled += 1
        if record.get("status") not in {"complete", "resumed"}:
            for summary in metric_summaries.values():
                summary["case_failures"] += 1
            continue
        successful += 1
        if record.get("final_decision_status") == "unresolved":
            final_unresolved += 1
        if record.get("final_decision_status") == "infrastructure_failure":
            final_infrastructure_failure += 1
        if record.get("l3_decision_status") == "unresolved":
            l3_unresolved_cases += 1
        if record.get("l3_decision_status") == "infrastructure_failure":
            l3_infrastructure_failure_cases += 1
        if record.get("l1_engineering_failure") is True:
            l1_engineering_failure_cases += 1
        binary_schema = record.get(
            "binary_response_schema_validation"
        )
        binary_schema = (
            binary_schema if isinstance(binary_schema, dict) else {}
        )
        binary_logical_judge_calls += int(
            binary_schema.get("logical_binary_judge_calls") or 0
        )
        binary_response_attempts += int(
            binary_schema.get("response_attempts") or 0
        )
        binary_schema_repair_retries += int(
            binary_schema.get("schema_repair_retries") or 0
        )
        binary_schema_repair_recoveries += int(
            binary_schema.get("schema_repair_recoveries") or 0
        )
        binary_schema_repair_failures += int(
            binary_schema.get("schema_repair_failures") or 0
        )
        latency = record.get("elapsed_seconds")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))
        comparison_path = Path(str(record["scene_comparison_path"]))
        comparison = deps.read_json(comparison_path)
        if record.get("grouping_status") not in {None, "complete"}:
            grouping_failures += 1
        report = deps.read_json(Path(str(record["scene_quality_report_path"])))
        report_metrics = report.get("metrics")
        report_metrics = report_metrics if isinstance(report_metrics, dict) else {}
        control = (
            deps.read_json(Path(str(record["control_manifest_path"])))
            if record.get("control_manifest_path")
            else {}
        )
        telemetry = deps.telemetry_by_metric(control)
        comparisons = comparison.get("metrics")
        comparisons = comparisons if isinstance(comparisons, dict) else {}
        for metric in metrics:
            item = comparisons.get(metric)
            item = item if isinstance(item, dict) else {}
            human = item.get("human")
            human = human if isinstance(human, dict) else {}
            model = item.get("model")
            model = model if isinstance(model, dict) else {}
            summary = metric_summaries[metric]
            if record.get("final_decision_status") != "resolved":
                summary["diagnostic_only_cases"] += 1
            expected = human.get("expected")
            predicted = model.get("prediction")
            if expected in {"valid", "invalid"}:
                summary["human_distribution"][expected] += 1
            if predicted in {"valid", "invalid"}:
                summary["predicted_distribution"][predicted] += 1
                summary["evaluated"] += 1
            elif predicted == "infrastructure_failure":
                summary["infrastructure_failure"] += 1
            else:
                summary["unresolved"] += 1
            if human.get("unclear") is True:
                summary["excluded_unclear"] += 1
            if item.get("included_in_accuracy") is True:
                if item.get("matches") is True:
                    summary["correct"] += 1
                else:
                    summary["incorrect"] += 1
            anomaly_level = item.get("anomaly_level")
            anomaly_level = (
                anomaly_level
                if isinstance(anomaly_level, dict)
                else {}
            )
            if anomaly_level.get("included_in_accuracy") is True:
                summary["anomaly_object_cases"] += 1
                if anomaly_level.get("exact_match") is True:
                    summary["anomaly_object_exact_correct"] += 1
                else:
                    summary["anomaly_object_exact_incorrect"] += 1
                summary["anomaly_object_true_positive"] += len(
                    anomaly_level.get("true_positive_object_ids") or []
                )
                summary["anomaly_object_false_negative"] += len(
                    anomaly_level.get("false_negative_object_ids") or []
                )
                summary["anomaly_object_false_positive"] += len(
                    anomaly_level.get("false_positive_object_ids") or []
                )
            metric_report = report_metrics.get(metric)
            metric_report = (
                metric_report if isinstance(metric_report, dict) else {}
            )
            failure_counts = deps.metric_failure_counts(metric_report)
            summary["camera_render_failures"] += failure_counts[
                "camera_render_failures"
            ]
            summary["judge_failures"] += failure_counts["judge_failures"]
            metric_telemetry = telemetry.get(metric, {})
            for key in (
                "judge_calls",
                "vlm_selector_calls",
                "preview_image_count",
                "final_image_count",
                "evidence_repair_count",
                "evidence_recovery_count",
            ):
                summary[key] += int(metric_telemetry.get(key) or 0)
            summary["initial_image_count"] += deps.initial_image_count(
                metric_report
            )
    for summary in metric_summaries.values():
        denominator = summary["correct"] + summary["incorrect"]
        summary["accuracy"] = (
            summary["correct"] / denominator if denominator else None
        )
        object_denominator = (
            summary["anomaly_object_exact_correct"]
            + summary["anomaly_object_exact_incorrect"]
        )
        summary["anomaly_object_exact_accuracy"] = (
            summary["anomaly_object_exact_correct"]
            / object_denominator
            if object_denominator
            else None
        )
        object_precision_denominator = (
            summary["anomaly_object_true_positive"]
            + summary["anomaly_object_false_positive"]
        )
        object_recall_denominator = (
            summary["anomaly_object_true_positive"]
            + summary["anomaly_object_false_negative"]
        )
        summary["anomaly_object_precision"] = (
            summary["anomaly_object_true_positive"] / object_precision_denominator
            if object_precision_denominator
            else None
        )
        summary["anomaly_object_recall"] = (
            summary["anomaly_object_true_positive"] / object_recall_denominator
            if object_recall_denominator
            else None
        )
        precision = summary["anomaly_object_precision"]
        recall = summary["anomaly_object_recall"]
        summary["anomaly_object_f1"] = (
            (
                2.0 * precision * recall / (precision + recall)
                if precision + recall > 0.0
                else 0.0
            )
            if isinstance(precision, float)
            and isinstance(recall, float)
            else None
        )
        total_judge_calls += summary["judge_calls"]
        total_selector_calls += summary["vlm_selector_calls"]
        summary["grouping_failures"] = grouping_failures
    api_usage = deps.api_usage_summary(all_api_call_records)
    operation_calls = (
        api_usage.get("operation_calls")
        if isinstance(api_usage.get("operation_calls"), dict)
        else {}
    )
    return {
        "schema_version": deps.summary_schema_version,
        "status": (
            "complete"
            if successful == len(case_records)
            else "partial"
            if successful
            else "failed"
        ),
        "source_prompt_used": False,
        "comparison_scopes": [
            "scene_level_metric_verdict",
            "anomaly_object_attribution",
        ],
        "elapsed_seconds": elapsed_seconds,
        "average_case_latency_seconds": (
            sum(latencies) / len(latencies) if latencies else None
        ),
        "totals": {
            "cases": len(case_records),
            "successful": successful,
            "failed": len(case_records) - successful - cancelled,
            "cancelled": cancelled,
            "grouping_failures": grouping_failures,
            "final_unresolved": final_unresolved,
            "final_infrastructure_failure": (
                final_infrastructure_failure
            ),
            "l3_unresolved_cases": l3_unresolved_cases,
            "l3_infrastructure_failure_cases": (
                l3_infrastructure_failure_cases
            ),
            "l1_engineering_failure_cases": (
                l1_engineering_failure_cases
            ),
            "binary_logical_judge_calls": binary_logical_judge_calls,
            "binary_response_attempts": (
                binary_response_attempts
            ),
            "binary_schema_repair_retries": (
                binary_schema_repair_retries
            ),
            "binary_schema_repair_recoveries": (
                binary_schema_repair_recoveries
            ),
            "binary_schema_repair_failures": (
                binary_schema_repair_failures
            ),
            "judge_calls": total_judge_calls,
            "vlm_camera_selector_calls": total_selector_calls,
            "functional_discovery_calls": int(
                operation_calls.get("functional_discovery") or 0
            ),
            "functional_affordance_calls": int(
                operation_calls.get("functional_affordance") or 0
            ),
            "functional_relation_calls": int(
                operation_calls.get("functional_relation") or 0
            ),
            "placement_discovery_calls": int(
                operation_calls.get("placement_discovery") or 0
            ),
            "usable_surface_decoder_calls": int(
                operation_calls.get("usable_surface_decoder") or 0
            ),
            "vlm_camera_selector_api_calls": int(
                operation_calls.get("camera_selector") or 0
            ),
            "judge_api_calls": int(
                operation_calls.get("judge") or 0
            ),
            "api_calls_number": api_usage["api_calls_number"],
            "tokens_usage": deepcopy(api_usage["tokens_usage"]),
        },
        "api_usage": api_usage,
        "metrics": metric_summaries,
    }


# Patchable package aliases.  ``default_reports_dependencies`` resolves these
# names at each call, so a package-level monkeypatch is not frozen at import.
read_json = _io_read_json
atomic_write_json = _io_atomic_write_json
utc_now = _io_utc_now
api_usage_summary = _telemetry_api_usage_summary
empty_metric_summary = _telemetry_empty_metric_summary
initial_image_count = _telemetry_initial_image_count
metric_failure_counts = _telemetry_metric_failure_counts
read_api_call_records = _telemetry_read_api_call_records
telemetry_by_metric = _telemetry_by_metric
response_schema_audit_from_exception = _response_schema_audit_from_exception


__all__ = [
    "CASE_SCHEMA_VERSION",
    "MIN_PUBLISHABLE_SCORE_COVERAGE",
    "ReportsDependencies",
    "SUMMARY_SCHEMA_VERSION",
    "binary_schema_validation_summary",
    "build_summary",
    "collect_l1_engineering_failures",
    "default_reports_dependencies",
    "l3_resolution_audit",
    "record_case_cancellation",
    "record_case_failure",
]
