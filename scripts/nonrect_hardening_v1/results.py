"""Execution result envelopes: accounting is distinct from scientific scoring.

Never create a synthetic official complete-room report or an aggregate score.
Skipped, infrastructure-failed and unstarted work stays in the denominator.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .safety import (IntegrityError, atomic_json, best_effort_record, diagnostic_projection,
                     digest, exception_record, identity, read_json, stamp)
from . import reasons, assurance

RESULT_VERSION = "nonrect_accounted_execution_result_v4"
METRICS = ("collision", "oob", "support", "scale_consistency", "style_consistency",
           "object_pairing_consistency", "functional_consistency", "semantic_placement_consistency")
LOCAL_HARD_ERRORS = (TypeError, KeyError, IndexError, AttributeError, AssertionError,
                     ZeroDivisionError, RecursionError)
HARD_TYPES = {t.__name__ for t in LOCAL_HARD_ERRORS} | {"ValueError", "ResponseSchemaRepairError",
    "RoomEvaluatorReportError", "RoomLayoutValidationError", "NonRectangularContractError",
    "NonRectangularMaterializationContractError"}


def failure_kind(failure: dict | None) -> str:
    failure = failure or {}
    category, kind = str(failure.get("category", "")), str(failure.get("error_type", ""))
    if any(word in category for word in ("identity_drift", "configuration", "interrupted", "channel_unavailable")):
        return "infra_failure"
    if kind in {"EndpointConfigurationError", "EndpointMalformedResponseError", "MissingAPIKeyError",
                "RetryBudgetExhausted", "ChannelUnavailable", "IntegrityError", "BlenderSourceSceneModifiedError"}:
        return "infra_failure"
    if any(word in category for word in ("transport", "http", "service_or_renderer", "execution_budget", "api_")):
        return "infra_failure"
    if category == "camera_or_renderer_failure":
        return "infra_failure"
    if kind in HARD_TYPES or category == "room_hard_failure":
        return "hard_failure"
    if category in {"semantic_or_contract", "evidence_exhaustion_unclosed", "metric_report_missing",
                    "judge_response_contract_failure", "camera_constraint_contract_invalid",
                    "evidence_budget_exhausted", "metric_not_evaluated"}:
        return "hard_failure"
    return "unresolved"


def hard_origin(raw: Any) -> bool:
    """Only positive local hard-failure evidence, with infrastructure vetoes."""
    kinds, stops = set(), set()
    ambiguous = False
    wrappers = {"BlenderRenderError", "EvidenceRenderFailure", "NonRectangularRoomMetricIncomplete"}
    def walk(value, chained=False):
        nonlocal ambiguous
        if isinstance(value, dict):
            if value.get("error_type"):
                kind = str(value["error_type"])
                if kind in wrappers:
                    nested = any(isinstance(value.get(k), (dict, list)) and value[k]
                                 for k in ("cause", "exception_chain", "root_cause", "causes"))
                    if not chained and not nested:
                        ambiguous = True
                else:
                    kinds.add(kind)
            stops.add(str(value.get("stop_reason", "")))
            for k, v in value.items():
                if k == "exception_chain" and isinstance(v, list):
                    for index, item in enumerate(v):
                        walk(item, chained=any(isinstance(later, dict) and later.get("error_type")
                                               for later in v[index + 1:]))
                    continue
                if k not in {"raw_response", "raw_request", "request_metadata"}:
                    walk(v)
        elif isinstance(value, (list, tuple)):
            for v in value: walk(v)
    walk(raw)
    if ambiguous or kinds - HARD_TYPES or any("integrity" in s for s in stops):
        return False
    return bool(kinds & HARD_TYPES or "camera_constraint_contract_invalid" in stops)


def capture_metrics(ctx, unit, evaluator, module, classify, enrich) -> None:
    """Normalize already observed data with the ORIGINAL pure normalization rules.

    Does not execute a missing metric, sample again, or combine different attempts.
    Snapshot failures must not mask the primary evaluation failure.
    """
    if ctx.attempt is None:
        return
    metrics = {}
    for name in METRICS:
        raw = ctx.failures.get(name)
        if raw is None:
            metrics[name] = {"status": "not_evaluated", "score": None}
            continue
        try:
            if name in METRICS[:3]:
                field = {"collision": "collision_count", "oob": "invalid_object_count",
                         "support": "unsupported_object_count"}[name]
                normalized = module._normalize_l1(name, raw, object_count=unit.generated_object_count,
                                                  invalid_count=int(raw.get(field) or 0))
            else:
                normalized = module._normalize_l3(name, raw, object_count=unit.generated_object_count,
                    evidence_continuity_context=evaluator.evidence_continuity_context)
            metrics[name] = {k: normalized[k] for k in
                             ("metric", "status", "score", "evaluated_object_count", "invalid_count")
                             if k in normalized}
        except Exception as exc:
            failure = classify(exc, stage="evaluation").public_dict()
            if hard_origin(enrich(raw, ctx)):
                failure = {**failure, "category": "room_hard_failure"}
            metrics[name] = {"status": failure_kind(failure), "score": None,
                             "failure": diagnostic_projection(failure),
                             "failure_details": reasons.describe(raw, ctx, stage="evaluation", metric=name)}
    payload = {"schema_version": RESULT_VERSION, "room_id": unit.room_id,
               "attempt_root": str(ctx.attempt), "execution_identity": ctx.run_identity,
               "materialization_identity_sha256": ctx.materialization_identity,
               "normalization": "original_frozen_rules_no_new_evaluation",
               "metrics": metrics}
    payload = diagnostic_projection(payload)
    best_effort_record(ctx.attempt / "execution_metric_results.json",
                       {**payload, "payload_sha256": identity(payload)})


def _metric_summary(metric):
    score = metric.get("score")
    if (metric.get("status") != "complete" or isinstance(score, bool)
            or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1):
        raise IntegrityError("invalid completed metric in execution result")
    return {k: metric[k] for k in ("metric", "status", "score", "evaluated_object_count", "invalid_count")
            if k in metric}


def room_result(root: Path, *, persist: bool = True, interrupted: bool = False) -> dict:
    summary = read_json(root / "summary.json")
    for key, expected in (("room_id", root.name), ("model", root.parents[3].name),
                          ("scene_id", root.parents[1].name)):
        if key in summary and summary[key] != expected:
            raise IntegrityError("room summary scope mismatch")
    row = {"room_id": root.name, "model": summary.get("model", root.parents[3].name),
           "scene_id": summary.get("scene_id", root.parents[1].name),
           "source_status": summary.get("status"), "score": None,
           "official_room_report": None, "failure": summary.get("latest_failure")}
    metrics = {name: {"status": "not_evaluated", "score": None} for name in METRICS}
    if summary.get("status") == "complete":
        pointer = read_json(root / "room_report_selected.json")
        path = Path(pointer["room_report_path"])
        if path.is_symlink() or digest(path) != pointer["room_report_sha256"]:
            raise IntegrityError("complete room report changed during result accounting")
        report = read_json(path)
        if (report.get("room_id") != root.name or report.get("status") != "complete"
                or report.get("schema_version") != "non_rectangular_complete_room_report_v1"
                or set(report.get("metrics", {})) != set(METRICS)
                or any(m.get("metric") != name for name, m in report["metrics"].items())):
            raise IntegrityError("complete room report scope mismatch")
        metrics = {name: _metric_summary(report["metrics"][name]) for name in METRICS}
        row.update(status="complete", official_room_report={"path": str(path), "sha256": digest(path)})
    else:
        terminal = summary.get("status") in {"failed_nonretryable", "failed_retry_exhausted"}
        kind = failure_kind(summary.get("latest_failure")) if terminal else "unresolved"
        count = summary.get("evaluation_attempt_count", 0)
        if type(count) is not int or count < 0:
            raise IntegrityError("invalid room attempt count")
        attempts = sorted((root / "evaluation_attempts").glob("attempt_*"))
        if any(p.is_symlink() or not p.is_dir() for p in attempts):
            raise IntegrityError("room attempt inventory must contain regular directories")
        if {p.name for p in attempts} != {f"attempt_{n:03d}" for n in range(1, len(attempts) + 1)}:
            raise IntegrityError("room attempt inventory has gaps or invalid names")
        if count != len(attempts):
            if not interrupted or count > len(attempts):
                raise IntegrityError("room summary attempt count is stale")
            # The process can die after opening a new retry but before updating
            # its room summary. Never silently select the previous attempt's
            # partial scores, even if they look more complete.
            row.update(source_status="interrupted_after_room_summary",
                       previous_summary_failure=row["failure"], failure=None,
                       summary_evaluation_attempt_count=count,
                       observed_evaluation_attempt_count=len(attempts))
            count, kind = len(attempts), "unresolved"
        attempt = root / "evaluation_attempts" / f"attempt_{count:03d}"
        snapshot = attempt / "execution_metric_results.json"
        details = attempt / "execution_diagnostics/failure_reasons.json"
        if count and details.is_file():
            detail_value = read_json(details)
            if detail_value.get("attempt_root") == str(attempt):
                row["failure_details"] = detail_value.get("details", [])
        if count and snapshot.is_file():
            value = read_json(snapshot)
            payload = {k: v for k, v in value.items() if k != "payload_sha256"}
            manifest = read_json(attempt / "attempt_manifest.json")
            if (identity(payload) != value.get("payload_sha256") or value.get("room_id") != root.name
                    or value.get("attempt_root") != str(attempt) or value.get("schema_version") != RESULT_VERSION
                    or manifest.get("materialization_identity_sha256") != value.get("materialization_identity_sha256")
                    or set(value.get("metrics", {})) != set(METRICS)):
                raise IntegrityError("partial metric snapshot identity mismatch")
            metrics = value["metrics"]
            for metric in metrics.values():
                if metric.get("status") == "complete":
                    _metric_summary(metric)
                elif metric.get("score") is not None or metric.get("status") not in {
                        "hard_failure", "infra_failure", "unresolved", "not_evaluated"}:
                    raise IntegrityError("invalid partial metric state")
            if any(m["status"] == "infra_failure" for m in metrics.values()):
                kind = "infra_failure"
            elif any(m["status"] == "unresolved" for m in metrics.values()):
                kind = "unresolved"
            row["partial_metric_snapshot"] = {"path": str(snapshot), "sha256": digest(snapshot)}
        row["status"] = "skipped_hard_failure" if kind == "hard_failure" else kind
        if kind == "hard_failure":
            metrics = {name: metric if metric["status"] == "complete" else
                       {**metric, "status": "skipped", "reason": metric.get("status", "not_evaluated")}
                       for name, metric in metrics.items()}
    row["metrics"] = metrics
    row["completed_metric_count"] = sum(m["status"] == "complete" for m in metrics.values())
    row["required_metric_count"] = len(METRICS)
    if persist:
        atomic_json(root / "execution_result.json", row)
    return row


def scene_result(root: Path, *, interrupted: bool = False) -> dict:
    summary_path = root / "summary.json"
    # The frozen coordinator writes preflight before starting the first room,
    # but only writes a scene summary after all its room threads return.
    summary = read_json(summary_path if summary_path.is_file() else root / "preflight.json")
    if not interrupted and not summary_path.is_file():
        raise IntegrityError("scene has no completed coordinator summary")
    order = summary.get("room_order", [])
    if (not isinstance(order, list) or any(not isinstance(r, str) or not r or r in {".", ".."}
            or Path(r).name != r for r in order) or len(order) != len(set(order))):
        raise IntegrityError("invalid or duplicate scene room inventory")
    for key, expected in (("model", root.parents[1].name), ("scene_id", root.name)):
        if key in summary and summary[key] != expected:
            raise IntegrityError("scene summary scope mismatch")
    rows = []
    for room in order:
        path = root / "rooms" / room
        if interrupted and not (path / "summary.json").is_file():
            rows.append({"room_id": room, "model": root.parents[1].name, "scene_id": root.name,
                         "source_status": "not_started_or_interrupted", "status": "unresolved",
                         "score": None, "official_room_report": None, "completed_metric_count": 0,
                         "required_metric_count": len(METRICS),
                         "metrics": {name: {"status": "not_evaluated", "score": None} for name in METRICS}})
        else:
            rows.append(room_result(path, interrupted=interrupted))
    aggregate = root / "evaluation_report.json"
    skip = root / "execution_aggregation_skip.json"
    aggregation = "complete" if aggregate.is_file() else "skipped" if skip.is_file() else "not_available"
    result = {"schema_version": RESULT_VERSION, "model": root.parents[1].name,
              "scene_id": summary.get("scene_id", root.name), "rooms": rows,
              "official_aggregation_status": aggregation, "aggregate_score": None}
    if aggregation == "complete":
        if read_json(aggregate).get("terminal_status") != "complete":
            raise IntegrityError("original scene aggregate is not complete")
        result["official_scene_report"] = {"path": str(aggregate), "sha256": digest(aggregate)}
    if aggregation == "skipped": result["aggregation_failure"] = read_json(skip)
    if not rows:
        result["rejected_scene_failure"] = summary.get("failure")
    atomic_json(root / "execution_result.json", result)
    return result


def lane_result(evaluation: Path, *, run_id: str, expected_rooms: int,
                execution_failure: dict | None = None) -> dict:
    if type(expected_rooms) is not int or expected_rooms < 0:
        raise IntegrityError("planned room count must be a non-negative integer")
    run = read_json(evaluation / "run_manifest.json")
    if identity(run["identity"]) != run["identity_sha256"]:
        raise IntegrityError("execution accounting run identity drift")
    scene_roots = {p.parent for p in evaluation.glob("models/*/scenes/*/summary.json")}
    if execution_failure is not None:
        scene_roots.update(p.parent for p in evaluation.glob("models/*/scenes/*/preflight.json"))
    scenes = [scene_result(path, interrupted=execution_failure is not None) for path in sorted(scene_roots)]
    rows = [r for scene in scenes for r in scene["rooms"]]
    counts = {key: sum(r["status"] == key for r in rows)
              for key in ("complete", "skipped_hard_failure", "infra_failure", "unresolved")}
    missing = max(0, expected_rooms - len(rows))
    if len(rows) > expected_rooms:
        raise IntegrityError("execution results exceed planned room inventory")
    aggregation_skips = sum(s["official_aggregation_status"] == "skipped" for s in scenes)
    unaggregated_complete = any(s["rooms"] and all(r["status"] == "complete" for r in s["rooms"])
                               and s["official_aggregation_status"] == "not_available" for s in scenes)
    unmapped_scenes = sum(not s["rooms"] for s in scenes)
    accounted = (not missing and not counts["unresolved"] and not counts["infra_failure"]
                 and not unaggregated_complete and not unmapped_scenes and execution_failure is None)
    status = ("completed_with_skips" if counts["skipped_hard_failure"] or aggregation_skips else "complete") if accounted else "incomplete"
    result = {"schema_version": RESULT_VERSION, "run_id": run_id, "status": status,
              "campaign_identity_sha256": run["identity_sha256"], "generated_at": stamp(),
              "planned_room_count": expected_rooms, "observed_room_count": len(rows),
              "room_counts": counts, "not_started_room_count": missing,
              "aggregation_skip_count": aggregation_skips, "all_work_accounted": accounted,
              "unmapped_scene_count": unmapped_scenes,
              "score_coverage": {"complete_rooms": counts["complete"], "planned_rooms": expected_rooms,
                  "completed_metrics": sum(r["completed_metric_count"] for r in rows),
                  "planned_metrics": expected_rooms * len(METRICS)},
              "aggregate_score": None, "skipped_items_are_not_scores": True, "scenes": scenes}
    if execution_failure is not None:
        result["execution_failure"] = diagnostic_projection(execution_failure)
    result.update(assurance.derive(result, METRICS))
    assurance.validate(result, METRICS)
    atomic_json(evaluation / "execution_result.json", result)
    return result
