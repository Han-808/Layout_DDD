"""Acceptance is distinct from observation coverage in the opt-in policy."""

from __future__ import annotations

from copy import deepcopy
from functools import wraps
from typing import Any, Callable
from benchmark.visual_judge.evidence_gap_v2 import enabled as fallback_v2_enabled

from benchmark.visual_judge.evidence_resolution import (
    ADAPTIVE_POLICY, adaptive_enabled, bind_resolution, finite_score,
    resolution_of,
)


def terminalize_adaptive_scope(record: dict[str, Any], *, phase: str) -> dict[str, Any]:
    if fallback_v2_enabled():
        from .consistency_acceptance_v2 import terminalize_scope
        return terminalize_scope(record, phase=phase)
    resolution = resolution_of(record)
    record["evidence_resolution_policy"] = ADAPTIVE_POLICY
    if resolution is None and record.get("status") == "evaluated" and finite_score(record.get("score")):
        aggregate = summarize_metric_resolution(record)
        record["resolution_coverage"] = aggregate
        if aggregate["complete"]:
            record["terminal_state"] = (
                "evaluated" if aggregate["visual_evidence_fraction"] == 1.0 else "evaluated_degraded"
            )
            return record
    if (
        resolution is not None and resolution.get("accepted") is True
        and record.get("status") == "evaluated" and finite_score(record.get("score"))
    ):
        bind_resolution(record, record)
        visual_complete = bool(resolution.get("visual_observation_complete"))
        record["terminal_state"] = "evaluated" if visual_complete else "evaluated_degraded"
        record["vlm_invoked"] = bool(resolution.get("model_invoked"))
        record["evidence_coverage"] = {
            **(record.get("evidence_coverage") or {}),
            "grounded": visual_complete,
            "empirically_grounded": visual_complete,
            "grounding_fraction": float(visual_complete),
            "policy_resolved": True,
            "coverage_kind": resolution.get("evidence_tier"),
        }
        record.pop("infrastructure_failure", None)
        return record
    judgement = record.get("judgement") or {}
    failure = judgement.get("failure") if isinstance(judgement, dict) else None
    failure = deepcopy(failure) if isinstance(failure, dict) else {
        "failure_category": "unresolved_required_judgement",
        "phase": phase,
        "error_type": judgement.get("error_type") if isinstance(judgement, dict) else None,
        "error": judgement.get("error") if isinstance(judgement, dict) else None,
    }
    record.update(
        status="failed", score=None, terminal_state="infrastructure_failure",
        reason=record.get("reason") or failure["failure_category"],
        infrastructure_failure={"phase": phase, "failure_kind": failure["failure_category"], **failure},
    )
    return record


def _resolution_units(value: Any, *, path: str = "metric") -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    # Persisted aggregate copies of the same scopes are de-duplicated later.
    own = resolution_of(value)
    if own is not None:
        judgement = value.get("judgement") or value
        rows = [
            row for key in ("functional_check_results", "placement_check_results")
            for row in judgement.get(key) or [] if isinstance(row, dict)
        ]
        ids = [str(row.get("check_id") or "") for row in rows]
        if not rows:
            return [{**deepcopy(own), "unit_id": path, "scope_path": path}]
        if any(not check_id for check_id in ids) or len(ids) != len(set(ids)):
            return [{"unit_id": path, "accepted": False, "failure_category": "check_id_integrity_failure"}]
        return [
            {
                **deepcopy(own), "unit_id": check_id, "scope_path": path,
                "accepted": bool(own.get("accepted")) and row.get("conclusion") in {
                    "valid", "invalid", "excluded_function_owned",
                },
                "visual_observation_complete": bool(own.get("images_used"))
                and row.get("observation_status") == "observed",
            }
            for check_id, row in zip(ids, rows)
        ]
    units: list[dict[str, Any]] = []
    # Judge-episode topology only; never treat acquisition/router verdicts as final.
    for key in (
        "judgement", "scene_global_judgement", "cross_group_relation_judgements",
        "group_judgements", "target_scope_judgements", "residual_global_placement_judgement",
        "group_results", "target_scope_results",
        "placement_global_handoff_reviews",
    ):
        nested = value.get(key)
        if isinstance(nested, dict):
            units.extend(_resolution_units(nested, path=path + "/" + key))
        elif isinstance(nested, list):
            for index, item in enumerate(nested):
                units.extend(_resolution_units(item, path=path + "/" + key + "/" + str(index)))
    if not units and (
        value.get("status") in {"failed", "unresolved", "infrastructure_failure", "not_evaluable"}
        or value.get("defaulted") is True
    ):
        units.append({"unit_id": path, "unit_key": path, "accepted": False,
                      "failure_category": "required_judgement_unresolved"})
    return units


def summarize_metric_resolution(report: dict[str, Any]) -> dict[str, Any]:
    units = _resolution_units(report)
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for unit in units:
        # A check is a stable obligation; copied reports must not inflate coverage.
        key = (
            unit.get("metric"), unit.get("unit_key"),
            None if str(unit.get("unit_id", "")).startswith("metric") else unit.get("unit_id"),
        )
        unique.setdefault(key, unit)
    units = list(unique.values())
    # Include unresolved checks in the denominator even if they never reached a Judge.
    known_ids = {str(unit.get("unit_id")) for unit in units}
    for ledger_key in ("functional_check_ledger", "placement_check_ledger"):
        ledger = report.get(ledger_key) or {}
        checks = ledger.get("checks") or [] if isinstance(ledger, dict) else []
        if isinstance(checks, dict):
            checks = list(checks.values())
        for check in checks:
            if not isinstance(check, dict):
                continue
            check_id = str(check.get("check_id") or "")
            if check_id and check_id not in known_ids:
                units.append({"unit_id": check_id, "accepted": False, "failure_category": "required_check_unresolved"})
                known_ids.add(check_id)
    eligible = len(units)
    accepted = sum(unit.get("accepted") is True for unit in units)
    model = sum(unit.get("model_judgement_completed") is True for unit in units)
    visual = sum(unit.get("visual_observation_complete") is True for unit in units)
    structured = sum(unit.get("evidence_tier") == "structured_fallback"
                     and unit.get("decision_source") == "model" for unit in units)
    style = sum(unit.get("decision_source") == "style_policy_no_deduction" for unit in units)
    complete = bool(eligible and accepted == eligible and report.get("status") == "evaluated")
    return {
        "schema_version": "adaptive_judgement_coverage_v1", "policy": ADAPTIVE_POLICY,
        "eligible_count": eligible, "accepted_count": accepted, "complete": complete,
        "acceptance_fraction": accepted / eligible if eligible else 0.0,
        "model_judgement_count": model, "model_judgement_fraction": model / eligible if eligible else 0.0,
        "visual_evidence_count": visual, "visual_evidence_fraction": visual / eligible if eligible else 0.0,
        "structured_fallback_count": structured, "structured_fallback_fraction": structured / eligible if eligible else 0.0,
        "style_policy_count": style, "style_policy_fraction": style / eligible if eligible else 0.0,
        "units": units,
    }


def adaptive_metric_result(function: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if fallback_v2_enabled():
            import inspect
            from .consistency_acceptance_v2 import original_scope_plan, finish_metric
            from benchmark.visual_judge.evidence_resolution import failure_record
            arguments = inspect.signature(function).bind(*args, **kwargs).arguments
            planned = original_scope_plan(arguments)
            try:
                report = function(*args, **kwargs)
            except Exception as exc:
                # A failed metric cannot erase independent completed metrics.
                # Hard errors remain failed, with their actual category.
                failure = failure_record(exc, phase="metric")
                config = arguments.get("metric_config") or {}
                report = {
                    "metric": arguments.get("metric_name"), "enabled": config.get("enabled", True),
                    "weight": float(config.get("weight", 1.0)),
                    "affects_score": bool(arguments.get("top_enabled", True) and config.get("enabled", True)
                                          and config.get("weight", 1.0) > 0),
                    "status": "not_evaluable" if failure["recoverable_acquisition"] else "failed",
                    "score": None, "reason": failure["failure_category"],
                    "judgement": {"failure": failure},
                    # An exception may precede base-report construction or
                    # follow real acquisition. Unknown is not "never invoked".
                    "evidence_request": {"provider_invoked": None, "renderer_invoked": None,
                                         "vlm_invoked": None},
                    "renderer_invoked": None, "preview_renderer_invoked": None,
                    "vlm_invoked": None, "invocation_audit_complete": False,
                    "exception_diagnostics": _exception_locations(exc),
                }
            if report.get("status") == "not_applicable":
                return report
            return finish_metric(report, planned)
        report = function(*args, **kwargs)
        if not adaptive_enabled() or report.get("status") == "not_applicable":
            return report
        report["evidence_resolution_policy"] = ADAPTIVE_POLICY
        coverage = summarize_metric_resolution(report)
        report["resolution_coverage"] = coverage
        if not coverage["complete"]:
            report.update(status="failed", score=None, terminal_state="infrastructure_failure",
                          reason=report.get("reason") or "required_judgements_unresolved")
        if coverage["style_policy_count"] == coverage["eligible_count"] and coverage["eligible_count"]:
            judgement = report.get("judgement")
            if isinstance(judgement, dict):
                judgement["confidence"] = None
        return report
    return wrapped


def _exception_locations(error: Exception) -> dict[str, Any]:
    """Retain locations, not messages, arguments, locals or model exchanges."""
    frames = []
    current = error.__traceback__
    while current is not None:
        code = current.tb_frame.f_code
        frames.append({"file": code.co_filename, "line": current.tb_lineno,
                       "function": code.co_name})
        current = current.tb_next
    return {"error_type": type(error).__name__, "frames": frames}


def adaptive_scene_result(function: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        report = function(*args, **kwargs)
        if not adaptive_enabled():
            return report
        if fallback_v2_enabled():
            from benchmark.visual_judge.evidence_gap_v2 import coverage_from_plan
            from benchmark.visual_judge.evidence_resolution import FALLBACK_POLICY
            selected = {name: item for name, item in report.get("metrics", {}).items()
                        if isinstance(item, dict) and item.get("affects_score")}
            records = [{
                "unit_id": name, "status": item.get("status"), "score": item.get("score"),
                "accepted": bool(item.get("resolution_coverage", {}).get("complete")),
                "execution_complete": True,
                "visual_observation_complete": item.get("resolution_coverage", {}).get("visual_evidence_complete") is True,
                "failure_category": None if item.get("resolution_coverage", {}).get("complete") else
                "evidence_gap" if item.get("status") == "not_evaluable" else "metric_failure",
            } for name, item in selected.items()]
            coverage = coverage_from_plan(selected, records)
            coverage.update(complete=coverage["evaluation_complete"], eligible_count=coverage["planned_count"],
                            accepted_count=coverage["evaluation_count"], acceptance_fraction=coverage["evaluation_fraction"])
            report.update(evidence_resolution_policy=FALLBACK_POLICY, resolution_coverage=coverage,
                          execution_complete=True)
            report.setdefault("coverage", {}).update(
                complete=coverage["evaluation_complete"], acceptance_complete=coverage["evaluation_complete"],
                score_grounding_complete=coverage["evaluation_complete"] and coverage["visual_evidence_complete"])
            return report
        from benchmark.evaluator.adaptive_audit import coverage_from_units
        metrics = [item for item in report.get("metrics", {}).values()
                   if isinstance(item, dict) and item.get("affects_score")]
        units = [unit for item in metrics for unit in item.get("resolution_coverage", {}).get("units", [])]
        complete = bool(metrics and all(item.get("resolution_coverage", {}).get("complete") for item in metrics)
                        and finite_score(report.get("score")))
        report["evidence_resolution_policy"] = ADAPTIVE_POLICY
        report["resolution_coverage"] = coverage_from_units(units, complete=complete)
        coverage = report.setdefault("coverage", {})
        coverage["score_grounding_complete"] = bool(
            complete and report["resolution_coverage"]["visual_evidence_fraction"] == 1.0
        )
        coverage["acceptance_complete"] = complete
        coverage["complete"] = complete
        return report
    return wrapped
