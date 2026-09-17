"""Fixed original scope accounting for the opt-in evidence-gap policy.

Coverage is not a verdict engine. Only already accepted metric results count;
missing planned scopes and subsequently registered typed obligations stay gaps.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from benchmark.visual_judge.evidence_gap_v2 import coverage_from_plan, gap_record
from benchmark.visual_judge.evidence_resolution import FALLBACK_POLICY, finite_score, resolution_of


def original_scope_plan(arguments: dict[str, Any]) -> list[str]:
    config = arguments.get("metric_config") or {}
    metric = arguments.get("metric_name")
    strategy = (config.get("evidence_plan") or {}).get("evidence_strategy")
    if strategy == "global_screen_then_local":
        # This retained Style contract is conditional routing, not a mandatory
        # global-plus-every-group scoring pass.
        return ["metric"]
    scope = (config.get("evidence_policy") or {}).get("camera_scope")
    global_first = strategy in {"global_discovery_then_group_local", "global_screen_then_local"}
    groups = arguments.get("groups")
    object_ids = set(arguments.get("object_ids") or [])
    minimum = 1 if metric == "scale_consistency" else 2
    planned = ["scene_global"] if global_first else []
    if groups is not None and (global_first or scope in {"group_local", "pair_local"}):
        for group in groups:
            members = set(group.get("object_ids") or []) & object_ids
            if len(members) >= minimum:
                planned.append("group:" + str(group["group_id"]))
    return planned or ["metric"]


def failure_of(record: Any) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    failures = []
    for key in ("failure", "infrastructure_failure", "coverage_gap"):
        value = record.get(key)
        if isinstance(value, dict) and value.get("failure_category"):
            failures.append(value)
    for key in ("judgement", "camera_control_audit", "audit"):
        nested = failure_of(record.get(key))
        if nested:
            failures.append(nested)
    for key in ("group_results", "group_judgements", "target_scope_judgements",
                "cross_group_relation_judgements", "placement_global_handoff_reviews", "check_episodes"):
        for value in record.get(key) or []:
            nested = failure_of(value)
            if nested:
                failures.append(nested)
    hard = [item for item in failures if item.get("failure_category") not in {"evidence_gap", "evidence_unavailable"}]
    return deepcopy((hard or failures)[0]) if failures else None


def terminalize_scope(record: dict[str, Any], *, phase: str) -> dict[str, Any]:
    from benchmark.evaluator.scene_quality.adaptive_acceptance import _resolution_units
    resolution = resolution_of(record)
    units = (_resolution_units(record) if "check_episodes" in record else
             [resolution] if resolution is not None else _resolution_units(record))
    accepted = bool(units and all(item.get("accepted") is True for item in units)
                    and record.get("status") == "evaluated" and finite_score(record.get("score")))
    record["evidence_resolution_policy"] = FALLBACK_POLICY
    record["execution_complete"] = True
    if accepted:
        visual = all(item.get("visual_observation_complete") is True for item in units)
        record.update(terminal_state="evaluated" if visual else "evaluated_degraded")
        record.pop("infrastructure_failure", None)
        return record
    failure = failure_of(record)
    if failure and failure.get("failure_category") in {"evidence_gap", "evidence_unavailable"}:
        reason = str(record.get("reason") or "required_observation_missing")
        record.update(status="not_evaluable", score=None, terminal_state="evidence_gap", verdict="unknown")
        record["coverage_gap"] = gap_record(unit_id=phase, reason=reason,
            source="bounded_scope_review", evidence=resolution)
        record.pop("infrastructure_failure", None)
    else:
        failure = failure or {"failure_category": "unresolved_required_judgement", "phase": phase}
        record.update(status="failed", score=None, terminal_state="infrastructure_failure",
                      infrastructure_failure=failure)
    return record


def _scope_record(unit_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    from benchmark.evaluator.scene_quality.adaptive_acceptance import _resolution_units
    units = [unit for record in records for unit in _resolution_units(record)]
    accepted = bool(records and units and all(unit.get("accepted") is True for unit in units)
                    and all(record.get("status") == "evaluated" and finite_score(record.get("score"))
                            for record in records))
    failures = [failure for record in records if (failure := failure_of(record))]
    hard = [item for item in failures if item.get("failure_category") not in {"evidence_gap", "evidence_unavailable"}]
    failure = (hard or failures or [{}])[0]
    category = failure.get("failure_category")
    if not accepted and not category:
        category = "unresolved_required_judgement"
    return {
        "unit_id": unit_id, "status": "evaluated" if accepted else "not_evaluable" if
        category in {"evidence_gap", "evidence_unavailable"} else "failed",
        "score": min(float(record["score"]) for record in records) if accepted else None,
        "accepted": accepted, "execution_complete": True,
        "visual_observation_complete": bool(accepted and units and all(
            unit.get("visual_observation_complete") is True for unit in units)),
        "model_judgement_completed": bool(units and all(unit.get("model_judgement_completed") for unit in units)),
        "failure_category": None if accepted else "evidence_gap" if category == "evidence_unavailable" else category,
        "fallback_source": "metric_owned_scope",
        "fallback_reason": None if accepted else category,
    }


def _stored_scope_outcome(record: dict[str, Any]) -> dict[str, Any]:
    """Adapt a bare Judge record without overriding an explicit outcome.

    A verdict alone is insufficient: an accepted, completed model judgement
    receipt is required. Metric-owned scores and failure statuses stay intact.
    """
    if "status" in record or "score" in record:
        return record
    resolution = resolution_of(record)
    accepted = bool(
        resolution and resolution.get("accepted") is True
        and resolution.get("model_judgement_completed") is True
        and record.get("verdict") in {"valid", "invalid"}
        and record.get("terminal_state") in {None, "evaluated", "evaluated_degraded"}
        and failure_of(record) is None
    )
    return {
        "status": "evaluated" if accepted else "failed",
        "score": (0.0 if record["verdict"] == "invalid" else 1.0) if accepted else None,
        "judgement": record,
    }


def summarize_metric(report: dict[str, Any], planned: list[str]) -> dict[str, Any]:
    judgement = report.get("judgement") or {}
    actual: dict[str, list[dict[str, Any]]] = {}
    global_record = judgement.get("scene_global_judgement")
    if not isinstance(global_record, dict):
        global_record = report.get("global_discovery")
    if isinstance(global_record, dict):
        actual["scene_global"] = [_stored_scope_outcome(global_record)]
    for key in ("group_results", "group_judgements"):
        rows = report.get(key) or judgement.get(key) or []
        for row in rows if isinstance(rows, list) else []:
            group_id = row.get("group_id")
            if group_id:
                bucket = actual.setdefault("group:" + str(group_id), [])
                if row not in bucket:
                    bucket.append(row)
    if planned == ["metric"]:
        actual["metric"] = [report]
    # Dynamic downstream episodes are additional obligations, never replacements
    # for any original group. Typed check coverage is tracked separately below.
    for key, label in (("cross_group_relation_judgements", "relation"),
                       ("target_scope_results", "target"),
                       ("placement_global_handoff_reviews", "handoff"),
                       ("residual_global_placement_judgement", "residual")):
        rows = report.get(key) or judgement.get(key) or []
        if label == "residual" and not rows:
            rows = report.get("residual_global_placement_review") or []
            if not rows and (report.get("residual_global_placement_phase") or {}).get("required"):
                rows = [{"status": "failed", "score": None,
                         "failure": {"failure_category": "unresolved_required_judgement"}}]
        if isinstance(rows, dict):
            rows = [rows]
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            name = label + ":" + str(row.get("relation_id") or row.get("target_id") or index)
            actual[name] = [_stored_scope_outcome(row)]
    original = list(planned)
    all_scopes = original + [name for name in actual if name not in original]
    scope_records = [_scope_record(name, rows) for name, rows in actual.items()]
    scopes = coverage_from_plan(all_scopes, scope_records)
    check_records = []
    for key in ("placement_check_ledger", "functional_check_ledger"):
        checks = (report.get(key) or {}).get("checks") or []
        if isinstance(checks, dict):
            checks = list(checks.values())
        for check in checks:
            row = check.get("result_row") or {}
            resolution = resolution_of(row) or resolution_of(check)
            accepted = bool(resolution and resolution.get("accepted") and
                row.get("conclusion") in {"valid", "invalid", "excluded_function_owned"})
            check_records.append({
                "unit_id": key + ":" + str(check["check_id"]),
                "status": "evaluated" if accepted else "not_evaluable",
                "score": (0.0 if row.get("conclusion") == "invalid" else 1.0) if accepted else None,
                "accepted": accepted, "execution_complete": bool(row),
                "visual_observation_complete": bool(accepted and resolution.get("images_used")
                                                    and row.get("observation_status") == "observed"),
                "failure_category": None if accepted else "evidence_gap",
                "fallback_reason": None if accepted else "typed_check_unresolved",
                "fallback_source": "typed_check_ledger",
            })
    checks = coverage_from_plan([row["unit_id"] for row in check_records], check_records)
    all_records = scope_records + check_records
    combined = coverage_from_plan(all_scopes + checks["planned_ids"], all_records)
    complete = bool(scopes["evaluation_complete"] and
                    (not check_records or checks["evaluation_complete"]) and report.get("status") == "evaluated")
    combined.update(
        original_planned_scope_ids=original, scope_coverage=scopes, typed_check_coverage=checks,
        complete=complete, eligible_count=combined["planned_count"],
        accepted_count=combined["evaluation_count"], acceptance_fraction=combined["evaluation_fraction"],
        model_judgement_count=sum(row.get("model_judgement_completed") is True for row in all_records),
        structured_fallback_count=0, style_policy_count=0,
    )
    return combined


def finish_metric(report: dict[str, Any], planned: list[str]) -> dict[str, Any]:
    if report.get("functional_ownership_ledger", {}) is None:
        # Optional absent ownership is not an empty certified ledger.
        report.pop("functional_ownership_ledger")
    coverage = summarize_metric(report, planned)
    report.update(evidence_resolution_policy=FALLBACK_POLICY, resolution_coverage=coverage,
                  execution_complete=True)
    if not coverage["complete"]:
        hard = bool(report.get("infrastructure_failures")) or coverage["failure_count"] > 0 or bool(failure_of(report) and
            failure_of(report).get("failure_category") not in {"evidence_gap", "evidence_unavailable"})
        report.update(status="failed" if hard else "not_evaluable", score=None,
                      terminal_state="infrastructure_failure" if hard else "evidence_gap")
    return report
