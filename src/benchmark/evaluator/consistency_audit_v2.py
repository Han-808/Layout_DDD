"""V2 fixed-plan accounting without changing metric-owned geometry rules."""
from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from typing import Any, Mapping

from benchmark.visual_judge.evidence_gap_v2 import coverage_from_plan
from benchmark.visual_judge.evidence_resolution import FALLBACK_POLICY, finite_score, resolution_of


def compatible_coverage(planned: list[str], records: list[dict[str, Any]]) -> dict[str, Any]:
    result = coverage_from_plan(planned, records)
    result.update(complete=result["evaluation_complete"], eligible_count=result["planned_count"],
                  accepted_count=result["evaluation_count"], acceptance_fraction=result["evaluation_fraction"])
    return result


def inventory_coverage(metrics: Mapping[str, Any], planned: list[str]) -> dict[str, Any]:
    records = []
    for name in planned:
        item = metrics.get(name)
        if not isinstance(item, dict):
            continue
        coverage = item.get("resolution_coverage") or {}
        accepted = coverage.get("complete") is True and finite_score(item.get("score"))
        category = None if accepted else "evidence_gap" if item.get("status") in {
            "not_evaluable", "finished_with_gaps",
        } else "metric_failure"
        records.append({
            "unit_id": name, "status": "evaluated" if accepted else "not_evaluable" if category == "evidence_gap" else "failed",
            "score": item.get("score") if accepted else None, "accepted": accepted,
            "execution_complete": item.get("execution_complete", True),
            "visual_observation_complete": bool(accepted and coverage.get("visual_evidence_complete")),
            "failure_category": category,
        })
    return compatible_coverage(planned, records)


def finish_l1(report: dict[str, Any], arguments: dict[str, Any], *, metric: str) -> dict[str, Any]:
    scene = arguments.get("scene") or {}
    ids = [str(item.get("id") or "") for item in scene.get("objects") or [] if isinstance(item, dict)]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("L1 fixed plan requires unique scene object IDs")
    pair = metric == "collision"
    identity = lambda a, b: "|".join(sorted((a, b)))
    planned = [identity(a, b) for a, b in combinations(ids, 2)] if pair else ids
    rows = report.get("pairs" if pair else "objects") or []
    records = []
    for row in rows:
        unit_id = identity(str(row.get("object_a") or ""), str(row.get("object_b") or "")) if pair else str(row.get("object_id") or "")
        resolution = resolution_of(row)
        failure = row.get("failure") or row.get("adjudication_failure") or {}
        accepted = bool((resolution or {}).get("accepted") if resolution else (
            row.get("final_verdict") in {"valid", "invalid"} and not row.get("requires_vlm")
            and not row.get("adjudication_error") and not failure))
        category = failure.get("failure_category")
        if not accepted and not category:
            category = "unresolved_required_judgement"
        category = "evidence_gap" if category == "evidence_unavailable" else category
        source = (resolution or {}).get("decision_source") if resolution else "deterministic_rule" if accepted else None
        audit = {
            **deepcopy(resolution or {}), "policy": FALLBACK_POLICY, "metric": metric,
            "unit_id": unit_id, "unit_key": unit_id, "accepted": accepted,
            "status": "evaluated" if accepted else "not_evaluable" if category == "evidence_gap" else "failed",
            "score": (0.0 if row.get("final_verdict") == "invalid" else 1.0) if accepted else None,
            "execution_complete": True, "decision_source": source,
            "visual_observation_complete": bool(accepted and resolution and resolution.get("visual_observation_complete")),
            "failure_category": None if accepted else category,
            "fallback_source": source or "bounded_metric_review",
            "fallback_reason": None if accepted else category,
        }
        row["evidence_resolution"] = deepcopy(audit)
        records.append(audit)
    if not planned:
        # E.g. one object gives no collision pairs. The existing metric's
        # deterministic empty-domain rule is a certificate, not a VLM fallback.
        planned = ["metric_empty_domain"]
        accepted = report.get("status") == "checked" and finite_score(report.get("score"))
        records = [{
            "unit_id": planned[0], "status": "evaluated" if accepted else "failed",
            "score": report.get("score") if accepted else None, "accepted": accepted,
            "execution_complete": True, "visual_observation_complete": False,
            "decision_source": "deterministic_rule" if accepted else None,
            "failure_category": None if accepted else "invalid_metric_input",
        }]
    coverage = compatible_coverage(planned, records)
    coverage["complete"] = bool(coverage["complete"] and report.get("status") == "checked" and finite_score(report.get("score")))
    report.update(evidence_resolution_policy=FALLBACK_POLICY, resolution_coverage=coverage, execution_complete=True)
    if not coverage["complete"]:
        hard = coverage["failure_count"] > 0 or bool(report.get("failure"))
        report.update(status="failed" if hard else "not_evaluable", score=None,
                      terminal_state="infrastructure_failure" if hard else "evidence_gap")
    return report


def finish_persisted(result: dict[str, Any], sources: dict[str, Any]) -> dict[str, Any]:
    records = result.get("metrics") or []
    active = [item for item in records if item.get("overall_weight", 0) > 0]
    coverage = inventory_coverage(sources, [item["metric"] for item in active])
    for item in records:
        source = sources.get(item["metric"]) or {}
        summary = source.get("resolution_coverage") or {}
        accepted = summary.get("complete") is True and finite_score(source.get("score"))
        score = float(source["score"]) if accepted else None
        visual = float(summary.get("visual_evidence_fraction") or 0)
        item.update(evidence_resolution_policy=FALLBACK_POLICY, resolution_coverage=deepcopy(summary),
                    accepted=accepted, score=score, observed_score=score,
                    score_status="complete" if accepted else "not_evaluable",
                    coverage_fraction=visual, coverage_complete=summary.get("visual_evidence_complete") is True,
                    coverage_threshold_passed=accepted,
                    grounded_overall_weight=float(item.get("overall_weight") or 0) * visual,
                    weighted_points=score * float(item.get("overall_weight") or 0) * 100 if accepted else None)
    total = sum(float(item["overall_weight"]) for item in active)
    score = sum(item["weighted_points"] for item in active) / total if coverage["complete"] and total else None
    result.update(evidence_resolution_policy=FALLBACK_POLICY, resolution_coverage=coverage,
                  execution_complete=True, combined_score_100=score, combined_observed_score_100=score,
                  combined_status="complete" if coverage["complete"] else "finished_with_gaps",
                  combined_coverage_fraction=coverage["visual_evidence_fraction"])
    for layer in result.get("layers") or []:
        selected = [item for item in active if item["layer"] == layer["layer"]]
        accepted = bool(selected and all(item["accepted"] for item in selected))
        weight = sum(float(item["local_weight"]) for item in selected)
        score = sum(item["score"] * float(item["local_weight"]) for item in selected) / weight if accepted and weight else None
        layer.update(score=score, observed_score=score, score_status="complete" if accepted else "not_evaluable")
        layer.setdefault("coverage", {}).update(complete=accepted, acceptance_complete=accepted,
                                                 coverage_threshold_passed=accepted)
    return result
