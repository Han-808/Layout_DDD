"""Evidence-adaptive audit for existing deterministic and model L1 decisions."""

from __future__ import annotations

from copy import deepcopy
from functools import wraps
from typing import Any, Callable
from benchmark.visual_judge.evidence_gap_v2 import enabled as fallback_v2_enabled
from benchmark.visual_judge.evidence_resolution import policy_of, FALLBACK_POLICY

from benchmark.visual_judge.evidence_resolution import (
    ADAPTIVE_POLICY, adaptive_enabled, finite_score, resolution_of,
)


def coverage_from_units(units: list[dict[str, Any]], *, complete: bool) -> dict[str, Any]:
    count = len(units)
    accepted = sum(unit.get("accepted") is True for unit in units)
    fields = {
        "accepted": accepted,
        "model_judgement": sum(unit.get("model_judgement_completed") is True for unit in units),
        "visual_evidence": sum(unit.get("visual_observation_complete") is True for unit in units),
        "structured_fallback": sum(unit.get("evidence_tier") == "structured_fallback"
                                   and unit.get("decision_source") == "model" for unit in units),
        "style_policy": sum(unit.get("decision_source") == "style_policy_no_deduction" for unit in units),
        "deterministic": sum(unit.get("decision_source") == "deterministic_rule" for unit in units),
    }
    return {
        "schema_version": "adaptive_judgement_coverage_v1", "policy": ADAPTIVE_POLICY,
        "eligible_count": count, "complete": bool(complete and accepted == count),
        **{name + "_count": value for name, value in fields.items()},
        **{("acceptance_fraction" if name == "accepted" else name + "_fraction"):
           value / count if count else float(complete and name == "accepted")
           for name, value in fields.items()},
        "units": deepcopy(units),
    }


def adaptive_l1_result(function: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if fallback_v2_enabled():
            import inspect
            from benchmark.evaluator.consistency_audit_v2 import finish_l1
            from benchmark.visual_judge.evidence_resolution import failure_record
            arguments = inspect.signature(function).bind(*args, **kwargs).arguments
            metric = function.__name__.removeprefix("check_")
            try:
                report = function(*args, **kwargs)
            except Exception as exc:
                report = {"metric": metric, "status": "failed", "score": None,
                          "failure": failure_record(exc, phase="metric"), "execution_complete": True}
            if report.get("status") == "not_applicable":
                return report
            return finish_l1(report, arguments, metric=metric)
        report = function(*args, **kwargs)
        if not adaptive_enabled() or report.get("status") == "not_applicable":
            return report
        units = []
        metric = str(report.get("metric") or "")
        for index, record in enumerate(report.get("pairs" if metric == "collision" else "objects") or []):
            resolution = resolution_of(record)
            if resolution is None:
                accepted = (
                    record.get("final_verdict") in {"valid", "invalid"}
                    and not record.get("requires_vlm")
                    and not record.get("adjudication_error")
                )
                resolution = {
                    "schema_version": "adaptive_evidence_resolution_v1", "policy": ADAPTIVE_POLICY,
                    "metric": metric, "unit_key": str(index),
                    "accepted": accepted, "decision_source": "deterministic_rule" if accepted else None,
                    "evidence_tier": "deterministic_geometry" if accepted else "unresolved",
                    "fallback": False, "images_used": [],
                    "model_invoked": False, "model_judgement_completed": False,
                    "visual_observation_complete": False, "trigger_reason": record.get("route"),
                }
            record["evidence_resolution"] = deepcopy(resolution)
            units.append(resolution)
        report["evidence_resolution_policy"] = ADAPTIVE_POLICY
        report["resolution_coverage"] = coverage_from_units(
            units, complete=report.get("status") == "checked" and finite_score(report.get("score")),
        )
        if not report["resolution_coverage"]["complete"]:
            report.update(status="requires_vlm", score=None)
        return report
    return wrapped


def adaptive_layer_accepted(report: dict[str, Any]) -> bool:
    coverage = report.get("resolution_coverage")
    if isinstance(coverage, dict):
        return coverage.get("complete") is True and finite_score(report.get("score"))
    metrics = report.get("metrics")
    if isinstance(metrics, dict):
        active = [value for value in metrics.values() if isinstance(value, dict)
                  and value.get("enabled", True) and value.get("status") != "not_applicable"]
        if any(value.get("evidence_resolution_policy") in {ADAPTIVE_POLICY, FALLBACK_POLICY} for value in active):
            return bool(active and all(adaptive_layer_accepted(value) for value in active)
                        and finite_score(report.get("score")))
    # Out-of-policy layers (L0/L2/L4) retain their original completion semantics.
    return finite_score(report.get("score")) and report.get("status") not in {
        "failed", "partial", "requires_vlm", "unresolved", "not_applicable", "not_implemented",
    }


def policy_code_identity() -> dict[str, str]:
    import hashlib
    from pathlib import Path
    package = Path(__file__).resolve().parents[1]
    files = [path for path in package.rglob("*.py") if "__pycache__" not in path.parts]
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    return {"policy": policy_of(), "source_tree_sha256": digest.hexdigest(),
            "source_root": str(package), "identity_scope": "benchmark_python_source"}


def apply_adaptive_layer_audit(report: dict[str, Any]) -> None:
    if not adaptive_enabled():
        return
    if fallback_v2_enabled():
        from benchmark.evaluator.consistency_audit_v2 import inventory_coverage
        metrics = report.get("metrics") or {}
        planned = [name for name, item in metrics.items()
                   if isinstance(item, dict) and item.get("enabled", True) and item.get("status") != "not_applicable"]
        coverage = inventory_coverage(metrics, planned)
        report.update(evidence_resolution_policy=FALLBACK_POLICY, resolution_coverage=coverage, execution_complete=True)
        if not coverage["complete"]:
            report["score"] = None
        report.setdefault("coverage", {}).update(
            complete=coverage["complete"], acceptance_complete=coverage["complete"],
            visual_evidence_fraction=coverage["visual_evidence_fraction"],
            coverage_threshold_passed=coverage["complete"],
        )
        return
    active = [item for item in (report.get("metrics") or {}).values()
              if isinstance(item, dict) and item.get("enabled", True)
              and item.get("status") != "not_applicable"]
    if not active or not all(item.get("evidence_resolution_policy") == ADAPTIVE_POLICY for item in active):
        return
    units = [unit for item in active for unit in item.get("resolution_coverage", {}).get("units", [])]
    complete = all(adaptive_layer_accepted(item) for item in active) and finite_score(report.get("score"))
    summary = coverage_from_units(units, complete=complete)
    report["evidence_resolution_policy"] = ADAPTIVE_POLICY
    report["resolution_coverage"] = summary
    report.setdefault("coverage", {}).update(
        acceptance_complete=complete, complete=complete,
        visual_evidence_fraction=summary["visual_evidence_fraction"],
        fraction=summary["visual_evidence_fraction"],
        grounded_score_fraction=summary["visual_evidence_fraction"],
        score_grounding_complete=complete and summary["visual_evidence_fraction"] == 1.0,
        coverage_threshold_passed=complete,
    )


def adaptive_nonrect_report_result(function: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        report = function(*args, **kwargs)
        rooms = [value.get("report") for value in (report.get("rooms") or {}).values()]
        marked = [room for room in rooms if isinstance(room, dict)
                  and room.get("evidence_resolution_policy") == ADAPTIVE_POLICY]
        if not marked:
            return report
        units = []
        for name, metric in report.get("aggregate", {}).get("metrics", {}).items():
            records = [room["metrics"][name]["raw_report"] for room in marked]
            selected = [unit for record in records for unit in record.get("resolution_coverage", {}).get("units", [])]
            units.extend(selected)
            complete = len(marked) == len(rooms) and all(adaptive_layer_accepted(record) for record in records)
            metric["resolution_coverage"] = coverage_from_units(selected, complete=complete)
            metric["evidence_resolution_policy"] = ADAPTIVE_POLICY
        report["evidence_resolution_policy"] = ADAPTIVE_POLICY
        report["resolution_coverage"] = coverage_from_units(
            units, complete=report.get("terminal_status") == "complete" and len(marked) == len(rooms),
        )
        report["coverage"]["acceptance_complete"] = report["resolution_coverage"]["complete"]
        report["coverage"]["visual_evidence_fraction"] = report["resolution_coverage"]["visual_evidence_fraction"]
        report["provenance"]["evidence_resolution_implementation"] = policy_code_identity()
        return report
    return wrapped


def adaptive_persisted_summary(function: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = function(*args, **kwargs)
        sources = {
            **(kwargs.get("l1_report", {}).get("metrics") or {}),
            **(kwargs.get("l3_report", {}).get("metrics") or {}),
        }
        if any('judgement_coverage_projection' in item for item in sources.values() if isinstance(item, dict)):
            from benchmark.evaluator.judgement_coverage import finish_persisted
            return finish_persisted(result, sources)
        if fallback_v2_enabled() or any(item.get("evidence_resolution_policy") == FALLBACK_POLICY
                                       for item in sources.values() if isinstance(item, dict)):
            from benchmark.evaluator.consistency_audit_v2 import finish_persisted
            return finish_persisted(result, sources)
        records = result.get("metrics") or []
        active_records = [item for item in records if item.get("overall_weight", 0) > 0]
        if not active_records or not all(sources.get(item["metric"], {}).get("evidence_resolution_policy") == ADAPTIVE_POLICY
                                         for item in active_records):
            return result
        result["evidence_resolution_policy"] = ADAPTIVE_POLICY
        units = []
        for item in records:
            source = sources.get(item["metric"]) or {}
            resolution = source.get("resolution_coverage") or {}
            accepted = adaptive_layer_accepted(source)
            visual = float(resolution.get("visual_evidence_fraction") or 0.0)
            score = float(source["score"]) if accepted else None
            item.update(
                evidence_resolution_policy=ADAPTIVE_POLICY, resolution_coverage=deepcopy(resolution),
                score=score, observed_score=score, score_status="complete" if accepted else "insufficient_metric_coverage",
                coverage_fraction=visual, coverage_complete=visual == 1.0,
                coverage_threshold_passed=accepted, accepted=accepted,
                grounded_overall_weight=float(item.get("overall_weight") or 0.0) * visual,
                weighted_points=score * float(item.get("overall_weight") or 0.0) * 100 if score is not None else None,
            )
            units.extend(resolution.get("units") or [])
        active = [item for item in records if item.get("overall_weight", 0) > 0]
        complete = bool(active and all(item["accepted"] for item in active))
        total_weight = sum(float(item["overall_weight"]) for item in active)
        result["resolution_coverage"] = coverage_from_units(units, complete=complete)
        result["combined_score_100"] = (
            sum(item["weighted_points"] for item in active) / total_weight if complete and total_weight else None
        )
        result["combined_observed_score_100"] = result["combined_score_100"]
        result["combined_status"] = "complete" if complete else "insufficient_metric_coverage"
        result["combined_coverage_fraction"] = (
            sum(item["grounded_overall_weight"] for item in active) / total_weight if total_weight else 0.0
        )
        for layer in result.get("layers") or []:
            selected = [item for item in active if item["layer"] == layer["layer"]]
            accepted = bool(selected and all(item["accepted"] for item in selected))
            weight = sum(float(item["local_weight"]) for item in selected)
            score = (sum(item["score"] * float(item["local_weight"]) for item in selected) / weight
                     if accepted and weight else None)
            visual = sum(float(item["local_weight"]) * item["coverage_fraction"] for item in selected) / weight if weight else 0.0
            layer.update(score=score, observed_score=score, score_status="complete" if accepted else "insufficient_metric_coverage")
            layer["coverage"].update(complete=accepted, acceptance_complete=accepted, fraction=visual,
                                     grounded_score_fraction=visual, coverage_threshold_passed=accepted,
                                     score_grounding_complete=accepted and visual == 1.0)
        return result
    return wrapped
