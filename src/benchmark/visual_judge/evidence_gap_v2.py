"""Explicit, opt-in evidence gaps; no generic geometry or semantic verdicts.

Metric-owned deterministic rules may retain their existing conclusions. This
module only accounts for evidence/coverage; it never creates a score from a
bounding box, a generator annotation, or a failed model call.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping

from benchmark.visual_judge.evidence_resolution import (
    AdaptiveEvidenceError, FALLBACK_POLICY, finite_score, policy_of,
)


GAP_SCHEMA = "evidence_coverage_gap_v2"
TERMINAL_INSTRUCTION = """Acquisition has ended. Reuse the supplied legal visual
evidence and metric-owned measurements without claiming unseen observations.
Missing evidence is neither validity nor invalidity. Return a supported
valid/invalid judgement only when the original metric rules and all required
checks are sufficiently supported. Otherwise retain insufficient/ambiguous
and unresolved typed rows; never guess a binary verdict. A structured
evidence_request may describe what remains missing, but it will be recorded as
a coverage gap, not executed. Do not change claim types, defect owners or
semantic conclusions to satisfy the response contract."""


def enabled(value: Any = None) -> bool:
    return policy_of(value) == FALLBACK_POLICY


class EvidenceGapError(AdaptiveEvidenceError):
    """Normal local exhaustion, not transport, input or renderer failure."""

    failure_category = "evidence_gap"

    def __init__(self, reason: str, *, audit: dict[str, Any] | None = None):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("evidence gap requires an explicit reason")
        super().__init__(reason, audit=audit)


_NORMAL_EXHAUSTION = frozenset({
    "placement_handoff_deferred",
    "trusted_candidate_bank_empty", "no_feasible_candidate", "vlm_no_feasible_candidate",
    "evidence_round_budget_exhausted", "max_evidence_rounds_exhausted",
    "max_camera_actions_exhausted", "max_selector_calls_exhausted",
    "max_total_images_exhausted", "fixed_views_insufficient",
    "evidence_packet_unchanged", "evidence_packet_already_judged",
    "judge_evidence_request_disabled",
})
_SERVICE_ERRORS = frozenset({
    "HTTPError", "URLError", "AuthenticationError", "EndpointPreflightError",
    "ModelRequestError", "OpenAICompatibleError", "OpenAICompatibleModelError",
    "EndpointStabilityPreflightError",
})
_ACQUISITION_GAPS = frozenset({"evidence_gap", "evidence_unavailable"})


def classify_failure(error: BaseException, *, phase: str) -> dict[str, Any]:
    """Only positively typed/structured normal exhaustion is a recoverable gap.

    In particular a generic RuntimeError, OSError, missing file, or render
    failure is not evidence exhaustion. Follow causes so a transport failure
    during response repair cannot be disguised as an ordinary schema failure.
    No exception prose is copied into the public classification record.
    """
    return _classify_failure(error, phase=phase, seen=set())


def _classify_failure(
    error: BaseException, *, phase: str, seen: set[int],
) -> dict[str, Any]:
    if id(error) in seen:
        return {
            "failure_category": "implementation_or_input_failure", "phase": phase,
            "error_type": type(error).__name__, "recoverable_acquisition": False,
        }
    seen.add(id(error))
    names = {cls.__name__ for cls in type(error).__mro__}
    category = getattr(error, "failure_category", None)
    response_failure = bool(names & {"ResponseSchemaRepairError", "JSONDecodeError"})
    bounded_camera = "NonRectangularCameraEvidenceExhausted" in names
    if not category and "EvidenceControlUnresolvedError" in names:
        result = getattr(error, "result", None)
        audit = getattr(result, "audit", {})
        failure = audit.get("failure", {}) if isinstance(audit, dict) else {}
        category = failure.get("failure_category") if isinstance(failure, dict) else None
        if not category and getattr(result, "stop_reason", None) in _NORMAL_EXHAUSTION:
            category = "evidence_unavailable"
        if not category and getattr(result, "stop_reason", None) == "evidence_missing":
            # No packet was supplied is distinct from a supplied file missing
            # on disk. Require the structured empty-packet gate certificate;
            # the stop-reason spelling alone must never hide broken artifacts.
            gates = [item.get("result") for item in audit.get("trace") or []
                     if item.get("stage") == "evidence_gate"]
            if gates and all(
                isinstance(gate, dict) and gate.get("reason_codes") == ["visual_evidence_missing"]
                and (gate.get("provenance") or {}).get("evidence_item_count") == 0
                for gate in gates
            ) and not ((audit.get("judge_request") or {}).get("visual_evidence")):
                category = "evidence_gap"
    if not category:
        if isinstance(error, (TimeoutError, ConnectionError)) or names & _SERVICE_ERRORS:
            category = "model_service_failure"
        elif isinstance(error, PermissionError):
            category = "permission_failure"
        elif bounded_camera:
            category = "evidence_unavailable"
        elif response_failure:
            category = "judge_response_failure"
        elif isinstance(error, FileNotFoundError):
            category = "input_integrity_failure"
        elif isinstance(error, OSError):
            category = "infrastructure_failure"
        else:
            category = "implementation_or_input_failure"
    if error.__cause__ is not None and error.__cause__ is not error:
        cause = _classify_failure(error.__cause__, phase=phase, seen=seen)
        expected_wrapper = (
            cause["failure_category"] == "implementation_or_input_failure"
            and (
                response_failure and isinstance(error.__cause__, (ValueError, TypeError, KeyError))
                or bounded_camera and type(error.__cause__) is RuntimeError
            )
        )
        if not cause["recoverable_acquisition"] and not expected_wrapper:
            category = cause["failure_category"]
        elif (
            cause["recoverable_acquisition"]
            and "EvidenceRenderFailure" in names
            and isinstance(getattr(error, "provenance", None), dict)
            and isinstance(error.provenance.get("provider_usage"), dict)
            and not error.provenance.get("usage_observation_error")
        ):
            # The legacy provider adapter wraps even typed local exhaustion.
            # Preserve that classification only after usage was observed;
            # arbitrary render errors and broken telemetry remain failures.
            category = cause["failure_category"]
    return {
        "failure_category": str(category), "phase": phase,
        "error_type": type(error).__name__,
        "recoverable_acquisition": category in _ACQUISITION_GAPS,
    }


def gap_record(
    *, unit_id: str, reason: str, source: str,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not all(isinstance(value, str) and value.strip() for value in (unit_id, reason, source)):
        raise ValueError("gap identity, reason and source are required")
    return {
        "schema_version": GAP_SCHEMA, "policy": FALLBACK_POLICY,
        "unit_id": unit_id, "status": "not_evaluable", "verdict": "unknown",
        "score": None, "execution_complete": True, "accepted": False,
        "visual_observation_complete": False, "decision_source": None,
        "fallback_source": source, "fallback_reason": reason,
        "evidence": deepcopy(dict(evidence or {})),
        "failure_category": "evidence_gap",
    }


def coverage_from_plan(
    planned_ids: Iterable[str], records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Keep the pre-execution denominator, including units with no result.

Duplicate/unplanned records are integrity errors, not last-write-wins merges.
Missing records are pending/unaccounted, not successful executions or scores.
"""
    planned = list(planned_ids)
    if any(not isinstance(item, str) or not item.strip() for item in planned):
        raise ValueError("planned units require nonempty string IDs")
    if len(planned) != len(set(planned)):
        raise ValueError("duplicate planned unit ID")
    planned_set = set(planned)
    by_id: dict[str, dict[str, Any]] = {}
    for item in records:
        unit = deepcopy(dict(item))
        unit_id = unit.get("unit_id")
        if unit_id not in planned_set:
            raise ValueError("result unit was not in the original plan")
        if unit_id in by_id:
            raise ValueError("duplicate result unit ID")
        if unit.get("accepted") is True and (
            not finite_score(unit.get("score"))
            or unit.get("status") not in {"evaluated", "complete"}
            or unit.get("failure_category")
        ):
            raise ValueError("accepted result requires a legal score and no failure")
        if unit.get("status") == "not_evaluable" and (
            unit.get("score") is not None or unit.get("accepted") is True
        ):
            raise ValueError("not_evaluable results must retain a null score")
        by_id[unit_id] = unit
    missing = [item for item in planned if item not in by_id]
    accepted = sum(item.get("accepted") is True for item in by_id.values())
    visual = sum(item.get("visual_observation_complete") is True for item in by_id.values())
    executed = sum(item.get("execution_complete") is True for item in by_id.values())
    gaps = [item for item in by_id.values() if item.get("failure_category") == "evidence_gap"]
    failures = [item for item in by_id.values() if item.get("failure_category") not in {None, "evidence_gap"}]
    total = len(planned)
    return {
        "schema_version": "planned_evidence_coverage_v2", "policy": FALLBACK_POLICY,
        "planned_count": total, "planned_ids": planned, "missing_ids": missing,
        "execution_complete_count": executed,
        "execution_complete": not missing and executed == total,
        "evaluation_complete": bool(total and accepted == total),
        "evaluation_count": accepted,
        "evaluation_fraction": accepted / total if total else None,
        "visual_evidence_count": visual,
        "visual_evidence_fraction": visual / total if total else None,
        "visual_evidence_complete": bool(total and visual == total),
        "gap_count": len(gaps), "failure_count": len(failures),
        "coverage_gaps": gaps, "failures": failures,
        "units": [by_id[item] for item in planned if item in by_id],
    }
