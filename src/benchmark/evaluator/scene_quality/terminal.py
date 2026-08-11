"""Terminal-state helpers for required Scene Quality Judge scopes.

Final global, relation, and group scopes are not routers.  They must either
produce a scientific binary result or expose an engineering failure.  Keeping
this distinction explicit prevents a selector, renderer, endpoint, or schema
failure from being reported as ordinary evidence ambiguity.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


TERMINAL_EVALUATED = "evaluated"
TERMINAL_EVALUATED_DEGRADED = "evaluated_degraded"
TERMINAL_INFRASTRUCTURE_FAILURE = "infrastructure_failure"

_ENGINEERING_STOP_REASONS = {
    "camera_selector_failed",
    "invalid_selector_response",
    "render_failed",
    "manifest_failure",
    "scene_contract_failure",
    "evidence_gate_failed",
}


def controller_stop_reason(audit: Any) -> str | None:
    """Return the normalized Controller stop reason from an audit record."""

    if not isinstance(audit, dict):
        return None
    nested = audit.get("audit")
    if isinstance(nested, dict):
        audit = nested
    value = audit.get("stop_reason")
    if value is None and isinstance(audit.get("camera_acquisition"), dict):
        value = audit["camera_acquisition"].get("stop_reason")
    normalized = str(value or "").strip()
    return normalized or None


def terminalize_required_scope(
    record: dict[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    """Attach a fail-closed terminal classification to a final Judge scope.

    A scientific ``need_more_evidence`` result should already have been
    finalized by the Controller.  If it reaches this boundary unresolved, the
    control contract failed; it is not a third scientific verdict.
    """

    status = str(record.get("status") or "").strip().lower()
    audit = record.get("camera_control_audit")
    stop_reason = controller_stop_reason(audit)
    forced = _forced_choice_applied(record)
    if status == "evaluated":
        degraded = bool(
            record.get("terminal_state") == TERMINAL_EVALUATED_DEGRADED
            or forced
            or stop_reason in _ENGINEERING_STOP_REASONS
            or (stop_reason and stop_reason.endswith("_forced_choice"))
        )
        record["terminal_state"] = (
            TERMINAL_EVALUATED_DEGRADED
            if degraded
            else TERMINAL_EVALUATED
        )
        if degraded:
            record["degradation_audit"] = {
                "phase": phase,
                "controller_stop_reason": stop_reason,
                "forced_binary": forced,
            }
        return record

    judgement = record.get("judgement")
    judgement = judgement if isinstance(judgement, dict) else {}
    error_type = str(judgement.get("error_type") or "").strip() or None
    error = str(judgement.get("error") or "").strip() or None
    original_reason = str(record.get("reason") or "").strip() or None
    explicit_engineering_failure = bool(
        status in {"failed", "error", "infrastructure_failure"}
        or error_type
        or stop_reason in _ENGINEERING_STOP_REASONS
        or _reason_is_engineering_failure(original_reason)
    )
    failure_kind = (
        "engineering_failure"
        if explicit_engineering_failure
        else "terminal_contract_failure"
    )
    record.update(
        status="failed",
        score=None,
        reason=original_reason or failure_kind,
        terminal_state=TERMINAL_INFRASTRUCTURE_FAILURE,
        infrastructure_failure={
            "phase": phase,
            "failure_kind": failure_kind,
            "original_status": status or "unknown",
            "original_reason": original_reason,
            "controller_stop_reason": stop_reason,
            "error_type": error_type,
            "error": error,
        },
    )
    return record


def required_scope_failure(
    *,
    phase: str,
    scope_id: str | None,
    reason: str,
    error_type: str | None = None,
    error: str | None = None,
    controller_stop: str | None = None,
) -> dict[str, Any]:
    """Create the compact immutable failure record used by aggregation."""

    return {
        "phase": phase,
        "scope_id": scope_id,
        "failure_kind": "engineering_failure",
        "reason": reason,
        "controller_stop_reason": controller_stop,
        "error_type": error_type,
        "error": error,
    }


def infrastructure_failure_from_scope(
    record: Any,
    *,
    phase: str,
    scope_id: str | None = None,
) -> dict[str, Any] | None:
    """Return an aggregation-safe failure record for a terminalized scope."""

    if not isinstance(record, dict):
        return required_scope_failure(
            phase=phase,
            scope_id=scope_id,
            reason="missing_required_scope_record",
        )
    if record.get("terminal_state") != TERMINAL_INFRASTRUCTURE_FAILURE:
        return None
    failure = record.get("infrastructure_failure")
    failure = deepcopy(failure) if isinstance(failure, dict) else {}
    judgement = record.get("judgement")
    judgement = judgement if isinstance(judgement, dict) else {}
    return {
        "phase": phase,
        "scope_id": scope_id,
        "failure_kind": str(
            failure.get("failure_kind") or "engineering_failure"
        ),
        "reason": str(
            failure.get("original_reason")
            or record.get("reason")
            or "required_scope_failed"
        ),
        "controller_stop_reason": failure.get("controller_stop_reason"),
        "error_type": failure.get("error_type") or judgement.get("error_type"),
        "error": failure.get("error") or judgement.get("error"),
    }


def _forced_choice_applied(value: Any) -> bool:
    if isinstance(value, dict):
        forced = value.get("budget_exhaustion_forced_choice")
        if isinstance(forced, dict) and forced.get("applied") is True:
            return True
        return any(_forced_choice_applied(child) for child in value.values())
    if isinstance(value, list):
        return any(_forced_choice_applied(child) for child in value)
    return False


def _reason_is_engineering_failure(reason: str | None) -> bool:
    normalized = str(reason or "").lower()
    return any(
        token in normalized
        for token in (
            "failed",
            "failure",
            "unavailable",
            "invalid_response",
            "render_error",
            "manifest_error",
            "schema",
            "endpoint",
        )
    )
