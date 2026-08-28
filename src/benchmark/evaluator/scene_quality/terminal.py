"""Terminal-state helpers for required Scene Quality Judge scopes.

Final scopes expose a binary scientific result whenever a legal input and at
least one retained evidence packet make that possible.  A non-hard control or
schema failure therefore becomes an audited, low-confidence default rather
than a third verdict.  Transport, input, and unexpected program failures stay
explicit infrastructure failures.
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

_HARD_STOP_REASONS = {
    "blank_evidence",
    "corrupt_evidence",
    "undecodable_evidence",
    "evidence_missing",
    "evidence_integrity_failure",
    "manifest_failure",
    "scene_contract_failure",
    "evidence_gate_failed",
    "renderer_followup_contract_invalid",
}
_SUCCESSFUL_STOP_REASONS = {
    "deterministic_conclusion",
    "judge_conclusion",
    "sufficient_evidence",
}
_RECOVERABLE_WITH_RETAINED_STOP_REASONS = {
    "camera_candidate_bank_failed",
    "camera_constraint_contract_invalid",
    "camera_preview_render_failed",
    "camera_preview_renderer_unavailable",
    "camera_selector_failed",
    "camera_selector_unavailable",
    "evidence_packet_already_judged",
    "evidence_packet_unchanged",
    "evidence_round_budget_exhausted",
    "fixed_views_insufficient",
    "judge_evidence_request_disabled",
    "judge_evidence_request_outside_group_scope",
    "max_camera_actions_exhausted",
    "max_evidence_rounds_exhausted",
    "max_selector_calls_exhausted",
    "max_total_images_exhausted",
    "no_feasible_candidate",
    "render_failed",
    "surface_target_invalid",
    "trusted_candidate_bank_empty",
    "usable_surface_decode_failed",
    "vlm_no_feasible_candidate",
}

_HARD_ERROR_TOKENS = (
    "endpoint",
    "http",
    "urlerror",
    "connection",
    "timeout",
    "authentication",
    "authorization",
    "rate_limit",
    "ratelimit",
    "file_not_found",
    "filenotfound",
    "unidentifiedimage",
    "scene_contract",
    "input_contract",
)
_HARD_REASON_TOKENS = (
    "object_grouping_unavailable",
    "vlm_judge_not_configured",
    "scene_contract_failure",
    "canonical_input",
    "render_evidence_unavailable",
    "evidence_packet_unavailable",
    "missing_required_evidence",
    "blank_evidence",
    "corrupt_evidence",
    "undecodable_evidence",
)
_RECOVERABLE_ERROR_TYPES = {
    "responseschemarepairerror",
}


def controller_stop_reason(audit: Any) -> str | None:
    """Return the normalized Controller stop reason from an audit record."""

    if not isinstance(audit, dict):
        return None
    value = audit.get("stop_reason")
    nested = audit.get("audit")
    if isinstance(nested, dict):
        audit = nested
    if value is None:
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
    """Finalize one scope as binary, degraded binary, or hard failure."""

    status = str(record.get("status") or "").strip().lower()
    audit = record.get("camera_control_audit")
    stop_reason = controller_stop_reason(audit)
    judgement = record.get("judgement")
    judgement = judgement if isinstance(judgement, dict) else {}
    forced = bool(
        _forced_choice_applied(record)
        or judgement.get("forced_binary") is True
        or judgement.get("defaulted") is True
        or judgement.get("evidence_ambiguous") is True
    )
    hard_failure = _is_hard_failure(
        record,
        error_type=(
            str(judgement.get("error_type") or "").strip() or None
        ),
        reason=str(record.get("reason") or "").strip() or None,
        stop_reason=stop_reason,
    )
    if status == "evaluated" and not hard_failure:
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

    error_type = str(judgement.get("error_type") or "").strip() or None
    error = str(judgement.get("error") or "").strip() or None
    original_reason = str(record.get("reason") or "").strip() or None
    if not hard_failure:
        failure = {
            "phase": phase,
            "original_status": status or "unknown",
            "original_reason": original_reason,
            "controller_stop_reason": stop_reason,
            "error_type": error_type,
            "error": error,
        }
        default_judgement = {
            "evidence_status": "sufficient",
            "verdict": "valid",
            "confidence": 0.0,
            "reason": (
                "No legal invalid finding survived the bounded recovery "
                "path; the episode therefore defaults to valid while "
                "retaining explicit ambiguity audit."
            ),
            "missing_evidence": [],
            "defects": [],
            "evidence_request": None,
            "evidence_ambiguous": True,
            "forced_binary": True,
            "defaulted": True,
            "decision_source": "default_valid_after_non_hard_failure",
            "recovery_failure": deepcopy(failure),
        }
        record.update(
            status="evaluated",
            score=1.0,
            reason=None,
            judgement=default_judgement,
            terminal_state=TERMINAL_EVALUATED_DEGRADED,
            terminal_decision={
                "forced_binary": True,
                "defaulted": True,
                "evidence_ambiguous": True,
                "decision_source": (
                    "default_valid_after_non_hard_failure"
                ),
                "failure": deepcopy(failure),
            },
            degradation_audit={
                "phase": phase,
                "controller_stop_reason": stop_reason,
                "forced_binary": True,
                "defaulted": True,
                "evidence_ambiguous": True,
            },
        )
        record.pop("infrastructure_failure", None)
        return record

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


def scope_was_defaulted(record: Any) -> bool:
    """Return whether a scope used the non-hard terminal fallback."""

    if not isinstance(record, dict):
        return False
    terminal = record.get("terminal_decision")
    if isinstance(terminal, dict) and terminal.get("defaulted") is True:
        return True
    judgement = record.get("judgement")
    return bool(
        isinstance(judgement, dict)
        and judgement.get("defaulted") is True
    )


def _is_hard_failure(
    record: dict[str, Any],
    *,
    error_type: str | None,
    reason: str | None,
    stop_reason: str | None,
) -> bool:
    """Classify only transport/input/config/program failures as hard.

    Schema, bounded-planning, evidence-selection, and ordinary evidence
    insufficiency failures are recoverable at the Judge-episode boundary.
    A schema-repair wrapper remains hard when its audit shows that the retry
    itself failed in transport.
    """

    normalized_type = str(error_type or "").replace("_", "").lower()
    normalized_reason = str(reason or "").lower()
    normalized_stop = str(stop_reason or "").strip().lower()
    if _stop_reason_is_hard(normalized_stop):
        return True
    if normalized_stop and normalized_stop not in _SUCCESSFUL_STOP_REASONS:
        known_recoverable = bool(
            normalized_stop in _RECOVERABLE_WITH_RETAINED_STOP_REASONS
            or normalized_stop.endswith("_forced_choice")
            or normalized_stop.endswith("_screen_deferred")
            or normalized_stop.endswith("_stage_rounds_exhausted")
        )
        if not known_recoverable:
            # Unknown Controller states are contract drift or unexpected
            # program states, not scientific uncertainty.
            return True
        if not _has_retained_gate_ready_evidence(record):
            return True
    if any(
        token.replace("_", "") in normalized_type
        for token in _HARD_ERROR_TOKENS
    ):
        return True
    if any(token in normalized_reason for token in _HARD_REASON_TOKENS):
        return True
    if (
        "evidence_paths" in record
        and not record.get("evidence_paths")
        and str(record.get("status") or "").lower() != "evaluated"
    ):
        # A retained packet lets the Judge make a forced choice after camera
        # acquisition fails.  With no decodable packet at all there is no
        # scientific input to project, so the failure remains infrastructure.
        return True
    evidence_resolution = record.get("evidence_resolution")
    if (
        isinstance(evidence_resolution, dict)
        and evidence_resolution.get("scope_satisfied") is False
        and not _uses_retained_global_forced_final(record)
    ):
        # A global overview cannot be relabelled as the required group/local
        # packet merely to avoid an infrastructure result.
        return True
    schema_audit = _schema_audit(record)
    if any(
        isinstance(attempt, dict)
        and str(attempt.get("failure_kind") or "").lower() == "transport"
        for attempt in schema_audit.get("attempts") or []
    ):
        return True
    if normalized_type in _RECOVERABLE_ERROR_TYPES:
        return False
    # Unexpected program failures must remain visible rather than being
    # converted into scientific validity.
    if normalized_type:
        return True
    return False


def _uses_retained_global_forced_final(record: dict[str, Any]) -> bool:
    """Recognize the explicit target-local global-anchor fallback.

    This does not relabel target-local evidence as satisfied. It permits an
    already retained global anchor to support an audited forced-final Judge;
    the resulting binary decision is score-grounded under the narrow retained-
    visual forced-choice policy.
    """

    if record.get("retained_global_forced_final") is not True:
        return False
    resolution = record.get("evidence_resolution")
    coverage = record.get("evidence_coverage")
    return bool(
        isinstance(resolution, dict)
        and resolution.get("global_anchor_satisfied") is True
        and resolution.get("local_scope_satisfied") is False
        and isinstance(coverage, dict)
        and coverage.get("grounded") is True
        and coverage.get("grounding_policy")
        == "real_judge_forced_choice_with_retained_visual_v1"
        and record.get("evidence_paths")
    )


def _stop_reason_is_hard(stop_reason: str | None) -> bool:
    normalized = str(stop_reason or "").strip().lower()
    return bool(
        normalized in _HARD_STOP_REASONS
        or any(
            normalized.startswith(reason + "_")
            for reason in _HARD_STOP_REASONS
        )
    )


def recoverable_validation_failure(exc: Exception) -> bool:
    """Isolate only explicit validation failures, never transport/program errors."""

    schema_audit = getattr(exc, "schema_audit", None)
    schema_audit = schema_audit if isinstance(schema_audit, dict) else {}
    if any(
        isinstance(attempt, dict)
        and str(attempt.get("failure_kind") or "").lower() == "transport"
        for attempt in schema_audit.get("attempts") or []
    ):
        return False
    if type(exc) is ValueError:
        return True
    return type(exc).__name__.replace("_", "").lower() == (
        "responseschemarepairerror"
    )


def _has_retained_gate_ready_evidence(record: dict[str, Any]) -> bool:
    """Require non-empty final evidence that an earlier gate accepted."""

    audit = record.get("camera_control_audit")
    audit = audit if isinstance(audit, dict) else {}
    nested = audit.get("audit")
    if isinstance(nested, dict):
        audit = nested
    final_refs = {
        str(item)
        for item in audit.get("images_used") or []
        if str(item).strip()
    }
    if not final_refs:
        final_refs = {
            str(item)
            for item in record.get("evidence_paths") or []
            if str(item).strip()
        }
    if not final_refs:
        return False
    trace = audit.get("trace")
    if not isinstance(trace, list):
        return False
    for event in trace:
        if not isinstance(event, dict) or event.get("stage") != "evidence_gate":
            continue
        result = event.get("result")
        if not isinstance(result, dict) or result.get("ready") is not True:
            continue
        gate_refs = {
            str(item)
            for item in event.get("images_used") or []
            if str(item).strip()
        }
        if gate_refs & final_refs:
            return True
    return False


def _schema_audit(record: dict[str, Any]) -> dict[str, Any]:
    for candidate in (
        record.get("response_schema_audit"),
        (record.get("judgement") or {}).get("response_schema_audit")
        if isinstance(record.get("judgement"), dict)
        else None,
    ):
        if isinstance(candidate, dict):
            return candidate
    return {}


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
