from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Callable

from benchmark.models import parse_json_object
from benchmark.visual_judge.contracts import ResponseSchemaRepairError


_RAW_RESPONSE_LIMIT = 20_000
_BINARY_SCHEMA_REPAIR_PROMPT = """Your previous response violated the required
JSON response schema. Use exactly the same visual evidence and preserve the same
semantic decision and explanation. Correct only the JSON structure. Return one
object with status valid, invalid, or need_more_evidence; confidence in [0,1];
a non-empty reason; defects exactly []; and evidence_request null for valid or
invalid, or the required structured evidence_request for need_more_evidence.
Return JSON only."""

_CANONICAL_SCHEMA_REPAIR_PROMPT = """Your previous response violated the
canonical metric response contract. Use exactly the same visual evidence and
structured context. This is contract reconciliation, not a second
adjudication. Preserve coherent typed check conclusions, defect target IDs,
evidence-request targets, and substantive explanations. If the response
envelope contradicts its explicit typed rows, reconcile only the envelope. If
any required typed row remains unresolved, use evidence_status=insufficient,
verdict=ambiguous, and defects=[]; keep already-observed invalid conclusions in
their typed rows for the next evidence round. Only after every required row is
resolved does a typed invalid row require verdict=invalid plus its explicit
defect. verdict=valid requires every required row to resolve without a defect;
insufficient evidence requires verdict=ambiguous;
sufficient evidence must clear missing_evidence and evidence_request. Judge only the requested metric
and use only the allowed defect scopes, target IDs,
and Camera DSL observation tokens from the previous user message. Do not
relabel an out-of-scope issue or invent a new defect. Return one object with
evidence_status sufficient or insufficient; verdict valid, invalid, or
ambiguous; confidence in [0,1]; a non-empty reason;
missing_evidence as a list; defects as a list; and evidence_request null unless
evidence_status is insufficient. If the original request requires
functional_check_results or placement_check_results, include every exact
required check ID once and preserve its target/subject/context IDs, observation
status, and conclusion. When all Functional checks are resolved, every invalid
required check must appear in exactly one defect.check_refs list; while any row
is unresolved, final defects must remain empty. One physical defect may
reference multiple invalid checks. Preserve the semantics of judge-originated Placement
checks, but assign unique proposal IDs and canonical typed check IDs when the
original identifiers collide. Preserve exact Function ownership references and
required causal attribution fields for an invalid clearance check. Invalid
requires at least one explicit
in-scope defect. If the original decision cannot be represented without
changing typed conclusions or defect identity, the caller will fail closed.
Return JSON only."""

_FORCED_CHOICE_CANONICAL_SCHEMA_REPAIR_PROMPT = """Your previous response
violated the terminal evidence-acquisition response contract. No more evidence can
be acquired. Use exactly the same visual evidence and structured context, then
choose the more defensible binary conclusion. Return one JSON object with
evidence_status="sufficient"; verdict exactly "valid" or "invalid"; confidence
in [0,1]; a non-empty reason; missing_evidence=[]; and
evidence_request=null. verdict="valid" requires defects=[].
If functional_check_results or placement_check_results are required, include
every exact check ID once.
Every invalid required Functional check must appear in exactly one
defect.check_refs list; one physical defect may reference multiple checks.
Use inferred_under_budget when the remaining views support only the required
terminal choice; do not omit a check. An invalid clearance check must retain
affected, causal, and scoring object IDs plus cause_kind.
verdict="invalid" requires one or more explicit in-scope defects using only the
allowed scopes and target IDs. "ambiguous", "insufficient", and any request for
more evidence are forbidden. Express residual uncertainty through confidence
and reason. Return JSON only."""


def repair_binary_response_schema_once(
    *,
    model: Any,
    messages: list[dict[str, Any]],
    response_format_json: bool,
    call_type: str,
    judge_label: str,
    validator: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate once, then permit one same-evidence schema-only repair."""

    return _repair_response_schema_once(
        model=model,
        messages=messages,
        response_format_json=response_format_json,
        call_type=call_type,
        judge_label=judge_label,
        validator=validator,
        repair_prompt=_BINARY_SCHEMA_REPAIR_PROMPT,
        policy="single_schema_repair_retry_v1",
        semantic_signature=_binary_semantic_signature,
        semantic_restore=_restore_binary_natural_language,
    )


def repair_canonical_response_schema_once(
    *,
    model: Any,
    messages: list[dict[str, Any]],
    response_format_json: bool,
    call_type: str,
    judge_label: str,
    validator: Callable[[dict[str, Any]], dict[str, Any]],
    force_binary_choice: bool = False,
    allowed_scopes: tuple[str, ...] = (),
    allowed_target_ids: tuple[str, ...] = (),
    allowed_missing_observations: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Permit one same-evidence repair for a canonical metric response."""

    return _repair_response_schema_once(
        model=model,
        messages=messages,
        response_format_json=response_format_json,
        call_type=call_type,
        judge_label=judge_label,
        validator=validator,
        repair_prompt=(
            _FORCED_CHOICE_CANONICAL_SCHEMA_REPAIR_PROMPT
            if force_binary_choice
            else _canonical_schema_repair_prompt(
                allowed_scopes=allowed_scopes,
                allowed_target_ids=allowed_target_ids,
                allowed_missing_observations=(
                    allowed_missing_observations
                ),
            )
        ),
        policy=(
            "single_forced_choice_decision_retry_v1"
            if force_binary_choice
            else "single_canonical_schema_repair_retry_v1"
        ),
        semantic_signature=(
            None
            if force_binary_choice
            else lambda value: _canonical_semantic_signature(
                value,
                allowed_scopes=allowed_scopes,
            )
        ),
        semantic_restore=(
            None
            if force_binary_choice
            else _restore_canonical_natural_language
        ),
    )


def _repair_response_schema_once(
    *,
    model: Any,
    messages: list[dict[str, Any]],
    response_format_json: bool,
    call_type: str,
    judge_label: str,
    validator: Callable[[dict[str, Any]], dict[str, Any]],
    repair_prompt: str,
    policy: str,
    semantic_signature: (
        Callable[[dict[str, Any]], dict[str, Any]] | None
    ),
    semantic_restore: (
        Callable[
            [dict[str, Any], dict[str, Any]],
            tuple[dict[str, Any], tuple[str, ...]],
        ]
        | None
    ),
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = model.chat_messages(
        messages,
        response_format_json=response_format_json,
        call_type=call_type,
    )
    first_metadata = dict(model.last_request_metadata)
    initial_value: dict[str, Any] | None = None
    locked_semantics: dict[str, Any] = {}
    try:
        initial_value = parse_json_object(raw)
        if semantic_signature is not None:
            locked_semantics = semantic_signature(initial_value)
        result = validator(initial_value)
    except (TypeError, ValueError, KeyError) as first_error:
        first_attempt = {
            "attempt": 1,
            "call_type": call_type,
            "raw_response": _bounded_raw_response(raw),
            "validation_error_type": type(first_error).__name__,
            "validation_error": str(first_error),
            "request_metadata": first_metadata,
        }
    else:
        return result, {
            "policy": policy,
            "attempt_count": 1,
            "repair_retry_count": 0,
            "recovered": False,
            "attempts": [
                {
                    "attempt": 1,
                    "call_type": call_type,
                    "validation_error": None,
                }
            ],
        }

    repair_call_type = f"{call_type}.schema_repair"
    repair_messages = [
        *deepcopy(messages),
        {"role": "assistant", "content": raw},
        {"role": "user", "content": repair_prompt},
    ]
    try:
        repaired_raw = model.chat_messages(
            repair_messages,
            response_format_json=response_format_json,
            call_type=repair_call_type,
        )
    except Exception as repair_transport_error:
        audit = {
            "policy": policy,
            "attempt_count": 2,
            "repair_retry_count": 1,
            "recovered": False,
            "attempts": [
                first_attempt,
                {
                    "attempt": 2,
                    "call_type": repair_call_type,
                    "raw_response": None,
                    "validation_error_type": type(
                        repair_transport_error
                    ).__name__,
                    "validation_error": str(repair_transport_error),
                    "failure_kind": "transport",
                },
            ],
        }
        raise ResponseSchemaRepairError(
            f"{judge_label} schema repair request failed: "
            f"{repair_transport_error}",
            schema_audit=audit,
        ) from repair_transport_error
    second_metadata = dict(model.last_request_metadata)
    restored_fields: tuple[str, ...] = ()
    try:
        repaired_value = parse_json_object(repaired_raw)
        if semantic_signature is not None and locked_semantics:
            _require_semantic_preservation(
                before=locked_semantics,
                after=semantic_signature(repaired_value),
            )
        if semantic_restore is not None and initial_value is not None:
            repaired_value, restored_fields = semantic_restore(
                initial_value,
                repaired_value,
            )
        repaired = validator(repaired_value)
    except (TypeError, ValueError, KeyError) as second_error:
        audit = {
            "policy": policy,
            "attempt_count": 2,
            "repair_retry_count": 1,
            "recovered": False,
            "attempts": [
                first_attempt,
                {
                    "attempt": 2,
                    "call_type": repair_call_type,
                    "raw_response": _bounded_raw_response(repaired_raw),
                    "validation_error_type": type(second_error).__name__,
                    "validation_error": str(second_error),
                    "request_metadata": second_metadata,
                },
            ],
        }
        raise ResponseSchemaRepairError(
            f"{judge_label} response schema remained invalid after one "
            f"repair retry: {second_error}",
            schema_audit=audit,
        ) from second_error
    return repaired, {
        "policy": policy,
        "attempt_count": 2,
        "repair_retry_count": 1,
        "recovered": True,
        "semantic_preservation": {
            "enforced": semantic_signature is not None,
            "locked_fields": deepcopy(locked_semantics),
            "changed": False,
            "restored_natural_language_fields": list(restored_fields),
        },
        "attempts": [
            first_attempt,
            {
                "attempt": 2,
                "call_type": repair_call_type,
                "raw_response": _bounded_raw_response(repaired_raw),
                "validation_error": None,
                "request_metadata": second_metadata,
            },
        ],
    }


def _canonical_semantic_signature(
    value: dict[str, Any],
    *,
    allowed_scopes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Capture structured decisions that a schema-only retry may not change.

    Free-form explanations are intentionally excluded.  A repair model may
    paraphrase them while correcting JSON, so the caller restores the original
    explanation text before returning the repaired response.
    """

    signature: dict[str, Any] = {}
    functional_rows_value = value.get("functional_check_results")
    functional_clearance_check_ids = _functional_clearance_check_ids(
        functional_rows_value
    )
    evidence_status = value.get("evidence_status")
    functional_pending = _functional_rows_have_unresolved(
        functional_rows_value
    )
    semantic_invalid = bool(
        not functional_pending
        and _has_explicit_typed_invalid_defect(value)
    )
    missing_evidence = value.get("missing_evidence")
    evidence_request = value.get("evidence_request")
    evidence_envelope_conflict = bool(
        (
            evidence_status == "insufficient"
            and semantic_invalid
        )
        or (functional_pending and evidence_status != "insufficient")
        or (
            evidence_status == "sufficient"
            and (
                bool(missing_evidence)
                or isinstance(evidence_request, dict)
            )
        )
    )
    if (
        evidence_status in {"sufficient", "insufficient"}
        and not evidence_envelope_conflict
    ):
        signature["evidence_status"] = evidence_status
    verdict = value.get("verdict")
    verdict_envelope_conflict = bool(
        (semantic_invalid and verdict != "invalid")
        or (functional_pending and verdict != "ambiguous")
    )
    if (
        verdict in {"valid", "invalid", "ambiguous"}
        and not verdict_envelope_conflict
    ):
        signature["verdict"] = verdict
    if (
        verdict == "invalid" and not functional_pending
    ) or semantic_invalid:
        defects = value.get("defects")
        if isinstance(defects, list):
            signature["defect_count"] = len(defects)
            target_sets = _defect_target_sets(defects)
            if len(target_sets) == len(defects):
                signature["defect_target_sets"] = target_sets
            allowed = set(allowed_scopes)
            scoped_claims = _defect_scopes_and_targets(
                defects,
                allowed_scopes=allowed,
            )
            if len(scoped_claims) == len(defects):
                signature["defect_scopes_and_targets"] = scoped_claims
            linked_claims = _defect_check_ref_claims(defects)
            if linked_claims:
                signature["defect_check_refs"] = linked_claims
            scoring_semantics = []
            for defect in defects:
                if not isinstance(defect, dict):
                    continue
                derived_scoring = bool(
                    set(_normalized_text_set(defect.get("check_refs")))
                    & functional_clearance_check_ids
                )
                scoring_semantics.append(
                    (
                        str(defect.get("category") or ""),
                        str(defect.get("severity") or ""),
                        (
                            ()
                            if derived_scoring
                            else _normalized_text_set(
                                defect.get("scoring_target_ids")
                            )
                        ),
                        str(defect.get("attribution_mode") or ""),
                    )
                )
            if (
                len(scoring_semantics) == len(defects)
                and any(any(item) for item in scoring_semantics)
            ):
                signature["defect_scoring_semantics"] = sorted(
                    scoring_semantics
                )
    if (
        (verdict == "ambiguous" or evidence_status == "insufficient")
        and not semantic_invalid
    ):
        if isinstance(evidence_request, dict):
            target_ids = _normalized_text_set(
                evidence_request.get("target_ids")
            )
            observations = _normalized_text_set(
                evidence_request.get("missing_observations")
            )
            if target_ids:
                signature["evidence_request_target_ids"] = target_ids
            if observations:
                signature["missing_observations"] = observations
    functional_rows = value.get("functional_check_results")
    if isinstance(functional_rows, list) and all(
        isinstance(item, dict) for item in functional_rows
    ):
        signature["functional_check_results"] = sorted(
            (
                str(item.get("check_id") or ""),
                tuple(
                    sorted(
                        str(target_id)
                        for target_id in item.get("target_ids") or []
                    )
                ),
                str(item.get("observation_status") or ""),
                str(item.get("conclusion") or ""),
                tuple(
                    sorted(
                        str(target_id)
                        for target_id in item.get(
                            "affected_object_ids"
                        )
                        or []
                    )
                ),
                str(item.get("cause_kind") or ""),
                tuple(
                    sorted(
                        str(target_id)
                        for target_id in item.get(
                            "causal_object_ids"
                        )
                        or []
                    )
                ),
                (
                    ()
                    if str(item.get("check_id") or "")
                    in functional_clearance_check_ids
                    else _normalized_text_set(
                        item.get("scoring_target_ids")
                    )
                ),
            )
            for item in functional_rows
        )
    placement_rows = value.get("placement_check_results")
    if isinstance(placement_rows, list) and all(
        isinstance(item, dict) for item in placement_rows
    ):
        signature["placement_check_results"] = sorted(
            (
                str(item.get("check_id") or ""),
                str(item.get("subject_id") or ""),
                tuple(
                    sorted(
                        str(context_id)
                        for context_id in item.get("context_ids") or []
                    )
                ),
                str(item.get("observation_status") or ""),
                str(item.get("conclusion") or ""),
                str(item.get("function_event_ref") or ""),
                bool(item.get("same_physical_event") is True),
            )
            for item in placement_rows
        )
    return signature


def _functional_rows_have_unresolved(value: Any) -> bool:
    """Return whether the typed Functional batch still needs evidence.

    Invalid observations already made in the same batch remain auditable rows,
    but they are not final defect claims until every required row is resolved.
    """

    return bool(
        isinstance(value, list)
        and any(
            isinstance(row, dict)
            and row.get("conclusion") == "unresolved"
            for row in value
        )
    )


def _has_explicit_typed_invalid_defect(value: dict[str, Any]) -> bool:
    """Return whether typed rows already establish one explicit defect.

    This does not infer a decision from prose.  It only recognizes a schema
    envelope that contradicts an already emitted invalid typed row plus an
    explicit defect record, allowing the repair retry to reconcile the outer
    verdict without re-adjudicating the scene.
    """

    defects = value.get("defects")
    if not isinstance(defects, list) or not any(
        isinstance(item, dict) for item in defects
    ):
        return False
    return any(
        isinstance(row, dict) and row.get("conclusion") == "invalid"
        for field in (
            "functional_check_results",
            "placement_check_results",
        )
        for row in (value.get(field) or [])
    )


def _binary_semantic_signature(
    value: dict[str, Any],
) -> dict[str, Any]:
    signature: dict[str, Any] = {}
    decision = value.get("status")
    if decision not in {"valid", "invalid", "need_more_evidence"}:
        decision = value.get("verdict")
    if decision in {"valid", "invalid", "need_more_evidence"}:
        signature["decision"] = decision
    if decision == "need_more_evidence":
        evidence_request = value.get("evidence_request")
        if isinstance(evidence_request, dict):
            target_ids = _normalized_text_set(
                evidence_request.get("target_ids")
            )
            observations = _normalized_text_set(
                evidence_request.get("missing_observations")
            )
            if target_ids:
                signature["evidence_request_target_ids"] = target_ids
            if observations:
                signature["missing_observations"] = observations
    return signature


def _require_semantic_preservation(
    *,
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    changed: dict[str, Any] = {}
    for key, expected in before.items():
        actual = after.get(key)
        if key in {
            "functional_check_results",
            "placement_check_results",
        }:
            # A schema retry may add omitted required rows, but every
            # structured check claim already made by the first response must
            # remain byte-for-byte equivalent at the semantic tuple level.
            if not isinstance(actual, list) or any(
                row not in actual for row in expected
            ):
                changed[key] = {
                    "before": deepcopy(expected),
                    "after": deepcopy(actual),
                }
            continue
        if actual != expected:
            changed[key] = {
                "before": deepcopy(expected),
                "after": deepcopy(actual),
            }
    if changed:
        raise ValueError(
            "response schema repair changed locked semantic fields: "
            + ", ".join(sorted(changed))
        )


def _defect_target_sets(
    value: Any,
) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        sorted(
            target_ids
            for defect in value
            if isinstance(defect, dict)
            if (
                target_ids := _normalized_text_set(
                    defect.get("target_ids")
                )
            )
        )
    )


def _defect_scopes_and_targets(
    value: Any,
    *,
    allowed_scopes: set[str],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not isinstance(value, list) or not allowed_scopes:
        return ()
    claims: list[tuple[str, tuple[str, ...]]] = []
    for defect in value:
        if not isinstance(defect, dict):
            continue
        scope = defect.get("scope")
        target_ids = _normalized_text_set(defect.get("target_ids"))
        if scope not in allowed_scopes or not target_ids:
            continue
        claims.append((str(scope), target_ids))
    return tuple(sorted(claims))


def _defect_check_ref_claims(
    value: Any,
) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    """Lock explicit typed-check linkage while allowing repair to add it."""

    if not isinstance(value, list):
        return ()
    claims: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for defect in value:
        if not isinstance(defect, dict):
            continue
        check_refs = _normalized_text_set(defect.get("check_refs"))
        target_ids = _normalized_text_set(defect.get("target_ids"))
        if check_refs and target_ids:
            claims.append((target_ids, check_refs))
    return tuple(sorted(claims))


def _functional_clearance_check_ids(value: Any) -> set[str]:
    """Identify rows whose scoring owner is deterministic bookkeeping."""

    if not isinstance(value, list):
        return set()
    return {
        str(row.get("check_id") or "")
        for row in value
        if isinstance(row, dict)
        and str(row.get("check_id") or "")
        and (
            row.get("cause_kind") in {"external_object", "self_layout"}
            or any(
                field in row
                for field in (
                    "affected_object_ids",
                    "causal_object_ids",
                )
            )
        )
    }


def _restore_canonical_natural_language(
    initial: dict[str, Any],
    repaired: dict[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Restore original prose after structured schema repair.

    The repair response supplies corrected schema tokens, confidence, and
    container shape.  It is not allowed to replace the original adjudication
    explanation or defect relation with a new semantic claim.
    """

    restored = deepcopy(repaired)
    restored_fields: list[str] = []
    _restore_text_field(
        initial,
        restored,
        "reason",
        restored_fields=restored_fields,
    )

    initial_functional_pending = _functional_rows_have_unresolved(
        initial.get("functional_check_results")
    )
    if not initial_functional_pending and (
        initial.get("verdict") == "invalid"
        or _has_explicit_typed_invalid_defect(initial)
    ):
        initial_defects = initial.get("defects")
        repaired_defects = restored.get("defects")
        if (
            not isinstance(initial_defects, list)
            or not isinstance(repaired_defects, list)
            or len(initial_defects) != len(repaired_defects)
        ):
            raise ValueError(
                "schema repair cannot preserve invalid-defect identity"
            )
        remaining = list(repaired_defects)
        preserved_defects: list[dict[str, Any]] = []
        for index, original in enumerate(initial_defects):
            if not isinstance(original, dict):
                raise ValueError(
                    "schema repair cannot preserve malformed defect semantics"
                )
            original_targets = _normalized_text_set(
                original.get("target_ids")
            )
            original_relation = _normalized_explanation(
                original.get("relation")
            )
            original_reason = _normalized_explanation(
                original.get("reason")
            )
            if (
                not original_targets
                or not original_relation
                or not original_reason
            ):
                raise ValueError(
                    "schema repair cannot invent missing defect semantics"
                )
            original_check_id = str(original.get("check_id") or "")
            original_check_refs = _normalized_text_set(
                original.get("check_refs")
            )
            match_index = next(
                (
                    candidate_index
                    for candidate_index, candidate in enumerate(remaining)
                    if isinstance(candidate, dict)
                    and _normalized_text_set(candidate.get("target_ids"))
                    == original_targets
                    and (
                        not original_check_id
                        or str(candidate.get("check_id") or "")
                        == original_check_id
                    )
                    and (
                        not original_check_refs
                        or _normalized_text_set(
                            candidate.get("check_refs")
                        )
                        == original_check_refs
                    )
                ),
                None,
            )
            if match_index is None:
                raise ValueError(
                    "schema repair changed defect target identity"
                )
            candidate = deepcopy(remaining.pop(match_index))
            is_placement_response = isinstance(
                initial.get("placement_check_results"),
                list,
            ) or isinstance(
                initial.get("judge_originated_placement_results"),
                list,
            )
            fields_to_restore = (
                ("target_ids", "reason")
                if is_placement_response
                else ("target_ids", "relation", "reason")
            )
            for field_name in fields_to_restore:
                original_value = deepcopy(original.get(field_name))
                if candidate.get(field_name) != original_value:
                    restored_fields.append(
                        f"defects[{index}].{field_name}"
                    )
                candidate[field_name] = original_value
            semantic_fields = [
                "category",
                "severity",
                "attribution_mode",
                "ownership_event_id",
            ]
            clearance_check_ids = _functional_clearance_check_ids(
                initial.get("functional_check_results")
            )
            is_derived_clearance_scoring = bool(
                set(original_check_refs) & clearance_check_ids
            )
            if not is_derived_clearance_scoring:
                semantic_fields.append("scoring_target_ids")
            for field_name in semantic_fields:
                if field_name not in original:
                    continue
                original_value = deepcopy(original[field_name])
                if candidate.get(field_name) != original_value:
                    restored_fields.append(
                        f"defects[{index}].{field_name}"
                    )
                candidate[field_name] = original_value
            preserved_defects.append(candidate)
        if remaining:
            raise ValueError(
                "schema repair added unapproved defect claims"
            )
        restored["defects"] = preserved_defects

    evidence_request = initial.get("evidence_request")
    repaired_request = restored.get("evidence_request")
    if (
        isinstance(evidence_request, dict)
        and isinstance(repaired_request, dict)
    ):
        _restore_text_field(
            evidence_request,
            repaired_request,
            "view_goal",
            restored_fields=restored_fields,
            field_path="evidence_request.view_goal",
        )

    return restored, tuple(dict.fromkeys(restored_fields))


def _restore_binary_natural_language(
    initial: dict[str, Any],
    repaired: dict[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    restored = deepcopy(repaired)
    restored_fields: list[str] = []
    _restore_text_field(
        initial,
        restored,
        "reason",
        restored_fields=restored_fields,
    )
    evidence_request = initial.get("evidence_request")
    repaired_request = restored.get("evidence_request")
    if (
        isinstance(evidence_request, dict)
        and isinstance(repaired_request, dict)
    ):
        _restore_text_field(
            evidence_request,
            repaired_request,
            "view_goal",
            restored_fields=restored_fields,
            field_path="evidence_request.view_goal",
        )
    return restored, tuple(dict.fromkeys(restored_fields))


def _restore_text_field(
    source: dict[str, Any],
    destination: dict[str, Any],
    field_name: str,
    *,
    restored_fields: list[str],
    field_path: str | None = None,
) -> None:
    original = source.get(field_name)
    if not isinstance(original, str) or not original.strip():
        return
    if destination.get(field_name) != original:
        restored_fields.append(field_path or field_name)
    destination[field_name] = original


def _canonical_schema_repair_prompt(
    *,
    allowed_scopes: tuple[str, ...],
    allowed_target_ids: tuple[str, ...],
    allowed_missing_observations: tuple[str, ...],
) -> str:
    constraints = [
        _CANONICAL_SCHEMA_REPAIR_PROMPT,
        "Use these exact enumerated values; do not invent aliases:",
        "allowed defect scopes: "
        + json.dumps(list(allowed_scopes), ensure_ascii=False),
        "allowed target IDs: "
        + json.dumps(list(allowed_target_ids), ensure_ascii=False),
        "allowed missing-observation tokens: "
        + json.dumps(
            list(allowed_missing_observations),
            ensure_ascii=False,
        ),
    ]
    return "\n".join(constraints)


def _normalized_explanation(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _normalized_text_set(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(
        sorted(
            {
                str(item).strip()
                for item in value
                if isinstance(item, (str, int))
                and str(item).strip()
            }
        )
    )


def _bounded_raw_response(value: Any) -> str:
    raw = str(value)
    if len(raw) <= _RAW_RESPONSE_LIMIT:
        return raw
    return raw[:_RAW_RESPONSE_LIMIT] + "\n...[truncated]"
