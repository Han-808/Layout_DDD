"""Typed obligations between placement discovery and metric judgement.

Placement discovery is a routing prior only.  This module turns accepted
discovery records into stable checks, validates exact Judge acknowledgements,
and merges their audit-only lifecycle.  It never decides a metric verdict.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Iterable


PLACEMENT_CHECK_LEDGER_VERSION = "placement_check_ledger_v1"
PLACEMENT_CHECK_RESULT_VERSION = "placement_check_results_v2"
RESIDUAL_GROUP_OBSERVATION_VERSION = (
    "placement_residual_group_observations_v1"
)
PLACEMENT_CHECK_TYPES = frozenset(
    {
        "support_and_height",
        "scene_zone",
        "contextual_anchor",
    }
)
PLACEMENT_CHECK_CONCLUSIONS = frozenset(
    {
        "valid",
        "invalid",
        "excluded_function_owned",
        "unresolved",
    }
)
_REQUIRED_OBSERVATIONS = {
    "support_and_height": (
        "target_visible",
        "contact_surface_visible",
        "group_context_visible",
    ),
    "scene_zone": (
        "target_visible",
        "global_context_preserved",
        "architecture_plane_visible",
    ),
    "contextual_anchor": (
        "joint_visibility",
        "group_context_visible",
        "global_context_preserved",
    ),
}
_OBSERVATION_STATUSES = {
    "observed",
    "inferred_under_budget",
    "missing",
}
_RESIDUAL_GROUP_EVIDENCE = {
    "sufficient",
    "partial_but_usable",
    "insufficient",
}
_RESIDUAL_GROUP_CANDIDATES = {
    "none",
    "scene_zone",
    "contextual_anchor",
}


def validate_residual_group_global_observations(
    value: dict[str, Any],
    *,
    groups: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Require one non-authoritative global-position observation per group."""

    expected = _residual_group_roster(groups)
    rows = value.get("group_global_observations")
    if not isinstance(rows, list):
        raise ValueError(
            "residual Placement requires group_global_observations"
        )
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(
                "group_global_observations must contain JSON objects"
            )
        group_id = str(row.get("group_id") or "").strip()
        if not group_id or group_id in by_id:
            raise ValueError(
                "group_global_observations require unique non-empty group_id"
            )
        if group_id not in expected:
            raise ValueError(
                "group_global_observations contain unknown group_id "
                f"{group_id!r} at index {index}"
            )
        _validate_residual_group_observation_row(
            row,
            expected_object_ids=expected[group_id],
            known_group_ids=set(expected),
        )
        by_id[group_id] = row
    if set(by_id) != set(expected):
        raise ValueError(
            "group_global_observations must cover every group exactly once: "
            f"expected {list(expected)}, got {list(by_id)}"
        )
    ungrounded = [
        group_id
        for group_id, row in by_id.items()
        if row.get("evidence_sufficiency") == "insufficient"
    ]
    grounded = len(expected) - len(ungrounded)
    return {
        "schema_version": RESIDUAL_GROUP_OBSERVATION_VERSION,
        "expected_group_count": len(expected),
        "grounded_group_count": grounded,
        "fraction": grounded / len(expected) if expected else 1.0,
        "complete": not ungrounded,
        "defaulted_group_ids": [],
        "ungrounded_group_ids": ungrounded,
        "decision_authority": "none",
    }


def salvage_residual_group_global_observations(
    value: dict[str, Any],
    *,
    groups: list[dict[str, Any]] | None,
    fallback_value: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Retain valid group rows and default only malformed/missing rows."""

    result = deepcopy(value)
    expected = _residual_group_roster(groups)
    sources = [
        fallback_value if isinstance(fallback_value, dict) else {},
        value,
    ]
    retained: list[dict[str, Any]] = []
    defaulted: list[str] = []
    source_by_group: dict[str, str] = {}
    for group_id, object_ids in expected.items():
        accepted = None
        for source_index, source in enumerate(sources):
            candidates = [
                deepcopy(row)
                for row in source.get("group_global_observations") or []
                if isinstance(row, dict)
                and str(row.get("group_id") or "").strip() == group_id
            ]
            if len(candidates) != 1:
                continue
            try:
                _validate_residual_group_observation_row(
                    candidates[0],
                    expected_object_ids=object_ids,
                    known_group_ids=set(expected),
                )
            except (TypeError, ValueError, KeyError):
                continue
            accepted = candidates[0]
            source_by_group[group_id] = (
                "initial" if source_index == 0 else "repair"
            )
            break
        if accepted is None:
            defaulted.append(group_id)
            accepted = {
                "group_id": group_id,
                "object_ids": list(object_ids),
                "global_position_observation": (
                    "No schema-valid group observation was retained."
                ),
                "related_group_ids": [],
                "inter_group_observation": (
                    "No schema-valid cross-group observation was retained."
                ),
                "evidence_sufficiency": "insufficient",
                "residual_issue_candidate": "none",
            }
            source_by_group[group_id] = "defaulted"
        retained.append(accepted)
    grounded = len(expected) - len(defaulted)
    result["group_global_observations"] = retained
    result["group_global_observation_coverage"] = {
        "schema_version": RESIDUAL_GROUP_OBSERVATION_VERSION,
        "expected_group_count": len(expected),
        "grounded_group_count": grounded,
        "fraction": grounded / len(expected) if expected else 1.0,
        "complete": not defaulted,
        "defaulted_group_ids": defaulted,
        "ungrounded_group_ids": list(defaulted),
        "source_by_group": source_by_group,
        "decision_authority": "none",
    }
    if defaulted:
        result["evidence_ambiguous"] = True
        result["forced_binary"] = True
        result["defaulted"] = True
    return result


def _residual_group_roster(
    groups: list[dict[str, Any]] | None,
) -> dict[str, list[str]]:
    roster: dict[str, list[str]] = {}
    for group in groups or []:
        if not isinstance(group, dict):
            raise TypeError("residual Placement groups must be JSON objects")
        group_id = str(group.get("group_id") or "").strip()
        object_ids = group.get("object_ids")
        if (
            not group_id
            or group_id in roster
            or not isinstance(object_ids, list)
            or not object_ids
            or any(
                not isinstance(item, str) or not item.strip()
                for item in object_ids
            )
            or len(object_ids) != len(set(object_ids))
        ):
            raise ValueError(
                "residual Placement requires a unique non-empty group roster"
            )
        roster[group_id] = list(object_ids)
    return roster


def _validate_residual_group_observation_row(
    row: dict[str, Any],
    *,
    expected_object_ids: list[str],
    known_group_ids: set[str],
) -> None:
    group_id = str(row.get("group_id") or "").strip()
    if row.get("object_ids") != expected_object_ids:
        raise ValueError(
            f"group {group_id!r} observation changed its object roster"
        )
    for field in (
        "global_position_observation",
        "inter_group_observation",
    ):
        if not str(row.get(field) or "").strip():
            raise ValueError(
                f"group {group_id!r} observation requires {field}"
            )
    related = row.get("related_group_ids")
    if (
        not isinstance(related, list)
        or any(
            not isinstance(item, str) or not item.strip()
            for item in related
        )
        or len(related) != len(set(related))
        or group_id in related
        or set(related) - known_group_ids
    ):
        raise ValueError(
            f"group {group_id!r} has invalid related_group_ids"
        )
    if row.get("evidence_sufficiency") not in _RESIDUAL_GROUP_EVIDENCE:
        raise ValueError(
            f"group {group_id!r} has invalid evidence_sufficiency"
        )
    if row.get("residual_issue_candidate") not in (
        _RESIDUAL_GROUP_CANDIDATES
    ):
        raise ValueError(
            f"group {group_id!r} has invalid residual_issue_candidate"
        )


def canonical_placement_check_type(value: Any) -> str:
    token = str(value or "").strip()
    if token not in PLACEMENT_CHECK_TYPES:
        raise ValueError(f"unsupported placement check type {token!r}")
    return token


def build_placement_check_ledger(
    discovery: dict[str, Any],
    *,
    groups: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Build stable checks from a validated canonical v2 discovery result."""

    if not isinstance(discovery, dict):
        raise TypeError("placement discovery must be a JSON object")
    object_ids = _discovery_object_ids(discovery)
    object_to_group, known_groups = _trusted_group_partition(
        groups,
        known_ids=set(object_ids),
    )
    pending: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}
    for index, raw in enumerate(discovery.get("candidates") or [], start=1):
        candidate = normalize_placement_candidate(
            raw,
            known_ids=set(object_ids),
        )
        subject_id = candidate["subject_id"]
        context_ids = candidate["context_ids"]
        check_type = candidate["check_type"]
        identity = (
            check_type,
            subject_id,
            tuple(sorted(context_ids)),
        )
        owner_stage, owning_group_id, group_ids = _check_owner(
            check_type=check_type,
            subject_id=subject_id,
            context_ids=context_ids,
            object_to_group=object_to_group,
            known_groups=known_groups,
        )
        record = pending.get(identity)
        source_kind = str(raw.get("check_type") or check_type)
        if record is None:
            record = {
                "check_id": placement_check_id(
                    check_type,
                    subject_id,
                    context_ids,
                ),
                "check_type": check_type,
                "subject_id": subject_id,
                "context_ids": list(context_ids),
                "target_ids": [subject_id],
                "owner_stage": owner_stage,
                "owning_group_id": owning_group_id,
                "group_ids": group_ids,
                "required_observations": list(
                    _REQUIRED_OBSERVATIONS[check_type]
                ),
                "observation_goals": [],
                "source_check_types": [],
                "source_discovery_refs": [],
                "origin": "placement_discovery",
                "lifecycle_status": "accepted",
                "obligation_lifecycle": [
                    {
                        "state": "discovered",
                        "source": "placement_discovery",
                    }
                ],
                "acquisition_status": "pending",
                "observation_complete": False,
                "judge_status": "pending",
                "judge_result_ref": None,
                "decision_authority": "none",
            }
            pending[identity] = record
        record["observation_goals"] = _stable_unique(
            [
                *record["observation_goals"],
                candidate["observation_goal"],
            ]
        )
        record["source_check_types"] = _stable_unique(
            [
                *record["source_check_types"],
                source_kind,
            ]
        )
        record["source_discovery_refs"] = _stable_unique(
            [
                *record["source_discovery_refs"],
                str(
                    raw.get("candidate_id")
                    or f"placement_candidate_{index:03d}"
                ),
            ]
        )
    checks = sorted(
        pending.values(),
        key=lambda item: (
            str(item["owner_stage"]),
            str(item.get("owning_group_id") or ""),
            str(item["check_type"]),
            str(item["subject_id"]),
            tuple(item["context_ids"]),
        ),
    )
    return {
        "schema_version": PLACEMENT_CHECK_LEDGER_VERSION,
        "checks": checks,
        "accepted_check_count": len(checks),
        "decision_authority": "none",
    }


def normalize_placement_candidate(
    value: Any,
    *,
    known_ids: set[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("placement candidate must be a JSON object")
    subject_id = str(value.get("subject_id") or "").strip()
    if not subject_id or subject_id not in known_ids:
        raise ValueError(
            "placement candidate references an unknown subject ID"
        )
    context_ids = _id_list(
        value.get("context_ids", []),
        known=known_ids,
        label="placement context_ids",
    )
    context_ids = sorted(context_ids)
    if subject_id in context_ids:
        raise ValueError(
            "placement subject cannot appear in its own context"
        )
    check_type = canonical_placement_check_type(value.get("check_type"))
    if check_type == "contextual_anchor" and not context_ids:
        raise ValueError(
            "contextual_anchor requires one or more context IDs"
        )
    goal = str(value.get("observation_goal") or "").strip()
    if not goal:
        raise ValueError(
            "placement candidate observation_goal must be non-empty"
        )
    return {
        "subject_id": subject_id,
        "context_ids": context_ids,
        "check_type": check_type,
        "observation_goal": goal[:1000],
    }


def placement_check_id(
    check_type: str,
    subject_id: str,
    context_ids: Iterable[str],
) -> str:
    payload = {
        "check_type": canonical_placement_check_type(check_type),
        "subject_id": str(subject_id),
        "context_ids": sorted(str(item) for item in context_ids),
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"placement_check_{digest}"


def canonicalize_placement_proposal_transport(
    value: Any,
    *,
    known_ids: set[str],
    source_ref: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Canonicalize only transport identity and one observed field alias."""

    if not isinstance(value, dict):
        raise ValueError("placement proposal must be a JSON object")
    normalized = deepcopy(value)
    warnings: list[dict[str, Any]] = []
    if "placement_check_type" in normalized:
        alias = normalized.get("placement_check_type")
        canonical = normalized.get("check_type")
        if canonical is not None and canonical != alias:
            raise ValueError(
                "placement_check_type conflicts with canonical check_type"
            )
        normalized["check_type"] = alias
        normalized.pop("placement_check_type", None)
        warnings.append(
            {
                "code": "placement_check_type_alias_canonicalized",
                "from": "placement_check_type",
                "to": "check_type",
            }
        )
    candidate = normalize_placement_candidate(
        normalized,
        known_ids=known_ids,
    )
    proposal_id = str(
        normalized.get("proposal_id") or source_ref or ""
    ).strip()
    if not proposal_id:
        proposal_id = "placement_proposal_" + placement_check_id(
            candidate["check_type"],
            candidate["subject_id"],
            candidate["context_ids"],
        ).removeprefix("placement_check_")
        warnings.append(
            {
                "code": "missing_proposal_id_generated",
                "proposal_id": proposal_id,
            }
        )
    normalized["proposal_id"] = proposal_id
    return normalized, warnings


def placement_global_checks(
    ledger: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    return [
        deepcopy(check)
        for check in (ledger or {}).get("checks") or []
        if check.get("owner_stage") == "scene_global"
    ]


def placement_checks_for_group(
    ledger: dict[str, Any] | None,
    group_id: str,
) -> list[dict[str, Any]]:
    return [
        deepcopy(check)
        for check in (ledger or {}).get("checks") or []
        if check.get("owner_stage") == "group_local"
        and str(check.get("owning_group_id") or "") == str(group_id)
        and check.get("judge_status") != "resolved"
    ]


def placement_target_checks(
    ledger: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return accepted checks whose subject has no trusted group owner."""

    return [
        deepcopy(check)
        for check in (ledger or {}).get("checks") or []
        if check.get("owner_stage") == "target_local"
        and check.get("judge_status") != "resolved"
    ]


def forced_group_ids_from_placement_checks(
    ledger: dict[str, Any] | None,
) -> list[str]:
    return sorted(
        {
            str(check.get("owning_group_id"))
            for check in (ledger or {}).get("checks") or []
            if check.get("owner_stage") == "group_local"
            and check.get("owning_group_id")
            and check.get("judge_status") != "resolved"
        }
    )


def placement_camera_targets_by_group(
    ledger: dict[str, Any] | None,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for check in (ledger or {}).get("checks") or []:
        if check.get("owner_stage") != "group_local":
            continue
        group_id = str(check.get("owning_group_id") or "")
        if not group_id:
            continue
        targets = result.setdefault(group_id, [])
        for object_id in [
            check.get("subject_id"),
            *(check.get("context_ids") or []),
        ]:
            normalized = str(object_id or "").strip()
            if normalized and normalized not in targets:
                targets.append(normalized)
    return result


def validate_placement_check_results(
    result: dict[str, Any],
    *,
    required_checks: list[dict[str, Any]],
    function_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Require exact rows, subject ownership, and exact same-event links.

    Placement still resolves its static-location claim independently.  An
    already-scored Functional event can remove duplicate Placement burden only
    when the row explicitly cites that stable event ID and confirms that it is
    the same physical event.  Shared object identity is never sufficient.
    """

    if not isinstance(result, dict):
        raise TypeError("placement Judge result must be a JSON object")
    expected = {
        str(check.get("check_id") or ""): check
        for check in required_checks
        if isinstance(check, dict) and check.get("check_id")
    }
    if len(expected) != len(required_checks):
        raise ValueError("required placement checks contain invalid IDs")
    rows = result.get("placement_check_results")
    if not expected and rows is None:
        rows = []
    if not isinstance(rows, list) or any(
        not isinstance(item, dict) for item in rows
    ):
        raise ValueError(
            "placement Judge response requires placement_check_results"
        )
    returned_ids = [str(item.get("check_id") or "") for item in rows]
    if (
        len(returned_ids) != len(set(returned_ids))
        or set(returned_ids) != set(expected)
    ):
        raise ValueError(
            "placement_check_results must cover every required check exactly once"
        )

    verdict = str(result.get("verdict") or "")
    defects_by_check: dict[str, list[dict[str, Any]]] = {}
    for defect in result.get("defects") or []:
        if not isinstance(defect, dict):
            raise TypeError("placement defects must contain JSON objects")
        check_id = str(defect.get("check_id") or "")
        if not check_id:
            raise ValueError(
                "every placement defect must reference a typed check"
            )
        if check_id not in expected:
            raise ValueError(
                f"placement defect references unknown check {check_id!r}"
            )
        defects_by_check.setdefault(check_id, []).append(defect)
    event_by_id = {
        str(item.get("event_id") or ""): item
        for item in function_events or []
        if isinstance(item, dict) and item.get("event_id")
    }
    resolved: list[str] = []
    unresolved: list[str] = []
    invalid: list[str] = []
    excluded: list[str] = []
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        check_id = str(row.get("check_id") or "")
        check = expected[check_id]
        _validate_exact_placement_row(row, check=check)
        status = str(row.get("observation_status") or "")
        conclusion = str(row.get("conclusion") or "")
        if status not in _OBSERVATION_STATUSES:
            raise ValueError(
                f"placement check {check_id} has invalid observation_status"
            )
        if conclusion not in PLACEMENT_CHECK_CONCLUSIONS:
            raise ValueError(
                f"placement check {check_id} has invalid conclusion"
            )
        if conclusion != "excluded_function_owned" and (
            row.get("function_event_ref") not in (None, "")
            or row.get("same_physical_event") is not None
        ):
            raise ValueError(
                f"placement check {check_id} may cite Function ownership "
                "only for exact same-event deduplication"
            )
        if not str(row.get("reason") or "").strip():
            raise ValueError(
                f"placement check {check_id} requires a non-empty reason"
            )
        if status == "missing" and conclusion != "unresolved":
            raise ValueError(
                f"missing placement check {check_id} must be unresolved"
            )
        if status != "missing" and conclusion == "unresolved":
            raise ValueError(
                f"resolved placement observation {check_id} requires a final "
                "or explicitly deduplicated conclusion"
            )
        mapped_defects = defects_by_check.get(check_id, [])
        if conclusion == "invalid":
            if verdict == "ambiguous" and not mapped_defects:
                invalid.append(check_id)
                resolved.append(check_id)
                normalized_rows.append(deepcopy(row))
                continue
            if len(mapped_defects) != 1:
                raise ValueError(
                    f"invalid placement check {check_id} requires exactly one "
                    "mapped defect"
                )
            _validate_placement_defect_for_check(
                mapped_defects[0],
                check=check,
            )
            invalid.append(check_id)
            resolved.append(check_id)
        elif conclusion == "excluded_function_owned":
            if mapped_defects:
                raise ValueError(
                    f"function-owned placement check {check_id} cannot retain "
                    "a placement defect"
                )
            event_ref = str(row.get("function_event_ref") or "").strip()
            if not event_ref or event_ref not in event_by_id:
                raise ValueError(
                    f"placement check {check_id} must reference one exact "
                    "known functional ownership event"
                )
            if row.get("same_physical_event") is not True:
                raise ValueError(
                    f"placement check {check_id} must explicitly confirm "
                    "same_physical_event before deduplication"
                )
            event_targets = {
                str(item)
                for field in (
                    "affected_object_ids",
                    "causal_object_ids",
                    "scoring_target_ids",
                    "counterpart_object_ids",
                )
                for item in event_by_id[event_ref].get(field) or []
            }
            if str(check["subject_id"]) not in event_targets:
                raise ValueError(
                    f"placement check {check_id} exact event reference has "
                    "no subject-role overlap"
                )
            excluded.append(check_id)
            resolved.append(check_id)
        elif conclusion == "valid":
            if mapped_defects:
                raise ValueError(
                    f"valid placement check {check_id} cannot retain a defect"
                )
            resolved.append(check_id)
        else:
            if mapped_defects:
                raise ValueError(
                    f"unresolved placement check {check_id} cannot retain a defect"
                )
            unresolved.append(check_id)
        normalized_row = deepcopy(row)
        normalized_row["context_ids"] = sorted(
            str(item) for item in row.get("context_ids") or []
        )
        normalized_rows.append(normalized_row)

    if verdict == "valid" and (invalid or unresolved):
        raise ValueError(
            "placement valid verdict requires every required check to resolve "
            "valid or be exactly deduplicated"
        )
    if verdict == "invalid" and not invalid:
        raise ValueError(
            "placement invalid verdict requires a resolved invalid check"
        )
    if unresolved and verdict != "ambiguous":
        raise ValueError(
            "unresolved placement checks require an ambiguous evidence-"
            "acquisition verdict; an early invalid cannot stop the loop"
        )
    pending_proposal = _pending_proposal_from_result(result)
    if verdict == "ambiguous" and (
        result.get("defects")
        or (not unresolved and pending_proposal is None)
    ):
        raise ValueError(
            "placement ambiguous verdict requires unresolved checks and cannot "
            "emit final defects"
        )
    return {
        "schema_version": PLACEMENT_CHECK_RESULT_VERSION,
        "required_check_count": len(expected),
        "resolved_check_ids": resolved,
        "unresolved_check_ids": unresolved,
        "invalid_check_ids": invalid,
        "excluded_function_owned_check_ids": excluded,
        "complete": not unresolved,
        "rows": normalized_rows,
        "decision_authority": "none",
    }


def salvage_placement_judge_response(
    value: Any,
    *,
    required_checks: list[dict[str, Any]],
    function_events: list[dict[str, Any]] | None = None,
    retain_invalid: bool = True,
    fallback_value: Any = None,
    known_ids: set[str] | None = None,
    groups: list[dict[str, Any]] | None = None,
    expected_owner_stage: str | None = None,
) -> dict[str, Any]:
    """Retain legal typed rows and default only malformed check rows valid."""

    sources = [
        (
            "initial",
            fallback_value if isinstance(fallback_value, dict) else {},
        ),
        ("repair", value if isinstance(value, dict) else {}),
    ]
    accepted_rows: list[dict[str, Any]] = []
    accepted_defects: list[dict[str, Any]] = []
    accepted_sources: dict[str, str] = {}
    defaulted_check_ids: list[str] = []
    rejected: list[dict[str, Any]] = []

    for check in required_checks:
        check_id = str(check.get("check_id") or "")
        accepted = False
        for source_name, source in sources:
            rows = source.get("placement_check_results")
            rows = rows if isinstance(rows, list) else []
            defects = source.get("defects")
            defects = defects if isinstance(defects, list) else []
            matching_rows = [
                deepcopy(row)
                for row in rows
                if isinstance(row, dict)
                and str(row.get("check_id") or "") == check_id
            ]
            matching_defects = [
                deepcopy(defect)
                for defect in defects
                if isinstance(defect, dict)
                and str(defect.get("check_id") or "") == check_id
            ]
            if len(matching_rows) != 1:
                rejected.append(
                    {
                        "source": source_name,
                        "check_id": check_id,
                        "reason": (
                            "missing_check_row"
                            if not matching_rows
                            else "duplicate_check_rows"
                        ),
                    }
                )
                continue
            row = matching_rows[0]
            conclusion = str(row.get("conclusion") or "")
            row_defects = (
                matching_defects
                if retain_invalid and conclusion == "invalid"
                else []
            )
            synthetic = {
                "verdict": "invalid" if row_defects else "valid",
                "defects": row_defects,
                "placement_check_results": [row],
            }
            try:
                validate_placement_check_results(
                    synthetic,
                    required_checks=[check],
                    function_events=function_events,
                )
            except (TypeError, ValueError, KeyError) as exc:
                rejected.append(
                    {
                        "source": source_name,
                        "check_id": check_id,
                        "reason": str(exc),
                    }
                )
                continue
            accepted_rows.append(row)
            accepted_defects.extend(row_defects)
            accepted_sources[check_id] = source_name
            accepted = True
            break
        if not accepted:
            accepted_rows.append(_default_valid_placement_row(check))
            defaulted_check_ids.append(check_id)

    accepted_judge_items: list[dict[str, Any]] = []
    accepted_judge_sources: dict[str, str] = {}
    dropped_judge_anchors: list[dict[str, Any]] = []
    if known_ids:
        initial = sources[0][1]
        raw_initial_items = initial.get("judge_originated_placement_results")
        raw_initial_items = (
            raw_initial_items if isinstance(raw_initial_items, list) else []
        )
        initial_anchors: list[tuple[str, tuple[str, ...], str]] = []
        for index, item in enumerate(raw_initial_items):
            try:
                anchor = _judge_originated_placement_identity_anchor(
                    item,
                    known_ids=known_ids,
                )
            except (TypeError, ValueError, KeyError) as exc:
                rejected.append(
                    {
                        "source": "initial",
                        "judge_item_index": index,
                        "reason": f"invalid_identity_anchor: {exc}",
                    }
                )
                continue
            if anchor not in initial_anchors:
                initial_anchors.append(anchor)

        accepted_judge_identities: set[
            tuple[str, tuple[str, ...], str]
        ] = set()
        for source_name, source in sources:
            raw_items = source.get("judge_originated_placement_results")
            raw_items = raw_items if isinstance(raw_items, list) else []
            source_defects = source.get("defects")
            source_defects = (
                source_defects if isinstance(source_defects, list) else []
            )
            for index, item in enumerate(raw_items):
                try:
                    identity = _judge_originated_placement_identity_anchor(
                        item,
                        known_ids=known_ids,
                    )
                except (TypeError, ValueError, KeyError) as exc:
                    rejected.append(
                        {
                            "source": source_name,
                            "judge_item_index": index,
                            "reason": f"invalid_identity_anchor: {exc}",
                        }
                    )
                    continue
                if identity not in initial_anchors:
                    rejected.append(
                        {
                            "source": source_name,
                            "judge_item_index": index,
                            "reason": (
                                "judge_candidate_has_no_initial_identity_anchor"
                            ),
                        }
                    )
                    continue
                if identity in accepted_judge_identities:
                    continue
                matching_defects = _judge_originated_candidate_defects(
                    item,
                    source_defects=source_defects,
                    identity=identity,
                )
                if not retain_invalid:
                    matching_defects = []
                synthetic = {
                    "verdict": "invalid",
                    "defects": deepcopy(matching_defects),
                    "placement_check_results": [],
                    "judge_originated_placement_results": [deepcopy(item)],
                }
                try:
                    adjusted, new_checks = (
                        normalize_judge_originated_placement_results(
                            synthetic,
                            known_ids=known_ids,
                            groups=groups,
                            existing_checks=deepcopy(required_checks),
                            expected_owner_stage=expected_owner_stage,
                        )
                    )
                    validate_placement_check_results(
                        adjusted,
                        required_checks=new_checks,
                        function_events=function_events,
                    )
                except (TypeError, ValueError, KeyError) as exc:
                    rejected.append(
                        {
                            "source": source_name,
                            "judge_item_index": index,
                            "reason": str(exc),
                        }
                    )
                    continue
                accepted_judge_identities.add(identity)
                accepted_judge_items.append(deepcopy(item))
                accepted_defects.extend(deepcopy(matching_defects))
                accepted_judge_sources[repr(identity)] = source_name
        dropped_judge_anchors = [
            {
                "subject_id": identity[0],
                "context_ids": list(identity[1]),
                "check_type": identity[2],
            }
            for identity in initial_anchors
            if identity not in accepted_judge_identities
        ]

    invalid = bool(accepted_defects)
    confidence = next(
        (
            source.get("confidence")
            for _, source in sources
            if isinstance(source.get("confidence"), (int, float))
        ),
        0.0,
    )
    if not isinstance(confidence, (int, float)) or not 0.0 <= float(
        confidence
    ) <= 1.0:
        confidence = 0.0
    eligible_atoms = len(required_checks) + len(accepted_judge_items) + len(
        dropped_judge_anchors
    )
    grounded_atoms = (
        len(required_checks)
        - len(defaulted_check_ids)
        + len(accepted_judge_items)
    )
    defaulted = bool(defaulted_check_ids or dropped_judge_anchors)
    return {
        "evidence_status": "sufficient",
        "verdict": "invalid" if invalid else "valid",
        "confidence": float(confidence),
        "reason": (
            "Legal Placement check rows were retained; malformed or missing "
            "rows defaulted valid after the bounded schema retry."
        ),
        "missing_evidence": [],
        "defects": accepted_defects if invalid else [],
        "evidence_request": None,
        "placement_check_results": accepted_rows,
        **(
            {
                "judge_originated_placement_results": accepted_judge_items,
            }
            if accepted_judge_items
            else {}
        ),
        "evidence_ambiguous": defaulted,
        "forced_binary": defaulted,
        "defaulted": defaulted,
        "decision_source": (
            "item_level_salvage"
            if defaulted
            else "retained_legal_rows"
        ),
        "placement_salvage_audit": {
            "policy": "valid_rows_plus_default_valid_v1",
            "required_check_count": len(required_checks),
            "grounded_check_count": (
                len(required_checks) - len(defaulted_check_ids)
            ),
            "defaulted_check_ids": defaulted_check_ids,
            "accepted_sources": accepted_sources,
            "accepted_judge_sources": accepted_judge_sources,
            "accepted_judge_originated_count": len(
                accepted_judge_items
            ),
            "dropped_judge_originated_count": len(
                dropped_judge_anchors
            ),
            "dropped_judge_originated_anchors": dropped_judge_anchors,
            "rejected_rows": rejected,
            "coverage": {
                "unit": "placement_check_or_judge_candidate",
                "eligible_count": eligible_atoms,
                "grounded_count": grounded_atoms,
                "fraction": (
                    grounded_atoms / eligible_atoms
                    if eligible_atoms
                    else 1.0
                ),
                "complete": grounded_atoms == eligible_atoms,
            },
        },
    }


def _judge_originated_placement_identity_anchor(
    value: Any,
    *,
    known_ids: set[str],
) -> tuple[str, tuple[str, ...], str]:
    if not isinstance(value, dict):
        raise ValueError("judge-originated placement row is not an object")
    subject_id = str(value.get("subject_id") or "").strip()
    if subject_id not in known_ids:
        raise ValueError("judge-originated placement subject is unknown")
    context_ids = _id_list(
        value.get("context_ids", []),
        known=known_ids,
        label="judge-originated placement context_ids",
    )
    if subject_id in context_ids:
        raise ValueError("judge-originated subject appears in its context")
    raw_type = value.get(
        "check_type",
        value.get("placement_check_type"),
    )
    check_type = canonical_placement_check_type(raw_type)
    return subject_id, tuple(sorted(context_ids)), check_type


def _judge_originated_candidate_defects(
    item: Any,
    *,
    source_defects: list[Any],
    identity: tuple[str, tuple[str, ...], str],
) -> list[dict[str, Any]]:
    if not isinstance(item, dict):
        return []
    subject_id, context_ids, check_type = identity
    proposal_id = str(item.get("proposal_id") or "").strip()
    check_id = placement_check_id(check_type, subject_id, context_ids)
    matches: list[dict[str, Any]] = []
    for defect in source_defects:
        if not isinstance(defect, dict):
            continue
        defect_ref = str(defect.get("check_id") or "").strip()
        if defect_ref in {proposal_id, check_id} - {""}:
            matches.append(deepcopy(defect))
            continue
        target_ids = {
            str(value)
            for value in defect.get("target_ids") or []
            if str(value).strip()
        }
        raw_type = defect.get("check_type", defect.get("relation"))
        try:
            defect_type = canonical_placement_check_type(raw_type)
        except ValueError:
            continue
        if subject_id in target_ids and defect_type == check_type:
            matches.append(deepcopy(defect))
    return matches


def _default_valid_placement_row(
    check: dict[str, Any],
) -> dict[str, Any]:
    return {
        "check_id": str(check["check_id"]),
        "subject_id": str(check["subject_id"]),
        "context_ids": sorted(
            str(item) for item in check.get("context_ids") or []
        ),
        "observation_status": "inferred_under_budget",
        "conclusion": "valid",
        "reason": (
            "No legal invalid finding survived the bounded recovery path; "
            "this atomic Placement check defaults valid with ambiguity."
        ),
    }


def canonicalize_placement_defect_linkage(
    result: dict[str, Any],
    *,
    required_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Canonicalize fields already determined by an exact typed check ID.

    ``relation`` and ``check_type`` carry no additional decision
    authority once a defect references a trusted check.  Normalizing these
    redundant fields avoids turning harmless wording drift into an unresolved
    Judge episode, while subject ownership and every trusted identity remain
    strictly validated downstream.
    """

    checks_by_id = {
        str(check.get("check_id") or ""): check
        for check in required_checks
        if isinstance(check, dict) and check.get("check_id")
    }
    normalized = deepcopy(result)
    defects = normalized.get("defects")
    if not isinstance(defects, list):
        return normalized
    for defect in defects:
        if not isinstance(defect, dict):
            continue
        check = checks_by_id.get(str(defect.get("check_id") or ""))
        if check is None:
            continue
        check_type = str(check.get("check_type") or "")
        if not check_type:
            continue
        defect["check_type"] = check_type
        defect["relation"] = check_type
    return normalized


def normalize_judge_originated_placement_results(
    result: dict[str, Any],
    *,
    known_ids: set[str],
    groups: list[dict[str, Any]] | None,
    existing_checks: list[dict[str, Any]],
    expected_owner_stage: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate and stabilize same-call recovery of discovery misses."""

    raw_items = result.get("judge_originated_placement_results")
    if raw_items is None:
        return deepcopy(result), []
    if not isinstance(raw_items, list) or any(
        not isinstance(item, dict) for item in raw_items
    ):
        raise ValueError(
            "judge_originated_placement_results must be a list of objects"
        )
    if result.get("verdict") != "invalid" and raw_items:
        raise ValueError(
            "judge-originated placement results require an invalid verdict"
        )
    canonical_items: list[dict[str, Any]] = []
    transport_warnings: list[dict[str, Any]] = []
    seen_proposal_ids: dict[str, tuple[str, tuple[str, ...], str]] = {}
    source_ref_by_identity: dict[
        tuple[str, tuple[str, ...], str], str
    ] = {}
    for raw_item in raw_items:
        source_proposal_id = str(raw_item.get("proposal_id") or "").strip()
        canonical_item, warnings = canonicalize_placement_proposal_transport(
            raw_item,
            known_ids=known_ids,
        )
        candidate = normalize_placement_candidate(
            canonical_item,
            known_ids=known_ids,
        )
        identity = (
            candidate["subject_id"],
            tuple(candidate["context_ids"]),
            candidate["check_type"],
        )
        proposal_id = str(canonical_item["proposal_id"])
        prior_identity = seen_proposal_ids.get(proposal_id)
        if prior_identity is not None:
            if prior_identity == identity:
                transport_warnings.append(
                    {
                        "code": "duplicate_identical_proposal_deduplicated",
                        "proposal_id": proposal_id,
                    }
                )
                continue
            proposal_id = "placement_proposal_" + placement_check_id(
                candidate["check_type"],
                candidate["subject_id"],
                candidate["context_ids"],
            ).removeprefix("placement_check_")
            canonical_item["proposal_id"] = proposal_id
            warnings.append(
                {
                    "code": "duplicate_proposal_id_regenerated",
                    "proposal_id": proposal_id,
                }
            )
        seen_proposal_ids[proposal_id] = identity
        source_ref_by_identity[identity] = source_proposal_id
        transport_warnings.extend(warnings)
        canonical_items.append(canonical_item)
    raw_items = canonical_items

    existing_identities = {
        _check_identity(check)
        for check in existing_checks
        if isinstance(check, dict)
    }
    object_to_group, known_groups = _trusted_group_partition(
        groups,
        known_ids=known_ids,
    )
    proposal_to_check: dict[str, dict[str, Any]] = {}
    proposal_reference_candidates: dict[str, list[dict[str, Any]]] = {}
    new_checks: list[dict[str, Any]] = []
    for item in raw_items:
        unknown = set(item) - {
            "proposal_id",
            "subject_id",
            "context_ids",
            "check_type",
            "observation_goal",
            "observation_status",
            "conclusion",
            "reason",
            "severity",
        }
        if unknown:
            raise ValueError(
                "judge-originated placement result contains unsupported "
                f"fields: {sorted(unknown)}"
            )
        proposal_id = str(item.get("proposal_id") or "").strip()
        if not proposal_id or proposal_id in proposal_to_check:
            raise ValueError(
                "judge-originated placement results require unique proposal IDs"
            )
        candidate = normalize_placement_candidate(
            item,
            known_ids=known_ids,
        )
        identity = (
            candidate["check_type"],
            candidate["subject_id"],
            tuple(sorted(candidate["context_ids"])),
        )
        if identity in existing_identities:
            raise ValueError(
                "judge-originated placement result duplicates a routed check"
            )
        if item.get("conclusion") != "invalid":
            raise ValueError(
                "judge-originated placement result must be a supported invalid "
                "finding; uncertainty must request evidence"
            )
        if item.get("observation_status") not in {
            "observed",
            "inferred_under_budget",
        }:
            raise ValueError(
                "judge-originated invalid placement result requires observed "
                "or inferred_under_budget evidence"
            )
        if not str(item.get("reason") or "").strip():
            raise ValueError(
                "judge-originated placement result requires a reason"
            )
        owner_stage, owning_group_id, group_ids = _check_owner(
            check_type=candidate["check_type"],
            subject_id=candidate["subject_id"],
            context_ids=candidate["context_ids"],
            object_to_group=object_to_group,
            known_groups=known_groups,
        )
        if (
            expected_owner_stage is not None
            and owner_stage != expected_owner_stage
        ):
            raise ValueError(
                "judge-originated placement check belongs to "
                f"{owner_stage!r}, not the active "
                f"{expected_owner_stage!r} phase"
            )
        check_id = placement_check_id(*identity)
        check = {
            "check_id": check_id,
            "check_type": candidate["check_type"],
            "subject_id": candidate["subject_id"],
            "context_ids": candidate["context_ids"],
            "target_ids": [candidate["subject_id"]],
            "owner_stage": owner_stage,
            "owning_group_id": owning_group_id,
            "group_ids": group_ids,
            "required_observations": list(
                _REQUIRED_OBSERVATIONS[candidate["check_type"]]
            ),
            "observation_goals": [candidate["observation_goal"]],
            "source_check_types": [candidate["check_type"]],
            "source_discovery_refs": [proposal_id],
            "origin": "judge_originated",
            "lifecycle_status": "resolved",
            "obligation_lifecycle": [
                {
                    "state": "discovered",
                    "source": "judge_originated_missed_check_sweep",
                },
                {
                    "state": "gate_ready",
                    "source": "current_judge_packet",
                },
                {
                    "state": "judged",
                    "source": "current_judge_packet",
                },
            ],
            "acquisition_status": "current_packet",
            "observation_complete": True,
            "observation_status": item["observation_status"],
            "judge_status": "resolved",
            "check_conclusion": "invalid",
            "judge_result_ref": None,
            "decision_authority": "none",
        }
        proposal_to_check[proposal_id] = check
        proposal_to_check[check_id] = check
        source_identity = (
            candidate["subject_id"],
            tuple(candidate["context_ids"]),
            candidate["check_type"],
        )
        for reference in {
            proposal_id,
            check_id,
            source_ref_by_identity.get(source_identity, ""),
        }:
            reference_checks = proposal_reference_candidates.setdefault(
                reference, []
            )
            if check not in reference_checks:
                reference_checks.append(check)
        existing_identities.add(identity)
        new_checks.append(check)

    adjusted = deepcopy(result)
    adjusted_items: list[dict[str, Any]] = []
    for item in raw_items:
        check = proposal_to_check[str(item["proposal_id"])]
        adjusted_items.append(
            {
                **deepcopy(item),
                "check_id": check["check_id"],
            }
        )
    adjusted.pop("judge_originated_placement_results", None)
    adjusted["judge_originated_placement_check_registrations"] = deepcopy(
        new_checks
    )
    for defect in adjusted.get("defects") or []:
        if not isinstance(defect, dict):
            continue
        defect_ref = str(defect.get("check_id") or "")
        candidates = list(
            proposal_reference_candidates.get(defect_ref) or []
        )
        if len(candidates) != 1:
            target_ids = {
                str(item)
                for item in defect.get("target_ids") or []
                if str(item).strip()
            }
            raw_type = defect.get("check_type", defect.get("relation"))
            try:
                defect_type = canonical_placement_check_type(raw_type)
            except ValueError:
                defect_type = None
            candidates = [
                check
                for check in new_checks
                if str(check.get("subject_id")) in target_ids
                and (
                    defect_type is None
                    or str(check.get("check_type")) == defect_type
                )
            ]
        check = candidates[0] if len(candidates) == 1 else None
        if check is None:
            continue
        defect["check_id"] = check["check_id"]
        defect["check_type"] = check["check_type"]
        defect["relation"] = check["check_type"]
        _validate_placement_defect_for_check(defect, check=check)
    referenced_new_ids = {
        str(defect.get("check_id") or "")
        for defect in adjusted.get("defects") or []
        if isinstance(defect, dict)
    }
    missing = sorted(
        str(check["check_id"])
        for check in new_checks
        if str(check["check_id"]) not in referenced_new_ids
    )
    if missing:
        raise ValueError(
            "judge-originated placement results require mapped defects: "
            f"{missing}"
        )
    rows = adjusted.get("placement_check_results")
    if rows is None:
        rows = []
    if not isinstance(rows, list):
        raise ValueError(
            "placement_check_results must be a list when Judge-originated "
            "checks are present"
        )
    existing_rows = {
        str(item.get("check_id") or ""): item
        for item in rows
        if isinstance(item, dict)
    }
    if (
        len(existing_rows) != len(rows)
        or any(not check_id for check_id in existing_rows)
    ):
        raise ValueError(
            "placement_check_results contain duplicate or empty check IDs"
        )
    additions: list[dict[str, Any]] = []
    for item in adjusted_items:
        check = proposal_to_check[str(item["proposal_id"])]
        normalized_row = {
            "check_id": check["check_id"],
            "subject_id": item["subject_id"],
            "context_ids": deepcopy(item.get("context_ids") or []),
            "observation_status": item["observation_status"],
            "conclusion": item["conclusion"],
            "reason": item["reason"],
        }
        prior_row = existing_rows.get(check["check_id"])
        if prior_row is not None:
            if prior_row != normalized_row:
                raise ValueError(
                    "judge-originated placement result disagrees with its "
                    "existing placement_check_results row"
                )
            continue
        additions.append(normalized_row)
    adjusted["placement_check_results"] = [
        *deepcopy(rows),
        *additions,
    ]
    if transport_warnings:
        adjusted["placement_transport_normalization"] = {
            "policy": "narrow_transport_only_v1",
            "warnings": transport_warnings,
        }
    return adjusted, new_checks


def build_pending_placement_check(
    proposal: Any,
    *,
    known_ids: set[str],
    groups: list[dict[str, Any]] | None,
    source_ref: str,
) -> dict[str, Any]:
    """Validate an evidence-request proposal before the next Judge call."""

    if not isinstance(proposal, dict):
        raise ValueError("placement check proposal must be a JSON object")
    proposal, _ = canonicalize_placement_proposal_transport(
        proposal,
        known_ids=known_ids,
        source_ref=source_ref,
    )
    unknown = set(proposal) - {
        "proposal_id",
        "subject_id",
        "context_ids",
        "check_type",
        "observation_goal",
    }
    if unknown:
        raise ValueError(
            "placement check proposal contains unsupported fields: "
            f"{sorted(unknown)}"
        )
    candidate = normalize_placement_candidate(
        proposal,
        known_ids=known_ids,
    )
    object_to_group, known_groups = _trusted_group_partition(
        groups,
        known_ids=known_ids,
    )
    owner_stage, owning_group_id, group_ids = _check_owner(
        check_type=candidate["check_type"],
        subject_id=candidate["subject_id"],
        context_ids=candidate["context_ids"],
        object_to_group=object_to_group,
        known_groups=known_groups,
    )
    proposal_id = str(proposal.get("proposal_id") or source_ref).strip()
    if not proposal_id:
        raise ValueError("placement check proposal requires a source reference")
    return {
        "check_id": placement_check_id(
            candidate["check_type"],
            candidate["subject_id"],
            candidate["context_ids"],
        ),
        "check_type": candidate["check_type"],
        "subject_id": candidate["subject_id"],
        "context_ids": list(candidate["context_ids"]),
        "target_ids": [candidate["subject_id"]],
        "owner_stage": owner_stage,
        "owning_group_id": owning_group_id,
        "group_ids": group_ids,
        "required_observations": list(
            _REQUIRED_OBSERVATIONS[candidate["check_type"]]
        ),
        "observation_goals": [candidate["observation_goal"]],
        "source_check_types": [candidate["check_type"]],
        "source_discovery_refs": [proposal_id],
        "origin": "judge_originated_evidence_request",
        "lifecycle_status": "evidence_requested",
        "obligation_lifecycle": [
            {
                "state": "discovered",
                "source": "judge_originated_missed_check_sweep",
            },
            {
                "state": "acquisition_attempted",
                "source": source_ref,
            },
        ],
        "acquisition_status": "pending",
        "observation_complete": False,
        "judge_status": "pending",
        "judge_result_ref": None,
        "decision_authority": "none",
    }


def merge_placement_checks(
    ledger: dict[str, Any],
    additions: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    result = deepcopy(ledger)
    checks = [
        deepcopy(item)
        for item in result.get("checks") or []
        if isinstance(item, dict)
    ]
    by_identity = {_check_identity(item): item for item in checks}
    for addition in additions:
        identity = _check_identity(addition)
        prior = by_identity.get(identity)
        if prior is None:
            item = deepcopy(addition)
            checks.append(item)
            by_identity[identity] = item
            continue
        prior["source_discovery_refs"] = _stable_unique(
            [
                *prior.get("source_discovery_refs", []),
                *addition.get("source_discovery_refs", []),
            ]
        )
        prior["observation_goals"] = _stable_unique(
            [
                *prior.get("observation_goals", []),
                *addition.get("observation_goals", []),
            ]
        )
    result["checks"] = sorted(
        checks,
        key=lambda item: str(item.get("check_id") or ""),
    )
    result["accepted_check_count"] = len(result["checks"])
    return result


def apply_placement_check_judgements(
    ledger: dict[str, Any],
    *,
    global_record: dict[str, Any] | None,
    group_results: list[dict[str, Any]],
    target_results: list[dict[str, Any]] | None = None,
    residual_records: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = deepcopy(ledger)
    checks_by_id = {
        str(check.get("check_id") or ""): check
        for check in result.get("checks") or []
        if isinstance(check, dict) and check.get("check_id")
    }
    rows_by_id: dict[str, tuple[dict[str, Any], str, bool]] = {}
    records: list[tuple[dict[str, Any], str]] = []
    if isinstance(global_record, dict):
        records.append((global_record, "global_discovery"))
    records.extend(
        (
            item,
            f"group_local_review:{item.get('group_id')}",
        )
        for item in group_results
        if isinstance(item, dict)
    )
    records.extend(
        (
            item,
            "residual_global_placement_review",
        )
        for item in residual_records or []
        if isinstance(item, dict)
    )
    records.extend(
        (
            item,
            f"target_local_confirmation:{item.get('target_id')}",
        )
        for item in target_results or []
        if isinstance(item, dict)
    )
    for record, phase in records:
        retained_visual_forced_check_ids = (
            _retained_visual_forced_placement_check_ids(record)
        )
        judgement = (
            record.get("judgement")
            if isinstance(record.get("judgement"), dict)
            else record
        )
        for row in judgement.get("placement_check_results") or []:
            if not isinstance(row, dict):
                continue
            check_id = str(row.get("check_id") or "")
            if check_id not in checks_by_id:
                raise ValueError(
                    f"Judge returned unknown placement check {check_id!r}"
                )
            if check_id in rows_by_id:
                raise ValueError(
                    f"placement check {check_id!r} was judged more than once"
                )
            rows_by_id[check_id] = (
                row,
                phase,
                check_id in retained_visual_forced_check_ids,
            )
        for item in judgement.get(
            "judge_originated_placement_results"
        ) or []:
            if not isinstance(item, dict):
                continue
            check_id = str(item.get("check_id") or "")
            if check_id not in checks_by_id:
                raise ValueError(
                    f"unknown judge-originated placement check {check_id!r}"
                )
            if check_id in rows_by_id:
                continue
            rows_by_id[check_id] = (
                {
                    "check_id": check_id,
                    "subject_id": item.get("subject_id"),
                    "context_ids": item.get("context_ids") or [],
                    "observation_status": item.get(
                        "observation_status"
                    ),
                    "conclusion": item.get("conclusion"),
                    "reason": item.get("reason"),
                },
                phase,
                check_id in retained_visual_forced_check_ids,
            )
    for check_id, check in checks_by_id.items():
        routed = rows_by_id.get(check_id)
        if routed is None:
            check["judge_status"] = "pending"
            continue
        row, phase, retained_visual_forced_choice = routed
        conclusion = str(row.get("conclusion") or "")
        observation_status = str(row.get("observation_status") or "")
        check["judge_status"] = (
            "resolved"
            if conclusion in {"valid", "invalid", "excluded_function_owned"}
            else "unresolved"
        )
        check["judge_result_ref"] = phase
        check["observation_status"] = observation_status
        check["grounded"] = bool(
            observation_status == "observed"
            or (
                observation_status == "inferred_under_budget"
                and conclusion in {"valid", "invalid"}
                and retained_visual_forced_choice
            )
        )
        check["observation_complete"] = observation_status in {
            "observed",
            "inferred_under_budget",
        }
        check["check_conclusion"] = conclusion
        check["result_row"] = deepcopy(row)
        check["result_row"]["context_ids"] = sorted(
            str(item) for item in row.get("context_ids") or []
        )
        if conclusion == "excluded_function_owned":
            check["function_event_ref"] = str(
                row.get("function_event_ref") or ""
            )
            check["same_physical_event"] = True
        if check["grounded"]:
            _append_obligation_transition(
                check,
                "gate_ready",
                source=phase,
            )
            _append_obligation_transition(check, "judged", source=phase)
        elif observation_status == "inferred_under_budget":
            _append_obligation_transition(
                check,
                "forced_or_defaulted",
                source=phase,
            )
        check["lifecycle_status"] = (
            "resolved"
            if check["judge_status"] == "resolved"
            else "unresolved"
        )
    unresolved_ids = [
        check_id
        for check_id, check in checks_by_id.items()
        if check.get("lifecycle_status") != "resolved"
    ]
    invalid_ids = [
        check_id
        for check_id, check in checks_by_id.items()
        if check.get("check_conclusion") == "invalid"
    ]
    excluded_ids = [
        check_id
        for check_id, check in checks_by_id.items()
        if check.get("check_conclusion") == "excluded_function_owned"
    ]
    grounded_ids = [
        check_id
        for check_id, check in checks_by_id.items()
        if check.get("grounded") is True
    ]
    resolved_ids = sorted(set(checks_by_id) - set(unresolved_ids))
    coverage = {
        "schema_version": PLACEMENT_CHECK_RESULT_VERSION,
        "required_check_count": len(checks_by_id),
        "resolved_check_count": len(resolved_ids),
        "resolved_check_ids": resolved_ids,
        "unresolved_check_ids": unresolved_ids,
        "invalid_check_ids": invalid_ids,
        "excluded_function_owned_check_ids": excluded_ids,
        "grounded_check_count": len(grounded_ids),
        "grounded_check_ids": grounded_ids,
        "grounding_fraction": (
            len(grounded_ids) / len(checks_by_id)
            if checks_by_id
            else 1.0
        ),
        "complete": not unresolved_ids,
        "decision_authority": "none",
    }
    return result, coverage


def _retained_visual_forced_placement_check_ids(
    record: Any,
) -> set[str]:
    """Return Placement checks forced by a real Judge over retained images.

    This mirrors Functional's narrow grounding rule. A terminal forced choice
    is grounded only when a Judge was actually invoked and the episode retained
    at least one visual artifact. Synthetic/default rows never qualify.
    """

    if not isinstance(record, dict):
        return set()
    raw_episodes = record.get("check_episodes")
    episodes = (
        [item for item in raw_episodes if isinstance(item, dict)]
        if isinstance(raw_episodes, list)
        else [record]
    )
    grounded_ids: set[str] = set()
    for episode in episodes:
        if episode.get("vlm_invoked") is not True:
            continue
        if int(episode.get("judge_episode_count") or 0) < 1:
            continue
        judgement = episode.get("judgement")
        judgement = (
            judgement if isinstance(judgement, dict) else episode
        )
        forced = judgement.get("budget_exhaustion_forced_choice")
        controller_forced = bool(
            isinstance(forced, dict)
            and forced.get("applied") is True
            and str(forced.get("final_verdict") or "")
            in {"valid", "invalid"}
        )
        target_retained_global_forced = bool(
            episode.get("retained_global_forced_final") is True
            and judgement.get("verdict") in {"valid", "invalid"}
        )
        if not controller_forced and not target_retained_global_forced:
            continue
        evidence_paths = {
            str(path)
            for path in episode.get("evidence_paths") or []
            if str(path).strip()
        }
        forced_evidence = {
            str(path)
            for path in (
                forced.get("evidence_artifacts")
                if isinstance(forced, dict)
                else []
            ) or []
            if str(path).strip()
        }
        if not evidence_paths and not forced_evidence:
            continue
        rows = [
            *(judgement.get("placement_check_results") or []),
            *(judgement.get("judge_originated_placement_results") or []),
        ]
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("observation_status") != "inferred_under_budget":
                continue
            if row.get("conclusion") not in {"valid", "invalid"}:
                continue
            check_id = str(row.get("check_id") or "")
            if check_id:
                grounded_ids.add(check_id)
    return grounded_ids


def _append_obligation_transition(
    check: dict[str, Any],
    state: str,
    *,
    source: str,
) -> None:
    lifecycle = check.setdefault("obligation_lifecycle", [])
    transition = {"state": str(state), "source": str(source)}
    if not lifecycle or lifecycle[-1] != transition:
        lifecycle.append(transition)


def _validate_exact_placement_row(
    row: dict[str, Any],
    *,
    check: dict[str, Any],
) -> None:
    check_id = str(check["check_id"])
    if set(row) - {
        "check_id",
        "subject_id",
        "context_ids",
        "observation_status",
        "conclusion",
        "reason",
        "function_event_ref",
        "same_physical_event",
    }:
        raise ValueError(
            f"placement check {check_id} returned unsupported fields"
        )
    if str(row.get("subject_id") or "") != str(check["subject_id"]):
        raise ValueError(
            f"placement check {check_id} returned the wrong subject"
        )
    context_ids = row.get("context_ids")
    if (
        not isinstance(context_ids, list)
        or tuple(sorted(str(item) for item in context_ids))
        != tuple(sorted(str(item) for item in check["context_ids"]))
    ):
        raise ValueError(
            f"placement check {check_id} returned the wrong context set"
        )


def _validate_placement_defect_for_check(
    defect: dict[str, Any],
    *,
    check: dict[str, Any],
) -> None:
    check_id = str(check["check_id"])
    if str(defect.get("check_id") or "") != check_id:
        raise ValueError(
            f"placement defect does not map to check {check_id}"
        )
    if str(defect.get("check_type") or "") != str(
        check["check_type"]
    ):
        raise ValueError(
            f"placement defect for {check_id} has the wrong check type"
        )
    if defect.get("target_ids") != [str(check["subject_id"])]:
        raise ValueError(
            f"placement defect for {check_id} must target only its subject"
        )
    if str(defect.get("relation") or "") != str(check["check_type"]):
        raise ValueError(
            f"placement defect for {check_id} must use its check type as "
            "the stable relation"
        )


def _discovery_object_ids(discovery: dict[str, Any]) -> list[str]:
    values = discovery.get("considered_object_ids")
    if not isinstance(values, list) or not values:
        raise ValueError(
            "placement discovery requires considered_object_ids"
        )
    result = [str(item).strip() for item in values]
    if any(not item for item in result) or len(result) != len(set(result)):
        raise ValueError(
            "placement discovery object IDs must be non-empty and unique"
        )
    return result


def _trusted_group_partition(
    groups: list[dict[str, Any]] | None,
    *,
    known_ids: set[str],
) -> tuple[dict[str, str], set[str]]:
    object_to_group: dict[str, str] = {}
    group_ids: set[str] = set()
    for group in groups or []:
        if not isinstance(group, dict):
            raise TypeError("placement groups must contain JSON objects")
        group_id = str(group.get("group_id") or "").strip()
        if not group_id or group_id in group_ids:
            raise ValueError(
                "placement groups require unique non-empty IDs"
            )
        group_ids.add(group_id)
        for object_id in group.get("object_ids") or []:
            normalized = str(object_id).strip()
            if normalized not in known_ids:
                raise ValueError(
                    f"placement group references unknown object {normalized!r}"
                )
            if normalized in object_to_group:
                raise ValueError(
                    f"placement object {normalized!r} belongs to multiple groups"
                )
            object_to_group[normalized] = group_id
    if group_ids and set(object_to_group) != known_ids:
        missing = sorted(known_ids - set(object_to_group))
        raise ValueError(
            "placement group partition does not cover every object: "
            f"{missing}"
        )
    return object_to_group, group_ids


def _check_owner(
    *,
    check_type: str,
    subject_id: str,
    context_ids: list[str],
    object_to_group: dict[str, str],
    known_groups: set[str],
) -> tuple[str, str | None, list[str]]:
    subject_group = object_to_group.get(subject_id)
    scoped_groups = list(
        dict.fromkeys(
            group_id
            for object_id in [subject_id, *context_ids]
            if (group_id := object_to_group.get(object_id))
        )
    )
    if known_groups and subject_group is None:
        raise ValueError(
            f"placement subject {subject_id!r} has no trusted group"
        )
    if not known_groups and subject_group is None and check_type != "scene_zone":
        return "target_local", None, []
    if check_type == "scene_zone":
        return "scene_global", None, scoped_groups
    if (
        check_type == "contextual_anchor"
        and len(scoped_groups) > 1
    ):
        return "scene_global", None, scoped_groups
    return "group_local", subject_group, scoped_groups


def _check_identity(
    check: dict[str, Any],
) -> tuple[str, str, tuple[str, ...]]:
    return (
        canonical_placement_check_type(check.get("check_type")),
        str(check.get("subject_id") or ""),
        tuple(
            sorted(str(item) for item in check.get("context_ids") or [])
        ),
    )


def _id_list(
    value: Any,
    *,
    known: set[str],
    label: str,
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON list")
    result = [str(item).strip() for item in value]
    if any(not item for item in result) or len(result) != len(set(result)):
        raise ValueError(f"{label} contains invalid or duplicate IDs")
    unknown = sorted(set(result) - known)
    if unknown:
        raise ValueError(f"{label} references unknown IDs: {unknown}")
    return result


def _stable_unique(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(deepcopy(value))
    return result


def _pending_proposal_from_result(
    result: dict[str, Any],
) -> dict[str, Any] | None:
    request = result.get("evidence_request")
    metadata = (
        request.get("metadata")
        if isinstance(request, dict)
        and isinstance(request.get("metadata"), dict)
        else {}
    )
    proposal = metadata.get("placement_check_proposal")
    return proposal if isinstance(proposal, dict) else None


__all__ = [
    "canonicalize_placement_defect_linkage",
    "PLACEMENT_CHECK_CONCLUSIONS",
    "PLACEMENT_CHECK_LEDGER_VERSION",
    "PLACEMENT_CHECK_RESULT_VERSION",
    "PLACEMENT_CHECK_TYPES",
    "apply_placement_check_judgements",
    "build_pending_placement_check",
    "build_placement_check_ledger",
    "canonical_placement_check_type",
    "canonicalize_placement_proposal_transport",
    "forced_group_ids_from_placement_checks",
    "merge_placement_checks",
    "normalize_judge_originated_placement_results",
    "normalize_placement_candidate",
    "placement_camera_targets_by_group",
    "placement_check_id",
    "placement_checks_for_group",
    "placement_global_checks",
    "salvage_placement_judge_response",
    "validate_placement_check_results",
]
