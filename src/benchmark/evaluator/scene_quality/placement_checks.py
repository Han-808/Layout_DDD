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
PLACEMENT_CHECK_RESULT_VERSION = "placement_check_results_v1"
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
    """Require exact rows and exact defect ownership for routed checks."""

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
                "or excluded conclusion"
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
            event_ref = str(row.get("function_event_ref") or "")
            if event_ref not in event_by_id:
                raise ValueError(
                    f"placement check {check_id} references an unknown "
                    "functional ownership event"
                )
            if row.get("same_physical_event") is not True:
                raise ValueError(
                    f"placement check {check_id} must explicitly confirm "
                    "same_physical_event before exclusion"
                )
            event_targets = {
                str(item)
                for field in (
                    "affected_object_ids",
                    "causal_object_ids",
                    "scoring_target_ids",
                )
                for item in event_by_id[event_ref].get(field) or []
            }
            if str(check["subject_id"]) not in event_targets:
                raise ValueError(
                    f"placement check {check_id} function event has no "
                    "object-role overlap"
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
            "valid or function-owned"
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
        check = proposal_to_check.get(str(defect.get("check_id") or ""))
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = deepcopy(ledger)
    checks_by_id = {
        str(check.get("check_id") or ""): check
        for check in result.get("checks") or []
        if isinstance(check, dict) and check.get("check_id")
    }
    rows_by_id: dict[str, tuple[dict[str, Any], str]] = {}
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
    for record, phase in records:
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
            rows_by_id[check_id] = (row, phase)
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
            )
    for check_id, check in checks_by_id.items():
        routed = rows_by_id.get(check_id)
        if routed is None:
            check["judge_status"] = "pending"
            continue
        row, phase = routed
        conclusion = str(row.get("conclusion") or "")
        observation_status = str(row.get("observation_status") or "")
        check["judge_status"] = (
            "resolved"
            if conclusion
            in {"valid", "invalid", "excluded_function_owned"}
            else "unresolved"
        )
        check["judge_result_ref"] = phase
        check["observation_status"] = observation_status
        check["observation_complete"] = observation_status in {
            "observed",
            "inferred_under_budget",
        }
        check["check_conclusion"] = conclusion
        check["result_row"] = deepcopy(row)
        check["result_row"]["context_ids"] = sorted(
            str(item) for item in row.get("context_ids") or []
        )
        check["lifecycle_status"] = (
            "resolved"
            if check["judge_status"] == "resolved"
            else "unresolved"
        )
        if row.get("function_event_ref"):
            check["function_event_ref"] = str(
                row["function_event_ref"]
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
    resolved_ids = sorted(set(checks_by_id) - set(unresolved_ids))
    coverage = {
        "schema_version": PLACEMENT_CHECK_RESULT_VERSION,
        "required_check_count": len(checks_by_id),
        "resolved_check_count": len(resolved_ids),
        "resolved_check_ids": resolved_ids,
        "unresolved_check_ids": unresolved_ids,
        "invalid_check_ids": invalid_ids,
        "excluded_function_owned_check_ids": excluded_ids,
        "complete": not unresolved_ids,
        "decision_authority": "none",
    }
    return result, coverage


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
    "forced_group_ids_from_placement_checks",
    "merge_placement_checks",
    "normalize_judge_originated_placement_results",
    "normalize_placement_candidate",
    "placement_camera_targets_by_group",
    "placement_check_id",
    "placement_checks_for_group",
    "placement_global_checks",
    "validate_placement_check_results",
]
