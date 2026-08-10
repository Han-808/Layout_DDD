"""Typed obligations between functional discovery and metric judgement.

Discovery remains non-judging: it proposes observable checks.  This module
turns accepted discovery records into an auditable ledger whose entries must
be routed to exactly one Judge stage.  The ledger never produces a metric
verdict.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from benchmark.visual_judge.functional_discovery_contract import (
    FUNCTIONAL_RELATION_PREDICATES,
    normalized_functional_relation_predicates,
)

FUNCTIONAL_CHECK_LEDGER_VERSION = "functional_check_ledger_v5"
FUNCTIONAL_CHECK_RESULT_VERSION = "functional_check_results_v2"

_RELATION_CHECK_OBSERVATIONS = {
    "directional_correspondence": (
        "target_visible",
        "joint_visibility",
        "interaction_side_visible",
        "front_back_disambiguated",
    ),
    "relative_use_geometry": (
        "target_visible",
        "joint_visibility",
        "relative_layout_visible",
        "interaction_region_visible",
    ),
}
_RELATION_ACQUISITION_OBSERVATIONS = {
    predicate: (
        *observations,
        "group_context_visible",
        "limited_local_context",
    )
    for predicate, observations in _RELATION_CHECK_OBSERVATIONS.items()
}


def functional_relation_required_observations(
    predicate: str,
    *,
    acquisition: bool = False,
) -> tuple[str, ...]:
    """Return the minimal deterministic observation recipe for one predicate."""

    normalized = str(predicate or "").strip()
    if normalized not in FUNCTIONAL_RELATION_PREDICATES:
        raise ValueError(
            f"unsupported functional relation predicate {normalized!r}"
        )
    source = (
        _RELATION_ACQUISITION_OBSERVATIONS
        if acquisition
        else _RELATION_CHECK_OBSERVATIONS
    )
    return tuple(source[normalized])
_ARCHITECTURE_ORIENTATION_OBSERVATIONS = (
    "target_visible",
    "interaction_side_visible",
    "front_back_disambiguated",
    "architecture_plane_visible",
    "global_context_preserved",
)
_CLEARANCE_OBSERVATIONS = (
    "target_visible",
    "interaction_side_visible",
    "approach_zone_visible",
)
_OPERATING_CLEARANCE_OBSERVATIONS = (
    "target_visible",
    "approach_zone_visible",
)
_LOCAL_CONFIRMATION_OBSERVATIONS = (
    "target_visible",
    "interaction_side_visible",
    "front_back_disambiguated",
)


def build_functional_check_ledger(
    discovery: dict[str, Any],
    *,
    groups: list[dict[str, Any]] | None = None,
    scene: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project trusted discovery records into typed, non-judging obligations."""

    if not isinstance(discovery, dict):
        raise TypeError("functional discovery must be a JSON object")
    known_ids = _known_object_ids(discovery)
    object_to_group, groups_by_id = _trusted_group_partition(
        groups,
        known_ids=known_ids,
    )
    directed_by_target: dict[str, dict[str, Any]] = {}
    for item in discovery.get("directed_surface_targets") or []:
        _require_record(item, "directed usable-side target")
        target_id = str(item.get("target_id") or "")
        _validated_targets([target_id], known_ids=known_ids)
        if target_id in directed_by_target:
            raise ValueError(
                f"duplicate directed usable-side target {target_id!r}"
            )
        if not isinstance(item.get("need_clearance"), bool):
            raise ValueError(
                "directed usable-side target need_clearance must be a "
                f"boolean; target_id={target_id!r}"
            )
        directed_by_target[target_id] = item
    pending: dict[
        tuple[str, str, tuple[str, ...], str | None, str],
        dict[str, Any],
    ] = {}

    def add(
        *,
        check_type: str,
        owner_stage: str,
        target_ids: list[str],
        owning_group_id: str | None,
        group_ids: list[str],
        relation: str,
        required_observations: tuple[str, ...],
        discovery_id: str,
        check_family: str,
        requires_dedicated_acquisition: bool,
        observation_goal: str,
        observation_kinds: list[str] | None = None,
        predicate: str | None = None,
        surface_roles: list[str] | None = None,
        need_clearance: bool = False,
    ) -> None:
        normalized_targets = _validated_targets(
            target_ids,
            known_ids=known_ids,
        )
        normalized_group_ids = _validated_group_scope(
            normalized_targets,
            declared_group_ids=group_ids,
            declared_owner=owning_group_id,
            owner_stage=owner_stage,
            object_to_group=object_to_group,
            groups_by_id=groups_by_id,
        )
        normalized_owner = (
            normalized_group_ids[0]
            if owner_stage == "group_local"
            else None
        )
        identity = (
            check_type,
            owner_stage,
            tuple(sorted(normalized_targets)),
            normalized_owner,
            relation,
        )
        current = pending.get(identity)
        if current is None:
            current = {
                "check_type": check_type,
                "check_family": check_family,
                "owner_stage": owner_stage,
                "route_scope": (
                    "cross_group"
                    if owner_stage == "cross_group_relation"
                    else "group_local"
                ),
                "target_ids": normalized_targets,
                "group_ids": normalized_group_ids,
                "owning_group_id": normalized_owner,
                "relation": relation,
                "predicate": predicate,
                "required_observations": [],
                "observation_goals": [],
                "observation_kinds": [],
                "surface_roles": [],
                "need_clearance": False,
                "target_affordances": [],
                "routing_discovery_ids": [],
                "source_discovery_ids": [],
                "requires_dedicated_acquisition": bool(
                    requires_dedicated_acquisition
                ),
                "acquisition_policy": (
                    "proactive_required"
                    if requires_dedicated_acquisition
                    else "judge_requested_on_demand"
                ),
                "lifecycle_status": "accepted",
                "acquisition_status": "pending",
                "artifact_rendered": False,
                "view_coverage_complete": False,
                "observation_complete": False,
                "judge_status": "pending",
                "judge_result_ref": None,
                "decision_authority": "none",
            }
            pending[identity] = current
        current["required_observations"] = _stable_unique(
            [
                *current["required_observations"],
                *required_observations,
            ]
        )
        current["observation_goals"] = _stable_unique(
            [
                *current["observation_goals"],
                observation_goal,
            ]
        )
        current["observation_kinds"] = _stable_unique(
            [
                *current["observation_kinds"],
                *(observation_kinds or []),
            ]
        )
        current["surface_roles"] = _stable_unique(
            [
                *current["surface_roles"],
                *(surface_roles or []),
            ]
        )
        current["need_clearance"] = bool(
            current["need_clearance"] or need_clearance
        )
        current["source_discovery_ids"] = _stable_unique(
            [
                *current["source_discovery_ids"],
                discovery_id,
            ]
        )
        current["routing_discovery_ids"] = _stable_unique(
            [
                *current["routing_discovery_ids"],
                discovery_id,
            ]
        )
        current["requires_dedicated_acquisition"] = bool(
            current["requires_dedicated_acquisition"]
            or requires_dedicated_acquisition
        )

    for item in discovery.get("cross_group_correspondences") or []:
        _require_record(item, "cross-group correspondence")
        for predicate in normalized_functional_relation_predicates(item):
            add(
                check_type=predicate,
                owner_stage="cross_group_relation",
                target_ids=list(item.get("target_ids") or []),
                owning_group_id=None,
                group_ids=list(item.get("group_ids") or []),
                relation=predicate,
                required_observations=(
                    functional_relation_required_observations(predicate)
                ),
                discovery_id=_required_discovery_id(item),
                check_family="cross_group_correspondence",
                requires_dedicated_acquisition=True,
                observation_goal=_observation_goal(item),
                observation_kinds=[predicate],
                predicate=predicate,
            )

    for item in discovery.get("within_group_correspondences") or []:
        _require_record(item, "within-group correspondence")
        group_ids = list(item.get("group_ids") or [])
        for predicate in normalized_functional_relation_predicates(item):
            add(
                check_type=predicate,
                owner_stage="group_local",
                target_ids=list(item.get("target_ids") or []),
                owning_group_id=(
                    str(group_ids[0]) if group_ids else None
                ),
                group_ids=group_ids,
                relation=predicate,
                required_observations=(
                    functional_relation_required_observations(predicate)
                ),
                discovery_id=_required_discovery_id(item),
                check_family="within_group_correspondence",
                requires_dedicated_acquisition=False,
                observation_goal=_observation_goal(item),
                observation_kinds=[predicate],
                predicate=predicate,
            )

    approach_by_target: dict[str, list[dict[str, Any]]] = {}
    boundary_by_target: dict[str, list[dict[str, Any]]] = {}
    for field in (
        "approach_clearance_targets",
        "boundary_sensitive_targets",
    ):
        for item in discovery.get(field) or []:
            _require_record(item, field)
            target_id = str(item.get("target_id") or "")
            _validated_targets([target_id], known_ids=known_ids)
            if (
                field == "approach_clearance_targets"
                and item.get("need_clearance") is not True
            ):
                raise ValueError(
                    "approach clearance target need_clearance must be true; "
                    f"target_id={target_id!r}"
                )
            destination = (
                approach_by_target
                if field == "approach_clearance_targets"
                else boundary_by_target
            )
            destination.setdefault(target_id, []).append(item)

    for target_id, directed_record in directed_by_target.items():
        declared = directed_record["need_clearance"]
        routed = bool(approach_by_target.get(target_id))
        if declared is not routed:
            raise ValueError(
                "directed usable-side clearance declaration must match one "
                "approach clearance target; "
                f"target_id={target_id!r}, need_clearance={declared!r}"
            )

    # Direction and free-space are independent predicates.  Every directed
    # object owns one architecture-orientation obligation.  Clearance is added
    # only when discovery says ordinary use requires free space (or when a
    # non-directed boundary-sensitive target explicitly requests that region).
    # The two obligations may share one side-conditioned acquisition unit, but
    # the Judge must return one result row for each check.
    for target_id, directed_record in directed_by_target.items():
        boundary_records = boundary_by_target.get(target_id, [])
        for record in [directed_record, *boundary_records]:
            add(
                check_type="architecture_orientation",
                owner_stage="group_local",
                target_ids=[target_id],
                owning_group_id=_optional_text(
                    record.get("owning_group_id")
                ),
                group_ids=[],
                relation="usable_side_toward_accessible_interior",
                required_observations=(
                    _ARCHITECTURE_ORIENTATION_OBSERVATIONS
                ),
                discovery_id=_required_discovery_id(record),
                check_family="architecture_orientation",
                requires_dedicated_acquisition=True,
                observation_goal=_observation_goal(record),
                surface_roles=list(
                    directed_record.get("surface_roles") or []
                ),
                need_clearance=False,
            )

    # ``need_clearance`` is the only discovery predicate that creates a
    # clearance obligation.  Boundary sensitivity may strengthen framing for
    # an existing clearance check, but it must not manufacture one: room
    # architecture is one possible blocker/source of context, not a separate
    # metric claim.  This is especially important for non-directed objects,
    # where a boundary-only routing hint previously created an unrelated
    # surrounding-clearance penalty path.
    clearance_target_ids = sorted(approach_by_target)
    for target_id in clearance_target_ids:
        directed_record = directed_by_target.get(target_id)
        approach_records = approach_by_target.get(target_id, [])
        boundary_records = boundary_by_target.get(target_id, [])
        records = [*approach_records, *boundary_records]
        required = list(
            _CLEARANCE_OBSERVATIONS
            if directed_record is not None
            else _OPERATING_CLEARANCE_OBSERVATIONS
        )
        if boundary_records:
            required.extend(
                [
                    "architecture_plane_visible",
                    "global_context_preserved",
                ]
            )
        for record in records:
            add(
                check_type="clearance",
                owner_stage="group_local",
                target_ids=[target_id],
                owning_group_id=_optional_text(
                    record.get("owning_group_id")
                ),
                group_ids=[],
                relation=(
                    "usable_side_clearance"
                    if directed_record is not None
                    else "surrounding_operating_clearance"
                ),
                required_observations=tuple(required),
                discovery_id=_required_discovery_id(record),
                check_family="clearance",
                requires_dedicated_acquisition=True,
                observation_goal=_observation_goal(record),
                surface_roles=(
                    list(directed_record.get("surface_roles") or [])
                    if directed_record is not None
                    else []
                ),
                need_clearance=True,
            )

    for item in discovery.get("unusual_unconfirmed") or []:
        _require_record(item, "local affordance confirmation")
        add(
            check_type="local_affordance_confirmation",
            owner_stage="group_local",
            target_ids=list(item.get("target_ids") or []),
            owning_group_id=_optional_text(item.get("owning_group_id")),
            group_ids=[],
            relation="local_affordance_confirmation",
            required_observations=_LOCAL_CONFIRMATION_OBSERVATIONS,
            discovery_id=_required_discovery_id(item),
            check_family="local_affordance_confirmation",
            requires_dedicated_acquisition=True,
            observation_goal=_observation_goal(item),
        )

    # A usable-side record may also be relevant to a correspondence or local
    # confirmation. Preserve that provenance in addition to its own access
    # check, without making the discovery record a decision authority.
    for record in pending.values():
        for target_id in record["target_ids"]:
            directed_record = directed_by_target.get(str(target_id))
            if directed_record is None:
                continue
            record["source_discovery_ids"] = _stable_unique(
                [
                    *record["source_discovery_ids"],
                    _required_discovery_id(directed_record),
                ]
            )
            record["observation_goals"] = _stable_unique(
                [
                    *record["observation_goals"],
                    _observation_goal(directed_record),
                ]
            )
            record["target_affordances"] = _stable_unique(
                [
                    *record["target_affordances"],
                    {
                        "target_id": str(target_id),
                        "directionality": directed_record.get(
                            "directionality"
                        ),
                        "surface_roles": list(
                            directed_record.get("surface_roles") or []
                        ),
                        "need_clearance": directed_record.get(
                            "need_clearance"
                        ),
                        "observation_goal": _observation_goal(
                            directed_record
                        ),
                    },
                ]
            )

    checks = []
    for index, (_, record) in enumerate(
        sorted(pending.items(), key=lambda item: item[0]),
        start=1,
    ):
        if record["check_type"] == "clearance":
            target_id = str(record["target_ids"][0])
            causal_candidates = _clearance_causal_candidates(
                target_id=target_id,
                known_ids=known_ids,
                scene=scene,
                limit=4,
            )
            record["causal_candidate_ids"] = [
                str(item["object_id"])
                for item in causal_candidates
            ]
            record["causal_candidates"] = causal_candidates
            record["causal_candidates_are_routing_prior"] = True
            record["allowed_causal_object_ids"] = sorted(known_ids)
            record["causal_owner_required_when_invalid"] = True
        checks.append(
            {
                "check_id": f"functional_check_{index:03d}",
                **record,
            }
        )
    routine_inventory: list[dict[str, Any]] = []
    return {
        "schema_version": FUNCTIONAL_CHECK_LEDGER_VERSION,
        "checks": checks,
        "accepted_check_count": len(checks),
        "routine_inventory": routine_inventory,
        "routine_inventory_count": len(routine_inventory),
        "advisory_clearance_inventory": [],
        "advisory_clearance_inventory_count": 0,
        "decision_authority": "none",
    }


def route_functional_check_ledger(
    ledger: dict[str, Any],
    *,
    selected_probe_units: list[dict[str, Any]],
    deferred_probe_units: list[dict[str, Any]],
) -> dict[str, Any]:
    """Record deterministic camera scheduling without resolving any check."""

    result = deepcopy(ledger)
    selected_ids = {
        str(check_id)
        for unit in selected_probe_units
        for check_id in unit.get("check_ids") or []
    }
    deferred_ids = {
        str(check_id)
        for unit in deferred_probe_units
        for check_id in unit.get("check_ids") or []
    }
    for check in result.get("checks") or []:
        check_id = str(check.get("check_id") or "")
        check["lifecycle_status"] = "routed"
        if check_id in selected_ids:
            check["acquisition_status"] = (
                "scheduled"
                if check.get("requires_dedicated_acquisition")
                else "shared_probe_scheduled"
            )
        elif check_id in deferred_ids:
            check["acquisition_status"] = (
                "deferred_budget"
                if check.get("requires_dedicated_acquisition")
                else "shared_probe_deferred"
            )
        else:
            check["acquisition_status"] = (
                "shared_scope_evidence"
                if check.get("requires_dedicated_acquisition")
                else "judge_requested_on_demand"
            )
    return result


def update_functional_check_evidence(
    ledger: dict[str, Any],
    *,
    probe_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Attach artifact state while leaving observation completion to the Judge."""

    result = deepcopy(ledger)
    results_by_check: dict[str, list[dict[str, Any]]] = {}
    for probe in probe_results:
        if not isinstance(probe, dict):
            continue
        for check_id in probe.get("check_ids") or []:
            results_by_check.setdefault(str(check_id), []).append(probe)
    for check in result.get("checks") or []:
        records = results_by_check.get(str(check.get("check_id") or ""), [])
        paths = _stable_unique(
            [
                str(path)
                for record in records
                for path in record.get("evidence_paths") or []
                if str(path).strip()
            ]
        )
        available = any(
            record.get("status") == "available" and record.get("evidence_paths")
            for record in records
        )
        failed = bool(records) and not available
        check["evidence_refs"] = paths
        check["artifact_rendered"] = bool(available)
        # A rendered image is not proof that the required relation or usable
        # side was decoded.  Only the Judge's per-check result can complete the
        # observation lifecycle.
        check["view_coverage_complete"] = False
        check["observation_complete"] = False
        if available:
            check["acquisition_status"] = "artifact_rendered"
        elif failed:
            check["acquisition_status"] = "failed"
    return result


def checks_for_group(
    ledger: dict[str, Any] | None,
    group_id: str,
) -> list[dict[str, Any]]:
    return [
        deepcopy(check)
        for check in (ledger or {}).get("checks") or []
        if check.get("owner_stage") == "group_local"
        and str(check.get("owning_group_id") or "") == str(group_id)
    ]


def checks_for_cross_group_relation(
    ledger: dict[str, Any] | None,
    *,
    target_ids: list[str],
    discovery_ids: list[str] | None = None,
    predicates: list[str] | None = None,
) -> list[dict[str, Any]]:
    target_set = tuple(sorted(str(item) for item in target_ids))
    discovery_set = {
        str(item) for item in discovery_ids or [] if str(item).strip()
    }
    predicate_set = {
        str(item) for item in predicates or [] if str(item).strip()
    }
    matches = [
        check
        for check in (ledger or {}).get("checks") or []
        if check.get("owner_stage") == "cross_group_relation"
        and tuple(sorted(str(item) for item in check.get("target_ids") or []))
        == target_set
        and (
            not predicate_set
            or str(check.get("predicate") or check.get("check_type") or "")
            in predicate_set
        )
        and (
            not discovery_set
            or discovery_set
            & {
                str(item)
                for item in check.get("source_discovery_ids") or []
            }
        )
    ]
    if not matches:
        raise ValueError(
            "cross-group relation must map to at least one functional check"
        )
    return [
        deepcopy(check)
        for check in sorted(
            matches,
            key=lambda item: str(item.get("check_id") or ""),
        )
    ]


def check_for_cross_group_relation(
    ledger: dict[str, Any] | None,
    *,
    target_ids: list[str],
    discovery_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Compatibility helper for frozen one-check relation episodes."""

    matches = checks_for_cross_group_relation(
        ledger,
        target_ids=target_ids,
        discovery_ids=discovery_ids,
    )
    if len(matches) != 1:
        raise ValueError(
            "singular cross-group relation lookup requires exactly one check"
        )
    return matches[0]


def validate_functional_check_results(
    result: dict[str, Any],
    *,
    required_checks: list[dict[str, Any]],
    invalid_verdict_requires_invalid_check: bool = False,
) -> dict[str, Any]:
    """Require exact per-check acknowledgement before accepting final valid."""

    if not required_checks:
        return {
            "schema_version": FUNCTIONAL_CHECK_RESULT_VERSION,
            "required_check_count": 0,
            "resolved_check_ids": [],
            "unresolved_check_ids": [],
            "invalid_check_ids": [],
            "complete": True,
            "decision_authority": "none",
        }
    expected = {
        str(check["check_id"]): check
        for check in required_checks
        if isinstance(check, dict) and check.get("check_id")
    }
    if len(expected) != len(required_checks):
        raise ValueError("required functional checks contain invalid IDs")
    rows = result.get("functional_check_results")
    if not isinstance(rows, list):
        raise ValueError(
            "functional Judge response requires functional_check_results"
        )
    if any(not isinstance(item, dict) for item in rows):
        raise ValueError("functional_check_results must contain JSON objects")
    returned_ids = [str(item.get("check_id") or "") for item in rows]
    if (
        len(returned_ids) != len(set(returned_ids))
        or set(returned_ids) != set(expected)
    ):
        raise ValueError(
            "functional_check_results must cover every required check exactly once"
        )

    resolved: list[str] = []
    unresolved: list[str] = []
    invalid: list[str] = []
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        check_id = str(row.get("check_id") or "")
        check = expected[check_id]
        expected_targets = tuple(
            sorted(str(item) for item in check["target_ids"])
        )
        returned_targets = row.get("target_ids")
        if (
            not isinstance(returned_targets, list)
            or tuple(sorted(str(item) for item in returned_targets))
            != expected_targets
        ):
            raise ValueError(
                f"functional check {check_id} returned the wrong target set"
            )
        observation_status = row.get("observation_status")
        conclusion = row.get("conclusion")
        if observation_status not in {
            "observed",
            "inferred_under_budget",
            "missing",
        }:
            raise ValueError(
                f"functional check {check_id} has invalid observation_status"
            )
        if conclusion not in {"valid", "invalid", "unresolved"}:
            raise ValueError(
                f"functional check {check_id} has invalid conclusion"
            )
        if not str(row.get("reason") or "").strip():
            raise ValueError(
                f"functional check {check_id} requires a non-empty reason"
            )
        if observation_status == "missing" and conclusion != "unresolved":
            raise ValueError(
                f"missing functional check {check_id} must be unresolved"
            )
        if observation_status != "missing" and conclusion == "unresolved":
            raise ValueError(
                f"resolved observation for {check_id} requires a binary conclusion"
            )
        if conclusion == "unresolved":
            unresolved.append(check_id)
        else:
            resolved.append(check_id)
        if conclusion == "invalid":
            if check.get("check_type") == "clearance":
                _validate_clearance_causal_attribution(
                    row,
                    check=check,
                )
            invalid.append(check_id)
        normalized_rows.append(deepcopy(row))

    verdict = result.get("verdict")
    if verdict not in {"valid", "invalid", "ambiguous"}:
        raise ValueError(
            "functional Judge verdict must be valid, invalid, or ambiguous"
        )
    if verdict == "valid" and (unresolved or invalid):
        raise ValueError(
            "functional valid verdict requires every required check to resolve valid"
        )
    if (
        verdict == "invalid"
        and invalid_verdict_requires_invalid_check
        and not invalid
    ):
        raise ValueError(
            "this functional Judge phase requires an invalid verdict to map "
            "to an invalid functional check"
        )
    if verdict == "ambiguous" and not unresolved:
        raise ValueError(
            "functional ambiguous verdict requires an unresolved functional check"
        )
    if verdict == "ambiguous" and invalid:
        raise ValueError(
            "functional ambiguous verdict cannot retain an invalid check; "
            "a supported invalid check is already a final defect"
        )
    defects = result.get("defects") or []
    if not isinstance(defects, list) or any(
        not isinstance(defect, dict) for defect in defects
    ):
        raise ValueError("functional defects must be a JSON list of objects")
    linked_defects: dict[str, dict[str, Any]] = {}
    for defect in defects:
        raw_refs = defect.get("check_refs")
        if raw_refs is None:
            # A baseline Judge may still discover a final Functional defect
            # that did not originate from a routed typed check.  Such a defect
            # remains legal, but it cannot close any required-check lifecycle.
            continue
        if (
            not isinstance(raw_refs, list)
            or not raw_refs
            or any(
                not isinstance(item, str) or not item.strip()
                for item in raw_refs
            )
        ):
            raise ValueError(
                "functional defect check_refs must contain check IDs"
            )
        check_refs = [str(item).strip() for item in raw_refs]
        if len(check_refs) != len(set(check_refs)):
            raise ValueError(
                "functional defect check_refs cannot contain duplicates"
            )
        for check_id in check_refs:
            if check_id not in expected:
                raise ValueError(
                    "functional defect references unknown check "
                    f"{check_id!r}"
                )
            if check_id not in invalid:
                raise ValueError(
                    "functional defect may reference only an invalid check: "
                    f"{check_id!r}"
                )
            if check_id in linked_defects:
                raise ValueError(
                    "invalid functional check is linked to more than one "
                    f"defect: {check_id!r}"
                )
            linked_defects[check_id] = defect
    for check_id in invalid:
        check = expected[check_id]
        defect = linked_defects.get(check_id)
        if defect is None:
            raise ValueError(
                f"invalid functional check {check_id} requires one explicit "
                "defect check_refs linkage"
            )
        defect_targets = {
            str(target_id)
            for target_id in defect.get("target_ids") or []
            if str(target_id).strip()
        }
        if check.get("check_type") == "clearance":
            check_targets = {
                str(target_id)
                for target_id in row_by_check_id(rows)[check_id].get(
                    "scoring_target_ids"
                )
                or []
            }
            exact_match_required = True
        else:
            check_targets = {
                str(target_id)
                for target_id in check.get("target_ids") or []
            }
            exact_match_required = False
        if not defect_targets or not (
            defect_targets == check_targets
            if exact_match_required
            else defect_targets <= check_targets
        ):
            raise ValueError(
                f"invalid functional check {check_id} requires object-level "
                "defect attribution within its target scope"
            )
    return {
        "schema_version": FUNCTIONAL_CHECK_RESULT_VERSION,
        "required_check_count": len(required_checks),
        "resolved_check_ids": resolved,
        "unresolved_check_ids": unresolved,
        "invalid_check_ids": invalid,
        "complete": not unresolved,
        "rows": normalized_rows,
        "decision_authority": "none",
    }


def canonicalize_functional_defect_check_linkage(
    result: dict[str, Any],
    *,
    required_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Split a safe multi-check union defect into atomic check linkages.

    Models sometimes describe one issue with the union of several atomic
    check scopes. That shape is deterministic to normalize when every
    referenced check is known, invalid, non-clearance, and intersects the
    returned target set. Unsafe shapes remain untouched so validation still
    fails closed.
    """

    if not isinstance(result, dict) or not required_checks:
        return result
    defects = result.get("defects")
    rows = result.get("functional_check_results")
    if not isinstance(defects, list) or not isinstance(rows, list):
        return result
    checks = {
        str(check.get("check_id") or ""): check
        for check in required_checks
        if isinstance(check, dict) and str(check.get("check_id") or "")
    }
    invalid_ids = {
        str(row.get("check_id") or "")
        for row in rows
        if isinstance(row, dict) and row.get("conclusion") == "invalid"
    }
    changed = False
    normalized_defects: list[Any] = []
    for defect in defects:
        if not isinstance(defect, dict):
            normalized_defects.append(defect)
            continue
        raw_refs = defect.get("check_refs")
        refs = (
            list(dict.fromkeys(str(item).strip() for item in raw_refs))
            if isinstance(raw_refs, list)
            and all(isinstance(item, str) and item.strip() for item in raw_refs)
            else []
        )
        defect_target_set = {
            str(item)
            for item in defect.get("target_ids") or []
            if str(item).strip()
        }
        referenced_checks = [checks.get(check_id) for check_id in refs]
        safe = bool(
            len(refs) > 1
            and defect_target_set
            and all(check is not None for check in referenced_checks)
            and set(refs) <= invalid_ids
            and all(
                check.get("check_type") != "clearance"
                for check in referenced_checks
                if check is not None
            )
        )
        check_target_sets = [
            {
                str(item)
                for item in check.get("target_ids") or []
                if str(item).strip()
            }
            for check in referenced_checks
            if check is not None
        ]
        if safe:
            safe = bool(
                check_target_sets
                and defect_target_set
                <= set().union(*check_target_sets)
                and all(
                    defect_target_set & target_set
                    for target_set in check_target_sets
                )
            )
        if not safe or all(
            defect_target_set <= target_set
            for target_set in check_target_sets
        ):
            normalized_defects.append(deepcopy(defect))
            continue
        for check_id, check in zip(refs, referenced_checks):
            assert check is not None
            atomic = deepcopy(defect)
            atomic["check_refs"] = [check_id]
            atomic["target_ids"] = [
                str(item)
                for item in check.get("target_ids") or []
                if str(item) in defect_target_set
            ]
            normalized_defects.append(atomic)
        changed = True
    if not changed:
        return result
    normalized = deepcopy(result)
    normalized["defects"] = normalized_defects
    return normalized


def canonicalize_typed_invalid_envelope(
    result: dict[str, Any],
) -> dict[str, Any]:
    """Derive the metric envelope from an asserted final typed defect.

    Exact defect/check linkage is still validated separately. This removes
    only a redundant schema contradiction: a response cannot remain
    ``ambiguous`` once it asserts a final defect, even if other required
    checks are unresolved.
    """

    defects = result.get("defects")
    if not isinstance(defects, list) or not defects:
        return result
    invalid_check_ids = {
        str(row.get("check_id") or row.get("proposal_id") or "")
        for field in (
            "functional_check_results",
            "placement_check_results",
            "judge_originated_placement_results",
        )
        for row in result.get(field) or []
        if isinstance(row, dict)
        and row.get("conclusion") == "invalid"
        and str(row.get("check_id") or row.get("proposal_id") or "").strip()
    }
    if not invalid_check_ids:
        return result
    linked_invalid_ids = {
        check_id
        for defect in defects
        if isinstance(defect, dict)
        for check_id in [
            *[
                str(item)
                for item in defect.get("check_refs") or []
                if str(item).strip()
            ],
            *(
                [str(defect.get("check_id"))]
                if str(defect.get("check_id") or "").strip()
                else []
            ),
        ]
        if check_id in invalid_check_ids
    }
    if not linked_invalid_ids:
        return result
    normalized = deepcopy(result)
    normalized["verdict"] = "invalid"
    normalized["evidence_status"] = "sufficient"
    normalized["missing_evidence"] = []
    normalized["evidence_request"] = None
    return normalized


def apply_functional_check_judgements(
    ledger: dict[str, Any],
    *,
    relation_results: list[dict[str, Any]],
    group_results: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge Judge acknowledgements back into the audit-only lifecycle."""

    result = deepcopy(ledger)
    checks_by_id = {
        str(check.get("check_id") or ""): check
        for check in result.get("checks") or []
        if isinstance(check, dict) and check.get("check_id")
    }
    rows_by_id: dict[str, tuple[dict[str, Any], str]] = {}
    for record, phase in [
        *[
            (
                item,
                "cross_group_relation_review:"
                f"{item.get('relation_id')}",
            )
            for item in relation_results
            if isinstance(item, dict)
        ],
        *[
            (
                item,
                f"group_local_review:{item.get('group_id')}",
            )
            for item in group_results
            if isinstance(item, dict)
        ],
    ]:
        judgement = (
            record.get("judgement")
            if isinstance(record.get("judgement"), dict)
            else {}
        )
        for row in judgement.get("functional_check_results") or []:
            if not isinstance(row, dict):
                continue
            check_id = str(row.get("check_id") or "")
            if check_id not in checks_by_id:
                raise ValueError(
                    f"Judge returned unknown functional check {check_id!r}"
                )
            if check_id in rows_by_id:
                raise ValueError(
                    f"functional check {check_id!r} was judged more than once"
                )
            rows_by_id[check_id] = (row, phase)

    for check_id, check in checks_by_id.items():
        routed = rows_by_id.get(check_id)
        if routed is None:
            check["judge_status"] = "pending"
            check["lifecycle_status"] = "routed"
            continue
        row, phase = routed
        observation_status = str(row.get("observation_status") or "")
        conclusion = str(row.get("conclusion") or "")
        check["judge_status"] = (
            "resolved" if conclusion in {"valid", "invalid"} else "unresolved"
        )
        check["judge_result_ref"] = phase
        check["observation_complete"] = (
            observation_status in {"observed", "inferred_under_budget"}
        )
        check["observation_status"] = observation_status
        check["check_conclusion"] = conclusion
        check["result_row"] = deepcopy(row)
        if check.get("check_type") == "clearance" and conclusion == "invalid":
            for field in (
                "affected_object_ids",
                "cause_kind",
                "causal_object_ids",
                "scoring_target_ids",
            ):
                check[field] = deepcopy(row[field])
        check["lifecycle_status"] = (
            "resolved"
            if conclusion in {"valid", "invalid"}
            else "unresolved"
        )

    resolved_ids = [
        check_id
        for check_id, check in checks_by_id.items()
        if check.get("lifecycle_status") == "resolved"
    ]
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
    coverage = {
        "schema_version": FUNCTIONAL_CHECK_RESULT_VERSION,
        "required_check_count": len(checks_by_id),
        "resolved_check_count": len(resolved_ids),
        "resolved_check_ids": resolved_ids,
        "unresolved_check_ids": unresolved_ids,
        "invalid_check_ids": invalid_ids,
        "complete": not unresolved_ids,
        "decision_authority": "none",
    }
    return result, coverage


def check_ids_for_discovery(
    ledger: dict[str, Any],
    discovery_ids: list[str],
) -> list[str]:
    source_ids = {str(item) for item in discovery_ids if str(item).strip()}
    return [
        str(check["check_id"])
        for check in ledger.get("checks") or []
        if source_ids
        & {
            str(item)
            for item in (
                check.get("routing_discovery_ids")
                or check.get("source_discovery_ids")
                or []
            )
        }
    ]


def forced_group_ids_from_checks(
    ledger: dict[str, Any] | None,
) -> list[str]:
    return sorted(
        {
            str(check.get("owning_group_id"))
            for check in (ledger or {}).get("checks") or []
            if check.get("owner_stage") == "group_local"
            and check.get("owning_group_id")
        }
    )


def _known_object_ids(discovery: dict[str, Any]) -> set[str]:
    inspected = [
        str(item)
        for item in discovery.get("inspected_object_ids") or []
        if str(item).strip()
    ]
    if len(inspected) != len(set(inspected)):
        raise ValueError("functional discovery object IDs must be unique")
    if inspected:
        return set(inspected)
    return {
        str(target_id)
        for field in (
            "directed_surface_targets",
            "approach_clearance_targets",
            "boundary_sensitive_targets",
            "within_group_correspondences",
            "cross_group_correspondences",
            "unusual_unconfirmed",
        )
        for item in discovery.get(field) or []
        if isinstance(item, dict)
        for target_id in (
            item.get("target_ids")
            or ([item.get("target_id")] if item.get("target_id") else [])
        )
        if str(target_id).strip()
    }


def _clearance_causal_candidates(
    *,
    target_id: str,
    known_ids: set[str],
    scene: dict[str, Any] | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Return a stable, routing-only blocker shortlist.

    The shortlist never limits Judge authority.  It only prioritizes nearby
    XY footprints so a bounded prompt can expose likely external blockers.
    """

    geometry = _scene_xy_geometry(scene)
    target = geometry.get(target_id)
    ranked: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for object_id in sorted(known_ids - {target_id}):
        candidate = geometry.get(object_id)
        if target is not None and candidate is not None:
            distance = _xy_aabb_gap(target, candidate)
            sort_key = (0, distance, object_id)
            record = {
                "object_id": object_id,
                "xy_aabb_gap": round(distance, 6),
                "geometry_status": "available",
            }
        else:
            sort_key = (1, math.inf, object_id)
            record = {
                "object_id": object_id,
                "xy_aabb_gap": None,
                "geometry_status": "unavailable",
            }
        ranked.append((sort_key, record))
    ranked.sort(key=lambda item: item[0])
    return [
        deepcopy(record)
        for _, record in ranked[: max(0, int(limit))]
    ]


def _scene_xy_geometry(
    scene: dict[str, Any] | None,
) -> dict[str, tuple[float, float, float, float]]:
    result: dict[str, tuple[float, float, float, float]] = {}
    for item in (scene or {}).get("objects") or []:
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("id") or "").strip()
        center = item.get("center")
        size = item.get("size")
        if not isinstance(size, (list, tuple)):
            proxy = (
                item.get("asset_proxy")
                if isinstance(item.get("asset_proxy"), dict)
                else {}
            )
            size = proxy.get("bbox_size")
        if (
            not object_id
            or not isinstance(center, (list, tuple))
            or not isinstance(size, (list, tuple))
            or len(center) < 2
            or len(size) < 2
        ):
            continue
        try:
            cx, cy = float(center[0]), float(center[1])
            sx, sy = abs(float(size[0])), abs(float(size[1]))
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (cx, cy, sx, sy)):
            continue
        result[object_id] = (cx, cy, sx, sy)
    return result


def _xy_aabb_gap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    dx = max(
        0.0,
        abs(left[0] - right[0]) - (left[2] + right[2]) / 2.0,
    )
    dy = max(
        0.0,
        abs(left[1] - right[1]) - (left[3] + right[3]) / 2.0,
    )
    return math.hypot(dx, dy)


def _validate_clearance_causal_attribution(
    row: dict[str, Any],
    *,
    check: dict[str, Any],
) -> None:
    affected = _strict_result_ids(
        row.get("affected_object_ids"),
        label="affected_object_ids",
    )
    expected_affected = sorted(
        str(item) for item in check.get("target_ids") or []
    )
    if affected != expected_affected:
        raise ValueError(
            f"clearance check {check['check_id']} must attribute the exact "
            "affected object set"
        )
    cause_kind = str(row.get("cause_kind") or "")
    if cause_kind not in {"external_object", "self_layout"}:
        raise ValueError(
            f"clearance check {check['check_id']} requires cause_kind "
            "external_object or self_layout"
        )
    causal = _strict_result_ids(
        row.get("causal_object_ids"),
        label="causal_object_ids",
    )
    scoring = _strict_result_ids(
        row.get("scoring_target_ids"),
        label="scoring_target_ids",
    )
    known = {
        str(item)
        for item in (
            check.get("allowed_causal_object_ids")
            or check.get("target_ids")
            or []
        )
    }
    unknown = sorted(set(causal) - known)
    if unknown:
        raise ValueError(
            f"clearance check {check['check_id']} references unknown causal "
            f"objects {unknown}"
        )
    if cause_kind == "self_layout":
        if causal != affected or scoring != affected:
            raise ValueError(
                f"self-layout clearance check {check['check_id']} must score "
                "the affected object itself"
            )
        return
    if not causal or set(causal) & set(affected):
        raise ValueError(
            f"external-object clearance check {check['check_id']} requires "
            "non-self causal objects"
        )
    if scoring != causal:
        raise ValueError(
            f"external-object clearance check {check['check_id']} must score "
            "the causal blocker set"
        )


def _strict_result_ids(value: Any, *, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"{label} must contain non-empty object IDs")
    normalized = sorted(str(item).strip() for item in value)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} contains duplicate object IDs")
    return normalized


def row_by_check_id(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("check_id") or ""): row
        for row in rows
        if isinstance(row, dict)
    }


def _trusted_group_partition(
    groups: list[dict[str, Any]] | None,
    *,
    known_ids: set[str],
) -> tuple[dict[str, str], set[str]]:
    object_to_group: dict[str, str] = {}
    group_ids: set[str] = set()
    for group in groups or []:
        if not isinstance(group, dict):
            raise TypeError("functional groups must contain JSON objects")
        group_id = str(group.get("group_id") or "").strip()
        if not group_id or group_id in group_ids:
            raise ValueError("functional groups require unique non-empty IDs")
        group_ids.add(group_id)
        for object_id in group.get("object_ids") or []:
            normalized = str(object_id)
            if known_ids and normalized not in known_ids:
                raise ValueError(
                    f"functional group contains unknown object ID {normalized!r}"
                )
            if normalized in object_to_group:
                raise ValueError(
                    f"functional object {normalized!r} belongs to multiple groups"
                )
            object_to_group[normalized] = group_id
    return object_to_group, group_ids


def _validated_targets(
    values: list[str],
    *,
    known_ids: set[str],
) -> list[str]:
    result = [str(item) for item in values if str(item).strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("functional check targets must be non-empty and unique")
    unknown = sorted(set(result) - known_ids) if known_ids else []
    if unknown:
        raise ValueError(f"functional check references unknown targets {unknown}")
    return result


def _validated_group_scope(
    target_ids: list[str],
    *,
    declared_group_ids: list[str],
    declared_owner: str | None,
    owner_stage: str,
    object_to_group: dict[str, str],
    groups_by_id: set[str],
) -> list[str]:
    trusted = list(
        dict.fromkeys(
            object_to_group.get(object_id)
            for object_id in target_ids
            if object_to_group.get(object_id)
        )
    )
    declared = [
        str(item) for item in declared_group_ids if str(item).strip()
    ]
    if groups_by_id:
        if len(trusted) != len(
            {
                object_to_group.get(object_id)
                for object_id in target_ids
            }
        ) or any(object_id not in object_to_group for object_id in target_ids):
            raise ValueError(
                "every functional check target requires trusted group ownership"
            )
        if declared and set(declared) != set(trusted):
            raise ValueError(
                "functional check group scope conflicts with trusted partition"
            )
        if declared_owner and declared_owner not in trusted:
            raise ValueError(
                "functional check owner conflicts with trusted partition"
            )
    else:
        trusted = declared or ([declared_owner] if declared_owner else [])
        if not trusted and owner_stage == "cross_group_relation":
            trusted = [
                f"legacy_unscoped_group:{object_id}"
                for object_id in target_ids
            ]
        elif not trusted and owner_stage == "group_local":
            trusted = ["legacy_unscoped_group"]
    if owner_stage == "cross_group_relation" and len(trusted) < 2:
        raise ValueError(
            "cross-group functional check must span at least two groups"
        )
    if owner_stage == "group_local" and len(trusted) != 1:
        raise ValueError(
            "group-local functional check requires exactly one owning group"
        )
    return trusted


def _required_discovery_id(item: dict[str, Any]) -> str:
    value = str(item.get("discovery_id") or "").strip()
    if value:
        return value
    # Additive read compatibility for early planner fixtures and frozen traces.
    # Current validated discovery always supplies a real discovery_id.
    targets = item.get("target_ids") or [
        item.get("target_id")
    ]
    normalized = "_".join(
        sorted(str(target) for target in targets if str(target).strip())
    )
    if not normalized:
        raise ValueError("functional discovery record requires discovery_id")
    return f"legacy_discovery:{normalized}"


def _observation_goal(item: dict[str, Any]) -> str:
    value = str(item.get("observation_goal") or "").strip()
    if not value:
        raise ValueError("functional discovery record requires observation_goal")
    return value[:1000]


def _require_record(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")


def _optional_text(value: Any) -> str | None:
    result = str(value or "").strip()
    return result or None


def _stable_unique(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        marker = repr(value)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result
