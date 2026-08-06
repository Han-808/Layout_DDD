"""Deterministic routing from functional discovery to camera probe units."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


FUNCTIONAL_ACQUISITION_PLAN_VERSION = "functional_acquisition_plan_v3"

_FRONTAGE_OBSERVATIONS = (
    "target_visible",
    "interaction_side_visible",
    "front_back_disambiguated",
    "approach_zone_visible",
    "limited_local_context",
)
_BOUNDARY_OBSERVATIONS = (
    "target_visible",
    "interaction_side_visible",
    "front_back_disambiguated",
    "approach_zone_visible",
    "architecture_plane_visible",
    "global_context_preserved",
    "limited_local_context",
)
_CORRESPONDENCE_OBSERVATIONS = (
    "target_visible",
    "joint_visibility",
    "interaction_side_visible",
    "front_back_disambiguated",
    "approach_zone_visible",
    "group_context_visible",
    "limited_local_context",
)
_CLEARANCE_OBSERVATIONS = (
    "target_visible",
    "interaction_side_visible",
    "approach_zone_visible",
    "limited_local_context",
)


def build_functional_acquisition_plan(
    discovery: dict[str, Any],
    *,
    max_probe_units: int,
    max_probe_units_source: str = "caller",
) -> dict[str, Any]:
    """Create a bounded plan without inventing new metric requirements."""

    limit = max(0, int(max_probe_units))
    directed = {
        str(item.get("target_id")): deepcopy(item)
        for item in discovery.get("directed_surface_targets") or []
        if isinstance(item, dict) and item.get("target_id")
    }
    boundary = {
        str(item.get("target_id")): deepcopy(item)
        for item in discovery.get("boundary_sensitive_targets") or []
        if isinstance(item, dict) and item.get("target_id")
    }
    approach = {
        str(item.get("target_id")): deepcopy(item)
        for item in discovery.get("approach_clearance_targets") or []
        if isinstance(item, dict) and item.get("target_id")
    }

    candidates: list[dict[str, Any]] = []
    sequence = 0

    def add(
        *,
        tier: int,
        kind: str,
        target_ids: list[str],
        route_scope: str,
        owning_group_id: str | None,
        observations: tuple[str, ...],
        view_goal: str,
        discovery_ids: list[str],
        trigger: str,
        audit_reason: str | None = None,
    ) -> None:
        nonlocal sequence
        sequence += 1
        surface_targets = [
            _surface_target_record(
                object_id,
                directed[object_id],
            )
            for object_id in target_ids
            if object_id in directed
        ]
        candidates.append(
            {
                "_tier": tier,
                "_sequence": sequence,
                "kind": kind,
                "target_ids": target_ids[:1],
                "related_target_ids": target_ids[1:],
                "required_observations": list(observations),
                "reason": str(view_goal)[:1000],
                "view_goal": str(view_goal)[:1000],
                "observation_goals": [str(view_goal)[:1000]],
                "route_scope": route_scope,
                "owning_group_id": owning_group_id,
                "surface_targets": surface_targets,
                "discovery_ids": list(discovery_ids),
                "observation_kinds": [],
                "acquisition_trigger": trigger,
                "acquisition_triggers": [trigger],
                "_audit_reasons": (
                    [str(audit_reason)[:1000]]
                    if audit_reason
                    else []
                ),
                "_group_confirmations": (
                    [
                        {
                            "neutral_observation_goal": str(view_goal)[
                                :1000
                            ],
                            "audit_reason": str(audit_reason or "")[
                                :1000
                            ]
                            or None,
                            "discovery_ids": list(discovery_ids),
                        }
                    ]
                    if trigger
                    == "unusual_unconfirmed_group_confirmation"
                    else []
                ),
            }
        )

    for item in discovery.get("cross_group_correspondences") or []:
        if not isinstance(item, dict):
            continue
        add(
            tier=0,
            kind="functional_correspondence",
            target_ids=[str(value) for value in item["target_ids"]],
            route_scope="cross_group",
            owning_group_id=None,
            observations=_CORRESPONDENCE_OBSERVATIONS,
            view_goal=str(item["observation_goal"]),
            discovery_ids=[str(item.get("discovery_id") or "")],
            trigger="cross_group_correspondence",
        )
        candidates[-1]["observation_kinds"] = list(
            item.get("observation_kinds") or []
        )

    for item in discovery.get("within_group_correspondences") or []:
        if not isinstance(item, dict):
            continue
        add(
            tier=0,
            kind="functional_correspondence",
            target_ids=[str(value) for value in item["target_ids"]],
            route_scope="group_local",
            owning_group_id=(
                str(item["group_ids"][0])
                if item.get("group_ids")
                else None
            ),
            observations=_CORRESPONDENCE_OBSERVATIONS,
            view_goal=str(item["observation_goal"]),
            discovery_ids=[str(item.get("discovery_id") or "")],
            trigger="within_group_correspondence",
        )
        candidates[-1]["observation_kinds"] = list(
            item.get("observation_kinds") or []
        )

    for object_id, item in boundary.items():
        add(
            tier=1,
            kind="functional_frontage",
            target_ids=[object_id],
            route_scope="group_local",
            owning_group_id=_optional_text(item.get("owning_group_id")),
            observations=_BOUNDARY_OBSERVATIONS,
            view_goal=str(item["observation_goal"]),
            discovery_ids=[
                str(item.get("discovery_id") or ""),
                *(
                    [str(directed[object_id].get("discovery_id") or "")]
                    if object_id in directed
                    else []
                ),
            ],
            trigger="boundary_sensitive_frontage",
        )

    for object_id, item in approach.items():
        add(
            tier=2,
            kind="approach_clearance",
            target_ids=[object_id],
            route_scope="group_local",
            owning_group_id=_optional_text(item.get("owning_group_id")),
            observations=_CLEARANCE_OBSERVATIONS,
            view_goal=str(item["observation_goal"]),
            discovery_ids=[str(item.get("discovery_id") or "")],
            trigger="approach_clearance",
        )

    for item in discovery.get("unusual_unconfirmed") or []:
        if not isinstance(item, dict):
            continue
        target_ids = [str(value) for value in item["target_ids"]]
        add(
            tier=3,
            kind=(
                "functional_correspondence"
                if len(target_ids) > 1
                else "functional_frontage"
            ),
            target_ids=target_ids,
            route_scope="group_local",
            owning_group_id=str(item["owning_group_id"]),
            observations=(
                _CORRESPONDENCE_OBSERVATIONS
                if len(target_ids) > 1
                else _FRONTAGE_OBSERVATIONS
            ),
            view_goal=str(item["observation_goal"]),
            discovery_ids=[str(item.get("discovery_id") or "")],
            trigger="unusual_unconfirmed_group_confirmation",
            audit_reason=str(item.get("audit_reason") or ""),
        )

    # One camera unit may satisfy multiple discovery records only when their
    # target set and routing ownership are exactly the same.  This prevents a
    # boundary, clearance, and neutral confirmation request for one object
    # from spending three images, while never merging unrelated group members.
    candidates = _merge_exact_target_candidates(candidates)

    # Routine directed affordances enrich relation/boundary/clearance units
    # through ``surface_targets`` above; they never consume a unit by
    # themselves.  This keeps the bounded budget focused on an observable need.
    for candidate in candidates:
        all_ids = [
            *candidate["target_ids"],
            *candidate["related_target_ids"],
        ]
        candidate["surface_targets"] = [
            _surface_target_record(
                object_id,
                directed[object_id],
            )
            for object_id in all_ids
            if object_id in directed
        ]
        candidate["_identity"] = _acquisition_identity(candidate)

    candidates.sort(
        key=lambda item: (item["_tier"], item["_identity"])
    )
    selected: list[dict[str, Any]] = []
    unscheduled: list[dict[str, Any]] = []
    seen_identities: set[
        tuple[str, tuple[str, ...], tuple[str, ...]]
    ] = set()
    for candidate in candidates:
        identity = candidate["_identity"]
        if identity in seen_identities:
            unscheduled.append(
                _unscheduled(candidate, "duplicate_acquisition_identity")
            )
            continue
        seen_identities.add(identity)
        if len(selected) >= limit:
            unscheduled.append(
                _unscheduled(candidate, "max_probe_units_exhausted")
            )
            continue
        selected.append(candidate)

    group_confirmations: list[dict[str, Any]] = []
    for candidate in selected:
        candidate.pop("_tier", None)
        candidate.pop("_sequence", None)
        candidate.pop("_identity", None)
        audit_reasons = candidate.pop("_audit_reasons", [])
        confirmation_records = candidate.pop(
            "_group_confirmations",
            [],
        )
        if (
            "unusual_unconfirmed_group_confirmation"
            in candidate.get("acquisition_triggers", [])
        ):
            for confirmation in confirmation_records or [{}]:
                group_confirmations.append(
                    {
                        "group_id": candidate["owning_group_id"],
                        "target_ids": [
                            *candidate["target_ids"],
                            *candidate["related_target_ids"],
                        ],
                        "neutral_observation_goal": str(
                            confirmation.get(
                                "neutral_observation_goal"
                            )
                            or candidate["view_goal"]
                        ),
                        "audit_reason": (
                            confirmation.get("audit_reason")
                            or (
                                audit_reasons[0]
                                if audit_reasons
                                else None
                            )
                        ),
                        "audit_reasons": list(audit_reasons),
                        "discovery_ids": list(
                            confirmation.get("discovery_ids")
                            or candidate["discovery_ids"]
                        ),
                        "decision_authority": "none",
                    }
                )
    for priority, candidate in enumerate(selected, start=1):
        candidate["priority"] = priority
        candidate["probe_id"] = f"functional_probe_{priority:02d}"

    return {
        "schema_version": FUNCTIONAL_ACQUISITION_PLAN_VERSION,
        "max_probe_units": limit,
        "budget": {
            "max_probe_units": {
                "requested": int(max_probe_units),
                "effective": limit,
                "source": str(max_probe_units_source),
            }
        },
        "probe_units": selected,
        "group_confirmations": group_confirmations,
        "unscheduled_discovery_items": unscheduled,
        "scheduled_probe_count": len(selected),
        "coverage_complete": not any(
            item.get("reason") == "max_probe_units_exhausted"
            for item in unscheduled
        ),
        "budget_exhausted": any(
            item.get("reason") == "max_probe_units_exhausted"
            for item in unscheduled
        ),
        "decision_authority": "none",
    }


def _surface_target_record(
    object_id: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "target_id": object_id,
        "surface_roles": list(record.get("surface_roles") or []),
    }
    hypothesis = record.get(
        "precomputed_usable_surface_hypothesis"
    )
    if isinstance(hypothesis, dict):
        result["precomputed_hypothesis"] = deepcopy(hypothesis)
    return result


def _acquisition_identity(
    candidate: dict[str, Any],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    return (
        str(candidate.get("kind") or ""),
        tuple(
            sorted(
                {
                    *[
                        str(item)
                        for item in candidate.get("target_ids") or []
                    ],
                    *[
                        str(item)
                        for item in candidate.get("related_target_ids") or []
                    ],
                }
            )
        ),
        tuple(
            sorted(
                str(item)
                for item in candidate.get("required_observations") or []
            )
        ),
    )


def _merge_exact_target_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Enrich, but never broaden, an already-triggered camera unit."""

    merged: list[dict[str, Any]] = []
    by_scope_and_targets: dict[
        tuple[str, str | None, tuple[str, ...]],
        dict[str, Any],
    ] = {}
    for candidate in candidates:
        target_ids = tuple(
            sorted(
                {
                    *[
                        str(item)
                        for item in candidate.get("target_ids") or []
                    ],
                    *[
                        str(item)
                        for item in candidate.get(
                            "related_target_ids"
                        )
                        or []
                    ],
                }
            )
        )
        key = (
            str(candidate.get("route_scope") or ""),
            _optional_text(candidate.get("owning_group_id")),
            target_ids,
        )
        existing = by_scope_and_targets.get(key)
        if existing is None:
            by_scope_and_targets[key] = candidate
            merged.append(candidate)
            continue

        existing["_tier"] = min(
            int(existing.get("_tier") or 0),
            int(candidate.get("_tier") or 0),
        )
        for list_key in (
            "required_observations",
            "observation_goals",
            "discovery_ids",
            "observation_kinds",
            "acquisition_triggers",
            "_audit_reasons",
            "_group_confirmations",
        ):
            existing[list_key] = _stable_unique(
                [
                    *list(existing.get(list_key) or []),
                    *list(candidate.get(list_key) or []),
                ]
            )
        # The highest-priority record remains the primary label and goal.
        # The complete list above preserves every acquisition need for audit
        # and for downstream prompt construction.
    return merged


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


def _unscheduled(
    candidate: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    identity = _acquisition_identity(candidate)
    return {
        "acquisition_identity": [
            identity[0],
            list(identity[1]),
            list(identity[2]),
        ],
        "discovery_ids": list(candidate.get("discovery_ids") or []),
        "target_ids": [
            *candidate.get("target_ids", []),
            *candidate.get("related_target_ids", []),
        ],
        "route_scope": candidate.get("route_scope"),
        "owning_group_id": candidate.get("owning_group_id"),
        "acquisition_trigger": candidate.get("acquisition_trigger"),
        "observation_goal": str(
            candidate.get("view_goal") or candidate.get("reason") or ""
        ),
        "reason": reason,
    }


def _optional_text(value: Any) -> str | None:
    result = str(value or "").strip()
    return result or None
