"""Deterministic routing from functional discovery to camera probe units."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from benchmark.evaluator.scene_quality.functional_checks import (
    build_functional_check_ledger,
    check_ids_for_discovery,
    functional_relation_required_observations,
    route_functional_check_ledger,
)
from benchmark.evaluator.scene_quality.functional_clearance import (
    apply_directional_clearance_profiles_to_ledger,
    build_directional_clearance_extensions,
)
from benchmark.evaluator.scene_quality.functional_measurements import (
    attach_functional_measurement_extension,
    build_functional_measurement_bank,
    compact_functional_measurements_for_checks,
)
from benchmark.visual_judge.functional_discovery_contract import (
    normalized_functional_relation_predicates,
)
from benchmark.visual_judge.functional_evidence import (
    FUNCTIONAL_PROBE_MAX_UNITS,
)

FUNCTIONAL_ACQUISITION_PLAN_VERSION = "functional_acquisition_plan_v12"

_FRONTAGE_OBSERVATIONS = (
    "target_visible",
    "interaction_side_visible",
    "front_back_disambiguated",
    "limited_local_context",
)
_BOUNDARY_OBSERVATIONS = (
    "target_visible",
    "interaction_side_visible",
    "front_back_disambiguated",
    "architecture_plane_visible",
    "global_context_preserved",
    "limited_local_context",
)
_LOCAL_MULTI_TARGET_CONFIRMATION_OBSERVATIONS = (
    "target_visible",
    "joint_visibility",
    "interaction_side_visible",
    "front_back_disambiguated",
    "group_context_visible",
    "limited_local_context",
)
_CLEARANCE_OBSERVATIONS = (
    "target_visible",
    "interaction_side_visible",
    "approach_zone_visible",
    "limited_local_context",
)
_DIRECTION_INDEPENDENT_CLEARANCE_OBSERVATIONS = (
    "target_visible",
    "approach_zone_visible",
    "limited_local_context",
)
_DIRECTION_INDEPENDENT_BOUNDARY_OBSERVATIONS = (
    "target_visible",
    "approach_zone_visible",
    "architecture_plane_visible",
    "global_context_preserved",
    "limited_local_context",
)


def build_functional_acquisition_plan(
    discovery: dict[str, Any],
    *,
    max_probe_units: int,
    max_probe_units_source: str = "caller",
    groups: list[dict[str, Any]] | None = None,
    scene: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a bounded plan without inventing new metric requirements."""

    requested_limit = max(0, int(max_probe_units))
    limit = min(requested_limit, FUNCTIONAL_PROBE_MAX_UNITS)
    functional_check_ledger = build_functional_check_ledger(
        discovery,
        groups=groups,
        scene=scene,
    )
    # Measurements belong to accepted checks, not rendered probes. Build one
    # deterministic record per accepted check before candidate construction or
    # camera-budget scheduling so an unscheduled check retains the same spatial
    # facts as a scheduled one.
    functional_measurement_bank = build_functional_measurement_bank(
        scene=scene,
        functional_check_ledger=functional_check_ledger,
        discovery=discovery,
    )
    directional_clearance_extensions = (
        build_directional_clearance_extensions(
            scene=scene,
            discovery=discovery,
            functional_check_ledger=functional_check_ledger,
        )
    )
    functional_measurement_bank = attach_functional_measurement_extension(
        functional_measurement_bank,
        namespace="directional_clearance",
        by_check_id=directional_clearance_extensions,
        source="deterministic_usable_side_forward_corridor_v1",
    )
    functional_check_ledger = apply_directional_clearance_profiles_to_ledger(
        functional_check_ledger,
        by_check_id=directional_clearance_extensions,
    )
    check_type_by_id = {
        str(item.get("check_id") or ""): str(
            item.get("check_type") or ""
        )
        for item in functional_check_ledger.get("checks") or []
        if isinstance(item, dict) and item.get("check_id")
    }
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
        check_family: str,
        audit_reason: str | None = None,
        include_surface_targets: bool = True,
    ) -> None:
        nonlocal sequence
        sequence += 1
        surface_targets = (
            [
                _surface_target_record(
                    object_id,
                    directed[object_id],
                )
                for object_id in target_ids
                if object_id in directed
            ]
            if include_surface_targets
            else []
        )
        check_ids = check_ids_for_discovery(
            functional_check_ledger,
            discovery_ids,
        )
        if not check_ids:
            raise ValueError(
                "accepted functional discovery did not map to a downstream "
                f"check: trigger={trigger!r}, discovery_ids={discovery_ids!r}"
            )
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
                "check_ids": check_ids,
                "functional_measurements": (
                    compact_functional_measurements_for_checks(
                        functional_measurement_bank,
                        check_ids,
                    )
                ),
                "check_types": _stable_unique(
                    [
                        check_type_by_id[check_id]
                        for check_id in check_ids
                        if check_id in check_type_by_id
                    ]
                ),
                "_check_family": check_family,
                "observation_kinds": [],
                "relation_predicates": [],
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
        for predicate in normalized_functional_relation_predicates(item):
            add(
                tier=0,
                kind="functional_correspondence",
                target_ids=[str(value) for value in item["target_ids"]],
                route_scope="cross_group",
                owning_group_id=None,
                observations=functional_relation_required_observations(
                    predicate,
                    acquisition=True,
                ),
                view_goal=str(item["observation_goal"]),
                discovery_ids=[_stable_discovery_id(item)],
                trigger="cross_group_correspondence",
                check_family="cross_group_correspondence",
                include_surface_targets=(
                    predicate == "directional_correspondence"
                ),
            )
            candidates[-1]["observation_kinds"] = [predicate]
            candidates[-1]["relation_predicates"] = [predicate]

    for object_id, item in directed.items():
        add(
            tier=2,
            kind="functional_frontage",
            target_ids=[object_id],
            route_scope="group_local",
            owning_group_id=_optional_text(item.get("owning_group_id")),
            observations=_FRONTAGE_OBSERVATIONS,
            view_goal=str(item["observation_goal"]),
            discovery_ids=[_stable_discovery_id(item)],
            trigger="directed_usable_side",
            check_family="architecture_orientation",
        )

    for object_id, item in boundary.items():
        # Architecture context may enrich an already-required orientation or
        # clearance view, but cannot create a probe/check by itself.  Directed
        # objects already own architecture-orientation; non-directed objects
        # enter this branch only when need_clearance created an approach row.
        if object_id not in directed and object_id not in approach:
            continue
        add(
            tier=3,
            kind="functional_frontage",
            target_ids=[object_id],
            route_scope="group_local",
            owning_group_id=_optional_text(item.get("owning_group_id")),
            observations=(
                _BOUNDARY_OBSERVATIONS
                if object_id in directed
                else _DIRECTION_INDEPENDENT_BOUNDARY_OBSERVATIONS
            ),
            view_goal=str(item["observation_goal"]),
            discovery_ids=[
                _stable_discovery_id(item),
                *(
                    [_stable_discovery_id(directed[object_id])]
                    if object_id in directed
                    else []
                ),
            ],
            trigger="boundary_sensitive_frontage",
            check_family=(
                "architecture_orientation"
                if object_id in directed
                else "clearance"
            ),
        )

    for object_id, item in approach.items():
        add(
            tier=4,
            kind="approach_clearance",
            target_ids=[object_id],
            route_scope="group_local",
            owning_group_id=_optional_text(item.get("owning_group_id")),
            observations=(
                _CLEARANCE_OBSERVATIONS
                if object_id in directed
                else _DIRECTION_INDEPENDENT_CLEARANCE_OBSERVATIONS
            ),
            view_goal=str(item["observation_goal"]),
            discovery_ids=[_stable_discovery_id(item)],
            trigger="approach_clearance",
            check_family="clearance",
        )

    for item in discovery.get("unusual_unconfirmed") or []:
        if not isinstance(item, dict):
            continue
        target_ids = [str(value) for value in item["target_ids"]]
        add(
            tier=5,
            kind=(
                "functional_correspondence"
                if len(target_ids) > 1
                else "functional_frontage"
            ),
            target_ids=target_ids,
            route_scope="group_local",
            owning_group_id=str(item["owning_group_id"]),
            observations=(
                _LOCAL_MULTI_TARGET_CONFIRMATION_OBSERVATIONS
                if len(target_ids) > 1
                else _FRONTAGE_OBSERVATIONS
            ),
            view_goal=str(item["observation_goal"]),
            discovery_ids=[_stable_discovery_id(item)],
            trigger="unusual_unconfirmed_group_confirmation",
            check_family="local_affordance_confirmation",
            audit_reason=str(item.get("audit_reason") or ""),
        )

    # One camera unit may satisfy multiple discovery records only when their
    # target set and routing ownership are exactly the same.  This prevents a
    # boundary, clearance, and neutral confirmation request for one object
    # from spending three images, while never merging unrelated group members.
    candidates = _merge_exact_target_candidates(candidates)

    # Directed affordances are actionable usable-side checks. They also enrich
    # relation, boundary, and clearance units through ``surface_targets``.
    for candidate in candidates:
        all_ids = [
            *candidate["target_ids"],
            *candidate["related_target_ids"],
        ]
        candidate["functional_measurements"] = (
            compact_functional_measurements_for_checks(
                functional_measurement_bank,
                list(candidate.get("check_ids") or []),
            )
        )
        needs_directional_surface = bool(
            candidate.get("kind") != "functional_correspondence"
            or "directional_correspondence"
            in (candidate.get("relation_predicates") or [])
        )
        candidate["surface_targets"] = (
            [
                _surface_target_record(
                    object_id,
                    directed[object_id],
                )
                for object_id in all_ids
                if object_id in directed
            ]
            if needs_directional_surface
            else []
        )
        candidate["evidence_reuse"] = _evidence_reuse_contract(candidate)
        candidate["_identity"] = _acquisition_identity(candidate)

    candidates.sort(key=lambda item: (item["_tier"], item["_identity"]))
    selected: list[dict[str, Any]] = []
    backfill: list[dict[str, Any]] = []
    unscheduled: list[dict[str, Any]] = []
    seen_identities: set[
        tuple[str, tuple[str, ...], tuple[str, ...]]
    ] = set()
    unique_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        identity = candidate["_identity"]
        if identity in seen_identities:
            unscheduled.append(
                _unscheduled(candidate, "duplicate_acquisition_identity")
            )
            continue
        seen_identities.add(identity)
        unique_candidates.append(candidate)

    # Every accepted cross-group relation owns an isolated Judge episode and
    # therefore a mandatory evidence reservation.  Optional object/group
    # probes use the remaining budget by marginal coverage.  Failing closed
    # here is preferable to admitting a relation check that can never run.
    mandatory_cross_group = [
        candidate
        for candidate in unique_candidates
        if candidate.get("route_scope") == "cross_group"
    ]
    optional_candidates = [
        candidate
        for candidate in unique_candidates
        if candidate.get("route_scope") != "cross_group"
    ]
    if len(mandatory_cross_group) > limit:
        raise ValueError(
            "accepted cross-group functional relations exceed the hard "
            "probe-unit capacity; silent relation starvation is forbidden: "
            f"required={len(mandatory_cross_group)}, capacity={limit}"
        )
    selected, remaining = _coverage_aware_selection(
        optional_candidates,
        limit=limit,
        directed_object_ids=set(directed),
        preselected=mandatory_cross_group,
    )
    for candidate in remaining:
        backfill.append(candidate)
        unscheduled.append(
            _unscheduled(candidate, "max_probe_units_exhausted")
        )

    group_confirmations: list[dict[str, Any]] = []
    for candidate in [*selected, *backfill]:
        candidate.pop("_tier", None)
        candidate.pop("_sequence", None)
        candidate.pop("_identity", None)
        candidate.pop("_check_family", None)
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
    for priority, candidate in enumerate(
        backfill,
        start=len(selected) + 1,
    ):
        candidate["priority"] = priority
        candidate["probe_id"] = (
            f"functional_probe_candidate_{priority:02d}"
        )
    functional_check_ledger = route_functional_check_ledger(
        functional_check_ledger,
        selected_probe_units=selected,
        deferred_probe_units=backfill,
    )
    accepted_cross_group_count = sum(
        candidate.get("route_scope") == "cross_group"
        for candidate in unique_candidates
    )
    scheduled_cross_group_count = sum(
        candidate.get("route_scope") == "cross_group"
        for candidate in selected
    )
    lazy_group_checks = [
        deepcopy(item)
        for item in functional_check_ledger.get("checks") or []
        if isinstance(item, dict)
        and item.get("owner_stage") == "group_local"
        and item.get("acquisition_policy") == "judge_requested_on_demand"
    ]

    return {
        "schema_version": FUNCTIONAL_ACQUISITION_PLAN_VERSION,
        "max_probe_units": limit,
        "budget": {
            "max_probe_units": {
                "requested": requested_limit,
                "effective": limit,
                "hard_cap": FUNCTIONAL_PROBE_MAX_UNITS,
                "clamped_to_hard_cap": requested_limit > limit,
                "source": str(max_probe_units_source),
            },
            "desired_unique_acquisition_identities": len(
                unique_candidates
            ),
            "scheduled_under_effective_budget": len(selected),
            "accepted_cross_group_acquisition_identities": (
                accepted_cross_group_count
            ),
            "scheduled_cross_group_acquisition_identities": (
                scheduled_cross_group_count
            ),
            "cross_group_reservation_complete": (
                accepted_cross_group_count == scheduled_cross_group_count
            ),
        },
        "scheduling_policy": {
            "policy": (
                "mandatory_cross_group_then_marginal_coverage_v2"
            ),
            "priority_order": [
                "all_accepted_cross_group_relations",
                "new_required_check",
                "new_object",
                "new_directed_object",
                "new_functional_family",
                "additional_required_checks",
                "least_repeated_targets",
                "legacy_tier_tiebreak",
            ],
            "family_is_primary_budget_authority": False,
            "decision_authority": "none",
        },
        "lazy_group_acquisition": {
            "policy": "required_check_driven_judge_requests_v1",
            "initial_evidence": [
                "one_angled_global_view",
                "one_group_local_view",
                "reused_existing_group_owned_probe_views",
            ],
            "required_check_ids": [
                str(item.get("check_id") or "")
                for item in lazy_group_checks
            ],
            "required_check_count": len(lazy_group_checks),
            "completion_condition": (
                "all_group_owned_required_checks_resolved_or_"
                "judge_episode_budget_exhausted"
            ),
            "decision_authority": "none",
        },
        "object_evidence_policy": {
            "baseline": (
                "every object entering group-local review remains covered by "
                "its owning-group visual packet"
            ),
            "directed": (
                "localize one usable side and acquire one side-conditioned "
                "probe view; pair it with one angled global anchor for the "
                "architecture-orientation Judge obligation"
            ),
            "architecture_orientation": {
                "predicate": (
                    "usable side points toward plausible accessible interior"
                ),
                "evidence": [
                    "one_angled_global_view",
                    "same_side_conditioned_local_view",
                ],
                "deterministic_direction_descriptor": (
                    "camera_routing_and_measurement_bank_judge_evidence"
                ),
            },
            "clearance": {
                "directed": (
                    "reuse the same side-conditioned probe to inspect the "
                    "usable-side-forward free-space region"
                ),
                "non_directed": (
                    "acquire one direction-independent local probe to inspect "
                    "the surrounding free-space region"
                ),
            },
            "relation_checks": {
                "directional_correspondence": (
                    "joint visibility plus functional-side direction"
                ),
                "relative_use_geometry": (
                    "joint visibility plus relative layout and interaction "
                    "region"
                ),
                "shared_episode_identity": (
                    "same route scope + exact target set"
                ),
                "atomic_results": True,
                "within_group_acquisition": (
                    "lazy_judge_requested_on_demand"
                ),
                "cross_group_acquisition": (
                    "proactive_isolated_relation_episode"
                ),
            },
            "reuse_identity": (
                "same route scope + owning group + exact target set"
            ),
            "decision_authority": "none",
        },
        "probe_units": selected,
        "backfill_probe_units": backfill,
        "group_confirmations": group_confirmations,
        "functional_check_ledger": functional_check_ledger,
        "functional_measurement_bank": functional_measurement_bank,
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
        "directionality": "directed",
        "surface_roles": list(record.get("surface_roles") or []),
        "need_clearance": bool(record.get("need_clearance", False)),
    }
    hypothesis = record.get(
        "precomputed_usable_surface_hypothesis"
    )
    if isinstance(hypothesis, dict):
        result["precomputed_hypothesis"] = deepcopy(hypothesis)
    return result


def _evidence_reuse_contract(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Describe when one rendered view satisfies multiple routed checks."""

    triggers = list(
        dict.fromkeys(
            str(item)
            for item in candidate.get("acquisition_triggers") or []
            if str(item).strip()
        )
    )
    side_conditioned = bool(candidate.get("surface_targets"))
    clearance_requested = any(
        str(item) == "clearance"
        for item in candidate.get("check_types") or []
    )
    architecture_orientation_requested = any(
        str(item) == "architecture_orientation"
        for item in candidate.get("check_types") or []
    )
    return {
        "policy": "same_target_same_scope_single_probe_v2",
        "shared_probe_view": len(triggers) > 1,
        "side_conditioned": side_conditioned,
        "reuses_usable_side_hypothesis": side_conditioned,
        "reuses_side_conditioned_view_for_architecture_orientation": bool(
            side_conditioned and architecture_orientation_requested
        ),
        "reuses_side_conditioned_view_for_clearance": bool(
            side_conditioned and clearance_requested
        ),
        "architecture_orientation_judge_packet": (
            {
                "global_anchor": "one_angled_global_view",
                "local_evidence": "same_side_conditioned_probe_view",
            }
            if architecture_orientation_requested
            else None
        ),
        "clearance_observation_frame": (
            "usable_side_forward_region"
            if side_conditioned and clearance_requested
            else "object_surrounding_region"
            if clearance_requested
            else None
        ),
        "satisfies_acquisition_triggers": triggers,
        "satisfies_check_ids": list(
            candidate.get("check_ids") or []
        ),
        "relation_predicates": list(
            candidate.get("relation_predicates") or []
        ),
        "additional_probe_image_budget": 1,
        "decision_authority": "none",
    }


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
            "check_ids",
            "check_types",
            "observation_kinds",
            "relation_predicates",
            "acquisition_triggers",
            "surface_targets",
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


def _coverage_aware_selection(
    candidates: list[dict[str, Any]],
    *,
    limit: int,
    directed_object_ids: set[str],
    preselected: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select bounded units by auditable marginal obligation coverage."""

    remaining = list(candidates)
    selected: list[dict[str, Any]] = []
    covered_checks: set[str] = set()
    covered_objects: set[str] = set()
    covered_directed: set[str] = set()
    covered_families: set[str] = set()
    for chosen in preselected or []:
        if len(selected) >= limit:
            raise ValueError(
                "mandatory functional evidence reservations exceed capacity"
            )
        gain = _candidate_coverage_gain(
            chosen,
            covered_checks=covered_checks,
            covered_objects=covered_objects,
            covered_directed=covered_directed,
            covered_families=covered_families,
            directed_object_ids=directed_object_ids,
        )
        chosen["scheduling_coverage_gain"] = {
            "selection_round": len(selected) + 1,
            "reservation": "accepted_cross_group_relation",
            "new_check_ids": sorted(gain["new_checks"]),
            "new_object_ids": sorted(gain["new_objects"]),
            "new_directed_object_ids": sorted(gain["new_directed"]),
            "first_family_coverage": bool(gain["new_family"]),
            "repeated_target_ids": sorted(gain["repeated_objects"]),
        }
        selected.append(chosen)
        covered_checks.update(gain["new_checks"])
        covered_objects.update(gain["new_objects"])
        covered_directed.update(gain["new_directed"])
        family = str(chosen.get("_check_family") or "")
        if family:
            covered_families.add(family)
    while remaining and len(selected) < limit:
        best_index = 0
        best_gain = _candidate_coverage_gain(
            remaining[0],
            covered_checks=covered_checks,
            covered_objects=covered_objects,
            covered_directed=covered_directed,
            covered_families=covered_families,
            directed_object_ids=directed_object_ids,
        )
        best_score = _coverage_gain_score(best_gain)
        for index, candidate in enumerate(remaining[1:], start=1):
            gain = _candidate_coverage_gain(
                candidate,
                covered_checks=covered_checks,
                covered_objects=covered_objects,
                covered_directed=covered_directed,
                covered_families=covered_families,
                directed_object_ids=directed_object_ids,
            )
            score = _coverage_gain_score(gain)
            if score > best_score:
                best_index = index
                best_gain = gain
                best_score = score
        chosen = remaining.pop(best_index)
        chosen["scheduling_coverage_gain"] = {
            "selection_round": len(selected) + 1,
            "new_check_ids": sorted(best_gain["new_checks"]),
            "new_object_ids": sorted(best_gain["new_objects"]),
            "new_directed_object_ids": sorted(
                best_gain["new_directed"]
            ),
            "first_family_coverage": bool(best_gain["new_family"]),
            "repeated_target_ids": sorted(best_gain["repeated_objects"]),
        }
        selected.append(chosen)
        covered_checks.update(best_gain["new_checks"])
        covered_objects.update(best_gain["new_objects"])
        covered_directed.update(best_gain["new_directed"])
        family = str(chosen.get("_check_family") or "")
        if family:
            covered_families.add(family)
    return selected, remaining


def _candidate_coverage_gain(
    candidate: dict[str, Any],
    *,
    covered_checks: set[str],
    covered_objects: set[str],
    covered_directed: set[str],
    covered_families: set[str],
    directed_object_ids: set[str],
) -> dict[str, Any]:
    check_ids = {
        str(item)
        for item in candidate.get("check_ids") or []
        if str(item).strip()
    }
    object_ids = {
        *[
            str(item)
            for item in candidate.get("target_ids") or []
            if str(item).strip()
        ],
        *[
            str(item)
            for item in candidate.get("related_target_ids") or []
            if str(item).strip()
        ],
    }
    directed_ids = {
        str(item.get("target_id") or "")
        for item in candidate.get("surface_targets") or []
        if isinstance(item, dict) and item.get("target_id")
    } & directed_object_ids
    family = str(candidate.get("_check_family") or "")
    return {
        "new_checks": check_ids - covered_checks,
        "new_objects": object_ids - covered_objects,
        "new_directed": directed_ids - covered_directed,
        "new_family": bool(family and family not in covered_families),
        "repeated_objects": object_ids & covered_objects,
        "tier": int(candidate.get("_tier") or 0),
    }


def _coverage_gain_score(gain: dict[str, Any]) -> tuple[int, ...]:
    """Lexicographic score; deterministic input order resolves exact ties."""

    new_checks = gain["new_checks"]
    new_objects = gain["new_objects"]
    new_directed = gain["new_directed"]
    repeated_objects = gain["repeated_objects"]
    return (
        int(bool(new_checks)),
        len(new_objects),
        len(new_directed),
        int(bool(gain["new_family"])),
        len(new_checks),
        -len(repeated_objects),
        -int(gain["tier"]),
    )


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
        "check_ids": list(candidate.get("check_ids") or []),
        "target_ids": [
            *candidate.get("target_ids", []),
            *candidate.get("related_target_ids", []),
        ],
        "route_scope": candidate.get("route_scope"),
        "owning_group_id": candidate.get("owning_group_id"),
        "acquisition_trigger": candidate.get("acquisition_trigger"),
        "observation_kinds": list(
            candidate.get("observation_kinds") or []
        ),
        "relation_predicates": list(
            candidate.get("relation_predicates") or []
        ),
        "observation_goal": str(
            candidate.get("view_goal") or candidate.get("reason") or ""
        ),
        "evidence_reuse": deepcopy(
            candidate.get("evidence_reuse") or {}
        ),
        "reason": reason,
    }


def _optional_text(value: Any) -> str | None:
    result = str(value or "").strip()
    return result or None


def _stable_discovery_id(item: dict[str, Any]) -> str:
    value = str(item.get("discovery_id") or "").strip()
    if value:
        return value
    targets = item.get("target_ids") or [item.get("target_id")]
    normalized = "_".join(
        sorted(str(target) for target in targets if str(target).strip())
    )
    if not normalized:
        raise ValueError("functional discovery record requires discovery_id")
    return f"legacy_discovery:{normalized}"
