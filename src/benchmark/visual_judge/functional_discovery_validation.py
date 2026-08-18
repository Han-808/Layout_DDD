"""Fail-closed validation and composition for functional discovery.

The helpers here are deliberately model- and transport-independent.  They
validate trusted object identity, finite discovery vocabularies, group
ownership, and the compatibility-shaped discovery result consumed by camera
acquisition.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from benchmark.visual_judge.functional_discovery_contract import (
    FUNCTIONAL_COUNTERPART_MODES,
    FUNCTIONAL_DIRECTIONALITY,
    FUNCTIONAL_ORDINARY_MOBILITY,
    FUNCTIONAL_RELATION_DEPENDENCIES,
    FUNCTIONAL_RELATION_PREDICATES,
    FUNCTIONAL_REVIEW_STATES,
    FUNCTIONAL_SURFACE_ROLES,
    normalized_functional_relation_predicates,
)
from benchmark.visual_judge.identity_evidence import (
    validate_identity_evidence,
)


_TOP_LEVEL_FIELDS = frozenset(
    {
        "inspected_object_ids",
        "directed_surface_targets",
        "functional_correspondences",
        "approach_clearance_targets",
        "boundary_sensitive_targets",
        "unusual_unconfirmed",
        "reason",
    }
)
_FORBIDDEN_FIELDS = frozenset(
    {
        "verdict",
        "validity",
        "score",
        "defect",
        "defects",
        "is_invalid",
        "metric_verdict",
        "metric_score",
        "status",
        "confidence",
        "camera",
        "camera_pose",
        "camera_action",
        "camera_actions",
        "pose",
        "location",
        "target",
        "lens",
        "lens_mm",
        "rotation",
        "direction",
        "normal",
        "azimuth",
        "elevation",
        "scene_patch",
        "scene_mutation",
        "mutation",
    }
)


def validate_functional_discovery_request(
    value: Any,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("functional discovery request must be a JSON object")
    if str(value.get("metric") or "") != "functional_consistency":
        raise ValueError(
            "functional discovery only supports functional_consistency"
        )
    image = value.get("global_image_path")
    if not isinstance(image, (str, Path)) or not str(image).strip():
        raise ValueError("functional discovery requires global_image_path")
    image_path = Path(str(image)).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(
            f"functional discovery image does not exist: {image_path}"
        )
    objects = _validated_objects(value.get("objects"))
    groups = _validated_groups(
        value.get("groups"),
        known_object_ids={item["id"] for item in objects},
    )
    identity = validate_identity_evidence(
        image_path=value.get("identity_image_path"),
        legend=value.get("identity_legend"),
        expected_object_ids=(item["id"] for item in objects),
        label="functional discovery",
    )
    architecture = value.get("architecture_context")
    if architecture is None:
        architecture = {
            "source": "unavailable",
            "logical_boundary_enabled": False,
            "logical_boundary_xy": [],
            "physical_walls_rendered": None,
            "physical_wall_ids": [],
        }
    if not isinstance(architecture, dict):
        raise ValueError(
            "functional discovery architecture_context must be an object"
        )
    return {
        "metric": "functional_consistency",
        "scene_id": (
            str(value["scene_id"])
            if value.get("scene_id") is not None
            else None
        ),
        "scene_type": (
            str(value["scene_type"])
            if value.get("scene_type") is not None
            else None
        ),
        "global_image_path": str(image_path),
        "architecture_context": deepcopy(architecture),
        "objects": objects,
        "groups": groups,
        **identity,
    }


def validate_functional_affordance_response(
    value: Any,
    *,
    object_ids: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("functional affordance response must be an object")
    _reject_forbidden_fields(value)
    unknown = set(value) - {"objects", "reason"}
    if unknown:
        raise ValueError(
            "functional affordance response returned unsupported fields: "
            f"{sorted(unknown)}"
        )
    rows = _json_object_list(value.get("objects"), "objects")
    if len(rows) != len(object_ids):
        raise ValueError(
            "functional affordance objects must cover every input object"
        )
    normalized: list[dict[str, Any]] = []
    normalization_warnings: list[dict[str, Any]] = []
    for expected_id, item in zip(object_ids, rows, strict=True):
        allowed = {
            "object_id",
            "directionality",
            "surface_roles",
            "need_clearance",
            "boundary_review_state",
            "review_state",
            "observation_goal",
            "boundary_observation_goal",
        }
        extra = set(item) - allowed
        if extra:
            raise ValueError(
                "functional affordance object contains unsupported fields: "
                f"{sorted(extra)}"
            )
        object_id = str(item.get("object_id") or "").strip()
        if object_id != expected_id:
            raise ValueError(
                "functional affordance objects must contain every input "
                "object exactly once in supplied order"
            )
        directionality = _allowed_token(
            item.get("directionality"),
            FUNCTIONAL_DIRECTIONALITY,
            "directionality",
        )
        raw_roles = item.get("surface_roles")
        if directionality == "directed":
            roles = _validated_text_list(
                raw_roles,
                allowed=FUNCTIONAL_SURFACE_ROLES,
                label="surface_roles",
            )
        else:
            if raw_roles not in (None, []):
                raise ValueError(
                    "non_directed affordances require empty surface_roles; "
                    f"object_id={object_id}"
                )
            roles = []
        need_clearance = _strict_bool(
            item.get("need_clearance"),
            "need_clearance",
        )
        boundary_state = _allowed_token(
            item.get("boundary_review_state"),
            FUNCTIONAL_REVIEW_STATES,
            "boundary_review_state",
        )
        review_state = _allowed_token(
            item.get("review_state"),
            FUNCTIONAL_REVIEW_STATES,
            "review_state",
        )
        observation_goal = _required_text(
            item.get("observation_goal"),
            "functional affordance observation_goal",
        )
        boundary_goal = str(
            item.get("boundary_observation_goal") or ""
        ).strip()[:1000]
        if boundary_state != "routine" and not boundary_goal:
            raise ValueError(
                "non-routine boundary review requires "
                f"boundary_observation_goal; object_id={object_id}"
            )
        if (
            boundary_state != "routine"
            and directionality == "non_directed"
            and not need_clearance
        ):
            raise ValueError(
                "boundary context cannot create an independent functional "
                "check; non_directed objects with need_clearance=false "
                "must use boundary_review_state=routine; "
                f"object_id={object_id}"
            )
        if boundary_state == "routine" and boundary_goal:
            # The explicit lifecycle state is authoritative. A model may
            # redundantly describe an ordinary boundary view even after
            # classifying the review as routine; clearing that text cannot
            # create a probe or alter a metric verdict. Preserve a structured
            # audit record instead of rejecting the otherwise valid ledger.
            normalization_warnings.append(
                {
                    "code": "routine_boundary_goal_cleared",
                    "object_id": object_id,
                    "field": "boundary_observation_goal",
                    "repair": "cleared",
                    "reason": "boundary_review_state_is_routine",
                }
            )
            boundary_goal = ""
        normalized.append(
            {
                "object_id": object_id,
                "directionality": directionality,
                "surface_roles": roles,
                "need_clearance": need_clearance,
                "boundary_review_state": boundary_state,
                "review_state": review_state,
                "observation_goal": observation_goal,
                "boundary_observation_goal": boundary_goal,
            }
        )
    return {
        "objects": normalized,
        "reason": _required_text(
            value.get("reason"),
            "functional affordance reason",
        ),
        "normalization_warnings": normalization_warnings,
    }


def salvage_functional_affordance_response(
    value: Any,
    *,
    object_ids: tuple[str, ...],
    fallback_value: Any = None,
) -> dict[str, Any]:
    """Retain valid object rows and default only malformed/missing rows.

    The strict validator above remains the authoritative contract.  This
    helper is used only after the single repair quota has been exhausted; it
    cannot create a usable surface or clearance requirement.  A defaulted row
    therefore means only "no specialised probe was scheduled for this
    object", never that the object's functionality was judged valid.
    """

    # A schema repair has no authority to revise an already-valid semantic
    # atom.  It may only fill a missing or malformed row for the same trusted
    # object identity.
    sources = [
        (
            "initial",
            fallback_value if isinstance(fallback_value, dict) else {},
        ),
        ("repair", value if isinstance(value, dict) else {}),
    ]
    known_ids = set(object_ids)
    rows_by_source: dict[str, dict[str, list[tuple[int, dict[str, Any]]]]] = {}
    rejected_items: list[dict[str, Any]] = []
    for source_name, response in sources:
        raw_rows = response.get("objects")
        raw_rows = raw_rows if isinstance(raw_rows, list) else []
        source_rows: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        rows_by_source[source_name] = source_rows
        for index, item in enumerate(raw_rows):
            if not isinstance(item, dict):
                rejected_items.append(
                    {
                        "source": source_name,
                        "index": index,
                        "object_id": None,
                        "reason": "row_is_not_an_object",
                    }
                )
                continue
            object_id = str(item.get("object_id") or "").strip()
            if object_id not in known_ids:
                rejected_items.append(
                    {
                        "source": source_name,
                        "index": index,
                        "object_id": object_id or None,
                        "reason": "unknown_or_empty_object_id",
                    }
                )
                continue
            source_rows.setdefault(object_id, []).append((index, item))

    normalized: list[dict[str, Any]] = []
    accepted_ids: list[str] = []
    accepted_sources: dict[str, str] = {}
    defaulted_ids: list[str] = []
    for object_id in object_ids:
        accepted = False
        for source_name, _ in sources:
            candidates = rows_by_source[source_name].get(object_id, [])
            if len(candidates) > 1:
                rejected_items.append(
                    {
                        "source": source_name,
                        "index": None,
                        "object_id": object_id,
                        "reason": "duplicate_object_rows",
                    }
                )
                continue
            if not candidates:
                continue
            index, candidate = candidates[0]
            try:
                validated = validate_functional_affordance_response(
                    {
                        "objects": [candidate],
                        "reason": "item-level salvage validation",
                    },
                    object_ids=(object_id,),
                )
            except (TypeError, ValueError, KeyError) as exc:
                rejected_items.append(
                    {
                        "source": source_name,
                        "index": index,
                        "object_id": object_id,
                        "reason": str(exc),
                    }
                )
                continue
            normalized.append(validated["objects"][0])
            accepted_ids.append(object_id)
            accepted_sources[object_id] = source_name
            accepted = True
            break
        if not accepted:
            rejected_items.append(
                {
                    "source": None,
                    "index": None,
                    "object_id": object_id,
                    "reason": "no_valid_object_row_after_retry",
                }
            )
            normalized.append(_default_affordance_row(object_id))
            defaulted_ids.append(object_id)

    return {
        "objects": normalized,
        "reason": (
            "Valid affordance rows were retained; malformed or missing rows "
            "defaulted to no specialised usable-surface or clearance probe."
        ),
        "item_salvage": {
            "policy": "valid_rows_plus_neutral_object_fallback_v1",
            "expected_object_count": len(object_ids),
            "accepted_object_ids": accepted_ids,
            "accepted_sources": accepted_sources,
            "accepted_object_count": len(accepted_ids),
            "defaulted_object_ids": defaulted_ids,
            "defaulted_object_count": len(defaulted_ids),
            "coverage_fraction": (
                len(accepted_ids) / len(object_ids) if object_ids else 0.0
            ),
            "rejected_items": rejected_items,
        },
    }


def validate_functional_relation_response(
    value: Any,
    *,
    object_ids: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("functional relation response must be an object")
    _reject_forbidden_fields(value)
    unknown = set(value) - {"considered_object_ids", "relations", "reason"}
    if unknown:
        raise ValueError(
            "functional relation response returned unsupported fields: "
            f"{sorted(unknown)}"
        )
    considered = _validated_id_list(
        value.get("considered_object_ids"),
        known=set(object_ids),
        label="considered_object_ids",
        minimum=1,
    )
    if tuple(considered) != object_ids:
        raise ValueError(
            "functional relation considered_object_ids must contain every "
            "input object exactly once in supplied order"
        )
    relations: list[dict[str, Any]] = []
    identities: set[tuple[tuple[str, ...], str]] = set()
    for item in _json_object_list(value.get("relations"), "relations"):
        extra = set(item) - {
            "target_ids",
            "predicate",
            "dependency",
            "counterpart_mode",
            "ordinary_mobility",
            "observation_goal",
        }
        if extra:
            raise ValueError(
                "functional relation contains unsupported fields: "
                f"{sorted(extra)}"
            )
        target_ids = _validated_id_list(
            item.get("target_ids"),
            known=set(object_ids),
            label="relations.target_ids",
            minimum=2,
        )
        if len(target_ids) != 2:
            raise ValueError(
                "relations.target_ids must contain exactly two trusted "
                "object IDs for one atomic direct-use relation"
            )
        predicate = _allowed_token(
            item.get("predicate"),
            FUNCTIONAL_RELATION_PREDICATES,
            "relations.predicate",
        )
        dependency = _allowed_token(
            item.get("dependency"),
            FUNCTIONAL_RELATION_DEPENDENCIES,
            "relations.dependency",
        )
        counterpart_mode = _allowed_token(
            item.get("counterpart_mode"),
            FUNCTIONAL_COUNTERPART_MODES,
            "relations.counterpart_mode",
        )
        ordinary_mobility = _allowed_token(
            item.get("ordinary_mobility"),
            FUNCTIONAL_ORDINARY_MOBILITY,
            "relations.ordinary_mobility",
        )
        identity = (tuple(sorted(target_ids)), predicate)
        if identity in identities:
            raise ValueError(
                "functional relation audit contains a duplicate atomic check"
            )
        identities.add(identity)
        relations.append(
            {
                "target_ids": target_ids,
                "predicate": predicate,
                "dependency": dependency,
                "counterpart_mode": counterpart_mode,
                "ordinary_mobility": ordinary_mobility,
                "observation_goal": _required_text(
                    item.get("observation_goal"),
                    "functional relation observation_goal",
                ),
            }
        )
    return {
        "considered_object_ids": considered,
        "relations": relations,
        "reason": _required_text(
            value.get("reason"),
            "functional relation reason",
        ),
    }


def salvage_functional_relation_response(
    value: Any,
    *,
    object_ids: tuple[str, ...],
    fallback_value: Any = None,
) -> dict[str, Any]:
    """Keep independently valid relation rows after one failed repair."""

    initial = fallback_value if isinstance(fallback_value, dict) else {}
    repair = value if isinstance(value, dict) else {}
    sources = [("initial", initial), ("repair", repair)]
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    identities: set[tuple[tuple[str, ...], str]] = set()
    known_ids = set(object_ids)
    initial_anchors: list[tuple[tuple[str, ...], str]] = []
    initial_rows = initial.get("relations")
    initial_rows = initial_rows if isinstance(initial_rows, list) else []
    for index, item in enumerate(initial_rows):
        try:
            anchor = _functional_relation_identity_anchor(
                item,
                known_ids=known_ids,
            )
        except (TypeError, ValueError, KeyError) as exc:
            rejected.append(
                {
                    "source": "initial",
                    "index": index,
                    "reason": f"invalid_identity_anchor: {exc}",
                }
            )
            continue
        if anchor not in initial_anchors:
            initial_anchors.append(anchor)

    considered_valid = False
    for source_name, response in sources:
        considered = response.get("considered_object_ids")
        considered_valid = considered_valid or bool(
            isinstance(considered, list)
            and tuple(str(item).strip() for item in considered) == object_ids
        )
        raw_relations = response.get("relations")
        raw_relations = raw_relations if isinstance(raw_relations, list) else []
        for index, item in enumerate(raw_relations):
            try:
                raw_identity = _functional_relation_identity_anchor(
                    item,
                    known_ids=known_ids,
                )
            except (TypeError, ValueError, KeyError) as exc:
                rejected.append(
                    {
                        "source": source_name,
                        "index": index,
                        "reason": f"invalid_identity_anchor: {exc}",
                    }
                )
                continue
            if raw_identity not in initial_anchors:
                rejected.append(
                    {
                        "source": source_name,
                        "index": index,
                        "reason": "relation_has_no_initial_identity_anchor",
                    }
                )
                continue
            try:
                validated = validate_functional_relation_response(
                    {
                        "considered_object_ids": list(object_ids),
                        "relations": [item],
                        "reason": "item-level salvage validation",
                    },
                    object_ids=object_ids,
                )
                row = validated["relations"][0]
                identity = (
                    tuple(sorted(row["target_ids"])),
                    str(row["predicate"]),
                )
            except (TypeError, ValueError, KeyError) as exc:
                rejected.append(
                    {
                        "source": source_name,
                        "index": index,
                        "reason": str(exc),
                    }
                )
                continue
            if identity in identities:
                continue
            identities.add(identity)
            accepted.append(row)
    dropped_anchors = [
        {
            "target_ids": list(identity[0]),
            "predicate": identity[1],
        }
        for identity in initial_anchors
        if identity not in identities
    ]
    return {
        "considered_object_ids": list(object_ids),
        "relations": accepted,
        "reason": (
            "Valid relation rows were retained; malformed rows were omitted "
            "without affecting other objects or checks."
        ),
        "item_salvage": {
            "policy": "initial_anchored_relation_atoms_v2",
            "consideration_contract_valid": considered_valid,
            "anchored_relation_count": len(initial_anchors),
            "accepted_relation_count": len(accepted),
            "dropped_relation_count": len(dropped_anchors),
            "dropped_relation_anchors": dropped_anchors,
            "rejected_relation_count": len(rejected),
            "rejected_items": rejected,
        },
    }


def _functional_relation_identity_anchor(
    value: Any,
    *,
    known_ids: set[str],
) -> tuple[tuple[str, ...], str]:
    """Read only the semantic identity needed to authorize schema repair."""

    if not isinstance(value, dict):
        raise ValueError("relation row is not an object")
    target_ids = value.get("target_ids")
    if (
        not isinstance(target_ids, list)
        or len(target_ids) != 2
        or any(not isinstance(item, str) or not item.strip() for item in target_ids)
    ):
        raise ValueError("relation target_ids are not one trusted pair")
    normalized_targets = tuple(sorted(str(item).strip() for item in target_ids))
    if len(set(normalized_targets)) != 2 or not set(normalized_targets) <= known_ids:
        raise ValueError("relation target_ids are unknown or duplicated")
    predicate = str(value.get("predicate") or "").strip()
    if predicate not in FUNCTIONAL_RELATION_PREDICATES:
        raise ValueError("relation predicate is unsupported")
    return normalized_targets, predicate


def _default_affordance_row(object_id: str) -> dict[str, Any]:
    return {
        "object_id": object_id,
        "directionality": "non_directed",
        "surface_roles": [],
        "need_clearance": False,
        "boundary_review_state": "routine",
        "review_state": "routine",
        "observation_goal": (
            "Use the fixed group/global evidence; no specialised affordance "
            "probe was scheduled for this object."
        ),
        "boundary_observation_goal": "",
    }


def compose_functional_discovery_result(
    *,
    affordance: dict[str, Any],
    relations: dict[str, Any],
    object_ids: tuple[str, ...],
    groups: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compose compatibility fields using only trusted IDs and group map."""

    rows = list(affordance["objects"])
    if tuple(item["object_id"] for item in rows) != object_ids:
        raise ValueError("affordance ledger does not match trusted object IDs")
    if tuple(relations["considered_object_ids"]) != object_ids:
        raise ValueError("relation audit does not match trusted object IDs")
    object_to_group = _object_to_group(groups)
    directed: list[dict[str, Any]] = []
    approach: list[dict[str, Any]] = []
    boundary: list[dict[str, Any]] = []
    unusual: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        object_id = row["object_id"]
        group_id = object_to_group.get(object_id)
        if row["directionality"] == "directed":
            directed.append(
                {
                    "discovery_id": f"directed_surface_{index:02d}",
                    "target_id": object_id,
                    "directionality": row["directionality"],
                    "surface_roles": deepcopy(row["surface_roles"]),
                    "observation_goal": row["observation_goal"],
                    "owning_group_id": group_id,
                    "review_state": row["review_state"],
                    "boundary_review_state": row[
                        "boundary_review_state"
                    ],
                    "need_clearance": row["need_clearance"],
                }
            )
        if row["need_clearance"]:
            approach.append(
                {
                    "discovery_id": f"approach_clearance_{index:02d}",
                    "target_id": object_id,
                    "observation_goal": row["observation_goal"],
                    "owning_group_id": group_id,
                    "need_clearance": True,
                }
            )
        if row["boundary_review_state"] != "routine":
            boundary.append(
                {
                    "discovery_id": f"boundary_sensitive_{index:02d}",
                    "target_id": object_id,
                    "observation_goal": row["boundary_observation_goal"],
                    "owning_group_id": group_id,
                    "boundary_review_state": row[
                        "boundary_review_state"
                    ],
                }
            )
        if row["review_state"] != "routine":
            if group_id is None:
                # No trusted local owner exists; retain the need in the ledger
                # instead of inventing a group.
                continue
            unusual.append(
                {
                    "discovery_id": f"unusual_unconfirmed_{index:02d}",
                    "target_ids": [object_id],
                    "owning_group_id": group_id,
                    "observation_goal": row["observation_goal"],
                    "audit_reason": (
                        "affordance requires local visual confirmation"
                    ),
                    "decision_authority": "none",
                    "confirmation_scope": "group_local",
                }
            )
    within: list[dict[str, Any]] = []
    cross: list[dict[str, Any]] = []
    for index, relation in enumerate(relations["relations"], start=1):
        group_ids = list(
            dict.fromkeys(
                object_to_group.get(object_id)
                for object_id in relation["target_ids"]
                if object_to_group.get(object_id)
            )
        )
        normalized = {
            **deepcopy(relation),
            "observation_kinds": [str(relation["predicate"])],
            "discovery_id": f"functional_correspondence_{index:02d}",
            "group_ids": group_ids,
            "scope": (
                "within_group" if len(group_ids) == 1 else "cross_group"
            ),
        }
        (within if len(group_ids) == 1 else cross).append(normalized)
    return {
        "inspected_object_ids": object_ids,
        "object_affordance_ledger": tuple(deepcopy(rows)),
        "directed_surface_targets": tuple(directed),
        "within_group_correspondences": tuple(within),
        "cross_group_correspondences": tuple(cross),
        "approach_clearance_targets": tuple(approach),
        "boundary_sensitive_targets": tuple(boundary),
        "unusual_unconfirmed": tuple(unusual),
        "reason": (
            f"{affordance['reason']} | {relations['reason']}"
        )[:1000],
    }


def validate_functional_discovery_response(
    value: Any,
    *,
    object_ids: tuple[str, ...],
    groups: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(
            "functional discovery response must be a JSON object"
        )
    _reject_forbidden_fields(value)
    unknown = set(value) - _TOP_LEVEL_FIELDS
    if unknown:
        raise ValueError(
            "functional discovery returned unsupported fields: "
            f"{sorted(unknown)}"
        )
    inspected = _validated_id_list(
        value.get("inspected_object_ids"),
        known=set(object_ids),
        label="inspected_object_ids",
        minimum=1,
    )
    if tuple(inspected) != object_ids:
        raise ValueError(
            "functional discovery inspected_object_ids must contain every "
            "input object exactly once in supplied order"
        )
    object_to_group = _object_to_group(groups)

    directed = _validated_directed_targets(
        value.get("directed_surface_targets"),
        known=set(object_ids),
        object_to_group=object_to_group,
    )
    raw_correspondences = _validated_relations(
        value.get("functional_correspondences"),
        known=set(object_ids),
        label="functional_correspondences",
    )
    within: list[dict[str, Any]] = []
    cross: list[dict[str, Any]] = []
    for index, item in enumerate(raw_correspondences, start=1):
        group_ids = list(
            dict.fromkeys(
                object_to_group.get(object_id)
                for object_id in item["target_ids"]
                if object_to_group.get(object_id)
            )
        )
        normalized = {
            **item,
            "observation_kinds": [str(item["predicate"])],
            "discovery_id": f"functional_correspondence_{index:02d}",
            "group_ids": group_ids,
            "scope": (
                "within_group"
                if len(group_ids) == 1
                else "cross_group"
            ),
        }
        (within if len(group_ids) == 1 else cross).append(normalized)

    approach = _validated_single_targets(
        value.get("approach_clearance_targets"),
        known=set(object_ids),
        label="approach_clearance_targets",
        prefix="approach_clearance",
        object_to_group=object_to_group,
    )
    boundary = _validated_single_targets(
        value.get("boundary_sensitive_targets"),
        known=set(object_ids),
        label="boundary_sensitive_targets",
        prefix="boundary_sensitive",
        object_to_group=object_to_group,
    )
    unusual = _validated_unusual(
        value.get("unusual_unconfirmed"),
        known=set(object_ids),
        object_to_group=object_to_group,
    )
    return {
        "inspected_object_ids": tuple(inspected),
        "directed_surface_targets": tuple(directed),
        "within_group_correspondences": tuple(within),
        "cross_group_correspondences": tuple(cross),
        "approach_clearance_targets": tuple(approach),
        "boundary_sensitive_targets": tuple(boundary),
        "unusual_unconfirmed": tuple(unusual),
        "reason": _required_text(
            value.get("reason"),
            "functional discovery reason",
        ),
    }


def _validated_objects(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(
            "functional discovery requires a non-empty object list"
        )
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) - {"id", "category"}:
            raise ValueError(
                "functional discovery objects permit only id and category"
            )
        object_id = str(item.get("id") or "").strip()
        category = str(item.get("category") or "").strip()
        if not object_id or not category or object_id in seen:
            raise ValueError(
                "functional discovery objects require unique non-empty id "
                "and category"
            )
        seen.add(object_id)
        result.append({"id": object_id, "category": category})
    return result


def _validated_groups(
    value: Any,
    *,
    known_object_ids: set[str],
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("functional discovery groups must be a JSON list")
    result: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    seen_objects: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(
                "functional discovery groups must contain JSON objects"
            )
        group_id = str(item.get("group_id") or "").strip()
        members = _validated_id_list(
            item.get("object_ids"),
            known=known_object_ids,
            label="group.object_ids",
            minimum=1,
        )
        if not group_id or group_id in seen_groups:
            raise ValueError(
                "functional discovery groups require unique non-empty group_id"
            )
        overlap = sorted(set(members) & seen_objects)
        if overlap:
            raise ValueError(
                "functional discovery grouping is not a partition; duplicate "
                f"object IDs: {overlap}"
            )
        seen_groups.add(group_id)
        seen_objects.update(members)
        result.append({"group_id": group_id, "object_ids": members})
    if result and seen_objects != known_object_ids:
        raise ValueError(
            "functional discovery grouping must cover every known object"
        )
    return result


def _validated_directed_targets(
    value: Any,
    *,
    known: set[str],
    object_to_group: dict[str, str],
) -> list[dict[str, Any]]:
    items = _json_object_list(value, "directed_surface_targets")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(items, start=1):
        if set(item) - {
            "target_id",
            "surface_roles",
            "need_clearance",
            "observation_goal",
        }:
            raise ValueError(
                "directed_surface_targets contains unsupported fields"
            )
        target_id = _known_id(item.get("target_id"), known)
        if target_id in seen:
            raise ValueError(
                "directed_surface_targets contains a duplicate target"
            )
        roles = _validated_text_list(
            item.get("surface_roles"),
            allowed=FUNCTIONAL_SURFACE_ROLES,
            label="surface_roles",
        )
        seen.add(target_id)
        result.append(
            {
                "discovery_id": f"directed_surface_{index:02d}",
                "target_id": target_id,
                "surface_roles": roles,
                "need_clearance": _strict_bool(
                    item.get("need_clearance"),
                    "directed_surface_targets.need_clearance",
                ),
                "observation_goal": _required_text(
                    item.get("observation_goal"),
                    "directed surface observation_goal",
                ),
                "owning_group_id": object_to_group.get(target_id),
            }
        )
    return result


def _validated_relations(
    value: Any,
    *,
    known: set[str],
    label: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    fingerprints: set[tuple[tuple[str, ...], str]] = set()
    for item in _json_object_list(value, label):
        if set(item) - {
            "target_ids",
            "predicate",
            "observation_kinds",
            "observation_goal",
        }:
            raise ValueError(f"{label} contains unsupported fields")
        target_ids = _validated_id_list(
            item.get("target_ids"),
            known=known,
            label=f"{label}.target_ids",
            minimum=2,
        )
        predicates = normalized_functional_relation_predicates(item)
        goal = _required_text(
            item.get("observation_goal"),
            f"{label} observation_goal",
        )
        for predicate in predicates:
            fingerprint = (tuple(sorted(target_ids)), predicate)
            if fingerprint in fingerprints:
                raise ValueError(
                    f"{label} contains a duplicate atomic relation check"
                )
            fingerprints.add(fingerprint)
            result.append(
                {
                    "target_ids": target_ids,
                    "predicate": predicate,
                    "observation_goal": goal,
                }
            )
    return result


def _validated_single_targets(
    value: Any,
    *,
    known: set[str],
    label: str,
    prefix: str,
    object_to_group: dict[str, str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(
        _json_object_list(value, label),
        start=1,
    ):
        allowed_fields = {"target_id", "observation_goal"}
        if label == "approach_clearance_targets":
            allowed_fields.add("need_clearance")
        if set(item) - allowed_fields:
            raise ValueError(f"{label} contains unsupported fields")
        target_id = _known_id(item.get("target_id"), known)
        if target_id in seen:
            raise ValueError(f"{label} contains a duplicate target")
        seen.add(target_id)
        normalized = {
            "discovery_id": f"{prefix}_{index:02d}",
            "target_id": target_id,
            "observation_goal": _required_text(
                item.get("observation_goal"),
                f"{label} observation_goal",
            ),
            "owning_group_id": object_to_group.get(target_id),
        }
        if label == "approach_clearance_targets":
            if _strict_bool(
                item.get("need_clearance"),
                "approach_clearance_targets.need_clearance",
            ) is not True:
                raise ValueError(
                    "approach_clearance_targets.need_clearance must be true"
                )
            normalized["need_clearance"] = True
        result.append(normalized)
    return result


def _validated_unusual(
    value: Any,
    *,
    known: set[str],
    object_to_group: dict[str, str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, item in enumerate(
        _json_object_list(value, "unusual_unconfirmed"),
        start=1,
    ):
        if set(item) - {
            "target_ids",
            "observation_goal",
            "audit_reason",
        }:
            raise ValueError(
                "unusual_unconfirmed contains unsupported fields"
            )
        target_ids = _validated_id_list(
            item.get("target_ids"),
            known=known,
            label="unusual_unconfirmed.target_ids",
            minimum=1,
        )
        group_ids = {
            object_to_group.get(object_id)
            for object_id in target_ids
        }
        if None in group_ids or len(group_ids) != 1:
            raise ValueError(
                "unusual_unconfirmed must map to exactly one owning group; "
                "cross-group uncertainty belongs in functional correspondence"
            )
        result.append(
            {
                "discovery_id": f"unusual_unconfirmed_{index:02d}",
                "target_ids": target_ids,
                "owning_group_id": next(iter(group_ids)),
                "observation_goal": _required_text(
                    item.get("observation_goal"),
                    "unusual_unconfirmed observation_goal",
                ),
                "audit_reason": _required_text(
                    item.get("audit_reason"),
                    "unusual_unconfirmed audit_reason",
                ),
                "decision_authority": "none",
                "confirmation_scope": "group_local",
            }
        )
    return result


def _object_to_group(
    groups: list[dict[str, Any]],
) -> dict[str, str]:
    return {
        str(object_id): str(group["group_id"])
        for group in groups
        for object_id in group.get("object_ids") or []
    }


def _json_object_list(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(
        isinstance(item, dict) for item in value
    ):
        raise ValueError(f"{label} must be a JSON list of objects")
    return value


def _validated_id_list(
    value: Any,
    *,
    known: set[str],
    label: str,
    minimum: int,
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON list")
    result = [str(item).strip() for item in value]
    if (
        len(result) < minimum
        or any(not item for item in result)
        or len(result) != len(set(result))
    ):
        raise ValueError(
            f"{label} requires at least {minimum} unique non-empty IDs"
        )
    unknown = sorted(set(result) - known)
    if unknown:
        raise ValueError(f"{label} references unknown object IDs: {unknown}")
    return result


def _known_id(value: Any, known: set[str]) -> str:
    result = str(value or "").strip()
    if not result or result not in known:
        raise ValueError(
            f"functional discovery references unknown object ID {result!r}"
        )
    return result


def _allowed_token(
    value: Any,
    allowed: frozenset[str],
    label: str,
) -> str:
    result = str(value or "").strip()
    if result not in allowed:
        raise ValueError(
            f"{label} must be one of {sorted(allowed)}, got {result!r}"
        )
    return result


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a JSON boolean")
    return value


def _validated_text_list(
    value: Any,
    *,
    allowed: frozenset[str],
    label: str,
) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty JSON list")
    result = [str(item).strip() for item in value]
    if any(not item for item in result) or len(result) != len(set(result)):
        raise ValueError(f"{label} must contain unique non-empty strings")
    unknown = sorted(set(result) - allowed)
    if unknown:
        raise ValueError(f"{label} contains unsupported values: {unknown}")
    return result


def _required_text(value: Any, label: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{label} must be non-empty")
    return result[:1000]


def _reject_forbidden_fields(value: Any, *, path: str = "response") -> None:
    if isinstance(value, dict):
        for raw_key, item in value.items():
            key = str(raw_key).strip().lower()
            if key in _FORBIDDEN_FIELDS:
                raise ValueError(
                    f"functional discovery may not return {path}.{raw_key}"
                )
            _reject_forbidden_fields(item, path=f"{path}.{raw_key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_fields(item, path=f"{path}[{index}]")


__all__ = [
    "compose_functional_discovery_result",
    "salvage_functional_affordance_response",
    "salvage_functional_relation_response",
    "validate_functional_affordance_response",
    "validate_functional_discovery_request",
    "validate_functional_discovery_response",
    "validate_functional_relation_response",
]
