"""Frozen reference annotations for benchmark scoring.

The natural-language converter is never benchmark ground truth. It may emit an
annotation *draft*, but official scoring must consume a *frozen* annotation that
records only confirmed prompt claims. A converter extraction failure therefore
never becomes a generator penalty: an unconfirmed or incomplete annotation is
excluded from official scoring instead of being silently evaluated.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from benchmark.relation_identity import normalize_relation_id, provisional_relation_id, with_relation_ids


REFERENCE_ANNOTATION_VERSION = "reference_annotation_v1"

CLAIM_STATES = ("confirmed", "unresolved_annotation", "not_mentioned")
INVENTORY_POLICIES = ("closed_world", "open_world")
ANNOTATION_STATUSES = ("draft", "confirmed", "reference_annotation_incomplete")
ANNOTATION_SOURCES = ("programmatic", "manual", "model_assisted")
REVIEW_STATUSES = ("pending", "approved")

CONFIRMED = "confirmed"
UNRESOLVED_ANNOTATION = "unresolved_annotation"
NOT_MENTIONED = "not_mentioned"


class ReferenceAnnotationError(ValueError):
    """Raised when a reference annotation is structurally malformed."""


def build_reference_annotation_draft(
    object_plan: dict,
    scene_request: dict | None = None,
    *,
    source: str = "programmatic",
) -> dict[str, Any]:
    """Project a converter object_plan into an annotation *draft*.

    The result is explicitly ``validation_status="draft"`` and is not eligible
    for official scoring. Freezing (review/confirmation, or preserving a
    programmatic structural spec) is a separate, deliberate step.
    """

    if not isinstance(object_plan, dict):
        raise ReferenceAnnotationError("object_plan must be a JSON object")
    request = scene_request if isinstance(scene_request, dict) else {}
    request_id = str(object_plan.get("request_id") or request.get("request_id") or "").strip()
    scene_type = str(object_plan.get("scene_type") or request.get("scene_type") or "room").strip() or "room"

    objects: list[dict[str, Any]] = []
    for index, obj in enumerate(object_plan.get("objects", []) if isinstance(object_plan.get("objects"), list) else []):
        if not isinstance(obj, dict):
            continue
        objects.append(
            {
                "id": str(obj.get("id") or f"obj_{index:03d}"),
                "category": str(obj.get("category") or "object"),
                "description": str(obj.get("description") or ""),
                "count": _positive_count(obj.get("count")),
                # A draft records the extracted claims as unconfirmed by default;
                # confirmation is a human/spec-owned decision, not a converter one.
                "claim_state": UNRESOLVED_ANNOTATION,
                "provenance": {
                    "origin": "converter_draft",
                    "object_plan_index": index,
                },
            }
        )

    oor_relations: list[dict[str, Any]] = []
    oar_relations: list[dict[str, Any]] = []
    relation_counters = {"oor": 0, "oar": 0}
    for relation in object_plan.get("relations", []) if isinstance(object_plan.get("relations"), list) else []:
        if not isinstance(relation, dict):
            continue
        family = str(relation.get("family") or "").strip().lower()
        resolved_family = "oar" if family == "oar" else "oor"
        relation_index = relation_counters[resolved_family]
        relation_counters[resolved_family] += 1
        relation_id = normalize_relation_id(relation.get("relation_id")) or provisional_relation_id(
            resolved_family,
            relation_index,
        )
        subject_id = _first_present(relation, ["subject_id", "subject"])
        if family == "oar":
            relation_type = str(_first_present(relation, ["type", "predicate", "relation"]) or "")
            architecture_target = str(
                _first_present(relation, ["architectural_element", "object_id", "object", "target"])
                or _default_architecture_target(relation_type, relation)
            )
            oar_relations.append(
                {
                    "relation_id": relation_id,
                    "subject_id": str(subject_id or ""),
                    "type": relation_type,
                    "architectural_element": architecture_target,
                    "claim_state": UNRESOLVED_ANNOTATION,
                    **_copy_relation_metadata(relation),
                }
            )
        else:
            claim: dict[str, Any] = {
                "relation_id": relation_id,
                "type": str(_first_present(relation, ["type", "predicate", "relation"]) or ""),
                "claim_state": UNRESOLVED_ANNOTATION,
                **_copy_relation_metadata(relation),
            }
            if subject_id is not None:
                claim["subject_id"] = str(subject_id)
            anchor = _first_present(relation, ["object_id", "object", "anchor_id", "target_id"])
            if anchor is not None:
                claim["object_id"] = str(anchor)
            for identity_key in ("subject_ids", "object_ids", "member_ids"):
                if isinstance(relation.get(identity_key), list):
                    claim[identity_key] = [str(value) for value in relation[identity_key]]
            oor_relations.append(claim)

    draft = {
        "annotation_version": REFERENCE_ANNOTATION_VERSION,
        "validation_status": "draft",
        "source": str(source),
        "request_id": request_id,
        "scene_type": scene_type,
        "objects": objects,
        "oor_relations": oor_relations,
        "oar_relations": oar_relations,
        "room_constraints": {"claim_state": NOT_MENTIONED},
        "provenance": {"origin": "converter_draft"},
    }
    if source == "model_assisted":
        draft["review"] = {
            "status": "pending",
            "reviewer": None,
            "reviewed_at": None,
        }
    return draft


def confirm_reference_annotation(
    annotation: dict,
    *,
    inventory_policy: str,
    confirmed_object_ids: list[str] | tuple[str, ...] | set[str] | None = None,
    reviewer: str | None = None,
    reviewed_at: str | None = None,
) -> dict[str, Any]:
    """Freeze a draft into a confirmed annotation.

    Programmatic/manual objects and relations are promoted to ``confirmed``
    (optionally restricted to an explicit id allow-list). Model-assisted drafts
    are different: they must already contain human-edited claim states, and this
    function delegates to approval without promoting any extraction.
    """

    if inventory_policy not in INVENTORY_POLICIES:
        raise ReferenceAnnotationError(f"inventory_policy must be one of {list(INVENTORY_POLICIES)}")
    if annotation.get("source") == "model_assisted":
        if confirmed_object_ids is not None:
            raise ReferenceAnnotationError(
                "model-assisted annotations cannot batch-confirm object IDs; "
                "edit claim_state values during human review"
            )
        return approve_reference_annotation(
            annotation,
            inventory_policy=inventory_policy,
            reviewer=str(reviewer or ""),
            reviewed_at=reviewed_at,
        )
    frozen = ensure_reference_relation_ids(annotation)
    allow = {str(value) for value in confirmed_object_ids} if confirmed_object_ids is not None else None
    for obj in frozen.get("objects", []) if isinstance(frozen.get("objects"), list) else []:
        if not isinstance(obj, dict):
            continue
        if allow is None or str(obj.get("id")) in allow:
            obj["claim_state"] = CONFIRMED
    confirmed_ids = {
        str(obj.get("id"))
        for obj in frozen.get("objects", []) if isinstance(frozen.get("objects"), list)
        if isinstance(obj, dict) and obj.get("claim_state") == CONFIRMED
    }
    for key in ("oor_relations", "oar_relations"):
        for relation in frozen.get(key, []) if isinstance(frozen.get(key), list) else []:
            if isinstance(relation, dict):
                required_ids = _relation_reference_ids(
                    relation,
                    family="oar" if key == "oar_relations" else "oor",
                )
                relation["claim_state"] = (
                    CONFIRMED if required_ids and required_ids <= confirmed_ids else UNRESOLVED_ANNOTATION
                )
    frozen["validation_status"] = "confirmed"
    frozen["inventory_policy"] = inventory_policy
    return validate_reference_annotation(frozen)


def approve_reference_annotation(
    annotation: dict,
    *,
    inventory_policy: str,
    reviewer: str,
    reviewed_at: str | None = None,
) -> dict[str, Any]:
    """Freeze a human-reviewed draft without inferring or promoting claims.

    The reviewer must explicitly edit each claim state before approval. This
    function only records the review decision and inventory policy; unresolved
    converter/model extractions remain unresolved and cannot become scoring
    ground truth by virtue of running the approval command.
    """

    validate_reference_annotation(annotation)
    if inventory_policy not in INVENTORY_POLICIES:
        raise ReferenceAnnotationError(
            f"inventory_policy must be one of {list(INVENTORY_POLICIES)}"
        )
    reviewer_name = str(reviewer or "").strip()
    if not reviewer_name:
        raise ReferenceAnnotationError("reference annotation approval requires a human reviewer")
    if annotation.get("known_incomplete") is True:
        raise ReferenceAnnotationError(
            "a known-incomplete annotation cannot be approved for official scoring"
        )

    frozen = ensure_reference_relation_ids(annotation)
    frozen["validation_status"] = "confirmed"
    frozen["inventory_policy"] = inventory_policy
    frozen["review"] = {
        "status": "approved",
        "reviewer": reviewer_name,
        "reviewed_at": str(reviewed_at).strip() if reviewed_at is not None else None,
    }
    provenance = frozen.get("provenance")
    frozen["provenance"] = {
        **(provenance if isinstance(provenance, dict) else {}),
        "approval_mode": "human_review_preserving_claim_states",
    }
    return validate_reference_annotation(frozen)


def validate_reference_annotation(annotation: dict) -> dict[str, Any]:
    """Structurally validate a reference annotation.

    Does not decide scoring eligibility (see :func:`is_official_scoreable`); a
    valid annotation can still be a ``draft`` or ``reference_annotation_incomplete``.
    """

    if not isinstance(annotation, dict):
        raise ReferenceAnnotationError("reference_annotation must be a JSON object")
    if annotation.get("annotation_version") != REFERENCE_ANNOTATION_VERSION:
        raise ReferenceAnnotationError(
            f"reference_annotation.annotation_version must be {REFERENCE_ANNOTATION_VERSION!r}"
        )
    status = annotation.get("validation_status")
    if status not in ANNOTATION_STATUSES:
        raise ReferenceAnnotationError(
            f"reference_annotation.validation_status must be one of {list(ANNOTATION_STATUSES)}"
        )
    if annotation.get("source") not in ANNOTATION_SOURCES:
        raise ReferenceAnnotationError(
            f"reference_annotation.source must be one of {list(ANNOTATION_SOURCES)}"
        )
    review = annotation.get("review")
    if review is not None:
        if not isinstance(review, dict):
            raise ReferenceAnnotationError("reference_annotation.review must be a JSON object")
        if review.get("status") not in REVIEW_STATUSES:
            raise ReferenceAnnotationError(
                f"reference_annotation.review.status must be one of {list(REVIEW_STATUSES)}"
            )
        if review.get("status") == "approved":
            _require_non_empty_string(review.get("reviewer"), "reference_annotation.review.reviewer")
        elif review.get("reviewer") is not None and str(review.get("reviewer")).strip():
            _require_non_empty_string(review.get("reviewer"), "reference_annotation.review.reviewer")
        if review.get("reviewed_at") is not None and str(review.get("reviewed_at")).strip():
            _require_non_empty_string(review.get("reviewed_at"), "reference_annotation.review.reviewed_at")
    if annotation.get("known_incomplete") is not None and not isinstance(annotation.get("known_incomplete"), bool):
        raise ReferenceAnnotationError("reference_annotation.known_incomplete must be boolean")
    _require_non_empty_string(annotation.get("request_id"), "reference_annotation.request_id")
    _require_non_empty_string(annotation.get("scene_type"), "reference_annotation.scene_type")

    objects = annotation.get("objects")
    if not isinstance(objects, list):
        raise ReferenceAnnotationError("reference_annotation.objects must be a JSON list")
    object_ids: set[str] = set()
    for index, obj in enumerate(objects):
        path = f"reference_annotation.objects[{index}]"
        if not isinstance(obj, dict):
            raise ReferenceAnnotationError(f"{path} must be a JSON object")
        object_id = _require_non_empty_string(obj.get("id"), f"{path}.id")
        if object_id in object_ids:
            raise ReferenceAnnotationError(f"{path}.id duplicates reference object id {object_id!r}")
        object_ids.add(object_id)
        _require_non_empty_string(obj.get("category"), f"{path}.category")
        if obj.get("description") is not None and str(obj.get("description")).strip():
            _require_non_empty_string(obj.get("description"), f"{path}.description")
        _require_positive_integer(obj.get("count"), f"{path}.count")
        _require_claim_state(obj.get("claim_state"), f"{path}.claim_state")
        for attr_index, attribute in enumerate(obj.get("attributes", []) if isinstance(obj.get("attributes"), list) else []):
            attr_path = f"{path}.attributes[{attr_index}]"
            if not isinstance(attribute, dict):
                raise ReferenceAnnotationError(f"{attr_path} must be a JSON object")
            _require_non_empty_string(attribute.get("name"), f"{attr_path}.name")
            _require_claim_state(attribute.get("claim_state"), f"{attr_path}.claim_state")

    seen_relation_ids: set[str] = set()
    for key in ("oor_relations", "oar_relations"):
        relations = annotation.get(key, [])
        if relations is None:
            continue
        if not isinstance(relations, list):
            raise ReferenceAnnotationError(f"reference_annotation.{key} must be a JSON list")
        for index, relation in enumerate(relations):
            path = f"reference_annotation.{key}[{index}]"
            if not isinstance(relation, dict):
                raise ReferenceAnnotationError(f"{path} must be a JSON object")
            relation_id = normalize_relation_id(relation.get("relation_id"))
            if relation_id is not None:
                if relation_id in seen_relation_ids:
                    raise ReferenceAnnotationError(
                        f"{path}.relation_id duplicates relation id {relation_id!r}"
                    )
                seen_relation_ids.add(relation_id)
            relation_type = _require_non_empty_string(relation.get("type"), f"{path}.type")
            claim_state = _require_claim_state(relation.get("claim_state"), f"{path}.claim_state")
            if key == "oar_relations":
                subject_id = _require_non_empty_string(relation.get("subject_id"), f"{path}.subject_id")
                if subject_id not in object_ids:
                    raise ReferenceAnnotationError(f"{path}.subject_id references unknown object id {subject_id!r}")
                _require_non_empty_string(relation.get("architectural_element"), f"{path}.architectural_element")
            else:
                _validate_reference_oor_identity_contract(
                    relation,
                    relation_type=relation_type,
                    valid_object_ids=object_ids,
                    path=path,
                )
            if claim_state == CONFIRMED:
                referenced_ids = _relation_reference_ids(
                    relation,
                    family="oar" if key == "oar_relations" else "oor",
                )
                confirmed_ids = {
                    str(obj.get("id"))
                    for obj in objects
                    if isinstance(obj, dict) and obj.get("claim_state") == CONFIRMED
                }
                if not referenced_ids <= confirmed_ids:
                    raise ReferenceAnnotationError(
                        f"{path} is confirmed but references an object claim that is not confirmed"
                    )

    room = annotation.get("room_constraints")
    if not isinstance(room, dict):
        raise ReferenceAnnotationError("reference_annotation.room_constraints must be a JSON object")
    _require_claim_state(room.get("claim_state"), "reference_annotation.room_constraints.claim_state")
    dimensions = room.get("dimensions")
    if dimensions is not None:
        if not isinstance(dimensions, dict):
            raise ReferenceAnnotationError("reference_annotation.room_constraints.dimensions must be a JSON object")
        for axis in ("width", "depth", "height"):
            if dimensions.get(axis) is not None:
                _require_positive_number(
                    dimensions[axis],
                    f"reference_annotation.room_constraints.dimensions.{axis}",
                )
    if room.get("shape") is not None:
        _require_non_empty_string(room.get("shape"), "reference_annotation.room_constraints.shape")

    if status == "confirmed":
        if annotation.get("known_incomplete") is True:
            raise ReferenceAnnotationError(
                "a confirmed reference_annotation must not also be flagged known_incomplete"
            )
        if annotation.get("inventory_policy") not in INVENTORY_POLICIES:
            raise ReferenceAnnotationError(
                f"a confirmed reference_annotation requires inventory_policy in {list(INVENTORY_POLICIES)}"
            )
        if annotation.get("inventory_policy") == "closed_world":
            unresolved_inventory = [
                str(obj.get("id") or "")
                for obj in objects
                if isinstance(obj, dict) and obj.get("claim_state") != CONFIRMED
            ]
            if unresolved_inventory:
                raise ReferenceAnnotationError(
                    "closed_world reference annotations require every listed object claim "
                    f"to be confirmed; unresolved object ids: {unresolved_inventory}"
                )
    elif annotation.get("inventory_policy") is not None and annotation.get("inventory_policy") not in INVENTORY_POLICIES:
        raise ReferenceAnnotationError(
            f"reference_annotation.inventory_policy must be one of {list(INVENTORY_POLICIES)}"
        )
    return annotation


def annotation_scoring_gate(annotation: dict) -> dict[str, Any]:
    """Return the frozen scoring-eligibility decision for an annotation."""

    validate_reference_annotation(annotation)
    status = annotation.get("validation_status")
    if annotation.get("known_incomplete") is True or status == "reference_annotation_incomplete":
        return {
            "official_scoreable": False,
            "status": "reference_annotation_incomplete",
            "reason": "reference_annotation_incomplete",
        }
    if status != "confirmed":
        return {
            "official_scoreable": False,
            "status": "unconfirmed_reference_annotation",
            "reason": "reference_annotation_not_confirmed",
        }
    if annotation.get("source") == "model_assisted":
        review = annotation.get("review")
        if not (
            isinstance(review, dict)
            and review.get("status") == "approved"
            and str(review.get("reviewer") or "").strip()
        ):
            return {
                "official_scoreable": False,
                "status": "human_review_required",
                "reason": "model_assisted_reference_annotation_not_human_reviewed",
            }
    return {"official_scoreable": True, "status": "confirmed", "reason": None}


def is_official_scoreable(annotation: dict) -> bool:
    return bool(annotation_scoring_gate(annotation)["official_scoreable"])


def ensure_reference_relation_ids(annotation: dict) -> dict[str, Any]:
    """Upgrade legacy annotations to the stable relation-ID contract.

    The function is deliberately copy-on-write. Official callers can accept old
    calibration fixtures without mutating the source artifact, while all new
    downstream routing and reports receive explicit claim identity.
    """

    if not isinstance(annotation, dict):
        raise ReferenceAnnotationError("reference_annotation must be a JSON object")
    upgraded = deepcopy(annotation)
    try:
        upgraded["oor_relations"] = with_relation_ids(
            upgraded.get("oor_relations", []) if isinstance(upgraded.get("oor_relations"), list) else [],
            family="oor",
        )
        upgraded["oar_relations"] = with_relation_ids(
            upgraded.get("oar_relations", []) if isinstance(upgraded.get("oar_relations"), list) else [],
            family="oar",
        )
    except ValueError as exc:
        raise ReferenceAnnotationError(str(exc)) from exc
    return upgraded


def confirmed_objects(annotation: dict) -> list[dict[str, Any]]:
    return [
        obj
        for obj in annotation.get("objects", []) if isinstance(annotation.get("objects"), list)
        if isinstance(obj, dict) and obj.get("claim_state") == CONFIRMED
    ]


def confirmed_oor_relations(annotation: dict) -> list[dict[str, Any]]:
    return _confirmed_relations(annotation, "oor_relations")


def confirmed_oar_relations(annotation: dict) -> list[dict[str, Any]]:
    return _confirmed_relations(annotation, "oar_relations")


def confirmed_room_constraints(annotation: dict) -> dict[str, Any] | None:
    room = annotation.get("room_constraints")
    if isinstance(room, dict) and room.get("claim_state") == CONFIRMED:
        return room
    return None


def object_plan_from_reference_annotation(annotation: dict) -> dict[str, Any]:
    """Build a deterministic-mapping object_plan from confirmed reference objects."""

    annotation = ensure_reference_relation_ids(annotation)
    objects = [
        {
            "id": str(obj.get("id")),
            "role": "",
            "category": str(obj.get("category") or ""),
            "description": str(obj.get("description") or obj.get("category") or ""),
            "count": _positive_count(obj.get("count")),
            "placement_intent": {"absolute_relations": [], "relative_relations": []},
            "metadata": {},
        }
        for obj in confirmed_objects(annotation)
    ]
    relations = [
        _project_reference_relation(relation, family="oor", target_key="object_plan")
        for relation in confirmed_oor_relations(annotation)
    ]
    relations.extend(
        _project_reference_relation(relation, family="oar", target_key="object_plan")
        for relation in confirmed_oar_relations(annotation)
    )
    return {
        "request_id": str(annotation.get("request_id") or ""),
        "scene_type": str(annotation.get("scene_type") or "room"),
        "scene_description": "",
        "prompt_granularity": "fine_grained",
        "explicit_claims": [],
        "objects": objects,
        "global_constraints": [],
        "relations": relations,
    }


def relationship_intents_from_reference_annotation(annotation: dict) -> dict[str, Any]:
    """Project only confirmed relation claims into deterministic routing input."""

    annotation = ensure_reference_relation_ids(annotation)
    validate_reference_annotation(annotation)
    return {
        "request_id": str(annotation.get("request_id") or ""),
        "status": "confirmed_reference_annotation",
        "oor_relations": [
            _project_reference_relation(relation, family="oor", target_key="intent")
            for relation in confirmed_oor_relations(annotation)
        ],
        "oar_relations": [
            _project_reference_relation(relation, family="oar", target_key="intent")
            for relation in confirmed_oar_relations(annotation)
        ],
        "unsupported_relations": [],
        "notes": ["Only confirmed frozen relation claims are eligible for routing."],
    }


def _confirmed_relations(annotation: dict, key: str) -> list[dict[str, Any]]:
    relations = annotation.get(key)
    if not isinstance(relations, list):
        return []
    return [item for item in relations if isinstance(item, dict) and item.get("claim_state") == CONFIRMED]


def _project_reference_relation(
    relation: dict[str, Any],
    *,
    family: str,
    target_key: str,
) -> dict[str, Any]:
    result = {
        key: deepcopy(value)
        for key, value in relation.items()
        if key not in {"claim_state", "provenance", "architectural_element"}
    }
    result["family"] = family
    if family == "oar":
        result["target"] = str(relation.get("architectural_element") or "")
    if target_key == "intent":
        result["source"] = "frozen_reference_annotation"
    return result


def _copy_relation_metadata(relation: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(relation[key])
        for key in (
            "raw_relation",
            "axis",
            "direction",
            "order",
            "wall",
            "corner",
            "region",
        )
        if relation.get(key) is not None
    }


def _default_architecture_target(relation_type: str, relation: dict[str, Any]) -> str:
    if relation.get("wall"):
        return str(relation["wall"])
    if relation.get("corner"):
        return str(relation["corner"])
    if relation.get("region"):
        return str(relation["region"])
    relation_targets = {
        "on_floor": "floor",
        "room_center": "center_region",
        "attached_to_ceiling": "ceiling",
        "hung_from_ceiling": "ceiling",
    }
    return relation_targets.get(str(relation_type), "room_architecture")


def _relation_reference_ids(relation: dict[str, Any], *, family: str) -> set[str]:
    values: list[Any] = []
    if relation.get("subject_id") is not None:
        values.append(relation["subject_id"])
    if family == "oor" and relation.get("object_id") is not None:
        values.append(relation["object_id"])
    if family == "oor":
        for key in ("subject_ids", "object_ids", "member_ids"):
            if isinstance(relation.get(key), list):
                values.extend(relation[key])
    return {str(value) for value in values if str(value).strip()}


def _validate_reference_oor_identity_contract(
    relation: dict[str, Any],
    *,
    relation_type: str,
    valid_object_ids: set[str],
    path: str,
) -> None:
    subject_ids = _reference_id_list(relation.get("subject_ids"), f"{path}.subject_ids", valid_object_ids)
    object_ids = _reference_id_list(relation.get("object_ids"), f"{path}.object_ids", valid_object_ids)

    if relation_type == "between":
        subject = _require_reference_id(relation.get("subject_id"), f"{path}.subject_id", valid_object_ids)
        if len(object_ids) != 2:
            raise ReferenceAnnotationError(f"{path}.object_ids must contain exactly two IDs for between")
        _ = subject
        return
    if relation_type == "ordered":
        if len(object_ids) < 2:
            raise ReferenceAnnotationError(f"{path}.object_ids must contain at least two IDs for ordered")
        if not str(relation.get("axis") or relation.get("direction") or "").strip():
            raise ReferenceAnnotationError(f"{path} requires axis or direction for ordered")
        return
    if relation_type == "around":
        if not subject_ids:
            raise ReferenceAnnotationError(f"{path}.subject_ids must contain at least one plan ID for around")
        _require_reference_id(relation.get("object_id"), f"{path}.object_id", valid_object_ids)
        return

    if relation.get("subject_id") is not None and relation.get("object_id") is not None:
        _require_reference_id(relation.get("subject_id"), f"{path}.subject_id", valid_object_ids)
        _require_reference_id(relation.get("object_id"), f"{path}.object_id", valid_object_ids)
        return
    if len(subject_ids) + len(object_ids) < 2:
        raise ReferenceAnnotationError(f"{path} must reference at least two object IDs")


def _reference_id_list(value: Any, path: str, valid_object_ids: set[str]) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ReferenceAnnotationError(f"{path} must be a JSON list")
    return [
        _require_reference_id(item, f"{path}[{index}]", valid_object_ids)
        for index, item in enumerate(value)
    ]


def _require_reference_id(value: Any, path: str, valid_object_ids: set[str]) -> str:
    object_id = _require_non_empty_string(value, path)
    if object_id not in valid_object_ids:
        raise ReferenceAnnotationError(f"{path} references unknown object id {object_id!r}")
    return object_id


def _require_claim_state(value: Any, path: str) -> str:
    if value not in CLAIM_STATES:
        raise ReferenceAnnotationError(f"{path} must be one of {list(CLAIM_STATES)}")
    return str(value)


def _require_non_empty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReferenceAnnotationError(f"{path} must be a non-empty string")
    return value


def _require_positive_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReferenceAnnotationError(f"{path} must be a positive integer")
    return value


def _require_positive_number(value: Any, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ReferenceAnnotationError(f"{path} must be a positive number")
    return float(value)


def _positive_count(value: Any) -> int:
    if isinstance(value, bool):
        return 1
    try:
        count = int(value or 1)
    except (TypeError, ValueError):
        return 1
    return max(1, count)


def _first_present(mapping: dict, keys: list[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None
