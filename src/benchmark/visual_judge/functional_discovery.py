"""Strict, non-judging discovery for functional visual evidence.

The discovery model answers *what should be inspected*, never whether the
scene is valid.  Object-to-group ownership is recomputed from the trusted
grouping partition so a VLM cannot silently redefine evidence scopes.
"""

from __future__ import annotations

import base64
import json
import time
from copy import deepcopy
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

from benchmark.models import OpenAICompatibleModel, parse_json_object


FUNCTIONAL_DISCOVERY_SCHEMA_VERSION = "functional_discovery_v3"
FUNCTIONAL_DISCOVERY_PROMPT_VERSION = "functional_discovery_v4"
FUNCTIONAL_AFFORDANCE_SCHEMA_VERSION = "functional_affordance_ledger_v2"
FUNCTIONAL_AFFORDANCE_PROMPT_VERSION = "functional_affordance_ledger_v3"
FUNCTIONAL_RELATION_SCHEMA_VERSION = "functional_relation_audit_v1"
FUNCTIONAL_RELATION_PROMPT_VERSION = "functional_relation_audit_v2"
FUNCTIONAL_DISCOVERY_MAX_TOKENS = 4096

FUNCTIONAL_SURFACE_ROLES = frozenset(
    {
        "access_side",
        "opening_side",
        "control_side",
        "display_side",
        "seating_side",
        "interaction_side",
        "reflective_side",
        "service_side",
    }
)
FUNCTIONAL_RELATION_OBSERVATIONS = frozenset(
    {
        "mutual_orientation",
        "cooperative_operation",
        "operational_access",
        "shared_task_reach",
        "attachment_or_service_relation",
    }
)
FUNCTIONAL_DIRECTIONALITY = frozenset(
    {
        "directed",
        "omnidirectional",
        "no_ordinary_operation",
        "uncertain",
    }
)
FUNCTIONAL_CLEARANCE_NEEDS = frozenset(
    {"none", "approach", "opening", "operation", "uncertain"}
)
FUNCTIONAL_REVIEW_STATES = frozenset(
    {"routine", "local_confirmation", "uncertain"}
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


FUNCTIONAL_AFFORDANCE_SYSTEM_PROMPT = """Inventory functional observability;
do not judge validity. For every input object, decide whether ordinary use has
a visually identifiable directed side, is direction-independent, has no
ordinary operation, or is uncertain. Record only the allowed surface roles and
the kind of user clearance that must be observable. Separately record whether
that use direction or approach region must be observed together with the
supplied logical room boundary; this is an observation need, not a validity
claim. Use local_confirmation only when another local view could resolve an
observable affordance. Category is a navigation hint; visible geometry and
appearance are the basis for affordance.
When the overview cannot distinguish directedness, use uncertain; never turn
missing visibility into omnidirectional or no_ordinary_operation. An uncertain
row may retain one or more plausible surface roles when the ordinary operation
is identifiable but its directed side is not. Leave surface_roles empty only
when even the role is not identifiable.

Copy every object ID exactly once in input order. Return exactly:
{"objects":[{"object_id":"id",
"directionality":"directed",
"surface_roles":["access_side"],
"clearance_need":"none",
"boundary_review_state":"routine",
"review_state":"routine",
"observation_goal":"neutral observable fact",
"boundary_observation_goal":""}],
"reason":"brief coverage summary"}

Use only the supplied vocabulary. Never return a defect, score, verdict, pose,
vector, camera action, or scene edit. Return no other fields."""


FUNCTIONAL_RELATION_SYSTEM_PROMPT = """Inventory direct ordinary joint-use
relations; do not judge whether the current arrangement is valid. A relation
requires at least one allowed observable dependency: mutual_orientation,
cooperative_operation, operational_access, shared_task_reach, or
attachment_or_service_relation. Distance and group membership affect framing
only and cannot suppress such a relation. Co-presence, semantic similarity,
style compatibility, or possible usefulness together are insufficient.
Actively test each object for a direct ordinary joint-use counterpart before
returning no relation. Current misorientation must not suppress the underlying
relation whose orientation needs inspection.

Copy every object ID exactly once in considered_object_ids. Return sparse
trusted-ID relation sets, allowed observation tokens, and neutral observation
goals. Return exactly:
{"considered_object_ids":["id"],
"relations":[{"target_ids":["id1","id2"],
"observation_kinds":["mutual_orientation"],
"observation_goal":"neutral joint-use observation"}],
"reason":"brief coverage summary"}

Never return a defect, score, verdict, pose, vector, camera action, or scene
edit. Return no other fields."""


@dataclass(frozen=True)
class FunctionalDiscoveryResult:
    inspected_object_ids: tuple[str, ...]
    object_affordance_ledger: tuple[dict[str, Any], ...] = ()
    directed_surface_targets: tuple[dict[str, Any], ...] = ()
    within_group_correspondences: tuple[dict[str, Any], ...] = ()
    cross_group_correspondences: tuple[dict[str, Any], ...] = ()
    approach_clearance_targets: tuple[dict[str, Any], ...] = ()
    boundary_sensitive_targets: tuple[dict[str, Any], ...] = ()
    unusual_unconfirmed: tuple[dict[str, Any], ...] = ()
    reason: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FUNCTIONAL_DISCOVERY_SCHEMA_VERSION,
            "inspected_object_ids": list(self.inspected_object_ids),
            "object_coverage": [
                {
                    "object_id": object_id,
                    "inspected": True,
                    **deepcopy(
                        next(
                            (
                                {
                                    key: value
                                    for key, value in item.items()
                                    if key != "object_id"
                                }
                                for item in self.object_affordance_ledger
                                if item.get("object_id") == object_id
                            ),
                            {},
                        )
                    ),
                }
                for object_id in self.inspected_object_ids
            ],
            "object_affordance_ledger": list(
                deepcopy(self.object_affordance_ledger)
            ),
            "directed_surface_targets": list(
                deepcopy(self.directed_surface_targets)
            ),
            "within_group_correspondences": list(
                deepcopy(self.within_group_correspondences)
            ),
            "cross_group_correspondences": list(
                deepcopy(self.cross_group_correspondences)
            ),
            "approach_clearance_targets": list(
                deepcopy(self.approach_clearance_targets)
            ),
            "boundary_sensitive_targets": list(
                deepcopy(self.boundary_sensitive_targets)
            ),
            "unusual_unconfirmed": list(
                deepcopy(self.unusual_unconfirmed)
            ),
            "reason": self.reason,
            "decision_authority": "none",
            "provenance": deepcopy(self.provenance),
        }


def discover_openai_compatible_functional_evidence(
    *,
    model: OpenAICompatibleModel,
    request: dict[str, Any],
    max_context_chars: int = 30000,
    response_format_json: bool | None = None,
) -> dict[str, Any]:
    """Run two focused non-judging discovery audits and compose one result."""

    normalized = validate_functional_discovery_request(request)
    common = {
        "metric": "functional_consistency",
        "decision_authority": "none",
        "scene_access": "read_only",
        "scene_id": normalized.get("scene_id"),
        "scene_type": normalized.get("scene_type"),
        "object_list": deepcopy(normalized["objects"]),
    }
    use_json_response = (
        bool(getattr(model, "response_format_json", True))
        if response_format_json is None
        else bool(response_format_json)
    )
    affordance_raw, affordance_meta = _run_discovery_call(
        model=model,
        normalized=normalized,
        context={
            **common,
            "role": "functional_affordance_ledger",
            "prompt_version": FUNCTIONAL_AFFORDANCE_PROMPT_VERSION,
            "schema_version": FUNCTIONAL_AFFORDANCE_SCHEMA_VERSION,
            "architecture_context": deepcopy(
                normalized["architecture_context"]
            ),
            "allowed_surface_roles": sorted(FUNCTIONAL_SURFACE_ROLES),
            "allowed_directionality": sorted(FUNCTIONAL_DIRECTIONALITY),
            "allowed_clearance_needs": sorted(FUNCTIONAL_CLEARANCE_NEEDS),
            "allowed_review_states": sorted(FUNCTIONAL_REVIEW_STATES),
        },
        system_prompt=FUNCTIONAL_AFFORDANCE_SYSTEM_PROMPT,
        call_type="vlm_camera_pose.functional_discovery.affordance",
        max_context_chars=max_context_chars,
        response_format_json=use_json_response,
    )
    relation_raw, relation_meta = _run_discovery_call(
        model=model,
        normalized=normalized,
        context={
            **common,
            "role": "functional_relation_audit",
            "prompt_version": FUNCTIONAL_RELATION_PROMPT_VERSION,
            "schema_version": FUNCTIONAL_RELATION_SCHEMA_VERSION,
            "allowed_observation_kinds": sorted(
                FUNCTIONAL_RELATION_OBSERVATIONS
            ),
        },
        system_prompt=FUNCTIONAL_RELATION_SYSTEM_PROMPT,
        call_type="vlm_camera_pose.functional_discovery.relations",
        max_context_chars=max_context_chars,
        response_format_json=use_json_response,
    )
    object_ids = tuple(item["id"] for item in normalized["objects"])
    affordance = validate_functional_affordance_response(
        parse_json_object(affordance_raw),
        object_ids=object_ids,
    )
    relations = validate_functional_relation_response(
        parse_json_object(relation_raw),
        object_ids=object_ids,
    )
    result = compose_functional_discovery_result(
        affordance=affordance,
        relations=relations,
        object_ids=tuple(item["id"] for item in normalized["objects"]),
        groups=normalized["groups"],
    )
    return FunctionalDiscoveryResult(
        **result,
        provenance={
            "prompt_version": FUNCTIONAL_DISCOVERY_PROMPT_VERSION,
            "schema_version": FUNCTIONAL_DISCOVERY_SCHEMA_VERSION,
            "backend": "openai_compatible",
            "affordance_prompt_version": (
                FUNCTIONAL_AFFORDANCE_PROMPT_VERSION
            ),
            "affordance_schema_version": (
                FUNCTIONAL_AFFORDANCE_SCHEMA_VERSION
            ),
            "relation_prompt_version": FUNCTIONAL_RELATION_PROMPT_VERSION,
            "relation_schema_version": FUNCTIONAL_RELATION_SCHEMA_VERSION,
            "calls": {
                "affordance": affordance_meta,
                "relations": relation_meta,
            },
            # Additive compatibility: points at the most recent call while the
            # complete per-call records above remain authoritative.
            "request_metadata": deepcopy(relation_meta),
        },
    ).to_dict()


def _run_discovery_call(
    *,
    model: OpenAICompatibleModel,
    normalized: dict[str, Any],
    context: dict[str, Any],
    system_prompt: str,
    call_type: str,
    max_context_chars: int,
    response_format_json: bool,
) -> tuple[str, dict[str, Any]]:
    context_text = json.dumps(
        context,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(context_text) > max(1000, int(max_context_chars)):
        raise ValueError(
            f"{call_type} context exceeds max_context_chars; "
            "implicit truncation is forbidden"
        )
    content = [
        {
            "type": "text",
            "text": "Perform the supplied non-judging audit.\n" + context_text,
        },
        {
            "type": "image_url",
            "image_url": {
                "url": _image_data_url(
                    Path(normalized["global_image_path"]),
                    alias="scene_global",
                )
            },
        },
    ]
    started = time.perf_counter()
    raw = model.chat_messages(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        response_format_json=response_format_json,
        call_type=call_type,
        max_tokens=max(
            FUNCTIONAL_DISCOVERY_MAX_TOKENS,
            int(getattr(model, "max_tokens", None) or 0),
        ),
        max_tokens_source=f"{call_type}.minimum",
        case={
            "case_id": str(
                normalized.get("scene_id") or "functional_discovery"
            ),
            "scene_id": str(normalized.get("scene_id") or ""),
            "objects": deepcopy(normalized["objects"]),
        },
    )
    metadata = dict(getattr(model, "last_request_metadata", {}))
    metadata.update(
        call_type=call_type,
        status="complete",
        latency_seconds=round(time.perf_counter() - started, 6),
    )
    return raw, metadata


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
    for expected_id, item in zip(object_ids, rows, strict=True):
        allowed = {
            "object_id",
            "directionality",
            "surface_roles",
            "clearance_need",
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
        if directionality in {"directed", "uncertain"}:
            if directionality == "directed":
                roles = _validated_text_list(
                    raw_roles,
                    allowed=FUNCTIONAL_SURFACE_ROLES,
                    label="surface_roles",
                )
            elif raw_roles in (None, []):
                roles = []
            else:
                roles = _validated_text_list(
                    raw_roles,
                    allowed=FUNCTIONAL_SURFACE_ROLES,
                    label="surface_roles",
                )
        else:
            if raw_roles not in (None, []):
                raise ValueError(
                    "only directed or uncertain affordances may return "
                    "surface_roles"
                )
            roles = []
        clearance_need = _allowed_token(
            item.get("clearance_need"),
            FUNCTIONAL_CLEARANCE_NEEDS,
            "clearance_need",
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
                "boundary_observation_goal"
            )
        if boundary_state == "routine" and boundary_goal:
            raise ValueError(
                "routine boundary review must not invent a boundary goal"
            )
        normalized.append(
            {
                "object_id": object_id,
                "directionality": directionality,
                "surface_roles": roles,
                "clearance_need": clearance_need,
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
    identities: set[tuple[str, ...]] = set()
    for item in _json_object_list(value.get("relations"), "relations"):
        extra = set(item) - {
            "target_ids",
            "observation_kinds",
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
        kinds = _validated_text_list(
            item.get("observation_kinds"),
            allowed=FUNCTIONAL_RELATION_OBSERVATIONS,
            label="relations.observation_kinds",
        )
        identity = tuple(sorted(target_ids))
        if identity in identities:
            raise ValueError("functional relation audit contains a duplicate")
        identities.add(identity)
        relations.append(
            {
                "target_ids": target_ids,
                "observation_kinds": kinds,
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
        if (
            row["directionality"] == "directed"
            or (
                row["directionality"] == "uncertain"
                and row["surface_roles"]
            )
        ):
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
                    "clearance_need": row["clearance_need"],
                }
            )
        if row["clearance_need"] != "none":
            approach.append(
                {
                    "discovery_id": f"approach_clearance_{index:02d}",
                    "target_id": object_id,
                    "observation_goal": row["observation_goal"],
                    "owning_group_id": group_id,
                    "clearance_need": row["clearance_need"],
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
        if set(item) - {"target_id", "surface_roles", "observation_goal"}:
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
    fingerprints: set[tuple[str, ...]] = set()
    for item in _json_object_list(value, label):
        if set(item) - {"target_ids", "observation_goal"}:
            raise ValueError(f"{label} contains unsupported fields")
        target_ids = _validated_id_list(
            item.get("target_ids"),
            known=known,
            label=f"{label}.target_ids",
            minimum=2,
        )
        fingerprint = tuple(sorted(target_ids))
        if fingerprint in fingerprints:
            raise ValueError(f"{label} contains a duplicate relation")
        fingerprints.add(fingerprint)
        result.append(
            {
                "target_ids": target_ids,
                "observation_goal": _required_text(
                    item.get("observation_goal"),
                    f"{label} observation_goal",
                ),
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
        if set(item) - {"target_id", "observation_goal"}:
            raise ValueError(f"{label} contains unsupported fields")
        target_id = _known_id(item.get("target_id"), known)
        if target_id in seen:
            raise ValueError(f"{label} contains a duplicate target")
        seen.add(target_id)
        result.append(
            {
                "discovery_id": f"{prefix}_{index:02d}",
                "target_id": target_id,
                "observation_goal": _required_text(
                    item.get("observation_goal"),
                    f"{label} observation_goal",
                ),
                "owning_group_id": object_to_group.get(target_id),
            }
        )
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


def _image_data_url(path: Path, *, alias: str) -> str:
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Pillow is required to sanitize discovery images"
        ) from exc
    try:
        with Image.open(path) as source:
            source.load()
            normalized = ImageOps.exif_transpose(source).convert("RGBA")
            flattened = Image.new("RGB", normalized.size, (255, 255, 255))
            flattened.paste(normalized, mask=normalized.getchannel("A"))
            output = BytesIO()
            flattened.save(output, format="PNG")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(
            f"functional discovery image {alias!r} is not decodable"
        ) from exc
    return (
        "data:image/png;base64,"
        + base64.b64encode(output.getvalue()).decode("ascii")
    )
