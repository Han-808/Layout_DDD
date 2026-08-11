"""Pre-judgement visual planning for global functional consistency.

The planner is deliberately narrower than a metric Judge.  It may identify
objects whose functional frontage, correspondence, or approach space deserves
an object-centred view.  It may not return a validity decision or author a
camera pose.  Camera placement remains a deterministic candidate-bank plus
selection problem.
"""

from __future__ import annotations

import base64
import json
import math
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any

from benchmark.models import OpenAICompatibleModel, parse_json_object
from benchmark.visual_judge.roles import (
    DecisionContract,
    VLMRole,
    vlm_audit_metadata,
)


FUNCTIONAL_PROBE_PLAN_VERSION = "functional_probe_plan_v2"
FUNCTIONAL_PROBE_PLANNER_PROMPT_VERSION = (
    "functional_probe_planner_v4"
)
FUNCTIONAL_PROBE_PLANNER_MAX_TOKENS = 3072
FUNCTIONAL_PROBE_KINDS = (
    "functional_frontage",
    "functional_correspondence",
    "approach_clearance",
)
# Cross-group relations receive isolated Judge episodes, while object-owned
# frontage/clearance checks reuse group-local packets.  Six units proved too
# small even for ordinary scenes (for example, three accepted relations plus
# several directed objects), leaving accepted required checks permanently
# unrouted.  Keep the policy bounded, but make the default large enough for a
# complete medium-size scene inventory.
FUNCTIONAL_PROBE_DEFAULT_UNITS = 32
FUNCTIONAL_PROBE_MAX_UNITS = 32

_OBSERVATIONS_BY_KIND = {
    "functional_frontage": frozenset(
        {
            "target_visible",
            "interaction_side_visible",
            "front_back_disambiguated",
            "approach_zone_visible",
            "architecture_plane_visible",
            "global_context_preserved",
            "limited_local_context",
        }
    ),
    "functional_correspondence": frozenset(
        {
            "target_visible",
            "joint_visibility",
            "interaction_side_visible",
            "front_back_disambiguated",
            "approach_zone_visible",
            "group_context_visible",
            "architecture_plane_visible",
            "global_context_preserved",
            "limited_local_context",
        }
    ),
    "approach_clearance": frozenset(
        {
            "target_visible",
            "interaction_side_visible",
            "approach_zone_visible",
            "architecture_plane_visible",
            "global_context_preserved",
            "limited_local_context",
        }
    ),
}
_UNIT_FIELDS = frozenset(
    {
        "kind",
        "target_ids",
        "related_target_ids",
        "required_observations",
        "priority",
        "reason",
    }
)
_FORBIDDEN_PLANNER_FIELDS = frozenset(
    {
        "verdict",
        "validity",
        "status",
        "score",
        "confidence",
        "defect",
        "defects",
        "anomaly",
        "is_invalid",
        "camera",
        "camera_pose",
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
        "mutation",
    }
)

FUNCTIONAL_PROBE_PLANNER_SYSTEM_PROMPT = """You plan bounded pre-judgement
visual evidence for functional_consistency. You are not the Judge.

Use the supplied scene-global image and trusted object list to identify only
the observable facts that need a closer camera view. A probe indicates an
evidence need, never a suspected defect.

Architecture-boundary policy:

- An enabled logical_boundary_xy is the authoritative limit of usable room
  floor and user-approach space. Physical wall meshes may be absent by policy.
- Do not treat a missing physical wall mesh as a defect and do not invent one.
  Space outside an enabled logical boundary is nevertheless unavailable for
  ordinary standing, approach, control access, or operation.
- For an object with a directed usable, opening, control, seating, display, or
  interaction side, check whether the current evidence jointly establishes
  that side, the nearest logical boundary, and an interior-side approach or
  operating region.
- If logical_boundary_enabled is false, do not infer a room boundary from an
  image crop, background transition, or floor edge.

Probe kinds:

1. functional_frontage
   Use for one object whose ordinary operation depends on a directed usable,
   opening, display, seating, control, or interaction side, when the global
   image does not reliably establish that side and the space it faces. When
   boundary proximity matters, the probe must also expose the nearest logical
   boundary or visible floor extent and the interior-side operating region.
2. functional_correspondence
   Use for two or more objects whose ordinary joint use depends on their
   relative facing, interaction orientation, or mutually accessible sides.
   Identify natural, direct functional counterparts before considering their
   distance or grouping membership. At this planning stage, physical distance
   is not a disqualifier: counterparts may occupy different local groups or
   distant parts of the same scene while their ordinary joint use still
   depends on mutual orientation. Semantic co-occurrence, broad category
   relatedness, or possible usefulness together is insufficient; there must be
   a specific ordinary joint-use relationship whose visual correspondence can
   be inspected.
3. approach_clearance
   Use for one object whose ordinary operation depends on an accessible user
   approach zone or operating clearance that the global image does not
   establish. When the object is near an enabled logical boundary, verify that
   the required standing and operating region remains inside that boundary.

Planning procedure:

1. Examine objects in supplied-list order.
2. Enumerate natural direct functional correspondences before deciding which
   facts need additional evidence. A global image that merely contains both
   objects does not establish their correspondence. Treat relative facing or
   interaction orientation as established only when the relevant usable sides
   and the relationship between them are visually unambiguous.
3. Determine which required visual fact is unresolved in the global image. Do
   not create a probe when the global image genuinely establishes that fact.
   Merely seeing open pixels beyond an object does not establish usable
   approach space when the logical boundary is not jointly interpretable.
4. Prefer the smallest probe that resolves the fact.
5. Avoid assigning the same object to multiple probes. When one
   functional_correspondence probe already exposes an object's frontage and
   approach zone, do not create redundant frontage or clearance probes for it.
6. Put natural functional counterparts in one correspondence probe whenever a
   wider or room-spanning view can expose their usable sides and mutual
   orientation. They need not be spatially close or belong to the same local
   group. Distance must affect camera framing, not whether the correspondence
   exists. Do not create a correspondence from semantic relatedness alone.
7. Rank probes by evidence utility: first an unresolved joint orientation or
   directed usable side necessary for ordinary use; then the number of
   distinct objects inspected without losing local detail; then an unresolved
   approach or clearance region; and finally supplied object order as the
   tie-break.
8. Assign unique contiguous priorities 1..N in that order.
9. Do not fill a quota. Return fewer than max_probe_units, including zero,
   when no additional observation is justified.

Use these exact required_observations templates:

functional_frontage:
["target_visible","interaction_side_visible","front_back_disambiguated",
"approach_zone_visible","limited_local_context"]

functional_frontage when boundary relation is unresolved:
["target_visible","interaction_side_visible","front_back_disambiguated",
"approach_zone_visible","architecture_plane_visible",
"global_context_preserved","limited_local_context"]

functional_correspondence:
["target_visible","joint_visibility","interaction_side_visible",
"front_back_disambiguated","approach_zone_visible",
"group_context_visible","limited_local_context"]

approach_clearance:
["target_visible","interaction_side_visible","approach_zone_visible",
"limited_local_context"]

approach_clearance when boundary relation is unresolved:
["target_visible","interaction_side_visible","approach_zone_visible",
"architecture_plane_visible","global_context_preserved",
"limited_local_context"]

Categories are navigation hints only. Do not infer a defect from category,
location, orientation, or probe inclusion. Do not output validity, anomaly,
score, confidence, grouping, camera pose, direction, lens, scene mutation, or
benchmark conclusion.

Each reason must state only which visual fact the probe is intended to expose.
Use only trusted object IDs and the supplied probe kinds and observations.

Return exactly:
{"probe_units":[{"kind":"functional_frontage",
"target_ids":["trusted_object_id"],"related_target_ids":[],
"required_observations":["target_visible","interaction_side_visible",
"front_back_disambiguated","approach_zone_visible",
"limited_local_context"],"priority":1,
"reason":"observable fact this probe should expose"}],
"reason":"brief evidence-coverage explanation"}

Return no other fields. The number of probe_units must not exceed
max_probe_units."""


def plan_openai_compatible_functional_evidence(
    *,
    model: OpenAICompatibleModel,
    request: dict[str, Any],
    max_context_chars: int = 30000,
    response_format_json: bool | None = None,
) -> dict[str, Any]:
    """Run and strictly validate one pre-judgement functional evidence plan."""

    normalized = validate_functional_probe_planning_request(request)
    global_path = Path(normalized["global_image_path"]).expanduser()
    audit = vlm_audit_metadata(
        VLMRole.FUNCTIONAL_EVIDENCE_PLANNER,
        decision_contract=DecisionContract.FUNCTIONAL_PROBE_PLAN,
        judge_method="plan_functional_evidence",
    )
    context = {
        **audit,
        "role": "functional_evidence_planner",
        "prompt_version": FUNCTIONAL_PROBE_PLANNER_PROMPT_VERSION,
        "metric": "functional_consistency",
        "decision_authority": "none",
        "scene_access": "read_only",
        "image_role": "scene_global_visual_context",
        "scene_id": normalized.get("scene_id"),
        "scene_type": normalized.get("scene_type"),
        "architecture_context": deepcopy(
            normalized["architecture_context"]
        ),
        "object_list": deepcopy(normalized["objects"]),
        "max_probe_units": normalized["max_probe_units"],
        "allowed_probe_kinds": list(FUNCTIONAL_PROBE_KINDS),
        "allowed_observations_by_kind": {
            key: sorted(value)
            for key, value in _OBSERVATIONS_BY_KIND.items()
        },
    }
    context_text = json.dumps(
        context,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(context_text) > max(1000, int(max_context_chars)):
        raise ValueError(
            "functional evidence planner context exceeds max_context_chars; "
            "implicit truncation is forbidden"
        )
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Plan pre-judgement functional visual probes from this trusted "
                "request.\n" + context_text
            ),
        },
        {
            "type": "image_url",
            "image_url": {
                "url": _image_data_url(
                    global_path,
                    alias="scene_global",
                )
            },
        },
    ]
    use_json_response = (
        bool(getattr(model, "response_format_json", True))
        if response_format_json is None
        else bool(response_format_json)
    )
    configured_max_tokens = getattr(model, "max_tokens", None)
    planner_max_tokens = max(
        FUNCTIONAL_PROBE_PLANNER_MAX_TOKENS,
        int(configured_max_tokens or 0),
    )
    raw = model.chat_messages(
        [
            {
                "role": "system",
                "content": FUNCTIONAL_PROBE_PLANNER_SYSTEM_PROMPT,
            },
            {"role": "user", "content": content},
        ],
        response_format_json=use_json_response,
        call_type="vlm_camera_pose.functional_probe_plan",
        max_tokens=planner_max_tokens,
        max_tokens_source="functional_probe_planner_minimum",
        case={
            "case_id": str(normalized.get("scene_id") or "functional_probe"),
            "scene_id": str(normalized.get("scene_id") or ""),
            "objects": deepcopy(normalized["objects"]),
        },
    )
    plan = validate_functional_probe_plan_response(
        parse_json_object(raw),
        known_object_ids={
            str(item["id"]) for item in normalized["objects"]
        },
        max_probe_units=int(normalized["max_probe_units"]),
    )
    return {
        **plan,
        "schema_version": FUNCTIONAL_PROBE_PLAN_VERSION,
        "planner_role": "visual_evidence_only_no_metric_verdict",
        "request_metadata": deepcopy(
            {
                **dict(getattr(model, "last_request_metadata", {})),
                **audit,
                "functional_probe_prompt_version": (
                    FUNCTIONAL_PROBE_PLANNER_PROMPT_VERSION
                ),
            }
        ),
    }


def validate_functional_probe_planning_request(
    value: Any,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(
            "functional evidence planning request must be a JSON object"
        )
    if str(value.get("metric") or "") != "functional_consistency":
        raise ValueError(
            "functional evidence planning only supports "
            "functional_consistency"
        )
    global_image = value.get("global_image_path")
    if not isinstance(global_image, (str, Path)) or not str(
        global_image
    ).strip():
        raise ValueError(
            "functional evidence planning requires one global_image_path"
        )
    global_path = Path(str(global_image)).expanduser()
    if not global_path.is_file():
        raise FileNotFoundError(
            "functional evidence planner global image does not exist: "
            f"{global_path}"
        )
    raw_objects = value.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        raise ValueError(
            "functional evidence planning requires a non-empty object list"
        )
    objects: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_objects:
        if not isinstance(item, dict):
            raise ValueError(
                "functional evidence planner objects must be JSON objects"
            )
        if set(item) - {"id", "category"}:
            raise ValueError(
                "functional evidence planner object list permits only id and "
                "category; transforms and grouping prose are forbidden"
            )
        object_id = str(item.get("id") or "").strip()
        category = str(item.get("category") or "").strip()
        if not object_id or not category or object_id in seen:
            raise ValueError(
                "functional evidence planner objects require unique non-empty "
                "id and category"
            )
        seen.add(object_id)
        objects.append({"id": object_id, "category": category})
    max_units = value.get(
        "max_probe_units",
        FUNCTIONAL_PROBE_DEFAULT_UNITS,
    )
    if (
        isinstance(max_units, bool)
        or not isinstance(max_units, int)
        or not 1 <= max_units <= FUNCTIONAL_PROBE_MAX_UNITS
    ):
        raise ValueError(
            "max_probe_units must be an integer between 1 and "
            f"{FUNCTIONAL_PROBE_MAX_UNITS}"
        )
    return {
        "metric": "functional_consistency",
        "scene_id": (
            str(value["scene_id"]) if value.get("scene_id") is not None else None
        ),
        "scene_type": (
            str(value["scene_type"])
            if value.get("scene_type") is not None
            else None
        ),
        "global_image_path": str(global_path),
        "architecture_context": _validated_architecture_context(
            value.get("architecture_context")
        ),
        "objects": objects,
        "max_probe_units": max_units,
    }


def _validated_architecture_context(value: Any) -> dict[str, Any]:
    unavailable = {
        "source": "unavailable",
        "logical_boundary_enabled": False,
        "logical_boundary_xy": [],
        "physical_walls_rendered": None,
        "physical_wall_ids": [],
    }
    if value is None:
        return unavailable
    if not isinstance(value, dict):
        raise ValueError(
            "functional evidence planner architecture_context must be an object"
        )
    allowed = {
        "source",
        "logical_boundary_enabled",
        "logical_boundary_xy",
        "physical_walls_rendered",
        "physical_wall_ids",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            "functional evidence planner architecture_context contains "
            f"unsupported fields: {sorted(unknown)}"
        )
    source = str(value.get("source") or "").strip()
    if not source:
        raise ValueError(
            "functional evidence planner architecture_context requires source"
        )
    enabled = value.get("logical_boundary_enabled")
    if not isinstance(enabled, bool):
        raise ValueError(
            "architecture_context.logical_boundary_enabled must be boolean"
        )
    raw_boundary = value.get("logical_boundary_xy")
    if not isinstance(raw_boundary, list):
        raise ValueError(
            "architecture_context.logical_boundary_xy must be a list"
        )
    boundary: list[list[float]] = []
    for point in raw_boundary:
        if (
            not isinstance(point, list)
            or len(point) < 2
            or isinstance(point[0], bool)
            or isinstance(point[1], bool)
        ):
            raise ValueError(
                "architecture_context.logical_boundary_xy requires XY points"
            )
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "architecture_context logical-boundary coordinates must be "
                "numeric"
            ) from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(
                "architecture_context logical-boundary coordinates must be "
                "finite"
            )
        boundary.append([x, y])
    if enabled and len(boundary) < 3:
        raise ValueError(
            "enabled logical boundary requires at least three XY points"
        )
    if not enabled and boundary:
        raise ValueError(
            "disabled logical boundary must not include boundary coordinates"
        )
    rendered = value.get("physical_walls_rendered")
    if rendered is not None and not isinstance(rendered, bool):
        raise ValueError(
            "architecture_context.physical_walls_rendered must be boolean or "
            "null"
        )
    wall_ids = value.get("physical_wall_ids")
    if (
        not isinstance(wall_ids, list)
        or any(
            not isinstance(item, str) or not item.strip()
            for item in wall_ids
        )
        or len(wall_ids) != len(set(wall_ids))
    ):
        raise ValueError(
            "architecture_context.physical_wall_ids must contain unique "
            "non-empty strings"
        )
    if rendered is False and wall_ids:
        raise ValueError(
            "physical wall IDs require physical_walls_rendered=true"
        )
    return {
        "source": source,
        "logical_boundary_enabled": enabled,
        "logical_boundary_xy": boundary,
        "physical_walls_rendered": rendered,
        "physical_wall_ids": list(wall_ids),
    }


def validate_functional_probe_plan_response(
    value: Any,
    *,
    known_object_ids: set[str],
    max_probe_units: int,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(
            "functional evidence planner response must be a JSON object"
        )
    _reject_forbidden_fields(value)
    unknown_top = set(value) - {"probe_units", "reason"}
    if unknown_top:
        raise ValueError(
            "functional evidence planner returned unsupported fields: "
            f"{sorted(unknown_top)}"
        )
    units = value.get("probe_units")
    if not isinstance(units, list):
        raise ValueError(
            "functional evidence planner probe_units must be a JSON list"
        )
    if len(units) > max_probe_units:
        raise ValueError(
            "functional evidence planner exceeded max_probe_units "
            f"({len(units)} > {max_probe_units})"
        )
    normalized_units: list[dict[str, Any]] = []
    fingerprints: set[tuple[Any, ...]] = set()
    for index, unit in enumerate(units):
        if not isinstance(unit, dict):
            raise ValueError(
                "functional evidence planner probe units must be JSON objects"
            )
        unknown = set(unit) - _UNIT_FIELDS
        if unknown:
            raise ValueError(
                "functional evidence probe returned unsupported fields: "
                f"{sorted(unknown)}"
            )
        kind = str(unit.get("kind") or "").strip()
        if kind not in FUNCTIONAL_PROBE_KINDS:
            raise ValueError(
                "functional evidence probe kind must be one of "
                f"{list(FUNCTIONAL_PROBE_KINDS)}"
            )
        targets = _validated_ids(
            unit.get("target_ids"),
            label="target_ids",
            known_object_ids=known_object_ids,
            required=True,
        )
        related = _validated_ids(
            unit.get("related_target_ids", []),
            label="related_target_ids",
            known_object_ids=known_object_ids,
            required=False,
        )
        if set(targets) & set(related):
            raise ValueError(
                "functional evidence probe target_ids and related_target_ids "
                "must be disjoint"
            )
        if kind == "functional_correspondence" and len(
            {*targets, *related}
        ) < 2:
            raise ValueError(
                "functional_correspondence requires at least two objects"
            )
        observations = _validated_observations(
            unit.get("required_observations"),
            kind=kind,
        )
        priority = unit.get("priority")
        if (
            isinstance(priority, bool)
            or not isinstance(priority, int)
            or not 1 <= priority <= max_probe_units
        ):
            raise ValueError(
                "functional evidence probe priority must be an integer "
                f"between 1 and {max_probe_units}"
            )
        reason = str(unit.get("reason") or "").strip()
        if not reason:
            raise ValueError(
                "functional evidence probe requires a non-empty reason"
            )
        fingerprint = (
            kind,
            tuple(sorted(targets)),
            tuple(sorted(related)),
        )
        if fingerprint in fingerprints:
            raise ValueError(
                "functional evidence planner returned a duplicate probe unit"
            )
        fingerprints.add(fingerprint)
        normalized_units.append(
            {
                "_source_index": index,
                "kind": kind,
                "target_ids": targets,
                "related_target_ids": related,
                "required_observations": observations,
                "priority": priority,
                "reason": reason[:1000],
            }
        )
    normalized_units.sort(
        key=lambda item: (item["priority"], item["_source_index"])
    )
    for index, unit in enumerate(normalized_units, start=1):
        unit.pop("_source_index", None)
        unit["probe_id"] = f"functional_probe_{index:02d}"
    return {
        "probe_units": normalized_units,
        "reason": str(value.get("reason") or "")[:1000],
    }


def functional_probe_selector_context(value: Any) -> dict[str, Any]:
    """Return a bounded selector-only view of one trusted probe unit."""

    source = value if isinstance(value, dict) else {}
    kind = str(source.get("kind") or "").strip()
    if kind not in FUNCTIONAL_PROBE_KINDS:
        return {}
    allowed = _OBSERVATIONS_BY_KIND[kind]
    observations = [
        str(item)
        for item in source.get("required_observations") or []
        if str(item) in allowed
    ]
    observation_goals = list(
        dict.fromkeys(
            str(item).strip()[:1000]
            for item in (
                source.get("observation_goals")
                or [source.get("view_goal")]
            )
            if str(item or "").strip()
        )
    )[:4]
    acquisition_triggers = list(
        dict.fromkeys(
            str(item).strip()[:120]
            for item in source.get("acquisition_triggers") or []
            if str(item or "").strip()
        )
    )[:4]
    categories = source.get("target_categories")
    category_map = (
        {
            str(key): str(item)
            for key, item in categories.items()
            if str(key).strip() and str(item).strip()
        }
        if isinstance(categories, dict)
        else {}
    )
    return {
        "probe_id": str(source.get("probe_id") or "")[:120],
        "kind": kind,
        "target_ids": [
            str(item)
            for item in source.get("target_ids") or []
            if str(item).strip()
        ][:12],
        "related_target_ids": [
            str(item)
            for item in source.get("related_target_ids") or []
            if str(item).strip()
        ][:12],
        "target_categories": category_map,
        "required_observations": list(dict.fromkeys(observations)),
        "view_goal": str(source.get("view_goal") or "")[:1000],
        "observation_goals": observation_goals,
        "acquisition_triggers": acquisition_triggers,
        "route_scope": str(source.get("route_scope") or "")[:80],
        "owning_group_id": (
            str(source.get("owning_group_id"))
            if source.get("owning_group_id")
            else None
        ),
        "group_member_ids": [
            str(item)
            for item in source.get("group_member_ids") or []
            if str(item).strip()
        ][:24],
        "camera_scope_composition": str(
            source.get("camera_scope_composition") or ""
        )[:120],
        "usable_surface_hypotheses": [
            {
                "target_id": str(item.get("target_id") or ""),
                "status": str(item.get("status") or ""),
                "bank_complete": bool(item.get("bank_complete", False)),
                "available_side_ids": [
                    str(side_id)
                    for side_id in item.get("available_side_ids") or []
                ],
                "surfaces": [
                    {
                        "surface_role": str(
                            surface.get("surface_role") or ""
                        ),
                        "side_id": str(surface.get("side_id") or ""),
                        "confidence": surface.get("confidence"),
                    }
                    for surface in item.get("surfaces") or []
                    if isinstance(surface, dict)
                ],
            }
            for item in source.get("usable_surface_hypotheses") or []
            if isinstance(item, dict)
        ],
        "functional_geometry": deepcopy(
            source.get("functional_geometry")
            if isinstance(source.get("functional_geometry"), dict)
            else {}
        ),
        "decision_authority": "none",
    }


def _validated_ids(
    value: Any,
    *,
    label: str,
    known_object_ids: set[str],
    required: bool,
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(
            f"functional evidence probe {label} must be a JSON list"
        )
    result = list(
        dict.fromkeys(
            str(item)
            for item in value
            if isinstance(item, (str, int)) and str(item).strip()
        )
    )
    if required and not result:
        raise ValueError(
            f"functional evidence probe {label} cannot be empty"
        )
    unknown = sorted(set(result) - known_object_ids)
    if unknown:
        raise ValueError(
            "functional evidence probe references unknown object IDs: "
            f"{unknown}"
        )
    return result


def _validated_observations(value: Any, *, kind: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(
            "functional evidence probe required_observations must be a "
            "non-empty JSON list"
        )
    result = list(
        dict.fromkeys(
            str(item)
            for item in value
            if isinstance(item, str) and str(item).strip()
        )
    )
    unknown = sorted(set(result) - _OBSERVATIONS_BY_KIND[kind])
    if unknown:
        raise ValueError(
            f"{kind} returned unsupported observations: {unknown}"
        )
    return result


def _reject_forbidden_fields(value: Any, *, path: str = "response") -> None:
    if isinstance(value, dict):
        for raw_key, item in value.items():
            key = str(raw_key).strip().lower()
            if key in _FORBIDDEN_PLANNER_FIELDS:
                raise ValueError(
                    "functional evidence planner may not return "
                    f"{path}.{raw_key}"
                )
            _reject_forbidden_fields(item, path=f"{path}.{raw_key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_fields(item, path=f"{path}[{index}]")


def _image_data_url(path: Path, *, alias: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(
            f"functional evidence planner image does not exist: {path}"
        )
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Pillow is required to sanitize functional planner images"
        ) from exc
    try:
        with Image.open(path) as source:
            source.load()
            normalized = ImageOps.exif_transpose(source).convert("RGBA")
            flattened = Image.new(
                "RGB",
                normalized.size,
                (255, 255, 255),
            )
            flattened.paste(
                normalized,
                mask=normalized.getchannel("A"),
            )
            output = BytesIO()
            flattened.save(output, format="PNG")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(
            f"functional planner image {alias!r} is not decodable"
        ) from exc
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
