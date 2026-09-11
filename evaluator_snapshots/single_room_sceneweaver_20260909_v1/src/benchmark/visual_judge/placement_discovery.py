"""Strict, non-judging discovery for semantic-placement evidence."""

from __future__ import annotations

import base64
import json
import time
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any

from benchmark.functional_spatial_context import (
    validate_functional_spatial_context,
)
from benchmark.models import OpenAICompatibleModel, parse_json_object
from benchmark.visual_judge.contracts import ResponseSchemaRepairError
from benchmark.visual_judge.roles import (
    DecisionContract,
    VLMRole,
    vlm_audit_metadata,
)
from benchmark.visual_judge.identity_evidence import (
    validate_identity_evidence,
)


PLACEMENT_DISCOVERY_SCHEMA_VERSION = "placement_discovery_v2"
PLACEMENT_DISCOVERY_PROMPT_VERSION = "placement_discovery_v6"
PLACEMENT_DISCOVERY_MAX_TOKENS = 3192
PLACEMENT_DISCOVERY_REPAIR_POLICY = (
    "single_placement_discovery_schema_retry_v1"
)
PLACEMENT_DISCOVERY_CHECK_TYPES = frozenset(
    {
        "support_and_height",
        "scene_zone",
        "contextual_anchor",
    }
)
_FORBIDDEN_FIELDS = frozenset(
    {
        "verdict",
        "validity",
        "status",
        "score",
        "confidence",
        "defect",
        "defects",
        "pose",
        "camera_pose",
        "vector",
        "direction",
        "normal",
        "scene_mutation",
        "mutation",
    }
)


PLACEMENT_DISCOVERY_SYSTEM_PROMPT = """Identify sparse typed visual checks
needed to evaluate semantic location, assuming each object's identity belongs
in the scene. This is routing, not exhaustive object classification: copying a
default check onto every object is forbidden. considered_object_ids records
which objects Discovery inspected; it does not prove omitted objects normal
and cannot suppress downstream review. Name exactly one
subject_id per candidate. context_ids provide non-owning visual context and
never become defect owners; contextual_anchor requires at least one context
object, while the other check types may use an empty list.

Use only:
- support_and_height: whether the subject is on a semantically appropriate
  supporting surface and at a plausible placement height;
- scene_zone: whether the subject occupies a plausible room region;
- contextual_anchor: a non-operational positional association, such as an
  object being meaningfully anchored to another scene element.

Emit only concrete location questions visible in this scene. Write a neutral
observation_goal; do not imply the arrangement passes or fails.

functional_spatial_context is Function-prerequisite guidance.
Clearance measurements are spatial facts, not pass/fail thresholds.
background_only relations are Function-owned; candidate_attention may suggest
an independently visible Placement question. A movable companion is not a
permanent blocker merely because it intersects a corridor. This context is
neither a verdict nor a required check.

Do not route orientation, facing, approach, opening, operation, reachability,
or action-required correspondence; those are Functional. Collision,
penetration, floating, physical support failure, and out-of-bounds geometry
are structural. Identity, style, and scale are also out of scope. Discovery
proposes checks only.

Copy every object ID exactly once in considered_object_ids. Return exactly:
{"considered_object_ids":["id"],
"candidates":[{"subject_id":"id","context_ids":[],
"check_type":"scene_zone",
"observation_goal":"neutral visual fact"}],
"reason":"brief coverage summary"}

Never return a defect, score, verdict, pose, vector, camera action, or scene
edit. Return no other fields."""

_PLACEMENT_DISCOVERY_REPAIR_PROMPT = """The previous placement-discovery
response violated the supplied JSON contract. Correct only the schema using
the same images and object IDs. Use check_type (never placement_check_type),
copy considered_object_ids exactly once in the supplied order, retain only
the three allowed check types, require at least one context_id for every
contextual_anchor, and return JSON only. Do not add a verdict, defect, score,
camera pose, or scene edit."""


def discover_openai_compatible_placement_evidence(
    *,
    model: OpenAICompatibleModel,
    request: dict[str, Any],
    max_context_chars: int = 30000,
    response_format_json: bool | None = None,
) -> dict[str, Any]:
    normalized = validate_placement_discovery_request(request)
    audit = vlm_audit_metadata(
        VLMRole.PLACEMENT_DISCOVERY,
        decision_contract=DecisionContract.PLACEMENT_DISCOVERY,
        judge_method="discover_placement_evidence",
    )
    context = {
        **audit,
        "role": "placement_discovery",
        "metric": "semantic_placement_consistency",
        "decision_authority": "none",
        "scene_access": "read_only",
        "prompt_version": PLACEMENT_DISCOVERY_PROMPT_VERSION,
        "schema_version": PLACEMENT_DISCOVERY_SCHEMA_VERSION,
        "scene_id": normalized.get("scene_id"),
        "scene_type": normalized.get("scene_type"),
        "object_list": deepcopy(normalized["objects"]),
        "identity_grounding": {
            "status": normalized["identity_grounding"],
            "image_role": "global_identity_overlay",
            "legend": deepcopy(normalized["identity_legend"]),
        },
        "allowed_check_types": sorted(
            PLACEMENT_DISCOVERY_CHECK_TYPES
        ),
    }
    if normalized["functional_spatial_context"] is not None:
        context["functional_spatial_context"] = deepcopy(
            normalized["functional_spatial_context"]
        )
    context_text = json.dumps(
        context,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(context_text) > max(1000, int(max_context_chars)):
        raise ValueError(
            "placement discovery context exceeds max_context_chars; "
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
                    Path(normalized["global_image_path"])
                )
            },
        },
    ]
    if normalized["identity_image_path"] is not None:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _image_data_url(
                        Path(normalized["identity_image_path"])
                    )
                },
            }
        )
    messages = [
        {"role": "system", "content": PLACEMENT_DISCOVERY_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    use_json_response = (
        bool(getattr(model, "response_format_json", True))
        if response_format_json is None
        else bool(response_format_json)
    )
    started = time.perf_counter()
    raw = model.chat_messages(
        messages,
        response_format_json=use_json_response,
        call_type="vlm_camera_pose.placement_discovery",
        max_tokens=max(
            PLACEMENT_DISCOVERY_MAX_TOKENS,
            int(getattr(model, "max_tokens", None) or 0),
        ),
        max_tokens_source="placement_discovery_minimum",
        case={
            "case_id": str(
                normalized.get("scene_id") or "placement_discovery"
            ),
            "scene_id": str(normalized.get("scene_id") or ""),
            "objects": deepcopy(normalized["objects"]),
        },
    )
    object_ids = tuple(item["id"] for item in normalized["objects"])
    result, schema_audit = _validate_placement_discovery_with_retry(
        model=model,
        messages=messages,
        raw=raw,
        object_ids=object_ids,
        response_format_json=use_json_response,
        case={
            "case_id": str(
                normalized.get("scene_id") or "placement_discovery"
            ),
            "scene_id": str(normalized.get("scene_id") or ""),
            "objects": deepcopy(normalized["objects"]),
        },
    )
    return {
        **result,
        "schema_version": PLACEMENT_DISCOVERY_SCHEMA_VERSION,
        "decision_authority": "none",
        "provenance": {
            **audit,
            "prompt_version": PLACEMENT_DISCOVERY_PROMPT_VERSION,
            "schema_version": PLACEMENT_DISCOVERY_SCHEMA_VERSION,
            "backend": "openai_compatible",
            "request_metadata": {
                **dict(getattr(model, "last_request_metadata", {})),
                **audit,
                "latency_seconds": round(
                    time.perf_counter() - started,
                    6,
                ),
                "status": "complete",
            },
            "schema_validation": schema_audit,
        },
    }


def validate_placement_discovery_request(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("placement discovery request must be an object")
    if str(value.get("metric") or "") != (
        "semantic_placement_consistency"
    ):
        raise ValueError(
            "placement discovery only supports "
            "semantic_placement_consistency"
        )
    image_path = Path(
        str(value.get("global_image_path") or "")
    ).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(
            f"placement discovery image does not exist: {image_path}"
        )
    objects = value.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("placement discovery requires a non-empty object list")
    normalized_objects: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in objects:
        if not isinstance(item, dict) or set(item) - {"id", "category"}:
            raise ValueError(
                "placement discovery objects permit only id and category"
            )
        object_id = str(item.get("id") or "").strip()
        category = str(item.get("category") or "").strip()
        if not object_id or not category or object_id in seen:
            raise ValueError(
                "placement discovery requires unique object IDs and categories"
            )
        seen.add(object_id)
        normalized_objects.append({"id": object_id, "category": category})
    identity = validate_identity_evidence(
        image_path=value.get("identity_image_path"),
        legend=value.get("identity_legend"),
        expected_object_ids=(
            item["id"] for item in normalized_objects
        ),
        label="placement discovery",
    )
    functional_spatial_context = value.get("functional_spatial_context")
    if functional_spatial_context is not None:
        functional_spatial_context = validate_functional_spatial_context(
            functional_spatial_context,
            known_object_ids=(
                item["id"] for item in normalized_objects
            ),
        )
    return {
        "scene_id": value.get("scene_id"),
        "scene_type": value.get("scene_type"),
        "global_image_path": str(image_path),
        "objects": normalized_objects,
        "functional_spatial_context": functional_spatial_context,
        **identity,
    }


def validate_placement_discovery_response(
    value: Any,
    *,
    object_ids: tuple[str, ...],
) -> dict[str, Any]:
    value, normalization_warnings = (
        _canonicalize_placement_check_type_alias(value)
    )
    if not isinstance(value, dict):
        raise ValueError("placement discovery response must be an object")
    _reject_forbidden_fields(value)
    extra = set(value) - {"considered_object_ids", "candidates", "reason"}
    if extra:
        raise ValueError(
            "placement discovery returned unsupported fields: "
            f"{sorted(extra)}"
        )
    considered = _id_list(
        value.get("considered_object_ids"),
        known=set(object_ids),
        minimum=1,
        label="considered_object_ids",
    )
    if tuple(considered) != object_ids:
        raise ValueError(
            "placement discovery considered_object_ids must contain every "
            "input object exactly once in supplied order"
        )
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or not all(
        isinstance(item, dict) for item in candidates
    ):
        raise ValueError("placement candidates must be a list of objects")
    normalized: list[dict[str, Any]] = []
    identities: set[tuple[str, tuple[str, ...], str]] = set()
    for item in candidates:
        unknown = set(item) - {
            "subject_id",
            "context_ids",
            "check_type",
            "observation_goal",
        }
        if unknown:
            raise ValueError(
                "placement candidate returned unsupported fields: "
                f"{sorted(unknown)}"
            )
        subject_id = str(item.get("subject_id") or "").strip()
        if subject_id not in set(object_ids):
            raise ValueError(
                "placement candidate references an unknown subject ID"
            )
        context_ids = _id_list(
            item.get("context_ids"),
            known=set(object_ids),
            minimum=0,
            label="context_ids",
        )
        if subject_id in context_ids:
            raise ValueError(
                "placement subject cannot appear in its own context"
            )
        kind = str(item.get("check_type") or "").strip()
        if kind not in PLACEMENT_DISCOVERY_CHECK_TYPES:
            raise ValueError("placement check_type is unsupported")
        if kind == "contextual_anchor" and not context_ids:
            raise ValueError(
                "contextual_anchor requires one or more context IDs"
            )
        identity = (subject_id, tuple(sorted(context_ids)), kind)
        if identity in identities:
            raise ValueError("placement discovery contains a duplicate")
        identities.add(identity)
        source_goal = str(item.get("observation_goal") or "").strip()
        if not source_goal:
            raise ValueError(
                "placement candidate observation_goal must be non-empty"
            )
        goal = _neutral_placement_observation_goal(kind)
        normalized.append(
            {
                "subject_id": subject_id,
                "context_ids": context_ids,
                "check_type": kind,
                "observation_goal": goal,
                "source_observation_goal": source_goal[:1000],
            }
        )
    reason = str(value.get("reason") or "").strip()
    if not reason:
        raise ValueError("placement discovery reason must be non-empty")
    return {
        "considered_object_ids": considered,
        "candidates": normalized,
        "reason": reason[:1000],
        "observation_goal_policy": (
            "deterministic_neutral_routing_question_v1"
        ),
        "normalization_warnings": normalization_warnings,
        "coverage": {
            "unit": "object_consideration_and_candidate_atom",
            "eligible_count": len(object_ids) + len(normalized),
            "grounded_count": len(object_ids) + len(normalized),
            "fraction": 1.0,
            "complete": True,
        },
    }


def _validate_placement_discovery_with_retry(
    *,
    model: OpenAICompatibleModel,
    messages: list[dict[str, Any]],
    raw: str,
    object_ids: tuple[str, ...],
    response_format_json: bool,
    case: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    initial_value: Any = None
    try:
        initial_value = parse_json_object(raw)
        result = validate_placement_discovery_response(
            initial_value,
            object_ids=object_ids,
        )
    except (TypeError, ValueError, KeyError) as first_error:
        first_attempt = {
            "attempt": 1,
            "validation_error_type": type(first_error).__name__,
            "validation_error": str(first_error),
        }
    else:
        return result, {
            "policy": PLACEMENT_DISCOVERY_REPAIR_POLICY,
            "attempt_count": 1,
            "repair_retry_count": 0,
            "recovered": False,
            "item_level_salvage": False,
        }

    repair_call_type = "vlm_camera_pose.placement_discovery.schema_repair"
    repair_messages = [
        *deepcopy(messages),
        {"role": "assistant", "content": raw},
        {
            "role": "user",
            "content": (
                _PLACEMENT_DISCOVERY_REPAIR_PROMPT
                + "\n\nValidator error: "
                + first_attempt["validation_error"]
            ),
        },
    ]
    try:
        repaired_raw = model.chat_messages(
            repair_messages,
            response_format_json=response_format_json,
            call_type=repair_call_type,
            max_tokens=max(
                PLACEMENT_DISCOVERY_MAX_TOKENS,
                int(getattr(model, "max_tokens", None) or 0),
            ),
            max_tokens_source="placement_discovery_schema_repair_minimum",
            case=case,
        )
    except Exception as exc:
        audit = {
            "policy": PLACEMENT_DISCOVERY_REPAIR_POLICY,
            "attempt_count": 2,
            "repair_retry_count": 1,
            "recovered": False,
            "attempts": [
                first_attempt,
                {
                    "attempt": 2,
                    "failure_kind": "transport",
                    "validation_error_type": type(exc).__name__,
                    "validation_error": str(exc),
                },
            ],
        }
        raise ResponseSchemaRepairError(
            f"placement discovery repair request failed: {exc}",
            schema_audit=audit,
        ) from exc

    repaired_value: Any = None
    try:
        repaired_value = parse_json_object(repaired_raw)
        repaired = validate_placement_discovery_response(
            repaired_value,
            object_ids=object_ids,
        )
    except (TypeError, ValueError, KeyError) as second_error:
        salvaged = _salvage_placement_discovery_response(
            repaired_value if isinstance(repaired_value, dict) else {},
            object_ids=object_ids,
            fallback_value=(
                initial_value if isinstance(initial_value, dict) else {}
            ),
        )
        return salvaged, {
            "policy": PLACEMENT_DISCOVERY_REPAIR_POLICY,
            "attempt_count": 2,
            "repair_retry_count": 1,
            "recovered": False,
            "item_level_salvage": True,
            "attempts": [
                first_attempt,
                {
                    "attempt": 2,
                    "validation_error_type": type(second_error).__name__,
                    "validation_error": str(second_error),
                },
            ],
            "salvage": deepcopy(salvaged.get("item_salvage") or {}),
        }
    merged = _salvage_placement_discovery_response(
        repaired_value if isinstance(repaired_value, dict) else repaired,
        object_ids=object_ids,
        fallback_value=(
            initial_value if isinstance(initial_value, dict) else {}
        ),
    )
    return merged, {
        "policy": PLACEMENT_DISCOVERY_REPAIR_POLICY,
        "attempt_count": 2,
        "repair_retry_count": 1,
        "recovered": True,
        "item_level_salvage": True,
        "fallback_mode": "initial_valid_atoms_plus_anchored_repair",
        "salvage": deepcopy(merged.get("item_salvage") or {}),
        "attempts": [first_attempt, {"attempt": 2, "validation_error": None}],
    }


def _canonicalize_placement_check_type_alias(
    value: Any,
) -> tuple[Any, list[dict[str, Any]]]:
    if not isinstance(value, dict):
        return value, []
    normalized = deepcopy(value)
    warnings: list[dict[str, Any]] = []
    candidates = normalized.get("candidates")
    if not isinstance(candidates, list):
        return normalized, warnings
    for index, item in enumerate(candidates):
        if not isinstance(item, dict) or "placement_check_type" not in item:
            continue
        alias = item.get("placement_check_type")
        canonical = item.get("check_type")
        if canonical is not None and canonical != alias:
            continue
        item["check_type"] = alias
        item.pop("placement_check_type", None)
        warnings.append(
            {
                "code": "placement_check_type_alias_canonicalized",
                "candidate_index": index,
                "from": "placement_check_type",
                "to": "check_type",
            }
        )
    return normalized, warnings


def _salvage_placement_discovery_response(
    value: dict[str, Any],
    *,
    object_ids: tuple[str, ...],
    fallback_value: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sources: list[tuple[str, dict[str, Any]]] = []
    alias_warnings: list[dict[str, Any]] = []
    for source_name, source_value in (
        ("initial", fallback_value or {}),
        ("repair", value),
    ):
        normalized_source, warnings = (
            _canonicalize_placement_check_type_alias(source_value)
        )
        sources.append((source_name, normalized_source))
        alias_warnings.extend(
            {**warning, "source": source_name} for warning in warnings
        )
    initial_source = sources[0][1]
    initial_candidates = initial_source.get("candidates")
    initial_candidates = (
        initial_candidates if isinstance(initial_candidates, list) else []
    )
    initial_anchors: list[tuple[str, tuple[str, ...], str]] = []
    rejected: list[dict[str, Any]] = []
    for index, item in enumerate(initial_candidates):
        try:
            anchor = _placement_candidate_identity_anchor(
                item,
                object_ids=object_ids,
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

    accepted: list[dict[str, Any]] = []
    identities: set[tuple[str, tuple[str, ...], str]] = set()
    grounded_ids: list[str] = []
    for source_name, normalized_source in sources:
        raw_considered = normalized_source.get("considered_object_ids")
        if isinstance(raw_considered, list):
            for item in raw_considered:
                object_id = str(item or "").strip()
                if (
                    object_id in object_ids
                    and object_id not in grounded_ids
                ):
                    grounded_ids.append(object_id)
        raw_candidates = normalized_source.get("candidates")
        raw_candidates = (
            raw_candidates if isinstance(raw_candidates, list) else []
        )
        for index, item in enumerate(raw_candidates):
            try:
                raw_identity = _placement_candidate_identity_anchor(
                    item,
                    object_ids=object_ids,
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
                        "reason": "candidate_has_no_initial_identity_anchor",
                    }
                )
                continue
            try:
                validated = validate_placement_discovery_response(
                    {
                        "considered_object_ids": list(object_ids),
                        "candidates": [item],
                        "reason": "item-level salvage validation",
                    },
                    object_ids=object_ids,
                )
                candidate = validated["candidates"][0]
                identity = (
                    str(candidate["subject_id"]),
                    tuple(sorted(candidate["context_ids"])),
                    str(candidate["check_type"]),
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
            accepted.append(candidate)
    dropped_anchors = [
        {
            "subject_id": identity[0],
            "context_ids": list(identity[1]),
            "check_type": identity[2],
        }
        for identity in initial_anchors
        if identity not in identities
    ]
    eligible_count = len(object_ids) + len(initial_anchors)
    grounded_count = len(grounded_ids) + len(identities)
    coverage = {
        "unit": "object_consideration_and_candidate_atom",
        "eligible_count": eligible_count,
        "grounded_count": grounded_count,
        "fraction": (
            grounded_count / eligible_count if eligible_count else 0.0
        ),
        "complete": (
            set(grounded_ids) == set(object_ids)
            and len(identities) == len(initial_anchors)
        ),
    }
    return {
        "considered_object_ids": list(object_ids),
        "candidates": accepted,
        "reason": (
            "Valid placement candidates were retained; malformed candidates "
            "were omitted without changing other objects or checks."
        ),
        "observation_goal_policy": (
            "deterministic_neutral_routing_question_v1"
        ),
        "normalization_warnings": alias_warnings,
        "coverage": coverage,
        "item_salvage": {
            "policy": "initial_anchored_placement_atoms_v2",
            "anchored_candidate_count": len(initial_anchors),
            "accepted_candidate_count": len(accepted),
            "dropped_candidate_count": len(dropped_anchors),
            "dropped_candidate_anchors": dropped_anchors,
            "rejected_candidate_count": len(rejected),
            "rejected_items": rejected,
        },
    }


def _placement_candidate_identity_anchor(
    value: Any,
    *,
    object_ids: tuple[str, ...],
) -> tuple[str, tuple[str, ...], str]:
    """Read candidate identity without accepting its observation content."""

    if not isinstance(value, dict):
        raise ValueError("placement candidate is not an object")
    known = set(object_ids)
    subject_id = str(value.get("subject_id") or "").strip()
    if subject_id not in known:
        raise ValueError("placement candidate subject is unknown")
    context_ids = _id_list(
        value.get("context_ids"),
        known=known,
        minimum=0,
        label="context_ids",
    )
    if subject_id in context_ids:
        raise ValueError("placement candidate subject appears in context")
    check_type = str(value.get("check_type") or "").strip()
    if check_type not in PLACEMENT_DISCOVERY_CHECK_TYPES:
        raise ValueError("placement candidate check_type is unsupported")
    if check_type == "contextual_anchor" and not context_ids:
        raise ValueError(
            "contextual_anchor requires one or more context IDs"
        )
    return subject_id, tuple(sorted(context_ids)), check_type


def _neutral_placement_observation_goal(kind: str) -> str:
    return {
        "support_and_height": (
            "Observe the subject's supporting surface and placement height "
            "without assuming semantic plausibility."
        ),
        "scene_zone": (
            "Observe the subject's room zone and architectural context "
            "without assuming semantic plausibility."
        ),
        "contextual_anchor": (
            "Observe the subject's non-operational positional relationship "
            "to the listed context objects without assuming plausibility."
        ),
    }[kind]


def placement_groups_to_confirm(
    result: dict[str, Any],
    *,
    groups: list[dict[str, Any]],
) -> set[str]:
    object_to_group = {
        str(object_id): str(group.get("group_id"))
        for group in groups
        if isinstance(group, dict) and group.get("group_id")
        for object_id in group.get("object_ids") or []
    }
    return {
        object_to_group[str(item.get("subject_id"))]
        for item in result.get("candidates") or []
        if isinstance(item, dict)
        and str(item.get("subject_id")) in object_to_group
    }


def _id_list(
    value: Any,
    *,
    known: set[str],
    minimum: int,
    label: str,
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON list")
    result = [str(item).strip() for item in value]
    if (
        len(result) < minimum
        or any(not item for item in result)
        or len(result) != len(set(result))
    ):
        raise ValueError(f"{label} contains invalid or duplicate IDs")
    unknown = sorted(set(result) - known)
    if unknown:
        raise ValueError(f"{label} references unknown object IDs: {unknown}")
    return result


def _reject_forbidden_fields(value: Any, *, path: str = "response") -> None:
    if isinstance(value, dict):
        for raw_key, item in value.items():
            if str(raw_key).strip().lower() in _FORBIDDEN_FIELDS:
                raise ValueError(
                    f"placement discovery may not return {path}.{raw_key}"
                )
            _reject_forbidden_fields(item, path=f"{path}.{raw_key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_fields(item, path=f"{path}[{index}]")


def _image_data_url(path: Path) -> str:
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for placement discovery") from exc
    try:
        with Image.open(path) as source:
            source.load()
            normalized = ImageOps.exif_transpose(source).convert("RGBA")
            flattened = Image.new("RGB", normalized.size, (255, 255, 255))
            flattened.paste(normalized, mask=normalized.getchannel("A"))
            output = BytesIO()
            flattened.save(output, format="PNG")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("placement discovery image is not decodable") from exc
    return (
        "data:image/png;base64,"
        + base64.b64encode(output.getvalue()).decode("ascii")
    )
