"""Strict, non-judging discovery for semantic-placement evidence."""

from __future__ import annotations

import base64
import json
import time
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
from benchmark.visual_judge.identity_evidence import (
    validate_identity_evidence,
)


PLACEMENT_DISCOVERY_SCHEMA_VERSION = "placement_discovery_v2"
PLACEMENT_DISCOVERY_PROMPT_VERSION = "placement_discovery_v3"
PLACEMENT_DISCOVERY_MAX_TOKENS = 3192
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
default check onto every object is forbidden. considered_object_ids provides
complete inspection coverage even when candidates is empty. Name exactly one
subject_id per candidate. Optional context_ids provide non-owning visual
context and never become defect owners.

Use only:
- support_and_height: whether the subject is on a semantically appropriate
  supporting surface and at a plausible placement height;
- scene_zone: whether the subject occupies a plausible room region;
- contextual_anchor: a non-operational positional association, such as an
  object being meaningfully anchored to another scene element.

Emit a candidate only for a concrete location question visible in this scene.
Write observation_goal as a neutral question about what must be observed; do
not state or imply that the current arrangement already passes or fails.

Do not route orientation, facing, approach, opening, operation, reachability,
or action-required correspondence; those are functional. Do not route
collision, penetration, floating, physical support failure, or out-of-bounds
geometry; those are structural. Do not route identity membership, style, or
scale. Discovery proposes checks only and never decides them.

Copy every object ID exactly once in considered_object_ids. Return exactly:
{"considered_object_ids":["id"],
"candidates":[{"subject_id":"id","context_ids":[],
"check_type":"scene_zone",
"observation_goal":"neutral visual fact"}],
"reason":"brief coverage summary"}

Never return a defect, score, verdict, pose, vector, camera action, or scene
edit. Return no other fields."""


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
    started = time.perf_counter()
    raw = model.chat_messages(
        [
            {"role": "system", "content": PLACEMENT_DISCOVERY_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        response_format_json=(
            bool(getattr(model, "response_format_json", True))
            if response_format_json is None
            else bool(response_format_json)
        ),
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
    result = validate_placement_discovery_response(
        parse_json_object(raw),
        object_ids=tuple(item["id"] for item in normalized["objects"]),
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
    return {
        "scene_id": value.get("scene_id"),
        "scene_type": value.get("scene_type"),
        "global_image_path": str(image_path),
        "objects": normalized_objects,
        **identity,
    }


def validate_placement_discovery_response(
    value: Any,
    *,
    object_ids: tuple[str, ...],
) -> dict[str, Any]:
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
    }


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
