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


PLACEMENT_DISCOVERY_SCHEMA_VERSION = "placement_discovery_v1"
PLACEMENT_DISCOVERY_PROMPT_VERSION = "placement_discovery_v1"
PLACEMENT_DISCOVERY_MAX_TOKENS = 3192
PLACEMENT_OBSERVATION_KINDS = frozenset(
    {
        "support_surface",
        "placement_height",
        "scene_zone",
        "adjacency_context",
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


PLACEMENT_DISCOVERY_SYSTEM_PROMPT = """Identify only visual observations
needed to evaluate semantic location, assuming each object's identity belongs
in the scene. Name exactly one subject_id per candidate; optional context_ids
only explain the surrounding support, zone, or adjacency and are not defect
owners. Route a subject when support-surface meaning, placement height, scene
zone, or immediate adjacency/context requires closer confirmation.

Do not route orientation, facing, access, opening clearance, or operability;
those are functional. Do not route collision, penetration, floating, support
failure, or out-of-bounds geometry; those are structural. Do not route identity
membership, style, or scale.

Copy every object ID exactly once in considered_object_ids. Return exactly:
{"considered_object_ids":["id"],
"candidates":[{"subject_id":"id","context_ids":[],
"observation_kind":"scene_zone",
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
    context = {
        "role": "placement_discovery",
        "metric": "semantic_placement_consistency",
        "decision_authority": "none",
        "scene_access": "read_only",
        "prompt_version": PLACEMENT_DISCOVERY_PROMPT_VERSION,
        "schema_version": PLACEMENT_DISCOVERY_SCHEMA_VERSION,
        "scene_id": normalized.get("scene_id"),
        "scene_type": normalized.get("scene_type"),
        "object_list": deepcopy(normalized["objects"]),
        "allowed_observation_kinds": sorted(
            PLACEMENT_OBSERVATION_KINDS
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
            "prompt_version": PLACEMENT_DISCOVERY_PROMPT_VERSION,
            "schema_version": PLACEMENT_DISCOVERY_SCHEMA_VERSION,
            "backend": "openai_compatible",
            "request_metadata": {
                **dict(getattr(model, "last_request_metadata", {})),
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
    return {
        "scene_id": value.get("scene_id"),
        "scene_type": value.get("scene_type"),
        "global_image_path": str(image_path),
        "objects": normalized_objects,
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
            "observation_kind",
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
        kind = str(item.get("observation_kind") or "").strip()
        if kind not in PLACEMENT_OBSERVATION_KINDS:
            raise ValueError("placement observation_kind is unsupported")
        identity = (subject_id, tuple(sorted(context_ids)), kind)
        if identity in identities:
            raise ValueError("placement discovery contains a duplicate")
        identities.add(identity)
        goal = str(item.get("observation_goal") or "").strip()
        if not goal:
            raise ValueError(
                "placement candidate observation_goal must be non-empty"
            )
        normalized.append(
            {
                "subject_id": subject_id,
                "context_ids": context_ids,
                "observation_kind": kind,
                "observation_goal": goal[:1000],
            }
        )
    reason = str(value.get("reason") or "").strip()
    if not reason:
        raise ValueError("placement discovery reason must be non-empty")
    return {
        "considered_object_ids": considered,
        "candidates": normalized,
        "reason": reason[:1000],
    }


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
