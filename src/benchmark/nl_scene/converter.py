from __future__ import annotations

import json
import os
import re
from typing import Any

from benchmark.models.openai_compatible_model import OpenAICompatibleModel


CONVERTER_SYSTEM_PROMPT = """Convert a natural-language scene/layout request into a canonical object_plan JSON object.
Use a SceneEval-style decomposition conceptually: explicit object requirements, object attributes, soft object-object relation intents, soft object-architecture relation intents, and global constraints.
Return object_plan JSON only. Do not output coordinates, positions, center, rotation, pose, target_pose, exact asset ids, jids, asset_ref, or ground-truth pose.
Relations are soft intent only; use finite relation names when possible, but relationship_mapper will finalize evaluator-ready OOR/OAR specs later.
Return exactly one JSON object and no Markdown.
""".strip()

FORBIDDEN_OBJECT_KEYS = {
    "center",
    "position",
    "rotation",
    "target_pose",
    "pose",
    "jid",
    "asset_jid",
    "asset_id",
    "asset_ref",
    "expected_relations",
}


class ObjectPlanConversionError(RuntimeError):
    """Raised when an NL instruction cannot be converted into a valid object_plan."""


SceneSpecConversionError = ObjectPlanConversionError


def convert_nl_to_object_plan(
    instruction: str,
    *,
    request_id: str = "request_001",
    scene_type: str | None = None,
    room: dict | None = None,
    model_config: dict | None = None,
) -> dict:
    """Convert a natural-language instruction into canonical object_plan."""

    clean_instruction = str(instruction or "").strip()
    if not clean_instruction:
        raise ValueError("instruction must be a non-empty string")
    messages = _converter_messages(clean_instruction, request_id=request_id, scene_type=scene_type, room=room)
    first_response = call_chat_model(messages, model_config=model_config, response_format_json=True, call_type="object_plan_converter")
    try:
        parsed = parse_json_object_from_text(first_response)
    except ValueError as first_error:
        correction_messages = [
            *messages,
            {"role": "assistant", "content": first_response},
            {
                "role": "user",
                "content": (
                    "The previous response was not valid JSON. Return one corrected object_plan JSON object only. "
                    "Still omit coordinates, positions, rotations, exact asset ids, jids, asset_ref, and pose fields."
                ),
            },
        ]
        second_response = call_chat_model(correction_messages, model_config=model_config, response_format_json=True, call_type="object_plan_converter_retry")
        try:
            parsed = parse_json_object_from_text(second_response)
        except ValueError as second_error:
            raise ObjectPlanConversionError(f"Model did not return valid object_plan JSON: {second_error}") from first_error
    return validate_object_plan_json(parsed, request_id=request_id, instruction=clean_instruction, scene_type=scene_type)


def convert_nl_to_scene_spec(*args: Any, **kwargs: Any) -> dict:
    """Temporary alias that returns canonical object_plan, not legacy scene_spec."""

    return convert_nl_to_object_plan(*args, **kwargs)


def parse_json_object_from_text(text: str) -> dict:
    """Parse a JSON object from plain text or a fenced Markdown JSON block."""

    stripped = _strip_markdown_fence(str(text or "").strip())
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("response does not contain a JSON object")
        loaded = json.loads(stripped[start : end + 1])
    if not isinstance(loaded, dict):
        raise ValueError("response JSON must be an object")
    return loaded


def validate_object_plan_json(plan: dict, *, request_id: str = "request_001", instruction: str = "", scene_type: str | None = None) -> dict:
    if not isinstance(plan, dict):
        raise ObjectPlanConversionError("object_plan must be a JSON object")
    normalized: dict[str, Any] = {
        "request_id": _string(plan.get("request_id") or request_id),
        "scene_type": _string(plan.get("scene_type") or scene_type or "room"),
        "scene_description": _string(plan.get("scene_description") or instruction),
        "objects": [],
        "global_constraints": _string_list(plan.get("global_constraints")),
        "relations": [],
    }
    objects = plan.get("objects")
    if not isinstance(objects, list):
        raise ObjectPlanConversionError("object_plan must contain an objects list")
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            raise ObjectPlanConversionError(f"objects[{index}] must be a JSON object")
        forbidden = sorted(key for key in FORBIDDEN_OBJECT_KEYS if key in item)
        if forbidden:
            raise ObjectPlanConversionError(f"objects[{index}] contains forbidden pose/asset keys: {forbidden}")
        placement_intent = item.get("placement_intent") if isinstance(item.get("placement_intent"), dict) else {}
        obj: dict[str, Any] = {
            "id": _string(item.get("id") or f"obj_{index:03d}"),
            "role": _string(item.get("role")),
            "category": _string(item.get("category") or "object"),
            "description": _string(item.get("description") or item.get("short_desc") or item.get("category") or "object"),
            "count": _positive_int(item.get("count"), default=1),
            "placement_intent": {
                "absolute_relations": _relation_list(placement_intent.get("absolute_relations")),
                "relative_relations": _relation_list(placement_intent.get("relative_relations")),
            },
            "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
        }
        estimated_size = _size_or_none(item.get("estimated_size"))
        if estimated_size is not None:
            obj["estimated_size"] = estimated_size
        normalized["objects"].append(obj)
    relations = plan.get("relations")
    if isinstance(relations, list):
        normalized["relations"] = [item for item in relations if isinstance(item, dict)]
    return normalized


def validate_scene_spec(spec: dict, *, instruction: str = "", scene_type: str | None = None) -> dict:
    """Temporary validation alias returning canonical object_plan."""

    return validate_object_plan_json(spec, instruction=instruction, scene_type=scene_type)


def call_chat_model(
    messages: list[dict[str, Any]],
    *,
    model_config: dict | None = None,
    response_format_json: bool = True,
    call_type: str = "chat",
) -> str:
    """Call a configured chat model, with simple mock hooks for tests."""

    config = model_config or {}
    if "mock_response" in config:
        return str(config["mock_response"])
    responses = config.get("mock_responses")
    if isinstance(responses, list) and responses:
        return str(responses.pop(0))
    chat_model = config.get("chat_model") or config.get("client")
    if chat_model is not None:
        if hasattr(chat_model, "chat_messages"):
            return str(chat_model.chat_messages(messages, response_format_json=response_format_json, call_type=call_type))
        if callable(chat_model):
            return str(chat_model(messages))
    endpoint = config.get("base_url") or config.get("endpoint") or os.environ.get("OPENAI_BASE_URL")
    model_id = config.get("model") or config.get("model_id") or os.environ.get("MODEL_NAME") or os.environ.get("OPENAI_MODEL")
    api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
    if not endpoint or not model_id:
        raise ObjectPlanConversionError(
            "No chat model configured. Provide model_config with base_url/model or set OPENAI_BASE_URL and MODEL_NAME/OPENAI_MODEL."
        )
    model = OpenAICompatibleModel(
        name="object_plan_converter",
        endpoint=str(endpoint),
        model_id=str(model_id),
        api_key=api_key,
        temperature=float(config.get("temperature", 0.2)),
        max_tokens=int(config.get("max_tokens", 2048)),
        response_format_json=response_format_json,
    )
    return model.chat_messages(messages, response_format_json=response_format_json, call_type=call_type)


def _converter_messages(instruction: str, *, request_id: str, scene_type: str | None, room: dict | None) -> list[dict[str, Any]]:
    payload = {"request_id": request_id, "instruction": instruction, "scene_type": scene_type, "room": room or None}
    schema_hint = {
        "request_id": request_id,
        "scene_type": scene_type or "room",
        "scene_description": instruction,
        "objects": [
            {
                "id": "obj_000",
                "role": "main seating",
                "category": "sofa",
                "description": "comfortable sofa",
                "estimated_size": [2.4, 0.9, 0.8],
                "count": 1,
                "placement_intent": {"absolute_relations": [], "relative_relations": []},
                "metadata": {},
            }
        ],
        "global_constraints": [],
        "relations": [],
    }
    return [
        {"role": "system", "content": CONVERTER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Convert this request into object_plan JSON with exactly these top-level keys: "
                "request_id, scene_type, scene_description, objects, global_constraints, relations. "
                "Do not output coordinates, pose, center, rotation, jid, asset_id, or asset_ref. "
                f"Schema example: {json.dumps(schema_hint, ensure_ascii=False)}\n"
                f"Input: {json.dumps(payload, ensure_ascii=False)}"
            ),
        },
    ]


def _strip_markdown_fence(text: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def _string(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_string(item) for item in value if _string(item)]


def _relation_list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, (str, dict))]


def _size_or_none(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        size = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return size if all(item > 0 for item in size) else None


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
