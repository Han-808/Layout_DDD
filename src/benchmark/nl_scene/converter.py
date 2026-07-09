from __future__ import annotations

import json
import os
import re
from typing import Any

from benchmark.models.openai_compatible_model import OpenAICompatibleModel


CONVERTER_SYSTEM_PROMPT = """Convert a natural-language scene/layout request into structured JSON for asset retrieval.
Infer scene_type, object count, object categories, retrieval descriptions, optional estimated sizes, and global style/function constraints.
Do not infer placement coordinates. Do not output center, rotation, target_pose, exact asset ids, jids, or expected_relations.
Return exactly one JSON object and no Markdown.
""".strip()

FORBIDDEN_OBJECT_KEYS = {"center", "position", "rotation", "target_pose", "pose", "jid", "asset_jid", "expected_relations"}


class SceneSpecConversionError(RuntimeError):
    """Raised when an NL instruction cannot be converted into a valid scene spec."""


def convert_nl_to_scene_spec(
    instruction: str,
    *,
    scene_type: str | None = None,
    room: dict | None = None,
    model_config: dict | None = None,
) -> dict:
    """Convert a natural-language instruction into an asset-request scene spec."""

    clean_instruction = str(instruction or "").strip()
    if not clean_instruction:
        raise ValueError("instruction must be a non-empty string")
    messages = _converter_messages(clean_instruction, scene_type=scene_type, room=room)
    first_response = call_chat_model(messages, model_config=model_config, response_format_json=True, call_type="nl_scene_converter")
    try:
        parsed = parse_json_object_from_text(first_response)
    except ValueError as first_error:
        correction_messages = [
            *messages,
            {"role": "assistant", "content": first_response},
            {
                "role": "user",
                "content": (
                    "The previous response was not valid JSON. Return one corrected JSON object only, "
                    "following the same schema and still omitting positions, rotations, and exact asset ids."
                ),
            },
        ]
        second_response = call_chat_model(correction_messages, model_config=model_config, response_format_json=True, call_type="nl_scene_converter_retry")
        try:
            parsed = parse_json_object_from_text(second_response)
        except ValueError as second_error:
            raise SceneSpecConversionError(f"Model did not return valid scene-spec JSON: {second_error}") from first_error
    return validate_scene_spec(parsed, instruction=clean_instruction, scene_type=scene_type)


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


def validate_scene_spec(spec: dict, *, instruction: str = "", scene_type: str | None = None) -> dict:
    if not isinstance(spec, dict):
        raise SceneSpecConversionError("scene spec must be a JSON object")
    normalized: dict[str, Any] = {
        "scene_type": _string(spec.get("scene_type") or scene_type or "room"),
        "scene_description": _string(spec.get("scene_description") or instruction),
        "objects": [],
        "global_constraints": _string_list(spec.get("global_constraints")),
        "relations": [],
    }
    objects = spec.get("objects")
    if not isinstance(objects, list):
        raise SceneSpecConversionError("scene spec must contain an objects list")
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            raise SceneSpecConversionError(f"objects[{index}] must be a JSON object")
        forbidden = sorted(key for key in FORBIDDEN_OBJECT_KEYS if key in item)
        if forbidden:
            raise SceneSpecConversionError(f"objects[{index}] contains forbidden placement/asset keys: {forbidden}")
        obj: dict[str, Any] = {
            "id": item.get("id", index),
            "role": _string(item.get("role")),
            "category": _string(item.get("category") or "object"),
            "description": _string(item.get("description") or item.get("short_desc") or item.get("category") or "object"),
            "count": _positive_int(item.get("count"), default=1),
        }
        estimated_size = _size_or_none(item.get("estimated_size"))
        if estimated_size is not None:
            obj["estimated_size"] = estimated_size
        normalized["objects"].append(obj)
    relations = spec.get("relations")
    if isinstance(relations, list):
        normalized["relations"] = [item for item in relations if isinstance(item, dict)]
    return normalized


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
        raise SceneSpecConversionError(
            "No chat model configured. Provide model_config with base_url/model or set OPENAI_BASE_URL and MODEL_NAME/OPENAI_MODEL."
        )
    model = OpenAICompatibleModel(
        name="nl_scene_mvp",
        endpoint=str(endpoint),
        model_id=str(model_id),
        api_key=api_key,
        temperature=float(config.get("temperature", 0.2)),
        max_tokens=int(config.get("max_tokens", 2048)),
        response_format_json=response_format_json,
    )
    return model.chat_messages(messages, response_format_json=response_format_json, call_type=call_type)


def _converter_messages(instruction: str, *, scene_type: str | None, room: dict | None) -> list[dict[str, Any]]:
    payload = {"instruction": instruction, "scene_type": scene_type, "room": room or None}
    return [
        {"role": "system", "content": CONVERTER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Convert this request into JSON with keys: scene_type, scene_description, objects, "
                "global_constraints, relations. Objects need id, role, category, description, estimated_size, count.\n"
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
