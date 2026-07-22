from __future__ import annotations

import json
import os
import re
from typing import Any

from benchmark.models.openai_compatible_model import OpenAICompatibleModel


FINE_GRAINED = "fine_grained"
COARSE_GRAINED = "coarse_grained"
AUTO_GRANULARITY = "auto"
PROMPT_GRANULARITIES = {FINE_GRAINED, COARSE_GRAINED}
PROMPT_GRANULARITY_OPTIONS = {AUTO_GRANULARITY, *PROMPT_GRANULARITIES}


GRANULARITY_CLASSIFIER_SYSTEM_PROMPT = """Classify the granularity of a natural-language 3D scene-generation request.
Return fine_grained only when the request describes a multi-object layout with both:
1. multiple concrete object requirements or detailed object attributes; and
2. at least one explicit object-object or object-architecture relationship.
Object-object relationships include relative placement, support, distance, direction, facing, or ordering between objects.
Object-architecture relationships in active v1 include placement relative only to the benchmark room's walls, floor, or ceiling.
Return coarse_grained for broad scene type, style, atmosphere, or function requests, and for requests that only mention one or a few objects without explicit layout relationships.
Use only evidence literally present in the request. Do not infer missing objects, relationships, attributes, or scene contents.
Return one JSON object with: prompt_granularity, object_evidence, relationship_evidence, reason. No Markdown.
""".strip()


FINE_GRAINED_CONVERTER_SYSTEM_PROMPT = """Convert a fine-grained natural-language scene/layout request into a canonical object_plan JSON object.
Use a SceneEval-style decomposition conceptually: explicit object requirements, object attributes, soft object-object relation intents, soft object-architecture relation intents, and global constraints.
Return object_plan JSON only. Do not output coordinates, positions, center, rotation, pose, target_pose, exact asset ids, jids, asset_ref, or ground-truth pose.
Relations are explicit prompt claims, not inferred commonsense. Map them to the following frozen predicates whenever their meaning matches exactly.
OOR predicates: left, right, in_front, behind, above, below, near, far, contact, on_top_of, within, contains, aligned, parallel, perpendicular, between, ordered, around.
OAR predicates: on_floor, against_wall, near_wall, at_corner, near_corner, room_center, room_region, along_wall, mounted_on_wall, attached_to_ceiling, hung_from_ceiling.
Every relation must include family="oor" or family="oar" and type. Binary OOR uses subject_id and object_id. between uses subject_id plus exactly two object_ids. ordered uses object_ids in the requested order and an explicit axis or direction. around uses subject_ids for surrounding members and object_id for the center anchor. OAR uses subject_id plus target and, when applicable, wall, corner, or region.
If an explicit relationship does not match a frozen predicate, do not discard or force-map it: emit a concise snake_case type, preserve its exact wording in raw_relation, and keep the relevant object IDs or architecture target. The evaluator will route that claim to a VLM.
Canonical architecture targets are north_wall, south_wall, east_wall, west_wall, floor, ceiling, northeast_corner, northwest_corner, southeast_corner, and southwest_corner. In the fixed room frame, right/left/back/front wall mean east/west/north/south wall respectively.
Record every explicit textual requirement in explicit_claims. Do not infer unstated objects, relations, attributes, or visual effects.
Return exactly one JSON object and no Markdown.
""".strip()


COARSE_GRAINED_CONVERTER_SYSTEM_PROMPT = """Convert a coarse-grained natural-language scene request into a canonical object_plan JSON object for benchmark evaluation.
Do not complete the scene or invent a conventional furniture list. Include an object only when it is explicitly named or unambiguously required by the text. Include a relation only when the text explicitly states it.
Preserve only high-level requirements stated in the source text in explicit_claims. Do not expand them into implied objects, relations, attributes, visual effects, or commonsense scene completions. For example, a bedroom request does not imply a required bed unless the text names one.
Keep global_constraints limited to explicit room-level constraints. Return object_plan JSON only. Do not output coordinates, positions, center, rotation, pose, target_pose, exact asset ids, jids, asset_ref, or ground-truth pose.
Return exactly one JSON object and no Markdown.
""".strip()


# Backward-compatible public name for code importing the old prompt constant.
CONVERTER_SYSTEM_PROMPT = FINE_GRAINED_CONVERTER_SYSTEM_PROMPT

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

_ARCHITECTURE_TARGET_ALIASES = {
    "north_wall": "north_wall",
    "south_wall": "south_wall",
    "east_wall": "east_wall",
    "west_wall": "west_wall",
    "right_wall": "east_wall",
    "left_wall": "west_wall",
    "back_wall": "north_wall",
    "rear_wall": "north_wall",
    "front_wall": "south_wall",
    "floor": "floor",
    "ceiling": "ceiling",
    "northeast_corner": "northeast_corner",
    "northwest_corner": "northwest_corner",
    "southeast_corner": "southeast_corner",
    "southwest_corner": "southwest_corner",
    "room_center": "center_region",
    "center": "center_region",
    "centre": "center_region",
    "middle": "center_region",
    "center_region": "center_region",
    "centre_region": "center_region",
    "north_region": "north_region",
    "south_region": "south_region",
    "east_region": "east_region",
    "west_region": "west_region",
    "northeast_region": "northeast_region",
    "northwest_region": "northwest_region",
    "southeast_region": "southeast_region",
    "southwest_region": "southwest_region",
}

_OAR_RELATION_TYPES = {
    "against_wall",
    "near_wall",
    "on_floor",
    "at_corner",
    "near_corner",
    "room_center",
    "room_region",
    "along_wall",
    "mounted_on_wall",
    "attached_to_ceiling",
    "hung_from_ceiling",
}

_OOR_RELATION_TYPES = {
    "left",
    "right",
    "in_front",
    "behind",
    "above",
    "below",
    "near",
    "far",
    "contact",
    "on_top_of",
    "within",
    "contains",
    "aligned",
    "parallel",
    "perpendicular",
    "between",
    "ordered",
    "around",
}


class ObjectPlanConversionError(RuntimeError):
    """Raised when an NL instruction cannot be converted into a valid object_plan."""


SceneSpecConversionError = ObjectPlanConversionError


_ROOM_TERM_RE = re.compile(r"\b(?:room|bedroom|living\s+room|cafe|interior\s+space)\b", re.IGNORECASE)
_ROOM_NUMBER = r"(?:\d+(?:\.\d+)?|\.\d+)"
_ROOM_UNIT = r"(?:m|meter(?:s)?|metre(?:s)?)"
_ROOM_SEPARATOR = r"(?:x|by)"
_ROOM_TRIPLE_RE = re.compile(
    rf"(?P<width>{_ROOM_NUMBER})\s*(?:{_ROOM_UNIT})?\s*{_ROOM_SEPARATOR}\s*"
    rf"(?P<depth>{_ROOM_NUMBER})\s*(?:{_ROOM_UNIT})?\s*{_ROOM_SEPARATOR}\s*"
    rf"(?P<height>{_ROOM_NUMBER})\s*(?:{_ROOM_UNIT})?",
    re.IGNORECASE,
)
_ROOM_PAIR_RE = re.compile(
    rf"(?P<width>{_ROOM_NUMBER})\s*(?:{_ROOM_UNIT})?\s*{_ROOM_SEPARATOR}\s*"
    rf"(?P<depth>{_ROOM_NUMBER})\s*(?:{_ROOM_UNIT})?",
    re.IGNORECASE,
)
_ROOM_MEASUREMENT_CUE_RE = re.compile(
    r"\b(?:measure(?:s|d|ment|ments|ing)?|size(?:d)?|dimensions?|of\s+size|is)\b",
    re.IGNORECASE,
)
_ROOM_CONTENT_CUE_RE = re.compile(
    r"\b(?:contain(?:s|ing)?|include(?:s|ing)?|feature(?:s|d|ing)?|hold(?:s|ing)?|"
    r"furnished\s+with)\b",
    re.IGNORECASE,
)
_ROOM_OBJECT_WITH_RE = re.compile(
    r"\bwith\s+(?:an?|one|two|three|four|the)\s+"
    r"(?!(?:size|dimensions?|width|depth|height|ceiling)\b)[a-z][a-z_-]*",
    re.IGNORECASE,
)


def extract_room_dimension_claims(instruction: str) -> dict[str, float]:
    """Extract only explicit room dimensions from controlled natural language.

    This parser is deliberately narrow. It recognizes room-linked metric
    dimensions but does not infer dimensions from object sizes, room type, or
    commonsense. Missing axes are resolved later by the benchmark contract.
    """

    text = re.sub(r"\s+", " ", str(instruction or "").replace("×", "x")).strip()
    if not text:
        return {}
    claims: dict[str, float] = {}
    for clause in re.split(r"(?<=[.!?;])\s+", text):
        if not _ROOM_TERM_RE.search(clause):
            continue
        occupied_spans: list[tuple[int, int]] = []
        for match in _ROOM_TRIPLE_RE.finditer(clause):
            if not _measurement_is_room_linked(clause, match):
                continue
            for axis in ("width", "depth", "height"):
                _record_room_dimension(claims, axis, match.group(axis))
            occupied_spans.append(match.span())
        for match in _ROOM_PAIR_RE.finditer(clause):
            if any(start <= match.start() and match.end() <= end for start, end in occupied_spans):
                continue
            if not _measurement_is_room_linked(clause, match):
                continue
            _record_room_dimension(claims, "width", match.group("width"))
            _record_room_dimension(claims, "depth", match.group("depth"))

        for axis, label in (("width", "width"), ("depth", "depth"), ("height", "(?:ceiling\\s+)?height")):
            label_then_value = re.compile(
                rf"\b{label}\b\s*(?:of|is|=|:)?\s*(?P<value>{_ROOM_NUMBER})\s*{_ROOM_UNIT}\b",
                re.IGNORECASE,
            )
            for match in label_then_value.finditer(clause):
                if _label_is_room_linked(clause, match):
                    _record_room_dimension(claims, axis, match.group("value"))

        suffix_axes = (("width", "wide"), ("depth", "deep"), ("height", "high"))
        for axis, suffix in suffix_axes:
            value_then_label = re.compile(
                rf"(?P<value>{_ROOM_NUMBER})\s*{_ROOM_UNIT}\s+{suffix}\b",
                re.IGNORECASE,
            )
            for match in value_then_label.finditer(clause):
                if axis == "height" and any(end <= match.start() for _, end in occupied_spans):
                    _record_room_dimension(claims, axis, match.group("value"))
                elif _label_is_room_linked(clause, match):
                    _record_room_dimension(claims, axis, match.group("value"))
    return claims


def _measurement_is_room_linked(clause: str, measurement: re.Match[str]) -> bool:
    terms = list(_ROOM_TERM_RE.finditer(clause))
    for term in terms:
        if measurement.end() <= term.start():
            between = clause[measurement.end() : term.start()]
            if len(between) <= 24:
                return True
        elif term.end() <= measurement.start():
            between = clause[term.end() : measurement.start()]
            if _contains_object_content_cue(between):
                continue
            if len(between) <= 20 or (
                len(between) <= 100 and _ROOM_MEASUREMENT_CUE_RE.search(between)
            ):
                return True
    return False


def _label_is_room_linked(clause: str, measurement: re.Match[str]) -> bool:
    for term in _ROOM_TERM_RE.finditer(clause):
        if term.end() <= measurement.start():
            between = clause[term.end() : measurement.start()]
        elif measurement.end() <= term.start():
            between = clause[measurement.end() : term.start()]
        else:
            between = ""
        if len(between) <= 100 and not _contains_object_content_cue(between):
            return True
    return False


def _contains_object_content_cue(text: str) -> bool:
    return bool(_ROOM_CONTENT_CUE_RE.search(text) or _ROOM_OBJECT_WITH_RE.search(text))


def _record_room_dimension(claims: dict[str, float], axis: str, raw_value: str) -> None:
    value = float(raw_value)
    previous = claims.get(axis)
    if previous is not None and abs(previous - value) > 1.0e-6:
        raise ObjectPlanConversionError(
            f"conflicting explicit room {axis} dimensions in natural-language input: "
            f"{previous:g}m versus {value:g}m"
        )
    claims[axis] = value


def classify_prompt_granularity(
    instruction: str,
    *,
    model_config: dict | None = None,
) -> dict:
    """Classify prompt routing without generating or completing scene content."""

    clean_instruction = str(instruction or "").strip()
    if not clean_instruction:
        raise ValueError("instruction must be a non-empty string")
    messages = [
        {"role": "system", "content": GRANULARITY_CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({"instruction": clean_instruction}, ensure_ascii=False)},
    ]
    response = call_chat_model(
        messages,
        model_config=model_config,
        response_format_json=True,
        call_type="prompt_granularity_classifier",
    )
    try:
        parsed = parse_json_object_from_text(response)
        return _validate_granularity_classification(parsed)
    except (ValueError, ObjectPlanConversionError) as first_error:
        correction_messages = [
            *messages,
            {"role": "assistant", "content": response},
            {
                "role": "user",
                "content": (
                    "Return corrected JSON only. prompt_granularity must be fine_grained or coarse_grained. "
                    "Do not add scene objects or relations that are not literal evidence from the request."
                ),
            },
        ]
        corrected = call_chat_model(
            correction_messages,
            model_config=model_config,
            response_format_json=True,
            call_type="prompt_granularity_classifier_retry",
        )
        try:
            return _validate_granularity_classification(parse_json_object_from_text(corrected))
        except (ValueError, ObjectPlanConversionError) as second_error:
            raise ObjectPlanConversionError(
                f"Model did not return a valid prompt granularity classification: {second_error}"
            ) from first_error


def convert_nl_to_object_plan(
    instruction: str,
    *,
    request_id: str = "request_001",
    scene_type: str | None = None,
    room: dict | None = None,
    prompt_granularity: str = FINE_GRAINED,
    model_config: dict | None = None,
) -> dict:
    """Convert a natural-language instruction into canonical object_plan."""

    clean_instruction = str(instruction or "").strip()
    if not clean_instruction:
        raise ValueError("instruction must be a non-empty string")
    granularity = _prompt_granularity(prompt_granularity)
    messages = _converter_messages(
        clean_instruction,
        request_id=request_id,
        scene_type=scene_type,
        room=room,
        prompt_granularity=granularity,
    )
    first_response = call_chat_model(
        messages,
        model_config=model_config,
        response_format_json=True,
        call_type=f"object_plan_converter_{granularity}",
    )
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
                    f"Apply the {granularity} conversion policy from the system message. "
                    "Still omit coordinates, positions, rotations, exact asset ids, jids, asset_ref, and pose fields."
                ),
            },
        ]
        second_response = call_chat_model(
            correction_messages,
            model_config=model_config,
            response_format_json=True,
            call_type=f"object_plan_converter_{granularity}_retry",
        )
        try:
            parsed = parse_json_object_from_text(second_response)
        except ValueError as second_error:
            raise ObjectPlanConversionError(f"Model did not return valid object_plan JSON: {second_error}") from first_error
    return validate_object_plan_json(
        parsed,
        request_id=request_id,
        instruction=clean_instruction,
        scene_type=scene_type,
        prompt_granularity=granularity,
    )


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


def validate_object_plan_json(
    plan: dict,
    *,
    request_id: str = "request_001",
    instruction: str = "",
    scene_type: str | None = None,
    prompt_granularity: str = FINE_GRAINED,
) -> dict:
    if not isinstance(plan, dict):
        raise ObjectPlanConversionError("object_plan must be a JSON object")
    granularity = _prompt_granularity(prompt_granularity)
    normalized: dict[str, Any] = {
        "request_id": _string(plan.get("request_id") or request_id),
        "scene_type": _string(plan.get("scene_type") or scene_type or "room"),
        "scene_description": _string(plan.get("scene_description") or instruction),
        "prompt_granularity": granularity,
        "explicit_claims": _string_list(plan.get("explicit_claims")),
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
        object_ids = {str(item["id"]) for item in normalized["objects"]}
        normalized["relations"] = [
            _normalize_object_plan_relation(item, object_ids=object_ids)
            for item in relations
            if isinstance(item, dict)
        ]
    return normalized


def _normalize_object_plan_relation(relation: dict, *, object_ids: set[str]) -> dict[str, Any]:
    """Normalize converter relation syntax without inventing a scene claim."""

    subject = _first_present(relation, ["subject_id", "subject"])
    relation_type = _first_present(relation, ["type", "predicate", "relation"])
    target = _first_present(
        relation,
        ["object_id", "object", "anchor_id", "target_id", "target", "architectural_element", "wall", "corner", "region"],
    )
    subject_ids = _relation_id_list(_first_present(relation, ["subject_ids", "member_ids"]))
    relation_object_ids = _relation_id_list(
        _first_present(relation, ["object_ids", "objects", "anchor_ids", "target_ids"])
    )
    subject_list = _relation_id_list(subject)
    target_list = _relation_id_list(target)
    raw_family = _string(relation.get("family")).lower().replace("-", "_").replace(" ", "_")
    family_aliases = {
        "oor": "oor",
        "oo": "oor",
        "object_object": "oor",
        "oar": "oar",
        "oa": "oar",
        "object_architecture": "oar",
    }
    family = family_aliases.get(raw_family, raw_family)
    canonical_target = _canonical_architecture_target(target) if not target_list else None
    clean_type = _string(relation_type).lower().replace("-", "_").replace(" ", "_")
    clean_target = "" if target_list else _string(target)
    group_subject_ids = subject_ids or subject_list
    group_object_ids = relation_object_ids or target_list
    if not family:
        if clean_type in _OAR_RELATION_TYPES:
            family = "oar"
        elif clean_type in _OOR_RELATION_TYPES or clean_target in object_ids or (
            (group_subject_ids or group_object_ids)
            and all(value in object_ids for value in [*group_subject_ids, *group_object_ids])
        ):
            family = "oor"
        elif canonical_target is not None:
            family = "oar"

    normalized: dict[str, Any] = {"family": family, "type": clean_type}
    clean_subject = "" if subject_list else _string(subject)
    if clean_subject:
        normalized["subject_id"] = clean_subject
    if family == "oar":
        normalized["target"] = canonical_target or clean_target
        if canonical_target is not None and canonical_target != clean_target:
            normalized["raw_target"] = clean_target
        for key in ("wall", "corner", "region"):
            if relation.get(key) is not None:
                normalized[key] = _string(relation[key]).lower().replace("-", "_").replace(" ", "_")
    else:
        if clean_type == "between":
            normalized["object_ids"] = group_object_ids
        elif clean_type == "ordered":
            normalized["object_ids"] = group_object_ids or group_subject_ids
        elif clean_type == "around":
            normalized["subject_ids"] = group_subject_ids or relation_object_ids
            normalized["object_id"] = clean_target
        elif group_subject_ids or group_object_ids:
            if group_subject_ids:
                normalized["subject_ids"] = group_subject_ids
            if group_object_ids:
                normalized["object_ids"] = group_object_ids
        else:
            normalized["object_id"] = clean_target
    for key in ("raw_relation", "confidence", "source", "reason", "axis", "direction", "order"):
        if relation.get(key) is not None:
            normalized[key] = relation[key]
    return normalized


def _relation_id_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [_string(item) for item in value if _string(item)]


def _canonical_architecture_target(value: Any) -> str | None:
    text = _string(value).lower().replace("-", "_").replace(" ", "_")
    return _ARCHITECTURE_TARGET_ALIASES.get(text)


def _first_present(mapping: dict, keys: list[str]) -> Any:
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return None


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
    if "api_key" in config:
        raise ObjectPlanConversionError(
            "model_config must not contain literal api_key; use api_key_env instead"
        )
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
    api_key_env = config.get("api_key_env")
    if not endpoint or not model_id:
        raise ObjectPlanConversionError(
            "No chat model configured. Provide model_config with base_url/model or set OPENAI_BASE_URL and MODEL_NAME/OPENAI_MODEL."
        )
    model = OpenAICompatibleModel(
        name="object_plan_converter",
        endpoint=str(endpoint),
        model_id=str(model_id),
        api_key_env=str(api_key_env) if api_key_env else None,
        temperature=float(config.get("temperature", 0.2)),
        max_tokens=int(config.get("max_tokens", 2048)),
        context_length=(
            int(config["context_length"])
            if config.get("context_length") is not None
            else None
        ),
        timeout_seconds=int(config.get("timeout_seconds", 180)),
        response_format_json=response_format_json,
        max_retries=int(config.get("max_retries", 0)),
        retry_backoff_seconds=float(config.get("retry_backoff_seconds", 1.0)),
        max_tokens_field=str(config.get("max_tokens_field", "max_tokens")),
        send_temperature=bool(config.get("send_temperature", True)),
        require_api_key=(
            bool(config["require_api_key"])
            if config.get("require_api_key") is not None
            else None
        ),
    )
    return model.chat_messages(messages, response_format_json=response_format_json, call_type=call_type)


def _converter_messages(
    instruction: str,
    *,
    request_id: str,
    scene_type: str | None,
    room: dict | None,
    prompt_granularity: str,
) -> list[dict[str, Any]]:
    granularity = _prompt_granularity(prompt_granularity)
    payload = {"request_id": request_id, "instruction": instruction, "scene_type": scene_type, "room": room or None}
    schema_hint = {
        "request_id": request_id,
        "scene_type": scene_type or "room",
        "scene_description": instruction,
        "prompt_granularity": granularity,
        "explicit_claims": ["one comfortable sofa"],
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
        "relations": [
            {
                "family": "oar",
                "subject_id": "obj_000",
                "type": "against_wall",
                "target": "east_wall",
            }
        ],
    }
    if granularity == COARSE_GRAINED:
        schema_hint["explicit_claims"] = [instruction]
        schema_hint["objects"] = []
    system_prompt = (
        COARSE_GRAINED_CONVERTER_SYSTEM_PROMPT
        if granularity == COARSE_GRAINED
        else FINE_GRAINED_CONVERTER_SYSTEM_PROMPT
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Convert this request into object_plan JSON with exactly these top-level keys: "
                "request_id, scene_type, scene_description, prompt_granularity, explicit_claims, "
                "objects, global_constraints, relations. "
                "Do not output coordinates, pose, center, rotation, jid, asset_id, or asset_ref. "
                f"Schema example: {json.dumps(schema_hint, ensure_ascii=False)}\n"
                f"Input: {json.dumps(payload, ensure_ascii=False)}"
            ),
        },
    ]


def _prompt_granularity(value: Any) -> str:
    granularity = _string(value)
    if granularity not in PROMPT_GRANULARITIES:
        raise ValueError(
            f"prompt_granularity must be one of {sorted(PROMPT_GRANULARITIES)}, got {granularity!r}"
        )
    return granularity


def _validate_granularity_classification(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ObjectPlanConversionError("prompt granularity classification must be a JSON object")
    granularity = _prompt_granularity(value.get("prompt_granularity"))
    return {
        "prompt_granularity": granularity,
        "object_evidence": _string_list(value.get("object_evidence")),
        "relationship_evidence": _string_list(value.get("relationship_evidence")),
        "reason": _string(value.get("reason")),
    }


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
