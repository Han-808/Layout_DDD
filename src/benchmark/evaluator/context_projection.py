"""Public evaluator-context projection for canonical scenes.

Generated scenes may retain private planning metadata for provenance.  That
metadata is not observable scene evidence and must never influence intrinsic
validity, camera acquisition, VLM selection, Judge prompts, or persisted
evaluator requests.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


GENERATOR_PRIVATE_NESTED_KEYS = frozenset(
    {
        "task_slot",
        "task_slots",
    }
)
GENERATOR_PRIVATE_SCENE_METADATA_KEYS = frozenset(
    {
        "generation_prompt",
        "original_prompt",
        "instruction",
        "object_plan",
    }
)


def project_scene_for_evaluator_context(
    scene: dict[str, Any],
) -> dict[str, Any]:
    """Return a geometry-preserving scene without generator-private intent.

    ``task_slot`` is removed recursively because canonical adapters may retain
    duplicate copies under both object metadata and the scene-level instance
    registry.  Prompt-like fields are removed only from scene metadata so
    unrelated schema fields named ``instruction`` are not silently rewritten.
    The input scene is never mutated.
    """

    if not isinstance(scene, dict):
        raise TypeError("evaluator scene context must be a JSON object")
    task_slots = _task_slots_by_object_id(scene)
    projected = deepcopy(scene)
    _replace_generator_aliases_with_asset_grounding(projected, task_slots)
    _drop_nested_keys(projected, GENERATOR_PRIVATE_NESTED_KEYS)
    metadata = projected.get("metadata")
    if isinstance(metadata, dict):
        for key in GENERATOR_PRIVATE_SCENE_METADATA_KEYS:
            metadata.pop(key, None)
    return projected


def _task_slots_by_object_id(
    scene: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for obj in scene.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        metadata = obj.get("metadata")
        slot = (
            metadata.get("task_slot")
            if isinstance(metadata, dict)
            and isinstance(metadata.get("task_slot"), dict)
            else None
        )
        object_id = str(obj.get("id") or "")
        if object_id and isinstance(slot, dict):
            result[object_id] = slot
    metadata = scene.get("metadata")
    registry = (
        metadata.get("instance_registry")
        if isinstance(metadata, dict)
        and isinstance(metadata.get("instance_registry"), dict)
        else {}
    )
    for instance in registry.get("instances") or []:
        if not isinstance(instance, dict):
            continue
        object_id = str(
            instance.get("evaluator_object_id")
            or instance.get("object_id")
            or ""
        )
        slot = instance.get("task_slot")
        if object_id and isinstance(slot, dict):
            result[object_id] = slot
    return result


def _replace_generator_aliases_with_asset_grounding(
    scene: dict[str, Any],
    task_slots: dict[str, dict[str, Any]],
) -> None:
    """Replace task-slot aliases while preserving asset-observed semantics.

    Some multi-room adapters copy ``task_slot.intended_category`` into the
    ordinary canonical ``category`` field.  Removing the nested task slot alone
    would leave the same private intent available under a scoring field.  When
    that exact alias is detected, prefer an explicit retrieval category or the
    catalog's short/long asset description.  If no independent asset semantic
    exists, omit the alias and let visual evidence resolve identity.
    """

    for obj in scene.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        slot = task_slots.get(str(obj.get("id") or ""))
        if not isinstance(slot, dict):
            continue
        intended_category = _text(slot.get("intended_category"))
        category = _text(obj.get("category"))
        if intended_category and _same_text(category, intended_category):
            replacement = _first_text(
                obj.get("retrieval_category"),
                obj.get("short_desc"),
                obj.get("desc"),
            )
            if replacement:
                obj["category"] = replacement
            else:
                obj.pop("category", None)
        slot_description = _text(slot.get("description"))
        description = _text(obj.get("description"))
        if slot_description and _same_text(description, slot_description):
            replacement = _first_text(
                obj.get("desc"),
                obj.get("short_desc"),
            )
            if replacement and not _same_text(replacement, slot_description):
                obj["description"] = replacement
            else:
                obj.pop("description", None)


def _first_text(*values: Any) -> str | None:
    for value in values:
        text = _text(value)
        if text:
            return text
    return None


def _text(value: Any) -> str:
    return str(value or "").strip()


def _same_text(left: str, right: str) -> bool:
    return " ".join(left.casefold().split()) == " ".join(
        right.casefold().split()
    )


def _drop_nested_keys(value: Any, keys: frozenset[str]) -> None:
    if isinstance(value, dict):
        for key in list(value):
            if str(key) in keys:
                value.pop(key, None)
                continue
            _drop_nested_keys(value[key], keys)
    elif isinstance(value, list):
        for item in value:
            _drop_nested_keys(item, keys)


__all__ = [
    "GENERATOR_PRIVATE_NESTED_KEYS",
    "GENERATOR_PRIVATE_SCENE_METADATA_KEYS",
    "project_scene_for_evaluator_context",
]
