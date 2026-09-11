from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable


# Only IDs belonging to observable scene entities may authorize Camera DSL
# targets. Operational IDs (scene_id, request_id, asset_id, run_id, etc.) are
# intentionally absent.
SCENE_ENTITY_COLLECTIONS = frozenset(
    {
        "objects",
        "object_instances",
        "scene_objects",
        "instances",
        "entities",
        "subjects",
        "members",
        "targets",
        "architecture",
        "architecture_elements",
        "architectural_elements",
        "walls",
        "floors",
        "ceilings",
        "rooms",
        "doors",
        "windows",
        "planes",
        "boundaries",
        "supports",
    }
)
SCENE_ENTITY_ID_FIELDS = frozenset(
    {
        "id",
        "object_id",
        "instance_id",
        "entity_id",
        "target_id",
        "architecture_id",
        "wall_id",
        "floor_id",
        "ceiling_id",
        "room_id",
        "door_id",
        "window_id",
        "plane_id",
        "boundary_id",
        "support_id",
    }
)
SCENE_ENTITY_ID_LIST_FIELDS = frozenset(
    {
        "object_ids",
        "instance_ids",
        "entity_ids",
        "target_ids",
        "architecture_ids",
        "wall_ids",
        "floor_ids",
        "ceiling_ids",
        "room_ids",
        "door_ids",
        "window_ids",
        "plane_ids",
        "boundary_ids",
        "support_ids",
    }
)


def authoritative_scene_target_ids(scene: Mapping[str, Any]) -> tuple[str, ...]:
    """Return only IDs rooted in explicit observable-entity collections."""

    if not isinstance(scene, Mapping):
        raise TypeError("scene target authority requires a mapping")
    values: list[str] = []
    for key, value in scene.items():
        key_name = str(key)
        if key_name in SCENE_ENTITY_ID_LIST_FIELDS:
            _add_scalar_ids(values, value)
        elif key_name in SCENE_ENTITY_COLLECTIONS:
            _visit_entity_container(value, values)
    return _ordered_ids(values)


def merge_authoritative_target_ids(
    explicit_ids: Iterable[Any],
    scene: Mapping[str, Any],
) -> tuple[str, ...]:
    """Combine claim-scoped IDs with independently discoverable scene IDs."""

    return _ordered_ids(
        [
            *(str(value) for value in explicit_ids if str(value).strip()),
            *authoritative_scene_target_ids(scene),
        ]
    )


def _visit_entity_container(value: Any, result: list[str]) -> None:
    if isinstance(value, Mapping):
        for key in SCENE_ENTITY_ID_FIELDS:
            if key in value:
                _add_scalar_ids(result, value[key])
        for key in SCENE_ENTITY_ID_LIST_FIELDS:
            if key in value:
                _add_scalar_ids(result, value[key])
        for key, child in value.items():
            if str(key) in SCENE_ENTITY_COLLECTIONS:
                _visit_entity_container(child, result)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _visit_entity_container(child, result)


def _add_scalar_ids(result: list[str], value: Any) -> None:
    values = value if isinstance(value, (list, tuple)) else (value,)
    for item in values:
        if isinstance(item, (str, int)) and str(item).strip():
            result.append(str(item).strip())


def _ordered_ids(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(value).strip()
            for value in values
            if str(value).strip()
        )
    )
