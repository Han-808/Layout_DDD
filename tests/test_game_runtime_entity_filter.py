from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.game_scene import (
    GameSceneExportError,
    build_scene_and_collision_geometry,
)


_CUBE_VERTICES = [
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, 0.5, 0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, -0.5],
    [0.5, 0.5, 0.5],
]
_CUBE_FACES = [
    [0, 1, 3],
    [0, 3, 2],
    [4, 6, 7],
    [4, 7, 5],
    [0, 4, 5],
    [0, 5, 1],
    [2, 3, 7],
    [2, 7, 6],
    [0, 2, 6],
    [0, 6, 4],
    [1, 5, 7],
    [1, 7, 3],
]


def _runtime_role(
    classification: str,
    *,
    source: str,
    declared: str | None,
    signal_keys: list[str] | None = None,
) -> dict:
    return {
        "classification": classification,
        "source": source,
        "declared_entity_kind": declared,
        "signal_keys": list(signal_keys or []),
        "family_graph_path": None,
    }


def _mesh(
    object_id: str,
    *,
    x: float,
    entity_kind: str = "static",
    runtime_role: dict | None = None,
    source_names: list[str] | None = None,
) -> dict:
    vertices = [
        [vertex[0] + x, vertex[1] + 0.5, vertex[2]]
        for vertex in _CUBE_VERTICES
    ]
    entry = {
        "id": object_id,
        "category": "level_geometry",
        "category_source": "declared",
        "entity_kind": entity_kind,
        "rotation_quaternion": [0.0, 0.0, 0.0, 1.0],
        "vertices": vertices,
        "faces": _CUBE_FACES,
        "mesh_complete": True,
        "world_bounds": {
            "min": [x - 0.5, -0.0, -0.5],
            "max": [x + 0.5, 1.0, 0.5],
        },
        "source_names": list(source_names or [object_id]),
    }
    if runtime_role is not None:
        entry["runtime_role"] = runtime_role
    return entry


def _payload(objects: list[dict]) -> dict:
    return {
        "schema_version": "game_scene_probe_v1",
        "up_axis": "y",
        "unit_scale": 1.0,
        "captured_at_tick": 60,
        "deterministic_seed": 20260727,
        "objects": objects,
    }


def _export(objects: list[dict], tmp_path: Path) -> tuple[dict, dict]:
    return build_scene_and_collision_geometry(
        _payload(objects),
        scene_id="runtime_filter_scene",
        request_id="runtime_filter_request",
        scene_type="game_level",
        mesh_dir=tmp_path / "collision_geometry",
        collapse_contained=False,
    )


def _runtime_filter(scene: dict) -> dict:
    return scene["metadata"]["game_scene_import"]["individualization"][
        "runtime_filter"
    ]


def test_explicit_dynamic_actor_is_excluded_from_scene_and_collision(
    tmp_path: Path,
) -> None:
    static = _mesh(
        "arena_wall",
        x=0.0,
        runtime_role=_runtime_role(
            "static",
            source="declared_benchmark_entity_kind",
            declared="static",
        ),
    )
    actor = _mesh(
        "combatant_body",
        x=3.0,
        entity_kind="dynamic_actor",
        runtime_role=_runtime_role(
            "dynamic_actor",
            source="declared_benchmark_entity_kind",
            declared="dynamic_actor",
        ),
    )

    scene, collision = _export([static, actor], tmp_path)

    assert {item["id"] for item in scene["objects"]} == {"arena_wall"}
    assert set(collision["objects"]) == {"arena_wall"}
    runtime_filter = _runtime_filter(scene)
    assert runtime_filter["input_mesh_count"] == 2
    assert runtime_filter["retained_mesh_count"] == 1
    assert runtime_filter["excluded_mesh_count"] == 1
    assert runtime_filter["excluded"] == [
        {
            "id": "combatant_body",
            "entity_kind": "dynamic_actor",
            "reason": "non_static_runtime_entity",
            "classification_source": "declared_benchmark_entity_kind",
            "signal_keys": [],
            "family_graph_path": None,
        }
    ]


def test_misleading_actor_like_names_do_not_remove_explicit_static_geometry(
    tmp_path: Path,
) -> None:
    misleading_static = _mesh(
        "enemy_bot_player_spawn_wall",
        x=0.0,
        runtime_role=_runtime_role(
            "static",
            source="declared_benchmark_entity_kind",
            declared="static",
        ),
        source_names=["player", "enemy_bot", "spawn_helper"],
    )

    scene, collision = _export([misleading_static], tmp_path)

    assert [item["id"] for item in scene["objects"]] == [
        "enemy_bot_player_spawn_wall"
    ]
    assert set(collision["objects"]) == {"enemy_bot_player_spawn_wall"}
    runtime_filter = _runtime_filter(scene)
    assert runtime_filter["excluded_mesh_count"] == 0
    assert runtime_filter["unknown_stable_kept_count"] == 0
    assert runtime_filter["static_environment_certified"] is True


def test_unknown_stable_mesh_is_retained_and_explicitly_audited(
    tmp_path: Path,
) -> None:
    # Legacy probes have no runtime_role. They remain usable, but the durable
    # report must make clear that their static classification was not certified.
    unknown_stable = _mesh("unnamed_mesh_17", x=0.0)

    scene, collision = _export([unknown_stable], tmp_path)

    assert [item["id"] for item in scene["objects"]] == ["unnamed_mesh_17"]
    assert set(collision["objects"]) == {"unnamed_mesh_17"}
    runtime_filter = _runtime_filter(scene)
    assert runtime_filter["unknown_stable_kept_count"] == 1
    assert runtime_filter["unknown_stable_kept_ids"] == ["unnamed_mesh_17"]
    assert runtime_filter["static_environment_certified"] is False
    assert runtime_filter["classification_counts"] == {"static": 1}


def test_all_runtime_entities_fail_loudly_instead_of_exporting_empty_level(
    tmp_path: Path,
) -> None:
    actor = _mesh(
        "bot_body",
        x=0.0,
        entity_kind="dynamic_actor",
        runtime_role=_runtime_role(
            "dynamic_actor",
            source="runtime_actor_family_signal",
            declared=None,
            signal_keys=["team", "hp"],
        ),
    )
    helper = _mesh(
        "collision_debug_proxy",
        x=3.0,
        entity_kind="transient_helper",
        runtime_role=_runtime_role(
            "transient_helper",
            source="runtime_transient_family_signal",
            declared=None,
            signal_keys=["lifetime"],
        ),
    )

    with pytest.raises(
        GameSceneExportError,
        match=(
            "runtime entity filtering removed every probed mesh; "
            "the level has no static environment geometry"
        ),
    ):
        _export([actor, helper], tmp_path)
