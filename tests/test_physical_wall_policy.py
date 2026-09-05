from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from benchmark.adapters.layout_json.prompt import build_layout_json_method_input
from benchmark.architecture_policy import (
    ARCHITECTURE_POLICY_VERSION,
    CANONICAL_WALL_IDS,
    architecture_contract_from_scene,
    require_generated_architecture_targets_active,
    resolve_architecture_activation,
    validate_architecture_contract,
)
from benchmark.evaluator.OAR.evaluator import evaluate_oar
from benchmark.evaluator.OAR.geometry import normalize_room
from benchmark.evaluator.generic_validity.oob import check_oob
from benchmark.evaluator.generic_validity.support import (
    _architecture_plane_clearances,
)
from benchmark.evaluator.generic_validity.geometry import (
    get_room_boundary,
    normalize_objects,
)
from benchmark.grouping import normalize_grouping_scene
from benchmark.nl_scene.generation_input import (
    build_generation_input,
    build_scene_request,
    build_generator_visible_payload,
)
from benchmark.rendering.blender_worker import (
    _active_wall_ids,
    _canonical_object_ids,
    _wall_id_for_edge,
)
from benchmark.rendering.camera_pose import generate_camera_pose_candidates
from benchmark.task_contract import resolve_room_contract
from benchmark.visual_judge.render_views import (
    CAMERA_EVIDENCE_CACHE_CONTRACT_VERSION,
    CameraEvidenceProvider,
    _event_key,
)


def _room() -> dict:
    return resolve_room_contract(
        {"dimensions": {"width": 7.0, "depth": 5.0, "height": 3.0}}
    )


def _contract(
    instruction: str = "",
    *,
    policy: str = "explicit_only",
    specification_contract: dict | None = None,
    reference_annotation: dict | None = None,
    object_plan: dict | None = None,
    visual_style_spec: dict | None = None,
) -> dict:
    return resolve_architecture_activation(
        _room(),
        instruction=instruction,
        physical_wall_policy=policy,
        specification_contract=specification_contract,
        reference_annotation=reference_annotation,
        object_plan=object_plan,
        visual_style_spec=visual_style_spec,
    )


def _scene(contract: dict, *, outside: bool = False) -> dict:
    return {
        "scene_id": "wall_policy",
        "scene_type": "bedroom",
        "boundary": deepcopy(_room()["boundary"]),
        "scene_height": 3.0,
        "objects": [
            {
                "id": "chair",
                "category": "chair",
                "description": "chair",
                "center": [7.1 if outside else 1.0, 1.0, 0.5],
                "size": [0.5, 0.5, 1.0],
                "rotation": [0.0, 0.0, 0.0],
            }
        ],
        "metadata": {"architecture_contract": deepcopy(contract)},
    }


def _generation_input(contract: dict) -> dict:
    scene_request = build_scene_request(
        request_id="wall_policy",
        instruction="Create a bedroom with one chair.",
        scene_type="bedroom",
        room=_room(),
        structure=True,
    )
    return build_generation_input(
        scene_request=scene_request,
        object_plan={
            "request_id": "wall_policy",
            "scene_type": "bedroom",
            "objects": [{"id": "chair", "category": "chair"}],
            "relations": [],
        },
        architecture_contract=contract,
    )


def test_default_is_logically_bounded_floor_without_physical_walls() -> None:
    contract = _contract("Create a bedroom with a bed and two lamps.")

    assert contract["architecture_policy_version"] == ARCHITECTURE_POLICY_VERSION
    assert contract["logical_boundary"] == {
        "enabled": True,
        "boundary": _room()["boundary"],
    }
    assert contract["floor"]["enabled"] is True
    assert contract["physical_walls"]["policy"] == "explicit_only"
    assert contract["physical_walls"]["active_wall_ids"] == []


def test_scene_type_and_indoor_commonsense_do_not_activate_walls() -> None:
    assert _contract("Create a cozy indoor bedroom.")["wall_count"] == 0


def test_named_wall_claim_activates_only_that_wall() -> None:
    assert _contract("Attach the painting to the north wall.")[
        "physical_walls"
    ]["active_wall_ids"] == ["north_wall"]


@pytest.mark.parametrize(
    "instruction",
    (
        "Place the bed against a wall.",
        "Mount the display on the wall.",
    ),
)
def test_generic_wall_claim_activates_all_walls(instruction: str) -> None:
    assert _contract(instruction)["physical_walls"]["active_wall_ids"] == list(
        CANONICAL_WALL_IDS
    )


def test_named_corner_activates_adjacent_walls() -> None:
    assert _contract("Put the cabinet in the northwest corner.")[
        "physical_walls"
    ]["active_wall_ids"] == ["north_wall", "west_wall"]


def test_explicit_wall_appearance_activates_all_walls() -> None:
    assert _contract(
        "Create a bedroom.",
        visual_style_spec={"wall_finish": "exposed brick walls"},
    )["physical_walls"]["active_wall_ids"] == list(CANONICAL_WALL_IDS)


def test_frozen_specification_claim_is_an_authoritative_activation_source() -> None:
    contract = _contract(
        specification_contract={
            "claims": {
                "oar": [
                    {
                        "claim_id": "oar_001",
                        "relation_type": "against_wall",
                        "architectural_element": "east_wall",
                    }
                ]
            }
        }
    )

    assert contract["physical_walls"]["active_wall_ids"] == ["east_wall"]
    assert contract["physical_walls"]["activation_sources"] == [
        "specification_contract"
    ]


def test_generated_output_is_not_an_architecture_activation_source() -> None:
    contract = _contract("Create a bedroom.")
    with pytest.raises(ValueError, match="inactive architecture"):
        require_generated_architecture_targets_active(
            [
                {
                    "family": "oar",
                    "subject_id": "chair",
                    "type": "against_wall",
                    "architectural_element": "north_wall",
                }
            ],
            contract,
        )
    assert contract["physical_walls"]["active_wall_ids"] == []


def test_generator_visible_tokens_exclude_inactive_walls() -> None:
    visible = build_generator_visible_payload(_generation_input(_contract()))

    assert visible["benchmark_environment"]["architecture"][
        "allowed_architecture_tokens"
    ] == ["floor", "ceiling"]


def test_layout_json_prompt_lists_only_active_architecture_tokens() -> None:
    method_input = build_layout_json_method_input(
        _generation_input(_contract("Use the north wall for the chair."))
    )
    user_prompt = method_input["messages"][1]["content"]

    assert '["floor","ceiling","north_wall"]' in user_prompt
    assert "south_wall" not in user_prompt


def test_unknown_policy_and_wall_id_fail_closed() -> None:
    with pytest.raises(ValueError, match="physical wall policy"):
        _contract(policy="invented")
    contract = _contract()
    contract["physical_walls"]["active_wall_ids"] = ["up_wall"]
    with pytest.raises(ValueError, match="unknown physical wall ID"):
        validate_architecture_contract(contract)


def test_always_enclosed_is_explicit_legacy_compatibility() -> None:
    contract = _contract(policy="always_enclosed")

    assert contract["physical_walls"]["active_wall_ids"] == list(
        CANONICAL_WALL_IDS
    )
    assert contract["physical_walls"]["compatibility_mode"] is True
    assert contract["wall_count"] == 4


def test_worker_wall_edge_mapping_is_canonical_and_activation_is_exact() -> None:
    boundary = _room()["boundary"]

    assert [_wall_id_for_edge(boundary, index) for index in range(4)] == [
        "south_wall",
        "east_wall",
        "north_wall",
        "west_wall",
    ]
    assert _active_wall_ids(_contract()) == []
    assert _active_wall_ids(_contract("Use only the east wall.")) == [
        "east_wall"
    ]


def test_oar_normalization_separates_logical_and_physical_segments() -> None:
    open_room = normalize_room(_scene(_contract()))
    north_room = normalize_room(_scene(_contract("Use the north wall.")))

    assert len(open_room.logical_boundary_segments) == 4
    assert open_room.wall_segments == []
    assert [wall.name for wall in north_room.wall_segments] == ["north"]


def test_oar_inactive_wall_is_structured_contract_mismatch() -> None:
    report = evaluate_oar(
        _scene(_contract()),
        [
            {
                "subject_id": "chair",
                "type": "against_wall",
                "wall": "north",
            }
        ],
    )

    assert report["status"] == "architecture_contract_mismatch"
    assert report["score"] is None
    assert report["checks"][0]["evidence"]["reason_code"] == (
        "inactive_architecture_target"
    )


def test_oar_active_wall_keeps_deterministic_geometry_behavior() -> None:
    scene = _scene(_contract("Use the west wall."))
    scene["objects"][0]["center"] = [0.25, 1.0, 0.5]
    report = evaluate_oar(
        scene,
        [{"subject_id": "chair", "type": "against_wall", "wall": "west"}],
    )

    assert report["status"] == "ok"
    assert report["checks"][0]["passed"] is True


def test_oob_is_identical_across_wall_policies() -> None:
    wall_free = check_oob(_scene(_contract(), outside=True), {"detector_only": True})
    enclosed = check_oob(
        _scene(_contract(policy="always_enclosed"), outside=True),
        {"detector_only": True},
    )

    assert wall_free["objects"] == enclosed["objects"]
    assert sum(
        bool(item.get("candidate_oob")) for item in wall_free["objects"]
    ) == sum(
        bool(item.get("candidate_oob")) for item in enclosed["objects"]
    )


def test_architecture_clearances_ignore_inactive_and_keep_active_walls() -> None:
    wall_free_scene = _scene(_contract())
    active_scene = _scene(_contract("Use the west wall."))
    for scene in (wall_free_scene, active_scene):
        scene["objects"][0]["center"] = [0.25, 1.0, 0.5]
    obj = normalize_objects(wall_free_scene)[0][0]
    boundary = np.asarray(get_room_boundary(wall_free_scene), dtype=float)

    inactive = _architecture_plane_clearances(
        wall_free_scene,
        obj,
        boundary,
        True,
    )
    active = _architecture_plane_clearances(
        active_scene,
        obj,
        boundary,
        True,
    )

    assert inactive["west"] is None
    assert active["west"] == pytest.approx(0.0)
    assert inactive["floor"] == active["floor"]


def test_camera_candidates_remain_bound_to_logical_room_not_wall_policy() -> None:
    requests = []
    for contract in (_contract(), _contract(policy="always_enclosed")):
        scene = _scene(contract)
        requests.append(
            {
                "metric": "oob",
                "event": {"object_id": "chair"},
                "scene": scene,
                "object_ids": ["chair"],
            }
        )

    wall_free = generate_camera_pose_candidates(requests[0])
    enclosed = generate_camera_pose_candidates(requests[1])

    assert wall_free == enclosed
    assert all(
        0.0 <= pose["location"][0] <= 7.0
        and 0.0 <= pose["location"][1] <= 5.0
        for pose in wall_free
    )


def test_grouping_and_identity_remain_object_only() -> None:
    scene = _scene(_contract(policy="always_enclosed"))
    normalized = normalize_grouping_scene(scene)

    assert normalized.object_ids == ("chair",)
    assert _canonical_object_ids(scene) == ["chair"]
    assert not set(CANONICAL_WALL_IDS) & set(normalized.object_ids)


class _Renderer:
    width = 128
    height = 128
    preview_width = 64
    preview_height = 64
    preview_render_engine = "BLENDER_WORKBENCH"
    preview_cycles_samples = 1


def test_camera_cache_identity_includes_policy_version_and_active_walls(
    tmp_path: Path,
) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"same-source-blend")
    providers = [
        CameraEvidenceProvider(
            renderer=_Renderer(),
            blend_file=blend,
            out_dir=tmp_path / name,
            mode="visibility_ranked",
            architecture_contract=contract,
        )
        for name, contract in (
            ("open", _contract()),
            ("north", _contract("Use the north wall.")),
        )
    ]

    assert CAMERA_EVIDENCE_CACHE_CONTRACT_VERSION.endswith("_v5")
    assert providers[0].policy_config["architecture_contract"][
        "architecture_policy_version"
    ] == ARCHITECTURE_POLICY_VERSION
    assert _event_key({"policy": providers[0].policy_config}) != _event_key(
        {"policy": providers[1].policy_config}
    )


@pytest.mark.parametrize(
    "camera_policy",
    ("fixed", "deterministic_only", "vlm_only", "deterministic_then_vlm"),
)
def test_camera_ablation_arms_receive_same_frozen_wall_contract(
    camera_policy: str,
) -> None:
    contract = _contract("Use the north wall.")
    request = {
        "camera_acquisition": {"policy": camera_policy},
        "scene": _scene(contract),
    }

    assert architecture_contract_from_scene(request["scene"]) == contract
