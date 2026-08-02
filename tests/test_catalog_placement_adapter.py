from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_scene_harness as scene_harness_module
from benchmark.adapters import get_adapter
from benchmark.adapters.catalog_placement.converter import (
    convert_catalog_placement_to_scene,
    extract_catalog_placement,
    public_slot_ids_from_generation_input,
    validate_catalog_placement,
)
from benchmark.adapters.catalog_placement.prompt import (
    build_catalog_placement_method_input,
)
from benchmark.adapters.defaults import DEFAULT_GENERATION_ADAPTER
from benchmark.api.generation import run_generate
from benchmark.io_contracts import O3_SCENE_PACKAGE
from benchmark.nl_scene.generation_input import (
    build_generation_input,
    build_scene_request,
)
from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import read_json, write_json
from scripts.run_scene_harness import run_scene_harness


def _generation_input(*, count: int = 2) -> dict:
    request = build_scene_request(
        request_id="catalog_case",
        instruction="Place two chairs, but use the frozen selected asset.",
        scene_type="room",
        room={
            "boundary": [[0, 0], [6, 0], [6, 5], [0, 5]],
            "height": 3.0,
            "unit": "meter",
        },
        structure=True,
    )
    object_plan = {
        "request_id": "catalog_case",
        "scene_type": "room",
        "scene_description": "two task-role chairs",
        "objects": [
            {
                "id": "public_chair_slot",
                "role": "seating",
                "category": "chair",
                "description": "task-side chair role",
                "count": count,
                "placement_intent": {
                    "absolute_relations": [],
                    "relative_relations": [],
                },
                "metadata": {},
            }
        ],
        "global_constraints": [],
        "relations": [],
    }
    asset_selection = {
        "request_id": "catalog_case",
        "objects": [
            {
                "object_id": "public_chair_slot",
                "object_spec": {
                    "role": "seating",
                    "category": "chair",
                    "description": "task-side chair role",
                    "estimated_size": [1.0, 1.0, 1.0],
                    "count": count,
                },
                "retrieval_query": {
                    "description": "task-side chair role",
                    "category": "chair",
                    "size_constraint": [1.0, 1.0, 1.0],
                },
                # Deliberately wrong for the task role. Generated semantics must
                # remain those of this actual frozen asset.
                "selected_asset": {
                    "jid": "wrong_lamp_asset",
                    "category": "lamp",
                    "retrieval_category": "lighting",
                    "desc": "a frozen brass floor lamp",
                    "short_desc": "brass floor lamp",
                    "size": [2.0, 4.0, 8.0],
                    "asset_ref": {
                        "source_db": "frozen_test_catalog",
                        "asset_key": "wrong_lamp_asset",
                        "mesh_uri": None,
                        "pointcloud_uri": None,
                        "metadata_uri": None,
                    },
                    "asset_proxy": {
                        "type": "canonical_catalog_bbox",
                        "bbox_center_local": [0.25, -0.5, 1.0],
                        "bbox_size": [2.0, 4.0, 8.0],
                    },
                    "metadata": {
                        "appearance": "frozen brass",
                        "interactive": False,
                    },
                },
                "candidates": [],
                "selection_action": "select",
                "selection_decision": {
                    "action": "select",
                    "selected_jid": "wrong_lamp_asset",
                    "reason": "intentional wrong-asset fixture",
                    "generation_request": None,
                },
                "selection_reason": "intentional wrong-asset fixture",
            }
        ],
    }
    return build_generation_input(
        scene_request=request,
        object_plan=object_plan,
        asset_selection=asset_selection,
        evaluator_output_type=O3_SCENE_PACKAGE,
    )


def _instance(
    instance_id: str,
    *,
    center: list[float],
    slot_id: str | None = None,
) -> dict:
    value = {
        "instance_id": instance_id,
        "asset_id": "wrong_lamp_asset",
        "center_m": center,
        "target_size_m": [1.0, 3.0, 10.0],
        "rotation_euler_xyz_deg": [0.0, 0.0, 90.0],
    }
    if slot_id is not None:
        value["slot_id"] = slot_id
    return value


def test_reorder_preserves_instance_and_evaluator_identity() -> None:
    generation_input = _generation_input()
    left = _instance("lamp_left", center=[1.0, 2.0, 1.0])
    right = _instance("lamp_right", center=[4.0, 2.0, 1.0])

    first = convert_catalog_placement_to_scene(
        {"schema_version": "catalog_placement_v1", "instances": [left, right]},
        generation_input,
    )
    reordered = convert_catalog_placement_to_scene(
        {"schema_version": "catalog_placement_v1", "instances": [right, left]},
        generation_input,
    )

    assert first == reordered
    assert [
        (item["instance_id"], item["evaluator_object_id"], item["asset_id"])
        for item in first["metadata"]["instance_registry"]["instances"]
    ] == [
        ("lamp_left", "lamp_left", "wrong_lamp_asset"),
        ("lamp_right", "lamp_right", "wrong_lamp_asset"),
    ]


def test_duplicate_asset_instances_remain_independent() -> None:
    scene = convert_catalog_placement_to_scene(
        {
            "instances": [
                _instance("copy_a", center=[1.0, 1.0, 1.0]),
                _instance("copy_b", center=[3.0, 1.0, 1.0]),
            ]
        },
        _generation_input(),
    )

    assert [item["id"] for item in scene["objects"]] == ["copy_a", "copy_b"]
    assert {item["jid"] for item in scene["objects"]} == {"wrong_lamp_asset"}


def test_only_literal_public_slot_is_allowed_and_never_repairs_wrong_asset() -> None:
    generation_input = _generation_input(count=2)
    assert public_slot_ids_from_generation_input(generation_input) == {
        "public_chair_slot"
    }
    method_input = build_catalog_placement_method_input(generation_input)
    assert method_input["public_slot_ids"] == ["public_chair_slot"]
    assert "public_chair_slot#1" not in json.dumps(method_input)

    placement = {
        "instances": [
            _instance(
                "stable_generator_id",
                center=[2.0, 2.0, 1.0],
                slot_id="public_chair_slot",
            )
        ]
    }
    scene = convert_catalog_placement_to_scene(placement, generation_input)
    obj = scene["objects"][0]
    assert obj["id"] == "stable_generator_id"
    assert obj["category"] == "lamp"
    assert obj["description"] == "a frozen brass floor lamp"
    assert obj["metadata"]["catalog_placement"]["slot_id"] == "public_chair_slot"

    hidden = {
        "instances": [
            _instance(
                "stable_generator_id",
                center=[2.0, 2.0, 1.0],
                slot_id="private_hidden_slot",
            )
        ]
    }
    with pytest.raises(ArtifactValidationError, match="generator-visible public slot"):
        convert_catalog_placement_to_scene(hidden, generation_input)


def test_contract_rejects_semantic_fields_nonfinite_and_nonpositive_sizes() -> None:
    base = {
        "instances": [_instance("one", center=[1.0, 2.0, 3.0])]
    }
    semantic = json.loads(json.dumps(base))
    semantic["instances"][0]["category"] = "chair"
    with pytest.raises(ArtifactValidationError):
        validate_catalog_placement(semantic)

    nonfinite = json.loads(json.dumps(base))
    nonfinite["instances"][0]["center_m"][0] = math.inf
    with pytest.raises(ArtifactValidationError, match="finite"):
        validate_catalog_placement(nonfinite)

    nonpositive = json.loads(json.dumps(base))
    nonpositive["instances"][0]["target_size_m"][1] = 0.0
    with pytest.raises(ArtifactValidationError):
        validate_catalog_placement(nonpositive)

    with pytest.raises(ArtifactValidationError, match="exactly one JSON object"):
        extract_catalog_placement(
            "Here is the placement:\n"
            + json.dumps(base)
        )


def test_uniform_fit_rotation_and_bounds_follow_frozen_convention() -> None:
    scene = convert_catalog_placement_to_scene(
        {
            "instances": [
                _instance("rotated", center=[10.0, 20.0, 30.0])
            ]
        },
        _generation_input(),
    )
    metadata = scene["objects"][0]["metadata"]["catalog_placement"]

    # min([1/2, 3/4, 10/8]) = 0.5; actual local size is separate
    # from the requested target envelope.
    assert metadata["uniform_scale"] == pytest.approx(0.5)
    assert metadata["actual_local_bbox_size_m"] == pytest.approx(
        [1.0, 2.0, 4.0]
    )
    expected_rotation = [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    for observed, expected in zip(
        metadata["world_obb"]["rotation_matrix_rz_ry_rx"],
        expected_rotation,
        strict=True,
    ):
        assert observed == pytest.approx(expected)
    assert metadata["world_aabb"]["size_m"] == pytest.approx([2.0, 1.0, 4.0])
    # The imported root translation maps the scaled canonical bbox center,
    # rather than the raw asset origin, to center_m.
    assert metadata["asset_root_translation_m"] == pytest.approx(
        [9.75, 19.875, 29.5]
    )


def test_adapter_preserves_raw_artifact_bytes(tmp_path: Path) -> None:
    source = tmp_path / "submitted.json"
    raw = (
        '{\n  "schema_version": "catalog_placement_v1",\n'
        '  "instances": [{"instance_id": "one", '
        '"asset_id": "wrong_lamp_asset", "center_m": [1, 2, 3], '
        '"target_size_m": [1, 3, 10], '
        '"rotation_euler_xyz_deg": [0, 0, 90]}]\n}\n'
    ).encode("utf-8")
    source.write_bytes(raw)
    output = tmp_path / "output"

    scene_path = get_adapter("catalog_placement").parse_output(
        source,
        _generation_input(count=1),
        output,
    )

    assert scene_path.is_file()
    assert (output / "catalog_placement_raw_artifact.json").read_bytes() == raw


def test_new_generation_default_is_catalog_placement() -> None:
    assert DEFAULT_GENERATION_ADAPTER == "catalog_placement"
    contract = get_adapter(DEFAULT_GENERATION_ADAPTER).resolve_io_contract(
        _generation_input()
    )
    assert contract.evaluator_output_type == O3_SCENE_PACKAGE


def test_generate_native_only_exposes_exact_raw_artifact(
    tmp_path: Path,
) -> None:
    raw_path = write_json(
        tmp_path / "native_placement.json",
        {
            "schema_version": "catalog_placement_v1",
            "instances": [
                _instance("one", center=[1.0, 2.0, 3.0])
            ],
        },
    )

    result = run_generate(
        generation_input=_generation_input(count=1),
        adapter_name="catalog_placement",
        out_dir=tmp_path / "generation",
        method_output=raw_path,
        materialize_native_output=False,
    )

    assert result["status"]["status"] == "native_output_available"
    assert result["generated_scene"] is None
    assert Path(result["raw_native_artifact"]).read_bytes() == raw_path.read_bytes()
    metadata = read_json(result["adapter_metadata"])
    assert metadata["raw_native_artifact_path"] == raw_path.resolve().as_posix()
    assert metadata["materialize_native_output"] is False


def test_active_default_harness_uses_only_trusted_prepared_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_path = write_json(
        tmp_path / "native_placement.json",
        {"instances": [_instance("one", center=[1.0, 2.0, 3.0])]},
    )
    calls: dict[str, object] = {}

    def fake_run_generate(**kwargs):
        calls["run_generate"] = kwargs
        assert kwargs["adapter_name"] == "catalog_placement"
        assert kwargs["materialize_native_output"] is False
        return {
            "adapter": "catalog_placement",
            "method_input": (tmp_path / "method_input.json").as_posix(),
            "generated_scene": None,
            "native_output": raw_path.as_posix(),
            "raw_native_artifact": raw_path.as_posix(),
            "workflow_status": (tmp_path / "workflow_status.json").as_posix(),
            "adapter_metadata": (tmp_path / "adapter_metadata.json").as_posix(),
            "status": {"status": "native_output_available"},
        }

    def fake_prepare_submission(**kwargs):
        calls["prepare"] = kwargs
        assert kwargs["artifact"] == raw_path.as_posix()
        assert (
            kwargs["generation_input"]["generation_contract"][
                "evaluator_output_type"
            ]
            == O3_SCENE_PACKAGE
        )
        prepared_root = Path(kwargs["out_dir"])
        prepared_scene = write_json(
            prepared_root / "generated_scene.json",
            {
                "schema_version": "canonical_scene_v1",
                "scene_id": "trusted_scene",
                "request_id": "trusted_case",
                "scene_type": "room",
                "boundary": [[0, 0], [6, 0], [6, 5], [0, 5]],
                "scene_height": 3.0,
                "objects": [],
                "metadata": {
                    "coordinate_frame": {
                        "origin": "room_min_corner_floor",
                        "axes": "x_width_y_depth_z_up",
                        "unit": "meter",
                        "rotation_unit": "degree",
                    }
                },
            },
        )
        return SimpleNamespace(
            normalized_scene_path=prepared_scene,
            as_dict=lambda: {
                "normalized_scene_path": prepared_scene.as_posix(),
                "trusted_render_source_path": (
                    prepared_root / "evaluation.blend"
                ).as_posix(),
            },
        )

    def fake_evaluate_prepared_submission(**kwargs):
        calls["evaluate_prepared"] = kwargs
        assert (
            kwargs["prepared_submission"].normalized_scene_path
            == Path(kwargs["out_dir"]) / "preparation" / "generated_scene.json"
        )
        assert Path(kwargs["asset_root"]).resolve() == (
            tmp_path / "assets"
        ).resolve()
        assert Path(kwargs["asset_csv"]).resolve() == (
            tmp_path / "catalog.csv"
        ).resolve()
        report = {
            "evaluation_status": "complete",
            "benchmark_score": 1.0,
            "benchmark_score_status": "complete",
            "overall_valid": True,
            "layer_reports": {},
            "reports": {},
        }
        evaluation_dir = Path(kwargs["out_dir"])
        write_json(evaluation_dir / "evaluation_report.json", report)
        render_dir = evaluation_dir / "renders"
        render_dir.mkdir(parents=True, exist_ok=True)
        overview = render_dir / "standardized_top.png"
        overview.write_bytes(b"trusted-render-evidence")
        render_manifest = write_json(
            render_dir / "prepared_render_manifest.json",
            {
                "backend": "blender_prepared_scene_read_only_v1",
                "views": [{"name": "top", "path": overview.as_posix()}],
            },
        )
        write_json(
            evaluation_dir / "submission_run_manifest.json",
            {
                "rendering": {
                    "input_policy": "benchmark_owned_sanitized_blend",
                    "input_path": (
                        evaluation_dir
                        / "evaluation_frozen_rematerialization"
                        / "evaluation.blend"
                    ).as_posix(),
                    "manifest_path": render_manifest.as_posix(),
                    "manifest_sha256": hashlib.sha256(
                        render_manifest.read_bytes()
                    ).hexdigest(),
                    "overview_views": [overview.as_posix()],
                },
                "evaluation_render_authority": {
                    "source": "fresh_frozen_catalog_rematerialization",
                    "path": (
                        evaluation_dir
                        / "evaluation_frozen_rematerialization"
                        / "evaluation.blend"
                    ).as_posix(),
                    "sha256": "a" * 64,
                },
            },
        )
        return report

    def forbidden_legacy_call(*args, **kwargs):
        raise AssertionError("legacy render/evaluate path must not be called")

    monkeypatch.setattr(scene_harness_module, "run_generate", fake_run_generate)
    monkeypatch.setattr(
        scene_harness_module,
        "prepare_submission",
        fake_prepare_submission,
    )
    monkeypatch.setattr(
        scene_harness_module,
        "evaluate_prepared_submission",
        fake_evaluate_prepared_submission,
    )
    monkeypatch.setattr(
        scene_harness_module,
        "run_evaluate",
        forbidden_legacy_call,
    )
    monkeypatch.setattr(
        scene_harness_module.BlenderRenderer,
        "render_scene",
        forbidden_legacy_call,
    )

    manifest = run_scene_harness(
        instruction="Place the selected asset.",
        scene_type="room",
        room={
            "boundary": [[0, 0], [6, 0], [6, 5], [0, 5]],
            "height": 3.0,
            "unit": "meter",
        },
        out_dir=tmp_path / "trusted_case",
        generator_structure=_generation_input(count=1)["object_plan"],
        asset_selection=_generation_input(count=1)["asset_selection"],
        asset_mode="retrieve",
        method_output=raw_path,
        case_bundle=tmp_path / "case_bundle",
        asset_root=tmp_path / "assets",
        asset_csv=tmp_path / "catalog.csv",
        blender_bin=tmp_path / "blender",
    )

    assert set(calls) == {
        "run_generate",
        "prepare",
        "evaluate_prepared",
    }
    assert (
        manifest["adapter"]["trusted_submission_route"]
        == "prepare_submission_then_evaluate_prepared_submission"
    )
    assert manifest["rendering"]["backend"] == "benchmark_owned_sanitized_blend"
    assert manifest["artifacts"]["raw_native_artifact"] == raw_path.as_posix()
    assert Path(manifest["artifacts"]["generated_scene"]).is_file()
    assert Path(manifest["artifacts"]["render_manifest"]).name == (
        "prepared_render_manifest.json"
    )
    assert manifest["artifacts"]["render_evidence"] == [
        (
            tmp_path
            / "trusted_case"
            / "renders"
            / "standardized_top.png"
        ).as_posix()
    ]
    trusted_attempt = manifest["self_reflexive"]["attempts"][0]
    assert trusted_attempt["render_manifest_sha256"] == hashlib.sha256(
        Path(trusted_attempt["render_manifest"]).read_bytes()
    ).hexdigest()
    assert trusted_attempt["evaluation_render_authority"]["source"] == (
        "fresh_frozen_catalog_rematerialization"
    )


def test_ready_but_unscored_catalog_evaluation_stays_incomplete() -> None:
    assert scene_harness_module._prepared_evaluation_attempt_status(
        {
            "evaluation_status": "incomplete",
            "benchmark_score": None,
            "benchmark_score_status": "incomplete",
            "layer_reports": {
                "l0_structural_validity": {
                    "status": "passed",
                    "readiness": {"status": "ready"},
                }
            },
        }
    ) == "prepared_evaluation_incomplete"
    assert scene_harness_module._prepared_evaluation_attempt_status(
        {
            "evaluation_status": "not_evaluable",
            "benchmark_score": None,
            "benchmark_score_status": "not_evaluable",
        }
    ) == "not_evaluable"


def test_active_catalog_harness_rejects_self_reflection_without_fallback(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="does not support self-reflection iterations",
    ):
        run_scene_harness(
            instruction="Place the selected asset.",
            scene_type="room",
            out_dir=tmp_path / "reflection",
            method_output=tmp_path / "native.json",
            iteration_limit=1,
        )
