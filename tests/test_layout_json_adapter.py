from __future__ import annotations

import json
from pathlib import Path
import urllib.request

import pytest

import benchmark.models.openai_compatible_model as openai_compatible_model_module
from benchmark.adapters import get_adapter
from benchmark.adapters.layout_json.converter import convert_layout_json_to_scene, validate_layout_json
from benchmark.nl_scene.generation_input import build_generation_input, build_scene_request
from benchmark.scene_io.validate import ArtifactValidationError, validate_generated_scene
from benchmark.utils.io import read_json
from generate import run_generate
from scripts.run_scene_harness import run_scene_harness


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _object_plan(request_id: str = "smoke") -> dict:
    return {
        "request_id": request_id,
        "scene_type": "bedroom",
        "scene_description": "A room with a red bed.",
        "objects": [
            {
                "id": "hidden_bed_id",
                "role": "",
                "category": "bed",
                "description": "red bed",
                "count": 1,
                "placement_intent": {"absolute_relations": [], "relative_relations": []},
                "metadata": {},
            }
        ],
        "global_constraints": [],
        "relations": [],
    }


def _generation_input(*, structure: bool = False, request_id: str = "smoke") -> dict:
    request = build_scene_request(
        request_id=request_id,
        instruction="A room with a red bed in the middle.",
        scene_type="bedroom",
        room={"boundary": [[0, 0], [8, 0], [8, 8], [0, 8]], "height": 2.8, "unit": "meter"},
        structure=structure,
    )
    return build_generation_input(
        scene_request=request,
        object_plan=_object_plan(request_id) if structure else None,
    )


def _asset_selection(request_id: str = "smoke", *, count: int = 1) -> dict:
    return {
        "request_id": request_id,
        "objects": [
            {
                "object_id": "hidden_bed_id",
                "object_spec": {
                    "role": "main sleeping surface",
                    "category": "bed",
                    "description": "red bed",
                    "estimated_size": [2.0, 2.0, 1.0],
                    "count": count,
                },
                "retrieval_query": {
                    "description": "red bed",
                    "category": "bed",
                    "size_constraint": [2.0, 2.0, 1.0],
                },
                "selected_asset": {
                    "jid": "retrieved_red_bed",
                    "category": "bed",
                    "retrieval_category": "bed",
                    "desc": "a tall red upholstered bed",
                    "short_desc": "red upholstered bed",
                    "size": [2.0, 2.0, 1.0],
                    "asset_ref": {
                        "source_db": "imaginarium",
                        "asset_key": "retrieved_red_bed",
                        "mesh_uri": None,
                        "pointcloud_uri": None,
                        "metadata_uri": None,
                    },
                    "asset_proxy": {
                        "type": "obb_from_metadata_or_csv",
                        "bbox_center_local": [0, 0, 0],
                        "bbox_size": [2.0, 2.0, 1.0],
                    },
                    "metadata": {
                        "interactive": False,
                        "inner_placement": False,
                        "align_to_wall_normal": False,
                        "scaling_strategy": None,
                    },
                },
                "candidates": [],
                "selection_action": "select",
                "selection_decision": {
                    "action": "select",
                    "selected_jid": "retrieved_red_bed",
                    "reason": "top semantic result",
                    "generation_request": None,
                },
                "selection_reason": "top semantic result",
            }
        ],
    }


def _asset_generation_input(*, request_id: str = "smoke", count: int = 1) -> dict:
    request = build_scene_request(
        request_id=request_id,
        instruction="A room with a red bed in the middle.",
        scene_type="bedroom",
        room={"boundary": [[0, 0], [8, 0], [8, 8], [0, 8]], "height": 2.8, "unit": "meter"},
        structure=True,
    )
    plan = _object_plan(request_id)
    plan["objects"][0]["count"] = count
    return build_generation_input(
        scene_request=request,
        object_plan=plan,
        asset_selection=_asset_selection(request_id, count=count),
    )


def _layout_json() -> dict:
    return {
        "schema_version": "layout_json_v1",
        "scene_type": "bedroom",
        "coordinate_frame": {
            "origin": "room_min_corner_floor",
            "axes": "x_width_y_depth_z_up",
            "unit": "meter",
            "rotation_unit": "degree",
        },
        "objects": [
            {
                "id": "bed_1",
                "category": "bed",
                "description": "red bed",
                "center": [4, 4, 0.5],
                "size": [2, 2, 1],
                "rotation": [0, 0, 0],
            },
            {
                "id": "drawer_1",
                "category": "drawer",
                "description": "wooden drawer",
                "center": [5.5, 4, 0.4],
                "size": [0.8, 0.6, 0.8],
                "rotation": [0, 0, 0],
            },
        ],
        "relationships": [
            {"family": "oor", "subject": "drawer_1", "predicate": "right", "object": "bed_1"}
        ],
    }


def _spatial_ontology() -> dict:
    bed_to_drawer = {"count": 40, "p_b_given_a": 0.2, "npmi": 0.1}
    drawer_to_bed = {"count": 40, "p_b_given_a": 0.8, "npmi": 0.1}
    return {
        "bed": {
            "count": 200,
            "dimensions": {
                "width_m": {"p5": 1.5, "median": 2.0, "p95": 2.4},
                "depth_m": {"p5": 1.4, "median": 1.8, "p95": 2.2},
                "height_m": {"p5": 0.4, "median": 0.7, "p95": 1.2},
                "n_width": 200,
                "n_depth": 200,
                "n_height": 200,
            },
            "cooccurrence": {"drawer": bed_to_drawer},
        },
        "drawer": {
            "count": 100,
            "dimensions": {
                "width_m": {"p5": 0.5, "median": 0.8, "p95": 1.2},
                "depth_m": {"p5": 0.4, "median": 0.6, "p95": 0.9},
                "height_m": {"p5": 0.5, "median": 0.8, "p95": 1.2},
                "n_width": 100,
                "n_depth": 100,
                "n_height": 100,
            },
            "cooccurrence": {"bed": drawer_to_bed},
        },
    }


def test_direct_layout_json_prompt_exposes_only_natural_language(tmp_path: Path) -> None:
    adapter = get_adapter("layout_json")
    method_input_path = adapter.prepare_input(_generation_input(structure=False), tmp_path)
    method_input = read_json(method_input_path)
    system_prompt = method_input["messages"][0]["content"]
    prompt = method_input["messages"][1]["content"]

    assert method_input["output_schema"] == "layout_json_v1"
    assert method_input["input_mode"] == "natural_language_direct"
    assert "A room with a red bed in the middle." in prompt
    assert "room_min_corner_floor" in system_prompt
    assert "room_center_floor" in system_prompt
    assert "x_width_y_depth_z_up" in prompt
    assert 'rotation_unit="degree"' in system_prompt
    assert '"rotation_unit":"degree"' in prompt
    assert "hidden_bed_id" not in prompt
    assert "object_plan" not in prompt


def test_asset_layout_prompt_requires_explicit_catalog_asset_ids(tmp_path: Path) -> None:
    adapter = get_adapter("layout_json")
    method_input_path = adapter.prepare_input(_asset_generation_input(), tmp_path)
    prompt = read_json(method_input_path)["messages"][1]["content"]

    assert "Match each generated object semantically" in prompt
    assert "Do not match or retrieve assets using object ids" in prompt
    assert "exact asset_id lookup only" in prompt
    assert '\"asset_id\":\"selected_asset.jid\"' in prompt
    assert '\"description\":\"red bed\"' in prompt


def test_layout_json_converter_builds_canonical_proxy_scene() -> None:
    scene = convert_layout_json_to_scene(_layout_json(), _generation_input())

    assert validate_generated_scene(scene)
    assert scene["schema_version"] == "canonical_scene_v1"
    assert scene["request_id"] == "smoke"
    assert scene["boundary"] == [[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]]
    assert "jid" not in scene["objects"][0]
    assert "asset_ref" not in scene["objects"][0]
    assert "asset_proxy" not in scene["objects"][0]
    assert scene["objects"][0]["description"] == "red bed"
    assert scene["relations"] == [
        {"family": "oor", "subject_id": "drawer_1", "type": "right", "object_id": "bed_1"}
    ]
    assert scene["metadata"]["source_coordinate_frame"]["origin"] == "room_min_corner_floor"
    assert scene["metadata"]["coordinate_frame"]["origin"] == "room_min_corner_floor"
    assert scene["metadata"]["coordinate_frame"]["rotation_unit"] == "degree"


def test_layout_json_description_is_optional_and_not_inferred() -> None:
    layout = _layout_json()
    layout["objects"] = [layout["objects"][0]]
    layout["objects"][0].pop("description")
    layout["relationships"] = []

    scene = convert_layout_json_to_scene(layout, _generation_input())

    assert "description" not in scene["objects"][0]
    assert validate_generated_scene(scene) is scene


def test_layout_json_exactly_resolves_explicit_retrieved_asset_id() -> None:
    layout = _layout_json()
    layout["objects"] = [layout["objects"][0]]
    layout["objects"][0]["asset_id"] = "retrieved_red_bed"
    layout["relationships"] = []

    scene = convert_layout_json_to_scene(layout, _asset_generation_input())

    bed = scene["objects"][0]
    assert bed["id"] == "bed_1"
    assert bed["jid"] == "retrieved_red_bed"
    assert bed["metadata"]["asset_binding"]["method"] == "explicit_asset_id_exact_lookup"
    assert bed["metadata"]["asset_binding"]["requested_asset_id"] == "retrieved_red_bed"
    assert bed["metadata"]["asset_binding"]["selection_object_ids"] == ["hidden_bed_id"]
    assert bed["asset_ref"]["asset_key"] == "retrieved_red_bed"
    assert bed["geometry_provenance"] == "asset_mesh"
    assert scene["metadata"]["asset_binding"]["bound_object_count"] == 1
    assert scene["metadata"]["asset_binding"]["unresolved_object_count"] == 0


def test_layout_json_marks_generated_selected_assets_as_generated_meshes() -> None:
    layout = _layout_json()
    layout["objects"] = [layout["objects"][0]]
    layout["objects"][0]["asset_id"] = "retrieved_red_bed"
    layout["relationships"] = []
    generation_input = _asset_generation_input()
    generation_input["asset_selection"]["objects"][0]["selected_asset"]["asset_ref"][
        "source_db"
    ] = "generated"

    scene = convert_layout_json_to_scene(layout, generation_input)

    assert scene["objects"][0]["geometry_provenance"] == "generated_mesh"


def test_layout_json_allows_explicit_reuse_of_selected_asset_for_multiple_instances() -> None:
    layout = _layout_json()
    layout["objects"] = [
        {**layout["objects"][0], "id": "model_bed_left", "description": "left red bed", "asset_id": "retrieved_red_bed"},
        {**layout["objects"][0], "id": "model_bed_right", "description": "right red bed", "asset_id": "retrieved_red_bed"},
        {**layout["objects"][0], "id": "model_bed_extra", "description": "extra red bed", "asset_id": "retrieved_red_bed"},
    ]
    layout["relationships"] = []

    scene = convert_layout_json_to_scene(layout, _asset_generation_input(count=2))

    assert [item["jid"] for item in scene["objects"]] == [
        "retrieved_red_bed",
        "retrieved_red_bed",
        "retrieved_red_bed",
    ]
    assert scene["metadata"]["asset_binding"]["selected_asset_count"] == 1
    assert scene["metadata"]["asset_binding"]["bound_object_count"] == 3


def test_layout_json_rejects_unselected_generator_asset_id() -> None:
    layout = _layout_json()
    layout["objects"][0]["asset_id"] = "invented_asset"

    layout["objects"][1]["asset_id"] = "retrieved_red_bed"

    with pytest.raises(ArtifactValidationError, match="not in the request's selected asset catalog"):
        convert_layout_json_to_scene(layout, _asset_generation_input())


def test_layout_json_rejects_missing_asset_id_in_structured_assets_mode() -> None:
    with pytest.raises(ArtifactValidationError, match="requires an explicit selected asset_id"):
        convert_layout_json_to_scene(_layout_json(), _asset_generation_input())


def test_layout_json_preserves_small_degree_rotations() -> None:
    layout = _layout_json()
    layout["objects"][0]["rotation"] = [1, 5, 3.14159]

    scene = convert_layout_json_to_scene(layout, _generation_input())

    assert scene["objects"][0]["rotation"] == [1.0, 5.0, 3.14159]


def test_layout_json_converter_canonicalizes_center_origin() -> None:
    layout = _layout_json()
    layout["coordinate_frame"]["origin"] = "room_center_floor"
    layout["objects"][0]["center"] = [0, 0, 0.5]
    layout["objects"][1]["center"] = [1.5, 0, 0.4]

    scene = convert_layout_json_to_scene(layout, _generation_input())

    assert scene["boundary"] == [[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]]
    assert scene["objects"][0]["center"] == [4.0, 4.0, 0.5]
    assert scene["objects"][1]["center"] == [5.5, 4.0, 0.4]
    assert scene["metadata"]["source_coordinate_frame"]["origin"] == "room_center_floor"
    assert scene["metadata"]["coordinate_frame"]["origin"] == "room_min_corner_floor"


def test_layout_json_conversion_renames_keys_without_changing_semantic_values() -> None:
    layout = _layout_json()
    layout["objects"][0]["description"] = "a plush crimson four-poster bed"

    scene = convert_layout_json_to_scene(layout, _generation_input())

    # Serialized field names are renamed (subject->subject_id, predicate->type,
    # object->object_id) but the semantic values are copied verbatim.
    assert scene["objects"][0]["category"] == "bed"
    assert scene["objects"][0]["description"] == "a plush crimson four-poster bed"
    assert scene["relations"] == [
        {"family": "oor", "subject_id": "drawer_1", "type": "right", "object_id": "bed_1"}
    ]


def test_layout_json_conversion_does_not_rewrite_category_values() -> None:
    layout = _layout_json()
    layout["objects"][1]["category"] = "nightstand"
    layout["objects"][1]["description"] = "bedside cabinet"

    scene = convert_layout_json_to_scene(layout, _generation_input())

    # "nightstand" must not be semantically rewritten to a synonym like "cabinet".
    nightstand = scene["objects"][1]
    assert nightstand["category"] == "nightstand"
    assert nightstand["description"] == "bedside cabinet"


def test_layout_json_conversion_keeps_object_ids_and_references_consistent() -> None:
    scene = convert_layout_json_to_scene(_layout_json(), _generation_input())

    object_ids = {obj["id"] for obj in scene["objects"]}
    assert object_ids == {"bed_1", "drawer_1"}
    # Every relation reference resolves to an object id present in the scene.
    for relation in scene["relations"]:
        assert relation["subject_id"] in object_ids
        assert relation["object_id"] in object_ids


def test_layout_json_coordinate_conversion_requires_declared_frame() -> None:
    # Conversion is allowed only when the source frame is explicitly declared.
    missing_frame = _layout_json()
    missing_frame.pop("coordinate_frame")
    with pytest.raises(ArtifactValidationError, match="coordinate_frame"):
        convert_layout_json_to_scene(missing_frame, _generation_input())

    # With a declared room_center_floor frame the coordinates convert to the
    # canonical min-corner frame; nothing is inferred.
    declared = _layout_json()
    declared["coordinate_frame"]["origin"] = "room_center_floor"
    declared["objects"][0]["center"] = [0, 0, 0.5]
    scene = convert_layout_json_to_scene(declared, _generation_input())
    assert scene["objects"][0]["center"] == [4.0, 4.0, 0.5]
    assert scene["metadata"]["source_coordinate_frame"]["origin"] == "room_center_floor"
    assert scene["metadata"]["coordinate_frame"]["origin"] == "room_min_corner_floor"


def test_layout_json_validation_rejects_missing_object_size() -> None:
    malformed = _layout_json()
    malformed["objects"][0].pop("size")

    with pytest.raises(ArtifactValidationError, match="layout_json_v1 validation failed"):
        validate_layout_json(malformed)


def test_layout_json_validation_rejects_missing_rotation() -> None:
    malformed = _layout_json()
    malformed["objects"][0].pop("rotation")

    with pytest.raises(ArtifactValidationError, match="rotation"):
        validate_layout_json(malformed)


def test_layout_json_validation_requires_explicit_relation_family() -> None:
    malformed = _layout_json()
    malformed["relationships"][0].pop("family")

    with pytest.raises(ArtifactValidationError, match="family"):
        validate_layout_json(malformed)


def test_layout_json_validation_rejects_missing_coordinate_frame() -> None:
    malformed = _layout_json()
    malformed.pop("coordinate_frame")

    with pytest.raises(ArtifactValidationError, match="coordinate_frame"):
        validate_layout_json(malformed)


def test_layout_json_validation_rejects_inconsistent_explicit_boundary_origin() -> None:
    malformed = _layout_json()
    malformed["room"] = {"boundary": [[-4, -4], [4, -4], [4, 4], [-4, 4]], "height": 2.8}

    with pytest.raises(ArtifactValidationError, match="boundary minima"):
        validate_layout_json(malformed)


def test_layout_json_adapter_calls_openai_compatible_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict = {}

    def fake_urlopen(request: urllib.request.Request, timeout: int):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"choices": [{"message": {"content": json.dumps(_layout_json())}, "finish_reason": "stop"}]})

    monkeypatch.setattr(openai_compatible_model_module, "_urlopen_no_redirect", fake_urlopen)
    result = run_generate(
        generation_input=_generation_input(),
        adapter_name="layout_json",
        out_dir=tmp_path,
        run_generation=True,
        adapter_config={
            "endpoint": "http://127.0.0.1:8298/v1",
            "model": "Qwen3-VL-32B-Instruct-64K",
            "timeout_seconds": 12,
        },
    )

    scene = read_json(result["generated_scene"])
    metadata = read_json(result["adapter_metadata"])
    assert captured["url"] == "http://127.0.0.1:8298/v1/chat/completions"
    assert captured["timeout"] == 12
    assert captured["payload"]["model"] == "Qwen3-VL-32B-Instruct-64K"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert scene["metadata"]["generator_output_schema"] == "layout_json_v1"
    assert metadata["generation_run"]["model"] == "Qwen3-VL-32B-Instruct-64K"
    assert (tmp_path / "generator" / "model_response.txt").exists()
    assert (tmp_path / "generator" / "layout_json_output.json").exists()
    assert (tmp_path / "generator" / "model_request_metadata.json").exists()


def test_layout_json_v1_raw_response_replays_with_frozen_semantics(
    tmp_path: Path,
) -> None:
    raw_response = tmp_path / "historical_model_response.txt"
    raw_response.write_text(
        json.dumps(_layout_json(), ensure_ascii=False, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    adapter = get_adapter("layout_json")

    replayed_path = adapter.materialize_output(
        raw_response,
        _generation_input(),
        tmp_path / "replay",
    )

    assert read_json(replayed_path) == convert_layout_json_to_scene(
        _layout_json(),
        _generation_input(),
    )


def test_layout_json_adapter_does_not_repair_schema_invalid_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payloads: list[dict] = []
    invalid_layout = _layout_json()
    invalid_layout["relationships"][0]["object"] = ["bed_1", "left_wall"]

    def fake_urlopen(request: urllib.request.Request, timeout: int):
        payloads.append(json.loads(request.data.decode("utf-8")))
        return _FakeResponse({"choices": [{"message": {"content": json.dumps(invalid_layout)}, "finish_reason": "stop"}]})

    monkeypatch.setattr(openai_compatible_model_module, "_urlopen_no_redirect", fake_urlopen)
    with pytest.raises(ArtifactValidationError, match="schema repair is disabled"):
        run_generate(
            generation_input=_generation_input(),
            adapter_name="layout_json",
            out_dir=tmp_path,
            run_generation=True,
            adapter_config={
                "endpoint": "http://127.0.0.1:8298/v1",
                "model": "Qwen3-VL-32B-Instruct-64K",
            },
        )

    # Zero schema-repair calls: only the single generation call was made.
    assert len(payloads) == 1
    # Raw generator output is preserved as-is for the failure.
    raw_response_path = tmp_path / "generator" / "model_response.txt"
    assert raw_response_path.is_file()
    assert json.loads(raw_response_path.read_text(encoding="utf-8")) == invalid_layout
    # No schema-repair artifacts are written.
    assert not (tmp_path / "generator" / "model_response_initial_invalid.txt").exists()
    assert not (tmp_path / "generator" / "model_response_schema_repair_01.txt").exists()


def test_layout_json_adapter_rejects_positive_schema_repair_attempts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict] = []

    def fake_urlopen(request: urllib.request.Request, timeout: int):
        calls.append(json.loads(request.data.decode("utf-8")))
        return _FakeResponse({"choices": [{"message": {"content": json.dumps(_layout_json())}, "finish_reason": "stop"}]})

    monkeypatch.setattr(openai_compatible_model_module, "_urlopen_no_redirect", fake_urlopen)
    with pytest.raises(ValueError, match="schema repair is disabled"):
        run_generate(
            generation_input=_generation_input(),
            adapter_name="layout_json",
            out_dir=tmp_path,
            run_generation=True,
            adapter_config={
                "endpoint": "http://127.0.0.1:8298/v1",
                "model": "Qwen3-VL-32B-Instruct-64K",
                "schema_repair_attempts": 1,
            },
        )

    # The adapter refuses before issuing any generation or repair model call.
    assert calls == []


def test_layout_json_adapter_rejects_literal_api_key_before_model_call(tmp_path: Path) -> None:
    adapter = get_adapter("layout_json")
    secret = "must-not-appear"

    with pytest.raises(ValueError, match="use api_key_env") as captured:
        adapter.run_generation(
            tmp_path / "missing_method_input.json",
            tmp_path,
            config={
                "endpoint": "https://api.openai.com/v1",
                "model": "gpt-test",
                "api_key": secret,
            },
        )

    assert secret not in str(captured.value)


def test_layout_json_adapter_translates_reflection_into_repair_prompt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict = {}

    def fake_urlopen(request: urllib.request.Request, timeout: int):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"choices": [{"message": {"content": json.dumps(_layout_json())}, "finish_reason": "stop"}]})

    monkeypatch.setattr(openai_compatible_model_module, "_urlopen_no_redirect", fake_urlopen)
    run_generate(
        generation_input=_generation_input(),
        adapter_name="layout_json",
        out_dir=tmp_path,
        run_generation=True,
        adapter_config={"endpoint": "http://127.0.0.1:8298/v1", "model": "Qwen3-VL-32B-Instruct-64K"},
        evaluation_report={"benchmark_score": 0.5, "reports": {"generic_validity": {"score": 0.5}}},
        previous_generated_scene={"scene_id": "previous", "objects": []},
        iteration=1,
    )

    prompt = captured["payload"]["messages"][1]["content"]
    assert "repair_context" in prompt
    assert "previous_generated_scene" in prompt
    assert "previous_evaluation" in prompt


def test_layout_json_runs_through_scene_harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = 0

    def fake_urlopen(request: urllib.request.Request, timeout: int):
        nonlocal calls
        calls += 1
        return _FakeResponse({"choices": [{"message": {"content": json.dumps(_layout_json())}, "finish_reason": "stop"}]})

    monkeypatch.setattr(openai_compatible_model_module, "_urlopen_no_redirect", fake_urlopen)
    out_dir = tmp_path / "harness_smoke"
    manifest = run_scene_harness(
        instruction="A room with a red bed in the middle.",
        scene_type="bedroom",
        structure=False,
        prompt_granularity="coarse_grained",
        asset_mode="off",
        adapter="layout_json",
        adapter_config={"endpoint": "http://127.0.0.1:8298/v1", "model": "Qwen3-VL-32B-Instruct-64K"},
        run_generation=True,
        out_dir=out_dir,
    )

    assert manifest["status"] == "generated_scene_available"
    assert manifest["adapter"]["generator_output_schema"] == "layout_json_v1"
    assert manifest["asset_resolution"]["mode"] == "off"
    assert manifest["prompt_granularity"]["resolved"] == "coarse_grained"
    assert manifest["evaluation"]["gate"] == {
        "workflow": "canonical_l0_l4",
        "prompt_granularity": "coarse_grained",
        "prompt_granularity_role": "metadata_only",
        "activation_source": "canonical_profile_plus_specification_contract",
        "active_layers": [
            "l0_structural_validity",
            "l1_physical_plausibility",
            "l2_specification_fidelity",
            "l3_scene_quality",
            "l4_downstream_task_functionality",
        ],
    }
    assert manifest["prompt_granularity"]["classifier_called"] is False
    assert manifest["converter"]["called"] is False
    assert manifest["converter"]["model"] is None
    assert calls == 1
    assert manifest["artifacts"]["generator_structure"] is None
    assert not (out_dir / "generator_structure.json").exists()
    assert read_json(out_dir / "generation_input.json")["generation_contract"]["input_mode"] == "natural_language_direct"
    assert read_json(out_dir / "generated_scene.json")["metadata"]["output_adapter"] == "layout_json"
    report = read_json(out_dir / "evaluation_report.json")
    assert set(report["category_reports"]) == {
        "l0_structural_validity",
        "l1_physical_plausibility",
        "l2_specification_fidelity",
        "l3_scene_quality",
        "l4_downstream_task_functionality",
    }
    assert report["evaluator_version"] == "scene_harness_evaluator_v2"


def test_scene_harness_rejects_retired_non_game_spatial_ontology(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: int):
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {"content": json.dumps(_layout_json())},
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    monkeypatch.setattr(
        openai_compatible_model_module,
        "_urlopen_no_redirect",
        fake_urlopen,
    )

    with pytest.raises(
        ValueError,
        match="spatial_fidelity_ontology belongs to the retired non-game workflow",
    ):
        run_scene_harness(
            instruction="Create a cozy bedroom.",
            scene_type="bedroom",
            structure=False,
            prompt_granularity="coarse_grained",
            asset_mode="off",
            adapter="layout_json",
            adapter_config={
                "endpoint": "http://127.0.0.1:8298/v1",
                "model": "Qwen3-VL-32B-Instruct-64K",
            },
            run_generation=True,
            spatial_fidelity_ontology=_spatial_ontology(),
            out_dir=tmp_path / "retired_ontology",
        )


def test_fine_grained_scene_harness_does_not_run_runtime_converter_or_classifier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = 0

    def fake_urlopen(request: urllib.request.Request, timeout: int):
        nonlocal calls
        calls += 1
        return _FakeResponse({"choices": [{"message": {"content": json.dumps(_layout_json())}, "finish_reason": "stop"}]})

    monkeypatch.setattr(openai_compatible_model_module, "_urlopen_no_redirect", fake_urlopen)
    out_dir = tmp_path / "harness_smoke"
    manifest = run_scene_harness(
        instruction="Put a blue velvet bed in the center, with a desk to its right.",
        scene_type="bedroom",
        structure=False,
        asset_mode="off",
        adapter="layout_json",
        adapter_config={"endpoint": "http://127.0.0.1:8298/v1", "model": "Qwen3-VL-32B-Instruct-64K"},
        run_generation=True,
        out_dir=out_dir,
    )

    assert manifest["status"] == "generated_scene_available"
    assert manifest["adapter"]["generator_output_schema"] == "layout_json_v1"
    assert manifest["asset_resolution"]["mode"] == "off"
    assert manifest["prompt_granularity"]["resolved"] == "fine_grained"
    assert manifest["prompt_granularity"]["classifier_called"] is False
    assert manifest["converter"] == {
        "called": False,
        "runtime_allowed": False,
        "role": "offline_reference_annotation_authoring_only",
        "endpoint": None,
        "model": None,
    }
    assert calls == 1
    assert read_json(out_dir / "generation_input.json")["generation_contract"]["input_mode"] == "natural_language_direct"
    assert read_json(out_dir / "generated_scene.json")["metadata"]["output_adapter"] == "layout_json"
    assert (out_dir / "evaluation_report.json").exists()


def test_harness_renders_current_scene_for_canonical_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: int):
        return _FakeResponse({"choices": [{"message": {"content": json.dumps(_layout_json())}, "finish_reason": "stop"}]})

    render_calls = []

    class FakeRenderer:
        def __init__(self, **config) -> None:
            self.config = config

        def render_scene(self, *, scene_path, out_dir, asset_root=None) -> dict:
            render_calls.append({"scene_path": str(scene_path), "out_dir": str(out_dir), "asset_root": asset_root})
            render_dir = Path(out_dir)
            render_dir.mkdir(parents=True, exist_ok=True)
            views = []
            for name in ["top", "perspective"]:
                path = render_dir / f"standardized_{name}.png"
                path.write_bytes(b"png")
                views.append({"name": name, "path": str(path)})
            (render_dir / "render_manifest.json").write_text(json.dumps({"views": views}), encoding="utf-8")
            return {"views": views}

    judge_calls = []

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return {"score": 0.8}

    monkeypatch.setattr(openai_compatible_model_module, "_urlopen_no_redirect", fake_urlopen)
    monkeypatch.setattr("scripts.run_scene_harness.BlenderRenderer", FakeRenderer)
    out_dir = tmp_path / "blender_vlm"

    manifest = run_scene_harness(
        instruction="Create a cozy bedroom.",
        scene_type="bedroom",
        structure=False,
        prompt_granularity="coarse_grained",
        asset_mode="off",
        adapter="layout_json",
        adapter_config={"endpoint": "http://127.0.0.1:8298/v1", "model": "Qwen3-VL-32B-Instruct-64K"},
        run_generation=True,
        evaluator_vlm_judge=judge,
        blender_bin="/mnt/group/cmh/tools/blender/blender",
        out_dir=out_dir,
    )

    assert len(render_calls) == 1
    assert render_calls[0]["scene_path"].endswith("generated_scene.json")
    assert manifest["rendering"]["enabled"] is True
    assert manifest["artifacts"]["render_manifest"].endswith("renders/render_manifest.json")
    assert len(manifest["artifacts"]["render_evidence"]) == 2
    # No canonical L2 contract or L3 asset-policy applicability was supplied,
    # so rendering is available evidence but cannot silently activate a judge.
    assert judge_calls == []
    report = read_json(out_dir / "evaluation_report.json")
    assert report["evaluator_version"] == "scene_harness_evaluator_v2"
    assert report["benchmark_score_status"] == "insufficient_metric_coverage"
