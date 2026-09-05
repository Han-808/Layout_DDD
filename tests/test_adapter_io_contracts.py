from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.adapters import (
    OUTPUT_CONVERTER,
    OUTPUT_LOADER,
    AdapterRegistry,
    GenerationAdapter,
    OutputMaterializationRequired,
    SceneOutputRoute,
    get_adapter,
    list_adapters,
)
from benchmark.io_contracts import (
    I1_NATURAL_LANGUAGE,
    I2_NATURAL_LANGUAGE_STRUCTURE,
    O1_OBJECT_STATE,
    O2_SCENE_PROGRAM,
    O3_SCENE_PACKAGE,
)
from benchmark.nl_scene.generation_input import build_generation_input, build_scene_request
from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import load_yaml, read_json, write_json
from benchmark.vlm_assistance import VLM_BUDGET_FIELDS, budget_for_output
from generate import run_generate
from scripts.run_scene_harness import run_scene_harness


@pytest.mark.parametrize(
    ("adapter_name", "structure", "evaluator_output_type", "expected_input", "expected_native"),
    [
        ("layout_json", False, O1_OBJECT_STATE, I1_NATURAL_LANGUAGE, O1_OBJECT_STATE),
        ("layout_json", True, O1_OBJECT_STATE, I2_NATURAL_LANGUAGE_STRUCTURE, O1_OBJECT_STATE),
        ("scene_program", False, O1_OBJECT_STATE, I1_NATURAL_LANGUAGE, O2_SCENE_PROGRAM),
        ("scene_program", True, O1_OBJECT_STATE, I2_NATURAL_LANGUAGE_STRUCTURE, O2_SCENE_PROGRAM),
        ("scene_package", False, O3_SCENE_PACKAGE, I1_NATURAL_LANGUAGE, O3_SCENE_PACKAGE),
        ("scene_package", True, O3_SCENE_PACKAGE, I2_NATURAL_LANGUAGE_STRUCTURE, O3_SCENE_PACKAGE),
    ],
)
def test_six_i1_i2_o1_o2_o3_combinations_are_declared(
    adapter_name: str,
    structure: bool,
    evaluator_output_type: str,
    expected_input: str,
    expected_native: str,
) -> None:
    adapter = get_adapter(adapter_name)
    generation_input = _generation_input(
        structure=structure,
        evaluator_output_type=evaluator_output_type,
    )

    contract = adapter.resolve_io_contract(generation_input)

    assert contract.input_type == expected_input
    assert contract.native_output_type == expected_native
    assert contract.evaluator_output_type == evaluator_output_type


def test_current_registry_contains_no_legacy_adapter_aliases() -> None:
    assert list_adapters() == [
        "catalog_placement",
        "direct_layout",
        "holodeck",
        "layout_gpt",
        "layout_json",
        "layout_vlm",
        "object_state",
        "respace",
        "scene_package",
        "scene_program",
        "scene_smith",
        "scene_weaver",
    ]
    with pytest.raises(KeyError):
        get_adapter("manual")
    with pytest.raises(KeyError):
        get_adapter("passthrough")


def test_builtin_adapters_declare_loader_or_converter_route() -> None:
    expected = {
        "catalog_placement": OUTPUT_CONVERTER,
        "direct_layout": OUTPUT_CONVERTER,
        "holodeck": OUTPUT_CONVERTER,
        "layout_json": OUTPUT_CONVERTER,
        "layout_gpt": OUTPUT_CONVERTER,
        "layout_vlm": OUTPUT_CONVERTER,
        "object_state": OUTPUT_LOADER,
        "respace": OUTPUT_CONVERTER,
        "scene_package": OUTPUT_LOADER,
        "scene_program": OUTPUT_LOADER,
        "scene_smith": OUTPUT_CONVERTER,
        "scene_weaver": OUTPUT_CONVERTER,
    }

    assert {
        name: get_adapter(name).scene_output_route().kind
        for name in list_adapters()
    } == expected


@pytest.mark.parametrize(
    ("route_factory", "expected_kind"),
    [
        (SceneOutputRoute.existing_loader, OUTPUT_LOADER),
        (SceneOutputRoute.converter, OUTPUT_CONVERTER),
    ],
)
def test_scene_output_route_calls_exactly_one_selected_handler(
    route_factory,
    expected_kind: str,
    tmp_path: Path,
) -> None:
    calls: list[tuple] = []

    def handler(source_path, generation_input, out_dir, config):
        calls.append((source_path, generation_input, out_dir, config))
        return out_dir / "generated_scene.json"

    route = route_factory(handler)
    result = route.materialize(
        tmp_path / "native.json",
        {"request_id": "request-1"},
        tmp_path / "out",
        {"adapter_option": True},
    )

    assert route.kind == expected_kind
    assert result == tmp_path / "out" / "generated_scene.json"
    assert calls == [
        (
            tmp_path / "native.json",
            {"request_id": "request-1"},
            tmp_path / "out",
            {"adapter_option": True},
        )
    ]


def test_external_harness_adapter_factory_is_pluggable() -> None:
    class HarnessXAdapter(GenerationAdapter):
        name = "harness_x"
        output_ingestion_kind = OUTPUT_CONVERTER

        def convert_output(self, method_output_path, generation_input, out_dir, config=None):
            raise AssertionError("the structural registry must not run the converter")

    registry = AdapterRegistry()
    registry.register("harness_x", HarnessXAdapter)

    first = registry.create("harness_x")
    second = registry.create("HARNESS_X")

    assert isinstance(first, HarnessXAdapter)
    assert isinstance(second, HarnessXAdapter)
    assert first is not second
    assert registry.names() == ["harness_x"]


def test_external_harness_adapter_must_choose_one_output_route() -> None:
    class IncompleteHarnessAdapter(GenerationAdapter):
        name = "incomplete_harness"

    registry = AdapterRegistry({"incomplete_harness": IncompleteHarnessAdapter})

    with pytest.raises(ValueError, match="output_ingestion_kind"):
        registry.create("incomplete_harness")


@pytest.mark.parametrize("adapter_name", ["scene_program", "scene_package"])
def test_o2_o3_adapters_declare_future_vlm_assistance(adapter_name: str, tmp_path: Path) -> None:
    adapter = get_adapter(adapter_name)
    evaluator_output_type = O1_OBJECT_STATE if adapter_name == "scene_program" else O3_SCENE_PACKAGE
    generation_input = _generation_input(structure=False, evaluator_output_type=evaluator_output_type)

    method_input = read_json(adapter.prepare_input(generation_input, tmp_path / adapter_name))

    assert adapter.capabilities.vlm_assistance_stages
    assert method_input["vlm_assistance"]["supported"] is True
    assert method_input["vlm_assistance"]["enabled"] is False
    assert method_input["vlm_assistance"]["status"] == "disabled_by_budget"
    assert set(method_input["vlm_assistance"]["budget"]) == set(VLM_BUDGET_FIELDS)
    assert set(method_input["vlm_assistance"]["budget"].values()) == {0}


def test_checked_in_o2_o3_vlm_budgets_are_zero() -> None:
    config = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "vlm_assistance_budget.yaml")

    for output_type in (O2_SCENE_PROGRAM, O3_SCENE_PACKAGE):
        budget = budget_for_output(config, output_type)
        assert budget.enabled is False
        assert set(budget.as_dict().values()) == {0}


@pytest.mark.parametrize("adapter_name", ["scene_program", "scene_package"])
def test_positive_vlm_budget_requires_explicit_handler(adapter_name: str) -> None:
    adapter = get_adapter(adapter_name)

    with pytest.raises(ValueError, match="no config.vlm_assistant handler"):
        adapter.resolve_vlm_assistance({"vlm_budget": {"max_calls": 1}})

    assistance = adapter.resolve_vlm_assistance(
        {"vlm_budget": {"max_calls": 1}, "vlm_assistant": object()}
    )
    assert assistance["enabled"] is True
    assert assistance["status"] == "configured"


@pytest.mark.parametrize(
    ("adapter_name", "evaluator_output_type"),
    [
        ("layout_json", O1_OBJECT_STATE),
        ("object_state", O1_OBJECT_STATE),
        ("scene_program", O1_OBJECT_STATE),
        ("scene_package", O3_SCENE_PACKAGE),
    ],
)
def test_i1_method_payload_does_not_leak_benchmark_structure(
    adapter_name: str,
    evaluator_output_type: str,
    tmp_path: Path,
) -> None:
    generation_input = _generation_input(structure=False, evaluator_output_type=evaluator_output_type)
    assert "object_plan" not in generation_input
    assert "reference_annotation" not in generation_input

    method_input = read_json(get_adapter(adapter_name).prepare_input(generation_input, tmp_path / adapter_name))
    model_visible = method_input["messages"] if adapter_name == "layout_json" else method_input["generator_input"]
    serialized = json.dumps(model_visible, sort_keys=True)

    assert "evaluation_context" not in serialized
    assert "io_contract_case" not in serialized
    if isinstance(model_visible, dict):
        assert "structure" not in model_visible


@pytest.mark.parametrize("private_key", ["reference_annotation", "evaluation_context"])
def test_generation_input_rejects_evaluator_private_fields(private_key: str, tmp_path: Path) -> None:
    generation_input = _generation_input(structure=False, evaluator_output_type=O1_OBJECT_STATE)
    generation_input[private_key] = {"private_marker": "HIDDEN_EVAL_SENTINEL"}

    with pytest.raises(ArtifactValidationError, match=private_key):
        get_adapter("object_state").prepare_input(generation_input, tmp_path / private_key)


def test_harness_rejects_runtime_converter_and_auto_granularity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="runtime NL conversion is disabled"):
        run_scene_harness(
            instruction="Place a chair near the wall.",
            scene_type="room",
            out_dir=tmp_path / "runtime_converter",
            converter_model_config={"endpoint": "http://127.0.0.1:8298/v1"},
        )

    with pytest.raises(ValueError, match="runtime auto classification is disabled"):
        run_scene_harness(
            instruction="Place a chair near the wall.",
            scene_type="room",
            out_dir=tmp_path / "auto_granularity",
            prompt_granularity="auto",
        )


def test_scene_program_materializes_through_external_export(tmp_path: Path) -> None:
    adapter = get_adapter("scene_program")
    generation_input = _generation_input(structure=False, evaluator_output_type=O1_OBJECT_STATE)
    program_path = tmp_path / "scene.py"
    program_path.write_text("# concrete adapter executes this later\n", encoding="utf-8")
    exported_path = write_json(tmp_path / "exported_scene.json", _scene())

    generated_path = adapter.materialize_output(
        program_path,
        generation_input,
        tmp_path / "out",
        config={"exported_scene_path": exported_path.as_posix()},
    )

    generated = read_json(generated_path)
    assert generated["metadata"]["native_output_type"] == O2_SCENE_PROGRAM
    assert adapter.last_materialization_metadata["requires_execution"] is True
    assert adapter.last_materialization_metadata["output_ingestion_kind"] == OUTPUT_LOADER
    assert adapter.last_materialization_metadata["executed_output_path"] == exported_path.as_posix()


def test_generation_dispatcher_accepts_native_o2_method_output(tmp_path: Path) -> None:
    generation_input = _generation_input(structure=True, evaluator_output_type=O1_OBJECT_STATE)
    program_path = tmp_path / "scene.py"
    program_path.write_text("# generated Blender program\n", encoding="utf-8")
    exported_path = write_json(tmp_path / "exported_scene.json", _scene())

    result = run_generate(
        generation_input=generation_input,
        adapter_name="scene_program",
        out_dir=tmp_path / "run",
        method_output=program_path,
        adapter_config={"exported_scene_path": exported_path.as_posix()},
    )

    assert result["status"]["status"] == "generated_scene_available"
    metadata = read_json(result["adapter_metadata"])
    assert metadata["output_ingestion_kind"] == OUTPUT_LOADER
    assert metadata["io_contract"]["native_output_type"] == O2_SCENE_PROGRAM
    assert metadata["materialization"]["canonical_output_path"] == result["generated_scene"]


def test_generic_adapter_does_not_leak_structure_to_i1(tmp_path: Path) -> None:
    adapter = get_adapter("scene_program")
    generation_input = _generation_input(structure=False, evaluator_output_type=O1_OBJECT_STATE)

    method_input = read_json(adapter.prepare_input(generation_input, tmp_path))

    visible = method_input["generator_input"]
    assert visible["natural_language"] == "Place a chair near the wall."
    assert "structure" not in visible
    assert "object_plan" not in visible


def test_generic_adapter_exposes_frozen_structure_to_i2(tmp_path: Path) -> None:
    adapter = get_adapter("scene_program")
    generation_input = _generation_input(structure=True, evaluator_output_type=O1_OBJECT_STATE)

    method_input = read_json(adapter.prepare_input(generation_input, tmp_path))

    visible = method_input["generator_input"]
    assert visible["structure"]["object_plan"]["objects"][0]["id"] == "chair"
    assert visible["structure"]["room"]["height"] == 3.0
    assert "evaluation_context" not in visible


def test_scene_program_executor_receives_only_generator_visible_input(tmp_path: Path) -> None:
    adapter = get_adapter("scene_program")
    generation_input = _generation_input(structure=False, evaluator_output_type=O1_OBJECT_STATE)
    assert "object_plan" not in generation_input
    program_path = tmp_path / "scene.py"
    program_path.write_text("# generated program\n", encoding="utf-8")
    exported_path = write_json(tmp_path / "exported_scene.json", _scene())
    received: dict = {}

    def executor(**kwargs):
        received.update(kwargs)
        return exported_path

    adapter.materialize_output(
        program_path,
        generation_input,
        tmp_path / "out",
        config={"executor": executor},
    )

    assert "generation_input" not in received
    assert received["generator_input"]["natural_language"] == "Place a chair near the wall."
    assert "reference_annotation" not in json.dumps(received["generator_input"])
    assert received["vlm_assistance"]["status"] == "disabled_by_budget"
    assert received["vlm_assistant"] is None


def test_scene_program_without_executor_writes_handoff(tmp_path: Path) -> None:
    adapter = get_adapter("scene_program")
    generation_input = _generation_input(structure=True, evaluator_output_type=O1_OBJECT_STATE)
    program_path = tmp_path / "scene.dsl"
    program_path.write_text("room {}\n", encoding="utf-8")

    with pytest.raises(OutputMaterializationRequired, match="executor"):
        adapter.materialize_output(program_path, generation_input, tmp_path)

    request = read_json(tmp_path / "execution_request.json")
    assert request["status"] == "executor_required"
    assert request["evaluator_output_type"] == O1_OBJECT_STATE


def test_scene_package_enforces_fixed_catalog_when_supplied(tmp_path: Path) -> None:
    adapter = get_adapter("scene_package")
    generation_input = _generation_input(structure=True, evaluator_output_type=O3_SCENE_PACKAGE)
    package_path = write_json(tmp_path / "scene_package.json", {"scene": _scene()})

    generated_path = adapter.materialize_output(
        package_path,
        generation_input,
        tmp_path / "accepted",
        config={"allowed_asset_ids": ["chair_asset"]},
    )
    assert read_json(generated_path)["metadata"]["native_output_type"] == O3_SCENE_PACKAGE

    with pytest.raises(ArtifactValidationError, match="outside the fixed catalog"):
        adapter.materialize_output(
            package_path,
            generation_input,
            tmp_path / "rejected",
            config={"allowed_asset_ids": ["another_asset"]},
        )


def test_scene_package_official_mode_requires_snapshot_and_nonempty_catalog(tmp_path: Path) -> None:
    adapter = get_adapter("scene_package")
    generation_input = _generation_input(structure=True, evaluator_output_type=O3_SCENE_PACKAGE)
    package_path = write_json(tmp_path / "scene_package.json", {"scene": _scene()})

    with pytest.raises(ValueError, match="catalog_snapshot_id"):
        adapter.materialize_output(
            package_path,
            generation_input,
            tmp_path / "missing_snapshot",
            config={"official_mode": True, "allowed_asset_ids": ["chair_asset"]},
        )

    with pytest.raises(ArtifactValidationError, match="non-empty fixed-catalog"):
        adapter.materialize_output(
            package_path,
            generation_input,
            tmp_path / "missing_allowlist",
            config={"official_mode": True, "catalog_snapshot_id": "catalog-2026-07"},
        )

    generated_path = adapter.materialize_output(
        package_path,
        generation_input,
        tmp_path / "official",
        config={
            "official_mode": True,
            "catalog_snapshot_id": "catalog-2026-07",
            "allowed_asset_ids": ["chair_asset"],
        },
    )
    metadata = read_json(generated_path)["metadata"]
    assert metadata["asset_catalog_snapshot_id"] == "catalog-2026-07"
    assert metadata["fixed_catalog_enforced"] is True


def test_harness_writes_private_artifacts_only_after_generator_returns(tmp_path: Path) -> None:
    out_dir = tmp_path / "isolated_harness"
    observed: dict[str, bool] = {}

    def runner(*, method_input_path: Path, out_dir: Path, config: dict) -> Path:
        del config
        benchmark_dir = out_dir.parent
        for name in [
            "scene_request.json",
            "generator_structure.json",
            "reference_annotation.json",
            "asset_selection.json",
            "generation_input.json",
        ]:
            observed[name] = (benchmark_dir / name).exists()
        observed["private_marker_in_method_input"] = (
            "HIDDEN_EVAL_SENTINEL" in method_input_path.read_text(encoding="utf-8")
        )
        native_scene = _scene()
        native_scene.pop("request_id")
        native_scene.pop("scene_id")
        return write_json(out_dir / "native_object_state.json", native_scene)

    manifest = run_scene_harness(
        instruction="Place a chair near the wall.",
        scene_type="room",
        out_dir=out_dir,
        room={"boundary": [[0, 0], [7, 0], [7, 5], [0, 5]], "height": 3.0},
        structure=False,
        reference_annotation=_reference_annotation(out_dir.name),
        adapter="object_state",
        adapter_config={"runner": runner},
        run_generation=True,
    )

    assert observed == {
        "scene_request.json": False,
        "generator_structure.json": False,
        "reference_annotation.json": False,
        "asset_selection.json": False,
        "generation_input.json": False,
        "private_marker_in_method_input": False,
    }
    generated_scene = read_json(out_dir / "generated_scene.json")
    assert generated_scene["request_id"] == "isolated_harness"
    assert generated_scene["scene_id"] == "generated_isolated_harness"
    assert not (out_dir / "generator_structure.json").exists()
    assert (out_dir / "reference_annotation.json").exists()
    assert manifest["data_isolation"]["benchmark_private_artifacts_written_after_generation"] is True


def test_harness_records_zero_vlm_budget_for_o3(tmp_path: Path) -> None:
    budget_config = load_yaml(
        Path(__file__).resolve().parents[1] / "configs" / "vlm_assistance_budget.yaml"
    )

    manifest = run_scene_harness(
        instruction="Place a chair near the wall.",
        scene_type="room",
        out_dir=tmp_path / "o3_budget",
        room={"boundary": [[0, 0], [7, 0], [7, 5], [0, 5]], "height": 3.0},
        structure=False,
        adapter="scene_package",
        evaluator_output_type=O3_SCENE_PACKAGE,
        vlm_budget_config=budget_config,
    )

    assert manifest["status"] == "generation_skipped"
    assert manifest["vlm_assistance"]["supported"] is True
    assert manifest["vlm_assistance"]["enabled"] is False
    assert set(manifest["vlm_assistance"]["budget"].values()) == {0}


def _generation_input(*, structure: bool, evaluator_output_type: str) -> dict:
    request = build_scene_request(
        request_id="io_contract_case",
        instruction="Place a chair near the wall.",
        scene_type="room",
        room={"boundary": [[0, 0], [7, 0], [7, 5], [0, 5]], "height": 3.0},
        structure=structure,
    )
    plan = {
        "request_id": "io_contract_case",
        "scene_type": "room",
        "scene_description": "Place a chair near the wall.",
        "objects": [
            {
                "id": "chair",
                "category": "chair",
                "description": "chair",
                "metadata": {},
                "placement_intent": {"absolute_relations": [], "relative_relations": []},
            }
        ],
        "global_constraints": [],
        "relations": [],
    }
    return build_generation_input(
        scene_request=request,
        object_plan=plan if structure else None,
        evaluator_output_type=evaluator_output_type,
    )


def _reference_annotation(request_id: str) -> dict:
    return {
        "annotation_version": "reference_annotation_v1",
        "validation_status": "confirmed",
        "source": "manual",
        "request_id": request_id,
        "scene_type": "room",
        "inventory_policy": "open_world",
        "objects": [
            {
                "id": "chair",
                "category": "chair",
                "description": "chair",
                "count": 1,
                "claim_state": "confirmed",
            }
        ],
        "oor_relations": [],
        "oar_relations": [],
        "room_constraints": {"claim_state": "not_mentioned"},
        "provenance": {"private_marker": "HIDDEN_EVAL_SENTINEL"},
    }


def _scene() -> dict:
    return {
        "scene_id": "io_contract_scene",
        "request_id": "io_contract_case",
        "scene_type": "room",
        "boundary": [[0, 0], [7, 0], [7, 5], [0, 5]],
        "scene_height": 3.0,
        "objects": [
            {
                "id": "chair",
                "jid": "chair_asset",
                "category": "chair",
                "description": "chair",
                "size": [0.7, 0.7, 1.0],
                "center": [2.0, 2.0, 0.5],
                "rotation": [0.0, 0.0, 0.0],
                "asset_ref": {"source_db": "imaginarium", "asset_key": "chair_asset"},
                "asset_proxy": {
                    "type": "obb_from_metadata",
                    "bbox_center_local": [0.0, 0.0, 0.0],
                    "bbox_size": [0.7, 0.7, 1.0],
                },
                "metadata": {"interactive": False},
            }
        ],
        "metadata": {
            "coordinate_frame": {
                "origin": "room_min_corner_floor",
                "axes": "x_width_y_depth_z_up",
                "unit": "meter",
                "rotation_unit": "degree",
            }
        },
    }
