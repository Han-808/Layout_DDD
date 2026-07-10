from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark.adapters import get_adapter
from benchmark.scene_io.validate import (
    validate_asset_selection,
    validate_generated_scene,
    validate_generation_input,
    validate_object_plan,
    validate_scene_request,
)
from benchmark.utils.io import read_json, write_json
from evaluate import run_evaluate
from generate import run_generate, run_generate_from_natural_language
from benchmark.nl_scene.generation_input import (
    build_direct_natural_language_generation_input,
    build_generation_input,
)


ROOT = Path(__file__).resolve().parents[1]


def _scene_request() -> dict:
    return {
        "request_id": "demo_001",
        "instruction": "Create a cozy living room.",
        "scene_type": "living room",
        "room": {"boundary": [[0, 0], [4, 0], [4, 3], [0, 3]], "height": 2.8, "unit": "meter"},
        "metadata": {},
    }


def _object_plan() -> dict:
    return {
        "request_id": "demo_001",
        "scene_type": "living room",
        "scene_description": "A cozy living room.",
        "objects": [
            {
                "id": "obj_000",
                "role": "main seating",
                "category": "sofa",
                "description": "comfortable sofa",
                "estimated_size": [2.0, 0.8, 0.8],
                "count": 1,
                "placement_intent": {"absolute_relations": [], "relative_relations": []},
                "metadata": {},
            }
        ],
        "global_constraints": ["walkable"],
        "relations": [],
    }


def _asset_selection() -> dict:
    return {
        "request_id": "demo_001",
        "objects": [
            {
                "object_id": "obj_000",
                "object_spec": {"category": "sofa", "description": "comfortable sofa", "estimated_size": [2.0, 0.8, 0.8]},
                "selected_asset": {
                    "jid": "sofa_asset",
                    "category": "sofa",
                    "retrieval_category": "sofa",
                    "desc": "A comfortable sofa",
                    "short_desc": "comfortable sofa",
                    "size": [2.0, 0.8, 0.8],
                    "asset_ref": {"source_db": "imaginarium", "asset_key": "sofa_asset", "mesh_uri": None, "pointcloud_uri": None, "metadata_uri": None},
                    "asset_proxy": {"type": "obb_from_metadata_or_csv", "bbox_center_local": [0, 0, 0], "bbox_size": [2.0, 0.8, 0.8]},
                    "metadata": {"interactive": False, "inner_placement": False, "align_to_wall_normal": False, "scaling_strategy": None},
                },
                "candidates": [],
                "selection_reason": "top-1 retrieval result",
            }
        ],
    }


def _generation_input(request_id: str = "demo_001") -> dict:
    request = {**_scene_request(), "request_id": request_id}
    plan = {**_object_plan(), "request_id": request_id}
    selection = {**_asset_selection(), "request_id": request_id}
    return {
        "request_id": request_id,
        "scene_request": request,
        "object_plan": plan,
        "asset_selection": selection,
        "generation_contract": {"output_format": "canonical_generated_scene_v1", "requires_pose": True},
    }


def _generated_scene(request_id: str = "demo_001") -> dict:
    return {
        "scene_id": f"generated_{request_id}",
        "request_id": request_id,
        "scene_type": "living room",
        "boundary": [[0, 0], [4, 0], [4, 3], [0, 3]],
        "scene_height": 2.8,
        "objects": [
            {
                "id": "obj_000",
                "jid": "sofa_asset",
                "category": "sofa",
                "retrieval_category": "sofa",
                "desc": "A comfortable sofa",
                "short_desc": "comfortable sofa",
                "size": [2.0, 0.8, 0.8],
                "center": [2.0, 1.0, 0.4],
                "rotation": [0, 0, 0],
                "asset_ref": {"source_db": "imaginarium", "asset_key": "sofa_asset", "mesh_uri": None, "pointcloud_uri": None, "metadata_uri": None},
                "asset_proxy": {"type": "obb_from_metadata_or_csv", "bbox_center_local": [0, 0, 0], "bbox_size": [2.0, 0.8, 0.8]},
                "metadata": {"interactive": False},
            }
        ],
        "metadata": {"generator": "test", "adapter": "passthrough"},
    }


def test_canonical_artifact_validation_accepts_valid_examples() -> None:
    assert validate_scene_request(_scene_request())
    assert validate_object_plan(_object_plan())
    assert validate_asset_selection(_asset_selection())
    assert validate_generated_scene(_generated_scene())


def test_direct_natural_language_generation_input_skips_asset_selection() -> None:
    request_id = "direct_nl"
    generation_input = build_direct_natural_language_generation_input(
        request_id=request_id,
        instruction="Create a cozy living room.",
        scene_type="living room",
        room=_scene_request()["room"],
        object_plan={**_object_plan(), "request_id": request_id},
    )

    assert validate_generation_input(generation_input)
    assert generation_input["scene_request"]["structure"] is False
    assert generation_input["generation_contract"]["input_mode"] == "natural_language_direct"
    assert generation_input["generator_input"]["instruction"] == "Create a cozy living room."
    assert "asset_selection" not in generation_input


def test_structured_generation_input_can_skip_asset_retrieval() -> None:
    generation_input = build_generation_input(
        scene_request={**_scene_request(), "structure": True},
        object_plan=_object_plan(),
        asset_selection=None,
    )

    assert validate_generation_input(generation_input)
    assert generation_input["generation_contract"]["input_mode"] == "natural_language_structured"
    assert generation_input["generation_contract"]["requires_asset_selection"] is False
    assert generation_input["generator_input"]["object_plan"]["objects"][0]["id"] == "obj_000"
    assert generation_input["evaluation_context"]["structure_available_to_generator"] is True
    assert "asset_selection" not in generation_input


def test_passthrough_adapter_copies_and_validates_generated_scene(tmp_path: Path) -> None:
    generated_scene_path = write_json(tmp_path / "input_scene.json", _generated_scene())
    adapter = get_adapter("passthrough")
    method_input = adapter.prepare_input(_generation_input(), tmp_path)
    generated_path = adapter.parse_output(generated_scene_path, _generation_input(), tmp_path)

    assert method_input.name == "generation_input.json"
    assert generated_path.name == "generated_scene.json"
    assert read_json(generated_path)["scene_id"] == "generated_demo_001"


def test_generate_dispatcher_stops_cleanly_when_generation_skipped(tmp_path: Path) -> None:
    result = run_generate(generation_input=_generation_input("skip_run"), adapter_name="passthrough", out_dir=tmp_path)

    status = read_json(result["workflow_status"])
    assert status == {
        "status": "generation_skipped",
        "reason": "No generated scene provided and --run-generation was not set.",
        "next_expected_input": "generated_scene.json",
    }
    assert result["generated_scene"] is None


def test_generate_from_natural_language_api_prepares_direct_method_input(tmp_path: Path) -> None:
    result = run_generate_from_natural_language(
        instruction="Place a red bed in front of the window.",
        scene_type="bedroom",
        room=_scene_request()["room"],
        request_id="nl_to_generator",
        adapter_name="passthrough",
        out_dir=tmp_path,
        object_plan={**_object_plan(), "request_id": "nl_to_generator"},
    )

    method_input = read_json(result["method_input"])
    assert method_input["generation_contract"]["input_mode"] == "natural_language_direct"
    assert method_input["generator_input"]["instruction"] == "Place a red bed in front of the window."
    assert method_input["evaluation_context"]["asset_retrieval_skipped"] is True
    assert "asset_selection" not in method_input


def test_generate_dispatcher_attaches_self_reflection_feedback(tmp_path: Path) -> None:
    generated_scene_path = write_json(tmp_path / "input_scene.json", _generated_scene())
    evaluation_report = {"overall_score": 0.25, "reports": {"generic_validity": {"overall_score": 0.25}}}

    result = run_generate(
        generation_input=_generation_input("reflective"),
        adapter_name="passthrough",
        out_dir=tmp_path / "reflective",
        generated_scene=generated_scene_path,
        evaluation_report=evaluation_report,
        previous_generated_scene=_generated_scene(),
        iteration=1,
    )

    method_input = read_json(result["method_input"])
    assert method_input["self_reflection"]["source"] == "evaluate.py"
    assert method_input["self_reflection"]["target"] == "generate.py"
    assert method_input["self_reflection"]["iteration"] == 1
    assert method_input["self_reflection"]["previous_evaluation"] == evaluation_report


def test_manual_output_adapter_enriches_method_output_from_asset_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "asset_info.csv"
    csv_path.write_text(
        "id,name_en,bbx,caption_en,short_desc,class_en,retrieval_class_en\n"
        '1,chair_asset,"[0.5, 0.6, 0.9]",A wooden chair,wood chair,chair,chair\n',
        encoding="utf-8",
    )
    raw_scene = _generated_scene()
    raw_scene["objects"][0] = {
        "id": "obj_000",
        "jid": "chair_asset",
        "center": [1.0, 1.0, 0.45],
        "rotation": [0, 0, 0],
        "asset_ref": {"source_db": "imaginarium", "asset_key": "chair_asset"},
    }
    raw_path = write_json(tmp_path / "raw_scene.json", raw_scene)

    adapter = get_adapter("manual")
    generated_path = adapter.parse_output(raw_path, _generation_input(), tmp_path / "out", config={"asset_csv": str(csv_path), "enrich_assets": True})
    generated = read_json(generated_path)

    assert generated["objects"][0]["size"] == [0.5, 0.6, 0.9]
    assert generated["objects"][0]["category"] == "chair"
    assert generated["objects"][0]["desc"] == "A wooden chair"


def test_evaluate_consumes_generated_scene_without_generation_artifacts(tmp_path: Path) -> None:
    report = run_evaluate(scene=_generated_scene("eval_only"), out=tmp_path / "evaluation_report.json")

    assert report["request_id"] == "eval_only"
    assert "generic_validity" in report["reports"]


def test_scene_harness_partial_run_with_supplied_plan_and_selection(tmp_path: Path) -> None:
    plan_path = write_json(tmp_path / "plan.json", _object_plan())
    selection_path = write_json(tmp_path / "selection.json", _asset_selection())
    out_dir = tmp_path / "partial"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_scene_harness.py"),
            "--instruction",
            "Create a room.",
            "--scene-type",
            "living room",
            "--object-plan",
            str(plan_path),
            "--asset-selection",
            str(selection_path),
            "--asset-mode",
            "retrieve",
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = read_json(out_dir / "run_manifest.json")
    assert manifest["status"] == "generation_skipped"
    assert (out_dir / "generation_input.json").exists()
    assert manifest["artifacts"]["generated_scene"] is None
    assert manifest["asset_resolution"]["mode"] == "retrieve"
    assert manifest["asset_resolution"]["retrieval_enabled"] is True
    assert manifest["asset_resolution"]["generation_enabled"] is False
    assert manifest["asset_resolution"]["selector"] == "top1"


def test_scene_harness_retrieve_generate_mode_is_recorded(tmp_path: Path) -> None:
    plan_path = write_json(tmp_path / "plan.json", _object_plan())
    selection_path = write_json(tmp_path / "selection.json", _asset_selection())
    plugin_path = tmp_path / "asset_plugin.py"
    plugin_path.write_text(
        "def generate_asset(request):\n"
        "    return {'jid': 'generated_asset', 'size': [1, 1, 1], 'mesh_uri': 'generated.glb'}\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "generation_enabled"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_scene_harness.py"),
            "--instruction",
            "Create a room.",
            "--scene-type",
            "living room",
            "--object-plan",
            str(plan_path),
            "--asset-selection",
            str(selection_path),
            "--asset-mode",
            "retrieve-generate",
            "--asset-generator-plugin",
            f"{plugin_path}:generate_asset",
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = read_json(out_dir / "run_manifest.json")
    assert manifest["asset_resolution"]["mode"] == "retrieve-generate"
    assert manifest["asset_resolution"]["retrieval_enabled"] is True
    assert manifest["asset_resolution"]["generation_enabled"] is True
    assert manifest["asset_resolution"]["generation_tool_configured"] is True


def test_scene_harness_no_structure_skips_retrieval_and_keeps_object_plan(tmp_path: Path) -> None:
    plan_path = write_json(tmp_path / "plan.json", _object_plan())
    out_dir = tmp_path / "direct_nl"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_scene_harness.py"),
            "--instruction",
            "Create a room with a red bed in front of the window.",
            "--scene-type",
            "bedroom",
            "--object-plan",
            str(plan_path),
            "--no-structure",
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = read_json(out_dir / "run_manifest.json")
    generation_input = read_json(out_dir / "generation_input.json")
    assert manifest["status"] == "generation_skipped"
    assert manifest["artifacts"]["asset_selection"] is None
    assert not (out_dir / "asset_selection.json").exists()
    assert generation_input["generation_contract"]["input_mode"] == "natural_language_direct"
    assert generation_input["scene_request"]["structure"] is False
    assert generation_input["scene_request"]["room"] == {
        "boundary": [[0, 0], [5, 0], [5, 5], [0, 5]],
        "height": 2.9,
        "unit": "meter",
    }
    assert generation_input["object_plan"]["objects"][0]["id"] == "obj_000"
    assert generation_input["generator_input"]["instruction"].startswith("Create a room")
    assert manifest["asset_resolution"]["mode"] == "off"
    assert manifest["asset_resolution"]["retrieval_enabled"] is False


def test_scene_harness_off_can_use_structure_without_assets(tmp_path: Path) -> None:
    plan_path = write_json(tmp_path / "plan.json", _object_plan())
    out_dir = tmp_path / "structured_without_assets"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_scene_harness.py"),
            "--instruction",
            "Create a room with a sofa against the wall.",
            "--scene-type",
            "living room",
            "--object-plan",
            str(plan_path),
            "--structure",
            "--asset-mode",
            "off",
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = read_json(out_dir / "run_manifest.json")
    generation_input = read_json(out_dir / "generation_input.json")
    assert manifest["asset_resolution"]["mode"] == "off"
    assert manifest["asset_resolution"]["retrieval_enabled"] is False
    assert manifest["asset_resolution"]["adapter_support"] == "optional"
    assert generation_input["generation_contract"]["input_mode"] == "natural_language_structured"
    assert generation_input["generator_input"]["object_plan"]["objects"][0]["id"] == "obj_000"
    assert not (out_dir / "asset_selection.json").exists()


def test_scene_harness_full_run_with_external_generated_scene(tmp_path: Path) -> None:
    plan_path = write_json(tmp_path / "plan.json", _object_plan())
    selection_path = write_json(tmp_path / "selection.json", _asset_selection())
    generated_path = write_json(tmp_path / "generated.json", _generated_scene("full"))
    out_dir = tmp_path / "full"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_scene_harness.py"),
            "--instruction",
            "Create a room.",
            "--scene-type",
            "living room",
            "--object-plan",
            str(plan_path),
            "--asset-selection",
            str(selection_path),
            "--asset-mode",
            "retrieve",
            "--generated-scene",
            str(generated_path),
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = read_json(out_dir / "run_manifest.json")
    assert manifest["status"] == "generated_scene_available"
    assert (out_dir / "generated_scene.json").exists()
    assert (out_dir / "evaluation_report.json").exists()
    assert read_json(out_dir / "evaluation_report.json")["request_id"] == "full"


def test_scene_harness_iteration_limit_writes_reflexive_generation_input(tmp_path: Path) -> None:
    plan_path = write_json(tmp_path / "plan.json", _object_plan())
    selection_path = write_json(tmp_path / "selection.json", _asset_selection())
    invalid_scene = _generated_scene("reflective_loop")
    invalid_scene["objects"][0]["center"] = [20.0, 20.0, 0.4]
    generated_path = write_json(tmp_path / "invalid_generated.json", invalid_scene)
    out_dir = tmp_path / "reflective_loop"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_scene_harness.py"),
            "--instruction",
            "Create a room.",
            "--scene-type",
            "living room",
            "--object-plan",
            str(plan_path),
            "--asset-selection",
            str(selection_path),
            "--asset-mode",
            "retrieve",
            "--generated-scene",
            str(generated_path),
            "--iteration-limit",
            "1",
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = read_json(out_dir / "run_manifest.json")
    history = read_json(out_dir / "self_reflexive_history.json")
    reflected_input = read_json(out_dir / "iterations" / "iter_001" / "generation_input.json")

    assert manifest["status"] == "reflection_generation_pending"
    assert history["attempts"][0]["valid"] is False
    assert history["attempts"][1]["status"] == "generation_skipped"
    assert reflected_input["self_reflection"]["iteration"] == 1
    assert reflected_input["self_reflection"]["previous_evaluation"]["overall_score"] < 1.0
