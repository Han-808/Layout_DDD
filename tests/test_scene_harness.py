from __future__ import annotations

from copy import deepcopy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from benchmark.adapters import get_adapter
from benchmark.visual_judge.interfaces.camera import CameraSelectionResult
from benchmark.scene_io.validate import (
    ArtifactValidationError,
    validate_asset_selection,
    validate_generated_scene,
    validate_generation_input,
    validate_object_plan,
    validate_scene_request,
)
from benchmark.utils.io import load_yaml, read_json, write_json
from evaluate import run_evaluate
from generate import run_generate, run_generate_from_natural_language
from scripts.run_scene_harness import run_scene_harness
import scripts.run_scene_harness as scene_harness_module
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
        "schema_version": "canonical_scene_v1",
        "scene_id": f"generated_{request_id}",
        "request_id": request_id,
        "scene_type": "living room",
        "boundary": [[0, 0], [7, 0], [7, 5], [0, 5]],
        "scene_height": 3.0,
        "objects": [
            {
                "id": "obj_000",
                "jid": "sofa_asset",
                "category": "sofa",
                "description": "A comfortable sofa",
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
        "metadata": {
            "generator": "test",
            "adapter": "object_state",
            "coordinate_frame": {
                "origin": "room_min_corner_floor",
                "axes": "x_width_y_depth_z_up",
                "unit": "meter",
                "rotation_unit": "degree",
            },
        },
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
    )

    assert validate_generation_input(generation_input)
    assert generation_input["scene_request"]["structure"] is False
    assert generation_input["generation_contract"]["input_mode"] == "natural_language_direct"
    assert generation_input["generator_input"]["instruction"] == "Create a cozy living room."
    assert "object_plan" not in generation_input
    assert "evaluation_context" not in generation_input
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
    assert "evaluation_context" not in generation_input
    assert "asset_selection" not in generation_input


def test_object_state_adapter_copies_and_validates_generated_scene(tmp_path: Path) -> None:
    generated_scene_path = write_json(tmp_path / "input_scene.json", _generated_scene())
    adapter = get_adapter("object_state")
    method_input = adapter.prepare_input(_generation_input(), tmp_path)
    generated_path = adapter.parse_output(generated_scene_path, _generation_input(), tmp_path)

    assert method_input.name == "method_input.json"
    assert "evaluation_context" not in read_json(method_input)["generator_input"]
    assert generated_path.name == "generated_scene.json"
    assert read_json(generated_path)["scene_id"] == "generated_demo_001"


def test_generate_dispatcher_stops_cleanly_when_generation_skipped(tmp_path: Path) -> None:
    result = run_generate(generation_input=_generation_input("skip_run"), adapter_name="object_state", out_dir=tmp_path)

    status = read_json(result["workflow_status"])
    assert status == {
        "status": "generation_skipped",
        "reason": "No method output provided and --run-generation was not set.",
        "next_expected_input": "method_output",
    }
    assert result["generated_scene"] is None


def test_generate_from_natural_language_api_prepares_direct_method_input(tmp_path: Path) -> None:
    result = run_generate_from_natural_language(
        instruction="Place a red bed in front of the window.",
        scene_type="bedroom",
        room=_scene_request()["room"],
        request_id="nl_to_generator",
        adapter_name="object_state",
        out_dir=tmp_path,
    )

    method_input = read_json(result["method_input"])
    assert method_input["io_contract"]["input_type"] == "i1_natural_language"
    assert method_input["generator_input"]["natural_language"] == "Place a red bed in front of the window."
    assert "structure" not in method_input["generator_input"]
    assert "evaluation_context" not in method_input["generator_input"]


def test_generate_dispatcher_attaches_self_reflection_feedback(tmp_path: Path) -> None:
    generated_scene_path = write_json(tmp_path / "input_scene.json", _generated_scene("reflective"))
    evaluation_report = {"benchmark_score": 0.25, "reports": {"generic_validity": {"score": 0.25}}}

    result = run_generate(
        generation_input=_generation_input("reflective"),
        adapter_name="object_state",
        out_dir=tmp_path / "reflective",
        method_output=generated_scene_path,
        evaluation_report=evaluation_report,
        previous_generated_scene=_generated_scene(),
        iteration=1,
    )

    method_input = read_json(result["method_input"])
    reflection = method_input["generator_input"]["self_reflection"]
    assert reflection["source"] == "evaluate.py"
    assert reflection["target"] == "generate.py"
    assert reflection["iteration"] == 1
    assert reflection["previous_evaluation"] == evaluation_report
    assert method_input["io_contract"]["feedback_assistance"] is True


def test_object_state_adapter_does_not_use_asset_csv_to_repair_missing_canonical_fields(tmp_path: Path) -> None:
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

    adapter = get_adapter("object_state")
    with pytest.raises(ArtifactValidationError, match="category"):
        adapter.parse_output(
            raw_path,
            _generation_input(),
            tmp_path / "out",
            config={"asset_csv": str(csv_path), "enrich_assets": True},
        )


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


def test_scene_harness_no_structure_skips_retrieval_and_omits_public_structure(tmp_path: Path) -> None:
    out_dir = tmp_path / "direct_nl"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_scene_harness.py"),
            "--instruction",
            "Create a room with a red bed in front of the window.",
            "--scene-type",
            "bedroom",
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
    assert generation_input["scene_request"]["room"]["dimensions"] == {
        "width": 7.0,
        "depth": 5.0,
        "height": 3.0,
    }
    assert generation_input["scene_request"]["room"]["resolution_policy"] == "room_dimension_policy_v1"
    assert "object_plan" not in generation_input
    assert manifest["artifacts"]["generator_structure"] is None
    assert not (out_dir / "generator_structure.json").exists()
    assert generation_input["generator_input"]["instruction"].startswith("Create a room")
    assert manifest["asset_resolution"]["mode"] == "off"
    assert manifest["asset_resolution"]["retrieval_enabled"] is False


def test_scene_harness_rejects_public_structure_in_i1_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be supplied with structure=false"):
        run_scene_harness(
            instruction="Create a room with a red bed.",
            scene_type="bedroom",
            out_dir=tmp_path / "invalid_i1",
            structure=False,
            generator_structure=_object_plan(),
        )


@pytest.mark.parametrize(
    "config_argument",
    ("asset_selector_model_config", "adapter_config"),
)
def test_scene_harness_rejects_literal_api_key_in_programmatic_configs(
    tmp_path: Path,
    config_argument: str,
) -> None:
    secret = "must-not-appear"
    kwargs = {
        "instruction": "A plain room.",
        "scene_type": "room",
        "out_dir": tmp_path / config_argument,
        config_argument: {"api_key": secret},
    }

    with pytest.raises(ValueError, match="use api_key_env") as captured:
        run_scene_harness(**kwargs)

    assert secret not in str(captured.value)


@pytest.mark.parametrize("granularity", ["fine_grained", "coarse_grained"])
def test_scene_harness_uses_one_canonical_workflow_for_both_granularities(
    tmp_path: Path,
    granularity: str,
) -> None:
    manifest = run_scene_harness(
        instruction="Create a bedroom.",
        scene_type="bedroom",
        prompt_granularity=granularity,
        out_dir=tmp_path / granularity,
    )

    assert manifest["evaluation"]["gate"]["workflow"] == "canonical_l0_l4"
    assert manifest["evaluation"]["gate"]["prompt_granularity_role"] == "metadata_only"
    assert "evaluation_mode" not in manifest["evaluation"]["gate"]
    assert manifest["prompt_granularity"] == {
        "requested": granularity,
        "resolved": granularity,
        "role": "metadata_only",
        "classifier_called": False,
        "classification": None,
    }


def test_scene_harness_rejects_retired_non_game_spatial_ontology(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="retired non-game workflow"):
        run_scene_harness(
            instruction="Create a bedroom.",
            scene_type="bedroom",
            spatial_fidelity_ontology={"categories": {}},
            out_dir=tmp_path / "retired",
        )


def test_scene_harness_preserves_the_frozen_legacy_game_gate(tmp_path: Path) -> None:
    profile = load_yaml(
        ROOT / "configs" / "evaluation" / "metric_profile_game_v1.yaml"
    )
    manifest = run_scene_harness(
        instruction="Run the frozen Game profile.",
        scene_type="game",
        prompt_granularity="fine_grained",
        evaluation_profile=profile,
        support_enabled=False,
        out_dir=tmp_path / "legacy_game",
    )

    assert manifest["evaluation"]["gate"] == {
        "source": "scene_request.prompt_granularity",
        "prompt_granularity": "fine_grained",
        "evaluation_mode": "fine_grained_mode",
        "category_2": "prompt_fidelity",
        "active_categories": [
            "prompt_fidelity",
            "structural_validity",
            "visual_quality",
        ],
    }
    assert manifest["evaluation"]["support_enabled"] is False
    assert "canonical_configs" not in manifest["evaluation"]
    assert manifest["prompt_granularity"] == {
        "requested": "fine_grained",
        "resolved": "fine_grained",
        "evaluation_mode": "fine_grained_mode",
        "classifier_called": False,
        "classification": None,
    }


def test_scene_harness_resolves_prompt_room_once_for_generator_and_manifest(tmp_path: Path) -> None:
    plan_path = write_json(tmp_path / "plan.json", _object_plan())
    out_dir = tmp_path / "prompt_room"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_scene_harness.py"),
            "--instruction",
            "Create a room measuring 8 m by 6 m with a ceiling height of 2.8 m.",
            "--scene-type",
            "living room",
            "--object-plan",
            str(plan_path),
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    generation_input = read_json(out_dir / "generation_input.json")
    manifest = read_json(out_dir / "run_manifest.json")
    resolved = generation_input["scene_request"]["room"]

    assert resolved["dimensions"] == {"width": 8.0, "depth": 6.0, "height": 2.8}
    assert generation_input["generation_contract"]["architecture"]["room"] == resolved
    assert manifest["task_contract"]["architecture"]["room"] == resolved
    assert manifest["task_contract"]["prompt_room_dimensions"] == {
        "width": 8.0,
        "depth": 6.0,
        "height": 2.8,
    }


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
            "--adapter",
            "object_state",
            "--method-output",
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


def test_scene_harness_executes_group_l3_deterministic_to_vlm_cascade(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_id = "harness_l3_camera"
    generated_path = write_json(
        tmp_path / "generated_l3.json",
        _generated_scene(request_id),
    )

    class _Renderer:
        def __init__(self, **kwargs):
            self.config = kwargs
            self.calls: list[dict] = []

        @staticmethod
        def _image(path: Path, color=(70, 100, 130)) -> str:
            path.parent.mkdir(parents=True, exist_ok=True)
            image = Image.new("RGB", (32, 32), color)
            image.putpixel((0, 0), (180, 50, 30))
            image.save(path)
            return str(path)

        def render_scene(self, *, scene_path, out_dir, asset_root=None):
            del scene_path, asset_root
            destination = Path(out_dir)
            destination.mkdir(parents=True, exist_ok=True)
            blend = destination / "scene.blend"
            blend.write_bytes(b"blend")
            perspective = self._image(
                destination / "global_perspective.png"
            )
            return {
                "blend_file": str(blend),
                "views": [
                    {
                        "name": "global_perspective",
                        "path": perspective,
                    }
                ],
                "collision_geometry": None,
            }

        def render_camera_views(
            self,
            *,
            blend_file,
            out_dir,
            camera_views,
            preview=False,
        ):
            del blend_file
            self.calls.append(
                {
                    "kind": "rgb",
                    "preview": preview,
                    "views": deepcopy(camera_views),
                }
            )
            views = []
            for index, pose in enumerate(camera_views):
                view_code = sum(
                    str(pose.get("id") or f"view_{index}").encode("utf-8")
                ) % 200
                path = self._image(
                    Path(out_dir) / f"rgb_{index:02d}.png",
                    (30 + view_code, 120, 160),
                )
                views.append(
                    {
                        "id": pose["id"],
                        "path": path,
                        "pose": deepcopy(pose),
                    }
                )
            return {"views": views, "render_gpu_time_seconds": 0.01}

        def render_focus_overlay_views(
            self,
            *,
            blend_file,
            out_dir,
            camera_views,
            overlay_spec,
            preview=False,
        ):
            del blend_file, overlay_spec
            self.calls.append(
                {
                    "kind": "focus",
                    "preview": preview,
                    "views": deepcopy(camera_views),
                }
            )
            views = []
            for index, pose in enumerate(camera_views):
                path = self._image(
                    Path(out_dir) / f"focus_{index:02d}.png",
                    (150, 70, 70),
                )
                views.append(
                    {
                        "id": pose["id"],
                        "path": path,
                        "pose": deepcopy(pose),
                    }
                )
            return {"views": views}

    class _Judge:
        vlm_control_enabled = True

        def __init__(self):
            self.scene_quality_requests: list[dict] = []

        def adjudicate_scene_quality(self, request):
            self.scene_quality_requests.append(deepcopy(request))
            if len(self.scene_quality_requests) == 1:
                return {
                    "evidence_status": "insufficient",
                    "verdict": "ambiguous",
                    "confidence": 0.2,
                    "reason": "Need the complete local context.",
                    "missing_evidence": [],
                    "defects": [],
                    "evidence_request": {
                        "target_ids": ["obj_000"],
                        "missing_observations": [
                            "group_context_visible"
                        ],
                        "view_goal": "show the object in its local context",
                        "metadata": {},
                    },
                }
            return {
                "evidence_status": "sufficient",
                "verdict": "valid",
                "confidence": 0.9,
                "reason": "The repaired view resolves scale.",
                "missing_evidence": [],
                "defects": [],
            }

        def adjudicate_p0b(self, request):
            del request
            return {
                "verdict": "valid",
                "confidence": 1.0,
                "reason": "valid",
            }

        def adjudicate_relation(self, request):
            del request
            return {
                "verdict": "valid",
                "confidence": 1.0,
                "reason": "valid",
            }

    class _VLMSelector:
        backend = "harness_vlm_selector"
        validated_internal_candidate_bank = True

        def __init__(self):
            self.requests = []

        def select(self, request):
            self.requests.append(request)
            pose = {
                "id": "harness_vlm_view",
                "location": [2.0, -2.0, 2.0],
                "target": [2.0, 1.0, 0.5],
                "lens_mm": 45.0,
                "sensor_width_mm": 36.0,
                "geometry_feasible": True,
                "geometry_feasibility_verified": True,
                "target_visibility_estimate": True,
                "joint_visibility_estimate": True,
                "projected_coverage_estimate": 0.4,
                "target_object_ids": ["obj_000"],
            }
            return CameraSelectionResult(
                outcome="selected",
                selected_view_ids=("harness_vlm_view",),
                selected_views=(pose,),
                reason_codes=("vlm_repair_selected",),
                reason="selected one group-local repair view",
                backend=self.backend,
                evidence_round=request.evidence_round,
            )

    monkeypatch.setattr(
        scene_harness_module,
        "BlenderRenderer",
        _Renderer,
    )
    judge = _Judge()
    selector = _VLMSelector()
    out_dir = tmp_path / request_id
    manifest = run_scene_harness(
        instruction="Create a living room.",
        scene_type="living room",
        adapter="object_state",
        method_output=generated_path,
        out_dir=out_dir,
        evaluator_vlm_judge=judge,
        blender_bin="/usr/bin/false",
        camera_active_selector=selector,
        vlm_evaluation_control={
            "camera_acquisition": {
                "policy": "deterministic_then_vlm",
            }
        },
        object_grouping_report={
            "status": "complete",
            "grouping_backend": "vlm",
            "grouping_policy_id": "vlm_visual_evidence_scope_v2",
            "object_groups": [
                {
                    "group_id": "group_001",
                    "object_ids": ["obj_000"],
                }
            ],
        },
        scene_quality_config={
            "metrics": {
                    "scale_consistency": {
                        "enabled": True,
                        # Keep this test focused on the Controller repair
                        # cascade after an initial visual Judge request.
                        "evidence_plan": {
                            "evidence_strategy": "global_and_local",
                            "router_options": None,
                        },
                    },
                "object_pairing_consistency": {"enabled": False},
                "style_consistency": {"enabled": False},
                "functional_consistency": {"enabled": False},
            }
        },
        asset_policy={
            "mode": "generated_or_open_assets",
            "identity_owner": "generator",
            "category_selection_owner": "generator",
            "scale_owner": "generator",
            "appearance_owner": "generator",
            "arrangement_owner": "generator",
        },
    )

    report = read_json(manifest["artifacts"]["evaluation_report"])
    metric = report["reports"]["scene_quality"]["metrics"][
        "scale_consistency"
    ]
    assert metric["judgement"]["verdict"] == "valid"
    assert metric["renderer_invoked"] is True
    assert metric["final_render_count"] >= 1
    assert report["reports"]["scene_quality"][
        "renderer_invoked"
    ] is True
    assert len(judge.scene_quality_requests) == 2
    assert len(selector.requests) == 1
    grouping_protocol = report["evaluation_config"]["object_grouping"][
        "input_protocol"
    ]
    assert grouping_protocol == {
        "input_mode": "caller_supplied_unknown",
        "provenance_status": "not_provided",
    }
    attempt = read_json(manifest["artifacts"]["self_reflexive_history"])[
        "attempts"
    ][0]
    assert attempt["l3_camera_control"] == {
        "mode": "judge_driven_independent_components",
        "deterministic_selector": "DeterministicLocalCameraSelector",
        "vlm_selector_configured": True,
        "renderer": "CameraViewEvidenceRenderer",
        "scene_access": "read_only",
        "initial_group_camera": {
            "mode": "visibility_ranked",
            "selector": "deterministic",
            "source": "default",
        },
    }
    controlled = report["evaluation_config"]["vlm_evaluation_control"][
        "integration"
    ]["runtime"]["controlled_calls"]
    trace = next(
        item
        for item in controlled
        if item["metric"] == "scale_consistency"
    )["audit"]["trace"]
    assert [item["stage"] for item in trace] == [
        "evidence_gate",
        "judge",
        "acquisition_planner",
        "trusted_candidate_bank",
        "camera_selector",
        "camera_escalation",
        "candidate_preview_render",
        "camera_selector",
        "render",
        "evidence_gate",
        "judge",
    ]


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
            "--adapter",
            "object_state",
            "--method-output",
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
    reflected_input = read_json(out_dir / "iterations" / "iter_001" / "generator" / "method_input.json")

    assert manifest["status"] == "reflection_generation_pending"
    assert history["attempts"][0]["valid"] is False
    assert history["attempts"][1]["status"] == "generation_skipped"
    reflection = reflected_input["generator_input"]["self_reflection"]
    assert reflection["iteration"] == 1
    assert reflection["previous_evaluation"]["benchmark_score"] is None
    assert reflected_input["io_contract"]["feedback_assistance"] is True
