from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from benchmark.architecture_policy import (
    CANONICAL_WALL_IDS,
    build_architecture_contract,
)
from benchmark.api.submission import (
    TrustedCaseBundle,
    evaluate_prepared_submission,
    prepare_submission,
)
from benchmark.evaluator.profile import (
    L1,
    L2,
    L3,
    L4,
    resolve_evaluation_profile,
)
from benchmark.io_contracts import O3_SCENE_PACKAGE
from benchmark.materialization import (
    NativeRegistryAuthority,
    write_benchmark_native_registry,
)
from benchmark.materialization.blender import (
    inspect_sanitized_blend,
    materialize_catalog_scene,
)
from benchmark.nl_scene.generation_input import (
    build_generation_input,
    build_scene_request,
)
from benchmark.rendering import BlenderRenderError, BlenderRenderer
from benchmark.utils.io import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]
BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")
ASSET_ROOT = ROOT / "Support" / "Assets" / "imaginarium_assets"
ASSET_CSV = ASSET_ROOT / "imaginarium_asset_info.csv"
ASSET_ID = "0_alarm_clock_01_2k_packed"
BLEND_INSPECTOR_WORKER = (
    ROOT / "src" / "benchmark" / "materialization" / "blend_inspector_worker.py"
)


def _generation_input() -> dict:
    scene_request = build_scene_request(
        request_id="materialization_smoke",
        instruction="Place the selected object.",
        scene_type="room",
        room={
            "boundary": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
            "height": 3.0,
            "unit": "meter",
        },
        structure=True,
    )
    return build_generation_input(
        scene_request=scene_request,
        object_plan={
            "request_id": "materialization_smoke",
            "scene_type": "room",
            "scene_description": "one selected clock",
            "objects": [
                {
                    "id": "clock_slot",
                    "role": "clock",
                    "category": "clock",
                    "description": "selected clock",
                    "count": 1,
                    "placement_intent": {
                        "absolute_relations": [],
                        "relative_relations": [],
                    },
                    "metadata": {},
                }
            ],
            "global_constraints": [],
            "relations": [],
        },
        asset_selection={
            "request_id": "materialization_smoke",
            "objects": [
                {
                    "object_id": "clock_slot",
                    "object_spec": {
                        "role": "clock",
                        "category": "clock",
                        "description": "selected clock",
                        "estimated_size": [0.4, 0.4, 0.4],
                        "count": 1,
                    },
                    "retrieval_query": {
                        "description": "selected clock",
                        "category": "clock",
                        "size_constraint": [0.4, 0.4, 0.4],
                    },
                    "selected_asset": {
                        "jid": ASSET_ID,
                        "category": "clock",
                        "retrieval_category": "Alarm_clock",
                        "desc": "selected frozen clock",
                        "short_desc": "frozen clock",
                        "size": [0.13165, 0.066748, 0.174156],
                        "asset_ref": {
                            "source_db": "imaginarium",
                            "asset_key": ASSET_ID,
                            "mesh_uri": None,
                            "pointcloud_uri": None,
                            "metadata_uri": None,
                        },
                        "asset_proxy": {
                            "type": "canonical_catalog_bbox",
                            "bbox_center_local": [0.0, 0.0, 0.0],
                            "bbox_size": [0.13165, 0.066748, 0.174156],
                        },
                        "metadata": {},
                    },
                    "candidates": [],
                    "selection_action": "select",
                    "selection_decision": {
                        "action": "select",
                        "selected_jid": ASSET_ID,
                        "reason": "fixture",
                        "generation_request": None,
                    },
                    "selection_reason": "fixture",
                }
            ],
        },
        evaluator_output_type=O3_SCENE_PACKAGE,
    )


def _bundle(tmp_path: Path) -> TrustedCaseBundle:
    manifest = tmp_path / "case_bundle.json"
    manifest.write_text("{}\n", encoding="utf-8")
    profile = resolve_evaluation_profile()
    profile["layer_weights"] = {L1: 1.0, L2: 0.0, L3: 0.0, L4: 0.0}
    profile[L3]["enabled"] = False
    for metric in profile[L3]["metrics"].values():
        metric["enabled"] = False
        metric["weight"] = 0.0
    return TrustedCaseBundle(
        root=tmp_path,
        manifest_path=manifest,
        manifest_sha256="b" * 64,
        case_id="materialization_smoke",
        evaluator_output_type=O3_SCENE_PACKAGE,
        scene_request={
            "request_id": "materialization_smoke",
            "instruction": "Place the selected object.",
            "scene_type": "room",
            "structure": True,
            "prompt_granularity": "fine_grained",
            "room": {
                "boundary": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
                "height": 3.0,
                "unit": "meter",
            },
        },
        reference_annotation=None,
        specification_contract=None,
        specification_activation_mode="none",
        functional_semantic_config=None,
        scene_quality_config=None,
        object_grouping_report=None,
        asset_policy=None,
        authorized_deviations=None,
        spatial_fidelity_ontology=None,
        visual_style_spec=None,
        evaluation_profile=resolve_evaluation_profile(profile),
        workflow="canonical_l0_l4",
        enabled_evaluators={},
        p0b_official_mode=True,
        camera_evidence={
            "mode": None,
            "metric_modes": {},
            "max_views": 2,
            "max_steps": 0,
            "collision_overlay": True,
            "collision_contour": True,
            "active_fallback": {
                "enabled": False,
                "max_steps": 0,
                "candidate_count": 0,
                "fail_on_exhausted": True,
                "shadow_mode": True,
            },
        },
        catalog_snapshot_id="imaginarium_test_snapshot",
        allowed_asset_ids=(ASSET_ID,),
        artifact_records={},
    )


class _Judge:
    def adjudicate_p0b(self, request: dict) -> dict:
        del request
        return {"verdict": "valid", "confidence": 1.0, "reason": "test"}

    def adjudicate_relation(self, request: dict) -> dict:
        del request
        return {"verdict": "valid", "confidence": 1.0, "reason": "test"}

    def adjudicate_functional_semantic(self, request: dict) -> dict:
        del request
        return {
            "evidence_status": "sufficient",
            "verdict": "valid",
            "confidence": 1.0,
            "reason": "test",
            "defects": [],
        }

    def adjudicate_scene_quality(self, request: dict) -> dict:
        del request
        return {
            "evidence_status": "sufficient",
            "verdict": "valid",
            "confidence": 1.0,
            "reason": "test",
            "defects": [],
        }


@pytest.mark.skipif(
    not BLENDER.is_file() or not ASSET_CSV.is_file(),
    reason="requires the bundled local Blender and frozen Imaginarium catalog",
)
def test_real_blender_materializes_and_independently_inspects_frozen_asset(
    tmp_path: Path,
) -> None:
    generation_input = _generation_input()
    result = prepare_submission(
        artifact={
            "schema_version": "catalog_placement_v1",
            "instances": [
                {
                    "instance_id": "clock_left",
                    "asset_id": ASSET_ID,
                    "center_m": [1.25, 1.5, 0.8],
                    "uniform_scale": 2.0,
                    "rotation_euler_xyz_deg": [10.0, 20.0, 30.0],
                    "slot_id": "clock_slot",
                }
            ],
        },
        case_bundle=_bundle(tmp_path),
        out_dir=tmp_path / "prepared",
        asset_root=ASSET_ROOT,
        asset_csv=ASSET_CSV,
        blender_bin=BLENDER,
        generation_input=generation_input,
        timeout_seconds=120,
    )

    readiness = read_json(result.readiness_report_path)
    consistency = read_json(result.consistency_report_path)
    scene = read_json(result.normalized_scene_path)
    registry = read_json(result.instance_registry_path)
    build_report = read_json(
        result.trusted_render_source_path.with_suffix(
            result.trusted_render_source_path.suffix + ".build.json"
        )
    )

    assert readiness["status"] == "ready"
    assert consistency["status"] == "passed"
    assert build_report["placement"]["scale_mode"] == (
        "exact_uniform_scale"
    )
    assert build_report["placement"][
        "requested_scale_equals_effective_scale"
    ] is True
    assert "fit_mode" not in build_report["placement"]
    assert result.trusted_render_source_path.is_file()
    assert result.hashes["trusted_render_source_sha256"]
    assert scene["objects"][0]["id"] == "clock_left"
    assert scene["objects"][0]["category"] == "clock"
    assert registry["instances"][0]["instance_id"] == "clock_left"
    assert registry["instances"][0]["render_enabled"] is True

    trusted_hash_before = hashlib.sha256(
        result.trusted_render_source_path.read_bytes()
    ).hexdigest()
    render_manifest = BlenderRenderer(
        blender_bin=BLENDER,
        timeout_seconds=120,
        width=128,
        height=128,
        render_engine="BLENDER_WORKBENCH",
        require_asset_mesh=True,
    ).render_prepared_scene(
        blend_file=result.trusted_render_source_path,
        normalized_scene_path=result.normalized_scene_path,
        out_dir=tmp_path / "prepared_render",
    )
    trusted_hash_after = hashlib.sha256(
        result.trusted_render_source_path.read_bytes()
    ).hexdigest()

    assert render_manifest["blend_file"] == (
        result.trusted_render_source_path.as_posix()
    )
    assert render_manifest["source_scene_saved"] is False
    assert render_manifest["source_blend_modified"] is False
    assert render_manifest["source_blend_sha256_before"] == trusted_hash_before
    assert render_manifest["source_blend_sha256_after"] == trusted_hash_after
    assert trusted_hash_after == trusted_hash_before
    assert render_manifest["collision_geometry"]["export_summary"][
        "complete_mesh_count"
    ] == 1

    render_state_tamper = tmp_path / "catalog_render_state_tamper.blend"
    shutil.copyfile(result.trusted_render_source_path, render_state_tamper)
    _mutate_catalog_render_state(render_state_tamper)
    render_state_inspection = inspect_sanitized_blend(
        blend_path=render_state_tamper,
        expected_registry_path=tmp_path
        / "prepared"
        / "materialization_plan.json",
        out_path=tmp_path / "catalog_render_state_inspection.json",
        blender_bin=BLENDER,
        timeout_seconds=120,
    )
    assert render_state_inspection["status"] == "failed"
    assert "registered_asset_render_state_override" in (
        render_state_inspection["reason_codes"]
    )
    with pytest.raises(BlenderRenderError):
        BlenderRenderer(
            blender_bin=BLENDER,
            timeout_seconds=120,
            width=128,
            height=128,
            render_engine="BLENDER_WORKBENCH",
            require_asset_mesh=True,
        ).render_prepared_scene(
            blend_file=render_state_tamper,
            normalized_scene_path=result.normalized_scene_path,
            out_dir=tmp_path / "catalog_render_state_rejected",
        )

    report = evaluate_prepared_submission(
        prepared_submission=result,
        case_bundle=_bundle(tmp_path),
        out_dir=tmp_path / "prepared_evaluation",
        renderer=BlenderRenderer(
            blender_bin=BLENDER,
            timeout_seconds=120,
            width=128,
            height=128,
            render_engine="BLENDER_WORKBENCH",
            require_asset_mesh=True,
        ),
        vlm_judge=_Judge(),
        asset_root=ASSET_ROOT,
        asset_csv=ASSET_CSV,
        generation_input=generation_input,
        official_mode=True,
    )
    assert report["layer_reports"]["l0_structural_validity"][
        "readiness"
    ]["status"] == "ready"
    assert report["evidence_provenance"]["render_input_policy"] == (
        "benchmark_owned_sanitized_blend"
    )
    assert report["evidence_provenance"][
        "trusted_render_source_rederived_at_evaluation"
    ] is True
    assert Path(
        report["evidence_provenance"]["trusted_render_source"]
    ).resolve() != result.trusted_render_source_path.resolve()
    assert report["evidence_provenance"][
        "submitted_native_blend_rendered_directly"
    ] is False
    evaluated_scene = read_json(
        tmp_path / "prepared_evaluation" / "generated_scene.json"
    )
    assert evaluated_scene["objects"][0]["metadata"]["materialization"] == (
        scene["objects"][0]["metadata"]["materialization"]
    )
    assert evaluated_scene["objects"][0]["metadata"][
        "appearance_provenance"
    ] == scene["objects"][0]["metadata"]["appearance_provenance"]
    schema = read_json(ROOT / "schemas" / "evaluation_report.schema.json")
    Draft202012Validator(schema).validate(report)

    trusted_inspection = read_json(
        tmp_path / "prepared" / "trusted_blend_inspection.json"
    )
    observed = trusted_inspection["instances"][0]
    native_appearance = tmp_path / "native_with_untrusted_appearance.blend"
    shutil.copyfile(result.trusted_render_source_path, native_appearance)
    _mutate_native_blend(native_appearance, unsupported=False)
    native_hash_before = hashlib.sha256(native_appearance.read_bytes()).hexdigest()
    native_registry_authority = NativeRegistryAuthority.from_secret(
        key_id="materialization-test-authority",
        secret=b"catalog-native-registry-test-secret-0001",
    )
    native_registry_instances = [
        {
            "instance_id": observed["instance_id"],
            "evaluator_object_id": observed["evaluator_object_id"],
            "asset_id": observed["asset_id"],
            "native_root_name": observed["root_object_name"],
            "center_m": observed["center_m"],
            "uniform_scale": observed["requested_uniform_scale"],
            "rotation_euler_xyz_deg": observed[
                "rotation_euler_xyz_deg"
            ],
            "geometry_sha256": observed["geometry_sha256"],
            "material_sha256": observed["material_sha256"],
        }
    ]
    native_registry_path = write_benchmark_native_registry(
        tmp_path / "native_registry.json",
        authority=native_registry_authority,
        source_blend_path=native_appearance,
        case_bundle_manifest_sha256="b" * 64,
        catalog_snapshot_id="imaginarium_test_snapshot",
        instances=native_registry_instances,
    )
    assert read_json(native_registry_path)["source_blend_sha256"] == (
        native_hash_before
    )

    native_prepared = prepare_submission(
        artifact=native_appearance,
        case_bundle=_bundle(tmp_path),
        out_dir=tmp_path / "native_prepared",
        asset_root=ASSET_ROOT,
        asset_csv=ASSET_CSV,
        blender_bin=BLENDER,
        native_registry_path=native_registry_path,
        native_registry_authority=native_registry_authority,
        timeout_seconds=120,
    )
    native_hash_after = hashlib.sha256(native_appearance.read_bytes()).hexdigest()
    native_readiness = read_json(native_prepared.readiness_report_path)
    native_provenance = read_json(native_prepared.provenance_path)
    rematerialized_inspection = read_json(
        tmp_path / "native_prepared" / "trusted_blend_inspection.json"
    )

    assert native_readiness["status"] == "ready"
    assert native_hash_after == native_hash_before
    assert native_provenance["source"][
        "original_native_source_integrity"
    ] == {
        "path": native_appearance.as_posix(),
        "sha256_before": native_hash_before,
        "sha256_after": native_hash_after,
        "modified": False,
    }
    # Submitted cameras and lights are inventory-only and never copied to the
    # official trusted render source.
    assert rematerialized_inspection["technical_state"]["camera_count"] == 0
    assert rematerialized_inspection["technical_state"]["light_count"] == 0
    assert native_prepared.trusted_render_source_path != native_appearance

    public_mapping_path = write_json(
        tmp_path / "public_native_mapping.json",
        {
            "schema_version": "public_native_instance_mapping_v1",
            "instances": [
                {
                    "instance_id": observed["instance_id"],
                    "asset_id": observed["asset_id"],
                    "native_root_name": observed["root_object_name"],
                    "center_m": observed["center_m"],
                    "uniform_scale": observed["requested_uniform_scale"],
                    "rotation_euler_xyz_deg": observed[
                        "rotation_euler_xyz_deg"
                    ],
                    "slot_id": observed["slot_id"],
                }
            ],
        },
    )
    public_native_prepared = prepare_submission(
        artifact=native_appearance,
        case_bundle=_bundle(tmp_path),
        out_dir=tmp_path / "public_native_prepared",
        asset_root=ASSET_ROOT,
        asset_csv=ASSET_CSV,
        blender_bin=BLENDER,
        native_instance_mapping_path=public_mapping_path,
        native_registry_authority=native_registry_authority,
        timeout_seconds=120,
    )
    public_native_provenance = read_json(
        public_native_prepared.provenance_path
    )
    derived_registry = read_json(
        tmp_path
        / "public_native_prepared"
        / "benchmark_derived_native_registry.json"
    )
    native_registry_authority.verify(derived_registry)
    assert read_json(public_native_prepared.readiness_report_path)[
        "status"
    ] == "ready"
    assert public_native_provenance["native_registry"]["origin"] == (
        "benchmark_derived_from_public_native_mapping"
    )
    assert public_native_provenance["native_source_inspection"][
        "inspection_mode"
    ] == "public_native"
    assert hashlib.sha256(native_appearance.read_bytes()).hexdigest() == (
        native_hash_before
    )
    assert public_native_prepared.trusted_render_source_path != (
        native_appearance
    )
    preserved_public_mapping = Path(
        public_native_provenance["public_native_mapping"]["path"]
    )
    assert preserved_public_mapping.parent == (
        tmp_path / "public_native_prepared"
    ).resolve()
    public_mapping_path.unlink()
    public_report = evaluate_prepared_submission(
        prepared_submission=public_native_prepared,
        case_bundle=_bundle(tmp_path),
        out_dir=tmp_path / "public_native_evaluation",
        renderer=BlenderRenderer(
            blender_bin=BLENDER,
            timeout_seconds=120,
            width=128,
            height=128,
            render_engine="BLENDER_WORKBENCH",
            require_asset_mesh=True,
        ),
        vlm_judge=_Judge(),
        asset_root=ASSET_ROOT,
        asset_csv=ASSET_CSV,
        blender_bin=BLENDER,
        native_registry_authority=native_registry_authority,
        official_mode=True,
    )
    assert public_report["layer_reports"]["l0_structural_validity"][
        "readiness"
    ]["status"] == "ready"

    preserved_public_mapping.write_text(
        '{"schema_version":"public_native_instance_mapping_v1",'
        '"instances":[]}\n',
        encoding="utf-8",
    )
    tampered_mapping_report = evaluate_prepared_submission(
        prepared_submission=public_native_prepared,
        case_bundle=_bundle(tmp_path),
        out_dir=tmp_path / "public_native_mapping_tamper",
        renderer=BlenderRenderer(
            blender_bin=BLENDER,
            timeout_seconds=120,
            width=128,
            height=128,
            render_engine="BLENDER_WORKBENCH",
            require_asset_mesh=True,
        ),
        vlm_judge=_Judge(),
        asset_root=ASSET_ROOT,
        asset_csv=ASSET_CSV,
        blender_bin=BLENDER,
        native_registry_authority=native_registry_authority,
        official_mode=False,
    )
    tampered_readiness = tampered_mapping_report["layer_reports"][
        "l0_structural_validity"
    ]["readiness"]
    assert tampered_readiness["status"] == "not_evaluable"
    assert {
        "invalid_prepared_native_mapping",
        "native_instance_mapping_hash_binding_mismatch",
    } <= set(tampered_readiness["reason_codes"])

    unsupported_native = tmp_path / "native_unsupported.blend"
    shutil.copyfile(native_appearance, unsupported_native)
    _mutate_native_blend(unsupported_native, unsupported=True)
    unsupported_hash_before = hashlib.sha256(
        unsupported_native.read_bytes()
    ).hexdigest()
    unsupported_registry_path = write_benchmark_native_registry(
        tmp_path / "unsupported_native_registry.json",
        authority=native_registry_authority,
        source_blend_path=unsupported_native,
        case_bundle_manifest_sha256="b" * 64,
        catalog_snapshot_id="imaginarium_test_snapshot",
        instances=native_registry_instances,
    )
    assert read_json(unsupported_registry_path)["source_blend_sha256"] == (
        unsupported_hash_before
    )
    rejected = prepare_submission(
        artifact=unsupported_native,
        case_bundle=_bundle(tmp_path),
        out_dir=tmp_path / "native_rejected",
        asset_root=ASSET_ROOT,
        asset_csv=ASSET_CSV,
        blender_bin=BLENDER,
        native_registry_path=unsupported_registry_path,
        native_registry_authority=native_registry_authority,
        timeout_seconds=120,
    )
    unsupported_hash_after = hashlib.sha256(
        unsupported_native.read_bytes()
    ).hexdigest()
    rejected_readiness = read_json(rejected.readiness_report_path)
    rejected_inspection = read_json(
        tmp_path / "native_rejected" / "native_source_inspection.json"
    )

    assert rejected_readiness["status"] == "not_evaluable"
    assert unsupported_hash_after == unsupported_hash_before
    assert not rejected.trusted_render_source_path.is_file()
    assert {
        "registered_asset_fingerprint_mismatch",
        "non_rigid_or_procedural_asset",
        "registered_instance_hidden",
        "external_file_reference",
    } <= set(rejected_inspection["reason_codes"])


@pytest.mark.skipif(
    not BLENDER.is_file(),
    reason="requires the bundled local Blender",
)
def test_material_fingerprint_recurses_groups_ramps_and_curves(
    tmp_path: Path,
) -> None:
    saved_blend = tmp_path / "material_fingerprint_probe.blend"
    expression = f"""
import bpy
import importlib.util
import json

spec = importlib.util.spec_from_file_location(
    "material_fingerprint_worker",
    {str(BLEND_INSPECTOR_WORKER)!r},
)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

cube = bpy.data.objects.get("Cube")
material = bpy.data.materials.new("recursive_material")
material.use_nodes = True
cube.data.materials.append(material)
cube[worker.INSTANCE_ID_PROPERTY] = "registered_instance"
cube[worker.EVALUATOR_ID_PROPERTY] = "registered_instance"
cube[worker.CANONICAL_ID_PROPERTY] = "registered_instance"
cube[worker.ASSET_ID_PROPERTY] = "catalog_asset"
cube[worker.ROLE_PROPERTY] = "instance_root"

outer = bpy.data.node_groups.new("fingerprint_outer", "ShaderNodeTree")
inner = bpy.data.node_groups.new("fingerprint_inner", "ShaderNodeTree")
outer_node = material.node_tree.nodes.new("ShaderNodeGroup")
outer_node.name = "fingerprint_outer_node"
outer_node.node_tree = outer
inner_node = outer.nodes.new("ShaderNodeGroup")
inner_node.name = "fingerprint_inner_node"
inner_node.node_tree = inner
ramp = inner.nodes.new("ShaderNodeValToRGB")
ramp.name = "fingerprint_ramp"
curve = inner.nodes.new("ShaderNodeRGBCurve")
curve.name = "fingerprint_curve"
curve_point = curve.mapping.curves[0].points.new(0.4, 0.6)
curve.mapping.update()

baseline = worker._material_fingerprint([cube])
baseline_repeat = worker._material_fingerprint([cube])
bpy.ops.wm.save_as_mainfile(filepath={str(saved_blend)!r})
bpy.ops.wm.open_mainfile(filepath={str(saved_blend)!r})
cube = bpy.data.objects["Cube"]
inner = bpy.data.node_groups["fingerprint_inner"]
ramp = inner.nodes["fingerprint_ramp"]
curve = inner.nodes["fingerprint_curve"]
curve_point = min(
    curve.mapping.curves[0].points,
    key=lambda point: abs(float(point.location[0]) - 0.4),
)
baseline_reopened = worker._material_fingerprint([cube])
ramp.color_ramp.elements[0].color = (0.9, 0.1, 0.2, 1.0)
after_ramp = worker._material_fingerprint([cube])
curve_point.location = (0.4, 0.2)
curve_point.handle_type = "VECTOR"
curve.mapping.update()
after_curve = worker._material_fingerprint([cube])

class FakeTree:
    bl_idname = "ShaderNodeTree"
    interface = None
    links = ()
    def __init__(self):
        self.nodes = []

class FakeNode:
    bl_idname = "ShaderNodeGroup"
    mute = False
    label = ""
    inputs = ()
    outputs = ()
    def __init__(self, name):
        self.name = name
        self.node_tree = None

tree_a = FakeTree()
tree_b = FakeTree()
node_a = FakeNode("to_b")
node_b = FakeNode("to_a")
node_a.node_tree = tree_b
node_b.node_tree = tree_a
tree_a.nodes.append(node_a)
tree_b.nodes.append(node_b)
cycle_one = worker._node_tree_fingerprint_payload(
    tree_a,
    active=[],
    memo={{}},
)
cycle_two = worker._node_tree_fingerprint_payload(
    tree_a,
    active=[],
    memo={{}},
)

geometry = worker._geometry_fingerprint(cube, [cube])
expected = {{
    "native_root_name": cube.name,
    "evaluator_object_id": "registered_instance",
    "asset_id": "catalog_asset",
    "_registry_record": {{
        "geometry_sha256": geometry,
        "material_sha256": baseline,
    }},
}}
report = worker._inspect(
    mode="registered_native",
    expected_records={{"registered_instance": expected}},
    expected_data=None,
    catalog_data=None,
    expected_path=None,
    catalog_path=None,
)

assembly_root = bpy.data.objects.new("assembly_root", None)
bpy.context.scene.collection.objects.link(assembly_root)
assembly_mesh = bpy.data.meshes.new("assembly_mesh")
assembly_mesh.from_pydata(
    [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
    [],
    [(0, 1, 2)],
)
left = bpy.data.objects.new("assembly_left", assembly_mesh.copy())
right = bpy.data.objects.new("assembly_right", assembly_mesh.copy())
bpy.context.scene.collection.objects.link(left)
bpy.context.scene.collection.objects.link(right)
left.parent = assembly_root
right.parent = assembly_root
left.location.x = -1.0
right.location.x = 1.0
bpy.context.view_layer.update()
red = bpy.data.materials.new("assembly_red")
red.diffuse_color = (1.0, 0.0, 0.0, 1.0)
blue = bpy.data.materials.new("assembly_blue")
blue.diffuse_color = (0.0, 0.0, 1.0, 1.0)
left.data.materials.append(red)
right.data.materials.append(blue)
assembly_geometry_before = worker._geometry_fingerprint(
    assembly_root,
    [left, right],
)
assembly_material_before = worker._material_fingerprint([left, right])
assembly_before = worker._asset_assembly_fingerprint(
    assembly_root,
    [left, right],
)
left.data.materials[0] = blue
right.data.materials[0] = red
assembly_geometry_after = worker._geometry_fingerprint(
    assembly_root,
    [left, right],
)
assembly_material_after = worker._material_fingerprint([left, right])
assembly_after = worker._asset_assembly_fingerprint(
    assembly_root,
    [left, right],
)
print(
    "MATERIAL_FINGERPRINT_PROBE="
    + json.dumps(
        {{
            "baseline": baseline,
            "baseline_repeat": baseline_repeat,
            "baseline_reopened": baseline_reopened,
            "after_ramp": after_ramp,
            "after_curve": after_curve,
            "cycle_stable": cycle_one == cycle_two,
            "report_status": report["status"],
            "reason_codes": report["reason_codes"],
            "observed_material_sha256": report["instances"][0][
                "material_sha256"
            ],
            "observed_asset_assembly_sha256": report["instances"][0][
                "asset_assembly_sha256"
            ],
            "assembly_geometry_unchanged": (
                assembly_geometry_before == assembly_geometry_after
            ),
            "assembly_material_unchanged": (
                assembly_material_before == assembly_material_after
            ),
            "asset_assembly_changed": assembly_before != assembly_after,
        }},
        sort_keys=True,
    )
)
"""
    completed = subprocess.run(
        [
            str(BLENDER),
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python-exit-code",
            "1",
            "--python-expr",
            expression,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    marker = "MATERIAL_FINGERPRINT_PROBE="
    encoded = next(
        line[len(marker) :]
        for line in completed.stdout.splitlines()
        if line.startswith(marker)
    )
    result = json.loads(encoded)

    assert result["baseline"] == result["baseline_repeat"]
    assert result["baseline"] == result["baseline_reopened"]
    assert result["baseline"] != result["after_ramp"]
    assert result["after_ramp"] != result["after_curve"]
    assert result["cycle_stable"] is True
    assert result["report_status"] == "failed"
    assert "registered_asset_fingerprint_mismatch" in result["reason_codes"]
    assert result["observed_material_sha256"] == result["after_curve"]
    assert len(result["observed_asset_assembly_sha256"]) == 64
    assert result["assembly_geometry_unchanged"] is True
    assert result["assembly_material_unchanged"] is True
    assert result["asset_assembly_changed"] is True


@pytest.mark.skipif(
    not BLENDER.is_file(),
    reason="requires the bundled local Blender",
)
def test_native_material_fingerprint_is_fail_closed_for_effective_state(
    tmp_path: Path,
) -> None:
    expression = f"""
import bpy
import importlib.util
import json

spec = importlib.util.spec_from_file_location(
    "native_material_fail_closed_worker",
    {str(BLEND_INSPECTOR_WORKER)!r},
)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

obj = bpy.data.objects["Cube"]
root = bpy.data.objects.new("native_material_root", None)
bpy.context.scene.collection.objects.link(root)
obj.parent = root

base = bpy.data.materials.new("native_material_base")
base.use_nodes = True
obj.data.materials.append(base)
baseline_binding = worker._object_material_binding_payload(obj)
override = bpy.data.materials.new("native_material_override")
override.diffuse_color = (0.9, 0.1, 0.2, 1.0)
slot = obj.material_slots[0]
slot.link = "OBJECT"
slot.material = override
override_binding = worker._object_material_binding_payload(obj)
slot.link = "DATA"

image = bpy.data.images.new(
    "native_generated_image",
    width=2,
    height=2,
    alpha=True,
    float_buffer=False,
)
image.pixels = [
    0.1, 0.2, 0.3, 1.0,
    0.4, 0.5, 0.6, 1.0,
    0.7, 0.8, 0.9, 1.0,
    0.2, 0.3, 0.4, 1.0,
]
image_node = base.node_tree.nodes.new("ShaderNodeTexImage")
image_node.image = image
generated_before = worker._material_signature(base)
image.pixels[0] = 0.95
image.update()
generated_after = worker._material_signature(base)
image.pack()
packed_before_dirty_edit = worker._material_signature(base)
image.pixels[1] = 0.05
image.update()
packed_after_dirty_edit = worker._material_signature(base)

unsupported_before = worker._unsupported_native_material_state([obj])
texcoord = base.node_tree.nodes.new("ShaderNodeTexCoord")
anchor = bpy.data.objects.new("submitted_shader_anchor", None)
bpy.context.scene.collection.objects.link(anchor)
texcoord.object = anchor
unsupported_pointer = worker._unsupported_native_material_state([obj])
texcoord.object = None

script = base.node_tree.nodes.new("ShaderNodeScript")
script.mode = "INTERNAL"
script.script = bpy.data.texts.new("submitted_osl_source")
script.script.write("shader submitted() {{}}")
ies = base.node_tree.nodes.new("ShaderNodeTexIES")
if hasattr(ies, "mode"):
    ies.mode = "INTERNAL"
if hasattr(ies, "ies"):
    ies.ies = bpy.data.texts.new("submitted_ies_source")
object_info = base.node_tree.nodes.new("ShaderNodeObjectInfo")
attribute = base.node_tree.nodes.new("ShaderNodeAttribute")
attribute.attribute_type = "OBJECT"
unsupported_implicit = worker._unsupported_native_material_state([obj])

assembly_before = worker._asset_assembly_fingerprint(root, [obj])
obj.color = (0.2, 0.3, 0.4, 1.0)
assembly_after_color = worker._asset_assembly_fingerprint(root, [obj])
obj.color = (1.0, 1.0, 1.0, 1.0)
obj.pass_index = 17
assembly_after_pass_index = worker._asset_assembly_fingerprint(root, [obj])
obj.pass_index = 0
obj["submitted_shader_attribute"] = "changed"
assembly_after_custom_property = worker._asset_assembly_fingerprint(
    root,
    [obj],
)
del obj["submitted_shader_attribute"]
obj["benchmark_instance_id"] = "materializer_owned_stamp"
assembly_after_benchmark_stamp = worker._asset_assembly_fingerprint(
    root,
    [obj],
)
del obj["benchmark_instance_id"]
extra = bpy.data.objects.new("submitted_extra_empty", None)
bpy.context.scene.collection.objects.link(extra)
extra.parent = root
bpy.context.view_layer.update()
assembly_after_extra_empty = worker._asset_assembly_fingerprint(root, [obj])

print(
    "NATIVE_MATERIAL_FAIL_CLOSED="
    + json.dumps(
        {{
            "effective_override_changed": (
                baseline_binding != override_binding
            ),
            "generated_pixels_changed": (
                generated_before != generated_after
            ),
            "dirty_packed_pixels_changed": (
                packed_before_dirty_edit != packed_after_dirty_edit
            ),
            "unsupported_before": unsupported_before,
            "pointer_states": [
                item["state"] for item in unsupported_pointer
            ],
            "implicit_node_types": sorted(
                {{
                    item["node_type"]
                    for item in unsupported_implicit
                }}
            ),
            "assembly_color_changed": (
                assembly_before != assembly_after_color
            ),
            "assembly_pass_index_changed": (
                assembly_before != assembly_after_pass_index
            ),
            "assembly_custom_property_changed": (
                assembly_before != assembly_after_custom_property
            ),
            "assembly_benchmark_stamp_ignored": (
                assembly_before == assembly_after_benchmark_stamp
            ),
            "assembly_extra_empty_changed": (
                assembly_before != assembly_after_extra_empty
            ),
        }},
        sort_keys=True,
    )
)
"""
    completed = subprocess.run(
        [
            str(BLENDER),
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python-exit-code",
            "1",
            "--python-expr",
            expression,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    marker = "NATIVE_MATERIAL_FAIL_CLOSED="
    encoded = next(
        line[len(marker) :]
        for line in completed.stdout.splitlines()
        if line.startswith(marker)
    )
    result = json.loads(encoded)

    assert result["effective_override_changed"] is True
    assert result["generated_pixels_changed"] is True
    assert result["dirty_packed_pixels_changed"] is True
    assert result["unsupported_before"] == []
    assert "unsupported_pointer_property" in result["pointer_states"]
    assert {
        "ShaderNodeAttribute",
        "ShaderNodeObjectInfo",
        "ShaderNodeScript",
        "ShaderNodeTexIES",
    } <= set(result["implicit_node_types"])
    assert result["assembly_color_changed"] is True
    assert result["assembly_pass_index_changed"] is True
    assert result["assembly_custom_property_changed"] is True
    assert result["assembly_benchmark_stamp_ignored"] is True
    assert result["assembly_extra_empty_changed"] is True


@pytest.fixture(scope="module")
def architecture_allowlist_fixture(tmp_path_factory):
    if not BLENDER.is_file():
        pytest.skip("requires the bundled local Blender")
    fixture_dir = tmp_path_factory.mktemp("architecture_allowlist")
    boundary = [
        [0.0, 0.0],
        [4.0, 0.0],
        [4.0, 4.0],
        [0.0, 4.0],
    ]
    room = {
        "boundary": boundary,
        "height": 3.0,
        "floor_z": 0.0,
        "unit": "meter",
    }
    architecture = build_architecture_contract(
        room,
        active_wall_ids=CANONICAL_WALL_IDS,
    )
    plan_path = write_json(
        fixture_dir / "materialization_plan.json",
        {
            "schema_version": "catalog_materialization_plan_v1",
            "materialization_revision": "fixed_catalog_materialization_v1",
            "adapter_contract_revision": "catalog_placement_v1",
            "catalog_snapshot_id": "architecture_allowlist_test",
            "request": {
                "request_id": "architecture_allowlist_test",
                "scene_type": "room",
                "boundary": boundary,
                "scene_height": 3.0,
                "architecture": architecture,
            },
            "instances": [],
        },
    )
    normalized_path = write_json(
        fixture_dir / "normalized_scene.json",
        {
            "objects": [],
            "relations": [],
            "boundary": boundary,
            "scene_height": 3.0,
            "metadata": {"architecture_contract": architecture},
        },
    )
    trusted_blend = fixture_dir / "trusted.blend"
    materialize_catalog_scene(
        plan_path=plan_path,
        out_blend_path=trusted_blend,
        inspection_path=fixture_dir / "trusted_inspection.json",
        blender_bin=BLENDER,
        timeout_seconds=120,
    )
    renderer = BlenderRenderer(
        blender_bin=BLENDER,
        timeout_seconds=120,
        width=64,
        height=64,
        render_engine="BLENDER_WORKBENCH",
        require_asset_mesh=True,
    )
    valid_manifest = renderer.render_prepared_scene(
        blend_file=trusted_blend,
        normalized_scene_path=normalized_path,
        out_dir=fixture_dir / "valid_render",
    )
    assert valid_manifest["source_blend_modified"] is False
    return {
        "plan_path": plan_path,
        "normalized_path": normalized_path,
        "trusted_blend": trusted_blend,
        "renderer": renderer,
    }


@pytest.mark.parametrize(
    ("mutation", "inspection_reason"),
    [
        ("role_only_mesh", "architecture_allowlist_mismatch"),
        ("duplicate_floor_mesh", "architecture_allowlist_mismatch"),
        ("forged_architecture_curve", "architecture_allowlist_mismatch"),
        ("visible_untagged_curve", "unregistered_renderable_mesh"),
        ("visible_instanced_curve", "unregistered_renderable_mesh"),
        ("floor_geometry", "architecture_allowlist_mismatch"),
        ("floor_material", "architecture_allowlist_mismatch"),
        ("floor_transform", "architecture_allowlist_mismatch"),
        (
            "view_layer_material_override",
            "sanitized_scene_render_state_override",
        ),
        (
            "layer_collection_holdout",
            "sanitized_scene_render_state_override",
        ),
        (
            "layer_collection_indirect_only",
            "sanitized_scene_render_state_override",
        ),
        ("world_nodes", "sanitized_scene_render_state_override"),
        ("compositor_nodes", "sanitized_scene_render_state_override"),
        ("sequencer_strip", "sanitized_scene_render_state_override"),
        ("scene_render_override", "sanitized_scene_render_state_override"),
    ],
)
def test_architecture_and_scene_renderable_allowlists_fail_closed(
    tmp_path: Path,
    architecture_allowlist_fixture,
    mutation: str,
    inspection_reason: str,
) -> None:
    mutated_blend = tmp_path / f"{mutation}.blend"
    shutil.copyfile(
        architecture_allowlist_fixture["trusted_blend"],
        mutated_blend,
    )
    _mutate_architecture_blend(mutated_blend, mutation=mutation)
    inspection = inspect_sanitized_blend(
        blend_path=mutated_blend,
        expected_registry_path=architecture_allowlist_fixture["plan_path"],
        out_path=tmp_path / f"{mutation}_inspection.json",
        blender_bin=BLENDER,
        timeout_seconds=120,
    )
    assert inspection["status"] == "failed"
    assert inspection_reason in inspection["reason_codes"]

    with pytest.raises(BlenderRenderError):
        architecture_allowlist_fixture["renderer"].render_prepared_scene(
            blend_file=mutated_blend,
            normalized_scene_path=architecture_allowlist_fixture[
                "normalized_path"
            ],
            out_dir=tmp_path / f"{mutation}_render",
        )


def _mutate_architecture_blend(path: Path, *, mutation: str) -> None:
    allowed = {
        "role_only_mesh",
        "duplicate_floor_mesh",
        "forged_architecture_curve",
        "visible_untagged_curve",
        "visible_instanced_curve",
        "floor_geometry",
        "floor_material",
        "floor_transform",
        "view_layer_material_override",
        "layer_collection_holdout",
        "layer_collection_indirect_only",
        "world_nodes",
        "compositor_nodes",
        "sequencer_strip",
        "scene_render_override",
    }
    if mutation not in allowed:
        raise ValueError(f"unsupported architecture mutation {mutation!r}")
    expression = f"""
import bpy
mutation = {mutation!r}
if mutation in {{"role_only_mesh", "duplicate_floor_mesh"}}:
    bpy.ops.mesh.primitive_cube_add(location=(2.0, 2.0, 1.0))
    obj = bpy.context.object
    obj.name = "submitted_architecture_mesh"
    obj["benchmark_role"] = "architecture"
    if mutation == "duplicate_floor_mesh":
        obj["benchmark_architecture_id"] = "floor"
elif mutation in {{"forged_architecture_curve", "visible_untagged_curve"}}:
    curve = bpy.data.curves.new("submitted_curve_data", type="CURVE")
    curve.dimensions = "3D"
    spline = curve.splines.new("POLY")
    spline.points.add(1)
    spline.points[0].co = (0.5, 0.5, 0.5, 1.0)
    spline.points[1].co = (3.5, 3.5, 2.0, 1.0)
    curve.bevel_depth = 0.1
    obj = bpy.data.objects.new("submitted_curve", curve)
    bpy.context.scene.collection.objects.link(obj)
    if mutation == "forged_architecture_curve":
        obj["benchmark_role"] = "architecture"
        obj["benchmark_architecture_id"] = "forged_wall"
elif mutation == "visible_instanced_curve":
    collection = bpy.data.collections.new("submitted_instance_collection")
    curve = bpy.data.curves.new("submitted_instanced_curve_data", type="CURVE")
    curve.dimensions = "3D"
    spline = curve.splines.new("POLY")
    spline.points.add(1)
    spline.points[0].co = (0.5, 0.5, 0.5, 1.0)
    spline.points[1].co = (3.5, 3.5, 2.0, 1.0)
    curve.bevel_depth = 0.1
    child = bpy.data.objects.new("submitted_instanced_curve", curve)
    collection.objects.link(child)
    obj = bpy.data.objects.new("submitted_collection_instance", None)
    obj.instance_type = "COLLECTION"
    obj.instance_collection = collection
    bpy.context.scene.collection.objects.link(obj)
elif mutation == "floor_geometry":
    bpy.data.objects["benchmark_floor"].data.vertices[0].co.x += 0.25
elif mutation == "floor_material":
    material = bpy.data.objects["benchmark_floor"].data.materials[0]
    color = (0.95, 0.02, 0.02, 1.0)
    material.diffuse_color = color
    principled = next(
        node
        for node in material.node_tree.nodes
        if node.type == "BSDF_PRINCIPLED"
    )
    principled.inputs["Base Color"].default_value = color
elif mutation == "floor_transform":
    bpy.data.objects["benchmark_floor"].location.x += 0.25
elif mutation == "view_layer_material_override":
    material = bpy.data.materials.new("submitted_material_override")
    material.diffuse_color = (0.95, 0.02, 0.02, 1.0)
    bpy.context.view_layer.material_override = material
elif mutation in {{
    "layer_collection_holdout",
    "layer_collection_indirect_only",
}}:
    layer = bpy.context.view_layer.layer_collection.children[
        "benchmark_architecture"
    ]
    if mutation == "layer_collection_holdout":
        layer.holdout = True
    else:
        layer.indirect_only = True
elif mutation == "world_nodes":
    background = next(
        node
        for node in bpy.context.scene.world.node_tree.nodes
        if node.type == "BACKGROUND"
    )
    background.inputs["Strength"].default_value = 7.0
elif mutation == "compositor_nodes":
    tree = bpy.data.node_groups.new(
        "submitted_compositor",
        "CompositorNodeTree",
    )
    bpy.context.scene.compositing_node_group = tree
    tree.nodes.new("CompositorNodeRGB")
elif mutation == "sequencer_strip":
    editor = bpy.context.scene.sequence_editor_create()
    strips = getattr(editor, "sequences", None)
    if strips is None:
        strips = editor.strips
    if bpy.app.version >= (5, 0, 0):
        strips.new_effect(
            name="submitted_color_strip",
            type="COLOR",
            channel=1,
            frame_start=1,
            length=1,
        )
    else:
        strips.new_effect(
            name="submitted_color_strip",
            type="COLOR",
            channel=1,
            frame_start=1,
            frame_end=2,
        )
elif mutation == "scene_render_override":
    bpy.context.scene.render.use_border = True
    bpy.context.scene.view_settings.exposure = 1.0
    bpy.context.view_layer.samples = 2
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
"""
    subprocess.run(
        [
            str(BLENDER),
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            str(path),
            "--python-exit-code",
            "1",
            "--python-expr",
            expression,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _mutate_catalog_render_state(path: Path) -> None:
    expression = """
import bpy
mesh = next(
    obj
    for obj in bpy.data.objects
    if obj.type == "MESH"
    and obj.get("benchmark_role") == "asset_descendant"
)
mesh.visible_camera = False
mesh.visible_shadow = False
mesh.is_holdout = True
mesh.is_shadow_catcher = True
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
"""
    subprocess.run(
        [
            str(BLENDER),
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            str(path),
            "--python-exit-code",
            "1",
            "--python-expr",
            expression,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _mutate_native_blend(path: Path, *, unsupported: bool) -> None:
    expression = """
import bpy
camera_data = bpy.data.cameras.new("submitted_camera")
camera = bpy.data.objects.new("submitted_camera", camera_data)
bpy.context.scene.collection.objects.link(camera)
light_data = bpy.data.lights.new("submitted_light", type="POINT")
light = bpy.data.objects.new("submitted_light", light_data)
bpy.context.scene.collection.objects.link(light)
if UNSUPPORTED:
    mesh = next(obj for obj in bpy.data.objects if obj.type == "MESH" and obj.get("benchmark_instance_id"))
    mesh.hide_render = True
    mesh.modifiers.new(name="submitted_modifier", type="SUBSURF")
    material = next(material for material in mesh.data.materials if material is not None)
    material.diffuse_color = (0.99, 0.01, 0.01, 1.0)
    image = bpy.data.images.new("submitted_external_image", width=1, height=1)
    image.source = "FILE"
    image.filepath = "/tmp/submitted_external_reference.png"
    if material.node_tree is not None:
        texture = material.node_tree.nodes.new("ShaderNodeTexImage")
        texture.image = image
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
""".replace("UNSUPPORTED", "True" if unsupported else "False")
    subprocess.run(
        [
            str(BLENDER),
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            str(path),
            "--python-exit-code",
            "1",
            "--python-expr",
            expression,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
