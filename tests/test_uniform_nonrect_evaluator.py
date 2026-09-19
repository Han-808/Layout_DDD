"""Geometry adapters must not introduce a second score/fallback protocol."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import socket
import subprocess

import pytest

from benchmark.camera_cal_scene_level import uniform, composition, orchestrator
from benchmark.evaluator.generic_validity.oob import check_oob
from benchmark.evaluator.generic_validity.support import check_support
from benchmark.evaluator.scoring import score_oob_report
from benchmark.evaluator.judgement_coverage import metric_projection, apply_report
from benchmark.non_rectangular.geometry import polygon_geometry_from_scene, PolygonRoomGeometryError
from benchmark.rendering.camera_pose import generate_camera_pose_candidates, generate_global_context_poses
from benchmark.visual_judge.evidence_resolution import FALLBACK_POLICY, evidence_policy_scope
from test_p0b_oob import _obj, _scene, _Judge
from test_evidence_adaptive_judgement import Model, complete_required_rows


RECTANGLE = [[0, 0], [4, 0], [4, 4], [0, 4]]
L_SHAPE = [[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]]
ANGLED = [[0, 0], [4, 0], [3, 4], [0, 4]]


def polygon_scene(objects, boundary=L_SHAPE, *, floor=0.0, ceiling_in_scope=False):
    result = _scene(objects, boundary=deepcopy(boundary), height=floor + 4)
    walls = []
    for i, (start, end) in enumerate(zip(boundary, boundary[1:] + boundary[:1])):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        walls.append({"wall_id": f"edge_{i}", "start_xy": list(start), "end_xy": list(end),
                      "inward_normal_xy": [-dy / length, dx / length], "height_m": 4.0, "thickness_m": 0.12})
    result["metadata"].update(evaluation_mode="non_rectangular_multi_room", coordinates_transformed=False,
        non_rectangular_room_geometry={"schema_version": "non_rectangular_polygon_room_geometry_v1",
            "room_id": "room_000", "floor_polygon_xy": deepcopy(boundary), "wall_segments": walls,
            "floor_z_m": floor, "ceiling_z_m": floor + 4, "ceiling_in_scope": ceiling_in_scope})
    return result


@pytest.mark.parametrize("center", [[2, 2, .5], [.3, 2, .5], [3.7, 2, .5],
    [2, .3, .5], [2, 3.7, .5], [3.7, 3.7, .5], [2, 2, .497],
    [2, 2, .48], [2, 2, 3.8]])
@pytest.mark.parametrize("rotation", [[0, 0, 0], [15, 5, 30]])
def test_rectangle_geometry_adapter_same_verdict_coverage_and_burden(center, rotation):
    obj = _obj("a", center, rotation=rotation)
    plain = _scene([obj], boundary=RECTANGLE, height=4)
    polygon = polygon_scene([deepcopy(obj)], RECTANGLE, ceiling_in_scope=True)
    results = []
    with evidence_policy_scope({"evidence_resolution_policy": FALLBACK_POLICY}):
        for scene in (plain, polygon):
            result = check_oob(scene, vlm_judge=_Judge("invalid"))
            projected = metric_projection("oob", result, ["a"])
            results.append((result, projected))
    left, right = results
    assert left[0]["objects"][0]["candidate_oob"] == right[0]["objects"][0]["candidate_oob"]
    assert left[0]["objects"][0]["final_verdict"] == right[0]["objects"][0]["final_verdict"]
    assert left[1]["judgement_fraction"] == right[1]["judgement_fraction"]
    assert left[1]["observed_score"] == pytest.approx(right[1]["observed_score"])


@pytest.mark.parametrize("boundary,center,candidate", [
    (L_SHAPE, [3, 3, .5], True),  # Inside AABB but outside the real room.
    (L_SHAPE, [1, 3, .5], False),
    (L_SHAPE, [3, 1, .5], False),  # Do not intersect all concave wall half-planes.
    (ANGLED, [3.6, 3, .5], True),
    (ANGLED, [1, 2, .5], False),
])
def test_real_polygon_not_bbox_or_infinite_halfplanes(boundary, center, candidate):
    report = check_oob(polygon_scene([_obj("a", center)], boundary), vlm_judge=_Judge("invalid"))
    record = report["objects"][0]
    assert record["candidate_oob"] is candidate
    assert (record["final_verdict"] == "invalid") is candidate
    if candidate:
        assert record["outside_area_m2"] > 0 and record["violated_edges"]
        assert score_oob_report(report, ordered_object_ids=["a"])["score"] < 1


@pytest.mark.parametrize("floor", [-10.0, -1.5, 0.0, 2.3])
def test_floor_height_enters_support_and_oob_without_coordinate_rewrite(floor):
    from benchmark.scene_io.validate import validate_generated_scene
    scene = polygon_scene([_obj("a", [1, 1, floor + .5])], floor=floor)
    validate_generated_scene(scene)
    before = deepcopy(scene)
    oob = check_oob(scene, vlm_judge=_Judge("invalid"))
    support = check_support(scene, vlm_judge=_Judge("valid"))
    assert oob["objects"][0]["floor_penetration_m"] == pytest.approx(0)
    record = support["objects"][0]
    assert record["architecture_plane_clearances_m"]["floor"] == pytest.approx(0)
    assert record["certified_grounded_support"] is True
    assert scene == before
    lowered = deepcopy(scene)
    lowered["objects"][0]["center"][2] -= .1
    assert check_oob(lowered, vlm_judge=_Judge("invalid"))["objects"][0]["floor_penetration_m"] == pytest.approx(.1)


def test_no_ceiling_is_explicit_geometry_not_a_different_fallback():
    scene = polygon_scene([_obj("a", [1, 1, 5])], ceiling_in_scope=False)
    assert not check_oob(scene, vlm_judge=_Judge("invalid"))["objects"][0]["candidate_oob"]
    scene["metadata"]["non_rectangular_room_geometry"]["ceiling_in_scope"] = True
    assert check_oob(scene, vlm_judge=_Judge("invalid"))["objects"][0]["candidate_oob"]


@pytest.mark.parametrize("fault", ["boundary", "normal", "edge", "duplicate", "height"])
def test_inconsistent_polygon_fails_before_judgement(fault):
    scene = polygon_scene([_obj("a", [1, 1, .5])])
    geometry = scene["metadata"]["non_rectangular_room_geometry"]
    if fault == "boundary":
        scene["boundary"] = RECTANGLE
    elif fault == "normal":
        geometry["wall_segments"][0]["inward_normal_xy"] = [0, -1]
    elif fault == "edge":
        geometry["wall_segments"][0]["end_xy"] = [3, 0]
    elif fault == "duplicate":
        geometry["wall_segments"][1]["wall_id"] = "edge_0"
    else:
        geometry["wall_segments"][0]["height_m"] = -1
    judge = _Judge("valid")
    with pytest.raises(PolygonRoomGeometryError):
        polygon_geometry_from_scene(scene)
    assert judge.calls == 0


def test_unjudged_polygon_oob_is_unknown_not_default_valid():
    scene = polygon_scene([_obj("a", [3, 3, .5])])
    with evidence_policy_scope({"evidence_resolution_policy": FALLBACK_POLICY}):
        result = check_oob(scene)
    assert result["score"] is None
    assert result["objects"][0]["final_verdict"] is None
    assert metric_projection("oob", result, ["a"])["judgement_fraction"] == 0


def test_polygon_transport_failure_is_not_eligible_score():
    class Failed(_Judge):
        def adjudicate_p0b(self, request):
            raise ConnectionError("offline transport fault")
    with evidence_policy_scope({"evidence_resolution_policy": FALLBACK_POLICY}):
        result = check_oob(polygon_scene([_obj("a", [3, 3, .5])]), vlm_judge=Failed("valid"))
    projection = metric_projection("oob", result, ["a"])
    assert projection["observed_score"] is None
    assert projection["infrastructure_failure"]


@pytest.mark.parametrize("metric", ["support", "oob", "semantic_placement_consistency"])
def test_polygon_camera_bank_preserves_real_room_and_floor(metric):
    scene = polygon_scene([_obj("a", [1, 3, 2.8])], floor=2.3)
    geometry = polygon_geometry_from_scene(scene)
    request = {"scene": scene, "metric": metric, "object_ids": ["a"],
               "event": {"object_id": "a"}, "detector_evidence": {}}
    poses = generate_camera_pose_candidates(request, max_candidates=6, policy="local")
    assert poses
    for pose in poses:
        assert geometry.contains_xy(pose["location"][:2])
        assert pose["location"][2] >= geometry.floor_z_m
        assert geometry.segment_visible_inside_room(pose["location"][:2], pose["target"][:2])
    assert generate_global_context_poses(scene)


def test_full_public_evaluator_polygon_uses_uniform_projection(tmp_path):
    from benchmark.api.evaluation import run_evaluate
    from benchmark.visual_judge import OpenAICompatibleVLMJudge
    from benchmark.camera_cal_scene_level.persisted_scoring import case_scoring_summary
    scene = polygon_scene([_obj("a", [1, 1, .5], category="chair", description="wood chair")])
    model = Model(complete_required_rows)
    result = run_evaluate(scene=scene, out=tmp_path / "raw.json",
        vlm_judge=OpenAICompatibleVLMJudge(model, evidence_resolution_policy=FALLBACK_POLICY,
            terminal_evidence_policy="best_available_final_v1"),
        asset_policy={"mode": "generated_or_open_assets", "arrangement_owner": "generator",
            "category_selection_owner": "generator", "scale_owner": "generator", "appearance_owner": "generator"},
        render_evidence=[], vlm_evaluation_control={"evidence_resolution_policy": FALLBACK_POLICY,
            "budgets": {"max_evidence_rounds": 0}})
    result = apply_report(result)
    summary = case_scoring_summary(case_id="polygon", case_manifest=result,
        l1_report=result["layer_reports"]["l1_physical_plausibility"],
        l3_report=result["reports"]["scene_quality"])
    assert summary["combined_score_100"] == result["benchmark_score_100"]
    assert result["judgement_coverage_summary"]["minimum_judgement_coverage"] == .8
    assert not any("nonrect_evidence_continuity" in json.dumps(m) for m in result["reports"]["scene_quality"]["metrics"].values())
    assert model.calls


def write_native_case(tmp_path, scene):
    dataset = tmp_path / "dataset"
    case = dataset / "nr.layout.room_000"
    def write(rel, value):
        path = case / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    write("case_manifest.json", {"case_id": case.name, "status": "ready", "scene_type": "room", "paths": {}})
    write("scene/canonical_scene.json", scene)
    write("annotation.json", {})
    write("evidence/prepared_render_manifest.json", {"identity_legend": {"1": "a"}})
    write("evidence/collision_geometry_manifest.json", {"schema_version": "collision_geometry_v1", "units": "meter", "up_axis": "z",
        "objects": {obj["id"]: {"representation": "bbox_proxy", "complete": False, "geometry_path": None} for obj in scene["objects"]}})
    for rel in ("prepared/evaluation.blend", "evidence/standardized_top.png",
                "evidence/standardized_perspective.png", "evidence/standardized_identity_map.png"):
        write(rel, {"offline_fixture": True})
    return dataset, case.name


@pytest.mark.parametrize("mode", ["open-space", "multi-room", "non-rect"])
@pytest.mark.parametrize("workers", [1, 2])
def test_same_native_judge_and_control_for_three_mode_labels(tmp_path, monkeypatch, mode, workers):
    scene = polygon_scene([_obj("a", [1, 1, .5])])
    dataset, case = write_native_case(tmp_path, scene)
    class Reached(BaseException):
        pass
    def boundary(**kwargs):
        from benchmark.visual_judge.control_config import resolve_vlm_evaluation_control
        control = resolve_vlm_evaluation_control(kwargs["vlm_evaluation_control"])
        judge = kwargs["vlm_judge"]
        assert control.evidence_resolution_policy == judge.evidence_resolution_policy == FALLBACK_POLICY
        assert judge.terminal_evidence_policy == "best_available_final_v1"
        assert control.max_evidence_rounds == 3 and control.max_total_images == 8
        assert polygon_geometry_from_scene(kwargs["scene"]).floor_polygon_xy == tuple(map(tuple, L_SHAPE))
        raise Reached()
    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected live network/render during offline test")
    monkeypatch.setattr(composition, "run_evaluate", boundary)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setenv("JUDGE_ENDPOINT", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("JUDGE_MODEL", "gpt-5.6-sol")
    monkeypatch.setenv("JUDGE_API_KEY_ENV", "UNIFORM_OFFLINE_KEY")
    monkeypatch.setenv("UNIFORM_OFFLINE_KEY", "offline")
    args = uniform.parser().parse_args(["--mode", mode, "--dataset-root", str(dataset),
        "--case-id", case, "--output-root", str(tmp_path / "out"), "--max-workers", str(workers)])
    assert uniform.inspect_inputs(args)[0]["input_limitations"]
    deps = uniform.dependencies(uniform.native_argv(args))
    deps = replace(deps, execution=replace(deps.execution,
        endpoint_preflight=lambda **kwargs: {"attempts_required": 1, "api_invocations": 0}))
    with pytest.raises(Reached):
        orchestrator.run_main(deps=deps)


def test_nonrect_mode_does_not_fabricate_missing_geometry(tmp_path):
    dataset, case = write_native_case(tmp_path, _scene([_obj("a", [1, 1, .5])]))
    args = uniform.parser().parse_args(["--mode", "non-rect", "--dataset-root", str(dataset), "--case-id", case])
    with pytest.raises(ValueError, match="authoritative polygon"):
        uniform.inspect_inputs(args)


def test_camera_projection_keeps_polygon_not_rectangular_formal_contract():
    from benchmark.visual_judge.p0b import _project_scene_for_camera_evidence
    from benchmark.non_rectangular.architecture import observable_architecture_from_scene
    from benchmark.evaluator.scene_quality.functional_probe import _functional_architecture_context
    scene = polygon_scene([_obj("a", [1, 1, 2.8])], floor=2.3)
    before = deepcopy(scene)
    projected = _project_scene_for_camera_evidence(scene)
    assert "architecture_contract" not in projected["metadata"]
    assert projected["metadata"]["non_rectangular_room_geometry"] == scene["metadata"]["non_rectangular_room_geometry"]
    architecture = observable_architecture_from_scene(projected)
    assert architecture["floor"]["z"] == 2.3
    assert architecture["logical_boundary"]["boundary"] == L_SHAPE
    assert len(architecture["physical_walls"]["wall_segments"]) == 6
    assert _functional_architecture_context(projected)["physical_wall_ids"] == [f"edge_{i}" for i in range(6)]
    assert scene == before


def test_polygon_overlays_use_ordered_walls_and_true_floor_not_bbox():
    from benchmark.rendering.collision_overlay import _architecture_plane_overlays
    scene = polygon_scene([_obj("a", [1, 1, 2.8])], floor=2.3)
    detector = {"violated_edges": [{"wall_id": "edge_3", "start_xy": [99, 99]}],
                "plane_flags": {"floor_oob": True, "ceiling_oob": True}}
    overlays = _architecture_plane_overlays(scene, detector)
    wall = next(item for item in overlays if item.get("wall_id") == "edge_3")
    assert wall["corners"][0] == [2, 2, 2.3]
    floor = next(item for item in overlays if item["name"] == "floor")
    assert [point[:2] for point in floor["corners"]] == L_SHAPE
    assert all(point[2] == 2.3 for point in floor["corners"])
    assert len(floor["edges"]) == 6
    assert not any(item["name"] == "ceiling" for item in overlays)


@pytest.mark.parametrize("fault", [ValueError("bad pose"), OSError("renderer failed")])
def test_polygon_hard_acquisition_fault_does_not_become_exhaustion(fault):
    from benchmark.visual_judge import OpenAICompatibleVLMJudge
    model = Model(AssertionError("Do not call after an infrastructure failure"))
    judge = OpenAICompatibleVLMJudge(model, evidence_resolution_policy=FALLBACK_POLICY,
                                    terminal_evidence_policy="best_available_final_v1")
    def failed(request):
        raise fault
    with evidence_policy_scope({"evidence_resolution_policy": FALLBACK_POLICY}):
        result = check_oob(polygon_scene([_obj("a", [3, 3, .5])]),
                           vlm_judge=judge, local_view_provider=failed)
    projection = metric_projection("oob", result, ["a"])
    assert projection["infrastructure_failure"] and projection["judgement_fraction"] == 0
    assert not model.calls


@pytest.mark.parametrize("ceiling", [False, True])
def test_support_and_camera_honor_explicit_ceiling_geometry(ceiling):
    from benchmark.non_rectangular.support_geometry import polygon_support_architecture_evidence
    obj = _obj("a", [1, 1, 5.8])
    scene = polygon_scene([obj], floor=2.3, ceiling_in_scope=ceiling)
    geometry = polygon_geometry_from_scene(scene)
    evidence = polygon_support_architecture_evidence(scene, raw_object=obj, tolerance_m=.01)
    assert ("ceiling" in evidence["architecture_plane_clearances_m"]) is ceiling
    assert any(c["plane"] == "ceiling" for c in evidence["architecture_contact_candidates"]) is ceiling
    assert (geometry.camera_max_z_m == geometry.ceiling_z_m) is ceiling


@pytest.mark.parametrize("ending", ["valid", "invalid"])
def test_polygon_empty_camera_bank_uses_real_common_model_fallback(ending):
    from benchmark.visual_judge import OpenAICompatibleVLMJudge
    from benchmark.visual_judge.acquisition_outcome import AcquisitionExhausted
    from test_evidence_adaptive_judgement import response
    scene = polygon_scene([_obj("a", [3, 3, 2.8])], floor=2.3)
    requests = []
    def exhausted(request):
        requests.append(deepcopy(request))
        raise AcquisitionExhausted("trusted_candidate_bank_empty")
    answer = response(ending)
    for defect in answer["defects"]:
        defect["target_ids"] = ["a"]
    model = Model(answer)
    judge = OpenAICompatibleVLMJudge(model, evidence_resolution_policy=FALLBACK_POLICY,
                                    terminal_evidence_policy="best_available_final_v1")
    with evidence_policy_scope({"evidence_resolution_policy": FALLBACK_POLICY}):
        result = check_oob(scene, vlm_judge=judge, local_view_provider=exhausted)
    assert requests and model.calls
    assert result["objects"][0]["final_verdict"] == ending
    projection = metric_projection("oob", result, ["a"])
    assert projection["judgement_fraction"] == 1
    assert not projection["infrastructure_failure"]
    # Inspect the actual delivered model context, not just the geometry helper.
    delivered = json.dumps(model.calls)
    assert "floor_polygon_xy" in delivered and "floor_z_m" in delivered and "2.3" in delivered


def test_schema_accepts_only_explicit_polygon_projection():
    from jsonschema import Draft202012Validator
    from benchmark.scene_io.validate import validate_generated_scene, ArtifactValidationError
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "schemas/scene.schema.json").read_text())
    assert (root / "schemas/scene.schema.json").read_bytes() == (root / "src/benchmark/_resources/schemas/scene.schema.json").read_bytes()
    scene = polygon_scene([_obj("a", [1, 1, .5])])
    scene["metadata"].pop("coordinate_frame")
    Draft202012Validator(schema).validate(scene)
    validate_generated_scene(scene)
    plain = deepcopy(scene)
    plain["metadata"].pop("evaluation_mode")
    with pytest.raises(ArtifactValidationError):
        validate_generated_scene(plain)
    assert list(Draft202012Validator(schema).iter_errors(plain))


@pytest.mark.requires_local_data
def test_retained_nonrect_room_geometry_without_replay_or_input_rewrite():
    from benchmark.scene_io.validate import validate_generated_scene
    from benchmark.non_rectangular.architecture import observable_architecture_from_scene
    source = Path("/Users/han_mohan/Desktop/Layout_DDD/Support/artifacts/outputs/complicated_agent_evaluation/sol_plus20_adaptive_room2_blender2_v1_r1/evaluation/models/api2-gpt-5-6-sol-agent-v1/scenes/scene_011634/rooms/room_000/materialization_attempts/attempt_001/materialization")
    path = source / "canonical_room_scene.json"
    if not path.is_file():
        pytest.skip("Retained local nonrect materialization unavailable")
    assert uniform.digest(path) == "5207191fea952bb718b9628a2807a92c63bf644a97f7b2de942154247b28c43e"
    scene = json.loads(path.read_text())
    before = deepcopy(scene)
    validate_generated_scene(scene)
    geometry = polygon_geometry_from_scene(scene)
    assert len(geometry.walls) == 8 and len(scene["objects"]) == 20
    assert min(x for x, _ in geometry.floor_polygon_xy) < 0
    architecture = observable_architecture_from_scene(scene)
    assert architecture["floor"]["z"] == geometry.floor_z_m
    assert not architecture["ceiling"]["enabled"]
    assert generate_global_context_poses(scene)
    assert scene == before
    # Existing materialization is NOT a complete native canary package.
    assert not (source / "evidence/collision_geometry_manifest.json").exists()
