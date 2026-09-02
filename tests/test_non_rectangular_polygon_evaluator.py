from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from shapely.geometry import LineString

import benchmark.non_rectangular.evaluator as nonrect_evaluator_module
from benchmark.api.evaluation import run_evaluate
from benchmark.evaluator.generic_validity.collision import check_collision
from benchmark.evaluator.generic_validity.support import check_support
from benchmark.non_rectangular.evaluator import (
    _finalize_nonrect_l1_continuity,
    NonRectangularRoomMetricIncomplete,
)
from benchmark.non_rectangular import (
    L1_METRICS,
    L3_METRICS,
    ROOM_REPORT_SCHEMA_VERSION,
    CanonicalNonRectangularRoomEvaluator,
    NonRectangularEvaluationInput,
    PolygonRoomGeometry,
    build_room_evaluation_units,
    check_polygon_oob,
    prepare_non_rectangular_evaluation,
    project_room_unit_to_canonical_scene,
    run_internal_non_rectangular_evaluation,
)
from benchmark.rendering.camera_pose import (
    apply_camera_action,
    generate_camera_pose_candidates,
    generate_global_context_poses,
)
from benchmark.rendering.collision_overlay import build_focus_overlay_spec
from benchmark.visual_judge.openai_compatible import (
    OpenAICompatibleVLMJudge,
)
from benchmark.visual_judge.p0b import adjudicate_p0b_event


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/non_rectangular"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _input() -> NonRectangularEvaluationInput:
    return NonRectangularEvaluationInput.from_artifacts(
        room_layout=_fixture("simple_multi_room.json"),
        room_program=_fixture("simple_multi_room_program.json"),
        object_plan=_fixture("simple_multi_room_object_plan.json"),
        generated_scene=_fixture("simple_multi_room_scene.json"),
    )


def _l_room_scene(*, object_center: list[float]) -> dict[str, Any]:
    layout = _fixture("l_shape_single.json")
    room = layout["rooms"][0]
    geometry = {
        "schema_version": "non_rectangular_polygon_room_geometry_v1",
        "room_id": room["room_id"],
        "floor_polygon_xy": room["floor_polygon_xy"],
        "wall_segments": room["wall_segments"],
        "floor_z_m": 0.0,
        "ceiling_z_m": 2.8,
        "tolerance_m": 1.0e-6,
    }
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": "l-room",
        "request_id": "l-room",
        "scene_type": "living room",
        "boundary": room["floor_polygon_xy"],
        "scene_height": 2.8,
        "objects": [
            {
                "id": "target",
                "category": "cabinet",
                "center": object_center,
                "size": [0.6, 0.8, 1.0],
                "rotation": [0.0, 0.0, 0.0],
            },
            {
                "id": "context",
                "category": "chair",
                "center": [0.65, 0.8, 0.5],
                "size": [0.5, 0.5, 1.0],
                "rotation": [0.0, 0.0, 0.0],
            },
        ],
        "metadata": {
            "evaluation_mode": "non_rectangular_multi_room",
            "non_rectangular_room_geometry": geometry,
        },
    }


def test_polygon_geometry_rejects_concavity_void_and_preserves_los() -> None:
    scene = _l_room_scene(object_center=[0.65, 2.5, 0.5])
    geometry = PolygonRoomGeometry.from_metadata(
        scene["metadata"]["non_rectangular_room_geometry"]
    )

    assert geometry.contains_xy([1.0, 3.0]) is True
    assert geometry.contains_xy([3.0, 3.0]) is False
    assert geometry.segment_visible_inside_room([0.5, 3.0], [0.5, 0.5])
    assert not geometry.segment_visible_inside_room([3.0, 1.0], [1.0, 3.0])


def test_polygon_ray_interval_ignores_empty_geos_line_component() -> None:
    scene = _l_room_scene(object_center=[0.65, 2.5, 0.5])
    geometry = PolygonRoomGeometry.from_metadata(
        scene["metadata"]["non_rectangular_room_geometry"]
    )

    class EmptyLineIntersection:
        def intersection(self, _line: Any) -> LineString:
            return LineString()

    intervals = geometry._xy_ray_intervals(
        EmptyLineIntersection(),
        target_vector=np.asarray([0.5, 0.5, 0.5]),
        direction=np.asarray([1.0, 0.0, 0.0]),
        maximum_trace=10.0,
    )

    assert intervals == []


def test_polygon_oob_detects_concave_notch_and_binds_actual_wall() -> None:
    scene = _l_room_scene(object_center=[1.55, 2.0, 0.5])

    report = check_polygon_oob(scene, {"detector_only": True})

    assert report["candidate_oob_count"] == 1
    record = report["objects"][0]
    assert record["outside_area_m2"] > 0.0
    assert record["violated_wall_ids"] == ["room_000.wall_003"]
    assert record["violated_edges"][0]["edge_local_frame"] == {
        "inward_normal_xy": [-1.0, 0.0],
        "tangent_xy": [0.0, 1.0],
    }


def test_polygon_oob_judge_receives_polygon_walls_and_no_ceiling() -> None:
    scene = _l_room_scene(object_center=[1.55, 2.0, 0.5])

    class Judge:
        vlm_control_enabled = False

        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        def adjudicate_p0b(self, request: dict[str, Any]) -> dict[str, Any]:
            self.requests.append(request)
            return {
                "verdict": "invalid",
                "confidence": 0.9,
                "reason": "mock",
            }

    judge = Judge()

    report = check_polygon_oob(scene, vlm_judge=judge)

    assert report["status"] == "checked"
    assert report["invalid_object_count"] == 1
    architecture = judge.requests[0]["architecture"]
    assert architecture["geometry_type"] == "non_rectangular_polygon"
    assert architecture["ceiling"] == {"enabled": False, "z": None}
    assert len(architecture["physical_walls"]["wall_segments"]) == 6


def test_nonrect_oob_overlay_legend_owns_exact_violated_wall() -> None:
    scene = _l_room_scene(object_center=[1.55, 2.0, 0.5])
    record = check_polygon_oob(scene, {"detector_only": True})["objects"][0]

    spec = build_focus_overlay_spec(
        scene=scene,
        metric="oob",
        object_ids=["target"],
        detector_evidence={"violated_edges": record["violated_edges"]},
        architecture_element="room_polygon_and_floor",
    )

    assert len(spec["architecture_planes"]) == 1
    plane = spec["architecture_planes"][0]
    assert plane["wall_id"] == "room_000.wall_003"
    assert plane["geometry_source"] == "ordered_nonrect_wall_segment"
    assert any(
        item["role"] == "architecture_plane"
        and item["id"] == plane["id"]
        for item in spec["legend"]
    )


def test_polygon_oob_projects_full_tilted_obb_not_only_midplane() -> None:
    scene = _l_room_scene(object_center=[0.35, 1.0, 1.0])
    target = scene["objects"][0]
    target["size"] = [0.4, 0.4, 2.0]
    target["rotation"] = [0.0, 45.0, 0.0]

    report = check_polygon_oob(scene, {"detector_only": True})

    record = report["objects"][0]
    assert record["candidate_oob"] is True
    assert record["outside_area_m2"] > 0.0
    assert "room_000.wall_005" in record["violated_wall_ids"]


def test_polygon_oob_multi_edge_crossing_remains_one_object_event() -> None:
    scene = _l_room_scene(object_center=[1.55, 1.55, 0.5])
    scene["objects"][0]["size"] = [0.8, 0.8, 1.0]

    report = check_polygon_oob(scene, {"detector_only": True})

    record = report["objects"][0]
    assert report["candidate_oob_count"] == 1
    assert set(record["violated_wall_ids"]) == {
        "room_000.wall_002",
        "room_000.wall_003",
    }
    assert len(record["violated_edges"]) == 2


def test_oob_edge_camera_bank_is_polygon_feasible() -> None:
    scene = _l_room_scene(object_center=[1.55, 2.0, 0.5])
    record = check_polygon_oob(scene, {"detector_only": True})["objects"][0]
    request = {
        "metric": "oob",
        "scene": scene,
        "object_ids": ["target"],
        "event": {"violated_edges": record["violated_edges"]},
        "detector_evidence": {
            "violated_edges": record["violated_edges"]
        },
    }

    candidates = generate_camera_pose_candidates(request, max_candidates=6)
    geometry = PolygonRoomGeometry.from_metadata(
        scene["metadata"]["non_rectangular_room_geometry"]
    )

    assert candidates
    assert all(
        candidate["focus_wall_id"] == "room_000.wall_003"
        for candidate in candidates
    )
    assert all(
        geometry.contains_xy(
            candidate["location"][:2],
            wall_clearance_m=0.02,
        )
        for candidate in candidates
    )
    assert all(
        geometry.segment_visible_inside_room(
            candidate["location"][:2],
            candidate["target"][:2],
        )
        for candidate in candidates
    )


def test_oob_empty_nominal_bank_uses_bounded_focus_inset_not_score_exemption() -> None:
    polygon = [[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0]]
    walls = [
        {
            "wall_id": f"wall_{index:03d}",
            "start_xy": start,
            "end_xy": end,
            "inward_normal_xy": normal,
            "height_m": 2.8,
            "thickness_m": 0.10,
        }
        for index, (start, end, normal) in enumerate(
            zip(
                polygon,
                polygon[1:] + polygon[:1],
                ([0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]),
            )
        )
    ]
    scene = {
        "schema_version": "canonical_scene_v1",
        "scene_id": "tight-bed",
        "request_id": "tight-bed",
        "scene_type": "bedroom",
        "boundary": polygon,
        "scene_height": 2.8,
        "objects": [
            {
                "id": "bed",
                "category": "bed",
                "center": [0.9998, 1.5, 0.4],
                "size": [2.0, 2.0, 0.8],
                "rotation": [0.0, 0.0, 0.0],
            }
        ],
        "metadata": {
            "evaluation_mode": "non_rectangular_multi_room",
            "non_rectangular_room_geometry": {
                "schema_version": "non_rectangular_polygon_room_geometry_v1",
                "room_id": "room_000",
                "floor_polygon_xy": polygon,
                "wall_segments": walls,
                "floor_z_m": 0.0,
                "ceiling_z_m": 2.8,
                "tolerance_m": 1.0e-6,
            },
        },
    }
    record = check_polygon_oob(scene, {"detector_only": True})["objects"][0]
    request = {
        "metric": "oob",
        "scene": scene,
        "object_ids": ["bed"],
        "event": {"violated_edges": record["violated_edges"]},
        "detector_evidence": {
            "violated_edges": record["violated_edges"]
        },
    }

    candidates = generate_camera_pose_candidates(request, max_candidates=6)

    assert 0.0 < record["maximum_horizontal_penetration_m"] < 0.001
    assert record["candidate_oob"] is True
    assert candidates
    assert candidates[0]["repair_tier"] == "r3_last_local"
    assert candidates[0]["camera_focus_inset_m"] == 0.10
    assert candidates[0]["camera_type"] == "ORTHO"
    assert candidates[0]["orthographic_framing_center_xy"] != (
        candidates[0]["edge_inset_focus_xy"]
    )
    assert candidates[0]["proxy_framing"][
        "wall_crossing_included_in_span"
    ] is True
    audit = request["_nonrect_camera_candidate_audit"]
    assert audit["selected_tier"] == "r3_last_local"
    assert audit["maximum_focus_inset_m"] == 0.10
    assert all(
        tier["accepted_candidate_count"] == 0
        for tier in audit["tiers"][:3]
    )


class _GeometryOnlyP0BModel:
    model_id = "mock-geometry-only"
    endpoint = "http://localhost.invalid/v1"
    response_format_json = True

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.last_request_metadata: dict[str, Any] = {}

    def chat_messages(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.calls.append({"messages": deepcopy(messages), "kwargs": dict(kwargs)})
        self.last_request_metadata = {"usage": {"total_tokens": 1}}
        return json.dumps(
            {
                "verdict": "invalid",
                "confidence": 0.8,
                "reason": "authoritative polygon geometry records a crossing",
            }
        )


class _MalformedThenBinaryP0BModel(_GeometryOnlyP0BModel):
    def chat_messages(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.calls.append({"messages": deepcopy(messages), "kwargs": dict(kwargs)})
        self.last_request_metadata = {"usage": {"total_tokens": 1}}
        if len(self.calls) == 1:
            return json.dumps(
                {
                    "status": "invalid",
                    "confidence": 0.8,
                    "reason": "wrong binary field name",
                }
            )
        return json.dumps(
            {
                "verdict": "invalid",
                "confidence": 0.8,
                "reason": "schema-repaired binary response",
            }
        )


def test_nonrect_oob_zero_local_visual_forces_geometry_binary() -> None:
    scene = _l_room_scene(object_center=[1.55, 2.0, 0.5])
    model = _GeometryOnlyP0BModel()

    report = check_polygon_oob(
        scene,
        vlm_judge=OpenAICompatibleVLMJudge(model),
        local_view_provider=lambda _request: [],
    )

    assert report["status"] == "checked"
    assert report["invalid_object_count"] == 1
    assert len(model.calls) == 1
    assert model.calls[0]["kwargs"]["call_type"] == (
        "vlm_judge.p0b.oob.forced_choice"
    )
    assert len(model.calls[0]["messages"][1]["content"]) == 1
    target = next(
        item for item in report["objects"] if item["object_id"] == "target"
    )
    forced = target["judge_result"]["judgement"][
        "budget_exhaustion_forced_choice"
    ]
    assert forced["applied"] is True
    assert forced["available_image_count"] == 0
    assert forced["decision_source"] == (
        "nonrect_geometry_or_retained_visual_forced_binary"
    )


def test_nonrect_forced_p0b_repairs_binary_schema_once() -> None:
    scene = _l_room_scene(object_center=[1.55, 2.0, 0.5])
    model = _MalformedThenBinaryP0BModel()

    report = check_polygon_oob(
        scene,
        vlm_judge=OpenAICompatibleVLMJudge(model),
        local_view_provider=lambda _request: [],
    )

    assert report["status"] == "checked"
    assert report["invalid_object_count"] == 1
    assert len(model.calls) == 2
    assert model.calls[0]["kwargs"]["call_type"].endswith(".forced_choice")
    assert model.calls[1]["kwargs"]["call_type"].endswith(
        ".forced_choice.schema_repair"
    )


def test_rectangular_forced_p0b_path_does_not_gain_nonrect_schema_repair() -> None:
    model = _MalformedThenBinaryP0BModel()
    judge = OpenAICompatibleVLMJudge(model)
    with pytest.raises(ValueError, match="verdict must be exactly"):
        judge._adjudicate_p0b_raw(
            {
                "metric": "oob",
                "render_evidence": [],
                "budget_exhaustion_finalization": {
                    "required": True,
                    "trigger_stop_reason": "max_evidence_rounds_exhausted",
                    "ambiguity_before_forcing": True,
                },
            },
            _allow_need_more_evidence=False,
        )
    assert len(model.calls) == 1


def test_nonrect_support_zero_local_routes_geometry_to_vlm_before_if_else() -> None:
    scene = _l_room_scene(object_center=[0.35, 2.0, 1.5])
    model = _GeometryOnlyP0BModel()

    report = check_support(
        scene,
        vlm_judge=OpenAICompatibleVLMJudge(model),
        local_view_provider=lambda _request: [],
    )

    target = next(
        item for item in report["objects"] if item["object_id"] == "target"
    )
    assert report["status"] == "checked"
    assert target["route"] == "vlm_adjudicated"
    assert target["final_verdict"] == "invalid"
    assert "zero_visual_support_fallback" not in target
    assert model.calls[0]["kwargs"]["call_type"] == (
        "vlm_judge.p0b.support.forced_choice"
    )
    forced = target["judge_result"]["judgement"][
        "budget_exhaustion_forced_choice"
    ]
    assert forced["applied"] is True
    assert forced["available_image_count"] == 0


def test_nonrect_collision_zero_local_forces_geometry_binary() -> None:
    scene = _l_room_scene(object_center=[0.65, 0.8, 0.5])
    model = _GeometryOnlyP0BModel()

    report = check_collision(
        scene,
        vlm_judge=OpenAICompatibleVLMJudge(model),
        local_view_provider=lambda _request: [],
    )

    assert report["status"] == "checked"
    assert report["collision_pair_count"] == 1
    assert model.calls[0]["kwargs"]["call_type"] == (
        "vlm_judge.p0b.collision.forced_choice"
    )
    pair = report["pairs"][0]
    assert pair["route"] == "vlm_adjudicated"
    assert pair["judge_result"]["judgement"][
        "budget_exhaustion_forced_choice"
    ]["applied"] is True


def test_zero_visual_deterministic_l1_fallbacks_are_binary_and_audited() -> None:
    collision_scene = _l_room_scene(object_center=[0.65, 0.8, 0.5])
    collision = _finalize_nonrect_l1_continuity(
        "collision",
        check_collision(collision_scene),
    )
    assert collision["status"] == "checked"
    collision_fallbacks = [
        item for item in collision["pairs"]
        if item.get("route") == "deterministic_zero_visual_fallback"
    ]
    assert collision_fallbacks
    assert all(
        item["final_verdict"] in {"valid", "invalid"}
        for item in collision_fallbacks
    )

    oob = _finalize_nonrect_l1_continuity(
        "oob",
        check_polygon_oob(_l_room_scene(object_center=[1.55, 2.0, 0.5])),
    )
    assert oob["status"] == "checked"
    target_oob = next(
        item for item in oob["objects"] if item["object_id"] == "target"
    )
    assert target_oob["final_verdict"] == "invalid"
    assert target_oob["nonrect_evidence_continuity"]["matched_rule"] == (
        "verified_polygon_or_floor_crossing"
    )

    support_scene = _l_room_scene(object_center=[0.65, 2.5, 1.5])
    support = _finalize_nonrect_l1_continuity(
        "support",
        check_support(support_scene),
    )
    assert support["status"] == "checked"
    target_support = next(
        item for item in support["objects"] if item["object_id"] == "target"
    )
    assert target_support["final_verdict"] == "invalid"
    assert target_support["nonrect_evidence_continuity"]["degraded"] is True

    detector_only = check_polygon_oob(
        _l_room_scene(object_center=[1.55, 2.0, 0.5]),
        {"detector_only": True},
    )
    assert _finalize_nonrect_l1_continuity(
        "oob",
        detector_only,
    )["status"] == "detector_only"


def test_l1_continuity_closes_safe_events_but_retains_hard_event_failure() -> None:
    report = _finalize_nonrect_l1_continuity(
        "oob",
        {
            "status": "requires_vlm",
            "score": None,
            "num_objects": 2,
            "objects": [
                {
                    "object_id": "hard",
                    "requires_vlm": True,
                    "final_verdict": None,
                    "horizontal_oob": True,
                    "floor_oob": False,
                    "adjudication_error": (
                        "ResponseSchemaRepairError: binary schema invalid"
                    ),
                },
                {
                    "object_id": "safe",
                    "requires_vlm": True,
                    "final_verdict": None,
                    "horizontal_oob": False,
                    "floor_oob": False,
                    "adjudication_error": None,
                },
            ],
        },
    )

    assert report["status"] == "requires_vlm"
    assert report["score"] is None
    hard, safe = report["objects"]
    assert hard["final_verdict"] is None
    assert hard["nonrect_evidence_continuity"] == {
        "schema_version": "nonrect_zero_visual_binary_v1",
        "applied": False,
        "decision_source": None,
        "fallback_rejection_reason": "judge_response_contract_failure",
        "event_key": "oob:hard",
        "visual_evidence_count": 0,
        "geometry_contract_version": (
            "non_rectangular_polygon_room_geometry_v1"
        ),
        "degraded": True,
    }
    assert safe["final_verdict"] == "valid"
    assert safe["nonrect_evidence_continuity"]["applied"] is True
    assert report["deterministic_zero_visual_fallback_count"] == 1
    assert report["deterministic_fallback_rejected_count"] == 1
    assert report["unresolved_event_keys"] == ["oob:hard"]


def test_metric_diagnostics_persist_before_failed_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = prepare_non_rectangular_evaluation(_input())
    unit = build_room_evaluation_units(preflight)[0]

    def incomplete_collision(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {
            "metric": "collision",
            "status": "requires_vlm",
            "score": None,
            "num_objects": unit.generated_object_count,
            "collision_count": 0,
            "pairs": [
                {
                    "object_a": unit.object_ids[0],
                    "object_b": unit.object_ids[-1],
                    "requires_vlm": True,
                    "final_verdict": None,
                    "route": "vlm_adjudication_failed",
                    "adjudication_error": (
                        "ResponseSchemaRepairError: secret response body"
                    ),
                }
            ],
        }

    monkeypatch.setattr(
        nonrect_evaluator_module,
        "check_collision",
        incomplete_collision,
    )
    evaluator = CanonicalNonRectangularRoomEvaluator(
        output_root=tmp_path,
        runtime_by_room=_grouping_runtime(preflight),
        scene_quality_evaluator=_fake_scene_quality,
    )

    with pytest.raises(NonRectangularRoomMetricIncomplete) as raised:
        evaluator.evaluate(unit)

    assert raised.value.metric_id == "collision"
    assert raised.value.failure_category == "judge_response_contract_failure"
    assert raised.value.source_status == "requires_vlm"
    diagnostic_path = (
        tmp_path
        / "rooms"
        / unit.room_id
        / "metric_diagnostics/collision.json"
    )
    diagnostic = json.loads(diagnostic_path.read_text())
    assert diagnostic["metric_id"] == "collision"
    assert diagnostic["source_status"] == "requires_vlm"
    assert diagnostic["failure_category"] == (
        "judge_response_contract_failure"
    )
    assert diagnostic["unresolved_event_keys"]
    assert diagnostic["records"][0]["adjudication_error_type"] == (
        "ResponseSchemaRepairError"
    )
    assert "secret response body" not in diagnostic_path.read_text()


def test_support_uses_polygon_walls_and_excludes_ceiling() -> None:
    scene = _l_room_scene(object_center=[0.35, 2.0, 0.5])

    report = check_support(scene)
    record = report["objects"][0]

    assert report["status"] == "checked"
    assert report["architecture_scope"] == (
        "floor_wall_object_support_no_ceiling"
    )
    assert report["logical_architecture_attachment_policy"][
        "ceiling_attachment_enabled"
    ] is False
    assert record["nearest_logical_wall_measurement"]["wall_id"] == (
        "room_000.wall_005"
    )
    assert all(
        item["mode"] == "wall_attachment"
        for item in record["architecture_contact_candidates"]
    )


def test_support_preserves_nonzero_global_floor_height() -> None:
    scene = _l_room_scene(object_center=[0.35, 2.0, 1.5])
    geometry = scene["metadata"]["non_rectangular_room_geometry"]
    geometry["floor_z_m"] = 1.0
    geometry["ceiling_z_m"] = 3.8
    scene["scene_height"] = 3.8

    report = check_support(scene)

    assert report["status"] == "checked"
    assert report["score"] == 1.0
    record = report["objects"][0]
    floor_hits = [
        hit
        for hit in record["representative_samples"]
        if hit.get("target") == "floor"
    ]
    assert floor_hits
    assert all(hit["position"][2] == 1.0 for hit in floor_hits)


def test_polygon_camera_gate_prunes_aabb_void_and_context_anchor_is_inside() -> None:
    scene = _l_room_scene(object_center=[0.65, 2.5, 0.5])
    request = {
        "metric": "collision",
        "scene": scene,
        "object_ids": ["target", "context"],
        "event": {"object_a": "target", "object_b": "context"},
    }
    geometry = PolygonRoomGeometry.from_metadata(
        scene["metadata"]["non_rectangular_room_geometry"]
    )

    candidates = generate_camera_pose_candidates(request, max_candidates=6)
    global_poses = generate_global_context_poses(scene)

    assert candidates
    assert all(
        candidate["polygon_geometry_gate"]["room_wall_los_checked"]
        for candidate in candidates
    )
    assert all(
        geometry.contains_xy(candidate["location"][:2])
        for candidate in candidates
    )
    perspective = next(
        pose for pose in global_poses if pose["id"] == "global_perspective"
    )
    assert geometry.contains_xy(perspective["location"][:2])


def test_global_context_degrades_to_top_when_perspective_search_is_empty(
    monkeypatch: Any,
) -> None:
    scene = _l_room_scene(object_center=[0.65, 2.5, 0.5])
    monkeypatch.setattr(
        PolygonRoomGeometry,
        "place_on_feasible_ray",
        lambda self, **kwargs: None,
    )

    poses = generate_global_context_poses(scene)

    assert [item["id"] for item in poses] == ["global_top"]
    assert poses[0]["polygon_context"]["perspective_status"] == (
        "unavailable_after_bounded_polygon_search"
    )


def test_active_camera_action_is_revalidated_against_polygon() -> None:
    scene = _l_room_scene(object_center=[0.65, 2.5, 0.5])
    request = {
        "metric": "scale_consistency",
        "scene": scene,
        "object_ids": ["target"],
    }
    pose = generate_camera_pose_candidates(request, max_candidates=4)[0]
    geometry = PolygonRoomGeometry.from_metadata(
        scene["metadata"]["non_rectangular_room_geometry"]
    )

    moved = apply_camera_action(pose, "orbit_left", scene=scene)

    assert moved["camera_action"] == "orbit_left"
    assert moved["polygon_geometry_gate"]["wall_clearance_checked"] is True
    assert geometry.contains_xy(moved["location"][:2])
    assert geometry.segment_visible_inside_room(
        moved["location"][:2],
        moved["target"][:2],
    )


def test_room_projection_drops_generator_private_plan_and_keeps_coordinates() -> None:
    unit = build_room_evaluation_units(
        prepare_non_rectangular_evaluation(_input())
    )[0]

    scene = project_room_unit_to_canonical_scene(unit)

    assert scene["boundary"] == [list(point) for point in unit.floor_polygon_xy]
    assert scene["objects"][0]["center"] == list(
        unit.generated_objects[0]["center"]
    )
    assert "task_slot" not in json.dumps(scene, sort_keys=True)
    assert scene["metadata"]["room_scope"] == (
        "current_room_objects_and_walls_only"
    )


def _fake_scene_quality(
    scene: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    del kwargs
    return {
        "status": "evaluated",
        "metrics": {
            metric: {
                "metric": metric,
                "status": "evaluated",
                "score": 0.9,
                "object_ids": [item["id"] for item in scene["objects"]],
            }
            for metric in L3_METRICS
        },
    }


def _fake_scene_quality_no_visual(
    scene: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    del scene, kwargs
    return {
        "status": "incomplete",
        "metrics": {
            metric: {
                "metric": metric,
                "status": "failed",
                "score": None,
                "terminal_state": "infrastructure_failure",
                "reason": "group_local_render_evidence_unavailable",
            }
            for metric in L3_METRICS
        },
    }


def _grouping_runtime(preflight: Any) -> dict[str, dict[str, Any]]:
    return {
        unit.room_id: {
            "object_grouping_report": {
                "object_groups": [
                    {
                        "group_id": f"{unit.room_id}.group_000",
                        "object_ids": list(unit.object_ids),
                    }
                ]
            }
        }
        for unit in build_room_evaluation_units(preflight)
    }


class _NoAPICanonicalJudge:
    vlm_control_enabled = False

    def screen_scene_quality(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        return self._response(request)

    def adjudicate_scene_quality(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        return self._response(request)

    @staticmethod
    def _response(request: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "evidence_status": "sufficient",
            "verdict": "valid",
            "confidence": 0.9,
            "reason": "no-API canonical mock",
            "missing_evidence": [],
            "defects": [],
        }
        functional = request.get("required_functional_checks")
        if isinstance(functional, list):
            result["functional_check_results"] = [
                {
                    "check_id": str(item.get("check_id") or ""),
                    "target_ids": list(item.get("target_ids") or []),
                    "observation_status": "observed",
                    "conclusion": "valid",
                    "reason": "no-API canonical mock",
                }
                for item in functional
            ]
        placement = request.get("required_placement_checks")
        if isinstance(placement, list):
            result["placement_check_results"] = [
                {
                    "check_id": str(item.get("check_id") or ""),
                    "subject_id": str(item.get("subject_id") or ""),
                    "context_ids": list(item.get("context_ids") or []),
                    "observation_status": "observed",
                    "conclusion": "valid",
                    "reason": "no-API canonical mock",
                }
                for item in placement
            ]
        return result


def test_concrete_room_evaluator_runs_no_api_full_workflow(tmp_path: Path) -> None:
    preflight = prepare_non_rectangular_evaluation(_input())
    runtime_by_room = _grouping_runtime(preflight)
    evaluator = CanonicalNonRectangularRoomEvaluator(
        output_root=tmp_path,
        runtime_by_room=runtime_by_room,
        scene_quality_evaluator=_fake_scene_quality,
    )

    report = run_internal_non_rectangular_evaluation(
        _input(),
        room_evaluator=evaluator,
    )

    assert report["terminal_status"] == "complete"
    assert report["coverage"]["complete_room_count"] == 2
    for room in report["rooms"].values():
        assert tuple(room["report"]["metrics"]) == (*L1_METRICS, *L3_METRICS)
    assert (tmp_path / "rooms/room_000/canonical_room_scene.json").is_file()
    assert (tmp_path / "rooms/room_001/room_evaluation_report.json").is_file()


def test_no_visual_mock_whole_workflow_finishes_with_audited_binary_fallback(
    tmp_path: Path,
) -> None:
    preflight = prepare_non_rectangular_evaluation(_input())
    evaluator = CanonicalNonRectangularRoomEvaluator(
        output_root=tmp_path,
        runtime_by_room=_grouping_runtime(preflight),
        evidence_continuity_context={
            "global_evidence": {
                "status": "unavailable_recovered",
                "visual_evidence_count": 0,
            }
        },
        scene_quality_evaluator=_fake_scene_quality_no_visual,
    )

    report = run_internal_non_rectangular_evaluation(
        _input(),
        room_evaluator=evaluator,
    )

    assert report["terminal_status"] == "complete"
    assert report["coverage"]["complete_room_count"] == 2
    for room in report["rooms"].values():
        for metric in L3_METRICS:
            normalized = room["report"]["metrics"][metric]
            assert normalized["status"] == "complete"
            assert normalized["score"] == 1.0
            fallback = normalized["raw_report"][
                "nonrect_evidence_continuity"
            ]
            assert fallback["forced_binary"] is True
            assert fallback["defaulted"] is True
            assert fallback["visual_evidence_count"] == 0
            assert fallback["degraded"] is True


def test_actual_scene_quality_pipeline_does_not_stop_on_zero_visual_evidence(
    tmp_path: Path,
) -> None:
    preflight = prepare_non_rectangular_evaluation(_input())
    evaluator = CanonicalNonRectangularRoomEvaluator(
        output_root=tmp_path,
        runtime_by_room=_grouping_runtime(preflight),
        vlm_judge=_NoAPICanonicalJudge(),
        evidence_continuity_context={
            "global_evidence": {
                "status": "unavailable_recovered",
                "visual_evidence_count": 0,
            }
        },
    )

    report = run_internal_non_rectangular_evaluation(
        _input(),
        room_evaluator=evaluator,
    )

    assert report["terminal_status"] == "complete"
    assert report["coverage"]["infrastructure_failure_count"] == 0
    for room in report["rooms"].values():
        metrics = room["report"]["metrics"]
        assert tuple(metrics) == (*L1_METRICS, *L3_METRICS)
        assert all(item["status"] == "complete" for item in metrics.values())
        for metric in (
            "style_consistency",
            "functional_consistency",
            "semantic_placement_consistency",
        ):
            fallback = metrics[metric]["raw_report"][
                "nonrect_evidence_continuity"
            ]
            assert fallback["decision_source"] == (
                "deterministic_zero_visual_fallback"
            )
            assert fallback["visual_evidence_count"] == 0
            assert fallback["degraded"] is True


def test_api_failure_is_not_converted_by_nonrect_evidence_fallback(
    tmp_path: Path,
) -> None:
    preflight = prepare_non_rectangular_evaluation(_input())

    def api_failure_scene_quality(
        scene: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        del scene, kwargs
        metrics = _fake_scene_quality_no_visual({},)["metrics"]
        metrics["functional_consistency"] = {
            "metric": "functional_consistency",
            "status": "failed",
            "score": None,
            "terminal_state": "infrastructure_failure",
            "reason": "vlm_judge_failed",
            "judgement": {
                "error_type": "EndpointHTTPError",
                "error": "HTTP transport failed",
            },
        }
        return {"status": "incomplete", "metrics": metrics}

    evaluator = CanonicalNonRectangularRoomEvaluator(
        output_root=tmp_path,
        runtime_by_room=_grouping_runtime(preflight),
        scene_quality_evaluator=api_failure_scene_quality,
    )

    report = run_internal_non_rectangular_evaluation(
        _input(),
        room_evaluator=evaluator,
    )

    assert report["terminal_status"] == "incomplete"
    assert report["coverage"]["complete_room_count"] == 0
    assert report["coverage"]["infrastructure_failure_count"] == 2

    raw_oob = check_polygon_oob(
        _l_room_scene(object_center=[1.55, 2.0, 0.5])
    )
    target = next(
        item for item in raw_oob["objects"]
        if item["object_id"] == "target"
    )
    target["adjudication_error"] = (
        "EndpointMalformedResponseError: binary schema invalid"
    )
    unchanged = _finalize_nonrect_l1_continuity("oob", raw_oob)
    assert unchanged["status"] == "requires_vlm"
    assert next(
        item for item in unchanged["objects"]
        if item["object_id"] == "target"
    )["final_verdict"] is None


class _CompleteFakeRoomEvaluator:
    def evaluate(self, unit: Any) -> dict[str, Any]:
        metrics: dict[str, dict[str, Any]] = {}
        for metric in (*L1_METRICS, *L3_METRICS):
            metrics[metric] = {
                "metric": metric,
                "status": "complete",
                "score": 1.0,
                "evaluated_object_count": unit.generated_object_count,
                "raw_report": {"fake": True},
                **({"invalid_count": 0} if metric in L1_METRICS else {}),
            }
        return {
            "schema_version": ROOM_REPORT_SCHEMA_VERSION,
            "room_id": unit.room_id,
            "status": "complete",
            "metrics": metrics,
        }


def test_public_mode_route_writes_report_without_canonical_dispatch(
    tmp_path: Path,
) -> None:
    out = tmp_path / "report.json"

    report = run_evaluate(
        evaluation_mode="non_rectangular_multi_room",
        evaluation_input=_input(),
        room_evaluator=_CompleteFakeRoomEvaluator(),
        out=out,
    )

    assert out.is_file()
    assert report["terminal_status"] == "complete"
    assert report["provenance"]["public_route_connected"] is True
    assert report["aggregate"]["overall_score"] == 1.0
    assert report["aggregate"]["scoring_profile"]["profile_id"] == (
        "non_rectangular_room_weighted_v1"
    )


def test_rectangular_camera_and_support_stay_on_original_branches() -> None:
    scene = {
        "schema_version": "canonical_scene_v1",
        "scene_id": "rectangle",
        "request_id": "rectangle",
        "scene_type": "room",
        "boundary": [[0, 0], [4, 0], [4, 3], [0, 3]],
        "scene_height": 2.8,
        "objects": [
            {
                "id": "chair",
                "category": "chair",
                "center": [1.0, 1.0, 0.5],
                "size": [0.6, 0.6, 1.0],
                "rotation": [0.0, 0.0, 0.0],
            }
        ],
    }
    request = {
        "metric": "scale_consistency",
        "scene": deepcopy(scene),
        "object_ids": ["chair"],
    }

    candidates = generate_camera_pose_candidates(request, max_candidates=2)
    moved = apply_camera_action(candidates[0], "orbit_left", scene=scene)
    globals_ = generate_global_context_poses(scene)
    support = check_support(scene)

    assert all(
        candidate["feasibility"]["method"] == "ray_box_interval_v2"
        for candidate in candidates
    )
    assert all("polygon_geometry_gate" not in item for item in candidates)
    assert "polygon_geometry_gate" not in moved
    assert globals_[0]["policy_source"] == "frozen_global_context_v1"
    assert "architecture_scope" not in support
    assert "ceiling_attachment_enabled" not in support[
        "logical_architecture_attachment_policy"
    ]

    class RectangularJudge:
        vlm_control_enabled = False

        def __init__(self) -> None:
            self.request: dict[str, Any] | None = None

        def adjudicate_p0b(
            self,
            request: dict[str, Any],
        ) -> dict[str, Any]:
            self.request = deepcopy(request)
            return {
                "verdict": "valid",
                "confidence": 0.5,
                "reason": "rectangular compatibility fixture",
            }

    rectangular_judge = RectangularJudge()
    adjudicate_p0b_event(
        metric="collision",
        event={"object_a": "chair", "object_b": "chair"},
        prompt="",
        relationships=[],
        scene=scene,
        detector_evidence={},
        judge=rectangular_judge,
        object_ids=["chair"],
        local_view_provider=lambda _request: [],
    )
    assert rectangular_judge.request is not None
    assert "nonrect_evidence_continuity" not in rectangular_judge.request
    assert "budget_exhaustion_finalization" not in rectangular_judge.request
