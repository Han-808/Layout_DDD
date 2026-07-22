from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark.evaluator import evaluate_generic_validity
from benchmark.evaluator.generic_validity.mesh_geometry import write_ascii_triangle_ply
from benchmark.evaluator.generic_validity.support import (
    DIRECT_CONTACT_TOLERANCE_CAP_M,
    SUPPORT_CANDIDATE_SELECTION_POLICY,
    SUPPORT_EVALUATOR_VERSION,
    SUPPORT_VLM_INSTRUCTION,
    SupportEvaluationError,
    _hard_contact_tolerance,
    _near_support_tolerance,
    check_support,
    disabled_support_report,
)
from benchmark.utils.io import read_json


ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _scene(objects: list[dict], boundary: list[list[float]] | None = None, height: float = 2.8) -> dict:
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": "support_test",
        "scene_type": "room",
        "boundary": boundary or [[0, 0], [4, 0], [4, 3], [0, 3]],
        "scene_height": height,
        "objects": objects,
        "metadata": {
            "coordinate_frame": {
                "origin": "room_min_corner_floor",
                "axes": "x_width_y_depth_z_up",
                "unit": "meter",
                "rotation_unit": "degree",
            }
        },
    }


def _obj(
    object_id: str,
    center: list[float],
    size: list[float] | None = None,
    *,
    category: str = "box",
    description: str = "box",
) -> dict:
    return {
        "id": object_id,
        "category": category,
        "description": description,
        "center": center,
        "size": size or [0.5, 0.5, 1.0],
        "rotation": [0, 0, 0],
    }


def _canonical_obj(object_id: str, center: list[float], size: list[float] | None = None) -> dict:
    jid = f"{object_id}_asset"
    resolved_size = size or [0.5, 0.5, 1.0]
    return {
        "id": object_id,
        "jid": jid,
        "category": "box",
        "description": "box",
        "retrieval_category": "box",
        "desc": "box",
        "short_desc": "box",
        "center": center,
        "size": resolved_size,
        "rotation": [0, 0, 0],
        "asset_ref": {"source_db": "imaginarium", "asset_key": jid, "mesh_uri": None, "pointcloud_uri": None, "metadata_uri": None},
        "asset_proxy": {"type": "obb_from_metadata_or_csv", "bbox_center_local": [0, 0, 0], "bbox_size": resolved_size},
        "metadata": {"interactive": False},
    }


def _by_id(report: dict, object_id: str) -> dict:
    return next(obj for obj in report["objects"] if obj["object_id"] == object_id)


class _Judge:
    """Minimal binary P0b judge mock recording the requests it receives."""

    model_id = "judge"
    endpoint = "http://127.0.0.1:8298/v1"

    def __init__(self, verdict: str = "valid") -> None:
        self.verdict = verdict
        self.requests: list[dict] = []

    def adjudicate_p0b(self, request: dict) -> dict:
        self.requests.append(request)
        return {"verdict": self.verdict, "confidence": 0.9, "reason": "mock"}


def _box_ply(path: Path, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> None:
    vertices = [
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ]
    faces = [
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 6, 5], [4, 7, 6],  # top
        [0, 4, 5], [0, 5, 1],
        [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3],
        [3, 7, 4], [3, 4, 0],
    ]
    write_ascii_triangle_ply(path, vertices, faces)


def _boxes_ply(path: Path, boxes: list[tuple[float, float, float, float, float, float]]) -> None:
    vertices = []
    faces = []
    for x0, x1, y0, y1, z0, z1 in boxes:
        offset = len(vertices)
        vertices.extend(
            [
                [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
            ]
        )
        faces.extend(
            [
                [offset + 0, offset + 1, offset + 2], [offset + 0, offset + 2, offset + 3],
                [offset + 4, offset + 6, offset + 5], [offset + 4, offset + 7, offset + 6],
                [offset + 0, offset + 4, offset + 5], [offset + 0, offset + 5, offset + 1],
                [offset + 1, offset + 5, offset + 6], [offset + 1, offset + 6, offset + 2],
                [offset + 2, offset + 6, offset + 7], [offset + 2, offset + 7, offset + 3],
                [offset + 3, offset + 7, offset + 4], [offset + 3, offset + 4, offset + 0],
            ]
        )
    write_ascii_triangle_ply(path, vertices, faces)


def _mesh_manifest(tmp_path: Path, object_id: str, ply: Path) -> dict:
    return {
        "schema_version": "collision_geometry_v1",
        "units": "meter",
        "up_axis": "z",
        "manifest_path": str(tmp_path / "collision_geometry.json"),
        "objects": {
            object_id: {
                "representation": "triangle_mesh",
                "geometry_path": str(ply),
                "transform_baked": True,
                "geometry_source": "asset_fbx",
                "complete": True,
            }
        },
    }


# --------------------------------------------------------------------------- #
# 1. Any reliable floor contact direct-passes without VLM
# --------------------------------------------------------------------------- #
def test_floor_contact_direct_passes_without_vlm() -> None:
    judge = _Judge("invalid")
    report = check_support(_scene([_obj("box", [1.0, 1.0, 0.5])]), vlm_judge=judge)
    record = report["objects"][0]

    assert record["contact_fraction"] == 1.0
    assert record["route"] == "direct_valid_contact"
    assert record["final_verdict"] == "valid"
    assert record["requires_vlm"] is False
    assert record["support_targets"] == ["floor"]
    assert judge.requests == []
    assert report["status"] == "checked"
    assert report["score"] == 1.0
    assert report["evaluator_version"] == SUPPORT_EVALUATOR_VERSION
    assert report["threshold_mode"] == "scale_aware_tolerance_contact"
    assert report["direct_contact_tolerance_bounds_m"] == [0.02, 0.035]
    assert report["direct_contact_tolerance_cap_m"] == pytest.approx(
        DIRECT_CONTACT_TOLERANCE_CAP_M
    )
    assert report["fixed_contact_tolerance_m"] is None
    assert report["legacy_contact_tolerance_affects_direct_valid"] is False
    assert report["hard_contact_tolerance_bounds_m"] == [0.02, 0.035]
    assert report["near_support_tolerance_bounds_m"] == [0.04, 0.08]
    assert report["candidate_selection_policy"] == "high_recall_candidate_no_label_prior"
    assert report["base_band_tolerance_m"] == 0.02
    assert report["minimum_contact_count"] == 1
    assert report["contact_fraction_affects_route"] is False
    assert record["base_contact_fraction"] == record["contact_fraction"]
    # box size_z=1.0 -> hard = clamp(0.02 + 0.005*1.0, 0.02, 0.035) = 0.025
    assert record["size_z_m"] == pytest.approx(1.0)
    assert record["hard_contact_tolerance_m"] == pytest.approx(0.025)
    assert record["contact_tolerance_m"] == pytest.approx(0.025)
    assert record["direct_contact_tolerance_m"] == pytest.approx(0.025)
    assert record["support_candidate_tolerance_m"] == pytest.approx(0.025)
    assert record["near_support_tolerance_m"] == pytest.approx(0.06)
    assert record["gap_band"] == "contact"


# --------------------------------------------------------------------------- #
# 2. Reliable object-on-object contact direct-passes
# --------------------------------------------------------------------------- #
def test_object_on_object_direct_passes() -> None:
    judge = _Judge("invalid")
    scene = _scene([_obj("table", [1.0, 1.0, 0.5]), _obj("book", [1.0, 1.0, 1.1], [0.3, 0.3, 0.2])])
    report = check_support(scene, vlm_judge=judge)
    book = _by_id(report, "book")

    assert book["route"] == "direct_valid_contact"
    assert book["final_verdict"] == "valid"
    assert book["support_targets"] == ["table"]
    assert book["certified_grounded_support"] is True
    assert book["grounding_status"] == "certified_tolerance_contact_path_to_floor"
    assert book["grounded_support_path"] == ["book", "table", "floor"]
    assert book["evidence_level"] == "obb"
    assert judge.requests == []
    assert report["score"] == 1.0


def test_floating_stack_local_contact_routes_without_grounded_ancestry() -> None:
    scene = _scene(
        [
            _obj("floating_base", [1.0, 1.0, 0.75], [1.0, 1.0, 1.0]),
            _obj("top", [1.0, 1.0, 1.35], [0.3, 0.3, 0.2]),
        ]
    )

    report = check_support(scene, {"detector_only": True})
    base = _by_id(report, "floating_base")
    top = _by_id(report, "top")

    assert base["requires_vlm"] is True
    assert top["contact_hit_count"] > 0
    assert top["support_targets"] == ["floating_base"]
    assert top["certified_grounded_support"] is False
    assert top["grounding_status"] == "local_object_contact_without_ground_path"
    assert top["grounded_support_path"] == []
    assert "object_contact_without_certified_ground_path" in top["routing_reasons"]
    assert top["requires_vlm"] is True
    assert top["route"] is None
    assert set(report["unresolved_grounding_object_ids"]) == {"floating_base", "top"}


def test_ungrounded_tolerance_contact_cycle_routes_to_vlm() -> None:
    # Two thin, coincident boxes see each other's surface within the legacy
    # candidate band. Their local gaps count as tolerance contacts, but the
    # component has no floor seed and therefore cannot direct-pass.
    scene = _scene(
        [
            _obj("cycle_a", [1.0, 1.0, 1.0], [0.5, 0.5, 0.02]),
            _obj("cycle_b", [1.0, 1.0, 1.0], [0.5, 0.5, 0.02]),
        ]
    )

    report = check_support(scene, {"detector_only": True})
    for object_id in ("cycle_a", "cycle_b"):
        record = _by_id(report, object_id)
        assert record["contact_hit_count"] > 0
        assert record["certified_grounded_support"] is False
        assert record["ungrounded_contact_cycle_reachable"] is True
        assert "object_contact_without_certified_ground_path" in record["routing_reasons"]
        assert "ungrounded_contact_cycle" in record["routing_reasons"]
        assert record["requires_vlm"] is True


def test_floor_contact_remains_direct_valid_seed() -> None:
    record = _by_id(
        check_support(_scene([_obj("grounded", [1.0, 1.0, 0.5])])),
        "grounded",
    )

    assert record["route"] == "direct_valid_contact"
    assert record["certified_grounded_support"] is True
    assert record["grounded_support_path"] == ["grounded", "floor"]


# --------------------------------------------------------------------------- #
# 3. Sparse support contact does not invoke VLM
# --------------------------------------------------------------------------- #
def test_sparse_contact_direct_passes_without_vlm() -> None:
    # The item rests on a block covering only three of four sample columns. It is
    # still not floating; contact area and static stability are separate metrics.
    scene = _scene(
        [
            _obj("block", [0.7, 1.0, 0.2], [1.4, 2.0, 0.4]),
            _obj("item", [1.0, 1.0, 0.6], [1.6, 1.0, 0.4]),
        ]
    )
    judge = _Judge("valid")
    report = check_support(scene, vlm_judge=judge)
    item = _by_id(report, "item")

    assert item["contact_fraction"] == 0.75
    assert item["contact_fraction_affects_route"] is False
    assert item["requires_vlm"] is False
    assert item["route"] == "direct_valid_contact"
    assert item["final_verdict"] == "valid"
    assert judge.requests == []


def test_single_contact_column_is_enough_for_non_floating_support() -> None:
    scene = _scene(
        [
            _obj("narrow_foot", [0.2, 1.0, 0.2], [0.12, 2.0, 0.4]),
            _obj("frame", [1.0, 1.0, 0.6], [1.6, 1.0, 0.4]),
        ]
    )
    judge = _Judge("invalid")

    frame = _by_id(check_support(scene, vlm_judge=judge), "frame")

    assert 0.0 < frame["contact_fraction"] < 0.8
    assert frame["contact_hit_count"] >= 1
    assert frame["route"] == "direct_valid_contact"
    assert frame["final_verdict"] == "valid"
    assert judge.requests == []


def test_contact_split_across_multiple_targets_direct_passes() -> None:
    scene = _scene(
        [
            _obj("left_support", [0.65, 1.0, 0.2], [0.5, 2.0, 0.4]),
            _obj("right_support", [1.35, 1.0, 0.2], [0.5, 2.0, 0.4]),
            _obj("shelf", [1.0, 1.0, 0.6], [1.2, 1.0, 0.4]),
        ]
    )
    judge = _Judge("invalid")

    shelf = _by_id(check_support(scene, vlm_judge=judge), "shelf")

    assert shelf["support_targets"] == ["left_support", "right_support"]
    assert shelf["measured_support_modes"] == ["object_contact"]
    assert shelf["route"] == "direct_valid_contact"
    assert judge.requests == []


# --------------------------------------------------------------------------- #
# 4. Zero-contact / floating evidence invokes VLM rather than direct-invalid
# --------------------------------------------------------------------------- #
def test_floating_object_routes_to_vlm_not_direct_invalid() -> None:
    scene = _scene([_obj("float", [1.0, 1.0, 1.2], [0.5, 0.5, 0.5])])

    detector = check_support(scene, {"detector_only": True})["objects"][0]
    assert detector["contact_fraction"] == 0.0
    assert detector["requires_vlm"] is True
    assert detector["final_verdict"] is None
    assert detector["route"] is None

    judge = _Judge("valid")
    resolved = _by_id(check_support(scene, vlm_judge=judge), "float")
    assert resolved["route"] == "vlm_adjudicated"
    assert resolved["final_verdict"] == "valid"
    assert len(judge.requests) == 1


# --------------------------------------------------------------------------- #
# 5. VLM valid and invalid verdicts affect the support score correctly
# --------------------------------------------------------------------------- #
def test_vlm_verdicts_change_support_score() -> None:
    scene = _scene(
        [
            _obj("rest", [1.0, 1.0, 0.5]),
            _obj("float", [2.5, 1.5, 1.2], [0.5, 0.5, 0.5]),
        ]
    )

    valid = check_support(scene, vlm_judge=_Judge("valid"))
    invalid = check_support(scene, vlm_judge=_Judge("invalid"))

    assert valid["score"] == 1.0
    assert valid["valid_support_object_count"] == 2
    assert invalid["score"] == 0.5
    assert _by_id(invalid, "rest")["route"] == "direct_valid_contact"
    assert _by_id(invalid, "float")["final_verdict"] == "invalid"


# --------------------------------------------------------------------------- #
# 6. Candidate supporting-object descriptions and detector evidence reach the VLM
# --------------------------------------------------------------------------- #
def test_candidate_support_objects_and_evidence_reach_vlm() -> None:
    scene = _scene(
        [
            _obj("table", [1.0, 1.0, 0.5], category="table", description="wooden dining table"),
            _obj("lamp", [1.0, 1.0, 1.4], [0.3, 0.3, 0.4], category="lamp", description="tall floor lamp"),
        ]
    )
    judge = _Judge("valid")
    report = check_support(
        scene,
        prompt="Put the lamp on the table.",
        relationships=[{"subject": "lamp", "predicate": "on", "object": "table"}],
        vlm_judge=judge,
    )
    lamp = _by_id(report, "lamp")
    assert lamp["requires_vlm"] is True
    assert lamp["candidate_support_object_ids"] == ["table"]

    request = judge.requests[0]
    object_ids = [obj["id"] for obj in request["objects"]]
    assert "lamp" in object_ids and "table" in object_ids
    table_obj = next(obj for obj in request["objects"] if obj["id"] == "table")
    assert table_obj["description"] == "wooden dining table"

    evidence = request["detector_evidence"]
    assert evidence["support_instruction"] == SUPPORT_VLM_INSTRUCTION
    assert evidence["contact_fraction"] == 0.0
    assert evidence["base_contact_fraction"] == 0.0
    assert evidence["center_ray_affects_route"] is False
    assert "gap_statistics_m" in evidence
    assert set(evidence["architecture_plane_clearances_m"]) == {
        "west", "east", "south", "north", "floor", "ceiling"
    }
    assert "representative_ray_hits" in evidence
    assert any(obj["id"] == "table" for obj in evidence["candidate_support_objects"])
    assert request["natural_language_prompt"].startswith("Put the lamp")
    assert request["extracted_relationships"][0]["predicate"] == "on"


def test_wall_attachment_routes_to_vlm_with_signed_plane_clearances() -> None:
    judge = _Judge("valid")
    scene = _scene(
        [_obj("television", [0.1, 1.5, 1.5], [0.2, 1.0, 0.6], category="television")]
    )
    report = check_support(
        scene,
        prompt="Mount the television on the west wall.",
        relationships=[{"subject": "television", "predicate": "mounted_on", "object": "west_wall"}],
        vlm_judge=judge,
    )
    television = _by_id(report, "television")

    assert television["route"] == "vlm_adjudicated"
    assert television["final_verdict"] == "valid"
    assert television["base_contact_fraction"] == 0.0
    evidence = judge.requests[0]["detector_evidence"]
    assert evidence["architecture_plane_clearances_m"]["west"] == pytest.approx(0.0)
    assert evidence["architecture_plane_clearances_m"]["floor"] == pytest.approx(1.2)
    assert evidence["architecture_contact_candidates"] == [
        {"plane": "west", "signed_clearance_m": 0.0, "mode": "wall_attachment"}
    ]
    assert "possible_architecture_attachment" in evidence["routing_reasons"]
    assert judge.requests[0]["event"]["architecture_element"] == "floor_walls_ceiling_and_supports"


# --------------------------------------------------------------------------- #
# 7. The injected local-view provider receives the support event and object IDs
# --------------------------------------------------------------------------- #
def test_local_view_provider_receives_support_event(tmp_path: Path) -> None:
    local = tmp_path / "lamp_table_local.png"
    local.write_bytes(b"png")
    provider_calls: list[dict] = []

    def provider(request: dict) -> list[Path]:
        provider_calls.append(request)
        return [local]

    scene = _scene(
        [
            _obj("table", [1.0, 1.0, 0.5], category="table", description="table"),
            _obj("lamp", [1.0, 1.0, 1.4], [0.3, 0.3, 0.4]),
        ]
    )
    judge = _Judge("valid")
    check_support(scene, vlm_judge=judge, local_view_provider=provider)

    assert provider_calls
    assert provider_calls[0]["metric"] == "support"
    assert "lamp" in provider_calls[0]["object_ids"]
    assert "table" in provider_calls[0]["object_ids"]
    assert str(local) in judge.requests[0]["local_render_evidence"]


# --------------------------------------------------------------------------- #
# 8. Missing judge in official mode raises a support evaluation error
# --------------------------------------------------------------------------- #
def test_missing_judge_official_mode_raises() -> None:
    scene = _scene([_obj("float", [1.0, 1.0, 1.2], [0.5, 0.5, 0.5])])
    with pytest.raises(SupportEvaluationError, match="official mode"):
        check_support(scene, {"official_mode": True})


# --------------------------------------------------------------------------- #
# 9. Non-binary VLM output raises
# --------------------------------------------------------------------------- #
def test_non_binary_vlm_output_raises() -> None:
    class _Bad:
        def adjudicate_p0b(self, request: dict) -> dict:
            return {"verdict": "maybe", "confidence": 0.5}

    scene = _scene([_obj("float", [1.0, 1.0, 1.2], [0.5, 0.5, 0.5])])
    with pytest.raises(SupportEvaluationError):
        check_support(scene, {"official_mode": True}, vlm_judge=_Bad())


def test_non_official_support_judge_failure_is_not_counted_as_adjudicated() -> None:
    class _Bad:
        def adjudicate_p0b(self, request: dict) -> dict:
            return {"verdict": "insufficient_evidence", "confidence": 0.5}

    report = check_support(
        _scene([_obj("float", [1.0, 1.0, 1.2], [0.5, 0.5, 0.5])]),
        vlm_judge=_Bad(),
    )
    record = report["objects"][0]

    assert record["route"] == "vlm_adjudication_failed"
    assert record["final_verdict"] is None
    assert report["status"] == "requires_vlm"
    assert report["coverage"]["vlm_adjudicated_objects"] == 0


# --------------------------------------------------------------------------- #
# 10. OBB fallback works without mesh geometry
# --------------------------------------------------------------------------- #
def test_obb_fallback_without_mesh_geometry() -> None:
    scene = _scene([_obj("table", [1.0, 1.0, 0.5]), _obj("book", [1.0, 1.0, 1.1], [0.3, 0.3, 0.2])])
    report = check_support(scene)  # no collision_geometry manifest
    book = _by_id(report, "book")

    assert book["evidence_level"] == "obb"
    assert book["route"] == "direct_valid_contact"
    assert book["representative_samples"][0]["target_representation"] == "obb"


# --------------------------------------------------------------------------- #
# 11. Mesh-backed ray evidence is used when valid triangle geometry exists
# --------------------------------------------------------------------------- #
def test_mesh_backed_ray_evidence_used(tmp_path: Path) -> None:
    ply = tmp_path / "table.ply"
    _box_ply(ply, 0.5, 1.5, 0.5, 1.5, 0.0, 0.5)
    manifest = _mesh_manifest(tmp_path, "table", ply)
    scene = _scene([_obj("table", [1.0, 1.0, 0.25], [1.0, 1.0, 0.5]), _obj("mug", [1.0, 1.0, 0.6], [0.3, 0.3, 0.2])])

    report = check_support(scene, collision_geometry=manifest)
    mug = _by_id(report, "mug")

    assert mug["evidence_level"] == "mixed"
    assert mug["support_targets"] == ["table"]
    assert mug["route"] == "direct_valid_contact"
    assert any(sample["target_representation"] == "mesh" for sample in mug["representative_samples"])


# --------------------------------------------------------------------------- #
# 12. Sinking no longer independently reduces the support score
# --------------------------------------------------------------------------- #
def test_sinking_is_not_an_independent_support_penalty() -> None:
    # Bottom sunk 0.02 m below the floor (within contact tolerance). Legacy support
    # forced this to unsupported; the probe now treats penetration as contact.
    sunk = _scene([_obj("sunk", [1.0, 1.0, 0.49], [0.5, 0.5, 1.02])])
    judge = _Judge("invalid")
    report = check_support(sunk, vlm_judge=judge)
    record = report["objects"][0]

    assert record["route"] == "direct_valid_contact"
    assert record["final_verdict"] == "valid"
    assert judge.requests == []
    assert report["score"] == 1.0
    assert "sinking" not in record


# --------------------------------------------------------------------------- #
# 13. --no-support-enabled performs no detector or VLM work
# --------------------------------------------------------------------------- #
def test_disabled_support_does_no_detector_or_vlm_work() -> None:
    scene = _scene([_obj("float", [1.0, 1.0, 1.2], [0.5, 0.5, 0.5])])
    judge = _Judge("valid")
    report = evaluate_generic_validity(
        scene,
        {
            "collision": {"enabled": False},
            "oob": {"enabled": False},
            "navigability": {"enabled": False},
            "accessibility": {"enabled": False},
        },
        vlm_judge=judge,
        support_enabled=False,
    )
    support = report["metrics"]["support"]

    assert support["status"] == "not_applicable"
    assert support["score"] is None
    assert support["reason"] == "disabled_by_configuration"
    assert support["enabled"] is False
    assert support["objects"] == []
    assert judge.requests == []


def test_disabled_support_report_shape() -> None:
    report = disabled_support_report()
    assert report["status"] == "not_applicable"
    assert report["score"] is None
    assert report["reason"] == "disabled_by_configuration"
    assert report["enabled"] is False


# --------------------------------------------------------------------------- #
# 14. Disabled support is excluded from generic-validity aggregation
# --------------------------------------------------------------------------- #
def test_disabled_support_excluded_from_aggregation() -> None:
    scene = _scene([_obj("a", [1.0, 1.0, 0.5], [0.5, 0.5, 1.0]), _obj("b", [2.5, 1.0, 0.5], [0.5, 0.5, 1.0])])
    report = evaluate_generic_validity(
        scene,
        {"navigability": {"enabled": False}, "accessibility": {"enabled": False}},
        support_enabled=False,
    )

    assert "support" in report["disabled_metrics"]
    assert report["metric_scores"]["support"] is None
    assert report["metrics"]["collision"]["score"] == 1.0
    assert report["metrics"]["oob"]["score"] == 1.0
    assert report["active_metric_count"] == 2
    assert report["score"] == 1.0


# --------------------------------------------------------------------------- #
# 15. Enabled-but-unresolved support is not silently excluded (not "disabled")
# --------------------------------------------------------------------------- #
def test_enabled_unresolved_support_is_not_treated_as_disabled() -> None:
    scene = _scene([_obj("float", [1.0, 1.0, 1.2], [0.5, 0.5, 0.5])])
    report = evaluate_generic_validity(
        scene,
        {"navigability": {"enabled": False}, "accessibility": {"enabled": False}},
    )  # support enabled by default, no judge configured
    support = report["metrics"]["support"]

    assert support["status"] == "requires_vlm"
    assert support["score"] is None
    assert support["enabled"] is True
    assert support.get("reason") != "disabled_by_configuration"
    assert "support" not in report["disabled_metrics"]
    assert report["metric_scores"]["support"] is None
    assert report["score"] is None
    assert report["status"] == "incomplete"
    assert "support" in report["unresolved_metrics"]


def test_mesh_lower_envelope_does_not_inherit_grounded_bbox_bottom(tmp_path: Path) -> None:
    ply = tmp_path / "floating_inside_bbox.ply"
    _box_ply(ply, 0.9, 1.1, 0.9, 1.1, 0.4, 0.6)
    manifest = _mesh_manifest(tmp_path, "tiny", ply)
    scene = _scene([_obj("tiny", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0])])

    report = check_support(scene, collision_geometry=manifest)
    record = _by_id(report, "tiny")

    assert record["source_representation"] == "mesh"
    assert record["source_sample_method"] == "mesh_lower_envelope"
    assert record["evidence_level"] == "mesh"
    assert record["contact_fraction"] == 0.0
    assert record["gap_statistics_m"]["min"] == pytest.approx(0.4)
    assert record["requires_vlm"] is True
    assert record["final_verdict"] is None
    assert report["score"] is None


def test_mesh_base_band_excludes_raised_tabletop_from_contact_denominator(tmp_path: Path) -> None:
    ply = tmp_path / "four_leg_table.ply"
    legs = [
        (0.50, 0.60, 0.50, 0.60, 0.00, 0.80),
        (1.40, 1.50, 0.50, 0.60, 0.00, 0.80),
        (0.50, 0.60, 1.40, 1.50, 0.00, 0.80),
        (1.40, 1.50, 1.40, 1.50, 0.00, 0.80),
    ]
    tabletop = [(0.50, 1.50, 0.50, 1.50, 0.80, 1.00)]
    _boxes_ply(ply, [*legs, *tabletop])
    manifest = _mesh_manifest(tmp_path, "table", ply)
    judge = _Judge("invalid")

    record = _by_id(
        check_support(
            _scene([_obj("table", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0])]),
            collision_geometry=manifest,
            vlm_judge=judge,
        ),
        "table",
    )

    assert record["source_sample_method"] == "mesh_lower_envelope"
    assert record["contact_sample_method"] == "lowest_base_contact_band"
    assert record["base_contact_sample_count"] < record["lower_envelope_sample_count"]
    assert record["base_contact_fraction"] == 1.0
    assert record["center_ray_supported"] is False
    assert record["center_ray_affects_route"] is False
    assert record["route"] == "direct_valid_contact"
    assert judge.requests == []


def test_invalid_support_mesh_degrades_to_obb_without_crashing(tmp_path: Path) -> None:
    malformed = tmp_path / "bad_indices.ply"
    malformed.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "element face 1",
                "property list uchar int vertex_indices",
                "end_header",
                "0 0 0",
                "1 0 0",
                "0 1 0",
                "3 0 1 99",
            ]
        ),
        encoding="utf-8",
    )
    manifest = _mesh_manifest(tmp_path, "box", malformed)
    judge = _Judge("valid")

    record = _by_id(
        check_support(
            _scene([_obj("box", [1.0, 1.0, 0.5])]),
            collision_geometry=manifest,
            vlm_judge=judge,
        ),
        "box",
    )

    assert record["source_representation"] == "obb"
    assert "out-of-range face indices" in record["source_mesh_load_error"]
    assert record["geometry_evidence_degraded"] is True
    assert record["route"] == "vlm_adjudicated"
    assert record["final_verdict"] == "valid"
    assert len(judge.requests) == 1


def test_stale_world_transform_mesh_is_rejected_and_routed_to_vlm(tmp_path: Path) -> None:
    stale = tmp_path / "stale_at_origin.ply"
    _box_ply(stale, -0.5, 0.5, -0.5, 0.5, 0.0, 1.0)
    manifest = _mesh_manifest(tmp_path, "box", stale)
    judge = _Judge("valid")

    record = _by_id(
        check_support(
            _scene([_obj("box", [2.0, 2.0, 0.5], [1.0, 1.0, 1.0])]),
            collision_geometry=manifest,
            vlm_judge=judge,
        ),
        "box",
    )

    assert record["source_representation"] == "obb"
    assert record["source_mesh_frame_validation"]["canonical_consistent"] is False
    assert "canonical mesh frame mismatch" in record["source_mesh_load_error"]
    assert record["geometry_evidence_degraded"] is True
    assert record["routing_reasons"] == ["geometry_evidence_degraded"]
    assert record["route"] == "vlm_adjudicated"
    evidence = judge.requests[0]["detector_evidence"]
    assert evidence["geometry_evidence_degraded"] is True
    assert evidence["source_mesh_frame_validation"]["canonical_consistent"] is False


def test_contact_on_degraded_target_geometry_routes_to_vlm(tmp_path: Path) -> None:
    stale_table = tmp_path / "stale_table_at_origin.ply"
    _box_ply(stale_table, -0.5, 0.5, -0.5, 0.5, 0.0, 1.0)
    manifest = _mesh_manifest(tmp_path, "table", stale_table)
    judge = _Judge("valid")
    scene = _scene(
        [
            _obj("table", [2.0, 2.0, 0.5], [1.0, 1.0, 1.0]),
            _obj("book", [2.0, 2.0, 1.1], [0.3, 0.3, 0.2]),
        ]
    )

    book = _by_id(check_support(scene, collision_geometry=manifest, vlm_judge=judge), "book")

    assert book["contact_hit_count"] > 0
    assert book["support_targets"] == ["table"]
    assert book["geometry_evidence_degraded"] is True
    assert any(reason.startswith("target:table:") for reason in book["geometry_degraded_reasons"])
    assert book["route"] == "vlm_adjudicated"


def test_legacy_contact_fraction_threshold_is_diagnostic_only() -> None:
    scene = _scene(
        [
            _obj("narrow_foot", [0.2, 1.0, 0.2], [0.12, 2.0, 0.4]),
            _obj("frame", [1.0, 1.0, 0.6], [1.6, 1.0, 0.4]),
        ]
    )
    judge = _Judge("invalid")

    frame = _by_id(
        check_support(scene, {"contact_fraction_threshold": 1.0}, vlm_judge=judge),
        "frame",
    )

    assert frame["contact_fraction"] < frame["contact_fraction_threshold"]
    assert frame["contact_fraction_affects_route"] is False
    assert frame["route"] == "direct_valid_contact"
    assert judge.requests == []


def test_mesh_support_sample_budget_is_independent_of_tessellation_density(tmp_path: Path) -> None:
    ply = tmp_path / "dense_plane.ply"
    side = 10
    vertices = [
        [0.8 + x * 0.04, 0.8 + y * 0.04, 0.4]
        for x in range(side)
        for y in range(side)
    ]
    faces = []
    for x in range(side - 1):
        for y in range(side - 1):
            a = x * side + y
            b = (x + 1) * side + y
            c = (x + 1) * side + y + 1
            d = x * side + y + 1
            faces.extend([[a, b, c], [a, c, d]])
    write_ascii_triangle_ply(ply, vertices, faces)
    manifest = _mesh_manifest(tmp_path, "dense", ply)

    record = _by_id(
        check_support(
            _scene([_obj("dense", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0])]),
            collision_geometry=manifest,
        ),
        "dense",
    )

    assert record["source_representation"] == "mesh"
    assert record["sample_count"] <= 16


def test_mesh_center_support_is_diagnostic_not_a_route_gate(tmp_path: Path) -> None:
    ply = tmp_path / "hollow_frame.ply"
    rectangles = [
        (0.5, 0.8, 0.5, 1.5),
        (1.2, 1.5, 0.5, 1.5),
        (0.8, 1.2, 0.5, 0.8),
        (0.8, 1.2, 1.2, 1.5),
    ]
    vertices = []
    faces = []
    for x0, x1, y0, y1 in rectangles:
        offset = len(vertices)
        vertices.extend([[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]])
        faces.extend([[offset, offset + 1, offset + 2], [offset, offset + 2, offset + 3]])
    write_ascii_triangle_ply(ply, vertices, faces)
    manifest = _mesh_manifest(tmp_path, "frame", ply)

    record = _by_id(
        check_support(
            _scene([_obj("frame", [1.0, 1.0, 0.1], [1.0, 1.0, 0.2])]),
            collision_geometry=manifest,
        ),
        "frame",
    )

    assert record["contact_fraction"] == 1.0
    assert record["center_source_available"] is False
    assert record["center_ray_supported"] is False
    assert record["center_ray_affects_route"] is False
    assert record["requires_vlm"] is False
    assert record["route"] == "direct_valid_contact"


def test_enabled_unresolved_support_blocks_structural_category(tmp_path: Path) -> None:
    from evaluate import run_evaluate

    scene = {
        **_scene([_canonical_obj("floating", [1.0, 1.0, 1.2], [0.5, 0.5, 0.5])]),
        "request_id": "incomplete_support",
    }
    report = run_evaluate(
        scene=scene,
        out=tmp_path / "report.json",
        eval_generic_validity=True,
        support_enabled=True,
    )

    structural = report["category_reports"]["structural_validity"]
    assert structural["status"] == "not_evaluable"
    assert structural["score"] is None
    assert structural["reason"] == "generic_validity_incomplete"
    assert report["benchmark_score"] is None


def test_top_level_official_mode_requires_binary_support_adjudication() -> None:
    scene = _scene([_obj("floating", [1.0, 1.0, 1.2], [0.5, 0.5, 0.5])])
    with pytest.raises(SupportEvaluationError, match="official mode"):
        evaluate_generic_validity(
            scene,
            {
                "collision": {"enabled": False},
                "oob": {"enabled": False},
                "navigability": {"enabled": False},
                "accessibility": {"enabled": False},
            },
            p0b_official_mode=True,
        )


# --------------------------------------------------------------------------- #
# Scale-aware thresholds, gap bands, and independent multi-label scoring
# --------------------------------------------------------------------------- #
def test_scale_aware_threshold_formulas_and_caps() -> None:
    # hard = clamp(0.02 + 0.005 * size_z, 0.02, 0.035)
    assert _hard_contact_tolerance(0.0) == pytest.approx(0.02)  # min cap
    assert _hard_contact_tolerance(1.0) == pytest.approx(0.025)
    assert _hard_contact_tolerance(3.0) == pytest.approx(0.035)  # 0.02+0.015 hits max
    assert _hard_contact_tolerance(10.0) == pytest.approx(0.035)  # max cap
    # near = clamp(0.03 + 0.03 * size_z, 0.04, 0.08)
    assert _near_support_tolerance(0.0) == pytest.approx(0.04)  # min cap (0.03 -> 0.04)
    assert _near_support_tolerance(0.5) == pytest.approx(0.045)
    assert _near_support_tolerance(1.0) == pytest.approx(0.06)
    assert _near_support_tolerance(5.0) == pytest.approx(0.08)  # max cap
    # near >= hard for every size.
    for size_z in (0.0, 0.2, 0.5, 1.0, 2.0, 5.0):
        assert _near_support_tolerance(size_z) >= _hard_contact_tolerance(size_z)


def test_fixed_legacy_contact_tolerance_changes_contact_band_with_safety_cap() -> None:
    # A large object whose scale-aware candidate threshold would be 0.035,
    # pinned to a narrower 0.02 direct-valid tolerance.
    scene = _scene([_obj("tall", [1.0, 1.0, 1.5], [0.5, 0.5, 3.0])])
    report = check_support(scene, {"contact_tolerance_m": 0.02})
    record = _by_id(report, "tall")

    assert report["threshold_mode"] == "fixed_tolerance_contact"
    assert report["fixed_contact_tolerance_m"] == pytest.approx(0.02)
    assert record["hard_contact_tolerance_m"] == pytest.approx(0.02)
    assert record["contact_tolerance_m"] == pytest.approx(0.02)
    assert record["legacy_contact_tolerance_affects_direct_valid"] is True
    # near stays scale-aware but is never below the fixed hard threshold.
    assert record["near_support_tolerance_m"] >= record["hard_contact_tolerance_m"]

    # A large fixed override may widen candidate discovery, while deterministic
    # contact remains capped at 3.5 cm.
    raised = _by_id(
        check_support(_scene([_obj("box", [1.0, 1.0, 0.5])]), {"contact_tolerance_m": 0.5}),
        "box",
    )
    assert raised["hard_contact_tolerance_m"] == pytest.approx(0.5)
    assert raised["near_support_tolerance_m"] == pytest.approx(0.5)
    assert raised["contact_tolerance_m"] == pytest.approx(DIRECT_CONTACT_TOLERANCE_CAP_M)


def test_positive_gap_inside_scale_aware_hard_band_direct_passes() -> None:
    # size_z=0.5 -> hard contact tolerance is 0.0225. A 0.02 m fitting
    # clearance remains within the deterministic contact band.
    scene = _scene([_obj("cube", [1.0, 1.0, 0.27], [0.5, 0.5, 0.5])])
    judge = _Judge("invalid")
    record = _by_id(check_support(scene, vlm_judge=judge), "cube")

    assert record["minimum_positive_clearance_m"] is None
    assert record["contact_gap_statistics_m"]["max"] == pytest.approx(0.02)
    assert record["gap_band"] == "contact"
    assert record["route"] == "direct_valid_contact"
    assert record["final_verdict"] == "valid"
    assert judge.requests == []


def test_submillimeter_contact_still_direct_valid() -> None:
    scene = _scene([_obj("cube", [1.0, 1.0, 0.2505], [0.5, 0.5, 0.5])])
    judge = _Judge("invalid")
    record = _by_id(check_support(scene, vlm_judge=judge), "cube")

    assert record["gap_statistics_m"]["max"] == pytest.approx(0.0005)
    assert record["gap_band"] == "contact"
    assert record["route"] == "direct_valid_contact"
    assert record["final_verdict"] == "valid"
    assert judge.requests == []


def test_borderline_positive_clearance_routes_to_vlm() -> None:
    # size_z=1.0 -> hard 0.025, near 0.06. A 0.05 m gap is borderline.
    scene = _scene([_obj("shelf", [1.0, 1.0, 0.55], [0.5, 0.5, 1.0])])

    detector = _by_id(check_support(scene, {"detector_only": True}), "shelf")
    assert detector["gap_band"] == "borderline_positive_clearance"
    assert detector["minimum_positive_clearance_m"] == pytest.approx(0.05)
    assert detector["contact_hit_count"] == 0
    assert detector["requires_vlm"] is True
    assert detector["route"] is None
    assert "borderline_positive_clearance" in detector["routing_reasons"]

    judge = _Judge("valid")
    resolved = _by_id(check_support(scene, vlm_judge=judge), "shelf")
    assert resolved["route"] == "vlm_adjudicated"
    assert resolved["final_verdict"] == "valid"
    assert len(judge.requests) == 1


def test_strong_positive_clearance_routes_to_vlm() -> None:
    # size_z=1.0 -> near 0.06. A 0.5 m gap is a strong positive clearance.
    scene = _scene([_obj("floater", [1.0, 1.0, 1.0], [0.5, 0.5, 1.0])])
    detector = _by_id(check_support(scene, {"detector_only": True}), "floater")

    assert detector["gap_band"] == "strong_positive_clearance"
    assert detector["minimum_positive_clearance_m"] == pytest.approx(0.5)
    assert detector["requires_vlm"] is True
    assert "strong_positive_clearance" in detector["routing_reasons"]


def test_no_deterministic_direct_invalid_route() -> None:
    scene = _scene([_obj("floater", [1.0, 1.0, 1.5], [0.5, 0.5, 1.0])])
    # No judge configured and not official: the detector never emits a verdict.
    record = _by_id(check_support(scene), "floater")

    assert record["gap_band"] == "strong_positive_clearance"
    assert record["requires_vlm"] is True
    assert record["final_verdict"] is None
    assert record["route"] is None
    assert not str(record["route"] or "").startswith("direct_invalid")


def test_normalized_minimum_positive_clearance() -> None:
    # size_z=1.0, min positive gap 0.5 -> normalized 0.5.
    tall = _by_id(
        check_support(_scene([_obj("tall", [1.0, 1.0, 1.0], [0.5, 0.5, 1.0])]), {"detector_only": True}),
        "tall",
    )
    assert tall["minimum_positive_clearance_m"] == pytest.approx(0.5)
    assert tall["size_z_m"] == pytest.approx(1.0)
    assert tall["normalized_minimum_positive_clearance"] == pytest.approx(0.5)

    # size_z=0.2, same 0.5 gap -> normalized 2.5.
    short = _by_id(
        check_support(_scene([_obj("short", [1.0, 1.0, 0.6], [0.5, 0.5, 0.2])]), {"detector_only": True}),
        "short",
    )
    assert short["minimum_positive_clearance_m"] == pytest.approx(0.5)
    assert short["normalized_minimum_positive_clearance"] == pytest.approx(2.5)

    # A grounded object has no positive clearance.
    grounded = _by_id(check_support(_scene([_obj("box", [1.0, 1.0, 0.5])])), "box")
    assert grounded["minimum_positive_clearance_m"] is None
    assert grounded["normalized_minimum_positive_clearance"] is None


def test_negative_gap_is_not_support_invalid() -> None:
    # Bottom is 0.5 m below the floor (deep penetration / negative gap).
    scene = _scene([_obj("embedded", [1.0, 1.0, 0.0], [0.5, 0.5, 1.0])])
    judge = _Judge("invalid")
    record = _by_id(check_support(scene, vlm_judge=judge), "embedded")

    assert record["gap_band"] == "contact"
    assert record["gap_statistics_m"]["min"] == pytest.approx(-0.5)
    assert record["minimum_positive_clearance_m"] is None
    assert record["route"] == "direct_valid_contact"
    assert record["final_verdict"] == "valid"
    assert judge.requests == []


def test_collision_invalid_does_not_imply_support_invalid() -> None:
    # Two boxes overlap in XY (collision candidate) but both rest on the floor,
    # so Support is independently valid. The shared judge returns invalid, which
    # only affects the collision metric.
    scene = _scene(
        [
            _obj("a", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0]),
            _obj("b", [1.4, 1.0, 0.5], [1.0, 1.0, 1.0]),
        ]
    )
    judge = _Judge("invalid")
    report = evaluate_generic_validity(
        scene,
        {"navigability": {"enabled": False}, "accessibility": {"enabled": False}},
        vlm_judge=judge,
    )
    support = report["metrics"]["support"]
    collision = report["metrics"]["collision"]

    assert collision["score"] < 1.0  # collision independently invalid
    assert support["score"] == 1.0  # support independently valid (floor contact)
    assert support["evaluated_object_count"] == 2
    for record in support["objects"]:
        assert record["route"] == "direct_valid_contact"
        assert "covered_by_collision" not in record
        assert "covered_by_oob" not in record


def test_simultaneous_collision_and_support_needs_independent_positive_gap() -> None:
    # "wedged" floats above the floor (independent positive base gap) while its
    # body overlaps the tall "post" in 3D (independent collision event). It is
    # therefore legitimately invalid for both metrics, with independent evidence.
    scene = _scene(
        [
            _obj("post", [1.0, 1.0, 1.0], [0.3, 0.3, 2.0]),
            _obj("wedged", [1.5, 1.0, 1.5], [1.0, 0.5, 0.5]),
        ]
    )
    judge = _Judge("invalid")
    report = evaluate_generic_validity(
        scene,
        {"navigability": {"enabled": False}, "accessibility": {"enabled": False}},
        vlm_judge=judge,
    )
    support = report["metrics"]["support"]
    collision = report["metrics"]["collision"]
    wedged = _by_id(support, "wedged")

    # Collision independently flags the overlapping pair.
    assert collision["collision_pair_count"] >= 1
    # Support independently flags the wedged object via its own positive base gap.
    assert wedged["contact_hit_count"] == 0
    assert wedged["gap_band"] == "strong_positive_clearance"
    assert wedged["minimum_positive_clearance_m"] == pytest.approx(1.25)
    assert wedged["route"] == "vlm_adjudicated"
    assert wedged["final_verdict"] == "invalid"
    # "post" rests on the floor and stays support-valid despite the collision.
    assert _by_id(support, "post")["route"] == "direct_valid_contact"
    # No ownership gate or denominator exclusion.
    assert "covered_by_collision" not in wedged
    assert support["evaluated_object_count"] == 2


def test_judge_request_includes_policy_and_threshold_context() -> None:
    scene = _scene(
        [_obj("lamp", [1.0, 1.0, 1.4], [0.3, 0.3, 0.4], category="lamp", description="floor lamp")]
    )
    judge = _Judge("valid")
    check_support(scene, prompt="A floating lamp.", vlm_judge=judge)
    request = judge.requests[0]

    assert request["candidate_selection_policy"] == SUPPORT_CANDIDATE_SELECTION_POLICY
    assert request["event"]["candidate_selection_policy"] == SUPPORT_CANDIDATE_SELECTION_POLICY
    assert request["event"]["gap_band"] in {
        "borderline_positive_clearance",
        "strong_positive_clearance",
        "unknown_clearance",
    }
    evidence = request["detector_evidence"]
    for key in (
        "candidate_selection_policy",
        "grounded_support_policy",
        "grounded_support_required_for_direct_valid",
        "certified_grounded_support",
        "grounding_status",
        "grounded_support_path",
        "grounding_contact_target_ids",
        "size_z_m",
        "hard_contact_tolerance_m",
        "near_support_tolerance_m",
        "gap_band",
        "minimum_positive_clearance_m",
        "normalized_minimum_positive_clearance",
        "representative_ray_hits",
        "candidate_support_objects",
        "architecture_plane_clearances_m",
        "architecture_contact_candidates",
    ):
        assert key in evidence, key
    assert evidence["hard_contact_tolerance_m"] == pytest.approx(_hard_contact_tolerance(0.4))
    assert evidence["near_support_tolerance_m"] == pytest.approx(_near_support_tolerance(0.4))


@pytest.mark.parametrize(
    "config,match",
    [
        ({"contact_tolerance_m": -0.1}, "contact_tolerance_m"),
        ({"base_band_tolerance_m": -0.1}, "base_band_tolerance_m"),
        ({"contact_fraction_threshold": 1.1}, "contact_fraction_threshold"),
        ({"minimum_contact_count": 0}, "minimum_contact_count"),
        ({"minimum_contact_count": 1.5}, "minimum_contact_count"),
        ({"mesh_bounds_tolerance_m": -0.1}, "mesh_bounds_tolerance_m"),
        ({"mesh_center_tolerance_m": -0.1}, "mesh_center_tolerance_m"),
        ({"bottom_sample_grid": [0, 4]}, "bottom_sample_grid"),
        ({"bottom_sample_grid": [1.5, 4]}, "bottom_sample_grid"),
        ({"max_representative_samples": 2.5}, "max_representative_samples"),
        ({"official_mode": True, "detector_only": True}, "mutually exclusive"),
    ],
)
def test_support_rejects_invalid_configuration(config: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        check_support(_scene([_obj("box", [1.0, 1.0, 0.5])]), config)


# --------------------------------------------------------------------------- #
# 16. CLI and scene harness propagate the option end to end
# --------------------------------------------------------------------------- #
def test_cli_propagates_no_support_enabled(tmp_path: Path) -> None:
    scene = {**_scene([_canonical_obj("floating", [1.0, 1.0, 1.2], [0.5, 0.5, 0.5])]), "request_id": "cli_support"}
    scene_path = tmp_path / "scene.json"
    out_path = tmp_path / "report.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "evaluate.py"),
            "--scene",
            str(scene_path),
            "--out",
            str(out_path),
            "--no-support-enabled",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    validity = read_json(out_path)["reports"]["generic_validity"]
    assert validity["metrics"]["support"]["status"] == "not_applicable"
    assert validity["metrics"]["support"]["reason"] == "disabled_by_configuration"
    assert "support" in validity["disabled_metrics"]


def test_scene_harness_propagates_no_support_enabled(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    selection_path = tmp_path / "selection.json"
    generated_path = tmp_path / "generated.json"
    plan_path.write_text(json.dumps(_harness_object_plan()), encoding="utf-8")
    selection_path.write_text(json.dumps(_harness_asset_selection()), encoding="utf-8")
    generated_path.write_text(json.dumps(_harness_generated_scene()), encoding="utf-8")
    # The harness derives request_id from the out-dir name, which must match the
    # request_id baked into the external generated scene.
    out_dir = tmp_path / "demo_001"

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
            "--no-support-enabled",
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    validity = read_json(out_dir / "evaluation_report.json")["reports"]["generic_validity"]
    assert validity["metrics"]["support"]["status"] == "not_applicable"
    assert validity["metrics"]["support"]["reason"] == "disabled_by_configuration"
    assert "support" in validity["disabled_metrics"]
    manifest = read_json(out_dir / "run_manifest.json")
    assert manifest["evaluation"]["support_enabled"] is False
    assert manifest["evaluation"]["p0b_official_mode"] is False
    assert manifest["evaluation"]["p0b_local_view_provider_configured"] is False


def test_run_evaluate_propagates_p0b_local_view_provider(tmp_path: Path) -> None:
    from evaluate import run_evaluate

    local = tmp_path / "floating_local.png"
    local.write_bytes(b"png")
    provider_calls: list[dict] = []

    def provider(request: dict) -> list[Path]:
        provider_calls.append(request)
        return [local]

    judge = _Judge("valid")
    scene = {
        **_scene([_canonical_obj("floating", [1.0, 1.0, 1.2], [0.5, 0.5, 0.5])]),
        "request_id": "provider_case",
    }
    run_evaluate(
        scene=scene,
        out=tmp_path / "provider_report.json",
        eval_generic_validity=True,
        vlm_judge=judge,
        p0b_local_view_provider=provider,
    )

    assert provider_calls
    assert provider_calls[0]["metric"] == "support"
    assert judge.requests[0]["local_render_evidence"] == [str(local)]


def test_p0b_shared_context_excludes_fine_only_reference_relations(tmp_path: Path) -> None:
    from evaluate import run_evaluate

    judge = _Judge("valid")
    scene = {
        **_scene([_canonical_obj("generated_box", [1.0, 1.0, 1.2], [0.5, 0.5, 0.5])]),
        "request_id": "aligned_relation_case",
    }
    reference_annotation = {
        "annotation_version": "reference_annotation_v1",
        "validation_status": "confirmed",
        "source": "manual",
        "request_id": "aligned_relation_case",
        "scene_type": "room",
        "inventory_policy": "open_world",
        "objects": [
            {
                "id": "plan_box",
                "category": "box",
                "description": "box",
                "count": 1,
                "claim_state": "confirmed",
            }
        ],
        "oor_relations": [],
        "oar_relations": [
            {
                "subject_id": "plan_box",
                "type": "above",
                "architectural_element": "floor",
                "claim_state": "confirmed",
            }
        ],
        "room_constraints": {"claim_state": "not_mentioned"},
    }
    run_evaluate(
        scene=scene,
        out=tmp_path / "aligned_relation_report.json",
        scene_request={
            "request_id": "aligned_relation_case",
            "instruction": "Place a box above the floor.",
            "prompt_granularity": "fine_grained",
        },
        reference_annotation=reference_annotation,
        eval_generic_validity=True,
        vlm_judge=judge,
    )

    assert judge.requests[0]["extracted_relationships"] == []
    assert judge.requests[0]["natural_language_prompt"] == "Place a box above the floor."


def _harness_object_plan() -> dict:
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


def _harness_asset_selection() -> dict:
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


def _harness_generated_scene() -> dict:
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": "generated_demo_001",
        "request_id": "demo_001",
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
                "center": [2.0, 1.0, 1.5],
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
