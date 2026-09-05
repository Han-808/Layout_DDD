from __future__ import annotations

import pytest

from benchmark.evaluator.scene_quality.functional_clearance import (
    apply_directional_clearance_profiles_to_ledger,
    build_directional_clearance_extensions,
    build_directional_clearance_profile,
)


def _scene() -> dict:
    return {
        "objects": [
            {
                "id": "cabinet",
                "center": [0.0, 0.0, 0.5],
                "size": [1.2, 0.5, 1.0],
                "rotation": [0.0, 0.0, 0.0],
            },
            {
                "id": "coffee_table",
                "center": [0.0, -1.35, 0.3],
                "size": [1.0, 0.6, 0.6],
                "rotation": [0.0, 0.0, 0.0],
            },
            {
                "id": "side_chair",
                "center": [0.8, -0.7, 0.45],
                "size": [0.5, 0.5, 0.9],
                "rotation": [0.0, 0.0, 0.0],
            },
            {
                "id": "rug",
                "center": [0.0, -0.7, 0.01],
                "size": [2.0, 2.0, 0.02],
                "rotation": [0.0, 0.0, 0.0],
            },
            {
                "id": "television",
                "center": [0.0, -0.25, 1.25],
                "size": [0.9, 0.1, 0.5],
                "rotation": [0.0, 0.0, 0.0],
            },
            {
                "id": "behind",
                "center": [0.0, 0.8, 0.5],
                "size": [0.4, 0.4, 1.0],
                "rotation": [0.0, 0.0, 0.0],
            },
            {
                "id": "ceiling_light",
                "center": [0.0, -0.7, 2.7],
                "size": [0.5, 0.5, 0.4],
                "rotation": [0.0, 0.0, 0.0],
            },
        ]
    }


def _surface(side_id: str = "local_neg_y") -> dict:
    return {
        "target_id": "cabinet",
        "status": "identified",
        "surfaces": [
            {
                "side_id": side_id,
                "surface_role": "opening_side",
                "confidence": 0.9,
            }
        ],
    }


def test_directional_clearance_uses_usable_side_and_filters_support_layers(
) -> None:
    profile = build_directional_clearance_profile(
        scene=_scene(),
        target_id="cabinet",
        usable_surface_hypothesis=_surface(),
        relation_records=[
            {
                "target_ids": ["cabinet", "side_chair"],
                "ordinary_mobility": "movable_companion",
            }
        ],
    )

    assert profile["status"] == "available"
    assert profile["world_outward_direction_xy"] == pytest.approx(
        [0.0, -1.0]
    )
    by_id = {
        item["object_id"]: item
        for item in profile["forward_intersections"]
    }
    assert "behind" not in by_id
    assert by_id["coffee_table"]["excluded_from_obstacle"] is False
    assert by_id["side_chair"]["ordinary_mobility"] == (
        "movable_companion"
    )
    assert by_id["rug"]["thin_floor_layer"] is True
    assert by_id["rug"]["excluded_from_obstacle"] is True
    assert by_id["television"]["support_relation"] == (
        "supported_by_target"
    )
    assert by_id["television"]["excluded_from_obstacle"] is True
    assert by_id["ceiling_light"]["vertical_relevant"] is False
    assert by_id["ceiling_light"]["excluded_from_obstacle"] is True
    assert profile["nearest_forward_obstacle_distance_m"] == pytest.approx(
        0.2
    )


def test_directional_clearance_rotates_local_side_into_world_space() -> None:
    scene = _scene()
    scene["objects"][0]["rotation"] = [0.0, 0.0, 90.0]
    profile = build_directional_clearance_profile(
        scene=scene,
        target_id="cabinet",
        usable_surface_hypothesis=_surface(),
    )

    assert profile["world_outward_direction_xy"] == pytest.approx(
        [1.0, 0.0]
    )


@pytest.mark.parametrize(
    "hypothesis,reason",
    [
        ({"status": "ambiguous", "surfaces": []}, (
            "usable_side_unavailable_or_ambiguous"
        )),
        ({"status": "identified", "surfaces": []}, (
            "usable_side_unavailable_or_ambiguous"
        )),
    ],
)
def test_directional_clearance_fails_soft_without_one_trusted_side(
    hypothesis: dict,
    reason: str,
) -> None:
    profile = build_directional_clearance_profile(
        scene=_scene(),
        target_id="cabinet",
        usable_surface_hypothesis=hypothesis,
    )

    assert profile["status"] == "unavailable"
    assert profile["unavailable_reason"] == reason
    assert profile["decision_authority"] == "none"


def test_extensions_cover_every_accepted_clearance_check_independently(
) -> None:
    discovery = {
        "directed_surface_targets": [
            {
                "target_id": "cabinet",
                "precomputed_usable_surface_hypothesis": _surface(),
            },
            {
                "target_id": "behind",
                "precomputed_usable_surface_hypothesis": {
                    "target_id": "behind",
                    "status": "ambiguous",
                    "surfaces": [],
                },
            },
        ],
        "relation_admission_audit": {
            "context_only_relations": [
                {
                    "target_ids": ["cabinet", "side_chair"],
                    "focal_id": "cabinet",
                    "counterpart_id": "side_chair",
                    "ordinary_mobility": "movable_companion",
                }
            ]
        },
    }
    ledger = {
        "checks": [
            {
                "check_id": "functional_check_001",
                "check_type": "clearance",
                "target_ids": ["cabinet"],
            },
            {
                "check_id": "functional_check_002",
                "check_type": "clearance",
                "target_ids": ["behind"],
            },
            {
                "check_id": "functional_check_003",
                "check_type": "architecture_orientation",
                "target_ids": ["cabinet"],
            },
        ]
    }

    extensions = build_directional_clearance_extensions(
        scene=_scene(),
        discovery=discovery,
        functional_check_ledger=ledger,
    )

    assert set(extensions) == {
        "functional_check_001",
        "functional_check_002",
    }
    assert extensions["functional_check_001"]["status"] == "available"
    side_chair = next(
        item
        for item in extensions["functional_check_001"][
            "forward_intersections"
        ]
        if item["object_id"] == "side_chair"
    )
    assert side_chair["ordinary_mobility"] == "movable_companion"
    assert extensions["functional_check_002"]["status"] == "unavailable"

    enriched = apply_directional_clearance_profiles_to_ledger(
        ledger,
        by_check_id=extensions,
    )
    by_check = {
        item["check_id"]: item for item in enriched["checks"]
    }
    directed = by_check["functional_check_001"]
    assert directed["causal_candidate_policy"] == (
        "usable_side_forward_corridor_v1"
    )
    assert "rug" not in directed["causal_candidate_ids"]
    assert "television" not in directed["causal_candidate_ids"]
    assert by_check["functional_check_002"]["causal_candidate_policy"] == (
        "directional_profile_unavailable_judge_not_restricted"
    )


def test_counterpart_mobility_is_not_reversed_onto_the_focal() -> None:
    profile = build_directional_clearance_profile(
        scene=_scene(),
        target_id="cabinet",
        usable_surface_hypothesis=_surface(),
        relation_records=[
            {
                "target_ids": ["side_chair", "cabinet"],
                "focal_id": "side_chair",
                "counterpart_id": "cabinet",
                "ordinary_mobility": "movable_companion",
            }
        ],
    )

    side_chair = next(
        item
        for item in profile["forward_intersections"]
        if item["object_id"] == "side_chair"
    )
    assert side_chair["ordinary_mobility"] == "unspecified"


def test_corridor_uses_full_sat_not_only_forward_lateral_projection() -> None:
    scene = {
        "objects": [
            {
                "id": "cabinet",
                "center": [0.0, 0.0, 0.5],
                "size": [1.0, 0.5, 1.0],
                "rotation": [0.0, 0.0, 0.0],
            },
            {
                "id": "corner_near_only",
                "center": [0.74, -0.35, 0.45],
                "size": [0.6, 0.16, 0.9],
                "rotation": [0.0, 0.0, 95.0],
            },
        ]
    }

    profile = build_directional_clearance_profile(
        scene=scene,
        target_id="cabinet",
        usable_surface_hypothesis=_surface(),
    )

    assert profile["corridor_half_width_m"] == pytest.approx(0.65)
    assert profile["forward_intersections"] == []


def test_clearance_reports_overlap_width_and_fraction_without_verdict() -> None:
    profile = build_directional_clearance_profile(
        scene=_scene(),
        target_id="cabinet",
        usable_surface_hypothesis=_surface(),
    )
    coffee_table = next(
        item
        for item in profile["forward_intersections"]
        if item["object_id"] == "coffee_table"
    )

    assert coffee_table["corridor_overlap_width_m"] > 0.0
    assert 0.0 < coffee_table["corridor_width_overlap_fraction"] <= 1.0
    assert coffee_table["corridor_overlap_area_proxy_m2"] > 0.0
    assert "verdict" not in profile
