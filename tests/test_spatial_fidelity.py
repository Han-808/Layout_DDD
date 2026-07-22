from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from benchmark.evaluator.spatial_fidelity import (
    DEFAULT_COOCCURRENCE_CONFIG,
    DEFAULT_FUNCTIONAL_GROUPING_CONFIG,
    DEFAULT_SCALE_CONFIG,
    evaluate_cooccurrence,
    evaluate_functional_grouping,
    evaluate_scale,
    evaluate_spatial_fidelity,
    load_ontology,
)


def _distribution(p5: float, median: float, p95: float) -> dict[str, float]:
    return {
        "p5": p5,
        "p25": (p5 + median) / 2.0,
        "median": median,
        "p75": (median + p95) / 2.0,
        "p95": p95,
        "mean": median,
        "std": (p95 - p5) / 4.0,
    }


def _dimensions(
    *,
    width: tuple[float, float, float],
    depth: tuple[float, float, float],
    height: tuple[float, float, float],
    count: int,
) -> dict[str, Any]:
    return {
        "width_m": _distribution(*width),
        "height_m": _distribution(*height),
        "depth_m": _distribution(*depth),
        "n_width": count,
        "n_height": count,
        "n_depth": count,
    }


def _ontology() -> dict[str, Any]:
    """Small, exact SceneOnto-style fixture with sparse top-k pair storage."""

    return {
        "schema_version": "sceneonto_test_v1",
        "bed": {
            "count": 240,
            "room_associations": {"bedroom": {"count": 220}},
            "dimensions": _dimensions(
                width=(1.8, 2.0, 2.2),
                depth=(0.9, 1.0, 1.1),
                height=(0.4, 0.6, 0.8),
                count=240,
            ),
            "cooccurrence": {
                "nightstand": {"count": 180, "p_b_given_a": 0.75, "npmi": 0.55},
                "refrigerator": {"count": 1, "p_b_given_a": 0.005, "npmi": -0.4},
            },
            "cooccurrence_by_room": {
                "bedroom": {
                    "nightstand": {"count": 180, "p_b_given_a": 0.82, "npmi": 0.6}
                }
            },
        },
        "nightstand": {
            "count": 210,
            "room_associations": {"bedroom": {"count": 190}},
            "dimensions": _dimensions(
                width=(0.4, 0.5, 0.7),
                depth=(0.35, 0.45, 0.6),
                height=(0.45, 0.6, 0.75),
                count=210,
            ),
            "cooccurrence": {
                "bed": {"count": 168, "p_b_given_a": 0.80, "npmi": 0.55}
            },
            "cooccurrence_by_room": {
                "bedroom": {
                    "bed": {"count": 167, "p_b_given_a": 0.88, "npmi": 0.6}
                }
            },
        },
        "refrigerator": {
            "count": 230,
            "dimensions": _dimensions(
                width=(0.7, 0.9, 1.1),
                depth=(0.6, 0.75, 0.9),
                height=(1.5, 1.8, 2.2),
                count=230,
            ),
            "cooccurrence": {
                "bed": {"count": 1, "p_b_given_a": 0.004, "npmi": -0.4}
            },
        },
        "wardrobe": {
            "count": 170,
            "dimensions": _dimensions(
                width=(1.2, 1.5, 2.0),
                depth=(0.45, 0.55, 0.7),
                height=(1.8, 2.0, 2.4),
                count=170,
            ),
            # Its top-k list deliberately omits bed. Absence is unknown, not zero.
            "cooccurrence": {},
        },
    }


def _object(object_id: str, category: str, size: list[float]) -> dict[str, Any]:
    return {
        "id": object_id,
        "category": category,
        "description": category,
        "center": [1.0, 1.0, size[2] / 2.0],
        "size": size,
        "rotation": [0.0, 0.0, 0.0],
    }


def _scene(*objects: dict[str, Any], scene_type: str = "bedroom") -> dict[str, Any]:
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": "spatial_fidelity_test",
        "scene_type": scene_type,
        "boundary": [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]],
        "scene_height": 3.0,
        "objects": list(objects),
    }


class _RecordingJudge:
    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        return {"verdict": self.verdict, "confidence": 0.9, "reason": "fixture"}


def test_sceneonto_shape_and_file_sha256_are_preserved(tmp_path: Path) -> None:
    payload = json.dumps(_ontology(), indent=2, sort_keys=True).encode("utf-8")
    path = tmp_path / "sceneonto.json"
    path.write_bytes(payload)

    ontology = load_ontology(path)

    assert ontology.identity == {
        "source": path.resolve().as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "available": True,
        "schema_version": "sceneonto_test_v1",
        "storage_semantics": "sparse_top_k",
        "category_count": 4,
    }
    assert ontology.dimension_stats("bed", "width") == {
        "p5": 1.8,
        "p95": 2.2,
        "median": 2.0,
        "n_samples": 240,
    }
    room_record = ontology.cooccurrence_record("bed", "nightstand", room_type="bedroom")
    assert room_record is not None
    assert room_record.as_dict() == {
        "p_other_given_anchor": 0.82,
        "pair_count": 180,
        "anchor_observation_count": 220,
        "npmi": 0.6,
        "context": "room_conditioned",
        "probability_source": "recorded_conditional_probability",
    }


def test_scale_uses_width_depth_height_and_accepts_horizontal_axis_swap() -> None:
    ontology = load_ontology(_ontology())
    direct = evaluate_scale(
        _scene(_object("bed-direct", "bed", [2.0, 1.0, 0.6])),
        ontology,
        deepcopy(DEFAULT_SCALE_CONFIG),
    )
    swapped = evaluate_scale(
        _scene(_object("bed-swapped", "bed", [1.0, 2.0, 0.6])),
        ontology,
        deepcopy(DEFAULT_SCALE_CONFIG),
    )

    direct_check = direct["checks"][0]
    swapped_check = swapped["checks"][0]
    assert direct_check["horizontal_assignment"] == "direct"
    assert [
        (axis["canonical_axis"], axis["ontology_axis"], axis["actual_m"])
        for axis in direct_check["axis_checks"]
    ] == [
        ("width", "width", 2.0),
        ("depth", "depth", 1.0),
        ("height", "height", 0.6),
    ]
    assert swapped_check["horizontal_assignment"] == "swapped"
    assert [
        (axis["canonical_axis"], axis["ontology_axis"], axis["actual_m"])
        for axis in swapped_check["axis_checks"]
    ] == [
        ("width", "depth", 1.0),
        ("depth", "width", 2.0),
        ("height", "height", 0.6),
    ]
    assert direct["score"] == swapped["score"] == 1.0


@pytest.mark.parametrize(
    ("height", "classification", "route", "score"),
    [
        (0.6, "typical", "direct_valid", 1.0),
        (1.0, "unusual", "requires_vlm", None),
        (2.0, "extreme", "requires_vlm", None),
    ],
)
def test_scale_typical_unusual_and_extreme_routing(
    height: float,
    classification: str,
    route: str,
    score: float | None,
) -> None:
    report = evaluate_scale(
        _scene(_object("bed", "bed", [2.0, 1.0, height])),
        load_ontology(_ontology()),
        deepcopy(DEFAULT_SCALE_CONFIG),
    )

    check = report["checks"][0]
    assert check["statistical_classification"] == classification
    assert check["route"] == route
    assert check["score"] == score
    assert report["score"] == score
    if classification != "typical":
        assert check["candidate_route"] == "requires_vlm"
        assert check["vlm_candidate"]["event"]["type"] == "scale_outlier"


def test_scale_outlier_is_scored_only_after_binary_vlm_adjudication() -> None:
    judge = _RecordingJudge("invalid")
    report = evaluate_scale(
        _scene(_object("oversized-bed", "bed", [2.0, 1.0, 2.0])),
        load_ontology(_ontology()),
        deepcopy(DEFAULT_SCALE_CONFIG),
        prompt="A bedroom with a bed",
        render_evidence=["render.png"],
        vlm_judge=judge,
    )

    assert report["status"] == "checked"
    assert report["score"] == 0.0
    assert report["checks"][0]["route"] == "vlm_adjudicated"
    assert len(judge.requests) == 1
    request = judge.requests[0]
    assert request["category"] == "spatial_fidelity_adjudication"
    assert request["metric"] == "scale"
    assert request["detector_evidence"]["statistical_classification"] == "extreme"
    assert request["required_response"] == {"verdict": "valid|invalid"}


def test_unknown_and_zero_check_scale_cases_never_receive_full_credit() -> None:
    ontology = load_ontology(_ontology())
    unknown = evaluate_scale(
        _scene(_object("unknown", "mystery_artifact", [1.0, 1.0, 1.0])),
        ontology,
        deepcopy(DEFAULT_SCALE_CONFIG),
    )
    empty_scale = evaluate_scale(_scene(), ontology, deepcopy(DEFAULT_SCALE_CONFIG))
    one_category_cooccurrence = evaluate_cooccurrence(
        _scene(_object("bed", "bed", [2.0, 1.0, 0.6])),
        ontology,
        deepcopy(DEFAULT_COOCCURRENCE_CONFIG),
    )

    assert unknown["status"] == "incomplete"
    assert unknown["score"] is None
    assert unknown["coverage"]["unknown_count"] == 1
    assert unknown["checks"][0]["reason"] == "unknown_ontology_category"
    for report in (empty_scale, one_category_cooccurrence):
        assert report["status"] == "not_applicable"
        assert report["score"] is None
        assert report["partial_score"] is None
        assert report["coverage"]["complete"] is False


def test_cooccurrence_scores_each_unique_unordered_category_pair_once() -> None:
    report = evaluate_cooccurrence(
        _scene(
            _object("bed-1", "bed", [2.0, 1.0, 0.6]),
            _object("bed-2", "bed", [2.1, 1.0, 0.6]),
            _object("nightstand-1", "nightstand", [0.5, 0.45, 0.6]),
            _object("nightstand-2", "nightstand", [0.6, 0.5, 0.65]),
        ),
        load_ontology(_ontology()),
        deepcopy(DEFAULT_COOCCURRENCE_CONFIG),
    )

    assert report["pair_unit"] == "unique_unordered_category_pair"
    assert report["distinct_semantic_categories"] == ["bed", "nightstand"]
    assert report["eligible_category_pair_count"] == 1
    assert report["coverage"]["eligible_count"] == 1
    assert len(report["checks"]) == 1
    check = report["checks"][0]
    assert set(check["object_ids_a"] + check["object_ids_b"]) == {
        "bed-1",
        "bed-2",
        "nightstand-1",
        "nightstand-2",
    }
    assert check["route"] == "direct_valid"
    assert report["score"] == 1.0


def test_cooccurrence_prefers_room_evidence_then_falls_back_to_global() -> None:
    room_report = evaluate_cooccurrence(
        _scene(
            _object("bed", "bed", [2.0, 1.0, 0.6]),
            _object("nightstand", "nightstand", [0.5, 0.45, 0.6]),
            scene_type="bedroom",
        ),
        load_ontology(_ontology()),
        deepcopy(DEFAULT_COOCCURRENCE_CONFIG),
    )
    room_check = room_report["checks"][0]
    assert room_check["evidence_context"] == "room_conditioned"
    assert room_check["best_directional_probability"] == 0.88
    assert room_check["fallback_reason"] is None

    fallback_fixture = _ontology()
    fallback_fixture["bed"]["cooccurrence_by_room"]["bedroom"]["nightstand"] = {
        "count": 1,
        "p_b_given_a": 0.005,
    }
    fallback_fixture["nightstand"]["cooccurrence_by_room"]["bedroom"].pop("bed")
    fallback_report = evaluate_cooccurrence(
        _scene(
            _object("bed", "bed", [2.0, 1.0, 0.6]),
            _object("nightstand", "nightstand", [0.5, 0.45, 0.6]),
            scene_type="bedroom",
        ),
        load_ontology(fallback_fixture),
        deepcopy(DEFAULT_COOCCURRENCE_CONFIG),
    )
    fallback_check = fallback_report["checks"][0]
    assert fallback_check["evidence_context"] == "global"
    assert fallback_check["best_directional_probability"] == 0.8
    assert fallback_check["fallback_reason"] == "room_conditioned_evidence_insufficient"
    assert fallback_check["route"] == "direct_valid"


def test_missing_sparse_cooccurrence_entry_is_unknown_not_negative() -> None:
    report = evaluate_cooccurrence(
        _scene(
            _object("bed", "bed", [2.0, 1.0, 0.6]),
            _object("wardrobe", "wardrobe", [1.5, 0.55, 2.0]),
        ),
        load_ontology(_ontology()),
        deepcopy(DEFAULT_COOCCURRENCE_CONFIG),
    )

    assert report["status"] == "incomplete"
    assert report["score"] is None
    assert report["partial_score"] is None
    assert report["coverage"]["unknown_count"] == 1
    check = report["checks"][0]
    assert check["route"] == "unknown"
    assert check["reason"] == "missing_sparse_cooccurrence_entry"
    assert check["ontology_storage_semantics"] == "sparse_top_k"


def test_bidirectionally_supported_low_frequency_pair_routes_to_vlm() -> None:
    scene = _scene(
        _object("bed", "bed", [2.0, 1.0, 0.6]),
        _object("fridge", "refrigerator", [0.9, 0.75, 1.8]),
        scene_type="studio",
    )
    ontology = load_ontology(_ontology())
    pending = evaluate_cooccurrence(
        scene,
        ontology,
        deepcopy(DEFAULT_COOCCURRENCE_CONFIG),
    )

    check = pending["checks"][0]
    assert pending["score"] is None
    assert pending["coverage"]["vlm_pending_count"] == 1
    assert check["route"] == "requires_vlm"
    assert check["best_directional_probability"] == 0.005
    assert check["rarity_support"] == {
        "sufficient": True,
        "both_directions_recorded": True,
        "authoritative_directional_probabilities": True,
        "anchor_observation_counts": [240, 230],
        "minimum_anchor_observation_count": 100,
        "adequate_anchor_observation_support": True,
        "joint_pair_count_recorded": True,
        "pair_counts": [1, 1],
    }

    judge = _RecordingJudge("invalid")
    adjudicated = evaluate_cooccurrence(
        scene,
        ontology,
        deepcopy(DEFAULT_COOCCURRENCE_CONFIG),
        vlm_judge=judge,
    )
    assert adjudicated["status"] == "checked"
    assert adjudicated["score"] == 0.0
    assert adjudicated["checks"][0]["route"] == "vlm_adjudicated"
    assert judge.requests[0]["metric"] == "cooccurrence_plausibility"
    assert judge.requests[0]["event"]["type"] == "rare_category_cooccurrence"


def test_functional_grouping_is_an_explicit_zero_weight_placeholder() -> None:
    report = evaluate_functional_grouping(deepcopy(DEFAULT_FUNCTIONAL_GROUPING_CONFIG))

    assert report["metric"] == "functional_grouping"
    assert report["implemented"] is False
    assert report["enabled"] is False
    assert report["status"] == "not_implemented"
    assert report["score"] is None
    assert report["affects_score"] is False
    assert report["checks"] == []
    with pytest.raises(ValueError, match="not implemented"):
        evaluate_functional_grouping({"enabled": True, "implemented": False})


def test_spatial_fidelity_aggregate_remains_incomplete_on_metric_coverage_gap() -> None:
    report = evaluate_spatial_fidelity(
        _scene(
            _object("bed", "bed", [2.0, 1.0, 0.6]),
            _object("wardrobe", "wardrobe", [1.5, 0.55, 2.0]),
        ),
        ontology=_ontology(),
    )

    assert report["metrics"]["scale"]["score"] == 1.0
    assert report["metrics"]["cooccurrence_plausibility"]["score"] is None
    assert report["metrics"]["functional_grouping"]["status"] == "not_implemented"
    assert report["status"] == "incomplete"
    assert report["score"] is None
    assert report["partial_score"] == 1.0
    assert report["coverage"] == {
        "covered_metric_weight": 0.5,
        "required_metric_weight": 1.0,
        "complete": False,
        "covered_metrics": ["scale"],
        "uncovered_metrics": ["cooccurrence_plausibility"],
        "not_applicable_metrics": [],
        "zero_weight_metrics": ["functional_grouping"],
    }
