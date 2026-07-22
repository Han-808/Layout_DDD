from __future__ import annotations

import json
from pathlib import Path

from benchmark.evaluator.OAR import evaluate_oar
from benchmark.evaluator.OOR import evaluate_oor
from benchmark.evaluator.generic_validity import evaluate_generic_validity
from benchmark.reference_annotation import validate_reference_annotation
from benchmark.scene_io.validate import validate_generated_scene, validate_scene_request
from scripts.aggregate_source_distortion_experiment import _source_row


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "configs" / "experiments" / "p0b_source_distortion5"
DETECTOR_CONFIG = {
    "collision": {"detector_only": True},
    "oob": {"detector_only": True},
    "support": {"detector_only": True},
    "navigability": {"enabled": False},
    "accessibility": {"enabled": False},
}


def test_source_scene_fixture_matrix_and_metric_isolation() -> None:
    manifest = _read(FIXTURE_ROOT / "cases.json")
    cases = manifest["cases"]
    assert manifest["source_case_count"] == 5
    assert manifest["total_case_count"] == 25
    assert {case["family"] for case in cases} == {"clean", "collision", "oob", "support", "oar"}

    by_base: dict[str, set[str]] = {}
    for case in cases:
        by_base.setdefault(case["base_case_id"], set()).add(case["family"])
        fixture_dir = FIXTURE_ROOT / case["fixture_dir"]
        scene = _read(fixture_dir / "generated_scene.json")
        request = _read(fixture_dir / "scene_request.json")
        annotation = _read(fixture_dir / "reference_annotation.json")
        distortion = _read(fixture_dir / "distortion_manifest.json")

        validate_generated_scene(scene)
        validate_scene_request(request)
        validate_reference_annotation(annotation)
        assert scene["request_id"] == request["request_id"] == annotation["request_id"] == case["case_id"]
        assert scene["metadata"]["generator_skipped"] is True
        assert request["metadata"]["generator_skipped"] is True

        generic = evaluate_generic_validity(scene, config=DETECTOR_CONFIG)["metrics"]
        actual_collision = sorted(
            sorted([item["object_a"], item["object_b"]])
            for item in generic["collision"]["pairs"]
            if item.get("requires_vlm")
        )
        actual_oob = sorted(item["object_id"] for item in generic["oob"]["objects"] if item.get("requires_vlm"))
        actual_support = sorted(
            item["object_id"] for item in generic["support"]["objects"] if item.get("requires_vlm")
        )
        expected = distortion["expected"]
        expected_collision_candidates = expected.get("collision_candidate_pairs") or [
            *expected.get("collision_invalid_pairs", []),
            *expected.get("collision_valid_pairs", []),
        ]
        assert actual_collision == sorted(sorted(pair) for pair in expected_collision_candidates)
        invalid_collision = {
            tuple(sorted(pair)) for pair in expected.get("collision_invalid_pairs", [])
        }
        valid_collision = {
            tuple(sorted(pair)) for pair in expected.get("collision_valid_pairs", [])
        }
        assert invalid_collision.isdisjoint(valid_collision)
        if expected.get("collision_candidate_pairs") is not None:
            assert {
                tuple(sorted(pair)) for pair in expected["collision_candidate_pairs"]
            } == invalid_collision | valid_collision
        assert actual_oob == sorted(expected["oob_invalid_object_ids"])
        # Support is intentionally a high-recall router: additional candidates
        # are acceptable, but a controlled known-invalid object must never be
        # deterministically bypassed.
        assert set(expected["support_invalid_object_ids"]).issubset(actual_support)

        oor = evaluate_oor(scene, relation_specs=_relations(annotation, "oor_relations"))
        assert all(check.get("passed") is not False for check in oor["checks"])
        oar = evaluate_oar(scene, relation_specs=_relations(annotation, "oar_relations"))
        failed_oar = {
            str(check["relation_id"])
            for check in oar["checks"]
            if check.get("passed") is False
        }
        expected_failed_oar = set(
            expected.get("oar_invalid_relation_ids")
            or [f"oar_{index:03d}" for index in expected["oar_invalid_relation_indices"]]
        )
        assert failed_oar == expected_failed_oar

    assert len(by_base) == 5
    assert all(families == {"clean", "collision", "oob", "support", "oar"} for families in by_base.values())


def test_distortion_severity_is_explicit_and_varied() -> None:
    cases = _read(FIXTURE_ROOT / "cases.json")["cases"]
    for family in ("oob", "support", "oar"):
        values = {
            round(
                float(case["severity"]["target_relation_fraction" if family == "oar" else "target_object_fraction"]),
                6,
            )
            for case in cases
            if case["family"] == family
        }
        assert len(values) >= 4


def test_source_summary_compares_exact_controlled_events(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixtures_root"
    fixture_dir = fixture_root / "fixtures" / "case_001"
    source_root = tmp_path / "reports"
    report_dir = source_root / "case_001"
    fixture_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    (fixture_dir / "distortion_manifest.json").write_text(
        json.dumps(
            {
                "expected": {
                    "collision_invalid_pairs": [["obj_b", "obj_a"]],
                    "oob_invalid_object_ids": ["obj_c"],
                    "support_invalid_object_ids": [],
                    "oar_invalid_relation_indices": [1],
                }
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "evaluation_report.json").write_text(
        json.dumps(
            {
                "reports": {
                    "generic_validity": {
                        "metrics": {
                            "collision": {
                                "pairs": [
                                    {
                                        "object_a": "obj_a",
                                        "object_b": "obj_b",
                                        "judge_result": {"verdict": "invalid"},
                                    }
                                ]
                            },
                            "oob": {
                                "objects": [
                                    {
                                        "object_id": "obj_c",
                                        "judge_result": {"verdict": "invalid"},
                                    }
                                ]
                            },
                            "support": {"objects": []},
                        }
                    },
                    "oar": {"checks": [{"passed": True}, {"passed": False}]},
                }
            }
        ),
        encoding="utf-8",
    )
    row = _source_row(
        {
            "case_id": "case_001",
            "base_case_id": "base_001",
            "family": "collision",
            "fixture_dir": "fixtures/case_001",
            "severity": {},
        },
        source_root,
        fixture_root,
    )
    assert row["collision_exact_match"] is True
    assert row["oob_exact_match"] is True
    assert row["support_exact_match"] is True
    assert row["oar_exact_match"] is True
    assert row["all_controlled_metrics_exact_match"] is True


def _relations(annotation: dict, key: str) -> list[dict]:
    return [
        {name: value for name, value in relation.items() if name != "claim_state"}
        for relation in annotation[key]
    ]


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value
