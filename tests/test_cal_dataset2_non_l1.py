from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.judge_cal_dataset2_non_l1_evidence import (
    _arm_selections,
    _judge_context,
    _validate_response,
)
from scripts.validate_cal_dataset2_non_l1 import (
    CANONICAL_METRICS,
    EXPECTED_AUTHORIZATION_CELLS,
    EXPECTED_CASE_COUNT,
    EXPECTED_METRIC_COUNTS,
    Report,
    _outbound_allowlists,
    _validate_exact_span,
    _validate_opaque_identifier,
    json_diff_paths,
    scene_geometry_sha256,
    validate_dataset,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "Support/datasets/cal_dataset2_non_l1_evidence"


def _scene(*, request_id: str, center_x: float = 1.0) -> dict:
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": f"scene_for_{request_id}",
        "request_id": request_id,
        "scene_type": "unspecified_room",
        "boundary": [[0, 0], [4, 0], [4, 4], [0, 4]],
        "scene_height": 2.8,
        "objects": [
            {
                "id": "obj_000",
                "category": "chair",
                "jid": "4_SM_Chair01",
                "asset_ref": {
                    "source_db": "imaginarium_assets",
                    "asset_key": "4_SM_Chair01",
                },
                "asset_proxy": {
                    "type": "obb",
                    "bbox_size": [0.6, 0.6, 0.9],
                    "bbox_center_local": [0.0, 0.0, 0.0],
                },
                "size": [0.6, 0.6, 0.9],
                "center": [center_x, 1.0, 0.45],
                "rotation": [0.0, 0.0, 0.0],
                "geometry_provenance": "asset_mesh",
                "metadata": {},
            }
        ],
        "metadata": {
            "coordinate_frame": {
                "origin": "room_min_corner_floor",
                "axes": "x_width_y_depth_z_up",
                "unit": "meter",
                "rotation_unit": "degree",
            },
            "case_id": request_id,
        },
    }


def test_registered_distribution_sums_to_108() -> None:
    assert set(EXPECTED_METRIC_COUNTS) == set(CANONICAL_METRICS)
    assert sum(EXPECTED_METRIC_COUNTS.values()) == EXPECTED_CASE_COUNT
    assert len(EXPECTED_AUTHORIZATION_CELLS) == 4


def test_prompt_span_requires_exact_offsets(tmp_path: Path) -> None:
    prompt = "Create a room and make the chair face the monitor."
    span = "make the chair face the monitor"
    start = prompt.index(span)
    report = Report(tmp_path)
    _validate_exact_span(
        prompt,
        {
            "source_span": span,
            "source_char_start": start,
            "source_char_end": start + len(span),
        },
        text_key="source_span",
        offset_prefix="source",
        path="claim",
        report=report,
    )
    assert report.errors == []


def test_prompt_span_rejects_correct_text_at_wrong_offsets(tmp_path: Path) -> None:
    prompt = "Place a mug beside a table."
    span = "mug beside a table"
    report = Report(tmp_path)
    _validate_exact_span(
        prompt,
        {
            "source_span": span,
            "source_char_start": 0,
            "source_char_end": len(span),
        },
        text_key="source_span",
        offset_prefix="source",
        path="claim",
        report=report,
    )
    assert {issue.code for issue in report.errors} == {"prompt_span.offset_mismatch"}


def test_opaque_case_id_rejects_label_leakage(tmp_path: Path) -> None:
    report = Report(tmp_path)
    _validate_opaque_identifier("case_invalid_001", "case", report)
    assert any(issue.code == "leakage.nonopaque_identifier" for issue in report.errors)


def test_opaque_case_and_event_ids_accept_numeric_names(tmp_path: Path) -> None:
    report = Report(tmp_path)
    _validate_opaque_identifier("case_0042", "case", report)
    _validate_opaque_identifier(
        "event_00042",
        "event",
        report,
        allow_event_prefix=True,
    )
    assert report.errors == []


def test_scene_geometry_hash_ignores_case_identity_but_not_pose() -> None:
    first = _scene(request_id="case_0001")
    renamed = _scene(request_id="case_0002")
    moved = _scene(request_id="case_0003", center_x=1.25)
    assert scene_geometry_sha256(first) == scene_geometry_sha256(renamed)
    assert scene_geometry_sha256(first) != scene_geometry_sha256(moved)


def test_json_diff_paths_reports_only_changed_leaves() -> None:
    left = {"objects": [{"center": [1, 2, 3], "size": [1, 1, 1]}], "prompt": "a"}
    right = {"objects": [{"center": [1, 4, 3], "size": [1, 1, 1]}], "prompt": "b"}
    assert json_diff_paths(left, right) == {
        "/objects/0/center/1",
        "/prompt",
    }


def test_nested_outbound_allowlist_is_discovered() -> None:
    value = {
        "packet": {
            "judge_context_allowlist": [
                "original_prompt",
                "target_ids",
                "global_visual_evidence",
            ]
        }
    }
    assert list(_outbound_allowlists(value)) == [
        (
            "/packet/judge_context_allowlist",
            ["original_prompt", "target_ids", "global_visual_evidence"],
        )
    ]


def test_non_l1_judge_context_preserves_exact_prompt_authorization() -> None:
    fixture = DATASET_ROOT / "fixtures" / "case_0099"
    card = {
        "metric": "scale_consistency",
        "review_question": (
            "After applying only any prompt-authorized exemption, should this "
            "chair/table scale relation be penalized by L3?"
        ),
        "prompt_granularity": "fine_grained",
        "target_objects": [
            {"id": "obj_001", "description": "chair"},
            {"id": "obj_000", "description": "table"},
        ],
    }
    context = _judge_context(fixture, card)
    assert context["original_prompt"].startswith("Place a dining chair")
    assert context["prompt_authorized_deviations"] == [
        {
            "metric": "scale_consistency",
            "target_ids": ["obj_001", "obj_000"],
            "relation": "much_larger_than",
            "source": "explicit_prompt_requirement",
            "prompt_span": (
                "make the chair intentionally much larger than the table"
            ),
            "source_claim_id": "oor::chair_much_larger_than_table",
        }
    ]
    serialized = json.dumps(context, sort_keys=True)
    assert "human_semantic_label" not in serialized
    assert "construction_proposal" not in serialized
    assert "controlled_delta" not in serialized


def test_non_l1_contour_arms_are_same_pose_presentation_swaps(
    tmp_path: Path,
) -> None:
    global_views = []
    local_views = []
    contour_views = []
    for family, count, destination in (
        ("global_raw", 2, global_views),
        ("local_raw", 3, local_views),
    ):
        for index in range(count):
            path = tmp_path / f"{family}_{index}.png"
            path.write_bytes(f"{family}:{index}".encode())
            destination.append(
                {
                    "role": family,
                    "view_name": f"view_{index}",
                    "path": str(path),
                }
            )
    for index in range(3):
        path = tmp_path / f"local_contour_{index}.png"
        path.write_bytes(f"local_contour:{index}".encode())
        contour_views.append(
            {
                "view_id": f"view_{index}",
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    arms = _arm_selections(
        metric="oor",
        fixture=DATASET_ROOT / "fixtures" / "case_0001",
        paths={"global": global_views, "local_raw": local_views},
        contour_manifest={"complete": True, "contour_views": contour_views},
    )
    assert set(arms) == {
        "production_default",
        "global_only",
        "local_raw_only",
        "full_raw",
        "production_raw_swap",
        "local_contour_only",
    }
    assert [item["view_name"] for item in arms["production_default"]] == [
        item["view_name"] for item in arms["production_raw_swap"]
    ]
    assert {item["role"] for item in arms["production_default"]} == {
        "global_raw",
        "local_contour",
    }
    assert {item["role"] for item in arms["production_raw_swap"]} == {
        "global_raw",
        "local_raw",
    }


def test_non_l1_response_contract_requires_abstention_when_insufficient() -> None:
    _validate_response(
        {
            "evidence_status": "insufficient",
            "verdict": "ambiguous",
            "confidence": 0.4,
            "reason": "The target side is occluded.",
            "missing_evidence": ["an unobstructed local view"],
        }
    )
    with pytest.raises(ValueError, match="requires ambiguous"):
        _validate_response(
            {
                "evidence_status": "insufficient",
                "verdict": "valid",
                "confidence": 0.4,
                "reason": "The target side is occluded.",
                "missing_evidence": ["an unobstructed local view"],
            }
        )


@pytest.mark.skipif(
    not (DATASET_ROOT / "dataset_manifest.json").is_file(),
    reason="cal_dataset2 has not been constructed yet",
)
def test_constructed_dataset_passes_read_only_validator() -> None:
    before = _tree_content_digest(DATASET_ROOT)
    report = validate_dataset(DATASET_ROOT)
    after = _tree_content_digest(DATASET_ROOT)
    assert after == before, "validator modified the dataset"
    assert not report.errors, "\n".join(
        f"{issue.code}: {issue.path}: {issue.message}" for issue in report.errors[:100]
    )
    assert report.summary["discovered_case_count"] == EXPECTED_CASE_COUNT
    assert report.summary["event_count"] == EXPECTED_CASE_COUNT
    assert report.summary["metric_counts"] == EXPECTED_METRIC_COUNTS


def _tree_content_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
