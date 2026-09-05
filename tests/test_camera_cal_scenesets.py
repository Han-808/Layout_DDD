from __future__ import annotations

import json

from scripts.build_camera_cal_scenesets import DEFAULT_OUTPUT_ROOT


def test_camera_cal_scenesets_is_complete_when_built() -> None:
    if not DEFAULT_OUTPUT_ROOT.is_dir():
        return
    manifest = json.loads(
        (DEFAULT_OUTPUT_ROOT / "dataset_manifest.json").read_text()
    )
    assert manifest["dataset_id"] == "camera_cal_scenesets"
    assert manifest["case_count"] == 30
    assert manifest["metric_count"] == 5
    assert manifest["case_ids"] == [
        f"N{number:03d}" for number in range(1, 31)
    ]
    assert manifest["annotations"]["all_scenes_reviewed"] is True
    assert manifest["annotations"]["render_integrity"] == "usable"
    assert manifest["annotations"]["status_counts"] == {
        "valid": 97,
        "invalid": 53,
        "unclear": 0,
        "unreviewed": 0,
    }
    assert manifest["all_cases_ready"] is True
    for case in manifest["cases"]:
        root = DEFAULT_OUTPUT_ROOT / case["path"]
        assert (root / "case_manifest.json").is_file()
        assert (root / "annotation.json").is_file()
        assert (DEFAULT_OUTPUT_ROOT / case["canonical_scene"]).is_file()
        assert (DEFAULT_OUTPUT_ROOT / case["blend"]).is_file()


def test_camera_cal_case_annotations_match_dataset_partition() -> None:
    if not DEFAULT_OUTPUT_ROOT.is_dir():
        return
    dataset = json.loads(
        (DEFAULT_OUTPUT_ROOT / "annotations.json").read_text()
    )
    assert set(dataset["answers"]) == {
        f"N{number:03d}" for number in range(1, 31)
    }
    for case_id, answer in dataset["answers"].items():
        per_case = json.loads(
            (DEFAULT_OUTPUT_ROOT / case_id / "annotation.json").read_text()
        )
        assert per_case["case_id"] == case_id
        assert per_case["reviewed"] is True
        assert per_case["render_integrity"] == "usable"
        assert per_case["metrics"] == answer["metrics"]
