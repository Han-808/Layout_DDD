from __future__ import annotations

from pathlib import Path

from jsonschema import Draft202012Validator

from benchmark.datasets.hssd_hab_converter import convert_hssd_hab
from benchmark.utils.io import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]


def test_hssd_converter_writes_prompt_and_structured_basic_cases(tmp_path: Path) -> None:
    hssd_root = tmp_path / "hssd"
    scene_dir = hssd_root / "scenes"
    scene_dir.mkdir(parents=True)
    write_json(
        scene_dir / "tiny.scene_instance.json",
        {
            "scene_id": "tiny_scene",
            "object_instances": [
                {"id": "chair_001", "category": "chair", "translation": [0.0, 0.0, 0.0]},
                {"id": "desk_001", "category": "desk", "translation": [1.0, 0.0, 0.0]},
            ],
        },
    )

    out_dir = tmp_path / "cases"
    paths = convert_hssd_hab(hssd_root=hssd_root, out_dir=out_dir, limit=1, levels=["prompt_only", "structured_basic"])

    assert len(paths) == 2
    validator = Draft202012Validator(read_json(ROOT / "schemas" / "bm_instance.schema.json"))
    for path in paths:
        assert list(validator.iter_errors(read_json(path))) == []


def test_hssd_converter_can_write_small_compact_case(tmp_path: Path) -> None:
    hssd_root = tmp_path / "hssd"
    scene_dir = hssd_root / "scenes"
    scene_dir.mkdir(parents=True)
    write_json(
        scene_dir / "large.scene_instance.json",
        {
            "scene_id": "large_scene",
            "object_instances": [
                {"template_name": "hash_a", "translation": [0.0, 0.0, 0.0]},
                {"template_name": "hash_b", "translation": [1.0, 0.0, 2.0]},
                {"template_name": "hash_c", "translation": [2.0, 0.0, 3.0]},
            ],
        },
    )

    out_dir = tmp_path / "cases"
    paths = convert_hssd_hab(
        hssd_root=hssd_root,
        out_dir=out_dir,
        limit=1,
        levels=["structured_relation"],
        max_objects=2,
        compact_object_ids=True,
    )

    assert len(paths) == 1
    case = read_json(paths[0])
    assert [obj["id"] for obj in case["objects"]] == ["object_001", "object_002"]
    assert [obj["category"] for obj in case["objects"]] == ["hssd_object_001", "hssd_object_002"]
    assert case["objects"][0]["source_id"] == "hash_a_001"
    assert len(case["objects"]) == 2
    assert len(case["relations"]) == 1
    assert case["relations"][0]["subject"] == "object_001"
    assert case["room"]["boundary"] == [[-1.5, -1.5], [2.5, -1.5], [2.5, 3.5], [-1.5, 3.5]]

    validator = Draft202012Validator(read_json(ROOT / "schemas" / "bm_instance.schema.json"))
    assert list(validator.iter_errors(case)) == []
