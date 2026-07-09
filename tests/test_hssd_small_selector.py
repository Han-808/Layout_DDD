from __future__ import annotations

from pathlib import Path

from benchmark.legend.hssd.hssd_small_selector import convert_selected_small_hssd_scene
from benchmark.utils.io import read_json, write_json


def test_small_hssd_selector_picks_small_complete_scene_without_truncation(tmp_path: Path) -> None:
    hssd_root = tmp_path / "hssd-hab"
    scene_dir = hssd_root / "scenes"
    scene_dir.mkdir(parents=True)
    _write_scene(scene_dir / "too_small.scene_instance.json", "too_small", 3)
    _write_scene(scene_dir / "z_tie.scene_instance.json", "z_tie", 8)
    _write_scene(scene_dir / "a_tie.scene_instance.json", "a_tie", 8)
    _write_scene(scene_dir / "too_large.scene_instance.json", "too_large", 25)

    selected, paths, manifest_path = convert_selected_small_hssd_scene(
        hssd_root=hssd_root,
        out_dir=tmp_path / "cases",
        min_objects=6,
        max_objects=20,
        levels=["structured_basic"],
        compact_object_ids=True,
        bbox_from_scale=True,
    )

    assert selected.scene_id == "a_tie"
    assert selected.object_count == 8
    assert paths == [tmp_path / "cases" / "selected_structured_basic.json"]
    assert manifest_path == tmp_path / "cases" / "selected_manifest.json"

    case = read_json(paths[0])
    manifest = read_json(manifest_path)
    assert len(case["objects"]) == 8
    assert case["source"]["raw_object_instance_count"] == 8
    assert case["source"]["imported_object_count"] == 8
    assert case["source"]["max_objects"] is None
    assert case["source"]["truncated"] is False
    assert manifest["truncated"] is False
    assert manifest["selected"]["scene_id"] == "a_tie"
    assert manifest["stable_case_paths"] == [str(paths[0])]


def _write_scene(path: Path, scene_id: str, object_count: int) -> None:
    write_json(
        path,
        {
            "scene_id": scene_id,
            "object_instances": [
                {
                    "template_name": f"{scene_id}_object_{index:03d}",
                    "translation": [float(index), 0.0, float(index % 3)],
                    "non_uniform_scale": [0.5, 0.8, 0.6],
                }
                for index in range(object_count)
            ],
        },
    )
