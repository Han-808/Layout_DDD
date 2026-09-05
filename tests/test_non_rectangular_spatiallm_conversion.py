from __future__ import annotations

import csv
import json
from pathlib import Path
import zipfile

import pytest
from shapely.geometry import LinearRing

from benchmark.non_rectangular import validate_room_layout
from benchmark.scene_generation.non_rectangular_multi_room.spatiallm_conversion import (
    SpatialLMConversionError,
    convert_spatiallm_cohort,
)


SCENE_ID = "scene_900000"


def _room_text(
    points: list[tuple[float, float]],
    *,
    bbox_label: str,
) -> str:
    lines = []
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        lines.append(
            f"wall_{index}=Wall({start[0]},{start[1]},0.0,"
            f"{end[0]},{end[1]},0.0,2.8,0.1)"
        )
    lines.extend(
        [
            f"door_0=Door(wall_0,0.0,0.0,0.0,0.8,2.0)",
            f"window_0=Window(wall_1,0.0,0.0,1.0,1.0,1.0)",
            f"bbox_0=Bbox({bbox_label},1.0,1.0,0.5,0.0,1.0,1.0,1.0)",
        ]
    )
    return "\n".join(lines) + "\n"


def _source_root(tmp_path: Path, *, mismatch: bool = False, train: bool = False) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    clockwise_square = [(0.0, 0.0), (0.0, 3.0), (3.0, 3.0), (3.0, 0.0)]
    changed_square = [(0.0, 0.0), (0.0, 3.0), (3.1, 3.0), (3.1, 0.0)]
    l_shape = [
        (3.2, 0.0),
        (6.2, 0.0),
        (6.2, 3.0),
        (5.0, 3.0),
        (5.0, 1.2),
        (3.2, 1.2),
    ]
    with zipfile.ZipFile(source / "chunk_000.zip", "w") as archive:
        archive.writestr(
            f"{SCENE_ID}_00_0.txt",
            _room_text(clockwise_square, bbox_label="kitchen_counter"),
        )
        archive.writestr(
            f"{SCENE_ID}_00_1.txt",
            _room_text(
                changed_square if mismatch else clockwise_square,
                bbox_label="kitchen_stool",
            ),
        )
        archive.writestr(
            f"{SCENE_ID}_01_0.txt",
            _room_text(l_shape, bbox_label="living_room_sofa"),
        )
        archive.writestr(
            f"{SCENE_ID}_01_1.txt",
            _room_text(l_shape, bbox_label="living_room_table"),
        )
    with (source / "split.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "room_type", "scene_id", "room_id", "sample", "split"],
        )
        writer.writeheader()
        for room_id, room_type in ((0, "kitchen"), (1, "living room")):
            for sample in (0, 1):
                writer.writerow(
                    {
                        "id": f"{SCENE_ID}_{room_id:02d}_{sample}",
                        "room_type": room_type,
                        "scene_id": SCENE_ID,
                        "room_id": room_id,
                        "sample": sample,
                        "split": "train" if train else "reserved",
                    }
                )
    return source


def test_conversion_writes_label_free_valid_layout_and_provenance(
    tmp_path: Path,
) -> None:
    source = _source_root(tmp_path)
    output = tmp_path / "cohort"
    preview = tmp_path / "preview"

    manifest = convert_spatiallm_cohort(
        source_root=source,
        output_root=output,
        preview_root=preview,
        cohort_id="fixture_spatiallm_cohort_v1",
        scene_ids=[SCENE_ID],
        dataset_revision="a" * 40,
    )

    layout = json.loads((output / SCENE_ID / "room_layout.json").read_text())
    provenance = json.loads(
        (output / SCENE_ID / "source_provenance.json").read_text()
    )
    report = json.loads(
        (output / SCENE_ID / "geometry_validation.json").read_text()
    )
    assert validate_room_layout(layout)["valid"] is True
    assert layout["room_order"] == ["room_000", "room_001"]
    assert all(
        LinearRing(room["floor_polygon_xy"]).is_ccw for room in layout["rooms"]
    )
    assert layout["rooms"][1]["floor_polygon_xy"][0] == [3.2, 0.0]
    assert report["adjacency"]["connected"] is True
    assert report["geometry_repair_applied"] is False
    assert report["coordinates_transformed"] is False
    assert provenance["conversion"]["winding_reversed_room_ids"] == ["room_000"]
    assert provenance["conversion"]["source_room_type_labels_in_output"] is False
    assert all(
        room["wall_geometry_identical_across_samples"]
        for room in provenance["source"]["rooms"]
    )
    assert manifest["totals"] == {
        "scene_count": 1,
        "room_count": 2,
        "wall_segment_count": 10,
    }
    assert (preview / "selected10_contact_sheet.png").is_file()
    serialized = json.dumps(
        {"manifest": manifest, "layout": layout, "provenance": provenance}
    )
    assert "kitchen_counter" not in serialized
    assert "living room" not in serialized


def test_conversion_rejects_wall_geometry_drift_across_samples(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        SpatialLMConversionError,
        match="wall geometry differs across source samples",
    ):
        convert_spatiallm_cohort(
            source_root=_source_root(tmp_path, mismatch=True),
            output_root=tmp_path / "cohort",
            cohort_id="fixture_spatiallm_cohort_v1",
            scene_ids=[SCENE_ID],
            dataset_revision="a" * 40,
        )


def test_conversion_rejects_train_scene(tmp_path: Path) -> None:
    with pytest.raises(SpatialLMConversionError, match="contains train rows"):
        convert_spatiallm_cohort(
            source_root=_source_root(tmp_path, train=True),
            output_root=tmp_path / "cohort",
            cohort_id="fixture_spatiallm_cohort_v1",
            scene_ids=[SCENE_ID],
            dataset_revision="a" * 40,
        )
