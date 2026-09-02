from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from benchmark.scene_generation.non_rectangular_multi_room.spatiallm_programs import (
    SpatialLMProgramError,
    build_spatiallm_program_cohort,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/non_rectangular"
SCENE_ID = "scene_900000"


def _json_bytes(value: dict) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _source_root(
    tmp_path: Path,
    *,
    inconsistent: bool = False,
    train: bool = False,
) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    with (source / "split.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "room_type", "scene_id", "room_id", "sample", "split"],
        )
        writer.writeheader()
        for room_id, room_type in ((0, "kitchen"), (1, "living room")):
            for sample in (0, 1):
                label = (
                    "bedroom"
                    if inconsistent and room_id == 1 and sample == 1
                    else room_type
                )
                writer.writerow(
                    {
                        "id": f"{SCENE_ID}_{room_id:02d}_{sample}",
                        "room_type": label,
                        "scene_id": SCENE_ID,
                        "room_id": room_id,
                        "sample": sample,
                        "split": "train" if train else "reserved",
                    }
                )
    return source


def _layout_cohort(tmp_path: Path, source: Path) -> Path:
    root = tmp_path / "layouts"
    scene_root = root / SCENE_ID
    scene_root.mkdir(parents=True)
    layout = json.loads(
        (FIXTURES / "simple_multi_room.json").read_text(encoding="utf-8")
    )
    layout["layout_id"] = SCENE_ID
    layout_bytes = _json_bytes(layout)
    layout_path = scene_root / "room_layout.json"
    layout_path.write_bytes(layout_bytes)
    split_sha256 = hashlib.sha256((source / "split.csv").read_bytes()).hexdigest()
    manifest = {
        "schema_version": "non_rectangular_spatiallm_layout_cohort_v1",
        "cohort_id": "fixture_layout_cohort_v1",
        "manifest_sha256": "f" * 64,
        "source": {
            "dataset_revision": "a" * 40,
            "split_sha256": split_sha256,
        },
        "selection": {"scene_order": [SCENE_ID]},
        "scenes": [
            {
                "scene_id": SCENE_ID,
                "room_layout_path": f"{SCENE_ID}/room_layout.json",
                "room_layout_sha256": hashlib.sha256(layout_bytes).hexdigest(),
            }
        ],
    }
    (root / "manifest_v1.json").write_bytes(_json_bytes(manifest))
    return root


def _density_reference(tmp_path: Path) -> Path:
    path = tmp_path / "density_reference.json"
    path.write_bytes(
        _json_bytes(
            {
                "schema_version": "non_rectangular_object_density_reference_v1",
                "policy_id": "fixture_area_density_v1",
                "objects_per_m2_target": {"min": 0.5, "max": 0.6},
                "integer_rounding_policy": "floor_x_plus_0.5_v1",
            }
        )
    )
    return path


def test_program_builder_reuses_exact_multiset_without_room_binding(
    tmp_path: Path,
) -> None:
    source = _source_root(tmp_path)
    layouts = _layout_cohort(tmp_path, source)
    output = tmp_path / "programs"

    manifest = build_spatiallm_program_cohort(
        source_root=source,
        layout_cohort_root=layouts,
        output_root=output,
        cohort_id="fixture_source_types_area_density_v1",
        density_reference_path=_density_reference(tmp_path),
    )

    program = json.loads(
        (output / SCENE_ID / "room_program.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (output / SCENE_ID / "source_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert program["program_order"] == ["kitchen_01", "living_room_01"]
    assert program["programs"] == [
        {"program_id": "kitchen_01", "room_type": "kitchen"},
        {"program_id": "living_room_01", "room_type": "living room"},
    ]
    assert program["target_total_instances"] == {"min": 7, "max": 9}
    assert provenance["conversion"]["source_room_id_to_type_mapping_emitted"] is False
    assert provenance["conversion"]["program_order_corresponds_to_room_order"] is False
    assert "rooms" not in provenance["source"]
    assert "room_mapping" not in provenance["source"]
    assert manifest["totals"]["program_count"] == 2


def test_program_builder_rejects_inconsistent_source_type_across_samples(
    tmp_path: Path,
) -> None:
    source = _source_root(tmp_path, inconsistent=True)
    layouts = _layout_cohort(tmp_path, source)
    with pytest.raises(
        SpatialLMProgramError,
        match="source room_type differs across samples",
    ):
        build_spatiallm_program_cohort(
            source_root=source,
            layout_cohort_root=layouts,
            output_root=tmp_path / "programs",
            cohort_id="fixture_source_types_area_density_v1",
            density_reference_path=_density_reference(tmp_path),
        )


def test_program_builder_rejects_train_scene(tmp_path: Path) -> None:
    source = _source_root(tmp_path, train=True)
    layouts = _layout_cohort(tmp_path, source)
    with pytest.raises(SpatialLMProgramError, match="contains train rows"):
        build_spatiallm_program_cohort(
            source_root=source,
            layout_cohort_root=layouts,
            output_root=tmp_path / "programs",
            cohort_id="fixture_source_types_area_density_v1",
            density_reference_path=_density_reference(tmp_path),
        )
