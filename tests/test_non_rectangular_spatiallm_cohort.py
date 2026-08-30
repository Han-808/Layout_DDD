from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmark.non_rectangular import validate_room_layout
from benchmark.scene_generation.non_rectangular_multi_room.architecture import (
    build_polygon_architecture,
)


ROOT = Path(__file__).resolve().parents[1]
COHORT_ROOT = (
    ROOT
    / "configs/generation_extensions/non_rectangular_multi_room_v1/layouts"
    / "spatiallm_selected10_v1"
)
EXPECTED_SCENES = [
    "scene_011760",
    "scene_012102",
    "scene_011910",
    "scene_011351",
    "scene_012088",
    "scene_011568",
    "scene_011388",
    "scene_011528",
    "scene_011972",
    "scene_011400",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_selected10_cohort_is_complete_hash_bound_and_generation_ready() -> None:
    manifest_path = COHORT_ROOT / "manifest_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["selection"]["scene_order"] == EXPECTED_SCENES
    assert manifest["selection"]["source_room_type_labels_in_output"] is False
    assert manifest["source"]["dataset_revision"] == (
        "ab8de0983f48490a0a8850baa32308982d6fbacc"
    )
    assert manifest["totals"] == {
        "scene_count": 10,
        "room_count": 50,
        "wall_segment_count": 283,
    }
    without_identity = dict(manifest)
    observed_identity = without_identity.pop("manifest_sha256")
    canonical = (
        json.dumps(
            without_identity,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == observed_identity

    for scene in manifest["scenes"]:
        scene_id = scene["scene_id"]
        scene_root = COHORT_ROOT / scene_id
        layout_path = scene_root / "room_layout.json"
        provenance_path = scene_root / "source_provenance.json"
        validation_path = scene_root / "geometry_validation.json"
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))

        report = validate_room_layout(layout)
        architecture = build_polygon_architecture(layout)
        assert report["valid"] is True
        assert report["room_count"] == scene["room_count"]
        assert report["wall_segment_count"] == scene["wall_segment_count"]
        assert _sha256(layout_path) == scene["room_layout_sha256"]
        assert _sha256(provenance_path) == scene["source_provenance_sha256"]
        assert _sha256(validation_path) == scene["geometry_validation_sha256"]
        assert provenance["output"]["room_layout_sha256"] == _sha256(layout_path)
        assert provenance["conversion"]["coordinates_transformed"] is False
        assert provenance["conversion"]["coordinates_rounded"] is False
        assert provenance["conversion"]["geometry_repair_applied"] is False
        assert provenance["conversion"]["source_room_type_labels_in_output"] is False
        assert validation["adjacency"]["connected"] is True
        assert validation["architecture_check"]["exact_shared_wall_count"] == 0
        assert validation["architecture_check"][
            "near_or_partial_wall_pairing_applied"
        ] is False
        assert all(
            set(room) == {
                "room_id",
                "floor_z_m",
                "floor_polygon_xy",
                "wall_segments",
            }
            for room in layout["rooms"]
        )
        assert len(architecture["rooms"]) == report["room_count"]
