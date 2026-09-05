from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from benchmark.non_rectangular import validate_room_layout, validate_room_program


ROOT = Path(__file__).resolve().parents[1]
EXTENSION_ROOT = (
    ROOT / "configs/generation_extensions/non_rectangular_multi_room_v1"
)
LAYOUT_ROOT = EXTENSION_ROOT / "layouts/spatiallm_selected10_v1"
PROGRAM_ROOT = EXTENSION_ROOT / "programs/spatiallm_source_types_area_density_v1"
DENSITY_PATH = EXTENSION_ROOT / "density_references/multi_room_sceneboard_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_without_identity(value: dict) -> str:
    copied = dict(value)
    observed = copied.pop("manifest_sha256")
    payload = (
        json.dumps(
            copied,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    assert hashlib.sha256(payload).hexdigest() == observed
    return observed


def test_selected10_program_cohort_reuses_types_and_area_density() -> None:
    layout_manifest = json.loads(
        (LAYOUT_ROOT / "manifest_v1.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (PROGRAM_ROOT / "manifest_v1.json").read_text(encoding="utf-8")
    )
    density = json.loads(DENSITY_PATH.read_text(encoding="utf-8"))
    _canonical_without_identity(manifest)

    assert manifest["scene_order"] == layout_manifest["selection"]["scene_order"]
    assert manifest["totals"]["scene_count"] == 10
    assert manifest["totals"]["program_count"] == 50
    assert manifest["totals"]["target_total_instances"] == {
        "min": 343,
        "max": 425,
    }
    assert manifest["policy"]["source_room_type_spelling_preserved"] is True
    assert manifest["policy"]["source_room_type_multiplicity_preserved"] is True
    assert manifest["policy"]["source_room_id_to_type_mapping_emitted"] is False
    assert manifest["policy"]["program_order_corresponds_to_room_order"] is False
    assert manifest["policy"]["density_reference"]["sha256"] == _sha256(
        DENSITY_PATH
    )
    assert not manifest["linked_layout_cohort"]["manifest_path"].startswith("/")
    assert not manifest["policy"]["density_reference"]["path"].startswith("/")

    density_min = float(density["objects_per_m2_target"]["min"])
    density_max = float(density["objects_per_m2_target"]["max"])
    layout_records = {
        item["scene_id"]: item for item in layout_manifest["scenes"]
    }
    for scene_record in manifest["scenes"]:
        scene_id = scene_record["scene_id"]
        program_root = PROGRAM_ROOT / scene_id
        program_path = program_root / "room_program.json"
        provenance_path = program_root / "source_provenance.json"
        validation_path = program_root / "program_validation.json"
        layout_path = LAYOUT_ROOT / layout_records[scene_id]["room_layout_path"]
        program = json.loads(program_path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        layout = json.loads(layout_path.read_text(encoding="utf-8"))

        program_report = validate_room_program(program)
        layout_report = validate_room_layout(layout)
        area = float(layout_report["total_floor_area_m2"])
        room_count = int(layout_report["room_count"])
        expected_min = max(room_count, int(math.floor(area * density_min + 0.5)))
        expected_max = max(
            expected_min,
            room_count,
            int(math.floor(area * density_max + 0.5)),
        )
        assert int(program_report["program_count"]) == room_count
        assert program["target_total_instances"] == {
            "min": expected_min,
            "max": expected_max,
        }
        assert scene_record["target_total_instances"] == {
            "min": expected_min,
            "max": expected_max,
        }
        assert _sha256(program_path) == scene_record["room_program_sha256"]
        assert _sha256(provenance_path) == scene_record[
            "source_provenance_sha256"
        ]
        assert _sha256(validation_path) == scene_record["validation_sha256"]
        assert provenance["output"]["room_program_sha256"] == _sha256(
            program_path
        )
        assert provenance["conversion"][
            "source_room_id_to_type_mapping_emitted"
        ] is False
        assert provenance["conversion"][
            "program_order_corresponds_to_room_order"
        ] is False
        assert not provenance["linked_layout"]["path"].startswith("/")
        assert "rooms" not in provenance["source"]
        observed_multiset = Counter(
            item["room_type"] for item in program["programs"]
        )
        expected_multiset = {
            item["room_type"]: item["count"]
            for item in scene_record["room_type_multiset"]
        }
        assert dict(observed_multiset) == expected_multiset
        ordered_types = [item["room_type"] for item in program["programs"]]
        assert ordered_types == sorted(ordered_types)
        assert validation["linked_room_layout_sha256"] == _sha256(layout_path)
