from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random

import pytest

from scripts.build_l3_mutation_dataset import (
    ANGLE_FAMILY,
    MIXED_FAMILY,
    SCALE_FAMILY,
    SWAP_FAMILY,
    apply_angle_mutation,
    apply_cross_scene_replacement,
    apply_scale_mutation,
    build_variant,
    load_config,
    mutation_diversity_summary,
    validate_config,
    validate_mutation_diversity,
)
from scripts.render_l3_mutation_dataset import _validate_render_manifest
from scripts.serve_l3_mutation_review import (
    HUMAN_REVIEW_SCHEMA_VERSION,
    validate_review_payload,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "experiments" / "l3_mutation90_v1.yaml"
)


def test_experiment_config_freezes_requested_counts_and_safe_bounds() -> None:
    config = load_config(CONFIG_PATH)
    validate_config(config)

    assert config["sample"]["main_scene_count"] == 20
    assert config["sample"]["extra_mixed_scene_count"] == 10
    assert config["sample"]["min_objects"] == 8
    assert config["sample"]["max_objects"] == 25
    assert config["sample"]["min_boundary_area_m2"] == 5.0
    assert config["sample"]["max_boundary_aspect_ratio"] == 4.0
    assert min(config["mutations"]["single_target_counts"]) == 1
    assert max(config["mutations"]["single_target_counts"]) == 8
    assert min(config["mutations"]["replacement_target_counts"]) >= 2
    factors = [
        factor
        for values in config["mutations"]["scale_factors"].values()
        for factor in values
    ]
    assert min(factors) >= 0.5
    assert max(factors) <= 1.8


def test_angle_mutation_changes_only_selected_z_rotations() -> None:
    scene = _scene("target", 8)
    before = deepcopy(scene)

    operations = apply_angle_mutation(
        scene,
        count=3,
        degrees=[30, 60, 90],
        rng=random.Random(7),
    )

    assert len(operations) == 3
    changed = {operation["object_ids"][0] for operation in operations}
    for old, new in zip(before["objects"], scene["objects"]):
        if old["id"] in changed:
            assert old["rotation"][:2] == new["rotation"][:2]
            assert old["rotation"][2] != new["rotation"][2]
        else:
            assert old == new


def test_scale_mutation_is_uniform_and_preserves_bottom_plane() -> None:
    scene = _scene("target", 8)

    operations = apply_scale_mutation(
        scene,
        count=4,
        factors=[0.6, 0.8, 1.3, 1.6],
        rng=random.Random(11),
    )

    assert len(operations) == 4
    assert {operation["factor"] < 1.0 for operation in operations} == {
        True,
        False,
    }
    for operation in operations:
        assert operation["before"]["bottom_z"] == pytest.approx(
            operation["after"]["bottom_z"]
        )
        before = operation["before"]["size"]
        after = operation["after"]["size"]
        factor = operation["factor"]
        assert after == pytest.approx([value * factor for value in before])


def test_cross_scene_replacement_uses_multiple_external_donors() -> None:
    scene = _scene("target", 10, category_offset=0)
    donors = {
        "donor_a": _scene("donor_a", 10, category_offset=30),
        "donor_b": _scene("donor_b", 10, category_offset=60),
    }
    catalog = _catalog(scene, *donors.values())
    original_jids = {item["id"]: item["jid"] for item in scene["objects"]}

    operations = apply_cross_scene_replacement(
        scene,
        target_count=5,
        rng=random.Random(19),
        asset_catalog=catalog,
        donor_scenes=donors,
    )

    assert len(operations) == 5
    assert {
        operation["operation"] for operation in operations
    } == {"cross_scene_asset_replacement"}
    assert all(
        operation["donor"]["source_id"] in donors
        for operation in operations
    )
    changed = {operation["object_ids"][0] for operation in operations}
    assert len(changed) == 5
    assert len(changed) < len(scene["objects"])
    assert all(
        scene["objects"][int(object_id.rsplit("_", 1)[1])]["jid"]
        != original_jids[object_id]
        for object_id in changed
    )


def test_mixed_variant_contains_all_operations_and_caps_unique_targets() -> None:
    config = load_config(CONFIG_PATH)
    source = _scene("source", 14, category_offset=0)
    donors = {
        "donor_a": _scene("donor_a", 14, category_offset=30),
        "donor_b": _scene("donor_b", 14, category_offset=60),
    }
    catalog = _catalog(source, *donors.values())

    result = build_variant(
        source,
        family=MIXED_FAMILY,
        schedule_index=19,
        source_id="source_001",
        config=config,
        asset_catalog=catalog,
        donor_scenes=donors,
    )

    names = {
        operation["operation"]
        for operation in result["mutation"]["operations"]
    }
    assert names == {
        "rotate_about_z",
        "cross_scene_asset_replacement",
        "uniform_scale",
    }
    assert 5 <= result["mutation"]["modified_object_count"] <= 8


def test_family_variants_preserve_object_identity() -> None:
    config = load_config(CONFIG_PATH)
    source = _scene("source", 12, category_offset=0)
    donors = {
        "donor_a": _scene("donor_a", 12, category_offset=30),
        "donor_b": _scene("donor_b", 12, category_offset=60),
    }
    catalog = _catalog(source, *donors.values())
    expected_ids = [item["id"] for item in source["objects"]]

    for family in (ANGLE_FAMILY, SWAP_FAMILY, SCALE_FAMILY):
        result = build_variant(
            source,
            family=family,
            schedule_index=8,
            source_id="source_001",
            config=config,
            asset_catalog=catalog,
            donor_scenes=donors,
        )
        assert [
            item["id"] for item in result["scene"]["objects"]
        ] == expected_ids


def test_render_validation_requires_full_asset_mesh_and_identity(
    tmp_path: Path,
) -> None:
    views = []
    for name in ("top", "perspective", "identity_map"):
        path = tmp_path / f"{name}.png"
        path.write_bytes(b"image")
        views.append({"name": name, "path": str(path)})
    manifest = {
        "views": views,
        "render_validation": {
            "blank_views": [],
            "identity_map": {"status": "verified"},
        },
        "asset_coverage": {
            "object_count": 8,
            "asset_mesh_count": 8,
            "bbox_proxy_count": 0,
        },
    }
    _validate_render_manifest(manifest, expected_object_count=8)

    broken = deepcopy(manifest)
    broken["asset_coverage"]["asset_mesh_count"] = 7
    with pytest.raises(ValueError, match="bbox proxies"):
        _validate_render_manifest(broken, expected_object_count=8)


def test_review_payload_is_blind_and_strict() -> None:
    review_data = {
        "experiment_id": "experiment",
        "dataset_fingerprint": "fingerprint",
        "cases": [{"review_id": "R001"}],
        "annotation_contract": {
            "overall_labels": ["", "valid", "invalid", "ambiguous"],
            "severity_labels": ["", "none", "minor", "moderate", "major"],
            "issue_labels": [
                "orientation_or_function",
                "object_compatibility",
                "scale_consistency",
                "style_consistency",
                "other",
            ],
            "evidence_labels": [
                "",
                "sufficient",
                "insufficient",
                "uncertain",
            ],
            "render_labels": ["", "works", "broken", "uncertain"],
        },
    }
    payload = {
        "schema_version": HUMAN_REVIEW_SCHEMA_VERSION,
        "experiment_id": "experiment",
        "dataset_fingerprint": "fingerprint",
        "answers": {
            "R001": {
                "overall_label": "invalid",
                "severity": "minor",
                "issues": ["scale_consistency"],
                "evidence_sufficiency": "sufficient",
                "render_integrity": "works",
                "notes": "",
                "reviewed": True,
            }
        },
    }

    assert validate_review_payload(
        payload, review_data=review_data
    ) == payload
    payload["answers"]["R001"]["metric_verdict"] = "invalid"
    with pytest.raises(ValueError, match="fields"):
        validate_review_payload(payload, review_data=review_data)


def _scene(
    scene_id: str,
    count: int,
    *,
    category_offset: int = 0,
) -> dict:
    return {
        "scene_id": scene_id,
        "scene_type": "test room",
        "boundary": [[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]],
        "scene_height": 3.0,
        "objects": [
            {
                "id": f"scene_object_{index:03d}",
                "object_id": f"scene_object_{index:03d}",
                "jid": f"asset_{scene_id}_{index:03d}",
                "desc": f"test asset {index}",
                "short_desc": f"asset {index}",
                "size": [
                    0.5 + (index % 3) * 0.08,
                    0.45 + (index % 4) * 0.07,
                    0.6 + (index % 5) * 0.1,
                ],
                "center": [
                    0.8 + (index % 4) * 1.5,
                    0.8 + (index // 4) * 1.5,
                    (0.6 + (index % 5) * 0.1) / 2.0,
                ],
                "rotation": [0.0, 0.0, float(index * 5)],
                "_category_index": category_offset + index,
            }
            for index in range(count)
        ],
    }


def _catalog(*scenes: dict) -> dict[str, dict[str, str]]:
    return {
        item["jid"]: {
            "name_en": item["jid"],
            "class_en": f"class_{item['_category_index']}",
            "retrieval_class_en": f"group_{item['_category_index']}",
        }
        for scene in scenes
        for item in scene["objects"]
    }
