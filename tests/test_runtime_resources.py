from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from benchmark.game_scene.mode import load_game_mode_config
from benchmark.resources import packaged_resource_path, runtime_resource_path
from benchmark.visual_judge.render_views import (
    _camera_evidence_implementation_contract,
)


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_RESOURCES = (
    "schemas/generator_catalog_placement_v1.schema.json",
    "schemas/generator_layout_v1.schema.json",
    "configs/evaluation/metric_profile_canonical_v2.yaml",
    "configs/evaluation/metric_profile_game_canonical_v1.yaml",
    "configs/grouping/vlm_visual_evidence_scope_v2.yaml",
    "configs/game/game_mode_canonical_v1.yaml",
    "configs/game/counter_strike_static_arena_style_v1.json",
    "configs/game/counter_strike/benchmark_v1.yaml",
)


@pytest.mark.parametrize("relative", RUNTIME_RESOURCES)
def test_packaged_runtime_resource_matches_source_bytes(relative: str) -> None:
    source = ROOT / relative
    packaged = packaged_resource_path(relative)
    assert packaged.read_bytes() == source.read_bytes()
    assert runtime_resource_path(relative) == packaged


@pytest.mark.parametrize(
    "relative",
    (
        "schemas/generator_catalog_placement_v1.schema.json",
        "schemas/generator_layout_v1.schema.json",
    ),
)
def test_packaged_adapter_schema_is_valid_draft_2020_12(relative: str) -> None:
    value = json.loads(packaged_resource_path(relative).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(value)


def test_packaged_game_config_keeps_relative_resource_topology() -> None:
    mode = load_game_mode_config(
        packaged_resource_path("configs/game/game_mode_canonical_v1.yaml")
    )
    assert mode.evaluation_profile_path == packaged_resource_path(
        "configs/evaluation/metric_profile_game_canonical_v1.yaml"
    )
    assert mode.default_visual_style_spec_path == packaged_resource_path(
        "configs/game/counter_strike_static_arena_style_v1.json"
    )
    assert mode.raw["evaluation"]["active_metrics"] == [
        "collision",
        "navigability",
        "style_consistency",
    ]


def test_camera_implementation_fingerprint_is_path_independent() -> None:
    contract = _camera_evidence_implementation_contract()
    manifest = json.loads(
        (ROOT / "configs/resources/runtime_resources_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["sha256"] == manifest[
        "camera_evidence_implementation_sha256"
    ]
    assert hashlib.sha256(
        json.dumps(
            contract["files"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest() == contract["sha256"]


def test_runtime_resource_rejects_path_escape() -> None:
    with pytest.raises(ValueError, match="package-relative"):
        runtime_resource_path("../metric_profile.yaml")
