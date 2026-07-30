from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from benchmark.game_scene.counter_strike import (
    COUNTER_STRIKE_BENCHMARK_CONFIG_SCHEMA,
    COUNTER_STRIKE_CASE_CONTRACT_SCHEMA,
    CounterStrikeConfigError,
    CounterStrikeContractError,
    load_counter_strike_benchmark_config,
    load_counter_strike_case_contract,
)


ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_CONFIG = ROOT / "configs" / "game" / "counter_strike" / "benchmark_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_scene(
    *,
    unit_scale: float = 0.5,
    source_up_axis: str = "y",
    translation: list[float] | None = None,
) -> dict:
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": "cs_test_scene",
        "request_id": "cs_test_request",
        "scene_type": "counter_strike_static_arena",
        "boundary": [[0.0, 0.0], [40.0, 0.0], [40.0, 50.0], [0.0, 50.0]],
        "scene_height": 10.0,
        "objects": [],
        "metadata": {
            "coordinate_frame": {
                "origin": "room_min_corner_floor",
                "axes": "x_width_y_depth_z_up",
                "unit": "meter",
                "rotation_unit": "degree",
            },
            "game_scene_import": {
                "probe_schema_version": "game_scene_probe_v1",
                "source_up_axis": source_up_axis,
                "unit_scale": unit_scale,
                "translation_applied": translation or [10.0, 20.0, 30.0],
            },
        },
    }


def _contract(source_path: Path, *, unit_scale: float = 0.5) -> dict:
    return {
        "contract_version": "counter_strike_case_contract_v1",
        "case_id": "cs_test",
        "corpus_id": "test-corpus",
        "source_frame": {"up_axis": "y", "unit_scale": unit_scale},
        "source_assertions": [
            {
                "path": source_path.name,
                "sha256": _sha256(source_path),
                "evidence": "frozen spawn declarations",
            }
        ],
        "team_spawns": {
            "team_a": {
                "points": [[2.0, 4.0, 6.0]],
                "jitter_radius_m": 1.25,
            },
            "team_b": {
                "points": [[-2.0, 0.0, -6.0]],
                "jitter_radius_m": 0.0,
            },
        },
        "annotation": {
            "source": "benchmark_audited_source_declaration",
            "score_authority": False,
        },
    }


def _write_contract(root: Path, payload: dict) -> Path:
    path = root / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_counter_strike_schemas_are_valid_draft_2020_12() -> None:
    Draft202012Validator.check_schema(COUNTER_STRIKE_BENCHMARK_CONFIG_SCHEMA)
    Draft202012Validator.check_schema(COUNTER_STRIKE_CASE_CONTRACT_SCHEMA)


def test_checked_in_benchmark_config_loads_as_frozen_profile() -> None:
    loaded = load_counter_strike_benchmark_config(BENCHMARK_CONFIG)

    assert loaded.path == BENCHMARK_CONFIG.resolve()
    assert loaded.sha256 == _sha256(BENCHMARK_CONFIG)
    assert loaded.raw["status"] == "frozen"
    assert set(loaded.raw["l4_metrics"]) == {
        "zone_clarity",
        "route_structure",
        "spawn_balance",
        "landmark_legibility",
        "cover_diversity",
    }


def test_benchmark_config_rejects_unknown_or_inert_fields(tmp_path: Path) -> None:
    text = BENCHMARK_CONFIG.read_text(encoding="utf-8")
    path = tmp_path / "benchmark.yaml"
    path.write_text(f"{text}\ninert_future_option: true\n", encoding="utf-8")

    with pytest.raises(CounterStrikeConfigError) as caught:
        load_counter_strike_benchmark_config(path)

    assert caught.value.code == "invalid_schema"
    assert "Additional properties are not allowed" in str(caught.value)


def test_case_loader_verifies_source_and_uses_recorded_y_up_import_transform(
    tmp_path: Path,
) -> None:
    source = tmp_path / "map.js"
    source.write_text("export const teamASpawn = [2, 4, 6];\n", encoding="utf-8")
    contract_path = _write_contract(tmp_path, _contract(source))

    loaded = load_counter_strike_case_contract(
        contract_path,
        source_root=tmp_path,
        canonical_scene=_canonical_scene(),
    )

    assert loaded.case_id == "cs_test"
    assert loaded.source_assertions[0].resolved_path == source.resolve()
    assert loaded.source_assertions[0].sha256 == _sha256(source)
    # Exporter's Y-up basis is (x, y, z) -> (x, -z, y), then translation.
    assert loaded.canonical_team_spawns["team_a"]["points"][0] == pytest.approx(
        [11.0, 17.0, 32.0]
    )
    assert loaded.canonical_team_spawns["team_b"]["points"][0] == pytest.approx(
        [9.0, 23.0, 30.0]
    )
    # The contract names this value in meters, so source unit scale is not
    # applied a second time.
    assert loaded.canonical_team_spawns["team_a"]["jitter_radius_m"] == 1.25


def test_case_loader_rejects_source_hash_drift(tmp_path: Path) -> None:
    source = tmp_path / "map.js"
    source.write_text("const spawn = [0, 0, 0];\n", encoding="utf-8")
    payload = _contract(source)
    contract_path = _write_contract(tmp_path, payload)
    source.write_text("const spawn = [1, 0, 0];\n", encoding="utf-8")

    with pytest.raises(CounterStrikeContractError) as caught:
        load_counter_strike_case_contract(
            contract_path,
            source_root=tmp_path,
            canonical_scene=_canonical_scene(),
        )

    assert caught.value.code == "source_sha256_mismatch"
    assert "expected" in str(caught.value)
    assert "got" in str(caught.value)


def test_case_loader_rejects_source_frame_disagreement(tmp_path: Path) -> None:
    source = tmp_path / "map.js"
    source.write_text("const spawn = [0, 0, 0];\n", encoding="utf-8")
    contract_path = _write_contract(tmp_path, _contract(source))

    with pytest.raises(CounterStrikeContractError) as caught:
        load_counter_strike_case_contract(
            contract_path,
            source_root=tmp_path,
            canonical_scene=_canonical_scene(unit_scale=1.0),
        )

    assert caught.value.code == "source_frame_mismatch"


def test_case_loader_rejects_missing_or_malformed_import_transform(
    tmp_path: Path,
) -> None:
    source = tmp_path / "map.js"
    source.write_text("const spawn = [0, 0, 0];\n", encoding="utf-8")
    contract_path = _write_contract(tmp_path, _contract(source))
    scene = _canonical_scene()
    scene["metadata"]["game_scene_import"]["translation_applied"] = [0.0, 1.0]

    with pytest.raises(CounterStrikeContractError) as caught:
        load_counter_strike_case_contract(
            contract_path,
            source_root=tmp_path,
            canonical_scene=scene,
        )

    assert caught.value.code == "canonical_import_transform_invalid"
    assert "translation_applied" in str(caught.value)


def test_case_contract_schema_rejects_authoritative_annotations_and_unknown_keys(
    tmp_path: Path,
) -> None:
    source = tmp_path / "map.js"
    source.write_text("const spawn = [0, 0, 0];\n", encoding="utf-8")
    authoritative = _contract(source)
    authoritative["annotation"]["score_authority"] = True
    authoritative["unscoped_metric_hint"] = "do not accept"
    contract_path = _write_contract(tmp_path, authoritative)

    with pytest.raises(CounterStrikeContractError) as caught:
        load_counter_strike_case_contract(
            contract_path,
            source_root=tmp_path,
            canonical_scene=_canonical_scene(),
        )

    assert caught.value.code == "invalid_schema"


def test_case_contract_rejects_path_traversal_before_source_read(
    tmp_path: Path,
) -> None:
    source = tmp_path / "map.js"
    source.write_text("const spawn = [0, 0, 0];\n", encoding="utf-8")
    payload = _contract(source)
    payload["source_assertions"][0]["path"] = "../map.js"
    contract_path = _write_contract(tmp_path, payload)

    with pytest.raises(CounterStrikeContractError) as caught:
        load_counter_strike_case_contract(
            contract_path,
            source_root=tmp_path,
            canonical_scene=_canonical_scene(),
        )

    assert caught.value.code == "invalid_schema"


def test_duplicate_source_assertions_are_explicit_contract_failure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "map.js"
    source.write_text("const spawn = [0, 0, 0];\n", encoding="utf-8")
    payload = _contract(source)
    payload["source_assertions"].append(
        deepcopy(payload["source_assertions"][0])
    )
    contract_path = _write_contract(tmp_path, payload)

    with pytest.raises(CounterStrikeContractError) as caught:
        load_counter_strike_case_contract(
            contract_path,
            source_root=tmp_path,
            canonical_scene=_canonical_scene(),
        )

    assert caught.value.code == "duplicate_source_assertion"
