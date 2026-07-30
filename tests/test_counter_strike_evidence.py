from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path

import pytest

from benchmark.game_scene.counter_strike import (
    GLOBAL_EVIDENCE_ROLE,
    REGIONAL_EVIDENCE_ROLE,
    CounterStrikeEvidenceError,
    load_counter_strike_benchmark_config,
    load_counter_strike_frozen_evidence,
)
from benchmark.rendering.browser import (
    BROWSER_RENDER_BACKEND,
    CONTROLLED_CAMERA_APPEARANCE_FIDELITY,
)


ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_CONFIG = ROOT / "configs" / "game" / "counter_strike" / "benchmark_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_capture(tmp_path: Path) -> tuple[Path, dict]:
    capture = tmp_path / "capture"
    capture.mkdir()
    exported_scene = capture / "probe_exported_scene.json"
    exported_scene.write_text(
        json.dumps(
            {
                "schema_version": "canonical_scene_v1",
                "scene_id": "cs_test",
                "request_id": "cs_test",
                "scene_type": "counter_strike_static_arena",
                "boundary": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "scene_height": 5,
                "objects": [],
            }
        ),
        encoding="utf-8",
    )
    global_views = []
    local_views = []
    paths = [exported_scene]
    for index in range(2):
        path = capture / f"global_global_oblique_{index:02d}.png"
        path.write_bytes(f"global-{index}".encode())
        paths.append(path)
        global_views.append(
            {
                "id": f"global_oblique_{index:02d}",
                "name": f"global_oblique_{index:02d}",
                "path": path.as_posix(),
                "scope": "global",
                "backend": "threejs_original_runtime",
                "appearance_fidelity": CONTROLLED_CAMERA_APPEARANCE_FIDELITY,
            }
        )
    for index in range(4):
        path = capture / f"local_style_region_{index:02d}.png"
        path.write_bytes(f"regional-{index}".encode())
        paths.append(path)
        local_views.append(
            {
                "id": f"style_region_{index:02d}",
                "name": f"style_region_{index:02d}",
                "path": path.as_posix(),
                "scope": "object_local",
                "role": "style_local_fallback",
                "backend": "threejs_original_runtime",
                "appearance_fidelity": CONTROLLED_CAMERA_APPEARANCE_FIDELITY,
            }
        )
    manifest = {
        "backend": BROWSER_RENDER_BACKEND,
        "exported_scene": exported_scene.as_posix(),
        "views": global_views,
        "controlled_camera": {
            "enabled": True,
            "status": "ready",
            "view_family": "canonical_high_oblique_pair_v1",
            "image_budget": 2,
            "appearance_fidelity": CONTROLLED_CAMERA_APPEARANCE_FIDELITY,
            "style_local_fallback": {
                "enabled": True,
                "status": "ready",
                "view_family": "canonical_style_region_quadrants_v1",
                "image_budget": 4,
                "views": local_views,
            },
        },
        "capture_artifacts": {
            path.relative_to(capture).as_posix(): _sha256(path)
            for path in paths
        },
    }
    (capture / "render_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return capture, manifest


def _config():
    return load_counter_strike_benchmark_config(BENCHMARK_CONFIG)


def _rewrite_manifest(capture: Path, manifest: dict) -> None:
    (capture / "render_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def test_loader_exposes_exact_ordered_immutable_six_view_bank(
    tmp_path: Path,
) -> None:
    capture, _ = _write_capture(tmp_path)

    evidence = load_counter_strike_frozen_evidence(
        capture,
        benchmark_config=_config(),
    )

    assert tuple(item.id for item in evidence.global_views) == (
        "global_oblique_00",
        "global_oblique_01",
    )
    assert tuple(item.id for item in evidence.regional_views) == (
        "style_region_00",
        "style_region_01",
        "style_region_02",
        "style_region_03",
    )
    assert tuple(item.role for item in evidence.ordered) == (
        GLOBAL_EVIDENCE_ROLE,
        GLOBAL_EVIDENCE_ROLE,
        REGIONAL_EVIDENCE_ROLE,
        REGIONAL_EVIDENCE_ROLE,
        REGIONAL_EVIDENCE_ROLE,
        REGIONAL_EVIDENCE_ROLE,
    )
    assert all(item.sha256 == _sha256(item.path) for item in evidence.ordered)
    with pytest.raises(FrozenInstanceError):
        evidence.global_views[0].role = "changed"  # type: ignore[misc]


def test_loader_rejects_incomplete_regional_bank(tmp_path: Path) -> None:
    capture, manifest = _write_capture(tmp_path)
    manifest["controlled_camera"]["style_local_fallback"]["views"].pop()
    _rewrite_manifest(capture, manifest)

    with pytest.raises(CounterStrikeEvidenceError) as caught:
        load_counter_strike_frozen_evidence(
            capture,
            benchmark_config=_config(),
        )

    assert caught.value.code == "evidence_budget_mismatch"


def test_loader_rejects_any_tampered_capture_artifact(tmp_path: Path) -> None:
    capture, manifest = _write_capture(tmp_path)
    # Tamper with a non-selected artifact too: every existing capture hash is
    # part of the trust boundary, not only the six image records.
    diagnostic = capture / "authored_diagnostic.png"
    diagnostic.write_bytes(b"before")
    manifest["capture_artifacts"][diagnostic.name] = _sha256(diagnostic)
    _rewrite_manifest(capture, manifest)
    diagnostic.write_bytes(b"after")

    with pytest.raises(CounterStrikeEvidenceError) as caught:
        load_counter_strike_frozen_evidence(
            capture,
            benchmark_config=_config(),
        )

    assert caught.value.code == "artifact_hash_mismatch"


def test_loader_rejects_capture_artifact_path_escape(tmp_path: Path) -> None:
    capture, manifest = _write_capture(tmp_path)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    manifest["capture_artifacts"]["../outside.png"] = _sha256(outside)
    _rewrite_manifest(capture, manifest)

    with pytest.raises(CounterStrikeEvidenceError) as caught:
        load_counter_strike_frozen_evidence(
            capture,
            benchmark_config=_config(),
        )

    assert caught.value.code == "artifact_path_outside_root"


def test_loader_rejects_view_that_is_not_hash_bound(tmp_path: Path) -> None:
    capture, manifest = _write_capture(tmp_path)
    replacement = capture / "unhashed.png"
    replacement.write_bytes(b"unhashed")
    manifest["views"][0]["path"] = replacement.as_posix()
    _rewrite_manifest(capture, manifest)

    with pytest.raises(CounterStrikeEvidenceError) as caught:
        load_counter_strike_frozen_evidence(
            capture,
            benchmark_config=_config(),
        )

    assert caught.value.code == "evidence_path_unhashed"


def test_loader_rejects_view_order_drift(tmp_path: Path) -> None:
    capture, manifest = _write_capture(tmp_path)
    manifest["views"].reverse()
    _rewrite_manifest(capture, manifest)

    with pytest.raises(CounterStrikeEvidenceError) as caught:
        load_counter_strike_frozen_evidence(
            capture,
            benchmark_config=_config(),
        )

    assert caught.value.code == "evidence_order_mismatch"


def test_loader_rejects_wrong_controlled_view_family(tmp_path: Path) -> None:
    capture, manifest = _write_capture(tmp_path)
    manifest["controlled_camera"]["view_family"] = "unfrozen_camera_bank"
    _rewrite_manifest(capture, manifest)

    with pytest.raises(CounterStrikeEvidenceError) as caught:
        load_counter_strike_frozen_evidence(
            capture,
            benchmark_config=_config(),
        )

    assert caught.value.code == "view_family_mismatch"


def test_loader_rejects_exported_scene_outside_capture_root(
    tmp_path: Path,
) -> None:
    capture, manifest = _write_capture(tmp_path)
    outside = tmp_path / "outside_scene.json"
    outside.write_text(
        json.dumps({"schema_version": "canonical_scene_v1"}),
        encoding="utf-8",
    )
    manifest["exported_scene"] = outside.as_posix()
    _rewrite_manifest(capture, manifest)

    with pytest.raises(CounterStrikeEvidenceError) as caught:
        load_counter_strike_frozen_evidence(
            capture,
            benchmark_config=_config(),
        )

    assert caught.value.code == "exported_scene_outside_root"


def test_loader_requires_prevalidated_benchmark_config(tmp_path: Path) -> None:
    capture, _ = _write_capture(tmp_path)

    with pytest.raises(CounterStrikeEvidenceError) as caught:
        load_counter_strike_frozen_evidence(
            capture,
            benchmark_config={},  # type: ignore[arg-type]
        )

    assert caught.value.code == "benchmark_config_unvalidated"
