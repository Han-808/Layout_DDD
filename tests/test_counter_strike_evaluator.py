from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmark.evaluator.profile import L0, L1, L2, L3, L4
from benchmark.game_scene.counter_strike.evaluator import (
    COUNTER_STRIKE_L4_METRICS,
    CounterStrikeEvaluationError,
    evaluate_counter_strike_l4,
    merge_counter_strike_evaluation,
)
from benchmark.game_scene.counter_strike.integration import (
    CounterStrikeIntegrationError,
    evaluate_counter_strike_frozen_capture,
)
from benchmark.game_scene.counter_strike.loader import (
    CanonicalSceneImportTransform,
    CounterStrikeCaseContract,
    VerifiedSourceAssertion,
    load_counter_strike_benchmark_config,
)
from benchmark.game_scene.counter_strike.judge import (
    CounterStrikeVisualJudgeError,
)
from benchmark.game_scene.counter_strike.topology import CounterStrikeTopology
from benchmark.rendering.browser import (
    BROWSER_RENDER_BACKEND,
    CONTROLLED_CAMERA_APPEARANCE_FIDELITY,
)
from benchmark.utils.io import read_json, write_json


ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_CONFIG = (
    ROOT / "configs" / "game" / "counter_strike" / "benchmark_v1.yaml"
)


def _config():
    return load_counter_strike_benchmark_config(BENCHMARK_CONFIG)


def _case_contract(tmp_path: Path) -> CounterStrikeCaseContract:
    source = tmp_path / "map.js"
    source.write_text("const spawn = [0, 0, 0];\n", encoding="utf-8")
    source_hash = _sha256(source)
    return CounterStrikeCaseContract(
        path=tmp_path / "contract.json",
        sha256="contract-sha256",
        source_root=tmp_path,
        raw={"case_id": "cs_unit"},
        source_assertions=(
            VerifiedSourceAssertion(
                declared_path="map.js",
                resolved_path=source,
                sha256=source_hash,
                evidence="unit test",
            ),
        ),
        import_transform=CanonicalSceneImportTransform(
            source_up_axis="y",
            unit_scale=1.0,
            translation_applied=(0.0, 0.0, 0.0),
        ),
        canonical_team_spawns={
            "team_a": {"points": [[1.0, 1.0, 0.0]], "jitter_radius_m": 0.0},
            "team_b": {"points": [[4.0, 4.0, 0.0]], "jitter_radius_m": 0.0},
        },
    )


def _topology() -> CounterStrikeTopology:
    free = np.ones((3, 3), dtype=bool)
    zeros = np.zeros((3, 3), dtype=bool)
    return CounterStrikeTopology(
        version="test_topology_v1",
        grid={
            "largest_component_cells": 9,
            "total_free_cells": 9,
            "boundary_source": "test",
            "inside_room": free,
            "occupied": zeros,
        },
        free=free,
        x_centers=np.asarray([0.0, 1.0, 2.0]),
        y_centers=np.asarray([0.0, 1.0, 2.0]),
        resolution=1.0,
        team_a_cells=((0, 0),),
        team_b_cells=((2, 2),),
        team_a_representative=(0, 0),
        team_b_representative=(2, 2),
        distance_a=np.zeros((3, 3), dtype=float),
        distance_b=np.zeros((3, 3), dtype=float),
        traffic=np.zeros((3, 3), dtype=float),
        main_engagement=zeros.copy(),
        team_a_spawn_zone=zeros.copy(),
        team_b_spawn_zone=zeros.copy(),
        team_a_preparation=zeros.copy(),
        team_b_preparation=zeros.copy(),
        flank_region=zeros.copy(),
        engagement_cell=(1, 1),
        primary_path=((0, 0), (1, 1), (2, 2)),
        routes=(),
        cover_candidates=(),
    )


def _deterministic_metrics() -> dict[str, dict[str, Any]]:
    return {
        "zone_clarity": {
            "metric": "zone_clarity",
            "status": "checked_deterministic_component",
            "score": 0.8,
        },
        "route_structure": {
            "metric": "route_structure",
            "status": "checked",
            "score": 0.7,
            "verdict": "valid",
        },
        "spawn_balance": {
            "metric": "spawn_balance",
            "status": "checked",
            "score": 0.6,
            "verdict": "valid",
        },
        "cover_diversity": {
            "metric": "cover_diversity",
            "status": "checked_deterministic_component",
            "score": 0.5,
            "verdict": "invalid",
        },
    }


def _diagram_renderer(_topology: CounterStrikeTopology, **kwargs: Any) -> Path:
    path = Path(kwargs["out_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"topology")
    return path


class _VisualJudge:
    def __init__(
        self,
        *,
        fail_metric: str | None = None,
        invalid_metric: str | None = None,
        failure_exception: Exception | None = None,
    ) -> None:
        self.fail_metric = fail_metric
        self.invalid_metric = invalid_metric
        self.failure_exception = failure_exception
        self.calls: list[str] = []
        self.kwargs: list[dict[str, Any]] = []

    def judge_metric(self, metric: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(metric)
        self.kwargs.append(kwargs)
        if metric == self.fail_metric:
            raise self.failure_exception or RuntimeError(
                "provider body must not be serialized"
            )
        if metric == self.invalid_metric:
            return {
                "metric": metric,
                "status": "checked",
                "score": 0.5,
                "verdict": "invalid",
            }
        score = 0.6 if metric == "zone_clarity" else 0.9
        return {
            "metric": metric,
            "status": "checked",
            "score": score,
            "verdict": "valid",
        }


def _canonical_report(
    *,
    collision_score: float | None = 0.8,
) -> dict[str, Any]:
    collision = {
        "metric": "collision",
        "status": "checked" if collision_score is not None else "unresolved",
        "score": collision_score,
    }
    if collision_score is None:
        collision["reason"] = "test_unresolved"
    return {
        "workflow": "canonical_l0_l4",
        "profile_version": "canonical_scene_evaluation_v1",
        "scene_id": "cs_unit",
        "request_id": "cs_unit",
        "layer_reports": {
            L0: {
                "layer": L0,
                "status": "passed",
                "score": None,
                "affects_score": False,
            },
            L1: {
                "layer": L1,
                "metrics": {
                    "collision": collision,
                    "navigability": {
                        "metric": "navigability",
                        "status": "checked",
                        "score": 0.6,
                    },
                },
            },
            L2: {
                "layer": L2,
                "status": "not_applicable",
                "score": None,
                "affects_score": False,
            },
            L3: {
                "layer": L3,
                "metrics": {
                    "style_consistency": {
                        "metric": "style_consistency",
                        "status": "evaluated",
                        "score": 0.9,
                    }
                },
            },
            L4: {
                "layer": L4,
                "status": "not_implemented",
                "score": None,
            },
        },
    }


def _complete_l4_report() -> dict[str, Any]:
    metrics = {
        name: {
            "metric": name,
            "status": "checked",
            "score": score,
            "verdict": "valid",
        }
        for name, score in zip(
            COUNTER_STRIKE_L4_METRICS,
            (0.71, 0.7, 0.6, 0.9, 0.5),
        )
    }
    return {
        "report_schema_version": "counter_strike_l4_report_v1",
        "layer": L4,
        "status": "evaluated",
        "score": 0.682,
        "metrics": metrics,
    }


def test_l4_evaluator_merges_exactly_five_metrics_with_frozen_weights(
    tmp_path: Path,
) -> None:
    judge = _VisualJudge()

    report = evaluate_counter_strike_l4(
        {
            "schema_version": "canonical_scene_v1",
            "scene_id": "cs_unit",
            "objects": [],
        },
        case_contract=_case_contract(tmp_path),
        benchmark_config=_config(),
        visual_judge=judge,
        frozen_evidence=object(),
        out_dir=tmp_path / "l4",
        topology_analyzer=lambda *_args, **_kwargs: (
            _topology(),
            _deterministic_metrics(),
        ),
        diagram_renderer=_diagram_renderer,
    )

    assert tuple(report["metrics"]) == COUNTER_STRIKE_L4_METRICS
    assert judge.calls == [
        "zone_clarity",
        "landmark_legibility",
        "cover_diversity",
    ]
    for kwargs in judge.kwargs:
        context = kwargs["topology_context"]
        serialized = json.dumps(context, sort_keys=True)
        assert context["schema_version"] == (
            "counter_strike_neutral_visual_context_v1"
        )
        assert "deterministic_metrics" not in serialized
        assert '"score"' not in serialized
        assert kwargs["topology_diagram"].name == (
            "judge_observation_diagram.png"
        )
    assert report["metrics"]["zone_clarity"]["score"] == pytest.approx(0.71)
    assert report["metrics"]["cover_diversity"]["score"] == pytest.approx(0.72)
    assert report["score"] == pytest.approx(0.726)
    assert report["coverage"]["complete"] is True


def test_explicit_visual_defect_cannot_be_overridden_by_proxy_score(
    tmp_path: Path,
) -> None:
    report = evaluate_counter_strike_l4(
        {"schema_version": "canonical_scene_v1", "objects": []},
        case_contract=_case_contract(tmp_path),
        benchmark_config=_config(),
        visual_judge=_VisualJudge(invalid_metric="zone_clarity"),
        frozen_evidence=object(),
        out_dir=tmp_path / "l4",
        topology_analyzer=lambda *_args, **_kwargs: (
            _topology(),
            _deterministic_metrics(),
        ),
        diagram_renderer=_diagram_renderer,
    )

    zone = report["metrics"]["zone_clarity"]
    assert zone["weighted_score_before_perceptual_veto"] == pytest.approx(
        0.665
    )
    assert zone["score"] == pytest.approx(0.5)
    assert zone["verdict"] == "invalid"
    assert zone["reason"] == "significant_perceptual_defect_veto"


def test_l4_visual_metric_failure_isolated_and_composite_stays_null(
    tmp_path: Path,
) -> None:
    report = evaluate_counter_strike_l4(
        {"schema_version": "canonical_scene_v1", "objects": []},
        case_contract=_case_contract(tmp_path),
        benchmark_config=_config(),
        visual_judge=_VisualJudge(fail_metric="zone_clarity"),
        frozen_evidence=object(),
        out_dir=tmp_path / "l4",
        topology_analyzer=lambda *_args, **_kwargs: (
            _topology(),
            _deterministic_metrics(),
        ),
        diagram_renderer=_diagram_renderer,
    )

    assert report["metrics"]["zone_clarity"]["status"] == "metric_failed"
    assert report["metrics"]["zone_clarity"]["failure"] == {
        "stage": "visual_judge.zone_clarity",
        "error_type": "RuntimeError",
    }
    assert report["metrics"]["landmark_legibility"]["score"] == 0.9
    assert report["metrics"]["route_structure"]["score"] == 0.7
    assert report["score"] is None
    assert report["coverage"]["complete"] is False


def test_l4_visual_metric_failure_preserves_safe_error_code_only(
    tmp_path: Path,
) -> None:
    visual_judge = _VisualJudge(
        fail_metric="cover_diversity",
        failure_exception=CounterStrikeVisualJudgeError(
            "verdict_score_inconsistent",
            "provider response body must not be persisted",
        ),
    )

    report = evaluate_counter_strike_l4(
        {"schema_version": "canonical_scene_v1", "objects": []},
        case_contract=_case_contract(tmp_path),
        benchmark_config=_config(),
        visual_judge=visual_judge,
        frozen_evidence=object(),
        out_dir=tmp_path / "l4",
        topology_analyzer=lambda *_args, **_kwargs: (
            _topology(),
            _deterministic_metrics(),
        ),
        diagram_renderer=_diagram_renderer,
    )

    failure = report["metrics"]["cover_diversity"]["failure"]
    assert failure == {
        "stage": "visual_judge.cover_diversity",
        "error_type": "CounterStrikeVisualJudgeError",
        "error_code": "verdict_score_inconsistent",
    }
    assert "provider response body" not in json.dumps(report)


def test_integrated_composite_uses_canonical_l1_l3_and_complete_l4() -> None:
    report = merge_counter_strike_evaluation(
        _canonical_report(),
        _complete_l4_report(),
        benchmark_config=_config(),
    )

    assert report["layer_reports"][L1]["score"] == pytest.approx(0.7)
    assert report["layer_reports"][L3]["score"] == pytest.approx(0.9)
    assert report["layer_reports"][L4]["score"] == pytest.approx(0.682)
    assert report["benchmark_score"] == pytest.approx(
        0.25 * 0.7 + 0.15 * 0.9 + 0.60 * 0.682
    )
    assert report["coverage"]["complete"] is True
    assert set(report["metric_vector"]) == {
        "collision",
        "navigability",
        "style_consistency",
        *COUNTER_STRIKE_L4_METRICS,
    }


def test_integrated_composite_never_reweights_an_unresolved_metric() -> None:
    report = merge_counter_strike_evaluation(
        _canonical_report(collision_score=None),
        _complete_l4_report(),
        benchmark_config=_config(),
    )

    assert report["layer_reports"][L1]["score"] is None
    assert report["layer_reports"][L3]["score"] == 0.9
    assert report["layer_reports"][L4]["score"] == pytest.approx(0.682)
    assert report["benchmark_score"] is None
    assert report["benchmark_score_status"] == "insufficient_metric_coverage"
    assert report["coverage"]["aggregation"] == (
        "complete_only_no_missing_metric_reweight"
    )


def test_integrated_composite_rejects_incomplete_l4_metric_set() -> None:
    l4 = _complete_l4_report()
    del l4["metrics"]["cover_diversity"]

    with pytest.raises(CounterStrikeEvaluationError) as caught:
        merge_counter_strike_evaluation(
            _canonical_report(),
            l4,
            benchmark_config=_config(),
        )

    assert caught.value.code == "l4_metric_set_invalid"


class _FrozenCollisionRenderer:
    is_frozen_capture_renderer = True

    def __init__(self, capture_dir: Path) -> None:
        self.capture_dir = capture_dir
        self.render_reuse_calls = 0

    def __call__(self, _request: dict[str, Any]) -> dict[str, Any]:
        return {"status": "insufficient", "render_evidence_items": []}

    def render_scene(
        self,
        *,
        scene_path: str | Path,
        out_dir: str | Path,
        asset_root: str | Path | None = None,
    ) -> dict[str, Any]:
        del scene_path, asset_root
        assert Path(out_dir).resolve() == self.capture_dir.resolve()
        self.render_reuse_calls += 1
        return read_json(self.capture_dir / "render_manifest.json")

    def provide_scene_quality_evidence(
        self,
        _request: dict[str, Any],
    ) -> dict[str, Any]:
        return {"status": "available", "render_evidence_items": []}


def test_capture_integration_invokes_canonical_once_without_recapture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "run"
    capture = _write_capture(out_dir / "renders")
    renderer = _FrozenCollisionRenderer(capture)
    canonical_calls = 0

    def canonical_evaluator(**kwargs: Any) -> dict[str, Any]:
        nonlocal canonical_calls
        canonical_calls += 1
        kwargs["renderer"].render_scene(
            scene_path=Path(
                read_json(capture / "render_manifest.json")["exported_scene"]
            ),
            out_dir=out_dir / "renders",
        )
        report = _canonical_report()
        write_json(out_dir / "evaluation_report.json", report)
        return {"evaluation_report": report, "manifest": {}}

    monkeypatch.setattr(
        "benchmark.game_scene.counter_strike.integration."
        "evaluate_counter_strike_l4",
        lambda *_args, **_kwargs: _complete_l4_report(),
    )

    result = evaluate_counter_strike_frozen_capture(
        out_dir=out_dir,
        capture_dir=capture,
        canonical_case_bundle=tmp_path / "case_bundle",
        benchmark_config=_config(),
        case_contract=_case_contract(tmp_path),
        canonical_vlm_judge=object(),
        counter_strike_visual_judge=object(),
        renderer=renderer,
        canonical_evaluator=canonical_evaluator,
    )

    assert canonical_calls == 1
    assert renderer.render_reuse_calls == 1
    assert result["evaluation_report"]["integration"]["capture_performed"] is False
    assert result["evaluation_report"]["integration"][
        "frozen_capture_reused"
    ] is True
    assert result["evaluation_report"]["integration"][
        "canonical_evaluator_invocations"
    ] == 1


def test_capture_integration_rejects_style_only_non_callable_renderer(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "run"
    capture = _write_capture(out_dir / "renders")

    class StyleOnly:
        is_frozen_capture_renderer = True
        capture_dir = capture

        def render_scene(self, **_kwargs: Any) -> dict[str, Any]:
            return {}

        def provide_scene_quality_evidence(
            self,
            _request: dict[str, Any],
        ) -> dict[str, Any]:
            return {}

    with pytest.raises(CounterStrikeIntegrationError) as caught:
        evaluate_counter_strike_frozen_capture(
            out_dir=out_dir,
            capture_dir=capture,
            canonical_case_bundle=tmp_path / "case_bundle",
            benchmark_config=_config(),
            case_contract=_case_contract(tmp_path),
            canonical_vlm_judge=object(),
            counter_strike_visual_judge=object(),
            renderer=StyleOnly(),
        )

    assert caught.value.code == "collision_evidence_provider_missing"


def _write_capture(capture: Path) -> Path:
    capture.mkdir(parents=True)
    exported_scene = capture / "probe_exported_scene.json"
    exported_scene.write_text(
        json.dumps(
            {
                "schema_version": "canonical_scene_v1",
                "scene_id": "cs_unit",
                "request_id": "cs_unit",
                "scene_type": "counter_strike_static_arena",
                "boundary": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "scene_height": 5,
                "objects": [],
            }
        ),
        encoding="utf-8",
    )
    paths = [exported_scene]
    global_views = []
    regional_views = []
    for index in range(2):
        path = capture / f"global_{index}.png"
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
        path = capture / f"regional_{index}.png"
        path.write_bytes(f"regional-{index}".encode())
        paths.append(path)
        regional_views.append(
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
                "views": regional_views,
            },
        },
        "capture_artifacts": {
            path.relative_to(capture).as_posix(): _sha256(path)
            for path in paths
        },
    }
    write_json(capture / "render_manifest.json", manifest)
    return capture


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
