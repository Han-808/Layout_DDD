"""Backward-compatibility tests for the deprecated ``visual_quality`` shim.

The L3 layer was reframed to Scene Quality (see
``tests/test_scene_quality_interfaces.py`` for canonical coverage). These tests
only guarantee that the historical ``visual_quality`` import names keep working
and delegate to the canonical Scene Quality implementation.
"""

from __future__ import annotations

from benchmark.evaluator.visual_quality import (
    DEFAULT_VISUAL_QUALITY_INTERFACE_CONFIG,
    VISUAL_QUALITY_INTERFACE_METRICS,
    VISUAL_QUALITY_INTERFACE_NAMESPACE,
    VISUAL_QUALITY_INTERFACE_VERSION,
    VisualQualityInterfaceConfigError,
    evaluate_visual_quality_interfaces,
    resolve_visual_quality_config,
)


def _scene() -> dict:
    return {"scene_id": "shim", "objects": [{"id": "bed"}]}


def test_shim_names_delegate_to_scene_quality() -> None:
    # Namespace and metric set now reflect the canonical Scene Quality layer.
    assert VISUAL_QUALITY_INTERFACE_NAMESPACE == "l3_scene_quality"
    assert VISUAL_QUALITY_INTERFACE_VERSION == "scene_quality_v7"
    assert VISUAL_QUALITY_INTERFACE_METRICS == (
        "style_consistency",
        "scale_consistency",
        "object_pairing_consistency",
        "functional_consistency",
        "semantic_placement_consistency",
    )
    assert DEFAULT_VISUAL_QUALITY_INTERFACE_CONFIG["implemented"] is True


def test_shim_resolve_and_evaluate_work() -> None:
    resolved = resolve_visual_quality_config()
    assert resolved["metrics"]["style_consistency"]["evidence_policy"][
        "camera_mode"
    ] == "global_oblique"

    report = evaluate_visual_quality_interfaces(_scene(), config={"enabled": True})
    assert report["category"] == "l3_scene_quality"
    assert report["interface_version"] == "scene_quality_v7"
    assert report["status"] == "failed"
    assert report["terminal_state"] == "infrastructure_failure"
    assert report["score"] is None
    assert report["affects_score"] is True


def test_shim_config_error_type_is_shared() -> None:
    try:
        resolve_visual_quality_config({"implemented": False})
    except VisualQualityInterfaceConfigError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("expected VisualQualityInterfaceConfigError")
