from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from benchmark.visual_judge.p0b import adjudicate_p0b_event
from benchmark.visual_judge.render_views import _cached_items, _highlighted_global_pose
from benchmark.visual_judge.visual_config import compose_default_p0b_visual_evidence


def _item(
    path: Path,
    role: str,
    view_id: str,
    *,
    boundary: bool = False,
) -> dict:
    legend = [{"id": "object", "role": "primary_target"}]
    if boundary:
        legend.append({"id": "room_east_plane", "role": "architecture_plane"})
    return {
        "path": str(path),
        "role": role,
        "view_id": view_id,
        "pose": {"id": view_id},
        "color_legend": legend,
    }


def _bundle(tmp_path: Path, *, boundary: bool = True) -> list[dict]:
    return [
        _item(tmp_path / "perspective.png", "metric_highlighted_global", "global_perspective"),
        _item(tmp_path / "top.png", "metric_highlighted_global", "global_top"),
        _item(tmp_path / "local_1_raw.png", "metric_local_rgb", "local_1"),
        _item(tmp_path / "local_1_contour.png", "metric_local_contour", "local_1"),
        _item(
            tmp_path / "local_1_highlight.png",
            "metric_local_highlight",
            "local_1",
            boundary=boundary,
        ),
        _item(tmp_path / "local_2_raw.png", "metric_local_rgb", "local_2"),
        _item(tmp_path / "local_2_contour.png", "metric_local_contour", "local_2"),
        _item(
            tmp_path / "local_2_highlight.png",
            "metric_local_highlight",
            "local_2",
            boundary=boundary,
        ),
    ]


@pytest.mark.parametrize(
    ("metric", "roles", "view_ids", "budget"),
    [
        (
            "collision",
            ["metric_local_rgb", "metric_local_contour"],
            ["local_1", "local_1"],
            2,
        ),
        (
            "oob",
            ["metric_highlighted_global", "metric_local_highlight"],
            ["global_top", "local_1"],
            2,
        ),
        (
            "support",
            ["metric_local_rgb", "metric_local_rgb", "metric_highlighted_global"],
            ["local_1", "local_2", "global_top"],
            3,
        ),
    ],
)
def test_metric_default_composes_exact_calibrated_budget(
    tmp_path: Path,
    metric: str,
    roles: list[str],
    view_ids: list[str],
    budget: int,
) -> None:
    selected, policy = compose_default_p0b_visual_evidence(metric, _bundle(tmp_path))

    assert [item["role"] for item in selected] == roles
    assert [item["view_id"] for item in selected] == view_ids
    assert policy["image_budget"] == budget
    assert policy["actual_image_count"] == budget
    assert policy["global_perspective_included"] is False
    assert policy["selected_view_ids"] == view_ids


def test_oob_default_requires_boundary_plane_in_local_highlight(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="flagged room boundary plane"):
        compose_default_p0b_visual_evidence("oob", _bundle(tmp_path, boundary=False))


def test_collision_default_requires_same_pose_contour(tmp_path: Path) -> None:
    items = [
        item
        for item in _bundle(tmp_path)
        if not (
            item["role"] == "metric_local_contour"
            and item["view_id"] == "local_1"
        )
    ]
    with pytest.raises(RuntimeError, match="same-pose segmentation contour"):
        compose_default_p0b_visual_evidence("collision", items)


def test_p0b_request_and_result_record_applied_metric_default(tmp_path: Path) -> None:
    captured: list[dict] = []

    def judge(request: dict) -> dict:
        captured.append(request)
        return {"verdict": "invalid", "confidence": 0.8, "reason": "visible gap"}

    report = adjudicate_p0b_event(
        metric="support",
        event={"object_id": "object"},
        prompt="",
        relationships=[],
        scene={"objects": [{"id": "object"}]},
        detector_evidence={"gap_m": 0.1},
        judge=judge,
        local_view_provider=lambda _request: _bundle(tmp_path),
    )

    request = captured[0]
    assert len(request["render_evidence"]) == 3
    assert (
        request["visual_evidence_policy"]["config_id"]
        == "support_local2_raw_global_top_budget3_v2"
    )
    assert report["visual_evidence_policy"] == request["visual_evidence_policy"]


def test_metric_default_does_not_allow_judge_to_silently_truncate(tmp_path: Path) -> None:
    class _Judge:
        max_images = 2

        def adjudicate_p0b(self, _request: dict) -> dict:  # pragma: no cover - must not be called
            raise AssertionError("judge should not be called")

    with pytest.raises(RuntimeError, match="below the support default VisualConfig budget=3"):
        adjudicate_p0b_event(
            metric="support",
            event={"object_id": "object"},
            prompt="",
            relationships=[],
            scene={"objects": [{"id": "object"}]},
            detector_evidence={"gap_m": 0.1},
            judge=_Judge(),
            local_view_provider=lambda _request: _bundle(tmp_path),
        )


def test_passthrough_preserves_frozen_experiment_bundle(tmp_path: Path) -> None:
    captured: list[dict] = []

    def judge(request: dict) -> dict:
        captured.append(request)
        return {"verdict": "valid", "confidence": 0.8, "reason": "experiment"}

    items = _bundle(tmp_path)
    adjudicate_p0b_event(
        metric="collision",
        event={"object_a": "object", "object_b": "other"},
        prompt="",
        relationships=[],
        scene={"objects": [{"id": "object"}, {"id": "other"}]},
        detector_evidence={},
        judge=judge,
        local_view_provider=lambda _request: items,
        visual_config_policy="passthrough",
    )

    assert captured[0]["render_evidence"] == [item["path"] for item in items]
    assert "visual_evidence_policy" not in captured[0]


def test_highlighted_global_pose_defaults_to_top_but_keeps_legacy_option() -> None:
    request = {
        "metric": "collision",
        "scene": {
            "boundary": [[0, 0], [4, 0], [4, 3], [0, 3]],
            "scene_height": 2.8,
            "objects": [],
        },
        "detector_evidence": {},
    }

    assert _highlighted_global_pose(request)["id"] == "global_top"
    assert _highlighted_global_pose(request, policy="legacy_metric")["id"] == "global_perspective"


def test_cached_rich_evidence_is_invalidated_when_global_pose_policy_changes(
    tmp_path: Path,
) -> None:
    image = tmp_path / "global.png"
    image.write_bytes(b"png")
    manifest = tmp_path / "camera_evidence_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "policy": {"highlighted_global_pose_policy": "legacy_metric"},
                "render_evidence_items": [{"path": str(image), "role": "metric_highlighted_global"}],
                "render_evidence_artifacts": [
                    {
                        "slot": 0,
                        "path": str(image),
                        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert _cached_items(
        manifest,
        expected_highlighted_global_pose_policy="legacy_metric",
    ) is not None
    assert _cached_items(
        manifest,
        expected_highlighted_global_pose_policy="global_top",
    ) is None
