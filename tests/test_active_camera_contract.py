from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from benchmark.rendering.camera_pose import apply_camera_action
from benchmark.visual_judge.active_fallback import (
    ConditionalActiveCameraEvidenceProvider,
    build_conditional_active_camera_evidence_provider,
)
from benchmark.visual_judge.active_policy import (
    generate_corrective_camera_proposals,
)
from benchmark.visual_judge.evidence_sufficiency import (
    INSUFFICIENT,
    SUFFICIENT,
    UNKNOWN,
    assess_preview_selection_sufficiency,
    assess_visual_evidence_sufficiency,
)
from benchmark.visual_judge.openai_compatible import (
    OpenAICompatibleVLMJudge,
)
from benchmark.visual_judge.render_views import CameraEvidenceProvider


def _evidence_file(directory: Path, name: str) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    Image.new("RGB", (4, 4), (64, 64, 64)).save(path)
    return path.as_posix()


def _oob_packet(
    directory: Path,
    *,
    target_fraction: float | None = 0.02,
    include_plane_legend: bool = True,
    local_path: str | None = None,
) -> list[dict]:
    visibility = (
        {
            "target_pixel_fractions": {"obj_001": target_fraction},
            "region_pixel_fractions": {"architecture_plane": 0.02},
        }
        if target_fraction is not None
        else None
    )
    return [
        {
            "path": _evidence_file(directory, "global.png"),
            "role": "metric_highlighted_global",
            "view_id": "global_top",
        },
        {
            "path": (
                _evidence_file(directory, "local.png")
                if local_path is None
                else local_path
            ),
            "role": "metric_local_highlight",
            "view_id": "oob_local_00",
            "target_ids": ["obj_001"],
            "color_legend": (
                [
                    {"id": "obj_001", "role": "primary_target"},
                    {"id": "east_wall", "role": "architecture_plane"},
                ]
                if include_plane_legend
                else [{"id": "obj_001", "role": "primary_target"}]
            ),
            "visibility": visibility,
        },
    ]


def _collision_packet(directory: Path) -> list[dict]:
    view_id = "collision_local_00"
    return [
        {
            "path": _evidence_file(directory, "collision_raw.png"),
            "role": "collision_rgb",
            "view_id": view_id,
            "target_ids": ["obj_a", "obj_b"],
            "visibility": {
                "status": "ok",
                "image_pixel_count": 100,
                "targets": {
                    "obj_a": {
                        "visible_pixels": 10,
                        "normalized_visibility": 0.1,
                    },
                    "obj_b": {
                        "visible_pixels": 10,
                        "normalized_visibility": 0.1,
                    },
                },
            },
        },
        {
            "path": _evidence_file(directory, "collision_contour.png"),
            "role": "metric_local_contour",
            "view_id": view_id,
        },
    ]


@pytest.mark.parametrize(
    ("case", "expected_status", "expected_repairability"),
    [
        ("camera", INSUFFICIENT, "camera"),
        ("presentation", INSUFFICIENT, "presentation"),
        ("rerender", INSUFFICIENT, "rerender"),
        ("geometry", UNKNOWN, "geometry"),
        ("nonvisual", UNKNOWN, "nonvisual"),
        ("unknown", UNKNOWN, "unknown"),
    ],
)
def test_sufficiency_repairability_taxonomy_controls_active_trigger(
    tmp_path: Path,
    case: str,
    expected_status: str,
    expected_repairability: str,
) -> None:
    request: dict = {"metric": "oob", "object_ids": ["obj_001"]}
    if case == "camera":
        metric = "oob"
        packet = _oob_packet(tmp_path / case, target_fraction=0.0)
    elif case == "presentation":
        metric = "oob"
        packet = _oob_packet(
            tmp_path / case,
            include_plane_legend=False,
        )
    elif case == "rerender":
        metric = "oob"
        packet = _oob_packet(
            tmp_path / case,
            local_path=(tmp_path / case / "missing.png").as_posix(),
        )
    elif case == "geometry":
        metric = "collision"
        packet = _collision_packet(tmp_path / case)
        request = {
            "metric": metric,
            "object_ids": ["obj_a", "obj_b"],
            "detector_evidence": {
                "mesh": {"containment_a_in_b": True},
            },
        }
    elif case == "nonvisual":
        metric = "scale"
        packet = _oob_packet(tmp_path / case)
    else:
        metric = "oob"
        packet = _oob_packet(tmp_path / case, target_fraction=None)

    assessment = assess_visual_evidence_sufficiency(
        metric,
        packet,
        request=request,
    )

    assert assessment["status"] == expected_status
    assert assessment["repairability"] == expected_repairability
    assert assessment["camera_repairable"] is (case == "camera")
    assert assessment["trigger_recommended"] is (case == "camera")
    assert {
        item["repairability"] for item in assessment["deficiencies"]
    } == {expected_repairability}


@pytest.mark.parametrize(
    (
        "focus_fields",
        "expected_status",
        "expected_repairability",
        "expected_reason",
    ),
    [
        (
            {},
            UNKNOWN,
            "unknown",
            "collision_focus_roi_unmeasured",
        ),
        (
            {
                "focus_measurement_status": "measured",
                "focus_in_frame": False,
                "focus_pixel_fraction": 0.0,
            },
            INSUFFICIENT,
            "camera",
            "focus_region_out_of_frame",
        ),
        (
            {
                "focus_measurement_status": "measured",
                "focus_in_frame": True,
                "focus_pixel_fraction": 0.000001,
            },
            INSUFFICIENT,
            "camera",
            "focus_region_too_small",
        ),
        (
            {
                "focus_measurement_status": "measured",
                "focus_in_frame": True,
                "focus_pixel_fraction": 0.001,
            },
            SUFFICIENT,
            "none",
            "calibrated_packet_and_visibility_sufficient",
        ),
    ],
)
def test_final_collision_focus_measurement_controls_camera_repair(
    tmp_path: Path,
    focus_fields: dict,
    expected_status: str,
    expected_repairability: str,
    expected_reason: str,
) -> None:
    packet = _collision_packet(tmp_path)
    packet[0]["visibility"].update(focus_fields)

    assessment = assess_visual_evidence_sufficiency(
        "collision",
        packet,
        request={
            "metric": "collision",
            "object_ids": ["obj_a", "obj_b"],
        },
    )

    assert assessment["status"] == expected_status
    assert assessment["repairability"] == expected_repairability
    assert assessment["reason_codes"] == [expected_reason]
    assert assessment["trigger_recommended"] is (
        expected_repairability == "camera"
    )


class _Provider:
    def __init__(self, items: list[dict]) -> None:
        self.items = items
        self.calls: list[dict] = []
        self.policy_config = {"provider": type(self).__name__}

    def __call__(self, request: dict) -> list[dict]:
        self.calls.append(request)
        return self.items


@pytest.mark.parametrize("deficiency", ["presentation", "rerender"])
def test_non_camera_insufficiency_does_not_invoke_active_provider(
    tmp_path: Path,
    deficiency: str,
) -> None:
    if deficiency == "presentation":
        base_items = _oob_packet(
            tmp_path / "base",
            include_plane_legend=False,
        )
    else:
        base_items = _oob_packet(
            tmp_path / "base",
            local_path=(tmp_path / "base" / "missing.png").as_posix(),
        )
    deterministic = _Provider(base_items)
    active = _Provider(_oob_packet(tmp_path / "active"))
    provider = ConditionalActiveCameraEvidenceProvider(
        deterministic_provider=deterministic,
        active_provider=active,
        out_dir=tmp_path / "fallback",
        max_views=2,
        max_steps=1,
        fail_on_exhausted=False,
    )

    returned = provider({"metric": "oob", "object_ids": ["obj_001"]})

    assert active.calls == []
    assert returned == base_items


def _candidate(view_id: str = "view_00") -> dict:
    return {
        "id": view_id,
        "name": view_id,
        "camera_type": "PERSP",
        "location": [2.0, 2.0, 1.5],
        "target": [0.0, 0.0, 0.5],
        "lens_mm": 52.0,
        "candidate_policy": "feasible_v2",
        "room_bounds": [-4.0, 4.0, -4.0, 4.0, 0.0, 3.0],
    }


@pytest.mark.parametrize(
    ("metric", "expected_roles"),
    [
        ("collision", {"collision_focus"}),
        ("oob", {"object_plane_boundary", "violated_architecture_plane"}),
        ("support", {"support_gap", "support_topology"}),
    ],
)
def test_corrective_proposals_are_metric_specific_and_preflighted(
    metric: str,
    expected_roles: set[str],
) -> None:
    candidate = _candidate()
    proposals = generate_corrective_camera_proposals(
        metric=metric,
        candidates=[candidate],
        deficiency={
            "deficiencies": [
                {
                    "code": "focus_region_too_small",
                    "repairability": "camera",
                }
            ]
        },
        max_proposals=20,
    )

    assert proposals
    assert {item["metric"] for item in proposals} == {metric}
    assert {item["look_at_role"] for item in proposals} == expected_roles
    assert all(item["room_feasible"] is True for item in proposals)
    assert all(item["result_pose_fingerprint"] for item in proposals)
    assert all(item["proposal_fingerprint"] for item in proposals)
    assert all(item["result_pose"]["parent_view_id"] == candidate["id"] for item in proposals)
    assert all(
        -4.0 <= item["result_pose"]["location"][0] <= 4.0
        and -4.0 <= item["result_pose"]["location"][1] <= 4.0
        and 0.0 <= item["result_pose"]["location"][2] <= 3.0
        for item in proposals
    )


def test_corrective_proposals_filter_non_camera_and_repeated_actions() -> None:
    candidate = _candidate()
    deficiency = {
        "deficiencies": [
            {
                "code": "target_occluded_or_too_small",
                "repairability": "camera",
            }
        ]
    }
    initial = generate_corrective_camera_proposals(
        metric="oob",
        candidates=[candidate],
        deficiency=deficiency,
        max_proposals=20,
    )
    assert initial
    repeated = initial[0]
    filtered = generate_corrective_camera_proposals(
        metric="oob",
        candidates=[candidate],
        deficiency=deficiency,
        history=[
            {
                "parent_view_id": repeated["parent_view_id"],
                "action_primitive": repeated["action_primitive"],
            }
        ],
        max_proposals=20,
    )

    assert (
        repeated["parent_view_id"],
        repeated["action_primitive"],
    ) not in {
        (item["parent_view_id"], item["action_primitive"])
        for item in filtered
    }
    assert generate_corrective_camera_proposals(
        metric="oob",
        candidates=[candidate],
        deficiency={
            "deficiencies": [
                {
                    "code": "oob_architecture_plane_highlight_missing",
                    "repairability": "presentation",
                }
            ]
        },
    ) == []


def test_corrective_proposal_budget_is_round_robin_across_candidates() -> None:
    candidates = [_candidate(f"view_{index:02d}") for index in range(5)]

    proposals = generate_corrective_camera_proposals(
        metric="support",
        candidates=candidates,
        deficiency={
            "deficiencies": [
                {
                    "code": "target_occluded_or_too_small",
                    "repairability": "camera",
                }
            ]
        },
        max_proposals=5,
    )

    assert [item["parent_view_id"] for item in proposals] == [
        item["id"] for item in candidates
    ]


def test_collision_context_pose_is_not_relabelled_as_contact_repair() -> None:
    context = {
        **_candidate("collision_context"),
        "focus_kind": "pair_context",
    }

    assert generate_corrective_camera_proposals(
        metric="collision",
        candidates=[context],
        deficiency={
            "deficiencies": [
                {
                    "code": "focus_region_too_small",
                    "repairability": "camera",
                }
            ]
        },
    ) == []


def test_camera_action_rechecks_object_clearance_and_proxy_framing() -> None:
    candidate = {
        **_candidate("clearance"),
        "location": [2.0, 0.0, 0.5],
        "target": [0.0, 0.0, 0.5],
    }
    scene = {
        "objects": [
            {
                "id": "blocking_proxy",
                "center": [1.88, 0.68, 0.5],
                "size": [0.5, 0.5, 0.5],
                "rotation": [0.0, 0.0, 0.0],
            }
        ]
    }
    with pytest.raises(ValueError, match="inside an object clearance proxy"):
        apply_camera_action(candidate, "orbit_left", scene=scene)

    clipping = {
        **_candidate("clipping"),
        "location": [1.0, 0.0, 0.5],
        "target": [0.0, 0.0, 0.5],
        "proxy_framing_bounds": [
            [-1.0, -1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        "proxy_framing": {"aspect_ratio": 1.0},
    }
    with pytest.raises(ValueError, match="clips the proxy framing bounds"):
        apply_camera_action(clipping, "dolly_in")


def test_preview_assessment_exposes_deterministic_evidence_gain() -> None:
    request = {"metric": "oob", "object_ids": ["obj_001"]}
    weak = assess_preview_selection_sufficiency(
        "oob",
        ["weak"],
        {
            "weak": {
                "target_pixel_fractions": {"obj_001": 0.0},
                "region_pixel_fractions": {"architecture_plane": 0.02},
            }
        },
        request=request,
    )
    strong = assess_preview_selection_sufficiency(
        "oob",
        ["strong"],
        {
            "strong": {
                "target_pixel_fractions": {"obj_001": 0.02},
                "region_pixel_fractions": {"architecture_plane": 0.02},
            }
        },
        request=request,
    )

    assert weak["status"] == INSUFFICIENT
    assert strong["status"] == "sufficient"
    assert strong["evidence_utility"] > weak["evidence_utility"]


def test_oob_preview_requires_visible_object_and_architecture_plane() -> None:
    request = {"metric": "oob", "object_ids": ["obj_001"]}
    assessment = assess_preview_selection_sufficiency(
        "oob",
        ["view_00"],
        {
            "view_00": {
                "target_pixel_fractions": {"obj_001": 0.02},
                "region_pixel_fractions": {"architecture_plane": 0.0},
            }
        },
        request=request,
    )

    assert assessment["status"] == INSUFFICIENT
    assert assessment["reason_codes"] == [
        "architecture_plane_not_visible"
    ]
    assert assessment["camera_repairable"] is True


def test_support_duplicate_views_are_camera_repairable() -> None:
    request = {"metric": "support", "object_ids": ["obj_001"]}
    visibility = {
        "target_pixel_fractions": {"obj_001": 0.02},
        "focus_pixel_fraction": 0.001,
    }
    pose = {
        "location": [2.0, 2.0, 1.0],
        "target": [0.0, 0.0, 0.2],
    }
    assessment = assess_preview_selection_sufficiency(
        "support",
        ["view_00", "view_01"],
        {"view_00": visibility, "view_01": visibility},
        request=request,
        poses_by_id={"view_00": pose, "view_01": pose},
    )

    assert assessment["status"] == INSUFFICIENT
    assert assessment["reason_codes"] == ["redundant_local_views"]
    assert assessment["camera_repairable"] is True


def test_shadow_mode_preserves_official_deterministic_packet(tmp_path: Path) -> None:
    base_items = _oob_packet(tmp_path / "base", target_fraction=0.0)
    active_items = _oob_packet(tmp_path / "active", target_fraction=0.02)
    deterministic = _Provider(base_items)
    active = _Provider(active_items)
    provider = ConditionalActiveCameraEvidenceProvider(
        deterministic_provider=deterministic,
        active_provider=active,
        out_dir=tmp_path / "fallback",
        max_views=2,
        max_steps=1,
        fail_on_exhausted=False,
        shadow_mode=True,
    )

    returned = provider({"metric": "oob", "object_ids": ["obj_001"]})

    assert len(active.calls) == 1
    assert returned == base_items
    assert all("active_camera_fallback" not in item for item in returned)


class _NoopRenderer:
    preview_render_engine = "BLENDER_WORKBENCH"
    preview_width = 64
    preview_height = 64
    preview_cycles_samples = 1


class _NoopSelector:
    max_images = 5

    def select_camera_views(self, request: dict) -> dict:
        return {
            "selected_view_ids": [
                str(request["candidates"][0]["id"]),
            ],
            "action": None,
        }


def test_conditional_builder_enables_active_repair_on_active_provider_only(
    tmp_path: Path,
) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")
    provider = build_conditional_active_camera_evidence_provider(
        renderer=_NoopRenderer(),
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        deterministic_mode="auto",
        selector=_NoopSelector(),
        max_views=2,
        max_steps=1,
    )

    assert provider.deterministic_provider.active_repair is False
    assert provider.active_provider.active_repair is True
    assert provider.deterministic_provider.candidate_count == 5
    assert provider.active_provider.candidate_count == 5


class _ActiveSelector:
    def __init__(self, *, request_action: bool) -> None:
        self.request_action = request_action
        self.calls: list[dict] = []

    def select_camera_views(self, request: dict) -> dict:
        self.calls.append(request)
        candidate_id = str(request["candidates"][0]["id"])
        action = None
        if self.request_action and request.get("corrective_proposals"):
            proposal = request["corrective_proposals"][0]
            action = {
                "proposal_id": proposal["proposal_id"],
                "view_id": proposal["parent_view_id"],
                "type": proposal["action_primitive"],
            }
        return {
            "selected_view_ids": [candidate_id],
            "action": action,
            "reason": "test selector",
        }


def _active_provider(
    tmp_path: Path,
    *,
    selector: _ActiveSelector,
    max_steps: int,
) -> CameraEvidenceProvider:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")
    return CameraEvidenceProvider(
        renderer=_NoopRenderer(),
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        mode="query_cov",
        selector=selector,
        max_views=1,
        max_steps=max_steps,
        candidate_count=1,
        active_repair=True,
    )


def _install_fake_overlay_renderer(
    monkeypatch: pytest.MonkeyPatch,
    provider: CameraEvidenceProvider,
) -> None:
    def render_overlay_views(
        *,
        request: dict,
        out_dir: Path,
        camera_views: list[dict],
        overlay_spec: dict,
        preview: bool,
    ) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        views = []
        for pose in camera_views:
            path = out_dir / f"{pose['id']}.png"
            path.write_bytes(b"preview")
            views.append({"id": pose["id"], "path": path.as_posix()})
        return {"views": views}

    monkeypatch.setattr(provider, "_render_overlay_views", render_overlay_views)


def _active_request() -> dict:
    return {
        "metric": "oob",
        "object_ids": ["obj_001"],
        "_camera_selection_phase": "active_fallback",
        "_camera_evidence_deficiency": {
            "status": "insufficient",
            "camera_repairable": True,
            "reason_codes": ["target_occluded_or_too_small"],
            "deficiencies": [
                {
                    "code": "target_occluded_or_too_small",
                    "repairability": "camera",
                }
            ],
        },
    }


def _overlay_spec() -> dict:
    return {
        "targets": [{"id": "obj_001", "color": [1.0, 0.0, 0.0]}],
        "colors": {
            "marker": [0.0, 1.0, 0.0],
            "architecture": [1.0, 0.0, 1.0],
        },
        "architecture_planes": [{"id": "east_wall"}],
        "legend": [
            {"id": "obj_001", "role": "primary_target"},
            {"id": "east_wall", "role": "architecture_plane"},
        ],
    }


def test_active_loop_stops_before_action_when_current_packet_is_sufficient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector = _ActiveSelector(request_action=True)
    provider = _active_provider(
        tmp_path,
        selector=selector,
        max_steps=2,
    )
    _install_fake_overlay_renderer(monkeypatch, provider)
    monkeypatch.setattr(
        "benchmark.visual_judge.render_views.measure_focus_visibility",
        lambda path, **kwargs: {
            "target_pixel_fractions": {"obj_001": 0.02},
            "region_pixel_fractions": {"architecture_plane": 0.02},
        },
    )

    selected, log = provider._active_repair_selection(
        _active_request(),
        [_candidate()],
        tmp_path / "event",
        overlay_spec=_overlay_spec(),
    )

    assert [item["id"] for item in selected] == ["view_00"]
    assert len(selector.calls) == 1
    assert log["stop_reason"] == "sufficient_evidence"
    assert log["camera_action_count"] == 0
    assert log["steps"][0]["sufficiency"]["status"] == "sufficient"


def test_active_loop_stops_on_no_measured_gain_and_retains_best_packet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector = _ActiveSelector(request_action=True)
    provider = _active_provider(
        tmp_path,
        selector=selector,
        max_steps=2,
    )
    _install_fake_overlay_renderer(monkeypatch, provider)
    monkeypatch.setattr(
        "benchmark.visual_judge.render_views.measure_focus_visibility",
        lambda path, **kwargs: {
            "target_pixel_fractions": {"obj_001": 0.0},
            "region_pixel_fractions": {"architecture_plane": 0.02},
        },
    )

    selected, log = provider._active_repair_selection(
        _active_request(),
        [_candidate()],
        tmp_path / "event",
        overlay_spec=_overlay_spec(),
    )

    assert len(selector.calls) == 2
    assert log["camera_action_count"] == 1
    assert log["stop_reason"] == "no_measured_evidence_gain"
    assert len(log["steps"]) == 2
    assert log["steps"][0]["action_execution"]["executed"] is True
    assert log["steps"][1]["gain_from_previous"] == {
        "comparable": True,
        "status_improved": False,
        "utility_delta": 0.0,
        "usable_view_delta": 0,
    }
    assert [item["id"] for item in selected] == ["view_00"]


def test_active_selector_uses_request_local_proposal_aliases_only(
    tmp_path: Path,
) -> None:
    image = tmp_path / "candidate.png"
    Image.new("RGB", (4, 4), (128, 128, 128)).save(image)
    model = _JudgeModel()
    judge = OpenAICompatibleVLMJudge(model)
    model.chat_messages = lambda messages, **kwargs: (
        model.calls.append({"messages": messages, "kwargs": kwargs})
        or json.dumps(
            {
                "selected_view_ids": ["candidate_00"],
                "action": {"proposal_id": "proposal_00"},
                "reason": "bounded repair",
            }
        )
    )
    proposal = {
        "proposal_id": "private-internal-proposal-id",
        "parent_view_id": "private-internal-candidate-id",
        "family": "plane_tangent_left",
        "action_primitive": "orbit_left",
        "target_evidence": "object-plane boundary",
        "repairs_deficiency_codes": [
            "target_occluded_or_too_small",
            "private-deficiency-sentinel",
        ],
        "room_feasible": True,
        "result_pose": {
            "location": [1.0, 2.0, 3.0],
            "private_pose_sentinel": True,
        },
        "proposal_fingerprint": "private-proposal-fingerprint",
    }

    result = judge.select_camera_views(
        {
            "selection_phase": "active_fallback",
            "metric": "oob",
            "evidence_deficiency": {
                "status": "insufficient",
                "reason_codes": [
                    "target_occluded_or_too_small",
                    "private-deficiency-sentinel",
                ],
                "required_local_view_count": 1,
                "measured_local_view_count": 1,
                "usable_local_view_count": 0,
            },
            "candidates": [
                {
                    "id": "private-internal-candidate-id",
                    "pose": {
                        "azimuth_degrees": 20.0,
                        "private_pose_sentinel": True,
                    },
                    "image_path": image.as_posix(),
                }
            ],
            "corrective_proposals": [proposal],
            "max_views": 1,
            "allow_adjustment": True,
            "allowed_actions": ["orbit_left"],
            "preview_role": "highlighted_focus",
        }
    )

    assert result["selected_view_ids"] == ["private-internal-candidate-id"]
    assert result["action"] == {
        "proposal_id": "private-internal-proposal-id",
        "view_id": "private-internal-candidate-id",
        "type": "orbit_left",
        "family": "plane_tangent_left",
    }
    context = json.loads(
        model.calls[0]["messages"][1]["content"][0]["text"].split("\n", 1)[1]
    )
    assert context["corrective_proposals"] == [
        {
            "id": "proposal_00",
            "source_candidate_id": "candidate_00",
            "family": "plane_tangent_left",
            "action": "orbit_left",
            "target_evidence": "object-plane boundary",
            "repairs": ["target_occluded_or_too_small"],
            "room_feasible": True,
        }
    ]
    assert context["evidence_deficiency"]["reason_codes"] == [
        "target_occluded_or_too_small"
    ]
    serialized = json.dumps(context)
    for private_value in (
        "private-internal-proposal-id",
        "private-internal-candidate-id",
        "private-deficiency-sentinel",
        "private_pose_sentinel",
        "private-proposal-fingerprint",
    ):
        assert private_value not in serialized


class _JudgeModel:
    model_id = "fake-judge"
    endpoint = "http://127.0.0.1:1/v1"
    last_request_metadata: dict = {}

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def chat_messages(self, messages, **kwargs) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return json.dumps(
            {
                "verdict": "valid",
                "confidence": 0.9,
                "reason": "packet inspected",
            }
        )


def test_final_judge_payload_strips_active_policy_lineage(tmp_path: Path) -> None:
    image = Path(_evidence_file(tmp_path, "judge.png"))
    model = _JudgeModel()
    judge = OpenAICompatibleVLMJudge(model)
    judge.adjudicate_p0b(
        {
            "metric": "collision",
            "event": {"object_a": "obj_a", "object_b": "obj_b"},
            "render_evidence": [image.as_posix()],
            "local_render_evidence_metadata": [
                {
                    "role": "collision_rgb",
                    "view_id": "private_view__orbit_left",
                    "active_camera_fallback": {
                        "used": True,
                        "private": "fallback-lineage-sentinel",
                    },
                    "pose": {
                        "camera_action": "orbit_left",
                        "camera_action_protocol": "private-protocol-sentinel",
                        "camera_action_parameters": {"private": 1},
                        "parent_view_id": "private-parent-sentinel",
                        "policy_source": "private-policy-sentinel",
                    },
                    "proposal_id": "private-proposal-sentinel",
                    "trajectory": ["private-trajectory-sentinel"],
                    "selection_phase": "active_fallback",
                    "diagnostic_degradation_reason": (
                        "RuntimeError: /private/render/path "
                        "private-degradation-sentinel"
                    ),
                    "safe_diagnostic": "retained-safe-sentinel",
                }
            ],
        }
    )

    context = json.loads(
        model.calls[0]["messages"][1]["content"][0]["text"].split("\n", 1)[1]
    )
    serialized = json.dumps(context)
    assert context["view_evidence"][0]["role"] == "collision_rgb"
    for forbidden in (
        "active_camera_fallback",
        "camera_action",
        "camera_action_protocol",
        "camera_action_parameters",
        "parent_view_id",
        "policy_source",
        "proposal_id",
        "trajectory",
        "selection_phase",
        "private_view__orbit_left",
        "fallback-lineage-sentinel",
        "private-protocol-sentinel",
        "private-parent-sentinel",
        "private-policy-sentinel",
        "private-proposal-sentinel",
        "private-trajectory-sentinel",
        "private-degradation-sentinel",
        "/private/render/path",
    ):
        assert forbidden not in serialized
