from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from benchmark.rendering.camera_pose import (
    CAMERA_ACTIONS,
    CAMERA_CANDIDATE_POLICIES,
    CAMERA_CANDIDATE_POLICY_ALIASES,
    DEFAULT_CAMERA_CANDIDATE_POLICY,
    DEFAULT_CAMERA_MODE_BY_METRIC,
    apply_camera_action,
    generate_camera_pose_candidates,
    parse_metric_camera_modes,
    resolve_camera_pose_mode,
    select_bbox_track_views,
)
from benchmark.visual_judge import CameraEvidenceProvider
from benchmark.rendering.collision_overlay import (
    build_focus_overlay_spec,
    measure_focus_visibility,
    rank_focus_candidates,
    rank_support_contact_candidates,
)


def test_camera_candidate_policy_names_are_local_and_legacy() -> None:
    assert CAMERA_CANDIDATE_POLICIES == ("local", "legacy")
    assert CAMERA_CANDIDATE_POLICY_ALIASES == {
        "feasible_v2": "local",
        "legacy_v1": "legacy",
    }
    assert DEFAULT_CAMERA_CANDIDATE_POLICY == "local"
    assert generate_camera_pose_candidates(
        _request(), policy="feasible_v2"
    ) == generate_camera_pose_candidates(_request(), policy="local")
    assert generate_camera_pose_candidates(
        _request(), policy="legacy_v1"
    ) == generate_camera_pose_candidates(_request(), policy="legacy")


def test_camera_action_accepts_persisted_alias_and_rejects_unknown_policy() -> None:
    pose = generate_camera_pose_candidates(
        _request(),
        max_candidates=1,
        policy="local",
    )[0]
    pose["candidate_policy"] = "feasible_v2"
    adjusted = apply_camera_action(pose, "orbit_left")
    assert adjusted["camera_action"] == "orbit_left"

    pose["candidate_policy"] = "invented"
    with pytest.raises(ValueError, match="camera candidate policy"):
        apply_camera_action(pose, "orbit_left")
    pose["candidate_policy"] = "  "
    with pytest.raises(ValueError, match="camera candidate policy"):
        apply_camera_action(pose, "orbit_left")


def _request(metric: str = "collision") -> dict:
    return {
        "metric": metric,
        "event": {"object_a": "bed", "object_b": "cabinet"},
        "scene": {
            "scene_id": "scene",
            "boundary": [[0, 0], [7, 0], [7, 5], [0, 5]],
            "scene_height": 3.0,
            "objects": [
                {
                    "id": "bed",
                    "category": "bed",
                    "center": [3.0, 2.5, 0.5],
                    "size": [2.0, 1.5, 1.0],
                    "rotation": [0.0, 0.0, 20.0],
                },
                {
                    "id": "cabinet",
                    "category": "cabinet",
                    "center": [4.0, 2.5, 0.55],
                    "size": [0.8, 0.8, 1.1],
                    "rotation": [0.0, 0.0, 0.0],
                },
            ],
        },
        "object_ids": ["bed", "cabinet"],
        "detector_evidence": {"normalized_overlap": 0.2},
        "natural_language_prompt": "Put the cabinet beside the bed.",
        "extracted_relationships": [{"subject": "cabinet", "predicate": "beside", "object": "bed"}],
    }


def _actual_pose_angles(pose: dict) -> tuple[float, float, float]:
    location = np.asarray(pose["location"], dtype=float)
    target = np.asarray(pose["target"], dtype=float)
    delta = location - target
    distance = float(np.linalg.norm(delta))
    azimuth = math.degrees(math.atan2(float(delta[1]), float(delta[0]))) % 360.0
    elevation = math.degrees(math.atan2(float(delta[2]), float(np.linalg.norm(delta[:2]))))
    return azimuth, elevation, distance


class _FakeRenderer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def render_camera_views(self, *, blend_file, out_dir, camera_views, preview=False):
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self.calls.append({"preview": preview, "views": camera_views, "blend_file": Path(blend_file)})
        views = []
        for index, pose in enumerate(camera_views):
            path = destination / f"view_{index:02d}.png"
            path.write_bytes(b"png")
            views.append({"id": pose["id"], "path": str(path), "pose": pose})
        return {"views": views}

    def render_focus_overlay_views(
        self,
        *,
        blend_file,
        out_dir,
        camera_views,
        overlay_spec,
        preview=False,
    ):
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self.calls.append(
            {
                "preview": preview,
                "views": camera_views,
                "blend_file": Path(blend_file),
                "pass": "identity",
            }
        )
        targets = [
            item
            for item in overlay_spec.get("targets", [])
            if isinstance(item, dict)
        ]
        views = []
        for index, pose in enumerate(camera_views):
            path = destination / f"identity_{index:02d}.png"
            image = Image.new("RGB", (64, 48), (20, 20, 20))
            stripe_width = max(1, 48 // max(1, len(targets)))
            for target_index, target in enumerate(targets):
                color = tuple(
                    round(float(value) * 255)
                    for value in target["color"]
                )
                x0 = 8 + target_index * stripe_width
                x1 = min(56, x0 + stripe_width)
                for x in range(x0, x1):
                    for y in range(8, 40):
                        image.putpixel((x, y), color)
            image.save(path)
            views.append(
                {
                    "id": pose["id"],
                    "path": str(path),
                    "pose": pose,
                }
            )
        return {"views": views}


class _FakeSelector:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def select_camera_views(self, request: dict) -> dict:
        self.calls.append(request)
        first = request["candidates"][0]["id"]
        if request["allow_adjustment"]:
            return {
                "selected_view_ids": [first],
                "action": {"view_id": first, "type": "orbit_left"},
                "reason": "separate the silhouettes",
            }
        return {"selected_view_ids": [first], "action": None, "reason": "best adjusted view"}


class _FocusRenderer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def render_camera_views(self, *, blend_file, out_dir, camera_views, preview=False):
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self.calls.append({"pass": "rgb", "preview": preview})
        views = []
        for index, pose in enumerate(camera_views):
            path = destination / f"rgb_{index:02d}.png"
            Image.new("RGB", (32, 32), (120, 140, 160)).save(path)
            views.append({"id": pose["id"], "path": str(path), "pose": pose})
        return {"views": views}

    def render_focus_overlay_views(self, *, blend_file, out_dir, camera_views, overlay_spec, preview=False):
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self.calls.append({"pass": "focus", "preview": preview})
        color = tuple(round(value * 255) for value in overlay_spec["targets"][0]["color"])
        views = []
        for index, pose in enumerate(camera_views):
            path = destination / f"focus_{index:02d}.png"
            image = Image.new("RGB", (32, 32), (80, 80, 80))
            for x in range(8, 24):
                for y in range(8, 24):
                    image.putpixel((x, y), color)
            image.save(path)
            views.append({"id": pose["id"], "path": str(path), "pose": pose})
        return {"views": views}


class _BackfillFocusRenderer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def render_camera_views(self, *, blend_file, out_dir, camera_views, preview=False):
        ids = [str(item["id"]) for item in camera_views]
        self.calls.append(("rgb", ids))
        if "bad" in ids:
            raise RuntimeError("blank camera evidence")
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        views = []
        for candidate_id in ids:
            path = destination / f"{candidate_id}.png"
            Image.new("RGB", (32, 32), (120, 140, 160)).save(path)
            views.append({"id": candidate_id, "path": str(path)})
        return {"views": views}

    def render_focus_evidence_bundle(
        self,
        *,
        blend_file,
        out_dir,
        local_camera_views,
        global_camera_views,
        overlay_spec,
    ):
        ids = [str(item["id"]) for item in local_camera_views]
        self.calls.append(("bundle", ids))
        if "bad" in ids:
            raise RuntimeError("blank bundled evidence")
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        rgb_views = []
        overlay_views = []
        global_views = []
        for candidate_id in ids:
            rgb = destination / f"{candidate_id}_rgb.png"
            overlay = destination / f"{candidate_id}_overlay.png"
            Image.new("RGB", (32, 32), (120, 140, 160)).save(rgb)
            Image.new("RGB", (32, 32), (180, 40, 40)).save(overlay)
            rgb_views.append({"id": candidate_id, "path": str(rgb)})
            overlay_views.append({"id": candidate_id, "path": str(overlay)})
        for pose in global_camera_views:
            path = destination / f"{pose['id']}_global.png"
            Image.new("RGB", (32, 32), (180, 40, 40)).save(path)
            global_views.append({"id": pose["id"], "path": str(path)})
        return {
            "rgb_views": rgb_views,
            "overlay_views": overlay_views,
            "global_overlay_views": global_views,
        }

def test_metric_aware_bbox_candidates_and_bounded_actions() -> None:
    candidates = generate_camera_pose_candidates(_request(), max_candidates=6)

    assert len(candidates) == 6
    assert candidates[0]["policy_source"] == "metric_aware_feasible_candidate_bank_v2"
    assert candidates[0]["candidate_policy"] == "local"
    assert candidates[0]["sensor_width_mm"] == 36.0
    assert "proxy_bounds_fit" in candidates[0]["proxy_framing"]
    assert candidates[0]["target_object_ids"] == ["bed", "cabinet"]
    assert candidates[0]["name"].startswith("collision_")
    assert len(select_bbox_track_views(candidates, max_views=2)) == 2

    adjusted = apply_camera_action(candidates[0], "orbit_left")
    assert adjusted["parent_view_id"] == candidates[0]["id"]
    assert adjusted["camera_action"] == "orbit_left"
    assert adjusted["location"] != candidates[0]["location"]
    actual_azimuth, actual_elevation, actual_distance = _actual_pose_angles(adjusted)
    assert adjusted["azimuth_degrees"] == pytest.approx(actual_azimuth)
    assert adjusted["elevation_degrees"] == pytest.approx(actual_elevation)
    assert adjusted["distance_m"] == pytest.approx(actual_distance)
    assert set(CAMERA_ACTIONS) >= {"orbit_left", "dolly_in"}


def test_bbox_track_provider_renders_frozen_views_without_selector(tmp_path: Path) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")
    renderer = _FakeRenderer()
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        mode="bbox_track",
        max_views=2,
    )

    paths = provider(_request())

    assert len(paths) == 2
    assert len(renderer.calls) == 1
    assert renderer.calls[0]["preview"] is False
    assert provider.policy_config["mode"] == "bbox_track"
    assert provider.policy_config["max_steps"] == 0
    assert provider.last_call_usage is not None
    assert provider.last_call_usage["metric"] == "collision"
    assert provider.last_call_usage["cache_hit"] is False
    assert provider.last_call_usage["evidence_refs"] == [
        str(path) for path in paths
    ]
    assert provider.last_call_usage["selector_calls"] == 0
    assert provider.last_call_usage["camera_actions"] == 0


def test_four_concrete_modes_and_auto_metric_resolution() -> None:
    assert resolve_camera_pose_mode("auto", "collision") == "visibility_ranked"
    assert resolve_camera_pose_mode("auto", "oob") == "visibility_ranked"
    assert resolve_camera_pose_mode("auto", "support") == "support_contact_plane"
    assert (
        resolve_camera_pose_mode("auto", "functional_semantic_fidelity")
        == "visibility_ranked"
    )
    assert resolve_camera_pose_mode("auto", "scale_consistency") == "visibility_ranked"
    assert (
        resolve_camera_pose_mode("auto", "object_pairing_consistency")
        == "visibility_ranked"
    )
    assert resolve_camera_pose_mode("auto", "style_consistency") == "global_only"
    assert (
        resolve_camera_pose_mode(
            "auto", "semantic_placement_consistency"
        )
        == "visibility_ranked"
    )
    assert DEFAULT_CAMERA_MODE_BY_METRIC["object_architecture_penetration"] == "visibility_ranked"
    overrides = parse_metric_camera_modes(
        [
            "collision=global_only",
            "support=query_cov",
            "scale_consistency=bbox_track",
        ]
    )
    assert resolve_camera_pose_mode("auto", "collision", metric_modes=overrides) == "global_only"
    assert resolve_camera_pose_mode("auto", "support", metric_modes=overrides) == "query_cov"
    assert (
        resolve_camera_pose_mode(
            "auto",
            "scale_consistency",
            metric_modes=overrides,
        )
        == "bbox_track"
    )


def test_auto_provider_renders_functional_semantic_claim_local_evidence(
    tmp_path: Path,
) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")
    renderer = _FocusRenderer()
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend,
        out_dir=tmp_path / "functional_evidence",
        mode="auto",
        max_views=2,
    )
    request = _request("functional_semantic_fidelity")
    request["object_ids"] = ["bed", "cabinet"]
    request["event"] = {
        "type": "local_functionality",
        "claim_id": "local_001",
        "object_ids": ["bed", "cabinet"],
    }

    evidence = provider(request)

    assert evidence
    assert any(call["pass"] == "focus" for call in renderer.calls)
    manifest_paths = list(
        (tmp_path / "functional_evidence").glob(
            "*/camera_evidence_manifest.json"
        )
    )
    assert len(manifest_paths) == 1
    manifest = json.loads(manifest_paths[0].read_text())
    assert manifest["resolved_mode"] == "visibility_ranked"
    assert manifest["metric"] == "functional_semantic_fidelity"


def test_auto_provider_renders_canonical_l3_local_evidence(
    tmp_path: Path,
) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")
    provider = CameraEvidenceProvider(
        renderer=_FocusRenderer(),
        blend_file=blend,
        out_dir=tmp_path / "l3_camera",
        mode="auto",
        max_views=1,
    )

    evidence = provider(_request("scale_consistency"))

    roles = {str(item.get("role")) for item in evidence if isinstance(item, dict)}
    assert "metric_local_rgb" in roles
    assert "metric_highlighted_global" in roles
    assert provider.policy_config["resolved_metric_modes"][
        "scale_consistency"
    ] == "visibility_ranked"


def test_support_contact_plane_mode_is_support_only() -> None:
    assert parse_metric_camera_modes(["support=support_contact_plane"]) == {
        "support": "support_contact_plane"
    }
    try:
        parse_metric_camera_modes(["collision=support_contact_plane"])
    except ValueError as exc:
        assert "only valid for the support metric" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("support-only camera mode was accepted for collision")


def test_support_contact_plane_candidates_target_measured_gap_midpoint() -> None:
    request = _request("support")
    request["object_ids"] = ["bed"]
    request["event"] = {"object_id": "bed"}
    request["_resolved_camera_pose_mode"] = "support_contact_plane"
    request["detector_evidence"] = {
        "base_min_z_m": 0.20,
        "minimum_positive_clearance_m": 0.20,
        "representative_ray_hits": [
            {
                "position": [3.0, 2.5, 0.0],
                "target": "floor",
                "gap_m": 0.20,
                "is_center": True,
            }
        ],
    }

    candidates = generate_camera_pose_candidates(request, max_candidates=6)

    assert len(candidates) == 6
    assert all(item["policy_source"] == "support_contact_plane_candidate_bank_v2" for item in candidates)
    assert all(item["target"] == [3.0, 2.5, 0.1] for item in candidates)
    assert all(item["support_contact_focus"]["gap_m"] == 0.20 for item in candidates)
    assert {round(item["azimuth_degrees"]) for item in candidates} == {0, 60, 120, 180, 240, 300}
    assert all(0.0 < item["elevation_degrees"] < 25.0 for item in candidates)


def test_feasible_collision_is_order_invariant_and_uses_detector_focus() -> None:
    request = _request("collision")
    request["detector_evidence"] = {
        "focus_region": {
            "center": [3.85, 2.4, 0.72],
            "source": "intersection_contact",
        }
    }

    base = generate_camera_pose_candidates(request, max_candidates=6)
    request_order_swapped = deepcopy(request)
    request_order_swapped["object_ids"].reverse()
    scene_order_swapped = deepcopy(request)
    scene_order_swapped["scene"]["objects"].reverse()
    by_request_order = generate_camera_pose_candidates(request_order_swapped, max_candidates=6)
    by_scene_order = generate_camera_pose_candidates(scene_order_swapped, max_candidates=6)

    def geometry(bank: list[dict]) -> list[tuple]:
        return [
            (
                item["name"],
                tuple(round(value, 8) for value in item["location"]),
                tuple(round(value, 8) for value in item["target"]),
                round(item["azimuth_degrees"], 8),
                round(item["elevation_degrees"], 8),
            )
            for item in bank
        ]

    assert geometry(base) == geometry(by_request_order) == geometry(by_scene_order)
    assert [item["focus_kind"] for item in base].count("collision_focus") == 4
    assert [item["focus_kind"] for item in base].count("pair_context") == 2
    assert all(
        item["target"] == [3.85, 2.4, 0.72]
        for item in base
        if item["focus_kind"] == "collision_focus"
    )
    assert all(item["event_focus_source"] == "intersection_contact" for item in base)
    assert all(item["collision_axis_source"] == "object_center_axis" for item in base)


def test_feasible_collision_degenerate_axis_uses_recorded_fallback() -> None:
    request = _request("collision")
    request["scene"]["objects"][1]["center"][:2] = request["scene"]["objects"][0]["center"][:2]

    candidates = generate_camera_pose_candidates(request, max_candidates=6)

    assert len(candidates) == 6
    assert {item["collision_axis_source"] for item in candidates} == {
        "longest_horizontal_obb_axis"
    }

    with_closest_points = deepcopy(request)
    with_closest_points["detector_evidence"] = {
        "closest_points": {
            "object_a": [3.0, 2.0, 0.5],
            "object_b": [3.5, 2.5, 0.5],
        }
    }
    closest_candidates = generate_camera_pose_candidates(with_closest_points, max_candidates=6)
    assert {item["collision_axis_source"] for item in closest_candidates} == {
        "detector_closest_points"
    }


@pytest.mark.parametrize(
    ("flag", "inward_normal"),
    [
        ("west_oob", [1.0, 0.0, 0.0]),
        ("east_oob", [-1.0, 0.0, 0.0]),
        ("south_oob", [0.0, 1.0, 0.0]),
        ("north_oob", [0.0, -1.0, 0.0]),
        ("floor_oob", [0.0, 0.0, 1.0]),
        ("ceiling_oob", [0.0, 0.0, -1.0]),
    ],
)
def test_feasible_oob_single_plane_candidates_are_inward_and_truthful(
    flag: str,
    inward_normal: list[float],
) -> None:
    request = _request("oob")
    request["object_ids"] = ["bed"]
    request["event"] = {"object_id": "bed", "plane_flags": {flag: True}}
    request["detector_evidence"] = {"plane_flags": {flag: True}}

    candidates = generate_camera_pose_candidates(request, max_candidates=6)

    assert len(candidates) == 6
    assert {item["focus_plane_flag"] for item in candidates} == {flag}
    for item in candidates:
        delta = np.asarray(item["location"]) - np.asarray(item["target"])
        assert float(np.dot(delta, np.asarray(inward_normal))) > 0.0
        actual_azimuth, actual_elevation, actual_distance = _actual_pose_angles(item)
        assert item["azimuth_degrees"] == pytest.approx(actual_azimuth)
        assert item["elevation_degrees"] == pytest.approx(actual_elevation)
        assert item["distance_m"] == pytest.approx(actual_distance)
        assert item["feasibility"]["ray_preserved"] is True
    if flag == "ceiling_oob":
        assert all(item["elevation_degrees"] < 0.0 for item in candidates)


def test_feasible_oob_multi_and_opposing_planes_are_all_represented_without_duplicates() -> None:
    flags = {
        "west_oob": True,
        "east_oob": True,
        "floor_oob": True,
        "ceiling_oob": True,
    }
    request = _request("oob")
    request["object_ids"] = ["bed"]
    request["event"] = {"object_id": "bed", "plane_flags": flags}
    request["detector_evidence"] = {"plane_flags": flags}

    candidates = generate_camera_pose_candidates(request, max_candidates=8)

    assert 1 <= len(candidates) <= 8
    assert set(flags).issubset({item["focus_plane_flag"] for item in candidates})
    fingerprints = {
        (
            tuple(round(value, 5) for value in item["location"]),
            tuple(round(value, 5) for value in item["target"]),
        )
        for item in candidates
    }
    assert len(fingerprints) == len(candidates)


def test_feasible_policy_returns_exact_requested_bank_size() -> None:
    for count in (1, 6, 12):
        assert len(generate_camera_pose_candidates(_request(), max_candidates=count)) == count


def test_functional_probe_candidates_are_low_wide_and_context_preserving() -> None:
    request = _request("functional_consistency")
    request["_resolved_camera_pose_mode"] = "query_cov"
    request["functional_probe"] = {
        "probe_id": "functional_probe_01",
        "kind": "functional_correspondence",
        "target_ids": ["bed"],
        "related_target_ids": ["cabinet"],
        "required_observations": [
            "joint_visibility",
            "interaction_side_visible",
            "approach_zone_visible",
        ],
    }

    candidates = generate_camera_pose_candidates(
        request,
        max_candidates=8,
    )

    assert len(candidates) == 8
    assert all(
        item["policy_source"]
        == "functional_required_observation_candidate_bank_v3"
        for item in candidates
    )
    assert all(
        item["event_focus_source"]
        == "functional_probe_relation_target_union"
        for item in candidates
    )
    assert all(
        item["view_family"] == "functional_relation_wide"
        for item in candidates
    )
    assert all(
        8.0 <= item["intended_elevation_degrees"] <= 16.0
        for item in candidates
    )
    assert all(item["lens_mm"] <= 32.0 for item in candidates)
    assert all(
        item["functional_context_margin_m"] == pytest.approx(1.25)
        for item in candidates
    )
    assert all(
        item["proxy_framing_bounds"][0][2] == pytest.approx(0.0)
        for item in candidates
    )
    assert all(
        item["target"][:2] == pytest.approx([3.10189614, 2.5])
        for item in candidates
    )
    assert all(
        item["proxy_framing"]["all_corners_in_front"] is True
        and item["proxy_framing"]["proxy_bounds_fit"] is True
        for item in candidates
    )
    target_min = np.asarray(candidates[0]["target_bounds"][0])
    target_max = np.asarray(candidates[0]["target_bounds"][1])
    framing_min = np.asarray(
        candidates[0]["proxy_framing_bounds"][0]
    )
    framing_max = np.asarray(
        candidates[0]["proxy_framing_bounds"][1]
    )
    assert np.all(framing_min[:2] <= target_min[:2])
    assert np.all(framing_max[:2] >= target_max[:2])


def test_functional_repair_routes_by_check_semantics() -> None:
    directed = _request("functional_consistency")
    directed["object_ids"] = ["bed"]
    directed["functional_repair"] = {
        "target_ids": ["bed"],
        "required_observations": [
            "interaction_side_visible",
            "front_back_disambiguated",
        ],
        "surface_targets": [
            {
                "target_id": "bed",
                "directionality": "directed",
                "surface_roles": ["interaction_side"],
            }
        ],
    }
    directed_candidates = generate_camera_pose_candidates(
        directed,
        max_candidates=4,
    )
    assert directed_candidates
    assert all(
        item["policy_source"]
        == "functional_judge_requested_elevated_side_repair_v1"
        for item in directed_candidates
    )

    non_directed = _request("functional_consistency")
    non_directed["object_ids"] = ["bed"]
    non_directed["functional_repair"] = {
        "target_ids": ["bed"],
        "required_observations": [
            "target_visible",
            "approach_zone_visible",
        ],
        "check_types": ["clearance"],
        "surface_targets": [
            {
                "target_id": "bed",
                "directionality": "non_directed",
                "surface_roles": [],
                "need_clearance": True,
            }
        ],
    }
    clearance_candidates = generate_camera_pose_candidates(
        non_directed,
        max_candidates=4,
    )
    assert clearance_candidates
    assert all(
        item["policy_source"]
        == "functional_required_observation_candidate_bank_v3"
        and item["view_family"] == "functional_frontage_probe"
        for item in clearance_candidates
    )

    relation = _request("functional_consistency")
    relation["object_ids"] = ["bed", "cabinet"]
    relation["functional_repair"] = {
        "target_ids": ["bed", "cabinet"],
        "required_observations": [
            "target_visible",
            "joint_visibility",
        ],
        "check_types": ["within_group_correspondence"],
        "relation_predicates": ["relative_use_geometry"],
        "group_member_ids": ["bed", "cabinet"],
    }
    relation_candidates = generate_camera_pose_candidates(
        relation,
        max_candidates=4,
    )
    assert relation_candidates
    assert all(
        item["view_family"] == "functional_relation_wide"
        for item in relation_candidates
    )

    usable_fallback = _request("functional_consistency")
    usable_fallback["object_ids"] = ["bed", "cabinet"]
    usable_fallback["functional_repair"] = {
        "schema_version": "functional_camera_repair_v3",
        "target_ids": ["bed", "cabinet"],
        "required_observations": [
            "interaction_side_visible",
            "front_back_disambiguated",
        ],
        "usable_side_fallback": True,
        "unresolved_usable_side_target_ids": ["bed", "cabinet"],
        "source_check_ids": ["relation-check"],
    }
    fallback_candidates = generate_camera_pose_candidates(
        usable_fallback,
        max_candidates=4,
    )
    assert len(fallback_candidates) == 4
    assert all(
        item["policy_source"]
        == "functional_usable_side_soft_fallback_v1"
        and item["usable_side_fallback"] is True
        for item in fallback_candidates
    )
    assert {
        (
            tuple(item["target_object_ids"]),
            item["fallback_local_side_id"],
        )
        for item in fallback_candidates
    } == {
        (("bed",), "local_pos_y"),
        (("cabinet",), "local_pos_y"),
        (("bed",), "local_neg_y"),
        (("cabinet",), "local_neg_y"),
    }


def test_cross_group_functional_probe_uses_global_context_when_local_pair_cannot_fit() -> None:
    request = _request("functional_consistency")
    request["_resolved_camera_pose_mode"] = "query_cov"
    request["scene"] = {
        "scene_id": "cross_room_relation",
        "boundary": [[0, 0], [7, 0], [7, 6], [0, 6]],
        "scene_height": 3.0,
        "objects": [
            {
                "id": "sectional_sofa",
                "category": "sofa",
                "center": [3.5, 4.5, 0.42],
                "size": [3.807188, 3.008588, 0.844084],
                "rotation": [0.0, 0.0, 180.0],
            },
            {
                "id": "television",
                "category": "television",
                "center": [3.5, 0.7, 0.97],
                "size": [1.3621, 0.119728, 0.788418],
                "rotation": [0.0, 0.0, 0.0],
            },
        ],
    }
    request["object_ids"] = ["sectional_sofa", "television"]
    request["functional_probe"] = {
        "probe_id": "functional_probe_01",
        "kind": "functional_correspondence",
        "target_ids": ["sectional_sofa"],
        "related_target_ids": ["television"],
        "route_scope": "cross_group",
        "surface_targets": [],
        "required_observations": [
            "joint_visibility",
            "interaction_side_visible",
        ],
    }

    candidates = generate_camera_pose_candidates(
        request,
        max_candidates=4,
    )

    assert [item["camera_type"] for item in candidates] == [
        "PERSP",
        "ORTHO",
    ]
    assert all(
        item["policy_source"]
        == "functional_cross_group_global_context_fallback_v1"
        for item in candidates
    )
    assert all(
        item["view_family"] == "functional_relation_global_context"
        and item["fallback_reason"]
        == "room_interior_joint_framing_infeasible"
        for item in candidates
    )
    assert all(
        item["target_object_ids"]
        == ["sectional_sofa", "television"]
        for item in candidates
    )


def test_functional_probe_refills_feasible_candidate_bank() -> None:
    request = _request("functional_consistency")
    request["_resolved_camera_pose_mode"] = "query_cov"
    request["object_ids"] = ["bed"]
    request["functional_probe"] = {
        "probe_id": "functional_probe_01",
        "kind": "functional_frontage",
        "target_ids": ["bed"],
        "related_target_ids": [],
        "required_observations": [
            "interaction_side_visible",
            "approach_zone_visible",
        ],
    }
    unblocked = generate_camera_pose_candidates(
        request,
        max_candidates=4,
    )
    request["scene"]["objects"].append(
        {
            "id": "camera_blocker",
            "category": "partition",
            "center": unblocked[0]["location"],
            "size": [0.1, 0.1, 0.1],
            "rotation": [0.0, 0.0, 0.0],
        }
    )

    candidates = generate_camera_pose_candidates(
        request,
        max_candidates=4,
    )

    assert 1 <= len(candidates) <= 4
    assert all(
        item["location"] != unblocked[0]["location"]
        for item in candidates
    )
    assert all(
        item["candidate_bank_requested_count"] == 4
        and item["candidate_bank_generated_count"] == 4
        and item["candidate_bank_complete"] is True
        for item in candidates
    )


def test_functional_correspondence_shortlists_six_geometry_poses_to_four():
    request = _request("functional_consistency")
    request["_resolved_camera_pose_mode"] = "query_cov"
    request["functional_probe"] = {
        "probe_id": "functional_probe_01",
        "kind": "functional_correspondence",
        "target_ids": ["bed"],
        "related_target_ids": ["cabinet"],
        "required_observations": [
            "joint_visibility",
            "interaction_side_visible",
        ],
    }

    candidates = generate_camera_pose_candidates(
        request,
        max_candidates=4,
    )

    assert 1 <= len(candidates) <= 4
    assert all(
        item["functional_probe_candidate_pool_count"]
        >= len(candidates)
        and item["functional_probe_shortlist_limit"] == 4
        and item["functional_probe_shortlist_policy"]
        == "local_proxy_framing_context_rank_v1"
        for item in candidates
    )
    assert [
        item["functional_probe_shortlist_rank"]
        for item in candidates
    ] == list(range(1, len(candidates) + 1))
    assert all(
        isinstance(item["functional_probe_shortlist_score"], float)
        for item in candidates
    )


def test_feasible_policy_uses_render_aspect_ratio_for_proxy_projection() -> None:
    request = _request()
    request["_camera_render"] = {"width": 800, "height": 400}

    candidates = generate_camera_pose_candidates(request, max_candidates=6)

    assert all(item["proxy_framing"]["aspect_ratio"] == 2.0 for item in candidates)
    assert all(item["sensor_width_mm"] == 36.0 for item in candidates)
    assert all(item["sensor_fit"] == "HORIZONTAL" for item in candidates)


def test_feasible_policy_rejects_camera_locations_inside_unrelated_object_obb() -> None:
    request = _request()
    blocked_location = generate_camera_pose_candidates(request, max_candidates=6)[0]["location"]
    request["scene"]["objects"].append(
        {
            "id": "camera_blocker",
            "category": "partition",
            "center": blocked_location,
            "size": [0.4, 0.4, 0.4],
            "rotation": [0.0, 0.0, 0.0],
        }
    )

    candidates = generate_camera_pose_candidates(request, max_candidates=6)

    lower = np.asarray(blocked_location, dtype=float) - 0.23
    upper = np.asarray(blocked_location, dtype=float) + 0.23
    assert len(candidates) == 6
    assert all(
        not np.all((lower <= np.asarray(item["location"])) & (np.asarray(item["location"]) <= upper))
        for item in candidates
    )


def test_support_uses_first_requested_subject_not_scene_serialization_order() -> None:
    request = _request("support")
    subject = {
        "id": "subject",
        "category": "cup",
        "center": [3.0, 2.5, 1.1],
        "size": [0.2, 0.2, 0.2],
        "rotation": [0.0, 0.0, 0.0],
    }
    support = {
        "id": "support",
        "category": "table",
        "center": [3.0, 2.5, 0.5],
        "size": [2.0, 2.0, 1.0],
        "rotation": [0.0, 0.0, 0.0],
    }
    request["scene"]["objects"] = [support, subject]
    request["object_ids"] = ["subject", "support"]
    request["event"] = {"object_id": "subject", "object_ids": ["subject", "support"]}
    request["_resolved_camera_pose_mode"] = "support_contact_plane"
    request["detector_evidence"] = {
        "representative_ray_hits": [
            {
                "position": [3.0, 2.5, 1.0],
                "target": "support",
                "gap_m": 0.0,
                "is_center": True,
            }
        ]
    }

    serialized_support_first = generate_camera_pose_candidates(request, max_candidates=6)
    reordered = deepcopy(request)
    reordered["scene"]["objects"].reverse()
    serialized_subject_first = generate_camera_pose_candidates(reordered, max_candidates=6)

    expected_bounds = [[2.9, 2.4, 1.0], [3.1, 2.6, 1.2000000000000002]]
    assert np.allclose(serialized_support_first[0]["target_bounds"], expected_bounds)
    assert np.allclose(
        [item["location"] for item in serialized_support_first],
        [item["location"] for item in serialized_subject_first],
    )


def test_legacy_policy_retains_frozen_direction_and_clamp_semantics() -> None:
    request = _request("oob")
    request["object_ids"] = ["bed"]
    request["event"] = {"object_id": "bed", "plane_flags": {"ceiling_oob": True}}
    request["detector_evidence"] = {"plane_flags": {"ceiling_oob": True}}

    legacy = generate_camera_pose_candidates(request, max_candidates=6, policy="legacy")
    feasible = generate_camera_pose_candidates(request, max_candidates=6, policy="local")

    assert legacy[0]["policy_source"] == "metric_aware_obb_candidate_bank_v1"
    assert legacy[0]["elevation_degrees"] == 65.0
    assert "candidate_policy" not in legacy[0]
    assert feasible[0]["elevation_degrees"] < 0.0


def test_global_only_provider_uses_external_overview_without_local_render(tmp_path: Path) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")
    renderer = _FakeRenderer()
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        mode="global_only",
    )

    assert provider(_request()) == []
    assert renderer.calls == []
    event_dir = next((tmp_path / "evidence").iterdir())
    assert (event_dir / "camera_evidence_manifest.json").is_file()


def test_auto_per_metric_override_can_enable_query_cov_with_same_selector(tmp_path: Path) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")
    renderer = _FakeRenderer()
    selector = _FakeSelector()
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        mode="auto",
        metric_modes={"collision": "query_cov"},
        selector=selector,
        max_views=1,
        max_steps=0,
    )

    paths = provider(_request())

    assert paths
    assert len(selector.calls) == 1
    assert provider.selector is selector
    assert provider.policy_config["resolved_metric_modes"]["collision"] == "query_cov"


def test_oob_focus_overlay_highlights_target_and_flagged_room_plane(tmp_path: Path) -> None:
    scene = _request()["scene"]
    spec = build_focus_overlay_spec(
        scene=scene,
        metric="oob",
        object_ids=["bed"],
        detector_evidence={"plane_flags": {"east_oob": True}},
        architecture_element="room_bounds",
    )

    assert spec["targets"][0]["id"] == "bed"
    assert spec["targets"][0]["required_for_visibility"] is True
    assert spec["architecture_planes"][0]["flag"] == "east_oob"
    assert any(entry["role"] == "architecture_plane" for entry in spec["legend"])

    target_color = tuple(round(channel * 255) for channel in spec["targets"][0]["color"])
    image = Image.new("RGB", (16, 16), (80, 80, 80))
    for x in range(4, 12):
        for y in range(4, 12):
            image.putpixel((x, y), target_color)
    path = tmp_path / "focus.png"
    image.save(path)
    stats = measure_focus_visibility(path, targets=spec["targets"])
    assert stats["target_pixel_fractions"]["bed"] > 0.0


def test_focus_visibility_survives_display_transform_and_shading(tmp_path: Path) -> None:
    path = tmp_path / "shaded_focus.png"
    image = Image.new("RGB", (20, 10), (55, 55, 58))
    for x in range(8):
        for y in range(10):
            image.putpixel((x, y), (107, 33, 30))
    image.save(path)

    stats = measure_focus_visibility(
        path,
        targets=[{"id": "target", "color": [1.0, 0.12, 0.12]}],
    )

    assert stats["target_pixel_fractions"]["target"] == pytest.approx(0.4)


def test_support_focus_overlay_marks_measured_vertical_gap() -> None:
    scene = _request("support")["scene"]
    spec = build_focus_overlay_spec(
        scene=scene,
        metric="support",
        object_ids=["bed"],
        detector_evidence={
            "representative_ray_hits": [
                {
                    "position": [3.0, 2.5, 0.0],
                    "target": "floor",
                    "gap_m": 0.20,
                    "is_center": True,
                }
            ]
        },
    )

    assert [marker["position"] for marker in spec["markers"]] == [
        [3.0, 2.5, 0.0],
        [3.0, 2.5, 0.2],
    ]
    assert spec["connectors"][0]["type"] == "measured_support_gap"
    assert spec["connectors"][0]["gap_m"] == 0.20
    assert spec["focus"]["center"] == [3.0, 2.5, 0.1]
    assert any(entry["role"] == "measured_support_gap" for entry in spec["legend"])


def test_visibility_rank_prefers_visible_then_angularly_distinct_views() -> None:
    candidates = [
        {"id": "front", "azimuth_degrees": 0.0, "elevation_degrees": 25.0},
        {"id": "near_front", "azimuth_degrees": 8.0, "elevation_degrees": 25.0},
        {"id": "side", "azimuth_degrees": 90.0, "elevation_degrees": 25.0},
    ]
    visibility = {
        "front": {"target_pixel_fractions": {"bed": 0.10}},
        "near_front": {"target_pixel_fractions": {"bed": 0.099}},
        "side": {"target_pixel_fractions": {"bed": 0.09}},
    }
    targets = [{"id": "bed", "required_for_visibility": True}]

    selected, log = rank_focus_candidates(
        candidates,
        visibility,
        targets=targets,
        max_views=2,
    )

    assert selected[0]["id"] == "front"
    assert selected[1]["id"] == "side"
    assert log["selector"] == "deterministic_visibility_framing_rank_v1"


def test_support_contact_rank_requires_visible_gap_and_prefers_diversity() -> None:
    candidates = [
        {"id": "front", "azimuth_degrees": 0.0, "elevation_degrees": 8.0},
        {"id": "near_front", "azimuth_degrees": 8.0, "elevation_degrees": 8.0},
        {"id": "side", "azimuth_degrees": 90.0, "elevation_degrees": 8.0},
        {"id": "hidden_gap", "azimuth_degrees": 180.0, "elevation_degrees": 8.0},
    ]
    visibility = {
        "front": {"target_pixel_fractions": {"bed": 0.10}, "focus_pixel_fraction": 0.010},
        "near_front": {"target_pixel_fractions": {"bed": 0.10}, "focus_pixel_fraction": 0.0099},
        "side": {"target_pixel_fractions": {"bed": 0.09}, "focus_pixel_fraction": 0.009},
        "hidden_gap": {"target_pixel_fractions": {"bed": 0.20}, "focus_pixel_fraction": 0.0},
    }

    selected, log = rank_support_contact_candidates(
        candidates,
        visibility,
        targets=[{"id": "bed", "required_for_visibility": True}],
        max_views=2,
    )

    assert [item["id"] for item in selected] == ["front", "side"]
    assert log["selector"] == "support_contact_plane_visibility_rank_v1"
    hidden = next(item for item in log["ranked"] if item["id"] == "hidden_gap")
    assert hidden["usable"] is False


def test_support_final_render_backfills_blank_selected_camera(tmp_path: Path) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")
    renderer = _BackfillFocusRenderer()
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        mode="auto",
        max_views=1,
    )
    bad = {"id": "bad", "location": [1, 1, 1], "target": [0, 0, 0]}
    good = {"id": "good", "location": [-1, 1, 1], "target": [0, 0, 0]}

    _, _, _, rendered, backfill, _ = provider._render_final_focus_evidence(
        request=_request("support"),
        event_dir=tmp_path / "event",
        selected=[bad],
        candidates=[bad, good],
        ranking_log={"ranked": [{"id": "bad"}, {"id": "good"}]},
        overlay_spec={"targets": [{"id": "bed", "color": [1.0, 0.1, 0.1]}]},
        resolved_mode="support_contact_plane",
    )

    assert [item["id"] for item in rendered] == ["good"]
    assert backfill["backfilled"] is True
    assert backfill["rendered_view_ids"] == ["good"]
    assert backfill["skipped_candidates"][0]["id"] == "bad"
    assert ("bundle", ["good"]) in renderer.calls


def test_auto_oob_emits_ranked_local_and_highlighted_global_without_pose_vlm(tmp_path: Path) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")
    renderer = _FocusRenderer()
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        mode="auto",
        max_views=2,
    )
    request = _request("oob")
    request["event"] = {"object_id": "bed", "plane_flags": {"east_oob": True}}
    request["object_ids"] = ["bed"]
    request["detector_evidence"] = {"plane_flags": {"east_oob": True}}

    items = provider(request)

    assert [item["role"] for item in items].count("metric_local_rgb") == 2
    assert [item["role"] for item in items].count("metric_local_highlight") == 2
    assert [item["role"] for item in items].count("metric_highlighted_global") == 1
    assert renderer.calls[0] == {"pass": "focus", "preview": True}
    assert all(call["pass"] in {"focus", "rgb"} for call in renderer.calls)


def test_query_cov_provider_uses_preview_selection_and_one_bounded_step(tmp_path: Path) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")
    renderer = _FakeRenderer()
    selector = _FakeSelector()
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        mode="query_cov",
        selector=selector,
        max_views=1,
        max_steps=1,
    )

    paths = provider(_request())

    assert len(paths) == 1
    assert [call["preview"] for call in renderer.calls] == [True, True, False]
    assert len(selector.calls) == 2
    assert selector.calls[0]["allow_adjustment"] is True
    assert selector.calls[1]["allow_adjustment"] is False
    assert renderer.calls[-1]["views"][0]["camera_action"] == "orbit_left"
    assert provider.policy_config["allowed_camera_actions"] == list(CAMERA_ACTIONS)

    first_usage = deepcopy(provider.last_call_usage)
    assert isinstance(first_usage["call_id"], str)
    assert first_usage["metric"] == "collision"
    assert first_usage["cache_hit"] is False
    assert first_usage["evidence_refs"] == [
        str(paths[0]["view_id"]),
    ]
    manifest_path = Path(first_usage["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["call_usage"] == first_usage
    assert first_usage["selector_calls"] == len(
        manifest["selection"]["steps"]
    ) == 2
    assert first_usage["camera_actions"] == sum(
        isinstance(step["decision"].get("action"), dict)
        for step in manifest["selection"]["steps"]
    ) == 1

    cached_paths = provider(_request())

    assert cached_paths == paths
    assert len(renderer.calls) == 3
    assert provider.last_call_usage["call_id"] != first_usage["call_id"]
    assert provider.last_call_usage["cache_hit"] is True
    assert provider.last_call_usage["evidence_refs"] == [
        str(paths[0]["view_id"]),
    ]
    assert provider.last_call_usage["selector_calls"] == 0
    assert provider.last_call_usage["camera_actions"] == 0
    cached_manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    assert cached_manifest["call_usage"] == provider.last_call_usage
    assert cached_manifest["selection"] == manifest["selection"]
    assert cached_manifest["render_evidence"] == manifest["render_evidence"]


def test_camera_provider_rehashes_same_size_same_mtime_source_blend(
    tmp_path: Path,
) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")
    original_stat = blend.stat()
    first = CameraEvidenceProvider(
        renderer=_FakeRenderer(),
        blend_file=blend,
        out_dir=tmp_path / "first_evidence",
        mode="bbox_track",
    )

    blend.write_bytes(b"BLEND")
    os.utime(
        blend,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    second = CameraEvidenceProvider(
        renderer=_FakeRenderer(),
        blend_file=blend,
        out_dir=tmp_path / "second_evidence",
        mode="bbox_track",
    )

    assert second.source_blend_sha256 != first.source_blend_sha256


def test_camera_provider_rehashes_same_size_same_mtime_cached_evidence(
    tmp_path: Path,
) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")
    renderer = _FakeRenderer()
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        mode="bbox_track",
        max_views=1,
    )
    request = _request()

    paths = provider(request)
    evidence_path = Path(paths[0])
    original_stat = evidence_path.stat()
    evidence_path.write_bytes(b"bad")
    os.utime(
        evidence_path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )

    rerendered_paths = provider(request)

    assert len(renderer.calls) == 2
    assert provider.last_call_usage["cache_hit"] is False
    assert rerendered_paths == paths
    assert evidence_path.read_bytes() == b"png"


def test_query_cov_can_render_frozen_vlm_selection_without_runtime_selector(
    tmp_path: Path,
) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")
    renderer = _FakeRenderer()
    candidates = generate_camera_pose_candidates(
        {**_request(), "_resolved_camera_pose_mode": "query_cov"},
        max_candidates=6,
    )
    selected_id = str(candidates[1]["id"])
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        mode="query_cov",
        selector=None,
        max_views=1,
        max_steps=0,
        candidate_count=6,
        frozen_view_ids=[selected_id],
    )

    paths = provider(_request())

    assert len(paths) == 1
    assert [call["preview"] for call in renderer.calls] == [False]
    assert renderer.calls[0]["views"][0]["id"] == selected_id
    assert provider.policy_config["selection_source"] == "frozen_vlm_selected_view_ids"
    assert provider.policy_config["allowed_camera_actions"] == []


def test_functional_probe_provider_returns_only_unmodified_rgb(
    tmp_path: Path,
) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")
    renderer = _FakeRenderer()
    selector = _FakeSelector()
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend,
        out_dir=tmp_path / "functional_probe",
        mode="query_cov",
        selector=selector,
        max_views=1,
        max_steps=0,
        candidate_count=6,
    )
    request = _request("functional_consistency")
    request.update(
        evidence_scope="pair_local",
        evidence_policy={
            "camera_scope": "pair_local",
            "camera_pose_mode": "query_cov",
            "presentation": "raw",
        },
        functional_probe={
            "probe_id": "functional_probe_01",
            "kind": "functional_correspondence",
            "target_ids": ["bed"],
            "related_target_ids": ["cabinet"],
            "target_categories": {
                "bed": "bed",
                "cabinet": "cabinet",
            },
            "required_observations": [
                "joint_visibility",
                "interaction_side_visible",
                "approach_zone_visible",
            ],
            "view_goal": "show the usable sides together",
        },
    )

    items = provider(request)

    assert len(items) == 1
    assert items[0]["role"] == "functional_probe_rgb"
    assert items[0]["evidence_style"] == "raw"
    assert items[0]["image_transform"] == "none"
    assert [call["preview"] for call in renderer.calls] == [
        True,
        False,
        False,
    ]
    assert [call.get("pass", "rgb") for call in renderer.calls] == [
        "identity",
        "rgb",
        "identity",
    ]
    assert len(selector.calls) == 1
    assert len(renderer.calls[0]["views"]) == 4
    assert selector.calls[0]["preview_role"] == "highlighted_focus"
    assert selector.calls[0]["functional_probe"]["kind"] == (
        "functional_correspondence"
    )
    manifest = json.loads(
        (
            next(
                (tmp_path / "functional_probe").glob(
                    "*/camera_evidence_manifest.json"
                )
            )
        ).read_text(encoding="utf-8")
    )
    assert manifest["judge_presentation"] == "raw_rgb_only"
    assert manifest["source_scene_pixels_modified"] is False
    assert manifest["candidate_budget"] == {
        "configured_max": 6,
        "requested": 4,
        "generated": 4,
        "pool_generated": 6,
        "shortlist_policy": "local_proxy_framing_context_rank_v1",
    }


def test_functional_frontage_probe_uses_four_candidates(
    tmp_path: Path,
) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")
    renderer = _FakeRenderer()
    selector = _FakeSelector()
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend,
        out_dir=tmp_path / "functional_frontage",
        mode="query_cov",
        selector=selector,
        max_views=1,
        max_steps=0,
        candidate_count=6,
    )
    request = _request("functional_consistency")
    request.update(
        object_ids=["bed"],
        evidence_scope="object_local",
        evidence_policy={
            "camera_scope": "object_local",
            "camera_pose_mode": "query_cov",
            "presentation": "raw",
        },
        functional_probe={
            "probe_id": "functional_probe_01",
            "kind": "functional_frontage",
            "target_ids": ["bed"],
            "related_target_ids": [],
            "target_categories": {"bed": "bed"},
            "required_observations": [
                "interaction_side_visible",
                "front_back_disambiguated",
                "approach_zone_visible",
            ],
            "view_goal": "show the usable face and outward context",
        },
    )

    items = provider(request)

    assert len(items) == 1
    assert len(renderer.calls[0]["views"]) == 4
    assert provider.last_call_usage["candidate_count_requested"] == 4
    assert provider.last_call_usage["candidate_count_generated"] == 4
