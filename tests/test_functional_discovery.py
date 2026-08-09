from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from PIL import Image
import pytest

from benchmark.evaluator.scene_quality.functional_acquisition import (
    build_functional_acquisition_plan,
)
from benchmark.evaluator.scene_quality.functional_boundary_evidence import (
    acquire_functional_boundary_evidence,
    discovery_with_boundary_hypotheses,
    qualifying_boundary_surface_targets,
    qualifying_direction_surface_targets,
)
from benchmark.evaluator.scene_quality.functional_geometry import (
    build_functional_geometry_observations,
)
from benchmark.evaluator.scene_quality.functional_probe import (
    acquire_functional_probe_evidence,
    functional_probe_judge_packet,
)
from benchmark.evaluator.scene_quality.group_scoped import (
    resolve_group_evidence_packets,
)
from benchmark.rendering.camera_pose import (
    generate_camera_pose_candidates,
    generate_usable_surface_side_bank,
    generate_usable_surface_side_repair_bank,
)
from benchmark.visual_judge.functional_discovery import (
    FUNCTIONAL_AFFORDANCE_SYSTEM_PROMPT,
    FUNCTIONAL_RELATION_SYSTEM_PROMPT,
    compose_functional_discovery_result,
    validate_functional_affordance_response,
    validate_functional_discovery_response,
    validate_functional_relation_response,
)
from benchmark.visual_judge.functional_evidence import (
    functional_probe_selector_context,
)
from benchmark.visual_judge.contracts import ResponseSchemaRepairError
from benchmark.visual_judge.openai_camera_selector import (
    OpenAICompatibleCameraSelector,
)
from benchmark.visual_judge.orchestration.audit import (
    evidence_artifact_refs,
)
from benchmark.visual_judge.orchestration.budget import (
    extend_acquisition_ledger,
)
from benchmark.visual_judge.render_views import CameraEvidenceProvider
from benchmark.visual_judge.placement_discovery import (
    PLACEMENT_DISCOVERY_SYSTEM_PROMPT,
    discover_openai_compatible_placement_evidence,
    validate_placement_discovery_response,
)
from benchmark.visual_judge.usable_surface import (
    DEFAULT_USABLE_SURFACE_DETECTOR_BACKEND,
    USABLE_SURFACE_SYSTEM_PROMPT,
    VLMTrustedSideUsableSurfaceDetector,
    usable_surface_cache_identity,
    validate_usable_surface_response,
)


GROUPS = [
    {"group_id": "seating", "object_ids": ["sofa"]},
    {"group_id": "media", "object_ids": ["television", "cabinet"]},
]


def _raw_discovery() -> dict:
    return {
        "inspected_object_ids": ["sofa", "television", "cabinet"],
        "directed_surface_targets": [
            {
                "target_id": "sofa",
                "surface_roles": ["seating_side"],
                "need_clearance": False,
                "observation_goal": "show the seating side and outward space",
            },
            {
                "target_id": "television",
                "surface_roles": ["display_side"],
                "need_clearance": False,
                "observation_goal": "show the display side and viewing area",
            },
            {
                "target_id": "cabinet",
                "surface_roles": ["opening_side"],
                "need_clearance": True,
                "observation_goal": "show the access side and approach area",
            },
        ],
        "functional_correspondences": [
            {
                "target_ids": ["sofa", "television"],
                "observation_goal": (
                    "show both usable sides and their mutual orientation"
                ),
            }
        ],
        "approach_clearance_targets": [
            {
                "target_id": "cabinet",
                "need_clearance": True,
                "observation_goal": "show ordinary opening clearance",
            }
        ],
        "boundary_sensitive_targets": [
            {
                "target_id": "cabinet",
                "observation_goal": (
                    "show the access side, interior region, and boundary"
                ),
            }
        ],
        "unusual_unconfirmed": [
            {
                "target_ids": ["cabinet"],
                "observation_goal": (
                    "show the cabinet access side in its group context"
                ),
                "audit_reason": "the global angle does not establish frontage",
            }
        ],
        "reason": "all objects inspected",
    }


def _normalized() -> dict:
    validated = validate_functional_discovery_response(
        _raw_discovery(),
        object_ids=("sofa", "television", "cabinet"),
        groups=GROUPS,
    )
    return {
        "schema_version": "functional_discovery_v3",
        **{
            key: list(deepcopy(value))
            if isinstance(value, tuple)
            else deepcopy(value)
            for key, value in validated.items()
        },
    }


def test_discovery_normalizes_cross_group_and_group_owned_targets() -> None:
    result = _normalized()

    assert result["within_group_correspondences"] == []
    relation = result["cross_group_correspondences"][0]
    assert relation["target_ids"] == ["sofa", "television"]
    assert relation["group_ids"] == ["seating", "media"]
    assert result["unusual_unconfirmed"][0]["owning_group_id"] == "media"
    assert result["directed_surface_targets"][2][
        "owning_group_id"
    ] == "media"


def test_discovery_rejects_incomplete_coverage_and_metric_decision() -> None:
    raw = _raw_discovery()
    raw["inspected_object_ids"] = ["sofa", "television"]
    with pytest.raises(ValueError, match="every input object"):
        validate_functional_discovery_response(
            raw,
            object_ids=("sofa", "television", "cabinet"),
            groups=GROUPS,
        )

    for forbidden, value in (
        ("camera_pose", {"location": [0, 0, 0]}),
        ("scene_mutation", {"objects": []}),
    ):
        raw = _raw_discovery()
        raw[forbidden] = value
        with pytest.raises(ValueError, match="may not return"):
            validate_functional_discovery_response(
                raw,
                object_ids=("sofa", "television", "cabinet"),
                groups=GROUPS,
            )

    raw = _raw_discovery()
    raw["directed_surface_targets"][0]["target_id"] = "unknown_object"
    with pytest.raises(ValueError, match="unknown object ID"):
        validate_functional_discovery_response(
            raw,
            object_ids=("sofa", "television", "cabinet"),
            groups=GROUPS,
        )

    raw = _raw_discovery()
    raw["verdict"] = "invalid"
    with pytest.raises(ValueError, match="may not return"):
        validate_functional_discovery_response(
            raw,
            object_ids=("sofa", "television", "cabinet"),
            groups=GROUPS,
        )


def test_discovery_rejects_cross_group_unusual_confirmation() -> None:
    raw = _raw_discovery()
    raw["unusual_unconfirmed"][0]["target_ids"] = [
        "sofa",
        "cabinet",
    ]
    with pytest.raises(ValueError, match="exactly one owning group"):
        validate_functional_discovery_response(
            raw,
            object_ids=("sofa", "television", "cabinet"),
            groups=GROUPS,
        )


def test_acquisition_routes_cross_group_and_neutral_group_confirmation() -> None:
    plan = build_functional_acquisition_plan(
        _normalized(),
        max_probe_units=4,
    )

    cross = next(
        item
        for item in plan["probe_units"]
        if item["route_scope"] == "cross_group"
    )
    assert cross["kind"] == "functional_correspondence"
    assert cross["surface_targets"] == [
        {
            "target_id": "sofa",
            "directionality": "directed",
            "surface_roles": ["seating_side"],
            "need_clearance": False,
        },
        {
            "target_id": "television",
            "directionality": "directed",
            "surface_roles": ["display_side"],
            "need_clearance": False,
        },
    ]
    group = next(
        item
        for item in plan["probe_units"]
        if item.get("target_ids") == ["cabinet"]
    )
    assert group["owning_group_id"] == "media"
    assert "global angle" not in group["view_goal"]
    confirmation = plan["group_confirmations"][0]
    assert confirmation["audit_reason"] == (
        "the global angle does not establish frontage"
    )
    assert confirmation["neutral_observation_goal"] == (
        "show the cabinet access side in its group context"
    )
    assert group["target_ids"] == ["cabinet"]
    assert group["acquisition_trigger"] == (
        "directed_usable_side"
    )
    assert group["acquisition_triggers"] == [
        "directed_usable_side",
        "boundary_sensitive_frontage",
        "approach_clearance",
        "unusual_unconfirmed_group_confirmation",
    ]
    assert group["evidence_reuse"][
        "reuses_side_conditioned_view_for_clearance"
    ] is True
    assert group["evidence_reuse"]["clearance_observation_frame"] == (
        "usable_side_forward_region"
    )
    assert group["evidence_reuse"]["additional_probe_image_budget"] == 1
    assert len(group["observation_goals"]) == 4
    selector_context = functional_probe_selector_context(group)
    assert selector_context["observation_goals"] == (
        group["observation_goals"]
    )
    assert "global angle" not in json.dumps(
        selector_context,
        sort_keys=True,
    )
    assert len(
        [
            item
            for item in plan["probe_units"]
            if item.get("owning_group_id") == "media"
        ]
    ) == 2
    assert plan["coverage_complete"] is True
    assert plan["budget_exhausted"] is False


def test_acquisition_budget_exhaustion_is_explicit() -> None:
    discovery = {
        "directed_surface_targets": [],
        "within_group_correspondences": [],
        "cross_group_correspondences": [
            {
                "discovery_id": f"relation_{index}",
                "target_ids": [
                    f"left_{index}",
                    f"right_{index}",
                ],
                "observation_goal": f"show relation {index}",
            }
            for index in range(5)
        ],
        "approach_clearance_targets": [],
        "boundary_sensitive_targets": [],
        "unusual_unconfirmed": [],
    }

    plan = build_functional_acquisition_plan(
        discovery,
        max_probe_units=4,
    )

    assert len(plan["probe_units"]) == 4
    assert plan["coverage_complete"] is False
    assert plan["budget_exhausted"] is True
    assert len(plan["unscheduled_discovery_items"]) == 1
    unscheduled = plan["unscheduled_discovery_items"][0]
    assert unscheduled["discovery_ids"] == ["relation_4"]
    assert unscheduled["target_ids"] == ["left_4", "right_4"]
    assert unscheduled["observation_goal"] == "show relation 4"
    assert unscheduled["reason"] == "max_probe_units_exhausted"
    assert unscheduled["acquisition_identity"][0] == (
        "functional_correspondence"
    )


def test_only_cross_group_relation_is_proactively_scheduled() -> None:
    discovery = {
        "directed_surface_targets": [],
        "within_group_correspondences": [
            {
                "discovery_id": "local_first_alphabetically",
                "target_ids": ["a", "b"],
                "group_ids": ["group_local"],
                "observation_kinds": ["mutual_orientation"],
                "observation_goal": "show the in-group relationship",
            }
        ],
        "cross_group_correspondences": [
            {
                "discovery_id": "cross_last_alphabetically",
                "target_ids": ["y", "z"],
                "group_ids": ["group_y", "group_z"],
                "observation_kinds": ["mutual_orientation"],
                "observation_goal": "show the cross-group relationship",
            }
        ],
        "approach_clearance_targets": [],
        "boundary_sensitive_targets": [],
        "unusual_unconfirmed": [],
    }

    plan = build_functional_acquisition_plan(
        discovery,
        max_probe_units=1,
    )

    assert plan["probe_units"][0]["route_scope"] == "cross_group"
    assert plan["probe_units"][0]["discovery_ids"] == [
        "cross_last_alphabetically"
    ]
    assert plan["backfill_probe_units"] == []
    local_checks = [
        item
        for item in plan["functional_check_ledger"]["checks"]
        if item["owner_stage"] == "group_local"
    ]
    assert local_checks
    assert all(
        item["acquisition_status"] == "judge_requested_on_demand"
        for item in local_checks
    )


def test_failed_probe_is_backfilled_by_next_deterministic_candidate(
    tmp_path: Path,
) -> None:
    global_image = tmp_path / "global.png"
    replacement_image = tmp_path / "replacement.png"
    Image.new("RGB", (16, 16), (40, 50, 60)).save(global_image)
    Image.new("RGB", (16, 16), (70, 80, 90)).save(
        replacement_image
    )
    discovery = {
        "schema_version": "functional_discovery_v3",
        "inspected_object_ids": ["a", "b", "c", "d"],
        "directed_surface_targets": [],
        "within_group_correspondences": [],
        "cross_group_correspondences": [
            {
                "discovery_id": "relation_01",
                "target_ids": ["a", "b"],
                "group_ids": ["group_a", "group_b"],
                "observation_kinds": ["mutual_orientation"],
                "observation_goal": "show a and b together",
            },
            {
                "discovery_id": "relation_02",
                "target_ids": ["c", "d"],
                "group_ids": ["group_c", "group_d"],
                "observation_kinds": ["mutual_orientation"],
                "observation_goal": "show c and d together",
            },
        ],
        "approach_clearance_targets": [],
        "boundary_sensitive_targets": [],
        "unusual_unconfirmed": [],
        "reason": "two deterministic candidates",
        "provenance": {},
    }

    class Planner:
        def discover_functional_evidence(self, request: dict) -> dict:
            return deepcopy(discovery)

    calls: list[list[str]] = []

    def provider(request: dict) -> list[dict]:
        target_ids = list(request["object_ids"])
        calls.append(target_ids)
        if target_ids == ["a", "b"]:
            raise RuntimeError("no feasible candidate")
        return [
            {
                "path": str(replacement_image),
                "role": "functional_probe_rgb",
                "evidence_style": "raw",
                "image_transform": "none",
            }
        ]

    paths, audit = acquire_functional_probe_evidence(
        planner=Planner(),
        provider=provider,
        scene={
            "scene_id": "scene",
            "scene_type": "living_room",
            "objects": [
                {"id": object_id, "category": "object"}
                for object_id in ("a", "b", "c", "d")
            ],
        },
        global_image_path=str(global_image),
        max_probe_units=1,
        groups=[
            {"group_id": f"group_{object_id}", "object_ids": [object_id]}
            for object_id in ("a", "b", "c", "d")
        ],
    )
    packet = functional_probe_judge_packet(
        global_paths=[str(global_image)],
        probe_paths=paths,
        acquisition_audit=audit,
    )

    assert calls == [["a", "b"], ["c", "d"]]
    assert paths == [str(replacement_image)]
    assert audit["attempted_backfill_count"] == 1
    assert audit["successful_probe_count"] == 1
    assert audit["failed_probe_count"] == 1
    assert audit["budget_exhausted"] is False
    assert audit["failed_discovery_items"][0]["target_ids"] == [
        "a",
        "b",
    ]
    relation = packet["relation_observation_requests"][0]
    assert relation["target_ids"] == ["c", "d"]
    assert relation["relation_predicates"] == [
        "directional_correspondence"
    ]
    assert relation["observation_kinds"] == [
        "directional_correspondence"
    ]
    assert relation["decision_authority"] == "none"


def test_directed_affordance_routes_to_usable_side_check() -> None:
    affordance = validate_functional_affordance_response(
        {
            "objects": [
                {
                    "object_id": "a",
                    "directionality": "directed",
                    "surface_roles": ["interaction_side"],
                    "need_clearance": False,
                    "boundary_review_state": "routine",
                    "review_state": "routine",
                    "observation_goal": "show the ordinary interaction side",
                    "boundary_observation_goal": "",
                },
                {
                    "object_id": "b",
                    "directionality": "non_directed",
                    "surface_roles": [],
                    "need_clearance": False,
                    "boundary_review_state": "routine",
                    "review_state": "routine",
                    "observation_goal": "show ordinary access",
                    "boundary_observation_goal": "",
                },
            ],
            "reason": "complete ledger",
        },
        object_ids=("a", "b"),
    )
    relations = validate_functional_relation_response(
        {
            "considered_object_ids": ["a", "b"],
            "relations": [],
            "reason": "no direct dependency",
        },
        object_ids=("a", "b"),
    )
    discovery = compose_functional_discovery_result(
        affordance=affordance,
        relations=relations,
        object_ids=("a", "b"),
        groups=[{"group_id": "g", "object_ids": ["a", "b"]}],
    )

    assert discovery["inspected_object_ids"] == ("a", "b")
    plan = build_functional_acquisition_plan(
        {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in discovery.items()
        },
        max_probe_units=4,
    )
    assert len(plan["probe_units"]) == 1
    assert plan["probe_units"][0]["target_ids"] == ["a"]
    assert plan["probe_units"][0]["acquisition_trigger"] == (
        "directed_usable_side"
    )


def test_routine_boundary_goal_is_deterministically_cleared() -> None:
    affordance = validate_functional_affordance_response(
        {
            "objects": [
                {
                    "object_id": "cabinet",
                    "directionality": "directed",
                    "surface_roles": ["opening_side"],
                    "need_clearance": True,
                    "boundary_review_state": "routine",
                    "review_state": "routine",
                    "observation_goal": "show the opening side",
                    "boundary_observation_goal": (
                        "show ordinary separation from the wall"
                    ),
                }
            ],
            "reason": "complete ledger with redundant boundary text",
        },
        object_ids=("cabinet",),
    )

    assert affordance["objects"][0]["boundary_observation_goal"] == ""
    assert affordance["normalization_warnings"] == [
        {
            "code": "routine_boundary_goal_cleared",
            "object_id": "cabinet",
            "field": "boundary_observation_goal",
            "repair": "cleared",
            "reason": "boundary_review_state_is_routine",
        }
    ]


def test_non_routine_boundary_review_still_requires_goal() -> None:
    with pytest.raises(
        ValueError,
        match="non-routine boundary review requires",
    ):
        validate_functional_affordance_response(
            {
                "objects": [
                    {
                        "object_id": "cabinet",
                        "directionality": "directed",
                        "surface_roles": ["opening_side"],
                        "need_clearance": True,
                        "boundary_review_state": "local_confirmation",
                        "review_state": "routine",
                        "observation_goal": "show the opening side",
                        "boundary_observation_goal": "",
                    }
                ],
                "reason": "missing required boundary goal",
            },
            object_ids=("cabinet",),
        )


def test_boundary_context_cannot_create_non_directed_clearance_check() -> None:
    with pytest.raises(
        ValueError,
        match="boundary context cannot create an independent functional check",
    ):
        validate_functional_affordance_response(
            {
                "objects": [
                    {
                        "object_id": "plant",
                        "directionality": "non_directed",
                        "surface_roles": [],
                        "need_clearance": False,
                        "boundary_review_state": "local_confirmation",
                        "review_state": "routine",
                        "observation_goal": "show the plant in local context",
                        "boundary_observation_goal": (
                            "show the plant relative to the room boundary"
                        ),
                    }
                ],
                "reason": "boundary-only routing must fail closed",
            },
            object_ids=("plant",),
        )


@pytest.mark.parametrize(
    "legacy_directionality",
    ["uncertain", "omnidirectional", "no_ordinary_operation"],
)
def test_directionality_contract_rejects_legacy_values(
    legacy_directionality: str,
) -> None:
    with pytest.raises(ValueError, match="directionality"):
        validate_functional_affordance_response(
            {
                "objects": [
                    {
                        "object_id": "fixture",
                        "directionality": legacy_directionality,
                        "surface_roles": ["control_side"],
                        "need_clearance": True,
                        "boundary_review_state": "routine",
                        "review_state": "routine",
                        "observation_goal": (
                            "identify the control side and operating space"
                        ),
                        "boundary_observation_goal": "",
                    }
                ],
                "reason": "legacy non-binary directionality",
            },
            object_ids=("fixture",),
        )


@pytest.mark.parametrize(
    "legacy_clearance",
    ["none", "approach", "opening", "operation", "uncertain"],
)
def test_clearance_contract_rejects_legacy_enum(
    legacy_clearance: str,
) -> None:
    with pytest.raises(ValueError, match="unsupported fields"):
        validate_functional_affordance_response(
            {
                "objects": [
                    {
                        "object_id": "fixture",
                        "directionality": "directed",
                        "surface_roles": ["control_side"],
                        "clearance_need": legacy_clearance,
                        "boundary_review_state": "routine",
                        "review_state": "routine",
                        "observation_goal": "inspect ordinary use",
                        "boundary_observation_goal": "",
                    }
                ],
                "reason": "legacy clearance subtype",
            },
            object_ids=("fixture",),
        )


def test_non_directed_affordance_may_independently_require_clearance() -> None:
    affordance = validate_functional_affordance_response(
        {
            "objects": [
                {
                    "object_id": "round_table",
                    "directionality": "non_directed",
                    "surface_roles": [],
                    "need_clearance": True,
                    "boundary_review_state": "routine",
                    "review_state": "routine",
                    "observation_goal": "show ordinary access around the table",
                    "boundary_observation_goal": "",
                }
            ],
            "reason": "direction and free-space need are independent",
        },
        object_ids=("round_table",),
    )
    discovery = compose_functional_discovery_result(
        affordance=affordance,
        relations={
            "considered_object_ids": ["round_table"],
            "relations": [],
            "reason": "no direct joint-use relation",
        },
        object_ids=("round_table",),
        groups=[
            {
                "group_id": "dining",
                "object_ids": ["round_table"],
            }
        ],
    )

    assert discovery["directed_surface_targets"] == ()
    assert discovery["approach_clearance_targets"][0][
        "need_clearance"
    ] is True


def test_boundary_prepass_ignores_routine_review_state_but_requires_clearance() -> None:
    discovery = {
        "directed_surface_targets": [
            {
                "target_id": "routine_target",
                "directionality": "directed",
                "surface_roles": ["opening_side"],
                "need_clearance": True,
                "boundary_review_state": "routine",
            },
            {
                "target_id": "uncertain_target",
                "directionality": "directed",
                "surface_roles": ["service_side"],
                "need_clearance": True,
                "boundary_review_state": "routine",
            },
            {
                "target_id": "no_clearance",
                "directionality": "directed",
                "surface_roles": ["display_side"],
                "need_clearance": False,
                "boundary_review_state": "local_confirmation",
            },
        ]
    }

    targets = qualifying_boundary_surface_targets(
        discovery,
        architecture_context={"logical_boundary_enabled": True},
    )

    assert [item["target_id"] for item in targets] == [
        "routine_target",
        "uncertain_target",
    ]


def test_direction_prepass_includes_every_directed_target() -> None:
    discovery = {
        "directed_surface_targets": [
            {
                "target_id": "display",
                "directionality": "directed",
                "surface_roles": ["display_side"],
                "need_clearance": False,
            },
            {
                "target_id": "cabinet",
                "directionality": "directed",
                "surface_roles": ["opening_side"],
                "need_clearance": True,
            },
        ]
    }

    targets = qualifying_direction_surface_targets(discovery)

    assert [item["target_id"] for item in targets] == [
        "display",
        "cabinet",
    ]
    assert [item["need_clearance"] for item in targets] == [False, True]


def test_direction_prepass_runs_without_a_logical_boundary() -> None:
    class Provider:
        def __init__(self) -> None:
            self.calls = 0

        def provide_functional_boundary_evidence(
            self,
            request: dict,
        ) -> dict:
            self.calls += 1
            target = deepcopy(request["surface_targets"][0])
            return {
                "status": "complete",
                "decision_authority": "none",
                "scene_access": "read_only",
                "surface_targets": [target],
                "usable_surface_hypotheses": [
                    {
                        "target_id": target["target_id"],
                        "status": "identified",
                        "surfaces": [
                            {
                                "surface_role": "display_side",
                                "side_id": "local_pos_y",
                                "visual_cues": ["screen plane"],
                                "confidence": 0.9,
                            }
                        ],
                        "reason": "screen plane is visible",
                    }
                ],
                "functional_geometry": {
                    "schema_version": "functional_geometry_v2",
                    "decision_authority": "none",
                    "scene_access": "read_only",
                    "logical_boundary_available": False,
                    "surface_observations": [],
                    "observation_status": "available",
                },
                "decoder_audit": {
                    "decoder_calls": 1,
                    "cache_hits": 0,
                    "preview_render_count": 4,
                },
            }

    provider = Provider()
    evidence = acquire_functional_boundary_evidence(
        provider=provider,
        scene={"objects": []},
        discovery={
            "directed_surface_targets": [
                {
                    "target_id": "display",
                    "directionality": "directed",
                    "surface_roles": ["display_side"],
                    "need_clearance": False,
                }
            ]
        },
        architecture_context={"logical_boundary_enabled": False},
    )

    assert provider.calls == 1
    assert evidence["provider_invoked"] is True
    assert evidence["logical_boundary_enabled"] is False
    assert evidence["requested_target_ids"] == ["display"]


def test_relation_units_are_stable_and_never_merge_by_group() -> None:
    discovery = {
        "directed_surface_targets": [],
        "within_group_correspondences": [
            {
                "discovery_id": "second",
                "target_ids": ["c", "d"],
                "group_ids": ["g"],
                "observation_kinds": ["cooperative_operation"],
                "observation_goal": "show c and d together",
            },
            {
                "discovery_id": "first",
                "target_ids": ["a", "b"],
                "group_ids": ["g"],
                "observation_kinds": ["mutual_orientation"],
                "observation_goal": "show a and b together",
            },
        ],
        "cross_group_correspondences": [],
        "approach_clearance_targets": [],
        "boundary_sensitive_targets": [],
        "unusual_unconfirmed": [],
    }
    plan = build_functional_acquisition_plan(
        discovery,
        max_probe_units=4,
    )

    assert plan["probe_units"] == []
    checks = plan["functional_check_ledger"]["checks"]
    assert {
        tuple(item["target_ids"])
        for item in checks
        if item["owner_stage"] == "group_local"
    } == {("a", "b"), ("c", "d")}
    assert all(
        item["acquisition_status"] == "judge_requested_on_demand"
        for item in checks
    )


def test_boundary_context_does_not_create_clearance_without_need() -> None:
    discovery = {
        "directed_surface_targets": [],
        "within_group_correspondences": [],
        "cross_group_correspondences": [],
        "approach_clearance_targets": [],
        "boundary_sensitive_targets": [
            {
                "discovery_id": "boundary_01",
                "target_id": "a",
                "owning_group_id": "g",
                "observation_goal": (
                    "show ordinary use direction relative to the boundary"
                ),
            }
        ],
        "unusual_unconfirmed": [],
    }
    plan = build_functional_acquisition_plan(
        discovery,
        max_probe_units=4,
    )

    assert plan["probe_units"] == []
    assert plan["functional_check_ledger"]["checks"] == []


def test_n025_n028_structural_acquisition_coverage_regressions() -> None:
    """Case names document coverage only; production has no category rules."""

    n025 = validate_functional_discovery_response(
        {
            "inspected_object_ids": ["sofa", "television"],
            "directed_surface_targets": [
                {
                    "target_id": "sofa",
                    "surface_roles": ["seating_side"],
                    "need_clearance": False,
                    "observation_goal": "show the seating side",
                },
                {
                    "target_id": "television",
                    "surface_roles": ["display_side"],
                    "need_clearance": False,
                    "observation_goal": "show the display side",
                },
            ],
            "functional_correspondences": [
                {
                    "target_ids": ["sofa", "television"],
                    "observation_goal": (
                        "show both usable sides and their relationship"
                    ),
                }
            ],
            "approach_clearance_targets": [],
            "boundary_sensitive_targets": [],
            "unusual_unconfirmed": [],
            "reason": "N025 acquisition coverage",
        },
        object_ids=("sofa", "television"),
        groups=[
            {"group_id": "seating", "object_ids": ["sofa"]},
            {"group_id": "media", "object_ids": ["television"]},
        ],
    )
    n025_plan = build_functional_acquisition_plan(
        {key: list(value) if isinstance(value, tuple) else value
         for key, value in n025.items()},
        max_probe_units=4,
    )
    assert n025_plan["probe_units"][0]["route_scope"] == "cross_group"
    assert set(
        [
            *n025_plan["probe_units"][0]["target_ids"],
            *n025_plan["probe_units"][0]["related_target_ids"],
        ]
    ) == {"sofa", "television"}

    n026 = validate_functional_discovery_response(
        {
            "inspected_object_ids": ["mirror"],
            "directed_surface_targets": [
                {
                    "target_id": "mirror",
                    "surface_roles": ["reflective_side"],
                    "need_clearance": False,
                    "observation_goal": "show the reflective side",
                }
            ],
            "functional_correspondences": [],
            "approach_clearance_targets": [],
            "boundary_sensitive_targets": [],
            "unusual_unconfirmed": [],
            "reason": "N026 acquisition coverage",
        },
        object_ids=("mirror",),
        groups=[{"group_id": "dressing", "object_ids": ["mirror"]}],
    )
    n026_plan = build_functional_acquisition_plan(
        {key: list(value) if isinstance(value, tuple) else value
         for key, value in n026.items()},
        max_probe_units=4,
    )
    assert len(n026_plan["probe_units"]) == 1
    assert n026_plan["probe_units"][0]["target_ids"] == ["mirror"]
    assert n026_plan["probe_units"][0]["acquisition_trigger"] == (
        "directed_usable_side"
    )

    n027 = validate_functional_discovery_response(
        {
            "inspected_object_ids": ["upright_piano", "piano_bench"],
            "directed_surface_targets": [],
            "functional_correspondences": [
                {
                    "target_ids": ["upright_piano", "piano_bench"],
                    "observation_goal": "show their ordinary joint-use relation",
                }
            ],
            "approach_clearance_targets": [],
            "boundary_sensitive_targets": [],
            "unusual_unconfirmed": [],
            "reason": "N027 acquisition coverage",
        },
        object_ids=("upright_piano", "piano_bench"),
        groups=[
            {
                "group_id": "music",
                "object_ids": ["upright_piano", "piano_bench"],
            }
        ],
    )
    n027_plan = build_functional_acquisition_plan(
        {key: list(value) if isinstance(value, tuple) else value
         for key, value in n027.items()},
        max_probe_units=4,
    )
    assert n027_plan["probe_units"] == []
    n027_checks = [
        item
        for item in n027_plan["functional_check_ledger"]["checks"]
        if item["owner_stage"] == "group_local"
    ]
    assert n027_checks
    assert all(
        item["acquisition_status"] == "judge_requested_on_demand"
        for item in n027_checks
    )

    n028 = validate_functional_discovery_response(
        {
            "inspected_object_ids": ["tool_cabinet"],
            "directed_surface_targets": [
                {
                    "target_id": "tool_cabinet",
                    "surface_roles": ["opening_side"],
                    "need_clearance": False,
                    "observation_goal": "show the opening side",
                }
            ],
            "functional_correspondences": [],
            "approach_clearance_targets": [],
            "boundary_sensitive_targets": [
                {
                    "target_id": "tool_cabinet",
                    "observation_goal": (
                        "show frontage with the logical boundary"
                    ),
                }
            ],
            "unusual_unconfirmed": [],
            "reason": "N028 acquisition coverage",
        },
        object_ids=("tool_cabinet",),
        groups=[
            {"group_id": "storage", "object_ids": ["tool_cabinet"]}
        ],
    )
    n028_plan = build_functional_acquisition_plan(
        {key: list(value) if isinstance(value, tuple) else value
         for key, value in n028.items()},
        max_probe_units=4,
    )
    assert n028_plan["probe_units"][0]["acquisition_trigger"] == (
        "directed_usable_side"
    )
    assert "boundary_sensitive_frontage" in n028_plan["probe_units"][0][
        "acquisition_triggers"
    ]
    assert "architecture_plane_visible" in n028_plan["probe_units"][0][
        "required_observations"
    ]


@pytest.mark.parametrize(
    ("status", "surfaces"),
    [
        (
            "identified",
            [
                {
                    "surface_role": "display_side",
                    "side_id": "local_pos_x",
                    "visual_cues": ["screen plane"],
                    "confidence": 0.9,
                }
            ],
        ),
        (
            "ambiguous",
            [
                {
                    "surface_role": "display_side",
                    "side_id": "local_pos_x",
                    "visual_cues": ["possible screen plane"],
                    "confidence": 0.55,
                },
                {
                    "surface_role": "display_side",
                    "side_id": "local_neg_x",
                    "visual_cues": ["opposite plane remains possible"],
                    "confidence": 0.45,
                },
            ],
        ),
        ("no_directed_surface", []),
    ],
)
def test_usable_surface_contract_statuses(
    status: str,
    surfaces: list[dict],
) -> None:
    result = validate_usable_surface_response(
        {
            "status": status,
            "surfaces": surfaces,
            "reason": "bounded surface observation",
        },
        allowed_surface_roles={"display_side"},
    )
    assert result["status"] == status


def test_usable_surface_rejects_untrusted_side_pose_and_verdict() -> None:
    base = {
        "status": "identified",
        "surfaces": [
            {
                "surface_role": "display_side",
                "side_id": "invented_side",
                "visual_cues": ["screen"],
                "confidence": 0.8,
            }
        ],
        "reason": "surface",
    }
    with pytest.raises(ValueError, match="trusted side"):
        validate_usable_surface_response(
            base,
            allowed_surface_roles={"display_side"},
        )
    for forbidden in ("verdict", "score", "pose", "scene_mutation"):
        value = deepcopy(base)
        value["surfaces"][0]["side_id"] = "local_pos_x"
        value[forbidden] = (
            {"objects": []}
            if forbidden == "scene_mutation"
            else 0.0
            if forbidden == "score"
            else "invalid"
        )
        with pytest.raises(ValueError, match="may not return"):
            validate_usable_surface_response(
                value,
                allowed_surface_roles={"display_side"},
            )


def test_partial_surface_bank_cannot_claim_no_directed_surface() -> None:
    with pytest.raises(ValueError, match="complete trusted side bank"):
        validate_usable_surface_response(
            {
                "status": "no_directed_surface",
                "surfaces": [],
                "reason": "no directed surface observed",
            },
            allowed_surface_roles={"display_side"},
            available_side_ids={"local_pos_x", "local_neg_x"},
            bank_complete=False,
        )
    result = validate_usable_surface_response(
        {
            "status": "insufficient_comparison",
            "surfaces": [],
            "reason": "the available subset is inconclusive",
        },
        allowed_surface_roles={"display_side"},
        available_side_ids={"local_pos_x"},
        bank_complete=False,
    )
    assert result["status"] == "insufficient_comparison"
    assert result["available_side_ids"] == ["local_pos_x"]


def test_placement_discovery_separates_subject_from_context() -> None:
    result = validate_placement_discovery_response(
        {
            "considered_object_ids": ["subject", "context"],
            "candidates": [
                {
                    "subject_id": "subject",
                    "context_ids": ["context"],
                    "observation_kind": "adjacency_context",
                    "observation_goal": (
                        "show the subject's location relative to context"
                    ),
                }
            ],
            "reason": "complete",
        },
        object_ids=("subject", "context"),
    )
    assert result["candidates"][0]["subject_id"] == "subject"
    assert result["candidates"][0]["context_ids"] == ["context"]
    assert result["candidates"][0]["observation_kind"] == (
        "contextual_anchor"
    )
    assert result["candidates"][0]["observation_goal"] == (
        "Observe the subject's non-operational positional relationship "
        "to the listed context objects without assuming plausibility."
    )
    assert result["candidates"][0]["source_observation_goal"] == (
        "show the subject's location relative to context"
    )
    assert result["observation_goal_policy"] == (
        "deterministic_neutral_routing_question_v1"
    )
    for forbidden in ("verdict", "pose", "scene_mutation"):
        value = {
            "considered_object_ids": ["subject", "context"],
            "candidates": [],
            "reason": "complete",
            forbidden: "invalid",
        }
        with pytest.raises(ValueError, match="may not return"):
            validate_placement_discovery_response(
                value,
                object_ids=("subject", "context"),
            )


def test_placement_camera_framing_uses_subject_and_context_without_reownership() -> None:
    captured: dict = {}

    def resolve(value, **kwargs):
        captured.update(kwargs)
        return ["/tmp/placement.png"], {
            "scope_satisfied": True,
            "provider_invoked": True,
            "provider_status": "available",
            "provider_reason": None,
            "provider_usage": {
                "acquired_artifact_paths": ["/tmp/placement.png"]
            },
        }

    packets = resolve_group_evidence_packets(
        None,
        metric_name="semantic_placement_consistency",
        policy={
            "camera_scope": "group_local",
            "image_budget": 2,
            "global_image_budget": 0,
            "scoped_image_budget": 1,
            "include_global_context": False,
        },
        scene={
            "objects": [
                {
                    "id": "subject",
                    "center": [1.0, 1.0, 0.5],
                    "size": [0.5, 0.5, 1.0],
                    "rotation": [0.0, 0.0, 0.0],
                },
                {
                    "id": "context",
                    "center": [3.0, 1.0, 0.5],
                    "size": [1.0, 1.0, 1.0],
                    "rotation": [0.0, 0.0, 0.0],
                },
            ]
        },
        prompt=None,
        groups=[{"group_id": "subject_group", "object_ids": ["subject"]}],
        grouping_report=None,
        camera_evidence_provider=object(),
        resolve_metric_evidence=resolve,
        camera_target_ids_by_group={
            "subject_group": ["subject", "context"]
        },
    )

    assert captured["selected_object_ids"] == ["subject", "context"]
    assert list(captured["group_scope"].member_ids) == [
        "subject",
        "context",
    ]
    assert list(packets[0]["group_scope"].member_ids) == ["subject"]
    assert packets[0]["camera_target_ids"] == ["subject", "context"]


def test_group_packet_resolution_uses_per_judge_episode_budget() -> None:
    calls: list[str] = []

    def resolve(value, **kwargs):
        group_id = kwargs["selected_group_ids"][0]
        calls.append(group_id)
        path = f"/tmp/{group_id}.png"
        artifacts = [
            path,
            f"/tmp/{group_id}_highlight_1.png",
            f"/tmp/{group_id}_rgb_2.png",
            f"/tmp/{group_id}_highlight_2.png",
            f"/tmp/{group_id}_global_highlight.png",
        ]
        return ["/tmp/global.png", path], {
            "scope_satisfied": True,
            "provider_invoked": True,
            "provider_status": "available",
            "provider_reason": None,
            "provider_usage": {
                "acquired_artifact_paths": artifacts
            },
        }

    existing_paths = [
        "/tmp/global.png",
        "/tmp/probe_1.png",
        "/tmp/probe_2.png",
        "/tmp/probe_3.png",
        "/tmp/probe_4.png",
    ]
    ledger = extend_acquisition_ledger(
        None,
        artifact_ids=evidence_artifact_refs(existing_paths),
    )
    packets = resolve_group_evidence_packets(
        None,
        metric_name="functional_consistency",
        policy={
            "camera_scope": "group_local",
            "image_budget": 2,
            "global_image_budget": 1,
            "scoped_image_budget": 1,
            "include_global_context": True,
        },
        scene={
            "objects": [
                {
                    "id": "a",
                    "center": [1.0, 1.0, 0.5],
                    "size": [0.5, 0.5, 1.0],
                    "rotation": [0.0, 0.0, 0.0],
                },
                {
                    "id": "b",
                    "center": [3.0, 1.0, 0.5],
                    "size": [0.5, 0.5, 1.0],
                    "rotation": [0.0, 0.0, 0.0],
                },
            ]
        },
        prompt=None,
        groups=[
            {"group_id": "g1", "object_ids": ["a"]},
            {"group_id": "g2", "object_ids": ["b"]},
        ],
        grouping_report=None,
        camera_evidence_provider=object(),
        resolve_metric_evidence=resolve,
        initial_acquisition_ledger=ledger,
        max_total_images=6,
    )

    assert calls == ["g1", "g2"]
    assert packets[0]["camera_acquisition_ledger_before"][
        "total_images_acquired"
    ] == 0
    assert packets[0]["camera_acquisition_ledger_after"][
        "total_images_acquired"
    ] == 2
    assert packets[1]["camera_acquisition_ledger_before"][
        "total_images_acquired"
    ] == 0
    assert packets[1]["camera_acquisition_ledger_after"][
        "total_images_acquired"
    ] == 2
    assert packets[0]["metric_camera_acquisition_ledger_before"][
        "total_images_acquired"
    ] == 5
    assert packets[0]["metric_camera_acquisition_ledger_after"][
        "total_images_acquired"
    ] == 10
    assert packets[1]["metric_camera_acquisition_ledger_after"][
        "total_images_acquired"
    ] == 15
    assert all(
        packet["resolution"]["scope_satisfied"] is True
        and packet["resolution"]["provider_reason"] is None
        and packet["resolution"]["acquisition_budget"]["scope"]
        == "group_judge_episode"
        and packet["resolution"]["acquisition_budget"]["counting"]
        == "judge_packet_seed_then_controller_acquired_artifacts"
        and packet["resolution"]["acquisition_budget"][
            "initial_judge_evidence_count"
        ]
        == 2
        and packet["resolution"]["acquisition_budget"][
            "acquired_artifact_count"
        ]
        == 6
        for packet in packets
    )


def test_group_packet_budget_overrun_reports_actual_episode_reason() -> None:
    def resolve(value, **kwargs):
        group_id = kwargs["selected_group_ids"][0]
        paths = [
            "/tmp/global.png",
            *[
                f"/tmp/{group_id}_evidence_{index}.png"
                for index in range(6)
            ],
        ]
        return paths, {
            "scope_satisfied": True,
            "provider_invoked": True,
            "provider_status": "available",
            "provider_reason": None,
            "provider_usage": {
                "acquired_artifact_paths": paths
            },
        }

    packets = resolve_group_evidence_packets(
        None,
        metric_name="functional_consistency",
        policy={
            "camera_scope": "group_local",
            "image_budget": 2,
            "global_image_budget": 1,
            "scoped_image_budget": 1,
            "include_global_context": True,
        },
        scene={
            "objects": [
                {
                    "id": "a",
                    "center": [1.0, 1.0, 0.5],
                    "size": [0.5, 0.5, 1.0],
                    "rotation": [0.0, 0.0, 0.0],
                }
            ]
        },
        prompt=None,
        groups=[{"group_id": "g1", "object_ids": ["a"]}],
        grouping_report=None,
        camera_evidence_provider=object(),
        resolve_metric_evidence=resolve,
        max_total_images=6,
    )

    assert packets[0]["resolution"]["scope_satisfied"] is False
    assert packets[0]["resolution"]["provider_status"] == "available"
    assert packets[0]["resolution"]["provider_reason"] == (
        "group_judge_episode_evidence_budget_exceeded"
    )
    assert packets[0]["resolution"]["acquisition_budget"][
        "initial_judge_evidence_count"
    ] == 7


def test_discovery_prompts_are_concise_and_case_agnostic() -> None:
    # The approved affordance prompt is intentionally kept byte-for-byte
    # unchanged even though its explicit field-consistency rules put it just
    # above the general compact-prompt target.
    assert len(FUNCTIONAL_AFFORDANCE_SYSTEM_PROMPT) < 2300
    other_prompts = (
        FUNCTIONAL_RELATION_SYSTEM_PROMPT,
        PLACEMENT_DISCOVERY_SYSTEM_PROMPT,
        USABLE_SURFACE_SYSTEM_PROMPT,
    )
    assert all(len(prompt) < 2200 for prompt in other_prompts)
    joined = "\n".join(
        (FUNCTIONAL_AFFORDANCE_SYSTEM_PROMPT, *other_prompts)
    ).lower()
    for case_specific_token in (
        "n021",
        "n022",
        "sofa",
        "television",
        "piano",
        "toilet",
        "washing machine",
    ):
        assert case_specific_token not in joined
    assert "distance and group membership" in (
        FUNCTIONAL_RELATION_SYSTEM_PROMPT.lower()
    )
    assert "not a candidate whitelist" in (
        FUNCTIONAL_RELATION_SYSTEM_PROMPT.lower()
    )
    assert "copy every object id exactly once" in (
        FUNCTIONAL_AFFORDANCE_SYSTEM_PROMPT.lower()
    )


def test_relation_audit_accepts_two_predicates_for_one_object_set() -> None:
    validated = validate_functional_relation_response(
        {
            "considered_object_ids": ["a", "b"],
            "relations": [
                {
                    "target_ids": ["a", "b"],
                    "predicate": "directional_correspondence",
                    "observation_goal": "show functional-side directions",
                },
                {
                    "target_ids": ["b", "a"],
                    "predicate": "relative_use_geometry",
                    "observation_goal": "show relative use geometry",
                },
            ],
            "reason": "complete",
        },
        object_ids=("a", "b"),
    )

    assert [item["predicate"] for item in validated["relations"]] == [
        "directional_correspondence",
        "relative_use_geometry",
    ]


def test_relation_audit_rejects_duplicate_atomic_check() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        validate_functional_relation_response(
            {
                "considered_object_ids": ["a", "b"],
                "relations": [
                    {
                        "target_ids": ["a", "b"],
                        "predicate": "directional_correspondence",
                        "observation_goal": "show functional-side directions",
                    },
                    {
                        "target_ids": ["b", "a"],
                        "predicate": "directional_correspondence",
                        "observation_goal": "show the same direction check",
                    },
                ],
                "reason": "complete",
            },
            object_ids=("a", "b"),
        )


def test_relation_audit_rejects_non_atomic_multi_object_target_set() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        validate_functional_relation_response(
            {
                "considered_object_ids": ["a", "b", "c"],
                "relations": [
                    {
                        "target_ids": ["a", "b", "c"],
                        "predicate": "directional_correspondence",
                        "observation_goal": "show the direct-use direction",
                    }
                ],
                "reason": "complete",
            },
            object_ids=("a", "b", "c"),
        )


def test_surface_cache_identity_ignores_instance_rotation() -> None:
    object_record = {
        "id": "tv",
        "jid": "asset_tv",
        "size": [1.0, 0.2, 0.7],
        "rotation": [0.0, 0.0, 0.0],
        "asset_ref": {"asset_key": "asset_tv"},
        "metadata": {
            "asset_metadata": {
                "catalog_hashes": {"mesh_sha256": "mesh-hash"}
            }
        },
    }
    rotated = deepcopy(object_record)
    rotated["rotation"] = [0.0, 0.0, 137.0]
    detector = {
        "implementation_id": "fake",
        "version": "v1",
        "configuration": {"threshold": 0.5},
        "model": "model-a",
    }
    kwargs = {
        "requested_surface_roles": ["display_side"],
        "trusted_preview_identity": {
            "policy": "trusted_side_bank",
            "renderer": "fake",
        },
        "detector_manifest": detector,
    }
    assert usable_surface_cache_identity(
        object_record,
        **kwargs,
    ) == (
        usable_surface_cache_identity(rotated, **kwargs)
    )
    resized = deepcopy(object_record)
    resized["size"] = [1.5, 0.2, 0.7]
    assert usable_surface_cache_identity(
        object_record,
        **kwargs,
    ) != (
        usable_surface_cache_identity(resized, **kwargs)
    )
    changed_detector = deepcopy(detector)
    changed_detector["version"] = "v2"
    assert usable_surface_cache_identity(
        object_record,
        **kwargs,
    ) != usable_surface_cache_identity(
        object_record,
        **{
            **kwargs,
            "detector_manifest": changed_detector,
        },
    )
    changed_config = deepcopy(detector)
    changed_config["configuration"]["threshold"] = 0.7
    assert usable_surface_cache_identity(
        object_record,
        **kwargs,
    ) != usable_surface_cache_identity(
        object_record,
        **{
            **kwargs,
            "detector_manifest": changed_config,
        },
    )
    changed_backend = deepcopy(detector)
    changed_backend["implementation_id"] = "other"
    assert usable_surface_cache_identity(
        object_record,
        **kwargs,
    ) != usable_surface_cache_identity(
        object_record,
        **{
            **kwargs,
            "detector_manifest": changed_backend,
        },
    )
    changed_model = deepcopy(detector)
    changed_model["model"] = "model-b"
    assert usable_surface_cache_identity(
        object_record,
        **kwargs,
    ) != usable_surface_cache_identity(
        object_record,
        **{
            **kwargs,
            "detector_manifest": changed_model,
        },
    )
    changed_roles = {
        **kwargs,
        "requested_surface_roles": ["control_side"],
    }
    assert usable_surface_cache_identity(
        object_record,
        **kwargs,
    ) != usable_surface_cache_identity(
        object_record,
        **changed_roles,
    )
    changed_preview = deepcopy(
        kwargs["trusted_preview_identity"]
    )
    changed_preview["renderer"] = "other"
    assert usable_surface_cache_identity(
        object_record,
        **kwargs,
    ) != usable_surface_cache_identity(
        object_record,
        **{
            **kwargs,
            "trusted_preview_identity": changed_preview,
        },
    )


def test_local_side_bank_rotates_with_object_and_informs_candidates() -> None:
    scene = {
        "scene_id": "scene",
        "boundary": [[0, 0], [8, 0], [8, 8], [0, 8]],
        "scene_height": 3.0,
        "objects": [
            {
                "id": "television",
                "category": "television",
                "center": [4.0, 4.0, 1.0],
                "size": [1.4, 0.3, 1.0],
                "rotation": [0.0, 0.0, 90.0],
            }
        ],
    }
    bank = generate_usable_surface_side_bank(
        scene,
        target_id="television",
    )
    assert [item["id"] for item in bank] == [
        "local_pos_x",
        "local_neg_x",
        "local_pos_y",
        "local_neg_y",
    ]
    pos_x = bank[0]
    assert pos_x["world_outward_axis"][:2] == pytest.approx([0.0, 1.0])

    request = {
        "metric": "functional_consistency",
        "scene": scene,
        "object_ids": ["television"],
        "_resolved_camera_pose_mode": "query_cov",
        "functional_probe": {
            "probe_id": "functional_probe_01",
            "kind": "functional_frontage",
            "target_ids": ["television"],
            "related_target_ids": [],
            "usable_surface_hypotheses": [
                {
                    "target_id": "television",
                    "status": "identified",
                    "surfaces": [
                        {
                            "surface_role": "display_side",
                            "side_id": "local_pos_x",
                            "confidence": 0.9,
                        }
                    ],
                }
            ],
        },
    }
    candidates = generate_camera_pose_candidates(
        request,
        max_candidates=4,
    )
    assert candidates[0]["usable_surface_informed"] is True
    assert candidates[0]["usable_surface_side_ids"] == ["local_pos_x"]
    assert any(
        item["intended_azimuth_degrees"] == pytest.approx(90.0)
        for item in candidates
    )


def test_local_side_bank_keeps_four_preview_only_sides_at_boundary() -> None:
    scene = {
        "scene_id": "scene",
        "boundary": [[0, 0], [8, 0], [8, 8], [0, 8]],
        "scene_height": 3.0,
        "objects": [
            {
                "id": "cabinet",
                "category": "cabinet",
                "center": [7.9, 4.0, 1.0],
                "size": [0.2, 0.5, 1.8],
                "rotation": [0.0, 0.0, 0.0],
            }
        ],
    }

    bank = generate_usable_surface_side_bank(
        scene,
        target_id="cabinet",
    )

    assert [item["id"] for item in bank] == [
        "local_pos_x",
        "local_neg_x",
        "local_pos_y",
        "local_neg_y",
    ]
    assert all(item["feasibility"]["preview_only"] is True for item in bank)
    assert any(
        item["feasibility"]["room_feasible"] is False for item in bank
    )


def test_local_side_bank_skips_only_nonhorizontal_intrinsic_sides() -> None:
    scene = {
        "scene_id": "scene",
        "boundary": [[0, 0], [8, 0], [8, 8], [0, 8]],
        "scene_height": 3.0,
        "objects": [
            {
                "id": "tilted",
                "category": "fixture",
                "center": [4.0, 4.0, 1.0],
                "size": [1.0, 1.0, 1.0],
                "rotation": [0.0, 90.0, 0.0],
            }
        ],
    }

    bank = generate_usable_surface_side_bank(
        scene,
        target_id="tilted",
    )

    assert [item["id"] for item in bank] == [
        "local_pos_y",
        "local_neg_y",
    ]


def test_ambiguous_surfaces_get_complementary_exact_candidates_and_none_falls_back() -> None:
    scene = {
        "scene_id": "scene",
        "boundary": [[0, 0], [8, 0], [8, 8], [0, 8]],
        "scene_height": 3.0,
        "objects": [
            {
                "id": "cabinet",
                "category": "cabinet",
                "center": [4.0, 4.0, 1.0],
                "size": [1.4, 0.5, 1.8],
                "rotation": [0.0, 0.0, 0.0],
            },
            {
                "id": "workbench",
                "category": "workbench",
                "center": [6.0, 4.0, 0.6],
                "size": [1.2, 0.7, 1.2],
                "rotation": [0.0, 0.0, 0.0],
            }
        ],
    }
    request = {
        "metric": "functional_consistency",
        "scene": scene,
        "object_ids": ["cabinet"],
        "_resolved_camera_pose_mode": "query_cov",
        "functional_probe": {
            "probe_id": "functional_probe_01",
            "kind": "functional_frontage",
            "target_ids": ["cabinet"],
            "related_target_ids": [],
            "group_id": "work_group",
            "group_member_ids": ["cabinet", "workbench"],
            "usable_surface_hypotheses": [
                {
                    "target_id": "cabinet",
                    "status": "ambiguous",
                    "surfaces": [
                        {
                            "surface_role": "opening_side",
                            "side_id": "local_pos_x",
                            "confidence": 0.5,
                        },
                        {
                            "surface_role": "opening_side",
                            "side_id": "local_pos_y",
                            "confidence": 0.5,
                        },
                    ],
                }
            ],
        },
    }
    candidates = generate_camera_pose_candidates(request, max_candidates=4)
    exact_azimuths = {
        round(float(item["intended_azimuth_degrees"]), 6)
        for item in candidates
    }
    assert {0.0, 90.0} <= exact_azimuths
    assert all(item["usable_surface_informed"] is True for item in candidates)
    assert all(
        item["functional_group_member_ids"]
        == ["cabinet", "workbench"]
        for item in candidates
    )
    assert all(
        float(item["functional_group_context_bounds"][1][0]) >= 6.6
        for item in candidates
    )
    assert all(
        float(item["functional_specific_target_bounds"][1][0]) < 5.0
        for item in candidates
    )

    request["functional_probe"]["usable_surface_hypotheses"] = [
        {
            "target_id": "cabinet",
            "status": "no_directed_surface",
            "surfaces": [],
        }
    ]
    fallback = generate_camera_pose_candidates(request, max_candidates=4)
    assert all(
        item["usable_surface_informed"] is False for item in fallback
    )
    assert all(item["usable_surface_side_ids"] == [] for item in fallback)


class _Model:
    model_id = "model"
    endpoint = "https://example.test/v1"
    response_format_json = True

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.last_request_metadata: dict = {}

    def chat_messages(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return json.dumps(self.responses.pop(0))


def test_openai_transport_separates_discovery_and_surface_calls(
    tmp_path: Path,
) -> None:
    global_image = tmp_path / "global.png"
    Image.new("RGB", (16, 16), (100, 110, 120)).save(global_image)
    previews = []
    for side_id in (
        "local_pos_x",
        "local_neg_x",
        "local_pos_y",
        "local_neg_y",
    ):
        path = tmp_path / f"{side_id}.png"
        Image.new("RGB", (16, 16), (80, 90, 100)).save(path)
        previews.append({"side_id": side_id, "image_path": str(path)})
    model = _Model(
        [
            {
                "objects": [
                    {
                        "object_id": "television",
                        "directionality": "directed",
                        "surface_roles": ["display_side"],
                        "need_clearance": False,
                        "boundary_review_state": "routine",
                        "review_state": "routine",
                        "observation_goal": "show the display side",
                        "boundary_observation_goal": (
                            "show ordinary separation from the boundary"
                        ),
                    }
                ],
                "reason": "complete",
            },
            {
                "considered_object_ids": ["television"],
                "relations": [],
                "reason": "no direct joint-use relation",
            },
            {
                "status": "identified",
                "surfaces": [
                    {
                        "surface_role": "display_side",
                        "side_id": "local_pos_y",
                        "visual_cues": ["screen plane"],
                        "confidence": 0.92,
                    }
                ],
                "reason": "screen plane is visible",
            },
        ]
    )
    selector = OpenAICompatibleCameraSelector(model)
    discovery = selector.discover_functional_evidence(
        {
            "metric": "functional_consistency",
            "scene_id": "scene",
            "scene_type": "living_room",
            "global_image_path": str(global_image),
            "objects": [
                {"id": "television", "category": "television"}
            ],
            "groups": [
                {
                    "group_id": "group_001",
                    "object_ids": ["television"],
                }
            ],
        }
    )
    decoded = selector.decode_usable_surface(
        {
            "scene_id": "scene",
            "target_id": "television",
            "target_category": "television",
            "surface_roles": ["display_side"],
            "previews": previews,
        }
    )

    assert discovery["object_coverage"][0]["object_id"] == "television"
    assert discovery["object_coverage"][0]["inspected"] is True
    assert discovery["object_coverage"][0]["directionality"] == "directed"
    assert discovery["boundary_sensitive_targets"] == []
    assert discovery["provenance"]["normalization_warnings"] == [
        {
            "code": "routine_boundary_goal_cleared",
            "object_id": "television",
            "field": "boundary_observation_goal",
            "repair": "cleared",
            "reason": "boundary_review_state_is_routine",
        }
    ]
    assert decoded["surfaces"][0]["side_id"] == "local_pos_y"
    assert [call["kwargs"]["call_type"] for call in model.calls] == [
        "vlm_camera_pose.functional_discovery.affordance",
        "vlm_camera_pose.functional_discovery.relations",
        "vlm_camera_pose.usable_surface_decode",
    ]
    affordance_text = model.calls[0]["messages"][1]["content"][0]["text"]
    relation_text = model.calls[1]["messages"][1]["content"][0]["text"]
    surface_text = model.calls[2]["messages"][1]["content"][0]["text"]
    assert (
        '"vlm_role":"functional_affordance_discovery"'
        in affordance_text
    )
    assert (
        '"decision_contract":"functional_affordance_ledger_v4"'
        in affordance_text
    )
    assert (
        '"vlm_role":"functional_relation_discovery"' in relation_text
    )
    assert (
        '"decision_contract":"functional_relation_audit_v3"'
        in relation_text
    )
    assert (
        '"policy":"soft_non_exhaustive_positive_prior"'
        in relation_text
    )
    assert (
        '"directed_surface_roles":{"television":["display_side"]}'
        in relation_text
    )
    assert (
        '"trusted_group_partition":[{"group_id":"group_001",'
        '"object_ids":["television"]}]'
        in relation_text
    )
    assert '"vlm_role":"usable_surface_decoder"' in surface_text
    assert (
        '"decision_contract":"usable_surface_decode_v2"' in surface_text
    )
    assert discovery["provenance"]["calls"]["affordance"][
        "vlm_role"
    ] == "functional_affordance_discovery"
    assert discovery["provenance"]["calls"]["affordance"][
        "decision_contract"
    ] == "functional_affordance_ledger_v4"
    assert discovery["provenance"]["calls"]["relations"][
        "vlm_role"
    ] == "functional_relation_discovery"
    relation_input = discovery["provenance"][
        "relation_input_contract"
    ]
    assert relation_input["visual_evidence_roles"] == ["scene_global"]
    assert relation_input["structured_context"][
        "trusted_group_partition"
    ] == [
        {
            "group_id": "group_001",
            "object_ids": ["television"],
        }
    ]
    assert relation_input["excluded_object_fields"] == [
        "center",
        "size",
        "rotation",
        "description",
    ]
    assert decoded["provenance"]["vlm_role"] == "usable_surface_decoder"
    assert (
        decoded["provenance"]["decision_contract"]
        == "usable_surface_decode_v2"
    )


def test_placement_discovery_records_explicit_vlm_role_and_contract(
    tmp_path: Path,
) -> None:
    global_image = tmp_path / "global.png"
    Image.new("RGB", (16, 16), (100, 110, 120)).save(global_image)
    model = _Model(
        [
            {
                "considered_object_ids": ["telephone"],
                "candidates": [
                    {
                        "subject_id": "telephone",
                        "context_ids": [],
                        "observation_kind": "scene_zone",
                        "observation_goal": (
                            "show the telephone's surrounding scene zone"
                        ),
                    }
                ],
                "reason": "complete",
            }
        ]
    )

    result = discover_openai_compatible_placement_evidence(
        model=model,
        request={
            "metric": "semantic_placement_consistency",
            "scene_id": "scene",
            "scene_type": "living_room",
            "global_image_path": str(global_image),
            "objects": [
                {"id": "telephone", "category": "telephone"}
            ],
        },
    )

    text = model.calls[0]["messages"][1]["content"][0]["text"]
    assert '"vlm_role":"placement_discovery"' in text
    assert '"decision_contract":"placement_discovery_v2"' in text
    assert result["provenance"]["vlm_role"] == "placement_discovery"
    assert (
        result["provenance"]["decision_contract"]
        == "placement_discovery_v2"
    )


def test_discovery_identity_overlay_is_additive_and_exactly_grounded(
    tmp_path: Path,
) -> None:
    global_image = tmp_path / "global.png"
    identity_image = tmp_path / "identity.png"
    Image.new("RGB", (16, 16), (100, 110, 120)).save(global_image)
    Image.new("RGB", (16, 16), (255, 0, 0)).save(identity_image)
    model = _Model(
        [
            {
                "objects": [
                    {
                        "object_id": "chair_a",
                        "directionality": "non_directed",
                        "surface_roles": [],
                        "need_clearance": False,
                        "boundary_review_state": "routine",
                        "review_state": "routine",
                        "observation_goal": "inspect ordinary use",
                        "boundary_observation_goal": "",
                    },
                    {
                        "object_id": "chair_b",
                        "directionality": "non_directed",
                        "surface_roles": [],
                        "need_clearance": False,
                        "boundary_review_state": "routine",
                        "review_state": "routine",
                        "observation_goal": "inspect ordinary use",
                        "boundary_observation_goal": "",
                    },
                ],
                "reason": "complete",
            },
            {
                "considered_object_ids": ["chair_a", "chair_b"],
                "relations": [],
                "reason": "complete",
            },
        ]
    )
    selector = OpenAICompatibleCameraSelector(model)
    selector.discover_functional_evidence(
        {
            "metric": "functional_consistency",
            "scene_id": "scene",
            "scene_type": "living_room",
            "global_image_path": str(global_image),
            "identity_image_path": str(identity_image),
            "identity_legend": {
                "red": "chair_a",
                "blue": "chair_b",
            },
            "objects": [
                {"id": "chair_a", "category": "chair"},
                {"id": "chair_b", "category": "chair"},
            ],
            "groups": [
                {
                    "group_id": "group_001",
                    "object_ids": ["chair_a", "chair_b"],
                }
            ],
        }
    )

    for call in model.calls:
        content = call["messages"][1]["content"]
        assert len(
            [item for item in content if item["type"] == "image_url"]
        ) == 2
        assert '"red":"chair_a"' in content[0]["text"]
        assert '"blue":"chair_b"' in content[0]["text"]


def test_discovery_identity_legend_must_cover_every_input_object(
    tmp_path: Path,
) -> None:
    global_image = tmp_path / "global.png"
    identity_image = tmp_path / "identity.png"
    Image.new("RGB", (16, 16), (100, 110, 120)).save(global_image)
    Image.new("RGB", (16, 16), (255, 0, 0)).save(identity_image)
    selector = OpenAICompatibleCameraSelector(_Model([]))

    with pytest.raises(ValueError, match="cover exactly"):
        selector.discover_functional_evidence(
            {
                "metric": "functional_consistency",
                "global_image_path": str(global_image),
                "identity_image_path": str(identity_image),
                "identity_legend": {"red": "chair_a"},
                "objects": [
                    {"id": "chair_a", "category": "chair"},
                    {"id": "chair_b", "category": "chair"},
                ],
                "groups": [
                    {
                        "group_id": "group_001",
                        "object_ids": ["chair_a", "chair_b"],
                    }
                ],
            }
        )


def test_placement_discovery_consumes_trusted_identity_overlay(
    tmp_path: Path,
) -> None:
    global_image = tmp_path / "global.png"
    identity_image = tmp_path / "identity.png"
    Image.new("RGB", (16, 16), (100, 110, 120)).save(global_image)
    Image.new("RGB", (16, 16), (255, 0, 0)).save(identity_image)
    model = _Model(
        [
            {
                "considered_object_ids": ["telephone_a", "telephone_b"],
                "candidates": [],
                "reason": "complete",
            }
        ]
    )

    discover_openai_compatible_placement_evidence(
        model=model,
        request={
            "metric": "semantic_placement_consistency",
            "scene_id": "scene",
            "global_image_path": str(global_image),
            "identity_image_path": str(identity_image),
            "identity_legend": {
                "red": "telephone_a",
                "blue": "telephone_b",
            },
            "objects": [
                {"id": "telephone_a", "category": "telephone"},
                {"id": "telephone_b", "category": "telephone"},
            ],
        },
    )

    content = model.calls[0]["messages"][1]["content"]
    assert len(
        [item for item in content if item["type"] == "image_url"]
    ) == 2
    assert '"red":"telephone_a"' in content[0]["text"]


def test_affordance_contract_conflict_gets_one_same_evidence_repair(
    tmp_path: Path,
) -> None:
    global_image = tmp_path / "global.png"
    Image.new("RGB", (16, 16), (100, 110, 120)).save(global_image)
    model = _Model(
        [
            {
                "objects": [
                    {
                        "object_id": "cabinet",
                        "directionality": "non_directed",
                        "surface_roles": ["opening_side"],
                        "need_clearance": True,
                        "boundary_review_state": "routine",
                        "review_state": "routine",
                        "observation_goal": "show ordinary cabinet use",
                        "boundary_observation_goal": "",
                    }
                ],
                "reason": "initial contradictory ledger",
            },
            {
                "objects": [
                    {
                        "object_id": "cabinet",
                        "directionality": "directed",
                        "surface_roles": ["opening_side"],
                        "need_clearance": True,
                        "boundary_review_state": "routine",
                        "review_state": "routine",
                        "observation_goal": "show ordinary cabinet use",
                        "boundary_observation_goal": "",
                    }
                ],
                "reason": "contract repaired from the same image",
            },
            {
                "considered_object_ids": ["cabinet"],
                "relations": [],
                "reason": "no direct joint-use relation",
            },
        ]
    )

    discovery = OpenAICompatibleCameraSelector(
        model
    ).discover_functional_evidence(
        {
            "metric": "functional_consistency",
            "scene_id": "scene",
            "scene_type": "living_room",
            "global_image_path": str(global_image),
            "objects": [{"id": "cabinet", "category": "cabinet"}],
            "groups": [
                {"group_id": "group_001", "object_ids": ["cabinet"]}
            ],
        }
    )

    assert discovery["directed_surface_targets"][0][
        "directionality"
    ] == "directed"
    schema_audit = discovery["provenance"]["calls"]["affordance"][
        "schema_validation"
    ]
    assert schema_audit["attempt_count"] == 2
    assert schema_audit["repair_retry_count"] == 1
    assert schema_audit["recovered"] is True
    assert schema_audit["attempts"][0]["validation_error"] == (
        "non_directed affordances require empty surface_roles; "
        "object_id=cabinet"
    )
    assert [call["kwargs"]["call_type"] for call in model.calls] == [
        "vlm_camera_pose.functional_discovery.affordance",
        "vlm_camera_pose.functional_discovery.affordance.schema_repair",
        "vlm_camera_pose.functional_discovery.relations",
    ]
    relation_text = model.calls[2]["messages"][1]["content"][0]["text"]
    assert (
        '"directed_surface_roles":{"cabinet":["opening_side"]}'
        in relation_text
    )


def test_affordance_contract_repair_fails_closed_after_one_retry(
    tmp_path: Path,
) -> None:
    global_image = tmp_path / "global.png"
    Image.new("RGB", (16, 16), (100, 110, 120)).save(global_image)
    invalid = {
        "objects": [
            {
                "object_id": "cabinet",
                "directionality": "non_directed",
                "surface_roles": ["opening_side"],
                "need_clearance": True,
                "boundary_review_state": "routine",
                "review_state": "routine",
                "observation_goal": "show ordinary cabinet use",
                "boundary_observation_goal": "",
            }
        ],
        "reason": "still contradictory",
    }
    model = _Model(
        [
            deepcopy(invalid),
            deepcopy(invalid),
        ]
    )

    with pytest.raises(ResponseSchemaRepairError) as exc_info:
        OpenAICompatibleCameraSelector(
            model
        ).discover_functional_evidence(
            {
                "metric": "functional_consistency",
                "scene_id": "scene",
                "scene_type": "living_room",
                "global_image_path": str(global_image),
                "objects": [
                    {"id": "cabinet", "category": "cabinet"}
                ],
                "groups": [
                    {
                        "group_id": "group_001",
                        "object_ids": ["cabinet"],
                    }
                ],
            }
        )

    assert exc_info.value.schema_audit["attempt_count"] == 2
    assert exc_info.value.schema_audit["recovered"] is False
    assert [call["kwargs"]["call_type"] for call in model.calls] == [
        "vlm_camera_pose.functional_discovery.affordance",
        "vlm_camera_pose.functional_discovery.affordance.schema_repair",
    ]


def test_relation_discovery_uses_sparse_positive_non_filtering_prior(
    tmp_path: Path,
) -> None:
    global_image = tmp_path / "global.png"
    Image.new("RGB", (16, 16), (100, 110, 120)).save(global_image)
    model = _Model(
        [
            {
                "objects": [
                    {
                        "object_id": "table",
                        "directionality": "non_directed",
                        "surface_roles": [],
                        "need_clearance": False,
                        "boundary_review_state": "routine",
                        "review_state": "routine",
                        "observation_goal": "inspect ordinary table use",
                        "boundary_observation_goal": "",
                    },
                    {
                        "object_id": "chair",
                        "directionality": "directed",
                        "surface_roles": ["seating_side"],
                        "need_clearance": True,
                        "boundary_review_state": "routine",
                        "review_state": "routine",
                        "observation_goal": "inspect the seating side",
                        "boundary_observation_goal": "",
                    },
                ],
                "reason": "complete affordance inventory",
            },
            {
                "considered_object_ids": ["table", "chair"],
                "relations": [
                    {
                        "target_ids": ["table", "chair"],
                        "predicate": "relative_use_geometry",
                        "observation_goal": (
                            "show ordinary seated access to the table"
                        ),
                    }
                ],
                "reason": (
                    "the non-directed table remains a relation candidate"
                ),
            },
        ]
    )

    discovery = OpenAICompatibleCameraSelector(
        model
    ).discover_functional_evidence(
        {
            "metric": "functional_consistency",
            "scene_id": "scene",
            "scene_type": "dining_room",
            "global_image_path": str(global_image),
            "objects": [
                {"id": "table", "category": "table"},
                {"id": "chair", "category": "chair"},
            ],
            "groups": [
                {
                    "group_id": "group_001",
                    "object_ids": ["table", "chair"],
                }
            ],
        }
    )

    assert [call["kwargs"]["call_type"] for call in model.calls] == [
        "vlm_camera_pose.functional_discovery.affordance",
        "vlm_camera_pose.functional_discovery.relations",
    ]
    relation_context = json.loads(
        model.calls[1]["messages"][1]["content"][0]["text"].split(
            "\n",
            1,
        )[1]
    )
    assert relation_context["affordance_prior"] == {
        "policy": "soft_non_exhaustive_positive_prior",
        "source_status": "validated",
        "coverage_requirement": (
            "objects omitted from this prior remain full relation candidates"
        ),
        "directed_surface_roles": {
            "chair": ["seating_side"],
        },
        "need_clearance_object_ids": ["chair"],
    }
    assert "table" not in json.dumps(
        relation_context["affordance_prior"]
    )
    assert discovery["within_group_correspondences"][0]["target_ids"] == [
        "table",
        "chair",
    ]
    assert discovery["provenance"]["discovery_order"] == [
        "affordance",
        "relations",
    ]
    assert discovery["provenance"]["relation_affordance_prior"] == {
        "policy": "soft_non_exhaustive_positive_prior",
        "source_status": "validated",
        "input_object_count": 2,
        "hint_object_count": 1,
    }


class _Renderer:
    preview_width = 128
    preview_height = 96

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def render_camera_views(
        self,
        *,
        blend_file,
        out_dir,
        camera_views,
        preview=False,
    ):
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self.calls.append(
            {
                "preview": preview,
                "ids": [str(item["id"]) for item in camera_views],
            }
        )
        views = []
        for index, pose in enumerate(camera_views):
            path = destination / f"view_{index:02d}.png"
            Image.new("RGB", (32, 24), (100, 110, 120)).save(path)
            views.append(
                {
                    "id": str(pose["id"]),
                    "path": str(path),
                    "pose": deepcopy(pose),
                }
            )
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
                "ids": [str(item["id"]) for item in camera_views],
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
                    "id": str(pose["id"]),
                    "path": str(path),
                    "pose": deepcopy(pose),
                }
            )
        return {"views": views}


class _Selector:
    max_images = 8

    def __init__(self) -> None:
        self.surface_calls: list[dict] = []
        self.selection_calls: list[dict] = []

    def decode_usable_surface(self, request: dict) -> dict:
        self.surface_calls.append(request)
        return {
            "schema_version": "usable_surface_decode_v1",
            "target_id": request["target_id"],
            "status": "identified",
            "surfaces": [
                {
                    "surface_role": request["surface_roles"][0],
                    "side_id": "local_pos_x",
                    "visual_cues": ["usable plane"],
                    "confidence": 0.9,
                }
            ],
            "reason": "usable plane",
            "provenance": {"prompt_version": "test"},
        }

    def select_camera_views(self, request: dict) -> dict:
        self.selection_calls.append(request)
        return {
            "selected_view_ids": [request["candidates"][0]["id"]],
            "action": None,
            "reason": "first surface-informed candidate",
        }


def _single_cabinet_scene() -> dict:
    return {
        "scene_id": "scene",
        "boundary": [[0, 0], [8, 0], [8, 8], [0, 8]],
        "scene_height": 3.0,
        "objects": [
            {
                "id": "cabinet",
                "jid": "asset_cabinet",
                "category": "cabinet",
                "center": [4.0, 4.0, 1.0],
                "size": [1.0, 0.5, 2.0],
                "rotation": [0.0, 0.0, 0.0],
                "asset_ref": {"asset_key": "asset_cabinet"},
                "metadata": {
                    "asset_metadata": {
                        "catalog_hashes": {
                            "mesh_sha256": "mesh-cabinet"
                        }
                    }
                },
            }
        ],
    }


def test_usable_surface_repair_bank_preserves_trusted_side_ids() -> None:
    scene = _single_cabinet_scene()

    primary = generate_usable_surface_side_bank(
        scene,
        target_id="cabinet",
    )
    repair = generate_usable_surface_side_repair_bank(
        scene,
        target_id="cabinet",
    )

    assert [item["id"] for item in repair] == [
        item["id"] for item in primary
    ]
    assert {
        item["policy_source"] for item in repair
    } == {"usable_surface_elevated_detail_repair_v1"}
    assert all(
        item["elevation_degrees"] > primary[index]["elevation_degrees"]
        for index, item in enumerate(repair)
    )


@pytest.mark.parametrize(
    ("second_status", "expected_resolution"),
    [
        ("identified", "resolved"),
        ("ambiguous", "pending"),
    ],
)
def test_usable_surface_pending_runs_one_bounded_evidence_repair(
    tmp_path: Path,
    second_status: str,
    expected_resolution: str,
) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")

    class PendingSelector(_Selector):
        def decode_usable_surface(self, request: dict) -> dict:
            self.surface_calls.append(deepcopy(request))
            status = (
                "ambiguous"
                if len(self.surface_calls) == 1
                else second_status
            )
            side_ids = (
                ["local_pos_x", "local_neg_y"]
                if status == "ambiguous"
                else ["local_neg_y"]
            )
            return {
                "schema_version": "usable_surface_decode_v2",
                "target_id": request["target_id"],
                "status": status,
                "surfaces": [
                    {
                        "surface_role": request["surface_roles"][0],
                        "side_id": side_id,
                        "visual_cues": ["visible opening geometry"],
                        "confidence": 0.6,
                    }
                    for side_id in side_ids
                ],
                "reason": "bounded comparison test",
                "provenance": {"prompt_version": "test"},
            }

    selector = PendingSelector()
    provider = CameraEvidenceProvider(
        renderer=_Renderer(),
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        mode="query_cov",
        selector=selector,
        max_views=1,
        max_steps=0,
        candidate_count=4,
        usable_surface_cache_dir=tmp_path / "surface_cache",
    )
    provider(
        {
            "metric": "functional_consistency",
            "scene": _single_cabinet_scene(),
            "object_ids": ["cabinet"],
            "evidence_scope": "object_local",
            "evidence_policy": {
                "camera_pose_mode": "query_cov",
                "scoped_image_budget": 1,
            },
            "functional_probe": {
                "probe_id": "functional_probe_01",
                "kind": "functional_frontage",
                "target_ids": ["cabinet"],
                "related_target_ids": [],
                "required_observations": [
                    "interaction_side_visible",
                    "approach_zone_visible",
                ],
                "surface_targets": [
                    {
                        "target_id": "cabinet",
                        "surface_roles": ["opening_side"],
                    }
                ],
                "view_goal": "show the opening side and its free-space region",
            },
        }
    )

    surface_audit = provider.last_call_usage["usable_surface"]
    assert len(selector.surface_calls) == 2
    assert surface_audit["decoder_calls"] == 2
    assert surface_audit["repair_rounds"] == 1
    assert surface_audit["preview_render_count"] == 16
    assert surface_audit["resolution_status"] == expected_resolution
    assert surface_audit["banks"][1]["evidence_round"] == 2
    assert {
        item["pose"]["policy_source"]
        for item in selector.surface_calls[1]["previews"]
    } == {"usable_surface_elevated_detail_repair_v1"}
    lifecycle = surface_audit["lifecycles"][0]
    assert [item["status"] for item in lifecycle["rounds"]] == [
        "ambiguous",
        second_status,
    ]
    assert lifecycle["state"] == expected_resolution
    assert surface_audit["pending_after_budget"] == (
        1 if expected_resolution == "pending" else 0
    )


def test_provider_decodes_surface_then_selects_and_renders(
    tmp_path: Path,
) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")
    renderer = _Renderer()
    selector = _Selector()
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        mode="query_cov",
        selector=selector,
        max_views=1,
        max_steps=0,
        candidate_count=4,
        usable_surface_cache_dir=tmp_path / "surface_cache",
    )
    assert isinstance(
        provider.usable_surface_detector,
        VLMTrustedSideUsableSurfaceDetector,
    )
    assert provider.policy_config["usable_surface"]["backend"] == (
        DEFAULT_USABLE_SURFACE_DETECTOR_BACKEND
    )
    assert provider.max_full_artifacts_for_controller_request(
        {
            "metric": "functional_consistency",
            "context": {},
        }
    ) == 3
    assert provider.max_full_artifacts_for_controller_request(
        {
            "metric": "functional_consistency",
            "context": {
                "camera_evidence_request": {
                    "functional_probe": {
                        "probe_id": "functional_probe_01",
                    },
                    "evidence_policy": {
                        "scoped_image_budget": 1,
                    },
                }
            },
        }
    ) == 2
    scene = {
        "scene_id": "scene",
        "boundary": [[0, 0], [8, 0], [8, 8], [0, 8]],
        "scene_height": 3.0,
        "objects": [
            {
                "id": "television",
                "jid": "asset_tv",
                "category": "television",
                "center": [4.0, 4.0, 1.0],
                "size": [1.4, 0.3, 1.0],
                "rotation": [0.0, 0.0, 90.0],
                "asset_ref": {"asset_key": "asset_tv"},
                "metadata": {
                    "asset_metadata": {
                        "catalog_hashes": {
                            "mesh_sha256": "mesh-tv"
                        }
                    }
                },
            }
        ],
    }
    request = {
        "metric": "functional_consistency",
        "scene": scene,
        "object_ids": ["television"],
        "evidence_scope": "object_local",
        "evidence_policy": {
            "camera_pose_mode": "query_cov",
            "scoped_image_budget": 1,
        },
        "functional_probe": {
            "probe_id": "functional_probe_01",
            "kind": "functional_frontage",
            "target_ids": ["television"],
            "related_target_ids": [],
            "required_observations": [
                "interaction_side_visible",
            ],
            "surface_targets": [
                {
                    "target_id": "television",
                    "surface_roles": ["display_side"],
                }
            ],
            "view_goal": "show the display side",
        },
    }
    result = provider(request)

    assert len(result) == 1
    assert len(selector.surface_calls) == 1
    assert len(selector.selection_calls) == 1
    assert selector.selection_calls[0]["functional_probe"][
        "usable_surface_hypotheses"
    ][0]["surfaces"][0]["side_id"] == "local_pos_x"
    usage = provider.last_call_usage
    assert usage["usable_surface"]["decoder_calls"] == 1
    assert usage["usable_surface"]["preview_render_count"] == 8
    assert usage["preview_render_count"] == 12
    assert usage["final_render_count"] == 2
    assert len(usage["acquired_artifact_paths"]) == 2
    ledger = extend_acquisition_ledger(
        None,
        artifact_ids=evidence_artifact_refs(
            usage["acquired_artifact_paths"]
        ),
    )
    assert ledger["total_images_acquired"] == 2

    assert provider(request) == result
    assert provider.last_call_usage["cache_hit"] is True
    assert len(
        provider.last_call_usage["acquired_artifact_paths"]
    ) == 2

    identity_path = Path(
        next(
            path
            for path in usage["acquired_artifact_paths"]
            if path != result[0]["path"]
        )
    )
    identity_path.unlink()
    provider(request)
    assert provider.last_call_usage["cache_hit"] is False
    assert identity_path.is_file()


def test_fake_surface_detector_drives_functional_probe_pipeline(
    tmp_path: Path,
) -> None:
    global_image = tmp_path / "global.png"
    Image.new("RGB", (32, 24), (90, 100, 110)).save(global_image)
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")

    class FakeDetector:
        implementation_id = "fake_catalog_affordance"
        version = "fake_v1"

        def __init__(self) -> None:
            self.calls: list[dict] = []

        def manifest(self) -> dict:
            return {
                "implementation_id": self.implementation_id,
                "version": self.version,
                "configuration": {"source": "test"},
                "model": None,
                "decision_authority": "none",
            }

        def detect(self, request: dict) -> dict:
            self.calls.append(deepcopy(request))
            return {
                "schema_version": "usable_surface_decode_v2",
                "target_id": request["target_id"],
                "status": "identified",
                "surfaces": [
                    {
                        "surface_role": request["surface_roles"][0],
                        "side_id": "local_pos_x",
                        "visual_cues": ["catalog-authored opening side"],
                        "confidence": 1.0,
                    }
                ],
                "reason": "test detector",
                "detector_implementation_id": self.implementation_id,
                "detector_version": self.version,
                "decision_authority": "none",
                "provenance": {"source": "fake"},
            }

    class SelectionOnly:
        max_images = 8

        def __init__(self) -> None:
            self.calls: list[dict] = []

        def select_camera_views(self, request: dict) -> dict:
            self.calls.append(deepcopy(request))
            return {
                "selected_view_ids": [request["candidates"][0]["id"]],
                "action": None,
                "reason": "first candidate",
            }

    scene = {
        "scene_id": "scene",
        "scene_type": "storage",
        "boundary": [[0, 0], [5, 0], [5, 5], [0, 5]],
        "scene_height": 3.0,
        "objects": [
            {
                "id": "cabinet",
                "jid": "asset_cabinet",
                "category": "cabinet",
                    "center": [2.5, 2.5, 1.0],
                "size": [1.0, 0.5, 2.0],
                "rotation": [0.0, 0.0, 0.0],
                "asset_ref": {"asset_key": "asset_cabinet"},
                "metadata": {
                    "asset_metadata": {
                        "catalog_hashes": {
                            "mesh_sha256": "mesh-cabinet"
                        }
                    }
                },
            }
        ],
    }
    discovery = {
        "schema_version": "functional_discovery_v3",
        "inspected_object_ids": ["cabinet"],
        "directed_surface_targets": [
            {
                "discovery_id": "directed_surface_01",
                "target_id": "cabinet",
                "directionality": "directed",
                "surface_roles": ["opening_side"],
                "need_clearance": True,
                "boundary_review_state": "routine",
                "owning_group_id": "g",
                "observation_goal": "show opening access",
            }
        ],
        "within_group_correspondences": [],
        "cross_group_correspondences": [],
        "approach_clearance_targets": [
            {
                "discovery_id": "approach_clearance_01",
                "target_id": "cabinet",
                "need_clearance": True,
                "owning_group_id": "g",
                "observation_goal": "show opening access",
            }
        ],
        "boundary_sensitive_targets": [],
        "unusual_unconfirmed": [],
        "reason": "complete",
        "provenance": {},
    }

    class Planner:
        def discover_functional_evidence(self, request: dict) -> dict:
            return deepcopy(discovery)

    detector = FakeDetector()
    selector = SelectionOnly()
    provider = CameraEvidenceProvider(
        renderer=_Renderer(),
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        mode="query_cov",
        selector=selector,
        usable_surface_detector=detector,
        max_views=1,
        max_steps=0,
        candidate_count=4,
    )
    paths, audit = acquire_functional_probe_evidence(
        planner=Planner(),
        provider=provider,
        scene=scene,
        global_image_path=str(global_image),
        max_probe_units=1,
        groups=[{"group_id": "g", "object_ids": ["cabinet"]}],
    )
    packet = functional_probe_judge_packet(
        global_paths=[str(global_image)],
        probe_paths=paths,
        acquisition_audit=audit,
    )

    assert len(detector.calls) == 1
    assert detector.calls[0]["read_only_provenance"][
        "scene_access"
    ] == "read_only"
    assert detector.calls[0]["available_side_ids"]
    assert selector.calls
    assert audit["functional_boundary_evidence"]["status"] == "complete"
    assert audit["functional_acquisition_plan"]["probe_units"]
    assert paths
    assert packet["planning_role"] == (
        "visual_evidence_only_no_metric_verdict"
    )
    assert packet["image_order"][1]["role"] == "functional_probe"


def test_malformed_surface_detector_output_fails_closed(
    tmp_path: Path,
) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")

    class MalformedDetector:
        def manifest(self) -> dict:
            return {
                "implementation_id": "malformed",
                "version": "v1",
                "configuration": {},
                "decision_authority": "none",
            }

        def detect(self, request: dict) -> dict:
            return {
                "status": "identified",
                "surfaces": [
                    {
                        "surface_role": request["surface_roles"][0],
                        "side_id": "untrusted_world_side",
                        "visual_cues": ["invalid"],
                        "confidence": 1.0,
                    }
                ],
                "reason": "invalid side",
            }

    class Selector:
        def select_camera_views(self, request: dict) -> dict:
            return {
                "selected_view_ids": [request["candidates"][0]["id"]],
                "action": None,
                "reason": "test selector",
            }

    provider = CameraEvidenceProvider(
        renderer=_Renderer(),
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        mode="query_cov",
        selector=Selector(),
        usable_surface_detector=MalformedDetector(),
    )
    result = provider.provide_functional_boundary_evidence(
        {
            "metric": "functional_consistency",
            "scene": {
                "scene_id": "scene",
                "boundary": [[0, 0], [5, 0], [5, 5], [0, 5]],
                "objects": [
                    {
                        "id": "cabinet",
                        "category": "cabinet",
                        "center": [2.5, 2.5, 1.0],
                        "size": [1.0, 0.5, 2.0],
                        "rotation": [0.0, 0.0, 0.0],
                    }
                ],
            },
            "surface_targets": [
                {
                    "target_id": "cabinet",
                    "surface_roles": ["opening_side"],
                }
            ],
        }
    )

    assert result["status"] == "failed"
    assert result["usable_surface_hypotheses"] == []
    assert result["decoder_audit"]["failures"][0]["reason"] == (
        "usable_surface_decode_failed"
    )


def test_boundary_prepass_computes_geometry_and_reuses_surface_decode(
    tmp_path: Path,
) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")
    renderer = _Renderer()
    selector = _Selector()
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        mode="query_cov",
        selector=selector,
        max_views=1,
        max_steps=0,
        candidate_count=4,
        usable_surface_cache_dir=tmp_path / "surface_cache",
    )
    scene = {
        "scene_id": "scene",
        "boundary": [[0, 0], [8, 0], [8, 8], [0, 8]],
        "scene_height": 3.0,
        "objects": [
            {
                "id": "cabinet",
                "jid": "asset_cabinet",
                "category": "cabinet",
                "center": [7.0, 4.0, 1.0],
                "size": [1.0, 0.5, 2.0],
                "rotation": [0.0, 0.0, 0.0],
                "asset_ref": {"asset_key": "asset_cabinet"},
                "metadata": {
                    "asset_metadata": {
                        "catalog_hashes": {
                            "mesh_sha256": "mesh-cabinet"
                        }
                    }
                },
            }
        ],
    }
    discovery = {
        "directed_surface_targets": [
            {
                "discovery_id": "directed_surface_01",
                "target_id": "cabinet",
                "directionality": "directed",
                "surface_roles": ["opening_side"],
                "need_clearance": True,
                "boundary_review_state": "routine",
                "owning_group_id": "g",
                "observation_goal": "show ordinary opening access",
            }
        ],
        "within_group_correspondences": [],
        "cross_group_correspondences": [],
        "approach_clearance_targets": [
            {
                "discovery_id": "approach_clearance_01",
                "target_id": "cabinet",
                "need_clearance": True,
                "owning_group_id": "g",
                "observation_goal": "show ordinary opening access",
            }
        ],
        "boundary_sensitive_targets": [],
        "unusual_unconfirmed": [],
    }
    boundary_evidence = acquire_functional_boundary_evidence(
        provider=provider,
        scene=scene,
        discovery=discovery,
        architecture_context={"logical_boundary_enabled": True},
    )

    assert boundary_evidence["status"] == "complete"
    assert len(selector.surface_calls) == 1
    assert all(call["preview"] is True for call in renderer.calls)
    surface = boundary_evidence["functional_geometry"][
        "surface_observations"
    ][0]
    assert surface["descriptor_kind"] == "usable_side_world_direction"
    assert surface["routing_only"] is True
    assert surface["architecture_orientation_applicable"] is True
    assert surface["clearance_applicable"] is True
    assert surface["outward_ray_boundary_distance_m"] == pytest.approx(
        0.499
    )
    assert surface["approach_samples"][0][
        "inside_logical_boundary"
    ] is False

    plan = build_functional_acquisition_plan(
        discovery_with_boundary_hypotheses(
            discovery,
            boundary_evidence,
        ),
        max_probe_units=1,
    )
    unit = plan["probe_units"][0]
    assert "precomputed_hypothesis" in unit["surface_targets"][0]
    provider(
        {
            "metric": "functional_consistency",
            "scene": scene,
            "object_ids": ["cabinet"],
            "evidence_scope": "object_local",
            "evidence_policy": {
                "camera_pose_mode": "query_cov",
                "scoped_image_budget": 1,
            },
            "functional_probe": {
                **unit,
                "group_id": "g",
                "group_member_ids": ["cabinet"],
            },
        }
    )

    assert len(selector.surface_calls) == 1
    assert provider.last_call_usage["usable_surface"][
        "precomputed_hypotheses"
    ] == 1


def test_zero_full_probe_budget_still_routes_boundary_measurements(
    tmp_path: Path,
) -> None:
    global_image = tmp_path / "global.png"
    global_image.write_bytes(b"image")
    scene = {
        "scene_id": "scene",
        "scene_type": "storage",
        "boundary": [[0, 0], [5, 0], [5, 5], [0, 5]],
        "objects": [
            {
                "id": "cabinet",
                "category": "cabinet",
                "center": [4.0, 2.5, 1.0],
                "size": [1.0, 0.5, 2.0],
                "rotation": [0.0, 0.0, 0.0],
            }
        ],
    }

    class Planner:
        def discover_functional_evidence(self, request: dict) -> dict:
            return {
                "schema_version": "functional_discovery_v3",
                "inspected_object_ids": ["cabinet"],
                "directed_surface_targets": [
                    {
                        "discovery_id": "directed_surface_01",
                        "target_id": "cabinet",
                        "directionality": "directed",
                        "surface_roles": ["opening_side"],
                        "need_clearance": True,
                        "boundary_review_state": "routine",
                        "owning_group_id": "g",
                        "observation_goal": "show ordinary opening access",
                    }
                ],
                "within_group_correspondences": [],
                "cross_group_correspondences": [],
                "approach_clearance_targets": [
                    {
                        "discovery_id": "approach_clearance_01",
                        "target_id": "cabinet",
                        "need_clearance": True,
                        "owning_group_id": "g",
                        "observation_goal": "show ordinary opening access",
                    }
                ],
                "boundary_sensitive_targets": [],
                "unusual_unconfirmed": [],
                "reason": "complete",
                "provenance": {},
            }

    class BoundaryProvider:
        def __init__(self) -> None:
            self.calls = 0

        def provide_functional_boundary_evidence(
            self,
            request: dict,
        ) -> dict:
            self.calls += 1
            return {
                "status": "complete",
                "decision_authority": "none",
                "scene_access": "read_only",
                "surface_targets": deepcopy(
                    request["surface_targets"]
                ),
                "usable_surface_hypotheses": [
                    {
                        "target_id": "cabinet",
                        "status": "identified",
                        "surfaces": [
                            {
                                "surface_role": "opening_side",
                                "side_id": "local_pos_x",
                                "visual_cues": ["opening plane"],
                                "confidence": 0.9,
                            }
                        ],
                        "reason": "opening plane",
                    }
                ],
                "functional_geometry": {
                    "schema_version": "functional_geometry_v1",
                    "decision_authority": "none",
                    "scene_access": "read_only",
                    "logical_boundary_available": True,
                    "surface_observations": [
                        {
                            "target_id": "cabinet",
                            "nearest_boundary_distance_m": 0.5,
                            "outward_ray_boundary_distance_m": 0.5,
                            "approach_samples": [],
                        }
                    ],
                    "observation_status": "available",
                },
                "decoder_audit": {
                    "decoder_calls": 1,
                    "cache_hits": 0,
                    "preview_render_count": 8,
                },
                "provenance": {"geometry_source": "deterministic"},
            }

    provider = BoundaryProvider()
    paths, audit = acquire_functional_probe_evidence(
        planner=Planner(),
        provider=provider,
        scene=scene,
        global_image_path=str(global_image),
        max_probe_units=0,
        groups=[{"group_id": "g", "object_ids": ["cabinet"]}],
    )
    packet = functional_probe_judge_packet(
        global_paths=[str(global_image)],
        probe_paths=paths,
        acquisition_audit=audit,
    )

    assert paths == []
    assert provider.calls == 1
    assert audit["status"] == "complete_no_probes"
    assert audit["usable_surface_decoder_calls"] == 1
    assert audit["rendered_probe_count"] == 0
    assert audit["forced_group_ids"] == ["g"]
    assert audit["group_probe_packets"]["g"][
        "boundary_clearance_evidence"
    ]["functional_geometry"]["surface_observations"][0][
        "outward_ray_boundary_distance_m"
    ] == 0.5
    assert packet["boundary_clearance_evidence"][
        "decision_authority"
    ] == "none"
    assert "decoder_audit" not in packet["boundary_clearance_evidence"]
    assert "provenance" not in json.dumps(
        packet["boundary_clearance_evidence"]
    )
    assert packet["image_order"] == [
        {
            "image_index": 0,
            "image_alias": "image_00",
            "role": "scene_global",
        }
    ]


def test_functional_geometry_uses_rotated_frontage_not_object_center() -> None:
    scene = {
        "boundary": [[0, 0], [6, 0], [6, 6], [0, 6]],
        "objects": [
            {
                "id": "cabinet",
                "center": [4.8, 3.0, 1.0],
                "size": [3.0, 1.0, 2.0],
                "rotation": [0.0, 0.0, 90.0],
            }
        ],
    }
    result = build_functional_geometry_observations(
        scene,
        {
            "probe_id": "frontage",
            "kind": "functional_frontage",
            "target_ids": ["cabinet"],
            "related_target_ids": [],
            "required_observations": [
                "architecture_plane_visible"
            ],
            "usable_surface_hypotheses": [
                {
                    "target_id": "cabinet",
                    "status": "identified",
                    "surfaces": [
                        {
                            "side_id": "local_pos_x",
                            "surface_role": "opening_side",
                        }
                    ],
                }
            ],
        },
    )

    surface = result["surface_observations"][0]
    assert surface["object_center_xy"] == [4.8, 3.0]
    assert surface["frontage_origin_xy"][1] > 4.49
    assert surface["frontage_support_extent_m"] == pytest.approx(1.5)
    assert surface["approach_samples"][0]["point_xy"][1] > 4.99
    # A 3m x 1m object rotated 90 degrees has a 1m x 3m world footprint.
    assert result["target_bounds"][0][:2] == pytest.approx([4.3, 1.5])
    assert result["target_bounds"][1][:2] == pytest.approx([5.3, 4.5])
