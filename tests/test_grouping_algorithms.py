from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from PIL import Image
import pytest
import yaml

from benchmark.grouping import (
    GROUPING_ROLE,
    AnchorGroupingAlgorithm,
    GroupingRequest,
    GroupingResult,
    TopologyGroupingAlgorithm,
    VLMGroupingAlgorithm,
    build_grouping_algorithm,
    group_scene,
    normalize_grouping_scene,
)
from benchmark.evaluator.scene_quality.interfaces import _normalize_groups
from benchmark.evaluator.evidence_contract import (
    GROUPING_ROLE as EVALUATOR_GROUPING_ROLE,
)


ROOT = Path(__file__).resolve().parents[1]


def _object(
    object_id: str | None,
    description: str,
    center: list[float],
    size: list[float],
    *,
    support_parent: str | None = None,
    region_id: str | None = None,
) -> dict:
    result = {
        "short_desc": description,
        "desc": description,
        "jid": f"asset_{description.replace(' ', '_')}",
        "center": center,
        "size": size,
        "rotation": [0.0, 0.0, 0.0],
    }
    if object_id is not None:
        result["id"] = object_id
    if support_parent is not None:
        result["support_parent"] = support_parent
    if region_id is not None:
        result["region_id"] = region_id
    return result


def _mixed_scene() -> dict:
    return {
        "scene_id": "mixed_room",
        "scene_type": "bedroom with home office",
        "boundary": [[0, 0], [10, 0], [10, 8], [0, 8]],
        "objects": [
            _object("bed", "queen bed", [2, 2, 0.5], [2, 2, 1]),
            _object(
                "nightstand",
                "wood nightstand",
                [3.3, 2, 0.3],
                [0.5, 0.5, 0.6],
            ),
            _object(
                "desk",
                "office desk",
                [8, 6, 0.4],
                [1.5, 0.7, 0.8],
            ),
            _object(
                "chair",
                "office chair",
                [7.5, 5, 0.5],
                [0.6, 0.6, 1],
            ),
        ],
    }


class _Model:
    model_id = "grouping-test-model"
    endpoint = "http://127.0.0.1:9999/v1"

    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[dict] = []
        self.last_request_metadata = {"image_count": 1}

    def chat_messages(self, messages, **kwargs) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return json.dumps(self.response)


def _vlm_response() -> dict:
    # Deliberately put the work group first. The shared result contract
    # canonicalizes group order using the scene object order.
    return {
        "object_groups": [
            {
                "object_ids": ["desk", "chair"],
                "label": "work context",
                "anchor_object_id": "desk",
                "reason": "Desk and office chair form one local work scope.",
            },
            {
                "object_ids": ["bed", "nightstand"],
                "label": "sleep context",
                "anchor_object_id": "bed",
                "reason": "Bed and nightstand form one local sleep scope.",
            },
        ],
        "reason": "Two conservative local evidence scopes.",
    }


def test_scene_normalization_derives_stable_ids_without_mutation() -> None:
    scene = {
        "scene_id": "converted",
        "boundary": [[0, 0], [4, 0], [4, 4], [0, 4]],
        "objects": [
            _object(None, "gray sofa", [1, 1, 0.5], [2, 1, 1]),
            _object(None, "gray sofa", [3, 1, 0.5], [2, 1, 1]),
        ],
    }
    original = deepcopy(scene)

    normalized = normalize_grouping_scene(scene)

    assert normalized.object_ids == (
        "scene_object_000",
        "scene_object_001",
    )
    assert normalized.derived_object_id_count == 2
    assert scene == original


def test_grouping_role_matches_existing_evidence_contract() -> None:
    assert GROUPING_ROLE == EVALUATOR_GROUPING_ROLE


def test_topology_backend_handles_converted_scene_shape_and_exact_partition():
    scene = {
        "scene_id": "converted",
        "boundary": [[0, 0], [8, 0], [8, 8], [0, 8]],
        "objects": [
            _object(None, "gray sofa", [1, 1, 0.5], [2, 1, 1]),
            _object(None, "coffee table", [2.4, 1, 0.25], [1, 1, 0.5]),
            _object(None, "wardrobe", [7, 7, 1], [1, 0.5, 2]),
        ],
    }

    result = TopologyGroupingAlgorithm().group(
        GroupingRequest(scene=scene)
    )
    report = result.to_dict()
    assigned = [
        object_id
        for group in report["object_groups"]
        for object_id in group["object_ids"]
    ]

    assert assigned == [
        "scene_object_000",
        "scene_object_001",
        "scene_object_002",
    ]
    assert len(assigned) == len(set(assigned))
    assert report["grouping_role"] == GROUPING_ROLE
    assert report["grouping_backend"] == "topology"
    assert report["provenance"]["model_calls"] == 0


def test_anchor_backend_builds_sleep_and_work_scopes() -> None:
    result = AnchorGroupingAlgorithm().group(
        GroupingRequest(scene=_mixed_scene())
    )
    groups = result.to_dict()["object_groups"]

    assert [group["object_ids"] for group in groups] == [
        ["bed", "nightstand"],
        ["desk", "chair"],
    ]
    assert groups[0]["anchor_object_id"] == "bed"
    assert groups[0]["anchor_family"] == "sleep"
    assert groups[1]["anchor_object_id"] == "desk"
    assert groups[1]["anchor_family"] == "work"
    assert result.provenance["model_calls"] == 0


def test_anchor_backend_preserves_explicit_support_even_when_far() -> None:
    scene = {
        "objects": [
            _object("desk", "office desk", [0, 0, 0.5], [2, 1, 1]),
            _object(
                "lamp",
                "desk lamp",
                [8, 8, 1.5],
                [0.2, 0.2, 0.5],
                support_parent="desk",
            ),
        ]
    }

    result = AnchorGroupingAlgorithm(
        {"anchor": {"max_assignment_gap_m": 0.1}}
    ).group(GroupingRequest(scene=scene))
    group = result.to_dict()["object_groups"][0]

    assert group["object_ids"] == ["desk", "lamp"]
    assignment = next(
        item
        for item in group["assignments"]
        if item["object_id"] == "lamp"
    )
    assert assignment["reason_codes"] == ["support_parent"]


def test_anchor_backend_retains_unmatched_object_as_singleton() -> None:
    scene = _mixed_scene()
    scene["objects"].append(
        _object(
            "remote_art",
            "small wall art",
            [30, 30, 1.5],
            [0.4, 0.05, 0.4],
        )
    )
    result = AnchorGroupingAlgorithm(
        {
            "anchor": {
                "max_assignment_gap_m": 1.0,
                "max_assignment_gap_ratio": 0.0,
            }
        }
    ).group(GroupingRequest(scene=scene))

    singleton = next(
        group
        for group in result.to_dict()["object_groups"]
        if group["object_ids"] == ["remote_art"]
    )
    assert singleton["group_source"] == "anchor_singleton"
    assert "without guessing" in singleton["reason"]


def test_vlm_backend_uses_partition_prompt_and_visual_context(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "global.png"
    image = Image.new("RGB", (4, 4), (80, 90, 100))
    image.putpixel((0, 0), (10, 20, 30))
    image.save(image_path)
    model = _Model(_vlm_response())
    scene = _mixed_scene()
    original = deepcopy(scene)

    result = VLMGroupingAlgorithm(model).group(
        GroupingRequest(
            scene=scene,
            visual_evidence=(
                {
                    "path": str(image_path),
                    "role": "global_identity_overlay",
                    "identity_overlay": True,
                    "object_ids": [
                        "bed",
                        "nightstand",
                        "desk",
                        "chair",
                    ],
                },
            ),
            context={
                "scene_intent": "sleep and work zones",
                "private_path": "/must/not/appear",
            },
        )
    )

    assert [
        group.object_ids for group in result.object_groups
    ] == [("bed", "nightstand"), ("desk", "chair")]
    assert result.object_groups[0].group_id == "group_001"
    assert result.object_groups[1].group_id == "group_002"
    assert result.provenance["images_used"] == ["view_00"]
    assert result.provenance["model"] == "grouping-test-model"
    assert scene == original
    call = model.calls[0]
    assert call["kwargs"]["response_format_json"] is True
    assert call["kwargs"]["call_type"] == "vlm_grouping.partition"
    system = call["messages"][0]["content"]
    assert "not a benchmark metric" in system
    assert "verdict, score, confidence" in system
    assert "make groups look more plausible" in system
    content = call["messages"][1]["content"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    prompt = content[0]["text"]
    assert '"role":"evidence_partition_not_metric_verdict"' in prompt
    assert '"identity_overlay":true' in prompt
    assert all(object_id in prompt for object_id in ("bed", "nightstand", "desk", "chair"))
    assert str(image_path) not in prompt
    assert "/must/not/appear" not in prompt


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (
            {
                **_vlm_response(),
                "verdict": "valid",
            },
            "must not contain metric outputs",
        ),
        (
            {
                "object_groups": [
                    {
                        "object_ids": ["bed", "nightstand"],
                        "label": "sleep",
                        "anchor_object_id": "bed",
                        "reason": "local sleep scope",
                        "score": 1.0,
                    },
                    {
                        "object_ids": ["desk", "chair"],
                        "label": "work",
                        "anchor_object_id": "desk",
                        "reason": "local work scope",
                    },
                ],
                "reason": "partition",
            },
            "must not contain metric outputs",
        ),
        (
            {
                "object_groups": [
                    {
                        "object_ids": ["bed", "nightstand"],
                        "label": "sleep",
                        "anchor_object_id": "bed",
                        "reason": "local sleep scope",
                    },
                    {
                        "object_ids": ["desk", "unknown"],
                        "label": "work",
                        "anchor_object_id": "desk",
                        "reason": "local work scope",
                    },
                ],
                "reason": "partition",
            },
            "unknown object IDs",
        ),
        (
            {
                "object_groups": [
                    {
                        "object_ids": ["bed", "nightstand"],
                        "label": "sleep",
                        "anchor_object_id": "bed",
                        "reason": "local sleep scope",
                    },
                    {
                        "object_ids": ["desk"],
                        "label": "work",
                        "anchor_object_id": "desk",
                        "reason": "local work scope",
                    },
                ],
                "reason": "partition",
            },
            "missing \\['chair'\\]",
        ),
        (
            {
                "object_groups": [
                    {
                        "object_ids": ["bed", "nightstand", "desk"],
                        "label": "first",
                        "anchor_object_id": "bed",
                        "reason": "scope",
                    },
                    {
                        "object_ids": ["desk", "chair"],
                        "label": "second",
                        "anchor_object_id": "chair",
                        "reason": "scope",
                    },
                ],
                "reason": "partition",
            },
            "more than once",
        ),
        (
            {
                "object_groups": [
                    {
                        "object_ids": ["bed", "nightstand"],
                        "label": "sleep",
                        "anchor_object_id": "desk",
                        "reason": "scope",
                    },
                    {
                        "object_ids": ["desk", "chair"],
                        "label": "work",
                        "anchor_object_id": "desk",
                        "reason": "scope",
                    },
                ],
                "reason": "partition",
            },
            "anchor_object_id must be a group member",
        ),
    ],
)
def test_vlm_backend_fails_closed_on_invalid_partition(
    response: dict,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        VLMGroupingAlgorithm(_Model(response)).group(
            GroupingRequest(scene=_mixed_scene())
        )


def test_grouping_result_rejects_overlap_and_missing_members() -> None:
    with pytest.raises(ValueError, match="multiple groups"):
        GroupingResult.create(
            groups=[
                {"object_ids": ["a", "b"]},
                {"object_ids": ["b", "c"]},
            ],
            expected_object_ids=("a", "b", "c"),
            backend="test",
            policy_id="test_v1",
            reason="test partition",
        )


def test_factory_backends_share_one_interface() -> None:
    assert isinstance(
        build_grouping_algorithm({"grouping": {"backend": "topology"}}),
        TopologyGroupingAlgorithm,
    )
    assert isinstance(
        build_grouping_algorithm({"grouping": {"backend": "anchor"}}),
        AnchorGroupingAlgorithm,
    )
    assert isinstance(
        build_grouping_algorithm(
            {"grouping": {"backend": "vlm"}},
            model=_Model(_vlm_response()),
        ),
        VLMGroupingAlgorithm,
    )
    with pytest.raises(ValueError, match="requires an injected chat model"):
        build_grouping_algorithm({"grouping": {"backend": "vlm"}})


def test_group_scene_default_is_topology_and_returns_result() -> None:
    result = group_scene(_mixed_scene())

    assert isinstance(result, GroupingResult)
    assert result.backend == "topology"


@pytest.mark.parametrize("backend", ["topology", "anchor"])
def test_new_reports_are_accepted_by_existing_group_consumer(
    backend: str,
) -> None:
    result = group_scene(
        _mixed_scene(),
        config={"grouping": {"backend": backend}},
    )

    normalized = _normalize_groups(
        result.to_dict(),
        valid_object_ids={"bed", "nightstand", "desk", "chair"},
    )

    assert normalized is not None
    assert {
        object_id
        for group in normalized
        for object_id in group["object_ids"]
    } == {"bed", "nightstand", "desk", "chair"}


def test_vlm_empty_scene_does_not_call_model() -> None:
    model = _Model(_vlm_response())

    result = VLMGroupingAlgorithm(model).group(
        GroupingRequest(scene={"objects": []})
    )

    assert result.object_groups == ()
    assert result.provenance["model_calls"] == 0
    assert model.calls == []


def test_vlm_request_config_overrides_constructor_defaults() -> None:
    model = _Model(_vlm_response())

    result = VLMGroupingAlgorithm(model).group(
        GroupingRequest(
            scene=_mixed_scene(),
            config={
                "grouping": {
                    "backend": "vlm",
                    "vlm": {"response_format_json": False},
                }
            },
        )
    )

    assert model.calls[0]["kwargs"]["response_format_json"] is False
    assert (
        result.resolved_grouping_config["response_format_json"]
        is False
    )


@pytest.mark.parametrize(
    ("filename", "backend"),
    [
        ("topology_metadata_geometry_v2.yaml", "topology"),
        ("anchor_object_v1.yaml", "anchor"),
        ("vlm_semantic_partition_v1.yaml", "vlm"),
    ],
)
def test_reference_grouping_configs_construct_expected_backend(
    filename: str,
    backend: str,
) -> None:
    config = yaml.safe_load(
        (ROOT / "configs" / "grouping" / filename).read_text(
            encoding="utf-8"
        )
    )

    algorithm = build_grouping_algorithm(
        config,
        model=_Model(_vlm_response()) if backend == "vlm" else None,
    )

    assert algorithm.backend == backend
