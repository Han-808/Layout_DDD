from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from PIL import Image
import pytest
import yaml

from benchmark.grouping import (
    ACTIVE_GROUPING_BACKENDS,
    DEFAULT_GROUPING_BACKEND,
    DEFAULT_GROUPING_FALLBACK_CONFIG,
    DEPRECATED_GROUPING_BACKENDS,
    GROUPING_ROLE,
    AnchorGroupingAlgorithm,
    GroupingRequest,
    GroupingResult,
    TopologyGroupingAlgorithm,
    VLMGroupingAlgorithm,
    VLMPrimaryGroupingAlgorithm,
    build_grouping_algorithm,
    group_scene,
    prepare_grouping_evidence,
    normalize_grouping_scene,
)
from benchmark.evaluator.scene_quality.interfaces import _normalize_groups
from benchmark.api.evaluation import _resolve_object_grouping_report
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


class _RaisingModel:
    model_id = "grouping-failure-model"
    endpoint = "http://127.0.0.1:9999/v1"

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def chat_messages(self, messages, **kwargs) -> str:
        self.calls += 1
        raise self.error


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
    assert result.provenance["vlm_role"] == "vlm_grouping"
    assert (
        result.provenance["decision_contract"]
        == "grouping_partition_v1"
    )
    assert scene == original
    call = model.calls[0]
    assert call["kwargs"]["response_format_json"] is True
    assert call["kwargs"]["call_type"] == "vlm_grouping.partition"
    system = call["messages"][0]["content"]
    assert "not a benchmark metric" in system
    assert "verdicts, scores, confidence" in system
    assert "make the partition appear more plausible" in system
    assert "smallest local scope" in system
    assert "Do not chain weak proximity links" in system
    assert "lower supplied source_index" in system
    assert 'Set label exactly to "local_scope:<anchor_object_id>"' in system
    assert result.provenance["prompt_version"] == "vlm_grouping_prompt_v3"
    content = call["messages"][1]["content"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    prompt = content[0]["text"]
    assert '"vlm_role":"vlm_grouping"' in prompt
    assert '"decision_contract":"grouping_partition_v1"' in prompt
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
        VLMPrimaryGroupingAlgorithm,
    )
    assert isinstance(
        build_grouping_algorithm({"grouping": {"backend": "vlm"}}),
        VLMPrimaryGroupingAlgorithm,
    )


def test_group_scene_default_is_vlm_primary_with_topology_fallback() -> None:
    assert DEFAULT_GROUPING_BACKEND == "vlm"
    assert DEFAULT_GROUPING_FALLBACK_CONFIG == {
        "enabled": True,
        "backend": "topology",
    }
    assert ACTIVE_GROUPING_BACKENDS == ("vlm",)
    assert DEPRECATED_GROUPING_BACKENDS == ("topology", "anchor")

    fallback = group_scene(_mixed_scene())
    assert fallback.backend == "topology"
    assert fallback.to_dict()["fallback_used"] is True
    route = fallback.provenance["grouping_fallback"]
    assert route["primary_outcome"] == "failed"
    assert route["vlm_role"] == "vlm_grouping"
    assert route["decision_contract"] == "grouping_partition_v1"
    assert route["primary_failure"]["error_type"] == "ValueError"
    assert route["fallback_backend"] == "topology"

    result = group_scene(
        _mixed_scene(),
        model=_Model(_vlm_response()),
    )

    assert isinstance(result, GroupingResult)
    assert result.backend == "vlm"
    assert result.to_dict()["fallback_used"] is False
    route = result.provenance["grouping_fallback"]
    assert route["primary_outcome"] == "complete"
    assert route["vlm_role"] == "vlm_grouping"
    assert route["decision_contract"] == "grouping_partition_v1"
    assert route["model"] == "grouping-test-model"
    assert route["endpoint"] == "http://127.0.0.1:9999/v1"
    assert route["last_request_metadata"] == {"image_count": 1}


def test_vlm_transport_failure_uses_audited_topology_fallback() -> None:
    result = group_scene(
        _mixed_scene(),
        model=_RaisingModel(RuntimeError("transport unavailable")),
    )

    assert result.backend == "topology"
    assert {
        object_id
        for group in result.object_groups
        for object_id in group.object_ids
    } == {"bed", "nightstand", "desk", "chair"}
    route = result.provenance["grouping_fallback"]
    assert route["fallback_used"] is True
    assert route["primary_failure"] == {
        "error_type": "RuntimeError",
        "message": "transport unavailable",
    }
    assert route["vlm_role"] == "vlm_grouping"
    assert route["decision_contract"] == "grouping_partition_v1"
    assert route["model"] == "grouping-failure-model"
    assert route["endpoint"] == "http://127.0.0.1:9999/v1"
    assert result.resolved_grouping_config["fallback"]["backend"] == (
        "topology"
    )


def test_grouping_fallback_audit_sanitizes_model_request_metadata() -> None:
    model = _RaisingModel(RuntimeError("transport unavailable"))
    model.endpoint = (
        "https://audit-user:credential@example.test/v1"
        "?api_key=endpoint-secret"
    )
    model.last_request_metadata = {
        "endpoint": model.endpoint,
        "url": (
            "https://audit-user:credential@example.test/v1/chat/completions"
            "?token=url-secret"
        ),
        "model": "grouping-failure-model",
        "call_type": "vlm_grouping.partition",
        "message_count": 2,
        "image_count": 1,
        "prompt_chars": 1234,
        "api_key_env": "LITELLM_MASTER_KEY",
        "authorization_configured": True,
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "raw_response": "must-not-be-copied",
        },
        "prompt_budget_report": {
            "call_type": "vlm_grouping.partition",
            "prompt_chars": 1234,
            "estimated_prompt_tokens": 308,
            "sections": [
                {
                    "name": "scene_context",
                    "chars": 900,
                    "estimated_tokens": 225,
                    "text": "full-prompt-must-not-be-copied",
                }
            ],
        },
        "api_key": "literal-api-secret",
        "authorization": "Bearer bearer-secret",
        "messages": [{"content": "full-prompt-must-not-be-copied"}],
        "image_data": "data:image/png;base64,full-image-must-not-be-copied",
    }

    result = group_scene(_mixed_scene(), model=model)

    route = result.provenance["grouping_fallback"]
    assert route["vlm_role"] == "vlm_grouping"
    assert route["decision_contract"] == "grouping_partition_v1"
    assert route["model"] == "grouping-failure-model"
    assert route["endpoint"] == "https://example.test/v1"
    metadata = route["last_request_metadata"]
    assert metadata["endpoint"] == "https://example.test/v1"
    assert metadata["url"] == (
        "https://example.test/v1/chat/completions"
    )
    assert metadata["api_key_env"] == "LITELLM_MASTER_KEY"
    assert metadata["usage"] == {
        "completion_tokens": 20,
        "prompt_tokens": 100,
        "total_tokens": 120,
    }
    assert metadata["prompt_budget_report"]["sections"] == [
        {
            "name": "scene_context",
            "chars": 900,
            "estimated_tokens": 225,
        }
    ]
    serialized = json.dumps(route, sort_keys=True)
    for secret in (
        "credential",
        "endpoint-secret",
        "url-secret",
        "literal-api-secret",
        "bearer-secret",
        "full-prompt-must-not-be-copied",
        "full-image-must-not-be-copied",
    ):
        assert secret not in serialized


def test_vlm_schema_failure_uses_audited_topology_fallback() -> None:
    invalid = _vlm_response()
    invalid["object_groups"][1]["object_ids"] = ["bed"]

    result = group_scene(_mixed_scene(), model=_Model(invalid))

    assert result.backend == "topology"
    route = result.provenance["grouping_fallback"]
    assert route["fallback_used"] is True
    assert route["primary_failure"]["error_type"] == "ValueError"
    assert "missing" in route["primary_failure"]["message"]


@pytest.mark.parametrize(
    "fallback",
    [
        True,
        {"enabled": "yes"},
        {"backend": "anchor"},
        {"enabled": True, "unknown": 1},
    ],
)
def test_grouping_fallback_config_fails_closed(fallback: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_grouping_algorithm(
            {"grouping": {"backend": "vlm", "fallback": fallback}},
            model=_Model(_vlm_response()),
        )


def test_explicit_custom_algorithm_does_not_silently_fallback() -> None:
    class _CustomFailure:
        backend = "custom"
        policy_id = "custom_v1"

        def group(self, request):
            raise RuntimeError("custom grouping failed")

    with pytest.raises(RuntimeError, match="custom grouping failed"):
        group_scene(
            _mixed_scene(),
            model=_Model(_vlm_response()),
            algorithm=_CustomFailure(),
        )


def test_canonical_grouping_route_uses_vlm_then_topology_fallback() -> None:
    resolved = _resolve_object_grouping_report(
        None,
        scene=_mixed_scene(),
        request={"instruction": "Create a bedroom and work area."},
        visual_evidence=[],
        model=_Model(_vlm_response()),
    )
    assert resolved["status"] == "complete"
    assert resolved["grouping_backend"] == "vlm"
    assert (
        resolved["grouping_policy_id"]
        == "vlm_visual_evidence_scope_v2"
    )
    assert resolved["fallback_used"] is False

    fallback = _resolve_object_grouping_report(
        None,
        scene=_mixed_scene(),
        request={},
        visual_evidence=[],
        model=None,
    )
    assert fallback["status"] == "complete"
    assert fallback["source"] == "canonical_runtime_default"
    assert fallback["grouping_backend"] == "topology"
    assert fallback["grouping_policy_id"] == (
        "topology_metadata_geometry_v2"
    )
    assert fallback["fallback_used"] is True
    assert fallback["provenance"]["grouping_fallback"][
        "primary_failure"
    ]["error_type"] == "ValueError"


def test_canonical_grouping_evidence_protocol_is_identity_aware(
    tmp_path: Path,
) -> None:
    perspective = tmp_path / "global_perspective.png"
    top = tmp_path / "global_top.png"
    identity = tmp_path / "global_identity.png"
    for index, path in enumerate((perspective, top, identity)):
        image = Image.new("RGB", (4, 4), (50 + index, 60, 70))
        image.putpixel((0, 0), (10, 20 + index, 30))
        image.save(path)

    packet = prepare_grouping_evidence(
        {
            "global_perspective": str(perspective),
            "global_top": str(top),
            "identity_map": str(identity),
        },
        identity_legend={
            "A": "bed",
            "B": "nightstand",
            "C": "desk",
            "D": "chair",
        },
    )

    assert packet.input_mode == "identity_aware_perspective_top"
    assert packet.degraded_reasons == ()
    assert packet.available_roles == (
        "global_perspective_rgb",
        "global_top_rgb",
        "global_identity_overlay",
    )
    identity_item = next(
        item
        for item in packet.visual_evidence
        if item["representation"] == "identity_map"
    )
    assert identity_item["identity_legend"]["A"] == "bed"
    assert packet.provenance()["protocol_version"] == (
        "grouping_evidence_protocol_v1"
    )
    model = _Model(_vlm_response())
    resolved = _resolve_object_grouping_report(
        None,
        scene=_mixed_scene(),
        request={"instruction": "Create two functional zones."},
        visual_evidence=list(packet.visual_evidence),
        grouping_input_protocol=packet.provenance(),
        identity_legend=packet.identity_legend,
        model=model,
    )
    assert resolved["status"] == "complete"
    call_text = model.calls[0]["messages"][1]["content"][0]["text"]
    assert "identity_overlay_legend" in call_text
    assert '"A":"bed"' in call_text


def test_canonical_grouping_records_degraded_input_protocol(
    tmp_path: Path,
) -> None:
    overview = tmp_path / "overview.png"
    Image.new("RGB", (4, 4), (80, 90, 100)).save(overview)
    packet = prepare_grouping_evidence([str(overview)])
    model = _Model(_vlm_response())

    resolved = _resolve_object_grouping_report(
        None,
        scene=_mixed_scene(),
        request={"instruction": "Create two functional zones."},
        visual_evidence=list(packet.visual_evidence),
        grouping_input_protocol=packet.provenance(),
        identity_legend=packet.identity_legend,
        model=model,
    )

    protocol = resolved["provenance"]["grouping_input_protocol"]
    assert protocol["input_mode"] == "generic_overview_degraded"
    assert "identity_map_missing" in protocol["degraded_reasons"]


def test_caller_grouping_does_not_inherit_current_run_evidence_protocol() -> None:
    report = {
        "status": "complete",
        "grouping_backend": "vlm",
        "grouping_policy_id": "vlm_visual_evidence_scope_v2",
        "object_groups": [
            {
                "group_id": "group_001",
                "object_ids": ["bed", "nightstand"],
            },
            {
                "group_id": "group_002",
                "object_ids": ["desk", "chair"],
            },
        ],
    }
    current_run_protocol = {
        "input_mode": "identity_aware_perspective_top",
        "protocol_version": "grouping_evidence_protocol_v1",
    }

    resolved = _resolve_object_grouping_report(
        report,
        scene=_mixed_scene(),
        request={},
        visual_evidence=[],
        grouping_input_protocol=current_run_protocol,
    )

    assert resolved["provenance"]["grouping_input_protocol"] == {
        "input_mode": "caller_supplied_unknown",
        "provenance_status": "not_provided",
    }

    report["provenance"] = {
        "grouping_input_protocol": {
            "input_mode": "caller_recorded_protocol",
            "protocol_version": "external_v1",
        }
    }
    resolved_with_provenance = _resolve_object_grouping_report(
        report,
        scene=_mixed_scene(),
        request={},
        visual_evidence=[],
        grouping_input_protocol=current_run_protocol,
    )
    assert resolved_with_provenance["provenance"][
        "grouping_input_protocol"
    ] == report["provenance"]["grouping_input_protocol"]


def test_non_vlm_frozen_grouping_is_rejected_unless_diagnostic() -> None:
    topology_report = {
        "status": "complete",
        "grouping_backend": "topology",
        "grouping_policy_id": "topology_grouping_v1",
        "object_groups": [
            {
                "group_id": "group_001",
                "object_ids": ["bed", "nightstand"],
            },
            {
                "group_id": "group_002",
                "object_ids": ["desk", "chair"],
            },
        ],
    }

    rejected = _resolve_object_grouping_report(
        topology_report,
        scene=_mixed_scene(),
        request={},
        visual_evidence=[],
    )
    assert rejected["status"] == "unavailable"
    assert rejected["non_canonical_grouping_input"] is True
    assert rejected["reported_grouping_backend"] == "topology"
    assert (
        rejected["reported_grouping_policy_id"]
        == "topology_grouping_v1"
    )
    assert rejected["expected_grouping_backend"] == "vlm"
    assert (
        rejected["expected_grouping_policy_id"]
        == "vlm_visual_evidence_scope_v2"
    )
    assert "grouping_backend_must_be_vlm" in rejected[
        "validation_errors"
    ]

    diagnostic = _resolve_object_grouping_report(
        topology_report,
        scene=_mixed_scene(),
        request={},
        visual_evidence=[],
        allow_non_canonical_input=True,
    )
    assert diagnostic["status"] == "complete"
    assert diagnostic["grouping_backend"] == "topology"
    assert diagnostic["non_canonical_grouping_input"] is True
    assert diagnostic["diagnostic_only"] is True


def test_validated_runtime_fallback_report_can_be_replayed() -> None:
    fallback = group_scene(_mixed_scene()).to_dict()
    fallback["status"] = "complete"

    resolved = _resolve_object_grouping_report(
        fallback,
        scene=_mixed_scene(),
        request={},
        visual_evidence=[],
    )

    assert resolved["status"] == "complete"
    assert resolved["grouping_backend"] == "topology"
    assert resolved["fallback_used"] is True
    assert resolved["non_canonical_grouping_input"] is False


def test_fallback_can_be_explicitly_disabled() -> None:
    config = {
        "grouping": {
            "backend": "vlm",
            "fallback": {"enabled": False, "backend": "topology"},
        }
    }
    with pytest.raises(ValueError, match="requires an injected chat model"):
        build_grouping_algorithm(config)

    model = _RaisingModel(RuntimeError("primary failed"))
    algorithm = build_grouping_algorithm(config, model=model)
    with pytest.raises(RuntimeError, match="primary failed"):
        algorithm.group(GroupingRequest(scene=_mixed_scene()))


def test_invalid_partition_is_rejected_even_in_diagnostic_mode() -> None:
    missing_member_report = {
        "status": "complete",
        "grouping_backend": "vlm",
        "grouping_policy_id": "vlm_visual_evidence_scope_v2",
        "object_groups": [
            {
                "group_id": "group_001",
                "object_ids": ["bed", "nightstand", "desk"],
            }
        ],
    }

    rejected = _resolve_object_grouping_report(
        missing_member_report,
        scene=_mixed_scene(),
        request={},
        visual_evidence=[],
        allow_non_canonical_input=True,
    )
    assert rejected["status"] == "unavailable"
    assert rejected["reason"] == "non_canonical_grouping_input_rejected"
    assert "missing_object_ids:chair" in rejected["validation_errors"]


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
        ("vlm_visual_evidence_scope_v2.yaml", "vlm"),
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


def test_grouping_identity_legend_must_cover_unique_scene_ids(
    tmp_path: Path,
) -> None:
    from PIL import Image
    from benchmark.grouping import prepare_grouping_evidence

    items = []
    for name, role in (
        ("perspective", "global_perspective_rgb"),
        ("top", "global_top_rgb"),
        ("identity_map", "global_identity_overlay"),
    ):
        path = tmp_path / f"{name}.png"
        Image.new("RGB", (8, 8), (20, 40, 60)).save(path)
        item = {
            "path": str(path),
            "role": role,
            "representation": (
                "identity_map" if name == "identity_map" else "rgb"
            ),
        }
        if name == "identity_map":
            item["identity_legend"] = {
                "red": "a",
                "blue": "a",
            }
        items.append(item)

    with pytest.raises(ValueError, match="unique canonical object ID"):
        prepare_grouping_evidence(
            items,
            expected_object_ids=("a", "b"),
        )
