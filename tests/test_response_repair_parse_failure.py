"""Regression for a failed JSON repair, not a model/Blender integration test."""
from copy import deepcopy
import json

import pytest
from PIL import Image

from benchmark.visual_judge.contracts import ResponseSchemaRepairError
from benchmark.visual_judge.functional_discovery import _validate_discovery_response_with_single_repair
from benchmark.visual_judge.openai_camera_selector import OpenAICompatibleCameraSelector
from benchmark.visual_judge.response_repair import _repair_response_schema_once
from benchmark.visual_judge.roles import DecisionContract, VLMRole


class Model:
    model_id = "offline-mock"
    endpoint = "https://invalid.test/v1"
    response_format_json = True

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.last_request_metadata = {}

    def chat_messages(self, messages, **kwargs):
        self.calls.append(deepcopy(kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        self.last_request_metadata = {"finish_reason": "length", "tokens_usage": {"completion_tokens": 4096}}
        return response if isinstance(response, str) else json.dumps(response)


@pytest.mark.parametrize("bad", ["", '{"objects":[', "[]", "null"])
@pytest.mark.parametrize("initial", ["", '{"keep":"initial","bad":true}'])
def test_discovery_repair_parse_failure_salvages_without_unbound_local(bad, initial):
    model = Model([bad])
    calls = []

    def validate(value):
        raise ValueError("malformed discovery")

    def salvage(repair, original):
        calls.append((deepcopy(repair), deepcopy(original)))
        return {"retained": original.get("keep"), "item_salvage": {"defaulted": True}}

    result, audit = _validate_discovery_response_with_single_repair(
        model=model, normalized={"scene_id": "mock", "objects": []},
        messages=[], initial_raw=initial, initial_metadata={"finish_reason": "length"},
        response_format_json=True, call_type="functional_discovery.affordance",
        role=VLMRole.FUNCTIONAL_AFFORDANCE_DISCOVERY,
        decision_contract=DecisionContract.FUNCTIONAL_AFFORDANCE_DISCOVERY,
        label="affordance", repair_prompt="same evidence", validator=validate, salvage=salvage,
    )
    assert calls == [({}, json.loads(initial) if initial else {})]
    assert result["retained"] == ("initial" if initial else None)
    assert audit["attempt_count"] == 2 and audit["repair_retry_count"] == 1
    assert audit["recovered"] is False and audit["item_level_salvage"] is True
    assert audit["attempts"][1]["validation_error_type"] == "ModelResponseError"
    assert audit["attempts"][1]["request_metadata"]["finish_reason"] == "length"
    assert len(model.calls) == 1


@pytest.mark.parametrize("phase", ["affordance", "relations"])
@pytest.mark.parametrize("bad", ["", '{"unfinished":'])
def test_public_functional_discovery_double_truncation_keeps_explicit_defaults(tmp_path, phase, bad):
    image = tmp_path / "global.png"
    Image.new("RGB", (16, 16), "white").save(image)
    affordance = {
        "objects": [{"object_id": "cabinet", "directionality": "non_directed",
                     "surface_roles": [], "need_clearance": False,
                     "boundary_review_state": "routine", "review_state": "routine",
                     "observation_goal": "ordinary use", "boundary_observation_goal": ""}],
        "reason": "mock",
    }
    relations = {"considered_object_ids": ["cabinet"], "relations": [], "reason": "mock"}
    responses = [bad, bad, relations] if phase == "affordance" else [affordance, bad, bad]
    model = Model(responses)
    result = OpenAICompatibleCameraSelector(model).discover_functional_evidence({
        "metric": "functional_consistency", "scene_id": "mock", "scene_type": "living_room",
        "global_image_path": str(image), "objects": [{"id": "cabinet", "category": "cabinet"}],
        "groups": [{"group_id": "one", "object_ids": ["cabinet"]}],
    })
    assert len(model.calls) == 3
    audit = result["provenance"]["calls"][phase]["schema_validation"]
    assert audit["attempt_count"] == 2 and audit["recovered"] is False
    assert audit["item_level_salvage"] is True
    if phase == "affordance":
        assert result["object_coverage"][0]["defaulted"] is True
        assert result["object_coverage"][0]["inspected"] is False
        assert result["coverage"]["fraction"] < 1.0


@pytest.mark.parametrize("bad", ["", '{"unfinished":'])
@pytest.mark.parametrize("fallback_mode", ["none", "accepted", "rejected"])
def test_judge_repair_parse_failure_keeps_fallback_or_schema_failure(bad, fallback_mode):
    model = Model([bad, bad])
    fallback_calls = []

    def validate(value):
        if value != {"retained": True}:
            raise ValueError("invalid response")
        return value

    def fallback(repair, initial):
        fallback_calls.append((repair, initial))
        return {"retained": fallback_mode == "accepted"}

    kwargs = dict(
        model=model, messages=[], response_format_json=True, call_type="mock.judge",
        judge_label="mock", validator=validate, repair_prompt="same evidence",
        policy="test", semantic_signature=None, semantic_restore=None,
        fail_soft_fallback=None if fallback_mode == "none" else fallback,
    )
    if fallback_mode == "accepted":
        result, audit = _repair_response_schema_once(**kwargs)
        assert result == {"retained": True}
        assert audit["recovered"] is False and audit["item_level_salvage"] is True
    else:
        with pytest.raises(ResponseSchemaRepairError) as caught:
            _repair_response_schema_once(**kwargs)
        assert not isinstance(caught.value.__cause__, UnboundLocalError)
    assert len(model.calls) == 2
    assert fallback_calls == ([] if fallback_mode == "none" else [({}, {})])
