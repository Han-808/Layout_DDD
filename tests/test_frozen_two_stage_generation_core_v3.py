from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.api3_anthropic_runner_v2.generation_runner import run_case as run_case_v2
from tools.api3_anthropic_runner_v3.generation_runner import (
    ModelConfig as ModelConfigV3,
    load_model_json_emission,
    run_case as run_case_v3,
)
from tools.api3_anthropic_runner_v3.strict_json import StrictJSONError

from tests.test_frozen_two_stage_generation_core import (
    _Context,
    _Retriever,
    _api_body,
    _brief,
    _placement,
    _plan,
)

# v3 is v2 plus tolerance for exactly one whole-response JSON code fence.  These
# tests pin both halves of that claim: the fence is accepted and recorded, and
# everything the fence rule deliberately excludes still fails.  v2 is covered
# too, because the whole point of shipping a second bundle is that the published
# cohort's generator keeps rejecting the fence.


def _fenced(value: dict, *, tag: str = "json") -> str:
    return f"```{tag}\n" + json.dumps(value, separators=(",", ":")) + "\n```"


def _model_v3(endpoint: str, *, retries: int = 0, timeout: float = 1.0) -> ModelConfigV3:
    # Mirrors `_model` in the v2 core tests, rebuilt with v3's own dataclass so
    # the sealed bundles stay independent of each other.
    return ModelConfigV3(
        key="test-model",
        label="Test Model",
        endpoint=endpoint,
        api_key="EMPTY",
        configured_model="openai/test-model",
        wire_model="test-model",
        timeout_seconds=timeout,
        max_infrastructure_retries=retries,
        retry_delay_seconds=0.0,
        temperature=0.9,
        top_p=1.0,
        top_k=-1,
        max_tokens=65536,
        repetition_penalty=1.0,
        reasoning_effort="high",
        preserved_thinking=True,
        strategy_type="ConsistentHash",
    )


def test_load_model_json_emission_strips_exactly_one_whole_response_fence() -> None:
    plan = _plan()
    for tag in ("json", "JSON", ""):
        value, normalized, envelope = load_model_json_emission(
            _fenced(plan, tag=tag).encode("utf-8")
        )
        assert value == plan
        assert envelope == "single_json_code_fence_v1"
        assert json.loads(normalized.decode("utf-8")) == plan

    raw = json.dumps(plan, separators=(",", ":"))
    value, normalized, envelope = load_model_json_emission(raw.encode("utf-8"))
    assert value == plan
    assert envelope == "raw_json"
    # The raw path must hand back the original bytes untouched, because the
    # accepted artifact is still asserted to be a byte copy of the emission.
    assert normalized == raw.encode("utf-8")


def test_load_model_json_emission_rejects_prose_and_partial_fences() -> None:
    plan = json.dumps(_plan(), separators=(",", ":"))
    rejected = (
        "Here is the plan:\n```json\n" + plan + "\n```",  # surrounding prose
        "```json " + plan + " ```",  # single-line fence
        "```json\n" + plan,  # unterminated
        plan + "\n```",  # no opening fence
    )
    for text in rejected:
        with pytest.raises(StrictJSONError):
            load_model_json_emission(text.encode("utf-8"))


@pytest.mark.requires_loopback
def test_fenced_stage_emissions_are_normalized_and_recorded(tmp_path: Path) -> None:
    plan, placement = _plan(), _placement()
    responses = [
        (200, _api_body(_fenced(plan)), 0.0),
        (200, _api_body(_fenced(placement)), 0.0),
    ]
    with _Context(responses) as (endpoint, _handler):
        result = run_case_v3(
            output_root=tmp_path,
            model=_model_v3(endpoint),
            brief=_brief(),
            retriever=_Retriever(),
            stage_a_prompt="same stage A prompt",
            stage_c_prompt="same stage C prompt",
        )

    case = tmp_path / "brief_00"
    assert result["status"] == "complete"
    assert result["eligible_for_strict_one_shot_evaluation"] is True

    # The fenced original survives verbatim; the accepted artifact is de-fenced.
    assert (case / "object_plan_first_emission.json").read_text().startswith("```json")
    assert (
        case / "catalog_placement_first_emission.json"
    ).read_text().startswith("```json")
    assert json.loads((case / "object_plan.json").read_text()) == plan
    assert json.loads((case / "catalog_placement_v1.json").read_text()) == placement

    assert json.loads((case / "object_plan_validation.json").read_text()) == {
        "valid": True,
        "response_envelope": "single_json_code_fence_v1",
        "syntactic_normalization": True,
    }
    assert json.loads((case / "placement_validation.json").read_text()) == {
        "valid": True,
        "response_envelope": "single_json_code_fence_v1",
        "syntactic_normalization": True,
    }

    audit = json.loads((case / "one_shot_audit.json").read_text())
    # Honest: the frozen placement is no longer a byte copy of the emission.
    assert audit["first_emission_equals_frozen_placement"] is False
    # Stripping an envelope is not an edit to the emitted value.
    assert audit["post_emission_transform_edit_count"] == 0
    assert audit["generator_semantic_retry_count"] == 0

    freeze = json.loads((case / "generation_freeze.json").read_text())
    assert freeze["hashes"]["placement_first_emission"] != freeze["hashes"]["placement_frozen"]


@pytest.mark.requires_loopback
def test_raw_json_case_is_byte_identical_and_carries_no_envelope_keys(
    tmp_path: Path,
) -> None:
    plan_text = json.dumps(_plan(), separators=(",", ":"))
    placement_text = json.dumps(_placement(), separators=(",", ":"))
    responses = [
        (200, _api_body(plan_text), 0.0),
        (200, _api_body(placement_text), 0.0),
    ]
    with _Context(responses) as (endpoint, _handler):
        result = run_case_v3(
            output_root=tmp_path,
            model=_model_v3(endpoint),
            brief=_brief(),
            retriever=_Retriever(),
            stage_a_prompt="same stage A prompt",
            stage_c_prompt="same stage C prompt",
        )

    case = tmp_path / "brief_00"
    assert result["status"] == "complete"
    # Byte-for-byte what v2 would have written: the envelope keys are omitted on
    # the raw path so a re-run of an already-passing case keeps its artifact
    # hashes comparable with the published cohort.
    assert (case / "object_plan.json").read_text() == plan_text
    assert (case / "catalog_placement_v1.json").read_bytes() == (
        case / "catalog_placement_first_emission.json"
    ).read_bytes()
    assert json.loads((case / "object_plan_validation.json").read_text()) == {"valid": True}
    assert json.loads((case / "placement_validation.json").read_text()) == {"valid": True}
    audit = json.loads((case / "one_shot_audit.json").read_text())
    assert audit["first_emission_equals_frozen_placement"] is True


@pytest.mark.requires_loopback
def test_fence_with_surrounding_prose_is_still_rejected(tmp_path: Path) -> None:
    prose = "Here is the object plan:\n```json\n" + json.dumps(
        _plan(), separators=(",", ":")
    ) + "\n```"
    with _Context([(200, _api_body(prose), 0.0)]) as (endpoint, _handler):
        result = run_case_v3(
            output_root=tmp_path,
            model=_model_v3(endpoint),
            brief=_brief(),
            retriever=_Retriever(),
            stage_a_prompt="same stage A prompt",
            stage_c_prompt="same stage C prompt",
        )
    case = tmp_path / "brief_00"
    assert result["status"] == "stage_a_schema_invalid"
    validation = json.loads((case / "object_plan_validation.json").read_text())
    assert validation["valid"] is False
    assert validation["error_type"] == "StrictJSONError"
    assert not (case / "object_plan.json").exists()


@pytest.mark.requires_loopback
def test_v2_core_still_rejects_a_fenced_emission(tmp_path: Path) -> None:
    # The published scene cohort was generated by v2.  v3 exists precisely so
    # that v2 can keep this behaviour unchanged.
    with _Context([(200, _api_body(_fenced(_plan())), 0.0)]) as (endpoint, _handler):
        from tests.test_frozen_two_stage_generation_core import _model as _model_v2

        result = run_case_v2(
            output_root=tmp_path,
            model=_model_v2(endpoint),
            brief=_brief(),
            retriever=_Retriever(),
            stage_a_prompt="same stage A prompt",
            stage_c_prompt="same stage C prompt",
        )
    case = tmp_path / "brief_00"
    assert result["status"] == "stage_a_schema_invalid"
    validation = json.loads((case / "object_plan_validation.json").read_text())
    assert validation["error_type"] == "StrictJSONError"
    assert validation["error_message"].startswith("Expecting value: line 1 column 1")
