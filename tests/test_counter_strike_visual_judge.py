from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image
import pytest

from benchmark.game_scene.counter_strike import (
    GLOBAL_EVIDENCE_ROLE,
    REGIONAL_EVIDENCE_ROLE,
    CounterStrikeEvidenceDescriptor,
    CounterStrikeFrozenEvidence,
    load_counter_strike_benchmark_config,
)
from benchmark.game_scene.counter_strike.judge import (
    CounterStrikeVisualJudge,
    CounterStrikeVisualJudgeError,
)


ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_CONFIG = ROOT / "configs" / "game" / "counter_strike" / "benchmark_v1.yaml"
REQUIRED_ROLES = (
    "team_a_spawn",
    "team_b_spawn",
    "preparation",
    "main_engagement",
    "flank",
)


class _FakeModel:
    model_id = "fake-cs-vlm"

    def __init__(
        self,
        responses: dict[str, list[dict[str, Any]]],
        *,
        request_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.responses = {
            key: list(values) for key, values in responses.items()
        }
        self.request_metadata = dict(request_metadata or {})
        self.last_request_metadata: dict[str, Any] = {}
        self.calls: list[dict[str, Any]] = []

    def chat_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format_json: bool,
        call_type: str,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "response_format_json": response_format_json,
                "call_type": call_type,
            }
        )
        self.last_request_metadata = dict(self.request_metadata)
        queue = self.responses.get(call_type)
        if not queue:
            raise AssertionError(f"unexpected model call {call_type!r}")
        return json.dumps(queue.pop(0))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_png(path: Path, color: tuple[int, int, int]) -> Path:
    Image.new("RGB", (8, 8), color).save(path)
    return path


def _evidence(tmp_path: Path) -> tuple[CounterStrikeFrozenEvidence, Path]:
    capture = tmp_path / "capture"
    capture.mkdir()
    global_views = []
    regional_views = []
    for index in range(2):
        path = _write_png(
            capture / f"global_{index}.png",
            (30 + index, 40, 50),
        )
        global_views.append(
            CounterStrikeEvidenceDescriptor(
                id=f"global_oblique_{index:02d}",
                role=GLOBAL_EVIDENCE_ROLE,
                path=path,
                sha256=_sha256(path),
            )
        )
    for index in range(4):
        path = _write_png(
            capture / f"regional_{index}.png",
            (60, 70 + index, 80),
        )
        regional_views.append(
            CounterStrikeEvidenceDescriptor(
                id=f"style_region_{index:02d}",
                role=REGIONAL_EVIDENCE_ROLE,
                path=path,
                sha256=_sha256(path),
            )
        )
    manifest = capture / "render_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    topology = _write_png(tmp_path / "topology.png", (255, 255, 255))
    return (
        CounterStrikeFrozenEvidence(
            capture_dir=capture,
            manifest_path=manifest,
            manifest_sha256=_sha256(manifest),
            global_views=tuple(global_views),
            regional_views=tuple(regional_views),
        ),
        topology,
    )


def _zone_roles(*, missing: str | None = None) -> list[dict[str, str]]:
    return [
        {
            "role": role,
            "status": "missing" if role == missing else "clear",
            "evidence": f"visible geometry for {role}",
        }
        for role in REQUIRED_ROLES
    ]


def _zone_response(
    *,
    verdict: str = "valid",
    score: float = 0.8,
    roles: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "evidence_status": "sufficient",
        "verdict": verdict,
        "score": score,
        "reason": "All required spatial roles are legible.",
        "defects": (
            [] if verdict != "invalid" else ["main engagement role is unclear"]
        ),
        "missing_evidence": [],
        "role_findings": roles if roles is not None else _zone_roles(),
    }


def _insufficient_response() -> dict[str, Any]:
    return {
        "evidence_status": "insufficient",
        "verdict": "ambiguous",
        "score": None,
        "reason": "The global views do not expose the transition space.",
        "defects": [],
        "missing_evidence": ["spawn transition geometry"],
        "role_findings": [],
    }


def _landmark_response(
    *,
    count: int = 3,
    regions: list[str] | None = None,
) -> dict[str, Any]:
    region_names = regions or [f"region_{index}" for index in range(count)]
    return {
        "evidence_status": "sufficient",
        "verdict": "valid",
        "score": 0.8,
        "reason": "Three distinct visible cues can be reused as callouts.",
        "defects": [],
        "missing_evidence": [],
        "landmarks": [
            {
                "name": f"landmark_{index}",
                "view_ids": ["global_oblique_00"],
                "visible_cue": f"distinct cue {index}",
                "spatial_region": region_names[index],
            }
            for index in range(count)
        ],
    }


def _cover_response(*, count: int = 4) -> dict[str, Any]:
    heights = ["waist", "standing", "low", "tall"]
    widths = ["narrow", "wide", "medium", "wide"]
    arrangements = ["isolated", "paired", "row", "compound"]
    return {
        "evidence_status": "sufficient",
        "verdict": "valid",
        "score": 0.8,
        "reason": "Four visibly distinct cover configurations are present.",
        "defects": [],
        "missing_evidence": [],
        "cover_findings": [
            {
                "form_name": f"cover_form_{index}",
                "view_ids": ["global_oblique_00"],
                "visible_cue": f"visible cover cue {index}",
                "spatial_region": f"region_{index}",
                "height_profile": heights[index],
                "width_profile": widths[index],
                "arrangement": arrangements[index],
            }
            for index in range(count)
        ],
    }


def _judge(
    model: _FakeModel,
    *,
    selector_model: _FakeModel | None = None,
) -> CounterStrikeVisualJudge:
    return CounterStrikeVisualJudge(
        model,
        benchmark_config=load_counter_strike_benchmark_config(
            BENCHMARK_CONFIG
        ),
        selector_model=selector_model,
    )


def _neutral_context(metric: str) -> dict[str, Any]:
    return {
        "schema_version": "counter_strike_neutral_visual_context_v1",
        "metric": metric,
        "scope": "static_3d_environment_only",
        "observation_aid": {
            "shows": [
                "walkable_free_space",
                "blocking_footprints",
                "declared_team_a_spawn_points",
                "declared_team_b_spawn_points",
            ],
            "omits": [
                "deterministic_scores",
                "deterministic_verdicts",
                "inferred_zone_roles",
                "inferred_routes",
                "cover_proposals",
                "engagement_anchor",
                "case_identity",
            ],
            "ground_truth": False,
        },
    }


def test_global_sufficient_packet_resolves_without_selector(
    tmp_path: Path,
) -> None:
    evidence, topology = _evidence(tmp_path)
    response = _zone_response()
    model = _FakeModel(
        {"cs_static_design.zone_clarity": [response, response]}
    )

    result = _judge(model).judge_metric(
        "zone_clarity",
        evidence=evidence,
        topology_diagram=topology,
        topology_context=_neutral_context("zone_clarity"),
    )

    assert result.status == "checked"
    assert result.verdict == "valid"
    assert result.score == pytest.approx(0.8)
    assert result.evidence_phase == "global"
    assert result.selected_regional_ids == ()
    assert [call["call_type"] for call in model.calls] == [
        "cs_static_design.zone_clarity",
        "cs_static_design.zone_clarity",
    ]


def test_insufficient_global_packet_uses_frozen_regional_selector_then_rejudges(
    tmp_path: Path,
) -> None:
    evidence, topology = _evidence(tmp_path)
    insufficient = _insufficient_response()
    sufficient = _zone_response()
    model = _FakeModel(
        {
            "cs_static_design.zone_clarity": [
                insufficient,
                insufficient,
                sufficient,
                sufficient,
            ]
        }
    )
    selector_model = _FakeModel(
        {
            "cs_static_design.selector.zone_clarity": [
                {
                    "evidence_status": "selected",
                    "selected_view_ids": ["style_region_02"],
                    "reason": "This view exposes the missing transition.",
                }
            ],
        }
    )

    result = _judge(model, selector_model=selector_model).judge_metric(
        "zone_clarity",
        evidence=evidence,
        topology_diagram=topology,
        topology_context=_neutral_context("zone_clarity"),
    )

    assert result.status == "checked"
    assert result.evidence_phase == "global_plus_regional_fallback"
    assert result.selected_regional_ids == ("style_region_02",)
    assert result.selector is not None
    assert result.selector["selected_view_ids"] == ["style_region_02"]
    assert [call["call_type"] for call in model.calls] == [
        "cs_static_design.zone_clarity",
        "cs_static_design.zone_clarity",
        "cs_static_design.zone_clarity",
        "cs_static_design.zone_clarity",
    ]
    assert [call["call_type"] for call in selector_model.calls] == [
        "cs_static_design.selector.zone_clarity"
    ]


@pytest.mark.parametrize(
    ("selector_response", "expected_code"),
    [
        (
            {
                "evidence_status": "selected",
                "selected_view_ids": ["style_region_00"],
                "reason": "useful",
                "verdict": "valid",
            },
            "selector_returned_verdict",
        ),
        (
            {
                "evidence_status": "selected",
                "selected_view_ids": ["not_in_frozen_bank"],
                "reason": "useful",
            },
            "selector_unknown_view_id",
        ),
    ],
)
def test_selector_cannot_return_verdict_or_unknown_ids(
    tmp_path: Path,
    selector_response: dict[str, Any],
    expected_code: str,
) -> None:
    evidence, topology = _evidence(tmp_path)
    insufficient = _insufficient_response()
    model = _FakeModel(
        {
            "cs_static_design.zone_clarity": [
                insufficient,
                insufficient,
            ]
        }
    )
    selector_model = _FakeModel(
        {
            "cs_static_design.selector.zone_clarity": [
                selector_response
            ],
        }
    )

    with pytest.raises(CounterStrikeVisualJudgeError) as caught:
        _judge(model, selector_model=selector_model).judge_metric(
            "zone_clarity",
            evidence=evidence,
            topology_diagram=topology,
            topology_context=_neutral_context("zone_clarity"),
        )

    assert caught.value.code == expected_code


def test_repeat_verdict_disagreement_is_unresolved_not_majority_vote(
    tmp_path: Path,
) -> None:
    evidence, topology = _evidence(tmp_path)
    valid = _zone_response()
    invalid = _zone_response(
        verdict="invalid",
        score=0.2,
        roles=_zone_roles(missing="main_engagement"),
    )
    model = _FakeModel(
        {"cs_static_design.zone_clarity": [valid, invalid]}
    )

    result = _judge(model).judge_metric(
        "zone_clarity",
        evidence=evidence,
        topology_diagram=topology,
        topology_context=_neutral_context("zone_clarity"),
    )

    assert result.status == "unresolved"
    assert result.verdict == "ambiguous"
    assert result.score is None
    assert result.to_dict()["repeat_agreement"] is False


def test_valid_zone_result_requires_configured_clear_role_count(
    tmp_path: Path,
) -> None:
    evidence, topology = _evidence(tmp_path)
    roles = _zone_roles()
    roles[0]["status"] = "weak"
    roles[1]["status"] = "weak"
    response = _zone_response(roles=roles)
    model = _FakeModel(
        {"cs_static_design.zone_clarity": [response, response]}
    )

    with pytest.raises(CounterStrikeVisualJudgeError) as caught:
        _judge(model).judge_metric(
            "zone_clarity",
            evidence=evidence,
            topology_diagram=topology,
            topology_context=_neutral_context("zone_clarity"),
        )

    assert caught.value.code == "metric_response_schema_invalid"
    assert "at least 4 clear roles" in str(caught.value)


def test_one_weakly_read_role_still_permits_a_valid_zone_result(
    tmp_path: Path,
) -> None:
    """The bar is a majority of clear roles, not unanimity.

    A role that is present but not delimited by its own geometry is what
    ``weak`` exists to record.  If one of those sank the verdict the status
    would be indistinguishable from ``missing``.
    """

    evidence, topology = _evidence(tmp_path)
    roles = _zone_roles()
    roles[-1]["status"] = "weak"
    response = _zone_response(roles=roles)
    model = _FakeModel(
        {"cs_static_design.zone_clarity": [response, response]}
    )

    result = _judge(model).judge_metric(
        "zone_clarity",
        evidence=evidence,
        topology_diagram=topology,
        topology_context=_neutral_context("zone_clarity"),
    )

    assert result.status == "checked"
    assert result.verdict == "valid"


def test_zone_prompt_states_the_bar_the_response_is_validated_against(
    tmp_path: Path,
) -> None:
    """The judge is told the same count that validation will enforce.

    Rejecting a response for a rule it was never given would spend a repeat on
    a question the judge had no way to answer.
    """

    evidence, topology = _evidence(tmp_path)
    response = _zone_response()
    model = _FakeModel(
        {"cs_static_design.zone_clarity": [response, response]}
    )

    _judge(model).judge_metric(
        "zone_clarity",
        evidence=evidence,
        topology_diagram=topology,
        topology_context=_neutral_context("zone_clarity"),
    )

    configured = load_counter_strike_benchmark_config(BENCHMARK_CONFIG).raw[
        "l4_metrics"
    ]["zone_clarity"]["min_clear_roles"]
    prompt = json.dumps(model.calls[0]["messages"])
    assert f"at least {configured} of the five roles" in prompt


def test_valid_landmark_result_requires_three_distinct_named_regions(
    tmp_path: Path,
) -> None:
    evidence, topology = _evidence(tmp_path)
    too_few = _landmark_response(count=2)
    model = _FakeModel(
        {
            "cs_static_design.landmark_legibility": [
                too_few,
                too_few,
            ]
        }
    )

    with pytest.raises(CounterStrikeVisualJudgeError) as caught:
        _judge(model).judge_metric(
            "landmark_legibility",
            evidence=evidence,
            topology_diagram=topology,
            topology_context=_neutral_context("landmark_legibility"),
        )

    assert caught.value.code == "metric_response_schema_invalid"
    assert "distinct named landmarks" in str(caught.value)


def test_valid_cover_result_requires_structured_visible_diversity(
    tmp_path: Path,
) -> None:
    evidence, topology = _evidence(tmp_path)
    response = _cover_response()
    model = _FakeModel(
        {"cs_static_design.cover_diversity": [response, response]}
    )

    result = _judge(model).judge_metric(
        "cover_diversity",
        evidence=evidence,
        topology_diagram=topology,
        topology_context=_neutral_context("cover_diversity"),
    )

    assert result.status == "checked"
    assert result.verdict == "valid"
    assert result.score == pytest.approx(0.8)
    assert len(result.repeats[0]["cover_findings"]) == 4


def test_visual_judge_rejects_deterministic_score_context(
    tmp_path: Path,
) -> None:
    evidence, topology = _evidence(tmp_path)
    response = _zone_response()
    model = _FakeModel(
        {"cs_static_design.zone_clarity": [response, response]}
    )

    with pytest.raises(CounterStrikeVisualJudgeError) as caught:
        _judge(model).judge_metric(
            "zone_clarity",
            evidence=evidence,
            topology_diagram=topology,
            topology_context={
                "deterministic_metrics": {
                    "zone_clarity": {"score": 1.0, "verdict": "valid"}
                }
            },
        )

    assert caught.value.code == "topology_context_invalid"
    assert model.calls == []


def test_zone_result_requires_exactly_one_finding_per_required_role(
    tmp_path: Path,
) -> None:
    evidence, topology = _evidence(tmp_path)
    duplicated = _zone_roles() + [_zone_roles()[0]]
    response = _zone_response(roles=duplicated)
    model = _FakeModel(
        {"cs_static_design.zone_clarity": [response, response]}
    )

    with pytest.raises(CounterStrikeVisualJudgeError) as caught:
        _judge(model).judge_metric(
            "zone_clarity",
            evidence=evidence,
            topology_diagram=topology,
            topology_context=_neutral_context("zone_clarity"),
        )

    assert caught.value.code == "metric_response_schema_invalid"
    assert "exactly one finding" in str(caught.value)


def test_result_metadata_is_allowlisted_and_never_contains_credentials_or_paths(
    tmp_path: Path,
) -> None:
    evidence, topology = _evidence(tmp_path)
    response = _zone_response()
    secret = "super-secret-value"
    metadata = {
        "endpoint": (
            f"https://user:{secret}@example.test/v1?api_key={secret}"
        ),
        "model": "fake-cs-vlm",
        "api_key_env": "OPENAI_API_KEY",
        "authorization_configured": True,
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "secret": secret,
        },
        "headers": {"Authorization": f"Bearer {secret}"},
        "api_key": secret,
        "local_path": "/private/sensitive/capture.png",
    }
    model = _FakeModel(
        {"cs_static_design.zone_clarity": [response, response]},
        request_metadata=metadata,
    )

    result = _judge(model).judge_metric(
        "zone_clarity",
        evidence=evidence,
        topology_diagram=topology,
        topology_context=_neutral_context("zone_clarity"),
    )

    serialized = json.dumps(result.to_dict(), sort_keys=True)
    assert secret not in serialized
    assert "/private/sensitive" not in serialized
    safe = result.repeats[0]["request_metadata"]
    assert safe["endpoint"] == "https://example.test/v1"
    assert safe["api_key_env"] == "OPENAI_API_KEY"
    assert safe["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    assert "headers" not in safe
    # Outbound messages use anonymized IDs and flattened data URLs, never
    # local file-system paths.
    messages = json.dumps(model.calls[0]["messages"])
    assert evidence.global_views[0].path.as_posix() not in messages
    assert topology.as_posix() not in messages


def test_judge_revalidates_frozen_evidence_hash_before_model_call(
    tmp_path: Path,
) -> None:
    evidence, topology = _evidence(tmp_path)
    evidence.global_views[0].path.write_bytes(b"tampered after evidence load")
    model = _FakeModel(
        {
            "cs_static_design.zone_clarity": [
                _zone_response(),
                _zone_response(),
            ]
        }
    )

    with pytest.raises(CounterStrikeVisualJudgeError) as caught:
        _judge(model).judge_metric(
            "zone_clarity",
            evidence=evidence,
            topology_diagram=topology,
            topology_context=_neutral_context("zone_clarity"),
        )

    assert caught.value.code == "evidence_hash_mismatch"
    assert model.calls == []
