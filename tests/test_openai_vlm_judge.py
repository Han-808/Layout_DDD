from __future__ import annotations

import base64
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

from benchmark.visual_judge import OpenAICompatibleVLMJudge, build_openai_compatible_vlm_judge, evaluate_vlm_category
from benchmark.visual_judge.openai_compatible import (
    _select_judge_visual_paths,
    _selector_candidate_order_key,
)
from benchmark.visual_judge.contracts import ResponseSchemaRepairError
from benchmark.visual_judge.l3_prompts import (
    L3_METRIC_BOUNDARY_RULES,
    L3_METRIC_PROMPT_VERSION,
)
from benchmark.visual_judge.runtime import EvidenceControlUnresolvedError


class FakeMultimodalModel:
    model_id = "Qwen3-VL-32B-Instruct-64K"
    endpoint = "http://127.0.0.1:8298/v1"

    def __init__(self, response: dict | list[dict]) -> None:
        self.responses = (
            list(response) if isinstance(response, list) else [response]
        )
        self.calls = []
        self.last_request_metadata = {"image_count": 1}

    def chat_messages(self, messages, **kwargs) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        self.last_request_metadata = {
            "image_count": 1,
            "call_type": kwargs.get("call_type"),
        }
        return json.dumps(self.responses[index])


def test_ordinary_rejudge_keeps_global_anchor_and_latest_repair_views() -> None:
    paths = [Path(f"/tmp/image_{index}.png") for index in range(8)]

    selected, audit = _select_judge_visual_paths(
        paths,
        max_images=3,
        forced_choice=False,
    )

    assert selected == [paths[0], paths[6], paths[7]]
    assert audit["visual_selection_policy"] == (
        "global_anchor_plus_most_recent"
    )


def _request(image_path: Path) -> dict:
    return {
        "category": "visual_quality",
        "prompt": None,
        "scene_summary": {
            "scene_id": "scene",
            "objects": [{"id": "bed", "rotation_degrees": [0, 0, 5]}],
        },
        "deterministic_evidence": {"generic_validity": {"score": 0.9}},
        "render_evidence": [str(image_path)],
    }


def _write_test_png(path: Path, *, private_text: str | None = None) -> None:
    metadata = None
    if private_text is not None:
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("private_source", private_text)
    image = Image.new("RGBA", (4, 3), (12, 34, 56, 128))
    image.putpixel((0, 0), (80, 90, 100, 255))
    image.save(path, pnginfo=metadata)


def test_openai_compatible_vlm_judge_sends_images_and_structured_prior(tmp_path: Path) -> None:
    image_path = tmp_path / "standardized_top.png"
    _write_test_png(image_path, private_text="must-not-leave-process")
    model = FakeMultimodalModel(
        {"applicable": True, "score": 0.75, "confidence": 0.8, "summary": "plausible", "issues": [], "evidence": ["top view"]}
    )
    judge = OpenAICompatibleVLMJudge(model)
    request = _request(image_path)
    original_request = deepcopy(request)

    result = judge.evaluate(request)

    content = model.calls[0]["messages"][1]["content"]
    assert content[0]["type"] == "text"
    assert "generic_validity" in content[0]["text"]
    context = json.loads(content[0]["text"].split("\n", 1)[1])
    assert context["vlm_role"] == "judge"
    assert context["decision_contract"] == "generic_visual_score_v1"
    assert context["judge_method"] == "evaluate"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    encoded = content[1]["image_url"]["url"].split(",", 1)[1]
    with Image.open(BytesIO(base64.b64decode(encoded))) as normalized:
        assert normalized.mode == "RGB"
        assert normalized.info == {}
    assert model.calls[0]["kwargs"]["response_format_json"] is True
    assert result["score"] == 0.75
    assert result["model"] == "Qwen3-VL-32B-Instruct-64K"
    assert result["vlm_role"] == "judge"
    assert result["decision_contract"] == "generic_visual_score_v1"
    assert result["judge_method"] == "evaluate"
    assert request == original_request


def test_generic_visual_evaluator_rejects_out_of_range_score(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "generic.png"
    _write_test_png(image_path)
    judge = OpenAICompatibleVLMJudge(
        FakeMultimodalModel(
            {
                "applicable": True,
                "score": 1.1,
                "confidence": 0.8,
                "summary": "invalid range",
                "issues": [],
                "evidence": [],
            }
        )
    )

    with pytest.raises(ValueError, match="score must be between 0 and 1"):
        judge.evaluate(_request(image_path))


def test_direct_generic_public_method_does_not_call_model_without_evidence():
    model = FakeMultimodalModel(
        {
            "applicable": True,
            "score": 1.0,
            "confidence": 1.0,
            "summary": "must not run",
            "issues": [],
            "evidence": [],
        }
    )

    result = OpenAICompatibleVLMJudge(model).evaluate(
        {
            "category": "visual_quality",
            "render_evidence": [],
        }
    )

    assert result["applicable"] is False
    assert result["score"] is None
    assert model.calls == []


def test_direct_canonical_public_method_returns_compatible_unresolved_without_model():
    model = FakeMultimodalModel(
        {
            "evidence_status": "sufficient",
            "verdict": "valid",
            "confidence": 1.0,
            "reason": "must not run",
            "missing_evidence": [],
            "defects": [],
        }
    )

    result = OpenAICompatibleVLMJudge(model).adjudicate_scene_quality(
        {
            "metric": "style_consistency",
            "render_evidence": [],
        }
    )

    assert result["evidence_status"] == "insufficient"
    assert result["verdict"] == "ambiguous"
    assert result["missing_evidence"]
    assert model.calls == []


@pytest.mark.parametrize(
    "metric",
    [
        "scale_consistency",
        "object_pairing_consistency",
    ],
)
def test_explicit_json_screen_calls_model_without_images(metric):
    model = FakeMultimodalModel(
        {
            "evidence_status": "sufficient",
            "verdict": "valid",
            "confidence": 0.9,
            "reason": "The structured scene data raises no concern.",
            "missing_evidence": [],
            "defects": [],
            "evidence_request": None,
        }
    )
    judge = OpenAICompatibleVLMJudge(model)

    result = judge.screen_scene_quality(
        {
            "metric": metric,
            "evidence_phase": "json_screen",
            "decision_mode": "screen",
            "target_object_ids": ["chair", "desk"],
            "judgment_scope": {
                "included": [
                    (
                        "significant_visible_category_relative_"
                        "scale_incoherence"
                    )
                    if metric == "scale_consistency"
                    else "group_member_category_compatibility"
                ]
            },
            "scene_summary": {
                "scene_type": "bedroom",
                "objects": [
                    {
                        "id": "chair",
                        "category": "chair",
                        "size": [0.5, 0.5, 0.9],
                    },
                    {
                        "id": "desk",
                        "category": "desk",
                        "size": [1.2, 0.6, 0.75],
                    },
                ],
            },
            "render_evidence": [],
        }
    )

    assert result["verdict"] == "valid"
    assert result["images_used"] == []
    assert len(model.calls) == 1
    content = model.calls[0]["messages"][1]["content"]
    assert len(content) == 1
    context = json.loads(content[0]["text"].split("\n", 1)[1])
    assert context["evidence_phase"] == "json_screen"
    assert context["decision_mode"] == "screen"
    assert "structured-data routing screen" in (
        context["phase_instruction"]
    )
    if metric == "scale_consistency":
        assert "physical dimensions remain materially suspicious" in (
            context["phase_instruction"]
        )
        assert "final scale defect" in context["phase_instruction"]
    else:
        assert "Apply the relocation test" in (
            context["phase_instruction"]
        )
        assert "final pairing defect" in context["phase_instruction"]
    assert "Additional visual evidence can be acquired" in (
        context["phase_instruction"]
    )
    system_prompt = model.calls[0]["messages"][0]["content"]
    assert "Additional visual evidence can be acquired" in system_prompt
    assert "Follow this decision order" in system_prompt
    assert "rearranging the authored layout" in system_prompt
    assert "do not default to valid" not in system_prompt
    assert "instead of guessing" not in context["phase_instruction"]
    assert model.calls[0]["kwargs"]["call_type"] == (
        f"vlm_judge.screen.{metric}"
    )


@pytest.mark.parametrize(
    ("metric", "required_text", "forbidden_text"),
    [
        (
            "scale_consistency",
            "physical-size mismatch remains material",
            "relocation test again",
        ),
        (
            "object_pairing_consistency",
            "Apply the relocation test again",
            "physical-size mismatch remains material",
        ),
    ],
)
def test_visual_confirmation_phase_instructions_are_metric_specific(
    tmp_path: Path,
    metric: str,
    required_text: str,
    forbidden_text: str,
) -> None:
    image_path = tmp_path / f"{metric}.png"
    _write_test_png(image_path)
    model = FakeMultimodalModel(
        {
            "evidence_status": "sufficient",
            "verdict": "valid",
            "confidence": 0.8,
            "reason": "The visual candidate is valid.",
            "missing_evidence": [],
            "defects": [],
            "evidence_request": None,
        }
    )

    OpenAICompatibleVLMJudge(model).adjudicate_scene_quality(
        {
            "metric": metric,
            "evidence_phase": "visual_confirmation",
            "decision_mode": "final",
            "target_object_ids": ["chair"],
            "judgment_scope": {"included": []},
            "scene_summary": {
                "scene_type": "room",
                "objects": [{"id": "chair", "category": "chair"}],
            },
            "render_evidence": [str(image_path)],
        }
    )

    context = json.loads(
        model.calls[0]["messages"][1]["content"][0]["text"].split(
            "\n", 1
        )[1]
    )
    assert required_text in context["phase_instruction"]
    assert forbidden_text not in context["phase_instruction"]


def test_canonical_json_screen_gets_one_same_evidence_schema_repair() -> None:
    invalid_scope = {
        "evidence_status": "sufficient",
        "verdict": "invalid",
        "confidence": 0.8,
        "reason": "The chair is materially undersized for its category.",
        "missing_evidence": [],
        "defects": [
            {
                "scope": "functional_access_obstruction",
                "target_ids": ["chair"],
                "relation": "category-relative physical scale",
                "reason": "The chair is materially undersized.",
            }
        ],
        "evidence_request": None,
    }
    repaired = {
        "evidence_status": "sufficient",
        "verdict": "invalid",
        "confidence": 0.8,
        "reason": "The chair is materially undersized for its category.",
        "missing_evidence": [],
        "defects": [
            {
                "scope": (
                    "significant_visible_category_relative_"
                    "scale_incoherence"
                ),
                "target_ids": ["chair"],
                "relation": "category-relative physical scale",
                "reason": "The chair is materially undersized.",
            }
        ],
        "evidence_request": None,
    }
    model = FakeMultimodalModel([invalid_scope, repaired])
    judge = OpenAICompatibleVLMJudge(model)

    result = judge.screen_scene_quality(
        {
            "metric": "scale_consistency",
            "evidence_phase": "json_screen",
            "decision_mode": "screen",
            "target_object_ids": ["chair"],
            "judgment_scope": {
                "included": [
                    "significant_visible_category_relative_scale_incoherence"
                ]
            },
            "scene_summary": {
                "scene_type": "office",
                "objects": [
                    {
                        "id": "chair",
                        "category": "chair",
                        "size": [0.6, 0.6, 1.0],
                    }
                ],
            },
            "render_evidence": [],
        }
    )

    assert result["verdict"] == "invalid"
    assert len(model.calls) == 2
    assert model.calls[1]["kwargs"]["call_type"] == (
        "vlm_judge.screen.scale_consistency.schema_repair"
    )
    assert model.calls[1]["messages"][:2] == model.calls[0]["messages"]
    assert model.calls[1]["messages"][2] == {
        "role": "assistant",
        "content": json.dumps(invalid_scope),
    }
    repair_prompt = model.calls[1]["messages"][3]["content"]
    assert "Judge only the requested metric" in repair_prompt
    assert "out-of-scope issue" in repair_prompt
    assert (
        'allowed defect scopes: '
        '["significant_visible_category_relative_scale_incoherence"]'
        in repair_prompt
    )
    audit = result["request_metadata"]["response_schema_validation"]
    assert audit["policy"] == "single_canonical_schema_repair_retry_v1"
    assert audit["attempt_count"] == 2
    assert audit["repair_retry_count"] == 1
    assert audit["recovered"] is True
    assert audit["semantic_preservation"]["locked_fields"] == {
        "evidence_status": "sufficient",
        "verdict": "invalid",
        "defect_count": 1,
        "defect_target_sets": (("chair",),),
    }
    assert audit["semantic_preservation"][
        "restored_natural_language_fields"
    ] == []


def test_canonical_schema_repair_rejects_semantic_verdict_change() -> None:
    invalid_scope = {
        "evidence_status": "sufficient",
        "verdict": "invalid",
        "confidence": 0.8,
        "reason": "The floor lamp is materially undersized.",
        "missing_evidence": [],
        "defects": [
            {
                "scope": "scale_consistency",
                "target_ids": ["floor_lamp"],
                "relation": "category-relative height",
                "reason": "The object is materially too short.",
            }
        ],
        "evidence_request": None,
    }
    changed_to_ambiguous = {
        "evidence_status": "insufficient",
        "verdict": "ambiguous",
        "confidence": 0.5,
        "reason": "A visual comparison is required.",
        "missing_evidence": ["target_visible"],
        "defects": [],
        "evidence_request": {
            "target_ids": ["floor_lamp"],
            "missing_observations": ["target_visible"],
            "view_goal": "show the floor lamp",
            "metadata": {},
        },
    }
    model = FakeMultimodalModel(
        [invalid_scope, changed_to_ambiguous]
    )

    with pytest.raises(ResponseSchemaRepairError) as raised:
        OpenAICompatibleVLMJudge(model).screen_scene_quality(
            {
                "metric": "scale_consistency",
                "evidence_phase": "json_screen",
                "decision_mode": "screen",
                "target_object_ids": ["floor_lamp"],
                "judgment_scope": {
                    "included": [
                        (
                            "significant_visible_category_relative_"
                            "scale_incoherence"
                        )
                    ]
                },
                "scene_summary": {
                    "scene_type": "lounge",
                    "objects": [
                        {
                            "id": "floor_lamp",
                            "category": "floor_lamp",
                            "size": [0.3, 0.3, 1.1],
                        }
                    ],
                },
                "render_evidence": [],
            }
        )

    assert "changed locked semantic fields" in str(raised.value)
    assert raised.value.schema_audit["recovered"] is False


def test_canonical_schema_repair_restores_original_explanation() -> None:
    invalid_scope = {
        "evidence_status": "sufficient",
        "verdict": "invalid",
        "confidence": 0.8,
        "reason": "The chair is materially undersized for its category.",
        "missing_evidence": [],
        "defects": [
            {
                "scope": "wrong_scope_token",
                "target_ids": ["chair"],
                "relation": "category-relative physical scale",
                "reason": "The chair is materially undersized.",
            }
        ],
        "evidence_request": None,
    }
    changed_explanation = {
        "evidence_status": "sufficient",
        "verdict": "invalid",
        "confidence": 0.8,
        "reason": "The chair blocks access to the room.",
        "missing_evidence": [],
        "defects": [
            {
                "scope": (
                    "significant_visible_category_relative_"
                    "scale_incoherence"
                ),
                "target_ids": ["chair"],
                "relation": "circulation access",
                "reason": "The chair blocks the doorway.",
            }
        ],
        "evidence_request": None,
    }
    model = FakeMultimodalModel(
        [invalid_scope, changed_explanation]
    )

    result = OpenAICompatibleVLMJudge(model).screen_scene_quality(
        {
            "metric": "scale_consistency",
            "evidence_phase": "json_screen",
            "decision_mode": "screen",
            "target_object_ids": ["chair"],
            "judgment_scope": {
                "included": [
                    (
                        "significant_visible_category_relative_"
                        "scale_incoherence"
                    )
                ]
            },
            "scene_summary": {
                "scene_type": "office",
                "objects": [
                    {
                        "id": "chair",
                        "category": "chair",
                        "size": [0.6, 0.6, 1.0],
                    }
                ],
            },
            "render_evidence": [],
        }
    )

    assert result["verdict"] == "invalid"
    assert result["reason"] == invalid_scope["reason"]
    assert result["defects"] == [
        {
            "scope": (
                "significant_visible_category_relative_"
                "scale_incoherence"
            ),
            "target_ids": ["chair"],
            "relation": "category-relative physical scale",
            "reason": "The chair is materially undersized.",
        }
    ]
    audit = result["request_metadata"]["response_schema_validation"]
    assert audit["recovered"] is True
    assert audit["semantic_preservation"][
        "restored_natural_language_fields"
    ] == [
        "reason",
        "defects[0].relation",
        "defects[0].reason",
    ]


def test_canonical_schema_repair_rejects_structured_claim_change(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "functional-repair.png"
    _write_test_png(image_path)
    initial = {
        "evidence_status": "sufficient",
        "verdict": "invalid",
        "confidence": 2.0,
        "reason": "The cabinet door cannot open.",
        "missing_evidence": [],
        "defects": [
            {
                "scope": "opening_clearance",
                "target_ids": ["cabinet"],
                "relation": "door opening clearance",
                "reason": "The cabinet door is blocked.",
            }
        ],
        "evidence_request": None,
    }
    changed_scope = {
        **initial,
        "confidence": 0.8,
        "defects": [
            {
                "scope": "circulation",
                "target_ids": ["cabinet"],
                "relation": "walking clearance",
                "reason": "The cabinet blocks circulation.",
            }
        ],
    }
    model = FakeMultimodalModel([initial, changed_scope])

    with pytest.raises(ResponseSchemaRepairError) as raised:
        OpenAICompatibleVLMJudge(model).screen_scene_quality(
            {
                "metric": "functional_consistency",
                "evidence_phase": "json_screen",
                "decision_mode": "screen",
                "target_object_ids": ["cabinet"],
                "judgment_scope": {
                    "included": [
                        "opening_clearance",
                        "circulation",
                    ]
                },
                "scene_summary": {
                    "objects": [
                        {"id": "cabinet", "category": "cabinet"}
                    ]
                },
                "render_evidence": [str(image_path)],
            }
        )

    assert "changed locked semantic fields" in str(raised.value)
    assert raised.value.schema_audit["recovered"] is False


def test_canonical_schema_repair_fails_closed_after_one_retry() -> None:
    invalid = {
        "evidence_status": "sufficient",
        "verdict": "invalid",
        "confidence": 0.8,
        "reason": "Still outside the requested metric.",
        "missing_evidence": [],
        "defects": [
            {
                "scope": "functional_access_obstruction",
                "target_ids": ["chair"],
                "relation": "access",
                "reason": "This remains outside scale consistency.",
            }
        ],
        "evidence_request": None,
    }
    model = FakeMultimodalModel(
        [
            invalid,
            invalid,
            {
                "evidence_status": "sufficient",
                "verdict": "valid",
                "confidence": 1.0,
                "reason": "must never be called",
                "missing_evidence": [],
                "defects": [],
                "evidence_request": None,
            },
        ]
    )
    judge = OpenAICompatibleVLMJudge(model)

    with pytest.raises(ResponseSchemaRepairError) as raised:
        judge.screen_scene_quality(
            {
                "metric": "scale_consistency",
                "evidence_phase": "json_screen",
                "decision_mode": "screen",
                "target_object_ids": ["chair"],
                "judgment_scope": {
                    "included": [
                        (
                            "significant_visible_category_relative_"
                            "scale_incoherence"
                        )
                    ]
                },
                "scene_summary": {
                    "scene_type": "office",
                    "objects": [
                        {
                            "id": "chair",
                            "category": "chair",
                            "size": [0.6, 0.6, 1.0],
                        }
                    ],
                },
                "render_evidence": [],
            }
        )

    assert len(model.calls) == 2
    assert raised.value.schema_audit["repair_retry_count"] == 1
    assert raised.value.schema_audit["recovered"] is False


@pytest.mark.parametrize(
    ("method_name", "request_payload"),
    [
        (
            "adjudicate_relation",
            {
                "family": "oor",
                "relation": {
                    "subject_id": "a",
                    "target_id": "b",
                },
                "render_evidence": [],
            },
        ),
        (
            "adjudicate_spatial_fidelity",
            {
                "metric": "scale",
                "event": {"object_id": "a"},
                "render_evidence": [],
            },
        ),
    ],
)
def test_direct_binary_public_methods_fail_closed_without_evidence(
    method_name: str,
    request_payload: dict,
):
    model = FakeMultimodalModel(
        {
            "verdict": "valid",
            "confidence": 1.0,
            "reason": "must not run",
        }
    )
    judge = OpenAICompatibleVLMJudge(model)

    with pytest.raises(EvidenceControlUnresolvedError):
        getattr(judge, method_name)(request_payload)

    assert model.calls == []


def test_canonical_scene_quality_adapter_preserves_style_contract(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "style.png"
    _write_test_png(image_path)
    model = FakeMultimodalModel(
        {
            "evidence_status": "sufficient",
            "verdict": "valid",
            "confidence": 0.9,
            "reason": "No significant style inconsistency.",
            "missing_evidence": [],
            "defects": [],
        }
    )
    judge = OpenAICompatibleVLMJudge(model)
    style_spec = {
        "spec_version": "visual_style_spec_v1",
        "source": "benchmark_owned",
        "frozen": True,
        "directives": [
            {
                "directive_id": "s1",
                "statement": "Use a consistent low-poly style.",
                "required": True,
            }
        ],
    }

    result = judge.adjudicate_scene_quality(
        {
            "metric": "style_consistency",
            "prompt": "A low-poly room.",
            "judgment_scope": {
                "included": ["significant_visible_style_incompatibility"]
            },
            "visual_style_spec": style_spec,
            "scene_summary": {"objects": [{"id": "chair"}]},
            "render_evidence": [str(image_path)],
        }
    )

    assert result["verdict"] == "valid"
    context = json.loads(
        model.calls[0]["messages"][1]["content"][0]["text"].split("\n", 1)[1]
    )
    assert context["visual_style_spec"] == style_spec
    assert context["vlm_role"] == "judge"
    assert context["decision_contract"] == "canonical_metric_v1"
    assert context["judge_method"] == "adjudicate_scene_quality"
    assert result["vlm_role"] == "judge"
    assert result["decision_contract"] == "canonical_metric_v1"
    assert result["judge_method"] == "adjudicate_scene_quality"
    assert model.calls[0]["kwargs"]["call_type"] == (
        "vlm_judge.canonical.style_consistency"
    )


def test_canonical_scene_quality_adapter_supports_functional_consistency(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "functional_group.png"
    probe_path = tmp_path / "functional_probe.png"
    _write_test_png(image_path)
    _write_test_png(probe_path)
    model = FakeMultimodalModel(
        {
            "evidence_status": "sufficient",
            "verdict": "valid",
            "confidence": 0.9,
            "reason": "The workstation is visibly usable.",
            "missing_evidence": [],
            "defects": [],
        }
    )

    result = OpenAICompatibleVLMJudge(
        model
    ).adjudicate_scene_quality(
        {
            "metric": "functional_consistency",
            "metric_prompt_version": L3_METRIC_PROMPT_VERSION,
            "metric_boundary_rules": list(L3_METRIC_BOUNDARY_RULES),
            "evidence_phase": "global_discovery",
            "decision_mode": "final",
            "judgment_scope": {
                "included": [
                    "group_real_world_usability",
                    "interaction_side_accessibility",
                    "opening_clearance",
                    "orientation_for_use",
                    "ensemble_operability",
                ]
            },
            "target_object_ids": ["chair", "desk"],
            "render_evidence": [
                str(image_path),
                str(probe_path),
            ],
            "structured_context_policy": {
                "object_fields": ["id", "category"],
                "excluded_object_fields": [
                    "center",
                    "size",
                    "rotation",
                ],
            },
            "functional_probe_evidence": {
                "schema_version": "functional_probe_judge_packet_v1",
                "planning_role": (
                    "visual_evidence_only_no_metric_verdict"
                ),
                "probe_inclusion_is_invalidity_prior": False,
                "image_order": [
                    {
                        "image_index": 0,
                        "image_alias": "image_00",
                        "role": "scene_global",
                    },
                    {
                        "image_index": 1,
                        "image_alias": "image_01",
                        "role": "functional_probe",
                        "probe_kind": "functional_frontage",
                        "target_ids": ["chair"],
                        "presentation": "raw_rgb",
                    },
                ],
            },
        }
    )

    assert result["verdict"] == "valid"
    context = json.loads(
        model.calls[0]["messages"][1]["content"][0][
            "text"
        ].split("\n", 1)[1]
    )
    assert context["metric"] == "functional_consistency"
    assert "ordinary real-world use" in context["rubric"]
    assert "Establish usable sides from visible geometry" in (
        context["rubric"]
    )
    assert context["metric_prompt_version"] == L3_METRIC_PROMPT_VERSION
    assert context["metric_boundary_rules"] == list(
        L3_METRIC_BOUNDARY_RULES
    )
    assert "overall scene-level functional pass" in (
        context["phase_instruction"]
    )
    assert "owned by later isolated cross-group episodes" in (
        context["phase_instruction"]
    )
    assert "Later phases still run" in (
        context["phase_instruction"]
    )
    assert "Additional visual evidence can be acquired" in (
        context["phase_instruction"]
    )
    assert context["functional_probe_evidence"][
        "probe_inclusion_is_invalidity_prior"
    ] is False
    assert context["functional_probe_evidence"]["image_order"][1][
        "presentation"
    ] == "raw_rgb"
    assert context["structured_context_policy"][
        "object_fields"
    ] == ["id", "category"]
    assert result["request_metadata"]["metric_prompt_version"] == (
        L3_METRIC_PROMPT_VERSION
    )
    system_prompt = model.calls[0]["messages"][0]["content"]
    assert "Semantic uncertainty is not evidence insufficiency" in (
        system_prompt
    )
    assert model.calls[0]["kwargs"]["call_type"] == (
        "vlm_judge.canonical.functional_consistency"
    )


def test_functional_required_check_is_repaired_and_preserved(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "functional_check.png"
    _write_test_png(image_path)
    required_check = {
        "check_id": "functional_check_001",
        "check_type": "usable_side_access",
        "owner_stage": "group_local",
        "target_ids": ["bookshelf"],
        "group_ids": ["storage"],
        "owning_group_id": "storage",
        "relation": "usable_side_access",
        "required_observations": [
            "interaction_side_visible",
            "approach_zone_visible",
        ],
        "surface_roles": ["access_side"],
        "decision_authority": "none",
    }
    base_response = {
        "evidence_status": "sufficient",
        "verdict": "valid",
        "confidence": 0.8,
        "reason": "The shelf frontage is accessible.",
        "missing_evidence": [],
        "defects": [],
        "evidence_request": None,
    }
    repaired_response = {
        **base_response,
        "functional_check_results": [
            {
                "check_id": "functional_check_001",
                "target_ids": ["bookshelf"],
                "observation_status": "observed",
                "conclusion": "valid",
                "reason": "The access face and approach region are visible.",
            }
        ],
    }
    model = FakeMultimodalModel([base_response, repaired_response])

    result = OpenAICompatibleVLMJudge(
        model
    ).adjudicate_scene_quality(
        {
            "metric": "functional_consistency",
            "metric_prompt_version": L3_METRIC_PROMPT_VERSION,
            "metric_boundary_rules": list(L3_METRIC_BOUNDARY_RULES),
            "evidence_phase": "group_local_review",
            "decision_mode": "final",
            "judgment_scope": {
                "included": ["interaction_side_accessibility"]
            },
            "target_object_ids": ["bookshelf"],
            "render_evidence": [str(image_path)],
            "required_functional_checks": [required_check],
            "functional_probe_evidence": {
                "required_checks": [required_check],
            },
        }
    )

    assert result["functional_check_results"] == (
        repaired_response["functional_check_results"]
    )
    assert len(model.calls) == 2
    context = json.loads(
        model.calls[0]["messages"][1]["content"][0]["text"].split("\n", 1)[1]
    )
    assert context["required_functional_checks"] == [
        {
            key: value
            for key, value in required_check.items()
            if key != "decision_authority"
        }
    ]
    assert "required_checks" not in context[
        "functional_probe_evidence"
    ]
    assert context["functional_probe_evidence"][
        "required_checks_reference"
    ] == "required_functional_checks"
    assert model.calls[1]["kwargs"]["call_type"].endswith(
        ".schema_repair"
    )


def test_functional_typed_defect_canonicalizes_ambiguous_envelope(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "functional_invalid_envelope.png"
    _write_test_png(image_path)
    required_check = {
        "check_id": "functional_check_direction",
        "check_type": "architecture_orientation",
        "owner_stage": "group_local",
        "target_ids": ["bookshelf"],
        "group_ids": ["storage"],
        "owning_group_id": "storage",
        "relation": "architecture_orientation",
        "required_observations": [
            "interaction_side_visible",
            "global_context_preserved",
        ],
    }
    invalid_row = {
        "check_id": required_check["check_id"],
        "target_ids": ["bookshelf"],
        "observation_status": "observed",
        "conclusion": "invalid",
        "reason": "The usable side faces inaccessible boundary space.",
    }
    defect = {
        "scope": "interaction_side_accessibility",
        "target_ids": ["bookshelf"],
        "relation": "architecture_orientation",
        "reason": "The usable side is not oriented to accessible interior.",
        "check_refs": [required_check["check_id"]],
    }
    initial = {
        "evidence_status": "sufficient",
        "verdict": "ambiguous",
        "confidence": 0.8,
        "reason": "The typed check is invalid.",
        "missing_evidence": [],
        "defects": [defect],
        "evidence_request": None,
        "functional_check_results": [invalid_row],
    }
    model = FakeMultimodalModel([initial])

    result = OpenAICompatibleVLMJudge(model).adjudicate_scene_quality(
        {
            "metric": "functional_consistency",
            "metric_prompt_version": L3_METRIC_PROMPT_VERSION,
            "metric_boundary_rules": list(L3_METRIC_BOUNDARY_RULES),
            "evidence_phase": "group_local_review",
            "decision_mode": "final",
            "judgment_scope": {
                "included": ["interaction_side_accessibility"]
            },
            "target_object_ids": ["bookshelf"],
            "render_evidence": [str(image_path)],
            "required_functional_checks": [required_check],
        }
    )

    assert result["verdict"] == "invalid"
    assert result["functional_check_results"] == [invalid_row]
    audit = result["request_metadata"]["response_schema_validation"]
    assert audit["recovered"] is False
    assert len(model.calls) == 1


def test_placement_check_id_canonicalizes_redundant_relation(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "placement_relation_repair.png"
    _write_test_png(image_path)
    check = {
        "check_id": "placement_check_zone",
        "check_type": "scene_zone",
        "subject_id": "phone",
        "context_ids": [],
        "owner_stage": "scene_global",
        "owning_group_id": "group-1",
        "group_ids": ["group-1"],
        "required_observations": [
            "target_visible",
            "global_context_preserved",
            "architecture_plane_visible",
        ],
        "observation_goals": ["Observe the subject's room zone."],
        "origin": "placement_discovery",
        "acquisition_status": "current_packet",
    }
    row = {
        "check_id": check["check_id"],
        "subject_id": "phone",
        "context_ids": [],
        "observation_status": "observed",
        "conclusion": "invalid",
        "reason": "The subject occupies an implausible room zone.",
    }
    initial_defect = {
        "scope": "semantically_inappropriate_scene_zone",
        "target_ids": ["phone"],
        "relation": "implausible location",
        "reason": "The subject occupies an implausible room zone.",
        "severity": "material_contextual_mismatch",
        "check_id": check["check_id"],
        "placement_check_type": "scene_zone",
    }
    base = {
        "evidence_status": "sufficient",
        "verdict": "invalid",
        "confidence": 0.8,
        "reason": "The typed scene-zone check fails.",
        "missing_evidence": [],
        "evidence_request": None,
        "placement_check_results": [row],
    }
    model = FakeMultimodalModel(
        [{**base, "defects": [initial_defect]}]
    )

    result = OpenAICompatibleVLMJudge(model).adjudicate_scene_quality(
        {
            "metric": "semantic_placement_consistency",
            "evidence_phase": "global_discovery",
            "decision_mode": "final",
            "judgment_scope": {
                "included": ["semantically_inappropriate_scene_zone"]
            },
            "target_object_ids": ["phone"],
            "scene_summary": {
                "scene_type": "living_room",
                "objects": [{"id": "phone", "category": "telephone"}],
            },
            "object_groups": [
                {"group_id": "group-1", "object_ids": ["phone"]}
            ],
            "required_placement_checks": [check],
            "render_evidence": [str(image_path)],
        }
    )

    assert result["verdict"] == "invalid"
    assert result["defects"][0]["relation"] == "scene_zone"
    assert result["request_metadata"]["response_schema_validation"][
        "recovered"
    ] is False
    assert len(model.calls) == 1


def test_functional_schema_repair_may_add_only_omitted_check_rows(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "functional_check_partial.png"
    _write_test_png(image_path)
    required_checks = [
        {
            "check_id": "functional_check_001",
            "check_type": "usable_side_access",
            "owner_stage": "group_local",
            "target_ids": ["bookshelf"],
            "group_ids": ["storage"],
            "owning_group_id": "storage",
            "relation": "usable_side_access",
            "required_observations": ["interaction_side_visible"],
        },
        {
            "check_id": "functional_check_002",
            "check_type": "architecture_clearance",
            "owner_stage": "group_local",
            "target_ids": ["bookshelf"],
            "group_ids": ["storage"],
            "owning_group_id": "storage",
            "relation": "architecture_clearance",
            "required_observations": ["architecture_plane_visible"],
        },
    ]
    first_row = {
        "check_id": "functional_check_001",
        "target_ids": ["bookshelf"],
        "observation_status": "observed",
        "conclusion": "valid",
        "reason": "The frontage is visible and accessible.",
    }
    second_row = {
        "check_id": "functional_check_002",
        "target_ids": ["bookshelf"],
        "observation_status": "observed",
        "conclusion": "valid",
        "reason": "The frontage has usable separation from the wall.",
    }
    base_response = {
        "evidence_status": "sufficient",
        "verdict": "valid",
        "confidence": 0.8,
        "reason": "The bookshelf is usable.",
        "missing_evidence": [],
        "defects": [],
        "evidence_request": None,
    }
    initial_response = {
        **base_response,
        "functional_check_results": [first_row],
    }
    repaired_response = {
        **base_response,
        "functional_check_results": [first_row, second_row],
    }
    model = FakeMultimodalModel(
        [initial_response, repaired_response]
    )

    result = OpenAICompatibleVLMJudge(
        model
    ).adjudicate_scene_quality(
        {
            "metric": "functional_consistency",
            "metric_prompt_version": L3_METRIC_PROMPT_VERSION,
            "metric_boundary_rules": list(L3_METRIC_BOUNDARY_RULES),
            "evidence_phase": "group_local_review",
            "decision_mode": "final",
            "judgment_scope": {
                "included": ["interaction_side_accessibility"]
            },
            "target_object_ids": ["bookshelf"],
            "render_evidence": [str(image_path)],
            "required_functional_checks": required_checks,
            "functional_probe_evidence": {
                "required_checks": required_checks,
            },
        }
    )

    assert result["functional_check_results"] == [
        first_row,
        second_row,
    ]
    assert len(model.calls) == 2


def test_required_functional_checks_fail_closed_before_context_truncation(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "functional_check_context_budget.png"
    _write_test_png(image_path)
    model = FakeMultimodalModel(
        {
            "evidence_status": "sufficient",
            "verdict": "valid",
            "confidence": 0.8,
            "reason": "must not be called",
            "missing_evidence": [],
            "defects": [],
            "evidence_request": None,
        }
    )
    required_check = {
        "check_id": "functional_check_001",
        "check_type": "usable_side_access",
        "owner_stage": "group_local",
        "target_ids": ["bookshelf"],
        "relation": "usable_side_access",
        "required_observations": ["interaction_side_visible"],
        "observation_goals": ["x" * 20_000],
    }

    with pytest.raises(ValueError, match="context budget"):
        OpenAICompatibleVLMJudge(
            model,
            max_context_chars=1_000,
        ).adjudicate_scene_quality(
            {
                "metric": "functional_consistency",
                "metric_prompt_version": L3_METRIC_PROMPT_VERSION,
                "metric_boundary_rules": list(
                    L3_METRIC_BOUNDARY_RULES
                ),
                "evidence_phase": "group_local_review",
                "decision_mode": "final",
                "judgment_scope": {
                    "included": [
                        "interaction_side_accessibility"
                    ]
                },
                "target_object_ids": ["bookshelf"],
                "render_evidence": [str(image_path)],
                "required_functional_checks": [required_check],
                "functional_probe_evidence": {
                    "required_checks": [required_check],
                },
            }
        )

    assert model.calls == []


def test_scene_quality_adapter_keeps_semantic_placement_out_of_l1(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "placement_group.png"
    _write_test_png(image_path)
    model = FakeMultimodalModel(
        {
            "evidence_status": "sufficient",
            "verdict": "invalid",
            "confidence": 0.9,
            "reason": "The phone is implausibly placed on the floor.",
            "missing_evidence": [],
                "defects": [
                    {
                        "scope": "semantically_inappropriate_support_surface",
                        "target_ids": ["phone"],
                        "relation": "support_and_height",
                        "reason": "The floor is not a plausible semantic surface.",
                        "severity": "clear_semantic_misplacement",
                        "check_id": "phone_support_proposal",
                    }
                ],
                "judge_originated_placement_results": [
                    {
                        "proposal_id": "phone_support_proposal",
                        "subject_id": "phone",
                        "context_ids": [],
                        "check_type": "support_and_height",
                        "observation_goal": (
                            "Inspect the phone's semantic support surface "
                            "and placement height."
                        ),
                        "observation_status": "observed",
                        "conclusion": "invalid",
                        "reason": (
                            "The floor is not a plausible semantic surface."
                        ),
                        "severity": "clear_semantic_misplacement",
                    }
                ],
            }
        )

    result = OpenAICompatibleVLMJudge(
        model
    ).adjudicate_scene_quality(
        {
            "metric": "semantic_placement_consistency",
            "evidence_phase": "group_local_review",
            "decision_mode": "final",
            "judgment_scope": {
                "included": [
                    "semantically_inappropriate_support_surface",
                    "implausible_placement_height",
                    "semantically_inappropriate_scene_zone",
                    "implausible_local_context",
                ],
                "excluded": [
                    "collision",
                    "physical_support",
                ],
            },
            "target_object_ids": ["phone", "side_table"],
            "scene_summary": {
                "scene_type": "living_room",
                "objects": [
                    {"id": "phone", "category": "telephone"},
                    {"id": "side_table", "category": "side_table"},
                ],
            },
            "render_evidence": [str(image_path)],
        }
    )

    assert result["verdict"] == "invalid"
    context = json.loads(
        model.calls[0]["messages"][1]["content"][0][
            "text"
        ].split("\n", 1)[1]
    )
    assert "current location" in context["rubric"]
    assert "out-of-bounds geometry belong to L1" in context["rubric"]
    assert "group-local pass" in (
        context["phase_instruction"]
    )
    assert "support-surface meaning" in context["phase_instruction"]
    assert "orientation or operability" in (
        context["phase_instruction"]
    )
    assert model.calls[0]["kwargs"]["call_type"] == (
        "vlm_judge.canonical.semantic_placement_consistency"
    )


def test_global_placement_prompt_defers_group_local_discovery_miss(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "placement_global.png"
    _write_test_png(image_path)
    model = FakeMultimodalModel(
        {
            "evidence_status": "sufficient",
            "verdict": "valid",
            "confidence": 0.9,
            "reason": "No scene-global placement defect is present.",
            "missing_evidence": [],
            "defects": [],
        }
    )
    deferred = {
        "check_id": "placement_check:phone-support",
        "check_type": "support_and_height",
        "subject_id": "phone",
        "context_ids": [],
        "owner_stage": "group_local",
        "owning_group_id": "group-1",
        "group_ids": ["group-1"],
        "required_observations": [
            "target_visible",
            "contact_surface_visible",
            "group_context_visible",
        ],
        "observation_goals": ["Inspect semantic support and height."],
        "origin": "judge_originated_evidence_request",
        "acquisition_status": "pending",
    }

    result = OpenAICompatibleVLMJudge(
        model
    ).adjudicate_scene_quality(
        {
            "metric": "semantic_placement_consistency",
            "evidence_phase": "global_discovery",
            "decision_mode": "final",
            "judgment_scope": {
                "included": [
                    "semantically_inappropriate_support_surface",
                    "implausible_placement_height",
                    "semantically_inappropriate_scene_zone",
                    "implausible_local_context",
                ],
                "excluded": ["collision", "physical_support"],
            },
            "scene_summary": {
                "scene_type": "living_room",
                "objects": [
                    {"id": "phone", "category": "telephone"},
                    {"id": "side_table", "category": "side_table"},
                ],
            },
            "object_groups": [
                {
                    "group_id": "group-1",
                    "object_ids": ["phone", "side_table"],
                }
            ],
            "required_placement_checks": [],
            "deferred_placement_checks": [deferred],
            "render_evidence": [str(image_path)],
        }
    )

    assert result["verdict"] == "valid"
    context = json.loads(
        model.calls[0]["messages"][1]["content"][0]["text"].split(
            "\n", 1
        )[1]
    )
    assert context["deferred_placement_checks"] == [deferred]
    assert "later group-local stage" in context["phase_instruction"]


@pytest.mark.parametrize(
    "response",
    [
        {
            "evidence_status": "sufficient",
            "verdict": "invalid",
            "confidence": 0.8,
            "reason": "bad",
            "missing_evidence": [],
            "defects": [{"scope": "wrong", "target_ids": ["chair"], "relation": "x", "reason": "bad"}],
        },
        {
            "evidence_status": "sufficient",
            "verdict": "invalid",
            "confidence": 0.8,
            "reason": "bad",
            "missing_evidence": [],
            "defects": [
                {
                    "scope": "significant_visible_style_incompatibility",
                    "target_ids": [],
                    "relation": "x",
                    "reason": "bad",
                }
            ],
        },
        {
            "evidence_status": "insufficient",
            "verdict": "ambiguous",
            "confidence": 0.2,
            "reason": "occluded",
            "missing_evidence": [],
            "defects": [],
        },
        {
            "evidence_status": "insufficient",
            "verdict": "valid",
            "confidence": 0.2,
            "reason": "occluded",
            "missing_evidence": ["closer_view"],
            "defects": [],
        },
    ],
)
def test_canonical_scene_quality_adapter_rejects_malformed_contract(
    tmp_path: Path,
    response: dict,
) -> None:
    image_path = tmp_path / "style.png"
    _write_test_png(image_path)
    judge = OpenAICompatibleVLMJudge(FakeMultimodalModel(response))
    with pytest.raises(ValueError):
        judge.adjudicate_scene_quality(
            {
                "metric": "style_consistency",
                "judgment_scope": {
                    "included": ["significant_visible_style_incompatibility"]
                },
                "render_evidence": [str(image_path)],
            }
        )


def test_budget_exhaustion_forced_choice_uses_context_bounded_visuals(
    tmp_path: Path,
) -> None:
    image_paths = []
    for index in range(8):
        image_path = tmp_path / f"view-{index}.png"
        _write_test_png(image_path)
        image_paths.append(image_path)
    model = FakeMultimodalModel(
        {
            "evidence_status": "sufficient",
            "verdict": "valid",
            "confidence": 0.55,
            "reason": "No significant in-scope defect is established.",
            "missing_evidence": [],
            "defects": [],
            "evidence_request": None,
        }
    )
    judge = OpenAICompatibleVLMJudge(model, max_images=3)

    result = judge._adjudicate_scene_quality_raw(
        {
            "metric": "style_consistency",
            "judgment_scope": {
                "included": [
                    "significant_visible_style_incompatibility"
                ]
            },
            "render_evidence": [str(path) for path in image_paths],
            "budget_exhaustion_finalization": {
                "required": True,
                "trigger_stop_reason": (
                    "max_evidence_rounds_exhausted"
                ),
                "previous_missing_observations": [
                    "group_context_visible"
                ],
            },
        }
    )

    assert result["verdict"] == "valid"
    assert result["images_used"] == [
        str(image_paths[0].resolve()),
        str(image_paths[6].resolve()),
        str(image_paths[7].resolve()),
    ]
    context = json.loads(
        model.calls[0]["messages"][1]["content"][0]["text"].split(
            "\n", 1
        )[1]
    )
    finalization = context["budget_exhaustion_finalization"]
    assert finalization["available_visual_count"] == 8
    assert finalization["selected_visual_count"] == 3
    assert finalization["dropped_visual_count"] == 5
    assert (
        finalization["visual_selection_policy"]
        == "global_anchor_plus_most_recent"
    )
    assert finalization["configured_visual_context_limit"] == 3
    assert "ambiguous" in model.calls[0]["messages"][0]["content"]
    assert "forbidden" in model.calls[0]["messages"][0]["content"]
    assert model.calls[0]["kwargs"]["call_type"].endswith(
        ".forced_choice"
    )


def test_budget_exhaustion_ambiguous_response_is_repaired_to_binary(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "forced-choice.png"
    _write_test_png(image_path)
    ambiguous = {
        "evidence_status": "insufficient",
        "verdict": "ambiguous",
        "confidence": 0.2,
        "reason": "A closer view would normally help.",
        "missing_evidence": ["group_context_visible"],
        "defects": [],
        "evidence_request": {
            "target_ids": ["scene"],
            "missing_observations": ["group_context_visible"],
            "view_goal": "show more context",
            "metadata": {},
        },
    }
    valid = {
        "evidence_status": "sufficient",
        "verdict": "valid",
        "confidence": 0.45,
        "reason": "The current packet does not establish a defect.",
        "missing_evidence": [],
        "defects": [],
        "evidence_request": None,
    }
    model = FakeMultimodalModel([ambiguous, valid])

    result = OpenAICompatibleVLMJudge(
        model
    )._adjudicate_scene_quality_raw(
        {
            "metric": "style_consistency",
            "judgment_scope": {
                "included": [
                    "significant_visible_style_incompatibility"
                ]
            },
            "render_evidence": [str(image_path)],
            "budget_exhaustion_finalization": {
                "required": True,
                "trigger_stop_reason": (
                    "max_evidence_rounds_exhausted"
                ),
            },
        }
    )

    assert result["verdict"] == "valid"
    assert len(model.calls) == 2
    assert model.calls[1]["kwargs"]["call_type"].endswith(
        ".forced_choice.schema_repair"
    )
    assert "ambiguous" in model.calls[1]["messages"][-1]["content"]
    assert "forbidden" in model.calls[1]["messages"][-1]["content"]
    assert result["request_metadata"]["response_schema_validation"][
        "policy"
    ] == "single_forced_choice_decision_retry_v1"


def test_functional_semantic_forces_terminal_choice_after_unrepairable_evidence(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "functional.png"
    _write_test_png(image_path)
    model = FakeMultimodalModel(
        [
            {
                "evidence_status": "insufficient",
                "verdict": "ambiguous",
                "confidence": 0.2,
                "reason": "The requested work area is occluded.",
                "missing_evidence": [],
                "defects": [],
                "evidence_request": {
                    "target_ids": ["scene"],
                    "missing_observations": [
                        "group_context_visible"
                    ],
                    "view_goal": "show the requested work area",
                    "metadata": {},
                },
            },
            {
                "evidence_status": "sufficient",
                "verdict": "valid",
                "confidence": 0.4,
                "reason": "No functional defect is established by the available packet.",
                "missing_evidence": [],
                "defects": [],
                "evidence_request": None,
            },
        ]
    )

    result = OpenAICompatibleVLMJudge(model).adjudicate_functional_semantic(
        {
            "metric": "functional_semantic_fidelity",
            "judgment_scope": {
                "included": ["this_benchmark_owned_prompt_claim_only"]
            },
            "render_evidence": [str(image_path)],
        }
    )

    context = json.loads(
        model.calls[0]["messages"][1]["content"][0]["text"].split("\n", 1)[1]
    )
    assert context["vlm_role"] == "judge"
    assert context["decision_contract"] == "canonical_metric_v1"
    assert context["judge_method"] == "adjudicate_functional_semantic"
    assert result["verdict"] == "valid"
    assert result["request_metadata"]["budget_exhaustion_finalization"][
        "ambiguity_before_forcing"
    ] is True
    assert result["request_metadata"]["budget_exhaustion_finalization"][
        "previous_evidence_request"
    ]["target_ids"] == ["scene"]
    assert result["request_metadata"]["budget_exhaustion_finalization"][
        "termination_kind"
    ] == "acquisition_unavailable"
    assert result["router_state"] == "not_suspicious"
    assert result["evidence_request"] is None
    assert result["vlm_role"] == "judge"
    assert result["decision_contract"] == "canonical_metric_v1"
    assert result["judge_method"] == "adjudicate_functional_semantic"
    assert "allowed_missing_observations" in context
    assert "allowed_evidence_request_target_ids" in context


def test_canonical_judge_rejects_free_form_missing_evidence(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "functional-free-form.png"
    _write_test_png(image_path)
    judge = OpenAICompatibleVLMJudge(
        FakeMultimodalModel(
            {
                "evidence_status": "insufficient",
                "verdict": "ambiguous",
                "confidence": 0.2,
                "reason": "The requested work area is occluded.",
                "missing_evidence": [
                    "please show a better angle"
                ],
                "defects": [],
            }
        )
    )

    with pytest.raises(ValueError, match="exact allowed Camera DSL"):
        judge.adjudicate_functional_semantic(
            {
                "metric": "functional_semantic_fidelity",
                "render_evidence": [str(image_path)],
            }
        )


def test_vlm_category_supports_not_applicable_response(tmp_path: Path) -> None:
    image_path = tmp_path / "view.png"
    _write_test_png(image_path)
    model = FakeMultimodalModel(
        {"applicable": False, "score": None, "confidence": 0.2, "summary": "object is occluded", "issues": [], "evidence": []}
    )

    report = evaluate_vlm_category(
        category="visual_quality",
        prompt=None,
        scene={"scene_id": "scene", "objects": []},
        render_evidence=[str(image_path)],
        judge=OpenAICompatibleVLMJudge(model),
    )

    assert report["status"] == "not_evaluable"
    assert report["score"] is None
    assert report["reason"] == "vlm_judge_insufficient_evidence"


def test_vlm_category_preserves_canonical_description_and_proxy_status(tmp_path: Path) -> None:
    image_path = tmp_path / "view.png"
    _write_test_png(image_path)
    model = FakeMultimodalModel(
        {"applicable": True, "score": 0.5, "confidence": 0.7, "summary": "layout only", "issues": [], "evidence": []}
    )

    evaluate_vlm_category(
        category="prompt_fidelity",
        prompt="Place a red bed in the room.",
        scene={
            "scene_id": "scene",
            "objects": [
                {
                    "id": "bed_1",
                    "category": "bed",
                    "description": "red bed",
                    "center": [1, 1, 0.5],
                    "size": [2, 1.5, 1],
                    "asset_proxy": {"type": "obb", "bbox_size": [2, 1.5, 1]},
                    "metadata": {"asset_resolution": "unresolved"},
                }
            ],
        },
        render_evidence=[str(image_path)],
        judge=OpenAICompatibleVLMJudge(model),
    )

    context = model.calls[0]["messages"][1]["content"][0]["text"]
    assert '"description":"red bed"' in context
    assert '"asset_resolution":"unresolved"' in context
    assert '"proxy_type":"obb"' in context


def test_judge_builder_supports_mnet_and_remote_api_config() -> None:
    local = build_openai_compatible_vlm_judge(
        {"endpoint": "http://127.0.0.1:8298/v1", "model": "Qwen3-VL-32B-Instruct-64K"}
    )
    remote = build_openai_compatible_vlm_judge(
        {"endpoint": "https://api.example.com/v1", "model": "remote-vlm", "api_key_env": "REMOTE_API_KEY"}
    )

    assert local.model.api_key_env is None
    assert remote.model.api_key_env == "REMOTE_API_KEY"
    assert remote.model.endpoint == "https://api.example.com/v1"
    assert local.model.max_tokens == 8192
    assert remote.model.max_tokens == 8192


def test_relation_judge_is_binary_and_receives_prompt_claim_and_image(tmp_path: Path) -> None:
    image_path = tmp_path / "local_relation_view.png"
    _write_test_png(image_path)
    model = FakeMultimodalModel(
        {"verdict": "valid", "confidence": 0.85, "reason": "the requested relation is visible"}
    )
    judge = OpenAICompatibleVLMJudge(model)

    result = judge.adjudicate_relation(
        {
            "family": "oor",
            "relation": {"type": "mirrors", "subject_id": "a", "object_id": "b"},
            "natural_language_prompt": "A mirrors B.",
            "involved_objects": [{"id": "a"}, {"id": "b"}],
            "scene_summary": {"scene_id": "scene", "objects": []},
            "detector_evidence": {"route": "requires_vlm", "proxy": "obb"},
            "render_evidence": [str(image_path)],
        }
    )

    content = model.calls[0]["messages"][1]["content"]
    assert "A mirrors B." in content[0]["text"]
    assert '"type":"mirrors"' in content[0]["text"]
    assert '"route":"requires_vlm"' in content[0]["text"]
    assert '"proxy":"obb"' in content[0]["text"]
    assert content[1]["type"] == "image_url"
    assert model.calls[0]["kwargs"]["call_type"] == "vlm_judge.relationship.oor"
    assert result["verdict"] == "valid"
    assert result["confidence"] == 0.85
    assert result["vlm_role"] == "judge"
    assert result["decision_contract"] == "relation_binary_v1"
    assert result["judge_method"] == "adjudicate_relation"


@pytest.mark.parametrize("verdict", ["ambiguous", "insufficient_evidence"])
def test_relation_judge_rejects_third_verdict(
    tmp_path: Path,
    verdict: str,
) -> None:
    image_path = tmp_path / "relation.png"
    _write_test_png(image_path)
    judge = OpenAICompatibleVLMJudge(
        FakeMultimodalModel(
            {"verdict": verdict, "confidence": 0.2, "reason": "occluded"}
        )
    )

    with pytest.raises(ValueError, match="exactly 'valid' or 'invalid'"):
        judge.adjudicate_relation(
            {
                "family": "oor",
                "relation": {"type": "mirrors"},
                "render_evidence": [str(image_path)],
            }
        )


@pytest.mark.parametrize(
    ("control_method", "payload", "expected_observation"),
    [
        (
            "_adjudicate_p0b_control",
            {
                "metric": "collision",
                "event": {"object_ids": ["a", "b"]},
            },
            "contact_surface_visible",
        ),
        (
            "_adjudicate_relation_control",
            {
                "family": "oor",
                "relation": {
                    "type": "mirrors",
                    "subject_id": "a",
                    "object_id": "b",
                },
            },
            "target_visible",
        ),
    ],
)
def test_binary_judges_have_internal_structured_need_more_path(
    tmp_path,
    control_method,
    payload,
    expected_observation,
):
    image_path = tmp_path / "binary-control.png"
    _write_test_png(image_path)
    model = FakeMultimodalModel(
        {
            "status": "need_more_evidence",
            "confidence": 0.2,
            "reason": "both targets are not jointly visible",
            "defects": [],
            "evidence_request": {
                "target_ids": ["a", "b"],
                "missing_observations": ["joint_visibility"],
                "view_goal": "show both targets in one view",
                "metadata": {"source": "binary_control"},
            },
        }
    )
    judge = OpenAICompatibleVLMJudge(model)

    result = getattr(judge, control_method)(
        {**payload, "render_evidence": [str(image_path)]}
    )

    assert result["status"] == "need_more_evidence"
    assert result["evidence_request"]["target_ids"] == ["a", "b"]
    system_prompt = model.calls[0]["messages"][0]["content"]
    user_text = model.calls[0]["messages"][1]["content"][0]["text"]
    assert "do not guess" in system_prompt
    assert "need_more_evidence" in system_prompt
    assert "allowed_missing_observations" in user_text
    assert expected_observation in user_text
    assert "Do not put prose in missing_observations" in system_prompt


@pytest.mark.parametrize(
    ("control_method", "payload"),
    [
        (
            "_adjudicate_p0b_control",
            {
                "metric": "collision",
                "event": {"object_ids": ["a", "b"]},
            },
        ),
        (
            "_adjudicate_relation_control",
            {
                "family": "oor",
                "relation": {
                    "type": "mirrors",
                    "subject_id": "a",
                    "object_id": "b",
                },
            },
        ),
    ],
)
def test_internal_binary_schema_failure_gets_one_same_evidence_repair(
    tmp_path: Path,
    control_method: str,
    payload: dict,
) -> None:
    image_path = tmp_path / "binary-repair.png"
    _write_test_png(image_path)
    first = {
        "status": "invalid",
        "confidence": 0.8,
        "reason": "The visible event is invalid.",
        "defects": ["free-form defect"],
        "evidence_request": None,
    }
    repaired = {
        **first,
        "reason": "The same event remains invalid.",
        "defects": [],
    }
    model = FakeMultimodalModel([first, repaired])
    judge = OpenAICompatibleVLMJudge(model)

    result = getattr(judge, control_method)(
        {**payload, "render_evidence": [str(image_path)]}
    )

    assert result["status"] == "invalid"
    assert result["reason"] == first["reason"]
    assert result["defects"] == []
    assert len(model.calls) == 2
    assert model.calls[1]["kwargs"]["call_type"].endswith(
        ".schema_repair"
    )
    assert model.calls[1]["messages"][:2] == (
        model.calls[0]["messages"]
    )
    assert model.calls[1]["messages"][2] == {
        "role": "assistant",
        "content": json.dumps(first),
    }
    assert "defects exactly []" in (
        model.calls[1]["messages"][3]["content"]
    )
    audit = result["request_metadata"][
        "response_schema_validation"
    ]
    assert audit["attempt_count"] == 2
    assert audit["repair_retry_count"] == 1
    assert audit["recovered"] is True
    assert audit["semantic_preservation"][
        "restored_natural_language_fields"
    ] == ["reason"]
    assert json.loads(audit["attempts"][0]["raw_response"]) == first


def test_internal_binary_schema_repair_fails_closed_after_one_retry(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "binary-repair-failure.png"
    _write_test_png(image_path)
    invalid = {
        "status": "invalid",
        "confidence": 0.8,
        "reason": "invalid event",
        "defects": ["still not allowed"],
        "evidence_request": None,
    }
    model = FakeMultimodalModel([invalid, invalid, {
        "status": "valid",
        "confidence": 1.0,
        "reason": "must never be called",
        "defects": [],
        "evidence_request": None,
    }])
    judge = OpenAICompatibleVLMJudge(model)

    with pytest.raises(ResponseSchemaRepairError) as raised:
        judge._adjudicate_p0b_control(
            {
                "metric": "collision",
                "event": {"object_ids": ["a", "b"]},
                "render_evidence": [str(image_path)],
            }
        )

    assert len(model.calls) == 2
    audit = raised.value.schema_audit
    assert audit["attempt_count"] == 2
    assert audit["repair_retry_count"] == 1
    assert audit["recovered"] is False
    assert len(audit["attempts"]) == 2


def test_spatial_fidelity_judge_is_binary_and_treats_rarity_as_routing_only(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "spatial_pair.png"
    _write_test_png(image_path)
    model = FakeMultimodalModel(
        {
            "verdict": "valid",
            "confidence": 0.9,
            "reason": "unusual but coherent for a studio bedroom",
        }
    )
    judge = OpenAICompatibleVLMJudge(model)

    result = judge.adjudicate_spatial_fidelity(
        {
            "metric": "cooccurrence_plausibility",
            "event": {
                "type": "rare_category_cooccurrence",
                "object_ids": ["bed", "fridge"],
            },
            "detector_evidence": {
                "best_directional_probability": 0.005,
                "candidate_route": "requires_vlm",
            },
            "involved_objects": [{"id": "bed"}, {"id": "fridge"}],
            "scene_summary": {"scene_id": "studio", "objects": []},
            "natural_language_prompt": "Create a compact studio bedroom.",
            "render_evidence": [str(image_path)],
        }
    )

    system = model.calls[0]["messages"][0]["content"]
    content = model.calls[0]["messages"][1]["content"]
    assert "dataset rarity or absence alone is never an error" in system
    assert '"best_directional_probability":0.005' in content[0]["text"]
    assert content[1]["type"] == "image_url"
    assert model.calls[0]["kwargs"]["call_type"] == (
        "vlm_judge.spatial_fidelity.cooccurrence_plausibility"
    )
    assert result["verdict"] == "valid"
    assert result["confidence"] == 0.9
    assert result["vlm_role"] == "judge"
    assert result["decision_contract"] == "spatial_fidelity_binary_v1"
    assert result["judge_method"] == "adjudicate_spatial_fidelity"


def test_relation_context_budget_preserves_detector_packet_ahead_of_large_scene(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "relation.png"
    _write_test_png(image_path)
    model = FakeMultimodalModel(
        {"verdict": "invalid", "confidence": 0.9, "reason": "proxy and image disagree"}
    )
    judge = OpenAICompatibleVLMJudge(model, max_context_chars=1000)

    judge.adjudicate_relation(
        {
            "family": "oar",
            "relation": {"type": "mounted_on_wall", "subject_id": "painting", "wall": "north"},
            "natural_language_prompt": "Mount the painting on the north wall. " + "detail " * 1000,
            "involved_objects": [{"id": "painting"}],
            "detector_evidence": {
                "route": "requires_vlm",
                "proxy_checks_passed": False,
                "signed_wall_clearance_m": 0.42,
            },
            "scene_summary": {
                "scene_id": "large",
                "objects": [{"id": f"object_{index}", "description": "x" * 300} for index in range(50)],
            },
            "render_evidence": [str(image_path)],
        }
    )

    text = model.calls[0]["messages"][1]["content"][0]["text"]
    context_text = text.split("\n", 1)[1]
    context = json.loads(context_text)
    assert len(context_text) <= 1000
    assert context["vlm_role"] == "judge"
    assert context["decision_contract"] == "relation_binary_v1"
    assert context["judge_method"] == "adjudicate_relation"
    assert context["explicit_relation_claim"]["type"] == "mounted_on_wall"
    assert context["detector_evidence"]["signed_wall_clearance_m"] == 0.42
    assert context["_benchmark_context_budget"]["truncated"] is True


def test_p0b_context_budget_reserves_every_priority_field(tmp_path: Path) -> None:
    image_path = tmp_path / "p0b.png"
    _write_test_png(image_path)
    model = FakeMultimodalModel(
        {"verdict": "valid", "confidence": 0.9, "reason": "test"}
    )
    judge = OpenAICompatibleVLMJudge(model, max_context_chars=1000)
    huge = "x" * 5000

    result = judge.adjudicate_p0b(
        {
            "metric": "collision",
            "event": {"event_detail": huge},
            "detector_evidence": {"detector_sentinel": huge},
            "natural_language_prompt": huge,
            "metric_rubric": huge,
            "candidate_selection_policy": "high_recall_candidate_no_label_prior",
            "collision_evidence_style_guide": huge,
            "local_render_evidence_metadata": {"metadata": huge},
            "objects": [{"description": huge}],
            "architecture": {"description": huge},
            "render_evidence": [str(image_path)],
        }
    )

    text = model.calls[0]["messages"][1]["content"][0]["text"]
    context_text = text.split("\n", 1)[1]
    context = json.loads(context_text)
    assert len(context_text) <= 1000
    for key in (
        "vlm_role",
        "decision_contract",
        "judge_method",
        "metric",
        "event",
        "detector_evidence",
        "natural_language_prompt",
        "metric_rubric",
        "candidate_selection_policy",
        "collision_evidence_style_guide",
        "view_names",
        "view_evidence",
    ):
        assert key in context
    assert "detector_sentinel" in context["detector_evidence"]["json_prefix"]
    assert context["vlm_role"] == "judge"
    assert context["decision_contract"] == "p0b_binary_v1"
    assert context["judge_method"] == "adjudicate_p0b"
    assert result["vlm_role"] == "judge"
    assert result["decision_contract"] == "p0b_binary_v1"
    assert result["judge_method"] == "adjudicate_p0b"


def test_p0b_outbound_context_hides_local_paths_hashes_and_dataset_linkage(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "private_case_name.png"
    _write_test_png(image_path)
    model = FakeMultimodalModel(
        {"verdict": "invalid", "confidence": 0.9, "reason": "visible gap"}
    )
    judge = OpenAICompatibleVLMJudge(model)

    judge.adjudicate_p0b(
        {
            "metric": "support",
            "event": {"object_id": "obj_000"},
            "detector_evidence": {"gap_m": 0.1},
            "natural_language_prompt": "Create a room.",
            "local_render_evidence_metadata": [
                {
                    "path": "/private/local/exp1_1/case.png",
                    "sha256": "private-content-hash",
                    "case_id": "private-case-id",
                    "geometry_provenance": "private-asset-source",
                    "role": "metric_local_highlight",
                    "view_id": "support_contact_060",
                }
            ],
            "render_evidence": [str(image_path)],
        }
    )

    text = model.calls[0]["messages"][1]["content"][0]["text"]
    context = json.loads(text.split("\n", 1)[1])
    serialized = json.dumps(context)
    assert context["view_names"] == ["image_00"]
    assert context["view_evidence"] == [
        {
            "role": "metric_local_highlight",
            "view_id": "slot_00",
        }
    ]
    assert "private/local" not in serialized
    assert "private-content-hash" not in serialized
    assert "private-case-id" not in serialized
    assert "private-asset-source" not in serialized


def test_query_cov_camera_selector_chooses_views_without_metric_verdict(tmp_path: Path) -> None:
    first = tmp_path / "candidate_a.png"
    second = tmp_path / "candidate_b.png"
    private_metadata = PngImagePlugin.PngInfo()
    private_metadata.add_text("private_source", "must-not-leave-process")
    first_image = Image.new("RGBA", (3, 2), (255, 0, 0, 64))
    first_image.putpixel((0, 0), (0, 255, 0, 255))
    first_image.save(first, pnginfo=private_metadata)
    second_image = Image.new("RGBA", (3, 2), (0, 0, 255, 128))
    second_image.putpixel((0, 0), (255, 255, 0, 255))
    second_image.save(second)
    candidates = [
        {
            "id": "generator_first_private_id",
            "pose": {
                "location": [1, 2, 3],
                "target": [0, 0, 0],
                "azimuth_degrees": 10.0,
                "elevation_degrees": 20.0,
                "target_object_ids": ["bed", "cabinet"],
            },
            "image_path": str(first),
        },
        {
            "id": "generator_second_private_id",
            "pose": {
                "location": [3, 2, 1],
                "target": [0, 0, 0],
                "azimuth_degrees": 200.0,
                "elevation_degrees": 15.0,
                "target_object_ids": ["bed", "cabinet"],
            },
            "image_path": str(second),
        },
    ]
    digest_order = sorted(candidates, key=_selector_candidate_order_key)
    candidates = list(reversed(digest_order))
    assert candidates[0]["id"] != digest_order[0]["id"]
    model = FakeMultimodalModel(
        {
            "selected_view_ids": ["candidate_00"],
            "action": {"view_id": "candidate_00", "type": "orbit_left"},
            "reason": "the overlap is visible but needs a small lateral change",
        }
    )
    judge = OpenAICompatibleVLMJudge(model)

    result = judge.select_camera_views(
        {
            "metric": "collision",
            "event": {"object_a": "bed", "object_b": "cabinet"},
            "object_ids": ["bed", "cabinet"],
            "detector_evidence": {"normalized_overlap": 0.2, "secret": "detector-secret"},
            "natural_language_prompt": "private-prompt-sentinel",
            "extracted_relationships": [{"private-relation-sentinel": True}],
            "candidates": candidates,
            "max_views": 1,
            "allow_adjustment": True,
            "allowed_actions": ["orbit_left", "orbit_right"],
            "preview_role": "highlighted_focus",
            "color_legend": [
                {
                    "id": "bed",
                    "category": "private-category",
                    "role": "object_a",
                    "color": [1.0, 0.0, 0.0],
                    "representation": "asset_mesh",
                }
            ],
            # This is the field emitted by the active CameraEvidenceProvider.
            "preview_degradation": "focus_preview_failed: RuntimeError: incomplete highlight coverage",
        }
    )

    system = model.calls[0]["messages"][0]["content"]
    assert "Do not judge whether the metric event is valid or invalid" in system
    assert model.calls[0]["kwargs"]["call_type"] == "vlm_camera_pose.query_cov"
    assert result["selected_view_ids"] == [digest_order[0]["id"]]
    assert result["action"]["view_id"] == digest_order[0]["id"]
    assert result["action"]["type"] == "orbit_left"
    assert result["images_used"] == ["candidate_00", "candidate_01"]
    assert result["request_metadata"]["selector_candidate_order_policy"] == "stable_pose_image_digest_v1"
    assert result["request_metadata"]["selector_candidate_alias_policy"] == "per_request_sequential_alias_v1"
    assert result["vlm_role"] == "vlm_camera_selector"
    assert result["decision_contract"] == "camera_selection_v1"
    assert result["judge_method"] == "select_camera_views"

    outbound = model.calls[0]["messages"][1]["content"]
    context = json.loads(outbound[0]["text"].split("\n", 1)[1])
    assert set(context) == {
        "candidates",
        "max_views",
        "allow_adjustment",
        "allowed_actions",
        "metric_family",
        "preview_role",
        "preview_warning_class",
        "color_legend",
        "vlm_role",
        "decision_contract",
        "judge_method",
    }
    assert context["vlm_role"] == "vlm_camera_selector"
    assert context["decision_contract"] == "camera_selection_v1"
    assert context["judge_method"] == "select_camera_views"
    assert [item["id"] for item in context["candidates"]] == ["candidate_00", "candidate_01"]
    assert context["metric_family"] == "collision"
    assert context["preview_warning_class"] == "incomplete_target_visibility"
    assert context["color_legend"] == [
        {"role": "object_a", "rgb": [1.0, 0.0, 0.0], "representation": "mesh"}
    ]
    serialized = json.dumps(context)
    for private_value in (
        "bed",
        "cabinet",
        "detector-secret",
        "private-prompt-sentinel",
        "private-relation-sentinel",
        "private-category",
        "generator_first_private_id",
        "generator_second_private_id",
        first.name,
        second.name,
        str(tmp_path),
    ):
        assert private_value not in serialized
    assert all("location" not in item["pose"] and "target" not in item["pose"] for item in context["candidates"])

    encoded = outbound[1]["image_url"]["url"].split(",", 1)[1]
    with Image.open(BytesIO(base64.b64decode(encoded))) as normalized:
        assert normalized.mode == "RGB"
        assert normalized.info == {}


def test_active_camera_selector_receives_only_sanitized_deficiency(
    tmp_path: Path,
) -> None:
    image = tmp_path / "candidate.png"
    rendered = Image.new("RGB", (3, 2), (128, 128, 128))
    rendered.putpixel((0, 0), (32, 64, 96))
    rendered.save(image)
    model = FakeMultimodalModel(
        {
            "selected_view_ids": ["candidate_00"],
            "action": None,
            "reason": "best available corrective view",
        }
    )
    judge = OpenAICompatibleVLMJudge(model)

    judge.select_camera_views(
        {
            "selection_phase": "active_fallback",
            "evidence_deficiency": {
                "status": "insufficient",
                "reason_codes": [
                    "measured_local_visibility_insufficient",
                    "private-case-sentinel",
                ],
                "required_local_view_count": 2,
                "usable_local_view_count": 0,
                "local_path": "/private/case/path",
            },
            "metric": "support",
            "candidates": [
                {
                    "id": "private_candidate_id",
                    "pose": {"azimuth_degrees": 20.0},
                    "image_path": image.as_posix(),
                }
            ],
            "max_views": 1,
            "allow_adjustment": False,
            "allowed_actions": [],
            "preview_role": "highlighted_focus",
        }
    )

    assert model.calls[0]["kwargs"]["call_type"] == "vlm_camera_pose.active_fallback"
    context = json.loads(
        model.calls[0]["messages"][1]["content"][0]["text"].split("\n", 1)[1]
    )
    assert context["selection_phase"] == "active_fallback"
    assert context["evidence_deficiency"] == {
        "status": "insufficient",
        "reason_codes": ["measured_local_visibility_insufficient"],
        "required_local_view_count": 2,
        "usable_local_view_count": 0,
    }
    assert "private-case-sentinel" not in json.dumps(context)
    assert "/private/case/path" not in json.dumps(context)


def test_query_cov_camera_selector_excludes_blank_candidate_previews(tmp_path: Path) -> None:
    blank = tmp_path / "blank.png"
    usable = tmp_path / "usable.png"
    blank.write_bytes(b"blank")
    rendered = Image.new("RGB", (2, 2), (12, 34, 56))
    rendered.putpixel((0, 0), (78, 90, 12))
    rendered.save(usable)
    model = FakeMultimodalModel(
        {
            "selected_view_ids": ["candidate_00"],
            "action": None,
            "reason": "the usable candidate exposes the target",
        }
    )
    judge = OpenAICompatibleVLMJudge(model)

    result = judge.select_camera_views(
        {
            "metric": "support",
            "candidates": [
                {
                    "id": "blank",
                    "pose": {"location": [1, 2, 3]},
                    "image_path": str(blank),
                    "render_status": "blank",
                },
                {
                    "id": "usable",
                    "pose": {"location": [3, 2, 1]},
                    "image_path": str(usable),
                    "render_status": "ok",
                },
            ],
            "max_views": 1,
            "allow_adjustment": False,
        }
    )

    assert result["selected_view_ids"] == ["usable"]
    assert result["images_used"] == ["candidate_00"]
    context = model.calls[0]["messages"][1]["content"][0]["text"]
    assert '"id": "blank"' not in context


def test_query_cov_camera_selector_rejects_invalid_preview_image(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.png"
    invalid.write_bytes(b"not-an-image")
    judge = OpenAICompatibleVLMJudge(
        FakeMultimodalModel({"selected_view_ids": ["candidate_00"], "action": None})
    )

    with pytest.raises(ValueError, match="candidate_00 is not a valid decodable image"):
        judge.select_camera_views(
            {
                "metric": "support",
                "candidates": [
                    {"id": "private_internal_id", "pose": {}, "image_path": str(invalid)}
                ],
                "max_views": 1,
                "allow_adjustment": False,
            }
        )


@pytest.mark.parametrize(
    ("forbidden_key", "forbidden_value"),
    [("verdict", None), ("score", 0.5)],
)
def test_camera_selector_rejects_metric_decision_fields(
    tmp_path: Path,
    forbidden_key: str,
    forbidden_value: object,
) -> None:
    image = tmp_path / "candidate.png"
    _write_test_png(image)
    response = {
        "selected_view_ids": ["candidate_00"],
        "action": None,
        "reason": "best view",
        forbidden_key: forbidden_value,
    }
    judge = OpenAICompatibleVLMJudge(FakeMultimodalModel(response))

    with pytest.raises(
        ValueError,
        match="must not contain verdict or score",
    ):
        judge.select_camera_views(
            {
                "metric": "support",
                "candidates": [
                    {
                        "id": "candidate",
                        "pose": {},
                        "image_path": str(image),
                    }
                ],
                "max_views": 1,
                "allow_adjustment": False,
            }
        )
