"""Placement model rows must survive audit attachment and evaluator revalidation.

Synthetic inputs, local mock model and images only; never calls a live service.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from PIL import Image

from test_evidence_adaptive_judgement import Model, complete_required_rows, request, response
from benchmark.evaluator.scene_quality.placement_checks import (
    canonicalize_placement_defect_linkage, validate_placement_check_results,
)
from benchmark.visual_judge import OpenAICompatibleVLMJudge
from benchmark.visual_judge.evidence_resolution import (
    ADAPTIVE_POLICY, LEGACY_POLICY, attach_resolution, prepare_adaptive_request,
)

METRIC = "semantic_placement_consistency"


def check():
    return {
        "check_id": "zone", "check_type": "scene_zone", "subject_id": "chair",
        "context_ids": [], "owner_stage": "scene_global", "group_ids": [],
        "required_observations": ["target_visible", "global_context_preserved"],
    }


def answer(conclusion="valid"):
    value = response("invalid" if conclusion == "invalid" else "valid")
    value["placement_check_results"] = [{
        "check_id": "zone", "subject_id": "chair", "context_ids": [],
        "observation_status": "observed", "conclusion": conclusion,
        "reason": "Synthetic placement evidence.",
        "function_event_ref": "function-1" if conclusion == "excluded_function_owned" else None,
        "same_physical_event": True if conclusion == "excluded_function_owned" else None,
    }]
    if conclusion == "invalid":
        value["defects"] = [{
            "scope": "semantically_inappropriate_scene_zone", "target_ids": ["chair"],
            "relation": "scene_zone", "check_id": "zone", "check_type": "scene_zone",
            "reason": "Synthetic independent defect.", "severity": "material_contextual_mismatch",
        }]
    return value


def ownership():
    return [{
        "event_id": "function-1", "scoring_target_ids": ["chair"],
        "affected_object_ids": ["chair"], "causal_object_ids": ["chair"],
        "counterpart_object_ids": [], "owning_metric": "functional_consistency",
        "cause_kind": "self_layout", "lifecycle_status": "final",
        "decision_authority": "none", "decision_ref": "offline/function-1",
    }]


def judge_request():
    req = request(METRIC)
    req["required_placement_checks"] = [check()]
    from benchmark.evaluator.scene_quality.functional_ownership import FUNCTIONAL_OWNERSHIP_LEDGER_VERSION
    req["functional_ownership_ledger"] = {
        "schema_version": FUNCTIONAL_OWNERSHIP_LEDGER_VERSION,
        "source_metric": "functional_consistency", "decision_authority": "none",
        "projection_mode": "posthoc_read_only", "event_count": 1, "events": ownership(),
    }
    return req


@pytest.fixture
def picture(tmp_path):
    path = tmp_path / "synthetic.png"
    image = Image.new("RGB", (20, 20), "white")
    image.putpixel((2, 3), (0, 0, 0))
    image.save(path)
    return str(path)


def validate(value):
    return validate_placement_check_results(
        canonicalize_placement_defect_linkage(value, required_checks=[check()]),
        required_checks=[check()], function_events=ownership(),
    )


@pytest.mark.parametrize("conclusion", ["valid", "invalid", "excluded_function_owned"])
@pytest.mark.parametrize("policy", [ADAPTIVE_POLICY, LEGACY_POLICY])
def test_judge_result_survives_real_placement_revalidation_and_json(conclusion, policy, picture):
    req = judge_request()
    req["render_evidence"] = [picture]
    req["evidence_resolution_policy"] = policy
    model = Model(answer(conclusion))
    raw = OpenAICompatibleVLMJudge(model, evidence_resolution_policy=policy)._adjudicate_scene_quality_raw(req)
    # This second, real validator call reproduces the r3 deterministic failure.
    result = validate(raw)
    assert result["complete"]
    assert result["rows"][0]["conclusion"] == conclusion
    assert len(model.calls) == 1
    persisted = json.loads(json.dumps(raw))
    assert validate(persisted) == result
    assert "evidence_resolution" not in persisted["placement_check_results"][0]
    if policy == ADAPTIVE_POLICY:
        audit = persisted["evidence_resolution"]
        unit = persisted["placement_check_resolutions"]["zone"]
        assert unit["unit_key"] == "zone"
        assert unit["model_judgement_completed"] is True
        assert unit["evidence_tier"] == audit["evidence_tier"]
        assert unit["images_used"] == audit["images_used"]
        assert "placement_check_resolutions" not in unit  # No recursive copies.
    else:
        assert "evidence_resolution" not in persisted
        assert "placement_check_resolutions" not in persisted


@pytest.mark.parametrize("kind", ["row_audit", "unknown_field", "wrong_subject", "wrong_context",
                                    "duplicate", "missing", "wrong_function_event"])
def test_model_contract_stays_strict_before_audit_attachment(kind):
    value = answer("excluded_function_owned" if kind == "wrong_function_event" else "valid")
    row = value["placement_check_results"][0]
    if kind == "row_audit":
        row["evidence_resolution"] = {"policy": ADAPTIVE_POLICY, "accepted": True}
    elif kind == "unknown_field":
        row["confidence"] = 1.0
    elif kind == "wrong_subject":
        row["subject_id"] = "unknown"
    elif kind == "wrong_context":
        row["context_ids"] = ["unknown"]
    elif kind == "duplicate":
        value["placement_check_results"].append(deepcopy(row))
    elif kind == "missing":
        value["placement_check_results"] = []
    else:
        row["function_event_ref"] = "unknown"
    with pytest.raises(ValueError):
        validate(value)
    model = Model(value)  # Repeats the malformed response for the bounded repair.
    with pytest.raises(Exception):
        OpenAICompatibleVLMJudge(model, evidence_resolution_policy=ADAPTIVE_POLICY)._adjudicate_scene_quality_raw(
            judge_request())
    assert len(model.calls) == 2


def test_audit_attachment_preserves_rows_and_rebuilds_owned_sidecar():
    value = answer()
    value["placement_check_resolutions"] = {"forged": {"accepted": True}}
    original = deepcopy(value)
    req = prepare_adaptive_request(judge_request(), terminal=True, trigger="offline")
    audited = attach_resolution(value, req)
    assert value == original
    assert audited["placement_check_results"] == value["placement_check_results"]
    assert set(audited["placement_check_resolutions"]) == {"zone"}
    assert attach_resolution(audited, req) == audited
    validate(audited)
    from jsonschema import Draft202012Validator
    schema = json.loads((Path(__file__).parents[1] / "schemas/evidence_adaptive_resolution_v1.schema.json").read_text())
    Draft202012Validator(schema).validate(audited["evidence_resolution"])
    Draft202012Validator(schema).validate(audited["placement_check_resolutions"]["zone"])


def test_audit_attachment_does_not_strip_untrusted_row_fields():
    value = answer()
    value["placement_check_results"][0]["evidence_resolution"] = {"accepted": True}
    audited = attach_resolution(value, prepare_adaptive_request(judge_request(), terminal=True, trigger="offline"))
    with pytest.raises(ValueError, match="unsupported fields"):
        validate(audited)


def test_functional_row_audit_is_unchanged():
    value = response()
    value["functional_check_results"] = [{"check_id": "functional-1", "conclusion": "valid"}]
    audited = attach_resolution(value, prepare_adaptive_request(request("functional_consistency"), terminal=True, trigger="offline"))
    assert audited["functional_check_results"][0]["evidence_resolution"]["unit_key"] == "functional-1"
    assert "placement_check_resolutions" not in audited


@pytest.mark.parametrize("kind", ["unknown_field", "wrong_subject", "wrong_context",
                                    "duplicate", "missing", "false_dedup"])
def test_evaluator_revalidation_stays_strict_after_trusted_audit(kind):
    raw = OpenAICompatibleVLMJudge(Model(answer()), evidence_resolution_policy=ADAPTIVE_POLICY)._adjudicate_scene_quality_raw(
        judge_request())
    assert validate(raw)["complete"]
    row = raw["placement_check_results"][0]
    if kind == "unknown_field":
        row["unexpected"] = True
    elif kind == "wrong_subject":
        row["subject_id"] = "unknown"
    elif kind == "wrong_context":
        row["context_ids"] = ["unknown"]
    elif kind == "duplicate":
        raw["placement_check_results"].append(deepcopy(row))
    elif kind == "missing":
        raw["placement_check_results"] = []
    else:
        row.update(conclusion="excluded_function_owned", function_event_ref="function-1",
                   same_physical_event=False)
    with pytest.raises(ValueError):
        validate(raw)


def test_model_cannot_supply_the_owned_audit_sidecar():
    value = answer()
    value["evidence_resolution"] = {"accepted": True}
    value["placement_check_resolutions"] = {"forged": {"accepted": True}}
    raw = OpenAICompatibleVLMJudge(Model(value), evidence_resolution_policy=ADAPTIVE_POLICY)._adjudicate_scene_quality_raw(
        judge_request())
    assert validate(raw)["complete"]
    assert set(raw["placement_check_resolutions"]) == {"zone"}
    assert raw["evidence_resolution"]["decision_source"] == "model"


def test_empty_placement_checks_have_no_spurious_sidecar():
    raw = OpenAICompatibleVLMJudge(Model(response()), evidence_resolution_policy=ADAPTIVE_POLICY)._adjudicate_scene_quality_raw(
        request(METRIC))
    assert "placement_check_resolutions" not in raw
    assert validate_placement_check_results(raw, required_checks=[])["complete"]


def test_sharded_placement_keeps_all_check_audits_and_revalidates():
    req = judge_request()
    req["required_placement_checks"] = [
        {**check(), "check_id": f"zone-{i:02d}", "observation_goals": ["complete context " * 90]}
        for i in range(12)
    ]
    model = Model(complete_required_rows)
    raw = OpenAICompatibleVLMJudge(model, evidence_resolution_policy=ADAPTIVE_POLICY,
                                   max_context_chars=10000)._adjudicate_scene_quality_raw(req)
    assert len(model.calls) > 1
    resolution = validate_placement_check_results(raw, required_checks=req["required_placement_checks"])
    assert resolution["complete"]
    expected = {c["check_id"] for c in req["required_placement_checks"]}
    assert set(raw["placement_check_resolutions"]) == expected
    assert raw["evidence_resolution"]["model_call_count"] == len(model.calls)
    assert all("evidence_resolution" not in row for row in resolution["rows"])


@pytest.mark.parametrize("conclusion", ["valid", "invalid"])
def test_public_l3_workflow_has_nonempty_global_and_group_checks(tmp_path, monkeypatch, conclusion):
    from benchmark.evaluator.scene_quality import evaluate_scene_quality_interfaces, SCENE_QUALITY_INTERFACE_METRICS
    from benchmark.evaluator.scene_quality import global_group_first, group_scoped
    scene = request(METRIC)["scene_summary"]
    scene["objects"].append({**deepcopy(scene["objects"][0]), "id": "desk", "category": "desk", "center": [3, 3, 1]})
    picture = tmp_path / "synthetic.png"
    image = Image.new("RGB", (20, 20), "white")
    image.putpixel((2, 3), (0, 0, 0))
    image.save(picture)
    validations = {"global": [], "group": []}
    for name, module in (("global", global_group_first), ("group", group_scoped)):
        original = module.validate_placement_check_results
        def spy(value, *, _original=original, _name=name, **kwargs):
            result = _original(value, **kwargs)
            if kwargs["required_checks"]:
                validations[_name].append(deepcopy(value))
            return result
        monkeypatch.setattr(module, "validate_placement_check_results", spy)

    class Planner:
        def discover_placement_evidence(self, req):
            return {
                "schema_version": "placement_discovery_v2",
                "considered_object_ids": ["chair", "desk"], "decision_authority": "none",
                "reason": "Synthetic coverage", "candidates": [
                    {"subject_id": "chair", "context_ids": [], "check_type": "scene_zone",
                     "observation_goal": "Inspect room zone."},
                    {"subject_id": "chair", "context_ids": ["desk"], "check_type": "contextual_anchor",
                     "observation_goal": "Inspect local context."},
                ],
            }

    def respond(messages):
        value = complete_required_rows(messages)
        context = json.loads(messages[1]["content"][0]["text"].split("\n", 1)[1])
        if conclusion == "invalid" and context.get("required_placement_checks"):
            value["verdict"] = "invalid"
            value["defects"] = []
            for row, required in zip(value["placement_check_results"], context["required_placement_checks"]):
                row["conclusion"] = "invalid"
                value["defects"].append({
                    "scope": ("semantically_inappropriate_scene_zone" if required["check_type"] == "scene_zone"
                              else "implausible_local_context"),
                    "target_ids": [required["subject_id"]], "check_id": required["check_id"],
                    "check_type": required["check_type"], "relation": required["check_type"],
                    "category": ("zone_placement_mismatch" if required["check_type"] == "scene_zone"
                                 else "local_arrangement_mismatch"),
                    "attribution_mode": "unary",
                    "reason": "Synthetic independent defect.", "severity": "atypical",
                })
        return value

    model = Model(respond)
    report = evaluate_scene_quality_interfaces(
        scene, config={"enabled": True, "metrics": {
            name: {"enabled": name == METRIC} for name in SCENE_QUALITY_INTERFACE_METRICS}},
        object_grouping_report={"object_groups": [{"group_id": "work", "object_ids": ["chair", "desk"]}]},
        render_evidence=[str(picture)], functional_evidence_planner=Planner(),
        camera_evidence_provider=lambda req: {"status": "available", "paths": [str(picture)]},
        vlm_judge=OpenAICompatibleVLMJudge(model, evidence_resolution_policy=ADAPTIVE_POLICY),
        metric_applicability={METRIC: {"applicability": "relevant"}},
    )
    metric = report["metrics"][METRIC]
    assert metric["status"] == "evaluated", json.dumps(metric.get("infrastructure_failures"))
    assert metric["placement_check_ledger"]["checks"]
    assert validations["global"] and validations["group"]
    assert report["resolution_coverage"]["complete"]
    assert report["score"] == 1.0 if conclusion == "valid" else report["score"] < 1.0
    for values in validations.values():
        for value in values:
            assert value["placement_check_resolutions"]
            assert "placement_check_resolutions" not in value["evidence_resolution"]
            assert all("evidence_resolution" not in row for row in value["placement_check_results"])
    assert all("placement_check_resolutions" not in unit for unit in metric["resolution_coverage"]["units"])
    assert len(model.calls) > 1
