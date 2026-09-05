from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

from PIL import Image
import pytest

from benchmark.evaluator.scene_quality.functional_acquisition import (
    FUNCTIONAL_ACQUISITION_PLAN_VERSION,
)
from benchmark.evaluator.scene_quality.functional_prejudgement import (
    DisabledFunctionalPrejudgementEvidenceSource,
    FrozenFunctionalPrejudgementEvidenceSource,
    FunctionalPrejudgementEvidenceRequest,
    FunctionalPrejudgementEvidenceResult,
    RuntimeFunctionalPrejudgementEvidenceSource,
    resolve_functional_prejudgement_evidence_source,
    validate_functional_prejudgement_evidence_config,
)
from benchmark.evaluator.scene_quality.interfaces import (
    resolve_scene_quality_config,
)
from benchmark.visual_judge.adapters.legacy_judge import (
    ControlledVLMJudge,
)
from benchmark.visual_judge.control_config import (
    resolve_vlm_evaluation_control,
)


def _request(tmp_path: Path) -> FunctionalPrejudgementEvidenceRequest:
    global_image = tmp_path / "global.png"
    Image.new("RGB", (32, 24), (90, 100, 110)).save(global_image)
    return FunctionalPrejudgementEvidenceRequest.create(
        scene={
            "scene_id": "scene",
            "objects": [
                {"id": "chair", "category": "chair"},
                {"id": "table", "category": "table"},
            ],
        },
        global_image_path=str(global_image),
        max_probe_units=4,
        groups=[
            {
                "group_id": "group_001",
                "object_ids": ["chair", "table"],
            }
        ],
        grouping_report={"status": "complete"},
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_payload(
    request: FunctionalPrejudgementEvidenceRequest,
    probe_path: Path,
) -> dict:
    return FunctionalPrejudgementEvidenceResult(
        status="complete",
        selected_judge_probe_paths=(str(probe_path),),
        cross_group_probe_paths=(str(probe_path),),
        cross_group_probe_packet={
            "schema_version": "functional_probe_judge_packet_v4",
            "planning_role": "visual_evidence_only_no_metric_verdict",
            "image_order": [],
        },
        group_owned_probe_packets={},
        telemetry={
            "planner_calls": 1,
            "usable_surface_detector_calls": 1,
            "selector_calls": 1,
            "preview_render_count": 4,
            "full_render_count": 1,
            "judge_facing_image_count": 1,
            "cache_hits": 0,
        },
        budget_usage={
            "max_probe_units": 4,
            "scheduled_probe_count": 1,
            "budget_exhausted": False,
        },
        source_identity=request.identity(),
        artifact_sha256={str(probe_path): _sha256(probe_path)},
        provenance={
            "acquisition_plan_version": (
                FUNCTIONAL_ACQUISITION_PLAN_VERSION
            ),
            "usable_surface_detector": {
                "implementation_id": "fake_detector",
                "version": "fake_v1",
            },
            "decision_authority": "none",
        },
        runtime_audit={
            "schema_version": "functional_probe_acquisition_v3",
            "decision_authority": "none",
            "selected_raw_rgb_paths": [str(probe_path)],
            "cross_group_evidence_paths": [str(probe_path)],
            "group_evidence_paths": {},
            "group_probe_packets": {},
        },
    ).to_dict()


def test_default_runtime_source_wraps_existing_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    probe = tmp_path / "probe.png"
    Image.new("RGB", (16, 16), (110, 120, 130)).save(probe)
    calls: list[dict] = []

    def acquire(**kwargs):
        calls.append(kwargs)
        return [str(probe)], {
            "schema_version": "functional_probe_acquisition_v3",
            "status": "complete",
            "decision_authority": "none",
            "planner_mode": "legacy_functional_probe_plan_v2",
            "cross_group_evidence_paths": [str(probe)],
            "group_probe_packets": {},
            "probe_results": [],
            "planned_probe_count": 1,
            "rendered_probe_count": 1,
            "unscheduled_discovery_items": [],
            "coverage_complete": True,
            "budget_exhausted": False,
        }

    monkeypatch.setattr(
        "benchmark.evaluator.scene_quality.functional_prejudgement."
        "acquire_functional_probe_evidence",
        acquire,
    )
    source = RuntimeFunctionalPrejudgementEvidenceSource(
        planner=object(),
        provider=object(),
    )
    result = source.prepare_functional_evidence(request)

    assert validate_functional_prejudgement_evidence_config(None)[
        "mode"
    ] == "runtime"
    assert result.selected_judge_probe_paths == (str(probe),)
    assert result.cross_group_probe_paths == (str(probe),)
    assert result.cross_group_probe_packet[
        "planning_role"
    ] == "visual_evidence_only_no_metric_verdict"
    assert result.decision_authority == "none"
    assert len(calls) == 1
    assert calls[0]["max_probe_units"] == 4
    assert calls[0]["groups"][0]["group_id"] == "group_001"


def test_disabled_source_performs_no_proactive_work(
    tmp_path: Path,
) -> None:
    result = DisabledFunctionalPrejudgementEvidenceSource().prepare_functional_evidence(
        _request(tmp_path)
    )

    assert result.status == "disabled"
    assert result.selected_judge_probe_paths == ()
    assert result.cross_group_probe_paths == ()
    assert result.telemetry == {
        "planner_calls": 0,
        "usable_surface_detector_calls": 0,
        "selector_calls": 0,
        "preview_render_count": 0,
        "full_render_count": 0,
        "judge_facing_image_count": 0,
        "cache_hits": 0,
    }


@pytest.mark.parametrize(
    "camera_policy",
    [
        "fixed",
        "deterministic_only",
        "vlm_only",
        "deterministic_then_vlm",
    ],
)
def test_prejudgement_mode_and_camera_policy_resolve_independently(
    camera_policy: str,
) -> None:
    scene_config = resolve_scene_quality_config(
        {
            "functional_prejudgement_evidence": {
                "mode": "disabled"
            }
        }
    )
    camera_control = resolve_vlm_evaluation_control(
        {
            "camera_acquisition": {
                "policy": camera_policy,
            }
        }
    )

    assert scene_config["functional_prejudgement_evidence"][
        "mode"
    ] == "disabled"
    assert camera_control.camera_acquisition_policy == camera_policy


def test_frozen_source_reuses_validated_artifacts_without_runtime(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    probe = tmp_path / "probe.png"
    Image.new("RGB", (16, 16), (120, 130, 140)).save(probe)
    payload = _frozen_payload(request, probe)
    source = FrozenFunctionalPrejudgementEvidenceSource(
        payload,
        expected_detector_implementation_id="fake_detector",
        expected_detector_version="fake_v1",
    )

    result = source.prepare_functional_evidence(request)

    assert result.selected_judge_probe_paths == (str(probe),)
    assert result.provenance["reuse_mode"] == "frozen"
    assert result.provenance["runtime_calls_performed"] is False


def test_frozen_source_preserves_exact_group_owned_packet(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    cross_probe = tmp_path / "cross.png"
    group_probe = tmp_path / "group.png"
    Image.new("RGB", (16, 16), (120, 130, 140)).save(
        cross_probe
    )
    Image.new("RGB", (16, 16), (140, 150, 160)).save(
        group_probe
    )
    payload = _frozen_payload(request, cross_probe)
    group_packet = {
        "schema_version": "functional_probe_judge_packet_v4",
        "planning_role": "visual_evidence_only_no_metric_verdict",
        "probe_inclusion_is_invalidity_prior": False,
        "group_id": "group_001",
        "observation_requests": [],
        "image_order": [
            {
                "image_alias": "group_probe_00",
                "role": "functional_probe",
                "artifact_id": str(group_probe),
            }
        ],
    }
    payload["selected_judge_probe_paths"] = [
        str(cross_probe),
        str(group_probe),
    ]
    payload["group_owned_probe_packets"] = {
        "group_001": deepcopy(group_packet)
    }
    payload["artifact_sha256"] = {
        str(cross_probe): _sha256(cross_probe),
        str(group_probe): _sha256(group_probe),
    }
    payload["runtime_audit"].update(
        {
            "selected_raw_rgb_paths": [
                str(cross_probe),
                str(group_probe),
            ],
            "group_evidence_paths": {
                "group_001": [str(group_probe)]
            },
            "group_probe_packets": {
                "group_001": deepcopy(group_packet)
            },
        }
    )
    source = FrozenFunctionalPrejudgementEvidenceSource(
        payload,
        expected_detector_implementation_id="fake_detector",
        expected_detector_version="fake_v1",
    )

    result = source.prepare_functional_evidence(request)

    assert result.group_owned_probe_packets == {
        "group_001": group_packet
    }
    assert result.selected_judge_probe_paths == (
        str(cross_probe),
        str(group_probe),
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "scene_hash",
        "object_ids",
        "group_ids",
        "schema_version",
        "planner_version",
        "detector_version",
        "artifact_hash",
        "missing_path",
        "cross_group_path_not_selected",
    ],
)
def test_frozen_source_fails_closed_on_identity_or_artifact_mismatch(
    tmp_path: Path,
    mutation: str,
) -> None:
    request = _request(tmp_path)
    probe = tmp_path / "probe.png"
    Image.new("RGB", (16, 16), (130, 140, 150)).save(probe)
    payload = _frozen_payload(request, probe)
    if mutation == "scene_hash":
        payload["source_identity"]["scene_sha256"] = "0" * 64
    elif mutation == "object_ids":
        payload["source_identity"]["object_ids"] = ["other"]
    elif mutation == "group_ids":
        payload["source_identity"]["group_ids"] = ["other"]
    elif mutation == "schema_version":
        payload["schema_version"] = "other"
    elif mutation == "planner_version":
        payload["provenance"]["acquisition_plan_version"] = "other"
    elif mutation == "detector_version":
        payload["provenance"]["usable_surface_detector"][
            "version"
        ] = "other"
    elif mutation == "artifact_hash":
        payload["artifact_sha256"][str(probe)] = "0" * 64
    elif mutation == "missing_path":
        missing = tmp_path / "missing.png"
        payload["selected_judge_probe_paths"] = [str(missing)]
        payload["cross_group_probe_paths"] = [str(missing)]
        payload["artifact_sha256"] = {str(missing): "0" * 64}
    else:
        other = tmp_path / "other.png"
        Image.new("RGB", (16, 16), (150, 160, 170)).save(other)
        payload["cross_group_probe_paths"] = [str(other)]
        payload["artifact_sha256"][str(other)] = _sha256(other)

    source = FrozenFunctionalPrejudgementEvidenceSource(
        payload,
        expected_detector_implementation_id="fake_detector",
        expected_detector_version="fake_v1",
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        source.prepare_functional_evidence(request)


def test_frozen_source_identity_includes_optional_overlay(
    tmp_path: Path,
) -> None:
    global_image = tmp_path / "global_identity_run.png"
    identity_image = tmp_path / "identity.png"
    probe = tmp_path / "probe_identity.png"
    Image.new("RGB", (32, 24), (90, 100, 110)).save(global_image)
    Image.new("RGB", (32, 24), (255, 0, 0)).save(identity_image)
    Image.new("RGB", (16, 16), (110, 120, 130)).save(probe)
    request = FunctionalPrejudgementEvidenceRequest.create(
        scene={
            "scene_id": "scene",
            "objects": [
                {"id": "chair", "category": "chair"},
                {"id": "table", "category": "table"},
            ],
        },
        global_image_path=str(global_image),
        identity_image_path=str(identity_image),
        identity_legend={"red": "chair", "blue": "table"},
        max_probe_units=4,
        groups=[
            {
                "group_id": "group_001",
                "object_ids": ["chair", "table"],
            }
        ],
        grouping_report={"status": "complete"},
    )
    payload = _frozen_payload(request, probe)
    payload["source_identity"]["identity_legend"] = {
        "red": "table",
        "blue": "chair",
    }

    source = FrozenFunctionalPrejudgementEvidenceSource(
        payload,
        expected_detector_implementation_id="fake_detector",
        expected_detector_version="fake_v1",
    )
    with pytest.raises(ValueError, match="identity_legend"):
        source.prepare_functional_evidence(request)


def test_frozen_mode_never_falls_back_to_runtime(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    probe = tmp_path / "probe.png"
    Image.new("RGB", (16, 16), (140, 150, 160)).save(probe)
    payload = _frozen_payload(request, probe)
    payload["source_identity"]["scene_sha256"] = "bad"
    source = resolve_functional_prejudgement_evidence_source(
        {
            "mode": "frozen",
            "frozen_result": payload,
            "expected_usable_surface_detector": {
                "implementation_id": "fake_detector",
                "version": "fake_v1",
            },
        },
        planner=object(),
        provider=object(),
    )

    with pytest.raises(ValueError, match="source mismatch"):
        source.prepare_functional_evidence(request)


def test_disabled_prejudgement_does_not_disable_judge_camera_controller(
    tmp_path: Path,
) -> None:
    initial = tmp_path / "initial.png"
    repair = tmp_path / "repair.png"
    initial_image = Image.new("RGB", (32, 24), (100, 110, 120))
    initial_image.putpixel((0, 0), (0, 0, 0))
    initial_image.save(initial)
    repair_image = Image.new("RGB", (32, 24), (120, 130, 140))
    repair_image.putpixel((0, 0), (255, 255, 255))
    repair_image.save(repair)

    class Selector:
        def __init__(self) -> None:
            self.calls = 0

        def select_camera_views(self, request):
            self.calls += 1
            return {
                "selected_view_ids": [request["candidates"][0]["id"]],
                "action": None,
                "reason": "selected",
            }

    class Provider:
        camera_selector_backend = "vlm"
        policy_config = {
            "max_selector_calls": 1,
            "max_camera_actions": 1,
        }

        def __init__(self, selector) -> None:
            self.selector = selector
            self.calls = 0
            self.last_call_usage = None

        def __call__(self, request):
            self.calls += 1
            self.selector.select_camera_views(
                {"candidates": [{"id": "provider-view"}]}
            )
            self.last_call_usage = {
                "call_id": f"provider-{self.calls}",
                "metric": request["metric"],
                "cache_hit": False,
                "evidence_refs": [str(repair)],
                "manifest_path": None,
                "selector_calls": 1,
                "camera_actions": 1,
            }
            return [str(repair)]

    class Judge:
        vlm_control_enabled = True

        def __init__(self) -> None:
            self.calls = 0

        def adjudicate_scene_quality(self, request):
            self.calls += 1
            if self.calls == 1:
                return {
                    "evidence_status": "insufficient",
                    "verdict": "ambiguous",
                    "confidence": 0.2,
                    "reason": "need another view",
                    "missing_evidence": ["interaction_side_visible"],
                    "defects": [],
                }
            return {
                "evidence_status": "sufficient",
                "verdict": "valid",
                "confidence": 0.9,
                "reason": "resolved",
                "missing_evidence": [],
                "defects": [],
            }

    disabled = DisabledFunctionalPrejudgementEvidenceSource()
    assert disabled.prepare_functional_evidence(
        _request(tmp_path)
    ).selected_judge_probe_paths == ()
    selector = Selector()
    provider = Provider(selector)
    wrapper = ControlledVLMJudge(
        Judge(),
        control=resolve_vlm_evaluation_control(
            {
                "camera_selector": {"backend": "vlm"},
                "budgets": {
                    "max_selector_calls": 2,
                    "max_camera_actions": 1,
                },
            }
        ),
        camera_provider=provider,
        camera_selector=selector,
    )

    response = wrapper.adjudicate_scene_quality(
        {
            "metric": "functional_consistency",
            "scene_summary": {"scene_id": "scene"},
            "render_evidence": [str(initial)],
            "judgment_scope": {
                "included": ["functional_consistency"]
            },
        }
    )

    assert response["verdict"] == "valid"
    assert provider.calls == 1
    assert selector.calls == 1
    assert wrapper.audit_records[0]["audit"][
        "selector_calls_used"
    ] == 1
