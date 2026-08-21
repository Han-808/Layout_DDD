from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pytest

from benchmark.evaluation_campaign.cli import _require_private_binding_path
from benchmark.evaluation_campaign.config import (
    AttemptPolicy,
    CampaignConfigError,
    CasePlan,
    DatasetSpec,
    EvaluationCampaignSpec,
    JudgeProfile,
    KernelSpec,
    LocalBinding,
    OutputSpec,
    PriorAttemptRoot,
    SelectionSpec,
    load_campaign,
    load_profile_registry,
)
from benchmark.evaluation_campaign.dataset_identity import (
    inspect_evaluation_dataset,
    prepare_portable_dataset_view,
)
from benchmark.evaluation_campaign.kernel import (
    FROZEN_RUNNER,
    FROZEN_SELECTOR,
    ProcessInvocation,
    build_final_selector_invocation,
    build_pending_selector_invocation,
    build_round_invocation,
    build_smoke_invocation,
)
from benchmark.evaluation_campaign.orchestrator import (
    EvaluationCampaignOrchestrator,
    ExecutionResult,
    _campaign_lock,
    _attempt_counts,
    _next_round_index,
    _round_directories,
)
from benchmark.evaluation_campaign.provenance import (
    _python_dependency_closure,
    evaluation_source_manifest,
    protocol_manifest,
    validate_prior_attempt,
    validate_final_selection,
    write_selection_provenance,
)
from benchmark.evaluation_campaign.routes import (
    ResolvedJudgeRoute,
    open_judge_route,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _selection_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(row["benchmark_score_100"]) for row in rows]
    coverages = [
        float(row["grounded_score_fraction"])
        for row in rows
        if isinstance(row.get("grounded_score_fraction"), (int, float))
        and not isinstance(row.get("grounded_score_fraction"), bool)
    ]
    return {
        "case_count": len(rows),
        "published_case_count": len(rows),
        "official_score_100": sum(scores) / len(scores),
        "mean_combined_coverage_fraction": (
            sum(coverages) / len(coverages)
            if len(coverages) == len(rows)
            else None
        ),
        "infrastructure_failure_case_count": 0,
        "metrics": [],
    }


def _dataset(root: Path, case_ids: tuple[str, ...] = ("S100", "S101")) -> Path:
    root.mkdir(parents=True)
    rows = []
    for index, case_id in enumerate(case_ids):
        case = root / case_id
        (case / "scene").mkdir(parents=True)
        (case / "prepared").mkdir()
        (case / "evidence/collision_geometry").mkdir(parents=True)
        scene = json.dumps({"scene_id": case_id, "objects": []}).encode()
        blend = f"blend-{case_id}".encode()
        perspective = f"perspective-{case_id}".encode()
        top = f"top-{case_id}".encode()
        identity = f"identity-{case_id}".encode()
        annotation = json.dumps({"case_id": case_id, "metrics": {}}).encode()
        files = {
            case / "scene/canonical_scene.json": scene,
            case / "prepared/evaluation.blend": blend,
            case / "annotation.json": annotation,
            case / "evidence/standardized_perspective.png": perspective,
            case / "evidence/standardized_top.png": top,
            case / "evidence/standardized_identity_map.png": identity,
            case / "evidence/collision_geometry/object.ply": b"ply\n" + case_id.encode(),
        }
        for path, data in files.items():
            path.write_bytes(data)
        _write_json(
            case / "evidence/prepared_render_manifest.json",
            {
                "backend": "test",
                "blender_version": "test",
                "render_engine": "test",
                "render_config": {"width": 16, "height": 16},
                "architecture_policy_version": "test",
                "identity_legend": {"#010203": "object"},
                "scene_json": str((root.parent / "source" / case_id).resolve()),
            },
        )
        geometry = case / "evidence/collision_geometry/object.ply"
        _write_json(
            case / "evidence/collision_geometry_manifest.json",
            {
                "schema_version": "collision_geometry_v1",
                "units": "meter",
                "up_axis": "z",
                "export_summary": {"complete": 1},
                "objects": {
                    "object": {
                        "complete": True,
                        "geometry_path": str(geometry.resolve()),
                        "source_uri": str((root.parent / "assets/object.fbx").resolve()),
                        "vertex_count": 3,
                        "face_count": 1,
                    }
                },
            },
        )
        scene_sha = _sha(scene)
        _write_json(
            case / "case_manifest.json",
            {
                "schema_version": "camera_cal_scene_case_v1",
                "dataset_id": "fixture-dataset",
                "case_id": case_id,
                "status": "ready",
                "semantic_content_fingerprint": scene_sha,
                "source": {
                    "namespace": "generation-profile-that-is-not-identity",
                    "original_case_root": str((root.parent / "source" / case_id).resolve()),
                },
                "paths": {
                    "canonical_scene": "scene/canonical_scene.json",
                    "blend": "prepared/evaluation.blend",
                    "annotation": "annotation.json",
                    "evidence": {
                        "perspective": "evidence/standardized_perspective.png",
                        "top": "evidence/standardized_top.png",
                        "identity": "evidence/standardized_identity_map.png",
                    },
                },
                "critical_artifact_hashes": {
                    "canonical_scene": scene_sha,
                    "blend": _sha(blend),
                    "evidence_perspective": _sha(perspective),
                    "evidence_top": _sha(top),
                    "evidence_identity": _sha(identity),
                },
            },
        )
        rows.append(
            {
                "case_id": case_id,
                "status": "ready",
                "semantic_content_fingerprint": scene_sha,
            }
        )
    _write_json(
        root / "dataset_manifest.json",
        {
            "schema_version": "camera_cal_scenesets_manifest_v1",
            "dataset_id": "fixture-dataset",
            "source_dataset": str((root.parent / "generation-source").resolve()),
            "case_count": len(case_ids),
            "case_ids": list(case_ids),
            "cases": rows,
            "all_cases_ready": True,
        },
    )
    return root


def _relocate_dataset(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    manifest = json.loads((target / "dataset_manifest.json").read_text())
    manifest["source_dataset"] = str((target.parent / "other-generation-source").resolve())
    _write_json(target / "dataset_manifest.json", manifest)
    for case in target.glob("S*"):
        case_manifest = json.loads((case / "case_manifest.json").read_text())
        case_manifest["source"]["original_case_root"] = str(
            (target.parent / "other-source" / case.name).resolve()
        )
        _write_json(case / "case_manifest.json", case_manifest)
        render = json.loads(
            (case / "evidence/prepared_render_manifest.json").read_text()
        )
        render["scene_json"] = str(
            (target.parent / "other-source" / case.name / "scene.json").resolve()
        )
        _write_json(case / "evidence/prepared_render_manifest.json", render)
        collision = json.loads(
            (case / "evidence/collision_geometry_manifest.json").read_text()
        )
        collision["objects"]["object"]["geometry_path"] = str(
            (case / "evidence/collision_geometry/object.ply").resolve()
        )
        collision["objects"]["object"]["source_uri"] = str(
            (target.parent / "other-assets/object.fbx").resolve()
        )
        _write_json(case / "evidence/collision_geometry_manifest.json", collision)
    return target


def _profile(*, managed: bool = False) -> JudgeProfile:
    payload = {
        "profile_id": ("profile-api2" if managed else "profile-api1"),
        "binding_id": ("binding-api2" if managed else "binding-api1"),
        "adapter": (
            "openai_compatible_managed_proxy_v1"
            if managed
            else "openai_compatible_direct_v1"
        ),
        "model_alias": "gpt-5.6-sol",
        "request_protocol": "openai_chat_completions_v1",
        "model_profile": {
            "send_temperature": False,
            "response_format_json": False,
            "max_tokens_field": "max_tokens",
        },
        "wire_policy": {
            "min_request_interval_seconds": 1.0 if managed else 0.0,
            "external_model_discovery": True,
            "external_multimodal_smoke": True,
        },
        "adapter_attestation": (
            {
                "schema_version": "litellm_model_entry_v1",
                "model_name": "gpt-5.6-sol",
                "provider_model": "openai/api_azure_openai_gpt-5.6-sol",
                "base_model": "gpt-5.6-sol",
                "reasoning_effort": "xhigh",
                "additional_drop_params": ["output_config", "temperature"],
                "drop_params": True,
                "num_retries": 1,
                "request_timeout_seconds": 3000,
            }
            if managed
            else None
        ),
    }
    return JudgeProfile(
        **payload,
        fingerprint_sha256=_sha(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode()
        ),
    )


def _campaign(
    tmp_path: Path,
    dataset_root: Path,
    dataset_sha: str,
    *,
    profile: JudgeProfile | None = None,
    max_attempts: int = 3,
) -> EvaluationCampaignSpec:
    selected_profile = profile or _profile()
    config_path = tmp_path / "campaign.json"
    config_path.write_text("{}\n")
    public_profile = selected_profile.public_dict()
    public_profile.pop("profile_fingerprint_sha256")
    _write_json(
        tmp_path / "profiles.json",
        {
            "schema_version": "public_judge_profile_registry_v1",
            "profiles": [public_profile],
        },
    )
    return EvaluationCampaignSpec(
        source_path=config_path,
        source_sha256=_sha(config_path.read_bytes()),
        campaign_id="fixture-campaign",
        model_label="fixture-model",
        profile_registry=tmp_path / "profiles.json",
        judge_profile_id=selected_profile.profile_id,
        dataset=DatasetSpec(
            root=dataset_root,
            expected_dataset_id="fixture-dataset",
            expected_fingerprint_sha256=dataset_sha,
            expected_case_ids=("S100", "S101"),
            smoke_case_id="S100",
        ),
        case_plan=CasePlan(
            run_case_ids=("S100", "S101"),
            selection_case_ids=("S100", "S101"),
            prior_attempt_roots=(),
        ),
        kernel=KernelSpec(
            profile="camera_cal_scene_level_v9_exact",
            grouping_config=ROOT / "configs/grouping/vlm_visual_evidence_scope_v2.yaml",
            metric_selection_mode="runner_default",
            metrics=(),
            functional_group_local_granularity="per_check",
            functional_group_local_evidence_policy="shared_group_bank",
            deduction_multiplier=2.0,
            l3_only=False,
            blender_timeout_seconds=1800,
            continue_on_error=True,
            terminal_progress=True,
            export_audit_graphs=True,
        ),
        attempt_policy=AttemptPolicy(
            max_new_attempts_per_case=max_attempts,
            retry_delay_seconds=0.0,
            max_workers=2,
            round0_preflight_attempts=10,
            retry_preflight_attempts=3,
            preflight_timeout_seconds=3000,
        ),
        outputs=OutputSpec(
            attempt_parent=tmp_path / "attempts",
            final_selection_root=tmp_path / "final",
        ),
        selection=SelectionSpec(
            policy="first_publishable_v1",
        ),
    )


def _resolved(profile: JudgeProfile | None = None) -> ResolvedJudgeRoute:
    selected = profile or _profile()
    return ResolvedJudgeRoute(
        profile=selected,
        deployment_kind="test",
        binding_fingerprint_sha256="1" * 64,
        route_fingerprint_sha256="2" * 64,
        endpoint="https://hidden.example.invalid/v1",
        api_key_env="HIDDEN_KEY_ENV",
        secret_environment={"HIDDEN_KEY_ENV": "sentinel-secret"},
    )


def _strict_final_fixture(
    final: Path,
    campaign: EvaluationCampaignSpec,
    sources: list[tuple[Path, str]],
) -> dict[Path, str]:
    records: list[dict[str, Any]] = []
    routes: dict[Path, str] = {}
    for index, (root, case_id) in enumerate(sources):
        case = root / "cases" / case_id
        case.mkdir(parents=True, exist_ok=True)
        _write_json(
            case / "case_run_manifest.json",
            {
                "status": "complete",
                "final_decision_status": "resolved",
                "l1_engineering_failure": False,
            },
        )
        _write_json(
            case / "evaluation_report.json",
            {
                "evaluation_status": "complete",
                "benchmark_score_status": "complete",
                "benchmark_score_100": 50 + index,
            },
        )
        _write_json(case / "l1_report.json", {})
        _write_json(case / "scene_quality_report.json", {})
        records.append(
            {
                "case_id": case_id,
                "selected_attempt_index": index,
                "source_run": str(root.resolve()),
                "source_case": str(case.resolve()),
                "storage": "absolute_directory_symlink",
                "status": "complete",
                "final_decision_status": "resolved",
                "benchmark_score_100": 50 + index,
                "benchmark_score_status": "complete",
                "evaluation_status": "complete",
                "grounded_score_fraction": None,
                "l1_engineering_failure": False,
                "case_manifest_sha256": _sha((case / "case_run_manifest.json").read_bytes()),
                "evaluation_report_sha256": _sha((case / "evaluation_report.json").read_bytes()),
                "l1_report_sha256": _sha((case / "l1_report.json").read_bytes()),
                "l3_report_sha256": _sha((case / "scene_quality_report.json").read_bytes()),
            }
        )
        routes[root.resolve()] = f"profile-{index}"
    final.mkdir(parents=True)
    (final / "cases").mkdir()
    for row in records:
        (final / "cases" / row["case_id"]).symlink_to(
            row["source_case"], target_is_directory=True
        )
    provider_route = "profiles:" + ",".join(dict.fromkeys(routes.values()))
    selection = {
        "schema_version": "scene_level_first_publishable_selection_v1",
        "status": "complete",
        "model_label": campaign.model_label,
        "evaluator_model": "gpt-5.6-sol",
        "provider_route": provider_route,
        "case_count": len(records),
        "attempt_roots": [str(root.resolve()) for root, _ in sources],
        "selection_policy": "first_publishable_attempt_only_no_score_selection",
        "publishability_policy": {
            "case_status": "complete",
            "final_decision_status": "resolved",
            "l1_engineering_failure": False,
            "evaluation_status": "complete",
            "benchmark_score_status": "complete",
            "benchmark_score_100": "finite_number",
        },
        "cases": records,
    }
    _write_json(final / "selection_manifest.json", selection)
    _write_json(final / "run_manifest.json", selection)
    _write_json(
        final / "summary.json",
        {
            "schema_version": "selected_scene_level_summary_v1",
            "status": "complete",
            "model_label": campaign.model_label,
            "evaluator_model": "gpt-5.6-sol",
            "provider_route": provider_route,
            "totals": {
                "cases": len(records),
                "successful": len(records),
                "failed": 0,
                "final_unresolved": 0,
                "final_infrastructure_failure": 0,
                "l1_engineering_failure_cases": 0,
                "retry_cases": sum(
                    int(row["selected_attempt_index"] > 0) for row in records
                ),
                "baseline_cases": sum(
                    int(row["selected_attempt_index"] == 0) for row in records
                ),
                "attempt_rounds": len(sources),
            },
            "aggregate": _selection_aggregate(records),
        },
    )
    return routes


def test_dataset_identity_is_portable_across_source_and_root_paths(
    tmp_path: Path,
) -> None:
    original = _dataset(tmp_path / "original")
    relocated = _relocate_dataset(original, tmp_path / "relocated")

    left = inspect_evaluation_dataset(original, expected_case_ids=("S100", "S101"))
    right = inspect_evaluation_dataset(relocated, expected_case_ids=("S100", "S101"))

    assert left.portable_fingerprint_sha256 == right.portable_fingerprint_sha256
    assert left.raw_manifest_sha256 != right.raw_manifest_sha256


@pytest.mark.parametrize(
    "relative",
    [
        "S100/scene/canonical_scene.json",
        "S100/annotation.json",
        "S100/prepared/evaluation.blend",
        "S100/evidence/standardized_perspective.png",
        "S100/evidence/collision_geometry/object.ply",
    ],
)
def test_dataset_identity_changes_when_consumed_artifact_changes(
    tmp_path: Path, relative: str
) -> None:
    original = _dataset(tmp_path / "original")
    baseline = inspect_evaluation_dataset(
        original, expected_case_ids=("S100", "S101")
    )
    target = original / relative
    if relative.endswith("annotation.json"):
        _write_json(
            target,
            {"case_id": "S100", "metrics": {}, "scene_type": "changed"},
        )
    else:
        target.write_bytes(target.read_bytes() + b"changed")
    if relative.endswith("canonical_scene.json"):
        manifest_path = original / "S100/case_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["semantic_content_fingerprint"] = _sha(target.read_bytes())
        manifest["critical_artifact_hashes"]["canonical_scene"] = _sha(
            target.read_bytes()
        )
        _write_json(manifest_path, manifest)
    elif relative.endswith("evaluation.blend"):
        manifest_path = original / "S100/case_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["critical_artifact_hashes"]["blend"] = _sha(target.read_bytes())
        _write_json(manifest_path, manifest)
    elif relative.endswith("standardized_perspective.png"):
        manifest_path = original / "S100/case_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["critical_artifact_hashes"]["evidence_perspective"] = _sha(
            target.read_bytes()
        )
        _write_json(manifest_path, manifest)
    changed = inspect_evaluation_dataset(
        original, expected_case_ids=("S100", "S101")
    )
    assert baseline.portable_fingerprint_sha256 != changed.portable_fingerprint_sha256


def test_public_configs_reject_deployment_fields_and_unknowns(tmp_path: Path) -> None:
    campaign = {
        "schema_version": "scene_evaluation_campaign_v1",
        "campaign_id": "test-campaign",
        "model_label": "model",
        "profile_registry": "profiles.json",
        "judge_profile_id": "profile-api1",
        "endpoint": "https://should-not-be-public.invalid/v1",
    }
    path = tmp_path / "campaign.json"
    _write_json(path, campaign)
    with pytest.raises(CampaignConfigError):
        load_campaign(path, repo_root=tmp_path)

    registry = {
        "schema_version": "public_judge_profile_registry_v1",
        "profiles": [
            {
                **_profile().public_dict(),
                "credential_env": "SHOULD_NOT_BE_PUBLIC",
            }
        ],
    }
    _write_json(tmp_path / "profiles.json", registry)
    with pytest.raises(CampaignConfigError, match="deployment-only|fields differ"):
        load_profile_registry(tmp_path / "profiles.json")


def test_static_check_allows_missing_dataset_and_does_not_load_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "benchmark.evaluation_campaign.orchestrator.evaluation_source_manifest",
        lambda root: {"manifest_sha256": "f" * 64},
    )
    for relative in (
        ".venv/bin/python",
        "scripts/run_camera_cal_scene_level.py",
        "scripts/select_first_publishable_scene_evaluations.py",
        "scripts/check_model_endpoint.py",
        "configs/grouping/grouping.yaml",
        "profiles.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    campaign = _campaign(
        tmp_path,
        tmp_path / "missing-dataset",
        "0" * 64,
    )
    campaign = replace(
        campaign,
        profile_registry=tmp_path / "profiles.json",
        kernel=replace(
            campaign.kernel,
            grouping_config=tmp_path / "configs/grouping/grouping.yaml",
        ),
    )
    orchestrator = EvaluationCampaignOrchestrator(
        campaign,
        _profile(),
        None,
        repo_root=tmp_path,
        python_executable=tmp_path / ".venv/bin/python",
    )
    result = orchestrator.check()
    assert result["dataset_status"] == "missing_allowed_for_static_check"
    assert result["binding_status"] == "not_loaded_static_check"
    assert result["network_used"] is False
    assert result["credential_read"] is False


def test_direct_route_keeps_endpoint_secret_and_env_selection_out_of_public_manifest(
    tmp_path: Path,
) -> None:
    profile = _profile()
    binding = LocalBinding(
        binding_id=profile.binding_id,
        adapter=profile.adapter,
        values={
            "endpoint": "https://private-deployment.example.invalid/v1",
            "credential_env": "PRIVATE_JUDGE_CREDENTIAL",
        },
    )
    probes = []
    session = open_judge_route(
        profile,
        binding,
        repo_root=tmp_path,
        environ={"PATH": "/bin", "PRIVATE_JUDGE_CREDENTIAL": "sentinel-secret"},
        model_probe=lambda endpoint, model, credential: probes.append(
            (endpoint, model, credential)
        ),
    )
    with session as route:
        public = json.dumps(route.public_manifest())
        assert "private-deployment" not in public
        assert "PRIVATE_JUDGE_CREDENTIAL" not in public
        assert "sentinel-secret" not in public
        env = route.evaluator_environment({"PATH": "/bin"})
        assert env["JUDGE_ENDPOINT"].startswith("https://private-deployment")
        assert env["PRIVATE_JUDGE_CREDENTIAL"] == "sentinel-secret"
    assert probes == [
        (
            "https://private-deployment.example.invalid/v1",
            "gpt-5.6-sol",
            "sentinel-secret",
        )
    ]


def test_managed_proxy_is_owned_loopback_and_upstream_secret_isolated(
    tmp_path: Path,
) -> None:
    launcher = tmp_path / "src/benchmark/evaluation_campaign/owned_proxy_launcher.py"
    launcher.parent.mkdir(parents=True)
    launcher.write_bytes(
        (ROOT / "src/benchmark/evaluation_campaign/owned_proxy_launcher.py").read_bytes()
    )
    adapter = tmp_path / "profile.yaml"
    adapter.write_text(
        "model_list:\n"
        "  - model_name: gpt-5.6-sol\n"
        "    litellm_params:\n"
        "      model: openai/api_azure_openai_gpt-5.6-sol\n"
        "      reasoning_effort: xhigh\n"
        "      additional_drop_params: [output_config, temperature]\n"
        "    model_info: {base_model: gpt-5.6-sol}\n"
        "litellm_settings: {drop_params: true, num_retries: 1, request_timeout: 3000}\n"
    )
    profile = _profile(managed=True)
    binding = LocalBinding(
        binding_id=profile.binding_id,
        adapter=profile.adapter,
        values={
            "launcher_path": "src/benchmark/evaluation_campaign/owned_proxy_launcher.py",
            "launcher_sha256": _sha(launcher.read_bytes()),
            "adapter_profile_path": "profile.yaml",
            "adapter_profile_sha256": _sha(adapter.read_bytes()),
            "upstream_environment": {
                "UPSTREAM_BASE_URL": "LOCAL_BASE_URL",
                "UPSTREAM_CREDENTIAL": "LOCAL_UPSTREAM_CREDENTIAL",
            },
            "local_master_key_env": "LOCAL_MASTER_KEY",
            "local_port_env": "LOCAL_PROXY_PORT",
            "local_port": 43123,
            "startup_timeout_seconds": 1,
        },
    )

    class FakeProcess:
        pid = 42

        def __init__(self) -> None:
            self.returncode = None
            self.terminated = False

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            return int(self.returncode or 0)

        def kill(self) -> None:
            self.returncode = -9

    process = FakeProcess()
    captured: dict[str, Any] = {}

    def factory(argv: list[str], **kwargs: Any) -> FakeProcess:
        captured.update(argv=argv, **kwargs)
        return process

    session = open_judge_route(
        profile,
        binding,
        repo_root=tmp_path,
        environ={
            "PATH": "/bin",
            "LOCAL_BASE_URL": "https://private-upstream.invalid",
            "LOCAL_UPSTREAM_CREDENTIAL": "upstream-secret",
        },
        process_factory=factory,
        readiness_probe=lambda endpoint, key, model: endpoint.endswith(":43123/v1")
        and bool(key)
        and model == "gpt-5.6-sol",
        port_probe=lambda host, port: False,
        sleep=lambda _: None,
    )
    with session as route:
        assert route.endpoint == "http://127.0.0.1:43123/v1"
        assert Path(captured["argv"][0]).name.startswith("python")
        assert captured["argv"][1:8] == [
            str(launcher),
            "--config",
            str(adapter),
            "--host",
            "127.0.0.1",
            "--port",
            "43123",
        ]
        assert captured["argv"][8] == "--ownership-token"
        assert len(captured["argv"][9]) >= 32
        assert captured["env"]["UPSTREAM_CREDENTIAL"] == "upstream-secret"
        evaluator_env = route.evaluator_environment(
            {
                "PATH": "/bin",
                "LOCAL_BASE_URL": "https://private-upstream.invalid",
                "LOCAL_UPSTREAM_CREDENTIAL": "upstream-secret",
                "UPSTREAM_BASE_URL": "https://private-upstream.invalid",
                "UPSTREAM_CREDENTIAL": "upstream-secret",
            }
        )
        assert "LOCAL_UPSTREAM_CREDENTIAL" not in evaluator_env
        assert "UPSTREAM_CREDENTIAL" not in evaluator_env
        assert "LOCAL_BASE_URL" not in evaluator_env
        assert "UPSTREAM_BASE_URL" not in evaluator_env
        assert evaluator_env["LOCAL_MASTER_KEY"]
        public = json.dumps(route.public_manifest())
        assert "43123" not in public
        assert "upstream-secret" not in public
    assert process.terminated is True


def test_managed_proxy_refuses_occupied_port_without_starting(tmp_path: Path) -> None:
    launcher = tmp_path / "src/benchmark/evaluation_campaign/owned_proxy_launcher.py"
    launcher.parent.mkdir(parents=True)
    launcher.write_bytes(
        (ROOT / "src/benchmark/evaluation_campaign/owned_proxy_launcher.py").read_bytes()
    )
    adapter = tmp_path / "profile.yaml"
    adapter.write_text(
        "model_list:\n"
        "  - model_name: gpt-5.6-sol\n"
        "    litellm_params:\n"
        "      model: openai/api_azure_openai_gpt-5.6-sol\n"
        "      reasoning_effort: xhigh\n"
        "      additional_drop_params: [output_config, temperature]\n"
        "    model_info: {base_model: gpt-5.6-sol}\n"
        "litellm_settings: {drop_params: true, num_retries: 1, request_timeout: 3000}\n"
    )
    profile = _profile(managed=True)
    binding = LocalBinding(
        binding_id=profile.binding_id,
        adapter=profile.adapter,
        values={
            "launcher_path": "src/benchmark/evaluation_campaign/owned_proxy_launcher.py",
            "launcher_sha256": _sha(launcher.read_bytes()),
            "adapter_profile_path": "profile.yaml",
            "adapter_profile_sha256": _sha(adapter.read_bytes()),
            "upstream_environment": {"UPSTREAM": "SOURCE_ENV"},
            "local_master_key_env": "LOCAL_MASTER_KEY",
            "local_port_env": "LOCAL_PROXY_PORT",
            "local_port": 43123,
            "startup_timeout_seconds": 1,
        },
    )
    started = []
    session = open_judge_route(
        profile,
        binding,
        repo_root=tmp_path,
        environ={"SOURCE_ENV": "secret"},
        process_factory=lambda *args, **kwargs: started.append(args),
        port_probe=lambda host, port: True,
    )
    with pytest.raises(CampaignConfigError, match="occupied"):
        session.__enter__()
    assert started == []


def test_exact_round_argv_and_retry_preflight_mapping(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "dataset")
    identity = inspect_evaluation_dataset(dataset, expected_case_ids=("S100", "S101"))
    campaign = _campaign(tmp_path, dataset, identity.portable_fingerprint_sha256)
    route = _resolved()
    round0 = build_round_invocation(
        campaign,
        route,
        repo_root=ROOT,
        python_executable=ROOT / ".venv/bin/python",
        case_ids=("S100", "S101"),
        round_index=0,
        round_root=tmp_path / "attempts/round_00",
        base_environment={"PATH": "/bin"},
    )
    retry = build_round_invocation(
        campaign,
        route,
        repo_root=ROOT,
        python_executable=ROOT / ".venv/bin/python",
        case_ids=("S101",),
        round_index=1,
        round_root=tmp_path / "attempts/round_01",
        base_environment={"PATH": "/bin"},
    )
    assert round0.argv[:2] == (str(ROOT / ".venv/bin/python"), FROZEN_RUNNER)
    assert round0.argv.count("--case-id") == 2
    assert "--metric" not in round0.argv
    assert round0.argv[round0.argv.index("--deduction-multiplier") + 1] == "2"
    assert round0.argv[round0.argv.index("--endpoint-preflight-attempts") + 1] == "10"
    assert retry.argv[retry.argv.index("--endpoint-preflight-attempts") + 1] == "3"
    assert retry.argv.count("--case-id") == 1
    assert "--no-resume" in retry.argv
    assert "--continue-on-error" in retry.argv
    assert "--terminal-progress" in retry.argv
    assert "--export-audit-graphs" in retry.argv


def test_smoke_and_selector_builders_preserve_exact_contract_without_public_secret(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path / "dataset")
    identity = inspect_evaluation_dataset(dataset, expected_case_ids=("S100", "S101"))
    campaign = _campaign(tmp_path, dataset, identity.portable_fingerprint_sha256)
    route = _resolved()
    smoke = build_smoke_invocation(
        campaign,
        route,
        repo_root=ROOT,
        python_executable=ROOT / ".venv/bin/python",
    )
    assert "--no-send-temperature" in smoke.argv
    assert "--no-response-format-json" in smoke.argv
    assert "--multimodal" in smoke.argv
    assert "hidden.example.invalid" in " ".join(smoke.argv)
    assert "hidden.example.invalid" not in json.dumps(smoke.public_dict())
    pending = build_pending_selector_invocation(
        campaign,
        repo_root=ROOT,
        python_executable=ROOT / ".venv/bin/python",
        chronological_attempt_roots=(tmp_path / "round_00",),
    )
    assert pending.argv[1] == FROZEN_SELECTOR
    assert pending.argv[-1] == "--pending-only"
    final = build_final_selector_invocation(
        campaign,
        repo_root=ROOT,
        python_executable=ROOT / ".venv/bin/python",
        chronological_attempt_roots=(tmp_path / "round_00", tmp_path / "round_01"),
    )
    assert "--provider-route" in final.argv
    assert final.argv.count("--attempt-root") == 2


def test_campaign_rounds_only_pending_and_selects_chronological_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset(tmp_path / "dataset")
    identity = inspect_evaluation_dataset(dataset, expected_case_ids=("S100", "S101"))
    campaign = _campaign(tmp_path, dataset, identity.portable_fingerprint_sha256)
    profile = _profile()
    route = _resolved(profile)
    invocations: list[tuple[str, ...]] = []
    pending_responses = iter(("S100\nS101\n", "S101\n", ""))
    monkeypatch.setattr(
        "benchmark.evaluation_campaign.orchestrator.write_selection_provenance",
        lambda *args, **kwargs: {},
    )

    class RouteSession:
        def __enter__(self) -> ResolvedJudgeRoute:
            return route

        def __exit__(self, *args: Any) -> None:
            return None

    def executor(
        invocation: ProcessInvocation,
        *,
        capture_output: bool,
        on_started: Any = None,
    ) -> ExecutionResult:
        invocations.append(invocation.argv)
        script = invocation.argv[1]
        if script.endswith("check_model_endpoint.py"):
            return ExecutionResult(0, "", "", 10)
        if script == FROZEN_RUNNER:
            if on_started:
                on_started(100 + len(invocations))
            out = Path(invocation.argv[invocation.argv.index("--output-root") + 1])
            case_ids = tuple(
                invocation.argv[index + 1]
                for index, value in enumerate(invocation.argv)
                if value == "--case-id"
            )
            for case_id in case_ids:
                case = out / "cases" / case_id
                case.mkdir(parents=True, exist_ok=True)
                _write_json(case / "case_run_manifest.json", {"status": "complete"})
                _write_json(case / "evaluation_report.json", {"score": 1})
            _write_json(out / "run_manifest.json", {"status": "complete"})
            return ExecutionResult(0, "", "", 200 + len(invocations))
        if script == FROZEN_SELECTOR and "--pending-only" in invocation.argv:
            return ExecutionResult(0, next(pending_responses), "", 20)
        if script == FROZEN_SELECTOR:
            roots = [
                Path(invocation.argv[index + 1])
                for index, value in enumerate(invocation.argv)
                if value == "--attempt-root"
            ]
            final_root = Path(
                invocation.argv[invocation.argv.index("--output-root") + 1]
            )
            final_root.mkdir(parents=True)
            _write_json(
                final_root / "selection_manifest.json",
                {
                    "cases": [
                        {
                            "case_id": "S100",
                            "source_run": str(roots[0]),
                            "source_case": str(roots[0] / "cases/S100"),
                        },
                        {
                            "case_id": "S101",
                            "source_run": str(roots[1]),
                            "source_case": str(roots[1] / "cases/S101"),
                        },
                    ]
                },
            )
            return ExecutionResult(0, "{}", "", 30)
        raise AssertionError(script)

    monkeypatch.setattr(
        "benchmark.evaluation_campaign.orchestrator.protocol_manifest",
        lambda *args, **kwargs: {
            "protocol_fingerprint_sha256": "3" * 64,
            "route_fingerprint_sha256": "2" * 64,
            "adapter_attestation_sha256": "0" * 64,
            "source_manifest": {},
        },
    )
    monkeypatch.setattr(
        "benchmark.evaluation_campaign.orchestrator.git_state",
        lambda root: {"commit": "test", "dirty": False},
    )
    orchestrator = EvaluationCampaignOrchestrator(
        campaign,
        profile,
        LocalBinding(profile.binding_id, profile.adapter, {}),
        repo_root=ROOT,
        python_executable=ROOT / ".venv/bin/python",
        executor=executor,
        route_session_factory=lambda *args, **kwargs: RouteSession(),
        sleep=lambda _: None,
    )
    result = orchestrator.run()
    assert result.status == "complete"
    runner_calls = [argv for argv in invocations if argv[1] == FROZEN_RUNNER]
    assert runner_calls[0].count("--case-id") == 2
    assert runner_calls[1].count("--case-id") == 1
    assert "S101" in runner_calls[1] and "S100" not in runner_calls[1]
    selection = json.loads(
        (campaign.outputs.final_selection_root / "selection_manifest.json").read_text()
    )
    assert [row["source_run"] for row in selection["cases"]] == [
        str((campaign.outputs.attempt_parent / "round_00").resolve()),
        str((campaign.outputs.attempt_parent / "round_01").resolve()),
    ]
    public_state = (
        campaign.outputs.attempt_parent / "campaign_manifest.json"
    ).read_text()
    assert str(tmp_path) not in public_state
    assert "runner_pid" not in public_state and "dirty_paths" not in public_state


def test_resume_preserves_interrupted_round_and_uses_new_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset(tmp_path / "dataset")
    identity = inspect_evaluation_dataset(dataset, expected_case_ids=("S100", "S101"))
    campaign = _campaign(tmp_path, dataset, identity.portable_fingerprint_sha256)
    profile = _profile()
    route = _resolved(profile)
    monkeypatch.setattr(
        "benchmark.evaluation_campaign.orchestrator.write_selection_provenance",
        lambda *args, **kwargs: {},
    )

    class RouteSession:
        def __enter__(self) -> ResolvedJudgeRoute:
            return route

        def __exit__(self, *args: Any) -> None:
            return None

    calls = []

    def crashing_executor(invocation: ProcessInvocation, **kwargs: Any) -> ExecutionResult:
        if invocation.argv[1] == FROZEN_SELECTOR:
            return ExecutionResult(0, "S100\nS101\n", "")
        if invocation.argv[1] == FROZEN_RUNNER:
            kwargs["on_started"](987654321)
            raise RuntimeError("simulated controller crash")
        return ExecutionResult(0, "", "")

    monkeypatch.setattr(
        "benchmark.evaluation_campaign.orchestrator.protocol_manifest",
        lambda *args, **kwargs: {
            "protocol_fingerprint_sha256": "3" * 64,
            "route_fingerprint_sha256": "2" * 64,
            "adapter_attestation_sha256": "0" * 64,
            "source_manifest": {},
        },
    )
    monkeypatch.setattr(
        "benchmark.evaluation_campaign.orchestrator.git_state",
        lambda root: {"commit": "test", "dirty": False},
    )
    first = EvaluationCampaignOrchestrator(
        campaign,
        profile,
        LocalBinding(profile.binding_id, profile.adapter, {}),
        repo_root=ROOT,
        python_executable=ROOT / ".venv/bin/python",
        executor=crashing_executor,
        route_session_factory=lambda *args, **kwargs: RouteSession(),
        sleep=lambda _: None,
    )
    with pytest.raises(RuntimeError, match="simulated"):
        first.run()

    pending_values = iter(("S100\nS101\n", ""))

    def resumed_executor(invocation: ProcessInvocation, **kwargs: Any) -> ExecutionResult:
        calls.append(invocation.argv)
        if invocation.argv[1].endswith("check_model_endpoint.py"):
            return ExecutionResult(0, "", "")
        if invocation.argv[1] == FROZEN_SELECTOR and "--pending-only" in invocation.argv:
            return ExecutionResult(0, next(pending_values), "")
        if invocation.argv[1] == FROZEN_RUNNER:
            out = Path(invocation.argv[invocation.argv.index("--output-root") + 1])
            for case_id in ("S100", "S101"):
                case = out / "cases" / case_id
                case.mkdir(parents=True, exist_ok=True)
                _write_json(case / "case_run_manifest.json", {})
                _write_json(case / "evaluation_report.json", {})
            _write_json(out / "run_manifest.json", {"status": "complete"})
            return ExecutionResult(0, "", "", 2)
        if invocation.argv[1] == FROZEN_SELECTOR:
            roots = [
                Path(invocation.argv[index + 1])
                for index, value in enumerate(invocation.argv)
                if value == "--attempt-root"
            ]
            final = campaign.outputs.final_selection_root
            final.mkdir(parents=True)
            _write_json(
                final / "selection_manifest.json",
                {
                    "cases": [
                        {
                            "case_id": case_id,
                            "source_run": str(roots[-1]),
                            "source_case": str(roots[-1] / "cases" / case_id),
                        }
                        for case_id in ("S100", "S101")
                    ]
                },
            )
            return ExecutionResult(0, "", "")
        raise AssertionError(invocation.argv)

    second = EvaluationCampaignOrchestrator(
        campaign,
        profile,
        LocalBinding(profile.binding_id, profile.adapter, {}),
        repo_root=ROOT,
        python_executable=ROOT / ".venv/bin/python",
        executor=resumed_executor,
        route_session_factory=lambda *args, **kwargs: RouteSession(),
        sleep=lambda _: None,
    )
    assert second.run().status == "complete"
    runner = next(argv for argv in calls if argv[1] == FROZEN_RUNNER)
    assert runner[runner.index("--output-root") + 1].endswith("round_01")
    record = json.loads(
        (campaign.outputs.attempt_parent / "round_00/campaign_round.json").read_text()
    )
    assert record["status"] == "abandoned_interrupted"


def test_campaign_stops_at_per_case_attempt_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset(tmp_path / "dataset")
    identity = inspect_evaluation_dataset(dataset, expected_case_ids=("S100", "S101"))
    campaign = _campaign(
        tmp_path,
        dataset,
        identity.portable_fingerprint_sha256,
        max_attempts=1,
    )
    profile = _profile()
    route = _resolved(profile)
    runner_calls = 0

    class RouteSession:
        def __enter__(self) -> ResolvedJudgeRoute:
            return route

        def __exit__(self, *args: Any) -> None:
            return None

    def executor(invocation: ProcessInvocation, **kwargs: Any) -> ExecutionResult:
        nonlocal runner_calls
        if invocation.argv[1].endswith("check_model_endpoint.py"):
            return ExecutionResult(0, "", "")
        if invocation.argv[1] == FROZEN_SELECTOR:
            return ExecutionResult(0, "S100\nS101\n", "")
        if invocation.argv[1] == FROZEN_RUNNER:
            runner_calls += 1
            out = Path(invocation.argv[invocation.argv.index("--output-root") + 1])
            _write_json(out / "run_manifest.json", {"status": "complete"})
            return ExecutionResult(0, "", "", 12)
        raise AssertionError(invocation.argv)

    monkeypatch.setattr(
        "benchmark.evaluation_campaign.orchestrator.protocol_manifest",
        lambda *args, **kwargs: {
            "protocol_fingerprint_sha256": "3" * 64,
            "route_fingerprint_sha256": "2" * 64,
            "adapter_attestation_sha256": "0" * 64,
            "source_manifest": {},
        },
    )
    monkeypatch.setattr(
        "benchmark.evaluation_campaign.orchestrator.git_state",
        lambda root: {"commit": "test", "dirty": False},
    )
    result = EvaluationCampaignOrchestrator(
        campaign,
        profile,
        LocalBinding(profile.binding_id, profile.adapter, {}),
        repo_root=ROOT,
        python_executable=ROOT / ".venv/bin/python",
        executor=executor,
        route_session_factory=lambda *args, **kwargs: RouteSession(),
        sleep=lambda _: None,
    ).run()
    assert result.status == "retries_exhausted"
    assert result.unresolved_case_ids == ("S100", "S101")
    assert runner_calls == 1
    assert not campaign.outputs.final_selection_root.exists()


def test_resume_rejects_dataset_protocol_or_config_mismatch(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "dataset")
    identity = inspect_evaluation_dataset(dataset, expected_case_ids=("S100", "S101"))
    campaign = _campaign(tmp_path, dataset, identity.portable_fingerprint_sha256)
    recorded_dataset = identity.public_dict()
    recorded_dataset["portable_fingerprint_sha256"] = "a" * 64
    state = {
        "schema_version": "scene_evaluation_campaign_state_v1",
        "campaign_id": campaign.campaign_id,
        "campaign_config_sha256": campaign.source_sha256,
        "dataset": recorded_dataset,
        "protocol": {"protocol_fingerprint_sha256": "b" * 64},
        "execution": {"execution_fingerprint_sha256": "e" * 64},
        "git": {"commit": "test", "dirty": False},
        "judge_profile": {
            **_profile().public_dict()
        },
        "status": "planned",
        "pending_case_ids": ["S100", "S101"],
        "rounds": [],
        "created_at": "test",
        "updated_at": "test",
    }
    orchestrator = EvaluationCampaignOrchestrator(
        campaign,
        _profile(),
        None,
        repo_root=ROOT,
        python_executable=ROOT / ".venv/bin/python",
    )
    with pytest.raises(RuntimeError, match="dataset changed"):
        orchestrator._validate_state_guards(
            state,
            identity,
            {"protocol_fingerprint_sha256": "b" * 64},
        )
    state["dataset"] = identity.public_dict()
    with pytest.raises(RuntimeError, match="protocol changed"):
        orchestrator._validate_state_guards(
            state,
            identity,
            {"protocol_fingerprint_sha256": "c" * 64},
        )
    changed_campaign = replace(campaign, source_sha256="d" * 64)
    changed = EvaluationCampaignOrchestrator(
        changed_campaign,
        _profile(),
        None,
        repo_root=ROOT,
        python_executable=ROOT / ".venv/bin/python",
    )
    with pytest.raises(RuntimeError, match="config changed"):
        changed._validate_state_guards(
            state,
            identity,
            {"protocol_fingerprint_sha256": "b" * 64},
        )


def test_campaign_file_lock_refuses_second_controller(tmp_path: Path) -> None:
    lock_path = tmp_path / ".campaign.lock"
    with _campaign_lock(lock_path):
        with pytest.raises(RuntimeError, match="already locked"):
            with _campaign_lock(lock_path):
                raise AssertionError("unreachable")


def test_selection_provenance_preserves_mixed_api1_api2_profile_ids(
    tmp_path: Path,
) -> None:
    dataset_root = _dataset(tmp_path / "dataset")
    identity = inspect_evaluation_dataset(
        dataset_root, expected_case_ids=("S100", "S101")
    )
    campaign = _campaign(
        tmp_path, dataset_root, identity.portable_fingerprint_sha256
    )
    api1 = tmp_path / "api1/round_00"
    api2 = tmp_path / "api2/round_00"
    records = []
    for index, (root, case_id) in enumerate(((api1, "S100"), (api2, "S101"))):
        case = root / "cases" / case_id
        case.mkdir(parents=True)
        _write_json(
            case / "case_run_manifest.json",
            {
                "status": "complete",
                "final_decision_status": "resolved",
                "l1_engineering_failure": False,
            },
        )
        _write_json(
            case / "evaluation_report.json",
            {
                "evaluation_status": "complete",
                "benchmark_score_status": "complete",
                "benchmark_score_100": 50,
            },
        )
        _write_json(case / "l1_report.json", {})
        _write_json(case / "scene_quality_report.json", {})
        records.append(
            {
                "case_id": case_id,
                "selected_attempt_index": index,
                "source_run": str(root.resolve()),
                "source_case": str(case.resolve()),
                "storage": "absolute_directory_symlink",
                "status": "complete",
                "final_decision_status": "resolved",
                "benchmark_score_100": 50,
                    "benchmark_score_status": "complete",
                    "evaluation_status": "complete",
                    "grounded_score_fraction": None,
                    "l1_engineering_failure": False,
                "case_manifest_sha256": _sha((case / "case_run_manifest.json").read_bytes()),
                "evaluation_report_sha256": _sha((case / "evaluation_report.json").read_bytes()),
                "l1_report_sha256": _sha((case / "l1_report.json").read_bytes()),
                "l3_report_sha256": _sha((case / "scene_quality_report.json").read_bytes()),
            }
        )
    final = campaign.outputs.final_selection_root
    final.mkdir(parents=True)
    (final / "cases").mkdir()
    for row in records:
        (final / "cases" / row["case_id"]).symlink_to(
            row["source_case"], target_is_directory=True
        )
    selection = {
        "schema_version": "scene_level_first_publishable_selection_v1",
        "status": "complete",
        "model_label": campaign.model_label,
        "evaluator_model": "gpt-5.6-sol",
        "provider_route": (
            "profiles:gpt56sol-api1-direct-v1,"
            "gpt56sol-api2-standard-xhigh-v1"
        ),
        "case_count": 2,
        "attempt_roots": [str(api1.resolve()), str(api2.resolve())],
        "selection_policy": "first_publishable_attempt_only_no_score_selection",
        "publishability_policy": {
            "case_status": "complete",
            "final_decision_status": "resolved",
            "l1_engineering_failure": False,
            "evaluation_status": "complete",
            "benchmark_score_status": "complete",
            "benchmark_score_100": "finite_number",
        },
        "cases": records,
    }
    _write_json(final / "selection_manifest.json", selection)
    _write_json(final / "run_manifest.json", selection)
    _write_json(
        final / "summary.json",
        {
            "schema_version": "selected_scene_level_summary_v1",
            "status": "complete",
            "model_label": campaign.model_label,
            "evaluator_model": "gpt-5.6-sol",
            "provider_route": (
                "profiles:gpt56sol-api1-direct-v1,"
                "gpt56sol-api2-standard-xhigh-v1"
            ),
            "totals": {
                "cases": 2,
                "successful": 2,
                "failed": 0,
                "final_unresolved": 0,
                "final_infrastructure_failure": 0,
                "l1_engineering_failure_cases": 0,
                "retry_cases": 1,
                "baseline_cases": 1,
                "attempt_rounds": 2,
            },
            "aggregate": _selection_aggregate(records),
        },
    )
    result = write_selection_provenance(
        final,
        campaign=campaign,
        dataset=identity,
        protocol_fingerprint_sha256="f" * 64,
        attempt_route_ids={
            api1.resolve(): "gpt56sol-api1-direct-v1",
            api2.resolve(): "gpt56sol-api2-standard-xhigh-v1",
        },
    )
    assert [row["judge_profile_id"] for row in result["cases"]] == [
        "gpt56sol-api1-direct-v1",
        "gpt56sol-api2-standard-xhigh-v1",
    ]
    assert "source_run" not in result["cases"][0]


def test_prior_attempt_adoption_fails_closed_on_plan_or_protocol_drift(
    tmp_path: Path,
) -> None:
    dataset_root = _dataset(tmp_path / "dataset")
    identity = inspect_evaluation_dataset(
        dataset_root, expected_case_ids=("S100", "S101")
    )
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    _write_json(
        legacy / "run_manifest.json",
        {"status": "complete", "experiment_plan_sha256": "a" * 64},
    )
    _write_json(
        legacy / "experiment_plan.json",
        {
            "cases": [
                {
                    "case_id": case.case_id,
                    "semantic_content_fingerprint": case.semantic_content_fingerprint,
                }
                for case in identity.cases
            ]
        },
    )
    with pytest.raises(ValueError, match="experiment plan drift"):
        validate_prior_attempt(
            PriorAttemptRoot(
                root=legacy,
                judge_profile_id="gpt56sol-api1-direct-v1",
                adoption_mode="legacy_experiment_plan",
                expected_experiment_plan_sha256="b" * 64,
            ),
            dataset=identity,
            protocol_fingerprint_sha256="c" * 64,
        )

    current = tmp_path / "current"
    current.mkdir()
    _write_json(current / "run_manifest.json", {"status": "complete"})
    _write_json(
        current / "campaign_round.json",
        {
            "schema_version": "scene_evaluation_campaign_round_v1",
            "protocol_fingerprint_sha256": "d" * 64,
            "dataset_fingerprint_sha256": identity.portable_fingerprint_sha256,
            "status": "complete",
            "started": True,
            "case_ids": ["S100", "S101"],
            "route": {"route_fingerprint_sha256": "f" * 64},
        },
    )
    with pytest.raises(ValueError, match="protocol mismatch"):
        validate_prior_attempt(
            PriorAttemptRoot(
                root=current,
                judge_profile_id="gpt56sol-api2-standard-xhigh-v1",
                adoption_mode="campaign_protocol",
                expected_protocol_fingerprint_sha256="d" * 64,
            ),
            dataset=identity,
            protocol_fingerprint_sha256="e" * 64,
        )


def test_campaign_package_has_no_generation_or_retrieval_profile_dependency() -> None:
    package = ROOT / "src/benchmark/evaluation_campaign"
    content = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(package.glob("*.py"))
    )
    assert "benchmark.scene_generation" not in content
    assert "from benchmark.api.evaluation" not in content
    assert "import benchmark.api.evaluation" not in content
    assert "from benchmark.evaluator" not in content
    assert "from benchmark.visual_judge" not in content
    assert "from benchmark.rendering" not in content
    forbidden = "retrieval" + "_profile"
    assert forbidden not in content


def test_local_binding_example_is_redacted_and_real_binding_is_ignored() -> None:
    example = ROOT / "configs/evaluation/campaigns/evaluation_bindings.local.example.json"
    raw = example.read_text(encoding="utf-8")
    assert "example.invalid" in raw
    assert "0000000000000000000000000000000000000000000000000000000000000000" in raw
    ignored = subprocess.run(
        [
            "git",
            "check-ignore",
            "-q",
            "configs/evaluation/campaigns/evaluation_bindings.local.json",
        ],
        cwd=ROOT,
        check=False,
    )
    assert ignored.returncode == 0
    _require_private_binding_path(
        ROOT / "configs/evaluation/campaigns/evaluation_bindings.local.json"
    )
    with pytest.raises(SystemExit):
        _require_private_binding_path(example)


def test_checked_in_public_profiles_and_campaigns_have_no_deployment_values() -> None:
    root = ROOT / "configs/evaluation/campaigns"
    profiles = load_profile_registry(root / "judge_profiles_v1.json")
    assert set(profiles) == {
        "gpt56sol-api1-direct-v1",
        "gpt56sol-api2-standard-xhigh-v1",
    }
    for name in ("glm53_api1_full10_v1.json", "glm53_api2_supplement_v1.json"):
        campaign = load_campaign(root / name, repo_root=ROOT)
        raw = campaign.source_path.read_text(encoding="utf-8").lower()
        assert "http://" not in raw and "https://" not in raw
        assert "credential_env" not in raw
        assert '"port"' not in raw


def test_byte_for_byte_relocated_dataset_runs_through_private_projection(
    tmp_path: Path,
) -> None:
    source = _dataset(tmp_path / "source")
    source_identity = inspect_evaluation_dataset(
        source, expected_case_ids=("S100", "S101")
    )
    copied = tmp_path / "copied"
    shutil.copytree(source, copied)
    copied_identity = inspect_evaluation_dataset(
        copied, expected_case_ids=("S100", "S101")
    )
    assert copied_identity.portable_fingerprint_sha256 == source_identity.portable_fingerprint_sha256
    guarded = source / "S100/evidence/collision_geometry_manifest.json"
    guarded_before = _sha(guarded.read_bytes())
    view = prepare_portable_dataset_view(copied, tmp_path / "private-view")
    projected = inspect_evaluation_dataset(view, expected_case_ids=("S100", "S101"))
    assert projected.portable_fingerprint_sha256 == source_identity.portable_fingerprint_sha256
    assert _sha(guarded.read_bytes()) == guarded_before
    collision = json.loads(
        (view / "S100/evidence/collision_geometry_manifest.json").read_text()
    )
    assert not Path(collision["objects"]["object"]["geometry_path"]).is_absolute()


def test_protocol_fingerprint_covers_profile_campaign_package_and_yaml(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    for relative in (
        "scripts/run_camera_cal_scene_level.py",
        "scripts/select_first_publishable_scene_evaluations.py",
        "scripts/check_model_endpoint.py",
        "scripts/build_vlm_evidence_viewer.py",
        "src/benchmark/evaluation_campaign/controller.py",
        "src/benchmark/camera_cal_scene_level/__init__.py",
        "src/benchmark/camera_cal_scene_level/io.py",
        "src/benchmark/camera_cal_scene_level/progress.py",
        "src/benchmark/camera_cal_scene_level/telemetry.py",
        "src/benchmark/api/evaluation.py",
        "src/benchmark/evaluator/module.py",
        "src/benchmark/models/module.py",
        "src/benchmark/rendering/module.py",
        "src/benchmark/visual_judge/module.py",
        "src/benchmark/grouping/module.py",
        "src/benchmark/scoring_profiles.py",
        "configs/grouping/policy.yaml",
        "configs/evaluation/policy.yaml",
        "src/benchmark/_resources/configs/evaluation/policy.yaml",
            "src/benchmark/_resources/configs/grouping/policy.yaml",
            "pyproject.toml",
            "uv.lock",
        ):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture:{relative}\n", encoding="utf-8")
    dataset_root = _dataset(tmp_path / "dataset")
    dataset = inspect_evaluation_dataset(
        dataset_root, expected_case_ids=("S100", "S101")
    )
    profile = _profile()
    campaign = _campaign(tmp_path, dataset_root, dataset.portable_fingerprint_sha256)
    campaign = replace(
        campaign,
        kernel=replace(campaign.kernel, grouping_config=repo / "configs/grouping/policy.yaml"),
    )
    route = _resolved(profile).public_manifest()
    first = protocol_manifest(
        campaign, dataset, repo_root=repo, profile=profile, route_public_manifest=route
    )
    camera_runtime_paths = {
        row["path"]
        for row in first["source_manifest"]["files"]
        if str(row["path"]).startswith(
            "src/benchmark/camera_cal_scene_level/"
        )
    }
    expected_camera_runtime_paths = {
        path.relative_to(repo).as_posix()
        for path in (
            repo / "src/benchmark/camera_cal_scene_level"
        ).rglob("*.py")
    }
    assert camera_runtime_paths == expected_camera_runtime_paths
    leaf = repo / "src/benchmark/camera_cal_scene_level/telemetry.py"
    leaf.write_text("changed-leaf\n", encoding="utf-8")
    leaf_changed = protocol_manifest(
        campaign,
        dataset,
        repo_root=repo,
        profile=profile,
        route_public_manifest=route,
    )
    assert (
        first["protocol_fingerprint_sha256"]
        != leaf_changed["protocol_fingerprint_sha256"]
    )
    source = repo / "src/benchmark/evaluation_campaign/controller.py"
    source.write_text("changed\n", encoding="utf-8")
    second = protocol_manifest(
        campaign, dataset, repo_root=repo, profile=profile, route_public_manifest=route
    )
    assert (
        leaf_changed["protocol_fingerprint_sha256"]
        != second["protocol_fingerprint_sha256"]
    )
    changed_profile = replace(profile, fingerprint_sha256="f" * 64)
    third = protocol_manifest(
        campaign,
        dataset,
        repo_root=repo,
        profile=changed_profile,
        route_public_manifest=route,
    )
    assert second["protocol_fingerprint_sha256"] != third["protocol_fingerprint_sha256"]
    yaml_path = repo / "configs/grouping/policy.yaml"
    yaml_path.write_text("changed-yaml\n", encoding="utf-8")
    fourth = protocol_manifest(
        campaign,
        dataset,
        repo_root=repo,
        profile=changed_profile,
        route_public_manifest=route,
    )
    assert third["protocol_fingerprint_sha256"] != fourth["protocol_fingerprint_sha256"]


def test_python_dependency_closure_resolves_imported_submodule_alias(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    entry = repo / "scripts/entry.py"
    package = repo / "src/benchmark/runtime"
    entry.parent.mkdir(parents=True)
    package.mkdir(parents=True)
    entry.write_text(
        "from benchmark.runtime import leaf as runtime_leaf\n",
        encoding="utf-8",
    )
    (package / "__init__.py").write_text("", encoding="utf-8")
    leaf = package / "leaf.py"
    leaf.write_text("VALUE = 1\n", encoding="utf-8")

    closure = _python_dependency_closure(
        repo,
        {entry.resolve()},
    )

    assert (package / "__init__.py").resolve() in closure
    assert leaf.resolve() in closure


def test_real_source_manifest_covers_all_tracked_camera_runtime_modules() -> None:
    tracked = {
        line
        for line in subprocess.check_output(
            [
                "git",
                "ls-files",
                "--",
                "src/benchmark/camera_cal_scene_level",
            ],
            cwd=ROOT,
            text=True,
        ).splitlines()
        if line.endswith(".py")
    }
    manifest_paths = {
        str(row["path"])
        for row in evaluation_source_manifest(ROOT)["files"]
    }

    assert tracked
    assert tracked.issubset(manifest_paths)


def test_managed_adapter_attestation_rejects_declared_effort_drift(
    tmp_path: Path,
) -> None:
    launcher = tmp_path / "src/benchmark/evaluation_campaign/owned_proxy_launcher.py"
    launcher.parent.mkdir(parents=True)
    launcher.write_bytes(
        (ROOT / "src/benchmark/evaluation_campaign/owned_proxy_launcher.py").read_bytes()
    )
    adapter = tmp_path / "adapter.yaml"
    adapter.write_text(
        "model_list:\n"
        "  - model_name: gpt-5.6-sol\n"
        "    litellm_params:\n"
        "      model: openai/api_azure_openai_gpt-5.6-sol\n"
        "      reasoning_effort: high\n"
        "      additional_drop_params: [output_config, temperature]\n"
        "    model_info: {base_model: gpt-5.6-sol}\n"
        "litellm_settings: {drop_params: true, num_retries: 1, request_timeout: 3000}\n",
        encoding="utf-8",
    )
    profile = _profile(managed=True)
    binding = LocalBinding(
        profile.binding_id,
        profile.adapter,
        {
            "launcher_path": "src/benchmark/evaluation_campaign/owned_proxy_launcher.py",
            "launcher_sha256": _sha(launcher.read_bytes()),
            "adapter_profile_path": "adapter.yaml",
            "adapter_profile_sha256": _sha(adapter.read_bytes()),
            "upstream_environment": {"UPSTREAM": "SOURCE"},
            "local_master_key_env": "LOCAL_MASTER",
            "local_port_env": "LOCAL_PORT",
            "local_port": 43124,
            "startup_timeout_seconds": 1,
        },
    )
    session = open_judge_route(
        profile,
        binding,
        repo_root=tmp_path,
        environ={"SOURCE": "secret"},
        process_factory=lambda *args, **kwargs: pytest.fail("must not launch"),
        port_probe=lambda host, port: False,
    )
    with pytest.raises(CampaignConfigError, match="attestation mismatch"):
        session.__enter__()


def test_managed_proxy_enter_exception_cleans_owned_process_and_lease(
    tmp_path: Path,
) -> None:
    launcher = tmp_path / "src/benchmark/evaluation_campaign/owned_proxy_launcher.py"
    launcher.parent.mkdir(parents=True)
    launcher.write_bytes(
        (ROOT / "src/benchmark/evaluation_campaign/owned_proxy_launcher.py").read_bytes()
    )
    adapter = tmp_path / "adapter.yaml"
    adapter.write_text(
        "model_list:\n"
        "  - model_name: gpt-5.6-sol\n"
        "    litellm_params:\n"
        "      model: openai/api_azure_openai_gpt-5.6-sol\n"
        "      reasoning_effort: xhigh\n"
        "      additional_drop_params: [output_config, temperature]\n"
        "    model_info: {base_model: gpt-5.6-sol}\n"
        "litellm_settings: {drop_params: true, num_retries: 1, request_timeout: 3000}\n",
        encoding="utf-8",
    )
    profile = _profile(managed=True)
    binding = LocalBinding(
        profile.binding_id,
        profile.adapter,
        {
            "launcher_path": "src/benchmark/evaluation_campaign/owned_proxy_launcher.py",
            "launcher_sha256": _sha(launcher.read_bytes()),
            "adapter_profile_path": "adapter.yaml",
            "adapter_profile_sha256": _sha(adapter.read_bytes()),
            "upstream_environment": {"UPSTREAM": "SOURCE"},
            "local_master_key_env": "LOCAL_MASTER",
            "local_port_env": "LOCAL_PORT",
            "local_port": 43126,
            "startup_timeout_seconds": 1,
        },
    )

    class Process:
        pid = 987654321
        terminated = False

        def poll(self) -> int | None:
            return 0 if self.terminated else None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def kill(self) -> None:
            self.terminated = True

    process = Process()
    lease_root = tmp_path / "lease"
    session = open_judge_route(
        profile,
        binding,
        repo_root=tmp_path,
        environ={"SOURCE": "secret"},
        process_factory=lambda *args, **kwargs: process,
        readiness_probe=lambda *args: (_ for _ in ()).throw(RuntimeError("probe")),
        port_probe=lambda host, port: False,
        ownership_root=lease_root,
    )
    with pytest.raises(RuntimeError, match="probe"):
        session.__enter__()
    assert process.terminated is True
    assert not list(lease_root.glob("*.lease.json"))


def test_owned_proxy_launcher_consumes_config_host_and_port(tmp_path: Path) -> None:
    config = tmp_path / "adapter.yaml"
    config.write_text("model_list: []\n", encoding="utf-8")
    launcher = ROOT / "src/benchmark/evaluation_campaign/owned_proxy_launcher.py"
    completed = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(launcher),
            "--config",
            str(config),
            "--host",
            "127.0.0.1",
            "--port",
            "43125",
            "--ownership-token",
            "a" * 48,
            "--verify-contract-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["config_sha256"] == _sha(config.read_bytes())
    assert result["host"] == "127.0.0.1" and result["port"] == 43125


def test_legacy_prior_adoption_is_nonempty_complete_and_immutable(
    tmp_path: Path,
) -> None:
    dataset_root = _dataset(tmp_path / "dataset")
    dataset = inspect_evaluation_dataset(
        dataset_root, expected_case_ids=("S100", "S101")
    )
    prior_root = tmp_path / "prior"
    plan = {
        "cases": [
            {
                "case_id": case.case_id,
                "semantic_content_fingerprint": case.semantic_content_fingerprint,
            }
            for case in dataset.cases
        ]
    }
    _write_json(prior_root / "experiment_plan.json", plan)
    _write_json(
        prior_root / "run_manifest.json",
        {"status": "complete", "experiment_plan_sha256": "a" * 64},
    )
    for case_id in dataset.ordered_case_ids:
        _write_json(prior_root / "cases" / case_id / "case_run_manifest.json", {"status": "complete"})
        _write_json(prior_root / "cases" / case_id / "evaluation_report.json", {"score": 1})
    prior = PriorAttemptRoot(
        root=prior_root,
        judge_profile_id="profile-api1",
        adoption_mode="legacy_experiment_plan",
        expected_experiment_plan_sha256="a" * 64,
    )
    adoption = tmp_path / "private/adoption.json"
    validate_prior_attempt(
        prior,
        dataset=dataset,
        protocol_fingerprint_sha256="b" * 64,
        judge_profile_fingerprint_sha256="c" * 64,
        adoption_manifest_path=adoption,
    )
    _write_json(prior_root / "cases/S100/evaluation_report.json", {"score": 2})
    with pytest.raises(ValueError, match="immutable prior adoption"):
        validate_prior_attempt(
            prior,
            dataset=dataset,
            protocol_fingerprint_sha256="b" * 64,
            judge_profile_fingerprint_sha256="c" * 64,
            adoption_manifest_path=adoption,
        )
    _write_json(prior_root / "experiment_plan.json", {"cases": []})
    with pytest.raises(ValueError, match="no cases"):
        validate_prior_attempt(
            prior,
            dataset=dataset,
            protocol_fingerprint_sha256="b" * 64,
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("duplicate", "case order/inventory"),
        ("hash", "source hash mismatch"),
        ("link", "link target mismatch"),
        ("summary", "summary status/totals"),
    ],
)
def test_existing_final_adoption_rejects_false_green(
    tmp_path: Path, mutation: str, match: str
) -> None:
    dataset_root = _dataset(tmp_path / "dataset")
    identity = inspect_evaluation_dataset(dataset_root, expected_case_ids=("S100", "S101"))
    campaign = _campaign(tmp_path, dataset_root, identity.portable_fingerprint_sha256)
    roots = [(tmp_path / "a", "S100"), (tmp_path / "b", "S101")]
    routes = _strict_final_fixture(campaign.outputs.final_selection_root, campaign, roots)
    final = campaign.outputs.final_selection_root
    if mutation == "duplicate":
        value = json.loads((final / "selection_manifest.json").read_text())
        value["cases"][1]["case_id"] = "S100"
        _write_json(final / "selection_manifest.json", value)
        _write_json(final / "run_manifest.json", value)
    elif mutation == "hash":
        source = roots[0][0] / "cases/S100/evaluation_report.json"
        value = json.loads(source.read_text())
        value["benchmark_score_100"] = 99
        _write_json(source, value)
    elif mutation == "link":
        link = final / "cases/S100"
        link.unlink()
        link.symlink_to(roots[1][0] / "cases/S101", target_is_directory=True)
    else:
        value = json.loads((final / "summary.json").read_text())
        value["totals"]["successful"] = 1
        _write_json(final / "summary.json", value)
    with pytest.raises(ValueError, match=match):
        validate_final_selection(final, campaign=campaign, attempt_route_ids=routes)


def test_round_100_resume_and_attempt_accounting_fail_closed(tmp_path: Path) -> None:
    rows = [
        {"round_index": index, "started": True, "case_ids": ["S100"]}
        for index in range(100)
    ]
    rows.append({"round_index": 100, "started": False, "case_ids": ["S100"]})
    assert _next_round_index(rows) == 101
    assert _attempt_counts(rows, ("S100",))["S100"] == 100
    for name in ("round_99", "round_100"):
        (tmp_path / name).mkdir()
    assert [path.name for path in _round_directories(tmp_path)] == ["round_99", "round_100"]


def test_resume_rejects_state_row_whose_round_directory_is_missing(
    tmp_path: Path,
) -> None:
    dataset_root = _dataset(tmp_path / "dataset")
    identity = inspect_evaluation_dataset(
        dataset_root, expected_case_ids=("S100", "S101")
    )
    campaign = _campaign(
        tmp_path, dataset_root, identity.portable_fingerprint_sha256
    )
    orchestrator = EvaluationCampaignOrchestrator(
        campaign,
        _profile(),
        None,
        repo_root=ROOT,
        python_executable=ROOT / ".venv/bin/python",
    )
    protocol = {
        "protocol_fingerprint_sha256": "a" * 64,
        "route_fingerprint_sha256": "b" * 64,
    }
    execution = {"execution_fingerprint_sha256": "c" * 64}
    state = orchestrator._reconstruct_state(identity, protocol, execution)
    state["rounds"] = [
        {
            "round_index": 0,
            "round_name": "round_00",
            "case_ids": ["S100"],
            "status": "planned",
            "exit_code": None,
            "started": False,
            "judge_profile_id": _profile().profile_id,
            "invocation": None,
        }
    ]
    _write_json(campaign.outputs.attempt_parent / "campaign_manifest.json", state)
    with pytest.raises(RuntimeError, match="missing round directory"):
        orchestrator._reconstruct_state(identity, protocol, execution)


def test_unstarted_round_is_excluded_and_terminal_artifacts_are_rejected(
    tmp_path: Path,
) -> None:
    dataset_root = _dataset(tmp_path / "dataset")
    identity = inspect_evaluation_dataset(
        dataset_root, expected_case_ids=("S100", "S101")
    )
    campaign = _campaign(
        tmp_path, dataset_root, identity.portable_fingerprint_sha256
    )
    orchestrator = EvaluationCampaignOrchestrator(
        campaign,
        _profile(),
        None,
        repo_root=ROOT,
        python_executable=ROOT / ".venv/bin/python",
    )
    excluded = {
        "round_index": 0,
        "round_name": "round_00",
        "case_ids": ["S100"],
        "status": "abandoned_not_started",
        "exit_code": None,
        "started": False,
        "judge_profile_id": _profile().profile_id,
        "invocation": None,
    }
    included = {
        **excluded,
        "round_index": 1,
        "round_name": "round_01",
        "status": "abandoned_interrupted",
        "started": True,
    }
    roots, _ = orchestrator._chronological_roots(
        {"rounds": [excluded, included]}
    )
    assert roots == ((campaign.outputs.attempt_parent / "round_01").resolve(),)

    protocol = {
        "protocol_fingerprint_sha256": "a" * 64,
        "route_fingerprint_sha256": "b" * 64,
    }
    execution = {"execution_fingerprint_sha256": "c" * 64}
    round_root = campaign.outputs.attempt_parent / "round_00"
    _write_json(
        round_root / "campaign_round.json",
        {
            "schema_version": "scene_evaluation_campaign_round_v1",
            "campaign_id": campaign.campaign_id,
            "campaign_config_sha256": campaign.source_sha256,
            "dataset_fingerprint_sha256": identity.portable_fingerprint_sha256,
            "protocol_fingerprint_sha256": protocol[
                "protocol_fingerprint_sha256"
            ],
            "route": {
                "route_fingerprint_sha256": protocol[
                    "route_fingerprint_sha256"
                ]
            },
            "round_index": 0,
            "case_ids": ["S100"],
            "status": "planned",
            "exit_code": None,
            "started": False,
            "updated_at": "test",
        },
    )
    _write_json(round_root / "run_manifest.json", {"status": "complete"})
    with pytest.raises(RuntimeError, match="unstarted.*terminal artifacts"):
        orchestrator._reconstruct_state(identity, protocol, execution)


def test_check_rejects_selector_incompatible_judge_alias(tmp_path: Path) -> None:
    dataset_root = _dataset(tmp_path / "dataset")
    identity = inspect_evaluation_dataset(
        dataset_root, expected_case_ids=("S100", "S101")
    )
    campaign = _campaign(
        tmp_path, dataset_root, identity.portable_fingerprint_sha256
    )
    registry = json.loads(campaign.profile_registry.read_text())
    registry["profiles"][0]["model_alias"] = "different-judge"
    _write_json(campaign.profile_registry, registry)
    changed_profile = load_profile_registry(campaign.profile_registry)[
        campaign.judge_profile_id
    ]
    orchestrator = EvaluationCampaignOrchestrator(
        campaign,
        changed_profile,
        None,
        repo_root=ROOT,
        python_executable=ROOT / ".venv/bin/python",
    )
    with pytest.raises(CampaignConfigError, match="requires one gpt-5.6-sol"):
        orchestrator.check()


def test_public_invocation_projection_contains_no_local_paths_or_secrets(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path / "dataset")
    identity = inspect_evaluation_dataset(dataset, expected_case_ids=("S100", "S101"))
    campaign = _campaign(tmp_path, dataset, identity.portable_fingerprint_sha256)
    invocation = build_round_invocation(
        campaign,
        _resolved(),
        repo_root=ROOT,
        python_executable=ROOT / ".venv/bin/python",
        case_ids=("S100",),
        round_index=0,
        round_root=tmp_path / "attempts/round_00",
    )
    public = json.dumps(invocation.public_dict())
    assert str(tmp_path) not in public
    assert "hidden.example.invalid" not in public
    assert "HIDDEN_KEY_ENV" not in public


@pytest.mark.parametrize("mode", ("overlap", "l3_only"))
def test_campaign_config_rejects_unsafe_output_topology_and_l3_only(
    tmp_path: Path, mode: str
) -> None:
    value = json.loads(
        (ROOT / "configs/evaluation/campaigns/glm53_api1_full10_v1.json").read_text()
    )
    value["profile_registry"] = "profiles.json"
    value["dataset"]["root"] = "dataset"
    value["kernel"]["grouping_config"] = "grouping.yaml"
    value["case_plan"]["prior_attempt_roots"] = []
    value["outputs"] = {
        "attempt_parent": "outputs/attempts",
        "final_selection_root": (
            "outputs/attempts/final" if mode == "overlap" else "outputs/final"
        ),
    }
    if mode == "l3_only":
        value["kernel"]["l3_only"] = True
    config = tmp_path / "campaign.json"
    _write_json(config, value)
    with pytest.raises(CampaignConfigError, match="disjoint|l3_only"):
        load_campaign(config, repo_root=tmp_path)


@pytest.mark.parametrize("mode", ("dataset_prior", "prior_prior"))
def test_campaign_config_rejects_overlapping_prior_roots(
    tmp_path: Path, mode: str
) -> None:
    value = json.loads(
        (
            ROOT
            / "configs/evaluation/campaigns/glm53_api2_supplement_v1.json"
        ).read_text()
    )
    value["profile_registry"] = "profiles.json"
    value["kernel"]["grouping_config"] = "grouping.yaml"
    value["dataset"]["root"] = "shared" if mode == "dataset_prior" else "dataset"
    first = dict(value["case_plan"]["prior_attempt_roots"][0])
    first["root"] = "shared/prior" if mode == "dataset_prior" else "priors"
    value["case_plan"]["prior_attempt_roots"] = [first]
    if mode == "prior_prior":
        second = dict(first)
        second["root"] = "priors/nested"
        value["case_plan"]["prior_attempt_roots"].append(second)
    value["outputs"] = {
        "attempt_parent": "outputs/attempts",
        "final_selection_root": "outputs/final",
    }
    config = tmp_path / "campaign.json"
    _write_json(config, value)
    with pytest.raises(CampaignConfigError, match="disjoint"):
        load_campaign(config, repo_root=tmp_path)
