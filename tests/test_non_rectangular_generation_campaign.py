from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.scene_generation.non_rectangular_multi_room.campaign import (
    PreparedNonRectangularCampaign,
    _revalidate_inputs,
    prepare_non_rectangular_campaign,
    run_prepared_non_rectangular_campaign,
)
import benchmark.scene_generation.non_rectangular_multi_room.campaign as campaign_module
from benchmark.scene_generation.non_rectangular_multi_room.cli import (
    main as campaign_main,
)
from benchmark.scene_generation.non_rectangular_multi_room.profiles import (
    load_non_rectangular_campaign_registry,
    load_non_rectangular_campaign_registry_v2,
)
from benchmark.scene_generation.non_rectangular_multi_room.provenance import (
    compatibility_source_manifest,
)
from benchmark.scene_generation.campaign.execution import repository_root
from benchmark.scene_generation.campaign.runtime import RuntimeProviderModel


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/non_rectangular"
CAMPAIGN_ID = "api2-kimi-k3-nonrect-global-v1"
CAMPAIGN_V2_ID = "api2-kimi-k3-nonrect-global-v2"
FULLRUN_CAMPAIGN_V2_ID = "api2-kimi-k3-nonrect-global-retry5-v2"


def _copy_inputs(tmp_path: Path) -> tuple[Path, Path]:
    layout = tmp_path / "room_layout.json"
    program = tmp_path / "room_program.json"
    layout.write_bytes((FIXTURES / "simple_multi_room.json").read_bytes())
    program.write_bytes(
        (FIXTURES / "simple_multi_room_program.json").read_bytes()
    )
    return layout, program


def test_independent_registry_exposes_reviewed_model_campaigns() -> None:
    registry = load_non_rectangular_campaign_registry(repository_root())

    assert len(registry) == 8
    assert registry[CAMPAIGN_ID].model_profile_id == "api2-kimi-k3"

    registry_v2 = load_non_rectangular_campaign_registry_v2(repository_root())
    assert len(registry_v2) == 14
    assert registry_v2[CAMPAIGN_V2_ID].model_profile_id == "api2-kimi-k3"
    assert registry_v2[
        "api3-sonnet5-nonrect-global-retry5-v2"
    ].model_profile_id == (
        "api3-claude-sonnet-5-nonrect-global-retry5-v1"
    )


def test_source_manifest_covers_standalone_campaign_prompts_and_registry() -> None:
    paths = {
        item["path"] for item in compatibility_source_manifest()["files"]
    }

    assert "campaign.py" in paths
    assert "prompts/stage_a_prompt_v1.txt" in paths
    assert "prompts/stage_c_prompt_v1.txt" in paths
    assert "prompts/stage_a_prompt_v3.txt" in paths
    assert "prompts/stage_c_prompt_v3.txt" in paths
    assert (
        "configs/generation_extensions/non_rectangular_multi_room_v1/registry_v1.json"
        in paths
    )
    assert (
        "configs/generation_extensions/non_rectangular_multi_room_v1/registry_v2.json"
        in paths
    )


def test_prepare_dispatches_only_with_both_new_artifacts(tmp_path: Path) -> None:
    layout, program = _copy_inputs(tmp_path)

    prepared = prepare_non_rectangular_campaign(
        CAMPAIGN_ID,
        room_layout_path=layout,
        room_program_path=program,
    )

    assert isinstance(prepared, PreparedNonRectangularCampaign)
    public = prepared.public_dict()
    assert public["generation_mode"] == "non_rectangular_multi_room_global_v1"
    assert public["layout_id"] == "fixture_simple_multi_room"
    assert public["credential_loaded"] is False
    assert public["network_used"] is False


def test_prepare_v2_dispatch_is_explicit_and_keeps_v1_default(tmp_path: Path) -> None:
    layout, program = _copy_inputs(tmp_path)

    prepared = prepare_non_rectangular_campaign(
        CAMPAIGN_V2_ID,
        room_layout_path=layout,
        room_program_path=program,
        contract_version="v2",
    )

    public = prepared.public_dict()
    assert public["contract_version"] == "v2"
    assert public["workflow_profile_id"] == (
        "non-rectangular-global-two-stage-v2"
    )
    assert public["generation_mode"] == "non_rectangular_multi_room_global_v2"


def test_nonrect_api2_stage_timeouts_do_not_modify_shared_gateway(
    tmp_path: Path,
) -> None:
    layout, program = _copy_inputs(tmp_path)
    prepared = prepare_non_rectangular_campaign(
        FULLRUN_CAMPAIGN_V2_ID,
        room_layout_path=layout,
        room_program_path=program,
        contract_version="v2",
    )
    base_model = RuntimeProviderModel.from_profile(
        prepared.model,
        endpoint="https://fixture.invalid/v1",
        api_key="fixture-app:fixture-key",
    )
    model = campaign_module._StageTimeoutProviderModel(
        base=base_model,
        stage_c_timeout_seconds=3600.0,
    )
    route = campaign_module._stage_timeout_provider_route(
        prepared.provider_route
    )

    stage_a = route.request_headers(model, "stage-a")["Authorization"]
    stage_c = route.request_headers(
        model.for_stage("stage_c"), "stage-c"
    )["Authorization"]

    assert "timeout=2400" in stage_a
    assert "timeout=3600" in stage_c
    assert prepared.provider_route.gateway.timeout_seconds == 600


def test_prepare_rejects_unknown_campaign(tmp_path: Path) -> None:
    layout, program = _copy_inputs(tmp_path)

    with pytest.raises(ValueError, match="unknown or incomplete"):
        prepare_non_rectangular_campaign(
            "unknown-nonrect-campaign",
            room_layout_path=layout,
            room_program_path=program,
        )


def test_prepared_input_toctou_is_rejected(tmp_path: Path) -> None:
    layout, program = _copy_inputs(tmp_path)
    prepared = prepare_non_rectangular_campaign(
        CAMPAIGN_ID,
        room_layout_path=layout,
        room_program_path=program,
    )
    value = json.loads(program.read_text(encoding="utf-8"))
    value["target_total_instances"]["max"] += 1
    program.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="room-program identity changed"):
        _revalidate_inputs(prepared)


def test_cli_check_accepts_new_artifact_pair(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout, program = _copy_inputs(tmp_path)

    code = campaign_main(
        [
            "check",
            "--campaign",
            CAMPAIGN_ID,
            "--room-layout",
            str(layout),
            "--room-program",
            str(program),
        ]
    )

    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] is True
    assert result["generation_mode"] == "non_rectangular_multi_room_global_v1"


def test_cli_check_accepts_explicit_v2_contract(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout, program = _copy_inputs(tmp_path)

    code = campaign_main(
        [
            "check",
            "--campaign",
            CAMPAIGN_V2_ID,
            "--contract-version",
            "v2",
            "--room-layout",
            str(layout),
            "--room-program",
            str(program),
        ]
    )

    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["contract_version"] == "v2"
    assert result["generation_mode"] == "non_rectangular_multi_room_global_v2"


def test_prepared_campaign_wires_standalone_runtime_without_old_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, program = _copy_inputs(tmp_path)
    prepared = prepare_non_rectangular_campaign(
        CAMPAIGN_ID,
        room_layout_path=layout,
        room_program_path=program,
    )
    fake_retriever = object()
    fake_binding = SimpleNamespace(
        endpoint="https://fixture.invalid/v1",
        credential=lambda environ: "fixture-key",
    )
    monkeypatch.setattr(
        campaign_module,
        "preflight_non_rectangular_campaign",
        lambda *args, **kwargs: ({"ok": True}, fake_retriever),
    )
    monkeypatch.setattr(
        campaign_module,
        "resolve_bindings",
        lambda *args, **kwargs: (fake_binding, object(), {}),
    )
    fake_core = object()
    monkeypatch.setattr(campaign_module, "load_frozen_core", lambda root: fake_core)
    fake_model = SimpleNamespace(key="api2-kimi-k3")
    monkeypatch.setattr(
        campaign_module,
        "RuntimeProviderModel",
        SimpleNamespace(from_profile=lambda *args, **kwargs: fake_model),
    )
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"status": "complete", "layout_id": "fixture_simple_multi_room"}

    monkeypatch.setattr(
        campaign_module,
        "run_non_rectangular_generation",
        fake_run,
    )

    summary, stopped, preflight = run_prepared_non_rectangular_campaign(
        prepared,
        output_root=tmp_path / "output",
    )

    assert summary["status"] == "complete"
    assert stopped is False
    assert preflight == {"ok": True}
    assert captured["core"] is fake_core
    assert captured["retriever"] is fake_retriever
    assert captured["room_layout"]["layout_id"] == "fixture_simple_multi_room"
    assert captured["room_program"]["layout_id"] == "fixture_simple_multi_room"


def test_prepared_v2_campaign_wires_only_v2_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, program = _copy_inputs(tmp_path)
    prepared = prepare_non_rectangular_campaign(
        CAMPAIGN_V2_ID,
        room_layout_path=layout,
        room_program_path=program,
        contract_version="v2",
    )
    fake_retriever = object()
    fake_binding = SimpleNamespace(
        endpoint="https://fixture.invalid/v1",
        credential=lambda environ: "fixture-key",
    )
    monkeypatch.setattr(
        campaign_module,
        "preflight_non_rectangular_campaign",
        lambda *args, **kwargs: ({"ok": True}, fake_retriever),
    )
    monkeypatch.setattr(
        campaign_module,
        "resolve_bindings",
        lambda *args, **kwargs: (fake_binding, object(), {}),
    )
    monkeypatch.setattr(
        campaign_module,
        "load_frozen_core",
        lambda root: object(),
    )
    monkeypatch.setattr(
        campaign_module,
        "RuntimeProviderModel",
        SimpleNamespace(
            from_profile=lambda *args, **kwargs: SimpleNamespace(
                key="api2-kimi-k3"
            )
        ),
    )
    captured = {}

    monkeypatch.setattr(
        campaign_module,
        "run_non_rectangular_generation",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("v1 runtime must not be called")
        ),
    )

    def fake_v2_run(**kwargs):
        captured.update(kwargs)
        return {"status": "complete", "layout_id": "fixture_simple_multi_room"}

    monkeypatch.setattr(
        campaign_module,
        "run_non_rectangular_generation_v2",
        fake_v2_run,
    )

    summary, stopped, preflight = run_prepared_non_rectangular_campaign(
        prepared,
        output_root=tmp_path / "output_v2",
    )

    assert summary["status"] == "complete"
    assert stopped is False
    assert preflight == {"ok": True}
    assert captured["workflow_profile_id"] == (
        "non-rectangular-global-two-stage-v2"
    )
    assert captured["room_layout"]["layout_id"] == "fixture_simple_multi_room"
