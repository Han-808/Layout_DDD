from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable

import pytest

pytestmark = pytest.mark.requires_git_history

from benchmark.scene_generation.legacy_cutover.forwarding import (
    BindingParityReport,
    LegacyCutoverBlocked,
    blocked_surfaces,
    contract_by_id,
    contracts,
    forward_if_attested,
    translate_legacy_argv,
    verify_binding_parity,
    verify_legacy_source_identity,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REPLAY_ROOT = (
    REPO_ROOT
    / "Support"
    / "legacy"
    / "frozen_replay"
    / "generation_pre_campaign_v1"
)


def _binding_file(tmp_path: Path, contract_id: str, *, match: bool = True) -> Path:
    contract = contract_by_id(contract_id)
    legacy = json.loads(
        (REPO_ROOT / contract.legacy_models_path).read_text(encoding="utf-8")
    )
    endpoint = legacy["endpoint"] if match else "https://different.invalid/v1"
    value = {
        "schema_version": "generation_route_bindings_v2",
        "bindings": {
            contract.route_profile_id: {
                "endpoint": endpoint,
                "credential_env": legacy["api_key_env"],
            }
        },
    }
    path = tmp_path / "generation_bindings.local.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("contract_id", "legacy_argv", "expected"),
    (
        (
            "api2-kimi-k3-full10-v1",
            ("check",),
            ("check", "--campaign", "api2-kimi-k3-scene10-v2"),
        ),
        (
            "api2-kimi-k3-full10-v1",
            ("preflight",),
            ("preflight", "--campaign", "api2-kimi-k3-scene10-v2"),
        ),
        (
            "api2-glm53-full10-v1",
            ("run", "--output-dir", "relative/output"),
            (
                "run",
                "--campaign",
                "api2-glm53-scene10-v2",
                "--output-dir",
                "relative/output",
            ),
        ),
        (
            "api3-opus48-high-full10-v1",
            ("run", "--output-dir", "/tmp/new-output"),
            (
                "run",
                "--campaign",
                "api3-opus48-high-scene10-v2",
                "--output-dir",
                "/tmp/new-output",
            ),
        ),
    ),
)
def test_common_full10_argv_translation_is_exact(
    contract_id: str,
    legacy_argv: tuple[str, ...],
    expected: tuple[str, ...],
) -> None:
    plan = translate_legacy_argv(contract_by_id(contract_id), legacy_argv)
    assert plan.campaign_argv == expected
    assert plan.public_dict()["cutover_ready"] is False


def test_opus_standalone_preflight_is_not_misrepresented_as_full10_preflight() -> None:
    contract = contract_by_id("api3-opus48-high-full10-v1")
    with pytest.raises(SystemExit) as raised:
        translate_legacy_argv(contract, ("preflight",))
    assert raised.value.code == 2


def test_non_equivalent_retry_surfaces_are_explicitly_blocked() -> None:
    values = blocked_surfaces()
    assert len(values) == 4
    assert {Path(value["entrypoint"]).name for value in values} == {
        "recovery_runner.py",
        "fresh_case_retry_runner.py",
        "preflight.py",
        "retry_runner.py",
    }


@pytest.mark.parametrize(
    "contract_id",
    (
        "api2-kimi-k3-full10-v1",
        "api2-glm53-full10-v1",
        "api3-opus48-high-full10-v1",
    ),
)
def test_private_binding_parity_does_not_read_a_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_id: str,
) -> None:
    monkeypatch.delenv("API2_APP_CREDENTIAL", raising=False)
    monkeypatch.delenv("FORGEAX_API_KEY", raising=False)
    path = _binding_file(tmp_path, contract_id)
    report = verify_binding_parity(
        contract_by_id(contract_id),
        repo_root=REPO_ROOT,
        generation_bindings_path=path,
    )
    assert report.matches is True
    assert report.public_dict()["matches"] is True


def test_binding_mismatch_fails_closed_without_disclosing_values(tmp_path: Path) -> None:
    contract = contract_by_id("api2-kimi-k3-full10-v1")
    report = verify_binding_parity(
        contract,
        repo_root=REPO_ROOT,
        generation_bindings_path=_binding_file(
            tmp_path, contract.contract_id, match=False
        ),
    )
    assert report.matches is False
    assert "endpoint" not in report.public_dict()
    assert "credential_env" not in report.public_dict()


def test_forwarding_refuses_before_campaign_execution() -> None:
    plan = translate_legacy_argv(
        contract_by_id("api2-kimi-k3-full10-v1"), ("check",)
    )
    called = False

    def campaign_main(argv: Iterable[str] | None) -> int:
        nonlocal called
        called = True
        return 0

    with pytest.raises(LegacyCutoverBlocked, match="artifact parity"):
        forward_if_attested(
            plan, binding_parity=None, campaign_main=campaign_main
        )
    assert called is False


def test_attested_helper_forwards_only_the_precomputed_argv() -> None:
    original = contract_by_id("api2-glm53-full10-v1")
    contract = replace(
        original, terminal_output_parity=True, artifact_parity=True
    )
    plan = translate_legacy_argv(contract, ("check",))
    seen: list[tuple[str, ...]] = []

    def campaign_main(argv: Iterable[str] | None) -> int:
        assert argv is not None
        seen.append(tuple(argv))
        return 7

    assert (
        forward_if_attested(
            plan, binding_parity=None, campaign_main=campaign_main
        )
        == 7
    )
    assert seen == [
        ("check", "--campaign", "api2-glm53-scene10-v2")
    ]


def test_attested_network_forward_pins_the_verified_private_binding(
    tmp_path: Path,
) -> None:
    original = contract_by_id("api2-kimi-k3-full10-v1")
    contract = replace(
        original, terminal_output_parity=True, artifact_parity=True
    )
    binding_path = _binding_file(tmp_path, original.contract_id)
    parity = verify_binding_parity(
        original,
        repo_root=REPO_ROOT,
        generation_bindings_path=binding_path,
    )
    plan = translate_legacy_argv(
        contract, ("run", "--output-dir", "new-output")
    )
    seen: list[tuple[str, ...]] = []

    def campaign_main(argv: Iterable[str] | None) -> int:
        assert argv is not None
        seen.append(tuple(argv))
        return 0

    assert (
        forward_if_attested(
            plan, binding_parity=parity, campaign_main=campaign_main
        )
        == 0
    )
    assert seen == [
        (
            "run",
            "--campaign",
            "api2-kimi-k3-scene10-v2",
            "--output-dir",
            "new-output",
            "--generation-bindings",
            str(binding_path.resolve()),
        )
    ]


def test_replay_snapshot_is_complete_and_hash_pinned() -> None:
    manifest = json.loads(
        (REPLAY_ROOT / "replay_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["snapshot_id"] == "generation-pre-campaign-v1"
    source_commit = manifest["source_commit"]
    assert (
        subprocess.run(
            ["git", "rev-parse", source_commit],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        == source_commit
    )
    assert len(manifest["files"]) == 7
    for value in manifest["files"]:
        snapshot = REPLAY_ROOT / value["snapshot_path"]
        payload = snapshot.read_bytes()
        assert len(payload) == value["bytes"]
        assert hashlib.sha256(payload).hexdigest() == value["sha256"]
        assert (
            subprocess.run(
                [
                    "git",
                    "rev-parse",
                    f"{source_commit}:{value['original_path']}",
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            == value["git_blob"]
        )
    for value in manifest["external_dependencies"]:
        payload = (REPO_ROOT / value["path"]).read_bytes()
        assert len(payload) == value["bytes"]
        assert hashlib.sha256(payload).hexdigest() == value["sha256"]
        assert (
            subprocess.run(
                ["git", "rev-parse", f"{source_commit}:{value['path']}"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            == value["git_blob"]
        )


def test_replay_manifest_contains_no_private_binding_content() -> None:
    manifest_text = (REPLAY_ROOT / "replay_manifest.json").read_text(
        encoding="utf-8"
    )
    assert "https://" not in manifest_text
    assert '"endpoint"' not in manifest_text
    assert '"credential_env"' not in manifest_text
    assert "/Users/" not in manifest_text


def test_contract_registry_is_small_and_model_explicit() -> None:
    assert {value.contract_id for value in contracts()} == {
        "api2-kimi-k3-full10-v1",
        "api2-glm53-full10-v1",
        "api3-opus48-high-full10-v1",
    }


@pytest.mark.parametrize("contract", contracts())
def test_translation_contract_is_pinned_to_reviewed_legacy_source(
    contract: object,
) -> None:
    report = verify_legacy_source_identity(contract, repo_root=REPO_ROOT)
    assert report.matches is True
    assert report.public_dict()["matches"] is True
