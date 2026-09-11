"""Fail-closed planning for legacy generation-command cutovers.

The current campaign runtime is the intended owner of new generation runs.
This module characterizes how a small set of historical full10 command lines
would translate to that runtime.  It deliberately does not execute a forward
until terminal-output and artifact parity have also been attested.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Callable, Iterable

from benchmark.scene_generation.campaign.bindings import LocalRouteBindings


class LegacyCutoverBlocked(RuntimeError):
    """Raised when a legacy command has not satisfied every cutover gate."""


@dataclass(frozen=True, slots=True)
class LegacyForwardContract:
    contract_id: str
    legacy_entrypoint: str
    campaign_id: str
    route_profile_id: str
    commands: tuple[str, ...]
    legacy_models_path: str
    legacy_entrypoint_sha256: str
    legacy_models_sha256: str
    replay_snapshot_id: str
    terminal_output_parity: bool = False
    artifact_parity: bool = False

    @property
    def cutover_ready(self) -> bool:
        return self.terminal_output_parity and self.artifact_parity


@dataclass(frozen=True, slots=True)
class LegacyForwardPlan:
    contract: LegacyForwardContract
    legacy_command: str
    campaign_argv: tuple[str, ...]

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "legacy_generation_forward_plan_v1",
            "contract_id": self.contract.contract_id,
            "legacy_entrypoint": self.contract.legacy_entrypoint,
            "legacy_command": self.legacy_command,
            "campaign_id": self.contract.campaign_id,
            "argv_parity": True,
            "terminal_output_parity": self.contract.terminal_output_parity,
            "artifact_parity": self.contract.artifact_parity,
            "cutover_ready": self.contract.cutover_ready,
            "replay_snapshot_id": self.contract.replay_snapshot_id,
        }


@dataclass(frozen=True, slots=True)
class BindingParityReport:
    contract_id: str
    route_profile_id: str
    binding_sha256: str
    endpoint_matches: bool
    credential_environment_matches: bool
    selected_binding_path: Path = field(repr=False, compare=False)

    @property
    def matches(self) -> bool:
        return self.endpoint_matches and self.credential_environment_matches

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "legacy_generation_binding_parity_v1",
            "contract_id": self.contract_id,
            "route_profile_id": self.route_profile_id,
            "binding_sha256": self.binding_sha256,
            "endpoint_matches": self.endpoint_matches,
            "credential_environment_matches": self.credential_environment_matches,
            "matches": self.matches,
        }


@dataclass(frozen=True, slots=True)
class LegacySourceIdentityReport:
    contract_id: str
    entrypoint_matches: bool
    model_transport_matches: bool

    @property
    def matches(self) -> bool:
        return self.entrypoint_matches and self.model_transport_matches

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "legacy_generation_source_identity_v1",
            "contract_id": self.contract_id,
            "entrypoint_matches": self.entrypoint_matches,
            "model_transport_matches": self.model_transport_matches,
            "matches": self.matches,
        }


_SNAPSHOT_ID = "generation-pre-campaign-v1"
_CONTRACTS = {
    "api2-kimi-k3-full10-v1": LegacyForwardContract(
        contract_id="api2-kimi-k3-full10-v1",
        legacy_entrypoint=(
            "tools/api2_kimi_k3_runner_v1/kimi_k3_generation_runner.py"
        ),
        campaign_id="api2-kimi-k3-scene10-v2",
        route_profile_id="api2-chat-top-level-reasoning-v1",
        commands=("check", "preflight", "run"),
        legacy_models_path="tools/api2_kimi_k3_runner_v1/models.pod.json",
        legacy_entrypoint_sha256=(
            "25ae25a6ddae36b4e66207d6accf4139a043123b46cb02f2009171264b08a006"
        ),
        legacy_models_sha256=(
            "52d2581601a2c3dd54f998069ad2a3b11189fdf0384e91fe92c3d40fef27f9ae"
        ),
        replay_snapshot_id=_SNAPSHOT_ID,
    ),
    "api2-glm53-full10-v1": LegacyForwardContract(
        contract_id="api2-glm53-full10-v1",
        legacy_entrypoint=(
            "tools/api2_glm53_runner_v1/glm53_generation_runner.py"
        ),
        campaign_id="api2-glm53-scene10-v2",
        route_profile_id="api2-responses-reasoning-v1",
        commands=("check", "preflight", "run"),
        legacy_models_path="tools/api2_glm53_runner_v1/models.pod.json",
        legacy_entrypoint_sha256=(
            "56aee6b201335f4f87952924f27a883f4b30732f61b01adc6ac774b018635bee"
        ),
        legacy_models_sha256=(
            "55921c496a8551bbf7d4a485845545a57ae065777a5c34b1db32343c97564b3a"
        ),
        replay_snapshot_id=_SNAPSHOT_ID,
    ),
    "api3-opus48-high-full10-v1": LegacyForwardContract(
        contract_id="api3-opus48-high-full10-v1",
        legacy_entrypoint=(
            "tools/api3_opus48_thinking_runner_v1/generation_runner.py"
        ),
        campaign_id="api3-opus48-high-scene10-v2",
        route_profile_id="api3-chat-adaptive-thinking-v1",
        commands=("check", "run"),
        legacy_models_path=(
            "tools/api3_opus48_thinking_runner_v1/models.pod.json"
        ),
        legacy_entrypoint_sha256=(
            "427a1e35ddc79b3ddb4742f7d085df8e75e64bb908ee062d7a83d395cd4e39c2"
        ),
        legacy_models_sha256=(
            "e75c29e024e0943eaa574aa3e46b71d4776ea1c2cb2bbf9389dd04fc8a5a0af6"
        ),
        replay_snapshot_id=_SNAPSHOT_ID,
    ),
}


def contracts() -> tuple[LegacyForwardContract, ...]:
    return tuple(_CONTRACTS[key] for key in sorted(_CONTRACTS))


def contract_by_id(contract_id: str) -> LegacyForwardContract:
    try:
        return _CONTRACTS[contract_id]
    except KeyError as exc:
        raise ValueError("unknown legacy generation cutover contract") from exc


def _parser(contract: LegacyForwardContract) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=contract.legacy_entrypoint)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in contract.commands:
        child = subparsers.add_parser(command)
        if command == "run":
            child.add_argument("--output-dir", type=Path, required=True)
    return parser


def translate_legacy_argv(
    contract: LegacyForwardContract,
    argv: Iterable[str],
) -> LegacyForwardPlan:
    """Translate only the argv surface already shared by legacy and v2.

    No endpoint, credential, prompt, request, response, or artifact is read.
    Unsupported retry/preflight surfaces are intentionally absent from the
    contract registry and therefore cannot be translated accidentally.
    """

    args = _parser(contract).parse_args(list(argv))
    translated = [args.command, "--campaign", contract.campaign_id]
    if args.command == "run":
        translated.extend(("--output-dir", str(args.output_dir)))
    return LegacyForwardPlan(
        contract=contract,
        legacy_command=args.command,
        campaign_argv=tuple(translated),
    )


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key in legacy model transport config")
        result[key] = value
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_legacy_source_identity(
    contract: LegacyForwardContract,
    *,
    repo_root: str | Path,
) -> LegacySourceIdentityReport:
    root = Path(repo_root).expanduser().resolve()
    return LegacySourceIdentityReport(
        contract_id=contract.contract_id,
        entrypoint_matches=hmac.compare_digest(
            _sha256(root / contract.legacy_entrypoint),
            contract.legacy_entrypoint_sha256,
        ),
        model_transport_matches=hmac.compare_digest(
            _sha256(root / contract.legacy_models_path),
            contract.legacy_models_sha256,
        ),
    )


def _legacy_private_binding(
    path: Path, *, expected_sha256: str
) -> tuple[str, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("legacy model transport config must be a regular file")
    if not hmac.compare_digest(_sha256(path), expected_sha256):
        raise LegacyCutoverBlocked(
            "legacy model transport identity differs from the reviewed contract"
        )
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object_pairs
    )
    if not isinstance(value, dict):
        raise ValueError("legacy model transport config must be an object")
    endpoint = value.get("endpoint")
    credential_environment = value.get("api_key_env")
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("legacy model transport endpoint is unavailable")
    if not isinstance(credential_environment, str) or not credential_environment:
        raise ValueError("legacy credential environment is unavailable")
    return endpoint, credential_environment


def verify_binding_parity(
    contract: LegacyForwardContract,
    *,
    repo_root: str | Path,
    generation_bindings_path: str | Path,
) -> BindingParityReport:
    """Compare private binding identities without reading the credential value."""

    root = Path(repo_root).expanduser().resolve()
    endpoint, credential_environment = _legacy_private_binding(
        root / contract.legacy_models_path,
        expected_sha256=contract.legacy_models_sha256,
    )
    selected_binding_path = Path(generation_bindings_path).expanduser().resolve()
    binding = LocalRouteBindings.load(selected_binding_path).require(
        contract.route_profile_id
    )
    return BindingParityReport(
        contract_id=contract.contract_id,
        route_profile_id=contract.route_profile_id,
        binding_sha256=binding.binding_sha256,
        endpoint_matches=hmac.compare_digest(endpoint, binding.endpoint),
        credential_environment_matches=hmac.compare_digest(
            credential_environment, binding.credential_env
        ),
        selected_binding_path=selected_binding_path,
    )


def forward_if_attested(
    plan: LegacyForwardPlan,
    *,
    binding_parity: BindingParityReport | None,
    campaign_main: Callable[[Iterable[str] | None], int],
) -> int:
    """Execute only after every compatibility gate is explicitly true."""

    if plan.legacy_command in {"preflight", "run"}:
        if binding_parity is None or not binding_parity.matches:
            raise LegacyCutoverBlocked("legacy route binding parity is not proven")
    if not plan.contract.cutover_ready:
        raise LegacyCutoverBlocked(
            "legacy terminal-output and artifact parity are not proven"
        )
    argv = plan.campaign_argv
    if plan.legacy_command in {"preflight", "run"}:
        assert binding_parity is not None
        argv = (
            *argv,
            "--generation-bindings",
            str(binding_parity.selected_binding_path),
        )
    return campaign_main(argv)


def blocked_surfaces() -> tuple[dict[str, str], ...]:
    """Return non-forwardable commands that require a future campaign contract."""

    return (
        {
            "entrypoint": "tools/api2_generation_recovery_v1/recovery_runner.py",
            "reason": "selected-brief timeout recovery is not a v2 campaign surface",
        },
        {
            "entrypoint": (
                "tools/api2_generation_recovery_v1/fresh_case_retry_runner.py"
            ),
            "reason": "fresh semantic retry chances are not a v2 campaign surface",
        },
        {
            "entrypoint": (
                "tools/api3_opus48_thinking_runner_v1/preflight.py"
            ),
            "reason": "the optional required-reasoning-signal mode has no v2 contract",
        },
        {
            "entrypoint": (
                "tools/api3_opus48_thinking_runner_v1/retry_runner.py"
            ),
            "reason": "historical failure-set selection is not a v2 campaign surface",
        },
    )
