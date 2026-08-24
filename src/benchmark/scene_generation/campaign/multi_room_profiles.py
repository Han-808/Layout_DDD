"""Strict additive profile fragments for multi-room generation campaigns."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

from benchmark.scene_generation.campaign.profiles import (
    CampaignProfileBundle,
    ExecutionPolicyContract,
    ModelProfile,
    RouteProfile,
)


ADDITIVE_REGISTRY_MANIFEST_SCHEMA_VERSION = (
    "generation_additive_registry_manifest_v1"
)
MULTI_ROOM_FRAGMENT_SCHEMA_VERSION = "generation_multi_room_campaign_fragment_v1"
MULTI_ROOM_WORKFLOW_PROFILE_ID = (
    "frozen-two-stage-multi-room-with-architecture-v1"
)
MULTI_ROOM_GENERATION_MODE = "multi_room_with_architecture_v1"
DEFAULT_ADDITIVE_REGISTRY_RELATIVE = Path(
    "configs/generation_extensions/additive_registry_manifest_v1.json"
)
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AdditiveProfileError(ValueError):
    """Raised when an additive generation profile is not exact and trusted."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise AdditiveProfileError(f"duplicate additive-profile key: {key}")
        value[key] = child
    return value


def _reject_constant(value: str) -> None:
    raise AdditiveProfileError(f"non-finite additive-profile number: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdditiveProfileError(
            f"invalid additive registry JSON: {type(exc).__name__}"
        ) from exc
    if not isinstance(value, dict):
        raise AdditiveProfileError("additive registry root must be an object")
    return value


def _exact(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise AdditiveProfileError(
            f"{path} keys mismatch: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )


def _string(value: Any, path: str, *, identifier: bool = False) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise AdditiveProfileError(f"{path} must be a non-empty trimmed string")
    if identifier and not _IDENTIFIER.fullmatch(value):
        raise AdditiveProfileError(f"{path} must be a portable identifier")
    return value


def _digest(value: Any, path: str) -> str:
    text = _string(value, path)
    if not _SHA256.fullmatch(text):
        raise AdditiveProfileError(f"{path} must be a lowercase sha256")
    return text


def _safe_relative(value: Any, path: str) -> Path:
    text = _string(value, path)
    relative = Path(text)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != text:
        raise AdditiveProfileError(f"{path} must be a safe relative path")
    return relative


@dataclass(frozen=True, slots=True)
class MultiRoomWorkflowContract:
    workflow_profile_id: str
    generation_mode: str
    core_bundle_id: str
    core_runner_version: str
    core_runner_source_sha256: str
    compatibility_runtime_version: str
    compatibility_source_manifest_sha256: str
    stage_a_prompt_sha256: str
    stage_c_prompt_sha256: str
    floor_plan_schema_sha256: str
    object_plan_schema_sha256: str
    multi_room_scene_schema_sha256: str
    compiled_architecture_schema_sha256: str
    room_evaluation_index_schema_sha256: str
    assembly_manifest_schema_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "workflow_profile_id",
            "generation_mode",
            "core_bundle_id",
            "core_runner_version",
            "compatibility_runtime_version",
        ):
            _string(getattr(self, name), name, identifier=True)
        if self.workflow_profile_id != MULTI_ROOM_WORKFLOW_PROFILE_ID:
            raise AdditiveProfileError("unsupported multi-room workflow profile")
        if self.generation_mode != MULTI_ROOM_GENERATION_MODE:
            raise AdditiveProfileError("unsupported multi-room generation mode")
        for name in (
            "core_runner_source_sha256",
            "compatibility_source_manifest_sha256",
            "stage_a_prompt_sha256",
            "stage_c_prompt_sha256",
            "floor_plan_schema_sha256",
            "object_plan_schema_sha256",
            "multi_room_scene_schema_sha256",
            "compiled_architecture_schema_sha256",
            "room_evaluation_index_schema_sha256",
            "assembly_manifest_schema_sha256",
        ):
            _digest(getattr(self, name), name)

    def public_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class MultiRoomArtifactContract:
    artifact_contract_id: str
    run_manifest_schema_version: str
    room_result_schema_version: str
    global_scene_schema_version: str
    compiled_architecture_schema_version: str
    room_evaluation_index_schema_version: str
    assembly_manifest_schema_version: str
    summary_schema_version: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _string(getattr(self, name), name, identifier=True)

    def public_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class MultiRoomCampaignProfile:
    campaign_id: str
    workflow_profile_id: str
    model_profile_id: str
    retrieval_profile_id: str
    execution_policy_id: str
    artifact_contract_id: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _string(getattr(self, name), name, identifier=True)
        if self.workflow_profile_id != MULTI_ROOM_WORKFLOW_PROFILE_ID:
            raise AdditiveProfileError("campaign references unsupported workflow")

    def public_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "workflow_profile_id": self.workflow_profile_id,
            "generation_mode": MULTI_ROOM_GENERATION_MODE,
            "model_profile_id": self.model_profile_id,
            "retrieval_profile_id": self.retrieval_profile_id,
            "execution_policy_id": self.execution_policy_id,
            "artifact_contract_id": self.artifact_contract_id,
        }


@dataclass(frozen=True, slots=True)
class ResolvedMultiRoomCampaign:
    campaign: MultiRoomCampaignProfile
    workflow: MultiRoomWorkflowContract
    artifact: MultiRoomArtifactContract
    model: ModelProfile
    route: RouteProfile
    execution: ExecutionPolicyContract


@dataclass(frozen=True, slots=True)
class MultiRoomProfileRegistry:
    manifest_path: Path
    manifest_sha256: str
    fragment_hashes: Mapping[str, str]
    workflows: Mapping[str, MultiRoomWorkflowContract]
    artifacts: Mapping[str, MultiRoomArtifactContract]
    campaigns: Mapping[str, MultiRoomCampaignProfile]
    base_bundle: CampaignProfileBundle

    def resolve(self, campaign_id: str) -> ResolvedMultiRoomCampaign:
        try:
            campaign = self.campaigns[campaign_id]
            workflow = self.workflows[campaign.workflow_profile_id]
            artifact = self.artifacts[campaign.artifact_contract_id]
            model = self.base_bundle.models.by_id[campaign.model_profile_id]
            route = self.base_bundle.routes.by_id[model.route_profile_id]
            execution = self.base_bundle.contracts.execution_by_id[
                campaign.execution_policy_id
            ]
        except KeyError as exc:
            raise AdditiveProfileError(
                f"unknown or incomplete multi-room campaign {campaign_id!r}: {exc}"
            ) from exc
        return ResolvedMultiRoomCampaign(
            campaign=campaign,
            workflow=workflow,
            artifact=artifact,
            model=model,
            route=route,
            execution=execution,
        )


def _workflow(value: Any, path: str) -> MultiRoomWorkflowContract:
    if not isinstance(value, dict):
        raise AdditiveProfileError(f"{path} must be an object")
    fields = set(MultiRoomWorkflowContract.__dataclass_fields__)
    _exact(value, fields, path)
    return MultiRoomWorkflowContract(
        **{name: _string(value[name], f"{path}.{name}") for name in fields}
    )


def _artifact(value: Any, path: str) -> MultiRoomArtifactContract:
    if not isinstance(value, dict):
        raise AdditiveProfileError(f"{path} must be an object")
    fields = set(MultiRoomArtifactContract.__dataclass_fields__)
    _exact(value, fields, path)
    return MultiRoomArtifactContract(
        **{name: _string(value[name], f"{path}.{name}") for name in fields}
    )


def _campaign(value: Any, path: str) -> MultiRoomCampaignProfile:
    if not isinstance(value, dict):
        raise AdditiveProfileError(f"{path} must be an object")
    fields = set(MultiRoomCampaignProfile.__dataclass_fields__)
    _exact(value, fields, path)
    return MultiRoomCampaignProfile(
        **{name: _string(value[name], f"{path}.{name}") for name in fields}
    )


def load_multi_room_profile_registry(
    repo_root: str | Path,
    base_bundle: CampaignProfileBundle,
    *,
    manifest_path: str | Path | None = None,
) -> MultiRoomProfileRegistry:
    """Load explicitly listed additive fragments in fixed manifest order."""

    root = Path(repo_root).expanduser().resolve()
    manifest = (
        root / DEFAULT_ADDITIVE_REGISTRY_RELATIVE
        if manifest_path is None
        else Path(manifest_path).expanduser().resolve()
    )
    value = _load_json(manifest)
    _exact(value, {"schema_version", "fragments"}, "additive manifest")
    if value["schema_version"] != ADDITIVE_REGISTRY_MANIFEST_SCHEMA_VERSION:
        raise AdditiveProfileError("unsupported additive registry manifest version")
    fragments = value["fragments"]
    if not isinstance(fragments, list) or not fragments:
        raise AdditiveProfileError("additive manifest fragments must be non-empty")

    workflows: dict[str, MultiRoomWorkflowContract] = {}
    artifacts: dict[str, MultiRoomArtifactContract] = {}
    campaigns: dict[str, MultiRoomCampaignProfile] = {}
    fragment_hashes: dict[str, str] = {}
    fragment_keys = {
        "fragment_id",
        "path",
        "sha256",
        "generation_mode",
        "workflow_profile_id",
    }
    for index, raw in enumerate(fragments):
        if not isinstance(raw, dict):
            raise AdditiveProfileError(f"fragments[{index}] must be an object")
        _exact(raw, fragment_keys, f"fragments[{index}]")
        fragment_id = _string(
            raw["fragment_id"], f"fragments[{index}].fragment_id", identifier=True
        )
        if fragment_id in fragment_hashes:
            raise AdditiveProfileError(f"duplicate additive fragment: {fragment_id}")
        if raw["generation_mode"] != MULTI_ROOM_GENERATION_MODE:
            raise AdditiveProfileError("unsupported additive generation mode")
        if raw["workflow_profile_id"] != MULTI_ROOM_WORKFLOW_PROFILE_ID:
            raise AdditiveProfileError("unsupported additive workflow profile")
        relative = _safe_relative(raw["path"], f"fragments[{index}].path")
        fragment_path = (manifest.parent / relative).resolve()
        try:
            fragment_path.relative_to(manifest.parent.resolve())
        except ValueError as exc:
            raise AdditiveProfileError("additive fragment escapes manifest root") from exc
        expected_sha = _digest(raw["sha256"], f"fragments[{index}].sha256")
        actual_sha = _sha256(fragment_path)
        if actual_sha != expected_sha:
            raise AdditiveProfileError(
                f"additive fragment hash mismatch for {fragment_id}"
            )
        fragment_hashes[fragment_id] = actual_sha
        fragment = _load_json(fragment_path)
        _exact(
            fragment,
            {"schema_version", "workflow", "artifact_contract", "campaigns"},
            fragment_id,
        )
        if fragment["schema_version"] != MULTI_ROOM_FRAGMENT_SCHEMA_VERSION:
            raise AdditiveProfileError(
                f"unsupported fragment schema for {fragment_id}"
            )
        workflow = _workflow(fragment["workflow"], f"{fragment_id}.workflow")
        artifact = _artifact(
            fragment["artifact_contract"], f"{fragment_id}.artifact_contract"
        )
        if workflow.workflow_profile_id in workflows:
            raise AdditiveProfileError(
                f"duplicate additive workflow: {workflow.workflow_profile_id}"
            )
        if artifact.artifact_contract_id in artifacts:
            raise AdditiveProfileError(
                f"duplicate additive artifact contract: {artifact.artifact_contract_id}"
            )
        workflows[workflow.workflow_profile_id] = workflow
        artifacts[artifact.artifact_contract_id] = artifact
        raw_campaigns = fragment["campaigns"]
        if not isinstance(raw_campaigns, list) or not raw_campaigns:
            raise AdditiveProfileError(f"{fragment_id}.campaigns must be non-empty")
        for campaign_index, raw_campaign in enumerate(raw_campaigns):
            campaign = _campaign(
                raw_campaign, f"{fragment_id}.campaigns[{campaign_index}]"
            )
            if campaign.campaign_id in campaigns:
                raise AdditiveProfileError(
                    f"duplicate additive campaign: {campaign.campaign_id}"
                )
            if campaign.workflow_profile_id != workflow.workflow_profile_id:
                raise AdditiveProfileError("campaign/workflow fragment mismatch")
            if campaign.artifact_contract_id != artifact.artifact_contract_id:
                raise AdditiveProfileError("campaign/artifact fragment mismatch")
            if campaign.model_profile_id not in base_bundle.models.by_id:
                raise AdditiveProfileError(
                    f"campaign references unknown model: {campaign.model_profile_id}"
                )
            if campaign.execution_policy_id not in base_bundle.contracts.execution_by_id:
                raise AdditiveProfileError(
                    "campaign references unknown execution policy: "
                    f"{campaign.execution_policy_id}"
                )
            model = base_bundle.models.by_id[campaign.model_profile_id]
            route = base_bundle.routes.by_id[model.route_profile_id]
            execution = base_bundle.contracts.execution_by_id[
                campaign.execution_policy_id
            ]
            if route.runner_version != workflow.core_runner_version:
                raise AdditiveProfileError("route/core runner version mismatch")
            if (
                execution.max_infrastructure_retries
                != model.transport_policy.max_infrastructure_retries
                or execution.retry_delay_seconds
                != model.transport_policy.retry_delay_seconds
            ):
                raise AdditiveProfileError("model/execution retry policy mismatch")
            campaigns[campaign.campaign_id] = campaign

    return MultiRoomProfileRegistry(
        manifest_path=manifest,
        manifest_sha256=_sha256(manifest),
        fragment_hashes=MappingProxyType(dict(fragment_hashes)),
        workflows=MappingProxyType(dict(workflows)),
        artifacts=MappingProxyType(dict(artifacts)),
        campaigns=MappingProxyType(dict(campaigns)),
        base_bundle=base_bundle,
    )
