"""Stable lifecycle facade for base and additive generation campaigns."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from benchmark.scene_generation.campaign.execution import (
    DEFAULT_PROFILE_RELATIVE,
    PreparedCampaign,
    gate_resources,
    preflight_campaign,
    prepare_campaign,
    repository_root,
    resolve_bindings,
    run_campaign,
)
from benchmark.scene_generation.campaign.loader import load_campaign_profile_bundle
from benchmark.scene_generation.campaign.multi_room_execution import (
    PreparedMultiRoomCampaign,
    preflight_multi_room_campaign,
    prepare_multi_room_campaign,
    run_prepared_multi_room_campaign,
)
from benchmark.scene_generation.retrieval import build_runtime


PreparedGenerationCampaign = PreparedCampaign | PreparedMultiRoomCampaign


def prepare_generation_campaign(
    campaign_id: str,
    *,
    floor_plan_path: str | Path | None = None,
    profile_root: str | Path | None = None,
    retrieval_catalog_path: str | Path | None = None,
    trust_manifest: str | Path | None = None,
) -> PreparedGenerationCampaign:
    """Explicitly dispatch by reviewed campaign registry membership."""

    root = repository_root()
    profiles = (
        root / DEFAULT_PROFILE_RELATIVE
        if profile_root is None
        else Path(profile_root).expanduser().resolve()
    )
    base = load_campaign_profile_bundle(profiles)
    if campaign_id in base.campaigns.by_id:
        if floor_plan_path is not None:
            raise ValueError(
                "single-room campaigns do not accept a multi-room floor plan"
            )
        return prepare_campaign(
            campaign_id,
            profile_root=profiles,
            retrieval_catalog_path=retrieval_catalog_path,
            trust_manifest=trust_manifest,
        )
    if floor_plan_path is None:
        raise ValueError(
            "additive multi-room campaigns require an explicit floor-plan artifact"
        )
    return prepare_multi_room_campaign(
        campaign_id,
        floor_plan_path=floor_plan_path,
        profile_root=profiles,
        retrieval_catalog_path=retrieval_catalog_path,
        trust_manifest=trust_manifest,
    )


def check_generation_campaign(
    campaign_id: str,
    **kwargs: Any,
) -> dict[str, Any]:
    return prepare_generation_campaign(campaign_id, **kwargs).public_dict()


def resolve_generation_campaign(
    prepared: PreparedGenerationCampaign,
    *,
    generation_bindings_path: str | Path | None = None,
    resource_bindings_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    return resolve_bindings(
        prepared,
        generation_bindings_path=generation_bindings_path,
        resource_bindings_path=resource_bindings_path,
        environ=environ,
    )


def resource_gate_generation_campaign(
    prepared: PreparedGenerationCampaign,
    *,
    resource_bindings_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    runtime_factory: Callable[..., Any] = build_runtime,
) -> tuple[Any | None, dict[str, Any]]:
    return gate_resources(
        prepared,
        resource_bindings_path=resource_bindings_path,
        environ=environ,
        runtime_factory=runtime_factory,
    )


def preflight_generation_campaign(
    prepared: PreparedGenerationCampaign,
    *,
    generation_bindings_path: str | Path | None = None,
    resource_bindings_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    runtime_factory: Callable[..., Any] = build_runtime,
    transport: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], Any | None]:
    if isinstance(prepared, PreparedMultiRoomCampaign):
        return preflight_multi_room_campaign(
            prepared,
            generation_bindings_path=generation_bindings_path,
            resource_bindings_path=resource_bindings_path,
            environ=environ,
            runtime_factory=runtime_factory,
            transport=transport,
        )
    return preflight_campaign(
        prepared,
        generation_bindings_path=generation_bindings_path,
        resource_bindings_path=resource_bindings_path,
        environ=environ,
        runtime_factory=runtime_factory,
        transport=transport,
    )


def run_generation_campaign(
    prepared: PreparedGenerationCampaign,
    *,
    output_root: str | Path,
    generation_bindings_path: str | Path | None = None,
    resource_bindings_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    runtime_factory: Callable[..., Any] = build_runtime,
    transport: Callable[..., Any] | None = None,
    resume: bool = False,
) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    if isinstance(prepared, PreparedMultiRoomCampaign):
        return run_prepared_multi_room_campaign(
            prepared,
            output_root=output_root,
            generation_bindings_path=generation_bindings_path,
            resource_bindings_path=resource_bindings_path,
            environ=environ,
            progress=progress,
            runtime_factory=runtime_factory,
            transport=transport,
            resume=resume,
        )
    if resume:
        raise ValueError(
            "--resume is additive multi-room behavior and is not enabled for "
            "the frozen single-room campaign"
        )
    return run_campaign(
        prepared,
        output_root=output_root,
        generation_bindings_path=generation_bindings_path,
        resource_bindings_path=resource_bindings_path,
        environ=environ,
        progress=progress,
        runtime_factory=runtime_factory,
        transport=transport,
    )


__all__ = [
    "PreparedGenerationCampaign",
    "check_generation_campaign",
    "preflight_generation_campaign",
    "prepare_generation_campaign",
    "resolve_generation_campaign",
    "resource_gate_generation_campaign",
    "run_generation_campaign",
]
