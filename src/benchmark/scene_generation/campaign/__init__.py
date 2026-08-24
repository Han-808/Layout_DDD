"""Portable profiles and the current model-agnostic generation campaign CLI."""

from benchmark.scene_generation.campaign.loader import (
    load_campaign_profile_bundle,
    load_campaign_contract_registry,
    load_campaign_profile_registry,
    load_model_profile_registry,
    load_route_profile_registry,
)
from benchmark.scene_generation.campaign.profiles import (
    CampaignProfile,
    CampaignProfileBundle,
    CampaignProfileRegistry,
    GatewayOptions,
    ModelProfile,
    ModelProfileRegistry,
    RequestOptions,
    RouteProfile,
    RouteProfileRegistry,
    TransportPolicy,
)


_LAZY_API_EXPORTS = frozenset(
    {
        "PreparedGenerationCampaign",
        "check_generation_campaign",
        "preflight_generation_campaign",
        "prepare_generation_campaign",
        "resolve_generation_campaign",
        "resource_gate_generation_campaign",
        "run_generation_campaign",
    }
)


def __getattr__(name: str):
    if name not in _LAZY_API_EXPORTS:
        raise AttributeError(name)
    from benchmark.scene_generation.campaign import api

    return getattr(api, name)


__all__ = [
    "PreparedGenerationCampaign",
    "CampaignProfile",
    "CampaignProfileBundle",
    "CampaignProfileRegistry",
    "GatewayOptions",
    "ModelProfile",
    "ModelProfileRegistry",
    "RequestOptions",
    "RouteProfile",
    "RouteProfileRegistry",
    "TransportPolicy",
    "check_generation_campaign",
    "load_campaign_profile_bundle",
    "load_campaign_contract_registry",
    "load_campaign_profile_registry",
    "load_model_profile_registry",
    "load_route_profile_registry",
    "preflight_generation_campaign",
    "prepare_generation_campaign",
    "resolve_generation_campaign",
    "resource_gate_generation_campaign",
    "run_generation_campaign",
]
