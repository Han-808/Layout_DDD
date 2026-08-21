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


__all__ = [
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
    "load_campaign_profile_bundle",
    "load_campaign_contract_registry",
    "load_campaign_profile_registry",
    "load_model_profile_registry",
    "load_route_profile_registry",
]
