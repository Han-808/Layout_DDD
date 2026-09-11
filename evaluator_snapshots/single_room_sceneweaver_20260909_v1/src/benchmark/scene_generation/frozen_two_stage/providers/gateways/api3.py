"""API3 header policy for the OpenAI-compatible generation gateway.

See ``docs/generation_transport_compatibility.md``.  Credentials remain
runtime-only and never appear in ``public_dict`` or source manifests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmark.scene_generation.frozen_two_stage.providers.base import ProviderModel


@dataclass(frozen=True)
class API3GatewayPolicy:
    """Build the frozen API3 Bearer/SessionID/StrategyType headers."""

    runner_version: str = "2.0.0"

    def request_headers(
        self, model: ProviderModel, session_id: str
    ) -> dict[str, str]:
        return {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "User-Agent": f"hy34-two-stage-generator/{self.runner_version}",
            "Authorization": f"Bearer {model.api_key}",
            "SessionID": session_id,
            "StrategyType": model.strategy_type,
        }

    def public_dict(self) -> dict[str, Any]:
        return {
            "gateway": "api3",
            "runner_version": self.runner_version,
            "session_header": "SessionID",
            "strategy_header": "StrategyType",
        }
