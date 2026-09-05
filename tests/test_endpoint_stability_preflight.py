from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchmark.models import EndpointConfigurationError
from benchmark.models.endpoint_preflight import (
    EndpointStabilityPreflightError,
    run_endpoint_stability_preflight,
)


def _png(path: Path) -> None:
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    )


class _HealthyModel:
    def __init__(self, **_: Any) -> None:
        self.last_request_metadata: dict[str, Any] = {}

    def chat_messages(self, messages: list[dict], **_: Any) -> str:
        assert any(
            isinstance(item.get("content"), list)
            for item in messages
            if isinstance(item, dict)
        )
        self.last_request_metadata = {
            "finish_reason": "stop",
            "usage": {"total_tokens": 7},
        }
        return '{"ok":true}'


class _BrokenRouteModel(_HealthyModel):
    def chat_messages(self, messages: list[dict], **_: Any) -> str:
        del messages
        raise EndpointConfigurationError(
            "HTTP 400: on-demand throughput isn't supported; use an "
            "inference profile"
        )


def test_repeated_multimodal_preflight_requires_every_attempt(tmp_path: Path) -> None:
    image = tmp_path / "scene.png"
    _png(image)

    report = run_endpoint_stability_preflight(
        endpoint="http://127.0.0.1:4010/v1",
        model_id="test-model",
        api_key_env="TEST_KEY",
        image_path=image,
        attempts=6,
        concurrency=2,
        model_factory=_HealthyModel,
    )

    assert report["status"] == "passed"
    assert report["completed_attempts"] == 6
    assert report["api_invocations"] == 6
    assert report["fatal_route_configuration"] is False


def test_route_configuration_failure_trips_preflight_gate(tmp_path: Path) -> None:
    image = tmp_path / "scene.png"
    _png(image)

    with pytest.raises(EndpointStabilityPreflightError) as captured:
        run_endpoint_stability_preflight(
            endpoint="http://127.0.0.1:4010/v1",
            model_id="claude-opus-5-aihub",
            api_key_env="TEST_KEY",
            image_path=image,
            attempts=10,
            concurrency=2,
            model_factory=_BrokenRouteModel,
        )

    report = captured.value.report
    assert report["status"] == "failed"
    assert report["fatal_route_configuration"] is True
    assert report["api_invocations"] < 10
    assert any(
        item.get("status") == "cancelled_after_route_failure"
        for item in report["results"]
    )
