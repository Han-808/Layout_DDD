from __future__ import annotations

import io
import json
from pathlib import Path
import urllib.error
import urllib.request

import pytest

import benchmark.models.openai_compatible_model as openai_compatible_model_module
from benchmark.models import (
    EndpointConfigurationError,
    MissingAPIKeyError,
    OpenAICompatibleModel,
)
from benchmark.models.openai_compatible_model import EndpointHTTPError, _RejectRedirectHandler
from benchmark.nl_scene.converter import ObjectPlanConversionError, call_chat_model
from benchmark.visual_judge import build_openai_compatible_vlm_judge


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _chat_response(content: str = '{"ok":true}') -> _FakeResponse:
    return _FakeResponse(
        {
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }
    )


def test_official_openai_endpoint_uses_openai_api_key_without_storing_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-key")

    def fake_urlopen(request: urllib.request.Request, timeout: int):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _chat_response()

    monkeypatch.setattr(openai_compatible_model_module, "_urlopen_no_redirect", fake_urlopen)
    model = OpenAICompatibleModel(
        name="openai-test",
        endpoint="https://api.openai.com/v1",
        model_id="gpt-test",
        max_tokens=123,
        max_tokens_field="max_completion_tokens",
        send_temperature=False,
    )

    assert model.chat_messages([{"role": "user", "content": "Return JSON."}]) == '{"ok":true}'
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-secret-key"
    assert captured["payload"]["max_completion_tokens"] == 123
    assert "max_tokens" not in captured["payload"]
    assert "temperature" not in captured["payload"]
    assert model.api_key_env == "OPENAI_API_KEY"
    assert model.last_request_metadata["authorization_configured"] is True
    assert "test-secret-key" not in json.dumps(model.last_request_metadata)


def test_official_openai_endpoint_fails_clearly_when_key_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    model = OpenAICompatibleModel(
        name="openai-test",
        endpoint="https://api.openai.com/v1",
        model_id="gpt-test",
    )

    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY"):
        model.list_models()


def test_local_openai_compatible_endpoint_remains_auth_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_urlopen(request: urllib.request.Request, timeout: int):
        captured["authorization"] = request.get_header("Authorization")
        return _FakeResponse({"object": "list", "data": []})

    monkeypatch.setattr(openai_compatible_model_module, "_urlopen_no_redirect", fake_urlopen)
    model = OpenAICompatibleModel(
        name="local-test",
        endpoint="http://127.0.0.1:8298/v1",
        model_id="local-model",
    )

    assert model.list_models() == {"object": "list", "data": []}
    assert captured["authorization"] is None
    assert model.api_key_env is None
    assert model.require_api_key is False


def test_explicit_custom_api_key_env_is_required_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COMPANY_OPENAI_KEY", raising=False)
    model = OpenAICompatibleModel(
        name="gateway-test",
        endpoint="https://gateway.example.com/v1",
        model_id="remote-model",
        api_key_env="COMPANY_OPENAI_KEY",
    )

    with pytest.raises(MissingAPIKeyError, match="COMPANY_OPENAI_KEY"):
        model.list_models()


def test_converter_shared_client_honors_custom_api_key_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    monkeypatch.setenv("COMPANY_OPENAI_KEY", "company-secret")

    def fake_urlopen(request: urllib.request.Request, timeout: int):
        captured["authorization"] = request.get_header("Authorization")
        return _chat_response()

    monkeypatch.setattr(openai_compatible_model_module, "_urlopen_no_redirect", fake_urlopen)
    result = call_chat_model(
        [{"role": "user", "content": "Return JSON."}],
        model_config={
            "endpoint": "https://gateway.example.com/v1",
            "model": "remote-model",
            "api_key_env": "COMPANY_OPENAI_KEY",
        },
    )

    assert result == '{"ok":true}'
    assert captured["authorization"] == "Bearer company-secret"


def test_vlm_builder_propagates_openai_request_compatibility_options() -> None:
    judge = build_openai_compatible_vlm_judge(
        {
            "endpoint": "https://api.openai.com/v1",
            "model": "gpt-test",
            "max_tokens_field": "max_completion_tokens",
            "send_temperature": False,
        }
    )

    assert judge.model.api_key_env == "OPENAI_API_KEY"
    assert judge.model.require_api_key is True
    assert judge.model.max_tokens_field == "max_completion_tokens"
    assert judge.model.send_temperature is False


def test_literal_api_keys_are_rejected_at_shared_config_boundaries() -> None:
    secret = "must-never-appear-in-error"
    with pytest.raises(ValueError, match="use api_key_env") as judge_error:
        build_openai_compatible_vlm_judge(
            {
                "endpoint": "https://api.openai.com/v1",
                "model": "gpt-test",
                "api_key": secret,
            }
        )
    assert secret not in str(judge_error.value)

    with pytest.raises(ObjectPlanConversionError, match="use api_key_env") as converter_error:
        call_chat_model(
            [{"role": "user", "content": "Return JSON."}],
            model_config={
                "endpoint": "https://api.openai.com/v1",
                "model": "gpt-test",
                "api_key": secret,
            },
        )
    assert secret not in str(converter_error.value)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        OpenAICompatibleModel(
            name="direct-secret-test",
            endpoint="https://api.openai.com/v1",
            model_id="gpt-test",
            api_key=secret,  # type: ignore[call-arg]
        )


def test_non_loopback_http_endpoint_is_rejected_but_loopback_http_remains_valid() -> None:
    with pytest.raises(ValueError, match="must use HTTPS"):
        OpenAICompatibleModel(
            name="insecure-remote",
            endpoint="http://gateway.example.com/v1",
            model_id="remote-model",
        )

    local = OpenAICompatibleModel(
        name="local",
        endpoint="http://localhost:8298/v1",
        model_id="local-model",
    )
    assert local.endpoint == "http://localhost:8298/v1"


def test_api_key_env_must_be_an_environment_variable_name_without_echoing_value() -> None:
    supplied_literal = "sk-live-secret-that-must-not-be-logged"
    with pytest.raises(ValueError, match="environment-variable name") as captured:
        OpenAICompatibleModel(
            name="invalid-env-name",
            endpoint="https://gateway.example.com/v1",
            model_id="remote-model",
            api_key_env=supplied_literal,
        )
    assert supplied_literal not in str(captured.value)


def test_redirect_handler_refuses_redirect_before_authorization_can_be_forwarded() -> None:
    request = urllib.request.Request(
        "https://gateway.example.com/v1/models",
        headers={"Authorization": "Bearer test-secret"},
    )
    target = "http://redirect-target.example/leak"
    with pytest.raises(urllib.error.HTTPError, match="redirect responses are not followed") as captured:
        _RejectRedirectHandler().redirect_request(
            request,
            io.BytesIO(b"redirect"),
            302,
            "Found",
            {"Location": target},
            target,
        )
    assert target not in str(captured.value)


def test_http_error_body_is_bounded_and_redacts_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "company-super-secret"
    monkeypatch.setenv("COMPANY_OPENAI_KEY", secret)
    body = (
        f'{{"authorization":"Bearer {secret}","api_key":"{secret}","detail":"'
        + ("x" * 20_000)
        + '"}'
    ).encode("utf-8")

    def fake_urlopen(request: urllib.request.Request, timeout: int):
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "bad request",
            {},
            io.BytesIO(body),
        )

    monkeypatch.setattr(openai_compatible_model_module, "_urlopen_no_redirect", fake_urlopen)
    model = OpenAICompatibleModel(
        name="bounded-error",
        endpoint="https://gateway.example.com/v1",
        model_id="remote-model",
        api_key_env="COMPANY_OPENAI_KEY",
    )

    with pytest.raises(EndpointHTTPError) as captured:
        model.list_models()
    message = str(captured.value)
    assert secret not in message
    assert "<redacted>" in message
    assert "<truncated>" in message
    assert len(message) < 2_200


def test_bedrock_on_demand_model_id_failure_is_typed_as_route_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = json.dumps(
        {
            "error": {
                "message": (
                    "Invocation of model ID anthropic.claude-opus-5 with "
                    "on-demand throughput isn’t supported. Retry your "
                    "request with the ID or ARN of an inference profile "
                    "that contains this model."
                )
            }
        }
    ).encode("utf-8")

    def fake_urlopen(request: urllib.request.Request, timeout: int):
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "bad request",
            {},
            io.BytesIO(body),
        )

    monkeypatch.setattr(
        openai_compatible_model_module,
        "_urlopen_no_redirect",
        fake_urlopen,
    )
    model = OpenAICompatibleModel(
        name="bad-bedrock-route",
        endpoint="http://127.0.0.1:4010/v1",
        model_id="claude-opus-5-aihub",
    )

    with pytest.raises(EndpointConfigurationError, match="inference profile"):
        model.chat_messages([{"role": "user", "content": "hello"}])


def test_credential_entrypoint_scripts_have_no_literal_key_channel() -> None:
    root = Path(__file__).resolve().parents[1]
    scripts = [
        "scripts/run_p0b_camera_ablation.py",
        "scripts/probe_internal_model_api.py",
        "scripts/check_model_endpoint.py",
        "scripts/run_scene_harness.py",
        "scripts/author_reference_annotation.py",
    ]
    for relative in scripts:
        source = (root / relative).read_text(encoding="utf-8")
        assert 'API = "xxx"' not in source
        assert '"--api-key",' not in source
        assert "'--api-key'," not in source
