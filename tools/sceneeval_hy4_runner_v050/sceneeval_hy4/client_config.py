"""Strict loading of the frozen OpenAI-client YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any


CLIENT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "openai_clients.yaml"
SESSION_ID_SENTINEL = "DYNAMIC_UNIQUE_PER_REQUEST"


class ClientConfigError(ValueError):
    pass


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ClientConfigError(f"{label} must be a mapping")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ClientConfigError(
            f"{label} keys must be exactly {sorted(expected)!r}; "
            f"received {sorted(value)!r}"
        )


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ClientConfigError(f"{label} must be numeric")
    return float(value)


def load_run_config(path: Path = CLIENT_CONFIG_PATH):
    """Load the exact approved YAML and return a runner RunConfig."""
    try:
        import yaml
    except ImportError as exc:
        raise ClientConfigError(
            "PyYAML is required to load openai_clients.yaml"
        ) from exc

    try:
        root = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ClientConfigError(f"cannot load client YAML {path}: {exc}") from exc

    root = _mapping(root, "root")
    _exact_keys(root, {"openai_clients"}, "root")
    clients = _mapping(root["openai_clients"], "openai_clients")
    _exact_keys(clients, {"main"}, "openai_clients")
    main = _mapping(clients["main"], "openai_clients.main")
    _exact_keys(main, {"name", "config"}, "openai_clients.main")
    if main["name"] != "litellm":
        raise ClientConfigError("openai_clients.main.name must be 'litellm'")

    config = _mapping(main["config"], "openai_clients.main.config")
    _exact_keys(config, {"client_args", "request_args"}, "main.config")
    client_args = _mapping(config["client_args"], "client_args")
    _exact_keys(client_args, {"api_key", "base_url"}, "client_args")
    request_args = _mapping(config["request_args"], "request_args")
    _exact_keys(
        request_args,
        {
            "model",
            "timeout",
            "max_retries",
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "repetition_penalty",
            "extra_body",
            "extra_headers",
        },
        "request_args",
    )

    api_key = client_args["api_key"]
    base_url = client_args["base_url"]
    configured_model = request_args["model"]
    if not isinstance(api_key, str) or not api_key:
        raise ClientConfigError("client_args.api_key must be a non-empty string")
    if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
        raise ClientConfigError("client_args.base_url must be an HTTP(S) URL")
    if not isinstance(configured_model, str) or not configured_model.startswith("openai/"):
        raise ClientConfigError("request_args.model must use the openai/ provider prefix")
    wire_model = configured_model.removeprefix("openai/")
    if not wire_model:
        raise ClientConfigError("request_args.model lacks a wire model name")

    timeout = _number(request_args["timeout"], "request_args.timeout")
    temperature = _number(request_args["temperature"], "request_args.temperature")
    top_p = _number(request_args["top_p"], "request_args.top_p")
    repetition_penalty = _number(
        request_args["repetition_penalty"],
        "request_args.repetition_penalty",
    )
    max_retries = request_args["max_retries"]
    top_k = request_args["top_k"]
    max_tokens = request_args["max_tokens"]
    for value, label, minimum in (
        (max_retries, "request_args.max_retries", 0),
        (max_tokens, "request_args.max_tokens", 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ClientConfigError(f"{label} must be an integer >= {minimum}")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ClientConfigError("request_args.top_k must be an integer")
    if timeout <= 0:
        raise ClientConfigError("request_args.timeout must be > 0")

    extra_body = _mapping(request_args["extra_body"], "request_args.extra_body")
    _exact_keys(extra_body, {"chat_template_kwargs"}, "extra_body")
    template = _mapping(
        extra_body["chat_template_kwargs"],
        "extra_body.chat_template_kwargs",
    )
    _exact_keys(
        template,
        {"reasoning_effort", "preserved_thinking"},
        "chat_template_kwargs",
    )
    if not isinstance(template["reasoning_effort"], str):
        raise ClientConfigError("reasoning_effort must be a string")
    if not isinstance(template["preserved_thinking"], bool):
        raise ClientConfigError("preserved_thinking must be boolean")

    extra_headers = _mapping(
        request_args["extra_headers"],
        "request_args.extra_headers",
    )
    _exact_keys(extra_headers, {"SessionID", "StrategyType"}, "extra_headers")
    if extra_headers["SessionID"] != SESSION_ID_SENTINEL:
        raise ClientConfigError(
            f"extra_headers.SessionID must be {SESSION_ID_SENTINEL!r}"
        )
    if not isinstance(extra_headers["StrategyType"], str):
        raise ClientConfigError("extra_headers.StrategyType must be a string")

    from .runner import RunConfig

    return RunConfig(
        endpoint=base_url.rstrip("/") + "/chat/completions",
        configured_model=configured_model,
        wire_model=wire_model,
        api_key=api_key,
        timeout_seconds=timeout,
        max_retries=max_retries,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        reasoning_effort=template["reasoning_effort"],
        preserved_thinking=template["preserved_thinking"],
        strategy_type=extra_headers["StrategyType"],
    )
