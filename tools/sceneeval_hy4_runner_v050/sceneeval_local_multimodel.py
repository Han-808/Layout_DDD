"""Local multi-model adapter for the frozen SceneEval-100 capture harness.

The adapter deliberately reuses the frozen input loader, prompt builder,
attempt lifecycle, retry policy, response capture, and immutable artifacts.
Only the provider-facing request profile differs from the HY4/MNET profile:

* the model is selected at runtime;
* only OpenAI-compatible request fields are sent;
* provider-specific routing headers are omitted; and
* the real bearer token is injected only at the transport boundary and is
  never returned to the frozen artifact writer.

The adapter is process-global while installed because the frozen runner was
not originally designed with injectable request/transport hooks. Public entry
points below install and restore the hooks synchronously; concurrent calls are
therefore intentionally unsupported.
"""

from __future__ import annotations

import json
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from sceneeval_hy4 import __version__
from sceneeval_hy4 import runner as frozen_runner
from sceneeval_hy4.artifacts import (
    request_json_bytes,
    sha256_bytes,
    write_json_exclusive,
)
from sceneeval_hy4.inputs import InputBatch, SceneInput, load_human100_jsonl
from sceneeval_hy4.prompt import build_system_prompt, build_user_prompt, protocol_text
from sceneeval_hy4.transport import post_once as wire_post_once


MODEL_ORDER = (
    "claude-opus-4-8",
    "kimi-k3",
    "gpt-5.6-luna",
)
DEFAULT_BASE_URL = "https://llm-proxy.forgeax.com/v1"
DEFAULT_MAX_TOKENS = 65536
DEFAULT_REASONING_EFFORT = "high"
REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh"}
)
CLAUDE_ADAPTIVE_THINKING_MODELS = frozenset({"claude-opus-4-8"})
CLAUDE_EFFORT_MAP = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "max",
}
EXPECTED_INPUT_SHA256 = (
    "4f632141aa5539e33a6e5f9b63c26f7d66a1882c48c07ba0fd5036af74fa892f"
)
RUNNER_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = RUNNER_ROOT / "data" / "sceneeval_human100_generation.jsonl"
LOCAL_PROFILE_PATH = RUNNER_ROOT / "local_openai_profile.yaml"
REDACTED_BEARER = "Bearer <runtime-secret-redacted>"


class LocalMultiModelError(RuntimeError):
    """Raised when the local suite configuration or provenance is invalid."""


class LocalRunConfig(frozen_runner.RunConfig):
    """RunConfig whose public representation contains no credential."""

    def public_dict(self) -> dict[str, Any]:
        if self.wire_model in CLAUDE_ADAPTIVE_THINKING_MODELS:
            request_fields = ["model", "messages", "max_tokens", "thinking", "stream"]
            if self.reasoning_effort != "none":
                request_fields.append("output_config")
            reasoning_interface = "anthropic_adaptive_thinking"
        else:
            request_fields = [
                "model",
                "messages",
                "max_tokens",
                "reasoning_effort",
                "stream",
            ]
            reasoning_interface = "openai_reasoning_effort"
        return {
            "endpoint": self.endpoint,
            "runner_version": __version__,
            "configured_client": "forgeax_openai_compatible",
            "transport_implementation": "auditable_direct_openai_compatible",
            "configured_model": self.configured_model,
            "wire_model": self.wire_model,
            "authentication": "runtime bearer token; never persisted",
            "request_fields": sorted(request_fields),
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
            "reasoning_interface": reasoning_interface,
            "stream": False,
            "omitted_request_fields": [
                "temperature",
                "top_p",
                "top_k",
                "repetition_penalty",
                "chat_template_kwargs",
            ],
            "provider_routing_headers": False,
            "message_roles": ["system", "user"],
            "system_message": True,
            "system_message_source": "exact frozen prompt_protocol.txt",
            "user_message": "exact SceneEval Description only",
            "scene_id_sent_to_model": False,
            "examples": False,
            "constrained_decoding": False,
            "automatic_retry": True,
            "max_retries": self.max_retries,
            "max_consecutive_infrastructure_failures": self.max_retries + 1,
            "short_output_retry_limit": "unbounded",
            "minimum_visible_output_tokens": (
                frozen_runner.MIN_VISIBLE_OUTPUT_TOKENS
            ),
            "visible_output_token_formula": (
                "usage.completion_tokens - reasoning_tokens when supplied; "
                "otherwise usage.completion_tokens"
            ),
            "retry_delay_seconds": self.retry_delay_seconds,
            "retry_attempt_statuses": sorted(
                frozen_runner.RETRYABLE_ATTEMPT_STATUSES | {"short_output"}
            ),
            "never_retry_attempt_statuses": [
                "captured",
                "token_count_unavailable",
                "transport_ambiguous",
            ],
            "model_content_validation": False,
            "reasoning_channel_normalization": False,
            "timeout_seconds": self.timeout_seconds,
        }


def _model_slug(model: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._-")
    if not slug:
        raise LocalMultiModelError(f"model cannot be represented as a path: {model!r}")
    return slug


def _make_config(
    *,
    model: str,
    base_url: str,
    timeout_seconds: float,
    max_retries: int,
    max_tokens: int,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> LocalRunConfig:
    if model not in MODEL_ORDER:
        raise LocalMultiModelError(f"model is outside the fixed suite order: {model}")
    if not base_url.startswith(("http://", "https://")):
        raise LocalMultiModelError("base URL must be HTTP(S)")
    if timeout_seconds <= 0:
        raise LocalMultiModelError("timeout must be positive")
    if isinstance(max_retries, bool) or max_retries < 0:
        raise LocalMultiModelError("max_retries must be a non-negative integer")
    if isinstance(max_tokens, bool) or max_tokens < 1:
        raise LocalMultiModelError("max_tokens must be a positive integer")
    if reasoning_effort not in REASONING_EFFORTS:
        raise LocalMultiModelError(
            "reasoning_effort must be one of: "
            + ", ".join(sorted(REASONING_EFFORTS))
        )
    return LocalRunConfig(
        endpoint=base_url.rstrip("/") + "/chat/completions",
        configured_model=f"openai/{model}",
        wire_model=model,
        api_key="<runtime-secret-redacted>",
        timeout_seconds=float(timeout_seconds),
        max_retries=int(max_retries),
        max_tokens=int(max_tokens),
        reasoning_effort=reasoning_effort,
    )


def _request_value(scene: SceneInput, config: LocalRunConfig) -> dict[str, Any]:
    """Build the minimal request while reusing both exact prompt messages."""

    value = {
        "model": config.wire_model,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(scene.description)},
        ],
        "max_tokens": config.max_tokens,
        "stream": False,
    }
    if config.wire_model in CLAUDE_ADAPTIVE_THINKING_MODELS:
        if config.reasoning_effort == "none":
            value["thinking"] = {"type": "disabled"}
        else:
            value["thinking"] = {"type": "adaptive"}
            value["output_config"] = {
                "effort": CLAUDE_EFFORT_MAP[config.reasoning_effort]
            }
    else:
        value["reasoning_effort"] = config.reasoning_effort
    return value


def _redacted_request_headers(
    config: LocalRunConfig,
    session_id: str,
) -> dict[str, str]:
    """Return safe artifact headers; the real bearer is injected later."""

    return {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
        "User-Agent": f"sceneeval-local-multimodel-runner/{__version__}",
        "Authorization": REDACTED_BEARER,
        # Required by the frozen interruption/resume provenance check. This is
        # a local attempt identifier and is not forwarded to ForgeAX.
        "SessionID": session_id,
    }


def _transport_with_runtime_secret(api_key: str) -> Callable[..., Any]:
    if not api_key:
        raise LocalMultiModelError("API key must be non-empty")

    def post_once(
        endpoint: str,
        request_body: bytes,
        *,
        connect_timeout: float,
        read_timeout: float,
        request_headers: dict[str, str] | None = None,
    ):
        artifact_headers = request_headers or {}
        if artifact_headers.get("Authorization") != REDACTED_BEARER:
            raise LocalMultiModelError(
                "refusing transport because artifact authorization is not redacted"
            )
        wire_headers = {
            "Content-Type": artifact_headers.get(
                "Content-Type", "application/json; charset=utf-8"
            ),
            "Accept": artifact_headers.get("Accept", "application/json"),
            "User-Agent": artifact_headers.get(
                "User-Agent", f"sceneeval-local-multimodel-runner/{__version__}"
            ),
            "Authorization": f"Bearer {api_key}",
        }
        return wire_post_once(
            endpoint,
            request_body,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            request_headers=wire_headers,
        )

    return post_once


def _visible_output_tokens(api_value: Any) -> tuple[int, int, int]:
    """Accept standard OpenAI usage when reasoning-token detail is absent."""

    if not isinstance(api_value, dict):
        raise frozen_runner.TokenCountUnavailable(
            "API response top-level value is not an object"
        )
    usage = api_value.get("usage")
    if not isinstance(usage, dict):
        raise frozen_runner.TokenCountUnavailable("usage must be an object")
    completion_tokens = usage.get("completion_tokens")
    if (
        isinstance(completion_tokens, bool)
        or not isinstance(completion_tokens, int)
        or completion_tokens < 0
    ):
        raise frozen_runner.TokenCountUnavailable(
            "usage.completion_tokens must be a non-negative integer"
        )
    reasoning_tokens = 0
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict) and details.get("reasoning_tokens") is not None:
        value = details["reasoning_tokens"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise frozen_runner.TokenCountUnavailable(
                "usage.completion_tokens_details.reasoning_tokens must be a "
                "non-negative integer when supplied"
            )
        reasoning_tokens = value
    if reasoning_tokens > completion_tokens:
        raise frozen_runner.TokenCountUnavailable(
            "reasoning_tokens cannot exceed completion_tokens"
        )
    return (
        completion_tokens - reasoning_tokens,
        completion_tokens,
        reasoning_tokens,
    )


_ORIGINAL_MANIFEST = frozen_runner._manifest


def _manifest(
    batch: InputBatch,
    config: LocalRunConfig,
    runner_source: dict[str, Any],
) -> dict[str, Any]:
    value = _ORIGINAL_MANIFEST(batch, config, runner_source)
    value["scope"] = (
        "SceneEval human-authored IDs 0-99; local ordered multi-model generation"
    )
    value["client_config"]["source"] = "local_openai_profile.yaml"
    value["protocol"]["system_message_is_frozen_protocol"] = True
    value["local_openai_compatibility"] = {
        "real_api_key_persisted": False,
        "authorization_artifact": REDACTED_BEARER,
        "provider_routing_headers_sent": False,
        "reasoning_tokens_optional": True,
        "adapter_source": "sceneeval_local_multimodel.py",
    }
    return value


@contextmanager
def _installed_adapter(api_key: str) -> Iterator[None]:
    names = {
        "_request_value": _request_value,
        "_request_headers": _redacted_request_headers,
        "_visible_output_tokens": _visible_output_tokens,
        "post_once": _transport_with_runtime_secret(api_key),
        "_manifest": _manifest,
        "CLIENT_CONFIG_PATH": LOCAL_PROFILE_PATH,
    }
    previous = {name: getattr(frozen_runner, name) for name in names}
    try:
        for name, value in names.items():
            setattr(frozen_runner, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(frozen_runner, name, value)


def _suite_manifest(
    batch: InputBatch,
    *,
    base_url: str,
    reasoning_effort: str,
) -> dict[str, Any]:
    adapter_bytes = Path(__file__).read_bytes()
    launcher_path = RUNNER_ROOT / "run_local_multimodel.py"
    return {
        "suite_manifest_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_order": list(MODEL_ORDER),
        "base_url": base_url.rstrip("/"),
        "reasoning_effort": reasoning_effort,
        "authentication": "runtime bearer token; never persisted",
        "input": {
            "row_count": len(batch.scenes),
            "ids": [batch.scenes[0].scene_id, batch.scenes[-1].scene_id],
            "sha256": batch.sha256,
        },
        "prompt_protocol_sha256": sha256_bytes(
            (protocol_text() + "\n").encode("utf-8")
        ),
        "adapter_sha256": sha256_bytes(adapter_bytes),
        "launcher_sha256": sha256_bytes(launcher_path.read_bytes()),
    }


def _verify_suite_manifest(
    output_root: Path,
    batch: InputBatch,
    *,
    base_url: str,
    reasoning_effort: str,
) -> None:
    path = output_root / "suite-manifest.json"
    try:
        existing = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalMultiModelError(f"cannot read suite manifest: {exc}") from exc
    current = _suite_manifest(
        batch,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
    )
    for key in (
        "suite_manifest_version",
        "model_order",
        "base_url",
        "reasoning_effort",
        "authentication",
        "input",
        "prompt_protocol_sha256",
        "adapter_sha256",
        "launcher_sha256",
    ):
        if existing.get(key) != current.get(key):
            raise LocalMultiModelError(f"suite resume provenance differs: {key}")


def validate_exact_input(input_jsonl: str | Path = DEFAULT_INPUT_PATH) -> InputBatch:
    batch = load_human100_jsonl(Path(input_jsonl).expanduser().resolve())
    if batch.sha256 != EXPECTED_INPUT_SHA256:
        raise LocalMultiModelError(
            "SceneEval-100 input bytes differ from the approved snapshot; "
            f"expected {EXPECTED_INPUT_SHA256}, received {batch.sha256}"
        )
    return batch


def run_ordered_suite(
    *,
    api_key: str,
    output_root: str | Path,
    input_jsonl: str | Path = DEFAULT_INPUT_PATH,
    base_url: str = DEFAULT_BASE_URL,
    timeout_seconds: float = 1800.0,
    max_retries: int = 2,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    resume: bool = False,
) -> dict[str, Any]:
    """Run all configured models serially over the exact SceneEval-100 snapshot."""

    batch = validate_exact_input(input_jsonl)
    root = Path(output_root).expanduser().resolve()
    if resume:
        if not root.is_dir():
            raise LocalMultiModelError(f"suite output does not exist: {root}")
        _verify_suite_manifest(
            root,
            batch,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
        )
    else:
        try:
            root.mkdir(parents=True, mode=0o750, exist_ok=False)
        except FileExistsError as exc:
            raise LocalMultiModelError(
                f"refusing to overwrite existing suite output: {root}"
            ) from exc
        write_json_exclusive(
            root / "suite-manifest.json",
            _suite_manifest(
                batch,
                base_url=base_url,
                reasoning_effort=reasoning_effort,
            ),
        )

    results: dict[str, Any] = {}
    with _installed_adapter(api_key):
        for model in MODEL_ORDER:
            config = _make_config(
                model=model,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )
            model_root = root / _model_slug(model)
            model_resume = resume and model_root.exists()
            summary, stopped = frozen_runner.run_batch(
                batch,
                model_root,
                config,
                resume=model_resume,
            )
            results[model] = {
                "output_root": str(model_root),
                "summary": summary,
                "stopped": stopped,
            }

    summary_root = root / "suite-execution-summaries"
    summary_root.mkdir(mode=0o750, exist_ok=True)
    summary_name = (
        datetime.now(timezone.utc).strftime("summary_%Y%m%dT%H%M%S_%fZ_")
        + uuid.uuid4().hex
        + ".json"
    )
    write_json_exclusive(
        summary_root / summary_name,
        {
            "model_order": list(MODEL_ORDER),
            "results": results,
            "resume": bool(resume),
        },
    )
    return results


def run_smoke_probe(
    *,
    api_key: str,
    output_root: str | Path,
    scene_id: int = 0,
    input_jsonl: str | Path = DEFAULT_INPUT_PATH,
    base_url: str = DEFAULT_BASE_URL,
    timeout_seconds: float = 1800.0,
    max_retries: int = 2,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> dict[str, Any]:
    """Run one exact SceneEval scene through each model in fixed order."""

    batch = validate_exact_input(input_jsonl)
    if isinstance(scene_id, bool) or not isinstance(scene_id, int):
        raise LocalMultiModelError("scene_id must be an integer")
    scene = next((item for item in batch.scenes if item.scene_id == scene_id), None)
    if scene is None:
        raise LocalMultiModelError(f"scene_id must be between 0 and 99: {scene_id}")

    root = Path(output_root).expanduser().resolve()
    try:
        root.mkdir(parents=True, mode=0o750, exist_ok=False)
    except FileExistsError as exc:
        raise LocalMultiModelError(
            f"refusing to overwrite existing smoke-probe output: {root}"
        ) from exc

    manifest = _suite_manifest(
        batch,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
    )
    manifest.update(
        {
            "suite_type": "single_scene_smoke_probe",
            "scene_id": scene.scene_id,
            "timeout_seconds": timeout_seconds,
            "max_retries": max_retries,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
        }
    )
    write_json_exclusive(root / "smoke-probe-manifest.json", manifest)

    results: dict[str, Any] = {}
    with _installed_adapter(api_key):
        for model in MODEL_ORDER:
            config = _make_config(
                model=model,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )
            model_root = root / _model_slug(model)
            # Reuse the frozen initializer so every probe still snapshots the
            # exact 100-row source input, prompt, schema, runner sources, and
            # credential-free local client profile.
            frozen_runner.initialize_run(model_root, batch, config)
            result = frozen_runner.run_scene(scene, model_root, config)
            result_value = {
                "scene_id": result.scene_id,
                "status": result.status,
                "attempt_count": result.attempt_count,
                "stop_batch": result.stop_batch,
                "output_root": str(model_root),
            }
            write_json_exclusive(model_root / "smoke-probe-result.json", result_value)
            results[model] = result_value

    write_json_exclusive(
        root / "smoke-probe-summary.json",
        {
            "model_order": list(MODEL_ORDER),
            "scene_id": scene.scene_id,
            "results": results,
        },
    )
    return results


def request_preview(model: str, description: str) -> bytes:
    """Return canonical request bytes for tests and offline review only."""

    config = _make_config(
        model=model,
        base_url=DEFAULT_BASE_URL,
        timeout_seconds=1800.0,
        max_retries=2,
        max_tokens=DEFAULT_MAX_TOKENS,
        reasoning_effort=DEFAULT_REASONING_EFFORT,
    )
    return request_json_bytes(
        _request_value(SceneInput(scene_id=0, description=description), config)
    )
