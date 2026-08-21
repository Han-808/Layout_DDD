from __future__ import annotations

import ast
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import threading
from typing import Any

import pytest

from benchmark.scene_generation.frozen_two_stage.orchestrator import (
    FrozenTwoStageOrchestrator,
)
from benchmark.scene_generation.frozen_two_stage.cli import check_config
from benchmark.scene_generation.frozen_two_stage.config import load_run_config
from benchmark.scene_generation.frozen_two_stage.providers import (
    ChatOptionPolicy,
    make_api2_chat_route,
    make_api2_responses_route,
    make_api3_chat_route,
)
from benchmark.scene_generation.frozen_two_stage.retry_policy import RetryPolicy
from benchmark.scene_generation.frozen_two_stage.spec import GenerationRunSpec
from benchmark.scene_generation.frozen_two_stage.trust import TrustError
from tools.api3_anthropic_runner_v2 import generation_runner as shared_core


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _model(**overrides: Any) -> SimpleNamespace:
    values = {
        "key": "test-model",
        "label": "Test Model",
        "endpoint": "http://127.0.0.1:1/v1/chat/completions",
        "api_key": "app-id:app-key?ignored=legacy-query",
        "configured_model": "openai/test-model",
        "wire_model": "test-model",
        "timeout_seconds": 1.0,
        "max_infrastructure_retries": 1,
        "retry_delay_seconds": 0.0,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "max_tokens": 65536,
        "repetition_penalty": None,
        "reasoning_effort": "max",
        "preserved_thinking": None,
        "strategy_type": "ConsistentHash",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_frozen_provider_request_bytes_match_expected_envelopes() -> None:
    user_value = {"z": 3, "a": {"value": 1}}
    user_json = _canonical_json_bytes(user_value).decode("utf-8")

    kimi_model = _model(wire_model="kimi-k3", reasoning_effort="max")
    kimi = make_api2_chat_route(
        provider="moonshot",
        gateway_model="kimi-k3",
        user_agent_suffix="api2-kimi-k3-v1",
        option_policy=ChatOptionPolicy.top_level_reasoning(),
    )
    kimi_expected = {
        "model": "kimi-k3",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": user_json},
        ],
        "reasoning_effort": "max",
        "max_tokens": 65536,
        "stream": False,
    }
    assert _canonical_json_bytes(
        kimi.request_value(
            model=kimi_model,
            system_prompt="system",
            user_value=user_value,
            canonical_json_bytes=_canonical_json_bytes,
        )
    ) == _canonical_json_bytes(kimi_expected)

    glm_model = _model(wire_model="glm-5.3", reasoning_effort="max")
    glm = make_api2_responses_route(
        provider="zhipu",
        gateway_model="glm-5.3",
        user_agent_suffix="api2-glm53-v1",
    )
    glm_expected = {
        "model": "glm-5.3",
        "instructions": "system",
        "input": [{"role": "user", "content": user_json}],
        "reasoning": {"effort": "max"},
        "max_output_tokens": 65536,
        "text": {"format": {"type": "json_object"}},
        "store": False,
        "stream": False,
    }
    assert _canonical_json_bytes(
        glm.request_value(
            model=glm_model,
            system_prompt="system",
            user_value=user_value,
            canonical_json_bytes=_canonical_json_bytes,
        )
    ) == _canonical_json_bytes(glm_expected)

    opus_model = _model(
        wire_model="claude-opus-4-8-aihub",
        api_key="api3-key",
        reasoning_effort="high",
        preserved_thinking=True,
    )
    opus = make_api3_chat_route(
        option_policy=ChatOptionPolicy.adaptive_thinking(reasoning_effort="high")
    )
    opus_expected = {
        "model": "claude-opus-4-8-aihub",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": user_json},
        ],
        "max_tokens": 65536,
        "stream": False,
        "thinking": {"type": "adaptive"},
        "reasoning_effort": "high",
    }
    assert _canonical_json_bytes(
        opus.request_value(
            model=opus_model,
            system_prompt="system",
            user_value=user_value,
            canonical_json_bytes=_canonical_json_bytes,
        )
    ) == _canonical_json_bytes(opus_expected)


def test_chat_and_responses_extractors_preserve_frozen_shapes() -> None:
    chat = make_api3_chat_route(option_policy=ChatOptionPolicy.legacy_core())
    chat_body = _canonical_json_bytes(
        {
            "choices": [
                {
                    "message": {
                        "content": "answer",
                        "reasoning": "reasoning",
                        "reasoning_content": "provider reasoning",
                    }
                }
            ],
            "usage": {"completion_tokens": 7},
        }
    )
    chat_message = chat.extract_api_message(chat_body)
    assert chat_message.as_legacy_tuple() == (
        b"answer",
        b"reasoning",
        b"provider reasoning",
        {"completion_tokens": 7},
    )
    with pytest.raises(ValueError, match="exactly one choice"):
        chat.extract_api_message(b'{"choices":[]}')

    responses = make_api2_responses_route(
        provider="zhipu",
        gateway_model="glm-5.3",
        user_agent_suffix="api2-glm53-v1",
    )
    responses_body = _canonical_json_bytes(
        {
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "content": [
                        {"type": "reasoning_text", "text": "first"},
                        {"type": "reasoning_text", "text": "second"},
                    ],
                },
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "output"},
                    ],
                },
            ],
            "usage": {"output_tokens": 9},
        }
    )
    responses_message = responses.extract_api_message(responses_body)
    assert responses_message.as_legacy_tuple() == (
        b"output",
        b"first\nsecond",
        None,
        {"output_tokens": 9},
    )
    with pytest.raises(ValueError, match="not complete: max_output_tokens"):
        responses.extract_api_message(
            _canonical_json_bytes(
                {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                }
            )
        )


def test_api2_gateway_headers_match_frozen_cache_and_auth_policy() -> None:
    route = make_api2_chat_route(
        provider="moonshot",
        gateway_model="kimi-k3",
        user_agent_suffix="api2-kimi-k3-v1",
        option_policy=ChatOptionPolicy.top_level_reasoning(),
        clock=lambda: 123.5,
    )
    expected_cache_id = hashlib.md5(b"123.5app-idsession-7").hexdigest()
    headers = route.request_headers(_model(), "session-7")
    assert headers == {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
        "User-Agent": "hy34-two-stage-generator/2.0.0 api2-kimi-k3-v1",
        "Authorization": (
            "Bearer app-id:app-key?provider=moonshot&model=kimi-k3&timeout=600"
            f"&cache_task_id={expected_cache_id}"
        ),
    }
    assert "app-key" not in json.dumps(route.public_dict())


def test_api3_gateway_headers_match_frozen_session_policy() -> None:
    route = make_api3_chat_route(option_policy=ChatOptionPolicy.legacy_core())
    headers = route.request_headers(
        _model(api_key="api3-key", strategy_type="ConsistentHash"),
        "session-8",
    )
    assert headers == {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
        "User-Agent": "hy34-two-stage-generator/2.0.0",
        "Authorization": "Bearer api3-key",
        "SessionID": "session-8",
        "StrategyType": "ConsistentHash",
    }


def test_two_provider_instances_do_not_mutate_shared_core_globals() -> None:
    frozen_hooks = (
        shared_core._request_value,
        shared_core._request_headers,
        shared_core._extract_api_message,
    )
    kimi = make_api2_chat_route(
        provider="moonshot",
        gateway_model="kimi-k3",
        user_agent_suffix="api2-kimi-k3-v1",
        option_policy=ChatOptionPolicy.top_level_reasoning(),
    )
    glm = make_api2_responses_route(
        provider="zhipu",
        gateway_model="glm-5.3",
        user_agent_suffix="api2-glm53-v1",
    )
    kimi_value = kimi.request_value(
        model=_model(wire_model="kimi-k3"),
        system_prompt="system",
        user_value={"case": 1},
        canonical_json_bytes=_canonical_json_bytes,
    )
    glm_value = glm.request_value(
        model=_model(wire_model="glm-5.3"),
        system_prompt="system",
        user_value={"case": 2},
        canonical_json_bytes=_canonical_json_bytes,
    )
    assert "messages" in kimi_value and "instructions" not in kimi_value
    assert "instructions" in glm_value and "messages" not in glm_value
    assert (
        shared_core._request_value,
        shared_core._request_headers,
        shared_core._extract_api_message,
    ) == frozen_hooks


def test_api3_legacy_route_is_model_agnostic_and_matches_shared_core() -> None:
    model = shared_core.ModelConfig(
        **vars(
            _model(
                wire_model="any-api3-model-alias",
                api_key="api3-key",
                temperature=0.4,
                top_p=0.9,
                top_k=12,
                repetition_penalty=1.05,
                reasoning_effort="high",
                preserved_thinking=True,
            )
        )
    )
    route = make_api3_chat_route(
        option_policy=ChatOptionPolicy.legacy_core(),
        route_key="api3-shared-chat-v1",
    )
    user_value = {"brief": {"id": 1}}
    expected = shared_core._request_value(
        model=model,
        system_prompt="system",
        user_value=user_value,
    )
    actual = route.request_value(
        model=model,
        system_prompt="system",
        user_value=user_value,
        canonical_json_bytes=shared_core.canonical_json_bytes,
    )
    assert shared_core.canonical_json_bytes(actual) == (
        shared_core.canonical_json_bytes(expected)
    )
    assert route.request_headers(model, "session") == shared_core._request_headers(
        model, "session"
    )


def _spec(
    output_root: Path,
    retry_policy: RetryPolicy,
    *,
    brief_ids: tuple[str, ...] = ("brief_00", "brief_01"),
) -> GenerationRunSpec:
    return GenerationRunSpec(
        provider_key="fake-route",
        model_key="test-model",
        wire_model="test-model",
        ordered_brief_ids=brief_ids,
        briefs_path=output_root.parent / "briefs.json",
        models_path=output_root.parent / "models.json",
        output_root=output_root,
        retry_policy=retry_policy,
        execution_policy={"schema_version": "test_execution_policy_v1"},
        generation_parameters={"reasoning_effort": "high"},
    )


def test_retry_policy_and_run_spec_reject_secret_or_raw_content_fields(
    tmp_path: Path,
) -> None:
    retry_policy = RetryPolicy(
        max_infrastructure_retries=2,
        retry_delay_seconds=0,
        continue_after_case_failure=True,
    )
    assert retry_policy.maximum_attempts_per_stage == 3
    assert retry_policy.to_public_dict()["semantic_retry_count"] == 0
    with pytest.raises(ValueError, match="transport_ambiguous membership"):
        RetryPolicy(
            max_infrastructure_retries=1,
            retryable_transport_statuses=frozenset(
                {"transport_failure", "transport_ambiguous"}
            ),
            retry_ambiguous_timeouts=False,
        )
    with pytest.raises(ValueError, match="unsupported retryable transport"):
        RetryPolicy(
            max_infrastructure_retries=1,
            retryable_transport_statuses=frozenset(
                {"transport_failure", "transport_maybe"}
            ),
        )
    with pytest.raises(ValueError, match="forbidden credential key"):
        GenerationRunSpec(
            provider_key="route",
            model_key="model",
            wire_model="wire",
            ordered_brief_ids=("brief_00",),
            briefs_path=tmp_path / "briefs.json",
            models_path=tmp_path / "models.json",
            output_root=tmp_path / "output-secret",
            retry_policy=retry_policy,
            execution_policy={"schema_version": "policy_v1"},
            generation_parameters={"nested": {"api_key": "do-not-record"}},
        )
    with pytest.raises(ValueError, match="forbidden credential key"):
        GenerationRunSpec(
            provider_key="route",
            model_key="model",
            wire_model="wire",
            ordered_brief_ids=("brief_00",),
            briefs_path=tmp_path / "briefs.json",
            models_path=tmp_path / "models.json",
            output_root=tmp_path / "output-env-name",
            retry_policy=retry_policy,
            execution_policy={"schema_version": "policy_v1"},
            generation_parameters={"api_key_env": "SENSITIVE_ENV_NAME"},
        )
    with pytest.raises(ValueError, match="forbidden raw-content key"):
        GenerationRunSpec(
            provider_key="route",
            model_key="model",
            wire_model="wire",
            ordered_brief_ids=("brief_00",),
            briefs_path=tmp_path / "briefs.json",
            models_path=tmp_path / "models.json",
            output_root=tmp_path / "output-prompt",
            retry_policy=retry_policy,
            execution_policy={"schema_version": "policy_v1"},
            generation_parameters={"system_prompt": "raw prompt"},
        )


class _FakeCore:
    def __init__(self, root: Path, statuses: list[str]) -> None:
        self.DEFAULT_STAGE_A_PROMPT = root / "stage_a.txt"
        self.DEFAULT_STAGE_C_PROMPT = root / "stage_c.txt"
        self.DEFAULT_STAGE_A_PROMPT.write_text("stage A", encoding="utf-8")
        self.DEFAULT_STAGE_C_PROMPT.write_text("stage C", encoding="utf-8")
        self.statuses = iter(statuses)
        self.calls: list[str] = []

    def write_json_exclusive(self, path: Path, value: Any) -> None:
        self.calls.append(f"write:{path.name}")
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")

    def initialize_run(self, **kwargs: Any) -> None:
        output_root = kwargs["output_root"]
        self.calls.append("initialize")
        output_root.mkdir(parents=True, exist_ok=False)
        self.write_json_exclusive(
            output_root / "run_manifest.json",
            {"schema_version": "fake_manifest_v1"},
        )

    def run_case(self, **kwargs: Any) -> dict[str, Any]:
        brief_id = kwargs["brief"]["brief_id"]
        self.calls.append(f"case:{brief_id}")
        assert kwargs["stage_a_prompt"] == "stage A"
        assert kwargs["stage_c_prompt"] == "stage C"
        assert kwargs["provider_route"].key == "fake-route"
        assert isinstance(kwargs["retry_policy"], RetryPolicy)
        status = next(self.statuses)
        return {
            "brief_id": brief_id,
            "status": status,
            "stop_batch": status != "complete",
            "eligible_for_strict_one_shot_evaluation": status == "complete",
        }

    @staticmethod
    def utc_now() -> str:
        return "2026-08-21T00:00:00Z"


@pytest.mark.parametrize(
    ("continue_after_failure", "expected_processed", "expected_stopped"),
    [(False, 1, True), (True, 2, False)],
)
def test_orchestrator_preserves_order_summary_and_continue_policy(
    tmp_path: Path,
    continue_after_failure: bool,
    expected_processed: int,
    expected_stopped: bool,
) -> None:
    root = tmp_path / f"run-{continue_after_failure}"
    core = _FakeCore(tmp_path, ["stage_a_failed", "complete"])
    retry_policy = RetryPolicy(
        max_infrastructure_retries=1,
        retry_delay_seconds=0,
        continue_after_case_failure=continue_after_failure,
    )
    route = SimpleNamespace(key="fake-route")
    orchestrator = FrozenTwoStageOrchestrator(core, route)
    events: list[dict[str, Any]] = []
    summary, stopped = orchestrator.run(
        spec=_spec(root, retry_policy),
        model=_model(),
        briefs=[{"brief_id": "brief_00"}, {"brief_id": "brief_01"}],
        retriever=object(),
        progress=lambda event: events.append(dict(event)),
    )
    assert stopped is expected_stopped
    assert summary["processed_briefs"] == expected_processed
    assert summary["failed"] == 1
    assert summary["complete"] == expected_processed - 1
    assert summary["stopped_early"] is expected_stopped
    expected_case_calls = ["case:brief_00"]
    if continue_after_failure:
        expected_case_calls.append("case:brief_01")
    assert [item for item in core.calls if item.startswith("case:")] == expected_case_calls
    assert core.calls.index("initialize") < core.calls.index("case:brief_00")
    assert core.calls[-1] == "write:summary.json"
    assert json.loads((root / "summary.json").read_text()) == summary
    first_terminal = next(
        event
        for event in events
        if event["event"] == "case_terminal"
        and event["brief_id"] == "brief_00"
    )
    assert first_terminal["continued_after_case"] is continue_after_failure


def _imported_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names.append(module)
            names.extend(
                f"{module}.{alias.name}" if module else alias.name
                for alias in node.names
            )
    return names


def test_evaluation_modules_do_not_import_frozen_generation_layer() -> None:
    paths = [
        Path("scripts/run_camera_cal_scene_level.py"),
        Path("src/benchmark/api/evaluation.py"),
        *Path("src/benchmark/evaluator").rglob("*.py"),
        *Path("src/benchmark/visual_judge").rglob("*.py"),
    ]
    assert paths
    violations: list[str] = []
    for path in paths:
        for imported in _imported_names(path):
            if "frozen_two_stage" in imported:
                violations.append(f"{path}:{imported}")
    assert not violations, "evaluation imported generation providers: " + ", ".join(
        violations
    )


def test_checked_in_run_configs_are_static_credential_free_and_multi_model_ready() -> None:
    config_paths = (
        Path("configs/generation/api2_kimi_k3_scene10_v2.json"),
        Path("configs/generation/api2_glm53_scene10_v2.json"),
        Path("configs/generation/api3_opus48_high_scene10_v2.json"),
    )
    reports = [check_config(load_run_config(path)) for path in config_paths]
    assert [report["model_key"] for report in reports] == [
        "api2-kimi-k3",
        "api2-glm-5-3",
        "api3-claude-opus-4-8-high",
    ]
    assert all(report["credential_loaded"] is False for report in reports)
    assert all(report["retriever_loaded"] is False for report in reports)
    assert all(report["network_used"] is False for report in reports)
    assert {
        report["retrieval_profile_id"] for report in reports
    } == {"imaginarium-qwen3-embedding-0.6b-stable-top1-v2"}

    from tools.api2_glm53_runner_v1 import glm53_generation_runner as glm_adapter
    from tools.api2_kimi_k3_runner_v1 import (
        kimi_k3_generation_runner as kimi_adapter,
    )
    from tools.api3_opus48_thinking_runner_v1 import (
        generation_runner as opus_adapter,
    )

    configured_routes = [
        load_run_config(path).route.build_route().public_dict()
        for path in config_paths
    ]
    adapter_routes = [
        kimi_adapter.provider_route().public_dict(),
        glm_adapter.provider_route().public_dict(),
        opus_adapter.provider_route().public_dict(),
    ]
    assert configured_routes == adapter_routes


def test_static_check_does_not_import_core_or_initialize_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmark.scene_generation.frozen_two_stage import cli as configured_cli

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("static check executed configurable runtime code")

    monkeypatch.setattr(configured_cli, "load_frozen_core", forbidden)
    monkeypatch.setattr(configured_cli, "load_runtime_inputs", forbidden)
    report = configured_cli.check_config(
        load_run_config(
            "configs/generation/api2_kimi_k3_scene10_v2.json"
        )
    )
    assert report["credential_loaded"] is False
    assert report["retriever_loaded"] is False
    assert report["network_used"] is False
    assert report["preflight_contract"] == "post_preflight_only"


def test_cli_import_does_not_preimport_untrusted_retrieval_package(
    tmp_path: Path,
) -> None:
    """Process startup must reach the trust gate before retrieval import."""

    sentinel = tmp_path / "retrieval-imported"
    script = "\n".join(
        (
            "import importlib.abc",
            "import pathlib",
            "import sys",
            f"sentinel = pathlib.Path({str(sentinel)!r})",
            "class BlockRetrieval(importlib.abc.MetaPathFinder):",
            "    def find_spec(self, fullname, path=None, target=None):",
            "        if fullname == 'benchmark.scene_generation.retrieval' or "
            "fullname.startswith('benchmark.scene_generation.retrieval.'):",
            "            sentinel.write_text(fullname, encoding='utf-8')",
            "            raise RuntimeError('retrieval imported before trust')",
            "        return None",
            "sys.meta_path.insert(0, BlockRetrieval())",
            "import benchmark.scene_generation.frozen_two_stage.cli",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert not sentinel.exists()


def test_shared_runtime_source_mutation_fails_run_before_import_or_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmark.scene_generation.frozen_two_stage import cli as configured_cli

    repo = tmp_path / "repo"
    copies = {
        "tools/api3_anthropic_runner_v2": Path(
            "tools/api3_anthropic_runner_v2"
        ),
        "tools/api3_opus48_thinking_runner_v1": Path(
            "tools/api3_opus48_thinking_runner_v1"
        ),
        "src/benchmark/scene_generation/retrieval": Path(
            "src/benchmark/scene_generation/retrieval"
        ),
        "configs/retrieval": Path("configs/retrieval"),
    }
    for relative, source in copies.items():
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative == "tools/api3_opus48_thinking_runner_v1":
            destination.mkdir()
            shutil.copy2(source / "models.pod.json", destination / "models.pod.json")
        else:
            shutil.copytree(
                source,
                destination,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
    generation = repo / "configs" / "generation"
    generation.mkdir(parents=True)
    shutil.copy2(
        "configs/generation/api3_opus48_high_scene10_v2.json",
        generation / "api3_opus48_high_scene10_v2.json",
    )

    def bundle(bundle_id: str, role: str, root_text: str) -> dict[str, Any]:
        root = repo / root_text
        return {
            "bundle_id": bundle_id,
            "role": role,
            "root": root_text,
            "files": [
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in sorted(root.rglob("*"))
                if path.is_file()
                and path.suffix != ".pyc"
                and "__pycache__" not in path.parts
            ],
        }

    trust_path = repo / "configs" / "runners" / "test_trust.json"
    trust_path.parent.mkdir()
    trust_path.write_text(
        json.dumps(
            {
                "schema_version": "active_generation_bundles_v1",
                "bundles": [
                    bundle(
                        "core",
                        "frozen_generation_core",
                        "tools/api3_anthropic_runner_v2",
                    ),
                    bundle(
                        "model",
                        "generation_model_config",
                        "tools/api3_opus48_thinking_runner_v1",
                    ),
                    bundle(
                        "runtime",
                        "shared_generation_retrieval_runtime",
                        "src/benchmark/scene_generation/retrieval",
                    ),
                    bundle(
                        "profiles",
                        "generation_retrieval_profiles",
                        "configs/retrieval",
                    ),
                    bundle(
                        "run-config",
                        "config_only_generation_routes",
                        "configs/generation",
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )
    config = load_run_config(
        generation / "api3_opus48_high_scene10_v2.json"
    )
    runtime_source = repo / "src/benchmark/scene_generation/retrieval/runtime.py"
    runtime_source.write_text(
        runtime_source.read_text(encoding="utf-8") + "\n# untrusted mutation\n",
        encoding="utf-8",
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("runtime import/credential path was reached")

    monkeypatch.setattr(configured_cli, "load_frozen_core", forbidden)
    monkeypatch.setattr(configured_cli, "load_runtime_inputs", forbidden)
    with pytest.raises(TrustError, match="trusted hash mismatch"):
        configured_cli.run_config(
            config,
            output_root=tmp_path / "must-not-exist",
            trust_manifest=trust_path,
        )


def test_legacy_v1_run_config_normalizes_to_current_retrieval_profile() -> None:
    loaded = load_run_config("configs/generation/api2_kimi_k3_scene10_v1.json")
    assert loaded.source_schema_version == "frozen_two_stage_run_config_v1"
    assert loaded.to_public_dict()["schema_version"] == "frozen_two_stage_run_config_v2"
    assert loaded.retrieval_profile_id == (
        "imaginarium-qwen3-embedding-0.6b-stable-top1-v2"
    )


def test_runtime_resources_are_gated_before_credential_loading(tmp_path: Path) -> None:
    from benchmark.scene_generation.frozen_two_stage.compatibility.loader import (
        load_runtime_inputs,
    )

    events: list[str] = []

    class Core:
        @staticmethod
        def _load_briefs(_path: Path) -> list[dict[str, str]]:
            return [{"brief_id": "brief_00"}]

        class RetrieverAdapter:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                events.append("retrieval_gate")

        @staticmethod
        def _load_model_config(_path: Path, _key: str) -> object:
            events.append("credential")
            return object()

    load_runtime_inputs(
        Core,
        models_path=tmp_path / "models.json",
        model_key="model",
        briefs_path=tmp_path / "briefs.json",
        ordered_brief_ids=("brief_00",),
        retriever_root=tmp_path,
        retrieval_catalog_path=tmp_path / "profiles.json",
        retrieval_profile_id="profile-v2",
        resource_bindings_path=tmp_path / "bindings.json",
    )
    assert events == ["retrieval_gate", "credential"]


def test_runtime_retrieval_identity_drift_fails_before_credential_loading(
    tmp_path: Path,
) -> None:
    from benchmark.scene_generation.frozen_two_stage.compatibility.loader import (
        load_runtime_inputs,
    )

    events: list[str] = []

    class Core:
        @staticmethod
        def _load_briefs(_path: Path) -> list[dict[str, str]]:
            return [{"brief_id": "brief_00"}]

        class RetrieverAdapter:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                events.append("retrieval_gate")
                self.public_provenance = {
                    "retrieval_profile_id": "profile-v2",
                    "catalog_sha256": "changed-after-static-trust",
                    "runtime_source_sha256": "runtime-v2",
                }

        @staticmethod
        def _load_model_config(_path: Path, _key: str) -> object:
            events.append("credential")
            return object()

    with pytest.raises(ValueError, match="catalog_sha256"):
        load_runtime_inputs(
            Core,
            models_path=tmp_path / "models.json",
            model_key="model",
            briefs_path=tmp_path / "briefs.json",
            ordered_brief_ids=("brief_00",),
            retriever_root=tmp_path,
            retrieval_catalog_path=tmp_path / "profiles.json",
            retrieval_profile_id="profile-v2",
            resource_bindings_path=tmp_path / "bindings.json",
            expected_retrieval_identity={
                "retrieval_profile_id": "profile-v2",
                "catalog_sha256": "trusted-catalog",
                "runtime_source_sha256": "runtime-v2",
            },
        )
    assert events == ["retrieval_gate"]


def test_resume_requires_v3_manifest_and_exact_retrieval_identity(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "run"
    output_root.mkdir()
    briefs = tmp_path / "briefs.json"
    models = tmp_path / "models.json"
    briefs.write_text("briefs", encoding="utf-8")
    models.write_text("models", encoding="utf-8")
    retrieval_identity = {
        "schema_version": "generation_retrieval_provenance_v2",
        "retrieval_profile_id": "profile-v2",
        "catalog_sha256": "catalog",
        "profile_sha256": "profile",
        "runtime_source_sha256": "runtime",
    }
    source_manifest = {"manifest_sha256": "source-manifest"}
    model = SimpleNamespace(key="model")
    retriever = SimpleNamespace(public_provenance=retrieval_identity)
    manifest = {
        "schema_version": shared_core.RUN_MANIFEST_SCHEMA_VERSION,
        "runner_version": shared_core.RUNNER_VERSION,
        "model": {"key": "model"},
        "briefs_sha256": hashlib.sha256(briefs.read_bytes()).hexdigest(),
        "models_config_sha256": hashlib.sha256(models.read_bytes()).hexdigest(),
        "stage_a_prompt_sha256": hashlib.sha256(
            shared_core.DEFAULT_STAGE_A_PROMPT.read_bytes()
        ).hexdigest(),
        "stage_c_prompt_sha256": hashlib.sha256(
            shared_core.DEFAULT_STAGE_C_PROMPT.read_bytes()
        ).hexdigest(),
        "source_manifest": source_manifest,
        "retrieval": retrieval_identity,
    }
    manifest_path = output_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    shared_core.verify_resume(
        output_root=output_root,
        model=model,
        briefs_path=briefs,
        models_path=models,
        retriever=retriever,
        source_manifest=source_manifest,
    )

    old = dict(manifest)
    old["schema_version"] = "hy34_two_stage_run_manifest_v2"
    old.pop("retrieval")
    manifest_path.write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(shared_core.ArtifactError, match="strict retrieval identity"):
        shared_core.verify_resume(
            output_root=output_root,
            model=model,
            briefs_path=briefs,
            models_path=models,
            retriever=retriever,
            source_manifest=source_manifest,
        )

    missing = dict(manifest)
    missing.pop("retrieval")
    manifest_path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(shared_core.ArtifactError, match="provenance mismatch"):
        shared_core.verify_resume(
            output_root=output_root,
            model=model,
            briefs_path=briefs,
            models_path=models,
            retriever=retriever,
            source_manifest=source_manifest,
        )


def test_config_only_onboarding_does_not_use_a_model_name_allowlist(
    tmp_path: Path,
) -> None:
    model_config = {
        "schema_version": "hy34_model_transport_config_v1",
        "endpoint": "https://generation.example.invalid/v1/responses",
        "api_key_env": "FUTURE_MODEL_API_KEY",
        "request": {
            "timeout_seconds": 600,
            "max_infrastructure_retries": 2,
            "retry_delay_seconds": 3,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "max_tokens": 65536,
            "repetition_penalty": None,
            "reasoning_effort": "max",
            "preserved_thinking": None,
            "strategy_type": "responses_api2_ali",
        },
        "models": {
            "future-qwen-alias": {
                "label": "Future Qwen Alias",
                "configured_model": "ali/future-qwen-alias",
                "wire_model": "future-qwen-alias",
            }
        },
    }
    models_path = tmp_path / "models.json"
    models_path.write_text(json.dumps(model_config), encoding="utf-8")
    core_root = Path("tools/api3_anthropic_runner_v2").resolve()
    run_config_value = {
        "schema_version": "frozen_two_stage_run_config_v1",
        "core_root": str(core_root),
        "models_path": str(models_path),
        "model_key": "future-qwen-alias",
        "ordered_brief_ids": ["brief_00"],
        "route": {
            "kind": "api2_responses",
            "key": "api2-ali-responses-v1",
            "provider": "ali",
            "gateway_model": "future-qwen-alias",
            "user_agent_suffix": "api2-ali-responses-v1",
            "default_reasoning_effort": "max",
        },
        "retry": {
            "retryable_transport_statuses": ["transport_failure"],
            "retryable_http_statuses": [429, 500, 503, 504],
            "retry_ambiguous_timeouts": False,
            "continue_after_case_failure": True,
        },
        "execution_policy": {"schema_version": "future_policy_v1"},
    }
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(run_config_value), encoding="utf-8")

    loaded = load_run_config(config_path)
    route = loaded.route.build_route()

    assert loaded.model_key == "future-qwen-alias"
    assert route.public_dict()["gateway"]["provider"] == "ali"
    with pytest.raises(TrustError, match="trusted bundle"):
        check_config(loaded)


def test_legacy_adapter_uses_private_core_and_reports_adapter_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_path = Path(
        "tools/api2_kimi_k3_runner_v1/kimi_k3_generation_runner.py"
    ).resolve()
    spec = importlib.util.spec_from_file_location(
        "_test_private_kimi_adapter", adapter_path
    )
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    monkeypatch.setenv("API2_APP_CREDENTIAL", "dummy-app:dummy-key")

    assert Path(adapter.core.__file__).resolve() == Path(
        "tools/api3_anthropic_runner_v2/generation_runner.py"
    ).resolve()
    report = adapter.core.check_runner(
        briefs_path=adapter.FROZEN_CORE_ROOT / "briefs.json",
        models_path=adapter.MODELS_PATH,
        retriever_root=None,
        source_manifest=adapter._source_manifest(),
    )
    assert report["source_manifest"]["adapter_version"] == (
        "api2-kimi-k3-chat-completions-v1"
    )
    assert report["source_manifest"]["provider_route"]["route_key"] == (
        "api2-kimi-k3-chat-completions-v1"
    )


class _LoopbackServer(ThreadingHTTPServer):
    def handle_error(self, request: Any, client_address: Any) -> None:
        pass


class _LoopbackContext:
    def __init__(self, responses: list[tuple[int, bytes]]) -> None:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            requests: list[bytes] = []
            headers_seen: list[dict[str, str]] = []

            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                type(self).requests.append(body)
                type(self).headers_seen.append(dict(self.headers.items()))
                index = min(len(type(self).requests) - 1, len(owner.responses) - 1)
                status, response = owner.responses[index]
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self.responses = responses
        self.handler = Handler
        self.server = _LoopbackServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> tuple[str, type[BaseHTTPRequestHandler]]:
        self.thread.start()
        port = self.server.server_address[1]
        return f"http://127.0.0.1:{port}/v1/chat/completions", self.handler

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


@pytest.mark.requires_loopback
def test_shared_core_uses_injected_route_for_request_response_and_retry(
    tmp_path: Path,
) -> None:
    successful_body = _canonical_json_bytes(
        {
            "choices": [
                {
                    "message": {
                        "content": '{"result":"ok"}',
                        "reasoning": "private",
                        "reasoning_content": None,
                    }
                }
            ],
            "usage": {"completion_tokens": 2},
        }
    )
    with _LoopbackContext(
        [(503, b'{"error":"busy"}'), (200, successful_body)]
    ) as (endpoint, handler):
        model = shared_core.ModelConfig(
            **vars(
                _model(
                    endpoint=endpoint,
                    api_key="loopback-key",
                    temperature=0.2,
                    top_p=0.8,
                    top_k=7,
                    repetition_penalty=1.1,
                    reasoning_effort="high",
                    preserved_thinking=True,
                )
            )
        )
        route = make_api3_chat_route(
            option_policy=ChatOptionPolicy.legacy_core(),
            route_key="loopback-route",
        )
        retry_policy = RetryPolicy(
            max_infrastructure_retries=1,
            retryable_http_statuses=frozenset({503}),
            retry_delay_seconds=0,
        )
        capture = shared_core.call_model_stage(
            stage="stage_a_object_plan",
            stage_dir=tmp_path / "stage_a",
            model=model,
            system_prompt="system",
            user_value={"brief": "value"},
            provider_route=route,
            retry_policy=retry_policy,
        )

    expected_request = route.request_value(
        model=model,
        system_prompt="system",
        user_value={"brief": "value"},
        canonical_json_bytes=shared_core.canonical_json_bytes,
    )
    assert capture.status == "captured"
    assert capture.attempt_count == 2
    assert capture.infrastructure_retry_count == 1
    assert capture.content == b'{"result":"ok"}'
    assert handler.requests == [
        shared_core.canonical_json_bytes(expected_request),
        shared_core.canonical_json_bytes(expected_request),
    ]
    assert handler.headers_seen[0]["Authorization"] == "Bearer loopback-key"
    assert handler.headers_seen[0]["StrategyType"] == "ConsistentHash"
    assert handler.headers_seen[0]["SessionID"] != handler.headers_seen[1][
        "SessionID"
    ]


@pytest.mark.requires_loopback
def test_shared_core_full_case_preserves_stage_order_and_frozen_placement(
    tmp_path: Path,
) -> None:
    from tests.test_frozen_two_stage_generation_core import (
        _Retriever,
        _brief,
        _placement,
        _plan,
    )

    plan_text = json.dumps(_plan(), separators=(",", ":"))
    placement_text = json.dumps(_placement(), separators=(",", ":"))

    def response(content: str) -> bytes:
        return _canonical_json_bytes(
            {
                "choices": [{"message": {"content": content}}],
                "usage": {"completion_tokens": 1},
            }
        )

    with _LoopbackContext(
        [(200, response(plan_text)), (200, response(placement_text))]
    ) as (endpoint, handler):
        model = shared_core.ModelConfig(
            **vars(
                _model(
                    endpoint=endpoint,
                    api_key="loopback-key",
                    max_infrastructure_retries=0,
                    reasoning_effort=None,
                    preserved_thinking=None,
                )
            )
        )
        route = make_api3_chat_route(
            option_policy=ChatOptionPolicy.legacy_core(),
            route_key="loopback-full-case-route",
        )
        retry_policy = RetryPolicy(
            max_infrastructure_retries=0,
            retry_delay_seconds=0,
            continue_after_case_failure=True,
        )
        result = shared_core.run_case(
            output_root=tmp_path,
            model=model,
            brief=_brief(),
            retriever=_Retriever(),
            stage_a_prompt="stage A",
            stage_c_prompt="stage C",
            provider_route=route,
            retry_policy=retry_policy,
        )

    case_root = tmp_path / "brief_00"
    assert result["status"] == "complete"
    assert len(handler.requests) == 2
    assert (case_root / "catalog_placement_first_emission.json").read_text() == (
        placement_text
    )
    assert (case_root / "catalog_placement_v1.json").read_bytes() == (
        case_root / "catalog_placement_first_emission.json"
    ).read_bytes()
    audit = json.loads((case_root / "one_shot_audit.json").read_text())
    assert audit["object_plan_response_count"] == 1
    assert audit["retrieval_total_invocations"] == 1
    assert audit["placement_response_count"] == 1
    assert audit["generator_semantic_retry_count"] == 0
