from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlencode

import pytest

from benchmark.scene_generation.campaign import execution as campaign_execution
from benchmark.scene_generation.campaign import load_campaign_profile_bundle
from benchmark.scene_generation.campaign.legacy_v1 import (
    map_legacy_v1_to_campaign,
    project_legacy_v1,
)
from benchmark.scene_generation.campaign.loader import (
    load_campaign_profile_registry,
    load_model_profile_registry,
    load_route_profile_registry,
)
from benchmark.scene_generation.campaign.bindings import (
    LocalRouteBindings,
    select_binding_path as select_generation_binding_path,
)
from benchmark.scene_generation.campaign.cli import main as campaign_main
from benchmark.scene_generation.campaign.execution import (
    preflight_campaign,
    prepare_campaign,
    resolve_bindings,
    run_campaign,
)
from benchmark.scene_generation.campaign.runtime import (
    RuntimeProviderModel,
    build_provider_route,
)
from benchmark.scene_generation.retrieval import RetrievalCatalog
from benchmark.scene_generation.frozen_two_stage.config import load_run_config
from benchmark.scene_generation.frozen_two_stage.compatibility.loader import load_frozen_core
from benchmark.scene_generation.frozen_two_stage.providers.base import ProviderRoute
from tools.api3_anthropic_runner_v2.transport import TransportResult


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = REPO_ROOT / "configs" / "generation" / "campaign_v2"
ROUTE_REGISTRY = PROFILE_ROOT / "route_profiles_v2.json"
MODEL_REGISTRY = PROFILE_ROOT / "model_profiles_v2.json"
CAMPAIGN_REGISTRY = PROFILE_ROOT / "campaigns_v2.json"
RETRIEVAL_PROFILE_ID = "imaginarium-qwen3-embedding-0.6b-stable-top1-v2"
RETRIEVAL_CATALOG = REPO_ROOT / "configs" / "retrieval" / "profiles_v2.json"


LEGACY_TO_V2 = {
    "api2_kimi_k3_scene10_v1.json": (
        "api2-kimi-k3-scene10-v2",
        "api2-kimi-k3",
        "api2-chat-top-level-reasoning-v1",
    ),
    "api2_glm53_scene10_v1.json": (
        "api2-glm53-scene10-v2",
        "api2-glm-5-3",
        "api2-responses-reasoning-v1",
    ),
    "api3_opus48_high_scene10_v1.json": (
        "api3-opus48-high-scene10-v2",
        "api3-claude-opus-4-8-high",
        "api3-chat-adaptive-thinking-v1",
    ),
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _public_strings(value: Any):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield key
            yield from _public_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _public_strings(child)
    elif isinstance(value, str):
        yield value


def _normalize_headers(headers: Mapping[str, str]) -> dict[str, str]:
    normalized = dict(headers)
    authorization = normalized.get("Authorization", "")
    if "?" in authorization:
        _, query = authorization.split("?", 1)
        values = dict(parse_qsl(query, keep_blank_values=True))
        if "cache_task_id" in values:
            values["cache_task_id"] = "<dynamic-cache-task-id>"
        normalized["Authorization"] = "Bearer <credential>?" + urlencode(
            sorted(values.items())
        )
    elif authorization:
        normalized["Authorization"] = "Bearer <credential>"
    if "SessionID" in normalized:
        normalized["SessionID"] = "<dynamic-session-id>"
    return normalized


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_checked_in_campaign_bundle_is_portable_and_immutable() -> None:
    bundle = load_campaign_profile_bundle(PROFILE_ROOT)

    assert len(bundle.routes.routes) == 4
    assert len(bundle.models.models) == 7
    assert len(bundle.campaigns.campaigns) == 3
    campaign, model, route = bundle.resolve_campaign(
        "api3-opus48-high-scene10-v2"
    )
    assert campaign.retrieval_profile_id == RETRIEVAL_PROFILE_ID
    assert route.option_contract_id == "chat_adaptive_thinking_v1"
    assert model.wire_model == "claude-opus-4-8-aihub"
    with pytest.raises(FrozenInstanceError):
        campaign.campaign_id = "mutated"  # type: ignore[misc]

    public_values: list[str] = []
    for profile in bundle.routes.routes:
        public_values.extend(_public_strings(profile.to_public_dict()))
    for profile in bundle.models.models:
        public_values.extend(_public_strings(profile.to_public_dict()))
    for profile in bundle.campaigns.campaigns:
        public_values.extend(_public_strings(profile.to_public_dict()))
    lowered = "\n".join(public_values).lower()
    assert "http://" not in lowered
    assert "https://" not in lowered
    assert "/users/" not in lowered
    assert "api_key" not in lowered
    assert "credential_env" not in lowered


@pytest.mark.parametrize(
    ("legacy_name", "expected"),
    sorted(LEGACY_TO_V2.items()),
)
def test_current_v1_campaigns_map_deterministically_to_v2(
    legacy_name: str,
    expected: tuple[str, str, str],
) -> None:
    bundle = load_campaign_profile_bundle(PROFILE_ROOT)
    legacy_path = REPO_ROOT / "configs" / "generation" / legacy_name

    first = map_legacy_v1_to_campaign(legacy_path, bundle)
    second = map_legacy_v1_to_campaign(legacy_path, bundle)

    _, campaign, model, route = first
    assert (
        campaign.campaign_id,
        model.model_profile_id,
        route.route_profile_id,
    ) == expected
    assert first == second
    assert campaign.retrieval_profile_id == RETRIEVAL_PROFILE_ID


def test_legacy_mapping_rejects_unreviewed_retry_semantic_drift(
    tmp_path: Path,
) -> None:
    source = REPO_ROOT / "configs" / "generation" / "api2_kimi_k3_scene10_v1.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["core_root"] = str(REPO_ROOT / "tools" / "api3_anthropic_runner_v2")
    raw["models_path"] = str(
        REPO_ROOT / "tools" / "api2_kimi_k3_runner_v1" / "models.pod.json"
    )
    raw["retry"]["retry_ambiguous_timeouts"] = True
    path = tmp_path / "drifted-run.json"
    _write_json(path, raw)

    with pytest.raises(
        ValueError,
        match="transport_ambiguous|no reviewed v2 execution contract",
    ):
        project_legacy_v1(path)


@pytest.mark.parametrize("legacy_name", sorted(LEGACY_TO_V2))
def test_v2_profiles_preserve_current_request_and_header_envelopes(
    legacy_name: str,
) -> None:
    bundle = load_campaign_profile_bundle(PROFILE_ROOT)
    legacy_path = REPO_ROOT / "configs" / "generation" / legacy_name
    _, _, model_profile, route_profile = map_legacy_v1_to_campaign(
        legacy_path, bundle
    )
    legacy_config = load_run_config(legacy_path)
    legacy_route = legacy_config.route.build_route()
    if legacy_config.route.kind.startswith("api2_"):
        legacy_route = ProviderRoute(
            codec=legacy_route.codec,
            gateway=replace(legacy_route.gateway, clock=lambda: 123.5),
            route_key=legacy_route.key,
        )
    v2_route = build_provider_route(
        route_profile,
        model_profile,
        clock=lambda: 123.5,
    )
    runtime_model = RuntimeProviderModel.from_profile(
        model_profile,
        endpoint="https://runtime.example.invalid/v1",
        api_key="test-app:test-secret",
    )
    inputs = {
        "model": runtime_model,
        "system_prompt": "frozen-system-prompt",
        "user_value": {"z": 2, "a": ["one", "two"]},
        "canonical_json_bytes": _canonical_json_bytes,
    }

    assert legacy_route.request_value(**inputs) == v2_route.request_value(**inputs)
    assert _normalize_headers(
        legacy_route.request_headers(runtime_model, "legacy-session")
    ) == _normalize_headers(
        v2_route.request_headers(runtime_model, "v2-session")
    )


def test_new_model_alias_uses_existing_protocol_grammar_without_code_branch(
    tmp_path: Path,
) -> None:
    routes = load_route_profile_registry(ROUTE_REGISTRY)
    raw = json.loads(MODEL_REGISTRY.read_text(encoding="utf-8"))
    template = next(
        item for item in raw["models"] if item["model_profile_id"] == "api2-kimi-k3"
    )
    new_model = json.loads(json.dumps(template))
    new_model.update(
        {
            "model_profile_id": "api2-future-model-2031",
            "display_label": "Future Model 2031",
            "configured_model": "future/future-model-2031",
            "wire_model": "future-model-2031",
        }
    )
    new_model["gateway_options"]["gateway_model"] = "future-model-2031"
    raw["models"].append(new_model)
    path = tmp_path / "models.json"
    _write_json(path, raw)

    models = load_model_profile_registry(path, routes)
    model = models.by_id["api2-future-model-2031"]
    route = routes.by_id[model.route_profile_id]
    provider_route = build_provider_route(route, model, clock=lambda: 1.0)
    runtime_model = RuntimeProviderModel.from_profile(
        model,
        endpoint="https://runtime.example.invalid/v1",
        api_key="app:credential",
    )
    request = provider_route.request_value(
        model=runtime_model,
        system_prompt="system",
        user_value={"instruction": "build"},
        canonical_json_bytes=_canonical_json_bytes,
    )

    assert request["model"] == "future-model-2031"
    assert request["reasoning_effort"] == "max"
    assert type(provider_route.codec).__name__ == "OpenAIChatCodec"


def test_multiple_models_share_one_retrieval_profile_without_resource_fields() -> None:
    raw = json.loads(CAMPAIGN_REGISTRY.read_text(encoding="utf-8"))
    campaigns = raw["campaigns"]

    assert {item["retrieval_profile_id"] for item in campaigns} == {
        RETRIEVAL_PROFILE_ID
    }
    assert len({item["model_profile_id"] for item in campaigns}) == 3
    allowed = {
        "campaign_id",
        "workflow_profile_id",
        "model_profile_id",
        "retrieval_profile_id",
        "brief_set_id",
        "ordered_brief_ids",
        "execution_policy_id",
        "artifact_contract_id",
    }
    for item in campaigns:
        assert set(item) == allowed
        assert not {
            "dataset",
            "dataset_id",
            "encoder",
            "encoder_id",
            "index",
            "index_id",
        }.intersection(item)


def test_campaign_retrieval_ids_resolve_through_phase_a_catalog() -> None:
    bundle = load_campaign_profile_bundle(PROFILE_ROOT)
    catalog = RetrievalCatalog.load(RETRIEVAL_CATALOG)

    for campaign in bundle.campaigns.campaigns:
        composed = catalog.compose(campaign.retrieval_profile_id)
        assert (
            composed.profile.retrieval_profile_id
            == campaign.retrieval_profile_id
            == RETRIEVAL_PROFILE_ID
        )


@pytest.mark.parametrize(
    ("registry_name", "mutation", "error"),
    [
        (
            "route",
            lambda value: value["routes"][0].__setitem__("surprise", True),
            "unknown fields",
        ),
        (
            "model",
            lambda value: value["models"][0].__setitem__(
                "api_key", "must-not-be-public"
            ),
            "forbidden field",
        ),
        (
            "model",
            lambda value: value["models"][0].__setitem__(
                "display_label", "/Users/example/private/model"
            ),
            "local path",
        ),
        (
            "campaign",
            lambda value: value["campaigns"][0].__setitem__(
                "dataset_id", "must-be-in-retrieval-profile"
            ),
            "unknown fields",
        ),
    ],
)
def test_public_registries_reject_unknown_secret_path_and_resource_fields(
    tmp_path: Path,
    registry_name: str,
    mutation: Any,
    error: str,
) -> None:
    routes = load_route_profile_registry(ROUTE_REGISTRY)
    models = load_model_profile_registry(MODEL_REGISTRY, routes)
    source = {
        "route": ROUTE_REGISTRY,
        "model": MODEL_REGISTRY,
        "campaign": CAMPAIGN_REGISTRY,
    }[registry_name]
    raw = json.loads(source.read_text(encoding="utf-8"))
    mutation(raw)
    path = tmp_path / source.name
    _write_json(path, raw)

    with pytest.raises(ValueError, match=error):
        if registry_name == "route":
            load_route_profile_registry(path)
        elif registry_name == "model":
            load_model_profile_registry(path, routes)
        else:
            load_campaign_profile_registry(path, models)


def test_unknown_protocol_grammar_is_rejected_not_inferred_from_alias(
    tmp_path: Path,
) -> None:
    raw = json.loads(ROUTE_REGISTRY.read_text(encoding="utf-8"))
    raw["routes"][0]["codec_id"] = "future_magic_codec_v1"
    path = tmp_path / "routes.json"
    _write_json(path, raw)

    with pytest.raises(ValueError, match="unsupported protocol grammar"):
        load_route_profile_registry(path)


def test_registry_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "routes.json"
    path.write_text(
        '{"schema_version":"generation_route_profile_registry_v2",'
        '"schema_version":"generation_route_profile_registry_v2",'
        '"routes":[]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_route_profile_registry(path)


def test_runtime_model_public_projection_never_serializes_credential() -> None:
    bundle = load_campaign_profile_bundle(PROFILE_ROOT)
    _, model, _ = bundle.resolve_campaign("api3-opus48-high-scene10-v2")
    runtime = RuntimeProviderModel.from_profile(
        model,
        endpoint="https://private-runtime.example.invalid/v1",
        api_key="credential-that-must-not-be-recorded",
    )

    public = json.dumps(runtime.to_public_dict(), sort_keys=True)
    assert "credential-that-must-not-be-recorded" not in public
    assert "private-runtime.example.invalid" not in public
    assert "api_key" not in runtime.to_public_dict()


@pytest.mark.parametrize(
    "bad_value",
    [
        r"\\server\share\private",
        "grpc://private.example:4011/model",
        "label contains /Users/alice/private/data",
        "private.example:4011",
        r"prefix C:\\Users\\alice\\secret",
    ],
)
def test_public_scanner_rejects_extended_endpoint_and_local_path_shapes(
    tmp_path: Path, bad_value: str
) -> None:
    raw = json.loads(MODEL_REGISTRY.read_text(encoding="utf-8"))
    raw["models"][0]["display_label"] = bad_value
    path = tmp_path / "models.json"
    _write_json(path, raw)
    routes = load_route_profile_registry(ROUTE_REGISTRY)
    with pytest.raises(ValueError, match="endpoint|local path"):
        load_model_profile_registry(path, routes)


def test_binding_precedence_and_public_projection_are_redacted(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit.json"
    env_selected = tmp_path / "env.json"
    value = {
        "schema_version": "generation_route_bindings_v2",
        "bindings": {
            "api3-chat-adaptive-thinking-v1": {
                "endpoint": "https://private.example.invalid/v1/chat/completions",
                "credential_env": "PRIVATE_API3_KEY",
            }
        },
    }
    _write_json(explicit, value)
    _write_json(env_selected, value)
    assert select_generation_binding_path(
        repo_root=REPO_ROOT,
        explicit_path=explicit,
        environ={"LAYOUT_DDD_GENERATION_BINDINGS": str(env_selected)},
    ) == explicit.resolve()
    assert select_generation_binding_path(
        repo_root=REPO_ROOT,
        environ={"LAYOUT_DDD_GENERATION_BINDINGS": str(env_selected)},
    ) == env_selected.resolve()
    binding = LocalRouteBindings.load(explicit).require(
        "api3-chat-adaptive-thinking-v1"
    )
    public = json.dumps(binding.public_dict(), sort_keys=True)
    assert "private.example" not in public
    assert "PRIVATE_API3_KEY" not in public

    with pytest.raises(ValueError, match="header-safe"):
        binding.credential({"PRIVATE_API3_KEY": "unsafe\nheader"})


def test_generation_binding_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    link = tmp_path / "binding.json"
    _write_json(
        target,
        {
            "schema_version": "generation_route_bindings_v2",
            "bindings": {
                "api2-chat-top-level-reasoning-v1": {
                    "endpoint": "https://private.example.invalid/v1/chat/completions",
                    "credential_env": "PRIVATE_API2_KEY",
                }
            },
        },
    )
    link.symlink_to(target)
    selected = select_generation_binding_path(
        repo_root=REPO_ROOT,
        explicit_path=link,
    )
    assert selected == link.absolute()
    with pytest.raises(ValueError, match="non-symlink"):
        LocalRouteBindings.load(selected)


def test_preflight_behavior_has_no_campaign_or_model_name_heuristics() -> None:
    source = (REPO_ROOT / "src/benchmark/scene_generation/campaign/execution.py").read_text(
        encoding="utf-8"
    )
    assert 'startswith("api2-")' not in source
    assert "removesuffix" not in source


def test_campaign_binding_resolution_accepts_shared_resource_registry_superset(
    tmp_path: Path,
) -> None:
    prepared = prepare_campaign("api2-kimi-k3-scene10-v2")
    route_bindings = tmp_path / "generation.json"
    _write_json(
        route_bindings,
        {
            "schema_version": "generation_route_bindings_v2",
            "bindings": {
                prepared.route.route_profile_id: {
                    "endpoint": "https://private.example.invalid/v1/chat/completions",
                    "credential_env": "PRIVATE_API2_KEY",
                }
            },
        },
    )
    composed = RetrievalCatalog.load(RETRIEVAL_CATALOG).compose(
        prepared.campaign.retrieval_profile_id
    )
    selected_ids = {
        composed.index.metadata_file.resource_id,
        composed.index.matrix_file.resource_id,
        composed.encoder.model_resource_id,
    }
    resource_bindings = tmp_path / "resources.json"
    _write_json(
        resource_bindings,
        {
            "schema_version": "generation_resource_bindings_v2",
            "bindings": {
                **{
                    resource_id: {"path": f"resources/{index}"}
                    for index, resource_id in enumerate(sorted(selected_ids))
                },
                "future-dataset-resource-v2": {"path": "resources/future"},
            },
        },
    )

    _, binding_registry, public = resolve_bindings(
        prepared,
        generation_bindings_path=route_bindings,
        resource_bindings_path=resource_bindings,
    )

    assert "future-dataset-resource-v2" in binding_registry.paths
    assert set(public["resource_binding"]["bound_resource_ids"]) == selected_ids
    assert "future-dataset-resource-v2" not in json.dumps(public, sort_keys=True)


def test_unknown_campaign_contract_reference_fails_closed(tmp_path: Path) -> None:
    for path in PROFILE_ROOT.glob("*.json"):
        shutil.copy2(path, tmp_path / path.name)
    raw = json.loads((tmp_path / "campaigns_v2.json").read_text(encoding="utf-8"))
    raw["campaigns"][0]["execution_policy_id"] = "unknown-policy-v9"
    _write_json(tmp_path / "campaigns_v2.json", raw)
    with pytest.raises(ValueError, match="unknown contract reference"):
        load_campaign_profile_bundle(tmp_path)


def test_codec_rejects_request_options_it_cannot_emit() -> None:
    bundle = load_campaign_profile_bundle(PROFILE_ROOT)
    _, model, route = bundle.resolve_campaign("api2-kimi-k3-scene10-v2")
    invalid = replace(model, request_options=replace(model.request_options, top_p=0.9))
    with pytest.raises(ValueError, match="cannot emit"):
        invalid.validate_for(route)


def test_artifact_contract_is_executable_not_a_label() -> None:
    prepared = prepare_campaign("api2-kimi-k3-scene10-v2")
    invalid = replace(
        prepared.artifact,
        run_manifest_schema_version="hy34_two_stage_run_manifest_v999",
    )
    with pytest.raises(ValueError, match="run-manifest schema"):
        campaign_execution._validate_static_artifact_contract(
            prepared.core_root, invalid
        )


class _FakeRetrievalRuntime:
    embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"
    profile_id = RETRIEVAL_PROFILE_ID

    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events if events is not None else []

    def gate(self, **_: Any) -> dict[str, Any]:
        self.events.append("resource_gate")
        return {
            "schema_version": "generation_retrieval_gate_report_v2",
            "status": "ready",
            "strict": True,
            "errors": [],
            "warnings": [],
            "observed": self.public_provenance(),
            "golden_results": [],
        }

    def public_provenance(self) -> dict[str, Any]:
        return {
            "schema_version": "generation_retrieval_provenance_v2",
            "retrieval_profile_id": self.profile_id,
            "catalog_sha256": "0" * 64,
            "profile_sha256": "1" * 64,
        }

    def retrieve_batch(self, request: Mapping[str, Any]) -> dict[str, Any]:
        results = []
        for order, row in enumerate(request["requests"]):
            slot = str(row["slot_id"])
            results.append(
                {
                    "order": order,
                    "slot_id": slot,
                    "retrieval_query": row["retrieval_query"],
                    "size_constraint": row["size_constraint"],
                    "invocation_count": 1,
                    "rank1": {
                        "rank": 1,
                        "jid": f"asset_{slot}",
                        "short_desc": "simple chair",
                        "size": [0.5, 0.5, 0.9],
                        "category": "chair",
                        "description": "simple chair with backrest",
                        "score": 0.75,
                        "index_row": order,
                    },
                    "accepted_as_frozen_outcome": True,
                }
            )
        return {
            "schema_version": "hy34_frozen_top1_results_v1",
            "total_invocations": len(results),
            "retry_count": 0,
            "asset_replacement_count": 0,
            "results": results,
        }


class _RecordingEnvironment(dict[str, str]):
    def __init__(self, *args: Any, events: list[str], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.events = events

    def get(self, key: str, default: Any = None) -> Any:
        if key == "PRIVATE_TEST_CREDENTIAL":
            self.events.append("credential_read")
        return super().get(key, default)


def test_resource_gate_precedes_credential_and_network(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    bindings = tmp_path / "generation.json"
    _write_json(
        bindings,
        {
            "schema_version": "generation_route_bindings_v2",
            "bindings": {
                "api2-chat-top-level-reasoning-v1": {
                    "endpoint": "https://private.example.invalid/v1/chat/completions",
                    "credential_env": "PRIVATE_TEST_CREDENTIAL",
                }
            },
        },
    )
    environment = _RecordingEnvironment(
        {"PRIVATE_TEST_CREDENTIAL": "app:key"}, events=events
    )

    def factory(**_: Any) -> _FakeRetrievalRuntime:
        return _FakeRetrievalRuntime(events)

    def transport(*_: Any, **__: Any) -> TransportResult:
        events.append("network")
        body = _canonical_json_bytes(
            {"choices": [{"message": {"content": '{"ok":true}'}}]}
        )
        return TransportResult(
            status="response",
            elapsed_seconds=0.01,
            stage="complete",
            http_status=200,
            response_body=body,
        )

    report, _ = preflight_campaign(
        prepare_campaign("api2-kimi-k3-scene10-v2"),
        generation_bindings_path=bindings,
        environ=environment,
        runtime_factory=factory,
        transport=transport,
    )
    assert report["ok"] is True
    assert events == ["resource_gate", "credential_read", "network"]


class _CampaignLoopback(ThreadingHTTPServer):
    def handle_error(self, request: Any, client_address: Any) -> None:
        pass


class _CampaignLoopbackContext:
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
        self.server = _CampaignLoopback(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> tuple[str, type[BaseHTTPRequestHandler]]:
        self.thread.start()
        return (
            f"http://127.0.0.1:{self.server.server_address[1]}/v1/chat/completions",
            self.handler,
        )

    def __exit__(self, *args: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def _chat_response(content: Mapping[str, Any], *, model: str) -> bytes:
    return _canonical_json_bytes(
        {
            "model": model,
            "choices": [
                {"message": {"content": json.dumps(content, separators=(",", ":"))}}
            ],
            "usage": {"completion_tokens": 1},
        }
    )


def _twenty_object_plan() -> dict[str, Any]:
    objects = []
    for index in range(20):
        slot = f"chair_{index:02d}"
        objects.append(
            {
                "id": slot,
                "category": "chair",
                "role": "seating",
                "description": "one chair",
                "count": 1,
                "estimated_size": [0.5, 0.5, 0.9],
                "metadata": {
                    "intended_role": "seating",
                    "zone": "main_zone",
                    "support": "floor",
                    "directed": True,
                    "functional_side": "local_neg_y",
                    "facing_intent": "face world -Y",
                    "retrieval_query": "simple chair with backrest",
                    "requested_count": 1,
                },
                "placement_intent": {
                    "absolute_relations": ["inside room"],
                    "relative_relations": [],
                },
            }
        )
    return {
        "schema_version": "hy34_object_plan_v2",
        "scene_type": "open_plan_room",
        "scene_description": "A coherent open plan room.",
        "prompt_granularity": "fine_grained",
        "global_constraints": ["keep circulation usable"],
        "zones": [
            {"id": "main_zone", "description": "main area", "extent_hint": "center"}
        ],
        "relations": [],
        "objects": objects,
    }


def _twenty_placement() -> dict[str, Any]:
    return {
        "schema_version": "catalog_placement_v1",
        "instances": [
            {
                "instance_id": f"chair_{index:02d}_01",
                "asset_id": f"asset_chair_{index:02d}",
                "slot_id": f"chair_{index:02d}",
                "center_m": [0.5 + (index % 10) * 0.8, 0.8 + (index // 10) * 2.0, 0.45],
                "uniform_scale": 1.0,
                "rotation_euler_xyz_deg": [0.0, 0.0, 0.0],
            }
            for index in range(20)
        ],
    }


@pytest.mark.requires_loopback
def test_campaign_loopback_preserves_retry_response_and_redacted_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = prepare_campaign("api3-opus48-high-scene10-v2")
    prepared = replace(
        prepared,
        campaign=replace(prepared.campaign, ordered_brief_ids=("brief_00",)),
    )
    responses = [
        (200, _chat_response({"ok": True}, model="claude-opus-4-8-aihub")),
        (503, b'{"error":"busy"}'),
        (200, _chat_response(_twenty_object_plan(), model="claude-opus-4-8-aihub")),
        (200, _chat_response(_twenty_placement(), model="claude-opus-4-8-aihub")),
    ]
    with _CampaignLoopbackContext(responses) as (endpoint, handler):
        bindings = tmp_path / "generation.json"
        _write_json(
            bindings,
            {
                "schema_version": "generation_route_bindings_v2",
                "bindings": {
                    "api3-chat-adaptive-thinking-v1": {
                        "endpoint": endpoint,
                        "credential_env": "PRIVATE_TEST_CREDENTIAL",
                    }
                },
            },
        )
        # Avoid the declared production backoff in this deterministic loopback
        # while retaining the exact retry count/status contract.
        core = load_frozen_core(prepared.core_root)
        monkeypatch.setattr(core.time, "sleep", lambda _: None)
        summary, stopped, preflight = run_campaign(
            prepared,
            output_root=tmp_path / "output",
            generation_bindings_path=bindings,
            environ={"PRIVATE_TEST_CREDENTIAL": "loopback-key"},
            runtime_factory=lambda **_: _FakeRetrievalRuntime(),
        )
    assert preflight["ok"] is True
    assert stopped is False
    assert summary["complete"] == 1 and summary["failed"] == 0
    assert len(handler.requests) == 4
    assert handler.headers_seen[0]["Authorization"] == "Bearer loopback-key"
    manifest = json.loads((tmp_path / "output" / "run_manifest.json").read_text())
    serialized = json.dumps(manifest, sort_keys=True)
    assert manifest["schema_version"] == "hy34_two_stage_run_manifest_v3"
    assert manifest["source_manifest"]["schema_version"] == (
        "generation_campaign_source_manifest_v3"
    )
    assert endpoint not in serialized
    assert "loopback-key" not in serialized
    assert "credential_env" not in serialized
    assert (tmp_path / "output" / "brief_00" / "catalog_placement_v1.json").is_file()


def test_cli_check_is_an_executable_single_entrypoint(capsys: pytest.CaptureFixture[str]) -> None:
    assert campaign_main(["check", "--campaign", "api2-kimi-k3-scene10-v2"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["valid"] is True
    assert value["credential_loaded"] is False
    assert value["trust"]["schema_version"] == "generation_campaign_trust_report_v1"


def test_top_level_scene_generation_module_delegates_to_campaign_cli() -> None:
    environment = {
        **os.environ,
        "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT}",
    }

    def invoke(module: str) -> dict[str, Any]:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                module,
                "check",
                "--campaign",
                "api2-kimi-k3-scene10-v2",
            ],
            cwd=REPO_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    assert invoke("benchmark.scene_generation") == invoke(
        "benchmark.scene_generation.campaign"
    )


def test_cli_error_surface_never_echoes_private_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_path = tmp_path / "private-endpoint-and-credential-binding.json"
    exit_code = campaign_main(
        [
            "resolve",
            "--campaign",
            "api2-kimi-k3-scene10-v2",
            "--generation-bindings",
            str(private_path),
            "--resource-bindings",
            str(private_path),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 3
    assert str(private_path) not in captured.err
    assert "generation campaign command failed" in captured.err
