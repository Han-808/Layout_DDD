from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tools.hy34_retrieval_conditioned_runner_v1.contracts import (
    ContractError,
    validate_brief,
)
from tools.hy34_retrieval_conditioned_runner_v1.generation_runner import (
    ModelConfig,
    _request_value,
    call_model_stage,
    run_case,
)
from tools.hy34_retrieval_conditioned_runner_v1.transport import post_once


def _api_body(content: str) -> bytes:
    return json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": content,
                        "reasoning": "private",
                        "reasoning_content": "provider-compatible-private",
                    }
                }
            ],
            "usage": {"completion_tokens": 100},
        },
        separators=(",", ":"),
    ).encode("utf-8")


class _Server(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        pass


class _Context:
    def __init__(self, plan: list[tuple[int, bytes, float]]) -> None:
        parent = self

        class Handler(BaseHTTPRequestHandler):
            requests: list[bytes] = []
            headers_seen: list[dict[str, str]] = []

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                type(self).requests.append(body)
                type(self).headers_seen.append(dict(self.headers.items()))
                index = min(len(type(self).requests) - 1, len(parent.plan) - 1)
                status, response, delay = parent.plan[index]
                if delay:
                    time.sleep(delay)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("X-Request-ID", f"loopback-{index + 1}")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, format, *args):
                pass

        self.plan = plan
        self.handler = Handler
        self.server = _Server(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1/chat/completions", self.handler

    def __exit__(self, exc_type, exc, tb):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def _model(endpoint: str, *, retries: int = 0, timeout: float = 1.0) -> ModelConfig:
    return ModelConfig(
        key="test-model",
        label="Test Model",
        endpoint=endpoint,
        api_key="EMPTY",
        configured_model="openai/test-model",
        wire_model="test-model",
        timeout_seconds=timeout,
        max_infrastructure_retries=retries,
        retry_delay_seconds=0.0,
        temperature=0.9,
        top_p=1.0,
        top_k=-1,
        max_tokens=65536,
        repetition_penalty=1.0,
        reasoning_effort="high",
        preserved_thinking=True,
        strategy_type="ConsistentHash",
    )


def _brief() -> dict:
    return validate_brief(
        {
            "brief_id": "brief_00",
            "room_type": "test room",
            "room_dimensions_m": [4.0, 4.0, 3.0],
            "target_instances": {"min": 1, "max": 1},
            "instruction": "Generate one useful chair.",
            "physical_wall_policy": "explicit_only",
            "active_wall_ids": [],
        }
    )


def _plan() -> dict:
    return {
        "schema_version": "hy34_object_plan_v1",
        "scene_description": "A single useful chair in a small room.",
        "global_constraints": ["keep the chair usable"],
        "zones": [{"id": "main_zone", "description": "main area", "extent_hint": "center"}],
        "objects": [
            {
                "id": "chair",
                "category": "chair",
                "role": "seating",
                "description": "one chair",
                "count": 1,
                "estimated_size": [0.5, 0.5, 0.9],
                "zone": "main_zone",
                "support": "floor",
                "directed": True,
                "facing_intent": "face world -Y",
                "retrieval_query": "simple chair with backrest",
                "absolute_relations": ["center of room"],
                "relative_relations": [],
            }
        ],
    }


def _placement(*, asset_id: str = "asset_chair") -> dict:
    return {
        "schema_version": "catalog_placement_v1",
        "instances": [
            {
                "instance_id": "chair_01",
                "asset_id": asset_id,
                "slot_id": "chair",
                "center_m": [2.0, 2.0, 0.45],
                "uniform_scale": 1.0,
                "rotation_euler_xyz_deg": [0.0, 0.0, 0.0],
            }
        ],
    }


class _Retriever:
    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, request):
        self.calls += 1
        row = request["requests"][0]
        return {
            "schema_version": "hy34_frozen_top1_results_v1",
            "total_invocations": 1,
            "retry_count": 0,
            "asset_replacement_count": 0,
            "results": [
                {
                    "order": 0,
                    "slot_id": "chair",
                    "retrieval_query": row["retrieval_query"],
                    "size_constraint": row["size_constraint"],
                    "invocation_count": 1,
                    "rank1": {
                        "rank": 1,
                        "jid": "asset_chair",
                        "short_desc": "simple chair",
                        "size": [0.5, 0.5, 0.9],
                        "category": "chair",
                        "description": "simple chair with backrest",
                        "score": 0.75,
                        "index_row": 4,
                    },
                    "accepted_as_frozen_outcome": True,
                }
            ],
        }


def test_two_stage_case_is_exactly_one_plan_one_retrieval_one_placement(tmp_path: Path) -> None:
    plan_text = json.dumps(_plan(), separators=(",", ":"))
    placement_text = json.dumps(_placement(), separators=(",", ":"))
    retriever = _Retriever()
    with _Context([(200, _api_body(plan_text), 0.0), (200, _api_body(placement_text), 0.0)]) as (
        endpoint,
        handler,
    ):
        result = run_case(
            output_root=tmp_path,
            model=_model(endpoint),
            brief=_brief(),
            retriever=retriever,
            stage_a_prompt="same stage A prompt",
            stage_c_prompt="same stage C prompt",
        )
    case = tmp_path / "brief_00"
    assert result["status"] == "complete"
    assert result["eligible_for_strict_one_shot_evaluation"] is True
    assert len(handler.requests) == 2
    assert retriever.calls == 1
    assert (case / "catalog_placement_first_emission.json").read_text() == placement_text
    assert (case / "catalog_placement_v1.json").read_bytes() == (
        case / "catalog_placement_first_emission.json"
    ).read_bytes()
    audit = json.loads((case / "one_shot_audit.json").read_text())
    assert audit["object_plan_response_count"] == 1
    assert audit["placement_response_count"] == 1
    assert audit["placement_emission_count"] == 1
    assert audit["generator_semantic_retry_count"] == 0
    assert audit["post_emission_transform_edit_count"] == 0
    assert audit["geometry_feedback_used_before_freeze"] is False
    for request in handler.requests:
        sent = json.loads(request)
        assert "test-model" not in sent["messages"][0]["content"]
        assert "Test Model" not in sent["messages"][1]["content"]


def test_invalid_placement_is_preserved_without_regeneration(tmp_path: Path) -> None:
    plan_text = json.dumps(_plan(), separators=(",", ":"))
    invalid_text = json.dumps(_placement(asset_id="replacement_asset"), separators=(",", ":"))
    retriever = _Retriever()
    with _Context([(200, _api_body(plan_text), 0.0), (200, _api_body(invalid_text), 0.0)]) as (
        endpoint,
        handler,
    ):
        result = run_case(
            output_root=tmp_path,
            model=_model(endpoint),
            brief=_brief(),
            retriever=retriever,
            stage_a_prompt="A",
            stage_c_prompt="C",
        )
    case = tmp_path / "brief_00"
    assert result["status"] == "placement_schema_invalid"
    assert len(handler.requests) == 2
    assert (case / "catalog_placement_first_emission.json").read_text() == invalid_text
    assert not (case / "catalog_placement_v1.json").exists()
    audit = json.loads((case / "one_shot_audit.json").read_text())
    assert audit["placement_emission_count"] == 1
    assert audit["eligible_for_strict_one_shot_evaluation"] is False


def test_http_infrastructure_retry_reuses_request_but_not_session_id(tmp_path: Path) -> None:
    content = json.dumps(_plan(), separators=(",", ":"))
    with _Context([(503, b'{"error":"busy"}', 0.0), (200, _api_body(content), 0.0)]) as (
        endpoint,
        handler,
    ):
        capture = call_model_stage(
            stage="stage_a_object_plan",
            stage_dir=tmp_path / "stage_a",
            model=_model(endpoint, retries=1),
            system_prompt="A",
            user_value={"brief": _brief()},
        )
    assert capture.status == "captured"
    assert capture.infrastructure_retry_count == 1
    assert handler.requests[0] == handler.requests[1]
    assert handler.headers_seen[0]["SessionID"] != handler.headers_seen[1]["SessionID"]


def test_schema_invalid_object_plan_is_not_retried(tmp_path: Path) -> None:
    invalid_plan = json.dumps({"schema_version": "hy34_object_plan_v1", "objects": []})
    retriever = _Retriever()
    with _Context([(200, _api_body(invalid_plan), 0.0)]) as (endpoint, handler):
        result = run_case(
            output_root=tmp_path,
            model=_model(endpoint, retries=2),
            brief=_brief(),
            retriever=retriever,
            stage_a_prompt="A",
            stage_c_prompt="C",
        )
    assert result["status"] == "stage_a_schema_invalid"
    assert len(handler.requests) == 1
    assert retriever.calls == 0


def test_contract_rejects_count_outside_brief() -> None:
    plan = _plan()
    plan["objects"][0]["count"] = 2
    from tools.hy34_retrieval_conditioned_runner_v1.contracts import validate_object_plan

    with pytest.raises(ContractError, match="outside"):
        validate_object_plan(plan, brief=_brief())


def test_static_prompts_exclude_agent_concurrency_and_model_identity() -> None:
    root = Path("tools/hy34_retrieval_conditioned_runner_v1")
    prompts = [
        (root / "stage_a_prompt.txt").read_text(),
        (root / "stage_c_prompt.txt").read_text(),
    ]
    banned = (
        "four agents",
        "sibling",
        "shard_dir",
        "grok4.6",
        "opus5",
        "sonnet5",
        "hy3-aw",
        "hy4-opus5",
    )
    for prompt in prompts:
        lowered = prompt.lower()
        assert all(token not in lowered for token in banned)


def test_hy3_hy4_requests_differ_only_in_model_identity() -> None:
    common = dict(
        endpoint="http://127.0.0.1:1/v1/chat/completions",
        api_key="EMPTY",
        timeout_seconds=1.0,
        max_infrastructure_retries=0,
        retry_delay_seconds=0.0,
        temperature=0.9,
        top_p=1.0,
        top_k=-1,
        max_tokens=65536,
        repetition_penalty=1.0,
        reasoning_effort="high",
        preserved_thinking=True,
        strategy_type="ConsistentHash",
    )
    left = ModelConfig(
        key="hy3-aw", label="HY3-AW", configured_model="openai/hy3", wire_model="hy3", **common
    )
    right = ModelConfig(
        key="hy4-opus5", label="HY4-Opus5", configured_model="openai/hy4", wire_model="hy4", **common
    )
    left_request = _request_value(
        model=left, system_prompt="identical", user_value={"brief": _brief()}
    )
    right_request = _request_value(
        model=right, system_prompt="identical", user_value={"brief": _brief()}
    )
    assert left_request.pop("model") == "hy3"
    assert right_request.pop("model") == "hy4"
    assert left_request == right_request


def test_timeout_after_request_delivery_is_ambiguous() -> None:
    with _Context([(200, b"{}", 0.2)]) as (endpoint, handler):
        result = post_once(
            endpoint,
            b"{}",
            connect_timeout=0.2,
            read_timeout=0.03,
            request_headers={"Content-Type": "application/json"},
        )
        time.sleep(0.25)
    assert result.status == "transport_ambiguous"
    assert len(handler.requests) == 1
