import hashlib
import json
import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from sceneeval_hy4.artifacts import ArtifactError
from sceneeval_hy4.constants import MODEL_ALIAS
from sceneeval_hy4.inputs import InputBatch, SceneInput
from sceneeval_hy4.prompt import protocol_text
from sceneeval_hy4.runner import (
    RunConfig,
    initialize_run,
    run_batch,
    run_scene,
    verify_resume,
)
from sceneeval_hy4.transport import post_once


VALID_LAYOUT_TEXT = json.dumps(
    {
        "rooms": [
            {
                "id": "room_1",
                "category": "office",
                "origin_m": [0, 0, 0],
                "size_m": [4, 4, 2.8],
            }
        ],
        "objects": [
            {
                "id": "desk_1",
                "room_id": "room_1",
                "category": "desk",
                "appearance": "plain oak desk",
                "position_m": [0, 0, 0],
                "size_m": [1.2, 0.6, 0.75],
                "yaw_deg": 0,
            }
        ]
    },
    separators=(",", ":"),
)
THINK_TAG = "</think:6124c78e>"


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        pass


class FakeHandler(BaseHTTPRequestHandler):
    response_plan = [(200, b"{}", 0.0)]
    requests = []
    request_headers = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length)
        request_index = len(type(self).requests)
        type(self).requests.append(body)
        type(self).request_headers.append(dict(self.headers.items()))
        status, response_body, delay = type(self).response_plan[
            min(request_index, len(type(self).response_plan) - 1)
        ]
        if delay:
            time.sleep(delay)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Request-Id", f"loopback-test-request-{request_index + 1}")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, format, *args):
        pass


class ServerContext:
    def __init__(self, response_body=None, *, delay=0.0, status=200, plan=None):
        if plan is None:
            if response_body is None:
                response_body = b"{}"
            plan = [(status, response_body, delay)]
        handler = type("PerTestHandler", (FakeHandler,), {})
        handler.response_plan = plan
        handler.requests = []
        handler.request_headers = []
        self.handler = handler
        self.server = QuietServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        port = self.server.server_address[1]
        return f"http://127.0.0.1:{port}/v1/chat/completions", self.handler

    def __exit__(self, exc_type, exc, tb):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def api_body(
    content,
    reasoning=None,
    reasoning_content=None,
    *,
    completion_tokens=20,
    reasoning_tokens=5,
    include_usage=True,
):
    value = {
            "choices": [
                {
                    "message": {
                        "content": content,
                        "reasoning": reasoning,
                        "reasoning_content": reasoning_content,
                    }
                }
            ],
        }
    if include_usage:
        value["usage"] = {
            "completion_tokens": completion_tokens,
            "completion_tokens_details": {
                "reasoning_tokens": reasoning_tokens,
            },
        }
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def config(endpoint, *, timeout=1.0, max_retries=2):
    return RunConfig(
        endpoint=endpoint,
        timeout_seconds=timeout,
        max_retries=max_retries,
        retry_delay_seconds=0.0,
    )


class TransportTests(unittest.TestCase):
    def test_connection_refused_is_transport_failure(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        result = post_once(
            f"http://127.0.0.1:{port}/completion",
            b"{}",
            connect_timeout=0.2,
            read_timeout=0.2,
        )
        self.assertEqual(result.status, "transport_failure")
        self.assertEqual(result.stage, "connect")

    def test_timeout_after_request_is_ambiguous(self):
        with ServerContext(b"{}", delay=0.2) as (endpoint, handler):
            result = post_once(
                endpoint,
                b"{}",
                connect_timeout=0.2,
                read_timeout=0.03,
            )
            time.sleep(0.25)
            self.assertEqual(result.status, "transport_ambiguous")
            self.assertEqual(len(handler.requests), 1)


class RunnerTests(unittest.TestCase):
    def test_capture_preserves_exact_artifacts_and_two_message_request(self):
        description = "A desk."
        response_body = api_body(
            VALID_LAYOUT_TEXT,
            reasoning="private reasoning",
            reasoning_content="provider compatibility reasoning",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with ServerContext(response_body) as (endpoint, handler):
                result = run_scene(
                    SceneInput(0, description),
                    output,
                    config(endpoint),
                )
            attempt_dir = output / "scene_000" / "attempt_01"
            self.assertEqual(result.status, "captured")
            self.assertEqual(result.attempt_count, 1)
            self.assertEqual(len(handler.requests), 1)
            self.assertEqual(
                (attempt_dir / "request.json").read_bytes(), handler.requests[0]
            )
            self.assertEqual(
                (attempt_dir / "api-response.body").read_bytes(), response_body
            )
            self.assertEqual(
                (attempt_dir / "raw-content.txt").read_text("utf-8"),
                VALID_LAYOUT_TEXT,
            )
            self.assertEqual(
                (attempt_dir / "logs" / "reasoning.txt").read_text("utf-8"),
                "private reasoning",
            )
            self.assertEqual(
                (attempt_dir / "logs" / "reasoning_content.txt").read_text("utf-8"),
                "provider compatibility reasoning",
            )
            capture = json.loads((attempt_dir / "capture.json").read_text("utf-8"))
            headers = json.loads(
                (attempt_dir / "response-headers.json").read_text("utf-8")
            )
            scene_result = json.loads(
                (output / "scene_000" / "scene.result.json").read_text("utf-8")
            )
            self.assertEqual(capture["status"], "captured")
            self.assertEqual(capture["x_request_id"], "loopback-test-request-1")
            self.assertEqual(headers["x_request_id"], "loopback-test-request-1")
            self.assertEqual(scene_result["attempt_statuses"], ["captured"])
            self.assertEqual(scene_result["accepted_attempt_number"], 1)
            self.assertEqual(
                scene_result["accepted_request_sha256"],
                json.loads(
                    (attempt_dir / "attempt.started.json").read_text("utf-8")
                )["request_sha256"],
            )
            self.assertEqual(
                scene_result["accepted_response_sha256"],
                hashlib.sha256(response_body).hexdigest(),
            )
            self.assertEqual(
                scene_result["accepted_raw_content_sha256"],
                hashlib.sha256(VALID_LAYOUT_TEXT.encode("utf-8")).hexdigest(),
            )
            self.assertEqual(
                scene_result["accepted_session_id"],
                json.loads(
                    (attempt_dir / "request-headers.json").read_text("utf-8")
                )["session_id"],
            )
            self.assertEqual(
                scene_result["accepted_x_request_id"],
                "loopback-test-request-1",
            )
            self.assertEqual(
                capture["raw_content_sha256"],
                hashlib.sha256(VALID_LAYOUT_TEXT.encode("utf-8")).hexdigest(),
            )
            self.assertFalse(capture["model_content_json_parse_performed"])
            self.assertFalse(capture["model_content_schema_validation_performed"])
            self.assertFalse(capture["reasoning_channel_normalization_performed"])
            self.assertEqual(capture["api_reasoning_state"], "nonempty_string")
            self.assertEqual(
                capture["api_reasoning_content_state"], "nonempty_string"
            )
            self.assertFalse(capture["reasoning_fields_compared"])
            self.assertEqual(capture["completion_tokens"], 20)
            self.assertEqual(capture["reasoning_tokens"], 5)
            self.assertEqual(capture["visible_output_tokens"], 15)
            self.assertFalse((attempt_dir / "normalized-content.txt").exists())
            self.assertFalse((attempt_dir / "validation.json").exists())

            sent = json.loads(handler.requests[0])
            self.assertEqual(sent["model"], MODEL_ALIAS)
            self.assertEqual(
                [message["role"] for message in sent["messages"]],
                ["system", "user"],
            )
            self.assertEqual(sent["messages"][0]["content"], protocol_text())
            self.assertEqual(sent["messages"][1]["content"], description)
            self.assertNotIn("Scene ID", sent["messages"][1]["content"])
            self.assertEqual(sent["temperature"], 0.9)
            self.assertEqual(sent["top_p"], 1.0)
            self.assertEqual(sent["top_k"], -1)
            self.assertEqual(sent["max_tokens"], 65536)
            self.assertEqual(sent["repetition_penalty"], 1.0)
            self.assertEqual(
                sent["chat_template_kwargs"],
                {"reasoning_effort": "high", "preserved_thinking": True},
            )
            request_headers = json.loads(
                (attempt_dir / "request-headers.json").read_text("utf-8")
            )
            header_map = dict(request_headers["headers"])
            self.assertEqual(header_map["Authorization"], "Bearer EMPTY")
            self.assertEqual(header_map["StrategyType"], "ConsistentHash")
            self.assertEqual(header_map["SessionID"], request_headers["session_id"])
            self.assertTrue(header_map["SessionID"])
            self.assertEqual(
                handler.request_headers[0]["SessionID"],
                header_map["SessionID"],
            )

    def test_arbitrary_model_content_is_captured_once_without_validation(self):
        cases = [
            "```json\n{}\n```",
            '{"objects":[{"id":"x"}]}',
            "plain prose with no JSON",
            "analysis" + THINK_TAG + VALID_LAYOUT_TEXT,
        ]
        for index, content in enumerate(cases):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                with ServerContext(api_body(content)) as (endpoint, handler):
                    result = run_scene(
                        SceneInput(index, "test"),
                        Path(directory),
                        config(endpoint),
                    )
                attempt_dir = Path(directory) / f"scene_{index:03d}" / "attempt_01"
                capture = json.loads(
                    (attempt_dir / "capture.json").read_text("utf-8")
                )
                self.assertEqual(result.status, "captured")
                self.assertFalse(result.stop_batch)
                self.assertEqual(result.attempt_count, 1)
                self.assertEqual(len(handler.requests), 1)
                self.assertEqual(
                    (attempt_dir / "raw-content.txt").read_text("utf-8"), content
                )
                self.assertFalse(capture["model_content_validation_performed"])
                self.assertFalse(
                    capture["reasoning_channel_normalization_performed"]
                )
                self.assertFalse((attempt_dir / "validation.json").exists())
                self.assertFalse((attempt_dir / "normalized-content.txt").exists())

    def test_http_error_is_retried_once_then_capture_wins(self):
        error_body = b'{"error":{"type":"model response empty"}}'
        with tempfile.TemporaryDirectory() as directory:
            with ServerContext(
                plan=[
                    (400, error_body, 0.0),
                    (200, api_body(VALID_LAYOUT_TEXT), 0.0),
                ]
            ) as (endpoint, handler):
                result = run_scene(
                    SceneInput(0, "test"),
                    Path(directory),
                    config(endpoint),
                )
            scene_dir = Path(directory) / "scene_000"
            self.assertEqual(result.status, "captured")
            self.assertEqual(result.attempt_count, 2)
            self.assertEqual(len(handler.requests), 2)
            first = json.loads(
                (scene_dir / "attempt_01" / "attempt.result.json").read_text("utf-8")
            )
            second = json.loads(
                (scene_dir / "attempt_02" / "attempt.result.json").read_text("utf-8")
            )
            final = json.loads((scene_dir / "scene.result.json").read_text("utf-8"))
            self.assertEqual(first["status"], "http_error")
            self.assertEqual(second["status"], "captured")
            self.assertEqual(final["attempt_statuses"], ["http_error", "captured"])
            self.assertEqual(final["accepted_attempt_number"], 2)
            first_headers = json.loads(
                (scene_dir / "attempt_01" / "request-headers.json").read_text("utf-8")
            )
            second_headers = json.loads(
                (scene_dir / "attempt_02" / "request-headers.json").read_text("utf-8")
            )
            self.assertNotEqual(
                first_headers["session_id"],
                second_headers["session_id"],
            )

    def test_short_output_retries_until_visible_tokens_reach_ten(self):
        with tempfile.TemporaryDirectory() as directory:
            with ServerContext(
                plan=[
                    (
                        200,
                        api_body(
                            "short one",
                            completion_tokens=69,
                            reasoning_tokens=61,
                        ),
                        0.0,
                    ),
                    (
                        200,
                        api_body(
                            "short two",
                            completion_tokens=30,
                            reasoning_tokens=21,
                        ),
                        0.0,
                    ),
                    (
                        200,
                        api_body(
                            "accepted output",
                            completion_tokens=40,
                            reasoning_tokens=30,
                        ),
                        0.0,
                    ),
                ]
            ) as (endpoint, handler):
                result = run_scene(
                    SceneInput(0, "test"),
                    Path(directory),
                    config(endpoint),
                )
            scene_dir = Path(directory) / "scene_000"
            self.assertEqual(result.status, "captured")
            self.assertEqual(result.attempt_count, 3)
            self.assertEqual(len(handler.requests), 3)
            final = json.loads((scene_dir / "scene.result.json").read_text("utf-8"))
            self.assertEqual(
                final["attempt_statuses"],
                ["short_output", "short_output", "captured"],
            )
            visible = []
            session_ids = []
            for attempt_number in range(1, 4):
                attempt_dir = scene_dir / f"attempt_{attempt_number:02d}"
                capture = json.loads(
                    (attempt_dir / "capture.json").read_text("utf-8")
                )
                headers = json.loads(
                    (attempt_dir / "request-headers.json").read_text("utf-8")
                )
                visible.append(capture["visible_output_tokens"])
                session_ids.append(headers["session_id"])
            self.assertEqual(visible, [8, 9, 10])
            self.assertEqual(len(set(session_ids)), 3)

    def test_missing_usage_stops_without_retry_or_token_estimation(self):
        with tempfile.TemporaryDirectory() as directory:
            with ServerContext(
                api_body("content exists", include_usage=False)
            ) as (endpoint, handler):
                result = run_scene(
                    SceneInput(0, "test"),
                    Path(directory),
                    config(endpoint),
                )
            attempt_dir = Path(directory) / "scene_000" / "attempt_01"
            capture = json.loads((attempt_dir / "capture.json").read_text("utf-8"))
            self.assertEqual(result.status, "token_count_unavailable")
            self.assertTrue(result.stop_batch)
            self.assertEqual(result.attempt_count, 1)
            self.assertEqual(len(handler.requests), 1)
            self.assertEqual(capture["status"], "token_count_unavailable")
            self.assertIsNone(capture["visible_output_tokens"])
            self.assertEqual(
                capture["token_count_error_type"],
                "TokenCountUnavailable",
            )

    def test_retryable_failure_uses_two_retries_then_exhausts_without_stop(self):
        invalid_envelope = b'{"choices":[]}'
        with tempfile.TemporaryDirectory() as directory:
            with ServerContext(invalid_envelope) as (endpoint, handler):
                result = run_scene(
                    SceneInput(0, "test"),
                    Path(directory),
                    config(endpoint),
                )
            scene_dir = Path(directory) / "scene_000"
            self.assertEqual(result.status, "retry_exhausted")
            self.assertFalse(result.stop_batch)
            self.assertEqual(result.attempt_count, 3)
            self.assertEqual(len(handler.requests), 3)
            final = json.loads((scene_dir / "scene.result.json").read_text("utf-8"))
            self.assertEqual(
                final["attempt_statuses"],
                [
                    "invalid_api_response",
                    "invalid_api_response",
                    "invalid_api_response",
                ],
            )
            self.assertIsNone(final["accepted_attempt_number"])
            self.assertIsNone(final["accepted_raw_content_sha256"])

    def test_pre_delivery_transport_failure_is_retried_once(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        with tempfile.TemporaryDirectory() as directory:
            result = run_scene(
                SceneInput(0, "test"),
                Path(directory),
                config(f"http://127.0.0.1:{port}/completion"),
            )
            scene_dir = Path(directory) / "scene_000"
            self.assertEqual(result.status, "retry_exhausted")
            self.assertFalse(result.stop_batch)
            self.assertEqual(result.attempt_count, 3)
            first = json.loads(
                (scene_dir / "attempt_01" / "attempt.result.json").read_text("utf-8")
            )
            second = json.loads(
                (scene_dir / "attempt_02" / "attempt.result.json").read_text("utf-8")
            )
            third = json.loads(
                (scene_dir / "attempt_03" / "attempt.result.json").read_text("utf-8")
            )
            self.assertEqual(first["status"], "transport_failure")
            self.assertEqual(second["status"], "transport_failure")
            self.assertEqual(third["status"], "transport_failure")

    def test_transport_ambiguous_is_never_retried_and_stops_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            with ServerContext(api_body(VALID_LAYOUT_TEXT), delay=0.2) as (
                endpoint,
                handler,
            ):
                result = run_scene(
                    SceneInput(0, "test"),
                    Path(directory),
                    config(endpoint, timeout=0.03),
                )
                time.sleep(0.25)
            scene_dir = Path(directory) / "scene_000"
            self.assertEqual(result.status, "transport_ambiguous")
            self.assertTrue(result.stop_batch)
            self.assertEqual(result.attempt_count, 1)
            self.assertEqual(len(handler.requests), 1)
            self.assertFalse((scene_dir / "attempt_02").exists())

    def test_tagged_reasoning_is_preserved_as_raw_and_never_split(self):
        content = (
            "The user wants a layout.\n"
            'Draft that must never be selected: {"objects":[]}\n'
            + THINK_TAG
            + VALID_LAYOUT_TEXT
        )
        with tempfile.TemporaryDirectory() as directory:
            with ServerContext(api_body(content)) as (endpoint, handler):
                result = run_scene(
                    SceneInput(0, "A desk."),
                    Path(directory),
                    config(endpoint),
                )
            attempt_dir = Path(directory) / "scene_000" / "attempt_01"
            self.assertEqual(result.status, "captured")
            self.assertEqual(len(handler.requests), 1)
            self.assertEqual(
                (attempt_dir / "raw-content.txt").read_text("utf-8"), content
            )
            capture = json.loads((attempt_dir / "capture.json").read_text("utf-8"))
            self.assertEqual(capture["api_reasoning_state"], "null")
            self.assertEqual(capture["api_reasoning_content_state"], "null")
            self.assertFalse(capture["reasoning_channel_normalization_performed"])
            self.assertFalse((attempt_dir / "logs").exists())

    def test_resume_refuses_changed_prompt_snapshot(self):
        exact_bytes = b'{"id":0,"description":"test"}\n'
        batch = InputBatch(
            scenes=(SceneInput(0, "test"),),
            exact_bytes=exact_bytes,
            sha256=hashlib.sha256(exact_bytes).hexdigest(),
        )
        run_config = RunConfig(
            endpoint="http://127.0.0.1:1/test",
            timeout_seconds=1.0,
            retry_delay_seconds=0.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            initialize_run(output, batch, run_config)
            (output / "prompt_protocol.txt").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactError, "prompt/schema differs"):
                verify_resume(output, batch, run_config)

    def test_run_persists_runner_source_provenance_and_execution_summary(self):
        exact_bytes = b'{"id":0,"description":"test"}\n'
        batch = InputBatch(
            scenes=(SceneInput(0, "test"),),
            exact_bytes=exact_bytes,
            sha256=hashlib.sha256(exact_bytes).hexdigest(),
        )
        response_body = api_body(VALID_LAYOUT_TEXT)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with ServerContext(response_body) as (endpoint, handler):
                run_config = config(endpoint)
                summary, stopped = run_batch(
                    batch,
                    output,
                    run_config,
                    resume=False,
                )
            self.assertFalse(stopped)
            self.assertEqual(len(handler.requests), 1)
            manifest = json.loads(
                (output / "run-manifest.json").read_text("utf-8")
            )
            source = json.loads(
                (output / "runner-source-manifest.json").read_text("utf-8")
            )
            self.assertEqual(manifest["runner"]["version"], "0.5.0")
            self.assertEqual(
                manifest["runner"]["source_manifest_sha256"],
                source["source_manifest_sha256"],
            )
            self.assertGreater(len(source["files"]), 1)
            summary_path = output / summary["execution_summary_file"]
            self.assertTrue(summary_path.is_file())
            persisted = json.loads(summary_path.read_text("utf-8"))
            self.assertEqual(persisted["runner_version"], "0.5.0")
            self.assertEqual(persisted["summary"], summary)
            self.assertEqual(summary["scene_status_counts"], {"captured": 1})
            self.assertEqual(summary["attempt_status_counts"], {"captured": 1})
            resumed_summary, resumed_stopped = run_batch(
                batch,
                output,
                run_config,
                resume=True,
            )
            self.assertFalse(resumed_stopped)
            self.assertEqual(
                resumed_summary["scene_status_counts"],
                {"captured": 1},
            )
            self.assertEqual(
                len(list((output / "execution-summaries").glob("summary_*.json"))),
                2,
            )

    def test_resume_refuses_changed_runner_source_snapshot(self):
        exact_bytes = b'{"id":0,"description":"test"}\n'
        batch = InputBatch(
            scenes=(SceneInput(0, "test"),),
            exact_bytes=exact_bytes,
            sha256=hashlib.sha256(exact_bytes).hexdigest(),
        )
        run_config = RunConfig(
            endpoint="http://127.0.0.1:1/test",
            timeout_seconds=1.0,
            retry_delay_seconds=0.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            initialize_run(output, batch, run_config)
            source_path = output / "runner-source-manifest.json"
            source = json.loads(source_path.read_text("utf-8"))
            source["files"][0]["sha256"] = "0" * 64
            source_path.write_text(
                json.dumps(source, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ArtifactError,
                "runner source differs",
            ):
                verify_resume(output, batch, run_config)


if __name__ == "__main__":
    unittest.main()
