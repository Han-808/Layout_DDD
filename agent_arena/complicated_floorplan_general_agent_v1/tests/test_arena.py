from __future__ import annotations

import http.client
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import sys
import threading
import unittest
import uuid


ARENA_ROOT = Path(__file__).resolve().parents[1]
TRUSTED = ARENA_ROOT / "trusted"
for path in (TRUSTED,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from arena import (  # noqa: E402
    ArenaError,
    WORKSPACE_FILES,
    create_episode,
    verify_episode_inputs,
    verify_fixed_suite,
)
from model_gateway import GatewayError, ScopedModelGateway  # noqa: E402
from api_profiles import ProfileRegistry, RouteRuntimeBinding  # noqa: E402
from verify_arena import verify_lock  # noqa: E402


class ArenaTests(unittest.TestCase):
    def test_lock_and_fixed_suite(self) -> None:
        lock = verify_lock()
        fixed = verify_fixed_suite()
        self.assertGreater(lock["file_count"], 40)
        self.assertEqual(fixed["scene_count"], 10)
        self.assertEqual(fixed["room_count"], 42)
        self.assertEqual(fixed["wall_segment_count"], 314)
        self.assertEqual(
            fixed["aggregate_target_total_instances"], {"min": 914, "max": 1133}
        )
        self.assertEqual(
            fixed["database_snapshot_id"],
            "imaginarium-shared-agent-db-v1-8ea5e21ef6c710f7",
        )

    def test_episode_contains_only_public_case_files(self) -> None:
        token = uuid.uuid4().hex[:12]
        episode = create_episode(
            agent_id=f"fixture-agent-{token}",
            scene_id="scene_012121",
            run_id="unit-test",
        )
        try:
            files = {
                path.name
                for path in episode.workspace.iterdir()
                if path.is_file()
            }
            self.assertEqual(files, WORKSPACE_FILES)
            self.assertTrue((episode.workspace / ".home").is_dir())
            self.assertTrue((episode.workspace / ".tmp").is_dir())
            self.assertFalse((episode.workspace / ".git").exists())
            task = json.loads(
                (episode.workspace / "task.json").read_text(encoding="utf-8")
            )
            self.assertEqual(task["layout_id"], "scene_012121")
            self.assertEqual(task["target_total_instances"], {"min": 85, "max": 106})
            complexity = task["complexity_contract"]
            self.assertEqual(complexity["count_unit"], "expanded_placed_instance")
            self.assertTrue(
                complexity["count_constraints_are_authoritative_integers"]
            )
            self.assertEqual(
                complexity["density_provenance"],
                {
                    "baseline": "original_multi_room_floorplan_planned_envelope_v1",
                    "historical_multiplier": 1.4,
                    "objects_per_m2_used_to_precompute_ranges": {
                        "min": 0.6509909031838855,
                        "max": 0.8073424301494476,
                    },
                    "provenance_only": True,
                    "additional_multiplier_allowed": False,
                },
            )
            self.assertEqual(
                complexity["room_instance_ranges"],
                [
                    {"room_id": "room_000", "min": 34, "max": 43},
                    {"room_id": "room_001", "min": 11, "max": 13},
                    {"room_id": "room_002", "min": 13, "max": 16},
                    {"room_id": "room_003", "min": 14, "max": 18},
                    {"room_id": "room_004", "min": 13, "max": 16},
                ],
            )
            self.assertEqual(
                sum(row["min"] for row in complexity["room_instance_ranges"]),
                task["target_total_instances"]["min"],
            )
            self.assertEqual(
                sum(row["max"] for row in complexity["room_instance_ranges"]),
                task["target_total_instances"]["max"],
            )
            self.assertEqual(
                task["geometry_contract"]["uniform_scale"],
                {"policy": "exact", "value": 1.0},
            )
            rendered_todo = (episode.workspace / "TODO.md").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "- `room_000`: 34 to 43 instances",
                rendered_todo,
            )
            self.assertIn(
                "This is provenance only; do not\n  apply an additional multiplier.",
                rendered_todo,
            )
            self.assertIn(
                "All counts refer to expanded placed instances, not object-plan "
                "rows, slots,\nasset bindings, or placement containers.",
                rendered_todo,
            )
            self.assertIn(
                "Multiple expanded instances for a slot whose `count` is greater "
                "than one are\n  required and are not duplicate placements.",
                rendered_todo,
            )
            self.assertIn(
                "Relation endpoints in the object plan are slot-level, not "
                "instance-level.",
                rendered_todo,
            )
            self.assertIn(
                "no hidden or\n  implicit one-to-one instance pairing is assumed",
                rendered_todo,
            )
            combined = "\n".join(
                path.read_text(encoding="utf-8", errors="replace")
                for path in episode.workspace.iterdir()
                if path.is_file()
            )
            self.assertNotIn("TOKENHUB_API_KEY", combined)
            self.assertNotIn("/Users/han_mohan", combined)
            self.assertNotIn("evaluation_preflight.json", combined)
        finally:
            scene_root = episode.root.parent
            agent_root = scene_root.parent
            shutil.rmtree(episode.root)
            scene_root.rmdir()
            agent_root.rmdir()

    def test_authoritative_input_tampering_is_rejected(self) -> None:
        token = uuid.uuid4().hex[:12]
        episode = create_episode(
            agent_id=f"tamper-agent-{token}",
            scene_id="scene_012121",
            run_id="unit-test",
        )
        try:
            floorplan = episode.workspace / "floorplan.json"
            floorplan.chmod(0o644)
            floorplan.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ArenaError, "Agent changed"):
                verify_episode_inputs(episode)
        finally:
            scene_root = episode.root.parent
            agent_root = scene_root.parent
            shutil.rmtree(episode.root)
            scene_root.rmdir()
            agent_root.rmdir()

    def test_scoped_gateway_hides_secret_and_enforces_model_budget(self) -> None:
        observed: dict[str, object] = {}
        registry = ProfileRegistry.load()
        model = registry.model("api2-gpt-5-6-sol-agent-v1")
        route = registry.route(model.route_profile_id)

        class UpstreamHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers["Content-Length"])
                observed["path"] = self.path
                observed["authorization"] = self.headers.get("Authorization")
                observed["body"] = json.loads(self.rfile.read(length))
                event = {
                    "model": "api_azure_openai_gpt-5.6-sol",
                    "choices": [{"delta": {"content": "ok"}}],
                }
                body = (
                    "data: "
                    + json.dumps(event, separators=(",", ":"))
                    + "\n\ndata: [DONE]\n\n"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        try:
            upstream_url = f"http://127.0.0.1:{upstream.server_address[1]}/v1"
            with ScopedModelGateway(
                route=route,
                model=model,
                runtime_binding=RouteRuntimeBinding(
                    route_profile_id=route.route_profile_id,
                    binding_profile_id="fixture-api2-sol-binding-v1",
                    upstream_base_url=upstream_url,
                    allow_insecure_upstream=True,
                ),
                runtime_credential="trusted-upstream-secret",
                max_requests=1,
            ) as gateway:
                unauthorized_status, _ = _gateway_request(
                    gateway.port,
                    "0" * 64,
                    route.client_path,
                    {"model": model.client_wire_model, "stream": True},
                )
                self.assertEqual(unauthorized_status, 401)
                status, body = _gateway_request(
                    gateway.port,
                    gateway.capability_token,
                    route.client_path,
                    {
                        "model": model.client_wire_model,
                        "messages": [{"role": "user", "content": "test"}],
                        "stream": True,
                        "reasoning_effort": "low",
                        "max_tokens": 1,
                    },
                )
                self.assertEqual(status, 200)
                self.assertIn("data:", body)
                self.assertEqual(observed["path"], "/v1/chat/completions")
                self.assertEqual(
                    observed["authorization"], "Bearer trusted-upstream-secret"
                )
                self.assertEqual(observed["body"]["model"], model.upstream_wire_model)
                self.assertEqual(observed["body"]["reasoning_effort"], "high")
                self.assertEqual(observed["body"]["max_tokens"], 1)
                self.assertNotIn("trusted-upstream-secret", body)

                wrong_status, _ = _gateway_request(
                    gateway.port,
                    gateway.capability_token,
                    route.client_path,
                    {"model": "different-model"},
                )
                self.assertEqual(wrong_status, 400)
                exhausted_status, _ = _gateway_request(
                    gateway.port,
                    gateway.capability_token,
                    route.client_path,
                    {"model": model.client_wire_model, "stream": True},
                )
                self.assertEqual(exhausted_status, 429)
                self.assertEqual(gateway.request_count, 1)
        finally:
            upstream.shutdown()
            upstream.server_close()
            thread.join(timeout=5.0)

    def test_gateway_rejects_invalid_upstream_timeout(self) -> None:
        registry = ProfileRegistry.load()
        original = registry.model("api2-gpt-5-6-sol-agent-v1")
        model = replace(original, request_timeout_seconds=0)
        route = registry.route(model.route_profile_id)
        with self.assertRaisesRegex(GatewayError, "request timeout"):
            ScopedModelGateway(
                route=route,
                model=model,
                runtime_binding=RouteRuntimeBinding(
                    route_profile_id=route.route_profile_id,
                    binding_profile_id="fixture-api2-sol-binding-v1",
                    upstream_base_url="http://127.0.0.1:1/v1",
                    allow_insecure_upstream=True,
                ),
                runtime_credential="fixture-secret",
                max_requests=1,
            )


def _gateway_request(
    port: int, capability: str, path: str, payload: dict[str, object]
) -> tuple[int, str]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    body = json.dumps(payload)
    connection.request(
        "POST",
        path,
        body=body,
        headers={
            "Authorization": f"Bearer {capability}",
            "Content-Type": "application/json",
        },
    )
    response = connection.getresponse()
    content = response.read().decode("utf-8")
    status = response.status
    connection.close()
    return status, content


if __name__ == "__main__":
    unittest.main()
