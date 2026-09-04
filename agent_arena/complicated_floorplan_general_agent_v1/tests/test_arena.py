from __future__ import annotations

import http.client
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
            self.assertEqual(
                task["complexity_contract"]["room_instance_ranges"],
                [
                    {"room_id": "room_000", "min": 34, "max": 43},
                    {"room_id": "room_001", "min": 11, "max": 13},
                    {"room_id": "room_002", "min": 13, "max": 16},
                    {"room_id": "room_003", "min": 14, "max": 18},
                    {"room_id": "room_004", "min": 13, "max": 16},
                ],
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

        class UpstreamHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers["Content-Length"])
                observed["path"] = self.path
                observed["authorization"] = self.headers.get("Authorization")
                observed["body"] = json.loads(self.rfile.read(length))
                body = json.dumps({"ok": True}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
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
                upstream_base_url=upstream_url,
                upstream_secret="trusted-upstream-secret",
                fixed_model="fixed-model",
                endpoint="/responses",
                max_requests=1,
                upstream_timeout_seconds=3000,
                allow_insecure_loopback_upstream=True,
            ) as gateway:
                self.assertEqual(gateway.upstream_timeout_seconds, 3000.0)
                unauthorized_status, _ = _gateway_request(
                    gateway.port,
                    "0" * 64,
                    "/responses",
                    {"model": "fixed-model"},
                )
                self.assertEqual(unauthorized_status, 401)
                status, body = _gateway_request(
                    gateway.port,
                    gateway.capability_token,
                    "/responses",
                    {"model": "fixed-model", "input": "test"},
                )
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body), {"ok": True})
                self.assertEqual(observed["path"], "/v1/responses")
                self.assertEqual(
                    observed["authorization"], "Bearer trusted-upstream-secret"
                )
                self.assertNotIn("trusted-upstream-secret", body)

                wrong_status, _ = _gateway_request(
                    gateway.port,
                    gateway.capability_token,
                    "/responses",
                    {"model": "different-model"},
                )
                self.assertEqual(wrong_status, 400)
                exhausted_status, _ = _gateway_request(
                    gateway.port,
                    gateway.capability_token,
                    "/responses",
                    {"model": "fixed-model"},
                )
                self.assertEqual(exhausted_status, 429)
                self.assertEqual(gateway.request_count, 1)
        finally:
            upstream.shutdown()
            upstream.server_close()
            thread.join(timeout=5.0)

    def test_gateway_rejects_invalid_upstream_timeout(self) -> None:
        with self.assertRaisesRegex(GatewayError, "upstream_timeout_seconds"):
            ScopedModelGateway(
                upstream_base_url="http://127.0.0.1:1/v1",
                upstream_secret="fixture-secret",
                fixed_model="fixture-model",
                endpoint="/chat/completions",
                max_requests=1,
                upstream_timeout_seconds=0,
                allow_insecure_loopback_upstream=True,
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
