from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ARENA_ROOT = Path(__file__).resolve().parents[1]
TRUSTED = ARENA_ROOT / "trusted"
if str(TRUSTED) not in sys.path:
    sys.path.insert(0, str(TRUSTED))

from isolated_exec import _exclusive_log, sanitize_process_logs  # noqa: E402


class LogSanitizerTests(unittest.TestCase):
    def test_process_log_is_private_from_its_first_byte(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sieve-private-log-") as temporary:
            path = Path(temporary) / "agent.log"
            with _exclusive_log(path) as handle:
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                handle.write(b"fixture\n")

    def test_hidden_reasoning_and_routing_are_removed_but_usage_counts_remain(self) -> None:
        secret = "fixture-episode-capability-1234567890"
        with tempfile.TemporaryDirectory(prefix="sieve-log-sanitizer-") as temporary:
            path = Path(temporary) / "agent.jsonl"
            rows = [
                {
                    "type": "assistant_message",
                    "content": "public completion",
                    "reasoning_content": "private chain of thought",
                    "request_id": "request-secret",
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 34,
                        "reasoning": 21,
                        "reasoning_tokens": 21,
                    },
                    "capability_token": secret,
                    "endpoint": "https://private.example/v1",
                },
                {
                    "type": "thinking_delta",
                    "delta": "private stream thought",
                },
            ]
            path.write_text(
                "\n".join(json.dumps(item) for item in rows) + "\n",
                encoding="utf-8",
            )
            sanitize_process_logs([path], [secret])
            text = path.read_text(encoding="utf-8")
            observed = [json.loads(line) for line in text.splitlines()]
        self.assertNotIn(secret, text)
        self.assertNotIn("private chain of thought", text)
        self.assertNotIn("private stream thought", text)
        self.assertNotIn("private.example", text)
        self.assertEqual(
            observed[0]["usage"],
            {
                "input_tokens": 12,
                "output_tokens": 34,
                "reasoning": 21,
                "reasoning_tokens": 21,
            },
        )
        self.assertEqual(observed[1]["redacted"], "hidden_reasoning_event")
        self.assertEqual(observed[0]["content"], "public completion")


if __name__ == "__main__":
    unittest.main()
