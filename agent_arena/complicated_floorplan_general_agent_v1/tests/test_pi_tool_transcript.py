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

from pi_tool_transcript import (  # noqa: E402
    PiToolTranscriptError,
    project_pi_tool_transcript,
    verify_pi_tool_transcript,
)


class PiToolTranscriptTests(unittest.TestCase):
    def test_projection_is_hash_chained_complete_and_source_bound(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pi-transcript-") as temporary:
            root = Path(temporary)
            source = root / "agent.stdout.jsonl"
            transcript = root / "pi_tool_transcript.jsonl"
            source.write_text(
                "\n".join(
                    json.dumps(item, separators=(",", ":"))
                    for item in (
                        {
                            "type": "tool_execution_start",
                            "toolCallId": "raw-private-call-id",
                            "toolName": "write",
                            "args": {"path": "submission.json", "content": "{}"},
                        },
                        {
                            "type": "tool_execution_end",
                            "toolCallId": "raw-private-call-id",
                            "toolName": "write",
                            "isError": False,
                            "result": {"ok": True},
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            summary = project_pi_tool_transcript(
                source_path=source,
                output_path=transcript,
                require_complete=True,
            )
            verified = verify_pi_tool_transcript(
                source_path=source,
                transcript_path=transcript,
                summary=summary,
                require_complete=True,
            )
            text = transcript.read_text(encoding="utf-8")
            self.assertTrue(summary["complete"])
            self.assertTrue(verified["complete"])
            self.assertEqual(transcript.stat().st_mode & 0o777, 0o444)
            self.assertNotIn("raw-private-call-id", text)

    def test_transcript_or_source_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pi-transcript-tamper-") as temporary:
            root = Path(temporary)
            source = root / "agent.stdout.jsonl"
            transcript = root / "pi_tool_transcript.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "type": "tool_execution_start",
                        "toolCallId": "call-1",
                        "toolName": "bash",
                        "args": {"command": "true"},
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "type": "tool_execution_end",
                        "toolCallId": "call-1",
                        "toolName": "bash",
                        "isError": False,
                        "result": {"output": ""},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            summary = project_pi_tool_transcript(
                source_path=source,
                output_path=transcript,
                require_complete=True,
            )
            source.write_text(source.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
            with self.assertRaisesRegex(PiToolTranscriptError, "source hash"):
                verify_pi_tool_transcript(
                    source_path=source,
                    transcript_path=transcript,
                    summary=summary,
                    require_complete=True,
                )

    def test_overlapping_tool_calls_are_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pi-transcript-overlap-") as temporary:
            root = Path(temporary)
            source = root / "agent.stdout.jsonl"
            transcript = root / "pi_tool_transcript.jsonl"
            rows = [
                {
                    "type": "tool_execution_start",
                    "toolCallId": "call-1",
                    "toolName": "read",
                    "args": {"path": "task.json"},
                },
                {
                    "type": "tool_execution_start",
                    "toolCallId": "call-2",
                    "toolName": "read",
                    "args": {"path": "floorplan.json"},
                },
                {
                    "type": "tool_execution_end",
                    "toolCallId": "call-1",
                    "toolName": "read",
                    "isError": False,
                    "result": {},
                },
                {
                    "type": "tool_execution_end",
                    "toolCallId": "call-2",
                    "toolName": "read",
                    "isError": False,
                    "result": {},
                },
            ]
            source.write_text(
                "\n".join(json.dumps(item) for item in rows) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PiToolTranscriptError, "incomplete or non-sequential"):
                project_pi_tool_transcript(
                    source_path=source,
                    output_path=transcript,
                    require_complete=True,
                )


if __name__ == "__main__":
    unittest.main()
