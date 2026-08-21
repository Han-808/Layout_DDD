import json
import tempfile
import unittest
from pathlib import Path

from sceneeval_hy4.inputs import InputError, load_human100_jsonl
from sceneeval_hy4.prompt import build_system_prompt, build_user_prompt


class InputTests(unittest.TestCase):
    def _write(self, rows):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "input.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        self.addCleanup(directory.cleanup)
        return path

    def test_accepts_exact_human100_shape(self):
        path = self._write(
            [{"id": index, "description": f"scene {index}"} for index in range(100)]
        )
        batch = load_human100_jsonl(path)
        self.assertEqual(len(batch.scenes), 100)
        self.assertEqual(batch.scenes[-1].scene_id, 99)

    def test_rejects_annotation_leakage(self):
        rows = [
            {"id": index, "description": f"scene {index}"} for index in range(100)
        ]
        rows[0]["annotations"] = "must not reach model"
        with self.assertRaises(InputError):
            load_human100_jsonl(self._write(rows))

    def test_rejects_subset_or_reordering(self):
        with self.assertRaises(InputError):
            load_human100_jsonl(
                self._write(
                    [{"id": index, "description": "x"} for index in range(99)]
                )
            )


class PromptTests(unittest.TestCase):
    def test_system_contains_protocol_and_user_is_exact_description_only(self):
        description = "A red chair near a table."
        system = build_system_prompt()
        user = build_user_prompt(description)
        self.assertEqual(user, description)
        self.assertNotIn("Scene ID", user)
        self.assertIn("bottom-center", system)
        self.assertIn("front faces -Y", system)
        self.assertIn("Decide the number, type, arrangement, and dimensions", system)
        self.assertIn("Rooms must not overlap in floor area", system)
        self.assertIn(
            "Output inferred room geometry and abstract object layouts only",
            system,
        )
        self.assertNotIn("Output abstract objects only", system)
        self.assertIn("every coordinate value must be non-negative", system)
        self.assertIn('"rooms"', system)
        self.assertIn('"room_id"', system)
        self.assertNotIn("annotation", system.lower())


if __name__ == "__main__":
    unittest.main()
