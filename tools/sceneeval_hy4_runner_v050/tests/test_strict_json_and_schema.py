import unittest

from sceneeval_hy4.layout_schema import validate_layout
from sceneeval_hy4.strict_json import StrictJSONError, loads_strict


def valid_layout():
    return {
        "rooms": [
            {
                "id": "room_1",
                "category": "dining_room",
                "origin_m": [0, 0, 0],
                "size_m": [4, 5, 2.8],
            }
        ],
        "objects": [
            {
                "id": "chair_1",
                "room_id": "room_1",
                "category": "chair",
                "appearance": "wooden dining chair",
                "position_m": [0, 0, 0],
                "size_m": [0.5, 0.5, 0.9],
                "yaw_deg": 0,
            }
        ]
    }


class StrictJSONTests(unittest.TestCase):
    def test_rejects_code_fence_and_surrounding_prose(self):
        for text in ('```json\n{"objects":[]}\n```', 'answer: {"objects":[]}'):
            with self.subTest(text=text), self.assertRaises(StrictJSONError):
                loads_strict(text)

    def test_rejects_duplicate_keys_and_nonstandard_numbers(self):
        for text in ('{"objects":[],"objects":[]}', '{"x":NaN}'):
            with self.subTest(text=text), self.assertRaises(StrictJSONError):
                loads_strict(text)

    def test_allows_only_json_whitespace_around_value(self):
        self.assertEqual(loads_strict(' \n {"objects":[]}\t'), {"objects": []})


class LayoutSchemaTests(unittest.TestCase):
    def test_valid_layout(self):
        self.assertEqual(validate_layout(valid_layout()), [])

    def test_rejects_extra_fields_duplicate_ids_bool_and_nonpositive_size(self):
        value = valid_layout()
        value["extra"] = 1
        value["objects"].append(dict(value["objects"][0]))
        value["objects"][0]["yaw_deg"] = True
        value["objects"][0]["size_m"][1] = 0
        codes = [item["code"] for item in validate_layout(value)]
        self.assertIn("extra_field", codes)
        self.assertIn("unique", codes)
        self.assertIn("type", codes)
        self.assertIn("exclusive_minimum", codes)

    def test_enforces_nonnegative_coordinates_but_not_room_containment(self):
        value = valid_layout()
        value["objects"][0]["position_m"] = [999, 999, 10]
        value["objects"][0]["yaw_deg"] = 123456
        self.assertEqual(validate_layout(value), [])
        value["objects"][0]["position_m"] = [1, -1, 0]
        codes = [item["code"] for item in validate_layout(value)]
        self.assertIn("minimum", codes)

    def test_rejects_unknown_room_reference(self):
        value = valid_layout()
        value["objects"][0]["room_id"] = "missing"
        codes = [item["code"] for item in validate_layout(value)]
        self.assertIn("reference", codes)


if __name__ == "__main__":
    unittest.main()
